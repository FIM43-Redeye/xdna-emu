//! FSM execution: the `step()` method and all helper methods it calls.

use super::*;
use crate::device::bank_arbiter::Requester;
use crate::device::dma::channel::PendingRelease;
use crate::interpreter::execute::cycle_accurate::{fire_watchpoint_events_with_origin, AccessOrigin};
use crate::interpreter::timing::MemoryQuadrant;

/// Map a resolved DMA target to the `AccessOrigin` the target tile's
/// WatchPoint comparator should see.
///
/// `Own` is a local DMA access. For cross-tile (West/East) targets, the
/// origin is described from the target's perspective: a write into the
/// West window lands on the col-1 neighbour, so that neighbour sees the
/// access arriving from the East -- and vice versa. The memtile WatchPoint
/// layout only models East/West quadrant filter bits (AM025: bits [25:24]),
/// which lines up naturally with the MemTile-to-MemTile shared-memory bus
/// topology.
/// Does the word at `addr` cost a memory access, given the address of the
/// previous word the same transfer moved (`last`, None before the first word)?
///
/// The tile DMA's memory side is one bank width wide (128-bit DATAMEMORY_WIDTH,
/// `BankLayout::access_granule_bytes`): it fetches/writes a whole 16-byte
/// granule in one bank slot and streams the four 32-bit beats through a staging
/// buffer. So a word only claims a bank if it opens a granule the previous word
/// did not already bring in -- one bank access per four stream beats, measured
/// on Phoenix NPU1 at 16.0 B / 4.00 beats
/// (docs/superpowers/findings/2026-07-14-dma-bank-access-width.md). Strided BDs
/// fall out of this for free: a stride that lands every word in a fresh granule
/// pays a bank access every word, which is what silicon does.
///
/// Shared by `channel_bank_mask` (the non-committing peek, walking a cloned
/// address generator) and `do_transfer_cycle` (the committing path) so the two
/// can never disagree about which cycles touch a bank. `last` lives on the
/// `Transfer` and only advances when a word actually moves, so a denied channel
/// -- whose FSM step never runs -- re-presents the identical demand next cycle.
#[inline]
fn word_opens_granule(last: Option<u64>, addr: u64, granule: u64) -> bool {
    match last {
        None => true,
        Some(prev) => prev / granule != addr / granule,
    }
}

fn dma_access_origin(target: MemTileTarget) -> AccessOrigin {
    match target {
        MemTileTarget::Own => AccessOrigin::Dma,
        MemTileTarget::West => AccessOrigin::Neighbour(MemoryQuadrant::East),
        MemTileTarget::East => AccessOrigin::Neighbour(MemoryQuadrant::West),
    }
}

impl DmaEngine {
    /// Physical bank layout of this engine's tile data memory.
    #[inline]
    fn bank_layout(&self) -> crate::device::banking::BankLayout {
        use crate::device::banking::BankLayout;
        match self.tile_kind {
            TileKind::Compute => BankLayout::Compute,
            TileKind::Mem => BankLayout::MemTile,
            TileKind::ShimNoc | TileKind::ShimPl => BankLayout::None,
        }
    }

    /// Banks each active channel intends to touch THIS cycle, without
    /// transferring -- the DMA half of the request/arbitrate/commit split
    /// (see `CycleAccurateExecutor::peek_bank_demand` for the core's half).
    ///
    /// Only a channel in `Transferring` can touch a bank this cycle -- every
    /// other FSM phase (BdSetup, AcquiringLock, MemoryLatency,
    /// HostPipelineLatency, BdSwitchBubble, StartupHold, DrainingEgress,
    /// ReleasingLock, BdChaining, Idle, Paused, Error) moves no data, so none
    /// of them can ever be denied by the arbiter -- there is no FSM path that
    /// could "withdraw" a demand this peek never made.
    ///
    /// Scoped to compute tiles: the bank arbiter models the compute tile's 8
    /// physical banks, and `Requester::S2mm`/`Mm2s` ordinals are sized for
    /// `NUM_DMA_CHANNELS` (2 per direction on a compute tile) -- feeding a
    /// MemTile's 6+6 channels through it would panic in `Requester::ordinal`.
    /// MemTile bank geometry is unvalidated (see `BankLayout::MemTile`), so
    /// non-compute engines report no demand at all.
    pub fn peek_bank_demand(&self, layout: crate::device::banking::BankLayout) -> Vec<(Requester, u16)> {
        if !self.tile_kind.is_compute() {
            return Vec::new();
        }

        let mut demand = Vec::new();
        for (ch_idx, ch) in self.channels.iter().enumerate() {
            let ChannelFsm::Transferring { transfer } = &ch.fsm else {
                continue;
            };
            let mask = self.channel_bank_mask(transfer, layout);
            if mask == 0 {
                continue;
            }
            demand.push((self.channel_requester(ch_idx as u8), mask));
        }
        demand
    }

    /// The `Requester` identity for a flat channel index, per direction.
    fn channel_requester(&self, ch_idx: u8) -> Requester {
        let dir_ch = self.per_direction_channel(ch_idx);
        match self.channel_type(ch_idx) {
            ChannelType::S2MM => Requester::S2mm(dir_ch),
            ChannelType::MM2S => Requester::Mm2s(dir_ch),
        }
    }

    /// Tile-local memory offset a byte address wraps to for a channel with NO
    /// MemTile windowing (compute or shim tile) -- hardware simply wraps the
    /// raw address at the tile's data-memory size, no West/Own/East decode.
    ///
    /// Shared by `resolve_mm2s_target`/`resolve_s2mm_target` (the committing
    /// path, called with `tile.data_memory().len()`) and `channel_bank_mask`
    /// (the non-committing peek, called with the compute-tile `MEMORY_SIZE`
    /// archspec constant) so the two can never derive a different address for
    /// the same byte address. Previously they only agreed because the
    /// compute-tile memory size (64 KiB) happens to keep every bank-selecting
    /// bit (4, 14, 15 -- see `BankLayout::Compute::physical_bank`) below bit
    /// 16: true today, but an unenforced coincidence, not a guarantee (task-4
    /// review FIX 2).
    #[inline]
    fn wrap_local_offset(addr: u64, mem_size: usize) -> usize {
        (addr as usize) % mem_size
    }

    /// Bitmask of physical banks `transfer` will touch THIS cycle, given
    /// current stream availability -- 0 if the channel would not actually
    /// move any data this cycle (drained, or stalled on stream data /
    /// backpressure). Purely a read: walks a CLONE of the transfer's address
    /// generator (`AddressGenerator` is `Clone`), never the live one, so this
    /// can never advance the real transfer.
    ///
    /// Derives the exact same per-cycle word count and stream-availability
    /// gates that `do_transfer_cycle` uses to actually move data
    /// (`words_per_cycle_for`, `can_push_stream_out_for_channel`,
    /// `stream_in_count_for_channel` / `has_stream_in_for_channel`), so peek
    /// and commit can never disagree about which banks a channel needs. The
    /// walked address is wrapped through the same `wrap_local_offset` helper
    /// the committing path uses (see its doc comment) -- `peek_bank_demand`'s
    /// compute-only gate means the compute-tile `MEMORY_SIZE` archspec
    /// constant is always the right divisor here.
    ///
    /// Compute tiles never zero-pad (padding is MemTile-MM2S only -- see
    /// `Transfer::new`), so this only has to model the standard per-word
    /// path -- `peek_bank_demand`'s compute-only gate keeps it that way.
    ///
    /// Compression note: when decompression is enabled the number of stream
    /// words consumed per output word is data-dependent (a mask word plus a
    /// variable count of packed data words), which can't be sized without
    /// actually popping the stream. This peek conservatively claims at most
    /// the FIRST word's address in that case -- an under-approximation of a
    /// rare multi-word-per-cycle case, not an over-claim (see task-4 report
    /// and `docs/known-fidelity-gaps.md`).
    fn channel_bank_mask(&self, transfer: &Transfer, layout: crate::device::banking::BankLayout) -> u16 {
        use crate::device::banking::banks_for_access;

        // An MM2S channel's memory side is the granule FETCH into its egress
        // staging FIFO, not the word it hands the stream port this cycle (see
        // `next_granule_fetch`). The two are decoupled by the FIFO, which is
        // the whole point of the staging model.
        if self.uses_egress_staging(transfer) {
            return self.next_granule_fetch(transfer, layout).map_or(0, |f| f.bank_mask);
        }

        let bytes_remaining = transfer.remaining_bytes() as usize;
        if bytes_remaining == 0 {
            return 0;
        }
        let words_per_cycle = self.words_per_cycle_for(transfer);
        let words_this_cycle = words_per_cycle.min((bytes_remaining + 3) / 4);
        if words_this_cycle == 0 {
            return 0;
        }

        let available_words = match transfer.direction {
            TransferDirection::MM2S => {
                if self.can_push_stream_out_for_channel(transfer.channel) {
                    words_this_cycle
                } else {
                    0
                }
            }
            TransferDirection::S2MM => {
                if self.is_decompression_enabled(transfer.channel) {
                    usize::from(self.has_stream_in_for_channel(transfer.channel))
                } else {
                    self.stream_in_count_for_channel(transfer.channel).min(words_this_cycle)
                }
            }
        };
        if available_words == 0 {
            return 0;
        }

        let mem_size = xdna_archspec::aie2::compute::MEMORY_SIZE as usize;
        let granule = layout.access_granule_bytes();
        let mut last = transfer.last_access_addr;
        let mut probe = transfer.address_gen.clone();
        let mut mask = 0u16;
        for _ in 0..available_words {
            let addr = probe.current();
            // Only the word that OPENS a granule costs a bank slot; the other
            // three beats of the granule stream through the staging buffer with
            // no memory access at all -- so they declare no demand and cannot
            // be denied.
            if word_opens_granule(last, addr, granule) {
                let offset = Self::wrap_local_offset(addr, mem_size) as u32;
                mask |= banks_for_access(offset, 4, layout);
            }
            last = Some(addr);
            if probe.next().is_none() {
                break;
            }
        }
        mask
    }

    /// Does this transfer run through the MM2S egress staging FIFO?
    ///
    /// The staging model describes a DMA whose memory side reads banked TILE
    /// memory a 128-bit granule at a time while its stream side emits 32-bit
    /// beats. That is every ordinary compute/mem-tile MM2S. It is NOT:
    ///
    /// - S2MM, whose staging is the ingress FIFO (`stream_in`) on the other side
    ///   of the same asymmetry, already modelled;
    /// - the shim MM2S, which reads host DDR through the NoC rather than a bank
    ///   and whose transient is the separately calibrated DDR cold-start model;
    /// - a zero-padding or compressing MM2S, whose stream output is not a 1:1
    ///   image of the words it reads from memory, so "words fetched but not yet
    ///   sent" is not a well-defined quantity.
    fn uses_egress_staging(&self, transfer: &Transfer) -> bool {
        transfer.direction == TransferDirection::MM2S
            && !transfer.involves_host_memory()
            && !transfer.has_zero_padding()
            && !self.is_compression_enabled(transfer.channel)
    }

    /// The granule fetch an MM2S channel would issue THIS cycle, or `None` if it
    /// would issue none (nothing left unfetched, or the staging FIFO has no room
    /// for a whole granule).
    ///
    /// This is the channel's entire memory-side demand: one bank access that
    /// brings in every word of one 16-byte granule. Pure -- walks a CLONE of the
    /// address generator -- so `peek_bank_demand` (which runs before arbitration)
    /// and `do_transfer_cycle` (which runs after it) can call it and get the same
    /// answer.
    ///
    /// The fetch starts at the first word the staging does not already hold: the
    /// drain cursor (`address_gen.current()`) advanced by `staged_words`. Because
    /// draining a word advances the cursor by one and drops `staged_words` by one,
    /// that position -- and hence the bank this returns -- is INVARIANT under a
    /// drain. A channel denied its fetch therefore re-presents the identical
    /// demand next cycle even though its stream side kept running, which is what
    /// the arbiter's retry contract requires (AM020 ch.2:166).
    fn next_granule_fetch(
        &self,
        transfer: &Transfer,
        layout: crate::device::banking::BankLayout,
    ) -> Option<GranuleFetch> {
        use crate::device::banking::banks_for_access;

        let words_to_drain = (transfer.remaining_bytes() as usize).div_ceil(4);
        let unfetched = words_to_drain.checked_sub(transfer.staged_words)?;
        if unfetched == 0 {
            return None;
        }

        // Walk the clone to the first unfetched word.
        let mut probe = transfer.address_gen.clone();
        for _ in 0..transfer.staged_words {
            probe.next()?;
        }
        let first = probe.current();

        // The fetch brings in every word of `first`'s granule that this transfer
        // visits next, contiguously. A strided BD whose next word lands in a
        // different granule therefore fetches one word and pays a bank access per
        // word -- which is what silicon does.
        let granule = layout.access_granule_bytes();
        let mut words = 1usize;
        let mut prev = first;
        while words < unfetched {
            if probe.next().is_none() {
                break;
            }
            let addr = probe.current();
            if word_opens_granule(Some(prev), addr, granule) {
                break;
            }
            prev = addr;
            words += 1;
        }

        if transfer.staged_words + words > self.egress_staging_capacity() {
            return None; // no room for the whole granule; wait for the stream to drain one
        }

        let mem_size = xdna_archspec::aie2::compute::MEMORY_SIZE as usize;
        let offset = Self::wrap_local_offset(first, mem_size) as u32;
        Some(GranuleFetch { bank_mask: banks_for_access(offset, 4, layout), words })
    }

    /// Step the DMA engine by one cycle.
    ///
    /// This processes all active channels, moving data between memory and streams.
    /// Returns the overall result of the step.
    pub fn step(
        &mut self,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> DmaResult {
        self.step_impl(&[], tile, neighbors, host_memory)
    }

    /// Step the DMA engine by one cycle, honouring bank-arbitration denial.
    ///
    /// `denied` names the channels (by `Requester::S2mm`/`Mm2s`) that lost
    /// this cycle's bank arbitration (see `peek_bank_demand`). A denied
    /// channel's FSM step is skipped ENTIRELY this cycle -- `step_channel_fsm`
    /// is simply never called for it, so every field of its in-flight
    /// `Transfer` (address generator, `bytes_transferred`, zero-pad state)
    /// and every other piece of per-channel bookkeeping (current BD, repeat
    /// count, task queue, lock-wait counters, `is_first_bd`, ...) is left
    /// byte-for-byte as it was. The channel re-presents the identical bank
    /// demand next cycle, honouring the AM020 ch.2:166 retry contract (see
    /// `bank_arbiter` module docs): a hold that skips the whole step can
    /// never partially mutate state, because nothing about the held channel
    /// runs at all.
    ///
    /// Only a channel in `Transferring` is ever denied in practice --
    /// `peek_bank_demand` never reports a demand for any other FSM phase, so
    /// `denied` should never name one. If it did anyway (a caller bug), this
    /// is a no-op for that channel: denial only takes effect on a channel
    /// actually in `Transferring`.
    pub fn step_with_denied(
        &mut self,
        denied: &[Requester],
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> DmaResult {
        self.step_impl(denied, tile, neighbors, host_memory)
    }

    /// Whether `ch_idx` lost this cycle's bank arbitration, i.e. its MEMORY-side
    /// transaction does not happen. Only a channel actually in `Transferring` can
    /// be meaningfully denied -- that is the only FSM phase that ever touches a
    /// bank (see `peek_bank_demand`), so every other phase always returns `false`
    /// here regardless of what `denied` contains.
    fn is_mem_denied(&self, ch_idx: usize, denied: &[Requester]) -> bool {
        if denied.is_empty() {
            return false;
        }
        if !matches!(self.channels[ch_idx].fsm, ChannelFsm::Transferring { .. }) {
            return false;
        }
        denied.contains(&self.channel_requester(ch_idx as u8))
    }

    /// Whether a denial holds the WHOLE channel inert this cycle.
    ///
    /// The arbiter denies a memory transaction, not a channel. What that stops
    /// depends on which side of the channel's staging FIFO the memory sits:
    ///
    /// - **S2MM**: the memory WRITE is the only data movement its FSM step
    ///   performs -- the stream side is the ingress FIFO, filled by the routing
    ///   pass, which the arbiter never sees. So a denied S2MM's step is skipped
    ///   ENTIRELY: nothing about it runs, its `Transfer` is untouched, and its
    ///   ingress keeps accumulating the words the stream keeps offering. That is
    ///   the absorption.
    /// - **MM2S**: the memory READ is the granule fetch into the egress staging
    ///   FIFO. The word the channel hands the stream port comes OUT of that FIFO
    ///   and is not a memory access, so a memory arbiter cannot stop it. A denied
    ///   MM2S therefore still runs its step, with the fetch suppressed -- and the
    ///   staging covers the delayed refill. Silicon does exactly this: an MM2S
    ///   that lost 112 bank arbitrations paid 5 cycles for them
    ///   (docs/superpowers/findings/2026-07-14-dma-bank-access-width.md). Holding
    ///   its stream side too would bubble the port on every loss.
    ///
    /// Either way the channel re-presents the identical demand next cycle (see
    /// `next_granule_fetch` for why draining does not move the MM2S fetch
    /// cursor), honouring the AM020 ch.2:166 retry contract.
    fn is_held_inert(&self, ch_idx: usize, denied: &[Requester]) -> bool {
        self.is_mem_denied(ch_idx, denied) && matches!(self.channel_type(ch_idx as u8), ChannelType::S2MM)
    }

    fn step_impl(
        &mut self,
        denied: &[Requester],
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> DmaResult {
        let mut any_active = false;
        let mut any_waiting = false;

        for ch_idx in 0..self.channels.len() {
            let phase_before = self.channels[ch_idx].fsm.phase_name();
            // Snapshot the egress backlog BEFORE the step pushes this cycle's
            // word, so `complete_or_drain` can tell a lagging sink from a
            // healthy one (see `ChannelContext::egress_backlog`).
            self.channels[ch_idx].egress_backlog = self.stream_out_len_for_channel(ch_idx as u8);

            match &self.channels[ch_idx].fsm {
                ChannelFsm::Idle => {
                    // Check for queued BD from chaining or task queue
                    if let Some(next_bd) = self.channels[ch_idx].queued_bd.take() {
                        let repeat_count = self.channels[ch_idx].repeat_count as u8;
                        log::debug!(
                            "DMA tile({},{}) ch{} starting queued BD {} (repeat={})",
                            self.col,
                            self.row,
                            ch_idx,
                            next_bd,
                            repeat_count
                        );
                        if let Err(e) = self.start_channel_with_repeat(ch_idx as u8, next_bd, repeat_count) {
                            log::warn!(
                                "DMA tile({},{}) ch{} failed to start BD {}: {:?}",
                                self.col,
                                self.row,
                                ch_idx,
                                next_bd,
                                e
                            );
                            self.channels[ch_idx].fsm = ChannelFsm::Error;
                        } else {
                            any_active = true;
                        }
                    }
                }
                ChannelFsm::Paused { .. } | ChannelFsm::Error => {}
                _ => {
                    if self.is_held_inert(ch_idx, denied) {
                        // Held by bank arbitration: skip the FSM step
                        // entirely (see `is_held_inert`) so the channel
                        // retries the identical demand next cycle.
                        any_active = true;
                    } else {
                        // Active channel -- run one FSM cycle. A denied MM2S
                        // runs with its granule fetch suppressed but its
                        // stream side live (see `is_held_inert`).
                        let mem_denied = self.is_mem_denied(ch_idx, denied);
                        self.step_channel_fsm(ch_idx, mem_denied, tile, neighbors, host_memory);
                        if matches!(
                            self.channels[ch_idx].fsm,
                            ChannelFsm::AcquiringLock { acquired: false, .. }
                        ) {
                            any_waiting = true;
                        } else {
                            any_active = true;
                        }
                    }
                }
            }

            // Log transitions
            let phase_after = self.channels[ch_idx].fsm.phase_name();
            if phase_before != phase_after {
                log::info!(
                    "DMA({},{}) ch{}: {} -> {} cycle={}",
                    self.col,
                    self.row,
                    ch_idx,
                    phase_before,
                    phase_after,
                    self.current_cycle,
                );
            }
        }

        // Apply any deferred cross-lock release whose swap has occurred and
        // whose release latency has now elapsed. Runs after the FSM pass so a
        // release whose swap landed this cycle (channel no longer blocked) is
        // serviced the same cycle once past its ready_cycle floor. `denied`
        // is threaded through so a held channel's swap-enable-release watch
        // stays frozen too (FIX 1 -- see `service_pending_releases`).
        self.service_pending_releases(denied, tile, neighbors);

        if any_active {
            DmaResult::InProgress
        } else if any_waiting {
            DmaResult::WaitingForLock(0)
        } else {
            DmaResult::Complete
        }
    }

    /// Service deferred memtile LockRelease TRACE events. The functional
    /// semaphore was already released inline at completion (see
    /// `begin_completion`); these entries carry only the observable trace event.
    ///
    /// Schedule (fallback): a slot that is never reused before the channel goes
    /// idle -- the last fills of a finite task -- emits its trace at the
    /// `ready_cycle` floor. Slots that DO recycle get `trace_at` set by the
    /// BD-reuse grant (see step()), where HW's LOCK_SEL*_REL actually fires.
    ///
    /// Emit: once `current_cycle` reaches `trace_at`, emit the trace event at
    /// that cycle and drop the entry.
    ///
    /// `denied` gates the swap-enable-release WATCH only (see the call below),
    /// not this emit loop: a trace event whose `trace_at` was already fixed on
    /// an earlier (non-denied) cycle is a past fact -- a lock-release event
    /// timing, independent of this cycle's bank arbitration -- and fires on
    /// schedule regardless of whether this channel wins arbitration this
    /// cycle. See `is_denied_this_cycle` / FIX 1 in the task-4 report.
    fn service_pending_releases(
        &mut self,
        denied: &[Requester],
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) {
        for ch_idx in 0..self.channels.len() {
            // End-of-stream release tail: while a producer is stream-stalled, a
            // consumer-free (its acquire lock incrementing) is the SWAP-enable
            // that fires the next deferred full-release -- HW emits LOCK_SEL*_REL
            // there even though the stalled producer never re-acquires the slot.
            //
            // A HELD-INERT channel must be inert for the WHOLE cycle, not just
            // its own transfer: `prev_starving` is only cleared by
            // `step_transferring_cycle`, which a held channel never runs, so
            // it can go stale (true from a prior real stall) while the
            // channel's actual data availability has since changed. Reading
            // that stale flag here would let a swap that happens on a held
            // cycle retire a `pending_releases` entry and fire its deferred
            // LockRelease TRACE EVENT on a cycle the channel never actually
            // ran -- so skip the watch entirely for it this cycle (see
            // `is_held_inert`). Deliberately NOT clearing `prev_starving`
            // instead: that flag is real information about whether the
            // STREAM_STARVATION signal is currently asserted, and
            // artificially clearing it on a held cycle would make the
            // channel's next real step think starvation is a fresh onset
            // (spurious assert/deassert trace pair) instead of a
            // continuation. Skipping the watch call leaves that bit of state
            // alone and freezes only the bookkeeping this cycle's denial
            // should not be allowed to touch. A bank-denied MM2S is NOT held
            // inert -- it ran its step and maintained `prev_starving` itself,
            // so its watch is not stale and must not be skipped.
            if !self.is_held_inert(ch_idx, denied) {
                self.schedule_swap_enable_releases(ch_idx, tile, neighbors);
            }

            let channel_idle = !self.channels[ch_idx].has_pending_work();
            let mut i = 0;
            while i < self.channels[ch_idx].pending_releases.len() {
                let p = self.channels[ch_idx].pending_releases[i];
                // Fallback: a never-reused slot emits at the ready_cycle floor
                // once its channel idles.
                if p.trace_at.is_none() && channel_idle {
                    self.channels[ch_idx].pending_releases[i].trace_at = Some(p.ready_cycle);
                }
                // Emit the deferred trace event once it is due.
                let p = self.channels[ch_idx].pending_releases[i];
                match p.trace_at {
                    Some(t) if self.current_cycle >= t => {
                        self.channels[ch_idx].pending_releases.remove(i);
                        self.emit_lock_release_trace_at(p.lock_id, t, tile, neighbors);
                    }
                    _ => i += 1,
                }
            }
        }
    }

    /// End-of-stream release tail (tenant-4, HW-pinned): the producer's deferred
    /// full-release fires on the SWAP-enable -- the next buffer becoming FREE,
    /// i.e. a consumer-free event -- not on the producer's actual re-acquire.
    ///
    /// In steady state the producer re-acquires the slot right after the consumer
    /// frees it, so the re-acquire-grant path (`retire_slot_trace`) already lands
    /// the release on the consumer cadence. But when the input stream is exhausted
    /// the producer stream-stalls and never issues that re-acquire, so a trailing
    /// fill's slot is never recycled and its release would be dropped (the probe's
    /// 7-of-8 tail). HW still fires it: the producer probe shows the final
    /// LOCK_SEL1_REL landing where the next consumer-free does, with no
    /// LOCK_SEL0_ACQ after it.
    ///
    /// So while a channel is stream-stalled, watch its acquire lock (FREE). Each
    /// increment is a consumer-free (the stalled producer cannot decrement it), so
    /// emit one oldest still-pending release per +1 at `max(ready_cycle, now)`.
    /// Gated on `prev_starving`, so steady state (where the re-acquire path owns
    /// the timing) is untouched, and the `trace_at`-already-set guard prevents any
    /// double-fire against the re-acquire path.
    fn schedule_swap_enable_releases(&mut self, ch_idx: usize, tile: &Tile, neighbors: &NeighborTiles<'_>) {
        if !self.channels[ch_idx].prev_starving {
            // Not stream-stalled: re-baseline so the next stall starts fresh.
            self.channels[ch_idx].swap_free_watch = None;
            return;
        }
        // The stalled fill holds an acquired FREE buffer; its acquire lock IS the
        // FREE semaphore the consumer replenishes.
        let acquire_lock = self.channels[ch_idx].fsm.transfer().and_then(|t| t.acquire_lock);
        let Some(lock_id) = acquire_lock else { return };
        let Some(value) = self.read_lock_value(lock_id, tile, neighbors) else {
            return;
        };

        let increments = match self.channels[ch_idx].swap_free_watch {
            Some((prev_lock, prev_val)) if prev_lock == lock_id && value > prev_val => {
                (value - prev_val) as usize
            }
            _ => 0,
        };
        self.channels[ch_idx].swap_free_watch = Some((lock_id, value));

        // Emit one oldest still-pending release per consumer-free.
        let now = self.current_cycle;
        for _ in 0..increments {
            if let Some(pr) = self.channels[ch_idx].pending_releases.iter_mut().find(|p| p.trace_at.is_none())
            {
                pr.trace_at = Some(pr.ready_cycle.max(now));
            } else {
                break;
            }
        }
    }

    /// Read a (possibly-neighbor) lock's committed value, read-only.
    fn read_lock_value(&self, lock_id: u8, tile: &Tile, neighbors: &NeighborTiles<'_>) -> Option<i8> {
        match self.resolve_lock_id(lock_id)? {
            LockTarget::Own(id) => tile.locks.get(id as usize).map(|l| l.value),
            LockTarget::West(id) => neighbors
                .west
                .as_deref()
                .and_then(|w| w.locks.get(id as usize))
                .map(|l| l.value),
            LockTarget::East(id) => neighbors
                .east
                .as_deref()
                .and_then(|e| e.locks.get(id as usize))
                .map(|l| l.value),
        }
    }

    /// Consume the first-BD-of-task bonus into the next MemoryLatency budget.
    ///
    /// Returns extra cycles to add to memory_latency_cycles when entering
    /// MemoryLatency for the first BD of a task.  Combines three terms:
    ///
    /// - `channel_start_cycles`: generic "start trigger to first data" cost.
    ///   Fires on every task (every Idle->BdSetup), tile-agnostic.
    /// - `shim_per_task_overhead_{mm2s,s2mm}_cycles`: per-task overhead on
    ///   shim DMA touching host memory.  Fires on every task -- covers BD
    ///   programming + per-task AXI burst arbitration.
    /// - `shim_ddr_cold_start_{mm2s,s2mm}_cycles` decayed by
    ///   `shim_warmup_decay_{mm2s,s2mm}_permille`: the warm-up transient.
    ///   The cold-start cost (DDR read pipeline fill on MM2S / AXI setup on
    ///   S2MM) is charged as `cold_start * (decay/1000)^i` at task index
    ///   `i = warm_task_index`, so it decays geometrically across the chain
    ///   instead of firing once.  MM2S fades over ~4 tasks (r~0.31); S2MM's
    ///   decay is 0, so only task 0 pays cold-start (pure one-shot).  Phase 2d.
    ///
    /// `is_first_bd` is reset to true by `after_transfer_done` (Idle re-entry)
    /// and by `stop_channel`; `warm_task_index` increments per task and is
    /// reset to 0 only by `stop_channel`.
    pub(crate) fn consume_first_bd_bonus(&mut self, ch_idx: usize, transfer: &Transfer) -> u16 {
        if !self.channels[ch_idx].is_first_bd {
            return 0;
        }
        self.channels[ch_idx].is_first_bd = false;
        let mut bonus = self.timing_config.channel_start_cycles as u16;
        if self.tile_kind.is_shim() && transfer.involves_host_memory() {
            // SP-4a (#140): when the shim S2MM cold-start drain THROTTLE is
            // enabled, the FIRST host-memory S2MM task charges its ENTIRE
            // cold-start -- both the per-task overhead AND the DDR cold-start --
            // as a metered POST-arrival ingress->DDR drain (see
            // `do_transfer_cycle`), NOT as a pre-transfer `MemoryLatency` hold.
            // HW starts Transferring promptly and starves on the empty ingress
            // (~+13); charging the overhead pre-transfer instead blocks the
            // ingress drain and inflates first-starvation (measured +190, of
            // which 179 is the per-task overhead residual -- FINDING.md). The
            // metered drain (cooldown/decay) is thus the sole model of the
            // cold-start transient for the throttled first task.
            let throttle_s2mm_first = transfer.direction == TransferDirection::S2MM
                && self.timing_config.shim_s2mm_cold_drain_cooldown_cycles > 0
                && self.channels[ch_idx].warm_task_index == 0;
            if throttle_s2mm_first {
                self.channels[ch_idx].cold_drain_armed = true;
                self.channels[ch_idx].cold_drain_cooldown =
                    self.timing_config.shim_s2mm_cold_drain_cooldown_cycles;
                self.channels[ch_idx].cold_drain_word_index = 0;
            } else {
                // Per-task overhead fires on every (non-throttled) task.
                bonus += match transfer.direction {
                    TransferDirection::MM2S => self.timing_config.shim_per_task_overhead_mm2s_cycles,
                    TransferDirection::S2MM => self.timing_config.shim_per_task_overhead_s2mm_cycles,
                };
                // Warm-up transient: the cold-start cost decays geometrically
                // across the task chain rather than firing once.  At task index
                // `i` the channel pays cold_start * (decay/1000)^i -- i=0 is the
                // full cold-start, and the tail fades over ~4 tasks.  MM2S has
                // a real decay (r~0.31); S2MM's decay is 0, so it pays only the
                // one-shot cold-start at i=0, preserving prior behavior.  Phase 2d.
                let (cold_start, decay_permille) = match transfer.direction {
                    TransferDirection::MM2S => (
                        self.timing_config.shim_ddr_cold_start_mm2s_cycles,
                        self.timing_config.shim_warmup_decay_mm2s_permille,
                    ),
                    TransferDirection::S2MM => (
                        self.timing_config.shim_ddr_cold_start_s2mm_cycles,
                        self.timing_config.shim_warmup_decay_s2mm_permille,
                    ),
                };
                let mut term = cold_start as u32;
                for _ in 0..self.channels[ch_idx].warm_task_index {
                    term = term * decay_permille as u32 / 1000;
                }
                bonus += term as u16;
            }
            self.channels[ch_idx].warm_task_index += 1;
        } else {
            // Non-shim DMA channels (memtile / compute): a one-time-per-
            // channel-session pipeline-FILL cost on the FIRST task.  This is
            // the non-shim analogue of the shim DDR cold-start -- the memtile/
            // core channels have a real STARTING phase (DMA `Status` state 01,
            // AM025) that EMU otherwise collapses.  On add_one's deeper
            // ObjectFifo ping-pong that collapse spreads ~1200cy of first-
            // output latency the steady-state cadence can't account for (#140).
            //
            // Crucially, this latency is latched as a POST-transfer hold
            // (`startup_hold_cycles` -> `StartupHold`), NOT added to the
            // pre-transfer MemoryLatency budget.  A pre-transfer stall would
            // keep an S2MM channel from draining its input stream, backpressur-
            // ing the shim MM2S upstream (measured: memtile startup inflated
            // the shim MM2S ~1:1).  Holding AFTER the transfer instead delays
            // only the downstream-visible completion -- first-output widens
            // (shim S2MM grows) while the shim MM2S is untouched.
            //
            // Gated on `warm_task_index == 0` so it fires ONCE per session;
            // re-armed tasks past the first stay warm.  Default 0 = no-op.
            if self.channels[ch_idx].warm_task_index == 0 {
                self.channels[ch_idx].startup_hold_cycles = if self.tile_kind.is_mem() {
                    self.timing_config.memtile_first_bd_startup_cycles
                } else if self.tile_kind.is_compute() {
                    self.timing_config.compute_first_bd_startup_cycles
                } else {
                    // Shim tile not touching host memory (rare local route):
                    // no startup model.
                    0
                };
            }
            self.channels[ch_idx].warm_task_index += 1;
        }
        bonus
    }

    /// BD-prefetch overlap (Phase 2d.2): while a channel is transferring the
    /// current task, if another task is already queued, HW loads that next BD
    /// into its second slot and fires its START_TASK event during the current
    /// transfer -- it does not wait for the current task to finish.  Emit that
    /// START ahead of time, once per queued task, and mark the channel so
    /// start_channel suppresses the duplicate when the task actually begins.
    ///
    /// The data path stays strictly serial (only one transfer moves data at a
    /// time); this moves only the *event* timing, which is what lets the next
    /// task's START precede the current task's FINISHED -- i.e. a negative
    /// inter-task gap when the current transfer is long.  When the controller
    /// dispatches slowly (queue empty during the transfer) this never fires,
    /// so short steady-state tasks keep their positive gaps.
    fn maybe_prefetch_next_task(&mut self, ch_idx: usize) {
        if !self.dma_model.supports_task_queue() {
            return;
        }
        let ch = &self.channels[ch_idx];
        if ch.prefetch_start_emitted || ch.task_queue.is_empty() {
            return;
        }
        self.channels[ch_idx].prefetch_start_emitted = true;
        self.trace(EventType::DmaStartTask { channel: ch_idx as u8 });
    }

    /// Step a single channel through one cycle of its unified FSM.
    ///
    /// Each match arm does ONE cycle of work and optionally transitions to
    /// a new state. This replaces the old step_channel() + step_channel_timed() +
    /// complete_transfer() + finish_complete_transfer() chain.
    ///
    /// `mem_denied` is set only for an MM2S channel that lost this cycle's bank
    /// arbitration: its granule fetch does not happen, but its stream side runs
    /// (see `is_held_inert`). A denied S2MM never reaches here at all.
    fn step_channel_fsm(
        &mut self,
        ch_idx: usize,
        mem_denied: bool,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) {
        // Take the FSM out temporarily so we can match on it while
        // mutating self (for stream buffers, do_transfer, etc.)
        let fsm = std::mem::take(&mut self.channels[ch_idx].fsm);

        let new_fsm = match fsm {
            ChannelFsm::BdSetup { cycles_remaining, mut transfer } => {
                if cycles_remaining <= 1 {
                    // BD setup done. Check if lock acquisition is needed.
                    // Packet header insertion is deferred until AFTER lock
                    // acquisition -- on real hardware, the DMA only emits
                    // the header when it starts transferring data, not before
                    // the lock is acquired. Emitting early would lock the
                    // stream switch arbiter while the DMA waits for locks,
                    // blocking other packets that share the same arbiter.
                    if let Some(lock_id) = transfer.acquire_lock {
                        ChannelFsm::AcquiringLock {
                            lock_id,
                            cycles_remaining: self.timing_config.lock_acquire_cycles as u16,
                            acquired: false,
                            transfer,
                        }
                    } else {
                        // No lock needed -- insert header now, go to MemoryLatency.
                        self.maybe_insert_packet_header_from_transfer(&mut transfer);
                        let bonus = self.consume_first_bd_bonus(ch_idx, &transfer);
                        ChannelFsm::MemoryLatency {
                            cycles_remaining: (self.timing_config.memory_latency_cycles as u16) + bonus,
                            transfer,
                        }
                    }
                } else {
                    ChannelFsm::BdSetup { cycles_remaining: cycles_remaining - 1, transfer }
                }
            }

            ChannelFsm::AcquiringLock { lock_id, cycles_remaining, acquired, mut transfer } => {
                if acquired {
                    // Lock already acquired, counting down latency
                    if cycles_remaining <= 1 {
                        // Lock acquired, latency done -- insert packet header
                        // now that we're about to start the actual transfer.
                        self.maybe_insert_packet_header_from_transfer(&mut transfer);
                        // Dispatch on is_first_bd: cold-start (first BD of a
                        // task) pays MemoryLatency for pipeline fill; chained
                        // BDs skip it -- hardware prefetched and warmed the
                        // memory pipeline during the prior BD's transferring
                        // tail. See enter_chained_bd / the BD-chain finding.
                        if self.channels[ch_idx].is_first_bd {
                            let bonus = self.consume_first_bd_bonus(ch_idx, &transfer);
                            ChannelFsm::MemoryLatency {
                                cycles_remaining: (self.timing_config.memory_latency_cycles as u16) + bonus,
                                transfer,
                            }
                        } else {
                            // Chained BD whose post-grant cooldown just elapsed:
                            // route through the BD-switch bubble.
                            self.enter_chained_transfer(ch_idx, transfer)
                        }
                    } else {
                        ChannelFsm::AcquiringLock {
                            lock_id,
                            cycles_remaining: cycles_remaining - 1,
                            acquired: true,
                            transfer,
                        }
                    }
                } else {
                    // Check if arbiter granted the acquire (submitted in pre-step pass)
                    if self.check_acquire_granted(lock_id, tile, neighbors, ch_idx) {
                        self.on_acquire_granted(ch_idx, lock_id, transfer.bd_index);
                        // For a chained BD with no post-grant cooldown
                        // (cycles_remaining=0 from enter_chained_bd), collapse
                        // the AcquiringLock{acquired=true,cr=0} -> Transferring
                        // intermediate state into this same step. Saves one
                        // dead cycle on the chained-lock critical path. The
                        // cold-start path (is_first_bd=true with full
                        // lock_acquire_cycles) is unchanged -- it still
                        // counts down the post-grant latency in the
                        // acquired=true arm below.
                        if cycles_remaining == 0 && !self.channels[ch_idx].is_first_bd {
                            self.maybe_insert_packet_header_from_transfer(&mut transfer);
                            let host_lat = self.timing_config.host_memory_latency_cycles;
                            if host_lat > 0 && transfer.involves_host_memory() {
                                // Shim host-DDR still pays its NoC/DDR
                                // pipeline-fill latency before data flows.
                                ChannelFsm::HostPipelineLatency { cycles_remaining: host_lat, transfer }
                            } else {
                                self.enter_transfer_after_lock_grant(ch_idx, transfer)
                            }
                        } else {
                            ChannelFsm::AcquiringLock { lock_id, cycles_remaining, acquired: true, transfer }
                        }
                    } else {
                        self.channels[ch_idx].stats.lock_wait_cycles += 1;
                        // Held level: assert on the rising edge of the
                        // lock-stall signal; the matching deassert fires when the
                        // acquire is granted (see check_acquire_granted arm).
                        if !self.channels[ch_idx].prev_lock_stalled {
                            self.trace(EventType::DmaStalledLock { channel: ch_idx as u8, active: true });
                            self.channels[ch_idx].prev_lock_stalled = true;
                        }
                        // No deadlock-break needed: functional lock releases fire
                        // inline at BD completion (only the trace event defers),
                        // so a blocked acquire is always satisfiable by the
                        // peer/core release as it happens -- nothing to flush.
                        ChannelFsm::AcquiringLock { lock_id, cycles_remaining, acquired: false, transfer }
                    }
                }
            }

            ChannelFsm::MemoryLatency { cycles_remaining, transfer } => {
                if cycles_remaining <= 1 {
                    // For shim tiles accessing host DDR, add NoC+DDR pipeline
                    // latency. This is the extra time for the first word to
                    // traverse the NoC to DDR and back. Once the pipeline
                    // fills, throughput is 1 word/cycle (same as tile memory).
                    let host_lat = self.timing_config.host_memory_latency_cycles;
                    if host_lat > 0 && transfer.involves_host_memory() {
                        ChannelFsm::HostPipelineLatency { cycles_remaining: host_lat, transfer }
                    } else {
                        ChannelFsm::Transferring { transfer }
                    }
                } else {
                    ChannelFsm::MemoryLatency { cycles_remaining: cycles_remaining - 1, transfer }
                }
            }

            ChannelFsm::HostPipelineLatency { cycles_remaining, transfer } => {
                if cycles_remaining <= 1 {
                    ChannelFsm::Transferring { transfer }
                } else {
                    ChannelFsm::HostPipelineLatency { cycles_remaining: cycles_remaining - 1, transfer }
                }
            }

            ChannelFsm::BdSwitchBubble { cycles_remaining, transfer } => {
                // No beat moves while bubbling -- the stream port idles, which
                // is exactly the PORT_RUNNING deassert we model at each BD
                // boundary. When it drains, resume data movement.
                if cycles_remaining <= 1 {
                    ChannelFsm::Transferring { transfer }
                } else {
                    ChannelFsm::BdSwitchBubble { cycles_remaining: cycles_remaining - 1, transfer }
                }
            }

            ChannelFsm::Transferring { transfer } => {
                self.maybe_prefetch_next_task(ch_idx);
                self.step_transferring_cycle(ch_idx, mem_denied, transfer, tile, neighbors, host_memory)
            }

            ChannelFsm::StartupHold { cycles_remaining, transfer } => {
                // Data is already fully moved; this is pure completion latency
                // (#140 non-shim pipeline fill).  No beat moves, no stream
                // port activity -- the input was drained during Transferring,
                // so nothing backpressures upstream.  When it elapses, run the
                // normal completion (release + FINISHED_BD + chaining), still
                // gated on the egress stream having drained (#140 SP-4a).
                if cycles_remaining <= 1 {
                    self.complete_or_drain(ch_idx, transfer, tile, neighbors)
                } else {
                    ChannelFsm::StartupHold { cycles_remaining: cycles_remaining - 1, transfer }
                }
            }

            ChannelFsm::DrainingEgress { transfer } => {
                // Hold until the egress stream has handshaked downstream: the BD
                // is not retired (lock release / FINISHED_BD / chaining) until
                // the last word has left the local `stream_out` FIFO, not merely
                // memory (#140 SP-4a).  `route_dma_to_tile_switches` drains
                // `stream_out` every cycle regardless of FSM state, so a healthy
                // link exits in ~1-2 cycles; a stream-gated terminal link holds
                // here (backpressuring upstream) until its sink drains.
                if self.stream_out_len_for_channel(ch_idx as u8) == 0 {
                    self.begin_completion(ch_idx, transfer, tile, neighbors)
                } else {
                    ChannelFsm::DrainingEgress { transfer }
                }
            }

            ChannelFsm::ReleasingLock { lock_id, release_value, cycles_remaining, completion } => {
                if cycles_remaining <= 1 {
                    // Execute the lock release
                    self.execute_lock_release(lock_id, release_value, tile, neighbors);
                    // Update stats and handle chaining/repeat
                    self.after_transfer_done(ch_idx, completion, tile, neighbors)
                } else {
                    ChannelFsm::ReleasingLock {
                        lock_id,
                        release_value,
                        cycles_remaining: cycles_remaining - 1,
                        completion,
                    }
                }
            }

            ChannelFsm::BdChaining { cycles_remaining, next_bd } => {
                if cycles_remaining <= 1 {
                    // Load next BD and start transfer
                    let bd_addr = self.bd_configs.get(next_bd as usize).map(|c| c.base_addr).unwrap_or(0);
                    log::info!(
                        "DMA tile({},{}) ch{} BD chain -> BD{} (base_addr=0x{:X})",
                        self.col,
                        self.row,
                        ch_idx,
                        next_bd,
                        bd_addr
                    );
                    match self.create_transfer_from_bd(next_bd, ch_idx as u8) {
                        Ok(transfer) => {
                            self.channels[ch_idx].current_bd = Some(next_bd);
                            ChannelFsm::BdSetup {
                                cycles_remaining: self.timing_config.bd_setup_cycles as u16,
                                transfer: Box::new(transfer),
                            }
                        }
                        Err(e) => {
                            log::warn!(
                                "DMA tile({},{}) ch{} BD chain to {} failed: {:?}",
                                self.col,
                                self.row,
                                ch_idx,
                                next_bd,
                                e
                            );
                            ChannelFsm::Error
                        }
                    }
                } else {
                    ChannelFsm::BdChaining { cycles_remaining: cycles_remaining - 1, next_bd }
                }
            }

            // Idle/Paused/Error handled in step(), not here
            other => other,
        };

        self.channels[ch_idx].fsm = new_fsm;
    }

    /// Run one cycle of data movement and return the next FSM state.
    ///
    /// Factored out of the `Transferring` match arm so chained-BD paths
    /// can inline the first data cycle into the same step as the
    /// arbiter grant. Handles all four `TransferCycleResult` outcomes:
    /// Continue (advance, transition to completion if drained), Stalled
    /// (stay in Transferring, fire edge-triggered starvation event),
    /// FotFinish (early completion on TLAST), Error.
    fn step_transferring_cycle(
        &mut self,
        ch_idx: usize,
        mem_denied: bool,
        mut transfer: Box<Transfer>,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> ChannelFsm {
        let result = self.do_transfer_cycle(ch_idx, mem_denied, &mut transfer, tile, neighbors, host_memory);

        match result {
            TransferCycleResult::Continue => {
                transfer.tick();
                // Stream data resumed -- starvation level deasserts.
                if self.channels[ch_idx].prev_starving {
                    self.trace(EventType::DmaStreamStarvation { channel: ch_idx as u8, active: false });
                    self.channels[ch_idx].prev_starving = false;
                }
                if transfer.remaining_bytes() == 0 {
                    self.complete_or_hold(ch_idx, transfer, tile, neighbors)
                } else {
                    ChannelFsm::Transferring { transfer }
                }
            }
            TransferCycleResult::Stalled => {
                // STREAM_STARVATION held level: assert on the rising edge; the
                // deassert fires on the next Continue (data resumed).
                if !self.channels[ch_idx].prev_starving {
                    self.trace(EventType::DmaStreamStarvation { channel: ch_idx as u8, active: true });
                    self.channels[ch_idx].prev_starving = true;
                }
                ChannelFsm::Transferring { transfer }
            }
            TransferCycleResult::MemoryStarved => {
                // The egress staging ran dry with the stream port able to take a
                // beat: DMA_MM2S_n_MEMORY_STARVATION, raised in `do_transfer_cycle`.
                // Distinct from `Stalled`, which is the STREAM side backing up --
                // the two are duals reading this FIFO from opposite ends and are
                // mutually exclusive on silicon (BACKPRESSURE n STREAM_STARVATION
                // = 0 exactly, over 3 runs; see the event-semantics finding). So
                // this must NOT touch the stream-side starvation level.
                ChannelFsm::Transferring { transfer }
            }
            TransferCycleResult::FotFinish => self.complete_or_hold(ch_idx, transfer, tile, neighbors),
            TransferCycleResult::Error => ChannelFsm::Error,
        }
    }

    /// Create a Transfer from a BD index without starting a channel.
    /// Used by BdChaining to load the next BD in a chain.
    fn create_transfer_from_bd(&self, bd_index: u8, channel: u8) -> Result<Transfer, DmaError> {
        if bd_index as usize >= self.bd_configs.len() {
            return Err(DmaError::InvalidBd(bd_index));
        }
        let bd_config = &self.bd_configs[bd_index as usize];
        let direction = match self.channel_type(channel) {
            ChannelType::S2MM => TransferDirection::S2MM,
            ChannelType::MM2S => TransferDirection::MM2S,
        };
        Transfer::new(bd_config, bd_index, channel, direction, self.col, self.row, self.tile_kind)
    }

    /// Enter a chained next BD with hardware-prefetch semantics.
    ///
    /// Called from `after_transfer_done` when `Use_Next_BD` is set on the
    /// completing BD. Skips the `BdChaining`/`BdSetup`/`MemoryLatency`
    /// pipeline: real hardware fetches the next BD's parameters and warms
    /// its memory pipeline while the prior BD is still streaming, so a
    /// back-to-back chained BD enters `Transferring` the cycle after the
    /// prior one completes -- no inter-BD overhead on the critical path.
    ///
    /// If the BD's acquire lock is granted at call time, transitions
    /// directly to `Transferring` (or `HostPipelineLatency` for shim tiles
    /// touching host DDR, which still pay the NoC/DDR pipeline-fill on
    /// each BD). If the lock isn't yet granted, enters `AcquiringLock`
    /// with `cycles_remaining=0` -- the post-grant latency was also
    /// overlapped, so once the arbiter grants it the FSM transitions
    /// straight to `Transferring`.
    fn enter_chained_bd(
        &mut self,
        ch_idx: usize,
        next_bd: u8,
        _tile: &mut Tile,
        _neighbors: &mut NeighborTiles<'_>,
    ) -> ChannelFsm {
        let bd_addr = self.bd_configs.get(next_bd as usize).map(|c| c.base_addr).unwrap_or(0);
        log::info!(
            "DMA tile({},{}) ch{} BD chain -> BD{} (base_addr=0x{:X}) [prefetched]",
            self.col,
            self.row,
            ch_idx,
            next_bd,
            bd_addr,
        );
        let mut transfer = match self.create_transfer_from_bd(next_bd, ch_idx as u8) {
            Ok(t) => Box::new(t),
            Err(e) => {
                log::warn!(
                    "DMA tile({},{}) ch{} BD chain to {} failed: {:?}",
                    self.col,
                    self.row,
                    ch_idx,
                    next_bd,
                    e
                );
                return ChannelFsm::Error;
            }
        };
        self.channels[ch_idx].current_bd = Some(next_bd);

        // (Recv-side BD-switch accept deassert is driven by the *accept* cursor
        // in `push_stream_in`/`note_stream_accepted`, NOT here: the memory-write
        // completion that reaches this point lags the recv pop, which front-loads
        // the double-buffer through the boundary. See `accept_words_remaining`.)

        // If the BD acquires a lock, we can't grant it inline -- the
        // arbiter only resolves requests submitted in the pre-step
        // submit_lock_requests pass. Enter AcquiringLock with
        // cycles_remaining=0 (no post-grant countdown -- already
        // overlapped during prior BD's transferring); the FSM will
        // submit the request next cycle, the arbiter will resolve,
        // and the acquired-arm's is_first_bd dispatch (false here, BD
        // is chained) routes straight to Transferring/HostPipelineLatency.
        if let Some(lock_id) = transfer.acquire_lock {
            return ChannelFsm::AcquiringLock { lock_id, cycles_remaining: 0, acquired: false, transfer };
        }

        // No lock to acquire. Insert packet header and start transfer (through
        // the BD-switch bubble -- this is a chained-BD boundary).
        self.maybe_insert_packet_header_from_transfer(&mut transfer);
        self.enter_chained_transfer(ch_idx, transfer)
    }

    /// Route a chained BD's transfer into its data phase from a state that has
    /// no inherent idle cycle of its own (the no-lock prefetch path, or a
    /// post-cooldown acquired lock). Inserts the minimum BD-switch bubble
    /// (`bd_switch_bubble_cycles`) so the stream port deasserts for ~1 cycle at
    /// the boundary, unless a longer host-pipeline latency already provides the
    /// gap (then the bubble is absorbed). `bubble == 0` restores the old
    /// back-to-back behavior.
    fn enter_chained_transfer(&mut self, ch_idx: usize, transfer: Box<Transfer>) -> ChannelFsm {
        let host_lat = self.timing_config.host_memory_latency_cycles;
        let bubble = self.take_bd_switch_bubble(ch_idx);
        if host_lat > 0 && transfer.involves_host_memory() {
            ChannelFsm::HostPipelineLatency { cycles_remaining: host_lat, transfer }
        } else if bubble > 0 {
            ChannelFsm::BdSwitchBubble { cycles_remaining: bubble, transfer }
        } else {
            ChannelFsm::Transferring { transfer }
        }
    }

    /// Route a chained locked BD into its data phase the cycle its acquire is
    /// granted. The grant cycle is itself a port-idle cycle (no beat moves), so
    /// it already counts as the first BD-switch bubble cycle:
    /// - `bubble <= 1`: the grant cycle *is* the bubble; data resumes next cycle.
    /// - `bubble >= 2`: idle for the remaining `bubble - 1` cycles.
    ///
    /// The grant cycle moves NO data, and specifically touches no memory bank.
    /// It cannot: a channel in `AcquiringLock` declares no bank demand
    /// (`peek_bank_demand` reports only `Transferring` channels) and this cycle's
    /// arbitration has already closed by the time the grant lands. Doing a
    /// granule fetch here anyway -- which is what the old `bubble == 0` arm did,
    /// inlining a whole transferring cycle -- takes a bank the arbiter never
    /// awarded it: a core storing to that same bank in the same cycle silently
    /// wins too, no CONFLICT_DM_BANK fires, and neither agent pays the cycle
    /// silicon charges them. "Declared no demand" is not the same as "consumed
    /// no bank" (`chained_bd_touches_no_bank_on_its_lock_grant_cycle`).
    fn enter_transfer_after_lock_grant(&mut self, ch_idx: usize, transfer: Box<Transfer>) -> ChannelFsm {
        match self.take_bd_switch_bubble(ch_idx) {
            0 | 1 => ChannelFsm::Transferring { transfer },
            n => ChannelFsm::BdSwitchBubble { cycles_remaining: n - 1, transfer },
        }
    }

    /// Begin the completion sequence after data movement is done.
    ///
    /// Pipelines the release with the final data cycle: applies the BD's
    /// release lock inline (bypassing the arbiter, since releases never
    /// fail and never have a precondition), then goes straight to
    /// `after_transfer_done`. HW also overlaps release with the transfer
    /// tail, so the prior `ReleasingLock` state was an EMU-only dead cycle.
    ///
    /// The `ReleasingLock` FSM state is preserved but no longer reached
    /// from this entry point. It remains valid in case a future scenario
    /// (e.g., arbitrated release with non-trivial latency) needs it.
    /// Route a drained transfer to completion, inserting the one-time non-shim
    /// startup hold (#140) if one is latched for this channel.  When
    /// `startup_hold_cycles > 0` (set once per session by
    /// `consume_first_bd_bonus` on the first BD of a memtile/compute channel),
    /// the channel enters `StartupHold` for that many cycles before its
    /// release/FINISHED_BD fire -- delaying downstream first-output without
    /// backpressuring upstream, since the data is already fully moved.  The
    /// hold is consumed here (cleared), so only the first BD of the session
    /// pays it.
    fn complete_or_hold(
        &mut self,
        ch_idx: usize,
        transfer: Box<Transfer>,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) -> ChannelFsm {
        let hold = self.channels[ch_idx].startup_hold_cycles;
        if hold > 0 {
            self.channels[ch_idx].startup_hold_cycles = 0;
            ChannelFsm::StartupHold { cycles_remaining: hold, transfer }
        } else {
            self.complete_or_drain(ch_idx, transfer, tile, neighbors)
        }
    }

    /// Gate `begin_completion` on the channel's egress stream having drained.
    ///
    /// An MM2S channel has not truly finished its BD until the last word has
    /// handshaked downstream (TVALID&&TREADY), not merely left memory. If the
    /// sink is BEHIND -- the egress FIFO still held words from previous cycles
    /// when this step began (`ChannelContext::egress_backlog`) -- hold the
    /// channel in `DrainingEgress`, deferring the lock release, FINISHED_BD and
    /// BD chaining, so send progress is coupled to the actual downstream drain
    /// rather than to memory-read completion. This stops the MM2S from running
    /// ahead of a slow/stalled sink and pre-filling the pipeline (#140 SP-4a).
    ///
    /// The word this cycle's step just pushed is NOT a backlog: the routing pass
    /// hands it downstream later in this same cycle, so a channel whose sink is
    /// keeping up is done when its last word is pushed. Waiting a cycle for that
    /// word too would spend a second port-idle cycle at the BD boundary, and HW
    /// only ever shows one (`on16 off1` -- the boundary bubble). That extra cycle
    /// used to be hidden by having the next BD's lock-grant cycle inline a whole
    /// transfer cycle, which touched a memory bank outside arbitration -- see
    /// `enter_transfer_after_lock_grant`. The boundary has exactly one idle
    /// cycle; it is the bubble, and no bank access hides in it.
    fn complete_or_drain(
        &mut self,
        ch_idx: usize,
        transfer: Box<Transfer>,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) -> ChannelFsm {
        let is_mm2s = matches!(self.channel_type(ch_idx as u8), ChannelType::MM2S);
        if is_mm2s && self.channels[ch_idx].egress_backlog > 0 {
            // The DrainingEgress cycle(s) are port-idle boundary cycles (the
            // last beat leaving the egress FIFO), so they spend the BD-switch
            // bubble budget -- the chained-BD entry must not add a full bubble
            // on top, or the send cadence double-counts to off2 (#140 SP-4a).
            self.channels[ch_idx].bubble_spent = true;
            ChannelFsm::DrainingEgress { transfer }
        } else {
            self.begin_completion(ch_idx, transfer, tile, neighbors)
        }
    }

    /// The effective BD-switch bubble for a chained-BD entry, net of any
    /// boundary-idle cycle already spent this boundary.  A completion routed
    /// through `DrainingEgress` has already idled the port for the last-beat
    /// handshake, so it consumes one cycle of the `bd_switch_bubble_cycles`
    /// budget (mirrors the lock-grant idle cycle counting as the bubble in
    /// `enter_transfer_after_lock_grant`).  Consumes (clears) the credit.
    /// (#140 SP-4a.)
    fn take_bd_switch_bubble(&mut self, ch_idx: usize) -> u16 {
        let base = self.timing_config.bd_switch_bubble_cycles;
        if self.channels[ch_idx].bubble_spent {
            self.channels[ch_idx].bubble_spent = false;
            base.saturating_sub(1)
        } else {
            base
        }
    }

    fn begin_completion(
        &mut self,
        ch_idx: usize,
        transfer: Box<Transfer>,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) -> ChannelFsm {
        // A transfer can complete directly from a stalled state (e.g. FotFinish
        // on TLAST) without passing through a Continue, so close any asserted
        // held-level here too -- otherwise the starvation/stall span would never
        // deassert and would run to end-of-segment.
        if self.channels[ch_idx].prev_starving {
            self.trace(EventType::DmaStreamStarvation { channel: ch_idx as u8, active: false });
            self.channels[ch_idx].prev_starving = false;
        }
        if self.channels[ch_idx].prev_lock_stalled {
            self.trace(EventType::DmaStalledLock { channel: ch_idx as u8, active: false });
            self.channels[ch_idx].prev_lock_stalled = false;
        }

        let completion = CompletionInfo {
            bd_index: transfer.bd_index,
            next_bd: transfer.next_bd,
            cycles_elapsed: transfer.cycles_elapsed,
            channel: ch_idx as u8,
        };

        if let Some(lock_id) = transfer.release_lock {
            let release_value = transfer.release_value;
            // The FUNCTIONAL semaphore release is ALWAYS prompt (inline at
            // completion): the lock value a waiting consumer acquires must be
            // available immediately. Deferring it deadlocks a DMA->core handoff
            // with no buffer slack -- the compute core is not a DMA channel, so
            // it can neither trigger a buffer swap nor be reached by the
            // deadlock-break flush (see PendingRelease).
            //
            // On a memtile S2MM cross-lock handoff the OBSERVABLE LockRelease
            // trace event is deferred to the BD slot's reuse (HW LOCK_SEL1_REL
            // reflects BD retirement, not completion); the functional release
            // above is unaffected. Other releases (non-memtile, MM2S, self-chain,
            // task-end) trace inline.
            let next_acquire = transfer
                .next_bd
                .and_then(|n| self.bd_configs.get(n as usize))
                .and_then(|nbd| nbd.acquire_lock);
            let cross_lock = matches!(next_acquire, Some(next_acq) if next_acq != lock_id);
            let latency =
                if self.tile_kind.is_mem() && matches!(self.channel_type(ch_idx as u8), ChannelType::S2MM) {
                    self.timing_config.memtile_lock_release_latency_cycles as u64
                } else {
                    0
                };
            if cross_lock && latency > 0 {
                // Functional release now; trace event deferred to BD-slot reuse.
                self.release_lock_value(lock_id, release_value, tile, neighbors, ch_idx);
                self.channels[ch_idx].pending_releases.push(PendingRelease {
                    lock_id,
                    bd_index: completion.bd_index,
                    ready_cycle: self.current_cycle + latency,
                    trace_at: None,
                });
            } else {
                // No trace deferral: functional release + trace event inline.
                self.apply_lock_release_direct(lock_id, release_value, tile, neighbors, ch_idx);
            }
        }
        self.after_transfer_done(ch_idx, completion, tile, neighbors)
    }

    /// Handle post-transfer completion: stats, chaining, repeat, task queue.
    ///
    /// Returns the next FSM state. For chained BDs (Use_Next_BD set), this
    /// skips the `BdChaining`/`BdSetup`/`AcquiringLock-countdown`/
    /// `MemoryLatency` sequence and transitions straight to the next BD's
    /// `Transferring` (or `AcquiringLock` if the lock isn't yet granted) --
    /// hardware prefetches the next BD's parameters during the prior BD's
    /// transferring tail, so the inter-BD overhead is off the critical
    /// path on real silicon. The cold-start path (first BD of a task) is
    /// unaffected; it still enters `BdSetup`/`AcquiringLock` from
    /// `start_channel` and pays the full pipeline-fill cost. See
    /// `docs/superpowers/findings/2026-05-11-emu-bd-chain-pipelining.md`.
    fn after_transfer_done(
        &mut self,
        ch_idx: usize,
        completion: CompletionInfo,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) -> ChannelFsm {
        self.channels[ch_idx].stats.transfers_completed += 1;
        self.channels[ch_idx].stats.cycles_spent += completion.cycles_elapsed;

        // Emit DMA_FINISHED_BD
        self.trace_events
            .push((self.current_cycle, EventType::DmaFinishedBd { channel: ch_idx as u8 }));
        log::info!(
            "DMA tile({},{}) ch{} BD{} FINISH cy={}",
            self.col,
            self.row,
            ch_idx,
            completion.bd_index,
            self.current_cycle
        );

        // Check for BD chaining.
        //
        // Hardware follows next_bd unconditionally when Use_Next_BD is set
        // (represented as next_bd = Some(n) in our model). The hardware does
        // NOT detect chain cycles -- a BD chain that loops (BD0->BD1->BD0)
        // runs indefinitely. This is correct for double-buffered DMA patterns.
        //
        // Chain termination only occurs when Use_Next_BD=0 (next_bd = None).
        // The repeat_count then controls how many additional times the chain
        // is re-executed from chain_start_bd.
        if let Some(next_bd) = completion.next_bd {
            log::debug!(
                "DMA tile({},{}) ch{} chaining to BD {} (from BD {}) [prefetched]",
                self.col,
                self.row,
                ch_idx,
                next_bd,
                completion.bd_index
            );
            return self.enter_chained_bd(ch_idx, next_bd, tile, neighbors);
        }

        // Chain ended (Use_Next_BD=0). Check repeat_count for re-execution.
        if self.channels[ch_idx].repeat_count > 0 {
            self.channels[ch_idx].repeat_count -= 1;
            if let Some(start_bd) = self.channels[ch_idx].chain_start_bd {
                log::debug!(
                    "DMA tile({},{}) ch{} repeating chain from BD {} ({} remaining)",
                    self.col,
                    self.row,
                    ch_idx,
                    start_bd,
                    self.channels[ch_idx].repeat_count
                );
                return ChannelFsm::BdChaining {
                    cycles_remaining: self.timing_config.bd_chain_cycles as u16,
                    next_bd: start_bd,
                };
            }
        }

        // Task complete (no chaining, no repeats)
        self.trace(EventType::DmaFinishedTask { channel: ch_idx as u8 });
        self.maybe_emit_task_token(ch_idx);

        // Check for more tasks in the queue (AIE2+ only).
        if self.dma_model.supports_task_queue() && !self.channels[ch_idx].task_queue.is_empty() {
            log::debug!(
                "DMA tile({},{}) ch{} task complete, {} tasks remaining in queue",
                self.col,
                self.row,
                ch_idx,
                self.channels[ch_idx].task_queue.len()
            );
            self.start_next_queued_task(ch_idx as u8);
            // Re-arm is_first_bd so the queued task pays its first-BD bonus
            // (channel_start + shim_per_task_overhead_* + the decayed warm-up
            // term).  The cold-start no longer fires in full on each task --
            // consume_first_bd_bonus charges cold_start * (decay/1000)^i at
            // the channel's warm_task_index, so back-to-back queued tasks pay
            // a geometrically shrinking warm-up cost as the DDR controller
            // warms.  See finding 2026-05-25-shim-bd-chain-amortization and
            // the Phase 2d warm-up-transient model.
            self.channels[ch_idx].is_first_bd = true;
            return std::mem::take(&mut self.channels[ch_idx].fsm);
        }

        // Channel goes cold; re-arm the first-BD bonus for any future task.
        self.channels[ch_idx].is_first_bd = true;
        ChannelFsm::Idle
    }

    /// Per-cycle word throughput for `transfer`, depending on the NARROWEST
    /// interface it crosses -- and crucially, a DMA touches TWO interfaces
    /// (one at each endpoint), so the bound is direction-specific:
    ///   - shim DMA <-> host DDR: the shim AXI master to DDR (1 word/cyc on
    ///     Phoenix, HW measurement 2026-05-25).
    ///   - MM2S egress (memory -> stream): metered to the 32-bit AXI4-Stream
    ///     beat width (1 word/cyc/port).  Without this the MM2S bursts the
    ///     4-word memory-read rate into the shallow stream FIFO, fragmenting
    ///     PORT_RUNNING at the opening; HW meters egress to 1 word/cyc and
    ///     stays continuously asserted (#140 relay-fill).
    ///   - S2MM ingress (stream -> memory): the memory WRITE side is the
    ///     128-bit data-memory bus (4 words/cyc), NOT the stream beat.  The
    ///     stream READ already metered the fill to 1 word/cyc upstream (the
    ///     routing layer / ingress FIFO); the DMA then drains the staged words
    ///     to memory at the bus rate, bounded by what the ingress holds
    ///     (`min(words_available, words_per_cycle)`, emergent via stall-on-
    ///     empty-FIFO).  Draining at the stream rate instead conflated the two
    ///     interfaces, keeping the ingress full-headroom between buffers and
    ///     inflating the producer's transient lead in the send cascade (#140).
    ///   - otherwise (memory<->memory): the tile data memory bus
    ///     (4 words/cyc, 128-bit DATAMEMORY_WIDTH).
    ///
    /// Shared by `do_transfer_cycle` (the committing path) and
    /// `channel_bank_mask` (the non-committing peek) so the two can never
    /// derive a different per-cycle word count.
    fn words_per_cycle_for(&self, transfer: &Transfer) -> usize {
        if self.tile_kind.is_shim() && transfer.involves_host_memory() {
            self.timing_config.shim_words_per_cycle as usize
        } else if transfer.involves_stream() && transfer.direction == TransferDirection::MM2S {
            self.timing_config.stream_words_per_cycle as usize
        } else {
            self.timing_config.words_per_cycle as usize
        }
    }

    /// Perform one cycle of data transfer for a channel in the Transferring state.
    ///
    /// Extracts transfer parameters, calls do_transfer, and advances the transfer.
    /// The Transfer is borrowed from the FSM via the caller (not from self.channels).
    fn do_transfer_cycle(
        &mut self,
        ch_idx: usize,
        mem_denied: bool,
        transfer: &mut Transfer,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> TransferCycleResult {
        let words_per_cycle = self.words_per_cycle_for(transfer);

        // MM2S memory side: refill the egress staging FIFO with one 16-byte
        // granule, which is this channel's entire bank demand for the cycle
        // (`next_granule_fetch`). A channel that lost the arbitration does not
        // get its granule -- but it keeps whatever the FIFO already holds, and
        // the drain below runs on that. `staged_words` is the only thing this
        // touches, so the fetch cursor (and therefore the demand it re-presents
        // next cycle) is unchanged by a denial.
        if self.uses_egress_staging(transfer) && !mem_denied {
            let layout = self.bank_layout();
            if let Some(fetch) = self.next_granule_fetch(transfer, layout) {
                self.cycle_dma_banks |= fetch.bank_mask;
                transfer.staged_words += fetch.words;
            }
        }

        // SP-4a (#140): shim S2MM cold-start DDR-write drain throttle.  While the
        // cooldown is counting down the DDR row-open / NoC path is still warming,
        // so no word can retire this cycle -- the ingress stays occupied and the
        // upstream memtile MM2S (`of_out` send) is backpressured per-object
        // instead of dumping the pre-filled backlog.  The cooldown decays per
        // drained word (below) so the throttle fades to the steady 1 word/cyc.
        let cold_throttle = self.channels[ch_idx].cold_drain_armed
            && self.tile_kind.is_shim()
            && transfer.involves_host_memory()
            && transfer.direction == TransferDirection::S2MM;
        if cold_throttle && self.channels[ch_idx].cold_drain_cooldown > 0 {
            self.channels[ch_idx].cold_drain_cooldown -= 1;
            return TransferCycleResult::Stalled;
        }

        if transfer.has_zero_padding() {
            // Padding-aware path: one word at a time
            let remaining = transfer.remaining_bytes();
            if remaining == 0 {
                return TransferCycleResult::Continue;
            }

            let action = transfer.next_output_action();
            let is_last_word = remaining <= 4;
            let channel = transfer.channel;
            let tlast_suppress = transfer.effective_tlast_suppress();

            match action {
                PadAction::Zero => {
                    let should_assert_tlast = is_last_word && !tlast_suppress;
                    self.push_stream_out(StreamData { data: 0, tlast: should_assert_tlast, channel });
                    transfer.advance(4);
                    self.channels[ch_idx].stats.bytes_transferred += 4;
                    TransferCycleResult::Continue
                }
                PadAction::Data(addr) => {
                    let source = transfer.source;
                    let dest = transfer.dest;
                    self.word_opens_granule = word_opens_granule(
                        transfer.last_access_addr,
                        addr,
                        self.bank_layout().access_granule_bytes(),
                    );
                    let result = self.do_transfer(
                        source,
                        dest,
                        addr,
                        4,
                        channel,
                        is_last_word,
                        tlast_suppress,
                        tile,
                        neighbors,
                        host_memory,
                    );

                    if result.stall {
                        return TransferCycleResult::Stalled;
                    }
                    if result.success {
                        transfer.advance(4);
                        self.channels[ch_idx].stats.bytes_transferred += 4;
                        if result.fot_finish {
                            TransferCycleResult::FotFinish
                        } else {
                            TransferCycleResult::Continue
                        }
                    } else {
                        TransferCycleResult::Error
                    }
                }
            }
        } else {
            // Standard path: transfer up to words_per_cycle words this cycle.
            //
            // Each word's address comes from the BD's address generator so
            // multi-dimensional stride/wrap configuration (matrix transpose,
            // nd_memcpy_transforms, etc.) is honored. Reading the chunk
            // linearly from a single base address would produce wrong data
            // for any non-contiguous BD.
            let bytes_remaining = transfer.remaining_bytes() as usize;
            if bytes_remaining == 0 {
                return TransferCycleResult::Continue;
            }
            let mut words_this_cycle = words_per_cycle.min((bytes_remaining + 3) / 4);

            let source = transfer.source;
            let dest = transfer.dest;
            let channel = transfer.channel;
            let tlast_suppress = transfer.effective_tlast_suppress();
            let mut fot_finished = false;

            // MM2S stream side: the port can only send words the staging FIFO
            // already holds. If it holds none and the port could have taken a
            // beat, the channel is memory-starved -- the egress ran dry waiting
            // on a granule the memory side has not delivered. That is the whole
            // of DMA_MM2S_n_MEMORY_STARVATION, and with three spare memory
            // cycles in four it should essentially never happen: silicon reads 0
            // on every workload measured, including one where the MM2S lost 112
            // bank arbitrations (2026-07-14-dma-bank-access-width.md).
            let staging = self.uses_egress_staging(transfer);
            if staging {
                words_this_cycle = words_this_cycle.min(transfer.staged_words);
                if words_this_cycle == 0 {
                    if self.can_push_stream_out_for_channel(channel) {
                        let dir_ch = self.per_direction_channel(channel) as usize;
                        if let Some(f) = self.mm2s_egress_empty_wanted.get_mut(dir_ch) {
                            *f = true;
                        }
                        return TransferCycleResult::MemoryStarved;
                    }
                    // The port is backpressured anyway: an empty FIFO the stream
                    // is not asking to drain is not starvation.
                    return TransferCycleResult::Stalled;
                }
            }

            let granule = self.bank_layout().access_granule_bytes();
            for w in 0..words_this_cycle {
                let addr = transfer.current_address();
                let is_last = transfer.remaining_bytes() <= 4;

                // The memory-side access happens once per 128-bit granule; the
                // remaining beats of that granule move through the staging
                // buffer and claim no bank (`word_opens_granule`).  Set BEFORE
                // the word moves: `transfer.advance` updates `last_access_addr`.
                //
                // A staged MM2S word claims NOTHING: its bank access already
                // happened, on the cycle the granule was fetched into the egress
                // FIFO above. Recording it again here would double-count the
                // access and put it in the wrong cycle.
                self.word_opens_granule =
                    !staging && word_opens_granule(transfer.last_access_addr, addr, granule);

                let result = self.do_transfer(
                    source,
                    dest,
                    addr,
                    4,
                    channel,
                    is_last,
                    tlast_suppress,
                    tile,
                    neighbors,
                    host_memory,
                );

                if result.stall {
                    // Mid-chunk stall: if any word made it through this
                    // cycle, report Continue so progress is visible; if
                    // we stalled on the very first word, report Stalled.
                    return if w == 0 {
                        TransferCycleResult::Stalled
                    } else {
                        TransferCycleResult::Continue
                    };
                }
                if !result.success {
                    return TransferCycleResult::Error;
                }

                transfer.advance(4);
                if staging {
                    transfer.staged_words -= 1;
                }
                self.channels[ch_idx].stats.bytes_transferred += 4;

                if result.fot_finish {
                    fot_finished = true;
                    break;
                }
            }

            // SP-4a (#140): a word retired under the cold-start throttle -> the
            // DDR/NoC path warmed one step.  Decay the per-word cooldown
            // geometrically; when it floors to 0 the throttle is done and the
            // steady 1 word/cyc rate resumes.
            if cold_throttle {
                self.channels[ch_idx].cold_drain_word_index += 1;
                let c = self.timing_config.shim_s2mm_cold_drain_cooldown_cycles as u64;
                let r = self.timing_config.shim_s2mm_cold_drain_decay_permille as u64;
                let k = self.channels[ch_idx].cold_drain_word_index as u64;
                let mut term = c;
                for _ in 0..k {
                    term = term * r / 1000;
                }
                if term == 0 {
                    self.channels[ch_idx].cold_drain_armed = false;
                    self.channels[ch_idx].cold_drain_cooldown = 0;
                } else {
                    self.channels[ch_idx].cold_drain_cooldown = term as u16;
                }
            }

            if fot_finished {
                TransferCycleResult::FotFinish
            } else {
                TransferCycleResult::Continue
            }
        }
    }

    /// Bookkeeping every granted DMA acquire performs, whichever FSM path
    /// consumed the grant (`AcquiringLock`, or `enter_chained_bd` consuming a
    /// grant submitted during the previous BD's egress drain): the lock-stall
    /// trace level deasserts, the gated lock recorder logs the acquire, and the
    /// BD slot this acquire RECYCLES has its deferred release-trace scheduled --
    /// HW's LOCK_SEL*_REL trace fires at the BD-ring recycle, not at completion.
    fn on_acquire_granted(&mut self, ch_idx: usize, lock_id: u8, bd_index: u8) {
        if self.channels[ch_idx].prev_lock_stalled {
            self.trace(EventType::DmaStalledLock { channel: ch_idx as u8, active: false });
            self.channels[ch_idx].prev_lock_stalled = false;
        }
        // Only `LockTarget::Own` acquires are in scope for E4.
        if let Some(LockTarget::Own(local_id)) = self.resolve_lock_id(lock_id) {
            if let Some(rec) = &mut self.lock_recorder {
                rec.push(LockEvent {
                    cycle: self.current_cycle,
                    channel_flat: ch_idx as u8,
                    lock_local_id: local_id,
                    op: LockOp::Acquire,
                });
            }
        }
        let now = self.current_cycle;
        if let Some(pr) = self.channels[ch_idx]
            .pending_releases
            .iter_mut()
            .find(|p| p.bd_index == bd_index && p.trace_at.is_none())
        {
            pr.trace_at = Some(pr.ready_cycle.max(now));
        }
    }

    /// Check if a lock acquire was granted by the arbiter.
    ///
    /// The request was submitted in `submit_lock_requests()` before stepping.
    /// This checks whether the arbiter granted it during resolution.
    fn check_acquire_granted(
        &mut self,
        lock_id: u8,
        tile: &Tile,
        neighbors: &NeighborTiles<'_>,
        ch_idx: usize,
    ) -> bool {
        let lock_target = match self.resolve_lock_id(lock_id) {
            Some(target) => target,
            None => return false,
        };

        let (target_tile, local_id): (&Tile, u8) = match lock_target {
            LockTarget::Own(id) => (tile, id),
            LockTarget::West(id) => match neighbors.west.as_deref() {
                Some(west) => (west, id),
                None => return false,
            },
            LockTarget::East(id) => match neighbors.east.as_deref() {
                Some(east) => (east, id),
                None => return false,
            },
        };

        let requestor = self.channel_requestor(ch_idx as u8);
        let granted = target_tile.lock_was_granted(requestor, local_id as usize);
        let lock_value = target_tile.locks.get(local_id as usize).map(|l| l.value).unwrap_or(i8::MIN);

        log::info!(
            "DMA check_acquire_granted tile({},{}) ch{} bd_lock={} target={:?} local_lock={} \
             lock_value={} granted={}",
            self.col,
            self.row,
            ch_idx,
            lock_id,
            lock_target,
            local_id,
            lock_value,
            granted
        );

        if let Some(ref mut timing) = self.lock_timing {
            timing.track_acquire(local_id as usize, granted);
        }

        granted
    }

    /// Execute a lock release operation (post-arbiter).
    ///
    /// The release was already submitted to the arbiter in `submit_lock_requests()`
    /// and resolved before `step()`. This method just logs for debugging.
    /// The lock value was already updated by the arbiter's `resolve()`.
    fn execute_lock_release(
        &mut self,
        lock_id: u8,
        release_value: i8,
        _tile: &mut Tile,
        _neighbors: &mut NeighborTiles<'_>,
    ) {
        log::info!(
            "DMA tile({},{}) lock release bd_lock={} delta={} (applied by arbiter)",
            self.col,
            self.row,
            lock_id,
            release_value
        );
    }

    /// Insert a packet header from a Transfer reference (used during BdSetup).
    ///
    /// Unlike the old maybe_insert_packet_header which accessed self.transfers[],
    /// this takes a Transfer reference directly from the FSM.
    ///
    /// Takes `&mut Transfer` so it can call `mark_packet_header_sent()` after
    /// successful insertion. This prevents double-insertion when the same
    /// Transfer passes through both `start_channel_with_repeat` and `BdSetup`
    /// completion (the no-lock path).
    pub(super) fn maybe_insert_packet_header_from_transfer(&mut self, transfer: &mut Transfer) {
        if !transfer.needs_packet_header() || transfer.direction != TransferDirection::MM2S {
            if transfer.enable_packet && transfer.direction == TransferDirection::MM2S {
                log::warn!(
                    "DMA({},{}) ch{} BD{}: enable_packet=true but needs_packet_header()={} header_sent={}",
                    self.col,
                    self.row,
                    transfer.channel,
                    transfer.bd_index,
                    transfer.needs_packet_header(),
                    transfer.packet_header_sent
                );
            }
            return;
        }

        if let Some(header_word) = transfer.generate_packet_header() {
            self.push_stream_out(StreamData { data: header_word, tlast: false, channel: transfer.channel });
            transfer.mark_packet_header_sent();
            let (hdr, _) = crate::device::stream_switch::PacketHeader::decode(header_word);
            log::info!(
                "DMA({},{}) ch{} BD{} packet header: 0x{:08X} (pkt_id={}, type={:?})",
                self.col,
                self.row,
                transfer.channel,
                transfer.bd_index,
                header_word,
                hdr.stream_id,
                hdr.packet_type
            );
        }
    }

    /// Perform a data transfer operation.
    ///
    /// For stream transfers (MM2S/S2MM), data is buffered in stream_out/stream_in
    /// and will be routed by the TileArray's stream router.
    ///
    /// # Arguments
    /// * `tlast_suppress` - If true, TLAST is not asserted even on the last word
    fn do_transfer(
        &mut self,
        source: TransferEndpoint,
        dest: TransferEndpoint,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> TransferResult {
        match (source, dest) {
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::Stream { .. }) => {
                // MM2S: Read from tile memory, queue to stream output.
                // Backpressure: stall when the DMA's own output queue has
                // saturated the downstream slave port FIFO depth. Models
                // STALL_STRM_STARV on real silicon, where MM2S blocks
                // until the consumer (next switch hop / S2MM / etc.)
                // pulls a word free.  Per-channel: each MM2S channel has
                // its own slave-port FIFO, so one stalled channel does
                // not gate another.
                if !self.can_push_stream_out_for_channel(channel) {
                    return TransferResult::stalled();
                }
                if self.transfer_mm2s(addr, bytes, channel, is_last, tlast_suppress, tile, neighbors) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::Stream { .. }, TransferEndpoint::TileMemory { .. }) => {
                // S2MM: Read from stream input, write to tile memory
                let result = self.transfer_s2mm(addr, bytes, channel, tile, neighbors);
                if result.stall {
                    // No stream data available - stall (not an error)
                    return TransferResult::stalled();
                }
                if result.success {
                    // Check FoT mode: if enabled and TLAST received, signal early finish
                    let fot_mode = self.get_channel_fot_mode(channel);
                    let fot_finish = result.tlast_received && fot_mode != 0;

                    if fot_finish {
                        log::debug!(
                            "DMA({},{}) ch{} FoT mode {} triggered by TLAST ({} bytes written)",
                            self.col,
                            self.row,
                            channel,
                            fot_mode,
                            result.bytes_written
                        );
                    }

                    TransferResult { success: true, stall: false, fot_finish }
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::HostMemory, TransferEndpoint::TileMemory { .. }) => {
                if Self::transfer_host_to_tile_static(addr, bytes, tile, host_memory) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::HostMemory) => {
                // Record bank access for conflict detection (tile read by DMA)
                tile.record_dma_bank_access(addr as u32, bytes);
                if Self::transfer_tile_to_host_static(addr, bytes, tile, host_memory) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::HostMemory, TransferEndpoint::Stream { .. }) => {
                // Shim MM2S: Read from host DDR, queue to stream output.
                // Same backpressure as tile MM2S above -- shim's outgoing
                // stream port stalls when the next slave FIFO is full,
                // which is what propagates the consumer drain rate (memtile
                // S2MM lock waits, compute kernel rate) back to the shim
                // and gates DMA_FINISHED_TASK timing.  Per-channel gate.
                if !self.can_push_stream_out_for_channel(channel) {
                    return TransferResult::stalled();
                }
                if self.transfer_host_to_stream(addr, bytes, channel, is_last, tlast_suppress, host_memory) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::Stream { .. }, TransferEndpoint::HostMemory) => {
                // Shim S2MM: Read from stream input, write to host DDR
                let result = self.transfer_stream_to_host(addr, bytes, channel, host_memory);
                if result.stall {
                    // No stream data available - stall (not an error)
                    return TransferResult::stalled();
                }
                if result.success {
                    // Check FoT mode: if enabled and TLAST received, signal early finish
                    let fot_mode = self.get_channel_fot_mode(channel);
                    let fot_finish = result.tlast_received && fot_mode != 0;

                    if fot_finish {
                        log::debug!(
                            "DMA({},{}) Shim S2MM ch{} FoT mode {} triggered by TLAST ({} bytes written)",
                            self.col,
                            self.row,
                            channel,
                            fot_mode,
                            result.bytes_written
                        );
                    }

                    TransferResult { success: true, stall: false, fot_finish }
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::TileMemory { .. }) => {
                // Tile to tile: Would need array access, mark as success for now
                TransferResult::success()
            }
            _ => TransferResult::failure(),
        }
    }

    /// Resolve an MM2S byte address to (target_tile, offset within it).
    ///
    /// MemTile DMAs use the windowed (West/Own/East) address space per
    /// `mlir-aie` (`getMemLocalBaseAddress`): West=`[0,mem_size)`,
    /// Own=`[mem_size,2*mem_size)`, East=`[2*mem_size,3*mem_size)`.
    ///
    /// **Missing-neighbour fallback:** mlir-aie's `dma_configure_task` path
    /// emits *flat* addresses on AIE2 (no window offset added) -- a BD with
    /// `addr=0x11000` on a column-0 MemTile is intended as Own[0x11000] even
    /// though it lands in the West window. Real hardware silently aliases
    /// the access to Own when the addressed neighbour does not exist (the
    /// neighbour bus has no other endpoint), so we mirror that: West/East
    /// targets with no neighbour fall back to Own at the same byte offset.
    /// Out-of-window addresses (>= 3*mem_size) are still fatal.
    ///
    /// Compute/shim DMAs are unchanged: they wrap at `mem_size`.
    fn resolve_mm2s_target<'a>(
        &mut self,
        addr: u64,
        mem_size: usize,
        own: &'a mut Tile,
        neighbors: &'a mut NeighborTiles<'_>,
    ) -> Option<(MemTileTarget, &'a mut Tile, usize)> {
        if !self.tile_kind.is_mem() {
            return Some((MemTileTarget::Own, own, Self::wrap_local_offset(addr, mem_size)));
        }
        match MemTileTarget::resolve(addr, mem_size) {
            Ok((MemTileTarget::Own, off)) => Some((MemTileTarget::Own, own, off)),
            Ok((MemTileTarget::West, off)) => {
                Some(neighbors.west.as_deref_mut().map(|t| (MemTileTarget::West, t, off)).unwrap_or((
                    MemTileTarget::Own,
                    own,
                    off,
                )))
            }
            Ok((MemTileTarget::East, off)) => {
                Some(neighbors.east.as_deref_mut().map(|t| (MemTileTarget::East, t, off)).unwrap_or((
                    MemTileTarget::Own,
                    own,
                    off,
                )))
            }
            Err(e) => {
                let msg = format!(
                    "DMA MemTile({},{}) MM2S addr=0x{:X} outside three-window MemTile space (window_size=0x{:X})",
                    self.col, self.row, e.byte_addr, e.mem_size,
                );
                log::error!("{}", msg);
                self.fatal_errors.push(msg);
                None
            }
        }
    }

    /// Resolve an S2MM byte address to (target_tile, offset within it).
    ///
    /// Mutable counterpart to `resolve_mm2s_target`; same windowing and
    /// missing-neighbour fallback semantics. Returns a `&mut Tile` so the
    /// caller can write through the MemTile-to-MemTile shared-memory bus
    /// (or the local data memory if the West/East target is absent).
    fn resolve_s2mm_target<'a>(
        &mut self,
        addr: u64,
        mem_size: usize,
        own: &'a mut Tile,
        neighbors: &'a mut NeighborTiles<'_>,
    ) -> Option<(MemTileTarget, &'a mut Tile, usize)> {
        if !self.tile_kind.is_mem() {
            return Some((MemTileTarget::Own, own, Self::wrap_local_offset(addr, mem_size)));
        }
        match MemTileTarget::resolve(addr, mem_size) {
            Ok((MemTileTarget::Own, off)) => Some((MemTileTarget::Own, own, off)),
            Ok((MemTileTarget::West, off)) => Some(match neighbors.west.as_deref_mut() {
                Some(t) => (MemTileTarget::West, t, off),
                None => (MemTileTarget::Own, own, off),
            }),
            Ok((MemTileTarget::East, off)) => Some(match neighbors.east.as_deref_mut() {
                Some(t) => (MemTileTarget::East, t, off),
                None => (MemTileTarget::Own, own, off),
            }),
            Err(e) => {
                let msg = format!(
                    "DMA MemTile({},{}) S2MM addr=0x{:X} outside three-window MemTile space (window_size=0x{:X})",
                    self.col, self.row, e.byte_addr, e.mem_size,
                );
                log::error!("{}", msg);
                self.fatal_errors.push(msg);
                None
            }
        }
    }

    /// MM2S: Read from tile memory and queue to stream output.
    ///
    /// For MemTile DMAs, the byte address is decoded into a West/Own/East
    /// window via `MemTileTarget::resolve()`. Compute/shim DMAs always read
    /// from the local tile (legacy `addr % mem_size` wrap).
    ///
    /// # Arguments
    /// * `tlast_suppress` - If true, TLAST is not asserted even on the last word
    pub(super) fn transfer_mm2s(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) -> bool {
        let mem_size = tile.data_memory().len();

        // Resolve target tile and offset.
        // MemTile uses the windowed (West/Own/East) address space; non-MemTile
        // tiles wrap at mem_size.
        let (target_kind, target_tile, offset) =
            match self.resolve_mm2s_target(addr, mem_size, tile, neighbors) {
                Some(r) => r,
                None => return false,
            };

        // Record bank access for conflict detection (offset is local to the
        // target tile), but only on the word that opens a memory-access granule
        // -- the other three beats of a 128-bit read come from the staging
        // buffer and never reach a bank (`word_opens_granule`).
        if self.word_opens_granule {
            self.cycle_dma_banks |=
                crate::device::banking::banks_for_access(offset as u32, bytes, self.bank_layout());
        }

        if offset + bytes > mem_size {
            let msg = format!(
                "DMA({},{}) MM2S addr=0x{:X} bytes={} wraps past memory end (size=0x{:X}) -- bus error",
                self.col, self.row, addr, bytes, mem_size,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            return false;
        }

        // When compression is enabled, read 32-byte blocks and compress each one.
        // Compressed output (mask + packed non-zero bytes) is pushed as 32-bit stream words.
        if self.is_compression_enabled(channel) {
            return self.transfer_mm2s_compressed(
                target_tile,
                target_kind,
                offset,
                bytes,
                channel,
                is_last,
                tlast_suppress,
            );
        }

        let data = target_tile.data_memory();

        // Uncompressed path: read data from tile memory in 32-bit words
        let word_count = (bytes + 3) / 4;

        log::debug!(
            "DMA({},{}) MM2S ch{}: addr=0x{:X} offset=0x{:X} bytes={} words={}",
            self.col,
            self.row,
            channel,
            addr,
            offset,
            bytes,
            word_count
        );

        for i in 0..word_count {
            let word_offset = offset + i * 4;
            let word = if word_offset + 4 <= data.len() {
                u32::from_le_bytes([
                    data[word_offset],
                    data[word_offset + 1],
                    data[word_offset + 2],
                    data[word_offset + 3],
                ])
            } else {
                // Partial word at end
                let mut word_bytes = [0u8; 4];
                for j in 0..4 {
                    if word_offset + j < data.len() {
                        word_bytes[j] = data[word_offset + j];
                    }
                }
                u32::from_le_bytes(word_bytes)
            };

            if i < 2 || i == word_count - 1 {
                log::debug!(
                    "DMA({},{}) MM2S ch{} word[{}]: offset=0x{:X} value=0x{:08X}",
                    self.col,
                    self.row,
                    channel,
                    i,
                    word_offset,
                    word
                );
            }

            if crate::debug::watch::is_watched(word_offset as u64, 4) {
                crate::debug::watch::log_dma_read(
                    self.current_cycle,
                    self.col,
                    self.row,
                    word_offset as u64,
                    word,
                    &format!("MM2S{}", channel),
                );
            }

            // TLAST is asserted on last word unless suppressed
            // AM025: TLAST_Suppress (Word 5, bit 31) prevents TLAST assertion
            let is_last_word = is_last && (i == word_count - 1);
            let should_assert_tlast = is_last_word && !tlast_suppress;
            self.push_stream_out(StreamData { data: word, tlast: should_assert_tlast, channel });
        }

        // Fire WATCHPOINT_N events on the target tile for each word the DMA
        // read. The HW comparator sits at the bank interface and sees DMA
        // traffic the same as core traffic; AM025 distinguishes them via the
        // DMA_Access filter bit. Cross-tile (West/East) reads through the
        // MemTile-to-MemTile shared bus fire on the *target* tile with
        // `Neighbour(<direction-from-target>)` so the neighbour-side
        // quadrant filter bits work as documented.
        let cycle = self.current_cycle;
        let origin = dma_access_origin(target_kind);
        for i in 0..word_count {
            let word_addr = (offset + i * 4) as u32;
            fire_watchpoint_events_with_origin(target_tile, word_addr, false, cycle, None, origin);
        }

        true
    }

    /// MM2S compressed path: read 32-byte blocks, compress, push to stream.
    ///
    /// Sparsity compression (AM020 Ch1) operates on 256-bit (32-byte) blocks.
    /// Each block produces a 32-bit mask followed by packed non-zero bytes,
    /// padded to a 4-byte boundary. The compressed output is pushed as 32-bit
    /// stream words.
    fn transfer_mm2s_compressed(
        &mut self,
        target_tile: &mut Tile,
        target_kind: MemTileTarget,
        offset: usize,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
    ) -> bool {
        const BLOCK_SIZE: usize = 32;

        log::debug!(
            "DMA({},{}) MM2S ch{} COMPRESSED: offset=0x{:X} bytes={}",
            self.col,
            self.row,
            channel,
            offset,
            bytes
        );

        let num_blocks = (bytes + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for block_idx in 0..num_blocks {
            let block_start = offset + block_idx * BLOCK_SIZE;
            let data = target_tile.data_memory();
            let block_end = (block_start + BLOCK_SIZE).min(offset + bytes).min(data.len());
            let block_len = block_end - block_start;

            // Pad to 32 bytes if this is a partial final block
            let mut block = [0u8; BLOCK_SIZE];
            block[..block_len].copy_from_slice(&data[block_start..block_end]);

            let compressed = match compression::compress(&block) {
                Some(c) => c,
                None => {
                    let msg = format!(
                        "DMA({},{}) MM2S ch{} compression failed at block {} -- data corruption",
                        self.col, self.row, channel, block_idx,
                    );
                    log::error!("{}", msg);
                    self.fatal_errors.push(msg);
                    return false;
                }
            };

            log::debug!(
                "DMA({},{}) MM2S ch{} block {}: {} bytes -> {} compressed bytes",
                self.col,
                self.row,
                channel,
                block_idx,
                BLOCK_SIZE,
                compressed.len()
            );

            // Push compressed bytes as 32-bit stream words
            let compressed_words = compressed.len() / 4;
            let is_last_block = is_last && block_idx == num_blocks - 1;

            for w in 0..compressed_words {
                let wi = w * 4;
                let word = u32::from_le_bytes([
                    compressed[wi],
                    compressed[wi + 1],
                    compressed[wi + 2],
                    compressed[wi + 3],
                ]);

                let is_last_word = is_last_block && w == compressed_words - 1;
                let should_assert_tlast = is_last_word && !tlast_suppress;
                self.push_stream_out(StreamData { data: word, tlast: should_assert_tlast, channel });
            }
        }

        // Fire WATCHPOINT_N events per word covered by the source range -- the
        // bank comparator sees the same byte addresses regardless of
        // compression. Cross-tile origins follow the same mapping as the
        // uncompressed path.
        let cycle = self.current_cycle;
        let origin = dma_access_origin(target_kind);
        let word_count = (bytes + 3) / 4;
        for i in 0..word_count {
            let word_addr = (offset + i * 4) as u32;
            fire_watchpoint_events_with_origin(target_tile, word_addr, false, cycle, None, origin);
        }

        true
    }

    /// S2MM: Read from stream input, write to tile memory.
    ///
    /// Only transfers data that is available in stream_in for the specified channel.
    /// Returns S2mmResult indicating success, TLAST reception, and bytes written.
    pub(super) fn transfer_s2mm(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) -> S2mmResult {
        let mem_size = tile.data_memory().len();

        // Resolve target tile and offset.
        // MemTile uses the windowed (West/Own/East) address space; non-MemTile
        // tiles wrap at mem_size.
        let (target_kind, target_tile, offset) = match self
            .resolve_s2mm_target(addr, mem_size, tile, neighbors)
        {
            Some(r) => r,
            None => {
                return S2mmResult { success: false, stall: false, tlast_received: false, bytes_written: 0 }
            }
        };

        if offset + bytes > mem_size {
            let msg = format!(
                "DMA({},{}) S2MM addr=0x{:X} bytes={} wraps past memory end (size=0x{:X}) -- bus error",
                self.col, self.row, addr, bytes, mem_size,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            return S2mmResult { success: false, stall: false, tlast_received: false, bytes_written: 0 };
        }

        // When decompression is enabled, consume compressed blocks from stream
        // and write decompressed 32-byte blocks to memory. Decompression has
        // its own word-by-word stall logic so skip the atomic beat check.
        if self.is_decompression_enabled(channel) {
            if !self.has_stream_in_for_channel(channel) {
                return S2mmResult { success: true, stall: true, tlast_received: false, bytes_written: 0 };
            }
            // Bank is touched only when we actually consume from the stream,
            // and only on the word that opens a memory-access granule.
            if self.word_opens_granule {
                self.cycle_dma_banks |=
                    crate::device::banking::banks_for_access(offset as u32, bytes, self.bank_layout());
            }
            return self.transfer_s2mm_decompressed(offset, bytes, channel, target_kind, target_tile);
        }

        // Uncompressed path: the DMA bus transfers atomically -- all words
        // for this beat must be available in the stream before we proceed.
        let words_needed = (bytes + 3) / 4;
        let words_available = self.stream_in_count_for_channel(channel);
        if words_available < words_needed {
            return S2mmResult { success: true, stall: true, tlast_received: false, bytes_written: 0 };
        }

        // Bank access is recorded only after the stream-availability check.
        // A stalled S2MM (no upstream data) does NOT issue a memory access on
        // real hardware, so it must not register a bank touch -- otherwise
        // every stall cycle produces a phantom CONFLICT_DM_BANK_N + MEMORY_STALL
        // when a core happens to load from the same bank. Likewise a word that
        // merely fills the current 128-bit granule's staging buffer issues no
        // memory access of its own (`word_opens_granule`).
        if self.word_opens_granule {
            self.cycle_dma_banks |=
                crate::device::banking::banks_for_access(offset as u32, bytes, self.bank_layout());
        }

        // On real hardware, S2MM ignores TLAST unless Finish-on-TLAST (FoT)
        // mode is enabled. Without FoT, the DMA continues accepting words
        // until the BD's transfer length is satisfied, regardless of TLAST
        // markers on the stream.
        let fot_enabled = self.get_channel_fot_mode(channel) != 0;

        // Write data to tile memory in 32-bit words.
        // For MemTile DMAs this may be a neighbour's memory via the shared bus.
        let data = target_tile.data_memory_mut();
        let mut bytes_written = 0;
        let word_count = (bytes + 3) / 4;
        let mut tlast_received = false;

        log::debug!(
            "DMA({},{}) S2MM ch{}: addr=0x{:X} offset=0x{:X} bytes={} words={}",
            self.col,
            self.row,
            channel,
            addr,
            offset,
            bytes,
            word_count
        );
        for word_idx in 0..word_count {
            // Get data from stream for this specific channel
            if let Some(stream_data) = self.pop_stream_in_for_channel(channel) {
                let word = stream_data.data;
                let word_bytes = word.to_le_bytes();

                if word_idx < 2 || word_idx == word_count - 1 {
                    log::debug!(
                        "DMA({},{}) S2MM ch{} word[{}]: offset=0x{:X} value=0x{:08X}",
                        self.col,
                        self.row,
                        channel,
                        word_idx,
                        offset + bytes_written,
                        word
                    );
                }
                for j in 0..4 {
                    if bytes_written + j < bytes && offset + bytes_written + j < data.len() {
                        data[offset + bytes_written + j] = word_bytes[j];
                    }
                }

                if crate::debug::watch::is_watched((offset + bytes_written) as u64, 4) {
                    crate::debug::watch::log_dma_write(
                        self.current_cycle,
                        self.col,
                        self.row,
                        (offset + bytes_written) as u64,
                        word,
                        &format!("S2MM{}", channel),
                    );
                }

                bytes_written += 4;

                // Track TLAST for FoT mode
                if stream_data.tlast {
                    tlast_received = true;
                    if fot_enabled {
                        break; // FoT: stop receiving at TLAST boundary
                    }
                }
            } else {
                // No more stream data for this channel - transfer partial, continue next step
                break;
            };
        }

        // Fire WATCHPOINT_N events for each word actually written. The bank
        // comparator sees writes through the same path as core stores; AM025
        // distinguishes them via the DMA_Access filter bit. Cross-tile
        // (West/East) writes via the MemTile-to-MemTile shared bus fire on
        // the *target* tile with `Neighbour(<direction-from-target>)`.
        let cycle = self.current_cycle;
        let origin = dma_access_origin(target_kind);
        let words_written = bytes_written / 4;
        for i in 0..words_written {
            let word_addr = (offset + i * 4) as u32;
            fire_watchpoint_events_with_origin(target_tile, word_addr, true, cycle, None, origin);
        }

        S2mmResult { success: true, stall: false, tlast_received, bytes_written }
    }

    /// S2MM decompressed path: read compressed blocks from stream, decompress, write to memory.
    ///
    /// Sparsity decompression (AM020 Ch1) consumes compressed blocks from the stream.
    /// Each block starts with a 32-bit mask word, followed by ceil(popcount(mask)/4)
    /// data words containing packed non-zero bytes. The decompressed 32-byte output
    /// is written to tile memory.
    ///
    /// `target_tile` is the resolved destination (own or neighbour MemTile);
    /// `offset` is local to that tile's data memory.
    fn transfer_s2mm_decompressed(
        &mut self,
        offset: usize,
        bytes: usize,
        channel: u8,
        target_kind: MemTileTarget,
        target_tile: &mut Tile,
    ) -> S2mmResult {
        const BLOCK_SIZE: usize = 32;

        log::debug!(
            "DMA({},{}) S2MM ch{} DECOMPRESSED: offset=0x{:X} bytes={}",
            self.col,
            self.row,
            channel,
            offset,
            bytes
        );

        let mut mem_bytes_written: usize = 0;
        let mut tlast_received = false;

        // Process compressed blocks until we have written enough decompressed bytes
        while mem_bytes_written < bytes {
            // Read the mask word (first word of compressed block)
            let mask_data = match self.pop_stream_in_for_channel(channel) {
                Some(sd) => sd,
                None => break, // No more stream data; partial transfer
            };

            if mask_data.tlast {
                // TLAST on the mask word itself -- unusual, but handle gracefully.
                // Treat as an empty block (mask only, no data bytes).
                tlast_received = true;
                break;
            }

            let mask = mask_data.data;
            let non_zero_count = mask.count_ones() as usize;
            let data_bytes_needed = non_zero_count;
            let data_words_needed = (data_bytes_needed + 3) / 4;

            // Collect the data words for this compressed block
            let mut compressed_buf = Vec::with_capacity(4 + data_words_needed * 4);
            compressed_buf.extend_from_slice(&mask.to_le_bytes());

            let mut got_tlast = false;
            for _ in 0..data_words_needed {
                match self.pop_stream_in_for_channel(channel) {
                    Some(sd) => {
                        compressed_buf.extend_from_slice(&sd.data.to_le_bytes());
                        if sd.tlast {
                            got_tlast = true;
                            break;
                        }
                    }
                    None => break, // Stream starved mid-block
                }
            }

            // Decompress (tolerates short input: missing bytes decompress as zero)
            match compression::decompress(&compressed_buf) {
                Some(decompressed) => {
                    let data = target_tile.data_memory_mut();
                    let write_len = BLOCK_SIZE.min(bytes - mem_bytes_written);
                    let dest_start = offset + mem_bytes_written;
                    let dest_end = (dest_start + write_len).min(data.len());
                    let actual_write = dest_end - dest_start;

                    data[dest_start..dest_end].copy_from_slice(&decompressed[..actual_write]);
                    mem_bytes_written += actual_write;

                    log::debug!("DMA({},{}) S2MM ch{} decompressed block: mask=0x{:08X} {} non-zero -> {} bytes to mem",
                        self.col, self.row, channel, mask, non_zero_count, actual_write);
                }
                None => {
                    let msg = format!(
                        "DMA({},{}) S2MM ch{} decompression failed (mask=0x{:08X}, buf_len={}) -- data corruption",
                        self.col, self.row, channel, mask, compressed_buf.len(),
                    );
                    log::error!("{}", msg);
                    self.fatal_errors.push(msg);
                    return S2mmResult {
                        success: false,
                        stall: false,
                        tlast_received: false,
                        bytes_written: mem_bytes_written,
                    };
                }
            }

            if got_tlast {
                tlast_received = true;
                break;
            }
        }

        // Fire WATCHPOINT_N events per word actually written. Cross-tile
        // origins follow the same mapping as the uncompressed S2MM path.
        let cycle = self.current_cycle;
        let origin = dma_access_origin(target_kind);
        let words_written = mem_bytes_written / 4;
        for i in 0..words_written {
            let word_addr = (offset + i * 4) as u32;
            fire_watchpoint_events_with_origin(target_tile, word_addr, true, cycle, None, origin);
        }

        S2mmResult {
            success: mem_bytes_written > 0,
            stall: false,
            tlast_received,
            bytes_written: mem_bytes_written,
        }
    }

    /// Shim MM2S: Read from host DDR and queue to stream output.
    ///
    /// # Arguments
    /// * `tlast_suppress` - If true, TLAST is not asserted even on the last word
    fn transfer_host_to_stream(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        host_memory: &HostMemory,
    ) -> bool {
        // When compression is enabled, read 32-byte blocks and compress
        if self.is_compression_enabled(channel) {
            return self.transfer_host_to_stream_compressed(
                addr,
                bytes,
                channel,
                is_last,
                tlast_suppress,
                host_memory,
            );
        }

        // Uncompressed path: read data from host memory in 32-bit words
        let word_count = (bytes + 3) / 4;

        log::debug!("MM2S transfer: addr=0x{:X} bytes={} words={}", addr, bytes, word_count);
        for i in 0..word_count {
            let word_addr = addr + (i * 4) as u64;
            let word = host_memory.read_u32(word_addr);

            if crate::debug::watch::is_watched(word_addr, 4) {
                crate::debug::watch::log_dma_read(
                    self.current_cycle,
                    self.col,
                    self.row,
                    word_addr,
                    word,
                    &format!("MM2S{}", channel),
                );
            }

            if i < 4 {
                log::debug!("  MM2S word[{}]: addr=0x{:X} -> 0x{:08X}", i, word_addr, word);
            }

            // TLAST is asserted on last word unless suppressed
            // AM025: TLAST_Suppress (Word 5, bit 31) prevents TLAST assertion
            let is_last_word = is_last && i == word_count - 1;
            let should_assert_tlast = is_last_word && !tlast_suppress;
            self.push_stream_out(StreamData { data: word, channel, tlast: should_assert_tlast });
        }

        true
    }

    /// Shim MM2S compressed path: read 32-byte blocks from host DDR, compress, push to stream.
    fn transfer_host_to_stream_compressed(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        host_memory: &HostMemory,
    ) -> bool {
        const BLOCK_SIZE: usize = 32;

        log::debug!("Shim MM2S COMPRESSED: addr=0x{:X} bytes={}", addr, bytes);

        let num_blocks = (bytes + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for block_idx in 0..num_blocks {
            let block_start = addr + (block_idx * BLOCK_SIZE) as u64;
            let block_remaining = bytes - block_idx * BLOCK_SIZE;
            let block_len = BLOCK_SIZE.min(block_remaining);

            // Read 32-byte block from host memory
            let mut block = [0u8; BLOCK_SIZE];
            for i in 0..block_len {
                let byte_addr = block_start + i as u64;
                // Read individual bytes using word-aligned reads
                let word_addr = byte_addr & !3;
                let byte_offset = (byte_addr & 3) as usize;
                let word = host_memory.read_u32(word_addr);
                block[i] = word.to_le_bytes()[byte_offset];
            }

            let compressed = match compression::compress(&block) {
                Some(c) => c,
                None => {
                    log::error!("Shim MM2S ch{} compression failed at block {}", channel, block_idx);
                    return false;
                }
            };

            let compressed_words = compressed.len() / 4;
            let is_last_block = is_last && block_idx == num_blocks - 1;

            for w in 0..compressed_words {
                let wi = w * 4;
                let word = u32::from_le_bytes([
                    compressed[wi],
                    compressed[wi + 1],
                    compressed[wi + 2],
                    compressed[wi + 3],
                ]);

                let is_last_word = is_last_block && w == compressed_words - 1;
                let should_assert_tlast = is_last_word && !tlast_suppress;
                self.push_stream_out(StreamData { data: word, channel, tlast: should_assert_tlast });
            }
        }

        true
    }

    /// Shim S2MM: Read from stream input and write to host DDR.
    ///
    /// Returns S2mmResult to properly handle stalls when no stream data is available.
    fn transfer_stream_to_host(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        host_memory: &mut HostMemory,
    ) -> S2mmResult {
        // Atomic beat: stall until stream has enough words for this transfer.
        let words_needed = (bytes + 3) / 4;
        let words_available = self.stream_in_count_for_channel(channel);
        if words_available < words_needed {
            return S2mmResult { success: true, stall: true, tlast_received: false, bytes_written: 0 };
        }

        // When decompression is enabled, consume compressed blocks from stream
        if self.is_decompression_enabled(channel) {
            return self.transfer_stream_to_host_decompressed(addr, bytes, channel, host_memory);
        }

        // On real hardware, S2MM ignores TLAST unless Finish-on-TLAST (FoT)
        // mode is enabled. Without FoT, the DMA continues accepting words
        // until the BD's transfer length is satisfied, regardless of TLAST
        // markers on the stream. Breaking unconditionally on TLAST causes
        // address pointer skips when TLAST falls mid-batch (e.g., 10-word
        // source chunks vs 4-word DMA bus width).
        let fot_enabled = self.get_channel_fot_mode(channel) != 0;

        // Uncompressed path: write data to host memory in 32-bit words
        let mut bytes_written = 0;
        let word_count = (bytes + 3) / 4;
        let mut tlast_received = false;

        for i in 0..word_count {
            let stream_data = if let Some(sd) = self.pop_stream_in_for_channel(channel) {
                sd
            } else {
                // No more stream data for this channel - transfer partial, continue next step
                break;
            };

            let word = stream_data.data;
            let word_addr = addr + (i * 4) as u64;
            log::debug!("Shim S2MM write: addr=0x{:X} word=0x{:08X}", word_addr, word);
            host_memory.write_u32(word_addr, word);

            if crate::debug::watch::is_watched(word_addr, 4) {
                crate::debug::watch::log_dma_write(
                    self.current_cycle,
                    self.col,
                    self.row,
                    word_addr,
                    word,
                    &format!("S2MM{}", channel),
                );
            }

            bytes_written += 4;

            // Track TLAST for FoT mode
            if stream_data.tlast {
                tlast_received = true;
                if fot_enabled {
                    break; // FoT: stop receiving at TLAST boundary
                }
            }
        }

        S2mmResult { success: bytes_written > 0, stall: false, tlast_received, bytes_written }
    }

    /// Shim S2MM decompressed path: read compressed blocks from stream, decompress, write to host DDR.
    fn transfer_stream_to_host_decompressed(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        host_memory: &mut HostMemory,
    ) -> S2mmResult {
        const BLOCK_SIZE: usize = 32;

        log::debug!("Shim S2MM DECOMPRESSED: addr=0x{:X} bytes={}", addr, bytes);

        let mut mem_bytes_written: usize = 0;
        let mut tlast_received = false;

        while mem_bytes_written < bytes {
            // Read the mask word
            let mask_data = match self.pop_stream_in_for_channel(channel) {
                Some(sd) => sd,
                None => break,
            };

            if mask_data.tlast {
                tlast_received = true;
                break;
            }

            let mask = mask_data.data;
            let non_zero_count = mask.count_ones() as usize;
            let data_words_needed = (non_zero_count + 3) / 4;

            let mut compressed_buf = Vec::with_capacity(4 + data_words_needed * 4);
            compressed_buf.extend_from_slice(&mask.to_le_bytes());

            let mut got_tlast = false;
            for _ in 0..data_words_needed {
                match self.pop_stream_in_for_channel(channel) {
                    Some(sd) => {
                        compressed_buf.extend_from_slice(&sd.data.to_le_bytes());
                        if sd.tlast {
                            got_tlast = true;
                            break;
                        }
                    }
                    None => break,
                }
            }

            match compression::decompress(&compressed_buf) {
                Some(decompressed) => {
                    let write_len = BLOCK_SIZE.min(bytes - mem_bytes_written);
                    let dest_addr = addr + mem_bytes_written as u64;

                    // Write decompressed bytes to host memory via word-aligned writes
                    let full_words = write_len / 4;
                    for w in 0..full_words {
                        let wi = w * 4;
                        let word = u32::from_le_bytes([
                            decompressed[wi],
                            decompressed[wi + 1],
                            decompressed[wi + 2],
                            decompressed[wi + 3],
                        ]);
                        host_memory.write_u32(dest_addr + wi as u64, word);
                    }
                    // Handle trailing bytes (partial last word)
                    let trailing = write_len % 4;
                    if trailing > 0 {
                        let wi = full_words * 4;
                        let mut word_bytes = [0u8; 4];
                        word_bytes[..trailing].copy_from_slice(&decompressed[wi..wi + trailing]);
                        host_memory.write_u32(dest_addr + wi as u64, u32::from_le_bytes(word_bytes));
                    }

                    mem_bytes_written += write_len;
                }
                None => {
                    log::error!("Shim S2MM ch{} decompression failed (mask=0x{:08X})", channel, mask);
                    return S2mmResult {
                        success: false,
                        stall: false,
                        tlast_received: false,
                        bytes_written: mem_bytes_written,
                    };
                }
            }

            if got_tlast {
                tlast_received = true;
                break;
            }
        }

        S2mmResult {
            success: mem_bytes_written > 0,
            stall: false,
            tlast_received,
            bytes_written: mem_bytes_written,
        }
    }

    /// Transfer data from host memory to tile memory (static version).
    fn transfer_host_to_tile_static(
        tile_addr: u64,
        bytes: usize,
        tile: &mut Tile,
        host_memory: &mut HostMemory,
    ) -> bool {
        let offset = tile_addr as usize;
        if offset + bytes > tile.data_memory().len() {
            return false;
        }

        // Record bank access for conflict detection
        tile.record_dma_bank_access(offset as u32, bytes);

        // Read from host memory
        let mut buf = vec![0u8; bytes];
        // Note: In a real implementation, we'd track a separate host address
        // For now, we use the tile address as an offset into host memory too
        host_memory.read_bytes(tile_addr, &mut buf);

        // Write to tile memory
        tile.data_memory_mut()[offset..offset + bytes].copy_from_slice(&buf);

        true
    }

    /// Transfer data from tile memory to host memory (static version).
    fn transfer_tile_to_host_static(
        tile_addr: u64,
        bytes: usize,
        tile: &Tile,
        host_memory: &mut HostMemory,
    ) -> bool {
        let offset = tile_addr as usize;
        if offset + bytes > tile.data_memory().len() {
            return false;
        }

        // Read from tile memory
        let data = &tile.data_memory()[offset..offset + bytes];

        // Write to host memory
        // Note: In a real implementation, we'd track a separate host address
        host_memory.write_bytes(tile_addr, data);

        true
    }

    // NOTE: complete_transfer() and finish_complete_transfer() have been
    // replaced by begin_completion() and after_transfer_done() in the FSM.

    /// Emit a task complete token if Enable_Token_Issue is set for this channel.
    fn maybe_emit_task_token(&mut self, ch_idx: usize) {
        let config = &self.channels[ch_idx].task_config;

        if config.enable_token_issue {
            log::debug!(
                "DMA tile({},{}) ch{} emitting task complete token (controller_id={})",
                self.col,
                self.row,
                ch_idx,
                config.controller_id
            );

            self.task_tokens.issue(ch_idx as u8, config.controller_id);

            // Clear enable_token_issue after issuing (it's set per-task via Start_Queue)
            self.channels[ch_idx].task_config.enable_token_issue = false;
        }
    }
}

#[cfg(test)]
mod bank_demand_tests {
    use super::*;
    use crate::device::banking::BankLayout;

    /// Build a compute-tile S2MM engine (channel 0) sitting mid-transfer
    /// inside `Transferring`, with more data already staged in `stream_in`
    /// for its next cycle of movement -- so the peek/denial tests exercise a
    /// channel that is genuinely active, not merely configured.
    ///
    /// BD 0 is 32 bytes (8 words) at tile-local 0x400, no lock. The compute
    /// S2MM memory-bus rate is 4 words/cycle (`words_per_cycle`), so the
    /// fixture consumes exactly the first 4 words (16 bytes) during setup,
    /// leaving the channel genuinely mid-transfer with the second 4 words
    /// already staged for the caller.
    fn fixture_s2mm_engine_mid_transfer() -> (DmaEngine, Tile, HostMemory) {
        let mut eng = DmaEngine::new_compute_tile(1, 2);
        let mut tile = Tile::compute(1, 2);
        let mut host_mem = HostMemory::new();

        eng.configure_bd(0, BdConfig::simple_1d(0x400, 32)).unwrap();
        eng.start_channel(0, 0).unwrap(); // channel 0 is S2MM on a compute tile

        // Walk BdSetup/MemoryLatency until Transferring (no lock configured,
        // so this is a fixed handful of cycles).
        let mut guard = 0;
        while eng.channel_phase(0) != "Transferring" {
            eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
            guard += 1;
            assert!(guard < 50, "fixture did not reach Transferring in time");
        }

        // Stage and consume the first cycle's 4 words so the channel is
        // genuinely mid-transfer.
        for i in 0..4u32 {
            eng.push_stream_in(StreamData { data: 0xAAAA_0000 | i, tlast: false, channel: 0 });
        }
        eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        assert_eq!(
            eng.channel_stats(0).unwrap().bytes_transferred,
            16,
            "fixture setup must consume exactly one cycle's words"
        );
        assert_eq!(eng.channel_phase(0), "Transferring", "fixture must still have work left");

        // Stage the second (final) cycle's 4 words for the caller's tests.
        for i in 4..8u32 {
            eng.push_stream_in(StreamData { data: 0xAAAA_0000 | i, tlast: false, channel: 0 });
        }

        (eng, tile, host_mem)
    }

    #[test]
    fn dma_peek_does_not_advance_state() {
        // peek_bank_demand takes &self -- the borrow checker already proves
        // it cannot mutate the engine, so there is no point Debug-comparing
        // the engine across this call (that used to be here). The meaningful
        // full-state inertness check belongs on a DENIED *mutating* call
        // (`step_with_denied`), where "nothing changed" is an actual claim
        // about the code, not a tautology the compiler already enforces --
        // see `denied_channel_is_inert_including_swap_enable_bookkeeping`.
        let (eng, _tile, _host_mem) = fixture_s2mm_engine_mid_transfer();

        let demand = eng.peek_bank_demand(BankLayout::Compute);
        assert!(!demand.is_empty(), "an active channel must declare a bank");
        assert!(
            demand.iter().any(|(r, mask)| *r == Requester::S2mm(0) && *mask != 0),
            "the active S2MM channel must be the one declaring a bank: {demand:?}"
        );
    }

    #[test]
    fn peek_bank_demand_matches_committed_banks_on_grant() {
        // THE peek/commit-agreement assertion: whatever `peek_bank_demand`
        // predicts a granted channel will touch this cycle must be exactly
        // the banks `cycle_dma_banks` records once that cycle actually
        // commits. This is what the whole arbiter's correctness rests on --
        // if peek and commit ever disagree, the arbiter is granting/denying
        // based on a demand that doesn't match reality.
        let (mut eng, mut tile, mut host_mem) = fixture_s2mm_engine_mid_transfer();
        let demand = eng.peek_bank_demand(BankLayout::Compute);
        let (_, predicted_mask) = *demand
            .iter()
            .find(|(r, _)| *r == Requester::S2mm(0))
            .expect("channel 0 must have a demand");

        // `cycle_dma_banks` is an accumulator the array layer resets every
        // cycle (see `dma_ops.rs`); reset it here so this cycle's step
        // records only this cycle's touches.
        eng.cycle_dma_banks = 0;
        eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);

        assert_eq!(
            eng.cycle_dma_banks, predicted_mask,
            "the banks a granted channel actually touches must match what peek predicted"
        );
    }

    /// The tile DMA performs ONE 128-bit memory access per four 32-bit stream
    /// beats -- measured on Phoenix NPU1 at 16.0 B / 4.00 beats
    /// (docs/superpowers/findings/2026-07-14-dma-bank-access-width.md). A
    /// streaming MM2S moves one word per cycle, so over the BD it must claim a
    /// bank exactly once per 16-byte granule and never per stream beat: the
    /// other three beats of each granule come out of the egress staging FIFO
    /// with no memory access at all.
    ///
    /// WHICH cycles those accesses land on is the staging FIFO's business, not a
    /// fixed 1-in-4 rhythm: an empty FIFO refills as fast as it is allowed to, so
    /// a transfer front-loads its fetches until the staging is full and only then
    /// settles into one fetch per four drained words. That is what a real
    /// prefetching FIFO does, and it is what gives the DMA the slack to absorb a
    /// lost arbitration. The invariant silicon pins is the DENSITY -- 16 B per
    /// access -- which is what the conflict-area inversion measured.
    #[test]
    fn streaming_mm2s_claims_a_bank_once_per_granule() {
        let mut eng = DmaEngine::new_compute_tile(1, 2);
        let mut tile = Tile::compute(1, 2);
        let mut host_mem = HostMemory::new();

        // 32 bytes (8 words) at 0x400: two 16-byte granules.
        eng.configure_bd(0, BdConfig::simple_1d(0x400, 32)).unwrap();
        eng.start_channel(2, 0).unwrap(); // channel 2 is MM2S ch0 on a compute tile

        let mut guard = 0;
        while eng.channel_phase(2) != "Transferring" {
            eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
            guard += 1;
            assert!(guard < 50, "fixture did not reach Transferring in time");
        }

        let mut claimed = Vec::new();
        for _ in 0..8 {
            assert_eq!(eng.channel_phase(2), "Transferring", "fixture must still have words to move");
            let demand = eng.peek_bank_demand(BankLayout::Compute);
            claimed.push(demand.iter().any(|(r, m)| *r == Requester::Mm2s(0) && *m != 0));
            eng.cycle_dma_banks = 0;
            eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
            // Drain the egress FIFO like the array's stream router does every
            // cycle, so the MM2S is never stream-backpressured -- this test is
            // about the MEMORY side.
            while eng.pop_stream_out_for_channel(2).is_some() {}
            // Peek and commit must agree every cycle, including the cycles
            // that claim nothing.
            let predicted = demand
                .iter()
                .find(|(r, _)| *r == Requester::Mm2s(0))
                .map(|(_, m)| *m)
                .unwrap_or(0);
            assert_eq!(eng.cycle_dma_banks, predicted, "peek/commit disagreement");
        }

        assert_eq!(
            claimed.iter().filter(|c| **c).count(),
            32 / 16,
            "one bank access per 16-byte granule, not one per stream beat: {claimed:?}"
        );
        // The BD is 8 words and the staging is deeper than that, so both granules
        // are prefetched back to back and the remaining six beats stream out of
        // the FIFO claiming nothing.
        assert_eq!(claimed, vec![true, true, false, false, false, false, false, false]);
    }

    /// A chained BD's FIRST data cycle must be arbitrated like every other data
    /// cycle. The peek/commit agreement is absolute: whatever touches a bank in
    /// a cycle must have declared that bank in the same cycle's peek. A channel
    /// sitting in `AcquiringLock` declares nothing (`peek_bank_demand` only
    /// reports `Transferring` channels), so it must not move memory on the cycle
    /// its lock is granted either -- otherwise a core storing to the same bank
    /// that cycle silently wins a bank the DMA is also using, no conflict is
    /// emitted, and nobody is charged the cycle silicon charges.
    ///
    /// This is not an exotic path: `bd_switch_bubble_cycles` is 1 and an MM2S
    /// that finishes a BD with words still in its egress FIFO spends one bubble
    /// cycle in `DrainingEgress`, so the next BD's lock grant sees a bubble of 0
    /// -- the ordinary objectFIFO chain.
    #[test]
    fn chained_bd_touches_no_bank_on_its_lock_grant_cycle() {
        let mut eng = DmaEngine::new_compute_tile(1, 2);
        let mut tile = Tile::compute(1, 2);
        let mut host_mem = HostMemory::new();

        // BD0 -> BD1, BD1 acquires a lock that is already available, so the
        // acquire grants the first cycle it is submitted.
        eng.configure_bd(0, BdConfig::simple_1d(0x400, 32).with_next(1)).unwrap();
        eng.configure_bd(1, BdConfig::simple_1d(0x800, 32).with_acquire(5, 1)).unwrap();
        tile.locks[5].value = 1;
        eng.start_channel(2, 0).unwrap(); // channel 2 is MM2S ch0 on a compute tile

        let mut saw_lock_grant_boundary = false;
        for _ in 0..80 {
            let phase_before = eng.channel_phase(2);
            let peeked = eng
                .peek_bank_demand(BankLayout::Compute)
                .iter()
                .find(|(r, _)| *r == Requester::Mm2s(0))
                .map_or(0, |(_, m)| *m);

            eng.cycle_dma_banks = 0;
            eng.reset_cycle_drain_counters();
            eng.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
            tile.resolve_lock_requests(0);
            eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);

            assert_eq!(
                eng.cycle_dma_banks, peeked,
                "peek/commit disagreement stepping out of {phase_before}: the channel touched                  banks {:#06b} in a cycle whose arbitration only knew about {peeked:#06b}",
                eng.cycle_dma_banks
            );
            if phase_before == "AcquiringLock" && eng.channel_phase(2) != "AcquiringLock" {
                saw_lock_grant_boundary = true;
            }

            // Downstream drains one beat per cycle, exactly like the array's
            // stream router -- this is what leaves words in the egress FIFO at
            // BD completion and spends the BD-switch bubble in DrainingEgress.
            eng.pop_stream_out_for_channel(2);
        }

        assert!(saw_lock_grant_boundary, "the fixture must actually cross a chained-BD lock grant");
    }

    /// DMA_S2MM_n_MEMORY_BACKPRESSURE is a FIFO-FULL signal, not a lock signal
    /// and not an arbitration signal
    /// (docs/superpowers/findings/2026-07-14-dma-memory-pressure-event-semantics.md).
    /// Nothing in this test tells the channel it is lock-stalled: the BD wants a
    /// lock we never grant, so the channel simply never drains its ingress, and
    /// an upstream that offers one beat per cycle fills the FIFO. Backpressure
    /// must appear exactly when -- and only when -- the FIFO can no longer take
    /// the beat being offered. The number of beats swallowed first is the FIFO's
    /// depth, which nothing here hardcodes.
    #[test]
    fn s2mm_backpressure_emerges_only_when_the_ingress_is_full() {
        let mut eng = DmaEngine::new_compute_tile(1, 2);
        let mut tile = Tile::compute(1, 2);
        let mut host_mem = HostMemory::new();

        // Lock 5 is left at 0, so this acquire (== 1) never grants: the channel
        // parks in AcquiringLock and never writes a word to memory.
        eng.configure_bd(0, BdConfig::simple_1d(0x400, 4096).with_acquire(5, 1))
            .unwrap();
        eng.start_channel(0, 0).unwrap();

        let cap = eng.input_fifo_capacity();
        let mut accepted = 0usize;
        let mut first_backpressure_at = None;

        // Drive the two phases the array driver drives, in the array driver's
        // order: the DMA steps (Phase 3), then the upstream offers a beat
        // (Phase 4).
        for cycle in 0..(cap + 8) {
            eng.reset_cycle_drain_counters();
            eng.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
            tile.resolve_lock_requests(0);
            eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);

            if eng.can_accept_stream_in_for_routing(0) {
                eng.push_stream_in(StreamData { data: 0xC0DE_0000 | cycle as u32, tlast: false, channel: 0 });
                accepted += 1;
            } else {
                eng.note_ingress_offer_refused(0);
            }

            if eng.s2mm_memory_backpressure(0) && first_backpressure_at.is_none() {
                first_backpressure_at = Some(cycle);
            }
        }

        assert_eq!(accepted, cap, "the ingress must swallow exactly its own depth in beats");
        assert_eq!(
            first_backpressure_at,
            Some(cap),
            "backpressure must assert on the first beat the FIFO cannot take, and not one cycle sooner"
        );

        // Free the buffer. The channel drains the ingress, a slot opens, the
        // next offered beat is accepted -- and backpressure clears on its own.
        // Free the buffer. Once the channel resumes and drains a word, a slot
        // opens and the next offered beat is accepted -- and backpressure clears
        // on its own, on exactly that cycle. (How long the channel takes to
        // resume is its own start latency -- this BD is cold, so it pays the full
        // lock-acquire + memory-latency pipeline. A steady-state chained BD
        // resumes through the 1-cycle BD-switch bubble, which is where silicon's
        // "deasserts 1 cycle after the lock" comes from. That latency is a
        // separate, already-modelled thing; what the FIFO owns is that
        // backpressure ends the moment the FIFO has room, and not a cycle either
        // side of it.)
        tile.locks[5].set(1);
        let mut cleared_at = None;
        let mut first_accept_at = None;
        for cycle in 0..40 {
            eng.reset_cycle_drain_counters();
            eng.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
            tile.resolve_lock_requests(0);
            eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
            if eng.can_accept_stream_in_for_routing(0) {
                eng.push_stream_in(StreamData { data: 0xFEED_0000, tlast: false, channel: 0 });
                if first_accept_at.is_none() {
                    first_accept_at = Some(cycle);
                }
            } else {
                eng.note_ingress_offer_refused(0);
            }
            if !eng.s2mm_memory_backpressure(0) && cleared_at.is_none() {
                cleared_at = Some(cycle);
            }
        }
        assert!(first_accept_at.is_some(), "the channel must resume once the buffer is free");
        assert_eq!(
            cleared_at, first_accept_at,
            "backpressure must clear on exactly the cycle the FIFO can take a beat again"
        );
    }

    /// The MM2S egress staging FIFO is what lets the tile DMA lose most of its
    /// bank arbitrations at ZERO throughput cost -- measured on Phoenix NPU1 as
    /// `T_collide / T_apart = 1.0000` with the MM2S losing 112 arbitrations and
    /// paying 5 cycles, and `MM2S_MEMORY_STARVATION` reading 0
    /// (docs/superpowers/findings/2026-07-14-dma-bank-access-width.md).
    ///
    /// Deny the channel every time it presents a bank demand; it re-presents and
    /// wins the retry the next cycle (a round-robin loss, which is exactly what
    /// silicon's DMA suffers). Once the staging has anything in it at all, the
    /// transfer must take the SAME number of cycles as the undenied baseline and
    /// must never report memory starvation, because the staging covers the
    /// delayed refill.
    ///
    /// The one denial that CAN cost a cycle is the very first fetch of a
    /// transfer, when the staging is genuinely empty and there is nothing to
    /// cover with -- and that is not a modelling artifact, it is the residual
    /// silicon measures: `collide` lost 112 arbitrations across 16 transfers and
    /// paid 5 cycles for them, i.e. a handful of cold fetches, not a per-denial
    /// cost.
    #[test]
    fn mm2s_egress_staging_absorbs_bank_denials_without_starving_or_slowing() {
        /// Run a 64-word MM2S transfer, draining the egress every cycle like the
        /// array's stream router does. `deny` picks which presented demands lose
        /// their arbitration, given (demand_index, currently_staged). Returns
        /// (cycles to finish, starvation cycles, granted bank accesses).
        fn run(deny: impl Fn(usize, usize) -> bool) -> (usize, usize, usize) {
            let mut eng = DmaEngine::new_compute_tile(1, 2);
            let mut tile = Tile::compute(1, 2);
            let mut host_mem = HostMemory::new();
            eng.configure_bd(0, BdConfig::simple_1d(0x400, 256)).unwrap();
            eng.start_channel(2, 0).unwrap(); // channel 2 = MM2S ch0

            let (mut cycles, mut starved, mut granted, mut presented) = (0usize, 0usize, 0usize, 0usize);
            let mut denied_last = false;
            while eng.channel_phase(2) != "Idle" && cycles < 500 {
                eng.reset_cycle_drain_counters();
                let demand = eng.peek_bank_demand(BankLayout::Compute);
                let mask = demand
                    .iter()
                    .find(|(r, _)| *r == Requester::Mm2s(0))
                    .map(|(_, m)| *m)
                    .unwrap_or(0);
                let staged = eng.staged_words(2);
                // A channel that lost a bank must re-present the IDENTICAL demand
                // next cycle -- even though its stream side kept draining, which
                // moved its address generator. That is the retry contract, and it
                // holds only because draining does not move the fetch cursor.
                if denied_last {
                    assert_ne!(mask, 0, "a denied channel must re-present its demand");
                }
                let lose = mask != 0 && !denied_last && deny(presented, staged);
                if mask != 0 {
                    presented += 1;
                    if !lose {
                        granted += 1;
                    }
                }
                denied_last = lose;
                let denied: &[Requester] = if lose { &[Requester::Mm2s(0)] } else { &[] };
                eng.step_with_denied(denied, &mut tile, &mut NeighborTiles::empty(), &mut host_mem);
                while eng.pop_stream_out_for_channel(2).is_some() {}
                if eng.mm2s_memory_starvation(0) {
                    starved += 1;
                }
                cycles += 1;
            }
            assert!(cycles < 500, "transfer did not finish");
            (cycles, starved, granted)
        }

        let (base_cycles, base_starved, base_granted) = run(|_, _| false);
        // Lose every arbitration the channel presents once its staging is warm.
        let (denied_cycles, denied_starved, denied_granted) = run(|_, staged| staged > 0);

        assert_eq!(base_starved, 0, "an undenied MM2S must never starve");
        assert_eq!(base_granted, 256 / 16, "one bank access per 16-byte granule over the whole BD");
        assert_eq!(denied_granted, base_granted, "the same granules get fetched, just later");
        assert_eq!(
            denied_starved, 0,
            "a bank denial must be absorbed by the egress staging, not reported as memory starvation"
        );
        assert_eq!(
            denied_cycles, base_cycles,
            "losing every bank arbitration must cost ZERO cycles (HW: T_collide/T_apart = 1.0000)"
        );

        // The cold fetch is the exception, and it is silicon's exception too: with
        // an empty staging there is nothing to cover the loss, so the port idles
        // for exactly the cycle the retry takes.
        let (cold_cycles, cold_starved, _) = run(|i, _| i == 0);
        assert_eq!(cold_starved, 1, "a denied FIRST fetch has no staging to hide behind");
        assert_eq!(cold_cycles, base_cycles + 1, "and costs exactly the one retry cycle");
    }

    #[test]
    fn dma_peek_reports_no_demand_for_idle_channels() {
        // Channels 1-3 have no BD configured and are Idle -- Idle never
        // touches a bank this cycle, so peek must never invent a demand for
        // them (this is also why `denied` can never legally name them).
        let (eng, _tile, _host_mem) = fixture_s2mm_engine_mid_transfer();
        let demand = eng.peek_bank_demand(BankLayout::Compute);
        assert!(!demand.iter().any(|(r, _)| matches!(r, Requester::S2mm(1) | Requester::Mm2s(_))));
    }

    #[test]
    fn denied_channel_does_not_transfer_and_retries_unchanged() {
        let (mut eng, mut tile, mut host_mem) = fixture_s2mm_engine_mid_transfer();
        let before_bytes = eng.channel_stats(0).unwrap().bytes_transferred;
        let demand_before = eng.peek_bank_demand(BankLayout::Compute);
        assert!(!demand_before.is_empty());

        // Lose arbitration for two cycles running -- the retry contract
        // demands the identical demand and zero progress every time.
        for cycle in 0..2 {
            eng.step_with_denied(
                &[Requester::S2mm(0)],
                &mut tile,
                &mut NeighborTiles::empty(),
                &mut host_mem,
            );
            assert_eq!(
                eng.channel_stats(0).unwrap().bytes_transferred,
                before_bytes,
                "a denied channel must not move data this cycle ({cycle})"
            );
            let demand_now = eng.peek_bank_demand(BankLayout::Compute);
            assert_eq!(
                demand_now, demand_before,
                "a losing channel must re-present the identical demand ({cycle})"
            );
        }

        // Next cycle, ungated, it proceeds normally and performs exactly the
        // transfer it would have performed had it won on the first cycle:
        // BD 0's remaining 16 bytes (4 words), all already staged in
        // stream_in by the fixture.
        eng.step_with_denied(&[], &mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        let after_bytes = eng.channel_stats(0).unwrap().bytes_transferred;
        assert!(after_bytes > before_bytes, "it must retry and progress once granted");
        assert_eq!(after_bytes, 32, "granted retry must perform exactly the withheld transfer");

        // Byte-count equality alone doesn't rule out a corrupted or
        // mis-addressed write -- diff the actual destination memory against
        // a fresh undenied baseline run from the identical starting fixture
        // state (the fixture is deterministic: same BD, same staged stream
        // words), so "granted retry == the undenied transfer" is proven at
        // the data level, not just the byte-count level.
        let (mut baseline_eng, mut baseline_tile, mut baseline_host) = fixture_s2mm_engine_mid_transfer();
        baseline_eng.step(&mut baseline_tile, &mut NeighborTiles::empty(), &mut baseline_host);
        assert_eq!(
            tile.data_memory(),
            baseline_tile.data_memory(),
            "a denied-then-granted transfer must write identical destination memory to the undenied baseline"
        );
    }

    #[test]
    fn denied_channel_is_inert_including_swap_enable_bookkeeping() {
        // FIX 1 regression guard. `service_pending_releases` runs for EVERY
        // channel, EVERY cycle (from `step_impl`, not gated on denial before
        // this fix), and calls `schedule_swap_enable_releases`, which reads
        // `prev_starving` and, when true, watches the channel's acquire lock
        // for a "swap enable" increment -- mutating `swap_free_watch` and
        // potentially retiring a `pending_releases` entry (firing a deferred
        // LockRelease TRACE EVENT). `prev_starving` is only cleared inside
        // `step_transferring_cycle`, which a denied channel never runs, so a
        // channel that genuinely starved on an earlier cycle still reads as
        // stream-stalled here even while held -- and a lock-value increment
        // during the held cycle (some OTHER actor freeing a buffer, entirely
        // independent of this channel's own bank arbitration) used to be
        // enough to mutate engine state and emit a trace event on a cycle
        // the channel never actually ran.
        //
        // `fixture_s2mm_engine_mid_transfer` never starves (its channel
        // always has stream data staged) and configures no lock, so it can't
        // exercise this path -- this test builds the minimal scenario that
        // does: a lock-gated channel, driven to a genuine stream stall, with
        // a pending deferred release and an external lock-value bump, then
        // denied for one cycle.
        let mut eng = DmaEngine::new_compute_tile(1, 2);
        let mut tile = Tile::compute(1, 2);
        let mut host_mem = HostMemory::new();

        // BD requires lock 5 == 1 (acq_eq) to start; pre-satisfy it so the
        // channel is granted on its first AcquiringLock step -- the acquire
        // dance itself isn't what's under test.
        tile.locks[5].set(1);
        eng.configure_bd(0, BdConfig::simple_1d(0x400, 32).with_acquire(5, 1)).unwrap();
        eng.start_channel(0, 0).unwrap();

        let mut guard = 0;
        loop {
            eng.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
            tile.resolve_lock_requests(0);
            eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
            if eng.channel_phase(0) == "Transferring" {
                break;
            }
            guard += 1;
            assert!(guard < 50, "fixture did not reach Transferring in time");
        }

        // Genuinely starve: no stream data staged, so this step actually
        // stalls and sets prev_starving=true from a real starvation onset,
        // not stale leftover state.
        eng.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        assert_eq!(
            eng.channel_stats(0).unwrap().bytes_transferred,
            0,
            "must still be starved, no data available"
        );

        // A deferred LockRelease trace event from an earlier (unrelated) BD
        // completion, awaiting the next buffer swap -- exactly the state
        // `service_pending_releases` retires on a swap-enable increment.
        eng.channels[0].pending_releases.push(PendingRelease {
            lock_id: 5,
            bd_index: 0,
            ready_cycle: 0,
            trace_at: None,
        });

        // The external swap: some other actor frees a buffer by incrementing
        // the SAME acquire lock this starving channel watches -- a real,
        // independent hardware event, nothing to do with bank arbitration.
        tile.locks[5].value += 1;

        let before = format!("{:?}", eng);

        // Denied this cycle: the channel must not observe or react to the
        // swap at all -- not the transfer (covered by the retry-contract
        // test above), and not this engine-wide bookkeeping either.
        eng.step_with_denied(&[Requester::S2mm(0)], &mut tile, &mut NeighborTiles::empty(), &mut host_mem);

        assert_eq!(
            format!("{:?}", eng),
            before,
            "a denied channel must be byte-for-byte inert, including swap-enable-release bookkeeping (FIX 1)"
        );
    }

    #[test]
    fn undenied_step_with_denied_matches_plain_step() {
        // step_with_denied(&[], ...) must be indistinguishable from step():
        // an empty denial list holds nothing back.
        let (mut eng_a, mut tile_a, mut host_a) = fixture_s2mm_engine_mid_transfer();
        let (mut eng_b, mut tile_b, mut host_b) = fixture_s2mm_engine_mid_transfer();

        eng_a.step(&mut tile_a, &mut NeighborTiles::empty(), &mut host_a);
        eng_b.step_with_denied(&[], &mut tile_b, &mut NeighborTiles::empty(), &mut host_b);

        assert_eq!(
            eng_a.channel_stats(0).unwrap().bytes_transferred,
            eng_b.channel_stats(0).unwrap().bytes_transferred,
        );
        assert_eq!(eng_a.channel_phase(0), eng_b.channel_phase(0));
    }
}
