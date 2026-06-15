//! FSM execution: the `step()` method and all helper methods it calls.

use super::*;
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
fn dma_access_origin(target: MemTileTarget) -> AccessOrigin {
    match target {
        MemTileTarget::Own => AccessOrigin::Dma,
        MemTileTarget::West => AccessOrigin::Neighbour(MemoryQuadrant::East),
        MemTileTarget::East => AccessOrigin::Neighbour(MemoryQuadrant::West),
    }
}

impl DmaEngine {
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
        let mut any_active = false;
        let mut any_waiting = false;

        for ch_idx in 0..self.channels.len() {
            let phase_before = self.channels[ch_idx].fsm.phase_name();

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
                    // Active channel -- run one FSM cycle
                    self.step_channel_fsm(ch_idx, tile, neighbors, host_memory);
                    if matches!(self.channels[ch_idx].fsm, ChannelFsm::AcquiringLock { acquired: false, .. })
                    {
                        any_waiting = true;
                    } else {
                        any_active = true;
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
        // serviced the same cycle once past its ready_cycle floor.
        self.service_pending_releases(tile, neighbors);

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
    fn service_pending_releases(&mut self, tile: &mut Tile, neighbors: &mut NeighborTiles<'_>) {
        for ch_idx in 0..self.channels.len() {
            // End-of-stream release tail: while a producer is stream-stalled, a
            // consumer-free (its acquire lock incrementing) is the SWAP-enable
            // that fires the next deferred full-release -- HW emits LOCK_SEL*_REL
            // there even though the stalled producer never re-acquires the slot.
            self.schedule_swap_enable_releases(ch_idx, tile, neighbors);

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
            // Per-task overhead fires on every task.
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
    fn step_channel_fsm(
        &mut self,
        ch_idx: usize,
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
                            self.enter_chained_transfer(transfer)
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
                        // Acquire granted -- lock stall level deasserts.
                        if self.channels[ch_idx].prev_lock_stalled {
                            self.trace(EventType::DmaStalledLock { channel: ch_idx as u8, active: false });
                            self.channels[ch_idx].prev_lock_stalled = false;
                        }
                        // This acquire grant RETIRES the BD slot it re-acquires:
                        // schedule the deferred TRACE event for the prior fill
                        // that used this same slot (its functional release
                        // already fired inline at completion). HW's LOCK_SEL*_REL
                        // trace fires at this BD-ring recycle, not at completion
                        // -- for a depth-2 fifo that is ~2 fill-periods later.
                        let reuse_bd = transfer.bd_index;
                        let now = self.current_cycle;
                        if let Some(pr) = self.channels[ch_idx]
                            .pending_releases
                            .iter_mut()
                            .find(|p| p.bd_index == reuse_bd && p.trace_at.is_none())
                        {
                            pr.trace_at = Some(pr.ready_cycle.max(now));
                        }
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
                                self.enter_transfer_after_lock_grant(
                                    ch_idx,
                                    transfer,
                                    tile,
                                    neighbors,
                                    host_memory,
                                )
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
                self.step_transferring_cycle(ch_idx, transfer, tile, neighbors, host_memory)
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
        mut transfer: Box<Transfer>,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> ChannelFsm {
        let result = self.do_transfer_cycle(ch_idx, &mut transfer, tile, neighbors, host_memory);

        match result {
            TransferCycleResult::Continue => {
                transfer.tick();
                // Stream data resumed -- starvation level deasserts.
                if self.channels[ch_idx].prev_starving {
                    self.trace(EventType::DmaStreamStarvation { channel: ch_idx as u8, active: false });
                    self.channels[ch_idx].prev_starving = false;
                }
                if transfer.remaining_bytes() == 0 {
                    self.begin_completion(ch_idx, transfer, tile, neighbors)
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
            TransferCycleResult::FotFinish => self.begin_completion(ch_idx, transfer, tile, neighbors),
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
        self.enter_chained_transfer(transfer)
    }

    /// Route a chained BD's transfer into its data phase from a state that has
    /// no inherent idle cycle of its own (the no-lock prefetch path, or a
    /// post-cooldown acquired lock). Inserts the minimum BD-switch bubble
    /// (`bd_switch_bubble_cycles`) so the stream port deasserts for ~1 cycle at
    /// the boundary, unless a longer host-pipeline latency already provides the
    /// gap (then the bubble is absorbed). `bubble == 0` restores the old
    /// back-to-back behavior.
    fn enter_chained_transfer(&self, transfer: Box<Transfer>) -> ChannelFsm {
        let host_lat = self.timing_config.host_memory_latency_cycles;
        if host_lat > 0 && transfer.involves_host_memory() {
            ChannelFsm::HostPipelineLatency { cycles_remaining: host_lat, transfer }
        } else if self.timing_config.bd_switch_bubble_cycles > 0 {
            ChannelFsm::BdSwitchBubble {
                cycles_remaining: self.timing_config.bd_switch_bubble_cycles,
                transfer,
            }
        } else {
            ChannelFsm::Transferring { transfer }
        }
    }

    /// Route a chained locked BD into its data phase the cycle its acquire is
    /// granted. The grant cycle is itself a port-idle cycle (no beat moves), so
    /// it already counts as the first BD-switch bubble cycle:
    /// - `bubble == 0`: inline the first beat now (the historical no-bubble
    ///   prefetch -- grant and first data in one step).
    /// - `bubble == 1`: stay idle this cycle and resume next cycle (the grant
    ///   cycle *is* the one-cycle bubble); the default, matching HW `off1`.
    /// - `bubble >= 2`: idle for the remaining `bubble - 1` cycles.
    fn enter_transfer_after_lock_grant(
        &mut self,
        ch_idx: usize,
        transfer: Box<Transfer>,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> ChannelFsm {
        match self.timing_config.bd_switch_bubble_cycles {
            0 => self.step_transferring_cycle(ch_idx, transfer, tile, neighbors, host_memory),
            1 => ChannelFsm::Transferring { transfer },
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
                self.release_lock_value(lock_id, release_value, tile, neighbors);
                self.channels[ch_idx].pending_releases.push(PendingRelease {
                    lock_id,
                    bd_index: completion.bd_index,
                    ready_cycle: self.current_cycle + latency,
                    trace_at: None,
                });
            } else {
                // No trace deferral: functional release + trace event inline.
                self.apply_lock_release_direct(lock_id, release_value, tile, neighbors);
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

    /// Perform one cycle of data transfer for a channel in the Transferring state.
    ///
    /// Extracts transfer parameters, calls do_transfer, and advances the transfer.
    /// The Transfer is borrowed from the FSM via the caller (not from self.channels).
    fn do_transfer_cycle(
        &mut self,
        ch_idx: usize,
        transfer: &mut Transfer,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> TransferCycleResult {
        // Shim DMA transfers touching host memory stream at a narrower rate
        // than tile-local DMAs.  HW measurement (2026-05-25): 1 word/cyc on
        // Phoenix shim AXI master; tile-local stays at 4 words/cyc (data
        // memory bus width).
        let words_per_cycle = if self.tile_kind.is_shim() && transfer.involves_host_memory() {
            self.timing_config.shim_words_per_cycle as usize
        } else {
            self.timing_config.words_per_cycle as usize
        };

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

            // DDR burst gate: a shim MM2S host-memory read delivers in bursts
            // (AXI/DDR cadence), not a uniform stream. Poll every Transferring
            // cycle so the gap countdown advances, and cap this cycle's delivery
            // to the burst allowance. During an inter-burst gap the cap is 0, so
            // the channel makes no progress this cycle (a DDR wait, NOT a stream
            // starvation -- we return Continue below, never Stalled, so no
            // starvation event fires on the producer). The downstream S2MM
            // consumer then drains and starves on its own, the HW mechanism.
            // Disabled by default (BurstParams::DISABLED -> words_allowed returns
            // u16::MAX), so this is a no-op unless opted in.
            let burst_gated = self.tile_kind.is_shim()
                && transfer.involves_host_memory()
                && transfer.direction == TransferDirection::MM2S
                && self.timing_config.ddr_burst.enabled();
            if burst_gated {
                let allowed =
                    self.channels[ch_idx].ddr_burst_gate.words_allowed(self.timing_config.ddr_burst) as usize;
                words_this_cycle = words_this_cycle.min(allowed);
            }

            let source = transfer.source;
            let dest = transfer.dest;
            let channel = transfer.channel;
            let tlast_suppress = transfer.effective_tlast_suppress();
            let mut fot_finished = false;

            for w in 0..words_this_cycle {
                let addr = transfer.current_address();
                let is_last = transfer.remaining_bytes() <= 4;

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
                self.channels[ch_idx].stats.bytes_transferred += 4;
                // Draw down the burst budget per word actually delivered, so an
                // early backpressure stall (mid-chunk) leaves the gap accounting
                // consistent with real delivery.
                if burst_gated {
                    self.channels[ch_idx].ddr_burst_gate.consume(1);
                }

                if result.fot_finish {
                    fot_finished = true;
                    break;
                }
            }

            if fot_finished {
                TransferCycleResult::FotFinish
            } else {
                TransferCycleResult::Continue
            }
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
            return Some((MemTileTarget::Own, own, (addr as usize) % mem_size));
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
            return Some((MemTileTarget::Own, own, (addr as usize) % mem_size));
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

        // Record bank access for conflict detection (offset is local to the target tile)
        self.cycle_dma_banks |=
            crate::device::banking::banks_for_access(offset as u32, bytes, self.num_banks);

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
            // Bank is touched only when we actually consume from the stream.
            self.cycle_dma_banks |=
                crate::device::banking::banks_for_access(offset as u32, bytes, self.num_banks);
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
        // when a core happens to load from the same bank.
        self.cycle_dma_banks |=
            crate::device::banking::banks_for_access(offset as u32, bytes, self.num_banks);

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
