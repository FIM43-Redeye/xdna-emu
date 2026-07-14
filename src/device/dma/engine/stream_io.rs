//! Stream interface methods for TileArray integration.

use super::*;

/// Bytes per AIE2 stream beat (32-bit AXI4-Stream word). Used to convert a BD's
/// byte length into the accepted-word count the recv accept cursor tracks.
const BYTES_PER_STREAM_WORD: u32 = 4;

impl DmaEngine {
    /// Maximum per-channel `stream_in` depth before that S2MM channel
    /// backpressures its upstream stream-switch master port.
    ///
    /// Mirrors `output_fifo_capacity()` on the MM2S side. The S2MM consumer
    /// reads from the tile's local master stream-switch port, whose FIFO AM020
    /// ch2 specifies as 2-deep for AIE2 ("Local master ports have one register
    /// slice with 1-cycle latency and a 2-deep FIFO").
    ///
    /// On **compute and mem tiles** the DMA has a deeper per-channel S2MM
    /// ingress (device-model `s2mmChannel.buffer_depth` = 12 plus the 16-deep
    /// `StreamSwitch.fifo_depth`, identical on both tile types) downstream of
    /// that master port, letting a recv port stage a full objectfifo BD ahead of
    /// a lock-stalled buffer write. HW co-captures (#140, 2026-06-27) confirm it
    /// on both: memtile recv decodes to a clean `[16,16,16,16]` and compute recv
    /// to `[8,8,8,8,8,8,8,8]`, where the pre-fix 2-deep model split every reused
    /// BD (`[16,16,2,14,16]` / `[8,8,2,6,...]`). EMU counts the master-port beat
    /// at the DMA-accept point, so this single capacity stands in for the
    /// combined ingress (= `DMA_S2MM_INGRESS_FIFO_DEPTH`, 16). The **shim** s2mm
    /// differs (`buffer_depth` 64) and its recv is governed by the separately
    /// calibrated DDR cold-start model, so it keeps the shallow master-port depth.
    ///
    /// History: this was a hardcoded 256-word buffer "to absorb scheduler
    /// jitter without blocking." That depth silently absorbed ~256/buffer_words
    /// extra fast iterations during double-buffer warmup, defeating the
    /// otherwise HW-faithful master-port backpressure and producing a warmup
    /// transient much longer than silicon (root-caused 2026-06-13 via the
    /// dma_passthrough buffer-size sweep). The depth-16 value is far shallower
    /// (~1 BD of warmup slack). Phase 4's `can_accept_stream_in_for_routing`
    /// enforces the registered-FIFO ordering invariant: it checks
    /// `current + drained_this_cycle < capacity` rather than `current < capacity`,
    /// so a slot freed by Phase 3 drain is not double-counted as available in
    /// Phase 4 of the same cycle (#140 phase-ordering fix, 2026-06-27).
    pub fn input_fifo_capacity(&self) -> usize {
        use xdna_archspec::aie2::timing;
        if self.tile_kind.is_mem() || self.tile_kind.is_compute() {
            timing::DMA_S2MM_INGRESS_FIFO_DEPTH as usize
        } else {
            timing::STREAM_LOCAL_MASTER_FIFO_DEPTH as usize
        }
    }

    /// Map a combined channel index (S2MM_count + MM2S_offset) to the
    /// per-channel `stream_out` slot.  Returns `usize::MAX` if the channel
    /// isn't an MM2S channel on this tile -- callers that hit that case have
    /// a programming error and the bounds-checked index access below will
    /// panic, which is what we want (impossible-on-HW invariant).
    #[inline]
    fn stream_out_idx(&self, channel: u8) -> usize {
        (channel as usize).saturating_sub(self.s2mm_count)
    }

    /// Push a word to the MM2S channel's stream output buffer.  The caller is
    /// expected to gate on `can_push_stream_out_for_channel` (or the
    /// `output_fifo_capacity` check); this assumes capacity has been verified.
    pub(super) fn push_stream_out(&mut self, data: StreamData) {
        let idx = self.stream_out_idx(data.channel);
        self.stream_out[idx].push_back(data);
    }

    /// Pop a word from any non-empty MM2S channel's stream output buffer.
    ///
    /// Used by callers that want to drain the engine without caring about
    /// which channel produced the data (tests, `execute_1d_transfer`).
    /// `route_dma_to_tile_switches` uses `pop_stream_out_for_channel` instead
    /// so each channel's words land in the correct slave port and per-channel
    /// backpressure is preserved.
    pub fn pop_stream_out(&mut self) -> Option<StreamData> {
        for q in &mut self.stream_out {
            if let Some(d) = q.pop_front() {
                return Some(d);
            }
        }
        None
    }

    /// Pop the next word for a specific MM2S channel.
    pub fn pop_stream_out_for_channel(&mut self, channel: u8) -> Option<StreamData> {
        let idx = self.stream_out_idx(channel);
        self.stream_out.get_mut(idx)?.pop_front()
    }

    /// Peek at the next word in any non-empty channel without removing it.
    /// Returns the first channel's front (round-robin would skew throughput
    /// metrics; this is for inspection only).
    pub fn peek_stream_out(&self) -> Option<&StreamData> {
        self.stream_out.iter().find_map(|q| q.front())
    }

    /// Peek at the next word for a specific MM2S channel.
    pub fn peek_stream_out_for_channel(&self, channel: u8) -> Option<&StreamData> {
        let idx = self.stream_out_idx(channel);
        self.stream_out.get(idx)?.front()
    }

    /// Maximum per-channel `stream_out` depth before that channel
    /// backpressures.
    ///
    /// Models the slave-port FIFO that the DMA's output bridges into on
    /// real silicon; AM020 ch2 specifies "Local slave ports are 2-cycle
    /// latency and a 4-deep FIFO" for AIE2 / Phoenix.  Each MM2S channel
    /// pushes into its own slave port FIFO with independent credit-based
    /// flow control, so the capacity is enforced per-channel -- one
    /// channel filling its FIFO must not stall any other channel.
    pub fn output_fifo_capacity(&self) -> usize {
        xdna_archspec::aie2::timing::STREAM_LOCAL_SLAVE_FIFO_DEPTH as usize
    }

    /// Whether the MM2S channel's `stream_out` has room for at least one
    /// more word (i.e. `len < output_fifo_capacity`).
    pub fn can_push_stream_out_for_channel(&self, channel: u8) -> bool {
        let idx = self.stream_out_idx(channel);
        let cap = self.output_fifo_capacity();
        self.stream_out.get(idx).map_or(false, |q| q.len() < cap)
    }

    /// Prepend retained words back to the MM2S channel's stream_out buffer.
    ///
    /// Used by `route_dma_to_tile_switches` to put back words that couldn't
    /// be delivered this cycle (target slave full).  Retained words go to
    /// the front of THAT channel's queue so they're retried first next
    /// cycle, preserving per-channel ordering without affecting any other
    /// channel.
    pub fn prepend_stream_out_for_channel(
        &mut self,
        channel: u8,
        mut retained: std::collections::VecDeque<StreamData>,
    ) {
        let idx = self.stream_out_idx(channel);
        if let Some(q) = self.stream_out.get_mut(idx) {
            retained.append(q);
            *q = retained;
        }
    }

    /// Push a word to the per-channel stream input buffer (for S2MM to consume).
    ///
    /// Each S2MM channel has its own FIFO with independent capacity, matching
    /// real hardware where each channel connects to a dedicated stream switch
    /// master port. Returns true if successful, false if channel buffer is full.
    pub fn push_stream_in(&mut self, data: StreamData) -> bool {
        let ch = data.channel as usize;
        if ch >= self.stream_in.len() {
            let msg = format!(
                "DMA({},{}) push_stream_in: channel {} out of range (have {} S2MM channels)",
                self.col,
                self.row,
                ch,
                self.stream_in.len()
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            return false;
        }
        if self.stream_in[ch].len() < self.input_fifo_capacity() {
            // This word is accepted only because the gate allowed it -- which, if
            // a next-BD drain block was armed, means the prior BD has drained and
            // this is the first word of the next BD. Clear the block so the rest
            // of this BD accepts normally (the one-BD-lookahead bound, #140).
            if let Some(c) = self.channels.get_mut(ch) {
                c.accept_awaiting_drain = false;
            }
            self.stream_in[ch].push_back(data);
            self.advance_accept_cursor(ch);
            true
        } else {
            let msg = format!(
                "DMA({},{}) stream_in buffer full ({}), dropping ch{} data: 0x{:08X} -- \
                 backpressure violation",
                self.col,
                self.row,
                self.input_fifo_capacity(),
                data.channel,
                data.data,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            false
        }
    }

    /// Advance the recv-side accept cursor after one word is accepted into the
    /// S2MM input FIFO (#140). HW gates the stream port (TREADY) per-BD: it
    /// accepts exactly the current BD's length, then deasserts for the
    /// reconfiguration before the next BD. We mirror that on the accept side --
    /// independent of the lagging memory-write completion -- so the recv port
    /// traces `on16 off1` instead of front-loading the double-buffer through the
    /// boundary. When the current BD's words are exhausted, arm the deassert
    /// (`bd_switch_accept_block`) and walk to the next BD in the chain.
    /// No-op until an S2MM task initializes the cursor (`accept_bd == None`).
    fn advance_accept_cursor(&mut self, ch: usize) {
        let (abd, rem) = match self.channels.get(ch) {
            Some(c) if c.accept_bd.is_some() => (c.accept_bd, c.accept_words_remaining),
            _ => return,
        };
        let rem = rem.saturating_sub(1);
        if rem > 0 {
            self.channels[ch].accept_words_remaining = rem;
            return;
        }
        // BD fully accepted: arm the per-BD deassert and walk to the next BD.
        let next = abd.and_then(|b| self.bd_configs.get(b as usize)).and_then(|c| c.next_bd);
        let next_words = next
            .and_then(|n| self.bd_configs.get(n as usize))
            .map(|c| c.length.div_ceil(BYTES_PER_STREAM_WORD))
            .unwrap_or(0);
        let bubble = self.timing_config.bd_switch_bubble_cycles;
        let c = &mut self.channels[ch];
        c.bd_switch_accept_block = bubble;
        // One-BD-lookahead bound (#140 send-cadence): the next BD's words are not
        // accepted until this just-completed BD has DRAINED from the ingress. The
        // fixed bubble above is the intra-BD reconfiguration gap; this is the
        // cross-BD drain gate that keeps the producer at most one BD ahead of a
        // lock-stalled consumer (HW: ~8-word send bursts, never two BDs). Only
        // when there IS a successor -- a single/final BD stages fully (cold
        // absorb), nothing downstream to gate.
        c.accept_awaiting_drain = next.is_some();
        c.accept_bd = next;
        c.accept_words_remaining = next_words;
    }

    /// Initialize the recv-side accept cursor for an S2MM channel starting a task
    /// at `start_bd` (#140). Sets the cursor to the first BD so the accept side
    /// can gate TREADY per-BD from the outset; the first BD itself takes no
    /// leading deassert (the gap is between BDs).
    pub(crate) fn init_accept_cursor(&mut self, ch: usize, start_bd: u8) {
        let words = self
            .bd_configs
            .get(start_bd as usize)
            .map(|c| c.length.div_ceil(BYTES_PER_STREAM_WORD))
            .unwrap_or(0);
        if let Some(c) = self.channels.get_mut(ch) {
            c.accept_bd = Some(start_bd);
            c.accept_words_remaining = words;
            c.bd_switch_accept_block = 0;
            c.accept_awaiting_drain = false;
        }
    }

    /// Consume one cycle of the recv-side BD-switch deassert, if armed. Called
    /// from the stream-routing pass when a channel refuses the stream so the
    /// 1-cycle reconfiguration gap elapses. No-op when not armed. #140.
    pub(crate) fn consume_bd_switch_accept_block(&mut self, ch: usize) {
        if let Some(c) = self.channels.get_mut(ch) {
            c.bd_switch_accept_block = c.bd_switch_accept_block.saturating_sub(1);
        }
    }

    /// Check if any MM2S channel's stream output buffer has data.
    pub fn has_stream_out(&self) -> bool {
        self.stream_out.iter().any(|q| !q.is_empty())
    }

    /// Check if a specific MM2S channel has stream output data.
    pub fn has_stream_out_for_channel(&self, channel: u8) -> bool {
        let idx = self.stream_out_idx(channel);
        self.stream_out.get(idx).map_or(false, |q| !q.is_empty())
    }

    /// Check if any S2MM channel's stream input buffer has space.
    ///
    /// Returns true if at least one channel can accept data. Callers that
    /// know the target channel should use `can_accept_stream_in_for_channel`
    /// for precise per-channel backpressure.
    pub fn can_accept_stream_in(&self) -> bool {
        self.stream_in.iter().any(|q| q.len() < self.input_fifo_capacity())
    }

    /// Check if a specific S2MM channel's stream input buffer has space.
    ///
    /// Also deasserts during a BD-switch reconfiguration: while
    /// `bd_switch_accept_block` is set (armed at each chained-BD boundary, see
    /// `enter_chained_bd`), the channel refuses the stream even with FIFO room,
    /// modeling HW's per-BD TREADY deassert so the recv port traces `on16 off1`
    /// instead of front-loading the double-buffer through the boundary. #140.
    pub fn can_accept_stream_in_for_channel(&self, channel: u8) -> bool {
        if self
            .channels
            .get(channel as usize)
            .map_or(false, |c| c.bd_switch_accept_block > 0)
        {
            return false;
        }
        let current = self.stream_in.get(channel as usize).map_or(0, |q| q.len());
        // One-BD-lookahead bound (#140): hold the next BD off until the prior BD
        // has drained from the ingress. The block is armed at BD completion and
        // cleared when the first word of the next BD is accepted; while it stands,
        // any residual prior-BD occupancy refuses the stream.
        if self.channels.get(channel as usize).map_or(false, |c| c.accept_awaiting_drain) && current > 0 {
            return false;
        }
        current < self.input_fifo_capacity()
    }

    /// Get the total number of words across all MM2S channel stream output
    /// buffers.  For per-channel inspection use `stream_out_len_for_channel`.
    pub fn stream_out_len(&self) -> usize {
        self.stream_out.iter().map(|q| q.len()).sum()
    }

    /// Get the number of words queued for a specific MM2S channel.
    pub fn stream_out_len_for_channel(&self, channel: u8) -> usize {
        let idx = self.stream_out_idx(channel);
        self.stream_out.get(idx).map_or(0, |q| q.len())
    }

    /// Get the total number of words across all stream input channel buffers.
    pub fn stream_in_len(&self) -> usize {
        self.stream_in.iter().map(|q| q.len()).sum()
    }

    /// Check if stream input buffer has data for a specific channel.
    pub fn has_stream_in_for_channel(&self, channel: u8) -> bool {
        self.stream_in.get(channel as usize).map_or(false, |q| !q.is_empty())
    }

    /// Count available stream input words for a specific channel.
    pub fn stream_in_count_for_channel(&self, channel: u8) -> usize {
        self.stream_in.get(channel as usize).map_or(0, |q| q.len())
    }

    /// Pop data from a specific channel's stream input buffer.
    ///
    /// Each channel has its own FIFO, so this is O(1) (front pop).
    /// Also increments `stream_in_drained_this_cycle` for the channel so that
    /// Phase 4's `can_accept_stream_in_for_routing` can enforce the registered-
    /// FIFO ordering invariant (#140 phase-ordering fix).
    pub(super) fn pop_stream_in_for_channel(&mut self, channel: u8) -> Option<StreamData> {
        let word = self.stream_in.get_mut(channel as usize)?.pop_front()?;
        if let Some(c) = self.stream_in_drained_this_cycle.get_mut(channel as usize) {
            *c += 1;
        }
        Some(word)
    }

    /// Reset per-channel drain counters at the start of each routing cycle.
    ///
    /// Must be called before Phase 3 (`step_all_dma`) in `step_data_movement`.
    /// Phase 4 (`route_streams`) then reads these counters via
    /// `can_accept_stream_in_for_routing` to enforce the registered-FIFO
    /// ordering invariant: a slot freed in Phase 3 is not available to the
    /// producer in Phase 4 of the same cycle.
    ///
    /// Also clears the per-cycle memory-pressure signals, which are asserted by
    /// Phase 3 / Phase 4 and consumed by the coordinator's event pass (Phase E)
    /// before the next cycle begins.
    pub fn reset_cycle_drain_counters(&mut self) {
        for c in &mut self.stream_in_drained_this_cycle {
            *c = 0;
        }
        for f in &mut self.s2mm_ingress_full_offered {
            *f = false;
        }
        for f in &mut self.mm2s_egress_empty_wanted {
            *f = false;
        }
    }

    /// The routing pass offered S2MM channel `ch` a beat and the channel refused
    /// it. Raise `DMA_S2MM_n_MEMORY_BACKPRESSURE` if -- and only if -- the reason
    /// was that the ingress FIFO is FULL: the stream has data, the channel cannot
    /// drain to memory, and the staging has run out of room. That is the whole
    /// semantics of the event on silicon
    /// (docs/superpowers/findings/2026-07-14-dma-memory-pressure-event-semantics.md);
    /// the *cause* of the channel not draining (a held lock, in practice) is not
    /// tested for here and must not be.
    ///
    /// The other reasons `can_accept_stream_in_for_routing` refuses a beat -- the
    /// one-cycle BD-switch TREADY gap, the one-BD-lookahead drain gate -- are
    /// control-path, not memory pressure, and deliberately do not raise it.
    pub fn note_ingress_offer_refused(&mut self, ch: u8) {
        let cap = self.input_fifo_capacity();
        let current = self.stream_in.get(ch as usize).map_or(0, |q| q.len());
        let drained = self.stream_in_drained_this_cycle.get(ch as usize).copied().unwrap_or(0);
        if current + drained >= cap {
            if let Some(f) = self.s2mm_ingress_full_offered.get_mut(ch as usize) {
                *f = true;
            }
        }
    }

    /// Is `DMA_S2MM_n_MEMORY_BACKPRESSURE` asserted for S2MM channel `ch` this
    /// cycle? (Ingress FIFO full with a beat on offer.)
    pub fn s2mm_memory_backpressure(&self, ch: u8) -> bool {
        self.s2mm_ingress_full_offered.get(ch as usize).copied().unwrap_or(false)
    }

    /// Is `DMA_MM2S_n_MEMORY_STARVATION` asserted for MM2S channel `ch` this
    /// cycle? (Egress staging FIFO empty with the stream port able to take a
    /// beat.) `ch` is the per-direction MM2S channel number, not the flat index.
    pub fn mm2s_memory_starvation(&self, ch: u8) -> bool {
        self.mm2s_egress_empty_wanted.get(ch as usize).copied().unwrap_or(false)
    }

    /// Words currently held in flat channel `ch`'s egress staging FIFO (MM2S;
    /// always 0 for an S2MM channel or an idle one). Inspection only.
    pub fn staged_words(&self, ch: u8) -> usize {
        self.channels
            .get(ch as usize)
            .and_then(|c| c.fsm.transfer())
            .map_or(0, |t| t.staged_words)
    }

    /// Is S2MM channel `ch` currently stalled waiting for its acquire lock?
    /// Not an event -- the DMA_S2MM_n_STALLED_LOCK trace event has its own edge
    /// bookkeeping. Exposed for the validation census, which needs the lock-stall
    /// WINDOWS to check that backpressure has silicon's shape (a sustained window
    /// starting well inside a lock stall) and not merely silicon's total.
    pub fn channel_lock_stalled(&self, ch: u8) -> bool {
        self.channels.get(ch as usize).is_some_and(|c| c.prev_lock_stalled)
    }

    /// Depth (32-bit words) of an MM2S channel's egress staging FIFO -- the
    /// buffer between the 128-bit memory read and the 32-bit stream port. The
    /// memory side fills it a whole 16-byte granule at a time in one bank slot;
    /// the stream side drains one word per cycle. That asymmetry is what gives
    /// the DMA three spare memory cycles in four, so a lost bank arbitration
    /// delays a refill the FIFO has slack to cover instead of bubbling the
    /// stream (`DMA_MM2S_EGRESS_FIFO_DEPTH`, see its archspec field doc).
    ///
    /// Shim MM2S reads host DDR, not banked tile memory, and is governed by the
    /// separately calibrated DDR cold-start model -- it does not stage.
    pub fn egress_staging_capacity(&self) -> usize {
        xdna_archspec::aie2::timing::DMA_MM2S_EGRESS_FIFO_DEPTH as usize
    }

    /// Check if a specific S2MM channel can accept a word from the routing phase.
    ///
    /// Like `can_accept_stream_in_for_channel` but enforces the registered-FIFO
    /// ordering invariant: when Phase 3 (step_all_dma) drained D words from a
    /// channel's ingress, Phase 4 (route_streams) treats those D slots as still
    /// consumed for the purpose of the capacity check. On real HW, TREADY is a
    /// registered signal: the producer sees the FULL state from the end of the
    /// prior cycle and cannot push even though the consumer drained in the same
    /// cycle. The effective start-of-cycle occupancy is `len + drained_this_cycle`
    /// and must be below capacity for the channel to accept.
    ///
    /// Also deasserts during a BD-switch reconfiguration (same as
    /// `can_accept_stream_in_for_channel`). Must be called AFTER
    /// `reset_cycle_drain_counters()` and Phase 3 have both run.
    pub fn can_accept_stream_in_for_routing(&self, channel: u8) -> bool {
        if self
            .channels
            .get(channel as usize)
            .map_or(false, |c| c.bd_switch_accept_block > 0)
        {
            return false;
        }
        let cap = self.input_fifo_capacity();
        let current = self.stream_in.get(channel as usize).map_or(0, |q| q.len());
        let drained = self.stream_in_drained_this_cycle.get(channel as usize).copied().unwrap_or(0);
        // One-BD-lookahead bound (#140): hold the next BD off until the prior BD
        // has drained. Use the registered start-of-cycle occupancy (current +
        // drained) so a slot freed by this cycle's Phase 3 drain isn't seen as
        // drained until next cycle -- the same registered-FIFO semantics as the
        // capacity check, so the next BD reopens one cycle after the prior BD's
        // last word leaves, matching HW's registered TREADY.
        if self.channels.get(channel as usize).map_or(false, |c| c.accept_awaiting_drain)
            && current + drained > 0
        {
            return false;
        }
        current + drained < cap
    }

    // === Stream Port Mapping Integration ===
    //
    // These methods integrate with the stream_io module's port mappings,
    // providing a unified interface for determining which stream switch
    // ports DMA channels connect to.

    /// Get the slave port that MM2S channel `ch` sends data TO.
    ///
    /// For compute tiles: ch0 -> slave port 1, ch1 -> slave port 2
    /// For memtiles: ch0-5 -> slave ports 0-5
    /// For shim tiles: ch0 -> slave port 2, ch1 -> slave port 3 (South ports)
    pub fn mm2s_slave_port(&self, ch: u8) -> u8 {
        match self.tile_kind {
            TileKind::Compute => super::super::stream_io::compute::mm2s_slave_port(ch),
            TileKind::Mem => super::super::stream_io::memtile::mm2s_slave_port(ch),
            TileKind::ShimNoc | TileKind::ShimPl => super::super::stream_io::shim::mm2s_slave_port(ch),
        }
    }

    /// Get the master port that S2MM channel `ch` receives data FROM.
    ///
    /// For compute tiles: ch0 <- master port 1, ch1 <- master port 2
    /// For memtiles: ch0-5 <- master ports 0-5
    /// For shim tiles: ch0 <- master port 2, ch1 <- master port 3 (South ports)
    pub fn s2mm_master_port(&self, ch: u8) -> u8 {
        match self.tile_kind {
            TileKind::Compute => super::super::stream_io::compute::s2mm_master_port(ch),
            TileKind::Mem => super::super::stream_io::memtile::s2mm_master_port(ch),
            TileKind::ShimNoc | TileKind::ShimPl => super::super::stream_io::shim::s2mm_master_port(ch),
        }
    }

    /// Get the number of S2MM channels for this tile.
    pub fn s2mm_channel_count(&self) -> usize {
        self.s2mm_count
    }

    /// Get the number of MM2S channels for this tile.
    pub fn mm2s_channel_count(&self) -> usize {
        self.mm2s_count
    }

    /// Convert a StreamWord to StreamData for a given channel.
    ///
    /// This bridges the stream_io module's `StreamWord` (used by stream switches)
    /// with the engine's `StreamData` (which tracks channel ownership).
    pub fn stream_word_to_data(word: super::super::stream_io::StreamWord, channel: u8) -> StreamData {
        StreamData { data: word.data, tlast: word.tlast, channel }
    }

    /// Convert a StreamData to StreamWord.
    ///
    /// Drops the channel information (stream switches don't track channel).
    /// Parity is computed from the data.
    pub fn stream_data_to_word(data: &StreamData) -> super::super::stream_io::StreamWord {
        super::super::stream_io::StreamWord {
            data: data.data,
            tlast: data.tlast,
            parity: super::super::stream_io::StreamWord::compute_parity(data.data),
        }
    }

    /// Pop stream output as StreamWord (for stream switch integration).
    ///
    /// Returns the word and the channel it came from.  Iterates per-channel
    /// queues -- if multiple channels have data, picks the first non-empty.
    pub fn pop_stream_out_as_word(&mut self) -> Option<(super::super::stream_io::StreamWord, u8)> {
        self.pop_stream_out()
            .map(|data| (Self::stream_data_to_word(&data), data.channel))
    }

    /// Push stream input from StreamWord (for stream switch integration).
    ///
    /// Requires specifying which S2MM channel should receive this data.
    pub fn push_stream_in_from_word(
        &mut self,
        word: super::super::stream_io::StreamWord,
        channel: u8,
    ) -> bool {
        self.push_stream_in(Self::stream_word_to_data(word, channel))
    }
}

// ============================================================================
// StreamData <-> StreamWord Conversions
// ============================================================================

impl From<StreamData> for super::super::stream_io::StreamWord {
    fn from(data: StreamData) -> Self {
        Self { data: data.data, tlast: data.tlast, parity: Self::compute_parity(data.data) }
    }
}

impl StreamData {
    /// Create StreamData from a StreamWord with specified channel.
    pub fn from_stream_word(word: super::super::stream_io::StreamWord, channel: u8) -> Self {
        Self { data: word.data, tlast: word.tlast, channel }
    }
}
