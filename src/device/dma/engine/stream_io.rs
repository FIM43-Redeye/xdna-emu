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
    /// On the **memtile** the DMA also has a deeper per-channel S2MM ingress
    /// buffer (device-model `s2mmChannel.buffer_depth` = 12) downstream of that
    /// master port; together they let a recv port stage a full objectfifo BD
    /// ahead of a lock-stalled buffer write (HW co-capture #140, 2026-06-27 --
    /// see `DMA_MEMTILE_S2MM_INGRESS_FIFO_DEPTH`). EMU counts the master-port
    /// beat at the DMA-accept point, so this single capacity stands in for the
    /// combined DMA-buffer + master-FIFO ingress (= 16). Compute/shim keep the
    /// shallow master-port depth until their ingress is separately calibrated.
    ///
    /// History: this was a hardcoded 256-word buffer "to absorb scheduler
    /// jitter without blocking." That depth silently absorbed ~256/buffer_words
    /// extra fast iterations during double-buffer warmup, defeating the
    /// otherwise HW-faithful master-port backpressure and producing a warmup
    /// transient much longer than silicon (root-caused 2026-06-13 via the
    /// dma_passthrough buffer-size sweep). The memtile value (16) is far shallower
    /// (~1 BD of warmup slack). The consume-before-produce cycle order
    /// (`step_all_dma` Phase 3 drains before `route_streams` Phase 4 fills) means
    /// the depth does not spuriously overflow.
    pub fn input_fifo_capacity(&self) -> usize {
        use xdna_archspec::aie2::timing;
        if self.tile_kind.is_mem() {
            timing::DMA_MEMTILE_S2MM_INGRESS_FIFO_DEPTH as usize
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
        self.stream_in
            .get(channel as usize)
            .map_or(false, |q| q.len() < self.input_fifo_capacity())
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
    pub(super) fn pop_stream_in_for_channel(&mut self, channel: u8) -> Option<StreamData> {
        self.stream_in.get_mut(channel as usize)?.pop_front()
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
