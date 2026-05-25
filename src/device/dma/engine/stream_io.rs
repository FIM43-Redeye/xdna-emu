//! Stream interface methods for TileArray integration.

use super::*;

/// Maximum number of words buffered per stream input FIFO.
///
/// Each S2MM channel has its own FIFO. Value determined empirically
/// from bridge-test traces; not an AM025 register (the hardware FIFO
/// is much smaller, but the emulator buffers more to absorb
/// scheduler jitter without blocking). If this proves too shallow
/// or too deep in practice, tune here in one place.
pub(super) const STREAM_BUFFER_CAPACITY_WORDS: usize = 256;

impl DmaEngine {
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
        if self.stream_in[ch].len() < STREAM_BUFFER_CAPACITY_WORDS {
            self.stream_in[ch].push_back(data);
            true
        } else {
            let msg = format!(
                "DMA({},{}) stream_in buffer full ({}), dropping ch{} data: 0x{:08X} -- \
                 backpressure violation",
                self.col, self.row, STREAM_BUFFER_CAPACITY_WORDS, data.channel, data.data,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            false
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
        self.stream_in.iter().any(|q| q.len() < STREAM_BUFFER_CAPACITY_WORDS)
    }

    /// Check if a specific S2MM channel's stream input buffer has space.
    pub fn can_accept_stream_in_for_channel(&self, channel: u8) -> bool {
        self.stream_in
            .get(channel as usize)
            .map_or(false, |q| q.len() < STREAM_BUFFER_CAPACITY_WORDS)
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
