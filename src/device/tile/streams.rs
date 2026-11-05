//! Stream and cascade I/O methods on Tile.

use super::*;

/// Words a cascade link holds between a producer's `vmov MCD` and the consumer's `vmov SCD`
/// taking them: 2. A write needs a free slot; a word is in the link from its write until the
/// consumer takes it (in transit or buffered at the consumer alike).
///
/// Measured on Phoenix silicon. Harness depth probe (`tools/gen_cascade_probe.py`; consumer idle
/// for 1500-5000 cycles, producer issues back-to-back MCD writes with a cycle-counter stamp after
/// each): the third write always stalls until the consumer reads, in 27/27 stalling runs over
/// South and East pairs. Timing probe (`tools/gen_cascade_timing_probe.py`, families `wrel` and
/// `rt`): a stalled write issues in the very cycle the consumer's hold bundle takes a word, and a
/// consumer reading back to back after a full link never waits, i.e. the slot is free at once.
/// Where the two words physically sit is not observable: every probe sees the link as this
/// credit window plus [`CASCADE_TRANSIT_CYCLES`].
pub const CASCADE_TOTAL_DEPTH: usize = 2;

/// Cycles from a `vmov MCD` issuing to its word being readable at the consumer: 2. A word
/// written in cycle t is taken by a consumer bundle issuing in t + 2 or later, held or not; a
/// bundle held on an empty SCD issues in the first cycle a word is readable (no restart
/// penalty). Silicon, `tools/gen_cascade_timing_probe.py` family `rel` (consumer stall
/// max(0, n - 4) over n = 0..24, South and East, r1 == r2), with the one-way split from the
/// shared-memory timer-phase probe (`mf*`/`mr*`: the two tiles of a column read the same
/// timer). The link routing realizes it as two end-of-cycle moves: MCD -> transit -> SCD.
pub const CASCADE_TRANSIT_CYCLES: usize = 2;

/// Words the core's stream input buffer holds before the Core master port is backpressured: 1.
/// AM020 ch.2:143 gives the core's AXI4-Stream interface no FIFO, so this is only the hand-off
/// register between the switch and the `mov r, SS` read; the buffering of the path is the Core
/// master port's own 2-deep FIFO (ch.2:82). Without a bound the buffer absorbed any number of
/// words and a held consumer never backpressured its producer (silicon: 18 words between two
/// cores one tile apart, `tools/gen_stream_absorb_probe.py`).
pub const CORE_STREAM_INPUT_DEPTH: usize = 1;

impl Tile {
    // === Stream Port Access (for core direct stream reads/writes) ===

    /// Push a word (no TLAST) to the stream input buffer for a port.
    pub fn push_stream_input(&mut self, port: u8, value: u32) {
        self.push_stream_input_beat(port, value, false);
    }

    /// Push a `(word, tlast)` beat to the stream input buffer for a port.
    ///
    /// Called by the stream router when data arrives for this tile.
    pub fn push_stream_input_beat(&mut self, port: u8, value: u32, tlast: bool) {
        if (port as usize) < self.stream_input.len() {
            self.stream_input[port as usize].push_back((value, tlast));
        }
    }

    /// Pop a word from the stream input buffer for a port, dropping its TLAST.
    pub fn pop_stream_input(&mut self, port: u8) -> Option<u32> {
        self.pop_stream_input_beat(port).map(|(word, _)| word)
    }

    /// Pop a `(word, tlast)` beat from the stream input buffer for a port.
    ///
    /// Called by StreamReadScalar when the core reads from a stream port.
    /// Returns None if no data is available (should stall if blocking).
    pub fn pop_stream_input_beat(&mut self, port: u8) -> Option<(u32, bool)> {
        if (port as usize) < self.stream_input.len() {
            self.stream_input[port as usize].pop_front()
        } else {
            None
        }
    }

    /// Whether the core's stream input buffer for `port` can take another beat.
    pub fn stream_input_has_room(&self, port: u8) -> bool {
        self.stream_input_len(port) < CORE_STREAM_INPUT_DEPTH
    }

    /// Check if stream input has data for a port.
    pub fn has_stream_input(&self, port: u8) -> bool {
        if (port as usize) < self.stream_input.len() {
            !self.stream_input[port as usize].is_empty()
        } else {
            false
        }
    }

    /// Get stream input queue length for a port.
    pub fn stream_input_len(&self, port: u8) -> usize {
        if (port as usize) < self.stream_input.len() {
            self.stream_input[port as usize].len()
        } else {
            0
        }
    }

    /// Push a word (no TLAST) to the stream output buffer for a port.
    pub fn push_stream_output(&mut self, port: u8, value: u32) {
        self.push_stream_output_beat(port, value, false);
    }

    /// Push a `(word, tlast)` beat to the stream output buffer for a port.
    ///
    /// Called by StreamWriteScalar when the core writes to a stream port.
    pub fn push_stream_output_beat(&mut self, port: u8, value: u32, tlast: bool) {
        if (port as usize) < self.stream_output.len() {
            self.stream_output[port as usize].push_back((value, tlast));
        }
    }

    /// Get stream output queue length for a port.
    pub fn stream_output_len(&self, port: u8) -> usize {
        if (port as usize) < self.stream_output.len() {
            self.stream_output[port as usize].len()
        } else {
            0
        }
    }

    /// Pop a `(word, tlast)` beat from the stream output buffer for a port.
    ///
    /// Called by the stream router to collect data from this tile.
    pub fn pop_stream_output_beat(&mut self, port: u8) -> Option<(u32, bool)> {
        if (port as usize) < self.stream_output.len() {
            self.stream_output[port as usize].pop_front()
        } else {
            None
        }
    }

    // === Cascade Stream Helpers ===

    /// True when the outgoing link holds [`CASCADE_TOTAL_DEPTH`] words, so `vmov MCD` stalls.
    /// Counts this tile's MCD stage plus the downstream occupancy the coordinator refreshed.
    pub fn cascade_output_full(&self) -> bool {
        self.cascade_output.len() + self.cascade_downstream_occupancy >= CASCADE_TOTAL_DEPTH
    }

    /// Words of the incoming link held at this tile: in transit and readable.
    pub fn cascade_inbound_occupancy(&self) -> usize {
        self.cascade_wire.len() + self.cascade_input.len()
    }

    /// Push a 512-bit value into the cascade input FIFO (SCD).
    pub fn push_cascade_input(&mut self, data: [u64; xdna_archspec::aie2::CASCADE_WORDS]) {
        self.cascade_input.push_back(data);
    }

    /// Pop a 512-bit value from the cascade input FIFO (SCD).
    /// Returns None if the FIFO is empty (core should stall).
    pub fn pop_cascade_input(&mut self) -> Option<[u64; xdna_archspec::aie2::CASCADE_WORDS]> {
        self.cascade_input.pop_front()
    }

    /// Push a 512-bit value into the cascade output FIFO (MCD).
    pub fn push_cascade_output(&mut self, data: [u64; xdna_archspec::aie2::CASCADE_WORDS]) {
        self.cascade_output.push_back(data);
    }

    /// Pop a 512-bit value from the cascade output FIFO (MCD).
    pub fn pop_cascade_output(&mut self) -> Option<[u64; xdna_archspec::aie2::CASCADE_WORDS]> {
        self.cascade_output.pop_front()
    }

    /// Check if cascade input has data available.
    pub fn has_cascade_input(&self) -> bool {
        !self.cascade_input.is_empty()
    }

    /// Check if cascade output has data (for routing to neighbor).
    pub fn has_cascade_output(&self) -> bool {
        !self.cascade_output.is_empty()
    }
}
