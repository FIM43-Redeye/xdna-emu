//! Stream and cascade I/O methods on Tile.

use super::*;

impl Tile {
    // === Stream Port Access (for core direct stream reads/writes) ===

    /// Push a word to the stream input buffer for a port.
    ///
    /// Called by the stream router when data arrives for this tile.
    pub fn push_stream_input(&mut self, port: u8, value: u32) {
        if (port as usize) < self.stream_input.len() {
            self.stream_input[port as usize].push_back(value);
        }
    }

    /// Pop a word from the stream input buffer for a port.
    ///
    /// Called by StreamReadScalar when the core reads from a stream port.
    /// Returns None if no data is available (should stall if blocking).
    pub fn pop_stream_input(&mut self, port: u8) -> Option<u32> {
        if (port as usize) < self.stream_input.len() {
            self.stream_input[port as usize].pop_front()
        } else {
            None
        }
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

    /// Push a word to the stream output buffer for a port.
    ///
    /// Called by StreamWriteScalar when the core writes to a stream port.
    pub fn push_stream_output(&mut self, port: u8, value: u32) {
        if (port as usize) < self.stream_output.len() {
            self.stream_output[port as usize].push_back(value);
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

    /// Pop a word from the stream output buffer for a port.
    ///
    /// Called by the stream router to collect data from this tile.
    pub fn pop_stream_output(&mut self, port: u8) -> Option<u32> {
        if (port as usize) < self.stream_output.len() {
            self.stream_output[port as usize].pop_front()
        } else {
            None
        }
    }

    // === Cascade Stream Helpers ===

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
