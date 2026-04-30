//! Mode-2 (INST_EXEC) trace bit-stream decoder.
//!
//! Ports `tools/trace_decoder/modes/mode2.py` to Rust. Used by:
//! - encoder unit tests (round-trip verification)
//! - the comparator (`compare_mode2.rs`)
//!
//! Original implementation. Frame tree was recovered once already
//! from libxv_trace_decoder_opt.so symbols; this re-implementation
//! consumes that recovered tree (mode2.py docstring) directly.

/// One decoded mode-2 frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode2Frame {
    Atom { executed: bool },
    Filler0,
    Filler1,
    Sync,
    NewPc { pc: u16 },
    Repeat0 { count: u8 },
    Repeat1 { count: u16 },
    Start { anchor_pc: u16 },
    Lc { flag: u8, count: u32 },
    Stop,
}

/// Decode a mode-2 byte stream into a sequence of frames.
///
/// Bit ordering is MSB-first within each byte. Stops at end-of-stream
/// or first frame that fails to decode (returns the frames decoded so
/// far). Unknown prefixes drain the rest of the stream silently --
/// real captures may interleave tiles configured for different modes.
pub fn decode(_bytes: &[u8]) -> Vec<Mode2Frame> {
    todo!("implemented in Task 4.2+")
}

#[cfg(test)]
mod tests {
    // Tests added in Task 4.2+.
}
