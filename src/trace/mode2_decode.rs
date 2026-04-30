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

/// Bit-level reader over a byte stream, MSB-first within each byte.
/// Used by the mode-2 decoder to walk the prefix tree.
struct BitStream<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitStream<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    fn next_bit(&mut self) -> Option<u8> {
        let byte_idx = self.bit_pos / 8;
        if byte_idx >= self.bytes.len() {
            return None;
        }
        let shift = 7 - (self.bit_pos % 8);
        let bit = (self.bytes[byte_idx] >> shift) & 1;
        self.bit_pos += 1;
        Some(bit)
    }

    fn take(&mut self, n: usize) -> Option<u32> {
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | self.next_bit()? as u32;
        }
        Some(val)
    }
}

/// Decode a mode-2 byte stream into a sequence of frames.
///
/// Bit ordering is MSB-first within each byte. Stops at end-of-stream
/// or first frame that fails to decode (returns the frames decoded so
/// far). Unknown prefixes drain the rest of the stream silently --
/// real captures may interleave tiles configured for different modes.
pub fn decode(bytes: &[u8]) -> Vec<Mode2Frame> {
    let mut bits = BitStream::new(bytes);
    let mut out = Vec::new();
    while let Some(frame) = decode_one(&mut bits) {
        out.push(frame);
    }
    out
}

fn decode_one(bits: &mut BitStream) -> Option<Mode2Frame> {
    let b0 = bits.next_bit()?;
    if b0 == 0 {
        // 0xxx prefix
        let b1 = bits.next_bit()?;
        if b1 == 0 {
            // 00xx
            let b2 = bits.next_bit()?;
            let b3 = bits.next_bit()?;
            return Some(match (b2, b3) {
                (0, 0) => Mode2Frame::Atom { executed: false },
                (0, 1) => Mode2Frame::Atom { executed: true },
                (1, 0) => Mode2Frame::Filler0,
                _ => return None, // 0011 unused; treat as EOF
            });
        }
        // 01x prefix -- LC (Task 4.4) or unknown
        return None;
    }
    // 1xxx prefix
    let b1 = bits.next_bit()?;
    if b1 == 1 {
        // 11xxxxxxxx prefixes
        let b2 = bits.next_bit()?;
        let b3 = bits.next_bit()?;
        if b2 == 1 && b3 == 1 {
            // 1111x -- Start (Task 4.5) / Filler1 / Sync
            let b4 = bits.next_bit()?;
            if b4 == 1 {
                // 11111xxx -- Filler1 / Sync
                let b5 = bits.next_bit()?;
                let b6 = bits.next_bit()?;
                let b7 = bits.next_bit()?;
                return match (b5, b6, b7) {
                    (1, 1, 1) => Some(Mode2Frame::Sync),
                    (1, 1, 0) => Some(Mode2Frame::Filler1),
                    _ => None, // unknown 11111xxx
                };
            }
            // 11110 -- Start (Task 4.5)
            return None;
        }
        // 1100/1101 -- Multiple/Repeat/Stop (Task 4.4)
        return None;
    }
    // 10x -- New_PC (Task 4.3)
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_e_atom() {
        // E_atom prefix is 0001 -- one E_atom in high nibble of a byte
        // packed with another E_atom in the low nibble produces 0x11.
        let bytes = [0x11];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], Mode2Frame::Atom { executed: true });
        assert_eq!(frames[1], Mode2Frame::Atom { executed: true });
    }

    #[test]
    fn decodes_n_atom() {
        // N_atom prefix is 0000 -- byte 0x00 is two N_atoms.
        let bytes = [0x00];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], Mode2Frame::Atom { executed: false });
        assert_eq!(frames[1], Mode2Frame::Atom { executed: false });
    }

    #[test]
    fn decodes_filler0() {
        // Filler0 prefix is 0010 -- byte 0x22 is two Filler0s.
        let bytes = [0x22];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], Mode2Frame::Filler0);
        assert_eq!(frames[1], Mode2Frame::Filler0);
    }

    #[test]
    fn decodes_sync() {
        // Sync prefix is 11111111 -- exactly one byte = 0xFF.
        let bytes = [0xFF];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], Mode2Frame::Sync);
    }
}
