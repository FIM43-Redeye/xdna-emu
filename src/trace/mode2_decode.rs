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
/// far). Unknown prefixes terminate decoding -- real captures may interleave
/// tiles configured for different modes, but in those cases the caller
/// should be filtering by tile-key before invoking decode().
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
        // 01x prefix
        let b2 = bits.next_bit()?;
        if b2 == 0 {
            // 010 -- LC: 1b flag + 28b count
            let flag = bits.next_bit()?;
            let count = bits.take(28)?;
            return Some(Mode2Frame::Lc { flag, count });
        }
        // 011 unused
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
        // 110x -- Repeat0, Repeat1, Stop
        if b2 == 1 && b3 == 0 {
            // 1110 -- Repeat0: 4b count
            let count = bits.take(4)? as u8;
            return Some(Mode2Frame::Repeat0 { count });
        }
        // b2==0 (i.e. 110x where x is bit b3)
        let b4 = bits.next_bit()?;
        let b5 = bits.next_bit()?;
        if b4 == 1 && b5 == 0 {
            // 110110 -- Repeat1: 10b count
            let count = bits.take(10)? as u16;
            return Some(Mode2Frame::Repeat1 { count });
        }
        if b4 == 1 && b5 == 1 {
            // 110111 -- Stop: 26b payload, discarded
            bits.take(26)?;
            return Some(Mode2Frame::Stop);
        }
        return None; // unknown 110x prefix
    }
    // 10 -- New_PC: 14-bit PC follows
    let pc = bits.take(14)? as u16;
    Some(Mode2Frame::NewPc { pc })
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

    #[test]
    fn decodes_new_pc() {
        // New_PC prefix is 10, then 14b PC. Pack PC = 0x1234 (binary
        // 01_0010_0011_0100). Full 16-bit frame: 10 01_0010_0011_0100
        // = 0x9234. As bytes (MSB-first): [0x92, 0x34].
        let bytes = [0x92, 0x34];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], Mode2Frame::NewPc { pc: 0x1234 });
    }

    #[test]
    fn decodes_new_pc_zero() {
        // PC = 0x0000: bits 10 00_0000_0000_0000 = 0x8000 -> [0x80, 0x00]
        let bytes = [0x80, 0x00];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], Mode2Frame::NewPc { pc: 0x0000 });
    }

    #[test]
    fn decodes_lc() {
        // LC prefix 010, then 1b flag + 28b count. Total 32 bits.
        // flag=1, count=0xABCDEF: bits 010 1 (28b 0x0ABCDEF aligned)
        // 010_1_0000_1010_1011_1100_1101_1110_1111
        // = 0x50ABCDEF. As BE bytes: [0x50, 0xAB, 0xCD, 0xEF]
        let bytes = [0x50, 0xAB, 0xCD, 0xEF];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], Mode2Frame::Lc { flag: 1, count: 0x0ABCDEF });
    }

    #[test]
    fn decodes_lc_flag_zero() {
        // flag=0, count=8: 010 0 (28b 0x0000008)
        // bits: 010_0_0000_0000_0000_0000_0000_0000_1000 = 0x40000008
        // BE bytes: [0x40, 0x00, 0x00, 0x08]
        let bytes = [0x40, 0x00, 0x00, 0x08];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], Mode2Frame::Lc { flag: 0, count: 8 });
    }

    #[test]
    fn decodes_repeat0() {
        // Repeat0 prefix 1110, then 4b count. count=5: 1110_0101 = 0xE5
        let bytes = [0xE5];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], Mode2Frame::Repeat0 { count: 5 });
    }

    #[test]
    fn decodes_repeat1() {
        // Repeat1 prefix 110110, then 10b count. count=0x1FF (511):
        // bits 110110 + 01_1111_1111 = 110110_0111111111
        // = 0xD9FF. BE bytes: [0xD9, 0xFF]
        let bytes = [0xD9, 0xFF];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], Mode2Frame::Repeat1 { count: 0x1FF });
    }

    #[test]
    fn decodes_stop() {
        // Stop prefix 110111, then 26b payload (consumes word).
        // Payload value doesn't matter; we discard it.
        // bits 110111 + 26 zeros = 110111_00_00000000_00000000_00000000
        // = 0xDC000000. BE bytes: [0xDC, 0x00, 0x00, 0x00]
        let bytes = [0xDC, 0x00, 0x00, 0x00];
        let frames = decode(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], Mode2Frame::Stop);
    }
}
