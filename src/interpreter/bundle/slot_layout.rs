//! VLIW bundle slot extraction based on AIE2CompositeFormats.td.
//!
//! Each bundle format has specific slot layouts. This module extracts
//! individual slot data from packed VLIW bundles.
//!
//! # Format Structure
//!
//! Bundles have format markers in their low bits and slot data packed
//! hierarchically. The marker determines the bundle size, and discriminator
//! bits identify which slot combination is present.
//!
//! # Slot Bit Widths
//!
//! | Slot | Width |
//! |------|-------|
//! | lda  | 21    |
//! | ldb  | 16    |
//! | alu  | 20    |
//! | mv   | 22    |
//! | st   | 21    |
//! | vec  | 26    |
//! | lng  | 42    |

use super::BundleFormat;

/// Slot widths in bits (from AIE2Slots.td).
pub const LDA_WIDTH: u8 = 21;
pub const LDB_WIDTH: u8 = 16;
pub const ALU_WIDTH: u8 = 20;
pub const MV_WIDTH: u8 = 22;
pub const ST_WIDTH: u8 = 21;
pub const VEC_WIDTH: u8 = 26;
pub const LNG_WIDTH: u8 = 42;

/// Which slot type was detected in a bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotType {
    Lda,
    Ldb,
    Alu,
    Mv,
    St,
    Vec,
    Lng,
    Nop,
}

/// Extracted slot data from a bundle.
#[derive(Debug, Clone)]
pub struct ExtractedSlot {
    /// The slot type.
    pub slot_type: SlotType,
    /// The raw slot bits (right-aligned).
    pub bits: u64,
    /// The bit width of this slot.
    pub width: u8,
}

/// Result of extracting slots from a bundle.
#[derive(Debug, Clone, Default)]
pub struct ExtractedBundle {
    /// All extracted slots.
    pub slots: Vec<ExtractedSlot>,
}

impl ExtractedBundle {
    /// Create an empty bundle.
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Add a slot to this bundle.
    pub fn add_slot(&mut self, slot_type: SlotType, bits: u64, width: u8) {
        self.slots.push(ExtractedSlot {
            slot_type,
            bits,
            width,
        });
    }

    /// Check if this bundle is empty (no slots extracted).
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// Extract slot data from a 32-bit bundle.
///
/// 32-bit bundles contain a single slot. The slot type is determined
/// by discriminator bits in the instruction word.
///
/// # Format Variants (from AIE2CompositeFormats.td)
///
/// - I32_VEC: `{vec[25:0], 0b00, 0b1001}` - bits 5:4 = 00
/// - I32_ST:  `{0b00001, st[20:0], 0b01, 0b1001}` - bits 31:27=00001, bits 5:4=01
/// - I32_MV:  `{0b00011, mv[21:0], 0b1, 0b1001}` - bits 31:27=00011, bit 4=1
/// - I32_LDB: `{0b00111, ldb[15:0], 0b0000001, 0b1001}` - bits 31:27=00111
/// - I32_LDA: `{0b00000, lda[20:0], 0b01, 0b1001}` - bits 31:27=00000, bits 5:4=01
/// - I32_ALU: `{0b00010, alu[19:0], 0b001, 0b1001}` - bits 31:27=00010, bits 6:4=001
pub fn extract_32bit(word: u32) -> ExtractedBundle {
    let mut bundle = ExtractedBundle::new();

    // Verify format marker (should be 0x9)
    if (word & 0xF) != BundleFormat::MARKER_SHORT32 as u32 {
        return bundle;
    }

    // Check discriminator bits
    let high5 = (word >> 27) & 0x1F; // bits 31:27
    let mid_bits = (word >> 4) & 0x7; // bits 6:4

    match high5 {
        0b00000 => {
            // Could be LDA (bits 5:4=01) or VEC (bits 5:4=00)
            if mid_bits & 0x3 == 0b01 {
                // I32_LDA: lda at bits 26:6
                let lda_bits = (word >> 6) & ((1 << LDA_WIDTH) - 1);
                bundle.add_slot(SlotType::Lda, lda_bits as u64, LDA_WIDTH);
            } else if mid_bits & 0x3 == 0b00 {
                // I32_VEC: vec at bits 31:6
                let vec_bits = (word >> 6) & ((1 << VEC_WIDTH) - 1);
                bundle.add_slot(SlotType::Vec, vec_bits as u64, VEC_WIDTH);
            }
        }
        0b00001 => {
            // I32_ST: st at bits 26:6
            let st_bits = (word >> 6) & ((1 << ST_WIDTH) - 1);
            bundle.add_slot(SlotType::St, st_bits as u64, ST_WIDTH);
        }
        0b00010 => {
            // I32_ALU: alu at bits 26:7
            let alu_bits = (word >> 7) & ((1 << ALU_WIDTH) - 1);
            bundle.add_slot(SlotType::Alu, alu_bits as u64, ALU_WIDTH);
        }
        0b00011 => {
            // I32_MV: mv at bits 26:5
            let mv_bits = (word >> 5) & ((1 << MV_WIDTH) - 1);
            bundle.add_slot(SlotType::Mv, mv_bits as u64, MV_WIDTH);
        }
        0b00111 => {
            // I32_LDB: ldb at bits 26:11
            let ldb_bits = (word >> 11) & ((1 << LDB_WIDTH) - 1);
            bundle.add_slot(SlotType::Ldb, ldb_bits as u64, LDB_WIDTH);
        }
        _ => {
            // Default to VEC if discriminator doesn't match known patterns
            // VEC uses bits 31:6 when bits 5:4 = 00
            if mid_bits & 0x3 == 0b00 {
                let vec_bits = (word >> 6) & ((1 << VEC_WIDTH) - 1);
                bundle.add_slot(SlotType::Vec, vec_bits as u64, VEC_WIDTH);
            }
        }
    }

    bundle
}

/// Extract slot data from a 48-bit (6-byte) bundle.
///
/// # Format Variants
///
/// - I48_LNG: `{lng[41:0], 0b010, 0b101}` - single lng slot
/// - I48_ST_ALU: `{st[20:0], alu[19:0], 0b0111, 0b101}` - bits 7:3=0b0111
/// - I48_LDB_ST: `{0b00001, ldb[15:0], st[20:0], 0b000, 0b101}` - bits 47:43=00001
/// - I48_LDB_MV: `{0b00000, ldb[15:0], mv[21:0], 0b00, 0b101}` - bits 47:43=00000
/// - I48_LDB_ALU: `{0b00010, ldb[15:0], alu[19:0], 0b0000, 0b101}` - bits 47:43=00010
/// - I48_LDA_ST: `{lda[20:0], st[20:0], 0b110, 0b101}` - bits 5:3=110
/// - I48_LDA_MV: `{lda[20:0], mv[21:0], 0b01, 0b101}` - bits 4:3=01
/// - I48_LDA_LDB: `{lda[20:0], ldb[15:0], 0b00001111, 0b101}` - bits 10:3=0b00001111
/// - I48_LDA_ALU: `{lda[20:0], alu[19:0], 0b0011, 0b101}` - bits 6:3=0011
pub fn extract_48bit(bytes: &[u8]) -> ExtractedBundle {
    let mut bundle = ExtractedBundle::new();

    if bytes.len() < 6 {
        return bundle;
    }

    // Read 48 bits as u64 (little-endian)
    let mut word: u64 = 0;
    for (i, &b) in bytes.iter().take(6).enumerate() {
        word |= (b as u64) << (i * 8);
    }

    // Verify format marker (low 3 bits should be 0b101 = 5)
    if (word & 0x7) != 0b101 {
        return bundle;
    }

    // Check discriminator patterns
    let bits_10_3 = ((word >> 3) & 0xFF) as u8;
    let bits_7_3 = ((word >> 3) & 0x1F) as u8;
    let bits_6_3 = ((word >> 3) & 0xF) as u8;
    let bits_5_3 = ((word >> 3) & 0x7) as u8;
    let high5 = ((word >> 43) & 0x1F) as u8; // bits 47:43

    // I48_LNG: lone lng slot uses bits 5:3 = 010
    if bits_5_3 == 0b010 {
        let lng_bits = (word >> 6) & ((1u64 << LNG_WIDTH) - 1);
        bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        return bundle;
    }

    // I48_LDA_LDB: bits 10:3 = 0b00001111
    if bits_10_3 == 0b00001111 {
        let ldb_bits = (word >> 11) & ((1u64 << LDB_WIDTH) - 1);
        let lda_bits = (word >> 27) & ((1u64 << LDA_WIDTH) - 1);
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    // I48_ST_ALU: bits 7:3 = 0b0111
    if bits_7_3 == 0b00111 {
        let alu_bits = (word >> 8) & ((1u64 << ALU_WIDTH) - 1);
        let st_bits = (word >> 28) & ((1u64 << ST_WIDTH) - 1);
        bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        return bundle;
    }

    // Check LDB variants by high5
    match high5 {
        0b00001 => {
            // I48_LDB_ST: {0b00001, ldb[15:0], st[20:0], 0b000, 0b101}
            let st_bits = (word >> 6) & ((1u64 << ST_WIDTH) - 1);
            let ldb_bits = (word >> 27) & ((1u64 << LDB_WIDTH) - 1);
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            return bundle;
        }
        0b00000 => {
            // I48_LDB_MV: {0b00000, ldb[15:0], mv[21:0], 0b00, 0b101}
            let mv_bits = (word >> 5) & ((1u64 << MV_WIDTH) - 1);
            let ldb_bits = (word >> 27) & ((1u64 << LDB_WIDTH) - 1);
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            return bundle;
        }
        0b00010 => {
            // I48_LDB_ALU: {0b00010, ldb[15:0], alu[19:0], 0b0000, 0b101}
            let alu_bits = (word >> 7) & ((1u64 << ALU_WIDTH) - 1);
            let ldb_bits = (word >> 27) & ((1u64 << LDB_WIDTH) - 1);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            return bundle;
        }
        _ => {}
    }

    // LDA variants - check discriminator bits carefully
    // I48_LDA_ST: {lda[20:0], st[20:0], 0b110, 0b101} - bits 5:3 = 110
    if bits_5_3 == 0b110 {
        let st_bits = (word >> 6) & ((1u64 << ST_WIDTH) - 1);
        let lda_bits = (word >> 27) & ((1u64 << LDA_WIDTH) - 1);
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    // I48_LDA_MV: {lda[20:0], mv[21:0], 0b01, 0b101} - bits 4:3 = 01
    let bits_4_3 = ((word >> 3) & 0x3) as u8;
    if bits_4_3 == 0b01 {
        let mv_bits = (word >> 5) & ((1u64 << MV_WIDTH) - 1);
        let lda_bits = (word >> 27) & ((1u64 << LDA_WIDTH) - 1);
        bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    // I48_LDA_ALU: {lda[20:0], alu[19:0], 0b0011, 0b101} - bits 6:3 = 0011
    if bits_6_3 == 0b0011 {
        let alu_bits = (word >> 7) & ((1u64 << ALU_WIDTH) - 1);
        let lda_bits = (word >> 27) & ((1u64 << LDA_WIDTH) - 1);
        bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
    }

    bundle
}

/// Extract slot data from a 64-bit (8-byte) bundle.
pub fn extract_64bit(bytes: &[u8]) -> ExtractedBundle {
    let mut bundle = ExtractedBundle::new();

    if bytes.len() < 8 {
        return bundle;
    }

    // Read 64 bits as u64 (little-endian)
    let word = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);

    // Verify format marker (low 4 bits should be 0b0011 = 3)
    if (word & 0xF) != 0b0011 {
        return bundle;
    }

    // Extract bits for pattern matching
    let bits_6_4 = ((word >> 4) & 0x7) as u8;
    let bits_2_0_after_marker = ((word >> 4) & 0x7) as u8; // bits 6:4

    // I64_ALU_MV: bits 6:4 = 100 (0b100)
    // I64_NOP_LNG: bits 6:4 = 100, but with lng slot
    if bits_6_4 == 0b100 {
        // Check if it's ALU+MV or NOP+LNG by checking the alumv/lng discriminator
        // alu_mv = {alumv, 0b1} for ALU_MV, alu_mv = {lng, 0b0} for NOP_LNG
        let alu_mv_lsb = (word >> 10) & 1; // bit 10 determines alumv vs lng

        if alu_mv_lsb == 1 {
            // I64_ALU_MV: {0b0000000000, alumv[41:0], 0b1, 0b0000100, 0b0011}
            let mv_bits = (word >> 11) & ((1u64 << MV_WIDTH) - 1);
            let alu_bits = (word >> 33) & ((1u64 << ALU_WIDTH) - 1);
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        } else {
            // I64_NOP_LNG: {nop, 0b000000000, lng[41:0], 0b0, 0b0000100, 0b0011}
            let lng_bits = (word >> 11) & ((1u64 << LNG_WIDTH) - 1);
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        }
        return bundle;
    }

    // I64_ALU_VEC: {0b0000000000, alu[19:0], 0b01, vec[25:0], 0b10, 0b0011}
    // bits 5:4 = 10 for VEC combos
    if (word >> 4) & 0x3 == 0b10 {
        let vec_bits = (word >> 6) & ((1u64 << VEC_WIDTH) - 1);
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        // Check what's paired with vec
        let next_bits = (word >> 32) & 0x3;
        if next_bits == 0b01 {
            // ALU_VEC
            let alu_bits = (word >> 34) & ((1u64 << ALU_WIDTH) - 1);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        }
        // Could also be LDA_VEC, LDB_VEC, ST_VEC, MV_VEC
        return bundle;
    }

    // I64_LDA_LDB_ST: {lda[20:0], ldb[15:0], st[20:0], 0b11, 0b0011}
    if (word >> 4) & 0x3 == 0b11 {
        let st_bits = (word >> 6) & ((1u64 << ST_WIDTH) - 1);
        let ldb_bits = (word >> 27) & ((1u64 << LDB_WIDTH) - 1);
        let lda_bits = (word >> 43) & ((1u64 << LDA_WIDTH) - 1);
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    // I64_LDA_LDB_ALU: {lda[20:0], ldb[15:0], alu[19:0], 0b001, 0b0011}
    if bits_6_4 == 0b001 {
        let alu_bits = (word >> 7) & ((1u64 << ALU_WIDTH) - 1);
        let ldb_bits = (word >> 27) & ((1u64 << LDB_WIDTH) - 1);
        let lda_bits = (word >> 43) & ((1u64 << LDA_WIDTH) - 1);
        bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    // I64_ST_LDB_ALU: {st[20:0], ldb[15:0], alu[19:0], 0b101, 0b0011}
    if bits_6_4 == 0b101 {
        let alu_bits = (word >> 7) & ((1u64 << ALU_WIDTH) - 1);
        let ldb_bits = (word >> 27) & ((1u64 << LDB_WIDTH) - 1);
        let st_bits = (word >> 43) & ((1u64 << ST_WIDTH) - 1);
        bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        return bundle;
    }

    bundle
}

/// Extract slot data from an 80-bit (10-byte) bundle.
///
/// # Format Variants (from AIE2CompositeFormats.td)
///
/// 80-bit bundles have marker 0xB (0b1011) in low 4 bits and 76 bits of data.
/// Common patterns include:
///
/// - I80_LDA_ST_VEC: `{lda, st, 0b001001, vec, 0b00}`
/// - I80_LDA_MV_VEC: `{lda, mv, 0b00010, vec, 0b00}`
/// - I80_ALU_MV_VEC: `{0b00000, {alumv, 0b1}, vec, 0b01}` where alumv = {alu, mv}
/// - I80_LNG_VEC: `{0b00000, {lng, 0b0}, vec, 0b01}`
/// - I80_LDA_ST_ALU: `{lda, st, 0b00000000000, alu, 0b111}`
/// - I80_LDA_ALU_MV: `{lda, 0b00000, {alumv, 0b1}, 0b0001011}`
/// - I80_LDA_LNG: `{lda, 0b00000, {lng, 0b0}, 0b0001011}`
pub fn extract_80bit(bytes: &[u8]) -> ExtractedBundle {
    let mut bundle = ExtractedBundle::new();

    if bytes.len() < 10 {
        return bundle;
    }

    // Read 80 bits (10 bytes) as u128 (little-endian)
    let mut word_bytes = [0u8; 16];
    word_bytes[..10].copy_from_slice(&bytes[..10]);
    let word = u128::from_le_bytes(word_bytes);

    // Verify format marker (low 4 bits should be 0b1011 = 0xB)
    if (word & 0xF) != 0xB {
        return bundle;
    }

    // Extract instr80 (76 bits after the marker)
    let instr80 = (word >> 4) & ((1u128 << 76) - 1);

    // Extract discriminator bits for pattern matching
    // bits [1:0] of instr80 determine several patterns
    let bits_1_0 = (instr80 & 0x3) as u8;
    // bits [7:0] for more specific patterns
    let bits_7_0 = (instr80 & 0xFF) as u8;
    // bits at higher positions
    let bits_75_71 = ((instr80 >> 71) & 0x1F) as u8; // High 5 bits

    // Pattern: *_VEC with discriminator 0b00 at bits [1:0]
    if bits_1_0 == 0b00 {
        // VEC is at bits [27:2]
        let vec_bits = ((instr80 >> 2) & ((1u128 << VEC_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        // Check bits [33:28] for sub-patterns
        let bits_33_28 = ((instr80 >> 28) & 0x3F) as u8;

        if bits_33_28 == 0b001001 {
            // I80_LDA_ST_VEC: st at [54:34], lda at [75:55]
            let st_bits = ((instr80 >> 34) & ((1u128 << ST_WIDTH) - 1)) as u64;
            let lda_bits = ((instr80 >> 55) & ((1u128 << LDA_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
            bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        } else {
            // Check bits [32:28] for other patterns
            let bits_32_28 = ((instr80 >> 28) & 0x1F) as u8;

            if bits_32_28 == 0b00010 {
                // I80_LDA_MV_VEC: mv at [54:33], lda at [75:55]
                let mv_bits = ((instr80 >> 33) & ((1u128 << MV_WIDTH) - 1)) as u64;
                let lda_bits = ((instr80 >> 55) & ((1u128 << LDA_WIDTH) - 1)) as u64;
                bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
                bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
            } else if bits_32_28 == 0b00100 {
                // I80_ST_MV_VEC: mv at [54:33], st at [75:55]
                let mv_bits = ((instr80 >> 33) & ((1u128 << MV_WIDTH) - 1)) as u64;
                let st_bits = ((instr80 >> 55) & ((1u128 << ST_WIDTH) - 1)) as u64;
                bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
                bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
            } else if bits_32_28 == 0b00110 {
                // I80_LDB_MV_VEC: ldb at [48:33], 0b00000 at [75:71], mv at [32:11] (need recalc)
                // Actually: {0b00000, ldb, mv, 0b00110, vec, 0b00}
                // mv at [54:33], ldb at [70:55]
                let mv_bits = ((instr80 >> 33) & ((1u128 << MV_WIDTH) - 1)) as u64;
                let ldb_bits = ((instr80 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
                bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
                bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            } else if bits_75_71 == 0b00000 {
                // Patterns starting with 0b00000 at high bits
                // I80_ST_LDB_VEC: {st, ldb, 0b00000, 0b000001, vec, 0b00}
                if bits_33_28 == 0b000001 {
                    let ldb_bits = ((instr80 >> 39) & ((1u128 << LDB_WIDTH) - 1)) as u64;
                    let st_bits = ((instr80 >> 55) & ((1u128 << ST_WIDTH) - 1)) as u64;
                    bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
                    bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
                }
            } else if ((instr80 >> 28) & 0x7FF) == 0b00000011101 {
                // I80_LDA_LDB_VEC: {lda, ldb, 0b00000011101, vec, 0b00}
                let ldb_bits = ((instr80 >> 39) & ((1u128 << LDB_WIDTH) - 1)) as u64;
                let lda_bits = ((instr80 >> 55) & ((1u128 << LDA_WIDTH) - 1)) as u64;
                bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
                bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
            }
        }
        return bundle;
    }

    // Pattern: *_VEC with discriminator 0b01 at bits [1:0] (ALU_MV_VEC or LNG_VEC)
    if bits_1_0 == 0b01 {
        // VEC is at bits [27:2]
        let vec_bits = ((instr80 >> 2) & ((1u128 << VEC_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        // alu_mv at bits [70:28], discriminator 0b00000 at bits [75:71]
        // alu_mv LSB (bit 28) determines ALU+MV (1) vs LNG (0)
        let alu_mv_lsb = ((instr80 >> 28) & 1) as u8;

        if alu_mv_lsb == 1 {
            // I80_ALU_MV_VEC: alumv at [69:28], alumv = {alu, mv}
            // mv at bits [49:28] (after removing LSB)
            // alu at bits [69:50]
            let mv_bits = ((instr80 >> 29) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr80 >> 51) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        } else {
            // I80_LNG_VEC: lng at [69:28]
            let lng_bits = ((instr80 >> 29) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        }
        return bundle;
    }

    // Pattern: *_VEC with discriminator 0b10 at bits [1:0]
    if bits_1_0 == 0b10 {
        // VEC is at bits [27:2]
        let vec_bits = ((instr80 >> 2) & ((1u128 << VEC_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        // Check bits [29:28] for sub-patterns
        let bits_29_28 = ((instr80 >> 28) & 0x3) as u8;

        if bits_29_28 == 0b01 {
            // I80_ST_ALU_VEC: {st, 0b00000, alu, 0b01, vec, 0b10}
            let alu_bits = ((instr80 >> 30) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            let st_bits = ((instr80 >> 55) & ((1u128 << ST_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        } else if bits_29_28 == 0b10 {
            // I80_LDB_ALU_VEC: {0b00000, ldb, 0b00000, alu, 0b10, vec, 0b10}
            let alu_bits = ((instr80 >> 30) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            let ldb_bits = ((instr80 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        } else if bits_29_28 == 0b00 {
            // I80_LDA_ALU_VEC: {lda, 0b00000, alu, 0b00, vec, 0b10}
            let alu_bits = ((instr80 >> 30) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            let lda_bits = ((instr80 >> 55) & ((1u128 << LDA_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
            bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        }
        return bundle;
    }

    // Pattern: Non-VEC patterns with discriminator 0b111 at bits [2:0]
    if (instr80 & 0x7) == 0b111 {
        // I80_LDA_ST_ALU: {lda, st, 0b00000000000, alu, 0b111}
        let alu_bits = ((instr80 >> 3) & ((1u128 << ALU_WIDTH) - 1)) as u64;
        let st_bits = ((instr80 >> 34) & ((1u128 << ST_WIDTH) - 1)) as u64;
        let lda_bits = ((instr80 >> 55) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    // Pattern: *_ALU_MV or *_LNG patterns
    if bits_7_0 == 0b0001011 {
        // I80_LDA_ALU_MV or I80_LDA_LNG
        // alu_mv at [63:21], discriminator at bits [20:14] = 0b0001011
        let alu_mv_lsb = ((instr80 >> 21) & 1) as u8;
        let lda_bits = ((instr80 >> 55) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);

        if alu_mv_lsb == 1 {
            // I80_LDA_ALU_MV: alumv = {alu, mv}
            let mv_bits = ((instr80 >> 22) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr80 >> 44) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        } else {
            // I80_LDA_LNG
            let lng_bits = ((instr80 >> 22) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        }
        return bundle;
    }

    if bits_7_0 == 0b0000011 {
        // I80_LDB_ALU_MV or I80_LDB_LNG
        // {0b00000, ldb, 0b00000, alu_mv, 0b0000011}
        let alu_mv_lsb = ((instr80 >> 14) & 1) as u8;
        let ldb_bits = ((instr80 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);

        if alu_mv_lsb == 1 {
            // I80_LDB_ALU_MV
            let mv_bits = ((instr80 >> 15) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr80 >> 37) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        } else {
            // I80_LDB_LNG
            let lng_bits = ((instr80 >> 15) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        }
        return bundle;
    }

    if bits_7_0 == 0b0010011 {
        // I80_ST_ALU_MV or I80_ST_LNG
        // {st, 0b00000, alu_mv, 0b0010011}
        let alu_mv_lsb = ((instr80 >> 14) & 1) as u8;
        let st_bits = ((instr80 >> 55) & ((1u128 << ST_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);

        if alu_mv_lsb == 1 {
            // I80_ST_ALU_MV
            let mv_bits = ((instr80 >> 15) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr80 >> 37) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        } else {
            // I80_ST_LNG
            let lng_bits = ((instr80 >> 15) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        }
        return bundle;
    }

    // More non-VEC patterns with specific discriminators
    if bits_7_0 == 0b01101011 {
        // I80_ST_LDB_MV: {st, ldb, 0b000000000, mv, 0b01101011}
        let mv_bits = ((instr80 >> 8) & ((1u128 << MV_WIDTH) - 1)) as u64;
        let ldb_bits = ((instr80 >> 39) & ((1u128 << LDB_WIDTH) - 1)) as u64;
        let st_bits = ((instr80 >> 55) & ((1u128 << ST_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        return bundle;
    }

    if bits_7_0 == 0b00101011 {
        // I80_LDA_ST_MV: {lda, st, 0b0000, mv, 0b00101011}
        let mv_bits = ((instr80 >> 8) & ((1u128 << MV_WIDTH) - 1)) as u64;
        let st_bits = ((instr80 >> 34) & ((1u128 << ST_WIDTH) - 1)) as u64;
        let lda_bits = ((instr80 >> 55) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    if bits_7_0 == 0b11101011 {
        // I80_LDA_LDB_MV: {lda, ldb, 0b000000000, mv, 0b11101011}
        let mv_bits = ((instr80 >> 8) & ((1u128 << MV_WIDTH) - 1)) as u64;
        let ldb_bits = ((instr80 >> 39) & ((1u128 << LDB_WIDTH) - 1)) as u64;
        let lda_bits = ((instr80 >> 55) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    bundle
}

/// Extract slot data from a 96-bit (12-byte) bundle.
///
/// 96-bit bundles have marker 0x7 (0b0111) in low 4 bits and 92 bits of data.
pub fn extract_96bit(bytes: &[u8]) -> ExtractedBundle {
    let mut bundle = ExtractedBundle::new();

    if bytes.len() < 12 {
        return bundle;
    }

    // Read 96 bits (12 bytes) as u128 (little-endian)
    let mut word_bytes = [0u8; 16];
    word_bytes[..12].copy_from_slice(&bytes[..12]);
    let word = u128::from_le_bytes(word_bytes);

    // Verify format marker (low 4 bits should be 0b0111 = 0x7)
    if (word & 0xF) != 0x7 {
        return bundle;
    }

    // Extract instr96 (92 bits after the marker)
    let instr96 = (word >> 4) & ((1u128 << 92) - 1);

    // Extract discriminator bits
    let bits_1_0 = (instr96 & 0x3) as u8;
    let bits_8_0 = (instr96 & 0x1FF) as u16;

    // Pattern: *_VEC with discriminator 0b01 at bits [1:0]
    if bits_1_0 == 0b01 {
        // VEC at bits [27:2]
        let vec_bits = ((instr96 >> 2) & ((1u128 << VEC_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        // Check alu_mv discriminator at bit 28
        let alu_mv_lsb = ((instr96 >> 28) & 1) as u8;

        if alu_mv_lsb == 1 {
            // I96_ST_ALU_MV_VEC: {st, alumv, 0b1, vec, 0b01}
            let mv_bits = ((instr96 >> 29) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr96 >> 51) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            let st_bits = ((instr96 >> 71) & ((1u128 << ST_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        } else {
            // I96_ST_LNG_VEC: {st, lng, 0b0, vec, 0b01}
            let lng_bits = ((instr96 >> 29) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            let st_bits = ((instr96 >> 71) & ((1u128 << ST_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        }
        return bundle;
    }

    // Pattern: *_VEC with discriminator 0b10 at bits [1:0]
    if bits_1_0 == 0b10 {
        // VEC at bits [27:2]
        let vec_bits = ((instr96 >> 2) & ((1u128 << VEC_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        // Check bits [32:28] for sub-patterns
        let bits_32_28 = ((instr96 >> 28) & 0x1F) as u8;

        if bits_32_28 == 0b00110 {
            // I96_ST_LDB_MV_VEC: {st, ldb, mv, 0b00110, vec, 0b10}
            let mv_bits = ((instr96 >> 33) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let ldb_bits = ((instr96 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            let st_bits = ((instr96 >> 71) & ((1u128 << ST_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        } else if bits_32_28 == 0b01010 {
            // I96_LDB_ALU_MV_VEC or I96_LDB_LNG_VEC
            let alu_mv_lsb = ((instr96 >> 33) & 1) as u8;
            let ldb_bits = ((instr96 >> 76) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);

            if alu_mv_lsb == 1 {
                let mv_bits = ((instr96 >> 34) & ((1u128 << MV_WIDTH) - 1)) as u64;
                let alu_bits = ((instr96 >> 56) & ((1u128 << ALU_WIDTH) - 1)) as u64;
                bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
                bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
            } else {
                let lng_bits = ((instr96 >> 34) & ((1u128 << LNG_WIDTH) - 1)) as u64;
                bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
            }
        } else if bits_32_28 == 0b00010 {
            // I96_LDA_LDB_MV_VEC: {lda, ldb, mv, 0b00010, vec, 0b10}
            let mv_bits = ((instr96 >> 33) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let ldb_bits = ((instr96 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            let lda_bits = ((instr96 >> 71) & ((1u128 << LDA_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        } else if bits_32_28 == 0b01110 {
            // I96_LDA_LDB_ST_VEC: {lda, ldb, st, 0b001110, vec, 0b10}
            let st_bits = ((instr96 >> 34) & ((1u128 << ST_WIDTH) - 1)) as u64;
            let ldb_bits = ((instr96 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            let lda_bits = ((instr96 >> 71) & ((1u128 << LDA_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        } else {
            // Check for I96_LDA_ST_ALU_VEC: {lda, st, alu, 0b00, vec, 0b10}
            let bits_29_28 = ((instr96 >> 28) & 0x3) as u8;
            if bits_29_28 == 0b00 {
                let alu_bits = ((instr96 >> 30) & ((1u128 << ALU_WIDTH) - 1)) as u64;
                let st_bits = ((instr96 >> 50) & ((1u128 << ST_WIDTH) - 1)) as u64;
                let lda_bits = ((instr96 >> 71) & ((1u128 << LDA_WIDTH) - 1)) as u64;
                bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
                bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
                bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
            } else if bits_29_28 == 0b11 {
                // I96_LDA_LDB_ALU_VEC: {lda, ldb, 0b00000, alu, 0b11, vec, 0b10}
                let alu_bits = ((instr96 >> 30) & ((1u128 << ALU_WIDTH) - 1)) as u64;
                let ldb_bits = ((instr96 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
                let lda_bits = ((instr96 >> 71) & ((1u128 << LDA_WIDTH) - 1)) as u64;
                bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
                bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
                bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
            }
        }
        return bundle;
    }

    // Pattern: *_VEC with discriminator 0b00 at bits [1:0]
    if bits_1_0 == 0b00 {
        // I96_LDA_ALU_MV_VEC or I96_LDA_LNG_VEC
        let vec_bits = ((instr96 >> 2) & ((1u128 << VEC_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        let alu_mv_lsb = ((instr96 >> 28) & 1) as u8;
        let lda_bits = ((instr96 >> 71) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);

        if alu_mv_lsb == 1 {
            let mv_bits = ((instr96 >> 29) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr96 >> 51) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        } else {
            let lng_bits = ((instr96 >> 29) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        }
        return bundle;
    }

    // Non-VEC patterns
    if bits_8_0 == 0b000010011 {
        // I96_LDA_LDB_ALU_ST: {lda, ldb, 0b00000, alu, st, 0b000010011}
        let st_bits = ((instr96 >> 9) & ((1u128 << ST_WIDTH) - 1)) as u64;
        let alu_bits = ((instr96 >> 30) & ((1u128 << ALU_WIDTH) - 1)) as u64;
        let ldb_bits = ((instr96 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
        let lda_bits = ((instr96 >> 71) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    if (instr96 & 0xFF) == 0b00001111 {
        // I96_LDA_LDB_ST_MV: {lda, ldb, st, 0b0000, mv, 0b00001111}
        let mv_bits = ((instr96 >> 8) & ((1u128 << MV_WIDTH) - 1)) as u64;
        let st_bits = ((instr96 >> 34) & ((1u128 << ST_WIDTH) - 1)) as u64;
        let ldb_bits = ((instr96 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
        let lda_bits = ((instr96 >> 71) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        return bundle;
    }

    if (instr96 & 0x7F) == 0b0000111 {
        // I96_LDA_ST_ALU_MV or I96_LDA_ST_LNG
        let alu_mv_lsb = ((instr96 >> 7) & 1) as u8;
        let st_bits = ((instr96 >> 50) & ((1u128 << ST_WIDTH) - 1)) as u64;
        let lda_bits = ((instr96 >> 71) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);

        if alu_mv_lsb == 1 {
            let mv_bits = ((instr96 >> 8) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr96 >> 30) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        } else {
            let lng_bits = ((instr96 >> 8) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        }
        return bundle;
    }

    if (instr96 & 0x7F) == 0b0001011 {
        // I96_LDA_LDB_ALU_MV or I96_LDA_LDB_LNG
        let alu_mv_lsb = ((instr96 >> 7) & 1) as u8;
        let ldb_bits = ((instr96 >> 55) & ((1u128 << LDB_WIDTH) - 1)) as u64;
        let lda_bits = ((instr96 >> 71) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);

        if alu_mv_lsb == 1 {
            let mv_bits = ((instr96 >> 8) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr96 >> 30) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        } else {
            let lng_bits = ((instr96 >> 8) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        }
        return bundle;
    }

    bundle
}

/// Extract slot data from a 112-bit (14-byte) bundle.
///
/// 112-bit bundles have marker 0xF (0b1111) in low 4 bits and 108 bits of data.
pub fn extract_112bit(bytes: &[u8]) -> ExtractedBundle {
    let mut bundle = ExtractedBundle::new();

    if bytes.len() < 14 {
        return bundle;
    }

    // Read 112 bits (14 bytes) as u128 (little-endian)
    let mut word_bytes = [0u8; 16];
    word_bytes[..14].copy_from_slice(&bytes[..14]);
    let word = u128::from_le_bytes(word_bytes);

    // Verify format marker (low 4 bits should be 0b1111 = 0xF)
    if (word & 0xF) != 0xF {
        return bundle;
    }

    // Extract instr112 (108 bits after the marker)
    let instr112 = (word >> 4) & ((1u128 << 108) - 1);

    // Extract discriminator bits
    let bits_1_0 = (instr112 & 0x3) as u8;

    // Pattern: *_VEC with discriminator 0b01 at bits [1:0]
    if bits_1_0 == 0b01 {
        // VEC at bits [27:2]
        let vec_bits = ((instr112 >> 2) & ((1u128 << VEC_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        // I112_ST_LDB_ALU_MV_VEC or I112_ST_LDB_LNG_VEC
        let alu_mv_lsb = ((instr112 >> 28) & 1) as u8;

        if alu_mv_lsb == 1 {
            let mv_bits = ((instr112 >> 29) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr112 >> 51) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            let ldb_bits = ((instr112 >> 71) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            let st_bits = ((instr112 >> 87) & ((1u128 << ST_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        } else {
            let lng_bits = ((instr112 >> 29) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            let ldb_bits = ((instr112 >> 71) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            let st_bits = ((instr112 >> 87) & ((1u128 << ST_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        }
        return bundle;
    }

    // Pattern: *_VEC with discriminator 0b10 at bits [1:0]
    if bits_1_0 == 0b10 {
        // VEC at bits [27:2]
        let vec_bits = ((instr112 >> 2) & ((1u128 << VEC_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        // Check bit 28 for further discrimination
        let bit_28 = ((instr112 >> 28) & 1) as u8;

        if bit_28 == 0 {
            // I112_LDA_LDB_ALU_ST_VEC: {lda, ldb, alu, 0b00, st, vec, 0b10}
            let st_bits = ((instr112 >> 29) & ((1u128 << ST_WIDTH) - 1)) as u64;
            let alu_bits = ((instr112 >> 52) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            let ldb_bits = ((instr112 >> 72) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            let lda_bits = ((instr112 >> 88) & ((1u128 << LDA_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        } else {
            // I112_LDA_ST_MV_VEC: {lda, st, 0b0000000000000001, mv, vec, 0b10}
            let mv_bits = ((instr112 >> 29) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let st_bits = ((instr112 >> 67) & ((1u128 << ST_WIDTH) - 1)) as u64;
            let lda_bits = ((instr112 >> 88) & ((1u128 << LDA_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
            bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        }
        return bundle;
    }

    // Pattern: *_VEC with discriminator 0b00 at bits [1:0]
    if bits_1_0 == 0b00 {
        // VEC at bits [27:2]
        let vec_bits = ((instr112 >> 2) & ((1u128 << VEC_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

        // I112_LDA_LDB_ALU_MV_VEC or I112_LDA_LDB_LNG_VEC
        let alu_mv_lsb = ((instr112 >> 28) & 1) as u8;

        if alu_mv_lsb == 1 {
            let mv_bits = ((instr112 >> 29) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr112 >> 51) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            let ldb_bits = ((instr112 >> 71) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            let lda_bits = ((instr112 >> 87) & ((1u128 << LDA_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        } else {
            let lng_bits = ((instr112 >> 29) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            let ldb_bits = ((instr112 >> 71) & ((1u128 << LDB_WIDTH) - 1)) as u64;
            let lda_bits = ((instr112 >> 87) & ((1u128 << LDA_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
            bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
            bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);
        }
        return bundle;
    }

    // Non-VEC patterns with discriminator 0b0000111 at bits [6:0]
    if (instr112 & 0x7F) == 0b0000111 {
        // I112_LDA_LDB_ALU_MV_ST or I112_LDA_LDB_LNG_ST
        let alu_mv_lsb = ((instr112 >> 7) & 1) as u8;
        let st_bits = ((instr112 >> 8) & ((1u128 << ST_WIDTH) - 1)) as u64;
        let ldb_bits = ((instr112 >> 71) & ((1u128 << LDB_WIDTH) - 1)) as u64;
        let lda_bits = ((instr112 >> 87) & ((1u128 << LDA_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);
        bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);
        bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);

        if alu_mv_lsb == 1 {
            let mv_bits = ((instr112 >> 29) & ((1u128 << MV_WIDTH) - 1)) as u64;
            let alu_bits = ((instr112 >> 51) & ((1u128 << ALU_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);
            bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
        } else {
            let lng_bits = ((instr112 >> 29) & ((1u128 << LNG_WIDTH) - 1)) as u64;
            bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
        }
        return bundle;
    }

    bundle
}

/// Extract slot data from a 128-bit (16-byte) bundle.
///
/// 128-bit bundles have marker 0 at bit 0 (LSB = 0) and 127 bits of data.
///
/// # Format Variants (from AIE2CompositeFormats.td)
///
/// Both variants have the same overall structure:
/// `{ldb, lda, st, alu_mv, vec}` where alu_mv is 43 bits.
///
/// The alu_mv LSB (bit 27 of Inst) distinguishes the two variants:
///
/// - I128_LDB_LDA_ST_LNG_VEC: `alu_mv = {lng, 0b0}` - bit 27 = 0
/// - I128_LDB_LDA_ST_ALU_MV_VEC: `alu_mv = {alumv, 0b1}` where `alumv = {alu, mv}` - bit 27 = 1
///
/// Bit positions in Inst:
/// - Inst[0] = 0 (marker)
/// - vec[25:0] at Inst[26:1]
/// - alu_mv LSB at Inst[27]
/// - For ALU+MV: mv[21:0] at Inst[49:28], alu[19:0] at Inst[69:50]
/// - For LNG: lng[41:0] at Inst[69:28]
/// - st[20:0] at Inst[90:70]
/// - lda[20:0] at Inst[111:91]
/// - ldb[15:0] at Inst[127:112]
pub fn extract_128bit(bytes: &[u8]) -> ExtractedBundle {
    let mut bundle = ExtractedBundle::new();

    if bytes.len() < 16 {
        return bundle;
    }

    // Read 128 bits (16 bytes) as u128 (little-endian)
    let word = u128::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11],
        bytes[12], bytes[13], bytes[14], bytes[15],
    ]);

    // Verify format marker (bit 0 should be 0)
    if (word & 0x1) != 0 {
        return bundle;
    }

    // Extract vec slot: bits [26:1] (26 bits)
    let vec_bits = ((word >> 1) & ((1u128 << VEC_WIDTH) - 1)) as u64;
    bundle.add_slot(SlotType::Vec, vec_bits, VEC_WIDTH);

    // Check alu_mv LSB at bit 27 to determine variant
    let alu_mv_lsb = ((word >> 27) & 1) as u8;

    if alu_mv_lsb == 1 {
        // I128_LDB_LDA_ST_ALU_MV_VEC: alumv = {alu, mv}
        // mv[21:0] at bits [49:28]
        let mv_bits = ((word >> 28) & ((1u128 << MV_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Mv, mv_bits, MV_WIDTH);

        // alu[19:0] at bits [69:50]
        let alu_bits = ((word >> 50) & ((1u128 << ALU_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Alu, alu_bits, ALU_WIDTH);
    } else {
        // I128_LDB_LDA_ST_LNG_VEC: alu_mv = {lng, 0b0}
        // lng[41:0] at bits [69:28]
        let lng_bits = ((word >> 28) & ((1u128 << LNG_WIDTH) - 1)) as u64;
        bundle.add_slot(SlotType::Lng, lng_bits, LNG_WIDTH);
    }

    // st[20:0] at bits [90:70]
    let st_bits = ((word >> 70) & ((1u128 << ST_WIDTH) - 1)) as u64;
    bundle.add_slot(SlotType::St, st_bits, ST_WIDTH);

    // lda[20:0] at bits [111:91]
    let lda_bits = ((word >> 91) & ((1u128 << LDA_WIDTH) - 1)) as u64;
    bundle.add_slot(SlotType::Lda, lda_bits, LDA_WIDTH);

    // ldb[15:0] at bits [127:112]
    let ldb_bits = ((word >> 112) & ((1u128 << LDB_WIDTH) - 1)) as u64;
    bundle.add_slot(SlotType::Ldb, ldb_bits, LDB_WIDTH);

    bundle
}

/// Extract slot data from any bundle based on its format.
pub fn extract_slots(bytes: &[u8]) -> ExtractedBundle {
    if bytes.len() < 2 {
        return ExtractedBundle::new();
    }

    let format = super::detect_format(bytes);

    match format {
        BundleFormat::Nop16 => {
            let mut bundle = ExtractedBundle::new();
            bundle.add_slot(SlotType::Nop, 0, 1);
            bundle
        }
        BundleFormat::Short32 => {
            if bytes.len() >= 4 {
                let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                extract_32bit(word)
            } else {
                ExtractedBundle::new()
            }
        }
        BundleFormat::Medium48 => extract_48bit(bytes),
        BundleFormat::Medium64 => extract_64bit(bytes),
        BundleFormat::Long80 => extract_80bit(bytes),
        BundleFormat::Long96 => extract_96bit(bytes),
        BundleFormat::Long112 => extract_112bit(bytes),
        BundleFormat::Full128 => extract_128bit(bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_32bit_alu() {
        // I32_ALU: {0b00010, alu[19:0], 0b001, 0b1001}
        // Construct a test word with alu = 0x12345 (20 bits)
        let alu_value: u32 = 0x12345;
        let word: u32 = (0b00010 << 27) | (alu_value << 7) | (0b001 << 4) | 0b1001;

        let bundle = extract_32bit(word);
        assert_eq!(bundle.slots.len(), 1);
        assert_eq!(bundle.slots[0].slot_type, SlotType::Alu);
        assert_eq!(bundle.slots[0].bits, alu_value as u64);
        assert_eq!(bundle.slots[0].width, ALU_WIDTH);
    }

    #[test]
    fn test_extract_32bit_lda() {
        // I32_LDA: {0b00000, lda[20:0], 0b01, 0b1001}
        let lda_value: u32 = 0x1ABCD; // 21 bits
        let word: u32 = (0b00000 << 27) | (lda_value << 6) | (0b01 << 4) | 0b1001;

        let bundle = extract_32bit(word);
        assert_eq!(bundle.slots.len(), 1);
        assert_eq!(bundle.slots[0].slot_type, SlotType::Lda);
        assert_eq!(bundle.slots[0].bits, lda_value as u64);
    }

    #[test]
    fn test_extract_32bit_vec() {
        // I32_VEC: {vec[25:0], 0b00, 0b1001}
        // Note: high5 will be part of vec for VEC format
        let vec_value: u32 = 0x2345678; // 26 bits
        let word: u32 = (vec_value << 6) | (0b00 << 4) | 0b1001;

        let bundle = extract_32bit(word);
        assert_eq!(bundle.slots.len(), 1);
        assert_eq!(bundle.slots[0].slot_type, SlotType::Vec);
        assert_eq!(bundle.slots[0].bits, vec_value as u64);
    }

    #[test]
    fn test_extract_48bit_lng() {
        // I48_LNG: {lng[41:0], 0b010, 0b101}
        let lng_value: u64 = 0x123456789AB; // 42 bits
        let word: u64 = (lng_value << 6) | (0b010 << 3) | 0b101;

        let bytes = word.to_le_bytes();
        let bundle = extract_48bit(&bytes[..6]);

        assert_eq!(bundle.slots.len(), 1);
        assert_eq!(bundle.slots[0].slot_type, SlotType::Lng);
        assert_eq!(bundle.slots[0].bits, lng_value);
    }

    #[test]
    fn test_extract_48bit_lda_st() {
        // I48_LDA_ST: {lda[20:0], st[20:0], 0b110, 0b101}
        // Use lda value with high bits that don't match LDB patterns (0b00000, 0b00001, 0b00010)
        // lda = 0x0E5555 has bits 20:16 = 0xE = 0b01110
        let st_value: u64 = 0x1ABCD; // 21 bits
        let lda_value: u64 = 0x0E5555; // 20 bits, high5 = 0b01110
        let word: u64 = (lda_value << 27) | (st_value << 6) | (0b110 << 3) | 0b101;

        let bytes = word.to_le_bytes();
        let bundle = extract_48bit(&bytes[..6]);

        assert_eq!(bundle.slots.len(), 2, "Expected 2 slots, got: {:?}", bundle.slots);

        let st_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::St);
        let lda_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Lda);

        assert!(st_slot.is_some(), "Expected St slot");
        assert!(lda_slot.is_some(), "Expected Lda slot");
        assert_eq!(st_slot.unwrap().bits, st_value);
        assert_eq!(lda_slot.unwrap().bits, lda_value);
    }

    #[test]
    fn test_extract_slots_nop16() {
        // 16-bit NOP: marker 0x1
        let bytes = [0x01, 0x00];
        let bundle = extract_slots(&bytes);

        assert_eq!(bundle.slots.len(), 1);
        assert_eq!(bundle.slots[0].slot_type, SlotType::Nop);
    }

    #[test]
    fn test_extract_slots_detects_format() {
        // 32-bit with ALU slot
        let alu_value: u32 = 0x12345;
        let word: u32 = (0b00010 << 27) | (alu_value << 7) | (0b001 << 4) | 0b1001;
        let bytes = word.to_le_bytes();

        let bundle = extract_slots(&bytes);
        assert_eq!(bundle.slots.len(), 1);
        assert_eq!(bundle.slots[0].slot_type, SlotType::Alu);
    }

    #[test]
    fn test_extract_128bit_alu_mv_vec() {
        // I128_LDB_LDA_ST_ALU_MV_VEC: {ldb, lda, st, {alu, mv}, 0b1, vec}
        // Bit positions:
        // - Inst[0] = 0 (marker)
        // - vec[25:0] at Inst[26:1]
        // - alu_mv LSB = 1 at Inst[27]
        // - mv[21:0] at Inst[49:28]
        // - alu[19:0] at Inst[69:50]
        // - st[20:0] at Inst[90:70]
        // - lda[20:0] at Inst[111:91]
        // - ldb[15:0] at Inst[127:112]

        let vec_value: u128 = 0x2ABCDEF;   // 26 bits
        let mv_value: u128 = 0x3ABCDE;     // 22 bits
        let alu_value: u128 = 0x89ABC;     // 20 bits
        let st_value: u128 = 0x1BCDEF;     // 21 bits
        let lda_value: u128 = 0x1CDEF0;    // 21 bits
        let ldb_value: u128 = 0xABCD;      // 16 bits

        // Construct the 128-bit word
        let word: u128 = 0                           // bit 0 = 0 (marker)
            | (vec_value << 1)                       // bits [26:1]
            | (1u128 << 27)                          // bit 27 = 1 (ALU+MV variant)
            | (mv_value << 28)                       // bits [49:28]
            | (alu_value << 50)                      // bits [69:50]
            | (st_value << 70)                       // bits [90:70]
            | (lda_value << 91)                      // bits [111:91]
            | (ldb_value << 112);                    // bits [127:112]

        let bytes = word.to_le_bytes();
        let bundle = extract_128bit(&bytes);

        // Should have 6 slots: vec, mv, alu, st, lda, ldb
        assert_eq!(bundle.slots.len(), 6, "Expected 6 slots, got: {:?}", bundle.slots);

        let vec_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Vec);
        let mv_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Mv);
        let alu_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Alu);
        let st_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::St);
        let lda_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Lda);
        let ldb_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Ldb);

        assert!(vec_slot.is_some(), "Expected Vec slot");
        assert!(mv_slot.is_some(), "Expected Mv slot");
        assert!(alu_slot.is_some(), "Expected Alu slot");
        assert!(st_slot.is_some(), "Expected St slot");
        assert!(lda_slot.is_some(), "Expected Lda slot");
        assert!(ldb_slot.is_some(), "Expected Ldb slot");

        assert_eq!(vec_slot.unwrap().bits, vec_value as u64);
        assert_eq!(mv_slot.unwrap().bits, mv_value as u64);
        assert_eq!(alu_slot.unwrap().bits, alu_value as u64);
        assert_eq!(st_slot.unwrap().bits, st_value as u64);
        assert_eq!(lda_slot.unwrap().bits, lda_value as u64);
        assert_eq!(ldb_slot.unwrap().bits, ldb_value as u64);
    }

    #[test]
    fn test_extract_128bit_lng_vec() {
        // I128_LDB_LDA_ST_LNG_VEC: {ldb, lda, st, {lng, 0b0}, vec}
        // Bit positions:
        // - Inst[0] = 0 (marker)
        // - vec[25:0] at Inst[26:1]
        // - alu_mv LSB = 0 at Inst[27]
        // - lng[41:0] at Inst[69:28]
        // - st[20:0] at Inst[90:70]
        // - lda[20:0] at Inst[111:91]
        // - ldb[15:0] at Inst[127:112]

        let vec_value: u128 = 0x1234567;       // 26 bits
        let lng_value: u128 = 0x2ABCDEF0123;   // 42 bits
        let st_value: u128 = 0x1ABCDE;         // 21 bits
        let lda_value: u128 = 0x1BCDEF;        // 21 bits
        let ldb_value: u128 = 0xCDEF;          // 16 bits

        // Construct the 128-bit word
        let word: u128 = 0                           // bit 0 = 0 (marker)
            | (vec_value << 1)                       // bits [26:1]
            | (0u128 << 27)                          // bit 27 = 0 (LNG variant)
            | (lng_value << 28)                      // bits [69:28]
            | (st_value << 70)                       // bits [90:70]
            | (lda_value << 91)                      // bits [111:91]
            | (ldb_value << 112);                    // bits [127:112]

        let bytes = word.to_le_bytes();
        let bundle = extract_128bit(&bytes);

        // Should have 5 slots: vec, lng, st, lda, ldb
        assert_eq!(bundle.slots.len(), 5, "Expected 5 slots, got: {:?}", bundle.slots);

        let vec_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Vec);
        let lng_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Lng);
        let st_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::St);
        let lda_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Lda);
        let ldb_slot = bundle.slots.iter().find(|s| s.slot_type == SlotType::Ldb);

        assert!(vec_slot.is_some(), "Expected Vec slot");
        assert!(lng_slot.is_some(), "Expected Lng slot");
        assert!(st_slot.is_some(), "Expected St slot");
        assert!(lda_slot.is_some(), "Expected Lda slot");
        assert!(ldb_slot.is_some(), "Expected Ldb slot");

        assert_eq!(vec_slot.unwrap().bits, vec_value as u64);
        assert_eq!(lng_slot.unwrap().bits, lng_value as u64);
        assert_eq!(st_slot.unwrap().bits, st_value as u64);
        assert_eq!(lda_slot.unwrap().bits, lda_value as u64);
        assert_eq!(ldb_slot.unwrap().bits, ldb_value as u64);
    }

    #[test]
    fn test_extract_slots_128bit() {
        // Test that extract_slots properly routes to extract_128bit
        // Use a 128-bit bundle with ALU+MV variant

        let vec_value: u128 = 0x1111111;
        let mv_value: u128 = 0x222222;
        let alu_value: u128 = 0x33333;
        let st_value: u128 = 0x144444;
        let lda_value: u128 = 0x155555;
        let ldb_value: u128 = 0x6666;

        let word: u128 = 0
            | (vec_value << 1)
            | (1u128 << 27)
            | (mv_value << 28)
            | (alu_value << 50)
            | (st_value << 70)
            | (lda_value << 91)
            | (ldb_value << 112);

        let bytes = word.to_le_bytes();
        let bundle = extract_slots(&bytes);

        // Should have 6 slots
        assert_eq!(bundle.slots.len(), 6);

        // Verify we got the expected slot types
        let slot_types: Vec<SlotType> = bundle.slots.iter().map(|s| s.slot_type).collect();
        assert!(slot_types.contains(&SlotType::Vec));
        assert!(slot_types.contains(&SlotType::Mv));
        assert!(slot_types.contains(&SlotType::Alu));
        assert!(slot_types.contains(&SlotType::St));
        assert!(slot_types.contains(&SlotType::Lda));
        assert!(slot_types.contains(&SlotType::Ldb));
    }
}
