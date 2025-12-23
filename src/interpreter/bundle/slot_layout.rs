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
        // TODO: Implement 80-128 bit extraction
        BundleFormat::Long80
        | BundleFormat::Long96
        | BundleFormat::Long112
        | BundleFormat::Full128 => {
            // For now, return empty - these need more complex handling
            ExtractedBundle::new()
        }
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
}
