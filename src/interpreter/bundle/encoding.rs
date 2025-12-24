//! VLIW bundle encoding and format detection.
//!
//! AIE2 instructions use variable-length VLIW bundles. The bundle size is
//! encoded in the low bits of the first word:
//!
//! | Format   | Low Bits | Size    | Marker   |
//! |----------|----------|---------|----------|
//! | 16-bit   | 0b0001   | 2 bytes | NOP slot |
//! | 32-bit   | 0b1001   | 4 bytes | Single   |
//! | 48-bit   | 0b0101   | 6 bytes | Long     |
//! | 64-bit   | 0b0011   | 8 bytes | 2 slots  |
//! | 80-bit   | 0b1011   | 10 bytes| 2+ slots |
//! | 96-bit   | 0b0111   | 12 bytes| 3 slots  |
//! | 112-bit  | 0b1111   | 14 bytes| 4+ slots |
//! | 128-bit  | 0b0000   | 16 bytes| Full     |
//!
//! The format marker is extracted from bits 3:0 of the first 16-bit halfword.

/// Bundle format - determines the instruction size and available slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BundleFormat {
    /// 16-bit NOP format.
    Nop16,
    /// 32-bit short format - single operation.
    #[default]
    Short32,
    /// 48-bit format - long slot or combined.
    Medium48,
    /// 64-bit format - two slots.
    Medium64,
    /// 80-bit format.
    Long80,
    /// 96-bit format.
    Long96,
    /// 112-bit format.
    Long112,
    /// 128-bit full VLIW format - all slots available.
    Full128,
}

impl BundleFormat {
    /// Format marker constants (low 4 bits of first halfword).
    pub const MARKER_NOP16: u8 = 0b0001;   // 0x1
    pub const MARKER_SHORT32: u8 = 0b1001; // 0x9
    pub const MARKER_MEDIUM48: u8 = 0b0101; // 0x5
    pub const MARKER_MEDIUM64: u8 = 0b0011; // 0x3
    pub const MARKER_LONG80: u8 = 0b1011;  // 0xB
    pub const MARKER_LONG96: u8 = 0b0111;  // 0x7
    pub const MARKER_LONG112: u8 = 0b1111; // 0xF
    pub const MARKER_FULL128: u8 = 0b0000; // 0x0

    /// Detect format from the low 4 bits of the first halfword.
    ///
    /// The 128-bit format only has a 1-bit marker (bit 0 = 0), so any even
    /// nibble (0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE) indicates 128-bit.
    /// All other formats have bit 0 = 1.
    #[inline]
    pub fn from_marker(marker: u8) -> Self {
        let nibble = marker & 0xF;

        // 128-bit format has only 1-bit marker (bit 0 = 0)
        // All other formats have bit 0 = 1
        if nibble & 1 == 0 {
            return BundleFormat::Full128;
        }

        match nibble {
            Self::MARKER_NOP16 => BundleFormat::Nop16,
            Self::MARKER_SHORT32 => BundleFormat::Short32,
            Self::MARKER_MEDIUM48 => BundleFormat::Medium48,
            Self::MARKER_MEDIUM64 => BundleFormat::Medium64,
            Self::MARKER_LONG80 => BundleFormat::Long80,
            Self::MARKER_LONG96 => BundleFormat::Long96,
            Self::MARKER_LONG112 => BundleFormat::Long112,
            // All odd values that don't match known markers default to 32-bit
            _ => BundleFormat::Short32,
        }
    }

    /// Get the format marker value.
    #[inline]
    pub const fn marker(self) -> u8 {
        match self {
            BundleFormat::Nop16 => Self::MARKER_NOP16,
            BundleFormat::Short32 => Self::MARKER_SHORT32,
            BundleFormat::Medium48 => Self::MARKER_MEDIUM48,
            BundleFormat::Medium64 => Self::MARKER_MEDIUM64,
            BundleFormat::Long80 => Self::MARKER_LONG80,
            BundleFormat::Long96 => Self::MARKER_LONG96,
            BundleFormat::Long112 => Self::MARKER_LONG112,
            BundleFormat::Full128 => Self::MARKER_FULL128,
        }
    }

    /// Get the size of this format in bytes.
    #[inline]
    pub const fn size_bytes(self) -> u8 {
        match self {
            BundleFormat::Nop16 => 2,
            BundleFormat::Short32 => 4,
            BundleFormat::Medium48 => 6,
            BundleFormat::Medium64 => 8,
            BundleFormat::Long80 => 10,
            BundleFormat::Long96 => 12,
            BundleFormat::Long112 => 14,
            BundleFormat::Full128 => 16,
        }
    }

    /// Get the size of this format in bits.
    #[inline]
    pub const fn size_bits(self) -> u16 {
        (self.size_bytes() as u16) * 8
    }

    /// Get the number of marker bits in this format.
    #[inline]
    pub const fn marker_bits(self) -> u8 {
        match self {
            BundleFormat::Nop16 => 4,    // 0b0001
            BundleFormat::Short32 => 4,  // 0b1001
            BundleFormat::Medium48 => 3, // 0b101
            BundleFormat::Medium64 => 4, // 0b0011
            BundleFormat::Long80 => 4,   // 0b1011
            BundleFormat::Long96 => 4,   // 0b0111
            BundleFormat::Long112 => 4,  // 0b1111
            BundleFormat::Full128 => 1,  // 0b0
        }
    }

    /// Check if this format supports the given slot count.
    #[inline]
    pub fn supports_slots(self, count: usize) -> bool {
        match self {
            BundleFormat::Nop16 => count <= 1,
            BundleFormat::Short32 => count <= 1,
            BundleFormat::Medium48 => count <= 2,
            BundleFormat::Medium64 => count <= 2,
            BundleFormat::Long80 => count <= 3,
            BundleFormat::Long96 => count <= 4,
            BundleFormat::Long112 => count <= 5,
            BundleFormat::Full128 => count <= 7,
        }
    }
}

/// Detect the bundle format from the first bytes.
///
/// The format is encoded in the low 4 bits of the first halfword.
/// We only need 2 bytes to detect the format.
///
/// # Arguments
///
/// * `bytes` - At least 2 bytes of instruction data
///
/// # Returns
///
/// The detected bundle format, or `Short32` if fewer than 2 bytes available.
pub fn detect_format(bytes: &[u8]) -> BundleFormat {
    if bytes.len() < 2 {
        return BundleFormat::Short32; // Not enough data, assume minimum
    }

    // Read first halfword (little-endian)
    let hw0 = u16::from_le_bytes([bytes[0], bytes[1]]);

    // Special case: all zeros is historically a 4-byte NOP
    // This takes precedence over the format marker interpretation
    if bytes.len() >= 4 {
        let word0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if word0 == 0 || word0 == NOP_ENCODINGS[1] {
            return BundleFormat::Short32;
        }
    } else if hw0 == 0 {
        return BundleFormat::Short32;
    }

    // Format marker is in the low 4 bits
    let marker = (hw0 & 0xF) as u8;

    BundleFormat::from_marker(marker)
}

/// Detect format and return size in bytes.
///
/// Convenience function that returns the bundle size directly.
#[inline]
pub fn detect_size(bytes: &[u8]) -> u8 {
    detect_format(bytes).size_bytes()
}

/// Legacy function for compatibility - detects format from 32-bit word.
///
/// Use `detect_format(bytes)` for new code.
#[inline]
pub fn detect_format_word(word0: u32, _bytes_available: usize) -> BundleFormat {
    let marker = (word0 & 0xF) as u8;
    BundleFormat::from_marker(marker)
}

/// Slot mask indicating which slots are active in a bundle.
///
/// This is a bitfield where bit N indicates slot N is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlotMask(pub u8);

impl SlotMask {
    /// No slots active.
    pub const EMPTY: SlotMask = SlotMask(0);

    /// All slots active.
    pub const ALL: SlotMask = SlotMask(0x7F);

    /// Create a mask with a single slot active.
    #[inline]
    pub const fn single(slot: u8) -> Self {
        SlotMask(1 << slot)
    }

    /// Check if a slot is active.
    #[inline]
    pub const fn is_active(self, slot: u8) -> bool {
        (self.0 & (1 << slot)) != 0
    }

    /// Set a slot as active.
    #[inline]
    pub fn set_active(&mut self, slot: u8) {
        self.0 |= 1 << slot;
    }

    /// Count the number of active slots.
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Check if this mask is compatible with the given format.
    #[inline]
    pub fn is_valid_for_format(self, format: BundleFormat) -> bool {
        format.supports_slots(self.count() as usize)
    }
}

/// Known NOP encodings in AIE2.
///
/// Multiple instruction encodings can represent a no-operation.
pub const NOP_ENCODINGS: &[u32] = &[
    0x0000_0000, // Zero word - simplest NOP
    0x1501_0040, // AIE canonical NOP
];

/// Check if the given word is a known NOP encoding.
#[inline]
pub fn is_nop_encoding(word: u32) -> bool {
    NOP_ENCODINGS.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_format_size() {
        assert_eq!(BundleFormat::Nop16.size_bytes(), 2);
        assert_eq!(BundleFormat::Short32.size_bytes(), 4);
        assert_eq!(BundleFormat::Medium48.size_bytes(), 6);
        assert_eq!(BundleFormat::Medium64.size_bytes(), 8);
        assert_eq!(BundleFormat::Long80.size_bytes(), 10);
        assert_eq!(BundleFormat::Long96.size_bytes(), 12);
        assert_eq!(BundleFormat::Long112.size_bytes(), 14);
        assert_eq!(BundleFormat::Full128.size_bytes(), 16);
    }

    #[test]
    fn test_bundle_format_marker() {
        assert_eq!(BundleFormat::Nop16.marker(), 0x1);
        assert_eq!(BundleFormat::Short32.marker(), 0x9);
        assert_eq!(BundleFormat::Medium48.marker(), 0x5);
        assert_eq!(BundleFormat::Medium64.marker(), 0x3);
        assert_eq!(BundleFormat::Long80.marker(), 0xB);
        assert_eq!(BundleFormat::Long96.marker(), 0x7);
        assert_eq!(BundleFormat::Long112.marker(), 0xF);
        assert_eq!(BundleFormat::Full128.marker(), 0x0);
    }

    #[test]
    fn test_bundle_format_from_marker() {
        assert_eq!(BundleFormat::from_marker(0x1), BundleFormat::Nop16);
        assert_eq!(BundleFormat::from_marker(0x9), BundleFormat::Short32);
        assert_eq!(BundleFormat::from_marker(0x5), BundleFormat::Medium48);
        assert_eq!(BundleFormat::from_marker(0x3), BundleFormat::Medium64);
        assert_eq!(BundleFormat::from_marker(0xB), BundleFormat::Long80);
        assert_eq!(BundleFormat::from_marker(0x7), BundleFormat::Long96);
        assert_eq!(BundleFormat::from_marker(0xF), BundleFormat::Long112);
        assert_eq!(BundleFormat::from_marker(0x0), BundleFormat::Full128);
    }

    #[test]
    fn test_bundle_format_supports_slots() {
        assert!(BundleFormat::Short32.supports_slots(0));
        assert!(BundleFormat::Short32.supports_slots(1));
        assert!(!BundleFormat::Short32.supports_slots(2));

        assert!(BundleFormat::Medium64.supports_slots(2));
        assert!(!BundleFormat::Medium64.supports_slots(3));

        assert!(BundleFormat::Full128.supports_slots(7));
        assert!(!BundleFormat::Full128.supports_slots(8));
    }

    #[test]
    fn test_detect_format() {
        // 16-bit NOP format (marker 0x1)
        assert_eq!(detect_format(&[0x01, 0x00]), BundleFormat::Nop16);
        assert_eq!(detect_format(&[0x11, 0x55]), BundleFormat::Nop16);

        // 32-bit format (marker 0x9)
        assert_eq!(detect_format(&[0x09, 0x00, 0x00, 0x00]), BundleFormat::Short32);
        assert_eq!(detect_format(&[0x19, 0xAB, 0xCD, 0xEF]), BundleFormat::Short32);

        // 48-bit format (marker 0x5)
        assert_eq!(detect_format(&[0x15, 0x00, 0x00, 0x40]), BundleFormat::Medium48);

        // 64-bit format (marker 0x3)
        assert_eq!(detect_format(&[0x03, 0x00, 0x00, 0x00]), BundleFormat::Medium64);

        // 80-bit format (marker 0xB)
        assert_eq!(detect_format(&[0x0B, 0x00]), BundleFormat::Long80);
        assert_eq!(detect_format(&[0xBB, 0x10]), BundleFormat::Long80);

        // 96-bit format (marker 0x7)
        assert_eq!(detect_format(&[0x37, 0x88, 0x00, 0x03]), BundleFormat::Long96);

        // 112-bit format (marker 0xF)
        assert_eq!(detect_format(&[0x7F, 0x00, 0x00, 0x00]), BundleFormat::Long112);

        // 128-bit format (marker 0x0) - but NOT when all zeros (that's a NOP)
        // Note: all zeros is treated as Short32 NOP
        assert_eq!(detect_format(&[0x00, 0x00, 0x00, 0x00]), BundleFormat::Short32);
        // Non-zero with low nibble 0 is 128-bit
        assert_eq!(detect_format(&[0xE0, 0xFF, 0xBB, 0x10]), BundleFormat::Full128);
    }

    #[test]
    fn test_detect_format_real_binary() {
        // From actual ELF test:
        // PC 0x0000: 0x40000115 - ends in 0x5 -> 48-bit
        assert_eq!(detect_format(&0x40000115u32.to_le_bytes()), BundleFormat::Medium48);

        // PC 0x0004: 0x00550001 - ends in 0x1 -> 16-bit NOP
        assert_eq!(detect_format(&0x00550001u32.to_le_bytes()), BundleFormat::Nop16);

        // PC 0x002C: 0x10BBFFE0 - ends in 0x0 -> 128-bit
        assert_eq!(detect_format(&0x10BBFFe0u32.to_le_bytes()), BundleFormat::Full128);

        // PC 0x0038: 0x9A0010BB - ends in 0xB -> 80-bit
        assert_eq!(detect_format(&0x9A0010BBu32.to_le_bytes()), BundleFormat::Long80);
    }

    #[test]
    fn test_slot_mask_operations() {
        let mut mask = SlotMask::EMPTY;
        assert_eq!(mask.count(), 0);

        mask.set_active(0);
        mask.set_active(2);
        assert!(mask.is_active(0));
        assert!(!mask.is_active(1));
        assert!(mask.is_active(2));
        assert_eq!(mask.count(), 2);
    }

    #[test]
    fn test_slot_mask_format_validity() {
        let single = SlotMask::single(0);
        assert!(single.is_valid_for_format(BundleFormat::Short32));
        assert!(single.is_valid_for_format(BundleFormat::Medium64));
        assert!(single.is_valid_for_format(BundleFormat::Full128));

        let all = SlotMask::ALL;
        assert!(!all.is_valid_for_format(BundleFormat::Short32));
        assert!(!all.is_valid_for_format(BundleFormat::Medium64));
        assert!(all.is_valid_for_format(BundleFormat::Full128));
    }

    #[test]
    fn test_nop_encodings() {
        assert!(is_nop_encoding(0x0000_0000));
        assert!(is_nop_encoding(0x1501_0040));
        assert!(!is_nop_encoding(0x4000_0000));
    }
}
