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
    pub const MARKER_NOP16: u8 = 0b0001; // 0x1
    pub const MARKER_SHORT32: u8 = 0b1001; // 0x9
    pub const MARKER_MEDIUM48: u8 = 0b0101; // 0x5
    pub const MARKER_MEDIUM64: u8 = 0b0011; // 0x3
    pub const MARKER_LONG80: u8 = 0b1011; // 0xB
    pub const MARKER_LONG96: u8 = 0b0111; // 0x7
    pub const MARKER_LONG112: u8 = 0b1111; // 0xF
    pub const MARKER_FULL128: u8 = 0b0000; // 0x0

    /// Detect format from the low 4 bits of the first halfword.
    ///
    /// The 128-bit format only has a 1-bit marker (bit 0 = 0), so any even
    /// nibble (0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE) indicates 128-bit.
    /// All other formats have bit 0 = 1.
    ///
    /// IMPORTANT: The 48-bit format uses only 3 bits for its marker (0b101),
    /// while other formats use 4 bits. This means we must check the 3-bit
    /// marker first before checking 4-bit markers.
    #[inline]
    pub fn from_marker(marker: u8) -> Self {
        let nibble = marker & 0xF;

        // 128-bit format has only 1-bit marker (bit 0 = 0)
        // All other formats have bit 0 = 1
        if nibble & 1 == 0 {
            return BundleFormat::Full128;
        }

        // 48-bit format has a 3-bit marker (0b101), so check it first
        // This allows 0x5, 0xD (0b0101, 0b1101) to both match 48-bit
        if (marker & 0x7) == Self::MARKER_MEDIUM48 {
            return BundleFormat::Medium48;
        }

        match nibble {
            Self::MARKER_NOP16 => BundleFormat::Nop16,
            Self::MARKER_SHORT32 => BundleFormat::Short32,
            Self::MARKER_MEDIUM64 => BundleFormat::Medium64,
            Self::MARKER_LONG80 => BundleFormat::Long80,
            Self::MARKER_LONG96 => BundleFormat::Long96,
            Self::MARKER_LONG112 => BundleFormat::Long112,
            // All odd values that don't match known markers default to 32-bit
            _ => BundleFormat::Short32,
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
}

/// The all-zero instruction word: unwritten program memory past the last bundle.
///
/// Its low nibble is 0, which by the marker rule would be a 128-bit bundle,
/// but no TableGen composite format matches it. Real kernels fetch it (the
/// datadriven CLEAN probe saw a 16-zero-byte read only in this form), so the
/// decoder keeps treating it as a 4-byte NOP. Known gap (PLAN bug 10): a real
/// 128-bit bundle whose first 32 bits are zero would be misread; none has been
/// seen. Needs a silicon probe before the rule can be tightened.
#[inline]
pub const fn is_zero_padding(word0: u32) -> bool {
    word0 == 0
}

/// Detect the bundle format from the first bytes.
///
/// The format is encoded in the low 4 bits of the first halfword, so only
/// 2 bytes are needed. Returns `Short32` if fewer than 2 bytes are available.
pub fn detect_format(bytes: &[u8]) -> BundleFormat {
    if bytes.len() < 2 {
        return BundleFormat::Short32;
    }
    let hw0 = u16::from_le_bytes([bytes[0], bytes[1]]);

    let zero = if bytes.len() >= 4 {
        is_zero_padding(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    } else {
        hw0 == 0
    };
    if zero {
        return BundleFormat::Short32;
    }

    BundleFormat::from_marker((hw0 & 0xF) as u8)
}

/// Slot mask indicating which slots are active in a bundle.
///
/// This is a bitfield where bit N indicates slot N is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlotMask(pub u8);

impl SlotMask {
    /// No slots active.
    pub const EMPTY: SlotMask = SlotMask(0);

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
}
