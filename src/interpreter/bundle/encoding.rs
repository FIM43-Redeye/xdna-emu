//! VLIW bundle encoding and format detection.
//!
//! AIE2 instructions come in three sizes:
//!
//! - **Short (32-bit)**: Single scalar or control operation
//! - **Medium (64-bit)**: Scalar + limited vector/memory
//! - **Full (128-bit)**: Complete VLIW with all slots available
//!
//! The format is determined by examining specific bits in the instruction word.

/// Bundle format - determines the instruction size and available slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BundleFormat {
    /// 32-bit short format - single operation.
    #[default]
    Short32,
    /// 64-bit medium format - two operations.
    Medium64,
    /// 128-bit full VLIW format - all slots available.
    Full128,
}

impl BundleFormat {
    /// Get the size of this format in bytes.
    #[inline]
    pub const fn size_bytes(self) -> u8 {
        match self {
            BundleFormat::Short32 => 4,
            BundleFormat::Medium64 => 8,
            BundleFormat::Full128 => 16,
        }
    }

    /// Get the size of this format in bits.
    #[inline]
    pub const fn size_bits(self) -> u16 {
        (self.size_bytes() as u16) * 8
    }

    /// Check if this format supports the given slot count.
    #[inline]
    pub fn supports_slots(self, count: usize) -> bool {
        match self {
            BundleFormat::Short32 => count <= 1,
            BundleFormat::Medium64 => count <= 2,
            BundleFormat::Full128 => count <= 7,
        }
    }
}

/// Detect the bundle format from the first instruction word.
///
/// This function examines the encoding bits to determine whether
/// we're looking at a 32-bit, 64-bit, or 128-bit instruction.
///
/// # Arguments
///
/// * `word0` - The first 32-bit word of the instruction
/// * `bytes_available` - How many bytes are available for reading
///
/// # Returns
///
/// The detected bundle format, or `Short32` if uncertain.
pub fn detect_format(word0: u32, bytes_available: usize) -> BundleFormat {
    // AIE2 encoding uses specific marker bits to indicate format.
    // These patterns are derived from observing real instruction streams.

    // Check if this looks like a VLIW header (high bits set in specific ways)
    let opcode_high = (word0 >> 28) & 0xF;

    // Full 128-bit VLIW bundles typically have specific header patterns
    if bytes_available >= 16 && is_full_vliw_header(word0) {
        return BundleFormat::Full128;
    }

    // Medium 64-bit format detection
    if bytes_available >= 8 && is_medium_format(word0) {
        return BundleFormat::Medium64;
    }

    // Default to short 32-bit format
    BundleFormat::Short32
}

/// Check if the instruction word looks like a full VLIW header.
///
/// Full VLIW bundles have specific marker bits set that indicate
/// the presence of multiple slot operations.
fn is_full_vliw_header(word0: u32) -> bool {
    // Vector/accumulator operations often indicate full VLIW
    // These typically have opcodes in the 0xB-0xF range with specific encoding
    let opcode_high = (word0 >> 28) & 0xF;

    // Check for VLIW marker bits
    // The exact pattern depends on the instruction encoding
    if opcode_high >= 0xB {
        // Additional check for VLIW header signature
        // This is a heuristic - the real decoder will be more precise
        return (word0 & 0x8000_0000) != 0;
    }

    false
}

/// Check if the instruction uses medium (64-bit) format.
fn is_medium_format(word0: u32) -> bool {
    // Medium format is used for certain combined scalar+memory operations
    // This is a heuristic - will be refined with TableGen data
    let opcode_high = (word0 >> 28) & 0xF;

    // Some memory operations with vector operands use 64-bit format
    matches!(opcode_high, 0x8..=0xA) && (word0 & 0x0080_0000) != 0
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
        assert_eq!(BundleFormat::Short32.size_bytes(), 4);
        assert_eq!(BundleFormat::Medium64.size_bytes(), 8);
        assert_eq!(BundleFormat::Full128.size_bytes(), 16);

        assert_eq!(BundleFormat::Short32.size_bits(), 32);
        assert_eq!(BundleFormat::Medium64.size_bits(), 64);
        assert_eq!(BundleFormat::Full128.size_bits(), 128);
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
    fn test_detect_format_short() {
        // Simple scalar instruction should be short format
        let word = 0x4000_0000; // Scalar add pattern
        assert_eq!(detect_format(word, 16), BundleFormat::Short32);

        // NOP is short
        assert_eq!(detect_format(0x0000_0000, 16), BundleFormat::Short32);
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
