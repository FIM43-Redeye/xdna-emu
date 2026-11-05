//! VLIW bundle slot extraction based on AIE2CompositeFormats.td.
//!
//! Each bundle format has specific slot layouts. This module extracts
//! individual slot data from packed VLIW bundles, driven entirely by the
//! composite format records parsed from TableGen (`FormatTable`).
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
        self.slots.push(ExtractedSlot { slot_type, bits, width });
    }

    /// Check if this bundle is empty (no slots extracted).
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

// ============================================================================
// Data-driven format table (built from tblgen Inst field)
// ============================================================================

/// A single composite format entry for runtime discrimination and extraction.
#[derive(Clone)]
pub(crate) struct FormatEntry {
    /// Format name (e.g., "I128_LDB_LDA_ST_ALU_MV_VEC").
    #[allow(dead_code)]
    name: String,
    /// Bitmask of fixed positions in the bundle word.
    fixed_mask: u128,
    /// Expected value at fixed positions.
    fixed_value: u128,
    /// Per-slot extraction info: (SlotType, start_bit, width).
    slots: Vec<(SlotType, u8, u8)>,
}

/// Data-driven VLIW bundle slot extraction table.
///
/// Built from the parsed `Inst` field of all composite format records.
/// Groups entries by `BundleFormat` (determined by bundle size) and
/// discriminates within each group via mask+compare on the raw bundle word.
#[derive(Clone, Default)]
pub struct FormatTable {
    /// Entries grouped by BundleFormat discriminant index (0..8).
    groups: [Vec<FormatEntry>; 8],
}

impl FormatTable {
    /// Build a FormatTable from parsed composite format definitions.
    ///
    /// Only formats that have Inst-derived bit-level layout (non-zero fixed_mask
    /// or non-empty slot_maps) are included.
    pub fn build(formats: &[xdna_archspec::aie2::isa::CompositeFormatDef]) -> Self {
        let mut groups: [Vec<FormatEntry>; 8] = Default::default();

        for fmt in formats {
            // Skip formats without Inst-derived layout
            if fmt.slot_maps.is_empty() && fmt.fixed_mask == 0 {
                continue;
            }

            let group_idx = match fmt.total_bytes {
                2 => 0,  // Nop16
                4 => 1,  // Short32
                6 => 2,  // Medium48
                8 => 3,  // Medium64
                10 => 4, // Long80
                12 => 5, // Long96
                14 => 6, // Long112
                16 => 7, // Full128
                _ => continue,
            };

            let slots: Vec<(SlotType, u8, u8)> = fmt
                .slot_maps
                .iter()
                .filter_map(|sm| {
                    let slot_type = slot_name_to_type(&sm.slot_name)?;
                    Some((slot_type, sm.start_bit, sm.width))
                })
                .collect();

            groups[group_idx].push(FormatEntry {
                name: fmt.name.clone(),
                fixed_mask: fmt.fixed_mask,
                fixed_value: fmt.fixed_value,
                slots,
            });
        }

        // Sort each group: most fixed 1-bits first (more distinctive patterns
        // match faster and avoid ambiguity with less specific patterns).
        for group in &mut groups {
            group.sort_by(|a, b| {
                let ones_a = (a.fixed_value & a.fixed_mask).count_ones();
                let ones_b = (b.fixed_value & b.fixed_mask).count_ones();
                ones_b
                    .cmp(&ones_a)
                    .then_with(|| a.fixed_mask.count_ones().cmp(&b.fixed_mask.count_ones()).reverse())
            });
        }

        FormatTable { groups }
    }

    /// Extract slot data from a raw bundle using the data-driven table.
    ///
    /// Returns `None` if no format matches (a malformed bundle).
    pub fn extract(&self, bytes: &[u8], format: BundleFormat) -> Option<ExtractedBundle> {
        let group_idx = format_to_group_index(format);
        let group = &self.groups[group_idx];
        if group.is_empty() {
            return None;
        }

        // Read bytes into u128 (little-endian, zero-padded)
        let word = bytes_to_u128(bytes);

        // Linear scan: first match wins
        for entry in group {
            if (word & entry.fixed_mask) == entry.fixed_value {
                let mut bundle = ExtractedBundle::new();
                for &(slot_type, start_bit, width) in &entry.slots {
                    let mask = if width >= 128 {
                        u128::MAX
                    } else {
                        (1u128 << width) - 1
                    };
                    let bits = ((word >> start_bit) & mask) as u64;
                    bundle.add_slot(slot_type, bits, width);
                }
                return Some(bundle);
            }
        }
        None
    }

    /// Check if the table has entries for a given format size.
    pub fn has_entries(&self, format: BundleFormat) -> bool {
        !self.groups[format_to_group_index(format)].is_empty()
    }

    /// Total number of format entries across all groups.
    pub fn total_entries(&self) -> usize {
        self.groups.iter().map(|g| g.len()).sum()
    }

    /// Get entry count for a specific format group (for testing).
    #[cfg(test)]
    pub(crate) fn group_entries(&self, format: BundleFormat) -> &[FormatEntry] {
        &self.groups[format_to_group_index(format)]
    }
}

/// Convert a slot name string to SlotType.
fn slot_name_to_type(name: &str) -> Option<SlotType> {
    match name {
        "lda" => Some(SlotType::Lda),
        "ldb" => Some(SlotType::Ldb),
        "alu" => Some(SlotType::Alu),
        "mv" => Some(SlotType::Mv),
        "st" => Some(SlotType::St),
        "vec" => Some(SlotType::Vec),
        "lng" => Some(SlotType::Lng),
        "nop" | "nop16" => Some(SlotType::Nop),
        _ => {
            log::trace!("Unknown slot name '{}', skipping", name);
            None
        }
    }
}

/// Map BundleFormat to group index (0..8).
fn format_to_group_index(format: BundleFormat) -> usize {
    match format {
        BundleFormat::Nop16 => 0,
        BundleFormat::Short32 => 1,
        BundleFormat::Medium48 => 2,
        BundleFormat::Medium64 => 3,
        BundleFormat::Long80 => 4,
        BundleFormat::Long96 => 5,
        BundleFormat::Long112 => 6,
        BundleFormat::Full128 => 7,
    }
}

/// Read up to 16 bytes as a little-endian u128.
fn bytes_to_u128(bytes: &[u8]) -> u128 {
    let mut buf = [0u8; 16];
    let len = bytes.len().min(16);
    buf[..len].copy_from_slice(&bytes[..len]);
    u128::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // FormatTable tests
    // ========================================================================

    /// Build a FormatTable from the build-time generated tblgen output.
    fn try_build_format_table() -> Option<FormatTable> {
        let output = xdna_archspec::aie2::isa::load_from_generated();
        Some(FormatTable::build(&output.composite_formats))
    }

    #[test]
    fn test_empty_table_matches_nothing() {
        let table = FormatTable::default();
        assert_eq!(table.total_entries(), 0);
        assert!(table.extract(&[0x09, 0, 0, 0], BundleFormat::Short32).is_none());
    }

    #[test]
    fn test_format_table_builds_all_78() {
        let table = match try_build_format_table() {
            Some(t) => t,
            None => {
                eprintln!("Skipping: llvm-aie not found");
                return;
            }
        };

        // Should have all 78 composite formats
        assert_eq!(table.total_entries(), 78, "Expected 78 format entries, got {}", table.total_entries());

        // Each size group should have entries
        assert!(table.has_entries(BundleFormat::Short32));
        assert!(table.has_entries(BundleFormat::Medium48));
        assert!(table.has_entries(BundleFormat::Medium64));
        assert!(table.has_entries(BundleFormat::Long80));
        assert!(table.has_entries(BundleFormat::Long96));
        assert!(table.has_entries(BundleFormat::Long112));
        assert!(table.has_entries(BundleFormat::Full128));
    }

    #[test]
    fn test_format_table_discrimination_uniqueness() {
        let table = match try_build_format_table() {
            Some(t) => t,
            None => {
                eprintln!("Skipping: llvm-aie not found");
                return;
            }
        };

        // For each size group, verify no two formats match the same word.
        // Build a synthetic word for each format using its fixed_value and
        // check that no other format in the same group also matches.
        for format in [
            BundleFormat::Nop16,
            BundleFormat::Short32,
            BundleFormat::Medium48,
            BundleFormat::Medium64,
            BundleFormat::Long80,
            BundleFormat::Long96,
            BundleFormat::Long112,
            BundleFormat::Full128,
        ] {
            let group = table.group_entries(format);
            for (i, entry_i) in group.iter().enumerate() {
                // Synthetic word with only the fixed bits set
                let test_word = entry_i.fixed_value;
                let mut matches = Vec::new();
                for (j, entry_j) in group.iter().enumerate() {
                    if (test_word & entry_j.fixed_mask) == entry_j.fixed_value {
                        matches.push(j);
                    }
                }
                // The first match should be this entry (or earlier, more-specific one)
                assert!(!matches.is_empty(), "Format {} should match its own fixed_value", entry_i.name);
                // Entry i should be among the matches
                assert!(
                    matches.contains(&i),
                    "Format {} (idx {}) not in its own match set {:?}",
                    entry_i.name,
                    i,
                    matches
                );
            }
        }
    }

    #[test]
    fn test_format_table_roundtrip_all_formats() {
        let output = xdna_archspec::aie2::isa::load_from_generated();
        let table = FormatTable::build(&output.composite_formats);

        // For each composite format, construct a synthetic bundle with known
        // slot values and verify extraction recovers them exactly.
        for fmt in &output.composite_formats {
            if fmt.slot_maps.is_empty() {
                continue;
            }

            let total_bits = fmt.total_bits;
            let total_bytes = fmt.total_bytes;

            // Start with the fixed value (has all discriminator bits set)
            let mut word: u128 = fmt.fixed_value;

            // Insert known test values for each slot
            let mut expected: Vec<(SlotType, u64, u8)> = Vec::new();
            for sm in &fmt.slot_maps {
                let slot_type = match slot_name_to_type(&sm.slot_name) {
                    Some(t) => t,
                    None => continue,
                };
                // Use a test pattern: alternating bits within the slot width
                let test_val = 0x5A5A5A5A5A5A5A5Au64 & ((1u64 << sm.width) - 1);
                // Insert into the word
                let mask = if sm.width >= 128 {
                    u128::MAX
                } else {
                    (1u128 << sm.width) - 1
                };
                word &= !(mask << sm.start_bit); // clear slot region
                word |= (test_val as u128) << sm.start_bit; // set test value

                expected.push((slot_type, test_val, sm.width));
            }

            // Convert to bytes
            let bytes = word.to_le_bytes();
            let bundle_bytes = &bytes[..total_bytes as usize];

            // Detect format from the bytes
            let format = crate::interpreter::bundle::detect_format(bundle_bytes);

            // Extract using the table
            let result = table.extract(bundle_bytes, format);
            assert!(
                result.is_some(),
                "Format {} ({}b): no table match for word=0x{:032X}",
                fmt.name,
                total_bits,
                word
            );
            let bundle = result.unwrap();

            // Verify each expected slot
            for (exp_type, exp_val, exp_width) in &expected {
                let slot = bundle.slots.iter().find(|s| s.slot_type == *exp_type);
                assert!(slot.is_some(), "Format {}: missing {:?} slot", fmt.name, exp_type);
                let slot = slot.unwrap();
                assert_eq!(
                    slot.bits, *exp_val,
                    "Format {}: {:?} slot value mismatch (got 0x{:X}, expected 0x{:X})",
                    fmt.name, exp_type, slot.bits, exp_val
                );
                assert_eq!(slot.width, *exp_width, "Format {}: {:?} slot width mismatch", fmt.name, exp_type);
            }
        }
    }

    #[test]
    fn test_format_table_backward_compat_32bit_alu() {
        let table = match try_build_format_table() {
            Some(t) => t,
            None => {
                eprintln!("Skipping: llvm-aie not found");
                return;
            }
        };

        // I32_ALU: {0b00010, alu[19:0], 0b001, 0b1001}
        let alu_value: u32 = 0x12345;
        let word: u32 = (0b00010 << 27) | (alu_value << 7) | (0b001 << 4) | 0b1001;
        let bytes = word.to_le_bytes();

        let driven = table.extract(&bytes, BundleFormat::Short32).unwrap();

        assert_eq!(driven.slots.len(), 1, "Slot count mismatch for I32_ALU");
        assert_eq!(driven.slots[0].slot_type, SlotType::Alu);
        assert_eq!(driven.slots[0].bits, alu_value as u64);
    }

    #[test]
    fn test_format_table_backward_compat_128bit() {
        let table = match try_build_format_table() {
            Some(t) => t,
            None => {
                eprintln!("Skipping: llvm-aie not found");
                return;
            }
        };

        // I128_LDB_LDA_ST_ALU_MV_VEC
        let vec_value: u128 = 0x2ABCDEF;
        let mv_value: u128 = 0x3ABCDE;
        let alu_value: u128 = 0x89ABC;
        let st_value: u128 = 0x1BCDEF;
        let lda_value: u128 = 0x1CDEF0;
        let ldb_value: u128 = 0xABCD;

        let word: u128 = 0
            | (vec_value << 1)
            | (1u128 << 27)
            | (mv_value << 28)
            | (alu_value << 50)
            | (st_value << 70)
            | (lda_value << 91)
            | (ldb_value << 112);

        let bytes = word.to_le_bytes();

        let driven = table.extract(&bytes, BundleFormat::Full128).unwrap();

        assert_eq!(driven.slots.len(), 6);
        let expect = [
            (SlotType::Vec, vec_value),
            (SlotType::Mv, mv_value),
            (SlotType::Alu, alu_value),
            (SlotType::St, st_value),
            (SlotType::Lda, lda_value),
            (SlotType::Ldb, ldb_value),
        ];
        for (slot_type, want) in expect {
            let got = driven.slots.iter().find(|s| s.slot_type == slot_type);
            assert_eq!(got.map(|s| s.bits), Some(want as u64), "{slot_type:?}");
        }
    }
}
