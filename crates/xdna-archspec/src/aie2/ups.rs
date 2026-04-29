//! UPS (Upshift / type-widening) mode table for the AIE2 vector unit.
//!
//! The four valid UPS modes encode the lane count and element widths
//! for the hardware's shift-and-widen operation. Each mode promotes
//! narrow integer lanes into wider accumulator lanes with an optional
//! left shift before multiply-accumulate chains.
//!
//! | Scale | Acc mode | Lanes | Input bits | Output bits |
//! |-------|----------|-------|------------|-------------|
//! | Half  | Acc32    | 32    | 8          | 32          |
//! | Full  | Acc32    | 32    | 16         | 32          |
//! | Half  | Acc64    | 16    | 16         | 64          |
//! | Full  | Acc64    | 16    | 32         | 64          |
//!
//! Source: AIE2 hardware model; cross-validated against the
//! aietools Python UPS model (`ups.py` scale/acc_mode table).

/// UPS scale modes.
///
/// Half-scale uses the narrower input type for the given accumulator width;
/// Full-scale uses the wider input type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsScale {
    /// Half-scale: narrower input type (8->32 or 16->64).
    Half,
    /// Full-scale: wider input type (16->32 or 32->64).
    Full,
}

/// UPS accumulator output width modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsAccMode {
    /// 32-bit accumulator output.
    Acc32,
    /// 64-bit accumulator output.
    Acc64,
}

/// Parameters for a single UPS mode, derived from the hardware mode table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpsMode {
    /// Number of SIMD lanes.
    pub lanes: u32,
    /// Input element width in bits.
    pub bits_in: u32,
    /// Output element width in bits.
    pub bits_out: u32,
}

/// Look up the UPS mode parameters from scale and accumulator settings.
///
/// Returns the `UpsMode` matching the hardware mode table for the given
/// `(scale, acc)` combination. All four combinations are valid; this
/// function always returns a result.
///
/// Source: AIE2 hardware mode table. Same table is represented in
/// the aietools Python model.
pub fn ups_mode(scale: UpsScale, acc: UpsAccMode) -> UpsMode {
    match (scale, acc) {
        (UpsScale::Half, UpsAccMode::Acc32) => UpsMode { lanes: 32, bits_in: 8, bits_out: 32 },
        (UpsScale::Full, UpsAccMode::Acc32) => UpsMode { lanes: 32, bits_in: 16, bits_out: 32 },
        (UpsScale::Half, UpsAccMode::Acc64) => UpsMode { lanes: 16, bits_in: 16, bits_out: 64 },
        (UpsScale::Full, UpsAccMode::Acc64) => UpsMode { lanes: 16, bits_in: 32, bits_out: 64 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drift-detection test: the four UPS mode entries must match the
    /// hardware table exactly. Any change here requires a verified hardware
    /// cross-check.
    #[test]
    fn ups_mode_table_all_four_entries() {
        let cases = [
            (UpsScale::Half, UpsAccMode::Acc32, UpsMode { lanes: 32, bits_in: 8, bits_out: 32 }),
            (UpsScale::Full, UpsAccMode::Acc32, UpsMode { lanes: 32, bits_in: 16, bits_out: 32 }),
            (UpsScale::Half, UpsAccMode::Acc64, UpsMode { lanes: 16, bits_in: 16, bits_out: 64 }),
            (UpsScale::Full, UpsAccMode::Acc64, UpsMode { lanes: 16, bits_in: 32, bits_out: 64 }),
        ];

        for (scale, acc, expected) in cases {
            let got = ups_mode(scale, acc);
            assert_eq!(got, expected, "UPS mode mismatch for ({scale:?}, {acc:?})");
        }
    }

    /// Sanity: acc_cmb modes (Acc64) produce wider output lanes than Acc32 modes.
    #[test]
    fn ups_mode_acc64_wider_than_acc32() {
        // Acc64 output is 64 bits/lane; Acc32 output is 32 bits/lane.
        // Both scale variants must follow this rule.
        for scale in [UpsScale::Half, UpsScale::Full] {
            let m32 = ups_mode(scale, UpsAccMode::Acc32);
            let m64 = ups_mode(scale, UpsAccMode::Acc64);
            assert_eq!(m32.bits_out, 32, "Acc32 output must be 32 bits");
            assert_eq!(m64.bits_out, 64, "Acc64 output must be 64 bits");
        }
    }
}
