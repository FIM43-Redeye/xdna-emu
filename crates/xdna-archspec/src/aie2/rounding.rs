//! Hardware rounding modes for the SRS instruction and bf16 conversion.
//!
//! The mode index is the 4-bit `crRnd` field (Core_CR Round_Mode). Mode names
//! and the defined indices 0-3 and 8-13 follow llvm-aie `aiev2_defines.h`.
//!
//! Indices 4-7, 14 and 15 are not defined by the toolchain headers, but they
//! are not faults on silicon either: each behaves as a fixed alias or as a
//! mode of its own. This was observed on NPU1 (Phoenix) silicon, captured
//! 2026-10-28 (`tools/golden/silicon/vec_srs_i32_r{4,5,6,7,14,15}_s1.json`):
//!
//! | raw | behaviour                                   | variant       |
//! |-----|---------------------------------------------|---------------|
//! | 4   | round to even whenever the result is inexact | `InexactEven` |
//! | 5   | round to odd whenever the result is inexact  | `InexactOdd`  |
//! | 6   | same as 4                                    | `InexactEven` |
//! | 7   | same as 5                                    | `InexactOdd`  |
//! | 14  | same as 12                                   | `ConvEven`    |
//! | 15  | same as 13                                   | `ConvOdd`     |
//!
//! Per-mode semantics are applied by `vector_srs::shift_round`, which both the
//! SRS pipeline and the f32->bf16 conversion share.

/// Hardware rounding modes for the SRS instruction and bf16 conversion.
///
/// "Dropped bits" below means the low `shift` bits discarded by the shift
/// (for bf16 conversion, the low 16 bits of the f32 magnitude pattern). All
/// descriptions are as observed on NPU1 silicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundingMode {
    /// Mode 0 (`rnd_floor`): truncate the dropped bits, i.e. round toward -infinity.
    Floor = 0,

    /// Mode 1 (`rnd_ceil`): round toward +infinity whenever any dropped bit is set.
    Ceil = 1,

    /// Mode 2 (`rnd_sym_floor`): truncate the magnitude, i.e. round toward zero.
    SymFloor = 2,

    /// Mode 3 (`rnd_sym_ceil`): round the magnitude up whenever any dropped bit is set, i.e. away from zero.
    SymCeil = 3,

    /// Mode 4 (and 6): when any dropped bit is set, pick whichever of the two
    /// neighbouring results is even; exact results are unchanged. Undefined by
    /// the toolchain; silicon-observed.
    InexactEven = 4,

    /// Mode 5 (and 7): when any dropped bit is set, pick whichever of the two
    /// neighbouring results is odd; exact results are unchanged. Undefined by
    /// the toolchain; silicon-observed.
    InexactOdd = 5,

    /// Mode 8 (`rnd_neg_inf`): round to nearest, ties toward -infinity.
    NegInf = 8,

    /// Mode 9 (`rnd_pos_inf`): round to nearest, ties toward +infinity.
    PosInf = 9,

    /// Mode 10 (`rnd_sym_zero`): round to nearest, ties toward zero.
    SymZero = 10,

    /// Mode 11 (`rnd_sym_inf`): round to nearest, ties away from zero.
    SymInf = 11,

    /// Mode 12 (and 14, `rnd_conv_even`): round to nearest, ties to the even result.
    ConvEven = 12,

    /// Mode 13 (and 15, `rnd_conv_odd`): round to nearest, ties to the odd result.
    ConvOdd = 13,
}

impl RoundingMode {
    /// Convert a raw `crRnd` index to a `RoundingMode`.
    ///
    /// Every 4-bit index maps to a mode (undefined indices 4-7, 14, 15 follow
    /// the silicon-observed table in the module docs). Returns `None` only for
    /// values that do not fit the 4-bit field.
    pub fn from_raw(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::Floor),
            1 => Some(Self::Ceil),
            2 => Some(Self::SymFloor),
            3 => Some(Self::SymCeil),
            4 | 6 => Some(Self::InexactEven),
            5 | 7 => Some(Self::InexactOdd),
            8 => Some(Self::NegInf),
            9 => Some(Self::PosInf),
            10 => Some(Self::SymZero),
            11 => Some(Self::SymInf),
            12 | 14 => Some(Self::ConvEven),
            13 | 15 => Some(Self::ConvOdd),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_mode_from_raw_defined_indices() {
        for (raw, expected) in [
            (0u8, RoundingMode::Floor),
            (1, RoundingMode::Ceil),
            (2, RoundingMode::SymFloor),
            (3, RoundingMode::SymCeil),
            (8, RoundingMode::NegInf),
            (9, RoundingMode::PosInf),
            (10, RoundingMode::SymZero),
            (11, RoundingMode::SymInf),
            (12, RoundingMode::ConvEven),
            (13, RoundingMode::ConvOdd),
        ] {
            assert_eq!(RoundingMode::from_raw(raw), Some(expected));
        }
    }

    #[test]
    fn undefined_indices_follow_silicon_aliases() {
        for (raw, expected) in [
            (4u8, RoundingMode::InexactEven),
            (6, RoundingMode::InexactEven),
            (5, RoundingMode::InexactOdd),
            (7, RoundingMode::InexactOdd),
            (14, RoundingMode::ConvEven),
            (15, RoundingMode::ConvOdd),
        ] {
            assert_eq!(RoundingMode::from_raw(raw), Some(expected), "raw={raw}");
        }
    }

    #[test]
    fn indices_wider_than_the_four_bit_field_return_none() {
        for raw in [16u8, 17, 200, 255] {
            assert_eq!(RoundingMode::from_raw(raw), None, "raw={raw} should be None");
        }
    }
}
