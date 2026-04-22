//! Hardware rounding modes for the SRS instruction and bf16 conversion.
//!
//! The mode index values match the AIE2 hardware encoding in the
//! configuration word. Valid indices are 0-3 and 8-13 (indices 4-7
//! are reserved).
//!
//! The same 10 modes are used by both the integer SRS pipeline
//! (accumulator shift-round-saturate) and the bf16 conversion path.
//! The sign-magnitude path used by bf16 conversion inverts the
//! symmetry direction compared to the integer path; this is a
//! call-site concern, not a change to the mode values themselves.

/// Hardware rounding modes for the SRS instruction and bf16 conversion.
///
/// The mode index values match the hardware encoding in the configuration word.
/// Valid indices are 0-3 and 8-13 (indices 4-7 are reserved/unused).
///
/// Rounding behavior per AIE2 hardware specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundingMode {
    /// Mode 0: Floor -- truncate toward negative infinity.
    /// Discards all fractional bits. Equivalent to arithmetic right shift.
    Floor = 0,

    /// Mode 1: Ceiling -- round toward positive infinity.
    /// Adds 1 if any discarded bits are nonzero and value is not already exact.
    Ceil = 1,

    /// Mode 2: Symmetric floor -- round toward zero (positive) or away (negative).
    /// "Symmetric" means sign-dependent: positive values truncate, negative
    /// values round away from zero. (For sign-magnitude input, this inverts.)
    SymFloor = 2,

    /// Mode 3: Symmetric ceiling -- round away from zero (positive) or toward (negative).
    /// Opposite of SymFloor.
    SymCeil = 3,

    /// Mode 8: Round half toward negative infinity.
    /// At the exact halfway point (grd=1, stk=0), rounds toward -inf.
    /// Otherwise rounds to nearest.
    NegInf = 8,

    /// Mode 9: Round half toward positive infinity.
    /// At the exact halfway point, rounds toward +inf.
    /// Otherwise rounds to nearest.
    PosInf = 9,

    /// Mode 10: Round half toward zero (symmetric).
    /// At the exact halfway point, rounds toward zero.
    /// Otherwise rounds to nearest.
    SymZero = 10,

    /// Mode 11: Round half away from zero (symmetric).
    /// At the exact halfway point, rounds away from zero.
    /// Otherwise rounds to nearest.
    SymInf = 11,

    /// Mode 12: Convergent rounding to even (IEEE 754 banker's rounding).
    /// At the exact halfway point, rounds to the nearest even value.
    /// Otherwise rounds to nearest.
    ConvEven = 12,

    /// Mode 13: Convergent rounding to odd.
    /// At the exact halfway point, rounds to the nearest odd value.
    /// Otherwise rounds to nearest.
    ConvOdd = 13,
}

impl RoundingMode {
    /// Convert a raw hardware mode index to a `RoundingMode`.
    ///
    /// Returns `None` for reserved indices (4-7, 14-15, etc.).
    pub fn from_raw(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::Floor),
            1 => Some(Self::Ceil),
            2 => Some(Self::SymFloor),
            3 => Some(Self::SymCeil),
            8 => Some(Self::NegInf),
            9 => Some(Self::PosInf),
            10 => Some(Self::SymZero),
            11 => Some(Self::SymInf),
            12 => Some(Self::ConvEven),
            13 => Some(Self::ConvOdd),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_mode_from_raw_valid_indices() {
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
    fn rounding_mode_from_raw_reserved_indices_return_none() {
        for raw in [4u8, 5, 6, 7, 14, 15, 200, 255] {
            assert_eq!(RoundingMode::from_raw(raw), None, "raw={raw} should be None");
        }
    }
}
