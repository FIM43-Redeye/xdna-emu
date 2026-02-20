//! AIE2 Shift-Round-Saturate (SRS) rounding modes.
//!
//! Implements the 10 hardware rounding modes used by the SRS instruction,
//! which converts wide accumulator values to narrower output types.
//!
//! # Hardware Behavior
//!
//! The SRS pipeline has three stages:
//! 1. **Shift**: Right-shift the accumulator value by `shift + BIAS` bits
//! 2. **Round**: Apply one of 10 rounding modes based on the discarded bits
//! 3. **Saturate**: Clamp the result to the output type range
//!
//! The BIAS of 4 bits provides extra precision in the accumulator -- the
//! hardware internally left-shifts by BIAS before the user-specified shift.
//!
//! # Rounding Decision
//!
//! The rounding decision is based on four signals extracted from the shifted-out
//! bits:
//! - **sgn**: Whether the input value is negative
//! - **lsb**: The least significant bit of the retained result
//! - **grd**: The guard bit (first bit below the truncation point)
//! - **stk**: The sticky bit (OR of all bits below the guard bit)
//!
//! Rounding behavior per AIE2 hardware specification.

/// Internal bias applied by the SRS hardware pipeline.
///
/// The accumulator value is effectively left-shifted by BIAS bits before
/// the user-specified shift is applied. This means the total right-shift
/// is `user_shift + BIAS`.
const BIAS: u32 = 4;

/// Hardware rounding modes for the SRS instruction.
///
/// The mode index values match the hardware encoding in the configuration word.
/// Valid indices are 0-3 and 8-13 (indices 4-7 are reserved/unused).
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
    /// Convert a raw hardware mode index to a RoundingMode.
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

/// Compute the rounding increment (0 or 1) for one lane.
///
/// The rounding decision depends on the mode and four bit-level signals:
/// - `sgn`: input is negative
/// - `lsb`: least significant bit of the retained (truncated) result
/// - `grd`: guard bit (first discarded bit)
/// - `stk`: sticky bit (OR of all discarded bits below the guard)
///
/// Rounding behavior per AIE2 hardware specification.
fn srs_round(mode: RoundingMode, sgn: bool, lsb: bool, grd: bool, stk: bool) -> bool {
    // Classify the mode along two axes:
    //
    // "halfway" modes only round when at or past the midpoint (grd=1).
    // Non-halfway modes (floor/ceil/sym variants) round based on whether
    // ANY discarded bits are nonzero.
    let is_halfway = matches!(
        mode,
        RoundingMode::NegInf
            | RoundingMode::PosInf
            | RoundingMode::SymInf
            | RoundingMode::SymZero
            | RoundingMode::ConvEven
            | RoundingMode::ConvOdd
    );

    // "symmetric" modes use sign to decide direction.
    let symmetric = matches!(
        mode,
        RoundingMode::SymFloor
            | RoundingMode::SymCeil
            | RoundingMode::SymInf
            | RoundingMode::SymZero
    );

    // "otherdir" modes round in the opposite direction from the default.
    // Default direction: floor=toward -inf, sym_floor=toward zero,
    // neg_inf=toward -inf, conv_even=to even.
    // "otherdir" flips to: ceil=toward +inf, sym_ceil=away from zero,
    // pos_inf=toward +inf, sym_inf=away from zero, conv_odd=to odd.
    let otherdir = matches!(
        mode,
        RoundingMode::Ceil
            | RoundingMode::SymCeil
            | RoundingMode::PosInf
            | RoundingMode::SymInf
            | RoundingMode::ConvOdd
    );

    let convergent = matches!(mode, RoundingMode::ConvEven | RoundingMode::ConvOdd);

    // Compute the "determinant" -- a direction signal that gets combined
    // with the discarded-bits signals.
    let det = if convergent {
        // For convergent modes, the tiebreaker is the LSB of the result.
        // ConvEven: round to even means round UP when lsb=1 (to make it 0).
        // ConvOdd: inverted by otherdir.
        lsb
    } else if symmetric {
        // For symmetric modes, direction depends on sign of the input.
        sgn
    } else {
        // For non-symmetric, non-convergent modes (floor/ceil, neg_inf/pos_inf),
        // the determinant is false (floor) or true (ceil, via otherdir).
        false
    };

    // Apply direction flip.
    let det = if otherdir { !det } else { det };

    // Final rounding decision.
    if is_halfway {
        // Halfway modes: round only when guard bit is set AND (determinant OR sticky).
        // At exact halfway (grd=1, stk=0), the determinant breaks the tie.
        // Past halfway (grd=1, stk=1), always round.
        grd && (det || stk)
    } else {
        // Non-halfway modes (floor/ceil/sym variants): round when determinant
        // is set AND any fractional bit is nonzero.
        det && (grd || stk)
    }
}

/// Perform shift-round-saturate on a single accumulator lane.
///
/// # Arguments
///
/// * `value` - The accumulator lane value (interpreted as signed 64-bit)
/// * `shift` - The user-specified shift amount (BIAS is added internally)
/// * `signed_output` - Whether the output type is signed
/// * `output_bits` - Bit width of the output type (8, 16, or 32)
/// * `saturate` - Whether to clamp out-of-range values (true for normal SRS)
/// * `symmetric_saturate` - Whether signed min is -(2^(n-1)-1) instead of -2^(n-1)
/// * `mode` - The rounding mode to apply
///
/// Returns the rounded and saturated result, truncated to `output_bits`.
///
/// Rounding behavior per AIE2 hardware specification.
pub fn srs_lane(
    value: i64,
    shift: u32,
    signed_output: bool,
    output_bits: u32,
    saturate: bool,
    symmetric_saturate: bool,
    mode: RoundingMode,
) -> i64 {
    // Stage 1: Shift
    //
    // Apply the internal BIAS: the hardware left-shifts the input by BIAS
    // to provide extra precision, then right-shifts by (user_shift + BIAS).
    let total_shift = shift + BIAS;

    // Work in i128 to avoid overflow during the bias shift.
    let a: i128 = (value as i128) << BIAS;
    let a_shifted = a >> total_shift;

    // Stage 2: Round
    //
    // Extract the guard, sticky, and LSB signals from the shifted-out bits.
    let (grd, stk, lsb) = if total_shift > 0 {
        // The value before shifting has been left-shifted by 1 relative to
        // the guard bit position. Guard bit is at position `total_shift`
        // in the doubled value (a << 1), sticky is the OR of bits below that.
        let doubled = a << 1;
        let stk = truncate_nonzero(doubled, total_shift);
        let grd = ((doubled >> total_shift) & 1) != 0;
        let lsb = ((doubled >> (total_shift + 1)) & 1) != 0;
        (grd, stk, lsb)
    } else {
        (false, false, ((a >> 1) & 1) != 0)
    };

    let sgn = a < 0;

    let a_rounded = if srs_round(mode, sgn, lsb, grd, stk) {
        a_shifted + 1
    } else {
        a_shifted
    };

    // Stage 3: Saturate
    let (vmin, vmax) = if signed_output {
        let vmax = (1i128 << (output_bits - 1)) - 1;
        let vmin = if symmetric_saturate {
            -(1i128 << (output_bits - 1)) + 1
        } else {
            -(1i128 << (output_bits - 1))
        };
        (vmin, vmax)
    } else {
        let vmax = (1i128 << output_bits) - 1;
        (0i128, vmax)
    };

    let a_saturated = if saturate {
        a_rounded.clamp(vmin, vmax)
    } else {
        a_rounded
    };

    // Truncate to output width.
    truncate_to_width(a_saturated, signed_output, output_bits)
}

/// Check whether any of the lower `bits` of `value` are nonzero.
fn truncate_nonzero(value: i128, bits: u32) -> bool {
    if bits == 0 {
        return false;
    }
    if bits >= 128 {
        return value != 0;
    }
    let mask = (1i128 << bits) - 1;
    (value & mask) != 0
}

/// Truncate a value to `bits` width, sign-extending if signed.
fn truncate_to_width(value: i128, signed: bool, bits: u32) -> i64 {
    if bits == 0 {
        return 0;
    }
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let truncated = (value as u64) & mask;

    if signed && bits < 64 {
        // Sign-extend from bit (bits-1).
        let sign_bit = 1u64 << (bits - 1);
        if truncated & sign_bit != 0 {
            (truncated | !mask) as i64
        } else {
            truncated as i64
        }
    } else {
        truncated as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: run SRS with default settings (signed 32-bit output, saturation on,
    // no symmetric saturation).
    fn srs_s32(value: i64, shift: u32, mode: RoundingMode) -> i64 {
        srs_lane(value, shift, true, 32, true, false, mode)
    }

    // Helper: run SRS for signed 16-bit output.
    fn srs_s16(value: i64, shift: u32, mode: RoundingMode) -> i64 {
        srs_lane(value, shift, true, 16, true, false, mode)
    }

    // Helper: run SRS for unsigned 8-bit output.
    fn srs_u8(value: i64, shift: u32, mode: RoundingMode) -> i64 {
        srs_lane(value, shift, false, 8, true, false, mode)
    }

    // -----------------------------------------------------------------------
    // Mode 0: Floor -- truncate toward -inf
    // -----------------------------------------------------------------------

    #[test]
    fn floor_positive_exact() {
        // 256 >> 4 = 16, no fractional bits
        assert_eq!(srs_s32(256, 4, RoundingMode::Floor), 16);
    }

    #[test]
    fn floor_positive_with_fraction() {
        // 260 >> 4 = 16.25, floor -> 16
        assert_eq!(srs_s32(260, 4, RoundingMode::Floor), 16);
    }

    #[test]
    fn floor_negative_with_fraction() {
        // -260 >> 4 = -16.25, floor -> -17
        assert_eq!(srs_s32(-260, 4, RoundingMode::Floor), -17);
    }

    #[test]
    fn floor_negative_exact() {
        // -256 >> 4 = -16, exact
        assert_eq!(srs_s32(-256, 4, RoundingMode::Floor), -16);
    }

    // -----------------------------------------------------------------------
    // Mode 1: Ceil -- round toward +inf
    // -----------------------------------------------------------------------

    #[test]
    fn ceil_positive_with_fraction() {
        // 260 >> 4 = 16.25, ceil -> 17
        assert_eq!(srs_s32(260, 4, RoundingMode::Ceil), 17);
    }

    #[test]
    fn ceil_positive_exact() {
        // 256 >> 4 = 16, exact -> 16
        assert_eq!(srs_s32(256, 4, RoundingMode::Ceil), 16);
    }

    #[test]
    fn ceil_negative_with_fraction() {
        // -260 >> 4 = -16.25, ceil -> -16 (toward +inf)
        assert_eq!(srs_s32(-260, 4, RoundingMode::Ceil), -16);
    }

    // -----------------------------------------------------------------------
    // Mode 2: SymFloor -- toward zero
    // -----------------------------------------------------------------------

    #[test]
    fn sym_floor_positive_with_fraction() {
        // 260 >> 4 = 16.25, toward zero -> 16
        assert_eq!(srs_s32(260, 4, RoundingMode::SymFloor), 16);
    }

    #[test]
    fn sym_floor_negative_with_fraction() {
        // -260 >> 4 = -16.25, toward zero -> -16
        assert_eq!(srs_s32(-260, 4, RoundingMode::SymFloor), -16);
    }

    // -----------------------------------------------------------------------
    // Mode 3: SymCeil -- away from zero
    // -----------------------------------------------------------------------

    #[test]
    fn sym_ceil_positive_with_fraction() {
        // 260 >> 4 = 16.25, away from zero -> 17
        assert_eq!(srs_s32(260, 4, RoundingMode::SymCeil), 17);
    }

    #[test]
    fn sym_ceil_negative_with_fraction() {
        // -260 >> 4 = -16.25, away from zero -> -17
        assert_eq!(srs_s32(-260, 4, RoundingMode::SymCeil), -17);
    }

    #[test]
    fn sym_ceil_exact() {
        // Exact values should not be rounded.
        assert_eq!(srs_s32(256, 4, RoundingMode::SymCeil), 16);
        assert_eq!(srs_s32(-256, 4, RoundingMode::SymCeil), -16);
    }

    // -----------------------------------------------------------------------
    // Mode 8: NegInf -- round half toward -inf
    // -----------------------------------------------------------------------

    #[test]
    fn neg_inf_below_half() {
        // 260 >> 4 = 16.25, below half -> 16
        assert_eq!(srs_s32(260, 4, RoundingMode::NegInf), 16);
    }

    #[test]
    fn neg_inf_at_half() {
        // 264 >> 4 = 16.5, at half -> toward -inf = 16
        assert_eq!(srs_s32(264, 4, RoundingMode::NegInf), 16);
    }

    #[test]
    fn neg_inf_above_half() {
        // 268 >> 4 = 16.75, above half -> 17
        assert_eq!(srs_s32(268, 4, RoundingMode::NegInf), 17);
    }

    #[test]
    fn neg_inf_negative_at_half() {
        // -264 >> 4 = -16.5, at half -> toward -inf = -17
        assert_eq!(srs_s32(-264, 4, RoundingMode::NegInf), -17);
    }

    // -----------------------------------------------------------------------
    // Mode 9: PosInf -- round half toward +inf
    // -----------------------------------------------------------------------

    #[test]
    fn pos_inf_at_half() {
        // 264 >> 4 = 16.5, at half -> toward +inf = 17
        assert_eq!(srs_s32(264, 4, RoundingMode::PosInf), 17);
    }

    #[test]
    fn pos_inf_negative_at_half() {
        // -264 >> 4 = -16.5, at half -> toward +inf = -16
        assert_eq!(srs_s32(-264, 4, RoundingMode::PosInf), -16);
    }

    #[test]
    fn pos_inf_below_half() {
        assert_eq!(srs_s32(260, 4, RoundingMode::PosInf), 16);
    }

    // -----------------------------------------------------------------------
    // Mode 10: SymZero -- round half toward zero
    // -----------------------------------------------------------------------

    #[test]
    fn sym_zero_positive_at_half() {
        // 264 >> 4 = 16.5, toward zero -> 16
        assert_eq!(srs_s32(264, 4, RoundingMode::SymZero), 16);
    }

    #[test]
    fn sym_zero_negative_at_half() {
        // -264 >> 4 = -16.5, toward zero -> -16
        assert_eq!(srs_s32(-264, 4, RoundingMode::SymZero), -16);
    }

    #[test]
    fn sym_zero_above_half() {
        // Above half always rounds to nearest.
        assert_eq!(srs_s32(268, 4, RoundingMode::SymZero), 17);
        assert_eq!(srs_s32(-268, 4, RoundingMode::SymZero), -17);
    }

    // -----------------------------------------------------------------------
    // Mode 11: SymInf -- round half away from zero
    // -----------------------------------------------------------------------

    #[test]
    fn sym_inf_positive_at_half() {
        // 264 >> 4 = 16.5, away from zero -> 17
        assert_eq!(srs_s32(264, 4, RoundingMode::SymInf), 17);
    }

    #[test]
    fn sym_inf_negative_at_half() {
        // -264 >> 4 = -16.5, away from zero -> -17
        assert_eq!(srs_s32(-264, 4, RoundingMode::SymInf), -17);
    }

    // -----------------------------------------------------------------------
    // Mode 12: ConvEven -- round half to even (banker's rounding)
    // -----------------------------------------------------------------------

    #[test]
    fn conv_even_half_to_even_down() {
        // 264 >> 4 = 16.5, 16 is even -> 16
        assert_eq!(srs_s32(264, 4, RoundingMode::ConvEven), 16);
    }

    #[test]
    fn conv_even_half_to_even_up() {
        // 280 >> 4 = 17.5, 17 is odd -> round to 18 (even)
        assert_eq!(srs_s32(280, 4, RoundingMode::ConvEven), 18);
    }

    #[test]
    fn conv_even_above_half() {
        // 268 >> 4 = 16.75 -> 17 (above half always rounds to nearest)
        assert_eq!(srs_s32(268, 4, RoundingMode::ConvEven), 17);
    }

    #[test]
    fn conv_even_below_half() {
        // 260 >> 4 = 16.25 -> 16
        assert_eq!(srs_s32(260, 4, RoundingMode::ConvEven), 16);
    }

    // -----------------------------------------------------------------------
    // Mode 13: ConvOdd -- round half to odd
    // -----------------------------------------------------------------------

    #[test]
    fn conv_odd_half_to_odd_up() {
        // 264 >> 4 = 16.5, 16 is even -> round to 17 (odd)
        assert_eq!(srs_s32(264, 4, RoundingMode::ConvOdd), 17);
    }

    #[test]
    fn conv_odd_half_to_odd_down() {
        // 280 >> 4 = 17.5, 17 is odd -> stay 17
        assert_eq!(srs_s32(280, 4, RoundingMode::ConvOdd), 17);
    }

    // -----------------------------------------------------------------------
    // Saturation tests
    // -----------------------------------------------------------------------

    #[test]
    fn saturate_signed_16_positive_overflow() {
        // Large value that exceeds i16::MAX after shift
        let value = 32768i64 * 16; // 32768 >> 0 = 32768 > 32767
        assert_eq!(srs_s16(value, 0, RoundingMode::Floor), 32767);
    }

    #[test]
    fn saturate_signed_16_negative_overflow() {
        let value = -32769i64 * 16; // after shift exceeds i16::MIN
        assert_eq!(srs_s16(value, 0, RoundingMode::Floor), -32768);
    }

    #[test]
    fn saturate_unsigned_8_overflow() {
        let value = 256i64 * 16; // 256 > u8::MAX
        assert_eq!(srs_u8(value, 0, RoundingMode::Floor), 255);
    }

    #[test]
    fn saturate_unsigned_8_negative() {
        // Negative value to unsigned -> clamp to 0
        let value = -16i64;
        assert_eq!(srs_u8(value, 0, RoundingMode::Floor), 0);
    }

    #[test]
    fn symmetric_saturation_clips_min() {
        // With symmetric saturation, signed min is -(2^(n-1)-1) not -(2^(n-1)).
        // For 16-bit: min is -32767 not -32768.
        let value = -32768i64 * 16;
        let result = srs_lane(value, 0, true, 16, true, true, RoundingMode::Floor);
        assert_eq!(result, -32767);
    }

    // -----------------------------------------------------------------------
    // Zero shift
    // -----------------------------------------------------------------------

    #[test]
    fn zero_shift_passthrough() {
        // With shift=0, total_shift=BIAS=4. The hardware BIAS still applies.
        // Input 160 with shift 0: total_shift = 4, so 160 >> 4 = 10.
        // But the BIAS pre-shift means: (160 << 4) >> 4 = 160.
        // So effectively shift=0 means the value passes through unchanged.
        assert_eq!(srs_s32(42, 0, RoundingMode::Floor), 42);
    }

    // -----------------------------------------------------------------------
    // RoundingMode::from_raw
    // -----------------------------------------------------------------------

    #[test]
    fn from_raw_valid_modes() {
        assert_eq!(RoundingMode::from_raw(0), Some(RoundingMode::Floor));
        assert_eq!(RoundingMode::from_raw(1), Some(RoundingMode::Ceil));
        assert_eq!(RoundingMode::from_raw(2), Some(RoundingMode::SymFloor));
        assert_eq!(RoundingMode::from_raw(3), Some(RoundingMode::SymCeil));
        assert_eq!(RoundingMode::from_raw(8), Some(RoundingMode::NegInf));
        assert_eq!(RoundingMode::from_raw(9), Some(RoundingMode::PosInf));
        assert_eq!(RoundingMode::from_raw(10), Some(RoundingMode::SymZero));
        assert_eq!(RoundingMode::from_raw(11), Some(RoundingMode::SymInf));
        assert_eq!(RoundingMode::from_raw(12), Some(RoundingMode::ConvEven));
        assert_eq!(RoundingMode::from_raw(13), Some(RoundingMode::ConvOdd));
    }

    #[test]
    fn from_raw_reserved_modes() {
        for i in [4, 5, 6, 7, 14, 15, 16, 255] {
            assert_eq!(
                RoundingMode::from_raw(i),
                None,
                "index {} should be None",
                i
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn large_shift_zeroes_result() {
        // Shifting by more bits than the input has -> 0 (or -1 for floor of negative)
        assert_eq!(srs_s32(1000, 60, RoundingMode::Floor), 0);
    }

    #[test]
    fn negative_floor_large_shift() {
        // -1 with large shift: floor of a tiny negative number is -1
        assert_eq!(srs_s32(-1, 60, RoundingMode::Floor), -1);
    }

    #[test]
    fn ceil_exact_negative() {
        // Exact negative values: ceil of -16.0 is -16
        assert_eq!(srs_s32(-256, 4, RoundingMode::Ceil), -16);
    }

    // -----------------------------------------------------------------------
    // Cross-mode comparison at the same input
    // -----------------------------------------------------------------------

    #[test]
    fn all_modes_exact_value() {
        // When the value is exact (no fractional bits), all modes agree.
        let exact = 256i64; // 256 >> 4 = 16 exactly
        for mode in [
            RoundingMode::Floor,
            RoundingMode::Ceil,
            RoundingMode::SymFloor,
            RoundingMode::SymCeil,
            RoundingMode::NegInf,
            RoundingMode::PosInf,
            RoundingMode::SymZero,
            RoundingMode::SymInf,
            RoundingMode::ConvEven,
            RoundingMode::ConvOdd,
        ] {
            assert_eq!(
                srs_s32(exact, 4, mode),
                16,
                "mode {:?} should give 16 for exact value",
                mode
            );
        }
    }

    #[test]
    fn halfway_modes_diverge_at_half() {
        // At exactly 16.5 (264 >> 4), the modes should diverge predictably.
        let half = 264i64;
        assert_eq!(srs_s32(half, 4, RoundingMode::NegInf), 16);
        assert_eq!(srs_s32(half, 4, RoundingMode::PosInf), 17);
        assert_eq!(srs_s32(half, 4, RoundingMode::SymZero), 16);
        assert_eq!(srs_s32(half, 4, RoundingMode::SymInf), 17);
        assert_eq!(srs_s32(half, 4, RoundingMode::ConvEven), 16); // 16 is even
        assert_eq!(srs_s32(half, 4, RoundingMode::ConvOdd), 17); // 17 is odd
    }

    #[test]
    fn negative_halfway_modes_diverge() {
        // At -16.5 (-264 >> 4), the modes diverge in the opposite direction.
        let half = -264i64;
        assert_eq!(srs_s32(half, 4, RoundingMode::NegInf), -17);
        assert_eq!(srs_s32(half, 4, RoundingMode::PosInf), -16);
        assert_eq!(srs_s32(half, 4, RoundingMode::SymZero), -16);
        assert_eq!(srs_s32(half, 4, RoundingMode::SymInf), -17);
        assert_eq!(srs_s32(half, 4, RoundingMode::ConvEven), -16); // -16 is even
        assert_eq!(srs_s32(half, 4, RoundingMode::ConvOdd), -17); // -17 is odd
    }
}
