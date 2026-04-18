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
const BIAS: u32 = crate::arch::processor::SRS_SHIFT_BIAS as u32;

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
    //
    // Clamp total_shift: if shift is absurdly large (e.g., random test data),
    // the result is just the sign extension (0 or -1). Cap at 127 to avoid
    // Rust's shift-by->=bitwidth panic on i128.
    let total_shift = shift.saturating_add(BIAS).min(127);

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
        //
        // When total_shift is very large (>= 126), virtually all bits are
        // shifted out. Guard/sticky/LSB signals are derived from what remains
        // after the massive shift, which is effectively the sign extension.
        // Clamp individual shift amounts to avoid Rust panics on >= 128.
        let doubled = a.wrapping_shl(1);
        let stk = truncate_nonzero(doubled, total_shift);
        let grd = if total_shift < 128 {
            ((doubled >> total_shift) & 1) != 0
        } else {
            false
        };
        let lsb_shift = total_shift.saturating_add(1);
        let lsb = if lsb_shift < 128 {
            ((doubled >> lsb_shift) & 1) != 0
        } else {
            false
        };
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
    // Use unsigned arithmetic for the mask to avoid overflow when bits=127
    // (1i128 << 127 = i128::MIN, subtracting 1 would overflow).
    let mask = ((1u128 << bits) - 1) as i128;
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

// --- VectorAlu dispatch and pipeline wrappers ---

use crate::interpreter::bundle::{ElementType, SlotOp};
use crate::interpreter::state::{ExecutionContext, SrsConfig};
use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    /// Dispatch entry point for SRS (Shift-Round-Saturate) operations.
    ///
    /// Handles both narrow (256-bit) and wide (512-bit / 1024-bit) paths,
    /// including AccumWidth detection for Half vs Full accumulator sources.
    pub(super) fn execute_srs(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let has_wide_acc_source = matches!(op.accum_width,
            Some(crate::interpreter::decode::register_map::AccumWidth::Full)
            | Some(crate::interpreter::decode::register_map::AccumWidth::Half));

        if op.is_wide_vector || has_wide_acc_source {
            Self::execute_srs_wide(op, ctx, et)
        } else {
            Self::execute_srs_narrow(op, ctx, et)
        }
    }

    /// Narrow SRS path: read 512-bit accumulator, SRS to 256-bit vector.
    fn execute_srs_narrow(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let acc_reg = Self::get_acc_source(op);
        let shift = Self::get_shift_amount(op, ctx);
        let from = op.from_type.unwrap_or(ElementType::Int32);
        let result = Self::vector_srs(ctx, acc_reg, shift, from, et);
        Self::write_vector_dest(op, ctx, result);
        true
    }

    /// Wide SRS path: read 1024-bit or 512-bit accumulator, SRS to 512-bit or 256-bit vector.
    fn execute_srs_wide(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let acc_reg = Self::get_acc_source(op);
        let shift = Self::get_shift_amount(op, ctx);
        let from = op.from_type.unwrap_or(ElementType::Int64);
        // Half (bml/bmh) is 512-bit = single bm register, use narrow
        // read. Full (cm) and None (legacy/wide default) use read_wide.
        let is_half = matches!(op.accum_width,
            Some(crate::interpreter::decode::register_map::AccumWidth::Half));

        if !is_half {
            // Wide SRS: read Acc1024 (cm-register), SRS each half,
            // write Vec512.
            let acc_wide = ctx.accumulator.read_wide(acc_reg);
            let acc_lo: [u64; 8] = acc_wide[..8].try_into().unwrap();
            let acc_hi: [u64; 8] = acc_wide[8..].try_into().unwrap();

            let result_lo = Self::vector_srs_from_acc(&acc_lo, shift, from, et, &ctx.srs_config);
            let result_hi = Self::vector_srs_from_acc(&acc_hi, shift, from, et, &ctx.srs_config);

            // Pack the two halves contiguously. For reduction SRS
            // (e.g., s8 from s32, s16 from s64), each half only fills
            // a fraction of its 8-word output.
            let from_bits = from.bits() as usize;
            let lanes_per_half = if from_bits <= 32 { 16 } else { 8 };
            let to_bits = et.bits() as usize;
            let words_per_half = (lanes_per_half * to_bits + 31) / 32;

            let mut result = [0u32; 16];
            let n = words_per_half.min(8);
            result[..n].copy_from_slice(&result_lo[..n]);
            result[n..n + n].copy_from_slice(&result_hi[..n]);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            // Half SRS (bml/bmh): read single 512-bit accum register,
            // SRS 8 lanes, write to 256-bit w-register.
            let acc = ctx.accumulator.read(acc_reg);
            let result = Self::vector_srs_from_acc(&acc, shift, from, et, &ctx.srs_config);
            Self::write_vector_dest(op, ctx, result);
        }
        true
    }

    /// Shift-Round-Saturate: convert accumulator lanes to narrower vector output.
    ///
    /// Thin wrapper that reads the accumulator register and delegates to
    /// `vector_srs_from_acc` for the actual conversion logic.
    fn vector_srs(
        ctx: &ExecutionContext,
        acc_reg: u8,
        shift: u32,
        from_type: ElementType,
        to_type: ElementType,
    ) -> [u32; 8] {
        let acc = ctx.accumulator.read(acc_reg);
        Self::vector_srs_from_acc(&acc, shift, from_type, to_type, &ctx.srs_config)
    }

    /// Core SRS logic operating on accumulator data directly.
    ///
    /// The accumulator always has 8 lanes of 64-bit each (one value per u64).
    /// SRS reads 8 accumulator values and converts them to narrower output,
    /// packing multiple values per u32 word for 16-bit and 8-bit outputs.
    ///
    /// `from_type` controls how many bits of each u64 lane are meaningful:
    /// in S32 mode only the low 32 bits matter (upper bits may be garbage),
    /// in S64 mode the full 64 bits are used.
    ///
    /// Delegates to the `vector_srs` module which implements the full 10-mode
    /// SRS pipeline (shift, round, saturate) per AIE2 hardware specification.
    /// Float types (BFloat16, Float32) are handled inline since they bypass
    /// the integer rounding pipeline.
    ///
    /// Used by the narrow `vector_srs`, wide SRS handler, and fused `vst.srs`.
    pub(crate) fn vector_srs_from_acc(
        acc: &[u64; 8],
        shift: u32,
        from_type: ElementType,
        to_type: ElementType,
        cfg: &SrsConfig,
    ) -> [u32; 8] {
        let mut result = [0u32; 8];

        // Read rounding and saturation from the SRS control register state.
        let mode = RoundingMode::from_raw(cfg.rounding_mode)
            .unwrap_or(RoundingMode::PosInf);
        let saturate = cfg.saturate();
        let sym_sat = cfg.symmetric_saturate();

        let signed_output = to_type.is_signed();

        // Mask accumulator values to from_type width.
        // In S32 mode, only low 32 bits are meaningful.
        // In S64 mode (or 64-bit types), use the full u64.
        let from_bits = from_type.bits() as u32;
        let mask_value = |raw: u64| -> i64 {
            if from_bits >= 64 {
                raw as i64
            } else {
                // Sign-extend from from_bits width.
                let shift_amt = 64 - from_bits;
                ((raw as i64) << shift_amt) >> shift_amt
            }
        };

        // 8 accumulator lanes, each u64 holds one value.
        match to_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
                for i in 0..8 {
                    let val = mask_value(acc[i]);
                    let out = srs_lane(
                        val, shift, signed_output, 32,
                        saturate, sym_sat, mode,
                    );
                    result[i] = out as u32;
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                if from_bits >= 64 {
                    // Acc64 -> 16-bit (4:1 reduction): 8 u64 lanes, each
                    // SRS'd from 64-bit to 16-bit. Pack 2 per u32 = 4 words.
                    for i in 0..4 {
                        let val0 = mask_value(acc[i * 2]);
                        let val1 = mask_value(acc[i * 2 + 1]);
                        let out0 = srs_lane(
                            val0, shift, signed_output, 16,
                            saturate, sym_sat, mode,
                        );
                        let out1 = srs_lane(
                            val1, shift, signed_output, 16,
                            saturate, sym_sat, mode,
                        );
                        result[i] = (out0 as u16 as u32) | ((out1 as u16 as u32) << 16);
                    }
                } else {
                    // Acc32 -> 16-bit (2:1 reduction): 16 lanes packed 2 per u64.
                    // Extract lo32 and hi32 from each u64 to get 16 acc32 lanes,
                    // then SRS each to 16-bit and pack 2 per u32 result word.
                    for i in 0..8 {
                        let lo_val = mask_value(acc[i] & 0xFFFF_FFFF);
                        let hi_val = mask_value(acc[i] >> 32);
                        let out_lo = srs_lane(
                            lo_val, shift, signed_output, 16,
                            saturate, sym_sat, mode,
                        );
                        let out_hi = srs_lane(
                            hi_val, shift, signed_output, 16,
                            saturate, sym_sat, mode,
                        );
                        result[i] = (out_lo as u16 as u32) | ((out_hi as u16 as u32) << 16);
                    }
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // Acc32 mode: 16 lanes packed 2 per u64 -> 16 x 8-bit output.
                // 16 bytes = 4 u32 result words.
                for i in 0..4 {
                    let mut word = 0u32;
                    for j in 0..4 {
                        let lane_idx = i * 4 + j;
                        let u64_idx = lane_idx / 2;
                        let is_hi = lane_idx % 2 == 1;
                        let raw = if is_hi {
                            acc[u64_idx] >> 32
                        } else {
                            acc[u64_idx] & 0xFFFF_FFFF
                        };
                        let val = mask_value(raw);
                        let out = srs_lane(
                            val, shift, signed_output, 8,
                            saturate, sym_sat, mode,
                        );
                        word |= (out as u8 as u32) << (j * 8);
                    }
                    result[i] = word;
                }
            }
            ElementType::BFloat16 => {
                // Acc32 float mode: each u64 holds two f32 values.
                // BFloat16 SRS: extract each f32 and truncate to bf16,
                // packing two bf16 per u32 result word -> 16 bf16 lanes.
                for i in 0..8 {
                    let f_lo = f32::from_bits(acc[i] as u32);
                    let f_hi = f32::from_bits((acc[i] >> 32) as u32);
                    let bf_lo = Self::f32_to_bf16(f_lo);
                    let bf_hi = Self::f32_to_bf16(f_hi);
                    result[i] = (bf_lo as u32) | ((bf_hi as u32) << 16);
                }
            }
            ElementType::Float32 => {
                // Acc32 float mode: each u64 holds two f32 values
                // (bits [31:0] = lo, bits [63:32] = hi).
                // Float SRS is a pass-through: extract the low f32 per lane.
                // 8 accumulator u64 words -> 8 f32 result words (low half).
                for i in 0..8 {
                    result[i] = acc[i] as u32;
                }
            }
        }

        result
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
    // Golden reference saturation tests (from ISA Spec 1.12 SRS model)
    //
    // These test vectors are ported from the AMD SRS reference model that
    // validates boundary behavior at i16 saturation limits. Each test uses
    // shift=8 with Floor rounding (mode 0) and 16-bit signed output.
    //
    // Test categories:
    // A. Exact boundary values (should pass through without saturation)
    // B. One-beyond-boundary (should saturate)
    // C. Boundary + half-point (rounding interacts with saturation)
    //
    // Repeated for both normal and symmetric saturation.
    // -----------------------------------------------------------------------

    /// Helper for golden reference tests: signed 16-bit output, shift=8.
    fn srs_golden(value: i64, sat: bool, sym_sat: bool) -> i64 {
        srs_lane(value, 8, true, 16, sat, sym_sat, RoundingMode::Floor)
    }

    // -- Category A: Exact boundary values, normal saturation --

    #[test]
    fn golden_normal_max_exact() {
        // 32767 * 256 >> 8 = 32767 (i16::MAX): no saturation
        assert_eq!(srs_golden(32767 * 256, true, false), 32767);
    }

    #[test]
    fn golden_normal_min_exact() {
        // -32768 * 256 >> 8 = -32768 (i16::MIN): no saturation
        assert_eq!(srs_golden(-32768 * 256, true, false), -32768);
    }

    #[test]
    fn golden_normal_sym_min_exact() {
        // -32767 * 256 >> 8 = -32767: no saturation
        assert_eq!(srs_golden(-32767 * 256, true, false), -32767);
    }

    // -- Category B: One beyond boundary, normal saturation --

    #[test]
    fn golden_normal_max_plus_one() {
        // 32768 * 256 >> 8 = 32768: saturates to 32767
        assert_eq!(srs_golden(32768 * 256, true, false), 32767);
    }

    #[test]
    fn golden_normal_min_minus_one() {
        // -32769 * 256 >> 8 = -32769: saturates to -32768
        assert_eq!(srs_golden(-32769 * 256, true, false), -32768);
    }

    #[test]
    fn golden_normal_sym_min_minus_one() {
        // -32768 * 256 >> 8 = -32768: equals normal min, no saturation
        assert_eq!(srs_golden(-32768 * 256, true, false), -32768);
    }

    // -- Category C: Boundary + half-point, normal saturation --

    #[test]
    fn golden_normal_max_plus_half() {
        // (32767 * 256) + 128 = 32767.5 * 256. Floor -> 32767, within range.
        assert_eq!(srs_golden(32767 * 256 + 128, true, false), 32767);
    }

    #[test]
    fn golden_normal_min_minus_half() {
        // (-32768 * 256) - 128 = -32768.5 * 256. Floor -> -32769, saturates to -32768.
        assert_eq!(srs_golden(-32768 * 256 - 128, true, false), -32768);
    }

    #[test]
    fn golden_normal_sym_min_minus_half() {
        // (-32767 * 256) - 128 = -32767.5 * 256. Floor -> -32768, within normal range.
        assert_eq!(srs_golden(-32767 * 256 - 128, true, false), -32768);
    }

    // -- Category A: Exact boundary values, symmetric saturation --

    #[test]
    fn golden_sym_max_exact() {
        // 32767 * 256 >> 8 = 32767: no saturation
        assert_eq!(srs_golden(32767 * 256, true, true), 32767);
    }

    #[test]
    fn golden_sym_min_exact() {
        // -32768 * 256 >> 8 = -32768: symmetric min is -32767, saturates.
        assert_eq!(srs_golden(-32768 * 256, true, true), -32767);
    }

    #[test]
    fn golden_sym_sym_min_exact() {
        // -32767 * 256 >> 8 = -32767: equals symmetric min, no saturation.
        assert_eq!(srs_golden(-32767 * 256, true, true), -32767);
    }

    // -- Category B: One beyond boundary, symmetric saturation --

    #[test]
    fn golden_sym_max_plus_one() {
        // 32768 * 256 >> 8 = 32768: saturates to 32767
        assert_eq!(srs_golden(32768 * 256, true, true), 32767);
    }

    #[test]
    fn golden_sym_min_minus_one() {
        // -32769 * 256 >> 8 = -32769: saturates to -32767 (symmetric min)
        assert_eq!(srs_golden(-32769 * 256, true, true), -32767);
    }

    #[test]
    fn golden_sym_sym_min_minus_one() {
        // -32768 * 256 >> 8 = -32768: saturates to -32767 (symmetric min)
        assert_eq!(srs_golden(-32768 * 256, true, true), -32767);
    }

    // -- Category C: Boundary + half-point, symmetric saturation --

    #[test]
    fn golden_sym_max_plus_half() {
        // (32767 * 256) + 128: Floor -> 32767, within range.
        assert_eq!(srs_golden(32767 * 256 + 128, true, true), 32767);
    }

    #[test]
    fn golden_sym_min_minus_half() {
        // (-32768 * 256) - 128: Floor -> -32769, saturates to -32767 (symmetric).
        assert_eq!(srs_golden(-32768 * 256 - 128, true, true), -32767);
    }

    #[test]
    fn golden_sym_sym_min_minus_half() {
        // (-32767 * 256) - 128: Floor -> -32768, saturates to -32767 (symmetric).
        assert_eq!(srs_golden(-32767 * 256 - 128, true, true), -32767);
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

    // -----------------------------------------------------------------------
    // truncate_nonzero -- internal helper
    // -----------------------------------------------------------------------

    #[test]
    fn truncate_nonzero_zero_bits() {
        assert!(!truncate_nonzero(0xFF, 0));
    }

    #[test]
    fn truncate_nonzero_all_zero() {
        assert!(!truncate_nonzero(0x100, 8)); // only bit 8 set, checking low 8 bits
    }

    #[test]
    fn truncate_nonzero_some_set() {
        assert!(truncate_nonzero(0x101, 8)); // bit 0 set in low 8 bits
    }

    #[test]
    fn truncate_nonzero_large_bits() {
        // bits >= 128: returns value != 0
        assert!(truncate_nonzero(1, 128));
        assert!(!truncate_nonzero(0, 128));
    }

    // -----------------------------------------------------------------------
    // truncate_to_width -- internal helper
    // -----------------------------------------------------------------------

    #[test]
    fn truncate_to_width_zero_bits() {
        assert_eq!(truncate_to_width(0xABCD, false, 0), 0);
    }

    #[test]
    fn truncate_to_width_unsigned_8bit() {
        assert_eq!(truncate_to_width(0xFF, false, 8), 0xFF);
        assert_eq!(truncate_to_width(0x1FF, false, 8), 0xFF);
    }

    #[test]
    fn truncate_to_width_signed_8bit() {
        // 0x80 in signed 8-bit = -128
        assert_eq!(truncate_to_width(0x80, true, 8), -128);
        // 0x7F in signed 8-bit = 127
        assert_eq!(truncate_to_width(0x7F, true, 8), 127);
    }

    #[test]
    fn truncate_to_width_64bit() {
        // bits >= 64: no masking
        assert_eq!(truncate_to_width(i64::MAX as i128, true, 64), i64::MAX);
    }

    // -----------------------------------------------------------------------
    // srs_lane with saturation disabled
    // -----------------------------------------------------------------------

    #[test]
    fn no_saturation_allows_overflow() {
        // 32768 exceeds i16::MAX (32767). With shift=0, BIAS cancels out
        // (value << BIAS >> BIAS = value), so result = 32768 truncated to
        // 16-bit signed = -32768 (wraps). With saturation on, it would
        // clamp to 32767.
        let result = srs_lane(32768, 0, true, 16, false, false, RoundingMode::Floor);
        assert_eq!(result, -32768);

        // Compare: with saturation on, same value clamps.
        let result_sat = srs_lane(32768, 0, true, 16, true, false, RoundingMode::Floor);
        assert_eq!(result_sat, 32767);
    }

    // -----------------------------------------------------------------------
    // ConvEven/ConvOdd with negative halfway values
    // -----------------------------------------------------------------------

    #[test]
    fn conv_even_negative_half_to_even() {
        // -280 >> 4 = -17.5, -18 is even -> -18
        assert_eq!(srs_s32(-280, 4, RoundingMode::ConvEven), -18);
    }

    #[test]
    fn conv_even_negative_half_already_even() {
        // -264 >> 4 = -16.5, -16 is even -> -16
        assert_eq!(srs_s32(-264, 4, RoundingMode::ConvEven), -16);
    }

    #[test]
    fn conv_odd_negative_half_to_odd() {
        // -264 >> 4 = -16.5, -16 is even -> -17 (odd)
        assert_eq!(srs_s32(-264, 4, RoundingMode::ConvOdd), -17);
    }

    #[test]
    fn conv_odd_negative_half_already_odd() {
        // -280 >> 4 = -17.5, -17 is odd -> -17
        assert_eq!(srs_s32(-280, 4, RoundingMode::ConvOdd), -17);
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

/// Integration tests that exercise SRS through the full VectorAlu dispatch path.
#[cfg(test)]
mod integration_tests {
    use crate::interpreter::bundle::{ElementType, Operand, SlotIndex, SlotOp};
    use crate::interpreter::execute::vector_dispatch::VectorAlu;
    use crate::interpreter::state::{ExecutionContext, SrsConfig};
    use crate::tablegen::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    #[test]
    fn test_vector_srs_int32() {
        let mut ctx = make_ctx();
        ctx.accumulator.write(0, [256, 512, 768, 1024, 1280, 1536, 1792, 2048]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(4));
        op.from_type = Some(ElementType::Int32);

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(0);
        assert_eq!(result[0], 16);
        assert_eq!(result[1], 32);
        assert_eq!(result[2], 48);
        assert_eq!(result[3], 64);
    }

    #[test]
    fn test_vector_srs_reads_config() {
        let mut ctx = make_ctx();
        ctx.accumulator.write(0, [264, 0, 0, 0, 0, 0, 0, 0]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(4));
        op.from_type = Some(ElementType::Int32);

        // PosInf (mode 9): 16.5 -> 17
        ctx.srs_config.rounding_mode = 9;
        ctx.srs_config.saturation_mode = 1;
        ctx.srs_config.srs_sign = true;
        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(0)[0], 17);

        // Floor (mode 0): 16.5 -> 16
        ctx.srs_config.rounding_mode = 0;
        ctx.accumulator.write(0, [264, 0, 0, 0, 0, 0, 0, 0]);
        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(0)[0], 16);

        // NegInf (mode 8): 16.5 -> 16
        ctx.srs_config.rounding_mode = 8;
        ctx.accumulator.write(0, [264, 0, 0, 0, 0, 0, 0, 0]);
        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(0)[0], 16);
    }

    #[test]
    fn test_vector_srs_saturation_from_config() {
        let mut ctx = make_ctx();
        let overflow_val = 32768i64 as u64;
        ctx.accumulator.write(0, [overflow_val, 0, 0, 0, 0, 0, 0, 0]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int16)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(0));
        op.from_type = Some(ElementType::Int32);

        ctx.srs_config.saturation_mode = 1;
        ctx.srs_config.srs_sign = true;
        VectorAlu::execute(&op, &mut ctx);
        let lo16 = ctx.vector.read(0)[0] as i16;
        assert_eq!(lo16, 32767);

        ctx.srs_config.saturation_mode = 0;
        ctx.accumulator.write(0, [overflow_val, 0, 0, 0, 0, 0, 0, 0]);
        VectorAlu::execute(&op, &mut ctx);
        let lo16_nowrap = ctx.vector.read(0)[0] as i16;
        assert_eq!(lo16_nowrap, -32768);
    }

    #[test]
    fn test_vector_srs_from_type_masks_accumulator() {
        let mut ctx = make_ctx();
        ctx.accumulator.write(0, [
            0xDEAD_BEEF_0000_0064, 0, 0, 0, 0, 0, 0, 0,
        ]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int16)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(0));
        op.from_type = Some(ElementType::Int32);

        ctx.srs_config.rounding_mode = 0;
        ctx.srs_config.saturation_mode = 1;
        ctx.srs_config.srs_sign = true;

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(0);
        let lo16 = result[0] as i16;
        assert_eq!(lo16, 100, "from_type=Int32 should mask to low 32 bits");
    }

    #[test]
    fn test_wide_srs_cm_to_x() {
        let mut ctx = make_ctx();

        let acc_lo: [u64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let acc_hi: [u64; 8] = [90, 100, 110, 120, 130, 140, 150, 160];
        ctx.accumulator.write(0, acc_lo);
        ctx.accumulator.write(1, acc_hi);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(4))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(0));
        op.from_type = Some(ElementType::Int64);
        op.is_wide_vector = true;

        ctx.srs_config.rounding_mode = 0;
        ctx.srs_config.saturation_mode = 0;
        ctx.srs_config.srs_sign = true;

        VectorAlu::execute(&op, &mut ctx);

        let v4 = ctx.vector.read(4);
        assert_eq!(v4, [10, 20, 30, 40, 50, 60, 70, 80]);
        let v5 = ctx.vector.read(5);
        assert_eq!(v5, [90, 100, 110, 120, 130, 140, 150, 160]);
    }

    #[test]
    fn test_srs_from_acc32_to_d16() {
        let mut acc = [0u64; 8];
        for i in 0..8usize {
            let lo = (100 + i * 2) as u64;
            let hi = (100 + i * 2 + 1) as u64;
            acc[i] = lo | (hi << 32);
        }

        let cfg = SrsConfig {
            rounding_mode: 0,
            saturation_mode: 0,
            srs_sign: true,
        };

        let result = VectorAlu::vector_srs_from_acc(
            &acc, 0, ElementType::Int32, ElementType::Int16, &cfg,
        );

        for i in 0..8 {
            let expected_lo = (100 + i * 2) as u16;
            let expected_hi = (100 + i * 2 + 1) as u16;
            let expected = (expected_lo as u32) | ((expected_hi as u32) << 16);
            assert_eq!(result[i], expected, "result[{}]: got {:#010x}, expected {:#010x}",
                i, result[i], expected);
        }
    }

    #[test]
    fn test_srs_bf16_truncates_f32() {
        // BFloat16 SRS: each u64 holds two f32, truncated to bf16.
        // f32 1.0 = 0x3F800000 -> bf16 = 0x3F80
        // f32 2.0 = 0x40000000 -> bf16 = 0x4000
        let mut acc = [0u64; 8];
        acc[0] = 0x3F80_0000u64 | (0x4000_0000u64 << 32); // lo=1.0, hi=2.0
        acc[1] = 0x4040_0000u64 | (0x4080_0000u64 << 32); // lo=3.0, hi=4.0

        let cfg = SrsConfig {
            rounding_mode: 0,
            saturation_mode: 0,
            srs_sign: true,
        };

        let result = VectorAlu::vector_srs_from_acc(
            &acc, 0, ElementType::Int32, ElementType::BFloat16, &cfg,
        );

        // result[0] = bf16(1.0) | bf16(2.0) << 16 = 0x3F80 | 0x4000_0000
        assert_eq!(result[0] & 0xFFFF, 0x3F80); // bf16 for 1.0
        assert_eq!(result[0] >> 16, 0x4000);     // bf16 for 2.0
        assert_eq!(result[1] & 0xFFFF, 0x4040);  // bf16 for 3.0
        assert_eq!(result[1] >> 16, 0x4080);      // bf16 for 4.0
    }

    #[test]
    fn test_srs_float32_passthrough() {
        // Float32 SRS: passes through the low 32 bits of each u64.
        let mut acc = [0u64; 8];
        acc[0] = 0xDEAD_BEEF_3F80_0000; // low 32 = 0x3F800000 (1.0f)
        acc[1] = 0xCAFE_BABE_4000_0000; // low 32 = 0x40000000 (2.0f)

        let cfg = SrsConfig {
            rounding_mode: 0,
            saturation_mode: 0,
            srs_sign: true,
        };

        let result = VectorAlu::vector_srs_from_acc(
            &acc, 0, ElementType::Int32, ElementType::Float32, &cfg,
        );

        assert_eq!(result[0], 0x3F80_0000); // 1.0f
        assert_eq!(result[1], 0x4000_0000); // 2.0f
    }

    #[test]
    fn test_srs_from_acc64_to_d16() {
        // Acc64 -> 16-bit: 8 u64 lanes, SRS'd to 16-bit, packed 2 per u32.
        let mut acc = [0u64; 8];
        for i in 0..8usize {
            acc[i] = (100 + i) as u64;
        }

        let cfg = SrsConfig {
            rounding_mode: 0,
            saturation_mode: 0,
            srs_sign: true,
        };

        let result = VectorAlu::vector_srs_from_acc(
            &acc, 0, ElementType::Int64, ElementType::Int16, &cfg,
        );

        // 8 lanes packed 2 per word = 4 words
        for i in 0..4 {
            let expected_lo = (100 + i * 2) as u16;
            let expected_hi = (100 + i * 2 + 1) as u16;
            let expected = (expected_lo as u32) | ((expected_hi as u32) << 16);
            assert_eq!(result[i], expected, "result[{}]", i);
        }
    }

    #[test]
    fn test_srs_from_acc32_to_d8() {
        let mut acc = [0u64; 8];
        for i in 0..8usize {
            let lo = (10 + i * 2) as u64;
            let hi = (10 + i * 2 + 1) as u64;
            acc[i] = lo | (hi << 32);
        }

        let cfg = SrsConfig {
            rounding_mode: 0,
            saturation_mode: 0,
            srs_sign: true,
        };

        let result = VectorAlu::vector_srs_from_acc(
            &acc, 0, ElementType::Int32, ElementType::Int8, &cfg,
        );

        for i in 0..4 {
            let mut expected = 0u32;
            for j in 0..4 {
                let lane_idx = i * 4 + j;
                let val = (10 + lane_idx) as u8;
                expected |= (val as u32) << (j * 8);
            }
            assert_eq!(result[i], expected, "result[{}]: got {:#010x}, expected {:#010x}",
                i, result[i], expected);
        }
    }
}
