//! AIE2 floating-point utilities for BF16 and FP32.
//!
//! Centralizes all floating-point type conversions, rounding, NaN propagation,
//! and saturation behavior specific to the AIE2 vector unit. AIE2's float
//! handling differs from strict IEEE 754 in several key ways:
//!
//! - **Flush-to-zero (FTZ)**: Denormalized values (exponent == 0, mantissa != 0)
//!   are flushed to signed zero. This applies to both bf16 and fp32 operands
//!   in the multiply-accumulate datapath.
//!
//! - **Canonical NaN**: AIE2 produces a specific NaN bit pattern for all NaN
//!   results: `sign | 0x7F800000 | 0x7F`. The mantissa is 0x7F (lowest 7 bits
//!   set), which differs from the IEEE 754 convention of setting the quiet NaN
//!   indicator bit (bit 22). AIE2 does not distinguish quiet and signaling NaN.
//!
//! - **Rounding for bf16 conversion**: Uses the same 10-mode rounding logic as
//!   the integer SRS pipeline. The `srs_bf_lane` path applies rounding with
//!   `sgn_mag=true`, which inverts the symmetry direction compared to the
//!   integer path.
//!
//! All behavioral facts are derived from the AIE2 hardware model in the
//! open-source toolchain and observed NPU hardware behavior.

use xdna_archspec::aie2::rounding::RoundingMode;

// ---------------------------------------------------------------------------
// BF16 bit layout constants
//
// BFloat16 shares fp32's exponent range (8 bits) but has only 7 mantissa bits:
//   [15]    sign
//   [14:7]  exponent (8 bits, bias 127)
//   [6:0]   mantissa (7 bits, implicit leading 1)
// ---------------------------------------------------------------------------

const BF16_SIGN_BIT: u16 = 1 << 15;
const BF16_EXP_MASK: u16 = 0xFF << 7;
const BF16_MAN_MASK: u16 = 0x7F;

// FP32 bit layout constants.
const FP32_SIGN_BIT: u32 = 1 << 31;
const FP32_EXP_MASK: u32 = 0xFF << 23;
const FP32_MAN_MASK: u32 = 0x7F_FFFF;

// ---------------------------------------------------------------------------
// BF16 type utilities
// ---------------------------------------------------------------------------

/// Split a bf16 bit pattern into (sign, exponent, mantissa).
///
/// Matches the `bf16_split` function in the AIE2 hardware model.
#[inline]
pub fn bf16_split(bits: u16) -> (bool, u8, u8) {
    let sign = (bits & BF16_SIGN_BIT) != 0;
    let exp = ((bits & BF16_EXP_MASK) >> 7) as u8;
    let man = (bits & BF16_MAN_MASK) as u8;
    (sign, exp, man)
}

/// Assemble a bf16 bit pattern from (sign, exponent, mantissa).
///
/// Matches the `bf16_make` function in the AIE2 hardware model.
#[inline]
pub fn bf16_make(sign: bool, exp: u8, man: u8) -> u16 {
    ((sign as u16) << 15) | ((exp as u16) << 7) | ((man as u16) & BF16_MAN_MASK)
}

/// Split an fp32 bit pattern into (sign, exponent, mantissa).
///
/// Matches the `fp32_split` function in the AIE2 hardware model.
#[inline]
pub fn fp32_split(bits: u32) -> (bool, u8, u32) {
    let sign = (bits & FP32_SIGN_BIT) != 0;
    let exp = ((bits & FP32_EXP_MASK) >> 23) as u8;
    let man = bits & FP32_MAN_MASK;
    (sign, exp, man)
}

/// Assemble an fp32 bit pattern from (sign, exponent, mantissa).
///
/// Matches the `fp32_make` function in the AIE2 hardware model.
#[inline]
pub fn fp32_make(sign: bool, exp: u8, man: u32) -> u32 {
    ((sign as u32) << 31) | ((exp as u32) << 23) | (man & FP32_MAN_MASK)
}

/// Convert bf16 bit pattern to f32.
///
/// This is an exact conversion with no precision loss: bf16's 7-bit mantissa
/// fits entirely within fp32's 23-bit mantissa. The conversion simply pads
/// the mantissa with 16 zero bits.
///
/// Matches `bf16_to_float` in the hardware model: `fp32_to_float(a << 16)`.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to bf16 with specified rounding mode.
///
/// Applies the same rounding logic as the SRS instruction's bf16 lane path
/// (`srs_bf_lane` in the hardware model). The guard, sticky, and LSB bits
/// are extracted from the 16 mantissa bits that will be discarded.
///
/// For the special case where the input is NaN but truncation would lose
/// the NaN status (mantissa bits only in the low 16 bits), the result is
/// incremented to preserve NaN-ness.
///
/// Rounding behavior per AIE2 hardware model (`srs_bf_lane`).
pub fn f32_to_bf16(value: f32, mode: RoundingMode) -> u16 {
    let bits = value.to_bits();
    let (sign, exp, man) = fp32_split(bits);

    // Start with simple truncation (drop lower 16 bits).
    let mut result = (bits >> 16) as u16;

    if exp != 255 {
        // Normal or zero: apply rounding to the truncation point.
        //
        // The 16 discarded bits provide the rounding signals:
        //   bit 15 = guard (first discarded bit)
        //   bits 14:0 = sticky source (OR of all lower bits)
        //   bit 16 = LSB of the retained bf16 mantissa
        let sticky = (man & 0x7FFF) != 0;
        let guard = (man >> 15) & 1 != 0;
        let lsb = (man >> 16) & 1 != 0;

        // The bf16 conversion path uses sgn_mag=true in the hardware model,
        // which inverts the symmetry behavior for symmetric rounding modes.
        // This means for symmetric modes, the sign is effectively flipped
        // before the rounding decision. We use `srs_round_bf16` to handle this.
        if srs_round_bf16(mode, sign, lsb, guard, sticky) {
            result = result.wrapping_add(1);
        }
    } else if man != 0 {
        // Input is NaN. If truncation would make the mantissa zero (losing
        // NaN status), increment to preserve it. This handles the case where
        // all nonzero mantissa bits are in the lower 16 bits.
        let result_man = result & BF16_MAN_MASK;
        if result_man == 0 {
            result = result.wrapping_add(1);
        }
    }
    // else: infinity (exp=255, man=0) -- truncation is exact, no rounding needed.

    result
}

/// Convert f32 to bf16 with simple truncation (no rounding).
///
/// Drops the lower 16 bits of the fp32 representation. This is the behavior
/// of `float_to_bf16` in the hardware model and is used by the MAC reference
/// paths.
#[inline]
pub fn f32_to_bf16_truncate(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// Check if a bf16 value is denormalized (exponent == 0, mantissa != 0).
///
/// Denormalized bf16 values have an exponent of zero and a nonzero mantissa.
/// True zero (exp=0, man=0) is NOT denormalized.
#[inline]
pub fn bf16_is_denorm(bits: u16) -> bool {
    let (_, exp, man) = bf16_split(bits);
    exp == 0 && man != 0
}

/// Check if an fp32 value is denormalized.
#[inline]
pub fn fp32_is_denorm(bits: u32) -> bool {
    let (_, exp, man) = fp32_split(bits);
    exp == 0 && man != 0
}

/// Flush a denormalized bf16 value to signed zero.
///
/// If the exponent is zero, the mantissa is cleared. The sign bit is
/// preserved, so negative denorms become -0.0.
///
/// Matches `bf16_denorm_to_0` in the hardware model: when exp==0,
/// the mantissa is zeroed regardless of its value.
#[inline]
pub fn bf16_flush_to_zero(bits: u16) -> u16 {
    let (sign, exp, man) = bf16_split(bits);
    if exp == 0 {
        bf16_make(sign, 0, 0)
    } else {
        bf16_make(sign, exp, man)
    }
}

/// Flush a denormalized fp32 value to signed zero.
///
/// Matches `fp32_denorm_to_0` in the hardware model.
#[inline]
pub fn fp32_flush_to_zero(bits: u32) -> u32 {
    let (sign, exp, man) = fp32_split(bits);
    if exp == 0 {
        fp32_make(sign, 0, 0)
    } else {
        fp32_make(sign, exp, man)
    }
}

/// Check if a bf16 value is NaN (exponent == 255, mantissa != 0).
#[inline]
pub fn bf16_is_nan(bits: u16) -> bool {
    let (_, exp, man) = bf16_split(bits);
    exp == 255 && man != 0
}

/// Check if an fp32 bit pattern is NaN.
#[inline]
pub fn fp32_is_nan(bits: u32) -> bool {
    let (_, exp, man) = fp32_split(bits);
    exp == 255 && man != 0
}

/// Check if a bf16 value is infinity (exponent == 255, mantissa == 0).
#[inline]
pub fn bf16_is_inf(bits: u16) -> bool {
    let (_, exp, man) = bf16_split(bits);
    exp == 255 && man == 0
}

/// Check if an fp32 bit pattern is infinity.
#[inline]
pub fn fp32_is_inf(bits: u32) -> bool {
    let (_, exp, man) = fp32_split(bits);
    exp == 255 && man == 0
}

// ---------------------------------------------------------------------------
// AIE2 canonical NaN and infinity construction
// ---------------------------------------------------------------------------

/// Create the AIE2 canonical NaN bit pattern (fp32).
///
/// AIE2 produces a specific NaN for all NaN results: mantissa = 1 (only bit 0
/// set), sign = 0 (always positive).  This was verified against real NPU1
/// hardware by observing `vadd.f` on NaN inputs through the SRS path.
///
/// The aietools Python model (`bfloat16.py:fp32_make_nan`) uses mantissa =
/// 0x7F (2^7-1), which matches the BF16 canonical NaN extended to f32.  The
/// real silicon produces mantissa = 1 instead.  Since hardware is ground
/// truth, we use mantissa = 1.
#[inline]
pub fn fp32_make_nan(sign: bool) -> u32 {
    ((sign as u32) << 31) | (0xFF << 23) | 0x01
}

/// Create an fp32 infinity bit pattern.
///
/// Matches `fp32_make_inf(sgn)` in the hardware model.
#[inline]
pub fn fp32_make_inf(sign: bool) -> u32 {
    ((sign as u32) << 31) | (0xFF << 23)
}

/// Create the AIE2 canonical NaN as an f32 value.
#[inline]
pub fn aie2_canonical_nan(sign: bool) -> f32 {
    f32::from_bits(fp32_make_nan(sign))
}

// ---------------------------------------------------------------------------
// AIE2 NaN propagation
// ---------------------------------------------------------------------------

/// AIE2 NaN propagation for binary operations.
///
/// When either operand is NaN, the result is the AIE2 canonical NaN
/// (positive, unsigned). The hardware model always returns
/// `fp32_make_nan(False)` -- a positive canonical NaN -- regardless of
/// which operand was NaN or what its sign was.
///
/// If neither operand is NaN, returns `None` (the caller should compute
/// the normal result).
pub fn aie2_nan_propagate(a: f32, b: f32) -> Option<f32> {
    if a.is_nan() || b.is_nan() {
        // AIE2 always produces positive canonical NaN, regardless of input signs.
        Some(aie2_canonical_nan(false))
    } else {
        None
    }
}

/// Canonicalize a NaN value to the AIE2 canonical form.
///
/// If the input is NaN, returns the AIE2 canonical NaN (positive, mantissa
/// = 0x7F). If the input is not NaN, returns it unchanged.
///
/// AIE2 does not preserve NaN payloads or distinguish quiet vs signaling.
pub fn aie2_canonicalize_nan(value: f32) -> f32 {
    if value.is_nan() {
        aie2_canonical_nan(false)
    } else {
        value
    }
}

// ---------------------------------------------------------------------------
// AIE2 fp32 arithmetic (FTZ + NaN propagation)
// ---------------------------------------------------------------------------

/// Perform fp32 addition treating NaN bit patterns (exp=0xFF, man!=0) as valid
/// floats with exponent 255. Uses f64 intermediate precision to avoid IEEE NaN
/// semantics in the host's f32 arithmetic.
///
/// The AIE2 accumulator ALU does not short-circuit on NaN when both operands
/// are NaN. Instead, both values have the same exponent (255), so the hardware
/// subtracts/adds their mantissas normally, producing a result with a lower
/// exponent. If the result overflows back to exponent 255 (e.g., NaN + NaN
/// where both have the same sign), the hardware produces canonical NaN instead.
/// Verified against real NPU1 hardware.
///
/// Returns the raw fp32 result (caller should OR bit 0 if result exp < 255).
/// Returns canonical NaN if the result overflows to exp >= 255.
#[inline]
pub fn fp32_add_treat_nan_as_normal(a_bits: u32, b_bits: u32) -> u32 {
    // Decode both fp32 bit patterns into f64, treating exp=255 as valid.
    let a_f64 = fp32_bits_to_f64_raw(a_bits);
    let b_f64 = fp32_bits_to_f64_raw(b_bits);
    let result = a_f64 + b_f64;
    let result_bits = f64_to_fp32_bits_raw(result);

    // If the result overflows to exp >= 255 (inf/NaN range), the hardware
    // falls back to canonical NaN. This happens when both NaN have the same
    // sign (addition doubles the magnitude, overflowing the exponent).
    let result_exp = (result_bits >> 23) & 0xFF;
    if result_exp >= 255 {
        return fp32_make_nan(false);
    }

    result_bits
}

/// Decode an fp32 bit pattern to f64, treating ALL exponent values (including
/// 255) as valid biased exponents. This bypasses IEEE NaN/infinity semantics.
#[inline]
fn fp32_bits_to_f64_raw(bits: u32) -> f64 {
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7FFFFF;

    if exp == 0 && man == 0 {
        return if sign != 0 { -0.0 } else { 0.0 };
    }

    // Treat exp=0 as denorm, all others (including 255) as normal.
    let (frac, e) = if exp == 0 {
        // Denormalized: no implicit leading 1, exponent = 1 - bias.
        ((man as f64) / ((1u64 << 23) as f64), -126i32)
    } else {
        (1.0 + (man as f64) / ((1u64 << 23) as f64), exp - 127)
    };

    let value = frac * (2.0f64).powi(e);
    if sign != 0 {
        -value
    } else {
        value
    }
}

/// Convert an f64 value back to fp32 bit pattern, allowing the full exponent
/// range. Values that overflow (biased exp >= 255) produce infinity (exp=255,
/// man=0) so the caller can detect and handle overflow.
#[inline]
fn f64_to_fp32_bits_raw(value: f64) -> u32 {
    if value == 0.0 {
        return if value.is_sign_negative() {
            0x80000000
        } else {
            0x00000000
        };
    }

    let sign = value < 0.0;
    let abs_val = value.abs();

    // Compute exponent: floor(log2(abs_val)).
    let exp_unbiased = abs_val.log2().floor() as i32;
    let biased_exp = exp_unbiased + 127;

    if biased_exp >= 255 {
        // Overflow: return infinity so caller can detect.
        return fp32_make_inf(sign);
    }
    if biased_exp <= 0 {
        // Underflow to signed zero.
        return if sign { 0x80000000 } else { 0x00000000 };
    }

    // Extract mantissa: abs_val / 2^exp_unbiased gives 1.fraction.
    let significand = abs_val / (2.0f64).powi(exp_unbiased);
    let man_f = (significand - 1.0) * ((1u64 << 23) as f64);
    let mut man = (man_f + 0.5) as u32; // Round to nearest.

    // Handle rounding overflow (man rounds up to 2^23).
    if man >= (1u32 << 23) {
        man = 0;
        let biased_exp = biased_exp + 1;
        if biased_exp >= 255 {
            return fp32_make_inf(sign);
        }
        return fp32_make(sign, biased_exp as u8, man);
    }

    fp32_make(sign, biased_exp as u8, man)
}

/// AIE2 fp32 addition with hardware-accurate FTZ and NaN handling.
///
/// AIE2 float operations differ from IEEE 754:
/// 1. Inputs are flushed to signed zero if denormalized (exp=0, man!=0)
/// 2. If exactly one input is NaN, result is positive canonical NaN
/// 3. If both inputs are NaN, the hardware treats exponent 0xFF as a valid
///    exponent and computes the addition normally (no NaN short-circuit).
///    The result's mantissa bit 0 is set as a NaN-origin sticky flag.
///    Verified against real NPU1 hardware via ISA test harness.
/// 4. Normal IEEE 754 addition otherwise
/// 5. Result is FTZ'd again (denormalized results flushed to signed zero)
///
/// Derived from aietools python_model/model/mulmac.py and constants.py,
/// with both-NaN behavior from NPU hardware observation.
#[inline]
pub fn aie2_fp32_add(a_bits: u32, b_bits: u32) -> u32 {
    // Step 1: Flush denormalized inputs to signed zero.
    let a_ftz = fp32_flush_to_zero(a_bits);
    let b_ftz = fp32_flush_to_zero(b_bits);

    let a_nan = fp32_is_nan(a_ftz);
    let b_nan = fp32_is_nan(b_ftz);

    // Step 2: NaN handling.
    if a_nan && b_nan {
        // Both NaN: hardware computes as if exp=255 is valid, then sets bit 0.
        let result = fp32_add_treat_nan_as_normal(a_ftz, b_ftz);
        return fp32_flush_to_zero(result | 0x01);
    }
    if a_nan || b_nan {
        return fp32_make_nan(false); // Positive canonical NaN
    }

    // Step 3: Normal float addition.
    let a = f32::from_bits(a_ftz);
    let b = f32::from_bits(b_ftz);
    let result = a + b;

    // Step 4: Flush denormalized result to signed zero.
    fp32_flush_to_zero(result.to_bits())
}

/// AIE2 fp32 accumulator ALU addition (VADD.f / VSUB.f datapath).
///
/// Bit-exact model of the hardware mantissa datapath (bfshiftcompute /
/// bfaccshift / bfnorm lane pipeline), validated against NPU1 silicon via
/// the vector differential fuzzer (seeds 4/8/34/41/47, bf16 chains with
/// denormal/NaN/Inf-heavy inputs). Key behaviors that differ from IEEE:
///
/// - Denormal INPUTS are not flushed: they enter the mantissa datapath
///   without an implicit leading 1 (0x80520000 + 0x807E0000 -> exact
///   normal sum 0x80D00000 on silicon).
/// - Denormal RESULTS underflow to signed zero (sign of the true sum).
/// - NaN inputs are flags, not short-circuits: the exponent-255 operand
///   runs through the same datapath. The result has exp=255 forced,
///   mantissa = datapath sum (zeroed on overflow/inf) with bit 0 set as
///   the NaN sticky, and the sum's sign (so -0 - NaN = 0xFF800001-class).
/// - +Inf + -Inf is NaN (mantissa datapath cancels to zero, bit 0 sticks).
///
/// Used by VADD_F, VSUB_F, VNEGSUB_F, and VNEGADD_F (the accumulator
/// add/subtract instructions); callers apply operand negation by sign flip
/// (zeros excluded).
pub fn aie2_acc_fp32_add(a_bits: u32, b_bits: u32) -> u32 {
    let dec = |bits: u32| -> (bool, u32, u32) { ((bits >> 31) != 0, (bits >> 23) & 0xFF, bits & 0x7F_FFFF) };
    let (asgn, aexp, aman) = dec(a_bits);
    let (bsgn, bexp, bman) = dec(b_bits);

    let a_zero = aexp == 0 && aman == 0;
    let b_zero = bexp == 0 && bman == 0;
    let a_inf = aexp == 255 && aman == 0;
    let b_inf = bexp == 255 && bman == 0;
    let a_nan = aexp == 255 && aman != 0;
    let b_nan = bexp == 255 && bman != 0;

    // Flag pre-compute (bfshiftcompute_lane): inf becomes a signed flag,
    // opposing infinities raise NaN.
    let pinf = (a_inf && !asgn) || (b_inf && !bsgn);
    let minf = (a_inf && asgn) || (b_inf && bsgn);
    let nan = a_nan || b_nan || (pinf && minf);

    // Biased 9-bit exponent: +0x80 for nonzero, low bit forced for denorms.
    let exp9 = |zero: bool, exp: u32, man: u32| -> u32 {
        if zero {
            0
        } else {
            (exp + 0x80) | u32::from(exp == 0 && man != 0)
        }
    };
    let aexp9 = exp9(a_zero, aexp, aman);
    let bexp9 = exp9(b_zero, bexp, bman);
    let omax = aexp9.max(bexp9);

    // 24-bit mantissa (implicit 1 only when exp != 0), aligned to omax with
    // round-to-nearest-even on the shifted-out bits (bfaccshift_lane).
    let align = |exp: u32, man: u32, exp9v: u32| -> i64 {
        let man24 = if exp != 0 { (1 << 23) | man } else { man } as u64;
        let s = omax - exp9v;
        let v = man24 << 1; // 25-bit
        let e = if s >= 64 { 0 } else { v >> s };
        let r = e >> 1;
        let grd = e & 1 != 0;
        let lsb = (e >> 1) & 1 != 0;
        let rmask = if s >= 64 { u64::MAX } else { !(u64::MAX << s) };
        let stk = man24 & (rmask >> 1) != 0;
        (r + u64::from(grd && (stk || lsb))) as i64
    };
    let sum = {
        let av = align(aexp, aman, aexp9);
        let bv = align(bexp, bman, bexp9);
        (if asgn { -av } else { av }) + (if bsgn { -bv } else { bv })
    };

    // Normalize (bfnorm_lane): 29-bit magnitude, RNE to 23-bit mantissa.
    let sgn_sum = sum < 0;
    let m = sum.unsigned_abs();
    let real_zero = m == 0;
    let fl: i32 = if real_zero {
        32
    } else {
        28 - (63 - m.leading_zeros() as i32)
    };
    let d = if real_zero { 0u64 } else { m << (fl & 0x1F) };
    let grd = (d >> 4) & 1 != 0;
    let lsb = (d >> 5) & 1 != 0;
    let stk = d & 0xF != 0;
    let rup = grd && (stk || lsb);
    let r = (((d >> 5) & 0x7F_FFFF) + u64::from(rup)) & 0x7F_FFFF;
    let exp_up = (d >> 4) & 0xFF_FFFF == 0xFF_FFFF;
    let exp = omax as i32 - 0x7B - fl + i32::from(exp_up);

    let overflow = exp >= 255 && !real_zero;
    let underflow = exp <= 0 && !real_zero;
    let out_zeros = !(pinf || minf || overflow || underflow);
    let exp_ones = nan || pinf || minf || overflow;
    let exp_zeros = !(underflow || real_zero);

    let sgn_out = if !nan && (pinf || minf) { minf } else { sgn_sum };
    let rr = (if out_zeros { r as u32 } else { 0 }) | u32::from(nan);
    let e8 = if exp_ones {
        0xFF
    } else if exp_zeros {
        exp as u32 & 0xFF
    } else {
        0
    };
    (u32::from(sgn_out) << 31) | (e8 << 23) | rr
}

// ---------------------------------------------------------------------------
// AIE2 rounding for bf16 conversion (SRS bf16 lane path)
// ---------------------------------------------------------------------------

/// Compute the rounding increment for the bf16 SRS lane path.
///
/// This is the same `srs_round` logic used by the integer SRS pipeline,
/// but with `sgn_mag=true`. The `sgn_mag` flag inverts the symmetric
/// direction: for symmetric modes (SymFloor, SymCeil, SymInf, SymZero),
/// the sign-based tiebreaker is flipped.
///
/// Rounding behavior per AIE2 hardware model (`srs_round` with `sgn_mag=True`).
fn srs_round_bf16(mode: RoundingMode, sign: bool, lsb: bool, guard: bool, sticky: bool) -> bool {
    // Classify the rounding mode.
    let is_halfway = matches!(
        mode,
        RoundingMode::NegInf
            | RoundingMode::PosInf
            | RoundingMode::SymInf
            | RoundingMode::SymZero
            | RoundingMode::ConvEven
            | RoundingMode::ConvOdd
    );

    let mut symmetric = matches!(
        mode,
        RoundingMode::SymFloor | RoundingMode::SymCeil | RoundingMode::SymInf | RoundingMode::SymZero
    );

    let otherdir = matches!(
        mode,
        RoundingMode::Ceil
            | RoundingMode::SymCeil
            | RoundingMode::PosInf
            | RoundingMode::SymInf
            | RoundingMode::ConvOdd
    );

    let convergent = matches!(mode, RoundingMode::ConvEven | RoundingMode::ConvOdd);

    // sgn_mag=true inverts the symmetric flag.
    // This is the key difference from the integer SRS path: the bf16
    // conversion path operates in sign-magnitude representation, so
    // the symmetry direction must be inverted.
    symmetric = !symmetric;

    // Compute the determinant (direction signal).
    let det = if convergent {
        lsb
    } else if symmetric {
        sign
    } else {
        false
    };

    let det = if otherdir { !det } else { det };

    // Final rounding decision.
    if is_halfway {
        guard && (det || sticky)
    } else {
        det && (guard || sticky)
    }
}

// ---------------------------------------------------------------------------
// Saturation
// ---------------------------------------------------------------------------

/// Saturate an fp32 value to the bf16 representable range.
///
/// BF16 shares fp32's exponent range, so the maximum finite bf16 value is
/// `sign | 0x7F7F` as bf16 = `sign | 0x7F7F0000` as fp32.
/// That is: exponent=254, mantissa=all-ones-in-7-bits = 3.3895314e+38.
///
/// Values exceeding this magnitude are clamped to the maximum finite bf16
/// (not infinity). Values within range are returned unchanged. NaN and
/// infinity pass through unmodified.
pub fn saturate_f32_to_bf16_range(value: f32) -> f32 {
    if value.is_nan() || value.is_infinite() {
        return value;
    }

    // Maximum finite bf16: exponent=254, mantissa=0x7F (all 7 bits set).
    // As fp32: 0x7F7F0000 (positive) = bf16 0x7F7F.
    let max_bf16 = f32::from_bits(0x7F7F_0000);

    if value > max_bf16 {
        max_bf16
    } else if value < -max_bf16 {
        -max_bf16
    } else {
        value
    }
}

// ---------------------------------------------------------------------------
// Accumulator float helpers (used by matmul and other vector ops)
// ---------------------------------------------------------------------------

/// Read an fp32 value from an accumulator lane (stored as u32 in a u64 pair).
///
/// The 8 u64 accumulator lanes hold 16 fp32 values. Each u64 contains two
/// fp32 values: low 32 bits = even index, high 32 bits = odd index.
#[inline]
pub fn read_acc_f32(acc: &[u64; 8], index: usize) -> f32 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    f32::from_bits(bits)
}

/// Write an fp32 value to an accumulator lane.
#[inline]
pub fn write_acc_f32(acc: &mut [u64; 8], index: usize, value: f32) {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = value.to_bits() as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (bits << shift);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // bf16 split and make roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_split_make_roundtrip() {
        // Test that splitting and reassembling preserves the bit pattern.
        for bits in [0x0000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FFF, 0xFFFF, 0x0001] {
            let (s, e, m) = bf16_split(bits);
            assert_eq!(bf16_make(s, e, m), bits, "roundtrip failed for 0x{:04X}", bits);
        }
    }

    #[test]
    fn test_bf16_split_known_values() {
        // bf16 +1.0 = 0x3F80: sign=0, exp=127, man=0
        let (s, e, m) = bf16_split(0x3F80);
        assert!(!s);
        assert_eq!(e, 127);
        assert_eq!(m, 0);

        // bf16 -1.0 = 0xBF80
        let (s, e, m) = bf16_split(0xBF80);
        assert!(s);
        assert_eq!(e, 127);
        assert_eq!(m, 0);

        // bf16 +inf = 0x7F80
        let (s, e, m) = bf16_split(0x7F80);
        assert!(!s);
        assert_eq!(e, 255);
        assert_eq!(m, 0);

        // bf16 zero = 0x0000
        let (s, e, m) = bf16_split(0x0000);
        assert!(!s);
        assert_eq!(e, 0);
        assert_eq!(m, 0);
    }

    #[test]
    fn test_fp32_split_make_roundtrip() {
        for bits in [
            0x0000_0000u32,
            0x3F80_0000,
            0xBF80_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
            0x0000_0001,
            0x7F7F_FFFF,
        ] {
            let (s, e, m) = fp32_split(bits);
            assert_eq!(fp32_make(s, e, m), bits, "roundtrip failed for 0x{:08X}", bits);
        }
    }

    // -----------------------------------------------------------------------
    // bf16 <-> f32 conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_to_f32_exact() {
        // bf16 +1.0 (0x3F80) -> fp32 +1.0 (0x3F800000)
        assert_eq!(bf16_to_f32(0x3F80), 1.0f32);

        // bf16 -1.0 (0xBF80) -> fp32 -1.0
        assert_eq!(bf16_to_f32(0xBF80), -1.0f32);

        // bf16 +2.0 (0x4000) -> fp32 +2.0
        assert_eq!(bf16_to_f32(0x4000), 2.0f32);

        // bf16 +0.0 (0x0000) -> fp32 +0.0
        assert_eq!(bf16_to_f32(0x0000), 0.0f32);
        assert_eq!(bf16_to_f32(0x0000).to_bits(), 0x0000_0000);

        // bf16 -0.0 (0x8000) -> fp32 -0.0
        assert!(bf16_to_f32(0x8000).is_sign_negative());
        assert_eq!(bf16_to_f32(0x8000).to_bits(), 0x8000_0000);
    }

    #[test]
    fn test_bf16_to_f32_infinity() {
        // +inf
        assert!(bf16_to_f32(0x7F80).is_infinite());
        assert!(bf16_to_f32(0x7F80).is_sign_positive());

        // -inf
        assert!(bf16_to_f32(0xFF80).is_infinite());
        assert!(bf16_to_f32(0xFF80).is_sign_negative());
    }

    #[test]
    fn test_bf16_to_f32_nan() {
        // Any bf16 NaN -> fp32 NaN
        assert!(bf16_to_f32(0x7F81).is_nan());
        assert!(bf16_to_f32(0x7FFF).is_nan());
        assert!(bf16_to_f32(0xFFFF).is_nan());
    }

    #[test]
    fn test_f32_to_bf16_truncate_basic() {
        // Simple truncation drops the lower 16 bits.
        assert_eq!(f32_to_bf16_truncate(1.0f32), 0x3F80);
        assert_eq!(f32_to_bf16_truncate(-1.0f32), 0xBF80);
        assert_eq!(f32_to_bf16_truncate(0.0f32), 0x0000);
        assert_eq!(f32_to_bf16_truncate(2.0f32), 0x4000);
    }

    #[test]
    fn test_f32_to_bf16_rounding_conv_even() {
        // Test the ConvEven (IEEE banker's rounding) mode.
        //
        // 1.0 in fp32: 0x3F800000 -- no fractional bits to round.
        let r = f32_to_bf16(1.0f32, RoundingMode::ConvEven);
        assert_eq!(r, 0x3F80);

        // -1.0: same, no fractional bits.
        let r = f32_to_bf16(-1.0f32, RoundingMode::ConvEven);
        assert_eq!(r, 0xBF80);

        // 0.0: no bits to round.
        let r = f32_to_bf16(0.0f32, RoundingMode::ConvEven);
        assert_eq!(r, 0x0000);
    }

    #[test]
    fn test_f32_to_bf16_truncate_mode() {
        // Floor mode (mode 0) truncates toward -infinity.
        // For positive values, this is the same as simple truncation.
        let r = f32_to_bf16(1.0f32, RoundingMode::Floor);
        assert_eq!(r, 0x3F80);
    }

    #[test]
    fn test_f32_to_bf16_nan_preservation() {
        // If fp32 NaN has mantissa bits only in the lower 16 bits, the
        // truncated bf16 would lose NaN status. The conversion must increment
        // to preserve NaN-ness.
        //
        // fp32 NaN with mantissa = 0x0001 (only bit 0 set):
        // truncation would give bf16 0x7F80 (infinity, not NaN).
        // Correction: increment to 0x7F81 (NaN).
        let nan_bits = 0x7F80_0001u32; // exponent=255, mantissa=1
        let nan = f32::from_bits(nan_bits);
        assert!(nan.is_nan());

        let r = f32_to_bf16(nan, RoundingMode::ConvEven);
        assert!(bf16_is_nan(r), "bf16 result should be NaN, got 0x{:04X}", r);
    }

    #[test]
    fn test_f32_to_bf16_inf_preserved() {
        // Infinity should truncate cleanly.
        let r = f32_to_bf16(f32::INFINITY, RoundingMode::ConvEven);
        assert_eq!(r, 0x7F80);

        let r = f32_to_bf16(f32::NEG_INFINITY, RoundingMode::ConvEven);
        assert_eq!(r, 0xFF80);
    }

    // -----------------------------------------------------------------------
    // Denorm detection and flush-to-zero
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_denorm_detection() {
        // Zero is NOT denorm.
        assert!(!bf16_is_denorm(0x0000));
        assert!(!bf16_is_denorm(0x8000)); // -0.0

        // Normal values are NOT denorm.
        assert!(!bf16_is_denorm(0x3F80)); // 1.0
        assert!(!bf16_is_denorm(0x0080)); // smallest normal bf16

        // Denorms: exponent=0, mantissa!=0.
        assert!(bf16_is_denorm(0x0001)); // smallest positive denorm
        assert!(bf16_is_denorm(0x007F)); // largest positive denorm
        assert!(bf16_is_denorm(0x8001)); // negative denorm
        assert!(bf16_is_denorm(0x807F)); // largest negative denorm

        // Infinity and NaN are NOT denorm.
        assert!(!bf16_is_denorm(0x7F80)); // +inf
        assert!(!bf16_is_denorm(0x7FFF)); // NaN
    }

    #[test]
    fn test_fp32_denorm_detection() {
        assert!(!fp32_is_denorm(0x0000_0000)); // +0
        assert!(!fp32_is_denorm(0x8000_0000)); // -0
        assert!(!fp32_is_denorm(0x3F80_0000)); // 1.0
        assert!(!fp32_is_denorm(0x0080_0000)); // smallest normal
        assert!(fp32_is_denorm(0x0000_0001)); // smallest denorm
        assert!(fp32_is_denorm(0x007F_FFFF)); // largest denorm
        assert!(fp32_is_denorm(0x8000_0001)); // negative denorm
    }

    #[test]
    fn test_bf16_flush_to_zero() {
        // Normal values pass through unchanged.
        assert_eq!(bf16_flush_to_zero(0x3F80), 0x3F80);
        assert_eq!(bf16_flush_to_zero(0x0080), 0x0080);

        // Zero passes through.
        assert_eq!(bf16_flush_to_zero(0x0000), 0x0000);
        assert_eq!(bf16_flush_to_zero(0x8000), 0x8000); // -0 preserved

        // Denorms are flushed to signed zero.
        assert_eq!(bf16_flush_to_zero(0x0001), 0x0000); // +denorm -> +0
        assert_eq!(bf16_flush_to_zero(0x007F), 0x0000);
        assert_eq!(bf16_flush_to_zero(0x8001), 0x8000); // -denorm -> -0
        assert_eq!(bf16_flush_to_zero(0x807F), 0x8000);

        // Inf and NaN are NOT flushed.
        assert_eq!(bf16_flush_to_zero(0x7F80), 0x7F80);
        assert_eq!(bf16_flush_to_zero(0x7FFF), 0x7FFF);
    }

    #[test]
    fn test_fp32_flush_to_zero() {
        assert_eq!(fp32_flush_to_zero(0x3F80_0000), 0x3F80_0000);
        assert_eq!(fp32_flush_to_zero(0x0000_0000), 0x0000_0000);
        assert_eq!(fp32_flush_to_zero(0x0000_0001), 0x0000_0000); // flush
        assert_eq!(fp32_flush_to_zero(0x8000_0001), 0x8000_0000); // preserve sign
        assert_eq!(fp32_flush_to_zero(0x007F_FFFF), 0x0000_0000); // largest denorm
    }

    #[test]
    fn test_bf16_ftz_preserves_sign() {
        // Sign must be preserved when flushing to zero.
        let flushed = bf16_flush_to_zero(0x8001);
        assert_eq!(flushed, 0x8000, "sign not preserved in FTZ");
        let f = bf16_to_f32(flushed);
        assert!(f.is_sign_negative(), "flushed -0 should be negative");
    }

    // -----------------------------------------------------------------------
    // NaN classification and canonical form
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_is_nan() {
        assert!(!bf16_is_nan(0x0000)); // zero
        assert!(!bf16_is_nan(0x3F80)); // 1.0
        assert!(!bf16_is_nan(0x7F80)); // +inf
        assert!(!bf16_is_nan(0xFF80)); // -inf
        assert!(bf16_is_nan(0x7F81)); // NaN
        assert!(bf16_is_nan(0x7FFF)); // NaN
        assert!(bf16_is_nan(0xFFFF)); // -NaN
    }

    #[test]
    fn test_fp32_is_nan() {
        assert!(!fp32_is_nan(0x0000_0000));
        assert!(!fp32_is_nan(0x7F80_0000)); // +inf
        assert!(fp32_is_nan(0x7F80_0001)); // NaN
        assert!(fp32_is_nan(0x7FC0_0000)); // qNaN
        assert!(fp32_is_nan(0xFFC0_0000)); // -qNaN
    }

    #[test]
    fn test_fp32_is_inf() {
        assert!(fp32_is_inf(0x7F80_0000)); // +inf
        assert!(fp32_is_inf(0xFF80_0000)); // -inf
        assert!(!fp32_is_inf(0x7F80_0001)); // NaN
        assert!(!fp32_is_inf(0x0000_0000)); // zero
    }

    #[test]
    fn test_canonical_nan_bits() {
        // AIE2 canonical NaN: exponent=255, mantissa=1 (HW-verified).
        let pos_nan = fp32_make_nan(false);
        assert_eq!(pos_nan, 0x7F80_0001);
        assert!(fp32_is_nan(pos_nan));

        let neg_nan = fp32_make_nan(true);
        assert_eq!(neg_nan, 0xFF80_0001);
        assert!(fp32_is_nan(neg_nan));

        // Verify the f32 representation is NaN.
        assert!(f32::from_bits(pos_nan).is_nan());
        assert!(f32::from_bits(neg_nan).is_nan());
    }

    #[test]
    fn test_fp32_make_inf_bits() {
        assert_eq!(fp32_make_inf(false), 0x7F80_0000);
        assert_eq!(fp32_make_inf(true), 0xFF80_0000);
    }

    // -----------------------------------------------------------------------
    // NaN propagation
    // -----------------------------------------------------------------------

    #[test]
    fn test_nan_propagate_both_normal() {
        assert!(aie2_nan_propagate(1.0, 2.0).is_none());
        assert!(aie2_nan_propagate(0.0, -0.0).is_none());
        assert!(aie2_nan_propagate(f32::INFINITY, 1.0).is_none());
    }

    #[test]
    fn test_nan_propagate_one_nan() {
        let nan = f32::from_bits(0x7FC0_0000);

        // NaN + normal -> canonical NaN
        let result = aie2_nan_propagate(nan, 1.0);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.is_nan());
        // Result should be the positive canonical NaN.
        assert_eq!(r.to_bits(), fp32_make_nan(false));

        // normal + NaN -> canonical NaN
        let result = aie2_nan_propagate(1.0, nan);
        assert!(result.is_some());
        assert_eq!(result.unwrap().to_bits(), fp32_make_nan(false));
    }

    #[test]
    fn test_nan_propagate_both_nan() {
        let nan1 = f32::from_bits(0x7FC0_0001);
        let nan2 = f32::from_bits(0xFFC0_0002);

        // Both NaN -> positive canonical NaN (sign and payload discarded).
        let result = aie2_nan_propagate(nan1, nan2);
        assert!(result.is_some());
        assert_eq!(result.unwrap().to_bits(), fp32_make_nan(false));
    }

    #[test]
    fn test_nan_propagate_negative_nan() {
        // Even a negative NaN input produces positive canonical NaN.
        let neg_nan = f32::from_bits(0xFFC0_0000);
        let result = aie2_nan_propagate(neg_nan, 1.0);
        assert_eq!(result.unwrap().to_bits(), fp32_make_nan(false));
    }

    #[test]
    fn test_canonicalize_nan() {
        // Normal values pass through.
        assert_eq!(aie2_canonicalize_nan(1.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(aie2_canonicalize_nan(0.0).to_bits(), 0.0f32.to_bits());

        // NaN values become canonical.
        let arbitrary_nan = f32::from_bits(0x7FFF_FFFF);
        assert!(arbitrary_nan.is_nan());
        let canonical = aie2_canonicalize_nan(arbitrary_nan);
        assert_eq!(canonical.to_bits(), fp32_make_nan(false));
    }

    // -----------------------------------------------------------------------
    // Infinity handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_infinity_roundtrip() {
        // +inf
        let inf_bf16 = f32_to_bf16_truncate(f32::INFINITY);
        assert_eq!(inf_bf16, 0x7F80);
        assert!(bf16_is_inf(inf_bf16));
        assert!(bf16_to_f32(inf_bf16).is_infinite());

        // -inf
        let neg_inf_bf16 = f32_to_bf16_truncate(f32::NEG_INFINITY);
        assert_eq!(neg_inf_bf16, 0xFF80);
        assert!(bf16_is_inf(neg_inf_bf16));
        assert!(bf16_to_f32(neg_inf_bf16).is_infinite());
        assert!(bf16_to_f32(neg_inf_bf16).is_sign_negative());
    }

    // -----------------------------------------------------------------------
    // Zero sign preservation
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_sign_preservation_bf16() {
        // +0.0
        let pos_zero = f32_to_bf16_truncate(0.0f32);
        assert_eq!(pos_zero, 0x0000);

        // -0.0
        let neg_zero = f32_to_bf16_truncate(-0.0f32);
        assert_eq!(neg_zero, 0x8000);

        // Verify f32 roundtrip preserves sign.
        assert!(bf16_to_f32(pos_zero).is_sign_positive());
        assert!(bf16_to_f32(neg_zero).is_sign_negative());
    }

    #[test]
    fn test_zero_sign_preservation_fp32() {
        let (s, e, m) = fp32_split(0x0000_0000);
        assert!(!s);
        assert_eq!(e, 0);
        assert_eq!(m, 0);

        let (s, e, m) = fp32_split(0x8000_0000);
        assert!(s);
        assert_eq!(e, 0);
        assert_eq!(m, 0);
    }

    // -----------------------------------------------------------------------
    // bf16 edge cases: max, min, smallest denorm
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_max_finite() {
        // Maximum finite bf16: exp=254, man=0x7F = bf16 0x7F7F.
        // As fp32: 0x7F7F0000 = 3.3895314e+38.
        let max_bf16 = bf16_make(false, 254, 0x7F);
        assert_eq!(max_bf16, 0x7F7F);
        let f = bf16_to_f32(max_bf16);
        assert!(f.is_finite());
        assert!(f > 0.0);

        // One exponent step higher is +inf.
        let inf_bf16 = bf16_make(false, 255, 0);
        assert!(bf16_is_inf(inf_bf16));
    }

    #[test]
    fn test_bf16_min_normal() {
        // Smallest positive normal bf16: exp=1, man=0 = bf16 0x0080.
        let min_normal = bf16_make(false, 1, 0);
        assert_eq!(min_normal, 0x0080);
        assert!(!bf16_is_denorm(min_normal));
        let f = bf16_to_f32(min_normal);
        assert!(f > 0.0);
        assert!(f.is_finite());
    }

    #[test]
    fn test_bf16_smallest_denorm() {
        // Smallest positive denorm: exp=0, man=1 = bf16 0x0001.
        let smallest = bf16_make(false, 0, 1);
        assert_eq!(smallest, 0x0001);
        assert!(bf16_is_denorm(smallest));

        // After FTZ, becomes zero.
        assert_eq!(bf16_flush_to_zero(smallest), 0x0000);
    }

    // -----------------------------------------------------------------------
    // Saturation
    // -----------------------------------------------------------------------

    #[test]
    fn test_saturate_within_range() {
        // Values within bf16 range should pass through unchanged.
        assert_eq!(saturate_f32_to_bf16_range(1.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(saturate_f32_to_bf16_range(-1.0).to_bits(), (-1.0f32).to_bits());
        assert_eq!(saturate_f32_to_bf16_range(0.0).to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn test_saturate_overflow() {
        // Max fp32 is larger than max bf16 -> clamp.
        let max_bf16_f32 = f32::from_bits(0x7F7F_0000);
        let result = saturate_f32_to_bf16_range(f32::MAX);
        assert_eq!(result, max_bf16_f32);

        // Negative overflow.
        let result = saturate_f32_to_bf16_range(f32::MIN);
        assert_eq!(result, -max_bf16_f32);
    }

    #[test]
    fn test_saturate_nan_passthrough() {
        let result = saturate_f32_to_bf16_range(f32::NAN);
        assert!(result.is_nan());
    }

    #[test]
    fn test_saturate_inf_passthrough() {
        let result = saturate_f32_to_bf16_range(f32::INFINITY);
        assert!(result.is_infinite());

        let result = saturate_f32_to_bf16_range(f32::NEG_INFINITY);
        assert!(result.is_infinite());
    }

    // -----------------------------------------------------------------------
    // Accumulator float helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_acc_f32_roundtrip() {
        let mut acc = [0u64; 8];

        // Write values to all 16 lanes and read back.
        let values = [
            1.0f32,
            -1.0,
            0.0,
            2.5,
            -3.14,
            100.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.5,
            -0.5,
            1e10,
            -1e10,
            1e-10,
            -1e-10,
            f32::MIN_POSITIVE,
            -0.0,
        ];
        for (i, &v) in values.iter().enumerate() {
            write_acc_f32(&mut acc, i, v);
        }
        for (i, &v) in values.iter().enumerate() {
            let read = read_acc_f32(&acc, i);
            assert_eq!(read.to_bits(), v.to_bits(), "lane {} mismatch: expected {:?} got {:?}", i, v, read);
        }
    }

    #[test]
    fn test_acc_f32_nan_roundtrip() {
        let mut acc = [0u64; 8];
        let canonical = aie2_canonical_nan(false);
        write_acc_f32(&mut acc, 3, canonical);
        let read = read_acc_f32(&acc, 3);
        assert_eq!(read.to_bits(), canonical.to_bits());
    }

    // -----------------------------------------------------------------------
    // SRS bf16 rounding mode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_srs_round_bf16_conv_even_ties() {
        // Convergent-even (banker's rounding) at the exact halfway point.
        // At halfway (guard=1, sticky=0):
        //   - lsb=0 (even) -> don't round up (stays even)
        //   - lsb=1 (odd) -> round up (becomes even)

        // Note: sgn_mag=true in bf16 path inverts the symmetric flag,
        // but ConvEven is convergent, not symmetric, so sgn_mag does
        // not affect it.

        // lsb=0 (even), guard=1, sticky=0 -> no round (stay even).
        assert!(!srs_round_bf16(RoundingMode::ConvEven, false, false, true, false));
        // lsb=1 (odd), guard=1, sticky=0 -> round up (become even).
        assert!(srs_round_bf16(RoundingMode::ConvEven, false, true, true, false));

        // Past halfway (guard=1, sticky=1) -> always round.
        assert!(srs_round_bf16(RoundingMode::ConvEven, false, false, true, true));
        assert!(srs_round_bf16(RoundingMode::ConvEven, false, true, true, true));

        // Below halfway (guard=0) -> never round.
        assert!(!srs_round_bf16(RoundingMode::ConvEven, false, false, false, false));
        assert!(!srs_round_bf16(RoundingMode::ConvEven, false, true, false, false));
        assert!(!srs_round_bf16(RoundingMode::ConvEven, false, false, false, true));
    }

    #[test]
    fn test_srs_round_bf16_floor() {
        // Floor (mode 0) in the bf16 path with sgn_mag=true:
        // symmetric flag is inverted (false -> true for Floor since Floor is
        // not symmetric). Wait -- Floor is not symmetric. Let me trace through:
        //   symmetric = matches!(Floor, SymFloor|SymCeil|SymInf|SymZero) = false
        //   sgn_mag inverts: symmetric = !false = true
        //   otherdir = matches!(Floor, Ceil|SymCeil|PosInf|SymInf|ConvOdd) = false
        //   convergent = false
        //   det = if symmetric { sign } = sign
        //   det = if otherdir { !det } else { det } = sign
        //   is_halfway = false
        //   result = det && (guard || sticky) = sign && (guard || sticky)
        //
        // So in bf16 Floor mode: positive values never round, negative values
        // round if any discarded bits are nonzero. This is truncation for
        // positive (toward zero = toward -inf) and round-away-from-zero for
        // negative (which is also toward -inf). Correct floor behavior.

        // Positive, any discarded bits -> no round (truncate toward -inf = toward zero).
        assert!(!srs_round_bf16(RoundingMode::Floor, false, true, true, true));
        assert!(!srs_round_bf16(RoundingMode::Floor, false, false, true, false));

        // Negative with discarded bits -> round (toward -inf = away from zero).
        assert!(srs_round_bf16(RoundingMode::Floor, true, false, true, false));
        assert!(srs_round_bf16(RoundingMode::Floor, true, true, false, true));

        // Negative, no discarded bits -> no round.
        assert!(!srs_round_bf16(RoundingMode::Floor, true, false, false, false));
    }

    #[test]
    fn test_srs_round_bf16_ceil() {
        // Ceiling (mode 1) in bf16 path with sgn_mag=true:
        //   symmetric = !false = true
        //   otherdir = true (Ceil matches)
        //   convergent = false
        //   det = if symmetric { sign } = sign
        //   det = !sign (otherdir flips)
        //   is_halfway = false
        //   result = !sign && (guard || sticky)
        //
        // Positive with discarded bits -> round up (toward +inf).
        assert!(srs_round_bf16(RoundingMode::Ceil, false, false, true, false));
        assert!(srs_round_bf16(RoundingMode::Ceil, false, true, false, true));

        // Negative with discarded bits -> no round (truncate toward zero = toward +inf).
        assert!(!srs_round_bf16(RoundingMode::Ceil, true, false, true, true));
    }

    #[test]
    fn test_f32_to_bf16_rounding_ties_example() {
        // Construct an fp32 value where the lower 16 bits are exactly at
        // the halfway point: bit 15 = 1, bits 14:0 = 0.
        //
        // Take 1.0 (0x3F800000) and set bit 15: 0x3F808000.
        // bf16 truncation gives 0x3F80 (mantissa LSB = 0, i.e. "even").
        // ConvEven at halfway with lsb=0 -> no round -> 0x3F80.
        let val = f32::from_bits(0x3F80_8000);
        let r = f32_to_bf16(val, RoundingMode::ConvEven);
        assert_eq!(r, 0x3F80, "ties-to-even: even LSB should stay");

        // Now 0x3F818000: truncated bf16 = 0x3F81 (mantissa LSB = 1, "odd").
        // ConvEven at halfway with lsb=1 -> round up -> 0x3F82.
        let val = f32::from_bits(0x3F81_8000);
        let r = f32_to_bf16(val, RoundingMode::ConvEven);
        assert_eq!(r, 0x3F82, "ties-to-even: odd LSB should round up");
    }

    #[test]
    fn test_f32_to_bf16_rounding_past_halfway() {
        // Past halfway: bit 15 = 1 AND at least one of bits 14:0 set.
        // All modes that are "halfway" should round up.
        let val = f32::from_bits(0x3F80_8001); // halfway + sticky
        let r = f32_to_bf16(val, RoundingMode::ConvEven);
        assert_eq!(r, 0x3F81, "past halfway should always round up");
    }

    #[test]
    fn test_f32_to_bf16_overflow_to_next_exp() {
        // bf16 0x3FFF = exp=127, man=0x7F (max mantissa at this exponent).
        // If we round up, mantissa overflows -> exp should increment.
        // fp32 = 0x3FFF8000 (halfway) with bf16_trunc = 0x3FFF (lsb=1, odd).
        // ConvEven at halfway with lsb=1 -> round up -> 0x3FFF + 1 = 0x4000.
        // 0x4000 = exp=128, man=0 = 2.0. Correct overflow behavior.
        let val = f32::from_bits(0x3FFF_8000);
        let r = f32_to_bf16(val, RoundingMode::ConvEven);
        assert_eq!(r, 0x4000, "rounding should overflow to next exponent");
    }

    // -- aie2_acc_fp32_add: VADD.f datapath, silicon-verified (fuzzer seed 4) --

    #[test]
    fn acc_add_preserves_denormal_inputs() {
        // Two fp32 denorms (from bf16 denorms via vconv) sum to an exact
        // normal: silicon does NOT flush the inputs.
        assert_eq!(aie2_acc_fp32_add(0x8052_0000, 0x807E_0000), 0x80D0_0000);
    }

    #[test]
    fn acc_add_flushes_denormal_result_with_sum_sign() {
        // -0x45 + 0x71 denorms: denormal sum -> +0 with the true sum's sign.
        assert_eq!(aie2_acc_fp32_add(0x8045_0000, 0x0071_0000), 0x0000_0000);
        // -0x6b + 0x2f: negative denormal sum -> -0.
        assert_eq!(aie2_acc_fp32_add(0x806B_0000, 0x002F_0000), 0x8000_0000);
    }

    #[test]
    fn acc_add_nan_keeps_datapath_sign_with_sticky() {
        // -0 + -NaN: exp-255 operand overflows the datapath -> sign|FF|1.
        assert_eq!(aie2_acc_fp32_add(0x8000_0000, 0xFFC0_0000), 0xFF80_0001);
        // NaN + max-normal: mantissa datapath survives -> exp FF forced,
        // mantissa from the (rounded) sum.
        let r = aie2_acc_fp32_add(0xFF80_007F, 0x7F7F_0000);
        assert_eq!(r >> 23, 0x1FF, "negative, exp=255");
        assert_ne!(r & 0x7F_FFFF, 0, "NaN sticky/mantissa nonzero");
    }

    #[test]
    fn acc_add_opposing_infinities_are_nan() {
        // +Inf + -Inf cancels in the mantissa datapath -> +NaN sticky only.
        assert_eq!(aie2_acc_fp32_add(0x7F80_0000, 0xFF80_0000), 0x7F80_0001);
    }

    #[test]
    fn acc_add_normal_matches_ieee() {
        for (a, b) in [(1.0f32, 2.0f32), (1.5, -0.25), (3.0e38, 3.0e37), (-1.0e-30, 1.0e-32)] {
            assert_eq!(f32::from_bits(aie2_acc_fp32_add(a.to_bits(), b.to_bits())), a + b, "{a} + {b}");
        }
    }
}
