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
//! - **Canonical NaN**: NaN results observed on NPU1 silicon carry mantissa 1
//!   (`0x7F800001`), not the IEEE quiet-NaN bit (bit 22); see
//!   `aie2_fp32_nan_bits`. Some datapaths emit other payloads (open gap).
//!
//! - **Rounding for bf16 conversion**: selected by the same crRnd modes as the
//!   integer SRS pipeline (see `f32_to_bf16`).
//!
//! Behavioral facts not confirmed on silicon are model-informed and tracked in
//! `docs/fidelity-gaps/vector-compute.md`.

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
#[inline]
pub fn bf16_split(bits: u16) -> (bool, u8, u8) {
    let sign = (bits & BF16_SIGN_BIT) != 0;
    let exp = ((bits & BF16_EXP_MASK) >> 7) as u8;
    let man = (bits & BF16_MAN_MASK) as u8;
    (sign, exp, man)
}

/// Assemble a bf16 bit pattern from (sign, exponent, mantissa).
#[inline]
pub fn bf16_make(sign: bool, exp: u8, man: u8) -> u16 {
    ((sign as u16) << 15) | ((exp as u16) << 7) | ((man as u16) & BF16_MAN_MASK)
}

/// Split an fp32 bit pattern into (sign, exponent, mantissa).
#[inline]
pub fn fp32_split(bits: u32) -> (bool, u8, u32) {
    let sign = (bits & FP32_SIGN_BIT) != 0;
    let exp = ((bits & FP32_EXP_MASK) >> 23) as u8;
    let man = bits & FP32_MAN_MASK;
    (sign, exp, man)
}

/// Assemble an fp32 bit pattern from (sign, exponent, mantissa).
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
/// bf16 is the upper half of an fp32, so widening is a 16-bit left shift.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to bf16 with the specified rounding mode.
///
/// Rules, all observed on NPU1 silicon (`tools/golden/silicon/vec_conv_bf16*`,
/// captured 2026-10-28):
///
/// - Finite values, denormals and zeros are rounded as a sign-magnitude bit
///   pattern: the low 16 bits are dropped with the same per-mode rule as an
///   SRS shift of 16 on the signed value (`vector_srs::shift_round`). A carry
///   propagates into the exponent, so rounding up the largest finite value
///   gives infinity. Denormals are NOT flushed; rounding can turn a bf16
///   denormal into a normal or a nonzero value into a zero, and the sign is
///   kept (a negative value rounded to zero gives -0).
/// - Infinities pass through unchanged.
/// - NaN is never rounded: the high 16 bits are kept (sign, exponent, top 7
///   payload bits) in every mode. If that leaves a zero payload, which would
///   read as infinity, the payload's lowest bit is set so it stays a NaN.
pub fn f32_to_bf16(value: f32, mode: RoundingMode) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) as u16) & BF16_SIGN_BIT;
    let high_half = (bits >> 16) as u16;

    if fp32_is_nan(bits) {
        return if high_half & BF16_MAN_MASK == 0 {
            high_half | 1
        } else {
            high_half
        };
    }

    // Round the signed value so directed modes (floor, ceil, ...) see the sign;
    // the magnitude of the result is the bf16 magnitude, the sign is kept.
    let magnitude = (bits & !FP32_SIGN_BIT) as i128;
    let signed = if sign != 0 { -magnitude } else { magnitude };
    let rounded = super::vector_srs::shift_round(signed, 16, mode);
    sign | rounded.unsigned_abs() as u16
}

/// Convert f32 to bf16 with simple truncation (no rounding).
///
/// Drops the lower 16 bits of the fp32 representation. Used where the datapath
/// truncates rather than rounds.
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
/// When exp==0 the mantissa is zeroed regardless of its value.
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
/// Mantissa = 1 is what the silicon produces; it is not the BF16 canonical
/// NaN (mantissa 0x7F) extended to f32.  Hardware is ground truth.
#[inline]
pub fn aie2_fp32_nan_bits(sign: bool) -> u32 {
    ((sign as u32) << 31) | (0xFF << 23) | 0x01
}

/// Create an fp32 infinity bit pattern.
///
/// This is the standard IEEE 754 infinity encoding (exponent all ones,
/// mantissa zero).
#[inline]
pub fn fp32_inf_bits(sign: bool) -> u32 {
    ((sign as u32) << 31) | (0xFF << 23)
}

/// Create the AIE2 canonical NaN as an f32 value.
#[inline]
pub fn aie2_canonical_nan(sign: bool) -> f32 {
    f32::from_bits(aie2_fp32_nan_bits(sign))
}

// ---------------------------------------------------------------------------
// AIE2 NaN propagation
// ---------------------------------------------------------------------------

/// AIE2 NaN propagation for binary operations.
///
/// When either operand is NaN, the result is the AIE2 canonical NaN
/// (positive), regardless of which operand was NaN or what its sign
/// was. Model-informed: silicon shows datapath-dependent NaN payloads (see
/// docs/fidelity-gaps/vector-compute.md).
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
        let pos_nan = aie2_fp32_nan_bits(false);
        assert_eq!(pos_nan, 0x7F80_0001);
        assert!(fp32_is_nan(pos_nan));

        let neg_nan = aie2_fp32_nan_bits(true);
        assert_eq!(neg_nan, 0xFF80_0001);
        assert!(fp32_is_nan(neg_nan));

        // Verify the f32 representation is NaN.
        assert!(f32::from_bits(pos_nan).is_nan());
        assert!(f32::from_bits(neg_nan).is_nan());
    }

    #[test]
    fn test_fp32_inf_bits_bits() {
        assert_eq!(fp32_inf_bits(false), 0x7F80_0000);
        assert_eq!(fp32_inf_bits(true), 0xFF80_0000);
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
        assert_eq!(r.to_bits(), aie2_fp32_nan_bits(false));

        // normal + NaN -> canonical NaN
        let result = aie2_nan_propagate(1.0, nan);
        assert!(result.is_some());
        assert_eq!(result.unwrap().to_bits(), aie2_fp32_nan_bits(false));
    }

    #[test]
    fn test_nan_propagate_both_nan() {
        let nan1 = f32::from_bits(0x7FC0_0001);
        let nan2 = f32::from_bits(0xFFC0_0002);

        // Both NaN -> positive canonical NaN (sign and payload discarded).
        let result = aie2_nan_propagate(nan1, nan2);
        assert!(result.is_some());
        assert_eq!(result.unwrap().to_bits(), aie2_fp32_nan_bits(false));
    }

    #[test]
    fn test_nan_propagate_negative_nan() {
        // Even a negative NaN input produces positive canonical NaN.
        let neg_nan = f32::from_bits(0xFFC0_0000);
        let result = aie2_nan_propagate(neg_nan, 1.0);
        assert_eq!(result.unwrap().to_bits(), aie2_fp32_nan_bits(false));
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
        assert_eq!(canonical.to_bits(), aie2_fp32_nan_bits(false));
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
}

#[cfg(test)]
mod bf16_conversion_tests {
    use super::*;
    use crate::interpreter::execute::silicon_golden::{kernels_with_prefix, SiliconCase};
    use RoundingMode::*;

    fn convert(bits: u32, mode: RoundingMode) -> u16 {
        f32_to_bf16(f32::from_bits(bits), mode)
    }

    /// Positive inputs around two adjacent bf16 values, A = 0x3F80 (even) and
    /// B = 0x3F81 (odd): a quarter, exactly half, and three quarters of the way up.
    const PATTERNS: [(u32, u16); 6] = [
        (0x3F80_4000, 0x3F80), // even base, below half
        (0x3F80_8000, 0x3F80), // even base, tie
        (0x3F80_C000, 0x3F80), // even base, above half
        (0x3F81_4000, 0x3F81), // odd base, below half
        (0x3F81_8000, 0x3F81), // odd base, tie
        (0x3F81_C000, 0x3F81), // odd base, above half
    ];

    /// Per mode: whether the magnitude rounds up (1) for each pattern, for
    /// positive and then negative inputs.
    const UP: [(RoundingMode, [u16; 6], [u16; 6]); 12] = [
        (Floor, [0; 6], [1; 6]),
        (Ceil, [1; 6], [0; 6]),
        (SymFloor, [0; 6], [0; 6]),
        (SymCeil, [1; 6], [1; 6]),
        (NegInf, [0, 0, 1, 0, 0, 1], [0, 1, 1, 0, 1, 1]),
        (PosInf, [0, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 1]),
        (SymZero, [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]),
        (SymInf, [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]),
        (ConvEven, [0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1]),
        (ConvOdd, [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1]),
        (InexactEven, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]),
        (InexactOdd, [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]),
    ];

    #[test]
    fn each_mode_rounds_quarter_half_and_three_quarter_points() {
        for (mode, pos_up, neg_up) in UP {
            for (i, (bits, base)) in PATTERNS.into_iter().enumerate() {
                assert_eq!(convert(bits, mode), base + pos_up[i], "{mode:?} +{bits:#010x}");
                assert_eq!(
                    convert(bits | 0x8000_0000, mode),
                    0x8000 | (base + neg_up[i]),
                    "{mode:?} -{bits:#010x}"
                );
            }
        }
    }

    #[test]
    fn exactly_representable_values_are_unchanged_in_every_mode() {
        for raw in 0..16u8 {
            let mode = RoundingMode::from_raw(raw).unwrap();
            assert_eq!(convert(0x3F81_0000, mode), 0x3F81);
            assert_eq!(convert(0xC0490000, mode), 0xC049);
        }
    }

    #[test]
    fn signed_zero_is_preserved() {
        for mode in [Floor, Ceil, ConvEven] {
            assert_eq!(convert(0x0000_0000, mode), 0x0000);
            assert_eq!(convert(0x8000_0000, mode), 0x8000);
        }
    }

    #[test]
    fn denormals_are_rounded_not_flushed() {
        assert_eq!(convert(0x0000_0001, Floor), 0x0000);
        assert_eq!(convert(0x0000_0001, Ceil), 0x0001);
        assert_eq!(convert(0x8000_0001, Floor), 0x8001);
        assert_eq!(convert(0x8000_0001, Ceil), 0x8000);
        // 0x0001_8000 is a tie between bf16 denormals 1 and 2.
        assert_eq!(convert(0x0001_8000, ConvEven), 0x0002);
        // 0x0000_8000 is a tie between 0 and the smallest denormal.
        assert_eq!(convert(0x0000_8000, ConvEven), 0x0000);
        assert_eq!(convert(0x0000_8000, ConvOdd), 0x0001);
    }

    #[test]
    fn rounding_up_the_largest_finite_value_carries_to_infinity() {
        assert_eq!(convert(0x7F7F_FFFF, Ceil), 0x7F80);
        assert_eq!(convert(0x7F7F_FFFF, ConvEven), 0x7F80);
        assert_eq!(convert(0x7F7F_FFFF, Floor), 0x7F7F);
        assert_eq!(convert(0xFF7F_FFFF, Floor), 0xFF80);
        assert_eq!(convert(0xFF7F_FFFF, Ceil), 0xFF7F);
    }

    #[test]
    fn infinities_pass_through() {
        for mode in [Floor, Ceil, SymCeil, ConvOdd] {
            assert_eq!(convert(0x7F80_0000, mode), 0x7F80);
            assert_eq!(convert(0xFF80_0000, mode), 0xFF80);
        }
    }

    #[test]
    fn nan_keeps_the_high_half_in_every_mode() {
        for raw in 0..16u8 {
            let mode = RoundingMode::from_raw(raw).unwrap();
            assert_eq!(convert(0x7FC0_0001, mode), 0x7FC0);
            assert_eq!(convert(0x7FFF_FFFF, mode), 0x7FFF);
            assert_eq!(convert(0xFFA7_FFFF, mode), 0xFFA7);
        }
    }

    #[test]
    fn nan_with_payload_only_in_the_dropped_bits_stays_nan() {
        for raw in 0..16u8 {
            let mode = RoundingMode::from_raw(raw).unwrap();
            assert_eq!(convert(0x7F80_0001, mode), 0x7F81);
            assert_eq!(convert(0x7F80_8000, mode), 0x7F81);
            assert_eq!(convert(0xFF80_FFFF, mode), 0xFF81);
        }
    }

    /// Every captured f32 -> bf16 kernel, lane by lane against silicon.
    #[test]
    fn matches_every_silicon_bf16_capture() {
        let kernels = kernels_with_prefix("vec_conv_bf16");
        assert!(kernels.len() >= 27, "expected the full convert sweep, found {}", kernels.len());
        for kernel in kernels {
            let case = SiliconCase::load(&kernel, "conv.cc");
            let mode = case.rounding();
            for (i, (&input, &want)) in case.inputs.iter().zip(&case.silicon).enumerate() {
                let got = f32_to_bf16(f32::from_bits(input as u32), mode);
                assert_eq!(got as i64, want, "{kernel} lane {i}: f32 bits {input:#010x}, {mode:?}");
            }
        }
    }
}
