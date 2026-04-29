//! UPS (Upshift / type-widening) for AIE2 vector operations.
//!
//! UPS promotes narrow integer lanes into wider accumulator lanes with an
//! optional left shift. This is the standard DSP "upshift" operation used
//! to load data into accumulators before multiply-accumulate chains.
//!
//! # Hardware behavior (derived from AIE2 architecture)
//!
//! The operation is parameterised by a *scale* and an *accumulator mode*:
//!
//! | Scale | Acc mode | Lanes | Input bits | Output bits |
//! |-------|----------|-------|------------|-------------|
//! | Half  | Acc32    | 32    | 8          | 32          |
//! | Full  | Acc32    | 32    | 16         | 32          |
//! | Half  | Acc64    | 16    | 16         | 64          |
//! | Full  | Acc64    | 16    | 32         | 64          |
//!
//! Per lane the operation is:
//! 1. Sign-extend the input value to its declared width.
//! 2. Left-shift by the shift amount.
//! 3. Optionally saturate to the output range.
//! 4. Truncate (mask) to the output width.
//!
//! The 256-bit source vector is treated as packed lanes of `from_bits` width.
//! The result occupies a wider register (256 or 512 bits depending on mode).
//! Because our register file stores 8 x u32 words, the caller is responsible
//! for writing the result to the appropriate accumulator register(s).

use xdna_archspec::aie2::isa::ElementType;
pub use xdna_archspec::aie2::ups::{UpsScale, UpsAccMode, UpsMode, ups_mode};

/// Infer the UPS mode from source and destination element types.
///
/// Returns `None` if the type pair does not correspond to a valid UPS mode.
pub fn ups_mode_from_types(from_type: ElementType, to_type: ElementType) -> Option<UpsMode> {
    let bits_in = from_type.bits() as u32;
    let bits_out = to_type.bits() as u32;

    // Match against the four valid UPS mode entries.
    match (bits_in, bits_out) {
        (8, 32) => Some(UpsMode { lanes: 32, bits_in: 8, bits_out: 32 }),
        (16, 32) => Some(UpsMode { lanes: 32, bits_in: 16, bits_out: 32 }),
        (16, 64) => Some(UpsMode { lanes: 16, bits_in: 16, bits_out: 64 }),
        (32, 64) => Some(UpsMode { lanes: 16, bits_in: 32, bits_out: 64 }),
        // Same-width is a degenerate UPS (just shift within same width).
        (w_in, w_out) if w_in == w_out => {
            let lanes = 256 / w_in;
            Some(UpsMode { lanes, bits_in: w_in, bits_out: w_out })
        }
        _ => None,
    }
}

/// Sign-extend a value from `bits` width to i64.
///
/// Treats the lowest `bits` bits of `val` as a two's-complement signed
/// integer and sign-extends to 64 bits.
fn sign_extend(val: u64, bits: u32) -> i64 {
    if bits == 0 || bits >= 64 {
        return val as i64;
    }
    let shift = 64 - bits;
    ((val as i64) << shift) >> shift
}

/// Truncate a value to `bits` width (mask to lowest `bits` bits).
fn truncate(val: i64, bits: u32) -> i64 {
    if bits >= 64 {
        return val;
    }
    let mask = (1i64 << bits) - 1;
    let masked = val & mask;
    // Sign-extend the truncated value.
    sign_extend(masked as u64, bits)
}

/// Perform UPS on a single lane.
///
/// 1. Extend input to `bits_in` width (sign-extend if signed, zero-extend if unsigned).
/// 2. Left-shift by `shift`.
/// 3. If `saturate` is true, clamp to the signed range of `bits_out`.
/// 4. Truncate to `bits_out` width.
pub fn ups_lane(value: i64, shift: u32, bits_in: u32, bits_out: u32, saturate: bool) -> i64 {
    ups_lane_signed(value, shift, bits_in, bits_out, saturate, true)
}

/// UPS per-lane with explicit signedness control.
///
/// When `signed_input` is false, the input value is zero-extended (masked)
/// instead of sign-extended, matching hardware behavior for unsigned UPS.
pub fn ups_lane_signed(
    value: i64,
    shift: u32,
    bits_in: u32,
    bits_out: u32,
    saturate: bool,
    signed_input: bool,
) -> i64 {
    // Extend input to its declared width.
    let extended = if signed_input {
        truncate(value, bits_in)
    } else {
        // Zero-extend: mask to bits_in width without sign extension.
        if bits_in >= 64 {
            value
        } else {
            value & ((1i64 << bits_in) - 1)
        }
    };

    // Left-shift for scaling into the wider accumulator.
    let shifted = extended.wrapping_shl(shift);

    // Optional saturation to the output range.
    let saturated = if saturate {
        let (vmin, vmax) = if bits_out >= 64 {
            (i64::MIN, i64::MAX)
        } else {
            (-(1i64 << (bits_out - 1)), (1i64 << (bits_out - 1)) - 1)
        };
        shifted.max(vmin).min(vmax)
    } else {
        shifted
    };

    // Truncate to output width with sign extension.
    truncate(saturated, bits_out)
}

/// Extract lane `index` from a packed 256-bit vector represented as `[u32; 8]`.
///
/// Lanes are packed little-endian: lane 0 occupies the lowest bits of word 0.
fn extract_lane(src: &[u32; 8], index: u32, lane_bits: u32) -> i64 {
    let bit_offset = index * lane_bits;
    let word_idx = (bit_offset / 32) as usize;
    let bit_within_word = bit_offset % 32;

    if lane_bits <= 32 - bit_within_word {
        // Lane fits entirely within one word.
        let raw = (src[word_idx] >> bit_within_word) as u64;
        let mask = (1u64 << lane_bits) - 1;
        sign_extend(raw & mask, lane_bits)
    } else {
        // Lane spans two words.
        let lo_bits = 32 - bit_within_word;
        let lo = (src[word_idx] >> bit_within_word) as u64;
        let hi = if word_idx + 1 < 8 {
            src[word_idx + 1] as u64
        } else {
            0
        };
        let raw = lo | (hi << lo_bits);
        let mask = (1u64 << lane_bits) - 1;
        sign_extend(raw & mask, lane_bits)
    }
}

/// Pack a lane value into a 256-bit result vector at `[u32; 8]`.
///
/// Writes `lane_bits` bits of `value` at the position for lane `index`.
fn pack_lane(dst: &mut [u32; 8], index: u32, lane_bits: u32, value: i64) {
    let bit_offset = index * lane_bits;
    let word_idx = (bit_offset / 32) as usize;
    let bit_within_word = bit_offset % 32;
    let mask = if lane_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << lane_bits) - 1
    };
    let raw = (value as u64) & mask;

    if lane_bits <= 32 - bit_within_word {
        // Fits in one word.
        let wmask = (mask as u32) << bit_within_word;
        dst[word_idx] = (dst[word_idx] & !wmask) | ((raw as u32) << bit_within_word);
    } else {
        // Spans two words.
        let lo_bits = 32 - bit_within_word;
        let lo_mask = ((1u64 << lo_bits) - 1) as u32;
        dst[word_idx] =
            (dst[word_idx] & !(lo_mask << bit_within_word)) | ((raw as u32 & lo_mask) << bit_within_word);
        if word_idx + 1 < 8 {
            let hi_bits = lane_bits - lo_bits;
            let hi_mask = if hi_bits >= 32 {
                u32::MAX
            } else {
                (1u32 << hi_bits) - 1
            };
            let hi_val = (raw >> lo_bits) as u32 & hi_mask;
            dst[word_idx + 1] = (dst[word_idx + 1] & !hi_mask) | hi_val;
        }
    }
}

/// Perform a full UPS operation on a 256-bit source vector.
///
/// The source vector contains `mode.lanes` elements of `mode.bits_in` width.
/// Each lane is sign-extended, left-shifted by `shift`, and written to the
/// result as `mode.bits_out`-width elements.
///
/// When `bits_out > 32`, the result needs more than 256 bits. The caller must
/// handle writing to multiple accumulator registers. This function fills up
/// to 8 words (256 bits) of the result; for 64-bit accumulator modes where
/// the full result is 16x64=1024 bits, the caller should invoke this
/// function with appropriate slicing or use [`ups_lane`] directly.
///
/// For the common 32-bit accumulator modes (8->32 and 16->32), the full
/// result fits in 256 bits (8 x u32).
pub fn ups_vector(src: &[u32; 8], shift: u32, from_type: ElementType, to_type: ElementType) -> [u32; 8] {
    let bits_in = from_type.bits() as u32;
    let bits_out = to_type.bits() as u32;
    let lanes = 256 / bits_in;

    // For output wider than 32 bits per lane, we can only fit
    // 256 / bits_out lanes in the result. Cap accordingly.
    let out_lanes = if bits_out > 0 {
        (256 / bits_out).min(lanes)
    } else {
        0
    };

    let mut result = [0u32; 8];

    let signed_input = from_type.is_signed();

    for i in 0..out_lanes {
        let val = extract_lane(src, i, bits_in);
        let out = ups_lane_signed(val, shift, bits_in, bits_out, false, signed_input);
        pack_lane(&mut result, i, bits_out, out);
    }

    result
}

/// Perform UPS and produce a 512-bit accumulator result.
///
/// The accumulator has 8 lanes of 64 bits each (`[u64; 8]`).
///
/// **Acc32 mode** (`bits_out <= 32`): hardware packs TWO 32-bit accumulator
/// values per u64 slot: `u64[i] = lane[2*i] | (lane[2*i+1] << 32)`.
/// With 16-bit input this produces 16 acc32 lanes; with 8-bit input, also 16
/// (the maximum that fit in 8 u64 words at 2 per word).
///
/// **Acc64 mode** (`bits_out > 32`): one 64-bit value per u64 slot, 8 lanes.
pub fn ups_vector_to_acc(
    src: &[u32; 8],
    shift: u32,
    from_type: ElementType,
    to_type: ElementType,
) -> [u64; 8] {
    let bits_in = from_type.bits() as u32;
    let bits_out = to_type.bits() as u32;
    let signed_input = from_type.is_signed();

    let mut result = [0u64; 8];

    if bits_out <= 32 {
        // Acc32 mode: pack two 32-bit values per u64 word.
        let in_lanes = (256 / bits_in).min(16);
        for i in 0..in_lanes {
            let val = extract_lane(src, i, bits_in);
            let out = ups_lane_signed(val, shift, bits_in, bits_out, false, signed_input);
            let out_u32 = (out as u64) & 0xFFFF_FFFF;
            let word_idx = (i / 2) as usize;
            if i % 2 == 0 {
                result[word_idx] |= out_u32;
            } else {
                result[word_idx] |= out_u32 << 32;
            }
        }
    } else {
        // Acc64 mode: one 64-bit value per u64 word.
        let in_lanes = (256 / bits_in).min(8);
        for i in 0..in_lanes {
            let val = extract_lane(src, i, bits_in);
            let out = ups_lane_signed(val, shift, bits_in, bits_out, false, signed_input);
            result[i as usize] = out as u64;
        }
    }

    result
}

/// Perform UPS from a 256-bit narrow source to a 1024-bit wide accumulator.
///
/// This is the "w2c" path: a narrow vector register (wl) contains enough
/// elements for a 4:1 upshift to fill a full cm register (bml + bmh).
///
/// - **D8/S8 -> S32 (Acc32)**: 32 x 8-bit values -> 32 x 32-bit, packed as
///   16 u64 words (two i32 per u64).
/// - **D16/S16 -> S64 (Acc64)**: 16 x 16-bit values -> 16 x 64-bit, one per
///   u64 word.
pub fn ups_vector_to_acc_wide(
    src: &[u32; 8],
    shift: u32,
    from_type: ElementType,
    to_type: ElementType,
) -> [u64; 16] {
    let bits_in = from_type.bits() as u32;
    let bits_out = to_type.bits() as u32;
    let signed_input = from_type.is_signed();

    let mut result = [0u64; 16];

    if bits_out <= 32 {
        // Acc32 mode: 32 narrow lanes -> 32 acc32 values in 16 u64 words.
        let in_lanes = (256 / bits_in).min(32);
        for i in 0..in_lanes {
            let val = extract_lane(src, i, bits_in);
            let out = ups_lane_signed(val, shift, bits_in, bits_out, false, signed_input);
            let out_u32 = (out as u64) & 0xFFFF_FFFF;
            let word_idx = (i / 2) as usize;
            if i % 2 == 0 {
                result[word_idx] |= out_u32;
            } else {
                result[word_idx] |= out_u32 << 32;
            }
        }
    } else {
        // Acc64 mode: 16 narrow lanes -> 16 acc64 values in 16 u64 words.
        let in_lanes = (256 / bits_in).min(16);
        for i in 0..in_lanes {
            let val = extract_lane(src, i, bits_in);
            let out = ups_lane_signed(val, shift, bits_in, bits_out, false, signed_input);
            result[i as usize] = out as u64;
        }
    }

    result
}

// --- VectorAlu dispatch wrapper ---

use crate::interpreter::bundle::{Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    /// Dispatch entry point for UPS (Upshift / type-widening) operations.
    ///
    /// Handles both narrow (256-bit) and wide (512-bit / 1024-bit) paths,
    /// including AccumWidth detection for Half vs Full accumulator destinations.
    pub(super) fn execute_ups(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let has_wide_acc_source = matches!(
            op.accum_width,
            Some(crate::interpreter::decode::register_map::AccumWidth::Full)
                | Some(crate::interpreter::decode::register_map::AccumWidth::Half)
        );

        if op.is_wide_vector || has_wide_acc_source {
            Self::execute_ups_wide(op, ctx, et)
        } else {
            Self::execute_ups_narrow(op, ctx, et)
        }
    }

    /// Narrow UPS path: widen narrow vector lanes into 512-bit accumulator.
    fn execute_ups_narrow(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let src = Self::get_vector_source(op, ctx, 0);
        let shift = Self::get_shift_amount(op, ctx);
        let from = op.from_type.unwrap_or(ElementType::Int16);
        let acc_result = ups_vector_to_acc(&src, shift, from, et);
        match &op.dest {
            Some(Operand::AccumReg(r)) => {
                ctx.accumulator.write(*r, acc_result);
            }
            other => {
                panic!(
                    "VUPS destination must be AccumReg, got {:?} (encoding={:?})",
                    other, op.encoding_name,
                );
            }
        }
        true
    }

    /// Wide UPS path: widen to 1024-bit accumulator (w2c or x2c).
    fn execute_ups_wide(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let shift = Self::get_shift_amount(op, ctx);
        let from = op.from_type.unwrap_or(ElementType::Int16);
        let is_half =
            matches!(op.accum_width, Some(crate::interpreter::decode::register_map::AccumWidth::Half));

        if !is_half {
            let acc_wide = if !op.is_wide_vector {
                // w2c path: narrow source (256-bit wl) fills full
                // 1024-bit cm register via 4:1 upshift.
                let src = Self::get_vector_source(op, ctx, 0);
                ups_vector_to_acc_wide(&src, shift, from, et)
            } else {
                // x2c path: wide source (512-bit x-reg), UPS each half.
                let src = Self::get_wide_vec_source(op, ctx, 0);
                let src_lo: [u32; 8] = src[..8].try_into().unwrap();
                let src_hi: [u32; 8] = src[8..].try_into().unwrap();
                let acc_lo = ups_vector_to_acc(&src_lo, shift, from, et);
                let acc_hi = ups_vector_to_acc(&src_hi, shift, from, et);
                let mut wide = [0u64; 16];
                wide[..8].copy_from_slice(&acc_lo);
                wide[8..].copy_from_slice(&acc_hi);
                wide
            };

            match &op.dest {
                Some(Operand::AccumReg(r)) => {
                    ctx.accumulator.write_wide(*r, acc_wide);
                }
                other => {
                    panic!(
                        "Wide VUPS destination must be AccumReg, got {:?} (encoding={:?})",
                        other, op.encoding_name,
                    );
                }
            }
        } else {
            // Half UPS (bml/bmh): narrow source -> single 512-bit
            // accum register.
            let src = Self::get_vector_source(op, ctx, 0);
            let acc_result = ups_vector_to_acc(&src, shift, from, et);
            match &op.dest {
                Some(Operand::AccumReg(r)) => {
                    ctx.accumulator.write(*r, acc_result);
                }
                other => {
                    panic!(
                        "VUPS destination must be AccumReg, got {:?} (encoding={:?})",
                        other, op.encoding_name,
                    );
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- sign_extend tests --

    #[test]
    fn test_sign_extend_positive() {
        // 0x7F in 8 bits = +127
        assert_eq!(sign_extend(0x7F, 8), 127);
    }

    #[test]
    fn test_sign_extend_negative() {
        // 0xFF in 8 bits = -1
        assert_eq!(sign_extend(0xFF, 8), -1);
        // 0x80 in 8 bits = -128
        assert_eq!(sign_extend(0x80, 8), -128);
    }

    #[test]
    fn test_sign_extend_16bit() {
        assert_eq!(sign_extend(0xFFFF, 16), -1);
        assert_eq!(sign_extend(0x7FFF, 16), 32767);
        assert_eq!(sign_extend(0x8000, 16), -32768);
    }

    // -- truncate tests --

    #[test]
    fn test_truncate_to_8bit() {
        assert_eq!(truncate(256, 8), 0);
        assert_eq!(truncate(255, 8), -1); // 0xFF sign-extended
        assert_eq!(truncate(127, 8), 127);
        assert_eq!(truncate(-1, 8), -1);
    }

    #[test]
    fn test_truncate_to_32bit() {
        assert_eq!(truncate(0x7FFFFFFF, 32), 0x7FFFFFFF);
        assert_eq!(truncate(-1, 32), -1);
    }

    // -- ups_lane tests --

    #[test]
    fn test_ups_lane_8_to_32_no_shift() {
        // Simple widening: 8-bit value 42 -> 32-bit value 42
        let result = ups_lane(42, 0, 8, 32, false);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_ups_lane_8_to_32_with_shift() {
        // 8-bit value 1, shift left by 8 -> 256 in 32-bit
        let result = ups_lane(1, 8, 8, 32, false);
        assert_eq!(result, 256);
    }

    #[test]
    fn test_ups_lane_negative_8_to_32() {
        // -1 in 8 bits (0xFF), shift 0 -> -1 in 32 bits
        let result = ups_lane(0xFF, 0, 8, 32, false);
        assert_eq!(result, -1);
    }

    #[test]
    fn test_ups_lane_negative_8_to_32_shifted() {
        // -1 in 8 bits, shift left by 4 -> -16 in 32 bits
        let result = ups_lane(0xFF, 4, 8, 32, false);
        assert_eq!(result, -16);
    }

    #[test]
    fn test_ups_lane_16_to_32() {
        // 16-bit value 1000, shift 8 -> 256000
        let result = ups_lane(1000, 8, 16, 32, false);
        assert_eq!(result, 256000);
    }

    #[test]
    fn test_ups_lane_16_to_64() {
        // 16-bit value 0x7FFF = 32767, shift 16 -> 32767 * 65536
        let result = ups_lane(0x7FFF, 16, 16, 64, false);
        assert_eq!(result, 32767 * 65536);
    }

    #[test]
    fn test_ups_lane_saturation() {
        // 16-bit max positive 32767, shift 20 -> would overflow 32-bit
        // With saturation, should clamp to i32::MAX
        let result = ups_lane(0x7FFF, 20, 16, 32, true);
        assert_eq!(result, i32::MAX as i64);
    }

    #[test]
    fn test_ups_lane_saturation_negative() {
        // -32768 in 16 bits, shift 20 -> large negative, clamp to i32::MIN
        let result = ups_lane(0x8000_u64 as i64, 20, 16, 32, true);
        assert_eq!(result, i32::MIN as i64);
    }

    #[test]
    fn test_ups_lane_no_saturation_wraps() {
        // Without saturation, large shifts wrap within the output width.
        // 32767 << 20 in 32 bits wraps around.
        let result = ups_lane(0x7FFF, 20, 16, 32, false);
        let expected = truncate(32767i64 << 20, 32);
        assert_eq!(result, expected);
    }

    // -- extract_lane / pack_lane tests --

    #[test]
    fn test_extract_pack_roundtrip_8bit() {
        let mut data = [0u32; 8];
        // Write values to each 8-bit lane
        for i in 0..32u32 {
            pack_lane(&mut data, i, 8, i as i64);
        }
        for i in 0..32u32 {
            let val = extract_lane(&data, i, 8);
            assert_eq!(val, i as i64, "lane {i} mismatch");
        }
    }

    #[test]
    fn test_extract_pack_roundtrip_16bit() {
        let mut data = [0u32; 8];
        for i in 0..16u32 {
            let val = (i as i64) * 1000 - 8000;
            pack_lane(&mut data, i, 16, val);
        }
        for i in 0..16u32 {
            let expected = (i as i64) * 1000 - 8000;
            let val = extract_lane(&data, i, 16);
            assert_eq!(val, expected, "lane {i} mismatch");
        }
    }

    #[test]
    fn test_extract_negative_8bit() {
        // Pack -1 (0xFF) into lane 0
        let mut data = [0u32; 8];
        pack_lane(&mut data, 0, 8, -1);
        assert_eq!(extract_lane(&data, 0, 8), -1);
    }

    // -- ups_vector integration tests --

    #[test]
    fn test_ups_vector_8_to_32_identity() {
        // 32 lanes of 8-bit values [0..31], shift 0
        // Result: 8 words of 32-bit values [0..7] (only 8 fit in 256 bits)
        let mut src = [0u32; 8];
        for i in 0..32u32 {
            pack_lane(&mut src, i, 8, i as i64);
        }

        let result = ups_vector(&src, 0, ElementType::Int8, ElementType::Int32);

        // With 32-bit output, only 8 lanes fit in 256 bits.
        for i in 0..8u32 {
            let val = extract_lane(&result, i, 32);
            assert_eq!(val, i as i64, "lane {i}");
        }
    }

    #[test]
    fn test_ups_vector_8_to_32_shifted() {
        // Pack value 5 into every 8-bit lane, shift by 4 -> each 32-bit lane = 80
        let mut src = [0u32; 8];
        for i in 0..32u32 {
            pack_lane(&mut src, i, 8, 5);
        }

        let result = ups_vector(&src, 4, ElementType::Int8, ElementType::Int32);

        for i in 0..8u32 {
            let val = extract_lane(&result, i, 32);
            assert_eq!(val, 80, "lane {i}");
        }
    }

    #[test]
    fn test_ups_vector_16_to_32() {
        // 16 lanes of 16-bit values, shift by 8
        let mut src = [0u32; 8];
        for i in 0..16u32 {
            pack_lane(&mut src, i, 16, (i as i64) + 1);
        }

        let result = ups_vector(&src, 8, ElementType::Int16, ElementType::Int32);

        // 32-bit output: 8 lanes fit in 256 bits
        for i in 0..8u32 {
            let expected = ((i as i64) + 1) << 8;
            let val = extract_lane(&result, i, 32);
            assert_eq!(val, expected, "lane {i}");
        }
    }

    #[test]
    fn test_ups_vector_negative_values() {
        // Pack -5 into 8-bit lanes, shift by 2 -> -20 in 32-bit
        let mut src = [0u32; 8];
        for i in 0..32u32 {
            pack_lane(&mut src, i, 8, -5);
        }

        let result = ups_vector(&src, 2, ElementType::Int8, ElementType::Int32);

        for i in 0..8u32 {
            let val = extract_lane(&result, i, 32);
            assert_eq!(val, -20, "lane {i}");
        }
    }

    // -- UPS mode lookup tests --

    #[test]
    fn test_ups_mode_table() {
        let m = ups_mode(UpsScale::Half, UpsAccMode::Acc32);
        assert_eq!((m.lanes, m.bits_in, m.bits_out), (32, 8, 32));

        let m = ups_mode(UpsScale::Full, UpsAccMode::Acc32);
        assert_eq!((m.lanes, m.bits_in, m.bits_out), (32, 16, 32));

        let m = ups_mode(UpsScale::Half, UpsAccMode::Acc64);
        assert_eq!((m.lanes, m.bits_in, m.bits_out), (16, 16, 64));

        let m = ups_mode(UpsScale::Full, UpsAccMode::Acc64);
        assert_eq!((m.lanes, m.bits_in, m.bits_out), (16, 32, 64));
    }

    #[test]
    fn test_ups_mode_from_types_valid() {
        let m = ups_mode_from_types(ElementType::Int8, ElementType::Int32).unwrap();
        assert_eq!((m.lanes, m.bits_in, m.bits_out), (32, 8, 32));

        let m = ups_mode_from_types(ElementType::Int16, ElementType::Int32).unwrap();
        assert_eq!((m.lanes, m.bits_in, m.bits_out), (32, 16, 32));
    }

    #[test]
    fn test_ups_mode_from_types_same_width() {
        // Same width is allowed as degenerate case.
        let m = ups_mode_from_types(ElementType::Int32, ElementType::Int32).unwrap();
        assert_eq!((m.lanes, m.bits_in, m.bits_out), (8, 32, 32));
    }

    #[test]
    fn test_ups_mode_from_types_invalid() {
        // 32->16 is not a valid UPS (that would be SRS).
        assert!(ups_mode_from_types(ElementType::Int32, ElementType::Int16).is_none());
    }

    // -- ups_vector_to_acc tests --

    #[test]
    fn test_ups_acc32_packing_two_per_u64() {
        // vups.s32.s16 with shift=0: 16 input lanes of 16-bit -> 16 acc32 values,
        // packed 2 per u64.  u64[i] = lane[2*i] | (lane[2*i+1] << 32).
        let src = [
            0x0002_0001u32,
            0x0004_0003,
            0x0006_0005,
            0x0008_0007,
            0x000A_0009,
            0x000C_000B,
            0x000E_000D,
            0x0010_000F,
        ];
        let acc = ups_vector_to_acc(&src, 0, ElementType::Int16, ElementType::Int32);
        // Lane 0=1, lane 1=2 -> acc[0] = 1 | (2<<32)
        assert_eq!(acc[0], 1 | (2u64 << 32), "acc[0]");
        assert_eq!(acc[1], 3 | (4u64 << 32), "acc[1]");
        assert_eq!(acc[2], 5 | (6u64 << 32), "acc[2]");
        assert_eq!(acc[3], 7 | (8u64 << 32), "acc[3]");
        assert_eq!(acc[4], 9 | (10u64 << 32), "acc[4]");
        assert_eq!(acc[5], 11 | (12u64 << 32), "acc[5]");
        assert_eq!(acc[6], 13 | (14u64 << 32), "acc[6]");
        assert_eq!(acc[7], 15 | (16u64 << 32), "acc[7]");
    }

    #[test]
    fn test_ups_acc32_negative() {
        // vups.s32.s16 with shift=0 and negative input: -1 in i16 = 0xFFFF.
        // -1 as i32 = 0xFFFF_FFFF. Two per u64 = 0xFFFF_FFFF_FFFF_FFFF.
        let src = [0xFFFF_FFFF_u32; 8]; // all 16 lanes = -1 as i16
        let acc = ups_vector_to_acc(&src, 0, ElementType::Int16, ElementType::Int32);
        for i in 0..8usize {
            // Both lo and hi halves are 0xFFFF_FFFF (i.e. -1 as u32).
            assert_eq!(acc[i], 0xFFFF_FFFF_FFFF_FFFF, "lane pair {}", i);
        }
    }

    #[test]
    fn test_ups_acc64_one_per_u64() {
        // vups.s64.s32 with shift=0: 8 input lanes of 32-bit -> 8 x 64-bit.
        // Acc64 mode: one value per u64 word.
        let src = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let acc = ups_vector_to_acc(&src, 0, ElementType::Int32, ElementType::Int64);
        for i in 0..8 {
            assert_eq!(acc[i], (i + 1) as u64, "lane {}", i);
        }
    }

    #[test]
    fn test_ups_acc32_from_d8() {
        // vups.s32.s8 with shift=0: up to 16 lanes of 8-bit input -> 16 acc32 values.
        // Input: 32 lanes of 8-bit in [u32; 8], only first 16 processed.
        let mut src = [0u32; 8];
        for i in 0..32u32 {
            pack_lane(&mut src, i, 8, (i + 1) as i64);
        }
        let acc = ups_vector_to_acc(&src, 0, ElementType::Int8, ElementType::Int32);
        // 16 acc32 lanes, packed 2 per u64.
        for pair in 0..8usize {
            let lo = (pair * 2 + 1) as u64;
            let hi = (pair * 2 + 2) as u64;
            assert_eq!(acc[pair], lo | (hi << 32), "acc[{}]", pair);
        }
    }

    #[test]
    fn test_ups_wide_d8_to_s32() {
        // w2c path: 32 x uint8 in 256 bits -> 32 x int32 in 1024 bits (Acc32).
        // Pack 32 consecutive byte values (1..=32) into 8 u32 words.
        let mut src = [0u32; 8];
        for i in 0u32..32 {
            let byte_idx = i / 4;
            let shift_amt = (i % 4) * 8;
            src[byte_idx as usize] |= (i + 1) << shift_amt;
        }

        let acc = ups_vector_to_acc_wide(&src, 0, ElementType::UInt8, ElementType::Int32);

        // 32 acc32 lanes packed as 16 u64 words, two i32 per u64.
        for pair in 0..16usize {
            let lo = (pair * 2 + 1) as u64;
            let hi = (pair * 2 + 2) as u64;
            assert_eq!(acc[pair], lo | (hi << 32), "acc[{}]", pair);
        }
    }

    #[test]
    fn test_ups_wide_s16_to_s64() {
        // w2c path: 16 x int16 in 256 bits -> 16 x int64 in 1024 bits (Acc64).
        let mut src = [0u32; 8];
        for i in 0u32..16 {
            let word_idx = (i / 2) as usize;
            let shift_amt = (i % 2) * 16;
            // Use small negative values to test sign extension.
            let val = (-(i as i16) - 1) as u16;
            src[word_idx] |= (val as u32) << shift_amt;
        }

        let acc = ups_vector_to_acc_wide(&src, 0, ElementType::Int16, ElementType::Int64);

        for i in 0..16usize {
            let expected = (-(i as i64) - 1) as u64;
            assert_eq!(acc[i], expected, "acc[{}]", i);
        }
    }
}

/// Integration tests that exercise UPS through the full VectorAlu dispatch path.
#[cfg(test)]
mod integration_tests {
    use crate::interpreter::bundle::{ElementType, Operand, SlotIndex, SlotOp};
    use crate::interpreter::execute::vector_dispatch::VectorAlu;
    use crate::interpreter::execute::vector_ups::ups_vector_to_acc;
    use crate::interpreter::execute::vector_ups::ups_vector_to_acc_wide;
    use crate::interpreter::state::{ExecutionContext, SrsConfig};
    use xdna_archspec::aie2::isa::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    #[test]
    fn test_vector_ups_srs_roundtrip() {
        let mut ctx = make_ctx();
        let src = [
            0x0002_0001u32,
            0x0004_0003,
            0x0006_0005,
            0x0008_0007,
            0x000A_0009,
            0x000C_000B,
            0x000E_000D,
            0x0010_000F,
        ];
        let acc = ups_vector_to_acc(&src, 0, ElementType::Int16, ElementType::Int32);
        ctx.accumulator.write(0, acc);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int16)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(0));
        op.from_type = Some(ElementType::Int32);

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(0);
        assert_eq!(result[0], 0x0002_0001, "lanes 0-1");
        assert_eq!(result[1], 0x0004_0003, "lanes 2-3");
        assert_eq!(result[2], 0x0006_0005, "lanes 4-5");
        assert_eq!(result[3], 0x0008_0007, "lanes 6-7");
        assert_eq!(result[4], 0x000A_0009, "lanes 8-9");
        assert_eq!(result[5], 0x000C_000B, "lanes 10-11");
        assert_eq!(result[6], 0x000E_000D, "lanes 12-13");
        assert_eq!(result[7], 0x0010_000F, "lanes 14-15");
    }

    #[test]
    fn test_wide_ups_srs_s8_s32_roundtrip() {
        let src = [
            0x04030201u32,
            0x08070605,
            0x0C0B0A09,
            0x100F0E0D,
            0x14131211,
            0x18171615,
            0x1C1B1A19,
            0x201F1E1D,
        ];

        let acc_wide = ups_vector_to_acc_wide(&src, 0, ElementType::Int8, ElementType::Int32);
        assert!(acc_wide.iter().any(|&v| v != 0), "UPS produced all zeros");

        let acc_lo: [u64; 8] = acc_wide[..8].try_into().unwrap();
        let acc_hi: [u64; 8] = acc_wide[8..].try_into().unwrap();

        let cfg = SrsConfig::default();
        let result_lo =
            VectorAlu::vector_srs_from_acc(&acc_lo, 0, ElementType::Int32, ElementType::Int8, &cfg);
        let result_hi =
            VectorAlu::vector_srs_from_acc(&acc_hi, 0, ElementType::Int32, ElementType::Int8, &cfg);

        let lanes_per_half = 16usize;
        let to_bits = 8usize;
        let words_per_half = (lanes_per_half * to_bits + 31) / 32;
        let n = words_per_half.min(8);
        assert_eq!(n, 4, "should be 4 words per half for 16 x i8");

        let mut packed = [0u32; 8];
        packed[..n].copy_from_slice(&result_lo[..n]);
        packed[n..n + n].copy_from_slice(&result_hi[..n]);
        assert_eq!(packed, src, "wide s8.s32 UPS->SRS round-trip failed");

        // Also test with non-zero shift
        let acc_shifted = ups_vector_to_acc_wide(&src, 4, ElementType::Int8, ElementType::Int32);
        let acc_s_lo: [u64; 8] = acc_shifted[..8].try_into().unwrap();
        let acc_s_hi: [u64; 8] = acc_shifted[8..].try_into().unwrap();
        let res_s_lo =
            VectorAlu::vector_srs_from_acc(&acc_s_lo, 4, ElementType::Int32, ElementType::Int8, &cfg);
        let res_s_hi =
            VectorAlu::vector_srs_from_acc(&acc_s_hi, 4, ElementType::Int32, ElementType::Int8, &cfg);
        let mut packed_s = [0u32; 8];
        packed_s[..4].copy_from_slice(&res_s_lo[..4]);
        packed_s[4..8].copy_from_slice(&res_s_hi[..4]);
        assert_eq!(packed_s, src, "wide s8.s32 UPS->SRS round-trip with shift=4 failed");
    }

    #[test]
    fn test_wide_ups_x2c() {
        let mut ctx = make_ctx();

        ctx.vector.write(4, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(5, [9, 10, 11, 12, 13, 14, 15, 16]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Ups)
            .as_vector(ElementType::Int64)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::VectorReg(4))
            .with_source(Operand::Immediate(0));
        op.from_type = Some(ElementType::Int32);
        op.is_wide_vector = true;

        VectorAlu::execute(&op, &mut ctx);

        let acc_lo = ctx.accumulator.read(0);
        for i in 0..8 {
            assert_eq!(acc_lo[i], (i as u64 + 1), "acc_lo lane {i}");
        }
        let acc_hi = ctx.accumulator.read(1);
        for i in 0..8 {
            assert_eq!(acc_hi[i], (i as u64 + 9), "acc_hi lane {i}");
        }
    }
}
