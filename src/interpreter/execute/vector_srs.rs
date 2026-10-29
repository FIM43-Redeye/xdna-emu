//! AIE2 Shift-Round-Saturate (SRS).
//!
//! Semantics are derived from the llvm-aie mode names, AM020, and the NPU1
//! silicon captures in `tools/golden/silicon/` (2026-10-28); see
//! `docs/fidelity-gaps/vector-compute.md` for the per-mode rules.

use xdna_archspec::aie2::rounding::RoundingMode;

/// Arithmetic right shift of `value` by `shift` bits, rounded per `mode`.
///
/// The result is exact (no width limit): it can exceed the output range, which
/// is `srs_lane`'s business. Shared with the f32->bf16 conversion, which is the
/// same operation with shift 16 applied to the signed f32 bit pattern.
pub(crate) fn shift_round(value: i128, shift: u32, mode: RoundingMode) -> i128 {
    if shift == 0 {
        return value;
    }
    // Inputs are at most 65 bits wide, so any shift past 120 behaves like 120
    // (nothing but sign and "inexact" survives) while staying inside i128.
    let shift = shift.min(120);

    let floor = value >> shift;
    let dropped = value & ((1i128 << shift) - 1);
    let half = 1i128 << (shift - 1);

    let inexact = dropped != 0;
    let above_half = dropped > half;
    let tie = dropped == half;
    let negative = value < 0;
    let floor_is_odd = floor & 1 == 1;

    let round_up = match mode {
        RoundingMode::Floor => false,
        RoundingMode::Ceil => inexact,
        RoundingMode::SymFloor => inexact && negative,
        RoundingMode::SymCeil => inexact && !negative,
        // Round to the even (odd) neighbour of an inexact value: the floor is
        // already the wanted parity, or one more is.
        RoundingMode::InexactEven => inexact && floor_is_odd,
        RoundingMode::InexactOdd => inexact && !floor_is_odd,
        RoundingMode::NegInf => above_half,
        RoundingMode::PosInf => above_half || tie,
        RoundingMode::SymZero => above_half || (tie && negative),
        RoundingMode::SymInf => above_half || (tie && !negative),
        RoundingMode::ConvEven => above_half || (tie && floor_is_odd),
        RoundingMode::ConvOdd => above_half || (tie && !floor_is_odd),
    };
    floor + round_up as i128
}

/// Perform shift-round-saturate on a single accumulator lane.
///
/// * `value` - The accumulator lane value (interpreted as signed 64-bit)
/// * `shift` - The shift amount from the SRS configuration
/// * `signed_output` - Whether the output type is signed
/// * `output_bits` - Bit width of the output type (8, 16, or 32)
/// * `saturate` - Whether to clamp out-of-range values (otherwise the result wraps)
/// * `symmetric_saturate` - Whether signed min is -(2^(n-1)-1) instead of -2^(n-1)
/// * `mode` - The rounding mode to apply
///
/// Returns the result truncated to `output_bits`: sign-extended to i64 for a
/// signed output, zero-extended for an unsigned one.
pub fn srs_lane(
    value: i64,
    shift: u32,
    signed_output: bool,
    output_bits: u32,
    saturate: bool,
    symmetric_saturate: bool,
    mode: RoundingMode,
) -> i64 {
    debug_assert!((1..=64).contains(&output_bits));
    let rounded = shift_round(value as i128, shift, mode);

    let (min, max) = if signed_output {
        let min = -(1i128 << (output_bits - 1));
        (if symmetric_saturate { min + 1 } else { min }, (1i128 << (output_bits - 1)) - 1)
    } else {
        (0, (1i128 << output_bits) - 1)
    };

    if saturate {
        return rounded.clamp(min, max) as i64;
    }

    // Without saturation the result is the low `output_bits` bits.
    let low_bits = rounded & ((1i128 << output_bits) - 1);
    if signed_output && low_bits > max {
        (low_bits - (1i128 << output_bits)) as i64
    } else {
        low_bits as i64
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
        let has_wide_acc_source = matches!(
            op.accum_width,
            Some(crate::interpreter::decode::register_map::AccumWidth::Full)
                | Some(crate::interpreter::decode::register_map::AccumWidth::Half)
        );

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
        let is_half =
            matches!(op.accum_width, Some(crate::interpreter::decode::register_map::AccumWidth::Half));

        if !is_half {
            // Read Acc1024 (cm-register = 16 lanes), SRS each 8-lane half.
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

            // A wide accumulator SOURCE (1024-bit cm) does NOT imply a wide
            // vector DEST. SRS reduces each acc lane to one output element, so
            // a full 16-lane cm yields 16 outputs: with int16 that is 256 bits
            // = a single 256-bit `mWa` w-register (`vsrs.s16.s64`); only int32
            // output (512 bits) needs the x-register pair. `is_wide_vector`
            // carries the decoder's dest-class width -- write wide only then.
            if op.is_wide_vector {
                Self::write_wide_vec_dest(op, ctx, result);
            } else {
                let narrow: [u32; 8] = result[..8].try_into().unwrap();
                Self::write_vector_dest(op, ctx, narrow);
            }
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
        let mode = RoundingMode::from_raw(cfg.rounding_mode).unwrap_or(RoundingMode::PosInf);
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
                    let out = srs_lane(val, shift, signed_output, 32, saturate, sym_sat, mode);
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
                        let out0 = srs_lane(val0, shift, signed_output, 16, saturate, sym_sat, mode);
                        let out1 = srs_lane(val1, shift, signed_output, 16, saturate, sym_sat, mode);
                        result[i] = (out0 as u16 as u32) | ((out1 as u16 as u32) << 16);
                    }
                } else {
                    // Acc32 -> 16-bit (2:1 reduction): 16 lanes packed 2 per u64.
                    // Extract lo32 and hi32 from each u64 to get 16 acc32 lanes,
                    // then SRS each to 16-bit and pack 2 per u32 result word.
                    for i in 0..8 {
                        let lo_val = mask_value(acc[i] & 0xFFFF_FFFF);
                        let hi_val = mask_value(acc[i] >> 32);
                        let out_lo = srs_lane(lo_val, shift, signed_output, 16, saturate, sym_sat, mode);
                        let out_hi = srs_lane(hi_val, shift, signed_output, 16, saturate, sym_sat, mode);
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
                        let out = srs_lane(val, shift, signed_output, 8, saturate, sym_sat, mode);
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

/// Integration tests that exercise SRS through the full VectorAlu dispatch path.
#[cfg(test)]
mod integration_tests {
    use crate::interpreter::bundle::{ElementType, Operand, SlotIndex, SlotOp};
    use crate::interpreter::execute::vector_dispatch::VectorAlu;
    use crate::interpreter::state::{ExecutionContext, SrsConfig};
    use xdna_archspec::aie2::isa::SemanticOp;

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
        ctx.accumulator.write(0, [0xDEAD_BEEF_0000_0064, 0, 0, 0, 0, 0, 0, 0]);

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

    /// `vsrs.s16.s64 wh0, cm5, s0`: a full 1024-bit `cm` source (16 acc64
    /// lanes) reduced to int16 yields 16 outputs = 256 bits = a SINGLE 256-bit
    /// `mWa` destination, not a 512-bit x-register pair. The accumulator source
    /// being wide must not force a wide vector write. Regression for the Half-B
    /// SRS capture kernel, which panicked here (`write_wide` to odd base 1).
    #[test]
    fn test_srs_full_cm_to_d16_narrow_dest() {
        use crate::interpreter::decode::register_map::AccumWidth;

        let mut ctx = make_ctx();
        // cm = {bml (AccumReg 0), bmh (AccumReg 1)} = 16 lanes.
        ctx.accumulator.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.accumulator.write(1, [9, 10, 11, 12, 13, 14, 15, 16]);

        // Mirror the decoder output: Srs, Int16, dest wh0 = VectorReg(1) (odd
        // half-register), source cm = AccumReg(0) Full, shift s0, from Int64,
        // is_wide_vector = false (the dest is a single 256-bit w-register).
        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int16)
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(40));
        op.from_type = Some(ElementType::Int64);
        op.accum_width = Some(AccumWidth::Full);
        op.is_wide_vector = false;

        ctx.srs_config.rounding_mode = 0;
        ctx.srs_config.saturation_mode = 0;
        ctx.srs_config.srs_sign = true;

        VectorAlu::execute(&op, &mut ctx);

        // 16 int16 [1..=16] packed 2-per-u32 into the single dest register.
        let v1 = ctx.vector.read(1);
        let expected: [u32; 8] = [
            1 | (2 << 16),
            3 | (4 << 16),
            5 | (6 << 16),
            7 | (8 << 16),
            9 | (10 << 16),
            11 | (12 << 16),
            13 | (14 << 16),
            15 | (16 << 16),
        ];
        assert_eq!(v1, expected, "full-cm s16.s64 must write 256 bits to VectorReg(1)");
    }

    #[test]
    fn test_srs_from_acc32_to_d16() {
        let mut acc = [0u64; 8];
        for i in 0..8usize {
            let lo = (100 + i * 2) as u64;
            let hi = (100 + i * 2 + 1) as u64;
            acc[i] = lo | (hi << 32);
        }

        let cfg = SrsConfig { rounding_mode: 0, saturation_mode: 0, srs_sign: true };

        let result = VectorAlu::vector_srs_from_acc(&acc, 0, ElementType::Int32, ElementType::Int16, &cfg);

        for i in 0..8 {
            let expected_lo = (100 + i * 2) as u16;
            let expected_hi = (100 + i * 2 + 1) as u16;
            let expected = (expected_lo as u32) | ((expected_hi as u32) << 16);
            assert_eq!(
                result[i], expected,
                "result[{}]: got {:#010x}, expected {:#010x}",
                i, result[i], expected
            );
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

        let cfg = SrsConfig { rounding_mode: 0, saturation_mode: 0, srs_sign: true };

        let result = VectorAlu::vector_srs_from_acc(&acc, 0, ElementType::Int32, ElementType::BFloat16, &cfg);

        // result[0] = bf16(1.0) | bf16(2.0) << 16 = 0x3F80 | 0x4000_0000
        assert_eq!(result[0] & 0xFFFF, 0x3F80); // bf16 for 1.0
        assert_eq!(result[0] >> 16, 0x4000); // bf16 for 2.0
        assert_eq!(result[1] & 0xFFFF, 0x4040); // bf16 for 3.0
        assert_eq!(result[1] >> 16, 0x4080); // bf16 for 4.0
    }

    #[test]
    fn test_srs_float32_passthrough() {
        // Float32 SRS: passes through the low 32 bits of each u64.
        let mut acc = [0u64; 8];
        acc[0] = 0xDEAD_BEEF_3F80_0000; // low 32 = 0x3F800000 (1.0f)
        acc[1] = 0xCAFE_BABE_4000_0000; // low 32 = 0x40000000 (2.0f)

        let cfg = SrsConfig { rounding_mode: 0, saturation_mode: 0, srs_sign: true };

        let result = VectorAlu::vector_srs_from_acc(&acc, 0, ElementType::Int32, ElementType::Float32, &cfg);

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

        let cfg = SrsConfig { rounding_mode: 0, saturation_mode: 0, srs_sign: true };

        let result = VectorAlu::vector_srs_from_acc(&acc, 0, ElementType::Int64, ElementType::Int16, &cfg);

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

        let cfg = SrsConfig { rounding_mode: 0, saturation_mode: 0, srs_sign: true };

        let result = VectorAlu::vector_srs_from_acc(&acc, 0, ElementType::Int32, ElementType::Int8, &cfg);

        for i in 0..4 {
            let mut expected = 0u32;
            for j in 0..4 {
                let lane_idx = i * 4 + j;
                let val = (10 + lane_idx) as u8;
                expected |= (val as u32) << (j * 8);
            }
            assert_eq!(
                result[i], expected,
                "result[{}]: got {:#010x}, expected {:#010x}",
                i, result[i], expected
            );
        }
    }
}

#[cfg(test)]
mod lane_tests {
    use super::*;
    use crate::interpreter::execute::silicon_golden::{kernels_with_prefix, SiliconCase};
    use RoundingMode::*;

    /// Inputs in quarters: v / 4 = -1.75, -1.5, -1.25, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.5.
    const QUARTERS: [i128; 13] = [-7, -6, -5, -2, -1, 0, 1, 2, 3, 5, 6, 7, 10];

    fn check_table(mode: RoundingMode, expected: [i128; 13]) {
        let got: Vec<i128> = QUARTERS.iter().map(|&v| shift_round(v, 2, mode)).collect();
        assert_eq!(got, expected, "{mode:?} on v/4 for v in {QUARTERS:?}");
    }

    #[test]
    fn floor_rounds_toward_negative_infinity() {
        check_table(Floor, [-2, -2, -2, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2]);
    }

    #[test]
    fn ceil_rounds_up_whenever_inexact() {
        check_table(Ceil, [-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]);
    }

    #[test]
    fn sym_floor_rounds_toward_zero() {
        check_table(SymFloor, [-1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2]);
    }

    #[test]
    fn sym_ceil_rounds_away_from_zero() {
        check_table(SymCeil, [-2, -2, -2, -1, -1, 0, 1, 1, 1, 2, 2, 2, 3]);
    }

    #[test]
    fn neg_inf_ties_go_down() {
        check_table(NegInf, [-2, -2, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn pos_inf_ties_go_up() {
        check_table(PosInf, [-2, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3]);
    }

    #[test]
    fn sym_zero_ties_go_toward_zero() {
        check_table(SymZero, [-2, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn sym_inf_ties_go_away_from_zero() {
        check_table(SymInf, [-2, -2, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 3]);
    }

    #[test]
    fn conv_even_ties_go_to_even() {
        check_table(ConvEven, [-2, -2, -1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn conv_odd_ties_go_to_odd() {
        check_table(ConvOdd, [-2, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 2, 3]);
    }

    #[test]
    fn inexact_even_picks_the_even_neighbour_for_any_inexact_value() {
        check_table(InexactEven, [-2, -2, -2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]);
    }

    #[test]
    fn inexact_odd_picks_the_odd_neighbour_for_any_inexact_value() {
        check_table(InexactOdd, [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 3]);
    }

    #[test]
    fn shift_zero_is_exact_in_every_mode() {
        for raw in 0..16u8 {
            let mode = RoundingMode::from_raw(raw).unwrap();
            for v in [-5i128, -1, 0, 1, 7] {
                assert_eq!(shift_round(v, 0, mode), v, "{mode:?}");
            }
        }
    }

    #[test]
    fn huge_shift_leaves_only_the_sign_or_a_rounding_carry() {
        assert_eq!(shift_round(-1, 200, Floor), -1);
        assert_eq!(shift_round(5, 200, Floor), 0);
        assert_eq!(shift_round(5, 200, Ceil), 1);
    }

    fn srs16(v: i64, sat: bool, sym: bool) -> i64 {
        srs_lane(v, 0, true, 16, sat, sym, Floor)
    }

    #[test]
    fn no_saturation_wraps_to_the_output_width() {
        assert_eq!(srs16(32768, false, false), -32768);
        assert_eq!(srs16(65537, false, false), 1);
        assert_eq!(srs16(-32769, false, false), 32767);
    }

    #[test]
    fn saturation_clamps_to_the_full_signed_range() {
        assert_eq!(srs16(40000, true, false), 32767);
        assert_eq!(srs16(-40000, true, false), -32768);
        assert_eq!(srs16(-32768, true, false), -32768);
    }

    #[test]
    fn symmetric_saturation_excludes_the_most_negative_value() {
        assert_eq!(srs16(40000, true, true), 32767);
        assert_eq!(srs16(-40000, true, true), -32767);
        assert_eq!(srs16(-32768, true, true), -32767);
        assert_eq!(srs16(-32767, true, true), -32767);
    }

    #[test]
    fn unsigned_output_clamps_to_zero_and_max_and_zero_extends() {
        assert_eq!(srs_lane(-5, 0, false, 8, true, false, Floor), 0);
        assert_eq!(srs_lane(300, 0, false, 8, true, false, Floor), 255);
        assert_eq!(srs_lane(300, 0, false, 8, true, true, Floor), 255);
        assert_eq!(srs_lane(300, 0, false, 8, false, false, Floor), 44);
        assert_eq!(srs_lane(-1, 0, false, 16, false, false, Floor), 65535);
    }

    #[test]
    fn rounding_carry_is_saturated_not_wrapped() {
        // 32767.5 rounds up to 32768, which then clamps.
        assert_eq!(srs_lane(65535, 1, true, 16, true, false, PosInf), 32767);
    }

    #[test]
    fn full_width_accumulator_values_do_not_overflow() {
        assert_eq!(srs_lane(i64::MAX, 0, true, 32, true, false, Floor), i32::MAX as i64);
        assert_eq!(srs_lane(i64::MIN, 0, true, 32, true, false, Floor), i32::MIN as i64);
        assert_eq!(srs_lane(i64::MAX, 1, true, 64, false, false, Ceil), 1i64 << 62);
    }

    /// Every captured int32 -> int16 SRS kernel, lane by lane against silicon.
    #[test]
    fn matches_every_silicon_srs_capture() {
        let kernels = kernels_with_prefix("vec_srs_i32");
        assert!(kernels.len() >= 70, "expected the full SRS sweep, found {}", kernels.len());
        for kernel in kernels {
            let case = SiliconCase::load(&kernel, "srs.cc");
            let shift = case.define("SRS_SHIFT");
            let (sat, sym) = case.saturation();
            let mode = case.rounding();
            for (i, (&input, &want)) in case.inputs.iter().zip(&case.silicon).enumerate() {
                let got = srs_lane(input, shift, true, 16, sat, sym, mode);
                assert_eq!(
                    got, want,
                    "{kernel} lane {i}: srs({input}, shift {shift}, {mode:?}, sat {sat}, sym {sym})"
                );
            }
        }
    }
}
