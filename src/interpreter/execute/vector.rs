//! Vector ALU execution unit.
//!
//! Handles SIMD operations on 256-bit vectors (8 x 32-bit, 16 x 16-bit, 32 x 8-bit).
//!
//! # Execution Flow
//!
//! ```text
//! CycleAccurateExecutor::execute_slot()
//!         |
//!         v
//!   execute_semantic(op, ctx)  <-- scalar register ops
//!         |
//!         | returns false (not a scalar op)
//!         v
//!   VectorAlu::execute(op, ctx)  <-- SIMD operations (this module)
//! ```
//!
//! ## Future Direction
//!
//! Many vector operations are element-wise versions of scalar operations.
//! The plan is to eventually have vector ops call scalar semantic handlers
//! per-element, reducing ~3,000 lines to ~200 lines of thin wrappers.
//!
//! # Operations
//!
//! - **Arithmetic**: vadd, vsub, vmul (element-wise)
//! - **MAC**: vmac (multiply-accumulate to 512-bit accumulator)
//! - **MatMul**: Dense/sparse matrix multiply for ML workloads
//! - **Compare**: vcmp, vmin, vmax
//! - **Shuffle**: vshuffle, vpack, vunpack

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::tablegen::SemanticOp;

use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    /// Execute one 256-bit half of a vector operation.
    pub(super) fn execute_half(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        _et: ElementType,
    ) -> bool {

        match semantic {
            // ========== Arithmetic ==========

            // Add, Sub, Mul, Min, Max: extracted to vector_dispatch.rs

            SemanticOp::Mac | SemanticOp::MatMul | SemanticOp::MatMulSub
            | SemanticOp::NegMul | SemanticOp::NegMatMul
            | SemanticOp::AddMac | SemanticOp::SubMac => {
                return super::vector_matmul::execute_matmul(op, ctx);
            }

            // Neg: extracted to vector_arith.rs

            // Shuffle, Pack, Unpack: extracted to vector_misc.rs / vector_pack.rs

            // Comparison ops (Cmp, SetGe, SetLt, SetEq, MaxLt, MinGe): extracted to vector_compare.rs

            // SRS/UPS/Convert: extracted to vector_srs.rs, vector_ups.rs, vector_convert.rs

            // Copy, VectorExtract, VectorInsert, VectorClear, VectorBroadcast,
            // Align: extracted to vector_misc.rs
            // VectorSelect: extracted to vector_compare.rs
            // And, Or, Xor, Not: extracted to vector_misc.rs (typeless dispatchers)
            // Shl, Srl, Sra: extracted to vector_arith.rs
            // AbsGtz, NegGtz, NegLtz, Accumulate/AccumSub/AccumNegAdd/AccumNegSub/NegAdd:
            // extracted to vector_arith.rs
            // SubLt, SubGe, MaxDiffLt: extracted to vector_arith.rs

            _ => false, // Not a vector operation handled here
        }
    }

/// Accumulator-to-accumulator add/subtract.
    ///
    /// Handles VADD, VSUB, VNEGADD, VNEGSUB and their .f (float) variants.
    /// These take two accumulator (cm-class, 1024-bit) sources and write
    /// a 1024-bit result.
    ///
    /// The config register (scalar operand) controls:
    /// - Bit  0: zero_acc1 (zero acc1 before operation)
    /// - Bit 10: shift16 (right-shift result by 16 bits)
    /// - Bit 11: sub_acc1 (negate acc1)
    /// - Bit 12: sub_acc2 (negate acc2)
    ///
    /// Base operations (before config modifiers):
    /// - vadd:    dst = acc1 + acc2
    /// - vsub:    dst = acc1 - acc2
    /// - vnegadd: dst = -acc1 + acc2
    /// - vnegsub: dst = -acc1 - acc2
    pub(super) fn execute_acc_add_sub(op: &SlotOp, ctx: &mut ExecutionContext) {
        // Collect the two AccumReg source indices.
        let mut acc_sources: Vec<u8> = Vec::new();
        for src in &op.sources {
            if let Operand::AccumReg(r) = src {
                acc_sources.push(*r);
            }
        }

        let acc1_reg = if !acc_sources.is_empty() { acc_sources[0] } else { 0 };
        let acc2_reg = if acc_sources.len() >= 2 { acc_sources[1] } else { acc1_reg };

        // Read config register.
        let conf = Self::get_config_register(op, ctx).unwrap_or(0);
        let zero_acc1 = (conf & 1) != 0;
        let shift16 = ((conf >> 10) & 1) != 0;
        let sub_acc1 = ((conf >> 11) & 1) != 0;
        let sub_acc2 = ((conf >> 12) & 1) != 0;

        // Determine md1/md2 encoding bits from SemanticOp.
        //
        // The hardware VACC ALU has two mode bits (md1, md2) that XOR with the
        // config register's sub_acc1/sub_acc2 bits to determine the effective
        // sign of each accumulator operand:
        //   negate_acc1 = md1 XOR sub_acc1
        //   negate_acc2 = md2 XOR sub_acc2
        //
        // Encoding:       md1  md2   Operation
        //   VADD           0    0    +acc1 + acc2
        //   VSUB           0    1    +acc1 - acc2
        //   VNEGSUB        1    0    -acc1 + acc2
        //   VNEGADD        1    1    -acc1 - acc2 = -(acc1 + acc2)
        //
        // Note: VNEGADD negates BOTH operands (the "add" refers to the
        // unsigned operation, "neg" flips both signs). VNEGSUB negates only
        // acc1 (the "sub" flips one sign, "neg" flips another, net one flip).
        let semantic = op.semantic.unwrap_or(SemanticOp::Accumulate);
        let enc_lower = op.encoding_name.as_deref().unwrap_or("").to_ascii_lowercase();
        // Map semantic -> (md1, md2).  Accept both AccumXxx (build-time) and
        // generic NegAdd (runtime mnemonic dispatch, no NegSub variant exists).
        let (md1, md2) = match semantic {
            SemanticOp::Accumulate => (false, false),                    // VADD
            SemanticOp::AccumSub => (false, true),                       // VSUB
            SemanticOp::AccumNegAdd => (true, true),                     // VNEGADD
            SemanticOp::AccumNegSub => (true, false),                    // VNEGSUB
            SemanticOp::NegAdd => {
                // Runtime mnemonic dispatch -- both vnegadd and vnegsub map to
                // NegAdd.  Distinguish via encoding name.
                if enc_lower.contains("negsub") {
                    (true, false)                                        // VNEGSUB
                } else {
                    (true, true)                                         // VNEGADD
                }
            }
            _ => (false, false),
        };
        // Float mode from element_type (set at build time from mnemonic ".f" suffix).
        let is_float = matches!(op.element_type, Some(ElementType::Float32) | Some(ElementType::BFloat16));

        // Wide (cm, 1024-bit) vs narrow (bm, 512-bit) accumulator.
        // Use AccumWidth metadata from decoder when available (structural signal).
        // Fallback: cm registers always use even base indices, so odd index implies bm.
        let dst_reg = Self::get_acc_dest(op);
        let is_wide = match op.accum_width {
            Some(crate::tablegen::decoder_ffi::AccumWidth::Full) => true,
            Some(crate::tablegen::decoder_ffi::AccumWidth::Half)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterLow)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterHigh) => false,
            None => {
                // Legacy fallback: if any index is odd, it must be a bm half.
                let any_odd = acc_sources.iter().any(|r| r % 2 != 0) || dst_reg % 2 != 0;
                !any_odd
            }
        };

        // Compute effective signs: md bits XOR config bits (mirrors hardware).
        let negate_acc1 = md1 ^ sub_acc1;
        let negate_acc2 = md2 ^ sub_acc2;

        if is_wide {
            // Wide path: 1024-bit cm registers (16 x 64-bit lanes).
            let a1 = ctx.accumulator.read_wide(acc1_reg);
            let a2 = ctx.accumulator.read_wide(acc2_reg);
            let mut result = [0u64; 16];

            if is_float {
                use super::vector_float::{fp32_flush_to_zero, aie2_acc_fp32_add};
                for i in 0..16 {
                    let mut a1_lo = if zero_acc1 { 0u32 } else { fp32_flush_to_zero(a1[i] as u32) };
                    let mut a1_hi = if zero_acc1 { 0u32 } else { fp32_flush_to_zero((a1[i] >> 32) as u32) };
                    let mut a2_lo = fp32_flush_to_zero(a2[i] as u32);
                    let mut a2_hi = fp32_flush_to_zero((a2[i] >> 32) as u32);

                    // Negate by flipping sign bit (works for zero, normal, inf, NaN).
                    if negate_acc1 { a1_lo ^= 0x8000_0000; a1_hi ^= 0x8000_0000; }
                    if negate_acc2 { a2_lo ^= 0x8000_0000; a2_hi ^= 0x8000_0000; }

                    // Use acc ALU add (no output FTZ): the accumulator
                    // register file preserves denormalized fp32 values.
                    let r_lo = aie2_acc_fp32_add(a1_lo, a2_lo);
                    let r_hi = aie2_acc_fp32_add(a1_hi, a2_hi);
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            } else {
                // Acc32 mode: each u64 holds two independent 32-bit accumulator
                // lanes.  The hardware ALU has no carry chain between the lo and
                // hi halves -- all arithmetic is per-lane 32-bit.
                for i in 0..16 {
                    let v1_lo = if zero_acc1 { 0i32 } else { a1[i] as i32 };
                    let v1_hi = if zero_acc1 { 0i32 } else { (a1[i] >> 32) as i32 };
                    let v2_lo = a2[i] as i32;
                    let v2_hi = (a2[i] >> 32) as i32;
                    let v1_lo = if negate_acc1 { v1_lo.wrapping_neg() } else { v1_lo };
                    let v1_hi = if negate_acc1 { v1_hi.wrapping_neg() } else { v1_hi };
                    let v2_lo = if negate_acc2 { v2_lo.wrapping_neg() } else { v2_lo };
                    let v2_hi = if negate_acc2 { v2_hi.wrapping_neg() } else { v2_hi };
                    let mut r_lo = v1_lo.wrapping_add(v2_lo);
                    let mut r_hi = v1_hi.wrapping_add(v2_hi);
                    if shift16 { r_lo >>= 16; r_hi >>= 16; }
                    result[i] = (r_lo as u32 as u64) | ((r_hi as u32 as u64) << 32);
                }
            }

            ctx.accumulator.write_wide(dst_reg, result);
        } else {
            // Narrow path: 512-bit bm registers (8 x 64-bit lanes).
            let a1 = ctx.accumulator.read(acc1_reg);
            let a2 = ctx.accumulator.read(acc2_reg);
            let mut result = [0u64; 8];

            if is_float {
                use super::vector_float::{fp32_flush_to_zero, aie2_acc_fp32_add};
                for i in 0..8 {
                    let mut a1_lo = if zero_acc1 { 0u32 } else { fp32_flush_to_zero(a1[i] as u32) };
                    let mut a1_hi = if zero_acc1 { 0u32 } else { fp32_flush_to_zero((a1[i] >> 32) as u32) };
                    let mut a2_lo = fp32_flush_to_zero(a2[i] as u32);
                    let mut a2_hi = fp32_flush_to_zero((a2[i] >> 32) as u32);

                    if negate_acc1 { a1_lo ^= 0x8000_0000; a1_hi ^= 0x8000_0000; }
                    if negate_acc2 { a2_lo ^= 0x8000_0000; a2_hi ^= 0x8000_0000; }

                    let r_lo = aie2_acc_fp32_add(a1_lo, a2_lo);
                    let r_hi = aie2_acc_fp32_add(a1_hi, a2_hi);
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            } else {
                // Acc32 mode: independent 32-bit lane operations (see wide path).
                for i in 0..8 {
                    let v1_lo = if zero_acc1 { 0i32 } else { a1[i] as i32 };
                    let v1_hi = if zero_acc1 { 0i32 } else { (a1[i] >> 32) as i32 };
                    let v2_lo = a2[i] as i32;
                    let v2_hi = (a2[i] >> 32) as i32;
                    let v1_lo = if negate_acc1 { v1_lo.wrapping_neg() } else { v1_lo };
                    let v1_hi = if negate_acc1 { v1_hi.wrapping_neg() } else { v1_hi };
                    let v2_lo = if negate_acc2 { v2_lo.wrapping_neg() } else { v2_lo };
                    let v2_hi = if negate_acc2 { v2_hi.wrapping_neg() } else { v2_hi };
                    let mut r_lo = v1_lo.wrapping_add(v2_lo);
                    let mut r_hi = v1_hi.wrapping_add(v2_hi);
                    if shift16 { r_lo >>= 16; r_hi >>= 16; }
                    result[i] = (r_lo as u32 as u64) | ((r_hi as u32 as u64) << 32);
                }
            }

            ctx.accumulator.write(dst_reg, result);
        }
    }

    /// Accumulator negate: dst = -src.
    ///
    /// Handles VNEG on both bm (512-bit) and cm (1024-bit) accumulators.
    /// The config word controls zero_acc (zero the source before negation,
    /// producing 0). Float mode is detected from element_type (Float32/BFloat16):
    /// each u64 lane holds two fp32 values (bits [31:0] and bits [63:32]).
    pub(super) fn execute_acc_negate(op: &SlotOp, ctx: &mut ExecutionContext) {
        let acc_reg = op.sources.iter().find_map(|s| match s {
            Operand::AccumReg(r) => Some(*r),
            _ => None,
        }).unwrap_or(0);
        let dst_reg = Self::get_acc_dest(op);

        let conf = Self::get_config_register(op, ctx).unwrap_or(0);
        let zero_acc = (conf & 1) != 0;
        // Float mode from element_type (set at build time from mnemonic ".f" suffix).
        let is_float = matches!(op.element_type, Some(ElementType::Float32) | Some(ElementType::BFloat16));

        // Wide (cm) vs narrow (bm) -- same detection as execute_acc_add_sub.
        // Use AccumWidth metadata from decoder when available (structural signal).
        let is_wide = match op.accum_width {
            Some(crate::tablegen::decoder_ffi::AccumWidth::Full) => true,
            Some(crate::tablegen::decoder_ffi::AccumWidth::Half)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterLow)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterHigh) => false,
            None => {
                let any_odd = acc_reg % 2 != 0 || dst_reg % 2 != 0;
                !any_odd
            }
        };

        if is_wide {
            let src = ctx.accumulator.read_wide(acc_reg);
            let mut result = [0u64; 16];
            if is_float {
                use super::vector_float::{fp32_flush_to_zero, fp32_is_nan, fp32_make_nan};
                for i in 0..16 {
                    let lo = fp32_flush_to_zero(src[i] as u32);
                    let hi = fp32_flush_to_zero((src[i] >> 32) as u32);
                    let r_lo = if zero_acc { 0u32 }
                        else if fp32_is_nan(lo) { fp32_make_nan(false) }
                        else { lo ^ 0x8000_0000 };
                    let r_hi = if zero_acc { 0u32 }
                        else if fp32_is_nan(hi) { fp32_make_nan(false) }
                        else { hi ^ 0x8000_0000 };
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            } else {
                // Acc32 mode: independent 32-bit negation per lane half.
                for i in 0..16 {
                    let lo = if zero_acc { 0i32 } else { src[i] as i32 };
                    let hi = if zero_acc { 0i32 } else { (src[i] >> 32) as i32 };
                    let r_lo = lo.wrapping_neg() as u32;
                    let r_hi = hi.wrapping_neg() as u32;
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            }
            ctx.accumulator.write_wide(dst_reg, result);
        } else {
            let src = ctx.accumulator.read(acc_reg);
            let mut result = [0u64; 8];
            if is_float {
                use super::vector_float::{fp32_flush_to_zero, fp32_is_nan, fp32_make_nan};
                for i in 0..8 {
                    let lo = fp32_flush_to_zero(src[i] as u32);
                    let hi = fp32_flush_to_zero((src[i] >> 32) as u32);
                    let r_lo = if zero_acc { 0u32 }
                        else if fp32_is_nan(lo) { fp32_make_nan(false) }
                        else { lo ^ 0x8000_0000 };
                    let r_hi = if zero_acc { 0u32 }
                        else if fp32_is_nan(hi) { fp32_make_nan(false) }
                        else { hi ^ 0x8000_0000 };
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            } else {
                // Acc32 mode: independent 32-bit negation per lane half.
                for i in 0..8 {
                    let lo = if zero_acc { 0i32 } else { src[i] as i32 };
                    let hi = if zero_acc { 0i32 } else { (src[i] >> 32) as i32 };
                    let r_lo = lo.wrapping_neg() as u32;
                    let r_hi = hi.wrapping_neg() as u32;
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            }
            ctx.accumulator.write(dst_reg, result);
        }
    }

    // vector_srs, vector_srs_from_acc: moved to vector_srs.rs
    // vector_convert: moved to vector_convert.rs

    /// Execute a 512-bit wide vector operation.
    ///
    /// Unlike execute_half which processes 256-bit chunks independently,
    /// this reads full 512-bit inputs and writes full 512-bit outputs.
    /// Element-wise ops use the bridge to reuse narrow math. Cross-half
    /// ops have dedicated implementations (added in later tasks).
    pub(super) fn execute_wide(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        et: ElementType,
    ) -> bool {
        match semantic {
            // Add, Sub, Mul, Min, Max: extracted to vector_dispatch.rs
            // VectorSelect: extracted to vector_compare.rs

            // Neg: extracted to vector_arith.rs

            // And, Or, Xor, Not: extracted to vector_misc.rs (typeless dispatchers)
            // Comparison ops (Cmp, SetGe, SetLt): extracted to vector_compare.rs

            // ========== Matrix Multiply (config-driven) ==========
            SemanticOp::Mac | SemanticOp::MatMul | SemanticOp::MatMulSub
            | SemanticOp::NegMul | SemanticOp::NegMatMul
            | SemanticOp::AddMac | SemanticOp::SubMac => {
                return super::vector_matmul::execute_matmul(op, ctx);
            }

            // Accumulate/AccumSub/AccumNegAdd/AccumNegSub/NegAdd: extracted to vector_arith.rs
            // SRS/UPS/Convert: extracted to vector_srs.rs, vector_ups.rs, vector_convert.rs

            // Copy, VectorClear, VectorExtract, VectorPush, VectorPushHi,
            // VectorInsert, VectorBroadcast, Align, Shuffle:
            // extracted to vector_misc.rs

            // Shl, Srl, Sra, MaxLt, MinGe, SubLt, SubGe, MaxDiffLt,
            // NegGtz, NegLtz, AbsGtz: extracted to vector_arith.rs
            // SetEq: extracted to vector_compare.rs

            // Pack, Unpack: extracted to vector_pack.rs

            // ========== Fallback ==========
            _ => Self::execute_wide_fallback(op, ctx, semantic, et),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;
    use crate::interpreter::state::SrsConfig;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    // Arithmetic tests (add, sub, mul, min, max) moved to vector_arith.rs
    // Shuffle, push, extract, insert, align, mov tests: moved to vector_misc.rs
    // test_vector_cmp: moved to vector_compare.rs
    // Float arithmetic tests (add_f32, mul_f32, min_max_f32, add_bf16) moved to vector_arith.rs

    #[test]
    fn test_vector_srs_int32() {
        let mut ctx = make_ctx();
        // Accumulator: one value per u64 lane.
        ctx.accumulator.write(0, [256, 512, 768, 1024, 1280, 1536, 1792, 2048]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(4)); // Shift right by 4
        op.from_type = Some(ElementType::Int32);

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(0);
        // 256 >> 4 = 16, 512 >> 4 = 32, etc (with rounding)
        assert_eq!(result[0], 16);
        assert_eq!(result[1], 32);
        assert_eq!(result[2], 48);
        assert_eq!(result[3], 64);
    }

    #[test]
    fn test_vector_ups_srs_roundtrip() {
        let mut ctx = make_ctx();
        // UPS+SRS round-trip: 16 x i16 input -> Acc32 (16 lanes, 2 per u64)
        // -> SRS back to 16 x i16. All 16 lanes should survive the round-trip.
        let src = [0x0002_0001u32, 0x0004_0003, 0x0006_0005, 0x0008_0007,
                    0x000A_0009, 0x000C_000B, 0x000E_000D, 0x0010_000F];
        // UPS with shift=0: sign-extend 16 i16 values to i32 in accumulator
        let acc = crate::interpreter::execute::vector_ups::ups_vector_to_acc(
            &src, 0, ElementType::Int16, ElementType::Int32,
        );
        ctx.accumulator.write(0, acc);

        // SRS with shift=0
        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int16)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(0));
        op.from_type = Some(ElementType::Int32);

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(0);
        // Acc32 mode processes all 16 lanes. SRS produces 16 x i16
        // packed into 8 u32 words -- a perfect round-trip.
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
        // Wide UPS+SRS round-trip: 32 x i8 input -> cm0 (Acc32 wide, 32 lanes)
        // -> SRS back to 32 x i8. Simulates vlda.ups.s32.s8 + vst.srs.s8.s32.
        //
        // 32 bytes of i8 data packed into 8 u32 words:
        let src = [
            0x04030201u32, 0x08070605, 0x0C0B0A09, 0x100F0E0D,
            0x14131211, 0x18171615, 0x1C1B1A19, 0x201F1E1D,
        ];

        // UPS wide: 32 x i8 -> 32 x i32 in cm (1024-bit accumulator)
        let acc_wide = crate::interpreter::execute::vector_ups::ups_vector_to_acc_wide(
            &src, 0, ElementType::Int8, ElementType::Int32,
        );

        // Verify UPS wrote non-zero data
        assert!(acc_wide.iter().any(|&v| v != 0), "UPS produced all zeros");

        // SRS each half (split at word 8 boundary, matching fused handler)
        let acc_lo: [u64; 8] = acc_wide[..8].try_into().unwrap();
        let acc_hi: [u64; 8] = acc_wide[8..].try_into().unwrap();

        let cfg = SrsConfig::default();
        let result_lo = VectorAlu::vector_srs_from_acc(
            &acc_lo, 0, ElementType::Int32, ElementType::Int8, &cfg,
        );
        let result_hi = VectorAlu::vector_srs_from_acc(
            &acc_hi, 0, ElementType::Int32, ElementType::Int8, &cfg,
        );

        // Pack like the fused handler does
        let lanes_per_half = 16usize; // from_bits=32 <= 32
        let to_bits = 8usize;
        let words_per_half = (lanes_per_half * to_bits + 31) / 32;
        let n = words_per_half.min(8);
        assert_eq!(n, 4, "should be 4 words per half for 16 x i8");

        let mut packed = [0u32; 8];
        packed[..n].copy_from_slice(&result_lo[..n]);
        packed[n..n + n].copy_from_slice(&result_hi[..n]);

        // The round-trip should recover the original input
        assert_eq!(packed, src, "wide s8.s32 UPS->SRS round-trip failed");

        // Also test with non-zero shift (common in real code)
        let acc_shifted = crate::interpreter::execute::vector_ups::ups_vector_to_acc_wide(
            &src, 4, ElementType::Int8, ElementType::Int32,
        );
        let acc_s_lo: [u64; 8] = acc_shifted[..8].try_into().unwrap();
        let acc_s_hi: [u64; 8] = acc_shifted[8..].try_into().unwrap();
        let res_s_lo = VectorAlu::vector_srs_from_acc(
            &acc_s_lo, 4, ElementType::Int32, ElementType::Int8, &cfg,
        );
        let res_s_hi = VectorAlu::vector_srs_from_acc(
            &acc_s_hi, 4, ElementType::Int32, ElementType::Int8, &cfg,
        );
        let mut packed_s = [0u32; 8];
        packed_s[..4].copy_from_slice(&res_s_lo[..4]);
        packed_s[4..8].copy_from_slice(&res_s_hi[..4]);
        assert_eq!(packed_s, src, "wide s8.s32 UPS->SRS round-trip with shift=4 failed");
    }

    #[test]
    fn test_vector_convert_bf16_to_f32() {
        let mut ctx = make_ctx();
        // Create bf16 values: 1.0, 2.0, 3.0, 4.0 (packed 2 per u32)
        let bf16_1 = VectorAlu::f32_to_bf16(1.0);
        let bf16_2 = VectorAlu::f32_to_bf16(2.0);
        let bf16_3 = VectorAlu::f32_to_bf16(3.0);
        let bf16_4 = VectorAlu::f32_to_bf16(4.0);
        ctx.vector.write(
            0,
            [
                (bf16_1 as u32) | ((bf16_2 as u32) << 16),
                (bf16_3 as u32) | ((bf16_4 as u32) << 16),
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        );

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Convert)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::VectorReg(0));
        op.from_type = Some(ElementType::BFloat16);

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(1);

        // First 4 lanes should be 1.0, 2.0, 3.0, 4.0
        let f0 = f32::from_bits(result[0]);
        let f1 = f32::from_bits(result[1]);
        let f2 = f32::from_bits(result[2]);
        let f3 = f32::from_bits(result[3]);
        assert!((f0 - 1.0).abs() < 0.01, "Expected 1.0, got {}", f0);
        assert!((f1 - 2.0).abs() < 0.01, "Expected 2.0, got {}", f1);
        assert!((f2 - 3.0).abs() < 0.01, "Expected 3.0, got {}", f2);
        assert!((f3 - 4.0).abs() < 0.01, "Expected 4.0, got {}", f3);
    }

    /// VFLOOR: basic floor computation with shift=0.
    #[test]
    fn test_vfloor_bf16_to_s32_shift0() {
        let result = VectorAlu::vector_floor_bf16_to_s32(
            &[
                // Pack BF16 pairs: (1.5, -2.5), (0.0, 100.0), zeros...
                ((VectorAlu::f32_to_bf16(1.5) as u32) | ((VectorAlu::f32_to_bf16(-2.5) as u32) << 16)),
                ((VectorAlu::f32_to_bf16(0.0) as u32) | ((VectorAlu::f32_to_bf16(100.0) as u32) << 16)),
                0, 0, 0, 0, 0, 0,
            ],
            0, // shift=0
        );
        assert_eq!(result[0] as i32, 1, "floor(1.5) = 1");
        assert_eq!(result[1] as i32, -3, "floor(-2.5) = -3");
        assert_eq!(result[2] as i32, 0, "floor(0.0) = 0");
        assert_eq!(result[3] as i32, 100, "floor(100.0) = 100");
    }

    /// VFLOOR: signed 6-bit shift masking. Hardware uses (shift & 0x3F) sign-extended.
    #[test]
    fn test_vfloor_signed_6bit_shift() {
        let bf16_10 = VectorAlu::f32_to_bf16(10.0);
        let src = [bf16_10 as u32, 0, 0, 0, 0, 0, 0, 0];

        // shift=3: floor(10.0 * 8) = 80
        assert_eq!(VectorAlu::vector_floor_bf16_to_s32(&src, 3)[0] as i32, 80);

        // shift=0x23 (35): low 6 bits = 0x23 = 35, signed 6-bit = 35-64 = -29
        // floor(10.0 * 2^-29) = floor(10.0 / 536870912) = 0
        assert_eq!(VectorAlu::vector_floor_bf16_to_s32(&src, 0x23)[0] as i32, 0);

        // shift=0xFFFFFF07 (-249 as i32): low 6 bits = 7, positive
        // floor(10.0 * 128) = 1280
        assert_eq!(VectorAlu::vector_floor_bf16_to_s32(&src, -249)[0] as i32, 1280);

        // shift with bit 5 set: 0x20 = 32, signed 6-bit = 32-64 = -32
        // floor(10.0 * 2^-32) = 0
        assert_eq!(VectorAlu::vector_floor_bf16_to_s32(&src, 0x20)[0] as i32, 0);
    }

    /// VFLOOR: NaN input saturates to INT_MAX.
    #[test]
    fn test_vfloor_nan_saturates_to_int_max() {
        // BF16 0xFFFF = NaN (all exponent + mantissa bits set)
        let src = [0xFFFF_FFFF, 0, 0, 0, 0, 0, 0, 0];
        let result = VectorAlu::vector_floor_bf16_to_s32(&src, 0);
        assert_eq!(result[0], i32::MAX as u32, "NaN -> INT_MAX");
        assert_eq!(result[1], i32::MAX as u32, "NaN -> INT_MAX");
    }

    /// VCONV_BF16_FP32: accumulator source routes to wide path and reads all 16 lanes.
    #[test]
    fn test_vconv_bf16_fp32_acc_source() {
        use crate::tablegen::decoder_ffi::AccumWidth;
        let mut ctx = make_ctx();

        // Write 16 FP32 values into accumulator bml0 as consecutive pairs per lane.
        // Lane[i] = {fp32[2i+1], fp32[2i]}
        let mut acc = [0u64; 8];
        for i in 0..8 {
            let f_lo = ((i * 2 + 1) as f32).to_bits();      // 1.0, 3.0, 5.0, ...
            let f_hi = ((i * 2 + 2) as f32).to_bits();      // 2.0, 4.0, 6.0, ...
            acc[i] = (f_lo as u64) | ((f_hi as u64) << 32);
        }
        ctx.accumulator.write(0, acc);

        // VCONV_BF16_FP32: Convert FP32 (in acc) to BF16 (in vector)
        let mut op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Convert)
            .as_vector(ElementType::BFloat16)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0));
        op.from_type = Some(ElementType::Float32);
        op.accum_width = Some(AccumWidth::Half);

        assert!(VectorAlu::execute(&op, &mut ctx));
        let result = ctx.vector.read(0);

        // Result should have 16 BF16 values packed: bf16(1.0), bf16(2.0), ..., bf16(16.0)
        for i in 0..8 {
            let lo_bf16 = (result[i / 2] >> ((i % 2) * 16)) as u16;
            let expected = VectorAlu::f32_to_bf16((i + 1) as f32);
            assert_eq!(lo_bf16, expected,
                "element {}: expected bf16({}.0)=0x{:04X}, got 0x{:04X}",
                i, i + 1, expected, lo_bf16);
        }
    }

    /// VCONV_FP32_BF16: BF16->FP32 expansion writes consecutive pairs to accumulator.
    #[test]
    fn test_vconv_fp32_bf16_acc_dest() {
        let mut ctx = make_ctx();

        // Write 16 BF16 values (1.0..16.0) into vector register
        let mut vec_data = [0u32; 8];
        for i in 0..8 {
            let bf_lo = VectorAlu::f32_to_bf16((i * 2 + 1) as f32);
            let bf_hi = VectorAlu::f32_to_bf16((i * 2 + 2) as f32);
            vec_data[i] = (bf_lo as u32) | ((bf_hi as u32) << 16);
        }
        ctx.vector.write(0, vec_data);

        // VCONV_FP32_BF16: Convert BF16 (in vector) to FP32 (in acc)
        let mut op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Convert)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::VectorReg(0));
        op.from_type = Some(ElementType::BFloat16);
        op.is_wide_vector = true;

        assert!(VectorAlu::execute(&op, &mut ctx));
        let acc = ctx.accumulator.read(0);

        // Each lane should have consecutive FP32 pair: {fp32[2i+1], fp32[2i]}
        for i in 0..8 {
            let lo = acc[i] as u32;
            let hi = (acc[i] >> 32) as u32;
            let f_lo = f32::from_bits(lo);
            let f_hi = f32::from_bits(hi);
            let expected_lo = (i * 2 + 1) as f32;
            let expected_hi = (i * 2 + 2) as f32;
            assert!((f_lo - expected_lo).abs() < 0.1,
                "lane {} lo: expected {}, got {}", i, expected_lo, f_lo);
            assert!((f_hi - expected_hi).abs() < 0.1,
                "lane {} hi: expected {}, got {}", i, expected_hi, f_hi);
        }
    }

    #[test]
    fn test_vector_convert_int32_to_f32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Convert)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::VectorReg(0));
        op.from_type = Some(ElementType::Int32);

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(1);

        for i in 0..8 {
            let f = f32::from_bits(result[i]);
            assert_eq!(f, (i + 1) as f32);
        }
    }

    // test_vector_mov: moved to vector_misc.rs

    #[test]
    fn test_vector_srs_reads_config() {
        // Verify SRS reads rounding mode from srs_config, not hardcoded.
        let mut ctx = make_ctx();
        // Accumulator lane 0 = 264 (264 >> 4 = 16.5)
        ctx.accumulator.write(0, [264, 0, 0, 0, 0, 0, 0, 0]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(4));
        op.from_type = Some(ElementType::Int32);

        // PosInf (mode 9), saturation on, signed.
        // 16.5 with PosInf -> 17 (round toward +inf at halfway)
        ctx.srs_config.rounding_mode = 9; // PosInf
        ctx.srs_config.saturation_mode = 1; // Saturate
        ctx.srs_config.srs_sign = true; // Signed
        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(0)[0], 17);

        // Switch to Floor rounding (mode 0): 16.5 -> 16
        ctx.srs_config.rounding_mode = 0; // Floor
        ctx.accumulator.write(0, [264, 0, 0, 0, 0, 0, 0, 0]);
        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(0)[0], 16);

        // Switch to NegInf (mode 8): 16.5 at half -> 16 (toward -inf)
        ctx.srs_config.rounding_mode = 8; // NegInf
        ctx.accumulator.write(0, [264, 0, 0, 0, 0, 0, 0, 0]);
        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(0)[0], 16);
    }

    #[test]
    fn test_vector_srs_saturation_from_config() {
        let mut ctx = make_ctx();
        // Value that overflows i16: 32768 (> 32767)
        let overflow_val = 32768i64 as u64;
        ctx.accumulator.write(0, [overflow_val, 0, 0, 0, 0, 0, 0, 0]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int16)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(0)); // shift=0
        op.from_type = Some(ElementType::Int32);

        // Saturation enabled, signed -> clamp to 32767
        ctx.srs_config.saturation_mode = 1; // Saturate
        ctx.srs_config.srs_sign = true; // Signed
        VectorAlu::execute(&op, &mut ctx);
        let lo16 = ctx.vector.read(0)[0] as i16;
        assert_eq!(lo16, 32767);

        // Disable saturation: value wraps
        ctx.srs_config.saturation_mode = 0; // No saturation
        ctx.accumulator.write(0, [overflow_val, 0, 0, 0, 0, 0, 0, 0]);
        VectorAlu::execute(&op, &mut ctx);
        let lo16_nowrap = ctx.vector.read(0)[0] as i16;
        // Without saturation, 32768 truncated to 16 bits wraps to -32768
        assert_eq!(lo16_nowrap, -32768);
    }

    // wide_vector_push, extract_wide_element, insert_wide_element,
    // wide_vector_align tests: moved to vector_misc.rs

    #[test]
    fn test_vector_srs_from_type_masks_accumulator() {
        // When from_type is Int32, only the low 32 bits of each u64
        // accumulator lane should be used. Upper bits should be ignored.
        let mut ctx = make_ctx();
        // Set acc lane 0: upper 32 bits = garbage, lower 32 bits = 100.
        // With saturation enabled and signed 16-bit output, value 100
        // should survive. But without masking, the garbage upper bits
        // (0xDEAD_BEEF) make the i64 value huge and saturation clamps
        // to i16::MAX (32767) instead of 100.
        ctx.accumulator.write(0, [
            0xDEAD_BEEF_0000_0064, // upper garbage, lower = 100
            0, 0, 0, 0, 0, 0, 0,
        ]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int16) // 16-bit output
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(0)); // shift=0
        op.from_type = Some(ElementType::Int32); // 32-bit accumulator mode

        // Floor rounding, saturation ON, signed output.
        // Without masking: i64 value is huge -> saturates to 32767.
        // With masking: value is 100 -> passes through as 100.
        ctx.srs_config.rounding_mode = 0;
        ctx.srs_config.saturation_mode = 1; // Saturate
        ctx.srs_config.srs_sign = true;

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(0);
        let lo16 = result[0] as i16;
        assert_eq!(lo16, 100, "from_type=Int32 should mask to low 32 bits");
    }

    #[test]
    fn test_wide_ups_x2c() {
        // Wide UPS: x-register (512-bit) -> cm-register (1024-bit)
        let mut ctx = make_ctx();

        // Write 512-bit input to x4 (v4+v5): simple 32-bit values
        // Low half (v4): [1,2,3,4,5,6,7,8]
        ctx.vector.write(4, [1, 2, 3, 4, 5, 6, 7, 8]);
        // High half (v5): [9,10,11,12,13,14,15,16]
        ctx.vector.write(5, [9, 10, 11, 12, 13, 14, 15, 16]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Ups)
            .as_vector(ElementType::Int64) // output: 64-bit accumulator
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::VectorReg(4))
            .with_source(Operand::Immediate(0)); // shift=0
        op.from_type = Some(ElementType::Int32); // input: 32-bit vector
        op.is_wide_vector = true;

        VectorAlu::execute(&op, &mut ctx);

        // Check low accumulator (acc0): should have lanes 0-7 = [1,2,...,8]
        let acc_lo = ctx.accumulator.read(0);
        for i in 0..8 {
            assert_eq!(acc_lo[i], (i as u64 + 1), "acc_lo lane {i}");
        }
        // Check high accumulator (acc1): should have lanes 8-15 = [9,10,...,16]
        let acc_hi = ctx.accumulator.read(1);
        for i in 0..8 {
            assert_eq!(acc_hi[i], (i as u64 + 9), "acc_hi lane {i}");
        }
    }

    #[test]
    fn test_wide_srs_cm_to_x() {
        // Wide SRS: cm-register (1024-bit) -> x-register (512-bit)
        let mut ctx = make_ctx();

        // Write 1024-bit accumulator to cm0 (acc0+acc1)
        let acc_lo: [u64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let acc_hi: [u64; 8] = [90, 100, 110, 120, 130, 140, 150, 160];
        ctx.accumulator.write(0, acc_lo);
        ctx.accumulator.write(1, acc_hi);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(4))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::Immediate(0)); // shift=0
        op.from_type = Some(ElementType::Int64);
        op.is_wide_vector = true;

        // Floor rounding, no saturation, signed
        ctx.srs_config.rounding_mode = 0;
        ctx.srs_config.saturation_mode = 0;
        ctx.srs_config.srs_sign = true;

        VectorAlu::execute(&op, &mut ctx);

        // Check output: x4 = v4+v5
        let v4 = ctx.vector.read(4);
        assert_eq!(v4, [10, 20, 30, 40, 50, 60, 70, 80]);
        let v5 = ctx.vector.read(5);
        assert_eq!(v5, [90, 100, 110, 120, 130, 140, 150, 160]);
    }

    // test_wide_setge_scalar_dest_i32, test_wide_setlt_scalar_dest_i32: moved to vector_compare.rs

    /// SRS from Acc32 (16 lanes packed 2 per u64) to 16-bit output.
    /// Verifies that vector_srs_from_acc correctly unpacks lo32/hi32 from
    /// each u64 and produces 16 x 16-bit results packed into 8 u32 words.
    #[test]
    fn test_srs_from_acc32_to_d16() {
        // Build acc data: 16 lanes of acc32, values 100..115, packed 2 per u64.
        let mut acc = [0u64; 8];
        for i in 0..8usize {
            let lo = (100 + i * 2) as u64;
            let hi = (100 + i * 2 + 1) as u64;
            acc[i] = lo | (hi << 32);
        }

        // SRS with shift=0, from S32 accumulator to Int16 output.
        let cfg = SrsConfig {
            rounding_mode: 0, // Floor
            saturation_mode: 0, // No saturation
            srs_sign: true, // Signed output
        };

        let result = VectorAlu::vector_srs_from_acc(
            &acc, 0, ElementType::Int32, ElementType::Int16, &cfg,
        );

        // Each result word packs 2 x 16-bit values.
        for i in 0..8 {
            let expected_lo = (100 + i * 2) as u16;
            let expected_hi = (100 + i * 2 + 1) as u16;
            let expected = (expected_lo as u32) | ((expected_hi as u32) << 16);
            assert_eq!(result[i], expected, "result[{}]: got {:#010x}, expected {:#010x}",
                i, result[i], expected);
        }
    }

    /// SRS from Acc32 (16 lanes packed 2 per u64) to 8-bit output.
    #[test]
    fn test_srs_from_acc32_to_d8() {
        // Build acc data: 16 lanes of acc32, values 10..25, packed 2 per u64.
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

        // 16 x 8-bit values packed into 4 u32 words (4 bytes each).
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

    // NaN and FTZ tests (vadd_f_nan_canonical, vadd_f_denormal_ftz) moved to vector_arith.rs
}
