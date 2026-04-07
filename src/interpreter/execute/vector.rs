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
use crate::interpreter::state::{ExecutionContext, SrsConfig};
use crate::tablegen::SemanticOp;

use super::vector_dispatch::VectorAlu;
use super::vector_srs::{self, RoundingMode};
use super::vector_ups;
use super::vector_pack;

impl VectorAlu {
    /// Execute one 256-bit half of a vector operation.
    pub(super) fn execute_half(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        et: ElementType,
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

            // ========== Shuffle/Pack/Unpack ==========

            SemanticOp::Shuffle => {
                // VSHUFFLE is a 512-bit operation handled in execute_wide.
                // If we reach here via the narrow (256-bit) fallback path,
                // it means execute_wide didn't handle it -- should not happen
                // for real VSHUFFLE instructions. Return false to signal
                // unhandled (avoids panic from calling wide helpers in narrow context).
                false
            }

            SemanticOp::Pack => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_pack(&a, &b);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Unpack => {
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_unpack_low(&src);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            // Comparison ops (Cmp, SetGe, SetLt, SetEq, MaxLt, MinGe): extracted to vector_compare.rs

            // ========== SRS/UPS/Convert ==========

            SemanticOp::Srs => {
                // Shift-Round-Saturate: convert accumulator to vector
                let acc_reg = Self::get_acc_source(op);
                let shift = Self::get_shift_amount(op, ctx);
                let from = op.from_type.unwrap_or(ElementType::Int32);
                let result = Self::vector_srs(ctx, acc_reg, shift, from, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Convert => {
                // Type conversion (e.g., bf16 <-> f32).
                // Expansion (bf16->fp32) writes to accumulator;
                // contraction (fp32->bf16) writes to vector register.
                let src = Self::get_vector_source(op, ctx, 0);
                let from = op.from_type.unwrap_or(ElementType::Int32);
                let is_expansion = from.bits() < et.bits();
                if is_expansion && matches!(op.dest, Some(Operand::AccumReg(_))) {
                    // BF16->FP32 expansion: 8 bf16 -> 8 fp32 packed into accum.
                    let result = Self::vector_convert(&src, from, et);
                    let dst_reg = Self::get_acc_dest(op);
                    let mut acc = [0u64; 8];
                    for i in 0..8 {
                        acc[i] = result[i] as u64;
                    }
                    ctx.accumulator.write(dst_reg, acc);
                } else {
                    let result = Self::vector_convert(&src, from, et);
                    Self::write_vector_dest(op, ctx, result);
                }
                true
            }

            SemanticOp::Ups => {
                // Vector upshift: widen narrow vector lanes into accumulator.
                // VUPS destination is always an accumulator register.
                let src = Self::get_vector_source(op, ctx, 0);
                let shift = Self::get_shift_amount(op, ctx);
                let from = op.from_type.unwrap_or(ElementType::Int16);
                let acc_result = vector_ups::ups_vector_to_acc(&src, shift, from, et);
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

            // ========== Copy/Move ==========

            SemanticOp::Copy => {
                let src = Self::get_vector_source(op, ctx, 0);
                Self::write_vector_dest(op, ctx, src);
                true
            }

            // ========== Element Operations ==========

            SemanticOp::VectorExtract => {
                // Extract single element from vector to scalar
                let src = Self::get_vector_source(op, ctx, 0);
                let index = Self::get_lane_index(op, ctx);
                let result = Self::vector_extract(&src, index, et);
                Self::write_scalar_dest(op, ctx, result);
                true
            }

            SemanticOp::VectorInsert => {
                // VINSERT.N dst, s1, idx, s0: copy s1 with s1[idx] replaced by s0.
                // The base vector is s1 (sources[0]), NOT the current dst value.
                // VPUSH (shift+insert) is handled in the execute_wide path.
                //
                // Decoded sources: [s1 (VectorReg), idx (ScalarReg r29), s0 (ScalarReg)].
                // Index is always r29 (implicit), value is the second scalar source.
                let mut base = Self::get_vector_source(op, ctx, 0);
                let index = ctx.scalar.read(29);  // r29: implicit index register
                let value = Self::get_nth_scalar_source(op, ctx, 1);  // s0 (skip idx)
                Self::vector_insert(&mut base, value, index, et);
                Self::write_vector_dest(op, ctx, base);
                true
            }

            // VectorSelect: extracted to vector_compare.rs

            SemanticOp::VectorClear => {
                Self::write_vector_dest(op, ctx, [0u32; 8]);
                true
            }

            SemanticOp::VectorBroadcast => {
                // Three variants:
                // 1. VBCST: broadcast scalar to all lanes: vbcst.8 $dst, $scalar
                // 2. VBCST_64: broadcast 64-bit scalar pair to all lanes:
                //    vbcst.64 $dst, $s0  (s0 is decoded as a register pair)
                // 3. VEXTBCST: extract element from vector, then broadcast:
                //    vextbcst.16 $dst, $src_vec, $idx
                let has_vector_source = op.sources.iter().any(|s| matches!(s, Operand::VectorReg(_)));

                if matches!(et, ElementType::Int64 | ElementType::UInt64) && !has_vector_source {
                    // 64-bit broadcast: read register pair (rN, rN+1) as lo:hi.
                    let val64 = Self::get_scalar_source_64(op, ctx);
                    let lo = val64 as u32;
                    let hi = (val64 >> 32) as u32;
                    let mut result = [0u32; 8];
                    for i in 0..4 {
                        result[i * 2] = lo;
                        result[i * 2 + 1] = hi;
                    }
                    Self::write_vector_dest(op, ctx, result);
                } else {
                    let value = if has_vector_source {
                        // VEXTBCST: extract element at index from vector source
                        let src = Self::get_vector_source(op, ctx, 0);
                        let index = Self::get_lane_index(op, ctx);
                        Self::extract_element_by_index(&src, index, et)
                    } else {
                        // VBCST: broadcast scalar value
                        Self::get_scalar_source(op, ctx)
                    };

                    let result = Self::vector_broadcast(value, et);
                    Self::write_vector_dest(op, ctx, result);
                }
                true
            }

            // Shl, Srl, Sra: extracted to vector_arith.rs

            SemanticOp::Align => {
                // Concatenate two vectors and shift
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let shift = Self::get_lane_index(op, ctx);
                let result = Self::vector_align(&a, &b, shift);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            // AbsGtz, NegGtz, NegLtz, Accumulate/AccumSub/AccumNegAdd/AccumNegSub/NegAdd:
            // extracted to vector_arith.rs

            // ========== Bitwise Operations ==========

            SemanticOp::And => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_bitwise_and(&a, &b);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Or => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_bitwise_or(&a, &b);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Xor => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_bitwise_xor(&a, &b);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Not => {
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_bitwise_not(&src);
                Self::write_vector_dest(op, ctx, result);
                true
            }

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

    /// Dense matrix multiply: acc += A * B (operates on 4x4 or 4x8 tiles).
    ///
    /// For AIE2, this processes matrix tiles where:
    /// - A: activation matrix (from vector register)
    /// - B: weight matrix (from vector register)
    /// - Result accumulates to 512-bit accumulator
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
    pub(super) fn vector_srs_from_acc(
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
                    let out = vector_srs::srs_lane(
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
                        let out0 = vector_srs::srs_lane(
                            val0, shift, signed_output, 16,
                            saturate, sym_sat, mode,
                        );
                        let out1 = vector_srs::srs_lane(
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
                        let out_lo = vector_srs::srs_lane(
                            lo_val, shift, signed_output, 16,
                            saturate, sym_sat, mode,
                        );
                        let out_hi = vector_srs::srs_lane(
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
                        let out = vector_srs::srs_lane(
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

    /// Vector type conversion.
    ///
    /// Used by standalone `VCONV` and fused `vlda.conv` / `vst.conv`.
    pub(super) fn vector_convert(src: &[u32; 8], from_type: ElementType, to_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match (from_type, to_type) {
            // BFloat16 -> Float32 (expand: 16 bf16 -> 8 f32, use lower half)
            (ElementType::BFloat16, ElementType::Float32) => {
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..8 {
                    // Take lower bf16 from each pair
                    let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
                    // AIE2 FTZ: bf16 denormals (exp=0) flush to signed zero.
                    result[i] = fp32_flush_to_zero(Self::bf16_to_f32(bf16).to_bits());
                }
            }
            // Float32 -> BFloat16 (pack: 8 f32 -> 16 bf16, store in lower 4 words)
            (ElementType::Float32, ElementType::BFloat16) => {
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..4 {
                    // AIE2 FTZ on inputs before conversion.
                    let f0 = f32::from_bits(fp32_flush_to_zero(src[i * 2]));
                    let f1 = f32::from_bits(fp32_flush_to_zero(src[i * 2 + 1]));
                    let bf0 = Self::f32_to_bf16(f0);
                    let bf1 = Self::f32_to_bf16(f1);
                    result[i] = (bf0 as u32) | ((bf1 as u32) << 16);
                }
            }
            // Int32 -> Float32
            (ElementType::Int32, ElementType::Float32) => {
                for i in 0..8 {
                    result[i] = (src[i] as i32 as f32).to_bits();
                }
            }
            // UInt32 -> Float32
            (ElementType::UInt32, ElementType::Float32) => {
                for i in 0..8 {
                    result[i] = (src[i] as f32).to_bits();
                }
            }
            // Float32 -> Int32
            (ElementType::Float32, ElementType::Int32) => {
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..8 {
                    let f = f32::from_bits(fp32_flush_to_zero(src[i]));
                    result[i] = f as i32 as u32;
                }
            }
            // Float32 -> UInt32
            (ElementType::Float32, ElementType::UInt32) => {
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..8 {
                    let f = f32::from_bits(fp32_flush_to_zero(src[i]));
                    result[i] = f.max(0.0) as u32;
                }
            }
            // Int16 -> Int32 (expand lower half)
            (ElementType::Int16, ElementType::Int32) => {
                for i in 0..8 {
                    let i16_val = (src[i / 2] >> ((i % 2) * 16)) as i16;
                    result[i] = i16_val as i32 as u32;
                }
            }
            // Int32 -> BFloat16 (VFLOOR_S32_BF16): 8 i32 -> 16 bf16 (pack into 4 words)
            (ElementType::Int32, ElementType::BFloat16) => {
                for i in 0..4 {
                    let f0 = src[i * 2] as i32 as f32;
                    let f1 = src[i * 2 + 1] as i32 as f32;
                    let bf0 = Self::f32_to_bf16(f0);
                    let bf1 = Self::f32_to_bf16(f1);
                    result[i] = (bf0 as u32) | ((bf1 as u32) << 16);
                }
            }
            // BFloat16 -> Int32: 16 bf16 -> 8 i32 (use lower half)
            // VFLOOR uses floor rounding (toward negative infinity).
            (ElementType::BFloat16, ElementType::Int32) => {
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..8 {
                    let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
                    // AIE2 FTZ: bf16 denormals flush to signed zero before floor.
                    let f = f32::from_bits(fp32_flush_to_zero(Self::bf16_to_f32(bf16).to_bits()));
                    result[i] = f.floor() as i32 as u32;
                }
            }
            // Same type: pass through
            _ if from_type == to_type => {
                result = *src;
            }
            (from, to) => {
                panic!(
                    "vector_convert: unhandled conversion {:?} -> {:?} (encoding={:?})",
                    from, to, "vector_convert",
                );
            }
        }

        result
    }

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

            // ========== Bitwise ops (no ElementType parameter) ==========
            SemanticOp::And => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let b_lo: [u32; 8] = b[..8].try_into().unwrap();
                let b_hi: [u32; 8] = b[8..].try_into().unwrap();
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&Self::vector_bitwise_and(&a_lo, &b_lo));
                result[8..].copy_from_slice(&Self::vector_bitwise_and(&a_hi, &b_hi));
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Or => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let b_lo: [u32; 8] = b[..8].try_into().unwrap();
                let b_hi: [u32; 8] = b[8..].try_into().unwrap();
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&Self::vector_bitwise_or(&a_lo, &b_lo));
                result[8..].copy_from_slice(&Self::vector_bitwise_or(&a_hi, &b_hi));
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Xor => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let b_lo: [u32; 8] = b[..8].try_into().unwrap();
                let b_hi: [u32; 8] = b[8..].try_into().unwrap();
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&Self::vector_bitwise_xor(&a_lo, &b_lo));
                result[8..].copy_from_slice(&Self::vector_bitwise_xor(&a_hi, &b_hi));
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Not => {
                let a = Self::get_wide_vec_source(op, ctx, 0);
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&Self::vector_bitwise_not(&a_lo));
                result[8..].copy_from_slice(&Self::vector_bitwise_not(&a_hi));
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            // Comparison ops (Cmp, SetGe, SetLt): extracted to vector_compare.rs

            // ========== Matrix Multiply (config-driven) ==========
            SemanticOp::Mac | SemanticOp::MatMul | SemanticOp::MatMulSub
            | SemanticOp::NegMul | SemanticOp::NegMatMul
            | SemanticOp::AddMac | SemanticOp::SubMac => {
                return super::vector_matmul::execute_matmul(op, ctx);
            }

            // Accumulate/AccumSub/AccumNegAdd/AccumNegSub/NegAdd: extracted to vector_arith.rs

            // ========== SRS/UPS (element-wise, split into halves) ==========

            SemanticOp::Srs => {
                let acc_reg = Self::get_acc_source(op);
                let shift = Self::get_shift_amount(op, ctx);
                let from = op.from_type.unwrap_or(ElementType::Int64);
                // Half (bml/bmh) is 512-bit = single bm register, use narrow
                // read. Full (cm) and None (legacy/wide default) use read_wide.
                let is_half = matches!(op.accum_width,
                    Some(crate::tablegen::decoder_ffi::AccumWidth::Half));

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

            SemanticOp::Ups => {
                let shift = Self::get_shift_amount(op, ctx);
                let from = op.from_type.unwrap_or(ElementType::Int16);
                let is_half = matches!(op.accum_width,
                    Some(crate::tablegen::decoder_ffi::AccumWidth::Half));

                if !is_half {
                    let acc_wide = if !op.is_wide_vector {
                        // w2c path: narrow source (256-bit wl) fills full
                        // 1024-bit cm register via 4:1 upshift.
                        let src = Self::get_vector_source(op, ctx, 0);
                        vector_ups::ups_vector_to_acc_wide(&src, shift, from, et)
                    } else {
                        // x2c path: wide source (512-bit x-reg), UPS each half.
                        let src = Self::get_wide_vec_source(op, ctx, 0);
                        let src_lo: [u32; 8] = src[..8].try_into().unwrap();
                        let src_hi: [u32; 8] = src[8..].try_into().unwrap();
                        let acc_lo = vector_ups::ups_vector_to_acc(&src_lo, shift, from, et);
                        let acc_hi = vector_ups::ups_vector_to_acc(&src_hi, shift, from, et);
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
                    let acc_result = vector_ups::ups_vector_to_acc(&src, shift, from, et);
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

            // ========== Copy / Clear ==========
            SemanticOp::Copy => {
                // Handle both vector-to-vector and accum-to-accum moves.
                let has_acc_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::AccumReg(_)));
                if has_acc_source {
                    let src_reg = op.sources.iter().find_map(|s| match s {
                        Operand::AccumReg(r) => Some(*r),
                        _ => None,
                    }).unwrap_or(0);
                    let is_half = matches!(op.accum_width,
                        Some(crate::tablegen::decoder_ffi::AccumWidth::Half));
                    if !is_half {
                        // Accumulator move: vmov cm_dst, cm_src
                        let data = ctx.accumulator.read_wide(src_reg);
                        Self::write_wide_acc_dest(op, ctx, data);
                    } else {
                        // Half-accum move: vmov bm_dst, bm_src
                        let data = ctx.accumulator.read(src_reg);
                        let dst = Self::get_acc_dest(op);
                        ctx.accumulator.write(dst, data);
                    }
                } else {
                    // Vector move: vmov x_dst, x_src
                    let a = Self::get_wide_vec_source(op, ctx, 0);
                    Self::write_wide_vec_dest(op, ctx, a);
                }
                true
            }
            SemanticOp::VectorClear => {
                // Handle both vector and accumulator clears.
                let has_acc_dest = matches!(&op.dest, Some(Operand::AccumReg(_)));
                if has_acc_dest {
                    let is_half = matches!(op.accum_width,
                        Some(crate::tablegen::decoder_ffi::AccumWidth::Half));
                    if !is_half {
                        Self::write_wide_acc_dest(op, ctx, [0u64; 16]);
                    } else {
                        let dst = Self::get_acc_dest(op);
                        ctx.accumulator.clear(dst);
                    }
                } else {
                    Self::write_wide_vec_dest(op, ctx, [0u32; 16]);
                }
                true
            }

            // ========== Cross-half operations ==========

            SemanticOp::VectorExtract => {
                // VEXTRACT operates on a full 512-bit source.
                // Cannot use fallback (execute_half twice) because the second
                // call overwrites the scalar dest with a result from the wrong half.
                let src = Self::get_wide_vec_source(op, ctx, 0);
                let index = Self::get_lane_index(op, ctx);
                let value = Self::extract_wide_element(&src, index, et);
                if et.bits() >= 64 {
                    // 64-bit extract: write register pair (rN, rN+1)
                    Self::write_scalar_dest(op, ctx, value as u32);
                    if let Some(Operand::ScalarReg(r)) = &op.dest {
                        ctx.scalar.write(r + 1, (value >> 32) as u32);
                    }
                } else {
                    Self::write_scalar_dest(op, ctx, value as u32);
                }
                true
            }

            SemanticOp::VectorPush | SemanticOp::VectorPushHi => {
                let src = Self::get_wide_vec_source(op, ctx, 0);
                // For 64-bit VPUSH, read a register pair (rN, rN+1).
                let value = if et.bits() >= 64 {
                    Self::get_scalar_source_64(op, ctx)
                } else {
                    Self::get_scalar_source(op, ctx) as u64
                };
                let is_hi = matches!(op.semantic, Some(SemanticOp::VectorPushHi));
                let result = Self::wide_vector_push(&src, value, is_hi, et);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            SemanticOp::VectorInsert => {
                // VINSERT.N dst, s1, idx, s0: copy s1 with s1[idx] = s0.
                // Cannot use fallback (execute_half twice) because it
                // inserts at the same index in BOTH halves.
                //
                // Decoded sources: [s1 (VectorReg), idx (ScalarReg r29), s0 (ScalarReg)].
                let base = Self::get_wide_vec_source(op, ctx, 0);
                let index = ctx.scalar.read(29);  // r29: implicit index register
                if matches!(et, ElementType::Int64 | ElementType::UInt64) {
                    // 64-bit: s0 is a register pair (rN+1:rN). Read both halves.
                    // get_nth_scalar_source returns the pair's base register value;
                    // we need to find the actual register number to read rN+1.
                    let mut s0_reg = None;
                    let mut scalar_count = 0;
                    for src in &op.sources {
                        if let Operand::ScalarReg(r) = src {
                            if scalar_count == 1 { s0_reg = Some(*r); break; }
                            scalar_count += 1;
                        }
                    }
                    let reg = s0_reg.unwrap_or(0);
                    let lo = ctx.scalar.read(reg);
                    let hi = ctx.scalar.read(reg + 1);
                    let result = Self::insert_wide_element_64(&base, index, lo, hi);
                    Self::write_wide_vec_dest(op, ctx, result);
                } else {
                    let value = Self::get_nth_scalar_source(op, ctx, 1);  // s0 (skip idx)
                    let result = Self::insert_wide_element(&base, index, value, et);
                    Self::write_wide_vec_dest(op, ctx, result);
                }
                true
            }

            SemanticOp::VectorBroadcast => {
                let has_vector_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::VectorReg(_)));
                if has_vector_source {
                    // VEXTBCST: extract element from 512-bit source, then broadcast
                    let src = Self::get_wide_vec_source(op, ctx, 0);
                    let index = Self::get_lane_index(op, ctx);
                    let value = Self::extract_wide_element(&src, index, et);
                    let mut result = [0u32; 16];
                    if et.bits() >= 64 {
                        // 64-bit broadcast: replicate lo:hi pairs across 512 bits
                        let lo = value as u32;
                        let hi = (value >> 32) as u32;
                        for i in 0..8 {
                            result[i * 2] = lo;
                            result[i * 2 + 1] = hi;
                        }
                    } else {
                        let narrow_result = Self::vector_broadcast(value as u32, et);
                        result[..8].copy_from_slice(&narrow_result);
                        result[8..].copy_from_slice(&narrow_result);
                    }
                    Self::write_wide_vec_dest(op, ctx, result);
                } else {
                    // VBCST / VBCSTSHFL: broadcast scalar to 512-bit vector.
                    // VBCSTSHFL additionally applies a 16-bit matrix transpose.

                    // Step 1: broadcast
                    let mut result = if matches!(et, ElementType::Int64 | ElementType::UInt64) {
                        // 64-bit: read register pair for full 64-bit value
                        let val64 = Self::get_scalar_source_64(op, ctx);
                        let lo = val64 as u32;
                        let hi = (val64 >> 32) as u32;
                        let mut r = [0u32; 16];
                        for i in 0..8 {
                            r[i * 2] = lo;
                            r[i * 2 + 1] = hi;
                        }
                        r
                    } else {
                        let value = Self::get_scalar_source(op, ctx);
                        let narrow_result = Self::vector_broadcast(value, et);
                        let mut r = [0u32; 16];
                        r[..8].copy_from_slice(&narrow_result);
                        r[8..].copy_from_slice(&narrow_result);
                        r
                    };

                    // Step 2: VBCSTSHFL applies an implicit 16-bit matrix
                    // transpose based on element size (observed on NPU1,
                    // r29=0). The transpose groups all copies of each
                    // 16-bit sub-component together:
                    //   .8:  T32_2x16Lo(broadcast, zeros)
                    //   .16: identity (1 component, no rearrangement)
                    //   .32: 128-bit blocks (4 copies x 2 components)
                    //   .64: 256-bit blocks (4 copies x 4 components)
                    let is_shfl = op.encoding_name.as_deref()
                        .map_or(false, |n| n.contains("SHFL") || n.contains("shfl"));

                    if is_shfl {
                        // Per aietools ISG (me_inline_primitives.h):
                        // VBCSTSHFL = s2v_interleave_sw(broadcast, ZEROS, r29)
                        // i.e., shuffle_vectors(broadcast, zeros, mode=r29)
                        //
                        // mode_decode: mask = u48(1) << (r29 & 0x3F)
                        // When mode >= 48, the shift exceeds the 48-bit field width
                        // and the hardware produces mask = 0 (no routing bits set).
                        // With mask = 0, the crossbar passes byte 0 through and
                        // zeros the remaining 63 bytes.
                        let r29 = ctx.scalar.read(29);
                        let mode_idx = (r29 & 0x3F) as u8;

                        if let Some(mode) = super::vector_permute::ShuffleMode::from_mode(mode_idx) {
                            let mut lo_bytes = [0u8; 64];
                            let hi_bytes = [0u8; 64];
                            for i in 0..16 {
                                lo_bytes[i * 4..i * 4 + 4].copy_from_slice(&result[i].to_le_bytes());
                            }
                            let shuffled = super::vector_permute::shuffle_vectors(&lo_bytes, &hi_bytes, mode);
                            for i in 0..16 {
                                result[i] = u32::from_le_bytes([
                                    shuffled[i * 4], shuffled[i * 4 + 1],
                                    shuffled[i * 4 + 2], shuffled[i * 4 + 3],
                                ]);
                            }
                        } else {
                            // mode >= 48: mode_decode overflows the 48-bit mask,
                            // producing mask = 0. The crossbar with no routing
                            // bits passes only byte 0 of the lo input; everything
                            // else is zero. Convert the broadcast to [byte0, 0..].
                            let byte0 = result[0] as u8;
                            result = [0u32; 16];
                            result[0] = byte0 as u32;
                        }
                    }

                    Self::write_wide_vec_dest(op, ctx, result);
                }
                true
            }

            SemanticOp::Align => {
                // VSHIFT / VSHIFT_ALIGN barrel shifter.
                //
                // VSHIFT:       sources = [s1, s2, shift(Scalar)]
                //               step = 0 (no pre-shift stage)
                //
                // VSHIFT_ALIGN: sources = [s1, step(Scalar), s2, shift(Scalar)]
                //               step = s-register value (pre-shift selector)
                //
                // The hardware uses a mask-based merge + barrel shift +
                // optional pre-shift merge. See wide_vector_shift() docs.
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);

                let n_scalars = op.sources.iter().filter(|s| {
                    matches!(s, Operand::ScalarReg(_) | Operand::Immediate(_))
                }).count();

                let (step, shift) = if n_scalars >= 2 {
                    // VSHIFT_ALIGN: first scalar = step, second = shift
                    let step = Self::get_nth_scalar_source(op, ctx, 0);
                    let shift = Self::get_nth_scalar_source(op, ctx, 1);
                    (step, shift)
                } else {
                    // VSHIFT: single scalar = shift, step = 0
                    (0, Self::get_nth_scalar_source(op, ctx, 0))
                };

                let result = Self::wide_vector_shift(&a, &b, step, shift);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            SemanticOp::Shuffle => {
                // VSHUFFLE = s2v_interleave_sw(s1, s2, mode)
                // Two 512-bit vector inputs shuffled through the 48-mode
                // crossbar, with mode from the scalar `mod` operand.
                let (s1, s2) = Self::get_two_wide_vec_sources(op, ctx);
                let mode_val = Self::get_scalar_source(op, ctx);
                let mode_idx = (mode_val & 0x3F) as u8;

                let mut lo_bytes = [0u8; 64];
                let mut hi_bytes = [0u8; 64];
                for i in 0..16 {
                    lo_bytes[i * 4..i * 4 + 4].copy_from_slice(&s1[i].to_le_bytes());
                    hi_bytes[i * 4..i * 4 + 4].copy_from_slice(&s2[i].to_le_bytes());
                }

                let out_bytes = if let Some(mode) = super::vector_permute::ShuffleMode::from_mode(mode_idx) {
                    super::vector_permute::shuffle_vectors(&lo_bytes, &hi_bytes, mode)
                } else {
                    // mode >= 48: mask overflows to 0, crossbar passes
                    // only byte 0 of lo input, rest is zero.
                    let mut z = [0u8; 64];
                    z[0] = lo_bytes[0];
                    z
                };

                let mut result = [0u32; 16];
                for i in 0..16 {
                    result[i] = u32::from_le_bytes([
                        out_bytes[i * 4], out_bytes[i * 4 + 1],
                        out_bytes[i * 4 + 2], out_bytes[i * 4 + 3],
                    ]);
                }
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            // Shl, Srl, Sra, MaxLt, MinGe, SubLt, SubGe, MaxDiffLt,
            // NegGtz, NegLtz, AbsGtz: extracted to vector_arith.rs
            // SetEq: extracted to vector_compare.rs

            // ========== Convert (wide path) ==========

            SemanticOp::Convert => {
                // VCONV/VFLOOR: type conversion.
                // Three source types:
                // A. Accumulator source (VCONV_FP32_BF16, VCONV_BF16_FP32): read acc,
                //    extract lower 32 bits per lane as the input vector.
                // B. Narrow vector source (VFLOOR bf16->s32): 256-bit in, 512-bit out.
                // C. Wide vector source: 512-bit in, 512-bit out.
                //
                // VFLOOR has a shift operand (scalar register) that scales:
                //   result[i] = saturate_s32(floor(bf16_value * 2^shift))
                let from = op.from_type.unwrap_or(ElementType::Int32);

                // Extract shift from scalar source (VFLOOR only).
                let shift_val: Option<i32> = op.sources.iter().find_map(|s| {
                    if let Operand::ScalarReg(r) = s {
                        Some(ctx.scalar.read(*r) as i32)
                    } else {
                        None
                    }
                });
                let is_vfloor = from == ElementType::BFloat16
                    && et == ElementType::Int32
                    && shift_val.is_some();
                let has_acc_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::AccumReg(_)));

                if has_acc_source {
                    // Accumulator source: read acc data as raw bits, then convert.
                    //
                    // Accumulator access widths (from LLVM register class):
                    //   QuarterLow  (amll/amhl): 256 bits = lanes 0-3 of a bm register
                    //   QuarterHigh (amlh/amhh): 256 bits = lanes 4-7 of a bm register
                    //   Half        (bml/bmh):   512 bits = 8 lanes
                    //   Full        (cm):       1024 bits = 16 lanes (2 bm registers)
                    //
                    // The raw bits are reinterpreted according to `from` type -- e.g.,
                    // VFLOOR reads a quarter as 16 packed BF16 values, not as lane values.
                    let acc_reg = Self::get_acc_source(op);
                    let is_quarter = matches!(op.accum_width,
                        Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterLow) |
                        Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterHigh));
                    // Only Full (cm-class) is truly wide (1024-bit, 16 lanes).
                    // Half (bml/bmh) is 512-bit = 8 lanes. When accum_width is
                    // None, default to half -- the even-register heuristic was
                    // wrong because bml registers ARE even-numbered.
                    let is_wide = matches!(op.accum_width,
                        Some(crate::tablegen::decoder_ffi::AccumWidth::Full));

                    let (src_lo, src_hi) = if is_quarter {
                        // Quarter-accumulator: 256 bits = 4 u64 lanes.
                        // Repack as 8 u32 words (raw byte reinterpretation).
                        let acc = ctx.accumulator.read(acc_reg);
                        let lane_start = match op.accum_width {
                            Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterHigh) => 4,
                            _ => 0,
                        };
                        let mut words = [0u32; 8];
                        for i in 0..4 {
                            words[i * 2] = acc[lane_start + i] as u32;
                            words[i * 2 + 1] = (acc[lane_start + i] >> 32) as u32;
                        }
                        // Split into two halves for vector_convert (4 words each).
                        let mut lo = [0u32; 8];
                        let mut hi = [0u32; 8];
                        lo[..4].copy_from_slice(&words[..4]);
                        hi[..4].copy_from_slice(&words[4..]);
                        (lo, hi)
                    } else if is_wide {
                        let acc = ctx.accumulator.read_wide(acc_reg);
                        let mut lo = [0u32; 8];
                        let mut hi = [0u32; 8];
                        for i in 0..8 {
                            lo[i] = acc[i] as u32;
                            hi[i] = acc[i + 8] as u32;
                        }
                        (lo, hi)
                    } else {
                        // Half (bml/bmh): 8 u64 lanes.
                        // Acc32 mode packs two 32-bit values per lane (consecutive):
                        //   lane[i] = {val[2i+1], val[2i]}
                        // Extract all 16 values in consecutive order, split into two
                        // halves of 8 for vector_convert processing.
                        let acc = ctx.accumulator.read(acc_reg);
                        let mut all = [0u32; 16];
                        for i in 0..8 {
                            all[i * 2] = acc[i] as u32;         // val[2i]
                            all[i * 2 + 1] = (acc[i] >> 32) as u32; // val[2i+1]
                        }
                        let mut lo = [0u32; 8];
                        let mut hi = [0u32; 8];
                        lo.copy_from_slice(&all[..8]);
                        hi.copy_from_slice(&all[8..]);
                        (lo, hi)
                    };
                    let (res_lo, res_hi) = if is_vfloor {
                        let s = shift_val.unwrap();
                        (Self::vector_floor_bf16_to_s32(&src_lo, s),
                         Self::vector_floor_bf16_to_s32(&src_hi, s))
                    } else {
                        (Self::vector_convert(&src_lo, from, et),
                         Self::vector_convert(&src_hi, from, et))
                    };
                    // VCONV may produce narrow output (e.g., f32->bf16 packs 8 f32
                    // into 4 words of bf16). Write to the appropriate dest width.
                    if is_quarter && from.bits() < et.bits() {
                        // Quarter expansion (e.g., VFLOOR bf16->s32): 256 bits ->
                        // 512 bits. Both halves produce full 8-word results.
                        let mut result = [0u32; 16];
                        result[..8].copy_from_slice(&res_lo);
                        result[8..].copy_from_slice(&res_hi);
                        Self::write_wide_vec_dest(op, ctx, result);
                    } else if et.bits() < from.bits() {
                        // Contraction: 512-bit acc -> 256-bit vector.
                        // Each half contracts; pack results into one 256-bit output.
                        let words_per_half = (8 * et.bits() as usize) / (from.bits() as usize);
                        let mut result = [0u32; 8];
                        result[..words_per_half].copy_from_slice(&res_lo[..words_per_half]);
                        result[words_per_half..words_per_half * 2]
                            .copy_from_slice(&res_hi[..words_per_half]);
                        Self::write_vector_dest(op, ctx, result);
                    } else {
                        // Same-width or expansion from half/full acc.
                        let mut result = [0u32; 16];
                        result[..8].copy_from_slice(&res_lo);
                        result[8..].copy_from_slice(&res_hi);
                        Self::write_wide_vec_dest(op, ctx, result);
                    }
                } else {
                    let is_expansion = from.bits() < et.bits();
                    if is_expansion {
                        // Narrow source (256-bit w-register) -> wide dest (512-bit x/acc).
                        let src = Self::get_vector_source(op, ctx, 0);
                        let mut lo_in = [0u32; 8];
                        lo_in[..4].copy_from_slice(&src[..4]);
                        let mut hi_in = [0u32; 8];
                        hi_in[..4].copy_from_slice(&src[4..]);
                        let (res_lo, res_hi) = if is_vfloor {
                            let s = shift_val.unwrap();
                            (Self::vector_floor_bf16_to_s32(&lo_in, s),
                             Self::vector_floor_bf16_to_s32(&hi_in, s))
                        } else {
                            (Self::vector_convert(&lo_in, from, et),
                             Self::vector_convert(&hi_in, from, et))
                        };
                        if matches!(op.dest, Some(Operand::AccumReg(_))) {
                            // Accumulator dest: pack 16 x u32 into 8 x u64.
                            // Acc32 layout: consecutive pairs per lane:
                            //   lane[i] = {val[2i+1], val[2i]}
                            let dst_reg = Self::get_acc_dest(op);
                            let mut all = [0u32; 16];
                            all[..8].copy_from_slice(&res_lo);
                            all[8..].copy_from_slice(&res_hi);
                            let mut acc = [0u64; 8];
                            for i in 0..8 {
                                acc[i] = (all[i * 2] as u64) | ((all[i * 2 + 1] as u64) << 32);
                            }
                            ctx.accumulator.write(dst_reg, acc);
                        } else {
                            let mut result = [0u32; 16];
                            result[..8].copy_from_slice(&res_lo);
                            result[8..].copy_from_slice(&res_hi);
                            Self::write_wide_vec_dest(op, ctx, result);
                        }
                    } else {
                        // Same-width: 512-bit source, 512-bit dest.
                        let src = Self::get_wide_vec_source(op, ctx, 0);
                        let src_lo: [u32; 8] = src[..8].try_into().unwrap();
                        let src_hi: [u32; 8] = src[8..].try_into().unwrap();
                        let res_lo = Self::vector_convert(&src_lo, from, et);
                        let res_hi = Self::vector_convert(&src_hi, from, et);
                        let mut result = [0u32; 16];
                        result[..8].copy_from_slice(&res_lo);
                        result[8..].copy_from_slice(&res_hi);
                        Self::write_wide_vec_dest(op, ctx, result);
                    }
                }
                true
            }

            // ========== Pack / Unpack (asymmetric widths) ==========

            SemanticOp::Pack => {
                // VPACK: 512-bit x-reg source -> 256-bit w-reg dest.
                // Each 256-bit half is packed independently, then the two
                // packed halves are concatenated into one 256-bit result.
                let name = op.encoding_name.as_deref().unwrap_or("");
                let (bits_i, bits_o, signed) = vector_pack::pack_widths_from_name(name);
                let src = Self::get_wide_vec_source(op, ctx, 0);
                let src_lo: [u32; 8] = src[..8].try_into().unwrap();
                let src_hi: [u32; 8] = src[8..].try_into().unwrap();

                let packed_lo = vector_pack::pack_half(
                    &src_lo, bits_i, bits_o, signed, vector_pack::PackMode::Truncate,
                );
                let packed_hi = vector_pack::pack_half(
                    &src_hi, bits_i, bits_o, signed, vector_pack::PackMode::Truncate,
                );

                // Each half produces (256/bits_i * bits_o) bits of packed data.
                // Concatenate the two halves into a single 256-bit w-register.
                let words_per_half = ((256 / bits_i) * bits_o / 32) as usize;
                let mut result = [0u32; 8];
                result[..words_per_half].copy_from_slice(&packed_lo[..words_per_half]);
                result[words_per_half..words_per_half * 2]
                    .copy_from_slice(&packed_hi[..words_per_half]);

                // Write to 256-bit w-register dest (NOT write_wide_vec_dest).
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Unpack => {
                // VUNPACK: 256-bit w-reg source -> 512-bit x-reg dest.
                // The source lanes are split: lower lanes fill the low half
                // of the output, upper lanes fill the high half.
                //
                // The compiler emits vldb+vunpack without NOPs because
                // hardware scoreboarding stalls the unpack until the load
                // completes. Force-commit all pending vector writes so the
                // source data from a preceding vldb is visible.
                ctx.force_commit_all_pending();

                let name = op.encoding_name.as_deref().unwrap_or("");
                let (bits_i, bits_o, signed) = vector_pack::unpack_widths_from_name(name);

                // Read the 256-bit source (NOT wide -- it's a w-register).
                let src = Self::get_vector_source(op, ctx, 0);

                // Each output half holds 256/bits_o lanes. The second half
                // reads from lane_start = 256/bits_o in the source.
                let lanes_per_half = (256 / bits_o) as usize;
                let result_lo = vector_pack::unpack_half(&src, 0, bits_i, bits_o, signed);
                let result_hi = vector_pack::unpack_half(
                    &src, lanes_per_half, bits_i, bits_o, signed,
                );

                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&result_lo);
                result[8..].copy_from_slice(&result_hi);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            // ========== Fallback ==========
            _ => Self::execute_wide_fallback(op, ctx, semantic, et),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    // Arithmetic tests (add, sub, mul, min, max) moved to vector_arith.rs

    #[test]
    fn test_vector_shuffle_mode0() {
        // VSHUFFLE with mode=0: verify the crossbar is invoked and
        // produces a non-trivial permutation of the input data.
        let mut ctx = make_ctx();
        let mut lo = [0u32; 16];
        let mut hi = [0u32; 16];
        for i in 0..16 {
            lo[i] = (i as u32) + 1;
            hi[i] = (i as u32) + 0x100;
        }
        ctx.vector.write_wide(0, lo);
        ctx.vector.write_wide(2, hi);
        ctx.scalar.write(5, 0);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Shuffle)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(4))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(2))
            .with_source(Operand::ScalarReg(5));
        op.is_wide_vector = true;

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read_wide(4);
        // The crossbar should produce a permutation that differs from
        // both inputs -- not identity, not zeros.
        assert_ne!(result, lo, "result should not be identity of lo");
        assert_ne!(result, hi, "result should not be identity of hi");
        assert_ne!(result, [0u32; 16], "result should not be all zeros");
    }

    #[test]
    fn test_vector_shuffle_overflow_mode() {
        // mode >= 48: mask overflows, only byte 0 of lo passes through
        let mut ctx = make_ctx();
        ctx.vector.write_wide(0, [0xDEADBEEF, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        ctx.vector.write_wide(2, [0xFF; 16]);
        ctx.scalar.write(3, 50); // mode 50 >= 48

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Shuffle)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(4))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(2))
            .with_source(Operand::ScalarReg(3));
        op.is_wide_vector = true;

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read_wide(4);
        // Only byte 0 (0xEF) passes, everything else zero
        assert_eq!(result[0], 0xEF);
        for i in 1..16 {
            assert_eq!(result[i], 0, "lane {} should be zero", i);
        }
    }

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
        let acc = super::vector_ups::ups_vector_to_acc(
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
        let acc_wide = super::vector_ups::ups_vector_to_acc_wide(
            &src, 0, ElementType::Int8, ElementType::Int32,
        );

        // Verify UPS wrote non-zero data
        assert!(acc_wide.iter().any(|&v| v != 0), "UPS produced all zeros");

        // SRS each half (split at word 8 boundary, matching fused handler)
        let acc_lo: [u64; 8] = acc_wide[..8].try_into().unwrap();
        let acc_hi: [u64; 8] = acc_wide[8..].try_into().unwrap();

        let cfg = super::SrsConfig::default();
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
        let acc_shifted = super::vector_ups::ups_vector_to_acc_wide(
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

    #[test]
    fn test_vector_mov() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Copy)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [10, 20, 30, 40, 50, 60, 70, 80]);
    }

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

    // ---- wide_vector_push ------------------------------------------------

    /// vpush.lo: scalar inserted at index 0, existing elements shift up.
    /// Verifies data crossing the 256-bit (word index 8) boundary.
    #[test]
    fn test_wide_vector_push_lo_32() {
        let mut src = [0u32; 16];
        for i in 0..16 { src[i] = (i as u32 + 1) * 100; }
        let result = VectorAlu::wide_vector_push(&src, 0xDEAD_BEEF_u64, false, ElementType::Int32);
        assert_eq!(result[0],  0xDEAD_BEEF, "inserted value at lo end");
        assert_eq!(result[1],  100,         "former element 0 shifted to 1");
        assert_eq!(result[8],  800,         "former element 7 crossed 256-bit boundary");
        assert_eq!(result[15], 1500,        "former element 14 at high end");
    }

    /// vpush.hi: scalar inserted at the highest position, existing elements
    /// shift down.  Verifies element that was in the high half appears in lo.
    #[test]
    fn test_wide_vector_push_hi_32() {
        let mut src = [0u32; 16];
        for i in 0..16 { src[i] = (i as u32 + 1) * 100; }
        let result = VectorAlu::wide_vector_push(&src, 0xCAFE_BABE_u64, true, ElementType::Int32);
        assert_eq!(result[0],  200,         "former element 1 shifted to 0");
        assert_eq!(result[7],  900,         "former element 8 crossed boundary to 7");
        assert_eq!(result[14], 1600,        "former element 15 at second-to-last");
        assert_eq!(result[15], 0xCAFE_BABE, "inserted value at hi end");
    }

    /// vpush.lo.64: 64-bit scalar inserted at the low end, consuming 8 bytes.
    #[test]
    fn test_wide_vector_push_lo_64() {
        let mut src = [0u32; 16];
        for i in 0..16 { src[i] = (i as u32 + 1) * 100; }
        let result = VectorAlu::wide_vector_push(&src, 0x1234_5678_ABCD_EF00_u64, false, ElementType::Int64);
        // Low 32 bits at word 0, high 32 bits at word 1.
        assert_eq!(result[0], 0xABCD_EF00, "64-bit value low word");
        assert_eq!(result[1], 0x1234_5678, "64-bit value high word");
        // Former element 0 (words 0-1) shifted to words 2-3.
        assert_eq!(result[2], 100, "former word 0 shifted to word 2");
        assert_eq!(result[3], 200, "former word 1 shifted to word 3");
    }

    /// vpush.hi.64: 64-bit scalar inserted at the high end, consuming 8 bytes.
    #[test]
    fn test_wide_vector_push_hi_64() {
        let mut src = [0u32; 16];
        for i in 0..16 { src[i] = (i as u32 + 1) * 100; }
        let result = VectorAlu::wide_vector_push(&src, 0xDEAD_BEEF_CAFE_0000_u64, true, ElementType::Int64);
        // High 8 bytes = inserted value.
        assert_eq!(result[14], 0xCAFE_0000, "64-bit value low word at hi end");
        assert_eq!(result[15], 0xDEAD_BEEF, "64-bit value high word at hi end");
        // Former words shifted down by 2.
        assert_eq!(result[0], 300, "former word 2 shifted to word 0");
        assert_eq!(result[1], 400, "former word 3 shifted to word 1");
    }

    // ---- extract_wide_element --------------------------------------------

    /// Element 8 of i32 lives in word index 8 (the high 256-bit half).
    #[test]
    fn test_extract_wide_element_high_half() {
        let mut src = [0u32; 16];
        src[8] = 0x4242_4242;
        let val = VectorAlu::extract_wide_element(&src, 8, ElementType::Int32);
        assert_eq!(val, 0x4242_4242);
    }

    /// Element 17 of i16: bit offset = 17*16 = 272.  Word index = 272/32 = 8,
    /// bit-in-word = 272%32 = 16, so it is the high 16 bits of word 8.
    #[test]
    fn test_extract_wide_element_16bit() {
        let mut src = [0u32; 16];
        src[8] = 0xBEEF_DEAD; // lo16 = 0xDEAD, hi16 = 0xBEEF
        // UInt16: no sign extension, raw 16-bit value
        let val = VectorAlu::extract_wide_element(&src, 17, ElementType::UInt16);
        assert_eq!(val, 0xBEEF);
        // Int16: sign-extended (0xBEEF is negative as i16)
        let val_signed = VectorAlu::extract_wide_element(&src, 17, ElementType::Int16);
        assert_eq!(val_signed, 0xBEEF_u16 as i16 as i32 as u32 as u64);
    }

    // ---- insert_wide_element ------------------------------------------------

    /// Insert i32 at element 8 (high half, word index 8).
    #[test]
    fn test_insert_wide_element_high_half() {
        let src = [0u32; 16];
        let result = VectorAlu::insert_wide_element(&src, 8, 0xDEAD_BEEF, ElementType::Int32);
        assert_eq!(result[8], 0xDEAD_BEEF);
        // Other words remain zero.
        assert_eq!(result[0], 0);
        assert_eq!(result[7], 0);
        assert_eq!(result[9], 0);
    }

    /// Insert i16 at element 17 (high 16 bits of word 8).
    #[test]
    fn test_insert_wide_element_16bit() {
        let mut src = [0u32; 16];
        src[8] = 0x0000_DEAD; // low 16 bits should be preserved
        let result = VectorAlu::insert_wide_element(&src, 17, 0xBEEF, ElementType::Int16);
        assert_eq!(result[8], 0xBEEF_DEAD); // hi16=BEEF, lo16=DEAD (preserved)
    }

    /// Round-trip: insert then extract should return the inserted value.
    #[test]
    fn test_insert_extract_roundtrip() {
        let src = [0xFFFF_FFFFu32; 16];
        let after = VectorAlu::insert_wide_element(&src, 5, 0x42, ElementType::Int32);
        let val = VectorAlu::extract_wide_element(&after, 5, ElementType::Int32);
        assert_eq!(val, 0x42);
        // Neighbor should be untouched.
        let neighbor = VectorAlu::extract_wide_element(&after, 4, ElementType::Int32);
        assert_eq!(neighbor, 0xFFFF_FFFF);
    }

    // ---- wide_vector_align -----------------------------------------------

    /// Zero shift returns src1 unchanged.
    #[test]
    fn test_wide_vector_align_no_shift() {
        let src1 = [1u32; 16];
        let src2 = [2u32; 16];
        let result = VectorAlu::wide_vector_shift(&src1, &src2, 0, 0);
        assert_eq!(result, [1u32; 16]);
    }

    /// Shift by exactly 64 bytes skips all of src1 and returns src2 unchanged.
    #[test]
    fn test_wide_vector_align_full_shift() {
        let src1 = [1u32; 16];
        let src2 = [2u32; 16];
        let result = VectorAlu::wide_vector_shift(&src1, &src2, 0, 64);
        assert_eq!(result, [2u32; 16]);
    }

    /// Shift by 60 bytes: result[0] = last word of src1, result[1] = first
    /// word of src2.  This exercises the cross-boundary stitch path.
    #[test]
    fn test_wide_vector_align_cross_boundary() {
        let mut src1 = [0u32; 16];
        let mut src2 = [0u32; 16];
        src1[15] = 0xAAAA_AAAA; // last word of src1
        src2[0]  = 0xBBBB_BBBB; // first word of src2
        let result = VectorAlu::wide_vector_shift(&src1, &src2, 0, 60);
        assert_eq!(result[0], 0xAAAA_AAAA, "last word of src1 at result[0]");
        assert_eq!(result[1], 0xBBBB_BBBB, "first word of src2 at result[1]");
    }

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
