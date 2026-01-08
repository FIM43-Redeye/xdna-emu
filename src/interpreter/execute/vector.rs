//! Vector ALU execution unit (LEGACY FALLBACK).
//!
//! Handles SIMD operations on 256-bit vectors (8 × 32-bit, 16 × 16-bit, 32 × 8-bit).
//!
//! # Architecture Note
//!
//! Like [`ScalarAlu`](super::ScalarAlu), this module is a **legacy fallback**.
//! The preferred execution path is through the TableGen-driven semantic
//! dispatcher in [`execute_semantic`](super::semantic::execute_semantic).
//!
//! ## Execution Flow
//!
//! ```text
//! CycleAccurateExecutor::execute_slot()
//!         |
//!         v
//!   execute_semantic(op, ctx)  <-- TableGen-driven, preferred
//!         |
//!         | returns false (no semantic info or vector type not handled)
//!         v
//!   ScalarAlu::execute(op, ctx)
//!         |
//!         | returns false (not a scalar op)
//!         v
//!   VectorAlu::execute(op, ctx)  <-- Legacy fallback (this module)
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

use crate::interpreter::bundle::{ElementType, Operation, Operand, ShufflePattern, SlotOp};
use crate::interpreter::state::ExecutionContext;

/// Vector ALU execution unit.
pub struct VectorAlu;

impl VectorAlu {
    /// Execute a vector operation.
    ///
    /// Returns `true` if the operation was handled, `false` if not a vector op.
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
        log::trace!("[VECTOR_ALU] Checking op={:?} dest={:?}", op.op, op.dest);
        match &op.op {
            Operation::VectorAdd { element_type } => {
                log::debug!("[VECTOR_ALU] Executing VectorAdd element_type={:?} dest={:?} sources={:?}",
                    element_type, op.dest, op.sources);
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_add(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorSub { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_sub(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMul { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_mul(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMac { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_mac(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            Operation::VectorMin { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_min(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMax { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_max(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorShuffle { pattern } => {
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_shuffle(&src, *pattern);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorPack => {
                // Pack two vectors into one (narrow)
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_pack(&a, &b);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorUnpack => {
                // Unpack one vector into two (widen) - writes low half
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_unpack_low(&src);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorCmp { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_cmp_eq(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMatMulDense { element_type } => {
                // Dense matrix multiply: acc += A * B
                // AIE2 processes 4x4 or 4x8 matrix tiles
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_matmul_dense(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            Operation::VectorMatMulSparse { element_type, .. } => {
                // Sparse matrix multiply: similar to dense but with sparse format
                // For now, treat as dense - sparse format optimization is a refinement
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_matmul_dense(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            // ========== Convolution-Related Operations (VMSC/VNEGMAC variants) ==========

            Operation::VectorMatMulSubDense { element_type } => {
                // Matrix multiply-subtract: acc -= A * B
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_matmul_sub(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            Operation::VectorMatMulSubSparse { element_type, .. } => {
                // Sparse matrix multiply-subtract (treat as dense for now)
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_matmul_sub(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            Operation::VectorNegMatMulDense { element_type } => {
                // Negated matrix multiply: acc += -(A * B)
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_neg_matmul(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            Operation::VectorNegMatMulSubDense { element_type } => {
                // Negated matrix multiply-subtract: acc -= -(A * B) = acc + (A * B)
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                // Note: acc -= -(A*B) is equivalent to acc += (A*B), same as dense
                Self::vector_matmul_dense(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            Operation::VectorMatMulAccFloat { .. } => {
                // BFloat16 matrix multiply-accumulate for CNN workloads
                // Operands are bf16, accumulator is fp32
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_matmul_bf16(ctx, acc_reg, &a, &b, true);
                true
            }

            Operation::VectorMatMulSubFloat { .. } => {
                // BFloat16 matrix multiply-subtract for CNN workloads
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_matmul_bf16(ctx, acc_reg, &a, &b, false);
                true
            }

            Operation::VectorAddMac { element_type } => {
                // Double accumulator: acc1 = acc1 + acc2 + A * B
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                let acc2_reg = Self::get_acc_source(op);
                Self::vector_double_acc_mac(ctx, acc_reg, acc2_reg, &a, &b, *element_type, true);
                true
            }

            Operation::VectorSubMac { element_type } => {
                // Double accumulator: acc1 = acc1 - acc2 + A * B
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                let acc2_reg = Self::get_acc_source(op);
                Self::vector_double_acc_mac(ctx, acc_reg, acc2_reg, &a, &b, *element_type, false);
                true
            }

            Operation::VectorSRS { from_type, to_type } => {
                // Shift-Round-Saturate: convert accumulator to vector
                let acc_reg = Self::get_acc_source(op);
                let shift = Self::get_shift_amount(op, ctx);
                let result = Self::vector_srs(ctx, acc_reg, shift, *from_type, *to_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorConvert { from_type, to_type } => {
                // Type conversion (e.g., bf16 <-> f32)
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_convert(&src, *from_type, *to_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMov { .. } => {
                // Vector register move
                let src = Self::get_vector_source(op, ctx, 0);
                Self::write_vector_dest(op, ctx, src);
                true
            }

            // ========== Vector Element Operations ==========

            Operation::VectorExtract { element_type } => {
                // Extract single element from vector to scalar
                // dst_scalar = src_vector[index]
                let src = Self::get_vector_source(op, ctx, 0);
                let index = Self::get_lane_index(op, ctx);
                let result = Self::vector_extract(&src, index, *element_type);
                Self::write_scalar_dest(op, ctx, result);
                true
            }

            Operation::VectorInsert { element_type } => {
                // Insert scalar into vector lane
                // dst_vector[index] = src_scalar; other lanes from src_vector
                let mut dst = Self::get_vector_dest_value(op, ctx);
                let value = Self::get_scalar_source(op, ctx);
                let index = Self::get_lane_index(op, ctx);
                Self::vector_insert(&mut dst, value, index, *element_type);
                Self::write_vector_dest(op, ctx, dst);
                true
            }

            Operation::VectorSelect { element_type } => {
                // Per-lane conditional select: dst[i] = sel[i] ? a[i] : b[i]
                // vector_select takes (mask, src_if_true, src_if_false)
                // Source order: mask, true_val, false_val
                let sel = Self::get_vector_source(op, ctx, 0);
                let a = Self::get_vector_source(op, ctx, 1);
                let b = Self::get_vector_source(op, ctx, 2);
                let result = Self::vector_select(&sel, &a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorClear => {
                // Clear vector register to zero
                Self::write_vector_dest(op, ctx, [0u32; 8]);
                true
            }

            Operation::VectorBroadcast { element_type } => {
                // Broadcast scalar value to all vector lanes
                let value = Self::get_scalar_source(op, ctx);
                let result = Self::vector_broadcast(value, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            // ========== Vector Shift Operations ==========

            Operation::VectorShiftLeft { element_type } => {
                // Vector logical left shift: dst[i] = a[i] << shift[i]
                let (a, shift) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_shift_left(&a, &shift, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorShiftRight { element_type } => {
                // Vector logical right shift: dst[i] = a[i] >> shift[i]
                let (a, shift) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_shift_right_logical(&a, &shift, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorArithShiftRight { element_type } => {
                // Vector arithmetic right shift: dst[i] = (signed)a[i] >> shift[i]
                let (a, shift) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_shift_right_arith(&a, &shift, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorAlign { .. } => {
                // Vector align: concatenate two vectors and shift
                // Result = (src1 || src2) >> (shift * element_size)
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let shift = Self::get_lane_index(op, ctx);
                let result = Self::vector_align(&a, &b, shift);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorUpshift { from_type, to_type } => {
                // Vector upshift: shift left with saturation for precision scaling
                let src = Self::get_vector_source(op, ctx, 0);
                let shift = Self::get_lane_index(op, ctx);
                let result = Self::vector_upshift(&src, shift, *from_type, *to_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            // ========== Conditional Vector Operations ==========

            Operation::VectorAbsGtz { element_type } => {
                // Absolute value if greater than zero: dst[i] = (src[i] > 0) ? abs(src[i]) : src[i]
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_abs_gtz(&src, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorNegGtz { element_type } => {
                // Negate if greater than zero: dst[i] = (src[i] > 0) ? -src[i] : src[i]
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_neg_gtz(&src, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorNegLtz { element_type } => {
                // Negate if less than zero: dst[i] = (src[i] < 0) ? -src[i] : src[i]
                // This is essentially abs()
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_neg_ltz(&src, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorAccumulate { element_type } => {
                // Vector accumulate: acc += src (add to accumulator without multiply)
                let src = Self::get_vector_source(op, ctx, 0);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_accumulate(ctx, acc_reg, &src, *element_type);
                true
            }

            Operation::VectorNegate { element_type } => {
                // Vector negate: dst = -src
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_negate(&src, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorNegAdd { element_type } => {
                // Vector negate and add: dst = -src1 + src2
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_neg_add(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorNegMul { element_type } => {
                // Vector negate multiply: acc += -(src1 * src2)
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_neg_mul(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            // ========== Vector Comparison Operations ==========

            Operation::VectorGe { element_type } => {
                // Greater than or equal: dst[i] = (a[i] >= b[i]) ? ~0 : 0
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_compare_ge(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorLt { element_type } => {
                // Less than: dst[i] = (a[i] < b[i]) ? ~0 : 0
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_compare_lt(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorEqz { element_type } => {
                // Equal to zero: dst[i] = (a[i] == 0) ? ~0 : 0
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_compare_eqz(&src, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMaxLt { element_type } => {
                // Maximum with less-than flag: dst = max(a, b), sets flags
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_max(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMinGe { element_type } => {
                // Minimum with greater-equal flag: dst = min(a, b), sets flags
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_min(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            // ========== Vector Bitwise Operations ==========

            Operation::VectorAnd { element_type: _ } => {
                // Bitwise AND: dst = a & b
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_bitwise_and(&a, &b);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorOr { element_type: _ } => {
                // Bitwise OR: dst = a | b
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_bitwise_or(&a, &b);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorXor { element_type: _ } => {
                // Bitwise XOR: dst = a ^ b
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_bitwise_xor(&a, &b);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorNot { element_type: _ } => {
                // Bitwise NOT: dst = ~a
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_bitwise_not(&src);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            // ========== Vector Conditional Arithmetic Operations ==========

            Operation::VectorSubLt { element_type } => {
                // Subtract if less-than: dst[i] = (a[i] < b[i]) ? a[i] - b[i] : a[i]
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_sub_lt(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorSubGe { element_type } => {
                // Subtract if greater-equal: dst[i] = (a[i] >= b[i]) ? a[i] - b[i] : a[i]
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_sub_ge(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMaxDiffLt { element_type } => {
                // Maximum difference if less-than: dst[i] = max(a[i] - b[i], 0) if a < b
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_maxdiff_lt(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            _ => false, // Not a vector operation
        }
    }

    /// Get two vector source operands.
    fn get_two_vector_sources(op: &SlotOp, ctx: &ExecutionContext) -> ([u32; 8], [u32; 8]) {
        let a = Self::get_vector_source(op, ctx, 0);
        let b = Self::get_vector_source(op, ctx, 1);
        (a, b)
    }

    /// Get a single vector source operand.
    fn get_vector_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> [u32; 8] {
        op.sources
            .get(idx)
            .map_or([0; 8], |src| Self::read_vector_operand(src, ctx))
    }

    /// Read a vector operand value.
    fn read_vector_operand(operand: &Operand, ctx: &ExecutionContext) -> [u32; 8] {
        match operand {
            Operand::VectorReg(r) => ctx.vector.read(*r),
            _ => [0; 8],
        }
    }

    /// Write result to vector destination.
    fn write_vector_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: [u32; 8]) {
        if let Some(Operand::VectorReg(r)) = &op.dest {
            ctx.vector.write(*r, value);
        }
    }

    /// Get accumulator destination register.
    fn get_acc_dest(op: &SlotOp) -> u8 {
        match &op.dest {
            Some(Operand::AccumReg(r)) => *r,
            _ => 0,
        }
    }

    /// Convert BFloat16 (upper 16 bits of f32) to f32.
    #[inline]
    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// Convert f32 to BFloat16 (truncate lower 16 bits).
    #[inline]
    fn f32_to_bf16(val: f32) -> u16 {
        (val.to_bits() >> 16) as u16
    }

    /// Get accumulator source register from operands.
    fn get_acc_source(op: &SlotOp) -> u8 {
        for src in &op.sources {
            if let Operand::AccumReg(r) = src {
                return *r;
            }
        }
        0
    }

    /// Get shift amount from operands (immediate or register).
    fn get_shift_amount(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // Look for immediate in sources
        for src in &op.sources {
            if let Operand::Immediate(imm) = src {
                return *imm as u32;
            }
            if let Operand::ScalarReg(r) = src {
                return ctx.scalar.read(*r);
            }
        }
        0 // Default: no shift
    }

    /// Dense matrix multiply: acc += A * B (operates on 4x4 or 4x8 tiles).
    ///
    /// For AIE2, this processes matrix tiles where:
    /// - A: activation matrix (from vector register)
    /// - B: weight matrix (from vector register)
    /// - Result accumulates to 512-bit accumulator
    fn vector_matmul_dense(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        a: &[u32; 8],
        b: &[u32; 8],
        elem_type: ElementType,
    ) {
        let current = ctx.accumulator.read(acc_reg);

        match elem_type {
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 x int8 elements: interpret as 4x8 * 8x4 = 4x4 output
                // Each u32 contains 4 int8 values
                // Simplified: element-wise multiply-accumulate for compatibility
                let mut new_acc = current;
                for i in 0..8 {
                    let mut acc = current[i];
                    for byte_idx in 0..4 {
                        let a_byte = ((a[i] >> (byte_idx * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (byte_idx * 8)) & 0xFF) as u8;
                        let prod = if matches!(elem_type, ElementType::Int8) {
                            (a_byte as i8 as i64) * (b_byte as i8 as i64)
                        } else {
                            (a_byte as i64) * (b_byte as i64)
                        };
                        acc = acc.wrapping_add(prod as u64);
                    }
                    new_acc[i] = acc;
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 16 x int16 elements: interpret as matrix multiply
                let mut new_acc = current;
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16 as i32;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16 as i32;
                    let b_lo = (b[i] & 0xFFFF) as i16 as i32;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16 as i32;
                    // Accumulate products
                    new_acc[i] = current[i]
                        .wrapping_add((a_lo * b_lo) as u64)
                        .wrapping_add((a_hi * b_hi) as u64);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::BFloat16 => {
                // 16 x bf16 matrix multiply
                let mut new_acc = current;
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16) as f64;
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16) as f64;
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16) as f64;
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + a_lo * b_lo + a_hi * b_hi).to_bits();
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Int32 | ElementType::UInt32 => {
                // 8 x int32 matrix multiply-accumulate
                let mut new_acc = current;
                for i in 0..8 {
                    let prod = (a[i] as u64) * (b[i] as u64);
                    new_acc[i] = current[i].wrapping_add(prod);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Float32 => {
                // 8 x f32 matrix multiply
                let mut new_acc = current;
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]) as f64;
                    let fb = f32::from_bits(b[i]) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + fa * fb).to_bits();
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
        }
    }

    /// Matrix multiply-subtract (VMSC): acc -= A * B
    ///
    /// Used for convolution operations where subtraction is needed.
    fn vector_matmul_sub(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        a: &[u32; 8],
        b: &[u32; 8],
        elem_type: ElementType,
    ) {
        let current = ctx.accumulator.read(acc_reg);

        match elem_type {
            ElementType::Int8 | ElementType::UInt8 => {
                let mut new_acc = current;
                for i in 0..8 {
                    let mut acc = current[i];
                    for byte_idx in 0..4 {
                        let a_byte = ((a[i] >> (byte_idx * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (byte_idx * 8)) & 0xFF) as u8;
                        let prod = if matches!(elem_type, ElementType::Int8) {
                            (a_byte as i8 as i64) * (b_byte as i8 as i64)
                        } else {
                            (a_byte as i64) * (b_byte as i64)
                        };
                        acc = acc.wrapping_sub(prod as u64);
                    }
                    new_acc[i] = acc;
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Int16 | ElementType::UInt16 => {
                let mut new_acc = current;
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16 as i32;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16 as i32;
                    let b_lo = (b[i] & 0xFFFF) as i16 as i32;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16 as i32;
                    new_acc[i] = current[i]
                        .wrapping_sub((a_lo * b_lo) as u64)
                        .wrapping_sub((a_hi * b_hi) as u64);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::BFloat16 => {
                let mut new_acc = current;
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16) as f64;
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16) as f64;
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16) as f64;
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f - a_lo * b_lo - a_hi * b_hi).to_bits();
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Int32 | ElementType::UInt32 => {
                let mut new_acc = current;
                for i in 0..8 {
                    let prod = (a[i] as u64) * (b[i] as u64);
                    new_acc[i] = current[i].wrapping_sub(prod);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Float32 => {
                let mut new_acc = current;
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]) as f64;
                    let fb = f32::from_bits(b[i]) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f - fa * fb).to_bits();
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
        }
    }

    /// Negated matrix multiply (VNEGMAC): acc += -(A * B)
    ///
    /// Adds the negated product to the accumulator.
    fn vector_neg_matmul(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        a: &[u32; 8],
        b: &[u32; 8],
        elem_type: ElementType,
    ) {
        // VNEGMAC: acc += -(A * B) is the same as acc -= A * B
        Self::vector_matmul_sub(ctx, acc_reg, a, b, elem_type);
    }

    /// BFloat16 matrix multiply-accumulate for CNN workloads (VMAC.f/VMSC.f)
    ///
    /// Operands are BFloat16, accumulator is Float32 for higher precision.
    /// This is the key instruction for neural network inference.
    fn vector_matmul_bf16(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        a: &[u32; 8],
        b: &[u32; 8],
        accumulate: bool, // true for VMAC.f, false for VMSC.f
    ) {
        let current = ctx.accumulator.read(acc_reg);
        let mut new_acc = current;

        // Each u32 contains 2 bf16 values (16 bf16 total in 256-bit vector)
        // Accumulator holds 8 x f32 values (as f64 for precision)
        for i in 0..8 {
            let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16) as f64;
            let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16) as f64;
            let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16) as f64;
            let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16) as f64;
            let current_f = f64::from_bits(current[i]);

            let product_sum = a_lo * b_lo + a_hi * b_hi;
            if accumulate {
                new_acc[i] = (current_f + product_sum).to_bits();
            } else {
                new_acc[i] = (current_f - product_sum).to_bits();
            }
        }

        ctx.accumulator.write(acc_reg, new_acc);
    }

    /// Double accumulator MAC (VADDMAC/VSUBMAC): acc1 = acc1 +/- acc2 + A * B
    ///
    /// These fused operations combine accumulator arithmetic with matrix multiply.
    fn vector_double_acc_mac(
        ctx: &mut ExecutionContext,
        acc1_reg: u8,
        acc2_reg: u8,
        a: &[u32; 8],
        b: &[u32; 8],
        elem_type: ElementType,
        add_acc2: bool, // true for VADDMAC, false for VSUBMAC
    ) {
        let acc1 = ctx.accumulator.read(acc1_reg);
        let acc2 = ctx.accumulator.read(acc2_reg);

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
                let mut new_acc = [0u64; 8];
                for i in 0..8 {
                    let prod = (a[i] as u64) * (b[i] as u64);
                    let acc2_contrib = if add_acc2 {
                        acc2[i]
                    } else {
                        0u64.wrapping_sub(acc2[i])
                    };
                    new_acc[i] = acc1[i].wrapping_add(acc2_contrib).wrapping_add(prod);
                }
                ctx.accumulator.write(acc1_reg, new_acc);
            }
            ElementType::BFloat16 | ElementType::Float32 => {
                let mut new_acc = [0u64; 8];
                for i in 0..8 {
                    let (a_lo, a_hi, b_lo, b_hi) = if matches!(elem_type, ElementType::BFloat16) {
                        (
                            Self::bf16_to_f32((a[i] & 0xFFFF) as u16) as f64,
                            Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16) as f64,
                            Self::bf16_to_f32((b[i] & 0xFFFF) as u16) as f64,
                            Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16) as f64,
                        )
                    } else {
                        (
                            f32::from_bits(a[i]) as f64,
                            0.0,
                            f32::from_bits(b[i]) as f64,
                            0.0,
                        )
                    };

                    let acc1_f = f64::from_bits(acc1[i]);
                    let acc2_f = f64::from_bits(acc2[i]);
                    let product = a_lo * b_lo + a_hi * b_hi;

                    let result = if add_acc2 {
                        acc1_f + acc2_f + product
                    } else {
                        acc1_f - acc2_f + product
                    };
                    new_acc[i] = result.to_bits();
                }
                ctx.accumulator.write(acc1_reg, new_acc);
            }
            _ => {
                // For other types, fall back to simple accumulate
                let mut new_acc = acc1;
                for i in 0..8 {
                    let acc2_contrib = if add_acc2 {
                        acc2[i]
                    } else {
                        0u64.wrapping_sub(acc2[i])
                    };
                    new_acc[i] = acc1[i].wrapping_add(acc2_contrib);
                }
                ctx.accumulator.write(acc1_reg, new_acc);
            }
        }
    }

    /// Shift-Round-Saturate: convert 64-bit accumulator lanes to narrower output.
    ///
    /// This is used after MAC operations to normalize results back to vector format.
    fn vector_srs(
        ctx: &ExecutionContext,
        acc_reg: u8,
        shift: u32,
        _from_type: ElementType,
        to_type: ElementType,
    ) -> [u32; 8] {
        let acc = ctx.accumulator.read(acc_reg);
        let mut result = [0u32; 8];

        match to_type {
            ElementType::Int32 | ElementType::UInt32 => {
                // 64-bit -> 32-bit with shift and saturation
                for i in 0..8 {
                    let shifted = if shift > 0 {
                        // Round: add 0.5 ULP before shift
                        let round_bit = 1u64 << (shift - 1);
                        acc[i].wrapping_add(round_bit) >> shift
                    } else {
                        acc[i]
                    };
                    // Saturate to 32 bits
                    result[i] = shifted.min(u32::MAX as u64) as u32;
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 64-bit -> 16-bit (pack 2 lanes into each u32)
                for i in 0..4 {
                    let val0 = {
                        let shifted = if shift > 0 {
                            let round_bit = 1u64 << (shift - 1);
                            acc[i * 2].wrapping_add(round_bit) >> shift
                        } else {
                            acc[i * 2]
                        };
                        shifted.min(u16::MAX as u64) as u16
                    };
                    let val1 = {
                        let shifted = if shift > 0 {
                            let round_bit = 1u64 << (shift - 1);
                            acc[i * 2 + 1].wrapping_add(round_bit) >> shift
                        } else {
                            acc[i * 2 + 1]
                        };
                        shifted.min(u16::MAX as u64) as u16
                    };
                    result[i] = (val0 as u32) | ((val1 as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                // 64-bit float -> bf16 (pack 2 per u32)
                for i in 0..4 {
                    let f0 = f64::from_bits(acc[i * 2]) as f32;
                    let f1 = f64::from_bits(acc[i * 2 + 1]) as f32;
                    let bf0 = Self::f32_to_bf16(f0);
                    let bf1 = Self::f32_to_bf16(f1);
                    result[i] = (bf0 as u32) | ((bf1 as u32) << 16);
                }
            }
            ElementType::Float32 => {
                // 64-bit float -> 32-bit float
                for i in 0..8 {
                    let f = f64::from_bits(acc[i]) as f32;
                    result[i] = f.to_bits();
                }
            }
            _ => {}
        }

        result
    }

    /// Vector type conversion.
    fn vector_convert(src: &[u32; 8], from_type: ElementType, to_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match (from_type, to_type) {
            // BFloat16 -> Float32 (expand: 16 bf16 -> 8 f32, use lower half)
            (ElementType::BFloat16, ElementType::Float32) => {
                for i in 0..8 {
                    // Take lower bf16 from each pair
                    let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
                    result[i] = Self::bf16_to_f32(bf16).to_bits();
                }
            }
            // Float32 -> BFloat16 (pack: 8 f32 -> 16 bf16, store in lower 4 words)
            (ElementType::Float32, ElementType::BFloat16) => {
                for i in 0..4 {
                    let f0 = f32::from_bits(src[i * 2]);
                    let f1 = f32::from_bits(src[i * 2 + 1]);
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
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
                    result[i] = f as i32 as u32;
                }
            }
            // Float32 -> UInt32
            (ElementType::Float32, ElementType::UInt32) => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
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
            // Same type: pass through
            _ if from_type == to_type => {
                result = *src;
            }
            _ => {
                // Unhandled conversion: pass through
                result = *src;
            }
        }

        result
    }

    /// Vector addition by element type.
    fn vector_add(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
                // 8 × 32-bit integer lanes
                for i in 0..8 {
                    result[i] = a[i].wrapping_add(b[i]);
                }
            }
            ElementType::Float32 => {
                // 8 × 32-bit IEEE 754 float lanes
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa + fb).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 16 × 16-bit integer lanes (2 per u32)
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = a_lo.wrapping_add(b_lo);
                    let r_hi = a_hi.wrapping_add(b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                // 16 × BFloat16 lanes (2 per u32)
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo + b_lo);
                    let r_hi = Self::f32_to_bf16(a_hi + b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 × 8-bit lanes (4 per u32)
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = a_byte.wrapping_add(b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector subtraction by element type.
    fn vector_sub(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = a[i].wrapping_sub(b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa - fb).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = a_lo.wrapping_sub(b_lo);
                    let r_hi = a_hi.wrapping_sub(b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo - b_lo);
                    let r_hi = Self::f32_to_bf16(a_hi - b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = a_byte.wrapping_sub(b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector multiplication by element type.
    fn vector_mul(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = a[i].wrapping_mul(b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa * fb).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = a_lo.wrapping_mul(b_lo);
                    let r_hi = a_hi.wrapping_mul(b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo * b_lo);
                    let r_hi = Self::f32_to_bf16(a_hi * b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = a_byte.wrapping_mul(b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector multiply-accumulate (adds to 512-bit accumulator).
    fn vector_mac(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        a: &[u32; 8],
        b: &[u32; 8],
        elem_type: ElementType,
    ) {
        let current = ctx.accumulator.read(acc_reg);

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
                // 8 × 32-bit → 8 × 64-bit accumulator lanes
                let mut new_acc = [0u64; 8];
                for i in 0..8 {
                    let prod = (a[i] as u64) * (b[i] as u64);
                    new_acc[i] = current[i].wrapping_add(prod);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Float32 => {
                // 8 × f32 MAC: accumulator holds f64 for precision
                let mut new_acc = current;
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]) as f64;
                    let fb = f32::from_bits(b[i]) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + fa * fb).to_bits();
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 16 × 16-bit → 16 products, accumulated
                let mut new_acc = current;
                for i in 0..8 {
                    let a_lo = a[i] & 0xFFFF;
                    let a_hi = (a[i] >> 16) & 0xFFFF;
                    let b_lo = b[i] & 0xFFFF;
                    let b_hi = (b[i] >> 16) & 0xFFFF;
                    let prod_lo = a_lo * b_lo;
                    let prod_hi = a_hi * b_hi;
                    // Accumulate both products into single 64-bit lane
                    new_acc[i] = current[i]
                        .wrapping_add(prod_lo as u64)
                        .wrapping_add(prod_hi as u64);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::BFloat16 => {
                // 16 × bf16 MAC: convert to f32, accumulate to f64
                let mut new_acc = current;
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16) as f64;
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16) as f64;
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16) as f64;
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + a_lo * b_lo + a_hi * b_hi).to_bits();
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 × 8-bit → 32 products, accumulated
                let mut new_acc = current;
                for i in 0..8 {
                    let mut sum = 0u64;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u64;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u64;
                        sum += a_byte * b_byte;
                    }
                    new_acc[i] = current[i].wrapping_add(sum);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
        }
    }

    /// Vector minimum by element type.
    fn vector_min(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    result[i] = std::cmp::min(a[i] as i32, b[i] as i32) as u32;
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = std::cmp::min(a[i], b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = fa.min(fb).to_bits();
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo.min(b_lo));
                    let r_hi = Self::f32_to_bf16(a_hi.min(b_hi));
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = std::cmp::min(a_lo, b_lo) as u16;
                    let r_hi = std::cmp::min(a_hi, b_hi) as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = std::cmp::min(a_lo, b_lo);
                    let r_hi = std::cmp::min(a_hi, b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = std::cmp::min(a_byte, b_byte) as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = std::cmp::min(a_byte, b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector maximum by element type.
    fn vector_max(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    result[i] = std::cmp::max(a[i] as i32, b[i] as i32) as u32;
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = std::cmp::max(a[i], b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = fa.max(fb).to_bits();
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo.max(b_lo));
                    let r_hi = Self::f32_to_bf16(a_hi.max(b_hi));
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = std::cmp::max(a_lo, b_lo) as u16;
                    let r_hi = std::cmp::max(a_hi, b_hi) as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = std::cmp::max(a_lo, b_lo);
                    let r_hi = std::cmp::max(a_hi, b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = std::cmp::max(a_byte, b_byte) as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = std::cmp::max(a_byte, b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector shuffle with pattern.
    fn vector_shuffle(src: &[u32; 8], pattern: ShufflePattern) -> [u32; 8] {
        match pattern {
            ShufflePattern::Identity => *src,

            ShufflePattern::Reverse => {
                let mut result = [0u32; 8];
                for i in 0..8 {
                    result[i] = src[7 - i];
                }
                result
            }

            ShufflePattern::Broadcast(lane) => {
                let val = src[(lane & 0x07) as usize];
                [val; 8]
            }

            ShufflePattern::InterleaveLow => {
                // Interleave low halves of two conceptual vectors
                // Here we just shuffle within single vector
                let mut result = [0u32; 8];
                for i in 0..4 {
                    result[i * 2] = src[i];
                    result[i * 2 + 1] = src[i + 4];
                }
                result
            }

            ShufflePattern::InterleaveHigh => {
                // Interleave high halves
                let mut result = [0u32; 8];
                for i in 0..4 {
                    result[i * 2] = src[i + 4];
                    result[i * 2 + 1] = src[i];
                }
                result
            }

            ShufflePattern::Custom(mask) => {
                // Each 3-bit field selects a source lane
                let mut result = [0u32; 8];
                for i in 0..8 {
                    let lane_sel = ((mask >> (i * 3)) & 0x7) as usize;
                    result[i] = src[lane_sel];
                }
                result
            }
        }
    }

    /// Pack two 32-bit vectors into one 16-bit vector.
    fn vector_pack(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];

        // Pack lower halves: take low 16 bits of each input lane
        for i in 0..4 {
            let a_lo = a[i * 2] & 0xFFFF;
            let a_hi = a[i * 2 + 1] & 0xFFFF;
            result[i] = a_lo | (a_hi << 16);
        }
        for i in 0..4 {
            let b_lo = b[i * 2] & 0xFFFF;
            let b_hi = b[i * 2 + 1] & 0xFFFF;
            result[i + 4] = b_lo | (b_hi << 16);
        }

        result
    }

    /// Unpack low half (sign-extend 16-bit to 32-bit).
    fn vector_unpack_low(src: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];

        for i in 0..4 {
            // Sign-extend low 16 bits
            let val = (src[i] & 0xFFFF) as i16 as i32 as u32;
            result[i * 2] = val;
            // Sign-extend high 16 bits
            let val = ((src[i] >> 16) & 0xFFFF) as i16 as i32 as u32;
            result[i * 2 + 1] = val;
        }

        result
    }

    /// Vector equality comparison (returns mask).
    fn vector_cmp_eq(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                for i in 0..8 {
                    result[i] = if a[i] == b[i] { 0xFFFF_FFFF } else { 0 };
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = a[i] & 0xFFFF;
                    let a_hi = (a[i] >> 16) & 0xFFFF;
                    let b_lo = b[i] & 0xFFFF;
                    let b_hi = (b[i] >> 16) & 0xFFFF;
                    let r_lo = if a_lo == b_lo { 0xFFFF } else { 0 };
                    let r_hi = if a_hi == b_hi { 0xFFFF } else { 0 };
                    result[i] = r_lo | (r_hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = (a[i] >> (j * 8)) & 0xFF;
                        let b_byte = (b[i] >> (j * 8)) & 0xFF;
                        let mask = if a_byte == b_byte { 0xFF } else { 0 };
                        r |= mask << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    // ========== Vector Element Manipulation Helper Functions ==========

    /// Get lane index from operands (immediate or register).
    fn get_lane_index(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // Look for immediate in sources (typically the last source)
        for src in op.sources.iter().rev() {
            if let Operand::Immediate(imm) = src {
                return *imm as u32;
            }
            if let Operand::ScalarReg(r) = src {
                return ctx.scalar.read(*r);
            }
        }
        0 // Default: lane 0
    }

    /// Get scalar source value from operands.
    fn get_scalar_source(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        for src in &op.sources {
            match src {
                Operand::ScalarReg(r) => return ctx.scalar.read(*r),
                Operand::Immediate(imm) => return *imm as u32,
                _ => {}
            }
        }
        0
    }

    /// Write scalar result to destination.
    fn write_scalar_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: u32) {
        if let Some(Operand::ScalarReg(r)) = &op.dest {
            ctx.scalar.write(*r, value);
        }
    }

    /// Get current value of vector destination (for read-modify-write ops like insert).
    fn get_vector_dest_value(op: &SlotOp, ctx: &ExecutionContext) -> [u32; 8] {
        if let Some(Operand::VectorReg(r)) = &op.dest {
            ctx.vector.read(*r)
        } else {
            [0; 8]
        }
    }

    /// Extract a single element from a vector.
    ///
    /// Returns the element at the given lane index, converted to a u32.
    fn vector_extract(src: &[u32; 8], index: u32, elem_type: ElementType) -> u32 {
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                // 8 lanes of 32-bit elements
                let lane = (index as usize) & 0x7;
                src[lane]
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 lanes of 16-bit elements (2 per u32)
                let word_idx = ((index as usize) >> 1) & 0x7;
                let sub_idx = (index as usize) & 0x1;
                let value = (src[word_idx] >> (sub_idx * 16)) & 0xFFFF;
                // Sign-extend for signed types
                if matches!(elem_type, ElementType::Int16) {
                    value as i16 as i32 as u32
                } else {
                    value
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 lanes of 8-bit elements (4 per u32)
                let word_idx = ((index as usize) >> 2) & 0x7;
                let sub_idx = (index as usize) & 0x3;
                let value = (src[word_idx] >> (sub_idx * 8)) & 0xFF;
                // Sign-extend for signed types
                if matches!(elem_type, ElementType::Int8) {
                    value as i8 as i32 as u32
                } else {
                    value
                }
            }
        }
    }

    /// Insert a scalar value into a vector at the given index.
    fn vector_insert(dst: &mut [u32; 8], value: u32, index: u32, elem_type: ElementType) {
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                // 8 lanes of 32-bit elements
                let lane = (index as usize) & 0x7;
                dst[lane] = value;
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 lanes of 16-bit elements (2 per u32)
                let word_idx = ((index as usize) >> 1) & 0x7;
                let sub_idx = (index as usize) & 0x1;
                let shift = sub_idx * 16;
                let mask = !(0xFFFFu32 << shift);
                dst[word_idx] = (dst[word_idx] & mask) | ((value & 0xFFFF) << shift);
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 lanes of 8-bit elements (4 per u32)
                let word_idx = ((index as usize) >> 2) & 0x7;
                let sub_idx = (index as usize) & 0x3;
                let shift = sub_idx * 8;
                let mask = !(0xFFu32 << shift);
                dst[word_idx] = (dst[word_idx] & mask) | ((value & 0xFF) << shift);
            }
        }
    }

    /// Per-element select: dst[i] = mask[i] != 0 ? src1[i] : src2[i]
    fn vector_select(
        mask: &[u32; 8],
        src1: &[u32; 8],
        src2: &[u32; 8],
        elem_type: ElementType,
    ) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                // 8 lanes of 32-bit elements
                for i in 0..8 {
                    result[i] = if mask[i] != 0 { src1[i] } else { src2[i] };
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 lanes of 16-bit elements (2 per u32)
                for i in 0..8 {
                    let mask_lo = mask[i] & 0xFFFF;
                    let mask_hi = (mask[i] >> 16) & 0xFFFF;
                    let src1_lo = src1[i] & 0xFFFF;
                    let src1_hi = (src1[i] >> 16) & 0xFFFF;
                    let src2_lo = src2[i] & 0xFFFF;
                    let src2_hi = (src2[i] >> 16) & 0xFFFF;
                    let r_lo = if mask_lo != 0 { src1_lo } else { src2_lo };
                    let r_hi = if mask_hi != 0 { src1_hi } else { src2_hi };
                    result[i] = r_lo | (r_hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 lanes of 8-bit elements (4 per u32)
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let m = (mask[i] >> (j * 8)) & 0xFF;
                        let s1 = (src1[i] >> (j * 8)) & 0xFF;
                        let s2 = (src2[i] >> (j * 8)) & 0xFF;
                        let val = if m != 0 { s1 } else { s2 };
                        r |= val << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Broadcast a scalar value to all vector lanes.
    fn vector_broadcast(value: u32, elem_type: ElementType) -> [u32; 8] {
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                // Broadcast 32-bit value to all 8 lanes
                [value; 8]
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // Broadcast 16-bit value to all 16 lanes (replicate in each u32)
                let val16 = value & 0xFFFF;
                let packed = val16 | (val16 << 16);
                [packed; 8]
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // Broadcast 8-bit value to all 32 lanes (replicate in each u32)
                let val8 = value & 0xFF;
                let packed = val8 | (val8 << 8) | (val8 << 16) | (val8 << 24);
                [packed; 8]
            }
        }
    }

    // ========== Vector Shift Operation Implementations ==========

    /// Vector logical left shift: each lane is shifted left by the corresponding shift amount.
    fn vector_shift_left(src: &[u32; 8], shift: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                for i in 0..8 {
                    let sh = shift[i] & 0x1F;
                    result[i] = src[i].wrapping_shl(sh);
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let src_lo = src[i] & 0xFFFF;
                    let src_hi = (src[i] >> 16) & 0xFFFF;
                    let sh_lo = shift[i] & 0xF;
                    let sh_hi = (shift[i] >> 16) & 0xF;
                    let r_lo = (src_lo << sh_lo) & 0xFFFF;
                    let r_hi = (src_hi << sh_hi) & 0xFFFF;
                    result[i] = r_lo | (r_hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let s = (src[i] >> (j * 8)) & 0xFF;
                        let sh = (shift[i] >> (j * 8)) & 0x7;
                        let v = (s << sh) & 0xFF;
                        r |= v << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }
        result
    }

    /// Vector logical right shift: unsigned shift right.
    fn vector_shift_right_logical(src: &[u32; 8], shift: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                for i in 0..8 {
                    let sh = shift[i] & 0x1F;
                    result[i] = src[i].wrapping_shr(sh);
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let src_lo = src[i] & 0xFFFF;
                    let src_hi = (src[i] >> 16) & 0xFFFF;
                    let sh_lo = shift[i] & 0xF;
                    let sh_hi = (shift[i] >> 16) & 0xF;
                    let r_lo = src_lo >> sh_lo;
                    let r_hi = src_hi >> sh_hi;
                    result[i] = r_lo | (r_hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let s = (src[i] >> (j * 8)) & 0xFF;
                        let sh = (shift[i] >> (j * 8)) & 0x7;
                        let v = s >> sh;
                        r |= v << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }
        result
    }

    /// Vector arithmetic right shift: signed shift right (preserves sign bit).
    fn vector_shift_right_arith(src: &[u32; 8], shift: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                for i in 0..8 {
                    let sh = (shift[i] & 0x1F) as u32;
                    result[i] = ((src[i] as i32).wrapping_shr(sh)) as u32;
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let src_lo = (src[i] & 0xFFFF) as i16;
                    let src_hi = ((src[i] >> 16) & 0xFFFF) as i16;
                    let sh_lo = (shift[i] & 0xF) as u32;
                    let sh_hi = ((shift[i] >> 16) & 0xF) as u32;
                    let r_lo = ((src_lo >> sh_lo) as u16) as u32;
                    let r_hi = ((src_hi >> sh_hi) as u16) as u32;
                    result[i] = r_lo | (r_hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let s = ((src[i] >> (j * 8)) & 0xFF) as i8;
                        let sh = (shift[i] >> (j * 8)) & 0x7;
                        let v = ((s >> sh) as u8) as u32;
                        r |= v << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }
        result
    }

    /// Vector align: concatenates two 256-bit vectors and extracts 256 bits at byte offset.
    /// Result = (src1 || src2) >> (byte_shift * 8), extracting lower 256 bits.
    fn vector_align(src1: &[u32; 8], src2: &[u32; 8], byte_shift: u32) -> [u32; 8] {
        // Treat as 64 bytes (512 bits total), shift right by byte_shift bytes
        // and take lower 32 bytes (256 bits)
        let shift = (byte_shift & 0x3F) as usize; // Max 63 bytes
        let mut result = [0u32; 8];

        // Build concatenated 64-byte array: [src2 || src1] (src2 is high, src1 is low)
        // Then shift right and take lower 32 bytes
        for i in 0..8 {
            let byte_idx = i * 4 + shift;

            // Get value from concatenated vector
            let get_byte = |idx: usize| -> u8 {
                let w = idx / 4;
                let b = idx % 4;
                if w < 8 {
                    ((src1[w] >> (b * 8)) & 0xFF) as u8
                } else if w < 16 {
                    ((src2[w - 8] >> (b * 8)) & 0xFF) as u8
                } else {
                    0
                }
            };

            let b0 = get_byte(byte_idx) as u32;
            let b1 = get_byte(byte_idx + 1) as u32;
            let b2 = get_byte(byte_idx + 2) as u32;
            let b3 = get_byte(byte_idx + 3) as u32;
            result[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        result
    }

    /// Vector upshift: shift left for precision scaling (e.g., int8 -> int16).
    /// Used when accumulating into wider types.
    fn vector_upshift(
        src: &[u32; 8],
        shift: u32,
        _from_type: ElementType,
        _to_type: ElementType,
    ) -> [u32; 8] {
        // Simple implementation: just shift left by shift amount
        // In practice, this would involve type widening but for now just shift
        let mut result = [0u32; 8];
        let sh = shift & 0x1F;
        for i in 0..8 {
            result[i] = src[i].wrapping_shl(sh);
        }
        result
    }

    // ========== Conditional Vector Operation Implementations ==========

    /// Absolute value if greater than zero: dst[i] = (src[i] > 0) ? abs(src[i]) : src[i]
    /// For positive values, abs() is identity, so this is really just a pass-through.
    /// Used for ReLU-like activations where only positive values are preserved as-is.
    fn vector_abs_gtz(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    let val = src[i] as i32;
                    // If val > 0, abs(val) = val (since val is positive)
                    // If val <= 0, keep val unchanged
                    result[i] = if val > 0 { val as u32 } else { src[i] };
                }
            }
            ElementType::UInt32 => {
                // Unsigned: all values are >= 0, so "greater than zero" means != 0
                // abs() is identity for unsigned
                result = *src;
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
                    result[i] = if f > 0.0 { f.abs().to_bits() } else { src[i] };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as i16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = if lo > 0 { lo as u16 } else { lo as u16 };
                    let r_hi = if hi > 0 { hi as u16 } else { hi as u16 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                result = *src;
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((src[i] & 0xFFFF) as u16);
                    let hi = Self::bf16_to_f32(((src[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = if lo > 0.0 { Self::f32_to_bf16(lo.abs()) } else { (src[i] & 0xFFFF) as u16 };
                    let r_hi = if hi > 0.0 { Self::f32_to_bf16(hi.abs()) } else { ((src[i] >> 16) & 0xFFFF) as u16 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = if byte > 0 { byte as u8 } else { byte as u8 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                result = *src;
            }
        }

        result
    }

    /// Negate if greater than zero: dst[i] = (src[i] > 0) ? -src[i] : src[i]
    fn vector_neg_gtz(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    let val = src[i] as i32;
                    result[i] = if val > 0 { (-val) as u32 } else { src[i] };
                }
            }
            ElementType::UInt32 => {
                // For unsigned, "negate" wraps around
                for i in 0..8 {
                    result[i] = if src[i] > 0 { 0u32.wrapping_sub(src[i]) } else { src[i] };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
                    result[i] = if f > 0.0 { (-f).to_bits() } else { src[i] };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as i16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = if lo > 0 { (-lo) as u16 } else { lo as u16 };
                    let r_hi = if hi > 0 { (-hi) as u16 } else { hi as u16 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as u16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = if lo > 0 { 0u16.wrapping_sub(lo) } else { lo };
                    let r_hi = if hi > 0 { 0u16.wrapping_sub(hi) } else { hi };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((src[i] & 0xFFFF) as u16);
                    let hi = Self::bf16_to_f32(((src[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = if lo > 0.0 { Self::f32_to_bf16(-lo) } else { (src[i] & 0xFFFF) as u16 };
                    let r_hi = if hi > 0.0 { Self::f32_to_bf16(-hi) } else { ((src[i] >> 16) & 0xFFFF) as u16 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = if byte > 0 { (-byte) as u8 } else { byte as u8 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = if byte > 0 { 0u8.wrapping_sub(byte) } else { byte };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Negate if less than zero: dst[i] = (src[i] < 0) ? -src[i] : src[i]
    /// This is essentially abs() for signed types.
    fn vector_neg_ltz(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    let val = src[i] as i32;
                    result[i] = val.abs() as u32;
                }
            }
            ElementType::UInt32 => {
                // Unsigned values are never < 0, so pass through
                result = *src;
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
                    result[i] = f.abs().to_bits();
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as i16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = lo.abs() as u16;
                    let r_hi = hi.abs() as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                result = *src;
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((src[i] & 0xFFFF) as u16);
                    let hi = Self::bf16_to_f32(((src[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(lo.abs());
                    let r_hi = Self::f32_to_bf16(hi.abs());
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = byte.abs() as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                result = *src;
            }
        }

        result
    }

    /// Vector accumulate: acc += src (add to accumulator without multiply).
    fn vector_accumulate(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        src: &[u32; 8],
        elem_type: ElementType,
    ) {
        let current = ctx.accumulator.read(acc_reg);
        let mut new_acc = current;

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
                for i in 0..8 {
                    new_acc[i] = current[i].wrapping_add(src[i] as u64);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + f).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as u64;
                    let hi = ((src[i] >> 16) & 0xFFFF) as u64;
                    new_acc[i] = current[i].wrapping_add(lo).wrapping_add(hi);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((src[i] & 0xFFFF) as u16) as f64;
                    let hi = Self::bf16_to_f32(((src[i] >> 16) & 0xFFFF) as u16) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + lo + hi).to_bits();
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut sum = 0u64;
                    for j in 0..4 {
                        sum += ((src[i] >> (j * 8)) & 0xFF) as u64;
                    }
                    new_acc[i] = current[i].wrapping_add(sum);
                }
            }
        }

        ctx.accumulator.write(acc_reg, new_acc);
    }

    /// Vector negate: dst = -src (per element negation).
    fn vector_negate(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    result[i] = (-(src[i] as i32)) as u32;
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = 0u32.wrapping_sub(src[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
                    result[i] = (-f).to_bits();
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as i16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = (-lo) as u16;
                    let r_hi = (-hi) as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as u16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = 0u16.wrapping_sub(lo);
                    let r_hi = 0u16.wrapping_sub(hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((src[i] & 0xFFFF) as u16);
                    let hi = Self::bf16_to_f32(((src[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(-lo);
                    let r_hi = Self::f32_to_bf16(-hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = (-byte) as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = 0u8.wrapping_sub(byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector negate and add: dst = -src1 + src2.
    fn vector_neg_add(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    result[i] = ((-(a[i] as i32)) as i64 + (b[i] as i32) as i64) as u32;
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = b[i].wrapping_sub(a[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (-fa + fb).to_bits();
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = (-a_lo).wrapping_add(b_lo) as u16;
                    let r_hi = (-a_hi).wrapping_add(b_hi) as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = b_lo.wrapping_sub(a_lo);
                    let r_hi = b_hi.wrapping_sub(a_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(-a_lo + b_lo);
                    let r_hi = Self::f32_to_bf16(-a_hi + b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = (-a_byte).wrapping_add(b_byte) as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = b_byte.wrapping_sub(a_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector negate multiply: acc += -(src1 * src2).
    fn vector_neg_mul(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        a: &[u32; 8],
        b: &[u32; 8],
        elem_type: ElementType,
    ) {
        // This is equivalent to vector_neg_matmul (acc -= A * B is same as acc += -(A * B))
        Self::vector_matmul_sub(ctx, acc_reg, a, b, elem_type);
    }

    // ========== Vector Comparison Operation Implementations ==========

    /// Vector compare greater-or-equal: dst[i] = (a[i] >= b[i]) ? ~0 : 0
    fn vector_compare_ge(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    result[i] = if (a[i] as i32) >= (b[i] as i32) { !0 } else { 0 };
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = if a[i] >= b[i] { !0 } else { 0 };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = if fa >= fb { !0 } else { 0 };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo: u16 = if a_lo >= b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi >= b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo: u16 = if a_lo >= b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi >= b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo: u16 = if a_lo >= b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi >= b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte: u8 = if a_byte >= b_byte { !0 } else { 0 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte: u8 = if a_byte >= b_byte { !0 } else { 0 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector compare less-than: dst[i] = (a[i] < b[i]) ? ~0 : 0
    fn vector_compare_lt(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    result[i] = if (a[i] as i32) < (b[i] as i32) { !0 } else { 0 };
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = if a[i] < b[i] { !0 } else { 0 };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = if fa < fb { !0 } else { 0 };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo: u16 = if a_lo < b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi < b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo: u16 = if a_lo < b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi < b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo: u16 = if a_lo < b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi < b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte: u8 = if a_byte < b_byte { !0 } else { 0 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte: u8 = if a_byte < b_byte { !0 } else { 0 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector compare equal-to-zero: dst[i] = (a[i] == 0) ? ~0 : 0
    fn vector_compare_eqz(a: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = if a[i] == 0 { !0 } else { 0 };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    result[i] = if fa == 0.0 { !0 } else { 0 };
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                for i in 0..8 {
                    let lo = (a[i] & 0xFFFF) as u16;
                    let hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let r_lo: u16 = if lo == 0 { !0 } else { 0 };
                    let r_hi: u16 = if hi == 0 { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let r_lo: u16 = if lo == 0.0 { !0 } else { 0 };
                    let r_hi: u16 = if hi == 0.0 { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte: u8 = if byte == 0 { !0 } else { 0 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    // ========== Vector Bitwise Operation Implementations ==========

    /// Vector bitwise AND: dst = a & b
    fn vector_bitwise_and(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = a[i] & b[i];
        }
        result
    }

    /// Vector bitwise OR: dst = a | b
    fn vector_bitwise_or(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = a[i] | b[i];
        }
        result
    }

    /// Vector bitwise XOR: dst = a ^ b
    fn vector_bitwise_xor(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = a[i] ^ b[i];
        }
        result
    }

    /// Vector bitwise NOT: dst = ~a
    fn vector_bitwise_not(a: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = !a[i];
        }
        result
    }

    // ========== Vector Conditional Arithmetic Implementations ==========

    /// Subtract if less-than: dst[i] = (a[i] < b[i]) ? a[i] - b[i] : a[i]
    fn vector_sub_lt(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    let va = a[i] as i32;
                    let vb = b[i] as i32;
                    result[i] = if va < vb { va.wrapping_sub(vb) as u32 } else { a[i] };
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = if a[i] < b[i] { a[i].wrapping_sub(b[i]) } else { a[i] };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = if fa < fb { (fa - fb).to_bits() } else { a[i] };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = if a_lo < b_lo { a_lo.wrapping_sub(b_lo) as u16 } else { a_lo as u16 };
                    let r_hi = if a_hi < b_hi { a_hi.wrapping_sub(b_hi) as u16 } else { a_hi as u16 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = if a_lo < b_lo { a_lo.wrapping_sub(b_lo) } else { a_lo };
                    let r_hi = if a_hi < b_hi { a_hi.wrapping_sub(b_hi) } else { a_hi };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = if a_lo < b_lo { Self::f32_to_bf16(a_lo - b_lo) } else { (a[i] & 0xFFFF) as u16 };
                    let r_hi = if a_hi < b_hi { Self::f32_to_bf16(a_hi - b_hi) } else { ((a[i] >> 16) & 0xFFFF) as u16 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = if a_byte < b_byte { a_byte.wrapping_sub(b_byte) as u8 } else { a_byte as u8 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = if a_byte < b_byte { a_byte.wrapping_sub(b_byte) } else { a_byte };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Subtract if greater-or-equal: dst[i] = (a[i] >= b[i]) ? a[i] - b[i] : a[i]
    fn vector_sub_ge(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    let va = a[i] as i32;
                    let vb = b[i] as i32;
                    result[i] = if va >= vb { va.wrapping_sub(vb) as u32 } else { a[i] };
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = if a[i] >= b[i] { a[i].wrapping_sub(b[i]) } else { a[i] };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = if fa >= fb { (fa - fb).to_bits() } else { a[i] };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = if a_lo >= b_lo { a_lo.wrapping_sub(b_lo) as u16 } else { a_lo as u16 };
                    let r_hi = if a_hi >= b_hi { a_hi.wrapping_sub(b_hi) as u16 } else { a_hi as u16 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = if a_lo >= b_lo { a_lo.wrapping_sub(b_lo) } else { a_lo };
                    let r_hi = if a_hi >= b_hi { a_hi.wrapping_sub(b_hi) } else { a_hi };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = if a_lo >= b_lo { Self::f32_to_bf16(a_lo - b_lo) } else { (a[i] & 0xFFFF) as u16 };
                    let r_hi = if a_hi >= b_hi { Self::f32_to_bf16(a_hi - b_hi) } else { ((a[i] >> 16) & 0xFFFF) as u16 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = if a_byte >= b_byte { a_byte.wrapping_sub(b_byte) as u8 } else { a_byte as u8 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = if a_byte >= b_byte { a_byte.wrapping_sub(b_byte) } else { a_byte };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Maximum difference if less-than: dst[i] = max(a[i] - b[i], 0) for unsigned,
    /// or clamped difference for signed types.
    fn vector_maxdiff_lt(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    let va = a[i] as i32;
                    let vb = b[i] as i32;
                    let diff = va.saturating_sub(vb);
                    result[i] = diff.max(0) as u32;
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    result[i] = a[i].saturating_sub(b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa - fb).max(0.0).to_bits();
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = a_lo.saturating_sub(b_lo).max(0) as u16;
                    let r_hi = a_hi.saturating_sub(b_hi).max(0) as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = a_lo.saturating_sub(b_lo);
                    let r_hi = a_hi.saturating_sub(b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16((a_lo - b_lo).max(0.0));
                    let r_hi = Self::f32_to_bf16((a_hi - b_hi).max(0.0));
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = a_byte.saturating_sub(b_byte).max(0) as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = a_byte.saturating_sub(b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    #[test]
    fn test_vector_add_i32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorAdd {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        assert!(VectorAlu::execute(&op, &mut ctx));
        assert_eq!(ctx.vector.read(2), [11, 22, 33, 44, 55, 66, 77, 88]);
    }

    #[test]
    fn test_vector_add_i16() {
        let mut ctx = make_ctx();
        // Pack 16-bit values: [1, 2] in lane 0, etc.
        ctx.vector.write(0, [0x0002_0001, 0, 0, 0, 0, 0, 0, 0]);
        ctx.vector.write(1, [0x0020_0010, 0, 0, 0, 0, 0, 0, 0]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorAdd {
                element_type: ElementType::Int16,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2)[0], 0x0022_0011);
    }

    #[test]
    fn test_vector_sub() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [100, 200, 300, 400, 500, 600, 700, 800]);
        ctx.vector.write(1, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorSub {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [90, 180, 270, 360, 450, 540, 630, 720]);
    }

    #[test]
    fn test_vector_mul() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [2, 3, 4, 5, 6, 7, 8, 9]);
        ctx.vector.write(1, [10, 10, 10, 10, 10, 10, 10, 10]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMul {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    fn test_vector_mac() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [2, 2, 2, 2, 2, 2, 2, 2]);

        // acc0 = v0 * v1
        let op = SlotOp::new(
            SlotIndex::Accumulator,
            Operation::VectorMac {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::AccumReg(0))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let acc = ctx.accumulator.read(0);
        assert_eq!(acc, [2, 4, 6, 8, 10, 12, 14, 16]);

        // Accumulate again
        VectorAlu::execute(&op, &mut ctx);
        let acc = ctx.accumulator.read(0);
        assert_eq!(acc, [4, 8, 12, 16, 20, 24, 28, 32]);
    }

    #[test]
    fn test_vector_min_max() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [5, 10, 15, 20, 25, 30, 35, 40]);
        ctx.vector.write(1, [10, 5, 20, 15, 30, 25, 40, 35]);

        // Min
        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMin {
                element_type: ElementType::UInt32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [5, 5, 15, 15, 25, 25, 35, 35]);

        // Max
        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMax {
                element_type: ElementType::UInt32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [10, 10, 20, 20, 30, 30, 40, 40]);
    }

    #[test]
    fn test_vector_shuffle_reverse() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorShuffle {
                pattern: ShufflePattern::Reverse,
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_vector_shuffle_broadcast() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorShuffle {
                pattern: ShufflePattern::Broadcast(2),
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [30, 30, 30, 30, 30, 30, 30, 30]);
    }

    #[test]
    fn test_vector_cmp() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [1, 0, 3, 0, 5, 0, 7, 0]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorCmp {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);
        assert_eq!(result[0], 0xFFFF_FFFF); // 1 == 1
        assert_eq!(result[1], 0); // 2 != 0
        assert_eq!(result[2], 0xFFFF_FFFF); // 3 == 3
        assert_eq!(result[3], 0); // 4 != 0
    }

    #[test]
    fn test_vector_add_f32() {
        let mut ctx = make_ctx();

        // Create float vectors: [1.0, 2.0, 3.0, ...] and [0.5, 0.5, ...]
        let a: [u32; 8] = [
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            4.0f32.to_bits(),
            5.0f32.to_bits(),
            6.0f32.to_bits(),
            7.0f32.to_bits(),
            8.0f32.to_bits(),
        ];
        let b: [u32; 8] = [0.5f32.to_bits(); 8];

        ctx.vector.write(0, a);
        ctx.vector.write(1, b);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorAdd {
                element_type: ElementType::Float32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);

        // Check results: 1.5, 2.5, 3.5, ...
        assert_eq!(f32::from_bits(result[0]), 1.5);
        assert_eq!(f32::from_bits(result[1]), 2.5);
        assert_eq!(f32::from_bits(result[2]), 3.5);
        assert_eq!(f32::from_bits(result[7]), 8.5);
    }

    #[test]
    fn test_vector_mul_f32() {
        let mut ctx = make_ctx();

        // [2.0, 3.0, 4.0, ...] * [0.5, 0.5, ...]
        let a: [u32; 8] = [
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            4.0f32.to_bits(),
            5.0f32.to_bits(),
            6.0f32.to_bits(),
            7.0f32.to_bits(),
            8.0f32.to_bits(),
            9.0f32.to_bits(),
        ];
        let b: [u32; 8] = [0.5f32.to_bits(); 8];

        ctx.vector.write(0, a);
        ctx.vector.write(1, b);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMul {
                element_type: ElementType::Float32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);

        // Check results: 1.0, 1.5, 2.0, ...
        assert_eq!(f32::from_bits(result[0]), 1.0);
        assert_eq!(f32::from_bits(result[1]), 1.5);
        assert_eq!(f32::from_bits(result[2]), 2.0);
    }

    #[test]
    fn test_vector_min_max_f32() {
        let mut ctx = make_ctx();

        // Test with positive and negative floats
        let a: [u32; 8] = [
            1.0f32.to_bits(),
            (-2.0f32).to_bits(),
            3.0f32.to_bits(),
            (-4.0f32).to_bits(),
            5.0f32.to_bits(),
            6.0f32.to_bits(),
            7.0f32.to_bits(),
            8.0f32.to_bits(),
        ];
        let b: [u32; 8] = [
            0.5f32.to_bits(),
            (-1.0f32).to_bits(),
            2.0f32.to_bits(),
            (-5.0f32).to_bits(),
            4.0f32.to_bits(),
            7.0f32.to_bits(),
            6.0f32.to_bits(),
            9.0f32.to_bits(),
        ];

        ctx.vector.write(0, a);
        ctx.vector.write(1, b);

        // Min
        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMin {
                element_type: ElementType::Float32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);
        assert_eq!(f32::from_bits(result[0]), 0.5);  // min(1.0, 0.5)
        assert_eq!(f32::from_bits(result[1]), -2.0); // min(-2.0, -1.0)
        assert_eq!(f32::from_bits(result[3]), -5.0); // min(-4.0, -5.0)

        // Max
        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMax {
                element_type: ElementType::Float32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);
        assert_eq!(f32::from_bits(result[0]), 1.0);  // max(1.0, 0.5)
        assert_eq!(f32::from_bits(result[1]), -1.0); // max(-2.0, -1.0)
        assert_eq!(f32::from_bits(result[3]), -4.0); // max(-4.0, -5.0)
    }

    #[test]
    fn test_vector_add_bf16() {
        let mut ctx = make_ctx();

        // BFloat16 is upper 16 bits of f32
        // Pack two bf16 values per u32: low half and high half
        fn pack_bf16(lo: f32, hi: f32) -> u32 {
            let lo_bits = (lo.to_bits() >> 16) as u16;
            let hi_bits = (hi.to_bits() >> 16) as u16;
            (lo_bits as u32) | ((hi_bits as u32) << 16)
        }

        // Create vectors with bf16 pairs
        let a: [u32; 8] = [
            pack_bf16(1.0, 2.0),
            pack_bf16(3.0, 4.0),
            0, 0, 0, 0, 0, 0,
        ];
        let b: [u32; 8] = [
            pack_bf16(0.5, 0.5),
            pack_bf16(0.5, 0.5),
            0, 0, 0, 0, 0, 0,
        ];

        ctx.vector.write(0, a);
        ctx.vector.write(1, b);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorAdd {
                element_type: ElementType::BFloat16,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);

        // Extract and check bf16 values
        let lo0 = VectorAlu::bf16_to_f32((result[0] & 0xFFFF) as u16);
        let hi0 = VectorAlu::bf16_to_f32(((result[0] >> 16) & 0xFFFF) as u16);

        // BFloat16 has limited precision, so check approximate equality
        assert!((lo0 - 1.5).abs() < 0.1, "Expected ~1.5, got {}", lo0);
        assert!((hi0 - 2.5).abs() < 0.1, "Expected ~2.5, got {}", hi0);
    }

    #[test]
    fn test_vector_matmul_dense_int32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [2, 2, 2, 2, 2, 2, 2, 2]);

        let op = SlotOp::new(
            SlotIndex::Accumulator,
            Operation::VectorMatMulDense {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::AccumReg(0))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let acc = ctx.accumulator.read(0);
        // Each lane: a[i] * b[i] accumulated
        assert_eq!(acc, [2, 4, 6, 8, 10, 12, 14, 16]);

        // Accumulate again
        VectorAlu::execute(&op, &mut ctx);
        let acc = ctx.accumulator.read(0);
        assert_eq!(acc, [4, 8, 12, 16, 20, 24, 28, 32]);
    }

    #[test]
    fn test_vector_srs_int32() {
        let mut ctx = make_ctx();
        // Set up accumulator with values that need shifting
        ctx.accumulator.write(0, [256, 512, 768, 1024, 1280, 1536, 1792, 2048]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorSRS {
                from_type: ElementType::Int32,
                to_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::AccumReg(0))
        .with_source(Operand::Immediate(4)); // Shift right by 4

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(0);
        // 256 >> 4 = 16, 512 >> 4 = 32, etc (with rounding)
        assert_eq!(result[0], 16);
        assert_eq!(result[1], 32);
        assert_eq!(result[2], 48);
        assert_eq!(result[3], 64);
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

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorConvert {
                from_type: ElementType::BFloat16,
                to_type: ElementType::Float32,
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

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

    #[test]
    fn test_vector_convert_int32_to_f32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorConvert {
                from_type: ElementType::Int32,
                to_type: ElementType::Float32,
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

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

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMov {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [10, 20, 30, 40, 50, 60, 70, 80]);
    }
}
