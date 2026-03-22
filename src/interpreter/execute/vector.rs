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

use crate::interpreter::bundle::{ElementType, Operand, ShufflePattern, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::state::{Vec512, Acc1024};
use crate::tablegen::SemanticOp;

use super::vector_config::MatMulConfig;
use super::vector_matmul;
use super::vector_srs::{self, RoundingMode};
use super::vector_ups;
use super::vector_pack;

/// Vector ALU execution unit.
pub struct VectorAlu;

impl VectorAlu {
    /// Execute a vector operation.
    ///
    /// Returns `true` if the operation was handled, `false` if not a vector op.
    /// Dispatches on `op.semantic` (SemanticOp) with metadata from SlotOp fields.
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
        let Some(semantic) = op.semantic else {
            return false;
        };

        // Only handle vector operations
        if !op.is_vector {
            return false;
        }

        let et = op.element_type.unwrap_or(ElementType::Int32);

        log::trace!("[VECTOR_ALU] Checking semantic={:?} element_type={:?} dest={:?}",
            semantic, op.element_type, op.dest);

        // TODO: MAC-family operations should use config-word-driven geometry
        // instead of element-wise multiply. The config word parser exists in
        // MatMulConfig::from_config_word() but the matmul kernel needs correct
        // element permutation tables before it can be wired up here.
        // See: vector_config.rs CONFIG_GEOMETRY_TABLE, vector_matmul.rs matmul_config_driven

        // Process the low 256-bit half.
        let handled = Self::execute_half(op, ctx, semantic, et);

        // For 512-bit (x-register) operations, also process the high half.
        // Create a temporary op with all VectorReg indices incremented by 1.
        if handled && op.is_wide_vector {
            let mut hi_op = op.clone();
            Self::increment_vector_regs(&mut hi_op);
            Self::execute_half(&hi_op, ctx, semantic, et);
        }

        handled
    }

    /// Increment all VectorReg indices by 1 (low half -> high half).
    fn increment_vector_regs(op: &mut SlotOp) {
        for src in &mut op.sources {
            if let Operand::VectorReg(r) = src {
                *r += 1;
            }
        }
        if let Some(Operand::VectorReg(r)) = &mut op.dest {
            *r += 1;
        }
    }

    /// Execute one 256-bit half of a vector operation.
    fn execute_half(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        et: ElementType,
    ) -> bool {

        match semantic {
            // ========== Arithmetic ==========

            SemanticOp::Add => {
                // Two variants:
                // 1. VADD_elem: simple vector add (2 sources)
                // 2. VADDSUB: conditional add/subtract (3 sources: s1, s2, sel)
                //    vaddsub.8 $d, $s1, $s2, $sel
                //    d[i] = (sel[i] & 1) ? s1[i] - s2[i] : s1[i] + s2[i]
                let vec_source_count = op.sources.iter()
                    .filter(|s| matches!(s, Operand::VectorReg(_)))
                    .count();

                if vec_source_count >= 3 {
                    let s1 = Self::get_vector_source(op, ctx, 0);
                    let s2 = Self::get_vector_source(op, ctx, 1);
                    let sel = Self::get_vector_source(op, ctx, 2);
                    let result = Self::vector_addsub(&s1, &s2, &sel, et);
                    Self::write_vector_dest(op, ctx, result);
                } else {
                    let (a, b) = Self::get_two_vector_sources(op, ctx);
                    let result = Self::vector_add(&a, &b, et);
                    Self::write_vector_dest(op, ctx, result);
                }
                true
            }

            SemanticOp::Sub => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_sub(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Mul => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_mul(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Mac => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_mac(ctx, acc_reg, &a, &b, et);
                true
            }

            SemanticOp::Min => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_min(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Max => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_max(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Neg => {
                // Two variants:
                // 1. VNEG (accumulator): vneg $dst, $acc1, $c
                //    Negate accumulator value: dst = -acc1
                // 2. Vector negate: vneg $d, $s1
                //    Per-element negate: d[i] = -s1[i]
                let has_acc_source = op.sources.iter().any(|s| matches!(s, Operand::AccumReg(_)));

                if has_acc_source {
                    // Accumulator negate: negate both halves of cm register.
                    // Config word controls zero_acc and shift16 (same as vadd).
                    let acc_reg = op.sources.iter().find_map(|s| match s {
                        Operand::AccumReg(r) => Some(*r),
                        _ => None,
                    }).unwrap_or(0);
                    let dst_reg = Self::get_acc_dest(op);
                    let conf = Self::get_config_register(op, ctx).unwrap_or(0);
                    let zero_acc1 = (conf & 1) != 0;
                    let is_float = op.encoding_name.as_ref()
                        .map_or(false, |n| n.contains("_F"));

                    for half in 0..2u8 {
                        let src = ctx.accumulator.read(acc_reg + half);
                        let mut result = [0u64; 8];

                        if is_float {
                            // Float: negate each fp32 value (two per u64 lane)
                            for i in 0..8 {
                                let lo = f32::from_bits(src[i] as u32);
                                let hi = f32::from_bits((src[i] >> 32) as u32);
                                let r_lo = if zero_acc1 { 0.0f32 } else { -lo };
                                let r_hi = if zero_acc1 { 0.0f32 } else { -hi };
                                result[i] = (r_lo.to_bits() as u64) | ((r_hi.to_bits() as u64) << 32);
                            }
                        } else {
                            // Integer: negate each i64 lane
                            for i in 0..8 {
                                let v = if zero_acc1 { 0i64 } else { src[i] as i64 };
                                result[i] = v.wrapping_neg() as u64;
                            }
                        }

                        ctx.accumulator.write(dst_reg + half, result);
                    }
                } else {
                    let src = Self::get_vector_source(op, ctx, 0);
                    let result = Self::vector_negate(&src, et);
                    Self::write_vector_dest(op, ctx, result);
                }
                true
            }

            // ========== Shuffle/Pack/Unpack ==========

            SemanticOp::Shuffle => {
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_shuffle(&src, op.shuffle_pattern);
                Self::write_vector_dest(op, ctx, result);
                true
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

            // ========== Comparison ==========

            SemanticOp::Cmp => {
                // Vector compare equal
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_cmp_eq(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::SetGe => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_compare_ge(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::SetLt => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_compare_lt(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::SetEq => {
                // Vector equal-to-zero (from VectorEqz)
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_compare_eqz(&src, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::MaxLt => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_max(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::MinGe => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_min(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            // ========== Matrix Operations ==========

            SemanticOp::MatMul => {
                // Dense/sparse matrix multiply: acc += A * B
                // BFloat16 variant uses dedicated bf16 path
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                match et {
                    ElementType::BFloat16 => Self::vector_matmul_bf16(ctx, acc_reg, &a, &b, true),
                    _ => Self::vector_matmul_dense(ctx, acc_reg, &a, &b, et),
                }
                true
            }

            SemanticOp::MatMulSub => {
                // Matrix multiply-subtract: acc -= A * B
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                match et {
                    ElementType::BFloat16 => Self::vector_matmul_bf16(ctx, acc_reg, &a, &b, false),
                    _ => Self::vector_matmul_sub(ctx, acc_reg, &a, &b, et),
                }
                true
            }

            SemanticOp::NegMatMul => {
                // Negated matrix multiply: acc += -(A * B)
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_neg_matmul(ctx, acc_reg, &a, &b, et);
                true
            }

            SemanticOp::AddMac => {
                // Double accumulator: acc1 = acc1 + acc2 + A * B
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                let acc2_reg = Self::get_acc_source(op);
                Self::vector_double_acc_mac(ctx, acc_reg, acc2_reg, &a, &b, et, true);
                true
            }

            SemanticOp::SubMac => {
                // Double accumulator: acc1 = acc1 - acc2 + A * B
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                let acc2_reg = Self::get_acc_source(op);
                Self::vector_double_acc_mac(ctx, acc_reg, acc2_reg, &a, &b, et, false);
                true
            }

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
                // Type conversion (e.g., bf16 <-> f32)
                let src = Self::get_vector_source(op, ctx, 0);
                let from = op.from_type.unwrap_or(ElementType::Int32);
                let result = Self::vector_convert(&src, from, et);
                Self::write_vector_dest(op, ctx, result);
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
                    _ => {
                        // Fallback: truncate to vector (shouldn't happen for real VUPS)
                        let mut v = [0u32; 8];
                        for i in 0..8 {
                            v[i] = acc_result[i] as u32;
                        }
                        Self::write_vector_dest(op, ctx, v);
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
                // TODO: VPUSH (shift+insert) operates on full 512-bit vectors
                // and cannot be properly handled in the 256-bit half-split path.
                // For now, fall through to legacy insert-at-index behavior.
                let mut dst = Self::get_vector_dest_value(op, ctx);
                let value = Self::get_scalar_source(op, ctx);
                let index = Self::get_lane_index(op, ctx);
                Self::vector_insert(&mut dst, value, index, et);
                Self::write_vector_dest(op, ctx, dst);
                true
            }

            SemanticOp::VectorSelect => {
                // VSEL.32 d, s1, s2, sel: per-lane conditional select.
                // input_order is [s1, s2, sel] where sel is a scalar register
                // whose bits control per-element selection.
                //
                // Hardware semantics (aietools aie_api select documentation):
                //   d[i] = (sel >> i) & 1 == 0 ? s1[i] : s2[i]
                //
                // When sel bit is 0: take from s1 (first vector source).
                // When sel bit is 1: take from s2 (second vector source).
                let s1 = Self::get_vector_source(op, ctx, 0);
                let s2 = Self::get_vector_source(op, ctx, 1);
                // Read the scalar select mask from the last source operand
                let sel_scalar = op.sources.get(2).map(|s| match s {
                    Operand::ScalarReg(r) => ctx.scalar.read(*r),
                    Operand::Immediate(v) => *v as u32,
                    _ => Self::get_scalar_source(op, ctx),
                }).unwrap_or(0);
                // Expand scalar mask to per-lane vector mask.
                let sel = Self::expand_select_mask(sel_scalar, et);
                // vector_select does: mask != 0 ? arg1 : arg2
                // Hardware wants: mask != 0 ? s2 : s1
                // So pass s2 as arg1 and s1 as arg2.
                let result = Self::vector_select(&sel, &s2, &s1, et);
                log::debug!(
                    "[VSEL] sel_scalar=0x{:X} sel={:?} s1={:?} s2={:?} -> {:?} (sources={:?}, dest={:?})",
                    sel_scalar, sel, s1, s2, result, op.sources, op.dest
                );
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::VectorClear => {
                Self::write_vector_dest(op, ctx, [0u32; 8]);
                true
            }

            SemanticOp::VectorBroadcast => {
                // Two variants:
                // 1. VBCST: broadcast scalar to all lanes: vbcst.8 $dst, $scalar
                // 2. VEXTBCST: extract element from vector, then broadcast:
                //    vextbcst.16 $dst, $src_vec, $idx
                let has_vector_source = op.sources.iter().any(|s| matches!(s, Operand::VectorReg(_)));

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
                true
            }

            // ========== Shift Operations ==========

            SemanticOp::Shl => {
                let (a, shift) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_shift_left(&a, &shift, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Srl => {
                let (a, shift) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_shift_right_logical(&a, &shift, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Sra => {
                let (a, shift) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_shift_right_arith(&a, &shift, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Align => {
                // Concatenate two vectors and shift
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let shift = Self::get_lane_index(op, ctx);
                let result = Self::vector_align(&a, &b, shift);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            // ========== Conditional Vector Operations ==========

            SemanticOp::AbsGtz => {
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_abs_gtz(&src, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::NegGtz => {
                // VNEG_GTZ: conditional negate based on another vector's sign.
                // vneg_gtz.16 $d, $cmp, $s1:
                //   d[i] = (cmp[i] > 0) ? -s1[i] : s1[i]
                let vec_source_count = op.sources.iter()
                    .filter(|s| matches!(s, Operand::VectorReg(_)))
                    .count();

                if vec_source_count >= 2 {
                    let cmp = Self::get_vector_source(op, ctx, 0);
                    let s1 = Self::get_vector_source(op, ctx, 1);
                    let result = Self::vector_bneg_gtz(&cmp, &s1, et);
                    Self::write_vector_dest(op, ctx, result);
                } else {
                    let src = Self::get_vector_source(op, ctx, 0);
                    let result = Self::vector_neg_gtz(&src, et);
                    Self::write_vector_dest(op, ctx, result);
                }
                true
            }

            SemanticOp::NegLtz => {
                // TODO: VBNEG_LTZ has two vector sources (cmp, s1) for
                // conditional negate. Currently using single-source abs path
                // which is correct when cmp == src but wrong otherwise.
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_neg_ltz(&src, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Accumulate => {
                // VADD/VSUB/VNEGADD/VNEGSUB: add/subtract two accumulator
                // registers. Format: `vadd $dst, $acc1, $acc2, $config`.
                //
                // Check if sources contain AccumReg (acc-to-acc operation)
                // or VectorReg (legacy vector accumulate path).
                let has_acc_source = op.sources.iter().any(|s| matches!(s, Operand::AccumReg(_)));

                if has_acc_source {
                    Self::execute_acc_add_sub(op, ctx);
                } else {
                    // Legacy: accumulate a vector source into an accumulator.
                    let src = Self::get_vector_source(op, ctx, 0);
                    let acc_reg = Self::get_acc_dest(op);
                    Self::vector_accumulate(ctx, acc_reg, &src, et);
                }
                true
            }

            SemanticOp::NegAdd => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_neg_add(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::NegMul => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_neg_mul(ctx, acc_reg, &a, &b, et);
                true
            }

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

            // ========== Conditional Arithmetic ==========

            SemanticOp::SubLt => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_sub_lt(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::SubGe => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_sub_ge(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::MaxDiffLt => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_maxdiff_lt(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            _ => false, // Not a vector operation handled here
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
    ///
    /// Handles both VectorReg (native 256-bit) and AccumReg (truncated from
    /// 512-bit to 256-bit by taking the low 32 bits of each u64 lane).
    /// This truncation matches hardware behavior for VMOV x, bml/bmh where
    /// the 256-bit accumulator half maps directly to 8 x u32 lanes.
    fn read_vector_operand(operand: &Operand, ctx: &ExecutionContext) -> [u32; 8] {
        match operand {
            Operand::VectorReg(r) => ctx.vector.read(*r),
            Operand::AccumReg(r) => {
                // Accumulator -> vector: take low 32 bits of each lane.
                let acc = ctx.accumulator.read(*r);
                let mut v = [0u32; 8];
                for i in 0..8 {
                    v[i] = acc[i] as u32;
                }
                v
            }
            Operand::Immediate(val) => {
                // Scalar broadcast into all lanes (for immediate vector operands).
                [*val as u32; 8]
            }
            other => {
                log::error!(
                    "[VECTOR] read_vector_operand: unexpected operand {:?} -- \
                     returning zeros, check decoder",
                    other
                );
                [0; 8]
            }
        }
    }

    /// Write result to vector destination.
    ///
    /// Handles both VectorReg (native 256-bit) and AccumReg (widened from
    /// 256-bit to 512-bit by zero-extending each u32 lane to u64).
    fn write_vector_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: [u32; 8]) {
        match &op.dest {
            Some(Operand::VectorReg(r)) => {
                ctx.vector.write(*r, value);
            }
            Some(Operand::AccumReg(r)) => {
                // Vector -> accumulator: zero-extend each u32 to u64.
                let mut acc = [0u64; 8];
                for i in 0..8 {
                    acc[i] = value[i] as u64;
                }
                ctx.accumulator.write(*r, acc);
            }
            Some(Operand::ScalarReg(r)) => {
                // Vector comparison -> scalar mask register (r16-r23).
                // Condense per-lane all-ones/all-zeros into a packed bitmask.
                let mask = Self::condense_comparison_mask(&value, op.element_type);
                ctx.scalar.write(*r, mask);
            }
            Some(other) => {
                log::error!(
                    "[VECTOR] write_vector_dest: unexpected dest {:?} -- \
                     result discarded, check decoder",
                    other
                );
            }
            None => {
                // Some vector operations have no explicit destination
                // (e.g., comparison that sets status flags).
            }
        }
    }

    /// Condense a per-lane comparison mask ([u32; 8] of all-ones/zeros)
    /// into a packed scalar bitmask. One bit per element lane.
    ///
    /// - 32-bit elements: 8 lanes -> 8-bit mask (bits [7:0])
    /// - 16-bit elements: 16 lanes -> 16-bit mask (bits [15:0])
    /// - 8-bit elements: 32 lanes -> 32-bit mask (bits [31:0])
    fn condense_comparison_mask(value: &[u32; 8], elem_type: Option<ElementType>) -> u32 {
        match elem_type {
            Some(ElementType::Int32) | Some(ElementType::UInt32)
            | Some(ElementType::Float32) | None => {
                // 8 x 32-bit lanes: one bit per u32 word
                let mut mask = 0u32;
                for i in 0..8 {
                    if value[i] != 0 {
                        mask |= 1 << i;
                    }
                }
                mask
            }
            Some(ElementType::Int16) | Some(ElementType::UInt16)
            | Some(ElementType::BFloat16) => {
                // 16 x 16-bit lanes: two lanes packed per u32 word
                let mut mask = 0u32;
                for i in 0..8 {
                    if value[i] & 0xFFFF != 0 {
                        mask |= 1 << (i * 2);
                    }
                    if value[i] & 0xFFFF_0000 != 0 {
                        mask |= 1 << (i * 2 + 1);
                    }
                }
                mask
            }
            Some(ElementType::Int8) | Some(ElementType::UInt8) => {
                // 32 x 8-bit lanes: four lanes packed per u32 word
                let mut mask = 0u32;
                for i in 0..8 {
                    for j in 0..4 {
                        if (value[i] >> (j * 8)) & 0xFF != 0 {
                            mask |= 1 << (i * 4 + j);
                        }
                    }
                }
                mask
            }
            _ => {
                // Unknown element type, try 32-bit default
                let mut mask = 0u32;
                for i in 0..8 {
                    if value[i] != 0 {
                        mask |= 1 << i;
                    }
                }
                mask
            }
        }
    }

    /// Check if a semantic operation is a MAC-family instruction.
    fn is_mac_family(semantic: SemanticOp) -> bool {
        matches!(
            semantic,
            SemanticOp::Mac
                | SemanticOp::MatMul
                | SemanticOp::MatMulSub
                | SemanticOp::NegMatMul
                | SemanticOp::AddMac
                | SemanticOp::SubMac
        )
    }

    /// Execute a MAC-family instruction with config-word-driven geometry.
    ///
    /// Unlike simple vector ops that can be split into independent 256-bit
    /// halves, MAC operations read full 512-bit vectors and write 1024-bit
    /// accumulators. The config word (from a scalar register) selects the
    /// tile geometry and operational flags.
    fn execute_mac_config_driven(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        et: ElementType,
    ) -> bool {
        // Extract config register value from sources.
        let conf_val = match Self::get_config_register(op, ctx) {
            Some(v) => v,
            None => {
                // No config register present (unit tests or legacy path).
                // Fall back to the old element-wise / half-split behavior.
                return Self::execute_mac_legacy(op, ctx, semantic, et);
            }
        };

        // Determine if this is a bf16 instruction.
        let is_bf16 = matches!(et, ElementType::BFloat16 | ElementType::Float32)
            || op.encoding_name.as_ref().map_or(false, |n| n.contains(".f"));

        // Parse config word into geometry and flags.
        let mut config = match MatMulConfig::from_config_word(conf_val, is_bf16) {
            Some(c) => c,
            None => {
                // Unknown geometry -- fall back to element-wise MAC.
                // This happens when the config word encodes an invalid
                // (amode, bmode, variant) triple.
                log::trace!(
                    "[VECTOR_MAC] unknown config word 0x{:08x} (amode={}, bmode={}, variant={}), \
                     falling back to element-wise",
                    conf_val,
                    (conf_val >> 1) & 3,
                    (conf_val >> 3) & 3,
                    (conf_val >> 5) & 7,
                );
                return Self::execute_mac_legacy(op, ctx, semantic, et);
            }
        };

        log::trace!(
            "[VECTOR_MAC] config=0x{:08x} -> {}x{}x{} bits_x={} bits_y={} \
             signed=({},{}) acc_width={:?} accum={} sub={}",
            conf_val,
            config.rows, config.inner, config.cols,
            config.a_type.bits(), config.b_type.bits(),
            config.x_signed, config.y_signed,
            config.acc_width, config.accumulate, config.subtract,
        );

        // Override subtract flag based on instruction type.
        match semantic {
            SemanticOp::MatMulSub => config.subtract = true,
            SemanticOp::NegMatMul => config.subtract = !config.subtract,
            _ => {}
        }

        // Get accumulator register index. For cm registers, the low half
        // is acc_reg and the high half is acc_reg + 1.
        let acc_reg = Self::get_acc_dest(op);
        let acc_lo_reg = acc_reg;
        let acc_hi_reg = acc_reg + 1;

        // Read current accumulator state (both halves for cm register).
        let mut acc_lo = ctx.accumulator.read(acc_lo_reg);
        let mut acc_hi = ctx.accumulator.read(acc_hi_reg);

        // Read full 512-bit vector inputs. Each x-register is two
        // consecutive 256-bit w-registers.
        let (a_lo, a_hi) = Self::get_wide_vector_source(op, ctx, 0);
        let (b_lo, b_hi) = Self::get_wide_vector_source(op, ctx, 1);

        // For AddMac/SubMac, handle the second accumulator source.
        if matches!(semantic, SemanticOp::AddMac | SemanticOp::SubMac) {
            let acc2_reg = Self::get_acc_source(op);
            let acc2_lo = ctx.accumulator.read(acc2_reg);
            let acc2_hi = ctx.accumulator.read(acc2_reg + 1);
            // Add or subtract the second accumulator.
            let negate = matches!(semantic, SemanticOp::SubMac);
            for i in 0..8 {
                if negate {
                    acc_lo[i] = acc_lo[i].wrapping_sub(acc2_lo[i]);
                    acc_hi[i] = acc_hi[i].wrapping_sub(acc2_hi[i]);
                } else {
                    acc_lo[i] = acc_lo[i].wrapping_add(acc2_lo[i]);
                    acc_hi[i] = acc_hi[i].wrapping_add(acc2_hi[i]);
                }
            }
        }

        // Perform the config-driven matrix multiply.
        vector_matmul::matmul_config_driven(
            &mut acc_lo, &mut acc_hi,
            &a_lo, &a_hi,
            &b_lo, &b_hi,
            &config,
        );

        // Write back both accumulator halves.
        ctx.accumulator.write(acc_lo_reg, acc_lo);
        ctx.accumulator.write(acc_hi_reg, acc_hi);

        true
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
    fn execute_acc_add_sub(op: &SlotOp, ctx: &mut ExecutionContext) {
        // Get the two AccumReg source indices.
        let mut acc_sources: Vec<u8> = Vec::new();
        for src in &op.sources {
            if let Operand::AccumReg(r) = src {
                acc_sources.push(*r);
            }
        }

        let acc1_reg = if !acc_sources.is_empty() { acc_sources[0] } else { 0 };
        let acc2_reg = if acc_sources.len() >= 2 { acc_sources[1] } else { acc1_reg };
        let dst_reg = Self::get_acc_dest(op);

        // Read config register.
        let conf = Self::get_config_register(op, ctx).unwrap_or(0);
        let zero_acc1 = (conf & 1) != 0;
        let shift16 = ((conf >> 10) & 1) != 0;
        let sub_acc1 = ((conf >> 11) & 1) != 0;
        let sub_acc2 = ((conf >> 12) & 1) != 0;

        // Determine base operation from encoding name.
        let enc_lower = op.encoding_name.as_ref().map_or(String::new(), |n| n.to_lowercase());
        let base_sub = enc_lower.starts_with("vsub");
        let base_neg = enc_lower.starts_with("vneg");
        let base_negsub = enc_lower.starts_with("vnegsub");
        let is_float = enc_lower.contains("_f");

        // Compute effective signs for acc1 and acc2.
        // Base operation sets initial signs, config modifiers can flip them.
        let negate_acc1 = base_neg || base_negsub || sub_acc1;
        let negate_acc2 = base_sub || base_negsub || sub_acc2;

        // Process both halves of the cm register (low=reg, high=reg+1).
        for half in 0..2u8 {
            let r1 = acc1_reg + half;
            let r2 = acc2_reg + half;
            let rd = dst_reg + half;

            let a1 = ctx.accumulator.read(r1);
            let a2 = ctx.accumulator.read(r2);
            let mut result = [0u64; 8];

            if is_float {
                for i in 0..8 {
                    let a1_lo = if zero_acc1 { 0.0 } else { f32::from_bits(a1[i] as u32) };
                    let a1_hi = if zero_acc1 { 0.0 } else { f32::from_bits((a1[i] >> 32) as u32) };
                    let a2_lo = f32::from_bits(a2[i] as u32);
                    let a2_hi = f32::from_bits((a2[i] >> 32) as u32);

                    let v1_lo = if negate_acc1 { -a1_lo } else { a1_lo };
                    let v1_hi = if negate_acc1 { -a1_hi } else { a1_hi };
                    let v2_lo = if negate_acc2 { -a2_lo } else { a2_lo };
                    let v2_hi = if negate_acc2 { -a2_hi } else { a2_hi };

                    let res_lo = v1_lo + v2_lo;
                    let res_hi = v1_hi + v2_hi;

                    result[i] = (res_lo.to_bits() as u64) | ((res_hi.to_bits() as u64) << 32);
                }
            } else {
                for i in 0..8 {
                    let v1 = if zero_acc1 { 0i64 } else { a1[i] as i64 };
                    let v2 = a2[i] as i64;

                    let v1 = if negate_acc1 { v1.wrapping_neg() } else { v1 };
                    let v2 = if negate_acc2 { v2.wrapping_neg() } else { v2 };

                    let mut res = v1.wrapping_add(v2);

                    // shift16: arithmetic right shift by 16 bits
                    if shift16 {
                        res >>= 16;
                    }

                    result[i] = res as u64;
                }
            }

            ctx.accumulator.write(rd, result);
        }
    }

    /// Legacy MAC execution path: half-split element-wise multiply.
    ///
    /// Used when no config register is present (unit tests) or the config
    /// word encodes an unrecognized geometry.
    fn execute_mac_legacy(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        et: ElementType,
    ) -> bool {
        let handled = Self::execute_half(op, ctx, semantic, et);

        if handled && op.is_wide_vector {
            let mut hi_op = op.clone();
            Self::increment_vector_regs(&mut hi_op);
            Self::execute_half(&hi_op, ctx, semantic, et);
        }

        handled
    }

    /// Get config register value from a scalar register in sources.
    ///
    /// The config register is the last scalar register source in the VMAC
    /// instruction (r0-r15 in `vmac cm0, cm0, x0, x0, r0`).
    /// Returns None if no scalar register is present (e.g., unit tests).
    fn get_config_register(op: &SlotOp, ctx: &ExecutionContext) -> Option<u32> {
        // Scan sources in reverse order -- the config register is typically last.
        for src in op.sources.iter().rev() {
            if let Operand::ScalarReg(r) = src {
                return Some(ctx.scalar.read(*r));
            }
        }
        None
    }

    /// Get a 512-bit wide vector source (two consecutive 256-bit registers).
    ///
    /// Returns (low_half, high_half) where each is [u32; 8] = 256 bits.
    /// For `x0`, this returns (wl0, wh0) = (vreg 0, vreg 1).
    fn get_wide_vector_source(
        op: &SlotOp,
        ctx: &ExecutionContext,
        source_index: usize,
    ) -> ([u32; 8], [u32; 8]) {
        // Find the nth VectorReg source.
        let mut vec_count = 0;
        for src in &op.sources {
            if let Operand::VectorReg(r) = src {
                if vec_count == source_index {
                    let lo = ctx.vector.read(*r);
                    // High half is the next register.
                    let hi = if op.is_wide_vector {
                        ctx.vector.read(r + 1)
                    } else {
                        [0u32; 8]
                    };
                    return (lo, hi);
                }
                vec_count += 1;
            }
        }
        // Fallback: return zeros.
        ([0u32; 8], [0u32; 8])
    }

    /// Get accumulator destination register.
    fn get_acc_dest(op: &SlotOp) -> u8 {
        match &op.dest {
            Some(Operand::AccumReg(r)) => *r,
            other => {
                log::error!(
                    "[VECTOR] get_acc_dest: expected AccumReg, got {:?} -- defaulting to acc0",
                    other
                );
                0
            }
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
        log::error!(
            "[VECTOR] get_acc_source: no AccumReg found in sources {:?} -- defaulting to acc0",
            op.sources
        );
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

    /// Shift-Round-Saturate: convert accumulator lanes to narrower vector output.
    ///
    /// The accumulator always has 8 lanes of 64-bit each (one value per u64).
    /// SRS reads 8 accumulator values and converts them to narrower output,
    /// packing multiple values per u32 word for 16-bit and 8-bit outputs.
    ///
    /// Delegates to the `vector_srs` module which implements the full 10-mode
    /// SRS pipeline (shift, round, saturate) per AIE2 hardware specification.
    /// Float types (BFloat16, Float32) are handled inline since they bypass
    /// the integer rounding pipeline.
    fn vector_srs(
        ctx: &ExecutionContext,
        acc_reg: u8,
        shift: u32,
        _from_type: ElementType,
        to_type: ElementType,
    ) -> [u32; 8] {
        let acc = ctx.accumulator.read(acc_reg);
        let mut result = [0u32; 8];

        // Read rounding and saturation from the SRS control register state.
        let cfg = &ctx.srs_config;
        let mode = RoundingMode::from_raw(cfg.rounding_mode)
            .unwrap_or(RoundingMode::PosInf);
        let saturate = cfg.saturate();
        let sym_sat = cfg.symmetric_saturate();

        let signed_output = matches!(
            to_type,
            ElementType::Int8 | ElementType::Int16 | ElementType::Int32
        );

        // 8 accumulator lanes, each u64 holds one value.
        match to_type {
            ElementType::Int32 | ElementType::UInt32 => {
                for i in 0..8 {
                    let val = acc[i] as i64;
                    let out = vector_srs::srs_lane(
                        val, shift, signed_output, 32,
                        saturate, sym_sat, mode,
                    );
                    result[i] = out as u32;
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 8 acc lanes -> 8 x 16-bit, packed 2 per u32 (4 words used)
                for i in 0..4 {
                    let val0 = acc[i * 2] as i64;
                    let val1 = acc[i * 2 + 1] as i64;
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
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..2 {
                    let mut word = 0u32;
                    for j in 0..4 {
                        let lane = i * 4 + j;
                        if lane < 8 {
                            let val = acc[lane] as i64;
                            let out = vector_srs::srs_lane(
                                val, shift, signed_output, 8,
                                saturate, sym_sat, mode,
                            );
                            word |= (out as u8 as u32) << (j * 8);
                        }
                    }
                    result[i] = word;
                }
            }
            ElementType::BFloat16 => {
                for i in 0..4 {
                    let f0 = f64::from_bits(acc[i * 2]) as f32;
                    let f1 = f64::from_bits(acc[i * 2 + 1]) as f32;
                    let bf0 = Self::f32_to_bf16(f0);
                    let bf1 = Self::f32_to_bf16(f1);
                    result[i] = (bf0 as u32) | ((bf1 as u32) << 16);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let f = f64::from_bits(acc[i]) as f32;
                    result[i] = f.to_bits();
                }
            }
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
    /// Conditional add/subtract: per-element add or subtract based on sel vector.
    ///
    /// d[i] = (sel[i] & 1) ? (s1[i] - s2[i]) : (s1[i] + s2[i])
    fn vector_addsub(
        s1: &[u32; 8],
        s2: &[u32; 8],
        sel: &[u32; 8],
        elem_type: ElementType,
    ) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                for i in 0..8 {
                    if sel[i] & 1 != 0 {
                        result[i] = s1[i].wrapping_sub(s2[i]);
                    } else {
                        result[i] = s1[i].wrapping_add(s2[i]);
                    }
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = (s1[i] & 0xFFFF) as u16;
                    let a_hi = ((s1[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (s2[i] & 0xFFFF) as u16;
                    let b_hi = ((s2[i] >> 16) & 0xFFFF) as u16;
                    let sel_lo = sel[i] & 1;
                    let sel_hi = (sel[i] >> 16) & 1;
                    let r_lo = if sel_lo != 0 { a_lo.wrapping_sub(b_lo) } else { a_lo.wrapping_add(b_lo) };
                    let r_hi = if sel_hi != 0 { a_hi.wrapping_sub(b_hi) } else { a_hi.wrapping_add(b_hi) };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut val = 0u32;
                    for j in 0..4u32 {
                        let a_byte = ((s1[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((s2[i] >> (j * 8)) & 0xFF) as u8;
                        let s_bit = (sel[i] >> (j * 8)) & 1;
                        let r = if s_bit != 0 {
                            a_byte.wrapping_sub(b_byte)
                        } else {
                            a_byte.wrapping_add(b_byte)
                        };
                        val |= (r as u32) << (j * 8);
                    }
                    result[i] = val;
                }
            }
        }

        result
    }

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

    /// Pack: narrow 32-bit lanes to 16-bit (truncation mode).
    ///
    /// Delegates to the `vector_pack` module. Takes two 256-bit vectors
    /// of 32-bit elements and produces one 256-bit vector of 16-bit elements.
    /// Uses truncation mode (no saturation) since the SlotOp does not
    /// currently carry pack saturation information.
    fn vector_pack(a: &[u32; 8], _b: &[u32; 8]) -> [u32; 8] {
        // The vector_pack module operates on a single register at a time,
        // narrowing from bits_i to bits_o. Pack the first source; the second
        // source would go into the upper half for a full 512-bit result,
        // but our register model is 256-bit so we pack just the first.
        vector_pack::pack_vector(a, 32, 16, false, vector_pack::PackMode::Truncate)
    }

    /// Unpack: widen 16-bit lanes to 32-bit (signed, sign-extend).
    ///
    /// Delegates to the `vector_pack` module. Takes a 256-bit vector of
    /// 16-bit elements and produces a 256-bit vector of 32-bit elements
    /// (lower half of the logical result).
    fn vector_unpack_low(src: &[u32; 8]) -> [u32; 8] {
        vector_pack::unpack_vector(src, 16, 32, true)
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
    /// Push a scalar into a vector, shifting existing elements.
    ///
    /// - `is_hi`: push into high end (shift elements towards low indices,
    ///   insert at highest position, discard lowest element)
    /// - `!is_hi`: push into low end (shift elements towards high indices,
    ///   insert at lowest position, discard highest element)
    fn vector_push(src: &[u32; 8], value: u32, is_hi: bool, et: ElementType) -> [u32; 8] {
        // Convert to byte array, shift, insert, convert back.
        let mut bytes = [0u8; 32];
        for (i, word) in src.iter().enumerate() {
            bytes[i * 4] = (*word & 0xFF) as u8;
            bytes[i * 4 + 1] = ((*word >> 8) & 0xFF) as u8;
            bytes[i * 4 + 2] = ((*word >> 16) & 0xFF) as u8;
            bytes[i * 4 + 3] = ((*word >> 24) & 0xFF) as u8;
        }

        let elem_bytes = et.bits() as usize / 8;
        let elem_bytes = if elem_bytes == 0 { 1 } else { elem_bytes }; // min 1 byte for 8-bit

        if is_hi {
            // Shift elements towards low indices (remove lowest element,
            // insert at highest position).
            bytes.copy_within(elem_bytes.., 0);
            // Insert value at the highest position
            let insert_pos = 32 - elem_bytes;
            let val_bytes = value.to_le_bytes();
            for i in 0..elem_bytes.min(4) {
                bytes[insert_pos + i] = val_bytes[i];
            }
        } else {
            // Shift elements towards high indices (remove highest element,
            // insert at lowest position).
            bytes.copy_within(..32 - elem_bytes, elem_bytes);
            // Insert value at position 0
            let val_bytes = value.to_le_bytes();
            for i in 0..elem_bytes.min(4) {
                bytes[i] = val_bytes[i];
            }
        }

        // Convert back to [u32; 8]
        let mut result = [0u32; 8];
        for (i, chunk) in bytes.chunks(4).enumerate() {
            result[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        result
    }

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

    /// Expand a scalar select mask to a per-lane vector mask.
    ///
    /// VSEL uses a scalar register where each bit selects the corresponding
    /// element. For 32-bit mode, bits 0-7 select 8 elements. For 16-bit,
    /// bits 0-15 select 16 elements (2 per u32 lane). For 8-bit, bits 0-31
    /// select 32 elements (4 per u32 lane).
    fn expand_select_mask(sel: u32, elem_type: ElementType) -> [u32; 8] {
        let mut mask = [0u32; 8];
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                // 8 elements, 1 bit each
                for i in 0..8 {
                    mask[i] = if (sel >> i) & 1 != 0 { 1 } else { 0 };
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 elements (2 per u32), 1 bit each
                for i in 0..8 {
                    let lo = if (sel >> (i * 2)) & 1 != 0 { 0xFFFF } else { 0 };
                    let hi = if (sel >> (i * 2 + 1)) & 1 != 0 { 0xFFFF } else { 0 };
                    mask[i] = lo | (hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 elements (4 per u32), 1 bit each
                for i in 0..8 {
                    let mut m = 0u32;
                    for j in 0..4 {
                        if (sel >> (i * 4 + j)) & 1 != 0 {
                            m |= 0xFF << (j * 8);
                        }
                    }
                    mask[i] = m;
                }
            }
        }
        mask
    }

    /// Broadcast a scalar value to all vector lanes.
    /// Extract a single element from a 256-bit vector by element index.
    ///
    /// Returns the element value (zero-extended to u32).
    fn extract_element_by_index(src: &[u32; 8], index: u32, et: ElementType) -> u32 {
        match et {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
                let idx = (index as usize) & 7;
                src[idx]
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 elements of 16-bit each in 256 bits
                let idx = (index as usize) & 15;
                let word = idx / 2;
                let half = idx % 2;
                (src[word] >> (half * 16)) & 0xFFFF
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 elements of 8-bit each in 256 bits
                let idx = (index as usize) & 31;
                let word = idx / 4;
                let byte_in_word = idx % 4;
                (src[word] >> (byte_in_word * 8)) & 0xFF
            }
        }
    }

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
    /// Conditional negate: d[i] = (cmp[i] > 0) ? -s1[i] : s1[i]
    fn vector_bneg_gtz(cmp: &[u32; 8], s1: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    let c = cmp[i] as i32;
                    let v = s1[i] as i32;
                    result[i] = if c > 0 { v.wrapping_neg() as u32 } else { s1[i] };
                }
            }
            ElementType::UInt32 => {
                for i in 0..8 {
                    let c = cmp[i];
                    result[i] = if c > 0 { (s1[i] as i32).wrapping_neg() as u32 } else { s1[i] };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let c = f32::from_bits(cmp[i]);
                    let v = f32::from_bits(s1[i]);
                    result[i] = if c > 0.0 { (-v).to_bits() } else { s1[i] };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let c_lo = (cmp[i] & 0xFFFF) as i16;
                    let c_hi = ((cmp[i] >> 16) & 0xFFFF) as i16;
                    let v_lo = (s1[i] & 0xFFFF) as i16;
                    let v_hi = ((s1[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = if c_lo > 0 { v_lo.wrapping_neg() } else { v_lo };
                    let r_hi = if c_hi > 0 { v_hi.wrapping_neg() } else { v_hi };
                    result[i] = (r_lo as u16 as u32) | ((r_hi as u16 as u32) << 16);
                }
            }
            ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let c_lo = (cmp[i] & 0xFFFF) as u16;
                    let c_hi = ((cmp[i] >> 16) & 0xFFFF) as u16;
                    let v_lo = (s1[i] & 0xFFFF) as i16;
                    let v_hi = ((s1[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = if c_lo > 0 { v_lo.wrapping_neg() } else { v_lo };
                    let r_hi = if c_hi > 0 { v_hi.wrapping_neg() } else { v_hi };
                    result[i] = (r_lo as u16 as u32) | ((r_hi as u16 as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut val = 0u32;
                    for j in 0..4u32 {
                        let c = ((cmp[i] >> (j * 8)) & 0xFF) as i8;
                        let v = ((s1[i] >> (j * 8)) & 0xFF) as i8;
                        let r = if c > 0 { v.wrapping_neg() } else { v };
                        val |= (r as u8 as u32) << (j * 8);
                    }
                    result[i] = val;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut val = 0u32;
                    for j in 0..4u32 {
                        let c = ((cmp[i] >> (j * 8)) & 0xFF) as u8;
                        let v = ((s1[i] >> (j * 8)) & 0xFF) as i8;
                        let r = if c > 0 { v.wrapping_neg() } else { v };
                        val |= (r as u8 as u32) << (j * 8);
                    }
                    result[i] = val;
                }
            }
        }

        result
    }

    /// Conditional negate: d[i] = (cmp[i] < 0) ? -s1[i] : s1[i]
    fn vector_bneg_ltz(cmp: &[u32; 8], s1: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    let c = cmp[i] as i32;
                    let v = s1[i] as i32;
                    result[i] = if c < 0 { v.wrapping_neg() as u32 } else { s1[i] };
                }
            }
            ElementType::UInt32 | ElementType::Float32 => {
                // Unsigned: cmp is never < 0 (treated as unsigned), pass through.
                // Float: check sign bit.
                if matches!(elem_type, ElementType::Float32) {
                    for i in 0..8 {
                        let c = f32::from_bits(cmp[i]);
                        let v = f32::from_bits(s1[i]);
                        result[i] = if c < 0.0 { (-v).to_bits() } else { s1[i] };
                    }
                } else {
                    result = *s1;
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let c_lo = (cmp[i] & 0xFFFF) as i16;
                    let c_hi = ((cmp[i] >> 16) & 0xFFFF) as i16;
                    let v_lo = (s1[i] & 0xFFFF) as i16;
                    let v_hi = ((s1[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = if c_lo < 0 { v_lo.wrapping_neg() } else { v_lo };
                    let r_hi = if c_hi < 0 { v_hi.wrapping_neg() } else { v_hi };
                    result[i] = (r_lo as u16 as u32) | ((r_hi as u16 as u32) << 16);
                }
            }
            ElementType::UInt16 | ElementType::BFloat16 => {
                result = *s1; // Unsigned: never negative
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut val = 0u32;
                    for j in 0..4u32 {
                        let c = ((cmp[i] >> (j * 8)) & 0xFF) as i8;
                        let v = ((s1[i] >> (j * 8)) & 0xFF) as i8;
                        let r = if c < 0 { v.wrapping_neg() } else { v };
                        val |= (r as u8 as u32) << (j * 8);
                    }
                    result[i] = val;
                }
            }
            ElementType::UInt8 => {
                result = *s1; // Unsigned: never negative
            }
        }

        result
    }

    fn vector_neg_ltz(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    let val = src[i] as i32;
                    // Hardware saturates: abs(MIN) = MAX (not overflow).
                    result[i] = val.saturating_abs() as u32;
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
                    // Hardware saturates: abs(-32768) = 32767.
                    let r_lo = lo.saturating_abs() as u16;
                    let r_hi = hi.saturating_abs() as u16;
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
                        // Hardware saturates: abs(-128) = 127.
                        let r_byte = byte.saturating_abs() as u8;
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
    /// Hardware uses two's complement wrapping: -MIN = MIN.
    fn vector_negate(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    result[i] = (src[i] as i32).wrapping_neg() as u32;
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
                    let r_lo = lo.wrapping_neg() as u16;
                    let r_hi = hi.wrapping_neg() as u16;
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
                        let r_byte = byte.wrapping_neg() as u8;
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

    // ========== Wide (512-bit / 1024-bit) Helpers ==========

    /// Read the nth VectorReg source as a full 512-bit value.
    ///
    /// Unlike get_vector_source which reads a single 256-bit register,
    /// this reads a pair of consecutive registers (x-register = two w-registers).
    /// Skips non-VectorReg sources when counting, so idx=0 is the first
    /// VectorReg in sources, idx=1 is the second, etc.
    fn get_wide_vec_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> Vec512 {
        let mut vec_count = 0;
        for src in &op.sources {
            if let Operand::VectorReg(r) = src {
                if vec_count == idx {
                    return ctx.vector.read_wide(*r);
                }
                vec_count += 1;
            }
        }
        [0u32; 16]
    }

    /// Read two wide vector sources.
    fn get_two_wide_vec_sources(
        op: &SlotOp,
        ctx: &ExecutionContext,
    ) -> (Vec512, Vec512) {
        let a = Self::get_wide_vec_source(op, ctx, 0);
        let b = Self::get_wide_vec_source(op, ctx, 1);
        (a, b)
    }

    /// Write a 512-bit result to the vector destination.
    fn write_wide_vec_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: Vec512) {
        if let Some(Operand::VectorReg(r)) = &op.dest {
            ctx.vector.write_wide(*r, value);
        } else {
            log::error!(
                "[VECTOR_WIDE] write_wide_vec_dest: expected VectorReg dest, got {:?}",
                op.dest
            );
        }
    }

    /// Read the accumulator destination register index and its current 1024-bit value.
    fn get_wide_acc_dest_value(op: &SlotOp, ctx: &ExecutionContext) -> (u8, Acc1024) {
        let reg = Self::get_acc_dest(op);
        (reg, ctx.accumulator.read_wide(reg))
    }

    /// Write a 1024-bit result to the accumulator destination.
    fn write_wide_acc_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: Acc1024) {
        let reg = Self::get_acc_dest(op);
        ctx.accumulator.write_wide(reg, value);
    }

    /// Read an AccumReg source as a 1024-bit cm-register.
    fn get_wide_acc_source(op: &SlotOp, ctx: &ExecutionContext) -> (u8, Acc1024) {
        let reg = Self::get_acc_source(op);
        (reg, ctx.accumulator.read_wide(reg))
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

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Int32)
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

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Int16)
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

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Sub)
            .as_vector(ElementType::Int32)
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

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Mul)
            .as_vector(ElementType::Int32)
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
        let op = SlotOp::from_semantic(SlotIndex::Accumulator, SemanticOp::Mac)
            .as_vector(ElementType::Int32)
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
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Min)
            .as_vector(ElementType::UInt32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [5, 5, 15, 15, 25, 25, 35, 35]);

        // Max
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Max)
            .as_vector(ElementType::UInt32)
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

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Shuffle)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::VectorReg(0));
        op.shuffle_pattern = ShufflePattern::Reverse;

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_vector_shuffle_broadcast() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [10, 20, 30, 40, 50, 60, 70, 80]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Shuffle)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::VectorReg(0));
        op.shuffle_pattern = ShufflePattern::Broadcast(2);

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [30, 30, 30, 30, 30, 30, 30, 30]);
    }

    #[test]
    fn test_vector_cmp() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [1, 0, 3, 0, 5, 0, 7, 0]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Cmp)
            .as_vector(ElementType::Int32)
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

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Float32)
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

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Mul)
            .as_vector(ElementType::Float32)
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
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Min)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);
        assert_eq!(f32::from_bits(result[0]), 0.5);  // min(1.0, 0.5)
        assert_eq!(f32::from_bits(result[1]), -2.0); // min(-2.0, -1.0)
        assert_eq!(f32::from_bits(result[3]), -5.0); // min(-4.0, -5.0)

        // Max
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Max)
            .as_vector(ElementType::Float32)
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

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::BFloat16)
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

        let op = SlotOp::from_semantic(SlotIndex::Accumulator, SemanticOp::MatMul)
            .as_vector(ElementType::Int32)
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
        // UPS+SRS round-trip: 16 x i16 input, UPS processes lower 8 lanes,
        // SRS converts back to i16. Only the lower 4 output words should
        // match the lower 4 input words (8 x i16 -> 8 acc lanes -> 8 x i16).
        let src = [0x0002_0001u32, 0x0004_0003, 0x0006_0005, 0x0008_0007,
                    0x000A_0009, 0x000C_000B, 0x000E_000D, 0x0010_000F];
        // UPS with shift=0: sign-extend 8 i16 values to i32 in accumulator
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
        // 8 acc lanes produce 8 x i16 packed into 4 u32 words.
        // UPS processes lanes 0-7 (first 8 of 16 input lanes).
        // Lanes 0-7 are: 1,2, 3,4, 5,6, 7,8
        assert_eq!(result[0], 0x0002_0001, "lanes 0-1");
        assert_eq!(result[1], 0x0004_0003, "lanes 2-3");
        assert_eq!(result[2], 0x0006_0005, "lanes 4-5");
        assert_eq!(result[3], 0x0008_0007, "lanes 6-7");
        // Upper words are zero (only 8 lanes processed)
        assert_eq!(result[4], 0, "upper should be zero");
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
}
