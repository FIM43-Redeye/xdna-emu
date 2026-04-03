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
use crate::interpreter::state::{ExecutionContext, SrsConfig};
use crate::interpreter::state::{Vec512, Acc1024};
use crate::tablegen::SemanticOp;

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

        // Skip fused ops that have memory operands -- MemoryUnit handles these.
        // Fused instructions (vlda.ups, vst.srs, vst.pack, vlda.conv, vst.conv)
        // combine memory access with compute and need tile access.
        //
        // Standalone SRS/Pack/UPS (e.g., VSRS in the ST slot, VUPS in the LDA
        // slot) are also on memory slots but have NO memory operand -- they
        // operate purely on registers and must be handled here.
        if op.slot.is_memory() && matches!(semantic,
            SemanticOp::Ups | SemanticOp::Srs | SemanticOp::Pack
            | SemanticOp::Unpack | SemanticOp::Convert)
            && op.sources.iter().any(|s| matches!(s, Operand::Memory { .. }))
        {
            return false;
        }

        let et = op.element_type.unwrap_or(ElementType::Int32);

        log::trace!("[VECTOR_ALU] Checking semantic={:?} element_type={:?} dest={:?}",
            semantic, op.element_type, op.dest);

        // Determine if this needs full-width (1024-bit) processing:
        // 1. is_wide_vector: instruction has Vector512 operands (x-registers)
        // 2. AccumReg-only ops (VADD, VSUB, VNEG) that use cm-class (Full)
        //    accumulators -- these need wide read/write.
        // 3. Wide accumulator source: SRS/UPS/Convert with cm-class source
        //    (AccumWidth::Full) need the wide path even though they write to
        //    a vector register, not an accumulator.
        let has_wide_acc_source = matches!(op.accum_width,
            Some(crate::tablegen::decoder_ffi::AccumWidth::Full));
        let is_accum_only = matches!(semantic,
            SemanticOp::Accumulate | SemanticOp::AccumSub
            | SemanticOp::AccumNegAdd | SemanticOp::AccumNegSub
            | SemanticOp::Neg | SemanticOp::NegAdd) && {
            let has_acc_source = op.sources.iter()
                .any(|s| matches!(s, Operand::AccumReg(_)));
            let has_vec_source = op.sources.iter()
                .any(|s| matches!(s, Operand::VectorReg(_)));
            has_acc_source && !has_vec_source
        };

        if op.is_wide_vector || is_accum_only || has_wide_acc_source {
            Self::execute_wide(op, ctx, semantic, et)
        } else {
            Self::execute_half(op, ctx, semantic, et)
        }
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
                // 2. VADDSUB: conditional add/subtract (2 vector + 1 scalar source)
                //    vaddsub.8 $d, $s1, $s2, $sel
                //    d[i] = (sel[i] & 1) ? s1[i] - s2[i] : s1[i] + s2[i]
                //    sel is a SCALAR register (bitmask), not a vector.
                let has_scalar_sel = op.sources.iter()
                    .any(|s| matches!(s, Operand::ScalarReg(_)))
                    && op.sources.iter()
                        .filter(|s| matches!(s, Operand::VectorReg(_)))
                        .count() == 2;

                if has_scalar_sel {
                    let s1 = Self::get_vector_source(op, ctx, 0);
                    let s2 = Self::get_vector_source(op, ctx, 1);
                    // Read the scalar selector bitmask
                    let sel_scalar = Self::get_nth_scalar_source(op, ctx, 0);
                    let sel = Self::expand_select_mask(sel_scalar, et);
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

            SemanticOp::Mac | SemanticOp::MatMul | SemanticOp::MatMulSub
            | SemanticOp::NegMul | SemanticOp::NegMatMul
            | SemanticOp::AddMac | SemanticOp::SubMac => {
                return super::vector_matmul::execute_matmul(op, ctx);
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
                // Vector negate: vneg $d, $s1 -- per-element negate.
                // AccumReg-only Neg ops are routed to execute_wide which
                // calls execute_acc_negate, so this arm only sees vector ops.
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_negate(&src, et);
                Self::write_vector_dest(op, ctx, result);
                true
            }

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
                // cmp = packed bitmask of the same comparison
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&result, et));
                true
            }

            SemanticOp::SetLt => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_compare_lt(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                // cmp = packed bitmask of the same comparison
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&result, et));
                true
            }

            SemanticOp::SetEq => {
                // Vector equal-to-zero (VEQZ): produces a comparison bitmask
                // written to scalar cmp register(s), not a vector result.
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_compare_eqz(&src, et);
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&result, et));
                true
            }

            SemanticOp::MaxLt => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_max(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                // cmp = per-element (a < b)
                let cmp = Self::vector_compare_lt(&a, &b, et);
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
                true
            }

            SemanticOp::MinGe => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_min(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                // cmp = per-element (a >= b)
                let cmp = Self::vector_compare_ge(&a, &b, et);
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
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
                // cmp = per-element (s > 0)
                let zero = [0u32; 8];
                let cmp = Self::vector_compare_lt(&zero, &src, et); // 0 < s  =>  s > 0
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
                true
            }

            SemanticOp::NegGtz => {
                let vec_source_count = op.sources.iter()
                    .filter(|s| matches!(s, Operand::VectorReg(_)))
                    .count();

                let src = if vec_source_count >= 2 {
                    let cond = Self::get_vector_source(op, ctx, 0);
                    let s1 = Self::get_vector_source(op, ctx, 1);
                    let result = Self::vector_bneg_gtz(&cond, &s1, et);
                    Self::write_vector_dest(op, ctx, result);
                    s1
                } else {
                    let src = Self::get_vector_source(op, ctx, 0);
                    let result = Self::vector_neg_gtz(&src, et);
                    Self::write_vector_dest(op, ctx, result);
                    src
                };
                // cmp = per-element (s > 0)
                let zero = [0u32; 8];
                let cmp = Self::vector_compare_lt(&zero, &src, et);
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
                true
            }

            SemanticOp::NegLtz => {
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_neg_ltz(&src, et);
                Self::write_vector_dest(op, ctx, result);
                // cmp = per-element (s < 0)
                let zero = [0u32; 8];
                let cmp = Self::vector_compare_lt(&src, &zero, et);
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
                true
            }

            SemanticOp::Accumulate | SemanticOp::AccumSub
            | SemanticOp::AccumNegAdd | SemanticOp::AccumNegSub
            | SemanticOp::NegAdd => {
                // VADD/VSUB/VNEGADD/VNEGSUB: add/subtract two accumulator
                // registers. Format: `vadd $dst, $acc1, $acc2, $config`.
                //
                // Check if sources contain AccumReg (acc-to-acc operation)
                // or VectorReg (legacy vector accumulate / regular NegAdd).
                let has_acc_source = op.sources.iter().any(|s| matches!(s, Operand::AccumReg(_)));

                if has_acc_source {
                    Self::execute_acc_add_sub(op, ctx);
                } else if matches!(semantic, SemanticOp::NegAdd) {
                    // Regular vector negate-add (no accumulator involvement).
                    let (a, b) = Self::get_two_vector_sources(op, ctx);
                    let result = Self::vector_neg_add(&a, &b, et);
                    Self::write_vector_dest(op, ctx, result);
                } else {
                    // Legacy: accumulate a vector source into an accumulator.
                    let src = Self::get_vector_source(op, ctx, 0);
                    let acc_reg = Self::get_acc_dest(op);
                    Self::vector_accumulate(ctx, acc_reg, &src, et);
                }
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
                // cmp = per-element (a < b)
                let cmp = Self::vector_compare_lt(&a, &b, et);
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
                true
            }

            SemanticOp::SubGe => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_sub_ge(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                // cmp = per-element (a >= b)
                let cmp = Self::vector_compare_ge(&a, &b, et);
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
                true
            }

            SemanticOp::MaxDiffLt => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_maxdiff_lt(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
                // cmp = per-element (a < b)
                let cmp = Self::vector_compare_lt(&a, &b, et);
                Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
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
            other => {
                panic!(
                    "condense_comparison_mask: unhandled element type {:?}",
                    other,
                );
            }
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
    fn execute_acc_add_sub(op: &SlotOp, ctx: &mut ExecutionContext) {
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
                use super::vector_float::{fp32_flush_to_zero, aie2_fp32_add};
                for i in 0..16 {
                    let mut a1_lo = if zero_acc1 { 0u32 } else { fp32_flush_to_zero(a1[i] as u32) };
                    let mut a1_hi = if zero_acc1 { 0u32 } else { fp32_flush_to_zero((a1[i] >> 32) as u32) };
                    let mut a2_lo = fp32_flush_to_zero(a2[i] as u32);
                    let mut a2_hi = fp32_flush_to_zero((a2[i] >> 32) as u32);

                    // Negate by flipping sign bit (works for zero, normal, inf, NaN).
                    if negate_acc1 { a1_lo ^= 0x8000_0000; a1_hi ^= 0x8000_0000; }
                    if negate_acc2 { a2_lo ^= 0x8000_0000; a2_hi ^= 0x8000_0000; }

                    let r_lo = aie2_fp32_add(a1_lo, a2_lo);
                    let r_hi = aie2_fp32_add(a1_hi, a2_hi);
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
                use super::vector_float::{fp32_flush_to_zero, aie2_fp32_add};
                for i in 0..8 {
                    let mut a1_lo = if zero_acc1 { 0u32 } else { fp32_flush_to_zero(a1[i] as u32) };
                    let mut a1_hi = if zero_acc1 { 0u32 } else { fp32_flush_to_zero((a1[i] >> 32) as u32) };
                    let mut a2_lo = fp32_flush_to_zero(a2[i] as u32);
                    let mut a2_hi = fp32_flush_to_zero((a2[i] >> 32) as u32);

                    if negate_acc1 { a1_lo ^= 0x8000_0000; a1_hi ^= 0x8000_0000; }
                    if negate_acc2 { a2_lo ^= 0x8000_0000; a2_hi ^= 0x8000_0000; }

                    let r_lo = aie2_fp32_add(a1_lo, a2_lo);
                    let r_hi = aie2_fp32_add(a1_hi, a2_hi);
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
    fn execute_acc_negate(op: &SlotOp, ctx: &mut ExecutionContext) {
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
                for i in 0..8 {
                    let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
                    let f = Self::bf16_to_f32(bf16);
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
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

    /// Vector minimum by element type.
    fn vector_min(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = std::cmp::min(a[i] as i32, b[i] as i32) as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
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
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = std::cmp::max(a[i] as i32, b[i] as i32) as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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

    /// Get 64-bit scalar source from a register pair (rN, rN+1).
    ///
    /// For 64-bit VPUSH, the scalar value spans two adjacent registers:
    /// rN holds the low 32 bits, rN+1 holds the high 32 bits.
    fn get_scalar_source_64(op: &SlotOp, ctx: &ExecutionContext) -> u64 {
        for src in &op.sources {
            if let Operand::ScalarReg(r) = src {
                let lo = ctx.scalar.read(*r) as u64;
                let hi = ctx.scalar.read(r + 1) as u64;
                return lo | (hi << 32);
            }
        }
        0
    }

    /// Get the Nth scalar source operand (0-indexed among scalars/immediates).
    ///
    /// For instructions with multiple scalar operands (e.g., VINSERT has both
    /// idx and s0), type-scanning heuristics pick the wrong one. This function
    /// returns the Nth scalar in source order.
    fn get_nth_scalar_source(op: &SlotOp, ctx: &ExecutionContext, n: usize) -> u32 {
        let mut count = 0;
        for src in &op.sources {
            match src {
                Operand::ScalarReg(r) => {
                    if count == n { return ctx.scalar.read(*r); }
                    count += 1;
                }
                Operand::Immediate(imm) => {
                    if count == n { return *imm as u32; }
                    count += 1;
                }
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

    /// Write a 64-bit scalar result to destination register pair (rN, rN+1).
    /// Used for 8-bit comparisons where the bitmask exceeds 32 bits.
    fn write_scalar_dest_wide(op: &SlotOp, ctx: &mut ExecutionContext, value: u64) {
        if let Some(Operand::ScalarReg(r)) = &op.dest {
            ctx.scalar.write(*r, value as u32);
            ctx.scalar.write(*r + 1, (value >> 32) as u32);
        }
    }

    /// Write comparison flags to the secondary destination register (cmp) of
    /// dual-result instructions (VSUB_LT, VABS_GTZ, VNEG_GTZ, etc.).
    ///
    /// The cmp register receives a per-element bitmask: bit i is set when the
    /// comparison is true for element i.
    fn write_cmp_dest(op: &SlotOp, ctx: &mut ExecutionContext, flags: u32) {
        if let Some(Operand::ScalarReg(r)) = op.extra_dests.first() {
            ctx.scalar.write(*r, flags);
        }
    }

    /// Write a 64-bit comparison bitmask to the cmp register pair (for 8-bit
    /// element comparisons that need 64 bits: 32 elements per half * 2 halves).
    fn write_cmp_dest_wide(op: &SlotOp, ctx: &mut ExecutionContext, flags: u64) {
        if let Some(Operand::ScalarReg(r)) = op.extra_dests.first() {
            // Write low 32 bits to rN, high 32 bits to rN+1.
            ctx.scalar.write(*r, flags as u32);
            ctx.scalar.write(*r + 1, (flags >> 32) as u32);
        }
    }

    /// Compute per-element comparison flags from a vector comparison result.
    ///
    /// Takes the expanded mask (all-ones or all-zeros per element) and packs
    /// it into a scalar bitmask: bit i = 1 if element i was non-zero.
    fn pack_comparison_flags(mask: &[u32; 8], elem_type: ElementType) -> u32 {
        let mut flags: u32 = 0;
        match elem_type {
            ElementType::Int32 | ElementType::UInt32
            | ElementType::Int64 | ElementType::UInt64
            | ElementType::Float32 => {
                // 8 elements of 32-bit
                for i in 0..8 {
                    if mask[i] != 0 { flags |= 1 << i; }
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 elements of 16-bit
                for i in 0..8 {
                    let lo = mask[i] & 0xFFFF;
                    let hi = (mask[i] >> 16) & 0xFFFF;
                    let bit = i * 2;
                    if lo != 0 { flags |= 1 << bit; }
                    if hi != 0 { flags |= 1 << (bit + 1); }
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 elements of 8-bit
                for i in 0..8 {
                    for j in 0..4 {
                        let byte = (mask[i] >> (j * 8)) & 0xFF;
                        let bit = i * 4 + j;
                        if byte != 0 { flags |= 1 << bit; }
                    }
                }
            }
        }
        flags
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
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
    /// Absolute value.  The GTZ suffix describes comparison flags written to
    /// the `cmp` register, NOT a condition on the primary result.  The primary
    /// output is always abs(s).
    fn vector_abs_gtz(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    let val = src[i] as i32;
                    result[i] = val.wrapping_abs() as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                // Unsigned: abs is identity.
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
                    let r_lo = lo.wrapping_abs() as u16;
                    let r_hi = hi.wrapping_abs() as u16;
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
                        let r_byte = byte.wrapping_abs() as u8;
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

    /// Unconditional negate.  The GTZ suffix describes comparison flags written
    /// to the `cmp` register, NOT a condition on the primary result.  The
    /// primary output is always -s.
    fn vector_neg_gtz(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    let val = src[i] as i32;
                    result[i] = val.wrapping_neg() as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = 0u32.wrapping_sub(src[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    // Flip sign bit.
                    result[i] = src[i] ^ 0x8000_0000;
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
                    // Flip sign bits for bf16 elements.
                    result[i] = src[i] ^ 0x8000_8000;
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

    /// Negate if less than zero: dst[i] = (src[i] < 0) ? -src[i] : src[i]
    /// This is essentially abs() for signed types.
    /// Conditional negate: d[i] = (cmp[i] > 0) ? -s1[i] : s1[i]
    fn vector_bneg_gtz(cmp: &[u32; 8], s1: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    let c = cmp[i] as i32;
                    let v = s1[i] as i32;
                    result[i] = if c > 0 { v.wrapping_neg() as u32 } else { s1[i] };
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
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
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    let c = cmp[i] as i32;
                    let v = s1[i] as i32;
                    result[i] = if c < 0 { v.wrapping_neg() as u32 } else { s1[i] };
                }
            }
            ElementType::UInt32 | ElementType::UInt64 | ElementType::Float32 => {
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

    /// Bitwise NOT (VBNEG_LTZ).  The "B" prefix means "bitwise", matching
    /// VBAND/VBOR.  The LTZ suffix describes comparison flags written to
    /// the `cmp` register.  The primary output is always ~s.
    fn vector_neg_ltz(src: &[u32; 8], _elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = !src[i];
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
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
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = (src[i] as i32).wrapping_neg() as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
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
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = ((-(a[i] as i32)) as i64 + (b[i] as i32) as i64) as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
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

    // ========== Vector Comparison Operation Implementations ==========

    /// Vector compare greater-or-equal: dst[i] = (a[i] >= b[i]) ? ~0 : 0
    fn vector_compare_ge(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = if (a[i] as i32) >= (b[i] as i32) { !0 } else { 0 };
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
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
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = if (a[i] as i32) < (b[i] as i32) { !0 } else { 0 };
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
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
    /// Unconditional subtraction: d = a - b.  The LT suffix describes
    /// comparison flags written to the `cmp` register, NOT a condition on
    /// the subtraction.
    fn vector_sub_lt(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        Self::vector_sub_unconditional(a, b, elem_type)
    }

    /// Unconditional subtraction: d = a - b.  The GE suffix describes
    /// comparison flags written to the `cmp` register, NOT a condition on
    /// the subtraction.
    fn vector_sub_ge(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        Self::vector_sub_unconditional(a, b, elem_type)
    }

    /// Shared unconditional subtraction for vsub_lt / vsub_ge.
    fn vector_sub_unconditional(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32
            | ElementType::Int64 | ElementType::UInt64 => {
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

    /// Maximum difference if less-than: dst[i] = max(a[i] - b[i], 0) for unsigned,
    /// or clamped difference for signed types.
    fn vector_maxdiff_lt(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    let va = a[i] as i32;
                    let vb = b[i] as i32;
                    let diff = va.saturating_sub(vb);
                    result[i] = diff.max(0) as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
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

    // ========== Wide dispatch bridges ==========

    /// Bridge: apply a narrow element-wise function to a wide vector.
    ///
    /// Splits Vec512 into two [u32; 8] halves, applies the function to
    /// each half independently, and concatenates the results. Works for
    /// any operation where each output element depends only on
    /// corresponding input elements.
    fn wide_element_wise_unary(
        a: &Vec512,
        et: ElementType,
        op_fn: fn(&[u32; 8], ElementType) -> [u32; 8],
    ) -> Vec512 {
        let a_lo: [u32; 8] = a[..8].try_into().unwrap();
        let a_hi: [u32; 8] = a[8..].try_into().unwrap();
        let r_lo = op_fn(&a_lo, et);
        let r_hi = op_fn(&a_hi, et);
        let mut result = [0u32; 16];
        result[..8].copy_from_slice(&r_lo);
        result[8..].copy_from_slice(&r_hi);
        result
    }

    /// Bridge: apply a narrow two-input element-wise function to wide vectors.
    fn wide_element_wise_binary(
        a: &Vec512,
        b: &Vec512,
        et: ElementType,
        op_fn: fn(&[u32; 8], &[u32; 8], ElementType) -> [u32; 8],
    ) -> Vec512 {
        let a_lo: [u32; 8] = a[..8].try_into().unwrap();
        let a_hi: [u32; 8] = a[8..].try_into().unwrap();
        let b_lo: [u32; 8] = b[..8].try_into().unwrap();
        let b_hi: [u32; 8] = b[8..].try_into().unwrap();
        let r_lo = op_fn(&a_lo, &b_lo, et);
        let r_hi = op_fn(&a_hi, &b_hi, et);
        let mut result = [0u32; 16];
        result[..8].copy_from_slice(&r_lo);
        result[8..].copy_from_slice(&r_hi);
        result
    }

    /// Fallback: split a wide op into two narrow halves.
    ///
    /// Used for SemanticOps not yet ported to execute_wide.
    /// Preserves the old clone+increment behavior.
    fn execute_wide_fallback(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        et: ElementType,
    ) -> bool {
        log::trace!(
            "[VECTOR_WIDE] fallback to half-split for {:?}",
            semantic
        );
        let handled = Self::execute_half(op, ctx, semantic, et);
        if handled {
            let mut hi_op = op.clone();
            Self::increment_vector_regs(&mut hi_op);
            Self::execute_half(&hi_op, ctx, semantic, et);
        }
        handled
    }

    /// Execute a 512-bit wide vector operation.
    ///
    /// Unlike execute_half which processes 256-bit chunks independently,
    /// this reads full 512-bit inputs and writes full 512-bit outputs.
    /// Element-wise ops use the bridge to reuse narrow math. Cross-half
    /// ops have dedicated implementations (added in later tasks).
    fn execute_wide(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        et: ElementType,
    ) -> bool {
        match semantic {
            // ========== Element-wise arithmetic (bridge) ==========
            SemanticOp::Add => {
                // Check for VADDSUB: 2 vector sources + 1 scalar selector.
                let has_scalar_sel = op.sources.iter()
                    .any(|s| matches!(s, Operand::ScalarReg(_)))
                    && op.sources.iter()
                        .filter(|s| matches!(s, Operand::VectorReg(_)))
                        .count() == 2;
                if has_scalar_sel {
                    // VADDSUB: read full 512-bit sources and scalar selector.
                    let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                    let sel_scalar = Self::get_nth_scalar_source(op, ctx, 0);
                    // Split wide vectors into lo/hi halves for per-half processing.
                    let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                    let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                    let b_lo: [u32; 8] = b[..8].try_into().unwrap();
                    let b_hi: [u32; 8] = b[8..].try_into().unwrap();
                    // Expand selector bits: lo half uses lower bits, hi uses upper.
                    // For 8-bit elements (32 per half), the selector is a 64-bit
                    // register pair (eL class: l_n = {r(16+2n), r(16+2n+1)}).
                    // LLVM decodes this as a single ScalarReg for the even register;
                    // we read reg+1 for the upper 32 bits.
                    let elems_per_half = 256 / et.bits() as u32;
                    let sel_lo = Self::expand_select_mask(sel_scalar, et);
                    let sel_hi_bits = if elems_per_half >= 32 {
                        // 8-bit: 64-bit selector, upper 32 bits in the next register
                        let sel_reg = op.sources.iter().find_map(|s| {
                            if let Operand::ScalarReg(r) = s { Some(*r) } else { None }
                        }).unwrap_or(0);
                        ctx.scalar.read(sel_reg + 1)
                    } else {
                        sel_scalar >> elems_per_half
                    };
                    let sel_hi = Self::expand_select_mask(sel_hi_bits, et);
                    let lo = Self::vector_addsub(&a_lo, &b_lo, &sel_lo, et);
                    let hi = Self::vector_addsub(&a_hi, &b_hi, &sel_hi, et);
                    let mut result = [0u32; 16];
                    result[..8].copy_from_slice(&lo);
                    result[8..].copy_from_slice(&hi);
                    Self::write_wide_vec_dest(op, ctx, result);
                    return true;
                }
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_add);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Sub => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_sub);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Mul => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_mul);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Min => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_min);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Max => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_max);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::VectorSelect => {
                // VSEL: 512-bit select with scalar bitmask.
                // Must split the selector across lo/hi halves.
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let sel_scalar = op.sources.iter().find_map(|s| match s {
                    Operand::ScalarReg(r) => Some(ctx.scalar.read(*r)),
                    _ => None,
                }).unwrap_or(0);
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let b_lo: [u32; 8] = b[..8].try_into().unwrap();
                let b_hi: [u32; 8] = b[8..].try_into().unwrap();
                let elems_per_half = 256 / et.bits() as u32;
                let sel_lo = Self::expand_select_mask(sel_scalar, et);
                let sel_hi_bits = if elems_per_half >= 32 {
                    // 8-bit elements: 64-bit mask from register pair.
                    // The operand decodes as a single ScalarReg (low reg
                    // of the pair). Read r+1 for the high 32 bits.
                    op.sources.iter().find_map(|s| match s {
                        Operand::ScalarReg(r) => Some(ctx.scalar.read(r + 1)),
                        _ => None,
                    }).unwrap_or(0)
                } else {
                    sel_scalar >> elems_per_half
                };
                let sel_hi = Self::expand_select_mask(sel_hi_bits, et);
                // Hardware: bit=0 -> s1, bit=1 -> s2
                let lo = Self::vector_select(&sel_lo, &b_lo, &a_lo, et);
                let hi = Self::vector_select(&sel_hi, &b_hi, &a_hi, et);
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&lo);
                result[8..].copy_from_slice(&hi);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            SemanticOp::Neg => {
                // Two sub-cases:
                // 1. AccumReg source: VNEG on cm-class accumulator (1024-bit)
                // 2. VectorReg source: element-wise vector negate (512-bit)
                let has_acc_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::AccumReg(_)));
                if has_acc_source {
                    Self::execute_acc_negate(op, ctx);
                } else {
                    let a = Self::get_wide_vec_source(op, ctx, 0);
                    let result = Self::wide_element_wise_unary(&a, et, Self::vector_negate);
                    Self::write_wide_vec_dest(op, ctx, result);
                }
                true
            }

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

            // ========== Comparison (bridge) ==========
            SemanticOp::Cmp => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_cmp_eq);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::SetGe => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_compare_ge);
                if matches!(op.dest, Some(Operand::ScalarReg(_))) {
                    // VGE: condense per-lane mask into scalar bitmask
                    let lo: [u32; 8] = result[..8].try_into().unwrap();
                    let hi: [u32; 8] = result[8..].try_into().unwrap();
                    let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                    let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                    let lanes_per_half = 256 / et.bits() as u32;
                    let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
                    // For 8-bit elements, the mask needs 64 bits (register pair).
                    if lanes_per_half >= 32 {
                        // Write to primary dest (VGE/VLT: cmp is the main dest).
                        Self::write_scalar_dest_wide(op, ctx, full_mask);
                        // Also write to extra_dests (dual-result ops like VSUB_LT).
                        Self::write_cmp_dest_wide(op, ctx, full_mask);
                    } else {
                        Self::write_scalar_dest(op, ctx, full_mask as u32);
                        // Also write to extra_dests (cmp register) for 16/32-bit.
                        Self::write_cmp_dest(op, ctx, full_mask as u32);
                    }
                } else {
                    Self::write_wide_vec_dest(op, ctx, result);
                    let lo: [u32; 8] = result[..8].try_into().unwrap();
                    let hi: [u32; 8] = result[8..].try_into().unwrap();
                    let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                    let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                    let lanes_per_half = 256 / et.bits() as u32;
                    let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
                    if lanes_per_half >= 32 {
                        Self::write_cmp_dest_wide(op, ctx, full_mask);
                    } else {
                        Self::write_cmp_dest(op, ctx, full_mask as u32);
                    }
                }
                true
            }
            SemanticOp::SetLt => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_compare_lt);
                if matches!(op.dest, Some(Operand::ScalarReg(_))) {
                    let lo: [u32; 8] = result[..8].try_into().unwrap();
                    let hi: [u32; 8] = result[8..].try_into().unwrap();
                    let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                    let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                    let lanes_per_half = 256 / et.bits() as u32;
                    let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
                    if lanes_per_half >= 32 {
                        Self::write_scalar_dest_wide(op, ctx, full_mask);
                        Self::write_cmp_dest_wide(op, ctx, full_mask);
                    } else {
                        Self::write_scalar_dest(op, ctx, full_mask as u32);
                        Self::write_cmp_dest(op, ctx, full_mask as u32);
                    }
                } else {
                    Self::write_wide_vec_dest(op, ctx, result);
                    let lo: [u32; 8] = result[..8].try_into().unwrap();
                    let hi: [u32; 8] = result[8..].try_into().unwrap();
                    let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                    let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                    let lanes_per_half = 256 / et.bits() as u32;
                    let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
                    if lanes_per_half >= 32 {
                        Self::write_cmp_dest_wide(op, ctx, full_mask);
                    } else {
                        Self::write_cmp_dest(op, ctx, full_mask as u32);
                    }
                }
                true
            }

            // ========== Matrix Multiply (config-driven) ==========
            SemanticOp::Mac | SemanticOp::MatMul | SemanticOp::MatMulSub
            | SemanticOp::NegMul | SemanticOp::NegMatMul
            | SemanticOp::AddMac | SemanticOp::SubMac => {
                return super::vector_matmul::execute_matmul(op, ctx);
            }

            // ========== Accumulator ops ==========
            SemanticOp::Accumulate | SemanticOp::AccumSub
            | SemanticOp::AccumNegAdd | SemanticOp::AccumNegSub
            | SemanticOp::NegAdd => {
                let has_acc_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::AccumReg(_)));
                if has_acc_source {
                    // VADD/VSUB/VNEGADD/VNEGSUB: accumulator-to-accumulator
                    Self::execute_acc_add_sub(op, ctx);
                } else if matches!(semantic, SemanticOp::NegAdd) {
                    // Regular vector negate-add (no accumulator involvement).
                    let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                    let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_neg_add);
                    Self::write_wide_vec_dest(op, ctx, result);
                } else {
                    // Legacy: vector-into-accumulator, use fallback
                    return Self::execute_wide_fallback(op, ctx, semantic, et);
                }
                true
            }

            // ========== SRS/UPS (element-wise, split into halves) ==========

            SemanticOp::Srs => {
                // Wide SRS: read Acc1024, SRS each half, write Vec512.
                let acc_reg = Self::get_acc_source(op);
                let shift = Self::get_shift_amount(op, ctx);
                let from = op.from_type.unwrap_or(ElementType::Int64);
                let acc_wide = ctx.accumulator.read_wide(acc_reg);
                let acc_lo: [u64; 8] = acc_wide[..8].try_into().unwrap();
                let acc_hi: [u64; 8] = acc_wide[8..].try_into().unwrap();

                let result_lo = Self::vector_srs_from_acc(&acc_lo, shift, from, et, &ctx.srs_config);
                let result_hi = Self::vector_srs_from_acc(&acc_hi, shift, from, et, &ctx.srs_config);

                // Pack the two halves contiguously. For reduction SRS (e.g.,
                // s8 from s32, s16 from s64), each half only fills a fraction
                // of its 8-word output. Compute how many valid u32 words each
                // half produces based on lane count and output element width.
                let from_bits = from.bits() as usize;
                let lanes_per_half = if from_bits <= 32 { 16 } else { 8 }; // Acc32 vs Acc64
                let to_bits = et.bits() as usize;
                let words_per_half = (lanes_per_half * to_bits + 31) / 32;

                let mut result = [0u32; 16];
                let n = words_per_half.min(8);
                result[..n].copy_from_slice(&result_lo[..n]);
                result[n..n + n].copy_from_slice(&result_hi[..n]);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            SemanticOp::Ups => {
                let shift = Self::get_shift_amount(op, ctx);
                let from = op.from_type.unwrap_or(ElementType::Int16);

                let acc_wide = if !op.is_wide_vector {
                    // w2c path: narrow source (256-bit wl) fills full 1024-bit
                    // cm register via 4:1 upshift (e.g., 32xi8->32xi32 or
                    // 16xi16->16xi64).
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
                true
            }

            // ========== Copy / Clear ==========
            SemanticOp::Copy => {
                // Handle both vector-to-vector and accum-to-accum moves.
                let has_acc_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::AccumReg(_)));
                if has_acc_source {
                    // Accumulator move: vmov cm_dst, cm_src
                    let src_reg = op.sources.iter().find_map(|s| match s {
                        Operand::AccumReg(r) => Some(*r),
                        _ => None,
                    }).unwrap_or(0);
                    let data = ctx.accumulator.read_wide(src_reg);
                    Self::write_wide_acc_dest(op, ctx, data);
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
                    Self::write_wide_acc_dest(op, ctx, [0u64; 16]);
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

            // ========== Shift operations (bridge) ==========
            SemanticOp::Shl => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_shift_left);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Srl => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_shift_right_logical);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Sra => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_shift_right_arith);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            // ========== Comparison variants (bridge) ==========
            // MaxLt and MinGe are synonyms for Max/Min -- the conditional suffix
            // describes the input ordering contract, not a distinct operation.
            SemanticOp::MaxLt => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_max);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::MinGe => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_min);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            // ========== Conditional arithmetic (bridge) ==========
            SemanticOp::SubLt => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_sub_lt);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::SubGe => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_sub_ge);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::MaxDiffLt => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_maxdiff_lt);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            // NegAdd and Sub are now handled in the accumulator dispatch above
            // (with has_acc_source check to distinguish vector vs accumulator ops).

            // ========== Conditional unary (bridge) ==========
            SemanticOp::NegGtz => {
                // Two-vector-source variant (VNEG_GTZ with select operand) uses fallback.
                let vec_source_count = op.sources.iter()
                    .filter(|s| matches!(s, Operand::VectorReg(_)))
                    .count();
                if vec_source_count >= 2 {
                    return Self::execute_wide_fallback(op, ctx, semantic, et);
                }
                let a = Self::get_wide_vec_source(op, ctx, 0);
                let result = Self::wide_element_wise_unary(&a, et, Self::vector_neg_gtz);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::NegLtz => {
                let a = Self::get_wide_vec_source(op, ctx, 0);
                let result = Self::wide_element_wise_unary(&a, et, Self::vector_neg_ltz);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::AbsGtz => {
                let a = Self::get_wide_vec_source(op, ctx, 0);
                let result = Self::wide_element_wise_unary(&a, et, Self::vector_abs_gtz);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::SetEq => {
                // VEQZ: compare each element to zero, produce scalar comparison bitmask.
                // VEQZ has the cmp register as primary dest (not extra_dests), so
                // write to dest via write_scalar_dest. Also write to extra_dests
                // for dual-result ops that might share this path.
                let a = Self::get_wide_vec_source(op, ctx, 0);
                let result = Self::wide_element_wise_unary(&a, et, Self::vector_compare_eqz);
                let lo: [u32; 8] = result[..8].try_into().unwrap();
                let hi: [u32; 8] = result[8..].try_into().unwrap();
                let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                let lanes_per_half = 256 / et.bits() as u32;
                let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
                if lanes_per_half >= 32 {
                    Self::write_scalar_dest_wide(op, ctx, full_mask);
                    Self::write_cmp_dest_wide(op, ctx, full_mask);
                } else {
                    Self::write_scalar_dest(op, ctx, full_mask as u32);
                    Self::write_cmp_dest(op, ctx, full_mask as u32);
                }
                true
            }

            // ========== Convert (wide path) ==========

            SemanticOp::Convert => {
                // VCONV/VFLOOR: type conversion.
                // Three source types:
                // A. Accumulator source (VCONV_FP32_BF16, VCONV_BF16_FP32): read acc,
                //    extract lower 32 bits per lane as the input vector.
                // B. Narrow vector source (VFLOOR bf16->s32): 256-bit in, 512-bit out.
                // C. Wide vector source: 512-bit in, 512-bit out.
                let from = op.from_type.unwrap_or(ElementType::Int32);
                let has_acc_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::AccumReg(_)));

                if has_acc_source {
                    // Accumulator source: read acc lanes, extract lower 32 bits each.
                    let acc_reg = Self::get_acc_source(op);
                    let acc = ctx.accumulator.read_wide(acc_reg);
                    // Each acc lane is u64; extract lower 32 bits for vector_convert.
                    let mut src_lo = [0u32; 8];
                    let mut src_hi = [0u32; 8];
                    for i in 0..8 {
                        src_lo[i] = acc[i] as u32;
                        src_hi[i] = acc[i + 8] as u32;
                    }
                    let res_lo = Self::vector_convert(&src_lo, from, et);
                    let res_hi = Self::vector_convert(&src_hi, from, et);
                    // VCONV may produce narrow output (e.g., f32->bf16 packs 8 f32
                    // into 4 words of bf16). Write to the appropriate dest width.
                    if et.bits() < from.bits() {
                        // Contraction: 512-bit acc -> 256-bit vector.
                        // Each half contracts; pack results into one 256-bit output.
                        let words_per_half = (8 * et.bits() as usize) / (from.bits() as usize);
                        let mut result = [0u32; 8];
                        result[..words_per_half].copy_from_slice(&res_lo[..words_per_half]);
                        result[words_per_half..words_per_half * 2]
                            .copy_from_slice(&res_hi[..words_per_half]);
                        Self::write_vector_dest(op, ctx, result);
                    } else {
                        // Same-width or expansion from acc.
                        let mut result = [0u32; 16];
                        result[..8].copy_from_slice(&res_lo);
                        result[8..].copy_from_slice(&res_hi);
                        Self::write_wide_vec_dest(op, ctx, result);
                    }
                } else {
                    let is_expansion = from.bits() < et.bits();
                    if is_expansion {
                        // Narrow source (256-bit w-register) -> wide dest (512-bit x-register).
                        let src = Self::get_vector_source(op, ctx, 0);
                        let mut lo_in = [0u32; 8];
                        lo_in[..4].copy_from_slice(&src[..4]);
                        let mut hi_in = [0u32; 8];
                        hi_in[..4].copy_from_slice(&src[4..]);
                        let res_lo = Self::vector_convert(&lo_in, from, et);
                        let res_hi = Self::vector_convert(&hi_in, from, et);
                        let mut result = [0u32; 16];
                        result[..8].copy_from_slice(&res_lo);
                        result[8..].copy_from_slice(&res_hi);
                        Self::write_wide_vec_dest(op, ctx, result);
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

    /// Push a scalar into a 512-bit vector, shifting existing elements.
    ///
    /// - `vpush.lo` (`is_hi=false`): shift elements toward high indices, insert
    ///   scalar at the lowest position (index 0).
    /// - `vpush.hi` (`is_hi=true`): shift elements toward low indices, insert
    ///   scalar at the highest position.
    ///
    /// The shift is element-size-aware: for i32, one push moves 4 bytes; for
    /// i16, 2 bytes; for i8, 1 byte.  The operation works on the full 64-byte
    /// (512-bit) vector, so elements cross the 256-bit lane boundary freely.
    fn wide_vector_push(src: &Vec512, value: u64, is_hi: bool, et: ElementType) -> Vec512 {
        // Flatten to bytes for element-size-agnostic shifting.
        let mut bytes = [0u8; 64];
        for (i, word) in src.iter().enumerate() {
            let b = word.to_le_bytes();
            bytes[i * 4..i * 4 + 4].copy_from_slice(&b);
        }

        let elem_bytes = (et.bits() as usize / 8).max(1);
        let val_bytes = value.to_le_bytes();

        if is_hi {
            // Shift towards low indices, open a slot at the high end.
            bytes.copy_within(elem_bytes.., 0);
            let insert_pos = 64 - elem_bytes;
            for i in 0..elem_bytes {
                bytes[insert_pos + i] = val_bytes[i];
            }
        } else {
            // Shift towards high indices, open a slot at the low end.
            bytes.copy_within(..64 - elem_bytes, elem_bytes);
            for i in 0..elem_bytes {
                bytes[i] = val_bytes[i];
            }
        }

        let mut result = [0u32; 16];
        for (i, chunk) in bytes.chunks(4).enumerate() {
            result[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        result
    }

    /// Extract a single element from any position in a 512-bit vector.
    ///
    /// The element index is masked to the valid range for the given element
    /// type (e.g., 0-31 for i16, 0-15 for i32).  Supports sub-word types
    /// stored in packed little-endian order within each u32 word.
    fn extract_wide_element(src: &Vec512, index: u32, et: ElementType) -> u64 {
        let bits = et.bits() as u32;
        if bits >= 64 {
            // 64-bit element: 8 elements in 512-bit register.
            let idx = (index as usize % 8) * 2;
            (src[idx] as u64) | ((src[idx + 1] as u64) << 32)
        } else {
            let max_elems = 512 / bits;
            let idx = index % max_elems;
            let bit_offset = idx * bits;
            let word_idx = (bit_offset / 32) as usize;
            let bit_in_word = bit_offset % 32;
            let mask = (1u64 << bits) - 1;
            let raw = ((src[word_idx] as u64 >> bit_in_word) & mask) as u32;
            // Sign-extend for signed types
            match et {
                ElementType::Int8 => (raw as u8 as i8 as i32 as u32) as u64,
                ElementType::Int16 => (raw as u16 as i16 as i32 as u32) as u64,
                _ => raw as u64,
            }
        }
    }

    /// Insert a scalar value at a specific element position in a 512-bit vector.
    ///
    /// Returns a copy of `src` with the element at `index` replaced by `value`.
    /// The index is masked to the valid range for the element type.
    /// Insert a 64-bit value (lo + hi words) at a specific 64-bit element position.
    fn insert_wide_element_64(src: &Vec512, index: u32, lo: u32, hi: u32) -> Vec512 {
        let mut result = *src;
        // 512 bits / 64 bits = 8 elements
        let idx = (index as usize) % 8;
        result[idx * 2] = lo;
        result[idx * 2 + 1] = hi;
        result
    }

    fn insert_wide_element(src: &Vec512, index: u32, value: u32, et: ElementType) -> Vec512 {
        let mut result = *src;
        let bits = et.bits() as u32;
        if bits >= 64 {
            // 64-bit element: only lower 32 bits available via this path.
            // Use insert_wide_element_64 for full 64-bit inserts.
            let idx = (index as usize % 8) * 2;
            result[idx] = value;
            return result;
        }
        let max_elems = 512 / bits;
        let idx = index % max_elems;
        let bit_offset = idx * bits;
        let word_idx = (bit_offset / 32) as usize;
        let bit_in_word = bit_offset % 32;
        let mask = (1u64 << bits) - 1;
        // Clear the target element bits and insert the new value.
        let word_val = result[word_idx] as u64;
        let cleared = word_val & !(mask << bit_in_word);
        let inserted = cleared | (((value as u64) & mask) << bit_in_word);
        result[word_idx] = inserted as u32;
        // Handle elements that span a word boundary (e.g., 64-bit at bit_in_word > 0).
        if bit_in_word + et.bits() as u32 > 32 && word_idx + 1 < 16 {
            let overflow_bits = bit_in_word + et.bits() as u32 - 32;
            let overflow_mask = (1u64 << overflow_bits) - 1;
            let hi_val = result[word_idx + 1] as u64;
            let hi_cleared = hi_val & !overflow_mask;
            let hi_inserted = hi_cleared | (((value as u64) >> (et.bits() as u32 - overflow_bits)) & overflow_mask);
            result[word_idx + 1] = hi_inserted as u32;
        }
        result
    }

    /// Concatenate two 512-bit vectors (128 bytes total) and extract 64 bytes
    /// starting at `byte_shift`.
    ///
    /// Logically: `result = (src1 || src2)[byte_shift .. byte_shift + 64]`.
    /// Bytes beyond the 128-byte window are zero-padded.  The shift is masked
    /// to 7 bits (0–127).
    /// VSHIFT/VSHIFT_ALIGN barrel shifter.
    ///
    /// Implements the hardware barrel shift with optional pre-shift stage.
    /// Derived from aietools ISG reference model (`vshift_hw` in
    /// `me_inline_primitives.h`). The hardware uses a mask-based merge
    /// of vectors a and b, followed by a progressive barrel shift, then
    /// an optional pre-shifted copy of a merged into the high bytes.
    ///
    /// Parameters:
    ///   a, b: 512-bit source vectors
    ///   step: 3-bit pre-shift selector (from s-register & 0x7):
    ///         0 = no pre-shift (plain VSHIFT behavior)
    ///         1 = pre-shift a by 4 bytes (32 bits)
    ///         2 = pre-shift a by 8 bytes (64 bits)
    ///         3 = pre-shift a by 16 bytes (128 bits)
    ///         4 = pre-shift a by 32 bytes (256 bits)
    ///   shift: byte shift amount from r-register (full 32-bit value)
    fn wide_vector_shift(a_orig: &Vec512, b: &Vec512, step: u32, shift: u32) -> Vec512 {
        // Convert to byte arrays for manipulation.
        let mut a_bytes = [0u8; 64];
        let mut b_bytes = [0u8; 64];
        for i in 0..16 {
            let aw = a_orig[i].to_le_bytes();
            let bw = b[i].to_le_bytes();
            a_bytes[i*4..i*4+4].copy_from_slice(&aw);
            b_bytes[i*4..i*4+4].copy_from_slice(&bw);
        }

        // Step 1: vshift_mask -- decompose shift register value.
        // Extract 6-bit shift amount and hi_shft flag.
        let shift_6 = (shift & 0x3F) as usize;
        let hi_shft = shift >= 64;

        // Build 64-bit mask: mask = (0xFFFF_FFFF_FFFF_FFFF << shift)[127:64]
        // This gives a mask where the high bits are set based on shift amount.
        let mask_128: u128 = 0xFFFF_FFFF_FFFF_FFFFu128 << (shift & 0x7F);
        let mask_64: u64 = (mask_128 >> 64) as u64;

        // Step 2: Compute pre-shifted version of a (for step != 0).
        let mut pre_bytes = [0u8; 64];
        let mut a_active = a_bytes;
        let step_val = step & 0x7;
        if step_val >= 1 && step_val <= 4 {
            // Pre-shift a right by 2^(step+1) bytes = {4, 8, 16, 32} bytes.
            let pre_shift_bits = match step_val {
                1 => 32,   // 4 bytes
                2 => 64,   // 8 bytes
                3 => 128,  // 16 bytes
                4 => 256,  // 32 bytes
                _ => 0,
            };
            // Right-shift a_orig by pre_shift_bits as a 512-bit value.
            let pre_shift_bytes = pre_shift_bits / 8;
            for i in 0..64 {
                let src_idx = i + pre_shift_bytes;
                pre_bytes[i] = if src_idx < 64 { a_bytes[src_idx] } else { 0 };
            }
            // When step != 0, a is zeroed for the main shift stage.
            a_active = [0u8; 64];
        }

        // Step 3: Build per-byte masks from the 64-bit mask.
        // maska: for vector a, maskb: for vector b.
        // If hi_shft: maska = 0 (all zero), else maska = ~mask (inverted).
        // maskb = mask.
        let maska_64: u64 = if hi_shft { 0 } else { !mask_64 };
        let maskb_64: u64 = mask_64;

        // Expand 1-bit-per-byte masks to full byte masks.
        let mut c_bytes = [0u8; 64];
        for i in 0..64 {
            let a_sel = (maska_64 >> i) & 1;
            let b_sel = (maskb_64 >> i) & 1;
            c_bytes[i] = if a_sel != 0 { a_active[i] }
                         else if b_sel != 0 { b_bytes[i] }
                         else { 0 };
        }

        // Step 4: Progressive barrel shift using shift[5:0] bit by bit.
        // Each bit rotates c by that power of 2 bytes.
        // Bit 5: rotate by 32 bytes
        // Bit 4: rotate by 16 bytes
        // Bit 3: rotate by 8 bytes
        // Bit 2: rotate by 4 bytes
        // Bit 1: rotate by 2 bytes
        // Bit 0: rotate by 1 byte
        // Each rotation: c = [c[0..N-1], c[N..63]] (low bytes move to top).
        for bit in (0..6).rev() {
            if (shift_6 >> bit) & 1 != 0 {
                let n = 1 << bit; // bytes to rotate
                let mut rotated = [0u8; 64];
                for i in 0..64 {
                    rotated[i] = c_bytes[(i + n) % 64];
                }
                c_bytes = rotated;
            }
        }

        // Step 5: Merge pre-shifted portion into result.
        // maskpre = bit-reversed maska (bit i of maskpre = bit (63-i) of maska).
        // result = c | (pre & maskpre)
        let mut maskpre_64: u64 = 0;
        for i in 0..64 {
            let bit = (maska_64 >> (63 - i)) & 1;
            maskpre_64 |= bit << i;
        }
        for i in 0..64 {
            if (maskpre_64 >> i) & 1 != 0 {
                c_bytes[i] |= pre_bytes[i];
            }
        }

        // Convert back to Vec512.
        let mut result = [0u32; 16];
        for i in 0..16 {
            result[i] = u32::from_le_bytes([
                c_bytes[i*4], c_bytes[i*4+1], c_bytes[i*4+2], c_bytes[i*4+3]
            ]);
        }
        result
    }

    /// Simple concat-and-extract shift (used by narrow VSHIFT path).
    fn wide_vector_align(src1: &Vec512, src2: &Vec512, byte_shift: u32) -> Vec512 {
        Self::wide_vector_shift(src1, src2, 0, byte_shift)
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

    // NOTE: test_vector_mac was removed -- it tested the old element-wise MAC
    // path which has been replaced by config-driven matrix multiply dispatch
    // (execute_matmul). MAC tests now belong in vector_matmul.rs.

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

    // NOTE: test_vector_matmul_dense_int32 was removed -- it tested the old
    // element-wise matmul path which has been replaced by config-driven matrix
    // multiply dispatch (execute_matmul). MatMul tests now belong in
    // vector_matmul.rs.

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
        let result = VectorAlu::wide_vector_align(&src1, &src2, 0);
        assert_eq!(result, [1u32; 16]);
    }

    /// Shift by exactly 64 bytes skips all of src1 and returns src2 unchanged.
    #[test]
    fn test_wide_vector_align_full_shift() {
        let src1 = [1u32; 16];
        let src2 = [2u32; 16];
        let result = VectorAlu::wide_vector_align(&src1, &src2, 64);
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
        let result = VectorAlu::wide_vector_align(&src1, &src2, 60);
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

    /// VGE with wide (x-register) sources and scalar dest produces a bitmask.
    /// 16 lanes of i32 across two halves -> 16-bit scalar mask.
    #[test]
    fn test_wide_setge_scalar_dest_i32() {
        let mut ctx = make_ctx();
        // x0 = v0:v1 (lo:hi), x2 = v2:v3
        // lo half (v0): [10, 5, 20, 3, 8, 8, 100, 0]
        // hi half (v1): [1, 2, 3, 4, 5, 6, 7, 8]
        ctx.vector.write(0, [10, 5, 20, 3, 8, 8, 100, 0]);
        ctx.vector.write(1, [1, 2, 3, 4, 5, 6, 7, 8]);
        // lo half (v2): [5, 10, 20, 4, 7, 8, 50, 1]
        // hi half (v3): [1, 3, 2, 4, 6, 5, 7, 9]
        ctx.vector.write(2, [5, 10, 20, 4, 7, 8, 50, 1]);
        ctx.vector.write(3, [1, 3, 2, 4, 6, 5, 7, 9]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::SetGe)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::ScalarReg(16))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(2));
        op.is_wide_vector = true;

        assert!(VectorAlu::execute(&op, &mut ctx));

        // lo half comparison (a >= b):
        //   10>=5=T, 5>=10=F, 20>=20=T, 3>=4=F, 8>=7=T, 8>=8=T, 100>=50=T, 0>=1=F
        //   mask_lo = 0b_0111_0101 = 0x75
        // hi half comparison:
        //   1>=1=T, 2>=3=F, 3>=2=T, 4>=4=T, 5>=6=F, 6>=5=T, 7>=7=T, 8>=9=F
        //   mask_hi = 0b_0110_1101 = 0x6D
        // full_mask = mask_lo | (mask_hi << 8) = 0x6D75
        let scalar_result = ctx.scalar.read(16);
        assert_eq!(scalar_result, 0x6D75, "wide VGE scalar bitmask mismatch: got {:#06x}", scalar_result);
    }

    /// VLT with wide (x-register) sources and scalar dest produces a bitmask.
    #[test]
    fn test_wide_setlt_scalar_dest_i32() {
        let mut ctx = make_ctx();
        // Same data as above
        ctx.vector.write(0, [10, 5, 20, 3, 8, 8, 100, 0]);
        ctx.vector.write(1, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(2, [5, 10, 20, 4, 7, 8, 50, 1]);
        ctx.vector.write(3, [1, 3, 2, 4, 6, 5, 7, 9]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::SetLt)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::ScalarReg(16))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(2));
        op.is_wide_vector = true;

        assert!(VectorAlu::execute(&op, &mut ctx));

        // VLT is the complement of VGE (for signed integers, no NaN).
        // lo: F,T,F,T,F,F,F,T -> 0b_1000_1010 = 0x8A
        // hi: F,T,F,F,T,F,F,T -> 0b_1001_0010 = 0x92
        // full_mask = 0x928A
        let scalar_result = ctx.scalar.read(16);
        assert_eq!(scalar_result, 0x928A, "wide VLT scalar bitmask mismatch: got {:#06x}", scalar_result);
    }

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

    /// VADD_F: NaN + NaN should produce canonical NaN with mantissa = 1.
    ///
    /// Negative s16 values sign-extended to s32 produce f32 NaN bit patterns
    /// (exponent = 0xFF, mantissa != 0).  The hardware canonical NaN is
    /// 0x7F800001 (mantissa = 1, sign = 0).  Verified against real NPU1 HW.
    #[test]
    fn test_vadd_f_nan_canonical() {
        use crate::tablegen::decoder_ffi::AccumWidth;
        let mut ctx = make_ctx();

        // Load NaN values: s16 = -1 -> s32 = 0xFFFFFFFF -> f32: NaN (exp=0xFF).
        // Pack two NaN values per u64 accumulator lane.
        let nan_s32 = 0xFFFFFFFFu32;
        let nan_u64 = (nan_s32 as u64) | ((nan_s32 as u64) << 32);
        let acc = [nan_u64; 8];
        ctx.accumulator.write(0, acc);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Accumulate)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(0));
        op.accum_width = Some(AccumWidth::Half);
        ctx.scalar.write(0, 0);

        VectorAlu::execute(&op, &mut ctx);

        let result = ctx.accumulator.read(0);
        // Each u64 lane should hold two copies of canonical NaN = 0x7F80007F.
        let expected = 0x7F80007F_7F80007Fu64;
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, expected,
                "acc lane {} should be canonical NaN pair, got {:#018x}", i, v);
        }
    }

    /// VADD_F: float accumulator add should flush denormals to zero.
    ///
    /// The test harness loads s16 data via `vups.s32.s16` then runs
    /// `vadd.f bml0, bml0, bml0, r0`.  Small positive s16 values
    /// (1..127) sign-extend to s32 = denormalized f32 bit patterns.
    /// FTZ should flush them to +0.0, so vadd.f(0, 0) = 0.
    /// SRS should then produce zero.
    #[test]
    fn test_vadd_f_denormal_ftz() {
        use crate::tablegen::decoder_ffi::AccumWidth;
        let mut ctx = make_ctx();

        // Simulate UPS: sign-extend 16 s16 values to s32, pack 2 per u64.
        // Values 0..15: all small positive -> denormalized f32 patterns.
        let src = [0x0002_0001u32, 0x0004_0003, 0x0006_0005, 0x0008_0007,
                    0x000A_0009, 0x000C_000B, 0x000E_000D, 0x0010_000F];
        let acc = super::vector_ups::ups_vector_to_acc(
            &src, 0, ElementType::Int16, ElementType::Int32,
        );
        ctx.accumulator.write(0, acc);

        // Build a vadd.f operation: bml0 = bml0 + bml0.
        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Accumulate)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(0)); // config = 0
        op.accum_width = Some(AccumWidth::Half);

        // Config register r0 = 0.
        ctx.scalar.write(0, 0);

        VectorAlu::execute(&op, &mut ctx);

        // Read back via SRS s16.s32 with shift=0.
        let result_acc = ctx.accumulator.read(0);
        // All values were denormalized f32 -> FTZ to 0 -> add(0,0) = 0.
        for (i, &v) in result_acc.iter().enumerate() {
            assert_eq!(v, 0, "acc lane {} should be 0 after FTZ, got {:#018x}", i, v);
        }
    }
}
