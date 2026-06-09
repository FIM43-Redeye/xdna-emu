//! Semantic operation inference from structured TableGen data.
//!
//! Infers semantic operations (Load, Store, Call, Ret, Branch, etc.) from
//! structural signals in TableGen instruction definitions, and refines them
//! using mnemonic conventions. Also infers element types, branch conditions,
//! and select variants.

use super::super::types::{BranchCondition, ElementType, SelectVariant, SemanticOp};

/// Infer semantic operation from structured TableGen data.
///
/// Uses register defs/uses, memory flags, delay slot, and parent class names
/// to classify instructions without parsing the mnemonic string. This is the
/// preferred inference path -- it relies on compiler-verified attributes from
/// llvm-aie rather than hand-maintained string matching tables.
///
/// Returns None for instructions that can't be classified structurally
/// (arithmetic, comparison, bitwise -- these need the mnemonic fallback).
pub fn infer_semantic_from_structure(
    defs: &[String],
    uses: &[String],
    may_load: bool,
    may_store: bool,
    has_delay_slot: bool,
    parents: &[String],
) -> Option<SemanticOp> {
    // Control flow: use Defs/Uses of lr and hasDelaySlot.
    // On AIE2 hardware encodings, isBranch/isCall/isReturn are only set on
    // Pseudo instructions, but these structural signals are reliable:
    //   JL/JL_IND: Defs=[lr], hasDelaySlot=1    -> Call
    //   RET:       Uses=[lr], hasDelaySlot=1     -> Ret
    //   J_jump_*:  hasDelaySlot=1, no lr         -> Br
    //   JNZD:      hasDelaySlot=1, no lr         -> BrCond (needs mnemonic for condition)
    let defs_lr = defs.iter().any(|r| r == "lr");
    let uses_lr = uses.iter().any(|r| r == "lr");

    if has_delay_slot {
        if defs_lr {
            return Some(SemanticOp::Call);
        }
        if uses_lr {
            return Some(SemanticOp::Ret);
        }
        // Other delay-slot instructions are branches. Return Br as the
        // baseline; callers upgrade to BrCond via `refine_branch_semantic()`
        // using the mnemonic to distinguish conditional vs unconditional.
        return Some(SemanticOp::Br);
    }

    // Memory operations: mayLoad/mayStore are authoritative flags from TableGen.
    // The slot (lda/ldb/st) and mnemonic (vlda/vst) refine the operation type,
    // but Load vs Store classification comes from these flags.
    if may_load && !may_store {
        return Some(SemanticOp::Load);
    }
    if may_store && !may_load {
        return Some(SemanticOp::Store);
    }

    // Parent class chain: format class names encode operation type.
    // These are compiler-internal names that follow a strict naming convention.
    for parent in parents {
        // Lock operations: AIE2_mLockId in parent chain
        if parent.contains("mLockId") || parent.contains("LockId") {
            // Can't distinguish acquire vs release from class alone;
            // leave for mnemonic fallback.
        }
        // Done (halt): AIE2_done_inst_alu
        if parent.contains("_done_") {
            return Some(SemanticOp::Done);
        }
    }

    None
}

/// Convert a shared TypeTag to the crate's ElementType.
fn tag_to_element_type(tag: super::super::element_type_logic::TypeTag) -> ElementType {
    use super::super::element_type_logic::TypeTag;
    match tag {
        TypeTag::Int8 => ElementType::Int8,
        TypeTag::UInt8 => ElementType::UInt8,
        TypeTag::Int16 => ElementType::Int16,
        TypeTag::UInt16 => ElementType::UInt16,
        TypeTag::Int32 => ElementType::Int32,
        TypeTag::UInt32 => ElementType::UInt32,
        TypeTag::Int64 => ElementType::Int64,
        TypeTag::UInt64 => ElementType::UInt64,
        TypeTag::BFloat16 => ElementType::BFloat16,
        TypeTag::Float32 => ElementType::Float32,
    }
}

/// Infer element type from a mnemonic suffix.
///
/// Delegates to shared logic in `element_type_logic.rs`.
pub fn infer_element_type(mnemonic: &str) -> Option<ElementType> {
    super::super::element_type_logic::infer_type_tag_from_mnemonic(mnemonic).map(tag_to_element_type)
}

/// Infer both element types for dual-type instructions (SRS/UPS/CONV/FLOOR).
///
/// Delegates to shared logic in `element_type_logic.rs`.
/// Returns `(element_type, from_type)` where element_type is the OUTPUT
/// type and from_type is the INPUT type.
pub fn infer_dual_element_types(name: &str) -> (Option<ElementType>, Option<ElementType>) {
    let (et, ft) = super::super::element_type_logic::infer_dual_type_tags(name);
    (et.map(tag_to_element_type), ft.map(tag_to_element_type))
}

/// Refine a Br semantic to BrCond when the mnemonic indicates a condition.
///
/// Structural inference returns `Br` for all delay-slot instructions without lr.
/// This function upgrades `Br` -> `BrCond` when the mnemonic encodes a condition
/// (jnz, jz, jnzd, b*). Unconditional jumps (j) stay as `Br`.
pub fn refine_branch_semantic(mnemonic: &str, semantic: Option<SemanticOp>) -> Option<SemanticOp> {
    if semantic != Some(SemanticOp::Br) {
        return semantic;
    }
    let mn = mnemonic.to_lowercase();
    if mn.starts_with("jnz") || mn.starts_with("jz") || (mn.starts_with('b') && !mn.starts_with("bswap")) {
        Some(SemanticOp::BrCond)
    } else {
        semantic
    }
}

/// Refine Load/Store/None into fused compute semantics using the TableGen
/// instruction name.
///
/// Fused instructions (vlda.ups, vst.srs, vst.pack, vldb.unpack, vlda.conv,
/// vst.conv) combine memory access with a compute step.  Structural inference
/// only sees `mayLoad`/`mayStore` and returns generic Load/Store.  The
/// instruction NAME from TableGen encodes the fused operation (e.g.
/// `VLDA_UPS_S64_S32_ag_idx`), so we can refine without runtime mnemonic
/// parsing.
///
/// Standalone VUNPACK/VPACK have neither mayLoad nor mayStore (they operate
/// purely on registers), so structural inference returns None.  We assign
/// Unpack/Pack based on the name prefix.
pub fn refine_fused_semantic(name: &str, semantic: Option<SemanticOp>) -> Option<SemanticOp> {
    match semantic {
        Some(SemanticOp::Load) => {
            // VLDA_UPS_* -> load + upshift to accumulator
            if name.contains("_UPS_") {
                return Some(SemanticOp::Ups);
            }
            // VLDB_*_UNPACK_* or VLDA_*_UNPACK_* -> load + unpack (widen)
            if name.contains("_UNPACK_") {
                return Some(SemanticOp::Unpack);
            }
            // VLDA_CONV_* -> load + convert (e.g., bf16 -> f32)
            if name.contains("_CONV_") {
                return Some(SemanticOp::Convert);
            }
            semantic
        }
        Some(SemanticOp::Store) => {
            // VST_SRS_* -> shift-round-saturate + store
            if name.contains("_SRS_") {
                return Some(SemanticOp::Srs);
            }
            // VST_PACK_* -> pack (narrow) + store
            if name.contains("_PACK_") {
                return Some(SemanticOp::Pack);
            }
            // VST_CONV_* -> convert + store (e.g., f32 -> bf16)
            if name.contains("_CONV_") {
                return Some(SemanticOp::Convert);
            }
            semantic
        }
        None => {
            // Standalone VUNPACK (register-to-register, no memory access)
            if name.starts_with("VUNPACK_") {
                return Some(SemanticOp::Unpack);
            }
            // Standalone VPACK (register-to-register, no memory access)
            if name.starts_with("VPACK_") {
                return Some(SemanticOp::Pack);
            }
            None
        }
        _ => semantic,
    }
}

/// Refine a `Mul` semantic to `MatMul` for the matrix-multiply VMUL.
///
/// In AIE2 there is no elementwise VMUL: the `vmul` mnemonic writing to a
/// cm/bm accumulator destination, with a config-word operand, IS a fresh
/// matrix multiply driven by the systolic MAC array (the encoding hardcodes
/// acc1 = 0b1111, i.e. no accumulator input). LLVM's Pat<>-based inference
/// assigns it the generic `Mul`, which the emulator would execute as an
/// elementwise multiply that never writes the accumulator. This upgrades it
/// to `MatMul` so it routes to the matrix-multiply execute path.
///
/// The matrix form is identified by the TableGen def name
/// (`VMUL_..._cm_core_...` for integer, `VMUL_F_..._bm_core_...` for bf16).
/// VMAC/VMSC/VNEGMAC already resolve to MAC-family semantics, and VNEGMUL
/// (negated fresh multiply, `NegMul`) keeps its own semantic -- only the plain
/// fresh-multiply `Mul` is refined here.
pub fn refine_matmul_semantic(name: &str, semantic: Option<SemanticOp>) -> Option<SemanticOp> {
    if semantic == Some(SemanticOp::Mul)
        && name.starts_with("VMUL_")
        && (name.contains("cm_core") || name.contains("bm_core"))
    {
        return Some(SemanticOp::MatMul);
    }
    semantic
}

/// Infer branch condition from mnemonic and semantic operation.
///
/// Only meaningful for BrCond instructions. Returns None for other semantics.
pub fn infer_branch_condition(mnemonic: &str, semantic: Option<SemanticOp>) -> Option<BranchCondition> {
    if semantic != Some(SemanticOp::BrCond) {
        return None;
    }
    let mn = mnemonic.to_lowercase();
    Some(if mn == "jnzd" {
        BranchCondition::NotZeroDecrement
    } else if mn == "jnz" {
        BranchCondition::NotZero
    } else if mn == "jz" {
        BranchCondition::Zero
    } else if mn.starts_with("beq") || mn == "bz" {
        BranchCondition::Equal
    } else if mn.starts_with("bne") || mn == "bnz" {
        BranchCondition::NotEqual
    } else if mn.starts_with("blt") {
        BranchCondition::Less
    } else if mn.starts_with("bge") {
        BranchCondition::GreaterEqual
    } else {
        BranchCondition::NotEqual // Fallback
    })
}

/// Infer select variant from mnemonic and semantic operation.
///
/// Only meaningful for Select instructions. Returns None for other semantics.
pub fn infer_select_variant(mnemonic: &str, semantic: Option<SemanticOp>) -> Option<SelectVariant> {
    if semantic != Some(SemanticOp::Select) {
        return None;
    }
    let mn = mnemonic.to_lowercase();
    Some(if mn.contains("eqz") {
        SelectVariant::EqualZero
    } else if mn.contains("nez") {
        SelectVariant::NotEqualZero
    } else {
        SelectVariant::Generic
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_inference_from_structure() {
        // Call: Defs=[lr] + hasDelaySlot (JL, JL_IND)
        assert_eq!(
            infer_semantic_from_structure(&["lr".into()], &[], false, false, true, &[],),
            Some(SemanticOp::Call),
        );

        // Return: Uses=[lr] + hasDelaySlot (RET)
        assert_eq!(
            infer_semantic_from_structure(&[], &["lr".into()], false, false, true, &[],),
            Some(SemanticOp::Ret),
        );

        // Load: mayLoad=true (VLDA, LDA, etc.)
        assert_eq!(infer_semantic_from_structure(&[], &[], true, false, false, &[],), Some(SemanticOp::Load),);

        // Store: mayStore=true (VST, ST, etc.)
        assert_eq!(
            infer_semantic_from_structure(&[], &[], false, true, false, &[],),
            Some(SemanticOp::Store),
        );

        // Done: parent class chain contains "_done_"
        assert_eq!(
            infer_semantic_from_structure(&[], &[], false, false, false, &["AIE2_done_inst_alu".into()],),
            Some(SemanticOp::Done),
        );

        // hasDelaySlot alone (without lr) returns Br (baseline branch).
        // Callers use refine_branch_semantic() to upgrade to BrCond.
        assert_eq!(
            infer_semantic_from_structure(&["srCarry".into()], &[], false, false, true, &[],),
            Some(SemanticOp::Br),
        );

        // Pure arithmetic: no structural signals -> None
        assert_eq!(infer_semantic_from_structure(&["srCarry".into()], &[], false, false, false, &[],), None,);
    }

    #[test]
    fn test_infer_element_type_64bit() {
        assert_eq!(infer_element_type("vpush.hi.64"), Some(ElementType::Int64));
        assert_eq!(infer_element_type("vpush.lo.64"), Some(ElementType::Int64));
        assert_eq!(infer_element_type("vinsert.64"), Some(ElementType::Int64));
        assert_eq!(infer_element_type("vextract.64"), Some(ElementType::Int64));
        // Existing types still work.
        assert_eq!(infer_element_type("vpush.hi.32"), Some(ElementType::Int32));
        assert_eq!(infer_element_type("vpush.lo.16"), Some(ElementType::Int16));
        assert_eq!(infer_element_type("vinsert.8"), Some(ElementType::Int8));
    }

    #[test]
    fn test_infer_dual_element_types_srs() {
        let (et, ft) = infer_dual_element_types("VSRS_S16_S32_mv_w_srs");
        assert_eq!(et, Some(ElementType::Int16));
        assert_eq!(ft, Some(ElementType::Int32));

        let (et, ft) = infer_dual_element_types("VSRS_D32_S64_mv_x_srs");
        assert_eq!(et, Some(ElementType::UInt32));
        assert_eq!(ft, Some(ElementType::Int64));

        let (et, ft) = infer_dual_element_types("VSRS_S8_S32_mv_w_srs");
        assert_eq!(et, Some(ElementType::Int8));
        assert_eq!(ft, Some(ElementType::Int32));
    }

    #[test]
    fn test_infer_dual_element_types_vsrsm() {
        // VSRSM (masked SRS) must also infer types from its name.
        let (et, ft) = infer_dual_element_types("VSRSM_D32_S64");
        assert_eq!(et, Some(ElementType::UInt32));
        assert_eq!(ft, Some(ElementType::Int64));

        let (et, ft) = infer_dual_element_types("VSRSM_S16_S32");
        assert_eq!(et, Some(ElementType::Int16));
        assert_eq!(ft, Some(ElementType::Int32));

        let (et, ft) = infer_dual_element_types("VSRSM_D16_S32");
        assert_eq!(et, Some(ElementType::UInt16));
        assert_eq!(ft, Some(ElementType::Int32));
    }

    #[test]
    fn test_infer_dual_element_types_ups() {
        let (et, ft) = infer_dual_element_types("VUPS_S32_D16_mv_ups_w2b");
        assert_eq!(et, Some(ElementType::Int32));
        assert_eq!(ft, Some(ElementType::UInt16));

        let (et, ft) = infer_dual_element_types("VUPS_S64_S32_mv_ups_w2b");
        assert_eq!(et, Some(ElementType::Int64));
        assert_eq!(ft, Some(ElementType::Int32));

        let (et, ft) = infer_dual_element_types("VUPS_S32_S16_mv_ups_x2c");
        assert_eq!(et, Some(ElementType::Int32));
        assert_eq!(ft, Some(ElementType::Int16));
    }

    #[test]
    fn test_infer_dual_element_types_fused() {
        // Pattern 2: with 2D/3D prefix
        let (et, ft) = infer_dual_element_types("VLDA_2D_UPS_S32_D16");
        assert_eq!(et, Some(ElementType::Int32));
        assert_eq!(ft, Some(ElementType::UInt16));

        let (et, ft) = infer_dual_element_types("VST_2D_SRS_D8_S32");
        assert_eq!(et, Some(ElementType::UInt8));
        assert_eq!(ft, Some(ElementType::Int32));

        // Pattern 3: without 2D/3D prefix (direct fused encoding)
        let (et, ft) = infer_dual_element_types("VLDA_UPS_S64_S32_ag_idx");
        assert_eq!(et, Some(ElementType::Int64));
        assert_eq!(ft, Some(ElementType::Int32));

        let (et, ft) = infer_dual_element_types("VLDA_UPS_S64_D32_ag_pstm_nrm");
        assert_eq!(et, Some(ElementType::Int64));
        assert_eq!(ft, Some(ElementType::UInt32));

        let (et, ft) = infer_dual_element_types("VST_SRS_S16_S32_ag_idx_imm");
        assert_eq!(et, Some(ElementType::Int16));
        assert_eq!(ft, Some(ElementType::Int32));
    }

    #[test]
    fn test_infer_dual_element_types_conv() {
        // Standalone VCONV: VCONV_{OUT}_{IN} (same convention as SRS/UPS)
        // VCONV_FP32_BF16 = bf16 input -> fp32 output
        let (et, ft) = infer_dual_element_types("VCONV_FP32_BF16");
        assert_eq!(et, Some(ElementType::Float32)); // output = FP32
        assert_eq!(ft, Some(ElementType::BFloat16)); // from = BF16

        let (et, ft) = infer_dual_element_types("VCONV_BF16_FP32");
        assert_eq!(et, Some(ElementType::BFloat16));
        assert_eq!(ft, Some(ElementType::Float32));

        // VFLOOR: VFLOOR_{OUT}_{IN}_* (same as SRS/UPS convention)
        // VFLOOR_S32_BF16 = floor BF16 input to S32 output
        let (et, ft) = infer_dual_element_types("VFLOOR_S32_BF16_mFl2FxSrc_AM");
        assert_eq!(et, Some(ElementType::Int32)); // output = S32
        assert_eq!(ft, Some(ElementType::BFloat16)); // from = BF16

        // Fused CONV: VLDA_2D_CONV_FP32_BF16
        // Same {OUT}_{IN} convention: FP32 is output, BF16 is input.
        let (et, ft) = infer_dual_element_types("VLDA_2D_CONV_FP32_BF16");
        assert_eq!(et, Some(ElementType::Float32));
        assert_eq!(ft, Some(ElementType::BFloat16));

        // Fused CONV without 2D: VLDA_CONV_FP32_BF16_ag_idx
        let (et, ft) = infer_dual_element_types("VLDA_CONV_FP32_BF16_ag_idx");
        assert_eq!(et, Some(ElementType::Float32));
        assert_eq!(ft, Some(ElementType::BFloat16));

        // Fused store CONV: VST_CONV_BF16_FP32_ag_idx
        // {OUT}_{IN}: BF16 output, FP32 input.
        let (et, ft) = infer_dual_element_types("VST_CONV_BF16_FP32_ag_idx");
        assert_eq!(et, Some(ElementType::BFloat16));
        assert_eq!(ft, Some(ElementType::Float32));

        // Fused store CONV with 2D: VST_CONV_2D_BF16_FP32
        let (et, ft) = infer_dual_element_types("VST_CONV_2D_BF16_FP32");
        assert_eq!(et, Some(ElementType::BFloat16));
        assert_eq!(ft, Some(ElementType::Float32));
    }

    #[test]
    fn test_infer_dual_element_types_non_srs_ups() {
        let (et, ft) = infer_dual_element_types("VADD_32");
        assert_eq!(et, None);
        assert_eq!(ft, None);

        let (et, ft) = infer_dual_element_types("VMAC_vmac_cm_core_dense");
        assert_eq!(et, None);
        assert_eq!(ft, None);
    }

    #[test]
    fn test_refine_fused_semantic_load_ups() {
        assert_eq!(
            refine_fused_semantic("VLDA_UPS_S64_S32_ag_idx", Some(SemanticOp::Load)),
            Some(SemanticOp::Ups),
        );
        assert_eq!(
            refine_fused_semantic("VLDA_UPS_S32_D16_ag_pstm_nrm_imm", Some(SemanticOp::Load)),
            Some(SemanticOp::Ups),
        );
    }

    #[test]
    fn test_refine_fused_semantic_store_srs() {
        assert_eq!(
            refine_fused_semantic("VST_SRS_S32_S64_ag_idx_imm", Some(SemanticOp::Store)),
            Some(SemanticOp::Srs),
        );
    }

    #[test]
    fn test_refine_fused_semantic_store_pack() {
        assert_eq!(
            refine_fused_semantic("VST_PACK_D4_D8_ag_idx", Some(SemanticOp::Store)),
            Some(SemanticOp::Pack),
        );
    }

    #[test]
    fn test_refine_fused_semantic_load_unpack() {
        assert_eq!(
            refine_fused_semantic("VLDB_2D_UNPACK_D16_D8", Some(SemanticOp::Load)),
            Some(SemanticOp::Unpack),
        );
    }

    #[test]
    fn test_refine_fused_semantic_standalone_pack_unpack() {
        // Standalone VUNPACK/VPACK have no mayLoad/mayStore -> semantic=None
        assert_eq!(refine_fused_semantic("VUNPACK_D16_D8", None), Some(SemanticOp::Unpack),);
        assert_eq!(refine_fused_semantic("VPACK_S4_S8", None), Some(SemanticOp::Pack),);
    }

    #[test]
    fn test_refine_fused_semantic_conv() {
        assert_eq!(
            refine_fused_semantic("VLDA_CONV_FP32_BF16_ag_idx", Some(SemanticOp::Load)),
            Some(SemanticOp::Convert),
        );
        assert_eq!(
            refine_fused_semantic("VST_CONV_BF16_FP32_ag_pstm_nrm", Some(SemanticOp::Store)),
            Some(SemanticOp::Convert),
        );
    }

    #[test]
    fn test_refine_matmul_semantic_vmul_to_matmul() {
        // The matrix-multiply VMUL (cm/bm accumulator destination, config-word
        // operand) is a fresh matrix multiply -- it must be MatMul, not the
        // elementwise Mul the Pat<>-based inference assigns. Integer + bf16.
        assert_eq!(
            refine_matmul_semantic("VMUL_vmac_cm_core_dense", Some(SemanticOp::Mul)),
            Some(SemanticOp::MatMul),
        );
        assert_eq!(
            refine_matmul_semantic("VMUL_F_vmac_bm_core_dense", Some(SemanticOp::Mul)),
            Some(SemanticOp::MatMul),
        );
        assert_eq!(
            refine_matmul_semantic("VMUL_vmac_cm_core_sparse_wide", Some(SemanticOp::Mul)),
            Some(SemanticOp::MatMul),
        );
    }

    #[test]
    fn test_refine_matmul_semantic_leaves_others_alone() {
        // VMAC already resolves to Mac (accumulate) -- must not be touched.
        assert_eq!(
            refine_matmul_semantic("VMAC_vmac_cm_core_dense", Some(SemanticOp::Mac)),
            Some(SemanticOp::Mac),
        );
        // VNEGMUL (negated fresh multiply) is not a plain VMUL: leave it.
        assert_eq!(
            refine_matmul_semantic("VNEGMUL_vmac_cm_core_dense", Some(SemanticOp::NegMul)),
            Some(SemanticOp::NegMul),
        );
        // A non-matrix Mul (no cm/bm-core accumulator destination) stays Mul.
        assert_eq!(
            refine_matmul_semantic("VMUL_elem_something", Some(SemanticOp::Mul)),
            Some(SemanticOp::Mul),
        );
        // Non-Mul semantics pass through unchanged.
        assert_eq!(refine_matmul_semantic("VADD_32", Some(SemanticOp::Add)), Some(SemanticOp::Add),);
    }

    #[test]
    fn test_refine_fused_semantic_passthrough() {
        // Plain load/store without fused suffix -> unchanged
        assert_eq!(refine_fused_semantic("VLDA_ag_idx_imm", Some(SemanticOp::Load)), Some(SemanticOp::Load),);
        assert_eq!(
            refine_fused_semantic("VST_dmw_sts_w_ag_idx", Some(SemanticOp::Store)),
            Some(SemanticOp::Store),
        );
        // Non-memory semantics -> unchanged
        assert_eq!(refine_fused_semantic("VADD_32", Some(SemanticOp::Add)), Some(SemanticOp::Add),);
    }
}
