//! Vector ALU dispatch table.
//!
//! Single entry point that routes SemanticOp variants to operation
//! implementations. Each match arm is a single function call.

use crate::interpreter::bundle::{ElementType, SlotOp};
use crate::interpreter::state::ExecutionContext;
use xdna_archspec::aie2::isa::SemanticOp;

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

        // Only handle vector operations.
        if !op.is_vector {
            return false;
        }

        // Skip fused ops that address memory -- MemoryUnit handles these.
        // Standalone SRS/Pack/UPS on memory slots with NO memory address are
        // register-only and must be handled here. `addresses_memory()` covers
        // both offset (Memory{}) and register-indirect (PointerReg, i.e.
        // post-increment / indexed) addressing -- matching only Memory{} let
        // post-increment fused loads fall through to the register path here,
        // before MemoryUnit ever saw them (the vec_srs_i32 bug).
        if op.slot.is_memory()
            && matches!(
                semantic,
                SemanticOp::Ups
                    | SemanticOp::Srs
                    | SemanticOp::Pack
                    | SemanticOp::Unpack
                    | SemanticOp::Convert
            )
            && op.addresses_memory()
        {
            return false;
        }

        // Fuzzer anti-folding sentinel: when armed, record that this vector
        // op actually executed (no-op otherwise). Mode captures the SRS
        // config for ops whose semantics depend on it.
        let mode = match semantic {
            SemanticOp::Srs | SemanticOp::Pack | SemanticOp::Convert | SemanticOp::Ups => {
                (ctx.srs_config.saturation_mode << 4) | ctx.srs_config.rounding_mode
            }
            _ => 0,
        };
        super::fuzz_recorder::record(semantic, op.element_type, mode);

        let et = op.element_type.unwrap_or(ElementType::Int32);

        log::trace!(
            "[VECTOR_ALU] Checking semantic={:?} element_type={:?} dest={:?}",
            semantic,
            op.element_type,
            op.dest
        );

        match semantic {
            // ========== Arithmetic ==========
            SemanticOp::Add => Self::execute_add(op, ctx, et),
            SemanticOp::Sub => Self::execute_binary_elementwise(op, ctx, et, Self::vector_sub),
            SemanticOp::Mul => Self::execute_binary_elementwise(op, ctx, et, Self::vector_mul),
            SemanticOp::Min => Self::execute_binary_elementwise(op, ctx, et, Self::vector_min),
            SemanticOp::Max => Self::execute_binary_elementwise(op, ctx, et, Self::vector_max),
            SemanticOp::Neg => Self::execute_neg(op, ctx, et),
            SemanticOp::NegAdd => Self::execute_accumulate(op, ctx, et, semantic),
            SemanticOp::Shl => Self::execute_shl(op, ctx, et),
            SemanticOp::Srl => Self::execute_srl(op, ctx, et),
            SemanticOp::Sra => Self::execute_sra(op, ctx, et),
            SemanticOp::AbsGtz => Self::execute_abs_gtz(op, ctx, et),
            SemanticOp::NegGtz => Self::execute_neg_gtz(op, ctx, et),
            SemanticOp::NegLtz => Self::execute_neg_ltz(op, ctx, et),
            SemanticOp::SubLt => Self::execute_sub_lt(op, ctx, et),
            SemanticOp::SubGe => Self::execute_sub_ge(op, ctx, et),
            SemanticOp::MaxDiffLt => Self::execute_maxdiff_lt(op, ctx, et),
            SemanticOp::Accumulate
            | SemanticOp::AccumSub
            | SemanticOp::AccumNegAdd
            | SemanticOp::AccumNegSub => Self::execute_accumulate(op, ctx, et, semantic),

            // ========== Comparison ==========
            SemanticOp::Cmp => Self::execute_cmp(op, ctx, et),
            SemanticOp::SetGe => Self::execute_setge(op, ctx, et),
            SemanticOp::SetLt => Self::execute_setlt(op, ctx, et),
            SemanticOp::SetEq => Self::execute_seteq(op, ctx, et),
            SemanticOp::MaxLt => Self::execute_maxlt(op, ctx, et),
            SemanticOp::MinGe => Self::execute_minge(op, ctx, et),
            SemanticOp::VectorSelect => Self::execute_select(op, ctx, et),

            // ========== Data movement ==========
            SemanticOp::Shuffle => Self::execute_shuffle(op, ctx, et),
            SemanticOp::VectorBroadcast => Self::execute_vector_broadcast(op, ctx, et),
            SemanticOp::VectorExtract => Self::execute_vector_extract(op, ctx, et),
            SemanticOp::VectorInsert => Self::execute_vector_insert(op, ctx, et),
            SemanticOp::VectorPush | SemanticOp::VectorPushHi => Self::execute_vector_push(op, ctx, et),
            SemanticOp::Align => Self::execute_align(op, ctx, et),
            SemanticOp::Copy => Self::execute_copy(op, ctx, et),
            SemanticOp::VectorClear => Self::execute_vector_clear(op, ctx, et),

            // ========== Bitwise ==========
            SemanticOp::And => Self::execute_binary_typeless(op, ctx, Self::vector_bitwise_and),
            SemanticOp::Or => Self::execute_binary_typeless(op, ctx, Self::vector_bitwise_or),
            SemanticOp::Xor => Self::execute_binary_typeless(op, ctx, Self::vector_bitwise_xor),
            SemanticOp::Not => Self::execute_unary_typeless(op, ctx, Self::vector_bitwise_not),

            // ========== Pack/Unpack ==========
            SemanticOp::Pack => Self::execute_pack(op, ctx, et),
            SemanticOp::Unpack => Self::execute_unpack(op, ctx, et),

            // ========== Pipelines ==========
            SemanticOp::Srs => Self::execute_srs(op, ctx, et),
            SemanticOp::Ups => Self::execute_ups(op, ctx, et),
            SemanticOp::Convert => Self::execute_convert(op, ctx, et),

            // ========== Matrix engine ==========
            SemanticOp::Mac
            | SemanticOp::MatMul
            | SemanticOp::MatMulSub
            | SemanticOp::NegMul
            | SemanticOp::NegMatMul
            | SemanticOp::AddMac
            | SemanticOp::SubMac => super::vector_matmul::execute_matmul(op, ctx),

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{Operand, SlotIndex};
    use crate::interpreter::execute::fuzz_recorder;

    /// Executing a vector op through the dispatcher records it when armed.
    #[test]
    fn test_dispatch_records_executed_op_when_armed() {
        let mut ctx = ExecutionContext::new();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        fuzz_recorder::arm();
        assert!(VectorAlu::execute(&op, &mut ctx));

        let keys = fuzz_recorder::take().expect("recorder was armed");
        assert!(keys.iter().any(|k| k.starts_with("Add/")), "expected an Add/ key, got {keys:?}");
    }
}
