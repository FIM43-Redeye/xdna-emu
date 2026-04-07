//! Vector ALU dispatch table.
//!
//! Single entry point that routes SemanticOp variants to operation
//! implementations. Each match arm is a single function call.
//!
//! During the dispatch inversion refactor, this file starts as a
//! scaffold that delegates to the old execute_half/execute_wide.
//! As operations are extracted, they replace arms in this table.

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::tablegen::SemanticOp;

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
        // 3. Accumulator source: SRS/UPS/Convert with accumulator source
        //    need the wide path because execute_half only reads VectorReg
        //    sources. Both Half (bml/bmh) and Full (cm) accumulators route here.
        let has_wide_acc_source = matches!(op.accum_width,
            Some(crate::tablegen::decoder_ffi::AccumWidth::Full)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::Half));
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
}
