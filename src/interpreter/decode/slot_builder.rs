//! SlotOp construction from decoded instructions.
//!
//! Builds `SlotOp` values with TableGen-derived metadata (semantic operation,
//! element type, memory width, implicit registers, etc.).

use crate::interpreter::bundle::{MemWidth, Operand, PostModify, SlotIndex, SlotOp};
use xdna_archspec::aie2::isa::{
    CompositeEncoder, InstrMemWidth, OperandType, RegisterKind, SemanticOp,
};

use super::decoder::{DecodedInstr, InstructionDecoder};

impl InstructionDecoder {
    /// Build a SlotOp with TableGen-derived information.
    ///
    /// Sources are already in canonical order from extract_operands().
    /// This method:
    /// 1. Sets the semantic operation from TableGen
    /// 2. Attaches implicit register uses/defs
    pub(super) fn build_slot_op(
        &self,
        slot_index: SlotIndex,
        decoded: &DecodedInstr,
        dest: Option<Operand>,
        sources: Vec<Operand>,
        extracted_pm: Option<PostModify>,
    ) -> SlotOp {
        let enc = &decoded.encoding;

        // Semantic is fully resolved at build time (see extract.rs Layers 1-5).
        // PointerAdd override: some PADDB/PADDA variants get Add from pattern
        // matching. The is_ptr_arithmetic flag (from mnemonic "padd*") corrects it.
        let effective_semantic = if enc.is_ptr_arithmetic {
            Some(SemanticOp::PointerAdd)
        } else if enc.semantic.is_none() && dest.is_some() && (
            enc.mnemonic == "mov" || enc.mnemonic == "mova"
            || enc.mnemonic == "movx" || enc.mnemonic == "movxm"
        ) && !enc.is_vector {
            // Move instructions that lack a Pat<> pattern (e.g., MOVX_mvx_scl
            // writes to a control register). LLVM marks these isMoveReg=1 or
            // isMoveImm=1 but the regex parser doesn't extract those flags.
            // Only applies when there's a destination (excludes stream push
            // operations like MOV_mv_scl2ms which write to master stream).
            Some(SemanticOp::Copy)
        } else {
            enc.semantic
        };
        let mut slot_op = if let Some(semantic) = effective_semantic {
            SlotOp::from_semantic(slot_index, semantic)
        } else if enc.mnemonic == "opcodestr" || enc.mnemonic.is_empty() {
            // "opcodestr" is a parser artifact from unresolved TableGen template
            // parameters in NOP class definitions; treat as NOP.
            SlotOp::nop(slot_index)
        } else {
            // No semantic -- unknown instruction. This will hit the executor's
            // "no semantic" error path and abort on first execution.
            log::error!(
                "[NO SEMANTIC] Instruction '{}' has no SemanticOp (no pattern or structural match)",
                enc.mnemonic
            );
            let mut s = SlotOp::nop(slot_index);
            s.semantic = None;
            s.raw_opcode = Some(decoded.operands.get("word0").copied().unwrap_or(0) as u32);
            s
        };

        // -- Populate metadata from InstrEncoding --
        slot_op.is_vector = enc.is_vector;
        // Detect 512+ bit operations: any operand uses Vector512, Vector1024,
        // or a composite register that can encode x-registers (MvBMXSrc/Dst).
        slot_op.is_wide_vector = enc.operand_fields.iter()
            .any(|f| matches!(f.operand_type,
                OperandType::Register(RegisterKind::Vector512)
                | OperandType::Register(RegisterKind::Vector1024)
                | OperandType::CompositeRegister(CompositeEncoder::MvBMXSrc)
                | OperandType::CompositeRegister(CompositeEncoder::MvBMXDst)
                | OperandType::CompositeRegister(CompositeEncoder::ShflDst)));
        // Detect 1024-bit (y-register) operations for sparse MAC wide variants.
        slot_op.is_quad_vector = enc.operand_fields.iter()
            .any(|f| f.operand_type == OperandType::Register(RegisterKind::Vector1024));
        slot_op.element_type = enc.element_type;
        slot_op.from_type = enc.from_type;
        slot_op.mem_width = match enc.mem_width {
            InstrMemWidth::Byte => MemWidth::Byte,
            InstrMemWidth::HalfWord => MemWidth::HalfWord,
            InstrMemWidth::Word => MemWidth::Word,
            InstrMemWidth::QuadWord => MemWidth::QuadWord,
            InstrMemWidth::Vector256 => MemWidth::Vector256,
        };
        slot_op.branch_condition = enc.branch_condition;
        slot_op.select_variant = enc.select_variant;
        // PostModify comes directly from AG field extraction -- no backpatching.
        slot_op.post_modify = extracted_pm.unwrap_or(PostModify::None);

        // Store encoding mnemonic for crossref/debugging
        slot_op.encoding_name = Some(enc.mnemonic.clone());

        // VBCSTSHFL: TableGen maps to Shuffle, but the instruction is a scalar
        // broadcast to 512-bit vector (with optional shuffle). Remap to
        // VectorBroadcast so the execution handler reads the scalar operand
        // correctly. The shuffle step is a no-op when all lanes are identical.
        if enc.mnemonic.starts_with("vbcstshfl") {
            slot_op.semantic = Some(SemanticOp::VectorBroadcast);
        }

        // Add implicit registers from TableGen
        if !enc.implicit_regs.is_empty() {
            slot_op = slot_op.with_implicit_regs(enc.implicit_regs.clone());
        }

        // Set destination and sources (already in TableGen canonical order)
        if let Some(d) = dest {
            slot_op = slot_op.with_dest(d);
        }
        for src in sources {
            slot_op = slot_op.with_source(src);
        }

        slot_op
    }
}
