//! AIE2 instruction set architecture: decoder tables, runtime model,
//! and LLVM MCDisassembler FFI.
//!
//! Content here is build-time extracted from llvm-aie (TableGen sources
//! + LLVM libraries). The runtime consumes generated constants via
//! `load_from_generated()`.
//!
//! The interpreter-coupled half of the old `decoder_ffi.rs` (MappedOperand,
//! RegisterMap, classify_reg_name, AccumWidth) lives in xdna-emu's
//! `interpreter::decode::register_map`, not here -- those types reference
//! the interpreter's `Operand` enum.

pub mod types;
pub mod resolver;
pub mod decoder_ffi;
pub mod element_type_logic;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/gen_tablegen.rs"));
}

/// Load the complete TableGen model from build-time generated constants.
///
/// This is the sole entry point for instruction decoder data. All 600+
/// instruction encodings, scheduling model, register
/// definitions, and composite format layouts are compiled in.
pub fn load_from_generated() -> types::TblgenOutput {
    generated::load_from_generated()
}

// Re-exports matching xdna-emu's src/tablegen/mod.rs surface so consumers
// can `use xdna_archspec::aie2::isa::*;` to get the full tablegen API.
pub use types::*;
pub use resolver::{
    build_decoder_tables, AddressingMode, CompositeEncoder, DecoderIndex, InstrEncoding, InstrMemWidth,
    OperandField, OperandType, RegisterKind, ResolveError, Resolver, SlotIndex, classify_operand_type,
    detect_addressing_mode, detect_mem_width, infer_branch_condition, infer_dual_element_types,
    infer_element_type, infer_select_variant, refine_branch_semantic,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// ACQ_mLockId_imm and ACQ_mLockId_reg must have distinguishing fixed
    /// bits in the build-time generated data, or they would decode
    /// ambiguously.
    #[test]
    fn test_acq_instruction_disambiguation() {
        let output = load_from_generated();
        let alu_encodings = output.encodings_by_slot.get("alu").expect("No alu slot encodings");
        let acq_imm = alu_encodings.iter().find(|e| e.name == "ACQ_mLockId_imm");
        let acq_reg = alu_encodings.iter().find(|e| e.name == "ACQ_mLockId_reg");
        if let (Some(imm), Some(reg)) = (acq_imm, acq_reg) {
            assert_ne!(
                imm.fixed_bits, reg.fixed_bits,
                "ACQ_mLockId_imm and ACQ_mLockId_reg should have different fixed_bits"
            );
            assert_ne!(
                imm.fixed_mask & imm.fixed_bits,
                reg.fixed_mask & reg.fixed_bits,
                "Masked bits should differ"
            );
        } else {
            let acq_names: Vec<_> = alu_encodings
                .iter()
                .filter(|e| e.name.to_lowercase().contains("acq"))
                .map(|e| &e.name)
                .collect();
            panic!("Expected both ACQ variants, found: {:?}", acq_names);
        }
    }

    /// Lock ID fields must be classified as LockId (unsigned), not signed
    /// immediate.
    #[test]
    fn test_acq_lock_id_field_type() {
        let output = load_from_generated();
        let alu = output.encodings_by_slot.get("alu").expect("No alu slot");
        let enc = alu
            .iter()
            .find(|e| e.name == "ACQ_mLockId_imm")
            .expect("ACQ_mLockId_imm should exist");
        let id_field = enc
            .operand_fields
            .iter()
            .find(|f| f.name == "id")
            .expect("ACQ_mLockId_imm should have 'id' field");
        assert_eq!(
            id_field.operand_type,
            OperandType::LockId,
            "Lock ID field should be LockId, not {:?}",
            id_field.operand_type
        );
    }

    /// Structural semantic inference: every instruction must have a
    /// SemanticOp assigned.
    #[test]
    fn test_structural_semantic_inference() {
        let output = load_from_generated();
        let all: Vec<_> = output.encodings_by_slot.values().flatten().collect();

        if let Some(jl) = all.iter().find(|e| e.name == "JL") {
            assert_eq!(jl.semantic, Some(SemanticOp::Call));
        }
        if let Some(ret) = all.iter().find(|e| e.name == "RET") {
            assert_eq!(ret.semantic, Some(SemanticOp::Ret));
        }
        if let Some(done) = all.iter().find(|e| e.name == "DONE") {
            assert_eq!(done.semantic, Some(SemanticOp::Done));
        }
        if let Some(add) = all.iter().find(|e| e.name == "ADD") {
            assert_eq!(add.semantic, Some(SemanticOp::Add));
        }
        if let Some(nopx) = all.iter().find(|e| e.name == "NOPX") {
            assert_eq!(nopx.semantic, Some(SemanticOp::Nop));
        }
        if let Some(paddb) = all.iter().find(|e| e.name == "PADDB_ldb_ptr_inc_nrm_imm") {
            assert_eq!(paddb.semantic, Some(SemanticOp::PointerAdd));
            assert!(paddb.is_ptr_arithmetic);
        }

        let total_with_semantic = all.iter().filter(|e| e.semantic.is_some()).count();
        assert_eq!(
            total_with_semantic,
            all.len(),
            "Expected 100% semantic coverage ({} missing)",
            all.len() - total_with_semantic
        );
    }

    /// Processor scheduling model matches AIE2SchedModel expectations.
    #[test]
    fn test_processor_model() {
        let output = load_from_generated();
        let model = output.processor_model.expect("Should have AIE2SchedModel");
        assert_eq!(model.load_latency, 5);
        assert_eq!(model.mispredict_penalty, 4);
        assert_eq!(model.high_latency, 37);
        assert_eq!(model.issue_width, 1000);
        assert_eq!(model.itinerary_name, "AIE2Itineraries");
    }

    /// Itinerary data was extracted from TableGen.
    #[test]
    fn test_itinerary_data() {
        let output = load_from_generated();
        let itin = &output.itineraries;
        assert!(!itin.is_empty());
        assert!(itin.contains_key("II_ABS"));
        assert!(itin.contains_key("II_ACQ"));
        if let Some(abs) = itin.get("II_ABS") {
            assert_eq!(abs.total_latency, 1);
            assert_eq!(abs.stages.len(), 1);
        }
    }

    /// Register model was extracted from TableGen.
    #[test]
    fn test_register_model() {
        let output = load_from_generated();
        let model = &output.register_model;
        assert!(model.registers.len() > 50);
        assert!(model.classes.len() > 5);
        if let Some(r0) = model.registers.get("r0") {
            assert_eq!(r0.hw_encoding, 0);
        }
        if let Some(lr) = model.registers.get("lr") {
            assert_eq!(lr.hw_encoding, 39);
        }
        if let Some(er) = model.classes.get("eR") {
            assert_eq!(er.members.len(), 32);
        }
    }

    /// Composite format metadata was extracted from TableGen.
    #[test]
    fn test_composite_formats() {
        let output = load_from_generated();
        let formats = &output.composite_formats;
        assert!(!formats.is_empty());
        if let Some(full) = formats.iter().find(|f| f.total_bytes == 16) {
            assert!(full.slots.len() >= 4);
        }
    }

    /// Pack/unpack encodings carry (OUT, IN) element types from their names,
    /// including the 4-bit operands, in the build-time generated data.
    #[test]
    fn pack_unpack_encodings_label_output_and_input_types() {
        let output = load_from_generated();
        let find = |name: &str| {
            output
                .encodings_by_slot
                .values()
                .flatten()
                .find(|e| e.name == name)
                .unwrap_or_else(|| panic!("no encoding {name}"))
        };
        let types = |name: &str| (find(name).element_type, find(name).from_type);
        assert_eq!(types("VPACK_S8_S16"), (Some(ElementType::Int8), Some(ElementType::Int16)));
        assert_eq!(types("VPACK_D4_D8"), (Some(ElementType::UInt4), Some(ElementType::UInt8)));
        assert_eq!(types("VUNPACK_S8_S4"), (Some(ElementType::Int8), Some(ElementType::Int4)));
        assert_eq!(types("VLDB_UNPACK_S16_S8_ag_idx"), (Some(ElementType::Int16), Some(ElementType::Int8)));
        assert_eq!(types("VLDB_2D_UNPACK_D16_D8"), (Some(ElementType::UInt16), Some(ElementType::UInt8)));
        assert_eq!(types("VST_PACK_S4_S8_ag_idx"), (Some(ElementType::Int4), Some(ElementType::Int8)));
        assert_eq!(types("VST_3D_PACK_D8_D16"), (Some(ElementType::UInt8), Some(ElementType::UInt16)));
    }
}
