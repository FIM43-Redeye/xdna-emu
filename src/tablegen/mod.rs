//! TableGen instruction model -- build-time extracted, runtime consumed.
//!
//! All instruction encodings, scheduling models, register definitions, and
//! decoder bytecode tables are extracted from llvm-aie at compile time by
//! `build_helpers/` and embedded as Rust constants in `gen_tablegen.rs`.
//!
//! At runtime, [`load_from_generated`] constructs the full `TblgenOutput`
//! from these constants. No llvm-aie, no filesystem access, no subprocess.
//!
//! The `resolver` module still provides the runtime types (`InstrEncoding`,
//! `OperandField`, etc.) and the `decoder_bytecode` module provides the
//! bytecode interpreter. Both are consumed by the instruction decoder.

pub mod decoder_bytecode;
pub mod decoder_ffi;
mod resolver;
mod types;

// Build-time generated instruction tables (from build_helpers -> gen_tablegen.rs)
mod generated {
    include!(concat!(env!("OUT_DIR"), "/gen_tablegen.rs"));
}

/// Load the complete TableGen model from build-time generated constants.
///
/// This is the sole entry point for instruction decoder data. All 600+
/// instruction encodings, decoder bytecode, scheduling model, register
/// definitions, and composite format layouts are compiled in.
pub fn load_from_generated() -> types::TblgenOutput {
    generated::load_from_generated()
}

pub use resolver::{
    build_decoder_tables, AddressingMode, CompositeEncoder, DecoderIndex, InstrEncoding,
    InstrMemWidth, OperandField, OperandType, RegisterKind, ResolveError, Resolver, SlotIndex,
    classify_operand_type, detect_addressing_mode, detect_mem_width,
    infer_branch_condition, infer_dual_element_types, infer_element_type, infer_select_variant,
    refine_branch_semantic,
};
pub use types::{
    BranchCondition, CompositeFormatDef, ElementType, EncodingPart, ImplicitReg,
    InstrAttributes, ItineraryInfo, OperandDef, PipelineStage, ProcessorModel,
    RegisterClassDef, RegisterDef, RegisterModel, SelectVariant, SemanticOp, SemanticPattern,
    SlotBitMap, TblgenOutput,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that ACQ_mLockId_imm and ACQ_mLockId_reg have distinguishing fixed bits
    /// in the build-time generated data.
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

    /// Lock ID fields must be classified as LockId (unsigned), not signed immediate.
    #[test]
    fn test_acq_lock_id_field_type() {
        let output = load_from_generated();
        let alu = output.encodings_by_slot.get("alu").expect("No alu slot");
        let enc = alu.iter().find(|e| e.name == "ACQ_mLockId_imm")
            .expect("ACQ_mLockId_imm should exist");
        for f in &enc.operand_fields {
            eprintln!("ACQ field: name={} width={} type={:?} signed={}",
                f.name, f.width, f.operand_type, f.signed);
        }
        let id_field = enc.operand_fields.iter().find(|f| f.name == "id")
            .expect("ACQ_mLockId_imm should have 'id' field");
        assert_eq!(id_field.operand_type, OperandType::LockId,
            "Lock ID field should be LockId, not {:?}", id_field.operand_type);
    }

    /// Verify structural semantic inference in the generated data.
    #[test]
    fn test_structural_semantic_inference() {
        let output = load_from_generated();
        let all: Vec<_> = output.encodings_by_slot.values().flatten().collect();

        // Control flow
        if let Some(jl) = all.iter().find(|e| e.name == "JL") {
            assert_eq!(jl.semantic, Some(SemanticOp::Call));
        }
        if let Some(ret) = all.iter().find(|e| e.name == "RET") {
            assert_eq!(ret.semantic, Some(SemanticOp::Ret));
        }
        if let Some(done) = all.iter().find(|e| e.name == "DONE") {
            assert_eq!(done.semantic, Some(SemanticOp::Done));
        }

        // Memory
        if let Some(add) = all.iter().find(|e| e.name == "ADD") {
            assert_eq!(add.semantic, Some(SemanticOp::Add));
        }

        // Move/NOP
        if let Some(nopx) = all.iter().find(|e| e.name == "NOPX") {
            assert_eq!(nopx.semantic, Some(SemanticOp::Nop));
        }

        // Pointer arithmetic via pseudo expansion
        if let Some(paddb) = all.iter().find(|e| e.name == "PADDB_ldb_ptr_inc_nrm_imm") {
            assert_eq!(paddb.semantic, Some(SemanticOp::PointerAdd));
            assert!(paddb.is_ptr_arithmetic);
        }

        // 100% semantic coverage
        let total_with_semantic = all.iter().filter(|e| e.semantic.is_some()).count();
        assert_eq!(
            total_with_semantic,
            all.len(),
            "Expected 100% semantic coverage ({} missing)",
            all.len() - total_with_semantic,
        );
    }

    /// Verify processor scheduling model from generated data.
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

    /// Verify itinerary data from generated data.
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

    /// Verify register model from generated data.
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

    /// Verify composite formats from generated data.
    #[test]
    fn test_composite_formats() {
        let output = load_from_generated();
        let formats = &output.composite_formats;

        assert!(!formats.is_empty());

        if let Some(full) = formats.iter().find(|f| f.total_bytes == 16) {
            assert!(full.slots.len() >= 4);
        }
    }
}
