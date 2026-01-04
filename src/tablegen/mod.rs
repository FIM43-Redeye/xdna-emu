//! TableGen parser module for extracting instruction definitions.
//!
//! This module parses LLVM TableGen (.td) files from the llvm-aie project
//! to automatically generate instruction decoder tables and semantic info.
//!
//! # Overview
//!
//! The AIE2 instruction set is defined in TableGen files:
//! - `AIE2Slots.td` - VLIW slot definitions (lda, ldb, alu, mv, st, vec, lng)
//! - `AIE2GenInstrFormats.td` - Instruction format classes with encoding patterns
//! - `AIE2GenInstrInfo.td` - Concrete instruction definitions
//! - `AIE2InstrPatterns.td` - SDNode patterns mapping ops to instructions
//!
//! # Semantic Information
//!
//! Beyond decoding, we extract:
//! - **Attributes**: `mayLoad`, `mayStore`, `hasSideEffects`, `Defs`, `Uses`
//! - **Patterns**: `Pat<(add ...), (ADD ...)>` tells us ADD performs addition
//!
//! This enables auto-generating execution logic, not just decoding.
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::tablegen::{load_from_llvm_aie, SemanticOp};
//!
//! let data = load_from_llvm_aie("../llvm-aie")?;
//!
//! // Find what ADD does
//! if let Some(pattern) = data.semantic_for_instruction("ADD") {
//!     println!("ADD performs: {:?}", pattern.operation); // SemanticOp::Add
//! }
//!
//! // Find all instructions that implement addition
//! for pattern in data.instructions_for_semantic(SemanticOp::Add) {
//!     println!("Addition: {}", pattern.instruction);
//! }
//! ```
//!
//! # File Locations in llvm-aie
//!
//! ```text
//! llvm-aie/llvm/lib/Target/AIE/
//! ├── AIE2Slots.td           # Slot definitions
//! ├── AIE2GenInstrFormats.td # Format classes
//! ├── AIE2GenInstrInfo.td    # Instruction defs
//! └── AIE2InstrPatterns.td   # Semantic patterns
//! ```

mod parser;
mod resolver;
mod types;

pub use parser::{
    parse_format_classes, parse_instructions, parse_patterns, parse_slots, parse_tablegen_files,
    parse_tablegen_files_with_patterns, ParseError,
};
pub use resolver::{
    build_decoder_tables, DecoderIndex, InstrEncoding, OperandField, ResolveError, Resolver,
    SlotIndex, SlotIndexStats,
};
pub use types::{
    EncodingPart, FormatClass, InstrAttributes, InstrDef, OperandDef, SemanticOp, SemanticPattern,
    SlotDef, TableGenData, TemplateParam,
};

use std::path::Path;

/// Load TableGen data from an llvm-aie repository clone.
///
/// This loads slots, format classes, instruction definitions, and semantic patterns.
///
/// # Arguments
///
/// * `llvm_aie_path` - Path to the llvm-aie repository root
///
/// # Example
///
/// ```ignore
/// let data = load_from_llvm_aie("../llvm-aie")?;
/// println!("Loaded {} slots, {} formats, {} instructions, {} patterns",
///     data.slots.len(), data.formats.len(), data.instructions.len(), data.patterns.len());
/// ```
pub fn load_from_llvm_aie(llvm_aie_path: impl AsRef<Path>) -> Result<TableGenData, std::io::Error> {
    let base = llvm_aie_path.as_ref().join("llvm/lib/Target/AIE");

    let slots_content = std::fs::read_to_string(base.join("AIE2Slots.td"))?;
    let formats_content = std::fs::read_to_string(base.join("AIE2GenInstrFormats.td"))?;

    // Load instruction definitions from all files
    // - AIE2GenInstrInfo.td: Generated instruction definitions
    // - AIE2InstrInfo.td: Main instruction definitions (NOPV, etc.)
    // - AIE2GenFixupInstrInfo.td: Fixup/vector instructions (VMOV, VADD, etc.)
    let gen_instrs = std::fs::read_to_string(base.join("AIE2GenInstrInfo.td"))?;
    let main_instrs = std::fs::read_to_string(base.join("AIE2InstrInfo.td"))
        .unwrap_or_default();
    let fixup_instrs = std::fs::read_to_string(base.join("AIE2GenFixupInstrInfo.td"))
        .unwrap_or_default();
    let instrs_content = format!("{}\n{}\n{}", gen_instrs, main_instrs, fixup_instrs);

    // Try to load patterns file (may not exist in all versions)
    let patterns_content = std::fs::read_to_string(base.join("AIE2InstrPatterns.td")).ok();

    Ok(parse_tablegen_files_with_patterns(
        &slots_content,
        &formats_content,
        &instrs_content,
        patterns_content.as_deref(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test: parse the actual llvm-aie files if available.
    #[test]
    fn test_load_from_llvm_aie() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        let data = load_from_llvm_aie(llvm_aie_path).expect("Failed to parse llvm-aie files");

        // Verify we got reasonable counts
        assert!(!data.slots.is_empty(), "Should have parsed some slots");
        assert!(
            !data.formats.is_empty(),
            "Should have parsed some format classes"
        );
        assert!(
            !data.instructions.is_empty(),
            "Should have parsed some instructions"
        );

        // Print summary for debugging
        eprintln!(
            "Parsed: {} slots, {} formats, {} instructions, {} patterns",
            data.slots.len(),
            data.formats.len(),
            data.instructions.len(),
            data.patterns.len()
        );

        // Verify known slots exist
        assert!(data.slots.contains_key("lda_slot"), "Should have lda_slot");
        assert!(data.slots.contains_key("alu_slot"), "Should have alu_slot");
        assert!(data.slots.contains_key("vec_slot"), "Should have vec_slot");

        // Verify slot properties
        let alu = data.slots.get("alu_slot").unwrap();
        assert_eq!(alu.bits, 20);
        assert_eq!(alu.field, "alu");

        let lda = data.slots.get("lda_slot").unwrap();
        assert_eq!(lda.bits, 21);

        // Check some known format classes
        if let Some(format) = data.formats.get("AIE2_add_r_ri_inst_alu") {
            assert_eq!(format.slot_field, Some("alu".to_string()));
            assert!(!format.encoding.is_empty());
        }

        // Check some known instructions
        let add_instrs: Vec<_> = data
            .instructions
            .keys()
            .filter(|k| k.starts_with("ADD"))
            .collect();
        eprintln!("Found {} ADD* instructions: {:?}", add_instrs.len(), add_instrs);
        assert!(!add_instrs.is_empty(), "Should have ADD instructions");

        // Check ADD_NC specifically
        if let Some(add_nc) = data.instructions.get("ADD_NC") {
            eprintln!("ADD_NC: format={}, mnemonic={}", add_nc.format, add_nc.mnemonic);
        }
    }

    #[test]
    fn test_slot_bits() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            return;
        }

        let data = load_from_llvm_aie(llvm_aie_path).unwrap();

        // Verify expected bit widths from AIE2Slots.td
        let expected = [
            ("lda_slot", 21),
            ("ldb_slot", 16),
            ("alu_slot", 20),
            ("mv_slot", 22),
            ("st_slot", 21),
            ("vec_slot", 26),
            ("lng_slot", 42),
            ("nop_slot", 1),
        ];

        for (name, bits) in expected {
            let slot = data
                .slots
                .get(name)
                .unwrap_or_else(|| panic!("Missing slot: {}", name));
            assert_eq!(slot.bits, bits, "Wrong bit width for {}", name);
        }
    }

    #[test]
    fn test_instruction_attributes() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            return;
        }

        let data = load_from_llvm_aie(llvm_aie_path).unwrap();

        // Count instructions with various attributes
        let with_defs: Vec<_> = data
            .instructions
            .values()
            .filter(|i| !i.attributes.defs.is_empty())
            .collect();
        let with_may_load: Vec<_> = data
            .instructions
            .values()
            .filter(|i| i.attributes.may_load)
            .collect();
        let with_may_store: Vec<_> = data
            .instructions
            .values()
            .filter(|i| i.attributes.may_store)
            .collect();

        eprintln!(
            "Attributes: {} with Defs, {} mayLoad, {} mayStore",
            with_defs.len(),
            with_may_load.len(),
            with_may_store.len()
        );

        // We should have some instructions with each attribute
        // (the actual counts depend on the llvm-aie version)
    }

    #[test]
    fn test_semantic_patterns() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            return;
        }

        let data = load_from_llvm_aie(llvm_aie_path).unwrap();

        eprintln!("Found {} semantic patterns", data.patterns.len());

        // Check that we found some common operations
        let add_patterns = data.instructions_for_semantic(SemanticOp::Add);
        let sub_patterns = data.instructions_for_semantic(SemanticOp::Sub);
        let and_patterns = data.instructions_for_semantic(SemanticOp::And);

        eprintln!(
            "Patterns: {} Add, {} Sub, {} And",
            add_patterns.len(),
            sub_patterns.len(),
            and_patterns.len()
        );

        // Print some pattern details
        for pattern in data.patterns.iter().take(10) {
            eprintln!("  {:?} -> {}", pattern.operation, pattern.instruction);
        }

        // We should have found at least the basic ALU operations
        assert!(!data.patterns.is_empty(), "Should have found some patterns");
    }

    #[test]
    fn test_resolve_real_instructions() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        let data = load_from_llvm_aie(llvm_aie_path).unwrap();
        let resolver = Resolver::new(&data);

        // Resolve all instructions
        let results: Vec<_> = resolver.resolve_all();
        let success_count = results.iter().filter(|r| r.is_ok()).count();
        let error_count = results.iter().filter(|r| r.is_err()).count();

        eprintln!(
            "Resolved: {} success, {} errors (out of {} instructions)",
            success_count,
            error_count,
            data.instructions.len()
        );

        // Print some successful encodings
        let encodings = resolver.resolve_all_ok();
        for enc in encodings.iter().take(10) {
            eprintln!(
                "  {} ({}): mask=0x{:X}, bits=0x{:X}, fields={:?}",
                enc.name,
                enc.slot,
                enc.fixed_mask,
                enc.fixed_bits,
                enc.operand_fields.iter().map(|f| &f.name).collect::<Vec<_>>()
            );
        }

        // Group by slot
        let by_slot = resolver.resolve_by_slot();
        eprintln!("Instructions by slot:");
        for (slot, instrs) in &by_slot {
            eprintln!("  {}: {} instructions", slot, instrs.len());
        }

        // Check ADD_NC specifically
        if let Some(add_nc) = encodings.iter().find(|e| e.name == "ADD_NC") {
            eprintln!(
                "ADD_NC: slot={}, mask=0x{:X}, bits=0x{:X}, mnemonic={}, fields={:?}",
                add_nc.slot,
                add_nc.fixed_mask,
                add_nc.fixed_bits,
                add_nc.mnemonic,
                add_nc.operand_fields.iter().map(|f| (&f.name, f.bit_position, f.width)).collect::<Vec<_>>()
            );
        } else {
            eprintln!("ADD_NC not found in resolved encodings!");
        }

        // Check the format class for ADD_NC
        if let Some(format) = data.formats.get("AIE2_mv_add_inst_mv") {
            eprintln!(
                "AIE2_mv_add_inst_mv format: slot_field={:?}, encoding={:?}",
                format.slot_field,
                format.encoding
            );
        } else {
            eprintln!("AIE2_mv_add_inst_mv format not found!");
        }

        // Check vec format classes
        let vec_formats: Vec<_> = data.formats.keys().filter(|k| k.contains("_vec")).collect();
        eprintln!("Vec format classes: {} total", vec_formats.len());
        for name in vec_formats.iter().take(5) {
            if let Some(format) = data.formats.get(*name) {
                eprintln!("  {}: slot_field={:?}, encoding.len={}", name, format.slot_field, format.encoding.len());
            }
        }

        // Check NOPV encoding
        if let Some(nopv) = encodings.iter().find(|e| e.name == "NOPV") {
            eprintln!(
                "NOPV: slot={}, mask=0x{:X}, bits=0x{:X}",
                nopv.slot, nopv.fixed_mask, nopv.fixed_bits
            );
        } else {
            eprintln!("NOPV not found in resolved encodings!");
        }

        // Check SBC specifically - debug why it's not being resolved
        if let Some(sbc) = data.instructions.get("SBC") {
            eprintln!("SBC instruction: format={}, template_args={:?}", sbc.format, sbc.template_args);
            if let Some(fmt) = data.formats.get(&sbc.format) {
                eprintln!("  Format found: template_params={:?}", fmt.template_params);
            } else {
                eprintln!("  Format NOT FOUND: {}", sbc.format);
                // List similar format names
                let similar: Vec<_> = data.formats.keys()
                    .filter(|k| k.contains("alu"))
                    .take(10)
                    .collect();
                eprintln!("  Similar formats: {:?}", similar);
            }
        } else {
            eprintln!("SBC instruction NOT FOUND in parsed instructions!");
        }

        // Check for MOVXM_lng_cg instruction
        if let Some(movxm) = data.instructions.get("MOVXM_lng_cg") {
            eprintln!("MOVXM_lng_cg instruction found: format={}, mnemonic={}",
                     movxm.format, movxm.mnemonic);
            if let Some(enc) = encodings.iter().find(|e| e.name == "MOVXM_lng_cg") {
                eprintln!("  RESOLVED: slot={}, mask=0x{:X}, bits=0x{:X}",
                         enc.slot, enc.fixed_mask, enc.fixed_bits);
                eprintln!("  fields: {:?}",
                         enc.operand_fields.iter().map(|f| (&f.name, f.bit_position, f.width)).collect::<Vec<_>>());
            } else {
                eprintln!("  MOVXM_lng_cg NOT in resolved encodings - resolution must have failed");
                // Try to resolve it directly to see the error
                if let Err(e) = resolver.resolve_instruction(movxm) {
                    eprintln!("  Resolution error: {:?}", e);
                }
            }
        } else {
            eprintln!("MOVXM_lng_cg NOT FOUND in parsed instructions!");
            // List all instruction names containing 'mov'
            let mov_instrs: Vec<_> = data.instructions.keys()
                .filter(|k| k.to_lowercase().contains("mov"))
                .take(20)
                .collect();
            eprintln!("  MOV-related instructions: {:?}", mov_instrs);
        }

        // We should have resolved at least some instructions
        assert!(success_count > 0, "Should resolve some instructions");
    }
}
