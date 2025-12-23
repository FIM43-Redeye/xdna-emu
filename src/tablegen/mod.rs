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
    build_decoder_tables, InstrEncoding, OperandField, ResolveError, Resolver,
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
    let instrs_content = std::fs::read_to_string(base.join("AIE2GenInstrInfo.td"))?;

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
        let add_count = data
            .instructions
            .keys()
            .filter(|k| k.starts_with("ADD"))
            .count();
        eprintln!("Found {} ADD* instructions", add_count);
        assert!(add_count > 0, "Should have ADD instructions");
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

        // We should have resolved at least some instructions
        assert!(success_count > 0, "Should resolve some instructions");
    }
}
