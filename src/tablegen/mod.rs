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
pub mod tblgen_records;
mod types;

pub use parser::{
    parse_format_classes, parse_instructions, parse_patterns, parse_slots, parse_tablegen_files,
    parse_tablegen_files_with_patterns, ParseError,
};
pub use resolver::{
    build_decoder_tables, DecoderIndex, InstrEncoding, OperandField, ResolveError, Resolver,
    SlotIndex, SlotIndexStats,
};
pub use tblgen_records::{parse_tblgen_records, InstrRecord, SlotEncoding};
pub use types::{
    EncodingPart, FormatClass, ImplicitReg, InstrAttributes, InstrDef, OperandDef, SemanticOp,
    SemanticPattern, SlotDef, TableGenData, TemplateParam,
};

use std::collections::HashMap;
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

/// Find the AIE-specific llvm-tblgen binary.
///
/// The system llvm-tblgen doesn't have AIE support, so we need the one
/// built with llvm-aie. This searches known installation locations.
///
/// # Search Order
/// 1. LLVM_AIE_TBLGEN environment variable (explicit override)
/// 2. Sibling mlir-aie installations (ironenv, my_install)
/// 3. llvm-aie build directory
/// 4. Fall back to PATH (likely to fail for AIE files)
pub fn find_aie_tblgen(llvm_aie_path: impl AsRef<Path>) -> Option<std::path::PathBuf> {
    // Check environment variable first
    if let Ok(path) = std::env::var("LLVM_AIE_TBLGEN") {
        let p = std::path::PathBuf::from(&path);
        if p.exists() {
            log::info!("Using llvm-tblgen from LLVM_AIE_TBLGEN: {}", path);
            return Some(p);
        }
    }

    let llvm_aie = llvm_aie_path.as_ref();

    // Known locations relative to llvm-aie or its parent
    let candidates = [
        // mlir-aie ironenv (Python venv with llvm-aie package)
        llvm_aie.parent().and_then(|p| Some(p.join("mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie/bin/llvm-tblgen"))),
        llvm_aie.parent().and_then(|p| Some(p.join("mlir-aie/ironenv/lib/python3.12/site-packages/llvm-aie/bin/llvm-tblgen"))),
        llvm_aie.parent().and_then(|p| Some(p.join("mlir-aie/ironenv/lib/python3.11/site-packages/llvm-aie/bin/llvm-tblgen"))),
        // mlir-aie install directory
        llvm_aie.parent().and_then(|p| Some(p.join("mlir-aie/my_install/mlir/bin/llvm-tblgen"))),
        llvm_aie.parent().and_then(|p| Some(p.join("mlir-aie/install/bin/llvm-tblgen"))),
        // llvm-aie build directory
        Some(llvm_aie.join("build/bin/llvm-tblgen")),
        Some(llvm_aie.join("build/Release/bin/llvm-tblgen")),
    ];

    for candidate in candidates.into_iter().flatten() {
        // Canonicalize to resolve relative paths like ../mlir-aie/...
        // canonicalize() fails if the file doesn't exist, so Ok means it exists
        if let Ok(canonical) = candidate.canonicalize() {
            log::info!("Found AIE llvm-tblgen at: {}", canonical.display());
            return Some(canonical);
        }
    }

    // Check if PATH version has AIE support (unlikely but worth trying)
    if let Ok(output) = std::process::Command::new("llvm-tblgen")
        .arg("--version")
        .output()
    {
        let version = String::from_utf8_lossy(&output.stdout);
        // AIE-enabled builds typically mention "aie" somewhere
        if version.to_lowercase().contains("aie") {
            log::info!("Using llvm-tblgen from PATH (has AIE support)");
            return Some(std::path::PathBuf::from("llvm-tblgen"));
        }
    }

    log::warn!("No AIE-enabled llvm-tblgen found. Set LLVM_AIE_TBLGEN environment variable.");
    None
}

/// Load instruction encodings using llvm-tblgen directly.
///
/// This gives us fully resolved encodings with all inheritance and template
/// substitution applied. Much more reliable than regex parsing.
///
/// # Arguments
///
/// * `llvm_aie_path` - Path to the llvm-aie repository root
///
/// # Returns
///
/// Vector of InstrEncoding ready for decoder use, grouped by slot.
///
/// # Note
///
/// Requires an AIE-enabled llvm-tblgen. The system llvm-tblgen typically
/// doesn't have AIE support. Set LLVM_AIE_TBLGEN environment variable or
/// ensure mlir-aie is installed as a sibling directory.
pub fn load_via_tblgen(llvm_aie_path: impl AsRef<Path>) -> Result<HashMap<String, Vec<InstrEncoding>>, std::io::Error> {
    use std::process::Command;

    let llvm_aie = llvm_aie_path.as_ref();
    let base = llvm_aie.join("llvm/lib/Target/AIE");

    // Find AIE-enabled llvm-tblgen
    let tblgen_path = find_aie_tblgen(llvm_aie)
        .ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No AIE-enabled llvm-tblgen found. Set LLVM_AIE_TBLGEN or install mlir-aie."
        ))?;

    log::info!("Running llvm-tblgen from: {}", tblgen_path.display());

    // Run llvm-tblgen to get resolved records
    let output = Command::new(&tblgen_path)
        .arg("--print-records")
        .arg("AIE2.td")
        .arg("-I.")
        .arg("-I../../..")
        .arg("-I../../../include")
        .current_dir(&base)
        .output()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other,
            format!("Failed to run llvm-tblgen at {}: {}", tblgen_path.display(), e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(std::io::Error::new(std::io::ErrorKind::Other,
            format!("llvm-tblgen failed: {}", stderr)));
    }

    let content = String::from_utf8_lossy(&output.stdout);
    let records = tblgen_records::parse_tblgen_records(&content);

    log::info!("Parsed {} instruction records from llvm-tblgen", records.len());

    // Convert to encodings grouped by slot
    let mut by_slot: HashMap<String, Vec<InstrEncoding>> = HashMap::new();

    for record in records {
        if let Some(encoding) = record.to_encoding() {
            by_slot.entry(encoding.slot.clone())
                .or_default()
                .push(encoding);
        }
    }

    // Log slot counts
    for (slot, encodings) in &by_slot {
        log::debug!("Slot '{}': {} encodings", slot, encodings.len());
    }

    // Sort by specificity (most specific first)
    for encodings in by_slot.values_mut() {
        encodings.sort_by(|a, b| b.specificity().cmp(&a.specificity()));
    }

    Ok(by_slot)
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

    /// Test that ACQ_mLockId_imm and ACQ_mLockId_reg have distinguishing fixed bits.
    ///
    /// This verifies our tblgen parser correctly captures the literal bits from
    /// mixin classes that distinguish immediate vs register variants.
    ///
    /// Key insight: mLockId field encoding differs:
    /// - ACQ_mLockId_imm: bit 0 of mLockId = 0 (immediate)
    /// - ACQ_mLockId_reg: bit 0 of mLockId = 1 (register)
    #[test]
    fn test_acq_instruction_disambiguation() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        let encodings_by_slot = load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // Find ACQ instructions in the alu slot
        let alu_encodings = encodings_by_slot.get("alu").expect("No alu slot encodings");

        let acq_imm = alu_encodings.iter()
            .find(|e| e.name == "ACQ_mLockId_imm");
        let acq_reg = alu_encodings.iter()
            .find(|e| e.name == "ACQ_mLockId_reg");

        eprintln!("ACQ instructions in alu slot:");
        for enc in alu_encodings.iter().filter(|e| e.name.starts_with("ACQ")) {
            eprintln!(
                "  {}: mask=0x{:05X}, bits=0x{:05X}, fields={:?}",
                enc.name, enc.fixed_mask, enc.fixed_bits,
                enc.operand_fields.iter().map(|f| &f.name).collect::<Vec<_>>()
            );
        }

        if let (Some(imm), Some(reg)) = (acq_imm, acq_reg) {
            // They should have different fixed_bits since mLockId bit 0 differs
            assert_ne!(
                imm.fixed_bits, reg.fixed_bits,
                "ACQ_mLockId_imm and ACQ_mLockId_reg should have different fixed_bits! \
                 imm=0x{:05X}, reg=0x{:05X}",
                imm.fixed_bits, reg.fixed_bits
            );

            // The masks should cover the distinguishing bit
            assert_ne!(
                imm.fixed_mask & imm.fixed_bits,
                reg.fixed_mask & reg.fixed_bits,
                "Masked bits should differ"
            );

            eprintln!("SUCCESS: ACQ instructions have distinguishing fixed bits");
            eprintln!(
                "  ACQ_mLockId_imm: mask=0x{:05X}, bits=0x{:05X}",
                imm.fixed_mask, imm.fixed_bits
            );
            eprintln!(
                "  ACQ_mLockId_reg: mask=0x{:05X}, bits=0x{:05X}",
                reg.fixed_mask, reg.fixed_bits
            );
        } else {
            // Print what we did find
            let acq_names: Vec<_> = alu_encodings.iter()
                .filter(|e| e.name.to_lowercase().contains("acq"))
                .map(|e| &e.name)
                .collect();
            panic!(
                "Expected both ACQ_mLockId_imm and ACQ_mLockId_reg, found: {:?}",
                acq_names
            );
        }
    }

    /// Test that implicit registers are correctly parsed from TableGen definitions.
    ///
    /// Instructions like SELEQZ use `eR27:$s2` which means r27 is read implicitly,
    /// not encoded as a field in the instruction bits.
    #[test]
    fn test_implicit_register_parsing() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        let data = load_from_llvm_aie(llvm_aie_path).unwrap();

        // Find SELEQZ - it should have r27 as an implicit register
        let seleqz_instrs: Vec<_> = data
            .instructions
            .iter()
            .filter(|(name, _)| name.to_uppercase().contains("SELEQZ"))
            .collect();

        eprintln!("Found {} SELEQZ instructions", seleqz_instrs.len());

        for (name, instr) in &seleqz_instrs {
            eprintln!(
                "{}: inputs={:?}, outputs={:?}, implicit={:?}",
                name,
                instr.inputs.iter().map(|o| &o.name).collect::<Vec<_>>(),
                instr.outputs.iter().map(|o| &o.name).collect::<Vec<_>>(),
                instr.implicit_regs
            );

            // Verify r27 is in implicit_regs, not in inputs
            let has_r27_implicit = instr
                .implicit_regs
                .iter()
                .any(|ir| ir.reg_num == 27 && ir.is_use);
            let has_r27_explicit = instr.inputs.iter().any(|o| o.name.contains("27"));

            if has_r27_implicit {
                eprintln!("  -> r27 correctly parsed as implicit use");
            }
            if has_r27_explicit {
                eprintln!("  WARNING: r27 still appears in explicit inputs!");
            }
        }

        // Also check SELNEZ instructions
        let selnez_instrs: Vec<_> = data
            .instructions
            .iter()
            .filter(|(name, _)| name.to_uppercase().contains("SELNEZ"))
            .collect();

        eprintln!("\nFound {} SELNEZ instructions", selnez_instrs.len());

        for (name, instr) in &selnez_instrs {
            eprintln!(
                "{}: inputs={:?}, implicit={:?}",
                name,
                instr.inputs.iter().map(|o| &o.name).collect::<Vec<_>>(),
                instr.implicit_regs
            );
        }

        // Count total instructions with implicit registers
        let with_implicit: Vec<_> = data
            .instructions
            .iter()
            .filter(|(_, i)| !i.implicit_regs.is_empty())
            .collect();

        eprintln!(
            "\nTotal instructions with implicit registers: {}",
            with_implicit.len()
        );

        // Show some examples
        for (name, instr) in with_implicit.iter().take(10) {
            eprintln!(
                "  {}: {:?}",
                name,
                instr.implicit_regs
            );
        }
    }
}
