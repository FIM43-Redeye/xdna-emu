//! TableGen instruction extraction via tblgen-rs (LLVM native binding).
//!
//! This module extracts instruction encodings, scheduling models, register
//! definitions, and semantic information from llvm-aie's TableGen files
//! using the `tblgen` crate (linked against LLVM's TableGen library).
//!
//! The only subprocess used is `llvm-tblgen -gen-disassembler` for the
//! decoder bytecode tables, with results cached to disk.
//!
//! Architecture constants (slot widths, register sizes) are extracted at
//! compile time by `xdna-archspec` -- see `crates/xdna-archspec/src/tablegen.rs`.
//! This module handles the runtime decoder table loading.
//!
//! # Entry Points
//!
//! - [`load_full_via_tblgen`] -- Complete model (encodings + scheduling + registers)
//! - [`load_via_tblgen`] -- Encodings only (legacy convenience wrapper)
//! - [`find_aie_tblgen`] -- Locate the AIE-enabled `llvm-tblgen` binary

mod cpp_switch;
pub mod decoder_bytecode;
pub mod native;
mod resolver;
pub mod tblgen_records;
mod types;

pub use resolver::{
    build_decoder_tables, AddressingMode, CompositeEncoder, DecoderIndex, InstrEncoding,
    InstrMemWidth, OperandField, OperandType, RegisterKind, ResolveError, Resolver, SlotIndex,
    classify_operand_type, detect_addressing_mode, detect_mem_width,
    infer_branch_condition, infer_element_type, infer_select_variant, refine_branch_semantic,
};
pub use tblgen_records::{InstrRecord, SlotEncoding};
pub use types::{
    BranchCondition, CompositeFormatDef, ElementType, EncodingPart, ImplicitReg,
    InstrAttributes, ItineraryInfo, OperandDef, PipelineStage, ProcessorModel,
    RegisterClassDef, RegisterDef, RegisterModel, SelectVariant, SemanticOp, SemanticPattern,
    SlotBitMap, TblgenOutput,
};

use std::collections::HashMap;
use std::path::Path;

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

    // Binary name is platform-dependent
    let exe = if cfg!(target_os = "windows") { "llvm-tblgen.exe" } else { "llvm-tblgen" };

    // Known locations relative to llvm-aie or its parent
    let mut candidates: Vec<Option<std::path::PathBuf>> = vec![
        // mlir-aie ironenv (Python venv with llvm-aie package) -- Linux layout
        llvm_aie.parent().map(|p| p.join(format!("mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie/bin/{exe}"))),
        llvm_aie.parent().map(|p| p.join(format!("mlir-aie/ironenv/lib/python3.12/site-packages/llvm-aie/bin/{exe}"))),
        llvm_aie.parent().map(|p| p.join(format!("mlir-aie/ironenv/lib/python3.11/site-packages/llvm-aie/bin/{exe}"))),
        // mlir-aie install directory
        llvm_aie.parent().map(|p| p.join(format!("mlir-aie/my_install/mlir/bin/{exe}"))),
        llvm_aie.parent().map(|p| p.join(format!("mlir-aie/install/bin/{exe}"))),
        // llvm-aie build directory
        Some(llvm_aie.join(format!("build/bin/{exe}"))),
        Some(llvm_aie.join(format!("build/Release/bin/{exe}"))),
    ];

    // Windows-specific: Python venv uses Lib/site-packages/ (not lib/pythonX.Y/...)
    if cfg!(target_os = "windows") {
        candidates.extend([
            llvm_aie.parent().map(|p| p.join(format!("mlir-aie/ironenv/Lib/site-packages/llvm-aie/bin/{exe}"))),
            llvm_aie.parent().map(|p| p.join(format!("mlir-aie/ironenv/Scripts/{exe}"))),
        ]);
    }

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
        .env_remove("LD_LIBRARY_PATH")
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
/// Load instruction encodings using llvm-tblgen directly (legacy API).
///
/// Returns only instruction encodings grouped by slot. For the full model
/// including scheduling, register, and format data, use [`load_full_via_tblgen`].
pub fn load_via_tblgen(llvm_aie_path: impl AsRef<Path>) -> Result<HashMap<String, Vec<InstrEncoding>>, std::io::Error> {
    let output = load_full_via_tblgen(llvm_aie_path)?;
    Ok(output.encodings_by_slot)
}

/// Load the complete TableGen model.
///
/// Extracts instruction encodings, scheduling model, register definitions,
/// and composite format layouts in-process via the tblgen crate (linked to
/// LLVM's TableGen library). Only the decoder bytecode tables still require
/// a subprocess (`llvm-tblgen -gen-disassembler`), cached to disk.
pub fn load_full_via_tblgen(llvm_aie_path: impl AsRef<Path>) -> Result<types::TblgenOutput, std::io::Error> {
    let llvm_aie = llvm_aie_path.as_ref();
    let base = llvm_aie.join("llvm/lib/Target/AIE");

    // Find AIE-enabled llvm-tblgen (needed only for -gen-disassembler)
    let tblgen_path = find_aie_tblgen(llvm_aie)
        .ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No AIE-enabled llvm-tblgen found. Set LLVM_AIE_TBLGEN or install mlir-aie."
        ))?;

    // Try disk cache for disassembler output
    let cache_key = compute_tblgen_cache_key(&tblgen_path);
    let disasm_text = if let Some(ref key) = cache_key {
        if let Some(cached) = try_load_cached_disasm(key) {
            log::info!("Using cached disassembler output (key: {}...)", &key[..12]);
            cached
        } else {
            let text = run_gen_disassembler(&tblgen_path, &base);
            write_disasm_cache(key, &text);
            text
        }
    } else {
        run_gen_disassembler(&tblgen_path, &base)
    };

    build_tblgen_output(&disasm_text, llvm_aie)
}

/// Run `llvm-tblgen -gen-disassembler` and return stdout text.
///
/// Best-effort: failures produce an empty string (logged as warnings).
fn run_gen_disassembler(tblgen_path: &Path, base: &Path) -> String {
    use std::process::Command;

    log::info!("Running llvm-tblgen -gen-disassembler from: {}", tblgen_path.display());
    let tblgen_args = ["AIE2.td", "-I.", "-I../../..", "-I../../../include"];

    match Command::new(tblgen_path)
        .arg("-gen-disassembler")
        .args(&tblgen_args)
        .current_dir(base)
        .env_remove("LD_LIBRARY_PATH")
        .output()
    {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).into_owned()
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            log::warn!("llvm-tblgen -gen-disassembler failed: {}", stderr);
            String::new()
        }
        Err(e) => {
            log::warn!("Failed to run -gen-disassembler: {}", e);
            String::new()
        }
    }
}

/// Build the complete TableGen model from native extraction + disassembler output.
fn build_tblgen_output(
    disasm_text: &str,
    llvm_aie_path: &Path,
) -> Result<types::TblgenOutput, std::io::Error> {
    let td_file = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2.td");
    let inc_path_bufs = vec![
        llvm_aie_path.join("llvm/include"),
        llvm_aie_path.join("llvm/lib/Target/AIE"),
    ];
    let inc_refs: Vec<&std::path::Path> = inc_path_bufs.iter().map(|p| p.as_path()).collect();

    // Extract instruction records in-process (full 607+ coverage).
    let records = native::load_instruction_records(&td_file, &inc_refs)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other,
            format!("TableGen instruction extraction failed: {}", e)))?;
    log::info!("Native TableGen: {} instruction records", records.len());

    // Convert to encodings grouped by slot
    let mut by_slot: HashMap<String, Vec<InstrEncoding>> = HashMap::new();
    for record in records {
        if let Some(encoding) = record.to_encoding() {
            by_slot.entry(encoding.slot.clone())
                .or_default()
                .push(encoding);
        }
    }

    // Apply Pat<>-derived semantics from fully resolved pattern records.
    let pattern_map = native::load_pattern_records(&td_file, &inc_refs)
        .unwrap_or_else(|e| { log::warn!("Pattern extraction failed: {}", e); HashMap::new() });
    let mut pattern_upgraded = 0usize;
    for encodings in by_slot.values_mut() {
        for enc in encodings.iter_mut() {
            if let Some(&op) = pattern_map.get(enc.name.as_str()) {
                enc.semantic = Some(op);
                pattern_upgraded += 1;
            }
        }
    }
    log::info!(
        "Applied {} pattern-based semantics from {} unique Pat<> records",
        pattern_upgraded,
        pattern_map.len(),
    );

    // Propagate semantics through pseudo -> concrete expansion maps.
    //
    // Pat<> patterns often target pseudo instructions (e.g., PADD_imm9_pseudo)
    // which expand to concrete encodings (e.g., PADDB_ldb_ptr_inc_nrm_imm) via
    // the `materializableInto` field. The concrete encodings have no Pat<> and
    // no structural flags, so they miss semantic assignment above.
    //
    // Build a reverse map: for each pseudo that has a known semantic (from
    // pattern_map), propagate that semantic to all its concrete alternatives.
    let pseudo_map = native::load_pseudo_expansion_map(&td_file, &inc_refs)
        .unwrap_or_else(|e| { log::warn!("Pseudo expansion failed: {}", e); HashMap::new() });
    let mut pseudo_propagated = 0usize;
    // Build concrete_name -> semantic from pseudo expansions
    let mut expansion_semantics: HashMap<String, types::SemanticOp> = HashMap::new();
    for (pseudo_name, concretes) in &pseudo_map {
        if let Some(&op) = pattern_map.get(pseudo_name.as_str()) {
            for concrete in concretes {
                expansion_semantics.insert(concrete.clone(), op);
            }
        }
    }
    // Apply to encodings that still lack a semantic
    for encodings in by_slot.values_mut() {
        for enc in encodings.iter_mut() {
            if enc.semantic.is_none() {
                if let Some(&op) = expansion_semantics.get(enc.name.as_str()) {
                    enc.semantic = Some(op);
                    pseudo_propagated += 1;
                }
            }
        }
    }
    log::info!(
        "Propagated {} semantics via pseudo expansion from {} pseudo records",
        pseudo_propagated,
        pseudo_map.len(),
    );

    // Layer 3: C++ selection propagation.
    //
    // Many instructions are selected via C++ switch statements in
    // AIE2InstrInfo::getOpCode() rather than Pat<> records. Parse those
    // switches to extract intrinsic -> opcode mappings, then propagate
    // semantics from the intrinsic to the concrete opcode.
    let cpp_map = cpp_switch::parse_cpp_opcode_switch(llvm_aie_path);
    let mut cpp_propagated = 0usize;
    for (intrinsic_stem, opcodes) in &cpp_map {
        if let Some(op) = types::SemanticOp::from_intrinsic(
            &format!("int_aie2_{}", intrinsic_stem),
        ) {
            for opcode_name in opcodes {
                for encodings in by_slot.values_mut() {
                    for enc in encodings.iter_mut() {
                        if enc.semantic.is_none() && enc.name == *opcode_name {
                            enc.semantic = Some(op);
                            cpp_propagated += 1;
                        }
                    }
                }
            }
        }
    }
    if cpp_propagated > 0 {
        log::info!(
            "Propagated {} semantics via C++ selection from {} intrinsic mappings",
            cpp_propagated,
            cpp_map.len(),
        );
    }

    // Layer 4: Itinerary-based inference for instructions without Pat<>, pseudo
    // expansion, or C++ selection. The scheduling model's Itinerary field groups
    // instructions by functional family, providing structural classification from
    // LLVM that doesn't depend on mnemonic string matching.
    //
    // Entries are checked as prefixes (starts_with), sorted longest-first to
    // ensure more specific prefixes match before less specific ones.
    const ITINERARY_SEMANTICS: &[(&str, types::SemanticOp)] = &[
        // Pointer arithmetic (2D/3D variants, stack frame adjustment)
        ("II_PADD", types::SemanticOp::PointerAdd),
        // Store half-byte variants (dual mayLoad+mayStore, skipped by structural)
        ("II_STHB", types::SemanticOp::Store),
        // Stream write: push to master stream (packet header and cascade variants)
        ("II_MOV_CPH", types::SemanticOp::StreamWritePacketHeader),
        ("II_MOV_PH", types::SemanticOp::StreamWritePacketHeader),
        ("II_ST_MS", types::SemanticOp::StreamWrite),
        // Stream read: slave stream -> scalar
        ("II_MOV_SS", types::SemanticOp::StreamRead),
        // Vector cascade stream access
        ("II_VMOV_CASCADE_READ", types::SemanticOp::CascadeRead),
        ("II_VMOV_CASCADE_WRITE", types::SemanticOp::CascadeWrite),
        // Vector move/copy
        ("II_VMOV", types::SemanticOp::Copy),
        // Pack/unpack (backup if C++ parse misses them)
        ("II_VPACK", types::SemanticOp::Pack),
        ("II_VUNPACK", types::SemanticOp::Unpack),
        // Vector sub-register insert
        ("II_VPUSH_HI", types::SemanticOp::VectorInsert),
        ("II_VPUSH_LO", types::SemanticOp::VectorInsert),
        // Masked SRS variants
        ("II_VSRSM", types::SemanticOp::Srs),
        // Extract-broadcast
        ("II_VEXTBCST", types::SemanticOp::VectorBroadcast),
        // Carry arithmetic
        ("II_ADC", types::SemanticOp::Adc),
        ("II_SBC", types::SemanticOp::Sbc),
        // Add no-carry variants
        ("II_ADD_NC", types::SemanticOp::Add),
        // Delay slot moves
        ("II_MOVd", types::SemanticOp::Copy),
        // Counter move
        ("II_MOV_CNTR", types::SemanticOp::Copy),
        // Float conversion (vfloor)
        ("II_VFLOORs32bf16", types::SemanticOp::Convert),
        // Division
        ("II_DIVS", types::SemanticOp::SDiv),
        // Vector add-subtract compound operations
        ("II_VADDSUB", types::SemanticOp::Add),
        // Sign/zero extension
        ("II_EXTENDs", types::SemanticOp::SignExtend),
        ("II_EXTENDu", types::SemanticOp::ZeroExtend),
    ];

    let mut itinerary_inferred = 0usize;
    for encodings in by_slot.values_mut() {
        for enc in encodings.iter_mut() {
            if enc.semantic.is_none() {
                if let Some(ref sched) = enc.sched_class {
                    for &(prefix, op) in ITINERARY_SEMANTICS {
                        if sched.starts_with(prefix) {
                            enc.semantic = Some(op);
                            itinerary_inferred += 1;
                            break;
                        }
                    }
                }
            }
        }
    }
    if itinerary_inferred > 0 {
        log::info!(
            "Inferred {} semantics from itinerary class grouping",
            itinerary_inferred,
        );
    }

    // Derive is_ptr_arithmetic from resolved semantics. This ensures the flag
    // is set for instructions whose PointerAdd semantic came from pseudo
    // expansion propagation or itinerary inference (both run after to_encoding()).
    for encodings in by_slot.values_mut() {
        for enc in encodings.iter_mut() {
            if enc.semantic == Some(types::SemanticOp::PointerAdd) {
                enc.is_ptr_arithmetic = true;
            }
        }
    }

    // Log slot counts
    for (slot, encodings) in &by_slot {
        log::debug!("Slot '{}': {} encodings", slot, encodings.len());
    }

    // Extract extended data
    let processor_model = native::load_processor_model(&td_file, &inc_refs)
        .unwrap_or_else(|e| { log::warn!("Processor model extraction failed: {}", e); None });
    let itineraries = native::load_itinerary_data(&td_file, &inc_refs)
        .unwrap_or_else(|e| { log::warn!("Itinerary extraction failed: {}", e); HashMap::new() });
    let register_model = native::load_register_model(&td_file, &inc_refs)
        .unwrap_or_else(|e| { log::warn!("Register model extraction failed: {}", e); Default::default() });

    // Composite VLIW bundle format definitions (fully resolved slot maps).
    let composite_formats = native::load_composite_formats(&td_file, &inc_refs)
        .unwrap_or_else(|e| { log::warn!("Composite format extraction failed: {}", e); Vec::new() });

    log::info!(
        "Extended data: model={}, {} itineraries, {} registers, {} classes, {} formats",
        processor_model.is_some(),
        itineraries.len(),
        register_model.registers.len(),
        register_model.classes.len(),
        composite_formats.len(),
    );

    // Parse decoder bytecode tables (empty string -> empty map)
    let decoder_tables = if disasm_text.is_empty() {
        log::warn!("No disassembler output available, falling back to heuristic disambiguation");
        HashMap::new()
    } else {
        let tables = decoder_bytecode::extract_all_tables(disasm_text);
        log::info!(
            "Extracted {} LLVM decoder tables ({} total bytes)",
            tables.len(),
            tables.values().map(|t| t.byte_count()).sum::<usize>(),
        );
        tables
    };

    Ok(types::TblgenOutput {
        encodings_by_slot: by_slot,
        processor_model,
        itineraries,
        register_model,
        composite_formats,
        decoder_tables,
    })
}

// ---------------------------------------------------------------------------
// Disk cache for llvm-tblgen output
// ---------------------------------------------------------------------------

/// Compute a cache key from the llvm-tblgen binary's canonical path and mtime.
///
/// Returns `None` if the path cannot be canonicalized or its metadata read
/// (e.g. the binary was deleted between finding and keying it).
fn compute_tblgen_cache_key(tblgen_path: &Path) -> Option<String> {
    use sha2::{Sha256, Digest};

    let canonical = tblgen_path.canonicalize().ok()?;
    let metadata = std::fs::metadata(&canonical).ok()?;
    let mtime = metadata.modified().ok()?;
    let mtime_secs = mtime
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut hasher = Sha256::new();
    hasher.update(canonical.to_string_lossy().as_bytes());
    hasher.update(b":");
    hasher.update(mtime_secs.to_le_bytes());
    Some(format!("{:x}", hasher.finalize()))
}

/// Return the cache directory, creating it if necessary.
fn tblgen_cache_dir() -> Option<std::path::PathBuf> {
    let base = dirs::cache_dir()?;
    let dir = base.join("xdna-emu").join("tblgen");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Try to load cached disassembler output matching `key`.
fn try_load_cached_disasm(key: &str) -> Option<String> {
    let dir = tblgen_cache_dir()?;
    let stored_key = std::fs::read_to_string(dir.join("key.sha256")).ok()?;
    if stored_key.trim() != key {
        return None;
    }
    std::fs::read_to_string(dir.join("disasm.txt")).ok()
}

/// Write disassembler output to disk cache.
fn write_disasm_cache(key: &str, disasm_text: &str) {
    let dir = match tblgen_cache_dir() {
        Some(d) => d,
        None => return,
    };

    let write_atomic = |name: &str, content: &str| -> std::io::Result<()> {
        let target = dir.join(name);
        let tmp = dir.join(format!("{}.tmp", name));
        std::fs::write(&tmp, content)?;
        std::fs::rename(&tmp, &target)?;
        Ok(())
    };

    if let Err(e) = write_atomic("disasm.txt", disasm_text) {
        log::debug!("Failed to cache disasm.txt: {}", e);
        return;
    }
    if let Err(e) = write_atomic("key.sha256", key) {
        log::debug!("Failed to cache key.sha256: {}", e);
        return;
    }

    log::info!("Cached disassembler output (key: {}...)", &key[..key.len().min(12)]);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Verify that structural semantic inference works for real instructions
    /// loaded from llvm-aie via tblgen.
    ///
    /// The tblgen path extracts hasDelaySlot, Defs/Uses, mayLoad/mayStore, and
    /// parent class chain. These structural signals should produce correct
    /// SemanticOps for control flow and memory instructions WITHOUT relying
    /// on mnemonic string parsing.
    #[test]
    fn test_structural_semantic_inference() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        let encodings_by_slot = load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");
        let all: Vec<_> = encodings_by_slot.values().flatten().collect();

        // JL: Call semantic from Defs=[lr] + hasDelaySlot
        if let Some(jl) = all.iter().find(|e| e.name == "JL") {
            assert_eq!(jl.semantic, Some(SemanticOp::Call),
                "JL should be Call (Defs=[lr] + hasDelaySlot)");
        }

        // RET: Return semantic from Uses=[lr] + hasDelaySlot
        if let Some(ret) = all.iter().find(|e| e.name == "RET") {
            assert_eq!(ret.semantic, Some(SemanticOp::Ret),
                "RET should be Ret (Uses=[lr] + hasDelaySlot)");
        }

        // DONE: Done semantic from parent class chain
        if let Some(done) = all.iter().find(|e| e.name == "DONE") {
            assert_eq!(done.semantic, Some(SemanticOp::Done),
                "DONE should be Done (parent chain contains _done_)");
        }

        // Load instructions: mayLoad=true
        let load_names = ["VLDA_128", "LDA_DM_S8_ag_idx_imm"];
        for name in &load_names {
            if let Some(enc) = all.iter().find(|e| e.name == *name) {
                assert_eq!(enc.semantic, Some(SemanticOp::Load),
                    "{} should be Load (mayLoad=true)", name);
                assert!(enc.may_load, "{} should have may_load flag", name);
            }
        }

        // Store instructions: mayStore=true && !mayLoad (pure stores).
        // Some stores have both mayLoad and mayStore (post-increment addressing),
        // which structural inference intentionally skips to avoid misclassifying.
        if let Some(st) = all.iter().find(|e| e.may_store && !e.may_load
            && (e.name.starts_with("ST_") || e.name.starts_with("VST")))
        {
            assert_eq!(st.semantic, Some(SemanticOp::Store),
                "{} should be Store (mayStore=true, mayLoad=false)", st.name);
        }

        // Arithmetic: ADD gets semantic from Pat<> records (no structural signal)
        if let Some(add) = all.iter().find(|e| e.name == "ADD") {
            assert_eq!(add.semantic, Some(SemanticOp::Add),
                "ADD should be Add (from Pat<> record)");
            assert!(!add.may_load && !add.may_store,
                "ADD should not have memory flags");
        }

        // Move instructions: isMoveImm/isMoveReg structural flags
        if let Some(mova) = all.iter().find(|e| e.name == "MOVA_lda_cg") {
            assert_eq!(mova.semantic, Some(SemanticOp::Copy),
                "MOVA_lda_cg should be Copy (isMoveImm=1)");
        }

        // NOP instructions: isSlotNOP structural flag
        if let Some(nopx) = all.iter().find(|e| e.name == "NOPX") {
            assert_eq!(nopx.semantic, Some(SemanticOp::Nop),
                "NOPX should be Nop (isSlotNOP=1)");
        }

        // Pointer arithmetic: PointerAdd propagated from pseudo expansion.
        // Pat<ptradd> -> PADD_imm9_pseudo -> materializableInto -> PADDB_*
        // The concrete PADDB has no Pat<> and no structural flags; semantic
        // comes entirely through the materializableInto chain.
        if let Some(paddb) = all.iter().find(|e| e.name == "PADDB_ldb_ptr_inc_nrm_imm") {
            assert_eq!(paddb.semantic, Some(SemanticOp::PointerAdd),
                "PADDB_ldb_ptr_inc_nrm_imm should be PointerAdd (via pseudo expansion)");
            assert!(paddb.is_ptr_arithmetic,
                "PADDB should have is_ptr_arithmetic derived from semantic");
        }

        // Count semantics by source
        let mut pattern_count = 0;
        let mut structural_count = 0;
        let mut total_with_semantic = 0;
        for enc in &all {
            if enc.semantic.is_some() {
                total_with_semantic += 1;
                // Classify source: structural signals vs pattern-derived
                match enc.semantic {
                    Some(SemanticOp::Load) if enc.may_load => structural_count += 1,
                    Some(SemanticOp::Store) if enc.may_store => structural_count += 1,
                    Some(SemanticOp::Call) | Some(SemanticOp::Ret) | Some(SemanticOp::Done)
                    | Some(SemanticOp::Copy) | Some(SemanticOp::Nop) =>
                        structural_count += 1,
                    _ => pattern_count += 1,
                }
            }
        }
        eprintln!(
            "Semantics: {}/{} instructions ({} structural, {} pattern-derived)",
            total_with_semantic, all.len(), structural_count, pattern_count,
        );
        // All real instructions should have semantics. VLIW bundle envelopes
        // (isComposite=1) are filtered out by to_encoding(), so everything
        // here is a real instruction.
        assert_eq!(
            total_with_semantic, all.len(),
            "Expected 100% semantic coverage ({} missing)",
            all.len() - total_with_semantic,
        );
    }

    /// Verify that the processor scheduling model is parsed correctly.
    #[test]
    fn test_processor_model() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }

        let output = load_full_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        let model = output.processor_model
            .expect("Should have parsed AIE2SchedModel");

        // These values come directly from AIE2Schedule.td
        assert_eq!(model.load_latency, 5, "AIE2 LoadLatency should be 5");
        assert_eq!(model.mispredict_penalty, 4, "AIE2 MispredictPenalty should be 4");
        assert_eq!(model.high_latency, 37, "AIE2 HighLatency should be 37");
        assert_eq!(model.issue_width, 1000, "AIE2 IssueWidth should be 1000 (unlimited)");
        assert_eq!(model.itinerary_name, "AIE2Itineraries");

        eprintln!("ProcessorModel: {:?}", model);
    }

    /// Verify that itinerary data is parsed with correct latencies.
    #[test]
    fn test_itinerary_data() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }

        let output = load_full_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        let itin = &output.itineraries;
        assert!(!itin.is_empty(), "Should have parsed itinerary classes");
        eprintln!("Parsed {} itinerary classes", itin.len());

        // Verify some known itinerary classes exist
        assert!(itin.contains_key("II_ABS"), "Should have II_ABS");
        assert!(itin.contains_key("II_ACQ"), "Should have II_ACQ");

        // II_ABS should have 1-cycle latency (single stage on R_WX_PORT)
        if let Some(abs) = itin.get("II_ABS") {
            assert_eq!(abs.total_latency, 1, "II_ABS should have 1-cycle latency");
            assert_eq!(abs.stages.len(), 1, "II_ABS should have 1 stage");
            eprintln!("II_ABS: latency={}, stages={:?}, cycles={:?}",
                abs.total_latency, abs.stages, abs.operand_cycles);
        }

        // Print a few itinerary classes for inspection
        for (name, info) in itin.iter().take(10) {
            eprintln!("  {}: latency={}, stages={}, operand_cycles={:?}",
                name, info.total_latency, info.stages.len(), info.operand_cycles);
        }
    }

    /// Verify register definitions are parsed with correct HWEncodings.
    #[test]
    fn test_register_model() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }

        let output = load_full_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");
        let model = &output.register_model;

        eprintln!("Register model: {} registers, {} classes",
            model.registers.len(), model.classes.len());

        // Verify basic register counts
        assert!(model.registers.len() > 50, "Should have many registers");
        assert!(model.classes.len() > 5, "Should have several register classes");

        // Verify known HWEncodings (from MEMORY.md)
        // r0: HWEncoding = 0
        if let Some(r0) = model.registers.get("r0") {
            assert_eq!(r0.hw_encoding, 0, "r0 HWEncoding should be 0");
        }

        // lr: HWEncoding = (4 << 3) | 0b111 = 39
        if let Some(lr) = model.registers.get("lr") {
            assert_eq!(lr.hw_encoding, 39, "lr HWEncoding should be 39 = (4<<3)|0b111");
            eprintln!("lr: hw_encoding={}, parents={:?}", lr.hw_encoding, lr.parents);
        }

        // LS: HWEncoding = (0 << 3) | 0b111 = 7
        if let Some(ls) = model.registers.get("LS") {
            assert_eq!(ls.hw_encoding, 7, "LS HWEncoding should be 7");
        }

        // p3: HWEncoding = 3
        if let Some(p3) = model.registers.get("p3") {
            assert_eq!(p3.hw_encoding, 3, "p3 HWEncoding should be 3");
        }

        // Verify register classes
        if let Some(er) = model.classes.get("eR") {
            assert_eq!(er.members.len(), 32, "eR should have 32 members (r0-r31)");
            assert!(er.members.contains(&"r0".to_string()));
            assert!(er.members.contains(&"r31".to_string()));
            eprintln!("eR: {} members, alignment={}", er.members.len(), er.alignment);
        }

        if let Some(ep) = model.classes.get("eP") {
            assert_eq!(ep.members.len(), 8, "eP should have 8 members (p0-p7)");
            eprintln!("eP: {} members, alignment={}", ep.members.len(), ep.alignment);
        }

        // Print register class summary
        for (name, cls) in &model.classes {
            eprintln!("  class {}: {} members, alignment={}",
                name, cls.members.len(), cls.alignment);
        }
    }

    /// Verify composite format definitions are parsed.
    #[test]
    fn test_composite_formats() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }

        let output = load_full_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        let formats = &output.composite_formats;
        assert!(!formats.is_empty(), "Should have parsed composite formats");
        eprintln!("Parsed {} composite formats", formats.len());

        // Should have formats for various sizes
        let sizes: Vec<u8> = formats.iter().map(|f| f.total_bytes).collect();
        eprintln!("Format sizes (bytes): {:?}", sizes);

        // The 128-bit (16-byte) format should exist with all 6 slots
        if let Some(full) = formats.iter().find(|f| f.total_bytes == 16) {
            eprintln!("128-bit format: {}, slots={:?}", full.name, full.slots);
            // Should have ldb(16), lda(21), st(21), alu(20), mv(22), vec(26)
            assert!(full.slots.len() >= 4,
                "128-bit format should have at least 4 slots, got {}", full.slots.len());
        }

        for fmt in formats {
            eprintln!("  {} ({}B = {}b): {:?}",
                fmt.name, fmt.total_bytes, fmt.total_bits, fmt.slots);
        }
    }
}
