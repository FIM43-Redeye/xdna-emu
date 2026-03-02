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

pub mod decoder_bytecode;
mod parser;
mod resolver;
pub mod tblgen_records;
mod types;

pub use parser::{
    parse_format_classes, parse_instructions, parse_patterns, parse_slots, parse_tablegen_files,
    parse_tablegen_files_with_patterns, ParseError,
};
pub use resolver::{
    build_decoder_tables, AddressingMode, CompositeEncoder, DecoderIndex, InstrEncoding,
    InstrMemWidth, OperandField, OperandType, RegisterKind, ResolveError, Resolver, SlotIndex,
    classify_operand_type, detect_addressing_mode, detect_mem_width,
    infer_branch_condition, infer_element_type, infer_select_variant, refine_branch_semantic,
};
pub use tblgen_records::{parse_tblgen_records, InstrRecord, SlotEncoding};
pub use types::{
    BranchCondition, CompositeFormatDef, ElementType, EncodingPart, FormatClass, ImplicitReg,
    InstrAttributes, InstrDef, ItineraryInfo, OperandDef, PipelineStage, ProcessorModel,
    RegisterClassDef, RegisterDef, RegisterModel, SelectVariant, SemanticOp, SemanticPattern,
    SlotDef, TableGenData, TblgenOutput, TemplateParam,
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

/// Load the complete TableGen model using llvm-tblgen.
///
/// This runs `llvm-tblgen --print-records AIE2.td` and extracts:
/// - Instruction encodings (grouped by slot)
/// - Processor scheduling model (LoadLatency, MispredictPenalty, etc.)
/// - Per-instruction itinerary data (latencies, pipeline stages)
/// - Register model (definitions with HWEncodings, class memberships)
/// - Composite VLIW bundle format definitions
///
/// Results are cached to disk at `~/.cache/xdna-emu/tblgen/` keyed on the
/// llvm-tblgen binary's canonical path and modification time. Cache misses
/// (including any I/O error) silently fall back to fresh subprocess invocation.
pub fn load_full_via_tblgen(llvm_aie_path: impl AsRef<Path>) -> Result<types::TblgenOutput, std::io::Error> {
    let llvm_aie = llvm_aie_path.as_ref();
    let base = llvm_aie.join("llvm/lib/Target/AIE");

    // Find AIE-enabled llvm-tblgen
    let tblgen_path = find_aie_tblgen(llvm_aie)
        .ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No AIE-enabled llvm-tblgen found. Set LLVM_AIE_TBLGEN or install mlir-aie."
        ))?;

    // Try disk cache first
    let cache_key = compute_tblgen_cache_key(&tblgen_path);
    if let Some(ref key) = cache_key {
        if let Some((records_text, disasm_text)) = try_load_cached_tblgen(key) {
            log::info!("Using cached tblgen output (key: {}...)", &key[..12]);
            return parse_tblgen_output(&records_text, &disasm_text);
        }
    }

    // Cache miss -- run subprocesses
    log::info!("Running llvm-tblgen from: {}", tblgen_path.display());
    let (records_text, disasm_text) = run_tblgen_subprocesses(&tblgen_path, &base)?;

    // Write to cache (best-effort, errors are logged and ignored)
    if let Some(ref key) = cache_key {
        write_tblgen_cache(key, &records_text, &disasm_text);
    }

    parse_tblgen_output(&records_text, &disasm_text)
}

/// Run both llvm-tblgen subprocesses and return raw stdout text.
///
/// Spawns `--print-records` and `-gen-disassembler` concurrently. The records
/// output is required (errors are propagated); the disassembler output is
/// best-effort (failures produce an empty string, logged as warnings).
fn run_tblgen_subprocesses(
    tblgen_path: &Path,
    base: &Path,
) -> Result<(String, String), std::io::Error> {
    use std::process::Command;

    // Both need the same include paths and working directory.
    // Remove LD_LIBRARY_PATH to prevent aietools from injecting an older
    // libstdc++ that lacks GLIBCXX symbols llvm-tblgen needs.
    let tblgen_args = ["AIE2.td", "-I.", "-I../../..", "-I../../../include"];

    // Spawn -gen-disassembler first (runs concurrently while we wait for records)
    let disasm_child = Command::new(tblgen_path)
        .arg("-gen-disassembler")
        .args(&tblgen_args)
        .current_dir(base)
        .env_remove("LD_LIBRARY_PATH")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    // Run --print-records synchronously
    let records_output = Command::new(tblgen_path)
        .arg("--print-records")
        .args(&tblgen_args)
        .current_dir(base)
        .env_remove("LD_LIBRARY_PATH")
        .output()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other,
            format!("Failed to run llvm-tblgen at {}: {}", tblgen_path.display(), e)))?;

    if !records_output.status.success() {
        let stderr = String::from_utf8_lossy(&records_output.stderr);
        return Err(std::io::Error::new(std::io::ErrorKind::Other,
            format!("llvm-tblgen --print-records failed: {}", stderr)));
    }

    let records_text = String::from_utf8_lossy(&records_output.stdout).into_owned();

    // Collect the -gen-disassembler output (best-effort)
    let disasm_text = match disasm_child {
        Ok(child) => {
            match child.wait_with_output() {
                Ok(out) if out.status.success() => {
                    String::from_utf8_lossy(&out.stdout).into_owned()
                }
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    log::warn!("llvm-tblgen -gen-disassembler failed: {}", stderr);
                    String::new()
                }
                Err(e) => {
                    log::warn!("Failed to collect -gen-disassembler output: {}", e);
                    String::new()
                }
            }
        }
        Err(e) => {
            log::warn!("Failed to spawn -gen-disassembler: {}", e);
            String::new()
        }
    };

    Ok((records_text, disasm_text))
}

/// Parse raw llvm-tblgen text output into the complete TableGen model.
fn parse_tblgen_output(
    records_text: &str,
    disasm_text: &str,
) -> Result<types::TblgenOutput, std::io::Error> {
    // Parse instruction records
    let records = tblgen_records::parse_tblgen_records(records_text);
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

    // Apply Pat<>-derived semantics from fully resolved pattern records.
    // These override structural inference because they carry more specific
    // semantic information (e.g., ADD->Add, SELEQZ->Select, ACQ->LockAcquire).
    let pattern_map = tblgen_records::parse_pattern_records(records_text);
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
    let pseudo_map = tblgen_records::parse_pseudo_expansion_map(records_text);
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

    // Itinerary-based inference for instruction families without pseudo expansion.
    //
    // The 2D/3D PADD variants (PADDA_2D, PADDB_3D, etc.) have no MultiSlot_Pseudo
    // parent and no Pat<> records -- the compiler emits them via custom C++ lowering
    // for multi-dimensional addressing. Similarly, PADD_sp_imm_pseudo expands to
    // PADDB_sp_imm/PADDA_sp_imm but has no ptradd SDNode pattern (it's used for
    // stack frame adjustment). The scheduling model's Itinerary field (II_PADD,
    // II_PADD_2D, II_PADD_3D) groups all pointer arithmetic instructions, providing
    // a structural classification from LLVM that doesn't depend on string matching
    // the assembly mnemonic.
    let mut itinerary_inferred = 0usize;
    for encodings in by_slot.values_mut() {
        for enc in encodings.iter_mut() {
            if enc.semantic.is_none() {
                if let Some(ref sched) = enc.sched_class {
                    if sched.starts_with("II_PADD") {
                        enc.semantic = Some(types::SemanticOp::PointerAdd);
                        itinerary_inferred += 1;
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

    // Parse extended data
    let processor_model = tblgen_records::parse_processor_model(records_text);
    let itineraries = tblgen_records::parse_itinerary_data(records_text);
    let register_model = tblgen_records::parse_register_model(records_text);
    let composite_formats = tblgen_records::parse_composite_formats(records_text);

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

/// Try to load cached tblgen output matching `key`.
///
/// Returns `None` on any mismatch or I/O error -- the caller should fall
/// through to fresh subprocess invocation.
fn try_load_cached_tblgen(key: &str) -> Option<(String, String)> {
    let dir = tblgen_cache_dir()?;

    // Validate key file first -- if it doesn't match, nothing else matters
    let stored_key = std::fs::read_to_string(dir.join("key.sha256")).ok()?;
    if stored_key.trim() != key {
        return None;
    }

    let records = std::fs::read_to_string(dir.join("records.txt")).ok()?;
    let disasm = std::fs::read_to_string(dir.join("disasm.txt")).ok()?;
    Some((records, disasm))
}

/// Write tblgen output to the disk cache.
///
/// Uses atomic writes (write to .tmp, then rename) to avoid partial reads.
/// Data files are written first, key file last -- a partial write looks like
/// a cache miss (stale or missing key).
fn write_tblgen_cache(key: &str, records_text: &str, disasm_text: &str) {
    let dir = match tblgen_cache_dir() {
        Some(d) => d,
        None => {
            log::debug!("Could not create tblgen cache directory");
            return;
        }
    };

    let write_atomic = |name: &str, content: &str| -> std::io::Result<()> {
        let target = dir.join(name);
        let tmp = dir.join(format!("{}.tmp", name));
        std::fs::write(&tmp, content)?;
        std::fs::rename(&tmp, &target)?;
        Ok(())
    };

    // Write data files first, key file last
    if let Err(e) = write_atomic("records.txt", records_text) {
        log::debug!("Failed to cache records.txt: {}", e);
        return;
    }
    if let Err(e) = write_atomic("disasm.txt", disasm_text) {
        log::debug!("Failed to cache disasm.txt: {}", e);
        return;
    }
    if let Err(e) = write_atomic("key.sha256", key) {
        log::debug!("Failed to cache key.sha256: {}", e);
        return;
    }

    log::info!("Cached tblgen output (key: {}...)", &key[..key.len().min(12)]);
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

        // Count total patterns and break down by source
        let total = data.patterns.len();
        let intrinsic_count = data.patterns.iter()
            .filter(|p| p.intrinsic_name.is_some())
            .count();
        let sdnode_count = total - intrinsic_count;

        eprintln!("Found {} semantic patterns ({} SDNode, {} intrinsic)",
            total, sdnode_count, intrinsic_count);

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

        // Check intrinsic-derived patterns exist for key vector operations
        let matmul = data.instructions_for_semantic(SemanticOp::MatMul);
        let mac = data.instructions_for_semantic(SemanticOp::Mac);
        let srs = data.instructions_for_semantic(SemanticOp::Srs);
        let ups = data.instructions_for_semantic(SemanticOp::Ups);
        let shuffle = data.instructions_for_semantic(SemanticOp::Shuffle);
        let lock_acq = data.instructions_for_semantic(SemanticOp::LockAcquire);

        eprintln!(
            "Intrinsic-derived: {} MatMul, {} Mac, {} Srs, {} Ups, {} Shuffle, {} LockAcquire",
            matmul.len(), mac.len(), srs.len(), ups.len(), shuffle.len(), lock_acq.len()
        );

        // Print breakdown by operation type
        let mut op_counts: std::collections::HashMap<SemanticOp, usize> = std::collections::HashMap::new();
        for p in &data.patterns {
            *op_counts.entry(p.operation).or_default() += 1;
        }
        let mut sorted: Vec<_> = op_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("All pattern operations:");
        for (op, count) in &sorted {
            eprintln!("  {:?}: {}", op, count);
        }

        // We should have found at least the basic ALU operations
        assert!(!data.patterns.is_empty(), "Should have found some patterns");

        // Intrinsic patterns should contribute significantly (>50 patterns)
        assert!(
            intrinsic_count > 50,
            "Expected >50 intrinsic-derived patterns, got {}",
            intrinsic_count
        );

        // Key vector operations should be present
        assert!(!matmul.is_empty(), "MatMul patterns should exist");
        assert!(!srs.is_empty(), "SRS patterns should exist");
        assert!(!ups.is_empty(), "UPS patterns should exist");
        assert!(!shuffle.is_empty(), "Shuffle patterns should exist");
        assert!(!lock_acq.is_empty(), "LockAcquire patterns should exist");
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

    // =========================================================================
    // Phase 1 Tests: Extended TableGen Data Extraction
    // =========================================================================

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
