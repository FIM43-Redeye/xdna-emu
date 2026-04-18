//! Build script for xdna-emu.
//!
//! Most arch-data code generation has moved to `crates/xdna-archspec/build.rs`.
//! What remains here pending later Subsystem 6 tasks:
//!
//! - `extract_aiert` + helpers + `gen_aiert_dma` / `gen_aiert_locks` / `gen_aiert_ports`
//!   (outputs consumed by `src/device/aiert_validation.rs`).
//!   These duplicate their archspec counterparts; the cross-validation
//!   side-effect on ArchModel is unique to archspec's copy.
//!
//! - `compile_llvm_decoder_ffi` + LLVM link (compiles
//!   `crates/xdna-archspec/decoder_ffi/aie2_decoder.cpp`). Moves to archspec in Task 9.
//!
//! - XRT plugin install logic (always belongs in xdna-emu).
//!
//! gen_tablegen.rs moved to xdna-archspec/build.rs in Task 7; consumed via
//! `xdna_archspec::aie2::isa::load_from_generated()`.
//!
//! Generated files (written to `$OUT_DIR/`):
//! - `gen_aiert_dma.rs`     -- aie-rt DMA module data (included by `aiert_validation`)
//! - `gen_aiert_locks.rs`   -- aie-rt lock module data (included by `aiert_validation`)
//! - `gen_aiert_ports.rs`   -- aie-rt port module data (included by `aiert_validation`)

use std::collections::HashMap;
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};

// AM025 register database types come from the graph crate.
// This eliminates the duplicate JSON parsing types that build.rs
// previously maintained separately (build scripts can now share
// types via workspace member crates).
use xdna_archspec::regdb::RegisterDb;

// ============================================================================
// Entry point
// ============================================================================

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Resolve AM025 JSON path: MLIR_AIE_PATH env or sibling directory
    let mlir_aie = env::var("MLIR_AIE_PATH").unwrap_or_else(|_| {
        manifest_dir
            .parent()
            .expect("Cargo manifest has no parent directory")
            .join("mlir-aie")
            .to_string_lossy()
            .to_string()
    });
    let am025_path =
        Path::new(&mlir_aie).join("lib/Dialect/AIE/Util/aie_registers_aie2.json");

    // Device model is in-repo
    let device_model_path = manifest_dir.join("tools/aie-device-models.json");

    // Rebuild triggers
    println!("cargo:rerun-if-changed={}", am025_path.display());
    println!("cargo:rerun-if-changed={}", device_model_path.display());
    println!("cargo:rerun-if-env-changed=MLIR_AIE_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    // Load AM025 register database via the graph crate's parser.
    let regdb = RegisterDb::from_file(&am025_path).unwrap_or_else(|e| {
        panic!(
            "Cannot load AM025 register database at {}:\n  {}\n\
             Set MLIR_AIE_PATH to override the mlir-aie location.",
            am025_path.display(),
            e
        )
    });

    // Build the full ArchModel (device topology + register data, cross-validated).
    // This is the graph as compile-time truth: Confirmed<T> panics on conflicts.
    let mut arch_model = xdna_archspec::build_arch_model(&device_model_path, &regdb, "npu1")
        .unwrap_or_else(|e| panic!("Failed to build ArchModel: {}", e));

    // Generate aie-rt extracted constants and cross-validate subsystem ranges.
    // This must happen before gen_arch() so the model's Confirmed<T> sources
    // reflect both AM025 and aie-rt when code is generated.
    extract_aiert(&manifest_dir, &out_dir, &mut arch_model);

    // Cross-validate processor model slot widths against llvm-aie TableGen.
    // The manual constants (from populate_aie2_manual_constants) are the baseline;
    // this extraction from AIE2Slots.td confirms them at compile time.
    let llvm_aie = env::var("LLVM_AIE_PATH").unwrap_or_else(|_| {
        manifest_dir
            .parent()
            .expect("Cargo manifest has no parent directory")
            .join("llvm-aie")
            .to_string_lossy()
            .to_string()
    });
    let llvm_aie_path = Path::new(&llvm_aie);
    let slots_td = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2Slots.td");
    println!("cargo:rerun-if-env-changed=LLVM_AIE_PATH");
    if slots_td.exists() {
        println!("cargo:rerun-if-changed={}", slots_td.display());
        match xdna_archspec::tablegen::confirm_processor_slots(
            &mut arch_model,
            llvm_aie_path,
        ) {
            Ok(count) => {
                println!(
                    "cargo:warning=TableGen: confirmed {} slot widths from {}",
                    count,
                    slots_td.display()
                );
            }
            Err(e) => {
                panic!(
                    "TableGen slot validation failed:\n  {}\n\
                     Set LLVM_AIE_PATH to override the llvm-aie location.",
                    e
                );
            }
        }
    } else {
        println!(
            "cargo:warning=llvm-aie not found at {} -- using manual slot widths only",
            llvm_aie_path.display()
        );
    }

    // gen_arch(), gen_subsystems(), gen_core_module(), gen_lock_request(),
    // gen_stream_ports(), gen_stream_ranges(), and gen_trace_events() have all
    // moved to xdna-archspec/build.rs (Tasks 4-8).
    //
    // gen_tablegen.rs now emits from xdna-archspec/build.rs (Task 7).
    // load_from_generated() is forwarded via pub use in src/tablegen/mod.rs.

    // ========================================================================
    // LLVM decoder FFI -- compile aie2_decoder.cpp and link LLVM libraries
    // ========================================================================
    //
    // Links LLVM's AIE2 MCDisassembler into the emulator so we get perfect
    // TRY_DECODE disambiguation (register class validation) without having
    // to reimplement LLVM's per-instruction decoder functions.
    compile_llvm_decoder_ffi(llvm_aie_path);

    // ========================================================================
    // Post-codegen: rebuild and install XRT plugin
    // ========================================================================
    //
    // The C++ plugin (libxrt_driver_emu.so) loads the Rust emulator at
    // runtime via dlopen -- there is no link-time dependency. This lets
    // us build and install the plugin from build.rs without circularity.
    //
    // The plugin .so is installed to /opt/xilinx/xrt/lib/ so XRT can
    // find it. The Rust lib is NOT copied -- the plugin resolves it at
    // runtime via XDNA_EMU_DIR/target/$profile/libxdna_emu.so.

    // Rebuild triggers for plugin C++ sources
    let plugin_src = manifest_dir.join("xrt-plugin/src");
    if plugin_src.exists() {
        for entry in fs::read_dir(&plugin_src).unwrap() {
            let entry = entry.unwrap();
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
        println!("cargo:rerun-if-changed=xrt-plugin/CMakeLists.txt");
    }

    // Only build the plugin if the cmake build directory exists.
    // First-time setup still requires: mkdir -p xrt-plugin/build && cd xrt-plugin/build && cmake ..
    let plugin_build = manifest_dir.join("xrt-plugin/build");
    let xrt_lib = Path::new("/opt/xilinx/xrt/lib");
    if plugin_build.join("CMakeCache.txt").exists() && xrt_lib.exists() {
        // Incremental cmake build (~2s when nothing changed).
        // Use .output() to capture stderr for diagnostics on failure.
        let result = std::process::Command::new("cmake")
            .args(["--build", "."])
            .current_dir(&plugin_build)
            .output();

        match result {
            Ok(output) if output.status.success() => {
                // Install plugin .so to XRT lib directory.
                // If the destination is already a symlink to the source,
                // skip the copy -- fs::copy follows symlinks and would
                // truncate the source file to 0 bytes.
                let src = plugin_build.join("libxrt_driver_emu.so.2.21.0");
                let dst = xrt_lib.join("libxrt_driver_emu.so.2.21.0");
                let link = xrt_lib.join("libxrt_driver_emu.so.2");
                if src.exists() {
                    let dst_resolves_to_src = dst.read_link()
                        .ok()
                        .and_then(|target| fs::canonicalize(xrt_lib.join(target)).ok())
                        .map_or(false, |resolved| {
                            fs::canonicalize(&src).map_or(false, |src_canon| resolved == src_canon)
                        });
                    if dst_resolves_to_src {
                        // Already symlinked -- nothing to do.
                    } else if let Err(e) = fs::copy(&src, &dst) {
                        println!("cargo:warning=Plugin install failed: {e}");
                    } else {
                        // Create/update symlink
                        let _ = fs::remove_file(&link);
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::symlink;
                            let _ = symlink("libxrt_driver_emu.so.2.21.0", &link);
                        }
                    }
                }
            }
            Ok(output) => {
                let code = output.status.code().unwrap_or(-1);
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("cargo:warning=Plugin cmake build failed (exit {code})");
                // Print first few lines of stderr for diagnostics.
                for line in stderr.lines().take(10) {
                    println!("cargo:warning=  plugin: {line}");
                }
            }
            Err(e) => {
                println!("cargo:warning=Plugin cmake build failed: {e}");
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// File header comment for generated source.
fn gen_header(source_desc: &str) -> String {
    format!(
        "// Auto-generated by build.rs from {}.\n\
         // Do not edit manually.\n\n",
        source_desc
    )
}

// ============================================================================
// Step 7: aie-rt header extraction (DMA, Lock, Stream Switch)
// ============================================================================

/// Top-level: locate aie-rt, run gcc -E, parse, generate all 3 files,
/// and cross-validate subsystem address ranges in the ArchModel.
fn extract_aiert(
    manifest_dir: &Path,
    out_dir: &Path,
    arch_model: &mut xdna_archspec::types::ArchModel,
) {
    use xdna_archspec::types::{
        ModuleKind, Source, SourceAttribution, SubsystemKind, TileKind,
    };

    // Rebuild triggers
    println!("cargo:rerun-if-env-changed=AIE_RT_PATH");

    let aiert_dir = env::var("AIE_RT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            manifest_dir
                .parent()
                .expect("Cargo manifest has no parent directory")
                .join("aie-rt/driver/src")
        });

    let reginit_path = aiert_dir.join("global/xaiemlgbl_reginit.c");
    if reginit_path.exists() {
        println!("cargo:rerun-if-changed={}", reginit_path.display());
    }

    let preprocessed = match run_aiert_preprocessor(&aiert_dir) {
        Some(text) => text,
        None => {
            write_aiert_stubs(out_dir);
            return;
        }
    };

    let dma_modules = parse_dma_modules(&preprocessed);
    let lock_modules = parse_lock_modules(&preprocessed);
    let port_maps = parse_port_maps(&preprocessed);

    // Cross-validate subsystem offset_start values against aie-rt.
    // DMA BaseAddr is the BD base offset, matching the DMA subsystem's
    // offset_start from AM025 register grouping.
    // Lock LockSetValBase is the lock value register base, matching the
    // Lock subsystem's offset_start from AM025 register grouping.
    let mut confirmations = Vec::new();

    // Map aie-rt DMA module names to (TileKind, ModuleKind)
    for m in &dma_modules {
        let (tile_kind, mod_kind) = if m.name.contains("MemTile") {
            (TileKind::Mem, ModuleKind::MemTile)
        } else if m.name.contains("Shim") {
            (TileKind::ShimNoc, ModuleKind::Shim)
        } else {
            (TileKind::Compute, ModuleKind::Memory)
        };

        let base_addr = get_field(&m.fields, "BaseAddr", &m.name);
        let source = SourceAttribution {
            origin: Source::AieRt,
            file: "xaiemlgbl_reginit.c".into(),
            detail: format!("{}.BaseAddr", m.name),
        };
        confirmations.push((tile_kind, mod_kind, SubsystemKind::Dma, base_addr, source));
    }

    // Map aie-rt Lock module names to (TileKind, ModuleKind)
    for m in &lock_modules {
        let (tile_kind, mod_kind) = if m.name.contains("MemTile") {
            (TileKind::Mem, ModuleKind::MemTile)
        } else if m.name.contains("Shim") {
            (TileKind::ShimNoc, ModuleKind::Shim)
        } else {
            (TileKind::Compute, ModuleKind::Memory)
        };

        let set_val_base = get_field(&m.fields, "LockSetValBase", &m.name);
        let source = SourceAttribution {
            origin: Source::AieRt,
            file: "xaiemlgbl_reginit.c".into(),
            detail: format!("{}.LockSetValBase", m.name),
        };
        confirmations.push((tile_kind, mod_kind, SubsystemKind::Lock, set_val_base, source));
    }

    xdna_archspec::confirm_subsystem_ranges(arch_model, &confirmations);

    gen_aiert_dma(&dma_modules, out_dir);
    gen_aiert_locks(&lock_modules, out_dir);
    gen_aiert_ports(&port_maps, out_dir);
}

/// Run gcc -E on xaiemlgbl_reginit.c with all aie-rt include paths.
fn run_aiert_preprocessor(aiert_dir: &Path) -> Option<String> {
    use std::process::Command;

    let reginit = aiert_dir.join("global/xaiemlgbl_reginit.c");
    if !reginit.exists() {
        eprintln!(
            "cargo:warning=aie-rt not found at {}, skipping aie-rt extraction",
            aiert_dir.display()
        );
        return None;
    }

    // All subdirectories that contain headers
    let subdirs = [
        "", "common", "core", "device", "dma", "events", "global",
        "interrupt", "io_backend", "lite", "locks", "memory", "npi",
        "perfcnt", "pl", "pm", "routing", "stream_switch", "timer",
        "trace", "util",
    ];

    let mut cmd = Command::new("gcc");
    cmd.arg("-E");
    for subdir in &subdirs {
        let inc = if subdir.is_empty() {
            aiert_dir.to_path_buf()
        } else {
            aiert_dir.join(subdir)
        };
        cmd.arg("-I").arg(&inc);
    }
    cmd.arg(&reginit);

    let output = match cmd.output() {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!(
                "cargo:warning=gcc -E failed ({}): {}, skipping aie-rt extraction",
                o.status, stderr
            );
            return None;
        }
        Err(e) => {
            eprintln!(
                "cargo:warning=Cannot run gcc ({}), skipping aie-rt extraction",
                e
            );
            return None;
        }
    };

    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

// ---- aie-rt preprocessor parsing (duplicated with xdna-archspec/build.rs) ----
//
// The archspec copy uses this logic to cross-validate ArchModel fields.
// This copy produces gen_aiert_dma/locks/ports.rs via gen_aiert_*(),
// which aiert_validation.rs includes. Both copies must stay in sync;
// delete this copy when gen_aiert_* consumers migrate.

// -- Data structures for parsed aie-rt structs --

struct DmaModData {
    name: String,
    fields: HashMap<String, String>,
}

struct LockModData {
    name: String,
    fields: HashMap<String, String>,
}

struct PortMapData {
    name: String,
    entries: Vec<(String, u8)>, // (PortType name, PortNum)
}

/// Parse all XAie_DmaMod struct initializers from preprocessed text.
fn parse_dma_modules(text: &str) -> Vec<DmaModData> {
    parse_struct_initializers(text, "XAie_DmaMod")
        .into_iter()
        .map(|(name, fields)| DmaModData { name, fields })
        .collect()
}

/// Parse all XAie_LockMod struct initializers from preprocessed text.
fn parse_lock_modules(text: &str) -> Vec<LockModData> {
    parse_struct_initializers(text, "XAie_LockMod")
        .into_iter()
        .map(|(name, fields)| LockModData { name, fields })
        .collect()
}

/// Parse all XAie_StrmSwPortMap array initializers from preprocessed text.
fn parse_port_maps(text: &str) -> Vec<PortMapData> {
    let mut results = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();
        // Look for: static const XAie_StrmSwPortMap <name>[] =
        if line.contains("XAie_StrmSwPortMap") && line.contains("[]") {
            // Extract the array name
            let name = extract_identifier(line, "XAie_StrmSwPortMap");
            if let Some(name) = name {
                let mut entries = Vec::new();
                let mut depth = 0;
                let mut current_port_type: Option<String> = None;
                let mut current_port_num: Option<u8> = None;

                // Advance to opening brace
                while i < lines.len() && !lines[i].contains('{') {
                    i += 1;
                }
                if i < lines.len() {
                    depth = 1;
                    i += 1;
                }

                while i < lines.len() && depth > 0 {
                    let l = lines[i].trim();
                    for ch in l.chars() {
                        match ch {
                            '{' => depth += 1,
                            '}' => {
                                depth -= 1;
                                if depth == 1 {
                                    // End of one entry
                                    if let (Some(pt), Some(pn)) =
                                        (current_port_type.take(), current_port_num.take())
                                    {
                                        entries.push((pt, pn));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    if let Some(val) = extract_field_value(l, "PortType") {
                        current_port_type = Some(val);
                    }
                    if let Some(val) = extract_field_value(l, "PortNum") {
                        current_port_num = parse_numeric_value(&val).map(|v| v as u8);
                    }

                    i += 1;
                }

                results.push(PortMapData { name, entries });
            }
        }
        i += 1;
    }

    results
}

/// Generic parser for C struct initializers of a given type name.
/// Returns Vec of (instance_name, field_map).
fn parse_struct_initializers(
    text: &str,
    type_name: &str,
) -> Vec<(String, HashMap<String, String>)> {
    let mut results = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();
        // Look for: static const <type_name> <name> =
        if line.contains(type_name) && line.contains('=') && !line.contains("[]") && !line.contains('(') {
            let name = extract_identifier(line, type_name);
            if let Some(name) = name {
                let mut fields = HashMap::new();
                let mut depth = 0;

                // Advance to opening brace
                while i < lines.len() && !lines[i].contains('{') {
                    i += 1;
                }
                if i < lines.len() {
                    depth = 1;
                    i += 1;
                }

                // Only parse top-level fields (depth == 1)
                while i < lines.len() && depth > 0 {
                    let l = lines[i].trim();
                    for ch in l.chars() {
                        match ch {
                            '{' => depth += 1,
                            '}' => depth -= 1,
                            _ => {}
                        }
                    }

                    // Only extract fields at depth 1 (top-level struct fields)
                    if depth == 1 && l.starts_with('.') {
                        if let Some((field_name, value)) = parse_field_assignment(l) {
                            fields.insert(field_name, value);
                        }
                    }

                    i += 1;
                }

                results.push((name, fields));
            }
        }
        i += 1;
    }

    results
}

/// Extract identifier name after the type name in a declaration line.
fn extract_identifier(line: &str, type_name: &str) -> Option<String> {
    let idx = line.find(type_name)? + type_name.len();
    let rest = line[idx..].trim();
    let ident: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if ident.is_empty() {
        None
    } else {
        Some(ident)
    }
}

/// Parse ".FieldName = value," from a line.
fn parse_field_assignment(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    if !line.starts_with('.') {
        return None;
    }
    let eq_idx = line.find('=')?;
    let field_name = line[1..eq_idx].trim().to_string();

    // Skip sub-field assignments like ".NxtBd.Idx = ..."
    if field_name.contains('.') {
        return None;
    }

    let value_part = line[eq_idx + 1..].trim();
    // Strip trailing comma and any trailing comment
    let value = value_part
        .trim_end_matches(',')
        .trim()
        .to_string();

    // Skip function pointer values
    if value.starts_with('&') || value.contains("((void *)0)") {
        return None;
    }

    Some((field_name, value))
}

/// Extract a field value from a line like ".FieldName = value,"
fn extract_field_value(line: &str, field: &str) -> Option<String> {
    let pattern = format!(".{} =", field);
    if let Some(idx) = line.find(&pattern) {
        let after = line[idx + pattern.len()..].trim();
        let value: String = after
            .chars()
            .take_while(|c| *c != ',' && *c != '}')
            .collect();
        Some(value.trim().to_string())
    } else {
        None
    }
}

/// Parse a numeric value string (hex or decimal, optional U suffix).
fn parse_numeric_value(s: &str) -> Option<u32> {
    let s = s.trim().trim_end_matches('U').trim_end_matches('u');
    if s.starts_with("0x") || s.starts_with("0X") {
        u32::from_str_radix(&s[2..], 16).ok()
    } else if s.starts_with('-') {
        // Handle negative values (e.g., LockValLowerBound = -64)
        s.parse::<i32>().ok().map(|v| v as u32)
    } else {
        s.parse::<u32>().ok()
    }
}

/// Get a u32 field from a DmaModData or LockModData field map.
fn get_field(fields: &HashMap<String, String>, name: &str, struct_name: &str) -> u32 {
    let val_str = fields
        .get(name)
        .unwrap_or_else(|| panic!("Field '{}' not found in {}", name, struct_name));
    parse_numeric_value(val_str)
        .unwrap_or_else(|| panic!("Cannot parse '{}' = '{}' in {}", name, val_str, struct_name))
}

/// Map an aie-rt struct instance name to a Rust module name.
fn dma_mod_name(name: &str) -> &str {
    if name.contains("MemTile") {
        "memtile_dma"
    } else if name.contains("Shim") {
        "shim_dma"
    } else {
        "compute_dma"
    }
}

fn lock_mod_name(name: &str) -> &str {
    if name.contains("MemTile") {
        "memtile_locks"
    } else if name.contains("Shim") {
        "shim_locks"
    } else {
        "compute_locks"
    }
}

fn port_map_rust_name(name: &str) -> &str {
    if name.contains("MemTile") && name.contains("Master") {
        "MEMTILE_MASTER_PORTS"
    } else if name.contains("MemTile") && name.contains("Slave") {
        "MEMTILE_SLAVE_PORTS"
    } else if name.contains("Shim") && name.contains("Master") {
        "SHIM_MASTER_PORTS"
    } else if name.contains("Shim") && name.contains("Slave") {
        "SHIM_SLAVE_PORTS"
    } else if name.contains("Master") {
        "COMPUTE_MASTER_PORTS"
    } else {
        "COMPUTE_SLAVE_PORTS"
    }
}

// -- Code generation functions --

fn gen_aiert_dma(modules: &[DmaModData], out_dir: &Path) {
    let mut out = gen_header("aie-rt xaiemlgbl_reginit.c DMA modules");

    for m in modules {
        let mod_name = dma_mod_name(&m.name);
        writeln!(out, "/// DMA constants from aie-rt {} ({})", m.name, mod_name).unwrap();
        writeln!(out, "pub mod {} {{", mod_name).unwrap();
        writeln!(out, "    pub const BD_BASE: u32 = {:#010X};", get_field(&m.fields, "BaseAddr", &m.name)).unwrap();
        writeln!(out, "    pub const BD_STRIDE: u32 = {:#06X};", get_field(&m.fields, "IdxOffset", &m.name)).unwrap();
        writeln!(out, "    pub const NUM_BDS: usize = {};", get_field(&m.fields, "NumBds", &m.name)).unwrap();
        writeln!(out, "    pub const NUM_LOCKS: usize = {};", get_field(&m.fields, "NumLocks", &m.name)).unwrap();
        writeln!(out, "    pub const START_QUEUE_BASE: u32 = {:#010X};", get_field(&m.fields, "StartQueueBase", &m.name)).unwrap();
        writeln!(out, "    pub const CH_CTRL_BASE: u32 = {:#010X};", get_field(&m.fields, "ChCtrlBase", &m.name)).unwrap();
        writeln!(out, "    pub const NUM_CHANNELS: usize = {};", get_field(&m.fields, "NumChannels", &m.name)).unwrap();
        writeln!(out, "    pub const CH_STRIDE: u32 = {:#06X};", get_field(&m.fields, "ChIdxOffset", &m.name)).unwrap();
        writeln!(out, "    pub const CH_STATUS_BASE: u32 = {:#010X};", get_field(&m.fields, "ChStatusBase", &m.name)).unwrap();
        writeln!(out, "    pub const CH_STATUS_STRIDE: u32 = {:#06X};", get_field(&m.fields, "ChStatusOffset", &m.name)).unwrap();
        writeln!(out, "    pub const NUM_ADDR_DIM: usize = {};", get_field(&m.fields, "NumAddrDim", &m.name)).unwrap();
        writeln!(out, "}}\n").unwrap();
    }

    fs::write(out_dir.join("gen_aiert_dma.rs"), out).unwrap();
}

fn gen_aiert_locks(modules: &[LockModData], out_dir: &Path) {
    let mut out = gen_header("aie-rt xaiemlgbl_reginit.c Lock modules");

    for m in modules {
        let mod_name = lock_mod_name(&m.name);
        writeln!(out, "/// Lock constants from aie-rt {} ({})", m.name, mod_name).unwrap();
        writeln!(out, "pub mod {} {{", mod_name).unwrap();
        writeln!(out, "    pub const BASE: u32 = {:#010X};", get_field(&m.fields, "BaseAddr", &m.name)).unwrap();
        writeln!(out, "    pub const NUM_LOCKS: usize = {};", get_field(&m.fields, "NumLocks", &m.name)).unwrap();
        writeln!(out, "    pub const LOCK_ID_STRIDE: u32 = {:#06X};", get_field(&m.fields, "LockIdOff", &m.name)).unwrap();
        writeln!(out, "    pub const REL_ACQ_OFFSET: u32 = {:#06X};", get_field(&m.fields, "RelAcqOff", &m.name)).unwrap();
        writeln!(out, "    pub const LOCK_VAL_OFFSET: u32 = {:#06X};", get_field(&m.fields, "LockValOff", &m.name)).unwrap();
        writeln!(out, "    pub const VAL_UPPER_BOUND: i32 = {};", get_field(&m.fields, "LockValUpperBound", &m.name) as i32).unwrap();
        writeln!(out, "    pub const VAL_LOWER_BOUND: i32 = {};", get_field(&m.fields, "LockValLowerBound", &m.name) as i32).unwrap();
        writeln!(out, "    pub const SET_VAL_BASE: u32 = {:#010X};", get_field(&m.fields, "LockSetValBase", &m.name)).unwrap();
        writeln!(out, "    pub const SET_VAL_STRIDE: u32 = {:#06X};", get_field(&m.fields, "LockSetValOff", &m.name)).unwrap();
        writeln!(out, "}}\n").unwrap();
    }

    fs::write(out_dir.join("gen_aiert_locks.rs"), out).unwrap();
}

fn gen_aiert_ports(port_maps: &[PortMapData], out_dir: &Path) {
    let mut out = gen_header("aie-rt xaiemlgbl_reginit.c stream switch port maps");

    // Port type enum
    writeln!(out, "/// Port type enum matching aie-rt XAie_StrmSwPortType.").unwrap();
    writeln!(out, "#[derive(Debug, Clone, Copy, PartialEq, Eq)]").unwrap();
    writeln!(out, "#[repr(u8)]").unwrap();
    writeln!(out, "pub enum AieRtPortType {{").unwrap();
    writeln!(out, "    Core = 0, Dma = 1, Ctrl = 2, Fifo = 3,").unwrap();
    writeln!(out, "    South = 4, West = 5, North = 6, East = 7, Trace = 8,").unwrap();
    writeln!(out, "}}\n").unwrap();

    for pm in port_maps {
        let const_name = port_map_rust_name(&pm.name);
        writeln!(out, "/// {} (from aie-rt {})", const_name, pm.name).unwrap();
        writeln!(
            out,
            "pub const {}: &[(AieRtPortType, u8)] = &[",
            const_name
        )
        .unwrap();

        for (i, (port_type, port_num)) in pm.entries.iter().enumerate() {
            let rust_variant = match port_type.as_str() {
                "CORE" => "Core",
                "DMA" => "Dma",
                "CTRL" => "Ctrl",
                "FIFO" => "Fifo",
                "SOUTH" => "South",
                "WEST" => "West",
                "NORTH" => "North",
                "EAST" => "East",
                "TRACE" => "Trace",
                other => panic!("Unknown port type '{}' in {}", other, pm.name),
            };
            writeln!(
                out,
                "    (AieRtPortType::{}, {}), // {}",
                rust_variant, port_num, i
            )
            .unwrap();
        }

        writeln!(out, "];\n").unwrap();
    }

    fs::write(out_dir.join("gen_aiert_ports.rs"), out).unwrap();
}

/// Write stub files when aie-rt is not available.
fn write_aiert_stubs(out_dir: &Path) {
    let dma_stub = "\
// aie-rt not available -- stub file.
pub mod memtile_dma {
    pub const BD_BASE: u32 = 0x000A0000;
    pub const BD_STRIDE: u32 = 0x0020;
    pub const NUM_BDS: usize = 48;
    pub const NUM_LOCKS: usize = 192;
    pub const START_QUEUE_BASE: u32 = 0x000A0604;
    pub const CH_CTRL_BASE: u32 = 0x000A0600;
    pub const NUM_CHANNELS: usize = 6;
    pub const CH_STRIDE: u32 = 0x0008;
    pub const CH_STATUS_BASE: u32 = 0x000A0660;
    pub const CH_STATUS_STRIDE: u32 = 0x0020;
    pub const NUM_ADDR_DIM: usize = 4;
}
pub mod compute_dma {
    pub const BD_BASE: u32 = 0x0001D000;
    pub const BD_STRIDE: u32 = 0x0020;
    pub const NUM_BDS: usize = 16;
    pub const NUM_LOCKS: usize = 16;
    pub const START_QUEUE_BASE: u32 = 0x0001DE04;
    pub const CH_CTRL_BASE: u32 = 0x0001DE00;
    pub const NUM_CHANNELS: usize = 2;
    pub const CH_STRIDE: u32 = 0x0008;
    pub const CH_STATUS_BASE: u32 = 0x0001DF00;
    pub const CH_STATUS_STRIDE: u32 = 0x0010;
    pub const NUM_ADDR_DIM: usize = 3;
}
pub mod shim_dma {
    pub const BD_BASE: u32 = 0x0001D000;
    pub const BD_STRIDE: u32 = 0x0020;
    pub const NUM_BDS: usize = 16;
    pub const NUM_LOCKS: usize = 16;
    pub const START_QUEUE_BASE: u32 = 0x0001D204;
    pub const CH_CTRL_BASE: u32 = 0x0001D200;
    pub const NUM_CHANNELS: usize = 2;
    pub const CH_STRIDE: u32 = 0x0008;
    pub const CH_STATUS_BASE: u32 = 0x0001D220;
    pub const CH_STATUS_STRIDE: u32 = 0x0008;
    pub const NUM_ADDR_DIM: usize = 3;
}
";

    let locks_stub = "\
// aie-rt not available -- stub file.
pub mod compute_locks {
    pub const BASE: u32 = 0x00040000;
    pub const NUM_LOCKS: usize = 16;
    pub const LOCK_ID_STRIDE: u32 = 0x0400;
    pub const REL_ACQ_OFFSET: u32 = 0x0200;
    pub const LOCK_VAL_OFFSET: u32 = 0x0004;
    pub const VAL_UPPER_BOUND: i32 = 63;
    pub const VAL_LOWER_BOUND: i32 = -64;
    pub const SET_VAL_BASE: u32 = 0x0001F000;
    pub const SET_VAL_STRIDE: u32 = 0x0010;
}
pub mod shim_locks {
    pub const BASE: u32 = 0x00040000;
    pub const NUM_LOCKS: usize = 16;
    pub const LOCK_ID_STRIDE: u32 = 0x0400;
    pub const REL_ACQ_OFFSET: u32 = 0x0200;
    pub const LOCK_VAL_OFFSET: u32 = 0x0004;
    pub const VAL_UPPER_BOUND: i32 = 63;
    pub const VAL_LOWER_BOUND: i32 = -64;
    pub const SET_VAL_BASE: u32 = 0x00014000;
    pub const SET_VAL_STRIDE: u32 = 0x0010;
}
pub mod memtile_locks {
    pub const BASE: u32 = 0x000D0000;
    pub const NUM_LOCKS: usize = 64;
    pub const LOCK_ID_STRIDE: u32 = 0x0400;
    pub const REL_ACQ_OFFSET: u32 = 0x0200;
    pub const LOCK_VAL_OFFSET: u32 = 0x0004;
    pub const VAL_UPPER_BOUND: i32 = 63;
    pub const VAL_LOWER_BOUND: i32 = -64;
    pub const SET_VAL_BASE: u32 = 0x000C0000;
    pub const SET_VAL_STRIDE: u32 = 0x0010;
}
";

    // Stub for ports is more involved but still needed for compilation
    let ports_stub = "\
// aie-rt not available -- stub file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AieRtPortType {
    Core = 0, Dma = 1, Ctrl = 2, Fifo = 3,
    South = 4, West = 5, North = 6, East = 7, Trace = 8,
}

pub const COMPUTE_MASTER_PORTS: &[(AieRtPortType, u8)] = &[];
pub const COMPUTE_SLAVE_PORTS: &[(AieRtPortType, u8)] = &[];
pub const SHIM_MASTER_PORTS: &[(AieRtPortType, u8)] = &[];
pub const SHIM_SLAVE_PORTS: &[(AieRtPortType, u8)] = &[];
pub const MEMTILE_MASTER_PORTS: &[(AieRtPortType, u8)] = &[];
pub const MEMTILE_SLAVE_PORTS: &[(AieRtPortType, u8)] = &[];
";

    fs::write(out_dir.join("gen_aiert_dma.rs"), dma_stub).unwrap();
    fs::write(out_dir.join("gen_aiert_locks.rs"), locks_stub).unwrap();
    fs::write(out_dir.join("gen_aiert_ports.rs"), ports_stub).unwrap();
}

/// Compile `crates/xdna-archspec/decoder_ffi/aie2_decoder.cpp` and link the LLVM AIE libraries.
///
/// This gives the emulator access to LLVM's MCDisassembler for perfect
/// instruction decoding, including TRY_DECODE register class validation
/// that our bytecode interpreter cannot replicate.
fn compile_llvm_decoder_ffi(llvm_aie_path: &Path) {
    let llvm_build = llvm_aie_path.join("build");
    let llvm_src = llvm_aie_path.join("llvm");
    let llvm_config = llvm_build.join("bin/llvm-config");

    if !llvm_config.exists() {
        println!(
            "cargo:warning=LLVM decoder FFI: llvm-config not found at {} -- skipping",
            llvm_config.display()
        );
        return;
    }

    let decoder_cpp = "crates/xdna-archspec/decoder_ffi/aie2_decoder.cpp";
    println!("cargo:rerun-if-changed={}", decoder_cpp);
    println!("cargo:rerun-if-changed=crates/xdna-archspec/decoder_ffi/aie2_decoder.h");

    // Compile aie2_decoder.cpp with the cc crate.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .file(decoder_cpp)
        .include(llvm_build.join("include"))
        .include(llvm_src.join("include"))
        .include(llvm_build.join("lib/Target/AIE"))
        .include(llvm_src.join("lib/Target/AIE"))
        .define("__STDC_LIMIT_MACROS", None)
        .define("__STDC_CONSTANT_MACROS", None)
        .define("_GNU_SOURCE", None)
        .flag("-fno-exceptions")
        .flag("-funwind-tables")
        .compile("aie2_decoder");

    // Link LLVM libraries. Use llvm-config to get the authoritative list.
    let llvm_libdir = run_llvm_config(&llvm_config, &["--libdir"]);
    println!("cargo:rustc-link-search=native={}", llvm_libdir.trim());

    // Parse --libs aie output: "-lLLVMAIEDisassembler -lLLVMMCDisassembler ..."
    let libs_output = run_llvm_config(&llvm_config, &["--libs", "aie"]);
    for token in libs_output.split_whitespace() {
        if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=static={}", lib);
        }
    }

    // System libraries (dynamic): "-lrt -ldl -lm -lz -lzstd -lxml2"
    let syslibs_output = run_llvm_config(&llvm_config, &["--system-libs"]);
    for token in syslibs_output.split_whitespace() {
        if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={}", lib);
        }
    }

    // C++ standard library
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// Run llvm-config with the given arguments and return stdout as a String.
fn run_llvm_config(llvm_config: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new(llvm_config)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run llvm-config: {}", e));
    if !output.status.success() {
        panic!(
            "llvm-config {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).unwrap()
}
