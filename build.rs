//! Build-time code generation from the NPU architecture specification.
//!
//! This build script constructs the full ArchModel at compile time (device
//! topology + AM025 register database, cross-validated via Confirmed<T>) and
//! generates Rust source files with validated `const` definitions.
//!
//! Generated files (written to `$OUT_DIR/`):
//! - `gen_arch.rs`          -- comprehensive architecture constants (included by `arch` module)
//! - `gen_subsystems.rs`    -- per-tile-type subsystem address ranges (included by `arch::subsystem`)
//! - `gen_core_module.rs`   -- core register offsets (included by `registers_spec.rs`)
//! - `gen_memory_lock.rs`   -- memory module lock constants (included by `registers_spec.rs`)
//! - `gen_memtile_lock.rs`  -- mem tile lock constants (included by `registers_spec.rs`)
//! - `gen_stream_ports.rs`  -- port type arrays (included by `arch` module)
//! - `gen_stream_ranges.rs` -- port range constants (included by `arch` module)
//! - `gen_tablegen.rs`      -- complete instruction decoder tables (included by `tablegen` module)

#[path = "build_helpers/mod.rs"]
mod build_helpers;

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
// Port type constants -- mirrors arch::port_type for codegen output
// ============================================================================
// These are our encoding convention, not hardware-defined. The generated code
// references `port_type::*` which is defined in the same module scope.

const PT_CORE: u8 = 0;
const PT_FIFO: u8 = 1;
const PT_TRACE: u8 = 2;
const PT_CTRL: u8 = 3;
const PT_NORTH_BASE: u8 = 10;
const PT_SOUTH_BASE: u8 = 20;
const PT_EAST_BASE: u8 = 30;
const PT_WEST_BASE: u8 = 40;
const PT_DMA_BASE: u8 = 50;

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

    // Bridge script for trace events
    let bridge_path = manifest_dir.join("tools/mlir-aie-bridge.py");
    println!("cargo:rerun-if-changed={}", bridge_path.display());

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

    // Generate all files from the graph crate's parsed data.
    // gen_arch() has moved to xdna-archspec/build.rs (Task 4).
    // gen_subsystems() has moved to xdna-archspec/build.rs (Task 5).
    // gen_core_module() and gen_lock_request() have moved to xdna-archspec/build.rs (Task 6).
    let port_data = gen_stream_ports(&regdb, &out_dir);
    gen_stream_ranges(&regdb, &port_data, &out_dir);
    gen_trace_events(&bridge_path, &out_dir);

    // ========================================================================
    // Full TableGen extraction for decoder tables (build-time)
    // ========================================================================

    // Rebuild triggers for build_helpers source files
    for helper in &["mod.rs", "extract.rs", "records.rs", "semantics.rs",
                     "cpp_switch.rs", "bytecode.rs", "codegen.rs"] {
        println!("cargo:rerun-if-changed=build_helpers/{}", helper);
    }

    let aie2_td = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2.td");
    if aie2_td.exists() {
        println!("cargo:rerun-if-changed={}", aie2_td.display());
        for td in &["AIE2InstrFormats.td", "AIE2InstrInfo.td", "AIE2InstrPatterns.td",
                     "AIE2Slots.td", "AIE2Schedule.td", "AIE2RegisterInfo.td"] {
            let p = llvm_aie_path.join(format!("llvm/lib/Target/AIE/{}", td));
            if p.exists() { println!("cargo:rerun-if-changed={}", p.display()); }
        }
        let cpp = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2InstrInfo.cpp");
        if cpp.exists() { println!("cargo:rerun-if-changed={}", cpp.display()); }

        match build_helpers::extract::extract_all(llvm_aie_path) {
            Ok(output) => {
                println!("cargo:warning=TableGen: extracted {} instructions across {} slots",
                    output.total_instructions(), output.slot_count());
                build_helpers::codegen::generate_tablegen_file(&output, &out_dir);
            }
            Err(e) => {
                panic!("TableGen extraction failed:\n  {}\n\
                        Set LLVM_AIE_PATH to override.", e);
            }
        }
    } else {
        panic!("llvm-aie not found at {} -- required for build-time TableGen extraction.\n\
                Set LLVM_AIE_PATH to override.", llvm_aie_path.display());
    }

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
// Step 3: Stream switch port type arrays
// (Steps 1 and 2 moved to xdna-archspec/build.rs in Task 6)
// ============================================================================

/// Collected port arrays for deriving ranges in Step 5.
struct PortArrayData {
    compute_master: Vec<PortEntry>,
    compute_slave: Vec<PortEntry>,
    memtile_master: Vec<PortEntry>,
    memtile_slave: Vec<PortEntry>,
    shim_master: Vec<PortEntry>,
    shim_slave: Vec<PortEntry>,
}

struct PortEntry {
    port_type_value: u8,
    port_type_expr: String,
    suffix: String,
}

fn gen_stream_ports(regdb: &RegisterDb, out_dir: &Path) -> PortArrayData {
    let mut out = gen_header("AM025 Stream_Switch_*_Config registers");
    // Note: the generated code references `port_type::*` which is defined
    // in the `arch` module (lib.rs), before the include!() point.

    let compute_master =
        collect_port_array(regdb, "core", "Stream_Switch_Master_Config_");
    let compute_slave =
        collect_port_array(regdb, "core", "Stream_Switch_Slave_Config_");
    let memtile_master =
        collect_port_array(regdb, "memory_tile", "Stream_Switch_Master_Config_");
    let memtile_slave =
        collect_port_array(regdb, "memory_tile", "Stream_Switch_Slave_Config_");
    let shim_master =
        collect_port_array(regdb, "shim", "Stream_Switch_Master_Config_");
    let shim_slave =
        collect_port_array(regdb, "shim", "Stream_Switch_Slave_Config_");

    write_port_array(
        &mut out,
        "COMPUTE_MASTER_PORTS",
        "Compute tile stream switch master",
        "CORE_MODULE",
        &compute_master,
    );
    write_port_array(
        &mut out,
        "COMPUTE_SLAVE_PORTS",
        "Compute tile stream switch slave",
        "CORE_MODULE",
        &compute_slave,
    );
    write_port_array(
        &mut out,
        "MEMTILE_MASTER_PORTS",
        "MemTile stream switch master",
        "MEMORY_TILE_MODULE",
        &memtile_master,
    );
    write_port_array(
        &mut out,
        "MEMTILE_SLAVE_PORTS",
        "MemTile stream switch slave",
        "MEMORY_TILE_MODULE",
        &memtile_slave,
    );
    write_port_array(
        &mut out,
        "SHIM_MASTER_PORTS",
        "Shim tile stream switch master",
        "PL_MODULE",
        &shim_master,
    );
    write_port_array(
        &mut out,
        "SHIM_SLAVE_PORTS",
        "Shim tile stream switch slave",
        "PL_MODULE",
        &shim_slave,
    );

    fs::write(out_dir.join("gen_stream_ports.rs"), out).unwrap();

    PortArrayData {
        compute_master,
        compute_slave,
        memtile_master,
        memtile_slave,
        shim_master,
        shim_slave,
    }
}

/// Collect and sort port entries for a given module and register prefix.
fn collect_port_array(
    regdb: &RegisterDb,
    module_name: &str,
    prefix: &str,
) -> Vec<PortEntry> {
    let module = regdb
        .module(module_name)
        .unwrap_or_else(|| panic!("AM025 JSON missing '{}' module", module_name));

    let mut entries: Vec<(u32, PortEntry)> = module
        .registers
        .iter()
        .filter(|r| r.name.starts_with(prefix))
        .map(|r| {
            let suffix = r.name[prefix.len()..].to_string();
            let (port_type_value, port_type_expr) = suffix_to_port_type(&suffix);
            (
                r.offset,
                PortEntry {
                    port_type_value,
                    port_type_expr,
                    suffix,
                },
            )
        })
        .collect();

    // Sort by offset to get the canonical hardware port ordering
    entries.sort_by_key(|(offset, _)| *offset);
    entries.into_iter().map(|(_, entry)| entry).collect()
}

/// Map a register name suffix to a port type value and Rust expression.
fn suffix_to_port_type(suffix: &str) -> (u8, String) {
    // Special names first
    if suffix == "AIE_Core0" {
        return (PT_CORE, "port_type::CORE".to_string());
    }
    if suffix == "Tile_Ctrl" {
        return (PT_CTRL, "port_type::CTRL".to_string());
    }
    if suffix.starts_with("FIFO") {
        return (PT_FIFO, "port_type::FIFO".to_string());
    }
    if suffix == "AIE_Trace" || suffix == "Mem_Trace" || suffix == "Trace" {
        return (PT_TRACE, "port_type::TRACE".to_string());
    }

    // Directional ports: "North0" or "North_0" formats
    let directions: &[(&str, u8, &str)] = &[
        ("North", PT_NORTH_BASE, "north"),
        ("South", PT_SOUTH_BASE, "south"),
        ("East", PT_EAST_BASE, "east"),
        ("West", PT_WEST_BASE, "west"),
        ("DMA", PT_DMA_BASE, "dma"),
    ];

    for &(dir_prefix, base, fn_name) in directions {
        if let Some(rest) = suffix.strip_prefix(dir_prefix) {
            // Handle both "North0" and "North_0" formats
            let num_str = rest.strip_prefix('_').unwrap_or(rest);
            let n: u8 = num_str.parse().unwrap_or_else(|e| {
                panic!("Cannot parse port index from suffix '{}': {}", suffix, e)
            });
            return (base + n, format!("port_type::{}({})", fn_name, n));
        }
    }

    panic!(
        "Unknown stream switch config suffix: '{}'. \
         Expected AIE_Core0, Tile_Ctrl, FIFO*, *Trace, or Direction[_]N.",
        suffix
    );
}

/// Write a single port type array constant.
fn write_port_array(
    out: &mut String,
    const_name: &str,
    doc_prefix: &str,
    am025_section: &str,
    entries: &[PortEntry],
) {
    writeln!(
        out,
        "/// {} port layout (AM025 {}).",
        doc_prefix, am025_section
    )
    .unwrap();
    writeln!(out, "pub const {}: &[u8] = &[", const_name).unwrap();
    for (i, entry) in entries.iter().enumerate() {
        writeln!(
            out,
            "    {:<26} // {}: {}",
            format!("{},", entry.port_type_expr),
            i,
            entry.suffix
        )
        .unwrap();
    }
    writeln!(out, "];\n").unwrap();
}

// ============================================================================
// Step 5: Stream switch port ranges and config bits
// ============================================================================

fn gen_stream_ranges(
    regdb: &RegisterDb,
    port_data: &PortArrayData,
    out_dir: &Path,
) {
    let mut out = gen_header("AM025 stream switch port ranges");

    // Extract ENABLE_BIT from any Master_Config register's Master_Enable field
    let enable_bit = find_master_enable_bit(regdb);
    writeln!(
        out,
        "/// Stream switch master enable bit position (AM025: Master_Enable)"
    )
    .unwrap();
    writeln!(out, "pub const ENABLE_BIT: u32 = {};", enable_bit).unwrap();
    writeln!(out).unwrap();
    // SLAVE_SELECT_MASK stays hardcoded -- it's a sub-field within the
    // Configuration field that the JSON doesn't break out separately.
    writeln!(
        out,
        "/// Slave select mask (5-bit sub-field, not individually specified in AM025)"
    )
    .unwrap();
    writeln!(out, "pub const SLAVE_SELECT_MASK: u32 = 0x1F;").unwrap();
    writeln!(out).unwrap();

    // Shim port ranges
    writeln!(out, "/// Shim tile port ranges").unwrap();
    writeln!(out, "pub mod shim {{").unwrap();
    write_direction_ranges(&mut out, "NORTH", &port_data.shim_master, &port_data.shim_slave);
    write_direction_ranges(&mut out, "SOUTH", &port_data.shim_master, &port_data.shim_slave);
    write_direction_ranges(&mut out, "EAST", &port_data.shim_master, &port_data.shim_slave);
    write_direction_ranges(&mut out, "WEST", &port_data.shim_master, &port_data.shim_slave);
    write_bundle_ranges(&mut out, "TRACE", PT_TRACE, &port_data.shim_master, &port_data.shim_slave);
    writeln!(out, "}}\n").unwrap();

    // MemTile port ranges
    writeln!(out, "/// MemTile port ranges").unwrap();
    writeln!(out, "pub mod mem_tile {{").unwrap();
    write_direction_ranges(
        &mut out,
        "SOUTH",
        &port_data.memtile_master,
        &port_data.memtile_slave,
    );
    write_direction_ranges(
        &mut out,
        "NORTH",
        &port_data.memtile_master,
        &port_data.memtile_slave,
    );
    write_bundle_ranges(&mut out, "DMA", PT_DMA_BASE, &port_data.memtile_master, &port_data.memtile_slave);
    write_bundle_ranges(&mut out, "TRACE", PT_TRACE, &port_data.memtile_master, &port_data.memtile_slave);
    writeln!(out, "}}\n").unwrap();

    // Compute tile port ranges
    writeln!(out, "/// Compute tile port ranges").unwrap();
    writeln!(out, "pub mod compute {{").unwrap();
    write_direction_ranges(
        &mut out,
        "SOUTH",
        &port_data.compute_master,
        &port_data.compute_slave,
    );
    write_direction_ranges(
        &mut out,
        "NORTH",
        &port_data.compute_master,
        &port_data.compute_slave,
    );
    write_direction_ranges(&mut out, "EAST", &port_data.compute_master, &port_data.compute_slave);
    write_direction_ranges(&mut out, "WEST", &port_data.compute_master, &port_data.compute_slave);
    write_bundle_ranges(&mut out, "DMA", PT_DMA_BASE, &port_data.compute_master, &port_data.compute_slave);
    write_bundle_ranges(&mut out, "TRACE", PT_TRACE, &port_data.compute_master, &port_data.compute_slave);
    writeln!(out, "}}").unwrap();

    fs::write(out_dir.join("gen_stream_ranges.rs"), out).unwrap();
}

/// Find the first and last port indices for a given direction in the arrays,
/// and write MASTER_START/END and SLAVE_START/END constants.
fn write_direction_ranges(
    out: &mut String,
    direction: &str,
    master_ports: &[PortEntry],
    slave_ports: &[PortEntry],
) {
    let dir_base = match direction {
        "NORTH" => PT_NORTH_BASE,
        "SOUTH" => PT_SOUTH_BASE,
        "EAST" => PT_EAST_BASE,
        "WEST" => PT_WEST_BASE,
        _ => panic!("Unknown direction: {}", direction),
    };

    // Find index range for this direction in master ports
    if let Some((start, end)) = find_port_range(master_ports, dir_base) {
        writeln!(
            out,
            "    /// {}-facing master ports: {}-{} ({} ports)",
            direction.to_lowercase(),
            start,
            end,
            end - start + 1
        )
        .unwrap();
        writeln!(
            out,
            "    pub const {}_MASTER_START: u8 = {};",
            direction, start
        )
        .unwrap();
        writeln!(
            out,
            "    pub const {}_MASTER_END: u8 = {};",
            direction, end
        )
        .unwrap();
    }

    // Find index range for this direction in slave ports
    if let Some((start, end)) = find_port_range(slave_ports, dir_base) {
        writeln!(
            out,
            "    /// {}-facing slave ports: {}-{} ({} ports)",
            direction.to_lowercase(),
            start,
            end,
            end - start + 1
        )
        .unwrap();
        writeln!(
            out,
            "    pub const {}_SLAVE_START: u8 = {};",
            direction, start
        )
        .unwrap();
        writeln!(
            out,
            "    pub const {}_SLAVE_END: u8 = {};",
            direction, end
        )
        .unwrap();
    }
}

/// Find port index range for a bundle type and write START/END constants.
///
/// For ranged types (DMA, directional), matches `base <= value < base + 10`.
/// For single-value types (TRACE, CORE, FIFO), matches `value == base` exactly.
fn write_bundle_ranges(
    out: &mut String,
    bundle: &str,
    base: u8,
    master_ports: &[PortEntry],
    slave_ports: &[PortEntry],
) {
    // TRACE/CORE/FIFO are single-value types; directional and DMA are ranged
    let is_ranged = base >= 10;

    if let Some((start, end)) = find_port_range_flex(master_ports, base, is_ranged) {
        writeln!(
            out,
            "    /// {} master ports: {}-{} ({} ports)",
            bundle.to_lowercase(),
            start,
            end,
            end - start + 1
        )
        .unwrap();
        writeln!(out, "    pub const {}_MASTER_START: u8 = {};", bundle, start).unwrap();
        writeln!(out, "    pub const {}_MASTER_END: u8 = {};", bundle, end).unwrap();
    }

    if let Some((start, end)) = find_port_range_flex(slave_ports, base, is_ranged) {
        writeln!(
            out,
            "    /// {} slave ports: {}-{} ({} ports)",
            bundle.to_lowercase(),
            start,
            end,
            end - start + 1
        )
        .unwrap();
        writeln!(out, "    pub const {}_SLAVE_START: u8 = {};", bundle, start).unwrap();
        writeln!(out, "    pub const {}_SLAVE_END: u8 = {};", bundle, end).unwrap();
    }
}

/// Find the first and last indices matching a port type.
///
/// When `ranged` is true, matches `base <= value < base + 10` (for directional/DMA).
/// When `ranged` is false, matches `value == base` exactly (for TRACE/CORE/FIFO).
fn find_port_range_flex(ports: &[PortEntry], base: u8, ranged: bool) -> Option<(u8, u8)> {
    let mut first = None;
    let mut last = None;

    for (i, entry) in ports.iter().enumerate() {
        let matches = if ranged {
            entry.port_type_value >= base && entry.port_type_value < base + 10
        } else {
            entry.port_type_value == base
        };
        if matches {
            if first.is_none() {
                first = Some(i as u8);
            }
            last = Some(i as u8);
        }
    }

    first.map(|f| (f, last.unwrap()))
}

/// Find the first and last indices where port_type_value has the given base
/// (e.g., PT_NORTH_BASE for any north(N) port).
fn find_port_range(ports: &[PortEntry], base: u8) -> Option<(u8, u8)> {
    let mut first = None;
    let mut last = None;

    for (i, entry) in ports.iter().enumerate() {
        if entry.port_type_value >= base && entry.port_type_value < base + 10 {
            if first.is_none() {
                first = Some(i as u8);
            }
            last = Some(i as u8);
        }
    }

    first.map(|f| (f, last.unwrap()))
}

/// Find the Master_Enable bit position from any Stream_Switch_Master_Config register.
fn find_master_enable_bit(regdb: &RegisterDb) -> u32 {
    for module in regdb.modules.values() {
        for reg in &module.registers {
            if reg.name.starts_with("Stream_Switch_Master_Config_") {
                if let Some(field) = reg.field("Master_Enable") {
                    // Graph crate's BitField has lsb/msb already parsed.
                    assert_eq!(
                        field.lsb, field.msb,
                        "Master_Enable should be a single bit"
                    );
                    return field.lsb as u32;
                }
            }
        }
    }
    panic!("Master_Enable bit field not found in any Stream_Switch_Master_Config register");
}

// ============================================================================
// Step 6: Trace event codes from mlir-aie bridge
// ============================================================================

fn gen_trace_events(bridge_path: &Path, out_dir: &Path) {
    use std::process::Command;

    if !bridge_path.exists() {
        write_trace_event_stub(out_dir);
        return;
    }

    // Find Python interpreter: prefer ironenv, fall back to system python3.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let npu_work = manifest_dir.parent().unwrap_or(Path::new("."));
    let ironenv_python = npu_work.join("mlir-aie/ironenv/bin/python3");
    let python = if ironenv_python.exists() {
        ironenv_python
    } else {
        PathBuf::from("python3")
    };

    let output = Command::new(&python)
        .arg(bridge_path)
        .arg("trace-events")
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!(
                "cargo:warning=mlir-aie bridge trace-events failed ({}), using stub: {}",
                o.status, stderr
            );
            write_trace_event_stub(out_dir);
            return;
        }
        Err(e) => {
            eprintln!(
                "cargo:warning=Could not run mlir-aie bridge ({}), using stub",
                e
            );
            write_trace_event_stub(out_dir);
            return;
        }
    };

    let json: serde_json::Value = match serde_json::from_slice(&output.stdout) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "cargo:warning=mlir-aie bridge returned invalid JSON ({}), using stub",
                e
            );
            write_trace_event_stub(out_dir);
            return;
        }
    };

    let enums = match json["enums"].as_object() {
        Some(e) => e,
        None => {
            eprintln!("cargo:warning=mlir-aie bridge JSON missing 'enums', using stub");
            write_trace_event_stub(out_dir);
            return;
        }
    };

    let mut out = gen_header("mlir-aie trace event enums (via mlir-aie-bridge.py)");
    writeln!(out, "// Source: mlir-aie Python API (aie.utils.trace_events)").unwrap();
    writeln!(out).unwrap();

    // Generate modules for each event enum.
    let enum_order = ["CoreEvent", "MemEvent", "MemTileEvent", "ShimTileEvent"];
    let mod_names = ["core_events", "mem_events", "memtile_events", "shim_events"];
    let fn_names = [
        "core_event_name",
        "mem_event_name",
        "memtile_event_name",
        "shim_event_name",
    ];

    for (i, enum_name) in enum_order.iter().enumerate() {
        let mod_name = mod_names[i];
        let fn_name = fn_names[i];

        if let Some(events) = enums.get(*enum_name).and_then(|v| v.as_object()) {
            // Collect and sort by value for deterministic output.
            let mut entries: Vec<(String, u64)> = events
                .iter()
                .filter_map(|(name, val)| val.as_u64().map(|v| (name.clone(), v)))
                .collect();
            entries.sort_by_key(|(_, v)| *v);

            // Module with const definitions.
            writeln!(out, "/// {} event codes from mlir-aie.", enum_name).unwrap();
            writeln!(out, "#[allow(dead_code)]").unwrap();
            writeln!(out, "pub mod {} {{", mod_name).unwrap();
            for (name, value) in &entries {
                // Sanitize names: replace leading digits, etc.
                let const_name = sanitize_const_name(name).to_ascii_uppercase();
                writeln!(out, "    pub const {}: u8 = {};", const_name, value).unwrap();
            }
            writeln!(out, "}}\n").unwrap();

            // Name lookup function.
            writeln!(
                out,
                "/// Look up {} event name by hardware code.",
                enum_name
            )
            .unwrap();
            writeln!(out, "pub fn {}(code: u8) -> &'static str {{", fn_name).unwrap();
            writeln!(out, "    match code {{").unwrap();
            for (name, value) in &entries {
                writeln!(out, "        {} => \"{}\",", value, name).unwrap();
            }
            writeln!(out, "        _ => \"UNKNOWN\",").unwrap();
            writeln!(out, "    }}").unwrap();
            writeln!(out, "}}\n").unwrap();
        }
    }

    fs::write(out_dir.join("trace_event_codes.rs"), out).unwrap();
}

/// Write a stub file when the bridge is not available.
fn write_trace_event_stub(out_dir: &Path) {
    let stub = "\
// Trace event codes not generated (mlir-aie bridge not available).
// Rebuild with mlir-aie installed for full event code tables.

pub fn core_event_name(_code: u8) -> &'static str { \"UNKNOWN\" }
pub fn mem_event_name(_code: u8) -> &'static str { \"UNKNOWN\" }
pub fn memtile_event_name(_code: u8) -> &'static str { \"UNKNOWN\" }
pub fn shim_event_name(_code: u8) -> &'static str { \"UNKNOWN\" }
";
    fs::write(out_dir.join("trace_event_codes.rs"), stub).unwrap();
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

/// Sanitize a Python enum name for use as a Rust const.
fn sanitize_const_name(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            result.push(ch);
        } else {
            result.push('_');
        }
    }
    // Rust consts can't start with a digit.
    if result.starts_with(|c: char| c.is_ascii_digit()) {
        result.insert(0, '_');
    }
    result
}

/// Compile `decoder_ffi/aie2_decoder.cpp` and link the LLVM AIE libraries.
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

    let decoder_cpp = "decoder_ffi/aie2_decoder.cpp";
    println!("cargo:rerun-if-changed={}", decoder_cpp);
    println!("cargo:rerun-if-changed=decoder_ffi/aie2_decoder.h");

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
