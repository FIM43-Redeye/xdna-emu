//! Build-time code generation from AMD AM025 register database and device model.
//!
//! This build script reads the same JSON sources that the runtime `regdb.rs`
//! uses, and generates Rust source files with `const` definitions that replace
//! hand-transcribed register constants.
//!
//! Generated files (written to `$OUT_DIR/`):
//! - `gen_core_module.rs`   -- core register offsets (included by `registers_spec.rs`)
//! - `gen_memory_lock.rs`   -- memory module lock constants (included by `registers_spec.rs`)
//! - `gen_memtile_lock.rs`  -- mem tile lock constants (included by `registers_spec.rs`)
//! - `gen_data_memory.rs`   -- data memory size constants (included by `registers_spec.rs`)
//! - `gen_stream_ports.rs`  -- port type arrays (included by `aie2_spec.rs`)
//! - `gen_stream_ranges.rs` -- port range constants (included by `aie2_spec.rs`)

use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};

// ============================================================================
// JSON deserialization types -- AM025 register database
// ============================================================================
// Cannot share types with regdb.rs because build scripts compile separately.

#[derive(Deserialize)]
struct RawRegisterDb {
    #[allow(dead_code)]
    version: String,
    modules: HashMap<String, RawModule>,
}

#[derive(Deserialize)]
struct RawModule {
    registers: Vec<RawRegister>,
}

#[derive(Deserialize)]
struct RawRegister {
    name: String,
    offset: String,
    description: String,
    bit_fields: Vec<RawBitField>,
}

#[derive(Deserialize)]
struct RawBitField {
    name: String,
    bit_range: Vec<u32>,
}

// ============================================================================
// JSON deserialization types -- device model
// ============================================================================

#[derive(Deserialize)]
struct DeviceModelSet {
    devices: HashMap<String, DeviceModel>,
}

#[derive(Deserialize)]
struct DeviceModel {
    local_memory_size: u64,
    mem_tile_size: u64,
}

// ============================================================================
// Port type constants -- mirrors aie2_spec::port_type for codegen output
// ============================================================================
// These are our encoding convention, not hardware-defined. The generated code
// references `port_type::*` which is defined in the same module scope.

const PT_CORE: u8 = 0;
const PT_FIFO: u8 = 1;
const PT_TRACE: u8 = 2;
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

    // Load AM025 register database
    let am025_text = fs::read_to_string(&am025_path).unwrap_or_else(|e| {
        panic!(
            "Cannot read AM025 register database at {}:\n  {}\n\
             Set MLIR_AIE_PATH to override the mlir-aie location.",
            am025_path.display(),
            e
        )
    });
    let regdb: RawRegisterDb = serde_json::from_str(&am025_text)
        .unwrap_or_else(|e| panic!("Failed to parse AM025 JSON: {}", e));

    // Load device model
    let device_text = fs::read_to_string(&device_model_path).unwrap_or_else(|e| {
        panic!(
            "Cannot read device model at {}: {}",
            device_model_path.display(),
            e
        )
    });
    let device_models: DeviceModelSet = serde_json::from_str(&device_text)
        .unwrap_or_else(|e| panic!("Failed to parse device model JSON: {}", e));

    // Bridge script for trace events
    let bridge_path = manifest_dir.join("tools/mlir-aie-bridge.py");
    println!("cargo:rerun-if-changed={}", bridge_path.display());

    // Generate all files
    gen_core_module(&regdb, &out_dir);
    gen_lock_request(&regdb, &out_dir, "memory", "gen_memory_lock.rs");
    gen_lock_request(&regdb, &out_dir, "memory_tile", "gen_memtile_lock.rs");
    gen_data_memory(&device_models, &out_dir);
    let port_data = gen_stream_ports(&regdb, &out_dir);
    gen_stream_ranges(&regdb, &port_data, &out_dir);
    gen_trace_events(&bridge_path, &out_dir);
}

// ============================================================================
// Helpers
// ============================================================================

/// Parse a hex offset string like "0x0000032000" to a u32 tile-local offset.
/// The JSON stores full 40-bit addresses; we mask to 20-bit tile-local offset.
fn parse_offset(s: &str) -> u32 {
    let s = s.trim_start_matches("0x").trim_start_matches("0X");
    let full = u64::from_str_radix(s, 16)
        .unwrap_or_else(|e| panic!("Bad hex offset '{}': {}", s, e));
    // Tile-local offset is the low 20 bits (0xFFFFF mask)
    (full & 0xFFFFF) as u32
}

/// File header comment for generated source.
fn gen_header(source_desc: &str) -> String {
    format!(
        "// Auto-generated by build.rs from {}.\n\
         // Do not edit manually.\n\n",
        source_desc
    )
}

// ============================================================================
// Step 1: Core module register offsets
// ============================================================================

fn gen_core_module(regdb: &RawRegisterDb, out_dir: &Path) {
    let core = regdb
        .modules
        .get("core")
        .expect("AM025 JSON missing 'core' module");

    // Map of JSON register name -> Rust constant name
    let register_map: &[(&str, &str)] = &[
        ("Core_Control", "CORE_CONTROL"),
        ("Core_Status", "CORE_STATUS"),
        ("Enable_Events", "CORE_ENABLE_EVENTS"),
        ("Reset_Event", "CORE_RESET_EVENT"),
        ("Debug_Control0", "CORE_DEBUG_CONTROL0"),
        ("Core_PC", "CORE_PC"),
        ("Core_SP", "CORE_SP"),
        ("Core_LR", "CORE_LR"),
        ("Tile_Control", "TILE_CONTROL"),
        ("Memory_Control", "MEMORY_CONTROL"),
    ];

    let mut out = gen_header("AM025 core module registers");

    for &(json_name, const_name) in register_map {
        let reg = core
            .registers
            .iter()
            .find(|r| r.name == json_name)
            .unwrap_or_else(|| {
                panic!("Core register '{}' not found in AM025 JSON", json_name)
            });
        let offset = parse_offset(&reg.offset);
        writeln!(out, "/// {} (AM025: {})", const_name, json_name).unwrap();
        writeln!(out, "pub const {}: u32 = {:#07X};", const_name, offset).unwrap();
        writeln!(out).unwrap();
    }

    // Compute OFFSET_START / OFFSET_END from all non-stream-switch core registers.
    // The core module address space runs from the lowest core register offset
    // to just below the stream switch region (which starts at 0x3F000).
    let core_proper_offsets: Vec<u32> = core
        .registers
        .iter()
        .filter(|r| !r.name.starts_with("Stream_Switch"))
        .map(|r| parse_offset(&r.offset))
        .filter(|&o| (0x30000..0x3F000).contains(&o))
        .collect();

    assert!(
        !core_proper_offsets.is_empty(),
        "No core-module registers found in 0x30000..0x3F000"
    );

    let min_offset = *core_proper_offsets.iter().min().unwrap();
    // Round start down to 4K page boundary for clean dispatch
    let offset_start = min_offset & !0xFFF;
    // End just before stream switch region (0x3F000)
    let offset_end = 0x3EFFF_u32;

    writeln!(out, "/// Core module offset range start (derived from AM025)").unwrap();
    writeln!(out, "pub const OFFSET_START: u32 = {:#07X};", offset_start).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "/// Core module offset range end (before stream switch at 0x3F000)"
    )
    .unwrap();
    writeln!(out, "pub const OFFSET_END: u32 = {:#07X};", offset_end).unwrap();

    fs::write(out_dir.join("gen_core_module.rs"), out).unwrap();
}

// ============================================================================
// Step 2: Lock_Request constants
// ============================================================================

fn gen_lock_request(
    regdb: &RawRegisterDb,
    out_dir: &Path,
    module_name: &str,
    output_file: &str,
) {
    let module = regdb
        .modules
        .get(module_name)
        .unwrap_or_else(|| panic!("AM025 JSON missing '{}' module", module_name));

    let lock_req = module
        .registers
        .iter()
        .find(|r| r.name == "Lock_Request")
        .unwrap_or_else(|| {
            panic!(
                "Lock_Request register not found in '{}' module",
                module_name
            )
        });

    let base_offset = parse_offset(&lock_req.offset);
    let desc = &lock_req.description;

    // Parse end address from description: "... address space: 0xBASE - 0xLAST, ..."
    // End (exclusive) = LAST + 4 (since registers are 4-byte aligned)
    let end_offset = parse_lock_end_address(desc, module_name);

    // Parse Lock_Id bit range: "Lock_Id [high:low]"
    let (id_high, id_low) = parse_desc_range(desc, "Lock_Id", module_name);
    let id_shift = id_low;
    let id_bits = id_high - id_low + 1;
    let id_mask = (1u32 << id_bits) - 1;

    // Parse Acq_Rel bit: "Acq_Rel (N)"
    let acq_rel_bit = parse_desc_single_bit(desc, "Acq_Rel", module_name);

    // Parse Change_Value range: "Change_Value [high:low]"
    let (cv_high, cv_low) = parse_desc_range(desc, "Change_Value", module_name);
    let cv_shift = cv_low;
    let cv_bits = cv_high - cv_low + 1;
    let cv_mask = (1u32 << cv_bits) - 1;

    let mut out = gen_header(&format!("AM025 {}/Lock_Request", module_name));

    writeln!(out, "/// Lock_Request base address").unwrap();
    writeln!(out, "pub const LOCK_REQUEST_BASE: u32 = {:#07X};", base_offset).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "/// Lock_Request end address (exclusive)").unwrap();
    writeln!(out, "pub const LOCK_REQUEST_END: u32 = {:#07X};", end_offset).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "/// Lock_Request address field: Lock_Id [{high}:{low}] ({bits} bits)",
        high = id_high,
        low = id_low,
        bits = id_bits
    )
    .unwrap();
    writeln!(out, "pub const LOCK_REQUEST_ID_SHIFT: u32 = {};", id_shift).unwrap();
    writeln!(out, "pub const LOCK_REQUEST_ID_MASK: u32 = {:#X};", id_mask).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "/// Lock_Request address field: Acq_Rel ({}) (1=acquire, 0=release)",
        acq_rel_bit
    )
    .unwrap();
    writeln!(
        out,
        "pub const LOCK_REQUEST_ACQ_REL_BIT: u32 = {};",
        acq_rel_bit
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "/// Lock_Request address field: Change_Value [{high}:{low}] ({bits} bits)",
        high = cv_high,
        low = cv_low,
        bits = cv_bits
    )
    .unwrap();
    writeln!(out, "pub const LOCK_REQUEST_VALUE_SHIFT: u32 = {};", cv_shift).unwrap();
    writeln!(out, "pub const LOCK_REQUEST_VALUE_MASK: u32 = {:#X};", cv_mask).unwrap();

    fs::write(out_dir.join(output_file), out).unwrap();
}

/// Parse "... 0xBASE - 0xLAST, ..." from Lock_Request description.
/// Returns exclusive end (LAST + 4).
fn parse_lock_end_address(desc: &str, module_name: &str) -> u32 {
    // Pattern: "0xBASE - 0xLAST"
    let dash_idx = desc.find(" - 0x").unwrap_or_else(|| {
        panic!(
            "Lock_Request in '{}' has no ' - 0x' in description: {}",
            module_name, desc
        )
    });
    let after_dash = &desc[dash_idx + 3..]; // skip " - "
    let hex_str: String = after_dash
        .chars()
        .take_while(|c| c.is_ascii_hexdigit() || *c == 'x' || *c == 'X')
        .collect();
    let hex_str = hex_str.trim_start_matches("0x").trim_start_matches("0X");
    let last_addr = u32::from_str_radix(hex_str, 16).unwrap_or_else(|e| {
        panic!(
            "Bad end address in Lock_Request description for '{}': {}",
            module_name, e
        )
    });
    last_addr + 4 // exclusive end
}

/// Parse "Name [high:low]" from description text.
fn parse_desc_range(desc: &str, field_name: &str, module_name: &str) -> (u32, u32) {
    let pattern = format!("{} [", field_name);
    let start = desc.find(&pattern).unwrap_or_else(|| {
        panic!(
            "Field '{}' not found in Lock_Request description for '{}': {}",
            field_name, module_name, desc
        )
    });
    let after = &desc[start + pattern.len()..];
    let bracket_end = after.find(']').unwrap();
    let range_str = &after[..bracket_end];
    let parts: Vec<&str> = range_str.split(':').collect();
    assert_eq!(parts.len(), 2, "Expected high:low in '{}'", range_str);
    let high: u32 = parts[0].parse().unwrap();
    let low: u32 = parts[1].parse().unwrap();
    (high, low)
}

/// Parse "Name (bit)" from description text.
fn parse_desc_single_bit(desc: &str, field_name: &str, module_name: &str) -> u32 {
    let pattern = format!("{} (", field_name);
    let start = desc.find(&pattern).unwrap_or_else(|| {
        panic!(
            "Field '{}' not found in Lock_Request description for '{}': {}",
            field_name, module_name, desc
        )
    });
    let after = &desc[start + pattern.len()..];
    let paren_end = after.find(')').unwrap();
    after[..paren_end].parse().unwrap()
}

// ============================================================================
// Step 3: Data memory sizes
// ============================================================================

fn gen_data_memory(device_models: &DeviceModelSet, out_dir: &Path) {
    let npu1 = device_models
        .devices
        .get("npu1")
        .expect("Device model missing 'npu1' device");

    let compute_end = npu1.local_memory_size - 1;
    let memtile_end = npu1.mem_tile_size - 1;

    let mut out = gen_header("device model (npu1) memory sizes");

    writeln!(
        out,
        "/// Data memory end offset for compute tile ({} KB)",
        npu1.local_memory_size / 1024
    )
    .unwrap();
    writeln!(
        out,
        "pub const COMPUTE_DATA_MEMORY_END: u32 = {:#07X};",
        compute_end
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "/// Data memory end offset for memory tile ({} KB)",
        npu1.mem_tile_size / 1024
    )
    .unwrap();
    writeln!(
        out,
        "pub const MEM_TILE_DATA_MEMORY_END: u32 = {:#07X};",
        memtile_end
    )
    .unwrap();

    fs::write(out_dir.join("gen_data_memory.rs"), out).unwrap();
}

// ============================================================================
// Step 4: Stream switch port type arrays
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

fn gen_stream_ports(regdb: &RawRegisterDb, out_dir: &Path) -> PortArrayData {
    let mut out = gen_header("AM025 Stream_Switch_*_Config registers");
    // Note: the generated code references `port_type::*` which is defined
    // in the including module (aie2_spec.rs), before the include!() point.

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
    regdb: &RawRegisterDb,
    module_name: &str,
    prefix: &str,
) -> Vec<PortEntry> {
    let module = regdb
        .modules
        .get(module_name)
        .unwrap_or_else(|| panic!("AM025 JSON missing '{}' module", module_name));

    let mut entries: Vec<(u32, PortEntry)> = module
        .registers
        .iter()
        .filter(|r| r.name.starts_with(prefix))
        .map(|r| {
            let suffix = r.name[prefix.len()..].to_string();
            let (port_type_value, port_type_expr) = suffix_to_port_type(&suffix);
            let offset = parse_offset(&r.offset);
            (
                offset,
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
    if suffix == "AIE_Core0" || suffix == "Tile_Ctrl" {
        return (PT_CORE, "port_type::CORE".to_string());
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
    regdb: &RawRegisterDb,
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
fn find_master_enable_bit(regdb: &RawRegisterDb) -> u32 {
    for module in regdb.modules.values() {
        for reg in &module.registers {
            if reg.name.starts_with("Stream_Switch_Master_Config_") {
                for field in &reg.bit_fields {
                    if field.name == "Master_Enable" {
                        // bit_range is [high, low] for a range, or [bit] for single
                        let bit = if field.bit_range.len() == 2 {
                            assert_eq!(
                                field.bit_range[0], field.bit_range[1],
                                "Master_Enable should be a single bit"
                            );
                            field.bit_range[0]
                        } else {
                            field.bit_range[0]
                        };
                        return bit;
                    }
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
