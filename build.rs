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

    // Generate all files from the graph crate's parsed data.
    gen_arch(&arch_model, &out_dir);
    gen_subsystems(&arch_model, &out_dir);
    gen_core_module(&regdb, &out_dir);
    gen_lock_request(&regdb, &out_dir, "memory", "gen_memory_lock.rs");
    gen_lock_request(&regdb, &out_dir, "memory_tile", "gen_memtile_lock.rs");
    let port_data = gen_stream_ports(&regdb, &out_dir);
    gen_stream_ranges(&regdb, &port_data, &out_dir);
    gen_trace_events(&bridge_path, &out_dir);

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
        // Incremental cmake build (~2s when nothing changed)
        let status = std::process::Command::new("cmake")
            .args(["--build", "."])
            .current_dir(&plugin_build)
            .status();

        match status {
            Ok(s) if s.success() => {
                // Install plugin .so to XRT lib directory
                let src = plugin_build.join("libxrt_driver_emu.so.2.21.0");
                let dst = xrt_lib.join("libxrt_driver_emu.so.2.21.0");
                let link = xrt_lib.join("libxrt_driver_emu.so.2");
                if src.exists() {
                    if let Err(e) = fs::copy(&src, &dst) {
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
            Ok(s) => {
                println!(
                    "cargo:warning=Plugin cmake build failed (exit {})",
                    s.code().unwrap_or(-1)
                );
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
// Step 0: Comprehensive architecture constants from ArchModel
// ============================================================================

fn gen_arch(model: &xdna_archspec::types::ArchModel, out_dir: &Path) {
    use xdna_archspec::types::TileKind;

    let mut out = gen_header("ArchModel (device model + AM025, cross-validated)");

    // Device-level constants
    if let Some(ref dc) = model.device_constants {
        writeln!(out, "/// Maximum lock counter value.").unwrap();
        writeln!(out, "pub const MAX_LOCK_VALUE: i32 = {};", dc.max_lock_value).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "/// Minimum lock counter value.").unwrap();
        writeln!(out, "pub const MIN_LOCK_VALUE: i32 = {};", dc.min_lock_value).unwrap();
        writeln!(out).unwrap();
    }

    // Array topology
    if let Some(ref topo) = model.array_topology {
        writeln!(out, "/// Number of columns in the NPU array.").unwrap();
        writeln!(out, "pub const COLUMNS: u8 = {};", topo.columns).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "/// Number of rows in the NPU array (including shim row).").unwrap();
        writeln!(out, "pub const ROWS: u8 = {};", topo.rows).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "/// Number of memory tile rows (row 1 through row N are mem tiles).").unwrap();
        writeln!(out, "pub const NUM_MEM_TILE_ROWS: u8 = {};", topo.num_mem_tile_rows).unwrap();
        writeln!(out).unwrap();

        // Tile address encoding (AM020 Ch2)
        writeln!(out, "/// Column shift for tile address encoding (bits 31:25).").unwrap();
        writeln!(out, "pub const TILE_COL_SHIFT: u32 = {};", topo.column_shift).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "/// Row shift for tile address encoding (bits 24:20).").unwrap();
        writeln!(out, "pub const TILE_ROW_SHIFT: u32 = {};", topo.row_shift).unwrap();
        writeln!(out).unwrap();

        // Derived from shifts
        let offset_bits = topo.row_shift;
        let row_bits = topo.column_shift - topo.row_shift;
        let col_bits = 32 - topo.column_shift;
        let offset_mask = (1u32 << offset_bits) - 1;
        writeln!(out, "/// Offset mask for tile-local addresses (bits {}:0).", offset_bits - 1).unwrap();
        writeln!(out, "pub const TILE_OFFSET_MASK: u32 = 0x{:X};", offset_mask).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "/// Row bits in tile address.").unwrap();
        writeln!(out, "pub const TILE_ROW_BITS: u32 = {};", row_bits).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "/// Column bits in tile address.").unwrap();
        writeln!(out, "pub const TILE_COL_BITS: u32 = {};", col_bits).unwrap();
        writeln!(out).unwrap();

        // Row indices (derived from topology)
        writeln!(out, "/// Shim tile row index.").unwrap();
        writeln!(out, "pub const SHIM_ROW: u8 = 0;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "/// First compute tile row index.").unwrap();
        writeln!(out, "pub const COMPUTE_ROW_START: u8 = {} + 1;", topo.num_mem_tile_rows).unwrap();
        writeln!(out).unwrap();
    }

    // Per-tile-type modules
    let tile_types: &[(&str, TileKind)] = &[
        ("compute", TileKind::Compute),
        ("memtile", TileKind::Mem),
        ("shim", TileKind::ShimNoc),
    ];

    for &(mod_name, kind) in tile_types {
        let tile = model.tile_types.iter().find(|t| t.kind == kind);
        let tile = match tile {
            Some(t) => t,
            None => continue,
        };

        writeln!(out, "/// {} tile constants.", tile.name).unwrap();
        writeln!(out, "pub mod {} {{", mod_name).unwrap();

        // Memory model
        if let Some(ref mem) = tile.memory {
            writeln!(out, "    /// Total data memory size in bytes.").unwrap();
            writeln!(out, "    pub const MEMORY_SIZE: u64 = {};", mem.size_bytes).unwrap();
            writeln!(out, "    /// Logical bank count (programmer/compiler view).").unwrap();
            writeln!(out, "    pub const LOGICAL_BANKS: u8 = {};", mem.logical.num_banks).unwrap();
            writeln!(out, "    /// Logical bank size in bytes.").unwrap();
            writeln!(out, "    pub const LOGICAL_BANK_SIZE: u64 = {};", mem.logical.bank_size).unwrap();
            if let Some(ref phys) = mem.physical {
                writeln!(out, "    /// Physical bank count (SRAM arrays, for conflict detection).").unwrap();
                writeln!(out, "    pub const PHYSICAL_BANKS: u8 = {};", phys.num_banks).unwrap();
                writeln!(out, "    /// Physical bank size in bytes.").unwrap();
                writeln!(out, "    pub const PHYSICAL_BANK_SIZE: u64 = {};", phys.bank_size).unwrap();
            }
            if let Some(pmem) = mem.program_memory_bytes {
                writeln!(out, "    /// Program (instruction) memory size in bytes.").unwrap();
                writeln!(out, "    pub const PROGRAM_MEMORY_SIZE: u64 = {};", pmem).unwrap();
            }
        }

        // Resource counts (from Confirmed<T> -- cross-validated)
        let inst = &tile.instances;
        writeln!(out).unwrap();
        writeln!(out, "    /// Number of locks (cross-validated).").unwrap();
        writeln!(out, "    pub const NUM_LOCKS: u8 = {};", inst.locks.value()).unwrap();
        writeln!(out, "    /// Number of buffer descriptors (cross-validated).").unwrap();
        writeln!(out, "    pub const NUM_BDS: u8 = {};", inst.bds.value()).unwrap();
        writeln!(out, "    /// Number of DMA channels per direction (cross-validated).").unwrap();
        writeln!(out, "    pub const NUM_DMA_CHANNELS: u8 = {};", inst.channels.value()).unwrap();

        // Core address map (compute tiles only)
        if let Some(ref cam) = tile.core_address_map {
            writeln!(out).unwrap();
            writeln!(out, "    /// Start of data memory in core address space.").unwrap();
            writeln!(out, "    /// Cardinal direction = address / MEMORY_SIZE.").unwrap();
            writeln!(out, "    /// Source: aie-rt AieMlCoreMod.DataMemAddr.").unwrap();
            writeln!(out, "    pub const DATA_MEM_ADDR: u32 = 0x{:X};", cam.data_mem_addr).unwrap();
            writeln!(out, "    /// Log2 of data memory size per quadrant (address shift).").unwrap();
            writeln!(out, "    /// Source: aie-rt AieMlCoreMod.DataMemShift.").unwrap();
            writeln!(out, "    pub const DATA_MEM_SHIFT: u8 = {};", cam.data_mem_shift).unwrap();
            writeln!(out, "    /// Whether memory modules alternate sides by row.").unwrap();
            writeln!(out, "    /// AIE1=true (checkerboard), AIE2=false (East always local).").unwrap();
            writeln!(out, "    pub const IS_CHECKERBOARD: bool = {};", cam.is_checkerboard).unwrap();
            writeln!(out, "    /// Program memory offset in host/CDO address space.").unwrap();
            writeln!(out, "    /// Source: aie-rt XAIEMLGBL_CORE_MODULE_PROGRAM_MEMORY.").unwrap();
            writeln!(out, "    pub const PROGRAM_MEM_HOST_OFFSET: u32 = 0x{:X};", cam.program_mem_host_offset).unwrap();
        }

        writeln!(out, "}}\n").unwrap();
    }

    // Cardinal direction constants for core data address space.
    // These are the quotient of (address / DataMemSize) as used by
    // aie-rt's _XAie_GetTargetTileLoc() for ELF loading and memory routing.
    writeln!(out, "/// Cardinal direction constants for core data memory addressing.").unwrap();
    writeln!(out, "///").unwrap();
    writeln!(out, "/// The core addresses data memory via `address / MEMORY_SIZE`,").unwrap();
    writeln!(out, "/// yielding a cardinal direction index 4-7.").unwrap();
    writeln!(out, "///").unwrap();
    writeln!(out, "/// Source: aie-rt `_XAie_GetTargetTileLoc()` (xaie_elfloader.c).").unwrap();
    writeln!(out, "pub mod cardinal {{").unwrap();
    writeln!(out, "    /// South neighbor (row - 1).").unwrap();
    writeln!(out, "    pub const SOUTH: u8 = 4;").unwrap();
    writeln!(out, "    /// West neighbor (col - 1 for non-checkerboard, or row-dependent).").unwrap();
    writeln!(out, "    pub const WEST: u8 = 5;").unwrap();
    writeln!(out, "    /// North neighbor (row + 1).").unwrap();
    writeln!(out, "    pub const NORTH: u8 = 6;").unwrap();
    writeln!(out, "    /// East / local (same tile for non-checkerboard, or row-dependent).").unwrap();
    writeln!(out, "    pub const EAST: u8 = 7;").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // Data memory host offset (always 0 for all tile types -- data memory
    // is at the start of the tile register space in the host/CDO view).
    writeln!(out, "/// Data memory offset in host/CDO address space (always 0).").unwrap();
    writeln!(out, "pub const DATA_MEM_HOST_OFFSET: u32 = 0;").unwrap();
    writeln!(out).unwrap();

    // -- Timing model --
    if let Some(ref t) = model.timing {
        writeln!(out, "/// Timing constants (AM020 prose + hardware observation).").unwrap();
        writeln!(out, "pub mod timing {{").unwrap();

        writeln!(out, "    // Lock timing").unwrap();
        writeln!(out, "    pub const LOCK_ACQUIRE_LATENCY: u8 = {};", t.lock.acquire_latency).unwrap();
        writeln!(out, "    pub const LOCK_RELEASE_LATENCY: u8 = {};", t.lock.release_latency).unwrap();
        writeln!(out, "    pub const LOCK_RETRY_INTERVAL: u8 = {};", t.lock.retry_interval).unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // Instruction timing").unwrap();
        writeln!(out, "    /// Data memory access pipeline depth.").unwrap();
        writeln!(out, "    pub const DATA_MEMORY_LATENCY: u8 = {};", t.instruction.data_memory_latency).unwrap();
        writeln!(out, "    /// Branch penalty: cycles lost on taken branch.").unwrap();
        writeln!(out, "    pub const BRANCH_PENALTY: u8 = {};", t.instruction.branch_penalty).unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // DMA timing").unwrap();
        writeln!(out, "    pub const DMA_BD_SETUP_CYCLES: u8 = {};", t.dma.bd_setup_cycles).unwrap();
        writeln!(out, "    pub const DMA_CHANNEL_START_CYCLES: u8 = {};", t.dma.channel_start_cycles).unwrap();
        writeln!(out, "    pub const DMA_WORDS_PER_CYCLE: u8 = {};", t.dma.words_per_cycle).unwrap();
        writeln!(out, "    pub const DMA_MEMORY_LATENCY_CYCLES: u8 = {};", t.dma.memory_latency_cycles).unwrap();
        writeln!(out, "    pub const DMA_LOCK_ACQUIRE_CYCLES: u8 = {};", t.dma.lock_acquire_cycles).unwrap();
        writeln!(out, "    pub const DMA_LOCK_RELEASE_CYCLES: u8 = {};", t.dma.lock_release_cycles).unwrap();
        writeln!(out, "    pub const DMA_BD_CHAIN_CYCLES: u8 = {};", t.dma.bd_chain_cycles).unwrap();
        writeln!(out, "    pub const DMA_HOST_MEMORY_LATENCY_CYCLES: u16 = {};", t.dma.host_memory_latency_cycles).unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // Stream switch timing").unwrap();
        writeln!(out, "    pub const STREAM_LOCAL_SLAVE_FIFO_DEPTH: u8 = {};", t.stream_switch.local_slave_fifo_depth).unwrap();
        writeln!(out, "    pub const STREAM_LOCAL_MASTER_FIFO_DEPTH: u8 = {};", t.stream_switch.local_master_fifo_depth).unwrap();
        writeln!(out, "    pub const STREAM_LOCAL_TO_LOCAL_LATENCY: u8 = {};", t.stream_switch.local_to_local_latency).unwrap();
        writeln!(out, "    pub const STREAM_LOCAL_TO_EXTERNAL_LATENCY: u8 = {};", t.stream_switch.local_to_external_latency).unwrap();
        writeln!(out, "    pub const STREAM_EXTERNAL_TO_EXTERNAL_LATENCY: u8 = {};", t.stream_switch.external_to_external_latency).unwrap();
        writeln!(out, "    pub const STREAM_EXTERNAL_TO_LOCAL_LATENCY: u8 = {};", t.stream_switch.external_to_local_latency).unwrap();
        writeln!(out, "    pub const PACKET_ARBITRATION_OVERHEAD: u8 = {};", t.stream_switch.packet_arbitration_overhead).unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // Route latency (derived from stream switch timing)").unwrap();
        writeln!(out, "    pub const ROUTE_LOCAL_TO_LOCAL: u8 = STREAM_LOCAL_TO_LOCAL_LATENCY;").unwrap();
        writeln!(out, "    pub const ROUTE_LOCAL_TO_EXTERNAL: u8 = STREAM_LOCAL_TO_EXTERNAL_LATENCY;").unwrap();
        writeln!(out, "    pub const ROUTE_EXTERNAL_TO_LOCAL: u8 = STREAM_EXTERNAL_TO_LOCAL_LATENCY;").unwrap();
        writeln!(out, "    pub const ROUTE_EXTERNAL_TO_EXTERNAL: u8 = STREAM_EXTERNAL_TO_EXTERNAL_LATENCY;").unwrap();
        writeln!(out, "    pub const ROUTE_PER_HOP: u8 = ROUTE_EXTERNAL_TO_EXTERNAL;").unwrap();

        writeln!(out, "}}\n").unwrap();
    }

    // -- Packet format --
    if let Some(ref p) = model.packet {
        writeln!(out, "/// Stream packet header bit layout (AM020 Table 2).").unwrap();
        writeln!(out, "pub mod packet {{").unwrap();
        writeln!(out, "    pub const STREAM_ID_MASK: u32 = {:#X};", p.stream.stream_id_mask).unwrap();
        writeln!(out, "    pub const TYPE_SHIFT: u8 = {};", p.stream.packet_type_shift).unwrap();
        writeln!(out, "    pub const TYPE_MASK: u32 = {:#X};", p.stream.packet_type_mask).unwrap();
        writeln!(out, "    pub const SRC_ROW_SHIFT: u8 = {};", p.stream.src_row_shift).unwrap();
        writeln!(out, "    pub const SRC_ROW_MASK: u32 = {:#X};", p.stream.src_row_mask).unwrap();
        writeln!(out, "    pub const SRC_COL_SHIFT: u8 = {};", p.stream.src_col_shift).unwrap();
        writeln!(out, "    pub const SRC_COL_MASK: u32 = {:#X};", p.stream.src_col_mask).unwrap();
        writeln!(out, "    pub const PARITY_SHIFT: u8 = {};", p.stream.parity_shift).unwrap();
        writeln!(out, "}}\n").unwrap();

        writeln!(out, "/// Control packet header format and operation codes (AM020 Table 3).").unwrap();
        writeln!(out, "pub mod ctrl_packet {{").unwrap();
        writeln!(out, "    pub const ADDRESS_MASK: u32 = {:#010X};", p.control.address_mask).unwrap();
        writeln!(out, "    pub const LENGTH_SHIFT: u8 = {};", p.control.length_shift).unwrap();
        writeln!(out, "    pub const LENGTH_MASK: u32 = {:#X};", p.control.length_mask).unwrap();
        writeln!(out, "    pub const OPERATION_SHIFT: u8 = {};", p.control.operation_shift).unwrap();
        writeln!(out, "    pub const OPERATION_MASK: u32 = {:#X};", p.control.operation_mask).unwrap();
        writeln!(out, "    pub const RESPONSE_ID_SHIFT: u8 = {};", p.control.response_id_shift).unwrap();
        writeln!(out, "    pub const RESPONSE_ID_MASK: u32 = {:#X};", p.control.response_id_mask).unwrap();
        writeln!(out, "    pub const PARITY_BIT: u8 = {};", p.control.parity_bit).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    pub const OP_WRITE: u8 = {};", p.control.op_write).unwrap();
        writeln!(out, "    pub const OP_READ: u8 = {};", p.control.op_read).unwrap();
        writeln!(out, "    pub const OP_WRITE_INCR: u8 = {};", p.control.op_write_incr).unwrap();
        writeln!(out, "    pub const OP_BLOCK_WRITE: u8 = {};", p.control.op_block_write).unwrap();
        writeln!(out, "}}\n").unwrap();

        writeln!(out, "/// DMA Finish-on-TLAST mode values (AM025 + aie-rt).").unwrap();
        writeln!(out, "pub mod fot {{").unwrap();
        writeln!(out, "    pub const DISABLED: u8 = {};", p.fot.disabled).unwrap();
        writeln!(out, "    pub const NO_COUNTS: u8 = {};", p.fot.no_counts).unwrap();
        writeln!(out, "    pub const COUNTS_WITH_TOKENS: u8 = {};", p.fot.counts_with_tokens).unwrap();
        writeln!(out, "    pub const COUNTS_FROM_REGISTER: u8 = {};", p.fot.counts_from_register).unwrap();
        writeln!(out, "}}\n").unwrap();
    }

    fs::write(out_dir.join("gen_arch.rs"), out).unwrap();
}

// ============================================================================
// Step 0b: Per-tile-type subsystem address ranges from ArchModel
// ============================================================================

/// Map SubsystemKind to a Rust module name for code generation.
fn subsystem_mod_name(kind: xdna_archspec::types::SubsystemKind) -> &'static str {
    use xdna_archspec::types::SubsystemKind;
    match kind {
        SubsystemKind::Dma => "dma",
        SubsystemKind::Lock => "lock",
        SubsystemKind::LockRequest => "lock_request",
        SubsystemKind::StreamSwitch => "stream_switch",
        SubsystemKind::Processor => "processor",
        SubsystemKind::ProgramMemory => "program_memory",
        SubsystemKind::DataMemory => "data_memory",
        SubsystemKind::Trace => "trace",
        SubsystemKind::Event => "event",
        SubsystemKind::Performance => "performance",
        SubsystemKind::Timer => "timer",
        SubsystemKind::WatchPoint => "watchpoint",
        SubsystemKind::Debug => "debug",
        SubsystemKind::ProgramCounter => "program_counter",
        SubsystemKind::Interrupt => "interrupt",
        SubsystemKind::NoC => "noc",
        SubsystemKind::ShimMux => "shim_mux",
        SubsystemKind::Unknown => "unknown",
    }
}

/// Generate per-tile-type, per-subsystem offset constants.
///
/// Produces `gen_subsystems.rs` with nested modules:
///   `compute::dma::OFFSET_START`, `memtile::lock::OFFSET_END`, etc.
///
/// Each constant is the cross-validated (AM025 + aie-rt) Confirmed<u32> value
/// from the SubsystemModel populated during ArchModel construction.
///
/// When a SubsystemKind appears in multiple modules within the same tile type
/// (e.g., compute tile has performance counters in both Core and Memory modules),
/// the module name is prefixed: `core_performance`, `memory_performance`.
/// Subsystems unique to a single module use the plain name: `dma`, `lock`.
fn gen_subsystems(model: &xdna_archspec::types::ArchModel, out_dir: &Path) {
    use xdna_archspec::types::{ModuleKind, SubsystemKind, TileKind};

    let mut out = gen_header("ArchModel subsystem address ranges");

    let tile_types: &[(&str, &str, TileKind)] = &[
        ("compute", "Compute", TileKind::Compute),
        ("memtile", "MemTile", TileKind::Mem),
        ("shim", "Shim", TileKind::ShimNoc),
    ];

    for &(mod_name, doc_name, kind) in tile_types {
        let tile = match model.tile_types.iter().find(|t| t.kind == kind) {
            Some(t) => t,
            None => continue,
        };

        writeln!(out, "/// {} tile subsystems.", doc_name).unwrap();
        writeln!(out, "pub mod {} {{", mod_name).unwrap();

        // Collect (module_kind, subsystem) pairs, filtering Unknown.
        let mut entries: Vec<(ModuleKind, &xdna_archspec::types::SubsystemModel)> = tile
            .modules
            .iter()
            .flat_map(|m| m.subsystems.iter().map(move |s| (m.kind, s)))
            .filter(|(_, s)| s.kind != SubsystemKind::Unknown)
            .collect();

        // Sort by offset for deterministic output.
        entries.sort_by_key(|(_, s)| *s.offset_start.value());

        // Detect which SubsystemKinds appear in more than one module.
        // Those need module-prefixed names to avoid collisions.
        let mut kind_modules: HashMap<SubsystemKind, Vec<ModuleKind>> = HashMap::new();
        for &(mk, sub) in &entries {
            let entry = kind_modules.entry(sub.kind).or_default();
            if !entry.contains(&mk) {
                entry.push(mk);
            }
        }

        for &(mk, sub) in &entries {
            let base_name = subsystem_mod_name(sub.kind);
            let needs_prefix = kind_modules
                .get(&sub.kind)
                .map_or(false, |mods| mods.len() > 1);

            let full_name = if needs_prefix {
                let prefix = match mk {
                    ModuleKind::Core => "core",
                    ModuleKind::Memory => "memory",
                    ModuleKind::MemTile => "memtile",
                    ModuleKind::Shim => "shim",
                };
                format!("{}_{}", prefix, base_name)
            } else {
                base_name.to_string()
            };

            let start = *sub.offset_start.value();
            let end = *sub.offset_end.value();
            let doc_label = full_name.replace('_', " ");

            writeln!(out, "    pub mod {} {{", full_name).unwrap();
            writeln!(
                out,
                "        /// Start of {} register space (inclusive).",
                doc_label
            )
            .unwrap();
            writeln!(out, "        pub const OFFSET_START: u32 = 0x{:X};", start).unwrap();
            writeln!(
                out,
                "        /// End of {} register space (exclusive).",
                doc_label
            )
            .unwrap();
            writeln!(out, "        pub const OFFSET_END: u32 = 0x{:X};", end).unwrap();
            writeln!(out, "    }}").unwrap();
        }

        writeln!(out, "}}\n").unwrap();
    }

    fs::write(out_dir.join("gen_subsystems.rs"), out).unwrap();
}

// ============================================================================
// Step 1: Core module register offsets
// ============================================================================

fn gen_core_module(regdb: &RegisterDb, out_dir: &Path) {
    let core = regdb
        .module("core")
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
            .register(json_name)
            .unwrap_or_else(|| {
                panic!("Core register '{}' not found in AM025 JSON", json_name)
            });
        writeln!(out, "/// {} (AM025: {})", const_name, json_name).unwrap();
        writeln!(out, "pub const {}: u32 = {:#07X};", const_name, reg.offset).unwrap();
        writeln!(out).unwrap();
    }

    // Compute OFFSET_START / OFFSET_END from all non-stream-switch core registers.
    // The core module address space runs from the lowest core register offset
    // to just below the stream switch region (which starts at 0x3F000).
    let core_proper_offsets: Vec<u32> = core
        .registers
        .iter()
        .filter(|r| !r.name.starts_with("Stream_Switch"))
        .map(|r| r.offset)
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
    regdb: &RegisterDb,
    out_dir: &Path,
    module_name: &str,
    output_file: &str,
) {
    let module = regdb
        .module(module_name)
        .unwrap_or_else(|| panic!("AM025 JSON missing '{}' module", module_name));

    let lock_req = module
        .register("Lock_Request")
        .unwrap_or_else(|| {
            panic!(
                "Lock_Request register not found in '{}' module",
                module_name
            )
        });

    let base_offset = lock_req.offset;
    let desc = lock_req.description.as_deref().unwrap_or("");

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
// Step 3: Stream switch port type arrays
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
