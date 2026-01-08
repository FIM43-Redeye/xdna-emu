//! Debug harness for any mlir-aie test
//!
//! Run with: cargo run --release --example debug_mlir_aie -- <test_name>
//!
//! Examples:
//!   cargo run --release --example debug_mlir_aie -- add_one_using_dma
//!   cargo run --release --example debug_mlir_aie -- add_256_using_dma_op_no_double_buffering
//!
//! This provides comprehensive debug output for any test in mlir-aie/build/test/npu-xrt/

use xdna_emu::parser::{Xclbin, AiePartition, Cdo, AieElf};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::cdo::find_cdo_offset;
use xdna_emu::interpreter::engine::{InterpreterEngine, EngineStatus};
use xdna_emu::npu::{NpuInstructionStream, NpuExecutor};
use std::path::{Path, PathBuf};

const MLIR_AIE_BASE: &str = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt";

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn")
    ).init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <test_name> [--max-cycles N]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} add_one_using_dma", args[0]);
        eprintln!("  {} add_256_using_dma_op_no_double_buffering --max-cycles 500", args[0]);
        eprintln!();
        eprintln!("Available tests:");
        list_tests()?;
        std::process::exit(1);
    }

    let test_name = &args[1];
    let max_cycles: u32 = args.iter()
        .position(|a| a == "--max-cycles")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let test_dir = PathBuf::from(MLIR_AIE_BASE).join(test_name);
    if !test_dir.exists() {
        eprintln!("Test directory not found: {}", test_dir.display());
        eprintln!();
        eprintln!("Available tests:");
        list_tests()?;
        std::process::exit(1);
    }

    run_debug(&test_dir, max_cycles)
}

fn list_tests() -> anyhow::Result<()> {
    for entry in std::fs::read_dir(MLIR_AIE_BASE)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // Check for xclbin
            let has_xclbin = path.join("aie.xclbin").exists()
                || path.join("final.xclbin").exists();
            if has_xclbin {
                if let Some(name) = path.file_name() {
                    println!("  {}", name.to_string_lossy());
                }
            }
        }
    }
    Ok(())
}

fn run_debug(test_dir: &Path, max_cycles: u32) -> anyhow::Result<()> {
    let test_name = test_dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("=== Debug {} ===\n", test_name);

    // Find xclbin
    let xclbin_path = if test_dir.join("aie.xclbin").exists() {
        test_dir.join("aie.xclbin")
    } else if test_dir.join("final.xclbin").exists() {
        test_dir.join("final.xclbin")
    } else {
        anyhow::bail!("No xclbin found in {}", test_dir.display());
    };

    // Find ELF files
    let elf_files = find_elf_files(test_dir);
    println!("Found {} ELF files:", elf_files.len());
    for (col, row, path) in &elf_files {
        println!("  ({},{}) -> {}", col, row, path.file_name().unwrap().to_string_lossy());
    }

    // Load XCLBIN
    println!("\n--- XCLBIN ---");
    println!("Path: {:?}", xclbin_path);
    let xclbin = Xclbin::from_file(&xclbin_path)?;
    println!("Sections: {}", xclbin.sections().count());

    // Get AIE partition
    let section = xclbin.find_section(SectionKind::AiePartition)
        .ok_or_else(|| anyhow::anyhow!("No AIE partition"))?;
    println!("AIE Partition: {} bytes", section.size());

    let partition = AiePartition::parse(section.data())?;
    println!("Column width: {}", partition.column_width());

    // Get PDI and parse CDO
    let pdi = partition.primary_pdi()
        .ok_or_else(|| anyhow::anyhow!("No primary PDI"))?;
    println!("PDI size: {} bytes", pdi.pdi_image.len());

    let cdo_offset = find_cdo_offset(pdi.pdi_image)
        .ok_or_else(|| anyhow::anyhow!("No CDO in PDI"))?;
    println!("CDO offset: {} (0x{:X})", cdo_offset, cdo_offset);

    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..])?;
    println!("CDO commands: {}", cdo.commands().count());

    // Create engine and apply CDO
    println!("\n--- Engine Setup ---");
    let mut engine = InterpreterEngine::new_npu1();
    engine.device_mut().apply_cdo(&cdo)?;

    // Print CDO stats
    let stats = &engine.device().stats;
    println!("CDO stats: {} commands, {} writes, {} dma_writes",
        stats.commands, stats.writes, stats.dma_writes);

    // Setup host memory with input data
    println!("\n--- Host Memory ---");
    let host_mem = engine.host_memory_mut();
    let _ = host_mem.allocate_region("input", 0x0, 4096);  // 1024 * 4 bytes
    let input: Vec<u32> = (1..=1024).collect();
    host_mem.write_slice(0x0, &input);
    let _ = host_mem.allocate_region("unused", 0x100, 256);
    let _ = host_mem.allocate_region("output", 0x1000, 4096);
    println!("Input: 1024 i32 values at 0x0 (first 4: [1, 2, 3, 4])");
    println!("Output: allocated at 0x1000");

    // Load and execute NPU instructions
    let insts_path = test_dir.join("insts.bin");
    if insts_path.exists() {
        println!("\n--- NPU Instructions ---");
        let insts_data = std::fs::read(&insts_path)?;
        println!("Size: {} bytes", insts_data.len());

        let stream = NpuInstructionStream::parse(&insts_data)
            .map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;
        println!("Parsed {} instructions", stream.len());

        let mut npu_executor = NpuExecutor::new();
        npu_executor.add_host_buffer(0x0, 4096);
        npu_executor.add_host_buffer(0x100, 256);
        npu_executor.add_host_buffer(0x1000, 4096);

        npu_executor.execute(&stream, engine.device_mut())
            .map_err(|e| anyhow::anyhow!("NPU execution failed: {}", e))?;
        println!("Executed {} NPU instructions", npu_executor.executed_count());
    } else {
        println!("\nNo insts.bin found");
    }

    // Print DMA state for all active tiles
    print_dma_state(&engine);
    print_lock_state(&engine);
    print_stream_routes(&engine);

    // Load ELF files
    println!("\n--- Loading ELFs ---");
    for (col, row, path) in &elf_files {
        println!("Loading ({},{}): {:?}", col, row, path.file_name().unwrap());
        let elf_data = std::fs::read(path)?;
        let elf = AieElf::parse(&elf_data)?;
        println!("  Entry: 0x{:X}, Segments: {}", elf.entry_point(), elf.load_segments().count());
        engine.load_elf_bytes(*col as usize, *row as usize, &elf_data)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    // Sync core state
    engine.sync_cores_from_device();
    let enabled = engine.enabled_cores();
    println!("\nEnabled cores after sync: {}", enabled);

    if enabled == 0 && elf_files.is_empty() {
        println!("\nWARNING: No cores enabled and no ELF files - test may not run");
    }

    // Run execution
    println!("\n=== Running (max {} cycles) ===", max_cycles);
    let mut last_pc: std::collections::HashMap<(usize, usize), u32> = std::collections::HashMap::new();
    let mut stall_counts: std::collections::HashMap<(usize, usize), u32> = std::collections::HashMap::new();
    let mut last_stream_report = 0u32;

    for cycle in 0..max_cycles {
        // Before step: check shim DMA stream_out queue
        if let Some(dma) = engine.device().array.dma_engine(0, 0) {
            let stream_out = dma.stream_out_len();
            if stream_out > 0 && cycle < 10 {
                println!("  Cycle {}: Shim DMA stream_out has {} items", cycle, stream_out);
            }
        }

        engine.step();

        // After step: check if any stream data moved
        let shim_switch = &engine.device().array.tile(0, 0).stream_switch;
        let memtile_switch = &engine.device().array.tile(0, 1).stream_switch;

        // Check shim slave[5] and master[16] FIFOs
        let shim_s5 = shim_switch.slaves.get(5).map(|s| s.fifo.len()).unwrap_or(0);
        let shim_m16 = shim_switch.masters.get(16).map(|m| m.fifo.len()).unwrap_or(0);

        // Check memtile south slaves (7-12)
        let mt_s11 = memtile_switch.slaves.get(11).map(|s| s.fifo.len()).unwrap_or(0);

        if (shim_s5 > 0 || shim_m16 > 0 || mt_s11 > 0) && cycle - last_stream_report >= 10 {
            println!("  Cycle {}: shim.slave[5]={} shim.master[16]={} memtile.slave[11]={}",
                cycle, shim_s5, shim_m16, mt_s11);
            last_stream_report = cycle;
        }

        // Track PC for all cores
        for (col, row, _) in &elf_files {
            let col = *col as usize;
            let row = *row as usize;
            if let Some(ctx) = engine.core_context(col, row) {
                let pc = ctx.pc();
                let key = (col, row);
                let last = last_pc.entry(key).or_insert(pc);
                let stalls = stall_counts.entry(key).or_insert(0);

                if pc == *last {
                    *stalls += 1;
                    if *stalls == 20 {
                        let status = engine.core_status(col, row);
                        println!("Cycle {}: ({},{}) STALLED at PC=0x{:04X} status={:?}",
                            cycle, col, row, pc, status);
                    }
                } else {
                    *stalls = 0;
                    *last = pc;
                }
            }
        }

        // Check status
        match engine.status() {
            EngineStatus::Halted => {
                println!("\nEngine HALTED at cycle {}", engine.total_cycles());
                break;
            }
            EngineStatus::Error => {
                println!("\nEngine ERROR at cycle {}", engine.total_cycles());
                print_error_details(&engine, &elf_files);
                break;
            }
            _ => {}
        }

        // Progress every 200 cycles
        if (cycle + 1) % 200 == 0 {
            print!("  Cycle {}: ", cycle + 1);
            for (col, row, _) in &elf_files {
                if let Some(ctx) = engine.core_context(*col as usize, *row as usize) {
                    print!("({},{})=0x{:04X} ", col, row, ctx.pc());
                }
            }
            println!();
        }
    }

    // Final state
    println!("\n=== Final State ===");
    println!("Total cycles: {}", engine.total_cycles());
    println!("Status: {:?}", engine.status());

    // Check output
    let host_mem = engine.host_memory_mut();
    let output: Vec<u32> = host_mem.read_slice(0x1000, 64);
    let non_zero: Vec<_> = output.iter().enumerate()
        .filter(|(_, &v)| v != 0)
        .take(10)
        .collect();

    if non_zero.is_empty() {
        println!("\nOutput: all zeros (no data written)");
    } else {
        println!("\nOutput (first non-zero values):");
        for (i, v) in non_zero {
            println!("  [{}] = {} (expected {})", i, v, i + 2);
        }
    }

    // Final DMA state
    print_dma_state(&engine);

    // Final lock state
    print_lock_state(&engine);

    Ok(())
}

fn find_elf_files(test_dir: &Path) -> Vec<(u8, u8, PathBuf)> {
    let mut elfs = Vec::new();

    // Look for project directories
    for pattern in &["aie_arch.mlir.prj", "aie.mlir.prj"] {
        let prj_dir = test_dir.join(pattern);
        if prj_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&prj_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name() {
                        let name_str = name.to_string_lossy();
                        if name_str.ends_with(".elf") && name_str.contains("core_") {
                            if let Some((col, row)) = parse_core_coords(&name_str) {
                                elfs.push((col, row, path));
                            }
                        }
                    }
                }
            }
        }
    }

    elfs
}

fn parse_core_coords(name: &str) -> Option<(u8, u8)> {
    let core_idx = name.find("core_")?;
    let after_core = &name[core_idx + 5..];
    let parts: Vec<&str> = after_core.split('_').take(2).collect();
    if parts.len() >= 2 {
        let col: u8 = parts[0].parse().ok()?;
        let row_str = parts[1].trim_end_matches(".elf");
        let row: u8 = row_str.parse().ok()?;
        return Some((col, row));
    }
    None
}

fn print_dma_state(engine: &InterpreterEngine) {
    println!("\n--- DMA State ---");

    // Shim tile (0,0)
    if let Some(dma) = engine.device().array.dma_engine(0, 0) {
        println!("Shim (0,0):");
        for ch in 0..4 {
            let state = dma.channel_state(ch);
            println!("  Ch{}: {:?}", ch, state);
        }
        println!("  Stream out queue: {} items", dma.stream_out_len());
        println!("  Stream in queue: {} items", dma.stream_in_len());
    }

    // MemTile (0,1)
    if let Some(dma) = engine.device().array.dma_engine(0, 1) {
        println!("MemTile (0,1):");
        for ch in 0..6 {
            let state = dma.channel_state(ch);
            println!("  S2MM[{}]: {:?}", ch, state);
        }
        for ch in 6..12 {
            let state = dma.channel_state(ch);
            println!("  MM2S[{}]: {:?}", ch - 6, state);
        }
        println!("  Stream in queue: {} items", dma.stream_in_len());
        println!("  Stream out queue: {} items", dma.stream_out_len());
    }

    // Compute tiles
    for row in 2..6 {
        if let Some(dma) = engine.device().array.dma_engine(0, row) {
            let mut has_activity = false;
            for ch in 0..4 {
                let state = dma.channel_state(ch);
                if !matches!(state, xdna_emu::device::dma::ChannelState::Idle) {
                    has_activity = true;
                    break;
                }
            }
            let stream_in = dma.stream_in_len();
            let stream_out = dma.stream_out_len();
            if has_activity || stream_in > 0 || stream_out > 0 {
                println!("Compute (0,{}):", row);
                for ch in 0..4 {
                    let state = dma.channel_state(ch);
                    println!("  Ch{}: {:?}", ch, state);
                }
                println!("  Stream in: {} items, Stream out: {} items", stream_in, stream_out);
            }
        }
    }
}

fn print_lock_state(engine: &InterpreterEngine) {
    println!("\n--- Lock State ---");

    for row in 1..6 {
        let tile = engine.device().array.tile(0, row);
        let mut has_non_zero = false;
        for lock_id in 0..16 {
            if tile.locks[lock_id].value != 0 {
                has_non_zero = true;
                break;
            }
        }
        if has_non_zero {
            println!("Tile (0,{}):", row);
            for lock_id in 0..16 {
                let value = tile.locks[lock_id].value;
                if value != 0 {
                    println!("  Lock[{}] = {}", lock_id, value);
                }
            }
        }
    }
}

fn print_stream_routes(engine: &InterpreterEngine) {
    println!("\n--- Stream Routes ---");

    let router_stats = engine.device().array.stream_router.stats();
    println!("Global: {} active routes", router_stats.active_routes);

    // Print local routes for active tiles
    for row in 0..6 {
        let tile = engine.device().array.tile(0, row);
        let count = tile.stream_switch.local_route_count();
        if count > 0 {
            println!("Tile (0,{}): {} local routes", row, count);
            for route in &tile.stream_switch.local_routes {
                if route.enabled {
                    println!("  slave[{}] -> master[{}]", route.slave_idx, route.master_idx);
                }
            }
        }
    }
}

fn print_error_details(engine: &InterpreterEngine, elf_files: &[(u8, u8, PathBuf)]) {
    for (col, row, _) in elf_files {
        let col = *col as usize;
        let row = *row as usize;
        if let Some(bundle) = engine.core_last_bundle(col, row) {
            println!("  ({},{}) last bundle: {:?}", col, row, bundle.slot_mask());
            for op in bundle.active_slots() {
                if let xdna_emu::interpreter::bundle::Operation::Unknown { opcode } = &op.op {
                    println!("    UNKNOWN opcode 0x{:08X} in slot {:?}", opcode, op.slot);
                }
            }
        }
    }
}
