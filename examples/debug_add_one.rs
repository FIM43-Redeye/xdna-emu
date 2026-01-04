//! Debug script to trace add_one_using_dma execution
//!
//! Run with: cargo run --release --example debug_add_one

use xdna_emu::parser::{Xclbin, AiePartition, Cdo, AieElf};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::cdo::find_cdo_offset;
use xdna_emu::interpreter::engine::{InterpreterEngine, EngineStatus};
use xdna_emu::npu::{NpuInstructionStream, NpuExecutor};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let test_dir = Path::new("/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma");
    let xclbin_path = test_dir.join("aie.xclbin");
    let elf_path = test_dir.join("aie_arch.mlir.prj/main_core_0_2.elf");

    println!("=== Debug add_one_using_dma ===\n");

    // Load XCLBIN
    println!("Loading XCLBIN from: {:?}", xclbin_path);
    let xclbin = Xclbin::from_file(&xclbin_path)?;
    println!("  Sections: {}", xclbin.sections().count());

    // Get AIE partition
    let section = xclbin.find_section(SectionKind::AiePartition)
        .ok_or_else(|| anyhow::anyhow!("No AIE partition"))?;
    println!("  AIE Partition: {} bytes", section.size());

    let partition = AiePartition::parse(section.data())?;
    println!("  Column width: {}", partition.column_width());

    // Get PDI and parse CDO
    let pdi = partition.primary_pdi()
        .ok_or_else(|| anyhow::anyhow!("No primary PDI"))?;
    println!("  PDI size: {} bytes", pdi.pdi_image.len());

    let cdo_offset = find_cdo_offset(pdi.pdi_image)
        .ok_or_else(|| anyhow::anyhow!("No CDO in PDI"))?;
    println!("  CDO offset: {} (0x{:X})", cdo_offset, cdo_offset);

    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..])?;
    println!("  CDO commands: {}", cdo.commands().count());

    // Create engine and apply CDO
    println!("\nCreating engine and applying CDO...");
    let mut engine = InterpreterEngine::new_npu1();
    engine.device_mut().apply_cdo(&cdo)?;

    // Print CDO stats
    let stats = &engine.device().stats;
    println!("  CDO stats: {} commands, {} writes, {} dma_writes",
        stats.commands, stats.writes, stats.dma_writes);

    // Setup host memory with input data
    println!("\nSetting up host memory...");
    let host_mem = engine.host_memory_mut();
    let _ = host_mem.allocate_region("input", 0x0, 4096);
    let input: Vec<u32> = (1..=1024).collect();
    host_mem.write_slice(0x0, &input);
    let _ = host_mem.allocate_region("output", 0x1000, 4096);
    println!("  Input: 64 i32 values at 0x0 (first 4: [1, 2, 3, 4])");

    // Load and execute NPU instructions (insts.bin)
    let insts_path = test_dir.join("insts.bin");
    if insts_path.exists() {
        println!("\nLoading NPU instructions from {:?}...", insts_path);
        let insts_data = std::fs::read(&insts_path)?;
        println!("  Size: {} bytes", insts_data.len());

        let stream = NpuInstructionStream::parse(&insts_data)
            .map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;
        println!("  Parsed {} instructions", stream.len());

        let mut npu_executor = NpuExecutor::new();
        // Set up host buffer addresses matching the test
        npu_executor.add_host_buffer(0x0, 4096);      // Input buffer
        npu_executor.add_host_buffer(0x1000, 4096);   // Output buffer

        npu_executor.execute(&stream, engine.device_mut())
            .map_err(|e| anyhow::anyhow!("NPU execution failed: {}", e))?;
        println!("  Executed {} NPU instructions", npu_executor.executed_count());

        // Check shim DMA state
        println!("\nShim DMA (0,0) state after NPU:");
        if let Some(dma) = engine.device().array.dma_engine(0, 0) {
            for ch in 0..4 {
                let state = dma.channel_state(ch);
                println!("  Channel {}: {:?}", ch, state);
            }
            println!("  Stream out queue: {} items", dma.stream_out_len());
        }

        // Check MemTile DMA state - critical for understanding data flow
        println!("\nMemTile DMA (0,1) state after NPU:");
        if let Some(dma) = engine.device().array.dma_engine(0, 1) {
            // MemTile has 6 S2MM (0-5) and 6 MM2S (6-11) channels
            for ch in 0..12 {
                let state = dma.channel_state(ch);
                let kind = if ch < 6 { "S2MM" } else { "MM2S" };
                println!("  Channel {} ({}): {:?}", ch, kind, state);
            }
            println!("  Stream in queue: {} items", dma.stream_in_len());
            println!("  Stream out queue: {} items", dma.stream_out_len());
        }

        // Check compute tile DMA state
        println!("\nCompute DMA (0,2) state:");
        if let Some(dma) = engine.device().array.dma_engine(0, 2) {
            for ch in 0..4 {
                let state = dma.channel_state(ch);
                println!("  Channel {}: {:?}", ch, state);
            }
        }

        // Check lock values on tile (0,2)
        println!("\nLock values on tile (0,2):");
        let tile = engine.device().array.tile(0, 2);
        for lock_id in 0..16 {
            let value = tile.locks[lock_id].value;
            if value != 0 {
                println!("  Lock {}: {}", lock_id, value);
            }
        }

        // Check stream router
        println!("\nStream router stats:");
        let router_stats = engine.device().array.stream_router.stats();
        println!("  Active routes: {}", router_stats.active_routes);
        println!("  Output buffered: {}", router_stats.output_buffered);
        println!("  Input buffered: {}", router_stats.input_buffered);

        // Check MemTile stream switch local routes (configured by CDO)
        println!("\nMemTile (0,1) stream switch local routes:");
        let memtile = engine.device().array.tile(0, 1);
        let local_routes = memtile.stream_switch.local_route_count();
        println!("  Total local routes: {}", local_routes);
        for route in &memtile.stream_switch.local_routes {
            println!("  slave[{}] -> master[{}]", route.slave_idx, route.master_idx);
        }

        // Global routes should be derived automatically from CDO stream switch configuration.
        // When a master port with external type (North/South) is enabled, the emulator
        // derives the inter-tile connection based on physical topology.
        //
        // MemTile port layout (AM025 register reference):
        //   Masters: 0-5 DMA, 6 Tile_Ctrl, 7-10 South, 11-16 North
        //   Slaves: 0-5 DMA, 6 Tile_Ctrl, 7-12 South, 13-16 North
        //
        // The CDO configures local routes within each tile's stream switch.
        // Global routes (inter-tile) are derived when external ports are enabled.

        println!("\nGlobal routes derived from CDO:");
        let router_stats2 = engine.device().array.stream_router.stats();
        println!("  Active routes: {}", router_stats2.active_routes);

        // Print all derived routes
        for route in engine.device().array.stream_router.routes() {
            println!("  ({},{}) master[{}] -> ({},{}) slave[{}]",
                route.src.col, route.src.row, route.src.port,
                route.dest.col, route.dest.row, route.dest.port);
        }
    } else {
        println!("\nNo insts.bin found - skipping NPU instruction execution");
    }

    // Sync core state
    engine.sync_cores_from_device();

    // Load ELF
    println!("\nLoading ELF: {:?}", elf_path);
    if elf_path.exists() {
        let elf_data = std::fs::read(&elf_path)?;
        let elf = AieElf::parse(&elf_data)?;
        println!("  Entry point: 0x{:X}", elf.entry_point());
        println!("  Segments: {}", elf.load_segments().count());

        engine.load_elf_bytes(0, 2, &elf_data).map_err(|e| anyhow::anyhow!("{}", e))?;
        println!("  Loaded into tile (0,2)");
    } else {
        println!("  ELF file not found, relying on CDO-loaded code");
    }

    // Check enabled cores after ELF load
    let enabled = engine.enabled_cores();
    println!("\nEnabled cores: {}", enabled);

    // Run for a few cycles and trace
    println!("\n=== Running (max 1000 cycles) ===");
    let mut last_pc = 0u32;
    let mut stall_count = 0;
    let mut last_stream_count = 16usize;

    for i in 0..1000 {
        engine.step();

        // Get PC of core (0,2)
        if let Some(ctx) = engine.core_context(0, 2) {
            let pc = ctx.pc();
            if pc == last_pc {
                stall_count += 1;
                if stall_count == 10 {
                    let status = engine.core_status(0, 2);
                    println!("Cycle {}: STALLED at PC=0x{:04X} for 10 cycles, status={:?}", i, pc, status);
                }
            } else {
                if stall_count > 0 {
                    stall_count = 0;
                }
                last_pc = pc;
            }
        }

        // Track stream consumption
        let tile = engine.device().array.tile(0, 2);
        let stream_count = tile.stream_input_len(0);
        if stream_count != last_stream_count && stream_count > 0 {
            println!("  Cycle {}: stream_input[0] changed: {} -> {}", i, last_stream_count, stream_count);
            last_stream_count = stream_count;
        } else if stream_count == 0 && last_stream_count > 0 {
            println!("  Cycle {}: stream_input[0] EMPTY (was {})", i, last_stream_count);
            last_stream_count = 0;
        }

        // Check status
        match engine.status() {
            EngineStatus::Halted => {
                println!("\nEngine HALTED at cycle {}", engine.total_cycles());
                break;
            }
            EngineStatus::Error => {
                println!("\nEngine ERROR at cycle {}", engine.total_cycles());

                // Try to get error details
                if let Some(bundle) = engine.core_last_bundle(0, 2) {
                    println!("  Last bundle slot mask: {:?}", bundle.slot_mask());
                }
                break;
            }
            _ => {}
        }

        // Print progress every 100 cycles
        if (i + 1) % 100 == 0 {
            if let Some(ctx) = engine.core_context(0, 2) {
                println!("  Cycle {}: PC=0x{:04X}", i + 1, ctx.pc());
            }
            // DMA diagnostics at cycle 100
            if i + 1 == 100 {
                if let Some(dma) = engine.device().array.dma_engine(0, 0) {
                    println!("    Shim DMA (0,0):");
                    println!("      Channel 0 (S2MM): {:?}", dma.channel_state(0));
                    println!("      Channel 2 (MM2S): {:?}", dma.channel_state(2));
                    println!("      stream_out: {} items", dma.stream_out_len());
                }
                // Check COMPUTE DMA (0,2) state
                if let Some(dma) = engine.device().array.dma_engine(0, 2) {
                    println!("    Compute DMA (0,2):");
                    println!("      Channel 0 (S2MM): {:?}", dma.channel_state(0));
                    println!("      Channel 2 (MM2S): {:?}", dma.channel_state(2));
                    println!("      stream_in: {} items", dma.stream_in_len());
                }
                // Check lock values
                let tile = engine.device().array.tile(0, 2);
                println!("    Tile (0,2) locks: 0={}, 1={}, 2={}, 3={}, 49={}",
                    tile.locks[0].value, tile.locks[1].value,
                    tile.locks[2].value, tile.locks[3].value,
                    tile.locks[49].value);
                // Check tile memory at S2MM destination (0x400) and output buffers (0x440, 0x460)
                let mem = tile.data_memory();
                let input_vals: Vec<u32> = (0..8).map(|i| {
                    let off = 0x400 + i * 4;
                    u32::from_le_bytes([mem[off], mem[off+1], mem[off+2], mem[off+3]])
                }).collect();
                let output_vals: Vec<u32> = (0..8).map(|i| {
                    let off = 0x440 + i * 4;
                    u32::from_le_bytes([mem[off], mem[off+1], mem[off+2], mem[off+3]])
                }).collect();
                println!("    Tile (0,2) input  @ 0x400: {:?}", input_vals);
                println!("    Tile (0,2) output @ 0x440: {:?}", output_vals);
                // Check stream router
                let stats = engine.device().array.stream_router.stats();
                println!("    Router: {} routes, out={}, in={}",
                    stats.active_routes, stats.output_buffered, stats.input_buffered);

                // Check tile (0,2) stream input buffers
                for port in 0..4u8 {
                    let len = tile.stream_input_len(port);
                    if len > 0 {
                        println!("    Tile (0,2) stream_input[{}]: {} items", port, len);
                    }
                }
            }
        }
    }

    // Final state
    println!("\n=== Final State ===");
    println!("Total cycles: {}", engine.total_cycles());
    println!("Status: {:?}", engine.status());

    Ok(())
}
