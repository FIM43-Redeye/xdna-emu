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
    // The kernel processes 64 i32 values (8 batches of 8 with ping-pong buffering)
    println!("\nSetting up host memory...");
    let host_mem = engine.host_memory_mut();
    let _ = host_mem.allocate_region("input", 0x0, 256);  // 64 * 4 bytes
    let input: Vec<u32> = (1..=64).collect();
    host_mem.write_slice(0x0, &input);
    let _ = host_mem.allocate_region("unused", 0x100, 128);  // arg1: unused buffer
    let _ = host_mem.allocate_region("output", 0x1000, 256);  // 64 * 4 bytes
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
        // Set up host buffer addresses matching the kernel's runtime_sequence signature:
        //   aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>)
        // The DdrPatch instructions use arg_idx to reference these buffers:
        //   arg0 = input buffer, arg1 = unused, arg2 = output buffer
        npu_executor.add_host_buffer(0x0, 256);       // arg0: Input buffer (64 * 4 bytes)
        npu_executor.add_host_buffer(0x100, 128);     // arg1: Unused middle buffer (32 * 4 bytes)
        npu_executor.add_host_buffer(0x1000, 256);    // arg2: Output buffer (64 * 4 bytes)

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
            println!("  slave[{}] -> master[{}] enabled={}", route.slave_idx, route.master_idx, route.enabled);
        }

        // Also print MemTile lock values
        println!("\nMemTile (0,1) lock values:");
        for lock_id in 0..4 {
            let value = memtile.locks[lock_id].value;
            println!("  Lock {}: {}", lock_id, value);
        }

        // Check Compute tile local routes
        println!("\nCompute (0,2) stream switch local routes:");
        let compute_tile = engine.device().array.tile(0, 2);
        println!("  Total local routes: {}", compute_tile.stream_switch.local_route_count());
        for route in &compute_tile.stream_switch.local_routes {
            println!("  slave[{}] -> master[{}] enabled={}", route.slave_idx, route.master_idx, route.enabled);
        }

        // Check Shim tile local routes
        println!("\nShim (0,0) stream switch local routes:");
        let shim_tile = engine.device().array.tile(0, 0);
        println!("  Total local routes: {}", shim_tile.stream_switch.local_route_count());
        for route in &shim_tile.stream_switch.local_routes {
            println!("  slave[{}] -> master[{}] enabled={}", route.slave_idx, route.master_idx, route.enabled);
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

    // Track lock acquires by PC to see if instructions are re-executed
    let mut lock_acquire_pcs: Vec<(u32, u32)> = Vec::new(); // (cycle, pc)
    let lock_acquire_addrs = [0x60u32, 0x74, 0xe8, 0xf2, 0x166, 0x170, 0x1e4, 0x1ee];
    let mut last_dump_cycle: i32 = -100;

    for i in 0..1000 {
        engine.step();

        // Get PC of core (0,2)
        if let Some(ctx) = engine.core_context(0, 2) {
            let pc = ctx.pc();

            // Track when we hit lock acquire instructions
            if lock_acquire_addrs.contains(&pc) && (lock_acquire_pcs.is_empty() || lock_acquire_pcs.last().unwrap().1 != pc) {
                lock_acquire_pcs.push((i as u32, pc));
                println!("  Cycle {}: HIT lock instruction at PC=0x{:04X}", i, pc);
            }

            // Dump memory at key points: before first store (0x9A), first store (0xA0), last store (0xCA), after stores (0xD4)
            if (pc == 0x9A || pc == 0xA0 || pc == 0xCA || pc == 0xD4) && (i as i32 - last_dump_cycle > 5) {
                last_dump_cycle = i as i32;
                let tile = engine.device().array.tile(0, 2);
                let mem = tile.data_memory();
                println!("  Cycle {}: Memory dump at PC=0x{:04X}:", i, pc);
                // Input buffer A: 0x400-0x41F
                print!("    Input@0x400: ");
                for j in 0..8 {
                    let addr = 0x400 + j * 4;
                    let val = u32::from_le_bytes([mem[addr], mem[addr+1], mem[addr+2], mem[addr+3]]);
                    print!("{} ", val);
                }
                println!();
                // Output buffer A: 0x440-0x45F
                print!("    Output@0x440: ");
                for j in 0..8 {
                    let addr = 0x440 + j * 4;
                    let val = u32::from_le_bytes([mem[addr], mem[addr+1], mem[addr+2], mem[addr+3]]);
                    print!("{} ", val);
                }
                println!();
                // Scalar registers r7-r14 (used for load/add/store)
                print!("    r7-r14: ");
                for r in 7..=14 {
                    print!("{} ", ctx.scalar_read(r));
                }
                println!();
            }

            // Track critical control flow points
            if (pc >= 0x140 && pc <= 0x2b0) || (pc >= 0 && pc <= 0x10) {
                println!("  Cycle {}: CRITICAL PC=0x{:04X} lr={:?}", i, pc, ctx.lr());
            }

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

    // Summary of lock acquire PCs hit
    println!("\n=== Lock Acquire PC Trace ===");
    println!("Total lock instruction hits: {}", lock_acquire_pcs.len());
    for (cycle, pc) in &lock_acquire_pcs {
        let lock_type = match *pc {
            0x60 | 0xe8 | 0x166 | 0x1e4 => "acq #0x31 (lock 1, input consumer)",
            0x74 | 0xf2 | 0x170 | 0x1ee => "acq #0x32 (lock 2, output producer)",
            _ => "unknown",
        };
        println!("  Cycle {}: PC=0x{:04X} - {}", cycle, pc, lock_type);
    }

    // =========================================================================
    // VERIFICATION: Check if the computation actually produced correct results
    // =========================================================================
    println!("\n=== Output Verification ===");

    // Read output from host memory (64 elements - matches kernel's loop)
    let host_mem = engine.host_memory_mut();
    let output: Vec<u32> = host_mem.read_slice(0x1000, 64);

    // Check first 16 values
    let show_count = 16.min(input.len());
    println!("Input  (first {}): {:?}", show_count, &input[..show_count]);
    println!("Output (first {}): {:?}", show_count, &output[..show_count]);

    // Verify: output should be input + 1
    let mut correct = 0;
    let mut incorrect = 0;
    let mut zero_count = 0;

    for i in 0..input.len() {
        let expected = input[i].wrapping_add(1);
        if output[i] == expected {
            correct += 1;
        } else if output[i] == 0 {
            zero_count += 1;
        } else {
            incorrect += 1;
            if incorrect <= 5 {
                println!("  MISMATCH at [{}]: input={}, expected={}, got={}",
                    i, input[i], expected, output[i]);
            }
        }
    }

    println!("\nResults: {} correct, {} incorrect, {} zeros (untouched)",
        correct, incorrect, zero_count);

    if correct == input.len() {
        println!("SUCCESS: All {} outputs are correct (input + 1)!", correct);
    } else if zero_count == input.len() {
        println!("FAILURE: Output buffer is all zeros - DMA didn't write results to host memory");
    } else {
        println!("PARTIAL: Some computation happened but results are incomplete");
    }

    Ok(())
}
