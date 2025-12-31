//! Trace execution of add_one_using_dma test to understand data flow.

use xdna_emu::parser::{Xclbin, AiePartition, Cdo};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::cdo::find_cdo_offset;
use xdna_emu::interpreter::engine::InterpreterEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Try static_L1_init which has statically initialized data
    let xclbin_path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/static_L1_init/aie.xclbin";

    // Load xclbin
    let xclbin = Xclbin::from_file(xclbin_path)?;
    let section = xclbin.find_section(SectionKind::AiePartition)
        .ok_or("No AIE partition")?;
    let partition = AiePartition::parse(section.data())?;
    let pdi = partition.primary_pdi().ok_or("No PDI")?;
    let cdo_offset = find_cdo_offset(pdi.pdi_image).ok_or("No CDO")?;
    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..])?;

    // Create engine and apply CDO
    let mut engine = InterpreterEngine::new_npu1();
    engine.device_mut().apply_cdo(&cdo)?;

    // Dump shim tile (0, 0) BD configurations
    println!("=== Shim Tile (0, 0) BD Configurations ===");
    if let Some(tile) = engine.device().tile(0, 0) {
        for (i, bd) in tile.dma_bds.iter().enumerate() {
            if bd.length > 0 || bd.addr_low > 0 {
                println!("BD {}: addr=0x{:08X}_{:08X}, len={}, ctrl=0x{:08X}",
                    i, bd.addr_high, bd.addr_low, bd.length, bd.control);
            }
        }

        // Check DMA channel states
        println!("\n=== Shim Tile (0, 0) DMA Channels ===");
        for (i, ch) in tile.dma_channels.iter().enumerate() {
            if ch.running || ch.control != 0 || ch.start_queue != 0 {
                println!("Channel {}: running={}, ctrl=0x{:08X}, start_queue=0x{:02X}",
                    i, ch.running, ch.control, ch.start_queue);
            }
        }
    }

    // Dump compute tile (0, 2) BD configurations
    println!("\n=== Compute Tile (0, 2) BD Configurations ===");
    if let Some(tile) = engine.device().tile(0, 2) {
        for (i, bd) in tile.dma_bds.iter().enumerate() {
            if bd.length > 0 || bd.addr_low > 0 {
                println!("BD {}: addr=0x{:08X}_{:08X}, len={}, ctrl=0x{:08X}",
                    i, bd.addr_high, bd.addr_low, bd.length, bd.control);
            }
        }

        // Check DMA channel states
        println!("\n=== Compute Tile (0, 2) DMA Channels ===");
        for (i, ch) in tile.dma_channels.iter().enumerate() {
            if ch.running || ch.control != 0 || ch.start_queue != 0 {
                println!("Channel {}: running={}, ctrl=0x{:08X}, start_queue=0x{:02X}",
                    i, ch.running, ch.control, ch.start_queue);
            }
        }

        // Check locks
        println!("\n=== Compute Tile (0, 2) Locks ===");
        for (i, lock) in tile.locks.iter().enumerate() {
            if lock.value > 0 {
                println!("Lock {}: value={}", i, lock.value);
            }
        }

        // Check for static data in tile memory
        println!("\n=== Compute Tile (0, 2) Data Memory (first 64 bytes) ===");
        let dm = tile.data_memory();
        let mut has_data = false;
        for i in 0..16 {
            let offset = i * 4;
            let val = u32::from_le_bytes([dm[offset], dm[offset+1], dm[offset+2], dm[offset+3]]);
            if val != 0 {
                has_data = true;
                println!("  DM[0x{:04X}] = {}", offset, val);
            }
        }
        if !has_data {
            println!("  (all zeros)");
        }
    }

    // Load ELF files
    let prj_dir = std::path::Path::new(xclbin_path).parent().unwrap().join("aie_arch.mlir.prj");
    if prj_dir.exists() {
        for entry in std::fs::read_dir(&prj_dir)? {
            let path = entry?.path();
            if path.extension().map(|e| e == "elf").unwrap_or(false) {
                if let Some(name) = path.file_name() {
                    let name_str = name.to_string_lossy();
                    if name_str.contains("core_") {
                        // Parse col and row from filename
                        if let Some((col, row)) = parse_core_coords(&name_str) {
                            let data = std::fs::read(&path)?;
                            match engine.load_elf_bytes(col as usize, row as usize, &data) {
                                Ok(entry) => {
                                    println!("\nLoaded ELF for ({}, {}): {:?}, entry=0x{:X}",
                                        col, row, path.file_name(), entry);
                                    // Check if core is enabled after load
                                    if let Some(ctx) = engine.core_context(col as usize, row as usize) {
                                        println!("  Core ({},{}) PC=0x{:X}", col, row, ctx.pc());
                                    }
                                }
                                Err(e) => {
                                    println!("\nFailed to load ELF for ({}, {}): {}", col, row, e);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Check if cores are enabled in engine state (set by load_elf_bytes)
    // NOTE: Do NOT call sync_cores_from_device() after loading ELFs - it would
    // overwrite the enabled state with the (un-enabled) device state.
    println!("\n=== Engine Core States (After ELF Load) ===");
    println!("is_core_enabled(0,2): {}", engine.is_core_enabled(0, 2));
    println!("is_core_enabled(0,3): {}", engine.is_core_enabled(0, 3));
    println!("enabled_cores(): {}", engine.enabled_cores());

    // Check data memory after ELF load
    println!("\n=== Tile (0, 2) Data Memory After ELF Load ===");
    if let Some(tile) = engine.device().tile(0, 2) {
        let dm = tile.data_memory();
        let mut found_data = false;
        // Check multiple regions - static data might be at various offsets
        for region_start in [0, 0x100, 0x200, 0x400, 0x800, 0x1000].iter() {
            let mut region_has_data = false;
            for i in 0..8 {
                let offset = region_start + i * 4;
                if offset + 4 > dm.len() { break; }
                let val = u32::from_le_bytes([dm[offset], dm[offset+1], dm[offset+2], dm[offset+3]]);
                if val != 0 {
                    if !region_has_data {
                        println!("  Region 0x{:04X}:", region_start);
                        region_has_data = true;
                        found_data = true;
                    }
                    println!("    DM[0x{:04X}] = {} (0x{:08X})", offset, val, val);
                }
            }
        }
        if !found_data {
            println!("  (no non-zero data found in sampled regions)");
        }
    }

    // Step to see where it stalls
    println!("\n=== Stepping Execution ===");
    let mut last_pc = 0u32;
    let mut stall_count = 0;

    for step in 0..1000 {
        engine.step();

        // Get PC of compute core
        let pc = engine.core_context(0, 2).map(|ctx| ctx.pc()).unwrap_or(0);

        // Detect stall (same PC for multiple cycles)
        if pc == last_pc {
            stall_count += 1;
            if stall_count == 5 {
                println!("Step {}: Core STALLED at PC=0x{:04X}", step, pc);

                // Check lock states when stalled
                if let Some(tile) = engine.device().tile(0, 2) {
                    println!("  Lock states:");
                    for (i, lock) in tile.locks.iter().enumerate().take(16) {
                        if lock.value != 0 {
                            println!("    Lock {}: value={}", i, lock.value);
                        }
                    }
                }
                break;
            }
        } else {
            stall_count = 0;
            last_pc = pc;
        }

        if step < 20 || step % 100 == 0 {
            println!("Step {}: PC(0,2)=0x{:04X}", step, pc);
        }

        if matches!(engine.status(), xdna_emu::interpreter::engine::EngineStatus::Halted) {
            println!("Engine halted at step {}", step);
            break;
        }
    }

    Ok(())
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
