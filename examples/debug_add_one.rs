//! Debug script to trace add_one_using_dma execution
//!
//! Run with: cargo run --release --example debug_add_one

use xdna_emu::parser::{Xclbin, AiePartition, Cdo, AieElf};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::cdo::find_cdo_offset;
use xdna_emu::interpreter::engine::{InterpreterEngine, EngineStatus};
use std::path::Path;

fn main() -> anyhow::Result<()> {
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

    // Setup host memory with input data
    println!("\nSetting up host memory...");
    let host_mem = engine.host_memory_mut();
    let _ = host_mem.allocate_region("input", 0x0, 4096);
    let input: Vec<u32> = (1..=1024).collect();
    host_mem.write_slice(0x0, &input);
    let _ = host_mem.allocate_region("output", 0x1000, 4096);
    println!("  Input: 64 i32 values at 0x0 (first 4: [1, 2, 3, 4])");

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

    for i in 0..1000 {
        engine.step();

        // Get PC of core (0,2)
        if let Some(ctx) = engine.core_context(0, 2) {
            let pc = ctx.pc();
            if pc == last_pc {
                stall_count += 1;
                if stall_count == 10 {
                    println!("Cycle {}: STALLED at PC=0x{:04X} for 10 cycles", i, pc);
                }
            } else {
                if stall_count > 0 {
                    stall_count = 0;
                }
                last_pc = pc;
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
        }
    }

    // Final state
    println!("\n=== Final State ===");
    println!("Total cycles: {}", engine.total_cycles());
    println!("Status: {:?}", engine.status());

    Ok(())
}
