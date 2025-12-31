// Quick diagnostic script - run with: cargo run --example debug_test

use xdna_emu::parser::{Xclbin, AiePartition, Cdo};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::cdo::find_cdo_offset;
use xdna_emu::interpreter::engine::{InterpreterEngine, EngineStatus};
use xdna_emu::interpreter::bundle::{detect_format, extract_slots};

fn main() {
    let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie.xclbin";
    let elf_path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie_arch.mlir.prj/main_core_0_2.elf";

    println!("Loading: {}", path);

    let xclbin = Xclbin::from_file(path).unwrap();
    let section = xclbin.find_section(SectionKind::AiePartition).unwrap();
    let partition = AiePartition::parse(section.data()).unwrap();
    let pdi = partition.primary_pdi().unwrap();
    let cdo_offset = find_cdo_offset(pdi.pdi_image).unwrap();
    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).unwrap();

    println!("CDO: {} commands", cdo.commands().count());

    let mut engine = InterpreterEngine::new_npu1();
    engine.device_mut().apply_cdo(&cdo).unwrap();
    engine.sync_cores_from_device();

    // Load ELF for core (0, 2)
    let elf_data = std::fs::read(elf_path).unwrap();
    let entry = engine.load_elf_bytes(0, 2, &elf_data).unwrap();
    println!("Loaded ELF, entry point: 0x{:04X}", entry);

    println!("Enabled cores: {}", engine.enabled_cores());

    // Print core status
    for col in 0..4 {
        for row in 2..6 {
            if let Some(ctx) = engine.core_context(col, row) {
                if engine.is_core_enabled(col, row) {
                    println!("Core ({},{}) enabled, PC=0x{:08X}", col, row, ctx.pc());
                }
            }
        }
    }

    // Run for 500 cycles and check status
    for i in 0..500 {
        engine.step();

        if engine.status() == EngineStatus::Halted {
            println!("Halted after {} cycles", i + 1);
            return;
        }

        if engine.status() == EngineStatus::Error {
            println!("Error after {} cycles", i + 1);

            // Print last bundle for each core
            for col in 0..4 {
                for row in 2..6 {
                    if engine.is_core_enabled(col, row) {
                        if let Some(ctx) = engine.core_context(col, row) {
                            let pc = ctx.pc();
                            println!("Core ({},{}) PC=0x{:04X}", col, row, pc);

                            // Dump program memory at PC
                            if let Some(tile) = engine.device().tile(col, row) {
                                if let Some(pm) = tile.program_memory() {
                                    let pc_byte = pc as usize;
                                    if pc_byte + 16 <= pm.len() {
                                        print!("  PM[0x{:04X}]: ", pc);
                                        for b in &pm[pc_byte..pc_byte+16] {
                                            print!("{:02X} ", b);
                                        }
                                        println!();

                                        // Debug: test slot extraction manually
                                        let test_bytes = &pm[pc_byte..pc_byte+16];
                                        let format = detect_format(test_bytes);
                                        println!("  Manual format detection: {:?} ({} bytes)", format, format.size_bytes());

                                        let word = u32::from_le_bytes([
                                            test_bytes[0], test_bytes[1],
                                            test_bytes[2], test_bytes[3]
                                        ]);
                                        println!("  Word: 0x{:08X}", word);
                                        println!("  Low nibble: 0x{:X}", word & 0xF);
                                        println!("  High5: 0b{:05b}", (word >> 27) & 0x1F);

                                        let extracted = extract_slots(test_bytes);
                                        println!("  Extracted {} slots:", extracted.slots.len());
                                        for slot in &extracted.slots {
                                            println!("    {:?}: bits=0x{:X}, width={}", slot.slot_type, slot.bits, slot.width);
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(bundle) = engine.core_last_bundle(col, row) {
                            println!("  Last bundle (format={:?}, size={}):",
                                bundle.format(), bundle.size());
                            print!("  Raw: ");
                            for b in bundle.raw_bytes() {
                                print!("{:02X} ", b);
                            }
                            println!();
                            for op in bundle.active_slots() {
                                println!("    {:?}: {:?}", op.slot, op.op);
                            }
                        }
                    }
                }
            }
            return;
        }

        // Print status every 100 cycles
        if (i + 1) % 100 == 0 {
            print!("Cycle {}: ", i + 1);
            for col in 0..4 {
                for row in 2..6 {
                    if let Some(status) = engine.core_status(col, row) {
                        if engine.is_core_enabled(col, row) {
                            if let Some(ctx) = engine.core_context(col, row) {
                                print!("({},{})={:?} PC=0x{:04X} ", col, row, status, ctx.pc());
                            }
                        }
                    }
                }
            }
            println!();
        }
    }

    println!("Ran 500 cycles, still running");

    // Print final status
    for col in 0..4 {
        for row in 2..6 {
            if let Some(status) = engine.core_status(col, row) {
                if engine.is_core_enabled(col, row) {
                    if let Some(ctx) = engine.core_context(col, row) {
                        println!("Core ({},{}) status={:?} PC=0x{:08X}", col, row, status, ctx.pc());
                    }
                    if let Some(bundle) = engine.core_last_bundle(col, row) {
                        println!("  Last bundle:");
                        for op in bundle.active_slots() {
                            println!("    {:?}: {:?}", op.slot, op.op);
                        }
                    }
                }
            }
        }
    }
}
