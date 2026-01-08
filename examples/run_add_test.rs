//! Generic add_N test runner
//!
//! Usage: cargo run --release --example run_add_test <test_dir> <add_value>

use xdna_emu::parser::{Xclbin, AiePartition, Cdo, AieElf};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::cdo::find_cdo_offset;
use xdna_emu::interpreter::engine::{InterpreterEngine, EngineStatus};
use xdna_emu::npu::{NpuInstructionStream, NpuExecutor};
use std::path::Path;
use std::env;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn")
    ).init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <test_dir> <add_value>", args[0]);
        eprintln!("Example: {} /path/to/add_314_using_dma_op 314", args[0]);
        std::process::exit(1);
    }

    let test_dir = Path::new(&args[1]);
    let add_value: u32 = args[2].parse()?;

    println!("=== Testing {} (add {}) ===\n", test_dir.file_name().unwrap().to_string_lossy(), add_value);

    // Load XCLBIN
    let xclbin_path = test_dir.join("aie.xclbin");
    println!("Loading: {:?}", xclbin_path);
    let xclbin = Xclbin::from_file(&xclbin_path)?;

    let section = xclbin.find_section(SectionKind::AiePartition)
        .ok_or_else(|| anyhow::anyhow!("No AIE partition"))?;
    let partition = AiePartition::parse(section.data())?;
    let pdi = partition.primary_pdi()
        .ok_or_else(|| anyhow::anyhow!("No primary PDI"))?;
    let cdo_offset = find_cdo_offset(pdi.pdi_image)
        .ok_or_else(|| anyhow::anyhow!("No CDO in PDI"))?;
    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..])?;
    println!("  CDO: {} commands", cdo.commands().count());

    // Create engine
    let mut engine = InterpreterEngine::new_npu1();
    engine.device_mut().apply_cdo(&cdo)?;

    // Setup host memory
    let host_mem = engine.host_memory_mut();
    let _ = host_mem.allocate_region("input", 0x0, 256);
    let input: Vec<u32> = (1..=64).collect();
    host_mem.write_slice(0x0, &input);
    let _ = host_mem.allocate_region("unused", 0x100, 128);
    let _ = host_mem.allocate_region("output", 0x1000, 256);

    // Load NPU instructions
    let insts_path = test_dir.join("insts.bin");
    if insts_path.exists() {
        let insts_data = std::fs::read(&insts_path)?;
        let stream = NpuInstructionStream::parse(&insts_data)
            .map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;

        let mut npu_executor = NpuExecutor::new();
        npu_executor.add_host_buffer(0x0, 256);
        npu_executor.add_host_buffer(0x100, 128);
        npu_executor.add_host_buffer(0x1000, 256);
        npu_executor.execute(&stream, engine.device_mut())
            .map_err(|e| anyhow::anyhow!("NPU execution failed: {}", e))?;
    }

    // Load ELF
    let elf_path = test_dir.join("aie_arch.mlir.prj/main_core_0_2.elf");
    if elf_path.exists() {
        let elf_data = std::fs::read(&elf_path)?;
        engine.load_elf_bytes(0, 2, &elf_data).map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    engine.sync_cores_from_device();

    // Run
    println!("\nRunning (max 2000 cycles)...");
    for _ in 0..2000 {
        engine.step();
        if engine.status() == EngineStatus::Halted {
            break;
        }
    }

    // Check output
    let host_mem = engine.host_memory_mut();
    let output: Vec<u32> = host_mem.read_slice(0x1000, 64);

    println!("\nResults:");
    println!("  Input  (first 8): {:?}", &input[0..8]);
    println!("  Output (first 8): {:?}", &output[0..8]);

    // Verify
    let mut correct = 0;
    let mut incorrect = 0;
    let mut zeros = 0;

    for (i, &out) in output.iter().enumerate() {
        let expected = input[i] + add_value;
        if out == expected {
            correct += 1;
        } else if out == 0 {
            zeros += 1;
        } else {
            incorrect += 1;
            if incorrect <= 5 {
                println!("  MISMATCH[{}]: input={}, expected={}, got={}", i, input[i], expected, out);
            }
        }
    }

    println!("\n{} correct, {} incorrect, {} zeros", correct, incorrect, zeros);

    if correct == 64 {
        println!("SUCCESS!");
    } else if zeros > 0 {
        println!("PARTIAL: Some outputs not computed (zeros)");
    } else {
        println!("FAILED: Incorrect outputs");
    }

    Ok(())
}
