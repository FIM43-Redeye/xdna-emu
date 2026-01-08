//! Run tests from TOML manifests extracted from mlir-aie.
//!
//! This example demonstrates the end-to-end test flow:
//! 1. Load a test manifest
//! 2. Generate input data
//! 3. Load the xclbin
//! 4. Run on the emulator
//! 5. Verify output matches expected
//!
//! # Usage
//!
//! ```bash
//! # Run a specific manifest
//! cargo run --release --example manifest_test -- \
//!     tests/mlir-aie-extracted/manifests/add_one_using_dma.toml \
//!     --mlir-aie /home/triple/npu-work/mlir-aie
//!
//! # Run all extracted manifests
//! cargo run --release --example manifest_test -- \
//!     tests/mlir-aie-extracted/manifests/*.toml \
//!     --mlir-aie /home/triple/npu-work/mlir-aie
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use xdna_emu::interpreter::engine::{InterpreterEngine, EngineStatus};
use xdna_emu::npu::{NpuInstructionStream, NpuExecutor};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::{AiePartition, Cdo, Xclbin};
use xdna_emu::parser::cdo::find_cdo_offset;
use xdna_emu::testing::manifest_runner::{
    ElementType, ManifestRunner, TestManifest, read_values,
};

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn"),
    )
    .init();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.toml>... [--mlir-aie <path>]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --mlir-aie <path>  Path to mlir-aie repository (for finding xclbin files)");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} tests/mlir-aie-extracted/manifests/add_one_using_dma.toml \\", args[0]);
        eprintln!("      --mlir-aie /home/triple/npu-work/mlir-aie");
        std::process::exit(1);
    }

    // Parse arguments
    let mut manifest_paths: Vec<PathBuf> = Vec::new();
    let mut mlir_aie_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--mlir-aie" && i + 1 < args.len() {
            mlir_aie_path = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else if !args[i].starts_with('-') {
            manifest_paths.push(PathBuf::from(&args[i]));
            i += 1;
        } else {
            eprintln!("Unknown option: {}", args[i]);
            std::process::exit(1);
        }
    }

    // Create runner
    let mut runner = ManifestRunner::new();
    if let Some(path) = mlir_aie_path.clone() {
        runner = runner.with_mlir_aie_path(path);
    }

    // Run all manifests
    let mut total = 0;
    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for manifest_path in &manifest_paths {
        total += 1;

        // Load manifest
        let manifest = match TestManifest::from_file(manifest_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[SKIP] {}: {}", manifest_path.display(), e);
                skipped += 1;
                continue;
            }
        };

        print!("[TEST] {} ... ", manifest.test.name);

        // Find xclbin
        let xclbin_path = match runner.find_xclbin(&manifest) {
            Some(p) => p,
            None => {
                println!("SKIP (no xclbin found)");
                skipped += 1;
                continue;
            }
        };

        // Find insts.bin (NPU instructions)
        let insts_path = xclbin_path.parent().unwrap().join("insts.bin");

        // Run the test
        match run_test(&manifest, &xclbin_path, &insts_path, mlir_aie_path.as_ref()) {
            Ok(result) => {
                if result.passed {
                    println!(
                        "PASS ({}/{} correct)",
                        result.correct_count, result.total_count
                    );
                    passed += 1;
                } else {
                    println!(
                        "FAIL ({}/{} correct)",
                        result.correct_count, result.total_count
                    );
                    if let Some(mismatch) = &result.first_mismatch {
                        println!(
                            "  First mismatch at index {}: expected {}, got {}",
                            mismatch.index, mismatch.expected, mismatch.actual
                        );
                    }
                    if let Some(error) = &result.error {
                        println!("  Error: {}", error);
                    }
                    failed += 1;
                }
            }
            Err(e) => {
                println!("ERROR: {}", e);
                failed += 1;
            }
        }
    }

    // Summary
    println!();
    println!("=== Summary ===");
    println!("Total: {}", total);
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    println!("Skipped: {}", skipped);

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Run a single test and return the result.
fn run_test(
    manifest: &TestManifest,
    xclbin_path: &Path,
    insts_path: &Path,
    mlir_aie_path: Option<&PathBuf>,
) -> Result<TestResult, String> {
    // Create the interpreter engine
    let mut engine = InterpreterEngine::new_npu1();

    // Load XCLBIN and apply CDO
    let xclbin = Xclbin::from_file(xclbin_path)
        .map_err(|e| format!("Failed to load xclbin: {}", e))?;

    let section = xclbin.find_section(SectionKind::AiePartition)
        .ok_or("No AIE partition in xclbin")?;

    let partition = AiePartition::parse(section.data())
        .map_err(|e| format!("Failed to parse AIE partition: {}", e))?;

    // Parse and apply CDO
    let pdi = partition.primary_pdi()
        .ok_or("No primary PDI in partition")?;

    let cdo_offset = find_cdo_offset(pdi.pdi_image)
        .ok_or("No CDO in PDI")?;

    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..])
        .map_err(|e| format!("Failed to parse CDO: {}", e))?;

    engine.device_mut().apply_cdo(&cdo)
        .map_err(|e| format!("Failed to apply CDO: {}", e))?;

    // Set up host memory
    let input_base = 0x0u64;
    let unused_base = 0x100u64;
    let output_base = 0x1000u64;

    // Generate input data
    let input_data = manifest
        .generate_input("input_a")
        .ok_or("Failed to generate input data")?;

    let input_buf = manifest.get_input("input_a")
        .ok_or("No input_a buffer defined")?;
    let elem_type = ElementType::from_str(&input_buf.element_type)
        .ok_or("Invalid input element type")?;

    // Write input to host memory
    let host_mem = engine.host_memory_mut();
    let _ = host_mem.allocate_region("input", input_base, input_data.len());
    host_mem.write_slice(input_base, &input_data);

    // Allocate unused middle buffer (some tests have this)
    let _ = host_mem.allocate_region("unused", unused_base, 128);

    // Allocate output buffer
    let output_buf = manifest.get_output()
        .ok_or("No output buffer defined")?;
    let output_elem_type = ElementType::from_str(&output_buf.element_type)
        .ok_or("Invalid output element type")?;
    let output_size = output_buf.size * output_elem_type.byte_size();
    let _ = host_mem.allocate_region("output", output_base, output_size);

    // Execute NPU instructions if present
    if insts_path.exists() {
        let insts_data = std::fs::read(insts_path)
            .map_err(|e| format!("Failed to read insts.bin: {}", e))?;

        let stream = NpuInstructionStream::parse(&insts_data)
            .map_err(|e| format!("Failed to parse NPU instructions: {}", e))?;

        let mut npu_executor = NpuExecutor::new();
        // Set up host buffer addresses matching the kernel signature
        npu_executor.add_host_buffer(input_base, input_data.len());
        npu_executor.add_host_buffer(unused_base, 128);
        npu_executor.add_host_buffer(output_base, output_size);

        npu_executor.execute(&stream, engine.device_mut())
            .map_err(|e| format!("NPU execution failed: {}", e))?;
    }

    // Load ELF if we can find it
    if let Some(mlir_aie) = mlir_aie_path {
        let elf_path = mlir_aie
            .join("build")
            .join(&manifest.test.source_dir)
            .join("aie_arch.mlir.prj")
            .join("main_core_0_2.elf");

        if elf_path.exists() {
            let elf_data = std::fs::read(&elf_path)
                .map_err(|e| format!("Failed to read ELF: {}", e))?;
            engine.load_elf_bytes(0, 2, &elf_data)
                .map_err(|e| format!("Failed to load ELF: {}", e))?;
        }
    }

    // Sync core state and run
    engine.sync_cores_from_device();

    // Run execution
    let max_cycles = 10_000;
    for _ in 0..max_cycles {
        engine.step();

        match engine.status() {
            EngineStatus::Halted => break,
            EngineStatus::Error => {
                return Ok(TestResult {
                    name: manifest.test.name.clone(),
                    passed: false,
                    correct_count: 0,
                    total_count: output_buf.size,
                    first_mismatch: None,
                    error: Some("Engine error during execution".to_string()),
                });
            }
            _ => {}
        }
    }

    // Read output
    let host_mem = engine.host_memory_mut();
    let output_data: Vec<u32> = host_mem.read_slice(output_base, output_buf.size);

    // Convert to i64 for comparison
    let actual_values: Vec<i64> = output_data.iter().map(|&v| v as i64).collect();

    // Generate expected values
    let input_values = read_values(&input_data, elem_type);
    let mut inputs: HashMap<String, Vec<i64>> = HashMap::new();
    inputs.insert("input_a".to_string(), input_values);

    let expected_values = manifest.generate_expected(&inputs)
        .ok_or("Failed to generate expected values")?;

    // Compare
    let total_count = expected_values.len().min(actual_values.len());
    let mut correct_count = 0;
    let mut first_mismatch = None;

    for i in 0..total_count {
        if actual_values[i] == expected_values[i] {
            correct_count += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some(MismatchInfo {
                index: i,
                expected: expected_values[i],
                actual: actual_values[i],
            });
        }
    }

    Ok(TestResult {
        name: manifest.test.name.clone(),
        passed: correct_count == total_count && total_count > 0,
        correct_count,
        total_count,
        first_mismatch,
        error: None,
    })
}

/// Result of running a test.
struct TestResult {
    name: String,
    passed: bool,
    correct_count: usize,
    total_count: usize,
    first_mismatch: Option<MismatchInfo>,
    error: Option<String>,
}

/// Information about a mismatched output value.
struct MismatchInfo {
    index: usize,
    expected: i64,
    actual: i64,
}
