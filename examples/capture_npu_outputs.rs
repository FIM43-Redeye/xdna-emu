//! Capture NPU hardware outputs for all discoverable tests.
//!
//! For each test with a test.cpp and corresponding built xclbin + insts.bin:
//! 1. Parses test.cpp for buffer metadata (sizes, types, patterns)
//! 2. Invokes the npu-runner C++ tool to run on real hardware
//! 3. Saves the raw output to tests/npu-outputs/<test_name>/output.bin
//!
//! Run:
//!   cargo run --example capture_npu_outputs
//!
//! Prerequisites:
//! - NPU hardware available (/dev/accel/accel0)
//! - npu-runner built: cd tools/npu-runner && cmake -B build && cmake --build build
//! - Test binaries built: ./scripts/build-mlir-aie-tests.sh

use std::path::PathBuf;
use xdna_emu::testing::test_cpp_parser;
use xdna_emu::testing::npu_runner;
use xdna_emu::testing::npu_test::TestOverrides;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    // Check prerequisites
    if !npu_runner::npu_available() {
        eprintln!("ERROR: NPU hardware not available.");
        eprintln!("  Ensure /dev/accel/accel0 exists (xdna-driver loaded).");
        std::process::exit(1);
    }

    if npu_runner::runner_binary().is_none() {
        eprintln!("ERROR: npu-runner binary not found.");
        eprintln!("  Build it with:");
        eprintln!("    cd tools/npu-runner && cmake -B build && cmake --build build");
        std::process::exit(1);
    }

    let config = xdna_emu::config::Config::get();
    let npu_xrt_dir = config.npu_xrt_test_dir();
    let test_src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../mlir-aie/test/npu-xrt");
    let output_base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");
    let overrides_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/test_overrides.toml");

    if !test_src_dir.exists() {
        eprintln!("ERROR: Test source directory not found: {}", test_src_dir.display());
        std::process::exit(1);
    }

    // Discover tests with test.cpp files
    let test_dirs = discover_test_dirs(&test_src_dir);

    let overrides = TestOverrides::load(&overrides_path);

    println!("=== NPU Output Capture ===");
    println!("Test source: {}", test_src_dir.display());
    println!("Binaries:    {}", npu_xrt_dir.display());
    println!("Output:      {}", output_base.display());
    println!("Found {} test directories\n", test_dirs.len());

    let mut captured = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for name in &test_dirs {
        print!("  {:<40} ", name);

        // Check overrides
        if let Some(reason) = overrides.skip.get(name.as_str()) {
            println!("SKIP ({})", reason);
            skipped += 1;
            continue;
        }

        // Parse test.cpp for buffer metadata
        let spec = match test_cpp_parser::parse_test_cpp(&test_src_dir.join(name)) {
            Some(s) => s,
            None => {
                println!("SKIP (no parseable test.cpp)");
                skipped += 1;
                continue;
            }
        };

        // Skip multi-kernel tests (not yet supported)
        if spec.multi_kernel {
            println!("SKIP (multi-kernel)");
            skipped += 1;
            continue;
        }

        // Find built xclbin
        let test_dir = npu_xrt_dir.join(name);
        let xclbin_path = test_dir.join("aie.xclbin");
        if !xclbin_path.exists() {
            println!("SKIP (no xclbin)");
            skipped += 1;
            continue;
        }

        // Find insts.bin
        let insts_path = if test_dir.join("insts.bin").exists() {
            test_dir.join("insts.bin")
        } else if test_dir.join("aie_run_seq.bin").exists() {
            test_dir.join("aie_run_seq.bin")
        } else {
            println!("SKIP (no insts.bin)");
            skipped += 1;
            continue;
        };

        // Run on NPU
        match npu_runner::run_on_npu(&spec, name, &xclbin_path, &insts_path, 30) {
            Ok(result) => {
                // Save output
                let out_dir = output_base.join(name);
                if let Err(e) = std::fs::create_dir_all(&out_dir) {
                    println!("FAIL (mkdir: {})", e);
                    failed += 1;
                    continue;
                }
                let out_path = out_dir.join("output.bin");
                if let Err(e) = std::fs::write(&out_path, &result.output) {
                    println!("FAIL (write: {})", e);
                    failed += 1;
                    continue;
                }
                println!("OK ({} bytes)", result.output.len());
                captured += 1;
            }
            Err(e) => {
                println!("FAIL ({})", e);
                failed += 1;
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Captured: {}", captured);
    println!("Failed:   {}", failed);
    println!("Skipped:  {}", skipped);
    println!("Total:    {}", test_dirs.len());

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Discover test directories that contain test.cpp files.
fn discover_test_dirs(src_dir: &std::path::Path) -> Vec<String> {
    let mut names = Vec::new();

    let entries = match std::fs::read_dir(src_dir) {
        Ok(e) => e,
        Err(_) => return names,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("test.cpp").exists() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                names.push(name.to_string());
            }
        }
    }

    names.sort();
    names
}
