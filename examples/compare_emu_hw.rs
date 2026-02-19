//! Cross-validation: emulator vs hardware reference output.
//!
//! Discovers all tests with test.cpp files, runs them through the emulator,
//! loads captured hardware outputs, and produces a comparison report.
//!
//! Run:
//!   cargo run --example compare_emu_hw
//!
//! Prerequisites:
//! - Test binaries built: ./scripts/build-mlir-aie-tests.sh
//! - (Optional) NPU outputs captured: cargo run --example capture_npu_outputs

use std::path::PathBuf;

use xdna_emu::testing::test_cpp_parser;
use xdna_emu::testing::xclbin_suite::{XclbinSuite, XclbinTest};
use xdna_emu::testing::hardware_comparison::{self, CrossValidation};
use xdna_emu::testing::npu_test::TestOverrides;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error"),
    )
    .init();

    let config = xdna_emu::config::Config::get();
    let npu_xrt_dir = config.npu_xrt_test_dir();
    let test_src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../mlir-aie/test/npu-xrt");
    let hw_output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");
    let overrides_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/test_overrides.toml");

    println!("=== Cross-Validation: Emulator vs Hardware ===\n");
    println!("Test source:  {}", test_src_dir.display());
    println!("Binaries:     {}", npu_xrt_dir.display());
    println!("HW captures:  {}", hw_output_dir.display());

    let has_hw_outputs = hw_output_dir.exists();
    if !has_hw_outputs {
        println!("  (no hardware captures found -- comparison will show N/A)");
        println!("  Run: cargo run --example capture_npu_outputs");
    }
    println!();

    // Discover tests from test source directory
    let test_dirs = match discover_test_dirs(&test_src_dir) {
        Ok(dirs) => dirs,
        Err(e) => {
            eprintln!("ERROR: {}", e);
            std::process::exit(1);
        }
    };

    let overrides = TestOverrides::load(&overrides_path);

    // Set up emulator suite
    let suite = XclbinSuite::new()
        .with_max_cycles(1_000_000);

    let mut cross_results: Vec<CrossValidation> = Vec::new();

    for name in &test_dirs {
        // Check overrides
        if let Some(reason) = overrides.skip.get(name.as_str()) {
            eprintln!("  {:<35} SKIP ({})", name, reason);
            continue;
        }

        // Find built xclbin
        let test_dir = npu_xrt_dir.join(name);
        let xclbin_path = test_dir.join("aie.xclbin");
        if !xclbin_path.exists() {
            continue;
        }

        // Parse test.cpp for buffer metadata
        let buffer_spec = test_cpp_parser::parse_test_cpp(&test_src_dir.join(name));

        // Create test and run through emulator
        let mut test = XclbinTest::from_path(&xclbin_path);
        test.buffer_spec = buffer_spec.clone();

        if let Some(reason) = overrides.expected_fail.get(name.as_str()) {
            test.expected_fail_reason = Some(reason.clone());
        }

        let (outcome, emu_output) = suite.run_single_with_output(&test);

        // Determine output element type
        let elem_type = buffer_spec.as_ref()
            .and_then(|s| s.buffers.iter().find(|b| b.direction == test_cpp_parser::BufferDir::Output))
            .map(|b| b.element_type)
            .unwrap_or(test_cpp_parser::ElementType::I32);

        // Load hardware output if available
        let hw_output = if has_hw_outputs {
            let hw_path = hw_output_dir.join(name).join("output.bin");
            std::fs::read(&hw_path).ok()
        } else {
            None
        };

        // Run comparison
        let cv = CrossValidation::compare(
            name,
            emu_output.as_deref(),
            hw_output.as_deref(),
            elem_type,
        );

        // Print per-test status line
        let emu_status = match &outcome {
            xdna_emu::testing::TestOutcome::Pass { cycles, .. } => {
                format!("PASS ({}c)", cycles)
            }
            xdna_emu::testing::TestOutcome::ValidationFail { cycles, correct, total, .. } => {
                format!("VFAIL {}/{} ({}c)", correct, total, cycles)
            }
            xdna_emu::testing::TestOutcome::ExpectedFail { cycles, .. } => {
                format!("XFAIL ({}c)", cycles)
            }
            xdna_emu::testing::TestOutcome::Timeout { cycles } => {
                format!("TIMEOUT ({}c)", cycles)
            }
            xdna_emu::testing::TestOutcome::UnknownOpcode { cycles, .. } => {
                format!("UNKNOWN ({}c)", cycles)
            }
            _ => format!("{:?}", std::mem::discriminant(&outcome)),
        };
        eprintln!("  {:<35} {}", name, emu_status);

        cross_results.push(cv);
    }

    // Print the full report
    println!();
    print!("{}", hardware_comparison::format_report(&cross_results));
}

/// Discover test directories that contain test.cpp files.
fn discover_test_dirs(src_dir: &std::path::Path) -> Result<Vec<String>, String> {
    let mut names = Vec::new();

    if !src_dir.exists() {
        return Err(format!("Test source directory not found: {}", src_dir.display()));
    }

    let entries = std::fs::read_dir(src_dir)
        .map_err(|e| format!("Cannot read directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("test.cpp").exists() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                names.push(name.to_string());
            }
        }
    }

    names.sort();
    Ok(names)
}
