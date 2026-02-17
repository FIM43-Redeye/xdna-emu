//! Three-way cross-validation: emulator vs manifest vs hardware.
//!
//! Runs all manifest-defined tests through the emulator, loads any
//! captured hardware outputs, and produces a comparison report showing
//! all three validation layers.
//!
//! Run:
//!   cargo run --example compare_emu_hw
//!
//! Prerequisites:
//! - Test binaries built: ./scripts/build-mlir-aie-tests.sh
//! - (Optional) NPU outputs captured: cargo run --example capture_npu_outputs

use std::path::PathBuf;

use xdna_emu::testing::manifest_runner::TestManifest;
use xdna_emu::testing::xclbin_suite::{XclbinSuite, XclbinTest};
use xdna_emu::testing::hardware_comparison::{self, CrossValidation};

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error"),
    )
    .init();

    let config = xdna_emu::config::Config::get();
    let npu_xrt_dir = config.npu_xrt_test_dir();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mlir-aie-extracted/manifests");
    let hw_output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");

    if !manifest_dir.exists() {
        eprintln!("ERROR: Manifest directory not found: {}", manifest_dir.display());
        std::process::exit(1);
    }

    println!("=== Cross-Validation: Emulator vs Hardware ===\n");
    println!("Manifests:    {}", manifest_dir.display());
    println!("Binaries:     {}", npu_xrt_dir.display());
    println!("HW captures:  {}", hw_output_dir.display());

    let has_hw_outputs = hw_output_dir.exists();
    if !has_hw_outputs {
        println!("  (no hardware captures found -- Layer 2 and 3 will show N/A)");
        println!("  Run: cargo run --example capture_npu_outputs");
    }
    println!();

    // Discover manifests
    let manifests = match discover_manifests(&manifest_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: {}", e);
            std::process::exit(1);
        }
    };

    // Set up emulator suite
    let suite = XclbinSuite::new()
        .with_max_cycles(1_000_000)
        .with_manifest_dir(manifest_dir.clone());

    let mut cross_results: Vec<CrossValidation> = Vec::new();

    for (name, manifest) in &manifests {
        // Skip tests marked as skip
        if manifest.test.skip {
            continue;
        }

        // Find built xclbin
        let test_dir = npu_xrt_dir.join(name);
        let xclbin_path = test_dir.join("aie.xclbin");
        if !xclbin_path.exists() {
            continue; // No binary available
        }

        // Create test and run through emulator
        let mut test = XclbinTest::from_path(&xclbin_path);
        test.manifest = Some(manifest.clone());

        let (outcome, emu_output) = suite.run_single_with_output(&test);

        // Generate input values for comparison
        let input_values = hardware_comparison::generate_input_values(manifest);

        // Load hardware output if available
        let hw_output = if has_hw_outputs {
            let hw_path = hw_output_dir.join(name).join("output.bin");
            if hw_path.exists() {
                std::fs::read(&hw_path).ok()
            } else {
                None
            }
        } else {
            None
        };

        // Run three-way comparison
        let cv = CrossValidation::compare(
            name,
            manifest,
            &input_values,
            emu_output.as_deref(),
            hw_output.as_deref(),
            if has_hw_outputs { Some(hw_output_dir.as_path()) } else { None },
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

/// Discover all TOML manifest files in a directory.
fn discover_manifests(dir: &std::path::Path) -> Result<Vec<(String, TestManifest)>, String> {
    let mut manifests = Vec::new();

    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Cannot read manifest directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "toml").unwrap_or(false) {
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            match TestManifest::from_file(&path) {
                Ok(manifest) => manifests.push((name, manifest)),
                Err(e) => {
                    eprintln!("  WARNING: Failed to parse {}: {}", path.display(), e);
                }
            }
        }
    }

    manifests.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(manifests)
}
