//! Capture NPU hardware outputs for all manifest-defined tests.
//!
//! For each test manifest with a corresponding built xclbin + insts.bin:
//! 1. Generates input data from the manifest pattern
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
use xdna_emu::testing::manifest_runner::TestManifest;
use xdna_emu::testing::npu_runner;

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
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mlir-aie-extracted/manifests");
    let output_base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");

    if !manifest_dir.exists() {
        eprintln!("ERROR: Manifest directory not found: {}", manifest_dir.display());
        std::process::exit(1);
    }

    // Discover manifests
    let manifests = match discover_manifests(&manifest_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: Failed to read manifests: {}", e);
            std::process::exit(1);
        }
    };

    println!("=== NPU Output Capture ===");
    println!("Manifests: {}", manifest_dir.display());
    println!("Binaries:  {}", npu_xrt_dir.display());
    println!("Output:    {}", output_base.display());
    println!("Found {} manifests\n", manifests.len());

    let mut captured = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for (name, manifest) in &manifests {
        print!("  {:<40} ", name);

        // Skip tests requiring different hardware platform
        if !manifest.test.platform.is_empty() {
            println!("PLATFORM (requires {})", manifest.test.platform);
            skipped += 1;
            continue;
        }

        // Skip tests marked as skip
        if manifest.test.skip {
            println!("SKIP ({})", manifest.test.skip_reason);
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

        // Find insts.bin (try both names).
        // For multi-kernel tests, insts_path is unused (instruction paths
        // come from the manifest's multi_kernel.runs), but we still need
        // a path for the API.  Use a dummy path and let run_on_npu handle it.
        let insts_path = if test_dir.join("insts.bin").exists() {
            test_dir.join("insts.bin")
        } else if test_dir.join("aie_run_seq.bin").exists() {
            test_dir.join("aie_run_seq.bin")
        } else if manifest.is_multi_kernel() {
            // Multi-kernel tests get insts paths from the manifest
            test_dir.join("insts.bin")  // placeholder, not actually read
        } else {
            println!("SKIP (no insts.bin)");
            skipped += 1;
            continue;
        };

        // Run on NPU
        match npu_runner::run_on_npu(manifest, &xclbin_path, &insts_path, 30) {
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
    println!("Total:    {}", manifests.len());

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Discover all TOML manifest files in a directory.
fn discover_manifests(dir: &std::path::Path) -> Result<Vec<(String, TestManifest)>, String> {
    let mut manifests = Vec::new();

    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Cannot read directory: {}", e))?;

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
