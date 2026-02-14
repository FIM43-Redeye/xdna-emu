//! Integration tests for hardware cross-validation.
//!
//! These tests run only when the `hardware-compare` feature is enabled AND
//! pre-captured NPU output files exist in tests/npu-outputs/.
//!
//! Run:
//!   cargo test --features hardware-compare
//!
//! Prerequisites:
//! - Test binaries built: ./scripts/build-mlir-aie-tests.sh
//! - NPU outputs captured: cargo run --example capture_npu_outputs

#![cfg(feature = "hardware-compare")]

use std::collections::HashMap;
use std::path::PathBuf;

use xdna_emu::testing::manifest_runner::{TestManifest, ElementType, read_values};
use xdna_emu::testing::xclbin_suite::{XclbinSuite, XclbinTest};
use xdna_emu::testing::hardware_comparison::{self, CrossValidation};

/// Base directory for captured hardware outputs.
fn hw_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/npu-outputs")
}

/// Manifest directory.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/mlir-aie-extracted/manifests")
}

/// Load a manifest by test name.
fn load_manifest(name: &str) -> Option<TestManifest> {
    let path = manifest_dir().join(format!("{}.toml", name));
    TestManifest::from_file(&path).ok()
}

/// Load captured hardware output for a test.
fn load_hw_output(name: &str) -> Option<Vec<u8>> {
    let path = hw_output_dir().join(name).join("output.bin");
    std::fs::read(&path).ok()
}

/// Generate input values from a manifest.
fn generate_inputs(manifest: &TestManifest) -> HashMap<String, Vec<i64>> {
    let mut inputs = HashMap::new();
    for (buf_name, buf_def) in &manifest.buffers {
        if buf_name == "output" {
            continue;
        }
        if let Some(elem_type) = ElementType::from_str(&buf_def.element_type) {
            if let Some(data) = manifest.generate_input(buf_name) {
                inputs.insert(buf_name.clone(), read_values(&data, elem_type));
            }
        }
    }
    inputs
}

/// Run a single test through the emulator and cross-validate.
fn cross_validate_test(test_name: &str) -> Option<CrossValidation> {
    let manifest = load_manifest(test_name)?;
    if manifest.test.skip {
        return None;
    }

    let config = xdna_emu::config::Config::get();
    let xclbin_path = config.npu_xrt_test_dir().join(test_name).join("aie.xclbin");
    if !xclbin_path.exists() {
        return None;
    }

    let suite = XclbinSuite::new().with_max_cycles(1_000_000);
    let mut test = XclbinTest::from_path(&xclbin_path);
    test.manifest = Some(manifest.clone());

    let (_outcome, emu_output) = suite.run_single_with_output(&test);
    let hw_output = load_hw_output(test_name);
    let inputs = generate_inputs(&manifest);

    Some(CrossValidation::compare(
        test_name,
        &manifest,
        &inputs,
        emu_output.as_deref(),
        hw_output.as_deref(),
    ))
}

// -- Individual test functions -----------------------------------------------
// Each test exercises cross-validation for a specific mlir-aie test.
// Tests are skipped (not failed) if the xclbin or hardware capture is missing.

#[test]
fn cross_validate_add_one_using_dma() {
    if let Some(cv) = cross_validate_test("add_one_using_dma") {
        if let Some(ref hw) = cv.hw_vs_manifest {
            assert!(hw.is_match(), "Hardware should match manifest for add_one_using_dma");
        }
        if let Some(ref emu_hw) = cv.emu_vs_hw {
            // This is informational -- emulator may not match hardware yet
            eprintln!("add_one_using_dma: emu vs hw = {}", emu_hw.summary());
        }
    }
}

#[test]
fn cross_validate_add_314_using_dma_op() {
    if let Some(cv) = cross_validate_test("add_314_using_dma_op") {
        if let Some(ref hw) = cv.hw_vs_manifest {
            assert!(hw.is_match(), "Hardware should match manifest for add_314_using_dma_op");
        }
        if let Some(ref emu_hw) = cv.emu_vs_hw {
            eprintln!("add_314_using_dma_op: emu vs hw = {}", emu_hw.summary());
        }
    }
}

#[test]
fn cross_validate_vec_vec_add_tile_init() {
    if let Some(cv) = cross_validate_test("vec_vec_add_tile_init") {
        if let Some(ref hw) = cv.hw_vs_manifest {
            assert!(hw.is_match(), "Hardware should match manifest for vec_vec_add_tile_init");
        }
        if let Some(ref emu_hw) = cv.emu_vs_hw {
            eprintln!("vec_vec_add_tile_init: emu vs hw = {}", emu_hw.summary());
        }
    }
}

/// Batch cross-validation: run all tests with captures and produce a report.
#[test]
fn cross_validate_all_with_report() {
    let manifest_path = manifest_dir();
    if !manifest_path.exists() {
        eprintln!("Skipping: no manifest directory");
        return;
    }

    let entries: Vec<_> = std::fs::read_dir(&manifest_path)
        .expect("Cannot read manifest dir")
        .flatten()
        .filter(|e| {
            e.path().extension().map(|ext| ext == "toml").unwrap_or(false)
        })
        .collect();

    let mut results: Vec<CrossValidation> = Vec::new();

    for entry in &entries {
        let name = entry.path()
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        if let Some(cv) = cross_validate_test(&name) {
            results.push(cv);
        }
    }

    if !results.is_empty() {
        let report = hardware_comparison::format_report(&results);
        eprintln!("\n{}", report);
    } else {
        eprintln!("No tests could be cross-validated (missing binaries or captures)");
    }
}
