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

use std::path::PathBuf;

use xdna_emu::testing::test_cpp_parser::{self, ElementType};
use xdna_emu::testing::xclbin_suite::{XclbinSuite, XclbinTest};
use xdna_emu::testing::hardware_comparison::{self, CrossValidation};

/// Base directory for captured hardware outputs.
fn hw_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/npu-outputs")
}

/// Load captured hardware output for a test.
fn load_hw_output(name: &str) -> Option<Vec<u8>> {
    let path = hw_output_dir().join(name).join("output.bin");
    std::fs::read(&path).ok()
}

/// Determine the output element type for a test from its test.cpp.
fn output_element_type(test_name: &str) -> ElementType {
    let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../mlir-aie/test/npu-xrt")
        .join(test_name);

    if let Some(spec) = test_cpp_parser::parse_test_cpp(&test_dir) {
        spec.buffers.iter()
            .find(|b| b.direction == test_cpp_parser::BufferDir::Output)
            .map(|b| b.element_type)
            .unwrap_or(ElementType::I32)
    } else {
        ElementType::I32
    }
}

/// Run a single test through the emulator and cross-validate against
/// hardware reference output.
fn cross_validate_test(test_name: &str) -> Option<CrossValidation> {
    let config = xdna_emu::config::Config::get();
    let xclbin_path = config.npu_xrt_test_dir().join(test_name).join("aie.xclbin");
    if !xclbin_path.exists() {
        return None;
    }

    // Parse test.cpp for buffer metadata
    let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../mlir-aie/test/npu-xrt")
        .join(test_name);
    let buffer_spec = test_cpp_parser::parse_test_cpp(&test_dir);

    let suite = XclbinSuite::new().with_max_cycles(1_000_000);
    let mut test = XclbinTest::from_path(&xclbin_path);
    test.buffer_spec = buffer_spec;

    let (_outcome, emu_output) = suite.run_single_with_output(&test);
    let hw_output = load_hw_output(test_name);
    let elem_type = output_element_type(test_name);

    Some(CrossValidation::compare(
        test_name,
        emu_output.as_deref(),
        hw_output.as_deref(),
        elem_type,
    ))
}

// -- Individual test functions -----------------------------------------------
// Each test exercises cross-validation for a specific mlir-aie test.
// Tests are skipped (not failed) if the xclbin or hardware capture is missing.

#[test]
fn cross_validate_add_one_using_dma() {
    if let Some(cv) = cross_validate_test("add_one_using_dma") {
        if let Some(ref result) = cv.emu_vs_reference {
            eprintln!("add_one_using_dma: emu vs hw = {}", result.summary());
        }
    }
}

#[test]
fn cross_validate_add_314_using_dma_op() {
    if let Some(cv) = cross_validate_test("add_314_using_dma_op") {
        if let Some(ref result) = cv.emu_vs_reference {
            eprintln!("add_314_using_dma_op: emu vs hw = {}", result.summary());
        }
    }
}

#[test]
fn cross_validate_vec_vec_add_tile_init() {
    if let Some(cv) = cross_validate_test("vec_vec_add_tile_init") {
        if let Some(ref result) = cv.emu_vs_reference {
            eprintln!("vec_vec_add_tile_init: emu vs hw = {}", result.summary());
        }
    }
}

/// Batch cross-validation: run all discoverable tests and produce a report.
#[test]
fn cross_validate_all_with_report() {
    let hw_dir = hw_output_dir();
    if !hw_dir.exists() {
        eprintln!("Skipping: no hardware output directory");
        return;
    }

    let entries: Vec<_> = std::fs::read_dir(&hw_dir)
        .expect("Cannot read hardware output dir")
        .flatten()
        .filter(|e| e.path().is_dir())
        .collect();

    let mut results: Vec<CrossValidation> = Vec::new();

    for entry in &entries {
        let name = entry.file_name().to_string_lossy().to_string();
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
