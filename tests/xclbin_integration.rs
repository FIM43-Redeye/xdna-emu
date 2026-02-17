//! Integration tests for xclbin binary execution with manifest validation.
//!
//! These tests require built mlir-aie test binaries. Build them first:
//!
//! ```sh
//! ./scripts/build-mlir-aie-tests.sh
//! ```
//!
//! Then run:
//!
//! ```sh
//! cargo test --features xclbin-tests
//! ```
//!
//! Each test loads a real xclbin binary, runs it through the emulator,
//! and validates the output against the manifest's expected transform.
//! Tests marked `expected_fail` in their manifest will pass as long as
//! they fail in the expected way. Tests marked `skip` are not run.

#![cfg(feature = "xclbin-tests")]

use std::path::PathBuf;

use xdna_emu::testing::manifest_runner::TestManifest;
use xdna_emu::testing::xclbin_suite::{XclbinTest, XclbinSuite, TestOutcome};

/// Find the mlir-aie build directory.
fn mlir_aie_build() -> Option<PathBuf> {
    // Try environment variable first
    if let Ok(path) = std::env::var("MLIR_AIE_BUILD") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    // Try relative to workspace
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../mlir-aie/build"),
    ];

    for c in &candidates {
        if c.exists() {
            return Some(c.clone());
        }
    }

    None
}

/// Find the manifest directory.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mlir-aie-extracted/manifests")
}

/// Load a test by name: find the xclbin, load the manifest, create XclbinTest.
fn load_test(name: &str) -> Option<(XclbinTest, TestManifest)> {
    let build = mlir_aie_build()?;
    let test_dir = build.join("test/npu-xrt").join(name);

    // Find xclbin
    let xclbin_path = if test_dir.join("aie.xclbin").exists() {
        test_dir.join("aie.xclbin")
    } else if test_dir.join("final.xclbin").exists() {
        test_dir.join("final.xclbin")
    } else {
        return None;
    };

    // Load manifest
    let manifest_path = manifest_dir().join(format!("{}.toml", name));
    let manifest = TestManifest::from_file(&manifest_path).ok()?;

    let test = XclbinTest::from_path(&xclbin_path).with_manifest(manifest.clone());

    Some((test, manifest))
}

/// Run a test and return the outcome, handling skip/expected_fail.
fn run_test(name: &str) {
    let Some((test, manifest)) = load_test(name) else {
        eprintln!("SKIP {}: binary or manifest not found (run ./scripts/build-mlir-aie-tests.sh)", name);
        return;
    };

    if manifest.test.skip {
        eprintln!("SKIP {}: {}", name, manifest.test.skip_reason);
        return;
    }

    let suite = XclbinSuite::new().with_max_cycles(2_000_000);
    let outcome = suite.run_single(&test);

    match &outcome {
        TestOutcome::Pass { cycles, correct, total } => {
            if manifest.test.expected_fail {
                // Expected to fail but passed -- manifest needs updating
                panic!(
                    "UNEXPECTED PASS for {}: passed ({} cycles, {:?}/{:?} correct) \
                     but manifest says expected_fail=true. Update the manifest!",
                    name, cycles, correct, total
                );
            }
            eprintln!(
                "PASS {}: {} cycles, {:?}/{:?} validated",
                name, cycles, correct, total
            );
        }
        TestOutcome::ExpectedFail { cycles, reason, .. } => {
            eprintln!(
                "EXPECTED FAIL {}: {} cycles - {}",
                name, cycles, reason
            );
            // This is OK -- test failed as documented
        }
        TestOutcome::UnexpectedPass { cycles, correct, total } => {
            panic!(
                "UNEXPECTED PASS for {}: {} cycles, {}/{} correct. \
                 Remove expected_fail from manifest!",
                name, cycles, correct, total
            );
        }
        TestOutcome::Skipped { reason } => {
            eprintln!("SKIP {}: {}", name, reason);
        }
        TestOutcome::ValidationFail { cycles, correct, total, first_mismatch } => {
            let mismatch_msg = first_mismatch
                .map(|(i, exp, act)| format!(" (first mismatch at [{}]: expected {}, got {})", i, exp, act))
                .unwrap_or_default();
            panic!(
                "VALIDATION FAIL for {}: {} cycles, {}/{} correct{}",
                name, cycles, correct, total, mismatch_msg
            );
        }
        TestOutcome::Fail { message, cycles } => {
            panic!("FAIL for {}: {} cycles - {}", name, cycles, message);
        }
        TestOutcome::UnknownOpcode { details, cycles } => {
            panic!(
                "UNKNOWN OPCODE for {}: {} cycles - slot {:?}, opcode 0x{:08X} at PC 0x{:04X}",
                name, cycles, details.slot, details.opcode, details.pc
            );
        }
        TestOutcome::Timeout { cycles } => {
            panic!("TIMEOUT for {}: {} cycles", name, cycles);
        }
        TestOutcome::LoadError { message } => {
            panic!("LOAD ERROR for {}: {}", name, message);
        }
        TestOutcome::Platform { required, reason } => {
            eprintln!("PLATFORM SKIP {}: requires {} - {}", name, required, reason);
        }
    }
}

// ===== Individual test functions =====
// Each maps to a manifest file and (potentially) a built test binary.
// Tests are ordered alphabetically for easy scanning.

#[test]
fn test_add_256_using_dma_op_no_double_buffering() {
    run_test("add_256_using_dma_op_no_double_buffering");
}

#[test]
fn test_add_314_using_dma_op() {
    run_test("add_314_using_dma_op");
}

#[test]
fn test_add_blockwrite() {
    run_test("add_blockwrite");
}

#[test]
fn test_add_one_objfifo() {
    run_test("add_one_objFifo");
}

#[test]
fn test_add_one_objfifo_elf() {
    run_test("add_one_objFifo_elf");
}

#[test]
fn test_add_one_two() {
    run_test("add_one_two");
}

#[test]
fn test_add_one_two_runlist() {
    run_test("add_one_two_runlist");
}

#[test]
fn test_add_one_two_txn() {
    run_test("add_one_two_txn");
}

#[test]
fn test_add_one_using_dma() {
    run_test("add_one_using_dma");
}

#[test]
fn test_cascade_flows() {
    run_test("cascade_flows");
}

#[test]
fn test_neighbor_tile_memory_access() {
    run_test("neighbor_tile_memory_access");
}

#[test]
fn test_packet_flow() {
    run_test("packet_flow");
}

#[test]
fn test_static_l1_init() {
    run_test("static_L1_init");
}

#[test]
fn test_vector_scalar_add() {
    run_test("vector_scalar_add");
}

#[test]
fn test_vector_scalar_add_runlist() {
    run_test("vector_scalar_add_runlist");
}

#[test]
fn test_vec_vec_add_memtile_init() {
    run_test("vec_vec_add_memtile_init");
}

#[test]
fn test_vec_vec_add_tile_init() {
    run_test("vec_vec_add_tile_init");
}
