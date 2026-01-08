//! Test harness for running XCLBIN binaries through the emulator.
//!
//! This module provides infrastructure for:
//! - Discovering and running xclbin test files
//! - Collecting unknown opcodes when execution fails
//! - Validating output against expected results
//! - Running tests from TOML manifests (extracted from mlir-aie)
//!
//! # Usage
//!
//! ```bash
//! # Run xclbin test suite
//! cargo run -- test-suite /path/to/mlir-aie/build/test/npu-xrt/
//!
//! # Run manifest-based test
//! cargo run --example manifest_test -- tests/mlir-aie-extracted/manifests/add_one_using_dma.toml
//! ```

pub mod xclbin_suite;
pub mod opcode_collector;
pub mod manifest_runner;

pub use xclbin_suite::{XclbinSuite, XclbinTest, TestOutcome};
pub use opcode_collector::{UnknownOpcode, OpcodeCollector};
pub use manifest_runner::{TestManifest, ManifestRunner, TestResult, ElementType};
