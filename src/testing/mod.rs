//! Test harness for running XCLBIN binaries through the emulator.
//!
//! This module provides infrastructure for:
//! - Discovering and running xclbin test files
//! - Collecting unknown opcodes when execution fails
//! - Validating output against expected results
//!
//! # Usage
//!
//! ```bash
//! cargo run -- test-suite /path/to/mlir-aie/build/test/npu-xrt/
//! ```

pub mod xclbin_suite;
pub mod opcode_collector;

pub use xclbin_suite::{XclbinSuite, XclbinTest, TestOutcome};
pub use opcode_collector::{UnknownOpcode, OpcodeCollector};
