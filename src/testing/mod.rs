//! Test harness for running XCLBIN binaries through the emulator.
//!
//! This module provides infrastructure for:
//! - Discovering and running xclbin test files
//! - Collecting unknown opcodes when execution fails
//! - Validating output against hardware reference captures
//! - Parsing buffer metadata from test.cpp files
//!
//! # Usage
//!
//! ```bash
//! # Run xclbin test suite
//! cargo run -- test-suite /path/to/mlir-aie/build/test/npu-xrt/
//! ```

pub mod xclbin_suite;
pub mod opcode_collector;
pub mod npu_runner;
pub mod hardware_comparison;
pub mod unit_test;
pub mod npu_test;
pub mod test_cpp_parser;
pub mod runner_config;
pub mod runner_stats;
pub mod hw_executor;
pub mod runner_display;
pub mod native_hw;

pub use xclbin_suite::{XclbinSuite, XclbinTest, TestOutcome, Compiler};
pub use opcode_collector::{UnknownOpcode, OpcodeCollector};
pub use test_cpp_parser::{BufferSpec, ElementType};
pub use hardware_comparison::{Diagnosis, HardwareValidation};
pub use unit_test::{UnitTest, UnitTestBuildResult};
pub use npu_test::NpuTestSource;
