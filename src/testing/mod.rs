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

pub mod artifacts;
pub mod xclbin_suite;
pub mod opcode_collector;
pub mod npu_runner;
pub mod hardware_comparison;
pub mod unit_test;
pub mod npu_test;
pub mod test_cpp_parser;
pub mod native_hw;
pub mod process_control;
pub mod host_defines;

/// Default maximum emulator cycles before timeout.
pub const DEFAULT_MAX_CYCLES: u64 = 1_000_000;

/// Default hardware execution timeout in seconds.
pub const DEFAULT_HW_TIMEOUT_SECS: u32 = 30;

pub use xclbin_suite::{XclbinSuite, XclbinTest, TestOutcome, Compiler};
pub use opcode_collector::{UnknownOpcode, OpcodeCollector};
pub use test_cpp_parser::{BufferSpec, ElementType};
pub use hardware_comparison::{Diagnosis, HardwareValidation};
pub use unit_test::{UnitTest, UnitTestBuildResult};
pub use npu_test::NpuTestSource;

/// Build an LD_LIBRARY_PATH that strips aietools entries.
///
/// aietools ships an outdated libstdc++ that shadows the system copy and
/// crashes XRT-linked executables. This helper filters those paths out so
/// child processes (npu-runner, native test.exe) link against the system
/// runtime.
pub fn sanitized_ld_library_path() -> String {
    let current = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    current
        .split(':')
        .filter(|p| !p.contains("aietools"))
        .collect::<Vec<_>>()
        .join(":")
}
