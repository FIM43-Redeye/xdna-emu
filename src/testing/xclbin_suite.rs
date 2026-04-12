//! XCLBIN test suite for running binary compatibility tests.
//!
//! This module provides test discovery and execution for xclbin files.

use std::path::{Path, PathBuf};
use anyhow::Result;

use crate::parser::{Xclbin, AiePartition, Cdo};
use crate::parser::xclbin::SectionKind;
use crate::parser::cdo::find_cdo_offset;
use crate::interpreter::engine::{InterpreterEngine, EngineStatus};
use crate::npu::{NpuInstructionStream, NpuExecutor, HostBuffer};

use super::opcode_collector::{OpcodeCollector, UnknownOpcode};
use super::test_cpp_parser::{BufferSpec, BufferDir, ElementType, read_values, generate_input_data};
use super::hardware_comparison::{
    self, CrossValidation, HardwareValidation, Diagnosis,
};
use super::native_hw::TestCppPattern;
use std::collections::HashMap;

/// Which compiler produced an xclbin artifact.
///
/// Tracks the provenance of each test so hardware execution statistics
/// are attributed to the correct compiler. Without this, Chess-only tests
/// that land in `primary_tests` would be mislabeled as Peano.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compiler {
    Peano,
    Chess,
}

/// Result of running a single xclbin test.
#[derive(Debug)]
pub enum TestOutcome {
    /// Test completed successfully with validated output.
    Pass {
        cycles: u64,
        /// Number of correct output values (if validation was performed)
        correct: Option<usize>,
        /// Total output values checked
        total: Option<usize>,
    },
    /// Test failed validation (output mismatch).
    ValidationFail {
        cycles: u64,
        correct: usize,
        total: usize,
        first_mismatch: Option<(usize, i64, i64)>, // (index, expected, actual)
    },
    /// Known-broken test failed as expected. The test has an
    /// expected_fail_reason in test_overrides.toml, and did produce wrong output.
    /// This is informational, not a test failure.
    ///
    /// The `actual` field preserves the underlying failure details so we can
    /// track progress without removing the expected_fail entry.
    ExpectedFail {
        cycles: u64,
        reason: String,
        /// Human-readable description of the actual failure (e.g. "ValidationFail: 0/64 correct")
        actual: String,
    },
    /// Known-broken test unexpectedly passed. The test has an
    /// expected_fail_reason in test_overrides.toml, but produced correct output.
    /// This means the emulator improved or the override needs removing.
    UnexpectedPass {
        cycles: u64,
        correct: usize,
        total: usize,
    },
    /// Test was skipped (e.g. requires xchesscc, missing binary).
    Skipped {
        reason: String,
    },
    /// Test requires a different hardware platform (e.g. npu2/Strix Point).
    /// Distinct from Skipped: these tests are gated by hardware generation,
    /// not by missing features or tooling.
    Platform {
        required: String,
        reason: String,
    },
    /// Test failed with an error.
    Fail {
        message: String,
        cycles: u64,
    },
    /// Test hit an unknown instruction.
    UnknownOpcode {
        details: UnknownOpcode,
        cycles: u64,
    },
    /// Test timed out (didn't halt within max cycles).
    Timeout {
        cycles: u64,
    },
    /// Test couldn't be loaded.
    LoadError {
        message: String,
    },
}

impl TestOutcome {
    /// Check if the test passed (or expected-fail matched expectations).
    pub fn is_pass(&self) -> bool {
        matches!(self, TestOutcome::Pass { .. } | TestOutcome::ExpectedFail { .. })
    }

    /// Check if the test failed validation.
    pub fn is_validation_fail(&self) -> bool {
        matches!(self, TestOutcome::ValidationFail { .. })
    }

    /// Check if the test was skipped.
    pub fn is_skipped(&self) -> bool {
        matches!(self, TestOutcome::Skipped { .. })
    }

    /// Check if a known-broken test unexpectedly passed.
    pub fn is_unexpected_pass(&self) -> bool {
        matches!(self, TestOutcome::UnexpectedPass { .. })
    }

    /// Get the cycle count (if available).
    pub fn cycles(&self) -> Option<u64> {
        match self {
            TestOutcome::Pass { cycles, .. } => Some(*cycles),
            TestOutcome::ValidationFail { cycles, .. } => Some(*cycles),
            TestOutcome::ExpectedFail { cycles, .. } => Some(*cycles),
            TestOutcome::UnexpectedPass { cycles, .. } => Some(*cycles),
            TestOutcome::Fail { cycles, .. } => Some(*cycles),
            TestOutcome::UnknownOpcode { cycles, .. } => Some(*cycles),
            TestOutcome::Timeout { cycles } => Some(*cycles),
            TestOutcome::Skipped { .. } | TestOutcome::Platform { .. } | TestOutcome::LoadError { .. } => None,
        }
    }
}

/// A single xclbin test case.
#[derive(Debug, Clone)]
pub struct XclbinTest {
    /// Test name (usually directory name).
    pub name: String,
    /// Path to the xclbin file.
    pub xclbin_path: PathBuf,
    /// Path to the project directory containing ELF files.
    pub project_dir: Option<PathBuf>,
    /// Optional expected output for validation.
    pub expected_output: Option<Vec<u8>>,
    /// Buffer metadata parsed from test.cpp (sizes, types, group IDs, input patterns).
    pub buffer_spec: Option<BufferSpec>,
    /// Skip reason from test_overrides.toml.
    pub skip_reason: Option<String>,
    /// Expected failure reason from test_overrides.toml.
    pub expected_fail_reason: Option<String>,
    /// Explicit path to NPU instructions file (overrides directory-based discovery).
    /// Set for multi-xclbin tests where each xclbin has a distinct insts file.
    pub insts_path: Option<PathBuf>,
    /// Which compiler produced this xclbin (Peano or Chess).
    /// Set by the build system in `collect_results()`. `None` for legacy
    /// tests discovered from pre-built directories (treated as Peano).
    pub compiler: Option<Compiler>,
    /// Path to compiled native test.exe for hardware execution.
    /// Set during the build phase when test.cpp is compiled.
    pub test_exe: Option<PathBuf>,
    /// Which argument pattern the test.cpp uses (cxxopts vs #define).
    pub test_cpp_pattern: Option<TestCppPattern>,
    /// Path to the original test source directory (for test.cpp location).
    pub source_dir: Option<PathBuf>,
}

impl XclbinTest {
    /// Create a new test from an xclbin path.
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let parent = path.parent();
        let name = parent
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Look for project directory with ELF files
        // Common patterns: aie_arch.mlir.prj/, aie.mlir.prj/, *.prj/
        let project_dir = parent.and_then(|p| {
            // Try known project directory names
            for pattern in &["aie_arch.mlir.prj", "aie.mlir.prj"] {
                let prj = p.join(pattern);
                if prj.exists() && prj.is_dir() {
                    return Some(prj);
                }
            }
            // Try to find any .prj directory
            if let Ok(entries) = std::fs::read_dir(p) {
                for entry in entries.flatten() {
                    let entry_path = entry.path();
                    if entry_path.is_dir() {
                        if let Some(name) = entry_path.file_name() {
                            if name.to_string_lossy().ends_with(".prj") {
                                return Some(entry_path);
                            }
                        }
                    }
                }
            }
            None
        });

        // Check for companion files in the same directory as the xclbin.
        // These are present in pre-built mlir-aie build trees and allow
        // --no-build discovery to find everything needed for HW execution.
        // Some lit tests produce "test.exe", others just "test" (Linux ELF).
        let test_exe = parent.and_then(|p| {
            let exe = p.join("test.exe");
            if exe.exists() { return Some(exe); }
            let test = p.join("test");
            if test.exists() && test.is_file() { return Some(test); }
            None
        });
        // Instruction file: most tests use insts.bin (raw NPU instructions),
        // but add_one_objFifo_elf uses insts.elf (ELF-wrapped instructions
        // loaded via xrt::elf experimental API).
        let insts_path = parent.and_then(|p| {
            let bin = p.join("insts.bin");
            if bin.exists() { return Some(bin); }
            let elf = p.join("insts.elf");
            if elf.exists() { return Some(elf); }
            None
        });

        Self {
            name,
            xclbin_path: path.to_path_buf(),
            project_dir,
            expected_output: None,
            buffer_spec: None,
            skip_reason: None,
            expected_fail_reason: None,
            insts_path,
            compiler: None,
            test_exe,
            test_cpp_pattern: None,
            source_dir: None,
        }
    }

    /// Set the expected output for validation.
    pub fn with_expected_output(mut self, output: Vec<u8>) -> Self {
        self.expected_output = Some(output);
        self
    }

    /// Set the buffer spec for input setup and validation.
    pub fn with_buffer_spec(mut self, spec: BufferSpec) -> Self {
        self.buffer_spec = Some(spec);
        self
    }

    /// Find ELF files for cores in the project directory.
    pub fn find_elf_files(&self) -> Vec<(u8, u8, PathBuf)> {
        let mut elfs = Vec::new();

        if let Some(ref prj_dir) = self.project_dir {
            if let Ok(entries) = std::fs::read_dir(prj_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name() {
                        let name_str = name.to_string_lossy();
                        // Pattern: main_core_X_Y.elf or core_X_Y.elf
                        if name_str.ends_with(".elf") && name_str.contains("core_") {
                            // Extract col and row from filename
                            if let Some((col, row)) = Self::parse_core_coords(&name_str) {
                                elfs.push((col, row, path));
                            }
                        }
                    }
                }
            }
        }

        elfs
    }

    /// Parse core coordinates from filename like "main_core_0_2.elf"
    fn parse_core_coords(name: &str) -> Option<(u8, u8)> {
        // Find "core_" and parse the numbers after it
        let core_idx = name.find("core_")?;
        let after_core = &name[core_idx + 5..];
        let parts: Vec<&str> = after_core.split('_').take(2).collect();
        if parts.len() >= 2 {
            let col: u8 = parts[0].parse().ok()?;
            // Remove .elf from row
            let row_str = parts[1].trim_end_matches(".elf");
            let row: u8 = row_str.parse().ok()?;
            return Some((col, row));
        }
        None
    }

    /// Find the NPU instructions file for this test.
    ///
    /// The insts file contains host-to-NPU commands that configure
    /// and trigger shim DMA transfers. Uses the explicit `insts_path` if set
    /// (for multi-xclbin tests), otherwise falls back to looking for
    /// `insts.bin` or `insts.elf` in the xclbin's parent directory.
    pub fn find_insts_file(&self) -> Option<PathBuf> {
        // Use explicit path if set (multi-xclbin tests)
        if let Some(ref path) = self.insts_path {
            if path.exists() {
                return Some(path.clone());
            }
        }
        // Fallback: look in the same directory as the xclbin
        self.xclbin_path.parent().and_then(super::artifacts::find_insts)
    }

    /// Count the number of compute tiles with embedded code in the xclbin.
    ///
    /// Parses the CDO and looks for DMA_WRITE commands to program memory
    /// (offset 0x20000). Returns the count of unique tiles with embedded code.
    pub fn count_embedded_cores(&self) -> usize {
        use crate::parser::cdo::CdoCommand;
        use crate::device::TileAddress;
        use std::collections::HashSet;

        let xclbin = match Xclbin::from_file(&self.xclbin_path) {
            Ok(x) => x,
            Err(_) => return 0,
        };

        let partition_section = match xclbin.find_section(SectionKind::AiePartition) {
            Some(s) => s,
            None => return 0,
        };

        let partition = match AiePartition::parse(partition_section.data()) {
            Ok(p) => p,
            Err(_) => return 0,
        };

        let mut embedded_cores: HashSet<(u8, u8)> = HashSet::new();

        for pdi in partition.pdis() {
            if let Some(cdo_offset) = find_cdo_offset(pdi.pdi_image) {
                if let Ok(cdo) = Cdo::parse(&pdi.pdi_image[cdo_offset..]) {
                    for cmd in cdo.commands() {
                        if let CdoCommand::DmaWrite { address, data } = &cmd {
                            let tile = TileAddress::decode(*address);
                            // ProgMem starts at offset 0x20000 for compute tiles
                            if tile.offset >= 0x20000 && tile.offset < 0x40000 && !data.is_empty() {
                                embedded_cores.insert((tile.col, tile.row));
                            }
                        }
                    }
                }
            }
        }

        embedded_cores.len()
    }
}

/// Test suite for running multiple xclbin tests.
pub struct XclbinSuite {
    /// Tests to run.
    tests: Vec<XclbinTest>,
    /// Collected unknown opcodes across all tests.
    collector: OpcodeCollector,
    /// Absolute cycle safety net. 0 means no limit (rely on TDR only).
    max_cycles: u64,
    /// Stall detection threshold. 0 means disabled (rely on max_cycles only).
    ///
    /// Number of consecutive cycles with no lock-release progress while
    /// pending syncs remain unsatisfied before declaring a stall.
    stall_threshold: u64,
    /// Results from completed tests.
    results: Vec<(String, TestOutcome, Option<HardwareValidation>)>,
    /// Directory containing hardware reference outputs.
    /// Each test's captured output is at `{reference_dir}/{test_name}/output.bin`.
    reference_dir: Option<PathBuf>,
    /// Directory containing captured hardware outputs for cross-validation.
    /// Each test's output is at `{npu_output_dir}/{test_name}/output.bin`.
    npu_output_dir: Option<PathBuf>,
}

impl XclbinSuite {
    /// Create a new test suite.
    ///
    /// Reads `max_cycles` from [`super::DEFAULT_MAX_CYCLES`] and
    /// `stall_threshold` from the global [`crate::config::Config`].
    pub fn new() -> Self {
        let stall_threshold = crate::config::Config::get().stall_threshold();
        Self {
            tests: Vec::new(),
            collector: OpcodeCollector::new(),
            max_cycles: super::DEFAULT_MAX_CYCLES,
            stall_threshold,
            results: Vec::new(),
            reference_dir: None,
            npu_output_dir: None,
        }
    }

    /// Create a test suite from a pre-built list of tests.
    ///
    /// Unlike `discover()` which walks a directory tree, this constructor
    /// accepts tests that have already been built (e.g. by `batch_build_peano()`).
    /// Tests are used as-is, with their buffer specs already attached.
    pub fn from_tests(tests: Vec<XclbinTest>) -> Self {
        Self {
            tests,
            ..Self::new()
        }
    }

    /// Set the absolute cycle safety net.
    ///
    /// This is a hard upper bound that catches anything the TDR misses.
    /// Set to 0 to disable entirely (rely on TDR no-progress detection only).
    pub fn with_max_cycles(mut self, max: u64) -> Self {
        self.max_cycles = max;
        self
    }

    /// Set the stall detection threshold in cycles.
    ///
    /// Number of consecutive cycles with no lock-release progress while
    /// pending syncs remain unsatisfied before declaring a stall.
    /// Set to 0 to disable stall detection.
    pub fn with_stall_threshold(mut self, threshold: u64) -> Self {
        self.stall_threshold = threshold;
        self
    }

    /// Set the directory containing hardware reference outputs.
    pub fn with_reference_dir(mut self, dir: PathBuf) -> Self {
        self.reference_dir = Some(dir);
        self
    }

    /// Set the directory containing captured NPU hardware outputs.
    ///
    /// Each test's captured output is expected at
    /// `{dir}/{test_name}/output.bin`. When set, the suite performs
    /// three-way cross-validation after each test run.
    pub fn with_npu_output_dir(mut self, dir: PathBuf) -> Self {
        self.npu_output_dir = Some(dir);
        self
    }

    /// Add a single test.
    pub fn add_test(&mut self, test: XclbinTest) {
        self.tests.push(test);
    }

    /// Discover tests from a directory.
    ///
    /// Recursively walks subdirectories looking for xclbin files matching
    /// the mlir-aie build structure. Handles both flat tests
    /// (e.g., `add_one_using_dma/aie.xclbin`) and nested tests inside
    /// parent directories (e.g., `core_dmas/writebd/aie.xclbin`).
    ///
    /// Test names are relative paths from the base directory, so nested
    /// tests get names like `core_dmas/writebd` instead of just `writebd`
    /// (which would collide with `tile_dmas/writebd`).
    pub fn discover(base_path: impl AsRef<Path>) -> Result<Self> {
        let artifacts = super::artifacts::discover_build_artifacts(base_path.as_ref());
        let mut suite = Self::new();

        for art in artifacts {
            let mut test = XclbinTest::from_path(&art.xclbin);
            test.name = art.name;
            test.insts_path = art.insts;
            suite.add_test(test);
        }

        // Sort tests by name for consistent ordering
        suite.tests.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(suite)
    }

    /// Get the number of tests.
    pub fn test_count(&self) -> usize {
        self.tests.len()
    }

    /// Get the tests as a slice.
    pub fn tests(&self) -> &[XclbinTest] {
        &self.tests
    }

    /// Get the tests as a mutable slice (for enriching after discovery).
    pub fn tests_mut(&mut self) -> &mut [XclbinTest] {
        &mut self.tests
    }

    /// Get the collected opcodes.
    pub fn collector(&self) -> &OpcodeCollector {
        &self.collector
    }

    /// Record an unknown opcode.
    pub fn record_unknown(&mut self, opcode: UnknownOpcode, test_name: &str) {
        self.collector.record(opcode, test_name);
    }

    /// Run all tests and return summary.
    pub fn run_all(&mut self) -> SuiteResult {
        let total = self.tests.len();
        let mut passed = 0;
        let mut failed = 0;
        let mut validation_failed = 0;
        let mut expected_fail = 0;
        let mut unexpected_pass = 0;
        let mut skipped = 0;
        let mut unknown = 0;
        let mut timeout = 0;
        let mut load_error = 0;
        let mut hw_validated = 0;
        let mut hw_correct = 0;
        let mut hw_compiler_bug = 0;
        let mut hw_emulator_bug = 0;

        // Clone tests to avoid borrow issues
        let tests: Vec<_> = self.tests.clone();

        for test in tests {
            let (outcome, raw_output, _, _) = self.run_single_inner(&test);

            match &outcome {
                TestOutcome::Pass { .. } => passed += 1,
                TestOutcome::ValidationFail { .. } => validation_failed += 1,
                TestOutcome::ExpectedFail { .. } => expected_fail += 1,
                TestOutcome::UnexpectedPass { .. } => unexpected_pass += 1,
                TestOutcome::Skipped { .. } | TestOutcome::Platform { .. } => skipped += 1,
                TestOutcome::Fail { .. } => failed += 1,
                TestOutcome::UnknownOpcode { details, .. } => {
                    unknown += 1;
                    self.collector.record(details.clone(), &test.name);
                }
                TestOutcome::Timeout { .. } => timeout += 1,
                TestOutcome::LoadError { .. } => load_error += 1,
            }

            // Cross-validate against hardware if npu_output_dir is set
            let hw_validation = self.cross_validate(&test, raw_output.as_deref());
            if let Some(ref hv) = hw_validation {
                if hv.diagnosis != Diagnosis::NoReference {
                    hw_validated += 1;
                    match hv.diagnosis {
                        Diagnosis::Correct => hw_correct += 1,
                        Diagnosis::CompilerBug => hw_compiler_bug += 1,
                        Diagnosis::EmulatorBug => hw_emulator_bug += 1,
                        _ => {}
                    }
                }
            }

            self.results.push((test.name.clone(), outcome, hw_validation));
        }

        SuiteResult {
            total,
            passed,
            failed,
            validation_failed,
            expected_fail,
            unexpected_pass,
            skipped,
            unknown,
            timeout,
            load_error,
            hw_validated,
            hw_correct,
            hw_compiler_bug,
            hw_emulator_bug,
        }
    }

    /// Run a single test.
    pub fn run_single(&self, test: &XclbinTest) -> TestOutcome {
        let (outcome, _, _, _) = self.run_single_inner(test);
        outcome
    }

    /// Run a single test and capture raw emulator output.
    ///
    /// Returns both the test outcome and the raw output bytes read from
    /// host memory at offset 0x1000 (the standard output buffer address).
    /// The output bytes are `None` if the engine could not be created or
    /// the test was skipped.
    pub fn run_single_with_output(&self, test: &XclbinTest) -> (TestOutcome, Option<Vec<u8>>) {
        let (outcome, output, _, _) = self.run_single_inner(test);
        (outcome, output)
    }

    /// Run a single test and return the binary trace buffer.
    ///
    /// Unlike `run_single()` which discards internal details, this exposes
    /// the binary trace buffer from the hardware trace units.
    /// The binary trace is in the same format as real NPU hardware produces,
    /// decodable by mlir-aie's `parse.py`.
    ///
    /// Returns (outcome, raw_output, binary_trace_buffer).
    pub fn run_single_with_trace(&self, test: &XclbinTest) -> (TestOutcome, Option<Vec<u8>>, Option<Vec<u8>>) {
        let (outcome, raw_output, trace_buf, _) = self.run_single_inner(test);
        (outcome, raw_output, trace_buf)
    }

    /// Run a single test with full cross-validation results.
    ///
    /// Returns the test outcome, raw output, and optional hardware
    /// cross-validation diagnosis.
    pub fn run_single_with_hw_validation(&self, test: &XclbinTest) -> (TestOutcome, Option<Vec<u8>>, Option<HardwareValidation>, Vec<String>) {
        let (outcome, raw_output, _, warnings) = self.run_single_inner(test);
        let hw_validation = self.cross_validate(test, raw_output.as_deref());
        (outcome, raw_output, hw_validation, warnings)
    }

    /// Cross-validate emulator output against captured hardware reference.
    ///
    /// Returns `None` if no npu_output_dir is configured or no buffer_spec
    /// is available. Returns `Some(HardwareValidation)` with a `NoReference`
    /// diagnosis if the hardware output file does not exist.
    fn cross_validate(&self, test: &XclbinTest, emu_output: Option<&[u8]>) -> Option<HardwareValidation> {
        let npu_dir = self.npu_output_dir.as_ref()?;
        let spec = test.buffer_spec.as_ref()?;

        // Find output buffer element type for comparison
        let output_type = spec.buffers.iter()
            .find(|b| b.direction == BufferDir::Output)
            .map(|b| b.element_type)
            .unwrap_or(ElementType::I32);

        // Load hardware reference output
        let hw_reference = hardware_comparison::load_hw_reference(npu_dir, &test.name);

        let cv = CrossValidation::compare(
            &test.name,
            emu_output,
            hw_reference.as_deref(),
            output_type,
        );

        Some(HardwareValidation::classify(cv))
    }

    /// Shared implementation for run_single and run_single_with_output.
    ///
    /// Returns (outcome, raw_output, binary_trace_buffer, warnings).
    /// The binary_trace_buffer contains raw packets from hardware trace units
    /// that flowed through the stream switch to host DDR, in the same format
    /// as real NPU hardware. Warnings are collected from the NPU executor.
    fn run_single_inner(&self, test: &XclbinTest) -> (TestOutcome, Option<Vec<u8>>, Option<Vec<u8>>, Vec<String>) {
        // Check if test overrides say to skip this test
        if let Some(ref reason) = test.skip_reason {
            return (TestOutcome::Skipped {
                reason: reason.clone(),
            }, None, None, Vec::new());
        }

        // Load xclbin
        let xclbin = match Xclbin::from_file(&test.xclbin_path) {
            Ok(x) => x,
            Err(e) => {
                return (TestOutcome::LoadError {
                    message: format!("Failed to load xclbin: {}", e),
                }, None, None, Vec::new());
            }
        };

        // Find AIE partition
        let section = match xclbin.find_section(SectionKind::AiePartition) {
            Some(s) => s,
            None => {
                return (TestOutcome::LoadError {
                    message: "No AIE partition in xclbin".to_string(),
                }, None, None, Vec::new());
            }
        };

        // Parse partition
        let partition = match AiePartition::parse(section.data()) {
            Ok(p) => p,
            Err(e) => {
                return (TestOutcome::LoadError {
                    message: format!("Failed to parse AIE partition: {}", e),
                }, None, None, Vec::new());
            }
        };

        // Get PDI and parse CDO
        let pdi = match partition.primary_pdi() {
            Some(p) => p,
            None => {
                return (TestOutcome::LoadError {
                    message: "No primary PDI in partition".to_string(),
                }, None, None, Vec::new());
            }
        };

        let cdo_offset = match find_cdo_offset(pdi.pdi_image) {
            Some(o) => o,
            None => {
                return (TestOutcome::LoadError {
                    message: "No CDO found in PDI".to_string(),
                }, None, None, Vec::new());
            }
        };

        let cdo = match Cdo::parse(&pdi.pdi_image[cdo_offset..]) {
            Ok(c) => c,
            Err(e) => {
                return (TestOutcome::LoadError {
                    message: format!("Failed to parse CDO: {}", e),
                }, None, None, Vec::new());
            }
        };

        // Create engine and apply CDO
        let mut engine = InterpreterEngine::new_npu1();
        engine.set_stall_threshold(self.stall_threshold);
        if let Err(e) = engine.device_mut().apply_cdo(&cdo) {
            return (TestOutcome::LoadError {
                message: format!("Failed to apply CDO: {}", e),
            }, None, None, Vec::new());
        }

        // Populate host memory - use buffer spec if available, else defaults.
        // host_buffers maps DDR patch arg_idx to actual host memory addresses.
        let (input_values, host_buffers) = if let Some(ref spec) = test.buffer_spec {
            self.setup_input_from_buffer_spec(&mut engine, spec)
        } else {
            self.setup_default_input(&mut engine);
            // Default layout: input(0x0, 4KB), middle(0x1000, 256B), output(0x2000, 4KB)
            let default_buffers = vec![
                HostBuffer { address: 0x0000, size: 4096 },
                HostBuffer { address: 0x1000, size: 256 },
                HostBuffer { address: 0x2000, size: 4096 },
            ];
            (None, default_buffers)
        };

        // Output address: find the first Output buffer's BO index.
        // With multiple output buffers (e.g. data + trace), we want the
        // primary data output, not the trace buffer.
        let output_addr = if let Some(ref spec) = test.buffer_spec {
            spec.buffers.iter()
                .find(|b| b.direction == BufferDir::Output)
                .map(|b| {
                    let bo_idx = b.group_id.saturating_sub(3) as usize;
                    host_buffers.get(bo_idx).map(|hb| hb.address).unwrap_or(0x1000u64)
                })
                .unwrap_or_else(|| host_buffers.last().map(|b| b.address).unwrap_or(0x1000u64))
        } else {
            // Default layout: output is the last buffer
            host_buffers.last().map(|b| b.address).unwrap_or(0x1000u64)
        };

        // Execute NPU instructions if insts.bin exists.
        // These configure and trigger shim DMA to move data to/from cores.
        // The NpuExecutor must survive until run_engine() so its collected
        // sync conditions can be checked as a completion signal.
        let mut npu_executor: Option<NpuExecutor> = None;
        let mut test_warnings: Vec<String> = Vec::new();

        if let Some(insts_path) = test.find_insts_file() {
            let insts_data = match std::fs::read(&insts_path) {
                Ok(d) => d,
                Err(e) => {
                    let msg = format!("Failed to read {}: {}",
                        insts_path.file_name().unwrap_or_default().to_string_lossy(), e);
                    log::warn!("[{}] {}", test.name, msg);
                    test_warnings.push(msg);
                    // Continue without NPU instructions - some tests may not need them
                    Vec::new()
                }
            };

            if !insts_data.is_empty() {
                match NpuInstructionStream::parse(&insts_data) {
                    Ok(stream) => {
                        log::debug!("Loaded {} NPU instructions from {:?}",
                            stream.len(), insts_path);

                        let mut executor = NpuExecutor::new();

                        // Set up host buffers from the buffer spec layout.
                        // These addresses must match the actual host memory allocation
                        // so DDR patches write correct addresses into shim DMA BDs.
                        executor.set_host_buffers(host_buffers.clone());

                        // Load instructions for interleaved execution.
                        // Instructions will be executed one-per-cycle in run_engine()
                        // alongside full system stepping, so DMA queue backpressure
                        // works correctly (the full array runs while waiting for
                        // queues to drain).
                        executor.load(&stream);

                        npu_executor = Some(executor);
                    }
                    Err(e) => {
                        let msg = format!("Failed to parse {}: {}",
                            insts_path.file_name().unwrap_or_default().to_string_lossy(), e);
                        log::warn!("[{}] {}", test.name, msg);
                        test_warnings.push(msg);
                    }
                }
            }
        }

        // Load ELF files if project directory exists
        // IMPORTANT: Load ELFs BEFORE sync_cores_from_device() so the engine
        // sees the cores as enabled after loading program code
        let elf_files = test.find_elf_files();
        log::debug!("Found {} ELF files for test {}", elf_files.len(), test.name);
        for (col, row, path) in &elf_files {
            let data = match std::fs::read(path) {
                Ok(d) => d,
                Err(e) => {
                    return (TestOutcome::LoadError {
                        message: format!("Failed to read ELF {:?}: {}", path, e),
                    }, None, None, Vec::new());
                }
            };

            if let Err(e) = engine.load_elf_bytes(*col as usize, *row as usize, &data) {
                return (TestOutcome::LoadError {
                    message: format!("Failed to load ELF into ({},{}): {}", col, row, e),
                }, None, None, Vec::new());
            }
        }

        // Sync core enabled state from device tiles to engine
        // Called AFTER loading ELFs so the engine sees the loaded cores
        engine.sync_cores_from_device();

        let enabled = engine.enabled_cores();
        if enabled == 0 && elf_files.is_empty() && npu_executor.is_none() {
            // No cores, no ELFs, and no NPU instructions -- truly nothing to
            // execute. DMA-only tests (which have insts.bin) still run: the
            // NPU instructions configure and enqueue DMA transfers, and the
            // engine steps DMA engines + stream switches until completion.
            return (TestOutcome::Skipped {
                reason: "No core code and no NPU instructions to execute".to_string(),
            }, None, None, Vec::new());
        }

        // Run until halt, sync completion, error, or timeout
        let outcome = self.run_engine(&mut engine, test, input_values, npu_executor.as_mut(), output_addr);

        // Flush trace units and route final binary trace packets to host DDR.
        engine.flush_trace_to_host();

        // Capture raw output from host memory at the output buffer address
        let raw_output = self.capture_output(&engine, test, output_addr);

        // Capture binary trace buffer from host memory if a trace buffer was allocated.
        // The NPU executor allocates trace buffers for DDR patches referencing arg_idx
        // beyond the known host buffers (typically arg_idx=3 for trace data).
        let binary_trace_buf = npu_executor.as_ref().and_then(|exec| {
            // Trace buffer is typically the last host buffer (beyond the user-specified ones)
            let bufs = exec.host_buffers();
            if bufs.len() > 3 {
                let tb = &bufs[3];
                let mut data = vec![0u8; tb.size];
                engine.host_memory().read_bytes(tb.address, &mut data);
                // Only return if there's actual data (not all zeros)
                if data.iter().any(|&b| b != 0) {
                    log::info!(
                        "Captured {} bytes of binary trace data from host memory at 0x{:X}",
                        data.len(), tb.address
                    );
                    Some(data)
                } else {
                    None
                }
            } else {
                None
            }
        });

        // Remap outcomes for expected-fail tests, preserving the actual failure details
        let final_outcome = if let Some(ref reason) = test.expected_fail_reason {
            match outcome {
                // Test failed as expected -- wrap with details for diagnostics
                TestOutcome::ValidationFail { cycles, correct, total, ref first_mismatch } => {
                    let actual = if let Some((idx, expected, got)) = first_mismatch {
                        format!("ValidationFail: {}/{} correct, first mismatch at [{}]: expected {}, got {}",
                            correct, total, idx, expected, got)
                    } else {
                        format!("ValidationFail: {}/{} correct", correct, total)
                    };
                    TestOutcome::ExpectedFail {
                        cycles,
                        reason: reason.clone(),
                        actual,
                    }
                }
                TestOutcome::Fail { ref message, cycles } => {
                    TestOutcome::ExpectedFail {
                        cycles,
                        reason: reason.clone(),
                        actual: format!("Fail: {}", message),
                    }
                }
                TestOutcome::UnknownOpcode { ref details, cycles } => {
                    let mnemonic = details.mnemonic.as_deref().unwrap_or("unknown");
                    TestOutcome::ExpectedFail {
                        cycles,
                        reason: reason.clone(),
                        actual: format!("UnknownOpcode: {:?} 0x{:04X} '{}' at PC {}",
                            details.slot, details.opcode, mnemonic, details.pc),
                    }
                }
                TestOutcome::Timeout { cycles } => {
                    TestOutcome::ExpectedFail {
                        cycles,
                        reason: reason.clone(),
                        actual: "Timeout".to_string(),
                    }
                }
                // Test passed when we expected failure -- override needs updating
                TestOutcome::Pass { cycles, correct, total } => {
                    TestOutcome::UnexpectedPass {
                        cycles,
                        correct: correct.unwrap_or(0),
                        total: total.unwrap_or(0),
                    }
                }
                // Other outcomes pass through unchanged
                other => other,
            }
        } else {
            outcome
        };

        // Merge test-level warnings with NPU executor warnings
        if let Some(ref exec) = npu_executor {
            test_warnings.extend(exec.warnings().iter().cloned());
        }

        (final_outcome, raw_output, binary_trace_buf, test_warnings)
    }

    /// Capture raw output bytes from host memory.
    ///
    /// Reads from the given output buffer address.
    /// The size is determined by the buffer spec output definition,
    /// or defaults to 4KB if no buffer spec is available.
    fn capture_output(&self, engine: &InterpreterEngine, test: &XclbinTest, output_addr: u64) -> Option<Vec<u8>> {
        let output_size = if let Some(ref spec) = test.buffer_spec {
            spec.buffers.iter()
                .find(|b| b.direction == BufferDir::Output)
                .map(|b| b.size_elements * b.element_type.byte_size())
                .unwrap_or(4096)
        } else {
            4096
        };

        let mut output_data = vec![0u8; output_size];
        engine.host_memory().read_bytes(output_addr, &mut output_data);
        Some(output_data)
    }

    /// Setup host memory from BufferSpec buffer definitions.
    ///
    /// Returns (input_values, host_buffers):
    /// - input_values: map of buffer name -> values (for debugging/display)
    /// - host_buffers: ordered list of HostBuffer entries matching DDR patch arg_idx
    ///
    /// Host buffers are ordered by group_id to match runtime_sequence arguments:
    /// DDR patches reference bo_index = group_id - 3.
    fn setup_input_from_buffer_spec(
        &self,
        engine: &mut InterpreterEngine,
        spec: &BufferSpec,
    ) -> (Option<HashMap<String, Vec<i64>>>, Vec<HostBuffer>) {
        let host_mem = engine.host_memory_mut();
        let mut inputs = HashMap::new();

        // Determine buffer object (BO) count from group_ids.
        // The MLIR_AIE kernel has: arg3=bo0(gid=3), arg4=bo1(gid=4), ..., arg7=bo4(gid=7).
        // DDR patches reference bo_index = group_id - 3.
        let max_gid = spec.buffers.iter()
            .map(|b| b.group_id)
            .max()
            .unwrap_or(5);
        let num_bos = (max_gid.saturating_sub(3) + 1) as usize;

        // Allocate buffers sequentially in host memory to avoid overlaps.
        let mut next_addr: u64 = 0x0;
        let mut buffer_addrs: Vec<(u64, usize)> = vec![(0, 0); num_bos]; // (addr, size) indexed by bo_idx

        // Allocate input buffers first
        for buf in &spec.buffers {
            if buf.direction == BufferDir::Output {
                continue;
            }
            let bo_idx = buf.group_id.saturating_sub(3) as usize;
            if bo_idx >= num_bos {
                continue;
            }

            let input_data = generate_input_data(buf);
            let data_size = input_data.len();
            let alloc_size = data_size.max(4096);

            let _ = host_mem.allocate_region(&buf.name, next_addr, alloc_size);
            host_mem.write_bytes(next_addr, &input_data);
            inputs.insert(buf.name.clone(), read_values(&input_data, buf.element_type));
            buffer_addrs[bo_idx] = (next_addr, data_size);
            next_addr += alloc_size as u64;
        }

        // If no inputs were placed, allocate a default input region
        if next_addr == 0 {
            let _ = host_mem.allocate_region("input", 0x0, 4096);
            buffer_addrs[0] = (0, 4096);
            next_addr = 4096;
        }

        // Allocate output buffers after all inputs, each at a unique address.
        next_addr = next_addr.max(0x1000);
        for buf in &spec.buffers {
            if buf.direction != BufferDir::Output {
                continue;
            }
            let bo_idx = buf.group_id.saturating_sub(3) as usize;
            if bo_idx >= num_bos {
                continue;
            }
            let output_size = (buf.size_elements * buf.element_type.byte_size()).max(4096);
            let _ = host_mem.allocate_region(&buf.name, next_addr, output_size);
            buffer_addrs[bo_idx] = (next_addr, output_size);
            next_addr += output_size as u64;
        }

        // Fill any gaps (unused BO indices) with dummy regions
        for i in 0..num_bos {
            if buffer_addrs[i] == (0, 0) {
                let _ = host_mem.allocate_region(&format!("unused_bo{}", i), next_addr, 256);
                buffer_addrs[i] = (next_addr, 256);
                next_addr += 256;
            }
        }

        // Build host buffer list indexed by BO index (= DDR patch arg_idx)
        let host_buffers: Vec<HostBuffer> = buffer_addrs
            .iter()
            .map(|(addr, size)| HostBuffer { address: *addr, size: *size })
            .collect();

        (Some(inputs), host_buffers)
    }

    /// Validate output against hardware reference capture.
    ///
    /// Loads the reference from `{reference_dir}/{test_name}/output.bin`
    /// and compares element-by-element against the emulator's output.
    fn validate_output(&self, engine: &InterpreterEngine, test: &XclbinTest, output_addr: u64) -> Option<(usize, usize, Option<(usize, i64, i64)>)> {
        let reference_dir = self.reference_dir.as_ref()?;
        let spec = test.buffer_spec.as_ref()?;

        // Find output buffer metadata
        let output_buf = spec.buffers.iter()
            .find(|b| b.direction == BufferDir::Output)?;
        let elem_type = output_buf.element_type;

        // Load hardware reference
        let hw_ref = hardware_comparison::load_hw_reference(reference_dir, &test.name)?;

        // Read emulator output from host memory
        let output_size = output_buf.size_elements * elem_type.byte_size();
        let mut output_data = vec![0u8; output_size];
        engine.host_memory().read_bytes(output_addr, &mut output_data);

        // Compare using hardware_comparison primitives
        let result = hardware_comparison::compare_buffers(&output_data, &hw_ref, elem_type);

        Some((result.matching, result.total, result.first_mismatch))
    }

    /// Build a completion outcome by validating output against hardware reference.
    ///
    /// Used by both the Halted and syncs-satisfied exit paths to avoid
    /// duplicating validation logic.
    fn make_completion_outcome(
        &self,
        engine: &InterpreterEngine,
        test: &XclbinTest,
        _input_values: Option<HashMap<String, Vec<i64>>>,
        cycles: u64,
        output_addr: u64,
    ) -> TestOutcome {
        if test.buffer_spec.is_some() && self.reference_dir.is_some() {
            if let Some((correct, total, first_mismatch)) =
                self.validate_output(engine, test, output_addr)
            {
                if correct == total && total > 0 {
                    return TestOutcome::Pass {
                        cycles,
                        correct: Some(correct),
                        total: Some(total),
                    };
                } else {
                    return TestOutcome::ValidationFail {
                        cycles,
                        correct,
                        total,
                        first_mismatch,
                    };
                }
            }
        }
        // No buffer spec, no reference, or validation could not run
        TestOutcome::Pass { cycles, correct: None, total: None }
    }

    /// Run an engine until completion.
    ///
    /// Exits when any of the following occur:
    /// 1. All cores halt (EngineStatus::Halted)
    /// 2. DMA sync conditions are satisfied (NpuExecutor::syncs_satisfied)
    /// 3. A core hits an error (EngineStatus::Error)
    /// 4. Quiescence detected (all subsystems terminal for 100 consecutive cycles)
    /// 5. Absolute cycle limit reached (safety net timeout)
    ///
    /// Condition 2 is the primary completion signal for real AIE2 workloads:
    /// kernels are infinite loops that process data as DMA feeds them, and the
    /// host determines completion by watching DMA channel status.
    ///
    /// Condition 4 detects deadlock by checking whether the entire system has
    /// settled: NPU executor done, all cores waiting/halted, all DMA terminal,
    /// and no data in flight. Sustained for 100 cycles = deadlock. This replaces
    /// the old TDR (DMA-bytes-only progress check) which failed to catch stalls
    /// where DMA bytes were moving but no completion was possible.
    fn run_engine(
        &self,
        engine: &mut InterpreterEngine,
        test: &XclbinTest,
        input_values: Option<HashMap<String, Vec<i64>>>,
        mut npu_executor: Option<&mut NpuExecutor>,
        output_addr: u64,
    ) -> TestOutcome {
        use super::quiescence::{
            QuiescenceDetector, QuiescenceStatus,
            StallDetector, StallStatus,
        };

        const QUIESCENCE_CYCLES: u64 = 100;

        let mut cycles = 0u64;
        let mut quiescence = QuiescenceDetector::new(QUIESCENCE_CYCLES);
        let mut stall = StallDetector::new(self.stall_threshold);
        // 0 means no hard limit -- rely on quiescence/stall detection only
        let cycle_limit = if self.max_cycles == 0 { u64::MAX } else { self.max_cycles };

        while cycles < cycle_limit {
            // Advance NPU instruction execution before engine step.
            // try_advance executes one instruction per cycle; when a DMA
            // queue is full it returns Blocked and the engine stepping
            // below drains the queue naturally via full system simulation.
            if let Some(executor) = npu_executor.as_mut() {
                let (device, host_mem) = engine.device_and_host_memory();
                if let crate::npu::AdvanceResult::Error(msg) = executor.try_advance(device, host_mem) {
                    log::error!("NPU executor fatal: {}", msg);
                    return TestOutcome::Fail { message: msg, cycles };
                }
            }

            engine.step();
            cycles = engine.total_cycles();

            match engine.status() {
                EngineStatus::Halted => {
                    // For DMA-only tests (no cores loaded), the engine halts
                    // immediately because no cores are enabled. But the NPU
                    // executor may still be issuing instructions that configure
                    // and trigger DMA. Keep running until the executor finishes
                    // and DMA syncs are satisfied.
                    let executor_pending = if let Some(executor) = npu_executor.as_mut() {
                        !executor.is_done() || !executor.syncs_satisfied(engine.device())
                    } else {
                        false
                    };
                    if executor_pending {
                        engine.force_running();
                    } else {
                        return self.make_completion_outcome(engine, test, input_values, cycles, output_addr);
                    }
                }
                EngineStatus::Error => {
                    if let Some(details) = self.extract_error_details(engine) {
                        return details;
                    }
                    return TestOutcome::Fail {
                        message: "Unknown execution error".to_string(),
                        cycles,
                    };
                }
                EngineStatus::Stalled => {
                    // Stall detection fired (adaptive timeout). Treat as a
                    // timeout -- the test suite will report failure with the
                    // cycle count so the caller can diagnose.
                    return TestOutcome::Fail {
                        message: format!(
                            "Stall detected: no progress for threshold cycles (cycle {})",
                            cycles
                        ),
                        cycles,
                    };
                }
                EngineStatus::Running | EngineStatus::Ready | EngineStatus::Paused => {
                    // Continue running
                }
            }

            // Check if DMA syncs are satisfied (NPU instruction sequence complete).
            // Only check after all NPU instructions have been executed -- Sync
            // instructions are interleaved one-per-cycle, so pending_syncs is
            // not fully populated until the executor reaches Done state.
            if let Some(executor) = npu_executor.as_mut() {
                if executor.is_done() && executor.syncs_satisfied(engine.device()) {
                    log::info!("DMA syncs satisfied after {} cycles for test {}", cycles, test.name);
                    return self.make_completion_outcome(engine, test, input_values, cycles, output_addr);
                }
            }

            // Quiescence detection: all subsystems terminal for 100 cycles.
            match quiescence.check(engine, npu_executor.as_deref()) {
                QuiescenceStatus::Quiescent(diagnosis) => {
                    log::info!(
                        "Quiescence detected after {} cycles in test {}: {}",
                        cycles, test.name, diagnosis,
                    );
                    return TestOutcome::Timeout { cycles };
                }
                QuiescenceStatus::Running => {}
            }

            // Stall detection: cores running but no DMA byte progress for
            // STALL_CYCLES while syncs remain unsatisfied. Catches livelocks
            // where quiescence can't fire because cores are still Running.
            match stall.check(engine, npu_executor.as_deref()) {
                StallStatus::Stalled(diagnosis) => {
                    let dma_bytes = engine.device().array.total_dma_bytes_transferred();
                    log::warn!(
                        "DMA stall after {} cycles in test {} ({} bytes transferred): {}",
                        cycles, test.name, dma_bytes, diagnosis,
                    );
                    return TestOutcome::Timeout { cycles };
                }
                StallStatus::Progressing => {}
            }
        }

        // Hard cycle limit reached -- produce diagnostic so we know why
        // neither quiescence nor stall detection fired.
        let diagnosis = QuiescenceDetector::diagnose(engine, npu_executor.as_deref());
        let dma_bytes = engine.device().array.total_dma_bytes_transferred();
        log::warn!(
            "Hard cycle limit ({}) reached in test {} ({} bytes transferred): {}",
            cycle_limit, test.name, dma_bytes, diagnosis,
        );
        TestOutcome::Timeout { cycles }
    }

    /// Try to extract error details from a failed engine.
    fn extract_error_details(&self, engine: &InterpreterEngine) -> Option<TestOutcome> {
        let cycles = engine.total_cycles();

        // Find the core that failed by checking all compute tiles.
        let cols = engine.device().cols();
        let rows = engine.device().rows();
        for col in 0..cols {
            for row in 2..rows {
                // Get PC from context
                let pc = engine.core_context(col, row)
                    .map(|ctx| ctx.pc())
                    .unwrap_or(0);

                // Check the last bundle for unknown operations (semantic: None)
                if let Some(bundle) = engine.core_last_bundle(col, row) {
                    for op in bundle.active_slots() {
                        if op.semantic.is_none() {
                            return Some(TestOutcome::UnknownOpcode {
                                details: UnknownOpcode {
                                    slot: op.slot,
                                    opcode: op.raw_opcode.unwrap_or(0),
                                    pc,
                                    tile: (col as u8, row as u8),
                                    mnemonic: None,
                                },
                                cycles,
                            });
                        }
                    }
                }
            }
        }

        None
    }

    /// Get all test results.
    pub fn results(&self) -> &[(String, TestOutcome, Option<HardwareValidation>)] {
        &self.results
    }

    /// Generate a summary report.
    pub fn summary_report(&self, result: &SuiteResult) -> String {
        let mut report = String::new();

        report.push_str("=== XCLBIN Test Suite Results ===\n\n");
        report.push_str(&format!(
            "Total: {}, Passed: {}, Validation Failed: {}, Expected Fail: {}, Unexpected Pass: {}, Skipped: {}, Failed: {}, Unknown Opcodes: {}, Timeout: {}, Load Error: {}\n\n",
            result.total, result.passed, result.validation_failed,
            result.expected_fail, result.unexpected_pass, result.skipped,
            result.failed, result.unknown, result.timeout, result.load_error
        ));

        // List failed tests
        if result.failed > 0 || result.validation_failed > 0 || result.unknown > 0 || result.timeout > 0 {
            report.push_str("--- Failed Tests ---\n");
            for (name, outcome, _) in &self.results {
                match outcome {
                    TestOutcome::ValidationFail { cycles, correct, total, first_mismatch } => {
                        report.push_str(&format!("{}: VALIDATION FAIL after {} cycles ({}/{} correct)\n", name, cycles, correct, total));
                        if let Some((idx, expected, actual)) = first_mismatch {
                            report.push_str(&format!("  First mismatch at [{}]: expected {}, got {}\n", idx, expected, actual));
                        }
                    }
                    TestOutcome::Fail { message, cycles } => {
                        report.push_str(&format!("{}: FAIL after {} cycles - {}\n", name, cycles, message));
                    }
                    TestOutcome::UnknownOpcode { details, cycles } => {
                        report.push_str(&format!(
                            "{}: UNKNOWN OPCODE after {} cycles - slot {:?}, opcode 0x{:08X} at PC 0x{:04X}\n",
                            name, cycles, details.slot, details.opcode, details.pc
                        ));
                    }
                    TestOutcome::Timeout { cycles } => {
                        report.push_str(&format!("{}: TIMEOUT after {} cycles\n", name, cycles));
                    }
                    TestOutcome::LoadError { message } => {
                        report.push_str(&format!("{}: LOAD ERROR - {}\n", name, message));
                    }
                    _ => {}
                }
            }
            report.push('\n');
        }

        // List expected-fail tests
        if result.expected_fail > 0 {
            report.push_str("--- Expected Failures (known-broken) ---\n");
            for (name, outcome, _) in &self.results {
                if let TestOutcome::ExpectedFail { cycles, reason, actual } = outcome {
                    report.push_str(&format!("{}: EXPECTED FAIL ({} cycles)\n", name, cycles));
                    if !reason.is_empty() {
                        report.push_str(&format!("  reason: {}\n", reason));
                    }
                    report.push_str(&format!("  actual: {}\n", actual));
                }
            }
            report.push('\n');
        }

        // List unexpected passes (known-broken tests that suddenly work)
        if result.unexpected_pass > 0 {
            report.push_str("--- Unexpected Passes (update test_overrides.toml!) ---\n");
            for (name, outcome, _) in &self.results {
                if let TestOutcome::UnexpectedPass { cycles, correct, total } = outcome {
                    report.push_str(&format!("{}: UNEXPECTED PASS ({} cycles, {}/{} correct) - remove from [expected_fail] in test_overrides.toml\n",
                        name, cycles, correct, total));
                }
            }
            report.push('\n');
        }

        // List passed tests
        if result.passed > 0 {
            report.push_str("--- Passed Tests ---\n");
            for (name, outcome, _) in &self.results {
                if let TestOutcome::Pass { cycles, correct, total } = outcome {
                    if let (Some(c), Some(t)) = (correct, total) {
                        report.push_str(&format!("{}: PASS ({} cycles, {}/{} validated)\n", name, cycles, c, t));
                    } else {
                        report.push_str(&format!("{}: PASS ({} cycles)\n", name, cycles));
                    }
                }
            }
            report.push('\n');
        }

        // List skipped tests
        if result.skipped > 0 {
            report.push_str("--- Skipped Tests ---\n");
            for (name, outcome, _) in &self.results {
                if let TestOutcome::Skipped { reason } = outcome {
                    if reason.is_empty() {
                        report.push_str(&format!("{}: SKIPPED\n", name));
                    } else {
                        report.push_str(&format!("{}: SKIPPED - {}\n", name, reason));
                    }
                }
            }
            report.push('\n');
        }

        // Hardware cross-validation summary
        if result.hw_validated > 0 {
            report.push_str("--- Hardware Cross-Validation ---\n");
            report.push_str(&format!(
                "Validated: {}, Correct: {}, Compiler Bug: {}, Emulator Bug: {}\n",
                result.hw_validated, result.hw_correct,
                result.hw_compiler_bug, result.hw_emulator_bug
            ));

            // List tests with non-trivial diagnoses
            for (name, _, hw) in &self.results {
                if let Some(hv) = hw {
                    match hv.diagnosis {
                        Diagnosis::CompilerBug | Diagnosis::EmulatorBug | Diagnosis::BothBroken => {
                            report.push_str(&format!("  {}: {}\n", name, hv.diagnosis));
                        }
                        _ => {}
                    }
                }
            }
            report.push('\n');
        }

        // Add opcode priority list if there are unknowns
        if self.collector.has_unknowns() {
            report.push_str(&self.collector.priority_report());
        }

        report
    }

    /// Setup default input data in host memory.
    ///
    /// Populates host memory with sequential integer data that most
    /// mlir-aie test kernels can process. This allows tests to make
    /// progress instead of timing out waiting for data.
    ///
    /// Memory layout (matching mlir-aie test patterns):
    /// - 0x0000: Input buffer (4KB - up to 1024 i32s)
    /// - 0x0100: Middle/unused buffer (256 bytes)
    /// - 0x1000: Output buffer (4KB - up to 1024 i32s)
    fn setup_default_input(&self, engine: &mut InterpreterEngine) {
        let host_mem = engine.host_memory_mut();

        // Allocate input region (4KB at address 0)
        let _ = host_mem.allocate_region("input", 0x0, 4096);

        // Write sequential i32 values: [1, 2, 3, ..., 1024]
        // This matches the common test pattern in mlir-aie examples
        let input: Vec<u32> = (1..=1024).collect();
        host_mem.write_slice(0x0, &input);

        // Allocate middle buffer after input (avoids overlap with input data)
        let _ = host_mem.allocate_region("middle", 0x1000, 256);

        // Allocate output region (4KB at 0x2000, after input + middle)
        let _ = host_mem.allocate_region("output", 0x2000, 4096);
    }

}

impl Default for XclbinSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of a test suite run.
#[derive(Debug, Clone)]
pub struct SuiteResult {
    /// Total number of tests.
    pub total: usize,
    /// Number of passed tests.
    pub passed: usize,
    /// Number of failed tests.
    pub failed: usize,
    /// Number of tests that failed output validation.
    pub validation_failed: usize,
    /// Number of known-broken tests that failed as expected.
    pub expected_fail: usize,
    /// Number of known-broken tests that unexpectedly passed.
    pub unexpected_pass: usize,
    /// Number of skipped tests.
    pub skipped: usize,
    /// Number of tests that hit unknown opcodes.
    pub unknown: usize,
    /// Number of tests that timed out.
    pub timeout: usize,
    /// Number of tests that failed to load.
    pub load_error: usize,
    /// Number of tests with hardware cross-validation data.
    pub hw_validated: usize,
    /// Tests where emulator matches hardware reference.
    pub hw_correct: usize,
    /// Tests where emulator matches hardware but both differ from expected.
    pub hw_compiler_bug: usize,
    /// Tests where emulator diverges from hardware reference.
    pub hw_emulator_bug: usize,
}

impl SuiteResult {
    /// Get pass rate as a percentage (excludes skipped tests from the denominator).
    pub fn pass_rate(&self) -> f64 {
        let effective_total = self.total - self.skipped;
        if effective_total == 0 {
            0.0
        } else {
            (self.passed as f64 / effective_total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_empty_dir() {
        let temp_dir = std::env::temp_dir().join("xdna_emu_test_empty");
        std::fs::create_dir_all(&temp_dir).ok();

        let suite = XclbinSuite::discover(&temp_dir);
        assert!(suite.is_ok());
        assert_eq!(suite.unwrap().test_count(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_xclbin_test_from_path() {
        let test = XclbinTest::from_path("/some/test_dir/aie.xclbin");
        assert_eq!(test.name, "test_dir");
    }

    #[test]
    fn test_suite_result_pass_rate() {
        let result = SuiteResult {
            total: 10,
            passed: 7,
            failed: 1,
            validation_failed: 0,
            expected_fail: 0,
            unexpected_pass: 0,
            skipped: 0,
            unknown: 1,
            timeout: 1,
            load_error: 0,
            hw_validated: 0,
            hw_correct: 0,
            hw_compiler_bug: 0,
            hw_emulator_bug: 0,
        };
        assert!((result.pass_rate() - 70.0).abs() < 0.01);
    }

    #[test]
    fn test_suite_result_pass_rate_with_skipped() {
        let result = SuiteResult {
            total: 10,
            passed: 7,
            failed: 1,
            validation_failed: 0,
            expected_fail: 0,
            unexpected_pass: 0,
            skipped: 2,
            unknown: 0,
            timeout: 0,
            load_error: 0,
            hw_validated: 0,
            hw_correct: 0,
            hw_compiler_bug: 0,
            hw_emulator_bug: 0,
        };
        // 7 passed out of 8 non-skipped = 87.5%
        assert!((result.pass_rate() - 87.5).abs() < 0.01);
    }
}
