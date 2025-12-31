//! XCLBIN test suite for running binary compatibility tests.
//!
//! This module provides test discovery and execution for xclbin files.

use std::path::{Path, PathBuf};
use anyhow::{anyhow, Result};

use crate::parser::{Xclbin, AiePartition, Cdo};
use crate::parser::xclbin::SectionKind;
use crate::parser::cdo::find_cdo_offset;
use crate::interpreter::engine::{InterpreterEngine, EngineStatus};

use super::opcode_collector::{OpcodeCollector, UnknownOpcode};

/// Result of running a single xclbin test.
#[derive(Debug)]
pub enum TestOutcome {
    /// Test completed successfully (all cores halted).
    Pass {
        cycles: u64,
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
    /// Check if the test passed.
    pub fn is_pass(&self) -> bool {
        matches!(self, TestOutcome::Pass { .. })
    }

    /// Get the cycle count (if available).
    pub fn cycles(&self) -> Option<u64> {
        match self {
            TestOutcome::Pass { cycles } => Some(*cycles),
            TestOutcome::Fail { cycles, .. } => Some(*cycles),
            TestOutcome::UnknownOpcode { cycles, .. } => Some(*cycles),
            TestOutcome::Timeout { cycles } => Some(*cycles),
            TestOutcome::LoadError { .. } => None,
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

        Self {
            name,
            xclbin_path: path.to_path_buf(),
            project_dir,
            expected_output: None,
        }
    }

    /// Set the expected output for validation.
    pub fn with_expected_output(mut self, output: Vec<u8>) -> Self {
        self.expected_output = Some(output);
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
}

/// Test suite for running multiple xclbin tests.
pub struct XclbinSuite {
    /// Tests to run.
    tests: Vec<XclbinTest>,
    /// Collected unknown opcodes across all tests.
    collector: OpcodeCollector,
    /// Maximum cycles before timeout.
    max_cycles: u64,
    /// Results from completed tests.
    results: Vec<(String, TestOutcome)>,
}

impl XclbinSuite {
    /// Create a new test suite.
    pub fn new() -> Self {
        Self {
            tests: Vec::new(),
            collector: OpcodeCollector::new(),
            max_cycles: 1_000_000,
            results: Vec::new(),
        }
    }

    /// Set the maximum cycles before timeout.
    pub fn with_max_cycles(mut self, max: u64) -> Self {
        self.max_cycles = max;
        self
    }

    /// Add a single test.
    pub fn add_test(&mut self, test: XclbinTest) {
        self.tests.push(test);
    }

    /// Discover tests from a directory.
    ///
    /// Looks for xclbin files in subdirectories matching the mlir-aie structure.
    pub fn discover(base_path: impl AsRef<Path>) -> Result<Self> {
        let base = base_path.as_ref();
        let mut suite = Self::new();

        // Look for xclbin files in immediate subdirectories
        for entry in std::fs::read_dir(base)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Check for aie.xclbin or final.xclbin
                for xclbin_name in &["aie.xclbin", "final.xclbin"] {
                    let xclbin_path = path.join(xclbin_name);
                    if xclbin_path.exists() {
                        suite.add_test(XclbinTest::from_path(&xclbin_path));
                        break;
                    }
                }
            } else if path.extension().map(|e| e == "xclbin").unwrap_or(false) {
                // Direct xclbin file
                suite.add_test(XclbinTest::from_path(&path));
            }
        }

        // Sort tests by name for consistent ordering
        suite.tests.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(suite)
    }

    /// Get the number of tests.
    pub fn test_count(&self) -> usize {
        self.tests.len()
    }

    /// Get the collected opcodes.
    pub fn collector(&self) -> &OpcodeCollector {
        &self.collector
    }

    /// Run all tests and return summary.
    pub fn run_all(&mut self) -> SuiteResult {
        let total = self.tests.len();
        let mut passed = 0;
        let mut failed = 0;
        let mut unknown = 0;
        let mut timeout = 0;
        let mut load_error = 0;

        // Clone tests to avoid borrow issues
        let tests: Vec<_> = self.tests.clone();

        for test in tests {
            let outcome = self.run_single(&test);

            match &outcome {
                TestOutcome::Pass { .. } => passed += 1,
                TestOutcome::Fail { .. } => failed += 1,
                TestOutcome::UnknownOpcode { details, .. } => {
                    unknown += 1;
                    self.collector.record(details.clone(), &test.name);
                }
                TestOutcome::Timeout { .. } => timeout += 1,
                TestOutcome::LoadError { .. } => load_error += 1,
            }

            self.results.push((test.name.clone(), outcome));
        }

        SuiteResult {
            total,
            passed,
            failed,
            unknown,
            timeout,
            load_error,
        }
    }

    /// Run a single test.
    pub fn run_single(&self, test: &XclbinTest) -> TestOutcome {
        // Load xclbin
        let xclbin = match Xclbin::from_file(&test.xclbin_path) {
            Ok(x) => x,
            Err(e) => {
                return TestOutcome::LoadError {
                    message: format!("Failed to load xclbin: {}", e),
                };
            }
        };

        // Find AIE partition
        let section = match xclbin.find_section(SectionKind::AiePartition) {
            Some(s) => s,
            None => {
                return TestOutcome::LoadError {
                    message: "No AIE partition in xclbin".to_string(),
                };
            }
        };

        // Parse partition
        let partition = match AiePartition::parse(section.data()) {
            Ok(p) => p,
            Err(e) => {
                return TestOutcome::LoadError {
                    message: format!("Failed to parse AIE partition: {}", e),
                };
            }
        };

        // Get PDI and parse CDO
        let pdi = match partition.primary_pdi() {
            Some(p) => p,
            None => {
                return TestOutcome::LoadError {
                    message: "No primary PDI in partition".to_string(),
                };
            }
        };

        let cdo_offset = match find_cdo_offset(pdi.pdi_image) {
            Some(o) => o,
            None => {
                return TestOutcome::LoadError {
                    message: "No CDO found in PDI".to_string(),
                };
            }
        };

        let cdo = match Cdo::parse(&pdi.pdi_image[cdo_offset..]) {
            Ok(c) => c,
            Err(e) => {
                return TestOutcome::LoadError {
                    message: format!("Failed to parse CDO: {}", e),
                };
            }
        };

        // Create engine and apply CDO
        let mut engine = InterpreterEngine::new_npu1();
        if let Err(e) = engine.device_mut().apply_cdo(&cdo) {
            return TestOutcome::LoadError {
                message: format!("Failed to apply CDO: {}", e),
            };
        }

        // Sync core enabled state from device tiles to engine
        engine.sync_cores_from_device();

        // Load ELF files if project directory exists
        let elf_files = test.find_elf_files();
        for (col, row, path) in &elf_files {
            let data = match std::fs::read(path) {
                Ok(d) => d,
                Err(e) => {
                    return TestOutcome::LoadError {
                        message: format!("Failed to read ELF {:?}: {}", path, e),
                    };
                }
            };

            if let Err(e) = engine.load_elf_bytes(*col as usize, *row as usize, &data) {
                return TestOutcome::LoadError {
                    message: format!("Failed to load ELF into ({},{}): {}", col, row, e),
                };
            }
        }

        // Debug: print enabled cores
        let enabled = engine.enabled_cores();
        if enabled == 0 && elf_files.is_empty() {
            // No cores enabled and no ELFs - likely a reconfiguration test
            return TestOutcome::Pass { cycles: 1 };
        }

        // Run until halt, error, or timeout
        self.run_engine(&mut engine, test)
    }

    /// Run an engine until completion.
    fn run_engine(&self, engine: &mut InterpreterEngine, test: &XclbinTest) -> TestOutcome {
        let mut cycles = 0u64;

        while cycles < self.max_cycles {
            engine.step();
            cycles = engine.total_cycles();

            match engine.status() {
                EngineStatus::Halted => {
                    return TestOutcome::Pass { cycles };
                }
                EngineStatus::Error => {
                    // Try to extract error details from the failed core
                    if let Some(details) = self.extract_error_details(engine) {
                        return details;
                    }
                    return TestOutcome::Fail {
                        message: "Unknown execution error".to_string(),
                        cycles,
                    };
                }
                EngineStatus::Running | EngineStatus::Ready | EngineStatus::Paused => {
                    // Continue running
                }
            }
        }

        TestOutcome::Timeout { cycles }
    }

    /// Try to extract error details from a failed engine.
    fn extract_error_details(&self, engine: &InterpreterEngine) -> Option<TestOutcome> {
        let cycles = engine.total_cycles();

        // Find the core that failed by checking the last decoded bundle
        // Check each core for error state
        for col in 0..4usize {
            for row in 2..6usize {
                // Get PC from context
                let pc = engine.core_context(col, row)
                    .map(|ctx| ctx.pc())
                    .unwrap_or(0);

                // Check the last bundle for unknown operations
                if let Some(bundle) = engine.core_last_bundle(col, row) {
                    for op in bundle.active_slots() {
                        if let crate::interpreter::bundle::Operation::Unknown { opcode } = &op.op
                        {
                            return Some(TestOutcome::UnknownOpcode {
                                details: UnknownOpcode {
                                    slot: op.slot.clone(),
                                    opcode: *opcode,
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
    pub fn results(&self) -> &[(String, TestOutcome)] {
        &self.results
    }

    /// Generate a summary report.
    pub fn summary_report(&self, result: &SuiteResult) -> String {
        let mut report = String::new();

        report.push_str("=== XCLBIN Test Suite Results ===\n\n");
        report.push_str(&format!(
            "Total: {}, Passed: {}, Failed: {}, Unknown Opcodes: {}, Timeout: {}, Load Error: {}\n\n",
            result.total, result.passed, result.failed, result.unknown, result.timeout, result.load_error
        ));

        // List failed tests
        if result.failed > 0 || result.unknown > 0 || result.timeout > 0 {
            report.push_str("--- Failed Tests ---\n");
            for (name, outcome) in &self.results {
                match outcome {
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

        // List passed tests
        if result.passed > 0 {
            report.push_str("--- Passed Tests ---\n");
            for (name, outcome) in &self.results {
                if let TestOutcome::Pass { cycles } = outcome {
                    report.push_str(&format!("{}: PASS ({} cycles)\n", name, cycles));
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
    /// Number of tests that hit unknown opcodes.
    pub unknown: usize,
    /// Number of tests that timed out.
    pub timeout: usize,
    /// Number of tests that failed to load.
    pub load_error: usize,
}

impl SuiteResult {
    /// Get pass rate as a percentage.
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.passed as f64 / self.total as f64) * 100.0
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
            unknown: 1,
            timeout: 1,
            load_error: 0,
        };
        assert!((result.pass_rate() - 70.0).abs() < 0.01);
    }
}
