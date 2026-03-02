//! Run statistics and hardware execution results.
//!
//! Centralizes all test run counters into `RunStats`, replacing the ~30
//! scattered `let mut` counter variables that previously lived in main().
//! Each `record_*` method enforces correct increment logic.

use serde::Serialize;

use crate::testing::xclbin_suite::{Compiler, TestOutcome};
use crate::testing::hardware_comparison::{
    Diagnosis, HardwareValidation, CompilerDiagnosis,
};

/// Structured outcome of a hardware test execution.
///
/// Separates semantics (what happened) from display (how to show it).
/// All branching on test outcome should use this enum, not string parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HwOutcome {
    /// Test executed and produced correct results.
    Pass,
    /// Test executed but produced wrong results.
    Fail,
    /// Test execution failed (timeout, crash, unknown error).
    Error,
    /// Device entered D-state (survived SIGKILL). Unrecoverable.
    Wedged,
}

/// Result from running a single test on real NPU hardware.
pub struct HwRunResult {
    /// Structured outcome for branching (pass/fail/error/wedged).
    pub outcome: HwOutcome,
    /// Display label (e.g. "PASS (64/64)", "FAIL (3/8)", "ERROR (...)").
    pub label: String,
    /// Raw output bytes from the NPU (empty on execution error).
    pub output: Vec<u8>,
    /// Wall-clock time in seconds.
    pub elapsed_secs: f64,
}

/// Accumulated statistics from a test run across all execution modes.
///
/// `record_*` methods enforce correct increment logic.
/// `print_summary` conditionally displays only non-zero sections.
#[derive(Default)]
pub struct RunStats {
    // Emulator outcomes
    pub passed: usize,
    /// Subset of `passed` where output was validated against reference data.
    pub passed_validated: usize,
    pub validation_failed: usize,
    pub expected_fail: usize,
    pub unexpected_pass: usize,
    pub skipped: usize,
    pub platform: usize,
    pub failed: usize,
    pub unknown: usize,
    pub timeout: usize,
    pub load_error: usize,
    // Chess emulator outcomes (mirrors Peano emulator counters above)
    pub chess_emu_passed: usize,
    pub chess_emu_passed_validated: usize,
    pub chess_emu_validation_fail: usize,
    pub chess_emu_expected_fail: usize,
    pub chess_emu_unexpected_pass: usize,
    pub chess_emu_skipped: usize,
    pub chess_emu_platform: usize,
    pub chess_emu_failed: usize,
    pub chess_emu_unknown: usize,
    pub chess_emu_timeout: usize,
    pub chess_emu_load_error: usize,
    // HW cross-validation (from captured npu-outputs dir)
    pub hw_validated: usize,
    pub hw_correct: usize,
    pub hw_compiler_bug: usize,
    pub hw_emulator_bug: usize,
    // Live hardware execution
    pub peano_hw_attempted: usize,
    pub peano_hw_pass: usize,
    pub peano_hw_fail: usize,
    pub peano_hw_error: usize,
    pub chess_hw_attempted: usize,
    pub chess_hw_pass: usize,
    pub chess_hw_fail: usize,
    pub chess_hw_error: usize,
    // Compiler comparison (Peano vs Chess)
    pub chess_compared: usize,
    pub chess_correct: usize,
    pub chess_peano_bugs: usize,
    pub chess_chess_bugs: usize,
    pub chess_emu_bugs: usize,
    // aiesimulator
    pub sim_attempted: usize,
    pub sim_correct: usize,
    pub sim_wrong: usize,
    pub sim_error: usize,
    // HW XFAIL tracking (additive overlay on record_hw counts)
    pub peano_hw_expected_fail: usize,
    pub peano_hw_unexpected_pass: usize,
    pub chess_hw_expected_fail: usize,
    pub chess_hw_unexpected_pass: usize,
    // Differential (hw-only: Peano vs Chess on HW)
    pub both_pass: usize,
    pub peano_only: usize,
    pub chess_only: usize,
    pub both_fail: usize,
    /// Subset of `both_fail` where the test has XFAIL annotation.
    pub both_fail_xfail: usize,
    // Cascade detection
    /// Test index (1-based) where HW execution was disabled due to cascade.
    pub hw_cascade_stopped_at: Option<usize>,
    /// Number of tests that were not attempted on hardware due to cascade.
    pub hw_cascade_skipped: usize,
    // Ground truth comparison
    pub gt_match: usize,
    pub gt_diverge: usize,
    pub gt_new: usize,
    // Per-test warnings (DMA queue full, parse errors, etc.)
    /// Total warnings across all tests.
    pub total_warnings: usize,
    /// Number of tests that had at least one warning.
    pub tests_with_warnings: usize,
}

impl RunStats {
    pub fn record_emu_outcome(&mut self, outcome: &TestOutcome) {
        match outcome {
            TestOutcome::Pass { correct, .. } => {
                self.passed += 1;
                if correct.is_some() {
                    self.passed_validated += 1;
                }
            }
            TestOutcome::ValidationFail { .. } => self.validation_failed += 1,
            TestOutcome::Fail { .. } => self.failed += 1,
            TestOutcome::UnknownOpcode { .. } => self.unknown += 1,
            TestOutcome::Timeout { .. } => self.timeout += 1,
            TestOutcome::LoadError { .. } => self.load_error += 1,
            TestOutcome::ExpectedFail { .. } => self.expected_fail += 1,
            TestOutcome::UnexpectedPass { .. } => self.unexpected_pass += 1,
            TestOutcome::Skipped { .. } => self.skipped += 1,
            TestOutcome::Platform { .. } => self.platform += 1,
        }
    }

    /// Record a Chess emulator outcome. Mirrors `record_emu_outcome()` but
    /// tracks into Chess-specific counters.
    pub fn record_chess_emu(&mut self, outcome: &TestOutcome) {
        match outcome {
            TestOutcome::Pass { correct, .. } => {
                self.chess_emu_passed += 1;
                if correct.is_some() {
                    self.chess_emu_passed_validated += 1;
                }
            }
            TestOutcome::ValidationFail { .. } => self.chess_emu_validation_fail += 1,
            TestOutcome::Fail { .. } => self.chess_emu_failed += 1,
            TestOutcome::UnknownOpcode { .. } => self.chess_emu_unknown += 1,
            TestOutcome::Timeout { .. } => self.chess_emu_timeout += 1,
            TestOutcome::LoadError { .. } => self.chess_emu_load_error += 1,
            TestOutcome::ExpectedFail { .. } => self.chess_emu_expected_fail += 1,
            TestOutcome::UnexpectedPass { .. } => self.chess_emu_unexpected_pass += 1,
            TestOutcome::Skipped { .. } => self.chess_emu_skipped += 1,
            TestOutcome::Platform { .. } => self.chess_emu_platform += 1,
        }
    }

    pub fn record_hw_validation(&mut self, hv: &HardwareValidation) {
        if hv.diagnosis == Diagnosis::NoReference {
            return;
        }
        self.hw_validated += 1;
        match hv.diagnosis {
            Diagnosis::Correct => self.hw_correct += 1,
            Diagnosis::CompilerBug => self.hw_compiler_bug += 1,
            Diagnosis::EmulatorBug => self.hw_emulator_bug += 1,
            _ => {}
        }
    }

    /// Record a hardware execution result, attributed to the correct compiler.
    pub fn record_hw(&mut self, compiler: Compiler, hw: &HwRunResult) {
        let (attempted, pass, fail, error) = match compiler {
            Compiler::Peano => (
                &mut self.peano_hw_attempted, &mut self.peano_hw_pass,
                &mut self.peano_hw_fail, &mut self.peano_hw_error,
            ),
            Compiler::Chess => (
                &mut self.chess_hw_attempted, &mut self.chess_hw_pass,
                &mut self.chess_hw_fail, &mut self.chess_hw_error,
            ),
        };
        *attempted += 1;
        match hw.outcome {
            HwOutcome::Pass => *pass += 1,
            HwOutcome::Fail => *fail += 1,
            HwOutcome::Error | HwOutcome::Wedged => *error += 1,
        }
    }

    /// Record XFAIL-adjusted hardware result. Call after `record_hw()`.
    ///
    /// When `is_xfail` is true: a failing HW result becomes "expected fail"
    /// and a passing HW result becomes "unexpected pass". These counters are
    /// additive overlays -- `record_hw()` still tracks raw pass/fail counts.
    pub fn record_hw_xfail(&mut self, compiler: Compiler, hw: &HwRunResult, is_xfail: bool) {
        if !is_xfail { return; }
        let (efail, upass) = match compiler {
            Compiler::Peano => (&mut self.peano_hw_expected_fail, &mut self.peano_hw_unexpected_pass),
            Compiler::Chess => (&mut self.chess_hw_expected_fail, &mut self.chess_hw_unexpected_pass),
        };
        match hw.outcome {
            HwOutcome::Pass => *upass += 1,
            _ => *efail += 1,
        }
    }

    pub fn record_compiler_diagnosis(&mut self, diag: CompilerDiagnosis) {
        if diag == CompilerDiagnosis::Incomplete {
            return;
        }
        self.chess_compared += 1;
        match diag {
            CompilerDiagnosis::Correct => self.chess_correct += 1,
            CompilerDiagnosis::PeanoCompilerBug => self.chess_peano_bugs += 1,
            CompilerDiagnosis::ChessCompilerBug => self.chess_chess_bugs += 1,
            CompilerDiagnosis::EmulatorBug => self.chess_emu_bugs += 1,
            _ => {}
        }
    }

    pub fn record_aiesim(&mut self, label: &str) {
        self.sim_attempted += 1;
        if label.starts_with("PASS") {
            self.sim_correct += 1;
        } else if label.starts_with("FAIL") {
            self.sim_wrong += 1;
        } else {
            self.sim_error += 1;
        }
    }

    pub fn record_warnings(&mut self, count: usize) {
        if count > 0 {
            self.total_warnings += count;
            self.tests_with_warnings += 1;
        }
    }

    pub fn record_differential(&mut self, peano_pass: bool, chess_pass: bool) {
        match (peano_pass, chess_pass) {
            (true, true) => self.both_pass += 1,
            (true, false) => self.peano_only += 1,
            (false, true) => self.chess_only += 1,
            (false, false) => self.both_fail += 1,
        }
    }

    /// Format a single HW summary line with XFAIL annotations.
    fn print_hw_line(
        &self, label: &str,
        pass: usize, attempted: usize,
        fail: usize, error: usize,
        expected_fail: usize, unexpected_pass: usize,
    ) {
        let real_fail = fail.saturating_sub(expected_fail);
        let mut detail = format!("{}/{} pass", pass, attempted);
        if real_fail > 0 {
            detail.push_str(&format!(", {} fail", real_fail));
        }
        if error > 0 {
            detail.push_str(&format!(", {} error", error));
        }
        if expected_fail > 0 {
            detail.push_str(&format!("  ({} expected fail)", expected_fail));
        }
        if unexpected_pass > 0 {
            detail.push_str(&format!("  ({} unexpected pass)", unexpected_pass));
        }
        println!("{:18}{}", label, detail);
    }

    /// Format emulator outcome lines for one compiler.
    fn print_emu_section(
        &self, effective: usize,
        passed: usize, passed_validated: usize,
        expected_fail: usize, unexpected_pass: usize,
        validation_fail: usize, failed: usize,
        unknown: usize, timeout: usize, load_error: usize,
    ) {
        let unvalidated = passed.saturating_sub(passed_validated);
        println!("Passed:           {} ({:.1}%){}", passed,
            100.0 * passed as f64 / effective.max(1) as f64,
            if unvalidated > 0 {
                format!("  ({} validated, {} no validation)", passed_validated, unvalidated)
            } else {
                String::new()
            });
        println!("Expected Fail:    {}", expected_fail);
        println!("Unexpected Pass:  {}", unexpected_pass);
        println!("Validation Fail:  {}", validation_fail);
        println!("Failed:           {}", failed);
        println!("Unknown:          {}", unknown);
        println!("Timeout:          {}", timeout);
        println!("Load Error:       {}", load_error);
    }

    /// Print run summary, conditionally showing only non-zero sections.
    ///
    /// In hw-only mode the emulator section is suppressed and the header
    /// changes to "HARDWARE-ONLY SUMMARY".
    pub fn print_summary(&self, total: usize, hw_only: bool) {
        println!("\n{:=<60}", "");

        if hw_only {
            println!("=== HARDWARE-ONLY SUMMARY ===");
            println!("Total:            {}", total);
            println!("Skipped:          {} (no buffer spec or insts.bin)", self.skipped);
        } else {
            println!("=== SUMMARY ===");
            let effective = total - self.skipped - self.platform;
            println!("Total:            {}", total);
            println!("Platform:         {} (requires different hardware)", self.platform);
            println!("Skipped:          {} (missing artifacts or no instructions)", self.skipped);

            let chess_emu_attempted = self.chess_emu_passed + self.chess_emu_validation_fail
                + self.chess_emu_expected_fail + self.chess_emu_unexpected_pass
                + self.chess_emu_failed + self.chess_emu_unknown
                + self.chess_emu_timeout + self.chess_emu_load_error;
            let has_chess_emu = chess_emu_attempted > 0;

            if has_chess_emu {
                // Per-compiler breakdown when Chess emulator is active
                println!("\n--- Peano Emulator ---");
            }
            self.print_emu_section(effective,
                self.passed, self.passed_validated,
                self.expected_fail, self.unexpected_pass,
                self.validation_failed, self.failed,
                self.unknown, self.timeout, self.load_error);

            if has_chess_emu {
                println!("\n--- Chess Emulator ---");
                self.print_emu_section(effective,
                    self.chess_emu_passed, self.chess_emu_passed_validated,
                    self.chess_emu_expected_fail, self.chess_emu_unexpected_pass,
                    self.chess_emu_validation_fail, self.chess_emu_failed,
                    self.chess_emu_unknown, self.chess_emu_timeout,
                    self.chess_emu_load_error);
            }
        }

        // Per-test warnings
        if self.total_warnings > 0 {
            println!("Warnings:         {} ({} tests)", self.total_warnings, self.tests_with_warnings);
        }

        // Hardware cross-validation (from captured npu-outputs dir)
        if self.hw_validated > 0 {
            println!("\n=== HARDWARE CROSS-VALIDATION ===");
            println!("Validated:        {}", self.hw_validated);
            println!("Correct:          {}", self.hw_correct);
            println!("Compiler Bug:     {}", self.hw_compiler_bug);
            println!("Emulator Bug:     {}", self.hw_emulator_bug);
        }

        // Chess vs Peano compiler comparison
        if self.chess_compared > 0 {
            println!("\n=== CHESS vs PEANO COMPARISON ===");
            println!("Compared:         {}", self.chess_compared);
            println!("Correct:          {} (both compilers match expected)", self.chess_correct);
            println!("Peano Bug:        {} (Chess correct, Peano wrong)", self.chess_peano_bugs);
            println!("Chess Bug:        {} (Peano correct, Chess wrong)", self.chess_chess_bugs);
            println!("Emulator Bug:     {} (both wrong, same way)", self.chess_emu_bugs);
        }

        // Live hardware execution
        if self.peano_hw_attempted > 0 || self.chess_hw_attempted > 0 {
            println!("\n=== LIVE HARDWARE EXECUTION ===");
            if self.peano_hw_attempted > 0 {
                self.print_hw_line("Peano HW:",
                    self.peano_hw_pass, self.peano_hw_attempted,
                    self.peano_hw_fail, self.peano_hw_error,
                    self.peano_hw_expected_fail, self.peano_hw_unexpected_pass);
            }
            if self.chess_hw_attempted > 0 {
                self.print_hw_line("Chess HW:",
                    self.chess_hw_pass, self.chess_hw_attempted,
                    self.chess_hw_fail, self.chess_hw_error,
                    self.chess_hw_expected_fail, self.chess_hw_unexpected_pass);
            }
        }

        // Cascade detection warning
        if let Some(stopped_at) = self.hw_cascade_stopped_at {
            println!("\n  WARNING: Hardware execution stopped at test {} due to device cascade failure.",
                stopped_at);
            println!("  {} tests were not attempted on hardware.", self.hw_cascade_skipped);
        }

        // aiesimulator cross-validation
        if self.sim_attempted > 0 {
            println!("\n=== AIESIMULATOR CROSS-VALIDATION ===");
            println!("Simulated:        {}", self.sim_attempted);
            println!("Correct:          {}", self.sim_correct);
            println!("Wrong:            {}", self.sim_wrong);
            println!("Error:            {} (build/invocation failures)", self.sim_error);
        }

        // Ground truth comparison
        let gt_total = self.gt_match + self.gt_diverge + self.gt_new;
        if gt_total > 0 {
            println!("\n=== GROUND TRUTH ===");
            println!("Compared:         {}", gt_total);
            println!("Match:            {}", self.gt_match);
            if self.gt_diverge > 0 {
                println!("Diverge:          {}", self.gt_diverge);
            }
            if self.gt_new > 0 {
                println!("New captures:     {}", self.gt_new);
            }
        }

        // Differential summary (Peano vs Chess on HW, typically hw-only mode)
        let compared = self.both_pass + self.peano_only + self.chess_only + self.both_fail;
        if compared > 0 {
            println!("\n=== DIFFERENTIAL (Peano vs Chess on HW) ===");
            println!("Compared:         {}", compared);
            println!("Both pass:        {}", self.both_pass);
            if self.peano_only > 0 {
                println!("Peano only:       {} (Chess bug?)", self.peano_only);
            }
            if self.chess_only > 0 {
                println!("Chess only:       {} (Peano bug?)", self.chess_only);
            }
            if self.both_fail > 0 {
                let annotation = if self.both_fail_xfail > 0 {
                    format!(" ({} XFAIL)", self.both_fail_xfail)
                } else {
                    String::new()
                };
                println!("Both fail:        {} (test issue?){}", self.both_fail, annotation);
            }
        }
    }

    /// Build a serializable summary from aggregate counters.
    pub fn to_summary(&self, total: usize) -> ReportSummary {
        ReportSummary {
            total,
            passed: self.passed,
            expected_fail: self.expected_fail,
            unexpected_pass: self.unexpected_pass,
            validation_fail: self.validation_failed,
            failed: self.failed,
            unknown: self.unknown,
            timeout: self.timeout,
            load_error: self.load_error,
            skipped: self.skipped,
            platform: self.platform,
            peano_hw_pass: self.peano_hw_pass,
            peano_hw_attempted: self.peano_hw_attempted,
            chess_hw_pass: self.chess_hw_pass,
            chess_hw_attempted: self.chess_hw_attempted,
        }
    }
}

// ---------------------------------------------------------------------------
// Machine-readable JSON report types
// ---------------------------------------------------------------------------

/// Top-level test report, written to `build/results.json`.
#[derive(Debug, Serialize)]
pub struct TestReport {
    /// Unix epoch seconds when the run started.
    pub timestamp: u64,
    /// Short git commit hash (or "unknown").
    pub git_commit: String,
    /// Execution targets (e.g. "emu", "emu,hw").
    pub runtime: String,
    /// Compiler selection (e.g. "peano", "peano,chess").
    pub compiler: String,
    /// Test suite selection (e.g. "npu-xrt", "npu-xrt,examples").
    pub suite: String,
    /// Per-test entries.
    pub tests: Vec<TestEntry>,
    /// Aggregate summary.
    pub summary: ReportSummary,
}

/// Per-test result entry.
#[derive(Debug, Serialize)]
pub struct TestEntry {
    pub name: String,
    /// Peano emulator result (None if emulator was not run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emulator: Option<EmulatorResult>,
    /// Peano hardware result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peano_hw: Option<HwReportResult>,
    /// Chess hardware result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chess_hw: Option<HwReportResult>,
    /// Chess emulator result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chess_emulator: Option<EmulatorResult>,
    /// Expected failure reason from test_overrides.toml (None if not XFAIL).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xfail_reason: Option<String>,
    /// Number of warnings emitted during emulation.
    pub warning_count: usize,
}

/// Emulator execution result for a single test.
#[derive(Debug, Serialize)]
pub struct EmulatorResult {
    /// Outcome category: "pass", "fail", "timeout", "unknown_opcode",
    /// "load_error", "validation_fail", "expected_fail", "unexpected_pass",
    /// "skipped", "platform".
    pub outcome: String,
    /// Cycle count (if the test ran).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycles: Option<u64>,
    /// Correct output values (if validation was performed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correct: Option<usize>,
    /// Total output values checked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<usize>,
    /// Reason string (for fail, skip, platform, expected_fail).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Hardware execution result for a single test.
#[derive(Debug, Serialize)]
pub struct HwReportResult {
    /// Outcome: "pass", "fail", "error", "wedged".
    pub outcome: String,
    /// Display label (e.g. "PASS (64/64)").
    pub label: String,
    /// Wall-clock execution time in seconds.
    pub elapsed_secs: f64,
    /// XFAIL status: "expected_fail", "unexpected_pass", or absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xfail_status: Option<String>,
}

/// Aggregate summary counters.
#[derive(Debug, Serialize)]
pub struct ReportSummary {
    pub total: usize,
    pub passed: usize,
    pub expected_fail: usize,
    pub unexpected_pass: usize,
    pub validation_fail: usize,
    pub failed: usize,
    pub unknown: usize,
    pub timeout: usize,
    pub load_error: usize,
    pub skipped: usize,
    pub platform: usize,
    pub peano_hw_pass: usize,
    pub peano_hw_attempted: usize,
    pub chess_hw_pass: usize,
    pub chess_hw_attempted: usize,
}

impl HwRunResult {
    /// Convert to a serializable report entry.
    pub fn to_report(&self, is_xfail: bool) -> HwReportResult {
        let outcome_str = match self.outcome {
            HwOutcome::Pass => "pass",
            HwOutcome::Fail => "fail",
            HwOutcome::Error => "error",
            HwOutcome::Wedged => "wedged",
        };
        let xfail_status = if is_xfail {
            Some(match self.outcome {
                HwOutcome::Pass => "unexpected_pass".to_string(),
                _ => "expected_fail".to_string(),
            })
        } else {
            None
        };
        HwReportResult {
            outcome: outcome_str.to_string(),
            label: self.label.clone(),
            elapsed_secs: self.elapsed_secs,
            xfail_status,
        }
    }
}

impl TestOutcome {
    /// Convert to a serializable emulator result.
    pub fn to_report(&self) -> EmulatorResult {
        match self {
            TestOutcome::Pass { cycles, correct, total } => EmulatorResult {
                outcome: "pass".to_string(),
                cycles: Some(*cycles),
                correct: *correct,
                total: *total,
                reason: None,
            },
            TestOutcome::ValidationFail { cycles, correct, total, .. } => EmulatorResult {
                outcome: "validation_fail".to_string(),
                cycles: Some(*cycles),
                correct: Some(*correct),
                total: Some(*total),
                reason: None,
            },
            TestOutcome::ExpectedFail { cycles, reason, actual } => EmulatorResult {
                outcome: "expected_fail".to_string(),
                cycles: Some(*cycles),
                correct: None,
                total: None,
                reason: Some(format!("{}: {}", reason, actual)),
            },
            TestOutcome::UnexpectedPass { cycles, correct, total } => EmulatorResult {
                outcome: "unexpected_pass".to_string(),
                cycles: Some(*cycles),
                correct: Some(*correct),
                total: Some(*total),
                reason: None,
            },
            TestOutcome::Fail { message, cycles } => EmulatorResult {
                outcome: "fail".to_string(),
                cycles: Some(*cycles),
                correct: None,
                total: None,
                reason: Some(message.clone()),
            },
            TestOutcome::UnknownOpcode { cycles, .. } => EmulatorResult {
                outcome: "unknown_opcode".to_string(),
                cycles: Some(*cycles),
                correct: None,
                total: None,
                reason: None,
            },
            TestOutcome::Timeout { cycles } => EmulatorResult {
                outcome: "timeout".to_string(),
                cycles: Some(*cycles),
                correct: None,
                total: None,
                reason: None,
            },
            TestOutcome::LoadError { message } => EmulatorResult {
                outcome: "load_error".to_string(),
                cycles: None,
                correct: None,
                total: None,
                reason: Some(message.clone()),
            },
            TestOutcome::Skipped { reason } => EmulatorResult {
                outcome: "skipped".to_string(),
                cycles: None,
                correct: None,
                total: None,
                reason: Some(reason.clone()),
            },
            TestOutcome::Platform { required, reason } => EmulatorResult {
                outcome: "platform".to_string(),
                cycles: None,
                correct: None,
                total: None,
                reason: Some(format!("{}: {}", required, reason)),
            },
        }
    }
}
