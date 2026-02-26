//! Run statistics and hardware execution results.
//!
//! Centralizes all test run counters into `RunStats`, replacing the ~30
//! scattered `let mut` counter variables that previously lived in main().
//! Each `record_*` method enforces correct increment logic.

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
    pub validation_failed: usize,
    pub expected_fail: usize,
    pub unexpected_pass: usize,
    pub skipped: usize,
    pub platform: usize,
    pub failed: usize,
    pub unknown: usize,
    pub timeout: usize,
    pub load_error: usize,
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
    // Differential (hw-only: Peano vs Chess on HW)
    pub both_pass: usize,
    pub peano_only: usize,
    pub chess_only: usize,
    pub both_fail: usize,
    // Cascade detection
    /// Test index (1-based) where HW execution was disabled due to cascade.
    pub hw_cascade_stopped_at: Option<usize>,
    /// Number of tests that were not attempted on hardware due to cascade.
    pub hw_cascade_skipped: usize,
}

impl RunStats {
    pub fn record_emu_outcome(&mut self, outcome: &TestOutcome) {
        match outcome {
            TestOutcome::Pass { .. } => self.passed += 1,
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

    pub fn record_differential(&mut self, peano_pass: bool, chess_pass: bool) {
        match (peano_pass, chess_pass) {
            (true, true) => self.both_pass += 1,
            (true, false) => self.peano_only += 1,
            (false, true) => self.chess_only += 1,
            (false, false) => self.both_fail += 1,
        }
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
            println!("Skipped:          {}", self.skipped);
            println!("Passed:           {} ({:.1}%)", self.passed,
                100.0 * self.passed as f64 / effective.max(1) as f64);
            println!("Expected Fail:    {}", self.expected_fail);
            println!("Unexpected Pass:  {}", self.unexpected_pass);
            println!("Validation Fail:  {}", self.validation_failed);
            println!("Failed:           {}", self.failed);
            println!("Unknown:          {}", self.unknown);
            println!("Timeout:          {}", self.timeout);
            println!("Load Error:       {}", self.load_error);
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
                let mut detail = format!("{}/{} pass", self.peano_hw_pass, self.peano_hw_attempted);
                if self.peano_hw_fail > 0 {
                    detail.push_str(&format!(", {} fail", self.peano_hw_fail));
                }
                if self.peano_hw_error > 0 {
                    detail.push_str(&format!(", {} error", self.peano_hw_error));
                }
                println!("Peano HW:         {}", detail);
            }
            if self.chess_hw_attempted > 0 {
                let mut detail = format!("{}/{} pass", self.chess_hw_pass, self.chess_hw_attempted);
                if self.chess_hw_fail > 0 {
                    detail.push_str(&format!(", {} fail", self.chess_hw_fail));
                }
                if self.chess_hw_error > 0 {
                    detail.push_str(&format!(", {} error", self.chess_hw_error));
                }
                println!("Chess HW:         {}", detail);
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
                println!("Both fail:        {} (test issue?)", self.both_fail);
            }
        }
    }
}
