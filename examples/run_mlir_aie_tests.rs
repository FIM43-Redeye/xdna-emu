//! Run all xclbin tests from mlir-aie npu-xrt directory.
//!
//! This is a quick test runner to see how many mlir-aie tests pass.
//! Supports manifest-based output validation when manifests are available.
//!
//! Usage:
//!   cargo run --example run_mlir_aie_tests [OPTIONS] [FILTER...]
//!
//! Options:
//!   --verbose, -v     Show full expected/actual output arrays for failures
//!   -j N              Run N tests in parallel (default: auto, up to 8)
//!   --elfanalyze      Run elfanalyzer on each test's ELFs (requires aietools)
//!   --no-chess        Disable Chess compiler builds (Chess is auto-detected by default)
//!   --no-hw           Disable NPU hardware validation (auto-detected by default)
//!   --aiesim          Run aiesimulator on Chess-built .prj (requires aietools)
//!   --unit-tests      Run mlir-aie chess_compiler_tests_aie2 unit tests (requires aietools)
//!   --full            Enable all validation: elfanalyze, aiesim, unit-tests, hw
//!   --no-build        Skip build phase, discover from build output directories
//!
//! Positional arguments are substring filters on test name. Multiple filters
//! are OR-ed (a test runs if it matches ANY filter).
//!
//! Examples:
//!   cargo run --example run_mlir_aie_tests                        # run all tests
//!   cargo run --example run_mlir_aie_tests -- add_blockwrite      # run one test
//!   cargo run --example run_mlir_aie_tests -- -v vec_vec          # verbose for matching
//!   cargo run --example run_mlir_aie_tests -- -v                  # verbose for all
//!   cargo run --example run_mlir_aie_tests -- -j 8                # run 8 tests at once
//!   cargo run --example run_mlir_aie_tests -- --elfanalyze add_one  # ELF analysis
//!   cargo run --example run_mlir_aie_tests -- --no-chess add_one    # Peano only (skip Chess)
//!   cargo run --example run_mlir_aie_tests -- --no-hw add_one       # skip real NPU
//!   cargo run --example run_mlir_aie_tests -- --full add_one        # full validation matrix
//!   cargo run --example run_mlir_aie_tests -- --full                # full validation, all tests

use std::path::{Path, PathBuf};
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use serde::Deserialize;
use xdna_emu::testing::xclbin_suite::{XclbinSuite, XclbinTest, TestOutcome};
use xdna_emu::testing::manifest_runner::{TestManifest, ElementType, read_values};
use xdna_emu::testing::hardware_comparison::{
    Diagnosis, HardwareValidation, CompilerComparison, CompilerDiagnosis,
    generate_input_values,
};
use xdna_emu::testing::npu_runner;
use xdna_emu::testing::npu_test::{self, NpuTestSource};
use xdna_emu::integration::aietools::AieTools;
use xdna_emu::integration::aiesimulator;
use xdna_emu::integration::elfanalyzer;
use xdna_emu::integration::chess_build::{BuildEnv, BuildOpts, BuildResult, find_all_xclbin_results};
use xdna_emu::testing::unit_test;

// ---------------------------------------------------------------------------
// Runner configuration (loaded from runner.toml)
// ---------------------------------------------------------------------------

/// Persistent configuration for the test runner.
///
/// Loaded from `runner.toml` at the project root. Missing keys fall back
/// to built-in defaults. CLI flags override config values.
#[derive(Debug, Deserialize)]
#[serde(default)]
struct RunnerConfig {
    execution: ExecutionConfig,
    build: BuildConfig,
    chess: ChessConfig,
    aiesim: AiesimConfig,
    unit_tests: UnitTestsConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct ExecutionConfig {
    /// Maximum emulator cycles before TDR-based timeout. 0 = TDR only.
    max_cycles: u64,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct BuildConfig {
    /// Nice level for compilation subprocesses (0-19).
    nice_level: i32,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct ChessConfig {
    /// Build tests with Chess compiler during batch build phase.
    build: bool,
    /// Run Chess-compiled binaries through the emulator.
    run_emulator: bool,
    /// Run Chess-compiled binaries on real NPU hardware.
    run_hardware: bool,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct AiesimConfig {
    /// Run aiesimulator on Chess-built .prj directories.
    enabled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct UnitTestsConfig {
    /// Enable unit test discovery and execution.
    enabled: bool,
    /// Run discovered unit tests through aiesimulator.
    aiesim: bool,
    /// Maximum simulation cycles for unit test aiesimulator runs.
    aiesim_timeout: u64,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            execution: ExecutionConfig::default(),
            build: BuildConfig::default(),
            chess: ChessConfig::default(),
            aiesim: AiesimConfig::default(),
            unit_tests: UnitTestsConfig::default(),
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self { max_cycles: 1_000_000 }
    }
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self { nice_level: 19 }
    }
}

impl Default for ChessConfig {
    fn default() -> Self {
        Self {
            build: false,
            run_emulator: false,
            run_hardware: true,
        }
    }
}

impl Default for AiesimConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

impl Default for UnitTestsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            aiesim: true,
            aiesim_timeout: 1_000_000,
        }
    }
}

impl RunnerConfig {
    /// Load config from `runner.toml` in the project root.
    ///
    /// Returns the built-in defaults if the file is missing or unparseable.
    fn load() -> Self {
        let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runner.toml");
        match std::fs::read_to_string(&config_path) {
            Ok(content) => {
                match toml::from_str(&content) {
                    Ok(config) => {
                        log::debug!("Loaded runner config from {}", config_path.display());
                        config
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to parse {}: {}", config_path.display(), e);
                        Self::default()
                    }
                }
            }
            Err(_) => Self::default(),
        }
    }

    /// Print a summary of non-default settings.
    fn print_active_overrides(&self) {
        let d = Self::default();
        let mut overrides = Vec::new();
        if self.chess.run_emulator != d.chess.run_emulator {
            overrides.push(format!("chess.run_emulator={}", self.chess.run_emulator));
        }
        if self.chess.run_hardware != d.chess.run_hardware {
            overrides.push(format!("chess.run_hardware={}", self.chess.run_hardware));
        }
        if self.aiesim.enabled != d.aiesim.enabled {
            overrides.push(format!("aiesim.enabled={}", self.aiesim.enabled));
        }
        if self.unit_tests.enabled != d.unit_tests.enabled {
            overrides.push(format!("unit_tests.enabled={}", self.unit_tests.enabled));
        }
        if self.build.nice_level != d.build.nice_level {
            overrides.push(format!("build.nice_level={}", self.build.nice_level));
        }
        if self.execution.max_cycles != d.execution.max_cycles {
            overrides.push(format!("execution.max_cycles={}", self.execution.max_cycles));
        }
        if !overrides.is_empty() {
            println!("config: {}", overrides.join(", "));
        }
    }
}

// ---------------------------------------------------------------------------
// CLI options (merged with config)
// ---------------------------------------------------------------------------

/// Parsed CLI options, merged with runner config.
struct Options {
    verbose: bool,
    jobs: usize,
    filters: Vec<String>,
    /// Run elfanalyzer on each test's ELF files (requires aietools).
    elfanalyze: bool,
    /// Build with Chess compiler (batch build phase).
    chess_build: bool,
    /// Run Chess-compiled binaries through the emulator.
    chess_emulator: bool,
    /// Run Chess-compiled binaries on real NPU hardware.
    chess_hardware: bool,
    /// Run tests on real NPU hardware (requires /dev/accel/accel0 + npu-runner).
    hw: bool,
    /// Run aiesimulator on Chess-built .prj directories.
    aiesim: bool,
    /// Nice level for build subprocesses.
    build_nice: i32,
    /// Maximum emulator cycles before timeout.
    max_cycles: u64,
    /// Run mlir-aie unit tests (chess_compiler_tests_aie2).
    unit_tests: bool,
    /// Run unit tests through aiesimulator.
    unit_tests_aiesim: bool,
    /// Maximum simulation cycles for unit test aiesimulator runs.
    unit_tests_aiesim_timeout: u64,
    /// Skip build phase entirely -- discover from build output directories.
    /// Useful for quick re-runs of already-built tests.
    no_build: bool,
    /// Hardware-only mode: skip emulator, run only on real NPU hardware.
    /// Useful for validating known-good tests quickly on real silicon.
    hw_only: bool,
}

/// Parse CLI arguments and merge with runner config.
fn parse_args(config: &RunnerConfig) -> Options {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut verbose = false;
    let mut jobs: usize = 0; // 0 = auto-detect
    let mut filters = Vec::new();
    let mut elfanalyze = false;
    let mut hw = true; // auto-detected, disabled by --no-hw
    let mut hw_only = false;
    let mut aiesim = false;
    let mut full = false;
    let mut unit_tests = false;
    let mut no_build = false;
    let mut no_chess = false;
    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--verbose" | "-v" => verbose = true,
            "--elfanalyze" => elfanalyze = true,
            // --chess is accepted for backward compat but is now the default
            "--chess" => {}
            "--no-chess" => no_chess = true,
            // --hw is accepted for backward compat but is now the default
            "--hw" => hw = true,
            "--no-hw" => hw = false,
            "--hw-only" => { hw_only = true; hw = true; },
            "--aiesim" => aiesim = true,
            "--full" => full = true,
            "--unit-tests" => unit_tests = true,
            "--no-build" => no_build = true,
            "-j" => {
                if let Some(n) = iter.next() {
                    jobs = n.parse().unwrap_or_else(|_| {
                        eprintln!("Invalid -j value: {}", n);
                        std::process::exit(1);
                    });
                    // -j 0 is nonsensical; treat as 1
                    if jobs == 0 { jobs = 1; }
                } else {
                    eprintln!("-j requires a number");
                    std::process::exit(1);
                }
            }
            _ if !arg.starts_with('-') => filters.push(arg.clone()),
            other => {
                eprintln!("Unknown option: {}", other);
                eprintln!("Usage: run_mlir_aie_tests [--verbose|-v] [-j N] [--elfanalyze] [--no-chess] [--no-hw] [--hw-only] [--aiesim] [--unit-tests] [--full] [--no-build] [FILTER...]");
                std::process::exit(1);
            }
        }
    }

    // --full enables everything
    if full {
        elfanalyze = true;
        no_chess = false;
        hw = true;
        aiesim = true;
        unit_tests = true;
    }

    // Auto-detect parallelism: use available CPU count, capped at 8 to avoid
    // overwhelming the system during builds. User can override with explicit -j N.
    if jobs == 0 {
        jobs = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);
    }

    // Chess builds are ON by default (auto-detected from aietools).
    // --no-chess explicitly disables. Config can also disable.
    // Actual availability is checked at runtime in main() after discovery.
    let chess_build = !no_chess && (config.chess.build || !no_chess);
    let chess_emulator = if chess_build { config.chess.run_emulator } else { false };
    let chess_hardware = if chess_build { config.chess.run_hardware } else { false };
    let aiesim = if aiesim { true } else if chess_build { config.aiesim.enabled } else { false };

    // Unit tests: CLI --unit-tests overrides config.unit_tests.enabled
    let unit_tests = unit_tests || config.unit_tests.enabled;

    Options {
        verbose,
        jobs,
        filters,
        elfanalyze,
        chess_build,
        chess_emulator,
        chess_hardware,
        hw,
        aiesim,
        build_nice: config.build.nice_level,
        max_cycles: config.execution.max_cycles,
        unit_tests,
        unit_tests_aiesim: config.unit_tests.aiesim,
        unit_tests_aiesim_timeout: config.unit_tests.aiesim_timeout,
        no_build,
        hw_only,
    }
}

/// Check if a test name matches any of the given filters.
fn matches_filter(name: &str, filters: &[String]) -> bool {
    if filters.is_empty() {
        return true;
    }
    filters.iter().any(|f| name.contains(f.as_str()))
}

/// Print full expected vs actual output arrays.
///
/// Uses the raw output bytes already captured from the test run, combined
/// with the manifest to generate expected values. No re-run needed.
fn print_verbose_comparison(
    raw_output: &[u8],
    manifest: &TestManifest,
) {
    let output_buf = match manifest.get_output() {
        Some(b) => b,
        None => { println!("      (no output buffer in manifest)"); return; }
    };

    let elem_type = match ElementType::from_str(&output_buf.element_type) {
        Some(t) => t,
        None => { println!("      (unknown element type: {})", output_buf.element_type); return; }
    };

    // Parse actual output from raw bytes
    let output_size = output_buf.size * elem_type.byte_size();
    let actual_bytes = &raw_output[..output_size.min(raw_output.len())];
    let actual = read_values(actual_bytes, elem_type);

    // Generate input values from manifest (needed for expected calculation)
    let mut inputs = std::collections::HashMap::new();
    for name in &["input_a", "input_b"] {
        if let Some(input_data) = manifest.generate_input(name) {
            if let Some(input_buf) = manifest.get_input(name) {
                if let Some(input_elem) = ElementType::from_str(&input_buf.element_type) {
                    inputs.insert(name.to_string(), read_values(&input_data, input_elem));
                }
            }
        }
    }

    // Generate expected values
    let reference_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");
    let ref_dir = if reference_dir.exists() { Some(reference_dir.as_path()) } else { None };
    let expected = match manifest.generate_expected(&inputs, ref_dir) {
        Some(e) => e,
        None => { println!("      (could not generate expected values)"); return; }
    };

    let total = expected.len().min(actual.len());
    let correct = (0..total).filter(|&i| actual[i] == expected[i]).count();
    let hex = elem_type.byte_size() >= 4;

    // Print header
    println!("      --- Output Detail ({}/{} correct, {} elements) ---", correct, total, total);

    // Show all elements, marking mismatches
    let show_max = total.min(128); // Cap at 128 for readability
    for i in 0..show_max {
        let marker = if actual[i] == expected[i] { " " } else { "X" };
        if hex {
            println!("      {} [{:4}] expected {:12} (0x{:08X})  got {:12} (0x{:08X})",
                marker, i, expected[i], expected[i] as u32, actual[i], actual[i] as u32);
        } else {
            println!("      {} [{:4}] expected {:6}  got {:6}", marker, i, expected[i], actual[i]);
        }
    }
    if show_max < total {
        let remaining_correct = ((show_max)..total).filter(|&i| actual[i] == expected[i]).count();
        println!("      ... ({} more elements, {} correct)", total - show_max, remaining_correct);
    }
    println!("      --- End Output Detail ---");
}

/// Result from running a single test, ready for display.
struct TestResult {
    idx: usize,
    name: String,
    elf_count: usize,
    embedded_count: usize,
    has_npu: bool,
    outcome: TestOutcome,
    raw_output: Option<Vec<u8>>,
    /// Hardware cross-validation result (if npu-outputs are available).
    hw_validation: Option<HardwareValidation>,
}

/// Format a test result line for display.
fn format_result(r: &TestResult, total: usize) -> String {
    let mut out = String::new();

    // Header: index, name, code sources
    out.push_str(&format!("[{:2}/{}] {:40} ... ", r.idx + 1, total,
        &r.name[..r.name.len().min(40)]));

    if r.elf_count == 0 && r.embedded_count == 0 {
        out.push_str("(no code) ");
    } else if r.elf_count > 0 && r.embedded_count > 0 {
        out.push_str(&format!("({} ELFs, {} CDO) ", r.elf_count, r.embedded_count));
    } else if r.elf_count > 0 {
        out.push_str(&format!("({} ELFs) ", r.elf_count));
    } else {
        out.push_str(&format!("({} CDO) ", r.embedded_count));
    }

    if !r.has_npu {
        out.push_str("(no NPU) ");
    }

    // Outcome
    match &r.outcome {
        TestOutcome::Pass { cycles, correct, total } => {
            if let (Some(c), Some(t)) = (correct, total) {
                out.push_str(&format!("PASS ({} cycles, {}/{} validated)", cycles, c, t));
            } else {
                out.push_str(&format!("PASS ({} cycles)", cycles));
            }
        }
        TestOutcome::ValidationFail { cycles, correct, total, first_mismatch } => {
            out.push_str(&format!("VALIDATION FAIL ({} cycles, {}/{} correct)", cycles, correct, total));
            if let Some((idx, expected, actual)) = first_mismatch {
                out.push_str(&format!("\n      First mismatch at [{}]: expected {}, got {}", idx, expected, actual));
            }
        }
        TestOutcome::Fail { message, cycles } => {
            out.push_str(&format!("FAIL ({} cycles)\n      {}", cycles, message));
        }
        TestOutcome::UnknownOpcode { details, cycles } => {
            out.push_str(&format!("UNKNOWN ({} cycles)\n      {:?}", cycles, details));
        }
        TestOutcome::Timeout { cycles } => {
            out.push_str(&format!("TIMEOUT ({} cycles)", cycles));
        }
        TestOutcome::LoadError { message } => {
            out.push_str(&format!("LOAD ERROR\n      {}", message));
        }
        TestOutcome::ExpectedFail { cycles, reason, actual } => {
            out.push_str(&format!("EXPECTED FAIL ({} cycles)", cycles));
            if !reason.is_empty() {
                out.push_str(&format!("\n      reason: {}", reason));
            }
            out.push_str(&format!("\n      actual: {}", actual));
        }
        TestOutcome::UnexpectedPass { cycles, correct, total } => {
            out.push_str(&format!("UNEXPECTED PASS ({} cycles, {}/{} correct)", cycles, correct, total));
            out.push_str("\n      Test was expected to fail but passed -- update manifest!");
        }
        TestOutcome::Skipped { reason } => {
            out.push_str(&format!("SKIP\n      {}", reason));
        }
        TestOutcome::Platform { required, reason } => {
            out.push_str(&format!("PLATFORM (requires {})", required));
            if !reason.is_empty() {
                out.push_str(&format!("\n      {}", reason));
            }
        }
    }

    // Append hardware cross-validation diagnosis if available
    if let Some(ref hv) = r.hw_validation {
        match hv.diagnosis {
            Diagnosis::NoReference => {} // Omit -- no useful info
            Diagnosis::Correct => {
                out.push_str("\n      hw: CORRECT (emulator matches hardware)");
            }
            Diagnosis::CompilerBug => {
                out.push_str("\n      hw: COMPILER BUG (emulator matches hardware, both wrong)");
            }
            Diagnosis::EmulatorBug => {
                out.push_str("\n      hw: EMULATOR BUG (hardware correct, emulator diverges)");
            }
            Diagnosis::BothBroken => {
                out.push_str("\n      hw: BOTH BROKEN (emulator, hardware, and manifest all disagree)");
            }
        }
    }

    out
}

/// Run elfanalyzer on a test's ELF files and cross-validate against our parser.
fn run_elfanalyzer(
    tools: &AieTools,
    test: &xdna_emu::testing::xclbin_suite::XclbinTest,
    _mlir_aie_path: &Path,
) {
    // find_elf_files returns Vec<(col, row, PathBuf)>
    let elf_files = test.find_elf_files();
    if elf_files.is_empty() {
        return;
    }

    for (_col, _row, elf_path) in &elf_files {
        let name = elf_path.file_name()
            .map(|n: &std::ffi::OsStr| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        match elfanalyzer::analyze(tools, elf_path) {
            Ok(analysis) => {
                print!("{}", elfanalyzer::format_analysis(&analysis, &name));

                // Cross-validate: compare elfanalyzer output against our parser
                match std::fs::read(elf_path) {
                    Ok(elf_data) => {
                        match elfanalyzer::cross_validate(&analysis, &elf_data) {
                            Ok(cv) => {
                                print!("{}", elfanalyzer::format_cross_validation(&cv, &name));
                            }
                            Err(e) => {
                                eprintln!("      cross-validation error for {}: {}", name, e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("      failed to read ELF for cross-validation: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("      elfanalyzer error for {}: {}", name, e);
            }
        }
    }
}

/// Artifacts from a Chess build, stored during the batch build phase.
struct ChessBuildArtifacts {
    /// Path to the Chess-compiled xclbin.
    xclbin: PathBuf,
    /// Path to the NPU instructions (insts.bin).
    insts: Option<PathBuf>,
    /// Path to the .prj directory (for aiesimulator).
    prj_dir: Option<PathBuf>,
}

/// Result from running a test on real NPU hardware.
struct HwRunResult {
    /// Display label (e.g. "PASS (64/64)", "FAIL (3/8)", "ERROR (...)").
    label: String,
    /// Raw output bytes from the NPU (empty on execution error).
    output: Vec<u8>,
    /// Whether the test passed (label starts with "PASS").
    passed: bool,
    /// Wall-clock time in seconds.
    elapsed_secs: f64,
}

/// Accumulated statistics from a test run across all execution modes.
///
/// Replaces the ~30 scattered `let mut` counter variables with a single
/// struct whose `record_*` methods enforce correct increment logic.
/// `print_summary` conditionally displays only non-zero sections.
#[derive(Default)]
struct RunStats {
    // Emulator outcomes
    passed: usize,
    validation_failed: usize,
    expected_fail: usize,
    unexpected_pass: usize,
    skipped: usize,
    platform: usize,
    failed: usize,
    unknown: usize,
    timeout: usize,
    load_error: usize,
    // HW cross-validation (from captured npu-outputs dir)
    hw_validated: usize,
    hw_correct: usize,
    hw_compiler_bug: usize,
    hw_emulator_bug: usize,
    // Live hardware execution
    peano_hw_attempted: usize,
    peano_hw_pass: usize,
    peano_hw_fail: usize,
    peano_hw_error: usize,
    chess_hw_attempted: usize,
    chess_hw_pass: usize,
    chess_hw_fail: usize,
    chess_hw_error: usize,
    // Compiler comparison (Peano vs Chess)
    chess_compared: usize,
    chess_correct: usize,
    chess_peano_bugs: usize,
    chess_chess_bugs: usize,
    chess_emu_bugs: usize,
    // aiesimulator
    sim_attempted: usize,
    sim_correct: usize,
    sim_wrong: usize,
    sim_error: usize,
    // Differential (hw-only: Peano vs Chess on HW)
    both_pass: usize,
    peano_only: usize,
    chess_only: usize,
    both_fail: usize,
}

impl RunStats {
    fn record_emu_outcome(&mut self, outcome: &TestOutcome) {
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

    fn record_hw_validation(&mut self, hv: &HardwareValidation) {
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

    fn record_peano_hw(&mut self, hw: &HwRunResult) {
        self.peano_hw_attempted += 1;
        if hw.passed {
            self.peano_hw_pass += 1;
        } else if hw.label.starts_with("FAIL") {
            self.peano_hw_fail += 1;
        } else {
            self.peano_hw_error += 1;
        }
    }

    fn record_chess_hw(&mut self, hw: &HwRunResult) {
        self.chess_hw_attempted += 1;
        if hw.passed {
            self.chess_hw_pass += 1;
        } else if hw.label.starts_with("FAIL") {
            self.chess_hw_fail += 1;
        } else {
            self.chess_hw_error += 1;
        }
    }

    fn record_compiler_diagnosis(&mut self, diag: CompilerDiagnosis) {
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

    fn record_aiesim(&mut self, label: &str) {
        self.sim_attempted += 1;
        if label.starts_with("PASS") {
            self.sim_correct += 1;
        } else if label.starts_with("FAIL") {
            self.sim_wrong += 1;
        } else {
            self.sim_error += 1;
        }
    }

    fn record_differential(&mut self, peano_pass: bool, chess_pass: bool) {
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
    fn print_summary(&self, total: usize, hw_only: bool) {
        println!("\n{:=<60}", "");

        if hw_only {
            println!("=== HARDWARE-ONLY SUMMARY ===");
            println!("Total:            {}", total);
            println!("Skipped:          {} (no manifest or insts.bin)", self.skipped);
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
                println!("Both fail:        {} (test/manifest issue?)", self.both_fail);
            }
        }
    }
}

/// Batch-compile all tests with their primary compiler.
///
/// Each test is built with its best-available compiler:
/// - Chess-only tests (REQUIRES: chess) -> Chess compiler
/// Create XclbinTest entries from a successful build result.
///
/// If the output directory contains a single xclbin, creates one XclbinTest
/// with the test's original name. If multiple xclbin files are present
/// (multi-variant tests like matrix_multiplication_using_cascade), creates
/// one XclbinTest per xclbin with names like "test_name/variant_name".
fn collect_xclbin_tests(
    test: &NpuTestSource,
    output_dir: &Path,
    _primary_result: &BuildResult,
    results: &mut Vec<XclbinTest>,
) {
    let all = find_all_xclbin_results(output_dir, &test.build_steps);

    if all.len() <= 1 {
        // Single xclbin (common case) -- use original name
        if let Some((ref xclbin, ref insts, _)) = all.first() {
            let mut t = XclbinTest::from_path(xclbin);
            t.name = test.name.clone();
            t.manifest = test.manifest.clone();
            t.insts_path = insts.clone();
            results.push(t);
        }
    } else {
        // Multiple xclbins -- create sub-tests with variant names
        for (xclbin, insts, variant) in &all {
            let mut t = XclbinTest::from_path(xclbin);
            t.name = format!("{}/{}", test.name, variant);
            // Manifest applies to the test as a whole; individual variants
            // may not match the manifest's expectations, so only attach it
            // if there's no variant-specific manifest available.
            t.manifest = test.manifest.clone();
            t.insts_path = insts.clone();
            results.push(t);
        }
    }
}

/// - Everything else -> Peano compiler
/// - npu2 tests are skipped (wrong hardware)
///
/// When Chess is unavailable, chess-only tests are reported as skipped
/// rather than failing the run.
///
/// Returns a Vec of XclbinTest ready for the emulator execution pipeline.
fn batch_build_primary(
    build_env: &BuildEnv,
    tests: &[&NpuTestSource],
    chess_available: bool,
    nice_level: i32,
) -> Vec<XclbinTest> {
    let mut skipped_npu2 = 0;
    let mut skipped_no_chess = 0;
    let mut skipped_no_steps = 0;

    // Categorize tests into build lists
    let mut peano_tests: Vec<&NpuTestSource> = Vec::new();
    let mut chess_tests: Vec<&NpuTestSource> = Vec::new();

    for test in tests.iter() {
        if test.requires_npu2() {
            skipped_npu2 += 1;
            continue;
        }
        if test.build_steps.is_empty() {
            skipped_no_steps += 1;
            continue;
        }
        if test.requires_chess() {
            if chess_available {
                chess_tests.push(test);
            } else {
                skipped_no_chess += 1;
            }
            continue;
        }
        peano_tests.push(test);
    }

    let total_buildable = peano_tests.len() + chess_tests.len();
    let total_skipped = skipped_npu2 + skipped_no_chess + skipped_no_steps;

    println!("\n=== PRIMARY BUILD PHASE ===");
    println!("{} buildable ({} Peano, {} Chess-only), {} skipped (npu2: {}, no chess: {}, no steps: {})",
        total_buildable, peano_tests.len(), chess_tests.len(),
        total_skipped, skipped_npu2, skipped_no_chess, skipped_no_steps);

    if total_buildable == 0 {
        return Vec::new();
    }

    let build_start = Instant::now();
    let mut results = Vec::new();
    let mut build_idx = 0;

    // Build Peano tests
    for test in &peano_tests {
        build_idx += 1;
        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("build/peano")
            .join(&test.name);

        let test_start = Instant::now();
        match build_env.build_npu_test(
            test,
            &output_dir,
            &BuildOpts {
                use_chess: false,
                gen_sim: false,
                device: String::new(),
                nice: if nice_level > 0 { Some(nice_level) } else { None },
            },
        ) {
            Ok(result) => {
                let cached = result.build_log == "(cached)";
                let elapsed = test_start.elapsed();
                let label = if cached { "cached" } else { "built" };
                println!("[{:2}/{}] {:40} Peano {} ({:.1}s)",
                    build_idx, total_buildable,
                    &test.name[..test.name.len().min(40)],
                    label, elapsed.as_secs_f64());

                collect_xclbin_tests(test, &output_dir, &result, &mut results);
            }
            Err(e) => {
                let elapsed = test_start.elapsed();
                let msg = e.lines().next().unwrap_or(&e);
                println!("[{:2}/{}] {:40} Peano FAILED ({:.1}s): {}",
                    build_idx, total_buildable,
                    &test.name[..test.name.len().min(40)],
                    elapsed.as_secs_f64(),
                    &msg[..msg.len().min(60)]);
            }
        }
    }

    // Build chess-only tests
    for test in &chess_tests {
        build_idx += 1;
        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("build/chess")
            .join(&test.name);

        let test_start = Instant::now();
        match build_env.build_npu_test(
            test,
            &output_dir,
            &BuildOpts {
                use_chess: true,
                gen_sim: false,
                device: String::new(),
                nice: if nice_level > 0 { Some(nice_level) } else { None },
            },
        ) {
            Ok(result) => {
                let cached = result.build_log == "(cached)";
                let elapsed = test_start.elapsed();
                let label = if cached { "cached" } else { "built" };
                println!("[{:2}/{}] {:40} Chess {} ({:.1}s)",
                    build_idx, total_buildable,
                    &test.name[..test.name.len().min(40)],
                    label, elapsed.as_secs_f64());

                collect_xclbin_tests(test, &output_dir, &result, &mut results);
            }
            Err(e) => {
                let elapsed = test_start.elapsed();
                let msg = e.lines().next().unwrap_or(&e);
                println!("[{:2}/{}] {:40} Chess FAILED ({:.1}s): {}",
                    build_idx, total_buildable,
                    &test.name[..test.name.len().min(40)],
                    elapsed.as_secs_f64(),
                    &msg[..msg.len().min(60)]);
            }
        }
    }

    let build_elapsed = build_start.elapsed();
    println!("Primary builds: {}/{} succeeded ({:.1}s)\n",
        results.len(), total_buildable, build_elapsed.as_secs_f64());

    results
}

/// Batch-compile Peano-primary tests with Chess for comparison.
///
/// Only builds tests that were already built with Peano (skips chess-only
/// tests since they already used Chess as their primary compiler, and
/// skips npu2 tests). Returns a map of test_name -> ChessBuildArtifacts.
///
/// Builds are cached: if xclbin exists and is newer than the entry file,
/// the build is skipped.
fn batch_build_chess_comparison(
    build_env: &BuildEnv,
    tests: &[&NpuTestSource],
    gen_sim: bool,
    nice_level: i32,
) -> std::collections::HashMap<String, ChessBuildArtifacts> {
    let mut artifacts = std::collections::HashMap::new();

    let buildable: Vec<&&NpuTestSource> = tests.iter()
        .filter(|t| {
            // Only build tests that used Peano as primary (not chess-only,
            // not npu2, and have build steps)
            !t.requires_npu2() && !t.requires_chess() && !t.build_steps.is_empty()
        })
        .collect();

    if buildable.is_empty() {
        println!("No Peano-primary tests to build Chess comparison for.");
        return artifacts;
    }

    println!("\n=== CHESS COMPARISON BUILD ({} tests) ===", buildable.len());
    let build_start = Instant::now();

    for (idx, test) in buildable.iter().enumerate() {
        let chess_output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("build/chess")
            .join(&test.name);

        let test_start = Instant::now();
        match build_env.build_npu_test(
            test,
            &chess_output_dir,
            &BuildOpts {
                use_chess: true,
                gen_sim,
                device: String::new(),
                nice: if nice_level > 0 { Some(nice_level) } else { None },
            },
        ) {
            Ok(result) => {
                let cached = result.build_log == "(cached)";
                let elapsed = test_start.elapsed();
                let label = if cached { "cached" } else { "built" };
                println!("[{:2}/{}] {:40} {} ({:.1}s)",
                    idx + 1, buildable.len(),
                    &test.name[..test.name.len().min(40)],
                    label, elapsed.as_secs_f64());
                artifacts.insert(test.name.clone(), ChessBuildArtifacts {
                    xclbin: result.xclbin,
                    insts: result.insts,
                    prj_dir: result.prj_dir,
                });
            }
            Err(e) => {
                let elapsed = test_start.elapsed();
                let msg = e.lines().next().unwrap_or(&e);
                println!("[{:2}/{}] {:40} FAILED ({:.1}s): {}",
                    idx + 1, buildable.len(),
                    &test.name[..test.name.len().min(40)],
                    elapsed.as_secs_f64(),
                    &msg[..msg.len().min(60)]);
            }
        }
    }

    let build_elapsed = build_start.elapsed();
    println!("Chess builds: {}/{} succeeded ({:.1}s)\n",
        artifacts.len(), buildable.len(), build_elapsed.as_secs_f64());

    artifacts
}

/// Run aiesimulator on a Chess-built .prj directory and compare output.
///
/// Returns a status label for display ("PASS (N/N)", "FAIL", "ERROR: ...", etc.).
fn run_aiesim_comparison(
    tools: &AieTools,
    test: &XclbinTest,
    prj_dir: &Path,
) -> String {
    let manifest = match test.manifest.as_ref() {
        Some(m) => m,
        None => return "SKIP (no manifest)".to_string(),
    };

    // Prepare input data from manifest
    let mut input_data = Vec::new();
    for (buf_name, buf_def) in &manifest.buffers {
        if buf_name == "output" {
            continue;
        }
        if let Some(data) = manifest.generate_input(buf_name) {
            // Write as binary file with the buffer name
            input_data.push((format!("{}.bin", buf_name), data));
        } else {
            // Write zero-filled buffer
            let elem_type = ElementType::from_str(&buf_def.element_type)
                .unwrap_or(ElementType::I32);
            let size_bytes = buf_def.size * elem_type.byte_size();
            input_data.push((format!("{}.bin", buf_name), vec![0u8; size_bytes]));
        }
    }

    // Run simulation with generous timeout (aiesimulator is slow)
    let sim_result = match aiesimulator::run_simulation(
        tools, prj_dir, &input_data, 1_000_000,
    ) {
        Ok(r) => r,
        Err(e) => return format!("ERROR: {}", e),
    };

    if sim_result.exit_code != 0 {
        // Truncate stderr for display
        let stderr_preview: String = sim_result.stderr
            .lines()
            .take(3)
            .collect::<Vec<_>>()
            .join(" | ");
        return format!("ERROR (exit {}): {}", sim_result.exit_code, stderr_preview);
    }

    // Read output and compare against expected
    let sim_output = match aiesimulator::read_output_data(&sim_result) {
        Ok(data) => data,
        Err(e) => return format!("ERROR reading output: {}", e),
    };

    // Compare against manifest expected values
    let output_buf = match manifest.get_output() {
        Some(b) => b,
        None => return "SKIP (no output buffer)".to_string(),
    };
    let elem_type = match ElementType::from_str(&output_buf.element_type) {
        Some(t) => t,
        None => return format!("SKIP (unknown type: {})", output_buf.element_type),
    };

    let input_values = generate_input_values(manifest);
    let reference_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");
    let ref_dir = if reference_dir.exists() { Some(reference_dir.as_path()) } else { None };

    let expected = match manifest.generate_expected(&input_values, ref_dir) {
        Some(e) => e,
        None => return "SKIP (cannot generate expected)".to_string(),
    };

    let actual = read_values(&sim_output, elem_type);
    let total = expected.len().min(actual.len());
    let correct = (0..total).filter(|&i| actual[i] == expected[i]).count();

    if correct == total && total > 0 {
        format!("PASS ({}/{})", correct, total)
    } else {
        format!("FAIL ({}/{})", correct, total)
    }
}

/// Run an xclbin on real NPU hardware and validate output against manifest.
///
/// Returns `Ok((label, raw_output))` on successful execution (even if
/// validation fails), or `Err(message)` if the hardware run itself failed.
fn run_on_hw_and_validate(
    manifest: &TestManifest,
    xclbin_path: &Path,
    insts_path: &Path,
) -> Result<(String, Vec<u8>), String> {
    // Timeout 0 = wait forever. Hardware TDR handles actual hangs.
    let result = npu_runner::run_on_npu(manifest, xclbin_path, insts_path, 0)
        .map_err(|e| {
            // Truncate to first line -- stderr from GLIBCXX errors can be 10+ lines
            let msg = e.to_string();
            msg.lines().next().unwrap_or(&msg).to_string()
        })?;

    // Validate output against manifest expected values
    let label = (|| -> Option<String> {
        let output_buf = manifest.get_output()?;
        let elem_type = ElementType::from_str(&output_buf.element_type)?;
        let input_values = generate_input_values(manifest);
        let reference_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/npu-outputs");
        let ref_dir = if reference_dir.exists() { Some(reference_dir.as_path()) } else { None };
        let expected = manifest.generate_expected(&input_values, ref_dir)?;

        let actual = read_values(&result.output, elem_type);
        let total = expected.len().min(actual.len());
        let correct = (0..total).filter(|&i| actual[i] == expected[i]).count();

        if correct == total && total > 0 {
            Some(format!("PASS ({}/{})", correct, total))
        } else {
            Some(format!("FAIL ({}/{})", correct, total))
        }
    })().unwrap_or_else(|| "DONE (no validation)".to_string());

    Ok((label, result.output))
}

/// Run an xclbin on real NPU hardware, display the result, and return it.
///
/// Deduplicates the Peano HW / Chess HW run-and-display logic. In compact
/// mode (hw-only), uses `print!` for single-line output. In normal mode,
/// uses `println!` with indentation.
fn run_hw_and_print(
    manifest: &TestManifest,
    xclbin: &Path,
    insts: &Path,
    prefix: &str,
    compact: bool,
) -> HwRunResult {
    let hw_start = Instant::now();
    match run_on_hw_and_validate(manifest, xclbin, insts) {
        Ok((label, output)) => {
            let elapsed = hw_start.elapsed().as_secs_f64();
            let passed = label.starts_with("PASS");
            if compact {
                print!("{}: {:18} ({:.1}s)", prefix, label, elapsed);
            } else {
                println!("      {}: {} ({:.1}s)", prefix, label, elapsed);
            }
            HwRunResult { label, output, passed, elapsed_secs: elapsed }
        }
        Err(e) => {
            let elapsed = hw_start.elapsed().as_secs_f64();
            // Extract the meaningful error: look for "ERROR:" lines from
            // npu-runner stderr, falling back to the NpuRunError message.
            let meaningful = e.lines()
                .filter(|l| l.contains("ERROR:") || l.starts_with("Kernel timed out"))
                .last()
                .unwrap_or_else(|| e.lines().next().unwrap_or(&e));
            let label = format!("ERROR ({})", &meaningful[..meaningful.len().min(60)]);
            if compact {
                print!("{}: {}", prefix, label);
            } else {
                println!("      {}: {}", prefix, label);
            }
            HwRunResult { label, output: Vec::new(), passed: false, elapsed_secs: elapsed }
        }
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error"),
    )
    .init();

    let runner_config = RunnerConfig::load();
    runner_config.print_active_overrides();
    let opts = parse_args(&runner_config);

    let config = xdna_emu::config::Config::get();
    let mlir_aie_path = PathBuf::from(config.mlir_aie_path());

    // Discover aietools (always attempt -- needed for auto Chess detection)
    let aietools = AieTools::discover(config);
    let chess_available = opts.chess_build && aietools.as_ref()
        .map_or(false, |t| t.xchesscc.is_some());

    if let Some(ref tools) = aietools {
        println!("aietools: {}", tools.summary());
    }

    // Validate explicitly-requested features that need aietools
    if opts.elfanalyze {
        if aietools.as_ref().map_or(true, |t| t.elfanalyzer.is_none()) {
            eprintln!("Warning: --elfanalyze requested but elfanalyzer not found in aietools");
        }
    }
    if (opts.aiesim || opts.unit_tests) && aietools.as_ref().map_or(true, |t| t.aiesimulator.is_none()) {
        eprintln!("Error: --aiesim/--unit-tests requires aiesimulator from aietools");
        std::process::exit(1);
    }

    // Discover build environment.
    // Source-driven builds (default) always need it. --no-build only needs
    // it for Chess/aiesim/unit-tests.
    let needs_build_env = !opts.no_build || chess_available || opts.aiesim || opts.unit_tests;
    let build_env = if needs_build_env {
        match BuildEnv::discover(config) {
            Ok(env) => {
                println!("build env: {}", env.summary());
                Some(env)
            }
            Err(e) => {
                if !opts.no_build {
                    eprintln!("Error: source-driven builds require build environment: {}", e);
                    eprintln!("Use --no-build to skip compilation and run pre-built tests.");
                    std::process::exit(1);
                }
                // --no-build without chess/aiesim: build_env not critical
                eprintln!("Warning: build environment not available: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Hardware validation: auto-detected by default, graceful fallback.
    // If opts.hw is true (default), check NPU availability and degrade gracefully.
    // If opts.hw is false (--no-hw), skip entirely.
    let hw_available = if opts.hw {
        if !npu_runner::npu_available() {
            println!("hw: NPU not available, hardware validation disabled");
            false
        } else {
            match npu_runner::runner_binary() {
                Some(path) => {
                    println!("hw: NPU available, npu-runner at {}", path.display());
                    true
                }
                None => {
                    println!("hw: NPU available but npu-runner not built, hardware validation disabled");
                    println!("    Build with: cd tools/npu-runner && cmake -B build && cmake --build build");
                    false
                }
            }
        }
    } else {
        false
    };

    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mlir-aie-extracted/manifests");

    let reference_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");

    let npu_output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");

    // ====================================================================
    // DISCOVERY AND BUILD
    //
    // Two modes:
    //   Default:    source tree -> build with Peano -> XclbinSuite
    //   --no-build: build tree  -> XclbinSuite (legacy, for quick re-runs)
    // ====================================================================

    // Source tests are stored here so they outlive the if/else block.
    // Needed for Chess builds which reference the source RUN lines.
    let mut source_tests_storage: Vec<NpuTestSource>;

    let (mut suite, tests, total_discovered) = if opts.no_build {
        // Legacy path: discover pre-built tests from cmake build tree
        let npu_xrt_path = config.npu_xrt_test_dir();
        if !npu_xrt_path.exists() {
            eprintln!("Error: npu-xrt build output not found at {}", npu_xrt_path.display());
            eprintln!("Either build mlir-aie with tests, or omit --no-build to build from source.");
            std::process::exit(1);
        }

        source_tests_storage = Vec::new();
        println!("Discovering pre-built tests in {}...", npu_xrt_path.display());

        let suite = match XclbinSuite::discover(&npu_xrt_path) {
            Ok(s) => {
                let mut s = s.with_max_cycles(opts.max_cycles);
                if reference_dir.exists() {
                    s = s.with_reference_dir(reference_dir.clone());
                }
                if npu_output_dir.exists() {
                    s = s.with_npu_output_dir(npu_output_dir.clone());
                }
                if manifest_path.exists() {
                    println!("Loading manifests from {}...", manifest_path.display());
                    s.with_manifest_dir(manifest_path.clone())
                } else {
                    s
                }
            }
            Err(e) => {
                eprintln!("Failed to discover tests: {}", e);
                std::process::exit(1);
            }
        };

        let all_tests: Vec<_> = suite.tests().to_vec();
        let discovered = all_tests.len();
        let tests: Vec<_> = all_tests.into_iter()
            .filter(|t| matches_filter(&t.name, &opts.filters))
            .collect();

        (suite, tests, discovered)
    } else {
        // Source-driven path: discover from source tree, build, create suite
        println!("Discovering tests from source tree...");
        source_tests_storage = npu_test::discover(&mlir_aie_path);

        // Load manifests onto source tests before building, so they
        // transfer to the XclbinTest objects created by batch_build_peano.
        if manifest_path.exists() {
            npu_test::load_manifests(&mut source_tests_storage, &manifest_path);
        }

        let total_source = source_tests_storage.len();
        println!("Found {} tests in source tree", total_source);

        // Filter source tests
        let filtered: Vec<&NpuTestSource> = source_tests_storage.iter()
            .filter(|t| matches_filter(&t.name, &opts.filters))
            .collect();

        // Primary build phase: Peano for most tests, Chess for chess-only
        let env = build_env.as_ref().unwrap();
        let built_tests = batch_build_primary(env, &filtered, chess_available, opts.build_nice);

        // Create suite from built tests
        let mut suite = XclbinSuite::from_tests(built_tests)
            .with_max_cycles(opts.max_cycles);
        if reference_dir.exists() {
            suite = suite.with_reference_dir(reference_dir.clone());
        }
        if npu_output_dir.exists() {
            suite = suite.with_npu_output_dir(npu_output_dir.clone());
        }
        // Load manifests from suite (for tests that got manifests from source
        // discovery, they're already attached; this handles the --no-build path)
        if manifest_path.exists() {
            suite = suite.with_manifest_dir(manifest_path.clone());
        }

        let tests: Vec<_> = suite.tests().to_vec();
        (suite, tests, total_source)
    };

    let total = tests.len();
    if !opts.filters.is_empty() {
        println!("Filter: {:?} -> {}/{} tests selected\n", opts.filters, total, total_discovered);
    } else {
        println!("{} tests ready to run\n", total);
    }

    if total == 0 && !opts.unit_tests {
        println!("No tests matched the filter.");
        return;
    }

    // === CHESS COMPARISON BUILD ===
    // Build Peano-primary tests with Chess too for dual-compiler comparison.
    // Chess-only tests are already built as primaries (no need to rebuild).
    let chess_builds = if chess_available {
        if let Some(ref env) = build_env {
            if !source_tests_storage.is_empty() {
                let filtered_source: Vec<&NpuTestSource> = source_tests_storage.iter()
                    .filter(|t| matches_filter(&t.name, &opts.filters))
                    .collect();
                batch_build_chess_comparison(env, &filtered_source, opts.aiesim, opts.build_nice)
            } else {
                std::collections::HashMap::new()
            }
        } else {
            std::collections::HashMap::new()
        }
    } else {
        std::collections::HashMap::new()
    };

    // === EXECUTE PHASE ===
    // Run emulator tests (skip in hw-only mode). Each element is Some(result)
    // when the emulator ran, or None in hw-only mode.
    let emu_results: Vec<Option<TestResult>> = if opts.hw_only {
        if !hw_available {
            eprintln!("--hw-only requires NPU hardware, but none was detected");
            std::process::exit(1);
        }
        println!("=== HARDWARE-ONLY MODE ===\n");
        (0..total).map(|_| None).collect()
    } else if opts.jobs > 1 {
        run_parallel(&suite, &tests, opts.jobs).into_iter().map(Some).collect()
    } else {
        run_sequential(&suite, &tests).into_iter().map(Some).collect()
    };

    // === UNIFIED PER-TEST LOOP ===
    // All execution modes (emulator, HW, Chess, aiesim) are handled in a
    // single pass. hw-only mode skips the emulator display and uses compact
    // single-line output for HW results.
    let mut stats = RunStats::default();

    for (idx, test) in tests.iter().enumerate() {
        let emu = &emu_results[idx];
        let compact = emu.is_none(); // hw-only mode: compact single-line output

        // --- Emulator display (skip in hw-only) ---
        if let Some(ref r) = emu {
            println!("{}", format_result(r, total));
            stats.record_emu_outcome(&r.outcome);
            if let Some(ref hv) = r.hw_validation {
                stats.record_hw_validation(hv);
            }
            if let TestOutcome::UnknownOpcode { ref details, .. } = r.outcome {
                suite.record_unknown(details.clone(), &r.name);
            }

            // Verbose mode: print full expected vs actual comparison
            if opts.verbose {
                if let Some(ref manifest) = test.manifest {
                    if let Some(ref output) = r.raw_output {
                        let show = matches!(&r.outcome,
                            TestOutcome::ValidationFail { .. } |
                            TestOutcome::ExpectedFail { .. } |
                            TestOutcome::UnexpectedPass { .. } |
                            TestOutcome::Pass { correct: Some(_), .. }
                        );
                        if show {
                            print_verbose_comparison(output, manifest);
                        }
                    }
                }
            }
        }

        // --- Skip gates ---
        let skip_hw = emu.as_ref().map_or(false, |r|
            matches!(&r.outcome, TestOutcome::Platform{..} | TestOutcome::Skipped{..}));

        let manifest = test.manifest.as_ref();
        let insts_path = test.find_insts_bin();

        // In hw-only mode, skip tests without manifest or insts
        if compact {
            if manifest.is_none() || insts_path.is_none() {
                let reason = if manifest.is_none() { "no manifest" } else { "no insts.bin" };
                println!("[{:2}/{}] {:45} SKIP ({})",
                    idx + 1, total, &test.name[..test.name.len().min(45)], reason);
                stats.skipped += 1;
                continue;
            }
            // Print compact header (results appended on same line)
            print!("[{:2}/{}] {:35} ",
                idx + 1, total, &test.name[..test.name.len().min(35)]);
        }

        // --- Peano HW ---
        let peano_hw: Option<HwRunResult> = if hw_available && !skip_hw {
            if let (Some(m), Some(ref insts)) = (manifest, &insts_path) {
                let prefix = if compact { "peano" } else { "peano-hw" };
                let hw = run_hw_and_print(m, &test.xclbin_path, insts, prefix, compact);
                stats.record_peano_hw(&hw);
                Some(hw)
            } else {
                None
            }
        } else {
            None
        };

        // --- elfanalyzer ---
        if opts.elfanalyze && !compact {
            if let Some(ref tools) = aietools {
                run_elfanalyzer(tools, test, &mlir_aie_path);
            }
        }

        // --- Chess emulator (sequential, usually disabled by config) ---
        let chess_emu_output: Option<Vec<u8>> = if chess_available && opts.chess_emulator && !compact {
            if let Some(chess_artifacts) = chess_builds.get(&test.name) {
                let mut chess_test = XclbinTest::from_path(&chess_artifacts.xclbin);
                chess_test.manifest = test.manifest.clone();
                let (chess_outcome, chess_raw, _hv) =
                    suite.run_single_with_hw_validation(&chess_test);

                let label = match &chess_outcome {
                    TestOutcome::Pass { correct, total, .. } => {
                        if let (Some(c), Some(t)) = (correct, total) {
                            format!("PASS ({}/{})", c, t)
                        } else {
                            "PASS".to_string()
                        }
                    }
                    TestOutcome::ValidationFail { correct, total, .. } =>
                        format!("VALIDATION FAIL ({}/{})", correct, total),
                    TestOutcome::Fail { message, .. } => format!("FAIL: {}", message),
                    TestOutcome::UnknownOpcode { .. } => "UNKNOWN OPCODE".to_string(),
                    TestOutcome::Timeout { .. } => "TIMEOUT".to_string(),
                    TestOutcome::LoadError { message } => format!("LOAD ERROR: {}", message),
                    other => format!("{:?}", other),
                };
                println!("      chess: {}", label);
                chess_raw
            } else {
                None
            }
        } else {
            None
        };

        // --- Chess HW ---
        let chess_hw: Option<HwRunResult> = if chess_available && !skip_hw {
            if let Some(chess_artifacts) = chess_builds.get(&test.name) {
                if let Some(ref chess_insts) = chess_artifacts.insts {
                    if let Some(m) = manifest {
                        // Run Chess HW in hw-only mode or when chess_hardware is enabled
                        if opts.chess_hardware || compact {
                            let prefix = if compact { "chess" } else { "chess-hw" };
                            if compact { print!("  "); }
                            let hw = run_hw_and_print(
                                m, &chess_artifacts.xclbin, chess_insts, prefix, compact,
                            );
                            stats.record_chess_hw(&hw);
                            Some(hw)
                        } else { None }
                    } else { None }
                } else { None }
            } else { None }
        } else { None };

        // --- Differential (when both compilers ran on HW) ---
        if let (Some(ref p), Some(ref c)) = (&peano_hw, &chess_hw) {
            stats.record_differential(p.passed, c.passed);
        }

        // --- Compiler comparison ---
        if chess_available {
            if chess_builds.contains_key(&test.name) {
                if let Some(m) = manifest {
                    let diag = (|| -> Option<CompilerDiagnosis> {
                        let output_buf = m.get_output()?;
                        let elem_type = ElementType::from_str(&output_buf.element_type)?;
                        let input_values = generate_input_values(m);
                        let reference_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                            .join("tests/npu-outputs");
                        let ref_dir = if reference_dir.exists() {
                            Some(reference_dir.as_path())
                        } else {
                            None
                        };
                        let expected = m.generate_expected(&input_values, ref_dir)?;

                        let comparison = CompilerComparison {
                            test_name: test.name.clone(),
                            peano_emu: emu.as_ref().and_then(|r| r.raw_output.clone()),
                            chess_emu: chess_emu_output.clone(),
                            peano_hw: peano_hw.as_ref().map(|h| h.output.clone()),
                            chess_hw: chess_hw.as_ref().map(|h| h.output.clone()),
                            manifest_expected: Some(expected),
                        };

                        let d = if comparison.peano_hw.is_some() || comparison.chess_hw.is_some() {
                            comparison.classify_full(elem_type)
                        } else {
                            comparison.classify(elem_type)
                        };
                        if d != CompilerDiagnosis::Incomplete && !compact {
                            println!("      compiler comparison: {}", d);
                        }
                        Some(d)
                    })();

                    if let Some(d) = diag {
                        stats.record_compiler_diagnosis(d);
                    }
                }
            }
        }

        // --- aiesimulator (needs Chess build's .prj, skip in hw-only) ---
        if opts.aiesim && !compact {
            if let Some(ref tools) = aietools {
                if let Some(prj) = chess_builds.get(&test.name).and_then(|a| a.prj_dir.as_ref()) {
                    let sim_label = run_aiesim_comparison(tools, test, prj);
                    println!("      aiesim: {}", sim_label);
                    stats.record_aiesim(&sim_label);
                }
            }
        }

        // Close compact (hw-only) line
        if compact {
            println!();
        }
    }

    // === UNIT TESTS ===
    // Self-contained chess_compiler_tests_aie2 tests, built and run through
    // aiesimulator. These use tile memory writes and lock-based sync, not
    // DDR/shim DMA. Executed as a separate section after NPU tests.
    let mut unit_discovered = 0;
    let mut unit_built = 0;
    let mut unit_build_failed = 0;
    let mut unit_skipped = 0;
    let mut unit_sim_pass = 0;
    let mut unit_sim_fail = 0;
    let mut unit_sim_error = 0;

    if opts.unit_tests {
        println!("\n{:=<60}", "");
        println!("=== UNIT TESTS (chess_compiler_tests_aie2) ===\n");

        let unit_tests = unit_test::discover(&mlir_aie_path);
        unit_discovered = unit_tests.len();

        if unit_tests.is_empty() {
            println!("No unit tests found.");
        } else {
            // Apply name filter (same filters as NPU tests)
            let filtered: Vec<_> = unit_tests.iter()
                .filter(|t| matches_filter(&t.name, &opts.filters))
                .collect();

            let filtered_count = filtered.len();
            if !opts.filters.is_empty() {
                println!("Filter: {:?} -> {}/{} unit tests selected",
                    opts.filters, filtered_count, unit_discovered);
            } else {
                println!("Found {} unit tests", unit_discovered);
            }

            // Build phase
            if let Some(ref env) = build_env {
                println!("\n--- Unit Test Build Phase ({} tests) ---", filtered_count);
                let build_start = Instant::now();

                // Collect build results: (test, prj_dir) for successful builds
                let mut build_results: Vec<(&unit_test::UnitTest, PathBuf)> = Vec::new();

                for (idx, test) in filtered.iter().enumerate() {
                    if let Some(ref reason) = test.skip_reason {
                        println!("[{:2}/{}] {:40} SKIP ({})",
                            idx + 1, filtered_count,
                            &test.name[..test.name.len().min(40)],
                            reason);
                        unit_skipped += 1;
                        continue;
                    }

                    let unit_output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("build/unit_tests")
                        .join(&test.name);

                    let test_start = Instant::now();
                    match env.build_unit_test(
                        test,
                        &unit_output_dir,
                        if opts.build_nice > 0 { Some(opts.build_nice) } else { None },
                    ) {
                        Ok(result) => {
                            let cached = result.build_log == "(cached)";
                            let elapsed = test_start.elapsed();
                            let label = if cached { "cached" } else { "built" };
                            println!("[{:2}/{}] {:40} {} ({:.1}s)",
                                idx + 1, filtered_count,
                                &test.name[..test.name.len().min(40)],
                                label, elapsed.as_secs_f64());
                            unit_built += 1;
                            build_results.push((test, result.prj_dir));
                        }
                        Err(e) => {
                            let elapsed = test_start.elapsed();
                            let msg = e.lines().next().unwrap_or(&e);
                            println!("[{:2}/{}] {:40} FAILED ({:.1}s): {}",
                                idx + 1, filtered_count,
                                &test.name[..test.name.len().min(40)],
                                elapsed.as_secs_f64(),
                                &msg[..msg.len().min(80)]);
                            unit_build_failed += 1;
                        }
                    }
                }

                let build_elapsed = build_start.elapsed();
                println!("Unit test builds: {}/{} succeeded ({:.1}s)",
                    unit_built, filtered_count - unit_skipped, build_elapsed.as_secs_f64());

                // Simulation phase (run through aiesimulator)
                if opts.unit_tests_aiesim && !build_results.is_empty() {
                    if let Some(ref tools) = aietools {
                        println!("\n--- Unit Test Simulation Phase ({} tests) ---",
                            build_results.len());

                        for (idx, (test, prj_dir)) in build_results.iter().enumerate() {
                            eprint!("\r[{:2}/{}] {}...",
                                idx + 1, build_results.len(),
                                &test.name[..test.name.len().min(40)]);
                            io::stderr().flush().unwrap();

                            match aiesimulator::run_unit_simulation(
                                tools,
                                prj_dir,
                                opts.unit_tests_aiesim_timeout,
                            ) {
                                Ok(result) => {
                                    let label = if result.passed {
                                        unit_sim_pass += 1;
                                        "PASS"
                                    } else {
                                        unit_sim_fail += 1;
                                        "FAIL"
                                    };
                                    eprint!("\r{:60}\r", "");
                                    println!("[{:2}/{}] {:40} {} ({:.1}s)",
                                        idx + 1, build_results.len(),
                                        &test.name[..test.name.len().min(40)],
                                        label, result.wall_time_secs);

                                    // In verbose mode, show stdout for failures
                                    if opts.verbose && !result.passed {
                                        for line in result.stdout.lines().take(20) {
                                            println!("      {}", line);
                                        }
                                    }
                                }
                                Err(e) => {
                                    unit_sim_error += 1;
                                    eprint!("\r{:60}\r", "");
                                    let msg = e.lines().next().unwrap_or(&e);
                                    println!("[{:2}/{}] {:40} ERROR: {}",
                                        idx + 1, build_results.len(),
                                        &test.name[..test.name.len().min(40)],
                                        &msg[..msg.len().min(60)]);
                                }
                            }
                        }
                        eprint!("\r{:60}\r", "");
                        io::stderr().flush().unwrap();
                    }
                }
            }
        }
    }

    // === SUMMARY ===
    stats.print_summary(total, opts.hw_only);

    // Unit test summary (separate counters, not part of RunStats)
    if unit_discovered > 0 {
        println!("\n=== UNIT TESTS ===");
        println!("Discovered:       {}", unit_discovered);
        println!("Skipped:          {}", unit_skipped);
        println!("Built:            {}", unit_built);
        println!("Build Failed:     {}", unit_build_failed);
        if unit_sim_pass + unit_sim_fail + unit_sim_error > 0 {
            let unit_sim_total = unit_sim_pass + unit_sim_fail + unit_sim_error;
            println!("Simulated:        {}", unit_sim_total);
            println!("Sim Pass:         {}", unit_sim_pass);
            println!("Sim Fail:         {}", unit_sim_fail);
            println!("Sim Error:        {}", unit_sim_error);
        }
    }

    // Show unknown opcodes if any
    let collector = suite.collector();
    let unknowns = collector.by_impact();
    if !unknowns.is_empty() {
        println!("\n=== UNKNOWN OPCODES (by impact) ===");
        for entry in unknowns.iter().take(10) {
            let op = &entry.first;
            let mnemonic = op.mnemonic.as_deref().unwrap_or("unknown");
            println!("  {:?} opcode 0x{:04X} '{}' (hits: {}, tests: {})",
                     op.slot, op.opcode, mnemonic, entry.count, entry.tests.len());
        }
    }
}

/// Run tests sequentially (original behavior, -j 1).
fn run_sequential(suite: &XclbinSuite, tests: &[xdna_emu::testing::xclbin_suite::XclbinTest]) -> Vec<TestResult> {
    let total = tests.len();
    let mut results = Vec::with_capacity(total);

    for (i, test) in tests.iter().enumerate() {
        // Print progress inline (overwritten by final output)
        eprint!("\r[{:2}/{}] {}...", i + 1, total, &test.name[..test.name.len().min(40)]);
        io::stderr().flush().unwrap();

        let elf_count = test.find_elf_files().len();
        let embedded_count = test.count_embedded_cores();
        let has_npu = test.find_insts_bin().is_some();
        let (outcome, raw_output, hw_validation) = suite.run_single_with_hw_validation(test);

        results.push(TestResult {
            idx: i,
            name: test.name.clone(),
            elf_count,
            embedded_count,
            has_npu,
            outcome,
            raw_output,
            hw_validation,
        });
    }
    eprint!("\r{:60}\r", ""); // Clear progress line
    io::stderr().flush().unwrap();

    results
}

/// Run tests in parallel across N worker threads.
///
/// Uses an atomic work counter so fast tests don't block behind slow ones.
/// Results are collected in arbitrary order, then sorted by index for display.
fn run_parallel(suite: &XclbinSuite, tests: &[xdna_emu::testing::xclbin_suite::XclbinTest], jobs: usize) -> Vec<TestResult> {
    let total = tests.len();
    let next_idx = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let results = Mutex::new(Vec::with_capacity(total));

    eprintln!("Running {} tests with {} threads...", total, jobs);

    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                loop {
                    let i = next_idx.fetch_add(1, Ordering::SeqCst);
                    if i >= total { break; }

                    let test = &tests[i];
                    let elf_count = test.find_elf_files().len();
                    let embedded_count = test.count_embedded_cores();
                    let has_npu = test.find_insts_bin().is_some();
                    let (outcome, raw_output, hw_validation) = suite.run_single_with_hw_validation(test);

                    let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
                    eprint!("\r  [{}/{}] completed", done, total);

                    results.lock().unwrap().push(TestResult {
                        idx: i,
                        name: test.name.clone(),
                        elf_count,
                        embedded_count,
                        has_npu,
                        outcome,
                        raw_output,
                        hw_validation,
                    });
                }
            });
        }
    });
    eprintln!(); // Newline after progress counter

    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.idx);
    results
}
