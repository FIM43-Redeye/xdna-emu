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
//!   -j N              Run N tests in parallel (default: 1)
//!   --elfanalyze      Run elfanalyzer on each test's ELFs (requires aietools)
//!   --chess           Build/run with Chess compiler alongside Peano (requires aietools)
//!   --hw              Run tests on real NPU hardware (requires /dev/accel/accel0 + npu-runner)
//!   --aiesim          Run aiesimulator on Chess-built .prj (implies --chess)
//!   --full            Enable all validation: --chess --elfanalyze --aiesim --hw
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
//!   cargo run --example run_mlir_aie_tests -- --chess vector_scalar  # dual-compiler
//!   cargo run --example run_mlir_aie_tests -- --hw add_one          # Peano on real NPU
//!   cargo run --example run_mlir_aie_tests -- --hw --chess add_one  # both compilers on NPU
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
use xdna_emu::integration::aietools::AieTools;
use xdna_emu::integration::aiesimulator;
use xdna_emu::integration::elfanalyzer;
use xdna_emu::integration::chess_build::{BuildEnv, BuildOpts};

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

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            execution: ExecutionConfig::default(),
            build: BuildConfig::default(),
            chess: ChessConfig::default(),
            aiesim: AiesimConfig::default(),
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
}

/// Parse CLI arguments and merge with runner config.
fn parse_args(config: &RunnerConfig) -> Options {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut verbose = false;
    let mut jobs: usize = 1;
    let mut filters = Vec::new();
    let mut elfanalyze = false;
    let mut chess = false;
    let mut hw = false;
    let mut aiesim = false;
    let mut full = false;
    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--verbose" | "-v" => verbose = true,
            "--elfanalyze" => elfanalyze = true,
            "--chess" => chess = true,
            "--hw" => hw = true,
            "--aiesim" => aiesim = true,
            "--full" => full = true,
            "-j" => {
                if let Some(n) = iter.next() {
                    jobs = n.parse().unwrap_or_else(|_| {
                        eprintln!("Invalid -j value: {}", n);
                        std::process::exit(1);
                    });
                    if jobs == 0 { jobs = 1; }
                } else {
                    eprintln!("-j requires a number");
                    std::process::exit(1);
                }
            }
            _ if !arg.starts_with('-') => filters.push(arg.clone()),
            other => {
                eprintln!("Unknown option: {}", other);
                eprintln!("Usage: run_mlir_aie_tests [--verbose|-v] [-j N] [--elfanalyze] [--chess] [--hw] [--aiesim] [--full] [FILTER...]");
                std::process::exit(1);
            }
        }
    }

    // --full enables everything
    if full {
        elfanalyze = true;
        chess = true;
        hw = true;
        aiesim = true;
    }

    // --aiesim implies --chess build (needs .prj from Chess build)
    if aiesim {
        chess = true;
    }

    // Merge CLI flags with config.
    // CLI --chess enables Chess build. Config controls Chess emulator/hw separately.
    // --aiesim on CLI overrides config.aiesim.enabled.
    let chess_build = chess || config.chess.build;
    let chess_emulator = if chess_build { config.chess.run_emulator } else { false };
    let chess_hardware = if chess_build { config.chess.run_hardware } else { false };
    let aiesim = if aiesim { true } else if chess_build { config.aiesim.enabled } else { false };

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

/// Result from a Chess comparison attempt (emulator + optional hardware).
struct ChessComparisonResult {
    /// Compiler diagnosis (if comparison was possible).
    diagnosis: Option<CompilerDiagnosis>,
    /// Label from Chess hardware run (e.g. "PASS (64/64)"), if attempted.
    chess_hw_label: Option<String>,
}

/// Batch-compile all tests with Chess compiler via aiecc.py.
///
/// Iterates all tests, finds MLIR sources, and builds each with Chess
/// before any test execution starts. Uses `nice 19` to avoid saturating
/// the system during heavy compilation. Builds are cached: if the xclbin
/// already exists and is newer than the MLIR source, the build is skipped.
///
/// Returns a map of test_name -> ChessBuildArtifacts for successful builds.
fn batch_build_chess(
    build_env: &BuildEnv,
    tests: &[XclbinTest],
    mlir_aie_path: &Path,
    gen_sim: bool,
    nice_level: i32,
) -> std::collections::HashMap<String, ChessBuildArtifacts> {
    let mut artifacts = std::collections::HashMap::new();

    // Collect buildable tests: those with manifests and MLIR source
    let buildable: Vec<(usize, &XclbinTest, PathBuf)> = tests.iter().enumerate()
        .filter_map(|(i, test)| {
            let manifest = test.manifest.as_ref()?;
            // Skip PLATFORM tests (no point building for wrong hardware)
            if !manifest.test.platform.is_empty() {
                return None;
            }
            let mlir_source = mlir_aie_path
                .join(&manifest.test.source_dir)
                .join(&manifest.build.mlir_file);
            if mlir_source.exists() {
                Some((i, test, mlir_source))
            } else {
                None
            }
        })
        .collect();

    if buildable.is_empty() {
        println!("No tests to build with Chess.");
        return artifacts;
    }

    println!("\n=== CHESS BUILD PHASE ({} tests) ===", buildable.len());
    let build_start = Instant::now();

    for (idx, (_i, test, mlir_source)) in buildable.iter().enumerate() {
        let manifest = test.manifest.as_ref().unwrap();
        let chess_output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("build/chess")
            .join(&test.name);

        let test_start = Instant::now();
        match build_env.build(
            mlir_source,
            &chess_output_dir,
            &BuildOpts {
                use_chess: true,
                gen_sim,
                device: manifest.build.device.clone(),
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
                // Truncate error to first line for display
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

/// Run a pre-built Chess test through the emulator and/or on hardware,
/// and compare against Peano results.
///
/// This function only handles execution and comparison -- all compilation
/// happens in the batch build phase via `batch_build_chess()`.
///
/// When `run_emulator` is false, the Chess emulator step is skipped entirely
/// (useful when Chess binaries always timeout in the emulator).
///
/// When `run_hw` is true and the Chess build produced an insts.bin, also
/// runs the Chess binary on real NPU hardware. `peano_hw_output` provides
/// the Peano hardware result for the full CompilerComparison matrix.
fn run_chess_test(
    suite: &XclbinSuite,
    test: &XclbinTest,
    chess_artifacts: &ChessBuildArtifacts,
    peano_output: Option<&[u8]>,
    run_emulator: bool,
    run_hw: bool,
    skip_hw: bool,
    peano_hw_output: Option<&[u8]>,
) -> ChessComparisonResult {
    let no_result = ChessComparisonResult { diagnosis: None, chess_hw_label: None };
    let manifest = match test.manifest.as_ref() {
        Some(m) => m,
        None => return no_result,
    };

    // Run Chess binary through the emulator (if enabled)
    let chess_raw_output = if run_emulator {
        let mut chess_test = XclbinTest::from_path(&chess_artifacts.xclbin);
        chess_test.manifest = test.manifest.clone();

        let (chess_outcome, chess_raw_output, _hw_validation) =
            suite.run_single_with_hw_validation(&chess_test);

        // Report Chess emulator outcome
        let chess_label = match &chess_outcome {
            TestOutcome::Pass { correct, total, .. } => {
                if let (Some(c), Some(t)) = (correct, total) {
                    format!("PASS ({}/{})", c, t)
                } else {
                    "PASS".to_string()
                }
            }
            TestOutcome::ValidationFail { correct, total, .. } => {
                format!("VALIDATION FAIL ({}/{})", correct, total)
            }
            TestOutcome::Fail { message, .. } => format!("FAIL: {}", message),
            TestOutcome::UnknownOpcode { .. } => "UNKNOWN OPCODE".to_string(),
            TestOutcome::Timeout { .. } => "TIMEOUT".to_string(),
            TestOutcome::LoadError { message } => format!("LOAD ERROR: {}", message),
            other => format!("{:?}", other),
        };
        println!("      chess: {}", chess_label);

        chess_raw_output
    } else {
        None
    };

    // Chess hardware execution (if requested, not PLATFORM-skipped, and insts.bin available)
    let mut chess_hw_output: Option<Vec<u8>> = None;
    let mut chess_hw_label: Option<String> = None;
    if run_hw && !skip_hw {
        if let Some(ref insts) = chess_artifacts.insts {
            let hw_start = Instant::now();
            match run_on_hw_and_validate(manifest, &chess_artifacts.xclbin, insts) {
                Ok((label, output)) => {
                    let elapsed = hw_start.elapsed();
                    println!("      chess-hw: {} ({:.1}s)", label, elapsed.as_secs_f64());
                    chess_hw_label = Some(label);
                    chess_hw_output = Some(output);
                }
                Err(e) => {
                    let label = format!("ERROR ({})", e);
                    println!("      chess-hw: {}", label);
                    chess_hw_label = Some(label);
                }
            }
        }
    }

    // Build CompilerComparison and classify
    let diagnosis = (|| -> Option<CompilerDiagnosis> {
        let output_buf = manifest.get_output()?;
        let elem_type = ElementType::from_str(&output_buf.element_type)?;
        let input_values = generate_input_values(manifest);
        let reference_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/npu-outputs");
        let ref_dir = if reference_dir.exists() { Some(reference_dir.as_path()) } else { None };
        let expected = manifest.generate_expected(&input_values, ref_dir)?;

        let comparison = CompilerComparison {
            test_name: test.name.clone(),
            peano_emu: peano_output.map(|b| b.to_vec()),
            chess_emu: chess_raw_output,
            peano_hw: peano_hw_output.map(|b| b.to_vec()),
            chess_hw: chess_hw_output,
            manifest_expected: Some(expected),
        };

        // Use classify_full() when hardware data is available for refined diagnosis
        let d = if comparison.peano_hw.is_some() || comparison.chess_hw.is_some() {
            comparison.classify_full(elem_type)
        } else {
            comparison.classify(elem_type)
        };
        // Only display when the diagnosis is meaningful (suppress INCOMPLETE)
        if d != CompilerDiagnosis::Incomplete {
            println!("      compiler comparison: {}", d);
        }
        Some(d)
    })();

    ChessComparisonResult { diagnosis, chess_hw_label }
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
    let result = npu_runner::run_on_npu(manifest, xclbin_path, insts_path, 10)
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
    let npu_xrt_path = config.npu_xrt_test_dir();

    // Discover aietools if any tool-dependent flags are set
    let needs_aietools = opts.elfanalyze || opts.chess_build || opts.aiesim;
    let aietools = if needs_aietools {
        match AieTools::discover(config) {
            Some(tools) => {
                println!("aietools: {}", tools.summary());
                if opts.elfanalyze && tools.elfanalyzer.is_none() {
                    eprintln!("Warning: --elfanalyze requested but elfanalyzer not found in aietools");
                }
                if opts.chess_build && tools.xchesscc.is_none() {
                    eprintln!("Error: --chess requires xchesscc from aietools");
                    std::process::exit(1);
                }
                if opts.aiesim && tools.aiesimulator.is_none() {
                    eprintln!("Error: --aiesim requires aiesimulator from aietools");
                    std::process::exit(1);
                }
                Some(tools)
            }
            None => {
                eprintln!("Error: --elfanalyze/--chess/--aiesim require aietools installation");
                eprintln!("Set XILINX_VITIS_AIETOOLS or aietools_path in config");
                std::process::exit(1);
            }
        }
    } else {
        AieTools::discover(config) // Discover silently for informational use
    };

    // Discover build environment if Chess build or aiesim is needed
    let build_env = if opts.chess_build || opts.aiesim {
        match BuildEnv::discover(config) {
            Ok(env) => {
                println!("build env: {}", env.summary());
                Some(env)
            }
            Err(e) => {
                eprintln!("Error: Chess build/aiesim require build environment: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // Validate hardware availability when --hw is requested
    let _runner_path = if opts.hw {
        if !npu_runner::npu_available() {
            eprintln!("Error: --hw requires NPU hardware (/dev/accel/accel0 not found)");
            std::process::exit(1);
        }
        match npu_runner::runner_binary() {
            Some(path) => {
                println!("hw: NPU available, npu-runner at {}", path.display());
                Some(path)
            }
            None => {
                eprintln!("Error: --hw requires npu-runner binary");
                eprintln!("Build with: cd tools/npu-runner && cmake -B build && cmake --build build");
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mlir-aie-extracted/manifests");

    if !npu_xrt_path.exists() {
        eprintln!("Error: mlir-aie npu-xrt tests not found at {}", npu_xrt_path.display());
        eprintln!("Make sure mlir-aie is built with tests enabled.");
        std::process::exit(1);
    }

    println!("Discovering tests in {}...", npu_xrt_path.display());

    let reference_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");

    let npu_output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/npu-outputs");

    let mut suite = match XclbinSuite::discover(&npu_xrt_path) {
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
                s.with_manifest_dir(manifest_path)
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
    let total_discovered = all_tests.len();

    // Apply filter
    let tests: Vec<_> = all_tests.into_iter()
        .filter(|t| matches_filter(&t.name, &opts.filters))
        .collect();

    let total = tests.len();
    if !opts.filters.is_empty() {
        println!("Filter: {:?} -> {}/{} tests selected\n", opts.filters, total, total_discovered);
    } else {
        println!("Found {} tests\n", total);
    }

    if total == 0 {
        println!("No tests matched the filter.");
        return;
    }

    // === BUILD PHASE ===
    // Batch-compile with Chess before any test execution. This separates
    // the slow compilation step (nice 19) from the fast execution step,
    // so all build errors surface upfront and test execution is uninterrupted.
    let chess_builds = if opts.chess_build {
        if let Some(ref env) = build_env {
            batch_build_chess(env, &tests, &mlir_aie_path, opts.aiesim, opts.build_nice)
        } else {
            std::collections::HashMap::new()
        }
    } else {
        std::collections::HashMap::new()
    };

    // === EXECUTE PHASE ===
    // Run tests -- parallel or sequential depending on -j flag
    let results: Vec<TestResult> = if opts.jobs > 1 {
        run_parallel(&suite, &tests, opts.jobs)
    } else {
        run_sequential(&suite, &tests)
    };

    // Display results in order and accumulate counters
    let mut passed = 0;
    let mut validation_failed = 0;
    let mut expected_fail = 0;
    let mut unexpected_pass = 0;
    let mut skipped = 0;
    let mut platform_count = 0;
    let mut failed = 0;
    let mut unknown_count = 0;
    let mut timeout_count = 0;
    let mut load_error = 0;
    let mut hw_validated = 0;
    let mut hw_correct = 0;
    let mut hw_compiler_bug = 0;
    let mut hw_emulator_bug = 0;
    let mut chess_compared = 0;
    let mut chess_correct = 0;
    let mut chess_peano_bugs = 0;
    let mut chess_chess_bugs = 0;
    let mut chess_emu_bugs = 0;
    let mut peano_hw_pass = 0;
    let mut peano_hw_fail = 0;
    let mut peano_hw_error = 0;
    let mut peano_hw_attempted = 0;
    let mut chess_hw_pass = 0;
    let mut chess_hw_fail = 0;
    let mut chess_hw_error = 0;
    let mut chess_hw_attempted = 0;
    let mut sim_attempted = 0;
    let mut sim_correct = 0;
    let mut sim_wrong = 0;
    let mut sim_error = 0;

    for r in &results {
        println!("{}", format_result(r, total));

        // Track hardware cross-validation stats
        if let Some(ref hv) = r.hw_validation {
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

        match &r.outcome {
            TestOutcome::Pass { .. } => passed += 1,
            TestOutcome::ValidationFail { .. } => validation_failed += 1,
            TestOutcome::Fail { .. } => failed += 1,
            TestOutcome::UnknownOpcode { details, .. } => {
                suite.record_unknown(details.clone(), &r.name);
                unknown_count += 1;
            }
            TestOutcome::Timeout { .. } => timeout_count += 1,
            TestOutcome::LoadError { .. } => load_error += 1,
            TestOutcome::ExpectedFail { .. } => expected_fail += 1,
            TestOutcome::UnexpectedPass { .. } => unexpected_pass += 1,
            TestOutcome::Skipped { .. } => skipped += 1,
            TestOutcome::Platform { .. } => platform_count += 1,
        }

        // Verbose mode: print full expected vs actual comparison
        if opts.verbose {
            if let Some(ref manifest) = tests[r.idx].manifest {
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

        // Peano hardware execution: run the Peano-compiled xclbin on real NPU
        // Skip for PLATFORM/SKIP tests -- no point running npu2 binaries on npu1
        let mut peano_hw_output: Option<Vec<u8>> = None;
        let skip_hw = matches!(&r.outcome,
            TestOutcome::Platform { .. } | TestOutcome::Skipped { .. });
        if opts.hw && !skip_hw {
            if let Some(ref manifest) = tests[r.idx].manifest {
                if let Some(insts_path) = tests[r.idx].find_insts_bin() {
                    peano_hw_attempted += 1;
                    let hw_start = Instant::now();
                    match run_on_hw_and_validate(manifest, &tests[r.idx].xclbin_path, &insts_path) {
                        Ok((label, output)) => {
                            let elapsed = hw_start.elapsed();
                            println!("      peano-hw: {} ({:.1}s)", label, elapsed.as_secs_f64());
                            if label.starts_with("PASS") {
                                peano_hw_pass += 1;
                            } else {
                                peano_hw_fail += 1;
                            }
                            peano_hw_output = Some(output);
                        }
                        Err(e) => {
                            println!("      peano-hw: ERROR ({})", e);
                            peano_hw_error += 1;
                        }
                    }
                }
            }
        }

        // elfanalyzer: run on each test's ELF files
        if opts.elfanalyze {
            if let Some(ref tools) = aietools {
                run_elfanalyzer(tools, &tests[r.idx], &mlir_aie_path);
            }
        }

        // Chess: run pre-built Chess binary through emulator and/or on hardware
        if opts.chess_build {
            if let Some(chess_artifacts) = chess_builds.get(&tests[r.idx].name) {
                let result = run_chess_test(
                    &suite, &tests[r.idx], chess_artifacts,
                    r.raw_output.as_deref(),
                    opts.chess_emulator,
                    opts.hw && opts.chess_hardware, skip_hw,
                    peano_hw_output.as_deref(),
                );

                if let Some(diag) = result.diagnosis {
                    if diag != CompilerDiagnosis::Incomplete {
                        match diag {
                            CompilerDiagnosis::PeanoCompilerBug => chess_peano_bugs += 1,
                            CompilerDiagnosis::ChessCompilerBug => chess_chess_bugs += 1,
                            CompilerDiagnosis::EmulatorBug => chess_emu_bugs += 1,
                            CompilerDiagnosis::Correct => chess_correct += 1,
                            _ => {}
                        }
                        chess_compared += 1;
                    }
                }

                // Track Chess hardware stats from the comparison result
                if let Some(ref chess_hw_label) = result.chess_hw_label {
                    chess_hw_attempted += 1;
                    if chess_hw_label.starts_with("PASS") {
                        chess_hw_pass += 1;
                    } else if chess_hw_label.starts_with("FAIL") {
                        chess_hw_fail += 1;
                    } else {
                        chess_hw_error += 1;
                    }
                }
            }
        }

        // aiesimulator comparison (needs Chess build's .prj)
        if opts.aiesim {
            if let Some(ref tools) = aietools {
                let chess_prj_dir = chess_builds.get(&tests[r.idx].name)
                    .and_then(|a| a.prj_dir.as_ref());
                if let Some(ref prj) = chess_prj_dir {
                    sim_attempted += 1;
                    let sim_label = run_aiesim_comparison(
                        tools, &tests[r.idx], prj,
                    );
                    println!("      aiesim: {}", sim_label);

                    if sim_label.starts_with("PASS") {
                        sim_correct += 1;
                    } else if sim_label.starts_with("FAIL") {
                        sim_wrong += 1;
                    } else {
                        sim_error += 1;
                    }
                }
            }
        }
    }

    println!("\n{:=<60}", "");
    println!("=== SUMMARY ===");
    let effective = total - skipped - platform_count;
    println!("Total:            {}", total);
    println!("Platform:         {} (requires different hardware)", platform_count);
    println!("Skipped:          {}", skipped);
    println!("Passed:           {} ({:.1}%)", passed, 100.0 * passed as f64 / effective.max(1) as f64);
    println!("Expected Fail:    {}", expected_fail);
    println!("Unexpected Pass:  {}", unexpected_pass);
    println!("Validation Fail:  {}", validation_failed);
    println!("Failed:           {}", failed);
    println!("Unknown:          {}", unknown_count);
    println!("Timeout:          {}", timeout_count);
    println!("Load Error:       {}", load_error);

    // Hardware cross-validation summary
    if hw_validated > 0 {
        println!("\n=== HARDWARE CROSS-VALIDATION ===");
        println!("Validated:        {}", hw_validated);
        println!("Correct:          {}", hw_correct);
        println!("Compiler Bug:     {}", hw_compiler_bug);
        println!("Emulator Bug:     {}", hw_emulator_bug);
    }

    // Chess compiler comparison summary
    if chess_compared > 0 {
        println!("\n=== CHESS vs PEANO COMPARISON ===");
        println!("Compared:         {}", chess_compared);
        println!("Correct:          {} (both compilers match expected)", chess_correct);
        println!("Peano Bug:        {} (Chess correct, Peano wrong)", chess_peano_bugs);
        println!("Chess Bug:        {} (Peano correct, Chess wrong)", chess_chess_bugs);
        println!("Emulator Bug:     {} (both wrong, same way)", chess_emu_bugs);
    }

    // Live hardware execution summary
    if peano_hw_attempted > 0 || chess_hw_attempted > 0 {
        println!("\n=== LIVE HARDWARE EXECUTION ===");
        if peano_hw_attempted > 0 {
            let mut peano_detail = format!("{}/{} pass", peano_hw_pass, peano_hw_attempted);
            if peano_hw_fail > 0 {
                peano_detail.push_str(&format!(", {} fail", peano_hw_fail));
            }
            if peano_hw_error > 0 {
                peano_detail.push_str(&format!(", {} error", peano_hw_error));
            }
            println!("Peano HW:         {}", peano_detail);
        }
        if chess_hw_attempted > 0 {
            let mut chess_detail = format!("{}/{} pass", chess_hw_pass, chess_hw_attempted);
            if chess_hw_fail > 0 {
                chess_detail.push_str(&format!(", {} fail", chess_hw_fail));
            }
            if chess_hw_error > 0 {
                chess_detail.push_str(&format!(", {} error", chess_hw_error));
            }
            println!("Chess HW:         {}", chess_detail);
        }
    }

    // aiesimulator summary
    if sim_attempted > 0 {
        println!("\n=== AIESIMULATOR CROSS-VALIDATION ===");
        println!("Simulated:        {}", sim_attempted);
        println!("Correct:          {}", sim_correct);
        println!("Wrong:            {}", sim_wrong);
        println!("Error:            {} (build/invocation failures)", sim_error);
    }

    // Show unknown opcodes if any
    let collector = suite.collector();
    let unknowns = collector.by_impact();
    if !unknowns.is_empty() {
        println!("\n=== UNKNOWN OPCODES (by impact) ===");
        for stats in unknowns.iter().take(10) {
            let op = &stats.first;
            let mnemonic = op.mnemonic.as_deref().unwrap_or("unknown");
            println!("  {:?} opcode 0x{:04X} '{}' (hits: {}, tests: {})",
                     op.slot, op.opcode, mnemonic, stats.count, stats.tests.len());
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
