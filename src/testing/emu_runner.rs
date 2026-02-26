//! Emulator + hardware validation runner.
//!
//! This is the primary test runner mode: discover tests from the mlir-aie
//! source tree (or pre-built cmake output), build them with Peano (and
//! optionally Chess), run through the emulator, validate against hardware
//! reference captures, and optionally execute on real NPU hardware.
//!
//! Entry point: [`run()`].

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::testing::xclbin_suite::{Compiler, XclbinSuite, XclbinTest, TestOutcome};
use crate::testing::test_cpp_parser::{BufferDir, read_values};
use crate::testing::hardware_comparison::{
    CompilerComparison, CompilerDiagnosis, load_hw_reference,
};
use crate::testing::npu_runner;
use crate::testing::npu_test::{self, NpuTestSource};
use crate::testing::runner_config::{self, Options};
use crate::testing::runner_stats::{RunStats, HwRunResult};
use crate::testing::hw_executor;
use crate::testing::native_hw;
use crate::testing::runner_display::{self, TestResult};
use crate::testing::unit_test;
use crate::integration::aietools::AieTools;
use crate::integration::chess_build::BuildEnv;
use crate::build_progress::{self, ParallelBuildConfig};

/// Type alias for Chess build artifacts (from build_progress module).
type ChessBuildArtifacts = build_progress::ChessArtifacts;

// ---------------------------------------------------------------------------
// Hardware recovery and wedge detection
// ---------------------------------------------------------------------------

/// Tracks device health across hardware test runs.
///
/// After each HW error (timeout, crash), the test runner already waits for
/// device recovery via `wait_for_device_idle(true)` in hw_executor/native_hw.
/// This state tracker then verifies the device actually recovered before
/// continuing. Only two conditions stop hardware execution:
///
/// - **D-state**: process survived SIGKILL, device is truly unrecoverable
///   without a reboot or bus-level reset.
/// - **Recovery failure**: device didn't return to healthy state after the
///   idle wait -- sysfs shows it's still active/wedged, or the device node
///   disappeared entirely.
///
/// Individual test timeouts (e.g. control packet tests that always hang) are
/// tolerated as long as the device recovers via TDR afterward.
struct HwCascadeState {
    hw_errors: usize,
    disabled_reason: Option<String>,
    stopped_at: Option<usize>,
    cascade_skipped: usize,
}

impl HwCascadeState {
    fn new() -> Self {
        Self {
            hw_errors: 0,
            disabled_reason: None,
            stopped_at: None,
            cascade_skipped: 0,
        }
    }

    fn is_disabled(&self) -> bool {
        self.disabled_reason.is_some()
    }

    /// Record a hardware execution result. After each error, verify the
    /// device recovered before allowing subsequent tests to run.
    fn record(&mut self, hw: &HwRunResult, test_idx: usize) {
        use super::runner_stats::HwOutcome;

        // D-state: process survived SIGKILL. Device is unrecoverable
        // without reboot or bus-level reset. Stop immediately.
        if hw.outcome == HwOutcome::Wedged && self.disabled_reason.is_none() {
            let detail = format!(
                "NPU device wedged (D-state) at test {} -- \
                 process survived SIGKILL, reboot required",
                test_idx + 1,
            );
            eprintln!("\nCRITICAL: {}", detail);
            eprintln!("Disabling hardware execution for remaining tests.\n");
            self.stopped_at = Some(test_idx + 1);
            self.disabled_reason = Some(detail);
            return;
        }

        // PASS or FAIL means the device executed the test. No recovery
        // check needed.
        if hw.outcome == HwOutcome::Pass || hw.outcome == HwOutcome::Fail {
            return;
        }

        // ERROR (timeout, crash, etc.) -- the device may need TDR recovery.
        // wait_for_device_idle(true) already ran in hw_executor/native_hw,
        // so the device has had up to 10s to recover. Verify it did.
        self.hw_errors += 1;

        if self.disabled_reason.is_some() {
            return;
        }

        if !npu_runner::probe_device_health() {
            // Device node gone -- driver crashed or device disappeared.
            let detail = format!(
                "NPU device node disappeared after error at test {} -- \
                 driver crash or device removal",
                test_idx + 1,
            );
            eprintln!("\nCRITICAL: {}", detail);
            eprintln!("Disabling hardware execution for remaining tests.\n");
            self.stopped_at = Some(test_idx + 1);
            self.disabled_reason = Some(detail);
            return;
        }

        // Device node exists. Check if runtime PM recovered (suspended
        // means device is idle and ready for next test).
        if !npu_runner::device_is_idle() {
            // Device still active after the idle wait timeout. TDR
            // recovery may have failed -- one more short wait.
            eprintln!(
                "\n  WARNING: Device still active after error at test {}. \
                 Waiting for TDR recovery...",
                test_idx + 1,
            );
            npu_runner::wait_for_device_idle(true);

            if !npu_runner::device_is_idle() && !npu_runner::probe_device_health() {
                let detail = format!(
                    "NPU device failed to recover after test {} -- \
                     TDR recovery did not complete",
                    test_idx + 1,
                );
                eprintln!("CRITICAL: {}", detail);
                eprintln!("Disabling hardware execution for remaining tests.\n");
                self.stopped_at = Some(test_idx + 1);
                self.disabled_reason = Some(detail);
                return;
            }
        }

        // Device recovered. Log and continue.
        eprintln!(
            "  (test {} error, device recovered via TDR -- continuing)",
            test_idx + 1,
        );
    }

    /// Transfer cascade counters to RunStats for summary display.
    fn apply_to_stats(&self, stats: &mut RunStats) {
        stats.hw_cascade_stopped_at = self.stopped_at;
        stats.hw_cascade_skipped = self.cascade_skipped;
    }
}

// ---------------------------------------------------------------------------
// Per-test processing context
// ---------------------------------------------------------------------------

/// Shared read-only context for per-test processing.
///
/// Bundles the ~10 parameters that every test needs access to, avoiding
/// a function signature with 15 arguments.
struct TestContext<'a> {
    opts: &'a Options,
    hw_available: bool,
    aietools: &'a Option<AieTools>,
    chess_builds: &'a HashMap<String, ChessBuildArtifacts>,
    chess_emu_results: &'a HashMap<String, (TestOutcome, Option<Vec<u8>>)>,
    reference_dir: &'a Path,
    mlir_aie_path: &'a Path,
    total: usize,
}

// ---------------------------------------------------------------------------
// Per-test processing
// ---------------------------------------------------------------------------

/// Process a single test: display emulator results, run hardware, do
/// comparisons. This is the body of the main test loop.
fn process_test(
    ctx: &TestContext,
    suite: &mut XclbinSuite,
    stats: &mut RunStats,
    cascade: &mut HwCascadeState,
    test: &XclbinTest,
    idx: usize,
    emu: &Option<TestResult>,
) {
    let compact = emu.is_none(); // hw-only mode: compact single-line output

    // --- Emulator display (skip in hw-only) ---
    if let Some(ref r) = emu {
        println!("{}", runner_display::format_result(r, ctx.total));
        stats.record_emu_outcome(&r.outcome);
        if let Some(ref hv) = r.hw_validation {
            stats.record_hw_validation(hv);
        }
        if let TestOutcome::UnknownOpcode { ref details, .. } = r.outcome {
            suite.record_unknown(details.clone(), &r.name);
        }

        if ctx.opts.verbose {
            if test.buffer_spec.is_some() {
                if let Some(ref output) = r.raw_output {
                    let show = matches!(&r.outcome,
                        TestOutcome::ValidationFail { .. } |
                        TestOutcome::ExpectedFail { .. } |
                        TestOutcome::UnexpectedPass { .. } |
                        TestOutcome::Pass { correct: Some(_), .. }
                    );
                    if show {
                        runner_display::print_verbose_comparison(output, test, ctx.reference_dir);
                    }
                }
            }
        }
    }

    // --- Skip gates ---
    // Only skip HW for Platform tests (wrong hardware generation). Skipped tests
    // (e.g. DMA-only, no core code) may still be valid on real hardware.
    let skip_hw = emu.as_ref().map_or(false, |r|
        matches!(&r.outcome, TestOutcome::Platform{..}));

    let spec = test.buffer_spec.as_ref();
    let insts_path = test.find_insts_file();

    // In hw-only mode, skip tests that cannot run on hardware at all.
    if compact {
        let has_native = test.test_exe.is_some();
        let has_runner = spec.is_some() && insts_path.is_some();
        if !has_native && !has_runner {
            let reason = if insts_path.is_none() { "no insts.bin" }
                else { "no buffer spec or test.exe" };
            println!("[{:2}/{}] {:45} SKIP ({})",
                idx + 1, ctx.total, &test.name[..test.name.len().min(45)], reason);
            stats.skipped += 1;
            return;
        }
        print!("[{:2}/{}] {:35} ",
            idx + 1, ctx.total, &test.name[..test.name.len().min(35)]);
    }

    // --- Chess HW first (ground truth, when Chess xclbin is available) ---
    //
    // In hw-only (compact) mode, Chess runs first as the primary/ground-truth
    // compiler. The host test.exe is compiler-agnostic -- we just point it at
    // the Chess-compiled xclbin and insts.bin.
    let chess_hw: Option<HwRunResult> = if !ctx.hw_available || cascade.is_disabled() || skip_hw {
        None
    } else if let Some(chess_artifacts) = ctx.chess_builds.get(&test.name) {
        if let Some(ref chess_insts) = chess_artifacts.insts {
            // Build a temporary test with Chess xclbin/insts but same test.exe
            let chess_insts_path = Some(chess_insts.clone());
            let mut chess_test = test.clone();
            chess_test.xclbin_path = chess_artifacts.xclbin.clone();
            chess_test.insts_path = Some(chess_insts.clone());
            run_hw_for_test(&chess_test, Compiler::Chess, compact, spec,
                &chess_insts_path, ctx, stats, cascade, idx)
        } else {
            None
        }
    } else {
        None
    };

    // --- Peano HW (secondary, always runs if available) ---
    let compiler = test.compiler.unwrap_or(Compiler::Peano);
    let primary_hw: Option<HwRunResult> = if cascade.is_disabled() {
        if chess_hw.is_none() { cascade.cascade_skipped += 1; }
        None
    } else if ctx.hw_available && !skip_hw {
        if compact && chess_hw.is_some() { print!("  "); }
        run_hw_for_test(test, compiler, compact, spec, &insts_path, ctx, stats, cascade, idx)
    } else {
        None
    };

    // --- Auto-capture reference output ---
    // Prefer Peano output (primary test path). Fall back to Chess if Peano
    // didn't run (e.g. cascade disabled for Peano but Chess already ran).
    let capture_hw = primary_hw.as_ref().or(chess_hw.as_ref());
    if let Some(hw) = capture_hw {
        auto_capture_reference(test, hw, ctx.reference_dir, compact);
    }

    // --- elfanalyzer ---
    if ctx.opts.elfanalyze && !compact {
        if let Some(ref tools) = ctx.aietools {
            runner_display::run_elfanalyzer(tools, test, ctx.mlir_aie_path);
        }
    }

    // --- Chess emulator (pre-computed in parallel phase 2b) ---
    let chess_emu_output: Option<Vec<u8>> = if let Some((outcome, raw)) =
        ctx.chess_emu_results.get(&test.name)
    {
        let label = format_chess_emu_label(outcome);
        if !compact {
            println!("      chess: {}", label);
        }
        raw.clone()
    } else {
        None
    };

    // --- Differential (when both compilers ran on HW) ---
    if let (Some(ref p), Some(ref c)) = (&primary_hw, &chess_hw) {
        stats.record_differential(
            p.outcome == super::runner_stats::HwOutcome::Pass,
            c.outcome == super::runner_stats::HwOutcome::Pass,
        );
    }

    // --- Compiler comparison ---
    if !ctx.chess_builds.is_empty() {
        if ctx.chess_builds.contains_key(&test.name) {
            if let Some(s) = spec {
                let diag = run_compiler_comparison(
                    s, test, compiler, emu, &chess_emu_output,
                    &primary_hw, &chess_hw, ctx.reference_dir, compact,
                );
                if let Some(d) = diag {
                    stats.record_compiler_diagnosis(d);
                }
            }
        }
    }

    // --- aiesimulator ---
    if ctx.opts.aiesim && !compact {
        if let Some(ref tools) = ctx.aietools {
            if let Some(prj) = ctx.chess_builds.get(&test.name).and_then(|a| a.prj_dir.as_ref()) {
                let sim_label = runner_display::run_aiesim_comparison(tools, test, prj, ctx.reference_dir);
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

/// Run primary hardware test (native test.exe preferred, npu-runner fallback).
fn run_hw_for_test(
    test: &XclbinTest,
    compiler: Compiler,
    compact: bool,
    spec: Option<&crate::testing::test_cpp_parser::BufferSpec>,
    insts_path: &Option<PathBuf>,
    ctx: &TestContext,
    stats: &mut RunStats,
    cascade: &mut HwCascadeState,
    idx: usize,
) -> Option<HwRunResult> {
    let prefix = match compiler {
        Compiler::Peano => if compact { "peano" } else { "peano-hw" },
        Compiler::Chess => if compact { "chess" } else { "chess-hw" },
    };

    // Try native test.exe first (handles custom host logic correctly)
    let hw = if let (Some(ref exe), Some(ref pattern), Some(ref insts)) =
        (&test.test_exe, &test.test_cpp_pattern, insts_path)
    {
        Some(native_hw::run_native_and_print(
            exe, &test.xclbin_path, insts, *pattern, prefix, compact,
        ))
    } else if let (Some(s), Some(ref insts)) = (spec, insts_path) {
        Some(hw_executor::run_hw_and_print(
            s, &test.name, &test.xclbin_path, insts, prefix, compact, ctx.reference_dir,
        ))
    } else {
        None
    };

    if let Some(ref hw) = hw {
        stats.record_hw(compiler, hw);
        cascade.record(hw, idx);
    }

    hw
}

/// Auto-capture hardware reference output for future emulator validation.
///
/// When a test passes on hardware but has no saved reference output, this
/// captures the raw binary output and saves it so subsequent emulator runs
/// can validate against it. Two paths:
///
/// 1. If the HW run already produced raw bytes (npu_runner path), save directly.
/// 2. If the HW run used native test.exe (no raw bytes), do a supplementary
///    npu_runner execution to capture the output buffer contents.
///
/// Only captures when the test passed -- we don't want wrong output as reference.
fn auto_capture_reference(
    test: &XclbinTest,
    hw: &HwRunResult,
    reference_dir: &Path,
    compact: bool,
) {
    use super::runner_stats::HwOutcome;

    // Only capture for tests that passed on hardware
    if hw.outcome != HwOutcome::Pass {
        return;
    }

    // Need buffer spec to know output element type for validation
    let spec = match test.buffer_spec.as_ref() {
        Some(s) => s,
        None => return,
    };

    // Need insts for npu_runner fallback
    let insts_path = match test.find_insts_file() {
        Some(p) => p,
        None => return,
    };

    // Skip if reference already exists
    let ref_path = reference_dir.join(&test.name).join("output.bin");
    if ref_path.exists() {
        return;
    }

    // Get raw output bytes: either from the HW result or via supplementary capture
    let output = if !hw.output.is_empty() {
        hw.output.clone()
    } else {
        // Native test.exe path -- no raw bytes. Do a npu_runner capture.
        match npu_runner::run_on_npu(
            spec, &test.name, &test.xclbin_path, &insts_path,
            super::runner_config::DEFAULT_HW_TIMEOUT_SECS,
        ) {
            Ok(result) => result.output,
            Err(e) => {
                log::debug!("auto-capture npu_runner failed for {}: {}", test.name, e);
                return;
            }
        }
    };

    if output.is_empty() {
        return;
    }

    // Save the reference
    if let Some(parent) = ref_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            log::warn!("auto-capture: failed to create dir for {}: {}", test.name, e);
            return;
        }
    }
    match std::fs::write(&ref_path, &output) {
        Ok(_) => {
            let label = format!("captured {} bytes", output.len());
            if compact {
                // hw-only mode: append to current line
                print!("  [{}]", label);
            } else {
                println!("      ref: {} -> {}", label, ref_path.display());
            }
        }
        Err(e) => {
            log::warn!("auto-capture: failed to write {}: {}", ref_path.display(), e);
        }
    }
}

/// Format a Chess emulator outcome as a display label.
fn format_chess_emu_label(outcome: &TestOutcome) -> String {
    match outcome {
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
    }
}

/// Run compiler comparison (Peano vs Chess) and return the diagnosis.
fn run_compiler_comparison(
    spec: &crate::testing::test_cpp_parser::BufferSpec,
    test: &XclbinTest,
    compiler: Compiler,
    emu: &Option<TestResult>,
    chess_emu_output: &Option<Vec<u8>>,
    primary_hw: &Option<HwRunResult>,
    chess_hw: &Option<HwRunResult>,
    reference_dir: &Path,
    compact: bool,
) -> Option<CompilerDiagnosis> {
    let output_buf = spec.buffers.iter().find(|b| b.direction == BufferDir::Output)?;
    let elem_type = output_buf.element_type;

    let expected_bytes = load_hw_reference(reference_dir, &test.name)?;
    let expected = read_values(&expected_bytes, elem_type);

    let peano_hw_output = if compiler == Compiler::Peano {
        primary_hw.as_ref().map(|h| h.output.clone())
    } else {
        None
    };
    let comparison = CompilerComparison {
        test_name: test.name.clone(),
        peano_emu: emu.as_ref().and_then(|r| r.raw_output.clone()),
        chess_emu: chess_emu_output.clone(),
        peano_hw: peano_hw_output,
        chess_hw: chess_hw.as_ref().map(|h| h.output.clone()),
        expected_values: Some(expected),
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
}

// ---------------------------------------------------------------------------
// Unit test execution
// ---------------------------------------------------------------------------

/// Results from the unit test phase, for summary display.
struct UnitTestStats {
    discovered: usize,
    built: usize,
    build_failed: usize,
    skipped: usize,
    sim_pass: usize,
    sim_fail: usize,
    sim_error: usize,
}

/// Run the unit test discovery/build/simulation phase.
fn run_unit_tests(
    opts: &Options,
    aietools: &Option<AieTools>,
    build_env: &Option<BuildEnv>,
    mlir_aie_path: &Path,
    manifest_dir: &str,
) -> UnitTestStats {
    let mut s = UnitTestStats {
        discovered: 0, built: 0, build_failed: 0,
        skipped: 0, sim_pass: 0, sim_fail: 0, sim_error: 0,
    };

    println!("\n{:=<60}", "");
    println!("=== UNIT TESTS (chess_compiler_tests_aie2) ===\n");

    let unit_tests = unit_test::discover(mlir_aie_path);
    s.discovered = unit_tests.len();

    if unit_tests.is_empty() {
        println!("No unit tests found.");
        return s;
    }

    let filtered: Vec<_> = unit_tests.iter()
        .filter(|t| runner_config::matches_filter(&t.name, &opts.filters))
        .collect();

    let filtered_count = filtered.len();
    if !opts.filters.is_empty() {
        println!("Filter: {:?} -> {}/{} unit tests selected",
            opts.filters, filtered_count, s.discovered);
    } else {
        println!("Found {} unit tests", s.discovered);
    }

    let env = match build_env {
        Some(e) => e,
        None => return s,
    };

    // Build phase
    println!("\n--- Unit Test Build Phase ({} tests) ---", filtered_count);
    let build_start = Instant::now();

    let mut build_results: Vec<(&unit_test::UnitTest, PathBuf)> = Vec::new();

    for (idx, test) in filtered.iter().enumerate() {
        if let Some(ref reason) = test.skip_reason {
            println!("[{:2}/{}] {:55} SKIP ({})",
                idx + 1, filtered_count,
                &test.name[..test.name.len().min(55)],
                reason);
            s.skipped += 1;
            continue;
        }

        let unit_output_dir = PathBuf::from(manifest_dir)
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
                println!("[{:2}/{}] {:55} {} ({:.1}s)",
                    idx + 1, filtered_count,
                    &test.name[..test.name.len().min(55)],
                    label, elapsed.as_secs_f64());
                s.built += 1;
                build_results.push((test, result.prj_dir));
            }
            Err(e) => {
                let elapsed = test_start.elapsed();
                let msg = e.lines().next().unwrap_or(&e);
                println!("[{:2}/{}] {:55} FAILED ({:.1}s): {}",
                    idx + 1, filtered_count,
                    &test.name[..test.name.len().min(55)],
                    elapsed.as_secs_f64(),
                    &msg[..msg.len().min(80)]);
                s.build_failed += 1;
            }
        }
    }

    let build_elapsed = build_start.elapsed();
    println!("Unit test builds: {}/{} succeeded ({:.1}s)",
        s.built, filtered_count - s.skipped, build_elapsed.as_secs_f64());

    // Simulation phase (run through aiesimulator)
    if opts.unit_tests_aiesim && !build_results.is_empty() {
        if let Some(ref tools) = aietools {
            use crate::integration::aiesimulator;

            println!("\n--- Unit Test Simulation Phase ({} tests) ---",
                build_results.len());

            for (idx, (test, prj_dir)) in build_results.iter().enumerate() {
                eprint!("\r[{:2}/{}] {}...",
                    idx + 1, build_results.len(),
                    &test.name[..test.name.len().min(55)]);
                io::stderr().flush().unwrap();

                match aiesimulator::run_unit_simulation(
                    tools,
                    prj_dir,
                    opts.unit_tests_aiesim_timeout,
                ) {
                    Ok(result) => {
                        let label = if result.passed {
                            s.sim_pass += 1;
                            "PASS"
                        } else {
                            s.sim_fail += 1;
                            "FAIL"
                        };
                        eprint!("\r{:60}\r", "");
                        println!("[{:2}/{}] {:55} {} ({:.1}s)",
                            idx + 1, build_results.len(),
                            &test.name[..test.name.len().min(55)],
                            label, result.wall_time_secs);

                        if opts.verbose && !result.passed {
                            for line in result.stdout.lines().take(20) {
                                println!("      {}", line);
                            }
                        }
                    }
                    Err(e) => {
                        s.sim_error += 1;
                        eprint!("\r{:60}\r", "");
                        let msg = e.lines().next().unwrap_or(&e);
                        println!("[{:2}/{}] {:55} ERROR: {}",
                            idx + 1, build_results.len(),
                            &test.name[..test.name.len().min(55)],
                            &msg[..msg.len().min(60)]);
                    }
                }
            }
            eprint!("\r{:60}\r", "");
            io::stderr().flush().unwrap();
        }
    }

    s
}

// ---------------------------------------------------------------------------
// Emulator execution helpers
// ---------------------------------------------------------------------------

/// Run tests sequentially (original behavior, -j 1).
fn run_sequential(suite: &XclbinSuite, tests: &[XclbinTest]) -> Vec<TestResult> {
    let total = tests.len();
    let mut results = Vec::with_capacity(total);

    for (i, test) in tests.iter().enumerate() {
        eprint!("\r[{:2}/{}] {}...", i + 1, total, &test.name[..test.name.len().min(55)]);
        io::stderr().flush().unwrap();

        let elf_count = test.find_elf_files().len();
        let embedded_count = test.count_embedded_cores();
        let has_npu = test.find_insts_file().is_some();
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
fn run_parallel(suite: &XclbinSuite, tests: &[XclbinTest], jobs: usize) -> Vec<TestResult> {
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
                    let has_npu = test.find_insts_file().is_some();
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

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run the emulator + hardware validation pipeline.
///
/// This is the default mode for `npu-test`. It discovers tests from the
/// mlir-aie source tree (or pre-built cmake output), builds them, runs
/// through the emulator, and optionally validates against real NPU hardware.
pub fn run(opts: &Options) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let config = crate::config::Config::get();
    let mlir_aie_path = PathBuf::from(config.mlir_aie_path());

    // Discover aietools (always attempt -- needed for auto Chess detection)
    let aietools = AieTools::discover(config);
    let chess_available = opts.chess_build && aietools.as_ref()
        .map_or(false, |t| t.xchesscc.is_some());

    if opts.chess_only && !chess_available {
        eprintln!("Error: --chess-only requires aietools with xchesscc");
        std::process::exit(1);
    }

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
                eprintln!("Warning: build environment not available: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Hardware validation: auto-detected by default, graceful fallback.
    let hw_available = if opts.hw {
        if !npu_runner::npu_available() {
            println!("hw: NPU not available, hardware validation disabled");
            false
        } else {
            let has_runner = npu_runner::runner_binary().is_some();
            if has_runner {
                println!("hw: NPU available (native test.exe + npu-runner fallback)");
            } else {
                println!("hw: NPU available (native test.exe only, no npu-runner)");
            }
            true
        }
    } else {
        false
    };

    let reference_dir = PathBuf::from(manifest_dir).join("tests/npu-outputs");
    let npu_output_dir = PathBuf::from(manifest_dir).join("tests/npu-outputs");
    let overrides_path = PathBuf::from(manifest_dir).join("tests/test_overrides.toml");

    // ====================================================================
    // DISCOVERY AND BUILD
    // ====================================================================

    // Declared here (not inside the else branch) so it outlives the if/else block.
    let mut source_tests_storage: Vec<NpuTestSource>;

    let (mut suite, tests, total_discovered, chess_builds) = if opts.no_build {
        // Legacy path: discover pre-built tests from cmake build tree
        let npu_xrt_path = config.npu_xrt_test_dir();
        if !npu_xrt_path.exists() {
            eprintln!("Error: npu-xrt build output not found at {}", npu_xrt_path.display());
            eprintln!("Either build mlir-aie with tests, or omit --no-build to build from source.");
            std::process::exit(1);
        }

        println!("Discovering pre-built tests in {}...", npu_xrt_path.display());

        let mut suite = match XclbinSuite::discover(&npu_xrt_path) {
            Ok(s) => {
                let mut s = s.with_max_cycles(opts.max_cycles);
                if reference_dir.exists() {
                    s = s.with_reference_dir(reference_dir.clone());
                }
                if npu_output_dir.exists() {
                    s = s.with_npu_output_dir(npu_output_dir.clone());
                }
                s
            }
            Err(e) => {
                eprintln!("Failed to discover tests: {}", e);
                std::process::exit(1);
            }
        };

        // Enrich discovered tests from the source tree. Build tree mirrors
        // source tree: build/test/npu-xrt/<name>/ -> test/npu-xrt/<name>/.
        //
        // From the source test.cpp we extract:
        // - buffer_spec: sizes, types, group IDs for emulator/npu-runner
        // - test_cpp_pattern: Cxxopts vs Define, so native test.exe gets
        //   the right arguments (--xclbin vs -DXCLBIN)
        let source_dir = config.npu_xrt_source_dir();
        {
            let tests_mut = suite.tests_mut();
            for test in tests_mut.iter_mut() {
                let test_source = source_dir.join(&test.name);

                // For multi-variant tests (e.g. matrix_multiplication_using_cascade/aie2_buffer),
                // the exact source path won't exist because variants don't have their own
                // directories. The test.cpp lives in the parent source directory.
                let effective_source = if test_source.exists() {
                    test_source
                } else if let Some(parent) = test_source.parent() {
                    if parent.join("test.cpp").exists() {
                        parent.to_path_buf()
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };

                test.source_dir = Some(effective_source.clone());
                if test.buffer_spec.is_none() {
                    test.buffer_spec =
                        super::test_cpp_parser::parse_test_cpp(&effective_source);
                }
                if test.test_cpp_pattern.is_none() {
                    if let Some((_, pattern)) =
                        native_hw::detect_test_cpp(&effective_source)
                    {
                        test.test_cpp_pattern = Some(pattern);
                    }
                }
            }
        }

        let all_tests: Vec<_> = suite.tests().to_vec();
        let discovered = all_tests.len();
        let tests: Vec<_> = all_tests.into_iter()
            .filter(|t| runner_config::matches_filter(&t.name, &opts.filters))
            .collect();

        // Discover pre-built Chess artifacts from build/chess/.
        // These are xclbin+insts pairs produced by a previous --chess-only run.
        // The host test.exe is compiler-agnostic, so we pair Chess xclbins
        // with the test.exe from the Peano lit tree.
        let chess_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("build/chess");
        let no_build_chess = discover_chess_artifacts(&chess_dir);
        if !no_build_chess.is_empty() {
            println!("{} Chess builds available in {}", no_build_chess.len(), chess_dir.display());
        }
        (suite, tests, discovered, no_build_chess)
    } else {
        // Source-driven path: discover from source tree, build, create suite
        println!("Discovering tests from source tree...");
        source_tests_storage = npu_test::discover(&mlir_aie_path);

        // Load test overrides (skip/expected_fail gates)
        npu_test::load_overrides(&mut source_tests_storage, &overrides_path);

        let total_source = source_tests_storage.len();
        println!("Found {} tests in source tree", total_source);

        // Filter source tests
        let filtered: Vec<&NpuTestSource> = source_tests_storage.iter()
            .filter(|t| runner_config::matches_filter(&t.name, &opts.filters))
            .collect();

        // Parallel build phase
        let env = build_env.as_ref().unwrap();
        let build_result = build_progress::run_parallel_builds(
            env,
            &filtered,
            chess_available,
            &ParallelBuildConfig {
                thread_count: opts.jobs,
                nice_level: opts.build_nice,
                verbose: opts.verbose,
                gen_sim: opts.aiesim,
                chess_only: opts.chess_only,
            },
        );

        // Compile native test.exe for hardware execution.
        let mut primary_tests = build_result.primary_tests;
        if hw_available {
            let test_exe_dir = PathBuf::from(manifest_dir).join("build/test_exe");
            let mut compiled = 0usize;
            let mut failed = 0usize;
            for test in primary_tests.iter_mut() {
                if test.test_cpp_pattern.is_none() {
                    continue;
                }
                let source_dir = match test.source_dir.as_ref() {
                    Some(d) => d,
                    None => continue,
                };
                let test_cpp = source_dir.join("test.cpp");
                if !test_cpp.exists() {
                    continue;
                }

                // Use artifact names from source test for correct preprocessor defines.
                let src = source_tests_storage.iter().find(|s| s.name == test.name);
                let xclbin_name = src.and_then(|s| s.artifact_names.xclbin_names.first().map(|n| n.as_str()));
                let insts_name = src.and_then(|s| s.artifact_names.insts_names.first().map(|n| n.as_str()));

                let output_dir = test_exe_dir.join(&test.name);
                match native_hw::compile_test_exe_with_artifacts(
                    &test_cpp, &output_dir, &mlir_aie_path,
                    xclbin_name, insts_name,
                ) {
                    Ok(exe_path) => {
                        test.test_exe = Some(exe_path);
                        compiled += 1;
                    }
                    Err(e) => {
                        log::warn!("Failed to compile test.exe for {}: {}", test.name, e);
                        failed += 1;
                    }
                }
            }
            if compiled > 0 || failed > 0 {
                println!("Native test.exe: {} compiled, {} failed", compiled, failed);
            }
        }

        // Create suite from built tests
        let mut suite = XclbinSuite::from_tests(primary_tests)
            .with_max_cycles(opts.max_cycles);
        if reference_dir.exists() {
            suite = suite.with_reference_dir(reference_dir.clone());
        }
        if npu_output_dir.exists() {
            suite = suite.with_npu_output_dir(npu_output_dir.clone());
        }

        let chess_builds_from_build = build_result.chess_artifacts;

        let tests: Vec<XclbinTest> = suite.tests().to_vec();
        (suite, tests, total_source, chess_builds_from_build)
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

    // === EMULATOR PHASE ===
    // Phase 2a: Peano emulator (parallel)
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

    // Phase 2b: Chess emulator (parallel)
    let chess_emu_results: HashMap<String, (TestOutcome, Option<Vec<u8>>)> =
        if chess_available && opts.chess_emulator && !opts.hw_only {
            let chess_tests: Vec<XclbinTest> = chess_builds.iter().filter_map(|(name, artifacts)| {
                let primary = tests.iter().find(|t| t.name == *name)?;
                let mut chess_test = XclbinTest::from_path(&artifacts.xclbin);
                chess_test.name = name.clone();
                chess_test.buffer_spec = primary.buffer_spec.clone();
                chess_test.skip_reason = primary.skip_reason.clone();
                chess_test.expected_fail_reason = primary.expected_fail_reason.clone();
                Some(chess_test)
            }).collect();

            if !chess_tests.is_empty() {
                let chess_count = chess_tests.len();
                eprintln!("Running {} Chess emulator tests with {} threads...", chess_count, opts.jobs);

                let results = if opts.jobs > 1 {
                    run_parallel(&suite, &chess_tests, opts.jobs)
                } else {
                    run_sequential(&suite, &chess_tests)
                };

                results.into_iter()
                    .map(|r| (r.name.clone(), (r.outcome, r.raw_output)))
                    .collect()
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

    // === DISPLAY + HARDWARE PHASE ===
    let mut stats = RunStats::default();
    let mut cascade = HwCascadeState::new();

    let ctx = TestContext {
        opts,
        hw_available,
        aietools: &aietools,
        chess_builds: &chess_builds,
        chess_emu_results: &chess_emu_results,
        reference_dir: &reference_dir,
        mlir_aie_path: &mlir_aie_path,
        total,
    };

    for (idx, test) in tests.iter().enumerate() {
        process_test(&ctx, &mut suite, &mut stats, &mut cascade, test, idx, &emu_results[idx]);
    }
    cascade.apply_to_stats(&mut stats);

    // === UNIT TESTS ===
    let unit_stats = if opts.unit_tests {
        Some(run_unit_tests(opts, &aietools, &build_env, &mlir_aie_path, manifest_dir))
    } else {
        None
    };

    // === SUMMARY ===
    stats.print_summary(total, opts.hw_only);

    if let Some(ref us) = unit_stats {
        if us.discovered > 0 {
            println!("\n=== UNIT TESTS ===");
            println!("Discovered:       {}", us.discovered);
            println!("Skipped:          {}", us.skipped);
            println!("Built:            {}", us.built);
            println!("Build Failed:     {}", us.build_failed);
            let sim_total = us.sim_pass + us.sim_fail + us.sim_error;
            if sim_total > 0 {
                println!("Simulated:        {}", sim_total);
                println!("Sim Pass:         {}", us.sim_pass);
                println!("Sim Fail:         {}", us.sim_fail);
                println!("Sim Error:        {}", us.sim_error);
            }
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

/// Discover pre-built Chess artifacts from a build directory.
///
/// Scans `chess_dir` for subdirectories containing xclbin+insts.bin pairs.
/// Handles both flat layout (`chess/<name>/aie.xclbin`) and nested layout
/// (`chess/<parent>/<subtest>/aie.xclbin`).
fn discover_chess_artifacts(chess_dir: &Path) -> HashMap<String, ChessBuildArtifacts> {
    let mut results = HashMap::new();
    if !chess_dir.exists() {
        return results;
    }

    // Walk the directory tree looking for xclbin files.
    fn walk(dir: &Path, prefix: &str, results: &mut HashMap<String, ChessBuildArtifacts>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                let full_name = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{}/{}", prefix, name)
                };

                // Check for xclbin in this directory
                let xclbin = find_xclbin(&path);
                if let Some(xclbin) = xclbin {
                    let insts = find_insts(&path);
                    let prj = find_prj_dir(&path);
                    results.insert(full_name.clone(), ChessBuildArtifacts {
                        xclbin,
                        insts,
                        prj_dir: prj,
                    });
                } else {
                    // Recurse into subdirectory (for nested test layout)
                    walk(&path, &full_name, results);
                }
            }
        }
    }

    fn find_insts(dir: &Path) -> Option<PathBuf> {
        let bin = dir.join("insts.bin");
        if bin.exists() { return Some(bin); }
        let elf = dir.join("insts.elf");
        if elf.exists() { return Some(elf); }
        None
    }

    fn find_xclbin(dir: &Path) -> Option<PathBuf> {
        // Prefer aie.xclbin, fall back to final.xclbin or any .xclbin
        let aie = dir.join("aie.xclbin");
        if aie.exists() { return Some(aie); }
        let fin = dir.join("final.xclbin");
        if fin.exists() { return Some(fin); }
        // Last resort: any xclbin
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "xclbin") {
                    return Some(p);
                }
            }
        }
        None
    }

    fn find_prj_dir(dir: &Path) -> Option<PathBuf> {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() && p.extension().is_some_and(|e| e == "prj") {
                    return Some(p);
                }
            }
        }
        None
    }

    walk(chess_dir, "", &mut results);
    results
}
