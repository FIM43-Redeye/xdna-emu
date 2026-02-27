//! Unified test runner for mlir-aie npu-xrt tests.
//!
//! Combines the emulator validation runner, LLVM lit wrapper, and trace
//! collection pipeline into a single binary with mode selection.
//!
//! # Modes
//!
//! - **(default)** Emulator + hardware validation: discover, build, run
//!   through emulator, optionally validate on real NPU hardware.
//! - **--lit** Wrap LLVM's `lit` for standard test execution with live
//!   progress display and process safety.
//! - **--trace** Inject hardware tracing, compile, and execute on real NPU.
//! - **--trace-all** Triple comparison: HW + emulator (+ optional aiesimulator).
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin npu-test                        # emulator + hardware
//! cargo run --bin npu-test -- --lit               # lit wrapper
//! cargo run --bin npu-test -- --trace             # trace collection
//! cargo run --bin npu-test -- --trace-all         # triple comparison
//! cargo run --bin npu-test -- add_one             # filter by name
//! cargo run --bin npu-test -- -v -j 4             # verbose, 4 threads
//! ```

use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use xdna_emu::testing::lit_wrapper::{self, LitConfig};
use xdna_emu::testing::lit_progress::ProgressTracker;
use xdna_emu::testing::lit_trace::{self, TraceConfig, TraceOutcome};
use xdna_emu::testing::trace_compare::{self, TraceCompareConfig};
use xdna_emu::testing::runner_config::{self, RunMode, RunnerConfig};
use xdna_emu::testing::emu_runner;

// ---------------------------------------------------------------------------
// Lit mode
// ---------------------------------------------------------------------------

/// Run tests via LLVM's lit framework with live progress display.
fn run_lit_mode(opts: &runner_config::Options) {
    let build_dir = match lit_wrapper::detect_build_dir(opts.build_dir.as_deref()) {
        Ok(dir) => dir,
        Err(msg) => {
            eprintln!("Error: {}", msg);
            process::exit(1);
        }
    };

    let test_dir = build_dir.join("test/npu-xrt");
    println!("Test directory: {}", test_dir.display());

    let config = LitConfig {
        test_dir,
        timeout_secs: opts.timeout_secs,
        jobs: opts.jobs,
        extra_args: opts.lit_args.clone(),
        filters: opts.filters.clone(),
        watchdog_secs: opts.watchdog_secs,
    };

    let mut tracker = ProgressTracker::new(opts.verbose);
    let max_failures = opts.max_failures;
    let mut failure_count: usize = 0;

    let callback: lit_wrapper::OnResult = Box::new(move |result, elapsed| {
        tracker.record(result, elapsed);

        if !result.code.is_success()
            && result.code != lit_wrapper::LitResultCode::Unsupported
        {
            failure_count += 1;
        }

        if let Some(max) = max_failures {
            if failure_count >= max {
                eprintln!("Reached --max-failures={}, stopping.", max);
            }
        }
    });

    let verbose_for_summary = opts.verbose;

    let (json_path, exit_code) = match lit_wrapper::run_lit(&config, callback) {
        Ok(result) => result,
        Err(msg) => {
            eprintln!("Error: {}", msg);
            process::exit(1);
        }
    };

    let json_summary = match lit_wrapper::parse_json_output(&json_path) {
        Ok(summary) => Some(summary),
        Err(msg) => {
            log::warn!("Could not parse lit JSON output: {}", msg);
            None
        }
    };

    let summary_tracker = ProgressTracker::new(verbose_for_summary);
    summary_tracker.print_summary(json_summary.as_ref());

    std::fs::remove_file(&json_path).ok();
    process::exit(exit_code);
}

// ---------------------------------------------------------------------------
// Trace mode
// ---------------------------------------------------------------------------

/// Build a TraceConfig from Options and discovered paths.
fn build_trace_config(opts: &runner_config::Options) -> TraceConfig {
    let config = xdna_emu::config::Config::get();
    let mlir_aie_source = PathBuf::from(config.mlir_aie_path());
    let mlir_aie_build = mlir_aie_source.join("build");
    let xdna_emu_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let runner_config = RunnerConfig::load(&xdna_emu_root.to_string_lossy());
    let use_chess = opts.chess_only || runner_config.chess.chess_only;

    if use_chess && config.aietools_path().is_none() {
        eprintln!("Error: --chess-only requires aietools with xchesscc");
        process::exit(1);
    }

    let python = detect_python(&mlir_aie_source);
    let aiecc = detect_aiecc(&mlir_aie_source);

    TraceConfig {
        xdna_emu_root,
        mlir_aie_source,
        mlir_aie_build,
        build_traced_dir: std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("build/traced-tests"),
        traces_dir: std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("build/traces"),
        trace_size: opts.trace_size,
        python,
        aiecc,
        inject_timeout: 120,
        compile_timeout: opts.timeout_secs.unwrap_or(600),
        execute_timeout: opts.timeout_secs.unwrap_or(120),
        aietools_path: config.aietools_path(),
        use_chess,
    }
    .resolve()
}

/// Discover and filter tests from the source tree.
fn discover_and_filter(
    trace_config: &TraceConfig,
    filters: &[String],
) -> Vec<(String, PathBuf)> {
    let test_source_dir = trace_config.mlir_aie_source.join("test/npu-xrt");
    if !test_source_dir.exists() {
        eprintln!("Error: test source directory not found: {}", test_source_dir.display());
        process::exit(1);
    }

    let all_tests = lit_trace::discover_tests(&trace_config.mlir_aie_source);
    println!("Discovered {} tests in {}", all_tests.len(), test_source_dir.display());

    let tests: Vec<_> = if filters.is_empty() {
        all_tests
    } else {
        all_tests
            .into_iter()
            .filter(|(name, _)| filters.iter().any(|f| name.contains(f)))
            .collect()
    };

    if tests.is_empty() {
        eprintln!("No tests match the given filters.");
        process::exit(1);
    }

    tests
}

/// Run trace collection mode: inject traces, compile, execute on NPU.
fn run_trace_mode(opts: &runner_config::Options) {
    let trace_config = build_trace_config(opts);
    let tests = discover_and_filter(&trace_config, &opts.filters);

    println!("Running trace pipeline on {} tests", tests.len());
    println!("  Compiler:     {}", if trace_config.use_chess { "Chess (--chess-only)" } else { "Peano" });
    println!("  Trace buffer: {} bytes", trace_config.trace_size);
    println!("  Build dir:    {}", trace_config.build_traced_dir.display());
    println!("  Traces dir:   {}", trace_config.traces_dir.display());
    println!();

    let total = tests.len();
    let mut success_count = 0usize;
    let mut skip_count = 0usize;
    let mut inject_fail = 0usize;
    let mut compile_fail = 0usize;
    let mut run_fail = 0usize;
    let mut timeout_count = 0usize;
    let mut wedged = false;

    for (i, (name, source_dir)) in tests.iter().enumerate() {
        if wedged {
            println!("[{:>3}/{}] {:<40}  SKIPPED (device wedged)", i + 1, total, name);
            skip_count += 1;
            continue;
        }

        let start = Instant::now();
        let outcome = lit_trace::run_trace_pipeline(name, source_dir, &trace_config);
        let elapsed = start.elapsed().as_secs_f64();

        let label = match &outcome {
            TraceOutcome::Success { .. } => {
                success_count += 1;
                "TRACE OK"
            }
            TraceOutcome::Skipped { reason } => {
                skip_count += 1;
                if opts.verbose {
                    println!("  skip: {}", reason);
                }
                "SKIPPED"
            }
            TraceOutcome::InjectFailed { stderr } => {
                inject_fail += 1;
                if opts.verbose {
                    let preview: String = stderr.lines().take(5).collect::<Vec<_>>().join("\n  ");
                    println!("  inject error:\n  {}", preview);
                }
                "INJECT FAIL"
            }
            TraceOutcome::CompileFailed { stderr } => {
                compile_fail += 1;
                if opts.verbose {
                    let preview: String = stderr.lines().take(5).collect::<Vec<_>>().join("\n  ");
                    println!("  compile error:\n  {}", preview);
                }
                "COMPILE FAIL"
            }
            TraceOutcome::RunFailed { stderr } => {
                run_fail += 1;
                if opts.verbose {
                    let preview: String = stderr.lines().take(5).collect::<Vec<_>>().join("\n  ");
                    println!("  run error:\n  {}", preview);
                }
                "RUN FAIL"
            }
            TraceOutcome::Timeout { stage } => {
                timeout_count += 1;
                if opts.verbose {
                    println!("  timeout in: {}", stage);
                }
                "TIMEOUT"
            }
            TraceOutcome::Wedged => {
                wedged = true;
                eprintln!("DEVICE WEDGED -- stopping all hardware tests");
                "WEDGED"
            }
        };

        let display_name = if name.len() > 40 {
            format!("{}...", &name[..37])
        } else {
            format!("{:<40}", name)
        };

        println!(
            "[{:>3}/{}] {}  {:<13} ({:.1}s)",
            i + 1, total, display_name, label, elapsed,
        );

        if let Some(max) = opts.max_failures {
            let failures = inject_fail + compile_fail + run_fail;
            if failures >= max {
                eprintln!("Reached --max-failures={}, stopping.", max);
                break;
            }
        }
    }

    // Summary
    println!();
    println!("Trace collection summary:");
    println!("  {} success", success_count);
    if skip_count > 0 { println!("  {} skipped", skip_count); }
    if inject_fail > 0 { println!("  {} inject failures", inject_fail); }
    if compile_fail > 0 { println!("  {} compile failures", compile_fail); }
    if run_fail > 0 { println!("  {} run failures", run_fail); }
    if timeout_count > 0 { println!("  {} timeouts", timeout_count); }
    if wedged { println!("  DEVICE WEDGED"); }

    let failures = inject_fail + compile_fail + run_fail + timeout_count;
    if failures > 0 || wedged {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Triple trace mode (--trace-all)
// ---------------------------------------------------------------------------

/// Run triple trace comparison: HW + emulator (+ optional aiesimulator).
fn run_trace_all_mode(opts: &runner_config::Options) {
    let config = xdna_emu::config::Config::get();

    if config.aietools_path().is_none() {
        eprintln!("Error: --trace-all requires aietools (Chess compiler + aiesimulator)");
        process::exit(1);
    }

    if opts.aiesim_trace {
        if let Some(tools) = xdna_emu::integration::aietools::AieTools::discover(&config) {
            if tools.aiesimulator.is_none() {
                eprintln!("Error: --aiesim-trace requires aiesimulator in aietools");
                process::exit(1);
            }
        } else {
            eprintln!("Error: --aiesim-trace requires aietools");
            process::exit(1);
        }
    }

    let trace_config = build_trace_config(opts);

    let xdna_emu_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let runner_cfg = RunnerConfig::load(&xdna_emu_root.to_string_lossy());

    let compare_config = TraceCompareConfig {
        normal_build_dir: xdna_emu_root.join("build/normal-chess"),
        aiesim_trace: opts.aiesim_trace,
        max_cycles: runner_cfg.execution.max_cycles,
        trace: trace_config,
    };

    let tests = discover_and_filter(&compare_config.trace, &opts.filters);

    println!("Triple trace comparison on {} tests", tests.len());
    println!("  Compiler:     Chess (--trace-all implies --chess-only)");
    println!("  Sources:      HW + Emulator{}", if opts.aiesim_trace { " + aiesimulator" } else { "" });
    println!("  Trace buffer: {} bytes", compare_config.trace.trace_size);
    println!("  Emulator:     reuses traced xclbin (single compilation)");
    if opts.aiesim_trace {
        println!("  aiesim build: {}", compare_config.normal_build_dir.display());
    }
    println!("  Traces dir:   {}", compare_config.trace.traces_dir.display());
    println!();

    let summary = trace_compare::run_trace_compare(&tests, &compare_config);
    summary.print();

    if summary.has_failures() || summary.wedged {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Find the Python interpreter from the mlir-aie venv.
fn detect_python(mlir_aie_source: &Path) -> PathBuf {
    let venv_python = mlir_aie_source.join("ironenv/bin/python3");
    if venv_python.exists() {
        return venv_python;
    }
    let venv_python = mlir_aie_source.join("ironenv/bin/python");
    if venv_python.exists() {
        return venv_python;
    }
    PathBuf::from("python3")
}

/// Find aiecc.py in the mlir-aie installation.
fn detect_aiecc(mlir_aie_source: &Path) -> PathBuf {
    let install = mlir_aie_source.join("install/bin/aiecc.py");
    if install.exists() {
        return install;
    }
    let my_install = mlir_aie_source.join("my_install/bin/aiecc.py");
    if my_install.exists() {
        return my_install;
    }
    PathBuf::from("aiecc.py")
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn"),
    )
    .init();

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let runner_config = RunnerConfig::load(manifest_dir);
    runner_config.print_active_overrides();
    let opts = runner_config::parse_args(&runner_config);

    match opts.mode {
        RunMode::EmuHw => emu_runner::run(&opts),
        RunMode::Lit => run_lit_mode(&opts),
        RunMode::Trace => run_trace_mode(&opts),
        RunMode::TraceAll => run_trace_all_mode(&opts),
    }
}
