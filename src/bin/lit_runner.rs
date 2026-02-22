//! Lit-based test runner for mlir-aie npu-xrt tests.
//!
//! Wraps LLVM's `lit` to handle test discovery, compilation, and execution
//! (via upstream RUN lines), while adding live progress display, process
//! safety, and optional trace collection.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin lit-runner                          # run all tests
//! cargo run --bin lit-runner -- add_one               # filter by name
//! cargo run --bin lit-runner -- -v                    # verbose (show failures)
//! cargo run --bin lit-runner -- --timeout 120         # per-test timeout
//! cargo run --bin lit-runner -- --lit-args "-vv"      # extra lit flags
//! cargo run --bin lit-runner -- --trace               # trace collection mode
//! cargo run --bin lit-runner -- --trace add_one       # trace a specific test
//! cargo run --bin lit-runner -- --trace --chess-only  # trace with Chess compiler
//! ```

use std::path::PathBuf;
use std::process;
use std::time::Instant;

use xdna_emu::testing::lit_wrapper::{self, LitConfig};
use xdna_emu::testing::lit_progress::ProgressTracker;
use xdna_emu::testing::lit_trace::{self, TraceConfig, TraceOutcome};
use xdna_emu::testing::runner_config::RunnerConfig;
use xdna_emu::testing::trace_compare::{self, TraceCompareConfig};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliArgs {
    verbose: bool,
    timeout_secs: Option<u32>,
    build_dir: Option<PathBuf>,
    max_failures: Option<usize>,
    lit_args: Vec<String>,
    filters: Vec<String>,
    jobs: usize,
    watchdog_secs: u32,
    /// Enable trace collection mode (bypasses lit, runs trace pipeline).
    trace: bool,
    /// Trace buffer size in bytes (default: 1MB).
    trace_size: usize,
    /// Use Chess as the primary compiler (skip Peano).
    chess_only: bool,
    /// Enable triple trace comparison mode (HW + emulator, implies --chess-only).
    trace_all: bool,
    /// Also collect aiesimulator VCD traces (requires aietools).
    aiesim_trace: bool,
}

fn parse_cli() -> CliArgs {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut verbose = false;
    let mut timeout_secs = None;
    let mut build_dir = None;
    let mut max_failures = None;
    let mut lit_args = Vec::new();
    let mut filters = Vec::new();
    let mut jobs: usize = 1;
    let mut watchdog_secs: u32 = 3600;
    let mut trace = false;
    let mut trace_size: usize = 1_048_576;
    let mut chess_only = false;
    let mut trace_all = false;
    let mut aiesim_trace = false;

    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--verbose" | "-v" => verbose = true,
            "--trace" => trace = true,
            "--trace-all" => { trace_all = true; chess_only = true; }
            "--aiesim-trace" => aiesim_trace = true,
            "--chess-only" => chess_only = true,
            "--trace-size" => {
                let val = iter.next().unwrap_or_else(|| {
                    eprintln!("--trace-size requires a value");
                    process::exit(1);
                });
                trace_size = val.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --trace-size value: {}", val);
                    process::exit(1);
                });
            }
            "--timeout" => {
                let val = iter.next().unwrap_or_else(|| {
                    eprintln!("--timeout requires a value");
                    process::exit(1);
                });
                timeout_secs = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --timeout value: {}", val);
                    process::exit(1);
                }));
            }
            "--build-dir" => {
                let val = iter.next().unwrap_or_else(|| {
                    eprintln!("--build-dir requires a value");
                    process::exit(1);
                });
                build_dir = Some(PathBuf::from(val));
            }
            "--max-failures" => {
                let val = iter.next().unwrap_or_else(|| {
                    eprintln!("--max-failures requires a value");
                    process::exit(1);
                });
                max_failures = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --max-failures value: {}", val);
                    process::exit(1);
                }));
            }
            "--lit-args" => {
                let val = iter.next().unwrap_or_else(|| {
                    eprintln!("--lit-args requires a value");
                    process::exit(1);
                });
                // Split on whitespace to allow --lit-args "-vv --param foo"
                lit_args.extend(val.split_whitespace().map(String::from));
            }
            "-j" => {
                let val = iter.next().unwrap_or_else(|| {
                    eprintln!("-j requires a value");
                    process::exit(1);
                });
                jobs = val.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid -j value: {}", val);
                    process::exit(1);
                });
                if jobs == 0 { jobs = 1; }
            }
            "--watchdog" => {
                let val = iter.next().unwrap_or_else(|| {
                    eprintln!("--watchdog requires a value");
                    process::exit(1);
                });
                watchdog_secs = val.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --watchdog value: {}", val);
                    process::exit(1);
                });
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            _ if arg.starts_with('-') => {
                eprintln!("Unknown option: {}", arg);
                print_usage();
                process::exit(1);
            }
            _ => filters.push(arg.clone()),
        }
    }

    CliArgs {
        verbose,
        timeout_secs,
        build_dir,
        max_failures,
        lit_args,
        filters,
        jobs,
        watchdog_secs,
        trace,
        trace_size,
        chess_only,
        trace_all,
        aiesim_trace,
    }
}

fn print_usage() {
    eprintln!("lit-runner: Run mlir-aie npu-xrt tests via lit");
    eprintln!();
    eprintln!("Usage: lit-runner [OPTIONS] [FILTER...]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -v, --verbose         Show full output for failing tests");
    eprintln!("  -j N                  Lit parallelism (default: 1)");
    eprintln!("  --timeout SECS        Per-test timeout (forwarded to lit)");
    eprintln!("  --build-dir PATH      mlir-aie build directory (auto-detected)");
    eprintln!("  --max-failures N      Stop after N failures");
    eprintln!("  --lit-args \"...\"      Extra arguments passed to lit verbatim");
    eprintln!("  --watchdog SECS       Outer watchdog for entire lit process (default: 3600)");
    eprintln!("  --trace               Enable trace collection mode (bypasses lit)");
    eprintln!("  --trace-size BYTES    Trace buffer size (default: 1048576 = 1MB)");
    eprintln!("  --chess-only          Use Chess compiler for trace builds (requires aietools)");
    eprintln!("                        Only affects --trace mode; lit mode reads RUN lines directly");
    eprintln!("  --trace-all           Triple trace mode: HW + emulator (implies --chess-only)");
    eprintln!("  --aiesim-trace        Also collect aiesimulator VCD traces (use with --trace-all)");
    eprintln!("  -h, --help            Show this help");
    eprintln!();
    eprintln!("Filters are substring matches on test names, OR-ed together.");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  lit-runner                             # run all tests via lit");
    eprintln!("  lit-runner add_one vec_scalar           # run matching tests");
    eprintln!("  lit-runner -v --timeout 120             # verbose with 2min timeout");
    eprintln!("  lit-runner --trace                      # trace all tests on NPU (HW only)");
    eprintln!("  lit-runner --trace add_one              # trace specific test");
    eprintln!("  lit-runner --trace --trace-size 2097152 # trace with 2MB buffer");
    eprintln!("  lit-runner --trace-all add_one          # triple trace: HW + emulator");
    eprintln!("  lit-runner --trace-all --aiesim-trace   # triple trace: HW + emu + aiesim");
}

// ---------------------------------------------------------------------------
// Trace mode
// ---------------------------------------------------------------------------

/// Run trace collection mode: inject traces, compile, execute on NPU.
fn run_trace_mode(cli: &CliArgs) {
    let config = xdna_emu::config::Config::get();
    let mlir_aie_source = PathBuf::from(config.mlir_aie_path());
    let mlir_aie_build = mlir_aie_source.join("build");

    // Detect xdna-emu project root (where tools/ lives)
    let xdna_emu_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Read runner.toml for config defaults, merge with CLI flags.
    let runner_config = RunnerConfig::load(&xdna_emu_root.to_string_lossy());
    let use_chess = cli.chess_only || runner_config.chess.chess_only;

    // Early validation: --chess-only needs aietools
    if use_chess && config.aietools_path().is_none() {
        eprintln!("Error: --chess-only requires aietools with xchesscc");
        process::exit(1);
    }

    // Detect Python interpreter from mlir-aie venv
    let python = detect_python(&mlir_aie_source);

    // Detect aiecc.py
    let aiecc = detect_aiecc(&mlir_aie_source);

    // Build config and resolve all paths to absolute so subprocesses
    // are independent of the runner's working directory.
    let trace_config = TraceConfig {
        xdna_emu_root: xdna_emu_root.clone(),
        mlir_aie_source: mlir_aie_source.clone(),
        mlir_aie_build,
        build_traced_dir: xdna_emu_root.join("build/traced-tests"),
        traces_dir: xdna_emu_root.join("build/traces"),
        trace_size: cli.trace_size,
        python,
        aiecc,
        inject_timeout: 120,
        compile_timeout: cli.timeout_secs.unwrap_or(600),
        execute_timeout: cli.timeout_secs.unwrap_or(120),
        aietools_path: config.aietools_path(),
        use_chess,
    }
    .resolve();

    // Discover tests from the source tree
    let test_source_dir = trace_config.mlir_aie_source.join("test/npu-xrt");
    if !test_source_dir.exists() {
        eprintln!("Error: test source directory not found: {}", test_source_dir.display());
        process::exit(1);
    }

    let all_tests = lit_trace::discover_tests(&test_source_dir);
    println!("Discovered {} tests in {}", all_tests.len(), test_source_dir.display());

    // Apply filters
    let tests: Vec<_> = if cli.filters.is_empty() {
        all_tests
    } else {
        all_tests
            .into_iter()
            .filter(|(name, _)| cli.filters.iter().any(|f| name.contains(f)))
            .collect()
    };

    if tests.is_empty() {
        eprintln!("No tests match the given filters.");
        process::exit(1);
    }

    println!("Running trace pipeline on {} tests", tests.len());
    println!("  Compiler:     {}", if trace_config.use_chess { "Chess (--chess-only)" } else { "Peano" });
    println!("  Trace buffer: {} bytes", trace_config.trace_size);
    println!("  Build dir:    {}", trace_config.build_traced_dir.display());
    println!("  Traces dir:   {}", trace_config.traces_dir.display());
    println!();

    // Run each test
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
                if cli.verbose {
                    println!("  skip: {}", reason);
                }
                "SKIPPED"
            }
            TraceOutcome::InjectFailed { stderr } => {
                inject_fail += 1;
                if cli.verbose {
                    let preview: String = stderr.lines().take(5).collect::<Vec<_>>().join("\n  ");
                    println!("  inject error:\n  {}", preview);
                }
                "INJECT FAIL"
            }
            TraceOutcome::CompileFailed { stderr } => {
                compile_fail += 1;
                if cli.verbose {
                    let preview: String = stderr.lines().take(5).collect::<Vec<_>>().join("\n  ");
                    println!("  compile error:\n  {}", preview);
                }
                "COMPILE FAIL"
            }
            TraceOutcome::RunFailed { stderr } => {
                run_fail += 1;
                if cli.verbose {
                    let preview: String = stderr.lines().take(5).collect::<Vec<_>>().join("\n  ");
                    println!("  run error:\n  {}", preview);
                }
                "RUN FAIL"
            }
            TraceOutcome::Timeout { stage } => {
                timeout_count += 1;
                if cli.verbose {
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

        // Pad or truncate name to 40 chars for alignment
        let display_name = if name.len() > 40 {
            format!("{}...", &name[..37])
        } else {
            format!("{:<40}", name)
        };

        println!(
            "[{:>3}/{}] {}  {:<13} ({:.1}s)",
            i + 1, total, display_name, label, elapsed,
        );

        // Check max failures
        if let Some(max) = cli.max_failures {
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

/// Find the Python interpreter from the mlir-aie venv.
fn detect_python(mlir_aie_source: &Path) -> PathBuf {
    // Check venv first
    let venv_python = mlir_aie_source.join("ironenv/bin/python3");
    if venv_python.exists() {
        return venv_python;
    }
    let venv_python = mlir_aie_source.join("ironenv/bin/python");
    if venv_python.exists() {
        return venv_python;
    }
    // Fall back to system Python
    PathBuf::from("python3")
}

/// Find aiecc.py in the mlir-aie installation.
fn detect_aiecc(mlir_aie_source: &Path) -> PathBuf {
    // Check install path
    let install = mlir_aie_source.join("install/bin/aiecc.py");
    if install.exists() {
        return install;
    }
    let my_install = mlir_aie_source.join("my_install/bin/aiecc.py");
    if my_install.exists() {
        return my_install;
    }
    // Check if aiecc.py is on PATH
    PathBuf::from("aiecc.py")
}

use std::path::Path;

// ---------------------------------------------------------------------------
// Triple trace mode (--trace-all)
// ---------------------------------------------------------------------------

/// Run triple trace comparison: HW + emulator (+ optional aiesimulator).
fn run_trace_all_mode(cli: &CliArgs) {
    let config = xdna_emu::config::Config::get();
    let mlir_aie_source = PathBuf::from(config.mlir_aie_path());
    let mlir_aie_build = mlir_aie_source.join("build");

    let xdna_emu_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let runner_config = RunnerConfig::load(&xdna_emu_root.to_string_lossy());

    // --trace-all implies Chess
    if config.aietools_path().is_none() {
        eprintln!("Error: --trace-all requires aietools (Chess compiler + aiesimulator)");
        process::exit(1);
    }

    if cli.aiesim_trace {
        // Verify aiesimulator is available
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

    let python = detect_python(&mlir_aie_source);
    let aiecc = detect_aiecc(&mlir_aie_source);

    let trace_config = TraceConfig {
        xdna_emu_root: xdna_emu_root.clone(),
        mlir_aie_source: mlir_aie_source.clone(),
        mlir_aie_build,
        build_traced_dir: xdna_emu_root.join("build/traced-tests"),
        traces_dir: xdna_emu_root.join("build/traces"),
        trace_size: cli.trace_size,
        python,
        aiecc,
        inject_timeout: 120,
        compile_timeout: cli.timeout_secs.unwrap_or(600),
        execute_timeout: cli.timeout_secs.unwrap_or(120),
        aietools_path: config.aietools_path(),
        use_chess: true,
    }
    .resolve();

    let compare_config = TraceCompareConfig {
        normal_build_dir: xdna_emu_root.join("build/normal-chess"),
        aiesim_trace: cli.aiesim_trace,
        max_cycles: runner_config.execution.max_cycles,
        trace: trace_config,
    };

    // Discover tests
    let test_source_dir = compare_config.trace.mlir_aie_source.join("test/npu-xrt");
    if !test_source_dir.exists() {
        eprintln!("Error: test source directory not found: {}", test_source_dir.display());
        process::exit(1);
    }

    let all_tests = lit_trace::discover_tests(&test_source_dir);
    println!("Discovered {} tests in {}", all_tests.len(), test_source_dir.display());

    // Apply filters
    let tests: Vec<_> = if cli.filters.is_empty() {
        all_tests
    } else {
        all_tests
            .into_iter()
            .filter(|(name, _)| cli.filters.iter().any(|f| name.contains(f)))
            .collect()
    };

    if tests.is_empty() {
        eprintln!("No tests match the given filters.");
        process::exit(1);
    }

    println!("Triple trace comparison on {} tests", tests.len());
    println!("  Compiler:     Chess (--trace-all implies --chess-only)");
    println!("  Sources:      HW + Emulator{}", if cli.aiesim_trace { " + aiesimulator" } else { "" });
    println!("  Trace buffer: {} bytes", compare_config.trace.trace_size);
    println!("  Emulator:     reuses traced xclbin (single compilation)");
    if cli.aiesim_trace {
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
// Lit mode (original behavior)
// ---------------------------------------------------------------------------

fn run_lit_mode(cli: &CliArgs) {
    // Detect build directory
    let build_dir = match lit_wrapper::detect_build_dir(cli.build_dir.as_deref()) {
        Ok(dir) => dir,
        Err(msg) => {
            eprintln!("Error: {}", msg);
            process::exit(1);
        }
    };

    let test_dir = build_dir.join("test/npu-xrt");
    println!("Test directory: {}", test_dir.display());

    // Configure lit
    let config = LitConfig {
        test_dir,
        timeout_secs: cli.timeout_secs,
        jobs: cli.jobs,
        extra_args: cli.lit_args.clone(),
        filters: cli.filters.clone(),
        watchdog_secs: cli.watchdog_secs,
    };

    // Progress tracking
    let mut tracker = ProgressTracker::new(cli.verbose);
    let max_failures = cli.max_failures;
    let mut failure_count: usize = 0;

    // Run lit with live progress callback
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

    // The tracker is moved into the callback, so we need a second one
    // for the summary. This is a limitation of the current design --
    // we'll use the JSON output for the summary instead.
    let verbose_for_summary = cli.verbose;

    let (json_path, exit_code) = match lit_wrapper::run_lit(&config, callback) {
        Ok(result) => result,
        Err(msg) => {
            eprintln!("Error: {}", msg);
            process::exit(1);
        }
    };

    // Parse JSON output for structured summary
    let json_summary = match lit_wrapper::parse_json_output(&json_path) {
        Ok(summary) => Some(summary),
        Err(msg) => {
            log::warn!("Could not parse lit JSON output: {}", msg);
            None
        }
    };

    // Print summary using a fresh tracker (the original was moved into callback)
    let summary_tracker = ProgressTracker::new(verbose_for_summary);
    summary_tracker.print_summary(json_summary.as_ref());

    // Clean up temp file
    std::fs::remove_file(&json_path).ok();

    // Exit with lit's exit code so CI detects failures
    process::exit(exit_code);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn"),
    )
    .init();

    let cli = parse_cli();

    if cli.trace_all {
        run_trace_all_mode(&cli);
    } else if cli.trace {
        run_trace_mode(&cli);
    } else {
        run_lit_mode(&cli);
    }
}
