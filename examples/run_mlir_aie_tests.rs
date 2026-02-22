//! Run all xclbin tests from mlir-aie npu-xrt directory.
//!
//! This is a quick test runner to see how many mlir-aie tests pass.
//! Validates output against hardware reference captures when available.
//!
//! Usage:
//!   cargo run --example run_mlir_aie_tests [OPTIONS] [FILTER...]
//!
//! Options:
//!   --verbose, -v     Show full expected/actual output arrays for failures
//!   -j N              Run N tests in parallel (default: auto, up to 8)
//!   --elfanalyze      Run elfanalyzer on each test's ELFs (requires aietools)
//!   --no-chess        Disable Chess compiler builds (Chess is auto-detected by default)
//!   --chess-only      Use Chess as the primary compiler for all tests (skip Peano)
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
//!   cargo run --example run_mlir_aie_tests -- --chess-only add_one   # Chess only (skip Peano)
//!   cargo run --example run_mlir_aie_tests -- --no-hw add_one       # skip real NPU
//!   cargo run --example run_mlir_aie_tests -- --full add_one        # full validation matrix
//!   cargo run --example run_mlir_aie_tests -- --full                # full validation, all tests

use std::path::PathBuf;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use xdna_emu::testing::xclbin_suite::{Compiler, XclbinSuite, XclbinTest, TestOutcome};
use xdna_emu::testing::test_cpp_parser::{BufferDir, read_values};
use xdna_emu::testing::hardware_comparison::{
    CompilerComparison, CompilerDiagnosis, load_hw_reference,
};
use xdna_emu::testing::npu_runner;
use xdna_emu::testing::npu_test::{self, NpuTestSource};
use xdna_emu::testing::runner_config::{self, RunnerConfig};
use xdna_emu::testing::runner_stats::{RunStats, HwRunResult};
use xdna_emu::testing::hw_executor;
use xdna_emu::testing::native_hw;
use xdna_emu::testing::runner_display::{self, TestResult};
use xdna_emu::integration::aietools::AieTools;
use xdna_emu::integration::chess_build::BuildEnv;
use xdna_emu::build_progress::{self, ParallelBuildConfig};
use xdna_emu::testing::unit_test;

/// Type alias for Chess build artifacts (from build_progress module).
type ChessBuildArtifacts = build_progress::ChessArtifacts;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error"),
    )
    .init();

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let runner_config = RunnerConfig::load(manifest_dir);
    runner_config.print_active_overrides();
    let opts = runner_config::parse_args(&runner_config);

    let config = xdna_emu::config::Config::get();
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
    // Native test.exe execution (preferred) only needs NPU + XRT.
    // npu-runner (fallback) also needs the npu_runner binary built.
    let hw_available = if opts.hw {
        if !npu_runner::npu_available() {
            println!("hw: NPU not available, hardware validation disabled");
            false
        } else {
            // NPU is available. Native test.exe path works regardless of npu-runner.
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
    // Only assigned in the source-driven (non --no-build) path.
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

        let suite = match XclbinSuite::discover(&npu_xrt_path) {
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

        let all_tests: Vec<_> = suite.tests().to_vec();
        let discovered = all_tests.len();
        let tests: Vec<_> = all_tests.into_iter()
            .filter(|t| runner_config::matches_filter(&t.name, &opts.filters))
            .collect();

        let no_build_chess: std::collections::HashMap<String, ChessBuildArtifacts> =
            std::collections::HashMap::new();
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
        // This is fast (~0.5s each) and sequential is fine.
        let mut primary_tests = build_result.primary_tests;
        if hw_available {
            let test_exe_dir = PathBuf::from(manifest_dir).join("build/test_exe");
            let mut compiled = 0usize;
            let mut failed = 0usize;
            for test in primary_tests.iter_mut() {
                // Only compile if the source test has a test.cpp
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

                let output_dir = test_exe_dir.join(&test.name);
                match native_hw::compile_test_exe(&test_cpp, &output_dir, &mlir_aie_path) {
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

        let tests: Vec<_> = suite.tests().to_vec();
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
    let chess_emu_results: std::collections::HashMap<String, (TestOutcome, Option<Vec<u8>>)> =
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
                std::collections::HashMap::new()
            }
        } else {
            std::collections::HashMap::new()
        };

    // === DISPLAY + HARDWARE PHASE ===
    let mut stats = RunStats::default();

    // Cascade detection: stop hammering hardware after consecutive errors.
    // A "FAIL (3/8)" means the device works (test just got wrong output).
    // Only "ERROR (...)" results indicate a potential device problem.
    const MAX_CONSECUTIVE_HW_ERRORS: usize = 3;
    let mut consecutive_hw_errors: usize = 0;
    let mut hw_disabled_reason: Option<String> = None;

    for (idx, test) in tests.iter().enumerate() {
        let emu = &emu_results[idx];
        let compact = emu.is_none(); // hw-only mode: compact single-line output

        // --- Emulator display (skip in hw-only) ---
        if let Some(ref r) = emu {
            println!("{}", runner_display::format_result(r, total));
            stats.record_emu_outcome(&r.outcome);
            if let Some(ref hv) = r.hw_validation {
                stats.record_hw_validation(hv);
            }
            if let TestOutcome::UnknownOpcode { ref details, .. } = r.outcome {
                suite.record_unknown(details.clone(), &r.name);
            }

            // Verbose mode: print full expected vs actual comparison
            if opts.verbose {
                if test.buffer_spec.is_some() {
                    if let Some(ref output) = r.raw_output {
                        let show = matches!(&r.outcome,
                            TestOutcome::ValidationFail { .. } |
                            TestOutcome::ExpectedFail { .. } |
                            TestOutcome::UnexpectedPass { .. } |
                            TestOutcome::Pass { correct: Some(_), .. }
                        );
                        if show {
                            runner_display::print_verbose_comparison(output, test, &reference_dir);
                        }
                    }
                }
            }
        }

        // --- Skip gates ---
        let skip_hw = emu.as_ref().map_or(false, |r|
            matches!(&r.outcome, TestOutcome::Platform{..} | TestOutcome::Skipped{..}));

        let spec = test.buffer_spec.as_ref();
        let insts_path = test.find_insts_bin();

        // In hw-only mode, skip tests that cannot run on hardware at all.
        // Native test.exe only needs insts; npu-runner also needs buffer spec.
        if compact {
            let has_native = test.test_exe.is_some();
            let has_runner = spec.is_some() && insts_path.is_some();
            if !has_native && !has_runner {
                let reason = if insts_path.is_none() { "no insts.bin" }
                    else { "no buffer spec or test.exe" };
                println!("[{:2}/{}] {:45} SKIP ({})",
                    idx + 1, total, &test.name[..test.name.len().min(45)], reason);
                stats.skipped += 1;
                continue;
            }
            // Print compact header (results appended on same line)
            print!("[{:2}/{}] {:35} ",
                idx + 1, total, &test.name[..test.name.len().min(35)]);
        }

        // --- Primary HW (native test.exe preferred, npu-runner fallback) ---
        let compiler = test.compiler.unwrap_or(Compiler::Peano);
        let hw_cascade_active = hw_disabled_reason.is_some();
        let primary_hw: Option<HwRunResult> = if hw_cascade_active {
            // Device was declared wedged -- skip silently (counted below)
            stats.hw_cascade_skipped += 1;
            None
        } else if hw_available && !skip_hw {
            let prefix = match compiler {
                Compiler::Peano => if compact { "peano" } else { "peano-hw" },
                Compiler::Chess => if compact { "chess" } else { "chess-hw" },
            };

            // Try native test.exe first (handles custom host logic correctly)
            let hw = if let (Some(ref exe), Some(ref pattern), Some(ref insts)) =
                (&test.test_exe, &test.test_cpp_pattern, &insts_path)
            {
                Some(native_hw::run_native_and_print(
                    exe, &test.xclbin_path, insts, *pattern, prefix, compact,
                ))
            } else if let (Some(s), Some(ref insts)) = (spec, &insts_path) {
                // Fall back to npu-runner for tests without test.exe
                Some(hw_executor::run_hw_and_print(
                    s, &test.name, &test.xclbin_path, insts, prefix, compact, &reference_dir,
                ))
            } else {
                None
            };

            if let Some(ref hw) = hw {
                stats.record_hw(compiler, hw);

                // Immediate wedge cutoff: D-state means the device is
                // unrecoverable without a reboot. Stop all HW immediately.
                if hw.wedged && hw_disabled_reason.is_none() {
                    let detail = format!(
                        "NPU device wedged (D-state) at test {} -- \
                         process survived SIGKILL, reboot required",
                        idx + 1,
                    );
                    eprintln!("\nCRITICAL: {}", detail);
                    eprintln!("Disabling hardware execution for remaining tests.\n");
                    stats.hw_cascade_stopped_at = Some(idx + 1);
                    hw_disabled_reason = Some(detail);
                }

                // Cascade detection: track consecutive ERROR results.
                // PASS or FAIL means the device is working (test passed or
                // produced wrong output). Only ERROR means the device itself
                // may be wedged.
                if hw.passed || hw.label.starts_with("FAIL") {
                    consecutive_hw_errors = 0;
                } else if hw_disabled_reason.is_none() {
                    consecutive_hw_errors += 1;
                    if consecutive_hw_errors >= MAX_CONSECUTIVE_HW_ERRORS {
                        let detail = if !npu_runner::probe_device_health() {
                            format!(
                                "NPU device unreachable after {} consecutive errors (test {})",
                                consecutive_hw_errors, idx + 1,
                            )
                        } else {
                            format!(
                                "NPU device wedged after {} consecutive errors at test {} \
                                 (device node exists but XRT operations fail)",
                                consecutive_hw_errors, idx + 1,
                            )
                        };
                        eprintln!("\nWARNING: {}", detail);
                        eprintln!("Disabling hardware execution for remaining tests.\n");
                        stats.hw_cascade_stopped_at = Some(idx + 1);
                        hw_disabled_reason = Some(detail);
                    }
                }
            }

            hw
        } else {
            None
        };

        // --- elfanalyzer ---
        if opts.elfanalyze && !compact {
            if let Some(ref tools) = aietools {
                runner_display::run_elfanalyzer(tools, test, &mlir_aie_path);
            }
        }

        // --- Chess emulator (pre-computed in parallel phase 2b) ---
        let chess_emu_output: Option<Vec<u8>> = if let Some((ref outcome, ref raw)) =
            chess_emu_results.get(&test.name)
        {
            let label = match outcome {
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
            if !compact {
                println!("      chess: {}", label);
            }
            raw.clone()
        } else {
            None
        };

        // --- Chess comparison HW (only for Peano-primary tests that also have Chess builds) ---
        // Chess-only primary tests already ran above via primary_hw.
        // Respect both --no-hw flag and cascade detection.
        let chess_hw: Option<HwRunResult> = if !hw_available || hw_disabled_reason.is_some() {
            // hw_available is false when --no-hw is passed; cascade means device is wedged
            None
        } else if chess_available && !skip_hw && compiler == Compiler::Peano {
            if let Some(chess_artifacts) = chess_builds.get(&test.name) {
                if let Some(ref chess_insts) = chess_artifacts.insts {
                    if let Some(s) = spec {
                        if opts.chess_hardware || compact {
                            let prefix = if compact { "chess" } else { "chess-hw" };
                            if compact { print!("  "); }
                            let hw = hw_executor::run_hw_and_print(
                                s, &test.name, &chess_artifacts.xclbin, chess_insts, prefix, compact, &reference_dir,
                            );
                            stats.record_hw(Compiler::Chess, &hw);

                            // Immediate wedge cutoff for Chess HW path.
                            if hw.wedged && hw_disabled_reason.is_none() {
                                let detail = format!(
                                    "NPU device wedged (D-state) at test {} (Chess HW) -- \
                                     process survived SIGKILL, reboot required",
                                    idx + 1,
                                );
                                eprintln!("\nCRITICAL: {}", detail);
                                eprintln!("Disabling hardware execution for remaining tests.\n");
                                stats.hw_cascade_stopped_at = Some(idx + 1);
                                hw_disabled_reason = Some(detail);
                            }

                            // Apply same cascade tracking for Chess HW errors.
                            if hw.passed || hw.label.starts_with("FAIL") {
                                consecutive_hw_errors = 0;
                            } else if hw_disabled_reason.is_none() {
                                consecutive_hw_errors += 1;
                                if consecutive_hw_errors >= MAX_CONSECUTIVE_HW_ERRORS {
                                    let detail = if !npu_runner::probe_device_health() {
                                        format!(
                                            "NPU device unreachable after {} consecutive errors (test {}, Chess HW)",
                                            consecutive_hw_errors, idx + 1,
                                        )
                                    } else {
                                        format!(
                                            "NPU device wedged after {} consecutive errors at test {} (Chess HW)",
                                            consecutive_hw_errors, idx + 1,
                                        )
                                    };
                                    eprintln!("\nWARNING: {}", detail);
                                    eprintln!("Disabling hardware execution for remaining tests.\n");
                                    stats.hw_cascade_stopped_at = Some(idx + 1);
                                    hw_disabled_reason = Some(detail);
                                }
                            }

                            Some(hw)
                        } else { None }
                    } else { None }
                } else { None }
            } else { None }
        } else { None };

        // --- Differential (when both compilers ran on HW) ---
        if let (Some(ref p), Some(ref c)) = (&primary_hw, &chess_hw) {
            if compiler == Compiler::Peano {
                stats.record_differential(p.passed, c.passed);
            }
        }

        // --- Compiler comparison ---
        if chess_available {
            if chess_builds.contains_key(&test.name) {
                if let Some(s) = spec {
                    let diag = (|| -> Option<CompilerDiagnosis> {
                        let output_buf = s.buffers.iter().find(|b| b.direction == BufferDir::Output)?;
                        let elem_type = output_buf.element_type;

                        let expected_bytes = load_hw_reference(&reference_dir, &test.name)?;
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
                    let sim_label = runner_display::run_aiesim_comparison(tools, test, prj, &reference_dir);
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
            let filtered: Vec<_> = unit_tests.iter()
                .filter(|t| runner_config::matches_filter(&t.name, &opts.filters))
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
                        use xdna_emu::integration::aiesimulator;

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
fn run_sequential(suite: &XclbinSuite, tests: &[XclbinTest]) -> Vec<TestResult> {
    let total = tests.len();
    let mut results = Vec::with_capacity(total);

    for (i, test) in tests.iter().enumerate() {
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
