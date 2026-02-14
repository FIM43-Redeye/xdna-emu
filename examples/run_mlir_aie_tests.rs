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
//!
//! Positional arguments are substring filters on test name. Multiple filters
//! are OR-ed (a test runs if it matches ANY filter).
//!
//! Examples:
//!   cargo run --example run_mlir_aie_tests                    # run all tests
//!   cargo run --example run_mlir_aie_tests -- add_blockwrite  # run one test
//!   cargo run --example run_mlir_aie_tests -- -v vec_vec      # verbose for matching
//!   cargo run --example run_mlir_aie_tests -- -v              # verbose for all
//!   cargo run --example run_mlir_aie_tests -- -j 8            # run 8 tests at once

use std::path::PathBuf;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use xdna_emu::testing::xclbin_suite::{XclbinSuite, TestOutcome};
use xdna_emu::testing::manifest_runner::{TestManifest, ElementType, read_values};

/// Parsed CLI options.
struct Options {
    verbose: bool,
    jobs: usize,
    filters: Vec<String>,
}

/// Parse CLI arguments.
fn parse_args() -> Options {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut verbose = false;
    let mut jobs: usize = 1;
    let mut filters = Vec::new();
    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--verbose" | "-v" => verbose = true,
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
                std::process::exit(1);
            }
        }
    }

    Options { verbose, jobs, filters }
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
    }

    out
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error"),
    )
    .init();

    let opts = parse_args();

    let config = xdna_emu::config::Config::get();
    let _mlir_aie_path = PathBuf::from(config.mlir_aie_path());
    let npu_xrt_path = config.npu_xrt_test_dir();
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/mlir-aie-extracted/manifests");

    if !npu_xrt_path.exists() {
        eprintln!("Error: mlir-aie npu-xrt tests not found at {}", npu_xrt_path.display());
        eprintln!("Make sure mlir-aie is built with tests enabled.");
        std::process::exit(1);
    }

    println!("Discovering tests in {}...", npu_xrt_path.display());

    let mut suite = match XclbinSuite::discover(&npu_xrt_path) {
        Ok(s) => {
            let s = s.with_max_cycles(1_000_000); // 1M cycle timeout
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
    let mut failed = 0;
    let mut unknown_count = 0;
    let mut timeout_count = 0;
    let mut load_error = 0;

    for r in &results {
        println!("{}", format_result(r, total));

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
    }

    println!("\n{:=<60}", "");
    println!("=== SUMMARY ===");
    let effective = total - skipped;
    println!("Total:            {}", total);
    println!("Skipped:          {}", skipped);
    println!("Passed:           {} ({:.1}%)", passed, 100.0 * passed as f64 / effective.max(1) as f64);
    println!("Expected Fail:    {}", expected_fail);
    println!("Unexpected Pass:  {}", unexpected_pass);
    println!("Validation Fail:  {}", validation_failed);
    println!("Failed:           {}", failed);
    println!("Unknown:          {}", unknown_count);
    println!("Timeout:          {}", timeout_count);
    println!("Load Error:       {}", load_error);

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
        let (outcome, raw_output) = suite.run_single_with_output(test);

        results.push(TestResult {
            idx: i,
            name: test.name.clone(),
            elf_count,
            embedded_count,
            has_npu,
            outcome,
            raw_output,
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
                    let (outcome, raw_output) = suite.run_single_with_output(test);

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
