//! Fuzz iteration loop: generate, compile, run, compare.
//!
//! Orchestrates the full fuzzing pipeline in two phases:
//! 1. **Compile phase** (parallel): generate C++ kernels and compile to xclbin
//!    using multiple threads.
//! 2. **Execute phase** (sequential): run each compiled case through the
//!    emulator and optionally on real NPU hardware, then compare outputs.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::fuzzer::gen;
use crate::fuzzer::lower_cpp;
use crate::fuzzer::params::{FuzzParams, ScalarType};
use crate::testing::runner_config::Options;
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};

/// Paths to external tools needed for compilation.
struct ToolPaths {
    /// Peano clang compiler.
    peano_clang: PathBuf,
    /// Python interpreter from ironenv.
    python: PathBuf,
    /// Path to aiecc.py.
    aiecc: PathBuf,
    /// PYTHONPATH for mlir-aie modules.
    pythonpath: String,
    /// Path to fuzz_template.py.
    template_script: PathBuf,
    /// Peano install dir (for PEANO_INSTALL_DIR env).
    peano_dir: PathBuf,
    /// mlir-aie bin dir (for PATH).
    mlir_aie_bin: PathBuf,
    /// aietools root (optional, for XILINX_VITIS_AIETOOLS).
    aietools_root: Option<PathBuf>,
}

impl ToolPaths {
    /// Discover tool paths from the environment.
    fn discover() -> Result<Self, String> {
        let config = crate::config::Config::get();
        let env = crate::integration::chess_build::BuildEnv::discover(config)?;

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let template_script = manifest_dir.join("tools/fuzz_template.py");
        if !template_script.exists() {
            return Err(format!("fuzz_template.py not found at {}", template_script.display()));
        }

        Ok(Self {
            peano_clang: env.peano_clang(),
            python: env.python().to_path_buf(),
            aiecc: env.aiecc().to_path_buf(),
            pythonpath: env.pythonpath().to_string(),
            template_script,
            peano_dir: env.peano_dir().to_path_buf(),
            mlir_aie_bin: env.mlir_aie_bin().to_path_buf(),
            aietools_root: env.aietools_root().map(Path::to_path_buf),
        })
    }

    /// Apply build environment variables to a Command.
    fn apply_env(&self, cmd: &mut Command) {
        cmd.env("PEANO_INSTALL_DIR", &self.peano_dir);

        // Build PATH with mlir-aie bin + peano bin
        let mut path = std::env::var("PATH").unwrap_or_default();
        path = format!("{}:{}:{}", self.mlir_aie_bin.display(),
            self.peano_dir.join("bin").display(), path);
        if let Some(ref aietools) = self.aietools_root {
            path = format!("{}:{}", aietools.join("bin").display(), path);
            cmd.env("XILINX_VITIS_AIETOOLS", aietools);
        }
        cmd.env("PATH", &path);
        cmd.env("PYTHONPATH", &self.pythonpath);

        // MLIR_AIE_DIR for include paths
        if let Some(mlir_aie_dir) = self.mlir_aie_bin.parent() {
            cmd.env("MLIR_AIE_DIR", mlir_aie_dir);
        }
    }
}

/// Trace buffer size in elements (i32). 1MB = 262144 x i32, matching the
/// standard trace buffer size used by npu-test and the NPU executor.
const TRACE_BUFFER_ELEMENTS: usize = 262_144;

fn dtype_str(dtype: ScalarType) -> &'static str {
    match dtype {
        ScalarType::I32 => "i32",
        ScalarType::I16 => "i16",
        ScalarType::I8 => "i8",
    }
}

/// A compiled fuzz case ready for execution.
struct CompiledCase {
    seed: u64,
    params: FuzzParams,
    case_dir: PathBuf,
    /// Per-group patched insts.bin paths (populated when --trace-sweep is active).
    group_insts_paths: Vec<PathBuf>,
}

/// Run the fuzz loop: batch-compile, then run on emulator + NPU and compare.
pub fn run_fuzz(opts: &Options) {
    let iterations = opts.fuzz_iterations;
    let base_seed = opts.fuzz_seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    println!("Fuzzing {} iterations, base seed {}", iterations, base_seed);

    if iterations == 0 {
        println!("Fuzz complete: 0 pass, 0 fail, 0 error");
        return;
    }

    // Discover tools once for all iterations.
    let tools = match ToolPaths::discover() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: failed to discover build tools: {}", e);
            std::process::exit(1);
        }
    };
    println!("  peano: {}", tools.peano_clang.display());
    println!("  aiecc: {}", tools.aiecc.display());
    println!("  template: {}", tools.template_script.display());

    let fuzz_dir = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("build/fuzz");
    std::fs::create_dir_all(&fuzz_dir).ok();

    let jobs = opts.jobs;
    let hw = opts.hw;
    let trace_sweep = opts.trace_sweep;

    // Phase 1: Generate and compile all cases (parallel).
    let compile_start = Instant::now();
    let (compiled, compile_errors) = compile_all(
        &tools, base_seed, iterations, &fuzz_dir, jobs, opts.verbose, trace_sweep,
    );
    let compile_elapsed = compile_start.elapsed().as_secs_f64();
    println!(
        "Compile: {} ok, {} error ({:.1}s, {} threads)",
        compiled.len(), compile_errors, compile_elapsed, jobs,
    );

    if compiled.is_empty() {
        println!("Fuzz complete: 0 pass, 0 fail, {} error", compile_errors);
        return;
    }

    // Phase 2: Execute on emulator (and optionally NPU), compare outputs.
    let exec_start = Instant::now();
    let (pass, fail, exec_errors) = execute_all(&compiled, opts.max_cycles, hw, opts.verbose, jobs);
    let exec_elapsed = exec_start.elapsed().as_secs_f64();

    let total_errors = compile_errors + exec_errors;
    println!(
        "Execute: {} pass, {} fail, {} error ({:.1}s)",
        pass, fail, exec_errors, exec_elapsed,
    );
    println!(
        "Fuzz complete: {} pass, {} fail, {} error",
        pass, fail, total_errors,
    );

    // Phase 3 (optional): Trace event group sweep.
    if trace_sweep {
        let sweep_start = Instant::now();
        execute_trace_sweep(&compiled, opts);
        let sweep_elapsed = sweep_start.elapsed().as_secs_f64();
        println!("Trace sweep: {:.1}s", sweep_elapsed);
    }

    if fail > 0 {
        std::process::exit(1);
    }
}

/// Phase 1: Generate C++ and compile to xclbin in parallel.
///
/// Returns the list of successfully compiled cases and the error count.
fn compile_all(
    tools: &ToolPaths,
    base_seed: u64,
    iterations: usize,
    fuzz_dir: &Path,
    jobs: usize,
    verbose: bool,
    trace_sweep: bool,
) -> (Vec<CompiledCase>, usize) {
    // Generate all case metadata up front (cheap, deterministic).
    let cases: Vec<(u64, FuzzParams, PathBuf)> = (0..iterations)
        .map(|i| {
            let seed = base_seed.wrapping_add(i as u64);
            let params = gen::generate(seed);
            let case_dir = fuzz_dir.join(format!("seed_{}", seed));
            (seed, params, case_dir)
        })
        .collect();

    // Write all C++ source files (fast, single-threaded).
    for (seed, params, case_dir) in &cases {
        std::fs::create_dir_all(case_dir).ok();
        let cpp = lower_cpp::lower_to_cpp(params);
        if let Err(e) = std::fs::write(case_dir.join("fuzz_kernel.cc"), &cpp) {
            eprintln!("seed {} write error: {}", seed, e);
        }
    }

    if verbose {
        println!("Generated {} kernels, compiling with {} threads...", cases.len(), jobs);
    }

    // Parallel compilation.
    // ToolPaths is not Clone, so wrap in Arc for sharing across threads.
    // We use scoped threads so the borrow is safe.
    let total = cases.len();
    let compiled = std::sync::Mutex::new(Vec::new());
    let errors = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);

    // Work queue: simple shared index.
    let next_idx = std::sync::atomic::AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..jobs {
            let cases = &cases;
            let tools = &tools;
            let compiled = &compiled;
            let errors = &errors;
            let done = &done;
            let next_idx = &next_idx;

            s.spawn(move || {
                loop {
                    let idx = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= cases.len() {
                        break;
                    }

                    let (seed, params, case_dir) = &cases[idx];

                    match compile_fuzz_case(tools, params, case_dir) {
                        Ok(_) => {
                            // Generate per-group patched insts files if sweep is active.
                            let group_insts_paths = if trace_sweep {
                                match generate_group_insts(case_dir, *seed, verbose) {
                                    Ok(paths) => paths,
                                    Err(e) => {
                                        errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        if verbose {
                                            eprintln!("seed {} group insts error: {}", seed, e);
                                        }
                                        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                                        if !verbose {
                                            eprint!("\rCompiling [{}/{}]", n, total);
                                        }
                                        continue;
                                    }
                                }
                            } else {
                                Vec::new()
                            };
                            compiled.lock().unwrap().push(CompiledCase {
                                seed: *seed,
                                params: params.clone(),
                                case_dir: case_dir.clone(),
                                group_insts_paths,
                            });
                        }
                        Err(e) => {
                            errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if verbose {
                                eprintln!("seed {} compile error: {}", seed, e);
                            }
                        }
                    }

                    let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if !verbose {
                        // Progress on stderr to avoid mixing with output.
                        eprint!("\rCompiling [{}/{}]", n, total);
                    }
                }
            });
        }
    });

    if !verbose {
        eprintln!(); // newline after progress
    }

    let error_count = errors.into_inner();
    let mut result = compiled.into_inner().unwrap();
    // Sort by seed for deterministic execution order.
    result.sort_by_key(|c| c.seed);
    (result, error_count)
}

/// Phase 2: Execute compiled cases on emulator (and optionally NPU), compare.
///
/// When `hw` is true, emulator and NPU run concurrently:
/// - NPU thread processes cases sequentially (fast, <1s each).
/// - Emulator workers fill result slots in parallel (~13s each), so
///   parallelism across `jobs` threads is critical for throughput.
/// The NPU thread grabs emulator results as they become available.
///
/// Returns (pass, fail, error) counts.
fn execute_all(
    cases: &[CompiledCase],
    max_cycles: u64,
    hw: bool,
    verbose: bool,
    jobs: usize,
) -> (usize, usize, usize) {
    let total = cases.len();

    if !hw {
        return execute_emulator_only(cases, max_cycles, verbose, jobs);
    }

    // HW comparison: run emulator (parallel) and NPU (sequential) independently,
    // then compare results. This decouples them so neither blocks the other.
    type CaseResult = Result<(Vec<u8>, Option<Vec<u8>>), String>;

    // Emulator results: parallel work-stealing.
    let emu_results: Vec<std::sync::Mutex<Option<CaseResult>>> =
        (0..total).map(|_| std::sync::Mutex::new(None)).collect();
    let emu_next = std::sync::atomic::AtomicUsize::new(0);
    let emu_done = std::sync::atomic::AtomicUsize::new(0);

    // NPU results: sequential (single device).
    let npu_results: Vec<std::sync::Mutex<Option<CaseResult>>> =
        (0..total).map(|_| std::sync::Mutex::new(None)).collect();
    let npu_done = std::sync::atomic::AtomicUsize::new(0);

    std::thread::scope(|s| {
        let emu_results = &emu_results;
        let npu_results = &npu_results;
        let emu_done = &emu_done;
        let npu_done = &npu_done;

        // Emulator workers: parallel, work-stealing.
        for _ in 0..jobs {
            let emu_next = &emu_next;
            s.spawn(move || {
                loop {
                    let idx = emu_next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= total { break; }
                    let case = &cases[idx];
                    let xclbin_path = case.case_dir.join("aie.xclbin");
                    let result = run_emulator(&xclbin_path, &case.params, max_cycles);
                    *emu_results[idx].lock().unwrap() = Some(result);
                    let n = emu_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if verbose {
                        eprintln!("  emu {}/{} (seed {})", n, total, case.seed);
                    }
                }
            });
        }

        // NPU thread: sequential (single device).
        s.spawn(move || {
            for (i, case) in cases.iter().enumerate() {
                let xclbin_path = case.case_dir.join("aie.xclbin");
                let insts_path = case.case_dir.join("insts.bin");
                let result = run_on_npu_raw(case, &xclbin_path, &insts_path);
                *npu_results[i].lock().unwrap() = Some(result);
                let n = npu_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if !verbose {
                    eprint!("\r  emu {}/{}, npu {}/{}",
                        emu_done.load(std::sync::atomic::Ordering::Relaxed), total,
                        n, total);
                }
            }
        });
    });

    if !verbose {
        eprintln!();
    }

    // Compare results.
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut error = 0usize;
    let mut vacuous = 0usize;

    for (i, case) in cases.iter().enumerate() {
        let emu_result = emu_results[i].lock().unwrap().take().unwrap();
        let npu_result = npu_results[i].lock().unwrap().take().unwrap();

        match (emu_result, npu_result) {
            (Ok((emu_output, emu_trace)), Ok((npu_output, npu_trace))) => {
                if emu_output.as_slice() == npu_output.as_slice() {
                    let all_zero = emu_output.iter().all(|&b| b == 0);
                    if all_zero {
                        vacuous += 1;
                        if verbose {
                            println!("seed {} MATCH (vacuous -- both zero, {} elements)",
                                case.seed,
                                emu_output.len() / case.params.dtype.byte_size());
                        }
                    } else {
                        pass += 1;
                        if verbose {
                            println!("seed {} MATCH ({} elements)",
                                case.seed,
                                emu_output.len() / case.params.dtype.byte_size());
                        }
                    }
                } else {
                    fail += 1;
                    let elem_type = match case.params.dtype {
                        ScalarType::I32 => crate::testing::test_cpp_parser::ElementType::I32,
                        ScalarType::I16 => crate::testing::test_cpp_parser::ElementType::I16,
                        ScalarType::I8 => crate::testing::test_cpp_parser::ElementType::I8,
                    };
                    let detail = format_mismatch(&emu_output, &npu_output, elem_type);
                    println!("seed {} MISMATCH: {}", case.seed, detail);
                    // Save outputs and trace data for post-mortem debugging.
                    let _ = std::fs::write(case.case_dir.join("emu_output.bin"), &emu_output);
                    let _ = std::fs::write(case.case_dir.join("npu_output.bin"), &npu_output);
                    if let Some(ref t) = emu_trace {
                        let _ = std::fs::write(case.case_dir.join("emu_trace.bin"), t);
                    }
                    if let Some(ref t) = npu_trace {
                        let _ = std::fs::write(case.case_dir.join("npu_trace.bin"), t);
                    }
                }
            }
            (Err(e), _) => {
                error += 1;
                if verbose {
                    println!("seed {} emulator error: {}", case.seed, e);
                }
            }
            (_, Err(e)) => {
                error += 1;
                if verbose {
                    println!("seed {} hw error: {}", case.seed, e);
                }
            }
        }
    }

    if vacuous > 0 {
        println!("  ({} vacuous matches -- both sides all zeros)", vacuous);
    }

    (pass, fail, error)
}

/// Emulator-only execution (no hardware comparison), parallel work-stealing.
fn execute_emulator_only(
    cases: &[CompiledCase],
    max_cycles: u64,
    verbose: bool,
    jobs: usize,
) -> (usize, usize, usize) {
    let total = cases.len();
    let pass = std::sync::atomic::AtomicUsize::new(0);
    let error = std::sync::atomic::AtomicUsize::new(0);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..jobs {
            let next = &next;
            let done = &done;
            let pass = &pass;
            let error = &error;
            s.spawn(move || {
                loop {
                    let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= total { break; }
                    let case = &cases[idx];
                    let xclbin_path = case.case_dir.join("aie.xclbin");
                    match run_emulator(&xclbin_path, &case.params, max_cycles) {
                        Ok((output, trace)) if !output.is_empty() => {
                            pass.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if verbose {
                                let trace_msg = trace.as_ref()
                                    .map(|t| format!(", trace {} bytes", t.len()))
                                    .unwrap_or_default();
                                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                                println!("[{}/{}] seed {} emulator: {} bytes{}",
                                    n, total, case.seed, output.len(), trace_msg);
                                if let Some(ref trace_data) = trace {
                                    let _ = std::fs::write(case.case_dir.join("emu_trace.bin"), trace_data);
                                }
                            } else {
                                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                                let p = pass.load(std::sync::atomic::Ordering::Relaxed);
                                let e = error.load(std::sync::atomic::Ordering::Relaxed);
                                eprint!("\r[{}/{}] {} pass, {} error", n, total, p, e);
                            }
                        }
                        Ok(_) => {
                            error.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            if verbose {
                                eprintln!("[{}/{}] seed {} emulator produced empty output", n, total, case.seed);
                            }
                        }
                        Err(e) => {
                            error.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            if verbose {
                                eprintln!("[{}/{}] seed {} emulator error: {}", n, total, case.seed, e);
                            }
                        }
                    }
                }
            });
        }
    });

    if !verbose {
        eprintln!();
    }
    // No fail category in emulator-only mode (only pass/error).
    (pass.into_inner(), 0, error.into_inner())
}

/// Phase 3: Trace event group sweep.
///
/// For each compiled case:
/// 1. Run NPU `trace_sweep_reps` times with group 0 for determinism check.
/// 2. Run NPU once per event group, collecting traces.
/// 3. Run emulator once per event group, collecting traces.
/// 4. Decode and compare NPU vs emulator event sequences per group.
/// 5. Write per-seed results to `trace_sweep/` subdirectory.
fn execute_trace_sweep(cases: &[CompiledCase], opts: &Options) {
    use crate::fuzzer::trace_sweep::{
        TRACE_EVENT_GROUPS, NUM_GROUPS, check_determinism, compare_event_sequences,
        decode_binary_trace, write_sweep_summary,
    };

    let reps = opts.trace_sweep_reps;
    let hw = opts.hw;
    let verbose = opts.verbose;
    let max_cycles = opts.max_cycles;

    println!("Trace sweep: {} seeds, {} groups, {} reps", cases.len(), NUM_GROUPS, reps);

    if !hw {
        println!("  (no hardware -- emulator-only trace capture, no comparison)");
    }

    let mut total_deterministic = 0usize;
    let mut total_nondeterministic = 0usize;
    let mut group_match_counts = vec![0usize; NUM_GROUPS];
    let mut group_mismatch_counts = vec![0usize; NUM_GROUPS];
    let mut nondeterministic_seeds = Vec::new();

    for case in cases {
        if case.group_insts_paths.len() != NUM_GROUPS {
            if verbose {
                eprintln!(
                    "seed {} skipped: expected {} group insts, found {}",
                    case.seed,
                    NUM_GROUPS,
                    case.group_insts_paths.len(),
                );
            }
            continue;
        }

        let sweep_dir = case.case_dir.join("trace_sweep");
        std::fs::create_dir_all(&sweep_dir).ok();
        let xclbin_path = case.case_dir.join("aie.xclbin");

        // Step 1: NPU determinism check (group 0 only).
        let determinism = if hw {
            let mut det_traces = Vec::with_capacity(reps);
            for rep in 0..reps {
                match run_on_npu_raw(case, &xclbin_path, &case.group_insts_paths[0]) {
                    Ok((_output, Some(trace))) => {
                        let trace_path = sweep_dir.join(format!("npu_det_rep{}.bin", rep));
                        let _ = std::fs::write(&trace_path, &trace);
                        det_traces.push(trace);
                    }
                    Ok((_output, None)) => {
                        det_traces.push(Vec::new());
                    }
                    Err(e) => {
                        if verbose {
                            eprintln!("seed {} det rep {} NPU error: {}", case.seed, rep, e);
                        }
                        break;
                    }
                }
            }

            let result = check_determinism(&det_traces);
            let det_text = match &result {
                Ok(()) => format!("PASS ({} reps, {} bytes each)",
                    det_traces.len(),
                    det_traces.first().map(|t| t.len()).unwrap_or(0)),
                Err(e) => format!("FAIL ({})", e),
            };
            let _ = std::fs::write(sweep_dir.join("determinism.txt"), &det_text);

            if result.is_ok() {
                total_deterministic += 1;
            } else {
                total_nondeterministic += 1;
                nondeterministic_seeds.push(case.seed);
            }

            if verbose {
                println!("seed {} determinism: {}", case.seed, det_text);
            }

            result
        } else {
            Ok(()) // No hardware => skip determinism check.
        };

        // Step 2+3: Per-group NPU and emulator traces.
        let mut comparisons = Vec::with_capacity(NUM_GROUPS);

        for (group_idx, group) in TRACE_EVENT_GROUPS.iter().enumerate() {
            // NPU trace for this group.
            let npu_trace = if hw {
                // For group 0, reuse the first determinism rep trace if available.
                let npu_trace_path = sweep_dir.join(format!("group_{}_npu.bin", group_idx));
                if group_idx == 0 {
                    // Copy from det_rep0 if it exists.
                    let rep0_path = sweep_dir.join("npu_det_rep0.bin");
                    if rep0_path.exists() {
                        let _ = std::fs::copy(&rep0_path, &npu_trace_path);
                    }
                    std::fs::read(&npu_trace_path).ok()
                } else {
                    match run_on_npu_raw(case, &xclbin_path, &case.group_insts_paths[group_idx]) {
                        Ok((_output, Some(trace))) => {
                            let _ = std::fs::write(&npu_trace_path, &trace);
                            Some(trace)
                        }
                        Ok((_output, None)) => {
                            let _ = std::fs::write(&npu_trace_path, &[]);
                            Some(Vec::new())
                        }
                        Err(e) => {
                            if verbose {
                                eprintln!(
                                    "seed {} group {} NPU error: {}",
                                    case.seed, group_idx, e
                                );
                            }
                            None
                        }
                    }
                }
            } else {
                None
            };

            // Emulator trace for this group.
            let emu_trace_path = sweep_dir.join(format!("group_{}_emu.bin", group_idx));
            let emu_trace = {
                let test = XclbinTest::from_path(&xclbin_path)
                    .with_buffer_spec(make_fuzz_buffer_spec(&case.params));
                // Override insts path to use the group-specific patched file.
                let test = XclbinTest {
                    insts_path: Some(case.group_insts_paths[group_idx].clone()),
                    ..test
                };
                let suite = XclbinSuite::new().with_max_cycles(max_cycles);
                let (_outcome, _raw_output, binary_trace) = suite.run_single_with_trace(&test);
                if let Some(ref trace) = binary_trace {
                    let _ = std::fs::write(&emu_trace_path, trace);
                }
                binary_trace
            };

            // Step 4: Compare.
            let comp = if let (Some(npu_data), Some(emu_data)) = (&npu_trace, &emu_trace) {
                let npu_events = decode_binary_trace(npu_data);
                let emu_events = decode_binary_trace(emu_data);
                let comp = compare_event_sequences(&npu_events, &emu_events, group.name);

                // Write comparison result.
                let comp_path = sweep_dir.join(format!("group_{}_compare.txt", group_idx));
                let _ = std::fs::write(&comp_path, format!("{}", comp));

                if comp.sequence_match {
                    group_match_counts[group_idx] += 1;
                } else {
                    group_mismatch_counts[group_idx] += 1;
                }

                if verbose {
                    println!("  seed {} {}", case.seed, comp);
                }
                Some(comp)
            } else {
                if verbose && emu_trace.is_some() {
                    println!("  seed {} group {} emulator-only ({} events)",
                        case.seed, group_idx,
                        emu_trace.as_ref()
                            .map(|t| decode_binary_trace(t).len())
                            .unwrap_or(0),
                    );
                }
                None
            };

            if let Some(c) = comp {
                comparisons.push(c);
            }
        }

        // Write sweep summary for this seed.
        let _ = write_sweep_summary(&sweep_dir, &determinism, &comparisons);
    }

    // Final report.
    println!("Trace sweep: {} seeds", cases.len());
    if hw {
        println!(
            "  Determinism: {}/{} deterministic",
            total_deterministic,
            total_deterministic + total_nondeterministic,
        );
        for (idx, group) in TRACE_EVENT_GROUPS.iter().enumerate() {
            let matches = group_match_counts[idx];
            let mismatches = group_mismatch_counts[idx];
            if matches + mismatches > 0 {
                println!(
                    "  {}: {}/{} match, {}/{} mismatch",
                    group.name, matches, matches + mismatches,
                    mismatches, matches + mismatches,
                );
            }
        }
        if !nondeterministic_seeds.is_empty() {
            println!(
                "  Non-deterministic seeds: {:?}",
                nondeterministic_seeds,
            );
        }
    }
    println!("  Trace data: build/fuzz/seed_*/trace_sweep/");
}

/// Build the standard fuzz buffer spec (shared between emulator and NPU paths).
fn make_fuzz_buffer_spec(params: &FuzzParams) -> crate::testing::test_cpp_parser::BufferSpec {
    use crate::testing::test_cpp_parser::{
        BufferDef, BufferDir, BufferSpec, ElementType, InputPattern,
    };

    let elem_type = match params.dtype {
        ScalarType::I32 => ElementType::I32,
        ScalarType::I16 => ElementType::I16,
        ScalarType::I8 => ElementType::I8,
    };

    BufferSpec {
        buffers: vec![
            BufferDef {
                name: "buf_in".to_string(),
                group_id: 3,
                size_elements: params.buffer_size,
                element_type: elem_type,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Sequential { start: 1, step: 1 },
            },
            BufferDef {
                name: "buf_scratch".to_string(),
                group_id: 4,
                size_elements: params.buffer_size,
                element_type: elem_type,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_out".to_string(),
                group_id: 5,
                size_elements: params.buffer_size,
                element_type: elem_type,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_trace".to_string(),
                group_id: 6,
                size_elements: TRACE_BUFFER_ELEMENTS,
                element_type: ElementType::I32,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
        ],
        multi_kernel: false,
    }
}

/// Run a fuzz case on real NPU hardware and return (output, trace) bytes.
fn run_on_npu_raw(
    case: &CompiledCase,
    xclbin_path: &Path,
    insts_path: &Path,
) -> Result<(Vec<u8>, Option<Vec<u8>>), String> {
    use crate::testing::npu_runner;

    if !npu_runner::npu_available() {
        return Err("NPU hardware not available".into());
    }

    let spec = make_fuzz_buffer_spec(&case.params);
    let test_name = format!("fuzz_seed_{}", case.seed);
    match npu_runner::run_on_npu(&spec, &test_name, xclbin_path, insts_path, 30) {
        Ok(result) => {
            let trace = result.extra_outputs.get("buf_trace").cloned();
            Ok((result.output, trace))
        }
        Err(e) => Err(format!("{:?}", e)),
    }
}

/// Format a human-readable mismatch summary.
fn format_mismatch(
    emu: &[u8],
    npu: &[u8],
    elem_type: crate::testing::test_cpp_parser::ElementType,
) -> String {
    let elem_size = elem_type.byte_size();
    let emu_elems = emu.len() / elem_size;
    let npu_elems = npu.len() / elem_size;

    if emu.len() != npu.len() {
        return format!(
            "size mismatch: emulator={} bytes ({} elems), npu={} bytes ({} elems)",
            emu.len(), emu_elems, npu.len(), npu_elems,
        );
    }

    // Find first differing element.
    let min_elems = emu_elems.min(npu_elems);
    for i in 0..min_elems {
        let offset = i * elem_size;
        let emu_slice = &emu[offset..offset + elem_size];
        let npu_slice = &npu[offset..offset + elem_size];
        if emu_slice != npu_slice {
            let emu_val = read_elem(emu_slice, elem_size);
            let npu_val = read_elem(npu_slice, elem_size);
            return format!(
                "element [{}]: emulator={}, npu={} ({} total elements)",
                i, emu_val, npu_val, emu_elems,
            );
        }
    }
    "outputs differ but no element mismatch found (padding?)".to_string()
}

fn read_elem(bytes: &[u8], size: usize) -> i64 {
    match size {
        1 => bytes[0] as i8 as i64,
        2 => i16::from_le_bytes([bytes[0], bytes[1]]) as i64,
        4 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64,
        _ => 0,
    }
}

/// Compile a fuzz case: kernel.cc -> kernel.o, template -> aie.mlir -> xclbin.
fn compile_fuzz_case(
    tools: &ToolPaths,
    params: &FuzzParams,
    case_dir: &Path,
) -> Result<(), String> {
    // Skip if already compiled (xclbin exists and is newer than source).
    let xclbin = case_dir.join("aie.xclbin");
    let kernel_cc = case_dir.join("fuzz_kernel.cc");
    if xclbin.exists() {
        if let (Ok(src_meta), Ok(xclbin_meta)) =
            (std::fs::metadata(&kernel_cc), std::fs::metadata(&xclbin))
        {
            if let (Ok(src_time), Ok(xclbin_time)) =
                (src_meta.modified(), xclbin_meta.modified())
            {
                if xclbin_time > src_time {
                    return Ok(());
                }
            }
        }
    }

    // Step 1: Compile kernel to .o with Peano clang
    let kernel_obj = case_dir.join("fuzz_kernel.cc.o");

    let mut compile_cmd = Command::new(&tools.peano_clang);
    compile_cmd
        .arg("--target=aie2-none-unknown-elf")
        .arg("-O2")
        .arg("-c")
        .arg(&kernel_cc)
        .arg("-o")
        .arg(&kernel_obj);
    tools.apply_env(&mut compile_cmd);

    let compile_out = compile_cmd.output()
        .map_err(|e| format!("Failed to spawn Peano clang: {}", e))?;
    if !compile_out.status.success() {
        let stderr = String::from_utf8_lossy(&compile_out.stderr);
        return Err(format!("Kernel compilation failed:\n{}",
            stderr.lines().take(10).collect::<Vec<_>>().join("\n")));
    }

    // Step 2: Generate MLIR template (always with trace instrumentation)
    let mut template_cmd = Command::new(&tools.python);
    template_cmd
        .arg(&tools.template_script)
        .arg("--kernel").arg("fuzz_kernel.cc")
        .arg("--size").arg(params.buffer_size.to_string())
        .arg("--dtype").arg(dtype_str(params.dtype))
        .arg("--outdir").arg(case_dir)
        .arg("--device").arg("npu1_1col")
        .arg("--trace");
    tools.apply_env(&mut template_cmd);

    let template_out = template_cmd.output()
        .map_err(|e| format!("Failed to spawn fuzz_template.py: {}", e))?;
    if !template_out.status.success() {
        let stderr = String::from_utf8_lossy(&template_out.stderr);
        return Err(format!("MLIR template generation failed:\n{}",
            stderr.lines().take(10).collect::<Vec<_>>().join("\n")));
    }

    // Step 3: Compile MLIR to xclbin via aiecc.py
    // Run from case_dir so aiecc.py can find the kernel .o and write outputs there.
    let mut aiecc_cmd = Command::new(&tools.python);
    aiecc_cmd
        .arg(&tools.aiecc)
        .arg("--no-xchesscc")
        .arg("--no-xbridge")
        .arg("--no-aiesim")
        .arg("--aie-generate-xclbin")
        .arg("--aie-generate-npu-insts")
        .arg("--no-compile-host")
        .arg("--alloc-scheme=basic-sequential")
        .arg("--xclbin-name=aie.xclbin")
        .arg("--npu-insts-name=insts.bin")
        .arg("aie.mlir");
    aiecc_cmd.current_dir(case_dir);
    tools.apply_env(&mut aiecc_cmd);

    let aiecc_out = aiecc_cmd.output()
        .map_err(|e| format!("Failed to spawn aiecc.py: {}", e))?;
    if !aiecc_out.status.success() {
        let stderr = String::from_utf8_lossy(&aiecc_out.stderr);
        let stdout = String::from_utf8_lossy(&aiecc_out.stdout);
        let combined = if stderr.is_empty() { stdout } else { stderr };
        return Err(format!("aiecc.py failed:\n{}",
            combined.lines().take(10).collect::<Vec<_>>().join("\n")));
    }

    // Verify outputs exist
    if !xclbin.exists() {
        return Err("aiecc.py succeeded but aie.xclbin not found".into());
    }
    let insts = case_dir.join("insts.bin");
    if !insts.exists() {
        return Err("aiecc.py succeeded but insts.bin not found".into());
    }

    Ok(())
}

/// Generate per-group patched insts.bin files for trace sweep.
///
/// Reads the original `insts.bin`, patches the trace event register values
/// for each group, and writes `insts_group_N.bin` files. Returns the paths.
fn generate_group_insts(
    case_dir: &Path,
    seed: u64,
    verbose: bool,
) -> Result<Vec<PathBuf>, String> {
    use crate::fuzzer::trace_sweep::{TRACE_EVENT_GROUPS, NUM_GROUPS, patch_insts_for_group};

    let insts_path = case_dir.join("insts.bin");
    let insts_bytes = std::fs::read(&insts_path)
        .map_err(|e| format!("Failed to read insts.bin: {}", e))?;

    let mut paths = Vec::with_capacity(NUM_GROUPS);

    for (idx, group) in TRACE_EVENT_GROUPS.iter().enumerate() {
        let patched = patch_insts_for_group(&insts_bytes, group)?;
        let group_path = case_dir.join(format!("insts_group_{}.bin", idx));
        std::fs::write(&group_path, &patched)
            .map_err(|e| format!("Failed to write {}: {}", group_path.display(), e))?;

        // Sanity check: group 0 should be identical to the original.
        if idx == 0 && patched != insts_bytes {
            if verbose {
                eprintln!(
                    "seed {} warning: group 0 patched insts differs from original \
                     (fuzz_template.py event config may have changed)",
                    seed,
                );
            }
        }

        paths.push(group_path);
    }

    Ok(paths)
}

/// Run a compiled fuzz case through the emulator.
///
/// Returns (output_bytes, binary_trace_bytes).
fn run_emulator(
    xclbin_path: &Path,
    params: &FuzzParams,
    max_cycles: u64,
) -> Result<(Vec<u8>, Option<Vec<u8>>), String> {
    let spec = make_fuzz_buffer_spec(params);
    let test = XclbinTest::from_path(xclbin_path).with_buffer_spec(spec);
    let suite = XclbinSuite::new().with_max_cycles(max_cycles);
    let (_outcome, raw_output, binary_trace) = suite.run_single_with_trace(&test);
    let output = raw_output.ok_or_else(|| "Emulator produced no output".to_string())?;
    Ok((output, binary_trace))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_fuzz_zero_iterations() {
        // Zero iterations should complete without error.
        let opts = crate::testing::runner_config::Options {
            mode: crate::testing::runner_config::RunMode::Fuzz,
            verbose: false,
            jobs: 1,
            filters: vec![],
            chess_only: false,
            elfanalyze: false,
            chess_build: false,
            chess_emulator: false,
            chess_hardware: false,
            hw: false,
            aiesim: false,
            build_nice: 19,
            max_cycles: 100_000,
            unit_tests: false,
            unit_tests_aiesim: false,
            unit_tests_aiesim_timeout: 0,
            no_build: false,
            hw_only: false,
            examples: false,
            rebuild: false,
            timeout_secs: None,
            build_dir: None,
            max_failures: None,
            lit_args: vec![],
            watchdog_secs: 3600,
            trace_size: 0,
            aiesim_trace: false,
            list_only: false,
            output_path: std::path::PathBuf::from("/dev/null"),
            format_json: false,
            fuzz_iterations: 0,
            fuzz_seed: Some(1),
            trace_sweep: false,
            trace_sweep_reps: 5,
        };
        run_fuzz(&opts);
    }

    #[test]
    fn test_dtype_str() {
        assert_eq!(dtype_str(ScalarType::I32), "i32");
        assert_eq!(dtype_str(ScalarType::I16), "i16");
        assert_eq!(dtype_str(ScalarType::I8), "i8");
    }

    #[test]
    fn test_format_mismatch_size_diff() {
        use crate::testing::test_cpp_parser::ElementType;
        let emu = vec![0u8; 16];
        let npu = vec![0u8; 32];
        let msg = format_mismatch(&emu, &npu, ElementType::I32);
        assert!(msg.contains("size mismatch"));
    }

    #[test]
    fn test_format_mismatch_element_diff() {
        use crate::testing::test_cpp_parser::ElementType;
        let emu = vec![1, 0, 0, 0, 2, 0, 0, 0]; // [1, 2] as i32
        let npu = vec![1, 0, 0, 0, 9, 0, 0, 0]; // [1, 9] as i32
        let msg = format_mismatch(&emu, &npu, ElementType::I32);
        assert!(msg.contains("element [1]"));
        assert!(msg.contains("emulator=2"));
        assert!(msg.contains("npu=9"));
    }

    #[test]
    fn test_format_mismatch_identical() {
        use crate::testing::test_cpp_parser::ElementType;
        let data = vec![1, 0, 0, 0];
        let msg = format_mismatch(&data, &data, ElementType::I32);
        assert!(msg.contains("no element mismatch"));
    }
}
