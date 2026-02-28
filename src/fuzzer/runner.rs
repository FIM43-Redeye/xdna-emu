//! Fuzz iteration loop: generate, compile, run, compare.
//!
//! Orchestrates the full fuzzing pipeline in two phases:
//! 1. **Compile phase** (parallel): generate C++ kernels and compile to xclbin
//!    using multiple threads.
//! 2. **Execute phase** (sequential): run each compiled case through the
//!    emulator and optionally on real NPU hardware, then compare outputs.

use std::io::Write;
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

    // Phase 1: Generate and compile all cases (parallel).
    let compile_start = Instant::now();
    let (compiled, compile_errors) = compile_all(
        &tools, base_seed, iterations, &fuzz_dir, jobs, opts.verbose,
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
                            compiled.lock().unwrap().push(CompiledCase {
                                seed: *seed,
                                params: params.clone(),
                                case_dir: case_dir.clone(),
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
        return execute_emulator_only(cases, max_cycles, verbose);
    }

    // HW comparison: pipeline emulator and NPU concurrently.
    //
    // Emulator result slots -- one per case, initially empty.  Emulator
    // workers fill them in parallel; the NPU thread reads them sequentially.
    let emu_slots: Vec<std::sync::Mutex<Option<Result<Vec<u8>, String>>>> =
        (0..total).map(|_| std::sync::Mutex::new(None)).collect();

    let pass = std::sync::atomic::AtomicUsize::new(0);
    let fail = std::sync::atomic::AtomicUsize::new(0);
    let error = std::sync::atomic::AtomicUsize::new(0);

    let emu_next = std::sync::atomic::AtomicUsize::new(0);

    std::thread::scope(|s| {
        let emu_slots = &emu_slots;
        let pass = &pass;
        let fail = &fail;
        let error = &error;

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
                    *emu_slots[idx].lock().unwrap() = Some(result);
                }
            });
        }

        // NPU thread: sequential (single device), compare with emulator.
        s.spawn(move || {
            for (i, case) in cases.iter().enumerate() {
                let xclbin_path = case.case_dir.join("aie.xclbin");
                let insts_path = case.case_dir.join("insts.bin");

                // Run on NPU (slow).
                let npu_result = run_on_npu_raw(case, &xclbin_path, &insts_path);

                // Wait for emulator result (usually already done).
                let emu_result = loop {
                    if let Some(r) = emu_slots[i].lock().unwrap().take() {
                        break r;
                    }
                    std::thread::yield_now();
                };

                // Compare.
                match (emu_result, npu_result) {
                    (Ok(emu_output), Ok(npu_output)) => {
                        if emu_output.as_slice() == npu_output.as_slice() {
                            pass.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if verbose {
                                println!("[{}/{}] seed {} MATCH", i + 1, total, case.seed);
                            }
                        } else {
                            fail.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let elem_type = match case.params.dtype {
                                ScalarType::I32 => crate::testing::test_cpp_parser::ElementType::I32,
                                ScalarType::I16 => crate::testing::test_cpp_parser::ElementType::I16,
                                ScalarType::I8 => crate::testing::test_cpp_parser::ElementType::I8,
                            };
                            let detail = format_mismatch(&emu_output, &npu_output, elem_type);
                            eprintln!("[{}/{}] seed {} MISMATCH: {}", i + 1, total, case.seed, detail);
                            if verbose {
                                let _ = std::fs::write(case.case_dir.join("emu_output.bin"), &emu_output);
                                let _ = std::fs::write(case.case_dir.join("npu_output.bin"), &npu_output);
                            }
                        }
                    }
                    (Err(e), _) => {
                        error.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if verbose {
                            eprintln!("[{}/{}] seed {} emulator error: {}", i + 1, total, case.seed, e);
                        }
                    }
                    (_, Err(e)) => {
                        error.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if verbose {
                            eprintln!("[{}/{}] seed {} hw error: {}", i + 1, total, case.seed, e);
                        }
                    }
                }

                if !verbose {
                    let p = pass.load(std::sync::atomic::Ordering::Relaxed);
                    let f = fail.load(std::sync::atomic::Ordering::Relaxed);
                    let e = error.load(std::sync::atomic::Ordering::Relaxed);
                    eprint!("\r[{}/{}] {} pass, {} fail, {} error", i + 1, total, p, f, e);
                }
            }
        });
    });

    if !verbose {
        eprintln!();
    }

    (
        pass.into_inner(),
        fail.into_inner(),
        error.into_inner(),
    )
}

/// Emulator-only execution (no hardware comparison).
fn execute_emulator_only(
    cases: &[CompiledCase],
    max_cycles: u64,
    verbose: bool,
) -> (usize, usize, usize) {
    let total = cases.len();
    let mut pass = 0usize;
    let mut error = 0usize;

    for (i, case) in cases.iter().enumerate() {
        let xclbin_path = case.case_dir.join("aie.xclbin");
        match run_emulator(&xclbin_path, &case.params, max_cycles) {
            Ok(output) if !output.is_empty() => {
                pass += 1;
                if verbose {
                    println!("[{}/{}] seed {} emulator: {} bytes", i + 1, total, case.seed, output.len());
                }
            }
            Ok(_) => {
                error += 1;
                if verbose {
                    eprintln!("[{}/{}] seed {} emulator produced empty output", i + 1, total, case.seed);
                }
            }
            Err(e) => {
                error += 1;
                if verbose {
                    eprintln!("[{}/{}] seed {} emulator error: {}", i + 1, total, case.seed, e);
                }
            }
        }
        if !verbose {
            print!("\r[{}/{}] {} pass, {} error", i + 1, total, pass, error);
            std::io::stdout().flush().ok();
        }
    }
    if !verbose {
        println!();
    }
    // No fail category in emulator-only mode (only pass/error).
    (pass, 0, error)
}

/// Run a fuzz case on real NPU hardware and return raw output bytes.
fn run_on_npu_raw(
    case: &CompiledCase,
    xclbin_path: &Path,
    insts_path: &Path,
) -> Result<Vec<u8>, String> {
    use crate::testing::npu_runner;
    use crate::testing::test_cpp_parser::{
        BufferDef, BufferDir, BufferSpec, ElementType, InputPattern,
    };

    if !npu_runner::npu_available() {
        return Err("NPU hardware not available".into());
    }

    let elem_type = match case.params.dtype {
        ScalarType::I32 => ElementType::I32,
        ScalarType::I16 => ElementType::I16,
        ScalarType::I8 => ElementType::I8,
    };

    let spec = BufferSpec {
        buffers: vec![
            BufferDef {
                name: "buf_in".to_string(),
                group_id: 3,
                size_elements: case.params.buffer_size,
                element_type: elem_type,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Sequential { start: 1, step: 1 },
            },
            BufferDef {
                name: "buf_scratch".to_string(),
                group_id: 4,
                size_elements: case.params.buffer_size,
                element_type: elem_type,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_out".to_string(),
                group_id: 5,
                size_elements: case.params.buffer_size,
                element_type: elem_type,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
        ],
        multi_kernel: false,
    };

    let test_name = format!("fuzz_seed_{}", case.seed);
    match npu_runner::run_on_npu(&spec, &test_name, xclbin_path, insts_path, 30) {
        Ok(result) => Ok(result.output),
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

    // Step 2: Generate MLIR template
    let mut template_cmd = Command::new(&tools.python);
    template_cmd
        .arg(&tools.template_script)
        .arg("--kernel").arg("fuzz_kernel.cc")
        .arg("--size").arg(params.buffer_size.to_string())
        .arg("--dtype").arg(dtype_str(params.dtype))
        .arg("--outdir").arg(case_dir)
        .arg("--device").arg("npu1_1col");
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

/// Run a compiled fuzz case through the emulator.
///
/// Returns the raw output buffer bytes.
fn run_emulator(
    xclbin_path: &Path,
    params: &FuzzParams,
    max_cycles: u64,
) -> Result<Vec<u8>, String> {
    use crate::testing::test_cpp_parser::{
        BufferDef, BufferDir, BufferSpec, ElementType, InputPattern,
    };

    let elem_type = match params.dtype {
        ScalarType::I32 => ElementType::I32,
        ScalarType::I16 => ElementType::I16,
        ScalarType::I8 => ElementType::I8,
    };

    // Build the same BufferSpec used for NPU execution so the emulator
    // allocates identically-sized buffers with the same input data.
    let spec = BufferSpec {
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
        ],
        multi_kernel: false,
    };

    let test = XclbinTest::from_path(xclbin_path).with_buffer_spec(spec);
    let suite = XclbinSuite::new().with_max_cycles(max_cycles);
    let (_outcome, raw_output) = suite.run_single_with_output(&test);
    raw_output.ok_or_else(|| "Emulator produced no output".into())
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
