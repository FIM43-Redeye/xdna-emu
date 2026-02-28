//! Fuzz iteration loop: generate, compile, run, compare.
//!
//! Orchestrates the full fuzzing pipeline. Each iteration:
//! 1. Generate random `FuzzParams` from a seed
//! 2. Lower to C++ kernel source
//! 3. Compile to xclbin via IRON template + aiecc.py
//! 4. Run on emulator
//! 5. (future) Run on real NPU hardware
//! 6. Compare outputs

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

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

/// Run the fuzz loop: generate, compile, run on emulator + NPU, compare.
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

    let mut pass = 0usize;
    let fail = 0usize;
    let mut error = 0usize;

    for i in 0..iterations {
        let seed = base_seed.wrapping_add(i as u64);
        let params = gen::generate(seed);
        let case_dir = fuzz_dir.join(format!("seed_{}", seed));
        std::fs::create_dir_all(&case_dir).ok();

        // 1. Lower to C++
        let cpp = lower_cpp::lower_to_cpp(&params);
        let kernel_path = case_dir.join("fuzz_kernel.cc");
        if let Err(e) = std::fs::write(&kernel_path, &cpp) {
            error += 1;
            if opts.verbose {
                eprintln!("[{}/{}] seed {} write error: {}", i + 1, iterations, seed, e);
            }
            continue;
        }

        if opts.verbose {
            println!("[{}/{}] seed {} generated {} ops, {} bytes C++",
                i + 1, iterations, seed, params.body.ops.len(), cpp.len());
        }

        // 2. Compile to xclbin
        match compile_fuzz_case(&tools, &params, &case_dir) {
            Ok(_) => {}
            Err(e) => {
                error += 1;
                if opts.verbose {
                    eprintln!("[{}/{}] seed {} compile error: {}", i + 1, iterations, seed, e);
                }
                continue;
            }
        }

        // 3. Run on emulator
        let xclbin_path = case_dir.join("aie.xclbin");
        let emu_output = match run_emulator(&xclbin_path, &params, opts.max_cycles) {
            Ok(output) => output,
            Err(e) => {
                error += 1;
                if opts.verbose {
                    eprintln!("[{}/{}] seed {} emulator error: {}", i + 1, iterations, seed, e);
                }
                continue;
            }
        };

        if opts.verbose {
            println!("[{}/{}] seed {} emulator produced {} output bytes",
                i + 1, iterations, seed, emu_output.len());
        }

        // 4. TODO: Run on real NPU hardware and compare.
        // For now, consider the emulator run a success if it produced output
        // without hanging or crashing. Hardware comparison is Phase 1.5.
        if !emu_output.is_empty() {
            pass += 1;
        } else {
            error += 1;
            if opts.verbose {
                eprintln!("[{}/{}] seed {} emulator produced empty output", i + 1, iterations, seed);
            }
        }

        // Progress (overwrite line in non-verbose mode)
        if !opts.verbose {
            print!("\r[{}/{}] {} pass, {} fail, {} error",
                i + 1, iterations, pass, fail, error);
            std::io::stdout().flush().ok();
        }
    }

    if !opts.verbose {
        println!(); // final newline after progress line
    }
    println!("Fuzz complete: {} pass, {} fail, {} error", pass, fail, error);

    if fail > 0 {
        std::process::exit(1);
    }
}

/// Compile a fuzz case: kernel.cc -> kernel.o, template -> aie.mlir -> xclbin.
fn compile_fuzz_case(
    tools: &ToolPaths,
    params: &FuzzParams,
    case_dir: &Path,
) -> Result<(), String> {
    // Step 1: Compile kernel to .o with Peano clang
    let kernel_cc = case_dir.join("fuzz_kernel.cc");
    let kernel_obj = case_dir.join("fuzz_kernel.cc.o");

    let mut compile_cmd = Command::new(&tools.peano_clang);
    compile_cmd
        .arg(format!("--target=aie2-none-unknown-elf"))
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
    let xclbin = case_dir.join("aie.xclbin");
    let insts = case_dir.join("insts.bin");
    if !xclbin.exists() {
        return Err("aiecc.py succeeded but aie.xclbin not found".into());
    }
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
    _params: &FuzzParams,
    max_cycles: u64,
) -> Result<Vec<u8>, String> {
    let test = XclbinTest::from_path(xclbin_path);
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
}
