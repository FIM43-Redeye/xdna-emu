//! Shared build-toolchain infrastructure for the fuzzer engine.
//!
//! Hosts `ToolPaths` (external tool discovery and environment setup),
//! `TRACE_BUFFER_ELEMENTS`, `catch_panic`, and `compile_kernel_case` -- the
//! four items shared between the scalar and vector fuzz runners.  Lifted out
//! of `fuzzer::runner` in the framework Step 1 refactor so both fuzzers and
//! the future generic engine share a single home.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Paths to external tools needed for compilation.
///
/// `pub(crate)` so the vector fuzz runner (`super::vector::runner`) reuses the
/// same compile pipeline.
pub(crate) struct ToolPaths {
    /// Peano clang compiler.
    pub(crate) peano_clang: PathBuf,
    /// Python interpreter from ironenv.
    python: PathBuf,
    /// Path to aiecc.py.
    pub(crate) aiecc: PathBuf,
    /// PYTHONPATH for mlir-aie modules.
    pythonpath: String,
    /// Path to fuzz_template.py.
    pub(crate) template_script: PathBuf,
    /// Peano install dir (for PEANO_INSTALL_DIR env).
    peano_dir: PathBuf,
    /// mlir-aie bin dir (for PATH).
    mlir_aie_bin: PathBuf,
    /// aietools root (optional, for XILINX_VITIS_AIETOOLS).
    aietools_root: Option<PathBuf>,
}

impl ToolPaths {
    /// Discover tool paths from the environment.
    pub(crate) fn discover() -> Result<Self, String> {
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

    /// Include dir for aie_api headers (`<mlir-aie install>/include`).
    /// Scalar fuzz kernels are plain C, but vector kernels need aie_api.
    pub(crate) fn aie_api_include(&self) -> Option<PathBuf> {
        self.mlir_aie_bin.parent().map(|root| root.join("include"))
    }

    /// Apply build environment variables to a Command.
    pub(crate) fn apply_env(&self, cmd: &mut Command) {
        cmd.env("PEANO_INSTALL_DIR", &self.peano_dir);

        // Build PATH with mlir-aie bin + peano bin
        let mut path = std::env::var("PATH").unwrap_or_default();
        path = format!("{}:{}:{}", self.mlir_aie_bin.display(), self.peano_dir.join("bin").display(), path);
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
pub(crate) const TRACE_BUFFER_ELEMENTS: usize = 262_144;

/// Run `f`, converting a panic into `Err(message)`.
///
/// The emulator workers run inside `std::thread::scope`, which re-raises a
/// scoped thread's panic when the scope joins -- so a single panicking seed
/// would otherwise abort the entire batch with no report.  A panic is the
/// highest-signal find a differential fuzzer can produce (an unambiguous
/// emulator bug), so callers catch it here and surface it as its own CRASH
/// category rather than letting it take the run down.
pub(crate) fn catch_panic<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(panic_to_string)
}

/// Extract a human-readable message from a caught panic payload.
fn panic_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Compile a `fuzz_kernel.cc` case directory to xclbin, parameterized by raw
/// buffer size (elements) and dtype string. Shared between the scalar fuzzer
/// (via [`compile_fuzz_case`]) and the vector fuzzer.
pub(crate) fn compile_kernel_case(
    tools: &ToolPaths,
    case_dir: &Path,
    buffer_size: usize,
    dtype: &str,
) -> Result<(), String> {
    // Skip if already compiled (xclbin exists and is newer than source).
    let xclbin = case_dir.join("aie.xclbin");
    let kernel_cc = case_dir.join("fuzz_kernel.cc");
    if xclbin.exists() {
        if let (Ok(src_meta), Ok(xclbin_meta)) = (std::fs::metadata(&kernel_cc), std::fs::metadata(&xclbin)) {
            if let (Ok(src_time), Ok(xclbin_time)) = (src_meta.modified(), xclbin_meta.modified()) {
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
        .arg("-std=c++20") // aie_api headers are C++20 (concepts)
        .arg("-c")
        .arg(&kernel_cc)
        .arg("-o")
        .arg(&kernel_obj);
    if let Some(inc) = tools.aie_api_include() {
        compile_cmd.arg("-I").arg(inc);
    }
    tools.apply_env(&mut compile_cmd);

    let compile_out = compile_cmd
        .output()
        .map_err(|e| format!("Failed to spawn Peano clang: {}", e))?;
    if !compile_out.status.success() {
        let stderr = String::from_utf8_lossy(&compile_out.stderr);
        return Err(format!(
            "Kernel compilation failed:\n{}",
            stderr.lines().take(10).collect::<Vec<_>>().join("\n")
        ));
    }

    // Step 2: Generate MLIR template (always with trace instrumentation)
    let mut template_cmd = Command::new(&tools.python);
    template_cmd
        .arg(&tools.template_script)
        .arg("--kernel")
        .arg("fuzz_kernel.cc")
        .arg("--size")
        .arg(buffer_size.to_string())
        .arg("--dtype")
        .arg(dtype)
        .arg("--outdir")
        .arg(case_dir)
        .arg("--device")
        .arg("npu1_1col")
        .arg("--trace");
    tools.apply_env(&mut template_cmd);

    let template_out = template_cmd
        .output()
        .map_err(|e| format!("Failed to spawn fuzz_template.py: {}", e))?;
    if !template_out.status.success() {
        let stderr = String::from_utf8_lossy(&template_out.stderr);
        return Err(format!(
            "MLIR template generation failed:\n{}",
            stderr.lines().take(10).collect::<Vec<_>>().join("\n")
        ));
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

    let aiecc_out = aiecc_cmd.output().map_err(|e| format!("Failed to spawn aiecc.py: {}", e))?;
    if !aiecc_out.status.success() {
        let stderr = String::from_utf8_lossy(&aiecc_out.stderr);
        let stdout = String::from_utf8_lossy(&aiecc_out.stdout);
        let combined = if stderr.is_empty() { stdout } else { stderr };
        return Err(format!(
            "aiecc.py failed:\n{}",
            combined.lines().take(10).collect::<Vec<_>>().join("\n")
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catch_panic_returns_ok_for_non_panicking_closure() {
        let r = catch_panic(|| 42);
        assert_eq!(r, Ok(42));
    }

    #[test]
    fn catch_panic_converts_str_panic_to_err() {
        let r: Result<i32, String> = catch_panic(|| panic!("boom-from-emulator"));
        let msg = r.expect_err("panic must be caught as Err");
        assert!(msg.contains("boom-from-emulator"), "panic message preserved, got: {}", msg);
    }

    #[test]
    fn catch_panic_converts_formatted_panic_to_err() {
        // The negate-overflow panic arrives as a String payload, not &str.
        let r: Result<i32, String> = catch_panic(|| panic!("value {}", 7));
        let msg = r.expect_err("panic must be caught as Err");
        assert!(msg.contains("value 7"), "formatted panic message preserved, got: {}", msg);
    }
}
