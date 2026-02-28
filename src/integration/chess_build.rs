//! Build orchestration for mlir-aie tests via aiecc.py.
//!
//! Supports both compiler backends:
//! - **Peano** (open-source LLVM fork): `--no-xchesscc`
//! - **Chess** (AMD proprietary): `--xchesscc` (requires aietools)
//!
//! The `BuildEnv` struct discovers and caches the build environment
//! (Python venv, aiecc.py, Peano, PYTHONPATH) once at startup, then
//! `build()` invocations reuse it for each test.
//!
//! # Prerequisites
//!
//! - mlir-aie must be built with `ironenv/` Python venv present
//! - `PEANO_INSTALL_DIR` must point to Peano (llvm-aie/install)
//! - For Chess builds: `XILINX_VITIS_AIETOOLS` must be set
//!
//! Source `activate-npu-env.sh` before using this module.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::aietools::AieTools;
use super::bridge;
use crate::config::Config;
use crate::testing::artifacts;

/// Result of a build (either Peano or Chess).
#[derive(Debug)]
pub struct BuildResult {
    /// Path to the produced xclbin.
    pub xclbin: PathBuf,
    /// Path to the NPU instruction binary.
    pub insts: Option<PathBuf>,
    /// Path to the .prj directory (needed for aiesimulator).
    pub prj_dir: Option<PathBuf>,
    /// Build output (stdout + stderr) for diagnostics.
    pub build_log: String,
    /// Synthetic aiecc.py commands from build manifest (for multi-xclbin).
    ///
    /// Populated by `build_example()` via the bridge `build-manifest`
    /// subcommand. Each entry is a synthetic command string containing
    /// `--xclbin-name=` and optionally `--npu-insts-name=` flags, in the
    /// format expected by `find_all_xclbin_results()`.
    pub build_commands: Vec<String>,
}

/// Options controlling a single build invocation.
#[derive(Debug, Clone)]
pub struct BuildOpts {
    /// Use Chess compiler (`--xchesscc`) instead of Peano (`--no-xchesscc`).
    pub use_chess: bool,
    /// Generate simulator artifacts (omit `--no-aiesim`).
    /// Required for aiesimulator runs. Only meaningful with Chess.
    pub gen_sim: bool,
    /// Device name for NPUDEVICE substitution (e.g. "npu1_1col").
    pub device: String,
    /// Unix nice level for the build process (0-19). When set, the build
    /// command is wrapped with `nice -n <level>` to reduce scheduling
    /// priority and avoid starving the system during heavy compilation.
    pub nice: Option<i32>,
    /// Force rebuild even if cached artifacts are fresh.
    pub force_rebuild: bool,
}

/// Cached build environment for aiecc.py invocations.
///
/// Discovered once at startup via `BuildEnv::discover()`, then reused
/// for each test build. All paths are validated at discovery time.
#[derive(Debug, Clone)]
pub struct BuildEnv {
    /// Python interpreter from ironenv venv.
    python: PathBuf,
    /// Path to aiecc.py script.
    aiecc: PathBuf,
    /// PYTHONPATH for mlir-aie Python modules (colon-separated).
    pythonpath: String,
    /// Peano compiler installation directory.
    peano_dir: PathBuf,
    /// mlir-aie bin directory (for PATH: aie-opt, aie-translate).
    mlir_aie_bin: PathBuf,
    /// aietools root, if available (for XILINX_VITIS_AIETOOLS).
    aietools_root: Option<PathBuf>,
    /// Canonicalized mlir-aie root (for deriving test_lib paths).
    /// Used at construction but not read back yet (pending Chess integration).
    #[allow(dead_code)]
    mlir_aie_root: PathBuf,
    /// Include path for test_lib (test_library.h), if available.
    test_lib_include: Option<PathBuf>,
    /// Library path for test_lib (libtest_lib.a), if available.
    test_lib_lib: Option<PathBuf>,
    /// Modification time of the toolchain (max mtime of key binaries like
    /// aie-opt). Used by cache checks to invalidate when mlir-aie is rebuilt
    /// even if test sources are unchanged.
    toolchain_mtime: Option<std::time::SystemTime>,
}

/// Check whether a cached xclbin is still valid given source and toolchain timestamps.
///
/// The xclbin is valid only if it is newer than BOTH the source file AND
/// the toolchain. This catches the case where mlir-aie is rebuilt (changing
/// buffer allocation, CDO generation, etc.) but the test source is unchanged.
fn is_cache_valid(
    xclbin_mtime: std::time::SystemTime,
    source_mtime: std::time::SystemTime,
    toolchain_mtime: Option<std::time::SystemTime>,
) -> bool {
    if xclbin_mtime <= source_mtime {
        return false;
    }
    if let Some(tc) = toolchain_mtime {
        if xclbin_mtime <= tc {
            return false;
        }
    }
    true
}

/// Find an xclbin produced by an example build.
///
/// Returns the first `*.xclbin` found alphabetically in the directory.
/// With the build manifest in place, this is a fallback -- the primary
/// xclbin comes from the manifest. Returns `None` if the directory
/// does not exist or contains no xclbin files.
fn find_example_xclbin(build_dir: &Path) -> Option<PathBuf> {
    artifacts::collect_xclbins(build_dir).into_iter().next()
}

impl BuildEnv {
    /// Discover the build environment from config and filesystem.
    ///
    /// Search order mirrors `build-mlir-aie-tests.sh`:
    /// - ironenv at `{mlir_aie}/ironenv/`
    /// - aiecc.py at `{mlir_aie}/install/bin/` or `{mlir_aie}/build/bin/`
    /// - PYTHONPATH from `install/python`, `build/python`, `my_install/python`
    /// - Peano from `PEANO_INSTALL_DIR` env or `../llvm-aie/install`
    ///
    /// Returns `Err` with a descriptive message if any required component
    /// is missing.
    pub fn discover(config: &Config) -> Result<Self, String> {
        let mlir_aie_raw = PathBuf::from(config.mlir_aie_path());
        if !mlir_aie_raw.exists() {
            return Err(format!(
                "mlir-aie not found at {} (set mlir_aie_path in config)",
                mlir_aie_raw.display()
            ));
        }
        // Canonicalize so all derived paths are absolute -- critical because
        // build() changes the working directory to the output dir.
        let mlir_aie = mlir_aie_raw.canonicalize()
            .map_err(|e| format!("Failed to canonicalize {}: {}", mlir_aie_raw.display(), e))?;

        // Python venv
        let ironenv = mlir_aie.join("ironenv");
        let python = ironenv.join("bin/python3");
        if !python.exists() {
            return Err(format!(
                "ironenv not found at {} -- build mlir-aie first",
                ironenv.display()
            ));
        }

        // aiecc.py: prefer install/bin over build/bin
        let aiecc = Self::find_first_existing(&[
            mlir_aie.join("install/bin/aiecc.py"),
            mlir_aie.join("build/bin/aiecc.py"),
        ]).ok_or_else(|| format!(
            "aiecc.py not found in {}/install/bin or {}/build/bin",
            mlir_aie.display(), mlir_aie.display()
        ))?;

        // PYTHONPATH: all Python module directories that exist (canonicalized)
        let python_dirs: Vec<String> = [
            mlir_aie.join("install/python"),
            mlir_aie.join("build/python"),
            mlir_aie.join("my_install/python"),
        ]
        .iter()
        .filter(|p| p.exists())
        .filter_map(|p| p.canonicalize().ok())
        .map(|p| p.to_string_lossy().to_string())
        .collect();

        if python_dirs.is_empty() {
            return Err(format!(
                "No Python module directories found under {} (install/python, build/python, my_install/python)",
                mlir_aie.display()
            ));
        }

        // Append any existing PYTHONPATH
        let mut pythonpath = python_dirs.join(":");
        if let Ok(existing) = std::env::var("PYTHONPATH") {
            if !existing.is_empty() {
                pythonpath.push(':');
                pythonpath.push_str(&existing);
            }
        }

        // mlir-aie bin directory (same as aiecc.py's parent)
        let mlir_aie_bin = aiecc.parent()
            .ok_or("aiecc.py has no parent directory")?
            .to_path_buf();

        // Peano: env var first, then sibling directory fallback
        let peano_dir = Self::find_peano(&mlir_aie)?;

        // aietools: optional, from env or config
        let aietools_root = config.aietools_path()
            .or_else(|| std::env::var("XILINX_VITIS_AIETOOLS").ok().map(PathBuf::from))
            .filter(|p| p.exists());

        // test_lib paths: needed for unit test compilation (test_library.h + libtest_lib.a).
        // These are built by mlir-aie's test_lib cmake target.
        let test_lib_include = {
            let p = mlir_aie.join("install/runtime_lib/x86_64/test_lib/include");
            if p.is_dir() { Some(p) } else { None }
        };
        let test_lib_lib = {
            let p = mlir_aie.join("install/runtime_lib/x86_64/test_lib/lib");
            if p.is_dir() { Some(p) } else { None }
        };

        // Toolchain mtime: use aie-opt as sentinel for mlir-aie rebuilds.
        // aie-opt is always rebuilt when mlir-aie changes; aiecc.py in the
        // install dir can have a stale timestamp from the initial copy.
        let toolchain_mtime = mlir_aie_bin.join("aie-opt")
            .metadata()
            .and_then(|m| m.modified())
            .ok();

        Ok(BuildEnv {
            python,
            aiecc,
            pythonpath,
            peano_dir,
            mlir_aie_bin,
            aietools_root,
            mlir_aie_root: mlir_aie,
            test_lib_include,
            test_lib_lib,
            toolchain_mtime,
        })
    }

    /// Build a test from MLIR source using aiecc.py.
    ///
    /// 1. Copies MLIR to output_dir with NPUDEVICE substitution
    /// 2. Invokes aiecc.py with the appropriate compiler flags
    /// 3. Returns paths to produced artifacts
    ///
    /// Caching: if `{output_dir}/aie.xclbin` already exists and is newer
    /// than the MLIR source, the build is skipped.
    pub fn build(
        &self,
        mlir_source: &Path,
        output_dir: &Path,
        opts: &BuildOpts,
    ) -> Result<BuildResult, String> {
        if !mlir_source.exists() {
            return Err(format!("MLIR source not found: {}", mlir_source.display()));
        }

        // Check cache: skip build if xclbin exists and is newer than source
        let xclbin_path = output_dir.join("aie.xclbin");
        if !opts.force_rebuild {
            if let Some(result) = self.check_cache(mlir_source, output_dir, &xclbin_path) {
                return Ok(result);
            }
        }

        // Create output directory
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed to create output dir {}: {}", output_dir.display(), e))?;

        // Copy MLIR with NPUDEVICE substitution
        let mlir_content = std::fs::read_to_string(mlir_source)
            .map_err(|e| format!("Failed to read {}: {}", mlir_source.display(), e))?;
        let substituted = mlir_content.replace("NPUDEVICE", &opts.device);
        let work_mlir = output_dir.join("aie_arch.mlir");
        std::fs::write(&work_mlir, &substituted)
            .map_err(|e| format!("Failed to write {}: {}", work_mlir.display(), e))?;

        // Build aiecc.py argument list
        let mut args = Vec::new();
        args.push(self.aiecc.to_string_lossy().to_string());

        // Compiler and linker selection.
        // Chess object files contain proprietary sections (.tctmemtab,
        // .chesstypeannotationtab, etc.) that Peano's ld.lld rejects via
        // --orphan-handling=error. Chess compilation requires Chess linking
        // (xbridge) too. Additionally, aiecc.py requires --xbridge for
        // --aiesim support.
        if opts.use_chess {
            args.push("--xchesscc".to_string());
            args.push("--xbridge".to_string());
        } else {
            args.push("--no-xchesscc".to_string());
            args.push("--no-xbridge".to_string());
        }

        // Simulator artifacts: aiecc.py defaults to no sim generation,
        // so we must explicitly pass --aiesim to produce .prj/sim/.
        if opts.gen_sim {
            args.push("--aiesim".to_string());
        }

        // Output generation
        args.push("--aie-generate-xclbin".to_string());
        args.push("--xclbin-name=aie.xclbin".to_string());
        args.push("--aie-generate-npu-insts".to_string());
        args.push("--npu-insts-name=insts.bin".to_string());
        args.push("--no-compile-host".to_string());

        // Source file (last argument)
        args.push(work_mlir.to_string_lossy().to_string());

        // Build environment.
        // When nice is set, wrap with `nice -n <level>` so compilation
        // runs at reduced priority and does not starve interactive work.
        let mut cmd = if let Some(n) = opts.nice {
            let mut c = Command::new("nice");
            c.args(["-n", &n.to_string()]);
            c.arg(&self.python);
            c
        } else {
            Command::new(&self.python)
        };
        cmd.args(&args);
        cmd.current_dir(output_dir);
        self.apply_env(&mut cmd);

        let compiler_label = if opts.use_chess { "Chess" } else { "Peano" };
        log::info!("Building with {} in {}", compiler_label, output_dir.display());

        let output = cmd.output()
            .map_err(|e| format!("Failed to execute aiecc.py: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let build_log = format!("--- stdout ---\n{}\n--- stderr ---\n{}", stdout, stderr);

        if !output.status.success() {
            // aiecc.py may return non-zero even when the xclbin was produced
            // successfully (e.g. the ps.so sim shim compilation fails due to
            // missing host libraries). If xclbin exists, treat as partial
            // success -- the emulator path works, aiesim may not.
            if xclbin_path.exists() {
                log::warn!(
                    "{} build exited with {} but xclbin was produced (partial success)",
                    compiler_label, output.status.code().unwrap_or(-1)
                );
            } else if let Some(found) = artifacts::collect_xclbins(output_dir).into_iter().next() {
                log::warn!(
                    "{} build exited with {} but found {} (partial success)",
                    compiler_label, output.status.code().unwrap_or(-1), found.display()
                );
                return Ok(BuildResult {
                    xclbin: found,
                    insts: artifacts::find_insts(output_dir),
                    prj_dir: artifacts::find_prj_dir(output_dir),
                    build_log,
                    build_commands: vec![],
                });
            } else {
                return Err(format!(
                    "{} build failed (exit {}):\n{}",
                    compiler_label,
                    output.status.code().unwrap_or(-1),
                    build_log
                ));
            }
        }

        // Find produced artifacts
        if !xclbin_path.exists() {
            // Try finding any xclbin in the output dir
            if let Some(found) = artifacts::collect_xclbins(output_dir).into_iter().next() {
                return Ok(BuildResult {
                    xclbin: found,
                    insts: artifacts::find_insts(output_dir),
                    prj_dir: artifacts::find_prj_dir(output_dir),
                    build_log,
                    build_commands: vec![],
                });
            }
            return Err(format!(
                "{} build succeeded but no xclbin found in {}",
                compiler_label, output_dir.display()
            ));
        }

        Ok(BuildResult {
            xclbin: xclbin_path,
            insts: artifacts::find_insts(output_dir),
            prj_dir: artifacts::find_prj_dir(output_dir),
            build_log,
            build_commands: vec![],
        })
    }

    /// Build a chess_compiler_tests_aie2 unit test for aiesimulator.
    ///
    /// Instead of hardcoding the build sequence, this method executes the
    /// `// RUN:` lines parsed from the test's MLIR file. Each test carries
    /// its own build recipe (the same approach LLVM's `lit` test runner
    /// uses), making us forward-compatible with any build pattern:
    ///
    /// - **Pre-compile kernel**: `xchesscc -c kernel.cc` then `aiecc.py`
    /// - **BCF two-step**: `aiecc.py` then `xchesscc +l <bcf> kernel.cc`
    /// - **MLIR-only**: just `aiecc.py`
    ///
    /// The flow:
    /// 1. Copy MLIR, test.cpp, and kernel files to output directory
    /// 2. For each build step: expand lit substitutions, execute via `sh -c`
    /// 3. Return the .prj directory path
    ///
    /// # Returns
    ///
    /// Path to the .prj directory on success, or an error message.
    pub fn build_unit_test(
        &self,
        test: &crate::testing::unit_test::UnitTest,
        output_dir: &Path,
        nice: Option<i32>,
    ) -> Result<crate::testing::unit_test::UnitTestBuildResult, String> {
        // Check test_lib availability (required for %test_lib_flags expansion)
        let test_lib_include = self.test_lib_include.as_ref()
            .ok_or("test_lib include dir not found -- build mlir-aie test_lib target")?;
        let test_lib_lib = self.test_lib_lib.as_ref()
            .ok_or("test_lib lib dir not found -- build mlir-aie test_lib target")?;

        // Check cache: if .prj/sim/ps/ps.so exists and is newer than all sources
        if let Some(result) = self.check_unit_test_cache(test, output_dir) {
            return Ok(result);
        }

        if test.build_steps.is_empty() {
            return Err(format!(
                "No build steps found in {} (no // RUN: lines)",
                test.mlir_file.display()
            ));
        }

        // Create output directory
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed to create {}: {}", output_dir.display(), e))?;

        // Copy source files to output directory
        let mlir_filename = test.mlir_file.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let work_mlir = output_dir.join(&mlir_filename);
        std::fs::copy(&test.mlir_file, &work_mlir)
            .map_err(|e| format!("Failed to copy MLIR: {}", e))?;

        std::fs::copy(&test.test_cpp, output_dir.join("test.cpp"))
            .map_err(|e| format!("Failed to copy test.cpp: {}", e))?;

        for kernel_src in &test.kernel_sources {
            let dest = output_dir.join(kernel_src.file_name().unwrap_or_default());
            std::fs::copy(kernel_src, &dest)
                .map_err(|e| format!("Failed to copy {}: {}", kernel_src.display(), e))?;
        }

        // Execute build steps from // RUN: lines
        let mut build_log = String::new();
        let step_count = test.build_steps.len();

        log::info!(
            "Building unit test {} ({} steps) in {}",
            test.name, step_count, output_dir.display()
        );

        for (step_idx, step) in test.build_steps.iter().enumerate() {
            let expanded = expand_lit_subs(
                step,
                &self.python,
                &self.aiecc,
                output_dir,
                &mlir_filename,
                Some(test_lib_include.as_path()),
                Some(test_lib_lib.as_path()),
                None,  // source_dir: use output_dir for %S (unit test behavior)
                None,  // aietools: not needed for unit tests
            );

            log::info!(
                "  step {}/{}: {}",
                step_idx + 1, step_count, expanded
            );

            // Execute via sh -c so shell constructs (pipes, quoting) work
            // and PATH from apply_env() is inherited by child processes.
            let mut cmd = if let Some(n) = nice {
                let mut c = Command::new("nice");
                c.args(["-n", &n.to_string(), "sh", "-c", &expanded]);
                c
            } else {
                let mut c = Command::new("sh");
                c.args(["-c", &expanded]);
                c
            };
            cmd.current_dir(output_dir);
            self.apply_env(&mut cmd);

            let output = cmd.output()
                .map_err(|e| format!(
                    "Build step {}/{} failed to execute: {}",
                    step_idx + 1, step_count, e
                ))?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            build_log.push_str(&format!(
                "--- step {}/{} ---\n{}\n--- stdout ---\n{}\n--- stderr ---\n{}\n",
                step_idx + 1, step_count, expanded, stdout, stderr
            ));

            if !output.status.success() {
                // Partial success: if .prj already exists, the failed step
                // may be ps.so compilation (known issue). Log and continue.
                if artifacts::find_prj_dir(output_dir).is_some() {
                    log::warn!(
                        "Build step {}/{} exited with {} but .prj exists (partial success)",
                        step_idx + 1, step_count,
                        output.status.code().unwrap_or(-1)
                    );
                    build_log.push_str(&format!(
                        "*** partial success: .prj exists despite exit code {} ***\n",
                        output.status.code().unwrap_or(-1)
                    ));
                } else {
                    return Err(format!(
                        "Build step {}/{} failed (exit {}):\n{}",
                        step_idx + 1, step_count,
                        output.status.code().unwrap_or(-1),
                        build_log
                    ));
                }
            }
        }

        // Find the .prj directory produced by aiecc.py
        let prj_dir = artifacts::find_prj_dir(output_dir).ok_or_else(|| format!(
            "Build completed but no .prj directory found in {}",
            output_dir.display()
        ))?;

        Ok(crate::testing::unit_test::UnitTestBuildResult {
            prj_dir,
            build_log,
        })
    }

    /// Build an NPU test from source using RUN-line-driven build steps.
    ///
    /// Follows the same RUN-line-driven approach as `build_unit_test()`: each
    /// test carries its own build recipe parsed from `run.lit` or `aie2.py`.
    /// The steps are filtered to build-only commands, expanded with lit
    /// substitutions, and executed sequentially.
    ///
    /// Unlike `build()` which takes raw MLIR and hardcodes the aiecc.py
    /// invocation, this method supports the full range of test patterns:
    /// - Static MLIR with NPUDEVICE substitution (cp + sed + aiecc.py)
    /// - Python MLIR generation (aie2.py + aiecc.py)
    /// - Chess kernel pre-compilation (xchesscc_wrapper + aiecc.py)
    ///
    /// `%S` expands to the source directory (not output_dir), matching how
    /// lit resolves paths. All build products go to output_dir (CWD).
    pub fn build_npu_test(
        &self,
        test: &crate::testing::npu_test::NpuTestSource,
        output_dir: &Path,
        opts: &BuildOpts,
    ) -> Result<BuildResult, String> {
        // Check cache: skip build if xclbin exists and is newer than entry file
        if !opts.force_rebuild {
            for name in &["aie.xclbin", "final.xclbin"] {
                let path = output_dir.join(name);
                if let Some(result) = self.check_npu_test_cache(test, output_dir, &path) {
                    return Ok(result);
                }
            }
        }

        if test.build_steps.is_empty() {
            return Err(format!(
                "No build steps for test {} (no RUN lines after filtering)",
                test.name
            ));
        }

        // Create output directory
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed to create {}: {}", output_dir.display(), e))?;

        // Execute build steps from filtered RUN lines
        let mut build_log = String::new();
        let step_count = test.build_steps.len();
        let compiler_label = if opts.use_chess { "Chess" } else { "Peano" };

        log::info!(
            "Building NPU test {} with {} ({} steps) in {}",
            test.name, compiler_label, step_count, output_dir.display()
        );

        // Entry filename for %s substitution (rarely used in NPU tests)
        let entry_filename = test.entry_file.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        for (step_idx, step) in test.build_steps.iter().enumerate() {
            // Apply compiler override (Peano/Chess flags) to aiecc.py and
            // xchesscc_wrapper lines
            let peano_clang = self.peano_dir.join("bin/clang");
            let overridden = apply_compiler_override(
                step, opts.use_chess, Some(&peano_clang),
            );

            // Expand lit substitutions (%S -> source_dir, %python, %aietools, etc.)
            let expanded = expand_lit_subs(
                &overridden,
                &self.python,
                &self.aiecc,
                output_dir,
                &entry_filename,
                self.test_lib_include.as_deref(),
                self.test_lib_lib.as_deref(),
                Some(test.source_dir.as_path()),
                self.aietools_root.as_deref(),
            );

            log::info!(
                "  step {}/{}: {}",
                step_idx + 1, step_count, expanded
            );

            // Execute via sh -c so shell constructs (pipes, redirects) work
            let mut cmd = if let Some(n) = opts.nice {
                let mut c = Command::new("nice");
                c.args(["-n", &n.to_string(), "sh", "-c", &expanded]);
                c
            } else {
                let mut c = Command::new("sh");
                c.args(["-c", &expanded]);
                c
            };
            cmd.current_dir(output_dir);
            self.apply_env(&mut cmd);

            let output = cmd.output()
                .map_err(|e| format!(
                    "Build step {}/{} failed to execute: {}",
                    step_idx + 1, step_count, e
                ))?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            build_log.push_str(&format!(
                "--- step {}/{} ---\n{}\n--- stdout ---\n{}\n--- stderr ---\n{}\n",
                step_idx + 1, step_count, expanded, stdout, stderr
            ));

            if !output.status.success() {
                // Partial success: if xclbin already exists, the failed step
                // may be non-critical (e.g. ps.so compilation).
                if !artifacts::collect_xclbins(output_dir).is_empty() {
                    log::warn!(
                        "Build step {}/{} exited with {} but xclbin exists (partial success)",
                        step_idx + 1, step_count,
                        output.status.code().unwrap_or(-1)
                    );
                } else {
                    return Err(format!(
                        "{} build step {}/{} failed (exit {}):\n{}",
                        compiler_label,
                        step_idx + 1, step_count,
                        output.status.code().unwrap_or(-1),
                        build_log
                    ));
                }
            }
        }

        // Find produced artifacts
        let xclbin = artifacts::collect_xclbins(output_dir).into_iter().next().ok_or_else(|| format!(
            "{} build completed but no xclbin found in {}",
            compiler_label, output_dir.display()
        ))?;

        Ok(BuildResult {
            xclbin,
            insts: artifacts::find_insts(output_dir),
            prj_dir: artifacts::find_prj_dir(output_dir),
            build_log,
            build_commands: vec![],
        })
    }

    /// Build a programming_example via its Makefile.
    ///
    /// For Peano: builds in `source_dir` (in-place, Makefile convention).
    /// For Chess: creates a symlinked workspace in `output_dir` so the
    /// in-place Peano build is not clobbered.
    ///
    /// Uses `CHESS=true|false` to select the compiler, which `makefile-common`
    /// translates into `--xchesscc` or `--no-xchesscc` for aiecc.py.
    pub fn build_example(
        &self,
        source_dir: &Path,
        python_source: &Path,
        output_dir: &Path,
        opts: &BuildOpts,
    ) -> Result<BuildResult, String> {
        let build_dir = output_dir.join("build");

        // Query the build manifest to discover all xclbin/insts pairs.
        // Runs `make -nBs` (dry run, no actual build) to parse aiecc.py flags.
        // Done before the cache check so cached results also get manifest info.
        let build_commands = bridge::BridgePath::discover()
            .and_then(|bp| bridge::query_build_manifest(&bp, source_dir, opts.use_chess))
            .unwrap_or_default();

        // Check cache
        if !opts.force_rebuild {
            if let Some(mut result) = self.check_example_cache(python_source, &build_dir) {
                result.build_commands = build_commands;
                return Ok(result);
            }
        }

        let compiler_label = if opts.use_chess { "Chess" } else { "Peano" };
        log::info!(
            "Building example with {} in {}",
            compiler_label, output_dir.display()
        );

        // For Chess: create symlinked workspace
        if opts.use_chess && output_dir != source_dir {
            std::fs::create_dir_all(output_dir)
                .map_err(|e| format!("Failed to create {}: {}", output_dir.display(), e))?;

            // Canonicalize source_dir so symlink targets are absolute.
            // source_dir may be relative (e.g. "../mlir-aie/programming_examples/...")
            // and relative symlink targets break when the link lives in a
            // different directory tree (build/chess_examples/).
            let abs_source = source_dir.canonicalize()
                .map_err(|e| format!("Cannot resolve {}: {}", source_dir.display(), e))?;

            // Symlink key files from source_dir into output_dir
            for entry in std::fs::read_dir(&abs_source)
                .map_err(|e| format!("Failed to read {}: {}", abs_source.display(), e))?
                .flatten()
            {
                let path = entry.path();
                let name = match path.file_name() {
                    Some(n) => n.to_os_string(),
                    None => continue,
                };
                let name_str = name.to_string_lossy();

                // Skip build dirs and hidden files
                if name_str == "build" || name_str == "_build" || name_str.starts_with('.') {
                    continue;
                }

                let dest = output_dir.join(&name);
                if !dest.exists() {
                    // Symlink with absolute target; srcdir in Makefile follows
                    // realpath so makefile-common includes resolve correctly.
                    #[cfg(unix)]
                    {
                        let _ = std::os::unix::fs::symlink(&path, &dest);
                    }
                }
            }

            // Some Makefiles include parent-level files via relative paths
            // (e.g. `include ${SELF_DIR}../makefile-common`). When SELF_DIR
            // uses $(lastword $(MAKEFILE_LIST)) without realpath, the include
            // resolves relative to the build tree, not the source tree.
            // Symlink parent-level makefile-common files so these includes work.
            if let Some(source_parent) = abs_source.parent() {
                if let Some(output_parent) = output_dir.parent() {
                    for name in &["makefile-common", "Makefile.common"] {
                        let src_file = source_parent.join(name);
                        let dst_file = output_parent.join(name);
                        if src_file.exists() && !dst_file.exists() {
                            #[cfg(unix)]
                            {
                                let _ = std::os::unix::fs::symlink(&src_file, &dst_file);
                            }
                        }
                    }
                }
            }
        }

        // Run make
        let chess_flag = if opts.use_chess { "true" } else { "false" };
        let mut cmd = if let Some(n) = opts.nice {
            let mut c = Command::new("nice");
            c.args(["-n", &n.to_string(), "make"]);
            c
        } else {
            Command::new("make")
        };

        cmd.args(["-C", &output_dir.to_string_lossy()]);
        // Use the default target (`all`), not a hardcoded path.
        // Most examples define `all: build/final.xclbin`, but some use
        // parameterized names or multiple xclbin targets.
        cmd.arg(format!("CHESS={}", chess_flag));
        self.apply_env(&mut cmd);

        let output = cmd.output()
            .map_err(|e| format!("Failed to execute make: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let build_log = format!("--- stdout ---\n{}\n--- stderr ---\n{}", stdout, stderr);

        if !output.status.success() {
            // Check if an xclbin was produced despite the error
            if find_example_xclbin(&build_dir).is_some() {
                log::warn!(
                    "{} example build exited with {} but xclbin was produced",
                    compiler_label, output.status.code().unwrap_or(-1)
                );
            } else {
                return Err(format!(
                    "{} example build failed (exit {}):\n{}",
                    compiler_label,
                    output.status.code().unwrap_or(-1),
                    build_log
                ));
            }
        }

        let xclbin_path = find_example_xclbin(&build_dir).ok_or_else(|| {
            format!(
                "{} example build completed but no xclbin found in {}",
                compiler_label, build_dir.display()
            )
        })?;

        Ok(BuildResult {
            xclbin: xclbin_path,
            insts: artifacts::find_insts(&build_dir),
            prj_dir: artifacts::find_prj_dir(&build_dir),
            build_log,
            build_commands,
        })
    }

    /// Check if a cached example build result exists and is still valid.
    fn check_example_cache(
        &self,
        python_source: &Path,
        build_dir: &Path,
    ) -> Option<BuildResult> {
        let xclbins = artifacts::collect_xclbins(build_dir);
        if xclbins.is_empty() {
            return None;
        }

        let source_modified = python_source.metadata().ok()?.modified().ok()?;

        // ALL xclbins must be newer than the source and toolchain.
        // If any single artifact is stale, the entire cache is invalid.
        for xclbin in &xclbins {
            let xclbin_modified = xclbin.metadata().ok()?.modified().ok()?;
            if !is_cache_valid(xclbin_modified, source_modified, self.toolchain_mtime) {
                return None;
            }
        }

        log::info!(
            "Using cached example build ({} xclbin(s)) in {}",
            xclbins.len(),
            build_dir.display()
        );
        Some(BuildResult {
            xclbin: xclbins.into_iter().next().unwrap(),
            insts: artifacts::find_insts(build_dir),
            prj_dir: artifacts::find_prj_dir(build_dir),
            build_log: "(cached)".to_string(),
            build_commands: vec![],
        })
    }

    /// Check if a cached NPU test build result exists and is still valid.
    ///
    /// Returns `Some(BuildResult)` if the xclbin is newer than the test's
    /// entry file (run.lit or aie2.py), meaning we can skip the build.
    fn check_npu_test_cache(
        &self,
        test: &crate::testing::npu_test::NpuTestSource,
        output_dir: &Path,
        xclbin_path: &Path,
    ) -> Option<BuildResult> {
        if !xclbin_path.exists() {
            return None;
        }

        let xclbin_modified = xclbin_path.metadata().ok()?.modified().ok()?;
        let entry_modified = test.entry_file.metadata().ok()?.modified().ok()?;

        if is_cache_valid(xclbin_modified, entry_modified, self.toolchain_mtime) {
            log::info!("Using cached NPU test build in {}", output_dir.display());
            Some(BuildResult {
                xclbin: xclbin_path.to_path_buf(),
                insts: artifacts::find_insts(output_dir),
                prj_dir: artifacts::find_prj_dir(output_dir),
                build_log: "(cached)".to_string(),
                build_commands: vec![],
            })
        } else {
            None
        }
    }

    /// Human-readable summary of the discovered environment.
    /// Whether the Chess compiler (aietools) is available for builds.
    pub fn has_chess(&self) -> bool {
        self.aietools_root.is_some()
    }

    pub fn summary(&self) -> String {
        let aietools_label = match &self.aietools_root {
            Some(p) => format!("aietools at {}", p.display()),
            None => "no aietools".to_string(),
        };
        let test_lib_label = if self.test_lib_include.is_some() {
            ", test_lib"
        } else {
            ""
        };
        format!(
            "python={}, aiecc={}, peano={}, {}{}",
            self.python.display(),
            self.aiecc.file_name().unwrap_or_default().to_string_lossy(),
            self.peano_dir.display(),
            aietools_label,
            test_lib_label,
        )
    }

    /// Check if a cached build result exists and is still valid.
    ///
    /// Returns `Some(BuildResult)` if the xclbin is newer than the MLIR source,
    /// meaning we can skip the build.
    fn check_cache(
        &self,
        mlir_source: &Path,
        output_dir: &Path,
        xclbin_path: &Path,
    ) -> Option<BuildResult> {
        if !xclbin_path.exists() {
            return None;
        }

        let source_modified = mlir_source.metadata().ok()?.modified().ok()?;
        let xclbin_modified = xclbin_path.metadata().ok()?.modified().ok()?;

        if is_cache_valid(xclbin_modified, source_modified, self.toolchain_mtime) {
            log::info!("Using cached build in {}", output_dir.display());
            Some(BuildResult {
                xclbin: xclbin_path.to_path_buf(),
                insts: artifacts::find_insts(output_dir),
                prj_dir: artifacts::find_prj_dir(output_dir),
                build_log: "(cached)".to_string(),
                build_commands: vec![],
            })
        } else {
            None
        }
    }

    /// Check if a cached unit test build result exists and is still valid.
    ///
    /// Returns `Some(UnitTestBuildResult)` if the .prj/sim/ps/ps.so is
    /// newer than all source files (MLIR, test.cpp, kernel files).
    fn check_unit_test_cache(
        &self,
        test: &crate::testing::unit_test::UnitTest,
        output_dir: &Path,
    ) -> Option<crate::testing::unit_test::UnitTestBuildResult> {
        let prj_dir = artifacts::find_prj_dir(output_dir)?;
        let ps_so = prj_dir.join("sim/ps/ps.so");
        if !ps_so.exists() {
            return None;
        }

        let ps_modified = ps_so.metadata().ok()?.modified().ok()?;

        // Check all source files are older than ps.so
        let mut sources = vec![&test.mlir_file, &test.test_cpp];
        let kernel_refs: Vec<&PathBuf> = test.kernel_sources.iter().collect();
        sources.extend(kernel_refs);

        for source in sources {
            let src_modified = source.metadata().ok()?.modified().ok()?;
            if !is_cache_valid(ps_modified, src_modified, self.toolchain_mtime) {
                return None; // Source or toolchain is newer, rebuild needed
            }
        }

        log::info!("Using cached unit test build in {}", output_dir.display());
        Some(crate::testing::unit_test::UnitTestBuildResult {
            prj_dir,
            build_log: "(cached)".to_string(),
        })
    }

    /// Apply standard environment variables to a Command.
    ///
    /// Sets PYTHONPATH, PEANO_INSTALL_DIR, PATH, XILINX_VITIS_AIETOOLS,
    /// and a cleaned LD_LIBRARY_PATH (with aietools lib dirs stripped).
    fn apply_env(&self, cmd: &mut Command) {
        cmd.env("PYTHONPATH", &self.pythonpath);
        cmd.env("PEANO_INSTALL_DIR", &self.peano_dir);

        // MLIR_AIE_DIR: makefile-common uses this for kernel include paths
        // (-I ${MLIR_AIE_DIR}/include). The activate script sets it to the
        // repo root, but aie_api/ headers only exist under install/include/.
        // mlir_aie_bin is install/bin or build/bin; parent is the correct root.
        if let Some(mlir_aie_dir) = self.mlir_aie_bin.parent() {
            cmd.env("MLIR_AIE_DIR", mlir_aie_dir);
        }

        // PATH: prepend mlir-aie bin, peano bin, and aietools bin
        let mut path = self.mlir_aie_bin.to_string_lossy().to_string();
        let peano_bin = self.peano_dir.join("bin");
        if peano_bin.exists() {
            path.push(':');
            path.push_str(&peano_bin.to_string_lossy());
        }
        if let Some(ref aietools) = self.aietools_root {
            let aietools_bin = aietools.join("bin");
            if aietools_bin.exists() {
                path.push(':');
                path.push_str(&aietools_bin.to_string_lossy());
            }
        }
        if let Ok(system_path) = std::env::var("PATH") {
            path.push(':');
            path.push_str(&system_path);
        }
        cmd.env("PATH", &path);

        // aietools environment
        if let Some(ref aietools) = self.aietools_root {
            cmd.env("XILINX_VITIS_AIETOOLS", aietools);
            cmd.env("XILINX_VITIS", aietools);
        }

        // Clean LD_LIBRARY_PATH: strip aietools lib entries
        let aietools_root_str = self.aietools_root.as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        let clean_ld_path: String = std::env::var("LD_LIBRARY_PATH")
            .unwrap_or_default()
            .split(':')
            .filter(|entry| {
                if aietools_root_str.is_empty() {
                    return true;
                }
                !entry.starts_with(&aietools_root_str)
            })
            .collect::<Vec<_>>()
            .join(":");
        cmd.env("LD_LIBRARY_PATH", &clean_ld_path);
    }

    /// Find Peano compiler installation.
    ///
    /// Search order:
    /// 1. `PEANO_INSTALL_DIR` environment variable
    /// 2. `../llvm-aie/install` (sibling directory)
    /// 3. ironenv pip package fallback
    fn find_peano(mlir_aie: &Path) -> Result<PathBuf, String> {
        // Env var first
        if let Ok(dir) = std::env::var("PEANO_INSTALL_DIR") {
            let p = PathBuf::from(&dir);
            if p.exists() {
                return p.canonicalize()
                    .map_err(|e| format!("Failed to canonicalize PEANO_INSTALL_DIR: {}", e));
            }
        }

        // Sibling llvm-aie/install
        if let Some(parent) = mlir_aie.parent() {
            let llvm_aie_install = parent.join("llvm-aie/install");
            if llvm_aie_install.join("bin").exists() {
                return llvm_aie_install.canonicalize()
                    .map_err(|e| format!("Failed to canonicalize llvm-aie path: {}", e));
            }
        }

        // ironenv pip package fallback
        let ironenv_peano = mlir_aie.join("ironenv/lib/python3.13/site-packages/llvm-aie");
        if ironenv_peano.exists() {
            return ironenv_peano.canonicalize()
                .map_err(|e| format!("Failed to canonicalize ironenv peano: {}", e));
        }

        Err(
            "Peano compiler not found. Set PEANO_INSTALL_DIR or build llvm-aie".to_string()
        )
    }

    /// Return the first path that exists from a list of candidates.
    fn find_first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
        candidates.iter().find(|p| p.exists()).cloned()
    }
}

/// Expand LLVM lit-style substitutions in a `// RUN:` line command.
///
/// Mirrors how LLVM's `lit` test runner expands variables in RUN directives.
/// The substitutions transform test-relative paths into absolute paths
/// suitable for execution in the build output directory.
///
/// Expansion order: `%S` before `%s` to prevent the shorter variable from
/// partially matching the longer one's prefix in output text. The `| FileCheck`
/// pipeline suffix is stripped since we don't run FileCheck.
///
/// # Substitutions
///
/// | Variable | Expansion |
/// |----------|-----------|
/// | `%PYTHON`, `%python` | Python interpreter path |
/// | `aiecc.py` | Full path to aiecc.py (Python needs file path, not PATH) |
/// | `%S` | Source directory (if provided) or build output directory |
/// | `%s` | MLIR filename (resolved via cwd) |
/// | `%test_lib_flags` | `-I<include> -L<lib> -ltest_lib` |
/// | `%aietools` | aietools root path (for `-I %aietools/include`) |
fn expand_lit_subs(
    cmd: &str,
    python: &Path,
    aiecc: &Path,
    output_dir: &Path,
    mlir_filename: &str,
    test_lib_include: Option<&Path>,
    test_lib_lib: Option<&Path>,
    source_dir: Option<&Path>,
    aietools_root: Option<&Path>,
) -> String {
    let mut expanded = cmd.to_string();

    // Strip FileCheck pipeline (we don't run FileCheck)
    if let Some(pos) = expanded.find("| FileCheck") {
        expanded.truncate(pos);
    }

    // %PYTHON and %python -> Python interpreter path
    expanded = expanded.replace("%PYTHON", &python.to_string_lossy());
    expanded = expanded.replace("%python", &python.to_string_lossy());

    // aiecc.py -> full path (Python needs a file path, not a PATH lookup)
    expanded = expanded.replace("aiecc.py", &aiecc.to_string_lossy());

    // %S -> source directory (for NPU tests) or output directory (for unit tests)
    // MUST expand before %s to avoid partial matches
    let s_dir = source_dir.unwrap_or(output_dir);
    expanded = expanded.replace("%S", &s_dir.to_string_lossy());

    // %s -> MLIR filename (resolved via cwd = output_dir)
    expanded = expanded.replace("%s", mlir_filename);

    // %test_lib_flags -> compiler/linker flags for test_library.h + libtest_lib.a
    let test_lib_flags = match (test_lib_include, test_lib_lib) {
        (Some(inc), Some(lib)) => {
            format!("-I{} -L{} -ltest_lib", inc.display(), lib.display())
        }
        _ => String::new(),
    };
    expanded = expanded.replace("%test_lib_flags", &test_lib_flags);

    // %aietools -> aietools root path (for -I %aietools/include in xchesscc lines)
    if let Some(aietools) = aietools_root {
        expanded = expanded.replace("%aietools", &aietools.to_string_lossy());
    }

    expanded.trim().to_string()
}

/// Apply compiler override to a build command.
///
/// For aiecc.py commands, ensures the correct compiler and linker flags are
/// present regardless of what the original RUN line specified.
///
/// For xchesscc_wrapper commands in Peano mode, rewrites to use Peano's clang
/// so kernel objects don't contain Chess-specific ELF sections (.tctmemtab,
/// .rtstab, etc.) that ld.lld rejects with --orphan-handling=error.
///
/// This operates on raw RUN line text (before lit expansion), so `aiecc.py`
/// appears as a literal string that's easy to match.
pub fn apply_compiler_override(cmd: &str, use_chess: bool, peano_clang: Option<&Path>) -> String {
    // Rewrite xchesscc_wrapper kernel compilation to Peano clang in Peano mode.
    // Pattern: xchesscc_wrapper <arch> [-I <path>]... -c <source>.cc -o <output>.o
    if !use_chess && cmd.contains("xchesscc_wrapper") {
        if let Some(clang) = peano_clang {
            return rewrite_xchesscc_to_peano(cmd, clang);
        }
    }

    if !cmd.contains("aiecc.py") {
        return cmd.to_string();
    }

    let mut result = cmd.to_string();

    // Strip existing compiler/linker flags (order: --no- before -- to avoid
    // partial matches, e.g. stripping "--xchesscc" from "--no-xchesscc")
    result = result.replace(" --no-xchesscc", "");
    result = result.replace(" --no-xbridge", "");
    result = result.replace(" --xchesscc", "");
    result = result.replace(" --xbridge", "");

    // Insert desired flags after aiecc.py
    let flags = if use_chess {
        " --xchesscc --xbridge"
    } else {
        " --no-xchesscc --no-xbridge"
    };
    result = result.replacen("aiecc.py", &format!("aiecc.py{}", flags), 1);

    // Clean up any double spaces introduced by stripping
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }

    result
}

/// Rewrite a `xchesscc_wrapper` command to use Peano's clang.
///
/// Input:  `xchesscc_wrapper aie2 -I /path/include -c source.cc -o output.o`
/// Output: `<clang> --target=aie2-none-unknown-elf -I /path/include -c source.cc -o output.o`
///
/// Extracts the architecture target from the first argument after xchesscc_wrapper,
/// preserves -I, -c, -o flags and their arguments, and drops Chess-specific flags
/// like +w, -d, -f, +l that don't apply to clang.
fn rewrite_xchesscc_to_peano(cmd: &str, clang: &Path) -> String {
    let parts: Vec<&str> = cmd.split_whitespace().collect();

    // Find xchesscc_wrapper position
    let wrapper_idx = match parts.iter().position(|p| p.contains("xchesscc_wrapper")) {
        Some(i) => i,
        None => return cmd.to_string(),
    };

    // Next token is architecture (aie2, aie2p, etc.)
    let arch = parts.get(wrapper_idx + 1).copied().unwrap_or("aie2");
    let target = format!("--target={}-none-unknown-elf", arch);

    let mut result = vec![clang.to_string_lossy().to_string(), target];

    // Walk remaining args, keeping -I, -c, -o and their values,
    // dropping Chess-specific flags (+w, -d, -f, +l, +a)
    let mut i = wrapper_idx + 2;
    while i < parts.len() {
        let arg = parts[i];
        match arg {
            // Flags with a following value to preserve
            "-I" | "-c" | "-o" => {
                result.push(arg.to_string());
                if let Some(&val) = parts.get(i + 1) {
                    result.push(val.to_string());
                    i += 1;
                }
            }
            // Chess-specific flags with a following value to skip
            "+w" | "+l" | "+a" => {
                i += 1; // skip the value too
            }
            // Chess-specific standalone flags to skip
            "-d" | "-f" => {}
            // Keep everything else (source files, -DFOO, etc.)
            _ if arg.starts_with('-') || arg.starts_with('+') => {
                // Unknown flag -- skip Chess-specific + flags, keep - flags
                if arg.starts_with('+') {
                    // +flags are Chess-specific
                } else {
                    result.push(arg.to_string());
                }
            }
            _ => {
                // Bare argument (source file, object file, etc.)
                result.push(arg.to_string());
            }
        }
        i += 1;
    }

    result.join(" ")
}

/// Required environment variables for Chess builds (legacy check).
const REQUIRED_ENV: &[(&str, &str)] = &[
    ("PEANO_INSTALL_DIR", "Peano compiler (needed by mlir-aie build system)"),
    ("XILINX_VITIS_AIETOOLS", "AMD aietools installation"),
];

/// Check that all required environment variables are set for Chess builds.
///
/// Returns a list of (var_name, description) pairs for missing variables.
pub fn check_environment() -> Vec<(&'static str, &'static str)> {
    REQUIRED_ENV
        .iter()
        .filter(|(var, _)| std::env::var(var).is_err())
        .copied()
        .collect()
}

/// Build a test with Chess, producing a xclbin alongside the Peano version.
///
/// This is a convenience wrapper around `BuildEnv::build()` for callers
/// that already have an `AieTools` reference and want the legacy signature.
///
/// For new code, prefer using `BuildEnv::discover()` + `BuildEnv::build()`
/// directly.
pub fn build_with_chess(
    _tools: &AieTools,
    build_env: &BuildEnv,
    mlir_source: &Path,
    output_dir: &Path,
    device: &str,
    gen_sim: bool,
) -> Result<BuildResult, String> {
    build_env.build(mlir_source, output_dir, &BuildOpts {
        use_chess: true,
        gen_sim,
        device: device.to_string(),
        nice: None,
        force_rebuild: false,
    })
}

/// Find ALL xclbin files in a directory, each paired with its insts file.
///
/// For multi-xclbin tests (like matrix_multiplication_using_cascade), the
/// build steps produce multiple `--xclbin-name=X --npu-insts-name=Y` pairs.
/// This function parses those flags from the build steps to match each xclbin
/// with its correct insts file. Falls back to filesystem heuristics.
///
/// Returns (xclbin_path, optional_insts_path, variant_name) tuples sorted
/// alphabetically by xclbin filename. The variant_name is the xclbin stem
/// (e.g., "aie2_plain" for "aie2_plain.xclbin").
pub fn find_all_xclbin_results(
    output_dir: &Path,
    build_steps: &[String],
) -> Vec<(PathBuf, Option<PathBuf>, String)> {
    // Parse xclbin -> insts mapping from build step flags
    let mut xclbin_to_insts: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for step in build_steps {
        if let (Some(xname), Some(iname)) = (
            extract_flag_value(step, "--xclbin-name="),
            extract_flag_value(step, "--npu-insts-name="),
        ) {
            xclbin_to_insts.insert(xname, iname);
        }
    }

    let xclbins = artifacts::collect_xclbins(output_dir);

    let mut results = Vec::new();
    for xclbin in &xclbins {
        let xclbin_name = xclbin.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let variant = xclbin.file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Try to find matching insts file
        let insts = if let Some(insts_name) = xclbin_to_insts.get(&xclbin_name) {
            // Known pairing from build steps
            let insts_path = output_dir.join(insts_name);
            if insts_path.exists() {
                Some(insts_path)
            } else {
                // Try .bin extension variant
                let bin_variant = insts_name
                    .replace(".txt", ".bin");
                let bin_path = output_dir.join(&bin_variant);
                if bin_path.exists() { Some(bin_path) } else { None }
            }
        } else {
            // No parsed mapping -- fall back to standard insts.bin
            artifacts::find_insts(output_dir)
        };

        results.push((xclbin.clone(), insts, variant));
    }

    results
}

/// Extract a flag value from a command line string.
/// For example, extract_flag_value("aiecc.py --xclbin-name=foo.xclbin", "--xclbin-name=")
/// returns Some("foo.xclbin").
fn extract_flag_value(cmd: &str, flag: &str) -> Option<String> {
    let idx = cmd.find(flag)?;
    let start = idx + flag.len();
    let rest = &cmd[start..];
    // Value ends at whitespace or end of string
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_environment_reports_missing() {
        // This test will report missing vars if the env isn't set up,
        // which is the expected behavior in a CI/test environment.
        let missing = check_environment();
        for (var, desc) in &missing {
            assert!(!var.is_empty());
            assert!(!desc.is_empty());
        }
    }

    // Basic artifact helper tests moved to testing::artifacts module.
    // These remain as integration-level checks that the imports work.
    #[test]
    fn test_artifact_helpers_accessible() {
        assert!(artifacts::collect_xclbins(Path::new("/nonexistent")).is_empty());
        assert!(artifacts::find_prj_dir(Path::new("/nonexistent")).is_none());
        assert!(artifacts::find_insts(Path::new("/nonexistent")).is_none());
    }

    #[test]
    fn test_build_opts_defaults() {
        let opts = BuildOpts {
            use_chess: false,
            gen_sim: false,
            device: "npu1_1col".to_string(),
            nice: None,
            force_rebuild: false,
        };
        assert!(!opts.use_chess);
        assert!(!opts.gen_sim);
        assert_eq!(opts.device, "npu1_1col");
        assert!(!opts.force_rebuild);
        assert!(opts.nice.is_none());
    }

    #[test]
    fn test_find_first_existing_none() {
        let result = BuildEnv::find_first_existing(&[
            PathBuf::from("/nonexistent/a"),
            PathBuf::from("/nonexistent/b"),
        ]);
        assert!(result.is_none());
    }

    #[test]
    fn test_expand_lit_subs_basic() {
        let result = expand_lit_subs(
            "%PYTHON aiecc.py --aiesim --xchesscc %s %test_lib_flags %S/test.cpp",
            Path::new("/venv/bin/python3"),
            Path::new("/install/bin/aiecc.py"),
            Path::new("/build/output"),
            "aie.mlir",
            Some(Path::new("/install/include")),
            Some(Path::new("/install/lib")),
            None,
            None,
        );
        assert_eq!(
            result,
            "/venv/bin/python3 /install/bin/aiecc.py --aiesim --xchesscc \
             aie.mlir -I/install/include -L/install/lib -ltest_lib /build/output/test.cpp"
        );
    }

    #[test]
    fn test_expand_lit_subs_filecheck_stripped() {
        let result = expand_lit_subs(
            "aie.mlir.prj/aiesim.sh | FileCheck %s",
            Path::new("/venv/bin/python3"),
            Path::new("/install/bin/aiecc.py"),
            Path::new("/build/output"),
            "aie.mlir",
            None,
            None,
            None,
            None,
        );
        assert_eq!(result, "aie.mlir.prj/aiesim.sh");
    }

    #[test]
    fn test_expand_lit_subs_kernel_compile() {
        let result = expand_lit_subs(
            "xchesscc_wrapper aie2 -c %S/kernel.cc",
            Path::new("/venv/bin/python3"),
            Path::new("/install/bin/aiecc.py"),
            Path::new("/build/test01"),
            "aie.mlir",
            None,
            None,
            None,
            None,
        );
        assert_eq!(result, "xchesscc_wrapper aie2 -c /build/test01/kernel.cc");
    }

    #[test]
    fn test_expand_lit_subs_bcf_step() {
        let result = expand_lit_subs(
            "xchesscc_wrapper aie2 +l aie.mlir.prj/main_core_1_3.bcf %S/kernel.cc -o custom_1_3.elf",
            Path::new("/venv/bin/python3"),
            Path::new("/install/bin/aiecc.py"),
            Path::new("/build/test02"),
            "aie.mlir",
            None,
            None,
            None,
            None,
        );
        assert_eq!(
            result,
            "xchesscc_wrapper aie2 +l aie.mlir.prj/main_core_1_3.bcf /build/test02/kernel.cc -o custom_1_3.elf"
        );
    }

    #[test]
    fn test_expand_lit_subs_no_test_lib() {
        let result = expand_lit_subs(
            "%PYTHON aiecc.py %s %test_lib_flags",
            Path::new("/python"),
            Path::new("/aiecc.py"),
            Path::new("/out"),
            "test.mlir",
            None,
            None,
            None,
            None,
        );
        // %test_lib_flags expands to empty string when paths are None
        assert_eq!(result, "/python /aiecc.py test.mlir");
    }

    #[test]
    fn test_expand_lit_subs_source_dir() {
        // When source_dir is provided, %S should point to it (not output_dir)
        let result = expand_lit_subs(
            "cp %S/aie.mlir aie_arch.mlir",
            Path::new("/python"),
            Path::new("/aiecc.py"),
            Path::new("/build/output"),
            "",
            None,
            None,
            Some(Path::new("/src/test/npu-xrt/add_one")),
            None,
        );
        assert_eq!(result, "cp /src/test/npu-xrt/add_one/aie.mlir aie_arch.mlir");
    }

    #[test]
    fn test_expand_lit_subs_aietools() {
        let result = expand_lit_subs(
            "xchesscc_wrapper aie2 -I %aietools/include -c %S/kernel.cc",
            Path::new("/python"),
            Path::new("/aiecc.py"),
            Path::new("/build"),
            "",
            None,
            None,
            Some(Path::new("/src/test")),
            Some(Path::new("/opt/aietools")),
        );
        assert_eq!(
            result,
            "xchesscc_wrapper aie2 -I /opt/aietools/include -c /src/test/kernel.cc"
        );
    }

    #[test]
    fn test_expand_lit_subs_lowercase_python() {
        let result = expand_lit_subs(
            "%python %S/aie2.py > ./aie2.mlir",
            Path::new("/venv/bin/python3"),
            Path::new("/aiecc.py"),
            Path::new("/build"),
            "",
            None,
            None,
            Some(Path::new("/src")),
            None,
        );
        assert_eq!(result, "/venv/bin/python3 /src/aie2.py > ./aie2.mlir");
    }

    #[test]
    fn test_compiler_override_peano() {
        let result = apply_compiler_override(
            "%python aiecc.py --xchesscc --xbridge --no-aiesim %s",
            false,
            None,
        );
        assert!(result.contains("--no-xchesscc"));
        assert!(result.contains("--no-xbridge"));
        assert!(!result.contains(" --xchesscc"));
        assert!(!result.contains(" --xbridge "));
    }

    #[test]
    fn test_compiler_override_chess() {
        let result = apply_compiler_override(
            "%python aiecc.py --no-xchesscc --no-xbridge --no-aiesim %s",
            true,
            None,
        );
        assert!(result.contains("--xchesscc"));
        assert!(result.contains("--xbridge"));
        assert!(!result.contains("--no-xchesscc"));
        assert!(!result.contains("--no-xbridge"));
    }

    #[test]
    fn test_compiler_override_adds_missing_flags() {
        // aiecc.py line with no compiler flags should get them added
        let result = apply_compiler_override(
            "%python aiecc.py --no-aiesim %s",
            false,
            None,
        );
        assert!(result.contains("--no-xchesscc"));
        assert!(result.contains("--no-xbridge"));
    }

    #[test]
    fn test_compiler_override_non_aiecc_passthrough() {
        // Non-aiecc.py commands pass through when no peano_clang provided
        let cmd = "xchesscc_wrapper aie2 -c %S/kernel.cc";
        let result = apply_compiler_override(cmd, false, None);
        assert_eq!(result, cmd);
    }

    #[test]
    fn test_compiler_override_no_double_spaces() {
        let result = apply_compiler_override(
            "%python aiecc.py --xchesscc --xbridge %s",
            false,
            None,
        );
        assert!(!result.contains("  "));
    }

    #[test]
    fn test_compiler_override_xchesscc_to_peano() {
        // xchesscc_wrapper lines should be rewritten to peano clang in Peano mode
        let clang = Path::new("/peano/bin/clang");
        let result = apply_compiler_override(
            "xchesscc_wrapper aie2 -I %aietools/include -c %S/vector_scalar_mul.cc -o vector_scalar_mul.o",
            false,
            Some(clang),
        );
        assert!(result.starts_with("/peano/bin/clang"));
        assert!(result.contains("--target=aie2-none-unknown-elf"));
        assert!(result.contains("-I %aietools/include"));
        assert!(result.contains("-c %S/vector_scalar_mul.cc"));
        assert!(result.contains("-o vector_scalar_mul.o"));
        assert!(!result.contains("xchesscc_wrapper"));
    }

    #[test]
    fn test_compiler_override_xchesscc_chess_mode_passthrough() {
        // xchesscc_wrapper lines should pass through in Chess mode
        let clang = Path::new("/peano/bin/clang");
        let cmd = "xchesscc_wrapper aie2 -I %aietools/include -c kernel.cc -o kernel.o";
        let result = apply_compiler_override(cmd, true, Some(clang));
        assert_eq!(result, cmd);
    }

    #[test]
    fn test_compiler_override_xchesscc_drops_chess_flags() {
        // Chess-specific flags (+w, -d, -f, +l) should be stripped
        let clang = Path::new("/peano/bin/clang");
        let result = apply_compiler_override(
            "xchesscc_wrapper aie2 +w /work -d -f input.o kernel.o +l core.bcf -o core.elf",
            false,
            Some(clang),
        );
        assert!(!result.contains("+w"));
        assert!(!result.contains("/work"));
        assert!(!result.contains("+l"));
        assert!(!result.contains("core.bcf"));
        // -d and -f are standalone Chess flags
        assert!(result.contains("-o core.elf"));
        assert!(result.contains("input.o"));
        assert!(result.contains("kernel.o"));
    }

    #[test]
    fn test_extract_flag_value() {
        assert_eq!(
            extract_flag_value(
                "aiecc.py --xclbin-name=aie2_plain.xclbin --npu-insts-name=insts2_plain.txt foo.mlir",
                "--xclbin-name="
            ),
            Some("aie2_plain.xclbin".to_string())
        );
        assert_eq!(
            extract_flag_value(
                "aiecc.py --xclbin-name=aie2_plain.xclbin --npu-insts-name=insts2_plain.txt foo.mlir",
                "--npu-insts-name="
            ),
            Some("insts2_plain.txt".to_string())
        );
        assert_eq!(
            extract_flag_value("aiecc.py --no-aiesim foo.mlir", "--xclbin-name="),
            None
        );
    }

    #[test]
    fn test_find_all_xclbin_results_single() {
        let dir = std::env::temp_dir().join("xdna_test_xclbin_single");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Create one xclbin and one insts file
        std::fs::write(dir.join("aie.xclbin"), b"xclbin").unwrap();
        std::fs::write(dir.join("insts.bin"), b"insts").unwrap();

        let build_steps = vec![
            "aiecc.py --xclbin-name=aie.xclbin --npu-insts-name=insts.bin aie.mlir".to_string(),
        ];

        let results = find_all_xclbin_results(&dir, &build_steps);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2, "aie");
        assert!(results[0].1.is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_find_all_xclbin_results_multi() {
        let dir = std::env::temp_dir().join("xdna_test_xclbin_multi");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Create multiple xclbin + insts files (like matrix_mult_cascade)
        std::fs::write(dir.join("aie2_buffer.xclbin"), b"xclbin").unwrap();
        std::fs::write(dir.join("aie2_cascade.xclbin"), b"xclbin").unwrap();
        std::fs::write(dir.join("aie2_plain.xclbin"), b"xclbin").unwrap();
        std::fs::write(dir.join("insts2_buffer.txt"), b"insts").unwrap();
        std::fs::write(dir.join("insts2_cascade.txt"), b"insts").unwrap();
        std::fs::write(dir.join("insts2_plain.txt"), b"insts").unwrap();

        let build_steps = vec![
            "aiecc.py --xclbin-name=aie2_plain.xclbin --npu-insts-name=insts2_plain.txt aie_plainx4.mlir".to_string(),
            "aiecc.py --xclbin-name=aie2_buffer.xclbin --npu-insts-name=insts2_buffer.txt aie_bufferx4.mlir".to_string(),
            "aiecc.py --xclbin-name=aie2_cascade.xclbin --npu-insts-name=insts2_cascade.txt aie_cascadex4.mlir".to_string(),
        ];

        let results = find_all_xclbin_results(&dir, &build_steps);
        assert_eq!(results.len(), 3);

        // Sorted alphabetically: buffer, cascade, plain
        assert_eq!(results[0].2, "aie2_buffer");
        assert_eq!(results[1].2, "aie2_cascade");
        assert_eq!(results[2].2, "aie2_plain");

        // Each should have its matching insts file
        assert!(results[0].1.as_ref().unwrap().ends_with("insts2_buffer.txt"));
        assert!(results[1].1.as_ref().unwrap().ends_with("insts2_cascade.txt"));
        assert!(results[2].1.as_ref().unwrap().ends_with("insts2_plain.txt"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_toolchain_mtime_detects_stale_cache() {
        // Simulate: xclbin is newer than source, but toolchain was rebuilt
        // after the xclbin was produced. Cache should be invalidated.
        use std::time::Duration;

        let dir = std::env::temp_dir().join("xdna_test_toolchain_cache");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let xclbin_path = dir.join("aie.xclbin");
        let source_path = dir.join("aie.mlir");

        // Create source first, then xclbin (xclbin is "newer" than source)
        std::fs::write(&source_path, b"source").unwrap();
        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(&xclbin_path, b"xclbin").unwrap();

        let xclbin_mtime = xclbin_path.metadata().unwrap().modified().unwrap();
        let source_mtime = source_path.metadata().unwrap().modified().unwrap();

        // Without toolchain check: cache is valid (xclbin > source)
        assert!(xclbin_mtime > source_mtime);

        // With toolchain rebuilt AFTER the xclbin: cache should be invalid
        let future_toolchain = xclbin_mtime + Duration::from_secs(100);
        assert!(is_cache_valid(xclbin_mtime, source_mtime, Some(future_toolchain)) == false);

        // With toolchain older than xclbin: cache is still valid
        let old_toolchain = source_mtime - Duration::from_secs(100);
        assert!(is_cache_valid(xclbin_mtime, source_mtime, Some(old_toolchain)) == true);

        // With no toolchain mtime (legacy behavior): cache is valid
        assert!(is_cache_valid(xclbin_mtime, source_mtime, None) == true);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_toolchain_mtime_source_newer_always_invalidates() {
        // Even with old toolchain, if source is newer than xclbin, cache is invalid.
        use std::time::{Duration, SystemTime};

        let now = SystemTime::now();
        let xclbin_mtime = now - Duration::from_secs(200);
        let source_mtime = now - Duration::from_secs(100);  // source newer than xclbin
        let toolchain_mtime = now - Duration::from_secs(300);  // toolchain older than both

        assert!(is_cache_valid(xclbin_mtime, source_mtime, Some(toolchain_mtime)) == false);
    }
}
