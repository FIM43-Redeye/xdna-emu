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
use crate::config::Config;

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

        Ok(BuildEnv {
            python,
            aiecc,
            pythonpath,
            peano_dir,
            mlir_aie_bin,
            aietools_root,
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
        if let Some(result) = self.check_cache(mlir_source, output_dir, &xclbin_path) {
            return Ok(result);
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
        cmd.env("PYTHONPATH", &self.pythonpath);
        cmd.env("PEANO_INSTALL_DIR", &self.peano_dir);

        // PATH: prepend mlir-aie bin and peano bin
        let mut path = self.mlir_aie_bin.to_string_lossy().to_string();
        let peano_bin = self.peano_dir.join("bin");
        if peano_bin.exists() {
            path.push(':');
            path.push_str(&peano_bin.to_string_lossy());
        }
        if let Ok(system_path) = std::env::var("PATH") {
            path.push(':');
            path.push_str(&system_path);
        }
        cmd.env("PATH", &path);

        // aietools environment (if available)
        //
        // Set XILINX_VITIS_AIETOOLS so aiecc.py can find xchesscc.
        // CRITICAL: Strip aietools lib directories from LD_LIBRARY_PATH.
        // aietools ships its own old libstdc++ that shadows the system
        // version and breaks Python's native MLIR bindings (nanobind).
        // activate-npu-env.sh may have already added these to the env.
        // The xchesscc subprocess manages its own library paths internally,
        // so removing them from the aiecc.py process is safe.
        if let Some(ref aietools) = self.aietools_root {
            cmd.env("XILINX_VITIS_AIETOOLS", aietools);
            cmd.env("XILINX_VITIS", aietools);
        }

        // Clean LD_LIBRARY_PATH: remove any aietools lib entries that
        // would shadow the system libstdc++ and break Python's MLIR bindings.
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
            } else if let Some(found) = find_xclbin_in(output_dir) {
                log::warn!(
                    "{} build exited with {} but found {} (partial success)",
                    compiler_label, output.status.code().unwrap_or(-1), found.display()
                );
                return Ok(BuildResult {
                    xclbin: found,
                    insts: find_insts_in(output_dir),
                    prj_dir: find_prj_dir(output_dir),
                    build_log,
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
            if let Some(found) = find_xclbin_in(output_dir) {
                return Ok(BuildResult {
                    xclbin: found,
                    insts: find_insts_in(output_dir),
                    prj_dir: find_prj_dir(output_dir),
                    build_log,
                });
            }
            return Err(format!(
                "{} build succeeded but no xclbin found in {}",
                compiler_label, output_dir.display()
            ));
        }

        Ok(BuildResult {
            xclbin: xclbin_path,
            insts: find_insts_in(output_dir),
            prj_dir: find_prj_dir(output_dir),
            build_log,
        })
    }

    /// Human-readable summary of the discovered environment.
    pub fn summary(&self) -> String {
        let aietools_label = match &self.aietools_root {
            Some(p) => format!("aietools at {}", p.display()),
            None => "no aietools".to_string(),
        };
        format!(
            "python={}, aiecc={}, peano={}, {}",
            self.python.display(),
            self.aiecc.file_name().unwrap_or_default().to_string_lossy(),
            self.peano_dir.display(),
            aietools_label,
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

        if xclbin_modified > source_modified {
            log::info!("Using cached build in {}", output_dir.display());
            Some(BuildResult {
                xclbin: xclbin_path.to_path_buf(),
                insts: find_insts_in(output_dir),
                prj_dir: find_prj_dir(output_dir),
                build_log: "(cached)".to_string(),
            })
        } else {
            None
        }
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
    })
}

/// Find the first .xclbin file in a directory.
fn find_xclbin_in(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "xclbin") {
            return Some(path);
        }
    }
    None
}

/// Find the first insts.bin file in a directory.
fn find_insts_in(dir: &Path) -> Option<PathBuf> {
    let path = dir.join("insts.bin");
    path.exists().then_some(path)
}

/// Find a .prj directory (created by aiecc.py for aiesimulator).
fn find_prj_dir(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.extension().map_or(false, |ext| ext == "prj") {
            return Some(path);
        }
    }
    None
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

    #[test]
    fn test_find_xclbin_in_empty_dir() {
        assert!(find_xclbin_in(Path::new("/nonexistent")).is_none());
    }

    #[test]
    fn test_find_prj_dir_nonexistent() {
        assert!(find_prj_dir(Path::new("/nonexistent")).is_none());
    }

    #[test]
    fn test_find_insts_nonexistent() {
        assert!(find_insts_in(Path::new("/nonexistent")).is_none());
    }

    #[test]
    fn test_build_opts_defaults() {
        let opts = BuildOpts {
            use_chess: false,
            gen_sim: false,
            device: "npu1_1col".to_string(),
            nice: None,
        };
        assert!(!opts.use_chess);
        assert!(!opts.gen_sim);
        assert_eq!(opts.device, "npu1_1col");
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
}
