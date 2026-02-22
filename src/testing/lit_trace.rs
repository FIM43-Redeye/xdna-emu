//! Trace pipeline orchestration for mlir-aie npu-xrt tests.
//!
//! Chains three stages for each test:
//! 1. **Inject** -- `trace-inject.py` adds packet flow + trace register writes
//! 2. **Compile** -- `aiecc.py` compiles the trace-enabled MLIR to xclbin
//! 3. **Execute** -- `trace-run.py` runs on the NPU and collects trace data
//!
//! Staleness checking via SHA256 skips inject+compile when sources haven't
//! changed, making re-runs near-instant (only the execute stage runs).
//!
//! All paths stored in [`TraceConfig`] must be absolute.  The
//! [`TraceConfig::resolve`] helper canonicalizes everything on construction
//! so the pipeline stages never depend on the process's working directory.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use crate::testing::process_control::{spawn_with_timeout, ProcessOutcome};
use crate::testing::sanitized_ld_library_path;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Outcome of a trace pipeline run for a single test.
pub enum TraceOutcome {
    /// Trace collected successfully.
    Success {
        trace_json: PathBuf,
        output_dir: PathBuf,
    },
    /// Injection stage failed (trace-inject.py error).
    InjectFailed { stderr: String },
    /// Compilation stage failed (aiecc.py error).
    CompileFailed { stderr: String },
    /// Execution stage failed (trace-run.py or NPU error).
    RunFailed { stderr: String },
    /// Test was skipped (blocklisted or already traced).
    Skipped { reason: String },
    /// Test timed out during one of the stages.
    Timeout { stage: String },
    /// Device is wedged (D-state after SIGKILL).
    Wedged,
}

/// Configuration for the trace pipeline.
///
/// All paths **must** be absolute.  Use [`TraceConfig::resolve`] to
/// canonicalize a config built from potentially-relative inputs.
pub struct TraceConfig {
    /// Root of the xdna-emu project (contains tools/).
    pub xdna_emu_root: PathBuf,
    /// mlir-aie source tree root (contains test/npu-xrt/).
    pub mlir_aie_source: PathBuf,
    /// mlir-aie build directory.
    pub mlir_aie_build: PathBuf,
    /// Where to put trace-enabled builds (build/traced-tests/).
    pub build_traced_dir: PathBuf,
    /// Where to put collected traces (build/traces/).
    pub traces_dir: PathBuf,
    /// Trace buffer size in bytes.
    pub trace_size: usize,
    /// Absolute path to Python interpreter (from mlir-aie venv).
    pub python: PathBuf,
    /// Absolute path to aiecc.py compiler driver.
    pub aiecc: PathBuf,
    /// Timeout for inject stage (seconds).
    pub inject_timeout: u32,
    /// Timeout for compile stage (seconds).
    pub compile_timeout: u32,
    /// Timeout for execute stage (seconds).
    pub execute_timeout: u32,
}

impl TraceConfig {
    /// Make every path in the config absolute so the pipeline is
    /// independent of the process working directory.
    ///
    /// Uses [`fs::canonicalize`] for directories (resolving symlinks is
    /// fine for directories).  For executables like the Python interpreter,
    /// uses [`make_absolute`] which preserves symlinks -- critical because
    /// a venv `python3` is a symlink to the system binary, and Python
    /// relies on the symlink path to locate its `sys.prefix` and
    /// site-packages.
    pub fn resolve(mut self) -> Self {
        self.xdna_emu_root = abs_dir(self.xdna_emu_root);
        self.mlir_aie_source = abs_dir(self.mlir_aie_source);
        self.mlir_aie_build = abs_dir(self.mlir_aie_build);
        self.build_traced_dir = abs_dir(self.build_traced_dir);
        self.traces_dir = abs_dir(self.traces_dir);

        // Preserve symlinks for executables -- venv python3 is a symlink
        // and resolving it loses the venv prefix detection.
        self.python = make_absolute(self.python);
        self.aiecc = make_absolute(self.aiecc);

        self
    }
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            xdna_emu_root: PathBuf::from("."),
            mlir_aie_source: PathBuf::from("../mlir-aie"),
            mlir_aie_build: PathBuf::from("../mlir-aie/build"),
            build_traced_dir: PathBuf::from("build/traced-tests"),
            traces_dir: PathBuf::from("build/traces"),
            trace_size: 1_048_576, // 1 MB
            python: PathBuf::from("python3"),
            aiecc: PathBuf::from("aiecc.py"),
            inject_timeout: 120,
            compile_timeout: 600,
            execute_timeout: 120,
        }
    }
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Make a path absolute by resolving symlinks (via [`fs::canonicalize`]).
///
/// Falls back to joining with the current directory if the path does not
/// exist on disk (e.g. build output directories that will be created later).
fn abs_dir(p: PathBuf) -> PathBuf {
    fs::canonicalize(&p).unwrap_or_else(|_| {
        if p.is_absolute() {
            p
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("/"))
                .join(p)
        }
    })
}

/// Make a path absolute **without** resolving symlinks.
///
/// This is critical for venv executables: `ironenv/bin/python3` is a symlink
/// to the system Python binary, and Python uses the *symlink* path to locate
/// its `sys.prefix` (and therefore site-packages).  Using `canonicalize`
/// would resolve through to `/usr/bin/python3.13`, losing the venv.
fn make_absolute(p: PathBuf) -> PathBuf {
    let full = if p.is_absolute() {
        p
    } else {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        cwd.join(&p)
    };

    // Clean up `..` and `.` components lexically (without following symlinks).
    let mut result = PathBuf::new();
    for component in full.components() {
        match component {
            std::path::Component::ParentDir => {
                result.pop();
            }
            std::path::Component::CurDir => {}
            other => result.push(other),
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Subprocess environment
// ---------------------------------------------------------------------------

/// Pre-computed environment variables for child processes.
///
/// Built once from a [`TraceConfig`] and applied to every subprocess so
/// that Python scripts, aiecc.py, and the Peano compiler all find their
/// dependencies regardless of the runner's cwd.
struct SubprocessEnv {
    path: String,
    pythonpath: String,
    ld_library_path: String,
}

impl SubprocessEnv {
    /// Build the environment from a resolved [`TraceConfig`].
    fn from_config(config: &TraceConfig) -> Self {
        // -- PATH --
        // Prepend: venv bin, mlir-aie install/bin, XRT bin
        let mut path_dirs: Vec<String> = Vec::new();

        // Venv bin directory (where python3, aiecc.py, lit, etc. live)
        if let Some(parent) = config.python.parent() {
            push_if_dir(&mut path_dirs, parent);
        }

        // mlir-aie install/bin (aie-opt, aie-translate, etc.)
        push_if_dir(&mut path_dirs, &config.mlir_aie_source.join("install/bin"));
        push_if_dir(&mut path_dirs, &config.mlir_aie_source.join("my_install/bin"));

        // Peano compiler
        push_if_dir(&mut path_dirs, &config.mlir_aie_source.join("install/peano/bin"));

        // XRT
        push_if_dir(&mut path_dirs, &PathBuf::from("/opt/xilinx/xrt/bin"));

        // Existing PATH
        if let Ok(existing) = std::env::var("PATH") {
            path_dirs.push(existing);
        }

        // -- PYTHONPATH --
        let mut py_dirs: Vec<String> = Vec::new();

        push_if_dir(&mut py_dirs, &config.mlir_aie_build.join("python"));
        push_if_dir(&mut py_dirs, &config.mlir_aie_source.join("install/python"));
        push_if_dir(&mut py_dirs, &PathBuf::from("/opt/xilinx/xrt/python"));

        if let Ok(existing) = std::env::var("PYTHONPATH") {
            py_dirs.push(existing);
        }

        // -- LD_LIBRARY_PATH --
        // Filter aietools (outdated libstdc++) AND cargo-injected debug paths
        // (target/debug/deps, rustup toolchain libs) which are irrelevant
        // for Python subprocesses.
        let ld_library_path = {
            let sanitized = sanitized_ld_library_path();
            sanitized
                .split(':')
                .filter(|p| !p.is_empty())
                .filter(|p| !p.contains("/target/debug"))
                .filter(|p| !p.contains("/target/release"))
                .filter(|p| !p.contains(".rustup"))
                .collect::<Vec<_>>()
                .join(":")
        };

        SubprocessEnv {
            path: path_dirs.join(":"),
            pythonpath: py_dirs.join(":"),
            ld_library_path,
        }
    }

    /// Apply this environment to a [`Command`].
    fn apply(&self, cmd: &mut Command) {
        cmd.env("PATH", &self.path)
            .env("PYTHONPATH", &self.pythonpath)
            .env("LD_LIBRARY_PATH", &self.ld_library_path);
    }
}

/// Push a directory path into the list if it exists on disk.
fn push_if_dir(dirs: &mut Vec<String>, path: &Path) {
    if path.is_dir() {
        dirs.push(path.to_string_lossy().into_owned());
    }
}

// ---------------------------------------------------------------------------
// Staleness checking
// ---------------------------------------------------------------------------

/// Check whether a test's traced build is up-to-date.
///
/// Compares SHA256 of the upstream source and the inject tool against
/// cached hashes in the test build directory.  If both match and the
/// xclbin exists, returns true (skip inject + compile).
fn is_up_to_date(test_build_dir: &Path, upstream_dir: &Path, inject_tool: &Path) -> bool {
    let xclbin = test_build_dir.join("aie.xclbin");
    if !xclbin.exists() {
        return false;
    }

    let source_hash_file = test_build_dir.join(".source-hash");
    let inject_hash_file = test_build_dir.join(".inject-hash");

    let current_source_hash = hash_directory_sources(upstream_dir);
    let current_inject_hash = hash_file(inject_tool);

    let cached_source = fs::read_to_string(&source_hash_file).unwrap_or_default();
    let cached_inject = fs::read_to_string(&inject_hash_file).unwrap_or_default();

    cached_source.trim() == current_source_hash && cached_inject.trim() == current_inject_hash
}

/// Write staleness hashes after a successful inject + compile.
fn write_hashes(test_build_dir: &Path, upstream_dir: &Path, inject_tool: &Path) {
    let source_hash = hash_directory_sources(upstream_dir);
    let inject_hash = hash_file(inject_tool);

    fs::write(test_build_dir.join(".source-hash"), &source_hash).ok();
    fs::write(test_build_dir.join(".inject-hash"), &inject_hash).ok();
}

/// SHA256 of all source files in a test directory (aie.mlir, aie2.py, test.cpp).
fn hash_directory_sources(dir: &Path) -> String {
    let mut hasher = Sha256::new();
    let mut found_any = false;

    for name in &["aie.mlir", "aie2.py", "test.cpp", "test.py"] {
        let path = dir.join(name);
        if let Ok(content) = fs::read(&path) {
            hasher.update(name.as_bytes());
            hasher.update(&content);
            found_any = true;
        }
    }

    if !found_any {
        return "empty".to_string();
    }

    format!("{:x}", hasher.finalize())
}

/// SHA256 of a single file.
fn hash_file(path: &Path) -> String {
    match fs::read(path) {
        Ok(content) => {
            let mut hasher = Sha256::new();
            hasher.update(&content);
            format!("{:x}", hasher.finalize())
        }
        Err(_) => "missing".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Blocklist
// ---------------------------------------------------------------------------

/// Tests that cannot be traced (control packet variants, special flows).
const TRACE_BLOCKLIST: &[&str] = &[
    "ctrl_packet",
    "loadpdi",
    "blockwrite",
    "maskwrite",
    "reconfigure",
];

/// Check if a test name matches any blocklist entry.
fn is_unsupported_for_tracing(name: &str) -> bool {
    TRACE_BLOCKLIST.iter().any(|blocked| name.contains(blocked))
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

/// Run the inject stage: trace-inject.py.
fn run_inject(
    test_name: &str,
    upstream_dir: &Path,
    test_build_dir: &Path,
    config: &TraceConfig,
    env: &SubprocessEnv,
) -> Result<(), TraceOutcome> {
    let inject_tool = config.xdna_emu_root.join("tools/trace-inject.py");

    let mut cmd = Command::new(&config.python);
    cmd.arg(&inject_tool)
        .arg(upstream_dir)
        .arg("--output")
        .arg(test_build_dir)
        .arg("--trace-size")
        .arg(config.trace_size.to_string());
    env.apply(&mut cmd);

    match spawn_with_timeout(&mut cmd, config.inject_timeout) {
        ProcessOutcome::Completed { exit_code, stderr, .. } => {
            if exit_code == 0 {
                Ok(())
            } else {
                Err(TraceOutcome::InjectFailed { stderr })
            }
        }
        ProcessOutcome::Timeout { .. } => {
            Err(TraceOutcome::Timeout { stage: format!("inject ({})", test_name) })
        }
        ProcessOutcome::Wedged { .. } => Err(TraceOutcome::Wedged),
        ProcessOutcome::SpawnError(msg) => {
            Err(TraceOutcome::InjectFailed { stderr: msg })
        }
    }
}

/// Copy or symlink kernel source files (.cc, .cpp, .h) from the upstream
/// test directory into the build directory so that aiecc.py can find them
/// during compilation (link_with = "kernel.o" requires the source nearby).
fn copy_kernel_sources(upstream_dir: &Path, build_dir: &Path) {
    let extensions = ["cc", "cpp", "h"];
    if let Ok(entries) = fs::read_dir(upstream_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let dominated = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| extensions.contains(&e))
                .unwrap_or(false);
            if !dominated {
                continue;
            }
            if let Some(name) = path.file_name() {
                let dest = build_dir.join(name);
                if !dest.exists() {
                    // Symlink to avoid copying large files
                    #[cfg(unix)]
                    {
                        std::os::unix::fs::symlink(&path, &dest).ok();
                    }
                    #[cfg(not(unix))]
                    {
                        fs::copy(&path, &dest).ok();
                    }
                }
            }
        }
    }
}

/// Read extra aiecc.py flags written by trace-inject.py (e.g. --dynamic-objFifos).
fn read_extra_aiecc_flags(test_build_dir: &Path) -> Vec<String> {
    let flags_file = test_build_dir.join(".aiecc-extra-flags");
    match fs::read_to_string(&flags_file) {
        Ok(content) => content
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Compile kernel .cc/.cpp files to .o using xchesscc_wrapper.
///
/// Many npu-xrt tests have external kernel code (kernel.cc, scale.cc, etc.)
/// that must be compiled to .o files before aiecc.py can link them into
/// the ELF.  The original lit tests use xchesscc_wrapper for this.
fn compile_kernels(
    test_build_dir: &Path,
    config: &TraceConfig,
    env: &SubprocessEnv,
) -> Result<(), String> {
    let aietools = std::env::var("XILINX_VITIS_AIETOOLS")
        .unwrap_or_else(|_| String::from("/home/triple/npu-work/aietools"));
    let aietools_include = PathBuf::from(&aietools).join("include");

    // Find all .cc and .cpp files in the build directory
    let cc_files: Vec<PathBuf> = fs::read_dir(test_build_dir)
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .filter_map(|entry| {
            let path = entry.path();
            let dominated = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "cc" || e == "cpp")
                .unwrap_or(false);
            // Skip test.cpp (host code, not kernel)
            let is_test = path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("test"))
                .unwrap_or(false);
            if dominated && !is_test { Some(path) } else { None }
        })
        .collect();

    if cc_files.is_empty() {
        return Ok(());
    }

    for cc_file in &cc_files {
        let stem = cc_file.file_stem().unwrap_or_default();
        let obj_file = test_build_dir.join(format!("{}.o", stem.to_string_lossy()));
        if obj_file.exists() {
            continue; // Already compiled
        }

        let mut cmd = Command::new("xchesscc_wrapper");
        cmd.arg("aie2")
            .arg("-I")
            .arg(&aietools_include)
            .arg("-c")
            .arg(cc_file)
            .arg("-o")
            .arg(&obj_file)
            .current_dir(test_build_dir);
        env.apply(&mut cmd);

        match spawn_with_timeout(&mut cmd, config.compile_timeout) {
            ProcessOutcome::Completed { exit_code, stderr, .. } => {
                if exit_code != 0 {
                    return Err(format!(
                        "xchesscc_wrapper failed for {}: {}",
                        cc_file.display(),
                        stderr,
                    ));
                }
            }
            ProcessOutcome::Timeout { .. } => {
                return Err(format!("kernel compile timed out: {}", cc_file.display()));
            }
            ProcessOutcome::Wedged { .. } => {
                return Err("device wedged during kernel compile".to_string());
            }
            ProcessOutcome::SpawnError(msg) => {
                return Err(format!("spawn error compiling {}: {}", cc_file.display(), msg));
            }
        }
    }

    Ok(())
}

/// Run the compile stage: aiecc.py on the traced MLIR.
fn run_compile(
    test_name: &str,
    test_build_dir: &Path,
    config: &TraceConfig,
    env: &SubprocessEnv,
) -> Result<(), TraceOutcome> {
    let traced_mlir = test_build_dir.join("aie_traced.mlir");

    let mut cmd = Command::new(&config.python);
    cmd.arg(&config.aiecc)
        .arg("--no-aiesim")
        .arg("--aie-generate-xclbin")
        .arg("--aie-generate-npu-insts")
        .arg("--no-compile-host")
        .arg("--xclbin-name=aie.xclbin")
        .arg("--npu-insts-name=insts.bin");

    // Apply extra flags from trace-inject.py (e.g. --dynamic-objFifos)
    for flag in read_extra_aiecc_flags(test_build_dir) {
        cmd.arg(&flag);
    }

    cmd.arg(&traced_mlir)
        .current_dir(test_build_dir);
    env.apply(&mut cmd);

    match spawn_with_timeout(&mut cmd, config.compile_timeout) {
        ProcessOutcome::Completed { exit_code, stderr, .. } => {
            if exit_code == 0 {
                Ok(())
            } else {
                Err(TraceOutcome::CompileFailed { stderr })
            }
        }
        ProcessOutcome::Timeout { .. } => {
            Err(TraceOutcome::Timeout { stage: format!("compile ({})", test_name) })
        }
        ProcessOutcome::Wedged { .. } => Err(TraceOutcome::Wedged),
        ProcessOutcome::SpawnError(msg) => {
            Err(TraceOutcome::CompileFailed { stderr: msg })
        }
    }
}

/// Run the execute stage: trace-run.py on the compiled xclbin.
fn run_execute(
    test_name: &str,
    test_build_dir: &Path,
    trace_output_dir: &Path,
    config: &TraceConfig,
    env: &SubprocessEnv,
) -> Result<PathBuf, TraceOutcome> {
    let run_tool = config.xdna_emu_root.join("tools/trace-run.py");
    let manifest = test_build_dir.join("manifest.json");

    let mut cmd = Command::new(&config.python);
    cmd.arg(&run_tool)
        .arg(&manifest)
        .arg("--output-dir")
        .arg(trace_output_dir);
    env.apply(&mut cmd);

    match spawn_with_timeout(&mut cmd, config.execute_timeout) {
        ProcessOutcome::Completed { exit_code, stderr, .. } => {
            if exit_code == 0 {
                let trace_json = trace_output_dir.join("trace.json");
                Ok(trace_json)
            } else {
                Err(TraceOutcome::RunFailed { stderr })
            }
        }
        ProcessOutcome::Timeout { .. } => {
            Err(TraceOutcome::Timeout { stage: format!("execute ({})", test_name) })
        }
        ProcessOutcome::Wedged { .. } => Err(TraceOutcome::Wedged),
        ProcessOutcome::SpawnError(msg) => {
            Err(TraceOutcome::RunFailed { stderr: msg })
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the full trace pipeline for a single test.
///
/// Steps:
/// 1. Check blocklist
/// 2. Check staleness (skip inject + compile if up-to-date)
/// 3. Inject traces into MLIR
/// 4. Compile with aiecc.py
/// 5. Execute on NPU and collect traces
pub fn run_trace_pipeline(
    test_name: &str,
    upstream_dir: &Path,
    config: &TraceConfig,
) -> TraceOutcome {
    // Blocklist check
    if is_unsupported_for_tracing(test_name) {
        return TraceOutcome::Skipped {
            reason: format!("blocklisted: {}", test_name),
        };
    }

    let env = SubprocessEnv::from_config(config);
    let test_build_dir = config.build_traced_dir.join(test_name);
    let trace_output_dir = config.traces_dir.join(test_name);
    let inject_tool = config.xdna_emu_root.join("tools/trace-inject.py");

    // Staleness check
    let needs_rebuild = !is_up_to_date(&test_build_dir, upstream_dir, &inject_tool);

    if needs_rebuild {
        // Create build directory
        if let Err(e) = fs::create_dir_all(&test_build_dir) {
            return TraceOutcome::InjectFailed {
                stderr: format!("Failed to create build dir: {}", e),
            };
        }

        // Stage 1: Inject
        if let Err(outcome) = run_inject(test_name, upstream_dir, &test_build_dir, config, &env) {
            return outcome;
        }

        // Check if inject produced a "skipped" manifest
        let manifest_path = test_build_dir.join("manifest.json");
        if manifest_path.exists() {
            if let Ok(content) = fs::read_to_string(&manifest_path) {
                if content.contains("\"skipped\"") {
                    return TraceOutcome::Skipped {
                        reason: "already has trace configuration".to_string(),
                    };
                }
            }
        }

        // Stage 1b: Copy kernel sources (.cc/.cpp/.h) from upstream so
        // compile_kernels and aiecc.py can find them.
        copy_kernel_sources(upstream_dir, &test_build_dir);

        // Stage 1c: Compile external kernel .cc files to .o
        if let Err(msg) = compile_kernels(&test_build_dir, config, &env) {
            return TraceOutcome::CompileFailed { stderr: msg };
        }

        // Stage 2: Compile
        if let Err(outcome) = run_compile(test_name, &test_build_dir, config, &env) {
            return outcome;
        }

        // Write staleness hashes after successful inject + compile
        write_hashes(&test_build_dir, upstream_dir, &inject_tool);
    }

    // Stage 3: Execute (always runs -- traces are the point)
    match run_execute(test_name, &test_build_dir, &trace_output_dir, config, &env) {
        Ok(trace_json) => TraceOutcome::Success {
            trace_json,
            output_dir: trace_output_dir,
        },
        Err(outcome) => outcome,
    }
}

/// Discover test names from the mlir-aie npu-xrt source tree.
///
/// A test directory must contain either `aie.mlir` or `aie2.py`.
/// Returns (test_name, source_dir) pairs sorted alphabetically.
pub fn discover_tests(test_source_root: &Path) -> Vec<(String, PathBuf)> {
    let mut tests = Vec::new();

    let entries = match fs::read_dir(test_source_root) {
        Ok(entries) => entries,
        Err(e) => {
            log::error!("Failed to read test source dir {}: {}", test_source_root.display(), e);
            return tests;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let has_mlir = path.join("aie.mlir").exists();
        let has_python = path.join("aie2.py").exists();

        if has_mlir || has_python {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                tests.push((name.to_string(), path.clone()));
            }
        }

        // Check for subdirectories (nested test structure)
        if let Ok(sub_entries) = fs::read_dir(&path) {
            for sub_entry in sub_entries.flatten() {
                let sub_path = sub_entry.path();
                if !sub_path.is_dir() {
                    continue;
                }
                let sub_has_mlir = sub_path.join("aie.mlir").exists();
                let sub_has_python = sub_path.join("aie2.py").exists();
                if sub_has_mlir || sub_has_python {
                    // Use parent/child as the test name
                    if let (Some(parent), Some(child)) = (
                        path.file_name().and_then(|n| n.to_str()),
                        sub_path.file_name().and_then(|n| n.to_str()),
                    ) {
                        let name = format!("{}/{}", parent, child);
                        tests.push((name, sub_path));
                    }
                }
            }
        }
    }

    tests.sort_by(|a, b| a.0.cmp(&b.0));
    tests
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocklist() {
        assert!(is_unsupported_for_tracing("ctrl_packet_basic"));
        assert!(is_unsupported_for_tracing("test_ctrl_packet_reconfig"));
        assert!(is_unsupported_for_tracing("loadpdi_test"));
        assert!(is_unsupported_for_tracing("blockwrite_example"));
        assert!(!is_unsupported_for_tracing("add_one_objFifo"));
        assert!(!is_unsupported_for_tracing("vec_scalar_mul"));
    }

    #[test]
    fn test_hash_file_missing() {
        let hash = hash_file(Path::new("/nonexistent/file.txt"));
        assert_eq!(hash, "missing");
    }

    #[test]
    fn test_hash_directory_empty() {
        let hash = hash_directory_sources(Path::new("/nonexistent/dir"));
        assert_eq!(hash, "empty");
    }

    #[test]
    fn test_hash_file_deterministic() {
        let tmp = std::env::temp_dir().join("test_trace_hash.txt");
        fs::write(&tmp, "hello world").unwrap();
        let h1 = hash_file(&tmp);
        let h2 = hash_file(&tmp);
        assert_eq!(h1, h2);
        assert_ne!(h1, "missing");
        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_staleness_no_xclbin() {
        // No xclbin means not up-to-date
        let tmp = std::env::temp_dir().join("test_staleness_no_xclbin");
        fs::create_dir_all(&tmp).ok();
        assert!(!is_up_to_date(
            &tmp,
            Path::new("/nonexistent"),
            Path::new("/nonexistent"),
        ));
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_resolve_makes_absolute() {
        let config = TraceConfig::default().resolve();
        assert!(config.xdna_emu_root.is_absolute());
        assert!(config.mlir_aie_source.is_absolute());
        assert!(config.build_traced_dir.is_absolute());
        assert!(config.traces_dir.is_absolute());
    }

    #[test]
    fn test_make_absolute_preserves_symlinks() {
        // make_absolute should NOT resolve symlinks -- critical for venv python
        let tmp = std::env::temp_dir().join("test_make_absolute_symlink");
        let target = tmp.join("target_file");
        let link = tmp.join("the_link");

        fs::create_dir_all(&tmp).ok();
        fs::write(&target, "hello").ok();

        // Create symlink: the_link -> target_file
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&target, &link).ok();
            let result = make_absolute(link.clone());
            // The result should point to 'the_link', not 'target_file'
            assert!(
                result.ends_with("the_link"),
                "make_absolute resolved symlink: got {:?}",
                result,
            );
        }

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_make_absolute_cleans_dotdot() {
        let result = make_absolute(PathBuf::from("/foo/bar/../baz"));
        assert_eq!(result, PathBuf::from("/foo/baz"));
    }
}
