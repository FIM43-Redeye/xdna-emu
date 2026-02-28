//! Native test.cpp execution for hardware testing.
//!
//! Compiles and runs each test's own test.cpp on real NPU hardware, exactly
//! as mlir-aie's lit infrastructure does. This handles all custom host logic
//! correctly because the test.cpp IS the hardware specification for how to
//! drive that test.
//!
//! Two test.cpp patterns are supported:
//!
//! - **cxxopts**: uses `test_utils::add_default_options()` for CLI arg parsing.
//!   Invoked with `--xclbin <path> --instr <path> --kernel MLIR_AIE`.
//!
//! - **#define**: uses `#ifndef XCLBIN` / `#define XCLBIN` for hardcoded paths.
//!   Compiled with `-DXCLBIN=...` / `-DINSTS_TXT=...` and run from the build
//!   directory (where the xclbin lives).
//!
//! # Architecture
//!
//! ```text
//! test.cpp (from mlir-aie source tree)
//!   |
//!   v
//! native_hw.rs       -- compiles test.cpp, runs the resulting binary
//!   |
//!   v
//! test.exe           -- native binary linked against XRT + test_utils
//!   |
//!   v
//! /dev/accel/accel0  -- real NPU hardware
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use super::npu_runner;
use super::process_control::{self, ProcessOutcome};

/// Check if an exit code indicates death by signal (negative on Unix).
fn is_signal_death(exit_code: Option<i32>) -> bool {
    // On Unix, processes killed by signal have exit code = 128 + signum
    // from wait(), but Rust's ExitStatus::code() returns None for signal
    // death and the raw signal is in .signal(). Our process_control uses
    // wait4 and returns raw exit codes, where signal death shows as
    // negative or > 128.
    match exit_code {
        Some(c) if c < 0 => true,       // negative = -(signal)
        Some(c) if c > 128 => true,     // 128 + signal (shell convention)
        _ => false,
    }
}

/// Get a human-readable signal name from an exit code.
fn signal_name(exit_code: i32) -> &'static str {
    let signum = if exit_code < 0 { -exit_code } else { exit_code - 128 };
    match signum {
        4 => "SIGILL",
        6 => "SIGABRT",
        7 => "SIGBUS",
        8 => "SIGFPE",
        9 => "SIGKILL",
        11 => "SIGSEGV",
        13 => "SIGPIPE",
        14 => "SIGALRM",
        15 => "SIGTERM",
        _ => "SIG?",
    }
}

/// Result of running a native test.exe on hardware.
pub struct NativeTestResult {
    /// Whether "PASS!" was found in stdout.
    pub passed: bool,
    /// Total element-level checks (Correct + Error lines).
    pub total_checks: usize,
    /// Number of correct element-level checks.
    pub correct_checks: usize,
    /// Full stdout from the test executable.
    pub stdout: String,
    /// Full stderr from the test executable.
    pub stderr: String,
    /// Wall-clock execution time in seconds.
    pub elapsed_secs: f64,
    /// Process exit code (None if killed by signal).
    pub exit_code: Option<i32>,
    /// Process survived SIGKILL -- device is in D-state.
    pub wedged: bool,
}

/// How the test.cpp handles its paths/arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestCppPattern {
    /// Uses `test_utils::add_default_options()` + cxxopts CLI parsing.
    /// Invoked with `--xclbin <path> --instr <path> --kernel MLIR_AIE`.
    Cxxopts,
    /// Uses `#ifndef XCLBIN` / `#define XCLBIN` preprocessor macros.
    /// Compiled with `-D` overrides, run from the xclbin directory.
    Define,
}

/// Detect whether a test directory contains a test.cpp and which pattern it uses.
///
/// Returns `Some((path, pattern))` if test.cpp exists, `None` otherwise.
/// Detection logic:
/// - `add_default_options` in source -> Cxxopts
/// - `#define XCLBIN` or `#ifndef XCLBIN` -> Define
/// - Neither found -> defaults to Cxxopts (the more common pattern)
pub fn detect_test_cpp(source_dir: &Path) -> Option<(PathBuf, TestCppPattern)> {
    let test_cpp = source_dir.join("test.cpp");
    if !test_cpp.exists() {
        return None;
    }

    let content = std::fs::read_to_string(&test_cpp).ok()?;
    let pattern = classify_pattern(&content);

    Some((test_cpp, pattern))
}

/// Classify a test.cpp source by its argument handling pattern.
fn classify_pattern(content: &str) -> TestCppPattern {
    // Check for cxxopts pattern first (more distinctive signal)
    if content.contains("add_default_options") {
        return TestCppPattern::Cxxopts;
    }

    // Check for #define pattern
    if content.contains("#define XCLBIN")
        || content.contains("#ifndef XCLBIN")
        || content.contains("#ifndef INSTS_TXT")
    {
        return TestCppPattern::Define;
    }

    // Default to cxxopts (the more common pattern, and safer --
    // CLI args are passed but ignored if the test doesn't use them)
    TestCppPattern::Cxxopts
}

/// Build preprocessor `-D` flags for `#define`-pattern test.cpp files.
///
/// Uses the provided artifact names (falling back to standard defaults).
/// These defines override the test.cpp's own `#ifndef XCLBIN` / `#define XCLBIN`
/// defaults so the binary finds the correct files at runtime.
///
/// XCLBIN is wrapped in `std::string()` to avoid ambiguity between the
/// `xrt::xclbin(const std::string&)` and `xrt::xclbin(const std::string_view&)`
/// constructors -- a bare `const char*` literal is equally convertible to both.
fn preprocessor_defines(
    xclbin_name: Option<&str>,
    insts_name: Option<&str>,
) -> Vec<String> {
    let xclbin = xclbin_name.unwrap_or("aie.xclbin");
    let insts = insts_name.unwrap_or("insts.bin");
    vec![
        format!("-DXCLBIN=std::string(\"{}\")", xclbin),
        format!("-DINSTS_TXT=\"{}\"", insts),
        "-DKERNEL_NAME=\"MLIR_AIE\"".to_string(),
    ]
}

/// Compile a test.cpp into a test executable.
///
/// Uses CMake when the example has a CMakeLists.txt (picks up buffer-size
/// defines automatically), falling back to direct g++ compilation.
/// Caches the result: skips recompilation if the executable is newer than
/// test.cpp.
///
/// # Arguments
/// * `test_cpp` - Path to the test.cpp source file
/// * `output_dir` - Directory to place the compiled executable
/// * `mlir_aie_path` - Root of the mlir-aie tree (for test_utils headers)
///
/// Returns the path to the compiled executable on success.
pub fn compile_test_exe(
    test_cpp: &Path,
    output_dir: &Path,
    mlir_aie_path: &Path,
) -> Result<PathBuf, String> {
    compile_test_exe_with_artifacts(test_cpp, output_dir, mlir_aie_path, None, None)
}

/// Compile test.exe via CMake using the example's own CMakeLists.txt.
///
/// Returns None if no CMakeLists.txt exists (caller should use g++ fallback).
/// Returns Some(Ok(path)) on success, Some(Err(msg)) on cmake failure.
///
/// CMake picks up buffer-size defines (IN1_SIZE, OUT_SIZE, etc.) from each
/// example's cache variables and target_compile_definitions. XCLBIN/INSTS_TXT
/// are not needed at compile time -- these tests use xrt_test_wrapper.h which
/// takes paths via CLI args at runtime.
fn compile_test_exe_cmake(
    source_dir: &Path,
    build_dir: &Path,
) -> Option<Result<PathBuf, String>> {
    let cmakelists = source_dir.join("CMakeLists.txt");
    if !cmakelists.exists() {
        return None;
    }

    if let Err(e) = std::fs::create_dir_all(build_dir) {
        return Some(Err(format!("Failed to create build dir: {}", e)));
    }

    // Configure. Use system default g++ (not g++-13) because XRT's
    // libxrt_coreutil.so requires CXXABI_1.3.15 from GCC 15's libstdc++.
    let configure = Command::new("cmake")
        .arg("-S").arg(source_dir)
        .arg("-B").arg(build_dir)
        .arg("-DTARGET_NAME=test")
        .arg("-DCMAKE_CXX_COMPILER=g++")
        .output();

    match configure {
        Err(e) => return Some(Err(format!("cmake configure failed: {}", e))),
        Ok(out) if !out.status.success() => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Some(Err(format!("cmake configure failed: {}",
                stderr.lines().take(5).collect::<Vec<_>>().join("\n"))));
        }
        _ => {}
    }

    // Build
    let build = Command::new("cmake")
        .arg("--build").arg(build_dir)
        .output();

    match build {
        Err(e) => return Some(Err(format!("cmake build failed: {}", e))),
        Ok(out) if !out.status.success() => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Some(Err(format!("cmake build failed: {}",
                stderr.lines().take(5).collect::<Vec<_>>().join("\n"))));
        }
        _ => {}
    }

    // Find the executable -- cmake names it "test" per TARGET_NAME
    let exe = build_dir.join("test");
    if exe.exists() {
        Some(Ok(exe))
    } else {
        // Some cmake configs may produce test.exe
        let exe_alt = build_dir.join("test.exe");
        if exe_alt.exists() {
            Some(Ok(exe_alt))
        } else {
            Some(Err(format!("cmake build succeeded but no executable in {}",
                build_dir.display())))
        }
    }
}

/// Compile a test.cpp with explicit artifact names for preprocessor defines.
///
/// Tries CMake first (when CMakeLists.txt exists) to pick up buffer-size
/// defines automatically, then falls back to direct g++ compilation.
///
/// When `xclbin_name` or `insts_name` are provided, they override the
/// default `-DXCLBIN="aie.xclbin"` / `-DINSTS_TXT="insts.bin"` defines.
/// This ensures `#define`-pattern tests find the correct files at runtime.
pub fn compile_test_exe_with_artifacts(
    test_cpp: &Path,
    output_dir: &Path,
    mlir_aie_path: &Path,
    xclbin_name: Option<&str>,
    insts_name: Option<&str>,
) -> Result<PathBuf, String> {
    let test_exe = output_dir.join("test.exe");
    let test_alt = output_dir.join("test");

    // Cache check: skip if executable is newer than test.cpp
    let cached = [&test_exe, &test_alt].iter()
        .find(|p| p.exists())
        .and_then(|exe_path| {
            let src_time = std::fs::metadata(test_cpp).ok()?.modified().ok()?;
            let exe_time = std::fs::metadata(exe_path).ok()?.modified().ok()?;
            if exe_time > src_time { Some(exe_path.to_path_buf()) } else { None }
        });
    if let Some(cached_path) = cached {
        return Ok(cached_path);
    }

    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create output dir: {}", e))?;

    // Try CMake first (picks up buffer-size defines automatically).
    let source_dir = test_cpp.parent().unwrap_or(Path::new("."));
    if let Some(result) = compile_test_exe_cmake(source_dir, output_dir) {
        return result;
    }

    // Fallback: direct g++ compilation (for tests without CMakeLists.txt).

    // test_utils include/library paths.
    // mlir-aie builds test_utils as a static library and installs headers.
    let test_lib_include = mlir_aie_path.join("runtime_lib/test_lib");
    let test_lib_lib = mlir_aie_path.join("build/runtime_lib/x86_64/test_lib/lib");

    // Fall back to install path if build path does not exist
    let test_lib_lib = if test_lib_lib.exists() {
        test_lib_lib
    } else {
        mlir_aie_path.join("install/runtime_lib/x86_64/test_lib/lib")
    };

    let xrt_include = PathBuf::from("/opt/xilinx/xrt/include");
    let xrt_lib = PathBuf::from("/opt/xilinx/xrt/lib");

    // Compile with g++. We link test_utils.cpp directly (like mlir-aie's
    // own CMakeLists does) rather than the static library, because the
    // library sometimes has link-order issues with cxxopts.
    let test_utils_cpp = mlir_aie_path.join("runtime_lib/test_lib/test_utils.cpp");

    let mut cmd = Command::new("g++");
    cmd.arg(test_cpp)
        .arg(&test_utils_cpp)
        .arg("-o").arg(&test_exe)
        .arg("-std=c++23")
        .arg("-include").arg("stdfloat")
        .arg("-DTEST_UTILS_USE_XRT")
        .arg(format!("-I{}", test_lib_include.display()))
        .arg(format!("-I{}", xrt_include.display()))
        .arg(format!("-L{}", xrt_lib.display()))
        .arg("-lxrt_coreutil")
        .arg("-luuid");

    // Preprocessor defines for #define-pattern tests.
    // Harmless for cxxopts tests (macros are guarded by #ifndef).
    for define in preprocessor_defines(xclbin_name, insts_name) {
        cmd.arg(define);
    }

    // Extra defines from Makefile defaults (tests without CMakeLists.txt).
    for define in super::host_defines::extra_defines(source_dir) {
        cmd.arg(define);
    }

    // Also add the test_lib static library path in case test_utils.cpp
    // references symbols from test_library.cpp.
    if test_lib_lib.exists() {
        cmd.arg(format!("-L{}", test_lib_lib.display()));
        cmd.arg("-ltest_lib");
    }

    log::debug!("Compiling test.cpp: {:?}", cmd);

    let output = cmd.output()
        .map_err(|e| format!("Failed to run g++: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "g++ failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            stderr.lines().take(5).collect::<Vec<_>>().join("\n"),
        ));
    }

    Ok(test_exe)
}

/// Run a native test.exe on real NPU hardware.
///
/// # Arguments
/// * `test_exe` - Path to the compiled test executable
/// * `xclbin` - Path to the xclbin file
/// * `insts` - Path to the NPU instructions file
/// * `pattern` - Which argument pattern this test uses
/// * `timeout_secs` - Maximum execution time before killing the process
pub fn run_native_test(
    test_exe: &Path,
    xclbin: &Path,
    insts: &Path,
    pattern: TestCppPattern,
    timeout_secs: u32,
) -> NativeTestResult {
    let start = Instant::now();

    // Canonicalize paths: test_exe and xclbin may be relative to the
    // original CWD. We need absolute paths because current_dir changes
    // the working directory for the child process.
    let abs_exe = std::fs::canonicalize(test_exe).unwrap_or_else(|_| test_exe.to_path_buf());
    let abs_xclbin = std::fs::canonicalize(xclbin).unwrap_or_else(|_| xclbin.to_path_buf());
    let abs_insts = std::fs::canonicalize(insts).unwrap_or_else(|_| insts.to_path_buf());

    let mut cmd = Command::new(&abs_exe);

    // Sanitize LD_LIBRARY_PATH: strip aietools paths that shadow system libstdc++
    cmd.env("LD_LIBRARY_PATH", sanitized_ld_library_path());

    // Always run from the xclbin's directory. Some tests use hardcoded
    // relative paths (e.g., load_instr_binary("insts.bin")), so they must
    // run from the build directory. Cxxopts tests with absolute path args
    // are unaffected by the working directory.
    if let Some(parent) = abs_xclbin.parent() {
        cmd.current_dir(parent);
    }

    match pattern {
        TestCppPattern::Cxxopts => {
            // Pass paths via CLI args (test_utils::add_default_options format)
            cmd.arg("-x").arg(&abs_xclbin)
                .arg("-i").arg(&abs_insts)
                .arg("-k").arg("MLIR_AIE");
        }
        TestCppPattern::Define => {
            // #define pattern: paths are baked in at compile time.
            // No CLI args needed -- paths resolved via current_dir above.
        }
    }

    // Spawn with timeout and D-state detection via shared process control.
    match process_control::spawn_with_timeout(&mut cmd, timeout_secs) {
        ProcessOutcome::Completed { stdout, stderr, exit_code } => {
            let elapsed_secs = start.elapsed().as_secs_f64();
            let (passed, total_checks, correct_checks) = parse_test_output(&stdout);

            NativeTestResult {
                passed,
                total_checks,
                correct_checks,
                stdout,
                stderr,
                elapsed_secs,
                exit_code: Some(exit_code),
                wedged: false,
            }
        }
        ProcessOutcome::Timeout { stdout, stderr } => {
            let elapsed_secs = start.elapsed().as_secs_f64();
            NativeTestResult {
                passed: false,
                total_checks: 0,
                correct_checks: 0,
                stdout,
                stderr: format!("{}\n(killed after {}s timeout)", stderr, timeout_secs),
                elapsed_secs,
                exit_code: None,
                wedged: false,
            }
        }
        ProcessOutcome::Wedged { pid, stdout, stderr } => {
            let elapsed_secs = start.elapsed().as_secs_f64();
            NativeTestResult {
                passed: false,
                total_checks: 0,
                correct_checks: 0,
                stdout,
                stderr: format!("{}\n(process {} wedged in D-state after {}s)", stderr, pid, timeout_secs),
                elapsed_secs,
                exit_code: None,
                wedged: true,
            }
        }
        ProcessOutcome::SpawnError(msg) => {
            NativeTestResult {
                passed: false,
                total_checks: 0,
                correct_checks: 0,
                stdout: String::new(),
                stderr: msg,
                elapsed_secs: start.elapsed().as_secs_f64(),
                exit_code: None,
                wedged: false,
            }
        }
    }
}

/// Build a sanitized LD_LIBRARY_PATH (reuses same logic as npu_runner).
///
/// Strips aietools paths that ship an ancient libstdc++ which would shadow
/// the system one and cause GLIBCXX errors.
use super::sanitized_ld_library_path;

// -- Output parsing ----------------------------------------------------------

/// Parse test stdout to determine pass/fail and element-level results.
///
/// Looks for:
/// - `PASS!` anywhere in stdout -> passed
/// - `Correct` lines -> correct element checks
/// - `Error` lines -> failed element checks
///
/// Returns `(passed, total_checks, correct_checks)`.
pub fn parse_test_output(stdout: &str) -> (bool, usize, usize) {
    let passed = stdout.contains("PASS!");
    let mut correct = 0usize;
    let mut errors = 0usize;

    for line in stdout.lines() {
        let trimmed = line.trim();
        // Standard pattern: "Correct(...)" / "Error(...)"
        // two_col pattern: "... is correct  : ..."
        // distribute_repeat pattern: "error at index[N]: expected X got Y"
        if trimmed.starts_with("Correct") || trimmed.contains("is correct") {
            correct += 1;
        } else if trimmed.starts_with("Error") || trimmed.starts_with("error at index") {
            errors += 1;
        }
    }

    let total = correct + errors;
    (passed, total, correct)
}

use super::runner_config::DEFAULT_HW_TIMEOUT_SECS;

/// Run a native test on hardware and produce an HwRunResult for display.
///
/// This is the high-level entry point that the test runner calls. It:
/// 1. Runs the test.exe
/// 2. Parses the output
/// 3. Waits for device idle
/// 4. Returns a result compatible with the existing display/stats infrastructure
pub fn run_native_and_print(
    test_exe: &Path,
    xclbin: &Path,
    insts: &Path,
    pattern: TestCppPattern,
    prefix: &str,
    compact: bool,
) -> super::runner_stats::HwRunResult {
    use super::runner_stats::{HwOutcome, HwRunResult};

    let result = run_native_test(test_exe, xclbin, insts, pattern, DEFAULT_HW_TIMEOUT_SECS);

    let (outcome, label) = if result.wedged {
        (HwOutcome::Wedged, "WEDGED (D-state)".to_string())
    } else if result.passed {
        let label = if result.total_checks > 0 {
            format!("PASS ({}/{})", result.correct_checks, result.total_checks)
        } else {
            "PASS".to_string()
        };
        (HwOutcome::Pass, label)
    } else if result.exit_code.is_none() {
        (HwOutcome::Error, "TIMEOUT".to_string())
    } else if let Some(code) = result.exit_code.filter(|&c| is_signal_death(Some(c))) {
        (HwOutcome::Error, format!("SIGNAL ({})", signal_name(code)))
    } else if result.total_checks > 0 {
        (HwOutcome::Fail, format!("FAIL ({}/{})", result.correct_checks, result.total_checks))
    } else {
        // No element checks and no PASS -- extract meaningful error
        let meaningful = result.stderr.lines()
            .filter(|l| !l.is_empty() && !l.contains("WARNING"))
            .last()
            .unwrap_or("unknown error");
        (HwOutcome::Error, format!("ERROR ({})", &meaningful[..meaningful.len().min(50)]))
    };

    let elapsed = result.elapsed_secs;

    if compact {
        print!("{}: {:18} ({:.1}s)", prefix, label, elapsed);
    } else {
        println!("      {}: {} ({:.1}s)", prefix, label, elapsed);
    }

    // Don't wait for device idle if wedged -- device is unrecoverable.
    if outcome != HwOutcome::Wedged {
        npu_runner::wait_for_device_idle(outcome == HwOutcome::Error);
    }

    // Native test.cpp binaries do their own validation internally and print
    // PASS/FAIL to stdout.  They don't expose raw output bytes -- the
    // comparison logic is baked into the C++ code.  This means the compiler
    // comparison pipeline (CompilerComparison::classify_full) can't get
    // peano_hw output from the native path; only the npu-runner path
    // (which reads raw bytes from a BO) can provide it.
    HwRunResult {
        outcome,
        label,
        output: Vec::new(),
        elapsed_secs: elapsed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_cxxopts_pattern() {
        let source = r#"
            #include "test_utils.h"
            int main(int argc, const char *argv[]) {
                cxxopts::Options options("test");
                test_utils::add_default_options(options);
            }
        "#;
        assert_eq!(classify_pattern(source), TestCppPattern::Cxxopts);
    }

    #[test]
    fn test_classify_define_pattern() {
        let source = r#"
            #include "test_utils.h"
            #ifndef XCLBIN
            #define XCLBIN "final.xclbin"
            #endif
            int main() {}
        "#;
        assert_eq!(classify_pattern(source), TestCppPattern::Define);
    }

    #[test]
    fn test_classify_define_pattern_ifndef_insts() {
        let source = r#"
            #ifndef INSTS_TXT
            #define INSTS_TXT "insts.bin"
            #endif
            int main() {}
        "#;
        assert_eq!(classify_pattern(source), TestCppPattern::Define);
    }

    #[test]
    fn test_classify_unknown_defaults_to_cxxopts() {
        let source = r#"
            #include "test_utils.h"
            int main() { return 0; }
        "#;
        assert_eq!(classify_pattern(source), TestCppPattern::Cxxopts);
    }

    #[test]
    fn test_parse_output_pass_with_checks() {
        let stdout = "\
Correct output[0] = 1 == 1
Correct output[1] = 2 == 2
Correct output[2] = 3 == 3
PASS!
";
        let (passed, total, correct) = parse_test_output(stdout);
        assert!(passed);
        assert_eq!(total, 3);
        assert_eq!(correct, 3);
    }

    #[test]
    fn test_parse_output_fail_with_errors() {
        let stdout = "\
Correct output[0] = 1 == 1
Error output[1] = 999 != 2
Correct output[2] = 3 == 3
FAIL!
";
        let (passed, total, correct) = parse_test_output(stdout);
        assert!(!passed);
        assert_eq!(total, 3);
        assert_eq!(correct, 2);
    }

    #[test]
    fn test_parse_output_pass_no_checks() {
        let stdout = "PASS!\n";
        let (passed, total, correct) = parse_test_output(stdout);
        assert!(passed);
        assert_eq!(total, 0);
        assert_eq!(correct, 0);
    }

    #[test]
    fn test_parse_output_empty() {
        let (passed, total, correct) = parse_test_output("");
        assert!(!passed);
        assert_eq!(total, 0);
        assert_eq!(correct, 0);
    }

    #[test]
    fn test_parse_output_only_errors() {
        let stdout = "\
Error output[0] = 0 != 1
Error output[1] = 0 != 2
FAIL!
";
        let (passed, total, correct) = parse_test_output(stdout);
        assert!(!passed);
        assert_eq!(total, 2);
        assert_eq!(correct, 0);
    }

    #[test]
    fn test_preprocessor_defines_default() {
        let defines = preprocessor_defines(None, None);
        assert!(defines.contains(&"-DXCLBIN=std::string(\"aie.xclbin\")".to_string()));
        assert!(defines.contains(&"-DINSTS_TXT=\"insts.bin\"".to_string()));
        assert!(defines.contains(&"-DKERNEL_NAME=\"MLIR_AIE\"".to_string()));
    }

    #[test]
    fn test_preprocessor_defines_custom_xclbin() {
        let defines = preprocessor_defines(Some("final.xclbin"), None);
        assert!(defines.contains(&"-DXCLBIN=std::string(\"final.xclbin\")".to_string()));
        assert!(defines.contains(&"-DINSTS_TXT=\"insts.bin\"".to_string()));
    }

    #[test]
    fn test_preprocessor_defines_custom_insts() {
        let defines = preprocessor_defines(None, Some("insts.elf"));
        assert!(defines.contains(&"-DXCLBIN=std::string(\"aie.xclbin\")".to_string()));
        assert!(defines.contains(&"-DINSTS_TXT=\"insts.elf\"".to_string()));
    }

    #[test]
    fn test_preprocessor_defines_both_custom() {
        let defines = preprocessor_defines(Some("aie2_plain.xclbin"), Some("insts2_plain.txt"));
        assert!(defines.contains(&"-DXCLBIN=std::string(\"aie2_plain.xclbin\")".to_string()));
        assert!(defines.contains(&"-DINSTS_TXT=\"insts2_plain.txt\"".to_string()));
    }

    #[test]
    fn test_detect_test_cpp_missing() {
        let dir = std::env::temp_dir().join("xdna_emu_native_hw_test_missing");
        let _ = std::fs::create_dir_all(&dir);
        assert!(detect_test_cpp(&dir).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_detect_test_cpp_cxxopts() {
        let dir = std::env::temp_dir().join("xdna_emu_native_hw_test_cxxopts");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("test.cpp"),
            "test_utils::add_default_options(options);\n",
        ).unwrap();

        let result = detect_test_cpp(&dir);
        assert!(result.is_some());
        let (_, pattern) = result.unwrap();
        assert_eq!(pattern, TestCppPattern::Cxxopts);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_detect_test_cpp_define() {
        let dir = std::env::temp_dir().join("xdna_emu_native_hw_test_define");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("test.cpp"),
            "#ifndef XCLBIN\n#define XCLBIN \"final.xclbin\"\n#endif\n",
        ).unwrap();

        let result = detect_test_cpp(&dir);
        assert!(result.is_some());
        let (_, pattern) = result.unwrap();
        assert_eq!(pattern, TestCppPattern::Define);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
