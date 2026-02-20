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

/// Compile a test.cpp into a test executable.
///
/// Uses the system g++ with XRT and test_utils headers/libraries.
/// Caches the result: skips recompilation if test.exe is newer than test.cpp.
///
/// # Arguments
/// * `test_cpp` - Path to the test.cpp source file
/// * `output_dir` - Directory to place the compiled test.exe
/// * `mlir_aie_path` - Root of the mlir-aie tree (for test_utils headers)
///
/// Returns the path to the compiled test.exe on success.
pub fn compile_test_exe(
    test_cpp: &Path,
    output_dir: &Path,
    mlir_aie_path: &Path,
) -> Result<PathBuf, String> {
    let test_exe = output_dir.join("test.exe");

    // Cache check: skip if test.exe is newer than test.cpp
    if test_exe.exists() {
        if let (Ok(src_meta), Ok(exe_meta)) =
            (std::fs::metadata(test_cpp), std::fs::metadata(&test_exe))
        {
            if let (Ok(src_time), Ok(exe_time)) =
                (src_meta.modified(), exe_meta.modified())
            {
                if exe_time > src_time {
                    return Ok(test_exe);
                }
            }
        }
    }

    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create output dir: {}", e))?;

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
        .arg("-std=c++17")
        .arg(format!("-I{}", test_lib_include.display()))
        .arg(format!("-I{}", xrt_include.display()))
        .arg(format!("-L{}", xrt_lib.display()))
        .arg("-lxrt_coreutil")
        .arg("-luuid")
        // Preprocessor defines for #define-pattern tests.
        // Harmless for cxxopts tests (macros are guarded by #ifndef).
        .arg("-DXCLBIN=\"aie.xclbin\"")
        .arg("-DINSTS_TXT=\"insts.bin\"")
        .arg("-DKERNEL_NAME=\"MLIR_AIE\"");

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

    let mut cmd = Command::new(test_exe);

    // Sanitize LD_LIBRARY_PATH: strip aietools paths that shadow system libstdc++
    cmd.env("LD_LIBRARY_PATH", sanitized_ld_library_path());

    match pattern {
        TestCppPattern::Cxxopts => {
            // Pass paths via CLI args (test_utils::add_default_options format)
            cmd.arg("-x").arg(xclbin)
                .arg("-i").arg(insts)
                .arg("-k").arg("MLIR_AIE");
        }
        TestCppPattern::Define => {
            // #define pattern: paths are baked in at compile time.
            // Run from the xclbin directory so relative paths resolve.
            if let Some(parent) = xclbin.parent() {
                cmd.current_dir(parent);
            }
        }
    }

    // Spawn the process with a timeout
    let child = match cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return NativeTestResult {
                passed: false,
                total_checks: 0,
                correct_checks: 0,
                stdout: String::new(),
                stderr: format!("Failed to spawn test.exe: {}", e),
                elapsed_secs: start.elapsed().as_secs_f64(),
                exit_code: None,
            };
        }
    };

    // Wait with timeout
    match wait_with_timeout(child, timeout_secs) {
        WaitResult::Completed { stdout, stderr, exit_code } => {
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
            }
        }
        WaitResult::Timeout { stdout, stderr } => {
            let elapsed_secs = start.elapsed().as_secs_f64();
            NativeTestResult {
                passed: false,
                total_checks: 0,
                correct_checks: 0,
                stdout,
                stderr: format!("{}\n(killed after {}s timeout)", stderr, timeout_secs),
                elapsed_secs,
                exit_code: None,
            }
        }
        WaitResult::Error(msg) => {
            NativeTestResult {
                passed: false,
                total_checks: 0,
                correct_checks: 0,
                stdout: String::new(),
                stderr: msg,
                elapsed_secs: start.elapsed().as_secs_f64(),
                exit_code: None,
            }
        }
    }
}

/// Build a sanitized LD_LIBRARY_PATH (reuses same logic as npu_runner).
///
/// Strips aietools paths that ship an ancient libstdc++ which would shadow
/// the system one and cause GLIBCXX errors.
use super::sanitized_ld_library_path;

// -- Timeout handling --------------------------------------------------------

enum WaitResult {
    Completed {
        stdout: String,
        stderr: String,
        exit_code: i32,
    },
    Timeout {
        stdout: String,
        stderr: String,
    },
    Error(String),
}

/// Wait for a child process with a timeout, killing it if it exceeds the limit.
fn wait_with_timeout(mut child: std::process::Child, timeout_secs: u32) -> WaitResult {
    use std::time::Duration;

    let timeout = Duration::from_secs(timeout_secs as u64);
    let start = Instant::now();
    let poll_interval = Duration::from_millis(100);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process finished
                let stdout = child.stdout.take()
                    .map(|mut s| {
                        let mut buf = String::new();
                        std::io::Read::read_to_string(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();
                let stderr = child.stderr.take()
                    .map(|mut s| {
                        let mut buf = String::new();
                        std::io::Read::read_to_string(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();

                return WaitResult::Completed {
                    stdout,
                    stderr,
                    exit_code: status.code().unwrap_or(-1),
                };
            }
            Ok(None) => {
                // Still running -- check timeout
                if start.elapsed() >= timeout {
                    // Kill the process
                    let _ = child.kill();
                    let _ = child.wait(); // Reap zombie

                    let stdout = child.stdout.take()
                        .map(|mut s| {
                            let mut buf = String::new();
                            std::io::Read::read_to_string(&mut s, &mut buf).ok();
                            buf
                        })
                        .unwrap_or_default();
                    let stderr = child.stderr.take()
                        .map(|mut s| {
                            let mut buf = String::new();
                            std::io::Read::read_to_string(&mut s, &mut buf).ok();
                            buf
                        })
                        .unwrap_or_default();

                    return WaitResult::Timeout { stdout, stderr };
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => {
                return WaitResult::Error(format!("Failed to wait for process: {}", e));
            }
        }
    }
}

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
        if trimmed.starts_with("Correct") {
            correct += 1;
        } else if trimmed.starts_with("Error") {
            errors += 1;
        }
    }

    let total = correct + errors;
    (passed, total, correct)
}

/// Default timeout for native test.exe execution (seconds).
/// 30s is generous for any test; matches npu_runner default.
pub const DEFAULT_NATIVE_TIMEOUT_SECS: u32 = 30;

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
    let result = run_native_test(test_exe, xclbin, insts, pattern, DEFAULT_NATIVE_TIMEOUT_SECS);

    let label = if result.passed {
        if result.total_checks > 0 {
            format!("PASS ({}/{})", result.correct_checks, result.total_checks)
        } else {
            "PASS".to_string()
        }
    } else if result.exit_code.is_none() {
        "TIMEOUT".to_string()
    } else if result.total_checks > 0 {
        format!("FAIL ({}/{})", result.correct_checks, result.total_checks)
    } else {
        // No element checks and no PASS -- extract meaningful error
        let meaningful = result.stderr.lines()
            .filter(|l| !l.is_empty() && !l.contains("WARNING"))
            .last()
            .unwrap_or("unknown error");
        format!("ERROR ({})", &meaningful[..meaningful.len().min(50)])
    };

    let passed = result.passed;
    let elapsed = result.elapsed_secs;

    if compact {
        print!("{}: {:18} ({:.1}s)", prefix, label, elapsed);
    } else {
        println!("      {}: {} ({:.1}s)", prefix, label, elapsed);
    }

    // Wait for device idle before next test
    let is_error = label.starts_with("ERROR") || label.starts_with("TIMEOUT");
    npu_runner::wait_for_device_idle(is_error);

    super::runner_stats::HwRunResult {
        label,
        output: Vec::new(), // Native tests don't expose raw output bytes
        passed,
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
