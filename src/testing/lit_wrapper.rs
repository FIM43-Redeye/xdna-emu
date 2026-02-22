//! Lit subprocess management and output parsing.
//!
//! Wraps LLVM's `lit` test runner as a child process, parsing its stdout
//! for live result updates and its `--output` JSON file for structured
//! per-test results.
//!
//! # Lit output format
//!
//! With `-a` (show all results), lit emits one line per test:
//! ```text
//! PASS: AIE_TEST :: npu-xrt/add_one_objFifo/run.lit (1 of 78)
//! ```
//!
//! The `--output=<file>` flag writes a JSON file with per-test elapsed
//! times and full script output for failures.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use regex::Regex;

use crate::testing::process_control::configure_process_group;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result codes that lit can produce for a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LitResultCode {
    /// Test passed.
    Pass,
    /// Test failed.
    Fail,
    /// Test failed as expected (marked XFAIL).
    XFail,
    /// Test passed unexpectedly (marked XFAIL but passed).
    XPass,
    /// Test is not supported on this platform/configuration.
    Unsupported,
    /// Test exceeded its time limit.
    Timeout,
    /// Test could not be resolved (e.g. missing dependencies).
    Unresolved,
}

impl LitResultCode {
    /// Parse a result code from its string representation in lit output.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "PASS" => Some(Self::Pass),
            "FAIL" => Some(Self::Fail),
            "XFAIL" => Some(Self::XFail),
            "XPASS" => Some(Self::XPass),
            "UNSUPPORTED" => Some(Self::Unsupported),
            "TIMEOUT" => Some(Self::Timeout),
            "UNRESOLVED" => Some(Self::Unresolved),
            _ => None,
        }
    }

    /// Short label for display.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::XFail => "XFAIL",
            Self::XPass => "XPASS",
            Self::Unsupported => "UNSUPPORTED",
            Self::Timeout => "TIMEOUT",
            Self::Unresolved => "UNRESOLVED",
        }
    }

    /// Whether this result counts as a test success (PASS or expected failure).
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Pass | Self::XFail)
    }
}

/// A single parsed test result from a lit stdout line.
///
/// Extracted from lines like:
/// `PASS: AIE_TEST :: npu-xrt/add_one_objFifo/run.lit (1 of 78)`
#[derive(Debug, Clone)]
pub struct LitTestResult {
    /// The result code (PASS, FAIL, etc.).
    pub code: LitResultCode,
    /// The test path as reported by lit (e.g. "npu-xrt/add_one_objFifo/run.lit").
    pub test_path: String,
    /// 1-based index of this test in the run.
    pub index: usize,
    /// Total number of tests in the run.
    pub total: usize,
}

impl LitTestResult {
    /// Extract a short test name from the full path.
    ///
    /// Strips the common "npu-xrt/" prefix and "/run.lit" suffix to get
    /// just the test directory name (e.g. "add_one_objFifo").
    pub fn short_name(&self) -> &str {
        let s = self.test_path.as_str();
        let s = s.strip_prefix("npu-xrt/").unwrap_or(s);
        let s = s.strip_suffix("/run.lit").unwrap_or(s);
        s
    }
}

/// Parse a single lit result line from stdout.
///
/// Returns `None` for lines that don't match the expected format (e.g.
/// script output, blank lines, summary lines).
pub fn parse_result_line(line: &str) -> Option<LitTestResult> {
    // Regex for lit result lines. The format has been stable across LLVM
    // releases since at least LLVM 14.
    //
    // Example: PASS: AIE_TEST :: npu-xrt/add_one_objFifo/run.lit (1 of 78)
    let re = Regex::new(
        r"^(PASS|FAIL|XFAIL|XPASS|UNSUPPORTED|TIMEOUT|UNRESOLVED): .+? :: (.+?) \((\d+) of (\d+)\)$"
    ).expect("lit result regex is valid");

    let caps = re.captures(line.trim())?;

    let code = LitResultCode::from_str(&caps[1])?;
    let test_path = caps[2].to_string();
    let index: usize = caps[3].parse().ok()?;
    let total: usize = caps[4].parse().ok()?;

    Some(LitTestResult {
        code,
        test_path,
        index,
        total,
    })
}

// ---------------------------------------------------------------------------
// JSON output parsing (from lit --output=<file>)
// ---------------------------------------------------------------------------

/// Structured test result from lit's JSON output file.
#[derive(Debug)]
pub struct LitJsonResult {
    /// Full test name (e.g. "AIE_TEST :: npu-xrt/add_one_objFifo/run.lit").
    pub name: String,
    /// Result code.
    pub code: LitResultCode,
    /// Elapsed time in seconds.
    pub elapsed: f64,
    /// Full test script output (useful for debugging failures).
    pub output: String,
}

/// Summary of a complete lit run, parsed from the JSON output file.
#[derive(Debug)]
pub struct LitRunSummary {
    /// Per-test results.
    pub results: Vec<LitJsonResult>,
    /// Total elapsed time reported by lit.
    pub total_elapsed: f64,
}

/// Parse lit's `--output` JSON file into structured results.
///
/// The JSON format is:
/// ```json
/// {
///   "__version__": [0, 1, 0],
///   "elapsed": 552.43,
///   "tests": [
///     {
///       "name": "AIE_TEST :: npu-xrt/add_one_objFifo/run.lit",
///       "code": "PASS",
///       "elapsed": 1.23,
///       "output": "..."
///     }
///   ]
/// }
/// ```
pub fn parse_json_output(path: &Path) -> Result<LitRunSummary, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read lit output file {}: {}", path.display(), e))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse lit output JSON: {}", e))?;

    let total_elapsed = json.get("elapsed")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let tests = json.get("tests")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Missing 'tests' array in lit output JSON".to_string())?;

    let mut results = Vec::with_capacity(tests.len());
    for test in tests {
        let name = test.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let code_str = test.get("code")
            .and_then(|v| v.as_str())
            .unwrap_or("UNRESOLVED");
        let code = LitResultCode::from_str(code_str)
            .unwrap_or(LitResultCode::Unresolved);

        let elapsed = test.get("elapsed")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let output = test.get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        results.push(LitJsonResult {
            name,
            code,
            elapsed,
            output,
        });
    }

    Ok(LitRunSummary {
        results,
        total_elapsed,
    })
}

// ---------------------------------------------------------------------------
// Lit subprocess runner
// ---------------------------------------------------------------------------

/// Configuration for spawning a lit subprocess.
pub struct LitConfig {
    /// Path to the lit test directory (e.g. `<build_dir>/test/npu-xrt/`).
    pub test_dir: PathBuf,
    /// Per-test timeout in seconds (forwarded to lit's --timeout).
    pub timeout_secs: Option<u32>,
    /// Parallelism level for lit (-j). Default: 1 (sequential for predictable progress).
    pub jobs: usize,
    /// Extra arguments passed verbatim to lit.
    pub extra_args: Vec<String>,
    /// Test name filters. If non-empty, passed as `--filter` regex to lit.
    pub filters: Vec<String>,
    /// Outer watchdog timeout for the entire lit process (seconds).
    /// If lit itself hangs, this kills it.
    pub watchdog_secs: u32,
}

impl Default for LitConfig {
    fn default() -> Self {
        Self {
            test_dir: PathBuf::new(),
            timeout_secs: None,
            jobs: 1,
            extra_args: Vec::new(),
            filters: Vec::new(),
            watchdog_secs: 3600, // 1 hour default watchdog
        }
    }
}

/// Callback invoked for each test result as it's parsed from lit stdout.
///
/// The `elapsed_secs` parameter is wall-clock time since the last result
/// (or since lit started, for the first result).
pub type OnResult = Box<dyn FnMut(&LitTestResult, f64)>;

/// Run lit as a subprocess, invoking `on_result` for each test as it completes.
///
/// Returns the path to the JSON output file (for post-run analysis) and
/// the lit process exit code.
pub fn run_lit(
    config: &LitConfig,
    mut on_result: OnResult,
) -> Result<(PathBuf, i32), String> {
    // Create temp file for JSON output
    let json_output = std::env::temp_dir().join(format!(
        "lit-runner-{}.json",
        std::process::id()
    ));

    let mut cmd = Command::new("lit");

    // -a: show all results (not just failures)
    // --no-progress-bar: don't emit progress bar escape sequences
    // --output: write structured JSON results
    cmd.arg("-a")
        .arg("--no-progress-bar")
        .arg("--output")
        .arg(&json_output)
        .arg("-j")
        .arg(config.jobs.to_string());

    if let Some(timeout) = config.timeout_secs {
        cmd.arg("--timeout").arg(timeout.to_string());
    }

    // Time reporting in summary
    cmd.arg("--time-tests");

    // Apply filters as a combined regex
    if !config.filters.is_empty() {
        let filter_regex = config.filters.join("|");
        cmd.arg("--filter").arg(&filter_regex);
    }

    // Extra arguments from user
    for arg in &config.extra_args {
        cmd.arg(arg);
    }

    // Test directory last
    cmd.arg(&config.test_dir);

    // Set up piped stdout for live parsing, stderr passthrough
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::inherit());

    // Process group for clean kill
    configure_process_group(&mut cmd);

    let mut child = cmd.spawn()
        .map_err(|e| format!("Failed to spawn lit: {}", e))?;

    // Read stdout line-by-line for live progress
    let stdout = child.stdout.take()
        .ok_or_else(|| "Failed to capture lit stdout".to_string())?;
    let reader = BufReader::new(stdout);

    let mut last_result_time = Instant::now();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                log::warn!("Error reading lit stdout: {}", e);
                continue;
            }
        };

        if let Some(result) = parse_result_line(&line) {
            let now = Instant::now();
            let elapsed = now.duration_since(last_result_time).as_secs_f64();
            last_result_time = now;
            on_result(&result, elapsed);
        }
    }

    // Wait for lit to exit
    let status = child.wait()
        .map_err(|e| format!("Failed to wait for lit: {}", e))?;

    let exit_code = status.code().unwrap_or(-1);

    Ok((json_output, exit_code))
}

/// Auto-detect the mlir-aie build directory.
///
/// Checks in order:
/// 1. Explicit `--build-dir` argument (if provided)
/// 2. `MLIR_AIE_BUILD_DIR` environment variable
/// 3. `../mlir-aie/build/` relative to the xdna-emu project root
pub fn detect_build_dir(explicit: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(dir) = explicit {
        if dir.join("test/npu-xrt").exists() {
            return Ok(dir.to_path_buf());
        }
        return Err(format!(
            "Build dir {} does not contain test/npu-xrt/",
            dir.display()
        ));
    }

    // Check environment variable
    if let Ok(dir) = std::env::var("MLIR_AIE_BUILD_DIR") {
        let path = PathBuf::from(&dir);
        if path.join("test/npu-xrt").exists() {
            return Ok(path);
        }
    }

    // Check config-based path (uses xdna-emu's config.toml)
    let config = crate::config::Config::get();
    let mlir_aie = PathBuf::from(config.mlir_aie_path());
    let build_dir = mlir_aie.join("build");
    if build_dir.join("test/npu-xrt").exists() {
        return Ok(build_dir);
    }

    Err(format!(
        "Could not find mlir-aie build directory. Checked:\n  \
         - MLIR_AIE_BUILD_DIR env var\n  \
         - {}/build/test/npu-xrt/\n\
         Use --build-dir to specify explicitly.",
        mlir_aie.display()
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pass() {
        let line = "PASS: AIE_TEST :: npu-xrt/add_one_objFifo/run.lit (1 of 78)";
        let result = parse_result_line(line).expect("should parse");
        assert_eq!(result.code, LitResultCode::Pass);
        assert_eq!(result.test_path, "npu-xrt/add_one_objFifo/run.lit");
        assert_eq!(result.index, 1);
        assert_eq!(result.total, 78);
    }

    #[test]
    fn test_parse_fail() {
        let line = "FAIL: AIE_TEST :: npu-xrt/some_test/run.lit (5 of 78)";
        let result = parse_result_line(line).expect("should parse");
        assert_eq!(result.code, LitResultCode::Fail);
        assert_eq!(result.test_path, "npu-xrt/some_test/run.lit");
        assert_eq!(result.index, 5);
        assert_eq!(result.total, 78);
    }

    #[test]
    fn test_parse_xfail() {
        let line = "XFAIL: AIE_TEST :: npu-xrt/expected_fail/run.lit (10 of 50)";
        let result = parse_result_line(line).expect("should parse");
        assert_eq!(result.code, LitResultCode::XFail);
        assert_eq!(result.index, 10);
        assert_eq!(result.total, 50);
    }

    #[test]
    fn test_parse_xpass() {
        let line = "XPASS: AIE_TEST :: npu-xrt/unexpected_pass/run.lit (3 of 10)";
        let result = parse_result_line(line).expect("should parse");
        assert_eq!(result.code, LitResultCode::XPass);
    }

    #[test]
    fn test_parse_unsupported() {
        let line = "UNSUPPORTED: AIE_TEST :: npu-xrt/needs_chess/run.lit (42 of 78)";
        let result = parse_result_line(line).expect("should parse");
        assert_eq!(result.code, LitResultCode::Unsupported);
        assert_eq!(result.index, 42);
    }

    #[test]
    fn test_parse_timeout() {
        let line = "TIMEOUT: AIE_TEST :: npu-xrt/hung_test/run.lit (7 of 78)";
        let result = parse_result_line(line).expect("should parse");
        assert_eq!(result.code, LitResultCode::Timeout);
    }

    #[test]
    fn test_parse_unresolved() {
        let line = "UNRESOLVED: AIE_TEST :: npu-xrt/broken/run.lit (1 of 1)";
        let result = parse_result_line(line).expect("should parse");
        assert_eq!(result.code, LitResultCode::Unresolved);
    }

    #[test]
    fn test_parse_non_result_lines() {
        // Script output, blank lines, summaries should return None
        assert!(parse_result_line("").is_none());
        assert!(parse_result_line("  ").is_none());
        assert!(parse_result_line("-- Testing: 78 tests --").is_none());
        assert!(parse_result_line("Testing Time: 552.43s").is_none());
        assert!(parse_result_line("  Passed    : 40").is_none());
        assert!(parse_result_line("Script:").is_none());
        assert!(parse_result_line("RUN: at line 1: ...").is_none());
    }

    #[test]
    fn test_parse_different_suite_names() {
        // lit suite name varies by configuration
        let line = "PASS: MLIR_AIE :: npu-xrt/add_one/run.lit (1 of 10)";
        let result = parse_result_line(line).expect("should parse");
        assert_eq!(result.code, LitResultCode::Pass);
        assert_eq!(result.test_path, "npu-xrt/add_one/run.lit");
    }

    #[test]
    fn test_short_name() {
        let result = LitTestResult {
            code: LitResultCode::Pass,
            test_path: "npu-xrt/add_one_objFifo/run.lit".to_string(),
            index: 1,
            total: 78,
        };
        assert_eq!(result.short_name(), "add_one_objFifo");
    }

    #[test]
    fn test_short_name_nested() {
        let result = LitTestResult {
            code: LitResultCode::Pass,
            test_path: "npu-xrt/e2e/add_one/run.lit".to_string(),
            index: 1,
            total: 10,
        };
        // Nested paths keep their structure
        assert_eq!(result.short_name(), "e2e/add_one");
    }

    #[test]
    fn test_short_name_no_prefix() {
        let result = LitTestResult {
            code: LitResultCode::Pass,
            test_path: "some_other/test.lit".to_string(),
            index: 1,
            total: 1,
        };
        // Without the npu-xrt prefix, returns as-is (minus /run.lit)
        assert_eq!(result.short_name(), "some_other/test.lit");
    }

    #[test]
    fn test_result_code_is_success() {
        assert!(LitResultCode::Pass.is_success());
        assert!(LitResultCode::XFail.is_success());
        assert!(!LitResultCode::Fail.is_success());
        assert!(!LitResultCode::XPass.is_success());
        assert!(!LitResultCode::Unsupported.is_success());
        assert!(!LitResultCode::Timeout.is_success());
        assert!(!LitResultCode::Unresolved.is_success());
    }

    #[test]
    fn test_result_code_labels() {
        assert_eq!(LitResultCode::Pass.label(), "PASS");
        assert_eq!(LitResultCode::Fail.label(), "FAIL");
        assert_eq!(LitResultCode::XFail.label(), "XFAIL");
        assert_eq!(LitResultCode::XPass.label(), "XPASS");
        assert_eq!(LitResultCode::Unsupported.label(), "UNSUPPORTED");
        assert_eq!(LitResultCode::Timeout.label(), "TIMEOUT");
        assert_eq!(LitResultCode::Unresolved.label(), "UNRESOLVED");
    }

    #[test]
    fn test_result_code_from_str_invalid() {
        assert!(LitResultCode::from_str("UNKNOWN").is_none());
        assert!(LitResultCode::from_str("pass").is_none()); // case sensitive
        assert!(LitResultCode::from_str("").is_none());
    }

    #[test]
    fn test_parse_json_output() {
        let json = r#"{
            "__version__": [0, 1, 0],
            "elapsed": 100.5,
            "tests": [
                {
                    "name": "AIE_TEST :: npu-xrt/add_one/run.lit",
                    "code": "PASS",
                    "elapsed": 1.23,
                    "output": "Script:\nRUN: something"
                },
                {
                    "name": "AIE_TEST :: npu-xrt/broken/run.lit",
                    "code": "FAIL",
                    "elapsed": 5.67,
                    "output": "error: compilation failed"
                }
            ]
        }"#;

        // Write to temp file
        let tmp = std::env::temp_dir().join("lit_test_output.json");
        std::fs::write(&tmp, json).unwrap();

        let summary = parse_json_output(&tmp).expect("should parse");
        assert_eq!(summary.total_elapsed, 100.5);
        assert_eq!(summary.results.len(), 2);
        assert_eq!(summary.results[0].code, LitResultCode::Pass);
        assert_eq!(summary.results[0].elapsed, 1.23);
        assert_eq!(summary.results[1].code, LitResultCode::Fail);
        assert!(summary.results[1].output.contains("compilation failed"));

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_parse_json_missing_fields() {
        // Graceful handling of missing optional fields
        let json = r#"{
            "tests": [
                {
                    "name": "test1",
                    "code": "PASS"
                }
            ]
        }"#;

        let tmp = std::env::temp_dir().join("lit_test_minimal.json");
        std::fs::write(&tmp, json).unwrap();

        let summary = parse_json_output(&tmp).expect("should parse");
        assert_eq!(summary.total_elapsed, 0.0);
        assert_eq!(summary.results.len(), 1);
        assert_eq!(summary.results[0].elapsed, 0.0);
        assert_eq!(summary.results[0].output, "");

        std::fs::remove_file(&tmp).ok();
    }
}
