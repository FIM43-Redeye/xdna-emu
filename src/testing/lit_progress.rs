//! Progress display formatting for lit test runs.
//!
//! Tracks test results as they arrive and formats live progress lines
//! and a final summary. Designed to give the same information density as
//! the old build progress grid but with lit doing the heavy lifting.

use std::io::{self, Write};
use std::time::Instant;

use super::lit_wrapper::{LitResultCode, LitTestResult, LitRunSummary};

// ---------------------------------------------------------------------------
// Live progress
// ---------------------------------------------------------------------------

/// Tracks progress during a lit run and formats output.
pub struct ProgressTracker {
    /// When the run started (for total elapsed time).
    start: Instant,
    /// Counts per result code.
    counts: ResultCounts,
    /// Whether to show full output for failures.
    verbose: bool,
}

/// Accumulated counts by result code.
#[derive(Debug, Default)]
pub struct ResultCounts {
    pub pass: usize,
    pub fail: usize,
    pub xfail: usize,
    pub xpass: usize,
    pub unsupported: usize,
    pub timeout: usize,
    pub unresolved: usize,
}

impl ResultCounts {
    /// Total number of results recorded.
    pub fn total(&self) -> usize {
        self.pass + self.fail + self.xfail + self.xpass
            + self.unsupported + self.timeout + self.unresolved
    }

    /// Number of successful results (PASS + XFAIL).
    pub fn successes(&self) -> usize {
        self.pass + self.xfail
    }

    /// Number of failure results (FAIL + XPASS + TIMEOUT + UNRESOLVED).
    pub fn failures(&self) -> usize {
        self.fail + self.xpass + self.timeout + self.unresolved
    }

    fn increment(&mut self, code: LitResultCode) {
        match code {
            LitResultCode::Pass => self.pass += 1,
            LitResultCode::Fail => self.fail += 1,
            LitResultCode::XFail => self.xfail += 1,
            LitResultCode::XPass => self.xpass += 1,
            LitResultCode::Unsupported => self.unsupported += 1,
            LitResultCode::Timeout => self.timeout += 1,
            LitResultCode::Unresolved => self.unresolved += 1,
        }
    }
}

impl ProgressTracker {
    /// Create a new progress tracker.
    pub fn new(verbose: bool) -> Self {
        Self {
            start: Instant::now(),
            counts: ResultCounts::default(),
            verbose,
        }
    }

    /// Record and display a test result.
    ///
    /// `elapsed_secs` is the wall-clock time for this individual test
    /// (measured as time since the previous result).
    pub fn record(&mut self, result: &LitTestResult, elapsed_secs: f64) {
        self.counts.increment(result.code);

        let line = format_progress_line(result, elapsed_secs);
        println!("{}", line);
        io::stdout().flush().ok();
    }

    /// Print the final summary after all tests have run.
    ///
    /// If a JSON output file is available, uses it for accurate per-test
    /// timing. Otherwise falls back to the accumulated counts.
    pub fn print_summary(&self, json_summary: Option<&LitRunSummary>) {
        println!();

        let total_time = if let Some(js) = json_summary {
            js.total_elapsed
        } else {
            self.start.elapsed().as_secs_f64()
        };

        // Use JSON counts if available (more accurate), fall back to live counts
        let counts = if let Some(js) = json_summary {
            let mut c = ResultCounts::default();
            for r in &js.results {
                c.increment(r.code);
            }
            c
        } else {
            // Copy live counts (can't move out of &self)
            ResultCounts {
                pass: self.counts.pass,
                fail: self.counts.fail,
                xfail: self.counts.xfail,
                xpass: self.counts.xpass,
                unsupported: self.counts.unsupported,
                timeout: self.counts.timeout,
                unresolved: self.counts.unresolved,
            }
        };

        println!("Testing Time: {:.1}s", total_time);
        println!();

        // Summary line with counts
        let mut parts = Vec::new();

        if counts.pass > 0 {
            parts.push(format!("{} passed", counts.pass));
        }
        if counts.fail > 0 {
            parts.push(format!("{} failed", counts.fail));
        }
        if counts.xfail > 0 {
            parts.push(format!("{} expected failures", counts.xfail));
        }
        if counts.xpass > 0 {
            parts.push(format!("{} unexpected passes", counts.xpass));
        }
        if counts.unsupported > 0 {
            parts.push(format!("{} unsupported", counts.unsupported));
        }
        if counts.timeout > 0 {
            parts.push(format!("{} timed out", counts.timeout));
        }
        if counts.unresolved > 0 {
            parts.push(format!("{} unresolved", counts.unresolved));
        }

        println!("Results: {}", parts.join(", "));

        // Print failing test names from JSON (if verbose and available)
        if self.verbose {
            if let Some(js) = json_summary {
                let failures: Vec<_> = js.results.iter()
                    .filter(|r| !r.code.is_success() && r.code != LitResultCode::Unsupported)
                    .collect();

                if !failures.is_empty() {
                    println!();
                    println!("Failing tests:");
                    for f in &failures {
                        println!("  {} :: {}", f.code.label(), f.name);
                    }
                }
            }
        }
    }

    /// Get the current counts.
    pub fn counts(&self) -> &ResultCounts {
        &self.counts
    }
}

/// Format a single progress line for display.
///
/// Output: `[42/78] add_one_objFifo               PASS    (1.2s)`
fn format_progress_line(result: &LitTestResult, elapsed_secs: f64) -> String {
    let name = result.short_name();

    // Pad or truncate name to 40 chars for alignment
    let display_name = if name.len() > 40 {
        format!("{}...", &name[..37])
    } else {
        format!("{:<40}", name)
    };

    let label = result.code.label();

    format!(
        "[{:>3}/{}] {}  {:<13} ({:.1}s)",
        result.index, result.total, display_name, label, elapsed_secs
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_progress_pass() {
        let result = LitTestResult {
            code: LitResultCode::Pass,
            test_path: "npu-xrt/add_one_objFifo/run.lit".to_string(),
            index: 1,
            total: 78,
        };

        let line = format_progress_line(&result, 1.23);
        assert!(line.contains("[  1/78]"));
        assert!(line.contains("add_one_objFifo"));
        assert!(line.contains("PASS"));
        assert!(line.contains("(1.2s)"));
    }

    #[test]
    fn test_format_progress_fail() {
        let result = LitTestResult {
            code: LitResultCode::Fail,
            test_path: "npu-xrt/broken_test/run.lit".to_string(),
            index: 42,
            total: 78,
        };

        let line = format_progress_line(&result, 5.67);
        assert!(line.contains("[ 42/78]"));  // right-aligned index
        assert!(line.contains("broken_test"));
        assert!(line.contains("FAIL"));
        assert!(line.contains("(5.7s)"));
    }

    #[test]
    fn test_format_progress_long_name() {
        let result = LitTestResult {
            code: LitResultCode::Pass,
            test_path: "npu-xrt/this_is_a_very_long_test_name_that_exceeds_40_characters/run.lit".to_string(),
            index: 1,
            total: 10,
        };

        let line = format_progress_line(&result, 0.5);
        // Name should be truncated with "..."
        assert!(line.contains("..."));
        // But result should still be there
        assert!(line.contains("PASS"));
    }

    #[test]
    fn test_result_counts() {
        let mut counts = ResultCounts::default();
        assert_eq!(counts.total(), 0);
        assert_eq!(counts.successes(), 0);
        assert_eq!(counts.failures(), 0);

        counts.increment(LitResultCode::Pass);
        counts.increment(LitResultCode::Pass);
        counts.increment(LitResultCode::Fail);
        counts.increment(LitResultCode::XFail);
        counts.increment(LitResultCode::Unsupported);

        assert_eq!(counts.total(), 5);
        assert_eq!(counts.successes(), 3); // 2 pass + 1 xfail
        assert_eq!(counts.failures(), 1);  // 1 fail
        assert_eq!(counts.pass, 2);
        assert_eq!(counts.fail, 1);
        assert_eq!(counts.xfail, 1);
        assert_eq!(counts.unsupported, 1);
    }

    #[test]
    fn test_progress_tracker_record() {
        let mut tracker = ProgressTracker::new(false);

        let result = LitTestResult {
            code: LitResultCode::Pass,
            test_path: "npu-xrt/test1/run.lit".to_string(),
            index: 1,
            total: 3,
        };
        tracker.record(&result, 1.0);

        let result = LitTestResult {
            code: LitResultCode::Fail,
            test_path: "npu-xrt/test2/run.lit".to_string(),
            index: 2,
            total: 3,
        };
        tracker.record(&result, 2.0);

        assert_eq!(tracker.counts().pass, 1);
        assert_eq!(tracker.counts().fail, 1);
        assert_eq!(tracker.counts().total(), 2);
    }
}
