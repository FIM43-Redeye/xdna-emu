//! Text and JSON report generation for VCD comparison results.
//!
//! This module converts a [`ComparisonResult`] (produced by the comparison
//! engine in [`compare`][crate::vcd::compare]) into two output formats:
//!
//! - **Text**: human-readable summary with per-subsystem breakdown, mismatch
//!   details, and missing signal lists. Suitable for terminal output or log
//!   files.
//! - **JSON**: machine-readable report for tooling integration. Uses
//!   `serde_json` (already a project dependency).
//!
//! # Usage
//!
//! ```
//! use xdna_emu::vcd::compare::{ComparisonResult, SignalResult};
//! use xdna_emu::vcd::report::{text_report, json_report};
//! use xdna_emu::vcd::tolerance::ToleranceConfig;
//!
//! let result = ComparisonResult { results: vec![] };
//! let tolerance = ToleranceConfig::strict();
//! let text = text_report(&result, &tolerance);
//! assert!(text.contains("Summary"));
//! let json = json_report(&result);
//! assert!(json.starts_with('{'));
//! ```

use crate::vcd::compare::{ComparisonResult, ComparisonSummary, SignalResult, Source};
use crate::vcd::state_path::Subsystem;
use crate::vcd::tolerance::ToleranceConfig;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ---------------------------------------------------------------------------
// Per-subsystem breakdown
// ---------------------------------------------------------------------------

/// Aggregate counts for a single subsystem.
#[derive(Debug, Default)]
struct SubsystemCounts {
    exact_match: usize,
    timing_offset: usize,
    mismatch: usize,
    missing_emu: usize,
    missing_sim: usize,
    both_empty: usize,
}

/// Build a map of per-subsystem counts from a `ComparisonResult`.
fn subsystem_breakdown(result: &ComparisonResult) -> HashMap<Subsystem, SubsystemCounts> {
    let mut map: HashMap<Subsystem, SubsystemCounts> = HashMap::new();
    for (path, signal_result) in &result.results {
        let sub = path.subsystem();
        let counts = map.entry(sub).or_default();
        match signal_result {
            SignalResult::ExactMatch => counts.exact_match += 1,
            SignalResult::TimingOffset { .. } => counts.timing_offset += 1,
            SignalResult::Mismatch { .. } => counts.mismatch += 1,
            // present_in: Emu means emu has it, sim does not => missing from sim
            SignalResult::Missing { present_in: Source::Emu } => counts.missing_sim += 1,
            // present_in: Sim means sim has it, emu does not => missing from emu
            SignalResult::Missing { present_in: Source::Sim } => counts.missing_emu += 1,
            SignalResult::BothEmpty => counts.both_empty += 1,
        }
    }
    map
}

// ---------------------------------------------------------------------------
// text_report
// ---------------------------------------------------------------------------

/// Generate a human-readable text report from comparison results.
///
/// The report contains:
/// 1. A summary table with aggregate counts and an overall PASS rate.
/// 2. A per-subsystem breakdown showing exact/offset/mismatch/missing counts.
/// 3. A mismatch detail section listing every differing signal with diff count
///    and the first divergence timestamp.
/// 4. A missing signal section listing signals absent from one source.
///
/// The `tolerance` parameter is included in the header so the report is
/// self-describing (the PASS rate depends on the tolerance thresholds used).
pub fn text_report(result: &ComparisonResult, tolerance: &ToleranceConfig) -> String {
    let summary = result.summary();
    let mut out = String::new();

    // -- Header -------------------------------------------------------------
    writeln!(out, "VCD Comparison Report").unwrap();
    writeln!(out, "=====================").unwrap();
    writeln!(out, "Default timing tolerance: {} cycles", tolerance.default_cycles).unwrap();
    writeln!(out).unwrap();

    // -- Summary table ------------------------------------------------------
    write_summary_table(&mut out, &summary);

    // -- Per-subsystem breakdown --------------------------------------------
    write_subsystem_breakdown(&mut out, result);

    // -- Mismatch detail ----------------------------------------------------
    write_mismatch_details(&mut out, result);

    // -- Missing signal detail ----------------------------------------------
    write_missing_details(&mut out, result);

    out
}

/// Write the top-level summary table into `out`.
fn write_summary_table(out: &mut String, summary: &ComparisonSummary) {
    writeln!(out, "Summary:").unwrap();
    writeln!(out, "  Exact match:    {:>6}", summary.exact_match).unwrap();
    writeln!(out, "  Timing offset:  {:>6}", summary.timing_offset).unwrap();
    writeln!(out, "  Mismatch:       {:>6}", summary.mismatch).unwrap();
    writeln!(out, "  Missing (emu):  {:>6}", summary.missing_emu).unwrap();
    writeln!(out, "  Missing (sim):  {:>6}", summary.missing_sim).unwrap();
    writeln!(out, "  Both empty:     {:>6}", summary.both_empty).unwrap();

    let total = summary.exact_match
        + summary.timing_offset
        + summary.mismatch
        + summary.missing_emu
        + summary.missing_sim
        + summary.both_empty;
    if total > 0 {
        // PASS = exact match + timing offset + both_empty (signals that are
        // structurally present but differ only within tolerance, or have no
        // activity to compare). Missing signals are not counted as pass or fail.
        let pass = summary.exact_match + summary.timing_offset + summary.both_empty;
        writeln!(
            out,
            "  PASS rate:      {:>5.1}%",
            pass as f64 / total as f64 * 100.0
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

/// Write a per-subsystem count table into `out`.
///
/// Subsystems are printed in `Subsystem::ALL` order so the output is
/// deterministic regardless of HashMap iteration order.
fn write_subsystem_breakdown(out: &mut String, result: &ComparisonResult) {
    let breakdown = subsystem_breakdown(result);
    if breakdown.is_empty() {
        return;
    }

    writeln!(out, "By subsystem:").unwrap();

    // Column header
    writeln!(
        out,
        "  {:<12}  {:>7}  {:>7}  {:>8}  {:>7}  {:>7}",
        "Subsystem", "exact", "offset", "mismatch", "miss/emu", "miss/sim"
    )
    .unwrap();
    writeln!(
        out,
        "  {:-<12}  {:-<7}  {:-<7}  {:-<8}  {:-<7}  {:-<7}",
        "", "", "", "", "", ""
    )
    .unwrap();

    for &sub in Subsystem::ALL {
        if let Some(c) = breakdown.get(&sub) {
            writeln!(
                out,
                "  {:<12}  {:>7}  {:>7}  {:>8}  {:>7}  {:>7}",
                sub.as_str(),
                c.exact_match,
                c.timing_offset,
                c.mismatch,
                c.missing_emu,
                c.missing_sim
            )
            .unwrap();
        }
    }
    writeln!(out).unwrap();
}

/// Write a list of mismatched signals with diagnostic info into `out`.
fn write_mismatch_details(out: &mut String, result: &ComparisonResult) {
    let mismatches: Vec<_> = result
        .results
        .iter()
        .filter(|(_, r)| matches!(r, SignalResult::Mismatch { .. }))
        .collect();

    writeln!(out, "Mismatches ({}):", mismatches.len()).unwrap();
    if mismatches.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        for (path, result) in &mismatches {
            if let SignalResult::Mismatch { diff_count, total_count, first_diff_time } = result {
                writeln!(
                    out,
                    "  {} -- {}/{} diffs, first at t={}",
                    path, diff_count, total_count, first_diff_time
                )
                .unwrap();
            }
        }
    }
    writeln!(out).unwrap();
}

/// Write a list of missing signals (present in one source but not the other).
fn write_missing_details(out: &mut String, result: &ComparisonResult) {
    let missing: Vec<_> = result
        .results
        .iter()
        .filter(|(_, r)| matches!(r, SignalResult::Missing { .. }))
        .collect();

    writeln!(out, "Missing signals ({}):", missing.len()).unwrap();
    if missing.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        for (path, result) in &missing {
            if let SignalResult::Missing { present_in } = result {
                let label = match present_in {
                    Source::Emu => "emu only (not in sim)",
                    Source::Sim => "sim only (not in emu)",
                };
                writeln!(out, "  {} -- {}", path, label).unwrap();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// json_report
// ---------------------------------------------------------------------------

/// Generate a machine-readable JSON report from comparison results.
///
/// The JSON object has the following structure:
///
/// ```json
/// {
///   "summary": {
///     "exact_match": 10,
///     "timing_offset": 2,
///     "mismatch": 1,
///     "missing_emu": 0,
///     "missing_sim": 1,
///     "both_empty": 5,
///     "total": 19,
///     "pass_rate": 89.47
///   },
///   "by_subsystem": {
///     "lock": { "exact_match": 10, "timing_offset": 0, "mismatch": 0, ... },
///     ...
///   },
///   "mismatches": [
///     { "path": "tile(0,1).lock.value[1]", "diff_count": 3, "total_count": 10, "first_diff_time": 42 }
///   ],
///   "missing": [
///     { "path": "tile(0,2).core.pc[1]", "present_in": "sim" }
///   ]
/// }
/// ```
///
/// Uses `serde_json` (already a project dependency) for correct JSON escaping.
pub fn json_report(result: &ComparisonResult) -> String {
    let summary = result.summary();
    let breakdown = subsystem_breakdown(result);

    // -- summary object -----------------------------------------------------
    let total = summary.exact_match
        + summary.timing_offset
        + summary.mismatch
        + summary.missing_emu
        + summary.missing_sim
        + summary.both_empty;
    let pass = summary.exact_match + summary.timing_offset + summary.both_empty;
    let pass_rate = if total > 0 {
        pass as f64 / total as f64 * 100.0
    } else {
        100.0
    };

    let summary_val = serde_json::json!({
        "exact_match": summary.exact_match,
        "timing_offset": summary.timing_offset,
        "mismatch": summary.mismatch,
        "missing_emu": summary.missing_emu,
        "missing_sim": summary.missing_sim,
        "both_empty": summary.both_empty,
        "total": total,
        "pass_rate": (pass_rate * 100.0).round() / 100.0,
    });

    // -- by_subsystem object ------------------------------------------------
    // Build in Subsystem::ALL order for stable output.
    let mut by_sub = serde_json::Map::new();
    for &sub in Subsystem::ALL {
        if let Some(c) = breakdown.get(&sub) {
            let sub_val = serde_json::json!({
                "exact_match": c.exact_match,
                "timing_offset": c.timing_offset,
                "mismatch": c.mismatch,
                "missing_emu": c.missing_emu,
                "missing_sim": c.missing_sim,
                "both_empty": c.both_empty,
            });
            by_sub.insert(sub.as_str().to_string(), sub_val);
        }
    }

    // -- mismatches array ---------------------------------------------------
    let mismatches: Vec<serde_json::Value> = result
        .results
        .iter()
        .filter_map(|(path, r)| {
            if let SignalResult::Mismatch { diff_count, total_count, first_diff_time } = r {
                Some(serde_json::json!({
                    "path": path.to_string(),
                    "diff_count": diff_count,
                    "total_count": total_count,
                    "first_diff_time": first_diff_time,
                }))
            } else {
                None
            }
        })
        .collect();

    // -- missing array ------------------------------------------------------
    let missing: Vec<serde_json::Value> = result
        .results
        .iter()
        .filter_map(|(path, r)| {
            if let SignalResult::Missing { present_in } = r {
                let source = match present_in {
                    Source::Emu => "emu",
                    Source::Sim => "sim",
                };
                Some(serde_json::json!({
                    "path": path.to_string(),
                    "present_in": source,
                }))
            } else {
                None
            }
        })
        .collect();

    // -- assemble root object -----------------------------------------------
    let root = serde_json::json!({
        "summary": summary_val,
        "by_subsystem": serde_json::Value::Object(by_sub),
        "mismatches": mismatches,
        "missing": missing,
    });

    serde_json::to_string_pretty(&root).unwrap_or_else(|e| {
        format!("{{\"error\": \"JSON serialisation failed: {}\"}}", e)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::compare::{ComparisonResult, SignalResult, Source};
    use crate::vcd::state_path::{DmaDir, StatePath};
    use crate::vcd::tolerance::ToleranceConfig;

    fn make_test_result() -> ComparisonResult {
        ComparisonResult {
            results: vec![
                (
                    StatePath::LockValue { col: 0, row: 1, idx: 0 },
                    SignalResult::ExactMatch,
                ),
                (
                    StatePath::LockValue { col: 0, row: 1, idx: 1 },
                    SignalResult::Mismatch {
                        diff_count: 3,
                        total_count: 10,
                        first_diff_time: 42,
                    },
                ),
                (
                    StatePath::DmaAddress { col: 0, row: 1, dir: DmaDir::S2mm, ch: 0 },
                    SignalResult::TimingOffset { offset_cycles: 2, change_count: 5 },
                ),
                (
                    StatePath::CorePc { col: 0, row: 2, stage: 1 },
                    SignalResult::Missing { present_in: Source::Sim },
                ),
            ],
        }
    }

    // -----------------------------------------------------------------------
    // text_report tests
    // -----------------------------------------------------------------------

    #[test]
    fn text_report_contains_summary() {
        let result = make_test_result();
        let tolerance = ToleranceConfig::strict();
        let text = text_report(&result, &tolerance);
        assert!(text.contains("Summary"));
        assert!(text.contains("Exact match"));
        assert!(text.contains("Mismatch"));
    }

    #[test]
    fn text_report_lists_mismatches() {
        let result = make_test_result();
        let tolerance = ToleranceConfig::strict();
        let text = text_report(&result, &tolerance);
        // The mismatch section must include diff_count/total_count and timestamp.
        assert!(text.contains("3/10 diffs"), "missing diff count: {}", text);
        assert!(text.contains("t=42"), "missing first_diff_time: {}", text);
    }

    #[test]
    fn text_report_lists_missing_signals() {
        let result = make_test_result();
        let tolerance = ToleranceConfig::strict();
        let text = text_report(&result, &tolerance);
        // The CorePc signal is sim-only (present_in: Source::Sim).
        assert!(text.contains("sim only"), "missing 'sim only' label: {}", text);
    }

    #[test]
    fn text_report_pass_rate_reasonable() {
        let result = make_test_result();
        let tolerance = ToleranceConfig::strict();
        let text = text_report(&result, &tolerance);
        // 1 exact + 1 timing offset out of 4 total = 50.0%
        // (missing counts in total but not in pass numerator)
        assert!(text.contains("50.0%"), "unexpected pass rate in: {}", text);
    }

    #[test]
    fn text_report_empty_result() {
        let result = ComparisonResult { results: vec![] };
        let tolerance = ToleranceConfig::strict();
        let text = text_report(&result, &tolerance);
        assert!(text.contains("Summary"));
        // No pass rate line when total == 0.
        assert!(!text.contains("PASS rate"));
    }

    #[test]
    fn text_report_no_mismatches_shows_none() {
        // A result with only exact matches should say "(none)" in the mismatch section.
        let result = ComparisonResult {
            results: vec![
                (StatePath::LockValue { col: 0, row: 0, idx: 0 }, SignalResult::ExactMatch),
            ],
        };
        let tolerance = ToleranceConfig::strict();
        let text = text_report(&result, &tolerance);
        assert!(text.contains("(none)"), "expected '(none)' for no mismatches: {}", text);
    }

    #[test]
    fn text_report_subsystem_breakdown_present() {
        let result = make_test_result();
        let tolerance = ToleranceConfig::strict();
        let text = text_report(&result, &tolerance);
        // Lock, dma, and core subsystems all appear in the test result.
        assert!(text.contains("lock"), "missing lock subsystem: {}", text);
        assert!(text.contains("dma"), "missing dma subsystem: {}", text);
        assert!(text.contains("core"), "missing core subsystem: {}", text);
    }

    // -----------------------------------------------------------------------
    // json_report tests
    // -----------------------------------------------------------------------

    #[test]
    fn json_report_is_valid_json() {
        let result = make_test_result();
        let json = json_report(&result);
        // serde_json round-trip: if our JSON is well-formed, parsing must succeed.
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("json_report produced invalid JSON");
        // Must be an object at the root.
        assert!(parsed.is_object(), "JSON root must be an object");
    }

    #[test]
    fn json_report_has_required_keys() {
        let result = make_test_result();
        let json = json_report(&result);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = parsed.as_object().unwrap();
        assert!(obj.contains_key("summary"), "missing 'summary' key");
        assert!(obj.contains_key("by_subsystem"), "missing 'by_subsystem' key");
        assert!(obj.contains_key("mismatches"), "missing 'mismatches' key");
        assert!(obj.contains_key("missing"), "missing 'missing' key");
    }

    #[test]
    fn json_report_summary_counts_correct() {
        let result = make_test_result();
        let json = json_report(&result);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let summary = &parsed["summary"];
        assert_eq!(summary["exact_match"], 1);
        assert_eq!(summary["timing_offset"], 1);
        assert_eq!(summary["mismatch"], 1);
        // CorePc is present_in: Sim => missing from emu
        assert_eq!(summary["missing_emu"], 1);
        assert_eq!(summary["missing_sim"], 0);
        assert_eq!(summary["total"], 4);
    }

    #[test]
    fn json_report_mismatches_array_populated() {
        let result = make_test_result();
        let json = json_report(&result);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let mismatches = parsed["mismatches"].as_array().unwrap();
        assert_eq!(mismatches.len(), 1);
        let m = &mismatches[0];
        assert_eq!(m["diff_count"], 3);
        assert_eq!(m["total_count"], 10);
        assert_eq!(m["first_diff_time"], 42);
    }

    #[test]
    fn json_report_missing_array_populated() {
        let result = make_test_result();
        let json = json_report(&result);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let missing = parsed["missing"].as_array().unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0]["present_in"], "sim");
    }

    #[test]
    fn json_report_empty_result() {
        let result = ComparisonResult { results: vec![] };
        let json = json_report(&result);
        assert!(json.starts_with('{'));
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("empty result JSON invalid");
        assert_eq!(parsed["summary"]["total"], 0);
        assert_eq!(parsed["mismatches"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["missing"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn json_report_by_subsystem_has_lock_entry() {
        let result = make_test_result();
        let json = json_report(&result);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let by_sub = parsed["by_subsystem"].as_object().unwrap();
        assert!(by_sub.contains_key("lock"), "missing 'lock' in by_subsystem");
        let lock = &by_sub["lock"];
        // 1 ExactMatch + 1 Mismatch for lock signals in test data.
        assert_eq!(lock["exact_match"], 1);
        assert_eq!(lock["mismatch"], 1);
    }

    // -----------------------------------------------------------------------
    // BothEmpty coverage
    // -----------------------------------------------------------------------

    #[test]
    fn text_report_both_empty_counted() {
        let result = ComparisonResult {
            results: vec![
                (StatePath::LockValue { col: 0, row: 0, idx: 0 }, SignalResult::BothEmpty),
            ],
        };
        let tolerance = ToleranceConfig::strict();
        let text = text_report(&result, &tolerance);
        assert!(text.contains("Both empty:"), "missing 'Both empty' line");
        // 1 both_empty out of 1 total => 100% pass rate.
        assert!(text.contains("100.0%"), "expected 100% pass rate for BothEmpty: {}", text);
    }
}
