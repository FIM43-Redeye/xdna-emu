//! Total activity-span extraction from a single aiesimulator VCD, in AIE cycles.
//!
//! This is the aiesim side of the three-way timing calibration (Option C in
//! `docs/coverage/three-way-timing-calibration.md`). It produces a single
//! "total active cycles" number per kernel, symmetric with `parse-trace.py
//! --out-cycles` for the HW and interpreter sides (which compute `max(ts) -
//! min(ts)` over trace events).
//!
//! ## Units
//!
//! aiesimulator VCDs use a picosecond timescale; the AIE core clock runs near
//! 1 GHz, so each cycle is ~952 ps. Rather than hardcode the period, we derive
//! it from the VCD's own time grid: the median gap between consecutive time
//! markers is one clock period (most state changes land one cycle apart). The
//! finest grid (gcd of all markers) is the half-cycle edge. This self-calibrates
//! to whatever frequency the installed aiesimulator runs at.
//!
//! ## Active span vs full sim
//!
//! The global VCD time range spans the entire simulation, including pre- and
//! post-kernel idle. To match `parse-trace`'s event-bounded span we measure the
//! range of value changes on *activity* signals only -- by default the trace
//! `event` subsystem, which is the closest analog to the trace-BO events the HW
//! side decodes. Callers can widen the included subsystems.

use crate::vcd::mapping::MappingTree;
use crate::vcd::state_path::{StatePath, Subsystem};

/// Activity span of an aiesimulator VCD, in both picoseconds and AIE cycles.
#[derive(Debug, Clone, PartialEq)]
pub struct CycleSpan {
    /// Earliest activity-signal change, in ps (timescale units).
    pub first_ps: u64,
    /// Latest activity-signal change, in ps.
    pub last_ps: u64,
    /// `last_ps - first_ps`.
    pub span_ps: u64,
    /// Derived AIE clock period in ps (median consecutive time-marker gap).
    pub period_ps: u64,
    /// `span_ps / period_ps`, the total active cycles. The comparable scalar.
    pub span_cycles: u64,
    /// Number of activity signals that contributed a change.
    pub n_signals: usize,
    /// Number of value changes counted across included signals.
    pub n_changes: usize,
}

/// Derive the AIE clock period (ps) from a sorted, de-duplicated time table.
///
/// Returns the median gap between consecutive distinct timestamps; for an AIE
/// VCD this is one clock period. Returns `None` if there are fewer than two
/// distinct timestamps (no timeline -- a dump-vars-only / degenerate VCD).
pub fn derive_period_ps(time_table: &[u64]) -> Option<u64> {
    if time_table.len() < 2 {
        return None;
    }
    let mut gaps: Vec<u64> = time_table.windows(2).map(|w| w[1] - w[0]).filter(|&g| g > 0).collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_unstable();
    Some(gaps[gaps.len() / 2])
}

/// Compute the activity span of an aiesimulator VCD in AIE cycles.
///
/// `include` selects which subsystems count as activity; if empty, defaults to
/// `[Subsystem::Event]` (the trace-event analog). Signals whose subsystem is not
/// included are ignored.
///
/// Returns an all-zero [`CycleSpan`] (with the derived period if available) when
/// the VCD has no value changes on the included signals -- e.g. a degenerate
/// dump-vars-only capture, which callers should treat as "no timing".
pub fn cycle_span(vcd_path: &str, tree: &MappingTree, include: &[Subsystem]) -> Result<CycleSpan, String> {
    if !std::path::Path::new(vcd_path).exists() {
        return Err(format!("VCD file not found: {}", vcd_path));
    }

    let default_include = [Subsystem::Event];
    let include: &[Subsystem] = if include.is_empty() {
        &default_include
    } else {
        include
    };

    let mut waveform =
        wellen::simple::read(vcd_path).map_err(|e| format!("Failed to read VCD '{}': {:?}", vcd_path, e))?;

    // Phase 1: collect mapped signal refs whose subsystem is included.
    let mapped: Vec<(StatePath, wellen::SignalRef)> = {
        let hierarchy = waveform.hierarchy();
        let mut v = Vec::new();
        for var in hierarchy.iter_vars() {
            let full_name = var.full_name(hierarchy);
            let segments: Vec<&str> = full_name.split('.').collect();
            if let Some(sp) = tree.resolve(&segments) {
                if include.contains(&sp.subsystem()) {
                    v.push((sp, var.signal_ref()));
                }
            }
        }
        v
    };

    // Period comes from the *global* time grid, not the filtered subset, so it
    // reflects the true clock even if the included signals are sparse.
    let period_ps = derive_period_ps(waveform.time_table()).unwrap_or(0);

    if mapped.is_empty() {
        return Ok(CycleSpan {
            first_ps: 0,
            last_ps: 0,
            span_ps: 0,
            period_ps,
            span_cycles: 0,
            n_signals: 0,
            n_changes: 0,
        });
    }

    let signal_refs: Vec<wellen::SignalRef> = mapped.iter().map(|(_, sr)| *sr).collect();
    waveform.load_signals(&signal_refs);

    let time_table = waveform.time_table();
    let mut first = u64::MAX;
    let mut last = 0u64;
    let mut n_changes = 0usize;
    let mut n_signals = 0usize;

    for (_, signal_ref) in &mapped {
        let signal = match waveform.get_signal(*signal_ref) {
            Some(s) => s,
            None => continue,
        };
        let mut contributed = false;
        for (time_idx, _sv) in signal.iter_changes() {
            let time = time_table.get(time_idx as usize).copied().unwrap_or(time_idx as u64);
            // The t=0 dump-vars initialization is not activity; skip it so the
            // span starts at the first real change (matching trace-event bounds).
            if time == 0 {
                continue;
            }
            n_changes += 1;
            contributed = true;
            if time < first {
                first = time;
            }
            if time > last {
                last = time;
            }
        }
        if contributed {
            n_signals += 1;
        }
    }

    if n_changes == 0 {
        return Ok(CycleSpan {
            first_ps: 0,
            last_ps: 0,
            span_ps: 0,
            period_ps,
            span_cycles: 0,
            n_signals: 0,
            n_changes: 0,
        });
    }

    let span_ps = last - first;
    let span_cycles = if period_ps > 0 { span_ps / period_ps } else { 0 };

    Ok(CycleSpan {
        first_ps: first,
        last_ps: last,
        span_ps,
        period_ps,
        span_cycles,
        n_signals,
        n_changes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_period_from_synthetic_grid() {
        // Grid mixing half-cycle (476) and full-cycle (952) edges.
        // Gaps: 952, 476, 476, 952, 952 -> sorted [476,476,952,952,952],
        // median (index 2) = 952 = one clock period.
        let tt = vec![0, 952, 1428, 1904, 2856, 3808];
        assert_eq!(derive_period_ps(&tt), Some(952));
    }

    #[test]
    fn derive_period_median_gap() {
        // Gaps: 952, 952, 952, 1904 -> sorted [952,952,952,1904], median = 952.
        let tt = vec![952, 1904, 2856, 3808, 5712];
        assert_eq!(derive_period_ps(&tt), Some(952));
    }

    #[test]
    fn derive_period_none_on_degenerate() {
        assert_eq!(derive_period_ps(&[]), None);
        assert_eq!(derive_period_ps(&[0]), None);
        assert_eq!(derive_period_ps(&[5, 5, 5]), None); // no positive gaps
    }
}
