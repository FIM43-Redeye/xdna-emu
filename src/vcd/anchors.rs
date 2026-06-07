//! Macro-anchor extraction from a single in-process NPU1 cluster VCD.
//!
//! This is the aiesim side of Option B (per-anchor) of the three-way timing
//! calibration (`docs/coverage/three-way-timing-calibration.md`). It reduces the
//! DMA channel FSM-state (`status`) timelines to a small, robust set of
//! canonical timing **anchors** that the real-HW trace BO can also yield -- by
//! name (`DMA_*_START_TASK` / `DMA_*_FINISHED_TASK`). Because the in-process
//! cluster VCD is native NPU1 geometry (#87/#88), HW and aiesim align directly
//! on `(col, row, kind)` with **zero geometry normalization**.
//!
//! ## Anchor vocabulary (v1)
//!
//! Per active DMA channel, up to two anchors derived purely from the typed
//! `status` signal ([`StatePath::DmaStatus`]), whose value is the channel FSM
//! state (`0` = idle, nonzero = active):
//!
//! | kind                  | derivation                          | HW trace BO event             |
//! |-----------------------|-------------------------------------|-------------------------------|
//! | `dma_{dir}{ch}_start` | first `status` change leaving idle  | `DMA_{DIR}_{ch}_START_TASK`    |
//! | `dma_{dir}{ch}_done`  | last `status` change                | `DMA_{DIR}_{ch}_FINISHED_TASK` |
//!
//! `{dir}` is `s2mm` or `mm2s`; `{ch}` is the channel index.
//!
//! Observed (add_one_using_dma, in-process VCD): shim channels run a clean
//! `0->1->2->0` (return-to-idle = done); compute/mem channels oscillate `1<->2`
//! per BD iteration and the capture ends mid-run, so "done" is the last change
//! rather than a return to idle. Both reduce to first-change / last-change --
//! the universal rule that needs no knowledge of the FSM-state encoding beyond
//! "0 is idle".
//!
//! ## Cycles, not picoseconds
//!
//! Anchor cycles are `change_time_ps / period_ps`, where `period_ps` is derived
//! from the VCD's own time grid (see [`crate::vcd::cycles::derive_period_ps`]).
//! Absolute cycle origins differ between sources (sim start vs trace start), so
//! the *comparator* normalizes each source to its earliest anchor before taking
//! per-anchor deltas -- only relative spacing is compared.

use crate::vcd::cycles::derive_period_ps;
use crate::vcd::mapping::MappingTree;
use crate::vcd::state_path::StatePath;

/// One canonical timing anchor: a named event at a tile, in AIE cycles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimingAnchor {
    /// Tile column (NPU1 geometry).
    pub col: u8,
    /// Tile row (NPU1 geometry).
    pub row: u8,
    /// Canonical anchor kind, e.g. `dma_s2mm0_start` / `dma_mm2s1_done`.
    pub kind: String,
    /// Anchor time in AIE cycles (`change_ps / period_ps`).
    pub cycle: u64,
}

/// Reduce one channel's change-times (ps) to `(start, done)` cycle anchors.
///
/// `start` is the first change that occurs after `t=0` (the leave-idle edge);
/// `done` is the last change. `done` is `None` when there is only a single
/// post-zero change (cannot distinguish start from completion) or when the
/// timeline is empty. Returns `(None, None)` if `period_ps == 0`.
///
/// The `t=0` dump-vars initialization is excluded -- it is the idle baseline,
/// not activity (mirrors the convention in [`crate::vcd::cycles::cycle_span`]).
pub fn reduce_channel_anchors(change_times_ps: &[u64], period_ps: u64) -> (Option<u64>, Option<u64>) {
    if period_ps == 0 {
        return (None, None);
    }
    let mut first = u64::MAX;
    let mut last = 0u64;
    let mut seen = false;
    for &t in change_times_ps {
        if t == 0 {
            continue;
        }
        seen = true;
        if t < first {
            first = t;
        }
        if t > last {
            last = t;
        }
    }
    if !seen {
        return (None, None);
    }
    let start = Some(first / period_ps);
    let done = if last > first {
        Some(last / period_ps)
    } else {
        None
    };
    (start, done)
}

/// Extract DMA start/done anchors from an in-process NPU1 cluster VCD.
///
/// Walks every variable, keeps those whose name resolves through `tree` to a
/// [`StatePath::DmaStatus`] (the channel FSM-state register), and reduces each
/// channel's `status` timeline to canonical anchors via [`reduce_channel_anchors`].
/// Inactive channels (no post-zero change) contribute nothing, so the NPU1
/// model's two unpopulated compute rows fall out naturally.
///
/// Returns the anchors sorted by `(col, row, kind, cycle)`. An empty result is
/// not an error -- it means no DMA channel showed activity (degenerate VCD).
pub fn extract_dma_anchors(vcd_path: &str, tree: &MappingTree) -> Result<Vec<TimingAnchor>, String> {
    extract_dma_anchors_with_period(vcd_path, tree).map(|(anchors, _period)| anchors)
}

/// As [`extract_dma_anchors`], also returning the derived clock period (ps).
///
/// The CLI echoes `period_ps` as measurement metadata; the cycle conversion is
/// otherwise an internal detail, so the common-case API drops it.
pub fn extract_dma_anchors_with_period(
    vcd_path: &str,
    tree: &MappingTree,
) -> Result<(Vec<TimingAnchor>, u64), String> {
    if !std::path::Path::new(vcd_path).exists() {
        return Err(format!("VCD file not found: {}", vcd_path));
    }

    let mut waveform =
        wellen::simple::read(vcd_path).map_err(|e| format!("Failed to read VCD '{}': {:?}", vcd_path, e))?;

    // Phase 1: collect (col, row, dir, ch, signal_ref) for every `status` signal.
    let mapped: Vec<(u8, u8, &'static str, u8, wellen::SignalRef)> = {
        let hierarchy = waveform.hierarchy();
        let mut v = Vec::new();
        for var in hierarchy.iter_vars() {
            let full_name = var.full_name(hierarchy);
            let segments: Vec<&str> = full_name.split('.').collect();
            if let Some(StatePath::DmaStatus { col, row, dir, ch }) = tree.resolve(&segments) {
                v.push((col, row, dir.as_str(), ch, var.signal_ref()));
            }
        }
        v
    };

    // Period from the global time grid (true clock even if status edges are sparse).
    let period_ps = derive_period_ps(waveform.time_table()).unwrap_or(0);

    if mapped.is_empty() || period_ps == 0 {
        return Ok((Vec::new(), period_ps));
    }

    let signal_refs: Vec<wellen::SignalRef> = mapped.iter().map(|m| m.4).collect();
    waveform.load_signals(&signal_refs);

    let time_table = waveform.time_table();
    let mut anchors = Vec::new();

    for (col, row, dir, ch, signal_ref) in &mapped {
        let signal = match waveform.get_signal(*signal_ref) {
            Some(s) => s,
            None => continue,
        };
        let times: Vec<u64> = signal
            .iter_changes()
            .map(|(time_idx, _sv)| time_table.get(time_idx as usize).copied().unwrap_or(time_idx as u64))
            .collect();
        let (start, done) = reduce_channel_anchors(&times, period_ps);
        if let Some(cycle) = start {
            anchors.push(TimingAnchor {
                col: *col,
                row: *row,
                kind: format!("dma_{}{}_start", dir, ch),
                cycle,
            });
        }
        if let Some(cycle) = done {
            anchors.push(TimingAnchor {
                col: *col,
                row: *row,
                kind: format!("dma_{}{}_done", dir, ch),
                cycle,
            });
        }
    }

    anchors.sort_by(|a, b| {
        (a.col, a.row, a.kind.as_str(), a.cycle).cmp(&(b.col, b.row, b.kind.as_str(), b.cycle))
    });
    Ok((anchors, period_ps))
}

/// Serialize anchors to the bare-measurement JSON consumed by the timing-record
/// glue (`{"period_ps":P,"anchors":[{"col":C,"row":R,"kind":"..","cycle":N}]}`).
///
/// Hand-rolled (no serde dependency), matching the JSON style of the `--cycles`
/// emitter in `src/bin/vcd_compare.rs`. The harness (#85) wraps this with
/// `kernel`/`compiler`/`source` to form a full timing record.
pub fn anchors_to_json(anchors: &[TimingAnchor], period_ps: u64) -> String {
    let mut s = String::new();
    s.push_str(&format!("{{\"period_ps\":{},\"anchors\":[", period_ps));
    for (i, a) in anchors.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"col\":{},\"row\":{},\"kind\":\"{}\",\"cycle\":{}}}",
            a.col, a.row, a.kind, a.cycle
        ));
    }
    s.push_str("]}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::dma_mapping::dma_mapping;
    use crate::vcd::lock_mapping::lock_mapping;
    use crate::vcd::mapping::MappingTree;

    // -- Pure reducer logic --

    #[test]
    fn reduce_start_and_done_from_shim_like_sequence() {
        // 0->1->2->0 at ps {0,100,200,500}, period 100. start=t100 (cycle 1),
        // done=last change t500 (cycle 5).
        let times = [0, 100, 200, 500];
        assert_eq!(reduce_channel_anchors(&times, 100), (Some(1), Some(5)));
    }

    #[test]
    fn reduce_compute_like_oscillation_done_is_last() {
        // Compute channel never returns to idle; done is simply the last change.
        let times = [0, 395, 396, 2575, 2577];
        // period 1: start=395, done=2577.
        assert_eq!(reduce_channel_anchors(&times, 1), (Some(395), Some(2577)));
    }

    #[test]
    fn reduce_single_change_has_no_done() {
        // Only one post-zero change: cannot distinguish start from completion.
        assert_eq!(reduce_channel_anchors(&[0, 300], 100), (Some(3), None));
    }

    #[test]
    fn reduce_empty_or_only_zero_is_none() {
        assert_eq!(reduce_channel_anchors(&[], 100), (None, None));
        assert_eq!(reduce_channel_anchors(&[0], 100), (None, None));
        assert_eq!(reduce_channel_anchors(&[0, 0, 0], 100), (None, None));
    }

    #[test]
    fn reduce_zero_period_is_none() {
        assert_eq!(reduce_channel_anchors(&[0, 100, 200], 0), (None, None));
    }

    // -- JSON serialization --

    #[test]
    fn json_empty_anchors() {
        assert_eq!(anchors_to_json(&[], 952), "{\"period_ps\":952,\"anchors\":[]}\n");
    }

    #[test]
    fn json_orders_and_formats_fields() {
        let anchors = vec![
            TimingAnchor { col: 1, row: 2, kind: "dma_s2mm0_start".into(), cycle: 415 },
            TimingAnchor { col: 1, row: 2, kind: "dma_s2mm0_done".into(), cycle: 2707 },
        ];
        let json = anchors_to_json(&anchors, 952);
        assert!(json.contains("\"period_ps\":952"));
        assert!(json.contains("{\"col\":1,\"row\":2,\"kind\":\"dma_s2mm0_start\",\"cycle\":415}"));
        assert!(json.contains("{\"col\":1,\"row\":2,\"kind\":\"dma_s2mm0_done\",\"cycle\":2707}"));
    }

    // -- End-to-end against a synthetic VCD (deterministic, no fixture needed) --

    /// Minimal VCD: one mem-tile S2MM channel `status` signal driven through a
    /// shim-like 0->1->2->0 sequence, plus a lock signal that must NOT produce
    /// an anchor (only DmaStatus does).
    fn make_synthetic_dma_vcd() -> String {
        "\
$timescale 1 ps $end\n\
$scope module top $end\n\
$scope module math_engine $end\n\
$scope module mem_row $end\n\
$scope module tile_0_1 $end\n\
$scope module dma $end\n\
$scope module s2mm_state0 $end\n\
$var wire 32 ! status [31:0] $end\n\
$upscope $end\n\
$upscope $end\n\
$scope module locks $end\n\
$var wire 32 \" value_0 $end\n\
$upscope $end\n\
$upscope $end\n\
$upscope $end\n\
$upscope $end\n\
$upscope $end\n\
$enddefinitions $end\n\
#0\n\
b0 !\n\
b0 \"\n\
#100\n\
b1 !\n\
#200\n\
b10 !\n\
#500\n\
b0 !\n\
"
        .to_string()
    }

    fn synthetic_tree() -> MappingTree {
        MappingTree::builder()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1)])
            .subsystem(dma_mapping(6, 6))
            .subsystem(lock_mapping(64))
            .done_tile_group()
            .build()
    }

    #[test]
    fn extract_anchors_from_synthetic_vcd() {
        let dir = std::env::temp_dir();
        let vcd_file = dir.join("xdna_emu_anchors_test.vcd");
        std::fs::write(&vcd_file, make_synthetic_dma_vcd()).expect("write synthetic VCD");

        let tree = synthetic_tree();
        let anchors = extract_dma_anchors(vcd_file.to_str().unwrap(), &tree).expect("extract should succeed");

        // Exactly two anchors from the one active channel; lock signal excluded.
        assert_eq!(anchors.len(), 2, "got {:?}", anchors);
        let start = anchors.iter().find(|a| a.kind == "dma_s2mm0_start").expect("start anchor");
        let done = anchors.iter().find(|a| a.kind == "dma_s2mm0_done").expect("done anchor");
        assert_eq!((start.col, start.row), (0, 1));
        // period = median gap of {0,100,200,500} grid -> gaps {100,100,300} -> 100.
        assert_eq!(start.cycle, 1, "start at t=100 / period 100");
        assert_eq!(done.cycle, 5, "done at t=500 / period 100");

        std::fs::remove_file(&vcd_file).ok();
    }

    #[test]
    fn extract_missing_file_is_error() {
        let tree = synthetic_tree();
        assert!(extract_dma_anchors("/nonexistent/x.vcd", &tree).is_err());
    }
}
