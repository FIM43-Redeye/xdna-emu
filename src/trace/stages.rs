//! Stage decomposition: per-side cycle deltas between named anchor events.
//!
//! Persists the ad-hoc analysis used to recover from the 2026-05-25 ts/soc
//! misdiagnosis. Consumes the events JSON produced by tools/parse-trace.py
//! (same input shape as [`crate::trace::compare`]) and reports per-stage HW
//! vs EMU cycle deltas using the `soc` field, falling back to `ts` only
//! when `soc` is absent (older snapshots).
//!
//! Default stages match the Phase C 5-stage decomposition for add_one-style
//! single-column flows: shim_dispatch -> shim_done -> memtile_s2mm_done ->
//! memtile_mm2s_done -> compute_s2mm_done -> kernel_acquired. Stages whose
//! anchor events are absent from the trace are skipped, not failed -- a
//! given test's trace config may not capture every anchor (e.g., the diag
//! `_diag_phase_b_add_one_instrumented` trace only covers shim row + core
//! row, omitting memtile DMA events).

use std::fmt::Write;
use std::fs;
use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize)]
struct RawEvent {
    col: u8,
    row: u8,
    name: String,
    ts: u64,
    #[serde(default)]
    soc: Option<u64>,
}

#[derive(Deserialize)]
struct RawFile {
    events: Vec<RawEvent>,
}

#[derive(Clone, Copy)]
pub enum TileFilter {
    Shim,    // row == 0
    Memtile, // row == 1
    Compute, // row >= 2
    Any,
}

impl TileFilter {
    fn matches(self, row: u8) -> bool {
        match self {
            TileFilter::Shim => row == 0,
            TileFilter::Memtile => row == 1,
            TileFilter::Compute => row >= 2,
            TileFilter::Any => true,
        }
    }
}

/// Match an event by name. `AllOf` requires every fragment to be a
/// substring of the event name; this lets one anchor cover any
/// `DMA_S2MM_<channel>_FINISHED_TASK` without enumerating channels.
pub enum NameMatch {
    Exact(&'static str),
    AllOf(&'static [&'static str]),
}

impl NameMatch {
    fn matches(&self, name: &str) -> bool {
        match self {
            NameMatch::Exact(s) => name == *s,
            NameMatch::AllOf(parts) => parts.iter().all(|p| name.contains(p)),
        }
    }
}

#[derive(Clone, Copy)]
pub enum PickKind {
    First,
    Last,
}

pub struct Anchor {
    pub label: &'static str,
    pub tile: TileFilter,
    pub event: NameMatch,
    pub pick: PickKind,
}

pub struct StageDef {
    pub name: &'static str,
    pub from: Anchor,
    pub to: Anchor,
}

/// Default 5-stage decomposition for add_one-style single-column flows.
///
/// Anchors:
/// - 1a: shim dispatch -> shim done (shim MM2S task lifetime)
/// - 1b: shim done -> memtile S2MM done (downstream propagation tail)
/// - 3: memtile S2MM done -> memtile MM2S done (memtile-internal turn)
/// - 4: memtile MM2S done -> compute S2MM done (memtile -> compute)
/// - 5: compute S2MM done -> kernel acquired (compute consume latency)
pub fn default_stages() -> &'static [StageDef] {
    &[
        StageDef {
            name: "1a: shim dispatch -> shim done",
            from: Anchor {
                label: "DMA_MM2S_0_START_TASK @ shim",
                tile: TileFilter::Shim,
                event: NameMatch::Exact("DMA_MM2S_0_START_TASK"),
                pick: PickKind::First,
            },
            to: Anchor {
                label: "DMA_MM2S_0_FINISHED_TASK @ shim",
                tile: TileFilter::Shim,
                event: NameMatch::Exact("DMA_MM2S_0_FINISHED_TASK"),
                pick: PickKind::First,
            },
        },
        StageDef {
            name: "1b: shim done -> memtile S2MM done",
            from: Anchor {
                label: "DMA_MM2S_0_FINISHED_TASK @ shim",
                tile: TileFilter::Shim,
                event: NameMatch::Exact("DMA_MM2S_0_FINISHED_TASK"),
                pick: PickKind::First,
            },
            to: Anchor {
                label: "DMA_S2MM_*_FINISHED_TASK @ memtile",
                tile: TileFilter::Memtile,
                event: NameMatch::AllOf(&["DMA_S2MM_", "_FINISHED_TASK"]),
                pick: PickKind::First,
            },
        },
        StageDef {
            name: "3: memtile S2MM done -> memtile MM2S done",
            from: Anchor {
                label: "DMA_S2MM_*_FINISHED_TASK @ memtile",
                tile: TileFilter::Memtile,
                event: NameMatch::AllOf(&["DMA_S2MM_", "_FINISHED_TASK"]),
                pick: PickKind::First,
            },
            to: Anchor {
                label: "DMA_MM2S_*_FINISHED_TASK @ memtile",
                tile: TileFilter::Memtile,
                event: NameMatch::AllOf(&["DMA_MM2S_", "_FINISHED_TASK"]),
                pick: PickKind::First,
            },
        },
        StageDef {
            name: "4: memtile MM2S done -> compute S2MM done",
            from: Anchor {
                label: "DMA_MM2S_*_FINISHED_TASK @ memtile",
                tile: TileFilter::Memtile,
                event: NameMatch::AllOf(&["DMA_MM2S_", "_FINISHED_TASK"]),
                pick: PickKind::First,
            },
            to: Anchor {
                label: "DMA_S2MM_*_FINISHED_TASK @ compute",
                tile: TileFilter::Compute,
                event: NameMatch::AllOf(&["DMA_S2MM_", "_FINISHED_TASK"]),
                pick: PickKind::First,
            },
        },
        StageDef {
            name: "5: compute S2MM done -> kernel acquired",
            from: Anchor {
                label: "DMA_S2MM_*_FINISHED_TASK @ compute",
                tile: TileFilter::Compute,
                event: NameMatch::AllOf(&["DMA_S2MM_", "_FINISHED_TASK"]),
                pick: PickKind::First,
            },
            to: Anchor {
                label: "INSTR_LOCK_ACQUIRE_REQ @ compute",
                tile: TileFilter::Compute,
                event: NameMatch::Exact("INSTR_LOCK_ACQUIRE_REQ"),
                pick: PickKind::First,
            },
        },
    ]
}

pub struct StageRow {
    pub name: &'static str,
    pub from_label: &'static str,
    pub to_label: &'static str,
    pub hw_from_cyc: Option<u64>,
    pub hw_to_cyc: Option<u64>,
    pub emu_from_cyc: Option<u64>,
    pub emu_to_cyc: Option<u64>,
}

impl StageRow {
    pub fn hw_cyc(&self) -> Option<i64> {
        match (self.hw_from_cyc, self.hw_to_cyc) {
            (Some(f), Some(t)) => Some(t as i64 - f as i64),
            _ => None,
        }
    }
    pub fn emu_cyc(&self) -> Option<i64> {
        match (self.emu_from_cyc, self.emu_to_cyc) {
            (Some(f), Some(t)) => Some(t as i64 - f as i64),
            _ => None,
        }
    }
    pub fn gap(&self) -> Option<i64> {
        match (self.hw_cyc(), self.emu_cyc()) {
            (Some(h), Some(e)) => Some(h - e),
            _ => None,
        }
    }
}

#[derive(Default)]
struct LoadedSide {
    /// Flat list of (col, row, name, cycle); cycle is `soc` if present else `ts`.
    events: Vec<(u8, u8, String, u64)>,
}

fn load_side(path: &Path) -> Result<LoadedSide, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let file: RawFile =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {}", path.display(), e))?;
    let mut out = LoadedSide::default();
    for rec in file.events {
        let cyc = rec.soc.unwrap_or(rec.ts);
        out.events.push((rec.col, rec.row, rec.name, cyc));
    }
    out.events.sort_by_key(|(_, _, _, c)| *c);
    Ok(out)
}

fn find_anchor(side: &LoadedSide, anchor: &Anchor) -> Option<u64> {
    let iter = side
        .events
        .iter()
        .filter(|(_, row, name, _)| anchor.tile.matches(*row) && anchor.event.matches(name));
    match anchor.pick {
        PickKind::First => iter.map(|(_, _, _, c)| *c).next(),
        PickKind::Last => iter.map(|(_, _, _, c)| *c).last(),
    }
}

pub fn compute_stages_from_paths(
    hw_path: &Path,
    emu_path: &Path,
    stages: &[StageDef],
) -> Result<Vec<StageRow>, String> {
    let hw = load_side(hw_path)?;
    let emu = load_side(emu_path)?;
    Ok(stages
        .iter()
        .map(|s| StageRow {
            name: s.name,
            from_label: s.from.label,
            to_label: s.to.label,
            hw_from_cyc: find_anchor(&hw, &s.from),
            hw_to_cyc: find_anchor(&hw, &s.to),
            emu_from_cyc: find_anchor(&emu, &s.from),
            emu_to_cyc: find_anchor(&emu, &s.to),
        })
        .collect())
}

/// Decompose a single side (e.g., EMU only with no HW reference). Useful
/// when probing test traces before kicking off the full HW campaign.
pub fn compute_stages_single_from_path(path: &Path, stages: &[StageDef]) -> Result<Vec<StageRow>, String> {
    let side = load_side(path)?;
    Ok(stages
        .iter()
        .map(|s| StageRow {
            name: s.name,
            from_label: s.from.label,
            to_label: s.to.label,
            hw_from_cyc: None,
            hw_to_cyc: None,
            emu_from_cyc: find_anchor(&side, &s.from),
            emu_to_cyc: find_anchor(&side, &s.to),
        })
        .collect())
}

pub fn format_stages_report(rows: &[StageRow]) -> String {
    let mut out = String::new();
    writeln!(out, "Stage decomposition (soc-based)").ok();
    writeln!(out, "{:<45}  {:>10}  {:>10}  {:>10}", "stage", "HW (cyc)", "EMU (cyc)", "gap").ok();
    writeln!(out, "{}", "-".repeat(45 + 2 + 10 + 2 + 10 + 2 + 10)).ok();
    let mut hw_total: i64 = 0;
    let mut emu_total: i64 = 0;
    let mut have_full = true;
    for r in rows {
        let hw = r.hw_cyc().map(|v| v.to_string()).unwrap_or_else(|| "-".into());
        let emu = r.emu_cyc().map(|v| v.to_string()).unwrap_or_else(|| "-".into());
        let gap = r.gap().map(|v| v.to_string()).unwrap_or_else(|| "-".into());
        writeln!(out, "{:<45}  {:>10}  {:>10}  {:>10}", r.name, hw, emu, gap).ok();
        match (r.hw_cyc(), r.emu_cyc()) {
            (Some(h), Some(e)) => {
                hw_total += h;
                emu_total += e;
            }
            _ => have_full = false,
        }
    }
    writeln!(out, "{}", "-".repeat(45 + 2 + 10 + 2 + 10 + 2 + 10)).ok();
    if have_full {
        writeln!(
            out,
            "{:<45}  {:>10}  {:>10}  {:>10}",
            "TOTAL (all stages present)",
            hw_total,
            emu_total,
            hw_total - emu_total
        )
        .ok();
    } else {
        writeln!(out, "TOTAL: some stages missing anchors -- partial result").ok();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(col: u8, row: u8, name: &str, soc: u64) -> RawEvent {
        RawEvent { col, row, name: name.into(), ts: soc, soc: Some(soc) }
    }

    fn dump_to_tmp(rec: &[RawEvent]) -> tempfile::NamedTempFile {
        let evs: Vec<_> = rec
            .iter()
            .map(|r| {
                serde_json::json!({
                    "col": r.col,
                    "row": r.row,
                    "pkt_type": 2,
                    "slot": 0,
                    "name": r.name,
                    "ts": r.ts,
                    "soc": r.soc.unwrap_or(r.ts),
                })
            })
            .collect();
        let file = serde_json::json!({
            "schema_version": 1,
            "events": evs,
        });
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), serde_json::to_string(&file).unwrap()).unwrap();
        f
    }

    #[test]
    fn stage_decompose_present_anchors() {
        let hw =
            dump_to_tmp(&[ev(1, 0, "DMA_MM2S_0_START_TASK", 100), ev(1, 0, "DMA_MM2S_0_FINISHED_TASK", 200)]);
        let emu =
            dump_to_tmp(&[ev(1, 0, "DMA_MM2S_0_START_TASK", 100), ev(1, 0, "DMA_MM2S_0_FINISHED_TASK", 150)]);
        let rows = compute_stages_from_paths(hw.path(), emu.path(), default_stages()).unwrap();
        // Stage 1a should resolve; others should have None anchors.
        assert_eq!(rows[0].hw_cyc(), Some(100));
        assert_eq!(rows[0].emu_cyc(), Some(50));
        assert_eq!(rows[0].gap(), Some(50));
        assert_eq!(rows[1].hw_cyc(), None);
    }

    #[test]
    fn allof_matches_dma_s2mm_variants() {
        let m = NameMatch::AllOf(&["DMA_S2MM_", "_FINISHED_TASK"]);
        assert!(m.matches("DMA_S2MM_0_FINISHED_TASK"));
        assert!(m.matches("DMA_S2MM_1_FINISHED_TASK"));
        assert!(!m.matches("DMA_MM2S_0_FINISHED_TASK"));
        assert!(!m.matches("DMA_S2MM_0_START_TASK"));
    }
}
