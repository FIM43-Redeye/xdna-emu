//! Efficient trace comparison: HW vs EMU binary trace analysis.
//!
//! Replaces Python `trace-compare.py` for the hot comparison path. Uses
//! compact Rust types (i64 vs Python's 28-byte int objects, contiguous Vec
//! vs fragmented list) reducing memory from 65GB+ to <10MB on dense
//! multi-tile traces like `packet_flow_fanout`.
//!
//! Two modes:
//! - **Single batch**: compare one HW vs EMU `trace_raw.bin` pair
//! - **Sweep directory**: compare all batches from `trace-sweep.py`

use serde::Deserialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

// ============================================================================
// Constants
// ============================================================================

/// Divergence threshold: deltas above this magnitude indicate real divergence
/// rather than micro-timing jitter. Matches Python trace-compare.py.
const DIVERGE_THRESHOLD: i64 = 10;

// ============================================================================
// Types
// ============================================================================

/// Tile identifier from trace packet header.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug, Ord, PartialOrd)]
pub struct TileKey {
    pub col: u8,
    pub row: u8,
    pub pkt_type: u8,
}

/// Decoded trace event with absolute cycle.
#[derive(Clone)]
pub struct TileEvent {
    pub slot: u8,
    pub abs_cycle: u64,
}

/// Per-tile decoded events, keyed by tile identifier.
pub type TileEvents = HashMap<TileKey, Vec<TileEvent>>;

/// Event name configuration from events.json.
///
/// Maps trace slot indices (0-7) to hardware event names. Used for
/// edge/level classification and human-readable output.
#[derive(Deserialize, Default, Clone)]
pub struct EventsConfig {
    #[serde(default)]
    pub core_events: Vec<String>,
    #[serde(default)]
    pub mem_events: Vec<String>,
    #[serde(default)]
    pub memtile_events: Vec<String>,
}

/// Result of analyzing one edge event type.
pub struct EdgeResult {
    pub name: String,
    pub hw_count: usize,
    pub emu_count: usize,
    pub paired: usize,
    /// Per-pair deltas (hw_cycle - emu_cycle). Compact i64 vector.
    pub deltas: Vec<i64>,
    /// First index where |delta| > DIVERGE_THRESHOLD.
    pub diverge_idx: Option<usize>,
    /// Drift classification: "none", "accumulating", "constant_offset", "irregular".
    pub drift_type: String,
    /// Context samples around divergence: (index, hw_cycle, emu_cycle, delta).
    pub samples: Vec<(usize, i64, i64, i64)>,
}

/// Result of analyzing one level event type (interval comparison).
pub struct LevelResult {
    pub name: String,
    pub hw_intervals: usize,
    pub emu_intervals: usize,
    pub paired: usize,
    pub diverge_idx: Option<usize>,
    /// Per-interval duration deltas (hw_dur - emu_dur).
    pub dur_deltas: Vec<i64>,
    /// Interval samples: (index, hw_start, hw_end, emu_start, emu_end).
    pub samples: Vec<(usize, i64, i64, i64, i64)>,
}

/// Options controlling which extended analysis passes to run.
#[derive(Clone, Default)]
pub struct AnalysisOptions {
    /// Enable per-iteration breakdown for recurring events.
    pub iterations: bool,
    /// Enable stall attribution (level stalls -> resolving events).
    pub stalls: bool,
    /// Enable cross-tile correlation (edge-to-edge across modules).
    pub cross_tile: bool,
    /// Remap physical columns to logical 0-indexed before comparison.
    /// Use for serial-vs-parallel HW traces where the driver assigns
    /// different physical columns to each run.
    pub remap_columns: bool,
    /// Enable PC-anchored set/multiset diff and perfcnt cycle bands.
    /// Applied to core tiles (pkt_type == 0) in mode-1 traces where
    /// abs_cycle carries the PC at event fire time.
    pub pc_anchored: bool,
}

impl AnalysisOptions {
    /// All extended analyses enabled.
    pub fn extended() -> Self {
        Self {
            iterations: true,
            stalls: true,
            cross_tile: true,
            remap_columns: false,
            pc_anchored: false,
        }
    }

    /// True if any extended analysis is enabled.
    pub fn any_enabled(&self) -> bool {
        self.iterations || self.stalls || self.cross_tile || self.pc_anchored
    }
}

// --- Per-Iteration Breakdown types ---

/// Timing for one iteration of a recurring event.
pub struct IterationTiming {
    pub iteration: usize,
    /// Cycle gap from previous occurrence (HW).
    pub hw_period: i64,
    /// Cycle gap from previous occurrence (EMU).
    pub emu_period: i64,
    /// hw_period - emu_period.
    pub period_delta: i64,
    /// Running sum of period_delta across iterations.
    pub cumulative_drift: i64,
}

/// Classification of how drift evolves across iterations.
pub enum IterationDriftType {
    /// All period deltas within threshold.
    Stable,
    /// Cumulative drift is monotonically increasing/decreasing.
    Accumulating,
    /// A sudden jump at one iteration.
    StepChange {
        at_iteration: usize,
        magnitude: i64,
    },
    /// No clear pattern.
    Irregular,
}

impl std::fmt::Display for IterationDriftType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "Stable"),
            Self::Accumulating => write!(f, "Accumulating"),
            Self::StepChange {
                at_iteration,
                magnitude,
            } => write!(f, "StepChange at #{} (magnitude {})", at_iteration, magnitude),
            Self::Irregular => write!(f, "Irregular"),
        }
    }
}

/// Per-iteration analysis result for one event on one tile.
pub struct IterationResult {
    pub name: String,
    pub tile: TileKey,
    pub iteration_count: usize,
    pub timings: Vec<IterationTiming>,
    pub drift_classification: IterationDriftType,
    /// First iteration where |period_delta| exceeds threshold.
    pub first_anomaly: Option<usize>,
}

// --- Stall Attribution types ---

/// Relationship between stall and resolution tiles.
#[derive(Clone, Copy, PartialEq)]
pub enum TileRelationship {
    /// Same (col, row), different pkt_type (core vs mem module).
    SameTile,
    /// Same column, adjacent row (e.g., memtile -> compute tile).
    SameColAdjacentRow,
}

/// Static rule mapping a stall event to potential resolution events.
struct StallRule {
    /// Name of the stall (level) event.
    stall: &'static str,
    /// pkt_type where the stall occurs.
    stall_pkt: u8,
    /// Candidate resolution event names.
    resolve: &'static [&'static str],
    /// pkt_type where the resolution occurs.
    resolve_pkt: u8,
    /// Tile relationship between stall and resolution.
    rel: TileRelationship,
}

/// One resolved stall instance.
pub struct StallResolution {
    /// Absolute cycle of stall end.
    pub stall_end_abs: u64,
    /// Absolute cycle of the resolving event.
    pub resolve_abs: u64,
    /// Gap: resolve_abs - stall_end_abs.
    pub gap: i64,
    /// Name of the resolving event.
    pub resolve_name: String,
}

/// Attribution result for one stall type across one tile pair.
pub struct StallAttribution {
    pub stall_name: String,
    pub stall_tile: TileKey,
    pub resolve_tile: TileKey,
    pub hw_resolutions: Vec<StallResolution>,
    pub emu_resolutions: Vec<StallResolution>,
    /// Per-pair gap deltas (hw_gap - emu_gap).
    pub gap_deltas: Vec<i64>,
}

// --- Cross-Tile Correlation types ---

/// Static rule pairing edge events across tiles.
struct CorrelationRule {
    /// Human-readable name for the correlation.
    name: &'static str,
    /// Source event name and pkt_type.
    src_name: &'static str,
    src_pkt: u8,
    /// Destination event name and pkt_type.
    dst_name: &'static str,
    dst_pkt: u8,
    /// Tile relationship.
    rel: TileRelationship,
}

/// One paired occurrence of correlated events.
pub struct CorrelationPair {
    pub src_abs: u64,
    pub dst_abs: u64,
    pub gap: i64,
}

/// Result for one correlation rule across one tile pair.
pub struct CorrelationResult {
    pub rule_name: String,
    pub src_tile: TileKey,
    pub dst_tile: TileKey,
    pub hw_pairs: Vec<CorrelationPair>,
    pub emu_pairs: Vec<CorrelationPair>,
    /// Per-pair gap deltas (hw_gap - emu_gap).
    pub gap_deltas: Vec<i64>,
}

/// Aggregate cross-tile correlation results for one batch.
pub struct CrossTileResult {
    pub correlations: Vec<CorrelationResult>,
}

// ============================================================================
// PC-Anchored types (Task 7)
// ============================================================================

/// Cycle estimate derived from perfcnt overflow anchoring.
///
/// The perfcnt counter overflows at a fixed period, emitting a PC sample
/// each time it wraps. We use those PC-stamped overflows to build a
/// monotone clock: each overflow tick is one period. Any other event PC
/// is linearly interpolated between the two bounding ticks.
#[derive(Debug, Clone, Copy)]
pub struct CycleBand {
    /// Estimated cycle for this PC in the HW trace.
    pub hw_cycle_est: u64,
    /// Estimated cycle for this PC in the EMU trace.
    pub emu_cycle_est: u64,
    /// hw_cycle_est as i64 - emu_cycle_est as i64.
    pub delta_cycles: i64,
    /// True when |delta_cycles| > period / 2.
    pub exceeds_tolerance: bool,
}

/// Per-tile PC-anchored comparison report.
///
/// Produced by `compare_pc_anchored_for_tile`. Covers one tile's
/// mode-1 trace data for one HW/EMU batch pair.
#[derive(Debug, Default, Clone)]
pub struct PCAnchoredReport {
    /// pkt_type from the TileKey (0 = Core, 3 = MemTile, etc.).
    pub pkt_type: u32,
    /// Per event-name PC set diff: (PCs only in HW, PCs only in EMU).
    pub set_diff: HashMap<String, (HashSet<u64>, HashSet<u64>)>,
    /// Per event-name multiset diff: PC -> (hw_count, emu_count, delta).
    pub multiset_diff: HashMap<String, HashMap<u64, (u32, u32, i32)>>,
    /// Per event-name, per-PC cycle band (populated when perfcnt slot found).
    pub cycle_bands: HashMap<String, HashMap<u64, CycleBand>>,
    /// Events dropped because abs_cycle == 0 (no-PC sentinel) on HW side.
    pub unanchored_count_hw: usize,
    /// Events dropped because abs_cycle == 0 (no-PC sentinel) on EMU side.
    pub unanchored_count_emu: usize,
}

/// Comparison result for one tile.
pub struct TileResult {
    pub hw_t0: u64,
    pub emu_t0: u64,
    pub edge_results: Vec<EdgeResult>,
    pub level_results: Vec<LevelResult>,
    /// Per-iteration breakdown (populated when AnalysisOptions.iterations is set).
    pub iteration_results: Vec<IterationResult>,
}

/// Comparison result for one batch (one HW/EMU file pair).
pub struct BatchResult {
    pub batch_idx: usize,
    pub config: EventsConfig,
    pub tiles: Vec<(TileKey, TileResult)>,
    /// Stall attribution results (populated when AnalysisOptions.stalls is set).
    pub stall_attributions: Vec<StallAttribution>,
    /// Cross-tile correlation (populated when AnalysisOptions.cross_tile is set).
    pub cross_tile: Option<CrossTileResult>,
    /// PC-anchored set/multiset diff per tile (populated when
    /// AnalysisOptions.pc_anchored is set). Keyed by TileKey.
    pub pc_anchored: HashMap<TileKey, PCAnchoredReport>,
}

// ============================================================================
// Level event classification
// ============================================================================

/// Level events fire every cycle a condition holds. Comparing them by
/// occurrence index is meaningless -- compare by interval structure instead.
fn is_level_event(name: &str) -> bool {
    matches!(
        name,
        "TRUE"
            | "ACTIVE"
            | "DISABLED"
            | "LOCK_STALL"
            | "MEMORY_STALL"
            | "STREAM_STALL"
            | "CASCADE_STALL"
            | "PORT_RUNNING_0"
            | "PORT_RUNNING_1"
            | "PORT_RUNNING_2"
            | "PORT_RUNNING_3"
            | "PORT_RUNNING_4"
            | "PORT_RUNNING_5"
            | "PORT_RUNNING_6"
            | "PORT_RUNNING_7"
            | "PORT_IDLE_0"
            | "PORT_IDLE_1"
            | "PORT_IDLE_2"
            | "PORT_IDLE_3"
            | "PORT_IDLE_4"
            | "PORT_IDLE_5"
            | "PORT_IDLE_6"
            | "PORT_IDLE_7"
            | "PORT_STALLED_0"
            | "PORT_STALLED_1"
            | "PORT_STALLED_2"
            | "PORT_STALLED_3"
            | "PORT_STALLED_4"
            | "PORT_STALLED_5"
            | "PORT_STALLED_6"
            | "PORT_STALLED_7"
            | "DMA_S2MM_0_STALLED_LOCK"
            | "DMA_S2MM_1_STALLED_LOCK"
            | "DMA_MM2S_0_STALLED_LOCK"
            | "DMA_MM2S_1_STALLED_LOCK"
            | "DMA_S2MM_0_STREAM_STARVATION"
            | "DMA_S2MM_1_STREAM_STARVATION"
            | "DMA_MM2S_0_STREAM_BACKPRESSURE"
            | "DMA_MM2S_1_STREAM_BACKPRESSURE"
            | "DMA_S2MM_0_MEMORY_BACKPRESSURE"
            | "DMA_S2MM_1_MEMORY_BACKPRESSURE"
            | "DMA_MM2S_0_MEMORY_STARVATION"
            | "DMA_MM2S_1_MEMORY_STARVATION"
            | "CONFLICT_DM_BANK_0"
            | "CONFLICT_DM_BANK_1"
            | "CONFLICT_DM_BANK_2"
            | "CONFLICT_DM_BANK_3"
            // MemTile stall events (SEL0/SEL1 naming from MemTileEvent enum)
            | "DMA_S2MM_SEL0_STALLED_LOCK"
            | "DMA_S2MM_SEL1_STALLED_LOCK"
            | "DMA_MM2S_SEL0_STALLED_LOCK"
            | "DMA_MM2S_SEL1_STALLED_LOCK"
            | "DMA_S2MM_SEL0_STREAM_STARVATION"
            | "DMA_S2MM_SEL1_STREAM_STARVATION"
            | "DMA_MM2S_SEL0_STREAM_BACKPRESSURE"
            | "DMA_MM2S_SEL1_STREAM_BACKPRESSURE"
    )
}

/// Map a slot index to its configured event name.
fn slot_name(slot: u8, names: &[String]) -> String {
    if (slot as usize) < names.len() {
        names[slot as usize].clone()
    } else {
        format!("slot{}", slot)
    }
}

// ============================================================================
// Events-JSON ingestion (produced by tools/parse-trace.py)
// ============================================================================
//
// Binary packet decoding lives in mlir-aie's aie.utils.trace module; we call
// it via parse-trace.py and consume the resulting JSON. Keeping the decoder
// single-sourced means HW-format evolution lands once, upstream, rather than
// drifting between two parallel decoders.

#[derive(Deserialize)]
struct EventRecord {
    col: u8,
    row: u8,
    pkt_type: u8,
    slot: u8,
    #[allow(dead_code)] // retained for debugging / future per-event name plumbing
    name: String,
    ts: u64,
}

#[derive(Deserialize)]
struct SlotNamesRecord {
    #[serde(default)]
    core: Vec<String>,
    #[serde(default)]
    mem: Vec<String>,
    #[serde(default)]
    memtile: Vec<String>,
    // "shim" field exists in the JSON but we don't currently consume it;
    // shim-tile slot naming would go through mem_events today.
}

#[derive(Deserialize)]
struct EventsFile {
    #[serde(default)]
    schema_version: u32,
    events: Vec<EventRecord>,
    #[serde(default)]
    slot_names: Option<SlotNamesRecord>,
}

/// Load a per-side events JSON produced by tools/parse-trace.py.
///
/// Returns (per-tile events, slot-name config). The config is populated from
/// the JSON's slot_names field when present; callers that want to override
/// names (legacy aiecc events.json) can substitute their own config.
pub fn load_events_json(path: &Path) -> Result<(TileEvents, EventsConfig), String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let file: EventsFile = serde_json::from_str(&text)
        .map_err(|e| format!("parse {}: {}", path.display(), e))?;
    if file.schema_version != 0 && file.schema_version != 1 {
        return Err(format!(
            "{}: unsupported schema_version {}",
            path.display(),
            file.schema_version
        ));
    }

    let mut tiles: TileEvents = HashMap::new();
    for rec in file.events {
        let key = TileKey {
            col: rec.col,
            row: rec.row,
            pkt_type: rec.pkt_type,
        };
        tiles.entry(key).or_default().push(TileEvent {
            slot: rec.slot,
            abs_cycle: rec.ts,
        });
    }
    for events in tiles.values_mut() {
        events.sort_by_key(|e| e.abs_cycle);
    }

    let config = match file.slot_names {
        Some(sn) => EventsConfig {
            core_events: sn.core,
            mem_events: sn.mem,
            memtile_events: sn.memtile,
        },
        None => EventsConfig::default(),
    };

    Ok((tiles, config))
}


/// Remap physical columns to 0-indexed logical columns.
///
/// The NPU driver assigns physical columns dynamically -- serial runs
/// might use column 1 while parallel runs use column 3 for the same
/// test.  This function normalizes column numbers so traces from
/// different runs can be compared by tile structure.
fn remap_tile_columns(tiles: &TileEvents) -> TileEvents {
    let mut cols: Vec<u8> = tiles.keys().map(|k| k.col).collect::<BTreeSet<_>>().into_iter().collect();
    cols.sort();
    let col_map: HashMap<u8, u8> = cols.into_iter().enumerate().map(|(i, c)| (c, i as u8)).collect();

    tiles
        .iter()
        .map(|(key, events)| {
            let new_key = TileKey {
                col: col_map[&key.col],
                row: key.row,
                pkt_type: key.pkt_type,
            };
            (new_key, events.to_vec())
        })
        .collect()
}

// ============================================================================
// Comparison
// ============================================================================

/// Find the first shared edge event for time alignment.
///
/// Returns (hw_t0, emu_t0) -- the absolute cycle of the first shared edge
/// event in each trace. Subtracting these aligns the timelines.
fn find_edge_anchor(
    hw_events: &[TileEvent],
    emu_events: &[TileEvent],
    slot_names: &[String],
) -> (u64, u64) {
    // First occurrence per slot.
    let mut hw_first: HashMap<u8, u64> = HashMap::new();
    for ev in hw_events {
        hw_first.entry(ev.slot).or_insert(ev.abs_cycle);
    }
    let mut emu_first: HashMap<u8, u64> = HashMap::new();
    for ev in emu_events {
        emu_first.entry(ev.slot).or_insert(ev.abs_cycle);
    }

    // Find first shared slot that is an edge event (not level, not TRUE).
    let shared: BTreeSet<u8> = hw_first
        .keys()
        .copied()
        .filter(|s| emu_first.contains_key(s))
        .collect();

    for slot in shared {
        let name = slot_name(slot, slot_names);
        if !is_level_event(&name) && name != "TRUE" {
            return (hw_first[&slot], emu_first[&slot]);
        }
    }

    // Fallback: first non-slot-0 event with cycle > 0.
    let hw_t0 = hw_events
        .iter()
        .find(|e| e.slot != 0 && e.abs_cycle > 0)
        .map(|e| e.abs_cycle)
        .unwrap_or(0);
    let emu_t0 = emu_events
        .iter()
        .find(|e| e.slot != 0 && e.abs_cycle > 0)
        .map(|e| e.abs_cycle)
        .unwrap_or(0);
    (hw_t0, emu_t0)
}

/// Convert a sorted list of event cycles into contiguous intervals.
///
/// Groups consecutive cycles (delta <= 1) into (start, end) intervals.
/// Non-consecutive firings produce separate intervals.
fn events_to_intervals(cycles: &[i64]) -> Vec<(i64, i64)> {
    if cycles.is_empty() {
        return Vec::new();
    }
    let mut intervals = Vec::new();
    let mut start = cycles[0];
    let mut prev = cycles[0];
    for &c in &cycles[1..] {
        if c - prev > 1 {
            intervals.push((start, prev));
            start = c;
        }
        prev = c;
    }
    intervals.push((start, prev));
    intervals
}

/// Analyze one edge event type: pair by occurrence index, compute deltas.
fn analyze_edge_event(
    name: String,
    hw_cycles: &[i64],
    emu_cycles: &[i64],
) -> EdgeResult {
    let paired = hw_cycles.len().min(emu_cycles.len());
    let deltas: Vec<i64> = (0..paired)
        .map(|i| hw_cycles[i] - emu_cycles[i])
        .collect();

    // Find divergence point.
    let diverge_idx = deltas.iter().position(|d| d.abs() > DIVERGE_THRESHOLD);

    // Classify drift pattern in the divergent tail.
    let drift_type = if let Some(div) = diverge_idx {
        if div < deltas.len() - 1 {
            let tail = &deltas[div..];
            let diffs: Vec<i64> = tail
                .windows(2)
                .map(|w| w[1].abs() - w[0].abs())
                .collect();
            let growing = diffs.iter().filter(|&&d| d > 0).count();
            if growing > diffs.len() * 6 / 10 {
                "accumulating"
            } else if diffs.iter().all(|d| d.abs() < DIVERGE_THRESHOLD) {
                "constant_offset"
            } else {
                "irregular"
            }
        } else {
            "none"
        }
    } else {
        "none"
    };

    // Context samples around the divergence point.
    let (ctx_start, ctx_end) = if let Some(div) = diverge_idx {
        (div.saturating_sub(2), paired.min(div + 5))
    } else {
        (0, paired.min(6))
    };
    let samples: Vec<(usize, i64, i64, i64)> = (ctx_start..ctx_end)
        .map(|i| (i, hw_cycles[i], emu_cycles[i], deltas[i]))
        .collect();

    EdgeResult {
        name,
        hw_count: hw_cycles.len(),
        emu_count: emu_cycles.len(),
        paired,
        deltas,
        diverge_idx,
        drift_type: drift_type.to_string(),
        samples,
    }
}

/// Analyze one level event type by interval structure.
fn analyze_level_event(
    name: String,
    hw_ivs: &[(i64, i64)],
    emu_ivs: &[(i64, i64)],
) -> LevelResult {
    let paired = hw_ivs.len().min(emu_ivs.len());

    let mut dur_deltas = Vec::with_capacity(paired);
    let mut diverge_idx = None;

    for i in 0..paired {
        let hw_dur = hw_ivs[i].1 - hw_ivs[i].0 + 1;
        let emu_dur = emu_ivs[i].1 - emu_ivs[i].0 + 1;
        let dd = hw_dur - emu_dur;
        dur_deltas.push(dd);
        if diverge_idx.is_none() && dd.abs() > DIVERGE_THRESHOLD {
            diverge_idx = Some(i);
        }
    }

    // Interval samples around divergence (or first few).
    let (ctx_start, ctx_end) = if let Some(div) = diverge_idx {
        (div.saturating_sub(1), paired.min(div + 4))
    } else {
        (0, paired.min(4))
    };
    let samples: Vec<(usize, i64, i64, i64, i64)> = (ctx_start..ctx_end)
        .map(|i| {
            (
                i,
                hw_ivs[i].0,
                hw_ivs[i].1,
                emu_ivs[i].0,
                emu_ivs[i].1,
            )
        })
        .collect();

    LevelResult {
        name,
        hw_intervals: hw_ivs.len(),
        emu_intervals: emu_ivs.len(),
        paired,
        diverge_idx,
        dur_deltas,
        samples,
    }
}

// ============================================================================
// Per-Iteration Breakdown (Feature 1)
// ============================================================================

/// Minimum number of paired occurrences to produce iteration analysis.
const MIN_ITERATION_PAIRS: usize = 4;

/// Period delta threshold: below this magnitude, a period is considered stable.
const PERIOD_DELTA_THRESHOLD: i64 = 5;

/// Classify how drift evolves across iteration timings.
fn classify_iteration_drift(timings: &[IterationTiming]) -> IterationDriftType {
    if timings.is_empty() {
        return IterationDriftType::Stable;
    }

    // Check if all period deltas are within threshold -> Stable.
    let all_stable = timings
        .iter()
        .all(|t| t.period_delta.abs() <= PERIOD_DELTA_THRESHOLD);
    if all_stable {
        return IterationDriftType::Stable;
    }

    // Step detection: find if any period_delta jumps by >10x the median.
    let mut abs_deltas: Vec<i64> = timings.iter().map(|t| t.period_delta.abs()).collect();
    abs_deltas.sort();
    let median = abs_deltas[abs_deltas.len() / 2];
    // Avoid division by zero: if median is 0, use 1.
    let step_threshold = (median.max(1)) * 10;
    for t in timings {
        if t.period_delta.abs() > step_threshold && t.period_delta.abs() > PERIOD_DELTA_THRESHOLD {
            return IterationDriftType::StepChange {
                at_iteration: t.iteration,
                magnitude: t.period_delta,
            };
        }
    }

    // Accumulating: cumulative_drift is monotonic across 80%+ of iterations.
    if timings.len() >= 3 {
        let monotonic_count = timings
            .windows(2)
            .filter(|w| {
                // Same sign of cumulative drift change
                let d0 = w[0].cumulative_drift;
                let d1 = w[1].cumulative_drift;
                (d1 - d0).signum() == timings[0].period_delta.signum()
                    || (d1 - d0).abs() <= PERIOD_DELTA_THRESHOLD
            })
            .count();
        if monotonic_count >= (timings.len() - 1) * 8 / 10 {
            return IterationDriftType::Accumulating;
        }
    }

    IterationDriftType::Irregular
}

/// Analyze per-iteration timing for an edge event with repeated occurrences.
fn analyze_iterations_edge(
    name: &str,
    tile: TileKey,
    hw_cycles: &[i64],
    emu_cycles: &[i64],
) -> Option<IterationResult> {
    let paired = hw_cycles.len().min(emu_cycles.len());
    if paired < MIN_ITERATION_PAIRS {
        return None;
    }

    let mut timings = Vec::with_capacity(paired - 1);
    let mut cumulative_drift: i64 = 0;

    for i in 1..paired {
        let hw_period = hw_cycles[i] - hw_cycles[i - 1];
        let emu_period = emu_cycles[i] - emu_cycles[i - 1];
        let period_delta = hw_period - emu_period;
        cumulative_drift += period_delta;
        timings.push(IterationTiming {
            iteration: i,
            hw_period,
            emu_period,
            period_delta,
            cumulative_drift,
        });
    }

    let drift_classification = classify_iteration_drift(&timings);
    let first_anomaly = timings
        .iter()
        .find(|t| t.period_delta.abs() > PERIOD_DELTA_THRESHOLD)
        .map(|t| t.iteration);

    Some(IterationResult {
        name: name.to_string(),
        tile,
        iteration_count: paired,
        timings,
        drift_classification,
        first_anomaly,
    })
}

/// Analyze per-iteration timing for a level event using interval durations.
fn analyze_iterations_level(
    name: &str,
    tile: TileKey,
    hw_ivs: &[(i64, i64)],
    emu_ivs: &[(i64, i64)],
) -> Option<IterationResult> {
    let paired = hw_ivs.len().min(emu_ivs.len());
    if paired < MIN_ITERATION_PAIRS {
        return None;
    }

    let mut timings = Vec::with_capacity(paired - 1);
    let mut cumulative_drift: i64 = 0;

    for i in 1..paired {
        let hw_period = hw_ivs[i].0 - hw_ivs[i - 1].0; // gap between interval starts
        let emu_period = emu_ivs[i].0 - emu_ivs[i - 1].0;
        let period_delta = hw_period - emu_period;
        cumulative_drift += period_delta;
        timings.push(IterationTiming {
            iteration: i,
            hw_period,
            emu_period,
            period_delta,
            cumulative_drift,
        });
    }

    let drift_classification = classify_iteration_drift(&timings);
    let first_anomaly = timings
        .iter()
        .find(|t| t.period_delta.abs() > PERIOD_DELTA_THRESHOLD)
        .map(|t| t.iteration);

    Some(IterationResult {
        name: name.to_string(),
        tile,
        iteration_count: paired,
        timings,
        drift_classification,
        first_anomaly,
    })
}

// ============================================================================
// Stall Attribution (Feature 2)
// ============================================================================

/// Maximum cycle gap to accept a resolution as matching a stall end.
const STALL_RESOLVE_MARGIN: i64 = 16;

/// Static table of stall-to-resolution rules.
const STALL_RULES: &[StallRule] = &[
    // DMA lock stalls (mem module) -> lock release (core module, same physical tile)
    StallRule {
        stall: "DMA_S2MM_0_STALLED_LOCK",
        stall_pkt: 1,
        resolve: &["INSTR_LOCK_RELEASE_REQ"],
        resolve_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    StallRule {
        stall: "DMA_S2MM_1_STALLED_LOCK",
        stall_pkt: 1,
        resolve: &["INSTR_LOCK_RELEASE_REQ"],
        resolve_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    StallRule {
        stall: "DMA_MM2S_0_STALLED_LOCK",
        stall_pkt: 1,
        resolve: &["INSTR_LOCK_RELEASE_REQ"],
        resolve_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    StallRule {
        stall: "DMA_MM2S_1_STALLED_LOCK",
        stall_pkt: 1,
        resolve: &["INSTR_LOCK_RELEASE_REQ"],
        resolve_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    // Stream starvation (mem) -> stream put (core)
    StallRule {
        stall: "DMA_S2MM_0_STREAM_STARVATION",
        stall_pkt: 1,
        resolve: &["INSTR_STREAM_PUT"],
        resolve_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    StallRule {
        stall: "DMA_S2MM_1_STREAM_STARVATION",
        stall_pkt: 1,
        resolve: &["INSTR_STREAM_PUT"],
        resolve_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    // Stream backpressure (mem) -> stream get (core)
    StallRule {
        stall: "DMA_MM2S_0_STREAM_BACKPRESSURE",
        stall_pkt: 1,
        resolve: &["INSTR_STREAM_GET"],
        resolve_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    StallRule {
        stall: "DMA_MM2S_1_STREAM_BACKPRESSURE",
        stall_pkt: 1,
        resolve: &["INSTR_STREAM_GET"],
        resolve_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    // Core LOCK_STALL -> DMA finished on mem module
    StallRule {
        stall: "LOCK_STALL",
        stall_pkt: 0,
        resolve: &[
            "DMA_S2MM_0_FINISHED_TASK",
            "DMA_MM2S_0_FINISHED_TASK",
            "DMA_S2MM_1_FINISHED_TASK",
            "DMA_MM2S_1_FINISHED_TASK",
        ],
        resolve_pkt: 1,
        rel: TileRelationship::SameTile,
    },
    // MemTile SEL variants -> core lock release (adjacent row).
    // MemTile trace packets use pkt_type=3 (PacketType.MEMTILE). Resolution
    // is on a Core tile (pkt_type=0) in an adjacent row (row 2 for row 1).
    StallRule {
        stall: "DMA_S2MM_SEL0_STALLED_LOCK",
        stall_pkt: 3,
        resolve: &["INSTR_LOCK_RELEASE_REQ"],
        resolve_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    StallRule {
        stall: "DMA_S2MM_SEL1_STALLED_LOCK",
        stall_pkt: 3,
        resolve: &["INSTR_LOCK_RELEASE_REQ"],
        resolve_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    StallRule {
        stall: "DMA_MM2S_SEL0_STALLED_LOCK",
        stall_pkt: 3,
        resolve: &["INSTR_LOCK_RELEASE_REQ"],
        resolve_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    StallRule {
        stall: "DMA_MM2S_SEL1_STALLED_LOCK",
        stall_pkt: 3,
        resolve: &["INSTR_LOCK_RELEASE_REQ"],
        resolve_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    // MemTile stream starvation -> core stream put (adjacent row)
    StallRule {
        stall: "DMA_S2MM_SEL0_STREAM_STARVATION",
        stall_pkt: 3,
        resolve: &["INSTR_STREAM_PUT"],
        resolve_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    StallRule {
        stall: "DMA_S2MM_SEL1_STREAM_STARVATION",
        stall_pkt: 3,
        resolve: &["INSTR_STREAM_PUT"],
        resolve_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    // MemTile stream backpressure -> core stream get (adjacent row)
    StallRule {
        stall: "DMA_MM2S_SEL0_STREAM_BACKPRESSURE",
        stall_pkt: 3,
        resolve: &["INSTR_STREAM_GET"],
        resolve_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    StallRule {
        stall: "DMA_MM2S_SEL1_STREAM_BACKPRESSURE",
        stall_pkt: 3,
        resolve: &["INSTR_STREAM_GET"],
        resolve_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
];

/// Check if two tiles match a given relationship.
fn tiles_match(stall: &TileKey, candidate: &TileKey, rel: TileRelationship) -> bool {
    match rel {
        TileRelationship::SameTile => stall.col == candidate.col && stall.row == candidate.row,
        TileRelationship::SameColAdjacentRow => {
            stall.col == candidate.col
                && ((stall.row as i16 - candidate.row as i16).abs() == 1)
        }
    }
}

/// Find the nearest resolving event at or after `after_abs` within margin.
///
/// `candidate_cycles_abs` must be sorted. Uses binary search.
fn find_resolution_event(
    after_abs: u64,
    candidate_cycles_abs: &[u64],
    margin: i64,
) -> Option<u64> {
    let idx = candidate_cycles_abs.partition_point(|&c| c < after_abs);
    if idx < candidate_cycles_abs.len() {
        let gap = candidate_cycles_abs[idx] as i64 - after_abs as i64;
        if gap.abs() <= margin {
            return Some(candidate_cycles_abs[idx]);
        }
    }
    // Also check the previous event (stall may end slightly after resolution).
    if idx > 0 {
        let gap = candidate_cycles_abs[idx - 1] as i64 - after_abs as i64;
        if gap.abs() <= margin {
            return Some(candidate_cycles_abs[idx - 1]);
        }
    }
    None
}

/// Build absolute cycle lists per (tile, event_name) from raw tile events.
///
/// Returns a map of (TileKey, event_name) -> sorted absolute cycle list.
fn build_absolute_event_map(
    tiles: &TileEvents,
    configs: &[(TileKey, &[String])],
) -> HashMap<(TileKey, String), Vec<u64>> {
    let mut map: HashMap<(TileKey, String), Vec<u64>> = HashMap::new();
    for (&key, events) in tiles {
        // Find the slot names for this tile.
        let names = configs
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, n)| *n)
            .unwrap_or(&[]);
        for ev in events {
            let name = slot_name(ev.slot, names);
            map.entry((key, name))
                .or_default()
                .push(ev.abs_cycle);
        }
    }
    // Sort all lists.
    for v in map.values_mut() {
        v.sort();
    }
    map
}

/// Run stall attribution across all tiles in a batch.
fn analyze_stall_attribution(
    hw_tiles: &TileEvents,
    emu_tiles: &TileEvents,
    config: &EventsConfig,
    tile_results: &[(TileKey, TileResult)],
) -> Vec<StallAttribution> {
    // Build tile -> slot_names mapping.
    let tile_configs: Vec<(TileKey, &[String])> = tile_results
        .iter()
        .map(|(key, _)| {
            let names: &[String] = if key.pkt_type == 3 && !config.memtile_events.is_empty() {
                &config.memtile_events
            } else if key.pkt_type == 0 {
                &config.core_events
            } else {
                &config.mem_events
            };
            (*key, names)
        })
        .collect();

    let hw_abs = build_absolute_event_map(hw_tiles, &tile_configs);
    let emu_abs = build_absolute_event_map(emu_tiles, &tile_configs);

    let all_keys: BTreeSet<TileKey> = hw_tiles
        .keys()
        .chain(emu_tiles.keys())
        .copied()
        .collect();

    let mut attributions = Vec::new();

    for rule in STALL_RULES {
        // Find stall tiles matching the rule's pkt_type.
        let stall_tiles: Vec<TileKey> = all_keys
            .iter()
            .filter(|k| k.pkt_type == rule.stall_pkt)
            .copied()
            .collect();

        for &stall_key in &stall_tiles {
            // Check if this tile has stall intervals (via level results).
            let tile_result = tile_results
                .iter()
                .find(|(k, _)| *k == stall_key);
            let stall_intervals = tile_result.and_then(|(_, tr)| {
                tr.level_results
                    .iter()
                    .find(|lr| lr.name == rule.stall)
            });
            if stall_intervals.is_none() {
                continue;
            }

            // Find candidate resolution tiles.
            let resolve_tiles: Vec<TileKey> = all_keys
                .iter()
                .filter(|k| {
                    k.pkt_type == rule.resolve_pkt
                        && tiles_match(&stall_key, k, rule.rel)
                })
                .copied()
                .collect();

            for &resolve_key in &resolve_tiles {
                // Get stall end cycles (absolute) from the tile events.
                let hw_stall_ivs = get_level_intervals_abs(
                    hw_tiles.get(&stall_key).map(|v| v.as_slice()).unwrap_or(&[]),
                    rule.stall,
                    tile_configs.iter().find(|(k, _)| *k == stall_key).map(|(_, n)| *n).unwrap_or(&[]),
                );
                let emu_stall_ivs = get_level_intervals_abs(
                    emu_tiles.get(&stall_key).map(|v| v.as_slice()).unwrap_or(&[]),
                    rule.stall,
                    tile_configs.iter().find(|(k, _)| *k == stall_key).map(|(_, n)| *n).unwrap_or(&[]),
                );

                // Get resolution event cycles (absolute).
                let mut hw_resolve_cycles: Vec<u64> = Vec::new();
                let mut emu_resolve_cycles: Vec<u64> = Vec::new();
                for &rname in rule.resolve {
                    if let Some(c) = hw_abs.get(&(resolve_key, rname.to_string())) {
                        hw_resolve_cycles.extend(c);
                    }
                    if let Some(c) = emu_abs.get(&(resolve_key, rname.to_string())) {
                        emu_resolve_cycles.extend(c);
                    }
                }
                hw_resolve_cycles.sort();
                emu_resolve_cycles.sort();

                if hw_stall_ivs.is_empty() && emu_stall_ivs.is_empty() {
                    continue;
                }

                // Match stall ends to resolution events.
                let hw_resolutions = resolve_stalls(&hw_stall_ivs, &hw_resolve_cycles, rule.resolve);
                let emu_resolutions = resolve_stalls(&emu_stall_ivs, &emu_resolve_cycles, rule.resolve);

                let paired = hw_resolutions.len().min(emu_resolutions.len());
                let gap_deltas: Vec<i64> = (0..paired)
                    .map(|i| hw_resolutions[i].gap - emu_resolutions[i].gap)
                    .collect();

                if !hw_resolutions.is_empty() || !emu_resolutions.is_empty() {
                    attributions.push(StallAttribution {
                        stall_name: rule.stall.to_string(),
                        stall_tile: stall_key,
                        resolve_tile: resolve_key,
                        hw_resolutions,
                        emu_resolutions,
                        gap_deltas,
                    });
                }
            }
        }
    }

    attributions
}

/// Get absolute-cycle intervals for a named level event on a tile.
fn get_level_intervals_abs(
    events: &[TileEvent],
    event_name: &str,
    slot_names: &[String],
) -> Vec<(u64, u64)> {
    // Find which slot(s) correspond to this event name.
    let target_slots: Vec<u8> = slot_names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.as_str() == event_name)
        .map(|(i, _)| i as u8)
        .collect();

    if target_slots.is_empty() {
        return Vec::new();
    }

    // Collect absolute cycles for the matching slot(s).
    let mut cycles: Vec<u64> = events
        .iter()
        .filter(|e| target_slots.contains(&e.slot))
        .map(|e| e.abs_cycle)
        .collect();
    cycles.sort();

    // Convert to intervals (consecutive cycles with gap <= 1).
    if cycles.is_empty() {
        return Vec::new();
    }
    let mut intervals = Vec::new();
    let mut start = cycles[0];
    let mut prev = cycles[0];
    for &c in &cycles[1..] {
        if c - prev > 1 {
            intervals.push((start, prev));
            start = c;
        }
        prev = c;
    }
    intervals.push((start, prev));
    intervals
}

/// Match stall interval ends to the nearest resolution event.
fn resolve_stalls(
    stall_ivs: &[(u64, u64)],
    resolve_cycles: &[u64],
    resolve_names: &[&str],
) -> Vec<StallResolution> {
    let mut results = Vec::new();
    for &(_, stall_end) in stall_ivs {
        if let Some(resolve_cycle) = find_resolution_event(stall_end, resolve_cycles, STALL_RESOLVE_MARGIN) {
            results.push(StallResolution {
                stall_end_abs: stall_end,
                resolve_abs: resolve_cycle,
                gap: resolve_cycle as i64 - stall_end as i64,
                resolve_name: resolve_names.first().unwrap_or(&"?").to_string(),
            });
        }
    }
    results
}

// ============================================================================
// Cross-Tile Correlation (Feature 3)
// ============================================================================

/// Static table of cross-tile correlation rules.
///
/// Includes both compute tile (pkt_type 1 = Mem module) and MemTile
/// (pkt_type 2) variants. Compute tile Mem module is rarely traced in
/// current configurations; MemTile rules use SEL naming and adjacent row.
const CORRELATION_RULES: &[CorrelationRule] = &[
    // --- Compute tile Mem module (pkt_type 1) -> Core (pkt_type 0), same tile ---
    // These will only match if mem module tracing is enabled.
    CorrelationRule {
        name: "DMA_START -> STREAM_GET",
        src_name: "DMA_MM2S_0_START_TASK",
        src_pkt: 1,
        dst_name: "INSTR_STREAM_GET",
        dst_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    CorrelationRule {
        name: "STREAM_PUT -> DMA_START",
        src_name: "INSTR_STREAM_PUT",
        src_pkt: 0,
        dst_name: "DMA_S2MM_0_START_TASK",
        dst_pkt: 1,
        rel: TileRelationship::SameTile,
    },
    CorrelationRule {
        name: "LOCK_RELEASE -> DMA_START",
        src_name: "INSTR_LOCK_RELEASE_REQ",
        src_pkt: 0,
        dst_name: "DMA_MM2S_0_START_TASK",
        dst_pkt: 1,
        rel: TileRelationship::SameTile,
    },
    CorrelationRule {
        name: "DMA_FINISH -> LOCK_ACQUIRE",
        src_name: "DMA_S2MM_0_FINISHED_TASK",
        src_pkt: 1,
        dst_name: "INSTR_LOCK_ACQUIRE_REQ",
        dst_pkt: 0,
        rel: TileRelationship::SameTile,
    },
    // --- MemTile (pkt_type 2) <-> Core (pkt_type 0), adjacent row ---
    //
    // NOTE: Cross-tile correlation only works within a single batch (sweep
    // batches are independent runs with different event configs). Rules are
    // designed to match event pairs that co-occur in the same batch config.
    //
    // Batch 0 pattern: MemTile DMA START + Core stall events
    // Batch 1 pattern: MemTile DMA FINISH + Core INSTR events (LOAD/STORE/STREAM_GET)
    // Batch 2 pattern: MemTile stall events + Core LOCK events

    // Batch 1: MemTile DMA finish -> Core instruction start (data delivery)
    CorrelationRule {
        name: "MT S2MM_FINISH -> STREAM_GET",
        src_name: "DMA_S2MM_SEL0_FINISHED_TASK",
        src_pkt: 3,
        dst_name: "INSTR_STREAM_GET",
        dst_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    CorrelationRule {
        name: "MT MM2S_FINISH -> STREAM_GET",
        src_name: "DMA_MM2S_SEL0_FINISHED_TASK",
        src_pkt: 3,
        dst_name: "INSTR_STREAM_GET",
        dst_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    CorrelationRule {
        name: "MT S2MM_FINISH -> LOAD",
        src_name: "DMA_S2MM_SEL0_FINISHED_TASK",
        src_pkt: 3,
        dst_name: "INSTR_LOAD",
        dst_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
    CorrelationRule {
        name: "MT MM2S_FINISH -> STORE",
        src_name: "DMA_MM2S_SEL0_FINISHED_TASK",
        src_pkt: 3,
        dst_name: "INSTR_STORE",
        dst_pkt: 0,
        rel: TileRelationship::SameColAdjacentRow,
    },
];

/// Run cross-tile correlation analysis.
fn analyze_cross_tile(
    hw_tiles: &TileEvents,
    emu_tiles: &TileEvents,
    config: &EventsConfig,
    tile_results: &[(TileKey, TileResult)],
) -> CrossTileResult {
    let tile_configs: Vec<(TileKey, &[String])> = tile_results
        .iter()
        .map(|(key, _)| {
            let names: &[String] = if key.pkt_type == 3 && !config.memtile_events.is_empty() {
                &config.memtile_events
            } else if key.pkt_type == 0 {
                &config.core_events
            } else {
                &config.mem_events
            };
            (*key, names)
        })
        .collect();

    let hw_abs = build_absolute_event_map(hw_tiles, &tile_configs);
    let emu_abs = build_absolute_event_map(emu_tiles, &tile_configs);

    let all_keys: BTreeSet<TileKey> = hw_tiles
        .keys()
        .chain(emu_tiles.keys())
        .copied()
        .collect();

    let mut correlations = Vec::new();

    for rule in CORRELATION_RULES {
        let src_tiles: Vec<TileKey> = all_keys
            .iter()
            .filter(|k| k.pkt_type == rule.src_pkt)
            .copied()
            .collect();

        for &src_key in &src_tiles {
            let dst_tiles: Vec<TileKey> = all_keys
                .iter()
                .filter(|k| {
                    k.pkt_type == rule.dst_pkt && tiles_match(&src_key, k, rule.rel)
                })
                .copied()
                .collect();

            for &dst_key in &dst_tiles {
                let hw_src = hw_abs
                    .get(&(src_key, rule.src_name.to_string()))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let hw_dst = hw_abs
                    .get(&(dst_key, rule.dst_name.to_string()))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let emu_src = emu_abs
                    .get(&(src_key, rule.src_name.to_string()))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let emu_dst = emu_abs
                    .get(&(dst_key, rule.dst_name.to_string()))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                let hw_pairs = pair_events(hw_src, hw_dst);
                let emu_pairs = pair_events(emu_src, emu_dst);

                if hw_pairs.is_empty() && emu_pairs.is_empty() {
                    continue;
                }

                let paired = hw_pairs.len().min(emu_pairs.len());
                let gap_deltas: Vec<i64> = (0..paired)
                    .map(|i| hw_pairs[i].gap - emu_pairs[i].gap)
                    .collect();

                correlations.push(CorrelationResult {
                    rule_name: rule.name.to_string(),
                    src_tile: src_key,
                    dst_tile: dst_key,
                    hw_pairs,
                    emu_pairs,
                    gap_deltas,
                });
            }
        }
    }

    CrossTileResult { correlations }
}

/// Pair source and destination events using two-pointer scan.
///
/// For each source event, finds the next destination event after it.
/// Both lists must be sorted by absolute cycle.
fn pair_events(src: &[u64], dst: &[u64]) -> Vec<CorrelationPair> {
    let mut pairs = Vec::new();
    let mut dst_idx = 0;

    for &s in src {
        // Advance dst pointer to first event at or after source.
        while dst_idx < dst.len() && dst[dst_idx] < s {
            dst_idx += 1;
        }
        if dst_idx >= dst.len() {
            break;
        }
        pairs.push(CorrelationPair {
            src_abs: s,
            dst_abs: dst[dst_idx],
            gap: dst[dst_idx] as i64 - s as i64,
        });
        dst_idx += 1; // Consume this destination event.
    }

    pairs
}

// ============================================================================
// PC-Anchored analysis (Task 7)
// ============================================================================

/// Candidate names for the perfcnt overflow event slot.
///
/// mlir-aie's trace injection may use either spelling depending on the
/// version of the event table. We accept both and pick the first match.
const PERFCNT_SLOT_NAMES: &[&str] = &["PERF_CNT_0", "PERF_CNT_0_EVENT"];

/// Return true if this event name is the perfcnt overflow sentinel.
fn is_perfcnt_event(name: &str) -> bool {
    PERFCNT_SLOT_NAMES.contains(&name)
}

/// Linearly interpolate a cycle estimate from a sorted perfcnt PC sequence.
///
/// `perfcnt_pcs` is a sorted slice of PC values at which the perfcnt
/// counter overflowed (one per overflow tick). Each tick corresponds to
/// exactly `period` cycles. A PC that falls between tick[i-1] and tick[i]
/// gets a fractional estimate by linear interpolation across that interval.
///
/// Returns 0 if `perfcnt_pcs` is empty or `pc` is before the first tick.
pub fn interpolate_cycle_from_perfcnt(pc: u64, perfcnt_pcs: &[u64], period: u64) -> u64 {
    if perfcnt_pcs.is_empty() {
        return 0;
    }
    // partition_point returns the index of the first element >= pc.
    let idx = perfcnt_pcs.partition_point(|&p| p < pc);
    if idx == 0 {
        // Before first tick: estimate 0.
        return 0;
    }
    if idx >= perfcnt_pcs.len() {
        // After last tick: clamp to last tick's cycle.
        return (perfcnt_pcs.len() as u64 - 1) * period;
    }
    let pc_below = perfcnt_pcs[idx - 1];
    let pc_above = perfcnt_pcs[idx];
    // Avoid division by zero if two ticks have the same PC (degenerate case).
    let span = pc_above.saturating_sub(pc_below).max(1);
    let frac = (pc - pc_below) as f64 / span as f64;
    ((idx as f64 - 1.0 + frac) * period as f64) as u64
}

/// Compare one tile's HW vs EMU mode-1 events using PC-set and multiset diff.
///
/// In mode-1 traces, `TileEvent.abs_cycle` carries the PC (program counter)
/// at which the traced event fired, not an absolute cycle timestamp. The
/// perfcnt overflow slot provides a deterministic PC clock: each overflow
/// at a known PC corresponds to one `period` of elapsed cycles. We use
/// those anchors to assign approximate cycle estimates to every other event.
///
/// Events with `abs_cycle == 0` are the "no-PC" sentinel emitted when the
/// trace packet carried no PC information (mode-2 / fallback frames). They
/// are excluded from the diff and counted separately.
pub fn compare_pc_anchored_for_tile(
    key: TileKey,
    hw_events: &[TileEvent],
    emu_events: &[TileEvent],
    slot_names: &[String],
) -> PCAnchoredReport {
    let mut report = PCAnchoredReport {
        pkt_type: key.pkt_type as u32,
        ..Default::default()
    };

    // Partition anchored (abs_cycle > 0) vs unanchored (abs_cycle == 0).
    let mut hw_anchored: Vec<&TileEvent> = Vec::new();
    let mut emu_anchored: Vec<&TileEvent> = Vec::new();
    for e in hw_events {
        if e.abs_cycle == 0 {
            report.unanchored_count_hw += 1;
        } else {
            hw_anchored.push(e);
        }
    }
    for e in emu_events {
        if e.abs_cycle == 0 {
            report.unanchored_count_emu += 1;
        } else {
            emu_anchored.push(e);
        }
    }

    // Group by event name (slot -> name lookup).
    let mut hw_by_name: HashMap<String, Vec<u64>> = HashMap::new();
    let mut emu_by_name: HashMap<String, Vec<u64>> = HashMap::new();
    for ev in &hw_anchored {
        let name = slot_name(ev.slot, slot_names);
        hw_by_name.entry(name).or_default().push(ev.abs_cycle);
    }
    for ev in &emu_anchored {
        let name = slot_name(ev.slot, slot_names);
        emu_by_name.entry(name).or_default().push(ev.abs_cycle);
    }

    // Sort each per-name list so set ops are deterministic.
    for v in hw_by_name.values_mut() {
        v.sort();
    }
    for v in emu_by_name.values_mut() {
        v.sort();
    }

    // Collect all event names seen in either side.
    let all_names: BTreeSet<String> = hw_by_name
        .keys()
        .chain(emu_by_name.keys())
        .cloned()
        .collect();

    // Locate perfcnt overflow PCs from HW side (HW is ground truth).
    // We use the first slot name that matches a known perfcnt spelling.
    let perfcnt_name: Option<String> = all_names
        .iter()
        .find(|n| is_perfcnt_event(n.as_str()))
        .cloned();

    let hw_perfcnt_pcs: Vec<u64> = perfcnt_name
        .as_ref()
        .and_then(|n| hw_by_name.get(n))
        .cloned()
        .unwrap_or_default();
    let emu_perfcnt_pcs: Vec<u64> = perfcnt_name
        .as_ref()
        .and_then(|n| emu_by_name.get(n))
        .cloned()
        .unwrap_or_default();

    // Estimate perfcnt period from the median inter-overflow PC distance.
    // Falls back to a wide default when there are fewer than 2 ticks.
    let period: u64 = estimate_perfcnt_period(&hw_perfcnt_pcs)
        .or_else(|| estimate_perfcnt_period(&emu_perfcnt_pcs))
        .unwrap_or(1024);

    // Set and multiset diff per event name.
    for name in &all_names {
        let hw_pcs: &[u64] = hw_by_name.get(name).map(|v| v.as_slice()).unwrap_or(&[]);
        let emu_pcs: &[u64] = emu_by_name.get(name).map(|v| v.as_slice()).unwrap_or(&[]);

        let hw_set: HashSet<u64> = hw_pcs.iter().copied().collect();
        let emu_set: HashSet<u64> = emu_pcs.iter().copied().collect();

        let hw_only: HashSet<u64> = hw_set.difference(&emu_set).copied().collect();
        let emu_only: HashSet<u64> = emu_set.difference(&hw_set).copied().collect();
        report
            .set_diff
            .insert(name.clone(), (hw_only, emu_only));

        // Per-PC multiset diff: count occurrences on each side.
        let union: HashSet<u64> = hw_set.union(&emu_set).copied().collect();
        let mut multiset: HashMap<u64, (u32, u32, i32)> = HashMap::new();
        for pc in union {
            let hw_count = hw_pcs.iter().filter(|&&p| p == pc).count() as u32;
            let emu_count = emu_pcs.iter().filter(|&&p| p == pc).count() as u32;
            let delta = hw_count as i32 - emu_count as i32;
            multiset.insert(pc, (hw_count, emu_count, delta));
        }
        report.multiset_diff.insert(name.clone(), multiset);

        // Cycle bands: skip perfcnt slot itself (it IS the clock reference).
        if perfcnt_name.as_deref() == Some(name.as_str()) {
            continue;
        }
        if hw_perfcnt_pcs.is_empty() && emu_perfcnt_pcs.is_empty() {
            continue;
        }

        // For every PC that appears in either side, interpolate cycle estimates.
        let all_pcs: HashSet<u64> = hw_pcs.iter().chain(emu_pcs.iter()).copied().collect();
        let tolerance = period / 2;
        let mut bands: HashMap<u64, CycleBand> = HashMap::new();
        for pc in all_pcs {
            let hw_est = interpolate_cycle_from_perfcnt(pc, &hw_perfcnt_pcs, period);
            let emu_est = interpolate_cycle_from_perfcnt(pc, &emu_perfcnt_pcs, period);
            let delta = hw_est as i64 - emu_est as i64;
            bands.insert(
                pc,
                CycleBand {
                    hw_cycle_est: hw_est,
                    emu_cycle_est: emu_est,
                    delta_cycles: delta,
                    exceeds_tolerance: delta.unsigned_abs() > tolerance,
                },
            );
        }
        if !bands.is_empty() {
            report.cycle_bands.insert(name.clone(), bands);
        }
    }

    report
}

/// Estimate the perfcnt overflow period from a sorted list of overflow PCs.
///
/// Takes the median inter-overflow PC distance. Returns `None` when fewer
/// than 2 ticks are available.
fn estimate_perfcnt_period(perfcnt_pcs: &[u64]) -> Option<u64> {
    if perfcnt_pcs.len() < 2 {
        return None;
    }
    let mut gaps: Vec<u64> = perfcnt_pcs
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]))
        .collect();
    gaps.sort();
    Some(gaps[gaps.len() / 2])
}

// ============================================================================
// Comparison (core)
// ============================================================================

/// Compare one tile's HW vs EMU event streams.
fn compare_tile_events(
    tile_key: TileKey,
    hw_events: &[TileEvent],
    emu_events: &[TileEvent],
    slot_names: &[String],
    opts: &AnalysisOptions,
) -> TileResult {
    let (hw_t0, emu_t0) = find_edge_anchor(hw_events, emu_events, slot_names);

    // Group events by slot, rebased to anchor.
    let mut hw_by_slot: HashMap<u8, Vec<i64>> = HashMap::new();
    let mut emu_by_slot: HashMap<u8, Vec<i64>> = HashMap::new();
    for ev in hw_events {
        hw_by_slot
            .entry(ev.slot)
            .or_default()
            .push(ev.abs_cycle as i64 - hw_t0 as i64);
    }
    for ev in emu_events {
        emu_by_slot
            .entry(ev.slot)
            .or_default()
            .push(ev.abs_cycle as i64 - emu_t0 as i64);
    }

    let all_slots: BTreeSet<u8> = hw_by_slot
        .keys()
        .chain(emu_by_slot.keys())
        .copied()
        .collect();

    let mut edge_results = Vec::new();
    let mut level_results = Vec::new();
    let mut iteration_results = Vec::new();

    for slot in all_slots {
        let name = slot_name(slot, slot_names);
        let hw_cycles = hw_by_slot.get(&slot).cloned().unwrap_or_default();
        let emu_cycles = emu_by_slot.get(&slot).cloned().unwrap_or_default();

        // Sort (should already be sorted from decode_per_tile, but be safe).
        let mut hw_sorted = hw_cycles;
        let mut emu_sorted = emu_cycles;
        hw_sorted.sort();
        emu_sorted.sort();

        if is_level_event(&name) {
            let hw_ivs = events_to_intervals(&hw_sorted);
            let emu_ivs = events_to_intervals(&emu_sorted);
            if opts.iterations {
                if let Some(ir) =
                    analyze_iterations_level(&name, tile_key, &hw_ivs, &emu_ivs)
                {
                    iteration_results.push(ir);
                }
            }
            level_results.push(analyze_level_event(name, &hw_ivs, &emu_ivs));
        } else {
            if opts.iterations {
                if let Some(ir) =
                    analyze_iterations_edge(&name, tile_key, &hw_sorted, &emu_sorted)
                {
                    iteration_results.push(ir);
                }
            }
            edge_results.push(analyze_edge_event(name, &hw_sorted, &emu_sorted));
        }
    }

    TileResult {
        hw_t0,
        emu_t0,
        edge_results,
        level_results,
        iteration_results,
    }
}

/// Compare one HW/EMU trace file pair.
pub fn compare_batch(
    hw_path: &Path,
    emu_path: &Path,
    config: &EventsConfig,
    batch_idx: usize,
) -> Result<BatchResult, String> {
    compare_batch_with_opts(hw_path, emu_path, config, batch_idx, &AnalysisOptions::default())
}

/// Compare one HW/EMU trace pair with extended analysis options.
///
/// Inputs are events-JSON paths produced by tools/parse-trace.py. If `config`
/// has any non-empty event list, it overrides the slot_names embedded in the
/// events JSON; otherwise the JSON's slot_names wins (HW-side takes priority
/// over EMU-side for name conflicts).
pub fn compare_batch_with_opts(
    hw_events_path: &Path,
    emu_events_path: &Path,
    config: &EventsConfig,
    batch_idx: usize,
    opts: &AnalysisOptions,
) -> Result<BatchResult, String> {
    let (hw_tiles, hw_config) = load_events_json(hw_events_path)?;
    let (emu_tiles, emu_config) = load_events_json(emu_events_path)?;

    // Resolve slot-name config: caller override > HW-side JSON > EMU-side JSON.
    let effective_config = if !config.core_events.is_empty()
        || !config.mem_events.is_empty()
        || !config.memtile_events.is_empty()
    {
        config.clone()
    } else if !hw_config.core_events.is_empty()
        || !hw_config.mem_events.is_empty()
        || !hw_config.memtile_events.is_empty()
    {
        hw_config
    } else {
        emu_config
    };
    let config = &effective_config;

    // When remap_columns is enabled, normalize physical columns to logical
    // 0-indexed so traces from different NPU column assignments can be compared.
    let (hw_tiles, emu_tiles) = if opts.remap_columns {
        (remap_tile_columns(&hw_tiles), remap_tile_columns(&emu_tiles))
    } else {
        (hw_tiles, emu_tiles)
    };

    let all_keys: BTreeSet<TileKey> =
        hw_tiles.keys().chain(emu_tiles.keys()).copied().collect();

    let mut tiles = Vec::new();
    for key in all_keys {
        let hw_ev = hw_tiles.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
        let emu_ev = emu_tiles.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
        let names = if key.pkt_type == 3 && !config.memtile_events.is_empty() {
            // MemTile (row 1) has its own event namespace
            &config.memtile_events
        } else if key.pkt_type == 0 {
            &config.core_events
        } else {
            &config.mem_events
        };
        let result = compare_tile_events(key, hw_ev, emu_ev, names, opts);
        tiles.push((key, result));
    }

    // Extended analyses (require cross-tile data).
    let stall_attributions = if opts.stalls {
        analyze_stall_attribution(&hw_tiles, &emu_tiles, config, &tiles)
    } else {
        Vec::new()
    };

    let cross_tile = if opts.cross_tile {
        Some(analyze_cross_tile(&hw_tiles, &emu_tiles, config, &tiles))
    } else {
        None
    };

    // PC-anchored analysis (core tiles only, pkt_type == 0).
    let pc_anchored = if opts.pc_anchored {
        let mut map = HashMap::new();
        for key in hw_tiles.keys().chain(emu_tiles.keys()).copied().collect::<BTreeSet<_>>() {
            // Only run on core tiles in mode-1 configuration.
            if key.pkt_type != 0 {
                continue;
            }
            let hw_ev = hw_tiles.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
            let emu_ev = emu_tiles.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
            let names = &config.core_events;
            let report = compare_pc_anchored_for_tile(key, hw_ev, emu_ev, names);
            map.insert(key, report);
        }
        map
    } else {
        HashMap::new()
    };

    Ok(BatchResult {
        batch_idx,
        config: config.clone(),
        tiles,
        stall_attributions,
        cross_tile,
        pc_anchored,
    })
}

// ============================================================================
// Report formatting
// ============================================================================

/// Count values in a sorted slice within [-threshold, +threshold].
fn count_within(sorted: &[i64], threshold: i64) -> usize {
    let lower = sorted.partition_point(|&d| d < -threshold);
    let upper = sorted.partition_point(|&d| d <= threshold);
    upper - lower
}

/// Format comparison results into a human-readable report.
///
/// Output format matches Python `trace-compare.py` so the bridge script
/// can parse it unchanged (greps for `Edge event types:` and `Pairs:`).
pub fn format_report(batch_results: &[BatchResult]) -> String {
    let mut out = String::new();

    let _ = writeln!(out, "{}", "=".repeat(76));
    let _ = writeln!(out, "Raw Binary Trace Comparison");
    let _ = writeln!(out, "  Alignment:  first shared edge event per tile");
    let _ = writeln!(
        out,
        "  Edge:       paired by occurrence index, divergence at |dt|>{}",
        DIVERGE_THRESHOLD
    );
    let _ = writeln!(out, "  Level:      compared by interval structure");
    let _ = writeln!(out, "{}", "=".repeat(76));
    let _ = writeln!(out);

    // Accumulators for aggregate summary.
    let mut all_edge_deltas_clean: Vec<i64> = Vec::new();
    let mut all_edge_deltas_total: Vec<i64> = Vec::new();
    let mut total_edge_clean: usize = 0;
    let mut total_edge_diverged: usize = 0;
    let mut total_edge_count_mismatch: usize = 0;
    let mut total_level_clean: usize = 0;
    let mut total_level_diverged: usize = 0;
    let mut total_level_count_mismatch: usize = 0;
    let mut divergence_details: Vec<(usize, String, String, usize, String)> = Vec::new();

    for batch in batch_results {
        let core_names = &batch.config.core_events;
        let mem_names = &batch.config.mem_events;
        let active_core: Vec<&str> = core_names
            .iter()
            .filter(|n| *n != "TRUE" && *n != "NONE")
            .map(|s| s.as_str())
            .collect();
        let active_mem: Vec<&str> = mem_names
            .iter()
            .filter(|n| *n != "TRUE" && *n != "NONE")
            .map(|s| s.as_str())
            .collect();

        let _ = writeln!(out, "--- Batch {} ---", batch.batch_idx);
        let _ = writeln!(
            out,
            "  Core: {}",
            if active_core.is_empty() {
                "(none)".to_string()
            } else {
                active_core.join(", ")
            }
        );
        let _ = writeln!(
            out,
            "  Mem:  {}",
            if active_mem.is_empty() {
                "(none)".to_string()
            } else {
                active_mem.join(", ")
            }
        );
        let _ = writeln!(out);

        for (key, tile_result) in &batch.tiles {
            let module = tile_module_name(key);
            let _ = writeln!(
                out,
                "  Tile ({},{}) {}  (anchor: HW cy {}, EMU cy {})",
                key.col, key.row, module, tile_result.hw_t0, tile_result.emu_t0,
            );

            // -- Edge events --
            for er in &tile_result.edge_results {
                if er.name == "TRUE" {
                    continue;
                }

                all_edge_deltas_total.extend_from_slice(&er.deltas);

                let count_ok = er.hw_count == er.emu_count;
                if !count_ok {
                    total_edge_count_mismatch += 1;
                }

                // Clean deltas = before divergence point.
                let clean_end = er.diverge_idx.unwrap_or(er.paired);
                let clean = &er.deltas[..clean_end];
                all_edge_deltas_clean.extend_from_slice(clean);

                if er.diverge_idx.is_some() {
                    total_edge_diverged += 1;
                } else {
                    total_edge_clean += 1;
                }

                // Format header line.
                let mut count_str = format!("{}/{}", er.hw_count, er.emu_count);
                if !count_ok {
                    count_str.push_str(" COUNTS DIFFER");
                }
                let status = if let Some(div) = er.diverge_idx {
                    format!("DIVERGES at #{}", div)
                } else {
                    "OK".to_string()
                };
                let _ = writeln!(
                    out,
                    "    [edge] {:<32} {:<20} {}",
                    er.name, count_str, status,
                );

                // Clean timing stats.
                if !clean.is_empty() {
                    let min_c = clean.iter().copied().min().unwrap();
                    let max_c = clean.iter().copied().max().unwrap();
                    let mean_c = clean.iter().sum::<i64>() as f64 / clean.len() as f64;
                    let _ = writeln!(
                        out,
                        "           Clean ({} pairs): min={:+} max={:+} mean={:+.1}",
                        clean.len(),
                        min_c,
                        max_c,
                        mean_c,
                    );
                }

                // Context samples.
                for &(idx, hw_c, emu_c, dt) in &er.samples {
                    let marker = if er.diverge_idx == Some(idx) {
                        "  <<< DIVERGENCE"
                    } else {
                        ""
                    };
                    let _ = writeln!(
                        out,
                        "           [{:>4}] HW={:<8} EMU={:<8} dt={:+}{}",
                        idx, hw_c, emu_c, dt, marker,
                    );
                }

                // Divergence annotation.
                if er.diverge_idx.is_some() {
                    let _ = writeln!(
                        out,
                        "           Drift pattern: {}",
                        er.drift_type
                    );
                    divergence_details.push((
                        batch.batch_idx,
                        format!("({},{}) {}", key.col, key.row, module),
                        er.name.clone(),
                        er.diverge_idx.unwrap(),
                        er.drift_type.clone(),
                    ));
                }
            }

            // -- Level events --
            for lr in &tile_result.level_results {
                if lr.name == "TRUE" {
                    continue;
                }

                let count_ok = lr.hw_intervals == lr.emu_intervals;
                if !count_ok {
                    total_level_count_mismatch += 1;
                }

                if lr.diverge_idx.is_some() {
                    total_level_diverged += 1;
                } else {
                    total_level_clean += 1;
                }

                let mut count_str = format!("{}/{} intervals", lr.hw_intervals, lr.emu_intervals);
                if !count_ok {
                    count_str.push_str(" DIFFER");
                }
                let status = if let Some(div) = lr.diverge_idx {
                    format!("DURATION DIVERGES at interval #{}", div)
                } else {
                    "OK".to_string()
                };
                let _ = writeln!(
                    out,
                    "    [level] {:<31} {:<20} {}",
                    lr.name, count_str, status,
                );

                // Clean duration stats.
                if !lr.dur_deltas.is_empty() {
                    let clean_end = lr.diverge_idx.unwrap_or(lr.paired);
                    let clean_dur = &lr.dur_deltas[..clean_end];
                    if !clean_dur.is_empty() {
                        let min_d = clean_dur.iter().copied().min().unwrap();
                        let max_d = clean_dur.iter().copied().max().unwrap();
                        let mean_d =
                            clean_dur.iter().sum::<i64>() as f64 / clean_dur.len() as f64;
                        let _ = writeln!(
                            out,
                            "            Clean durations ({}): min={:+} max={:+} mean={:+.1}",
                            clean_dur.len(),
                            min_d,
                            max_d,
                            mean_d,
                        );
                    }
                }

                // Interval samples.
                for &(idx, hw_s, hw_e, emu_s, emu_e) in &lr.samples {
                    let hw_d = hw_e - hw_s + 1;
                    let emu_d = emu_e - emu_s + 1;
                    let dd = hw_d - emu_d;
                    let marker = if lr.diverge_idx == Some(idx) {
                        "  <<< DIVERGENCE"
                    } else {
                        ""
                    };
                    let _ = writeln!(
                        out,
                        "            [{:>2}] HW={}-{} ({}cy)  EMU={}-{} ({}cy)  dt_dur={:+}{}",
                        idx, hw_s, hw_e, hw_d, emu_s, emu_e, emu_d, dd, marker,
                    );
                }

                if lr.diverge_idx.is_some() {
                    divergence_details.push((
                        batch.batch_idx,
                        format!("({},{}) {}", key.col, key.row, module),
                        lr.name.clone(),
                        lr.diverge_idx.unwrap(),
                        "duration".to_string(),
                    ));
                }
            }

            let _ = writeln!(out);
        }
    }

    // ---- Divergence summary ----
    if !divergence_details.is_empty() {
        let _ = writeln!(out, "{}", "=".repeat(76));
        let _ = writeln!(
            out,
            "Divergence Points (where HW and EMU first disagree)"
        );
        let _ = writeln!(out, "{}", "=".repeat(76));
        for (batch, tile, name, idx, drift) in &divergence_details {
            let _ = writeln!(
                out,
                "  Batch {}  {:<16} {:<34} #{:<5} ({})",
                batch, tile, name, idx, drift,
            );
        }
        let _ = writeln!(out);
    }

    // ---- Overall summary ----
    let _ = writeln!(out, "{}", "=".repeat(76));
    let _ = writeln!(out, "Summary");
    let _ = writeln!(out, "{}", "=".repeat(76));
    let _ = writeln!(out, "Batches:             {}", batch_results.len());
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Edge event types:    {} clean, {} diverged, {} count mismatch",
        total_edge_clean, total_edge_diverged, total_edge_count_mismatch,
    );
    let _ = writeln!(
        out,
        "Level event types:   {} clean, {} diverged, {} count mismatch",
        total_level_clean, total_level_diverged, total_level_count_mismatch,
    );

    // Timing stats for clean edge events.
    if !all_edge_deltas_clean.is_empty() {
        let mut sorted_c = all_edge_deltas_clean;
        sorted_c.sort();
        let n = sorted_c.len();
        let _ = writeln!(out);
        let _ = writeln!(out, "Edge timing (CLEAN pairs only -- before divergence):");
        let _ = writeln!(out, "  Pairs:           {}", n);
        let _ = writeln!(out, "  Min:             {:+}", sorted_c[0]);
        let _ = writeln!(out, "  Max:             {:+}", sorted_c[n - 1]);
        let _ = writeln!(
            out,
            "  Mean:            {:+.1}",
            sorted_c.iter().sum::<i64>() as f64 / n as f64
        );
        let _ = writeln!(out, "  Median:          {:+}", sorted_c[n / 2]);
        let _ = writeln!(out, "  Spread:          {}", sorted_c[n - 1] - sorted_c[0]);
        let _ = writeln!(out);
        for &threshold in &[0i64, 1, 2, 5, 10] {
            let within = count_within(&sorted_c, threshold);
            let pct = within as f64 / n as f64 * 100.0;
            let _ = writeln!(
                out,
                "  Within +/-{:>3}:    {:>4}/{} ({:.1}%)",
                threshold, within, n, pct,
            );
        }

        // Total stats (including post-divergence) if different from clean.
        all_edge_deltas_total.sort();
        if all_edge_deltas_total.len() != sorted_c.len() {
            let sorted_t = &all_edge_deltas_total;
            let nt = sorted_t.len();
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "Edge timing (ALL pairs including post-divergence):"
            );
            let _ = writeln!(out, "  Pairs:           {}", nt);
            let _ = writeln!(
                out,
                "  Spread:          {}",
                sorted_t[nt - 1] - sorted_t[0]
            );
            for &threshold in &[0i64, 1, 2, 5, 10, 50] {
                let within = count_within(sorted_t, threshold);
                let pct = within as f64 / nt as f64 * 100.0;
                let _ = writeln!(
                    out,
                    "  Within +/-{:>3}:    {:>4}/{} ({:.1}%)",
                    threshold, within, nt, pct,
                );
            }
        }
    }

    let _ = writeln!(out, "{}", "=".repeat(76));

    // ---- Extended analysis sections (appended after standard report) ----

    // Per-Iteration Breakdown
    let has_iterations = batch_results
        .iter()
        .any(|b| b.tiles.iter().any(|(_, tr)| !tr.iteration_results.is_empty()));
    if has_iterations {
        let _ = writeln!(out);
        let _ = writeln!(out, "{}", "=".repeat(76));
        let _ = writeln!(out, "Per-Iteration Breakdown");
        let _ = writeln!(out, "{}", "=".repeat(76));
        let _ = writeln!(out);

        for batch in batch_results {
            for (key, tile_result) in &batch.tiles {
                for ir in &tile_result.iteration_results {
                    let module = tile_module_name(key);
                    let _ = writeln!(
                        out,
                        "Tile ({},{}) {} -- {} ({} iterations, {})",
                        key.col, key.row, module, ir.name, ir.iteration_count,
                        ir.drift_classification,
                    );
                    let _ = writeln!(
                        out,
                        "  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}",
                        "Iter", "HW_period", "EMU_period", "dt_period", "cum_drift",
                    );
                    for t in &ir.timings {
                        let marker = if ir.first_anomaly == Some(t.iteration) {
                            "  <<< ANOMALY"
                        } else {
                            match &ir.drift_classification {
                                IterationDriftType::StepChange { at_iteration, .. }
                                    if *at_iteration == t.iteration =>
                                {
                                    "  <<< STEP"
                                }
                                _ => "",
                            }
                        };
                        let _ = writeln!(
                            out,
                            "  {:>4}  {:>10}  {:>10}  {:>+10}  {:>+10}{}",
                            t.iteration,
                            t.hw_period,
                            t.emu_period,
                            t.period_delta,
                            t.cumulative_drift,
                            marker,
                        );
                    }
                    let _ = writeln!(out);
                }
            }
        }
    }

    // Stall Attribution
    let has_stalls = batch_results
        .iter()
        .any(|b| !b.stall_attributions.is_empty());
    if has_stalls {
        let _ = writeln!(out, "{}", "=".repeat(76));
        let _ = writeln!(out, "Stall Attribution");
        let _ = writeln!(out, "{}", "=".repeat(76));
        let _ = writeln!(out);

        for batch in batch_results {
            if batch.stall_attributions.is_empty() {
                continue;
            }
            for sa in &batch.stall_attributions {
                let stall_mod = tile_module_name(&sa.stall_tile);
                let resolve_mod = tile_module_name(&sa.resolve_tile);
                let _ = writeln!(
                    out,
                    "{} on ({},{}) {} -> {} on ({},{}) {}",
                    sa.stall_name,
                    sa.stall_tile.col, sa.stall_tile.row, stall_mod,
                    sa.hw_resolutions.first().map(|r| r.resolve_name.as_str()).unwrap_or("?"),
                    sa.resolve_tile.col, sa.resolve_tile.row, resolve_mod,
                );
                let _ = writeln!(
                    out,
                    "  HW resolutions: {}  EMU resolutions: {}",
                    sa.hw_resolutions.len(),
                    sa.emu_resolutions.len(),
                );
                if !sa.hw_resolutions.is_empty() {
                    let hw_gaps: Vec<i64> = sa.hw_resolutions.iter().map(|r| r.gap).collect();
                    let _ = writeln!(
                        out,
                        "  HW gap:  mean={:.1}cy  min={}  max={}",
                        hw_gaps.iter().sum::<i64>() as f64 / hw_gaps.len() as f64,
                        hw_gaps.iter().min().unwrap(),
                        hw_gaps.iter().max().unwrap(),
                    );
                }
                if !sa.emu_resolutions.is_empty() {
                    let emu_gaps: Vec<i64> = sa.emu_resolutions.iter().map(|r| r.gap).collect();
                    let _ = writeln!(
                        out,
                        "  EMU gap: mean={:.1}cy  min={}  max={}",
                        emu_gaps.iter().sum::<i64>() as f64 / emu_gaps.len() as f64,
                        emu_gaps.iter().min().unwrap(),
                        emu_gaps.iter().max().unwrap(),
                    );
                }
                if !sa.gap_deltas.is_empty() {
                    let mean_delta =
                        sa.gap_deltas.iter().sum::<i64>() as f64 / sa.gap_deltas.len() as f64;
                    let direction = if mean_delta > 0.5 {
                        "EMU faster"
                    } else if mean_delta < -0.5 {
                        "EMU slower"
                    } else {
                        "matched"
                    };
                    let _ = writeln!(
                        out,
                        "  Delta:   mean={:+.1}cy ({})",
                        mean_delta, direction,
                    );
                }
                let _ = writeln!(out);
            }
        }
    }

    // Cross-Tile Correlation
    let has_cross_tile = batch_results
        .iter()
        .any(|b| b.cross_tile.as_ref().map_or(false, |ct| !ct.correlations.is_empty()));
    if has_cross_tile {
        let _ = writeln!(out, "{}", "=".repeat(76));
        let _ = writeln!(out, "Cross-Tile Correlation");
        let _ = writeln!(out, "{}", "=".repeat(76));
        let _ = writeln!(out);

        for batch in batch_results {
            if let Some(ct) = &batch.cross_tile {
                for cr in &ct.correlations {
                    let src_mod = tile_module_name(&cr.src_tile);
                    let dst_mod = tile_module_name(&cr.dst_tile);
                    let _ = writeln!(
                        out,
                        "{}  (Tile ({},{}) {} -> Tile ({},{}) {})",
                        cr.rule_name,
                        cr.src_tile.col, cr.src_tile.row, src_mod,
                        cr.dst_tile.col, cr.dst_tile.row, dst_mod,
                    );

                    let hw_count = cr.hw_pairs.len();
                    let emu_count = cr.emu_pairs.len();
                    let _ = writeln!(
                        out,
                        "  Pairs: HW={}, EMU={}",
                        hw_count, emu_count,
                    );

                    if !cr.hw_pairs.is_empty() {
                        let hw_gaps: Vec<i64> = cr.hw_pairs.iter().map(|p| p.gap).collect();
                        let mean = hw_gaps.iter().sum::<i64>() as f64 / hw_gaps.len() as f64;
                        let _ = writeln!(
                            out,
                            "  HW gap:  mean={:.1}cy  min={}  max={}",
                            mean,
                            hw_gaps.iter().min().unwrap(),
                            hw_gaps.iter().max().unwrap(),
                        );
                    }
                    if !cr.emu_pairs.is_empty() {
                        let emu_gaps: Vec<i64> = cr.emu_pairs.iter().map(|p| p.gap).collect();
                        let mean = emu_gaps.iter().sum::<i64>() as f64 / emu_gaps.len() as f64;
                        let _ = writeln!(
                            out,
                            "  EMU gap: mean={:.1}cy  min={}  max={}",
                            mean,
                            emu_gaps.iter().min().unwrap(),
                            emu_gaps.iter().max().unwrap(),
                        );
                    }
                    if !cr.gap_deltas.is_empty() {
                        let mean_delta =
                            cr.gap_deltas.iter().sum::<i64>() as f64 / cr.gap_deltas.len() as f64;
                        let direction = if mean_delta > 0.5 {
                            "EMU faster"
                        } else if mean_delta < -0.5 {
                            "EMU slower"
                        } else {
                            "matched"
                        };
                        let _ = writeln!(
                            out,
                            "  Delta:   mean={:+.1}cy ({}, {} pairs compared)",
                            mean_delta, direction, cr.gap_deltas.len(),
                        );
                    }
                    let _ = writeln!(out);
                }
            }
        }
    }

    out
}

/// Human-readable module name from tile key.
///
/// Trace packet pkt_type field (per mlir-aie PacketType enum):
///   0 = Core module
///   1 = Mem module (compute tile DMA/locks -- rarely traced)
///   2 = ShimTile module (row 0)
///   3 = MemTile module (row 1 in NPU1)
fn tile_module_name(key: &TileKey) -> &'static str {
    match key.pkt_type {
        0 => "Core",
        1 => "Mem",
        2 => "ShimTile",
        3 => "MemTile",
        _ => "?",
    }
}

// ============================================================================
// Sweep directory support
// ============================================================================

#[derive(Deserialize)]
struct SweepManifest {
    batches: Vec<SweepBatchInfo>,
}

#[derive(Deserialize)]
struct SweepBatchInfo {
    batch: usize,
    #[serde(default)]
    status: String,
    #[serde(default)]
    hw_status: String,
    #[serde(default)]
    emu_status: String,
}

/// Compare all batches in a trace-sweep.py output directory.
///
/// Reads `sweep-manifest.json`, iterates successful batches, and produces
/// a unified comparison report.
pub fn compare_sweep_dir(sweep_dir: &Path) -> Result<String, String> {
    compare_sweep_dir_with_opts(sweep_dir, &AnalysisOptions::default())
}

/// Compare all batches with extended analysis options.
pub fn compare_sweep_dir_with_opts(
    sweep_dir: &Path,
    opts: &AnalysisOptions,
) -> Result<String, String> {
    let manifest_path = sweep_dir.join("sweep-manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("read {}: {}", manifest_path.display(), e))?;
    let manifest: SweepManifest = serde_json::from_str(&manifest_text)
        .map_err(|e| format!("parse manifest: {}", e))?;

    let mut batch_results = Vec::new();

    for info in &manifest.batches {
        if info.status != "ok" || info.hw_status != "ok" || info.emu_status != "ok" {
            continue;
        }

        let batch_dir = sweep_dir.join(format!("batch_{:02}", info.batch));
        // Sweep mode expects per-batch events JSONs produced upstream
        // (tools/parse-trace.py). The legacy layout put raw .bin files here;
        // the sweep orchestrator now writes .events.json alongside.
        let hw_events = batch_dir.join("hw").join("trace.events.json");
        let emu_events = batch_dir.join("emu").join("trace.events.json");

        if !hw_events.exists() || !emu_events.exists() {
            continue;
        }

        // Legacy aiecc events.json (slot_names override) may still exist per
        // batch; prefer it over the slot_names embedded in the events JSON
        // when present, since it matches what aiecc itself emitted.
        let events_json = batch_dir.join("events.json");
        let config = if events_json.exists() {
            let text = fs::read_to_string(&events_json)
                .map_err(|e| format!("read {}: {}", events_json.display(), e))?;
            serde_json::from_str(&text)
                .map_err(|e| format!("parse {}: {}", events_json.display(), e))?
        } else {
            EventsConfig::default()
        };

        match compare_batch_with_opts(&hw_events, &emu_events, &config, info.batch, opts) {
            Ok(result) => batch_results.push(result),
            Err(e) => {
                eprintln!("Warning: batch {} failed: {}", info.batch, e);
            }
        }
    }

    Ok(format_report(&batch_results))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_events_to_intervals() {
        let cycles = vec![10, 11, 12, 20, 21];
        let ivs = events_to_intervals(&cycles);
        assert_eq!(ivs, vec![(10, 12), (20, 21)]);
    }

    #[test]
    fn test_events_to_intervals_empty() {
        assert!(events_to_intervals(&[]).is_empty());
    }

    #[test]
    fn test_events_to_intervals_single() {
        assert_eq!(events_to_intervals(&[42]), vec![(42, 42)]);
    }

    #[test]
    fn test_analyze_edge_event_clean() {
        let hw = vec![100, 200, 300];
        let emu = vec![100, 201, 301];
        let result = analyze_edge_event("TEST".to_string(), &hw, &emu);
        assert_eq!(result.paired, 3);
        assert_eq!(result.deltas, vec![0, -1, -1]);
        assert!(result.diverge_idx.is_none());
        assert_eq!(result.drift_type, "none");
    }

    #[test]
    fn test_analyze_edge_event_diverges() {
        let hw = vec![100, 200, 300, 400];
        let emu = vec![100, 200, 250, 350]; // delta at [2] = 50
        let result = analyze_edge_event("TEST".to_string(), &hw, &emu);
        assert_eq!(result.diverge_idx, Some(2));
        assert_eq!(result.deltas[2], 50);
    }

    #[test]
    fn test_slot_name_with_names() {
        let names = vec!["INSTR_VECTOR".to_string(), "MEMORY_STALL".to_string()];
        assert_eq!(slot_name(0, &names), "INSTR_VECTOR");
        assert_eq!(slot_name(1, &names), "MEMORY_STALL");
        assert_eq!(slot_name(2, &names), "slot2");
    }

    #[test]
    fn test_is_level_event() {
        assert!(is_level_event("ACTIVE"));
        assert!(is_level_event("PORT_RUNNING_0"));
        assert!(is_level_event("DMA_S2MM_0_STALLED_LOCK"));
        assert!(is_level_event("DMA_S2MM_SEL0_STALLED_LOCK"));
        assert!(is_level_event("DMA_MM2S_SEL1_STREAM_BACKPRESSURE"));
        assert!(!is_level_event("INSTR_VECTOR"));
        assert!(!is_level_event("slot3"));
    }

    #[test]
    fn test_count_within() {
        let sorted = vec![-5, -2, -1, 0, 0, 1, 3, 10];
        assert_eq!(count_within(&sorted, 0), 2); // just the 0s
        assert_eq!(count_within(&sorted, 1), 4); // -1, 0, 0, 1
        assert_eq!(count_within(&sorted, 2), 5); // -2, -1, 0, 0, 1
        assert_eq!(count_within(&sorted, 5), 7); // all except 10
        assert_eq!(count_within(&sorted, 10), 8); // all
    }

    // ---- Per-Iteration Breakdown tests ----

    #[test]
    fn test_classify_iteration_drift_stable() {
        let timings = vec![
            IterationTiming { iteration: 1, hw_period: 25, emu_period: 25, period_delta: 0, cumulative_drift: 0 },
            IterationTiming { iteration: 2, hw_period: 25, emu_period: 26, period_delta: -1, cumulative_drift: -1 },
            IterationTiming { iteration: 3, hw_period: 25, emu_period: 24, period_delta: 1, cumulative_drift: 0 },
        ];
        assert!(matches!(classify_iteration_drift(&timings), IterationDriftType::Stable));
    }

    #[test]
    fn test_classify_iteration_drift_step_change() {
        let timings = vec![
            IterationTiming { iteration: 1, hw_period: 25, emu_period: 25, period_delta: 0, cumulative_drift: 0 },
            IterationTiming { iteration: 2, hw_period: 25, emu_period: 25, period_delta: 0, cumulative_drift: 0 },
            IterationTiming { iteration: 3, hw_period: 25, emu_period: 75, period_delta: -50, cumulative_drift: -50 },
            IterationTiming { iteration: 4, hw_period: 25, emu_period: 25, period_delta: 0, cumulative_drift: -50 },
        ];
        match classify_iteration_drift(&timings) {
            IterationDriftType::StepChange { at_iteration, magnitude } => {
                assert_eq!(at_iteration, 3);
                assert_eq!(magnitude, -50);
            }
            other => panic!("Expected StepChange, got {}", other),
        }
    }

    #[test]
    fn test_analyze_iterations_edge_too_few() {
        let tile = TileKey { col: 1, row: 2, pkt_type: 0 };
        let hw = vec![100, 200, 300]; // only 3 pairs, below MIN_ITERATION_PAIRS
        let emu = vec![100, 200, 300];
        assert!(analyze_iterations_edge("TEST", tile, &hw, &emu).is_none());
    }

    #[test]
    fn test_analyze_iterations_edge_stable() {
        let tile = TileKey { col: 1, row: 2, pkt_type: 0 };
        let hw = vec![100, 125, 150, 175, 200];
        let emu = vec![100, 126, 151, 176, 201];
        let result = analyze_iterations_edge("TEST", tile, &hw, &emu).unwrap();
        assert_eq!(result.iteration_count, 5);
        assert_eq!(result.timings.len(), 4);
        assert!(matches!(result.drift_classification, IterationDriftType::Stable));
        assert!(result.first_anomaly.is_none());
    }

    #[test]
    fn test_analyze_iterations_level() {
        let tile = TileKey { col: 1, row: 2, pkt_type: 1 };
        // Intervals with consistent spacing.
        let hw = vec![(10, 15), (30, 35), (50, 55), (70, 75)];
        let emu = vec![(10, 15), (30, 35), (50, 55), (70, 75)];
        let result = analyze_iterations_level("DMA_STALL", tile, &hw, &emu).unwrap();
        assert_eq!(result.iteration_count, 4);
        assert!(matches!(result.drift_classification, IterationDriftType::Stable));
    }

    // ---- Stall Attribution tests ----

    #[test]
    fn test_tiles_match_same_tile() {
        let a = TileKey { col: 1, row: 2, pkt_type: 0 };
        let b = TileKey { col: 1, row: 2, pkt_type: 1 };
        assert!(tiles_match(&a, &b, TileRelationship::SameTile));
        let c = TileKey { col: 1, row: 3, pkt_type: 1 };
        assert!(!tiles_match(&a, &c, TileRelationship::SameTile));
    }

    #[test]
    fn test_tiles_match_adjacent_row() {
        let a = TileKey { col: 1, row: 1, pkt_type: 3 };
        let b = TileKey { col: 1, row: 2, pkt_type: 0 };
        assert!(tiles_match(&a, &b, TileRelationship::SameColAdjacentRow));
        let c = TileKey { col: 1, row: 3, pkt_type: 0 };
        assert!(!tiles_match(&a, &c, TileRelationship::SameColAdjacentRow));
        let d = TileKey { col: 2, row: 2, pkt_type: 0 };
        assert!(!tiles_match(&a, &d, TileRelationship::SameColAdjacentRow));
    }

    #[test]
    fn test_find_resolution_event() {
        let cycles = vec![100, 200, 300, 400];
        // Exact match.
        assert_eq!(find_resolution_event(200, &cycles, STALL_RESOLVE_MARGIN), Some(200));
        // Within margin (after).
        assert_eq!(find_resolution_event(195, &cycles, STALL_RESOLVE_MARGIN), Some(200));
        // Within margin (before -- check previous).
        assert_eq!(find_resolution_event(205, &cycles, STALL_RESOLVE_MARGIN), Some(200));
        // Too far from any.
        assert_eq!(find_resolution_event(250, &cycles, STALL_RESOLVE_MARGIN), None);
    }

    #[test]
    fn test_resolve_stalls() {
        let stall_ivs = vec![(10, 20), (50, 60)];
        let resolve_cycles = vec![21, 62];
        let results = resolve_stalls(&stall_ivs, &resolve_cycles, &["LOCK_RELEASE"]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].stall_end_abs, 20);
        assert_eq!(results[0].resolve_abs, 21);
        assert_eq!(results[0].gap, 1);
        assert_eq!(results[1].stall_end_abs, 60);
        assert_eq!(results[1].resolve_abs, 62);
        assert_eq!(results[1].gap, 2);
    }

    // ---- Cross-Tile Correlation tests ----

    #[test]
    fn test_pair_events_basic() {
        let src = vec![10, 30, 50];
        let dst = vec![15, 35, 55];
        let pairs = pair_events(&src, &dst);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].gap, 5);
        assert_eq!(pairs[1].gap, 5);
        assert_eq!(pairs[2].gap, 5);
    }

    #[test]
    fn test_pair_events_uneven() {
        let src = vec![10, 30, 50, 70]; // 4 sources
        let dst = vec![15, 55]; // only 2 destinations
        let pairs = pair_events(&src, &dst);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].src_abs, 10);
        assert_eq!(pairs[0].dst_abs, 15);
        assert_eq!(pairs[1].src_abs, 30);
        assert_eq!(pairs[1].dst_abs, 55);
    }

    #[test]
    fn test_pair_events_empty() {
        assert!(pair_events(&[], &[10, 20]).is_empty());
        assert!(pair_events(&[10, 20], &[]).is_empty());
    }

    // ---- AnalysisOptions tests ----

    #[test]
    fn test_analysis_options_defaults() {
        let opts = AnalysisOptions::default();
        assert!(!opts.iterations);
        assert!(!opts.stalls);
        assert!(!opts.cross_tile);
        assert!(!opts.pc_anchored);
        assert!(!opts.any_enabled());
    }

    #[test]
    fn test_analysis_options_extended() {
        let opts = AnalysisOptions::extended();
        assert!(opts.iterations);
        assert!(opts.stalls);
        assert!(opts.cross_tile);
        assert!(!opts.pc_anchored); // pc_anchored is opt-in via --pc-anchored, not --extended
        assert!(opts.any_enabled());
    }

    #[test]
    fn test_analysis_options_pc_anchored_any_enabled() {
        let opts = AnalysisOptions {
            pc_anchored: true,
            ..Default::default()
        };
        assert!(opts.any_enabled());
    }

    #[test]
    fn test_format_report_summary_line() {
        // Verify the report contains the summary line the bridge script parses.
        let batch = BatchResult {
            batch_idx: 0,
            config: EventsConfig::default(),
            tiles: Vec::new(),
            stall_attributions: Vec::new(),
            cross_tile: None,
            pc_anchored: HashMap::new(),
        };
        let report = format_report(&[batch]);
        assert!(report.contains("Edge event types:"));
        assert!(report.contains("Level event types:"));
        assert!(report.contains("Batches:             1"));
    }

    // ---- PC-Anchored analysis tests (Task 7) ----

    #[test]
    fn pc_anchored_set_diff_finds_hw_only_and_emu_only() {
        // HW: INSTR_VECTOR fires at PCs {100, 200, 300}
        // EMU: INSTR_VECTOR fires at PCs {100, 250, 300}
        // Expected hw_only={200}, emu_only={250}.
        let hw_events = vec![
            TileEvent { slot: 0, abs_cycle: 100 },
            TileEvent { slot: 0, abs_cycle: 200 },
            TileEvent { slot: 0, abs_cycle: 300 },
        ];
        let emu_events = vec![
            TileEvent { slot: 0, abs_cycle: 100 },
            TileEvent { slot: 0, abs_cycle: 250 },
            TileEvent { slot: 0, abs_cycle: 300 },
        ];
        let names = vec!["INSTR_VECTOR".to_string()];
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        let report = compare_pc_anchored_for_tile(key, &hw_events, &emu_events, &names);
        let (hw_only, emu_only) = &report.set_diff["INSTR_VECTOR"];
        assert_eq!(*hw_only, HashSet::from([200u64]));
        assert_eq!(*emu_only, HashSet::from([250u64]));
    }

    #[test]
    fn pc_anchored_multiset_diff_counts_repeated_pcs() {
        // PC 100 fires twice on HW, once on EMU -> delta = +1.
        // PC 200 fires once on each -> delta = 0.
        let hw_events = vec![
            TileEvent { slot: 0, abs_cycle: 100 },
            TileEvent { slot: 0, abs_cycle: 100 },
            TileEvent { slot: 0, abs_cycle: 200 },
        ];
        let emu_events = vec![
            TileEvent { slot: 0, abs_cycle: 100 },
            TileEvent { slot: 0, abs_cycle: 200 },
        ];
        let names = vec!["INSTR_VECTOR".to_string()];
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        let report = compare_pc_anchored_for_tile(key, &hw_events, &emu_events, &names);
        let ms = &report.multiset_diff["INSTR_VECTOR"];
        let (hw_c, emu_c, delta) = ms[&100];
        assert_eq!(hw_c, 2);
        assert_eq!(emu_c, 1);
        assert_eq!(delta, 1);
        let (hw_c2, emu_c2, delta2) = ms[&200];
        assert_eq!(hw_c2, 1);
        assert_eq!(emu_c2, 1);
        assert_eq!(delta2, 0);
    }

    #[test]
    fn pc_anchored_unanchored_events_excluded_from_diff() {
        // Events with abs_cycle == 0 are the "no-PC" sentinel: they should
        // accumulate into unanchored_count_* and NOT appear in set_diff.
        let hw_events = vec![
            TileEvent { slot: 0, abs_cycle: 0 },   // unanchored
            TileEvent { slot: 0, abs_cycle: 100 },
        ];
        let emu_events = vec![
            TileEvent { slot: 0, abs_cycle: 0 },   // unanchored
            TileEvent { slot: 0, abs_cycle: 0 },   // unanchored
            TileEvent { slot: 0, abs_cycle: 100 },
        ];
        let names = vec!["INSTR_VECTOR".to_string()];
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        let report = compare_pc_anchored_for_tile(key, &hw_events, &emu_events, &names);
        assert_eq!(report.unanchored_count_hw, 1);
        assert_eq!(report.unanchored_count_emu, 2);
        // Only PC 100 should appear; zero should not be in set_diff.
        let (hw_only, emu_only) = &report.set_diff["INSTR_VECTOR"];
        assert!(hw_only.is_empty(), "no hw-only PCs expected");
        assert!(emu_only.is_empty(), "no emu-only PCs expected");
    }

    #[test]
    fn cycle_band_linear_interpolation() {
        // perfcnt PCs at [10, 60, 110] correspond to cycles [0, 50, 100] (period=50).
        // PC 35 is between tick[0]=10 and tick[1]=60.
        // Linear: cycle ~= 0 + 50 * (35-10)/(60-10) = 25.
        let perfcnt_pcs = vec![10u64, 60, 110];
        let period = 50u64;
        let est = interpolate_cycle_from_perfcnt(35, &perfcnt_pcs, period);
        assert_eq!(est, 25);
        // PC 85 is between tick[1]=60 and tick[2]=110.
        // Linear: cycle ~= 50 + 50 * (85-60)/(110-60) = 75.
        let est2 = interpolate_cycle_from_perfcnt(85, &perfcnt_pcs, period);
        assert_eq!(est2, 75);
    }

    #[test]
    fn cycle_band_boundary_cases() {
        let perfcnt_pcs = vec![10u64, 60, 110];
        let period = 50u64;
        // Exactly at tick[0]: returns 0 cycles.
        assert_eq!(interpolate_cycle_from_perfcnt(10, &perfcnt_pcs, period), 0);
        // Exactly at tick[1]: returns 1 * 50 = 50 cycles.
        assert_eq!(interpolate_cycle_from_perfcnt(60, &perfcnt_pcs, period), 50);
        // Before first tick: returns 0.
        assert_eq!(interpolate_cycle_from_perfcnt(5, &perfcnt_pcs, period), 0);
        // After last tick: clamps to (len-1)*period = 100.
        assert_eq!(interpolate_cycle_from_perfcnt(200, &perfcnt_pcs, period), 100);
        // Empty perfcnt: always 0.
        assert_eq!(interpolate_cycle_from_perfcnt(50, &[], period), 0);
    }

    #[test]
    fn pc_anchored_cycle_bands_with_perfcnt_anchor() {
        // Perfcnt slot (slot 1) fires at PCs [0x100, 0x200] -> period ~256.
        // INSTR_VECTOR (slot 0) fires at PC 0x180 on HW, PC 0x180 on EMU.
        // Both traces have the same perfcnt PCs -> delta_cycles == 0.
        let hw_events = vec![
            TileEvent { slot: 1, abs_cycle: 0x100 }, // perfcnt tick 0
            TileEvent { slot: 1, abs_cycle: 0x200 }, // perfcnt tick 1
            TileEvent { slot: 0, abs_cycle: 0x180 }, // INSTR_VECTOR
        ];
        let emu_events = vec![
            TileEvent { slot: 1, abs_cycle: 0x100 }, // perfcnt tick 0
            TileEvent { slot: 1, abs_cycle: 0x200 }, // perfcnt tick 1
            TileEvent { slot: 0, abs_cycle: 0x180 }, // INSTR_VECTOR
        ];
        let names = vec!["INSTR_VECTOR".to_string(), "PERF_CNT_0".to_string()];
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        let report = compare_pc_anchored_for_tile(key, &hw_events, &emu_events, &names);
        // INSTR_VECTOR set diff should be empty (both see 0x180).
        let (hw_only, emu_only) = &report.set_diff["INSTR_VECTOR"];
        assert!(hw_only.is_empty());
        assert!(emu_only.is_empty());
        // Cycle band for INSTR_VECTOR at 0x180: delta_cycles should be 0.
        let bands = report.cycle_bands.get("INSTR_VECTOR").expect("cycle_bands populated");
        let band = bands[&0x180];
        assert_eq!(band.delta_cycles, 0);
        assert!(!band.exceeds_tolerance);
    }

    #[test]
    fn pc_anchored_perfcnt_slot_name_variant() {
        // Verify both spellings of the perfcnt event name are recognized.
        assert!(is_perfcnt_event("PERF_CNT_0"));
        assert!(is_perfcnt_event("PERF_CNT_0_EVENT"));
        assert!(!is_perfcnt_event("INSTR_VECTOR"));
        assert!(!is_perfcnt_event("PERF_CNT_1"));
    }

    #[test]
    fn estimate_perfcnt_period_median() {
        // Gaps: [90, 100, 110] -> sorted -> median = 100.
        let pcs = vec![0u64, 90, 190, 300];
        let period = estimate_perfcnt_period(&pcs).unwrap();
        // Gaps: 90, 100, 110 -> median (idx 1 of 3) = 100.
        assert_eq!(period, 100);
    }
}
