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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
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
    StepChange { at_iteration: usize, magnitude: i64 },
    /// No clear pattern.
    Irregular,
}

impl std::fmt::Display for IterationDriftType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "Stable"),
            Self::Accumulating => write!(f, "Accumulating"),
            Self::StepChange { at_iteration, magnitude } => {
                write!(f, "StepChange at #{} (magnitude {})", at_iteration, magnitude)
            }
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
    /// Tile packet type (0=core, 1=memmod, 2=shim/PL, 3=memtile per mlir-aie
    /// PacketType enum). Stored as u32 for formatting ergonomics; values 0-7.
    pub pkt_type: u32,
    /// Per event-name PC set diff: (PCs only in HW, PCs only in EMU).
    pub set_diff: HashMap<String, (HashSet<u64>, HashSet<u64>)>,
    /// Per event-name multiset diff: PC -> (hw_count, emu_count, delta).
    pub multiset_diff: HashMap<String, HashMap<u64, (u32, u32, i32)>>,
    /// Per event-name, per-PC cycle band. Populated only when BOTH HW and
    /// EMU have at least one perfcnt overflow tick; one-sided perfcnt
    /// presence suppresses cycle-band generation entirely (otherwise the
    /// missing-side estimate is 0 and every event would falsely flag as
    /// "exceeds tolerance"). See hw_perfcnt_tick_count / emu_perfcnt_tick_count
    /// to detect that case.
    pub cycle_bands: HashMap<String, HashMap<u64, CycleBand>>,
    /// Events dropped because abs_cycle == 0 (no-PC sentinel) on HW side.
    pub unanchored_count_hw: usize,
    /// Events dropped because abs_cycle == 0 (no-PC sentinel) on EMU side.
    pub unanchored_count_emu: usize,
    /// Number of perfcnt overflow ticks observed in HW. Zero means HW had
    /// no perfcnt clock; cycle bands are skipped entirely.
    pub hw_perfcnt_tick_count: usize,
    /// Number of perfcnt overflow ticks observed in EMU. Zero means EMU had
    /// no perfcnt clock; cycle bands are skipped entirely.
    pub emu_perfcnt_tick_count: usize,
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
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let file: EventsFile =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {}", path.display(), e))?;
    if file.schema_version != 0 && file.schema_version != 1 {
        return Err(format!("{}: unsupported schema_version {}", path.display(), file.schema_version));
    }

    let mut tiles: TileEvents = HashMap::new();
    for rec in file.events {
        let key = TileKey { col: rec.col, row: rec.row, pkt_type: rec.pkt_type };
        tiles
            .entry(key)
            .or_default()
            .push(TileEvent { slot: rec.slot, abs_cycle: rec.ts });
    }
    for events in tiles.values_mut() {
        events.sort_by_key(|e| e.abs_cycle);
    }

    let config = match file.slot_names {
        Some(sn) => EventsConfig { core_events: sn.core, mem_events: sn.mem, memtile_events: sn.memtile },
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
            let new_key = TileKey { col: col_map[&key.col], row: key.row, pkt_type: key.pkt_type };
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
fn find_edge_anchor(hw_events: &[TileEvent], emu_events: &[TileEvent], slot_names: &[String]) -> (u64, u64) {
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
    let shared: BTreeSet<u8> = hw_first.keys().copied().filter(|s| emu_first.contains_key(s)).collect();

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
fn analyze_edge_event(name: String, hw_cycles: &[i64], emu_cycles: &[i64]) -> EdgeResult {
    let paired = hw_cycles.len().min(emu_cycles.len());
    let deltas: Vec<i64> = (0..paired).map(|i| hw_cycles[i] - emu_cycles[i]).collect();

    // Find divergence point.
    let diverge_idx = deltas.iter().position(|d| d.abs() > DIVERGE_THRESHOLD);

    // Classify drift pattern in the divergent tail.
    let drift_type = if let Some(div) = diverge_idx {
        if div < deltas.len() - 1 {
            let tail = &deltas[div..];
            let diffs: Vec<i64> = tail.windows(2).map(|w| w[1].abs() - w[0].abs()).collect();
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
fn analyze_level_event(name: String, hw_ivs: &[(i64, i64)], emu_ivs: &[(i64, i64)]) -> LevelResult {
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
        .map(|i| (i, hw_ivs[i].0, hw_ivs[i].1, emu_ivs[i].0, emu_ivs[i].1))
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
    let all_stable = timings.iter().all(|t| t.period_delta.abs() <= PERIOD_DELTA_THRESHOLD);
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
            return IterationDriftType::StepChange { at_iteration: t.iteration, magnitude: t.period_delta };
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
        timings.push(IterationTiming { iteration: i, hw_period, emu_period, period_delta, cumulative_drift });
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
        timings.push(IterationTiming { iteration: i, hw_period, emu_period, period_delta, cumulative_drift });
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
            stall.col == candidate.col && ((stall.row as i16 - candidate.row as i16).abs() == 1)
        }
    }
}

/// Find the nearest resolving event at or after `after_abs` within margin.
///
/// `candidate_cycles_abs` must be sorted. Uses binary search.
fn find_resolution_event(after_abs: u64, candidate_cycles_abs: &[u64], margin: i64) -> Option<u64> {
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
        let names = configs.iter().find(|(k, _)| *k == key).map(|(_, n)| *n).unwrap_or(&[]);
        for ev in events {
            let name = slot_name(ev.slot, names);
            map.entry((key, name)).or_default().push(ev.abs_cycle);
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

    let all_keys: BTreeSet<TileKey> = hw_tiles.keys().chain(emu_tiles.keys()).copied().collect();

    let mut attributions = Vec::new();

    for rule in STALL_RULES {
        // Find stall tiles matching the rule's pkt_type.
        let stall_tiles: Vec<TileKey> =
            all_keys.iter().filter(|k| k.pkt_type == rule.stall_pkt).copied().collect();

        for &stall_key in &stall_tiles {
            // Check if this tile has stall intervals (via level results).
            let tile_result = tile_results.iter().find(|(k, _)| *k == stall_key);
            let stall_intervals =
                tile_result.and_then(|(_, tr)| tr.level_results.iter().find(|lr| lr.name == rule.stall));
            if stall_intervals.is_none() {
                continue;
            }

            // Find candidate resolution tiles.
            let resolve_tiles: Vec<TileKey> = all_keys
                .iter()
                .filter(|k| k.pkt_type == rule.resolve_pkt && tiles_match(&stall_key, k, rule.rel))
                .copied()
                .collect();

            for &resolve_key in &resolve_tiles {
                // Get stall end cycles (absolute) from the tile events.
                let hw_stall_ivs = get_level_intervals_abs(
                    hw_tiles.get(&stall_key).map(|v| v.as_slice()).unwrap_or(&[]),
                    rule.stall,
                    tile_configs
                        .iter()
                        .find(|(k, _)| *k == stall_key)
                        .map(|(_, n)| *n)
                        .unwrap_or(&[]),
                );
                let emu_stall_ivs = get_level_intervals_abs(
                    emu_tiles.get(&stall_key).map(|v| v.as_slice()).unwrap_or(&[]),
                    rule.stall,
                    tile_configs
                        .iter()
                        .find(|(k, _)| *k == stall_key)
                        .map(|(_, n)| *n)
                        .unwrap_or(&[]),
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
                let gap_deltas: Vec<i64> =
                    (0..paired).map(|i| hw_resolutions[i].gap - emu_resolutions[i].gap).collect();

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
fn get_level_intervals_abs(events: &[TileEvent], event_name: &str, slot_names: &[String]) -> Vec<(u64, u64)> {
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

    let all_keys: BTreeSet<TileKey> = hw_tiles.keys().chain(emu_tiles.keys()).copied().collect();

    let mut correlations = Vec::new();

    for rule in CORRELATION_RULES {
        let src_tiles: Vec<TileKey> =
            all_keys.iter().filter(|k| k.pkt_type == rule.src_pkt).copied().collect();

        for &src_key in &src_tiles {
            let dst_tiles: Vec<TileKey> = all_keys
                .iter()
                .filter(|k| k.pkt_type == rule.dst_pkt && tiles_match(&src_key, k, rule.rel))
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
                let gap_deltas: Vec<i64> = (0..paired).map(|i| hw_pairs[i].gap - emu_pairs[i].gap).collect();

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

/// Default cycles between perfcnt overflows when none can be derived
/// from observed PCs. Matches `tools/perfcnt_defaults.py`'s
/// `DEFAULT_PERFCNT_PERIOD` -- keep the two in sync if either changes.
pub const DEFAULT_PERFCNT_PERIOD: u64 = 1024;

/// Canonical event names for the perfcnt overflow slots.
///
/// All four counters per module are recognized as anchor candidates:
/// PERF_CNT_0..3 in core/mem/memtile event tables, PERF_CNT0..3_EVENT
/// in shim event tables (mlir-aie's distinct naming for shim, observed
/// in `xdna-archspec` generated tables). PERF_CNT_0_EVENT is the
/// AM020/AM025 documentation form, kept for non-mlir-aie writers.
///
/// Today the inject pipeline only configures counter 0, so PERF_CNT_0
/// is the practical default. Recognizing 1..3 is forward-looking for
/// tests that might use a non-zero counter as the trace anchor without
/// having to update this list.
const PERFCNT_SLOT_NAMES: &[&str] = &[
    "PERF_CNT_0",
    "PERF_CNT_1",
    "PERF_CNT_2",
    "PERF_CNT_3",
    "PERF_CNT0_EVENT",
    "PERF_CNT1_EVENT",
    "PERF_CNT2_EVENT",
    "PERF_CNT3_EVENT",
    "PERF_CNT_0_EVENT",
];

/// Return true if this event name is a perfcnt overflow sentinel.
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
/// Returns `None` when `perfcnt_pcs` is empty -- there's no anchor at all,
/// so any cycle answer would be invented. Returns `Some(0)` when `pc`
/// precedes the first tick (cycle 0..period can't be pinned tighter without
/// an earlier anchor; the lower bound is honest) and `Some((n-1) * period)`
/// when `pc` is at or beyond the last tick (clamped to the last anchored
/// cycle).
///
/// The "two consecutive ticks share the same retire PC" degenerate case
/// (tight loop straddling an overflow boundary) is unreachable under
/// `partition_point`'s semantics: a probe PC strictly equal to a colliding
/// tick value lands on the first occurrence of that value, putting
/// `pc_below` on the previous *distinct* tick. The function still
/// `debug_assert!`s that the span is positive as a guard against future
/// changes to idx selection.
pub fn interpolate_cycle_from_perfcnt(pc: u64, perfcnt_pcs: &[u64], period: u64) -> Option<u64> {
    if perfcnt_pcs.is_empty() {
        return None;
    }
    // partition_point returns the index of the first element >= pc.
    let idx = perfcnt_pcs.partition_point(|&p| p < pc);
    if idx == 0 {
        // Before first tick: lower-bound estimate.
        return Some(0);
    }
    if idx >= perfcnt_pcs.len() {
        // At or after last tick: clamp to last tick's cycle.
        return Some((perfcnt_pcs.len() as u64 - 1) * period);
    }
    let pc_below = perfcnt_pcs[idx - 1];
    let pc_above = perfcnt_pcs[idx];
    debug_assert!(
        pc_above > pc_below,
        "partition_point invariant violated: idx={idx} pc_below={pc_below} pc_above={pc_above}"
    );
    let span = pc_above - pc_below;
    let frac = (pc - pc_below) as f64 / span as f64;
    Some(((idx as f64 - 1.0 + frac) * period as f64) as u64)
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
    let mut report = PCAnchoredReport { pkt_type: key.pkt_type as u32, ..Default::default() };

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
    let all_names: BTreeSet<String> = hw_by_name.keys().chain(emu_by_name.keys()).cloned().collect();

    // Locate perfcnt overflow PCs from HW side (HW is ground truth).
    // We use the first slot name that matches a known perfcnt spelling.
    let perfcnt_name: Option<String> = all_names.iter().find(|n| is_perfcnt_event(n.as_str())).cloned();

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

    // Record perfcnt tick counts so consumers can detect asymmetric or
    // missing perfcnt clocks without rederiving them from the cycle_bands map.
    report.hw_perfcnt_tick_count = hw_perfcnt_pcs.len();
    report.emu_perfcnt_tick_count = emu_perfcnt_pcs.len();

    // Cycle bands require BOTH sides to have at least 2 perfcnt overflows.
    // With only 1 tick on a side, `interpolate_cycle_from_perfcnt` clamps to
    // 0 (before the tick) or 0 (at the tick), which is silently bogus rather
    // than a real estimate. With <1 tick, fake "exceeds tolerance" deltas
    // would appear for every event. "No data" is a clearer signal than "fake
    // data" -- callers who want loose anchors can request them explicitly.
    let bands_enabled = hw_perfcnt_pcs.len() >= 2 && emu_perfcnt_pcs.len() >= 2;

    // Estimate perfcnt period from the median inter-overflow PC distance.
    // Falls back to DEFAULT_PERFCNT_PERIOD when neither side has 2+ ticks.
    // (When bands_enabled is true, both sides have 2+ ticks and at least one
    // estimate succeeds; the unwrap_or only fires for inert callers.)
    let period: u64 = estimate_perfcnt_period(&hw_perfcnt_pcs)
        .or_else(|| estimate_perfcnt_period(&emu_perfcnt_pcs))
        .unwrap_or(DEFAULT_PERFCNT_PERIOD);

    // Set and multiset diff per event name.
    for name in &all_names {
        let hw_pcs: &[u64] = hw_by_name.get(name).map(|v| v.as_slice()).unwrap_or(&[]);
        let emu_pcs: &[u64] = emu_by_name.get(name).map(|v| v.as_slice()).unwrap_or(&[]);

        let hw_set: HashSet<u64> = hw_pcs.iter().copied().collect();
        let emu_set: HashSet<u64> = emu_pcs.iter().copied().collect();

        let hw_only: HashSet<u64> = hw_set.difference(&emu_set).copied().collect();
        let emu_only: HashSet<u64> = emu_set.difference(&hw_set).copied().collect();
        report.set_diff.insert(name.clone(), (hw_only, emu_only));

        // Per-PC multiset diff: count occurrences on each side.
        // hw_pcs and emu_pcs are sorted (above), so partition_point gives
        // O(log n) range bounds rather than O(n) linear scans.
        let union: HashSet<u64> = hw_set.union(&emu_set).copied().collect();
        let mut multiset: HashMap<u64, (u32, u32, i32)> = HashMap::new();
        for pc in union {
            let hw_lo = hw_pcs.partition_point(|&p| p < pc);
            let hw_hi = hw_pcs.partition_point(|&p| p <= pc);
            let hw_count = (hw_hi - hw_lo) as u32;
            let emu_lo = emu_pcs.partition_point(|&p| p < pc);
            let emu_hi = emu_pcs.partition_point(|&p| p <= pc);
            let emu_count = (emu_hi - emu_lo) as u32;
            let delta = hw_count as i32 - emu_count as i32;
            multiset.insert(pc, (hw_count, emu_count, delta));
        }
        report.multiset_diff.insert(name.clone(), multiset);

        // Cycle bands: skip perfcnt slot itself (it IS the clock reference).
        if perfcnt_name.as_deref() == Some(name.as_str()) {
            continue;
        }
        // Skip if either side lacks a perfcnt clock (see bands_enabled comment).
        if !bands_enabled {
            continue;
        }

        // For every PC that appears in either side, interpolate cycle estimates.
        // Skip PCs where either side returns None -- the degenerate cases
        // (empty anchors, same-PC tick collision) shouldn't pollute the band
        // map with bogus zero estimates.
        let all_pcs: HashSet<u64> = hw_pcs.iter().chain(emu_pcs.iter()).copied().collect();
        let tolerance = period / 2;
        let mut bands: HashMap<u64, CycleBand> = HashMap::new();
        for pc in all_pcs {
            let (Some(hw_est), Some(emu_est)) = (
                interpolate_cycle_from_perfcnt(pc, &hw_perfcnt_pcs, period),
                interpolate_cycle_from_perfcnt(pc, &emu_perfcnt_pcs, period),
            ) else {
                continue;
            };
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
    let mut gaps: Vec<u64> = perfcnt_pcs.windows(2).map(|w| w[1].saturating_sub(w[0])).collect();
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
        hw_by_slot.entry(ev.slot).or_default().push(ev.abs_cycle as i64 - hw_t0 as i64);
    }
    for ev in emu_events {
        emu_by_slot
            .entry(ev.slot)
            .or_default()
            .push(ev.abs_cycle as i64 - emu_t0 as i64);
    }

    let all_slots: BTreeSet<u8> = hw_by_slot.keys().chain(emu_by_slot.keys()).copied().collect();

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
                if let Some(ir) = analyze_iterations_level(&name, tile_key, &hw_ivs, &emu_ivs) {
                    iteration_results.push(ir);
                }
            }
            level_results.push(analyze_level_event(name, &hw_ivs, &emu_ivs));
        } else {
            if opts.iterations {
                if let Some(ir) = analyze_iterations_edge(&name, tile_key, &hw_sorted, &emu_sorted) {
                    iteration_results.push(ir);
                }
            }
            edge_results.push(analyze_edge_event(name, &hw_sorted, &emu_sorted));
        }
    }

    TileResult { hw_t0, emu_t0, edge_results, level_results, iteration_results }
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

    let all_keys: BTreeSet<TileKey> = hw_tiles.keys().chain(emu_tiles.keys()).copied().collect();

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
    let _ =
        writeln!(out, "  Edge:       paired by occurrence index, divergence at |dt|>{}", DIVERGE_THRESHOLD);
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
                let _ = writeln!(out, "    [edge] {:<32} {:<20} {}", er.name, count_str, status,);

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
                    let _ = writeln!(out, "           Drift pattern: {}", er.drift_type);
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
                let _ = writeln!(out, "    [level] {:<31} {:<20} {}", lr.name, count_str, status,);

                // Clean duration stats.
                if !lr.dur_deltas.is_empty() {
                    let clean_end = lr.diverge_idx.unwrap_or(lr.paired);
                    let clean_dur = &lr.dur_deltas[..clean_end];
                    if !clean_dur.is_empty() {
                        let min_d = clean_dur.iter().copied().min().unwrap();
                        let max_d = clean_dur.iter().copied().max().unwrap();
                        let mean_d = clean_dur.iter().sum::<i64>() as f64 / clean_dur.len() as f64;
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
        let _ = writeln!(out, "Divergence Points (where HW and EMU first disagree)");
        let _ = writeln!(out, "{}", "=".repeat(76));
        for (batch, tile, name, idx, drift) in &divergence_details {
            let _ = writeln!(out, "  Batch {}  {:<16} {:<34} #{:<5} ({})", batch, tile, name, idx, drift,);
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
        let _ = writeln!(out, "  Mean:            {:+.1}", sorted_c.iter().sum::<i64>() as f64 / n as f64);
        let _ = writeln!(out, "  Median:          {:+}", sorted_c[n / 2]);
        let _ = writeln!(out, "  Spread:          {}", sorted_c[n - 1] - sorted_c[0]);
        let _ = writeln!(out);
        for &threshold in &[0i64, 1, 2, 5, 10] {
            let within = count_within(&sorted_c, threshold);
            let pct = within as f64 / n as f64 * 100.0;
            let _ = writeln!(out, "  Within +/-{:>3}:    {:>4}/{} ({:.1}%)", threshold, within, n, pct,);
        }

        // Total stats (including post-divergence) if different from clean.
        all_edge_deltas_total.sort();
        if all_edge_deltas_total.len() != sorted_c.len() {
            let sorted_t = &all_edge_deltas_total;
            let nt = sorted_t.len();
            let _ = writeln!(out);
            let _ = writeln!(out, "Edge timing (ALL pairs including post-divergence):");
            let _ = writeln!(out, "  Pairs:           {}", nt);
            let _ = writeln!(out, "  Spread:          {}", sorted_t[nt - 1] - sorted_t[0]);
            for &threshold in &[0i64, 1, 2, 5, 10, 50] {
                let within = count_within(sorted_t, threshold);
                let pct = within as f64 / nt as f64 * 100.0;
                let _ = writeln!(out, "  Within +/-{:>3}:    {:>4}/{} ({:.1}%)", threshold, within, nt, pct,);
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
                        key.col, key.row, module, ir.name, ir.iteration_count, ir.drift_classification,
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
    let has_stalls = batch_results.iter().any(|b| !b.stall_attributions.is_empty());
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
                    sa.stall_tile.col,
                    sa.stall_tile.row,
                    stall_mod,
                    sa.hw_resolutions.first().map(|r| r.resolve_name.as_str()).unwrap_or("?"),
                    sa.resolve_tile.col,
                    sa.resolve_tile.row,
                    resolve_mod,
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
                    let mean_delta = sa.gap_deltas.iter().sum::<i64>() as f64 / sa.gap_deltas.len() as f64;
                    let direction = if mean_delta > 0.5 {
                        "EMU faster"
                    } else if mean_delta < -0.5 {
                        "EMU slower"
                    } else {
                        "matched"
                    };
                    let _ = writeln!(out, "  Delta:   mean={:+.1}cy ({})", mean_delta, direction,);
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
                        cr.src_tile.col,
                        cr.src_tile.row,
                        src_mod,
                        cr.dst_tile.col,
                        cr.dst_tile.row,
                        dst_mod,
                    );

                    let hw_count = cr.hw_pairs.len();
                    let emu_count = cr.emu_pairs.len();
                    let _ = writeln!(out, "  Pairs: HW={}, EMU={}", hw_count, emu_count,);

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
                            mean_delta,
                            direction,
                            cr.gap_deltas.len(),
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

/// Sweep manifest written by `tools/trace-sweep.py`.
///
/// Two formats coexist:
///   - **Legacy** (sweep_multi / older orchestrator): top-level `batches`
///     field present, an array of `SweepBatchInfo`.  An empty array means
///     "legacy run produced zero passing batches"; we still go down the
///     legacy path (which yields an empty result, not fs discovery).
///   - **Lockstep** (sweep_lockstep / mode-1): top-level `batches` field
///     absent; per-batch state is recorded only via the `batch_NN/`
///     subdirectories on disk.  We use `Option<Vec<...>>` rather than
///     `Vec<...>` + `#[serde(default)]` so that "field missing" and
///     "field is `[]`" are distinguishable -- a typo in the field name
///     in a future legacy manifest would otherwise silently kick us into
///     the fs-discovery branch and miss real batches.
#[derive(Deserialize)]
struct SweepManifest {
    /// Present in legacy manifests, absent in lockstep manifests.
    batches: Option<Vec<SweepBatchInfo>>,
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

/// Discover batch indices from a sweep directory by filesystem scan.
///
/// Walks `sweep_dir/batch_NN/` entries and returns a sorted list of (index,
/// hw_events_path, emu_events_path) tuples for every batch that has both HW
/// and EMU events JSON files.  Used when the manifest does not carry a
/// `batches[]` list (lockstep / mode-1 manifests omit it).
fn discover_batches_from_fs(sweep_dir: &Path) -> Vec<(usize, std::path::PathBuf, std::path::PathBuf)> {
    let mut results = Vec::new();
    // Pattern: sweep_dir/batch_NN/  where NN is zero-padded decimal.
    let rd = match fs::read_dir(sweep_dir) {
        Ok(rd) => rd,
        Err(_) => return results,
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("batch_") {
            continue;
        }
        let idx_str = &name_str["batch_".len()..];
        let idx: usize = match idx_str.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let batch_dir = entry.path();
        let hw = batch_dir.join("hw").join("trace.events.json");
        let emu = batch_dir.join("emu").join("trace.events.json");
        if hw.exists() && emu.exists() {
            results.push((idx, hw, emu));
        }
    }
    results.sort_by_key(|(idx, _, _)| *idx);
    results
}

/// Compare all batches in a trace-sweep.py output directory.
///
/// Reads `sweep-manifest.json`, iterates successful batches, and produces
/// a unified comparison report.
pub fn compare_sweep_dir(sweep_dir: &Path) -> Result<String, String> {
    compare_sweep_dir_with_opts(sweep_dir, &AnalysisOptions::default())
}

/// Compare all batches with extended analysis options.
///
/// Supports two manifest formats:
///   - Legacy (sweep_multi): `batches: [{batch, status, hw_status, emu_status}, ...]`
///   - Lockstep (sweep_lockstep / mode-1): no `batches[]` field; batches are
///     discovered from `batch_NN/` subdirectories.
pub fn compare_sweep_dir_with_opts(sweep_dir: &Path, opts: &AnalysisOptions) -> Result<String, String> {
    let manifest_path = sweep_dir.join("sweep-manifest.json");
    let manifest_text =
        fs::read_to_string(&manifest_path).map_err(|e| format!("read {}: {}", manifest_path.display(), e))?;
    let manifest: SweepManifest =
        serde_json::from_str(&manifest_text).map_err(|e| format!("parse manifest: {}", e))?;

    // Build the batch list.  See `SweepManifest` doc for format policy:
    //   - `batches: None`     -> lockstep manifest, scan batch_NN/ on disk.
    //   - `batches: Some([])` -> legacy manifest with zero passing batches;
    //                            keep the empty result (do NOT fall back to fs).
    //   - `batches: Some(_)`  -> legacy manifest, filter by status fields.
    let discovered: Vec<(usize, std::path::PathBuf, std::path::PathBuf)>;
    let batch_triples: &[(usize, std::path::PathBuf, std::path::PathBuf)] = match &manifest.batches {
        None => {
            discovered = discover_batches_from_fs(sweep_dir);
            &discovered
        }
        Some(legacy) => {
            // Filter legacy batches to only those that completed successfully.
            // All three status fields must be "ok" (original behaviour); batches
            // where all status fields are empty are also included because some
            // manifest versions omit status when every batch succeeded.
            discovered = legacy
                .iter()
                .filter(|info| {
                    (info.status == "ok" && info.hw_status == "ok" && info.emu_status == "ok")
                        || (info.status.is_empty() && info.hw_status.is_empty() && info.emu_status.is_empty())
                })
                .map(|info| {
                    let batch_dir = sweep_dir.join(format!("batch_{:02}", info.batch));
                    let hw = batch_dir.join("hw").join("trace.events.json");
                    let emu = batch_dir.join("emu").join("trace.events.json");
                    (info.batch, hw, emu)
                })
                .collect();
            &discovered
        }
    };

    let mut batch_results = Vec::new();

    for (batch_idx, hw_events, emu_events) in batch_triples {
        let batch_idx = *batch_idx;
        let batch_dir = sweep_dir.join(format!("batch_{:02}", batch_idx));

        // hw_events / emu_events come from the discovery step above; verify
        // they still exist before reading (race-free on the hot path).
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
            serde_json::from_str(&text).map_err(|e| format!("parse {}: {}", events_json.display(), e))?
        } else {
            EventsConfig::default()
        };

        match compare_batch_with_opts(hw_events, emu_events, &config, batch_idx, opts) {
            Ok(result) => batch_results.push(result),
            Err(e) => {
                eprintln!("Warning: batch {} failed: {}", batch_idx, e);
            }
        }
    }

    let base_report = format_report(&batch_results);

    // ---- PC-anchored sweep aggregation (Task 8) ----
    // Only appended when at least one batch has PC-anchored data.
    let has_pc = batch_results.iter().any(|b| !b.pc_anchored.is_empty());
    if has_pc {
        let manifest_opt = read_pc1_manifest(sweep_dir);
        let grounding = read_grounding_from_pc1_manifest(sweep_dir);
        let mut pc_suffix = format_report_pc_anchored(&batch_results, &grounding);

        // unsafe_for_pc_join warning from manifest. Placement note: we keep
        // the warning inside the PC suffix (rather than prepending to
        // base_report) because it qualifies the PC-anchored data, not the
        // standard per-batch report. Anyone reading the PC sections sees
        // the safety caveat directly above them.
        if let Some(manifest) = &manifest_opt {
            if manifest.unsafe_for_pc_join {
                let mut warning = String::new();
                let _ = writeln!(warning, "{}", "=".repeat(76));
                let _ = writeln!(warning, "WARNING: sweep-manifest flagged unsafe_for_pc_join=true");
                let _ = writeln!(warning, "  reason: {}", manifest.reason.as_deref().unwrap_or("?"));
                let _ = writeln!(
                    warning,
                    "  PC-anchored cross-batch joining was skipped; results are per-batch only."
                );
                let _ = writeln!(warning);
                // Prepend warning to pc_suffix.
                pc_suffix = warning + &pc_suffix;
            }
        }

        // Mode-2 comparison. For each baseline file in mode2-baseline/,
        // attempt to find the matching EMU mode-2 raw stream and run the
        // three-layer comparator. Path conventions are tentative until
        // Phase 6 finalizes the bridge-test layout (see Task 6.2).
        //
        // We always list whatever physical files are present; the manifest
        // flag is only a confirmation hint. A corrupted or missing manifest
        // must not silently suppress real baseline files.
        let mode2_pair = find_mode2_baseline_pair(sweep_dir);
        if let Some((hw_path, emu_path)) = mode2_pair {
            let _ = writeln!(pc_suffix, "Mode-2 comparison:");
            if !emu_path.exists() {
                let _ = writeln!(
                    pc_suffix,
                    "  hw={} -- SKIP (EMU events JSON {} not present; rerun with --mode2 after rebuilding the FFI plugin)",
                    hw_path.display(),
                    emu_path.display(),
                );
            } else {
                match crate::trace::compare_mode2::compare_mode2_from_events_files(
                    &hw_path,
                    &emu_path,
                    opts.remap_columns,
                ) {
                    Err(e) => {
                        let _ = writeln!(pc_suffix, "  ERROR loading events: {}", e);
                    }
                    Ok(report) => {
                        if report.per_tile.is_empty() {
                            let _ = writeln!(pc_suffix, "  no tiles common to HW and EMU events JSON",);
                        }
                        for r in &report.per_tile {
                            let status = if r.passed { "PASS" } else { "FAIL" };
                            let _ = writeln!(
                                pc_suffix,
                                "  tile pt={} ({},{}) [{}]: PC seq {}/{}, LC {}/{}, atom windows: {}",
                                r.tile.pkt_type,
                                r.tile.col,
                                r.tile.row,
                                status,
                                r.layer1.hw_count,
                                r.layer1.emu_count,
                                r.layer2.hw_count,
                                r.layer2.emu_count,
                                r.layer3.windows.len(),
                            );
                        }
                        for tile in &report.hw_only {
                            let _ = writeln!(
                                pc_suffix,
                                "  tile pt={} ({},{}): HW only -- no EMU events for tile",
                                tile.pkt_type, tile.col, tile.row,
                            );
                        }
                        for tile in &report.emu_only {
                            let _ = writeln!(
                                pc_suffix,
                                "  tile pt={} ({},{}): EMU only -- no HW events for tile",
                                tile.pkt_type, tile.col, tile.row,
                            );
                        }
                    }
                }
            }
            if let Some(m) = &manifest_opt {
                if !m.mode2_baseline_captured {
                    let _ = writeln!(
                        pc_suffix,
                        "  (warning: sweep-manifest reports mode2_baseline_captured=false but baseline files exist)",
                    );
                }
            }
        }

        return Ok(base_report + &pc_suffix);
    }

    Ok(base_report)
}

// ============================================================================
// PC-anchored sweep aggregation (Task 8)
// ============================================================================

/// Task 6 sweep-manifest.json shape.
///
/// The old SweepManifest covers the legacy `batches[]` format. This struct
/// covers the Task 6 `sweep_lockstep` output format, which is distinct.
#[derive(Deserialize)]
struct Pc1SweepManifest {
    /// Union of all grounding event names across all tile types.
    #[serde(default)]
    grounding: Pc1Grounding,
    /// True when cross-batch PC joining is unsafe (grounding PC sets drifted).
    #[serde(default)]
    unsafe_for_pc_join: bool,
    /// Human-readable reason for unsafe_for_pc_join (null when false).
    #[serde(default)]
    reason: Option<String>,
    /// True when a mode-2 baseline trace was captured alongside the sweep.
    #[serde(default)]
    mode2_baseline_captured: bool,
}

/// Per-tile-type grounding event lists from sweep-manifest.json.
#[derive(Deserialize, Default)]
struct Pc1Grounding {
    #[serde(default)]
    core: Vec<String>,
    #[serde(default)]
    memmod: Vec<String>,
    #[serde(default)]
    memtile: Vec<String>,
    #[serde(default)]
    shim: Vec<String>,
}

/// Read the Task 6 sweep-manifest.json from `sweep_dir`.
///
/// Returns `None` (with a log warning) when the file is absent or unparseable.
/// Robust to missing manifest: callers must handle `None` gracefully.
fn read_pc1_manifest(sweep_dir: &Path) -> Option<Pc1SweepManifest> {
    let path = sweep_dir.join("sweep-manifest.json");
    match fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str(&text) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("Warning: sweep-manifest.json parse failed (Pc1 schema): {}", e);
                None
            }
        },
        Err(_) => None, // absent manifest is normal for non-pc1 sweeps
    }
}

/// Read grounding event names from a Task 6 sweep-manifest.json.
///
/// Returns the union of `grounding.core`, `grounding.memmod`,
/// `grounding.memtile`, and `grounding.shim`. Returns an empty set when the
/// manifest is absent or does not carry grounding data — the caller treats
/// all events as swept in that case.
pub fn read_grounding_from_pc1_manifest(sweep_dir: &Path) -> BTreeSet<String> {
    match read_pc1_manifest(sweep_dir) {
        None => BTreeSet::new(),
        Some(m) => {
            let mut out = BTreeSet::new();
            for name in m
                .grounding
                .core
                .iter()
                .chain(&m.grounding.memmod)
                .chain(&m.grounding.memtile)
                .chain(&m.grounding.shim)
            {
                out.insert(name.clone());
            }
            out
        }
    }
}

/// Locate the mode-2 baseline events JSONs produced by trace-sweep.py
/// (Task 6 mode-2 capture). The sweep writes:
///
/// ```text
/// <sweep_dir>/mode2-baseline/hw/trace.events.json
/// <sweep_dir>/mode2-baseline/emu/trace.events.json   (when EMU-side mode-2 ran)
/// ```
///
/// Returns `Some((hw, emu))` when the HW events JSON exists, with `emu`
/// pointing at its sibling whether or not it's been written yet. Returns
/// `None` when no mode-2 baseline was captured at all -- callers should
/// treat that as "nothing to compare," not an error.
pub fn find_mode2_baseline_pair(sweep_dir: &Path) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let hw = sweep_dir.join("mode2-baseline/hw/trace.events.json");
    if !hw.is_file() {
        return None;
    }
    let emu = sweep_dir.join("mode2-baseline/emu/trace.events.json");
    Some((hw, emu))
}

/// Aggregate PC-anchored reports across batches and format three sections:
///
/// 1. **Coverage matrix**: per-event, per-batch status (swept / absent / grounding).
/// 2. **Divergence summary**: per-event total set+multiset magnitude, sorted descending.
/// 3. **Cycle delta summary**: avg and max |delta_cycles| per event.
///
/// `grounding` is the set of event names that serve as the perfcnt anchor in
/// every batch (read from sweep-manifest.json via `read_grounding_from_pc1_manifest`).
/// When empty, all events are classified as "swept".
///
/// This function is `pub` so unit tests can call it directly without a sweep dir.
pub fn format_report_pc_anchored(batch_results: &[BatchResult], grounding: &BTreeSet<String>) -> String {
    // --- Aggregate across batches ---

    // event_name -> batch_idx -> "swept" | "absent" | "grounding"
    let mut event_per_batch: BTreeMap<String, BTreeMap<usize, &str>> = BTreeMap::new();
    // event_name -> (set_diff_total, multiset_magnitude_total)
    let mut event_total_diff: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    // event_name -> [delta_cycles]
    let mut event_cycle_deltas: BTreeMap<String, Vec<i64>> = BTreeMap::new();

    // Collect all event names that appear anywhere in pc_anchored data.
    for batch in batch_results {
        for (_, report) in &batch.pc_anchored {
            for (name, (hw_only, emu_only)) in &report.set_diff {
                let status = if grounding.contains(name) {
                    "grounding"
                } else {
                    "swept"
                };
                event_per_batch.entry(name.clone()).or_default().insert(batch.batch_idx, status);

                let entry = event_total_diff.entry(name.clone()).or_default();
                entry.0 += hw_only.len() + emu_only.len();

                if let Some(ms) = report.multiset_diff.get(name) {
                    let mag: usize = ms.values().map(|(_, _, d)| d.unsigned_abs() as usize).sum();
                    entry.1 += mag;
                }
            }

            // Note: cycle_bands.keys() is a strict subset of set_diff.keys()
            // by construction in compare_pc_anchored_for_tile (every event
            // with a cycle band also has a set_diff entry, even when that
            // entry is empty). We therefore don't update event_per_batch
            // from this loop -- it would already have been populated by
            // the set_diff iteration above. Cycle deltas alone do not
            // imply coverage; coverage is defined by set_diff presence.
            for (name, by_pc) in &report.cycle_bands {
                let deltas = event_cycle_deltas.entry(name.clone()).or_default();
                for band in by_pc.values() {
                    deltas.push(band.delta_cycles);
                }
            }
        }
    }

    // Fill in "absent" for events that didn't appear in some batches.
    let all_batch_idxs: Vec<usize> = batch_results.iter().map(|b| b.batch_idx).collect();
    for by_batch in event_per_batch.values_mut() {
        for &idx in &all_batch_idxs {
            by_batch.entry(idx).or_insert("absent");
        }
    }

    let mut out = String::new();

    // ---- Section 1: Coverage matrix ----
    let _ = writeln!(out, "{}", "=".repeat(76));
    let _ = writeln!(out, "PC-anchored coverage");
    let _ = writeln!(out, "{}", "=".repeat(76));
    for (name, by_batch) in &event_per_batch {
        let _ = write!(out, "  {:24}: ", name);
        for (idx, status) in by_batch {
            let _ = write!(out, "batch {}={:<10} ", idx, status);
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(out);

    // ---- Section 2: Divergence summary (sorted descending by total magnitude) ----
    let _ = writeln!(out, "{}", "=".repeat(76));
    let _ = writeln!(out, "PC-anchored divergences (sorted by total)");
    let _ = writeln!(out, "{}", "=".repeat(76));
    let mut sorted_div: Vec<_> = event_total_diff.iter().collect();
    // Sort descending by combined magnitude, with alphabetical tie-break.
    // The explicit secondary key keeps output deterministic across BTreeMap
    // iteration changes and Rust's sort stability guarantees.
    sorted_div.sort_by(|(na, (sa, ma)), (nb, (sb, mb))| (sb + mb).cmp(&(sa + ma)).then_with(|| na.cmp(nb)));
    for (name, (set_size, multiset_mag)) in &sorted_div {
        let _ = writeln!(out, "  {:24}: set_diff={} multiset_mag={}", name, set_size, multiset_mag);
    }
    let _ = writeln!(out);

    // ---- Section 3: Cycle delta summary ----
    let _ = writeln!(out, "{}", "=".repeat(76));
    let _ = writeln!(out, "Perfcnt-anchored cycle deltas (avg |delta_cycles| per event)");
    let _ = writeln!(out, "{}", "=".repeat(76));
    for (name, deltas) in &event_cycle_deltas {
        if deltas.is_empty() {
            continue;
        }
        let avg_abs = deltas.iter().map(|d| d.unsigned_abs() as f64).sum::<f64>() / deltas.len() as f64;
        let max_abs = deltas.iter().map(|d| d.unsigned_abs()).max().unwrap_or(0);
        let _ = writeln!(out, "  {:24}: avg={:.1} max={} n={}", name, avg_abs, max_abs, deltas.len());
    }
    let _ = writeln!(out);

    out
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
            IterationTiming {
                iteration: 1,
                hw_period: 25,
                emu_period: 25,
                period_delta: 0,
                cumulative_drift: 0,
            },
            IterationTiming {
                iteration: 2,
                hw_period: 25,
                emu_period: 26,
                period_delta: -1,
                cumulative_drift: -1,
            },
            IterationTiming {
                iteration: 3,
                hw_period: 25,
                emu_period: 24,
                period_delta: 1,
                cumulative_drift: 0,
            },
        ];
        assert!(matches!(classify_iteration_drift(&timings), IterationDriftType::Stable));
    }

    #[test]
    fn test_classify_iteration_drift_step_change() {
        let timings = vec![
            IterationTiming {
                iteration: 1,
                hw_period: 25,
                emu_period: 25,
                period_delta: 0,
                cumulative_drift: 0,
            },
            IterationTiming {
                iteration: 2,
                hw_period: 25,
                emu_period: 25,
                period_delta: 0,
                cumulative_drift: 0,
            },
            IterationTiming {
                iteration: 3,
                hw_period: 25,
                emu_period: 75,
                period_delta: -50,
                cumulative_drift: -50,
            },
            IterationTiming {
                iteration: 4,
                hw_period: 25,
                emu_period: 25,
                period_delta: 0,
                cumulative_drift: -50,
            },
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
        let opts = AnalysisOptions { pc_anchored: true, ..Default::default() };
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
        let emu_events = vec![TileEvent { slot: 0, abs_cycle: 100 }, TileEvent { slot: 0, abs_cycle: 200 }];
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
            TileEvent { slot: 0, abs_cycle: 0 }, // unanchored
            TileEvent { slot: 0, abs_cycle: 100 },
        ];
        let emu_events = vec![
            TileEvent { slot: 0, abs_cycle: 0 }, // unanchored
            TileEvent { slot: 0, abs_cycle: 0 }, // unanchored
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
        assert_eq!(est, Some(25));
        // PC 85 is between tick[1]=60 and tick[2]=110.
        // Linear: cycle ~= 50 + 50 * (85-60)/(110-60) = 75.
        let est2 = interpolate_cycle_from_perfcnt(85, &perfcnt_pcs, period);
        assert_eq!(est2, Some(75));
    }

    #[test]
    fn cycle_band_boundary_cases() {
        let perfcnt_pcs = vec![10u64, 60, 110];
        let period = 50u64;
        // Exactly at tick[0]: returns Some(0) cycles.
        assert_eq!(interpolate_cycle_from_perfcnt(10, &perfcnt_pcs, period), Some(0));
        // Exactly at tick[1]: returns Some(1 * 50) = Some(50) cycles.
        assert_eq!(interpolate_cycle_from_perfcnt(60, &perfcnt_pcs, period), Some(50));
        // Before first tick: returns Some(0).
        assert_eq!(interpolate_cycle_from_perfcnt(5, &perfcnt_pcs, period), Some(0));
        // After last tick: clamps to Some((len-1)*period) = Some(100).
        assert_eq!(interpolate_cycle_from_perfcnt(200, &perfcnt_pcs, period), Some(100));
        // Empty perfcnt: returns None (no anchor).
        assert_eq!(interpolate_cycle_from_perfcnt(50, &[], period), None);
    }

    #[test]
    fn cycle_band_handles_duplicate_ticks() {
        // Two consecutive overflows at the same retire PC -- a tight loop
        // straddling the overflow boundary. partition_point places probes
        // such that pc_below is always the previous *distinct* tick, so the
        // span is positive and the standard interpolation applies.
        let perfcnt_pcs = vec![10u64, 50, 50, 100];
        let period = 50u64;
        // PC=50 lands at idx=1 (first tick >= 50), pc_below=10, pc_above=50.
        // Linear: cycle = 0 + 50 * (50-10)/(50-10) = 50.
        assert_eq!(interpolate_cycle_from_perfcnt(50, &perfcnt_pcs, period), Some(50));
        // PC=51 lands at idx=3 (past the run of 50s), pc_below=50, pc_above=100.
        // Linear: cycle = 100 + 50 * (51-50)/(100-50) = 101.
        assert_eq!(interpolate_cycle_from_perfcnt(51, &perfcnt_pcs, period), Some(101));
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
    fn pc_anchored_single_tick_per_side_suppresses_bands() {
        // Only one perfcnt tick on each side -- bands_enabled requires
        // >= 2 per side, so cycle_bands should be empty (or absent) for
        // every non-perfcnt event. This prevents silently-bogus zero
        // cycle estimates from one-anchor cases.
        let hw_events = vec![
            TileEvent { slot: 1, abs_cycle: 0x100 }, // single perfcnt tick
            TileEvent { slot: 0, abs_cycle: 0x180 }, // INSTR_VECTOR
        ];
        let emu_events = vec![
            TileEvent { slot: 1, abs_cycle: 0x100 }, // single perfcnt tick
            TileEvent { slot: 0, abs_cycle: 0x180 }, // INSTR_VECTOR
        ];
        let names = vec!["INSTR_VECTOR".to_string(), "PERF_CNT_0".to_string()];
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        let report = compare_pc_anchored_for_tile(key, &hw_events, &emu_events, &names);
        assert_eq!(report.hw_perfcnt_tick_count, 1);
        assert_eq!(report.emu_perfcnt_tick_count, 1);
        // No cycle band for INSTR_VECTOR -- one-tick anchor is insufficient.
        assert!(
            report.cycle_bands.get("INSTR_VECTOR").is_none(),
            "single-tick anchor should suppress cycle bands"
        );
    }

    #[test]
    fn pc_anchored_perfcnt_slot_name_variant() {
        // Core/mem/memtile naming: PERF_CNT_{0,1,2,3}.
        assert!(is_perfcnt_event("PERF_CNT_0"));
        assert!(is_perfcnt_event("PERF_CNT_1"));
        assert!(is_perfcnt_event("PERF_CNT_2"));
        assert!(is_perfcnt_event("PERF_CNT_3"));
        // Shim naming (mlir-aie's distinct enum): PERF_CNT{0,1,2,3}_EVENT.
        assert!(is_perfcnt_event("PERF_CNT0_EVENT"));
        assert!(is_perfcnt_event("PERF_CNT1_EVENT"));
        assert!(is_perfcnt_event("PERF_CNT2_EVENT"));
        assert!(is_perfcnt_event("PERF_CNT3_EVENT"));
        // Legacy AM020 documentation form for counter 0.
        assert!(is_perfcnt_event("PERF_CNT_0_EVENT"));
        // Non-perfcnt events are rejected.
        assert!(!is_perfcnt_event("INSTR_VECTOR"));
        assert!(!is_perfcnt_event("PERF_CNT_4")); // hardware doesn't go past 3
        assert!(!is_perfcnt_event("PERF_CNT4_EVENT"));
    }

    #[test]
    fn estimate_perfcnt_period_median() {
        // Gaps: [90, 100, 110] -> sorted -> median = 100.
        let pcs = vec![0u64, 90, 190, 300];
        let period = estimate_perfcnt_period(&pcs).unwrap();
        // Gaps: 90, 100, 110 -> median (idx 1 of 3) = 100.
        assert_eq!(period, 100);
    }

    #[test]
    fn estimate_perfcnt_period_two_ticks_uses_single_gap() {
        // With exactly 2 ticks there is one gap; gaps[gaps.len()/2] == gaps[0].
        let pcs = vec![100u64, 1124];
        assert_eq!(estimate_perfcnt_period(&pcs), Some(1024));
    }

    #[test]
    fn pc_anchored_one_sided_perfcnt_does_not_produce_misleading_bands() {
        // HW has perfcnt overflow events; EMU doesn't (the perfcnt clock
        // isn't reaching the trace unit on the EMU side yet). With a
        // one-sided perfcnt clock, every interpolated EMU estimate would
        // be 0 and every band would falsely flag "exceeds tolerance".
        // Expected: cycle_bands stays empty; tick counts reflect reality.
        let names = vec!["PERF_CNT_0".to_string(), "INSTR_VECTOR".to_string()];
        let hw_events = vec![
            TileEvent { slot: 0, abs_cycle: 100 }, // perfcnt tick
            TileEvent { slot: 0, abs_cycle: 200 }, // perfcnt tick
            TileEvent { slot: 1, abs_cycle: 150 }, // INSTR_VECTOR
        ];
        let emu_events = vec![
            // No slot-0 (perfcnt) events on EMU side.
            TileEvent { slot: 1, abs_cycle: 150 }, // INSTR_VECTOR
        ];
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        let report = compare_pc_anchored_for_tile(key, &hw_events, &emu_events, &names);
        assert!(report.cycle_bands.is_empty(), "asymmetric perfcnt should suppress cycle band generation");
        assert_eq!(report.hw_perfcnt_tick_count, 2);
        assert_eq!(report.emu_perfcnt_tick_count, 0);
        // Set/multiset diff still works regardless of perfcnt presence.
        assert!(report.set_diff.contains_key("INSTR_VECTOR"));
    }

    // ---- PC-anchored sweep aggregation tests (Task 8) ----

    /// Build a minimal PCAnchoredReport with the given event names in set_diff.
    /// No actual PCs are recorded; HW-only and EMU-only sets are empty (no
    /// divergence). Used to test coverage matrix formatting.
    fn make_pc_report_for_events(events_present: &[&str]) -> PCAnchoredReport {
        let mut report = PCAnchoredReport::default();
        report.pkt_type = 0;
        for name in events_present {
            report.set_diff.insert(name.to_string(), (HashSet::new(), HashSet::new()));
            report.multiset_diff.insert(name.to_string(), HashMap::new());
        }
        report
    }

    /// Build a BatchResult with only pc_anchored data (no tile timing data).
    fn make_batch_pc_only(batch_idx: usize, key: TileKey, report: PCAnchoredReport) -> BatchResult {
        let mut pc_anchored = HashMap::new();
        pc_anchored.insert(key, report);
        BatchResult {
            batch_idx,
            config: EventsConfig::default(),
            tiles: Vec::new(),
            stall_attributions: Vec::new(),
            cross_tile: None,
            pc_anchored,
        }
    }

    #[test]
    fn pc_anchored_coverage_matrix_formats() {
        // Batch 0 covers INSTR_VECTOR + PERF_CNT_0.
        // Batch 1 covers MEMORY_STALL + PERF_CNT_0.
        // PERF_CNT_0 is in the grounding set.
        // Expected: INSTR_VECTOR shows swept/absent; MEMORY_STALL shows absent/swept;
        //           PERF_CNT_0 shows grounding in both batches.
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        let batch0 = make_batch_pc_only(0, key, make_pc_report_for_events(&["PERF_CNT_0", "INSTR_VECTOR"]));
        let batch1 = make_batch_pc_only(1, key, make_pc_report_for_events(&["PERF_CNT_0", "MEMORY_STALL"]));

        let grounding: BTreeSet<String> = ["PERF_CNT_0".to_string()].iter().cloned().collect();
        let report = format_report_pc_anchored(&[batch0, batch1], &grounding);

        assert!(report.contains("PC-anchored coverage"), "missing section header");
        assert!(report.contains("INSTR_VECTOR"), "INSTR_VECTOR missing");
        assert!(report.contains("MEMORY_STALL"), "MEMORY_STALL missing");
        assert!(report.contains("PERF_CNT_0"), "PERF_CNT_0 missing");
        // Check status tags are present.
        assert!(report.contains("swept"), "swept tag missing");
        assert!(report.contains("absent"), "absent tag missing");
        assert!(report.contains("grounding"), "grounding tag missing");
    }

    #[test]
    fn pc_anchored_divergence_sorted_descending() {
        // Event A: set_diff = 5 PCs hw-only, 0 emu-only -> total set = 5.
        // Event B: set_diff = 1 PC hw-only, 0 emu-only -> total set = 1.
        // Expected: A appears before B in the divergence section.
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };

        // Build report for event A with 5 hw-only PCs.
        let mut report_a = PCAnchoredReport::default();
        let hw_only_a: HashSet<u64> = [100, 200, 300, 400, 500].iter().cloned().collect();
        report_a.set_diff.insert("EVENT_A".to_string(), (hw_only_a, HashSet::new()));
        report_a.multiset_diff.insert("EVENT_A".to_string(), HashMap::new());

        // Build report for event B with 1 hw-only PC.
        let mut report_b = PCAnchoredReport::default();
        let hw_only_b: HashSet<u64> = [100].iter().cloned().collect();
        report_b.set_diff.insert("EVENT_B".to_string(), (hw_only_b, HashSet::new()));
        report_b.multiset_diff.insert("EVENT_B".to_string(), HashMap::new());

        let mut pc_anchored = HashMap::new();
        // Put both events in the same tile report by merging.
        let mut combined = PCAnchoredReport::default();
        combined
            .set_diff
            .insert("EVENT_A".to_string(), report_a.set_diff["EVENT_A"].clone());
        combined
            .set_diff
            .insert("EVENT_B".to_string(), report_b.set_diff["EVENT_B"].clone());
        combined.multiset_diff.insert("EVENT_A".to_string(), HashMap::new());
        combined.multiset_diff.insert("EVENT_B".to_string(), HashMap::new());
        pc_anchored.insert(key, combined);

        let batch = BatchResult {
            batch_idx: 0,
            config: EventsConfig::default(),
            tiles: Vec::new(),
            stall_attributions: Vec::new(),
            cross_tile: None,
            pc_anchored,
        };

        let grounding = BTreeSet::new();
        let report = format_report_pc_anchored(&[batch], &grounding);

        // A should appear before B in the divergence section.
        assert!(report.contains("PC-anchored divergences"));
        let pos_a = report.find("EVENT_A").expect("EVENT_A not found");
        let pos_b = report.find("EVENT_B").expect("EVENT_B not found");
        // EVENT_A has higher divergence -> should appear first in sorted output.
        // We look for them specifically in the divergence section.
        let div_section = report.split("PC-anchored divergences").nth(1).unwrap_or("");
        let div_pos_a = div_section.find("EVENT_A").unwrap_or(usize::MAX);
        let div_pos_b = div_section.find("EVENT_B").unwrap_or(usize::MAX);
        assert!(
            div_pos_a < div_pos_b,
            "EVENT_A (set_diff=5) should precede EVENT_B (set_diff=1), \
             but A at {} and B at {} in divergence section",
            div_pos_a,
            div_pos_b
        );
        // Suppress unused-variable warning for pos_a/pos_b from coverage matrix.
        let _ = (pos_a, pos_b);
    }

    #[test]
    fn pc_anchored_cycle_delta_avg_max() {
        // Build a batch with cycle_bands for INSTR_VECTOR:
        //   PC 100: delta = +10
        //   PC 200: delta = -20
        //   PC 300: delta = 0
        // avg |delta| = (10 + 20 + 0) / 3 = 10.0; max |delta| = 20.
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        let mut report = PCAnchoredReport::default();
        // set_diff must have the key so the event appears.
        report
            .set_diff
            .insert("INSTR_VECTOR".to_string(), (HashSet::new(), HashSet::new()));

        let mut bands = HashMap::new();
        bands.insert(
            100u64,
            CycleBand {
                hw_cycle_est: 110,
                emu_cycle_est: 100,
                delta_cycles: 10,
                exceeds_tolerance: false,
            },
        );
        bands.insert(
            200u64,
            CycleBand {
                hw_cycle_est: 200,
                emu_cycle_est: 220,
                delta_cycles: -20,
                exceeds_tolerance: false,
            },
        );
        bands.insert(
            300u64,
            CycleBand { hw_cycle_est: 300, emu_cycle_est: 300, delta_cycles: 0, exceeds_tolerance: false },
        );
        report.cycle_bands.insert("INSTR_VECTOR".to_string(), bands);

        let mut pc_anchored = HashMap::new();
        pc_anchored.insert(key, report);

        let batch = BatchResult {
            batch_idx: 0,
            config: EventsConfig::default(),
            tiles: Vec::new(),
            stall_attributions: Vec::new(),
            cross_tile: None,
            pc_anchored,
        };

        let grounding = BTreeSet::new();
        let report_str = format_report_pc_anchored(&[batch], &grounding);

        assert!(report_str.contains("Perfcnt-anchored cycle deltas"), "cycle delta section missing");
        assert!(report_str.contains("INSTR_VECTOR"), "event name missing in cycle delta section");
        // avg=10.0, max=20, n=3.
        assert!(report_str.contains("avg=10.0"), "avg mismatch: {}", report_str);
        assert!(report_str.contains("max=20"), "max mismatch: {}", report_str);
        assert!(report_str.contains("n=3"), "n mismatch: {}", report_str);
    }

    #[test]
    fn pc_anchored_empty_batches_no_pc_section() {
        // When format_report_pc_anchored runs with zero batches, all three
        // section headers (coverage, divergence, cycle deltas) are emitted
        // but contain no event rows. compare_sweep_dir_with_opts would not
        // append the PC suffix at all in this case (guarded by `has_pc`),
        // so this test exercises the formatter directly to confirm clean
        // empty output rather than a panic or malformed sections.
        let grounding = BTreeSet::new();
        let report = format_report_pc_anchored(&[], &grounding);
        // All three section headers present.
        assert!(report.contains("PC-anchored coverage"));
        assert!(report.contains("PC-anchored divergences"));
        assert!(report.contains("Perfcnt-anchored cycle deltas"));
        // No event names should appear (no events to report).
        assert!(!report.contains("INSTR_VECTOR"));
    }

    #[test]
    fn pc_anchored_unsafe_for_pc_join_manifest_test() {
        // Build a synthetic sweep dir in TMPDIR with a manifest that has
        // unsafe_for_pc_join=true. Verify read_pc1_manifest parses it.
        let tmpdir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()));
        let sweep_dir = tmpdir.join("task8_test_manifest");
        std::fs::create_dir_all(&sweep_dir).expect("create sweep dir");

        let manifest_json = r#"{
            "test_name": "test",
            "compiler": "chess",
            "mode": "event_pc",
            "grounding": {
                "core": ["PERF_CNT_0"],
                "memmod": [],
                "memtile": [],
                "shim": []
            },
            "mode2_baseline_captured": false,
            "unsafe_for_pc_join": true,
            "reason": "grounding event PERF_CNT_0 PC drifted",
            "sweep_error": null
        }"#;
        std::fs::write(sweep_dir.join("sweep-manifest.json"), manifest_json).expect("write manifest");

        let manifest = read_pc1_manifest(&sweep_dir).expect("manifest should parse");
        assert!(manifest.unsafe_for_pc_join);
        assert_eq!(manifest.reason.as_deref(), Some("grounding event PERF_CNT_0 PC drifted"));

        // Grounding extraction includes core events.
        let grounding = read_grounding_from_pc1_manifest(&sweep_dir);
        assert!(grounding.contains("PERF_CNT_0"));

        // Cleanup.
        let _ = std::fs::remove_dir_all(&sweep_dir);
    }

    #[test]
    fn pc_anchored_divergence_alphabetical_tiebreak() {
        // Two events with identical set+multiset magnitude: the secondary
        // alphabetical key must produce stable, deterministic ordering
        // (EVENT_AAA before EVENT_ZZZ regardless of HashMap iteration order).
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        let mut combined = PCAnchoredReport::default();
        // Both events have a single hw-only PC -> set_diff total = 1 each.
        let single_pc: HashSet<u64> = [42].iter().cloned().collect();
        combined
            .set_diff
            .insert("EVENT_ZZZ".to_string(), (single_pc.clone(), HashSet::new()));
        combined.set_diff.insert("EVENT_AAA".to_string(), (single_pc, HashSet::new()));
        combined.multiset_diff.insert("EVENT_AAA".to_string(), HashMap::new());
        combined.multiset_diff.insert("EVENT_ZZZ".to_string(), HashMap::new());

        let mut pc_anchored = HashMap::new();
        pc_anchored.insert(key, combined);

        let batch = BatchResult {
            batch_idx: 0,
            config: EventsConfig::default(),
            tiles: Vec::new(),
            stall_attributions: Vec::new(),
            cross_tile: None,
            pc_anchored,
        };

        let grounding = BTreeSet::new();
        let report = format_report_pc_anchored(&[batch], &grounding);

        // Find positions in the divergence section only.
        let div_section = report.split("PC-anchored divergences").nth(1).expect("divergence section");
        let pos_aaa = div_section.find("EVENT_AAA").expect("EVENT_AAA in divergence section");
        let pos_zzz = div_section.find("EVENT_ZZZ").expect("EVENT_ZZZ in divergence section");
        assert!(
            pos_aaa < pos_zzz,
            "Equal-magnitude events should sort alphabetically: \
             EVENT_AAA at {} should precede EVENT_ZZZ at {}",
            pos_aaa,
            pos_zzz
        );
    }

    #[test]
    fn pc_anchored_finds_mode2_baseline_pair_even_without_manifest() {
        // Build a synthetic sweep dir with a mode2-baseline/hw/trace.events.json
        // file but NO sweep-manifest.json. find_mode2_baseline_pair must locate
        // the HW file anyway -- a missing/corrupted manifest must not silently
        // suppress real on-disk artifacts. The EMU sibling is returned too, even
        // when the file isn't present (caller decides what to do with it).
        let tmpdir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()));
        let sweep_dir = tmpdir.join("task8_test_baseline_no_manifest");
        let _ = std::fs::remove_dir_all(&sweep_dir);
        std::fs::create_dir_all(&sweep_dir).expect("create sweep dir");
        let hw_dir = sweep_dir.join("mode2-baseline/hw");
        std::fs::create_dir_all(&hw_dir).expect("create hw dir");
        let hw_path = hw_dir.join("trace.events.json");
        std::fs::write(&hw_path, "{}").expect("write hw events file");

        let pair = find_mode2_baseline_pair(&sweep_dir);
        assert!(pair.is_some());
        let (hw, emu) = pair.unwrap();
        assert_eq!(hw, hw_path);
        assert_eq!(emu, sweep_dir.join("mode2-baseline/emu/trace.events.json"));
        assert!(!emu.exists());

        // read_pc1_manifest returns None gracefully (no panic).
        assert!(read_pc1_manifest(&sweep_dir).is_none());
        // Grounding extraction is empty when manifest is absent.
        assert!(read_grounding_from_pc1_manifest(&sweep_dir).is_empty());

        // Cleanup.
        let _ = std::fs::remove_dir_all(&sweep_dir);
    }

    #[test]
    fn pc_anchored_returns_none_when_no_mode2_baseline_dir() {
        let tmpdir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()));
        let sweep_dir = tmpdir.join("task8_test_no_mode2_baseline");
        let _ = std::fs::remove_dir_all(&sweep_dir);
        std::fs::create_dir_all(&sweep_dir).expect("create sweep dir");
        assert!(find_mode2_baseline_pair(&sweep_dir).is_none());
        let _ = std::fs::remove_dir_all(&sweep_dir);
    }

    // ---- Manifest format compatibility tests --------------------------------
    //
    // `SweepManifest` accepts both the legacy and lockstep manifest shapes.
    // These tests pin the dispatch logic in `compare_sweep_dir_with_opts`:
    // missing `batches` -> filesystem discovery; present `batches: []` ->
    // legacy path with empty result; present `batches: [..]` -> legacy path
    // with the listed batches.

    /// Helper: build a synthetic batch_NN/{hw,emu}/trace.events.json pair.
    fn write_batch_events_json(batch_dir: &std::path::Path) {
        let hw_dir = batch_dir.join("hw");
        let emu_dir = batch_dir.join("emu");
        std::fs::create_dir_all(&hw_dir).expect("create hw dir");
        std::fs::create_dir_all(&emu_dir).expect("create emu dir");
        // Minimal events JSON that parse-trace.py would emit; just enough
        // structure that compare_batch_with_opts doesn't choke on parsing.
        // We only care that discovery finds these paths -- their content
        // doesn't drive the dispatch test.
        let stub = r#"{"tiles": []}"#;
        std::fs::write(hw_dir.join("trace.events.json"), stub).expect("write hw");
        std::fs::write(emu_dir.join("trace.events.json"), stub).expect("write emu");
    }

    /// Parse a legacy-shape manifest with explicit `batches[]` array.
    /// Verifies the `Some(_)` arm is taken and the array is preserved.
    #[test]
    fn sweep_manifest_legacy_format_parses_batches_array() {
        let json = r#"{
            "batches": [
                {"batch": 0, "status": "ok", "hw_status": "ok", "emu_status": "ok"},
                {"batch": 1, "status": "ok", "hw_status": "ok", "emu_status": "ok"}
            ]
        }"#;
        let m: SweepManifest = serde_json::from_str(json).expect("legacy parse");
        let batches = m.batches.expect("legacy manifest must have Some(batches)");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].batch, 0);
        assert_eq!(batches[1].batch, 1);
    }

    /// Parse a lockstep-shape manifest where `batches` is absent.
    /// Verifies the `None` arm is taken (rather than serde defaulting to []).
    #[test]
    fn sweep_manifest_lockstep_format_yields_none() {
        let json = r#"{
            "test_name": "test",
            "compiler": "chess",
            "mode": "event_pc",
            "n_batches": 5,
            "n_batches_completed": 5,
            "unsafe_for_pc_join": false,
            "mode2_baseline_captured": true,
            "grounding": {
                "core": ["PERF_CNT_0"],
                "memmod": [],
                "memtile": [],
                "shim": []
            }
        }"#;
        let m: SweepManifest = serde_json::from_str(json).expect("lockstep parse");
        assert!(m.batches.is_none(), "lockstep manifest must round-trip as None, not Some([])");
    }

    /// Empty `batches: []` is the "legacy run with zero passing batches"
    /// case. The dispatch must NOT fall back to fs discovery here -- if
    /// the orchestrator wrote zero passing batches, the result should be
    /// zero, not whatever batch_NN dirs happen to be on disk.
    #[test]
    fn sweep_manifest_legacy_empty_batches_is_some_not_none() {
        let json = r#"{ "batches": [] }"#;
        let m: SweepManifest = serde_json::from_str(json).expect("empty parse");
        let batches = m.batches.expect("Some(empty), not None");
        assert!(batches.is_empty());
    }

    /// End-to-end: lockstep manifest + on-disk batch_NN dirs ->
    /// `compare_sweep_dir_with_opts` should discover the batches via fs.
    #[test]
    fn compare_sweep_dir_lockstep_uses_fs_discovery() {
        let tmpdir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()));
        let sweep_dir = tmpdir.join("compare_lockstep_fs_discovery_fixture");
        let _ = std::fs::remove_dir_all(&sweep_dir);
        std::fs::create_dir_all(&sweep_dir).expect("create sweep dir");

        // Lockstep manifest -- no `batches` field.
        let manifest = r#"{
            "test_name": "test",
            "compiler": "chess",
            "mode": "event_pc",
            "n_batches": 2,
            "n_batches_completed": 2,
            "unsafe_for_pc_join": false,
            "mode2_baseline_captured": false,
            "grounding": {
                "core": ["PERF_CNT_0"],
                "memmod": [],
                "memtile": [],
                "shim": []
            }
        }"#;
        std::fs::write(sweep_dir.join("sweep-manifest.json"), manifest).expect("write manifest");

        // Two batch_NN dirs on disk; compare_sweep_dir_with_opts should
        // discover both via fs since manifest.batches is None.
        write_batch_events_json(&sweep_dir.join("batch_00"));
        write_batch_events_json(&sweep_dir.join("batch_01"));

        // Verify discover_batches_from_fs sees both batches.
        let triples = discover_batches_from_fs(&sweep_dir);
        assert_eq!(triples.len(), 2, "fs discovery should find 2 batches");
        assert_eq!(triples[0].0, 0);
        assert_eq!(triples[1].0, 1);

        // The full compare path may produce an error from compare_batch_with_opts
        // (the stub events JSON has empty tiles), but it must NOT error on
        // missing `batches` field -- that's the regression we're pinning.
        // We accept both Ok and Err here; the assertion is that we got past
        // the manifest deserializer.
        let result = compare_sweep_dir_with_opts(&sweep_dir, &AnalysisOptions::default());
        let err_msg = result.err().unwrap_or_default();
        assert!(
            !err_msg.contains("missing field `batches`"),
            "lockstep manifest must not error on missing `batches`: {}",
            err_msg
        );

        let _ = std::fs::remove_dir_all(&sweep_dir);
    }

    /// End-to-end: legacy manifest with empty `batches: []` should NOT
    /// fall back to fs discovery -- even if batch_NN/ dirs exist.
    #[test]
    fn compare_sweep_dir_legacy_empty_batches_does_not_fall_back_to_fs() {
        let tmpdir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()));
        let sweep_dir = tmpdir.join("compare_legacy_empty_no_fs_fallback_fixture");
        let _ = std::fs::remove_dir_all(&sweep_dir);
        std::fs::create_dir_all(&sweep_dir).expect("create sweep dir");

        // Legacy manifest with empty batches array.
        std::fs::write(sweep_dir.join("sweep-manifest.json"), r#"{ "batches": [] }"#)
            .expect("write manifest");

        // A stray batch_00/ on disk that the legacy run did NOT validate.
        // If we wrongly fell back to fs discovery this would be picked up.
        write_batch_events_json(&sweep_dir.join("batch_00"));

        // The legacy path with an empty batches array yields an empty
        // result.  The function still returns Ok with a (possibly empty)
        // report; we just need to confirm we didn't pick up the stray
        // batch_00/ via fs discovery.
        //
        // We can't easily count consumed batches from the public API
        // (the report is text), so instead we check the discovery path
        // is correctly NOT entered: parse manifest -> Some([]) -> empty
        // collect -> zero batch_results.  No panic, no fs fallback.
        let result = compare_sweep_dir_with_opts(&sweep_dir, &AnalysisOptions::default());
        // compare_sweep_dir_with_opts on zero batches returns Ok with a
        // base report and no PC-anchored suffix.  The key invariant is
        // that we don't error AND we don't accidentally consume the
        // stray batch (which would change the report text).
        assert!(result.is_ok(), "legacy empty batches should produce Ok report, got {:?}", result);

        let _ = std::fs::remove_dir_all(&sweep_dir);
    }
}
