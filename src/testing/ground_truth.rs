//! NPU ground truth: canonical trace capture, storage, and comparison.
//!
//! Captures what the real NPU does for each test kernel, strips
//! nondeterministic noise, and compares emulator output against it.
//! Three comparison layers: functional output, event sequence, timing deltas.
//!
//! Ground truth files live in `build/ground-truth/<test>/` and are
//! regenerated from the NPU on every run (NPU execution is cheap).

use std::collections::HashMap;
use std::path::Path;

use serde::{Serialize, Deserialize};

/// Canonical trace version. Bump when the format changes.
pub const CURRENT_VERSION: u32 = 1;

/// Which event configuration was used for capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventConfig {
    /// Base 8 events (single NPU run).
    Base8,
    /// Full 13-group sweep (statistical classification).
    FullSweep,
}

/// Identifies a tile and trace module.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileId {
    pub col: u8,
    pub row: u8,
    pub trace_type: TraceModule,
}

impl std::fmt::Display for TileId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let module = match self.trace_type {
            TraceModule::Core => "core",
            TraceModule::Mem => "mem",
        };
        write!(f, "{}({},{})", module, self.col, self.row)
    }
}

/// Which trace module within a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceModule {
    Core,
    Mem,
}

/// A single event in the canonical trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalEvent {
    /// Event name (e.g., "INSTR_VECTOR").
    pub name: String,
    /// Cycles since previous event (0 for first event).
    pub delta: u64,
}

/// Canonical trace for a single tile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileTrace {
    /// Deterministic events in order, with stable inter-event deltas.
    pub events: Vec<CanonicalEvent>,
    /// Event names that were filtered as nondeterministic.
    pub filtered_events: Vec<String>,
}

/// Complete canonical trace for all tiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalTrace {
    /// Format version (for forward compatibility).
    pub version: u32,
    /// Which event configuration produced this trace.
    pub event_config: EventConfig,
    /// Per-tile canonical event sequences.
    ///
    /// Serialized as a JSON map with string keys like "core(2,0)".
    #[serde(serialize_with = "serialize_tiles", deserialize_with = "deserialize_tiles")]
    pub tiles: HashMap<TileId, TileTrace>,
}

fn serialize_tiles<S: serde::Serializer>(
    tiles: &HashMap<TileId, TileTrace>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(tiles.len()))?;
    for (id, trace) in tiles {
        map.serialize_entry(&format!("{}", id), trace)?;
    }
    map.end()
}

fn deserialize_tiles<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<HashMap<TileId, TileTrace>, D::Error> {
    let string_map: HashMap<String, TileTrace> = HashMap::deserialize(deserializer)?;
    let mut tiles = HashMap::new();
    for (key, trace) in string_map {
        let id = parse_tile_id(&key).map_err(serde::de::Error::custom)?;
        tiles.insert(id, trace);
    }
    Ok(tiles)
}

/// Parse a TileId from its Display format: "core(2,0)" or "mem(1,3)".
fn parse_tile_id(s: &str) -> Result<TileId, String> {
    let (module_str, rest) = if let Some(r) = s.strip_prefix("core(") {
        (TraceModule::Core, r)
    } else if let Some(r) = s.strip_prefix("mem(") {
        (TraceModule::Mem, r)
    } else {
        return Err(format!("invalid tile id: {}", s));
    };
    let rest = rest.strip_suffix(')').ok_or_else(|| format!("missing ')': {}", s))?;
    let parts: Vec<&str> = rest.split(',').collect();
    if parts.len() != 2 {
        return Err(format!("expected col,row: {}", s));
    }
    let col: u8 = parts[0].parse().map_err(|_| format!("bad col: {}", s))?;
    let row: u8 = parts[1].parse().map_err(|_| format!("bad row: {}", s))?;
    Ok(TileId { col, row, trace_type: module_str })
}

/// Capture metadata (written alongside the canonical trace).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureMetadata {
    pub captured_at: String,
    pub kernel_hash: String,
    pub driver_version: String,
    pub event_config: EventConfig,
}

// ---------------------------------------------------------------------------
// Canonicalization
// ---------------------------------------------------------------------------

/// Known nondeterministic event names (filtered in single-run mode).
///
/// These events fire unpredictably due to runtime timing variance
/// (DMA pipeline depth, lock contention, memory bank conflicts).
pub const NONDETERMINISTIC_EVENTS: &[&str] = &[
    "LOCK_STALL",
    "MEMORY_STALL",
    "STREAM_STALL",
    "CASCADE_STALL",
];

/// Canonicalize decoded events for a single tile.
///
/// Filters out events whose slot name is in `nondeterministic`, aligns
/// remaining events to delta-from-previous, and returns a `TileTrace`.
pub fn canonicalize_tile_events(
    events: &[crate::fuzzer::trace_sweep::DecodedEvent],
    slot_names: &[&str; 8],
    nondeterministic: &[&str],
) -> TileTrace {
    let mut filtered_set: Vec<String> = Vec::new();
    let mut kept: Vec<(u64, &str)> = Vec::new();

    for ev in events {
        let idx = ev.slot as usize;
        if idx >= slot_names.len() {
            continue;
        }
        let name = slot_names[idx];
        if nondeterministic.contains(&name) {
            if !filtered_set.contains(&name.to_string()) {
                filtered_set.push(name.to_string());
            }
        } else {
            kept.push((ev.abs_cycle, name));
        }
    }

    let mut canonical_events = Vec::with_capacity(kept.len());
    let mut prev_cycle = kept.first().map(|(c, _)| *c).unwrap_or(0);

    for (i, (cycle, name)) in kept.iter().enumerate() {
        let delta = if i == 0 { 0 } else { cycle.saturating_sub(prev_cycle) };
        canonical_events.push(CanonicalEvent {
            name: name.to_string(),
            delta,
        });
        prev_cycle = *cycle;
    }

    filtered_set.sort();

    TileTrace {
        events: canonical_events,
        filtered_events: filtered_set,
    }
}

// ---------------------------------------------------------------------------
// Comparison: output buffers
// ---------------------------------------------------------------------------

/// Result of comparing functional output buffers.
#[derive(Debug)]
pub struct OutputComparison {
    pub total_bytes: usize,
    pub differing_bytes: usize,
    pub first_diff_byte: Option<usize>,
}

impl OutputComparison {
    pub fn is_match(&self) -> bool { self.differing_bytes == 0 }
}

/// Byte-for-byte comparison of DMA output buffers.
pub fn compare_output(expected: &[u8], actual: &[u8]) -> OutputComparison {
    let total_bytes = expected.len().max(actual.len());
    let mut differing_bytes = 0;
    let mut first_diff_byte = None;

    for i in 0..total_bytes {
        let a = expected.get(i).copied().unwrap_or(0);
        let b = actual.get(i).copied().unwrap_or(0);
        if a != b {
            differing_bytes += 1;
            if first_diff_byte.is_none() {
                first_diff_byte = Some(i);
            }
        }
    }

    OutputComparison { total_bytes, differing_bytes, first_diff_byte }
}

// ---------------------------------------------------------------------------
// Comparison: event sequences
// ---------------------------------------------------------------------------

/// Result of comparing event sequences.
#[derive(Debug)]
pub struct EventComparison {
    pub expected_len: usize,
    pub actual_len: usize,
    /// Index of first event name mismatch.
    pub first_divergence: Option<usize>,
}

impl EventComparison {
    pub fn is_match(&self) -> bool {
        self.first_divergence.is_none() && self.expected_len == self.actual_len
    }
}

/// Compare event sequences by name (ignoring timing).
pub fn compare_events(expected: &[CanonicalEvent], actual: &[CanonicalEvent]) -> EventComparison {
    let min_len = expected.len().min(actual.len());
    let mut first_divergence = None;

    for i in 0..min_len {
        if expected[i].name != actual[i].name {
            first_divergence = Some(i);
            break;
        }
    }

    if first_divergence.is_none() && expected.len() != actual.len() {
        first_divergence = Some(min_len);
    }

    EventComparison {
        expected_len: expected.len(),
        actual_len: actual.len(),
        first_divergence,
    }
}

// ---------------------------------------------------------------------------
// Comparison: timing deltas
// ---------------------------------------------------------------------------

/// Result of comparing timing deltas.
#[derive(Debug)]
pub struct TimingComparison {
    pub events_compared: usize,
    /// First event where delta differs beyond tolerance: (index, expected, actual).
    pub first_divergence: Option<(usize, u64, u64)>,
    pub max_delta_diff: u64,
}

impl TimingComparison {
    pub fn is_match(&self) -> bool { self.first_divergence.is_none() }
}

/// Compare inter-event deltas with configurable tolerance.
pub fn compare_timing(
    expected: &[CanonicalEvent],
    actual: &[CanonicalEvent],
    tolerance: u64,
) -> TimingComparison {
    let min_len = expected.len().min(actual.len());
    let mut first_divergence = None;
    let mut max_delta_diff = 0u64;

    for i in 0..min_len {
        let diff = expected[i].delta.abs_diff(actual[i].delta);
        max_delta_diff = max_delta_diff.max(diff);
        if diff > tolerance && first_divergence.is_none() {
            first_divergence = Some((i, expected[i].delta, actual[i].delta));
        }
    }

    TimingComparison {
        events_compared: min_len,
        first_divergence,
        max_delta_diff,
    }
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// Overall ground truth verdict for a test.
#[derive(Debug)]
pub enum GroundTruthVerdict {
    /// All three layers match.
    Match,
    /// Match, but NPU-vs-stored-truth shows drift (informational).
    MatchWithDrift { detail: String },
    /// DMA output buffers differ.
    DivergeOutput { first_diff: usize, differing: usize, total: usize },
    /// Event sequence differs.
    DivergeEvents { tile: String, position: usize },
    /// Timing deltas differ beyond tolerance.
    DivergeTiming { tile: String, position: usize, expected: u64, actual: u64 },
    /// First capture (no prior ground truth).
    New,
}

impl std::fmt::Display for GroundTruthVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Match => write!(f, "MATCH"),
            Self::MatchWithDrift { detail } => write!(f, "MATCH (drift: {})", detail),
            Self::DivergeOutput { first_diff, differing, total } =>
                write!(f, "DIVERGE output (byte {}, {}/{} differ)", first_diff, differing, total),
            Self::DivergeEvents { tile, position } =>
                write!(f, "DIVERGE events in {} at position {}", tile, position),
            Self::DivergeTiming { tile, position, expected, actual } =>
                write!(f, "DIVERGE timing in {} at position {} (expected {}, got {})",
                    tile, position, expected, actual),
            Self::New => write!(f, "NEW (first capture)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Full comparison pipeline
// ---------------------------------------------------------------------------

/// Compare emulator tile trace against NPU ground truth tile trace.
///
/// Checks layers in priority order: output first (most critical),
/// then events, then timing.
pub fn compare_ground_truth(
    npu_tile: &TileTrace,
    emu_tile: &TileTrace,
    npu_output: &[u8],
    emu_output: &[u8],
    timing_tolerance: u64,
) -> GroundTruthVerdict {
    // Layer 1: functional output
    let output_cmp = compare_output(npu_output, emu_output);
    if !output_cmp.is_match() {
        return GroundTruthVerdict::DivergeOutput {
            first_diff: output_cmp.first_diff_byte.unwrap_or(0),
            differing: output_cmp.differing_bytes,
            total: output_cmp.total_bytes,
        };
    }

    // Layer 2: event sequence
    let event_cmp = compare_events(&npu_tile.events, &emu_tile.events);
    if !event_cmp.is_match() {
        return GroundTruthVerdict::DivergeEvents {
            tile: "tile".into(),
            position: event_cmp.first_divergence.unwrap_or(0),
        };
    }

    // Layer 3: timing deltas
    let timing_cmp = compare_timing(&npu_tile.events, &emu_tile.events, timing_tolerance);
    if !timing_cmp.is_match() {
        if let Some((pos, exp, act)) = timing_cmp.first_divergence {
            return GroundTruthVerdict::DivergeTiming {
                tile: "tile".into(),
                position: pos,
                expected: exp,
                actual: act,
            };
        }
    }

    GroundTruthVerdict::Match
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Save ground truth files to a directory.
pub fn save_ground_truth(
    dir: &Path,
    trace: &CanonicalTrace,
    output: &[u8],
    metadata: &CaptureMetadata,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let trace_json = serde_json::to_string_pretty(trace)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(dir.join("canonical.json"), trace_json)?;
    std::fs::write(dir.join("output.bin"), output)?;
    let meta_json = serde_json::to_string_pretty(metadata)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(dir.join("metadata.json"), meta_json)?;
    Ok(())
}

/// Load ground truth from a directory. Returns None if files don't exist.
pub fn load_ground_truth(dir: &Path) -> Option<(CanonicalTrace, Vec<u8>)> {
    let trace_path = dir.join("canonical.json");
    let output_path = dir.join("output.bin");
    if !trace_path.exists() || !output_path.exists() {
        return None;
    }
    let trace_json = std::fs::read_to_string(&trace_path).ok()?;
    let trace: CanonicalTrace = serde_json::from_str(&trace_json).ok()?;
    let output = std::fs::read(&output_path).ok()?;
    Some((trace, output))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple timestamp without chrono dependency.
pub fn timestamp_now() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", duration.as_secs())
}

/// Read kernel version from /proc/version.
pub fn kernel_version() -> String {
    std::fs::read_to_string("/proc/version")
        .ok()
        .and_then(|v| v.split_whitespace().nth(2).map(String::from))
        .unwrap_or_else(|| "unknown".into())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_trace_round_trip() {
        let trace = CanonicalTrace {
            version: 1,
            event_config: EventConfig::Base8,
            tiles: {
                let mut m = HashMap::new();
                m.insert(
                    TileId { col: 2, row: 0, trace_type: TraceModule::Core },
                    TileTrace {
                        events: vec![
                            CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
                            CanonicalEvent { name: "INSTR_LOAD".into(), delta: 3 },
                        ],
                        filtered_events: vec!["LOCK_STALL".into()],
                    },
                );
                m
            },
        };

        let json = serde_json::to_string_pretty(&trace).unwrap();
        let parsed: CanonicalTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.tiles.len(), 1);

        let tile_id = TileId { col: 2, row: 0, trace_type: TraceModule::Core };
        let tile = &parsed.tiles[&tile_id];
        assert_eq!(tile.events.len(), 2);
        assert_eq!(tile.events[0].name, "INSTR_VECTOR");
        assert_eq!(tile.events[0].delta, 0);
        assert_eq!(tile.events[1].delta, 3);
        assert_eq!(tile.filtered_events, vec!["LOCK_STALL"]);
    }

    #[test]
    fn test_tile_id_display() {
        let id = TileId { col: 2, row: 0, trace_type: TraceModule::Core };
        assert_eq!(format!("{}", id), "core(2,0)");
        let id = TileId { col: 1, row: 3, trace_type: TraceModule::Mem };
        assert_eq!(format!("{}", id), "mem(1,3)");
    }

    #[test]
    fn test_canonicalize_filters_nondeterministic() {
        use crate::fuzzer::trace_sweep::DecodedEvent;

        let events = vec![
            DecodedEvent { slot: 0, abs_cycle: 10 },
            DecodedEvent { slot: 3, abs_cycle: 12 },  // LOCK_STALL
            DecodedEvent { slot: 0, abs_cycle: 15 },
            DecodedEvent { slot: 1, abs_cycle: 18 },
        ];

        let slot_names = [
            "INSTR_VECTOR", "INSTR_LOAD", "INSTR_STORE", "LOCK_STALL",
            "INSTR_CALL", "PORT_RUNNING_0", "PORT_RUNNING_1", "ACTIVE",
        ];

        let tile_trace = canonicalize_tile_events(&events, &slot_names, NONDETERMINISTIC_EVENTS);

        assert_eq!(tile_trace.events.len(), 3);
        assert_eq!(tile_trace.events[0].name, "INSTR_VECTOR");
        assert_eq!(tile_trace.events[0].delta, 0);
        assert_eq!(tile_trace.events[1].name, "INSTR_VECTOR");
        assert_eq!(tile_trace.events[1].delta, 5);  // 15 - 10
        assert_eq!(tile_trace.events[2].name, "INSTR_LOAD");
        assert_eq!(tile_trace.events[2].delta, 3);  // 18 - 15
        assert!(tile_trace.filtered_events.contains(&"LOCK_STALL".to_string()));
    }

    #[test]
    fn test_canonicalize_empty_events() {
        let events = vec![];
        let slot_names = [
            "A", "B", "C", "D", "E", "F", "G", "H",
        ];
        let tile = canonicalize_tile_events(&events, &slot_names, NONDETERMINISTIC_EVENTS);
        assert!(tile.events.is_empty());
        assert!(tile.filtered_events.is_empty());
    }

    #[test]
    fn test_compare_output_match() {
        let result = compare_output(b"hello", b"hello");
        assert!(result.is_match());
    }

    #[test]
    fn test_compare_output_diverge() {
        let result = compare_output(b"hello", b"hxllo");
        assert!(!result.is_match());
        assert_eq!(result.first_diff_byte, Some(1));
        assert_eq!(result.total_bytes, 5);
        assert_eq!(result.differing_bytes, 1);
    }

    #[test]
    fn test_compare_output_different_lengths() {
        let result = compare_output(b"hello", b"hel");
        assert!(!result.is_match());
        assert_eq!(result.total_bytes, 5);
        assert_eq!(result.differing_bytes, 2);  // 'l' and 'o' missing
    }

    #[test]
    fn test_compare_events_match() {
        let a = vec![
            CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
            CanonicalEvent { name: "INSTR_LOAD".into(), delta: 3 },
        ];
        let b = a.clone();
        let result = compare_events(&a, &b);
        assert!(result.is_match());
    }

    #[test]
    fn test_compare_events_diverge_name() {
        let a = vec![
            CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
            CanonicalEvent { name: "INSTR_LOAD".into(), delta: 3 },
        ];
        let b = vec![
            CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
            CanonicalEvent { name: "INSTR_STORE".into(), delta: 3 },
        ];
        let result = compare_events(&a, &b);
        assert!(!result.is_match());
        assert_eq!(result.first_divergence, Some(1));
    }

    #[test]
    fn test_compare_events_diverge_length() {
        let a = vec![
            CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
            CanonicalEvent { name: "INSTR_LOAD".into(), delta: 3 },
        ];
        let b = vec![
            CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
        ];
        let result = compare_events(&a, &b);
        assert!(!result.is_match());
        assert_eq!(result.first_divergence, Some(1));
    }

    #[test]
    fn test_compare_timing_match() {
        let a = vec![
            CanonicalEvent { name: "A".into(), delta: 0 },
            CanonicalEvent { name: "B".into(), delta: 5 },
        ];
        let b = a.clone();
        let result = compare_timing(&a, &b, 0);
        assert!(result.is_match());
    }

    #[test]
    fn test_compare_timing_within_tolerance() {
        let a = vec![
            CanonicalEvent { name: "A".into(), delta: 0 },
            CanonicalEvent { name: "B".into(), delta: 5 },
        ];
        let b = vec![
            CanonicalEvent { name: "A".into(), delta: 0 },
            CanonicalEvent { name: "B".into(), delta: 7 },
        ];
        assert!(!compare_timing(&a, &b, 0).is_match());
        assert!(compare_timing(&a, &b, 2).is_match());
    }

    #[test]
    fn test_compare_timing_max_diff() {
        let a = vec![
            CanonicalEvent { name: "A".into(), delta: 0 },
            CanonicalEvent { name: "B".into(), delta: 10 },
            CanonicalEvent { name: "C".into(), delta: 20 },
        ];
        let b = vec![
            CanonicalEvent { name: "A".into(), delta: 0 },
            CanonicalEvent { name: "B".into(), delta: 13 },
            CanonicalEvent { name: "C".into(), delta: 18 },
        ];
        let result = compare_timing(&a, &b, 5);
        assert!(result.is_match());
        assert_eq!(result.max_delta_diff, 3);
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", GroundTruthVerdict::Match), "MATCH");
        assert_eq!(format!("{}", GroundTruthVerdict::New), "NEW (first capture)");

        let v = GroundTruthVerdict::DivergeOutput {
            first_diff: 128,
            differing: 16,
            total: 1024,
        };
        assert!(format!("{}", v).contains("DIVERGE output"));
        assert!(format!("{}", v).contains("128"));
    }

    #[test]
    fn test_save_and_load_ground_truth() {
        let dir = std::path::PathBuf::from(
            std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp/claude-1000".into())
        ).join("gt-test-save-load");
        let _ = std::fs::remove_dir_all(&dir);

        let trace = CanonicalTrace {
            version: 1,
            event_config: EventConfig::Base8,
            tiles: HashMap::new(),
        };
        let output = b"test output data";
        let meta = CaptureMetadata {
            captured_at: "2026-03-01T00:00:00Z".into(),
            kernel_hash: "sha256:abc".into(),
            driver_version: "6.19.0".into(),
            event_config: EventConfig::Base8,
        };

        save_ground_truth(&dir, &trace, output, &meta).unwrap();

        let loaded = load_ground_truth(&dir);
        assert!(loaded.is_some());
        let (loaded_trace, loaded_output) = loaded.unwrap();
        assert_eq!(loaded_trace.version, 1);
        assert_eq!(loaded_output, output);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_nonexistent_ground_truth() {
        let dir = std::path::Path::new("/tmp/claude-1000/gt-nonexistent-test");
        assert!(load_ground_truth(dir).is_none());
    }

    #[test]
    fn test_full_comparison_match() {
        use crate::fuzzer::trace_sweep::DecodedEvent;

        let events = vec![
            DecodedEvent { slot: 0, abs_cycle: 10 },
            DecodedEvent { slot: 1, abs_cycle: 15 },
        ];
        let slot_names = [
            "INSTR_VECTOR", "INSTR_LOAD", "INSTR_STORE", "LOCK_STALL",
            "INSTR_CALL", "PORT_RUNNING_0", "PORT_RUNNING_1", "ACTIVE",
        ];

        let npu_trace = canonicalize_tile_events(&events, &slot_names, NONDETERMINISTIC_EVENTS);
        let emu_trace = canonicalize_tile_events(&events, &slot_names, NONDETERMINISTIC_EVENTS);

        let verdict = compare_ground_truth(
            &npu_trace, &emu_trace,
            b"result", b"result",
            0,
        );
        assert!(matches!(verdict, GroundTruthVerdict::Match));
    }

    #[test]
    fn test_full_comparison_output_diverge() {
        let npu_trace = TileTrace { events: vec![], filtered_events: vec![] };
        let emu_trace = npu_trace.clone();

        let verdict = compare_ground_truth(
            &npu_trace, &emu_trace,
            b"correct", b"wrong!!",
            0,
        );
        assert!(matches!(verdict, GroundTruthVerdict::DivergeOutput { .. }));
    }

    #[test]
    fn test_full_comparison_events_diverge() {
        let npu_trace = TileTrace {
            events: vec![
                CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
                CanonicalEvent { name: "INSTR_LOAD".into(), delta: 3 },
            ],
            filtered_events: vec![],
        };
        let emu_trace = TileTrace {
            events: vec![
                CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
                CanonicalEvent { name: "INSTR_STORE".into(), delta: 3 },
            ],
            filtered_events: vec![],
        };

        let verdict = compare_ground_truth(
            &npu_trace, &emu_trace,
            b"same", b"same",
            0,
        );
        assert!(matches!(verdict, GroundTruthVerdict::DivergeEvents { .. }));
    }

    #[test]
    fn test_full_comparison_timing_diverge() {
        let npu_trace = TileTrace {
            events: vec![
                CanonicalEvent { name: "A".into(), delta: 0 },
                CanonicalEvent { name: "B".into(), delta: 10 },
            ],
            filtered_events: vec![],
        };
        let emu_trace = TileTrace {
            events: vec![
                CanonicalEvent { name: "A".into(), delta: 0 },
                CanonicalEvent { name: "B".into(), delta: 20 },
            ],
            filtered_events: vec![],
        };

        let verdict = compare_ground_truth(
            &npu_trace, &emu_trace,
            b"same", b"same",
            0,
        );
        assert!(matches!(verdict, GroundTruthVerdict::DivergeTiming { .. }));
    }
}
