# NPU Ground Truth Comparison -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automated capture and comparison of canonical NPU traces and output
buffers, running every time npu-test executes, with no special flags required.

**Architecture:** A new `ground_truth` module in `src/testing/` defines the
canonical trace format (serde JSON), canonicalization pipeline (reusing
`trace_sweep` classification), and comparison logic (three layers: output,
events, timing). The existing `emu_runner` test loop gains ground truth
capture/compare as a post-processing step after each test.

**Tech Stack:** Rust, serde_json, sha2 (already a dependency), existing
`trace_sweep::decode_binary_trace` / `trace_sweep::SlotClass`.

---

## Task 1: Define canonical trace types

**Files:**
- Create: `src/testing/ground_truth.rs`
- Modify: `src/testing/mod.rs`

**Step 1: Write the failing test**

In `src/testing/ground_truth.rs`, add a test module with a round-trip test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_trace_round_trip() {
        let trace = CanonicalTrace {
            version: 1,
            event_config: EventConfig::Base8,
            tiles: {
                let mut m = std::collections::HashMap::new();
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

        let tile = parsed.tiles.values().next().unwrap();
        assert_eq!(tile.events.len(), 2);
        assert_eq!(tile.events[0].name, "INSTR_VECTOR");
        assert_eq!(tile.events[0].delta, 0);
        assert_eq!(tile.events[1].delta, 3);
        assert_eq!(tile.filtered_events, vec!["LOCK_STALL"]);
    }
}
```

**Step 2: Write the types to make the test pass**

```rust
//! NPU ground truth: canonical trace capture, storage, and comparison.
//!
//! Captures what the real NPU does for each test kernel, strips
//! nondeterministic noise, and compares emulator output against it.
//! Three comparison layers: functional output, event sequence, timing deltas.

use std::collections::HashMap;
use std::path::Path;

use serde::{Serialize, Deserialize};

/// Canonical trace version. Bump when the format changes.
const CURRENT_VERSION: u32 = 1;

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
    pub tiles: HashMap<TileId, TileTrace>,
}

/// Capture metadata (written alongside the canonical trace).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureMetadata {
    pub captured_at: String,
    pub kernel_hash: String,
    pub driver_version: String,
    pub event_config: EventConfig,
}
```

**Step 3: Register the module**

Add `pub mod ground_truth;` to `src/testing/mod.rs`.

**Step 4: Run tests**

Run: `cargo test --lib ground_truth`
Expected: PASS (round-trip serialization works)

**Step 5: Commit**

```
feat(testing): canonical trace types for NPU ground truth comparison
```

---

## Task 2: Canonicalization pipeline

**Files:**
- Modify: `src/testing/ground_truth.rs`

Converts raw binary trace data into a `CanonicalTrace` by decoding packets,
filtering nondeterministic events, and computing inter-event deltas.

**Step 1: Write the failing test**

```rust
#[test]
fn test_canonicalize_filters_nondeterministic() {
    // Build fake decoded events with known slots
    use crate::fuzzer::trace_sweep::DecodedEvent;

    let events = vec![
        DecodedEvent { slot: 0, abs_cycle: 10 },  // slot 0 = deterministic
        DecodedEvent { slot: 3, abs_cycle: 12 },  // slot 3 = nondeterministic
        DecodedEvent { slot: 0, abs_cycle: 15 },  // slot 0 again
        DecodedEvent { slot: 1, abs_cycle: 18 },  // slot 1 = deterministic
    ];

    // Slot names: base8 standard order
    let slot_names = [
        "INSTR_VECTOR", "INSTR_LOAD", "INSTR_STORE", "LOCK_STALL",
        "INSTR_CALL", "PORT_RUNNING_0", "PORT_RUNNING_1", "ACTIVE",
    ];

    let nondeterministic_slots = ["LOCK_STALL", "MEMORY_STALL", "STREAM_STALL", "CASCADE_STALL"];

    let tile_trace = canonicalize_tile_events(
        &events,
        &slot_names,
        &nondeterministic_slots,
    );

    // LOCK_STALL (slot 3) should be filtered
    assert_eq!(tile_trace.events.len(), 3);
    assert_eq!(tile_trace.events[0].name, "INSTR_VECTOR");
    assert_eq!(tile_trace.events[0].delta, 0);
    assert_eq!(tile_trace.events[1].name, "INSTR_VECTOR");
    assert_eq!(tile_trace.events[1].delta, 5);  // 15 - 10
    assert_eq!(tile_trace.events[2].name, "INSTR_LOAD");
    assert_eq!(tile_trace.events[2].delta, 3);  // 18 - 15
    assert!(tile_trace.filtered_events.contains(&"LOCK_STALL".to_string()));
}
```

**Step 2: Implement `canonicalize_tile_events`**

```rust
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
        let name = slot_names[ev.slot as usize];
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
        let delta = if i == 0 { 0 } else { cycle - prev_cycle };
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
```

**Step 3: Run tests**

Run: `cargo test --lib test_canonicalize_filters_nondeterministic`
Expected: PASS

**Step 4: Commit**

```
feat(ground-truth): canonicalization pipeline for trace events
```

---

## Task 3: Three-layer comparison logic

**Files:**
- Modify: `src/testing/ground_truth.rs`

**Step 1: Write the failing tests**

```rust
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
fn test_compare_events_diverge() {
    let a = vec![
        CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
        CanonicalEvent { name: "INSTR_LOAD".into(), delta: 3 },
    ];
    let b = vec![
        CanonicalEvent { name: "INSTR_VECTOR".into(), delta: 0 },
        // missing INSTR_LOAD
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
    // tolerance=0: diverges
    assert!(!compare_timing(&a, &b, 0).is_match());
    // tolerance=2: matches
    assert!(compare_timing(&a, &b, 2).is_match());
}
```

**Step 2: Implement comparison functions**

```rust
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

/// Result of comparing timing deltas.
#[derive(Debug)]
pub struct TimingComparison {
    pub events_compared: usize,
    /// First event where delta differs beyond tolerance.
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
```

**Step 3: Run tests**

Run: `cargo test --lib ground_truth`
Expected: All 7 tests pass

**Step 4: Commit**

```
feat(ground-truth): three-layer comparison (output, events, timing)
```

---

## Task 4: Ground truth verdict and file I/O

**Files:**
- Modify: `src/testing/ground_truth.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn test_verdict_display() {
    let v = GroundTruthVerdict::Match;
    assert_eq!(format!("{}", v), "MATCH");

    let v = GroundTruthVerdict::DivergeOutput {
        first_diff: 128,
        differing: 16,
        total: 1024,
    };
    assert!(format!("{}", v).contains("DIVERGE output"));
}

#[test]
fn test_save_and_load_ground_truth() {
    let dir = std::env::temp_dir().join("gt-test-save-load");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

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
```

**Step 2: Implement verdict, save, load**

```rust
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
```

**Step 3: Run tests**

Run: `cargo test --lib ground_truth`
Expected: All 9 tests pass

**Step 4: Commit**

```
feat(ground-truth): verdict types and file I/O
```

---

## Task 5: Full comparison pipeline

**Files:**
- Modify: `src/testing/ground_truth.rs`

Ties together canonicalization + comparison + verdict into a single function
that takes a test name, NPU trace data, emulator trace data, NPU output,
emulator output, and produces a `GroundTruthVerdict`.

**Step 1: Write the failing test**

```rust
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

    let npu_output = b"result";
    let emu_output = b"result";

    let verdict = compare_ground_truth(
        &npu_trace, &emu_trace,
        npu_output, emu_output,
        0, // timing tolerance
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
```

**Step 2: Implement `compare_ground_truth`**

```rust
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
```

**Step 3: Run tests**

Run: `cargo test --lib ground_truth`
Expected: All 11 tests pass

**Step 4: Commit**

```
feat(ground-truth): full comparison pipeline (output -> events -> timing)
```

---

## Task 6: Integrate into emu_runner

**Files:**
- Modify: `src/testing/ground_truth.rs` (add `run_ground_truth_cycle`)
- Modify: `src/testing/emu_runner.rs` (call after each test)
- Modify: `src/testing/runner_display.rs` (add GT verdict to output)

This is the integration task. It adds a top-level function
`run_ground_truth_cycle()` that takes the test name, the emulator's raw
output, and the emulator's binary trace buffer, and:
1. Runs the NPU (if available)
2. Decodes both traces
3. Canonicalizes both
4. Compares against stored ground truth (if exists)
5. Saves new ground truth
6. Compares emulator against ground truth
7. Returns `GroundTruthVerdict`

**Step 1: Add `run_ground_truth_cycle` function**

This function is the integration point. It calls out to the existing
NPU execution infrastructure (`native_hw` or `hw_executor`) for NPU
output and trace, then runs the comparison pipeline.

```rust
/// Run the full ground truth cycle for a single test.
///
/// 1. If NPU available: execute on NPU, capture output + binary trace
/// 2. Decode + canonicalize NPU trace
/// 3. If prior ground truth exists: compare NPU vs stored (detect drift)
/// 4. Save new NPU canonical as ground truth
/// 5. Decode + canonicalize emulator trace
/// 6. Compare emulator vs ground truth
/// 7. Return verdict
pub fn run_ground_truth_cycle(
    test_name: &str,
    emu_output: Option<&[u8]>,
    emu_binary_trace: Option<&[u8]>,
    npu_output: Option<&[u8]>,
    npu_binary_trace: Option<&[u8]>,
    ground_truth_dir: &Path,
    slot_names: &[&str; 8],
    timing_tolerance: u64,
) -> Option<GroundTruthVerdict> {
    // Need both NPU and emulator data to compare
    let emu_out = emu_output?;
    let npu_out = npu_output?;

    let test_dir = ground_truth_dir.join(test_name);

    // Decode traces (if available)
    let npu_trace = npu_binary_trace.map(|data| {
        let events = crate::fuzzer::trace_sweep::decode_binary_trace(data);
        canonicalize_tile_events(&events, slot_names, NONDETERMINISTIC_EVENTS)
    });

    let emu_trace = emu_binary_trace.map(|data| {
        let events = crate::fuzzer::trace_sweep::decode_binary_trace(data);
        canonicalize_tile_events(&events, slot_names, NONDETERMINISTIC_EVENTS)
    });

    // Check for drift against stored ground truth
    let prior = load_ground_truth(&test_dir);
    let drift = if let Some((prior_trace, prior_output)) = &prior {
        let first_tile = prior_trace.tiles.values().next();
        let npu_first = npu_trace.as_ref();
        if let (Some(pt), Some(nt)) = (first_tile, npu_first) {
            let out_cmp = compare_output(&prior_output, npu_out);
            if !out_cmp.is_match() {
                Some(format!("output changed ({} bytes differ)", out_cmp.differing_bytes))
            } else {
                let timing_cmp = compare_timing(&pt.events, &nt.events, 0);
                if !timing_cmp.is_match() {
                    Some(format!("timing +{}cy max drift", timing_cmp.max_delta_diff))
                } else {
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // Save new ground truth (NPU is always authoritative)
    if let Some(ref npu_t) = npu_trace {
        let trace = CanonicalTrace {
            version: CURRENT_VERSION,
            event_config: EventConfig::Base8,
            tiles: {
                let mut m = HashMap::new();
                m.insert(
                    TileId { col: 0, row: 0, trace_type: TraceModule::Core },
                    npu_t.clone(),
                );
                m
            },
        };
        let meta = CaptureMetadata {
            captured_at: chrono_or_fallback(),
            kernel_hash: "unknown".into(),
            driver_version: kernel_version(),
            event_config: EventConfig::Base8,
        };
        let _ = save_ground_truth(&test_dir, &trace, npu_out, &meta);
    }

    // Compare emulator vs ground truth
    let verdict = if let Some(ref emu_t) = emu_trace {
        if let Some(ref npu_t) = npu_trace {
            let mut v = compare_ground_truth(npu_t, emu_t, npu_out, emu_out, timing_tolerance);
            // Annotate with drift if present
            if let (GroundTruthVerdict::Match, Some(d)) = (&v, drift) {
                v = GroundTruthVerdict::MatchWithDrift { detail: d };
            }
            v
        } else {
            // No NPU trace, compare output only
            let out_cmp = compare_output(npu_out, emu_out);
            if out_cmp.is_match() {
                GroundTruthVerdict::Match
            } else {
                GroundTruthVerdict::DivergeOutput {
                    first_diff: out_cmp.first_diff_byte.unwrap_or(0),
                    differing: out_cmp.differing_bytes,
                    total: out_cmp.total_bytes,
                }
            }
        }
    } else {
        // No emulator trace, output-only comparison
        let out_cmp = compare_output(npu_out, emu_out);
        if out_cmp.is_match() {
            GroundTruthVerdict::Match
        } else {
            GroundTruthVerdict::DivergeOutput {
                first_diff: out_cmp.first_diff_byte.unwrap_or(0),
                differing: out_cmp.differing_bytes,
                total: out_cmp.total_bytes,
            }
        }
    };

    if prior.is_none() {
        return Some(GroundTruthVerdict::New);
    }

    Some(verdict)
}

fn chrono_or_fallback() -> String {
    // Use simple timestamp without requiring chrono crate
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", duration.as_secs())
}

fn kernel_version() -> String {
    std::fs::read_to_string("/proc/version")
        .ok()
        .and_then(|v| v.split_whitespace().nth(2).map(String::from))
        .unwrap_or_else(|| "unknown".into())
}
```

**Step 2: Add GT verdict to `TestResult` display**

In `runner_display.rs`, add a `ground_truth` field to `TestResult`:

```rust
pub struct TestResult {
    // ... existing fields ...
    /// Ground truth comparison verdict (if NPU available).
    pub ground_truth: Option<GroundTruthVerdict>,
}
```

And in `format_result`, append the verdict:

```rust
// After existing outcome display
if let Some(ref gt) = r.ground_truth {
    out.push_str(&format!("  [GT: {}]", gt));
}
```

**Step 3: Wire into emu_runner's process_test**

In `emu_runner.rs`, after the test runs and hardware validation completes,
call `run_ground_truth_cycle` if NPU output is available. This is a
post-processing step -- it doesn't change existing test flow.

The exact wiring depends on how `process_test` currently handles
hardware results. The key insertion point is after
`hw_executor::run_on_npu()` returns, where both `emu_output` and
`npu_output` are available.

**Step 4: Run full test suite**

Run: `cargo test --lib`
Expected: All existing tests pass, no regressions

**Step 5: Commit**

```
feat(ground-truth): integrate into emu_runner test loop
```

---

## Task 7: Add `--full` flag for full sweep mode

**Files:**
- Modify: `src/testing/runner_config.rs` (add `full_sweep` to Options)
- Modify: `src/testing/ground_truth.rs` (use `trace_sweep::classify_slots_from_reps` when `--full`)

**Step 1: Add CLI flag**

In `runner_config.rs`, add to `Options`:

```rust
/// Use full 13-group trace sweep for ground truth (requires multiple NPU runs).
pub full_sweep: bool,
```

And in `parse_args`, add:

```rust
"--full" => full_sweep = true,
```

**Step 2: Extend canonicalization**

When `full_sweep` is true, run multiple NPU executions per test and use
`trace_sweep::classify_slots_from_reps()` for statistical slot
classification instead of the hardcoded nondeterministic list.

This is a natural extension -- `canonicalize_tile_events` already
accepts the nondeterministic list as a parameter. In full sweep mode,
we derive that list from statistical analysis instead of using the
hardcoded constant.

**Step 3: Run tests**

Run: `cargo test --lib`
Expected: PASS (flag is additive, no behavioral change without it)

**Step 4: Commit**

```
feat(ground-truth): --full flag for statistical slot classification
```

---

## Task 8: Integration test with real binary trace data

**Files:**
- Modify: `src/testing/ground_truth.rs`

**Step 1: Write integration test**

This test uses the existing trace infrastructure to verify the full
pipeline works end-to-end with realistic data. If we have any saved
binary trace files in the build directory, use them. Otherwise, skip.

```rust
#[test]
fn test_canonicalize_real_trace_data() {
    // Try to load a real binary trace from a previous test run
    let trace_dir = std::path::Path::new("build/traces");
    if !trace_dir.exists() {
        eprintln!("  (no build/traces directory, skipping real-data test)");
        return;
    }

    // Find any emu-trace_raw.bin or hw-trace_raw.bin
    let mut found = false;
    if let Ok(entries) = std::fs::read_dir(trace_dir) {
        for entry in entries.flatten() {
            let bin = entry.path().join("emu-trace_raw.bin");
            if bin.exists() {
                let data = std::fs::read(&bin).unwrap();
                if data.iter().all(|&b| b == 0) { continue; }

                let events = crate::fuzzer::trace_sweep::decode_binary_trace(&data);
                assert!(!events.is_empty(), "decoded zero events from {}", bin.display());

                let slot_names = [
                    "INSTR_VECTOR", "INSTR_LOAD", "INSTR_STORE", "LOCK_STALL",
                    "INSTR_CALL", "PORT_RUNNING_0", "PORT_RUNNING_1", "ACTIVE",
                ];
                let tile = canonicalize_tile_events(&events, &slot_names, NONDETERMINISTIC_EVENTS);
                assert!(!tile.events.is_empty());

                eprintln!("  canonicalized {} -> {} events ({} filtered types)",
                    bin.display(), tile.events.len(), tile.filtered_events.len());
                found = true;
                break;
            }
        }
    }
    if !found {
        eprintln!("  (no binary trace files found, skipping real-data test)");
    }
}
```

**Step 2: Run it**

Run: `cargo test --lib test_canonicalize_real_trace_data -- --nocapture`
Expected: PASS (either finds data and canonicalizes, or gracefully skips)

**Step 3: Commit**

```
test(ground-truth): integration test with real binary trace data
```

---

## Execution Notes

- Tasks 1-5 are pure library code with unit tests -- no side effects,
  no NPU dependency, fast iteration.
- Task 6 is the integration task that wires everything together. It
  touches the main test loop and display formatting. Take care not
  to break existing test output format.
- Task 7 is an additive feature flag -- can be deferred if time is tight.
- Task 8 validates against real data -- run after at least one
  `--trace` run has populated `build/traces/`.

The `ground_truth.rs` module should be self-contained (~300-400 lines).
All trace decoding is delegated to `trace_sweep::decode_binary_trace()`.
All NPU execution is delegated to existing `hw_executor`/`native_hw`.
