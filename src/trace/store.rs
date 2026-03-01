//! Multi-source trace loading and navigation.
//!
//! Loads Perfetto JSON traces from hardware, emulator, and aiesimulator
//! into a unified store with a global cycle cursor.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;

/// Trace type distinguishing core module traces from memory module traces.
/// Matches the PID grouping convention used by mlir-aie's trace parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TraceType {
    Core = 0,
    Mem = 1,
}

/// Phase of a trace event (Begin or End of a duration).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Begin,
    End,
}

/// Source that produced a trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceSource {
    Hardware,
    Emulator,
    Aiesimulator,
}

/// A single trace event loaded from Perfetto JSON.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Cycle number (from Perfetto "ts" field).
    pub cycle: u64,
    /// Tile column.
    pub col: u8,
    /// Tile row.
    pub row: u8,
    /// Event name (e.g., "INSTR_VECTOR", "DMA_START_TASK").
    pub name: String,
    /// Begin or End phase.
    pub phase: Phase,
    /// Thread ID -- event lane within the tile.
    pub tid: u32,
    /// Core or Memory module trace.
    pub trace_type: TraceType,
    /// Original args from Perfetto JSON (preserved for display).
    pub args: serde_json::Value,
}

/// A single loaded trace with its source label and sorted events.
pub struct LoadedTrace {
    /// Human-readable label ("Hardware run 1", "Emulator", "aiesim").
    pub label: String,
    /// Source type for display/color coding.
    pub source: TraceSource,
    /// Events sorted by (cycle, col, row, tid).
    pub events: Vec<TraceEvent>,
    /// Cycle range: (min_cycle, max_cycle). (0, 0) if empty.
    pub cycle_range: (u64, u64),
    /// Which tiles have events.
    pub active_tiles: HashSet<(u8, u8)>,
}

impl LoadedTrace {
    /// Build a LoadedTrace from unsorted events.
    /// Sorts events and computes cycle_range and active_tiles.
    pub fn from_events(
        label: String,
        source: TraceSource,
        mut events: Vec<TraceEvent>,
    ) -> Self {
        events.sort_by(|a, b| {
            a.cycle
                .cmp(&b.cycle)
                .then(a.col.cmp(&b.col))
                .then(a.row.cmp(&b.row))
                .then(a.tid.cmp(&b.tid))
        });

        let cycle_range = if events.is_empty() {
            (0, 0)
        } else {
            (events.first().unwrap().cycle, events.last().unwrap().cycle)
        };

        let active_tiles: HashSet<(u8, u8)> =
            events.iter().map(|e| (e.col, e.row)).collect();

        Self {
            label,
            source,
            events,
            cycle_range,
            active_tiles,
        }
    }

    /// Binary search for the contiguous slice of events at a given cycle.
    /// Returns an empty slice if no events at that cycle.
    pub fn events_at_cycle(&self, cycle: u64) -> &[TraceEvent] {
        let start = self.events.partition_point(|e| e.cycle < cycle);
        let end = self.events.partition_point(|e| e.cycle <= cycle);
        &self.events[start..end]
    }

    /// Slice of events in [start_cycle, end_cycle] inclusive.
    pub fn events_in_range(
        &self,
        start_cycle: u64,
        end_cycle: u64,
    ) -> &[TraceEvent] {
        let start = self.events.partition_point(|e| e.cycle < start_cycle);
        let end = self.events.partition_point(|e| e.cycle <= end_cycle);
        &self.events[start..end]
    }

    /// Parse a Perfetto JSON trace from a reader.
    ///
    /// Extracts PID-to-tile mapping from process_name metadata, then
    /// converts B/E events into TraceEvents with tile coordinates.
    pub fn from_perfetto_json(
        reader: impl Read,
        label: String,
        source: TraceSource,
    ) -> Result<Self, String> {
        let raw: serde_json::Value = serde_json::from_reader(reader)
            .map_err(|e| format!("JSON parse error: {}", e))?;

        let array = raw
            .as_array()
            .ok_or_else(|| "Expected JSON array".to_string())?;

        // Phase 1: Build PID -> (col, row, trace_type) map from metadata.
        let mut pid_map: HashMap<u64, (u8, u8, TraceType)> = HashMap::new();
        for entry in array {
            if entry.get("ph").and_then(|v| v.as_str()) != Some("M") {
                continue;
            }
            if entry.get("name").and_then(|v| v.as_str())
                != Some("process_name")
            {
                continue;
            }
            let pid =
                entry.get("pid").and_then(|v| v.as_u64()).unwrap_or(0);
            let pname = entry
                .get("args")
                .and_then(|a| a.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            if let Some(parsed) = parse_process_name(pname) {
                pid_map.insert(pid, parsed);
            }
        }

        // Phase 2: Convert B/E events into TraceEvents.
        let mut events = Vec::new();
        for entry in array {
            let ph = match entry.get("ph").and_then(|v| v.as_str()) {
                Some("B") => Phase::Begin,
                Some("E") => Phase::End,
                _ => continue, // skip metadata and other phases
            };

            let pid =
                entry.get("pid").and_then(|v| v.as_u64()).unwrap_or(0);
            let (col, row, trace_type) = match pid_map.get(&pid) {
                Some(v) => *v,
                None => continue, // skip events with unknown PID
            };

            let name = entry
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN")
                .to_string();

            let cycle =
                entry.get("ts").and_then(|v| v.as_u64()).unwrap_or(0);
            let tid = entry.get("tid").and_then(|v| v.as_u64()).unwrap_or(0)
                as u32;
            let args = entry
                .get("args")
                .cloned()
                .unwrap_or(serde_json::Value::Null);

            events.push(TraceEvent {
                cycle,
                col,
                row,
                name,
                phase: ph,
                tid,
                trace_type,
                args,
            });
        }

        Ok(Self::from_events(label, source, events))
    }
}

/// Parse a process_name string into (col, row, TraceType).
///
/// Handles two formats:
/// - Our format: "core_trace for tile2,0" (row=2, col=0)
/// - mlir-aie:   "core_trace for column 0, row 2" (col=0, row=2)
///
/// Also handles prefixed names like "Emulator: core_trace for tile2,0".
fn parse_process_name(name: &str) -> Option<(u8, u8, TraceType)> {
    // Determine trace type from prefix.
    let trace_type = if name.contains("core_trace") {
        TraceType::Core
    } else if name.contains("mem_trace") {
        TraceType::Mem
    } else {
        return None;
    };

    // Try "tileROW,COL" format first (our format and hw traces).
    if let Some(pos) = name.find("tile") {
        let rest = &name[pos + 4..];
        let coords: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == ',')
            .collect();
        let parts: Vec<&str> = coords.split(',').collect();
        if parts.len() == 2 {
            let row: u8 = parts[0].parse().ok()?;
            let col: u8 = parts[1].parse().ok()?;
            return Some((col, row, trace_type));
        }
    }

    // Try "column COL, row ROW" format (mlir-aie).
    if let Some(col_pos) = name.find("column ") {
        let after_col = &name[col_pos + 7..];
        let col_str: String = after_col
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Some(row_pos) = name.find("row ") {
            let after_row = &name[row_pos + 4..];
            let row_str: String = after_row
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            let col: u8 = col_str.parse().ok()?;
            let row: u8 = row_str.parse().ok()?;
            return Some((col, row, trace_type));
        }
    }

    None
}

/// Multi-source trace store with global cycle cursor.
pub struct TraceStore {
    /// All loaded traces.
    pub traces: Vec<LoadedTrace>,
    /// Global cursor position in cycles.
    pub cursor: u64,
    /// Merged cycle range across all traces. (0, 0) if empty.
    pub cycle_range: (u64, u64),
}

impl Default for TraceStore {
    fn default() -> Self {
        Self {
            traces: Vec::new(),
            cursor: 0,
            cycle_range: (0, 0),
        }
    }
}

impl TraceStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Recompute merged cycle_range from all loaded traces.
    /// Only considers traces that have events (non-empty).
    fn update_cycle_range(&mut self) {
        let mut min_cycle = None;
        let mut max_cycle = None;
        for trace in &self.traces {
            if trace.events.is_empty() {
                continue;
            }
            min_cycle =
                Some(min_cycle.map_or(trace.cycle_range.0, |m: u64| {
                    m.min(trace.cycle_range.0)
                }));
            max_cycle =
                Some(max_cycle.map_or(trace.cycle_range.1, |m: u64| {
                    m.max(trace.cycle_range.1)
                }));
        }
        self.cycle_range = match (min_cycle, max_cycle) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => (0, 0),
        };
    }

    /// Add a loaded trace to the store.
    pub fn add_trace(&mut self, trace: LoadedTrace) {
        self.traces.push(trace);
        self.update_cycle_range();
    }

    /// Move cursor to a specific cycle, clamped to the merged range.
    pub fn seek(&mut self, cycle: u64) {
        if self.traces.is_empty() {
            self.cursor = 0;
            return;
        }
        self.cursor = cycle.clamp(self.cycle_range.0, self.cycle_range.1);
    }

    /// Find the next cycle (after cursor) that has any event in any trace.
    /// Returns None if cursor is at or past the last event.
    pub fn next_event_cycle(&self) -> Option<u64> {
        let mut best: Option<u64> = None;
        for trace in &self.traces {
            // Binary search for first event with cycle > cursor.
            let idx = trace.events.partition_point(|e| e.cycle <= self.cursor);
            if idx < trace.events.len() {
                let candidate = trace.events[idx].cycle;
                best = Some(best.map_or(candidate, |b: u64| b.min(candidate)));
            }
        }
        best
    }

    /// Find the previous cycle (before cursor) that has any event in any trace.
    /// Returns None if cursor is at or before the first event.
    pub fn prev_event_cycle(&self) -> Option<u64> {
        let mut best: Option<u64> = None;
        for trace in &self.traces {
            // Binary search for last event with cycle < cursor.
            let idx = trace.events.partition_point(|e| e.cycle < self.cursor);
            if idx > 0 {
                let candidate = trace.events[idx - 1].cycle;
                best = Some(best.map_or(candidate, |b: u64| b.max(candidate)));
            }
        }
        best
    }

    /// All events at the current cursor cycle, per loaded trace.
    /// Returns (trace_ref, event_slice) pairs. Slice may be empty for
    /// traces with no events at this cycle.
    pub fn events_at_cursor(&self) -> Vec<(&LoadedTrace, &[TraceEvent])> {
        self.traces
            .iter()
            .map(|t| (t, t.events_at_cycle(self.cursor)))
            .collect()
    }

    /// Events at the cursor for a specific tile, per loaded trace.
    pub fn tile_events_at_cursor(
        &self,
        col: u8,
        row: u8,
    ) -> Vec<(&LoadedTrace, Vec<&TraceEvent>)> {
        self.traces
            .iter()
            .map(|t| {
                let tile_events: Vec<&TraceEvent> = t
                    .events_at_cycle(self.cursor)
                    .iter()
                    .filter(|e| e.col == col && e.row == row)
                    .collect();
                (t, tile_events)
            })
            .collect()
    }

    /// Events in a cycle range [start, end] inclusive, per loaded trace.
    pub fn events_in_range(
        &self,
        start: u64,
        end: u64,
    ) -> Vec<(&LoadedTrace, &[TraceEvent])> {
        self.traces
            .iter()
            .map(|t| (t, t.events_in_range(start, end)))
            .collect()
    }

    /// Load a Perfetto JSON trace file into the store.
    pub fn load(
        &mut self,
        path: &Path,
        label: String,
        source: TraceSource,
    ) -> Result<(), String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
        let reader = std::io::BufReader::new(file);
        let trace =
            LoadedTrace::from_perfetto_json(reader, label, source)?;
        self.add_trace(trace);
        Ok(())
    }

    /// All tiles with events in any loaded trace.
    pub fn all_active_tiles(&self) -> HashSet<(u8, u8)> {
        let mut tiles = HashSet::new();
        for trace in &self.traces {
            tiles.extend(&trace.active_tiles);
        }
        tiles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_trace_store() {
        let store = TraceStore::new();
        assert_eq!(store.cursor, 0);
        assert_eq!(store.cycle_range, (0, 0));
        assert!(store.traces.is_empty());
        assert!(store.all_active_tiles().is_empty());
    }

    /// Helper to build a TraceEvent with minimal boilerplate.
    fn make_event(cycle: u64, col: u8, row: u8, name: &str) -> TraceEvent {
        TraceEvent {
            cycle,
            col,
            row,
            name: name.to_string(),
            phase: Phase::Begin,
            tid: 0,
            trace_type: TraceType::Core,
            args: serde_json::Value::Null,
        }
    }

    #[test]
    fn test_loaded_trace_from_events() {
        let events = vec![
            TraceEvent {
                cycle: 10,
                col: 0,
                row: 2,
                name: "INSTR_VECTOR".to_string(),
                phase: Phase::Begin,
                tid: 0,
                trace_type: TraceType::Core,
                args: serde_json::Value::Null,
            },
            TraceEvent {
                cycle: 5,
                col: 1,
                row: 2,
                name: "DMA_START_TASK".to_string(),
                phase: Phase::Begin,
                tid: 0,
                trace_type: TraceType::Mem,
                args: serde_json::Value::Null,
            },
        ];
        let trace = LoadedTrace::from_events(
            "test".to_string(),
            TraceSource::Emulator,
            events,
        );
        // Should be sorted by cycle
        assert_eq!(trace.events[0].cycle, 5);
        assert_eq!(trace.events[1].cycle, 10);
        assert_eq!(trace.cycle_range, (5, 10));
        assert!(trace.active_tiles.contains(&(0, 2)));
        assert!(trace.active_tiles.contains(&(1, 2)));
        assert_eq!(trace.active_tiles.len(), 2);
    }

    #[test]
    fn test_seek_clamps_to_range() {
        let mut store = TraceStore::new();
        let events = vec![
            make_event(10, 0, 2, "A"),
            make_event(20, 0, 2, "B"),
            make_event(30, 0, 2, "C"),
        ];
        store.add_trace(LoadedTrace::from_events(
            "t1".into(),
            TraceSource::Emulator,
            events,
        ));

        store.seek(15);
        assert_eq!(store.cursor, 15);

        store.seek(100);
        assert_eq!(store.cursor, 30); // clamp to max

        store.seek(0);
        assert_eq!(store.cursor, 10); // clamp to min
    }

    #[test]
    fn test_next_event_cycle() {
        let mut store = TraceStore::new();
        let events = vec![
            make_event(10, 0, 2, "A"),
            make_event(20, 0, 2, "B"),
            make_event(30, 0, 2, "C"),
        ];
        store.add_trace(LoadedTrace::from_events(
            "t1".into(),
            TraceSource::Emulator,
            events,
        ));

        store.seek(10);
        assert_eq!(store.next_event_cycle(), Some(20));

        store.seek(20);
        assert_eq!(store.next_event_cycle(), Some(30));

        store.seek(30);
        assert_eq!(store.next_event_cycle(), None); // at end
    }

    #[test]
    fn test_prev_event_cycle() {
        let mut store = TraceStore::new();
        let events = vec![
            make_event(10, 0, 2, "A"),
            make_event(20, 0, 2, "B"),
            make_event(30, 0, 2, "C"),
        ];
        store.add_trace(LoadedTrace::from_events(
            "t1".into(),
            TraceSource::Emulator,
            events,
        ));

        store.seek(30);
        assert_eq!(store.prev_event_cycle(), Some(20));

        store.seek(20);
        assert_eq!(store.prev_event_cycle(), Some(10));

        store.seek(10);
        assert_eq!(store.prev_event_cycle(), None); // at start
    }

    #[test]
    fn test_next_event_merges_traces() {
        let mut store = TraceStore::new();
        store.add_trace(LoadedTrace::from_events(
            "emu".into(),
            TraceSource::Emulator,
            vec![make_event(10, 0, 2, "A"), make_event(30, 0, 2, "C")],
        ));
        store.add_trace(LoadedTrace::from_events(
            "hw".into(),
            TraceSource::Hardware,
            vec![make_event(20, 0, 2, "B")],
        ));

        store.seek(10);
        // Next event is at cycle 20 (from hw trace), not 30
        assert_eq!(store.next_event_cycle(), Some(20));

        store.seek(20);
        assert_eq!(store.next_event_cycle(), Some(30));
    }

    #[test]
    fn test_events_at_cursor() {
        let mut store = TraceStore::new();
        store.add_trace(LoadedTrace::from_events(
            "t1".into(),
            TraceSource::Emulator,
            vec![
                make_event(10, 0, 2, "A"),
                make_event(10, 1, 2, "B"),
                make_event(20, 0, 2, "C"),
            ],
        ));

        store.seek(10);
        let results = store.events_at_cursor();
        assert_eq!(results.len(), 1); // one trace
        assert_eq!(results[0].1.len(), 2); // two events at cycle 10
    }

    #[test]
    fn test_events_at_cursor_multi_trace() {
        let mut store = TraceStore::new();
        store.add_trace(LoadedTrace::from_events(
            "emu".into(),
            TraceSource::Emulator,
            vec![make_event(10, 0, 2, "A")],
        ));
        store.add_trace(LoadedTrace::from_events(
            "hw".into(),
            TraceSource::Hardware,
            vec![make_event(10, 0, 2, "X"), make_event(10, 1, 2, "Y")],
        ));

        store.seek(10);
        let results = store.events_at_cursor();
        assert_eq!(results.len(), 2); // two traces
        assert_eq!(results[0].1.len(), 1); // emu: 1 event
        assert_eq!(results[1].1.len(), 2); // hw: 2 events
    }

    #[test]
    fn test_tile_events_at_cursor() {
        let mut store = TraceStore::new();
        store.add_trace(LoadedTrace::from_events(
            "t1".into(),
            TraceSource::Emulator,
            vec![
                make_event(10, 0, 2, "A"),
                make_event(10, 1, 2, "B"),
                make_event(10, 0, 2, "C"),
            ],
        ));

        store.seek(10);
        let results = store.tile_events_at_cursor(0, 2);
        assert_eq!(results.len(), 1); // one trace
        assert_eq!(results[0].1.len(), 2); // events A and C for tile (0,2)
    }

    #[test]
    fn test_events_in_range() {
        let mut store = TraceStore::new();
        store.add_trace(LoadedTrace::from_events(
            "t1".into(),
            TraceSource::Emulator,
            vec![
                make_event(5, 0, 2, "A"),
                make_event(10, 0, 2, "B"),
                make_event(15, 0, 2, "C"),
                make_event(20, 0, 2, "D"),
                make_event(25, 0, 2, "E"),
            ],
        ));

        let results = store.events_in_range(10, 20);
        assert_eq!(results.len(), 1);
        // Should include events at cycles 10, 15, 20
        assert_eq!(results[0].1.len(), 3);
        assert_eq!(results[0].1[0].name, "B");
        assert_eq!(results[0].1[2].name, "D");
    }

    #[test]
    fn test_events_at_cursor_empty_cycle() {
        let mut store = TraceStore::new();
        store.add_trace(LoadedTrace::from_events(
            "t1".into(),
            TraceSource::Emulator,
            vec![make_event(10, 0, 2, "A"), make_event(20, 0, 2, "B")],
        ));

        store.seek(15); // no events at cycle 15
        let results = store.events_at_cursor();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.len(), 0); // empty slice
    }

    // -- Task 4: Perfetto JSON loader tests --

    #[test]
    fn test_load_perfetto_json_basic() {
        let json = r#"[
{"name":"process_name","ph":"M","pid":0,"args":{"name":"core_trace for tile2,0"}},
{"name":"thread_name","ph":"M","pid":0,"tid":0,"args":{"name":"INSTR_VECTOR"}},
{"name":"INSTR_VECTOR","ph":"B","pid":0,"tid":0,"ts":10,"args":{}},
{"name":"INSTR_VECTOR","ph":"E","pid":0,"tid":0,"ts":11,"args":{}}
]"#;

        let trace = LoadedTrace::from_perfetto_json(
            json.as_bytes(),
            "test".to_string(),
            TraceSource::Emulator,
        )
        .unwrap();

        assert_eq!(trace.events.len(), 2);
        assert_eq!(trace.events[0].cycle, 10);
        assert_eq!(trace.events[0].col, 0);
        assert_eq!(trace.events[0].row, 2);
        assert_eq!(trace.events[0].name, "INSTR_VECTOR");
        assert_eq!(trace.events[0].phase, Phase::Begin);
        assert_eq!(trace.events[0].trace_type, TraceType::Core);
        assert_eq!(trace.events[1].phase, Phase::End);
        assert!(trace.active_tiles.contains(&(0, 2)));
    }

    #[test]
    fn test_load_perfetto_json_multi_tile() {
        let json = r#"[
{"name":"process_name","ph":"M","pid":0,"args":{"name":"core_trace for tile2,0"}},
{"name":"process_name","ph":"M","pid":1,"args":{"name":"mem_trace for tile2,0"}},
{"name":"process_name","ph":"M","pid":2,"args":{"name":"core_trace for tile2,1"}},
{"name":"INSTR_VECTOR","ph":"B","pid":0,"tid":0,"ts":10,"args":{}},
{"name":"DMA_START_TASK","ph":"B","pid":1,"tid":0,"ts":15,"args":{}},
{"name":"INSTR_LOAD","ph":"B","pid":2,"tid":1,"ts":10,"args":{}}
]"#;

        let trace = LoadedTrace::from_perfetto_json(
            json.as_bytes(),
            "test".to_string(),
            TraceSource::Hardware,
        )
        .unwrap();

        assert_eq!(trace.events.len(), 3);
        // Sorted by cycle, then col
        assert_eq!(trace.events[0].cycle, 10);
        assert_eq!(trace.events[2].cycle, 15);
        assert!(trace.active_tiles.contains(&(0, 2)));
        assert!(trace.active_tiles.contains(&(1, 2)));
        // Check trace types resolved from PID metadata
        let mem_event = trace
            .events
            .iter()
            .find(|e| e.name == "DMA_START_TASK")
            .unwrap();
        assert_eq!(mem_event.trace_type, TraceType::Mem);
    }

    #[test]
    fn test_load_perfetto_json_skips_metadata() {
        let json = r#"[
{"name":"process_name","ph":"M","pid":0,"args":{"name":"core_trace for tile2,0"}},
{"name":"thread_name","ph":"M","pid":0,"tid":0,"args":{"name":"INSTR_VECTOR"}},
{"name":"INSTR_VECTOR","ph":"B","pid":0,"tid":0,"ts":5,"args":{}}
]"#;

        let trace = LoadedTrace::from_perfetto_json(
            json.as_bytes(),
            "test".into(),
            TraceSource::Emulator,
        )
        .unwrap();

        // Only the B event, not the M metadata events
        assert_eq!(trace.events.len(), 1);
    }

    #[test]
    fn test_load_from_file() {
        // Use a real trace file from the build directory if available.
        let path =
            std::path::Path::new("build/traces/add_one_using_dma/emu-trace.json");
        if !path.exists() {
            // Skip test if no trace files built.
            return;
        }
        let mut store = TraceStore::new();
        store
            .load(path, "emu".into(), TraceSource::Emulator)
            .unwrap();
        assert_eq!(store.traces.len(), 1);
        assert!(!store.traces[0].events.is_empty());
        assert!(store.cycle_range.1 > 0);
    }

    #[test]
    fn test_parse_process_name_tile_format() {
        let result = parse_process_name("core_trace for tile2,0");
        assert_eq!(result, Some((0, 2, TraceType::Core)));

        let result = parse_process_name("mem_trace for tile1,3");
        assert_eq!(result, Some((3, 1, TraceType::Mem)));
    }

    #[test]
    fn test_parse_process_name_column_row_format() {
        let result =
            parse_process_name("core_trace for column 0, row 2");
        assert_eq!(result, Some((0, 2, TraceType::Core)));
    }

    #[test]
    fn test_parse_process_name_unknown() {
        assert_eq!(parse_process_name("something else"), None);
    }

    #[test]
    fn test_empty_trace_does_not_corrupt_range() {
        let mut store = TraceStore::new();
        // Add a real trace first.
        store.add_trace(LoadedTrace::from_events(
            "real".into(),
            TraceSource::Emulator,
            vec![make_event(100, 0, 2, "A"), make_event(200, 0, 2, "B")],
        ));
        assert_eq!(store.cycle_range, (100, 200));

        // Add an empty trace -- should NOT drag min to 0.
        store.add_trace(LoadedTrace::from_events(
            "empty".into(),
            TraceSource::Hardware,
            vec![],
        ));
        assert_eq!(store.cycle_range, (100, 200));
    }

    #[test]
    fn test_default_trait() {
        let store = TraceStore::default();
        assert_eq!(store.cursor, 0);
        assert_eq!(store.cycle_range, (0, 0));
        assert!(store.traces.is_empty());
    }
}
