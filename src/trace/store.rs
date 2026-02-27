//! Multi-source trace loading and navigation.
//!
//! Loads Perfetto JSON traces from hardware, emulator, and aiesimulator
//! into a unified store with a global cycle cursor.

use std::collections::HashSet;
use std::path::Path;

/// Re-use the existing TraceType from the export module.
pub use super::TraceType;

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

impl TraceStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            traces: Vec::new(),
            cursor: 0,
            cycle_range: (0, 0),
        }
    }

    /// Recompute merged cycle_range from all loaded traces.
    fn update_cycle_range(&mut self) {
        if self.traces.is_empty() {
            self.cycle_range = (0, 0);
            return;
        }
        let min = self
            .traces
            .iter()
            .map(|t| t.cycle_range.0)
            .min()
            .unwrap();
        let max = self
            .traces
            .iter()
            .map(|t| t.cycle_range.1)
            .max()
            .unwrap();
        self.cycle_range = (min, max);
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
}
