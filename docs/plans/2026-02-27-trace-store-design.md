# Trace Store: Multi-Source Trace Loading Framework

## Context

The emulator has three trace data sources, all converging on Perfetto JSON:

- **Emulator**: `src/trace/mod.rs` exports `export_perfetto()` from live events
- **Hardware**: mlir-aie's `parse.py` converts raw NPU trace buffers to JSON
- **aiesimulator**: `src/trace/vcd.rs` converts VCD files to JSON

All three produce Chrome Trace Event Format JSON with matching event names
(INSTR_VECTOR, DMA_START_TASK, etc.) and compatible PID/TID conventions.

The GUI debugger (`src/visual/`) has live emulator state views (tile grid,
registers, memory, controls) but zero trace visualization. Viewing traces
requires opening Perfetto UI externally -- completely disconnected from the
debugger.

### Goal

Build the framework layer that enables:

1. **Post-mortem replay**: Load a trace file, scrub through cycles, see tile
   states at each point in time.
2. **Side-by-side comparison**: Load emulator + hardware traces, step through
   in sync, spot divergences.
3. **Live trace overlay**: During emulator stepping, show what hardware did at
   the same cycle from a pre-loaded trace ("ghost" mode).
4. **Trace merging**: Combine multiple hardware traces from the same
   deterministic kernel (NPU has limited trace slots per run, so multiple
   runs with different trace configurations can be merged).

This design covers the **framework** (data model, loading, cursor, queries)
and minimal GUI integration (menu item, cursor bar). Timeline rendering
(swimlane view) is a separate future feature.

## Design Decisions

### Data format: Native Perfetto JSON

Load Perfetto JSON directly as the canonical format. All three sources already
produce it. No conversion layer needed. Perfetto's schema is well-defined.

If lookup performance becomes a bottleneck for very large traces, per-tile
BTreeMap indexes can be layered on without changing the data model -- the
sorted event vectors support both binary search (current) and index
construction (future).

### Module home: src/trace/

`src/trace/` already owns trace export. Adding trace loading/model alongside
it makes `src/trace/` the trace data hub: export AND import. The GUI imports
from `src/trace/`. Clean separation: `trace/` owns the data, `visual/` owns
the rendering.

### Time axis: Cycle-based with event skip

Default to cycle stepping. Provide next-event/prev-event navigation for fast
scrubbing. All three trace sources use cycles as the timestamp unit (emulator
and mlir-aie natively, VCD converted via clock period).

**Global cursor, no desync.** The cursor position is a single cycle number.
When jumping to the next event on any tile, the entire trace advances. Tiles
never desync.

### Multi-trace from day one

The store holds `Vec<LoadedTrace>`. Each trace has a source label, and all
queries return results tagged by trace. This supports:

- Loading emulator + hardware traces simultaneously
- Merging multiple hardware runs of the same kernel
- Adding aiesimulator traces as a third comparison point

## Data Model

```rust
/// A single trace event loaded from Perfetto JSON.
pub struct TraceEvent {
    /// Cycle number (from Perfetto "ts" field).
    pub cycle: u64,
    /// Tile column.
    pub col: u8,
    /// Tile row.
    pub row: u8,
    /// Event name (e.g., "INSTR_VECTOR", "DMA_START_TASK").
    pub name: String,
    /// Phase: Begin or End (from Perfetto "ph" field).
    pub phase: Phase,
    /// Thread ID -- event lane within the tile (from Perfetto "tid").
    pub tid: u32,
    /// Trace type: Core or Mem (derived from Perfetto PID metadata).
    pub trace_type: TraceType,
    /// Original args from Perfetto JSON (preserved for display).
    pub args: serde_json::Value,
}

pub enum Phase {
    Begin,
    End,
}

/// A single loaded trace with its source label and events.
pub struct LoadedTrace {
    /// Human-readable label ("Hardware run 1", "Emulator", "aiesim").
    pub label: String,
    /// Source type for display/color coding.
    pub source: TraceSource,
    /// Events sorted by cycle (primary), then col, row, tid.
    pub events: Vec<TraceEvent>,
    /// Cycle range: (min_cycle, max_cycle).
    pub cycle_range: (u64, u64),
    /// Which tiles have events (for grid highlighting).
    pub active_tiles: HashSet<(u8, u8)>,
}

pub enum TraceSource {
    Hardware,
    Emulator,
    Aiesimulator,
}

/// The trace store: holds all loaded traces and the global cursor.
pub struct TraceStore {
    /// All loaded traces (multi-trace from day one).
    pub traces: Vec<LoadedTrace>,
    /// Global cursor position in cycles.
    pub cursor: u64,
    /// Merged cycle range across all loaded traces.
    pub cycle_range: (u64, u64),
}
```

### Design invariants

- `LoadedTrace.events` is always sorted by cycle. This enables binary search
  now and per-tile indexing later.
- `TraceEvent` is flat and self-contained (no references, no lifetime
  parameters). Cheap to sort, search, and filter.
- `TraceStore` owns all traces with a single global cursor. All traces share
  the same cycle space.
- `active_tiles` is computed on load for fast "which tiles have data" queries
  without scanning events.

## Query API

```rust
impl TraceStore {
    /// Load a Perfetto JSON trace file.
    fn load(&mut self, path: &Path, label: String, source: TraceSource) -> Result<()>;

    /// Move cursor to a specific cycle.
    fn seek(&mut self, cycle: u64);

    /// All events at the current cursor cycle, across all loaded traces.
    fn events_at_cursor(&self) -> Vec<(&LoadedTrace, &[TraceEvent])>;

    /// Events at cursor for a specific tile, across all traces.
    fn tile_events_at_cursor(&self, col: u8, row: u8)
        -> Vec<(&LoadedTrace, Vec<&TraceEvent>)>;

    /// Jump to next cycle that has any event (globally). Returns None at end.
    fn next_event_cycle(&self) -> Option<u64>;

    /// Jump to previous cycle with any event. Returns None at start.
    fn prev_event_cycle(&self) -> Option<u64>;

    /// Events in a cycle range (for rendering a visible timeline window).
    fn events_in_range(&self, start: u64, end: u64)
        -> Vec<(&LoadedTrace, &[TraceEvent])>;

    /// All tiles that have events in any loaded trace.
    fn all_active_tiles(&self) -> HashSet<(u8, u8)>;
}
```

All queries use binary search on the sorted `events` vec. The `&[TraceEvent]`
slice returns are possible because events are contiguous by cycle in the
sorted vector -- binary search finds the start index, scan forward to find
the end. Zero allocation for the common case.

`next_event_cycle` / `prev_event_cycle` merge across all loaded traces --
find the next cycle with events in *any* trace, maintaining the global cursor
invariant.

### Future indexing (provisions for Approach B)

If trace sizes grow to millions of events and binary search latency becomes
noticeable during interactive scrubbing, add:

```rust
/// Per-tile cycle index, built lazily on first query.
tile_index: HashMap<(u8, u8), BTreeMap<u64, Range<usize>>>
```

This maps (col, row) -> (cycle -> event slice range in the sorted vec).
The sorted-vec invariant makes this a pure acceleration structure -- the
underlying data model does not change.

## Perfetto JSON Loading

The loader:

1. Parses the JSON array via serde_json.
2. Scans for `process_name` metadata events (`"ph":"M"`) to build a PID-to-
   tile mapping: `HashMap<u32, (u8, u8, TraceType)>`.
3. Handles both naming conventions:
   - Our format: `"core_trace for tile2,0"` (row, col)
   - mlir-aie format: `"core_trace for column 0, row 2"` (col, row)
4. Converts each `"B"` / `"E"` event into a `TraceEvent` with tile
   coordinates resolved from the PID map.
5. Sorts events by `(cycle, col, row, tid)`.
6. Computes `cycle_range` and `active_tiles`.

Metadata-only events (`"M"` phase) are consumed for PID mapping but not
stored in the event vector.

### Source auto-detection

If not specified by the caller, `TraceSource` can be inferred from
process_name prefixes:
- Contains "Emulator:" -> `TraceSource::Emulator`
- Contains "aiesimulator:" -> `TraceSource::Aiesimulator`
- Otherwise -> `TraceSource::Hardware`

## GUI Integration

### EmulatorApp changes

```rust
pub struct EmulatorApp {
    // ... existing fields ...
    /// Loaded trace data (None = no traces loaded, live emulator mode).
    pub trace_store: Option<TraceStore>,
}
```

### Minimal GUI additions

1. **File > Load Trace** menu item: opens file picker for `.json` files,
   calls `TraceStore::load()`, sets status message.
2. **Trace cursor bar**: when `trace_store.is_some()`, show cycle number
   with Prev/Next Event buttons and a cycle input field. Appears below the
   existing control bar.
3. **Tile grid overlay**: tiles with trace data get a subtle border or
   indicator via `all_active_tiles()`. Existing tile status coloring is
   unaffected.

The actual timeline rendering (swimlane view with event bars per tile) is a
separate future feature. This design delivers the data model, loading, cursor
navigation, and enough GUI to prove the framework works.

## Files

| File | Changes |
|------|---------|
| `src/trace/store.rs` | **NEW** -- TraceStore, LoadedTrace, TraceEvent, loading, cursor, queries |
| `src/trace/mod.rs` | Add `pub mod store;`, re-export key types |
| `src/visual/app.rs` | Add `trace_store: Option<TraceStore>`, menu item, cursor bar |
| `src/visual/controls.rs` | Add trace step/skip buttons when trace is loaded |

No changes to existing trace export code, emulator logic, or test runner.

## Verification

1. `cargo test --lib -- trace::store` -- unit tests for loading, sorting,
   cursor navigation, multi-trace queries
2. Load a real emulator trace JSON and verify `events_at_cursor` returns
   correct data
3. Load two traces (emulator + hardware) and verify `next_event_cycle`
   merges correctly
4. GUI builds and shows "Load Trace" menu item with functional cursor bar
