# Trace Visualizer Design

## Goal

Replace the atrophied emulator GUI with a trace comparison visualizer that
renders HW vs EMU event timelines side by side, highlights divergences, and
lets a human see exactly where and how the emulator deviates from real
silicon.

## Architecture

Clean-slate the `src/visual/` module. Keep egui 0.31 framework and the
module boundary. The new app's primary purpose is trace analysis. Emulator
run controls return later when live mode is wired up.

Data flows from the existing `trace::compare` library (the same code that
powers the `trace-compare` CLI) into a `TraceSource` trait that the
rendering layer consumes. No new file formats -- we load the raw binary
traces that the bridge test already generates and run comparison in-process.

## Tech Stack

- **egui 0.31** via eframe (pure Rust, immediate mode, cross-platform)
- **trace::compare** library (BatchResult, TileResult, EdgeResult, LevelResult)
- Raw binary trace format (32-bit packet headers, 28-byte payloads)
- EventsConfig JSON (slot index to event name mapping)

---

## Module Structure

### Deleted (old GUI, ~1,085 lines)

- `src/visual/app.rs` -- emulator-centric app
- `src/visual/tile_grid.rs` -- tile array grid
- `src/visual/tile_detail.rs` -- tile detail panels
- `src/visual/controls.rs` -- emulation run controls
- `src/visual/memory_view.rs` -- hex memory viewer
- `src/visual/mod.rs` -- old module root

### Created

```
src/visual/
    mod.rs              Module root, re-exports TraceViewerApp
    app.rs              eframe::App impl, layout, file loading, state
    timeline.rs         Core widget: event lanes, zoom/pan, divergence overlay
    tile_selector.rs    Sidebar: tile list with divergence severity indicators
    event_detail.rs     Bottom panel: selected event/region details
    theme.rs            Colors, spacing, style constants
```

### Untouched

- `src/trace/compare.rs` -- comparison engine (called as library)
- `src/trace/store.rs` -- Perfetto JSON loading (future use)
- `src/main.rs` -- CLI entry point (`--gui` launches the new app)
- Cargo.toml egui/eframe deps

---

## Data Abstraction

```rust
/// Trait abstracting the source of trace comparison data.
/// File-loaded now; live emulator stream later.
pub trait TraceSource {
    fn tiles(&self) -> &[(TileKey, TileResult)];
    fn batch_config(&self) -> &EventsConfig;
    fn hw_events(&self, tile: &TileKey) -> &[TileEvent];
    fn emu_events(&self, tile: &TileKey) -> &[TileEvent];
    fn anchor(&self, tile: &TileKey) -> (u64, u64); // (hw_t0, emu_t0)
}
```

### v1 Implementation

```rust
pub struct LoadedComparison {
    pub batch: BatchResult,
    pub hw_events: HashMap<TileKey, Vec<TileEvent>>,
    pub emu_events: HashMap<TileKey, Vec<TileEvent>>,
    pub alignment: AlignmentMap,
}

impl TraceSource for LoadedComparison { /* direct field access */ }
```

### Future: Live Mode

```rust
pub struct LiveComparison {
    pub reference: LoadedComparison,                     // pre-recorded HW trace
    pub live_events: HashMap<TileKey, Vec<TileEvent>>,   // growing as emulator runs
    // comparison re-computed incrementally
}

impl TraceSource for LiveComparison { /* same trait, different backing */ }
```

The timeline widget renders from `&dyn TraceSource` and doesn't know which
implementation it's using.

---

## Piecewise Alignment Model

The relationship between HW and EMU cycle numbers is not a simple global
offset. Different execution phases (DMA setup, compute, lock handshake) can
have systematically different timing ratios. The alignment model handles
this with anchor pairs.

```rust
pub struct AlignmentMap {
    /// Ordered anchor pairs: (hw_cycle, emu_cycle).
    /// Between anchors, cycles are linearly interpolated.
    /// Before first anchor: constant offset from first pair.
    /// After last anchor: constant offset from last pair.
    anchors: Vec<(u64, u64)>,
}

impl AlignmentMap {
    /// Map a HW cycle to unified timeline coordinates.
    fn map_hw_to_unified(&self, hw_cycle: u64) -> f64;

    /// Map an EMU cycle to unified timeline coordinates.
    fn map_emu_to_unified(&self, emu_cycle: u64) -> f64;

    /// Add a user-defined anchor pair.
    fn add_anchor(&mut self, hw_cycle: u64, emu_cycle: u64);
}
```

**v1**: Auto-populated with the single anchor from `trace::compare` (first
shared edge event per tile). One anchor = constant offset.

**Future**: Interactive anchoring. Click an HW event, click the
corresponding EMU event, press "Anchor". The timeline re-renders with
piecewise alignment. Events between anchors stretch/compress to stay
synchronized. Divergence display becomes alignment-aware -- deltas are
recomputed relative to the local alignment, revealing phase-dependent
timing differences vs real divergences.

---

## App Layout

### v1 Primary Layout

```
+------------------------------------------------------------------+
| Menu: File | View                              [Batch 0/3 v]     |
+------------------------------------------------------------------+
| Tile List     |  Timeline                                        |
| (sidebar)     |  [--- minimap / viewport indicator ---]          |
|               |  +- HW ----------------------------------------+ |
| > (0,2) Core  |  | INSTR_VECTOR  ||||  |||| ||||  ||||         | |
|   2 diverge   |  | DMA_START     | |  | |  | |                 | |
|   (0,2) Mem   |  +- EMU ----------------------------------------+ |
|   0 diverge   |  | INSTR_VECTOR  ||||  ||||  |||| ||||         | |
|   (0,1) MTile  |  | DMA_START     | |  | |   | |               | |
|   1 diverge   |  +--------------------------------------------------+
+---------------+---------------------------------------------------+
| Detail: INSTR_VECTOR #42  HW cy=5678  EMU cy=5670  dt=+8        |
+------------------------------------------------------------------+
```

- **Sidebar**: Scrollable tile list sorted by divergence count (worst first).
  Click to select tile, which updates the timeline.
- **Timeline**: Horizontal event lanes. HW block on top (blue tint), EMU
  block on bottom (green tint). Shared x-axis in cycle numbers.
  Zoom with scroll wheel, pan with click+drag.
- **Minimap**: Thin bar above the timeline showing the full trace extent
  with current viewport highlighted.
- **Detail panel**: Info for hovered/clicked event -- cycle, delta, event
  name, paired HW/EMU values.
- **Batch selector**: Dropdown when a sweep directory is loaded (multiple
  event configurations).

### Planned View Modes (architecture now, build later)

- **Dual Grid**: Two 5x6 NPU grids side by side (HW left, EMU right).
  Tiles colored by activity at the current timeline cursor. Clicking a
  tile selects it in the timeline.
- **Single Grid + Toggle**: One 5x6 grid with HW/EMU toggle button.
- **Split Timeline**: Two independent timeline panels that can scroll
  independently or be locked. This is where interactive piecewise
  alignment lives -- drag anchor points between the two panels.

---

## Timeline Rendering

### Coordinate System

- X-axis: cycle number, rebased from shared anchor (0 = first matched event)
- Y-axis: event lanes grouped into HW block and EMU block
- Each event type gets one lane

### Event Rendering

- **Edge events** (point-in-time): vertical tick marks at the cycle
- **Level events** (intervals): filled rectangles from begin to end cycle
- At high zoom-out, individual ticks collapse to a density heatmap
  (color intensity = event frequency per pixel)

### Divergence Visualization

- Matching event pairs connected by faint vertical lines between HW and
  EMU lanes
- When |delta| > threshold (default 10 cycles): connecting line turns red,
  thicker
- Divergent regions get a subtle red background wash
- First divergence point per event type gets a marker (diamond)

### Colors

- HW lane backgrounds: blue-tinted
- EMU lane backgrounds: green-tinted
- Edge ticks: white/bright
- Level bars: saturated lane color
- Divergent pairs: red connecting lines
- Selected event: yellow highlight

### Performance

- Only render events in the current viewport (binary search on sorted
  cycle arrays)
- Density heatmap mode at high zoom-out
- Comparison computed once on load, stored in memory

---

## Loading Sequence

1. User selects File > Open Trace Pair (or File > Open Sweep Directory)
2. For a trace pair: select HW dir and EMU dir (each containing
   `trace_raw.bin` + `events.json`)
3. For a sweep: select the sweep root directory, app discovers all batches
4. App calls `trace::compare::compare_batch()` in-process
5. Result: `BatchResult` + retained raw `TileEvent` vectors
6. Wrapped in `LoadedComparison` implementing `TraceSource`
7. Sidebar populates with tiles sorted by divergence
8. Timeline renders the first (worst) tile

---

## Scope Boundaries

### v1 (what we build)

- File > Open (trace pair or sweep directory)
- Dual-lane timeline per tile (HW top, EMU bottom)
- Edge ticks + level bars
- Divergence highlighting (red lines, wash, markers)
- Zoom/pan/minimap
- Click-to-inspect detail panel
- Tile sidebar sorted by divergence severity
- Batch selector dropdown for sweep directories
- Single auto-anchor alignment (from trace::compare)

### Explicitly NOT v1

- Emulator run controls (run/pause/step/reset)
- Live comparison mode
- Interactive piecewise alignment UI
- Dual grid / single grid views
- Breakpoints, watchpoints, memory inspection
- Perfetto JSON loading (raw binary traces only)
- Export/save functionality
- Density heatmap at high zoom-out (render as clipped ticks initially)

---

## Future Roadmap

1. **Interactive alignment**: Click-to-anchor UI for piecewise alignment
2. **Dual grid view**: 5x6 NPU grids showing tile state at cursor position
3. **Live mode**: Wire emulator trace output into `LiveComparison`,
   overlay against loaded HW reference in real-time
4. **Stall attribution overlay**: Draw arrows from stall intervals to
   their resolving events (cross-tile)
5. **Emulator controls**: Bring back run/pause/step, wired to live mode
