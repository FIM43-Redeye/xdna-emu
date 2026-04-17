# Trace Visualizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the atrophied emulator GUI with a trace comparison visualizer
that renders HW vs EMU event timelines side by side, highlights divergences,
and lets a human see exactly where the emulator deviates from real silicon.

**Architecture:** Clean-slate `src/visual/` while keeping the egui 0.31
framework. Data flows from the existing `trace::compare` library into a
`TraceSource` trait. Timeline widget renders dual-lane event lanes (HW top,
EMU bottom) with zoom/pan/minimap. Piecewise `AlignmentMap` handles
HW-to-EMU cycle mapping. Tile sidebar sorts by divergence severity.

**Tech Stack:** Rust, egui 0.31 (eframe), trace::compare library, rfd (native
file dialogs), raw binary trace format (32-bit packet headers).

**Design Doc:** `docs/plans/2026-03-08-trace-visualizer-design.md`

---

## Phase 1: Teardown and Foundation

### Task 1: Delete old GUI module

Remove all old visual module files. The emulator core is completely decoupled
from the GUI -- these files have no dependents outside `src/visual/mod.rs`
and `src/main.rs`.

**Files:**
- Delete: `src/visual/app.rs`
- Delete: `src/visual/tile_grid.rs`
- Delete: `src/visual/tile_detail.rs`
- Delete: `src/visual/controls.rs`
- Delete: `src/visual/memory_view.rs`
- Modify: `src/visual/mod.rs` (gut contents)
- Modify: `src/main.rs` (stub out `run_gui`)

**Step 1: Delete the five old submodule files**

```bash
rm src/visual/app.rs src/visual/tile_grid.rs src/visual/tile_detail.rs \
   src/visual/controls.rs src/visual/memory_view.rs
```

**Step 2: Replace mod.rs with a stub**

```rust
//! Trace comparison visualizer for AMD XDNA NPU emulation.
//!
//! Replaces the old emulator GUI with a trace-focused tool that renders
//! HW vs EMU event timelines side by side, highlights divergences, and
//! supports piecewise alignment for phase-aware comparison.

// Modules will be added in subsequent tasks.
```

**Step 3: Stub out run_gui in main.rs**

Replace the `run_gui` function and remove all `visual::EmulatorApp` usage.
The function should print a message and return Ok for now.

```rust
fn run_gui(_file_path: Option<&str>) -> anyhow::Result<()> {
    println!("Trace visualizer is under construction.");
    println!("Use --help for CLI options.");
    Ok(())
}
```

Remove the `use xdna_emu::visual::EmulatorApp;` import.

**Step 4: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: successful build (warnings OK, no errors)

**Step 5: Run tests to verify nothing broke**

Run: `cargo test --lib 2>&1 | tail -10`
Expected: all existing tests pass (visual module had zero tests)

**Step 6: Commit**

```bash
git add -u src/visual/ src/main.rs
git commit -m "refactor(visual): delete old GUI, prepare for trace visualizer"
```

---

### Task 2: Add rfd dependency and create theme module

**Files:**
- Modify: `Cargo.toml` (add `rfd` dependency)
- Create: `src/visual/theme.rs`
- Modify: `src/visual/mod.rs` (add `mod theme`)

**Step 1: Add rfd to Cargo.toml**

Add after the `egui_extras` line in `[dependencies]`:

```toml
rfd = "0.15"                                            # Native file dialogs
```

**Step 2: Create theme.rs**

```rust
//! Color and style constants for the trace visualizer.
//!
//! Centralizes all visual constants so the rendering code stays clean
//! and themes can be adjusted in one place.

use eframe::egui;

/// Background tint for hardware trace lanes.
pub const HW_LANE_BG: egui::Color32 = egui::Color32::from_rgba_premultiplied(30, 50, 80, 255);

/// Background tint for emulator trace lanes.
pub const EMU_LANE_BG: egui::Color32 = egui::Color32::from_rgba_premultiplied(30, 70, 50, 255);

/// Edge event tick mark color.
pub const EDGE_TICK: egui::Color32 = egui::Color32::from_rgb(220, 220, 230);

/// Level event bar color (HW).
pub const HW_LEVEL_BAR: egui::Color32 = egui::Color32::from_rgb(80, 130, 200);

/// Level event bar color (EMU).
pub const EMU_LEVEL_BAR: egui::Color32 = egui::Color32::from_rgb(80, 180, 120);

/// Divergent pair connecting line.
pub const DIVERGE_LINE: egui::Color32 = egui::Color32::from_rgb(220, 60, 60);

/// Matched pair connecting line (within threshold).
pub const MATCH_LINE: egui::Color32 = egui::Color32::from_rgba_premultiplied(100, 100, 100, 80);

/// Selected event highlight.
pub const SELECTED_HIGHLIGHT: egui::Color32 = egui::Color32::from_rgb(240, 200, 40);

/// Divergence background wash.
pub const DIVERGE_WASH: egui::Color32 = egui::Color32::from_rgba_premultiplied(180, 40, 40, 30);

/// First-divergence marker color.
pub const DIVERGE_MARKER: egui::Color32 = egui::Color32::from_rgb(255, 80, 80);

/// Minimap viewport indicator color.
pub const MINIMAP_VIEWPORT: egui::Color32 = egui::Color32::from_rgba_premultiplied(255, 255, 255, 60);

/// Minimap background.
pub const MINIMAP_BG: egui::Color32 = egui::Color32::from_rgb(30, 30, 35);

/// Height of each event lane in pixels.
pub const LANE_HEIGHT: f32 = 20.0;

/// Gap between HW and EMU lane blocks.
pub const BLOCK_GAP: f32 = 8.0;

/// Height of the minimap bar.
pub const MINIMAP_HEIGHT: f32 = 24.0;

/// Minimum pixels per cycle before ticks become too dense.
pub const MIN_PX_PER_TICK: f32 = 2.0;

/// Divergence threshold in cycles (matches trace::compare::DIVERGE_THRESHOLD).
pub const DIVERGE_THRESHOLD: i64 = 10;
```

**Step 3: Update mod.rs**

```rust
//! Trace comparison visualizer for AMD XDNA NPU emulation.
//!
//! Replaces the old emulator GUI with a trace-focused tool that renders
//! HW vs EMU event timelines side by side, highlights divergences, and
//! supports piecewise alignment for phase-aware comparison.

pub mod theme;
```

**Step 4: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: successful build

**Step 5: Commit**

```bash
git add Cargo.toml src/visual/theme.rs src/visual/mod.rs
git commit -m "feat(visual): add theme constants and rfd dependency"
```

---

### Task 3: AlignmentMap data structure

The alignment map translates between HW and EMU cycle numbers using
piecewise-linear interpolation between anchor pairs. This is a pure data
structure with no GUI dependency -- fully testable.

**Files:**
- Create: `src/visual/alignment.rs`
- Modify: `src/visual/mod.rs` (add `mod alignment`)

**Step 1: Write failing tests**

Create `src/visual/alignment.rs` with the test module first:

```rust
//! Piecewise-linear alignment between HW and EMU cycle timelines.
//!
//! Anchor pairs define known correspondences between HW and EMU cycles.
//! Between anchors, cycles are linearly interpolated. Before/after the
//! anchor range, a constant offset from the nearest anchor is applied.

/// Maps between HW and EMU cycle numbers using piecewise-linear interpolation.
///
/// # Examples
///
/// Single anchor (constant offset):
/// ```
/// # use xdna_emu::visual::alignment::AlignmentMap;
/// let map = AlignmentMap::from_single_anchor(1000, 950);
/// assert_eq!(map.hw_to_unified(1000), 950.0);
/// assert_eq!(map.emu_to_unified(950), 950.0);
/// // Both map to the same unified coordinate
/// assert_eq!(map.hw_to_unified(1100), 1050.0);
/// assert_eq!(map.emu_to_unified(1050), 1050.0);
/// ```
#[derive(Debug, Clone)]
pub struct AlignmentMap {
    /// Ordered anchor pairs: (hw_cycle, emu_cycle).
    anchors: Vec<(u64, u64)>,
}

// Implementation placeholder -- tests define the contract.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_map_is_identity() {
        let map = AlignmentMap::identity();
        assert_eq!(map.hw_to_unified(100), 100.0);
        assert_eq!(map.emu_to_unified(100), 100.0);
    }

    #[test]
    fn test_single_anchor_constant_offset() {
        // HW cycle 1000 = EMU cycle 950 -> offset of -50
        let map = AlignmentMap::from_single_anchor(1000, 950);
        // At the anchor: both map to the same unified coordinate
        assert_eq!(map.hw_to_unified(1000), map.emu_to_unified(950));
        // Before anchor: constant offset maintained
        assert_eq!(map.hw_to_unified(500), map.emu_to_unified(450));
        // After anchor: constant offset maintained
        assert_eq!(map.hw_to_unified(2000), map.emu_to_unified(1950));
    }

    #[test]
    fn test_two_anchors_interpolation() {
        let mut map = AlignmentMap::identity();
        // Anchor 1: HW=100 <-> EMU=100 (identity)
        map.add_anchor(100, 100);
        // Anchor 2: HW=200 <-> EMU=300 (EMU runs 2x faster in this region)
        map.add_anchor(200, 300);

        // At anchors: exact match
        let u1_hw = map.hw_to_unified(100);
        let u1_emu = map.emu_to_unified(100);
        assert!((u1_hw - u1_emu).abs() < 0.01);

        let u2_hw = map.hw_to_unified(200);
        let u2_emu = map.emu_to_unified(300);
        assert!((u2_hw - u2_emu).abs() < 0.01);

        // Midpoint: HW=150 should map to same unified as EMU=200
        let mid_hw = map.hw_to_unified(150);
        let mid_emu = map.emu_to_unified(200);
        assert!((mid_hw - mid_emu).abs() < 0.01);
    }

    #[test]
    fn test_add_anchor_maintains_sort_order() {
        let mut map = AlignmentMap::identity();
        map.add_anchor(300, 310);
        map.add_anchor(100, 105);
        map.add_anchor(200, 208);
        // Should be sorted by hw_cycle
        assert_eq!(map.anchor_count(), 3);
    }

    #[test]
    fn test_before_first_anchor_constant_extrapolation() {
        let mut map = AlignmentMap::identity();
        map.add_anchor(500, 480);
        // Before first anchor: constant offset from first pair
        // Offset = 480 - 500 = -20
        let hw0 = map.hw_to_unified(0);
        let emu0 = map.emu_to_unified(0);
        // HW=0 -> unified = 0 + (-20) = -20 (or 0 - 20 = -20)
        // EMU=0 -> unified = 0 (identity for EMU side)
        // The key: hw_to_unified(500) == emu_to_unified(480)
        let at_anchor_hw = map.hw_to_unified(500);
        let at_anchor_emu = map.emu_to_unified(480);
        assert!((at_anchor_hw - at_anchor_emu).abs() < 0.01);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib visual::alignment 2>&1 | tail -15`
Expected: FAIL (methods not implemented)

**Step 3: Implement AlignmentMap**

Fill in the implementation above the test module. Key methods:

- `identity() -> Self` -- empty anchors, everything maps 1:1
- `from_single_anchor(hw: u64, emu: u64) -> Self` -- one anchor pair
- `add_anchor(&mut self, hw: u64, emu: u64)` -- insert sorted
- `anchor_count(&self) -> usize`
- `hw_to_unified(&self, hw_cycle: u64) -> f64` -- map HW cycle to unified
  timeline coordinates (uses EMU cycle space as the unified basis)
- `emu_to_unified(&self, emu_cycle: u64) -> f64` -- map EMU cycle to unified
  coordinates (identity in unified space)

The unified coordinate system uses EMU cycles as the basis. `hw_to_unified`
applies the piecewise offset to translate HW cycles into this space.

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib visual::alignment 2>&1 | tail -15`
Expected: all PASS

**Step 5: Update mod.rs**

Add `pub mod alignment;` to `src/visual/mod.rs`.

**Step 6: Commit**

```bash
git add src/visual/alignment.rs src/visual/mod.rs
git commit -m "feat(visual): add AlignmentMap for piecewise timeline alignment"
```

---

### Task 4: TraceSource trait and LoadedComparison

The data abstraction layer. `TraceSource` defines what the rendering code
needs; `LoadedComparison` implements it by wrapping `trace::compare` output.

**Files:**
- Create: `src/visual/data.rs`
- Modify: `src/visual/mod.rs`

**Step 1: Write the trait and struct with tests**

```rust
//! Data abstraction for trace comparison sources.
//!
//! `TraceSource` defines the interface the timeline widget consumes.
//! `LoadedComparison` implements it for file-loaded trace pairs.

use std::collections::HashMap;
use std::path::Path;

use crate::trace::compare::{
    AnalysisOptions, BatchResult, EventsConfig, TileEvent, TileEvents, TileKey,
    compare_batch_with_opts, decode_per_tile,
};
use super::alignment::AlignmentMap;

/// Abstraction over trace comparison data sources.
///
/// File-loaded now (`LoadedComparison`); live emulator stream later.
pub trait TraceSource {
    /// All tiles with their comparison results, sorted by divergence severity.
    fn tile_keys(&self) -> &[TileKey];

    /// Batch event configuration (slot -> event name mapping).
    fn batch_config(&self) -> &EventsConfig;

    /// Raw HW events for a tile.
    fn hw_events(&self, tile: &TileKey) -> &[TileEvent];

    /// Raw EMU events for a tile.
    fn emu_events(&self, tile: &TileKey) -> &[TileEvent];

    /// The comparison result for a tile.
    fn batch_result(&self) -> &BatchResult;

    /// Current alignment map.
    fn alignment(&self) -> &AlignmentMap;

    /// Mutable access to alignment map (for interactive anchoring).
    fn alignment_mut(&mut self) -> &mut AlignmentMap;
}

/// File-loaded trace pair comparison.
///
/// Wraps a `BatchResult` from `trace::compare` alongside the raw decoded
/// events needed for timeline rendering.
pub struct LoadedComparison {
    pub batch: BatchResult,
    pub hw_events: TileEvents,
    pub emu_events: TileEvents,
    pub alignment: AlignmentMap,
    /// Tile keys sorted by divergence severity (most divergent first).
    pub sorted_keys: Vec<TileKey>,
}

impl LoadedComparison {
    /// Load and compare a trace pair from HW and EMU directories.
    ///
    /// Each directory must contain `trace_raw.bin`. An `events.json` file
    /// is loaded from the HW directory (or parent) for event name mapping.
    pub fn from_trace_dirs(
        hw_dir: &Path,
        emu_dir: &Path,
    ) -> Result<Self, String> {
        let hw_bin = hw_dir.join("trace_raw.bin");
        let emu_bin = emu_dir.join("trace_raw.bin");

        // Load events config
        let config = load_events_config(hw_dir)?;

        // Run comparison
        let opts = AnalysisOptions::default();
        let batch = compare_batch_with_opts(&hw_bin, &emu_bin, &config, 0, &opts)?;

        // Decode raw events (retained for timeline rendering)
        let hw_data = std::fs::read(&hw_bin)
            .map_err(|e| format!("read {}: {}", hw_bin.display(), e))?;
        let emu_data = std::fs::read(&emu_bin)
            .map_err(|e| format!("read {}: {}", emu_bin.display(), e))?;
        let hw_events = decode_per_tile(&hw_data);
        let emu_events = decode_per_tile(&emu_data);

        // Build alignment from batch t0 values
        let alignment = if let Some((_, tile_result)) = batch.tiles.first() {
            AlignmentMap::from_single_anchor(tile_result.hw_t0, tile_result.emu_t0)
        } else {
            AlignmentMap::identity()
        };

        // Sort tiles by divergence count (descending)
        let mut sorted_keys: Vec<TileKey> = batch.tiles.iter().map(|(k, _)| *k).collect();
        sorted_keys.sort_by(|a, b| {
            let a_div = divergence_count(&batch, a);
            let b_div = divergence_count(&batch, b);
            b_div.cmp(&a_div)
        });

        Ok(Self {
            batch,
            hw_events,
            emu_events,
            alignment,
            sorted_keys,
        })
    }
}

impl TraceSource for LoadedComparison {
    fn tile_keys(&self) -> &[TileKey] {
        &self.sorted_keys
    }

    fn batch_config(&self) -> &EventsConfig {
        &self.batch.config
    }

    fn hw_events(&self, tile: &TileKey) -> &[TileEvent] {
        self.hw_events.get(tile).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn emu_events(&self, tile: &TileKey) -> &[TileEvent] {
        self.emu_events.get(tile).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn batch_result(&self) -> &BatchResult {
        &self.batch
    }

    fn alignment(&self) -> &AlignmentMap {
        &self.alignment
    }

    fn alignment_mut(&mut self) -> &mut AlignmentMap {
        &mut self.alignment
    }
}

/// Count total divergences for a tile across all edge and level results.
fn divergence_count(batch: &BatchResult, key: &TileKey) -> usize {
    batch.tiles.iter()
        .find(|(k, _)| k == key)
        .map(|(_, result)| {
            let edge_divs = result.edge_results.iter()
                .filter(|e| e.diverge_idx.is_some())
                .count();
            let level_divs = result.level_results.iter()
                .filter(|l| l.diverge_idx.is_some())
                .count();
            edge_divs + level_divs
        })
        .unwrap_or(0)
}

/// Load EventsConfig from a directory (looks for events.json).
fn load_events_config(dir: &Path) -> Result<EventsConfig, String> {
    // Try dir/events.json, then dir/../events.json (sweep layout)
    let candidates = [
        dir.join("events.json"),
        dir.parent().map(|p| p.join("events.json")).unwrap_or_default(),
    ];
    for path in &candidates {
        if path.exists() {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("read {}: {}", path.display(), e))?;
            let config: EventsConfig = serde_json::from_str(&text)
                .map_err(|e| format!("parse {}: {}", path.display(), e))?;
            return Ok(config);
        }
    }
    // No events.json found -- return empty config (all events unnamed)
    Ok(EventsConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divergence_count_empty_batch() {
        let batch = BatchResult {
            batch_idx: 0,
            config: EventsConfig::default(),
            tiles: vec![],
            stall_attributions: vec![],
            cross_tile: None,
        };
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        assert_eq!(divergence_count(&batch, &key), 0);
    }

    #[test]
    fn test_load_events_config_missing_returns_default() {
        let dir = std::path::Path::new("/nonexistent/path");
        let config = load_events_config(dir).unwrap();
        assert!(config.core_events.is_empty());
    }
}
```

**Step 2: Run tests**

Run: `cargo test --lib visual::data 2>&1 | tail -10`
Expected: PASS

**Step 3: Update mod.rs**

Add `pub mod data;` to `src/visual/mod.rs`.

**Step 4: Commit**

```bash
git add src/visual/data.rs src/visual/mod.rs
git commit -m "feat(visual): add TraceSource trait and LoadedComparison"
```

---

## Phase 2: App Shell

### Task 5: Tile selector sidebar

The sidebar lists all tiles sorted by divergence count. Clicking a tile
selects it for timeline display.

**Files:**
- Create: `src/visual/tile_selector.rs`
- Modify: `src/visual/mod.rs`

**Step 1: Create tile_selector.rs**

```rust
//! Sidebar widget listing tiles sorted by divergence severity.
//!
//! Shows each tile's (col, row) identifier, tile type (Core/Mem/MemTile/Shim),
//! and divergence count. Selected tile is highlighted. Click to select.

use eframe::egui;
use crate::trace::compare::TileKey;
use super::data::TraceSource;

/// Render the tile selector sidebar.
///
/// Returns `Some(TileKey)` if the user clicked a new tile.
pub fn show_tile_selector(
    ui: &mut egui::Ui,
    source: &dyn TraceSource,
    selected: Option<&TileKey>,
) -> Option<TileKey> {
    let mut new_selection = None;

    ui.heading("Tiles");
    ui.separator();

    let batch = source.batch_result();

    egui::ScrollArea::vertical().show(ui, |ui| {
        for key in source.tile_keys() {
            let div_count = batch.tiles.iter()
                .find(|(k, _)| k == key)
                .map(|(_, r)| {
                    r.edge_results.iter().filter(|e| e.diverge_idx.is_some()).count()
                    + r.level_results.iter().filter(|l| l.diverge_idx.is_some()).count()
                })
                .unwrap_or(0);

            let tile_type = match key.pkt_type {
                0 => "Core",
                1 => "Mem",
                3 => "MTile",
                _ => "Shim",
            };

            let is_selected = selected == Some(key);
            let label = format!(
                "({},{}) {}  {} diverge",
                key.col, key.row, tile_type, div_count
            );

            let response = ui.selectable_label(is_selected, &label);
            if response.clicked() {
                new_selection = Some(*key);
            }
        }
    });

    new_selection
}
```

**Step 2: Update mod.rs**

Add `pub mod tile_selector;` to `src/visual/mod.rs`.

**Step 3: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: successful build

**Step 4: Commit**

```bash
git add src/visual/tile_selector.rs src/visual/mod.rs
git commit -m "feat(visual): add tile selector sidebar widget"
```

---

### Task 6: Event detail panel

Bottom panel showing info for the hovered/selected event.

**Files:**
- Create: `src/visual/event_detail.rs`
- Modify: `src/visual/mod.rs`

**Step 1: Create event_detail.rs**

```rust
//! Bottom panel showing details for a selected event or region.
//!
//! Displays cycle number, delta (HW-EMU), event name, and paired values
//! for the currently hovered or clicked timeline element.

use eframe::egui;

/// Information about a selected event for the detail panel.
#[derive(Debug, Clone)]
pub struct SelectedEvent {
    /// Event name (e.g., "INSTR_VECTOR").
    pub name: String,
    /// Occurrence index within this event type.
    pub index: usize,
    /// HW absolute cycle.
    pub hw_cycle: Option<u64>,
    /// EMU absolute cycle.
    pub emu_cycle: Option<u64>,
    /// Delta (hw - emu), if both sides present.
    pub delta: Option<i64>,
    /// Whether this is an edge event or level event.
    pub is_level: bool,
    /// For level events: duration in HW.
    pub hw_duration: Option<u64>,
    /// For level events: duration in EMU.
    pub emu_duration: Option<u64>,
}

/// Render the event detail panel.
pub fn show_event_detail(ui: &mut egui::Ui, event: Option<&SelectedEvent>) {
    match event {
        Some(ev) => {
            ui.horizontal(|ui| {
                ui.strong(&ev.name);
                ui.label(format!("#{}", ev.index));
                ui.separator();

                if let Some(hw) = ev.hw_cycle {
                    ui.label(format!("HW cy={}", hw));
                }
                if let Some(emu) = ev.emu_cycle {
                    ui.label(format!("EMU cy={}", emu));
                }
                if let Some(delta) = ev.delta {
                    let sign = if delta >= 0 { "+" } else { "" };
                    ui.label(format!("dt={}{}", sign, delta));
                }

                if ev.is_level {
                    ui.separator();
                    if let Some(dur) = ev.hw_duration {
                        ui.label(format!("HW dur={}", dur));
                    }
                    if let Some(dur) = ev.emu_duration {
                        ui.label(format!("EMU dur={}", dur));
                    }
                }
            });
        }
        None => {
            ui.label("Hover or click an event in the timeline for details.");
        }
    }
}
```

**Step 2: Update mod.rs, verify compilation**

Add `pub mod event_detail;` to `src/visual/mod.rs`.

Run: `cargo build 2>&1 | tail -5`

**Step 3: Commit**

```bash
git add src/visual/event_detail.rs src/visual/mod.rs
git commit -m "feat(visual): add event detail panel widget"
```

---

### Task 7: App shell with file loading

The main `TraceViewerApp` struct and `eframe::App` implementation. Handles
layout, file open dialogs, and wires the sidebar and detail panel together.
The timeline is a placeholder rectangle at this stage.

**Files:**
- Create: `src/visual/app.rs`
- Modify: `src/visual/mod.rs` (add re-export)
- Modify: `src/main.rs` (wire `run_gui`)

**Step 1: Create app.rs**

```rust
//! Main trace visualizer application.

use eframe::egui;
use std::path::PathBuf;

use crate::trace::compare::TileKey;
use super::data::{LoadedComparison, TraceSource};
use super::event_detail::{self, SelectedEvent};
use super::tile_selector;

/// Application state for the trace visualizer.
pub struct TraceViewerApp {
    /// Loaded trace comparison (None until user opens files).
    source: Option<LoadedComparison>,
    /// Currently selected tile.
    selected_tile: Option<TileKey>,
    /// Currently selected/hovered event for detail panel.
    selected_event: Option<SelectedEvent>,
    /// Status message for the status bar.
    status: String,
    /// Error message popup.
    error: Option<String>,
}

impl Default for TraceViewerApp {
    fn default() -> Self {
        Self {
            source: None,
            selected_tile: None,
            selected_event: None,
            status: "Ready. Use File > Open Trace Pair to load traces.".into(),
            error: None,
        }
    }
}

impl TraceViewerApp {
    /// Open a trace pair via native file dialog.
    fn open_trace_pair(&mut self) {
        // Select HW trace directory
        let hw_dir = rfd::FileDialog::new()
            .set_title("Select HW trace directory")
            .pick_folder();
        let hw_dir = match hw_dir {
            Some(d) => d,
            None => return, // cancelled
        };

        // Select EMU trace directory
        let emu_dir = rfd::FileDialog::new()
            .set_title("Select EMU trace directory")
            .pick_folder();
        let emu_dir = match emu_dir {
            Some(d) => d,
            None => return,
        };

        self.load_trace_pair(&hw_dir, &emu_dir);
    }

    /// Load a trace pair from two directories.
    fn load_trace_pair(&mut self, hw_dir: &PathBuf, emu_dir: &PathBuf) {
        match LoadedComparison::from_trace_dirs(hw_dir, emu_dir) {
            Ok(comparison) => {
                let tile_count = comparison.tile_keys().len();
                self.selected_tile = comparison.tile_keys().first().copied();
                self.status = format!(
                    "Loaded: {} tiles from {} / {}",
                    tile_count,
                    hw_dir.display(),
                    emu_dir.display(),
                );
                self.source = Some(comparison);
                self.error = None;
            }
            Err(e) => {
                self.error = Some(format!("Failed to load traces: {}", e));
            }
        }
    }
}

impl eframe::App for TraceViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Trace Pair...").clicked() {
                        ui.close_menu();
                        self.open_trace_pair();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });

        // Status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
            });
        });

        // Detail panel
        egui::TopBottomPanel::bottom("detail_panel")
            .resizable(true)
            .default_height(32.0)
            .show(ctx, |ui| {
                event_detail::show_event_detail(ui, self.selected_event.as_ref());
            });

        // Error popup
        if let Some(error) = self.error.clone() {
            egui::Window::new("Error")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(&error);
                    if ui.button("OK").clicked() {
                        self.error = None;
                    }
                });
        }

        // Sidebar (tile selector)
        if let Some(source) = &self.source {
            egui::SidePanel::left("tile_sidebar")
                .default_width(180.0)
                .resizable(true)
                .show(ctx, |ui| {
                    if let Some(new_tile) = tile_selector::show_tile_selector(
                        ui,
                        source as &dyn TraceSource,
                        self.selected_tile.as_ref(),
                    ) {
                        self.selected_tile = Some(new_tile);
                        self.selected_event = None;
                    }
                });
        }

        // Central panel (timeline placeholder)
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.source.is_some() {
                ui.centered_and_justified(|ui| {
                    ui.label("Timeline will render here.");
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("No traces loaded.\nUse File > Open Trace Pair to begin.");
                });
            }
        });
    }
}
```

**Step 2: Update mod.rs with re-export**

```rust
//! Trace comparison visualizer for AMD XDNA NPU emulation.

pub mod alignment;
pub mod app;
pub mod data;
pub mod event_detail;
pub mod theme;
pub mod tile_selector;

pub use app::TraceViewerApp;
```

**Step 3: Wire into main.rs**

Replace the `run_gui` stub and update the import:

Change `use xdna_emu::visual::EmulatorApp;` to
`use xdna_emu::visual::TraceViewerApp;` (or remove it since `run_gui`
constructs it inline).

```rust
fn run_gui(_file_path: Option<&str>) -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 800.0])
            .with_title("xdna-emu Trace Visualizer"),
        ..Default::default()
    };

    eframe::run_native(
        "xdna-emu",
        options,
        Box::new(|_cc| Ok(Box::new(xdna_emu::visual::TraceViewerApp::default()))),
    ).map_err(|e| anyhow::anyhow!("GUI error: {}", e))
}
```

**Step 4: Verify it compiles and runs**

Run: `cargo build 2>&1 | tail -5`
Expected: successful build

Manual test: `cargo run -- --gui` should open a window with menu bar, status
bar, and "No traces loaded" message.

**Step 5: Commit**

```bash
git add src/visual/app.rs src/visual/mod.rs src/main.rs
git commit -m "feat(visual): app shell with file loading, sidebar, detail panel"
```

---

## Phase 3: Timeline Rendering

### Task 8: Timeline viewport and zoom state

The timeline needs coordinate transforms: cycle-to-pixel, pixel-to-cycle,
zoom level, pan offset. This is a pure data structure.

**Files:**
- Create: `src/visual/viewport.rs`
- Modify: `src/visual/mod.rs`

**Step 1: Create viewport.rs with tests**

```rust
//! Timeline viewport: maps between cycle numbers and screen coordinates.
//!
//! Handles zoom (pixels per cycle) and pan (scroll offset in cycles).
//! Provides efficient cycle-to-pixel and pixel-to-cycle conversions.

/// Viewport state for the timeline widget.
#[derive(Debug, Clone)]
pub struct Viewport {
    /// First visible cycle (left edge of viewport).
    pub start_cycle: f64,
    /// Pixels per cycle (zoom level). Higher = more zoomed in.
    pub px_per_cycle: f64,
    /// Total width of the viewport in pixels.
    pub width_px: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            start_cycle: 0.0,
            px_per_cycle: 0.1,
            width_px: 1000.0,
        }
    }
}

impl Viewport {
    /// Last visible cycle (right edge of viewport).
    pub fn end_cycle(&self) -> f64 {
        self.start_cycle + (self.width_px as f64) / self.px_per_cycle
    }

    /// Number of cycles visible in the viewport.
    pub fn visible_cycles(&self) -> f64 {
        (self.width_px as f64) / self.px_per_cycle
    }

    /// Convert a cycle number to pixel x-coordinate within the viewport.
    pub fn cycle_to_px(&self, cycle: f64) -> f32 {
        ((cycle - self.start_cycle) * self.px_per_cycle) as f32
    }

    /// Convert a pixel x-coordinate to cycle number.
    pub fn px_to_cycle(&self, px: f32) -> f64 {
        self.start_cycle + (px as f64) / self.px_per_cycle
    }

    /// Returns true if a cycle falls within the visible viewport.
    pub fn is_visible(&self, cycle: f64) -> bool {
        cycle >= self.start_cycle && cycle <= self.end_cycle()
    }

    /// Zoom by a factor, keeping the given pixel position anchored.
    ///
    /// `factor` > 1.0 zooms in, < 1.0 zooms out.
    pub fn zoom_at(&mut self, factor: f64, anchor_px: f32) {
        let anchor_cycle = self.px_to_cycle(anchor_px);
        self.px_per_cycle *= factor;
        // Clamp zoom level
        self.px_per_cycle = self.px_per_cycle.clamp(0.0001, 100.0);
        // Adjust start_cycle so the anchor stays at the same pixel position
        self.start_cycle = anchor_cycle - (anchor_px as f64) / self.px_per_cycle;
    }

    /// Pan by a number of pixels (negative = pan left).
    pub fn pan_px(&mut self, delta_px: f32) {
        self.start_cycle -= (delta_px as f64) / self.px_per_cycle;
    }

    /// Fit a cycle range into the viewport.
    pub fn fit_range(&mut self, min_cycle: f64, max_cycle: f64) {
        let range = (max_cycle - min_cycle).max(1.0);
        self.px_per_cycle = (self.width_px as f64) / range;
        self.start_cycle = min_cycle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_to_px_roundtrip() {
        let vp = Viewport {
            start_cycle: 100.0,
            px_per_cycle: 0.5,
            width_px: 500.0,
        };
        let cycle = 300.0;
        let px = vp.cycle_to_px(cycle);
        let back = vp.px_to_cycle(px);
        assert!((back - cycle).abs() < 0.001);
    }

    #[test]
    fn test_visible_cycles() {
        let vp = Viewport {
            start_cycle: 0.0,
            px_per_cycle: 0.1,
            width_px: 1000.0,
        };
        assert!((vp.visible_cycles() - 10000.0).abs() < 0.01);
    }

    #[test]
    fn test_zoom_at_preserves_anchor() {
        let mut vp = Viewport {
            start_cycle: 0.0,
            px_per_cycle: 1.0,
            width_px: 1000.0,
        };
        let anchor_px = 500.0;
        let cycle_at_anchor = vp.px_to_cycle(anchor_px);
        vp.zoom_at(2.0, anchor_px);
        let cycle_after = vp.px_to_cycle(anchor_px);
        assert!((cycle_at_anchor - cycle_after).abs() < 0.01);
    }

    #[test]
    fn test_fit_range() {
        let mut vp = Viewport::default();
        vp.width_px = 1000.0;
        vp.fit_range(500.0, 1500.0);
        assert!((vp.start_cycle - 500.0).abs() < 0.01);
        assert!((vp.px_per_cycle - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_is_visible() {
        let vp = Viewport {
            start_cycle: 100.0,
            px_per_cycle: 1.0,
            width_px: 200.0,
        };
        assert!(vp.is_visible(150.0));
        assert!(vp.is_visible(100.0));
        assert!(vp.is_visible(300.0));
        assert!(!vp.is_visible(99.0));
        assert!(!vp.is_visible(301.0));
    }
}
```

**Step 2: Run tests, update mod.rs, commit**

Run: `cargo test --lib visual::viewport`
Expected: all PASS

Add `pub mod viewport;` to `src/visual/mod.rs`.

```bash
git add src/visual/viewport.rs src/visual/mod.rs
git commit -m "feat(visual): add Viewport for timeline zoom/pan state"
```

---

### Task 9: Timeline widget -- event lane rendering

The core visualization: horizontal event lanes showing edge ticks and level
bars for HW and EMU traces.

This is the largest task. It renders into egui's Painter API.

**Files:**
- Create: `src/visual/timeline.rs`
- Modify: `src/visual/app.rs` (integrate timeline)
- Modify: `src/visual/mod.rs`

**Step 1: Create timeline.rs**

This file implements the timeline rendering. Key responsibilities:

- Accept a `&dyn TraceSource`, selected `TileKey`, and `&mut Viewport`
- Compute event lanes (one per event name) for both HW and EMU blocks
- Render edge events as vertical ticks at cycle positions
- Render level events as filled rectangles (begin-to-end)
- Handle scroll-wheel zoom and click-drag pan
- Binary-search events to only render the visible viewport
- Draw connecting lines between matched HW/EMU event pairs
- Color divergent pairs red, matched pairs gray

The file will be ~200-300 lines. Key structure:

```rust
//! Timeline widget: dual-lane event visualization with zoom/pan.

use eframe::egui;
use crate::trace::compare::TileKey;
use super::data::TraceSource;
use super::viewport::Viewport;
use super::event_detail::SelectedEvent;
use super::theme;

/// Persistent state for the timeline widget (lives in app).
pub struct TimelineState {
    pub viewport: Viewport,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            viewport: Viewport::default(),
        }
    }
}

/// Render the timeline for a selected tile.
///
/// Returns `Some(SelectedEvent)` if the user hovered/clicked an event.
pub fn show_timeline(
    ui: &mut egui::Ui,
    source: &dyn TraceSource,
    tile: &TileKey,
    state: &mut TimelineState,
) -> Option<SelectedEvent> {
    // Implementation: see design doc for rendering rules.
    // This function:
    // 1. Gets the batch result for this tile to find event names
    // 2. Allocates vertical space: HW block (n lanes), gap, EMU block (n lanes)
    // 3. Iterates visible events via binary search on sorted abs_cycle
    // 4. Draws ticks/bars using the egui Painter
    // 5. Handles zoom (scroll) and pan (drag) input
    // 6. Draws divergence connecting lines between paired events
    //
    // Full implementation written by developer in this task.
    todo!("Timeline rendering -- implement per design doc")
}
```

The developer implementing this task should reference:
- `docs/plans/2026-03-08-trace-visualizer-design.md` section "Timeline Rendering"
- `src/trace/compare.rs` for `EdgeResult` (has `deltas`, `samples`) and
  `LevelResult` (has `dur_deltas`, `samples`)
- `src/visual/theme.rs` for all colors and dimensions
- `src/visual/viewport.rs` for coordinate transforms

Key egui APIs needed:
- `ui.allocate_rect()` / `ui.allocate_space()` for the timeline area
- `ui.painter()` for drawing primitives (line, rect, circle)
- `ui.input()` for scroll wheel (zoom) and pointer drag (pan)
- Binary search: `partition_point()` on sorted `TileEvent.abs_cycle`

**Step 2: Integrate into app.rs**

Replace the "Timeline will render here" placeholder in the central panel
with a call to `timeline::show_timeline()`. Add `TimelineState` to
`TraceViewerApp`.

**Step 3: Verify it compiles (with todo!)**

The `todo!()` compiles -- it just panics at runtime. The next step is to
implement the rendering.

**Step 4: Implement timeline rendering**

This is where the real work happens. Implement `show_timeline` following the
design doc. Test by running `cargo run -- --gui` and loading a trace pair.

Suggested approach -- build incrementally:
1. First: just draw the lane backgrounds (colored rectangles)
2. Then: add edge event ticks (vertical lines at cycle positions)
3. Then: add zoom/pan interaction
4. Then: add divergence lines between HW/EMU pairs
5. Then: add click-to-inspect (return SelectedEvent)
6. Then: add minimap

**Step 5: Manual visual testing**

Run: `cargo run -- --gui`
- Open a trace pair from a bridge test (e.g., `add_one_using_dma`)
- Verify: tiles appear in sidebar sorted by divergence
- Verify: timeline shows colored lanes with event ticks
- Verify: scroll wheel zooms, drag pans
- Verify: clicking an event shows details in bottom panel

**Step 6: Commit**

```bash
git add src/visual/timeline.rs src/visual/app.rs src/visual/mod.rs
git commit -m "feat(visual): timeline widget with event lanes, zoom/pan, divergence overlay"
```

---

## Phase 4: Polish

### Task 10: Minimap

A thin bar above the timeline showing the full trace extent with the current
viewport highlighted.

**Files:**
- Modify: `src/visual/timeline.rs` (add minimap rendering)

**Step 1: Add minimap rendering function**

```rust
/// Draw the minimap bar showing full trace extent.
fn draw_minimap(
    ui: &mut egui::Ui,
    total_min: f64,
    total_max: f64,
    viewport: &Viewport,
) {
    let (rect, _response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), theme::MINIMAP_HEIGHT),
        egui::Sense::click_and_drag(),
    );

    let painter = ui.painter_at(rect);

    // Background
    painter.rect_filled(rect, 0.0, theme::MINIMAP_BG);

    // Viewport indicator
    let total_range = (total_max - total_min).max(1.0);
    let vp_start_frac = ((viewport.start_cycle - total_min) / total_range).clamp(0.0, 1.0);
    let vp_end_frac = ((viewport.end_cycle() - total_min) / total_range).clamp(0.0, 1.0);

    let vp_rect = egui::Rect::from_min_max(
        egui::pos2(
            rect.left() + vp_start_frac as f32 * rect.width(),
            rect.top(),
        ),
        egui::pos2(
            rect.left() + vp_end_frac as f32 * rect.width(),
            rect.bottom(),
        ),
    );
    painter.rect_filled(vp_rect, 0.0, theme::MINIMAP_VIEWPORT);
}
```

**Step 2: Wire minimap into show_timeline, before the main lanes**

**Step 3: Manual test, commit**

```bash
git add src/visual/timeline.rs
git commit -m "feat(visual): add minimap to timeline"
```

---

### Task 11: Batch selector for sweep directories

When a sweep directory is loaded, show a dropdown to switch between batches.
This is a future enhancement -- for v1, we support single trace pair loading
only. Add the UI hook but leave sweep loading as a TODO.

**Files:**
- Modify: `src/visual/app.rs` (add batch selector to menu bar area)

**Step 1: Add batch index and selector to app state**

Add `batch_names: Vec<String>` and `selected_batch: usize` to
`TraceViewerApp`. Show a `ComboBox` in the top panel when
`batch_names.len() > 1`.

**Step 2: Commit**

```bash
git add src/visual/app.rs
git commit -m "feat(visual): add batch selector UI for sweep directories"
```

---

### Task 12: Keyboard shortcuts and final polish

Add keyboard shortcuts for common operations.

**Files:**
- Modify: `src/visual/app.rs`

**Step 1: Add keyboard handling in the update loop**

- `Home` / `F` -- fit timeline to full trace range
- `Left/Right` arrows -- pan
- `+/-` -- zoom in/out
- `Escape` -- deselect event

**Step 2: Add drag-and-drop support for trace directories**

Check `ctx.input()` for dropped files/folders. If a pair of directories
is dropped, load them.

**Step 3: Final manual test**

Run: `cargo run -- --gui`
- Open trace pair from bridge test results
- Verify all features work end-to-end
- Test keyboard shortcuts
- Test zoom at different levels

**Step 4: Commit**

```bash
git add src/visual/app.rs
git commit -m "feat(visual): keyboard shortcuts and drag-drop support"
```

---

## Phase 5: Integration

### Task 13: CLI trace viewer launch mode

Add a `--trace-view` CLI flag that opens the GUI pre-loaded with a trace pair.

**Files:**
- Modify: `src/main.rs`

**Step 1: Add CLI parsing**

```rust
// In argument parsing:
let trace_view_hw: Option<&str> = /* parse --trace-view-hw <dir> */;
let trace_view_emu: Option<&str> = /* parse --trace-view-emu <dir> */;
```

**Step 2: Pass to run_gui, load on startup**

If both `--trace-view-hw` and `--trace-view-emu` are provided, construct a
`TraceViewerApp` and call `load_trace_pair` before entering the event loop.

**Step 3: Update help text**

Add the new flags to `print_help()`.

**Step 4: Test**

```bash
# From a bridge test with traces:
cargo run -- --trace-view-hw /path/to/hw --trace-view-emu /path/to/emu
```

**Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): add --trace-view-hw/emu flags for direct trace loading"
```

---

### Task 14: Final cleanup and test pass

**Step 1: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

All tests must pass.

**Step 2: Run clippy**

```bash
cargo clippy 2>&1 | head -30
```

Fix any warnings in new code.

**Step 3: Verify GUI launches clean**

```bash
cargo run -- --gui
```

No panics, no warnings.

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "chore(visual): clippy fixes and final cleanup"
```

---

## Summary

| Task | Description | Files | Est. Lines |
|------|------------|-------|------------|
| 1 | Delete old GUI | -5 files, mod 2 | -1085 |
| 2 | Theme + rfd dep | +1 file, mod 2 | ~60 |
| 3 | AlignmentMap | +1 file | ~120 |
| 4 | TraceSource + LoadedComparison | +1 file | ~160 |
| 5 | Tile selector sidebar | +1 file | ~60 |
| 6 | Event detail panel | +1 file | ~80 |
| 7 | App shell + file loading | +1 file, mod 2 | ~170 |
| 8 | Viewport state | +1 file | ~120 |
| 9 | Timeline widget | +1 file, mod 1 | ~300 |
| 10 | Minimap | mod 1 | ~30 |
| 11 | Batch selector | mod 1 | ~20 |
| 12 | Keyboard shortcuts | mod 1 | ~40 |
| 13 | CLI launch mode | mod 1 | ~30 |
| 14 | Cleanup + test pass | mod N | ~10 |
| **Total** | | **+8 new, -5 deleted** | **~1200 new** |

Net: replace ~1,085 lines of dead GUI with ~1,200 lines of trace visualizer.
