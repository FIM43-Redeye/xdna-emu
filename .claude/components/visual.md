# Visual Layer

egui-based trace comparison viewer. The visual layer was retargeted from
a generic emulator state debugger to a HW vs EMU trace timeline tool.

Read this file when working on anything in `src/visual/`.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports `TraceViewerApp` |
| `app.rs` | `TraceViewerApp` -- main egui application; top-level layout and state |
| `data.rs` | Loaded trace data (events, tiles, metadata) |
| `theme.rs` | Color palette, typography, layout constants |
| `viewport.rs` | Viewport / pan-zoom state for the timeline |
| `timeline.rs` | Per-tile event-track rendering (HW track + EMU track + divergence overlay) |
| `tile_selector.rs` | Tile picker / filter UI |
| `event_detail.rs` | Detail panel for the currently selected event |
| `alignment.rs` | Piecewise alignment between HW and EMU timelines (phase-aware comparison) |

## Key Types

- `TraceViewerApp` -- implements `eframe::App`, owns loaded HW/EMU
  traces and the alignment state

## What It Shows

- Two horizontal tracks per tile (HW above, EMU below)
- Divergence highlighting where the two streams disagree
- Piecewise alignment: anchor points let the user re-align segments to
  see whether divergences are timing drift or genuine semantic
  differences

## What It Does Not Do

The trace viewer is a comparison tool, not a live debugger. It does not
attach to a running emulator instance, set breakpoints, or step
execution. Those Phase-3-Developer-Experience features are not built;
this viewer was the higher-leverage GUI investment.

## Launch

```bash
cargo run --release -- --visual <hw_events.json> <emu_events.json>
```

(See `src/main.rs` for current CLI flags.)
