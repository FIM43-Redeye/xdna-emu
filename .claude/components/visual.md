# Visual Debugger

egui-based GUI for visualizing NPU emulation state: tile grid, core details, memory hex view, and emulation controls.

Read this file when working on anything in `src/visual/`.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports `EmulatorApp` |
| `app.rs` | `EmulatorApp` -- main egui application, top-level layout and state management |
| `tile_grid.rs` | Tile array visualization -- grid of tiles with status-colored cells |
| `tile_detail.rs` | Detail panels -- core state (PC, registers), locks, DMA buffer descriptors |
| `controls.rs` | Emulation controls -- Run, Step, Pause, Reset buttons |
| `memory_view.rs` | Hex memory viewer -- displays tile memory contents |

## Key Types

- `EmulatorApp` -- implements `eframe::App`, holds all GUI state and a reference to the emulator engine

## GUI Framework

Uses **egui** via `eframe`:
- Immediate-mode rendering (no retained widget tree)
- Pure Rust, no system GUI dependencies
- Cross-platform (Linux, Windows, macOS, web via WASM)

## Current State

The GUI framework renders and the tile grid displays. However:
- No debugging features (breakpoints, watchpoints) are implemented
- No execution timeline or profiling views exist
- The GUI was last exercised several weeks ago and may need updates to match current data structures in the device and interpreter modules
- This is Phase 3 work (Developer Experience) and is not the current priority

## Launch

```bash
# Launch with GUI (requires a display server)
cargo run -- --gui path/to/binary.xclbin
```
