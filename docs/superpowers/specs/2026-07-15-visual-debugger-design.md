# xdna-emu Visual Debugger -- v1 Design

**Date:** 2026-07-15
**Status:** design (brainstormed with Maya; awaiting spec review before planning)
**Supersedes:** the existing `src/visual/` trace-comparison viewer, which is
retired (see "Disposition of the old viewer").

## Vision

A live, interactive visual debugger for the XDNA NPU array: load a real
`.xclbin`, step the emulator, and *see* what every part of the array is doing.
The north star is "visualize literally everything the emulator does" -- the
architecture must let each new thing we render (DMA, locks, routing, memory,
compute) slot in at the level where it belongs, incrementally, without a
rewrite.

This is the CLAUDE.md project vision's visual debugger, finally built for real.
It is developer-facing (lowers the barrier to NPU programming), not just an
internal validation tool.

## Scope of v1 (the first slice)

A **live, steppable, multi-panel** debugger showing the array and one selected
tile's internals as it executes. Concretely, v1 is done when:

1. The app loads an `.xclbin` and shows the **array** (5x6 for NPU1) as a
   spatial grid of type-colored tiles, laid out from the device model.
2. **Time controls** (step 1 cycle / step N / run / pause / reset) drive the
   emulator; the display updates as it steps.
3. Clicking a tile selects it; a **detail panel** shows that tile's live state
   as functional rows of values (not yet pretty): core status + PC, both DMA
   channels' state, the 64 locks, local-memory size, stream ports.

v1 is functional-first. Legibility/polish is an explicit later pass.

## Interaction model (the decided architecture)

**Multi-panel master-detail.** NOT continuous/semantic zoom -- that was
considered and rejected (hard to keep legible at every scale; not wanted).

- **Overview panel** -- the array as a fixed spatial grid that fits the window.
  Tiles are type-colored boxes (shim / mem / core) labeled with `(col,row)`.
  The selected tile is highlighted. Click to select.
- **Detail panel** -- the selected tile's internals, as rows of live values.
- **Control bar** -- load, and the time controls, plus a cycle counter /
  engine status readout.

Discrete levels, each a purpose-built legible layout -- this is why panels beat
zoom here, and why "rows of values" drop straight in. It is also the most
egui-native and cheapest-to-build-well option.

**Deferred (not v1, but the architecture must not preclude them):**
- Overview **pan/zoom** (turning the fixed grid into a navigable map) -- an
  additive enhancement, not a re-architecture.
- **Connections/routing lines** between tiles (drawn from `StreamPort.route_to`
  / configured flows) -- the natural second slice, once tiles are solid.
- Animation, memory hex editor, breakpoints, per-resource drill-down tabs.

## Architecture

egui/eframe app. The app **owns an `InterpreterEngine`** and drives it directly
(the run-to-completion `xclbin_suite` entry points are not usable for
stepping -- the GUI replicates the load sequence and steps the engine itself).

**Frame loop (per egui repaint):**
1. If in "running" mode, advance the engine (`engine.run(budget)` or N x
   `engine.step()`), bounded per frame so the UI stays responsive; if the
   design loads NPU insts, interleave `executor.try_advance(device, host_mem)`
   before each `engine.step()` (mirrors `xclbin_suite::run_engine`).
2. Read current state from the engine/array (below) and render both panels.
3. Repaint.

Pure read-and-render: **no new emulator execution logic**. The GUI is a viewer
over existing state.

### Data sources (all confirmed to exist)

Layout (static, per load):
- Array topology from the device model (`tools/aie-device-models.json`, already
  data-driven) -- tile grid, per-tile type. No hardcoded topology.

Execution control (`InterpreterEngine`, public):
- `new_npu1()`, `step()`, `run(max_cycles) -> u64`, `pause()`, `resume()`,
  `reset()`, `status() -> EngineStatus`, `total_cycles() -> u64`,
  `active_cores()`, `enabled_cores()`.

Load sequence (public today, but hand-assembled privately in
`xclbin_suite::run_single_inner`): `Xclbin::from_file` -> `AiePartition::parse`
-> `Cdo::parse` -> `new_npu1()` -> `assign_partition_columns` -> `apply_cdo` ->
`load_elf_bytes(col,row,..)` per tile -> `sync_cores_from_device` -> optional
`NpuExecutor`.

Per-tile state reads:
- **Core:** `engine.core_status(col,row) -> CoreStatus`
  (Ready/Running/WaitingLock/WaitingDma/WaitingStream/WaitBank/Halted/Error) +
  `engine.core_context(col,row)?.pc()`. (Read live PC here, NOT `Tile.core.pc`,
  which is load-time only.)
- **Locks:** `array.get(col,row)?.locks[i].value` (all 64) + `effective_lock_value`.
- **Local memory:** `Tile.data_memory()`, `read_data_u32`, `data_memory_gen()`
  (generation counter -> skip re-reading unchanged memory).
- **Streams:** `Tile.stream_switch` masters/slaves/`local_routes`;
  `StreamPort.cycle_active/cycle_stalled/route_to` for per-cycle activity.
- **DMA (coarse, live):** `array.dma_engine(col,row)?` -> `channel_state`,
  `channel_active`, `channel_has_pending_work`, `task_queue_size`, `get_bd(id)`.

### Emulator-side additions v1 needs (small, in-crate)

1. **A shared load helper.** Extract the load sequence from
   `xclbin_suite::run_single_inner` into a reusable `pub` function (e.g.
   `InterpreterEngine::load_xclbin(path) -> Result<Self, _>` or a free function
   in a shared module) so the GUI and the suite share ONE load path (DRY). This
   also de-risks the GUI tracking future load-sequence changes.
2. **Live DMA accessors.** `DmaEngine.channels` is `pub(super)`, so the live
   **current BD id** and **task-queue contents** are unreachable. Add small
   public accessors: `current_bd(ch) -> Option<u8>`, `queued_bd(ch) -> Option<u8>`,
   and a task-queue iterator. Read-only, no behavior change.

Everything else the detail panel needs is already public.

## Components / file structure (proposed)

New `src/visual/` (after clearing the old one):
- `app.rs` -- `DebuggerApp: eframe::App`; owns `InterpreterEngine`, run/pause
  state, selected tile; top-level layout (control bar + side panel + central).
- `engine_host.rs` -- the load + per-frame drive wrapper around
  `InterpreterEngine` (frame-bounded stepping, NPU-inst interleave).
- `overview.rs` -- array grid rendering + click-to-select (device-model-driven).
- `detail.rs` -- selected-tile detail panel (rows of live values).
- `controls.rs` -- control bar (load, step/run/pause/reset, cycle/status).
- `theme.rs` -- salvaged from the old viewer (palette/typography).

## Disposition of the old viewer

The current `src/visual/` is a HW-vs-EMU trace-comparison timeline (~2000 LOC):
`timeline.rs`, `alignment.rs`, `viewport.rs`, `data.rs`, `event_detail.rs`,
`tile_selector.rs`, `app.rs`. It is being retired (reported broken; wrong tool
for this direction). **Salvage `theme.rs`** (and glance at `tile_selector.rs`
for the click-select idiom); delete the rest. The CLI `--visual` path and
`main.rs` wiring get repointed to the new app.

## Success criteria (v1)

- `cargo run -- --visual <path.xclbin>` opens the app, loads the design, and
  renders the array from the device model (correct tile count/types/positions).
- Step/run/pause/reset visibly advance the emulator; the cycle counter and tile
  states change as it steps.
- Selecting any tile shows its live core status+PC, both DMA channels' state,
  64 lock values, memory size, and stream ports -- values that update as the
  emulator steps.
- No regression to the emulator: `cargo test --lib` stays green; the added DMA
  accessors and shared load helper have unit coverage; the suite still uses the
  same (now-shared) load path.

## Explicitly out of scope for v1

Overview pan/zoom; connection/routing lines; animation; prettiness/polish;
breakpoints; memory hex editor/editing; per-resource drill-down; multi-device
(NPU2) beyond what `new_npu1` gives; saving/loading debugger sessions.

## Open questions (for spec review)

1. **Detail-panel memory:** show just size in v1, or a small scrollable
   hex/word peek? (Leaning: size + a collapsed word peek, cheap via
   `read_data_u32`.)
2. **Run cadence:** fixed cycles-per-frame budget, or a target cycles/second
   with a slider? (Leaning: fixed budget in v1, slider later.)
3. **Load helper shape:** method on `InterpreterEngine` vs. free function in a
   new `src/loading/` module shared by suite + GUI? (Leaning: shared free
   function so the suite and GUI both call it without circular structure.)
