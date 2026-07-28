# DMA Real-Time Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live, clickable per-channel DMA progress bars and an on-demand channel deep-dive to the existing egui debugger without changing DMA behavior.

**Architecture:** The device layer exposes two pure read-only projections: a typed stall reason and FIFO-ordered queued BD ids. The egui-free debugger model snapshots light per-frame bar data and builds full BD/position/stat detail only for the selected channel; the GUI paints and wires that data through the existing floor-plan and detail panel.

**Tech Stack:** Rust, existing DMA engine/model types, egui/eframe 0.31, built-in unit tests.

## Global Constraints

- Do not launch the GUI; compile it only with `cargo build --features gui`.
- Keep `src/device` and `src/debugger` egui-free and `cargo build --no-default-features` green.
- Add no dependency and no DMA behavior/timing mutation.
- Put every new color in `src/visual/theme.rs`; reuse existing palette tokens otherwise.
- Do not commit; the controller commits.
- Preserve the unrelated untracked `tools/experiments/producer_emu_gate.py`.
- Finish with exactly these four gates: `cargo build`, `cargo build --features gui`, `cargo build --no-default-features`, `cargo test --lib`.

---

### Task 1: Read-only DMA projections

**Files:**
- Modify: `src/device/dma/engine/types.rs`
- Modify: `src/device/dma/token.rs`
- Modify: `src/device/dma/engine/task_queue_ops.rs`
- Test: `src/device/dma/engine/tests.rs`

**Interfaces:**
- Produces: `DmaStall::{LockWait, Backpressure, Starved, Other}`.
- Produces: `DmaEngine::channel_stall_reason(ChannelId) -> Option<DmaStall>`.
- Produces: `DmaEngine::queued_bd_ids(ChannelId) -> Vec<u8>`.

- [ ] **Step 1: Write failing engine tests.** Configure a lock-acquiring BD and step it into `AcquiringLock { acquired: false }`; assert `LockWait`. Step an unstalled MM2S BD into `Transferring`; assert `None`. Start one task and enqueue BDs 3 then 5 behind it; assert `[3, 5]`.
- [ ] **Step 2: Verify RED.** Run `cargo test --lib device::dma::engine::tests::dma_read_only -- --nocapture`; expect unresolved `DmaStall`, `channel_stall_reason`, and `queued_bd_ids` errors.
- [ ] **Step 3: Implement the minimum projections.** Add a crate-visible iterator over `TaskQueue` entries, then map FIFO entries to `start_bd`. Match the channel FSM: unacquired lock -> `LockWait`; `DrainingEgress` -> `Backpressure`; `Transferring + prev_starving` -> `Starved` for S2MM and `Backpressure` for MM2S, except the explicit shim cold-drain throttle -> `Other`; setup/latency/switch/chaining/startup bubbles and MM2S memory starvation -> `Other`; idle, paused, release, error, and progressing transfer -> `None`. Do not misuse `backoff_left`: bank denial has no persistent read-only latch, so report that limitation instead of emitting false telemetry.
- [ ] **Step 4: Verify GREEN.** Re-run the focused tests and expect pass.

### Task 2: Egui-free snapshots and selected-channel detail

**Files:**
- Modify: `src/debugger/model.rs`

**Interfaces:**
- Consumes: the three Task 1 device types/methods plus existing transfer, BD, and stats accessors.
- Produces: enriched `ChannelSnapshot { progress, phase, stall, queue_bds, .. }`.
- Produces: `DmaChannelDetail` and `dma_channel_detail(&InterpreterEngine, u8, u8, u8) -> Option<DmaChannelDetail>`.

- [ ] **Step 1: Write failing model tests.** Construct a real engine DMA channel with a configured current BD and two queued BDs. Assert the tile snapshot carries `progress`, `phase`, `stall`, and capped queue ids; assert the deep detail preserves the BD id/config, FIFO order, live transfer position, and stats.
- [ ] **Step 2: Verify RED.** Run `cargo test --lib debugger::model::tests::dma_ -- --nocapture`; expect missing fields/type/function errors.
- [ ] **Step 3: Implement the minimum model.** Read transfer progress once per channel and cap only `queue_bds` to eight; retain `queue_len` for overflow. Build detail only after validating tile/engine/channel, clone the current `BdConfig`, and copy live transfer/stat counters into plain fields.
- [ ] **Step 4: Verify GREEN.** Re-run the focused tests and expect pass.

### Task 3: Palette-backed clickable floor-plan bars

**Files:**
- Modify: `src/visual/theme.rs`
- Modify: `src/visual/floorplan.rs`

**Interfaces:**
- Consumes: enriched `ChannelSnapshot` and `DmaStall`.
- Produces: `FloorplanPresentation::selected_dma: Option<u8>`.
- Produces: `FloorplanResponse { dma_channel_clicked: Option<u8> }` from `floorplan`.

- [ ] **Step 1: Write failing GUI tests.** Cover phase/stall-to-palette mapping, render every required phase/stall in `__run_test_ui`, and feed pointer press/release events over a known bar to assert its channel id is returned.
- [ ] **Step 2: Verify RED.** Run `cargo test --lib visual::floorplan::tests::dma_ -- --nocapture`; expect missing response/palette/bar behavior.
- [ ] **Step 3: Implement the minimum painter.** Stack one full-width bar per channel; tint the background from the phase color, fill `progress` left-to-right, overlay a compact current/queued BD strip when space allows and `qN` when it does not, interact each bar with a stable id, and stroke the selected bar. Add only `dma_starved` to all three palettes; reuse `route_stalled` for backpressure.
- [ ] **Step 4: Verify GREEN.** Re-run focused GUI tests and expect pass.

### Task 4: Detail and app selection wiring

**Files:**
- Modify: `src/visual/detail.rs`
- Modify: `src/visual/app.rs`

**Interfaces:**
- Consumes: `FloorplanResponse` and `dma_channel_detail`.
- Produces: `detail::show(..., selected_dma: &mut Option<u8>)`.
- Produces: `DebuggerApp::selected_dma: Option<u8>` reset whenever `selected` changes.

- [ ] **Step 1: Write a failing selection-reset test.** Given a selected DMA channel, changing the selected tile clears it; retaining the tile preserves it.
- [ ] **Step 2: Verify RED.** Run the focused app test and expect the missing reset helper/field to fail.
- [ ] **Step 3: Implement the minimum wiring.** Apply a bar click immediately after floor-plan rendering. Preserve the existing all-channel text when no channel is selected; otherwise render current BD dimensions/iteration/locks, live position, queue ids, and cycle stats plus a small `show all channels` button. Compare tile selection before/after overview rendering and clear the channel on change.
- [ ] **Step 4: Verify GREEN.** Re-run the focused app/detail tests and expect pass.

### Task 5: Full verification and handoff

**Files:**
- Inspect only: all modified files and `git diff`.

- [ ] **Step 1: Format only touched Rust files.** Run `cargo fmt -- <explicit touched Rust paths>` or the repository formatter without changing unrelated files; inspect `git status` immediately afterward.
- [ ] **Step 2: Run the four fresh gates in the required order.** Run `cargo build`, `cargo build --features gui`, `cargo build --no-default-features`, and `cargo test --lib`; require exit code 0 for each.
- [ ] **Step 3: Inspect scope and mechanics.** Run `git diff --check`, `git diff --stat`, and `git status --short`; confirm no GUI launch, commit, dependency, timing mutation, or unrelated-file edit.
- [ ] **Step 4: Report contract coverage.** State exact stall derivation fields/phases, tested versus unexercised variants, any unavoidable derivation limitation, and the six requested human visual-review points.
