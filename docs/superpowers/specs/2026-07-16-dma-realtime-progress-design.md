# DMA Real-Time Progress Visualization -- Design

**Date:** 2026-07-16
**Arc:** Visual debugger, architecture view (extends the tile floor-plan)
**Status:** Approved (design), pending implementation

## Problem

The floor-plan's DMA block lights up when a channel is active but does not show
it *working*: no transfer progress, no sense of a buffer descriptor advancing,
and -- most importantly -- no way to see a **stall as it happens** or *why*. DMA
waiting (on locks, on stream backpressure, on starvation) is where most of a
kernel's time goes, and today it is invisible.

## Goal

Show each DMA channel's operation in real time: a progress bar that fills as the
current buffer descriptor (BD) transfers, colored by the channel's live phase so
a stall is a frozen bar in a stall color the instant it occurs; a compact view
of the BD queue behind it; and a click-through per-channel deep-dive in the
detail panel.

## What the model already gives us (verified)

Progress is fully tracked and public -- the intra-BD bar is pure GUI:

- `DmaEngine::get_transfer(ch) -> Option<&Transfer>`; `Transfer::progress() -> f32`
  (0.0-1.0), `bytes_transferred`, `total_bytes`, `remaining_bytes()`,
  `address_gen.current()` (byte address), `remaining()`.
- `DmaEngine::channel_phase(ch) -> &'static str` -- the real 13-state FSM phase
  (Idle / BdSetup / AcquiringLock / MemoryLatency / HostPipelineLatency /
  BdSwitchBubble / Transferring / StartupHold / DrainingEgress / ReleasingLock /
  BdChaining / Paused / Error). `ChannelState` alone is lossy (everything but
  Idle/Paused/Error/WaitingForLock collapses to Active).
- `current_bd(ch)`, `queued_bd(ch)`, `task_queue_size(ch)`, `channel_stats(ch)`
  (`bytes_transferred`, `transfers_completed`, `lock_wait_cycles`, `cycles_spent`),
  `get_bd(bd) -> Option<&BdConfig>` (base_addr, length, d0-d3 dims/strides,
  iteration, acquire/release locks).

Two things are NOT exposed and are added here (both **read-only**, egui-free,
in the device crate, each with a test):

## Emulator additions (two typed read-only accessors)

1. **`DmaStall` enum** (device crate, egui-free), the distinct no-progress
   reasons worth a color:
   `LockWait`, `Backpressure` (MM2S downstream not accepting / draining egress),
   `Starved` (S2MM no upstream data), `Other` (bank-arbitration loss, cold-start
   throttle, setup/latency bubbles -- folded so the enum stays small).
2. **`DmaEngine::channel_stall_reason(ch) -> Option<DmaStall>`** -- `None` while
   actively progressing, idle, or cleanly transitioning; `Some(reason)` when the
   channel made no data progress this cycle. Derived internally from the FSM
   phase plus the existing edge flags (`prev_lock_stalled` at channel.rs:393,
   `prev_starving` at channel.rs:387): `AcquiringLock{acquired:false}` /
   WaitingForLock -> LockWait; `DrainingEgress` or MM2S egress stall ->
   Backpressure; `Transferring` with `prev_starving` -> Starved; bank-denied /
   cold-start / latency bubbles -> Other.
3. **`DmaEngine::queued_bd_ids(ch) -> Vec<u8>`** -- the task queue's BD ids in
   order, so the queue strip and the deep-dive can list actual BDs (today only
   the depth escapes via `task_queue_size`).

IMPLEMENTER NOTE: verify exact field/queue access against the real code before
deriving; the citations above are from a read-only survey and may drift. Add
unit tests: for `channel_stall_reason`, drive a channel into lock-wait and into
transferring and assert the reason (starvation/backpressure paths may only be
reachable via a scenario test -- cover what is feasible and note what is not).
For `queued_bd_ids`, enqueue BDs and assert ids in order.

## View-model (model.rs)

**Enrich `ChannelSnapshot`** (light, built every frame for the bars) with:
`progress: f32`, `phase: &'static str`, `stall: Option<DmaStall>`,
`queue_bds: Vec<u8>` (capped to a small display bound, e.g. first 8, with a
separate `queue_len` already present for the "+N" overflow). Keep existing
fields (`index`, `state`, `current_bd`, `queued_bd`, `queue_len`).

**New on-demand deep-dive**, built only for the selected channel (avoids putting
every BD's dimensions in every frame's snapshot):
`dma_channel_detail(engine, col, row, ch) -> Option<DmaChannelDetail>` returning
the current BD's full `BdConfig` (base addr, length, d0-d3 dims + strides,
iteration, acquire/release locks), the queued BD ids, live position
(bytes_transferred / total_bytes, current address, remaining), and stats
(lock_wait_cycles, cycles_spent, transfers_completed). Pure/egui-free.

## Bar anatomy (approved: progress bar + compact queue strip)

Per channel, one **clickable horizontal bar**, stacked vertically (a core's
channels stack as full-width bars; squat mem-tile blocks hold short stacked bars
-- the wordy detail lives in the panel, not the block, so no text-into-walls):

- **Progress fill** = `progress()`, drawn left-to-right over the bar background.
- **Fill/bar color = phase/stall** via the palette:
  - Transferring, progressing -> green (moving)
  - LockWait -> amber
  - Backpressure -> orange (route_stalled)
  - Starved -> a distinct stall color (new palette token)
  - ReleasingLock -> blue (handing off)
  - Idle -> grey; setup/latency/chaining -> pale (warming, not stalled)
  - Error -> red
  A stall is a *frozen* fill in a stall color -- the real-time stall signal.
- **Compact queue strip**: the current BD highlighted, the next few queued BDs
  as small cells, then "+N" when deeper than the display bound. On the tightest
  core bars this collapses to a "qN" count without dropping the progress bar.
- **Selection**: each bar is a click target; clicking sets the selected DMA
  channel (see wiring). The selected bar is outlined.

All colors are palette tokens in theme.rs (add `dma_starved`, and
`dma_backpressure` if reusing `route_stalled` reads wrong); no scattered
literals.

## Detail-panel deep-dive (in-panel, no floating window)

The detail panel's existing "DMA channels:" section becomes channel-aware:

- **No channel selected** -> the current all-channels summary (unchanged).
- **A channel selected** (via clicking its bar) -> the `DmaChannelDetail`
  deep-dive for that channel: current BD (id, base addr, length, d0-d3
  dims/strides, iteration, acquire/release locks), live position (bytes/total,
  address, remaining), the queued BD list, and the wait/cycle stats. A small
  "show all channels" affordance clears the selection.

This reuses the master/detail already in place (click a tile -> tile detail; now
click a channel -> channel detail). No floating-window machinery.

## Wiring

- `floorplan()` returns a small `FloorplanResponse { dma_channel_clicked:
  Option<u8> }` (it currently returns `()`); `paint_dma` interacts each bar rect
  and reports a click. detail.rs uses it to update selection.
- `DebuggerApp` gains `selected_dma: Option<u8>`; passed `&mut` into
  `detail::show`, which updates it from the floor-plan response and clears it on
  tile change (selecting a different tile resets the channel selection).
- Live-ness is free: the app already `request_repaint()`s while running, so
  reading `progress()` each frame animates the bar with no new machinery. This
  is live-value reading, NOT the deferred flowing-dot motion animation.

## Testing

- Emu: `channel_stall_reason` and `queued_bd_ids` unit tests (as above).
- View-model: `dma_channel_detail` returns the expected BD config / queue / stats
  for a constructed engine state; enriched `ChannelSnapshot` carries progress/
  phase/stall.
- GUI: `paint_dma` / `floorplan` render for a channel in each phase (idle,
  transferring, each stall) without panic via `__run_test_ui`; a synthesized
  click on a bar rect yields the right `dma_channel_clicked`.
- **Gates:** `cargo build` (default / `--features gui` / `--no-default-features`)
  and `cargo test --lib` all green. GUI never launched; `src/debugger` egui-free;
  the two new emu accessors keep the device crate egui-free.

## Non-Goals (deferred)

- Flowing-dot / pulse **motion** animation (the later animation arc). The bar
  reads a live value each frame; it does not animate travelling data.
- Channel-label annotations on the edge-port arrows; FIFO rendering. (Noted for
  a later slice.)
- Floating/tear-off DMA window (in-panel deep-dive is the decision for now).
- Per-dimension address-generator index vector (private; the flattened address
  + remaining is enough for v1).

## Human-Only Review (Maya)

Bar legibility on squat mem-tile blocks and stacked core bars; whether a stall
reads instantly (frozen + color); stall-color distinctness (lock vs backpressure
vs starved) across palettes; queue-strip vs "qN" collapse behavior; deep-dive
usefulness and whether the click-to-select flow feels right.
