# Pass-2 deep dive: multi-tile timer sync gap

**Status**: gap fixed 2026-05-04. See "Implementation" section below for
the actual landed shape; a bridge re-run is queued to confirm mode-2
divergence drops.
**Authoritative source**: aie-rt `driver/src/timer/xaie_timer.c::XAie_SyncTimer`.
**Likely impact**: prime suspect for the suite-wide mode-2 anchor / PC-sequence
drift seen in 2026-05-04 bridge runs (#318/#321 family).

## How the hardware does it

`XAie_SyncTimer(BcastChannelId)` aligns all per-tile timers to cycle 0
in the same hardware cycle by exploiting the broadcast event network:

1. Enumerate all ungated tiles in the partition.
2. Configure broadcast-channel `BcastChannelId` so an event fired at
   shim col=0 propagates to every tile.
3. On every tile, write `Timer_Control.Reset_Event = <broadcast event ID>`.
   This arms each tile's timer to auto-reset to 0 the moment that event
   reaches it.
4. Generate the broadcast event once at shim col=0.
5. Broadcast propagates outward; every tile's timer resets in the same
   hardware cycle (modulo broadcast-network propagation latency).
6. Clear configuration.

The mlir-aie / aie-rt host layer typically calls `XAie_SyncTimer`
right before any trace-controller-driven workload, so the trace cycle
counts on every tile share a common t=0.

## What we model

`src/device/timer.rs::TileTimer` is per-module:

- 64-bit value, register read/write, threshold trigger ✓
- `Timer_Control.Reset_Event` field is *stored* (`reset_event()` getter) ✓
- `reset()` method exists ✓
- Direct register-write `Reset` bit (bit 31) → `reset()` ✓

Per-cycle ticking (`coordinator.rs:1079-1080`):

```rust
tile.core_timer.tick();
tile.mem_timer.tick();
```

Only `core_timer` and `mem_timer` advance. Memtile / Shim-PL timer
fields are not present on the `Tile` struct (`tile/mod.rs:209-211`):

```rust
pub core_timer: super::timer::TileTimer,
pub mem_timer: super::timer::TileTimer,
```

## What's missing

1. **No event-driven reset path.** When the event subsystem fires an
   event whose ID matches a tile's `core_timer.reset_event()` or
   `mem_timer.reset_event()`, nothing resets the timer. The
   `XAie_SyncTimer` protocol therefore has zero effect in the
   emulator: all timers keep running freely from their independent
   start cycles.
2. **No memtile / shim-PL timer modules** as distinct fields. Per
   AM025: compute tile has two timer modules (core + memory),
   memtile has one (mem-tile module), shim has one (PL module).
   We collapse memtile + shim onto the existing `mem_timer`/`core_timer`
   fields. Probably OK in practice, but worth re-checking once the
   reset wiring lands so cross-tile-type sync still works.
3. **No broadcast-propagation latency model.** Real HW takes a small
   number of cycles for the broadcast wave to reach far tiles. We
   would reset all timers in the same emulator cycle; HW spreads the
   reset by ~1 cycle per hop. Cycle-accurate to within ±N cycles
   where N = max_distance_to_shim — small but observable.

## Concrete fix sketch

**Where to plug in**: per-cycle event-processing phase in `coordinator.rs`,
after `generate_event` calls but before `commit_cycle`.

```rust
// Per-tile, per-cycle: any pending event matching a configured
// timer reset event resets that timer to zero.
let core_rst = tile.core_timer.reset_event();
let mem_rst  = tile.mem_timer.reset_event();
if core_rst != 0 && tile.core_events.fired_this_cycle(core_rst) {
    tile.core_timer.reset();
}
if mem_rst != 0 && tile.mem_events.fired_this_cycle(mem_rst) {
    tile.mem_timer.reset();
}
```

`fired_this_cycle` may need a small accessor if not present.

**Test plan**:
1. Unit test: configure a tile's `Timer_Control.Reset_Event = E`,
   advance N cycles, fire event E, assert timer is 0.
2. Integration test: replay the `XAie_SyncTimer` sequence (set up
   broadcast, configure all tiles' Reset_Event, fire shim event,
   assert all tile timers are 0 after one event-propagation cycle).
3. Bridge regression: re-run mode-2 sweep on add_one_using_dma
   (or any test that fails today). PC-sequence count divergence
   should drop.

**Estimated effort**: ~30-50 LOC + 2-3 unit tests. Should be a
half-day, not a multi-day investigation.

## Implementation (landed 2026-05-04)

`src/device/timer.rs`:

- `TileTimer` gains a `pending_reset: bool` latch.
- New `notify_event(event_id: u8)` method: if `event_id` is non-zero
  and matches `reset_event()`, sets `pending_reset = true`.
- `tick()` checks `pending_reset`; if set, value is forced to 0 (no
  increment) and the latch clears. Otherwise normal increment. The
  latch ordering means producer/consumer order within a cycle does
  not matter — final post-tick value is 0 either way.
- Explicit `reset()` also clears `pending_reset` so a stale latch
  cannot carry into the next cycle.

`src/device/tile/mod.rs`:

- `notify_core_trace_event` calls `core_timer.notify_event(hw_id)`.
- `notify_mem_trace_event` calls `mem_timer.notify_event(hw_id)`.

These two functions are already on the per-event-fire path that
feeds the trace unit and edge detectors, so every fired hardware
event passes through them — no new instrumentation needed.

Tests added:

- `notify_event_no_reset_event_configured` — Reset_Event=0 means
  "disabled"; no notify latches.
- `notify_event_id_zero_never_latches` — event id 0 is the
  unconfigured sentinel.
- `notify_matching_event_latches_pending_reset`.
- `notify_nonmatching_event_does_not_latch`.
- `pending_reset_cleared_by_tick_and_resets_value` — the per-cycle
  semantics: value goes to 0, latch clears, subsequent ticks
  count up from 1.
- `explicit_reset_clears_pending_flag` — register-driven Reset bit
  cleans up a stale latch.
- `sync_timer_protocol_aligns_independent_timers` — replays the
  `XAie_SyncTimer` shape: three timers diverged by orders of
  magnitude, all reset to 0 in the same cycle when the configured
  broadcast event fires on each.

Lib test count went from 2856 → 2863 (the seven new tests above);
no regressions.

## Open follow-ups

1. Memtile-only and shim-only tile types reuse `mem_timer` and
   `core_timer` respectively. The fix works because the right field
   is touched on each tile type, but the naming is misleading.
   Cosmetic; tracked as an architecture-index verification.
2. Broadcast-propagation latency model. We currently reset all
   tiles' timers in the same emulator cycle if they all see the
   broadcast event simultaneously. Real HW has ~1 cycle per hop
   propagation. Affects cycle-accuracy by < N cycles where N =
   max distance to shim. Not blocking mode-2 unblock.

## Why we missed this earlier

The timer module landed cleanly with full register-level coverage and
extensive tests. The reset-event plumbing was *almost* there
(field stored, `reset_event()` getter exposed), but the consumer side
— the per-cycle event-fire scan that checks against the stored event
ID — was never wired. This is the same shape as the missing-control-
packets miss: a subsystem looked complete on its own surface but
lacked one connection to the rest of the model.

Architecturally, this argues for a **subsystem-integration check** as
part of the coverage index: "this subsystem has registers; what
*reads* the configured fields per cycle?" If nothing does, the
subsystem is functionally a no-op.
