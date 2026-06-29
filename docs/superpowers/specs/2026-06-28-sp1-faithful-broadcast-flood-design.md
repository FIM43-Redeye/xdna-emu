# SP-1: Faithful Broadcast Flood (emulator core)

**Design spec.** 2026-06-28. First sub-project of the
[timer-sync faithful-broadcast arc](2026-06-28-timer-sync-faithful-broadcast-arc.md).

SP-1 makes the emulator's BROADCAST_15 timer-reset flood model real per-hop
propagation: each tile's timer resets at `origin_D = n_h*d_h + n_v*d_v` from the
flood source (plus an intra-tile core/mem offset), instead of the current
all-tiles-same-cycle flood. It is the foundation the rest of the arc builds on:
no hardware, no validation kernel, unit-testable in isolation.

This spec is implementable as written; the exact per-tick edge semantics and
test bodies are pinned in the implementation plan (`writing-plans` next).

---

## 1. Scope and the governing constraint

**In scope:** the flood mechanism (Dijkstra wavefront computing `origin_D`), the
deferred timer reset that consumes it, the `BroadcastTiming` constants home in
archspec, the intra-tile core/mem asymmetry, the broadcast event-ID de-dup, and
the three route-1 Component-3 correctness hazards. Pure unit tests.

**Out of scope (later SPs):** wiring `origin_D` into trace Start-frame origins
(SP-2); the validation kernel (SP-3); the cross-domain gate and inference export
(SP-4); measuring the real latency constants on silicon (SP-5); full build-time
derivation of broadcast event IDs from the aie-rt header (deferred hygiene
follow-up, see 3.5).

**Governing constraint -- behavior-neutral on ship (Maya, 2026-06-28).** The
`BroadcastTiming` constants default to **zero** (`d_h = d_v = 0`, intra-tile
offsets `= 0`). With those defaults the flood is byte-identical to today's
same-cycle behavior: the full existing trace sweep stays a clean regression
gate, and the mechanism is exercised only by unit tests that pass explicit
nonzero constants. Real values arrive in SP-5; this is the more principled
choice under "derive from the toolchain" -- a measured zero placeholder beats an
unvalidated guess shipped as default.

---

## 2. Current state (what SP-1 changes)

- **`src/device/state/effects.rs::propagate_broadcasts`** floods with **zero**
  per-hop delay -- every reached tile is notified at `current_cycle` (around
  effects.rs:521-522) via a **LIFO** frontier (`Vec` + `.pop()`, around
  effects.rs:502/592; an in-code comment mislabels it "BFS" -- fix while here).
  Per-module broadcast event IDs come from hardcoded bases
  (`CORE_BROADCAST_BASE=107`, `SHIM_PL_BROADCAST_BASE=110`,
  `MEMTILE_BROADCAST_BASE=142`; effects.rs:480-482), `id = base + channel`.
- **`src/device/events/mod.rs::broadcast_event_base`** holds a *second* copy of
  the same bases (107/107/110/142; events/mod.rs:119-122) with the aie-rt
  reference comments. The duplication is the real bug-class risk.
- **`src/device/timer.rs::TileTimer`** is a bare `value: u64` counter:
  `tick()` increments, or zeroes when `pending_reset` is latched;
  `notify_event(id)` latches `pending_reset` if `id == reset_event()`. There is
  **no wall-clock awareness and no reset delay** -- the reset is immediate on the
  next tick. The flood reaches it through `tile/mod.rs` (`notify_*_trace_event ->
  core_timer.notify_event`, around tile/mod.rs:807-811/862-866).

---

## 3. Components

### 3.1 Broadcast wavefront -- Dijkstra `origin_D`

Replace the LIFO frontier with a **min-cost wavefront** (Dijkstra over the
broadcast adjacency). From the flood source `(src_col, src_row)`, each reached
module's `origin_D` is the minimum cumulative cost over all paths, edge cost =
`d_v` for a vertical (north/south) hop and `d_h` for a horizontal (east/west)
hop. The AIE-ML broadcast OR-tree (AM020 Ch.2) means an event arriving from *any*
direction re-broadcasts onward, so the earliest arrival wins -- which is exactly
shortest-path. Dijkstra (not plain BFS) is required because `d_h != d_v` makes
hop-count and latency diverge.

The wavefront must honor the existing per-direction block masks (the adjacency
the current flood already walks); for the timer-sync flood these are unblocked
(reset defaults), so `origin_D` reduces to weighted-Manhattan
`|row-src_row|*d_v + |col-src_col|*d_h`, but the implementation stays general.

With the default `d_h = d_v = 0`, every `origin_D = 0` -- identical to today.

### 3.2 Per-module delay = `origin_D` + intra-tile offset

Within one tile the core and memory modules receive the broadcast through
different input-pipeline depths, so each module type carries an additive offset
on top of its tile's `origin_D`:

```
delay(module) = origin_D(tile)  +  intra_tile_offset(module_type)
```

`intra_tile_offset` defaults to 0 for every module type. The add_one silicon
signature (`core-memtile=+2`, `memmod-memtile=+4`, `core-memmod=-2`) is the
SP-5 target for these offsets; SP-1 only plumbs them.

### 3.3 `BroadcastTiming` -- the constants home (archspec)

A new struct in `crates/xdna-archspec`, per-device, mirroring the
`inter_tile_hop_latency` crossing-fix precedent (`StreamSwitchTiming`) but as its
own type because the event-broadcast network is a distinct fabric from the
stream switch:

```
struct BroadcastTiming {
    per_hop_horizontal: u8,        // d_h
    per_hop_vertical:   u8,        // d_v
    intra_tile_core_offset: u8,    // core module of a compute tile
    intra_tile_mem_offset:  u8,    // memory module of a compute tile
    // memtile and shim are single-module tile types: no core/mem intra-tile
    // split, so they carry no intra-tile offset in SP-1. A per-tile-type
    // baseline for them, if silicon shows one, is an SP-5 refinement.
}
```

All fields default to 0. Plumbed through `types.rs` (struct),
`model_builder.rs` (per-device values, NPU1 = all-zero placeholder), and
`build.rs` (const emission), consumed by `propagate_broadcasts`. The
placeholder-zero values carry a doc comment naming SP-5 as the calibration
source and pointing at the add_one signature.

### 3.4 Deferred timer reset (the mechanism that consumes the delay)

`TileTimer` gains a **local deferred-reset countdown** -- not a global scheduler.
When the flood delivers the broadcast to a module with `delay = D`:

- `D == 0` -> latch `pending_reset` exactly as today (reset on next tick).
- `D > 0` -> latch a countdown `D`. `tick()` decrements it while letting the
  timer keep free-running its **old** value (faithful: the distant tile's timer
  has not yet received the reset), and zeroes the timer on the tick the
  countdown expires (timer reads 0 at `flood_cycle + D`).

The sub-`D` window where the timer still shows its old free-running value is
**unobservable** because of the flood-before-first-event invariant (3.6.ii);
that is what lets us avoid a global event scheduler. This is the "baseline
shifts by `origin_D`" model: post-reset, two modules' timers differ by exactly
their `delay` difference -- the modeled skew.

**Interface:** the delay must travel from `propagate_broadcasts` (which knows
`origin_D`) to the timer. The broadcast-delivery path
(`notify_*_trace_event` -> `timer.notify_event`) gains a delay argument that
defaults to 0, so every non-flood call site and the existing
`sync_timer_protocol_aligns_independent_timers` test stay behavior-neutral.

### 3.5 Broadcast event-ID de-dup (defer the parser)

The authoritative AIE2/Phoenix source for the IDs is the aie-rt header
`xaie_events_aieml.h` (`XAIEML_EVENTS_CORE_BROADCAST_0 = 107` .. `_15 = 122`;
MEM same; `PL_BROADCAST_A_0 = 110`; `MEM_TILE_BROADCAST_0 = 142`). The AM025
register JSON has the broadcast register *structure* but **not** the ID values,
so the arc's "derive from regdb" was inaccurate.

The current values are correct; the risk is the duplication. **SP-1 de-dups:**
`effects.rs` consumes the single `events/mod.rs::broadcast_event_base` accessor
(which keeps the aie-rt reference comments) and its local
`CORE_/SHIM_PL_/MEMTILE_BROADCAST_BASE` consts are removed. **Deferred** (noted
follow-up, not SP-1): full build-time derivation of the bases from
`xaie_events_aieml.h`. Rationale: event IDs for a fixed architecture do not
evolve, the values are already correct, and a C-header build parser is real
complexity for no current correctness gain (YAGNI).

### 3.6 Route-1 Component-3 correctness hazards

All three are code-confirmed live against the wired flood->timer path
(tile/mod.rs:807-811/862-866) and must be handled in SP-1:

- **(i) Clock-gated module reset.** The deferred-reset latch/countdown must apply
  even to a module that is clock-gated and therefore not `tick()`-ed -- otherwise
  `origin_D` corrupts (the reset is lost or mistimed). The reset bookkeeping
  cannot depend on the module being actively ticked.
- **(ii) Flood completes before the first in-window traced event.** The flood
  (and all deferred resets it schedules) must resolve before any traced event
  fires, mirroring HW doing timer-sync during config. This invariant is what
  makes the 3.4 sub-`D` transient unobservable; SP-1 must assert/guarantee it,
  not assume it.
- **(iii) Shim self-reset.** The shim that *generates* the sync broadcast must
  reset its own timer when it fires the event (it is the flood origin, `origin_D
  = 0`), not only the downstream tiles.

---

## 4. Data flow (end to end)

```
flood fires at (src_col, src_row), cycle C
  -> Dijkstra wavefront over broadcast adjacency (block masks honored)
       -> origin_D(tile) for every reached tile          [3.1]
  -> per module: delay = origin_D + intra_tile_offset    [3.2, consts from 3.3]
  -> deliver broadcast event id (from de-duped accessor) [3.5]
       with delay to each module's timer                 [3.4 interface]
  -> TileTimer: D==0 immediate reset; D>0 countdown,
       timer reads 0 at C + delay                        [3.4]
```

With default-zero `BroadcastTiming`, every `delay = 0` and this collapses to the
current behavior exactly.

---

## 5. Testing strategy

Pure unit tests, no hardware:

1. **Wavefront / `origin_D`.** With explicit `d_h != d_v` (e.g. 2 and 3), assert
   a tile at `(Δcol, Δrow)` from the source gets `origin_D = Δcol*d_h +
   Δrow*d_v`; assert Dijkstra picks the min-cost path where two routes exist;
   assert block-masked directions are not traversed.
2. **Intra-tile offset.** With nonzero `intra_tile_core_offset` /
   `intra_tile_mem_offset`, assert core vs mem timers within one tile differ by
   the offset delta after the flood.
3. **Deferred reset.** With `delay = D > 0`, assert the timer reads 0 at
   `C + D` and free-runs its old value in `[C, C+D)`; with `delay = 0`, assert
   reset on the next tick (current semantics).
4. **Behavior-neutral regression.** With default-zero `BroadcastTiming`, assert
   the flood is byte-identical to today -- all reached timers reset on the same
   tick. The existing `sync_timer_protocol_aligns_independent_timers` test must
   stay green unchanged.
5. **Route-1 hazards.** (i) a clock-gated (un-ticked) module still resets
   correctly; (ii) an assertion/guard that the flood resolves before the first
   traced event; (iii) the shim's own timer resets when it generates the event.
6. **Event-ID de-dup.** Assert `effects.rs` and `events/mod.rs` resolve to the
   same per-module IDs (single source); the existing `broadcast_event_base`
   value test stays green.

Plus the full `cargo test --lib` and the trace sweep as the behavior-neutral
regression gate.

---

## 6. Open questions and risks

- **Per-tick edge of the countdown.** The exact tick on which a `D>0` countdown
  zeroes the timer (and whether the expiry tick increments-then-zeroes or just
  zeroes) is pinned in the plan via TDD; the spec fixes only the observable
  contract (reads 0 at `C+D`).
- **Flood source identity.** Confirm the flood origin in `propagate_broadcasts`
  is the shim/sync generator for the timer-sync case so `origin_D = 0` there
  (ties to hazard iii).
- **Clock-gating representation.** Hazard (i) depends on how clock-gated modules
  are modeled in the tick path; the plan must locate that to guarantee the latch
  survives un-ticked cycles.

---

## 7. Next step

`writing-plans` -> a TDD implementation plan for SP-1 (RED tests first per
component, behavior-neutral regression as the standing gate).
