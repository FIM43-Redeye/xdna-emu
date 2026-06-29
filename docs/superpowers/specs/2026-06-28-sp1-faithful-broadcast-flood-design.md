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

**Revised 2026-06-28 after design review.** The original §3.4 used a tick-driven
countdown; review proved it unfaithful in situ -- the timer-sync flood fires at
**config time** (`dispatch.rs:219`, register-write dispatch) where no `tick()`
runs, and clock-gated modules never tick at all (`coordinator.rs:1580/1622`), so
a countdown cannot drain before execution and would smear the skew across the
traced window. §3.4 now establishes the skew as a constant baseline at flood
time via a reset-*target* latch. Other review fixes folded in below (intra-tile
sign disentangle, 3-site de-dup, reset-clears-target, test re-scoping).

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

**Factor the wavefront to return a per-tile `origin_D` map** rather than burying
it in the delivery loop. SP-1 consumes it immediately (to compute `max_delay`
and the per-module targets), but SP-2 (trace Start-frame origins) and SP-4b (the
skew solver, which the arc says reads `origin_D` directly) reuse it without
recomputation. The reached-set is identical to today's flood at `d=0` because
`visited` is marked on enqueue -- a connectivity property independent of
LIFO-vs-Dijkstra order (review-confirmed).

### 3.2 Per-module delay = `origin_D` + intra-tile offset

Within one tile the core and memory modules receive the broadcast through
different input-pipeline depths, so each module type carries an additive offset
on top of its tile's `origin_D`:

```
delay(module) = origin_D(tile)  +  intra_tile_offset(module_type)
```

`intra_tile_offset` defaults to 0 for every module type.

**Disentangle the silicon signature (review fix).** Of the add_one terms
(`core-memtile=+2`, `memmod-memtile=+4`, `core-memmod=-2`), only `core-memmod`
is a true *intra-tile* comparison (same physical tile, shared `origin_D`); the
other two compare a compute tile against a *different* mem tile and are carried
by `origin_D` (the wavefront), not the intra-tile offset. So the only intra-tile
constraint is `mem_offset - core_offset = +2` -> `core_offset = 0`,
`mem_offset = 2`. SP-5 must set the intra-tile offsets from the `-2` difference
alone and **not** re-add `+2/+4` on top of an `origin_D` that already covers the
compute<->memtile hops, or it double-counts. SP-1 only plumbs the fields (all 0).

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

**`u8` is correct, not `i8` (review-confirmed).** `delay = origin_D + offset`
must be non-negative (it feeds a timer-baseline target); `origin_D >= 0`
(Dijkstra, non-negative weights) and `offset >= 0` (u8) guarantee that. The
invariant: *the earliest-resetting module type is the zero baseline; all offsets
are non-negative cycles relative to it.* `model_builder` should assert it. If
future silicon ever shows a module earlier than the chosen baseline, **rebaseline**
-- never switch to `i8` (a negative delay is meaningless for the mechanism).

### 3.4 Baseline-offset timer reset (the mechanism that consumes the delay)

The skew must be established as a **constant baseline at flood (config) time**,
not drained over execution ticks. Why: the flood fires from register-write
dispatch (`dispatch.rs:219`) during config, and `tick()` runs only in the
execution loop (`coordinator.rs:1581/1623`) and is skipped for clock-gated
modules. So between the flood and the first executed cycle there are **zero
ticks** -- a tick-driven countdown could not drain, and would surface the skew as
a staggered transient inside the traced window (the opposite of HW, where all
resets complete during config and the skew is a constant baseline from execution
cycle 0).

**Mechanism: extend the existing `pending_reset` latch into a reset-*target*
latch.** Today `pending_reset: bool` is latched at flood time and consumed by the
first `tick()`, which sets `value = 0`. SP-1 generalizes it to carry a target
value (default 0):

- The flood, after computing the wavefront, finds `max_delay` over all reached
  modules and latches each matching module's target = `max_delay - delay(module)`.
- `tick()` consumes the latch and sets `value = target` (instead of always 0),
  then increments normally on subsequent ticks -- exactly the current control
  flow, only the reset value changes.

This is faithful and clean:

- **Constant baseline, no transient.** The latch persists across the
  config->execution boundary and is consumed on each module's first execution
  tick (cycle 0), which runs *before* that cycle's trace notifies
  (`coordinator.rs:1581` precedes `:1616`). So every clocked module holds its
  baseline `max_delay - delay` from execution cycle 0 -- a module that reset
  earlier (smaller `delay`) reads higher, exactly the HW skew. The
  flood-before-first-event invariant (3.6.ii) is satisfied *structurally*, not by
  assertion.
- **Byte-neutral at `d=0`.** `max_delay = 0`, every `delay = 0`, every
  `target = 0` -> the latch sets `value = 0` on the first tick, identical to
  today. The existing `sync_timer_protocol_aligns_independent_timers` test and
  every non-flood reset path are unchanged.
- **No worse on clock-gating.** Consumption is still one tick (at un-gate for a
  gated module), exactly the current latch's behavior -- not the multi-tick drain
  the countdown would have needed. Making the gated-during-config edge case
  (rare: config runs with clocks active) faithful is a pre-existing nuance, out
  of SP-1 scope; SP-1 must only not regress it.

`max_delay` is the relative-correct, minimal non-negative baseline (latest-reset
module reads 0); the *absolute* origin is SP-2's job, so any `K >= max_delay`
would do and SP-1 picks `max_delay`.

**Interface:** the flood computes the per-module `delay` from the wavefront
(3.1) and `max_delay`, then latches the target through the broadcast-delivery
path (`notify_*_trace_event` -> `timer.notify_event`), which gains a target
argument defaulting to 0 so every non-flood call site stays behavior-neutral.
`reset()` and the control-bit-31 path must also clear the new target state
(parallel to how they clear `pending_reset` today; see 3.7).

### 3.5 Broadcast event-ID de-dup (defer the parser)

The authoritative AIE2/Phoenix source for the IDs is the aie-rt header
`xaie_events_aieml.h` (`XAIEML_EVENTS_CORE_BROADCAST_0 = 107` .. `_15 = 122`;
MEM same; `PL_BROADCAST_A_0 = 110`; `MEM_TILE_BROADCAST_0 = 142`). The AM025
register JSON has the broadcast register *structure* but **not** the ID values,
so the arc's "derive from regdb" was inaccurate.

The current values are correct; the risk is the duplication. **SP-1 de-dups:**
`effects.rs` consumes the single `events/mod.rs::broadcast_event_base` accessor
(which keeps the aie-rt reference comments) and its local
`CORE_/SHIM_PL_/MEMTILE_BROADCAST_BASE` consts are removed. The bases are used in
**three** sites in `effects.rs` (the core/mem/memtile hw_id around :517-519 and
the `SHIM_PL_BROADCAST_BASE` L1 tap around :540) -- all three move to the
accessor, via a `TileKind -> EventModuleType` mapping (Compute -> {Core, Memory},
Mem -> MemTile, Shim -> Pl) with a width cast if `EventId` is wider than `u8`.
**Deferred** (noted follow-up, not SP-1): full build-time derivation of the bases
from `xaie_events_aieml.h`. Rationale: event IDs for a fixed architecture do not
evolve, the values are already correct, and a C-header build parser is real
complexity for no current correctness gain (YAGNI).

### 3.6 Route-1 Component-3 correctness hazards

The reset-target mechanism (3.4) dissolves most of these, since it preserves the
existing latch's control flow rather than introducing a countdown:

- **(i) Clock-gated module reset -- preserved, not newly solved.** The
  reset-target latch is consumed by the first `tick()` exactly like today's
  `pending_reset`; a clock-gated module's latch simply waits for un-gate, the
  current shipped behavior. SP-1 introduces no new dependence on ticking beyond
  what already exists, so it does not regress clock-gating. (The countdown would
  have needed `D` ticks to drain and *would* have corrupted under gating -- that
  is why 3.4 was revised.) Making gated-during-config faithful is a pre-existing
  nuance, out of SP-1 scope.
- **(ii) Flood completes before the first in-window traced event -- satisfied
  structurally.** The latch persists across config->execution and is consumed on
  the first execution tick, which runs before that cycle's trace notifies
  (`coordinator.rs:1581` precedes `:1616`). So the baseline is in place before any
  traced event fires -- no separate assertion or scheduler needed.
- **(iii) Shim self-reset -- already satisfied; must not regress.** The flood
  seeds the frontier with the source `(col, source_row)` and visits it first
  (`effects.rs:502-521`), so the source's own timer already gets the broadcast
  (with `delay = 0` -> `origin_D = 0`). SP-1 only must keep the source at
  `delay 0` and not drop its self-delivery.

### 3.7 Reset paths must clear the new target state

`reset()` (`timer.rs:233`) and the control-bit-31 write path (`timer.rs:287`)
clear `pending_reset` today; they must also clear the new reset-target latch, or
a stale target could set the timer to a nonzero baseline after an explicit reset.
Mechanical, but a real omission if missed.

---

## 4. Data flow (end to end)

```
flood fires at config (dispatch.rs:219), source (src_col, src_row)
  phase 1: Dijkstra wavefront over broadcast adjacency (block masks honored)
       -> origin_D map for every reached tile             [3.1]
       -> per module: delay = origin_D + intra_tile_offset[3.2, consts from 3.3]
       -> max_delay = max(delay) over reached modules     [3.4]
  phase 2: for each reached module whose reset_event matches the broadcast id
       (id from the de-duped accessor)                    [3.5]
       -> latch reset-target = max_delay - delay          [3.4]
  ...config completes (no ticks)...
execution cycle 0: first tick() consumes the latch
       -> value = max_delay - delay  (before trace notifies that cycle)
       -> baseline present from cycle 0; increments thereafter
```

With default-zero `BroadcastTiming`, every `delay = 0`, `max_delay = 0`, every
target `= 0`, and the first tick sets `value = 0` -- byte-identical to today.

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
3. **Reset-target latch.** On `TileTimer`: a latched target `T` is applied by
   the first `tick()` (`value = T`, then increments); target `0` reproduces the
   current `pending_reset` semantics exactly (`value = 0` on first tick). Assert
   both.
4. **Behavior-neutral regression.** With default-zero `BroadcastTiming`, assert
   the flood is byte-identical to today: the **reached-set** is identical (not
   just the timing), and all reached timers read 0 at execution cycle 0. The
   existing `sync_timer_protocol_aligns_independent_timers` test must stay green
   unchanged.
5. **Constant baseline (the faithful-mechanism test).** Through the device-state
   flood path (not a bare `TileTimer`), with explicit nonzero constants: assert
   two modules with different `delay` hold a constant offset equal to their
   `delay` difference from execution cycle 0, with **no staggered transient**
   (the earlier-reset module reads higher). This is the test the original
   countdown design would have failed.
6. **Route-1 hazards.** (i) a module clock-gated through cycle K gets its target
   applied at its first tick (un-gate), matching the existing latch's behavior;
   (iii) the source/shim's own timer is delivered the broadcast (`delay 0`).
   ((ii) is structural -- no separate test; covered by test 5's "from cycle 0".)
7. **Reset clears target.** A latched target survives neither `reset()` nor a
   control-bit-31 write (parallel to `pending_reset`).
8. **Delivery coexistence.** A module receiving both a `delay=0` and a `delay>0`
   delivery (multi-channel / fixpoint re-entry) resolves to a single defined
   target -- assert the precedence the implementation picks.
9. **Event-ID de-dup.** Assert all three `effects.rs` sites and `events/mod.rs`
   resolve to the same per-module IDs (single source); the existing
   `broadcast_event_base` value test stays green.

Plus the full `cargo test --lib` and the trace sweep as the behavior-neutral
regression gate.

---

## 6. Open questions and risks

- **Delivery-coexistence precedence.** If a module receives both a `delay=0` and
  a `delay>0` target in one flood fixpoint, the precedence (last-wins / min-delay
  / max-delay) is undefined; the plan picks one and test 8 locks it. Likely
  min-delay (earliest arrival wins, consistent with the OR-tree), but confirm.
- **Flood source identity.** Confirm the flood origin in `propagate_broadcasts`
  is the shim/sync generator for the timer-sync case so `origin_D = 0` there
  (ties to hazard iii).
- **Gated-during-config edge case (pre-existing).** A module clock-gated across
  the entire config flood gets its target applied only at un-gate. Faithful or
  not, this is the *current* latch's behavior and out of SP-1 scope; flagged so
  SP-5/SP-4 know it exists.
- **Regression gate diffs `trace.log`, not pass/fail.** Per
  `feedback_verify_against_baseline_trace`, the bridge "CLEAN" verdict can miss
  level-event duration regressions; the behavior-neutral gate diffs the full
  `trace.log` against a pre-change baseline. The off-by-one that motivated this
  is closed by the latch-target reproducing `value=0` at cycle 0, but the diff
  stays the gate.

---

## 7. Next step

`writing-plans` -> a TDD implementation plan for SP-1 (RED tests first per
component, behavior-neutral regression as the standing gate).
