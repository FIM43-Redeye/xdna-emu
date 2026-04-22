# Performance Counter Level-Event Semantics Audit

## Problem

HW AIE2 perf counters configured with `XAIE_EVENT_ACTIVE_CORE` as start
event tick only during cycles when the core is in Execute state. This is
a *level* signal, not a pulse — the counter advances while the signal is
asserted, not on signal transitions.

EMU's current `PerfCounterBank` (`src/device/perf_counters/mod.rs:326-383`)
implements a pulse model: `handle_event(start)` transitions the counter
to `Active`, and `tick()` (called unconditionally every cycle from
`coordinator.rs:964`) increments every `Active` counter regardless of
whether the core is still active.

Consequence: once started, an ACTIVE_CORE-configured EMU counter ticks
every cycle until stopped, even when the core is stalled on lock-wait,
cascade-wait, or disabled. The counter overcounts.

## Which events are affected

Level-valued events reachable from EMU state, ordered by how close their
semantics are to "tick only while X":

| Event | Level semantic | EMU state source |
|-------|----------------|------------------|
| `ACTIVE_CORE` | Core in Execute state | `Core::is_running()` / `!is_blocked()` |
| `ACTIVE_MEMORY_STALL` | Core blocked on memory-bank contention | Not modeled in EMU (no bank contention) |
| `ACTIVE_LOCK_STALL` | Core blocked on lock acquire | `Core::is_waiting_on_lock()` |
| `ACTIVE_CASCADE_STALL` | Core blocked on cascade full/empty | `Core::is_waiting_on_cascade()` |
| `DISABLED_CORE` | Core is disabled | `!Core::is_enabled()` |

`ACTIVE_MEMORY_STALL` has no EMU backing — skip. The others map to
existing blocked-state checks.

## Pulse vs level today

`handle_event()` is called for pulse events — `LOCK_ACQUIRE`,
`PORT_RUNNING`, `BRANCH_TAKEN`, etc. — and correctly transitions
`Idle`/`Stopped` <-> `Active`. That path stays as-is for pulse events.

The fix targets only the `tick()` path: rather than ticking every
`Active` counter unconditionally, a counter with a level-valued
start_event should tick only when that level is asserted *this cycle*.

## Fix: Option 2 (selected)

Decision (from spec open questions): **Option 2 — move the tick gate
into the caller.** Instead of changing `tick()`'s signature, the
coordinator checks core state before calling `tick()`:

- Core is in Execute state → `core_perf_counters.tick_active_cycles()`
- Core is blocked/disabled → `core_perf_counters.tick_idle_cycles()`

`tick_active_cycles()` is the current behavior: increment all Active
counters, return threshold fires.

`tick_idle_cycles()` is new: increment only counters whose `start_event`
is NOT a level-valued event (i.e., preserves pulse-counter behavior for
non-ACTIVE_CORE counters that happen to be running on the core module),
but does not increment counters started by `ACTIVE_CORE`. Threshold
checks still apply.

This is the narrowest possible change: zero diff to `handle_event()`,
zero diff to pulse-event counters, no new plumbing.

## Out-of-scope for this work

- Stall events (`ACTIVE_LOCK_STALL`, etc.) as start events. Spec says:
  "If fixing these falls out of the same refactor (same plumbing, same
  predicate), include them. If any needs significant separate plumbing,
  defer to a follow-up." For Option 2, adding stall-event tick gating
  is symmetric work: the coordinator would need to query `is_waiting_on_lock()`
  etc. and conditionally tick a different subset. Leave as a follow-up.
- DMA delay modeling — orthogonal.
- NoC latency — orthogonal.
