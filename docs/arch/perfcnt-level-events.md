# Performance Counter Level-Event Semantics

## Model

AIE2 perf counters configured with `XAIE_EVENT_ACTIVE_CORE` (event 0x1C)
or `XAIE_EVENT_TRUE` (event 0x01) as start_event are **duration counters**.
ACTIVE_CORE arms the counter on transition to Execute state, and the
counter ticks every cycle until a stop event (typically DISABLED_CORE)
fires. The counter is **not** gated per-cycle on Execute state once armed.

This matches `aie-rt`'s perfcnt programming model and was confirmed
empirically against real Phoenix hardware.

## EMU implementation

Per-cycle in the coordinator
(`src/interpreter/engine/coordinator.rs:1075-1170`):

1. Re-assert always-on level events at the top of each cycle:
   - `handle_event(TRUE_EVENT)` for every tile (always asserted).
   - `handle_event(ACTIVE_CORE_EVENT)` only when the core executed an
     instruction this cycle.
2. Call `tile.core_perf_counters.tick()` and `tile.mem_perf_counters.tick()`
   once. `tick()` advances every Active counter and returns the indices
   that crossed their threshold.
3. Each fired counter is fed back through `handle_event(PERF_CNT_N)` (so
   self-resetting configs can recycle) and notified to the owning module's
   trace unit, stamped with the core's pipeline-adjusted PC.

`handle_event()` is idempotent on level events: an already-Active counter
stays Active when the start event re-asserts; counters whose start event
doesn't match are unaffected. The pulse-vs-level distinction is handled
entirely in the coordinator's per-cycle re-assertion phase, not by
special-casing the `tick()` method.

## Ordering

Phase 3e (this perfcnt phase) runs *before* Phase 3f (commit_cycle) so
that perfcnt `notify_event()` calls accumulate into the same cycle's
`pending_slot_mask` as TRUE/edge-detector events. AM020 specifies one
trace frame per cycle: when multiple events fire in cycle N they must
be coalesced into a single Multiple frame, not split across two frames.

## History

An earlier audit (2026-04-XX) framed this as an overcounting bug --
"ACTIVE_CORE counters tick every cycle while stalled, even though
ACTIVE_CORE is a level signal that should only assert during Execute" --
and proposed splitting `tick()` into `tick_active_cycles()` /
`tick_idle_cycles()` driven by a coordinator-side gate ("Option 2" in
the original audit, since archived).

Subsequent investigation (#354 trail) reframed the problem: real-HW
ACTIVE_CORE counters are duration counters, not per-cycle-gated counters.
The shipped fix re-asserts level events each cycle in the coordinator
and lets `tick()` stay simple, which matches HW semantics without the
proposed plumbing.

## Out of scope

- Stall events (`ACTIVE_LOCK_STALL`, etc.) as start events -- not yet
  modeled. Would follow the same re-assertion pattern: query
  `Core::is_waiting_on_lock()` etc. and conditionally `handle_event` the
  corresponding event id each cycle.
- Memory-bank contention -- not modeled in EMU at all.
- DMA delay modeling -- orthogonal.
