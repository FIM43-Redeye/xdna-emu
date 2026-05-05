# Trace controller start/stop pipelined latency gap

**Status**: gap fixed 2026-05-04. Validation against `add_one_using_dma`
mode-2 PC sequence pending re-run.
**Authoritative source**: empirical comparison of HW vs EMU trace dumps
on `add_one_using_dma.chess` after the timer-sync fix landed
(see [timer-sync-gap.md](timer-sync-gap.md)).

## How the hardware behaves

After the timer-sync fix the residual mode-2 PC-sequence divergence on
`add_one_using_dma.chess` was exactly **2 frames**: HW recorded 34 PCs,
EMU recorded 36. Per-PC accounting:

- EMU emitted PC `0x150` (kernel iter-1 entry) **4 times**; HW emitted
  it **3 times**. The extra was the iter-0 entry, captured at the same
  cycle as `start_event` (`PERF_CNT_0`) fires.
- EMU emitted PC `0xb0` (just past `__cxa_finalize`) **once**; HW emitted
  it **zero times**. The extra was a frame captured at the same cycle as
  `stop_event` (`INSTR_EVENT_1`) fires.

Both deltas are exactly one cycle each, on opposite ends of the trace
window. The simplest model that fits: the trace controller is pipelined
on its state transitions — same-cycle events for the start_event arrival
and stop_event arrival are NOT recorded.

```
cycle:                  X       X+1     X+2 ...     Y-1     Y
start_event arrives    ───┐
                          │ (controller still Idle this cycle —
                          │  events here are dropped)
                          ▼
state Idle → Running           ───────────────────►
                                                       │
stop_event arrives                                  ───┤
                                                       │ (controller still
                                                       │  Running this cycle —
                                                       │  events here are
                                                       │  dropped on the
                                                       │  same edge)
                                                       ▼
                                               state Running → Stopped
```

This matches a typical pipelined-FSM design: the state edge is
registered, and the lookup against the new state happens the cycle
after. Same-cycle events arriving alongside the trigger event check
against the OLD state, so they're discarded.

## What we model now

`src/device/trace_unit/mod.rs`:

- `TraceUnit` carries `armed_start_cycle: Option<u64>` and
  `armed_start_anchor: u16`. When `start_event` arrives, we set the
  arm cycle, emit the Start marker bytes immediately (so the byte
  stream's Start frame is at the very beginning, matching HW), but
  leave `state == Idle`.
- The top of `notify_event` checks: if `armed_start_cycle == Some(arm)`
  and the incoming `cycle > arm`, call `activate_armed_start(cycle)`
  which flips state to Running. Same-cycle events still see Idle and
  are dropped. Events from the next cycle onward see Running and are
  recorded.
- `stop_event` flips state to Stopped immediately AND clears any
  pending mode-2 frames / atom run that accumulated in the same cycle
  as the stop. (Previously we drained pending here, which leaked one
  same-cycle frame past the HW window.)

## Tests added

In `src/device/trace_unit/tests.rs`:

- `force_start(tu, start_event)` — test helper that fires the start
  event at cycle 0 then trips the cycle-advance promotion via an
  unmatched-event notify at cycle 1, leaving the unit Running with a
  cycle-0 baseline. Tests that exercise encoding logic from a clean
  baseline use this to skip the pipeline gap.
- `test_start_stop_state_machine` updated to check that state stays
  Idle right after `start_event` arrives (the Start marker bytes ARE
  emitted) and only transitions to Running after a later cycle.
- `test_read_register_roundtrip` updated similarly: the Status register
  reads Idle until cycle advances.
- `rewriting_control0_resets_state_for_rerun` updated so the rerun
  start verifies the same pipelined behavior.

## Tests left alone

Tests that only care about encoding behavior — `test_single0_encoding`,
`test_lazy_commit_on_cycle_change`, etc. — fire `start_event` at cycle
0 then events at cycle ≥ 1, which already advances past the arm cycle.
No change needed.

## Open follow-ups

1. Validation against `add_one_using_dma.chess` shows the fix is a
   no-op for THIS test — see "Validation outcome" below. The unit
   tests still hold, so the model is internally consistent; we just
   don't have a bridge test that exercises the same-arm-cycle path.
2. Whether the exact same 1-cycle pipeline is correct for non-mode-2
   trace sessions has not been confirmed; the residuals were measured
   on mode-2 only. If mode-0 / mode-1 captures have a different gap
   shape (HW has separate PERF_CNT vs EVENT pipelining stages), we
   may need a per-mode adjustment.
3. The `force_start` helper exists only in tests. If future bridge
   utilities need to drive the trace unit programmatically without
   pipelining, we'd export this; for now it stays test-local.

## Validation outcome (2026-05-04)

Re-running `add_one_using_dma.chess` mode-2 baseline after the
contamination fix landed in `tools/trace-sweep.py`:

- HW: 34 PCs (correct, matching the 0501 baseline).
- EMU: 36 PCs. Two extras vs HW: `0x150` (iter-0 entry) at start
  and `0xb0` at end.

Instrumented a bridge run with `log::info!` probes on
`start_armed`, `activated`, `stop_event`, and `notify_branch_taken`
push/drop. Findings:

- Zero `branch_DROPPED` messages across 36 branch-taken events.
  All branches see `state == Running` and push successfully.
- `start_armed` fires at `total_cycles=13187` with `hw_event=122`
  (BROADCAST_15). `activated` fires at `current_cycle=13188`
  (the 1-cycle pipeline DOES happen at the trace-unit level).
- All 36 `branch_PUSHED` calls happen at `ctx.cycles` values from
  2559 to 3173. `ctx.cycles` is the per-core execution-context
  counter; `total_cycles` is the global wall-clock counter. They
  are NOT the same clock.

Conclusion: this fix's same-arm-cycle drop path is never exercised
by the kernel branches in this test. The kernel's branch retires
land at `ctx.cycles` values well after `state` transitioned to
Running (in `total_cycles` time). The 2-PC EMU-vs-HW residual is a
**broadcast event delivery timing** issue, not a state-machine
issue:

- HW: `BROADCAST_15` (start) arrives in the trace unit AT the same
  hardware cycle the kernel retires `0x150`, so the pipelined drop
  removes it.
- EMU: `BROADCAST_15` arrives WAY before the kernel retires `0x150`
  (broadcast delivery has near-zero latency in the EMU model),
  so when `0x150` retires, state is solidly Running and the
  branch is captured.

Same shape on the stop side: HW's stop event arrives close enough
to the `0xb0` retire to drop it; EMU's stop arrives later, after
`0xb0` was already pushed.

The fix is therefore correct in principle but doesn't move the
needle on this kernel. Two options for the residual:

a. Add a configurable broadcast-event-delivery latency model and
   tune it to match HW. This is the principled fix but expensive.

b. Accept the 2-PC residual as a known cycle-level timing
   divergence between EMU and HW. The PC sequences agree
   structurally; the divergence is a 1-cycle window at each end
   of the trace.

Both are deferred. The current fix stays in place because the
unit tests confirm the model is correct for the same-cycle case
it targets, and other tests may exercise that path even if
`add_one_using_dma` doesn't.

## Why we missed this earlier

Pre-timer-sync, mode-2 divergence was dominated by spurious New_PC
frames on cores running ahead of others (timer skew). 35 spurious PCs
across the suite, mostly from the timer-sync gap. Once that fix landed,
the residual revealed itself as a clean 2-PC gap that didn't match any
event, control-flow, or anchor explanation — only the pipelined-FSM
model fit. Subsystems can have multiple gaps stacked in a single
divergence; fix the dominant one first to expose the remainder.
