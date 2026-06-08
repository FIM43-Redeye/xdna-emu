# LOCK_STALL trace emission: level-event model (design)

**Date:** 2026-06-07
**Status:** design / pending implementation plan
**Supersedes:** `docs/superpowers/findings/2026-05-25-lock-stall-cadence-h1-refuted.md`
(its "period=1 is correct, fix is wire compression" conclusion rested on a
since-fixed decoder bug -- see Evidence).
**Related:** `docs/superpowers/findings/2026-06-07-lock-stall-overemission-interp-vs-hw.md`
(the measured gap), `docs/known-fidelity-gaps.md`.

## Problem

Our interpreter emits the LOCK_STALL core-trace event ~100-375x too often
versus real NPU1 hardware, and with the wrong span durations. Measured on the
SAME peano xclbin, same decoder, both sides:

| Kernel | HW LOCK_STALL | EMU LOCK_STALL |
|---|---|---|
| `vec_mul_trace_distribute_lateral` | 19 | 7112 |
| `_diag_phase_b_add_one_instrumented` | 46 (peano) / 40 (chess) | 4449 |

This is the dominant trace-fidelity gap in the interpreter. Because trace
comparison against HW is how we validate the emulator, a 375x event-count error
(and wrong span durations) poisons any LOCK_STALL-based validation.

## Evidence: HW is a level signal, ours is a pulse

### HW decomposition (conclusive)
Decomposing distribute_lateral's HW LOCK_STALL spans by duration and correlating
each with surrounding lock transactions (`build/experiments/bcast-bridge/trace_hw.json`):

```
2 LONG-wait spans + 16 short-arb (1ns) spans + 1 startup = 19
  startup:  ts 1 -> 6355   (6354 ns)  initial wait before first acquire
  wait:     ts 6372 -> 8302 (1930 ns) acquire that could not complete (lock unavailable)
  16x 1ns:  each immediately trails a lock transaction (9 ACQUIRE + 9 RELEASE = 18 traced)
```

The rule: **LOCK_STALL is a traced level signal.** It asserts on every lock
acquire/release arbitration (brief -> 1ns span if the lock is free) and the
assertion *extends* through a genuine wait (lock unavailable -> long span). One
span per assertion. add_one (46) is consistent: releases are not assigned a
trace slot there, yet still produce arbitration spans -- confirming the span
fires on the *arbitration*, independent of whether the REQ event is traced.

### Why we over-emit (root cause)
- HW (and the AIE trace format generally): a core trace event is a **level**.
  The trace controller emits a frame only when the active-event *set changes* (a
  transition). The decoder (`tools/trace_decoder/decode.py` `rebuild_perfetto_mode0`,
  `_emit_be`) reconstructs spans by level transitions: slot bit 0->1 = "B", 1->0
  = "E". A held stall = one rising + one falling edge = one B..E span of the
  correct duration.
- Ours: `LOCK_STALL_TRACE_PERIOD = 1` (`src/interpreter/core/interpreter.rs:101`)
  re-emits a LOCK_STALL **point event every cycle** of held stall
  (`interpreter.rs` ~737-742). Each point event sets the slot bit for one frame
  then clears, so the decoder sees a 1-cycle 0->1->0 -> a 1ns B..E span. The
  7112 measured spans are each ~1ns, 2ns apart. So we emit one isolated pulse per
  cycle instead of one held level per assertion -- wrong count AND wrong duration.

### The 2026-05-25 supersession
That finding measured HW add_one LOCK_STALL = 2233-2766 and concluded period=1
was correct. Re-decoding the *same kernel's* current HW capture
(`build/bridge-test-results/20260606/..._add_one_instrumented.peano.hw/events.json`)
with the same `ours` decoder now gives **46** -- the decoder used to *expand*
HW's held-level skip-token runs into thousands of phantom per-cycle events,
coincidentally matching our pulse spam. The decoder was fixed; the 46 matches the
original Phase-C "44 events" claim that finding had dismissed. Its "fix = wire
compression, interpreter is correct" conclusion is therefore wrong: the
interpreter over-emits, and the fix is level-event semantics, not wire
compression. (Skip-token wire compression is a separate, optional size
optimization and is out of scope here.)

## Design (approach B: model the level signal)

The principle: **represent LOCK_STALL as a held level driven by edges, emitted
on transition -- matching HW -- rather than a per-cycle pulse.** Two cooperating
changes plus a calibration loop.

### 1. Trace unit: support held-level events
`src/device/trace_unit/`. Today `notify_event(hw_id, cycle, pc)` treats every
notification as a one-cycle pulse (sets the slot bit for the pending frame, which
then clears). Add a level/state notion for the level-class core events so a bit
stays asserted across cycles until explicitly deasserted, and a frame is emitted
on the rising and falling transitions (the existing one-frame-per-cycle,
delta-encoded path already produces the right bytes once the bit is held).

Interface shape (to finalize in the plan): an assert/deassert pair, e.g.
`notify_event_level(hw_id, cycle, pc, active: bool)`, or an explicit
begin/end. Point events (INSTR_LOCK_ACQUIRE_REQ, INSTR_EVENT_*) keep the existing
pulse path unchanged.

### 2. Interpreter: drive the lock-stall level edges
- **Delete** the per-cycle re-emission (`interpreter.rs` ~729-743,
  `LOCK_STALL_TRACE_PERIOD`).
- **Assert** the lock-stall level on each lock arbitration: every acquire
  (`control.rs:224` LockAcquire) and every release (`control.rs:308`
  LockRelease). For an immediate (lock-free) transaction, assert then deassert
  the next cycle -> 1ns span (the 16 short-arb spans). The release path already
  emits a point LockStall (`control.rs:367`); reconcile it into the level model.
- **Hold** the level through a genuine wait: when an acquire becomes
  `WaitLock` (`cycle_accurate.rs:818`), the level stays asserted from stall-entry
  until the lock is acquired (`try_resume_stall` success, `interpreter.rs:721-726`),
  where it deasserts -> one long span of the correct duration (the 2 wait spans +
  startup).

### 3. Scope boundary
LOCK_STALL only. PORT_RUNNING (1 vs 6) and STREAM_STARVATION (1 vs 2) are the
same *level-event* shape and will benefit from the trace-unit level support, but
their remaining gap is driven by DMA bursty-delivery timing (the deferred
DDR-sim axis) -- not fixed here. MEMORY_STALL / STREAM_STALL are also level
events; migrate them to the level path opportunistically if low-risk, otherwise
leave for a follow-up. Do not expand scope into the DMA timing model.

## Components and files

| Change | File(s) | Note |
|---|---|---|
| Held-level event support | `src/device/trace_unit/mod.rs` | assert/deassert; frame-on-transition; point events unchanged |
| Drive stall-level edges; delete periodic | `src/interpreter/core/interpreter.rs` (101, 721-743), `src/interpreter/execute/cycle_accurate.rs` (818-822), `src/interpreter/execute/control.rs` (224-306 acquire, 308-368 release) | assert on arb + wait-enter, deassert on arb-done + lock-acquired |
| Coordinator drain | `src/interpreter/engine/coordinator.rs` (860-872) | may need to forward level transitions, not just point notifies |

## Testing / calibration (test-first)

Calibration targets are now HW ground truth and must match **count and
decomposition**, not just totals:
- `vec_mul_trace_distribute_lateral`: 19 = 1 startup + 2 waits + 16 arb. The 2
  long spans must have the right durations (6354 / 1930 ns), not 1ns.
- `_diag_phase_b_add_one_instrumented`: 46 (peano) / 40 (chess).

Procedure:
1. Capture HW targets (already in hand: `trace_hw.json`; add_one events.json).
2. TDD: add trace-unit tests for level assert/deassert -> single span; assert
   point events still produce pulses.
3. Implement; re-run both kernels EMU-side (`run_distlat_emu.sh` + an add_one
   equivalent), decode, diff vs HW until count+decomposition match.
4. Regression: full `cargo test --lib` (trace_unit tests), and the broader trace
   test suite -- ensure no other event class regressed by the trace-unit change.
5. Verify against baseline trace.log per the team rule (a "CLEAN" bridge verdict
   can miss level-event duration regressions).

## Risks
- **Trace-unit blast radius:** the level change touches the shared encoder. Point
  events must remain byte-identical. Mitigate with targeted trace_unit tests
  before/after.
- **Double-counting on contended acquire:** an acquire that stalls must produce
  ONE span (assert at entry, deassert at acquire), not an entry pulse + a
  success pulse. The decomposition test guards this.
- **Off-by-one cycle:** EMU shows 12298 vs HW 12297 per invocation -- negligible,
  out of scope, noted so it is not mistaken for a regression.
