# Trace level-event emission: a unified held-level model (design)

**Date:** 2026-06-07
**Status:** design / pending implementation plan
**Scope:** ALL core/mem trace events, not just LOCK_STALL. LOCK_STALL is the
exemplar and primary calibration anchor; the fix is a single unified
level-vs-pulse emission model driven by the existing `is_level_event()`
classification.
**Supersedes:** `docs/superpowers/findings/2026-05-25-lock-stall-cadence-h1-refuted.md`
(its "period=1 is correct, fix is wire compression" conclusion rested on a
since-fixed decoder bug -- see Evidence).
**Related:** `docs/superpowers/findings/2026-06-07-lock-stall-overemission-interp-vs-hw.md`
(the measured gap), `docs/known-fidelity-gaps.md`.

## Problem

Trace emission does not respect the level-vs-pulse nature of events. The
emulator already *classifies* events correctly for trace *comparison*
(`is_level_event()` in `src/trace/compare.rs:350-410`), but the *emission* path
handles level events ad-hoc and inconsistently:
- **LOCK_STALL** (LEVEL) is emitted as a per-cycle **pulse** -> ~100-375x too
  many events, wrong span durations. This is the exemplar bug.
- **MEMORY_STALL, STREAM_STALL, CASCADE_STALL** (LEVEL) share the same
  pulse-emission shape and the same latent bug.
- **PORT_RUNNING / PORT_IDLE / PORT_STALLED** (LEVEL) are already emitted as held
  spans, but via separate ad-hoc logic; their residual count gap (PORT_RUNNING 1
  vs 6) is DMA bursty-delivery timing, NOT emission shape.
- **DMA `*_STREAM_STARVATION` / `*_STALLED_LOCK` / backpressure** (LEVEL) are
  edge-emitted on the rising edge only, missing clean falling-edge pairing.

The goal: one held-level emission mechanism for every event `is_level_event()`
calls LEVEL, and the existing pulse path for the rest. LOCK_STALL measured on
the SAME peano xclbin, same decoder, both sides (the exemplar):

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

## Design (approach B: unified held-level model)

The principle: **every event the existing `is_level_event()` classifier calls
LEVEL is emitted as a held signal -- one frame on the rising edge, one on the
falling edge -- producing a single B..E span of the real held duration. Pulse
events keep the existing one-cycle path.** `is_level_event()` is the single
source of truth for the classification; emission and comparison both consult it
(no duplicate list).

### 1. Trace unit: a held-level mask alongside the pulse mask
`src/device/trace_unit/`. Today the unit accumulates a per-cycle
`pending_slot_mask` and commits exactly one frame per cycle (already a
sampled-active-set shape). Generalize it to two inputs:
- a **held mask** -- level-event slot bits that stay set across cycles until
  explicitly cleared;
- a **pulse mask** -- one-cycle bits (the existing behavior).

Each cycle the active set = `held | pulse`. Commit a frame whenever the active
set *changes* from the last committed set (exactly what HW does and what the
decoder's `_emit_be` already pairs on). A held level thus yields one rising-edge
frame (B) and one falling-edge frame (E); the long delta between is the span.

Interface shape: finalize against the existing `notify_event` and
`pending_slot_mask` -- the natural fit is a level setter (e.g.
`set_event_level(hw_id, cycle, pc, active: bool)`) that sets/clears the held-mask
bit, with `notify_event` retained for pulses. The trace unit decides held vs
pulse by consulting `is_level_event(hw_id)` (or an hw_id-keyed equivalent), so
callers cannot get it inconsistent. Point events stay byte-identical.

### 2. Interpreter / device: drive level edges for every level event
Replace ad-hoc per-event emission with explicit assert(rising)/deassert(falling)
at each level condition's transitions. The exemplar, LOCK_STALL:
- **Delete** the per-cycle re-emission (`interpreter.rs` ~729-743,
  `LOCK_STALL_TRACE_PERIOD`).
- **Assert** on each lock arbitration (every acquire `control.rs:224`, every
  release `control.rs:308`); immediate transactions assert then deassert next
  cycle -> 1ns span (the 16 short-arb spans). Reconcile the existing release
  point-emit (`control.rs:367`) into the level model.
- **Hold** through a genuine wait: assert at `WaitLock` entry
  (`cycle_accurate.rs:818`), deassert at lock-acquired (`try_resume_stall`
  success, `interpreter.rs:721-726`) -> one long span of the real duration.

Then apply the same enter/exit edge treatment to the other level classes:
- **MEMORY_STALL, STREAM_STALL, CASCADE_STALL** (core): assert on stall entry,
  deassert on resolution -- the same shape as LOCK_STALL, currently pulse-bugged.
- **PORT_RUNNING / PORT_IDLE / PORT_STALLED** (core stream ports): re-express
  the existing held-span logic through the unified path. Emission shape becomes
  consistent; the residual count gap stays (see Scope honesty).
- **DMA `*_STREAM_STARVATION` / `*_STALLED_LOCK` / backpressure / memory
  starvation** (mem): the DMA engine already edge-fires on the rising edge
  (`stepping.rs:438-444` via `prev_starving`); add the falling-edge deassert so
  the trace unit closes the span. Generalize the `prev_*` edge pattern to all
  these level conditions.

### 3. Scope honesty (what "everything" does and does not close)
The unified mechanism covers ALL level events, but closure differs by cause:
- **Fully closed by this work** (their gap *is* the emission shape): LOCK_STALL,
  MEMORY_STALL, STREAM_STALL, CASCADE_STALL, and clean B..E pairing for the DMA
  stall/starvation/backpressure family.
- **Mechanism unified, count residual remains**: PORT_RUNNING (1 vs 6),
  STREAM_STARVATION counts (1 vs 2). These are driven by *when* the port/channel
  goes idle->active -- DMA bursty-delivery timing, the deferred DDR-sim axis.
  This work makes their emission consistent and their span durations correct for
  the activity we *do* model, but the count gap waits on the DMA timing work.
  Do NOT pull DMA timing modeling into this change.

## Components and files

| Change | File(s) | Note |
|---|---|---|
| Held-level mask + frame-on-change; classification via `is_level_event()` | `src/device/trace_unit/mod.rs`; consult `src/trace/compare.rs:350-410` | held \| pulse active set; pulse path byte-identical |
| LOCK_STALL edges; delete periodic | `src/interpreter/core/interpreter.rs` (101, 721-743), `src/interpreter/execute/cycle_accurate.rs` (818-822), `src/interpreter/execute/control.rs` (224-306, 308-368) | assert on arb + wait-enter, deassert on arb-done + lock-acquired |
| MEMORY/STREAM/CASCADE_STALL edges | core stall emission sites (audit during plan) | same enter/exit shape as LOCK_STALL |
| PORT_* re-express via unified path | core stream-port emission sites | shape only; count residual is DMA-timing |
| DMA stall/starvation/backpressure falling edge | `src/device/dma/engine/stepping.rs` (438-444), `src/device/dma/channel.rs` | generalize `prev_starving` edge to deassert |
| Coordinator drain forwards level transitions | `src/interpreter/engine/coordinator.rs` (860-872) | forward assert/deassert, not only point notifies |

## Testing / calibration (test-first)

Two tiers: mechanism (unit) and fidelity (HW calibration).

**Mechanism (trace_unit unit tests, TDD before code):**
- A held-level assert then deassert N cycles later -> exactly one B..E span of
  duration N (not N pulses, not a 1ns span).
- Interleaved level + pulse events in the same window -> level holds across the
  pulses; pulses still produce 1-cycle spans.
- Pulse events (INSTR_*, DMA START/FINISHED, LOCK_n_ACQ/REL, PERF_CNT) remain
  byte-identical to current output (golden-bytes guard).
- `is_level_event()` drives the held/pulse decision (a level hw_id holds, a pulse
  hw_id does not) -- single source of truth, no duplicate list.

**Fidelity (HW calibration -- count AND decomposition, not just totals):**
- LOCK_STALL exemplar: `vec_mul_trace_distribute_lateral` -> 19 = 1 startup + 2
  waits + 16 arb, with the 2 long spans at the real durations (6354 / 1930 ns),
  not 1ns. `_diag_phase_b_add_one_instrumented` -> 46 (peano) / 40 (chess).
- MEMORY_STALL / STREAM_STALL / CASCADE_STALL: pick kernels that exercise each
  (audit during plan), diff EMU vs HW.
- PORT_* and DMA starvation: assert emission *shape* is correct (proper B..E
  spans, durations match the activity we model) and document the residual count
  delta as expected (DMA-timing-bound), so it is not chased as a regression.

Procedure: capture HW targets (LOCK_STALL already in hand); write mechanism tests
first; implement trace_unit held-level support; migrate LOCK_STALL, verify 19 /
46; migrate the remaining level classes one at a time, re-running EMU-side and
diffing; full `cargo test --lib` + the broader trace test suite each step; verify
against baseline trace.log (a "CLEAN" bridge verdict can miss level-event
duration regressions).

## Risks
- **Trace-unit blast radius:** the held-mask change touches the shared encoder
  for ALL events. Pulse events must remain byte-identical -- golden-bytes guard
  before/after. This is the highest-risk part; land the trace_unit change with
  its tests before migrating any emission site.
- **Scope size:** this is a multi-event-class refactor of the emission path, not
  a one-constant fix. Sequence it: trace_unit mechanism -> LOCK_STALL (calibrate
  to 19/46) -> one level class at a time. Each class is independently verifiable;
  the feature is not "done" until every level class is migrated and passing (per
  the finish-what-you-start policy).
- **Double-counting on contended acquire:** an acquire that stalls must produce
  ONE span (assert at entry, deassert at acquire), not entry pulse + success
  pulse. The decomposition test guards this.
- **PORT_*/starvation false-regression:** their count residual is DMA-timing, not
  this work. Document the expected delta so a reviewer does not read it as a bug.
- **Off-by-one cycle:** EMU shows 12298 vs HW 12297 per invocation -- negligible,
  out of scope, noted so it is not mistaken for a regression.
