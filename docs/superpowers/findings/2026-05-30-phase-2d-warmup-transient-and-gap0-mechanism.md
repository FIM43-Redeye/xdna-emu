---
name: 'Phase 2d: shim warm-up transient (durations) + gap[0] is a two-part BD-prefetch / controller-pacing mechanism'
description: 'Phase 2d modeled the MM2S warm-up transient as a geometric decay of the cold-start across the task chain -- per-task transfer durations now match HW within ~3% (K=8 MM2S 1725/807/522/... vs HW 1739/804/497/...), steady-state inter-START gaps held at ~3%. But MM2S gap[0] stayed positive (+1326) vs HW negative (-433/-812). Diagnosis: gap[0] needs TWO coupled changes, neither sufficient alone -- (1) the DMA channel must emit START_TASK at BD-load/prefetch (during the prior transfer), and (2) the controller must pace TQ writes by queue occupancy so TQ[i+1] lands during task[i] rather than after it. Part 1 (prefetch START emission) is implemented + committed (b33517f); it is inert until Part 2. Part 2 (occupancy-dependent dispatch gate, calibration targets below) is designed but NOT yet implemented.'
type: project
---

# Phase 2d: warm-up transient (durations) + the gap[0] two-part mechanism

## TL;DR

Phase 2d set out to close the one structural gap left by Phase 2c: the
EMU did not model the **MM2S warm-up transient** (HW per-task transfer
durations decay across a chain). That is now fixed and committed. While
validating, the remaining `gap[0]` divergence resolved into a
**two-part mechanism** that is groundable but bigger than a constant
tweak. Part 1 is committed; Part 2 is designed and pending.

- **Durations: fixed** (commit `7568857`). K=8 MM2S EMU
  1725/807/522/427/408/399/396/398 vs HW 1739/804/497/422/398/399/364/370
  -- within ~3% at every index.
- **Steady-state inter-START gaps: held** (no regression). K=8 MM2S
  -3.7%, K=8 S2MM -2.7%.
- **`gap[0]`: still positive** (+1326) vs HW negative (-433 K=8, -812 K=4).
  Root cause is structural, not a duration error -- see below.

## Part A: the warm-up transient (DONE, committed `7568857`)

HW per-task MM2S transfer durations decay geometrically from the cold
value toward steady state over ~4 tasks (K=8: 1739, 804, 497, 422, 398,
399, 364, 369; asymptote ~383). The Phase 2c model charged the
cold-start once on task 0 and a flat per-task cost after, so it could
not reproduce the decay.

**Model**: charge `cold_start * (decay/1000)^i` at task index `i` (a
shim+host channel's `warm_task_index`), instead of the one-shot
cold-start. `i=0` reproduces the full cold-start; the tail fades over
~4 tasks. Fit: `T0 = cold_start_mm2s = 1330` (the existing constant,
within 2% of the free-fit 1356), `decay = 0.310`. S2MM decay = 0 (no
measurable tail past task 0 -- excess at i=1 is within noise), so S2MM
keeps the pure one-shot cold-start.

New per-direction constants `shim_warmup_decay_{mm2s,s2mm}_permille`
(310 / 0), threaded model_builder -> build.rs codegen -> aie2 DmaModel
-> runtime DmaTimingConfig. Applied in
`consume_first_bd_bonus` (src/device/dma/engine/stepping.rs). Retired the
obsolete `has_paid_cold_start` boolean (the geometric decay subsumes the
one-shot gate).

Validation (EMU campaign `2d-emu` vs HW baseline
`2026-05-27T04-19-33`, N=50): durations within ~3% everywhere;
steady-state inter-START gaps within ~3% (no regression).

## Part B: why gap[0] is still wrong -- the two-part mechanism

`gap[i]` here means `START_TASK[i+1] - FINISHED_TASK[i]`. HW MM2S
`gap[0]` is **negative** (the next task's START fires before the prior
task's FINISHED). The EMU produces `+1326`.

Tracing the EMU FSM (`after_transfer_done` -> `start_next_queued_task`):

- The controller gates TQ[1] to `T0 + dispatch_overhead(3050)`.
- `FINISHED[0]` is at ~1739, so at the moment task[0] finishes the queue
  is **empty** (TQ[1] not yet written) -> channel goes Idle -> START[1]
  fires only when TQ[1] lands at 3050. Hence `gap[0] = 3050 - 1739 = +1311`.

Each half-fix alone is insufficient:

| Fix | gap[0] | Why |
|-----|-------:|-----|
| Controller writes TQ[1] early (~1086) only | **0** | Channel is serial -- START[1] waits for FINISHED[0] to dequeue. max(1086, 1739)=1739. |
| DMA emits START at BD-load only | **+1311** | Controller still writes TQ[1] at 3050; nothing to load until then. |
| **Both** | **-653** | START[1]=max(TQ[1]=1086, load-slot-free~200)=1086; 1086-1739 = -653. |

So gap[0] requires **both** a DMA prefetch-START change and a controller
queue-occupancy pacing change.

### Grounding (why this is not pure speculation)

An Explore pass over aie-rt (`dma/xaie_dma_aieml.c`), AM020 ch.2/ch.5,
and the AIE2 register DB confirmed the **documented dual-state**: a DMA
channel status register exposes `Cur_BD` (the executing BD) AND
`Task_Queue_Size` (0-7 queued BDs) as separate fields, and BDs carry
`Use_Next_BD`/`Next_BD`. So a channel genuinely holds the next BD ready
while the current one drains. What is **undocumented** is the precise
START_TASK timing (BD-load vs data-move-start) -- so the *mechanism* is
grounded in the documented dual-state, and the *magnitude* (the prefetch
offset / the controller pacing ramp) is calibrated from our HW trace
data. That is the legitimate emulator pattern (derive mechanism from the
toolchain, calibrate timing from HW observation), not a speculative
controller state machine.

## Part 1 (DONE, committed `b33517f`): BD-prefetch START emission

`maybe_prefetch_next_task` (stepping.rs): while a channel is
`Transferring`, if a task is queued and START hasn't already been
emitted ahead, emit `DmaStartTask` and set
`channels[].prefetch_start_emitted`. `start_channel` suppresses the
duplicate START when that task actually begins, then clears the gate.
The data path stays strictly serial -- only the event timing moves.

Inert until Part 2: in the current campaign the controller writes TQ[i+1]
at the cold gate (3050), after task[i] (dur < 3050) has finished, so the
queue is empty during the transfer and the prefetch never fires. This
also preserves steady-state positive gaps (short tasks finish before the
next BD is queued).

Test: `test_queued_task_start_emitted_during_prior_transfer` -- a shim
MM2S channel with two tasks queued up front emits S,S,F,F (both STARTs
before either FINISH) instead of the serial S,F,S,F.

## Part 2 (DESIGNED, NOT yet implemented): controller queue-occupancy pacing

The controller must write TQ[i+1] early enough to be in the queue during
task[i]'s transfer. The HW inter-START (≈ inter-TQ-write) sequence for
K=8 MM2S **ramps**, it is not constant:

```
inter-START(i) = gap(i) + dur(i):
  1086, 2118, 3373, 3127, 3078, 3085, 3129   (rises from 1086, plateaus ~3100)
```

So the controller dispatches the first task fast (~1086) and throttles
toward a plateau (~3050) as the queue fills -- queue-occupancy
backpressure. A flat fast gate is wrong: it would make gap[1] negative
too (HW gap[1] is +1314, positive).

**Proposed model**: replace the binary cold(3050)/pipelined(520)
`dispatch_overhead` with an occupancy-dependent gate
`gate(occ) = min(base + occ*slope, plateau)`, where `occ` = outstanding
tasks (in-flight + queued) at the moment of the TQ write.

**Initial calibration targets** (from the HW ramp):
- `base ≈ 1086` (gate at occ=0 -> TQ[1] at 1086 -> gap[0] = 1086-1739 = -653)
- `slope ≈ 1032` (gate at occ=1 ≈ 2118 -> matches inter-START[1->2])
- `plateau ≈ 3050` (existing steady value; gate at occ>=2 caps here)

`classify_task_dispatch` (executor.rs) currently returns
`Option<bool>` (idle/busy); it would return enough to compute `occ`
(e.g. `task_queue_size + channel_running`). The gate is then
`base + occ*slope` capped at `plateau`, set into
`controller_next_taskq_cycle`.

This is an **empirical calibration loop**: implement the occupancy gate,
rebuild the FFI .so, run `multirun-trace-campaign --emu`, compare against
the HW baseline with `compare-dispatch-overhead.py`, and tune
base/slope/plateau until gap[0..] converge AND steady-state + durations
do not regress. Constants live in `src/npu/cycle_cost.rs`.

## Risks / watch items for Part 2

- **Steady-state regression**: the plateau must stay ~3050 so the
  steady-state inter-START match (~3%) is preserved. Guard with the
  comparator's `steady_gaps` rows.
- **gap[1] sign**: HW gap[1] is positive (+1314); the ramp must throttle
  fast enough that task[1]'s prefetch of task[2] does NOT fire (TQ[2]
  must land after FINISHED[1]). This is the main constraint distinguishing
  a correct ramp from a flat fast gate.
- **Trace-test sensitivity**: Part 1 changed START emission order for
  queued-during-transfer cases; full `cargo test --lib` passed (3220),
  but bridge/trace HW-comparison tests were not run (need HW) -- they
  should *improve* (EMU now matches HW prefetch) but verify on next HW run.

## Artifacts

- Commits: `7568857` (Part A warm-up durations), `b33517f` (Part 1
  prefetch START emission)
- EMU campaign: `build/experiments/dispatch-overhead-multirun/2d-emu/`
  (git rev 75688571c9b7, durations validated)
- HW baseline: `build/experiments/dispatch-overhead-multirun/2026-05-27T04-19-33/` (N=50)
- Decompose: `python3 tools/decompose-stalls.py <session>`
- Compare: `python3 tools/aggregate-dispatch-overhead.py <emu>` then
  `python3 tools/compare-dispatch-overhead.py <hw> <emu>`
- Prior finding:
  [`2026-05-27-phase-2c-dispatch-overhead-recalibration`](2026-05-27-phase-2c-dispatch-overhead-recalibration.md)
  ("Remaining gap" -- this finding closes Part A of it)
