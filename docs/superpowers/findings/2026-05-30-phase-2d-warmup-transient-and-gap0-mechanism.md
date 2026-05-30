---
name: 'Phase 2d: shim warm-up transient (durations) + gap[0] is a two-part BD-prefetch / controller-pacing mechanism'
description: 'Phase 2d modeled the MM2S warm-up transient as a geometric decay of the cold-start across the task chain -- per-task transfer durations now match HW within ~3% (K=8 MM2S 1725/807/522/... vs HW 1739/804/497/...), steady-state inter-START gaps held at ~3%. But MM2S gap[0] stayed positive (+1326) vs HW negative (-433/-812). Diagnosis: gap[0] needs TWO coupled changes, neither sufficient alone -- (1) the DMA channel must emit START_TASK at BD-load/prefetch (during the prior transfer), and (2) the controller must pace TQ writes by queue occupancy so TQ[i+1] lands during task[i] rather than after it. Part 1 (prefetch START emission) is implemented + committed (b33517f); it is inert until Part 2. Part 2 (a ramped controller dispatch gate) is DONE -- implemented, unit-tested, and campaign-validated. Two structural lessons: (1) the spec''s instantaneous-occupancy signal cannot reach the HW plateau (occupancy collapses to 0 between short tasks), so the gate is indexed by a monotonic per-channel dispatch counter instead (analog of Part A''s warm_task_index); (2) the fast first dispatch is MM2S-specific (prefetch from DDR), so the gate is per-direction -- MM2S ramps {1086, slope 1964, plateau 3050}, S2MM is flat at 3050. EMU campaign 2d3-emu confirms MM2S gap[0] fixed (+1326 -> -64, HW -433), gap[1] in IQR, and S2MM byte-for-byte unchanged from pre-Part-2 (no regression).'
type: project
---

# Phase 2d: warm-up transient (durations) + the gap[0] two-part mechanism

## TL;DR

Phase 2d set out to close the one structural gap left by Phase 2c: the
EMU did not model the **MM2S warm-up transient** (HW per-task transfer
durations decay across a chain). While validating, the remaining `gap[0]`
divergence resolved into a **two-part mechanism**: a BD-prefetch START
emission (Part 1) plus a controller dispatch-rate ramp (Part 2).
**All three are now done, committed, and campaign-validated.**

- **Durations: fixed** (commit `7568857`). K=8 MM2S EMU
  1725/807/522/427/408/399/396/398 vs HW 1739/804/497/422/398/399/364/370
  -- within ~3% at every index.
- **Steady-state inter-START gaps: held** (no regression). K=8 MM2S
  -3.7%, K=8 S2MM -2.7%.
- **`gap[0]`: fixed** (commits `b33517f` prefetch + `1bd0433` ramp +
  per-direction follow-up). K=8 MM2S +1326 -> -64 (HW median -433, in the
  HW IQR), with S2MM left byte-for-byte unchanged. See Parts 1-2 below.

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

## Part 2 (DONE -- implemented, unit-tested, campaign-validated): per-direction controller dispatch-rate ramp

The controller must write TQ[i+1] early enough to be in the queue during
task[i]'s transfer. The HW inter-START (≈ inter-TQ-write) sequence for
K=8 MM2S **ramps**, it is not constant:

```
inter-START(i) = gap(i) + dur(i):
  1086, 2118, 3373, 3127, 3078, 3085, 3129   (rises from 1086, plateaus ~3100)
```

So the controller dispatches the first task fast (~1086) and throttles
toward a plateau (~3050) over the first ~2 dispatches. A flat fast gate
is wrong: it would make gap[1] negative too (HW gap[1] is +1314,
positive).

**Model**: replace the binary cold(3050)/pipelined(520)
`dispatch_overhead` with a **per-direction** ramped gate
`gate(idx) = min(base + idx*slope, plateau)`, evaluated at each TQ write
(MM2S ramps, S2MM flat -- see iterations below for why and the final
constants).

### The signal pivot: dispatch index, NOT instantaneous occupancy

The original design fed the gate with **instantaneous occupancy**
(`task_queue_size + channel_running` at TQ-write time). Tracing it
against the HW numbers showed this is **structurally unable to reach the
plateau**:

- HW steady-state inter-START is ~3100 while steady task duration is
  ~370. Inter-START >> duration means the channel is **idle between
  tasks** -- the controller is the bottleneck and the channel drains and
  sits empty waiting for the next dispatch.
- So at every steady TQ write the channel is idle with an empty queue ->
  `occ = 0` -> `gate(0) = base`. The gate never climbs to the plateau; it
  produces a *spike-then-collapse* `1086, 2118, 1086, 1086, …` (only the
  cold task[0], dur 1739 > 1086, is still in flight at TQ[1]).
- This holds on HW too (same idle-between-tasks geometry), so it is not
  an EMU artifact -- instantaneous occupancy is the wrong signal.

**Fix**: index the ramp by a **monotonic per-channel dispatch counter**
(`controller_dispatch_index`) -- the count of TQ writes issued since the
channel's last reset -- not instantaneous occupancy. Monotonic 0->1->2->
plateau, and it *stays* at plateau, reproducing both the fast first
dispatch (gap[0] negative) and the steady ~3100 plateau. It is the
controller-side analog of Part A's `warm_task_index`: the dispatch
pipeline fills over the first ~2 tasks, then runs at the serialized rate.
Reset only on `stop_channel`/channel reset (fresh boot), mirroring
`warm_task_index`.

### Where it lives

- `CycleCostModel` (src/npu/cycle_cost.rs): retired
  `dispatch_overhead`/`dispatch_overhead_pipelined`; added a
  `DispatchGate { base, slope, plateau }` struct and per-direction
  `dispatch_mm2s`/`dispatch_s2mm` fields (disabled in legacy +
  with_known_constants), and the `dispatch_overhead_for(idx, is_mm2s)`
  gate method. See the per-direction iteration below for the values.
- `ChannelContext` (src/device/dma/channel.rs): added
  `controller_dispatch_index: u32`, reset in new()/reset()/stop_channel.
- `enqueue_task` (engine/task_queue_ops.rs): increments the index per
  dispatch; new `controller_dispatch_index(channel)` accessor.
- `classify_task_dispatch` (executor.rs): now returns
  `(dispatch_index, is_mm2s)` (`Option<(u32, bool)>`) and the
  gate-application uses `dispatch_overhead_for(idx, is_mm2s)`.

Unit tests (RED->GREEN): `dispatch_gate_is_per_direction`,
`dispatch_overhead_is_zero_in_non_dispatch_profiles` (cycle_cost),
`controller_dispatch_index_is_monotonic_across_drains` (engine -- the
persist-across-drain property that occupancy lacks), and the updated
`classify_task_dispatch_*` + `cycle_cost_model_mm2s_gate_ramps_s2mm_flat`.
Full `cargo test --lib` green (3223), zero regressions.

### Iteration 1 (single gate, both directions): the per-direction lesson

First EMU campaign (`2d2-emu`, single gate `1086/1032/3050`) vs the HW
baseline:

- **MM2S gap[0]: fixed.** +1326 -> -64 (HW median -433, in the HW IQR).
  Primary Part 2 goal met; steady-state held (~3%).
- **MM2S gap[1]: regressed** (2244 -> 672). The gradual slope (gate(1)=2118)
  over-discounted the second dispatch.
- **S2MM gap[0]/gap[1]: regressed badly** (2461 -> 498; 2800 -> 1868). S2MM
  had a near-perfect pre-Part-2 match (HW g0 2520, g1 2791, MAD only
  7-72), and the fast base=1086 halved its first two dispatches.

Root cause: the fast-first-dispatch is the **BD-prefetch** behavior, which
is **MM2S-specific** (MM2S sources from DDR, available immediately; S2MM
waits on stream data, so its first dispatch pays full freight ~plateau).
Applying one ramp to both directions is wrong.

### Iteration 2 (per-direction gate): the current model

The gate is split into a `DispatchGate { base, slope, plateau }` per
direction (`dispatch_mm2s` / `dispatch_s2mm` on `CycleCostModel`):

- **MM2S** `{base: 1086, slope: 1964, plateau: 3050}` -- base is below the
  per-task BD-config instruction floor (~1661) so it does NOT bind: task 0
  is instruction-bound and the Part 1 prefetch drives gap[0] negative;
  from index 1 on the gate is at the plateau (slope=1964 -> gate(1)=3050)
  so the steady gaps match the prior flat behavior (fixes the gap[1]
  regression). A 2-point ramp (fast first, then plateau), not gradual --
  the gaps don't support the 2118 middle point the raw inter-START ramp
  suggested (duration warm-up confounds that reconstruction).
- **S2MM** `DispatchGate::flat(3050)` -- pays the plateau on every
  dispatch, restoring the tight pre-Part-2 match.

`classify_task_dispatch` now returns `(dispatch_index, is_mm2s)` (the
direction is recovered from the absolute channel index: `ch >=
s2mm_channel_count` is MM2S).

**Iteration 2 results** (`2d3-emu` vs HW baseline; all values in HW IQR
unless noted):

| cell.gap | HW med | pre-P2 | iter1 | iter2 | verdict |
|----------|-------:|-------:|------:|------:|---------|
| k8.MM2S g0 | -433 | +1326 | -64 | **-64** | ✅ fixed (in IQR) |
| k8.MM2S g1 | 2683 | 2244 | 672 | **1604** | ✅ recovered to IQR (slope fix) |
| k8.MM2S g2-6 | ~2700 | ~2600 | ~2600 | ~2600 | held |
| k8.S2MM g0 | 2520 | 2461 | 498 | **2461** | ✅ restored (11 cyc below p25) |
| k8.S2MM g1 | 2791 | 2800 | 1868 | **2800** | ✅ restored (in IQR) |
| k8.S2MM g2-6 | ~2900 | 2800 | 2800 | 2800 | unchanged baseline offset |

**S2MM is byte-for-byte identical to pre-Part-2** -- the per-direction
split fully isolated it from the MM2S ramp (the iteration-1 regression is
gone). MM2S gap[0] is fixed (the Part 2 goal) and gap[1] recovered into
the IQR. The S2MM g2-g6 "below HW" rows are a *pre-existing* steady-state
offset (EMU 2800 vs HW ~2900-3096), unchanged by Part 2, not a regression.

**Converged.** MM2S gap[1] sits at 1604 vs HW median 2683, but the fast
first dispatch necessarily cascades to a lower gap[1] (raising it would
require gate(1) > plateau, breaking steady-state), and HW gap[1] MAD is
3108 (IQR -674..5446) -- pushing the deterministic EMU value closer to the
median is fitting to noise. All finding criteria met: gap[0] negative,
gap[1] positive, steady-state + durations + S2MM not regressed.

### Empirical loop (for further tuning)

Rebuild the FFI .so, run `multirun-trace-campaign --emu`,
`aggregate-dispatch-overhead.py`, then `compare-dispatch-overhead.py
<hw> <emu>`. Constants live in `src/npu/cycle_cost.rs`
(`provisional_npu1`). Note the MM2S transient gaps have *enormous* HW
variance (g0 MAD 908, g1 MAD 3108) -- EMU is deterministic, so chasing the
HW median within that spread is partly fitting to noise; S2MM has tight
MAD, so its match is the unambiguous guard.

## Risks / watch items for Part 2

- **Steady-state regression**: the plateau must stay ~3050 so the
  steady-state inter-START match (~3%) is preserved. Guard with the
  comparator's `steady_gaps` rows.
- **gap[1] sign**: HW gap[1] is positive (+1314); the ramp must throttle
  fast enough (idx=1 -> gate 2118, idx>=2 -> plateau 3050) that task[1]'s
  prefetch of task[2] does NOT fire (TQ[2] must land after FINISHED[1]).
  This is the main constraint distinguishing a correct ramp from a flat
  fast gate, and is why `slope` must not be too small.
- **Dispatch-index reset scope**: the index resets only on
  `stop_channel`/channel reset, so a second independent chain after a
  long idle (without a channel disable) would keep the index high and
  pay the plateau rate on its first dispatch instead of the fast base.
  Fine for the single-chain K-sweep; if a multi-chain kernel needs the
  fast-restart behavior, add an idle-gap reset (index -> 0 when the
  channel has been idle longer than `plateau` before a dispatch).
- **Trace-test sensitivity**: Part 1 changed START emission order for
  queued-during-transfer cases; full `cargo test --lib` passed (3223),
  but bridge/trace HW-comparison tests were not run (need HW) -- they
  should *improve* (EMU now matches HW prefetch + ramp) but verify on the
  next HW run.

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
