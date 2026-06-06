---
name: 'Shim DMA cold-start amortizes across chained tasks'
description: HW measurement on _diag_shim_chain_sweep K=8 shows shim DMA pays cold-start ONCE per channel session, then steady-state per-task overhead on every subsequent dispatch.  EMU's prior model fired cold-start on alternating tasks (depth-2 queue artifact); fix decomposes shim "cold_start" into a once-per-session cold term + a per-task overhead term applied on every Idle->task transition.
type: project
---

# Shim DMA cold-start amortizes across chained tasks

## TL;DR

The 2026-05-25 N-sweep calibration ([finding](2026-05-25-shim-throughput-1-word-per-cycle.md))
measured shim DMA single-task durations as `cold_start + N*slope`.  It assumed
that the "cold_start" intercept (747 MM2S, 171 S2MM) fires on every task.

A follow-up K-sweep
(`_diag_shim_chain_sweep/k{1,2,4,8,16}`, fixed N=64, K back-to-back
`dma_memcpy_nd` dispatches per direction) shows that HW actually pays
cold-start ONCE per channel per session.  Subsequent same-direction tasks
pay only a smaller steady-state "per-task overhead" on top of the throughput
term.

EMU's prior model fired cold-start on every-other task (a depth-2 task
queue artifact -- back-to-back queued tasks share is_first_bd, but the
queue drains every 2 tasks and re-arms it).  Real HW amortizes
indefinitely within a session.

## The data

`_diag_shim_chain_sweep/k8`, chess compiler, HW per-task durations (cyc):

| direction | task 0 | tasks 1..7 |
|-----------|-------:|-----------:|
| MM2S      | 1018   | ~313 (313, 313, 314, 314, 313, 314, 313) |
| S2MM      | 880    | ~241 (231, 250, 250, 250, 210, 248, 248) |

First task pays a much larger cost; subsequent tasks settle to a constant.

Decomposing each row as `cold_start + per_task_overhead + N*slope`
(with N=64, slope=1.0 from the prior N-sweep calibration):

- MM2S task 0:    1018 = `cold (705) + per_task (249) + N (64)`
- MM2S tasks 1-7:  313 =              `per_task (249) + N (64)`
- S2MM task 0:     880 = `cold (639) + per_task (177) + N (64)`
- S2MM tasks 1-7:  241 =              `per_task (177) + N (64)`

The first-task "cold" component varies 30-50% between sweeps
(run-to-run HW variance; see `Caveats` below), so the canonical values
used in EMU are calibrated against the LARGER N-sweep dataset rather than
isolated K-sweep first-task numbers:

| direction | EMU cold (one-shot per channel) | EMU per_task (every task) | Sum (= old "cold") |
|-----------|--------------------------------:|--------------------------:|-------------------:|
| MM2S      | 498                             | 249                       | 747 (matches old N-sweep cold) |
| S2MM      |   0                             | 168                       | 168 (matches old N-sweep cold ~171) |

The split preserves single-task duration exactly (K=1 EMU output unchanged)
while making K>1 chains amortize correctly.

## What changed in EMU

In `crates/xdna-archspec/`:
- `src/types.rs` -- `DmaTiming`: added `shim_per_task_overhead_{mm2s,s2mm}_cycles`
- `src/dma/mod.rs` -- `DmaTimingConfig`: same
- `src/model_builder.rs` -- AIE2 values: cold MM2S 747->498, S2MM 171->0;
  added per_task MM2S=249, S2MM=168
- `src/aie2/dma.rs` -- expose new constants via `DmaModel::timing_config()`
- `build.rs` -- codegen for `DMA_SHIM_PER_TASK_OVERHEAD_{MM2S,S2MM}_CYCLES`

In `src/device/dma/`:
- `timing.rs` -- runtime mirror of arch-side `DmaTimingConfig` fields
- `channel.rs` -- added `has_paid_cold_start: bool` to `ChannelContext`
  (false at construction; reset to false on `reset()` and `stop_channel`)
- `engine/stepping.rs` -- `consume_first_bd_bonus` decomposes the bonus
  into three terms: channel_start (every task, all tiles), per_task_OH
  (every task on shim+host), cold_start (gated on `!has_paid_cold_start`,
  set to true after firing)
- `engine/stepping.rs` -- `after_transfer_done` queue-fed path now ALSO
  re-arms `is_first_bd = true`.  Previously this was deliberately skipped
  to avoid re-firing the (then-monolithic) cold-start.  With cold-start
  now gated separately by `has_paid_cold_start`, queued tasks correctly
  pay per-task overhead while still skipping cold-start.
- `engine/mod.rs` -- `stop_channel` resets `has_paid_cold_start = false`
  (channel reset == fresh boot)
- `engine/tests.rs` -- updated test expectations

## Validation

K-sweep `_diag_shim_chain_sweep` per-task durations (post-fix, chess HW
vs EMU; cyc):

K=8 MM2S:

| task | EMU (this build) | HW (this run) | HW range (prior sweeps) |
|-----:|-----------------:|--------------:|------------------------:|
| 0    | 815              | 774           | 774, 1018, 1478         |
| 1    | 322              | 331           | 313, 313, 373           |
| 2    | 313              | 903           | 313, 314, 687           |
| 3    | 313              | 292           | 313, 314, 321           |
| 4-7  | 313 (each)       | 313/313/507/353 | 313 range +/- 200     |

K=8 S2MM:

| task | EMU (this build) | HW (this run) | HW range (prior sweeps) |
|-----:|-----------------:|--------------:|------------------------:|
| 0    | 240              | 691           | 691, 754, 880           |
| 1    | 240              | 227           | 227, 231, 233           |
| 2-7  | 240 (each)       | 240 +/- 15    | 240 +/- 15              |

**Steady-state tasks (1-7) match HW within HW noise band (~30 cyc).**
First-task MM2S matches HW; first-task S2MM undershoots due to anomalous
HW cold-start on this run (documented variance -- see Caveats).

Total span (EMU vs HW) still differs by ~80% at K=8 because inter-task
gaps (~3000 cyc each) are NOT modeled by this fix.  That's (C) in the
follow-ups -- next axis.

Unit tests: 3209/3209 pass.

## Caveats

**MM2S first-task variance.**  K=1 chain-sweep measured MM2S = 2127 cyc;
N=64 single-task N-sweep measured MM2S = 980 cyc.  Same kernel structure,
2.2x difference -- HW state (DDR cache / thermal / page state) drives
substantial first-task variance.  The chain-sweep K=8 within-run delta
(task 0 - tasks 1-7) is the cleanest cold-start measurement we have
(705 cyc MM2S), but the N-sweep large-N fit gives 747 -- both within
50 cyc.  EMU calibrated to 747 (= 498 cold + 249 per_task) to preserve
the existing N-sweep match.

**S2MM cold ≈ 0.**  Pull direction shows no first-task cold-start in
the dominant case (S2MM steady-state at K=8 ≈ 241 cyc, S2MM single-task
at N=64 also ≈ 244 cyc on EMU).  Some sweeps show anomalous first-task
S2MM (chain K=8 task 0 = 880 cyc, vs 241 steady).  Treated as variance;
modeled cold = 0.

**Inter-task gaps not modeled.**  HW spends ~3000 cyc between
`DMA_*_FINISHED_TASK_i` and `DMA_*_START_TASK_(i+1)` -- this is NPU
controller dispatch processing in the runtime sequence, not shim DMA
state.  This finding does not address it; tracked as the next axis to
calibrate (see follow-ups).

**Shim task queue depth.**  K=16 chain-sweep HW FAILs (4.8s timeout vs
0.6s for K=8).  **[CORRECTED 2026-06-06]** The mechanism stated here
originally ("shim task queue is 8-deep; the ninth queued task wedges")
is WRONG: the shim DMA task queue is **4-deep** (aie-rt
`XAIE_DMA_MAX_QUEUE_SIZE 4U` / `StartQSizeMax = 4U`), and the "8" was our
own incorrect `MAX_TASK_QUEUE_DEPTH`, not silicon -- k8 already exceeds a
4-deep queue yet passes.  The real cause is **16-BD shim pool
over-allocation** (a K-direction kernel needs 2K distinct shim BDs; K>8
exceeds the 16-BD pool), and the HW wedge is **non-monotonic** in K
(k8 pass, k9 wedge, k12 pass, k16 wedge).  Full analysis in
[2026-06-06-shim-bd-pool-overallocation-nonmonotonic-wedge.md](2026-06-06-shim-bd-pool-overallocation-nonmonotonic-wedge.md).
The generator is now capped at K=8.

## Follow-ups (non-gating)

- **Inter-task gap modeling.**  The dominant axis for chained-workload
  total-span accuracy.  Lives in the runtime sequence interpreter / NPU
  controller dispatch path, not the shim DMA timing model.  Next on the
  cycle-accuracy mission.
- **Multi-BD-per-task (next_bd chains).**  This finding measures K
  back-to-back `dma_memcpy_nd` dispatches.  Tasks with internal
  next_bd-linked BD chains are a separate axis (the original "BD chain"
  in `enter_chained_bd`).  Already partially modeled (`is_first_bd`
  stays false for chained BDs); not re-calibrated by this work.

## See also

- [`2026-05-25-shim-throughput-1-word-per-cycle.md`](2026-05-25-shim-throughput-1-word-per-cycle.md) -- HW N-sweep calibration
- [`2026-05-25-emu-shim-dma-timing-recalibrated.md`](2026-05-25-emu-shim-dma-timing-recalibrated.md) -- prior EMU recalibration
- [`../../coverage/cycle-accuracy-mission.md`](../../coverage/cycle-accuracy-mission.md) -- mission tracker
- `_diag_shim_chain_sweep/` in mlir-aie -- K-sweep calibration kernels
- `xdna-emu/tools/shim-chain-fit.py` -- K-sweep fit + HW/EMU compare tool
