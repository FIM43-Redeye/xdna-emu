# Our interpreter over-emits LOCK_STALL 375x vs HW (hardcoded trace period)

**Date:** 2026-06-07
**Kernel:** `vec_mul_trace_distribute_lateral` (peano), tile(2,1) core trace
**Method:** ran the SAME peano xclbin through our interpreter (EMU-side bridge,
`XDNA_EMU=1`, interpreter backend) with trace enabled, decoded with the same
`parse_trace`/`parse.py` path as the HW + aiesim oracles, diffed event counts.
Artifacts in `build/experiments/bcast-bridge/`: `run_distlat_emu.sh`,
`trace_emu.{txt,json}`, `run_emu.log`; HW reference `trace_hw.json`.

## The measured gap (our interpreter vs real NPU1)

| Event (B-phase count) | Ours | HW | aiesim | Verdict |
|---|---|---|---|---|
| INSTR_EVENT_0 / _1 | 4 / 4 | 4 / 4 | 4 / 4 | exact |
| INSTR_LOCK_ACQUIRE_REQ / _RELEASE_REQ | 9 / 9 | 9 / 9 | 9 / 9 | exact |
| cycles per invocation (EVENT_0->EVENT_1) | 12298 | 12297 | 12297 | off-by-1, negligible |
| **LOCK_STALL** | **7112** | **19** | 3 | **over-emit ~375x** |
| PORT_RUNNING_0 | 1 | 6 | 6 | under-segment |
| PORT_RUNNING_1 | 4 | 5 | 4 | miss one |
| DMA_S2MM_0/1_STREAM_STARVATION | 1 / 1 | 2 / 2 | 3 / 2 | miss one edge each |

Key contrast: **aiesim *under*-counts LOCK_STALL (3), we *over*-count (7112).**
Our gaps are NOT aiesim's gaps -- measuring our own model was essential.

## Root cause: `LOCK_STALL_TRACE_PERIOD = 1`

`src/interpreter/core/interpreter.rs:101`. The held-stall path
(`try_resume_stall`, ~line 737) re-emits a LOCK_STALL every
`LOCK_STALL_TRACE_PERIOD` cycles while the core sits in `WaitingLock`. With the
period at **1**, we emit one LOCK_STALL per stall-cycle.

Confirmed from the decoded timeline:
- 7100 of 7112 events are spaced **2 ns apart** (one per cycle).
- **All 7112 fall OUTSIDE the compute windows** (EVENT_0->EVENT_1) -- entirely in
  the idle gaps where the core waits for the next input batch (long
  pre-first-invocation fill + short inter-invocation gaps). Compute-region trace
  is clean.

### History: a hardcoded-constant tug-of-war
- `2d84e55` added periodic emission at `PERIOD = 1024`, calibrated to "HW's
  perf-counter-driven cadence (PERF_CTRL0 counting ACTIVE_CORE, threshold 1024)"
  -- finding `2026-05-05-355-cycle-divergence-diagnosis`.
- `bcea78f` changed it to `1` ("emit LOCK_STALL per polling cycle (was 1024)").

distribute_lateral's HW shows **19 = edge events only** (1 startup stall + 18
pulses, one per lock transaction: 9 acquire + 9 release). Neither `1` nor `1024`
reproduces that. **A fixed period cannot be right for all kernels because the
behavior is config-dependent:** HW emits periodic samples only when the kernel's
trace/perf-counter config programs a cycle-driven sample event at a threshold.
distribute_lateral programs none, so HW = edges only. The 1024<->1 flip-flop is
two calibrations of the same wrong knob against two different kernels.

## Fix direction (derive from config, per the cardinal rule)

Periodic trace emission must be **gated on the actual trace/perf-counter
configuration** -- whether a cycle-driven sample event is programmed, and at what
threshold -- not a hardcoded constant. Then both kernels fall out of one model:
distribute_lateral (no counter -> edges only -> 19) and the 1024-calibrated
kernel (counter programmed -> periodic).

**RESOLVED (2026-06-07).** HW LOCK_STALL is a traced **level signal**, edge-emitted
(B on rising / E on falling), NOT periodic and NOT config-dependent. Proof:
re-decoding the current `add_one` HW capture (`build/bridge-test-results/20260606/
..._add_one_instrumented.peano.hw/events.json`) with the same `ours` decoder gives
**LOCK_STALL = 46** (HW) vs **4449** (EMU) -- and add_one *has* a perf counter
programmed (PERF_CNT_2 = 12) yet stays at 46 edges, so it is not config-driven.
The 2026-05-25 "2233-2766, period=1 correct" was a since-fixed decoder bug
(expanded HW's held-level skip-token runs into phantom per-cycle events). The
distribute_lateral HW decomposition is conclusive: 19 = 1 startup wait + 2 genuine
waits + 16 per-arbitration 1ns spans (9 acquire + 9 release transactions).

Fix is NOT "delete periodic" alone -- our LOCK_STALL is a per-cycle **pulse**
event (each emission -> isolated 1ns span), so even one point event at entry
gives a 1ns span, not a span covering the real wait. The fix is to model
LOCK_STALL as a **held level** (assert at arbitration / wait-entry, deassert at
resolution), emitted on transition like HW. Full design:
`docs/superpowers/specs/2026-06-07-lock-stall-level-emission-design.md`.

## Cautions
1. Do not just flip `1 -> 1024`; that re-loses whatever `bcea78f` fixed. The fix
   is the config-derived (or edge-only) model, calibrated against re-measured HW
   for BOTH kernels.
2. **PORT_RUNNING (1 vs 6) and STREAM_STARVATION (1 vs 2) are one shared root
   cause** and a DIFFERENT axis: our DMA delivers input too smoothly, so the
   stream port never cycles idle->active the 6 times HW does and we miss the
   second starvation edge. Bundle these with the deferred DDR-fill-latency /
   bursty-delivery model -- NOT this LOCK_STALL work.
