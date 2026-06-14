# HW DDR-delivery stochasticity characterization (task #140)

**Date:** 2026-06-14
**Question (Maya):** is HW DDR delivery deterministic? If so, the burst-delivery
model could reproduce HW exactly, not just ballpark.
**Answer:** No -- DDR-boundary events are genuinely stochastic and *memoryless*.
But there is a deterministic compute skeleton, and end-to-end runtime is stable.

## Method

20 HW captures of the same binary (`add_one_using_dma`, chess), same boot
session, back-to-back (`build/experiments/ddr-stochasticity/run_01..20.json`,
`analyze.py`). Per event family: count distribution, run-order structure (lag-1
autocorrelation, linear drift vs run index, cold-first z-score), and
per-position timing jitter for count-stable events.

## Results (n=20)

| event | module | mean | std | CV% | range | autocorr | drift/run | run1 z |
|---|---|---:|---:|---:|---|---:|---:|---:|
| DMA_S2MM_0_STREAM_STARVATION | shim | 28.1 | 2.7 | 9.6 | 20-33 | -0.03 | -0.04 | 0.3 |
| DMA_S2MM_1_STREAM_STARVATION | shim | 33.6 | 2.0 | 5.8 | 29-38 | 0.23 | 0.00 | 0.2 |
| PORT_RUNNING_0 | memtile | 11.0 | 0.0 | 0.0 | 11-11 | - | 0 | 0 |
| PORT_RUNNING_1 | memtile | 16.0 | 0.0 | 0.0 | 16-16 | - | 0 | 0 |
| PORT_RUNNING_4 | memtile | 16.7 | 0.5 | 2.7 | 16-17 | -0.14 | 0 | -1.7 |
| PORT_RUNNING_5 | memtile | 8.0 | 0.0 | 0.0 | 8-8 | - | 0 | 0 |
| LOCK_STALL | core(m1) | 3027 | 748 | 24.7 | 1960-5210 | -0.20 | -37 | 0.4 |
| PERF_CNT_2 | core(m1) | 11.3 | 0.9 | 8.0 | 11-14 | -0.12 | -0.02 | -0.3 |

S2MM_0 series: `29,28,24,29,31,27,26,30,27,27,33,29,31,28,31,28,28,20,28,28`

- **Total span (max soc):** mean 344523, std 6833, **CV 2.0%**.
- **Timing jitter, count-stable PORT_RUNNING:** per-position soc std max=6cy
  (PR_0), 1cy (PR_1), 1cy (PR_5).

## Interpretation

The system decomposes into three regimes:

1. **Deterministic compute/on-chip skeleton.** `PORT_RUNNING_{0,1,5}` counts are
   *exactly* constant across all 20 runs (std 0), and their event timing barely
   moves (<=6 cycles). The kernel's compute structure -- loop iterations, memtile
   buffer fills, on-chip port toggles -- is reproducible. This is matchable
   *exactly* by the emulator.

2. **Memoryless-stochastic DDR-boundary events.** `STREAM_STARVATION` and
   `LOCK_STALL` jitter run-to-run with CV ~6-25%. Crucially the noise has **no
   exploitable structure**: lag-1 autocorrelation ~0, no drift across runs, run 1
   is not an outlier (z~0). It is not per-boot-fixed (all runs differ in one
   session), not cold-start (run 1 normal), not thermally drifting. It is genuine
   IID-ish noise from DDR controller arbitration/refresh timing. **No hidden
   determinism to exploit.**

3. **Stable end-to-end runtime.** Total cycles vary only CV 2.0% -- the kernel
   finishes in ~the same time every run. The DDR jitter reshuffles *when* the
   internal stalls/starvations land, not the aggregate work.

## Consequences

**For the burst-delivery model (#140 calibration).** Exact reproduction is
impossible because HW itself is not reproducible on these events. The right
target is the **distribution mean**: aim the emulator's (deterministic) output
at S2MM_0~=28, S2MM_1~=34 (EMU-off baseline 20/29). The model stays
deterministic; we center it on the HW mean and accept it lands as a point inside
the HW spread.

**For the trace-comparison pipeline (Maya's forward note).** The comparator
should classify events by *regime* and tolerance accordingly:
- Deterministic events (PORT_RUNNING counts, compute structure) -> exact match
  expected; a mismatch is a real bug.
- Stochastic DDR-boundary events (STREAM_STARVATION, LOCK_STALL) -> a **tolerance
  band** (e.g. HW mean +/- 2 std, or "EMU within observed HW range") is CLEAN;
  only an out-of-band value is DIVERGE. Comparing a stochastic event to a single
  HW sample (as today) manufactures phantom divergence ~half the time.
  This needs the HW *distribution* (multi-run), not a single capture -- a future
  pipeline change, scoped when we get there.

This is the same axis as known-fidelity-gaps row 50/117 and the aiesim
"compute-region cycle-EXACT, DDR fill latency under-modeled" finding -- now
quantified: the under-modeled axis is not just optimistic, it is *stochastic*,
so the honest goal is mean-calibration + band-comparison, not point-matching.

## Artifacts

`build/experiments/ddr-stochasticity/` -- 20 HW captures (`run_*.json`),
`capture_more.sh`, `analyze.py`.
