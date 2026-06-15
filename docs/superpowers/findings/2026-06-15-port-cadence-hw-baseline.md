# Memtile port-cadence HW baseline — variance-aware calibration target (#140)

**Date:** 2026-06-15
**Kernels:** `add_one_using_dma` (primary), `_diag_shim_chain_sweep/k{1,2,4,8}`,
`vec_vec_add_memtile_init`
**Method:** `tools/multirun-trace-campaign.py --kernels ... --n-runs 15` (real
NPU1, FW 1.5.5.391), aggregated by `tools/port-cadence-baseline.py`. Session:
`build/experiments/port-running-baseline/baseline-2026-06-15/` (90/90 OK, 44s).

## Why

`docs/known-fidelity-gaps.md:50` ("Held-level count under-emission / bursty
delivery") and finding `2026-06-14-port-running-under-emission.md` scoped the
memtile receive-port cadence gap (slot0 3-vs-6, slot4 5-vs-8) as a DMA-delivery
*calibration* axis: "calibrate against HW, not chase in the trace encoder." The
derivable part (the `cycle_beat` recv-port fix) already landed. Closing the
residual needs a **trustworthy, variance-aware HW target** before touching the
broad DMA-timing constants (the BurstGate class we reverted on a bad single
calibration). This baseline is that target.

## `add_one_using_dma` — the calibration target (15 runs)

| event port | role | HW mean | std | distinct | verdict | EMU now |
|------------|------|--------:|----:|----------|---------|--------:|
| PORT_RUNNING_0 | S2MM <- shim (recv)    | 5.73 | 0.44 | [5,6]   | stochastic ±1 | 3 |
| PORT_RUNNING_1 | MM2S -> compute (send) | 8.00 | 0.00 | [8]     | **deterministic** | 8 ✓ |
| PORT_RUNNING_4 | S2MM <- compute (recv) | 7.07 | 0.25 | [7,8]   | stochastic ±1 | 5 |
| PORT_RUNNING_5 | MM2S -> shim (send)    | 4.00 | 0.00 | [4]     | **deterministic** | 4 ✓ |

Modal sub-burst gap patterns (the cadence shape, `--gap 2`):
- **slot0** (6 bursts, 10/15 runs): `[16, 16, 7, 9, 13]`
- **slot4** (7 bursts, 7/15 runs): `[8, 9, 14, 38, 65, 73]`
- slot1 (8, det.): `[67, 75, 66, 75, 66, 74, 68]`  (one per ~70-cy compute iter)
- slot5 (4, det.): `[143, 140, 143]`

## Three load-bearing conclusions

1. **Send ports are deterministic and EMU already matches them exactly**
   (slot1=8, slot5=4). The gap is **only** the receive ports, and the receive
   ports are **mildly stochastic (±1)**. The calibration target is therefore a
   *distribution* (slot0 6±1, slot4 7±1), not a fixed count — validate within
   the band (mean ± k·std), not by exact match. This refines the 6/14 finding's
   "deterministic, std 0" (too few runs); structural cadence carries ±1 jitter
   that propagates one level from the DDR arrival.

2. **The aie-rt mechanism is confirmed in silicon.** slot4's early gaps
   `[8, 9, 14]` are the 8-word objfifo-consumer sub-burst micro-stalls (compute
   consumes `in1_cons_buff` 8xi32, double-buffered); the later `38/65/73` are
   core-processing waits. So HW breaks a 16-word memtile MM2S BD into 8-word
   sub-bursts gated by the downstream compute lock handshake — exactly the
   lock-gated-delivery model the Explore dive derived. EMU streams the BD
   continuously because the memtile receive FIFO (depth 2-4) absorbs the
   compute-side per-BD bubble.

3. **The shim STREAM_STARVATION is the dominant stochastic axis** (slot0-side
   std ~4, distinct 15-28) — memoryless DDR arrival jitter, cleanly separable
   from the structural PORT_RUNNING cadence. Do not try to make this
   deterministic; the band-sigma comparator is the right tool.

## Chain-sweep scaling (clean slot0/slot4 delivery probe)

`_diag_shim_chain_sweep/k{1,2,4,8}` (single self-chained shim BD ×K):
slot0 ≈ K (k1→1, k2→2.3, k4→4.1, k8→8.3; mildly stochastic from k2 up),
slot4 ≈ 2K (k1→2, k2→4, k4→8, k8→15.3), shim STREAM_STARVATION ≈ 7K (stochastic).
Confirms slot0 cadence tracks delivery-chain length structurally.

## Calibration plan (next)

Target: lift EMU recv-port cadence into the HW band — slot0 3→6±1, slot4 5→7±1 —
by surfacing the 8-word sub-burst backpressure that the receive FIFO currently
absorbs, **without** disturbing the deterministic send ports (slot1/slot5) or
regressing the 15 honest-CLEAN kernels / chain sweeps. Gates:
- **Keep green:** send-port cadence (slot1=8, slot5=4) exact; full lib suite;
  the 15 CLEAN kernels via current-comparator re-check against preserved HW.
- **New target:** recv-port interval counts within HW band (this baseline) for
  add_one + chain sweeps, validated EMU-vs-HW with the band-sigma comparator.
- **Lesson from BurstGate:** any broad DMA-timing constant change is validated
  against this multi-run baseline (mean±std), never a single capture.
