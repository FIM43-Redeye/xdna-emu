# SP-5c Phase 3: the mid-column south-flood is silicon-confirmed -- the two-sided R1 spine is buildable

**Date:** 2026-07-02
**Issue:** #140 timer-sync arc, SP-5c Phase 3 (vertical anisotropy).
**Status:** Buildability gate PASSED on Phoenix. Nothing flipped (`calibrated` stays false).

## TL;DR

The Phase-3 two-sided R1 spine (falsify `d_vN != d_vS`) needs a single **mid-column**
broadcast source to flood **both** north and south through the fabric. That was the one
genuinely-open composite (the Fable review's load-bearing Q1: neither Phase-2 source was
mid-column). A cheap R3b-PC probe -- the validated bring-up kernel with `s2` relocated from
the top corner `(2,5)` to the middle of the col-1 vertical spine, `(1,3)` -- ran N=3 on
Phoenix. **The tile one south hop below the source, `(1,2)`, read a small deterministic
range-0 interval (119, 119, 119); the two north tiles and both horizontal tiles also read
clean range-0.** A mid-column source floods south. **Option B is buildable.** Unlike the
Phase-2 shim-row premise, this paper mechanism was *confirmed* by silicon, not refuted.

## The probe

Kernel `sp5_skew_r3b_pc_dvprobe` (mlir-aie, branch `xdna-emu-cycle-budget`, untracked): a
copy of the silicon-validated `sp5_skew_r3b_pc` with `s2` relocated by editing exactly two
`write32`s (`Event_Broadcast14` config `0x34048` + `Event_Generate` `0x34008`) from
`column=2,row=5` to `column=1,row=3`. Nothing else changed -- same counters, same OP_READ
readback, same s1=shim(0,0)/ch15. The existing col-1 vertical spine tiles now straddle the
source: `(1,2)` south, `(1,4)`/`(1,5)` north.

Readback is the R3b-PC interval `r_X = D(s2->X) - D(s1->X) + T0` on each tile's
`Performance_Counter0` (START=BROADCAST_15, STOP=BROADCAST_14). The interpretation is
binary and had a built-in negative control:

- **South floods** => `(1,2)` gets both floods => a clean, small, deterministic interval.
- **South confined** => `(1,2)` gets START but never STOP => the counter free-runs to a
  large **varying** value (the confined-`d_h` signature from the Phase-2 capture,
  2021/3400/2507).

## Result (Phoenix NPU1, N=3, all clean rc=0 / zero TDR / zero IOMMU)

| counter_index | tile | role | run1 | run2 | run3 | range |
|---|---|---|---|---|---|---|
| 0 | (0,3) | h_west | 123 | 123 | 123 | 0 |
| 1 | (1,3) | **SOURCE (ignored)** | 7506 | 5074 | 4717 | 2789 |
| 2 | (2,3) | h_east | 115 | 115 | 115 | 0 |
| **3** | **(1,2)** | **SOUTH -- the question** | **119** | **119** | **119** | **0** |
| 4 | (1,4) | north | 115 | 115 | 115 | 0 |
| 5 | (1,5) | north | 115 | 115 | 115 | 0 |

- **`(1,2)` (south): range-0 at 119.** The mid-column source floods south. **PASS.**
- All four flooded neighbours (south + 2 north + 2 horizontal) read clean range-0 in a tight
  115-123 band -> the source floods **omnidirectionally** through the fabric.
- **The source tile `(1,3)` free-runs to a large varying value (7506/5074/4717).** This is
  the negative control working exactly as designed: a tile without a clean START->STOP
  interval reads large + varying, making the surrounding tiles' range-0 unambiguous.

## What it settles (and what it doesn't)

**Settles (Fable Q1, the buildability gate):** a single mid-column source floods both
vertical directions through the fabric -- the one open composite. The other two primitives
of the two-sided R1 mechanism (core-sourced flood; `Timer_Control.Reset_Event` off a
broadcast, i.e. `XAie_SyncTimer`'s own mechanism) were already silicon-proven. So the
two-sided R1 spine (Phase-3 Option B) is **GO** to build.

**Does not settle (not this probe's job):** the R1-specific readout path (reconfigure the
measured tiles' timer reset onto the synthetic mid-column broadcast, read origins via
decoded trace) is exercised end-to-end only when the R1 kernel is built. This probe used the
R3b-PC perf-counter readout, which isolates the *topology* question. Per the Fable framing,
topology was the only open composite; the readout path is standard.

**Bonus (not a measurement):** the intervals cluster in a tight 115-123 band across all
directions. These fold both floods + `T0`, so they are **not** clean per-direction `d_v` --
but the absence of any *gross* asymmetry (south 119 vs north 115) is an early, informal
gross-anisotropy floor consistent with the isotropic expectation. Do not read it as a value.

## Provenance

- Silicon: `build/experiments/sp5-skew/dvprobe_n3.log` + `dvprobe/run_*/counters.bin`
  (3 clean runs, N=3, 2026-07-02). Gate script `build/experiments/sp5-skew/dvprobe_gate.sh`
  (gitignored).
- Kernel: `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc_dvprobe/` (untracked; two-`write32`
  relocation of the validated `sp5_skew_r3b_pc`, insts.bin rebuilt, emu smoke clean
  `halt_reason=completed`).
- Plan: `docs/superpowers/plans/2026-07-02-sp5c-phase3-r1-two-sided-spine.md` Sec. 7 step 0.
- Fable review (Q1): recorded in that plan's Sec. 9.
