# SP-5c Phase 3: vertical anisotropy is blocked on the SP-4a cold-start fill gap (not a ceiling)

**Date:** 2026-07-02
**Issue:** #140 timer-sync arc, SP-5c Phase 3.
**Status:** Two-sided R1 spine (Option B) NOT built -- its blocker is a known, tracked,
fixable emulator fidelity gap. Vertical anisotropy is deferred-as-blocked, not conceded.

## TL;DR

Running the existing single-sided R1 (Option A) for a `d_v` *value* returned `−73.5`
with per-hop skews `−54/−93/−147` -- ~30x the real reset `d_v≈2` (R3b, clean,
Delta_wall-free). Diagnosis: **R1's "skew" is dominated by the emulator's Delta_wall
prediction error, not the reset skew.** And that error is not a new gap or a measurement
ceiling -- **it is the already-tracked SP-4a cold-start fill-state gap** (`known-fidelity-
gaps.md`, the "Inter-tile send-port cadence" row + its SP-4a sub-finding). The match is
exact: SP-4a records `prod->consA first-LOCK_STALL +2 HW vs −52 EMU`; the R1 row2(Prod)
->row3(ConsA) skew is `−54`. Same tiles, same event, same number. The R1 anisotropy probe
literally re-measured the −52 offset.

## Quantified diagnosis (run_01, N=3 gate clean)

- **The emu mispredicts cross-row dataflow wall-times.** On HW the three spine rows' first
  LOCK_STALL fire within ~4cy (near-simultaneous; the real reset skew is only ~2cy/hop,
  consistent with R3b's `d_v=2`); the emu spreads them by ~150cy. First-occ emu error:
  row3 `−56cy`, row4 `−151cy`.
- **LOCK_STALL is a bad anchor** -- HW fires ~2x more than the emu on consumer rows (row3
  `78 vs 40`, row4 `103 vs 54`), a lock-arbitration/fill fidelity gap; unfaithful at
  cold-start AND steady-state.
- **DMA-BD anchor is better but still not clean.** Steady-state (late-occurrence)
  DMA_MM2S_FINISHED_BD skews converge to small stable values (`−5/−8`, `+10`) -- a
  ~15-20x improvement over cold-start -- but the residual steady-state emu Delta_wall
  error is ~7-8cy/hop, still >the `d_v≈2` signal.

## Why this blocks the anisotropy split (and why it is not a permanent ceiling)

- **R3b (clean, Delta_wall-free) structurally cannot split N/S** (identifiability theorem;
  its two-source interval yields the SUM `d_vN+d_vS`, confirmed by the buildability probe:
  south−north = 4 = `d_vN+d_vS` = 2x symmavg).
- **R1 (direction-capable) needs a faithful Delta_wall**, but the best achievable floor
  (~7cy steady-state) exceeds the `d_v≈2` signal, so it cannot resolve a ~cycle-scale
  split. Building the two-sided R1 spine (Option B) would inherit the same ~7cy floor --
  **it cannot beat what Option A already measured**, so it was not built.
- **BUT the floor is the SP-4a cold-start fill-state gap -- a fixable emulator bug, not a
  fact of nature.** Closing it (the tracked send-cadence / device-model-audit resume
  target) drops the Delta_wall floor below `d_v` and re-enables the split as a byproduct.

**The convergence:** measuring vertical anisotropy faithfully and closing the cold-start
fill-state gap are the *same work*. The ~50cy cold-start Delta_wall error dwarfs the ~2cy
skew -- so the fill gap is the larger cycle-accuracy target, and its closure is the unlock.

## Disposition

- **Reset `d_v≈2`/hop stands** (R3b free-flood, clean; gate 5 green; HW-side reconstruction
  corroborates). This is the value the SP-5c flip uses.
- **Vertical anisotropy = BLOCKED on the SP-4a cold-start fill gap**, tracked (not conceded).
  Gross anisotropy (>~7cy split) is bounded out; sub-~7cy is unresolvable until the gap closes.
- **Option B (two-sided R1 spine) not built** -- same floor, no gain.
- **The SP-5c flip proceeds** on measured `d_h≈4`/`d_v≈2` + disclosed assumptions (vertical
  isotropy among them, now with a quantitative bound and a named blocker).
- **Next major cycle-accuracy campaign** (post-flip, Maya's sequencing): the cold-start
  fill-state gap (`known-fidelity-gaps.md` send-cadence row; resume hypothesis already
  written), now doubly-motivated -- largest Delta_wall error AND the anisotropy unlock.

## Provenance

- Silicon: `build/experiments/sp5-skew/task6/` (R1 N=3, gate clean), `r1_optionA_n3.log`.
- The gap: `docs/known-fidelity-gaps.md` "Inter-tile send-port cadence" row + its SP-4a
  sub-finding `findings/2026-06-29-coldstart-headstart-trace-baseline.md`.
- Plan: `docs/superpowers/plans/2026-07-02-sp5c-phase3-r1-two-sided-spine.md`.
