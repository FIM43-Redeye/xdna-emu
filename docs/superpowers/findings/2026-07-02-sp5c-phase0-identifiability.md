# SP-5c Phase 0: Identifiability Proofs + IF Decisions (#140)

**Date:** 2026-07-02
**Status:** Phase-0 theory lock-down for SP-5c. Pure software; no `calibrated` flip.

## Decoupled captures (the load-bearing decision)
- `d_h` <- block-replicated / shim-row capture, scoped `d_h`-ONLY.
- `d_v` <- free-flood R3b (straddling sources) + R1.
- Proof (executable): `tools/test_skew_r3b_identifiability.py`
  - `test_reset_routed_vertical_term_is_constant_dv_collapses`
  - `test_block_routed_capture_is_rank_deficient_for_dh_dv`
  - `test_free_flood_straddling_capture_identifies_dv`
- Algebra: reset routing gives interval vertical coefficient `(s2.row - s1.row)`,
  constant across tiles -> `d_v` unidentifiable from a block capture. Horizontal
  coefficient `|s2.col - t.col| - |s1.col - t.col|` varies -> `d_h` identifiable.

## IF decisions
- IF-1 (path fidelity): `d_h` MUST be measured on the shim-row path (block
  replication or shim-row tiles); bring-up `d_h~=4` was the AIE-row path.
- IF-2 (linearity): tested by enriched geometry residuals in Phase 2 (accessible
  cols only; power ceiling acknowledged).
- IF-3 (arrival->latch uniformity): uniform -> cancels; per-kind -> folds into the
  gauge intra-offset. Disclosed assumption.

## Determinism basis
XDNA is globally clocked (single clock domain); broadcast transport deterministic;
async only at NoC egress (`grounding.py:is_async_cdc`). Range-0 is predicted.
Full argument: design Sec.1.5.

## Magnitude / tolerance estimate (not a kill gate)
Per-hop skew is single-digit cycles (bring-up `d_h~=4`, `d_v~=2`; mlir-aie
benchmark corroborates ~4/col-hop horizontal via 2cy x 2 modules, ~2/tile
vertical). Array-span skew is tens of cycles; `Delta_wall` is hundreds. Tolerances
for the Phase-5 held-out gate must sit strictly above the held-out kernel's known
`Delta_wall` residual (design Sec.5).

## Ceiling components (disclosed, not measured)
E/W anisotropy; absolute intra-offset (gauge); `d_h^{ch15}` in isolation;
per-module horizontal split; clock-tree phase skew; structured OR-tree asymmetry.
Toolchain sweep dispositions: design Sec.4.
