# R3b-PC Silicon Bring-Up: the apparatus works on Phoenix (#140 SP-5b)

**Date:** 2026-07-01
**Status:** SP-5b R3b-PC apparatus SILICON-VALIDATED (bring-up capture). All three
HW-gated unknowns resolved positively. This is a **bring-up datapoint, NOT a
calibrated skew** — no `calibrated` flip, no constant set, no number asserted.
The interpretation + validation + flip are SP-5c.

## What was run

- **Kernel:** `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/aie.mlir` (branch
  `xdna-emu-cycle-budget`, commit `e98c5420e`). Apparatus merged to xdna-emu
  `master` at `57c58ac0`.
- **Capture:** `build/experiments/sp5-skew/r3b_pc_gate.sh 3` — 3 serial runs on
  real Phoenix NPU1 (`env -u XDNA_EMU`), preceded by `xrt-smi validate` (exit 0,
  device healthy).
- **Geometry** (`geometry.json`): two floods, `s1=(0,0)` on broadcast ch15
  (`BROADCAST_15`=122), `s2=(2,5)` on ch14 (`BROADCAST_14`=121); six measured
  cores arming `Performance_Counter0` START=122 STOP=121. `dn_h = |col-2|-|col|`,
  `dn_v = |row-5|-|row|`.

## Result

All 3 runs clean: `rc=0`, `tdr_delta=0`, `iommu_delta=0` (no wedge, no fault).
Raw `Performance_Counter0` readback (counter_index order), **identical across all
three runs** (range-0):

```
[137, 129, 121, 133, 125, 121]
```

Fit through the merged `{d_h, d_v}` extractor: **`d_h ≈ 4`, `d_v ≈ 2`** (raw
counter-cycle units), `fit_residual ≈ 7e-15`. The fit is exact against the
geometry with `const = 131`:

| ci | tile | dn_h | dn_v | predicted `131 + 4·dn_h + 2·dn_v` | measured |
|----|------|------|------|-----------------------------------|----------|
| 0 | (0,3) | +2 | -1 | 137 | 137 |
| 1 | (1,3) |  0 | -1 | 129 | 129 |
| 2 | (2,3) | -2 | -1 | 121 | 121 |
| 3 | (1,2) |  0 | +1 | 133 | 133 |
| 4 | (1,4) |  0 | -3 | 125 | 125 |
| 5 | (1,5) |  0 | -5 | 121 | 121 |

## The three HW-gated unknowns — all resolved

1. **Does a control-packet OP_READ of a perf-counter register (`0x31520`) return
   the live value on Phoenix?** **YES** — non-zero live values. **Fallback (b)
   (core `LDA → DMA out`) is NOT needed.** This was the linchpin unknown.
2. **Do the ch14/15 broadcast floods arrive as events that drive the counter?**
   **YES** — the counters vary by tile in the geometry-predicted way.
3. **Do the six one-word responses arrive at the shim S2MM in issue (=
   counter_index) order?** **YES** — the six values fit the geometry structure at
   **zero residual**; a reorder would have broken the fit. **Fallback (c) is NOT
   needed.**

## What this is NOT

**This is not a calibrated skew measurement.** `d_h≈4`, `d_v≈2` are raw per-hop
*interval-differential* constants in counter-cycle units; they fold clock-skew
together with broadcast-network (stream-switch + wire) latency. Disentangling the
skew component, reconciling signs, resolving the structure questions on richer
geometry, and validating causal-vs-HW are **SP-5c**. SP-5b's contract holds: the
apparatus **runs and reproduces**; it produces no number and no evidence any
number is the true skew. Nothing was flipped; the printed `d_h/d_v` are
visibility-only.

## What it vindicates / de-risks

- **The instrument-role flip (R3b primary over R1 for `d_h`).** R3b's *local
  single-clock* perf-counter interval came back integer-exact and range-0 — none
  of the ~30-cycle cross-column trace jitter that defeated R1's `d_h`. The
  jitter-immunity that motivated making R3b primary is confirmed on silicon.
- **First positive model-structure data point.** Six tiles fit a single
  `{d_h, d_v}` + const at zero residual → per-hop uniformity and cross-axis
  additivity *hold on this geometry*. This is a **fit, not yet a falsification**:
  the geometry is minimal (3 collinear per axis). SP-5c needs richer geometry to
  actually try to break the shape.
- **The whole readback path is silicon-proven**, so SP-5c can be scoped from a
  position of knowing the plumbing works — no fallback detour.

## Fragility noted

`r3b_pc_gate.sh` clears `run_*/` on each invocation, so this capture's raw
`counters.bin` files are clobbered by the next run. The values are preserved in
this finding (the durable record); the capture is deterministic (range-0), so it
reproduces on demand.

## SP-5c scope preview (not started — Maya-gated, one-way at the flip)

1. **Richer geometry to FALSIFY the structure** (per-hop uniformity, cross-axis
   additivity) — not just fit it. Add off-axis / longer-baseline tiles.
2. **Skew-vs-latency disentangling** — decide what fraction of the per-hop
   interval-differential is clock skew vs broadcast propagation, and how the
   model consumes it.
3. **Pre-flip gates** (SP-4b design §9a): held-out validation kernel, joint sign
   anchors, cross-column b-vector gate.
4. **Then** measure constants, flip `calibrated`, update the 3 regression guards,
   validate causal-vs-HW.

## Provenance

Bring-up capture on the hardware we own (Phoenix NPU1), the ground truth. The
apparatus was built + emulator-verified first (which caught three encoding/routing
bugs before silicon: parity polarity, readback packet format, packet-ID width);
this is its first hardware run, and it ran clean on the first try.
