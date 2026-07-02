# R3b Two-Source Identifiability Limit (#140 SP-5b)

**Date:** 2026-07-01
**Status:** OPEN design decision. Blocks R3b-PC Task 3 (enriched geometry). Found
during TDD execution of the R3b-PC plan; the rank-sufficiency guard test failed
`RankDeficientError: rank 2 < required 5` before any hardware was touched.

**One line:** The R3b two-flood interval method **cannot** identify within-axis
direction anisotropy (`d_hE` vs `d_hW`, `d_vN` vs `d_vS`) from two point sources
at *any* placement — so the soundness audit's Q1 remediation ("enriched geometry
makes isotropy falsifiable") is partly unrealizable as specified in the rev3
design docs and the R3b-PC plan.

---

## 1. The theorem

R3b measures, per tile X, the two-flood interval `r_X = D(s2,X) - D(s1,X)`, where
`D(s,X)` is the broadcast path cost from source `s`. The observation bridge
(`r3b_observe.py`) decomposes each path into monotone hop counts
`(e,w,n,s) = hops(src, tile)` with `e = max(tile.col - src.col, 0)`,
`w = max(src.col - tile.col, 0)`, etc., and emits the signed design-row
coefficients `a_hE = e2-e1`, `a_hW = w2-w1`, `a_vN = n2-n1`, `a_vS = s2h-s1h`.

For **any** source `s` and tile X, the identity `max(dx,0) - max(-dx,0) = dx`
gives `e - w = tile.col - src.col` unconditionally. Therefore:

```
a_hE - a_hW = (e2 - w2) - (e1 - w1) = (tile.col - s2.col) - (tile.col - s1.col)
            = s1.col - s2.col           # CONSTANT for every tile, any placement
```

and likewise `a_vN - a_vS = s1.row - s2.row`. So the vectors `(1,-1,0,0,0)` and
`(0,0,1,-1,0)` in `(d_hE, d_hW, d_vN, d_vS, d_turn)` space lie in the design
matrix's null space **for any two-source geometry**: after differencing against a
reference tile, the East and West columns become identical, as do North and South.
The rank ceiling is **3**, before tile placement is even considered.

With sources at **opposite corners** of the array (the plan's `s1=(0,0)`,
`s2=(2,5)` on `npu1_3col` = 3 cols x 6 rows), every physical tile has fixed hop
signs, and the turn coefficient collapses to a linear function of the others:

```
a_turn = (2 - tile.col)*(5 - tile.row) - tile.col*tile.row
       = 5*a_hW + 2*a_vS - 10        # a THIRD null direction
```

leaving **rank 2** — exactly what the guard test observed, confirmed by brute force
over the full `[0,2] x [0,5]` grid. Because the array *is* that grid (no off-array
tile exists), **no tile addition can raise the rank**. This is an algebraic
property of the two-source hop model, not an under-spanned geometry the plan's
"add a spanning tile" remedy could fix.

## 2. What R3b two-flood intervals CAN and CANNOT identify

| Quantity | Identifiable from 2-source interval? | Note |
|---|---|---|
| `d_h` vs `d_v` (**cross-axis**) | **YES** | Recovered as per-axis sums; the physically plausible asymmetry (horizontal vs vertical hops route/space differently) |
| `d_hE` vs `d_hW`, `d_vN` vs `d_vS` (**within-axis direction**) | **NO — fundamentally** | 2 null directions for any 2-source placement (Sec.1) |
| `d_turn` (cross-axis **additivity**) | Only with **non-corner** sources | Corner sources add a 3rd null direction; a source placement with tiles on both sides could expose it |
| `d_h^{ch1}` vs `d_h^{ch2}` (**per-channel** blend) | Separate concern | Needs the channel-control mechanism (read same tile armed on different channels), which the position-only bridge does not yet model |

**Physics reading of the within-axis gap.** A per-hop broadcast delay going North
vs South (or East vs West) is the same stream-switch + wire latency traversed in
opposite directions; by hardware symmetry it is very likely equal. Within-axis
direction isotropy is therefore a *defensible physical assumption* — the audit was
right that it was unremarked, but wrong to imply the two-flood instrument could
falsify it. The honest posture (audit's own "don't launder assumed as measured"
ethos) is to mark within-axis isotropy **ASSUMED and structurally unmeasurable by
R3b**, not to pretend enriched geometry tests it.

## 3. Downstream contamination of already-built work

- **rev3 design docs** (`2026-06-30-sp5b-kernel-hw-bringup-design.md` Sec.5.1/5.2/
  Sec.13; `2026-06-30-sp5b-measurement-apparatus-design.md` Sec.2/5.4) prescribe
  the signed N/S + E/W + interaction solver columns as the isotropy/additivity
  falsification hook. The N/S and E/W split is not realizable by this instrument.
- **The R3b-PC plan** (`docs/superpowers/plans/2026-07-01-sp5b-r3b-pc-enriched.md`)
  Task 1/2/3 build the 5-param solver + bridge + the failing guard.
- **Committed WIP** on branch `feat/sp5b-r3b-pc-software` (pushed): Task 1
  (`r3b_extract.py` 5-param) and Task 2 (`r3b_observe.py`) are green, but Task 1's
  `test_recovers_all_five_params` passes only on **hand-synthesized** coefficient
  rows that violate the `a_hE - a_hW = const` identity — data the real bridge can
  never emit. The solver is correct in the abstract but cannot be fed identifiable
  input from any real 2-source geometry. Task 3 was reverted (uncommitted);
  the branch sits clean at Task 2.

The soundness-audit finding (`2026-07-01-sp5b-soundness-audit.md`) Q1 remediation
and both rev3 docs carry an errata pointer to this document.

## 4. The decision (RESOLVED 2026-07-01: Option 1 + R1 reallocation)

1. **Accept the rank-2 reality.** Revise the solver to `{d_h, d_v}` (+ a turn term
   only if non-corner sources are used for additivity), document within-axis
   isotropy as an assumption R3b cannot test, and roll that correction back through
   the rev3 docs and the plan. Cleanest and buildable; concedes the audit's Q1
   over-reach. The channel-uniformity control (a distinct concern) still stands.
2. **Add a genuine third source** (or a working per-direction channel mechanism) to
   separate within-axis directions — a real instrument redesign, larger scope,
   breaks the two-source assumption baked into `observe_r3b`'s `sources["s1"/"s2"]`
   and `_hops`. Only worth it if within-axis anisotropy is judged physically real
   enough to measure rather than assume.
3. **Hybrid:** keep `{d_h, d_v}` from R3b, and if within-axis anisotropy must be
   checked at all, do it with a separate targeted experiment rather than folding it
   into the primary instrument.

**Recommendation (Opus, this session):** Option 1. Within-axis direction isotropy
is physically defensible by symmetry and structurally unmeasurable by R3b; forcing
a third source to measure a delay that is almost certainly symmetric spends a
scarce Phoenix redesign on a near-certainty. Concede the audit's Q1 over-reach
explicitly, keep the honest provenance.

**Resolution (2026-07-01): Option 1 adopted, and sharpened.** A fresh-context
Fable-agent adjudication independently reached Option 1 and added two points that
survive scrutiny (a third claimed point — that the coefficient line in Sec.1 is
wrong — did **not** survive: it was re-verified exact over all 18 `npu1_3col`
tiles, guarded by `test_skew_r3b_identifiability.py`):

1. **Option 2 is not merely expensive, it is self-defeating.** A third *flood*
   does not help — the within-axis anisotropy term enters every interval as a
   per-pair constant and is nulled at *any* source count. Separating within-axis
   directions requires an *absolute* timing observable, which for the horizontal
   axis is exactly the ~30-cycle cross-column jitter R3b's interval design exists
   to avoid. So a third source cannot rescue horizontal anisotropy identifiability.
2. **Vertical within-axis anisotropy IS falsifiable — on R1, not R3b.** R1
   measures *absolute* within-column arrival offsets (jitter-free, range-0 on HW).
   A two-sided mid-column R1 spine (source with cores both North and South) turns
   `d_vN ≠ d_vS` into a slope kink the single-`d_v` fit cannot absorb — residual
   grows (`test_skew_r1_diff_extract.py::test_two_sided_spine_falsifies_vertical_anisotropy`).
   Phoenix's 4 core rows make the two-sided span thin (enough to falsify, not to
   precisely identify), but it is strictly better than R3b's structural zero. So
   the genuinely-*assumed* surface shrinks to **horizontal direction isotropy
   alone** — a second-order asymmetry on a single-digit-cycle per-hop cost.

**Consequence:** R3b fits `{d_h, d_v}` (cross-axis, identifiable). Horizontal
direction isotropy is ASSUMED and must be disclosed as such at the `calibrated`
flip (SP-5c), tied to the fact that the emulator model already assumes it
(`effects.rs`) — same assumption on both sides, so nothing is laundered. Optional
`d_turn` (cross-axis additivity) is deferred until non-corner-source geometry
exists. The code rollback (commit `77180706`) implements this.

## 5. Provenance

The methodology worked: the paper audit flagged Q1 as a risk; implementing the
audit's *fix* exposed that the fix was itself partly unrealizable; and TDD's
rank-check caught it before any hardware. Escalated rather than worked around,
per the plan-flaw escalation rule.
