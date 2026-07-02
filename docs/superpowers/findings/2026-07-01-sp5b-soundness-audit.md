# SP-5b Soundness Audit — Skew-Inference Apparatus (#140 timer-sync arc)

**Date:** 2026-07-01
**Method:** Adversarial multi-Opus panel (15 agents: 5 Sonnet grounding readers →
4 Opus adversaries, one per identifiability question → 5 Opus cross-examiners
[4 rebutting each other + 1 red-team completeness critic] → 1 Opus synthesizer).
Run against the merged design docs + code; no HW. ~1.45M subagent tokens.
Panel run ID `wf_df21e1ae-f78`.

**Panel-integrity note.** Two round-1 reviewers (Q2-joint, Q3-deltawall) hit the
StructuredOutput retry cap and produced no round-1 verdict. Coverage held: the
cross-examiners who own Q2 and Q3 did their own independent code verification
(`cross:Q2-joint` re-derived from `effects.rs:657-658, r1_observe.py:52,
r1_diff_extract.py:35-38, r3b_extract.py:20`; `cross:Q3-deltawall` worked the
contamination through the least-squares projection), and the synthesizer
re-verified every load-bearing anchor. **Independently spot-verified by Opus
(this session) after the run:** `effects.rs:491-494` (isotropy — one `d_v` for
N/S, one `d_h` for E/W), `:497` (`ncost = cost + step`, additive), `:485`
(channel selects direction only, not hop cost), and `r1_diff_extract.py:35-36`
(design matrix `[dn_v_diff, core_ind_diff]`, **no intercept column**). All
confirmed exactly as reported.

---

> **ERRATA (2026-07-01, added post-implementation).** Q1's remediation below —
> "enriched geometry (two-sided + diagonal) makes direction isotropy falsifiable"
> — is **partly unrealizable**. Implementing it exposed that the R3b two-flood
> interval method cannot identify within-axis direction anisotropy (`d_hE` vs
> `d_hW`, `d_vN` vs `d_vS`) from two point sources at *any* placement (a rank
> theorem: `a_hE - a_hW = s1.col - s2.col`, constant for every tile). R3b *can*
> separate cross-axis `d_h` vs `d_v` (the physically plausible asymmetry) but not
> within-axis direction. See
> `docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md`.
> **RESOLVED (2026-07-01): Option 1** — assume horizontal within-axis isotropy;
> R3b fits `{d_h, d_v}`. Vertical within-axis anisotropy is reallocated to a
> two-sided R1 spine (falsifiable there, not on R3b). So Q1's isotropy-via-R3b
> remediation is conceded for horizontal, realized for vertical via R1. Code
> rollback: commit `77180706`. The Q1 text below stands as the audit *found* it,
> corrected by that finding + resolution.

## 1. Bottom line

**Build R3b — do not flip `calibrated`.** SP-5b is honest, well-gauged
apparatus: the differencing solvers correctly quotient out the one true gauge
freedom (`r1_diff_extract.py` recovers `{d_v, intra_contrast}`, not the
unobservable absolute intras), the `>=3-collinear-tiles` hook genuinely
falsifies *one* structural assumption (per-hop uniformity along a monotonic
axis), and the whole thing produces no number and flips no flag. Proceeding to
build the R3b-PC kernel is the right next step and costs no irreversibility. But
the inference **as designed cannot support the SP-5c calibrated flip**, and this
is not a hedge: of the four continuous structural assumptions the 4-knob linear
model bakes in — per-hop uniformity, direction isotropy, cross-axis additivity,
and per-channel uniformity of hop cost — the shipped geometry and solvers can
falsify only the first; the other three are **unfalsifiable by construction**
and the design docs are silent on two of them. On top of that, R3b's `d_h` (the
money quantity it was promoted to own) is a single-instrument estimate riding on
unfalsifiable isotropy, an *unmeasured* two-channel blend, and an unanchored
silicon sign; R1's `intra_contrast` is Delta_wall-contaminated with no
cross-check; and the terminal "causal-vs-HW" gate is definitionally circular if
run on the calibration geometry, which no document forbids. None of these blocks
building. All of them block the flip until the geometry is enriched, the sign
anchors are jointly resolved, and validation moves to held-out kernels.

## 2. Per-question verdict

### Q1 — Model-structure identifiability: UNDER-IDENTIFIED. HARD BLOCKER.

The linear model `origin_X = dn_v*d_v + dn_h*d_h + intra(kind)` bakes in four
structural assumptions; the apparatus falsifies one.

- **Per-hop uniformity (falsifiable — NOT circular).** `>=3` collinear same-kind
  tiles over-determine the axis line; `fit_residual` grows on an injected
  non-uniform hop. Confirmed by
  `test_skew_r1_diff_extract.py::test_fit_residual_grows_on_nonuniform_hops` and
  the negative control `test_two_points_per_axis_cannot_falsify`
  (`assert r['fit_residual'] < 1e-9`). This part of the design is sound.
- **Direction isotropy (unfalsifiable, unremarked).** `effects.rs:491-494` uses
  one `d_v` for both North and South, one `d_h` for both East/West. R1's spine is
  one-sided (cores at rows 2/3/4, all North of the source), so `d_v_south` is
  never exercised; a one-sided spine fits a zero-residual line under any
  anisotropy. R3b's opposite-corner floods do **not** rescue it: on a collinear
  row the anisotropic truth is still exactly linear in column index, so it
  absorbs into an averaged scalar with zero residual (`r3b_extract.py:20`, single
  unsigned column). Neither design doc addresses direction anisotropy.
- **Cross-axis additivity (unfalsifiable as specified).** `effects.rs:497` is
  pure additive; the geometry requirement is "collinear *per axis*", and every
  synthetic test uses only pure-axis points `(1,0),(2,0),(0,1),(0,2)`. An
  interaction/turn term contributes identically zero to every axis-collinear
  observation, so no residual can ever detect it.
- **2-category kind grouping (honestly punted).** `r1_extract.py` docstring:
  "Provisional emulator model; SP-5c may revise the grouping." Correctly
  deferred, not claimed validated — not a defect.

**Decisive argument:** `measurement-apparatus-design.md:214-216` reads a large
residual as "the shape is wrong," which invites the inverse laundering — a green
residual read as "4-knob shape confirmed" — flipping `calibrated` on a shape
that was 1/4 tested.

### Q2 — Joint identifiability of the two instruments: UNDER-IDENTIFIED / UNCERTAIN. HARD BLOCKER.

The gauge accounting is correct: only three combinations — `d_h`, `d_v`,
`contrast` — are ever observable; `core_off`/`mem_off` individually are not; the
differencing solvers implement exactly this. The *joint* story is thin:

- **The redundancy is 1/3, not 2×.** `d_v` is the only quantity both instruments
  see. `d_h` is R3b-only (`r3b_extract.py` has no kind column). `contrast` is
  R1-only. "The two instruments validate each other" is true only for `d_v`.
- **The one overlap is confounded.** R1's `d_v` is `d_v_north` (one-sided spine);
  R3b's `d_v` is structurally the `(north+south)/2` average. Under vertical
  anisotropy these are different physical quantities. The design defines **no
  adjudication rule or threshold** for disagreement — so agreement is misread
  confirmation and disagreement would be misattributed to row-51/noise.
- **Sign reconciliation is two anchors, treated as one.** R1's `d_v` sign is
  fixed by the reflected-trace convention (`r1_extract.py` negates
  `-o['origin']`); R3b's is fixed by perf-counter start/stop ordering and the
  geometric definition of `dn_v` — a *different* silicon fact. A single silicon
  anchor fixes one, not both.
- **Reference-tile fragility.** `r3b_extract.py:11,15` differences against a
  single hardcoded `reference=0`; a deterministic bias on that one tile shifts
  every b-row and silently tilts `d_h`/`d_v` while passing the range-0 gate.

### Q3 — Delta_wall contamination of R1: CONTAMINATES (under-identified). HARD BLOCKER for R1 `d_v`/`contrast`.

`observe_r1` (`r1_observe.py:52`) forms `skew = (sa_m - sb_m) - (sa_d - sb_d)`,
so the error `err_p = Delta_wall_true - Delta_wall_emu` (the OPEN row-51
residual, EMU −52 vs HW +2) adds directly into the b-vector `r1_diff_extract`
fits. The design docs get the structure *backwards*:

- **Constant Delta_wall bias is largely self-detecting, not "indistinguishable
  from skew."** `r1_diff_extract.py:35-36` has no intercept/all-ones column
  (observe_r1 already differenced within each pair). A constant bias across pairs
  is **not** in `span(A)`, so it projects into `fit_residual`.
  `measurement-apparatus-design.md Sec.2` *overstates* the constant-offset risk.
- **The lethal mode is a Delta_wall error linear in `dn_v`.**
  `err_p = gamma*(dn_v(b)-dn_v(a))` lies exactly in `span(A)` column 0 → absorbed
  fully into `d_v` (`d_v_recovered = d_v_true + gamma`), zero residual, silent,
  unbounded. Row 51 is a pipeline-fill-cadence effect whose per-module error
  plausibly scales with pipeline depth ~ `dn_v` — exactly this shape. Its
  `dn_v`-correlated magnitude is **unmeasured** (only a single lean-kernel pair).
- **`intra_contrast` is worse.** Same-tile pairs (`dn_v`-diff = 0) put the error
  wholly on the kind column, and there is **no R3b cross-check** for contrast.
  Consistent with the PROVISIONAL outcome the R1 HW gate already produced.

R3b *is* genuinely Delta_wall-free, so it can cross-check `d_v` (confounded per
Q2) but never `contrast`.

### Q4 — Instrument-role flip / R3b immunity: UNCERTAIN → UNDER-IDENTIFIED. HARD BLOCKER for `d_h`.

- **Jitter sub-question: sound-by-gate.** R3b's counter is started by s1 arrival
  and stopped by s2 arrival; if cross-column *arrival* jitter were run-to-run
  random, `r_X` varies and the Phase-2 range-0 b-vector gate goes RED before the
  flip. Caught — but on the *gate*, not the rationale. The stated rationale
  ("single-tile local clock reading, immune to cross-tile phase") is a **category
  error**: R3b never compares two tiles' clock origins; its counter is triggered
  by events that traverse the very crossing fabric where the ~30cy lives.
- **Deterministic channel-dependence: un-gated blocker (confirmed in code).** s1
  and s2 are on *distinct* broadcast channels. But `broadcast_origin_d` takes
  `d_h`/`d_v` as **channel-agnostic scalars** — `channel` only selects
  `allowed_directions` (`effects.rs:485,491-494`), never hop cost — and
  `r3b_extract.py:20` fits a single `d_h` against net displacement. On the
  straight opposite-corner row the two-channel blend is exactly linear in column,
  so the fit yields `d_h = (d_h^{s1}+d_h^{s2})/2` with **zero residual and
  range-0** — passing both gates. But the model *applies* that `d_h` to the
  channel-15 timer flood, which R3b can never measure in isolation. Horizontal
  channel-uniformity is measured and cross-checked **nowhere**; R1 cannot measure
  `d_h` at all.
- **The ~30cy was never measured on a broadcast.** It was measured between two
  cores' first LOCK_STALL events — a lock/DMA compute signal, not an
  EventBroadcast arrival. Whether cross-column broadcast-arrival jitter is even
  the same size is unmeasured anywhere in the repo.

## 3. Required changes before SP-5c

### Must fix before *building* R3b (cheap now, expensive as a retrofit)

1. **Bake two-sided + off-axis geometry into the R3b kernel and split the solver
   columns.** Place `>=1` measured same-kind tile on the *opposite* side of a
   source along each axis, and `>=1` off-axis diagonal tile
   (`dn_h != 0` AND `dn_v != 0`). Change `r3b_extract.py` (and
   `r1_diff_extract.py`) to signed North/South and East/West indicator columns
   plus a corner/interaction column, so anisotropy and additivity become
   over-determined, testable columns. Add synthetic tests mirroring
   `test_fit_residual_grows_*` where an injected anisotropic or interaction term
   must drive `fit_residual > threshold`.
2. **Design the channel-uniformity probe into the R3b kernel.** Either isolate a
   single-channel measurement of `d_h` (so the ch15 hop cost is measured on ch15,
   not a blend), or add an explicit same-corner/same-channel control pair whose
   residual would expose `d_h^{s1} != d_h^{s2}`. A geometry/config change, not a
   solver change.
3. **Add the cross-column broadcast-arrival-jitter pre-check to the Phase-2 HW
   gate.** Arm START=s1/STOP=s1 (or two co-observed broadcast events) at a
   cross-column tile and measure run-to-run range of a pure broadcast-arrival
   interval — directly measuring the quantity the flip assumes small.
4. **Rewrite the immunity text** (`apparatus Sec.5.1`, `kernel Sec.5.1`): strike
   "immune to cross-column jitter" as an architectural claim; state that immunity
   is **empirically gated by the range-0 b-vector check, not guaranteed by the
   single-clock property**; remove "trace" from "cross-column trace jitter."

### Must fix before flipping `calibrated`

5. **Realize the enriched geometry and require its residuals green.** The flip
   may cite a green residual as shape-validation *only* if the two-sided +
   diagonal + channel-control geometry of items 1–2 is built and its residuals
   are also green. Absent that, recorded provenance must read
   "uniformity-validated; isotropy, additivity, and channel-uniformity ASSUMED,
   not measured."
6. **Resolve both sign anchors jointly on one silicon frame.** A shared silicon
   anchor must place both `d_v` conventions in one frame before their agreement
   is trusted.
7. **Specify and run a held-out, geometrically-distinct validation kernel.**
   Because `skew := HW_offset - Delta_wall_emu` (`r1_observe.py:52`), a calibrated
   emulator reconstructs the exact HW offset on the *same* geometry it was fit to
   — the terminal "causal-vs-HW" gate is a **tautology** on the calibration
   kernel, and no doc forbids running it there. SP-5c must validate on a kernel
   whose Delta_wall varies non-collinearly with per-module skew. (Contingent on
   the currently-unspecified SP-5c procedure, not a confirmed code defect — but
   must be written down and enforced before the flip.)
8. **Quantify or dismiss the `dn_v`-correlated Delta_wall component.** Close row
   51's `dn_v`-correlated residual, or measure it on R1's actual kernel/window
   and show it empirically vanishes. A green R1 `fit_residual` does **not** clear
   this — the lethal mode is absorbed with zero residual.
9. **Hard-gate the flip on the cross-column b-vector.** The `d_h` flip must block
   on b-vector range over `N>=20` runs being strictly `< the single-digit d_h`
   being measured — a red result must block, not merely inform a go/no-go.

## 4. Resolved on paper vs needs HW

**Resolved (confirmed, code-anchored):**
- Solvers' gauge handling is correct; `core_off`/`mem_off` legitimately
  unobservable, correctly reduced to `contrast` (not a defect).
- Per-hop uniformity *is* genuinely falsifiable with `>=3` collinear same-kind
  tiles (not circular).
- Isotropy (`effects.rs:491-494`), additivity (`:497`), and channel-uniformity
  (`:485`) are baked in and **unfalsifiable by the shipped geometry** —
  structural facts of the current code, not speculation.
- The Delta_wall linear-in-`dn_v` absorption into `d_v` (zero residual) is a
  structural fact of `span(A)`; the constant-offset risk is *over*stated by docs.
- `intra_contrast` has no cross-check instrument.
- The run-to-run *jitter* failure mode is caught by the range-0 gate before the
  flip.

**Needs HW / experiment (speculative until measured):**
- Whether real Phoenix silicon *is* anisotropic (N≠S, E≠W) or channel-dependent.
  If silicon happens to be isotropic and channel-uniform, the unfalsifiable
  assumptions are harmlessly true — but that is an unmeasured bet, and the point
  is the apparatus cannot tell.
- The magnitude of the `dn_v`-correlated Delta_wall component (row 51) — only a
  single lean-kernel pair number exists.
- Whether cross-column broadcast-arrival jitter is small (the ~30cy was
  LOCK_STALL-based; item 3 settles this cheaply on HW).
- Whether `d_h` and `d_v` actually collapse — deferred to SP-5c, meaningful only
  once items 1–2 make the estimates un-blended.

## 5. Orientation

The #140 timer-sync arc exists to let the trace inference engine ground
cross-column causal edges: `raw_offset = Delta_wall + skew`, one equation, two
unknowns, resolved by using a within-domain-byte-exact emulator to supply
`Delta_wall` so the residual isolates `skew`. SP-1..SP-4 built and merged the
mechanism, all behavior-neutral behind zero constants and `calibrated=false`.
SP-5a/SP-5b-software-core (merged) added the runtime-override seam and the
extractor/solver stack. R1 (within-column) is merged and gated; **R3b-PC (the
primary `d_h`/`d_v` instrument) is not yet built — it is the next thing to
build.** SP-5c is the one-way, Phoenix-gated door that flips `calibrated`;
downstream sit framework tenants 4/5 and `clean_release(Aie2)`.

Verdict: **the door is not ready, but the next build is.** The apparatus is
unusually honest — it ships no numbers and self-flags most of these risks (the
NEXT-STEPS Sec.D list named all four questions). What the panel adds is that
three of the four are **confirmed** structural under-identifications in the
current code, not open worries. The correct move is to build R3b **with the
enriched geometry (items 1–3) designed in from the start** — retrofitting a
one-sided single-channel kernel later wastes the scarcer resource (a Phoenix
window). The `calibrated` flip stays red until items 5–9 are satisfied.
