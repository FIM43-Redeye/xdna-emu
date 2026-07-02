# SP-5c: Comprehensive Skew Characterization + Calibrated Go-Live -- Design

> **Status: DRAFT for review (2026-07-01).** Successor to SP-5b (measurement
> apparatus built + silicon bring-up validated, `57c58ac0`; bring-up finding
> `docs/superpowers/findings/2026-07-01-r3b-pc-silicon-bringup.md`). SP-5c is the
> Phoenix characterization campaign and the one-way `calibrated` flip. This doc
> was decomposed with a Fable analysis agent and its load-bearing claims were
> re-verified against code; it is written to be bounced off Fable again before a
> plan is drawn.

Issue #140, timer-sync faithful-broadcast arc. Sub-project SP-5c of the SP-5
decomposition (5a calibration enablement [done], 5b measurement apparatus +
bring-up [done], 5c comprehensive characterization + go-live [this]).

---

## 0. Mission

**Characterize the entire per-tile clock-skew field of the Phoenix (NPU1/AIE2)
array, honestly and completely, then flip the emulator's `calibrated` model live
on measured constants.** "Completely" is the operative word and a deliberate
scope choice by the architect: the smallness of skew (single-digit cycles per
hop) is *not* a licence to skip characterizing it. This is a cycle-accurate
emulator; the skew field is a real hardware quantity, and our job is to measure
all of it that is measurable and to *disclose*, with physical justification, the
part that provably is not.

**There is no value-of-information kill gate.** An earlier framing asked whether
the correction is worth flipping given that skew (tens of cycles across the
array) is small next to `Delta_wall` (hundreds of cycles of real compute/DMA
duration). The answer is that skew's size relative to `Delta_wall` sets
*tolerances and provenance*, not whether the work happens. Scope expansion, where
it buys more honest coverage, is desirable rather than something to minimize.

## 1. What skew is, and why "skew vs latency" was a red herring

On AIE2 every tile timer is reset synchronously by a channel-15 broadcast, but
the reset signal *propagates* across the array, so a tile farther from the source
resets its timer later in wall-clock time. Tile X's timer therefore reads zero at
wall-clock `T0 + D(source -> X)`, and the per-module skew is *defined* as

```
origin_D[X] = D(source -> X)   (the ch15 reset broadcast's propagation delay)
```

The emulator consumes this in exactly one place -- `tools/inference/grounding.py:65-79`:

```
causal_offset(child, parent, raw) = raw - (origin_D[parent] - origin_D[child])
```

a pure additive per-tile offset, gated behind `calibrated`. Re-deriving what
`raw` contains, with each tile's trace timestamps counted from its own post-reset
origin:

```
raw = soc_child - soc_parent = Delta_wall + (origin_D[parent] - origin_D[child])
```

So the model needs only *the difference in ch15 reset-arrival times between
tiles* -- which is `origin_D` by definition. **The reset broadcast's transport
latency IS the clock skew; they are the same physical quantity.** There is no
"skew fraction" buried inside a larger "latency" to split out. The SP-5b
scope-preview phrase "skew-vs-latency disentangling" is retired and replaced by
the two questions that actually have teeth (Sec. 3).

*(Verified: `grounding.py:65-79` consumes `origin_D` as a pure additive offset;
`effects.rs:446-506` accumulates `d_v`/`d_h` per hop with source `origin_D=0`.)*

## 2. What R3b measures, and the two IFs it rests on

R3b runs two floods (s1 on ch15, s2 on ch14); each measured tile arms
`Performance_Counter0` START on one arrival, STOP on the other, recording the
interval `r_X = D(s2->X) - D(s1->X) + (T0_2 - T0_1)`. A least-squares fit
recovers `{d_h, d_v}`; the `(T0_2 - T0_1)` host-gap constant cancels by
differencing against a reference tile (`r3b_extract.py`). R3b's per-hop constants
equal the ch15 timer-reset per-hop skew **if and only if**:

- **IF-1 (channel fidelity):** the measurement floods traverse the same broadcast
  tree with the same per-hop cost as the real ch15 reset flood.
- **IF-2 (linearity):** transport is hop-count-linear -- no per-tile fixed
  latency that is not proportional to hop count.

SP-5c's scientific content is discharging IF-1 and IF-2 on silicon, plus
extending coverage from the minimal bring-up geometry to the whole array.

## 3. The three seams (how the identity breaks) and the honest ceiling

Verified against code; ranked by impact x likelihood-of-late-discovery.

**Seam 1 -- channel blend (HIGH, the new #1 risk).** R3b's counter is
START=ch15 / STOP=ch14, and `d_h`/`d_v` are fitted as **channel-agnostic
scalars**: `effects.rs:485` uses `channel` only to select `allowed_directions`;
the hop cost at `:491-494` is `d_v`/`d_h` regardless of channel, and
`r3b_extract.py:20` fits a single `d_h` against net displacement. So the
recovered `d_h` is a ch14/ch15 blend, but the model *applies* it to a ch15-only
flood. On a collinear tile set this fits at **zero residual** with a blended,
biased `d_h` -- so the bring-up's ~7e-15 residual does not (and cannot) prove
channel uniformity. **Under the "characterize everywhere" mission, per-channel
hop cost becomes a first-class measured quantity, not a confound to null.**
Falsification / characterization: hold s1=ch15, vary s2's channel across runs;
any shift in `d_h` is a per-channel dependence -- a *finding to record*.

**Seam 2 -- `d_h` has no absolute-frame cross-check (HIGH, structural,
un-retireable).** R1 can cross-check `d_v` (both instruments see the vertical
reset transport), but R1 cannot measure `d_h` at all -- horizontal absolute
offsets ride the ~30-cycle cross-column jitter that R3b's interval design exists
to dodge. So `d_h` is a single-instrument estimate. The right response is
disclosure and the hard b-vector gate (Sec. 6), not more measurement.

**Seam 3 -- non-hop-linear per-tile transport (MEDIUM impact, HIGHEST
late-discovery).** OR-tree re-broadcast may add fixed per-node latency; edge/
corner tiles may differ. A linear model absorbs the hop-linear part and hides the
rest in the (cancelled) const or the residual. A green residual on minimal
geometry cannot distinguish "truly hop-linear" from "linear-enough on 6 tiles" --
this is exactly the inverse-laundering trap the audit warns of. Retired by richer
geometry (longer baselines, off-axis tiles) that gives non-linearity a column to
fail on.

**The honest ceiling (provably unmeasurable from R3b/R1).** Two components are
not identifiable from our two-source interval instruments at any placement, by
the rank-2 theorem (`findings/2026-07-01-r3b-two-source-identifiability-limit.md`):

- **Horizontal within-axis direction anisotropy** (`d_hE` vs `d_hW`):
  `a_hE - a_hW = s1.col - s2.col` is a per-tile constant, so E/W directions
  collapse. (Vertical `d_vN` vs `d_vS` IS falsifiable via the two-sided R1 spine
  -- see Phase 3.)
- **Absolute intra-tile offset** is gauge; only the contrast `core_off - mem_off`
  is observable.

"Characterize everywhere" therefore resolves to **"characterize everything
identifiable from our instruments; find independent toolchain-derived
observability for the rest (Sec. 4); and disclose, with physical justification,
whatever remains assumed."** Disclosure at the point of the flip, not silent
zeros, is how honesty is preserved.

## 4. Toolchain-derived observability for the ceiling components (NEW line of inquiry)

Per the project's prime directive -- *derive from the toolchain* -- a component
unmeasurable from R3b/R1 may still be documented or independently observable
elsewhere. SP-5c opens a dedicated, bounded investigation (pure-desk, no Phoenix
window) before conceding any component to "assumed":

- **aie-rt** (`../aie-rt/driver/src/`): does the HAL encode per-hop or
  per-stream-switch broadcast latency anywhere (event routing, broadcast network
  timing)?
- **AM025 register database / AM020-025 manuals**: is stream-switch per-hop
  latency, broadcast OR-tree depth, or per-module-kind pipeline delay documented
  as a hardware constant? An E/W asymmetry or an absolute intra-offset stated in
  the manual would independently pin a ceiling component.
- **llvm-aie / mlir-aie**: any latency model in the scheduler or device model
  that distinguishes E/W hops or names module-kind fixed offsets.
- **aiesimulator (read-only reference)**: its cycle-accurate model may expose a
  per-tile broadcast-arrival delay we can read as an independent oracle (never
  copied -- read the hardware fact, implement originally).
- **Direct HW register readback**: is there any absolute cross-column timer
  observable (beyond the interval instrument) that sidesteps the ~30-cycle jitter
  -- e.g. a synchronized reference readback -- that could give `d_h` direction a
  second handle?

Outcome per component: either an independent measurement/derivation (promote from
"assumed" to "toolchain-derived"), or a documented dead-end that justifies the
disclosed assumption.

## 5. Decomposition (ordered; front-loads every structural showstopper)

Ordering principle (the rank-2 lesson): prove identifiability on paper and retire
structural risk in pure-software phases *before* spending any Phoenix window.

| Phase | Kind | Purpose / risk retired | Gate |
|-------|------|------------------------|------|
| **0. Theory lock-down + toolchain-observability sweep** | SW | Identifiability proof for *every* solver column before any kernel goes to silicon; magnitude estimate for tolerances/provenance (NOT a kill gate); the Sec. 4 toolchain sweep for ceiling components. | Rank proof per column; written skew-vs-latency + assumption decision; toolchain sweep concluded per component. |
| **1. Rust ingestion wire + inert §9a plumbing** | SW | Build the missing `skew_constants.json -> model` path (Sec. 7); §9a fixpoint-ch15 multi-source test; wire the sweep sidecar consumption. All while `calibrated=false`. | Round-trip write->read->model populated, still inert; §9a tests green. |
| **2. Channel-fidelity + linearity + full-array characterization** | HW | Discharge Seams 1 & 3; per-channel hop cost as first-class; whole-array skew map (all cols x rows); hard b-vector gate; empirical jitter-immunity. | Enriched residuals green OR provenance downgraded; per-channel result recorded; b-vector range strictly `<` measured `d_h` over N>=20 (RED blocks). |
| **3. Vertical anisotropy + joint sign anchors + d_v cross-check** | HW | Falsify `d_vN != d_vS` via two-sided mid-column R1 spine; place both `d_v` sign conventions in one silicon frame; run the (now-clean) R1/R3b `d_v` cross-check. | Two-sided spine residual green OR model gains `d_vN/d_vS` split; signs agree; R1 `d_v` ~= R3b `d_v`. |
| **4. `dn_v`-correlated `Delta_wall` component** | HW | Close/bound the lethal mode a green `fit_residual` does NOT clear (the row-51 residual): a `Delta_wall` term linear in `dn_v` absorbed into `d_v` at zero residual. May fold into Phase 3. | Component measured `<` skew resolution, or bounded by the `Delta_wall`-free R3b cross-check. |
| **5. Held-out validation kernel** | SW build -> HW run | Break the causal-vs-HW tautology (`skew := HW_offset - Delta_wall_emu` is circular on the fit geometry). Validate on a geometrically-distinct kernel whose `Delta_wall` varies non-collinearly with skew. | `causal_offset` predictions match HW trace deltas on the held-out kernel within tolerance. |
| **6. Measure, flip `calibrated`, regression guards** | ONE-WAY, HW-confirmed | The door. Set constants via the Phase-1 wire; flip; 3 regression guards; honest provenance (measured vs toolchain-derived vs assumed). | Full regression + held-out causal-vs-HW + all prior gates green. |

**Dependencies:** `0 -> {1, 2}` (1 pure-SW, 2 HW, parallel); `2 -> 3 -> 4`;
`{3,4} + 5 -> 6`. Phase 0 gates everything (no kernel to silicon without its
proofs). Phase 5's kernel *build* can run parallel to 2-4; its *validation run*
co-locates with Phase 6.

## 6. Pre-flip gate mapping (all 7 must hold before the flip)

| # | Gate | Retired in |
|---|------|-----------|
| 1 | Held-out validation kernel (break the tautology) | Phase 5 |
| 2 | Enriched-geometry residuals green (else provenance = "assumed") | Phase 2 (structure/channel) + Phase 3 (vertical) |
| 3 | Joint sign anchors in one silicon frame | Phase 3 |
| 4 | `dn_v`-correlated `Delta_wall` quantified or dismissed | Phase 4 |
| 5 | Hard cross-column b-vector gate (RED blocks, not informs) | Phase 2 |
| 6 | §9a(a) fixpoint channel-15 multi-source test | Phase 1 |
| 7 | §9a(b) wire sweep sidecar consumption | Phase 1 |

## 7. The missing Rust ingestion wire

**Confirmed gap.** `tools/calibration/skew/schema.py` is a Python-only read/write
for `skew_constants.json`. The only path constants take into the Rust model today
is the test-only runtime seam `set_broadcast_timing_override` (`effects.rs:563`);
compile-time constants are all zero with `calibrated:false`
(`model_builder.rs:270-279`). There is no production reader.

**Recommended shape (Phase 1):** a build-time codegen step that reads the
committed `skew_constants.json` and emits the archspec `BROADCAST_*` consts +
`calibrated` flag -- so the flip is a **reviewable git diff** and the provenance
string travels with it (matches the "derive, schema-first" ethos). Retain the
runtime override for tests. **Extend the schema first** to carry what the honest
gates require: per-channel result, b-vector range, jitter range, and an explicit
`assumptions` field (e.g. `horizontal_direction_isotropy: assumed`,
`absolute_intra_offset: gauge`). Schema-first: define these before the flip.

## 8. Ranked risk register

1. **Channel blend `d_h != d_h_ch15` (Seam 1)** -- HIGH impact, HIGH
   late-discovery. Retire Phase 2.
2. **Non-hop-linear transport masked by minimal geometry (Seam 3)** -- MED-HIGH
   impact, HIGHEST late-discovery (zero residual reads as total success). Phase 2.
3. **`d_h` single-instrument, no absolute cross-check (Seam 2)** -- HIGH impact,
   structural/un-fixable; disclose (Phase 0/6) + b-vector gate (Phase 2).
4. **Tautology in causal-vs-HW gate** -- HIGH impact (sole backstop is fake on the
   fit kernel). Phase 5.
5. **`Delta_wall`-linear-in-`dn_v` absorbed into `d_v` (row 51)** -- MED impact,
   HIGH late-discovery (silent, unbounded). Phase 3/4.
6. **Vertical anisotropy corrupting the R1/R3b cross-check** -- MED/MED. Phase 3.
7. **Rank-deficiency recurrence in a new enriched column** -- HIGH impact if it
   recurs; now LOW late-discovery because Phase 0 proves rank first.
8. **Per-tile frequency (not phase) skew, unmodeled** -- LOW impact on
   microsecond windows, LOW likelihood. Document as a limit; do not chase.

## 9. What NOT to do

- **Do not resurrect the 5-param signed E/W + N/S solver.** The rank-2 theorem
  proved those columns are unidentifiable from any two-source interval; master's
  `r3b_extract` is correctly rolled back to `{d_h, d_v}`. Honor the errata
  pointers in the rev3 docs.
- **Do not run the terminal causal-vs-HW gate on the calibration kernel** -- it is
  a tautology there. Held-out kernel only (Phase 5).
- **Do not scope-creep the `d_turn`/interaction column back in** unless a
  deliberate non-corner source placement is chosen and its rank is proven in
  Phase 0. With corner sources it adds a third null direction (rank collapses).
- **Do not read a green residual on minimal geometry as shape validation.** That
  is the inverse-laundering trap; enriched geometry is what earns the claim.

## 10. Open questions to bounce off Fable (review targets)

1. Is the Phase ordering right, or should the toolchain-observability sweep
   (Sec. 4) be its own phase gating Phase 2's geometry design (if the manual
   documents E/W asymmetry, the geometry should be built to confirm it)?
2. Is per-channel characterization (Seam 1 first-class) best done as a dedicated
   sweep, or folded into the Phase-2 enriched geometry?
3. Full-array coverage: is a whole-array skew map the right ambition, or does the
   rank/identifiability structure mean a well-chosen sparse set carries the same
   information at lower Phoenix cost? (Characterize-everywhere vs
   characterize-sufficient.)
4. Are there ceiling components we have NOT listed (Sec. 3) that the
   "characterize everywhere" mission should chase -- e.g. per-column clock-tree
   skew independent of broadcast transport?
5. Does build-time codegen vs runtime JSON load for the ingestion wire (Sec. 7)
   have a failure mode we're missing?
