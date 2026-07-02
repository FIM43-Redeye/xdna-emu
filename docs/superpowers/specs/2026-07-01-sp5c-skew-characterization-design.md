# SP-5c: Comprehensive Skew Characterization + Calibrated Go-Live -- Design

> **Status: DRAFT rev2 (2026-07-02) for review.** Successor to SP-5b (measurement
> apparatus built + silicon bring-up validated, `57c58ac0`; bring-up finding
> `docs/superpowers/findings/2026-07-01-r3b-pc-silicon-bringup.md`). SP-5c is the
> Phoenix characterization campaign and the one-way `calibrated` flip.
>
> **rev2 folds in an independent adversarial Fable review** (2026-07-02). Two
> headline additions, both verified against source: (1) the mlir-aie
> `test/benchmarks/` broadcast-delay numbers independently explain and corroborate
> the bring-up `d_h≈4/d_v≈2` (Sec. 4); (2) the AIE2 timer-reset broadcast is a
> **shaped, E/W-blocked tree** (`XAie_SyncTimer`), not a free flood, so both R3b's
> kernel and the emulator's `origin_D` table must be computed on that tree
> (Sec. 1, 7). rev2 also corrects two over-claimed mitigations (Seams 1 & 2), adds
> a Phase 3/4 attribution rule, defines the Phase 5 tolerance, and hardens the
> ingestion wire against stale/silent-zero modes. rev1 -> rev2 changes are tagged
> **[rev2]** inline.

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
all of it that is measurable, to find independent toolchain-derived handles for
the components our instruments cannot reach, and to *disclose*, with physical
justification, whatever provably remains.

**There is no value-of-information kill gate.** An earlier framing asked whether
the correction is worth flipping given that skew (tens of cycles across the
array) is small next to `Delta_wall` (hundreds of cycles of real compute/DMA
duration). The answer is that skew's size relative to `Delta_wall` sets
*tolerances and provenance*, not whether the work happens. Scope expansion, where
it buys more honest coverage, is desirable rather than something to minimize.

## 1. What skew is, why "skew vs latency" was a red herring, and the reset-path IFs

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
scope-preview phrase "skew-vs-latency disentangling" is retired.

**[rev2] The identity's real content, from the toolchain.** The reason the
identity holds -- and it is a derive-from-toolchain *fact*, not an assumption --
is the actual reset mechanism: `XAie_SyncTimer` /
`XAie_SyncTimerWithTwoBcstChannel` (`aie-rt/driver/src/timer/xaie_timer.c:823,727`)
points each tile's `Timer_Control.Reset_Event` at a broadcast-channel event
(`_XAie_SetupTimerConfig`, `:559`) and fires one `XAie_EventGenerate` at shim
(0,0) (`:789`). R3b's perf counter is armed START by the *same class* of
broadcast-event arrival. Both "timer reaches zero" and "counter starts" are
driven by the identical event-network arrival -- which is precisely why the
arrival interval R3b measures equals the timer-origin difference. (R3b's
two-channel design ch15+ch14 even mirrors `XAie_SyncTimerWithTwoBcstChannel`,
which uses two channels because the shaped single-channel tree cannot reach every
tile -- see below.)

The identity rests on three IFs, not two:

- **IF-1 (channel + tree fidelity):** the measurement floods traverse the same
  broadcast tree, with the same per-hop cost, as the real ch15 reset flood. This
  is now sharper than rev1 stated -- see the tree-topology point below.
- **IF-2 (linearity):** transport is hop-count-linear -- no per-tile fixed latency
  that is not proportional to hop count.
- **[rev2] IF-3 (arrival->latch uniformity):** between broadcast-event *arrival*
  and the timer *latching to zero* there is a `Reset_Event`->latch step. If that
  delay is uniform across tiles it cancels in the `origin_D` difference; if it is
  per-module-kind it folds into the (already gauge/contrast) intra-tile offset --
  harmless, but by argument, not by luck. Named explicitly so the cancellation is
  reasoned.

**[rev2] The reset tree is shaped, not free -- a correctness item.**
`_XAie_SetupBroadcastConfig` (`xaie_timer.c:472-542`) for AIE2 blocks
**MEM-module EAST and CORE-module WEST on every AIE tile** (`:512-525`) and blocks
both E/W on memtile mem modules (`:526-534`). The real timer-reset broadcast is a
deliberately shaped directed tree, so it does **not** propagate horizontally the
way a free flood does. Two consequences:

1. **R3b fidelity (IF-1):** R3b's kernel must replicate `XAie_SyncTimer`'s block
   config, or it measures a *different topology* than the real reset. (Per-hop
   *cost* is plausibly blocking-independent -- blocking prunes edges, shortest-path
   per-hop cost is unchanged -- but that argument must be made, not skipped; it is
   a Phase-0 item, not an assumption.)
2. **Emulator `origin_D` correctness:** `effects.rs:578-580` confirms trace-prepare
   emits no block-mask writes, so the emulator floods **freely** (masks at reset =
   0). The real AIE2 reset uses the E/W-blocked tree, so even with perfect
   `d_h/d_v` the per-tile `origin_D` is computed on the *wrong* shortest paths for
   some tiles. The fix is tractable: `broadcast_origin_d` *already consults block
   masks per hop* (`effects.rs:573-576`) -- the machinery exists; the calibrated
   `origin_D` computation must *program* the `XAie_SyncTimer` block config rather
   than leave masks at 0. This belongs in the Sec. 7 wire/table generation, not
   just the constants.

*(Verified: `grounding.py:65-79` pure additive offset; `effects.rs:446-506`
per-hop accumulation, source `origin_D=0`; `effects.rs:573-580` mask-consulting
flood that today floods freely; `xaie_timer.c:472-542` AIE2 E/W block config.)*

## 2. What R3b measures

R3b runs two floods (s1 on ch15, s2 on ch14); each measured tile arms
`Performance_Counter0` START on one arrival, STOP on the other, recording the
interval `r_X = D(s2->X) - D(s1->X) + (T0_2 - T0_1)`. A least-squares fit
recovers `{d_h, d_v}`; the `(T0_2 - T0_1)` host-gap constant cancels by
differencing against a reference tile (`r3b_extract.py`). R3b's per-hop constants
equal the ch15 timer-reset per-hop skew iff IF-1, IF-2, IF-3 hold. SP-5c's
scientific content is discharging those IFs on silicon and extending coverage
from the minimal bring-up geometry across the (accessible) array.

## 3. The seams (how the identity breaks) and the honest ceiling

Verified against code; ranked by impact x likelihood-of-late-discovery.

**Seam 1 -- channel blend (HIGH, the #1 risk).** R3b's counter is
START=ch15 / STOP=ch14, and `d_h`/`d_v` are fitted as **channel-agnostic
scalars**: `effects.rs:485` uses `channel` only to select `allowed_directions`;
the hop cost at `:491-494` is `d_v`/`d_h` regardless of channel, and
`r3b_extract.py:20` fits a single `d_h` against net displacement. So the
recovered `d_h` is a ch14/ch15 blend, applied by the model to a ch15-only flood.
On a collinear (and even the corner-source bring-up) geometry this fits at
**zero residual** with a biased average -- so the bring-up's ~7e-15 residual does
not, and cannot, prove channel uniformity.

**[rev2] Honest bound on what a channel sweep retires.** ch15 appears *only* as
one leg of the blend, so `d_h^{ch15}`-in-isolation is **structurally unmeasurable
by R3b** -- exactly like horizontal E/W anisotropy. The Phase-2 "vary s2 across
ch14/13/12, hold s1=ch15" sweep detects differences *among measurement channels*
(`d_h^{ch14} != d_h^{ch13}`) but is blind to `d_h^{ch15}` differing from the
rest. So the sweep is worth running (a per-channel difference is a finding), but
**ch15 hop-cost uniformity stays a disclosed assumption after it.** rev1's
"channel becomes first-class and findable" over-claimed; corrected here.

**Seam 2 -- `d_h` has no absolute-frame cross-check (HIGH, structural,
un-retireable).** R1 can cross-check `d_v` (both instruments see the vertical
reset transport), but R1 cannot measure `d_h` at all -- horizontal absolute
offsets ride the ~30-cycle cross-column jitter that R3b's interval design exists
to dodge. **[rev2] Disclosure is the *only* mitigation for `d_h`'s value.** rev1
wrongly listed the b-vector gate here: that gate checks run-to-run *range*
(jitter/determinism), and a channel-blended or reference-biased `d_h` is perfectly
*deterministic* -- it passes range-0 untouched. The b-vector gate belongs to the
jitter risk (Sec. 8 #1), not to Seam 2's systematic-bias concern. `d_h`'s *value*
is cross-checked nowhere; we disclose that plainly.

**Seam 3 -- non-hop-linear / non-per-column transport (MED impact, HIGHEST
late-discovery).** OR-tree re-broadcast may add fixed per-node latency; edge/
corner tiles may differ; a linear model absorbs the hop-linear part and hides the
rest in the (cancelled) const or the residual. A green residual on minimal
geometry cannot distinguish "truly hop-linear" from "linear-enough on a few
tiles" -- the inverse-laundering trap. **[rev2] Two refinements:** (a) the "hop" is
mis-modeled as per-column -- the toolchain benchmark (Sec. 4) shows a horizontal
hop is physically *two module traversals* (core + mem, ~2 cy each); the
emulator's single per-column `d_h` (`effects.rs:493`) coarsens a per-module
structure, and if core-vs-mem traversal costs differ that lives inside `d_h`
unmodeled. (b) Phoenix's *usable* array is small -- physical col 0 is inaccessible
(rewritten-inaccessible), leaving ~4 columns and ~4 core rows, so "longer
baselines / off-axis tiles" have <=4 hops of leverage: enough to catch gross
non-linearity, not subtle curvature or an axis-aligned OR-tree offset. Phase 2's
"whole-array map" means **all accessible cols**, and its power ceiling is
acknowledged, not assumed away.

**The honest ceiling (provably unmeasurable from R3b/R1).** Not identifiable from
two-source interval instruments at any placement:

- **Horizontal within-axis direction anisotropy** (`d_hE` vs `d_hW`):
  `a_hE - a_hW = s1.col - s2.col` is a per-tile constant, so E/W directions
  collapse. (Vertical `d_vN` vs `d_vS` IS falsifiable via the two-sided R1 spine
  -- Phase 3.)
- **Absolute intra-tile offset** is gauge; only the contrast `core_off - mem_off`
  is observable.
- **[rev2] `d_h^{ch15}` in isolation** (Seam 1): the reset channel's own hop cost,
  measurable only blended.
- **[rev2] Per-module horizontal cost** (core-module vs mem-module traversal): the
  single-`d_h` model cannot split the two series traversals a horizontal hop
  physically comprises.
- **[rev2] Clock-tree / PLL *phase* skew independent of the event network:** a
  fixed per-column clock-tree offset makes "timer latches zero" lag "event
  arrives" by an amount R3b (event-network transport) does not see. *Partially*
  mooted because reset and counter are driven by the same event -- but only under
  IF-3. One disclosed sentence.
- **[rev2] Structured OR-tree asymmetry:** a re-broadcast offset that is a global
  pattern relative to the tree root (e.g. +k for every tile past the root column)
  can align with an axis and evade the longer-baseline test, absorbing into
  `const` or `d_h`. Distinct from per-tile jitter.

"Characterize everywhere" therefore resolves to **"characterize everything
identifiable from our instruments; find independent toolchain-derived
observability for the rest (Sec. 4); and disclose, with physical justification,
whatever remains assumed."** Disclosure at the flip, not silent zeros, preserves
honesty.

**[rev2] The skew / `Delta_wall` boundary.** `grounding.py:78` applies `origin_D`
to *every* cross-domain edge, but for edges mediated by a stream/DMA hop the true
offset includes stream-switch pipeline latency (the `inter_tile_hop_latency` /
`packet_arbitration` terms, `model_builder.rs:260-268`) that is `Delta_wall`'s job
(emu-supplied), not `origin_D`'s. `is_async_cdc` (`grounding.py:82`) carves out
the NoC-egress worst case; the on-chip stream-latency boundary between "skew" and
"`Delta_wall`" needs one explicit sentence in the flip provenance so the two
corrections neither double-count nor gap.

## 4. Toolchain-derived observability for the ceiling components

Per the prime directive -- *derive from the toolchain* -- SP-5c ran a desk sweep
(no Phoenix window) across the four sources before conceding any component to
"assumed." Results:

**[rev2] mlir-aie `test/benchmarks/` -- the corroboration rev1 missed, and the
single highest-value find.** `test/benchmarks/README.md:61-74` carries *empirical*
perf-counter-measured broadcast delays (VCK190/AIE1 @1 GHz), explicitly labeled
calibration measurements: **horizontal 2 cy per core/memory module -- and
horizontal signals must pass through *both* the core and mem module to reach the
next tile, so ~4 cy per column-hop** (benchmark 10, with the physical reason: 16
horizontal broadcast wires vs 32 vertical); **vertical 2 cy per tile** (11);
**shim->shim 4 cy** (9); **stream 2 cy/node** (12). Restated in
`mlir_exercises/tutorial-4/flow/README.md:147`; the reset skew itself is
acknowledged in `python/utils/trace/__init__.py:165` ("a slight delay ... a few
clock cycles between tiles"). This **independently explains the bring-up**
`d_h≈4` (2 modules x 2 cy) and `d_v≈2` (1 tile x 2 cy) -- a hardware-fact
cross-check the design previously lacked. Honest caveats: (a) AIE1/VCK190, not
AIE2/Phoenix -- the *structure* (2/module, horizontal-through-both) transfers; the
*magnitude* is what R3b confirms on Phoenix; (b) the benchmark measures
"horizontal" without splitting E/W, so it **confirms** the E/W-anisotropy
dead-end rather than removing it.

**aie-rt (`driver/src/`).** No per-hop / propagation / skew *constant* anywhere
("skew" is absent from the tree). BUT the reset *topology* is fully specified
(`XAie_SyncTimer` + `_XAie_SetupBroadcastConfig`, Sec. 1) -- a derive-from-toolchain
handle for the tree *shape*, actionable now (feed into R3b geometry + emu
`origin_D`). `Timer_Control.Reset_Event` is the event-select mechanism only.

**llvm-aie (Peano TableGen).** Definitive dead end -- models only intra-core VLIW
pipeline latency (`AIE2Schedule.td`); zero cross-tile / broadcast / per-hop / E/W
content.

**AM025 register database.** Dead end for *values* -- pure bit layouts; no
latency/delay/skew/propagation fields (only the unrelated adaptive-clock-gate idle
timeout carries a time unit).

**aiesimulator (read-only reference).** Not exercised live; low priority now that
the mlir-aie benchmark supplies the independent magnitude cross-check. A possible
*fourth* corroboration, not a new capability.

**Disposition per ceiling component:**

| Component | Toolchain disposition |
|-----------|----------------------|
| Per-hop magnitude / linearity | **Corroborated** (mlir-aie benchmark) + Phoenix-confirmed by R3b; physically explained |
| Reset tree topology | **Toolchain-derived** (aie-rt `XAie_SyncTimer`); actionable in R3b geometry + emu `origin_D` |
| E/W horizontal anisotropy | Confirmed dead-end (all 4 sources) -> disclosed assumption justified, now with a documented magnitude/structure to disclose |
| Absolute intra-offset (gauge) | No source pins it -> stays gauge, disclosed |
| `d_h^{ch15}` in isolation | No source; structurally blended -> disclosed assumption |
| Per-module horizontal split | Benchmark *documents the structure* (2/module) but not a Phoenix E/W or core/mem split -> disclosed, with the structure named |

## 5. Decomposition (ordered; front-loads every structural showstopper)

Ordering principle (the rank-2 lesson): prove identifiability on paper and retire
structural risk in pure-software phases *before* spending any Phoenix window.

| Phase | Kind | Purpose / risk retired | Gate |
|-------|------|------------------------|------|
| **0. Theory lock-down + toolchain sweep** | SW | Rank/identifiability proof for *every* solver column before any kernel goes to silicon; the per-hop-cost-blocking-independence argument (IF-1); magnitude estimate for tolerances/provenance (NOT a kill gate); the Sec. 4 sweep (essentially concluded -- its outputs feed Phase 2 geometry). | Rank proof per column; written skew-vs-latency + IF-1/2/3 + assumption decisions; sweep dispositions recorded. |
| **1. Rust ingestion wire + inert §9a plumbing** | SW | Build the missing `skew_constants.json -> model` path (Sec. 7); §9a fixpoint-ch15 multi-source test; wire the sweep sidecar consumption. All while `calibrated=false`. | Round-trip write->read->model populated, still inert; provenance-hash + non-null-when-calibrated build assertion in place; §9a tests green. |
| **2. Channel + linearity + accessible-array characterization** | HW | Discharge Seams 1 (inter-channel only) & 3; whole *accessible*-array skew map; hard b-vector jitter gate; empirical jitter-immunity. R3b kernel replicates the `XAie_SyncTimer` block tree (IF-1). | Enriched residuals green OR provenance downgraded; per-channel result recorded (ch15-uniformity disclosed-assumed); b-vector range strictly `<` measured `d_h` over N>=20 (RED blocks). |
| **3. Vertical anisotropy + joint sign anchors + d_v cross-check** | HW | Falsify `d_vN != d_vS` via two-sided mid-column R1 spine; both `d_v` sign conventions in one silicon frame; the R1/R3b `d_v` cross-check. | Two-sided spine residual green OR model gains `d_vN/d_vS` split (per attribution rule below); signs agree; R1 `d_v` ~= R3b `d_v` on the *symmetric* component. |
| **4. `dn_v`-correlated `Delta_wall` component** | HW | Bound the lethal mode a green `fit_residual` does NOT clear (row 51): a `Delta_wall` term linear in `dn_v` absorbed into `d_v`. Shares captures with Phase 3. | Symmetric component refuted by the R3b clean average; anti-symmetric residual disclosed as a limit (attribution rule below). |
| **5. Held-out validation kernel** | SW build -> HW run | Break the causal-vs-HW tautology (`skew := HW_offset - Delta_wall_emu` is circular on the fit geometry). | See tolerance definition below. |
| **6. Measure, flip `calibrated`, regression guards** | ONE-WAY, HW-confirmed | The door. Set constants via the Phase-1 wire (on the shaped-tree `origin_D`); flip; 3 regression guards; honest provenance (measured / toolchain-derived / assumed). | Full regression + held-out causal-vs-HW + all prior gates green. |

**Dependencies:** `0 -> {1, 2}` (1 pure-SW, 2 HW, parallel); `2 -> 3`; Phases 3
and 4 share silicon captures; `{3,4} + 5 -> 6`. **[rev2]** Phase 2's R3b enriched
capture and Phase 3's two-sided R1 spine are different instruments with no shared
silicon state -- their *captures* can co-locate in one Phoenix window; only the
`d_v` *reconciliation* is a true 2->3 dependency. Phase 0 gates everything.
Phase 5's kernel *build* parallels 2-4; its *validation run* co-locates with
Phase 6.

**[rev2] Phase 3/4 attribution rule (was "may fold 4 into 3").** R3b's `d_v` is
`Delta_wall`-free but is the `(d_vN+d_vS)/2` average (rank-2 limit); R1's two-sided
`d_v` splits N/S but is `Delta_wall`-contaminated. Comparing R3b's clean average
against R1's `(d_vN+d_vS)/2` isolates and refutes the **symmetric** `Delta_wall`
contamination (R3b is the clean referee). The **anti-symmetric** part -- true
anisotropy vs an anti-symmetric `Delta_wall` error -- remains confounded on R1
alone, because R3b structurally cannot see anti-symmetric vertical structure. So a
grown two-sided-spine residual does **not** uniquely mean anisotropy; it is
attributed as "anisotropy-or-anti-symmetric-`Delta_wall`" and disclosed as a
residual limit unless an independent handle separates them.

**[rev2] Phase 5 tolerance.** The gate is: `causal_offset` predictions match HW
trace deltas on a geometrically-distinct held-out kernel whose `Delta_wall` varies
non-collinearly with per-module skew. Because the prediction is `raw - skew` and
`raw`'s reconstruction uses the emulator's `Delta_wall` on the *new* geometry, the
held-out kernel MUST be one whose `Delta_wall_emu` is *independently* within-domain
byte-exact-verified, and the tolerance MUST be set strictly above that kernel's
known `Delta_wall` residual -- otherwise the gate validates `skew + Delta_wall_emu`
jointly and the tautology-break merely relocates the circularity into an
unverified `Delta_wall`.

## 6. Pre-flip gate mapping (all 7 must hold before the flip)

| # | Gate | Retired in |
|---|------|-----------|
| 1 | Held-out validation kernel (break the tautology) | Phase 5 |
| 2 | Enriched-geometry residuals green (else provenance = "assumed") | Phase 2 (structure/channel) + Phase 3 (vertical) |
| 3 | Joint sign anchors in one silicon frame | Phase 3 |
| 4 | `dn_v`-correlated `Delta_wall` quantified or dismissed | Phase 4 |
| 5 | Hard cross-column b-vector *jitter* gate (RED blocks, not informs) | Phase 2 |
| 6 | §9a(a) fixpoint channel-15 multi-source test | Phase 1 |
| 7 | §9a(b) wire sweep sidecar consumption | Phase 1 |

## 7. The Rust ingestion wire (+ shaped-tree `origin_D`)

**Confirmed gap.** `tools/calibration/skew/schema.py` is a Python-only read/write
for `skew_constants.json`. The only path constants take into the Rust model today
is the test-only runtime seam `set_broadcast_timing_override` (`effects.rs:563`);
compile-time constants are all zero with `calibrated:false`
(`model_builder.rs:270-279`). There is no production reader.

**Recommended shape (Phase 1):** a build-time codegen step that reads the
committed `skew_constants.json` and emits the archspec `BROADCAST_*` consts +
`calibrated` flag -- so the flip is a **reviewable git diff** and the provenance
travels with it (matches the derive/schema-first ethos). Retain the runtime
override test-only. **Extend the schema first** to carry what the honest gates
require: per-channel result, b-vector range, jitter range, and an explicit
`assumptions` field (`horizontal_direction_isotropy: assumed`,
`absolute_intra_offset: gauge`, `ch15_hop_cost: assumed_via_blend`,
`per_module_horizontal_split: not_measured`).

**[rev2] Kill the stale/silent-zero modes** (the `.so`-staleness class we've been
bitten by):
1. Commit a **provenance/content hash** of the measured `skew_constants.json`;
   check it at build time so regenerated-but-not-rebuilt constants fail loud
   rather than silently diverge.
2. Build-time assertion: `calibrated=true` **=>** all constants non-null and
   provenance present -- **fail the build**, never silently fall back to zeros
   (the worse failure mode of a runtime loader: a build that *claims* calibrated
   while uncalibrated).
3. Compile the provenance string in, queryable at runtime.
The hash also closes the deeper gap: neither approach otherwise *verifies the
constants equal the JSON actually measured on silicon*.

**[rev2] Shaped-tree `origin_D`.** The calibrated `origin_D` table generation
(`effects.rs` flood + the SP-4b sidecar) must apply the `XAie_SyncTimer` AIE2
block config (mem-blocks-E / core-blocks-W; memtile blocks E/W), not the current
free flood (Sec. 1). The mask-consulting machinery already exists
(`effects.rs:573-576`); this wires the correct masks at calibration time.

## 8. Ranked risk register

1. **Channel blend `d_h != d_h_ch15` (Seam 1)** -- HIGH impact, HIGH
   late-discovery; ch15-in-isolation is un-retireable, disclosed. Sweep + disclose
   (Phase 2/6).
2. **Non-hop-linear / per-module transport (Seam 3)** -- MED-HIGH impact, HIGHEST
   late-discovery; power-limited by the small accessible array. Phase 2.
3. **[rev2] Free-vs-shaped reset-tree `origin_D` error** -- MED-HIGH impact,
   previously *invisible* (rev1 missed it). Retired by the Sec. 7 shaped-tree fix +
   IF-1 argument (Phase 0/1).
4. **`d_h` single-instrument, no absolute cross-check (Seam 2)** -- HIGH impact,
   structural; disclosure is the sole mitigation.
5. **Tautology in causal-vs-HW gate** -- HIGH impact; Phase 5 held-out kernel +
   independently-verified `Delta_wall`.
6. **`Delta_wall`-linear-in-`dn_v` absorbed into `d_v` (row 51)** -- MED impact,
   HIGH late-discovery; symmetric part refuted by R3b, anti-symmetric disclosed.
   Phase 3/4.
7. **Rank-deficiency recurrence in a new enriched column** -- HIGH if it recurs;
   now LOW late-discovery because Phase 0 proves rank first.
8. **b-vector run-to-run jitter** -- the stochastic-determinism risk the b-vector
   gate actually addresses (distinct from Seam 2's bias). Phase 2, RED blocks.
9. **Per-tile frequency (not phase) skew; clock-tree phase skew** -- LOW impact on
   microsecond windows / partly mooted by shared-event driving. Document; don't
   chase.

## 9. What NOT to do

- **Do not resurrect the 5-param signed E/W + N/S solver.** The rank-2 theorem
  proved those columns unidentifiable from any two-source interval; master's
  `r3b_extract` is correctly `{d_h, d_v}`. Honor the rev3 errata pointers.
- **Do not run the terminal causal-vs-HW gate on the calibration kernel** -- it is
  a tautology there. Held-out kernel only (Phase 5).
- **Do not scope-creep the `d_turn`/interaction column back in** unless a
  deliberate non-corner source placement is chosen and its rank is proven in
  Phase 0 (corner sources add a third null direction -> rank collapses).
- **Do not read a green residual on minimal geometry as shape validation** -- the
  inverse-laundering trap; enriched (accessible-array) geometry earns the claim.
- **[rev2] Do not treat the per-channel sweep as retiring ch15 uniformity** -- it
  retires inter-measurement-channel differences only; ch15-in-isolation stays
  disclosed-assumed.
- **[rev2] Do not compute the calibrated `origin_D` on a free flood** -- use the
  `XAie_SyncTimer` shaped tree.

## 10. Review history

- **rev1 (2026-07-01):** decomposed with a Fable analysis agent; load-bearing
  claims verified against code. Open questions parked for a second Fable round.
- **rev2 (2026-07-02):** independent adversarial Fable review. Resolutions:
  (Q1) toolchain sweep stays in Phase 0 but its *outputs* feed Phase-2 geometry --
  not a blocking gate; the sweep is essentially concluded (Sec. 4). (Q2)
  per-channel sweep is dedicated + minimal + labeled epistemically weaker (retires
  inter-channel only). (Q3) whole-*accessible*-array pass, framed as *falsification
  coverage* (does one `{d_h,d_v}` predict every accessible tile?) not
  identification. (Q4) added ceiling components: per-module horizontal cost,
  clock-tree phase skew, structured OR-tree asymmetry, skew/`Delta_wall` boundary.
  (Q5) build-time codegen confirmed, hardened with provenance-hash +
  non-null-when-calibrated build assertion. Plus the reset-tree-topology
  correctness item (Sec. 1, 7) and the mlir-aie benchmark corroboration (Sec. 4)
  as the two headline additions.
