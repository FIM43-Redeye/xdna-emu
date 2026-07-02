# SP-5c: Comprehensive Skew Characterization + Calibrated Go-Live -- Design

> **Status: DRAFT rev3 (2026-07-02) for review.** Successor to SP-5b (measurement
> apparatus built + silicon bring-up validated, `57c58ac0`; bring-up finding
> `docs/superpowers/findings/2026-07-01-r3b-pc-silicon-bringup.md`). SP-5c is the
> Phoenix characterization campaign and the one-way `calibrated` flip. **This is
> the last Phoenix-HW-gated work before the Strix devbox swap retires the silicon
> (a one-way door); complete, correct characterization while we still have the
> hardware is the mission.**
>
> **rev2 -> rev3 folds in a second independent adversarial Fable review**
> (2026-07-02), which found rev2 had *over-corrected*. rev3 is mostly subtractive.
> Changes, tagged **[rev3]** inline: (1) the "shaped-tree `origin_D`" item is
> demoted from a correctness blocker to inert-in-model; the real requirement is
> that R3b replicate the reset block so `d_h` measures the **shim-row** cost
> (the bring-up measured the AIE-row cost). (2) The `XAie_SyncTimerWithTwoBcstChannel`
> "two channels for reachability" corroboration is **struck** -- verified false
> (both API variants are single-logical-source; the 2nd channel is a col-0 trigger
> relay). (3) The determinism basis is now physical: **XDNA is globally clocked
> (single clock domain, architect-confirmed)**, so broadcast transport is
> deterministic by construction and range-0 is predicted, not lucky. (4) Drift vs
> jitter: the b-vector gate is spaced + re-sampled. (5) The async carve-out is
> widened to name on-chip arbitration. (6) The Phase 3/4 attribution rule gets a
> straddling-geometry precondition; cross-axis additivity is removed as a
> falsification target (structurally true under the shaped tree). (7) Phase 5 must
> name a concrete held-out kernel or it is a paper gate.
>
> rev1 (initial Fable decomposition) and rev2 (first adversarial review) history in
> Sec. 11.

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
justification, whatever provably remains. **We do it now because the Strix swap is
a one-way door that removes the Phoenix silicon.**

**There is no value-of-information kill gate.** Skew's size relative to
`Delta_wall` (hundreds of cycles of real compute/DMA duration) sets *tolerances
and provenance*, not whether the work happens. Scope expansion, where it buys more
honest coverage, is desirable rather than something to minimize.

## 1. What skew is, why "skew vs latency" was a red herring, and the reset path

On AIE2 every tile timer is reset synchronously by a channel-15 broadcast, but the
reset signal *propagates* across the array, so a tile farther from the source
resets its timer later in wall-clock time. Tile X's timer reads zero at wall-clock
`T0 + D(source -> X)`, and the per-module skew is *defined* as

```
origin_D[X] = D(source -> X)   (the ch15 reset broadcast's propagation delay)
```

The emulator consumes this in exactly one place -- `tools/inference/grounding.py:65-79`:

```
causal_offset(child, parent, raw) = raw - (origin_D[parent] - origin_D[child])
```

a pure additive per-tile offset, gated behind `calibrated`. With each tile's trace
timestamps counted from its own post-reset origin,
`raw = Delta_wall + (origin_D[parent] - origin_D[child])`, so the model needs only
*the difference in ch15 reset-arrival times between tiles* -- which is `origin_D`
by definition. **The reset broadcast's transport latency IS the clock skew; the
same physical quantity.** There is no "skew fraction" to split out; the
"skew-vs-latency disentangling" phrasing is retired.

**The reset mechanism (from source), and what it is NOT.** `XAie_SyncTimer`
(`aie-rt/driver/src/timer/xaie_timer.c:823`) fires ONE `XAie_EventGenerate` at shim
`(0,0)` (`:789`), distributes the trigger along the **shim row** via
`XAie_EventBroadcast` across all row-0 columns (`:860-864`), and re-fires the
timer-reset broadcast channel up each column. The reset is a **single logical
source at (0,0)** -- which is why the emulator's single-source contract
(`src/device/state/mod.rs:106-121`) and `grounding.py`'s `CrossDomainModelError`
guard (`:74-77`) are correct. **[rev3] Struck:** rev2 claimed R3b's ch15+ch14
design "mirrors `XAie_SyncTimerWithTwoBcstChannel`, which uses two channels because
one tree can't reach everything." Verified false: the *single*-channel
`XAie_SyncTimer` also uses two channels (`BcastChannelId` and `+1`, `:857`), and
the 2nd channel is configured only at column 0 (`:865-867`) as a trigger relay,
not a reachability extension. R3b's two floods (interval differencing) and the
reset's trigger relay are unrelated mechanisms; no corroboration there.

The identity rests on three IFs:

- **IF-1 (path fidelity):** R3b must measure the per-hop cost on the *same physical
  path* the reset uses (see the shaped tree below).
- **IF-2 (linearity):** transport is hop-count-linear -- no per-tile fixed latency
  disproportionate to hop count.
- **IF-3 (arrival->latch uniformity):** between broadcast-event arrival and the
  timer latching to zero there is a `Reset_Event`->latch step; if uniform across
  tiles it cancels in the `origin_D` difference, if per-module-kind it folds into
  the (gauge/contrast) intra-tile offset. Named so the cancellation is argued.

**[rev3] The shaped reset tree, correctly framed.** `_XAie_SetupBroadcastConfig`
(`xaie_timer.c:512-534`) for AIE2 blocks **MEM-module EAST and CORE-module WEST on
every AIE tile** and both E/W on memtiles; shim tiles (row 0) are unblocked. Since
a horizontal broadcast physically traverses *both* the mem and core module to
reach the next tile (mlir-aie benchmark 10, Sec. 4), blocking both severs
horizontal handoff at AIE/memtile rows. **The reset therefore routes horizontally
only on the shim row, then climbs each column** -- matching the trigger-relay
mechanism above.

Two consequences, corrected from rev2:

1. **[rev3] The emulator `origin_D` "shaped-tree fix" is inert in the current
   model, NOT a correctness blocker.** The model uses one scalar `d_h` for all
   rows (`effects.rs:491-494`), so the free-flood Dijkstra cost `c*d_h + r*d_v`
   equals the shaped horizontal-then-vertical cost `c*d_h + r*d_v` exactly (for the
   shim-row source; blocking only prunes equal-cost redundant paths, changing no
   shortest-path cost and no reachability). Programming the block masks in the
   `origin_D` computation is therefore a no-op on the numbers -- retained only for
   generality (non-corner source / a future non-uniform model), not as a flip
   gate. rev2's risk #3 ("MED-HIGH, previously invisible") over-scoped this.
2. **[rev3] The real requirement: `d_h` must be the shim-row cost, and the bring-up
   measured the AIE-row cost.** Because the reset routes horizontally only on the
   shim row, the physically-correct value for the model's `d_h` is the **shim-row**
   per-hop cost (benchmark 9: shim->shim = 4 cy). The bring-up `d_h≈4` was measured
   with a free-flood R3b kernel whose cores sit on AIE row 3, so it measured
   **AIE-row** horizontal transport (benchmark 10: ~4 cy through both modules) -- a
   different mechanism that coincides in magnitude but is not guaranteed equal on
   Phoenix. **Fix (IF-1): R3b's kernel replicates the `XAie_SyncTimer` block config,
   forcing its horizontal hops onto the shim row so `d_h` is measured as the cost
   `origin_D` actually consumes.** This makes the rev2 "per-hop cost is
   blocking-independent" hand-wave unnecessary -- consistent block replication makes
   self-consistency automatic.

*(Verified: `grounding.py:65-79`; `effects.rs:446-506,573-580`;
`xaie_timer.c:472-542,789,823,857-868`; single-source `mod.rs:106-121`.)*

## 1.5. [rev3] Determinism basis (the physical foundation, not just a gate)

The method assumes the R3b perf-counter interval is deterministic. That is a
*prediction*, not a lucky bring-up artifact, and rev3 states why:

**XDNA is globally clocked -- the entire AIE array runs on one clock domain
(architect-confirmed).** Under a single clock domain, event-broadcast transport is
registered/combinational with fixed setup/hold: no metastability, no run-to-run
variation. Genuine async metastability appears only at a *frequency* boundary,
which on this path is exactly the NoC egress (AIE clock <-> NoC clock) that
`is_async_cdc` already carves out (`grounding.py:82-92`). So **range-0 is what the
single-domain model predicts**; the bring-up (range-0, N=3) confirms it. The
~30-cycle figure that once spooked us was measured between two cores' `LOCK_STALL`
events (data-dependent compute/lock, soundness audit Q4) -- **not** a broadcast
arrival; there is no evidence broadcast transport carries it. The b-vector gate
(Sec. 6) is a **backstop** to this physical argument, not the whole case.

**What the physical argument does NOT cover (and rev3 defends explicitly):**

- **Drift, not jitter.** A bias that varies slowly across runs (thermal settling, a
  DVFS P-state change, PLL wander over minutes) is neither fast jitter (caught by
  max-min if runs straddle it) nor a fixed deterministic bias. The b-vector gate's
  N runs must therefore be **spaced across the drift-relevant timescale** (not just
  back-to-back microseconds) and **re-sampled at the Phase-6 flip**, not only at
  Phase-2 calibration. A P-state shift that scales the counter clock is first-order
  cancelled in the *differenced* b-vector but not in the *absolute* per-hop
  constants `origin_D` consumes -- a disclosed second-order line.
- **On-chip contention, not just NoC egress.** `is_async_cdc` covers frequency-
  boundary metastability only. On-chip stream-switch packet arbitration under
  contention (`model_builder.rs:267`, `packet_arbitration_overhead`) has
  arrival-order nondeterminism that is neither `origin_D` nor a deterministic
  `Delta_wall`, and is *not* carved out. **The calibration and held-out kernels must
  avoid contended cross-domain edges** (the bring-up's control-packet readback path
  was uncontended -- part of why it was clean); rev3 makes this a kernel-design
  constraint and widens the carve-out *reasoning* to name on-chip arbitration as a
  distinct nondeterministic class.

## 2. What R3b measures

R3b runs two floods (s1 on ch15, s2 on ch14); each measured tile arms
`Performance_Counter0` START on one arrival, STOP on the other, recording
`r_X = D(s2->X) - D(s1->X) + (T0_2 - T0_1)`. A least-squares fit recovers
`{d_h, d_v}`; the host-gap constant cancels by differencing against a reference tile
(`r3b_extract.py`). R3b's per-hop constants equal the reset per-hop skew iff IF-1,
IF-2, IF-3 hold -- and IF-1 now requires **block replication** so `d_h` is the
shim-row cost (Sec. 1).

## 3. The seams and the honest ceiling

Ranked by impact x likelihood-of-late-discovery.

**Seam 1 -- channel blend (HIGH).** R3b's counter is START=ch15 / STOP=ch14, and
`d_h`/`d_v` are fitted as channel-agnostic scalars (`effects.rs:485,491-494`;
`r3b_extract.py:20`), so the recovered `d_h` is a ch14/ch15 blend applied to a
ch15-only flood. On collinear (and the corner-source bring-up) geometry this fits
at zero residual with a biased average, so the ~7e-15 bring-up residual does not
prove channel uniformity. **ch15-in-isolation is structurally unmeasurable by R3b**
(it is always one leg of the blend), like horizontal E/W anisotropy. The Phase-2
vary-s2-channel sweep detects inter-*measurement*-channel differences only;
**ch15 hop-cost uniformity stays a disclosed assumption.**

**Seam 2 -- `d_h` has no absolute-frame cross-check (HIGH, structural).** R1 can
cross-check `d_v`; R1 cannot measure `d_h` (horizontal absolute offsets ride the
cross-column path R3b's interval design dodges). **Disclosure is the only mitigation
for `d_h`'s value.** The b-vector gate checks run-to-run *range* (jitter), and a
biased `d_h` is deterministic -- it passes range-0; the gate addresses the
determinism backstop (Sec. 1.5), not systematic bias.

**Seam 3 -- non-hop-linear / non-per-column transport (MED impact, HIGHEST
late-discovery).** A linear model absorbs the hop-linear part and hides the rest in
the (cancelled) const or residual; a green residual on minimal geometry cannot
distinguish "truly hop-linear" from "linear-enough on a few tiles." Two refinements:
(a) the "hop" is per-*module*, not per-column -- a horizontal hop is two module
traversals (core + mem, ~2 cy each); the single per-column `d_h` coarsens this, and
core-vs-mem differences live in `d_h` unmodeled. (b) Phoenix's *usable* array is
small -- physical col 0 is inaccessible (rewritten-inaccessible), leaving ~4 cols
and ~4 core rows, so "longer baselines" have <=4 hops of leverage. Phase 2's map is
**all accessible cols**, and its power ceiling is acknowledged.

**The honest ceiling (provably unmeasurable from R3b/R1):**

- **Horizontal within-axis direction anisotropy** (`d_hE` vs `d_hW`):
  `a_hE - a_hW = s1.col - s2.col` is a per-tile constant -> E/W collapse. (Vertical
  `d_vN` vs `d_vS` IS falsifiable via the two-sided R1 spine -- Phase 3.)
- **Absolute intra-tile offset** is gauge; only `core_off - mem_off` observable.
- **`d_h^{ch15}` in isolation** (Seam 1).
- **Per-module horizontal cost** (core-module vs mem-module traversal).
- **Clock-tree phase skew** independent of the event network: partly mooted because
  reset and counter are driven by the same event (under IF-3, and the single clock
  domain of Sec. 1.5). One disclosed sentence.
- **Structured OR-tree asymmetry** aligned with an axis: can evade the
  longer-baseline test, absorbing into `const` or `d_h`.

"Characterize everywhere" resolves to **"characterize everything identifiable; find
independent toolchain-derived handles for the rest (Sec. 4); disclose the remainder
with justification, at the flip, not as silent zeros."**

**The skew / `Delta_wall` boundary.** `grounding.py:78` applies `origin_D` to every
cross-domain edge, but stream/DMA-mediated edges carry stream-switch pipeline
latency (`model_builder.rs:260-268`) that is `Delta_wall`'s job, not `origin_D`'s.
One explicit provenance sentence so the two corrections neither double-count nor
gap (and see the on-chip-contention nondeterminism of Sec. 1.5).

## 4. Toolchain-derived observability for the ceiling components

Desk sweep across four sources (no Phoenix window):

**mlir-aie `test/benchmarks/` -- the corroboration and the single highest-value
find.** `README.md:61-74` carries empirical perf-counter-measured broadcast delays
(VCK190/AIE1 @1 GHz): **horizontal 2 cy per core/memory module, through both
modules to the next tile => ~4 cy/column-hop** (benchmark 10, with the physical
reason: 16 horizontal wires vs 32 vertical); **vertical 2 cy/tile** (11);
**shim->shim 4 cy** (9); **stream 2 cy/node** (12). Restated in
`mlir_exercises/tutorial-4/flow/README.md:147`; reset skew acknowledged in
`python/utils/trace/__init__.py:165`. This independently explains the bring-up
`d_h≈4` / `d_v≈2`. Caveats: AIE1/VCK190 not AIE2/Phoenix (structure transfers,
magnitude confirmed by R3b); "horizontal" is not E/W-split, confirming the
E/W-anisotropy dead-end. **[rev3] Note the shim->shim (benchmark 9) vs
AIE-row-horizontal (benchmark 10) distinction is exactly the Sec. 1 point: the
reset uses shim-row horizontal, so `d_h` must be calibrated there.**

**aie-rt.** No per-hop/skew *constant* (the word "skew" is absent), but the reset
*topology* is fully specified (`XAie_SyncTimer` + `_XAie_SetupBroadcastConfig`) --
derive-from-toolchain handle for the tree shape and the shim-row horizontal path.

**llvm-aie (Peano TableGen).** Dead end -- only intra-core VLIW pipeline latency.

**AM025 register database.** Dead end for values -- pure bit layouts.

**aiesimulator (read-only).** Not exercised; a possible fourth corroboration, low
priority now.

| Component | Toolchain disposition |
|-----------|----------------------|
| Per-hop magnitude / linearity | Corroborated (mlir-aie) + Phoenix-confirmed by R3b; physically explained |
| Reset tree topology + shim-row horizontal path | Toolchain-derived (aie-rt); shapes R3b geometry + `d_h` calibration target |
| E/W horizontal anisotropy | Confirmed dead-end (all 4) -> disclosed assumption |
| Absolute intra-offset (gauge) | No source -> stays gauge, disclosed |
| `d_h^{ch15}` in isolation | Structurally blended -> disclosed assumption |
| Per-module horizontal split | Structure documented (2/module), no Phoenix core/mem split -> disclosed |

## 5. Decomposition (front-loads every structural showstopper)

Prove identifiability on paper and retire structural risk in pure-SW phases before
any Phoenix window.

| Phase | Kind | Purpose / risk retired | Gate |
|-------|------|------------------------|------|
| **0. Theory lock-down + toolchain sweep** | SW | Rank/identifiability proof per solver column; the determinism physical basis (Sec. 1.5) written down; IF-1/2/3 + assumption decisions; magnitude estimate for tolerances (NOT a kill gate); sweep dispositions (Sec. 4). | Rank proof per column; determinism basis + IF decisions recorded; sweep concluded. |
| **1. Rust ingestion wire + inert §9a plumbing** | SW | Build the missing `skew_constants.json -> model` path (Sec. 7); §9a fixpoint-ch15 multi-source test; wire the sweep sidecar consumption. All while `calibrated=false`. | Round-trip write->read->model populated, still inert; provenance-hash + non-null-when-calibrated build assertion; §9a tests green. |
| **2. Channel + linearity + accessible-array characterization** | HW | Discharge Seams 1 (inter-channel only) & 3; whole *accessible*-array map; **R3b kernel replicates the `XAie_SyncTimer` block so `d_h` = shim-row cost (IF-1)**; b-vector jitter gate spaced across the drift timescale (Sec. 1.5); uncontended cross-domain edges only. | Enriched residuals green OR provenance downgraded; per-channel result recorded (ch15 disclosed-assumed); b-vector range strictly `<` measured `d_h` over N>=20 *spaced* runs (RED blocks). |
| **3. Vertical anisotropy + joint sign anchors + d_v cross-check** | HW | Falsify `d_vN != d_vS` via two-sided mid-column R1 spine (**sources must straddle the measured tiles vertically**, or R3b's average degenerates -- see attribution rule); both `d_v` signs in one frame; R1/R3b `d_v` cross-check. | Two-sided spine residual green OR `d_vN/d_vS` split; signs agree; R1 `d_v` ~= R3b `d_v` on the symmetric component. |
| **4. `dn_v`-correlated `Delta_wall` component** | HW | Bound the mode a green residual does NOT clear (row 51). Shares captures with Phase 3. | Symmetric component refuted by R3b's clean average; anti-symmetric residual disclosed (attribution rule). |
| **5. Held-out validation kernel** | SW build -> HW run | Break the causal-vs-HW tautology. **[rev3] Blocked on naming a concrete kernel that is BOTH `Delta_wall`-distinct AND already within-domain byte-exact-verified** (byte-exact on arbitrary geometry is not yet achieved -- `clean_release(Aie2)` red -- so this is a real dependency, not a formality). | `causal_offset` predictions match HW deltas on the named kernel; tolerance strictly above that kernel's known `Delta_wall` residual. |
| **6. Measure, flip `calibrated`, regression guards** | ONE-WAY, HW-confirmed | The door. Set constants via Phase-1 wire; **re-sample the b-vector at the flip** (Sec. 1.5 drift); 3 regression guards; honest provenance (measured / toolchain-derived / assumed). | Full regression + held-out causal-vs-HW + all prior gates green. |

**Dependencies:** `0 -> {1, 2}`; `2 -> 3`; Phases 3 & 4 share silicon captures;
`{3,4} + 5 -> 6`. Phase 2's R3b capture and Phase 3's R1 spine are different
instruments -- captures can co-locate in one Phoenix window; only the `d_v`
reconciliation is a true 2->3 dependency. Phase 0 gates everything. Phase 5's kernel
*build* parallels 2-4; its *validation run* co-locates with Phase 6.

**Phase 3/4 attribution rule.** R3b's `d_v` is `Delta_wall`-free but is the
`(d_vN+d_vS)/2` **symmetric average** *when the sources straddle the measured tiles
vertically* (bring-up: s1 below, s2 above -- **[rev3] a geometry precondition, not a
free property**). R1's two-sided `d_v` splits N/S but is `Delta_wall`-contaminated
(`d_vN+eN`, `d_vS+eS`). R1's average - R3b's clean average = `(eN+eS)/2` = the
**symmetric** contamination -> refutable by R3b. R1's difference =
`(d_vN-d_vS)+(eN-eS)` -> the **anti-symmetric** part confounds true anisotropy with
anti-symmetric `Delta_wall` error, and R3b is structurally blind to it. So a grown
two-sided residual is attributed as "anisotropy-or-anti-symmetric-`Delta_wall`" and
disclosed as a residual limit unless an independent handle separates them. (Fable
re-derived this algebra; correct.)

## 6. Pre-flip gate mapping (all 7 must hold before the flip)

| # | Gate | Retired in |
|---|------|-----------|
| 1 | Held-out validation kernel (concrete, byte-exact, `Delta_wall`-distinct) | Phase 5 |
| 2 | Enriched-geometry residuals green (else provenance = "assumed") | Phase 2 + Phase 3 |
| 3 | Joint sign anchors in one silicon frame | Phase 3 |
| 4 | `dn_v`-correlated `Delta_wall` quantified or dismissed | Phase 4 |
| 5 | Hard b-vector *jitter* gate, spaced + re-sampled at flip (RED blocks) | Phase 2 + Phase 6 |
| 6 | §9a(a) fixpoint channel-15 multi-source test | Phase 1 |
| 7 | §9a(b) wire sweep sidecar consumption | Phase 1 |

## 7. The Rust ingestion wire (+ shim-row `d_h`)

**Confirmed gap.** `tools/calibration/skew/schema.py` is Python-only; constants
reach the model today only via the test-only `set_broadcast_timing_override`
(`effects.rs:563`); compile-time constants are zero with `calibrated:false`
(`model_builder.rs:270-279`). No production reader.

**Recommended shape (Phase 1):** build-time codegen reading the committed
`skew_constants.json` -> archspec `BROADCAST_*` consts + `calibrated` flag (the flip
is a reviewable git diff; provenance travels). Retain the runtime override
test-only. **Extend the schema first** to carry: per-channel result, b-vector range,
jitter range, and an `assumptions` field (`horizontal_direction_isotropy: assumed`,
`absolute_intra_offset: gauge`, `ch15_hop_cost: assumed_via_blend`,
`per_module_horizontal_split: not_measured`, `d_h_path: shim_row`).

**Kill the stale/silent-zero modes** (the `.so`-staleness class): (1) commit a
provenance/content hash of the measured JSON, checked at build; (2) build assertion
`calibrated=true => all constants non-null and provenance present` -- **fail the
build**, never silently zero; (3) compile the provenance string in, runtime-queryable.

**[rev3] `origin_D` tree note (demoted):** programming the `XAie_SyncTimer` block
masks in the `origin_D` flood is inert on the numbers under the single-scalar model
(Sec. 1) -- retain for generality, not as a flip gate. The load-bearing action is on
the *measurement* side: R3b measures `d_h` on the shim row.

## 8. Ranked risk register

1. **Channel blend `d_h != d_h_ch15` (Seam 1)** -- HIGH/HIGH; ch15-in-isolation
   un-retireable, disclosed. Sweep + disclose (Phase 2/6).
2. **Non-hop-linear / per-module transport (Seam 3)** -- MED-HIGH impact, HIGHEST
   late-discovery; power-limited by the small accessible array. Phase 2.
3. **[rev3] `d_h` measured on the wrong horizontal path** (AIE-row free flood vs the
   reset's shim-row path) -- MED-HIGH; retired by R3b block replication (Phase 2,
   IF-1). (Supersedes rev2's mis-scoped "free-vs-shaped `origin_D`" risk, which is
   inert.)
4. **`d_h` single-instrument, no absolute cross-check (Seam 2)** -- HIGH, structural;
   disclosure only.
5. **Tautology in causal-vs-HW gate** -- HIGH; Phase 5 named held-out kernel +
   independently-verified `Delta_wall`.
6. **`Delta_wall`-linear-in-`dn_v` absorbed into `d_v` (row 51)** -- MED, HIGH
   late-discovery; symmetric refuted by R3b, anti-symmetric disclosed. Phase 3/4.
7. **[rev3] Drift (thermal/DVFS/PLL) across the calibrate->use timescale** -- MED,
   HIGH late-discovery; the back-to-back b-vector gate is blind to it. Spaced runs +
   flip-time re-sample (Sec. 1.5).
8. **[rev3] On-chip stream-arbitration nondeterminism on contended cross-domain
   edges** -- MED; not covered by `is_async_cdc`. Kernel-design constraint: avoid
   contended edges (Sec. 1.5).
9. **Rank-deficiency recurrence in a new enriched column** -- HIGH if it recurs; LOW
   late-discovery (Phase 0 proves rank first).
10. **b-vector run-to-run jitter** -- the stochastic risk the gate directly
    addresses; predicted absent by the single-clock-domain basis (Sec. 1.5). Phase 2,
    RED blocks.
11. **Per-tile frequency / clock-tree phase skew** -- LOW on microsecond windows /
    partly mooted by the single domain + shared-event driving. Document; don't chase.

## 9. What NOT to do

- **Do not resurrect the 5-param signed E/W + N/S solver** (rank-2 unidentifiable;
  master's `r3b_extract` is correctly `{d_h, d_v}`).
- **Do not run the terminal causal-vs-HW gate on the calibration kernel** -- a
  tautology; named held-out kernel only (Phase 5).
- **[rev3] Do not list cross-axis additivity as a falsification target** -- under the
  shaped tree (horizontal-then-vertical routing) additivity is structurally true;
  there is no turn to penalize. (This dissolves soundness-audit Q1's "additivity
  unfalsifiable-and-assumed" worry -- it's guaranteed by the reset topology.) The
  optional `d_turn` column stays deferred and now moot for the reset.
- **Do not read a green residual on minimal geometry as shape validation** -- the
  inverse-laundering trap; accessible-array geometry earns it.
- **Do not treat the per-channel sweep as retiring ch15 uniformity** -- inter-channel
  only; ch15 stays disclosed-assumed.
- **[rev3] Do not sell the shaped-tree `origin_D` mask-programming as a correctness
  fix** -- it is inert in the single-scalar model; the real action is measuring `d_h`
  on the shim row.
- **[rev3] Do not gate determinism solely on the b-vector range** -- lead with the
  single-clock-domain physical argument; the gate is a spaced/re-sampled backstop.

## 10. [rev3] Open questions for the third review round

1. Is the single-clock-domain determinism basis (Sec. 1.5) stated strongly enough,
   and is any nondeterministic path still mis-classified as deterministic (beyond NoC
   egress + on-chip arbitration)?
2. Does R3b block-replication (IF-1, Phase 2) have a hidden failure -- e.g. does
   forcing horizontal onto the shim row change reachability of the measured cores or
   the rank of the fit, given Phoenix's small accessible array?
3. Is there a concrete existing kernel that satisfies Phase 5 (both `Delta_wall`-
   distinct and byte-exact-verified), or must one be built -- and does that make
   Phase 5 the campaign's critical path?
4. Any remaining rev1/rev2/rev3 internal contradiction or over-correction.
5. Is the Phase 0 + Phase 1 pure-SW slice safe to build and plan now, independent of
   the HW-phase corrections?

## 11. Review history

- **rev1 (2026-07-01):** Fable analysis-agent decomposition; claims verified against
  code.
- **rev2 (2026-07-01/02):** first adversarial Fable review -- added the mlir-aie
  benchmark corroboration, the reset-tree topology, IF-3, Seam 1/2 corrections, the
  Phase 3/4 attribution rule, the Phase 5 tolerance, and the ingestion-wire guards.
- **rev3 (2026-07-02):** second adversarial Fable review -- *subtractive*. Demoted the
  shaped-tree `origin_D` fix to inert (real requirement: shim-row `d_h` via block
  replication); struck the false two-channel corroboration; added the single-clock-
  domain determinism basis (architect-confirmed) with drift + on-chip-contention
  defenses; pinned the straddling-geometry precondition; removed additivity as a
  falsification target; made Phase 5 depend on a concrete named kernel. Verdict: GO to
  build/plan Phase 0 + Phase 1 now; rev3 required before HW phases and the flip.
