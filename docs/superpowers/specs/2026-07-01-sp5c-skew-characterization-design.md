# SP-5c: Comprehensive Skew Characterization + Calibrated Go-Live -- Design

> **Status: DRAFT rev4 (2026-07-02), ready to plan.** Successor to SP-5b
> (measurement apparatus built + silicon bring-up validated, `57c58ac0`; bring-up
> finding `docs/superpowers/findings/2026-07-01-r3b-pc-silicon-bringup.md`). SP-5c
> is the Phoenix characterization campaign and the one-way `calibrated` flip.
> **This is the last Phoenix-HW-gated work before the Strix devbox swap retires the
> silicon (a one-way door); complete, correct characterization while we still have
> the hardware is the mission.**
>
> **Three adversarial Fable review rounds** (rev1->rev4). rev4 folds in round 3,
> which found a real silicon-corrupting defect: rev3's IF-1 "one block-replicated
> R3b capture fits `{d_h, d_v}`" **silently destroys `d_v` identifiability** (the
> vertical spine collapses to a common-mode constant under block routing). rev4
> **decouples the captures** -- `d_h` from a block-replicated / shim-row capture
> (scoped `d_h`-only), `d_v` from a free-flood R3b capture and/or R1 -- because each
> constant must be measured where the reset actually realizes it. Changes tagged
> **[rev4]**. Full round history in Sec. 11.

Issue #140, timer-sync faithful-broadcast arc. Sub-project SP-5c of the SP-5
decomposition (5a calibration enablement [done], 5b measurement apparatus +
bring-up [done], 5c comprehensive characterization + go-live [this]).

---

## 0. Mission

**Characterize the entire per-tile clock-skew field of the Phoenix (NPU1/AIE2)
array, honestly and completely, then flip the emulator's `calibrated` model live
on measured constants.** "Completely" is a deliberate scope choice: the smallness
of skew (single-digit cycles per hop) is *not* a licence to skip it. This is a
cycle-accurate emulator; the skew field is a real hardware quantity. We measure all
of it that is measurable, find independent toolchain-derived handles for the rest,
and *disclose*, with physical justification, whatever provably remains. **We do it
now because the Strix swap is a one-way door that removes the Phoenix silicon.**

**No value-of-information kill gate.** Skew's size relative to `Delta_wall`
(hundreds of cycles of real compute/DMA duration) sets *tolerances and provenance*,
not whether the work happens. Scope expansion that buys more honest coverage is
desirable.

## 1. What skew is, why "skew vs latency" was a red herring, and the reset path

On AIE2 every tile timer is reset synchronously by a channel-15 broadcast, but the
reset signal *propagates*, so a farther tile resets its timer later in wall-clock
time. Tile X's timer reads zero at `T0 + D(source -> X)`, so the per-module skew is
*defined* as

```
origin_D[X] = D(source -> X)   (the ch15 reset broadcast's propagation delay)
```

The emulator consumes this in one place -- `grounding.py:65-79`:
`causal_offset(child, parent, raw) = raw - (origin_D[parent] - origin_D[child])`, a
pure additive per-tile offset gated behind `calibrated`. Since
`raw = Delta_wall + (origin_D[parent] - origin_D[child])`, the model needs only the
difference in reset-arrival times between tiles -- which is `origin_D`. **The reset
broadcast's transport latency IS the clock skew.** "skew-vs-latency disentangling"
is retired.

**The reset mechanism (from source).** `XAie_SyncTimer` (`xaie_timer.c:823`) fires
ONE `XAie_EventGenerate` at shim `(0,0)` (`:789`), distributes the trigger along the
**shim row** across all row-0 columns (`:860-864`), and re-fires the reset channel
up each column. It is a **single logical source at (0,0)** -- which is why the
emulator's single-source contract (`mod.rs:106-121`) and `grounding.py`'s
`CrossDomainModelError` guard (`:74-77`) are correct. *(Struck in rev3: the "R3b
mirrors the two-channel API" corroboration -- both API variants use two channels,
the 2nd is a col-0 trigger relay, not reachability.)*

The identity rests on three IFs: **IF-1 (path fidelity)** -- each constant measured
on the same physical path the reset uses; **IF-2 (linearity)** -- hop-count-linear
transport; **IF-3 (arrival->latch uniformity)** -- the `Reset_Event`->latch step
uniform (cancels) or per-kind (folds into the gauge intra-offset).

**The shaped reset tree.** `_XAie_SetupBroadcastConfig` (`xaie_timer.c:512-534`) for
AIE2 blocks **MEM-EAST + CORE-WEST on every AIE tile** and both E/W on memtiles;
shim row 0 is unblocked. Since a horizontal broadcast physically traverses *both*
modules to reach the next tile (mlir-aie benchmark 10, Sec. 4), the reset **routes
horizontally only on the shim row, then climbs each column**.

1. **`origin_D` mask-programming is inert (not a correctness blocker).** The model
   uses one scalar `d_h` for all rows (`effects.rs:491-494`), so free-flood Dijkstra
   cost `c*d_h + r*d_v` equals the shaped horizontal-then-vertical cost exactly for
   the shim-row source; blocking only prunes equal-cost paths. Retained for
   generality only. **[rev4] Caveat:** `broadcast_origin_d` reads a *single* module
   mask per tile (`effects.rs:482`), modeling one node per tile, so it cannot
   represent the two-module horizontal severing -- inert for the corner reset, but
   the "retain for generality" claim over-promises for a non-corner source without
   per-module nodes. Do not bank on it.
2. **[rev4] The real requirement -- and the `d_v` trap.** Because the reset routes
   horizontally only on the shim row, the model's `d_h` must be the **shim-row**
   per-hop cost (benchmark 9: shim->shim = 4 cy). The bring-up `d_h≈4` was a
   free-flood measurement on AIE row 3 -- **AIE-row** transport (benchmark 10: ~4 cy
   through both modules), coincident in magnitude, different mechanism. **But
   replicating the block to fix `d_h` breaks `d_v`:** under block routing both floods
   reach a measured tile by climbing its column from the shim, so the vertical climb
   is common-mode and cancels in the interval -- the whole vertical spine collapses
   to a constant, leaving `d_v` resting on a single tile (rank survives at 2 but with
   no redundancy or linearity check; a wrong-but-clean `d_v`). **Therefore the
   captures are decoupled (Sec. 2, Phase 2/3):** a block-replicated / shim-row
   capture measures `d_h` **only**; `d_v` is measured by a **free-flood** capture
   (vertical is unblocked in the real reset, so free-flood vertical *is* the reset's
   vertical mechanism) and/or R1.

*(Verified: `grounding.py:65-79`; `effects.rs:446-506,573-580`;
`xaie_timer.c:472-542,789,823,857-868`; `mod.rs:106-121`. The `d_v`-collapse algebra
re-derived on `sp5_skew_r3b_pc/geometry.json`: col-1 spine -> constant `5*d_v`.)*

## 1.5. Determinism basis (the physical foundation, not just a gate)

**XDNA is globally clocked -- the entire AIE array is one clock domain
(architect-confirmed).** Event-broadcast transport is then registered/combinational
with fixed setup/hold: no metastability, no run-to-run variation. Metastability
appears only at a *frequency* boundary -- the NoC egress (AIE<->NoC clock) that
`is_async_cdc` already carves out (`grounding.py:82-92`). So **range-0 is what the
single-domain model predicts**; the bring-up (range-0, N=3) confirms it. The
~30-cycle figure that once worried us was measured between two cores' `LOCK_STALL`
events (data-dependent compute, audit Q4) -- not broadcast arrival. The b-vector gate
is a **backstop** to this argument.

**[rev4] The range-0 gate also covers readback *ordering* determinism:**
`observe_r3b` indexes by `counter_index` (buffer position), so a run-to-run response
reorder at the shim S2MM would break range-0 -- ordering-determinism is gated, not
just value-jitter. State this explicitly.

**What the physical argument does NOT cover, defended explicitly:**

- **Drift, not jitter** (thermal settling, DVFS P-state change, PLL wander over
  minutes): neither fast jitter nor fixed bias. The b-vector `N` runs must be
  **spaced across the drift timescale** (not back-to-back microseconds) and
  **re-sampled at the Phase-6 flip**. A clock-scaling P-state shift is first-order
  cancelled in the *differenced* b-vector but not in the *absolute* per-hop constants
  `origin_D` consumes -- a disclosed second-order line.
- **On-chip contention, not just NoC egress:** stream-switch packet arbitration under
  contention (`model_builder.rs:267`) has arrival-order nondeterminism that is
  neither `origin_D` nor deterministic `Delta_wall` and is not carved out. **The
  calibration and held-out kernels must avoid contended cross-domain edges** (the
  bring-up readback path was uncontended -- part of why it was clean). Widen the
  carve-out *reasoning* to name on-chip arbitration as a distinct nondeterministic
  class.

## 2. What the R3b captures measure (decoupled)

R3b runs two floods (s1 on ch15, s2 on ch14); each measured tile arms
`Performance_Counter0` START on one arrival, STOP on the other, recording
`r_X = D(s2->X) - D(s1->X) + (T0_2 - T0_1)`; a least-squares fit over tiles recovers
per-hop constants after the host-gap constant cancels by reference differencing
(`r3b_extract.py`). **[rev4] Two captures, each valid for one axis:**

- **`d_h` capture (block-replicated or shim-row).** Replicates the `XAie_SyncTimer`
  block so horizontal is realized on the shim row (IF-1); `d_h` is the cost `origin_D`
  consumes. **Scoped `d_h`-only** -- `d_v` is not identifiable here (Sec. 1 trap).
- **`d_v` capture (free-flood).** Vertical is unblocked in the real reset, so a
  free-flood R3b measures the reset's vertical transport directly, with full spine
  leverage (>=3 collinear tiles). Cross-checked against R1 (Phase 3).

**[rev4] `observe_r3b` must match the programmed routing.** `r3b_observe.py:18-21,
41-44` computes coefficients from free-flood Manhattan hops; the `d_h` (block-routed)
capture needs a coefficient model that reflects shim-row-then-climb routing, or the
fit runs against mismatched coefficients. This is a Phase-0/Phase-1 code task.

R3b's per-hop constants equal the reset per-hop skew iff IF-1, IF-2, IF-3 hold.

## 3. The seams and the honest ceiling

Ranked by impact x likelihood-of-late-discovery.

**Seam 1 -- channel blend (HIGH).** The `d_h` capture's counter is
START=ch15 / STOP=ch14, and `d_h` is fitted channel-agnostic (`effects.rs:485,
491-494`), so it is a ch14/ch15 blend applied to a ch15-only flood; a collinear fit
hides it at zero residual. **ch15-in-isolation is structurally unmeasurable** (always
one leg of the blend). The Phase-2 vary-s2-channel sweep detects inter-*measurement*-
channel differences only; **ch15 hop-cost uniformity stays a disclosed assumption.**

**Seam 2 -- `d_h` has no absolute-frame cross-check (HIGH, structural).** R1 can
cross-check `d_v`, not `d_h`. **Disclosure is the only mitigation for `d_h`'s value.**
The b-vector gate checks *range* (jitter); a biased `d_h` is deterministic and passes
range-0 -- the gate is the determinism backstop, not a bias check.

**Seam 3 -- non-hop-linear / non-per-column transport (MED impact, HIGHEST
late-discovery).** A green residual on minimal geometry cannot distinguish
"hop-linear" from "linear-enough on a few tiles." (a) The "hop" is per-*module*: a
horizontal hop is two module traversals (~2 cy each); the single per-column `d_h`
coarsens this. (b) Phoenix's usable array is small -- col 0 inaccessible, ~4 cols and
~4 core rows, so baselines have <=4 hops of leverage. Phase 2's map is **all
accessible cols**, ceiling acknowledged.

**The honest ceiling (provably unmeasurable from R3b/R1):** horizontal E/W anisotropy
(`a_hE-a_hW = s1.col-s2.col` const -> E/W collapse; vertical `d_vN`/`d_vS` IS
falsifiable via the two-sided R1 spine); absolute intra-offset (gauge, only
`core_off-mem_off` observable); `d_h^{ch15}` in isolation (Seam 1); per-module
horizontal cost; clock-tree phase skew (partly mooted by the single clock domain +
shared-event driving under IF-3); structured OR-tree axis-aligned asymmetry.
"Characterize everywhere" = "characterize everything identifiable; find
toolchain-derived handles for the rest (Sec. 4); disclose the remainder at the flip,
not as silent zeros."

**Skew / `Delta_wall` boundary.** `grounding.py:78` applies `origin_D` to every
cross-domain edge, but stream/DMA-mediated edges carry stream-switch pipeline latency
(`model_builder.rs:260-268`) that is `Delta_wall`'s job. One provenance sentence so
the corrections neither double-count nor gap.

## 4. Toolchain-derived observability for the ceiling components

Desk sweep, four sources, no Phoenix window:

**mlir-aie `test/benchmarks/` -- the corroboration + highest-value find.**
`README.md:61-74`: **horizontal 2 cy per core/memory module, through both modules =>
~4 cy/column-hop** (bench 10, reason: 16 horizontal wires vs 32 vertical);
**vertical 2 cy/tile** (11); **shim->shim 4 cy** (9); **stream 2 cy/node** (12).
Independently explains bring-up `d_h≈4`/`d_v≈2`. Caveats: AIE1/VCK190 not
AIE2/Phoenix (structure transfers, magnitude confirmed by R3b); "horizontal" not
E/W-split -> confirms the E/W dead-end. **The shim->shim (bench 9) vs AIE-row (bench
10) distinction is exactly the Sec. 1 point: `d_h` must be calibrated on the shim
row.**

**aie-rt:** no per-hop constant, but reset topology fully specified (`XAie_SyncTimer`)
-- derive-from-toolchain handle for the tree shape + shim-row horizontal path.
**llvm-aie:** dead end (intra-core VLIW pipeline only). **AM025 regdb:** dead end
(bit layouts only). **aiesimulator:** possible fourth corroboration, low priority.

| Component | Toolchain disposition |
|-----------|----------------------|
| Per-hop magnitude / linearity | Corroborated (mlir-aie) + Phoenix-confirmed; physically explained |
| Reset topology + shim-row path | Toolchain-derived (aie-rt); shapes geometry + `d_h` calibration target |
| E/W anisotropy | Dead-end (all 4) -> disclosed assumption |
| Absolute intra-offset (gauge) | No source -> stays gauge, disclosed |
| `d_h^{ch15}` isolation | Structurally blended -> disclosed |
| Per-module horizontal split | Structure documented, no Phoenix split -> disclosed |

## 5. Decomposition (front-loads every structural showstopper)

Prove identifiability on paper and retire structural risk in pure-SW phases before
any Phoenix window.

| Phase | Kind | Purpose / risk retired | Gate |
|-------|------|------------------------|------|
| **0. Theory lock-down + toolchain sweep** | SW | Rank/identifiability proof per solver column **(including the decoupled `d_h`/`d_v` capture split -- rev4)**; the determinism basis (Sec. 1.5) written down; IF-1/2/3 + assumption decisions; magnitude estimate for tolerances (NOT a kill gate); sweep dispositions (Sec. 4); the `observe_r3b` per-routing coefficient models. | Rank proof per column + per capture; determinism basis + IF decisions recorded; sweep concluded. |
| **1. Rust ingestion wire + inert §9a plumbing** | SW | Build the `skew_constants.json -> model` path (Sec. 7); §9a fixpoint-ch15 multi-source test; wire the sweep sidecar consumption; the `observe_r3b` routing rewrite. All while `calibrated=false`. | Round-trip write->read->model populated, still inert; provenance-hash + non-null-when-calibrated build assertion; §9a tests green. |
| **2. `d_h` characterization (block-replicated / shim-row)** | HW | `d_h` = shim-row cost (IF-1) via block replication, **scoped `d_h`-only**; channel sweep (Seam 1, inter-channel only); linearity across accessible cols (Seam 3); b-vector jitter gate spaced across the drift timescale; uncontended edges only. | `d_h` residual green OR provenance downgraded; per-channel recorded (ch15 disclosed-assumed); b-vector range strictly `<` measured `d_h` over N>=20 *spaced* runs (RED blocks). |
| **3. `d_v` characterization + anisotropy + sign anchors** | HW | `d_v` via **free-flood** R3b (full spine leverage) *and* R1; falsify `d_vN != d_vS` via two-sided mid-column R1 spine (**sources must straddle the measured tiles vertically** -- a free-flood property); both `d_v` signs in one frame; R1/R3b `d_v` cross-check. | Free-flood `d_v` residual green; two-sided spine residual green OR `d_vN/d_vS` split; signs agree; R1 `d_v` ~= free-flood R3b `d_v` on the symmetric component. |
| **4. `dn_v`-correlated `Delta_wall` component** | HW | Bound the mode a green residual does NOT clear (row 51). Shares captures with Phase 3. | Symmetric component refuted by R3b's clean (free-flood) average; anti-symmetric residual disclosed (attribution rule). |
| **5. Held-out validation kernel** | SW build -> HW run | Break the causal-vs-HW tautology. **[rev4] Candidate: `matrix_multiplication_using_cascade`** (multi-column, output-byte-exact per `known-fidelity-gaps.md:71,79) -- BUT its trace-prep is documented as failing, so cross-domain *timing-edge* capturability is an open dependency. Phase 5 is on/near the critical path: either fix its trace-prep or build/rework a kernel that is BOTH `Delta_wall`-distinct AND within-domain byte-exact. | `causal_offset` predictions match HW deltas on the named kernel; tolerance strictly above that kernel's known `Delta_wall` residual. |
| **6. Measure, flip `calibrated`, regression guards** | ONE-WAY, HW-confirmed | The door. Set constants via Phase-1 wire; **re-sample the b-vector at the flip** (drift); 3 regression guards; honest provenance (measured / toolchain-derived / assumed). | Full regression + held-out causal-vs-HW + all prior gates green. |

**Dependencies:** `0 -> {1, 2}`; `2 -> 3` (`d_h`/`d_v` captures are different
instruments -- captures can co-locate in one Phoenix window; only the reconciliation
is a true dependency); Phases 3 & 4 share silicon captures; `{3,4} + 5 -> 6`. Phase 0
gates everything. Phase 5's kernel *build/fix* parallels 2-4; its *validation run*
co-locates with Phase 6.

**Phase 3/4 attribution rule (free-flood capture only).** The `d_v` **free-flood**
capture's `d_v` is `Delta_wall`-free and is the `(d_vN+d_vS)/2` **symmetric average**
*when the sources straddle the measured tiles vertically* (s1 below, s2 above -- a
free-flood property; the block-replicated `d_h` capture cannot straddle, which is
another reason `d_v` comes from the free-flood capture). R1's two-sided `d_v` splits
N/S but is `Delta_wall`-contaminated (`d_vN+eN`, `d_vS+eS`). R1 average - R3b average
= `(eN+eS)/2` = **symmetric** contamination -> refutable by R3b. R1 difference =
`(d_vN-d_vS)+(eN-eS)` -> **anti-symmetric** part confounds anisotropy with
anti-symmetric `Delta_wall`, and R3b is blind to it -> disclosed residual limit.
(Fable re-derived the algebra; correct.)

## 6. Pre-flip gate mapping (all 7 must hold before the flip)

| # | Gate | Retired in |
|---|------|-----------|
| 1 | Held-out validation kernel (named, byte-exact, `Delta_wall`-distinct, timing-edge-capturable) | Phase 5 |
| 2 | Enriched-geometry residuals green (else provenance = "assumed") | Phase 2 (`d_h`) + Phase 3 (`d_v`) |
| 3 | Joint sign anchors in one silicon frame | Phase 3 |
| 4 | `dn_v`-correlated `Delta_wall` quantified or dismissed | Phase 4 |
| 5 | Hard b-vector *jitter* gate, spaced + re-sampled at flip (RED blocks) | Phase 2 + Phase 6 |
| 6 | §9a(a) fixpoint channel-15 multi-source test | Phase 1 |
| 7 | §9a(b) wire sweep sidecar consumption | Phase 1 |

## 7. The Rust ingestion wire (+ shim-row `d_h`)

**Confirmed gap.** `schema.py` is Python-only; constants reach the model only via the
test-only `set_broadcast_timing_override` (`effects.rs:563`); compile-time constants
are zero with `calibrated:false` (`model_builder.rs:270-279`). No production reader.

**Recommended shape (Phase 1):** build-time codegen reading the committed
`skew_constants.json` -> archspec `BROADCAST_*` consts + `calibrated` flag (the flip
is a reviewable git diff; provenance travels). Retain the runtime override test-only.
**Extend the schema first** to carry: per-channel result, b-vector range, jitter
range, and an `assumptions` field (`horizontal_direction_isotropy: assumed`,
`absolute_intra_offset: gauge`, `ch15_hop_cost: assumed_via_blend`,
`per_module_horizontal_split: not_measured`, `d_h_path: shim_row`).

**Kill the stale/silent-zero modes** (the `.so`-staleness class): (1) commit a
provenance/content hash of the measured JSON, checked at build; (2) build assertion
`calibrated=true => all constants non-null and provenance present` -- **fail the
build**, never silently zero; (3) compile the provenance string in, runtime-queryable.

**`origin_D` tree note (demoted):** programming the block masks in the `origin_D`
flood is inert on the numbers under the single-scalar model (Sec. 1) -- retain for
generality, not as a flip gate. The load-bearing action is on the *measurement* side:
`d_h` on the shim row, `d_v` free-flood.

## 8. Ranked risk register

1. **Channel blend `d_h != d_h_ch15` (Seam 1)** -- HIGH/HIGH; ch15-isolation
   un-retireable, disclosed. Phase 2/6.
2. **Non-hop-linear / per-module transport (Seam 3)** -- MED-HIGH impact, HIGHEST
   late-discovery; power-limited by the small array. Phase 2.
3. **[rev4] `d_v` collapse under block replication** -- HIGH impact, was *silent*
   (rank-2 survives; a wrong-but-clean `d_v`); retired by the decoupled free-flood
   `d_v` capture (Phase 3). (Supersedes rev3's "one capture fits `{d_h,d_v}`.")
4. **`d_h` on the wrong horizontal path** (AIE-row vs reset's shim-row) -- MED-HIGH;
   retired by block replication / shim-row measurement (Phase 2, IF-1).
5. **`d_h` single-instrument, no absolute cross-check (Seam 2)** -- HIGH, structural;
   disclosure only.
6. **Tautology in causal-vs-HW gate** -- HIGH; Phase 5 named kernel + independently-
   verified `Delta_wall` (+ open timing-edge-capturability dependency).
7. **`Delta_wall`-linear-in-`dn_v` absorbed into `d_v` (row 51)** -- MED, HIGH
   late-discovery; symmetric refuted by R3b, anti-symmetric disclosed. Phase 3/4.
8. **Drift (thermal/DVFS/PLL)** -- MED, HIGH late-discovery; spaced runs + flip-time
   re-sample (Sec. 1.5).
9. **On-chip stream-arbitration nondeterminism on contended edges** -- MED; not covered
   by `is_async_cdc`. Kernel-design constraint: avoid contended edges.
10. **Rank-deficiency recurrence in a new enriched column** -- HIGH if it recurs; LOW
    late-discovery (Phase 0 proves rank per capture first).
11. **b-vector run-to-run jitter** -- predicted absent by the single-clock-domain basis
    (Sec. 1.5). Phase 2, RED blocks.
12. **Per-tile frequency / clock-tree phase skew** -- LOW / partly mooted by the single
    domain + shared-event driving. Document; don't chase.

## 9. What NOT to do

- **Do not resurrect the 5-param signed E/W + N/S solver** (rank-2 unidentifiable).
- **[rev4] Do not fit `d_v` from a block-replicated capture** -- the vertical spine
  collapses to common-mode; `d_v` comes from a free-flood capture and/or R1.
- **Do not run the terminal causal-vs-HW gate on the calibration kernel** -- a
  tautology; named held-out kernel only (Phase 5).
- **Do not list cross-axis additivity as a falsification target** -- under the shaped
  tree (horizontal-then-vertical routing) additivity is structurally true; no turn to
  penalize (dissolves audit Q1). `d_turn` stays deferred and moot for the reset.
- **Do not read a green residual on minimal geometry as shape validation** -- the
  inverse-laundering trap; accessible-array geometry earns it.
- **Do not treat the per-channel sweep as retiring ch15 uniformity** -- inter-channel
  only; ch15 stays disclosed-assumed.
- **Do not sell the shaped-tree `origin_D` mask-programming as a correctness fix** --
  inert in the single-scalar model.
- **Do not gate determinism solely on the b-vector range** -- lead with the
  single-clock-domain physical argument; the gate is a spaced/re-sampled backstop.

## 10. Third-round resolutions (rev4)

- **`d_v`-collapse defect:** fixed by decoupling the captures (Sec. 1 pt 2, Sec. 2,
  Phases 2/3). The one true blocker round 3 found; resolved.
- **`origin_D` mask inert:** confirmed; caveat added on single-node-per-tile modeling.
- **Additivity structurally true:** confirmed; stays struck as a target.
- **Determinism:** range-0 gate also covers readback ordering (stated); no path
  mis-classified beyond NoC egress + on-chip arbitration.
- **Phase 5:** named candidate `matrix_multiplication_using_cascade` + open
  timing-edge-capturability caveat recorded; Phase 5 is on/near critical path.
- **Verdict:** GO to plan + build Phase 0 + Phase 1 now (pure-SW, nothing
  irreversible); rev4 is the basis for the HW-phase plans, drawn after Phase 0's
  per-capture identifiability proofs land.

## 11. Review history

- **rev1 (2026-07-01):** Fable analysis-agent decomposition; claims verified vs code.
- **rev2 (2026-07-01/02):** 1st adversarial review -- mlir-aie benchmark corroboration,
  reset-tree topology, IF-3, Seam 1/2 corrections, Phase 3/4 attribution rule, Phase 5
  tolerance, ingestion-wire guards.
- **rev3 (2026-07-02):** 2nd adversarial review (subtractive) -- demoted the shaped-tree
  `origin_D` fix to inert (real requirement: shim-row `d_h` via block replication);
  struck the false two-channel corroboration; added the single-clock-domain determinism
  basis + drift/contention defenses; pinned the straddling precondition; removed
  additivity as a target; made Phase 5 depend on a concrete kernel.
- **rev4 (2026-07-02):** 3rd adversarial review -- found rev3's IF-1 "one block-replicated
  capture fits `{d_h,d_v}`" silently destroys `d_v` identifiability; **decoupled the
  captures** (`d_h` block/shim-row-only; `d_v` free-flood + R1); flagged the `observe_r3b`
  per-routing coefficient rewrite; re-pointed the attribution rule at the free-flood
  capture; recorded the Phase 5 candidate + timing-edge caveat; added the two Phase-0
  refinements. Verdict: ready to plan.
