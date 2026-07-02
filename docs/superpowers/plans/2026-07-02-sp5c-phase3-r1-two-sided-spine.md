# SP-5c Phase 3: the two-sided R1 spine (`d_vN` vs `d_vS`) -- execution plan

> **Status: DRAFT (2026-07-02).** Second HW-phase plan of SP-5c, drawn after the
> Phase-3 free-flood `d_v` half landed green (gate 5 fully green: jitter N=20
> back-to-back range-0 + drift N=20 spaced 90s over ~30min range-0, `d_v≈2`
> byte-identical). This plan covers the *other* half of Phase 3: falsify vertical
> anisotropy (`d_vN != d_vS`) and pin both `d_v` signs in one silicon frame
> (design `specs/2026-07-01-sp5c-skew-characterization-design.md` rev5, Sec.5 Phase
> 3, gates 2/3; risk register #7). Nothing here flips `calibrated`; the flip is
> Phase 6.
>
> **This plan opens by RETIRING a banked premise** (the "two mirror captures with
> different reset sources" recommendation in the session memory). Reading the code
> model + the reset topology shows that premise is unrealizable. The corrected
> mechanism and the real design fork are Sec. 1-3.
>
> **Fable-reviewed (2026-07-02): verdict BUILD-WITH-CHANGES, all changes folded in**
> (Sec. 9). The build is now gated on a cheap silicon buildability probe (Sec. 7 step
> 0) before any kernel work. A factual error about the extractor (it fits a single
> `d_v` + residual, not a `d_vN/d_vS` split) was caught and corrected throughout.

## 0. What "two-sided spine" must produce (the code model is authoritative)

`extract_r1_diff` (`tools/calibration/skew/r1_diff_extract.py`) and its falsification
test (`tools/test_skew_r1_diff_extract.py:88-132`) pin exactly what a two-sided spine
is. The model per module M:

```
module_delay(M) = dn_v(M) * slope + intra(kind_M),   slope = d_vN if dn_v >= 0 else d_vS
```

- The **source sits mid-column at `dn_v = 0`.** Tiles NORTH have `dn_v > 0` (per-hop
  `d_vN`); tiles SOUTH have `dn_v < 0` (per-hop `d_vS`).
- A genuine `d_vN != d_vS` becomes a **slope kink at the source** that a single-`d_v`
  fit cannot absorb -> `fit_residual` grows (`test_two_sided_spine_falsifies_vertical_anisotropy`,
  residual > 0.5 for `d_vN=3, d_vS=5`). Isotropic -> residual ~0
  (`test_two_sided_spine_isotropic_zero_residual`).
- **A one-sided spine cannot falsify anisotropy** (`test_one_sided_spine_cannot_falsify_anisotropy`,
  residual ~0 even when `d_vN != d_vS`): with all tiles North of the source, `d_vS` is
  never sampled. **This is precisely why the existing `sp5_skew_r1` kernel (source
  shim(0,0), all cores North) cannot do this job.**

So the physical requirement is a **single source with measured tiles on BOTH vertical
sides of it.** Hold that requirement against the reset topology next.

## 1. The correction: the real reset is north-only; you cannot get a two-sided spine from it

R1 as-built reads the **real `XAie_SyncTimer` reset**: sourced at the (0,0) corner,
channel `B` climbing each column from its shim (row 0) **north only** (rev5 finding
`findings/2026-07-02-sp5c-phase2-shim-row-topology.md`). Every compute tile's timer
origin reflects a north climb from its column's shim. **There is no south-going reset
leg among the compute tiles.** Consequences:

1. **The banked "two mirror captures (source core(0,2)->north, source core(0,5)->south)"
   recommendation is unrealizable.** It assumed we could relocate the reset source. We
   cannot -- `XAie_SyncTimer` fixes it at the corner (toolchain-derived, not our call) --
   and even if we could, the real reset climbs north from wherever it is re-sourced.
   **Retired.**
2. The existing single-sided R1 kernel measures **`d_vN` only** (north climb). Its
   gate-6 pass proved *runnability + range-0*, never a `d_v` value or an anisotropy test.
3. The design's Phase-3 phrasing ("sources must straddle the measured tiles vertically
   -- a free-flood property") conflates two distinct geometries: a *straddling pair*
   (s1 below + s2 above) gives the R3b **symmetric average** `(d_vN+d_vS)/2`; a *single
   mid-column source* gives the R1 **N/S split**. Same intent (see both directions),
   different apparatus. This plan disambiguates.

## 2. The realizable mechanism: a synthetic mid-column reset-trigger, read via trace

The only way R1's readout (per-tile timer-origin offset, via decoded-trace `soc`) can
reflect a mid-column bidirectional source is to **make that source reset the measured
tiles' timers.** Concretely:

- Emit a synthetic broadcast (`Event_Generate` on a chosen channel) from a **mid-column
  core** (the source at `dn_v=0`).
- **Reconfigure each measured tile's `Timer_Control` reset trigger** (register `0x34000`,
  `Reset_Event` field) to fire on that channel's broadcast arrival, replacing the corner
  reset. Hand-authored `aiex.npu.write32`, the same class of register programming R3b
  already does for its floods + counter config (`5.3` of the apparatus design).
- Each tile's timer then zeroes when the mid-column broadcast arrives, so its origin
  offset = `D(mid-source -> tile)`: north tiles get `dn_v * d_vN`, south tiles
  `|dn_v| * d_vS`. R1's trace readout + `observe_r1` + `extract_r1_diff` then fit a
  **single `d_v`** and a genuine `d_vN != d_vS` surfaces as a **slope kink at the
  source that grows `fit_residual`** -- see the instrument caveat next.

**Instrument caveat (do not overclaim): `extract_r1_diff` does NOT recover a per-direction
split.** Its design matrix is **2 columns** `[dn_v_diff, core_indicator_diff]`
(`r1_diff_extract.py:35-38`), fitting `{d_v, intra_contrast}` -- one `d_v`, not
`{d_vN, d_vS}`. Anisotropy is detected **only** as a nonzero residual under the single-slope
model (`test_two_sided_spine_falsifies_vertical_anisotropy`), never as two fitted numbers.
Recovering an actual split would need a 3-column matrix `{max(dn_v,0), min(dn_v,0),
contrast}` that **nobody has built** -- and Sec. 3 (fork) + Sec. 9 Q5 conclude the model
can't use a split anyway, so we do **not** build one (YAGNI). Gate language must say
"residual-based isotropy falsification," never "clean split."

**The emulator already models this for free.** `broadcast_origin_d`
(`effects.rs:489-503`) is a symmetric Dijkstra from any `source_row`: it floods north
(`r+1`, cost `d_v`) AND south (`r-1`, cost `d_v`). So a mid-column source is a
one-parameter change (`source_row`), and the emu is **isotropic by construction** (one
`d_v` both ways, `:491-492`) -- which is exactly the assumption this capture exists to
test on silicon. The `dwall` side is a zero-constants emu run (origins all 0,
source-independent), so **only the HW side carries the synthetic source**; `dwall` is
just the normal zero-constants run of the same kernel (write32s inert at
`calibrated=false`, same pattern as the existing R1 gate's dwall).

**Why R3b cannot substitute.** R3b's two-source interval provably cannot separate
within-axis direction at any placement (`test_skew_r3b_identifiability.py`; the SP-5b
identifiability theorem, `a_hE-a_hW` const / corner sources rank-2). A single-source
absolute-origin readout is the *only* instrument that sees the N/S split -- and that is
R1's readout, not R3b's counter interval. This is why the design reallocated anisotropy
to R1 (memory: "VERTICAL anisotropy reallocated to a two-sided mid-column R1 spine").

## 3. The fork: cheap cross-check (A) vs true residual falsification (B)

Two ways to get at `d_vN` vs `d_vS`, differing in rigor and cost.

### Option A -- combine already-green instruments (no new kernel)
Run the **existing single-sided R1** (real reset, north) for a `d_v` *value*
(`d_vN + eN`), and difference it against the **existing free-flood R3b straddling**
symmetric average `(d_vN+d_vS)/2` (already measured `d_v≈2`, gate-5 green):
`d_vS(inferred) = 2 * symmavg - d_vN`.

- **Pro:** uses only built + gated apparatus; one added R1 *value* extraction (the gate
  only proved runnability). Immediate gross-anisotropy signal.
- **Con:** a two-point consistency check, **not** a residual-based falsification.
  Per the design attribution rule (rev5 Sec.5): R1 is `Delta_wall`-contaminated
  (`d_vN+eN`), R3b symmavg is clean, so `2*symmavg - R1 = d_vS - eN` -- the
  contamination does **not** cleanly cancel. It bounds gross anisotropy; it does not
  earn gate 2's "residual green" for the vertical axis.

### Option B -- true two-sided R1 spine (new kernel, Sec. 2 mechanism) -- RECOMMENDED for the gate
Build the synthetic-mid-column-reset R1 kernel; `extract_r1_diff` yields a
residual-based isotropy falsification (kink-at-source), the design's actual gate-2/3
instrument. Cross-check its symmetric component against the free-flood R3b symmavg
(refutes symmetric `Delta_wall` contamination per the attribution rule); its
anti-symmetric part `(d_vN-d_vS)+(eN-eS)` is the disclosed residual limit (R3b is blind
to it).

- **Pro:** the rigorous, design-intended falsification; earns gate 2 (vertical) + gate 3.
- **Con:** a genuinely **new HW mechanism** (synthetic reset-trigger R1 -- the existing
  R1 uses the real reset). Carries **Phase-2-class buildability risk**: a paper mechanism
  silicon may refute (Phase 2 just taught us this the expensive-cheap way). Emu side is
  nearly free; HW side is unproven.

**Recommendation:** do **A first** (cheap, already-built -- an immediate reality check on
whether anisotropy is even gross), then build **B** for the gate. A is a de-risking probe,
not a substitute; the flip needs B's residual falsification (or an honest "assumed
isotropic, disclosed" if B proves unbuildable). Sequencing in Sec. 7.

## 4. Geometry under the Phoenix 4-core-row constraint (the binding limit)

Phoenix has only **4 core rows (2-5)** in the virtualized frame (memtile row 1, shim row
0). Core points per side for the two candidate mid-column sources (the source tile itself
counts as a `dn_v=0` observation, so add it to both sides -- see below):

| Source row | North core pts (`d_vN`) | South pts (`d_vS`) |
|---|---|---|
| core 3 | rows 4,5 (+ source at 0) | row 2 (core), row 1 (memtile), row 0 (shim) |
| core 4 | row 5 (+ source at 0) | rows 3,2 (core), row 1 (memtile), row 0 (shim) |

**Recommended: source at core row 3.** `extract_r1_diff` fits `{d_v, intra_contrast}` (2
params, see Sec. 2 caveat); kind-mixing across sides is absorbed by the contrast indicator.
North/south leverage is **better than a naive count suggests** (Fable correction):

- **The source tile is itself a `dn_v = 0` observation** (`origin_D[source] = 0`,
  `effects.rs:467`; Sec. 6a puts a trace unit on it). So the **north** core-group line is
  `{source(0), row4(+1), row5(+2)}` = 3 collinear = **2 hops = a genuine north-uniformity
  check**, not a bare 2-point line.
- **`origin_d_table` emits BOTH a core and a mem observation per compute tile**
  (`effects.rs:527-528`). So each of rows 2,4,5 yields two obs; the **south** side carries a
  core-group line `{source 0, row2 -1, shim -3}` and a mem-group line `{source 0, row2 -1,
  memtile -2}`.

Disclosed dependence (Sec. 9 Q4 / D2): the south's extra reach leans on the extractor's
intra-group lumping (`shim` grouped with `core`, `memtile` with `mem`;
`r1_diff_extract.py:14-15`). A `shim != core` intra mismatch on the `dn_v=-3` point biases
the apparent south slope -- disclose it; do not treat the south line as free of that
assumption. Pairs (`geometry.json`): north rows 4,5 vs source; south row 2 / memtile / shim
vs source; a `crosses-source` pair (row 4 vs row 2) to lock the kink; a same-tile
core<->mem intra pair for the contrast. `dn_v` signed relative to the mid-column source;
frame is virtualized (relative col 0) per apparatus design Sec.3 -- hop distances are
relocation-invariant. Mirror the existing `sp5_skew_r1` geometry schema.

## 5. Sign anchors in one frame (gate 3)

Gate 3 wants both `d_v` signs pinned in a single silicon capture. The two-sided spine is
naturally that frame: north tiles resolve `+d_vN`, south tiles `-d_vS`, and the
`crosses-source` pair fixes their relative sign against the same origin. Reconcile against
`origin_D` via the reflected-sign unit test already mandated (`r1_extract` sign-pin,
apparatus design Sec.4.2) -- the trace carries `max_delay - module_delay`, inverted vs
`origin_D`. Add the sign-anchor assertion to the extractor test if not already covered.

## 6. Code deltas (all `calibrated=false`, nothing irreversible)

The extractor and observe layers **already exist and are tested** -- the two-sided model
is `extract_r1_diff` + the four `test_skew_r1_diff_extract.py` cases. So the deltas are
mostly the kernel + geometry + gate:

### 6a. New kernel `sp5_skew_r1_2sided` (mlir-aie, branch `xdna-emu-cycle-budget`)
Copy `sp5_skew_r1/sp5_skew_r1.py` (the Q=0 objectfifo spine -- keep the pure-lock/DMA,
no-compute discipline that avoids the MEMORY_STALL gap). Add: (i) a synthetic
`Event_Generate` broadcast from the mid-column source (core row 3); (ii) `write32`
reconfiguration of the measured tiles' `Timer_Control.Reset_Event` to that channel;
(iii) trace units on the source + both-side tiles. **mlir-aie is a separate repo -- named
paths only, never `git add -A`.** Emu smoke-run must complete clean (`halt_reason=completed`,
LOCK_STALL on all spine cores, decodable trace); counters/origins read consistent with
zero constants on emu (the isotropic-by-construction dwall).

**Two config constraints (Fable Q3 -- load-bearing):**
- **Channel: reuse `ch15`** (the measured channel family -- matches both the real reset and
  the free-flood `d_v` capture). Do not introduce a fresh channel and a fresh unverified
  hop cost.
- **Switch masks: FREE-FLOOD / unblocked, never the reset's block config.** The real reset
  *blocks* E/W to shape a north climb (`_XAie_SetupBroadcastConfig`, finding:52-57). If this
  kernel inherits any reset-style block masks, the mid-column source is **prevented from
  flooding south** and you reproduce a confinement artifact indistinguishable from a
  topology failure (the Phase-2 trap wearing a different hat). Leave all switch masks at
  reset-default-open; the free-flood config is what silicon already proved reaches every row.

**Emu smoke is decode/plumbing ONLY -- it is NOT buildability evidence.** The emu asserts the
bidirectional flood by construction (`effects.rs:491-492`), the exact circularity that let
the shim-E/W edge survive three reviews until silicon refuted it. Buildability is settled by
the HW probe in Sec. 7 step 0, not by the smoke-run.

### 6b. `geometry.json` + `trace_config.json` (mlir-aie, same dir)
Signed `dn_v` relative to the mid-column source; pairs per Sec. 4; `trace_config.json`
slot->event mapping for the new trace units (drives `_slot_names_from_trace_config` in
`r1_emu_recover.py`). No `routing` key needed (R1 has no block routing).

### 6c. Extractor: confirm, don't rebuild
`extract_r1_diff` already handles signed `dn_v` two-sided data (its tests prove it). Add
only: the **sign-anchor** assertion (Sec. 5) if missing, and a frozen-fixture test on the
*actual* Phoenix geometry (source row 3, the Sec. 4 pairs) so the gate's rank/kink checks
run on the real design matrix, not just the synthetic `d_vN=3,d_vS=5` fixture.

### 6d. Gate `r1_2sided_gate.sh` + tally (xdna-emu `build/experiments/sp5-skew/`, gitignored)
Fork `r1_gate.sh` (it already does the R1 HW-run -> decode -> normalize -> dwall ->
`observe_r1` -> `extract_r1_diff` pipeline). Checks: rc-0, zero TDR/IOMMU delta, decodable
trace, **range-0 across N runs** (jitter -- the b-vector determinism gate for R1),
non-degeneracy (>=2 distinct `dn_v` per side), and report `fit_residual`. **Do NOT assert a
residual value** -- a nonzero residual is a catch-all (Fable D2): it can be driven by (a)
true `d_vN != d_vS`; (b) anti-symmetric `Delta_wall` `(eN - eS)` (R3b blind to it); (c)
intra-group-lumping error (`shim != core` or `memtile != mem`, `r1_diff_extract.py:14-15`);
or (d) nonlinearity. So **GREEN is informative (nothing misfits -> isotropic-enough,
proceed); RED is ambiguous** -> investigate the four confounds, then disclose-and-defer (Sec.
8 rule), never "RED = measured anisotropy." Serial only, no `xrt-smi` inside. Add a
spaced-drift variant (mirror `r3b_pc_drift_gate.sh`) for the drift half.

## 7. HW capture protocol + sequencing (Phoenix)

Preflight: `xrt-smi validate` exit 0 (alone, before any HW loop), dmesg TDR/IOMMU
baseline, no concurrent HW suite.

0. **BUILDABILITY PROBE -- hard gate before building B (Fable Q1, the plan's biggest gap).**
   Do **not** build the R1 trace kernel to discover whether the bidirectional flood works.
   Instead **relocate the already-silicon-validated R3b-PC single source to mid-column core
   row 3** (a one-register edit to the validated `sp5_skew_r3b_pc` kernel -- no
   trace-prepare/inject/decode), arm `Performance_Counter0` on **one north tile (row 4)** and
   **one south tile (row 2)**, read counters back, **N=3**. **Pass = the south tile registers
   a deterministic non-zero count** (the flood reaches south). If south stays 0, the
   mid-column flood is north-biased -> **B is unbuildable**; fall back to Option A + "assumed
   isotropic, disclosed." Cost: one bring-up capture, minutes of Phoenix time -- vs a full
   trace-kernel build to learn the same thing. This gates steps 2-4.
1. **Option A cheap probe (already-built apparatus).** Extract a `d_v` *value* from the
   existing single-sided R1 (real reset, north) and difference against the free-flood R3b
   symmavg (`d_v≈2`): `d_vS ~ 2*symmavg - d_vN`. Read strictly as a **gross-anisotropy floor**
   (`d_vS - eN`, not clean -- Sec. 3): catches only large anisotropy (e.g. `d_vS ~ 2*d_vN`),
   never positive evidence *for* isotropy at flip resolution. Zero-cost sanity signal.
2. **Build + emu-validate B** (Sec. 6, subagent-driven) -- only if step 0 passed.
3. **Option B capture:** `r1_2sided_gate.sh 20` (jitter) + the spaced-drift variant
   (drift half of gate 5, mirroring the free-flood drift gate already green). RED (range
   >= `d_v`) blocks.
4. **Cross-checks:** B's symmetric component vs free-flood R3b symmavg (attribution rule);
   sign anchors consistent in the one frame (gate 3).

Co-locates in the same Phoenix window as the free-flood `d_v` captures (design Sec. 5
dependency note: `d_h`/`d_v`/spine captures share a window; only reconciliation is a true
dependency).

## 8. Phase-3 spine exit gates + the anisotropy decision rule (design Sec.6)

- **Gate 2 (vertical `d_v` leg):** two-sided-spine `fit_residual` green under the
  **single-`d_v`** fit -> vertical uniformity confirmed. (There is **no** "clean split"
  branch -- the instrument does not fit `{d_vN, d_vS}`, Sec. 2 caveat.) RED -> the Sec. 8
  decision rule below, not an automatic block.
- **Gate 3:** both `d_v` signs pinned in the one silicon frame; signs agree with `origin_D`
  convention (reflected-sign test).
- **Gate 5 (R1 vertical, partial):** b-vector range strictly `<` measured `d_v` over N>=20
  spaced runs (RED blocks); re-sampled at the Phase-6 flip.
- **R1/R3b cross-check:** R1 symmetric component ~= free-flood R3b `d_v` on the symmetric
  average; anti-symmetric residual disclosed (attribution rule -- R3b blind to it).

**The anisotropy decision rule (Fable Q5 -- fix this BEFORE capture).** The emu model is
single-`d_v` isotropic by construction (`effects.rs:491-492`) and **cannot represent
`d_vN != d_vS` without a new parameter.** The single-`d_v` value IS the symmetric average
`(d_vN+d_vS)/2` -- the correct first-order value *regardless* of anisotropy. Per design Sec.0
("no value-of-information kill gate"), a split finding is a **disclosure event, not a flip
blocker**, until the held-out kernel says otherwise:

1. `|d_vN - d_vS|` within b-vector jitter / measurement resolution -> **declare isotropic,
   single-`d_v` honest, proceed.**
2. Resolvable but small vs the Phase-5 held-out kernel's `Delta_wall` tolerance -> **adopt
   the R3b free-flood symmavg as the single `d_v`** (already `d_v≈2`, gate-5 green),
   **disclose the anisotropy as a known assumption, defer the model parameter (YAGNI),
   proceed to the flip.**
3. Large enough that Phase-5 gate-1 (held-out causal-vs-HW) fails under single-`d_v` ->
   **only then** does it block and force a model parameter.

So the spine's real deliverable is "confirm/bound isotropy so the single-`d_v` model is
honest," not "produce a split the model can't use."

## 9. Fable overview -- resolutions (2026-07-02, verdict BUILD-WITH-CHANGES)

A curated Fable agent reviewed this plan against the design, the Phase-2 finding, the
extractor + its tests, and the emu Dijkstra model. Verdict: **BUILD-WITH-CHANGES** -- the
mechanism decomposes into two silicon-proven primitives (core-sourced flood, Phase-2 `s2`;
timer-reset-off-broadcast, `XAie_SyncTimer`'s own mechanism), leaving one genuinely-open
question (bidirectional flood from a *mid* source -- neither Phase-2 source was mid-column).
All Q1-Q5 folded into the plan above:

1. **Q1 (mechanism sound?) -> YES, gated on a cheap probe.** The buildability probe is now
   Sec. 7 step 0 (relocate validated R3b-PC to mid-column, read a north + a south counter,
   N=3). This was the plan's biggest gap -- it asked Q1 but never answered it.
2. **Q2 (Option A) -> run it as a gross floor, not misleading if read as such** (Sec. 3 /
   Sec. 7 step 1). No cleaner cheap split-probe exists (R3b can't split within-axis).
3. **Q3 (channel) -> reuse ch15 AND free-flood/unblocked masks** -- the sharper risk is the
   block masks, not the channel number (Sec. 6a constraints).
4. **Q4 (leverage) -> better than I claimed; the source is a `dn_v=0` obs** so north is
   3-collinear (uniformity tested); south leverage leans on intra-lumping, disclosed (Sec. 4).
5. **Q5 (does the split matter?) -> NO; deliverable is confirm/bound isotropy** -- anisotropy
   = disclose + defer, decision rule now in Sec. 8.

**Defects Fable found and this rev fixed:**
- **D1 (factual, was in this plan twice):** `extract_r1_diff` fits a **single `d_v`** + a
  residual, **not** a per-direction split. "Clean split" dropped from Gate 2; Sec. 2/4/8
  corrected; no split extractor will be built (Sec. 2 caveat, Q5).
- **D2 (identifiability):** a RED residual is a 4-way catch-all (true anisotropy /
  anti-symmetric `Delta_wall` / intra-lumping / nonlinearity) -> GREEN confirms, RED is
  ambiguous (Sec. 6d, Sec. 7 step 5, Sec. 8 rule).
- **D3 (Phase-2-class premise):** the bidirectional-mid-source flood is emu-asserted, not
  silicon-proven; mitigation hardened from "cheapest-probe-first" to the concrete Sec. 7
  step-0 hard gate, and the emu smoke-run is explicitly de-rated to plumbing-only (Sec. 6a).

Full review archived in the session transcript (Fable agent, 2026-07-02).

## 10. Subagent decomposition + model tiering

| Task | Model | Scope |
|---|---|---|
| A. Kernel `sp5_skew_r1_2sided` + geometry/trace_config | Sonnet | copy R1 spine, add mid-column `Event_Generate` + `Timer_Control` reset-reconfig write32, trace units, signed-`dn_v` geometry, build both compilers, emu smoke-run. mlir-aie: named paths only. |
| B. Extractor confirm + real-geometry fixture + sign-anchor test | Sonnet | frozen-fixture test on the source-row-3 design matrix; sign-pin assertion. `pytest tools/test_skew_*.py` green. |
| C. Gate + tally (jitter + spaced-drift) | Haiku | fork `r1_gate.sh` + `r1_2sided_gate.sh` + drift variant + tally. |
| Whole-delta review | Opus (me) | review A+B+C against this plan + the code model before any HW. |
| Mechanism/plan pressure-test | Fable agent | **DONE (2026-07-02)** -- verdict BUILD-WITH-CHANGES, folded into Sec. 9. |
| Buildability probe (Sec. 7 step 0) | me | relocate validated R3b-PC to mid-column, N=3, gate the build. |
| HW capture | me | Sec. 7 protocol on Phoenix, only if step 0 passes. |

## 11. Risks carried

- **Bidirectional-mid-source flood unbuildable on silicon (Phase-2-class)** -- HIGH; the one
  unproven composite. Mitigation: the Sec. 7 step-0 hard gate (cheap R3b-PC probe) settles it
  before any kernel build; Option A fallback; honest "assumed isotropic, disclosed" if it
  fails.
- **RED residual is a 4-way catch-all (D2)** -- structural; GREEN confirms, RED -> investigate
  + disclose-and-defer, never "measured anisotropy."
- **Inheriting reset block-masks confines the south flood** -- reproduces a false topology
  failure; mitigated by the free-flood/unblocked-masks constraint (Sec. 6a).
- **Thin north leverage** -- 4-row ceiling; north is uniformity-*tested* (3-collinear via the
  `dn_v=0` source) but only 2 hops -- disclosed.
- **Model cannot represent anisotropy** -- if `d_vN != d_vS`, single-`d_v` is
  wrong-but-inexpressible; decision rule (Sec. 8) set before capture -> disclose + defer.
