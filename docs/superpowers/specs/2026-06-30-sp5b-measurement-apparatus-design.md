# SP-5b: Skew Measurement Apparatus -- Design (rev3)

> **rev3 (2026-07-01) folds in the SP-5b soundness audit**
> (`docs/superpowers/findings/2026-07-01-sp5b-soundness-audit.md`). Verdict: build
> R3b, **do not flip `calibrated`**. The 4-knob model bakes in four structural
> assumptions (per-hop uniformity, direction isotropy, cross-axis additivity,
> per-channel hop-cost uniformity); the rev2 geometry falsifies only the first.
> The build-level corrections (enriched R3b geometry, gate-gated immunity, signed
> solver columns, SP-5c pre-flip gates) live in the **kernel bring-up spec rev3**
> (`2026-06-30-sp5b-kernel-hw-bringup-design.md` Sec.13); this doc corrects the
> assumption/role text those build on, tagged **[rev3]** inline.
>
> **ERRATA (post-implementation, 2026-07-01):** the rev3 claim that enriched
> geometry falsifies direction *isotropy* is partly unrealizable -- R3b's two-flood
> interval cannot separate within-axis directions (`d_hE` vs `d_hW`, `d_vN` vs
> `d_vS`) at any two-source placement. Cross-axis `d_h` vs `d_v` is fine. OPEN
> decision: `docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md`.

Issue #140, timer-sync faithful-broadcast arc. Sub-project SP-5b of the SP-5
decomposition (5a calibration enablement [done], 5b measurement apparatus
[this], 5c Phoenix campaign + go-live).

**rev2 folds in two adversarial Opus reviews (epistemics + feasibility).** The
material changes from rev1: the instrument roles are flipped (R3b is primary for
the hop constants, R1 narrows to within-column `d_v`/intra) because HW data shows
R1 cannot measure `d_h` from a trace; the "no new emulator features" claim is
withdrawn (SP-5b adds a runtime-override seam for the timing constants); the R1
emu loop is re-labeled a plumbing/regression check (not a physics test); R1's
cross-domain `Delta_wall` faithfulness is stated as assumed (not SP-4a-proven);
the falsification geometry requires >=3 collinear tiles per axis; and several
mechanical fixes (sign convention, pointer drift, flood ordering).

**This is alpha hardware-characterization apparatus, not production software.**
Engineering is pragmatic; the honesty hooks (falsification, `ModelDerived`
provenance, the no-laundering gate) stay, because those are what keep a wrong
measurement from ever reaching a causal claim.

Prior art: `2026-06-28-timer-sync-faithful-broadcast-arc.md` (the arc),
`docs/trace/cross-domain-skew-limit.md` (epistemic boundary; routes in Sec.8),
`2026-06-30-sp4b-skew-export-design.md` (engine gate this feeds),
`2026-06-30-sp5a-calibration-enablement-design.md` (plumbing this precedes),
`docs/superpowers/findings/2026-06-29-coldstart-headstart-trace-baseline.md`
(SP-4a), `build/experiments/sp3-spike-trace/SP4A-HW-TARGETS.md` (the HW jitter
data driving the flip).

---

## 1. Scope and non-goals

SP-5b builds and validates the two skew-measurement instruments so SP-5c can
point them at Phoenix silicon, read `d_h / d_v / intra_core / intra_mem`, and flip
`BroadcastTiming.calibrated`. **SP-5b produces apparatus, not numbers.**

The 4-knob model is exactly `BROADCAST_PER_HOP_HORIZONTAL` (`d_h`),
`BROADCAST_PER_HOP_VERTICAL` (`d_v`), `BROADCAST_INTRA_TILE_CORE_OFFSET`,
`BROADCAST_INTRA_TILE_MEM_OFFSET` (`crates/xdna-archspec/src/model_builder.rs:270-279`).

| In SP-5b | Out (deferred to SP-5c) |
|---|---|
| Build all three kernels (R1 + both R3b mechanisms) | The numeric Phoenix skew measurement |
| A runtime-override seam for the 4 constants + `calibrated` | Ingesting *measured* constants; production ingestion path |
| R1 emu inject-and-recover **plumbing/regression** check | Flipping `calibrated` + updating the 3 regression guards |
| R3b kernel correctness + rank-2 solve on synthetic data | R3b numeric silicon validation |
| Cheap HW *runnability/repro* gate on all three kernels | Causal-vs-HW validation; P1 round-trip gate |
| Fit-residual (structure-falsification) reporting | Resolving the structure questions on silicon |

**Non-goals.** No numeric skew measurement. No `calibrated` flip. No
engine/inference changes (SP-4b/SP-5a built the consuming side, dormant).
**Withdrawn from rev1:** the "no emulator feature work" non-goal -- SP-5b *does*
modify the emulator (the runtime-override seam, Sec.6). This is explicitly fine:
the emulator is characterization tooling, not a production program, and the
default (all-zero constants, `calibrated=false`) keeps behavior byte-identical,
so the three existing neutrality guards stay green.

**Why the arc has not solved this, and the safety property.** The skew is
undeterminable *from a trace alone* (skew-limit Sec.6, three walls). Each
instrument beats one wall with one load-bearing assumption (Sec.2). The safety
property is narrower than rev1 claimed: SP-5b **cannot auto-launder a wrong
number into a causal claim** -- everything downstream is gated behind a `calibrated`
flip that happens only in SP-5c under human causal-vs-HW validation, and the
emitted fact is provenance-tagged `ModelDerived`, never "measured" (SP-4b). It is
*not* "structurally impossible to be wrong": a numerically-wrong measurement that
passes the fit-residual check could still be flipped live by a human who trusts a
green structural gate. That is why the falsification geometry (Sec.4.5/5.4) must
actually be able to fire.

---

## 2. The two instruments, their roles, and their load-bearing assumptions

Roles are set by what each instrument can actually measure on silicon (rev2 flip):

- **R3b (two-source perf-counter interval) -- PRIMARY for `d_h` and `d_v`.** It
  reads a jitter-free *local* single-clock interval per tile (no trace, no
  emulator dependence), so it is immune to the cross-column trace jitter that
  defeats R1 (below) and it is the only instrument that can measure `d_h`. Beats
  Wall 2 (a single co-observed flood cancels: `soc_A(E)=soc_B(E)`) with a *second*
  flood `s2` from a different corner. **Load-bearing assumption: `s2 != s1`
  genuinely, the two-flood + perf-counter config is valid on silicon, and `s1`
  arrives before `s2` at every measured tile** (all untested -> the HW gate).

- **R1 (trace tile-distance sweep) -- `d_v` + intra, and the reproduction-path
  carrier.** HW data (`SP4A-HW-TARGETS.md:50-55`, 20 runs of R1's skeleton) shows
  only *within-column vertical* cross-domain pairs are range-0; *every
  cross-column pair carries ~30-cycle irreducible stream-switch-crossing jitter,
  present on the first event* -- and `d_h` is single-digit cycles, so R1 cannot
  isolate `d_h` from a trace (and we forbid statistical averaging). R1 therefore
  runs **within-column** (Sec.4), cleanly measuring `d_v` and the intra offsets,
  and additionally lands the never-shipped SP-3 gate-carrier. It beats Wall 1
  (underdetermination) with the emulator's `Delta_wall` as an independent
  equation. **Load-bearing assumption: the emulator's *cross-domain* `Delta_wall`
  is faithful in the chosen steady-state window.** This is *assumed*, NOT
  established -- SP-4a proved only *within-domain* rate faithfulness and explicitly
  retired its one cross-domain check (`-52` EMU vs `+2` HW, a fill-cadence
  residual: known-fidelity-gaps row 51, OPEN, head-start-invariant). **[rev3 --
  corrected mechanism]** rev2 said the danger is "a persistent constant phase
  offset indistinguishable from skew." The differencing solver
  (`r1_diff_extract`, no intercept column) makes the actual failure modes sharper
  and different: a *globally-constant* Delta_wall error **cancels** in the
  within-pair differencing (harmless); a *kind-dependent* constant lands wholly in
  `intra_contrast` (which has no R3b cross-check); and the mode that **silently
  corrupts `d_v`** is one **correlated with `dn_v`** -- `gamma*(dn_v_b - dn_v_a)`
  is collinear with the `d_v` design column, absorbed as `d_v += gamma` at **zero
  residual**, unbounded. Row 51 is a pipeline-fill effect whose per-module error
  plausibly scales with pipeline depth ~ `dn_v` -- exactly this shape -- and its
  `dn_v`-correlated magnitude is **unmeasured**. So R1's silicon `d_v` is
  contingent on row 51 closing (or its `dn_v`-correlated component empirically
  vanishing in the window), a green `fit_residual` does **not** clear it, and it is
  cross-checked by R3b only for `d_v` (never for `intra_contrast`). See kernel spec
  rev3 Sec.4.2.

**Structure vs values.** R1 and R3b share the *same linear per-hop functional
form*. So their cross-agreement validates the *values* under a shared linearity
assumption; it does **not** validate the *structure*. **[rev3 -- the shape has
four assumptions, not one.]** rev2 framed structure validation as the single
question "is per-hop delay uniform?" The audit found the 4-knob linear model
`origin = dn_v*d_v + dn_h*d_h + intra(kind)` bakes in **four** continuous
assumptions: (1) **per-hop uniformity**, (2) **direction isotropy** -- one `d_v`
for North and South, one `d_h` for East and West (`effects.rs:491-494`), (3)
**cross-axis additivity** -- no turn/interaction term (`:497`), and (4)
**per-channel hop-cost uniformity** -- hop cost is channel-agnostic (`:485`). The
`>=3-collinear-tiles-per-axis` redundancy (Sec.4.5/5.4) falsifies **only (1)**;
(2)/(3)/(4) are unfalsifiable by that geometry (a one-sided collinear axis fits a
zero-residual line under any anisotropy; an interaction term contributes zero on
axis-collinear tiles; a channel blend is linear in displacement). Falsifying all
four requires the **enriched geometry** -- two-sided per axis, off-axis diagonal,
channel-uniformity control -- specified in kernel spec rev3 Sec.5.1. Without it,
both instruments fit a shape they cannot reject and mislabel an untested-3/4
structure as "validated."

---

## 3. Column frames (load-bearing for all kernel geometry)

Phoenix NPU1 has two topology representations with different column numbering;
conflating them is the source of recurring "column weirdness."

- **Physical Phoenix**: 5 columns; absolute column 0 has **no shim tile**
  (`docs/device-model-audit.md:58`, aiesim `NPU1.json`).
- **mlir-aie (virtualized frame)**: the only NPU1 model is
  `VirtualizedNPU1TargetModel` (`include/aie/Dialect/AIE/IR/AIETargetModel.h:736`,
  comment "A sub-portion of the Phoenix NPU"), variants `1Col..4Col`; no
  `NPU1_5Col`. `getTileType` returns `ShimNOCTile` for row 0 of *every* modeled
  column (line 752-758) -- the shim-less column does not exist in this frame.

Firmware relocates the virtualized partition onto physical cols 1..N at load
(hence the emulator's absolute-frame "partition starting at column 1").

**Design rule:** specify all kernel geometry in the **virtualized frame**
(relative cols 0..N-1, all shim-bearing, N<=4; rows 0 shim / 1 memtile / 2-5
core). Hop *distances* are relocation-invariant. Never reason about a shim at
physical column 0.

(Out of scope: physical col-0 tiles are not binned -- per Maya they were
accessible and rewritten inaccessible, plausibly re-enableable. Separate later
investigation.)

---

## 4. Route 1 -- within-column trace sweep (`d_v` + intra)

### 4.1 The model being fit

For a module X, the broadcast reset arrives at
`origin_X = Dn_v(X)*d_v + intra(kind_X)` (R1 is within-column, so `Dn_h=0` and
`d_h` drops out; `d_h` is R3b's job). `intra(kind)` is one of the two intra
offsets (`core`, `mem`), with `memtile`/`shim` as reference or additional
indicators. The extraction is a **design matrix** (`origin` = linear combination
of `[Dn_v, kind-indicators]`), so the parameterization is not hardcoded; the fit
residual (Sec.4.5) reports whether the shape holds. SP-5b validates the apparatus
recovers whatever is injected; SP-5c settles the true silicon structure.

### 4.2 Measured -> residual -> solve

For a within-column cross-domain event pair (x in domain A, y in domain B) firing
deterministically together, the trace gives `soc(x)-soc(y) = Delta_wall + skew(A,B)`.
R1 isolates `skew(A,B) = measured_offset - Delta_wall`, with `Delta_wall` from the
emulator run at zero constants. Many pairs down the column -> an over-determined
linear system in `{d_v, intra_core, intra_mem}` -> least-squares + fit-residual norm.

**Sign convention (rev2, load-bearing).** The trace-embedded offset is the
*reflected* quantity `origin_offset = max_delay - module_delay`
(`src/device/state/effects.rs:628-629`), whereas the sidecar/engine `origin_D`
is `module_delay` directly (SP-4b Sec.4b). R1 reads the *decoded trace*, so its
recovered skew is in the reflected convention -- **sign-inverted relative to
`origin_D`**. The `max_delay` reference cancels in cross-domain differences, but
the sign does not. The R1 extractor MUST reconcile this and pin the sign with a
unit test against a known-`origin_D` injection (mirroring SP-4b Sec.2a's
known-`Delta_wall` sign-pinning), or SP-5c ingests sign-flipped constants.

### 4.3 The emulator inject-and-recover loop (a plumbing/regression check, not physics)

Injected skew constants are code-proven to touch **only** the trace timestamp
label (`origin_offset`, applied solely in `encode_start`) and the tile-timer
register (`reset_target`); the Dijkstra reached-set is `d_h/d_v`-invariant and no
scheduler/lock/DMA/PC path reads any skew value. So a run with injected constants
and a run at zero constants have **byte-identical execution timelines**;
`Delta_wall` cancels exactly *by construction*. The loop therefore validates the
extraction arithmetic and that SP-2's origin offset surfaces in the decoded trace
(the latter already wired: `trace_unit/mod.rs:1378`, `tools/trace_decoder/decode.py`);
it validates **no physics**. It is a regression/plumbing gate, not "the
load-bearing check" (rev1 overstated this).

Built as a Rust integration test driving `propagate_broadcasts_with_timing`
through the new runtime-override seam (Sec.6) with known constants, then decoding
and running `r1_extract`, asserting recovered == injected exactly. Contend with
the run's own zero-const trace-start flood (last-writer-wins, SP-4b Sec.4d): the
override must be in force at that flood.

### 4.4 Steady-state pair selection (honest contingency)

On silicon (SP-5c) there is no zero-const run; you subtract the emulator's
`Delta_wall`, faithful only if row 51 (the `-52` fill-cadence residual) is closed
or empirically absent in the chosen window. The extraction filters to a stable
REPS iteration window (`of_q0_rich` runs REPS=16), citing the SP-4a finding
in-code. **This filter drops warm-up iterations; it does NOT remove a persistent
inter-domain phase offset** (Sec.2). R1's silicon `d_v` is presented as contingent
on that gap, cross-checked by R3b -- not as a settled measurement.

### 4.5 Geometry and the falsification hook

R1 is **within-column**: a single column, vertical tile span. This escapes both
the cross-column ~30-cycle jitter and the cross-column trace-arbitration
truncation (SP-3's ConsB tail loss, `TASK3-RESULT.md:19-37`) in one move.

Geometry requirement: **>=3 collinear tiles at >=3 distinct vertical
hop-distances** (not merely a "pair") -- two points fit any line with zero
residual, so >=3 is required for the fit-residual to detect per-hop
*non-uniformity* at all. Include the memtile and shim (as the distance-0 / 4th-kind
anchors) and >=1 same-tile core<->memmod pair for the intra offsets.

The **fit-residual norm is a first-class output**: a large residual means the
linear model does not fit (non-uniform per-hop `d_v`) -- SP-5c reads that as "the
shape is wrong." Degenerate/rank-deficient geometry fails loud at extraction.
**[rev3 -- guard the inverse reading.]** A *green* residual on this within-column
one-sided geometry validates **per-hop uniformity only** (assumption 1 of 4,
Sec.2); it must **not** be read as "the 4-knob shape is confirmed" at the
`calibrated` flip. Isotropy, additivity, and channel-uniformity leave a green
residual regardless (they are unfalsifiable here), so the flip may cite a residual
as shape-validation only against the enriched geometry (kernel spec rev3 Sec.5.1);
otherwise the recorded provenance must say "uniformity-validated; isotropy,
additivity, channel-uniformity ASSUMED."

**Open risk (memmod traceability):** SP-3 deliberately did *not* trace the memmod
(its port events are memtile `DMA_Event_Channel_Selection`-dependent and
unreliable, `sp3-spike-trace/README.md:91-92`). Measuring `intra_mem` via trace
inherits that unreliability; the plan must either find a reliable memmod trace
event or source `intra_mem` from R3b. Flagged, not yet resolved.

### 4.6 Kernel

Built on the HW-proven `of_q0_rich` objectfifo skeleton (Q=0, reproducible, no
TDR), reduced to a single column with a vertical tile span. Traceable via
`mlir-trace-inject`. Also lands the SP-3 rank-2 gate-carrier.

---

## 5. Route 3b -- two-source perf-counter interval (primary for `d_h`, `d_v`)

### 5.1 The physics

Keep flood `s1` from one corner; add a second flood `s2` from a different corner.
The interval between wavefronts at tile X is
`r_X = (T0_2 - T0_1) + D(s2,X) - D(s1,X) = const + Dn_h*d_h + Dn_v*d_v`.
Differencing `r_X - r_Y` across tiles drops `const` and leaves a rank-2 system
solving `d_h, d_v` **directly on silicon, zero emulator dependence, no trace
decode**. `s2 != s1` is the trick (Wall 2). **`s1` must arrive before `s2` at
every measured tile** (else the counter measures nothing) -- a route-layout
constraint the plan must satisfy.

The measured interval is a **single-tile local** clock reading, so it does not
rely on cross-tile *phase* alignment (the array is a single clock domain --
uniform *rate*). That is R3b's genuine advantage over R1. **[rev3 -- "immune to
cross-column jitter" is a category error.]** The counter is *triggered* by the
`s1`/`s2` broadcast events, which traverse the same stream-switch crossing fabric
where the ~30cy variance lives, so cross-column *arrival* jitter perturbs the
counter's start/stop directly. R3b's immunity to run-to-run cross-column jitter is
therefore **empirical -- gated by the range-0 `b`-vector check** (kernel spec rev3
Sec.5.4), not guaranteed by the single-clock property. The ~30cy itself was
measured on `LOCK_STALL` events, not broadcast arrivals; broadcast-arrival jitter
is unmeasured and gets a direct pre-check (kernel rev3 Sec.5.4). A *deterministic*
cross-column bias would pass range-0 silently -- addressed by the two-sided /
channel-control geometry, not the jitter gate.

### 5.2 Mechanisms (both built; PerfCounter primary)

Neither is known-working; build both -- PerfCounter first (jitter-free), `LDA_TM`
second as corroborating cross-check and fallback.

- **PerfCounter interval (primary).** A performance counter (register block
  `0x31500`, a *separate* HW unit from the reset Timer at `0x34000`) configured
  START on `s1`'s broadcast event, STOP on `s2`'s. The counter value is `r_X`, a
  hardware interval -- no core polling, no schedule perturbation. Broadcast events
  are in the selectable event space (`broadcast_15` = 122 core/mem, 125 shim; no
  event whitelist in `XAie_PerfCounterControlSet`).

- **`LDA_TM` read (fallback/cross-check).** Core, event-triggered on `s2`,
  executes `__builtin_aiev2_read_tm` -> `LDA_TM` -> store -> DMA out
  (`llvm-aie .../IntrinsicsAIE2.td:469`; `Timer_Low @ 0x340F8`). Carries the
  skew-limit Sec.8 "roughly constant added latency" jitter caveat (why PerfCounter
  is primary). **This kernel reads its own timer, so it must never be run inside
  the R1 differential loop (Sec.4.3)** -- there it would break the `Delta_wall`
  cancellation.

### 5.3 Honest build scope (rev2 correction)

R3b is **not** a reuse of SP-3's `PERF_CNT` machinery -- SP-3's counter is a
free-running trace anchor, not broadcast-triggered. There is no declarative op
for a second flood and no perf-counter-config op in the dialect. R3b requires
**hand-authored `aiex.npu.write32` register programming** for (a) the second flood
(`Event_Broadcast{N}_A` / `Event_Generate` / `Timer_Control.Reset_Event`) and
(b) the perf-counter start/stop config (`Performance_Control*`), templated off
`AIEInsertTraceFlows.cpp` but net-new. Budget it as net-new kernel authoring, not
reuse. (The emulator flood model already handles two independent floods, and a
distinct-channel `s2` does not trip the ch15 single-source guard,
`effects.rs:604` -- so no emulator work is needed for R3b's HW-only validation.)

### 5.4 What SP-5b builds for R3b

Per the asymmetric-validation decision (R3b is emulator-independent by design, so
an in-emulator loop would be circular): kernel correctness + the rank-2 solve
unit-tested on synthetic data + HW runnability gate. **No numeric measurement, no
emulator model of R3b.** The synthetic solve test must also exercise the
falsification hook -- and because `const`-differencing removes one degree of
freedom, R3b needs **>=3 collinear tiles per axis** to over-determine uniformity
(one more than the naive rank-2 count). Numeric silicon validation is SP-5c.
**[rev3]** >=3-collinear-per-axis is necessary but **not sufficient** -- it
falsifies uniformity only. The full R3b geometry adds two-sided-per-axis,
off-axis-diagonal, and channel-uniformity control, with signed N/S+E/W and
interaction solver columns (`r3b_extract` becomes modified), so isotropy,
additivity, and channel-uniformity are over-determined too; synthetic tests must
inject each violation and require `fit_residual` growth. Full build spec: kernel
rev3 Sec.5.1/5.2.

---

## 6. Components and file layout

SP-5b modifies the emulator once: the **runtime-override seam**. `propagate_broadcasts`
(`src/device/state/effects.rs:564`) currently reads the four constants and
`calibrated` from compile-time archspec consts. rev2 adds an optional
runtime-settable override on emulator state that, when unset (default),
reads the zero consts (byte-identical, neutrality guards green), and when set,
supersedes them for a run. This is the only emulator change; it also gives SP-5c
a non-recompile path to load measured constants.

| Unit | Location |
|---|---|
| Runtime-override seam (4 consts + `calibrated`) | `src/device/state/effects.rs` + emulator state |
| R1 within-column kernel (= SP-3 gate-carrier) | `mlir-aie/test/npu-xrt/sp5_skew_r1/` |
| R3b PerfCounter kernel (primary) | `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/` |
| R3b `LDA_TM` kernel (fallback/cross-check) | `mlir-aie/test/npu-xrt/sp5_skew_r3b_tm/` |
| R1 extraction (residual -> `d_v,intra` + sign-pin + fit-residual) | `tools/calibration/skew/r1_extract.py` |
| R3b extraction (interval-difference -> rank-2; shared by both R3b kernels) | `tools/calibration/skew/r3b_extract.py` |
| R1 emu inject-and-recover harness (Rust integration test) | Rust test + `tools/calibration/skew/` glue |
| HW runnability gate (all three kernels) | `build/experiments/sp5-skew/*_gate.sh` |
| Measured-constants output schema | `tools/calibration/skew/skew_constants.json` (SP-5c populates) |

---

## 7. Testing tiers

1. **Synthetic-data unit tests (Python; no HW, no emu).** Both extraction solvers:
   inject known constants, recover exactly, confirm the fit-residual is ~0 on clean
   input and *grows* on structure-violating input **using >=3 collinear points per
   axis** (so the hook can actually fire). Rank-deficient geometry fails loud. R1's
   sign-pin test against a known-`origin_D` injection.
2. **Runtime-override seam neutrality test (Rust).** Default (unset) override ->
   byte-identical flood/trace vs today; the three existing neutrality guards stay
   green.
3. **R1 emu inject-and-recover (plumbing/regression).** Set constants via the seam,
   drive `propagate_broadcasts_with_timing`, decode, extract, assert recovered ==
   injected exactly. Validates arithmetic + SP-2 surfacing, not physics.
4. **HW runnability gates (cheap; all three kernels).** N serial runs on Phoenix --
   rc-0, no TDR, reproducible tile/event set, well-formed *non-degenerate*
   differential output. First real test of the two-flood config (R3b) and the
   within-column geometry (R1). Not measurement.

`cargo test --lib` stays green throughout (default behavior unchanged).

---

## 8. Risks

Load-bearing (adversarial-review priorities):

- **[R1] Cross-domain `Delta_wall` faithfulness is assumed, contingent on
  known-fidelity-gaps row 51** (Sec.2/4.4). If the `-52` fill-cadence residual is
  not steady-state-clean, R1's silicon `d_v` inherits it. Mitigation: R3b
  cross-check; do not present R1 silicon values as settled.
- **[R3b] Two-flood config + `s1`-before-`s2` ordering + PerfCounter
  start/stop-on-`BROADCAST_*`** all untested on Phoenix (Sec.5). The HW gate is
  the check; `LDA_TM` is the fallback.
- **[both] Structure-vs-values** (Sec.2): >=3 collinear tiles per axis are required
  or neither instrument can falsify per-hop non-uniformity -- the exact SP-5c
  question. Baked into Sec.4.5/5.4/7.

Secondary:

- [R1] memmod traceability for `intra_mem` (Sec.4.5) -- SP-3 dropped it on purpose.
- [R1] sign-convention reconciliation (Sec.4.2) -- reflected trace offset vs `origin_D`.
- [R3b] kernel authoring is hand-written register programming, not SP-3 reuse
  (Sec.5.3) -- budget accordingly.
- [R3b] `LDA_TM` per-tile read-latency jitter (why PerfCounter is primary); and
  the `LDA_TM` kernel must never enter the R1 loop (Sec.5.2).

---

## 9. SP-5b -> SP-5c handoff

SP-5b ships apparatus that provably recovers known constants (R1 emu loop, plumbing
tier) and provably runs on silicon (all three kernels, HW gate). No measured number
is produced in SP-5b.

SP-5c (Phoenix): runs the apparatus, reads `d_h/d_v` (R3b primary) and `d_v`/intra
(R1, cross-checking R3b's `d_v`), resolves the structure questions (per-hop
uniformity via the >=3-collinear residual; whether `d_h/d_v` collapse; intra-tile
sign vs the add_one `+2/+4/-2` signature -- only `-2` is intra-tile, `+2/+4` are
`origin_D`), writes `skew_constants.json`, loads them via the runtime-override seam
(then bakes them into archspec for the permanent flip), flips `calibrated`, updates
the three regression guards -- **verified current locations**:
`crates/xdna-archspec/src/runtime.rs:807`, `src/interpreter/engine/coordinator.rs:4078`,
`src/device/state/effects.rs:1355` -- and validates causal-vs-HW. The two SP-4b Sec.9a
prerequisites (fixpoint ch15 multi-source test -- largely covered by SP-5a; wiring
the sweep sidecar consumption) and the P1 round-trip gate fold in here.

**[rev3] The causal-vs-HW gate must run on a held-out kernel.** Because
`skew := HW_offset - Delta_wall_emu`, a calibrated emulator reconstructs the exact
HW offset on the *same* geometry it was fit to -- the terminal "causal-vs-HW" gate
is a **tautology** on the calibration kernel, and nothing currently forbids running
it there. SP-5c MUST validate on a geometrically-distinct, held-out kernel whose
`Delta_wall` varies non-collinearly with per-module skew, or the sole backstop
cannot fail. The full pre-flip gate list (held-out kernel, enriched-geometry
residuals green, joint sign anchors, `dn_v`-correlated Delta_wall quantified, hard
cross-column b-vector gate) is in kernel spec rev3 Sec.11.

`skew_constants.json` schema (SP-5b defines; SP-5c populates):

```json
{
  "d_h": null, "d_v": null,
  "intra": { "core": null, "mem": null },
  "fit_residual": null,
  "source_route": "r1 | r3b_pc | r3b_tm",
  "provenance": "measured-silicon"
}
```

All `null` until SP-5c measures; SP-5b's synthetic + emu tests exercise the schema
round-trip with injected non-null values.

---

## 10. Reference map

- Arc + decomposition: `2026-06-28-timer-sync-faithful-broadcast-arc.md`
- Epistemic boundary + routes: `docs/trace/cross-domain-skew-limit.md` (Sec.6 walls,
  Sec.8 routes)
- SP-4a cold-start finding + row 51:
  `docs/superpowers/findings/2026-06-29-coldstart-headstart-trace-baseline.md`,
  `docs/known-fidelity-gaps.md`
- HW jitter data driving the flip: `build/experiments/sp3-spike-trace/SP4A-HW-TARGETS.md`,
  `TASK3-RESULT.md`, `README.md`
- Engine gate this feeds: `2026-06-30-sp4b-skew-export-design.md`
- Plumbing this precedes: `2026-06-30-sp5a-calibration-enablement-design.md`
- Column frames: `include/aie/Dialect/AIE/IR/AIETargetModel.h:736`,
  `docs/device-model-audit.md:58`
- Model constants: `crates/xdna-archspec/src/model_builder.rs:270-279`; flood
  application `src/device/state/effects.rs:564`
- `LDA_TM`: `llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2.td:469`; perf counter
  `Performance_Control*` @ `0x31500` (AM025)
- Kernel skeleton + repro recipe: `build/experiments/sp3-spike-trace/`,
  `mlir-aie/test/npu-xrt/spike_bringup/`
