# SP-5b: Skew Measurement Apparatus -- Design

Issue #140, timer-sync faithful-broadcast arc. Sub-project SP-5b of the SP-5
decomposition (5a calibration enablement [done], 5b measurement apparatus
[this], 5c Phoenix campaign + go-live).

Prior art: `2026-06-28-timer-sync-faithful-broadcast-arc.md` (the arc),
`docs/trace/cross-domain-skew-limit.md` (the epistemic boundary; routes in Sec.8),
`2026-06-30-sp4b-skew-export-design.md` (the engine-side gate this feeds),
`2026-06-30-sp5a-calibration-enablement-design.md` (the plumbing this precedes).

---

## 1. Scope and non-goals

SP-5b builds and validates the two skew-measurement instruments so SP-5c can
point them at Phoenix silicon, read `d_h / d_v / intra-tile`, and flip
`BroadcastTiming.calibrated`. **SP-5b produces apparatus, not numbers.**

| In SP-5b | Out (deferred to SP-5c) |
|---|---|
| Build both kernels (R1 + both R3b mechanisms) | The numeric Phoenix skew measurement |
| R1: full in-emulator inject-known-constants -> recover loop | Ingesting measured constants into `BroadcastTiming` |
| R3b: kernel correctness + rank-2 solve on synthetic data | R3b numeric silicon validation |
| Cheap HW *runnability/repro* gate on all three kernels | Flipping `calibrated` + updating the 3 regression guards |
| Fit-residual (structure-falsification) reporting | Causal-vs-HW validation; P1 round-trip gate |

**Non-goals.** No numeric skew measurement. No emulator feature work (Sec.6
establishes this design needs none). No `calibrated` flip. No engine/inference
changes -- SP-4b/SP-5a already built the consuming side, dormant behind
`calibrated=false`.

**Why the arc has not solved this before, and why we can now.** The skew is
information-theoretically undeterminable *from a trace alone* (skew-limit Sec.6,
three independent walls). Every prior framing that stayed trace-only failed on
those walls. The two instruments here each beat a specific wall with one
load-bearing assumption (Sec.2); a design that quietly reverts to trace-only
will fail again. The safety net: the whole SP-5 arc **cannot fail into
wrongness, only into no-improvement** -- everything downstream is gated behind
`calibrated=false`, so a bad measurement leaves today's behavior
(`reproduction_offset` emitted, no causal fact) untouched. There is no path
where a bad skew number launders into a wrong causal claim; the `ModelDerived`
provenance leaf and the `calibrated` gate (SP-4b) make that structurally
impossible.

---

## 2. The two instruments and their load-bearing assumptions

Recorded up front because these are what the adversarial review must hammer.

- **Route 1 (trace tile-distance sweep, primary/reproduction path)** beats Wall 1
  (trace underdetermination: one equation, two unknowns per domain pair) by
  bringing an *independent second equation* -- the emulator's `Delta_wall` from a
  within-domain-exact forward model. **Load-bearing assumption: the emulator
  `Delta_wall` is trustworthy, which SP-4a established holds only in
  steady-state** (the forward spine is bit-faithful; cold-start is
  trace-perturbed). R1's extraction MUST use steady-state event pairs or it
  re-inherits SP-4a's contamination and fails exactly as before. Cite
  `docs/superpowers/findings/2026-06-29-coldstart-headstart-trace-baseline.md`.

- **Route 3b (two-source timer read, emulator-independent cross-check)** beats
  Wall 2 (no globally-simultaneous traceable signal: a co-observed single flood
  cancels, `soc_A(E) = soc_B(E)`) by using a *second* flood `s2` from a different
  corner, so its arrival delay does not cancel against `s1`'s. **Load-bearing
  assumption: `s2 != s1` genuinely (delays do not cancel) and the two-flood
  configuration is valid on silicon** (untested). Secondary: the per-tile read
  latency is constant across tiles (Sec.5 retires this for the primary
  mechanism).

R3b is the only measurement with **zero emulator dependence** -- it is what breaks
R1's soft-identifiability (skew-limit Sec.7): otherwise the emulator would both
produce and validate the skew numbers.

---

## 3. Column frames (load-bearing for all kernel geometry)

Phoenix NPU1 has two topology representations that do not share a column
numbering, and conflating them is the source of recurring "column weirdness".

- **Physical Phoenix**: 5 columns; absolute column 0 has **no shim tile** (its
  mem + core tiles exist but have no NoC/DDR path). Recorded at
  `docs/device-model-audit.md:58` (aiesim `NPU1.json` fact sheet:
  "5 cols (4+1, col 0 has no shim tile)").

- **mlir-aie toolchain (virtualized frame)**: the *only* NPU1 model is
  `VirtualizedNPU1TargetModel` (`include/aie/Dialect/AIE/IR/AIETargetModel.h:736`,
  class comment "A sub-portion of the Phoenix NPU"), variants `1Col..4Col`
  (`4Col` = "whole array"); there is no `NPU1_5Col`. `getTileType(col,row)`
  returns `ShimNOCTile` for row 0 of *every* modeled column unconditionally
  (line 752-758). So in the toolchain frame the shim-less column does not exist:
  every modeled column has a shim, spanning at most 4 cols x 6 rows (row 0 shim,
  row 1 memtile, rows 2-5 core).

Firmware relocates the virtualized partition onto physical columns 1..N at load
(hence the emulator's absolute-frame "partition starting at column 1" comments,
e.g. `src/device/state/dispatch.rs:517`).

**Design rule for SP-5b:** specify all kernel geometry in the **virtualized
frame** (relative cols 0..N-1, all shim-bearing, N<=4; rows 0..5). Hop
*distances* are relocation-invariant (relative spacing preserved under partition
relocation), so distances specified this way are valid on silicon. Never reason
about a shim at physical column 0.

(Aside, out of scope: physical col-0 tiles are not binned -- per Maya they were
accessible and rewritten inaccessible, plausibly re-enableable. Tracked as a
separate later investigation, not part of SP-5b.)

---

## 4. Route 1 -- trace tile-distance sweep

### 4.1 The model being fit

Each module X (a tile's core or memmod, a memtile, a shim) receives the
BROADCAST_15 reset at:

```
origin_X  =  Dn_h(X)*d_h  +  Dn_v(X)*d_v  +  intra(kind_X)
```

`Dn_h, Dn_v` are hop counts from source `s1` (known from geometry); `d_h, d_v`
are per-hop delays; `intra(kind)` is the within-tile module offset. This is the
nominal 4-knob shape (`d_h`, `d_v`, and the intra-tile offsets, with one module
kind chosen as the reference at 0). The extraction is built as a **design
matrix** -- each module's `origin` is a linear combination of
`[Dn_h, Dn_v, kind-indicators]` -- so the intra parameterization and knob count are
not hardcoded; whatever is parameterized, the solver fits, and the fit residual
(Sec.4.5) reports whether the shape holds. SP-5b only validates the apparatus
recovers whatever is injected; SP-5c settles the true silicon structure (Sec.9).

### 4.2 Measured -> residual -> solve

For a cross-domain event pair (x in domain A, y in domain B) that fire
deterministically together, the trace gives
`soc(x) - soc(y) = Delta_wall(x,y) + skew(A,B)`. R1 isolates skew:

```
skew(A,B)  =  measured_offset(x,y)  -  Delta_wall(x,y)
```

`Delta_wall` comes from the emulator run with `calibrated=false` (skew=0, so the
emulator offset *is* `Delta_wall`). Many (A,B) pairs across the geometry -> an
over-determined linear system in `{d_h, d_v, intra}` -> least-squares solve plus
a fit-residual norm.

### 4.3 The emulator inject-and-recover loop (the headline no-HW check)

Two emulator runs of the same kernel:

```
run T (calibrated=true, injected consts):  offset_T(x,y) = Delta_wall + injected_skew
run F (calibrated=false):                  offset_F(x,y) = Delta_wall
recovered_skew(A,B) = offset_T - offset_F = injected_skew(A,B)   -> solve -> assert == injected
```

The differencing cancels `Delta_wall` **and any common cold-start perturbation**
(same kernel, deterministic emulator), so recovery is **exact**. This is the
acceptance test for the whole R1 pipeline and needs no hardware. It is clean
despite SP-4a's cold-start finding precisely because it is differential.

### 4.4 Steady-state pair selection (load-bearing, SP-4a)

On silicon (SP-5c) there is no `run F`; you subtract the emulator's `Delta_wall`,
and per SP-4a the emulator `Delta_wall` is bit-faithful only in **steady state**.
So the extraction's pair-selector MUST filter to steady-state events -- concretely,
drop the warm-up iterations and use a stable iteration window of the REPS loop
(the `of_q0_rich` skeleton runs REPS=16). This filter is built into the tool now
so the identical tool works unchanged on HW; the emulator loop does not need it
(it cancels), but shipping the tool without it means SP-5c silently re-inherits
SP-4a's contamination. The filter cites the SP-4a finding in-code.

### 4.5 Geometry and the falsification hook

Geometry requirement (concrete placement worked out in the plan against the
virtualized-frame topology of Sec.3 and objectfifo routing feasibility, which
SP-3 proved flexible): >=2 linearly-independent hop-difference vectors (a
purely-vertical tile pair to isolate `d_v`, a purely-horizontal pair to isolate
`d_h`, >=1 diagonal to over-determine), same-tile core<->memmod pairs to isolate
`intra`, all four module kinds present (shim + memtile as the structure-validation
extras). Source `s1` at a corner shim (virtualized frame).

The **fit-residual norm is a first-class output**: on silicon a large residual
means the linear 4-knob model does not fit (non-uniform per-hop delay, or
`d_h/d_v` do not cleanly separate) -- SP-5c reads that as "the *shape* is wrong",
not merely "the *values* differ". Without it we would fit noise and never know.
Degenerate geometry (rank < 2) fails loud at extraction time.

### 4.6 Kernel

Built on the HW-proven `of_q0_rich` objectfifo skeleton (Q=0, cross-column,
reproducible, no TDR -- SP-3 Task 3). Bidirectional on-chip, >=2 hop counts,
in-window events, traceable via the `mlir-trace-inject` pipeline. This artifact
also lands the SP-3 rank-2 validation kernel that never shipped -- one kernel
serves as both the R1 measurement instrument and the SP-3 gate-carrier.

---

## 5. Route 3b -- two-source timer read

### 5.1 The physics

Keep flood `s1` from one corner; add a second flood `s2` from a different corner.
The interval between the two wavefronts at tile X is:

```
r_X = (T0_2 - T0_1) + D(s2,X) - D(s1,X)  =  const + Dn_h*d_h + Dn_v*d_v
```

Differencing `r_X - r_Y` across tiles drops `const` and leaves a rank-2 linear
system solving `d_h, d_v` **directly on silicon, zero emulator dependence, no
trace decode**. `s2 != s1` is the whole trick (Wall 2); same source and the
delays cancel, zero signal.

### 5.2 Two mechanisms (both built; PerfCounter primary)

Neither is known-working; both are "reasoned, untested". Build both -- PerfCounter
first (skips the jitter), `LDA_TM` second as corroborating cross-check and
fallback if PerfCounter's event triggering does not work on Phoenix.

- **PerfCounter interval (primary).** A performance counter configured to START
  on `s1`'s broadcast event and STOP on `s2`'s broadcast event. The counter value
  *is* `r_X` -- a hardware-measured interval, no core polling, no compute-schedule
  perturbation. Reuses the `PERF_CNT` machinery SP-3's spike already anchored on
  broadcast events. Risk: start/stop-on-`BROADCAST_*` + readout unconfirmed on
  Phoenix.

- **`LDA_TM` read (fallback/cross-check).** The core, event-triggered on `s2`,
  executes `__builtin_aiev2_read_tm` (`llvm-aie .../IntrinsicsAIE2.td:469` ->
  `LDA_TM`, an MMIO load of `Timer_Low @ 0x340F8` / `Timer_High @ 0x340FC`) ->
  store -> DMA out. Exactly as skew-limit Sec.8 documents. Risk: the "roughly
  constant added latency" caveat -- core-side polling has +-1-iteration jitter that
  masquerades as skew (this is why PerfCounter is primary).

Both share the same rank-2 extraction (Sec.5.3) and the same genuine hard part:
two independently-configured floods (`s2` on a distinct broadcast channel from
another corner). That config is R3b's untested crux; the HW gate (Sec.7) is its
first real test.

### 5.3 What SP-5b builds for R3b

Per the asymmetric-validation decision (R3b's whole point is
emulator-independence, so an in-emulator loop would be circular and force
modeling the exact timing R3b measures): kernel correctness + the rank-2 solve
unit-tested on synthetic data + HW runnability gate. **No numeric skew
measurement, no emulator model of R3b.** Numeric silicon validation is SP-5c.

---

## 6. Components and file layout

No new emulator features: the flood, `BroadcastTiming` knobs, and trace-origin
offset already exist (SP-1/2/4b). SP-5b is kernels + Python extraction + a test
harness + HW gates.

| Unit | Location |
|---|---|
| R1 unified measurement kernel (= SP-3 gate-carrier) | `mlir-aie/test/npu-xrt/sp5_skew_r1/` |
| R3b PerfCounter kernel (primary) | `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/` |
| R3b `LDA_TM` kernel (fallback/cross-check) | `mlir-aie/test/npu-xrt/sp5_skew_r3b_tm/` |
| R1 extraction (residual -> `d_h,d_v,intra` + fit-residual) | `tools/calibration/skew/r1_extract.py` |
| R3b extraction (interval-difference -> rank-2; shared by both R3b kernels) | `tools/calibration/skew/r3b_extract.py` |
| R1 emu inject-and-recover harness | Rust test + `tools/calibration/skew/` glue |
| HW runnability gate (all three kernels) | `build/experiments/sp5-skew/*_gate.sh` (SP-3 Task-3 style) |
| Measured-constants output schema | `tools/calibration/skew/skew_constants.json` (SP-5c ingests) |

---

## 7. Testing tiers

1. **Synthetic-data unit tests (Python; no HW, no emu).** Both extraction
   solvers: inject known constants into the forward formula, recover exactly, and
   confirm the fit-residual is ~0 on clean input and *grows* on structure-violating
   input (falsification hook works). Degenerate/rank-deficient geometry fails loud.

2. **R1 emu inject-and-recover (the load-bearing no-HW check).** Set
   `BroadcastTiming` constants in-emulator, run the R1 kernel through emu
   (`calibrated=true` vs `false`), extract, assert recovered == injected exactly.
   Exercises the real broadcast-flood + trace-origin path (verify during impl that
   the SP-2 origin offset surfaces in the decoded trace).

3. **HW runnability gates (cheap; all three kernels).** N serial runs on Phoenix
   -- rc-0, no TDR, reproducible tile/event set, well-formed *non-degenerate*
   differential output (reads/intervals must differ across tiles). Explicitly
   **not** measurement. First real test of the two-flood config (R3b) and the
   new geometry (R1). "HW is the cheap oracle" -- err toward more runs.

`cargo test --lib` stays green throughout (no emulator behavior change; the R1
harness adds tests, does not alter the flood).

---

## 8. Risks

Load-bearing (flagged for the adversarial reviewer):

- **[R1] Steady-state pair selection** (Sec.4.4). Without the steady-state
  filter, R1 re-inherits SP-4a cold-start contamination on silicon.
- **[R3b] `s2 != s1` + two-flood config validity** (Sec.5.1-5.2). The second
  flood from another corner on a distinct broadcast channel is the untested crux;
  if it cancels or misconfigures, R3b yields nothing. HW gate is the check.

Secondary:

- [R3b] PerfCounter start/stop-on-`BROADCAST_*` + readout unconfirmed on Phoenix
  (why `LDA_TM` is the built-in fallback).
- [R3b] `LDA_TM` per-tile read-latency jitter (why PerfCounter is primary).
- [R1] The emulator trace-origin offset (SP-2) must surface in the decoded trace
  for the emu loop to exercise the real path -- verify during impl.
- [geometry] Virtualized-frame span is only 4 cols x 6 rows; confirm it yields
  >=2 independent hop-difference vectors for rank-2 (Sec.4.5 / Sec.3).
- [scope] R3b builds two kernels; keep the shared rank-2 solver DRY across them.

---

## 9. SP-5b -> SP-5c handoff

SP-5b ships apparatus that *provably recovers known constants* (R1 in-emu, Tier 2)
and *provably runs on silicon* (all three kernels, Tier 3). No measured number is
produced in SP-5b.

SP-5c (Phoenix, HW-gated) then: runs the apparatus on silicon, reads real
`d_h/d_v/intra`, writes `skew_constants.json`, resolves the structure questions
(per-hop uniformity; whether `d_h/d_v` collapse; the intra-tile sign vs the
add_one `+2/+4/-2` signature -- only `-2` is intra-tile, `+2/+4` are `origin_D`),
sets the constants into `BroadcastTiming`, flips `calibrated`, updates the three
regression guards (`runtime.rs:799`, `coordinator.rs:4077`, `effects.rs:1248`),
and validates causal-vs-HW. The two SP-5c prerequisites from SP-4b design Sec.9a
(fixpoint ch15 multi-source test -- largely covered by SP-5a's provenance fix; and
wiring the sweep sidecar consumption) fold in there. P1 round-trip gate folds in
there.

`skew_constants.json` schema (SP-5b defines it; SP-5c populates it):

```json
{
  "d_h": null, "d_v": null,
  "intra": { "core": null, "memmod": null, "memtile": null, "shim": null },
  "fit_residual": null,
  "source_route": "r1 | r3b_pc | r3b_tm",
  "provenance": "measured-silicon"
}
```

All values `null` until SP-5c measures them; SP-5b's synthetic + emu tests
exercise the schema round-trip with injected non-null values.

---

## 10. Reference map

- Arc + decomposition: `2026-06-28-timer-sync-faithful-broadcast-arc.md`
- Epistemic boundary + routes: `docs/trace/cross-domain-skew-limit.md` (Sec.6 walls,
  Sec.8 routes 1/3b)
- SP-4a cold-start finding (steady-state constraint):
  `docs/superpowers/findings/2026-06-29-coldstart-headstart-trace-baseline.md`
- Engine gate this feeds: `2026-06-30-sp4b-skew-export-design.md`
- Plumbing this precedes: `2026-06-30-sp5a-calibration-enablement-design.md`
- Column frames: `include/aie/Dialect/AIE/IR/AIETargetModel.h:736`,
  `docs/device-model-audit.md:58`
- `LDA_TM`: `llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2.td:469`; timer regs
  `Timer_Low @ 0x340F8` / `Timer_High @ 0x340FC` (AM025)
- Kernel skeleton + repro recipe: `build/experiments/sp3-spike-trace/`,
  `mlir-aie/test/npu-xrt/spike_bringup/`
