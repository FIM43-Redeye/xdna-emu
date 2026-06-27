# Timer-Sync via Route-1: Emulator Broadcast Forward Model — Design

**Status:** approved design (brainstorm + two 3-lens adversarial-review rounds +
a toolchain-derivation keystone spike, 2026-06-27). First implementable
sub-project of the timer-sync arc.
**Issue:** #140 (byte-identical emulator/HW trace reports).
**Read first:** [`../../trace/cross-domain-skew-limit.md`](../../trace/cross-domain-skew-limit.md)
(the epistemic boundary; §7-9 are the design rule this spec implements) and
[`../../trace/capture-load-sensitivity.md`](../../trace/capture-load-sensitivity.md)
(host-load contamination of trace timing).
**Supersedes:** [`2026-06-27-broadcast15-skew-silicon-characterization-design.md`](2026-06-27-broadcast15-skew-silicon-characterization-design.md)
(the measure-first detour — see its banner for why).
**Prior art landed:** the engine-side reproduction-target annotation
(`tools/inference/grounding.py`: `reproduction_offset` + `is_async_cdc`) is
already merged; this spec does not touch it.

## How we got here (decision record)

The emulator currently gives every traced module the same trace-time origin, so
cross-domain trace-timestamp offsets carry **zero skew**. Real HW resets each
module's trace timer via a BROADCAST_15 flood that arrives a few cycles apart
(per-module `origin_D`), producing a fixed cross-domain skew (add_one measured:
core−memtile +2, memmod−memtile +4, core−memmod −2). Reproducing that skew is
the goal.

The path to *this* design ran through two rejected ones, both killed by
adversarial review (the same pattern that killed two prior attempts before this
arc):

1. **Emulator-first with an add_one gate** — rejected: the gate was *vacuous*.
   add_one's three pairs span only 3 modules → 2 independent skews, against a
   2-parameter model = exact fit by construction. The only falsifiable gate
   (in-domain) never touches the broadcast code.
2. **Measure-first active-probe characterization** — rejected: it inverted the
   skew-limit doc's own route priority (promoted the doc's speculative
   *fallback* route-3 to primary), demanded range-0 determinism from the shim
   (provably impossible on add_one — it's DMA-only and DDR-fed), only de-vacuumed
   the skew half (leaving coupling latency), and the probe instrument was the
   larger, wedge-riskier build.

The **keystone spike** then resolved the question both rejections turned on —
*are inter-tile coupling latencies pinnable from the toolchain?* — with a precise
yes-for-what-matters:

- **On-chip (non-DDR) coupling latency IS pinned**, already as emulator
  constants: stream-switch hop `ROUTE_PER_HOP = 4` cy
  (`crates/xdna-archspec/src/model_builder.rs:254`, AM020 ch2; applied in
  `src/device/array/routing.rs:1278`); DMA pipeline `bd_setup(4) +
  channel_start(2) + memory_latency(5) ≈ 11` cy (aie-rt `DATAMEMORY_WIDTH` +
  AM020).
- **Shim/DDR coupling latency is NOT pinned** — it is hardware-measured
  (`shim_ddr_cold_start_*`, a 50-run calibration) and the DDR crossing is
  non-deterministic by design.

That is exactly enough: on-chip cross-domain pairs have a pinned coupling
latency (zero free param), so their cross-domain residual *is* the skew; shim/DDR
pairs stay **gap-only** (`is_async_cdc`), which we already accept. This is the
skew-limit doc's **route-1** (§8.1, the doc's *primary, no-new-instrument*
route), made concrete.

## The epistemic foundation (what is determinable, and how)

Per the skew-limit doc §6, a trace *in isolation* cannot split the recorded
cross-module offset `Δwall + skew`. The escape (§7) is an
**independently-verified compute model**: the emulator runs on one global clock
and predicts a wall-time `W_sim(e)` for every event, so `ΔW_sim(x,y)` is its
prediction of `Δwall(x,y)` for *any* pair, cross-domain included. Verify that
prediction **skew-free within each domain** (where `skew ≡ 0`), then the
cross-domain residual isolates the skew:

```
skew(A,B) = [HW raw cross-domain offset] − ΔW_sim(x,y)
          =  measured  −  (verified) Δwall
```

This is traces **plus a verified compute model**, not traces alone — non-circular
because `ΔW_sim` is validated where skew cannot reach. The one degeneracy that
survives in-domain verification is the *alignment between two domains* =
inter-tile coupling latency + skew; the spike shows the on-chip coupling latency
is pinned, so for on-chip pairs the residual is pure skew.

**Two distinct "per-hop" quantities — do not conflate them:**
- **Stream/data coupling latency** (`ROUTE_PER_HOP = 4` cy + DMA pipeline) —
  propagation of *data* through the stream network. **Pinned**; subtracted to get
  `ΔW_sim` right.
- **Broadcast-flood per-hop delay `d`** — propagation of the *BROADCAST_15
  timer-reset flood*. **Unspecified by the toolchain** (IRON docs: "a few clock
  cycles between tiles"); this is the unknown we *solve* from the cross-domain
  residual.

## Goal and success criteria

**Goal:** the emulator reproduces HW's cross-domain trace-timer raw offsets for
the deterministic on-chip content, by modeling the BROADCAST_15 flood's per-hop
delay, validated by an *over-determined* (hence falsifiable) gate.

**Success:**
1. The emulator's BROADCAST_15 flood resets each module's timer at a per-module
   `origin_D` that emerges from a cycle-stepped flood with a per-hop delay.
2. **In-domain gate:** the emulator reproduces every within-domain segment of the
   validation kernel exactly (Q=0), validating `W_sim` skew-free.
3. **Over-determined cross-domain gate:** on a *hop-diverse* kernel, every on-chip
   cross-domain pair's raw offset matches HW (`emu raw == hw raw`, Q=0), with a
   *single* solved broadcast per-hop delay `d` (+ intra-tile core/mem offset)
   consistent across pairs of *different hop counts*. This can fail → falsifiable.
4. Shim/DDR cross-domain pairs are correctly treated gap-only and excluded from
   the grounded gate.
5. No constant tuned to pass; `d` is solved from the residual, the coupling
   latencies are the existing toolchain-derived constants (tested, not fit).

## Why the gate is non-vacuous (parameter counting)

The vacuity that killed the prior designs was *measurements ≤ free parameters*.
Here:

- **Free parameters:** broadcast per-hop delay `d` (vertical), possibly a
  distinct horizontal `d_h` (cross-column), and the intra-tile core/mem offset.
  ~2-3 free params.
- **Pinned (not free):** the on-chip coupling latencies (`ROUTE_PER_HOP`, DMA
  pipeline). They are *tested* by the gate, not fit: a pair at 1 hop and a pair
  at 3 hops constrain `ROUTE_PER_HOP` separately from `d`, so a wrong coupling
  constant cannot hide behind `d`.
- **Independent measurements:** every on-chip cross-domain pair on a hop-diverse
  kernel — many, with varied `Δhops`.

When the kernel supplies more independent on-chip cross-domain pairs (varied hop
counts) than free params, the joint fit is **over-determined**: no single `d` +
intra-offset can satisfy all pairs unless the compute model, the coupling
constants, *and* the broadcast model are all right. That is the falsifiable test
the prior designs lacked. **add_one's single column (≈2 hops, 3 modules) is
structurally too degenerate** — the validation kernel must be hop-diverse.

## Components

The first review established the flood **already exists and is correct**; this is
mostly *adding delay* and *reconciling the trace origin*, not building new
machinery.

**1. Per-hop-delay scheduling on the existing flood.** Today
`seed_broadcasts_for_event` → `propagate_broadcasts`
(`src/device/state/effects.rs:457-553`) already fires on shim USER_EVENT_1,
floods tile-to-tile, and calls `TileTimer::notify_event` with the **correct
per-module event IDs** (core 122, memtile 142+15=157, shim PL 110+15) — but it
notifies every recipient on the *same* cycle (zero skew). The change: schedule
each recipient's `notify_event` at `t_fire + d·hops`, so the wave arrives
staggered and `origin_D` emerges. **Reuse the existing per-module ID resolution;
do not hardcode a single id or channel** (the memtile-157-vs-122 and dynamic-
channel traps the review flagged). One cleanup the review surfaced: the broadcast
bases are currently hardcoded (`effects.rs:480-482`, `107/110/142`) rather than
regdb-derived — fold that toward derivation rather than extending the hardcode.

**2. Trace-origin reconciliation.** The trace unit's own `timer` field is **dead
code** (`src/device/trace_unit/mod.rs:159`, written never read); the real origin
is the global `cycle` argument threaded into `notify_event` / `commit_cycle`. The
change: the per-module trace timestamp must reflect `origin_D` — i.e. the Start
frame and deltas derive from the flood-reset per-module timer value, not absolute
`cycle`. Cleanest realization: offset `cycle` per-module at the coordinator
boundary by `origin_D` (a small change at the notify call sites), rather than
plumbing `TileTimer` through the whole trace API. **Within-domain deltas stay
byte-identical** (delta = `pending_cycle − last_event_cycle`, origin-invariant);
only the Start frame's absolute value shifts — `test_start_marker_encoding`
(asserts all-zero Start) must be updated to the per-module origin (HW shows a
nonzero Start, e.g. `f0 00 00 00 00 32 67 c5`).

**3. Ordering + clock-gating fixes (the correctness hazards the review found).**
- `TileTimer::notify_event` *latches* `pending_reset`, applied on the next
  `tick()`; a clock-gated module is not ticked
  (`coordinator.rs:1491-1492`), so a flood arriving during gating would apply the
  reset at the *un-gate* cycle, corrupting `origin_D`. Fix: apply the reset
  independent of the gate, or guarantee the flood completes before the module is
  gated/active.
- The flood must complete (all `origin_D` set) **before** that module's first
  in-window traced event. On HW timers sync during config, before the workload.
  Establish and assert this ordering invariant; the monotonicity of the
  skip-token/held-level encoder depends on the timer not resetting mid-stream.
- The shim's own-timer self-reset is bypassed today (the local `Event_Generate`
  path calls the trace unit directly, not `notify_*_trace_event`); the shim's
  timer (Reset_Event = USER_EVENT_1) must reset when it generates the event.

**4. In-domain verification gate.** Run the validation kernel on EMU, decode,
assert every within-domain segment grounds exactly (Q=0) — validates `W_sim`
*before* any skew is read. This is the genuinely load-bearing, falsifiable
compute-model gate. **First implementation step:** establish the in-domain
baseline on the candidate kernels (does EMU reproduce them within-domain today?);
a failure here is a compute-fidelity bug to surface and fix/scope, not paper over.

**5. Skew solver + over-determined cross-domain gate.** Offline: for each on-chip
cross-domain pair, `skew = HW_raw − ΔW_sim`; solve the single `d` (+ intra-offset)
across all pairs (least-squares over `Δhops`); the fit *residual* is the
uniform-per-hop test. The gate asserts `emu raw == hw raw` for every on-chip pair
once `d` is set. Reuses the inference engine's grounding/verifier (Q=0, the shared
comparator). Shim/DDR pairs are `is_async_cdc` and excluded.

## Validation kernel (hop diversity is a requirement, not a detail)

The gate's non-vacuity *requires* on-chip cross-domain pairs with varied hop
counts. add_one is disqualified (single column, degenerate). Candidates already
in the trace corpus: **two_col** (cross-column distribute/gather — horizontal +
vertical hops; used in the connectivity work) and **matrix_transpose** (already
traced). The **first planning step** selects the kernel by *verifying it actually
produces hop-diverse, deterministic, on-chip cross-domain pairs* (anchor-relative
offsets range-0 on a quiet host) — not assuming it. If neither suffices, a
purpose-built multi-row-column kernel is the fallback.

## Testing

- **Offline units (`cargo test --lib`, TDD red-first):** per-hop-delay scheduling
  produces staggered `origin_D` in flood order; the existing per-module ID
  resolution is preserved (regression guard against the 157-vs-122 trap);
  trace-origin reconciliation shifts the Start absolute value while leaving
  within-domain deltas byte-identical (update `test_start_marker_encoding`);
  clock-gating-vs-latched-reset ordering; shim self-reset.
- **In-domain gate (HW + EMU):** every within-domain segment of the validation
  kernel reproduces exactly (Q=0).
- **Over-determined cross-domain gate (HW + EMU):** all on-chip cross-domain
  pairs match HW under a single solved `d`; the cross-pair fit residual is
  reported as the uniform-per-hop verdict. Quiet-host precondition (assert range-0
  before trusting any HW offset; flag contamination, never average it).
- **Within-domain regression:** the trace sweep / matrix regression stays green
  (it keys on origin-invariant deltas, so an origin shift must not perturb it).

## Honesty fences

- **Q = 0, no tuning.** `d` is *solved* from the residual; coupling latencies are
  the existing toolchain-derived constants, *tested* by hop-diversity, not fit. A
  non-zero cross-pair residual is a falsification to diagnose (wrong coupling
  constant, wrong compute model, or non-uniform `d`), never an adjustable
  tolerance.
- **Derive IDs from the toolchain**, per module, for every broadcast channel and
  event — never hardcode 15/122/157 (the indexed-mapping bug class). Fold the
  existing hardcoded broadcast bases toward regdb derivation.
- **Shim/DDR stays gap-only.** The shim's measured range-23 non-determinism is
  *expected and fine* — those pairs were never grounded targets. The deterministic
  on-chip content stays cycle-accurate across every DMA gap (skew-limit §7).
- **Coupling-latency provenance documented honestly:** the on-chip constants are
  AM020-prose-derived (hand-written), not data-driven extraction; the gate's
  over-determination is what validates them empirically.

## Out of scope (named follow-ons)

- **Cross-column horizontal `d_h` as its own study**, if the validation kernel
  shows vertical and horizontal per-hop delays differ enough to need separate
  solving beyond what the chosen kernel constrains.
- **The active probe-broadcast instrument** (route-3) — only ever a gated
  corroboration if route-1's residual `d` comes out ambiguous, which this design
  is structured to avoid. Carries the wedge risk and the larger build the review
  flagged; not pursued unless route-1 genuinely under-determines `d`.
- **Async DDR/NoC egress causal timing** — irreducibly non-deterministic;
  gap-only, never promised.
- **Migrating the hand-written coupling constants to data-driven extraction** —
  desirable for the derive-from-toolchain ethos, but a separate effort; this spec
  uses and tests the existing constants.

## Correctness principle

Reproduce what the silicon does by deriving the model from the toolchain and
measuring only the one physical constant the toolchain genuinely does not specify
(the broadcast per-hop delay `d`) — and measure it *through an over-determined
gate*, never fit it to a single target. The compute model is validated skew-free
in-domain; the on-chip coupling latencies are toolchain-derived and tested by
hop-diversity; the cross-domain skew falls out as the residual. The async DDR
boundary is gap-only by design. `Q = 0`, no statistical inference.
