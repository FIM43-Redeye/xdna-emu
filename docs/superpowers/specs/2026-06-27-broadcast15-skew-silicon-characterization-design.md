# BROADCAST_15 Trace-Timer Skew: Silicon Characterization (Measure-First) — Design

**Status:** approved design (brainstorm + 3-lens adversarial review of the
*emulator* approach, 2026-06-27). This spec is the **measure-first** first
sub-project of the timer-sync arc. It deliberately precedes any emulator code.
**Issue:** #140 (byte-identical emulator/HW trace reports).
**Read first:** [`../../trace/cross-domain-skew-limit.md`](../../trace/cross-domain-skew-limit.md)
(the epistemic boundary) and
[`../../trace/capture-load-sensitivity.md`](../../trace/capture-load-sensitivity.md)
(host-load contamination of trace timing).
**Prior art landed:** the engine-side reproduction-target annotation
(`tools/inference/grounding.py`: `reproduction_offset` + `is_async_cdc`) is
already merged; this spec does not touch it.

## Why this sub-project exists (the inversion)

The original plan was emulator-first: give the emulator a per-hop broadcast
flood model, fit two constants to add_one's measured skew, and assert
`emu raw offset == hw raw offset`. A three-lens adversarial review killed that
plan's *validation*, not its mechanism:

- **The cross-domain gate was vacuous by construction.** add_one's three
  characterized pairs span only **3 modules** (core, memmod, memtile) → **2
  independent** skews. Against a **2-parameter** model (per-hop delay +
  intra-tile offset) that is an exact fit; the gate cannot fail. The "additive
  consistency" cited as physical evidence —
  `(core−memtile) − (memmod−memtile) ≡ core−memmod` — is an arithmetic
  tautology, true for any three numbers.
- **The only falsifiable gate (in-domain) never touches the broadcast code.**
  Within a domain `skew = 0` by definition, so the in-domain segment-reproduction
  gate validates the *compute* model and is blind to the broadcast model.
- Net: the emulator-first plan would have shipped broadcast code with **no
  falsifiable test of that code** — the same failure mode that falsified the two
  prior attempts at this (see the skew-limit doc's banner).

The review's converging fix: falsifiability needs **the shim (a 4th module) and
≥2 hops sharing one per-hop constant**, so a uniform-per-hop model makes a
prediction that can fail. We have never actually measured whether per-hop delay
is uniform — we measured two numbers and have two knobs.

So we invert the order: **characterize the skew on silicon first.** Cheap HW,
honest order, and it converts the eventual emulator gate from theater into a
real over-determined test. The emulator mechanism (which the review also showed
is *smaller* than first designed — the flood already exists in
`propagate_broadcasts` with correct per-module IDs) becomes a follow-on,
informed by a measured per-hop delay.

## The epistemic constraint this design respects

Per the skew-limit doc §6: **a trace in isolation cannot split the recorded
cross-module offset `Δwall + skew` into its parts.** Three independent walls
forbid it. This sub-project does not pretend otherwise. It is structured in two
stages with explicitly different epistemic status:

- **Stage 1 (passive)** records raw offsets (`Δwall + skew`, fused) as an
  *over-determined reproduction target*. It does **not** isolate skew. Its
  falsifiable content is limited to cross-run/structural *consistency*.
- **Stage 2 (active)** uses the skew-limit doc's **route-3** instrument (§8): a
  second probe broadcast on a *different path* than the BROADCAST_15 sync, whose
  arrival is **traced, not reset**. Because the probe's path differs from the
  sync's, the per-hop delays do **not** cancel (wall #2 bites only a *same-path*
  second flood), and the differential arrival pattern is linear in the per-hop
  delay `d` → `d` is solved directly on silicon, emulator-independent.

## Goal and success criteria

**Goal:** an over-determined, deterministic silicon characterization of the
AIE2 (NPU1 / Phoenix) trace-timer broadcast skew structure, sufficient to (a)
give the eventual emulator a non-degenerate reproduction target and (b) yield a
silicon-measured per-hop flood delay `d` with a uniform-per-hop
consistency/falsification verdict.

**Success:**
1. Stage 1: a documented raw-offset table covering **all four column-0 modules**
   (shim, memtile, compute core, compute memmod), each with cross-run range,
   captured on a verified-quiet host, with a range-0 determinism verdict.
2. Stage 2: a silicon-measured per-hop delay `d` (and intra-tile core/mem offset
   where measurable), with an explicit verdict on whether per-hop delay is
   uniform across the measured hops. A non-uniform result is a *successful*
   outcome — it falsifies the uniform-per-hop hypothesis and records the true
   structure.
3. No constant in either stage is tuned to hit a target; `d` is solved from the
   measurement.

## Stage 1 — Passive corpus

### What it does

Records each traced module's anchor-relative trace-timer offset for add_one,
across runs, including the shim. The shim is the load-bearing addition: it is the
4th module that breaks the 3-module/2-knob degeneracy.

### Steps

1. **Mine the existing 20-run capture** (`build/experiments/gap140/`
   `nondeterminism/add_one_using_dma/`, runs 00–19) offline for whatever
   per-module coverage it already holds. Determine which of {shim(0,0),
   memtile(0,1), core(0,2), memmod(0,2)} already have anchor-relative offsets and
   their cross-run range. (memmod fires only lock events on add_one — see the
   compute-memmod finding — but those are deterministic and *anchor-relative
   offsets do not require connectivity orientation*, so they are usable here.)
2. **Capture fresh to fill gaps**, on a verified-quiet host (see contamination
   guard below), so all four modules are covered with ≥ the existing run count.
   Reuse the existing capture/decode pipeline; no new instrument in Stage 1.
3. **Assert determinism:** each module's offset must agree across runs to range 0
   (`Q=0`, the shared comparator). A non-zero range is a contamination flag, not
   an averageable measurement — diagnose host load before proceeding.

### Deliverable

A raw-offset table `module → (anchor-relative offset, cross-run range)` plus a
determinism verdict, written to the experiment output tree and summarized in a
findings doc. **Explicitly labeled:** raw offsets are `Δwall + skew` fused; this
stage does not isolate skew. Its purpose is the over-determined target set and a
clean-determinism baseline.

### What Stage 1 can and cannot falsify

- **Can:** cross-run determinism (a non-zero range falsifies "the skew structure
  is fixed" *or* flags contamination); gross structural sanity (e.g. monotonic
  north-flood ordering of origins).
- **Cannot:** the per-hop-uniformity hypothesis, nor any skew value in isolation.
  That is Stage 2.

## Stage 2 — Active probe-broadcast (pins `d`)

### What it does

After the standard BROADCAST_15 sync establishes a common timer zero, inject a
**second probe broadcast** from a *different origin and/or direction* than the
sync, configured so each module **traces its arrival** (the probe event is in the
module's trace event set) **without** resetting on it (the probe event is *not*
the module's `Timer_Control.Reset_Event`). Each module records the probe's
arrival cycle in its (ch15-synced) timer.

### Why this isolates `d`

Let the BROADCAST_15 sync flood from the shim (col 0, row 0) northward; module
D's origin is `origin_D = d · hops(shim→D) + intra(D)`. Let the probe flood from
a different origin P (e.g. the column top, flooding south, or an adjacent
column). Module D records the probe arrival as
`rec_D = W(probe@D) − origin_D = [d · hops(P→D) + intra'(D)] − [d · hops(shim→D)
+ intra(D)]`. Differencing `rec_D` across modules along the flood axis yields
equations **linear in `d`**, with the `hops(P→D)` and `hops(shim→D)` terms
carrying **opposite signs** for opposing flood directions — so `d` is
recoverable from the slope of `rec_D` vs module position, independent of the
absolute `intra` terms and independent of any emulator. (This is the skew-limit
doc's route-3 fallback, here used as the primary `d` instrument; it does **not**
violate wall #2, which forbids only a *same-path* second flood whose delay
cancels the sync's.)

### Steps

1. **Derive the probe config from the toolchain.** The probe broadcast reuses the
   same broadcast-channel + event-generate register mechanism the sync uses
   (the registers `seed_broadcasts_for_event` / `propagate_broadcasts` already
   read). Channel and event IDs are looked up per module from the arch/regdb —
   **no hardcoded 15 / 122 / 157.** Choose a free broadcast channel and a free
   USER_EVENT for the probe; configure each traced module's trace event set to
   include the probe's per-module broadcast event; leave `Reset_Event` =
   BROADCAST_15 untouched.
2. **Generate the custom `patch_spec`** (the one new instrument) that lays down:
   the probe channel register at the chosen origin, the probe event-generate, and
   each module's trace-event-set addition for the probe arrival.
3. **Capture on real NPU1**, quiet host, ≥N runs; decode with the existing
   pipeline.
4. **Solve for `d`** from the slope of recorded probe-arrival vs module position
   along the flood axis. Repeat with at least two probe origins/directions to
   over-constrain and check sign behavior.
5. **Verdict:** is `d` consistent across modules/hops and across probe
   configurations? Record the value(s) and the uniform-vs-non-uniform conclusion.
   Where the column also exposes the intra-tile core/mem split, report it.

### Deliverable

A silicon-measured per-hop delay `d` (+ intra-tile offset where measurable) with
a consistency/falsification verdict, in the findings doc. `d` is **solved**, not
fit.

### What Stage 2 can falsify

The uniform-per-hop hypothesis directly: if the recovered `d` is not consistent
across modules/hops or across probe configurations, uniform-per-hop is false on
silicon and the true structure (per-hop variation, larger intra-tile asymmetry,
edge effects) is recorded instead. Either outcome is a successful measurement.

## Components & tooling

- **Stage 1:** analysis over the existing capture/decode pipeline
  (`tools/trace_capture.py`, `configure_batch`, upstream `parse_trace`, the
  `trace_join` offset math). No new instrument; a small analysis script that
  tabulates per-module anchor-relative offsets + cross-run range.
- **Stage 2 probe-config generator** (the only genuinely new code): emits the
  custom `patch_spec` for the probe broadcast, with all channel/event IDs derived
  per module from the arch/regdb. Lives under `tools/` alongside the existing
  trace-config tooling. Unit-tested offline (the patch it emits is deterministic
  given a chosen channel/event and origin), with an HW capture as the acceptance.
- **Solver:** a small offline routine that fits `d` to the slope of recorded
  probe-arrival vs position. "Fit" here is a *measurement* (least-squares slope of
  a directly-observed linear relation), not a tuning of the model to a target;
  the residual of the linear fit is itself the uniform-per-hop test.

## Honesty fences

- **Quiet-host precondition** (capture-load-sensitivity doc): assert range-0
  before trusting any offset; a non-zero range is contamination to diagnose, not
  to average. No autonomous load classifier — flag for human classification.
- **No tuning:** `d` is solved from the probe measurement; no constant is
  adjusted to make a target match. Per-hop delay is documented as
  silicon-measured (source-derivation policy #2: hardware observation), not
  toolchain-specified (the IRON docs explicitly say the per-hop delay is not in
  the toolchain).
- **Epistemic labels:** Stage 1 offsets are `Δwall + skew`, never presented as
  isolated skew. Only Stage 2 isolates `d`.
- **Derive IDs from the toolchain:** every broadcast channel and event ID
  (sync and probe) is resolved per module from the arch/regdb, never hardcoded —
  this is exactly the indexed-mapping bug class (memtile BROADCAST_15 = 157, not
  122; channel dynamically allocated) the adversarial review flagged.

## Out of scope (named follow-ons)

- **The emulator mechanism** — per-hop-delay scheduling in the existing
  `propagate_broadcasts` path; the trace-origin reconciliation (the trace Start
  frame must read the flood-reset per-module timer, not absolute cycle — note the
  trace unit's own `timer` field is dead code and the real coupling is the global
  `cycle` argument); the clock-gating-vs-latched-reset ordering hazard; the shim
  own-timer self-reset bypass. A separate sub-project, now informed by a measured
  `d` and an over-determined target.
- **Cross-column horizontal characterization** — Stage 2 may extend to a second
  column (horizontal hops) to over-constrain `d` further, or this may be its own
  follow-on once single-column `d` is in hand.
- **Any emulator validation gate** — deferred to the emulator sub-project, where
  the over-determined target from this work makes it falsifiable.

## Correctness principle

Measure what silicon does before modeling it. The trace cannot split `Δwall`
from `skew` (skew-limit doc), so Stage 1 honestly records the fused sum as a
reproduction target and Stage 2 uses an active, different-path probe to solve the
per-hop delay directly on hardware. Every broadcast channel/event ID is derived
from the toolchain per module; the one physical constant the toolchain does not
specify (per-hop delay) is measured from silicon, not fit to a target. No
statistical inference, `Q = 0`.
