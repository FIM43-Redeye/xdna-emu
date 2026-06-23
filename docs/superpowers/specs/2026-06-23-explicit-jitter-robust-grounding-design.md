# Explicit Jitter-Robust Grounding -- Design (measured rewrite)

**Status:** approved model (brainstorm + 2 HW spikes, 2026-06-23). Supersedes the
first draft of this file, whose "decompose at MM2S_FINISHED" premise the spike
proved backwards.
**Issue:** #140 (byte-identical emulator/HW trace reports), next tier after the
trace-experimenter-loop (Plan 4).
**Evidence:** `build/experiments/spike-jitter-grounding/FINDINGS.md` (+ spike.py,
spike2.py, logs).

## Goal

Replace the inference verifier's brittle statistical grounding rule
(`std <= eps` over cross-run offsets) with an **explicit, structural** rule that
extracts cycle-accurate data from the deterministic skeleton of a pipeline while
DMA-delivery jitter swirls around it. No statistical inference: no median, no
MAD, no outlier tolerance, no tuned epsilon.

This is the project's core purpose: "ground events even in the face of flaky DMA
and get cycle-accurate data on all the actually deterministic stuff."

## The measured model (this is the spec's foundation -- not assumed, measured)

Two HW spikes on add_one_using_dma (10 then 20 runs) established:

- **The deterministic, cycle-accurate unit is a segment bounded by milestone
  events WITHIN one timer domain.** The core compute segment
  `INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ` was **22 cycles in all 20
  runs, range 0** -- surviving every span outlier (a run whose shim span was
  3354, a run whose acquire fired at cycle 5029, a run whose span was 900).
- **Jitter is the WAIT, not the work.** The lock-acquire's absolute time scattered
  990..5029 cycles (delivery-gated). That is a gap to NAME, never a fixed count.
- **A within-domain offset that does NOT agree exactly is itself jitter.** The
  shim `MM2S_START -> S2MM_START` span (both shim, same domain) bundles
  issue + wait + compute; it scattered 900..3354. Same-domain is necessary but
  not sufficient -- exact cross-run agreement is the discriminator.
- **Cross-domain (cross-tile) timer skew is real.** One run's entire shim
  pipeline issued +3000 cycles late relative to the core PERF_CNT_2. Within-domain
  offsets cancel it; cross-domain offsets do not. Cross-domain grounding needs a
  synchronized timebase -- a LATER plan, out of scope here.
- **Observer effect:** heavy trace perturbs timing (light-trace span stable ~935;
  heavy-trace span scattered 900..3354). BUT the within-domain compute segment
  stayed exactly 22 even under heavy trace -- so **within-domain exact-agreement
  grounding is observer-effect-robust**. Low-frequency event selection is a
  refinement (deferred), not a correctness requirement.

## The problem being fixed

`tools/inference/verifier.py::correlates` conflates two jobs into one statistical
test: (1) does a causal edge exist + which way, (2) what is its offset. Both are
gated by cross-run `std <= eps=2.0`. That couples edge existence to offset
stability, so jitter -- which should only affect (2) -- destroys (1). The
trace-experimenter-loop (Plan 4) shipped a retry stopgap for this. This plan
removes the stopgap by fixing the rule.

## Architecture

Edge **existence + orientation** come from the static binary analysis already
built and HW-superset-verified (config_path / program_path facts). The runtime
trace's only job is **measuring offsets** along known edges, with a single
discriminating gate:

```
for a static edge (parent -> child):
  if parent and child are in the SAME timer domain (same tile)
     AND their per-run offset agrees EXACTLY across runs (range <= Q):
        -> DETERMINISTIC SEGMENT: ground the exact cycle offset
  else:
        -> NAMED GAP: keep existence + orientation only, no cycle count
```

The exact-agreement test is self-discriminating: it simultaneously classifies
(deterministic vs jitter) and verifies the magnitude. A within-domain edge that
bundles a wait fails exact agreement and correctly falls to a gap; the compute
segment passes and grounds. A through-core span is reported as
`gap + (exact segment) + gap`, never as one deterministic number.

## Dependency to verify first (feasibility gate)

The rule grounds within-domain milestone edges (e.g. core
`INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ`). For that to be groundable it
must arrive as an **oriented candidate pair** -- i.e. the static layer
(generator / route-graph / core_relay) must expose it, and both endpoints must be
in the configured + fired event set. The plan's first task verifies this against
`selfmodel.candidate_pairs_from_dump` + `generate_ledger` for add_one; if the
within-domain compute edge is not produced, exposing it (in the generator or as a
program_path entry) is part of this plan, not a surprise during implementation.

## Components (tight scope)

### 1. `tools/inference/grounding.py` (new)

- `same_domain(a, b) -> bool` -- a and b share a timer domain iff same tile
  (`col|row` equal); derived from the event keys, no new dump field.
- `ground_edge(run_dirs, child, parent) -> Grounding` -- returns one of:
  - `Segment(child, parent, offset)` if `same_domain` and the per-run offset
    range is `<= Q`;
  - `Gap(child, parent)` otherwise (cross-domain, or within-domain but
    non-exact = bundles jitter).
- `assemble(edges, groundings) -> Timeline` -- a static causal chain becomes a
  timeline of exact segments interleaved with named gaps.

### 2. `tools/inference/verifier.py` (rewritten primitives)

The `std <= eps` gate is removed. New primitives:

- `offset_exact(run_dirs, a, b) -> Optional[int]` -- the offset iff the per-run
  offset range is `<= Q` (Q=0; see below). Replaces `correlates`.
- `anchor_rigid(run_dirs, e) -> bool` -- e's anchored absolute time agrees
  exactly across runs (range `<= Q`). Replaces std-based `deterministic`.
- Falsifier triad (each returns `RejectedRule` on violation):
  - `check_ordering` -- `parent.ts <= child.ts` on every static edge, every run.
  - `check_additivity` -- `offset(a,c) == offset(a,b) + offset(b,c)` exactly,
    over a within-domain segment chain (vacuous where a domain has one segment).
  - `check_lock_handoff` -- `release.ts <= acquire.ts` on every LockPair edge.

### 3. `tools/inference/rules.py` (`try_derives` rewrite)

- Orientation: from the static config_path/program_path fact (unchanged).
- Grounding: from `grounding.ground_edge(...)`.
- Gate: exact-agreement (in `ground_edge`) AND the falsifier triad.
- Keep the existing shape (stochastic-parent, stable relative offset) -- the
  compute segment's endpoints are both absolute-jittery but exact-relative, which
  is exactly that shape -- but with EXACT agreement and within-domain restriction
  replacing `std <= eps`.
- Output: a `Segment` derive (exact offset) or a `Gap` derive
  (existence + orientation only).

### 4. Facts (`tools/inference/facts.py` + minimal consumers)

The `derives` Fact gains a `kind` discriminator (`segment` | `gap`) and, for
segments, the exact offset. Keep this MINIMAL -- a structured payload migration
is a deferred follow-on; this plan adds only what segment-vs-gap requires, with a
backward-compatible accessor so existing report fields keep working.

### 5. Report (`tools/inference/run_experiment.py::write_report`)

Report each derived edge as a segment (cycle-accurate offset) or a named gap, plus
any `RejectedRule`s. The determinism partition consumers (`mark_determinism`,
`classify_events`, `is_stochastic_root`, engine degeneracy, loop ranking) move
from std-based to `anchor_rigid` / exact-agreement -- this rewiring is in scope
because removing `std` breaks them otherwise.

## Q -- the measurement floor

**Q = 0, empirically confirmed.** The within-domain compute segment agreed
exactly (range 0) across 20 runs. Within-domain offsets share a timer domain, so
a deterministic relationship has zero cross-run delta. Q is documented as 0 (a
toolchain/measurement property, not a tuned value); if a future kernel exposes a
genuine discrete trace-frame quantum it is documented as that quantum, never a
value chosen to pass a test. (The earlier draft's decoder-coalescing
justification was wrong -- that artifact is already fixed; dropped.)

## Honest-failure handling (preserves the terminal-state model)

A within-domain edge expected deterministic that does NOT agree exactly is not
tolerated: it becomes a `Gap` (if it bundles a wait) or, if the design predicted
it deterministic and it isn't, surfaces as `halted_unexplained` (the existing
bug-signal terminal state). Terminal states unchanged: `placed`,
`halted_falsifiable`, `halted_unexplained`.

## Data flow

```
loop -> run_experiment -> engine (chainer -> rules -> grounding/verifier) -> report
```

Report gains per-edge segments (exact offsets), named gaps, and `RejectedRule`s.

## Testing

- **through-core HW** -> the Plan-4 retry stopgap is removed. New assertion: the
  within-domain compute segment grounds to an exact offset that agrees across the
  run set (range <= Q); the surrounding delivery/transit waits are reported as
  named gaps. No retry, no tolerance.
- **falsifiability HW** -> perturb structurally (break ordering / additivity /
  lock-handoff, or inject a per-run-varying offset that breaks exact agreement)
  and assert rejection.
- **offline units**: `same_domain`, `offset_exact` (range<=Q), `anchor_rigid`,
  the falsifier triad (each catches its violation), `ground_edge`
  (segment vs gap), `assemble` (segments + gaps), facts segment/gap round-trip.
- **suite** (add_one_using_dma + add_one_objFifo + vector_scalar_using_dma)
  re-verified under the new grounding; zero regressions; each reaches `placed`
  or `halted_falsifiable`.

## Out of scope (deferred, named follow-ons -- one serious thing at a time)

- **Cross-domain timer-sync** (BROADCAST_15 timer-reset) for cross-tile grounding.
  The biggest deferred piece; its own plan.
- **Active event selection / observer-effect handling** -- the solver preferring
  low-frequency milestone events and scheduling segment brackets. A refinement
  (within-domain grounding is already observer-effect-robust), its own plan.
- **Full structured facts-schema migration** beyond the minimal segment/gap kind.
- New trace-event-menu types; multi-column; the Plan-4 folds.

## Correctness principle

Domain classification is the event key (`col|row`); the deterministic/jitter
discrimination is the measured exact-agreement test; Q is measured. Nothing is
hardcoded to a kernel. DERIVE FROM THE TOOLCHAIN holds throughout.
