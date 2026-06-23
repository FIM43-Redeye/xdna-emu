# Explicit Jitter-Robust Grounding -- Design

**Status:** approved (brainstorm 2026-06-23)
**Issue:** #140 (byte-identical emulator/HW trace reports), next tier after the
trace-experimenter-loop (Plan 4).
**Predecessor:** `docs/superpowers/specs/2026-06-22-trace-experimenter-loop-design.md`

## Goal

Replace the inference verifier's brittle statistical grounding rule
(`std <= eps` over cross-run offsets) with an **explicit, structural** grounding
rule that grounds the deterministic skeleton of a pipeline *through* DMA-delivery
jitter -- cycle-accurate on everything deterministic, with DMA gaps named (not
numbered) rather than tolerated. No statistical inference: no median, no MAD, no
outlier tolerance, no tuned epsilon.

This is the project's core purpose stated plainly: "ground events even in the
face of flaky DMA and get cycle-accurate data on all the actually deterministic
stuff."

## The problem

The current grounding primitive (`tools/inference/verifier.py::correlates`)
conflates two distinct jobs into one statistical test:

1. **Does a causal edge exist, and which way does it point?**
2. **What is the cycle offset along it?**

Today both are answered by one gate: the cross-run offset's standard deviation
must be `<= eps` (2.0 cycles). That couples edge *existence* to offset
*stability*, so DMA jitter -- which should only perturb (2) -- destroys (1).

The concrete failure is the through-core pair
`DMA_S2MM_0_START_TASK <- DMA_MM2S_0_START_TASK`. The `core_relay` edge collapses
input -> core -> output into a **single edge that spans the whole pipeline**, and
that span contains the one genuinely non-deterministic thing in the system:
**input DMA delivery from DDR**. Usually delivery is ~0-jitter (std < 1), but an
occasional DDR/NoC hiccup pushes one run's offset past `eps=2.0` and the entire
derive is rejected -- even though the compute-and-output portion of that span was
perfectly deterministic on that run.

The trace-experimenter-loop (Plan 4) shipped with a retry-tolerant stopgap for
this. This plan removes the stopgap by fixing the grounding rule.

## Where jitter actually lives

Jitter is not smeared across the pipeline. It is confined to one segment that the
static config identifies explicitly:

```
MM2S_0_START --[DMA delivery: JITTER]--> MM2S_0_FINISHED --[compute+output: DETERMINISTIC]--> S2MM_0_START
 (input start)                            (input landed)                                       (output start)
```

Everything after input lands is deterministic and grounds to an exact cycle count
from a single run. The jitter is one config-known segment gated by a DMA-delivery
boundary -- a thing to *name*, not a statistic to tolerate.

## Architecture

**Separate the two jobs.**

- **Edge existence + orientation** come from the static binary analysis already
  built and HW-superset-verified (config_path / program_path edges:
  `DmaBufferRelay`, `LockPair`, `CoreLockRelay`). Explicit, no statistics.
- **Offset measurement** is the runtime trace's only job: read offsets along
  *known* edges. Grounding is gated by **edge classification** and an **explicit
  falsifier**, never a tolerance band.

**Both layers** (the chosen treatment for a jitter-spanning edge):

```
if edge crosses a known jitter boundary:
    decompose -> exact deterministic segments + named jitter gap
else:
    edge-trust -> measured offset, falsified only by structural checks
```

**The self-model knows what each event provides.** Decomposition is *active*, not
opportunistic: when a jitter-spanning edge is a candidate, the self-model marks
its bracketing events (e.g. `MM2S_0_FINISHED`) as **required** for that edge's
decomposition, so the planner schedules them into a measure-next batch and then
assembles event combos across batches into one timeline. Deterministic segments
are cycle-accurate; DMA gaps are marked as gaps.

## Components

### 1. `tools/inference/grounding.py` (new)

The explicit grounding rule.

- `classify_edge(edge, dump) -> EdgeClass` -- `DETERMINISTIC` or
  `JITTER_SPANNING`. An edge is jitter-spanning iff its static path crosses a
  DDR-sourced DMA delivery: a shim/memtile MM2S sourced from an external buffer
  whose completion gates a downstream lock (read off the route graph +
  `bd_configs`). `CoreLockRelay` through-core edges are inherently
  jitter-spanning; intra-tile buffer relays (`DmaBufferRelay` within a tile) are
  deterministic.
- `boundary_events(edge, dump) -> (gap_start, gap_end)` -- for a jitter-spanning
  edge, the config-known bracket: the input DMA's `..._START_TASK` (gap start)
  and `..._FINISHED_TASK` (gap end = delivery complete). The post-delivery event
  re-anchors the deterministic remainder.
- `required_events(edge, dump) -> List[str]` -- the bracketing events a
  jitter-spanning edge needs traced in order to decompose. Consumed by the
  self-model/planner so decomposition is scheduled, not hoped for.
- `ground(run_dirs, child, parent, edge, dump) -> Grounding` -- returns a
  structured result: **deterministic segments** `[(a, b, offset)]` (exact
  offsets) and **named jitter gaps** `[(start, end)]` (existence + orientation
  only, explicitly *not* a cycle count). For a deterministic edge this is one
  segment and no gap; for a jitter-spanning edge with brackets traced, it is the
  deterministic remainder segment(s) plus the named gap; for a jitter-spanning
  edge whose brackets did not fire, it falls back to edge-trust (one segment, the
  span, flagged as containing an unbracketed gap).

### 2. `tools/inference/verifier.py` (rewritten primitives)

The std gate is removed from `correlates` / `deterministic` / `coincident`.

- `offset_exact(run_dirs, a, b) -> Optional[int]` -- the offset iff the cross-run
  delta is `<= Q` (the derived measurement floor; see below). A single run
  trivially returns the measured value. Replaces std-based `correlates`.
- Falsifier triad (each returns `RejectedRule` on violation, else `None`):
  - `check_ordering(run_dirs, parent, child)` -- `parent.ts <= child.ts` on every
    static edge, every run.
  - `check_additivity(run_dirs, a, b, c)` -- `offset(a,c) == offset(a,b) +
    offset(b,c)` exactly, along deterministic segments.
  - `check_lock_handoff(run_dirs, release, acquire)` -- `release.ts <=
    acquire.ts` on every `LockPair` edge, every run.
- `deterministic` / `coincident` re-expressed via `offset_exact` (delta `<= Q`),
  not std.

### 3. `tools/inference/rules.py` (`try_derives` rewrite)

- Orientation: from the static config_path/program_path edge (unchanged).
- Offset: from `grounding.ground(...)` rather than `correlates`.
- Gate: the falsifier triad. No std, no eps tolerance.
- Output: a structured `derives` Fact (segments + jitter_gaps) for a
  jitter-spanning edge; a single-segment `derives` Fact for a deterministic edge.

### 4. Facts schema (`tools/inference/facts.py` + consumers)

The `derives` Fact payload becomes structured:
`segments: [(a, b, offset)]` + `jitter_gaps: [(start, end)]`. Engine
(`engine.py`) and report (`run_experiment.py::write_report`) consumers report
segments cycle-accurately and gaps by name. Backward-compatible accessor for the
common "single offset" case so existing report fields keep working.

### 5. Self-model / planner decomposition-awareness (`tools/inference/selfmodel.py`, `loop.py`)

- `selfmodel` consults `grounding.required_events(edge, dump)` so that when a
  jitter-spanning candidate edge is present, its bracketing events are surfaced
  as high-priority measure-next targets (within the `<= 8`-slot legality the
  planner already enforces).
- The loop assembles event combos across batches into one timeline; a derive that
  needs a bracket not yet traced schedules that bracket rather than failing.

## Q -- the measurement floor

Derived structurally, not measured statistically. The AIE2 trace timer is
cycle-accurate, and a truly deterministic same-execution offset has *constant*
per-tile timer skew -- so its cross-run delta is exactly 0. The only nonzero
source is the decoder's same-frame coalescing artifact (the
`test_trace_decoder.py` "8 -> 76 cycle span" inflation), which is fixed **in the
decoder**, never by widening Q.

**Therefore Q derives to 0** (literal cross-run equality), documented with that
justification. If characterization of the decoder/trace-timer semantics reveals a
genuine fixed quantization, Q absorbs exactly that value and is documented as a
measurement-resolution constant. Q is never a knob tuned to make a test pass.

A research/characterization step confirms Q from decoder + trace-timer semantics
before the gate depends on it.

## Honest-failure handling (preserves the terminal-state model)

A segment classified `DETERMINISTIC` whose measured cross-run delta exceeds Q is a
**contradiction**, never tolerated:

- If a jitter boundary is found on its static path -> reclassify as
  `JITTER_SPANNING` and decompose.
- Otherwise -> surface as `halted_unexplained` (the existing bug-signal terminal
  state).

This is what keeps "no statistics" from collapsing into "trust whatever the trace
says." The terminal states are unchanged: `placed` (full placement),
`halted_falsifiable` (honest halt with provenance constraints), `halted_unexplained`
(bug signal).

## Data flow

```
loop -> run_experiment -> engine (chainer -> rules -> grounding/verifier) -> report
```

The report gains: per-edge deterministic segments (cycle-accurate offsets), named
jitter gaps, and any `RejectedRule`s from the falsifier triad.

## Testing

- **through-core HW** -> reverts to a **single-run** assertion: the post-delivery
  deterministic segment grounds to an exact offset; the input-delivery gap is
  named, not numbered. Replaces the Plan-4 retry stopgap.
- **falsifiability HW** -> rewritten to perturb *structurally* (break ordering /
  additivity / lock-handoff) and assert rejection. No longer std-dependent.
- **offline units**: `classify_edge` (deterministic vs jitter-spanning),
  `boundary_events`, `required_events`, `offset_exact` (delta <= Q, single-run),
  the falsifier triad (each catches its violation), `ground` decomposition
  (segments + gap), facts-schema round-trip.
- **suite** (add_one_using_dma + add_one_objFifo + vector_scalar_using_dma)
  re-verified under the new grounding; zero regressions; `placed` or
  `halted_falsifiable` for each.

## Out of scope (one reasonable thing per plan)

- New trace-event-menu types beyond what's needed to bracket the add_one-class
  jitter boundary (full per-device event tables remain a follow-up).
- Per-tile timer-skew modeling beyond the constant-skew assumption Q rests on.
- Multi-column tracing.
- The deferred folds (groups/Z3, per-tile mode) from the Plan-4 design.

## Correctness principle

Boundary detection, event semantics, and Q all DERIVE FROM THE TOOLCHAIN
(route graph + `bd_configs` for boundaries, regdb/trace-unit for event semantics
and Q). Nothing about the jitter boundary or the measurement floor is hardcoded
to a kernel.
