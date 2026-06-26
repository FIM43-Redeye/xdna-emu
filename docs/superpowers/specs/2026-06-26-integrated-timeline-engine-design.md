# Integrated Timeline Engine — Design

**Date:** 2026-06-26
**Status:** Draft (design); pending adversarial review then implementation plan
**Issue:** #140 (true-accuracy arc)

## Motivation

The inference engine's purpose, stated plainly: **feed it a traced kernel and it
produces, automatically, a fully integrated all-tiles timeline in which every
event and every gap is accounted for** — deterministic events characterized
cycle-by-cycle, nondeterministic events bounded and ordered exactly as much as
the silicon allows, nothing silently dropped. That timeline is the ground-truth
specification the emulator must replicate: we cannot say the emulator diverges on
an event whose *hardware* value itself wiggles, so we must first characterize the
hardware's own determinism, honestly and completely.

The plumbing toward that vision is partly built and tested, but the vision itself
is not delivered. Today:

- **Per-edge grounding works.** `ground_edge(child, parent)` (grounding.py)
  classifies a single causal edge as a cycle-exact within-domain `Segment`, or a
  typed `Gap` (`within_domain_nonexact` WARN / `cross_domain` / `async_cdc`). The
  contamination canary, determinism detection (`anchor_rigid`, Q=0), and the
  trace-slot sweep that captures >8 events/tile all function and are tested.
- **There is no composition.** `assemble()`/`Timeline` exist in grounding.py but
  are never called. `run_engine` returns flat lists (`segments`, `gaps`,
  `derives`) — a bag of independent edges, not a timeline. Each event is placed
  only relative to its immediate parent.
- **The heart is missing.** "Deterministic events ground nondeterministic ones"
  requires walking from a nondeterministic event to a deterministic reference and
  expressing its position relative to that reference, bounded by a typed gap. That
  composition does not exist; `reproduction_offset` is a raw number, not anchored.
- **Only single-column has ever run end-to-end.** `add_one_using_dma` (one column)
  is the only kernel run through the full path. No multi-column fixture exists, so
  cross-tile / cross-column integration is untested.

## Goal / Definition of Done

`run_engine` emits an `IntegratedTimeline` for a kernel: an ordered structure of
**deterministic periods** (each a rigid frame recorded cycle-by-cycle in its own
coordinates) and **nondeterministic periods** (each an order-relation of
windowed events, closed by a grounding event), covering every configured event on
every traced tile, with every gap typed and every unaccounted weirdness flagged.
Validated end-to-end on a **multi-column** kernel (`two_col`), not just
single-column.

"Accounted for" means each event lands in exactly one of: a deterministic period
(exact local cycle), a nondeterministic period (window + order/concurrency +
typed reason), or the intermittent bucket (appearance set). No event is silently
absent; no order is asserted that the runs contradict; no absolute cycle is
claimed across a nondeterministic gap.

## Non-Goals

- **Visualization.** Emitting a Perfetto/GUI rendering of the timeline is out of
  scope. The deliverable is the structured `IntegratedTimeline` plus a plain-text
  renderer for tests and eyeballing. We already have trace-viz paths; wiring this
  model into them is future work.
- **Emulator comparison.** This characterizes *hardware* determinism (HW-vs-HW).
  Comparing the emulator against this ground truth is the next arc, not this one.
- **Fixing divergences / new captures campaign.** This builds the engine. Running
  it across the whole corpus to produce the determinism map is the subsequent
  step, scoped from this engine.
- **Autonomous load-vs-HW classification.** A range>0 within-domain span is still
  verified manually (re-capture on a quiet host). The engine flags; a human
  classifies. (Unchanged from the canary work.)

## Conceptual Model

**One global anchor.** Every event has an `anchored_ts` per run, measured against a
single anchor event (`1|2|0|PERF_CNT_2`) that sits in every trace batch by design.
This is the measurement reference and the zero of the first deterministic period.

**The timeline is an alternating sequence of periods.**

```
local cycle: 0    24   48                          0     12    40
             |----|----|·····?·····?·····|          |-----|-----|
             [ DETERMINISTIC (frame A) ] [ NONDET ] [ DETERMINISTIC (frame C) ]
             anchor a    b      c    d   e→grounds  f     g     h
             exact cycles       windows in A's clock  exact cycles, RE-ZEROED
```

**Deterministic periods re-zero.** You cannot carry an absolute cycle across a
nondeterministic period, because the gap's *duration* is exactly what varies. So
each deterministic period is its own rigid frame with cycle 0 at its grounding
event. Period A is anchored at the global zero; every later deterministic period
re-zeroes. What lives between frames is a nondeterministic period whose **windows
are measured in the upstream frame's clock** (the last trusted reference) — the
window *is* the honest statement of the inter-frame uncertainty. We never assert
an absolute cycle for where a downstream frame sits.

**Determinism is local, which is strictly more capable.** "Deterministic" means
range-0 relative to *this period's* reference, not the one global anchor. Hardware
re-synchronizes after jittery stretches (locks, barriers); those events are rigid
relative to each other even though their offset to the global anchor inherited the
gap's smear. Local re-anchoring recovers them as a clean deterministic period;
global-only detection would mislabel them nondeterministic.

**Represent things only as weird as they are.** Deterministic → exact cycles.
Orderable-but-jittery → ordered windows. Genuinely racing → marked concurrent, no
fabricated sequence. Fires-only-sometimes → intermittent, with its appearance set.
The representation's fidelity matches the behavior's fidelity — never more, never
less.

## Data Model

Per-event classification, computed from the event's `anchored_ts` vector across
the N runs:

- **deterministic** — belongs to a rigid cluster (below); carries an exact local
  `cycle` within its period.
- **nondeterministic** — rigid to nothing; carries a `window = [min, max]`
  (offset to the upstream frame's reference, across runs) and a `reason`
  (`within_domain_nonexact` / `cross_domain` / `async_cdc`).
- **intermittent** — not present in every run; carries the set of runs it appeared
  in. Never forced onto the skeleton.

Period types:

- **`DeterministicPeriod`** — an ordered list of events of one rigid frame, each
  with its exact local `cycle` (zero at the frame's earliest event = the grounding
  event). Inter-event gaps are exact (both endpoints rigid).
- **`NondeterministicPeriod`** — the events between two deterministic frames. Holds
  each event with its `window` + `reason`, plus an **order relation**: a set of
  `order_edges`, each tagged `causal` (from the derives graph) or `stable_position`
  (a < b in *every* run). Pairs connected by no edge are **concurrent**. Closed by
  a **`grounding_event`** = the first event of the next deterministic frame.

Order is stored as edges, not a flattened list, so a consumer can layer it into
"rungs" (each rung a set of mutually-concurrent events); the rung shape *shows*
how much order was earned. All singletons = fully ordered; one fat rung = total
race.

`IntegratedTimeline` — the ordered list of periods, plus the intermittent bucket
and a list of honesty flags (below).

**Honesty flags (carried, never swallowed):**

- `ungrounded_tail` — a trailing nondeterministic period where determinism never
  resumes: `grounding_event = None`.
- `overlaps_frame` — a nondeterministic event whose window overlaps a deterministic
  frame instead of slotting cleanly between frames: attached as concurrent-with-
  frame, not given a forced position.
- intermittent events — surfaced in the bucket with their appearance set.

## The Algorithm

Inputs: N clean run dirs (HW captures of one kernel), the configured-event set,
the causal `derives` graph (already built by the engine), the global anchor.

1. **Jitter-vectors.** For each configured event, collect its `anchored_ts` in each
   run (existing `_first_firsts` / `batch_firsts` path). Present in all N runs →
   full vector. Present in a subset → **intermittent**, set aside with its
   appearance set.

2. **Rigid clusters (the core).** For each full-vector event, compute its
   *jitter-vector* = `anchored_ts - anchored_ts[run 0]`. Rigid linkage (mutual
   offset range-0 every run) holds **iff two events' jitter-vectors are identical**
   — exact integer equality, Q=0, no tolerance. Group events by jitter-vector:
   - the all-zero group (contains the anchor) → Period A's frame;
   - any shared non-zero group → a re-synchronized frame;
   - a unique non-zero jitter-vector → a nondeterministic event.

   Rigid linkage is transitive (`offset(a,c) = offset(a,b) + offset(b,c)`, both
   constant ⇒ constant), so this grouping is a true equivalence partition. Cluster
   confidence scales with N: coincidental jitter-vector agreement among unrelated
   events becomes vanishingly unlikely as runs accumulate — never tuned, always
   re-verifiable by more captures.

3. **Internal cycles.** Within each cluster, local zero = earliest event; each
   other event's local `cycle` = its exact pairwise offset to that zero. Guard with
   `check_additivity` (sum-consistency of offsets within the frame).

4. **Sequence & build periods.** Order all clusters and nondeterministic events by
   mean `anchored_ts` — a *sequencing key only*, never a reported cycle. Walk
   left→right: a contiguous block of same-cluster events → a `DeterministicPeriod`;
   a run of nondeterministic events between two frames → a `NondeterministicPeriod`,
   windows measured against the upstream frame's reference, grounding event = the
   first event of the next frame.

5. **Intra-period order.** Inside each nondeterministic period, build `order_edges`
   from (a) causal `derives` edges among its events and (b) strictly-stable
   cross-run position (a < b in every run). Pairs with neither → concurrent.

6. **Honesty flags.** Emit `ungrounded_tail`, `overlaps_frame`, and the intermittent
   bucket as above.

**Determinism detection moves** from the single `anchor_rigid`-vs-global call to
jitter-vector equivalence. Same Q=0 primitive; it is what unlocks re-sync frames.
The existing `anchor_rigid` (range-0 vs global anchor) becomes the special case
"jitter-vector is all-zero."

## Components / Where It Lives

- **`tools/inference/timeline.py` (new).** The period data model
  (`DeterministicPeriod`, `NondeterministicPeriod`, `IntegratedTimeline`, the event
  records and honesty flags) and `assemble_timeline(run_dirs, derives, configured,
  anchor)`. This is the home of the segmentation algorithm.
- **`tools/inference/verifier.py` (extend).** Add a window helper
  (`offset_window` / per-run anchored_ts vector accessor) beside `offset_exact`,
  reusing the same `tj.batch_firsts` path so the timeline and the edge grounder
  agree on every measured number.
- **`tools/inference/engine.py` (wire).** Call `assemble_timeline` over the final
  run dirs and add `timeline` to the report dict, alongside the existing
  `segments` / `gaps` / `warnings`.
- **`tools/inference/run_experiment.py` (propagate).** Thread `timeline` through the
  returned report (best-effort, like the existing engine block).
- **`tools/inference/grounding.py` (retire dead code).** The unused
  `Timeline` / `assemble` (edge-list-in-chain-order) are superseded by the period
  model; remove them rather than leave two `Timeline` types. `ground_edge`, `Gap`,
  `Segment`, and the reason constants stay — the timeline reuses the typed reasons.
- **A plain-text renderer** (`render_timeline`) producing the A→B→C view with local
  cycles, windows, concurrency, and flags — used by tests and for eyeballing.

## Validation — Multi-Column

The single biggest untested risk is cross-column integration. Validation:

- **Generate a `two_col` config fixture** via the existing `config_extract` path
  (the tool that produced `add_one_using_dma.config.json`). `two_col` spans two
  columns, so its events live on multiple tiles and get placed on the one global
  anchor axis — exercising cross-tile placement and cross-domain gaps inside
  periods.
- **Offline, TDD.** Synthetic run-dir fixtures with hand-constructed
  deterministic / re-sync / nondeterministic / intermittent / concurrent /
  overlaps-frame / ungrounded-tail patterns — assert the exact period structure,
  local cycles, windows, order edges, concurrency, grounding events, and flags.
  Include a multi-column synthetic fixture (events across two columns).
- **HW-gated, end-to-end.** Run `two_col` through `run_experiment` on the real NPU
  (controller-run, witness-gated for clean capture), assert a well-formed
  `IntegratedTimeline` with the expected frame count and no swallowed events.

## Testing Strategy

Pure Python (`tools/`); `cargo test --lib` not required. TDD throughout. Offline
unit coverage for: the window helper; jitter-vector computation; rigid-cluster
partition (incl. re-sync frame, singleton nondeterministic, coincidental-agreement
note); internal-cycle assignment with additivity guard; sequencing; period
construction incl. boundaries and grounding events; intra-period ordering with
causal + stable-position edges and concurrent-by-omission; every honesty flag;
multi-column placement; full `assemble_timeline` on multi-period synthetic
fixtures. HW-gated E2E on `two_col` is controller-run, gated like existing HW
tests, never in the offline suite.

## Risks / Open Questions for Review

The adversarial review must probe, at minimum:

1. **Does jitter-vector equality actually capture rigid linkage on real data?**
   Co-tracing and the slot sweep mean not all events appear in the same batch; the
   anchored_ts is each event's first-co-tracing-batch value. Is that vector
   coherent enough that equality means rigidity, or does batch selection inject
   spurious differences?
2. **Coincidental clusters at small N.** Two unrelated events with equal jitter
   over few runs. Is the N-confidence argument sufficient, or do we need a minimum
   N or a guard?
3. **Sequencing by mean position.** Could the mean key misorder periods when
   windows are wide, producing a wrong period boundary? Is a more robust key
   needed?
4. **The interleaving case.** A nondeterministic event whose mean falls between
   same-frame deterministic events — is `overlaps_frame`/concurrent the right
   honest representation, and does the algorithm actually produce it rather than
   silently splitting a frame?
5. **Re-sync frames vs genuine independent jitter.** Two events sharing a non-zero
   jitter-vector — always a real re-sync frame, or can this be an artifact?
6. **Intermittent events that are actually deterministic-when-present.** Should a
   sometimes-firing event that is rigid in the runs where it appears be partly
   characterized, or stay fully in the intermittent bucket?

## Deliverables

- `tools/inference/timeline.py` — period model + `assemble_timeline` + `render_timeline`
- `offset_window` helper in `verifier.py`
- `timeline` wired into `engine.py` and `run_experiment.py` reports
- `two_col` config fixture + offline multi-period/multi-column tests + HW-gated E2E
- Dead `Timeline`/`assemble` removed from `grounding.py`

## Future Work (noted, out of scope)

- Perfetto/GUI rendering of the `IntegratedTimeline`.
- Running the engine across the full corpus to produce the per-event NPU
  determinism map (the ground-truth spec for emulator fidelity).
- Emulator-vs-timeline comparison (the fidelity arc that sits on top).
