# Integrated Timeline Engine — Design

**Date:** 2026-06-26
**Status:** Draft v2 (post-adversarial-review); pending user review then implementation plan
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
  `derives`) — a bag of independent edges, not a timeline.
- **The heart is missing.** "Deterministic events ground nondeterministic ones"
  requires composing edges into a structure where each event is positioned within
  a rigid frame and each gap is typed and bounded. That composition does not exist.
- **Only single-column has ever run end-to-end.** `add_one_using_dma` (one column,
  one batch) is the only kernel run through the full path. Cross-tile and
  multi-batch behavior is untested — and, per the review below, is exactly where
  the naive single-axis design fails.

## Design History (why this is v2)

Draft v1 proposed a **single global linear timeline**: one axis, all events placed
by their anchored timestamp, one alternating period sequence. A three-lens
adversarial review (math, measurement-pipeline, model-faithfulness) converged on
the same fatal defect: **a single linear axis reinterprets a single-anchor
measurement convenience as physical structure, and that contradicts the project's
own #140 cross-domain-skew result.** Three independent failures:

1. **Cross-domain offsets are not cycles.** The anchor lives in one timer domain
   (tile (1,2) core). An event in another domain has anchored offset `Δwall +
   skew` — provably un-decomposable (cross-domain-skew-limit.md), never a causal
   cycle. v1 would have assigned these as "exact local cycles," and worse,
   `check_additivity` is blind to the error because skew is additively consistent
   (+2/+4/−2 sum correctly). The existing `ground_edge` already gates this with
   `same_domain()`; v1 dropped the gate.
2. **Concurrent columns can't be a single sequence.** Multi-column kernels run
   tiles in parallel; a single left-to-right period list fabricates a total order
   over physically concurrent events and lets an unrelated tile's event "ground" a
   jitter period it has no causal link to.
3. **Cross-batch clustering is unsound.** Each capture *batch* is a separate
   silicon execution. Two events in different batches have jitter sampled from
   different executions, paired only by run index — so non-zero "re-sync" clusters
   are undetectable or fabricatable across batch boundaries. (The deterministic,
   all-zero backbone *does* survive cross-batch.) This bites on multi-batch
   kernels, never on single-batch `add_one` — which is why it was invisible.

v2 keeps the validated core (within-domain, within-batch jitter-vector clustering
reduces exactly to the HW-proven `offset_exact` Q=0 primitive) and the period
schema, but moves them onto the honest topology: **per-domain tracks woven by the
typed cross-domain gaps #140 already built.**

## Goal / Definition of Done

`run_engine` emits an `IntegratedTimeline` for a kernel: **one period sequence per
timer-domain track**, each track's deterministic periods recorded cycle-by-cycle
in their own re-zeroed coordinates and its nondeterministic periods recorded as
windowed, partially-ordered events; tracks **woven together by typed cross-domain
edges** (`reproduction_offset` / `cross_domain` / `async_cdc`) wherever a causal
edge crosses domains; every event characterized as an **occurrence sequence** (not
just its first firing); every unaccounted weirdness flagged. Validated end-to-end
on a **multi-column** kernel (`two_col`).

"Accounted for" means each event is, within its track, in exactly one of: a
deterministic period (count- and position-characterized), a nondeterministic
period (windows + partial order + typed reason), or a presence-equivalence class
in the intermittent set. No event silently absent; no order asserted that the runs
contradict; no absolute cycle claimed across a gap; **no cross-domain offset ever
reported as a cycle.**

## Non-Goals

- **Visualization.** The deliverable is the structured `IntegratedTimeline` plus a
  plain-text renderer for tests and eyeballing. Perfetto/GUI rendering is future.
- **Emulator comparison.** This characterizes *hardware* determinism (HW-vs-HW).
  Emulator-vs-timeline fidelity is the next arc.
- **Corpus campaign.** This builds the engine; running it across the corpus to
  produce the determinism map is the subsequent step.
- **Autonomous load-vs-HW classification.** A range>0 within-domain span is
  verified manually (re-capture on a quiet host). The engine flags; a human
  classifies. Unchanged from the canary work.
- **Input-space coverage.** The timeline characterizes determinism for **one fixed
  test input**. Value-dependent timing (data-dependent branches/loops) is rigid
  across runs of the same input and is reported as deterministic; the output is
  explicitly labeled input-conditioned. Characterizing across inputs is future.

## Conceptual Model

**Timer-domain tracks.** A *track* is one timer domain — the `(col, row, pkt_type)`
prefix already used by `grounding.same_domain`. The trace timer resets per domain,
so a track is the largest region where a raw offset is a true cycle count. A tile
has several tracks (core module, mem module, …). Each track gets its own period
sequence.

**Within a track: the period schema.** Within one domain, the alternating-period
model holds cleanly:

```
track 1|2|0 (core):  [DET frame A]→[NONDET]→[DET frame C]→ …
                      local cyc 0.. windows   local cyc 0.. (re-zeroed)
```

- **Deterministic period** — a maximal rigid frame; events recorded cycle-by-cycle
  in the frame's own coordinates (zero at the frame's grounding event).
- **Nondeterministic period** — events between two frames, each a window, partially
  ordered, closed by the grounding event that resumes determinism.
- **Re-zeroing** — each deterministic period zeros at its own grounding event,
  because the duration of the preceding nondeterministic period is exactly what
  varies; we never claim an absolute cycle across it.

**The single anchor is the measurement zero, not a frame.** Every event's
`anchored_ts` is measured against the one global anchor (present in every batch).
Within a track this subtraction is skew-free for *mutual* offsets (the common skew
to the anchor cancels), so within-track clustering is clean. Across tracks the
anchored value carries skew and is **never** used as a position — only as a coarse
sequencing hint constrained by causal edges.

**Cross-track weave.** Tracks relate only through real couplings — a DMA flow, a
lock handoff, a stream — which appear in the causal `derives` graph
(`config_path` / `program_path`). Each cross-track causal edge becomes a **typed
gap**: `cross_domain` carrying a `reproduction_offset` when the raw cross-domain
offset is range-0 (else None), or `async_cdc` for shim NoC egress. This is exactly
`ground_edge`'s existing cross-domain behavior; the weave reuses it. The
`IntegratedTimeline` is therefore a *set of per-track period sequences plus a graph
of typed cross-track edges* — not one global sequence.

**Occurrence sequences (not first-firing points).** The events we most need to
characterize — `PORT_RUNNING`, `LOCK_STALL` spans, held-level B/E phases, repeated
DMA bursts — fire multiple times with structured cadence. Each event is therefore
characterized as a **per-run sequence of firings**, and determinism is assessed at
two levels: **count** (does it fire the same number of times every run?) and
**per-occurrence position** (is occurrence *k* rigid or windowed?). A first-firing
scalar would be silent on exactly the behavior the emulator is being calibrated
against.

**Represent things only as weird as they are.** Deterministic count + position →
exact. Stable count, jittery position → per-occurrence windows. Jittery count →
count-window, flagged. Orderable → ordered; genuinely racing → concurrent.
Fires-only-sometimes → a presence class. Never more structure than the silicon
shows, never less.

## Data Model

**Event record (per event, per track):**
- `occurrences`: per run, the ordered list of `anchored_ts` firings.
- `count`: `count_rigid(n)` if the firing count is identical across all runs, else
  `count_window(min, max)` (flagged).
- per-occurrence classification (only when count-rigid, so occurrence *k* is
  well-defined across runs): each occurrence *k* is **deterministic** (range-0
  → exact cycle within its frame) or **nondeterministic** (window `[min,max]`).
- `domain`: the `(col,row,pkt_type)` track id.

**Track:** an ordered list of periods over that domain's events:
- **`DeterministicPeriod`** — a rigid frame (single-domain by construction), events
  with exact local cycles, zero at the grounding event. A `floating` window
  relative to the prior frame when the frame is mutually-rigid but not anchor-rigid
  (see "mutually-rigid floating cluster" below).
- **`NondeterministicPeriod`** — events with windows + `reason`, an **order
  relation** (`order_edges` tagged `causal` or `stable_position`; unconnected pairs
  = concurrent), closed by a `grounding_event` (None ⇒ `ungrounded_tail`).

**Cross-track edges:** typed gaps from `ground_edge` for causal edges whose
endpoints are in different domains — `cross_domain(reproduction_offset|None)` or
`async_cdc`. These are the only cross-track positional statements.

**Intermittent set:** events not present in every run, grouped into
**presence-equivalence classes** (identical appearance-set). Complementary classes
are noted as candidate mutual-exclusion (a conditional branch). Within a class,
events are characterized (rigid/window) over the runs where they appear. Distinct
from **anchor dropout** (a whole batch missing its anchor → a capture-health flag,
not per-event intermittency).

**`IntegratedTimeline`:** `{ tracks: [Track], cross_track_edges: [TypedGap],
intermittent: [PresenceClass], flags: [HonestyFlag], capture: {witness, runs,
input_id} }`.

**Honesty flags:** `ungrounded_tail` (per track), `count_window` (jittery firing
count), `load_suspect` (cluster split by ≤ a documented load band — re-capture),
`overlaps_frame` (window straddles a frame; carries the partial order it *does*
earn against individual frame events, blanket-concurrent only where no stable edge
exists), `additivity_unverifiable` (frame membership not confirmable by direct
co-traced measurement), `batch_flip` (event's winning batch index varies across
runs — excluded from clustering).

## The Algorithm

Inputs: N clean (witness-certified) run dirs for one kernel + one input; the
configured-event set; the causal `derives` graph; the global anchor.

1. **Capture occurrences.** For each event, in each run, collect the ordered
   `anchored_ts` of *all* firings within each batch (new occurrence-capturing
   accessor beside `batch_firsts`, sharing the same anchoring path). Determine
   `count` per run. Tag `count_rigid` / `count_window`.

2. **Eligibility gates (before any clustering).**
   - **Batch-stability:** an event whose first-co-tracing batch index varies across
     runs is `batch_flip` → excluded from clustering, routed to intermittent/gap.
     (Reconciles `loader`'s eps=2.0 cross-batch tolerance with the clusterer's Q=0:
     only same-batch-every-run events are clustered.)
   - **Presence:** events absent in some runs → intermittent presence classes.
   - **Anchor dropout:** batches with no anchor firing → capture-health flag, not
     per-event intermittency.

3. **Partition into tracks** by `(col,row,pkt_type)` domain.

4. **Within each track, within co-tracing batches: rigid clusters.** For
   count-rigid events, per occurrence index, compute the jitter-vector
   (`anchored_ts − anchored_ts[run 0]`). Rigid linkage ⟺ identical jitter-vector
   (exact integer, Q=0). Because all members share a domain, the common skew
   cancels and offsets are pure cycles. Group:
   - all-zero jitter-vector (anchor-rigid) → the track's anchored deterministic
     frame;
   - shared non-zero jitter-vector → a **mutually-rigid floating cluster** (rigid
     internally; position floats by a window relative to the prior frame). *Not*
     asserted to be a hardware "re-sync" — jitter-vector equality cannot distinguish
     a true re-synchronization from common-cause co-jitter (both downstream of one
     jittery parent), so the label stays neutral.
   - **Corroboration gate for non-zero clusters:** emit a multi-member floating
     cluster only if (a) a `derives` causal edge links members, or (b) N meets a
     documented minimum and the data-estimated false-cluster bound
     (`≈ pairs · p_c^(N−1)`, `p_c` the measured per-component collision rate) is
     below threshold. Otherwise demote members to nondeterministic. (Low-entropy
     jitter makes coincidental clusters *likely*, not rare, at small N — this gate
     is mandatory, not belt-and-suspenders.)

5. **Internal cycles.** Within a cluster, local zero = earliest occurrence; each
   member's cycle = its exact same-domain pairwise offset. Verify with
   `check_additivity`; a `None` (not directly co-traced) result is
   `additivity_unverifiable`, **not** a pass — demote unverifiable members.

6. **Build each track's period sequence.** Group events by cluster identity first
   (so a single rigid frame is never fragmented by sequencing), then order frames
   and nondeterministic events within the track by intra-track position. A run of
   nondeterministic events between two frames → a `NondeterministicPeriod`, windows
   measured against the upstream frame's reference, grounding event = first event of
   the next frame; trailing run with no next frame → `ungrounded_tail`.

7. **Intra-period order.** `order_edges` from causal `derives` edges plus
   strictly-stable cross-run position (a < b in *every* run) — the latter subject to
   the same N/corroboration discipline as clusters (stable order over few runs is
   weak evidence). Unconnected pairs → concurrent.

8. **Weave tracks.** For each `derives` edge crossing domains, emit the typed gap
   from `ground_edge` (`cross_domain` with `reproduction_offset|None`, or
   `async_cdc`). These connect periods across tracks; no shared-axis position is
   ever asserted.

9. **Honesty flags & intermittent classes** as in the data model.

**Determinism detection** is jitter-vector equivalence *within a domain and batch*;
the existing `anchor_rigid` (range-0 vs global anchor) is the special case
"all-zero jitter-vector," and is shared via the same `_anchored_per_run` path so the
two never silently diverge. Equivalence holds for events present in all N runs;
intermittent-but-rigid events are characterized within their presence class.

## Components / Where It Lives

- **`tools/inference/timeline.py` (new).** The data model (`EventRecord`, `Track`,
  `DeterministicPeriod`, `NondeterministicPeriod`, `PresenceClass`,
  `IntegratedTimeline`, flags) and `assemble_timeline(run_dirs, derives, configured,
  anchor)` — the per-track segmentation, cross-track weave, and occurrence analysis.
- **`tools/inference/verifier.py` (extend).** An occurrence accessor (all firings
  per event/run) and a `offset_window` helper beside `offset_exact`, reusing the
  `tj.batch_firsts` anchoring path; factor out `_anchored_per_run` for sharing.
- **`tools/trace_join.py` (extend).** A `batch_occurrences` (all firings, not just
  first) beside `batch_firsts`.
- **`tools/inference/engine.py` (wire).** Call `assemble_timeline`; add `timeline`
  to the report alongside `segments`/`gaps`/`warnings`.
- **`tools/inference/run_experiment.py` (propagate).** Thread `timeline` through the
  report (best-effort, like the existing engine block).
- **`tools/inference/grounding.py` (retire dead code).** Remove the unused
  `Timeline`/`assemble`; keep `ground_edge`, `Segment`, `Gap`, reasons,
  `same_domain` — the weave reuses them.
- **A plain-text renderer** (`render_timeline`): per-track A→B→C view with local
  cycles, occurrence windows, concurrency, cross-track typed edges, and flags.

## Validation — Multi-Column

The largest untested risk is cross-track/multi-batch integration.

- **`two_col` config fixture** via the existing `config_extract` path. Two columns
  → multiple domains across tiles → exercises per-track partition, cross-track
  weave, and (because shim has 9 DMA events > 8 slots) genuine multi-batch capture.
- **Offline, TDD.** Synthetic run-dir fixtures with hand-built patterns: clean
  single-domain frames; a mutually-rigid floating cluster (with and without
  corroboration); cross-domain coincidence that must become a typed edge, not a
  cycle; multi-batch split of a rigid pair (must NOT cluster across batches);
  `batch_flip`; count-rigid burst vs count-window; per-occurrence window;
  presence-equivalence classes incl. a complementary (mutual-exclusion) pair;
  `overlaps_frame` retaining partial order; `ungrounded_tail`; multi-track weave.
- **HW-gated, end-to-end.** `two_col` through `run_experiment` on the real NPU
  (controller-run, witness-gated), asserting a well-formed multi-track
  `IntegratedTimeline`: no cross-domain cycles, no swallowed events, expected track
  count.
- **Decisive A/B checks the review demanded** (run on real captured data, not just
  synthetic): (i) on existing 20-run `add_one`, confirm an event co-traced in ≥2
  batches has range-0 batch-to-batch anchored_ts (else cross-batch comparison is
  contaminated and the co-trace restriction is doing real work); (ii) quiet-20 vs
  loaded-20 `add_one` — clusters stable on quiet, fragment on loaded (the canary,
  working).

## Testing Strategy

Pure Python (`tools/`); `cargo test --lib` not required. TDD throughout. Offline
coverage for every algorithm step and every honesty flag, plus the synthetic
fixtures above. HW-gated E2E on `two_col` is controller-run, gated like existing HW
tests, never in the offline suite.

## Open Questions for User / Implementation

1. **Track granularity:** per timer domain `(col,row,pkt_type)` (chosen — it's the
   cycle-coherent unit) vs per tile `(col,row)` (coarser, mixes modules). Confirm
   per-domain.
2. **Minimum N and the false-cluster threshold** for non-zero floating clusters:
   the corroboration gate needs a concrete N floor and `p_c` bound. Derive from the
   measured jitter entropy of the corpus; pick during implementation against real
   data, documented, never tuned to pass a specific test.
3. **Held-level (B/E phase) events** vs pulse events: a level event's "occurrence"
   is a (begin,end) span; characterize span determinism. Confirm the occurrence
   model covers both pulses and spans, or split the event kinds.

## Deliverables

- `tools/inference/timeline.py` — data model + `assemble_timeline` + `render_timeline`
- occurrence + `offset_window` helpers in `verifier.py`; `batch_occurrences` in `trace_join.py`
- `timeline` wired into `engine.py` and `run_experiment.py`
- `two_col` config fixture + offline synthetic suite + real-data A/B checks + HW-gated E2E
- Dead `Timeline`/`assemble` removed from `grounding.py`

## Future Work (noted, out of scope)

- Perfetto/GUI rendering of the `IntegratedTimeline`.
- Running the engine across the corpus to produce the per-event NPU determinism map.
- Emulator-vs-timeline fidelity comparison (the arc that sits on top).
- Cross-input characterization (value-dependent timing as a coordinate).
