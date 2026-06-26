# Integrated Timeline Engine — Design

**Date:** 2026-06-26
**Status:** Draft v3 (post second review); pending final confirmation pass + user review
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
  are never called. `run_engine` returns flat lists — a bag of independent edges.
- **The heart is missing.** "Deterministic events ground nondeterministic ones"
  requires composing edges into a structure where each event is positioned within
  a rigid frame and each gap is typed and bounded. That does not exist.
- **Only single-column, single-batch has run end-to-end.** `add_one_using_dma` is
  the only kernel run through the full path — exactly the configuration that hides
  the cross-tile and multi-batch failures the reviews below exposed.

## Design History

**v1 — single global linear axis.** One axis, all events placed by anchored
timestamp, one period sequence. A three-lens adversarial review (math /
measurement / model) converged on one fatal root: a single linear axis
reinterprets a measurement convenience as physical structure. Three failures: (A)
cross-domain offsets are `Δwall + skew`, never cycles (contradicts
cross-domain-skew-limit.md; `check_additivity` blind to it because skew is
additively consistent); (C) concurrent columns flattened into a false total order
with spurious grounding events; (B) cross-batch jitter-vectors sampled from
independent executions make non-zero clusters fabricatable.

**v2 — per-domain tracks.** Moved the validated core (within-domain, within-batch
jitter clustering == `offset_exact` Q=0) onto per-timer-domain tracks woven by the
existing typed cross-domain gaps; added occurrence sequences, a corroboration gate,
eligibility gates, the "mutually-rigid floating cluster" rename. A second review
(closure audit + fresh-holes hunt) confirmed v1's fatals A/B/C **genuinely closed**
but found the new machinery introduced its own crop, fixed here in v3:

- **F1** the weave consumed `derives` *facts*, which `try_derives` refuses to emit
  for a deterministic parent — so a fully-deterministic multi-tile kernel weaves to
  nothing and the tracks disconnect. → weave over cross-domain *candidate pairs*
  via `ground_edge` directly.
- **F2** per-frame re-zeroing discarded the frame's offset-to-anchor while
  cross-track offsets are anchor-relative → non-composable coordinates. → every
  frame carries its **offset-to-anchor as a (possibly degenerate) window**; a
  "floating cluster" is simply a frame whose offset-window is non-degenerate.
- **F3** "demote on additivity-unverifiable" collided with the 8-slot co-trace
  ceiling and would discard determinism recoverable through the always-on anchor. →
  internal cycles come straight from the anchored/jitter-vector data (no
  co-tracing); additivity is an optional tri-state cross-check, never a demotion
  gate.
- **F4** the corroboration gate's causal path was dead for its own motivating case
  (co-jittering siblings have no edge to each other). → corroborate via a
  **common-parent** edge to the shared jittery root.
- **F5** `count_window` discarded all per-occurrence determinism wholesale, and
  "count" is partly trace-buffer-truncation-driven. → characterize the longest
  common rigid prefix; flag only the variable tail; detect buffer overflow to
  separate captured-count from silicon-count.
- **D** the deterministic verdict itself had no N-floor. → it is provisional like
  every Q=0 verdict (confidence from N + witness re-capture; documented min-N).
- **F6/F7** occurrence well-definedness across batches; eligibility-gate ordering.
  → pin one batch per event; order gates dropout → presence → batch-stability →
  count.

## Goal / Definition of Done

`run_engine` emits an `IntegratedTimeline`: **one period sequence per timer-domain
track**, each track's deterministic periods recorded cycle-by-cycle in their own
re-zeroed coordinates and carrying their offset-to-anchor window, its
nondeterministic periods recorded as windowed partially-ordered events; tracks
**woven together by typed cross-domain edges** wherever a causal coupling crosses
domains; every event characterized as an **occurrence sequence**; every unaccounted
weirdness flagged. The weave must connect every pair of physically-coupled tracks
(checked explicitly). Validated end-to-end on a **multi-column** kernel (`two_col`).

"Accounted for" means each event is, within its track, in exactly one of: a
deterministic period (count- and position-characterized), a nondeterministic period
(windows + partial order + typed reason), or a presence-equivalence class. No event
silently absent; no order asserted that the runs contradict; no absolute cycle
claimed across a gap; **no cross-domain offset ever reported as a cycle.**

## Non-Goals

- **Visualization.** Deliverable is the structured `IntegratedTimeline` + a
  plain-text renderer. Perfetto/GUI is future.
- **Emulator comparison.** This characterizes *hardware* (HW-vs-HW). Fidelity is
  the next arc.
- **Corpus campaign.** This builds the engine; the corpus determinism map is next.
- **Autonomous load-vs-HW classification.** range>0 within-domain is verified
  manually (quiet-host re-capture). Engine flags; human classifies.
- **Input-space coverage.** Characterizes one **fixed input**; value-dependent
  timing is reported deterministic and the output labeled input-conditioned.

## Conceptual Model

**Timer-domain tracks.** A *track* is one timer domain — the `(col,row,pkt_type)`
prefix used by `grounding.same_domain`. The trace timer resets per domain, so a
track is the largest region where a raw offset is a true cycle count. A tile has
several tracks; each gets its own period sequence.

**Within a track: the period schema.** Within one domain the alternating model
holds cleanly:

```
track 1|2|0 (core):  [DET frame A]→[NONDET]→[DET frame C]→ …
                      local cyc 0.. windows   local cyc 0.. (re-zeroed)
                      off2anchor=exact        off2anchor=window (floats)
```

- **Deterministic period (frame)** — a maximal rigid cluster; events recorded
  cycle-by-cycle in the frame's own coordinates (zero at the grounding event).
  Carries `offset_to_anchor`, a window across runs: **degenerate (exact)** when the
  frame is anchor-rigid, **non-degenerate** when it floats (a "mutually-rigid
  floating cluster" is exactly this — internally rigid, position uncertain). The
  window *is* the inter-frame uncertainty, kept first-class so cross-track edges
  compose.
- **Nondeterministic period** — events between two frames, each a window, partially
  ordered, closed by the grounding event that begins the next frame.
- **Re-zeroing** — each frame zeros at its own grounding event; absolute position is
  carried only as the `offset_to_anchor` window, never as a claimed cycle.

**The single anchor is the measurement zero, not a frame.** Every event's
`anchored_ts` is measured against the one global anchor (present in every batch).
*Within* a track the common skew to the anchor cancels in mutual offsets, so
within-track cycles are skew-free and recoverable through the anchor without
co-tracing. *Across* tracks the anchored value carries skew and is never a position.

**Cross-track weave.** Tracks relate only through real couplings — DMA, lock,
stream — which appear as **cross-domain candidate pairs**. Each is typed by
`ground_edge` into a `cross_domain` gap (with `reproduction_offset` when range-0,
else None) or `async_cdc`. The weave iterates candidate pairs directly via
`ground_edge` (which has no stochastic-parent gate), so **deterministic couplings
are drawn too** — distinct from the `derives` jitter graph, whose stochastic-parent
gate exists for a different purpose. The `IntegratedTimeline` is a set of per-track
period sequences plus a graph of typed cross-track edges.

**Occurrence sequences.** Events fire repeatedly with structured cadence
(`PORT_RUNNING`, `LOCK_STALL`, DMA bursts). Each event is characterized as a per-run
firing sequence from one **pinned batch** (the lowest always-on co-tracing batch);
other batches are independent samples, never concatenated. Determinism is assessed
over the **longest common rigid prefix**: the maximal leading run of occurrences
present and rigid in every run is characterized cycle-by-cycle; the variable tail is
flagged (`count_window`). Cross-run, occurrence *k* is matched by index, valid only
when firings are monotonic in ts within the pinned batch (else flagged
`occurrences_reorderable` and characterized as a set, not a sequence). A captured
count at the trace-buffer ceiling is flagged `count_truncated` — captured count is
not silicon count.

**Represent things only as weird as they are.** Exact when exact; windowed when it
jitters; concurrent when it races; presence-class when it sometimes fires; floating
when internally rigid but absolutely uncertain. Never more structure than the
silicon shows, never less.

## Data Model

**Event record (per event):**
- `domain`: the `(col,row,pkt_type)` track id.
- `pinned_batch`: the batch its occurrences are read from.
- `occurrences`: per run, the ordered firing `anchored_ts` from the pinned batch.
- `rigid_prefix_len`: length of the longest common rigid prefix; occurrences `[0,
  rigid_prefix_len)` are position-characterized, the rest are the variable tail.
- per-occurrence classification (over the prefix): each occurrence **deterministic**
  (range-0 → cycle within its frame) or **nondeterministic** (window `[min,max]`).
- flags: `count_window` (variable tail nonempty), `count_truncated`,
  `occurrences_reorderable`.

**Track:** ordered list of periods over that domain's events:
- **`DeterministicPeriod`** — single-domain rigid frame; events with exact local
  cycles (zero at grounding event); `offset_to_anchor` window (degenerate ⇒
  anchored frame, non-degenerate ⇒ floating frame).
- **`NondeterministicPeriod`** — events with windows + `reason`; an `order_edges`
  relation (tagged `causal` from a common-parent/chain edge, or `stable_position`;
  unconnected pairs = concurrent); a `grounding_event` (None ⇒ `ungrounded_tail`;
  present-but-no-causal-edge-into-the-period ⇒ `resumption_unattested`).

**Cross-track edges:** typed gaps from `ground_edge` over cross-domain candidate
pairs — `cross_domain(reproduction_offset|None)` or `async_cdc`. Composition: with
`offset_to_anchor` retained on both endpoints' frames, a consumer can place both
ends (exact when both frames anchored, windowed when either floats). An edge whose
endpoint lies inside a nondeterministic period (a windowed event) carries no
position — it is **existence/typing-only**, never positional.

**Intermittent set:** events absent in some runs, grouped into
**presence-equivalence classes** (identical appearance-set); complementary classes
noted as candidate mutual-exclusion (a conditional branch); within a class, events
characterized over the runs where they appear. Distinct from **anchor dropout** (a
whole batch missing its anchor → capture-health flag).

**`IntegratedTimeline`:** `{ tracks, cross_track_edges, intermittent, flags,
capture: {witness, n_runs, input_id} }`.

## The Algorithm

Inputs: N clean (witness-certified) run dirs for one kernel + one input; the
configured-event set; the causal `derives` graph + cross-domain candidate pairs;
the global anchor.

1. **Capture occurrences.** For each event, pin the lowest always-on co-tracing
   batch; read its per-run ordered firing `anchored_ts` (new `batch_occurrences`
   beside `batch_firsts`, same anchoring path). Compute the longest common rigid
   prefix; flag `count_window` / `count_truncated` / `occurrences_reorderable`.

2. **Eligibility gates, in order:**
   1. **Anchor dropout** — batches with no anchor firing → capture-health flag;
      those batch-runs do not count as event "absence."
   2. **Presence** — events absent in some (non-dropout) runs → intermittent
      presence classes.
   3. **Batch-stability** — among survivors, an event whose pinned batch index
      varies across runs → `batch_flip`, excluded from clustering, routed to
      intermittent/gap. (Reconciles `loader`'s eps=2.0 cross-batch tolerance with
      the clusterer's Q=0.)
   4. **Count** — `count_window` events are still position-characterized over their
      rigid prefix (never wholesale-excluded).

3. **Partition into tracks** by domain.

4. **Within each track: rigid clusters.** For each occurrence index in the rigid
   prefix, compute the jitter-vector (`anchored_ts − anchored_ts[run 0]`). Rigid
   linkage ⟺ identical jitter-vector (exact integer, Q=0). Same-domain ⇒ common
   skew cancels ⇒ pure cycles. Group: all-zero jitter-vector ⇒ anchored frame;
   shared non-zero ⇒ floating frame; unique non-zero ⇒ nondeterministic event.
   - **Determinism is provisional.** A rigid verdict (anchored or floating) is a
     Q=0 verdict: confidence scales with N and is reconfirmed by witness
     re-capture. A multi-member non-zero (floating) cluster additionally requires
     **corroboration**: a **common-parent** `derives` edge to a shared jittery root
     (∃P: derives(P,a) ∧ derives(P,b) — the real causal signal for co-jittering
     siblings; a sibling↔sibling edge does not exist and is not required), or, in
     its absence, N at/above the documented floor with the estimated false-cluster
     bound (`≈ pairs · p_c^(N−1)`, `p_c` measured) below threshold. Else demote
     members to nondeterministic. (Low-entropy jitter makes coincidental clusters
     likely at small N — this gate is mandatory.)

5. **Internal cycles (no co-tracing required).** Within a cluster, local zero =
   earliest occurrence; member m's cycle = `anchored_ts(m) − anchored_ts(zero)` in
   any run (constant by cluster definition; same-domain ⇒ skew-free). Where a pair
   *is* directly co-traced, `check_additivity` is run as a **tri-state cross-check**
   (PASS / VIOLATION / UNVERIFIABLE — distinguishing success from the current
   `None`-on-success-and-on-gap conflation, and treating a <3-member chain as
   vacuous-PASS not a failure); a VIOLATION demotes the cluster, UNVERIFIABLE does
   not.

6. **Build each track's period sequence.** Group by cluster identity first (a frame
   is never fragmented by sequencing), then order frames and nondeterministic events
   within the track by intra-track position. A run of nondeterministic events
   between two frames → a `NondeterministicPeriod` (windows vs the upstream frame's
   reference; grounding event = next frame's first event; trailing → `ungrounded_
   tail`; grounding frame with no causal edge into the period → `resumption_
   unattested`).

7. **Intra-period order.** `order_edges` from causal edges (common-parent/chain via
   `derives`) plus strictly-stable cross-run position (a < b every run), the latter
   under the same N/corroboration discipline as clusters. Unconnected pairs →
   concurrent.

8. **Weave tracks.** For each **cross-domain candidate pair**, emit the typed gap
   from `ground_edge` directly. After weaving, **verify connectivity**: every pair
   of tracks with a known physical coupling must be connected in the edge graph;
   report any disconnected coupled tracks as a defect (this is the F1 guard the DoD
   checks).

9. **Honesty flags & intermittent classes** as in the data model.

**Determinism detection** is jitter-vector equivalence within a domain; the
existing `anchor_rigid` is the special case "all-zero jitter-vector," shared via the
same `_anchored_per_run` path (factored out, not reimplemented) so the two never
silently diverge. Equivalence holds for events present in all N runs;
intermittent-but-rigid events are characterized within their presence class.

## Components / Where It Lives

- **`tools/inference/timeline.py` (new).** Data model (`EventRecord`, `Track`,
  `DeterministicPeriod`, `NondeterministicPeriod`, `PresenceClass`,
  `IntegratedTimeline`, flags) + `assemble_timeline(...)` + connectivity check +
  `render_timeline`.
- **`tools/trace_join.py` (extend).** `batch_occurrences` (all firings per event in
  a batch, ordered) beside `batch_firsts`.
- **`tools/inference/verifier.py` (extend).** Occurrence accessor + `offset_window`
  beside `offset_exact`; factor out `_anchored_per_run` for sharing; make
  `check_additivity` tri-state.
- **`tools/inference/engine.py` (wire).** Call `assemble_timeline`; add `timeline`
  to the report.
- **`tools/inference/run_experiment.py` (propagate).** Thread `timeline` through.
- **`tools/inference/grounding.py` (retire dead code).** Remove unused
  `Timeline`/`assemble`; keep `ground_edge`, `Segment`, `Gap`, reasons,
  `same_domain`.

## Validation — Multi-Column

- **`two_col` config fixture** via `config_extract`. Two columns → multiple
  domains → exercises per-track partition, the weave (incl. a deterministic
  column-to-column handoff that **must** produce a connecting edge — the F1 guard),
  and genuine multi-batch capture (shim 9 DMA events > 8 slots).
- **Offline, TDD.** Synthetic fixtures: clean anchored frame; floating frame (with
  and without common-parent corroboration); cross-domain coincidence that must
  become a typed edge not a cycle; multi-batch split of a rigid pair (must NOT
  cluster across batches); deterministic cross-track coupling (weave must connect,
  not disconnect); `batch_flip`; count-rigid burst; count-variable with a rigid
  prefix + flagged tail; `occurrences_reorderable`; per-occurrence window;
  presence-equivalence classes incl. a complementary pair; cross-track edge into a
  nondeterministic period (existence-only); `overlaps_frame` retaining partial
  order; `ungrounded_tail`; `resumption_unattested`; offset-to-anchor composition
  across a cross-track edge.
- **HW-gated, end-to-end.** `two_col` through `run_experiment` (controller-run,
  witness-gated): well-formed multi-track timeline, no cross-domain cycles, no
  swallowed events, **connectivity holds**.
- **Real-data A/B checks the reviews demanded:** (i) on existing 20-run `add_one`,
  an event co-traced in ≥2 batches has range-0 batch-to-batch anchored_ts (else the
  cross-batch restriction is load-bearing); (ii) quiet-20 vs loaded-20 — clusters
  stable on quiet, fragment on loaded (the canary working).

## Testing Strategy

Pure Python (`tools/`); `cargo test --lib` not required. TDD throughout. Offline
coverage for every algorithm step, every honesty flag, the connectivity check, and
the synthetic fixtures above. HW-gated E2E on `two_col` is controller-run, gated
like existing HW tests, never in the offline suite.

## Open Questions for Implementation

1. **Minimum N and false-cluster threshold** for floating clusters / stable-position
   edges / the provisional deterministic verdict: derive from the corpus's measured
   jitter entropy against real data, documented, never tuned to pass a test.
2. **Held-level (B/E phase) events**: a level event's "occurrence" is a
   (begin,end) span; characterize span determinism (begin-cycle, span-length) within
   the same occurrence model, or split the event kind. Decide against real
   level-event captures.
3. **Track granularity** confirmed per timer domain `(col,row,pkt_type)` (the
   cycle-coherent unit), not per tile.

## Deliverables

- `tools/inference/timeline.py` — model + `assemble_timeline` + connectivity check + `render_timeline`
- `batch_occurrences` (trace_join), occurrence + `offset_window` helpers + tri-state `check_additivity` (verifier)
- `timeline` wired into `engine.py` and `run_experiment.py`
- `two_col` fixture + offline synthetic suite + real-data A/B checks + HW-gated E2E
- Dead `Timeline`/`assemble` removed from `grounding.py`

## Future Work (noted, out of scope)

- Perfetto/GUI rendering of the `IntegratedTimeline`.
- Corpus-wide determinism map (the emulator's ground-truth spec).
- Emulator-vs-timeline fidelity comparison.
- Cross-input characterization (value-dependent timing as a coordinate).
