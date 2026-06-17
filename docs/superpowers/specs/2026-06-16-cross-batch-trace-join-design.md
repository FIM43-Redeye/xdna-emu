# Derivability-driven cross-batch trace join (task #140) — design

**Date:** 2026-06-16
**Status:** approved design, pre-plan
**Predecessor:** `2026-06-16-dma-nondeterminism-characterization-design.md` and its
results (`docs/superpowers/findings/2026-06-16-dma-nondeterminism-characterization-results.md`).
The characterization proved that the corpus-wide trace divergence is dominated
by **DMA milestone timing jitter** the silicon does not reproduce run-to-run,
while a deterministic backbone exists. This design consumes that result.

## Goal

Merge the per-batch traces of a single kernel sweep into **one complete
every-event trace** — covering every event that fires on every tile — on a
single anchored timeline, despite the 8-slot-per-tile hardware trace limit that
forbids tracing every event in one run. Deterministic and derivable events are
placed exactly; genuinely stochastic events (DMA milestones) are placed as a
real observed sample carrying a measured uncertainty band.

This is the artifact that is "nominally impossible": a kernel run never produces
a trace of all its events at once, because the hardware can trace only ~8 events
per tile per execution. The join reconstructs the complete picture from many
partial executions by aligning them on events that fire reliably.

## Scope

**In scope (this cycle):** the HW-only complete-trace builder. Input is one
sweep's batch directories plus the multi-run variance characterization; output
is one merged HW trace plus the analysis artifacts (derivability graph, batch
plan) that justify it. Self-consistency across runs is the validation oracle.

**Deferred (separate cycle):** EMU-vs-merged comparison. The merged HW trace is
the oracle a future comparator will diff an EMU run against (masking the
stochastic band, holding the backbone strict). Not built here.

**Non-goals:** generalizing beyond NPU1/AIE2; multi-column kernels (single
column for now, matching `add_one_using_dma`); changing the capture/decode path
(upstream owns decode; we own sweep/merge glue).

## Why this shape — the empirical findings that ground it

All measured on `add_one_using_dma`, 20 contention-free HW runs, anchored to
core-tile `PERF_CNT_2` first-fire (`build/experiments/gap140/nondeterminism/`):

| event | mean (anchored) | std | reading |
|-------|-----------------|-----|---------|
| core (1.2) S2MM/MM2S START | ~5601 | ~1995 | stochastic |
| shim (1.0) S2MM/MM2S START | ~6667 | ~1966 | stochastic |
| memtile (1.1) PORT_RUNNING_0 | −830 | 179 | near-deterministic |

1. **Nondeterminism is not confined to the shim.** Core DMA is independently
   stochastic at the same order of magnitude as shim DMA. Every tile has its own
   DMA engine; each is a *candidate* independent stochastic source. The
   optimistic "track only the shim and derive the rest" model is unproven and
   the data leans against it. The number of independent sources must be
   **measured**, not assumed.

2. **`PERF_CNT_2` first-fire is a valid anchor.** If the anchor itself were the
   jitter source, every anchored event would carry ~2000 std; the memtile port
   carries only 179. So anchor noise is bounded ≤179 and the ~2000 on the DMAs
   is genuine DMA jitter. This also resolves characterization finding **I-1**:
   the join anchors on the real grounding event (`PERF_CNT_2`), never the
   self-referential `min(soc)`.

3. **The candidate sources are not co-traced in the current sweep.** No batch
   captures shim DMA and core DMA in one execution, so their cross-correlation
   is unmeasurable from existing data. Marginals show several stochastic events
   but cannot distinguish *one shared DOF* from *several independent DOFs*.
   Proving derivability requires deliberate **co-tracing** — which trace config
   fully permits, since we choose the per-batch slot assignment.

4. **Trace slots are per-tile.** Tiles 1.0/1.1/1.2 each fill their own 8-slot
   trace unit independently in one run. Monitoring DMA on the shim costs *shim*
   slots only; it does not compete with the memtile port sweep or the core event
   sweep. Co-tracing across tiles is therefore cheap.

5. **The join's event set is small.** Only ~26 events fire for this kernel
   (7 shim + 7 memtile + 12 core); the 25-batch sweep cost is full-*catalog*
   discovery overhead, most of which never fires. A targeted join needs only a
   handful of batches.

## Core model

### Windowed timeline with single-DOF stochastic gaps

A kernel's timeline is **alternating deterministic windows and stochastic
gaps**, not a single global frame. Compute events are deterministic relative to
*compute-start* (gated on input DMA completion), not relative to the boot-clocked
`PERF_CNT_2`; anchored globally on the perfcnt they would smear. So:

- A **deterministic window** has exact internal layout, anchored on a local
  landmark.
- A **stochastic gap** between windows is a DMA-delivery wait of banded width.
- Hardware law collapses each gap to **one stochastic degree of freedom** (the
  DDR/NoC wait): the characterization found the late DMA milestones share a
  *single* value, and `FINISHED = START + transfer`, `PORT_RUNNING span == words`
  are deterministic-by-law relations. We ground one edge of each gap and derive
  the rest.

**Bracketing requirement:** every stochastic event must be bracketed by a
deterministic/derivable landmark on each side, so both adjacent windows are
independently grounded and the gap is a measured spacer. An unbracketed
stochastic event flags a window the join cannot confidently place (surfaced, not
masked).

### Derivability and the minimal always-on set

Event X is **derivable** from predictor S when `(X − S)` is deterministic
(std ≤ eps) *measured within one execution* across runs. The **derivability
graph** has events as nodes and deterministic-offset edges; its **roots** —
events with no deterministic predictor among the traced set — are the **minimal
independent stochastic DOF set**.

Key leverage: if X derives from root R with a once-measured constant offset, then
pinning R in every batch makes X placeable **wherever** it is traced, even in a
batch that never co-traced X with R. So the **always-on set collapses to
{anchor} ∪ {roots}**, and the entire derivable closure becomes joinable. The
roots are few (one per independent stochastic source); per-tile slots make
pinning them nearly free.

## Architecture — four phases

### Phase 0: Discovery sweep

Run the existing full-catalog sweep once to obtain the **active event set** (the
events that actually fire on each tile for this kernel). Existing
`trace-sweep.py` capability; no new code beyond reading its output.

Output: `active-events.json` — `{(col,row): [event names]}`.

### Phase 1: Derivability analysis

Deliberately **co-trace the candidate stochastic sources** (every tile's DMA
engine milestones, plus any other event the discovery sweep flagged stochastic)
in shared batches, repeated N runs. For each candidate pair (X, S) measure
`(X − S)` within each execution and aggregate std across runs.

Build the **derivability graph**:
- node per active event;
- directed edge `S → X` with offset `mean(X − S)` when `std(X − S) ≤ eps`;
- **roots** = nodes with no incoming deterministic edge = minimal independent
  stochastic DOF set;
- **derivable closure** = all non-root nodes, each annotated with its predictor
  and offset.

The anchor (`PERF_CNT_2`) is the boot-clock root; the DMA-gap DDR-waits are the
work-clock roots. The graph reports how many independent roots exist — answering
empirically the "is it just the shim?" question per kernel.

Output: `derivability-graph.json` — nodes, edges (with offsets and the std that
justified them), roots, and for each root its band stats (mean/std/min/max from
the runs).

### Phase 2: Batch-plan synthesis

From the graph, compute the trace schedule that covers every active event in the
fewest batches while keeping every batch placeable:

- **Always-on, every batch:** `{anchor} ∪ {roots}`, assigned to their tiles'
  slots. **Hard constraint:** if `{anchor} ∪ {roots}` does not fit the per-tile
  slot budget simultaneously, the planner **panics** with a diagnostic (the
  offending tile, the roots assigned to it, the budget overage) rather than
  silently dropping a root and producing an unplaceable join.
- **Swept payload:** all remaining active events, packed into the slots that
  remain on *their* tiles, grouped so a derivable event sits with its predictor
  in at least one batch when the offset still needs confirmation.
- Batch count = max over tiles of `ceil(per-tile payload / remaining slots)`.

Output: `batch-plan.json` — per-batch, per-tile explicit slot→event assignment,
plus the always-on reservation and the resulting batch count.

### Phase 3: Join

Run the planned sweep (sweep gains a `--plan <batch-plan.json>` mode that honors
the explicit assignment instead of catalog-order slicing), then merge:

1. **Anchor** each batch on its core `PERF_CNT_2` first-fire; subtract from every
   event in that batch (all tiles share the batch's silicon clock).
   - *Empirical gate:* deterministic/derivable events observed in multiple
     batches must land at the same `ts_anchored` (±eps); spread beyond eps is a
     hard error (the determinism claim is false for that event).
2. **Place** each event:
   - root/stochastic → real observed `ts_anchored` from its source batch, tagged
     `stochastic: true` with band overlay `{mean, std, min, max}` from the graph;
   - derivable → exact position via predictor + offset (or its own observed
     `ts_anchored`, which must agree within eps — also gated);
   - deterministic → exact `ts_anchored`.
3. **Reconcile** multi-batch observations: deterministic/derivable events take
   the median and assert spread ≤ eps; stochastic roots keep each distinct
   sample tagged by `source_batch`.
4. **Level/span events** (`PORT_RUNNING*`, `PORT_STALLED*`): place by window
   anchor; their *durations* are deterministic-by-law. Lane naming must key by
   `(col, row, slot)` — resolving characterization finding **I-2** (the
   `(pkt_type, slot)` keying that collapsed cross-tile lanes). The span-law
   check `sum(PORT_RUNNING spans) == words` is a validation gate here.
5. **Emit** `merged.events.json` and `merged.perfetto.json`, sorted by
   `ts_anchored`, every event tagged `source_batch`.

## Output schemas

**`merged.events.json`** — one record per placed event:
```json
{
  "col": 1, "row": 0, "name": "DMA_S2MM_0_START_TASK", "slot": 3,
  "ts_anchored": 6431, "source_batch": 12,
  "class": "stochastic",            // deterministic | derivable | stochastic
  "predictor": null,                // for derivable: {"name":..., "offset":...}
  "band": {"mean": 6667, "std": 1966, "min": 3829, "max": 11438}  // stochastic only
}
```

**`merged.perfetto.json`** — Perfetto JSON for ui.perfetto.dev, same placement;
stochastic events carry the band in their args for inspection.

**Analysis artifacts** (persisted, inspectable, fixture-testable):
`active-events.json`, `derivability-graph.json`, `batch-plan.json`.

## Validation oracle

- **Cross-run skeleton identity:** `join(run_A)` deterministic+derivable events
  == `join(run_B)` (exact `ts_anchored` match). The stochastic roots differ but
  must each fall within their measured band.
- **Coverage:** every active event from Phase 0 appears exactly once in the
  merged trace; none dropped, none duplicated.
- **Bracketing:** every stochastic event has a deterministic/derivable landmark
  on each side; unbracketed ones are reported.
- **Span law:** `sum(PORT_RUNNING spans) == words` per port lane.
- **Anchor gate / reconcile gate:** the ±eps assertions in Phase 3 hold; any
  violation fails the join with a diagnostic.

## Tooling and boundaries

- **`tools/trace-join.py`** (new): Phases 1–3 analysis and merge. Pure function
  of (sweep output + variance report). Standalone; does not capture. Independently
  testable against fixture batch-dirs.
- **`tools/trace-sweep.py`** (extend): a `--plan <batch-plan.json>` mode that
  honors an explicit per-batch slot assignment, replacing catalog-order slicing
  for the join sweep. The existing `_anchor_events` / `_merge_anchored` /
  `_check_grounding_pc_invariance` are refactored/reused — anchoring pinned to
  `PERF_CNT_2`-on-core, invariance check comparing *anchored-relative* values
  rather than absolute PCs.
- **`tools/trace_variance.py`** (existing): supplies the per-event band stats the
  graph and overlays consume. Phase 1 builds on its aggregation primitives.

## Relationship to prior open issues

- **I-1 (anchor self-referential):** resolved — anchor is `PERF_CNT_2` first-fire,
  validated bounded ≤179 anchor noise.
- **I-2 (span path cross-tile name collapse):** resolved within Phase 3 — lane
  naming keyed by `(col, row, slot)`; span law a validation gate.
- **`unsafe_for_pc_join` flag:** the existing check compares *absolute* PCs and
  always trips on boot-offset drift. The join makes it informational; correctness
  comes from anchored-relative alignment plus the Phase 3 gates, not from the
  absolute-PC invariant.
