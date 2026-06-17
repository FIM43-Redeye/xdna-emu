# DMA nondeterminism characterization — results (task #140)

**Date:** 2026-06-16
**Harness:** `tools/trace_variance.py` + `tools/trace-variance-sweep.py` (spec
`docs/superpowers/specs/2026-06-16-dma-nondeterminism-characterization-design.md`,
plan `docs/superpowers/plans/2026-06-16-dma-nondeterminism-characterization.md`).
**Data:** N=20 full-event HW sweeps of `add_one_using_dma` on NPU1, 25 lockstep
batches/run, HW-only, col 0. Output under
`build/experiments/gap140/nondeterminism/add_one_using_dma/`.

## Headline: DMA milestone timing is genuinely stochastic; the backbone is not

Across the 20 contention-free HW runs, the shim DMA `*_TASK` milestone events
are **strongly stochastic**, while the grounding/structural events are
**deterministic** (std 0). Measured on `batch_12` (the richest shim-milestone
batch) across the 20 runs, re-anchored within-tile:

| event | class | n | mean | std | min | max | range |
|-------|-------|---|------|-----|-----|-----|-------|
| DMA_S2MM_0_START_TASK     | stochastic    | 20 | 8498 | 1869 | 3829 | 11438 | 7609 |
| DMA_MM2S_0_START_TASK     | stochastic    | 20 | 8498 | 1869 | 3829 | 11438 | 7609 |
| DMA_S2MM_1_FINISHED_TASK  | stochastic    | 20 | 8498 | 1869 | 3829 | 11438 | 7609 |
| DMA_S2MM_0_STREAM_STARVATION | stochastic | 20 | 8498 | 1869 | 3829 | 11438 | 7609 |
| DMA_S2MM_1_START_TASK     | deterministic | 20 | 0 | 0 | 0 | 0 | 0 |
| DMA_MM2S_0_FINISHED_TASK  | deterministic | 20 | 0 | 0 | 0 | 0 | 0 |
| PERF_CNT_2 (grounding)    | deterministic | 20 | 0 | 0 | 0 | 0 | 0 |

**Caveat — the anchor event is "deterministic" by construction (not backbone
signal).** The milestone path anchors each tile to `min(soc)` over that tile's
milestone events, so whichever event holds the per-tile minimum is subtracted
from itself in every run and forced to std 0. Here that pins `DMA_S2MM_1_START_TASK`
and `DMA_MM2S_0_FINISHED_TASK` (and, on the offline-20 captures,
`DMA_S2MM_1_STREAM_STARVATION`) to 0 and labels them "deterministic" — an
artifact of the reduction, not evidence those events are reliable. The genuine
backbone (`PERF_CNT_2`, `LOCK_STALL`) is std 0 on its own merits; the anchor
event is not distinguishable from it in the current report. The spec called for
anchoring to a stable **grounding** event rather than `min(soc)` (design §2);
the milestone loader does not yet do this. **Consequence for the join work:** do
not treat a min-anchor event as a reliable correlation key without re-checking
it against a true grounding anchor. (Whole-branch review, 2026-06-16, finding I-1.)

**What the identical stochastic stats mean (real, not an artifact).** The shim
`*_TASK` milestones are co-timestamped by HW into two clusters at BD-task
boundaries — an early cluster (`MM2S_0_FINISHED`, `S2MM_1_START`) and a late
cluster (the other six). The first-occurrence-anchored reduction places the
early cluster at 0 and the late cluster at the inter-cluster interval, so the
four late events share one value: **the inter-cluster interval == the actual
data-movement duration**. That interval is what swings 3829→11438 cycles
(std ~1869) run-to-run — a **~3x spread on the same kernel with zero
contention**. This is the silicon-level nondeterminism that, comparing EMU
against a *single* HW capture, manufactures the corpus-wide 171/212 DIVERGE.
The offline-20 pass (single-config captures) agreed qualitatively: every
stochastic event was a DMA milestone, the backbone deterministic.

**Conclusion for #140:** the 171 is dominated by DMA milestone-timing jitter
that the silicon itself does not reproduce run-to-run. The deterministic
backbone is genuinely deterministic. A comparator that masks the stochastic
DMA intervals and holds the backbone strict will reclaim much of the 171 — the
exact decomposition the spec set out to produce.

## The deeper purpose: reliable events are cross-batch correlation anchors

The 8-slot hardware trace limit means no single run can trace every event. The
sweep beats this by tracing different event subsets per batch and **merging**
them into one complete every-event trace — but each batch is a separate HW run
with its own trace clock, so the merge is only possible by aligning batches on
events that fire *identically* across runs. **The deterministic events ARE the
correlation keys.** This characterization therefore does double duty: it both
quantifies the stochasticity (above) and identifies which events are reliable
enough to anchor the cross-batch join.

The sweep already exposes the live problem here: every run's manifest reports
`unsafe_for_pc_join: true` with `reason: "grounding event PERF_CNT_2 PC
drifted"`. So the single anchor the sweep currently relies on (`PERF_CNT_2`) is
itself not reliable enough to join on. Finding better anchors — or a *set* of
deterministic events that jointly pin the timeline — is the natural next step,
and it is exactly what the deterministic/stochastic classification feeds.

## Open issue: the span-law path does not reproduce sum==64

The deferred strict span-law proof (`sum(PORT_RUNNING spans) == words == 64`,
per the 2026-06-16 `port-cadence-metric-was-frame-records` finding) does **not**
hold through our `load_spans` on freshly-decoded perfetto. On `batch_05` (six
memtile `PORT_RUNNING` ports), re-decoded via `parse-trace.py --out-perfetto`:

```
PORT_RUNNING_0: sum=14982   PORT_RUNNING_1: sum=14982   (expected ~64)
PORT_RUNNING_2: sum=14934   PORT_RUNNING_3: sum=14934
PORT_RUNNING_4: sum=153     PORT_RUNNING_5: sum=153
```

Two concrete leads:
1. **Preamble span inclusion.** Each inflated lane's first B/E pair is a giant
   ~7300-cycle span (`327559→334879`) before the real 1-cycle-per-word port
   activity; our grouping sums it in. The finding's `be_spans` got a clean 64,
   so our span path diverges from the established reference here.
2. **Cross-tile name collapse (root cause of the byte-identical lanes).**
   `build_lane_name_map` keys names by `(pkt_type, slot)`, dropping col/row. When
   two physical tiles share a pkt_type, the dict-comprehension keeps only the
   last tile's name per `(pkt_type, slot)`, and *every* perfetto `pid` of that
   pkt_type is then mapped to the same name set — so distinct lanes
   (`PORT_RUNNING_0`==`_1`, `_2`==`_3`, `_4`==`_5`) collapse to one name and
   `load_spans_from_events` `.extend()`s their spans together, inflating the sum.
   The fix is to key by `(col, row, slot)` (or thread the tile coordinate through
   from the perfetto `process_name`, which already carries row/col). Reconcile
   against `be_spans`. (Whole-branch review, 2026-06-16, finding I-2.)

This is a real harness bug (not a stale fixture, as in Task 3). The HW span-law
itself is already established (the prior finding); proving it *through this
harness* is an open follow-up, folded into the correlation/join work below.

## Status and next step

- **Built and validated:** the milestone variance path (Tasks 1-2, 4-6) — the
  headline decomposition is sound, cross-checked between the offline-20 and the
  fresh-20 sweeps.
- **Open:** the span path (`load_spans`) inflates held-level sums vs the
  `be_spans` reference (above).
- **Next (Maya, 2026-06-16):** pivot to the cross-batch **join** — use the
  reliable (deterministic) events identified here to merge the 25 per-batch
  sweeps into one complete every-event trace, and find anchors better than the
  drifting `PERF_CNT_2`. The span-path reconciliation rides along as a sub-task
  (correct spans are needed for the joined trace's held-level events).

Artifacts: `build/experiments/gap140/nondeterminism/add_one_using_dma/`
(`run_00..19/`, `batch12-milestones/report.{md,json}`, `sweep.log`); fresh
perfetto for the span-law diagnostic at `run_{00,05,10}/batch_05/hw/trace.perfetto.json`.
