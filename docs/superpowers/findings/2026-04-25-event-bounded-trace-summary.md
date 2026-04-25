# C.1 Findings: event-bounded trace prototype (2026-04-25)

## Summary

The event-bounded trace prototype was already partly executed on
2026-04-24 (`build/experiments/cdo-preamble-test/test_event_bounded.py`)
and the results recorded in `docs/observability-leads.md` lead #1.
This document summarizes the conclusions visibly and records what's
still unexplored.

## The mechanism

Trace_Control0's start_event and stop_event fields gate when the
trace unit captures. mlir-aie's compiled output uses
`start_event=BROADCAST_15 (122)` and `stop_event=BROADCAST_14 (121)`,
fired by the host via shim control packets. Latency variance on the
broadcast path translates 1:1 into trace-window-span jitter.

`tools/trace-patch-events.py` lets us override these fields without
rebuilding the xclbin (and adds a `--mode` knob for trace mode 0/1/2).

## Results: 3 of 5 options exercised

`add_one_objFifo` test, N=20 iterations under hwctx-reuse + CDO
preamble (kernel ~3160 cy):

| Mode | start | stop | mean cy | sd cy | range cy | n_events sd |
|------|-------|------|---------|-------|----------|-------------|
| broadcast (default) | BROADCAST_15 | BROADCAST_14 | 3160 | **221** | 1049 | **0.0** |
| call-return | INSTR_CALL=35 | INSTR_RETURN=36 | 3109 | 199 | 772 | 1.2 |
| call-disabled | INSTR_CALL=35 | CORE_DISABLED=29 | 9911 | 2664 | 8944 | 1.6 |
| **PERF_CNT-based** | BROADCAST_15 | PERF_CNT_0 (threshold N) | -- | -- | -- | -- |
| **PC-trace (mode 1)** | n/a (changes axis) | n/a | -- | -- | -- | -- |

### Verdict on the first three

**Broadcast wins for event-count stability** (sd=0.0 -- exactly 146
events every iteration), and is competitive with call-return for span
(220 vs 200 cy sd). Call-disabled is broken under hwctx-reuse:
CORE_DISABLED doesn't fire predictably, span runs 3x longer with 12x
more variance.

The remaining ~7% span jitter is residual broadcast-latency variance
on the broadcast path. We can't reduce it without re-anchoring on
something tile-internal.

### PERF_CNT-based: would need perfcnt-enable plumbing (deferred)

Concept: replace `stop_event=BROADCAST_14` with `stop_event=PERF_CNT_0`,
configured to fire at threshold N cycles after BROADCAST_15. This
gives a tile-internal "exactly N cycles after start" stop, removing
the host-broadcast variance.

Why it's deferred: existing insts.bin doesn't write to perfcnt
registers; CDO preamble doesn't either. Implementing this means
extending `tools/trace-patch-events.py` (or adding a sibling tool) to
*insert* new Write32 ops into insts.bin -- header has `num_ops` and
`total_size` fields that must be updated. Worth doing when we
genuinely need fixed-window traces for some workload; not worth doing
on speculation. The `~7%` residual variance is acceptable for the
cycle-diff classifier under current bounds.

### PC-trace (mode 1): orthogonal, deferred to A.2

Mode 1 doesn't reduce span variance -- it changes what the trace
records. Each event captures the PC at fire-time instead of a cycle
delta. That makes (event, PC) a stable identity across runs, which is
exactly what A.2's PC-anchored joining needs.

Tracked under A.2 (thin defaults + sweep + PC-anchored joining).
That's where it belongs structurally; running mode-1 captures here
without the join machinery would be wasted work.

## Per the user direction (2026-04-25)

> I think we want to *try* for every option we've got available, but
> if broadcast is the only option, we can accept it.

We have data on three options. Broadcast is near-optimal among them.
The two unexplored options are real but each requires substantial
prerequisites (perfcnt insertion plumbing, or A.2's join framework).
**Verdict: accept broadcast as the default for the cycle-diff
pipeline.** Re-open if a workload surfaces where the residual variance
matters.

## What's actionable today

Nothing requiring code changes. The current bridge-test pipeline uses
broadcast by default (because mlir-aie's lowering produces it), and
the cycle-diff classifier's bounds (default `[0.5, 2.0]`, per-test
overrides via `cycle-drift-overrides.txt`) already accommodate the
~7% broadcast-latency variance.

If we want to formalize the broadcast-as-best decision in code, the
most natural place is a comment in `mlir-trace-inject.py` referencing
this finding. Optional polish.
