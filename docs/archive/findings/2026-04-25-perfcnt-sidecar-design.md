# C.2 Findings: perfcnt-span sidecar (2026-04-25)

## Original framing

`docs/observability-leads.md` lead #2:
> Performance counters with start/stop/reset events. Four 32-bit
> counters per tile. Hard constraint: NPU instruction set has no
> Read32 op, so we cannot read counter values back to host through
> the runtime sequence. The only way to surface a perfcnt measurement
> in our XRT path is via the PERF_CNT_N event into the trace.

Action priority: "Independent ground truth for 'did the kernel body
take a constant number of cycles?'"

## Status: design uncovered, scoped, deferred

Per the lead's "downgrade" finding (perfcnt and trace share the
same 64-bit free-running timer), **perfcnt-via-trace is not an
independent cycle source.** What it can still buy us:

1. **Deterministic tile-internal trace stop** -- route PERF_CNT_0 to
   Trace_Control0's stop_event slot, with perfcnt configured to fire
   at threshold N cycles after BROADCAST_15. Replaces host-fired
   BROADCAST_14 (variable latency) with a tile-internal "exactly N
   cycles after start" stop. Fixes the broadcast-latency variance
   surfaced in C.1 findings.
2. **Periodic timing markers** -- small threshold (e.g., 100), counter
   not auto-resetting. Fires PERF_CNT_0 every 100 cycles, gives
   cycle-stride markers in the trace.
3. **Threshold-as-bound check** -- threshold = expected cycle count;
   PERF_CNT firing iff actual count met threshold.

## The implementation prerequisite

All three uses require **inserting new Write32 ops into insts.bin to
program perfcnt registers**. Existing insts.bin doesn't write to
perfcnt; CDO preamble doesn't either (verified by scanning init and
enable blobs).

`tools/trace-patch-events.py` currently *modifies* existing Write32
values (the trace_control register's start_event/stop_event/mode
fields). It doesn't *insert* new ops. Insertion would need:

1. Parse insts.bin header layout (`num_ops`, `total_size` fields).
2. Decode existing op stream to find a safe insertion point (e.g.,
   right before the runtime-sequence body so perfcnt is enabled
   before kernel runs).
3. Construct new Write32 ops:
   - `0x31500` (PerfCounter0_Control) -- start/stop event selectors.
   - `0x31580` (PerfCounter0_Threshold) -- N for one-shot fire.
   - `0x31520` (PerfCounter0_Value) -- 0 to reset.
4. Insert the ops, update header `num_ops` and `total_size`.
5. Re-write the binary.

Each compute tile in the partition needs its own Write32 sequence
(perfcnt registers are per-tile).

This is a real ~1 day implementation task -- doable but not on the
critical path until a workload surfaces where the C.1 broadcast-latency
variance (~7%) actually matters.

## Trade-off vs the C.1 broadcast finding

C.1 settled on: broadcast wins for span stability under hwctx-reuse,
with ~7% residual variance from broadcast-fire latency. The
cycle-diff classifier's bounds (default `[0.5, 2.0]`, per-test
overrides) easily accommodate that.

Implementing perfcnt-based stop would shrink the ~7% residual to
near-zero (deterministic tile-internal counter trip), giving us
tighter MATCH/DRIFT discrimination. But the value of that depends on
finding workloads where that tighter discrimination changes a
classification. We don't have evidence yet.

## What this means for thread C

C.2 closed as designed-not-implemented. Deferred until either:
- a workload's MATCH/DRIFT classification flips because of broadcast
  latency variance (concrete evidence we need tighter bounds), or
- we want the periodic-timing-markers use case for richer trace data.

If we ever build the insts.bin Write32 insertion machinery for any
other reason, this becomes a near-free follow-on commit since the
mechanism is the same.

## Pointers for the implementation pass

- `tools/trace-patch-events.py` -- the natural home for an
  `insert_perfcnt_init` function (next to the existing `patch_*`
  functions). Header field offsets and op layout already understood.
- `aie-rt/driver/src/perfcnt/xaie_perfcnt.h` -- reference for the
  control word layout (`StartEvent` / `StopEvent` field positions,
  `ResetEvent` field).
- AM025 register database JSON has the exact bit positions for
  `PerfCounter0_Control` field decomposition.
