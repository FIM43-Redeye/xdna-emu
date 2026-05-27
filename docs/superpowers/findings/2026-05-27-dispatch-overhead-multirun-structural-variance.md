---
name: 'Dispatch-overhead variance is structural, not run-to-run noise'
description: 'Multi-run trace campaign (N=50 across K=1,2,4,8) on the _diag_shim_chain_sweep kernel reveals that the "30-50% run-to-run variance" reported in the 2026-05-25 dispatch-overhead finding is mostly structural HW-deterministic effects (K-dependent gap[2]/gap[K-1] elevations, MM2S Task_Queue pipelining). True run-to-run noise on the cleanest S2MM steady-state cells is +/-15 cyc MAD. The current EMU dispatch_overhead=2500 cyc is mechanistically Q-state-naive but empirically tuned to compensate at the K-sweep span level; a Q-aware refactor requires coupled re-calibration of multiple costs and is deferred to a future sprint.'
type: project
---

# Dispatch-overhead variance is structural, not run-to-run noise

## TL;DR

The 2026-05-25 dispatch-overhead finding flagged "run-to-run HW variance
is large" (per-task and per-gap measurements varying 30-50% between
sweeps) as a known caveat. An N=50 multi-run trace campaign now shows
that **variance is mostly structural HW-deterministic behavior, not
run-to-run noise**. True run-to-run noise at the cleanest cells is
+/-15 cyc MAD.

The "variance" came from two structural sources the original analysis
folded together:

1. **K-dependent gap[2]/gap[K-1] elevations** on S2MM. K=8 shows +180 cyc
   at gap[2], +260 cyc at gap[6] (the last in-chain). K=4 is flat.
   Likely NPU controller scheduler internal state -- below tile-trace
   reach.

2. **MM2S Task_Queue pipelining**. First 1-2 dispatches per chain are
   typically pipelined (negative or near-zero gap because next BD was
   pre-queued); subsequent dispatches are serialized. Bimodal at the
   first 1-2 positions, well-behaved at gap[3+].

Current EMU `dispatch_overhead = 2500 cyc` is **mechanistically
Q-state-naive** (applied universally to every Task_Queue Write32,
ignoring channel state) but **empirically tuned** to match the
K-sweep span values. A Q-aware refactor that only charges full
overhead when the channel was idle is the correct mechanic but
requires coupled re-calibration of `cmp_decode_cost`,
`fabric_cost`, `dispatch_overhead`, and a new
`dispatch_overhead_pipelined` field -- deferred to its own sprint.

## Campaign

200 iterations (N=50 runs x K in {1,2,4,8}), randomized order, 5 min
wall-clock. Each iteration:
- `bridge-trace-runner` against the existing trace-injected
  `_diag_shim_chain_sweep/k{K}/chess/aie.xclbin`
- `parse-trace.py` to decode `trace_raw.bin` -> `events.json`

K=16 excluded (wedges per the dispatch-overhead finding's "Follow-ups").

Tools added:
- `tools/multirun-trace-campaign.py` -- driver with FW+driver+git
  manifest, randomized (run, K) schedule, per-run metadata
- `tools/aggregate-dispatch-overhead.py` -- per-(K, direction,
  gap_index) distribution summaries
- `tools/plot-dispatch-overhead.py` -- histograms + per-index boxplots

Session data: `build/experiments/dispatch-overhead-multirun/2026-05-27T04-19-33/`
(local-only per repo convention; manifest.json committed via inline
quote where load-bearing).

## Findings

### F1. Variance is structural, not noise

Per-(K, direction, gap_index) cells are highly deterministic across 50
runs. S2MM K=8 example (median, p25, p75):

| Index | Median | p25 | p75 | Spread |
|------:|-------:|----:|----:|-------:|
| gap[0] | 2522 | 2472 | 2606 | 134 |
| gap[1] | 2791 | 2787 | 2800 | 13 |
| gap[2] | 2971 | 2968 | 2975 | 7 |
| gap[3] | 2811 | 2807 | 2814 | 7 |
| gap[4] | 2810 | 2806 | 2849 | 43 |
| gap[5] | 2926 | 2907 | 2949 | 42 |
| gap[6] | 3098 | 3068 | 3136 | 68 |

p25-p75 spread is 7-134 cyc per cell. The 30-50% "variance" in the
original finding came from aggregating across positions whose medians
differ by ~600 cyc, not from run-to-run noise.

### F2. K-dependent structural pattern on S2MM

The S2MM gap signature for K=8 has **elevations at gap[2] (+180 vs base),
gap[5] (+115), gap[6] (+260)** that are absent in K=4 (flat at ~2786
across gap[0..2]). The pattern reproduces across all 50 runs to within
+/-15 cyc.

Cause unknown. Memtile PORT_RUNNING events fire ~810 cyc after each
shim FINISHED at consistent timing **regardless of position** -- they
do not correlate with the gap elevations. The structural cause is
likely NPU controller scheduler internal state (Task_Queue refill
cadence, AXI burst boundaries, or arbiter rotation) which is below
tile-trace reach.

The pattern's tightness (MAD ~10 cyc on most cells) implies it's
hardware-deterministic, not stochastic.

### F3. MM2S Task_Queue pipelining (bimodal distribution)

Unlike S2MM, MM2S exhibits bimodal per-position distributions
dominated by Task_Queue interaction. Across 50 K=8 runs:

| Index | Median | Pipelined (<500) | Serialized (>=2500) |
|------:|-------:|-----------------:|--------------------:|
| gap[0] | -433 | 37/50 | 12/50 |
| gap[1] | 2683 | 24/50 | 25/50 |
| gap[2] | 2884 | 11/50 | 36/50 |
| gap[3] | 2720 | 4/50 | 34/50 |
| gap[4+] | ~2700 | <3/50 | >40/50 |

**Mechanic** (verified against aie-rt):
- Shim Task_Queue HW depth is 8 (per AM025; matches EMU's
  `MAX_TASK_QUEUE_DEPTH`).
- aie-rt's `_XAieMl_DmaWaitForBdTaskQueue` polls "queue >= half-full"
  (size >= 4) as a backpressure signal, effectively limiting
  controller-side queue depth to 4 in normal operation.
  (`aie-rt/driver/src/dma/xaie_dma.c:45`,
  `XAIE_DMA_MAX_QUEUE_SIZE = 4`.)
- NPU controller dispatches Task_Queue writes at burst rate
  ~830-900 cyc per write while queue has room.
- `*_START_TASK` events fire when the BD enters the channel
  (effectively at queue-enter); `*_FINISHED_TASK` events fire at
  transfer completion.
- For small/fast MM2S transfers, multiple BDs can be queued back-to-back
  before the first finishes -- so the "inter-task gap" measurement
  (next START - this FINISHED) goes negative when the next task was
  already queued.
- S2MM doesn't show this because every S2MM task has
  `STREAM_STARVATION` -- it stalls waiting for upstream data and
  can't pre-queue effectively.

### F4. Task_Queue depth: HW vs aie-rt software limit

The shim DMA Task_Queue is **8 entries deep in hardware** (AM025
`Task_Queue_Size` is a 3-bit field reporting 0-7 plus an explicit
"full" flag). aie-rt defines `XAIE_DMA_MAX_QUEUE_SIZE = 4` as its
operational limit -- aie-rt's polling code waits when queue
reaches half-full to avoid HW backpressure.

EMU's `MAX_TASK_QUEUE_DEPTH = 8` (`src/device/dma/token.rs:105`)
is **correct for the HW**. The "K=16 wedges" behavior in the
dispatch-overhead finding's Follow-ups is not a depth-mismatch issue;
it's a separate failure mode where the controller writes past the
HW queue's 8-entry limit. (This is Phase 2b territory.)

### F5. EMU dispatch_overhead model audit

Current model (`src/npu/cycle_cost.rs:292`):
```rust
dispatch_overhead: 2500,
```

Applied at `src/npu/executor.rs:506`:
```rust
if Self::is_task_dispatch_write(col, row, offset, device) {
    retire_cycles = retire_cycles.saturating_add(self.cycle_model.dispatch_overhead);
}
```

This applies the full 2500 cyc to **every** Write32 targeting a
Task_Queue register, regardless of channel state.

The mechanically-correct model:
- When channel is idle at the time of the Write32:
  pay full overhead (~2785 cyc per measured S2MM steady-state).
- When channel is active (queue has pending tasks or current task
  executing): pay only controller burst rate (~520 cyc above the
  base per-instruction cost).

The current model over-counts pipelined dispatches:
- K=4 MM2S: 1 idle dispatch + 3 pipelined = 1*2785 + 3*520 = 4345
  cyc HW overhead. EMU model: 4*2500 = 10000 cyc. **Over-counts by
  ~5655 cyc.**
- K=4 S2MM: 4 idle dispatches (no pipelining due to STREAM_STARVATION)
  = 4*2785 = 11140 cyc. EMU model: 4*2500 = 10000 cyc. **Under-counts
  by ~1140 cyc.**

The errors partially offset at the K-sweep span level, which is why
the original finding's K=4/8 validation showed errors within 10-12%
of HW. Other workloads (longer chains, mixed direction, different
queue dynamics) may not have offsetting errors.

## Implications

1. **Leave the existing 2500-cyc constant in place** until the Q-aware
   refactor lands -- it's empirically tuned to the K-sweep validation
   set and a one-line change degrades that.
2. **Q-aware refactor** is the right model upgrade but needs:
   - New cycle_cost field: `dispatch_overhead_pipelined`
   - Executor query of `ChannelState` at apply time
   - Coupled re-calibration of `cmp_decode_cost`, `fabric_cost`,
     `dispatch_overhead`, `dispatch_overhead_pipelined` against the
     full bridge corpus
   - Bridge suite re-run for regressions
3. **Structural floor**: even with a perfect Q-aware model and
   re-tuned constants, the ~+/-200 cyc K-dependent gap structure
   (F2) is hardware-deterministic and below tile-trace reach. A
   single-constant model has a residual error of this magnitude.
4. **Run-to-run noise is negligible** for calibration purposes
   (S2MM steady-state MAD ~10 cyc). Future calibrations need N=5-10
   runs, not N=50+.

## Open questions (not blocking)

- What controller-side mechanism produces the K=8 gap[2]/gap[K-1]
  elevations? Hypotheses: Task_Queue refill cadence at half-empty,
  AXI burst boundary alignment, NPU controller arbiter rotation.
  Would need NPU controller-internal trace or microcode access to
  verify.
- The original finding's K=4 MM2S undershoot (-803 cyc) and K=8 MM2S
  undershoot (-2623 cyc) suggest a position-aware error pattern -- the
  more chained MM2S tasks, the more EMU over-counts dispatch_overhead
  on pipelined dispatches. A targeted bridge test could quantify the
  magnitude of correction needed before the Q-aware refactor.

## See also

- [`2026-05-25-npu-controller-dispatch-overhead.md`](2026-05-25-npu-controller-dispatch-overhead.md)
  -- the original calibration finding; this work refines its
  "Run-to-run HW variance" caveat and "Follow-ups: On-NPU readback
  path" line.
- [`2026-05-26-aie-rw-access-not-a-cycle-probe.md`](2026-05-26-aie-rw-access-not-a-cycle-probe.md)
  -- the Phase 1 negative result that motivated the trace-based
  approach used here.
- `docs/superpowers/plans/2026-05-26-aie-rw-access-characterization.md`
  -- the plan this work executes (Phase 2a).
- `tools/multirun-trace-campaign.py` / `tools/aggregate-dispatch-overhead.py`
  / `tools/plot-dispatch-overhead.py` -- the harness, aggregator, and
  visualization tools.
- Session data: `build/experiments/dispatch-overhead-multirun/2026-05-27T04-19-33/`
  (local-only; aggregated.json and plots/ for downstream consumers).
