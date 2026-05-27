---
name: 'Phase 2c: dispatch_overhead recalibration + non-stalling controller refactor'
description: 'Recalibrated five shim-DMA / dispatch constants against multi-run K-sweep HW data after restructuring the EMU controller model to decouple controller rate-limiting from per-instruction retirement. Result: steady-state inter-START gaps now agree with HW to 0.3-5.3% across MM2S/S2MM/K=4/K=8; S2MM cold-start gap[0] within 2-4%. Surfaced a previously hidden HW behavior the model still does not capture: per-task transfer durations DECREASE through a chain (K=8 MM2S task[0]=1739 cyc -> task[7]=370 cyc), a warm-up transient that explains the MM2S gap[0] sign-flip we had attributed to pipelining. Decomposing the transient further is blocked on Phoenix: AIE_RW_ACCESS cannot program perf counters, and MM2S-direction level trace events wedge the amdxdna mgmt mailbox after ~30s of sustained dispatch.'
type: project
---

# Phase 2c: dispatch_overhead recalibration + non-stalling controller refactor

## TL;DR

Phase 2c restructured the EMU's controller model and re-derived
five shim-DMA / dispatch timing constants against the multi-run
HW K-sweep data captured in Phase 2a. Net outcome:

- **Steady-state inter-START gaps** (MM2S/S2MM, K=4/K=8) now
  match HW to **0.3-5.3%**, mean drift ~3%.
- **S2MM cold-start gap[0]** within **2-4%** of HW (K=4, K=8).
- The **K=2 "anomaly"** is no longer an anomaly: it's the cold-
  to-warm-1 transition with no chain to amortize against, and
  the per-task data shows it sitting on the same curve as larger
  K values.

Three structural changes shipped:

1. **Non-stalling controller refactor** (`2c.A`): the controller
   rate-limit gate is decoupled from instruction retirement.
   Task_Queue writes block on the gate; non-TQ writes (BD config,
   etc.) issue freely during controller-busy time. Mirrors HW
   behavior where the controller is not the bottleneck for other
   register writes.

2. **Q-aware dispatch_overhead** (`2c.1`): two distinct overhead
   constants based on Task_Queue state at the time of the TQ
   write -- `dispatch_overhead` (cold, queue empty + channel
   idle) and `dispatch_overhead_pipelined` (queue busy). The
   model previously used a single constant for both regimes.

3. **5-constant recalibration** (`2c.3`): cold-start and per-task
   shim-DDR constants re-derived against multi-K HW data, fixing
   the prior K=8-only fit that under-modeled K=1 by 50-58%.

Phase 2c left one HW behavior structurally unmodeled and brought
two calibration-source paths to dead ends. See "Remaining gap" and
"Blocked calibration sources" below.

## What was wrong before

The prior model had three connected issues:

### 1. Single dispatch_overhead conflated cold vs pipelined

The original `dispatch_overhead = 2500 cyc` constant was applied
to every Task_Queue write regardless of context. HW data shows
the gap between consecutive START_TASK events on a single shim
channel is bimodal:

- **Cold dispatch** (queue empty, channel idle): ~3050 cyc
- **Pipelined dispatch** (queue already has work in flight):
  ~520 cyc (and on MM2S K=8 frequently *negative*, i.e., the
  next START fires before the prior FINISHED)

A single constant cannot reproduce both regimes.

### 2. dispatch_overhead was charged as retirement cost

The cost was added to instruction retirement cycles, which meant
the EMU stalled the entire instruction stream during a dispatch.
Real HW does not work that way: the controller is rate-limited
on TQ enqueues, not on register accesses generally. BD config
writes for the next dispatch can proceed while the controller is
mid-dispatch on the previous one. Modeling it as retirement cost
forced the EMU to a more serial execution than HW exhibits.

### 3. Shim-DDR constants fit against K=8-only

The cold-start and per-task overheads were calibrated against
K=8 steady-state data (the dominant signal in the original
single-run trace). That made K=8 totals look right but
under-modeled K=1 transfer duration by 50-58%:

| Cell | Old EMU | HW (N=50) | Drift |
|---|---:|---:|---:|
| K=1 MM2S transfer | 822 | 1654 | -50% |
| K=1 S2MM transfer | 243 | 584 | -58% |

K=1 is the cold-start case in pure form (no chain to amortize),
so the gap was specifically in the cold-start constants.

## What changed

### Non-stalling controller (`src/npu/executor.rs`, `2c.A`)

Added two fields to `NpuExecutor`:

```rust
npu_cycle: u64,                    // bumps once per try_advance
controller_next_taskq_cycle: u64,  // rate gate
```

Before executing each instruction, we check whether it's a
Task_Queue write (via `classify_task_dispatch`). If so:

```rust
if dispatch_was_idle.is_some() && self.npu_cycle < self.controller_next_taskq_cycle {
    return AdvanceResult::Blocked;
}
```

This stalls only the TQ write itself, not the surrounding
instruction stream. After the write executes, the gate is set
based on cold vs pipelined state:

```rust
let overhead = if was_idle { self.cycle_model.dispatch_overhead }
               else { self.cycle_model.dispatch_overhead_pipelined };
self.controller_next_taskq_cycle = self.npu_cycle.saturating_add(overhead);
```

The retirement cycles calculation no longer includes
`dispatch_overhead` -- it lives entirely in the gate now.

### Q-aware classification (`2c.1`)

`classify_task_dispatch(col, row, offset, device) -> Option<bool>`
returns `Some(true)` when the target shim channel is idle and
its queue is empty (cold), `Some(false)` when the queue is busy
(pipelined), and `None` if the address is not a Task_Queue write.

The pre-execute classification captures the state BEFORE the
write lands, so the cost charged is the cost of *this* dispatch,
not the next one.

### Recalibrated constants (`crates/xdna-archspec/src/model_builder.rs`, `2c.3`)

Tuned against the multi-run HW K-sweep data:

| Constant | Old | New | Calibration anchor |
|---|---:|---:|---|
| `shim_ddr_cold_start_mm2s_cycles` | 498 | 1330 | HW K=1 MM2S = 1654 |
| `shim_ddr_cold_start_s2mm_cycles` | 0 | 341 | HW K=1 S2MM = 584 |
| `shim_per_task_overhead_mm2s_cycles` | 249 | 325 | HW K=4+ MM2S steady ~400 |
| `shim_per_task_overhead_s2mm_cycles` | 168 | 179 | HW K=4+ S2MM steady 254 |
| `dispatch_overhead` (NpuCycleCost) | 2500 | 3050 | HW S2MM K=8 steady gap |
| `dispatch_overhead_pipelined` (new) | n/a | 520 | HW MM2S K=8 short gap |

Approach: fit per-task and cold-start against K=1 (pure cold) +
K=4+ (pure steady-state) as orthogonal anchors. Steady-state cells
constrain per-task; K=1 minus K=4+ constrains cold-start. Single
linear fit, not iterative.

## Validation

Re-ran the EMU campaign post-recalibration and compared to HW
multi-run baseline (`2026-05-27T04-19-33`, N=50). Per-cell median
delta (EMU vs HW):

### Steady-state inter-START gaps (gap[1..])

| Cell | HW | EMU | Δ% |
|---|---:|---:|---:|
| K=4 MM2S steady | 2803 | 2655 | -5.3% |
| K=4 S2MM steady | 2785 | 2800 | +0.5% |
| K=8 MM2S steady | 2734 | 2655 | -2.9% |
| K=8 S2MM steady | 2876 | 2800 | -2.7% |

### S2MM cold-start gap[0]

| Cell | HW | EMU | Δ% |
|---|---:|---:|---:|
| K=4 S2MM gap[0] | 2562 | 2461 | -3.9% |
| K=8 S2MM gap[0] | 2520 | 2461 | -2.3% |

### What didn't converge: MM2S cold-start gap[0]

| Cell | HW | EMU | Δ |
|---|---:|---:|---:|
| K=4 MM2S gap[0] | **-812** | 1326 | +2138 |
| K=8 MM2S gap[0] | **-433** | 1326 | +1759 |

HW shows *negative* gaps on MM2S (next START fires before prior
FINISHED). EMU produces large positive gaps. Section "Remaining
gap" explains why.

## What we learned about HW (the new finding)

The stall-decomposition aggregator
(`tools/decompose-stalls.py`, committed in `fc31c3f`) revealed a
behavior the dispatch-gap-only view had hidden:

**MM2S per-task transfer durations DECREASE through the chain.**

K=8 MM2S durations from HW (N=50 median, in cycles):

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dur | 1739 | 804 | 497 | 422 | 398 | 399 | 364 | 369 |

S2MM is also affected but much more weakly:

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dur | 509 | 233 | 250 | 250 | 248 | 249 | 249 | 248 |

This is a **warm-up transient** -- the first transfer pays
significant overhead beyond cold-start, and subsequent transfers
asymptote downward over several iterations before reaching
steady state.

### Reframing the MM2S "pipelining" finding

We previously interpreted MM2S `gap[0] = -812` cyc on K=4 (and
similar negatives on K=8) as evidence of aggressive HW pipelining
-- the controller dispatching `task[1]` before `task[0]`
finishes. The per-task data forces a different reading: **gap[0]
is negative because `task[0]` runs long (1739 cyc) while
`task[1]` runs short (804 cyc), so the FINISHED_TASK[0] event
fires LATER than START_TASK[1].** Pipelining is real in the
sense that the controller doesn't wait for task[0] to finish
before issuing task[1], but it's pipelining of *variable-duration
tasks*, not fixed-duration tasks. The EMU's current model assumes
fixed steady-state duration after cold-start, so the negative
gap cannot emerge.

The "K=2 anomaly" is also explained: K=2 S2MM `gap[0]=49` cyc on
HW is just the cold->warm-1 transition with no chain. K>2 cases
all sit on the same per-task duration curve.

## Remaining gap

EMU does not yet model the warm-up transient. After cold-start,
the model charges the same per-task overhead to every task,
giving constant duration. HW's behavior is monotonically
decreasing for at least 3-4 tasks before flattening.

Two possible model shapes (not yet implemented):

1. **Empirical curve fit**: fit task duration as
   `cold_start + per_task + transient(i)` where `transient(i)`
   decays geometrically or exponentially over `i` tasks. Uses
   the existing HW campaign data; no new measurements needed.

2. **Controller setup state machine**: model an explicit "warming
   up" controller state that gradually transitions to "warm",
   with the transition driven by something physical (TLB/cache
   fill, BD prefetch pipeline depth, etc.). Requires a hypothesis
   about the underlying mechanism.

Without controller-internal visibility, only (1) is grounded.
(2) would be speculation.

## Blocked calibration sources

Three paths to "what causes the warm-up transient" closed during
Phase 2c/2d:

1. **AIE_RW_ACCESS perf counter readback** -- blocked by the
   2026-05-26 finding: `read_aie_reg` returns FW-internal ~12000-
   tick-per-call artifacts, `write_aie_reg` to Timer_Control
   silently fails. The XRT API is functional; the FW path is
   broken for this use case.

2. **Trace-event decomposition of MM2S transfer duration** --
   the obvious next step (add MM2S_BACKPRESSURE / MM2S_STALLED_LOCK
   to the shim trace) wedges the amdxdna mgmt mailbox after ~30s
   of sustained dispatch. See
   [`2026-05-27-mm2s-level-trace-event-wedges-mgmt-mailbox`](2026-05-27-mm2s-level-trace-event-wedges-mgmt-mailbox.md).
   S2MM-direction level events work fine, but the warm-up
   transient is much weaker on S2MM, so decomposing the S2MM
   side gives much less signal.

3. **Trace-event level resolution** -- even when we successfully
   capture an MM2S level event in a single-shot run (before the
   wedge), it fires only 1 cyc per task at the transition
   boundary. The trace pipeline cannot resolve sustained
   sub-task stall duration on this hardware. STREAM_STARVATION,
   STREAM_BACKPRESSURE, and STALLED_LOCK all show the same
   1-cyc-per-task pattern.

The remaining candidate (Phase 2e) is the **firmware DPT trace
framework** (`AMDXDNA_DPT_FW_TRACE`), which captures the
controller's own runtime log. Decode cost is high (format is
FW-internal) and may not be feasible at all.

## Artifacts

- Commits: `12ec4e9` (`2c.1`), `6b3c838` (`2c.2`), `8dd8c26`
  (`2c.A`), `c496824` (`2c.3`), `fc31c3f` (decompose-stalls +
  env var)
- Multirun campaign: `tools/multirun-trace-campaign.py`
- HW vs EMU comparator: `tools/compare-dispatch-overhead.py`
- Stall decomposition: `tools/decompose-stalls.py`
- HW baseline: `build/experiments/dispatch-overhead-multirun/
  2026-05-27T04-19-33/` (N=50, gitignored, reproducible)
- Post-recalibration EMU: `build/experiments/dispatch-overhead-
  multirun/2c-3-emu/`
- Related finding:
  [`2026-05-27-mm2s-level-trace-event-wedges-mgmt-mailbox`](2026-05-27-mm2s-level-trace-event-wedges-mgmt-mailbox.md)
