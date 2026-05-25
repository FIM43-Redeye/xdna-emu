---
name: 'Trace ts vs soc -- cross-test cycle comparisons must use soc, not ts'
description: The trace decoder emits both `ts` (trace-position-inflated, schema-compatible with HW's reordered packet stream) and `soc` (true SoC cycle). Using `ts` for cycle measurements across tests with different LOCK_STALL emission counts produces phantom multi-thousand-cycle "regressions" that are pure event-position arithmetic. P3 chosen: tighten measurement discipline now, defer proper perf-counter modeling to a HW-ground-truth phase.
type: project
---

# Trace `ts` vs `soc` -- a measurement gotcha that tricked us twice

## TL;DR

The trace decoder produces two cycle-like fields per event:

- **`soc`** -- true SoC cycle when the event fired. This is what you
  want for "when did this happen on the silicon clock."
- **`ts`** -- `soc + 1 + events_before_in_packet_stream`. Schema-
  compatible with HW's reordered trace packet stream, suitable for
  in-test event-ordering but **NOT for cross-test cycle comparison**.

If test A has 2701 LOCK_STALL events before its kernel-acquire event
and test B has 44, the same SoC-cycle kernel-acquire shows up at
`ts = soc + 2702` on A and `ts = soc + 45` on B. Diffing those `ts`
values across A and B yields a fake 2657-cyc "regression." That is
exactly the trap we fell into today.

We also fell into a version of it on 2026-05-11. The "DMA pipeline 2.4x
too fast" framing in `docs/archive/findings/2026-05-11-emu-dma-pipeline-too-fast-misses-stalls.md`
was retracted *because of this same artifact*, but the code change it
motivated (LOCK_STALL_TRACE_PERIOD: 1024 -> 1) was left in place. That
left EMU over-emitting LOCK_STALL by ~60x on tests with long stall
windows -- which is what re-tripped us today.

## The trap, concretely

Cycle-accuracy work on `_diag_phase_b_add_one_instrumented.chess`,
re-baselining after the 2026-05-25 backpressure / per-channel
`stream_out` refactor. Phase C (2026-05-10) had decomposed the
HW/EMU gap into 5 stages and found that stages 1+2 owned 98.9% of
it (shim DDR cold-start under-modeled).

Today's re-baseline showed:

| Stage | HW (ts) | Phase C EMU (ts) | EMU now (ts) |
|-------|------:|------:|------:|
| 1+2: shim -> memtile S2MM | 2699 | 162 | 2027 |
| 3: memtile S2MM -> MM2S | 13 | 7 | 16 |
| 4: memtile MM2S -> compute S2MM | 21 | 9 | 10 |
| 5: compute S2MM -> kernel acquired | 15 | 4 | **2690** |

Stage 5 had apparently *exploded* from 4 cyc to 2690 cyc -- a phantom
regression that looked enormous and suggested a real new bug in the
lock-release path. Suspects flagged: chained-BD residual fixes (89ab81e,
589af07), the per-route SS backpressure (92fac90), the per-channel
`stream_out` refactor (36dc890).

The actual data on the kernel-acquired event:

```
ts=8659, soc=5973  INSTR_LOCK_ACQUIRE_REQ
```

`soc=5973` is just 5 cyc after `compute_s2mm_done` at `soc=5968`. Stage 5
in SoC terms is **5 cyc**, basically unchanged from Phase C's 4 cyc.

The 2686-cyc inflation came from 1378 LOCK_STALL events sitting in the
packet stream ahead of the kernel-acquired event. With `ts = soc + 1 +
events_before`, that inflated the `ts` field by exactly the LOCK_STALL
count.

Re-decomposed using `soc`:

| Stage | HW | EMU (SoC) | Gap |
|-------|---:|---:|---:|
| 1+2 | 2699 | 2027 | 672 |
| 3 | 13 | 15 | -2 |
| 4 | 21 | 11 | 10 |
| 5 | 15 | 5 | 10 |
| Total | 2748 | 2058 | 690 |

So: no stage-5 regression, no chained-BD residual fix to blame. The
cold-start primitive (`shim_ddr_cold_start_cycles=1500`) closed 74% of
the original stage-1+2 gap, exactly as Phase C predicted.

## Why this trap exists at all

The trace decoder reproduces HW's behavior of writing trace packets to
host memory in a serialized, reordered stream. The `ts` field is
generated so that downstream consumers can reconstruct event ordering
within a single test's trace stream. For that purpose, monotonicity
matters and uniqueness is convenient -- hence `ts = soc + 1 +
events_before_in_stream`.

But the field's name reads like a timestamp, and on tests with sparse
trace events `ts ~ soc` so the difference is invisible. That's the
trap: you don't notice the semantics until you compare across tests
or across emission-count regimes within one test.

Documented in `docs/archive/findings/9cf60f9` (commit
`9cf60f9 findings: trace decoder ts is SoC + 1 + events_before, not
SoC cycle`).

## Why P3 (don't change emission) rather than (P1) period=1024

`add_one_using_dma` on 2026-05-11 measured HW LOCK_STALL = 24 events,
EMU = 22 events with period=1. That suggested EMU and HW agree on
*per-cycle* emission for short-stall kernels.

Phase C measured HW LOCK_STALL = 44 events for
`_diag_phase_b_add_one_instrumented`, described as "mostly 1024-cyc
spaced." That suggested *perf-counter-rollover* emission for
long-stall kernels.

These are not contradictory *if* HW's LOCK_STALL emission depends on
the trace controller's perf-counter configuration -- which it does,
on real silicon. The two tests configure perf counters differently.
`add_one_using_dma`'s short per-iter stalls (~hundreds of cycles)
look the same under either emission model. `_diag_phase_b_*`'s long
diagnostic stalls (5921 cyc) look very different.

So flipping period back to 1024 today would over-correct: it would
match HW on diagnostic tests while breaking the close match on
normal tests like `add_one_using_dma`. We need HW-side data on more
tests before committing to either fixed value.

## P3 in practice

Two discipline rules:

1. **Cycle measurements across tests use `soc`, not `ts`.** This is
   the rule that would have saved us this afternoon, and the one that
   would have saved 2026-05-11 too.

2. **Stage-decomposition tooling should consume `soc` by default.**
   If we ever script the trace-comparison logic for the bridge sweep
   matrix (the "B-cycle option C" direction we discussed), it must
   read `soc` from the events JSON, not `ts`.

The ad-hoc Python comparator I used today is not persistent tooling;
when we build that, the rule above is the design constraint. Until
then, just remember the gotcha.

## What "do it right" looks like (eventual P2)

Model the perf-counter machinery properly so EMU LOCK_STALL emission
reflects the test's actual trace-controller configuration:

1. Read each tile's PERF_CTRL0 / PERF_CTRL1 from the CDO load (which
   events to count, and the threshold).
2. When the configured "ACTIVE_CORE" event is asserted, tick the
   counter. The core is "active" while stalled (per AM020 -- stall
   doesn't reset core-active status), so the counter ticks during
   WaitingLock just like HW.
3. On rollover, fire the configured event (typically PERF_CNT_0, slot 0)
   AND the correlated state event (LOCK_STALL when the core happens
   to be in lock-stall, MEMORY_STALL when in memory-stall, etc.).
4. The "initial entry edge" of LOCK_STALL stays where it is today
   (`cycle_accurate.rs:821`).

This is structural -- a perf-counter subsystem, gated by trace-
controller config. Not a knob bump.

**Prerequisite for P2**: HW LOCK_STALL counts on a calibration set of
tests, decoded under the correct ts/soc understanding. The 2026-05-11
"~4400 LOCK_STALL events" count cannot be trusted because that session
made the same `ts` vs `soc` mistake we did. The Phase C count
(44 events on `_diag_phase_b_add_one_instrumented`) is more
credible because it predates the framework regression and matches
the "~1024-cyc periodicity" qualitative observation. But we should
remeasure all of it once we're committing to the modeling work.

## See also

- `9cf60f9` -- original ts semantic finding ("ts is SoC + 1 + events_before")
- `docs/archive/findings/2026-05-10-phase-c-stage-attribution.md` --
  Phase C decomposition, the comparison target this afternoon
- `docs/archive/findings/2026-05-11-emu-dma-pipeline-too-fast-misses-stalls.md` --
  the retracted prior framing; the LOCK_STALL_TRACE_PERIOD=1 change
  it motivated is now suspect
- `docs/coverage/cycle-accuracy-mission.md` -- item being added for the
  P2 perf-counter modeling work
- `src/interpreter/core/interpreter.rs:97-101` -- `LOCK_STALL_TRACE_PERIOD`
  constant, currently 1, to be revisited under P2
- `src/interpreter/core/interpreter.rs:728-743` -- periodic emit logic
- `src/interpreter/execute/cycle_accurate.rs:818-822` -- initial-entry emit
