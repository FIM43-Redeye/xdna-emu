---
name: 'Shim stage 1a/1b residual: cold-start knob cannot fix wrong-shape model'
description: After the shim_ddr_cold_start_cycles primitive closed 74% of the Phase C stage 1+2 gap, a 672-cyc residual remains -- structurally not closable by the cold-start knob alone. The remaining gap is the wrong SHAPE of the model: EMU treats cold-start as pure pre-data delay then streams fast; HW's task duration extends over the streaming because the DDR rate IS the bottleneck. Stage 1a (shim dispatch -> shim done) and stage 1b (shim done -> memtile S2MM done) are anti-correlated under the cold-start knob -- no setting closes both. Real fix needs shim streaming throughput modeling, which needs HW ground truth.
type: project
---

# Shim stage 1a/1b structural limit -- why cold-start tuning saturates here

## TL;DR

The `shim_ddr_cold_start_cycles=1500` primitive (commit `3357b7c`,
following Phase C's 2026-05-10 closure plan) closed 74% of the
`_diag_phase_b_add_one_instrumented.chess` stage 1+2 gap -- from
2537 cyc to 672 cyc. The remaining 672 cyc cannot be closed by the
cold-start knob alone, because the model has the **wrong shape**.

| Sub-stage | HW | EMU (SoC) | Gap |
|-----------|---:|---:|---:|
| 1a: shim dispatch -> shim FINISHED_TASK | 1682 | 2054 | -372 (EMU slower) |
| 1b: shim FINISHED_TASK -> memtile S2MM FINISHED_BD | 1017 | -27 | +1044 (EMU wrong direction) |
| Total 1+2 | 2699 | 2027 | 672 |

Stage 1b is *negative* in EMU -- the memtile S2MM finishes 27 cyc
before the shim DMA reports task-done. In HW the memtile finishes
1017 cyc after the shim. EMU's wrong-direction artifact has shrunk
dramatically from Phase C's -777, but it's still wrong-direction.

## Why the cold-start knob cannot fix this

`shim_ddr_cold_start_cycles` adds N cyc to the MemoryLatency state on
the first BD of a shim DMA task (gated to shim + `transfer.
involves_host_memory()`, via `consume_first_bd_bonus`). This delays
the MemoryLatency -> Transferring transition by N. Concretely:

- Shifts data egress by +N
- Shifts shim FINISHED_TASK by +N
- Shifts memtile S2MM FINISHED_BD by +N (same downstream chain)

So **stage 1a grows by N**, and **stage 1b is invariant to N** (both
endpoints shift by the same amount). No setting of the cold-start
knob can close 1b. The trade-off the knob offers is:

| Cold-start value | 1a gap | 1b gap | Total 1+2 gap |
|---:|---:|---:|---:|
| 1500 (current) | -372 | +1044 | -672 |
| 1128 (optimal for 1a) | 0 | +1044 | -1044 |
| 2172 (optimal for total) | +1044 | +1044 | 0 |

No knob value can be both right for sub-stages AND right for the
total. The current choice (1500) is roughly the "minimize total
gap" tuning that happens to leave the sub-stage decomposition
visibly broken.

## What the wrong shape means

In real silicon:

```
shim_dispatch -> [DDR setup: ~tRCD + tRP + tRAS] -> [data streaming: throughput-bound] -> shim FINISHED_TASK
                                                          |
                                                          +-> memtile S2MM receives in parallel
                                                          +-> memtile commits last word ~tail cycles after last word leaves shim
```

HW's task duration is roughly `cold_start + transfer_words / shim_egress_rate`.
For a 64-word BD with ~1000 cyc cold-start + 1 word/cyc streaming, the shim's
task is ~1064 cyc, with data flowing continuously from cycle ~1000 to ~1064,
and the memtile finishing slightly after shim_done.

EMU's current model:

```
shim_dispatch -> BdSetup(4) -> AcquiringLock(1) -> MemoryLatency(5 + cold_start) -> Transferring(words/words_per_cycle) -> FINISHED_TASK
                                                                                          |
                                                                                          +-> data egress at 4 words/cyc
                                                                                          +-> memtile receives within a few cycles
```

`words_per_cycle = 4` in `DmaTiming` -- the configured streaming rate.
For a 64-word BD, Transferring takes 16 cyc. Total shim task in
EMU is ~1525 cyc (4 + 1 + 5 + 1500 + 16). The 16-cyc streaming phase
is fast enough that the memtile S2MM, which receives at the same rate,
finishes essentially co-incident with shim_done.

The model is missing **throughput-bound streaming**. Shim's actual
DDR egress rate is closer to 1 word/cyc (estimated, unmeasured) than
the configured 4 words/cyc -- but this is not just a `words_per_cycle`
knob bump, because the rate varies between cold-start and steady-state.

## The real fix (gated on HW ground truth)

Item #10 in `docs/coverage/cycle-accuracy-mission.md` -- shim streaming
throughput modeling. Sketch:

1. Model the shim DMA's data egress as a **rate**, not an instant
   discharge of a full BD's worth of words. The rate should match HW's
   DDR controller egress rate for the BD's access pattern (linear vs
   strided, single-bank vs cross-bank).
2. Cold-start primitive stays, but represents the row-open / precharge
   cost paid before the first word leaves -- not the entire pre-data
   delay we have today.
3. Streaming rate fires word-by-word with appropriate per-word cycle
   cost. Memtile S2MM receives at the matching rate, finishing slightly
   after the last word leaves the shim.

**Prerequisite**: HW measurement of shim egress rate. Vary BD size and
access pattern; measure shim_dispatch -> shim_done and shim_dispatch ->
memtile_s2mm_done deltas. Solve for cold_start (intercept) and
per-word rate (slope).

This is structurally bigger than the cold-start primitive. It changes
the Transferring phase from "consume N words instantly" to "consume
N words over N/rate cycles," with corresponding effects on stream
switch backpressure (the producer is now actually slow).

## Decision: 48a + 48c

48a -- leave `shim_ddr_cold_start_cycles=1500`. It's the local optimum
that minimizes total residual under the wrong-shape model. Tuning
to 1128 would make 1a sub-stage match HW but open the total gap
from 672 to 1044 cyc.

48c -- defer the structural fix to a streaming-throughput-modeling
item gated on HW ground truth. Added to cycle-accuracy-mission.md
as item #10.

Locking in current state with the analysis documented, rather than
chasing a local optimum that will be undone when the structural fix
lands.

## See also

- `docs/archive/findings/2026-05-10-phase-c-stage-attribution.md` --
  original 5-stage decomposition that defined this work
- `docs/superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md` --
  measurement-discipline finding from earlier today; the SoC-based
  decomposition above depends on it
- `docs/coverage/cycle-accuracy-mission.md` item #5 (NoC / AXI / DMA
  pipeline timings, DEFERRED) -- broader umbrella for DMA pipeline
  modeling
- `docs/coverage/cycle-accuracy-mission.md` item #10 (this finding's
  forward link) -- specific shim streaming throughput modeling item
- `src/device/dma/engine/stepping.rs` `consume_first_bd_bonus` --
  cold-start emission site
- `crates/xdna-archspec/src/model_builder.rs:189` -- where the 1500
  value is set
- Commit `3357b7c` -- the cold-start primitive that closed 74% of
  the Phase C gap
