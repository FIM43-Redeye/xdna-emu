---
name: 'Shim BD-pool over-allocation wedges hardware non-monotonically (k>8 chain sweep)'
description: The _diag_shim_chain_sweep K>8 variants over-allocate the 16-BD shim pool (2K distinct BDs needed). On real HW and AMD aiesim this wedges, but NON-MONOTONICALLY (k8 pass, k9 wedge, k12 pass, k16 wedge) -- an allocation-specific BD-lifecycle interaction with no clean closed form. Our interpreter completes all of them (re-parses BDs on reuse, never models the in-flight-BD-reuse hazard). Documented fidelity gap; generator capped at K=8 rather than modeled.
type: project
---

# Shim BD-pool over-allocation wedges hardware non-monotonically

## TL;DR

The `_diag_shim_chain_sweep` calibration kernels fire K back-to-back
`dma_memcpy_nd` dispatches per direction. Each direction needs K distinct shim
BDs, so a run needs **2K distinct BDs** -- but an AIE2 shim/NoC tile has exactly
**16 BDs** (aie-rt `NumBds = 16U`; BD-ID is 4-bit). For **K > 8** the compiler
wraps the input BDs onto the output BDs, and the kernel **wedges on real
hardware**.

Critically, the wedge is **non-monotonic in K**:

| K | input BDs | output BDs | in/out BD overlap | real HW / aiesim |
|---|-----------|------------|-------------------|------------------|
| 8 | 8-15 | 0-7 | none (0) | **PASS** |
| 9 | 9-15,0,1 | 0-8 | {0,1} (2) | **WEDGE** |
| 12 | 12-15,0-7 | 0-11 | {0-7} (8) | **PASS** |
| 16 | 0-15 | 0-15 | all 16 | **WEDGE** |

Overlap count goes 0, 2, 8, 16 -> verdict pass, wedge, pass, wedge. k9 wedges
with only 2 shared BDs while k12 passes with 8. There is **no clean
single-resource model** -- not "2K > 16", not overlap count, not K parity. It is
a subtle, allocation-specific BD-lifecycle / queue / lock-timing interaction
that the silicon and AMD's aiesim model but that does not reduce to a constant.

**Our emulator's interpreter passes all of them** (it re-parses a BD on reuse
and completes) -- a fidelity gap. We **cap the generator at K=8** and document
the gap rather than model a non-monotonic wedge for a synthetic calibration
kernel.

## How we got here

`k16` was the last `_diag_shim_chain_sweep` variant that failed somewhere in the
test matrix. Initial belief (from a stale caveat -- see "Corrects the record"
below) was that it failed because of an "8-deep shim task queue". Investigation
showed:

1. **Real HW fails k16.** `./test.exe -x aie.xclbin -i insts.bin` on the NPU1
   returns `ERT_CMD_STATE_TIMEOUT` (status 8) at ~4 s. k8 passes (~0.6 s). This
   is the ground truth.
2. **aiesim bridge faithfully reproduces it** (k16 wedges output-side; the
   shim S2MM output queue fills and the `dma_wait @out` never satisfies).
3. **Our interpreter completes k16** (`halt_reason=completed cycles=199237`,
   `PASS!`). So for this kernel our interpreter is *less* faithful than both HW
   and aiesim -- the inverse of the matmul/buffer case
   ([2026-06-06-aiesim-aie2-cross-core-shared-memory-limitation.md](2026-06-06-aiesim-aie2-cross-core-shared-memory-limitation.md)).

## Root cause: 16-BD shim pool, non-monotonic over-allocation

The decisive data is the BD allocation (from the interpreter's
`task_queue_ops` debug log, `enqueued task: BD=...`). The allocator
(`--alloc-scheme=basic-sequential`) gives output `BD0..K-1` and input
`BD K..2K-1`, **wrapping mod 16**. For K <= 8 the two halves are disjoint and
the kernel runs. For K > 8 they overlap, and because the **single-slot memtile
buffer overlaps the input and output phases** (the memtile MM2S drains
`cons_lock` *during* input -- observed directly in the probe: `cons_lock`
climbed to 13 while input was still flowing), the colliding BDs interact.

But the failure is **not** a clean function of either K or overlap count -- see
the table above. The k9/k12 discriminator (compiled + run through the aiesim
oracle, confirmed real at the default poll cap, not a cap artifact) proves the
non-monotonicity: k9 wedges, k12 passes. We could not reduce this to a single
modeled resource, and validating an emulator change against a jagged map with no
pattern is not tractable for a synthetic kernel.

## Why we cap instead of model

- `_diag_shim_chain_sweep` is **our own calibration kernel family** (shim DMA
  cold-start amortization, fixed N=64, sweeping K). Its purpose is meaningful
  only while the dispatches are well-formed; K>8 over-allocates the BD pool and
  the resulting numbers reflect compiler-allocation accidents, not DMA timing.
- Faithfully reproducing the wedge would mean modeling detailed shim
  BD-reuse-while-live + queue + TCT lifecycle semantics, then matching a
  non-monotonic pass/fail map -- a large effort with broad regression risk
  (touches every `issue_token` kernel) for zero real-workload payoff.
- The generator (`scripts/gen-shim-chain-sweep.py`) now caps K at 8 (`MAX_K`,
  `2*MAX_K == 16`) and refuses K>8 with an explanatory error.

## The fidelity gap (known, documented, not fixed)

Our interpreter's NPU executor **back-pressures** a full shim task queue
(`BlockedOnQueue` parks and retries until the queue drains) and **re-parses a BD
on reuse**, so it completes kernels that real HW cannot. Real HW drops on task-
queue overflow (AM025 `Task_Queue_Overflow` sticky bit) and has finite TCT
backpressure (`Stalled_TCT`; our `TokenState` is explicitly unbounded -- see the
comment in `src/device/dma/token.rs`). Related, also unfixed:

- `MAX_TASK_QUEUE_DEPTH = 8` in `src/device/dma/token.rs` but the hardware queue
  is **4** (aie-rt `XAIE_DMA_MAX_QUEUE_SIZE 4U`, `StartQSizeMax = 4U`). Correct
  in isolation is easy; it does not by itself reproduce the k16 wedge (our peak
  occupancy is 1-2), so it is left as part of this same gap.

These are tracked together as "shim DMA finite-resource fidelity" -- worth a
consolidated pass someday, but low priority versus real-workload behavior.

## Corrects the record

This supersedes the caveat in
[2026-05-25-shim-bd-chain-amortization.md](2026-05-25-shim-bd-chain-amortization.md)
("Shim task queue depth"), which said k16 HW fails because "the shim DMA task
queue is 8-deep on AIE2 per AM025; the ninth queued task wedges." That is wrong
twice: the queue is **4-deep**, not 8 (the "8" was our own wrong
`MAX_TASK_QUEUE_DEPTH`, not silicon), and the real cause is **16-BD pool
over-allocation**, not queue depth -- k8 already exceeds a 4-deep queue and
passes fine.

## Reproduction

```bash
# generator now refuses K>8:
python3 scripts/gen-shim-chain-sweep.py --k 16   # -> error, max K is 8

# BD allocation (any built variant, native interpreter):
cd <build>/k16/chess
XDNA_EMU=1 RUST_LOG=xdna_emu::device::dma::engine::task_queue_ops=debug \
  ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin 2>&1 | grep 'enqueued task'

# HW ground truth (real NPU, no XDNA_EMU): k16 -> "status: 8" (timeout); k8 PASS.
# aiesim oracle (XDNA_BACKEND=aiesim) reproduces: k8 pass, k9 wedge, k12 pass, k16 wedge.
```
