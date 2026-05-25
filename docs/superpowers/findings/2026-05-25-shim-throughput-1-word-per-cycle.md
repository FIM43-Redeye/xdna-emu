---
name: 'HW shim DMA throughput is 1 word/cycle'
description: HW shim DMA on Phoenix sustains ~1 i32-word/cycle in both MM2S (push) and S2MM (pull) directions, measured via a parameterized calibration kernel that sweeps single-BD shim transfers from N=8 to N=16384 i32 words.  Cold-start dominates at small N (typical shim DMA task time 700-2000 cyc) and a linear throughput term becomes visible above N~1024.  Large-N sub-fit (N >= 1024): MM2S cold_start=747 cyc + 1.05 cyc/word; S2MM cold_start=171 cyc + 1.00 cyc/word, R^2 = 0.996 / 1.000.  Numbers feed item #10 of the cycle-accuracy mission (shim throughput modeling in EMU's DMA stepping).
type: project
---

# Phoenix NPU shim DMA throughput characterization

## TL;DR

Calibration sweep on the existing-HW NPU shows the shim DMA controller
sustains essentially **1 i32-word per cycle** in both directions:

| Direction | Large-N cold_start | Large-N slope | Words/cyc | R^2 |
|-----------|-------------------:|--------------:|----------:|----:|
| MM2S (push: DDR -> memtile) | 747 cyc | 1.05 cyc/word | 0.95 | 0.996 |
| S2MM (pull: memtile -> DDR) | 171 cyc | 1.00 cyc/word | 1.00 | 1.000 |

Large-N sub-fit uses N in {1024, 2048, 4096, 8192, 16384}.  All-N fit
(adds N in {8..512}) gives the same slope but inflated cold_start
(1155 cyc MM2S, 569 cyc S2MM) because per-BD task overhead is
size-independent and proportionally larger on small BDs.

The directional asymmetry in cold_start (MM2S ~750 cyc, S2MM ~170 cyc)
reflects that the push direction has to issue a DDR read before the
data starts streaming, while the pull direction is downstream of an
already-warm memtile buffer.

## Method

Calibration kernel: shim MM2S 0 -> memtile S2MM 0 -> memtile buffer
-> memtile MM2S 0 -> shim S2MM 0 (single column, no compute core,
passive memtile loopback).  One BD per direction per invocation; BD
length N parameterized at compile time across {8, 16, 32, 64, 128,
256, 512, 1024, 2048, 4096, 8192, 16384} i32 words.

Kernel source lives at `mlir-aie/test/npu-xrt/_diag_shim_throughput_sweep/n64/`
(untracked in mlir-aie tree, same convention as `_diag_phase_b_add_one_instrumented`).
Sibling sizes materialize via `xdna-emu/scripts/gen-shim-throughput-sweep.sh`.
The runtime sequence forces sequential dispatch with an
`aiex.npu.dma_wait{@in}` between the two `dma_memcpy_nd` ops; without
this, the host queues both BDs and runs them concurrently, which
confounds per-direction timing.

Trace event capture from the bridge harness default shim event set:
`DMA_MM2S_0_START_TASK`, `DMA_MM2S_0_FINISHED_TASK`,
`DMA_S2MM_0_START_TASK`, `DMA_S2MM_0_FINISHED_TASK`.  Durations are
`FINISHED.soc - START.soc` (the ts/soc gotcha discipline applies --
soc only).

Regression: `xdna-emu/tools/shim-throughput-fit.py`, ordinary
least squares fit of `duration = cold_start + N * slope` per
direction.

## Data

Single-shot measurements per BD size, run sequentially via the
bridge harness on the local NPU:

| N | MM2S dur | S2MM dur |
|--:|--------:|--------:|
| 8 | 1854 | 830 |
| 16 | 2064 | 845 |
| 32 | 963 | 872 |
| 64 | 1741 | 696 |
| 128 | 1207 | 862 |
| 256 | 838 | 726 |
| 512 | 1314 | 1001 |
| 1024 | 1247 | 1187 |
| 2048 | 3421 | 2217 |
| 4096 | 5304 | 4261 |
| 8192 | 9121 | 8357 |
| 16384 | 17963 | 16525 |

CSV at `data/hw-shim-throughput-2026-05-25.csv`; per-direction fit at
`data/hw-shim-throughput-2026-05-25-fit.json`.

Single-shot variance at small N (8..1024) is high -- duration jitters
in a 700-2000 cyc band that masks the per-word throughput term until
N >= 2048.  This is per-BD task overhead variance, not noise on the
throughput measurement itself; the large-N points sit on a tight
linear curve.

## Implications for EMU (item #10)

Current EMU stage 1+2 model uses `shim_ddr_cold_start_cycles = 1500`
applied per shim DMA event (per
`findings/2026-05-25-shim-stage1a-1b-structural-limit.md` and the
Phase C primitive).

The HW measurement says:

- MM2S cold_start should drop from 1500 to ~750-1150 cyc (depending
  on whether the model meant per-BD or per-dispatch).
- S2MM cold_start should be much lower than MM2S, ~170-570 cyc.
- A per-word throughput term should be added at ~1 word/cyc (256
  bit per cycle of memref<i32>, plausibly matching the AXI4-Stream
  256-bit interface width).

Tuning the EMU primitive based on these numbers is the next step
(item #10 modeling, gated on this finding).  Validation gate from
the campaign doc still applies: EMU stage 1a/1b residuals should
drop within stage 3 noise floor on `_diag_phase_b_add_one_instrumented`
after the model update.

## Cold-start is per-dispatch, not per-BD

The kernel's runtime sequence issues exactly one `dma_memcpy_nd`
per direction.  Each generates one or more shim BDs internally
(depending on alignment / max-BD-length); the shim DMA's
`START_TASK` and `FINISHED_TASK` events bracket the entire
`dma_memcpy_nd` command, not the individual BDs.  So `cold_start`
in this fit is per-`dma_memcpy_nd`, not per-BD.

This matters for modeling kernels with chained BDs (e.g.,
`_diag_phase_b_add_one_instrumented` issues a single
`dma_memcpy_nd` that internally chains 4 BDs of 16 words).  The
EMU should treat the dma_memcpy_nd as the unit of cold-start, not
the individual BD.

## See also

- `docs/coverage/hw-measurement-campaign.md` -- the campaign that
  led to this measurement
- `docs/coverage/cycle-accuracy-mission.md` item #10 -- the EMU
  modeling work this unblocks
- `docs/superpowers/findings/2026-05-25-shim-stage1a-1b-structural-limit.md`
  -- prior finding identifying the gap
- `mlir-aie/test/npu-xrt/_diag_shim_throughput_sweep/n64/`
  (untracked in mlir-aie) -- kernel source
- `xdna-emu/scripts/gen-shim-throughput-sweep.sh` -- materializes
  sibling sizes
- `xdna-emu/tools/shim-throughput-fit.py` -- regression
- `xdna-emu/data/hw-shim-throughput-2026-05-25.csv` -- measurements
