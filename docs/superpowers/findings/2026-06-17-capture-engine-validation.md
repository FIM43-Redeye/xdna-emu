# Trace-Capture Engine — HW Validation (Task 9, #140)

**Date:** 2026-06-17
**Kernel:** `add_one_using_dma` (chess), real NPU1 (Phoenix), 6 runs.
**Result:** PASS — the engine captures correctly-labeled `events.json` per run and
the cross-run derivability classification is hardware-correct.

## What ran

`trace_capture.run_loop('add_one_using_dma', SEED_ACTIVE_PLAN, n_runs=6, ...)`:
per run, `HwRunner` (a thin adapter over trace-sweep's `RunnerSession`) drives
RESET -> patch -> run -> decode -> label for one co-traced batch of the seed
active set, writing `run_NN/batch_00/hw/trace.events.json`. After the runs:
N-run coverage union, the cross-run derivability graph, the synthesized plan,
and a joined Perfetto record stream.

## Validation outcome

- **6/6 runs clean**, 11 slots fired each, no hard errors.
- **Stochastic roots = the DMA-delivery events** (the #140 thesis, confirmed on
  HW): shim `DMA_MM2S_0_FINISHED/START_TASK`, `DMA_S2MM_0_FINISHED_TASK`;
  memtile `PORT_RUNNING_0`; core `LOCK_STALL`. All DMA-driven nondeterminism.
- **6 deterministic roots** form a stable backbone identical across runs within
  eps (the cross-run skeleton identity).
- **Coverage: 11/26 covered, 15 named gaps.** The engine surfaces non-firing
  configured slots by name instead of hiding them.

## Two root causes fixed at HW integration (unit tests could not catch these)

1. **Trace-mode mismatch (the blocker).** Kernels compile the *core* trace unit
   to EVENT_PC (mode 1: `Trace_Control0 = 0x797a0001`). Decoding a mode-1 stream
   with the mode-0 (EVENT_TIME) decoder misreads PC bytes as slot bitmasks ->
   phantom fires on slots configured NONE -> the unconfigured-slot hard error.
   Fix: `configure_batch` now emits `"mode": 0` per tile so the patcher rewrites
   `Trace_Control0` to EVENT_TIME, matching `parse_trace(EVENT_TIME)`. (Trace
   config is owned entirely by npu_insts; the CDO has zero trace writes — so
   patching insts is the correct lever, we were just not patching the control
   register. trace-sweep always sets `"mode"`, which is why it never hit this.)

2. **start_col reconcile (CRITICAL-1 from the design review, confirmed real).**
   The patcher operates in RELATIVE column space (the MLIR/insts logical column,
   0 for a 1-col kernel); the decoder reports ABSOLUTE columns (the driver places
   the partition at column 1). `SEED_ACTIVE_PLAN` is therefore keyed on relative
   col 0; `traced_col=1` and the join anchor (`1|2|0|PERF_CNT_2`) are absolute.
   The column-free `label_map` (keyed `(pkt,row,slot)`) bridges the two — exactly
   the property the design built it for.

Also fixed: `configure_batch` pads each tile's patch events to 8 with NONE(0) so
the patcher zeroes slots we don't configure (else the kernel's compile-time
trace events in the trailing slots fire and trip the guard); a py3.13
`@dataclass`/importlib quirk in `HwRunner`'s lazy `RunnerSession` load.

## Open follow-ups (not blocking; candidates for the trace-system review)

- **memmod (pkt=1, row=2) produced no packets** across all 6 runs — the entire
  memmod active set is a coverage gap. The row-2 tile has both a core and a
  memmod trace unit sharing one egress; whether they can be co-traced in one
  batch (vs split across batches) needs a dedicated look.
- **Seed over-counts.** The seed is the union across the prior 25-batch catalog
  characterization; in a single co-traced batch many events legitimately do not
  fire (workload/config-dependent), which is why 15/26 are gaps. Honest, but a
  discovery sweep (deferred) would seed more precisely.
- **Event GROUPS** (e.g. `GROUP_DMA_ACTIVITY`) as a way to fit more coverage per
  8-slot budget — promising, specificity tradeoff; see the trace-system review.
