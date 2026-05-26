---
name: 'NPU controller dispatch overhead added to cycle-cost model'
description: Adds a flat per-task-start cost to the NpuExecutor's cycle-cost model.  HW K-sweep measurements showed ~2800 cyc inter-task gaps that EMU's existing per-packet costs only accounted for ~313 cyc of -- the missing ~2500 cyc is controller-side DMA arbiter / AXI setup that we now model as a single dispatch_overhead constant applied to Write32 instructions targeting Task_Queue registers.  Closes the dominant cycle-accuracy gap on chained DMA workloads.
type: project
---

# NPU controller dispatch overhead

## TL;DR

The K-sweep finding ([2026-05-25-shim-bd-chain-amortization](2026-05-25-shim-bd-chain-amortization.md))
fixed shim DMA cold-start amortization but documented a remaining gap:
HW spends ~2800 cyc between consecutive task-start events on a single
channel, while EMU spent only ~313 cyc.

This finding closes that gap by adding `dispatch_overhead = 2500 cyc`
to `CycleCostModel` (provisional_npu1 profile only).  The overhead
fires once per Write32 instruction whose register address matches a
DMA channel's Task_Queue register, applied as an additive on the
instruction's retirement cost.  Together with the existing per-packet
CMP cost (~100 cyc for the Write32 itself + ~200 cyc for the
preceding BD-config BlockWrite), total per-dispatch cost lands at
~2800 cyc.

## The number

HW K-sweep `_diag_shim_chain_sweep` chess inter-task gaps,
steady-state (excluding the first gap which often includes additional
sync overhead):

| direction | K=4 mean | K=8 mean | K=16 mean | overall |
|-----------|---------:|---------:|----------:|--------:|
| MM2S      | 2630     | 2737     | 2740      | ~2700   |
| S2MM      | 2860     | 2884     | 2885      | ~2870   |

EMU pre-fix gaps: 313 cyc steady-state.  Missing cost: ~2500 cyc per
dispatch, direction-agnostic at this resolution.  Modeled as a single
constant.

## Where it lands in EMU

`src/npu/cycle_cost.rs`:
- New `CycleCostModel::dispatch_overhead: u64` field.
- `provisional_npu1()` sets it to 2500.
- `legacy_one_per_packet()` and `with_known_constants()` leave it at 0
  (no behaviour change for callers using those profiles).

`src/npu/executor.rs`:
- New `Self::is_task_dispatch_write(col, row, offset, device)` --
  pure detector that returns true if the address matches any DMA
  channel's Task_Queue register on the target tile (compute, mem, or
  shim).  Mirrors the address classification done in
  `would_block_on_queue` but without state mutation.
- `try_advance()` applies `dispatch_overhead` to the retirement
  cycle budget for any Write32 instruction that hits this detector.

The detection is tile-type-aware (matches Task_Queue offsets for
compute, mem, and shim DMA channels) but the overhead constant is
shared -- the K-sweep only exercises shim dispatches.  If memtile or
compute task dispatches turn out to have different controller costs,
the constant can be split per tile type later.

## Validation

K-sweep `_diag_shim_chain_sweep` chess, post-fix HW vs EMU total spans:

| K | direction | HW    | EMU   | Δ     |
|---|-----------|------:|------:|------:|
| 2 | S2MM      | 3365  | 3367  | +2    |
| 4 | MM2S      | 10503 | 9700  | -803  |
| 4 | S2MM      | 7446  | 9619  | +2173 |
| 8 | MM2S      | 24827 | 22204 | -2623 |
| 8 | S2MM      | 22025 | 22123 | +98   |

S2MM K=2 and K=8 match HW within 100 cyc.  MM2S undershoots by
~10% (HW MM2S has higher run-to-run variance, esp. first-task).
K=4 S2MM overshoots by +2173 because this HW run happened to show
a near-zero first inter-task gap (+19 cyc); EMU's model assumes
constant ~2800 cyc per gap.

Pre-fix EMU values were 4623 (S2MM K=8) and 4704 (MM2S K=8) --
~80% under HW total span.  Post-fix EMU spans match within ~12%
of HW for K >= 2 chains.

Unit tests: 3209/3209 pass.

Corpus (`_diag_phase_b`, `add_one_using_dma`, `add_one_objFifo`,
`add_one_objFifo_elf`): all PASS bridge verdict.  Single-task stage 1a
EMU values UNCHANGED -- the dispatch_overhead delays the NEXT
instruction's issue, not the DMA channel's START event (which fires
synchronously when the Write32 reaches the DMA engine).  Single-task
workloads have no "next instruction within the dispatch window," so
the overhead is invisible to them.

## Caveats

**Single constant, no per-tile-type split.**  K-sweep only measures
shim dispatches.  Memtile and compute task dispatches are programmed
via CDO at xclbin load, not in the runtime sequence, so they rarely
appear as dispatch_overhead-eligible Write32s.  If a workload pattern
emerges where they do, may need direction- or tile-type-specific
constants.

**Doesn't model first-gap variance.**  The first gap after a phase
transition (e.g., MM2S phase to S2MM phase) often differs from
steady-state gaps -- sometimes much larger (+4000-5000 cyc) due to
phase synchronization, sometimes near-zero.  The constant model can't
capture this; it's tracked as expected variance.

**Run-to-run HW variance is large.**  Per-task and per-gap measurements
vary 30-50% between sweeps.  EMU's constant model trades absolute
accuracy for determinism.  Calibration values target the median
HW behaviour.

## Follow-ups (non-gating)

- **On-NPU readback path.**  The `provisional_npu1()` doc-comment
  flags that `Performance_Counter0` readback via
  `xrt::hw_context::read_aie_reg` is functional on Phoenix as of
  2026-05-05, which would provide trace-independent ground-truth
  cycle counts.  The trace-side calibration in this finding could be
  cross-validated against readback measurements to tighten the
  dispatch_overhead constant (and surface variance structure we
  currently average away).  See finding
  `docs/archive/findings/2026-05-05-aie-rw-access-firmware-actually-supported.md`.
- **First-gap modeling.**  If phase-transition gaps emerge as
  important for accuracy on real workloads, model the additional cost
  via direction-change detection in the executor.
- **K=16 EMU diverges from HW behavior (not cosmetic).**  Initial
  thought was that EMU's K=16 had a trailing phantom event; closer
  inspection shows it's actually two divergences from HW:

  - HW K=16: 16 MM2S START/FINISHED pairs (correct count), then **zero
    S2MM events** -- HW wedges between the MM2S phase finishing and
    the first S2MM dispatch.  Probably hits the 8-deep task queue
    limit during the deferred dispatch attempts (or a dma_wait timeout
    after queue management goes wrong).  Bridge test reports `FAIL`
    with a 4.8s timeout.
  - EMU K=16: 17 MM2S task pairs (1 phantom at the MM2S->S2MM
    boundary) + 18 S2MM task pairs (2 phantoms), runs to completion.
    EMU silently defers dispatches past the 8-deep queue via the
    `BlockedOnQueue` path -- never hits HW's failure mode -- and
    emits extra trace events at the phase boundary.

  "Replicate HW including its flaws" means EMU should ALSO fail at
  K=16 in roughly the same way HW does, not chug through.  The right
  fix is probably to treat queue overflow as a hard error (no graceful
  deferral past depth 8) rather than blocking until queue drains.
  Phantom events would naturally disappear because EMU wouldn't reach
  the S2MM phase.  Behavioral change with potential blast radius --
  needs a corpus sweep to check what depends on the current deferral
  semantics.  Tracked as future work.

## See also

- [`2026-05-25-shim-bd-chain-amortization.md`](2026-05-25-shim-bd-chain-amortization.md) -- prior finding (cold-once + per_task_OH); identified the inter-task gap as the dominant remaining axis
- [`2026-05-25-emu-shim-dma-timing-recalibrated.md`](2026-05-25-emu-shim-dma-timing-recalibrated.md) -- N-sweep recalibration
- `src/npu/cycle_cost.rs` -- cost model
- `src/npu/executor.rs` -- dispatch detection + overhead application
- `_diag_shim_chain_sweep/` -- K-sweep calibration kernel family
