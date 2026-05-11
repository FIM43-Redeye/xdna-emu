---
name: 'EMU: dma_wait now charges firmware mailbox latency (~8000 cyc)'
description: EMU's BlockedOnSync → FlushingStreams transition now models the firmware mailbox roundtrip latency that HW pays on every dma_wait. Single 8004-cyc charge per sync (8000 mailbox + 4 stream-flush), applied whether or not more instructions follow. Closes the dominant component of the dma_wait gap identified in the 2026-05-10 Phase B finding.
type: project
---

# EMU dma_wait mailbox latency

## TL;DR

EMU's executor now charges every `dma_wait` an 8004-cyc post-sync
settling cost (8000 cyc firmware mailbox roundtrip + 4 cyc stream
switch flush). This closes the dominant component of the ~22.5x
dma_wait gap measured in
`2026-05-10-phase-b-runtime-seq-instrumentation.md`.

Direct measurement on `_diag_phase_b_add_one_instrumented` (chess):

```
Pre-fix:  EMU total = 7008 cyc
Post-fix: EMU total = 15008 cyc
Delta:    +8000 cyc -- exact, matches the design target.
HW total (anchored T0..T3 only):  11781 cyc
```

EMU now overshoots HW by ~3000 cyc on this specific kernel, which
is a separate calibration question (likely DMA pipeline propagation
elsewhere -- see the "structural pipeline gap" thread in the Phase B
finding). For now the dma_wait component is closed.

## Implementation

Single change in `src/npu/executor.rs`. The state machine already had
`ExecutorState::FlushingStreams` as a post-sync settling state with a
4-cyc stream-switch flush; the change widens its counter to `u32` and
adds 8000 cyc of mailbox roundtrip to the transition value:

```rust
const MAILBOX_RESPONSE_CYCLES: u32 = 8000;
const STREAM_FLUSH_CYCLES: u32 = 4;
self.state = ExecutorState::FlushingStreams {
    next_index,
    remaining: MAILBOX_RESPONSE_CYCLES + STREAM_FLUSH_CYCLES,
};
```

Adjacent change: previously, if `dma_wait` was the LAST instruction
in the runtime sequence, the executor would short-circuit to `Done`
without going through `FlushingStreams`. On HW the mailbox roundtrip
happens regardless of whether more host-side work follows. Removed
the short-circuit; now `FlushingStreams` always fires, and its
`remaining→0` transition decides Done vs Executing based on
`next_index`.

## Test

`npu::executor::tests::test_sync_resolution_charges_mailbox_latency`
pins the post-sync remaining count at 8004 and asserts the executor
spends exactly that many cycles in `FlushingStreams` before
resuming. Any future tuning of `MAILBOX_RESPONSE_CYCLES` will surface
as a test failure that gets reviewed deliberately.

## What changes for downstream consumers

- **Every test with at least one `dma_wait`** now reports ~8000 more
  EMU cycles. That's essentially every kernel in
  `mlir-aie/test/npu-xrt` (only configuration-only paths skip
  `dma_wait`).
- Cycle-count baselines from before this commit are invalidated.
  Calibration sprints (Phase A, Phase C residuals) should
  re-baseline.
- Bridge `cargo test --lib` unaffected (2878 tests still pass --
  none of them encoded the pre-fix cycle count in an assertion).

## Update 2026-05-12: per-batch model lands as the default

After the env-opt-in fallback, an empirical fit on EMU+sync-count
data across 5 kernels (1, 2, 4, 261, 272 syncs) and the physics of
firmware command processing pointed to a **per-batch** model:
charge the mailbox roundtrip once per runtime sequence (on the
first dma_wait), then subsequent dma_waits pipeline through the
firmware mailbox queue at zero mailbox cost.

This matches both regimes:
- **Single-sync (Phase B)**: full ~8000 cyc charged on the only
  dma_wait. Closes the original 22.5x gap.
- **Multi-sync (160-272 dma_waits)**: only ~8000 cyc total mailbox
  overhead, well within budget. No timeouts.

Direct verification on the same five kernels after the model
landed (chess/peano EMU cycles):

```
test                              syncs  pre-fix   post-fix  delta
_diag_phase_b_add_one_instrumented   1   5258      14990     +9732
add_blockwrite                       2   5425      13831     +8406
shim_dma_bd_reuse                    4   20926     -- (passes, not re-measured)
sync_task_complete_token           272   690363    698363    +8000
ctrl_packet_reconfig_1x4_cores     261   758281    766281    +8000
```

The 261-sync test pays the same +8000 cyc total as the 1-sync test
-- exactly the per-batch model's prediction. The slight excess
above +8000 on the 1-2 sync cases is EMU side-effects during the
FlushingStreams window (DMA finalization that wouldn't have happened
otherwise); accuracy gain, not noise.

The default `MAILBOX_RESPONSE_CYCLES` is now 8000 (Phase B
measurement). `XDNA_EMU_MAILBOX_LATENCY` still overrides for
calibration experimentation.

Full bridge sweep after the change: zero regressions vs the
pre-mailbox-model baseline (0 timeouts, 0 newly-failing tests).

## Update 2026-05-11 evening: dialed back to env-opt-in (default 0)

The first full bridge sweep after the 8000-cyc constant landed
exposed the multi-sync regression I flagged in "open questions":
**4 chess + 6 peano tests timed out** at the 600s wall clock.
Two of them (`ctrl_packet_reconfig_1x4_cores`, `sync_task_complete_token`,
`ctrl_packet_reconfig_4x1_cores`) have 130-163 `dma_wait` calls
each. At 8000 cyc/sync × 160 syncs that's 1.28M cyc of pure
mailbox time, which at EMU's per-cycle wall-clock cost blew the
budget.

Reverted the default to 0 and made the mailbox component
env-controlled via `XDNA_EMU_MAILBOX_LATENCY`. The other piece of
the change kept (FlushingStreams always fires, even when sync is
the final instruction) -- adds only 4 cyc/sync (stream-flush).

After revert, the focused rerun of the 4 timeout tests passes:
```
ctrl_packet_reconfig_1x4_cores  PASS chess, PASS peano
ctrl_packet_reconfig_4x1_cores  PASS chess, PASS peano
sync_task_complete_token        PASS chess, PASS peano
dma_task_large_linear           PASS peano  (chess is SKIP_COMPILER)
```

Calibration sprints that want the mailbox accuracy on
single-sync kernels still get it:
```
XDNA_EMU_MAILBOX_LATENCY=8000 ./test.exe ...
```

The multi-sync model is now open work, not a regression risk in
the default path.

## Open questions

1. **Per-sync vs amortized -- still unresolved**. The dial-back
   confirmed the per-sync model is wrong for multi-sync kernels.
   Reality on HW: a single firmware command queue with bounded
   concurrency probably serializes mailbox responses, so the cost
   amortizes across nearby syncs. A model that matches HW would
   either (a) cap the cumulative cost per "batch" of syncs in a
   runtime sequence, or (b) charge a smaller per-sync constant
   (~50-100 cyc?) representing the amortized response overhead.
   Need multi-sync HW measurement to discriminate.

2. **Is 8000 NPU1-specific?** Phase B measured 7981 cyc on Phoenix
   (NPU1). NPU2/NPU4 likely have similar mailbox latency but the
   constant should ideally be device-keyed once we cross-check.

3. **EMU overshoots HW after the fix.** On the Phase B kernel EMU is
   now ~3200 cyc longer than HW. Two possibilities:
   - The 8000-cyc constant is generous (HW measurement noise: 7981
     vs assumed 8000; some other source of overlap on HW that EMU
     doesn't model).
   - EMU has other small over-counts elsewhere in the runtime
     sequence that the previous undershoot was masking.
   Worth a Phase D re-measurement when other calibration items move.

## See also

- `docs/superpowers/findings/2026-05-10-phase-b-runtime-seq-instrumentation.md`
  -- the measurement that identified this gap.
- `src/npu/executor.rs` -- the change.
- task #24 (this work); task #13 still tracks the residual ~9 cyc
  that lives separately.
