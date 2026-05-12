---
name: 'EMU DMA pipeline runs ~2.4x faster than HW, causing compute core to miss subsequent stall windows after the head-of-line wait'
description: On add_one_using_dma the EMU compute core stalls once (cycles 473-605 of trace) waiting for the initial DMA fill, then sails through 4 iterations × 4 acquires (16 logical acquires) without ever stalling again. HW stalls at multiple iteration boundaries (cycles 32, 9410, 20693, 25724, 26459+, 27281+) because the DMA refill rate falls behind the compute drain rate. Root cause: EMU's DMA fill cycle per buffer (~35 cyc/BD on compute S2MM) is ~2.4x faster than HW's effective rate (~2625 cyc/buffer in HW vs ~1088 in EMU after head-of-line). Downstream symptom: EMU emits 22 LOCK_STALL and 4 INSTR_LOCK_ACQUIRE_REQ events vs HW's 24 and 2339 -- but only because EMU has one stall window where HW has six. The event-emission code is roughly correct; the trace-content gap is a DMA-modeling gap. Probable contributors: stream-switch traversal latency unmodeled (each hop is ~1 cyc in EMU), per-BD turnover at memtile possibly over-pipelined by recent #26 fix, DDR refill rate too aggressive on the shim path. Calibration task on the scale of #24 (mailbox latency), not a quick fix. Stream-switch traversal latency is the most likely single-knob contributor and the planned next step.
type: project
---

# EMU DMA pipeline too fast -- compute misses subsequent stall windows

## TL;DR

On `add_one_using_dma`, the EMU compute core stalls *once* at the
head-of-line and then runs all 4 iterations × 4 acquires (16 logical
acquires) without re-stalling. HW stalls at *multiple* iteration
boundaries. The reason is not a trace-emission bug: it is that the
EMU's DMA pipeline (shim → memtile → compute) is **~2.4x faster per
buffer than HW**, so the producer is always ahead of the consumer
and the consumer never waits.

This is upstream of three open trace-divergence symptoms
(NEXT-STEPS.md naming):

- **#353** LOCK_STALL undercount -- 22 in EMU vs 24 in HW
- **#354** PERF_CNT_0 / PERF_CNT_2 undercount -- 63 vs 56 (actually
  *over*count post my recent fixes, but the count is roughly right)
- **#355** EMU kernel-activity cycle range short vs HW -- 14655 vs
  ~27340

All three are downstream of the same root cause.

## Investigation context

Came from cross-checking the open NEXT-STEPS.md tasks (#321, #353,
#354, #355) against current EMU behavior. #353 and #354 are
substantially closed (events do emit). #355 is partially closed
(EMU at 14655 cyc vs HW ~27340 -- no longer 47x off, but EMU is
~46% faster than HW overall).

Cross-checked event counts on `add_one_using_dma` with both HW and
EMU traces decoded via `tools/parse-trace.py`:

| event | HW (sane <100k) | EMU | ratio |
|------|---:|---:|---:|
| INSTR_LOCK_ACQUIRE_REQ | 2339 | 4 | 0.0017 |
| LOCK_STALL | 24 | 22 | 0.92 |
| MEMORY_STALL | 16 | 1 | 0.06 |
| STREAM_STALL | 6 | 16 | 2.67 |
| PERF_CNT_2 | 56 | 63 | 1.12 |

Tried fixing `INSTR_LOCK_ACQUIRE_REQ` directly by emitting it on
every cycle of WaitLock polling (`interpreter/core/interpreter.rs`).
EMU's count stayed at 4 (events coalesced into 4 cycles by something
in the trace pipeline) but `LOCK_STALL` count *exploded* from 22 to
4954 -- adding one unrelated event somehow caused the trace
pipeline to commit thousands more LockStall frames. Reverted.

Realized the symptom was upstream: the 4 InstrLockAcquireReq events
correspond to 4 *cycles* of acquire activity in EMU, because the
kernel's 16 logical acquires actually issue across only 4 distinct
cycle clusters. HW's 2339 events span thousands of cycles because
HW's acquire instructions sit in WaitLock for thousands of cycles
across multiple stall windows.

## Empirical data

`add_one_using_dma` kernel structure:

- 4 iterations × 2 halves × 2 acquires + 2 releases = 16 acquires,
  16 releases per kernel run
- objfifo input depth 2, output depth 2
- compute kernel inner body: `for i in 0..8 { load, add, store }`

**EMU lock acquire log** (from `bridge.log`):

```
17 LockAcquire calls total: 1 WAIT (head-of-line at lock 1, raw=49),
16 SUCCESS (4 acquires × 4 iterations).
```

After the head-of-line wait completes at cycle ~5949, the next 15
acquires all succeed immediately. No subsequent stalls.

**EMU compute S2MM DMA fill rate** (from `DMA(1,2) ch0` state log):

```
cycle=5949 Transferring -> AcquiringLock
cycle=5950 AcquiringLock -> Transferring   (acquire took 1 cyc)
cycle=5957 Transferring -> AcquiringLock   (Transferring window: 7 cyc)
cycle=5977 AcquiringLock -> Transferring   (acquire wait: 20 cyc)
cycle=6012 AcquiringLock -> Transferring   (acquire wait: 34 cyc)
cycle=6047 AcquiringLock -> Transferring   (34 cyc)
cycle=6082 AcquiringLock -> Transferring   (35 cyc)
cycle=6117 ...
cycle=6152 ...
cycle=6187 ...
```

Steady-state: **~35 cycles per buffer fill on the DMA side.** That
gives the compute core a new buffer every 35 cycles, far faster than
the core can drain its inner-loop body (which takes ~30 cycles per
buffer half including 4 lock ops + 8 inner iterations).

**Per-buffer cycle budget:**

| | EMU | HW |
|---|---:|---:|
| Cold-start (head-of-line) | ~5949 cyc | ~5000 cyc (estimate) |
| Active processing window | 14655 - 5949 = 8706 cyc | ~21000 cyc |
| Buffers processed | 8 (4 iter × 2 halves) | 8 |
| **Per-buffer steady-state** | **~1088 cyc** | **~2625 cyc** |
| Ratio | 1x | **2.4x slower** |

HW LOCK_STALL events distributed across the kernel (sane-filtered
timestamps):

```
[32, 9410, 9423, 20693, 20706, 25724, 25741, 26459, 26475, 26477,
 26480, 26482, 26490, 26495, 26496, 26511, 26512, 26516, 26526,
 26540, 27281, 27284, 27307, 27338]
```

Six distinct burst-windows: cycle 32 (head-of-line), 9410-9423,
20693-20706, 25724-25741, 26459-26540 (big burst, end-of-iter
cleanup), 27281-27338 (final). The 9k/20k/25k spacing matches
~5-6k cycles per iteration in HW, consistent with the per-buffer
estimate of 2625 cyc × 2 halves per iter ≈ 5250 cyc per iteration.

EMU LOCK_STALL events:

```
[473, 477, 478, 480, 481, 483, 484, 486, 487, 489, 490, 492, 493,
 495, 496, 498, 499, 501, 502, 504, ..., 605]
```

Single burst of 22 events in cycles 473-605, then nothing for the
remaining ~14000 cycles of execution.

## Why the rate mismatch

The EMU DMA pipeline has been progressively optimized this week
(`#13` chain pipelining, `#26` inline release + grant). Those fixes
correctly closed dead cycles that the EMU was paying but HW was not.
But they may have over-corrected: HW probably has *some* inter-BD
latency (arbiter handshake, stream-switch turnover) that EMU now
zero-pays.

Concretely, the candidate sources of the missing 2.4x:

### 1. Stream switch traversal latency (most likely single contributor)

`STREAM_LOCAL_TO_LOCAL_LATENCY`, `STREAM_LOCAL_TO_EXTERNAL_LATENCY`,
`STREAM_EXTERNAL_TO_EXTERNAL_LATENCY` constants exist in archspec
but I have not verified they are enforced on the data path
end-to-end. If a word traverses shim → memtile → compute through
two stream-switch hops with, say, 2-4 cyc per hop, that adds 4-8
cyc per buffer-byte-stream, summing to large per-buffer overhead.

### 2. Per-BD turnover at memtile (possibly over-pipelined)

The `#26` chained-BD work closed the AcquiringLock-to-Transferring
intermediate cycle. In synthetic tests (no backpressure), this
matched HW exactly. Under realistic backpressure on shared locks,
HW might still pay a cycle or two between BDs for arbiter
turnover. Worth re-measuring memtile MM2S 16w under backpressure
specifically.

### 3. DDR refill rate (shim path)

The shim DMA's host-memory pipeline has cold-start (~3000 cyc) plus
~1 word/cyc thereafter (modeled in `host_memory_latency_cycles` and
`shim_ddr_cold_start_cycles`). HW DDR might have burst patterns
and refill stalls on consecutive buffer reads that EMU does not
capture.

### 4. NoC traversal cycles (unmodeled entirely)

The shim → memtile path crosses the NoC. EMU treats this as
zero-latency after cold-start. Real NoC has per-hop latency and
contention.

## Downstream effects

This DMA-too-fast root cause produces several visible symptoms:

- **#353**: LOCK_STALL undercount -- EMU stalls once where HW stalls
  six times. Per-stall events count similarly (~5 events per
  stall-burst on both sides), but EMU has 1/6th the stall windows.
- **#354**: PERF_CNT_2 actually overshoots HW slightly (63 vs 56),
  likely because EMU's compute runs ~46% faster and the perf
  counter completes more cycles in its active window.
- **#355**: 14655 vs 27340 total kernel cycles. ~46% faster, same
  ratio as the per-iteration rate.

The fix surface for all three is **the same**: make EMU's DMA
pipeline run at the right rate relative to compute. The event
emission code itself is roughly correct.

## What I tried and reverted

Adding `InstrLockAcquireReq` emission to the WaitLock polling path
in `interpreter/core/interpreter.rs::try_resume_stall` caused
`LOCK_STALL` count to explode from 22 to 4954 -- not because I
touched LockStall emission code, but because the trace pipeline
has some emergent behavior where adding events to one path
triggers more frame commits on others. I did not fully understand
this and backed out. The trace pipeline's `pending_slot_mask`
coalescing and the events-log circular buffer dropping interact in
ways that need their own investigation if we want to do
per-cycle-of-stall event emission cleanly.

For now: the per-stall-window event count is approximately right
(22 vs 24). The actual problem is the missing stall windows.

## What to do next

Recommended sequence, in priority order:

1. **Stream-switch traversal latency** (next step planned). Verify
   that `STREAM_LOCAL_TO_LOCAL_LATENCY` etc. are applied on the
   data path from MM2S → S2MM through one or more switch hops.
   If they are not, wire them in. Measure impact on
   `add_one_using_dma` cycle count.

2. **Memtile DMA inter-BD turnover under backpressure**. The `#26`
   fix matched HW in no-backpressure synthetic tests, but the
   relevant case for `add_one_using_dma` is sustained backpressure.
   Re-measure memtile MM2S 16w specifically under that workload
   and see if HW pays cycles EMU has zero'd out.

3. **DDR refill / NoC latency**. Lower priority -- one-shot cost
   per kernel-cycle, the per-iteration impact is small compared to
   stream switch and DMA turnover.

4. **Trace pipeline coalescing investigation** (for future
   per-cycle event work). The mystery of "adding InstrLockAcquireReq
   emission inflates LockStall count 200x" needs to be resolved
   before any HW-rate event emission work can land. Not blocking
   the DMA-rate work above.

## Related findings

- `2026-05-11-emu-bd-chain-pipelining.md` -- the `#13` fix that
  closed chain setup overlap. Verify under backpressure.
- `2026-05-11-emu-chained-bd-spec-acquire-attempt.md` -- the `#26`
  fix that closed the inline release + inline grant residuals.
  Possibly over-aggressive in the backpressured case.
- `2026-05-11-emu-dma-wait-mailbox-latency.md` -- the `#24`
  calibration for firmware mailbox latency. Same shape of fix
  (calibrated constant for a HW timing component) is what (1) and
  (2) above would look like.
- `2026-05-05-355-cycle-divergence-diagnosis.md` -- original
  diagnosis of #355. The 50x gap is mostly closed; the remaining
  46% is what this finding decomposes.
