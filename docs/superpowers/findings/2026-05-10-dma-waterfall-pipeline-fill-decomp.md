---
name: '#355a DMA waterfall: pipeline propagation is fine; gap lives in runtime-sequence dispatch + core boot'
description: Per-channel FSM phase residency on add_one_using_dma shows shim MM2S 0 idle for 3911 cyc waiting for the runtime sequence to push its BD task. After it starts, propagation through memtile->compute is ~547 cyc, which is in the ballpark of HW's documented chain latency. The DMA model is essentially correct; the gap lives in NPU instruction dispatch timing and compute boot timing, neither of which is DMA-related.
type: project
---

# `#355a` DMA waterfall on `add_one_using_dma`

## TL;DR

Built `tools/dma-waterfall.py` to reconstruct per-channel DMA FSM phase
residency from a bridge-test EMU log.  Ran fresh `add_one_using_dma`
EMU-only and built the per-channel timeline.  Two findings flip the
calibration target away from DMA:

1. **Shim MM2S 0 (host->memtile input) is idle for 3911 EMU cycles**
   waiting for the runtime sequence to dispatch its BD task.  That single
   bubble dominates EMU's pipeline-fill and is purely an NPU-instruction
   timing artifact, not a DMA model gap.
2. **Once the shim starts, propagation through memtile->compute takes
   only ~547 cyc** (cycle 3911 -> 4458), roughly matching the theoretical
   serial chain latency from the documented per-stage costs.

This is consistent with the May 5/6 split: kernel-execution is ~1.0x
already; the visible cycle gap lives outside the DMA pipeline.  After
this measurement the leading suspects are NPU instruction execution
timing (runtime-sequence dispatch) and compute core boot timing.

## Methodology

1. Added a single `log::info!` line at `start_channel_with_repeat` that
   matches the FSM step-loop's transition format
   (`DMA(c,r) chN: Idle -> {BdSetup|AcquiringLock} cycle=N`).  Without
   this, channel-start is invisible to the waterfall because it happens
   outside the per-cycle step loop.
2. Wrote `tools/dma-waterfall.py`.  Parses transition lines and emits
   per-channel time-in-phase totals plus optional CSV.
3. Ran `./scripts/emu-bridge-test.sh --no-hw --no-trace --chess-only
   'add_one_using_dma$'`.  Total emu run = 5060 cyc, 218 transitions.

## Per-channel residency (chess EMU, end_cycle=5060)

```
    tile        ch  AcquiringLock  BdChaining   BdSetup  HostPipelineLatency      Idle  MemoryLatency  ReleasingLock  Transferring     total
   (1,0)         0                                    2                  500      4537              5                           16      5060
   (1,0)         1                                    2                  496      3187              5                         1370      5060
   (1,0)         2                                    2                  491      4532              5                           30      5060
   (1,1)         0              9           8        16                                            25              4          4998      5060
   (1,1)         1              9           8        16                                            25              4          4998      5060
   (1,1)         6           4988           8        16                                            20              4            24      5060
   (1,1)         7           4988           8        16                                            20              4            24      5060
   (1,2)         0            342          16        32                                            45              8          4617      5060
   (1,2)         2           4948          16        32                                            40              8            16      5060
```

Channel mapping (col 1 = our test column, after start_col=1 offset):
- (1,0) = shim:    ch0=S2MM 0 (output), ch1=S2MM 1, ch2=MM2S 0 (input)
- (1,1) = memtile: ch0=S2MM 0 (in), ch1=S2MM 1, ch6=MM2S 0 (out), ch7=MM2S 1
- (1,2) = compute: ch0=S2MM 0 (kernel input), ch2=MM2S 0 (kernel output)

Massive `Idle` columns on shim channels show the dispatch wait.  Massive
`Transferring` on S2MM channels is "waiting for stream data" -- the
state stays put while no word arrives.  Massive `AcquiringLock` on MM2S
channels (ch6, ch7, ch2 of compute) is "waiting for cons_lock release."

## First-pass timeline (the part that matters for pipeline fill)

| Cycle  | Event                                                              |
|-------:|--------------------------------------------------------------------|
|      0 | All compute + memtile DMAs go Idle->Active (CDO setup)             |
|      0 | Compute S2MM 0 acquires prod_lock (init=2), enters MemoryLatency   |
|      6 | Compute S2MM 0 enters Transferring -- now waits for stream data    |
|      6 | Memtile S2MM 0 enters Transferring -- waits for shim to feed it    |
|   2085 | (Per May 6) compute core hits its first lock acquire               |
| **3911** | **Shim MM2S 0 (host->memtile) Idle -> BdSetup -- dispatched at last** |
|   3913 | Shim BdSetup -> MemoryLatency (2 cyc, cf. 4 expected -- worth noting) |
|   3918 | Shim MemoryLatency -> HostPipelineLatency (5 cyc check)            |
|   4409 | Shim HostPipelineLatency -> Transferring (491 cyc, ~500 expected)  |
|   4434 | Memtile S2MM 0 Transferring -> ReleasingLock (releases cons_lock)  |
|   4436 | Memtile MM2S 0 (ch6) AcquiringLock -> MemoryLatency (acquired)     |
|   4441 | Memtile MM2S 0 enters Transferring -- starts pushing to compute    |
|   4458 | Compute S2MM 0 Transferring -> ReleasingLock (releases cons_lock)  |

So the **DMA-side propagation chain** from "shim starts" to "first compute
S2MM lock release" is **547 cyc** (3911 -> 4458).  That covers:

- 2 cyc shim BdSetup
- 5 cyc shim MemoryLatency
- 491 cyc shim HostPipelineLatency (host_memory_latency_cycles=500)
- ~10 cyc shim transfer + stream switch routing to memtile
- ~14 cyc memtile S2MM 0 fills 16-word BD + lock release
- ~7 cyc memtile MM2S 0 lock acquire + BdSetup + MemLat
- ~10 cyc memtile transfer + stream switch routing to compute
- ~8 cyc compute S2MM 0 fills 8-word BD + lock release

These match the model's per-stage costs.  Nothing about the DMA chain
itself is short by thousands of cycles.

## Where the gap actually lives

EMU's measured pipeline fill (consumer's first WAIT to first SUCCESS) was
2719 cyc post host_lat=500 tune (May 6).  Decomposed:

- Compute hits first acquire at **cycle ~2085** (May 6 measurement).
- Lock signal arrives at **cycle 4458**.
- Wait time: 4458 - 2085 = **2373 cyc**.  Close to May 6's 2719 (delta
  is which lock is observed and when the trace controller ticks).

The 2373-cyc consumer wait decomposes as:
- 1826 cyc of "compute boots and runtime sequence dispatches in
  parallel, but shim MM2S 0 doesn't start until cycle 3911 because the
  runtime sequence takes that long to issue its BD push"
- 547 cyc of "DMA propagation chain"

On HW, per the May 6 inference, compute boots at cycle ~6000 and the
trace shows ~6000 cyc of pre-acquire stalls.  HW first acquire issues
later in absolute cycles than EMU's, which means HW's runtime sequence
dispatch + boot is *slower* than EMU's, not faster.  Two of the three
suspects from the prior finding are now ruled in:

| Suspect                                  | Verdict                                |
|------------------------------------------|----------------------------------------|
| DMA engine internal pipeline cost gap    | **Refuted.** Chain is ~547 cyc and matches the model. |
| EMU core boots too fast (~2.9x)          | **Confirmed.** Compute hits first acquire at 2085 EMU vs ~6000 HW. |
| Runtime sequence dispatch timing too fast| **New suspect.** Shim ch2 starts at 3911 EMU; HW likely later (firmware-paced). |
| Stream-switch fabric latency             | Already correctly modeled (May 7).    |
| Memtile broadcast/fanout                 | Not exercised by this 1D test.        |

## Loose ends worth noting (small, don't change the conclusion)

- **BdSetup observed = 2 cyc, model says 4.**  Off-by-2 in transition
  timing, possibly a `current_cycle` advancement vs `cycles_remaining`
  decrement ordering issue at start_channel.  Not material to the gap.
- **HostPipelineLatency observed = 491 cyc, model says 500.**  Same
  family of off-by-9 in the FSM countdown vs cycle-counter alignment.
  Worth tightening the FSM to make residency match the constant exactly,
  but again not material.
- **`channel_start_cycles=2` defined but never consumed by the FSM.**
  Audit-discovered dead constant.  Adding it back would add ~2 cyc per
  channel start, ~30 cyc total across all DMAs.  Trivial.
- **`packet_arbitration_overhead=1` similarly orphaned.**  Doesn't
  affect this test (no packet flows).

## Recommended next direction

The DMA model is essentially correct for this workload; the calibration
gap is now a runtime-sequence + core-boot timing problem.  Two avenues:

1. **Audit NPU instruction execution timing** in `src/npu/executor.rs`
   (and the runtime sequence interpreter).  The 3911-cyc dispatch wait
   from xclbin-load to first shim BD push is the bulk of EMU's
   "pipeline fill" measurement.  Each `aiex.npu.write32`,
   `aiex.npu.writebd`, `aiex.npu.dma_memcpy_nd` etc. should consume
   cycles consistent with HW firmware command processing rates.  Get
   ground truth via `read_aie_reg(TIMER_LOW)` reads on a compute tile,
   straddling the runtime sequence -- sweep through enabling/disabling
   subsections.

2. **Audit compute-core boot timing.**  Compute hits first acquire at
   2085 EMU vs ~6000 HW (May 6).  Where does the ~3915 cyc of
   "boot before kernel runs" go on HW?  Suspects: PROGRAM_COUNTER reset,
   instruction-fetch warmup, `.text` section relocation, configure-time
   register write quiescence.  Same `read_aie_reg(TIMER_LOW)`
   methodology should bound this.

Both paths require functioning compute-tile `read_aie_reg` (which we
have on Phoenix per the May 5 + May 7 findings) and avoid memtile reads
(which are firmware-broken on Phoenix per the May 7 finding).

## See also

- `docs/superpowers/findings/2026-05-05-355-cycle-divergence-diagnosis.md`
  -- original gap decomposition
- `docs/superpowers/findings/2026-05-06-355a-host-latency-response.md`
  -- host_memory_latency calibration; named DMA-engine internal pipeline
  cost as the leading suspect (now refuted by this finding)
- `docs/superpowers/findings/2026-05-07-aie-rw-access-memtile-dm-half-impl.md`
  -- read_aie_reg works on compute tiles, fails on memtiles
- `tools/dma-waterfall.py` -- script used here
- `crates/xdna-archspec/src/model_builder.rs:152-198` -- DMA + stream
  switch timing constants under audit
- task #355a -- still open; this finding redirects effort
