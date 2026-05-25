# Stream Switch Backpressure Regression: bd_chain_repeat_on_memtile

**Date**: 2026-05-25
**Status**: Resolved — root cause was DMA-side head-of-line blocking on a
shared `stream_out` queue, not a lock-arbiter modeling gap.
**Related commits**:
- stream_switch local-route backpressure (commit 92fac90, surfaced the bug)
- archspec external master FIFO = 4 per AM020 (this fix part 1)
- dma: per-channel stream_out (this fix part 2)

---

## Context

`memtile_dmas/blockwrite_using_locks.chess` was wedging with TIMEOUT at 600s,
producing the `lock_value=0 bd_lock=64` symptom on memtile (1,1) ch0 in
`AcquiringLock`. Root cause was upstream: the memtile MM2S BD self-chained on
its own cons-lock and produced data forever. On silicon, that would
backpressure-stall once shim S2MM ch0 completed its 4096-word receive and the
FIFOs filled. In the emulator, `stream_switch::step()` popped slave ports
unconditionally into the unbounded `switch_pipeline: Vec<InSwitchWord>`
regardless of downstream capacity, so upstream MM2S never saw backpressure
and the engine never reached Halted/Stalled.

Fixed by gating the slave pop on the per-route in-flight budget
`peer.latency + master.fifo_capacity` (counting `pipeline_to_master + master
FIFO occupancy`). For multicast: all destinations must have room.

After the fix: `memtile_dmas/blockwrite_using_locks.chess` passes in 16.8s.
The DMA MM2S now correctly stalls in `Transferring` once the downstream FIFO
chain fills; natural completion fires when the sync target (shim ch0) goes
idle; the engine reaches Stalled and the run loop exits cleanly.

All 3207 lib tests pass (+1 new backpressure unit test
`test_per_route_backpressure_caps_pipeline_at_latency_plus_master_fifo`).

---

## Regression: bd_chain_repeat_on_memtile

`bd_chain_repeat_on_memtile` (chess and peano) was PASS before the fix, now
FAILs with `Error count: 32569` of 32768 expected bytes (only 199 correct).
Engine wedges with `halt_reason=wedge_recovered cycles=108223` after running
all 4 channels into `AcquiringLock` on producer-side locks.

### Topology

Two-column objectFifo pipeline from `bd_chain_repeat_on_memtile/aie2.py`:

```
shim col 0 row 0
  MM2S ch0 (in)   ─── object_fifo "in" iter_count=8 ────►  memtile col 0 row 1
                                                              │
                                            ┌─ split_0 iter=8 repeat=2 ──► compute (0,2)
                                            │                                  │
                                            └─ split_1 iter=8 repeat=2 ──► compute (0,3)
                                                                               │
                                            ┌─ join_0 iter=16 ◄────────────────┘
  S2MM ch0 (out)  ◄── object_fifo "out" iter_count=16 ─── memtile (0,1)
                                            └─ join_1 iter=16 ◄────────────────┘
```

Compute tiles run a `passThroughLine` kernel; the test verifies each input
chunk appears `REPEAT_COUNT=2` times in the output.

### Pre-fix vs post-fix per-channel beat counts

Memtile (1,1) MM2S beats by slave port:

| slave | pre-fix | post-fix | role |
|-------|---------|----------|------|
| 0 | 4096 | 1302 | split_0 → compute (1,2) |
| 1 | 4096 | 832 | split_1 → compute (1,3) |
| 2 | 8192 | **18** | out → shim ch0 (this is the bottleneck) |

Shim ch0 S2MM beats:

| | pre-fix | post-fix |
|---|---|---|
| beats received | 8192 | **18** |
| BD cycles | 11667 | (never completes) |

Post-fix shim ch0 receives the first 18 data words correctly
(`0x03020100`, `0x07060504`, ..., `0x47464544`), then dries up. The first 18
bytes of `bo_out` are valid; the rest stays at memset(0). Verify counts
32569 byte mismatches against the expected `i % 256` pattern.

### Stuck state

End-of-run channels in `AcquiringLock`:

```
tile(1,1) ch0 bd_lock=66 target=Own(2) — split_0 prod lock
tile(1,1) ch1 bd_lock=68 target=Own(4) — split_1 prod lock
tile(1,1) ch2 bd_lock=70 target=Own(6) — join_0 prod lock
tile(1,2) ch0 bd_lock=0  target=Own(0) — compute's split_0 cons-data lock
```

Memtile MM2S ch8 (the OUT fifo producer) ends in `Transferring` with no
forward progress — slave[2] backpressured, route `slave[2] → master[7]` at
budget, downstream `master[7] → shim slave[14] → shim master[4] → shim ch0
stream_in` chain has room but the route is gated by something. Shim ch0 stays
in `Transferring` (consuming at 1 word/cycle, `stream_in_len=1` per beat),
ready to receive more.

The same `AcquiringLock` deadlock pattern exists at the *end* of the pre-fix
run too (compute (1,2)/(1,3) ch2 stuck on `Own(3)`). Pre-fix it surfaces only
*after* all 8192 OUT words have flowed; post-fix it surfaces after 18.

### Hypothesis: leaky pipeline masked a lock/backpressure race

The leaky pre-fix pipeline let MM2S produce data ahead of the downstream
arbiter / lock pipeline. Pre-fix:

- Producer MM2S emits 28,672 total beats across the column (vs 4919 post-fix).
- Most go to trace (2304 to shim ch1 trace port), the rest spread across
  split/join/out.
- 8192 OUT beats reach shim ch0, BD completes, sync satisfies, run halts
  cleanly. The trailing `AcquiringLock` state on a few channels is benign —
  TDR sees Halted/Stalled with `syncs_satisfied=true` and classifies natural
  completion.

Post-fix the producer can't run ahead of the lock pipeline. When the OUT
fifo's first iteration completes and tries to start its second iteration, it
needs the next prod-lock release from somewhere upstream. With backpressure
holding the producer in `Transferring` (no `Stalled_Stream_Backpressure`
asserted today), the lock-vs-stream interaction stalls before any further
data words flow.

The leaky behavior masked this in two ways:

1. **Buffer overshoot.** Unbounded `switch_pipeline` let many words sit
   between producer and consumer, so when the producer momentarily stalled
   on a lock, the downstream had backlog to drain — appearance of forward
   progress.
2. **Race on lock release vs data consumption.** With proper backpressure,
   the channel state machine and the lock arbiter need to reach a fixed
   point in a single cycle. If the channel is in `Transferring(stalled
   on slave full)` and the lock release path isn't observing that, the
   tile arbiter doesn't see the activity needed to grant the next acquire.

Investigation should start at:

- `dma::engine::stepping::transfer_mm2s` — does the channel emit
  `Stalled_Stream_Backpressure` (bit 4 of `DMA_MM2S_Status_0`) when its
  `stream_out` push to slave is rejected? The coverage doc claims it does
  (`docs/coverage/aie2/...`), but the stuck channels' lack of progress
  notification to the lock arbiter suggests the signal isn't being
  observed by whatever drives the next BD-chain step.
- `array::routing::route_dma_to_tile_switches` — when `slave.can_accept()`
  returns false, is `cycle_stalled` set on the MM2S side? Today it just
  retains the data in `stream_out`.
- Lock arbiter: is there a path where an MM2S channel in stalled-Transferring
  state would still need to release a lock (e.g., release pipelined with
  last data cycle, but data isn't flowing)?

### Trade-off

Landing the fix as-is because the behavior is now silicon-faithful. The
blockwrite_using_locks wedge converted to a `bd_chain_repeat_on_memtile`
deadlock: both are real modeling gaps but the new failure surfaces a deeper
gap (lock-arbiter / backpressure interaction) that was being papered over
by unbounded pipeline buffering.

### Next steps

1. Characterize whether `Stalled_Stream_Backpressure` is fired on the
   memtile MM2S OUT (ch8) at the point it stops producing. If not, the
   signal-emission path needs work.
2. Check the lock arbiter's view of stalled channels: a producer that
   can't push data still needs to advance its release-after-transfer
   bookkeeping when the chain resumes; verify the release-after-transfer
   path doesn't require contiguous `Transferring` cycles.
3. If both look right, the deadlock is intrinsic to the test's BD chain
   pattern and the fix is the resolution itself — `bd_chain_repeat_on_memtile`
   should be reclassified as a test that depends on faster-than-silicon
   propagation. (Would want HW correlation before concluding this.)

---

## Resolution (2026-05-25)

The hypothesis above was wrong.  The deadlock was **not** in the lock arbiter
or pipeline-vs-backpressure modelling, it was a **DMA-side head-of-line
blocking bug** that the new backpressure fix exposed.

### What actually happened

A cycle-by-cycle state dump (added to `step_data_movement`, removed once
diagnosis was complete) revealed at the wedge:

```
ch8=Transferring(76/512) stream_out=4 chans=[6, 6, 6, 6]
slv0=4 slv1=0 slv2=0
```

- Memtile DMA's `stream_out` was full (4 / 4)
- All 4 slots held data for ch 6 (`split_0`), whose slave[0] was full
  because compute(1,2) was legitimately backpressured
- Ch 8 (OUT) wanted to push to slave[2] -- which was empty -- but
  couldn't, because the capacity check on the **shared** `stream_out`
  queue saw `len == 4` and returned `Stalled` for *every* MM2S channel

So one stalled MM2S channel (ch 6) was freezing all the others (ch 7, ch 8)
on the same tile.  This is a classic head-of-line blocking issue at the
producer side.  AM020 Ch2 specifies "Local slave ports are ... a 4-deep
FIFO" *per port*, with independent credit-based flow control from each
MM2S channel to its own slave port -- not a shared 4-deep queue across
channels.

### Fix

Two parts:

1. **External master FIFO depth = 4** (per AM020).  Local master is 2-deep,
   external master is 4-deep -- we previously modelled both as 2.  Added a
   separate `STREAM_EXTERNAL_MASTER_FIFO_DEPTH` constant and dispatched on
   `PortType::is_external()` in `StreamPort::new`.  This moved the wedge
   point by exactly +1 word (18 -> 19), confirming the budget formula was
   correct but FIFO sizing was off.

2. **Per-channel `stream_out`** -- the actual fix.  Changed
   `DmaEngine::stream_out` from `VecDeque<StreamData>` to
   `Vec<VecDeque<StreamData>>` indexed by `channel - s2mm_count`.  Each
   MM2S channel gets its own 4-deep FIFO.  Capacity gates
   (`output_fifo_capacity`) became per-channel via
   `can_push_stream_out_for_channel`.  `route_dma_to_tile_switches` now
   iterates channels independently, resolving the destination slave once
   per channel (it's a function of channel index, not data).  Retained
   words go back to the channel's own queue, preserving FIFO order
   per-channel without coupling channels to each other.

### What was wrong with the pre-diagnosis hypothesis

The leading edge of the stall was MM2S ch 8 stuck at 18 words, with all
other channels also mid-transfer.  This *looked* like a downstream lock
deadlock cascading upstream, but the data flow was actually the opposite:
ch 6 stalled (legitimately, downstream backpressure), ch 6's data filled
the shared `stream_out`, and then ch 7 / ch 8 inherited the stall without
any of their own downstream chains being affected.  All FIFOs along OUT's
downstream chain (slave[2], pipeline, master[7], inter-tile, shim slave[14],
shim pipeline, shim master[4], shim ch 0 stream_in) were EMPTY at wedge
time -- the chain had drained completely.  We just couldn't push more
into it because of the head-of-line block.

The lock-arbiter hypothesis was wrong because all downstream queues were
empty; there was no cascade.

### Verification

- `cargo test --lib`: 3207 passed (no regressions vs the per-port-type
  FIFO change baseline).
- `bd_chain_repeat_on_memtile.chess`: PASS (was failing 32569/32768).
- `memtile_dmas/blockwrite_using_locks.chess`: PASS (still passes -- the
  backpressure fix from the prior commit is preserved).

Broader bridge sweep TBD.
