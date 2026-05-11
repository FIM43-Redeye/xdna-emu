---
name: 'Task #26 attempt: speculative lock pre-arm for chained BD acquires breaks producer/consumer pacing'
description: Tried to close the +1 cyc per-BD residual on chained-BD-with-lock paths by speculatively submitting the next BD's acquire request to the arbiter while the prior BD is still in Transferring. The unit test for back-to-back chained BDs on a private lock pair passed cleanly (interval matches HW data cycles). But on real kernels with counting-semaphore locks shared across tiles, the lock value gets decremented thousands of cycles too early -- the consumer "outruns" the producer's filling pace. Five bridge kernels regressed with data corruption (add_one_using_dma, _diag_phase_b_add_one_instrumented, dmabd_task_queue, vec_mul_trace_distribute_lateral, vec_vec_add_memtile_init). Reverted. The +1 cyc residual remains.
type: project
---

# Task #26 attempt: speculative chained-BD lock pre-arm breaks cross-tile pacing

## TL;DR

Tried to close the +1 cyc per-BD residual identified in
`2026-05-11-emu-bd-chain-pipelining.md` (memtile MM2S 16w: EMU=16 cyc
vs HW=15 cyc) by adding a `pre_acquired: Option<PreAcquired>` slot
per DMA channel, speculatively submitting the next chained BD's
acquire request during the prior BD's `Transferring`, and consuming
the granted slot in `enter_chained_bd` to skip `AcquiringLock`.

Worked in isolation: a private-lock-pair unit test that confirmed
4 cyc interval before, 2 cyc interval after (matches HW expectation).
2882/2882 lib tests pass with the change.

Broke 5 of 7 representative bridge kernels with data corruption.
Reverted. The fundamental issue is that **speculative acquire on a
counting-semaphore lock that's shared across tiles changes the lock
value visibility window in a way that decouples lock state from
actual data availability**.

## What the attempt did

1. Added `pre_acquired: Option<PreAcquired>` to `ChannelContext`.
   `PreAcquired { for_bd, bd_lock_id, acquire_value, granted }`.
2. Extended `submit_lock_requests` pre-step pass: when a channel is in
   `Transferring` with `transfer.next_bd = Some(N)` and BD N has an
   acquire lock, submit the acquire request to the arbiter (if not
   already granted). Record the in-flight slot.
3. Added `capture_pre_acquired_grants` post-resolve pass: read the
   arbiter's `was_granted` and latch `pre_acquired.granted = true`.
4. Modified `enter_chained_bd`: if `pre_acquired` matches the next BD's
   lock AND `granted` is true, skip `AcquiringLock` entirely and go
   straight to `Transferring`/`HostPipelineLatency`. Clear the slot.
5. On `stop_channel`: warn-and-leak (no compensating release; rare
   edge case).

The wired-in flow inserted a Phase 2.5 in `step_data_movement` between
`resolve_lock_requests` and `step_all_dma`.

## Why the unit test passed

`chained_bds_with_lock_acquire_have_no_dead_cycles` was the
failing-then-passing test driving the change. It has:

- Two MM2S BDs in a single chain on one channel.
- Each BD acquires its own dedicated lock (lock 0 vs lock 1), releases
  the other. Pure double-buffer ping-pong.
- Both locks are seeded by the test harness; no other party touches
  them.

Result: BD#1 finishes at cycle 10, BD#2 at cycle 12 (interval = 2 = BD
data cycles). The speculative acquire on lock 1 grants immediately
because nothing else contends; consuming it in `enter_chained_bd`
collapses the `AcquiringLock` round-trip cycle to zero.

## Why real kernels broke

Real objectfifo locks are **counting semaphores shared between
producer and consumer tiles**. The producer (e.g., memtile MM2S) fills
buffers and emits `release` on the consumer's "full" lock; the
consumer (e.g., compute S2MM) `acquire`s the same lock to consume.

Sample log from `add_one_using_dma`:

```
DMA tile(1,1) ch0 cycle=9    SPEC_ACQ for_bd=1 lock=64 av=-1
DMA tile(1,1) ch0 cycle=9    SPEC_GRANT for_bd=1 lock=64
DMA tile(1,1) ch0 cycle=5924 SPEC_CONSUME for_bd=1 lock=64
```

The consumer speculatively decremented lock 64 at cycle 9 and held the
"acquired" state for 5915 cycles before actually using the buffer.

During those 5915 cycles, the producer was filling buffers and
incrementing lock 64. The lock value at any given cycle was 1 less
than it would have been without speculation. Anything else that reads
the lock value (a core polling a fifo depth, another channel's
arbiter contention, the producer's own decision logic via separate
locks) sees a "consumer is further ahead" state that isn't true.

The corruption pattern in `add_one_using_dma`:

```
Error in output 13 != 29
Error in output 14 != 30
...
Correct output 34 == 34
...
Error in output 10 != 58
```

First chunk wrong, middle chunks correct, last chunk wrong --
consistent with the consumer reading buffer N before the producer
has finished filling it, on the first and last iterations where the
pipeline isn't yet/anymore in steady state.

## What real silicon almost certainly does

HW does pipeline the chained-BD arbiter grant. But it must do it in a
way that **the lock's externally-observable value reflects what's
actually decremented when the consumer reads the buffer**, not when
the arbitration completes. Possible implementations:

- **Two-phase lock state**: the arbiter logically "reserves" the next
  grant during the prior BD's transfer tail, but the lock counter only
  decrements at the cycle the chained BD actually begins streaming
  data. Other readers see the "committed" value, not the "reserved"
  one.
- **Tight pipeline depth**: HW only pipelines the grant by 1-2 cycles
  (the +1 cyc residual we measure), not thousands. Limits the
  reservation window so the externally-visible mismatch is too small
  to perturb cross-tile timing.

Either approach would require redesigning the EMU's lock arbiter to
distinguish "reserved/pending" grants from "applied" grants -- a
much larger structural change than fits the scope of #26.

## What stays as-is

The +1 cyc per-BD residual on memtile MM2S 16w (EMU=16 vs HW=15)
remains. On `_diag_phase_b_add_one_instrumented` chess: 16 BDs on the
memtile side x +1 = +16 cyc total; on `vec_vec_add_memtile_init`
chess: +2 cyc per BD on memtile MM2S 16w. Compute S2MM 8w already
matches HW exactly thanks to the #13 fix.

Total impact: a few tens of cycles per kernel iteration, dwarfed by
larger residuals elsewhere (mailbox latency, cold-start, etc.).

## What I'd do differently

If revisiting:

1. Limit the speculation window to the **last data cycle of the prior
   BD** -- not the entire Transferring. This restricts the cross-tile
   timing perturbation to 1 cycle. (Tried this once during debugging;
   the "last cycle" gate eliminated the cycle savings entirely
   because the consumer's spec-grant and the chain-consume now have
   to fit in the same cycle as the prior BD's last data, which the
   FSM doesn't currently allow.)
2. Add a per-arbiter "reservation queue" distinct from the live lock
   counter. The reservation logically holds a slot but the visible
   counter only updates when the holder commits. This is the HW model
   per my analysis, but it's a 200+ line change spanning the arbiter,
   lock state, and all callers of `lock_value`.
3. Skip the optimization entirely. +1 cyc per chained BD is below the
   trace decoder's measurement noise floor for most kernels.

For now, option 3 is the right call. Documented and deferred.

## Reproducing

Failing test (now reverted but captured in `git log`):

```rust
#[test]
fn chained_bds_with_lock_acquire_have_no_dead_cycles() {
    // Two chained MM2S BDs acquiring lock 0 and lock 1 respectively,
    // releasing the other. With_acquire(N, 1) waits for lock==1 and
    // decrements. Pre-seed lock 0=1, lock 1=0 so BD#0 acquires cleanly
    // and BD#1 waits for BD#0's release.
    //
    // HW would produce FINISHED_BD events 2 cycles apart (data cycles
    // only). EMU produces them 4 cycles apart: data + ReleasingLock +
    // AcquiringLock-grant + AcquiringLock-acquired-transition.
    //
    // Asserts interval == 2.
}
```

Run with the speculative attempt: 2882/2882 lib pass, this test
passes. But run `./scripts/emu-bridge-test.sh --no-hw --peano-only
'add_one_using_dma|_diag_phase_b'` and get data-corruption failures.

## See also

- `2026-05-11-emu-bd-chain-pipelining.md` -- Task #13 fix that closed
  BdSetup + MemoryLatency overlap; established the +1 cyc residual
  this attempt tried to close.
- `2026-05-11-emu-dma-wait-mailbox-latency.md` -- separate +cyc gap,
  much larger magnitude.
- Branch `dev` at commit `85c5692` -- last green state before this
  attempt. The full pre_acquired implementation can be re-derived from
  the descriptions above if a future contributor wants to try the
  proper reservation-queue redesign.
