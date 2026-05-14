---
name: 'Task #13 reframe: EMU pays BD-chain setup serially where HW overlaps it'
description: Re-measurement of Phase C "Stage 4" with drift-corrected SoC timestamps and chunk-boundary-aware anchor pairing shows the original ~11 cyc gap was a methodology artefact (BD geometry mismatch: 16w memtile BD vs 8w compute BD). Properly paired, end-to-end transit gap is 3 cyc (within noise). The real systematic gap is per-BD chain overhead: EMU runs `BdChaining -> BdSetup -> AcquiringLock -> MemoryLatency` after the prior BD's `Transferring` ends, exposing 2-8 cyc on the critical path; HW overlaps the chain setup with the prior BD's tail and pays ~0. Verified across two kernels and three channel types (memtile MM2S, compute S2MM 8w, compute S2MM 16w). Gap shrinks as BD size grows -- with big-enough BDs the chain overhead fully hides.
type: project
---

# Task #13 reframe: BD-chain setup is serial on EMU, overlapped on HW

## TL;DR

Task #13 was framed as "Stage 4 leaves a ~9 cyc residual after Phase C
closure" -- the path memtile MM2S done -> compute S2MM done.
Re-measuring with the drift-corrected SoC timestamps (post `0b2df14`)
and properly pairing anchors by chunk boundary shows that framing was
an artefact of mismatched BD geometry. The end-to-end transit gap is
3 cyc on iter 1, within the trace decoder's noise floor.

The real systematic gap is **per-BD chain overhead**: EMU's FSM runs
the full `BdChaining (2) + BdSetup (4) + AcquiringLock (1) +
MemoryLatency (5)` = 12-cyc sequence *after* `Transferring` ends.
Real HW overlaps chain setup with the prior BD's transferring tail and
pays ~0 cyc on critical path. The gap shrinks as BD size grows --
small BDs can't hide the full 12-cyc setup behind their data, big BDs
can.

Confirmed across two kernels and three channels. Reframe task #13 from
"close 9-cyc Stage 4 residual" to "model BD-chain setup overlap."

## Where the original framing went wrong

Phase C measured "Stage 4 = memtile_mm2s_done #1 → compute_s2mm_done #1"
as 22 cyc HW / 10 cyc EMU (= 12 cyc gap; later 9 after `channel_start_cycles`
plumb-through). The implicit assumption was that those two anchors fire at
the same chunk boundary.

They don't. On `_diag_phase_b_add_one_instrumented`:

- Memtile MM2S BD = **16 words**
- Compute S2MM BD = **8 words**

One memtile MM2S BD feeds **two** compute S2MM BDs. So
`compute_s2mm_done #1` corresponds to *the first half* of
`memtile_mm2s_done #1`'s data, not the same chunk. The right pairing
is `memtile_mm2s_done #i ↔ compute_s2mm_done #2i` (chunk boundary).

Under the corrected pairing and drift-corrected SoC timestamps:

| pairing                                | HW | EMU | gap |
| -------------------------------------- | --: | --: | --: |
| (wrong) memtile #1 → compute #1        | 22 |  11 |  11 |
| (right) memtile #1 → compute #2        | 30 |  27 |   3 |

3 cyc is within the trace decoder's measurement noise. The "11-cyc
gap" was the time it takes one half-chunk of data to traverse the
inter-tile path plus the EMU paying chain overhead on the compute side
for BD #2 -- two effects entangled.

## The real systematic gap: per-BD chain cadence

Independent of cross-tile pairing, we can look at how long a single
channel takes between consecutive `DMA_*_FINISHED_BD` events when BDs
chain back-to-back (no kernel-induced backpressure). This is a pure
DMA self-measurement.

### Three data points

Phase C harness on `_diag_phase_b_add_one_instrumented` chess:

```
memtile MM2S 0 (16-word BD):
  HW:  340991 341006 341021 341036       intervals: 15 15 15
  EMU: 5959   5979   5999   6019         intervals: 20 20 20   gap +5/BD

compute S2MM 0 (8-word BD):
  HW:  341013 341021 (then kernel-bound) interval:  8
  EMU: 5970   5986   (kernel-bound)      interval:  16          gap +8/BD
```

Phase C harness on `vec_vec_add_memtile_init` chess (probe kernel,
different BD geometry):

```
compute S2MM 0 (16-word BD):
  HW:  371810 371826 (then kernel-bound) interval:  16
  EMU: 5841   5859   (kernel-bound)      interval:  18          gap +2/BD
```

### The pattern

Plot gap vs BD data cycles:

| channel + size                    | data cyc | EMU interval | HW interval | EMU critical-path overhead |
| --------------------------------- | -------: | -----------: | ----------: | -------------------------: |
| compute S2MM 8w  (Phase B)        |        8 |           16 |           8 |                          8 |
| memtile MM2S 16w (Phase B)        |       16 |           20 |          15 |                          5 |
| compute S2MM 16w (vec_vec_add)    |       16 |           18 |          16 |                          2 |

All three show **EMU > HW by a positive amount**, and that amount
**shrinks as BD size grows**. That's the smoking-gun signature of
"prior-BD tail not hiding all of next-BD chain setup."

In numbers: EMU's FSM has 12 cyc of inter-BD work (`BdChaining +
BdSetup + AcquiringLock + MemoryLatency`). HW overlaps it with the
prior BD's tail. EMU overlaps *some* but not all -- the residual shows
up as critical-path overhead. With BDs big enough (~256 words or
more, untested here but predicted) the full 12 cyc fits inside the
prior BD's transferring and the gap should collapse to 0.

HW values match "data cycles + 0" tightly:
- 16w / 1-word-per-cycle stream = 16 cyc; HW measures 15-16
- 8w / 1-word-per-cycle stream = 8 cyc; HW measures 8

That's HW running effectively at the stream-rate bound with zero
inter-BD critical path. Whatever HW's chain-setup pipeline depth
actually is, it fully hides behind prior-BD transferring for these
sizes.

## Why this matters for total runtime

The 12 cyc inter-BD setup in EMU isn't on the critical path *most* of
the time -- the FSM has stages that decrement in parallel with other
state, and the lock-acquire is usually immediate. But the residual 2-8
cyc per BD compounds:

- `_diag_phase_b_add_one_instrumented`: 8 iterations × 2 BDs/iter on
  the compute side = 16 BDs. At +8 cyc/BD that's +128 cyc total.
- Same kernel memtile side: 16 BDs × +5 cyc = +80 cyc.

Total +200 cyc on EMU vs HW just from chain overhead on the input
path. Matches the magnitude of "EMU overshoots HW by ~3200 cyc" open
question in the mailbox finding (most of that 3200 was the mailbox
constant; this chain overhead accounts for a chunk of the residual).

## Implementation (2026-05-11)

Landed in `src/device/dma/engine/stepping.rs` (`enter_chained_bd` helper
plus an `is_first_bd` dispatch in the `AcquiringLock`-acquired arm).

When `after_transfer_done` runs with `next_bd: Some(_)`, the FSM no
longer transitions through `BdChaining -> BdSetup -> AcquiringLock(cycles=lock_acquire_cycles)
-> MemoryLatency` for the chained BD. Instead it goes straight to the
next BD's `Transferring` (or `HostPipelineLatency` for shim+host_mem
BDs, which still pay DDR pipeline fill per BD). If the BD has an
acquire lock, the FSM stages in `AcquiringLock { cycles_remaining: 0,
acquired: false }` -- the post-grant countdown is folded away as part
of the prefetch, but the arbiter grant check still costs the usual two
cycles (one for the request to be picked up by the next pre-step
`submit_lock_requests` pass, one for the grant transition).

Cold start (first BD of a task, entered via `start_channel`) is
unaffected: `is_first_bd=true` keeps the `AcquiringLock` arm routing to
`MemoryLatency` with the existing `channel_start_cycles` /
`shim_ddr_cold_start_cycles` bonuses. The dispatch on `is_first_bd`
is consumed at the first MemoryLatency entry, so every subsequent BD
in the chain takes the prefetched path.

`ReleasingLock` is intentionally **not** skipped on chained BDs. An
earlier draft fired the lock release inline at `begin_completion`,
which broke data correctness in the bridge tests (Phase B's
`_diag_phase_b_add_one_instrumented` output came back all-zeros).
Keeping `ReleasingLock` as a real one-cycle stage preserves the
existing arbiter ordering invariants; only the prefetched
chain-setup window collapses.

### Measured impact

`_diag_phase_b_add_one_instrumented` chess, EMU back-to-back BD
interval re-measured with the same Phase C trace harness:

| channel        | BD size | HW | EMU pre-fix | EMU post-fix |
| -------------- | ------: | -: | ----------: | -----------: |
| memtile MM2S 0 |     16w | 15 |          20 |       **16** |
| compute S2MM 0 |      8w |  8 |          16 |        **8** |

Compute S2MM 8w lands exactly on HW. Memtile MM2S 16w is +1 cyc -- the
residual is the `AcquiringLock` grant-cycle that we can't fully hide
without bypassing the arbiter (which would re-introduce the
correctness bug). Below trace-decoder noise.

Total EMU runtime on the kernel: 14990 -> 14680 cyc (saved 310 cyc;
~2% on this workload, more on workloads with many small chained BDs).

All 2878 library unit tests pass; six varied bridge kernels pass
(`add_one_using_dma`, `add_256_using_dma_op_no_double_buffering`,
`vec_vec_add_memtile_init`, `vector_scalar_using_dma`, `cascade_flows`,
`matrix_transpose`).

## What the fix would look like

Hardware doesn't actually run BdChaining -> BdSetup -> Lock ->
MemLatency strictly serially. Once a chain is configured (`Use_Next_BD`
set), the engine can prefetch the next BD's parameters while the
current BD is still streaming data. Lock acquisition can speculate.
Memory latency on the next BD can start partway through the prior
BD's transfer.

In the FSM, this would mean:
1. Begin `BdChaining` state as soon as the *first byte* of current BD
   has been pushed to stream_out (not when last byte fires
   FINISHED_BD).
2. `BdSetup` runs while current BD's transfer continues -- separate
   state machine threads, or a "pending next BD" slot on the channel.
3. `AcquiringLock` can begin as soon as setup completes; if the lock
   isn't ready yet, that's still parallel with current BD's tail.
4. `MemoryLatency` is also parallelizable with prior BD's tail.

Simplest implementation: add a "pre-armed next BD" state. When
current BD enters `Transferring`, start a parallel countdown of the
next BD's `BdSetup + AcquiringLock + MemoryLatency`. When current BD
finishes, if the parallel countdown is also done, transition directly
to the next BD's `Transferring` (no inter-BD pause). If the parallel
countdown isn't done, pay the residual cycles.

This is a real structural change, not a knob bump. Worth implementing
because the gap is empirically confirmed and the fix mirrors how the
hardware actually behaves.

## What this invalidates

- The closure commit `3357b7c`'s claim of "Stage 4: HW=21 EMU=10 gap
  +11 vs 12" was based on the wrong anchor pairing AND legacy
  drift-contaminated `ts`. Both error sources happened to cancel
  partially. The corrected number (3 cyc, paired by chunk boundary,
  drift-corrected SoC) is essentially within noise -- `channel_start_cycles`
  plumb-through closed Stage 4 transit. Good news; that part is done.
- Task #13's framing as "Stage 4 ~9 cyc residual" is wrong. The
  residual lives elsewhere (per-BD chain overhead).

## What this confirms

- Phase C's cold-start work and `channel_start_cycles` plumb-through
  closed the dominant gap. Stage 4 transit itself is ~noise.
- The remaining systematic EMU-vs-HW gap on small-BD workloads is in
  the inter-BD chain pipeline, not in transit latency, not in lock
  arbitration, not in stream switch hop counts.

## See also

- `../../archive/findings/2026-05-10-phase-c-stage-attribution.md` -- original Phase C
  attribution; Stage 4 numbers need re-reading under the corrected
  pairing.
- `2026-05-10-trace-decoder-event-density-drift.md` -- the drift
  correction methodology that made this re-measurement possible.
- `2026-05-11-emu-dma-wait-mailbox-latency.md` -- open question 3
  ("EMU overshoots HW by ~3200 cyc on Phase B kernel"); a portion of
  that 3200 is accounted for by the chain overhead documented here.
- Commit `3357b7c` -- closure commit whose Stage 4 numbers used the
  wrong pairing. Implementation is sound; the closure-residual
  diagnostic in the commit message is wrong.
- Task #13 -- should be reframed from "close Stage 4 9 cyc" to
  "model BD-chain setup overlap."

## Reproducing the measurement

```bash
XDNA_TRACE_MODE=event_time \
XDNA_TRACE_MEMTILE_EVENTS="DMA_S2MM_SEL0_FINISHED_BD,DMA_MM2S_SEL0_FINISHED_BD" \
XDNA_TRACE_MEMTILE_SEL_CHANNELS="S2MM_SEL0:0,MM2S_SEL0:0" \
XDNA_TRACE_MEMMOD_EVENTS="DMA_S2MM_0_FINISHED_BD,DMA_MM2S_0_FINISHED_BD" \
./scripts/emu-bridge-test.sh --chess-only -v _diag_phase_b_add_one_instrumented

# Then decode both sides:
python3 tools/parse-trace.py \
  --trace-bin build/bridge-test-results/<run>/.../trace_raw.bin \
  --xclbin-mlir mlir-aie/build/.../input_with_addresses.mlir \
  --out-events /tmp/events.json

# Cadence per channel from .soc field (drift-corrected):
jq '.events[] | select(.name == "DMA_MM2S_SEL0_FINISHED_BD") | .soc' events.json
```
