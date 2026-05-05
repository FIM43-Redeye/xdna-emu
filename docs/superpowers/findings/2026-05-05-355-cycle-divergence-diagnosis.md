---
name: '#355 EMU/HW cycle divergence -- diagnosis on add_one_using_dma'
description: Decomposing the apparent 12.7x EMU/HW trace-span ratio. Within the kernel's execution window the two are roughly equal; the visible gap is dominated by (a) EMU's DMA pipeline fill running ~2.6x faster than HW (host_memory_latency / stream-switch fabric likely under-counted), and (b) EMU's trace controller starting AFTER the fill, so HW's pre-fill stall events never get captured in EMU. Splits into two distinct fixes.
type: project
---

# Cycle divergence on add_one_using_dma -- decomposition

## TL;DR

Surface symptom: HW trace spans 7484 cyc, EMU spans 587 cyc -- a 12.7x
gap that read like "EMU is 13x too fast". The actual decomposition:

| Phase                                | HW (cyc) | EMU (cyc) | Ratio |
|--------------------------------------|---------:|----------:|------:|
| Pipeline fill (1st lock wait -> success) | ~6000   | 2319     | 2.6x  |
| Kernel execution (16 lock acquires)  | ~500    | ~587     | ~1.0x |
| Trace span (first event -> last)     | 7484    | 587      | 12.7x |

The 12.7x trace-span ratio is **misleading**: the trace span includes
HW's pre-fill LOCK_STALL/PERF_CNT_0 events (one set per ~1024 cyc during
the wait), but EMU's trace controller doesn't start until *after* the
pipeline has filled, so those events never appear in EMU. The two
problems compound visually but are independent.

## Where the data came from

```bash
$ ls build/bridge-test-results/latest/add_one_using_dma.chess.{hw,emu}/
trace_config.json  trace_raw.bin
```

Decoded with `tools/parse-trace.py` -> events JSON, cycles scalar.

**HW events (chess HW):** 67 events.
- slot 0 PERF_CNT_0:           8 events, 1024-cyc spaced (perf counter rollovers)
- slot 6 LOCK_STALL:           44 events, mostly 1024-cyc spaced
- slot 7 INSTR_LOCK_ACQUIRE_REQ: 15 events, all clustered at the END
  (ts=486229..486726, a 497-cyc burst)

The first 6000 cyc of HW trace contain only s0/s6 events at ~1024-cyc
periodicity -- those are the perf counter rollover and corresponding
stall-during-stall events. After ~6000 cyc, the first acquire succeeds
and the kernel runs in 497 cyc.

**EMU events (chess EMU):** 19 events at ts=2337..2924, span 587.
- slot 7 INSTR_LOCK_ACQUIRE_REQ: 16 events
- slot 4 MEMORY_STALL:           3 events
- slot 0/6:                      0 events

EMU recorded the kernel-execution window (16 acquires over 587 cyc)
but missed everything before the trace controller started.

## Pipeline fill measurement

Counted from the EMU bridge log:

```
log line 183:  LockAcquire raw=49 ... current=0 -> WAIT
log line 7294: LockAcquire raw=49 ... current=1 -> 0 SUCCESS
```

Counting per-cycle DMA `check_acquire_granted` lines on the consumer
channel (`tile(0,2) ch2`) between the two: **2319 cycles**.

HW pipeline fill, derived from the trace: ~6000 cyc (PERF_CNT_0
periodicity of 1024 x 6 anchors before first acquire).

**Ratio: 2.6x.** EMU is a little under 3x faster than HW for the
host-DDR -> shim -> memtile -> compute chain to fill.

## Why is EMU's fill faster?

Cycle accounting from `crates/xdna-archspec/src/model_builder.rs:152`:

```rust
DmaTiming {
    bd_setup_cycles: 4,
    channel_start_cycles: 2,
    words_per_cycle: 4,
    memory_latency_cycles: 5,
    lock_acquire_cycles: 1,
    lock_release_cycles: 1,
    bd_chain_cycles: 2,
    host_memory_latency_cycles: 100,   // <-- the suspect
}
```

Per-stage round-trip cost: 4 + 2 + 1 + 5 + 2 + 1 + 2 = 17 cyc.
Chain: shim(17 + host_memory_latency=100) + memtile-in(17) + memtile-out(17) + compute(17) = 168 cyc theoretical fill.

Observed EMU fill: 2319 cyc, so the model picks up 14x the headline-sum
somewhere (probably in serialization of multiple BDs, lock-wait
re-checks, and stream-switch fabric stalling on backpressure). Even so,
it's **2.6x short** of HW.

Most likely culprits:
1. **host_memory_latency_cycles=100** is too low. PCIe + NoC + DDR
   round-trip on Phoenix-class systems is closer to 500-1000 cyc per
   transaction at this transfer size. The 100-cyc value was an early
   guess; we never validated it against HW.
2. **No stream-switch fabric per-hop latency** in the data path. The
   `StreamSwitchTiming.local_to_local_latency: 3` etc. constants exist
   in the timing model but aren't applied as cycle costs to data flowing
   through the switch -- they're modelled implicitly by the FSM
   transitions of producer/consumer DMA, which don't account for
   in-flight buffering.
3. **Memtile DMA broadcast/fanout**: when memtile produces to compute,
   any per-port arbitration latency isn't counted.

## Trace-controller-start mismatch (orthogonal issue)

Even if we fix the cycle accounting, the trace span gap won't fully
close because the EMU trace controller starts **after** the first
acquire succeeds, missing the preceding stall window entirely.

In HW, the trace controller is configured at xclbin-load time. By the
time the first acquire fires, the controller is already running and
recording events from its internal cycle counter. Stall-during-stall
events fire at the perf-counter rollover boundaries (every 1024 cyc),
producing the periodic LOCK_STALL events visible in HW's pre-acquire
window.

In EMU, the trace controller seems to either (a) not be running yet
when the first stall happens, or (b) emit only one LockStall event on
the initial WaitLock entry rather than periodically while still in
WaitingLock state.

Looking at `src/interpreter/execute/cycle_accurate.rs:556-559`:

```rust
ExecuteResult::WaitLock { .. } => {
    ctx.timing_context_mut()
        .record_event(start_cycle, EventType::LockStall { cycles: 1 });
}
```

LockStall is emitted ONCE on the WaitLock execute_result. The
re-checking path in `src/interpreter/core/interpreter.rs:590` (for
CoreStatus::WaitingLock) records a 1-cycle stall but does NOT emit
another trace event. So in EMU, a long lock wait produces exactly 1
LockStall event regardless of duration.

In HW, the perf counter (configured to count ACTIVE_CORE) increments
while the core is stalled (because stall doesn't reset core-active
status), and the rollover at 1024 fires PERF_CNT_0. The resulting
event chain looks like periodic LOCK_STALL in the trace.

## Two distinct fixes

This finding effectively splits #355 into two sub-tasks:

### #355a -- DMA pipeline fill cycle accounting

Calibrate host_memory_latency, stream-switch fabric latency, and
memtile broadcast costs to bring EMU pipeline fill closer to HW.
Strategy:
1. Pick 3-4 representative tests with different DMA patterns
   (single-input add_one, multi-input cascade, large-buffer matmul).
2. Measure HW pipeline fill on each (LOCK_STALL rate during pre-acquire
   window scaled by perf-counter threshold).
3. Measure EMU pipeline fill via the bridge-log counting trick used here.
4. Solve for the model parameters that minimize the cross-test residual.
5. Validate: bridge-test sweep cycle ratio should drop from ~12.7x to
   <2x on this test, and similar improvements elsewhere.

### #355b -- EMU trace controller pre-kernel event capture

Make EMU emit periodic LOCK_STALL events during long lock waits, the
way HW's perf-counter-driven trace-event chain does. Probably:
1. Add a per-cycle hook in the WaitingLock loop that increments a
   stall-cycle counter.
2. When the counter crosses N (matching the HW perf-counter threshold,
   typically 1024), emit a LockStall trace event and reset the counter.
3. Verify: EMU's pre-kernel slot 6 event count should match HW's within
   ~1 event.

Overlaps strongly with #321 (trace-stop timing) and #353 (LOCK_STALL
emission) -- consider treating as one workstream "EMU trace event
fidelity".

## What we did NOT do today

- Quantitative cross-test calibration (just one test, point estimate).
- Code changes to host_memory_latency or trace event emission.
- Verification on peano-side or other test variants.

The bridge-trace-runner lifecycle bug from `2026-05-05-aie-rw-access-firmware-actually-supported.md`
is in the way of doing on-NPU calibration via PERF_COUNTER0 readback;
that path would give us ground-truth cycle counts to anchor the
calibration. Worth fixing first before attempting #355a in earnest.

## See also

- task #355 (this finding)
- task #353 (EMU LOCK_STALL emission -- subsumed by #355b)
- task #354 (EMU PERF_CNT_0 emission -- related; the perf counter does
  fire in HW during stalls, so #355b would emit those as well if the
  trace controller is configured for slot 0)
- task #321 (trace-stop timing) -- different stage, but same trace
  controller code
- `2026-05-04-control-path-cycle-calibration.md` -- prior calibration
  work for control-path packets; the DMA-path equivalent has not been
  attempted
- `crates/xdna-archspec/src/model_builder.rs:152-178` -- the timing
  model values in question
- `src/interpreter/execute/cycle_accurate.rs:556-559` -- single-shot
  LockStall emission point
- `src/interpreter/core/interpreter.rs:590` -- WaitingLock retry path
  that should be emitting periodic events
