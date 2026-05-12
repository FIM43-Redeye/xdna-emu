---
name: 'EMU compute trace cycle counter jumps ~247k mid-stream (NOT a DMA pipeline rate gap; original framing was wrong)'
description: On add_one_using_dma the EMU compute trace emits a tight cluster of events at cycle ~247060, even though the entire EMU coordinator run is 14673 cycles. The original framing of this finding -- "EMU DMA pipeline 2.4x faster than HW, causing missed stall windows" -- was based on misinterpreting these inflated trace timestamps as real timing. A clean HW+EMU bridge run on 2026-05-11 showed the event-content comparison is CLEAN (6/6 pre-jump pairs within +/-5 cyc), and the visible divergence is a cycle-stamp jump of ~247k cyc inside the EMU trace pipeline. The DMA pipeline isn't too fast, the per-buffer rate isn't 2.4x off, and the recent #26 fix didn't over-correct. The actual bug is in the EMU's compute-tile trace timestamp generation, likely a tile-local cycle counter that desynchronizes from the coordinator clock partway through the run.
type: project
---

# Originally: EMU DMA pipeline too fast. Actually: EMU compute trace cycle-stamp bug.

## TL;DR (revised 2026-05-11)

**The "2.4x DMA gap" framing in the original write-up was wrong.**

A clean HW + EMU bridge run on `add_one_using_dma` after the doc was
written shows:

- Bridge test: PASS, both sides
- Trace comparison: **CLEAN**
- First 6 event pairs: within +/-5 cyc (HW vs EMU)
- Divergence at pair 5: EMU's compute-tile trace cycle jumps from
  cy 18 to cy 247060 between consecutive events, then all subsequent
  events cluster in cy 247060-247170
- EMU coordinator total: **14673 cyc** (XDNA_EMU_STATUS) -- the
  "247060 cyc" event timestamp is ~17x larger than the entire run

That jump is a trace-pipeline artifact, not a real timing gap. The
DMA pipeline rate, the per-BD turnover, and the stream-switch
latencies are all roughly correct. The previous "HW = 27340 cyc"
total in this doc was derived from the inflated EMU trace stamps,
not real cycles -- the actual HW kernel-activity window is in the
low single thousands of cycles, similar to EMU.

## What was originally claimed (and is now retracted)

- ❌ "EMU DMA pipeline is 2.4x faster per buffer than HW"
- ❌ "Compute core misses 5/6 stall windows because producer is too
  fast"
- ❌ "Stream-switch traversal latency is the most likely single-knob
  contributor"
- ❌ Recommendations to recalibrate DMA / lock-arbiter / DDR rates

These were derived from comparing EMU's inflated compute-tile trace
timestamps against HW's normal timestamps. The comparison wasn't
apples-to-apples, so the gap it showed wasn't real.

## Update 2026-05-12: spurious MEMORY_STALL root cause + fix

One of the three "real fidelity bugs" surfaced after the trace-decoder
mode fix (HW=0 / EMU=1 MEMORY_STALL interval on `add_one_using_dma`)
is now closed.

**Root cause**: `transfer_s2mm()` in `src/device/dma/engine/stepping.rs`
recorded the bank access into `cycle_dma_banks` *before* checking
whether stream data was available. When the S2MM was stalled waiting
for upstream data (which is the steady state during the kernel's
warm-up window, ~6500 cyc on add_one_using_dma), every cycle it would
phantom-claim a memory access at the BD's start offset. The conflict
detector then ANDs `cycle_core_banks & cycle_dma_banks` and fires
CONFLICT_DM_BANK_N + MEMORY_STALL whenever the core happens to load
from the same bank -- e.g., a constructor-table load at offset 0x488
(bank 0) coinciding with a stalled S2MM "writing" to offset 0x400
(also bank 0).

In HW the stalled DMA does not issue a memory transaction, so no bank
arbitration occurs and no conflict event fires.

**Fix**: Move `cycle_dma_banks |= banks_for_access(...)` to *after*
the `stream_in_count_for_channel(channel) < words_needed` check (and
inside the decompression branch, after the `has_stream_in_for_channel`
check). MM2S already gates correctly via the
`stream_out.len() >= output_fifo_capacity()` check happening *before*
`transfer_mm2s` is called in `do_transfer`, so no change there.

After the fix, `add_one_using_dma` chess: 0 MEMORY_STALL events, 0
CONFLICT_DM_BANK_N events (down from 1 stall interval + 2 CONFLICT
events), matching HW. 2885/2885 lib tests pass. Bridge test passes.

## What is actually true

1. The bridge test passes on Chess for `add_one_using_dma`.
2. The trace-comparison's pre-divergence portion (the apples-to-apples
   part) is clean.
3. **The visible "divergence" is a trace-decoder mode mismatch in the
   bridge pipeline.** Both HW and EMU encode the compute-tile trace
   in mode 1 (EventPC -- the Trace_Control0 value 0x79860081 has
   low bits = 0b01 = mode 1, confirmed by 0xf1 Start opcode in both
   raw trace BOs). But `scripts/emu-bridge-test.sh` invokes
   `parse-trace.py` without `--trace-mode auto`, so the script
   defaults to `--trace-mode event_time` (mode 0). The mode-0
   decoder happily parses mode-1 EventPC frames as Multiple/Single
   opcodes with cycle deltas, accumulating bogus cycle counters
   like 247060 on the EMU side and noisy-but-smaller values on the
   HW side. Both sides are misdecoded, but in different ways,
   producing the false visible "gap."
4. **The real EMU trace fidelity issues, surfaced once mode-1 is
   decoded correctly:**
   - HW emits LOCK_STALL ~4400 times at PC 832 (every WaitLock
     poll cycle). EMU emits LOCK_STALL twice at PC 0.
     `event_pc(LockStall) -> None` in `src/trace/mod.rs:865`, so
     mode-1 encoding stamps it with PC=0 instead of the WaitLock
     instruction PC. Also EMU is edge-triggered (one event per
     entry into WaitLock) rather than level (one per polling cyc).
   - HW PC for `INSTR_LOCK_ACQUIRE_REQ` is 828; EMU PC is 816 (12
     instructions / 24 bytes off). Same compiled binary, so this
     is an EMU-side bookkeeping bug -- likely capturing PC at the
     wrong pipeline stage (pre-decode vs post-issue).
   - HW emits MEMORY_STALL at... unclear in the raw data. EMU
     emits MEMORY_STALL at PC=0 (sentinel) three times.
5. Two sensitivity probes (stream-switch latency +5x, per-BD turnover
   +50 cyc) both moved EMU total cycles by less than 1%, confirming
   those are not the gap drivers either.

## Probes I ran while pursuing the wrong framing

For the record, since the data is informative on what *doesn't*
drive EMU/HW divergence:

### Probe 1: Stream-switch traversal latency

Bumped `STREAM_LOCAL_TO_LOCAL_LATENCY` 3->15,
`STREAM_LOCAL_TO_EXTERNAL_LATENCY` 4->20, and
`STREAM_EXTERNAL_TO_EXTERNAL_LATENCY` 4->20 (5x amplification across
the board). Re-ran `add_one_using_dma` on Chess:

- Baseline: 14673 cyc
- 5x latency: 14762 cyc
- Delta: +89 cyc (0.6%)

The constants are correctly wired (intra-tile latency in
`stream_switch::step`/`switch_pipeline`, inter-tile in
`propagate_inter_tile`/`ROUTE_PER_HOP`), but the +89 cyc is all
head-of-line / cold-start. Once the pipeline is primed, words flow
at the slowest stage's rate independent of traversal latency. Not a
per-buffer knob.

### Probe 2: Per-BD turnover under backpressure

Added +50 cyc post-grant cooldown in `enter_chained_bd` (stepping.rs
line 441). This injects 50 cyc after each chained-BD lock grant
before transfer begins. Re-ran:

- Baseline: 14673 cyc
- +50 cyc cooldown: 14742 cyc
- Delta: +69 cyc (0.5%)

The probe was absorbed because the DMA channels in
`add_one_using_dma` are *consumer-bound*: each channel waits ~130
cyc per buffer for compute to release the lock, and the +50 cyc
post-grant just compresses that wait time on the next acquire. Not
a per-buffer knob either.

### Probe 3: Compute bundle latency

Added `ctx.cycles += 2` after `record_instruction(1)` in
`cycle_accurate.rs::execute_internal`. Zero effect on total cycles:
the coordinator's `total_cycles` advances per coordinator step, not
per core bundle, and `add_one_using_dma` is mailbox-bound (~8000
cyc of `#24` modeling at the tail). Slowing the core internally
doesn't slow the coordinator wall-clock.

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

### 1. Stream switch traversal latency (RULED OUT as a single-knob fix)

Investigated 2026-05-11 after the original write-up. Constants are
wired: `STREAM_LOCAL_TO_LOCAL_LATENCY=3` and
`STREAM_EXTERNAL_TO_EXTERNAL_LATENCY=4` (model_builder.rs:210-213) are
applied per-route in the intra-tile pipeline
(`stream_switch::step` -> `switch_pipeline` with `peer.latency`
cycles_remaining), and `ROUTE_PER_HOP=4` is applied to the
inter-tile pipeline (routing.rs:1063 `propagate_inter_tile`).

Sensitivity test: bumped all three to 15/20/20 (5x amplification)
and re-ran `add_one_using_dma` on Chess build. Cycle count moved
from 14673 -> 14762, a delta of +89 cyc (0.6%). At 1x amplification
the realistic delta would be ~18 cyc.

Conclusion: stream-switch latency is a head-of-line / cold-start
cost, not a per-buffer steady-state cost. The pipeline is largely
backpressure-bounded once primed, so adding traversal latency just
shifts when the first word arrives; subsequent words flow at the
slowest stage's rate. To move the per-buffer rate we need a knob
that adds cost *per word* or *per BD turnover*, not per pipeline
fill.

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

## Downstream effects (RETRACTED)

The original framing claimed three downstream symptoms (#353, #354,
#355) all rooted in a "DMA too fast" cause. With the bridge run
showing CLEAN traces and the pre-jump pairs matching within +/-5
cyc, none of this is real:

- LOCK_STALL counts (22 vs 24) differ by 2, well within trace-frame
  coalescing noise -- not 1/6 of the stall windows missing.
- PERF_CNT_2 counts are within ~10% on both sides.
- The "27340 vs 14655 total" gap was the inflated EMU stamps vs
  HW stamps -- the actual coordinator total is 14673 cyc, and HW
  is similar order of magnitude (TBD precise measurement).

The fix surface is **not** "make EMU's DMA pipeline slower". The
fix surface is **find why the EMU compute-tile trace clock jumps
to ~247060 partway through the run**.

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

## What to do next (revised 2026-05-11 after isolating root cause)

Three concrete bugs surfaced, in order of impact:

1. **Bridge script uses wrong trace-decoder mode** (5-minute fix).
   `scripts/emu-bridge-test.sh:875` calls `parse-trace.py` without
   `--trace-mode auto`. Change it. Once both HW and EMU traces are
   decoded with their actual modes (compute = mode 1, shim/memtile
   = mode 0), the false "247060 cycle gap" disappears and the real
   event-content divergences become legible. This unblocks every
   downstream trace-fidelity investigation.

2. **EMU emits LockStall edge-triggered with no PC** (real fidelity
   bug). HW emits LOCK_STALL on every cycle of WaitLock polling
   (~4400 events in `add_one_using_dma`); EMU emits it twice. The
   variant `EventType::LockStall { cycles: u8 }` carries no PC, so
   `event_pc()` returns None and the mode-1 encoded frame uses
   PC=0 instead of the WaitLock instruction PC. Two sub-fixes:
   - Add a PC field to LockStall (and MemoryStall, StreamStall) so
     mode-1 trace stamps carry the real waiting instruction PC.
   - Change emission from edge-triggered (once per stall entry) to
     level (once per cycle while WaitLock is unresolved). The
     previous attempt at this exploded LockStall count 200x; the
     trace-pipeline coalescing investigation (item 4) needs to land
     first to understand why.

3. **INSTR_LOCK_ACQUIRE_REQ PC is off by ~12 instructions** (smaller
   fidelity bug). HW=828, EMU=816. Same compiled binary, so it's
   an EMU bookkeeping issue -- probably capturing PC at a different
   pipeline stage than HW does (pre-decode vs post-issue, or
   pre/post-delay-slot resolution). Lower priority than (1) and (2)
   but trivial once those are unblocked.

## Update 2026-05-12: INSTR_LOCK_ACQUIRE_REQ count investigation

Dug into the "HW 15 / EMU 16" off-by-one count mismatch. Cross-kernel
data:

| Kernel | Dynamic acquires (EMU) | HW count | EMU count |
|---|---|---|---|
| add_one_using_dma | 16 | 15 | 16 |
| add_one_objFifo | 16 | 15 | 16 |
| add_one_ctrl_packet | 5 | 5 | 5 |

First hypothesis: HW only fires INSTR_LOCK_ACQUIRE_REQ on immediate
success (not on post-stall resumption). Implemented retry-suppression
in EMU via a `lock_acquire_retry_pending` flag on ExecutionContext --
matched add_one_using_dma to 15/15 but broke add_one_ctrl_packet
(undercounted to 3/5: EMU has 3 stall-resolved acquires that HW DOES
capture). Reverted.

Real conclusion: HW's "drop 1" pattern is specific to kernels with a
long head-of-line stall followed by tightly-packed acquires. Likely
a trace-controller pipeline artifact -- either the first event is
dropped during pipeline startup, or the last event is dropped because
trace shutdown happens before it flushes. Same root-cause family as
the #33 PC-pipeline-depth bug (HW=0x33C vs EMU=0x330): both are
manifestations of HW's trace pipeline timing that EMU doesn't model.

EMU's count of 16 is the *logically correct* count of acquires
issued. HW's 15 is an underreport. **No EMU code change is the right
answer here** -- modeling HW's trace-controller pipeline drop is
much more work than the +1 discrepancy is worth.

## Update 2026-05-12: #33 PC pipeline depth CLOSED

The "INSTR_LOCK_ACQUIRE_REQ PC is off by ~12 instructions" item from
the 2026-05-11 next-steps list is now fixed. Cross-kernel verification
after the patch (HW vs EMU on the post-fix `.so`):

| Kernel              |  HW PC | EMU PC | Match |
|---------------------|-------:|-------:|------:|
| add_one_using_dma   |  0x33C |  0x33C | ✓ |
| add_one_objFifo     |  0x31C |  0x31C | ✓ |
| add_one_ctrl_packet |  0x60C |  0x60C | ✓ |

Pre-fix, `add_one_using_dma` was 0x330 (the acq's own PC). The other
two were not measured pre-fix.

The +12-byte offset on `add_one_using_dma` (3 bundles past `acq`: `ret`
+ 2× `nop` + the 4th delay-slot `nop`'s issue PC) looks like a constant
in bytes but it's actually a constant in *cycles*: the trace controller
observes the acq-success signal at a late pipeline stage and stamps the
frame with the current issue-stage PC; by the time the signal lands,
the core has moved 4 cycles past the `acq` (one `ret` bundle + three
1-cycle delay-slot nops, all of which issue at 1 cycle each in this
helper layout). +12 bytes is just what those four 1-cycle bundles
happen to sum to in the standard `aie_runtime::acquire` helper. A
kernel with different post-acq bundle widths would see a different
byte offset but the same 4-cycle delay -- which is why the fix
modeled cycles, not bytes.

The fix lives in three files:

- `src/interpreter/state/timing_context.rs`: a `pending_deferred_pc`
  queue holding `(trigger_cycle, record_at_cycle, kind)`, plus
  `record_event_with_pc_delay` (scheduler) and `drain_deferred_pc`
  (drainer that stamps the event with the current issue PC). The
  pipeline-depth constant lives here as `TRACE_PC_PIPELINE_DEPTH = 4`.

- `src/interpreter/execute/control.rs`: the InstrLockAcquireReq
  emission on lock-acquire success now calls `record_event_with_pc_delay`
  instead of `record_event` -- the event is committed 4 cycles later
  with whatever PC is at issue at that point.

- `src/interpreter/core/interpreter.rs`: both step paths (`step_internal`
  and `step`) call `drain_deferred_pc(current_cycle, current_pc)` after
  resolving any stall, so the drain consistently fires at start-of-bundle
  with the issue PC. The post-stall path matters because if the deferred
  event lands during a subsequent stall (rare in practice but possible
  when a kernel back-to-backs acquires) we don't want to stamp the
  stalled PC.

This same mechanism can be extended to InstrLockReleaseReq, InstrLoad,
InstrStore, InstrCall, and InstrReturn if/when those events show similar
HW/EMU PC offsets -- the deferred-PC plumbing is generic. For now only
InstrLockAcquireReq goes through it, because that's the only Instr event
with confirmed HW data to fit against.

The HW=15/EMU=16 count discrepancy is unchanged by this fix and remains
the HW-pipeline drop pattern documented in the section above, not an
EMU bug.

## Update 2026-05-12: PERF_CNT_2 0 -> 8 events (HW=11)

EMU was emitting 0 PERF_CNT_2 events on `add_one_using_dma` (HW=11).
After the fix, EMU emits 8 events. Two semantic bugs in the perf
counter model:

1. **Level-gating in `tick`**. `PerfCounterBank::tick_idle_cycles()`
   skipped counters whose `start_event == ACTIVE_CORE`, on the
   (incorrect) theory that ACTIVE_CORE is a level signal asserted
   only during Execute state. ACTIVE_CORE is actually an edge-style
   start trigger: `EventMonitor pc0(... XAIE_EVENT_ACTIVE_CORE,
   XAIE_EVENT_DISABLED_CORE, ...)` in mlir-aie's `05_Core_Startup`
   benchmark uses it to measure total core lifetime cycles --
   meaning the counter must keep ticking through stalls. The fix
   collapses `tick_active_cycles`/`tick_idle_cycles` into a single
   `tick()` that increments every Active counter unconditionally;
   the old names remain as `#[deprecated]` shims to ease migration
   of test call sites.

2. **Reset-event stopped the counter**. `handle_event(reset_event)`
   set the counter to `Idle`, which made the canonical self-reset
   pattern (`reset_event = PERF_CNT_N`, `threshold = period`) fire
   exactly once per kernel and then go silent. Per aie-rt's
   `XAie_PerfCounterReset`, reset zeros the counter register but
   does not halt the counter -- the run state machine is start/stop
   only. Fix: reset zeros the value, leaves state unchanged.

PCs are now stamped from `tile.core.pc` (the pipeline-adjusted
trace PC) rather than the `None` sentinel that decoded to PC=0,
matching HW's behavior of capturing the in-flight PC when the
counter fires. EMU PC=0xBC (188) on this kernel vs HW PCs 0xCC
(204) and 0x340 (832) -- different specific PCs because of the
underlying cycle-count divergence, same general pattern (compute
body + lock-stall pipeline PC).

Residual EMU=8/HW=11 gap is the kernel running ~30% fewer total
cycles in EMU than HW, leaving ~3 fewer 1024-cycle windows for the
threshold to fire in. That's the broader cycle-fidelity issue
discussed elsewhere in this finding, not a perf-counter bug.

Trace stays CLEAN on `add_one_using_dma`/`add_one_objFifo`/
`add_one_ctrl_packet` after the fix.

## Update 2026-05-12: PORT_RUNNING/IDLE/STALLED/TLAST level-triggered (#36)

EMU was emitting **zero** PORT_RUNNING/PORT_IDLE/PORT_STALLED/PORT_TLAST
events; HW emits thousands of PORT_RUNNING per active port. On
`add_one_using_dma` chess HW batch 00, PORT_RUNNING_2 and PORT_RUNNING_6
each fire 10848 times across ~6991 kernel cycles.

Root cause: coordinator.rs Phase 3b treated all four events as rising-edge
triggered, with `prev_port_state` per slot for transition detection.
This was based on the (incorrect) reasoning that level-triggered emit
would "flood small trace BDs in milliseconds" -- it doesn't, because
`TraceUnit::notify_event` filters by `event_slots` at trace_unit/mod.rs:587,
so only events the kernel asked to capture reach the trace BD. Level
emission only walks the in-memory event list once per cycle (bounded
8 slots * configured tiles).

Per AM025 and aie-rt's event enumeration, all four are slot-mapped
signals derived from combinational port state -- they assert continuously
while the condition holds, and the trace controller stamps a frame each
cycle. Empirical HW data confirms (10848 events / kernel = many per
active cycle, not transitions).

Fix: read `port.cycle_active` / `cycle_stalled` / `cycle_tlast` directly
each cycle in Phase 3b. `begin_routing_cycle` resets these at the start
of every `step_data_movement`, so reading them gives the live level
signal. PORT_TLAST is a 1-cycle pulse (cycle_tlast cleared each cycle),
so level == edge for it. `prev_port_state` field on `Tile` and its
reset in `state/effects.rs` are no longer needed; removed.

Edge detector pipeline is unaffected: `notify_*_trace_event` still sets
`curr_active=true` on each fire, and `evaluate_edge_detectors` compares
against `prev_active` then resets `curr_active` each cycle. So
level-triggered notify still produces correct rising/falling edges for
downstream edge detectors.

After fix on `add_one_using_dma` chess (full trace.log):

```
PORT_RUNNING_1   8/8 intervals  OK   (HW=1cy each, EMU=8cy each)
PORT_RUNNING_5   4/4 intervals  OK   (HW=1cy each, EMU=8cy each)
PORT_RUNNING_0   6/1 intervals  DIFFER (duration)
PORT_RUNNING_4   7/3 intervals  DIFFER (duration)
```

Two slots' interval counts now match HW exactly; two slots are still
short. The pattern across "OK" slots is a constant `dt_dur=-7` cycles
(EMU's active period is 8 cycles, HW's is 1 cycle, repeated identically
across 8 and 4 intervals respectively). This is a separate stream-switch
fidelity issue -- EMU's `Port::cycle_active` stays true for the whole
duration of a FIFO drain (~8-word burst) where HW asserts PORT_RUNNING
for one cycle per beat. Tracked as #38; the level-triggered emission is
faithfully transcribing whatever the underlying port state model says.

cargo test --lib: 2889/2889 pass. Broad bridge sweep (chess-only):
60 pass / 0 regressions / 36 trace clean / 0 diverged. Two pre-existing
fails unchanged (`vec_mul_event_trace` NPU2-only, `vec_mul_trace_distribute_lateral`
Chess compile segfault).

## Other items

5. **Spurious MEMORY_STALL (CLOSED 2026-05-12)**: S2MM bank record
   fired before the stall check, so every stalled cycle phantom-claimed
   a bank touch. See "Update 2026-05-12" above.

4. **Trace pipeline coalescing investigation** (prerequisite for
   item 2's level-triggered fix). The mystery of "adding
   InstrLockAcquireReq emission inflates LockStall count 200x"
   needs its own root-cause before the LockStall fidelity work
   above can land cleanly.

The DMA-rate calibration that was originally next is fully
shelved -- the DMA rate isn't the problem.

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
