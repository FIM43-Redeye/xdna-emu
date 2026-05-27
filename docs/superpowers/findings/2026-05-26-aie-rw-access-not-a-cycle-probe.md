---
name: 'AIE_RW_ACCESS is not a usable cycle-counter probe on Phoenix'
description: 'Empirical Phase 1 calibration shows read_aie_reg on Timer_Low advances by a fixed ~12,000 ticks per call regardless of wall-clock duration between calls or whether a kernel ran in between. Multiple write paths (write_aie_reg, runtime_sequence aiex.npu.write32 from the trace pipeline) all fail to change what the read returns. Conclusion: AIE_RW_ACCESS is useful for state inspection (the wedge survey use case) but cannot serve as the dispatch_overhead cross-validation tool the earlier finding speculated about. Cycle-accuracy work must rely on the trace unit.'
type: project
---

# AIE_RW_ACCESS is not a usable cycle-counter probe on Phoenix

## TL;DR

Phase 1 calibration of `read_aie_reg` on Phoenix produced a clear
negative result. Timer_Low values returned by AIE_RW_ACCESS advance
by a fixed ~12,000 ticks per call irrespective of wait time or
kernel activity, multiple write paths to Timer_Control silently
fail to alter the readback, and the per-call ~12,000 tick advance
appears to be a FW-handler-internal artifact rather than a cycle
counter we can use.

The earlier dispatch-overhead finding speculated about using
AIE_RW_ACCESS to cross-validate the `dispatch_overhead = 2500 cyc`
constant via on-NPU cycle-counter readback. That speculation does
not survive empirical contact with the FW path. Cycle-accuracy
work must remain trace-unit-based.

AIE_RW_ACCESS remains useful as a **state inspection probe** for
what it already proved in the wedge survey: reading tile registers
at a host-chosen moment to find out what HW state looks like (e.g.,
Task_Queue depth at the K=16 wedge, BD pointer values, lock counts).

## Phase 1 results

All experiments below were on Phoenix NPU1 FW 1.5.5.391 / protocol
5.8 via the xdna-driver `emu-shim-base` branch (with the memtile-
block safety patch from
[`2026-05-26-aie-rw-access-memtile-wedge-mechanism`](2026-05-26-aie-rw-access-memtile-wedge-mechanism.md)).
Probe: `tools/rw-access-probe`. Default tile: (0, 2) compute core,
Timer_Low at 0x340F8.

### T1: roundtrip distribution (10k tight-loop reads)

| Percentile | Roundtrip |
|-----------:|----------:|
| min        | 60.8 us   |
| p50        | 72.5 us   |
| p90        | 90.3 us   |
| p99        | 192 us    |
| p99.9      | 648 us    |
| max        | 3,115 us  |

p50 ~73us is tighter than the wedge-survey eyeball estimate of
~200us. Effective tight-loop rate ~12,150 reads/s.

### T2: Timer_Low advance is independent of wall-clock

| Sleep between reads | Wall-clock duration | Per-read Δ (mean ± std) |
|-------:|------:|------:|
|   1 ms | 22 ms | 11,858 ± 16 |
|  10 ms | 194 ms | 11,876 ± 14 |
| 100 ms | 1.90 s | 11,878 ± 16 |
|1000 ms | 19.00 s | 11,872 ± 15 |

Wall-clock varied by 1000x; Timer_Low advance varied by <0.2%.
This rules out "Timer_Low free-running at any single fixed clock
that AIE_RW_ACCESS would let us observe": if it were free-running,
the 1000-ms-sleep case should have produced ~1.4M ticks (at 144
MHz, the rate that would explain 11,870 ticks per ~82us tight-loop
call). It does not.

### T3: kernel dispatch doesn't change the picture either

Built a traced variant of `add_one_using_dma` via the bridge-test
trace pipeline. Confirmed in the lowered MLIR
(`chess/aie_arch.mlir.prj/input_with_addresses.mlir`) that the
`aiex.runtime_sequence` block contains explicit `aiex.npu.write32`
to Timer_Control on the core, memory, memtile, and shim:

```
aiex.npu.write32 {address = 212992, column = 0, row = 2, value = 31232}   # core   Timer_Control = 0x7A00
aiex.npu.write32 {address = 606208, column = 0, row = 1, value = 40192}   # memtile
aiex.npu.write32 {address = 81920,  column = 0, row = 2, value = 31232}   # memory
aiex.npu.write32 {address = 212992, column = 0, row = 0, value = 32512}   # shim
```

(Value 0x7A00 sets `Reset_Event = 122` and `Reset = 0` — i.e.,
the timer will reset when event 122 fires, else free-run.)

Ran `rw-dispatch-probe` against this build with 5 pre-samples + 5
post-samples bracketing a verified-PASS kernel run (output check
PASS, 0/64 errors). Kernel wall-clock 1099us. Results:

| Delta | Value | Should be (if Timer_Low ticked during run) |
|-------|------:|-------------------------------------------|
| pre[1] - pre[0] | 11,924 | ~12k (per-call idle baseline) |
| post[1] - post[0] | 12,053 | ~12k (per-call idle baseline) |
| **post[0] - pre[4]** | **12,109** | **~1,099,000 (if 1 GHz) or ~158,000 (if 144 MHz)** |

The bracket delta across a 1099us kernel run is indistinguishable
from a single idle AIE_RW_ACCESS call. Timer_Low at the compute
tile that ran the kernel did not advance during execution as
observed via AIE_RW_ACCESS.

### T3 follow-up: direct verification fails

Read Timer_Control directly via AIE_RW_ACCESS after the kernel
run: returns **0x00000000**. The `runtime_sequence`'s
`aiex.npu.write32` did not produce a visible change in the
register we read.

Attempted writes via `write_aie_reg` (XRT API path, confirmed to
go through the same MSG_OP_AIE_RW_ACCESS handler with
`aie2_access_type = REG_WRITE`):

- `write_aie_reg(... Timer_Control, 0x7A00)`: succeeds at the
  API level (no exception, ~98us roundtrip), but readback of
  Timer_Control still returns 0.
- `write_aie_reg(... Timer_Control, 0x80000000)` (set Reset
  bit 31, which per the AM025 spec should immediately zero
  Timer_Low): no effect — Timer_Low continues to accumulate
  from where it was, not from 0.

The write call returns success but the write has no visible effect
on what subsequent reads return. The FW handler decompile from
the earlier capability survey
([`2026-05-06-npu1-msg-op-capability-survey`](../archive/findings/2026-05-06-npu1-msg-op-capability-survey.md))
showed FUN_08ad98c4 dispatching on the access-type enum with both
REG_READ and REG_WRITE as reachable cases, but the empirical
behavior of REG_WRITE on Phoenix appears to be "ack the call
without writing the hardware register" — or write to a different
register space than REG_READ reads from.

### What the ~12,000-per-call advance actually is

Unresolved. Three candidate explanations:

1. **FW-internal artifact**: the FW handler does roughly 12,000
   tile-clock-cycles' worth of work per AIE_RW_ACCESS call
   servicing the request (stream switch routing, AXI access,
   response framing). The returned Timer_Low value reflects this
   internal activity rather than the kernel's separate execution.
2. **Read-via-cached-mirror**: AIE_RW_ACCESS reads a cached or
   aliased shadow of the tile registers maintained by FW for
   debug purposes, which advances on FW activity rather than tile
   activity.
3. **Real Timer_Low gated to AIE_RW_ACCESS calls**: the tile clock
   is power-gated except during AIE_RW_ACCESS handling, so the
   timer only ticks during the brief window when FW has the tile
   awake to service a register access. Kernel execution wakes the
   tile but the timer's source clock is independently gated.

The tight per-call delta std (~0.1%) suggests an FW-side
deterministic process more than a hardware variable. Either way,
the ~12,000-per-call advance is not a wall-clock-correlated cycle
counter.

## Implications for the broader cycle-accuracy mission

The dispatch-overhead finding's "Follow-ups" section speculated:

> The provisional_npu1() doc-comment flags that
> Performance_Counter0 readback via xrt::hw_context::read_aie_reg
> is functional on Phoenix as of 2026-05-05, which would provide
> trace-independent ground-truth cycle counts.

This proves out as wishful thinking under empirical inspection.
The XRT API is functional, the FW path is functional in the
narrow sense of "the call completes," but the values returned are
not what the dispatch-overhead question needs. Cross-validating
the 2500-cyc dispatch_overhead constant requires either:

1. **Trace-unit measurements**, which already work and are how
   the constant was originally calibrated. Cross-validation via
   the same mechanism doesn't add independent signal.
2. **In-kernel instrumentation**: have the kernel write Timer_Low
   to memory at known points. Requires kernel modification and
   either Chess or Peano support for the Timer_Low intrinsic.
   The trace unit already does this internally via event packets,
   so we'd be re-implementing trace from scratch.
3. **Statistical noise reduction on existing trace data**: many
   runs + outlier detection + per-tile event decorrelation. Likely
   the highest-leverage path that doesn't require new HW probes.

## What AIE_RW_ACCESS is still good for

The wedge survey (2026-05-26) demonstrated AIE_RW_ACCESS's actual
usable role: **state inspection at a host-chosen moment**.

- Reading Task_Queue depth, BD pointer, and channel status on a
  shim DMA to characterize HW state at a wedge or steady-state.
- Reading lock counters between dispatches to verify
  synchronization state.
- Reading PC / register snapshots from a halted core.
- Verifying tile reachability post-driver-load (smoke test for the
  rw_access feature itself).

For these uses, the per-call ~12,000-tick advance on Timer_Low is
irrelevant — we're reading other registers whose values are
genuinely the hardware state, not Timer_Low's pathological case.

## Open questions (not load-bearing, kept for future curiosity)

- Why does `write_aie_reg` succeed at the API level but produce no
  visible effect? Is there a write-protect, or does the FW handler
  route writes to a different address space than reads?
- What's the exact mechanism producing the 11,870-tick (± 15) per-
  call advance with such tight determinism?
- Does any tile-local register show different behavior than
  Timer_Low under AIE_RW_ACCESS? Trying a known-changeable register
  (e.g., a lock counter that we can deterministically nudge from a
  kernel) might disambiguate the "what view does AIE_RW_ACCESS read
  from" question.

These are interesting puzzles but not blocking. The cycle-accuracy
work has its tool already (the trace unit); we don't need to
unravel AIE_RW_ACCESS's internals to make progress.

## See also

- [`2026-05-26-aie-rw-access-memtile-wedge-mechanism`](2026-05-26-aie-rw-access-memtile-wedge-mechanism.md) — the wedge survey that established AIE_RW_ACCESS's real reach
- [`2026-05-25-npu-controller-dispatch-overhead`](2026-05-25-npu-controller-dispatch-overhead.md) — the 2500-cyc dispatch overhead finding that motivated this calibration
- `docs/superpowers/plans/2026-05-26-aie-rw-access-characterization.md` — the plan that produced this work; Phase 2 will be amended to pivot to trace-unit-based cycle-accuracy work
- `tools/rw-access-probe/rw-access-probe.cpp` and `rw-dispatch-probe.cpp` — the probes used
- `tools/analyze-rw-latency.py` — Python analysis for the CSV outputs
