# Task 319 cause (b): the 60-cycle drift comes from NPU instruction cycle accounting

## Summary

After fixing cause (a) (pipeline-PC during ACQ stall, commit `c0abfd0`),
the remaining mode-2 anchor divergence on `add_one_using_dma.chess` is
that EMU's BROADCAST_15 fires while the kernel is in the *first*
acquire (1st JL #816 stub still stalling), while HW's BROADCAST_15
fires after the kernel has already cleared the 1st acquire and is
stalling in the *second*. This is cause (b) from the original
investigation: a ~60-cycle drift in when BROADCAST_15 reaches the
trace unit relative to kernel progression.

## Where the latency comes from

The cause is **EMU's NPU executor processes one control packet per
simulation cycle, while real HW takes more cycles per packet.**

Concrete data from `add_one_using_dma.chess` EMU run:

```
Executing 25 NPU instructions
Tile(0,0) Event_Generate: event_id=127 (offset=0x34008) cycle=51
   -> BROADCAST_15
Propagating BROADCAST_15 from tile (0,0) to column 0 at cycle 51
...
Tile(0,0) Event_Generate: event_id=126 (offset=0x34008) cycle=790
   -> BROADCAST_14
halt_reason=completed cycles=791
```

The 25 NPU instructions for this test compile down to:
- a handful of `trace.start_config` control packets that program
  the compute tile's trace_unit registers and ultimately fire
  Event_Generate=127 -> BROADCAST_15
- two `dma_memcpy_nd` sequences that program shim-side BD configs
  and trigger DMA channels
- a `dma_wait` for the output DMA to drain

In `src/testing/xclbin_suite.rs:1186`, the engine main loop calls
`executor.try_advance()` once per cycle, and the executor in
`src/npu/executor.rs:366` issues exactly one instruction per call
(returning `Blocked` only when a DMA queue is full or a sync poll is
pending). So in EMU, the trace-setup control packets retire at ~1
cycle per packet -- meaning BROADCAST_15 fires after ~10-12 packets
of trace setup, observed empirically at cycle 51.

The bridge-report PC-anchored comparison shows:

```
Tile (0,2) Core (anchor: HW cy 204, EMU cy 144)
```

The "anchor cycle" is the cycle of the first matched edge event after
trace start. Both EMU and HW have ~93 cycles between BROADCAST_15
arrival and the first anchored edge event (a stable feature of the
kernel's progression after trace start). So HW's BROADCAST_15 fires
at roughly cycle 204 - 93 = 111. EMU fires it at cycle 51. **HW takes
~60 more cycles than EMU to reach Event_Generate=127** -- about
~10-12 cycles per control packet vs EMU's 1 cycle.

This is the source of the latency.

## Why it matters for trace fidelity

The 60-cycle gap means:

- In EMU, by cycle 51 the kernel has had time to traverse `_main_init`
  prologue (~30 instructions), `JL #224` to `core_0_2`, the
  `core_0_2` prologue (~25 instructions), and the `JL #816` to enter
  the 1st acquire stub. It then stalls at PC=816 (1st ACQ) waiting
  for input DMA data.
- In HW, by the equivalent cycle 111, the kernel has done all of
  the above *plus* gone further: the 1st acquire's data has arrived
  (because the host's `dma_memcpy_nd` packets were processed in
  parallel-ish with the trace setup, so input DMA started delivering
  data earlier in real time), the 1st ACQ has been granted, the
  `JL #816` at PC 336 fired, and the kernel is now in the *2nd*
  acquire stub stalling at PC=816 again.

Both EMU and HW report anchor_pc=832 (with cause-(a)'s pipeline-PC
fix), but they're "in different acquires." The result is EMU's mode-2
trace has one extra `New_PC=336` in iteration 1 that HW doesn't have
(because HW's anchor was AFTER the 1st RET-to-336 happened).

## Why "1 cycle per NPU instruction" is wrong

Real HW's path from "host CPU writes a control packet to the IPU
command queue" -> "control packet decoded by CMP" -> "CMP issues an
AXI write to the target tile register" has multiple latencies:
PCIe doorbell, command queue read, CMP decode, AXI fabric traversal,
target tile register write. Per AMD's NPU architecture docs (AM020
Ch.2 stall handling, Ch.6 boot/config), each control packet writes
through the same memory-mapped AXI4 path that the host uses for
register access. That path isn't single-cycle.

The exact per-packet cycle cost isn't documented in AM020 -- it
varies with packet type and tile distance. But it's clearly more
than 1.

EMU's `try_advance()` running at 1 cycle per packet is a simplification
that produces correct architectural behavior (the right registers get
written, the right events fire) but compresses the wall-clock-derived
cycle accounting.

## Possible fixes

1. **Empirical per-packet cycle cost.** Add a constant cycle charge
   per control packet (e.g., `try_advance` charges N cycles before
   issuing the next instruction). Calibrate N against HW timing
   observations on a few representative tests. Simplest model;
   probably good enough for trace bracket alignment.

2. **Per-packet-type cycle cost.** Different control packets have
   different costs in HW. Register writes are cheap, DMA descriptor
   programming (multi-word writes) takes longer, sync ops longer
   still. A per-type table would be more accurate but harder to
   calibrate.

3. **Model the CMP pipeline.** Most accurate. Add a model of the
   IPU command processor's fetch-decode-issue stages with the
   actual AXI/PCIe latencies. Significant work; only worth doing
   if cycle-accurate IPU modeling is a goal beyond mode-2 trace.

4. **Side-channel: charge cycles only for the packets that affect
   trace timing.** `Event_Generate` writes (which fire broadcasts)
   are the main ones that matter for trace bracketing. Adding a
   per-Event_Generate cycle delay gets us closer to HW without
   touching unrelated paths.

## Recommendation

**Fix (1) or (4) -- empirical constant charge.** It's the smallest
change that addresses the observed symptom. The exact value of N
should be calibrated against HW timing for at least 2-3 distinct
mode-2 baselines (sweep with different kernel structures) so we
don't overfit to `add_one_using_dma`. Document the chosen value
and where it came from.

Fix (3) would be ideal but is genuinely a multi-week project that
would need its own roadmap entry.

## Concrete next step

Implement an experiment:
- Add a `NPU_INSTR_CYCLE_COST` config knob (default = 1, current
  behavior).
- Re-run `add_one_using_dma.chess` with N = 5, 10, 15, 20.
- Find the N that lands EMU's BROADCAST_15 at cycle ~111 (matching
  inferred HW timing), and see if iter-1 `New_PC=336` disappears.
- If the trace matches at some N, repeat on 2-3 other mode-2
  baselines (need #305 unblocked first to have multi-test data,
  OR run the experiment manually).
- If a single N works across tests, ship it. If different tests
  need different N, switch to per-packet-type modeling.

## Status

Cause identified. Logged for #319. Fix is straightforward to
prototype but needs HW timing data (or empirical sweep) to
calibrate the cycle cost. Same root cause likely fixes #321
(BROADCAST_14 firing too early relative to kernel halt).
