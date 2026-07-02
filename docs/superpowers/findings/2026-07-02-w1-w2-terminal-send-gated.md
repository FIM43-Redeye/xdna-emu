# W1/W2: the SP-4a gate is the terminal memtile->shim send, not dataflow engagement

**Date:** 2026-07-02  **Issue:** #140 (SP-4a cold-start fill-state; W1 mechanism)
**Status:** MECHANISM LOCALIZED by trace-free in-kernel-timer measurement. The
pre-drain gate on HW is SPECIFICALLY the terminal `memtile->shim` (of_out) send;
the producer and consumer both flow and fill upstream before the drain. Corrects
the earlier A2 read (which the trace confounded).
**Tool:** the in-kernel-timer oracle (`2026-07-02-in-kernel-timer-oracle-works.md`).
**Artifacts:** `build/experiments/pathA-cntr-spike/{w1,w2-consa}/` (gitignored):
instrumented lean kernels + `w1_host.cpp`/`w2_consa_host.cpp` XRT drivers.

## Method

Instrument a core in the lean SP-4a kernel (ProdCore(0,2)->MemTile(0,1)->
ConsA(0,3)->shim of_out drain) with `read_cntr()` timestamps, drain them to DDR,
and reset the tile's timer (Timer_Control bit31) RIGHT BEFORE the of_out drain
dispatch. Then each timestamp reads LARGE (pre-drain, free-running) or SMALL
(post-drain, post-reset) -- so the pre/post-drain split of each core's dataflow
is read directly, trace-free. of_src (W1, producer) and of_d/of_j (W2, ConsA)
lock sequences were diffed BIT-IDENTICAL to the un-instrumented lean kernel
(additive instrumentation only).

## Results (HW)

**W1 -- Producer (of_src release), reset at drain:**
- loop_entry + 12 releases (iter0..11) all PRE-drain (large, ~368330..369823).
- 839cy backpressure STALL at iter7 (~55cy/iter before, ~65cy after).
- Reset/drain crossed after iter11; iters 12..31 post-drain (small).

**W2 -- ConsA (of_j release), reset at drain:**
- loop_entry + 5 releases (iter0..4) all PRE-drain (large, ~372335..372747).
- NO backpressure stall (smooth ~80cy/iter throughout).
- Reset/drain crossed after iter4; iters 5..31 post-drain (small).

**Sweep (prior finding) -- of_out at drain-start:** HW shim S2MM starves at
t+13 (of_out ~empty), EMU at t+1683 (of_out backlog/full).

## The convergence

- Producer FLOWS pre-drain (12 releases, backpressure).
- ConsA FLOWS pre-drain (5 of_j releases, smooth) -- data reaches ConsA and it
  produces of_j BEFORE the drain.
- of_out is EMPTY at drain-start.

So on HW the pipeline fills UPSTREAM (of_src, of_d, of_j all take data pre-drain)
but the terminal `memtile->shim` (of_out) send stays EMPTY until the shim drain
dispatches. The gate is localized to the LAST stage by elimination: everything up
through ConsA's of_j production runs pre-drain; only the shim-facing send is held.

The producer-stalls-but-ConsA-doesn't asymmetry fits a sink-gated pipeline:
backpressure fills from the gated sink BACKWARD, so the furthest-upstream producer
(faster, earlier) piles up and stalls at iter7, while ConsA -- one hop from the
gate -- was still within of_j+of_out capacity (~4) at 5 releases when the drain
relieved it.

## Implication for the EMU defect

The sweep's EMU of_out is FULL at the drain (starve t+1683) -- the EMU PRE-FILLS
the terminal memtile->shim send during the pre-drain window; HW leaves it EMPTY
until the shim drain pulls. So the defect is NOT "cores start too early" (producer
+ ConsA both faithfully run from CDO, confirmed) and NOT the control-path
constants -- it is the terminal send's ENGAGEMENT: the EMU advances data into the
of_out send path with no active shim consumer; HW does not.

**This also corrects the earlier A2 read.** The trace-based +43 "producer stalls
after the drain" signal was a trace-unit confound; trace-free, the producer
plainly produces 12 buffers BEFORE the drain. The dataflow-engagement framing was
too broad -- it is specifically the terminal send.

## Certainty boundary

SOLID (measured): producer flows pre-drain (12); ConsA flows pre-drain (5,
produces of_j); of_out empty at drain-start. INFERRED (mechanism): upstream fills
while the terminal send stays empty; the EMU pre-fills the terminal send. The
"of_out empty while of_j full" buffer accounting rests on objectfifo-link
semantics that a toolchain (aie-rt) read must confirm. The memtile relays are
pure DMA (no core to instrument), so this hop is derivation-plus-elimination, not
direct measurement.

## Next

Derive from aie-rt: WHEN does a memtile MM2S paired with an un-started shim S2MM
drain actually engage its send? + read the EMU's memtile-of_out MM2S engagement
code. The measured HW behavior (terminal send empty until the drain) is the oracle
the fix must match.
