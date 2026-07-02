# SP-4a mechanism, derived: the terminal shim S2MM is unconfigured pre-drain; the EMU accept-path lacks a "started" term

**Date:** 2026-07-02  **Issue:** #140 (SP-4a cold-start fill-state)
**Status:** MECHANISM DERIVED from the toolchain (aie-rt/mlir-aie) AND the EMU
code -- two independent derivations converged on the same "stream-gated,
unconfigured destination" mechanism. Fix PROPOSED (one predicate), under
adversarial review before implementation.
**Builds on:** `2026-07-02-w1-w2-terminal-send-gated.md` (the HW measurement that
localized the gate to the terminal send).

## The toolchain truth (what HW does + why)

`object_fifo_link([of_j], of_out, [0], [])` is a zero-copy 1:1 link: one shared
2-slot buffer + one PROD/CONS lock pair (prod init=2, cons init=0), used by two
DMA channels (S2MM in, MM2S out). Buffer-owner selection at
`AIEObjectFifoStatefulTransform.cpp:606-654`; single lock pair per link at
`:422-451`.

**The of_j split (resolves the "of_j full / of_out empty" tension):** because
ConsA<->MemTile are not shared-memory tiles, `of_j` is split
(`AIEObjectFifoStatefulTransform.cpp:1876-1958`) into a core-side `j` (ConsA-local
2-slot buffer, what the core's of_j.acquire/release touches) and a memtile-side
`j_cons`/`out` shared buffer (2 slots). These are SEPARATE storage one stream-hop
apart. So the measured "of_j full" (ConsA-side, ~5) and "of_out empty" (the
memtile shared buffer feeding the shim) are different storage. ConsA fills local
`j` (2) + shared `j_cons`/`out` (2) = 4 complete cycles + hangs on the 5th ~= the
measured ~5; the producer, one more independent depth-2 domain deeper, accumulates
~12. All capped by the same shim-dispatch event.

**Why the terminal send stays empty:** the shim S2MM (of_out drain) gets NO static
CDO configuration -- `createShimDMA` (`AIEObjectFifoStatefulTransform.cpp:1011-1018`)
early-returns when there are no registered external buffers, and of_out's DDR
buffer is supplied DYNAMICALLY by the runtime `npu_dma_memcpy_nd`
(`of_q0_lean.py:74-76`). So there is no ShimDMAOp, no BD, no
`pushToBdQueueAndEnable`, and no enable for the shim S2MM until the runtime
sequence executes at `xrt::run()` time -- much later than the static CDO apply
that enables every on-chip channel (`AIERT.cpp:494-510`, `:743-789`). The memtile
of_out MM2S IS CDO-enabled and attempts to send, but stalls mid-transfer holding
its lock, because its downstream (the disabled/unconfigured shim S2MM) offers no
stream-consumer readiness (no TREADY). **STREAM-gated, mechanism (b).** So the
shared buffer holds ~2 slots and the shim, once dispatched, drains them and
starves at t+13.

(One inference not directly citable from aie-rt, which only pokes config
registers: "a disabled destination asserts no readiness and a stalled BD holds
its lock indefinitely" is standard DMA-engine semantics, matching what the EMU
already models for BD completion. The fix's HW gate -- offset -52->+2 -- is the
confirmation.)

## The EMU defect (converged from the EMU side)

The memtile of_out MM2S is CDO-started and moves data whenever it holds the
of_out consumer lock and has local `stream_out` room. Critically it releases the
of_out consumer lock when the buffer drains into the LOCAL `stream_out` (not on
shim consumption), so each buffer frees on local drain -- the pre-fill enabler.
The terminal accept guard `can_accept_stream_in_for_routing`
(`src/device/dma/engine/stream_io.rs:371`, used at `src/device/array/routing.rs:967`)
checks only `bd_switch_accept_block`, `accept_awaiting_drain`, and FIFO capacity
(`current + drained < cap`) -- it has NO "is this S2MM channel started?" term. So
the Idle, unconfigured shim S2MM still accepts into its 2-deep FIFO from cy0, and
the fabric slack (16-deep switch FIFOs via `input_fifo_capacity`, inter-tile,
memtile master) absorbs ~5 buffers -> starve t+1683.

## Proposed fix (under review)

Add a "channel started" term to the accept guard: refuse when the target S2MM is
unstarted -- `task_queue_size(ch) > 0 || channel_fsm(ch).is_active()`
(`task_queue_ops.rs:150`, `channel.rs:198`), both already available. HW-faithful
(an unconfigured/disabled S2MM asserts no TREADY); safe by construction because
interior memtile/compute S2MMs are CDO-started (pass from cy0), only the
runtime-started shim S2MM is held.

**Open question under adversarial review:** the EMU releases the of_out lock on
LOCAL drain, so the accept-gate alone may bound the pre-fill to the fabric-FIFO
depth (16-deep) rather than HW's ~2 slots -- possibly closing the -52 oracle,
possibly needing the lock-release-on-local-drain semantics changed too. Fix shape
(one predicate vs two changes) TBD by the review + the HW gate.

## Gate (unchanged)

lean oracle offset -52 -> +2; write32-sweep EMU of_out goes empty-at-drain
(starve 1683 -> small); `cargo test --lib` no regressions; bridge-corpus
spot-check (the accept path is the send-cadence fixes' territory --
f4009413/788e3d70/b5ec0404/eb683bc4 -- so watch for recv/send-cadence regressions).
