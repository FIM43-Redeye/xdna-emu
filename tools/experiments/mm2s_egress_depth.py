#!/usr/bin/env python3
"""mm2s_egress_depth -- pin DMA_MM2S_EGRESS_FIFO_DEPTH (currently 12), Experiment B.

A COMPUTE-tile MM2S (where the constant lives) whose BD/lock structure fills
the egress FIFO, then stalls the memory-side FETCH on a lock while the
stream side keeps draining. This is the exact mirror of the S2MM ingress
finding (docs/superpowers/findings/2026-07-14-dma-memory-pressure-event-
semantics.md), which pinned ingress depth at ~15-16 beats from "backpressure
asserts +15" -- here we read DMA_MM2S_0_STALLED_LOCK onset to
DMA_MM2S_0_MEMORY_STARVATION onset instead.

The crux (see the plan's Experiment B section): STARVATION only reads the
true ceiling if the FIFO is genuinely full at the moment the fetch stalls. A
naive single-tile design cannot guarantee that (the stream can drain the
FIFO just as fast as fetch fills it, so a lock stall that happens only
*between* transfers finds an already-empty FIFO). Two tile shapes exist here:

SINGLE-TILE (bankdisc/producer_probe skeleton: core fills dma_buf, releases
lk_full; MM2S self-loops on lk_full/lk_empty; shim S2MM drains to host):

  fill_stall    -- naive baseline. No engineered mechanism at all; exists to
                   show the baseline onset delay is small (motivates escalation).
  never_stall   -- control. lk_full is a RATCHET pre-loaded with REPS credit
                   (memtile_bankwidth's A1 idiom), so MM2S never waits between
                   BDs; STARVATION is expected to stay ~0 (nothing to drain
                   into an idle gap that never opens).
  cold          -- control. REPS=1: the only STALLED_LOCK before this design's
                   first (and only) BD is a genuine cold start (FIFO has never
                   held anything); a second, permanent STALLED_LOCK follows
                   FINISHED_BD once the core has no more reps to give (the
                   same "ratchet-of-1" a single-BD self-loop always produces).
  fetch_starve  -- escalation lever 2. Reuses producer_probe's collide DENSE
                   MARCH-STORE hammer verbatim, on the SAME logical bank as
                   dma_buf, so the fetch loses bank arbitration on (close to)
                   every cycle it competes with the core.

TWO-TILE (escalation lever 1, "decouple fill from drain", and lever 3, "sweep
the dwell", turn out to be the SAME mechanism at different K):

  stream_backpressure, dwell_sweep_{K} -- source tile (0,2) fetches src_buf
  into its own egress FIFO under a SINGLE release (lk_src_full, never
  released again -- so it goes permanently STALLED_LOCK the moment its one
  BD finishes, exactly `cold`'s mechanism). Sink tile (0,3) withholds its OWN
  accept lock (lk_sink_go) for K busy-cycles -- producer_probe's sticky_body(K)
  K-dwell machinery, reused verbatim -- before releasing it. While withheld,
  the physical stream cannot drain, so flow-control backpressure should climb
  all the way back into the source's egress FIFO. Once lk_sink_go releases,
  the sink drains while the source's fetch is ALREADY permanently stalled, so
  onset(source STALLED_LOCK) -> onset(source MEMORY_STARVATION) reads exactly
  the residual occupancy left when the hold ended -- the ceiling, once K is
  generous enough to have saturated it. `stream_backpressure` is one
  generously-sized K (the "clean" escalation-1 variant); `dwell_sweep_{K}`
  sweeps K broadly so the measure script can read the plateau (the ceiling)
  directly, per the plan's escalation-3 lever.

  UNVERIFIED, flagged per the task brief rather than guessed past: (a) whether
  a direct compute-to-compute DMA flow, row 2 -> row 3 (no intervening
  memtile hop), physically routes on Phoenix silicon -- aie-opt's structural
  verifier does not check switchbox routability, only op legality; (b)
  whether the sink's OWN ingress FIFO absorbs part of the hold before
  backpressure reaches the source, which would mean K needs to be larger than
  a single-FIFO model predicts to saturate the source side. Both are hardware
  questions for the interactive capture task (Task 6), not decidable here.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from producer_probe import MARCH_BODY, MARCH_ELEMS, MARCH_N, sticky_body  # noqa: E402

REPS = 16              # fill_stall / never_stall / fetch_starve repeat count
OBJ = 256               # i32 words per MM2S transfer
STACK_SIZE = 0x400
SRC_ADDR = 0x2400       # source buffer address (logical bank 0) -- matches
                        # bankdisc.py / producer_probe.py's own dma_buf choice
MARCH_ADDR = 0x0400     # fetch_starve hammer buffer -- SAME logical bank as
                        # SRC_ADDR, matching producer_probe's "collide" choice

DWELL_SWEEP_KS = (2, 4, 8, 16, 32, 64, 128)
STREAM_BACKPRESSURE_K = 64   # generous fixed dwell for the escalation-1 variant

SINGLE_TILE_VARIANTS = ("fill_stall", "never_stall", "cold", "fetch_starve")
TWO_TILE_K = {"stream_backpressure": STREAM_BACKPRESSURE_K}
TWO_TILE_K.update({f"dwell_sweep_{k}": k for k in DWELL_SWEEP_KS})

VARIANTS = tuple(sorted(set(SINGLE_TILE_VARIANTS) | set(TWO_TILE_K)))

assert (MARCH_ADDR >> 14) & 3 == (SRC_ADDR >> 14) & 3, \
    "fetch_starve's hammer must collide with the source buffer's logical bank"


def _emit_single_tile(variant: str) -> str:
    reps = 1 if variant == "cold" else REPS
    total = reps * OBJ

    if variant == "never_stall":
        buffers = (
            f'    %dma_buf = aie.buffer(%core_0_2) '
            f'{{sym_name = "dma_buf", address = {SRC_ADDR} : i32}} : memref<{OBJ}xi32>'
        )
        locks = (
            f'    %lk_full = aie.lock(%core_0_2, 0) '
            f'{{init = {REPS} : i32, sym_name = "lk_full"}}'
        )
        core_body = f"""
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %cOBJ = arith.constant {OBJ} : index
      scf.for %i = %c0 to %cOBJ step %c1 {{
        %iv = arith.index_cast %i : index to i32
        memref.store %iv, %dma_buf[%i] : memref<{OBJ}xi32>
      }}
      // Single fill; REPS BDs replay it (data is stale after the first BD,
      // but this is a timing-only probe -- same convention as
      // memtile_bankwidth.py's A1 ratchet ("data unchecked").
      aie.end"""
        mem_body = f"""
    %mem = aie.mem(%core_0_2) {{
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lk_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%dma_buf : memref<{OBJ}xi32>, 0, {OBJ}) {{bd_id = 0 : i32}}
      aie.next_bd ^bd0
    ^end:
      aie.end
    }}"""
    else:
        march_decl = ""
        march_const = ""
        hammer = ""
        if variant == "fetch_starve":
            march_decl = (
                f'    %march_buf = aie.buffer(%core_0_2) '
                f'{{sym_name = "march_buf", address = {MARCH_ADDR} : i32}} '
                f': memref<{MARCH_ELEMS}xi32>\n'
            )
            march_const = f'\n      %cMARCH = arith.constant {MARCH_N} : index'
            hammer = MARCH_BODY
        buffers = (
            f'{march_decl}'
            f'    %dma_buf = aie.buffer(%core_0_2) '
            f'{{sym_name = "dma_buf", address = {SRC_ADDR} : i32}} : memref<{OBJ}xi32>'
        )
        locks = (
            f'    %lk_empty = aie.lock(%core_0_2, 0) {{init = 1 : i32, sym_name = "lk_empty"}}\n'
            f'    %lk_full  = aie.lock(%core_0_2, 1) {{init = 0 : i32, sym_name = "lk_full"}}'
        )
        core_body = f"""
      %c0     = arith.constant 0 : index
      %c1     = arith.constant 1 : index
      %cREPS  = arith.constant {reps} : index
      %cOBJ   = arith.constant {OBJ} : index{march_const}

      scf.for %r = %c0 to %cREPS step %c1 {{
        aie.use_lock(%lk_empty, AcquireGreaterEqual, 1)
        scf.for %i = %c0 to %cOBJ step %c1 {{
          %iv = arith.index_cast %i : index to i32
          memref.store %iv, %dma_buf[%i] : memref<{OBJ}xi32>
        }}
        // Release: the MM2S may now drain dma_buf. fetch_starve's hammer
        // (if any) runs CONCURRENTLY with that drain.
        aie.use_lock(%lk_full, Release, 1){hammer}
      }}
      aie.end"""
        mem_body = f"""
    %mem = aie.mem(%core_0_2) {{
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lk_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%dma_buf : memref<{OBJ}xi32>, 0, {OBJ}) {{bd_id = 0 : i32}}
      aie.use_lock(%lk_empty, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }}"""

    return f"""//===- mm2s_egress_depth {variant} ------------------------------*- MLIR -*-===//
// Single-tile MM2S egress-FIFO-depth probe. dma_buf @ {SRC_ADDR:#06x}
// (logical bank {(SRC_ADDR >> 14) & 3}); reps = {reps}.
//===----------------------------------------------------------------------===//
module {{
  aie.device(npu1_2col) {{
    %shim_0_0 = aie.tile(0, 0) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}}
    %core_0_2 = aie.tile(0, 2)

{buffers}
{locks}

    aie.flow(%core_0_2, DMA : 0, %shim_0_0, DMA : 0)
    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)

    %core = aie.core(%core_0_2) {{{core_body}
    }} {{stack_size = {STACK_SIZE} : i32}}
{mem_body}

    aie.runtime_sequence(%comp_buf: memref<{total}xi32>) {{
      aiex.npu.dma_memcpy_nd(%comp_buf[0, 0, 0, 0][1, 1, 1, {total}][0, 0, 0, 1]) {{id = 0 : i64, metadata = @drain}} : memref<{total}xi32>
      aiex.npu.dma_wait {{symbol = @drain}}
    }}
  }}
}}
"""


def _emit_two_tile(variant: str, K: int) -> str:
    total = OBJ
    dwell = sticky_body(K)

    return f"""//===- mm2s_egress_depth {variant} ------------------------------*- MLIR -*-===//
// Two-tile hold-then-release probe (escalation lever 1 / 3). Source (0,2)
// fetches src_buf under a single release -- permanently STALLED_LOCK the
// moment its one BD finishes (the `cold` ratchet-of-1 idiom). Sink (0,3)
// withholds its own accept lock for K = {K} busy-cycles (producer_probe's
// sticky_body(K), reused verbatim) before releasing it, so the stream cannot
// drain during the hold; the residual occupancy at release time is read via
// onset(source STALLED_LOCK) -> onset(source MEMORY_STARVATION).
// UNVERIFIED: whether a direct row-2 -> row-3 DMA flow (no memtile hop)
// physically routes, and whether the sink's own ingress FIFO absorbs part of
// the hold before backpressure reaches the source -- both are hardware
// questions for the interactive capture task, flagged rather than guessed.
//===----------------------------------------------------------------------===//
module {{
  aie.device(npu1_2col) {{
    %shim_0_0 = aie.tile(0, 0) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}}
    %core_0_2 = aie.tile(0, 2)
    %core_0_3 = aie.tile(0, 3)

    %src_buf  = aie.buffer(%core_0_2) {{sym_name = "src_buf", address = {SRC_ADDR} : i32}} : memref<{OBJ}xi32>
    %lk_src_full = aie.lock(%core_0_2, 0) {{init = 0 : i32, sym_name = "lk_src_full"}}

    %march_buf = aie.buffer(%core_0_3) {{sym_name = "march_buf", address = {MARCH_ADDR} : i32}} : memref<{MARCH_ELEMS}xi32>
    %sink_buf  = aie.buffer(%core_0_3) {{sym_name = "sink_buf", address = {SRC_ADDR} : i32}} : memref<{OBJ}xi32>
    %lk_sink_go    = aie.lock(%core_0_3, 0) {{init = 0 : i32, sym_name = "lk_sink_go"}}
    %lk_sink_ready = aie.lock(%core_0_3, 1) {{init = 0 : i32, sym_name = "lk_sink_ready"}}

    aie.flow(%core_0_2, DMA : 0, %core_0_3, DMA : 0)
    aie.flow(%core_0_3, DMA : 0, %shim_0_0, DMA : 0)
    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)

    %core_src = aie.core(%core_0_2) {{
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %cOBJ = arith.constant {OBJ} : index
      scf.for %i = %c0 to %cOBJ step %c1 {{
        %iv = arith.index_cast %i : index to i32
        memref.store %iv, %src_buf[%i] : memref<{OBJ}xi32>
      }}
      aie.use_lock(%lk_src_full, Release, 1)
      aie.end
    }} {{stack_size = {STACK_SIZE} : i32}}

    %mem_src = aie.mem(%core_0_2) {{
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lk_src_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%src_buf : memref<{OBJ}xi32>, 0, {OBJ}) {{bd_id = 0 : i32}}
      aie.next_bd ^bd0
    ^end:
      aie.end
    }}

    %core_sink = aie.core(%core_0_3) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index{dwell}
      aie.use_lock(%lk_sink_go, Release, 1)
      aie.end
    }} {{stack_size = {STACK_SIZE} : i32}}

    %mem_sink = aie.mem(%core_0_3) {{
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%lk_sink_go, AcquireGreaterEqual, 1)
      aie.dma_bd(%sink_buf : memref<{OBJ}xi32>, 0, {OBJ}) {{bd_id = 0 : i32}}
      aie.use_lock(%lk_sink_ready, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:
      aie.use_lock(%lk_sink_ready, AcquireGreaterEqual, 1)
      aie.dma_bd(%sink_buf : memref<{OBJ}xi32>, 0, {OBJ}) {{bd_id = 1 : i32}}
      aie.next_bd ^bb3
    ^bb4:
      aie.end
    }}

    aie.runtime_sequence(%comp_buf: memref<{total}xi32>) {{
      aiex.npu.dma_memcpy_nd(%comp_buf[0, 0, 0, 0][1, 1, 1, {total}][0, 0, 0, 1]) {{id = 0 : i64, metadata = @drain}} : memref<{total}xi32>
      aiex.npu.dma_wait {{symbol = @drain}}
    }}
  }}
}}
"""


def emit(variant: str) -> str:
    if variant in SINGLE_TILE_VARIANTS:
        return _emit_single_tile(variant)
    if variant in TWO_TILE_K:
        return _emit_two_tile(variant, TWO_TILE_K[variant])
    raise KeyError(f"unknown variant {variant!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    print(emit(ap.parse_args().variant), end="")
