#!/usr/bin/env python3
"""tct_token_depth -- characterize the TCT completion-token buffer depth
(Experiment C, docs/superpowers/specs/2026-07-15-phoenix-dma-constant-
captures.md). Characterize-only: pin the number if the capture is clean,
otherwise document the mechanism/bound -- do not force a number.

**WHY THIS IS A LOWER-BOUND PROBE, not a saturation probe.** The device
model's oracle (NPU1.json task_complete_queue_size = 128, unvalidated) would
need 128 completed-but-undrained tokens to trigger DMA_TASK_TOKEN_STALL. That
is NOT safely reachable on real silicon, and the reason is toolchain-derived,
not guessed:

  1. `aiex.dma_start_task` lowers to an UNCONDITIONAL 32-bit register write
     (mlir-aie AIEDmaToNpu.cpp:142-191) -- there is NO queue-availability
     poll. aie-rt has a poll-for-slot helper (_XAieMl_DmaWaitForBdTaskQueue,
     xaie_dma_aieml.c:1257) but the lowering never calls it. Overrunning the
     4-deep task queue is a SILENTLY-DROPPED BD plus a sticky HW error bit
     ("Attempt to write to full task queue", aie_registers_aie2.json:26910),
     NOT back-pressure. So firing 128+ tasks no-await does not "self-pace" --
     it corrupts.
  2. `aiex.dma_free_task` is compile-time allocator bookkeeping only
     (AIEAssignRuntimeSequenceBDIDs.cpp:92-136) -- zero runtime completion
     guarantee.
  3. Reusing a BD id without an intervening await is compiler-FORBIDDEN
     ("Specified buffer descriptor ID N is already in use. Emit an
     aiex.dma_free_task operation to reuse BDs.", same file :54-58); freeing
     without awaiting drops that guard while giving no completion guarantee,
     so BD reuse then races live hardware. The canonical >16-task reuse test
     (shim_dma_bd_reuse/aie.mlir) ALWAYS awaits per batch -- it never relies
     on queue-depth ordering.

So the three routes to 128 undrained tokens all fail: reuse-without-await
races the NPU; reuse-with-await DRAINS the very tokens we want to accumulate
(and hits a counting-semaphore ambiguity -- dma_await_task waits for ONE
token on the channel, satisfied by an earlier sibling's completion); and
no-reuse caps at the 16 BD ids a tile has.

**The binding safety limit is the 4-deep task queue, not the 16 BD ids.**
XAIE_DMA_MAX_QUEUE_SIZE = 4 (aie-rt xaie_dma.c:45). A `dma_start_task` push
onto a FULL queue is silently dropped (finding above), and the host uC issues
pushes far faster than a multi-word transfer drains, so firing N tasks with no
await between them overflows the queue for any N that outpaces completion.
The ONLY safe pacing signal is `dma_await_task` -- which drains a token. So:

  - N = 4 is GUARANTEED safe: it fills the 4-deep queue exactly and issues no
    5th push, so no overflow is possible regardless of completion timing.
  - N = 8 is PRECEDENT-safe (not model-guaranteed): shim_dma_bd_reuse fires 8
    MM2S tasks back-to-back before its first await and is a passing E2E test,
    so 8-back-to-back is empirically safe on this HW class.
  - N > 8 is NEITHER -- do not add it. It would rely on unverified timing and
    can silently drop a BD -> the receive never completes -> await(recv) hangs
    -> NPU wedge.

This generator therefore fires N in {4, 8} fire-and-forget MM2S tasks (ids
0..N-1, reserving id 15 for the receive), each issue_token=true, NEVER
awaited, NO reuse -- a clean LOWER BOUND on the buffer depth ("> N undrained
tokens did not stall"). Like Experiment B, this most likely reveals that the
token transport out-paces generation so the tile buffer never fills at all
(there is no dialect-level knob to throttle the token return route -- it
transits a fixed column-control overlay to the shim, configured by the
compiler, not the kernel author).

**Why shim, not memtile/memmod.** DMA_TASK_TOKEN_STALL exists on memtile
(event 140), memmod/compute-local (event 102), and shim (event 75). A REAL,
checked-in upstream test -- mlir-aie/test/npu-xrt/shim_dma_bd_reuse/aie.mlir
-- already demonstrates fire-and-forget shim MM2S tasks via the low-level
`aiex.dma_configure_task` / `dma_start_task` API, each independently marked
`{issue_token = true}` (the Enable_Token_Issue bit, token.rs:70-72). Building
the memtile/memmod variant would mean reverse-engineering the same idiom on a
tile type with no confirmed working precedent -- shim is the toolchain-derived
choice, not a coin flip.

**Multi-task, not self-looping single-BD.** A self-looping single-BD chain
(`aie.next_bd` back to itself) is ONE task-queue push whose BD chain never
reaches a terminal block, so it never completes and FINISHED_TASK fires zero
times (token.rs task-queue model + stepping.rs's DmaFinishedTask emission,
which fires only when a queued task's chain reaches ^end). This generator
pushes many DISTINCT, FINITE tasks (each one BD, `^end` after a single pass)
via repeated `aiex.dma_configure_task` + `dma_start_task`.

**Also required for issue_token to route at all**: the `aie.packet_flow`
routing the shim's own TileControl source to its South port (copied verbatim
from shim_dma_bd_reuse/aie.mlir:31-35) -- per docs/arch/tct-completion-
model.md this is the token's physical egress path; omitting it would silently
strand every issued token.
"""
import argparse

# Shim tile under test (module="shim" in the decoded trace; pt_name "shim"
# per trace_decoder.decode._PT_CODE_TO_NAME). Compute-tile passthrough
# provides the stream sink/source the shim MM2S/S2MM channels need to
# actually drain/fill (mirrors shim_dma_bd_reuse's own topology).
SHIM_ROW, SHIM_COL = 0, 0
CORE_ROW, CORE_COL = 2, 0

# Reserve the top BD id for the S2MM receive (one tile has 16 ids, Start_BD_ID
# 4-bit, shared across both directions per shim_dma_bd_reuse). The MM2S
# fire-and-forget tasks take 0..N-1 with NO reuse. N_MAX is set by the 4-deep
# task queue's safety envelope, NOT the BD-id count: see the module docstring
# -- N=4 is queue-guaranteed safe, N=8 is precedent-safe, N>8 risks a wedge.
RECV_BD_ID = 15
N_MAX = 8  # precedent-safe ceiling (shim_dma_bd_reuse fires 8 back-to-back)

OBJ_SIZES = {"small": 16, "large": 256}  # i32 words per MM2S task. Smaller
                                         # finishes faster -> tokens produced
                                         # at a higher rate for the same count.

# variant -> (n_tasks, obj key). n_tasks undrained MM2S tokens (ids 0..n-1),
# all issue_token=true, none awaited. Sweep the count (does more undrained
# get closer to a stall?) and the task size (does a faster token rate push
# occupancy up before transport drains it?). N stays in {4, 8}: 4 is
# queue-guaranteed safe, 8 is precedent-safe (see module docstring).
VARIANTS = {
    "n4_small": dict(n_tasks=4, obj="small"),
    "n8_small": dict(n_tasks=8, obj="small"),
    "n4_large": dict(n_tasks=4, obj="large"),
    "n8_large": dict(n_tasks=8, obj="large"),
}


def _mm2s_tasks_mlir(n_tasks: int, obj: int) -> str:
    """N fire-and-forget MM2S tasks, bd_id 0..n-1, each issue_token=true,
    reading the same source buffer. No await, no free, no reuse -- the
    HW-safe lower-bound design."""
    lines = []
    for i in range(n_tasks):
        lines.append(
            f"      %t{i} = aiex.dma_configure_task(%shim_0_0, MM2S, 0) {{\n"
            f"        aie.dma_bd(%dma_buf : memref<{obj}xi32>, 0, {obj}) {{bd_id = {i} : i32}}\n"
            f"        aie.end\n"
            f"      }} {{issue_token = true}}\n"
            f"      aiex.dma_start_task(%t{i})"
        )
    return "\n".join(lines)


def emit(variant: str) -> str:
    cfg = VARIANTS[variant]
    obj = OBJ_SIZES[cfg["obj"]]
    n = cfg["n_tasks"]
    total = n * obj
    tasks = _mm2s_tasks_mlir(n, obj)

    return f"""//===- tct_token_depth {variant} -------------------------------*- MLIR -*-===//
// TCT completion-token buffer depth probe (Experiment C, characterize-only,
// HW-safe LOWER-BOUND design). Shim tile (0,0) MM2S ch0 fires {n}
// fire-and-forget tasks (bd ids 0-{n - 1}, obj={cfg['obj']} = {obj} words/task),
// each issue_token=true and NEVER awaited -- so {n} completion tokens go
// undrained. If DMA_TASK_TOKEN_STALL does not fire, buffer depth > {n}.
// No BD reuse -> no race -> HW-safe. See module docstring for why 128 is
// unreachable safely.
//===----------------------------------------------------------------------===//
module {{
  aie.device(npu1_2col) {{
    %shim_0_0 = aie.tile({SHIM_COL}, {SHIM_ROW}) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}}
    %core_0_2 = aie.tile({CORE_COL}, {CORE_ROW}) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}}

    %core_buf = aie.buffer(%core_0_2) {{sym_name = "core_buf"}} : memref<{obj}xi32>

    // Input path: shim MM2S ch0 -> core S2MM ch0
    aie.flow(%shim_0_0, DMA : 0, %core_0_2, DMA : 0)
    // Output path: core MM2S ch0 -> shim S2MM ch0
    aie.flow(%core_0_2, DMA : 0, %shim_0_0, DMA : 0)

    // Packet flow for issue_token on shim MM2S -- the token's physical
    // egress path (AIEGenerateColumnControlOverlay); without this, every
    // issued token is silently stranded. Copied verbatim from the proven
    // precedent (shim_dma_bd_reuse/aie.mlir).
    aie.packet_flow(0x3) {{
      aie.packet_source<%shim_0_0, "TileControl" : 0>
      aie.packet_dest<%shim_0_0, "South" : 0>
    }}

    // Core-less passthrough: the compute tile's OWN dma engine ping-pongs
    // core_buf between S2MM-fill and MM2S-drain via lock handshake, no
    // aie.core region needed (proven idiom, shim_dma_bd_reuse/aie.mlir).
    // This is what lets the shim MM2S tasks actually complete (and so issue
    // their tokens) -- without a stream drain they would backpressure forever.
    %lock_in = aie.lock(%core_0_2, 0) {{init = 1 : i32, sym_name = "lock_in"}}
    %lock_out = aie.lock(%core_0_2, 1) {{init = 0 : i32, sym_name = "lock_out"}}

    %mem_0_2 = aie.mem(%core_0_2) {{
      %0 = aie.dma_start(S2MM, 0, ^s2mm, ^mm2s_entry)
    ^s2mm:
      aie.use_lock(%lock_in, AcquireGreaterEqual, 1)
      aie.dma_bd(%core_buf : memref<{obj}xi32>, 0, {obj})
      aie.use_lock(%lock_out, Release, 1)
      aie.next_bd ^s2mm
    ^mm2s_entry:
      %1 = aie.dma_start(MM2S, 0, ^mm2s, ^end)
    ^mm2s:
      aie.use_lock(%lock_out, AcquireGreaterEqual, 1)
      aie.dma_bd(%core_buf : memref<{obj}xi32>, 0, {obj})
      aie.use_lock(%lock_in, Release, 1)
      aie.next_bd ^mm2s
    ^end:
      aie.end
    }}

    aie.runtime_sequence(%dma_buf: memref<{obj}xi32>, %output: memref<{total}xi32>) {{
      // Pre-configure shim S2MM ch0: one long-lived task, an N-dimensional
      // BD that receives all {n} passthrough transfers into consecutive
      // {obj}-word slices of %output. Reserved bd_id ({RECV_BD_ID}) stays
      // outside the MM2S 0-{n - 1} range. This IS awaited (its token drains),
      // so it does not count toward the undrained accumulation.
      %recv = aiex.dma_configure_task(%shim_0_0, S2MM, 0) {{
        aie.dma_bd(%output : memref<{total}xi32>, 0, {total}, [<size = {n}, stride = {obj}>, <size = {obj}, stride = 1>]) {{bd_id = {RECV_BD_ID} : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%recv)

{tasks}

      // Await ONLY the receive -- guarantees the passthrough drained every
      // MM2S transfer (so all {n} MM2S tasks completed and issued their
      // tokens), while leaving those {n} MM2S tokens themselves undrained.
      aiex.dma_await_task(%recv)
    }}
  }}
}}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    print(emit(ap.parse_args().variant), end="")
