#!/usr/bin/env python3
"""tct_token_depth -- characterize the TCT completion-token buffer depth
(Experiment C, docs/superpowers/specs/2026-07-15-phoenix-dma-constant-
captures.md). Characterize-only: pin the number if the capture is clean,
otherwise document the mechanism/bound -- do not force a number.

**Why shim, not memtile/memmod.** The brief lists three candidate tiles for
DMA_TASK_TOKEN_STALL: memtile (event 140), memmod/compute-local (event 102),
shim (event 75). A REAL, checked-in upstream test --
mlir-aie/test/npu-xrt/shim_dma_bd_reuse/aie.mlir -- already demonstrates the
exact mechanism this experiment needs on a SHIM MM2S channel: firing MANY
fire-and-forget tasks via the low-level `aiex.dma_configure_task` /
`dma_start_task` / `dma_await_task` / `dma_free_task` API, each task
independently marked `{issue_token = true}` (the Enable_Token_Issue bit,
token.rs:70-72, exposed directly at the MLIR level), reusing a small BD-id
budget across many "waves" of tasks. Building the memtile/memmod variant
would mean reverse-engineering the same idiom on a tile type with no
confirmed working precedent in this toolchain checkout -- shim removes that
guess entirely, so it is the toolchain-derived choice, not a coin flip.

**Multi-task, not self-looping single-BD.** A self-looping single-BD chain
(`aie.next_bd` pointing back at itself, e.g. every OTHER sibling probe in
this file family: bankdisc.py, producer_probe.py, mm2s_egress_depth.py) is
ONE task-queue push whose BD chain never reaches a terminal block, so it
never completes and FINISHED_TASK fires zero times (confirmed by
inspecting token.rs's task-queue model and stepping.rs's DmaFinishedTask
emission, which fires only when a queued task's chain reaches ^end). This
generator instead pushes many DISTINCT, FINITE tasks (each one BD, `^end`
after a single pass) via repeated `aiex.dma_configure_task` +
`dma_start_task` calls -- confirmed live usage, not synthesized from
scratch, in `shim_dma_bd_reuse/aie.mlir:76-218` (20 fire-and-forget MM2S
tasks on one shim channel, reusing BD ids in batches of 8).

**The "wave" mechanism.** Start_BD_ID is a 4-bit field on compute/shim tiles
(token.rs START_BD_ID_WIDTH_COMPUTE = 4 -> 16 ids, shared across BOTH
directions of one tile per the precedent's own numbering: MM2S wave tasks
take ids 0-7, the S2MM receive task takes id 8). A "wave" pushes WAVE_SIZE
MM2S tasks (bd_id 0..WAVE_SIZE-1) back to back with NO await between them,
then awaits ONLY the wave's last (terminal) task and frees the others --
mirroring the precedent's batch-of-8 exactly (in-order completion per
channel means the terminal task finishing implies every earlier task in
the wave already finished too, so freeing them without individually
awaiting is safe). Reusing bd_id 0..WAVE_SIZE-1, WAVE_SIZE=8 matches the
proven precedent's own batch size verbatim rather than probing the
untested 9-15 range; NUM_WAVES is unbounded by the BD-id budget (each wave
reuses the same ids after freeing), so it is set generously instead
(NUM_WAVES=20).

**Throttle interpretation (the one interpretive judgment call here,
flagged rather than silently assumed).** The brief frames the sweep as
"the token-return stream route is throttled to a slow/backpressured
consumer" -- language mirroring Experiment B's physically-routed stream
throttle (mm2s_egress_depth.py's sink tile withholding a lock). But TCT
tokens are NOT user-routed via `aie.flow`/stream-switch config at all: per
docs/arch/tct-completion-model.md and AIEGenerateColumnControlOverlay.cpp,
they transit a FIXED column-control-overlay path to the shim (`col<<21 |
row<<16 | actor_id` header), configured automatically by the compiler, not
by kernel authors. There is no dialect-level "throttle this route" knob.
The mechanically faithful stand-in implemented here: TOKENS_PER_WAVE, how
many of each wave's WAVE_SIZE tasks carry `issue_token = true` (the actual
Enable_Token_Issue control). A low fraction (the "control" variant: only
the wave-terminal task, exactly the precedent's own default pattern) means
every issued token is immediately consumed by that same wave's mandatory
await -- nothing accumulates. A high fraction (the "all_*" variants: every
task in the wave) means most tokens are NEVER individually drained, so
they should pile up in the per-tile completion-token queue across
successive waves -- the same net effect ("tokens back up faster than they
drain") the brief describes, achieved through the real hardware control
that exists, rather than a stream-switch throttle that doesn't apply to
TCT transport.

**Sweep axes.**
  - OBJ (task size / token rate): "small" (16 words) vs "large" (256
    words) per-task MM2S transfer -- smaller finishes faster, so tokens
    are produced at a higher real-time rate for the same task count.
  - TOKENS_PER_WAVE (return-route throttle, see above): 1 (control), half
    (WAVE_SIZE // 2), all (WAVE_SIZE).

**Deadlock risk, flagged not guessed past.** Once the completion-token
queue is genuinely full, does a task that itself requests
Enable_Token_Issue but can't get a queue slot fail to complete at all (a
real per-task stall), or does only the TOKEN DELIVERY stall while the
underlying DMA transfer still finishes? If the former, this design's own
per-wave `dma_await_task` on the (token-carrying) terminal task could hang
past the wave where the queue first saturates, since reclaiming that
wave's BD ids depends on exactly the token-issuing completion event the
experiment is trying to starve. `all_small_safe_reclaim` is the offered
mitigation: its wave-terminal task carries `issue_token = false`, so
`dma_await_task` must fall back to a non-token completion signal to make
progress (mirroring the emulator's own documented Channel_Running
fallback, docs/arch/tct-completion-model.md -- NOT independently confirmed
on real hardware). Task 7 should watch for a hang on `all_small`/`all_large`
at the wave where the outstanding count crosses ~128 and fall back to the
`_safe_reclaim` variant (or reduce NUM_WAVES) if so.

**Also required for issue_token to route at all**: the `aie.packet_flow`
routing the shim's own TileControl source to its South port (copied
verbatim from shim_dma_bd_reuse/aie.mlir:31-35) -- per
docs/arch/tct-completion-model.md this is the token's physical egress path;
omitting it would silently strand every issued token.
"""
import argparse

# Shim tile under test (module="shim" in the decoded trace; pt_name "shim"
# per trace_decoder.decode._PT_CODE_TO_NAME). Compute-tile passthrough
# provides the stream sink/source the shim MM2S/S2MM channels need to
# actually drain/fill (mirrors shim_dma_bd_reuse's own topology).
SHIM_ROW, SHIM_COL = 0, 0
CORE_ROW, CORE_COL = 2, 0

WAVE_SIZE = 8       # matches shim_dma_bd_reuse's own batch size verbatim
RECV_BD_ID = 8      # one id reserved for the S2MM receive task (id 8, same
                    # numbering the precedent uses -- ids 0-7 stay free for
                    # the MM2S wave)
NUM_WAVES = 20      # generous: (WAVE_SIZE-1)*NUM_WAVES = 140 possible
                    # outstanding tokens by design's end for the "all_*"
                    # variants -- safely past the 128 aie-rt/NPU1.json
                    # oracle number (docs/device-model-audit.md's
                    # task_complete_queue_size row), without probing the
                    # untested >8-tasks-per-wave BD-id range.
NUM_TOTAL_TASKS = NUM_WAVES * WAVE_SIZE
STACK_SIZE = 0x400

OBJ_SIZES = {"small": 16, "large": 256}  # i32 words per MM2S task -- the
                                         # task-size / token-rate sweep axis

# variant -> (tokens_per_wave, obj key, reclaim_token). tokens_per_wave: how
# many of the wave's WAVE_SIZE tasks (counting from the END, i.e. always
# including or excluding the terminal per reclaim_token below) carry
# issue_token=true. reclaim_token: explicit override for the wave-terminal
# task specifically (the one dma_await_task/dma_free_task rely on).
VARIANTS = {
    # CONTROL: exactly shim_dma_bd_reuse's own pattern (only the
    # wave-terminal task ever issues a token, immediately consumed by that
    # same wave's mandatory await). Expect ~0 outstanding at all times and
    # TOKEN_STALL never firing -- confirms raw task VOLUME (160 tasks)
    # alone isn't the trigger, only genuinely undrained tokens are.
    "control":    dict(tokens_per_wave=1, obj="small", reclaim_token=True),
    "all_small":  dict(tokens_per_wave=WAVE_SIZE,      obj="small", reclaim_token=True),
    "all_large":  dict(tokens_per_wave=WAVE_SIZE,      obj="large", reclaim_token=True),
    "half_small": dict(tokens_per_wave=WAVE_SIZE // 2, obj="small", reclaim_token=True),
    "half_large": dict(tokens_per_wave=WAVE_SIZE // 2, obj="large", reclaim_token=True),
    # Deadlock mitigation (see module docstring) -- offered as a ready-made
    # escape hatch for Task 7, not silently assumed safe.
    "all_small_safe_reclaim": dict(tokens_per_wave=WAVE_SIZE, obj="small", reclaim_token=False),
}


def _issue_token(i: int, tokens_per_wave: int, reclaim_token: bool) -> bool:
    """Whether wave-local task index i (0..WAVE_SIZE-1) carries issue_token.

    The terminal task (i == WAVE_SIZE - 1) is controlled independently by
    reclaim_token (it is what dma_await_task/dma_free_task rely on to
    safely reuse the wave's BD ids); every other task gets a token iff it
    is among the last `tokens_per_wave` indices.
    """
    if i == WAVE_SIZE - 1:
        return reclaim_token
    return i >= WAVE_SIZE - tokens_per_wave


def _wave_mlir(wave: int, tokens_per_wave: int, reclaim_token: bool, obj: int) -> str:
    lines = []
    task_names = [f"%t{wave}_{i}" for i in range(WAVE_SIZE)]
    for i, tname in enumerate(task_names):
        issue = _issue_token(i, tokens_per_wave, reclaim_token)
        attr = " {issue_token = true}" if issue else ""
        lines.append(
            f"      {tname} = aiex.dma_configure_task(%shim_0_0, MM2S, 0) {{\n"
            f"        aie.dma_bd(%dma_buf : memref<{obj}xi32>, 0, {obj}) {{bd_id = {i} : i32}}\n"
            f"        aie.end\n"
            f"      }}{attr}\n"
            f"      aiex.dma_start_task({tname})"
        )
    lines.append(f"      aiex.dma_await_task({task_names[-1]})")
    for tname in task_names[:-1]:
        lines.append(f"      aiex.dma_free_task({tname})")
    return "\n".join(lines)


def emit(variant: str) -> str:
    cfg = VARIANTS[variant]
    obj = OBJ_SIZES[cfg["obj"]]
    total = NUM_TOTAL_TASKS * obj
    waves = "\n".join(
        _wave_mlir(w, cfg["tokens_per_wave"], cfg["reclaim_token"], obj)
        for w in range(NUM_WAVES)
    )

    return f"""//===- tct_token_depth {variant} -------------------------------*- MLIR -*-===//
// TCT completion-token buffer depth probe (Experiment C, characterize-only).
// Shim tile (0,0) MM2S ch0 fires {NUM_TOTAL_TASKS} fire-and-forget tasks
// ({NUM_WAVES} waves of {WAVE_SIZE}, BD ids 0-{WAVE_SIZE - 1} reused per wave),
// tokens_per_wave={cfg['tokens_per_wave']}, obj={cfg['obj']} ({obj} words/task),
// reclaim_token={cfg['reclaim_token']}. See module docstring for the full
// design rationale and the deadlock-risk flag.
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
      // BD that receives all {NUM_TOTAL_TASKS} passthrough transfers into
      // consecutive {obj}-word slices of %output. Reserved bd_id
      // ({RECV_BD_ID}) stays outside the MM2S wave's 0-{WAVE_SIZE - 1} range.
      %recv = aiex.dma_configure_task(%shim_0_0, S2MM, 0) {{
        aie.dma_bd(%output : memref<{total}xi32>, 0, {total}, [<size = {NUM_TOTAL_TASKS}, stride = {obj}>, <size = {obj}, stride = 1>]) {{bd_id = {RECV_BD_ID} : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%recv)

{waves}

      aiex.dma_await_task(%recv)
    }}
  }}
}}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    print(emit(ap.parse_args().variant), end="")
