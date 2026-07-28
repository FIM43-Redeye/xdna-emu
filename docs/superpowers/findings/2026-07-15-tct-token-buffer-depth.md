# TCT completion-token buffer depth (Experiment C) -- characterized, 128 kept unvalidated

**Date:** 2026-07-15
**Branch:** `feature/phoenix-dma-captures`
**Experiment:** C of the Phoenix DMA-constant capture arc
(`docs/superpowers/specs/2026-07-15-phoenix-dma-constant-captures.md`).
**Deliverables:** `tools/experiments/tct_token_depth.py`, `..._measure.py`,
`c_tct_capture.py`. Related: `docs/arch/tct-completion-model.md`,
`docs/superpowers/findings/2026-07-08-tct-token-transport-hw-validation.md`.

## Question

`token.rs` models the per-tile completion-token buffer (the resource behind
`DMA_TASK_TOKEN_STALL` / status bit[5] `Stalled_TCT`) as **unbounded**.
NPU1.json's `task_complete_queue_size` gives **128**, but that number is
unvalidated. Can we pin it on real Phoenix silicon?

## Outcome: characterize-only (as the spec pre-accepted)

**128 cannot be pinned by any HW-safe trace method. HW establishes a lower
bound of `depth > 8` and confirms the saturation mechanism never engages.**
The constant stays as-is: EMU token buffer **unbounded** (deliberate),
NPU1.json's **128** kept as the documented-but-unvalidated oracle.

## Why 128 is unreachable safely (toolchain-derived)

Triggering `TOKEN_STALL` needs 128 completed-but-undrained tokens. Every route
there is unsafe, and the reasons are primary-source toolchain facts, not
guesses (full citations in `tct_token_depth.py`'s module docstring):

1. **`dma_start_task` does not self-pace.** It lowers to an *unconditional*
   32-bit register write (`AIEDmaToNpu.cpp:142-191`) -- no queue-availability
   poll. A poll-for-slot helper exists in aie-rt
   (`_XAieMl_DmaWaitForBdTaskQueue`, `xaie_dma_aieml.c:1257`) but the lowering
   never calls it. Overrunning the **4-deep** task queue
   (`XAIE_DMA_MAX_QUEUE_SIZE = 4`, `xaie_dma.c:45`) **silently drops the BD**
   plus latches a sticky HW error ("Attempt to write to full task queue",
   `aie_registers_aie2.json:26910`). It is not back-pressure. So firing 128+
   no-await tasks corrupts; it does not accumulate.
2. **BD reuse without await is compiler-forbidden.** Reusing an id while it is
   still allocated errors ("Specified buffer descriptor ID N is already in
   use. Emit an aiex.dma_free_task operation to reuse BDs.",
   `AIEAssignRuntimeSequenceBDIDs.cpp:54-58`); `dma_free_task` is compile-time
   allocator bookkeeping only (no runtime completion guarantee), so freeing
   without awaiting drops the guard while BD reuse then races live silicon.
   The canonical >16-task reuse test (`shim_dma_bd_reuse/aie.mlir`) **always
   awaits per batch** -- it never relies on queue-depth ordering.
3. **The only safe pacing signal is `dma_await_task`, which drains a token**
   -- defeating the accumulation -- and additionally carries a
   counting-semaphore ambiguity (it waits for *one* token on the channel, not
   a specific task's).

So the safe firing ceiling is set by the 4-deep queue, not the 16 BD ids:
**N = 4** is queue-guaranteed safe (fills the queue exactly, no 5th push);
**N = 8** is precedent-safe (`shim_dma_bd_reuse` fires 8 back-to-back before
its first await); **N > 8** risks a wedge.

## What was run

Shim tile (0,0 declared -> **(0,1)** decoded, column virtualization confirmed
on HW) MM2S ch0 fires N fire-and-forget tasks, each `issue_token=true`, NEVER
awaited (no reuse, no race). A compute-tile passthrough drains the stream so
the tasks actually complete and issue their tokens; a long-lived shim S2MM
receive (awaited once, on a separate channel) confirms end-to-end drain.
Sweep N in {4, 8} x obj in {small=16w, large=256w}, 3 reps each, Chess.

Trace events (shim): `DMA_TASK_TOKEN_STALL` (the crux), `DMA_MM2S_0_FINISHED_TASK`
+ `DMA_S2MM_0_FINISHED_TASK` (completed-token edges), `DMA_MM2S_0_START_TASK`.

## HW results (Phoenix, 3 reps each, identical across reps)

| variant  | undrained tokens N | MM2S finished | recv finished | `TOKEN_STALL` |
|----------|:---:|:---:|:---:|:---:|
| n4_small | 4 | 4 | 1 | **never** |
| n8_small | 8 | 8 | 1 | **never** |
| n4_large | 4 | 4 | 1 | **never** |
| n8_large | 8 | 8 | 1 | **never** |

Every task completed and issued its token (MM2S-finished == N in all cases --
no silent drops, no wedge), yet the per-tile completion-token buffer never
signaled full, even at the safe ceiling of 8 undrained tokens, for both task
sizes.

## Interpretation

- **Lower bound: `depth > 8`** on Phoenix, consistent with the NPU1.json
  oracle of 128. Nothing observed contradicts 128; nothing safely reachable
  can confirm it.
- **The saturation mechanism never engages -- the Experiment-B parallel.**
  Just as B's egress FIFO never starved (memory out-bandwidths the 32-bit
  stream), C's completion-token buffer never fills: the token return route (a
  fixed column-control overlay to the shim, configured by the compiler, with
  no dialect-level throttle knob) out-paces token generation. Even 8
  completed-but-unawaited tokens drain to the shim/host faster than they are
  produced, so occupancy never approaches any buffer bound. Task size makes no
  difference (small finishes faster but the transport still keeps up).
- **`Stalled_TCT` is a saturation-only signal.** With no way to safely force
  saturation and no trace event exposing buffer *occupancy* (only the binary
  "full" event), the depth is not trace-observable below its own ceiling --
  structurally the same limitation B hit.

## Decision

- **EMU: keep the token buffer unbounded** (`token.rs`). It was made unbounded
  deliberately because aie-rt exposes only the 1-bit `Stalled_TCT` flag with
  no numeric depth; nothing here changes that, and the gap "only bites at >128
  outstanding TCTs" -- a regime the safe probe shows the hardware itself keeps
  the buffer far below.
- **Keep NPU1.json's 128** as the documented-unvalidated oracle. Do not fit a
  number: the spec forbids calibrating to a target, and the HW cannot settle
  it either way.
- **Gap-doc row -> CHARACTERIZED** (`docs/device-model-audit.md`
  task-complete-queue row + item 3): HW lower-bounded `> 8`, saturation
  mechanism shown to not engage, 128 unpinnable safely.
