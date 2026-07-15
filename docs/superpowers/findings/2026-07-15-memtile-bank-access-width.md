# Memtile bank access width, interleave, and the input task-queue bound: AM020 derivation

Date: 2026-07-15
Task: Phoenix-retirement arc, Task 1 (HW-free: comments/docs upgrade + investigation)
Related: [`2026-07-14-dma-bank-access-width.md`](2026-07-14-dma-bank-access-width.md) (the compute-tile
measurement this finding extends to the memtile via architecture-manual derivation)

## AM020 derivation (ch.5)

Source: `docs/xdna/am020-aie-ml/chapter-5-aie-ml-memory-tile-architecture.md`.

Three facts pin the memtile's bank width, interleave, and per-channel port count
without requiring a new hardware capture:

- **Line 31 / 105 -- bank width.** "512 KB memory arranged into 16 banks (each
  128-bit wide and 2k words deep)... Each bank allows one read or one write
  every cycle." This is the same 128-bit granule the compute-tile measurement
  ([`2026-07-14-dma-bank-access-width.md`](2026-07-14-dma-bank-access-width.md))
  found empirically for `PHYSICAL_BANK_WIDTH_BITS` on a compute tile --
  `memtile::PHYSICAL_BANK_WIDTH_BITS` (also 128) is therefore a derived value,
  not a placeholder.
- **Line 137 -- interleave.** "Interleaving is done at a 128-bit granularity,
  such that sequential 128-bit accesses map to different banks and wrap around
  after the 16 banks, every 256B." This is exactly `(addr >> 4) & 0xF` --
  `COMPUTE_INTERLEAVE_SHIFT` (trailing_zeros of 16 bytes = 4) masked to 16
  banks -- confirming `BankLayout::MemTile::physical_bank` in
  `src/device/banking.rs` is derived, not the "preserve the previous flat
  interleave" placeholder the comment used to claim.
- **Line 153 -- single shared interface.** "[Each channel] can load a BD,
  generate address, access memory over a shared interface" -- singular, per
  channel. This grounds the single-granule-per-cycle cap in
  `src/device/dma/engine/stepping.rs` (`granule_capped_words`) for
  `BankLayout::MemTile`: one 128-bit memory port per channel, the same shape
  the compute-tile measurement found for the compute-tile DMA. Combined with
  the 128-bit bank width above, the memtile arm of `access_granule_bytes` is
  now grounded in the architecture manual rather than carried over from the
  compute-tile inference alone.

**What remains open.** AM020 ch.5 describes the steady-state/contiguous shape;
it does not by itself pin the cap's exact per-cycle behavior under a **strided**
single-channel access (whether a strided memtile S2MM still claims exactly one
granule per cycle, or something the manual text doesn't distinguish). That
corner is the residual noted in the updated code comments and the
`docs/fidelity-gaps/dma-stream-resources.md` gap row, and is HW-confirmed by
Experiment A2 below.

## Input task-queue bound (Step 3 investigation)

**Finding: the emulator already bounds the input task queue at 4, and it is
already derived from the toolchain, not guessed.** `src/device/dma/token.rs`
defines `MAX_TASK_QUEUE_DEPTH: usize = 4` (token.rs:109), justified in its own
doc comment by aie-rt's `XAIE_DMA_MAX_QUEUE_SIZE 4U` (`xaie_dma.c:45`) and the
per-tile-type `StartQSizeMax = 4U` channel property, uniform across compute,
memtile, and shim/NoC modules. `TaskQueue` (token.rs:212-218) wraps a
`VecDeque<TaskQueueEntry>` with a `capacity` field; `push()` (token.rs:239-246)
returns `Err(QueueFull)` and sets a sticky overflow flag once
`entries.len() >= capacity`, matching the AM025 `Task_Queue_Overflow` status
bit. `TaskQueue::new_default()` (token.rs:230-232) constructs it at
`MAX_TASK_QUEUE_DEPTH`. Per-channel, `ChannelContext` holds this as
`task_queue: TaskQueue` (`src/device/dma/channel.rs:314-315`), and
`DmaEngine::enqueue_task` (`src/device/dma/engine/task_queue_ops.rs:12`) is the
enqueue path that pushes into it and rejects on `QueueFull`. A unit test
(`token.rs:848-864`, `queue_full_at_hardware_depth`) asserts the bound directly.

This matches AM020 ch.5:51/65 ("Support task queue and task-complete-tokens;
queue depth is four tasks per channel") exactly -- the emulator's bound was
already toolchain-derived (from aie-rt) before this task, and AM020 now
independently corroborates the same number for the memtile specifically.

**One stale comment, not fixed here.** `src/device/dma/engine/task_queue_ops.rs:8`
says "each channel has an 8-deep task queue" -- this contradicts the enforced
`MAX_TASK_QUEUE_DEPTH = 4` and its own passing test. Per this task's scope
(comment/doc upgrade + read-only investigation, no behavior or unrelated-comment
changes), the stale "8-deep" comment is left for a future cleanup task to
correct; it does not reflect an actual unbounded or differently-bounded queue,
only a doc typo/copy-paste artifact in a file this task did not otherwise touch.

Distinct from the task queue: the *completion token* buffer (`TokenState`,
token.rs, backed by an unbounded `VecDeque`) is a separate structure and is
deliberately modeled as unbounded -- already documented in its own doc comment
as a known simplification (real hardware backpressures the channel instead of
dropping tokens). It is not the structure AM020 ch.5:51/65 describes and is out
of scope for this note.

## A1 HW result (pending Task 5)

## A2 HW result (pending Task 5)
