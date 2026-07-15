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

## A2 HW result: a single memtile MM2S is stream-bound (strided ~= contiguous)

Method: one memtile MM2S channel, a strided `aie.dma_bd` sweeping the byte
stride, 256-word transfer, span bracketed by `DMA_MM2S_SEL0_FINISHED_BD`
(preceding `STALLED_LOCK` falling edge to `FINISHED_BD` rising edge). Real
Phoenix NPU1, 1 rep.

| stride | 4B (contiguous) | 16B | 64B | 256B |
|--------|-----------------|-----|-----|------|
| span (cycles) | 251 | 258 | 258 | 258 |

`span(strided)/span(contiguous) ~= 258/251 ~= 1.03` -- **~1, not ~4.** A strided
single-channel transfer is NOT serialized relative to a contiguous one. The
transfer moves 256 words in ~256 cycles = **1 word/cycle**, independent of
stride.

**Why the discriminator can't distinguish 1-vs-4 here (the confound):** a single
memtile MM2S drains to one 32-bit AIE stream, which physically caps it at 1
word/cycle. The memory-side granule (which could move 4 words = 128 bits per
bank access) is therefore never the bottleneck, so span is stream-bound and
insensitive to memory-side parallelism. This is the same reason the compute-tile
`bankdisc` measured bank width via CONFLICT AREA, not span
([`2026-07-14-dma-bank-access-width.md`](2026-07-14-dma-bank-access-width.md)).
The clean fact A2 does establish: strided is not slower than contiguous on HW.

## A1 HW result: real bank contention, but minimal -- stream-bound

The original A1 (contiguous buffers "pinned to bank 0") was broken: a memtile
interleaves every 16B over 16 banks, so a 256-word (1KB) buffer spans ALL 16
banks and a single traced bank saw almost no conflict (an initial HW capture
read `CONFLICT_DM_BANK_0 = 0`). Redesigned (commit `b951c3fe`): each channel's
BD is **strided 256B** (`[<size=64, stride=64>, <size=4, stride=1>]`), so every
access lands on the SAME physical bank `(base>>4)&0xF`. Two channels
(S2MM fill + MM2S drain) run concurrently; variants place them on the same bank
(`collide`), different banks (`apart`), or drain-only (`solo`). Real Phoenix, 1 rep.

| variant | CONFLICT_DM_BANK_0 (cycles) | drain span |
|---------|-----------------------------|------------|
| a1_solo (floor)               | 0  | 251 |
| a1_apart (fill bank 8, drain bank 0) | 0  | 251 |
| a1_collide (both bank 0)      | **10** | 251 |

**Validity gates pass:** contention is real and bank-specific -- `collide` fires
conflict, `apart`/`solo` do not. **But the conflict is tiny (10 cycles) and there
is zero mutual slowdown** (`collide` span == `solo` span == 251). Quantitatively
consistent with stream-bound DMAs: each channel makes ~64 granule accesses to
bank 0 over ~251 cycles (~25% bank occupancy, stream-paced at 1 word/cycle), so
`P(both same cycle) ~= 0.25^2 ~= 6%` -> ~16 expected conflict cycles, 10
observed. Two stream-bound DMAs cannot saturate a memtile bank.

## Synthesis: memtile DMA is stream-bound; memory is never the bottleneck

A1 and A2 converge on one conclusion:

> A memtile DMA<->stream transfer is capped at **1 word/cycle** by the 32-bit AIE
> stream. The memtile memory subsystem -- 16x 128-bit banks, 9 read + 9 write
> 128-bit interfaces, 30 GB/s (AM020 ch.5:35/105) -- vastly out-bandwidths any
> single stream, so the memory side is never the limiter for a stream-fed DMA.

Consequences:
- **The original open corner is MOOT for stream-fed transfers.** "Does a strided
  memtile S2MM move 1 word/cycle or up to 4?" -- on HW it is 1 word/cycle
  regardless of stride OR contention, because memory is never the bottleneck.
  Bank width = 16B stands (AM020-derived, Task 1); these stream-bound
  measurements cannot contradict it.
- **The emulator's granule cap is only wrong if the emulator is not itself
  stream-bound.** DECISIVE CHECK (pending, next session): run the same xclbins
  under `XDNA_EMU=1` and compare spans to HW's ~251. EMU ~= 251 for all -> the
  emulator is already stream-bound, the cap is harmless, gap closes as "no
  fidelity error." EMU makes strided slow or exceeds 1 word/cycle -> that is the
  real gap and the memtile granule cap should be lifted.

**Caveats.** 1 rep per variant -- the finding is qualitative (a flat span pattern
and a bank-specific-but-minimal conflict), robust to rep noise; 3-rep pooling is
a formality to add with the EMU comparison. **Column-virtualization offset
confirmed:** the trace reports the memtile at col 1 for a declared `aie.tile(0,1)`
(physical column offset, cf. the Phoenix col-0 virtualization); harmless here --
the measure keys on the `"memtile"` process-name prefix, not `(col,row)`.
