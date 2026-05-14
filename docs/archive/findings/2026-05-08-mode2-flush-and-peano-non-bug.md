# Mode-2 trace empty-BO is mode-2 specific, not Peano-specific

**Status:** Phase 0's "Peano-traced builds produce empty trace BO"
finding (2026-04-30) is corrected here. Peano is fine; mode-2 has an
inherent threshold that small kernels don't cross under any compiler.

## Empirical matrix

Same `runtime_loop` fixture (single-pass), same harness
(`bridge-trace-runner` against the traced xclbin), same `N=8` input.
Captured trace.bin sizes (non-zero bytes / total):

| Trace mode | Compiler | Single-pass | 4-pass wrapper | 64-pass wrapper |
|------------|----------|-------------|----------------|-----------------|
| 0: event_time (cycle deltas + slot events) | Peano | **64** | -- | -- |
| 1: event_pc   (PCs + slot events)          | Peano | **128** | -- | -- |
| 2: inst_exec  (atoms + branches + LC)      | Peano | **0** | 64 | 576 |
| 2: inst_exec  (atoms + branches + LC)      | Chess | **0** | 64 | (Phase 0: 576) |

The Phase 0 doc claimed "Peano-traced builds produce empty trace BOs
regardless of mode." That's not what's happening:

- **Peano single-pass + mode 0/1 produces real trace bytes.** The
  toolchain wiring is fine.
- **Mode 2 single-pass produces 0 bytes on BOTH compilers.** Phase 0
  only tested with `--mode2`, so the all-empty-on-Peano observation
  was real but mis-attributed.

## Why mode 2 fails specifically for tiny kernels

Modes 0 and 1 record one trace frame per slot-event firing. Each
event = a few bytes of frame data, plus periodic Sync (~1 byte) and
delta encoding. Even a tiny kernel generates several events
(STREAM_STALL while waiting on objectfifo, INSTR_VECTOR on the
first vector op, etc.); 8 + events crosses the 28-byte byte-buffer
threshold and the trace controller emits one or more 32-byte packets
to the shim DMA, which writes them to DDR.

Mode 2 records core execution at cycle granularity:
- **E_atom / N_atom per cycle** (1-4 bits each, RLE-compressed)
- **New_PC on branch** (16 bits)
- **LC frame on ZOL start** (32 bits, once per ZOL invocation)

Slot events are *not* recorded in mode 2. A single-pass kernel that
runs ~30 instructions in ~50 cycles generates maybe 30 atom-bits +
2 New_PC frames + 1 LC frame = around 12-20 bytes. Below the 28-byte
trace-controller packet threshold, the partial buffer is held
internally, and on `Stop_Event` the bytes are discarded -- there is
no flush register in the trace control register set.

## Tested HW workarounds that did NOT work

1. **Fire INSTR_EVENT_0 (id=33) via Event_Generate writes between
   `dma_wait` and the stop broadcast.** Patched 16, 64, 192 fires
   into the runtime_sequence post-lowering MLIR; rebuilt insts.bin
   and reran. 0 bytes captured every time. Mode-2 doesn't record
   slot events, so writing to `EVENT_GENERATE` (0x34008) on the core
   tile does nothing observable in mode-2's output.

2. **Lower the trace BD `buffer_length`** (currently 2048 words /
   8KB). Per user feedback this would degrade the normal multi-pass
   path, where 8KB transfers are more efficient. Not pursued.

3. **`XAie_DmaChannelReset` on the shim trace channel.** Returns
   `XAIE_INVALID_TILE` for shim tiles -- aie-rt doesn't expose this.

## What actually fixes the symptom

For mode-2 specifically, the only path is **more kernel activity**.
The `heavy_zol` 64-pass wrapper is the canonical pattern; any fixture
that wraps a small ZOL in N >= 4 outer passes generates enough atoms
to cross the threshold. `tools/mode2_capture_fixtures/lc_overflow_probe`
takes a runtime-controlled wrapper count via `in[1]` for exactly this
reason.

For modes 0 and 1, no fix is needed -- they work on single-pass tiny
kernels out of the box.

## Practical implications

- The default trace mode in `mlir-trace-inject.py` and
  `build_fixture.sh` is now `event_pc` (mode 1). Most users won't
  hit the mode-2 threshold issue.
- `--mode2` is opt-in for the LC / atom / branch-level probes that
  inherently need it. Those should always use a multi-pass wrapper.
- We should NOT spend further effort trying to "flush" mode-2 from
  outside the kernel. There's no HW mechanism, and the alternative
  workarounds (lower buffer_length, lower burst_length) compromise
  the normal high-volume capture path.

## Code / doc changes from this finding

- `docs/superpowers/findings/2026-04-30-mode2-lc-flag-semantics.md`
  -- prepended a correction pointer to this doc on the "Why Peano
  builds didn't trace" section.
- This doc.

No emulator or mlir-aie source changes are needed: the previous
emitter rule (mode-2 single-pass = empty trace) is correct HW
behavior, just mis-attributed in the original write-up.
