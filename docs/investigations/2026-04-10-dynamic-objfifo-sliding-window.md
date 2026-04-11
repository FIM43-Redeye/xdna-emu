# Dynamic Object FIFO Sliding Window Investigation

**Date**: 2026-04-10
**Status**: Root cause identified, fix implemented (2026-04-11)
**Affected tests**: 5 of 6 dynamic_object_fifo bridge tests

## Summary

Five `dynamic_object_fifo` bridge tests fail on EMU (both Chess and Peano)
while passing on real HW. The root cause is a **shim DMA S2MM TLAST handling
bug** -- `transfer_stream_to_host()` and `transfer_s2mm()` unconditionally
break on TLAST even when Finish-on-TLAST (FoT) mode is disabled, causing
the address pointer to skip ahead when TLAST falls mid-batch.

Fix: gate the TLAST break on `fot_mode != 0` in both functions.

## Results After Fix (2026-04-11)

| Test | Chess EMU (before) | Chess EMU (after) | Peano EMU (before) | Peano EMU (after) |
|------|--------------------|--------------------|--------------------|--------------------|
| nested_loops | FAIL | PASS | FAIL | PASS |
| ping_pong | PASS | PASS | PASS | PASS |
| reduction | FAIL | PASS | FAIL | PASS |
| sliding_window | FAIL | FAIL* | FAIL | PASS |
| sliding_window_conditional | FAIL | FAIL* | FAIL | PASS |
| two_core_sliding_window | FAIL | FAIL* | FAIL | FAIL* |

*Remaining failures show a DIFFERENT pattern: only the first element of
some rows is wrong (off by 1-2), not the systematic address-gap pattern.
This is a separate bug, likely a core/DMA timing race in the sliding
window acquire pattern. The original stride bug is fully resolved.

`ping_pong` was unaffected because it uses simple acquire-1/release-1
with depth-2 FIFO. The original failing tests use `acquire(Consume, 2)`
with depth-3 FIFO (sliding window pattern).

## Symptom

Output data is shifted and contains 0-gaps. Example from `sliding_window`:

```
row 1: expected 1, got: 0, 0, 1, 1, 1, 1, 1, 1, 1, 1
row 2: expected 3, got: 1, 1, 0, 0, 5, 3, 3, 3, 3, 3
row 3: expected 5, got: 3, 3, 3, 3, 0, 0, 7, 5, 5, 5
```

The 0-gaps shift by 2 positions each row. Previous row's values appear in
the first positions. Data from a future row occasionally appears (e.g., 5
where 3 is expected in row 2 = 2+3, which is row 3's computation).

## Root Cause: Shim DMA Address Stride Bug

### Evidence

The shim S2MM DMA is configured with a single BD transferring 400 bytes
(100 x i32) to host memory at `0x800000003000`. This should produce 100
consecutive 4-byte writes at offsets 0, 4, 8, ..., 396.

**Actual write addresses observed** (from RUST_LOG debug output):

```
Elements written: 0-9, 12-21, 24-33, 36-45, 48-57, 60-69, 72-81, 84-93, 96-99
Elements MISSING: 10-11, 22-23, 34-35, 46-47, 58-59, 70-71, 82-83, 94-95
```

Only **84 of 100 words** are written. The write address jumps by 48 bytes
(12 words) every 40 bytes (10 words). The 8-byte (2-word) gap per chunk
matches the 0-gaps in the test output exactly.

### Why 12 instead of 10?

The compute tile's MM2S DMA sends data through the stream switch in 10-word
chunks (one output buffer = `memref<10xi32>` = 40 bytes). The shim S2MM
receives these words and writes them to DDR.

The shim S2MM BD is configured as a flat 400-byte linear transfer -- no
multi-dimensional addressing. The write pointer should simply increment by
4 bytes per word. But something causes it to advance by an extra 8 bytes
at every 10-word boundary.

### Where to look

The bug is in the shim DMA S2MM transfer path. Likely candidates:

1. **`src/device/dma/engine/stepping.rs`** -- `do_transfer_cycle` for S2MM
   on shim tiles. Check how the write address is computed for each word.

2. **`src/device/dma/transfer/`** -- Transfer state tracking. The
   `remaining_bytes` and address pointer logic.

3. **Stream-to-memory address calculation** -- The shim S2MM receives words
   from the stream switch and writes them to host memory. If the transfer
   tracks progress based on the *compute tile's* BD dimensions rather than
   the *shim's* BD dimensions, the address stride from the source BD
   (which has dimensional addressing for 10-element buffers) could leak
   into the shim's flat write.

### Confirmed Root Cause

`transfer_stream_to_host()` and `transfer_s2mm()` unconditionally
`break` on TLAST, even when Finish-on-TLAST (FoT) mode is disabled.
The caller (`do_transfer_cycle`) then advances the address generator by
the full `bytes_to_transfer` (batch size = 4 words = 16 bytes), not the
actual bytes written.

The compute tile MM2S sends 10-word chunks with TLAST on word 10. The
shim S2MM transfers 4 words per cycle (128-bit DMA bus). When TLAST
falls at word 10 (mid-batch for the 3rd 4-word cycle), the S2MM writes
only 2 words but the address pointer skips 4 word positions -- leaving
a 2-word gap.

On real AIE2 hardware, S2MM ignores TLAST unless FoT mode is configured
on the channel. The unconditional break was incorrect.

**Fix**: Gate the TLAST break on `fot_mode != 0` in both
`transfer_stream_to_host()` and `transfer_s2mm()` in
`src/device/dma/engine/stepping.rs`.

## Test Setup (for reproducing)

```bash
cd /home/triple/npu-work/mlir-aie/build/test/npu-xrt/dynamic_object_fifo/sliding_window/chess
XDNA_EMU=debug RUST_LOG=xdna_emu::device::dma::engine::stepping=debug \
  ./test.exe 2>/tmp/sliding_window_debug.log
```

Key log patterns to grep:
- `Shim S2MM write:` -- shows actual host memory write addresses and values
- `DMA(0,2)` -- compute tile DMA state transitions
- `DMA(0,0)` -- shim tile DMA state transitions

## Architecture of the Test

### MLIR source

```
aie.objectfifo @in(ShimTile, ComputeTile, 3, memref<10xi32>)   -- depth 3
aie.objectfifo @out(ComputeTile, ShimTile, 2, memref<10xi32>)  -- depth 2
```

### Lock protocol (tile 0,2)

| Lock | Init | Used by | Semantics |
|------|------|---------|-----------|
| lock 0 (`in_cons_prod_lock_0`) | 3 | DMA S2MM acq(-1), Core rel(+1) | Free input buffers |
| lock 1 (`in_cons_cons_lock_0`) | 0 | DMA S2MM rel(+1), Core acq(-1) | Filled input buffers |
| lock 2 (`out_prod_lock_0`) | 2 | Core acq(-1), DMA MM2S rel(+1) | Free output buffers |
| lock 3 (`out_cons_lock_0`) | 0 | Core rel(+1), DMA MM2S acq(-1) | Filled output buffers |

### DMA BDs (tile 0,2)

- S2MM ch0: BD0 (buff_0, 40B) -> BD1 (buff_1, 40B) -> BD2 (buff_2, 40B) -> BD0 (loop)
- MM2S ch2: BD3 (out_buff_0, 40B) -> BD4 (out_buff_1, 40B) -> BD3 (loop)

### DMA BDs (shim tile 0,0)

- S2MM ch0: BD0 (host output, 400B, linear, no locks)
- MM2S ch2: BD1 (host input, 400B, linear, no locks)

### Core sliding window logic

1. Acquire 1 input buffer, process `add_10_i32(buff[i], buff[i], out)` (first row)
2. Loop 8x: Acquire 1 more input buffer (now holding 2), process `add_10_i32(buff[i], buff[i+1], out)`, release 1
3. Acquire 1 more (holding 2), process, release 2 (final)

The "acquire 2" in the IRON API is lowered to a single `use_lock(cons_lock, AcquireGreaterEqual, 1)` per iteration, with software index tracking via `_anonymous0` buffer.

## Initial Misdirection

### Core/DMA ordering theory (RULED OUT)

Initially suspected the coordinator's execution order (cores step before
DMA in each cycle) was causing the core to read stale buffer contents.
Investigation revealed:

1. The coordinator comment says "DMA steps before cores" but the code does
   cores (Phase 2) then DMA (Phase 3). The comment is stale/wrong.

2. However, this doesn't cause the bug because DMA data writes complete
   in previous cycles before the lock is released. By the time the core
   acquires the lock, the data is fully written.

3. The AM020 docs confirm core and DMA run truly in parallel on the same
   clock with per-bank arbiters. Our sequential stepping is functionally
   correct for the lock-protected producer/consumer pattern.

### Core/DMA parallelism (DEFERRED)

The coordinator's sequential core-then-DMA stepping is architecturally
incorrect (should be parallel with bank arbitration), but this is a
separate concern from the current bug. The per-bank arbiter model would
improve cycle-accuracy and may be needed for future memory-conflict-heavy
workloads, but is not the cause of the sliding_window failures.

**TODO (separate task)**: Fix coordinator comment to match code, or
refactor to true parallel model with per-bank arbiters.

## Also Discovered

### Regression: add_21_i8_using_dma_op_with_padding

Previously passing, now fails (both compilers). Data at wrong offsets,
similar shifted pattern. Possibly related to the same shim DMA stride bug
since it also involves padded DMA transfers. Worth checking after fixing
the sliding window issue.

### New test: packet_flow_fanin.peano

Previously failing, now passes. Likely fixed by the Peano toolchain update
(we updated llvm-aie/mlir-aie to latest upstream).

### Stale coordinator comment

`coordinator.rs:509-515` says "DMA steps before cores" but the code does
the opposite. Either fix the comment or (better) refactor to match the
documented design.
