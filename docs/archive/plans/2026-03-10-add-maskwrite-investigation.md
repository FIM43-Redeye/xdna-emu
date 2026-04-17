# add_maskwrite Investigation - 2026-03-10

## Status: IN PROGRESS (3 bugs found, 2 fixed, 1 remaining)

## Test Description

`add_maskwrite` tests NPU maskwrite32 instructions that modify tile data
memory at runtime. The core initializes input_buffer with 0x37373737, then
the host maskwrites two words before releasing a lock. The core reads the
modified values, adds 1, and DMAs the result to host.

HW: PASS. EMU: FAIL.

## Bug 1: NPU Instruction Timing (FIXED in this session)

**Root cause**: The emulator's run loop interleaved NPU instructions 1:1
with core cycles. On real hardware, the core runs for thousands of cycles
(CDO enables it) before NPU instructions arrive through firmware + NoC.
Maskwrites fired before the core's init loop completed.

**Fix**: Added warm-up phase in `xdna_emu_run()` (ffi/mod.rs) that steps
the engine until `all_cores_blocked()` before processing NPU instructions.
Added `all_cores_blocked()` to InterpreterEngine (coordinator.rs).

**Files changed**:
- `src/ffi/mod.rs` -- warm-up loop before main interleaved loop
- `src/interpreter/engine/coordinator.rs` -- `all_cores_blocked()` method

## Bug 2: PADDB dest=None (FIXED in this session)

**Root cause**: The decoder produces PADDB (pointer add) instructions with
`dest=None` and `srcs=[PointerReg(N), Immediate(offset)]`. The destination
pointer register is implicit (tied operand -- it's the same as the source).
But `execute_pointer_add` routed `dest=None` to the SP-only path, so the
actual pointer register was never modified. This caused p6 to retain its
initial value through the entire prologue, corrupting all subsequent pointer
arithmetic.

**Evidence**: The compute loop loaded from offset 0x410 (element 4) at the
first LDA instruction, instead of offset 0x400 (element 0). Traced back to
r22 = 0x70410 (should be 0x70408), caused by the first PADDB [p6], #-8 at
address 250 not modifying p6.

**Fix**: In `execute_pointer_add` (semantic.rs), when dest=None, infer the
destination from the first PointerReg source operand. Only fall back to SP
when no pointer source is found.

**File changed**: `src/interpreter/execute/semantic.rs`

**Decoder TODO**: The proper fix is in the decoder -- set dest = first
PointerReg source when the encoding has `is_ptr_arithmetic=true` and
dest=None. The semantic.rs fix is a workaround.

## Bug 3: Output Buffer All Zeros (CURRENT -- not yet diagnosed)

After fixing bugs 1 and 2, the test output changed from partial failures
to ALL outputs being `1` (= 0 + 1, meaning output_buffer is zero).
Expected values are now correct in the reference (17375778, 3a3c3e31,
37373738...) but the emulator produces all 1s.

This suggests either:
- The core's stores to output_buffer use wrong addresses (PADDB fix may
  have changed pointer arithmetic in unexpected ways)
- The DMA reads from wrong addresses
- The warm-up phase runs too many iterations

**Next steps**:
1. Check if PADDB fix causes second PADDB to double-decrement (latency
   interaction -- queue_pointer_write uses latency 1, but the old code
   was writing immediately via apply_post_modify)
2. Verify the core's store addresses for output_buffer
3. Check if other tests regress with the PADDB fix

## Infrastructure Fixes (also done this session)

### Debug Build Workflow
- `rebuild-plugin.sh` now defaults to debug builds (10x faster: 4s vs 2min)
- `--release` flag available when needed
- Script always rebuilds Rust lib + C++ plugin + installs

### XDNA_EMU_DIR / Profile Selection
- `activate-npu-env.sh` now sets `XDNA_EMU_DIR` instead of `XDNA_EMU_LIB`
- Plugin resolves `$XDNA_EMU_DIR/target/{debug|release}/libxdna_emu.so`
  based on the `XDNA_EMU` env var value:
  - `XDNA_EMU=debug` (or `1` or any truthy) -> debug lib
  - `XDNA_EMU=release` -> release lib
  - `XDNA_EMU_LIB` still works as explicit override
- Bridge test defaults to `XDNA_EMU=debug`
- Plugin logs which library it loads

**Files changed**:
- `scripts/rebuild-plugin.sh`
- `scripts/emu-bridge-test.sh`
- `xrt-plugin/src/pdev_emu.cpp`
- `toolchain-build/activate-npu-env.sh` (in npu-work, not xdna-emu)

### Bridge Test Caching Fix
- Tests no longer recompile every run (was copying MLIR unconditionally,
  invalidating timestamp cache). Now conditional on source changes.
- **File changed**: `scripts/emu-bridge-test.sh` (by background agent)

## Key Debugging Insight

The `XDNA_EMU_LIB` env var from activate-npu-env.sh was pointing to the
release lib, overriding our debug builds. We spent significant time adding
debug instrumentation that never appeared in output because the wrong .so
was loaded. The LD_DEBUG=libs trace revealed this:
```
calling init: /home/triple/npu-work/xdna-emu/target/release/libxdna_emu.so
```
The new XDNA_EMU_DIR mechanism prevents this class of issue.
