# Cascade Matmul Timeout Investigation

**Date**: 2026-04-11
**Test**: `matrix_multiplication_using_cascade/buffer` (Chess compiler)
**Status**: HW=PASS (1.4ms), EMU=TIMEOUT
**Also affected**: `cascade/plain` (FAIL/data error), `cascade/cascade` (FAIL/data error)

## Summary

The cascade matmul test times out in the emulator but passes on real hardware.
The root cause is NOT a simple lock deadlock -- the cross-tile objectFifo lock
recycling works correctly for ~271 cycles before the system enters a livelock
where cores execute but never complete the matmul function.

## Test Architecture

4 compute tiles in a cascade chain:
```
tile(0,2) --cascade--> tile(1,2) --cascade--> tile(2,2) --cascade--> tile(3,2)
```

Each tile receives A/B matrix data via DMA (S2MM), computes a partial matmul,
sends results downstream via cascade (`vmov MCD`), and the final tile sends
results back via DMA (MM2S) through memtiles to the shim.

**MLIR source**: `mlir-aie/test/npu-xrt/matrix_multiplication_using_cascade/aie_cascadex4.mlir`
**Kernel source**: `mlir-aie/test/npu-xrt/matrix_multiplication_using_cascade/mm.cc`
**Lowered MLIR**: `chess/aie_bufferx4.mlir.prj/input_with_addresses.mlir`
**LLVM IR**: `chess/aie_bufferx4.mlir.prj/segment_0_core_0_2.ll`
**ELF disasm**: saved at `~/.claude/projects/.../tool-results/b9lg8qeit.txt`

## Lock Layout (per compute tile)

MLIR defines locks 0-3 for data DMA synchronization:
```
Lock 0 (init=0): DMA S2MM0 releases -> core acquires (data A ready)
Lock 1 (init=1): core releases -> DMA S2MM0 acquires (buffer A free)
Lock 2 (init=0): DMA S2MM1 releases -> core acquires (data B ready)
Lock 3 (init=1): core releases -> DMA S2MM1 acquires (buffer B free)
```

objectFifo lowering adds locks 4-5 for cross-tile cascade staging:
```
Lock 4 (init=1): of0_prod_lock_0 -- "buffer available for writing"
Lock 5 (init=0): of0_cons_lock_0 -- "data ready for reading"
```

Lock 4 init=1 comes from CDO (confirmed in raw binary at word 320 of
`segment_0_aie_cdo_init.bin`, address 0x0021F040, value 0x00000001).

## ObjectFifo Cross-Tile Lock Protocol

The lowered MLIR (`input_with_addresses.mlir`) has BOTH sides:

**Producer (bb24, on tile N):**
```mlir
acquire of0_prod_lock_0 (lock 4, AcquireGreaterEqual 1)
store result to of0_buff_0 (1xi32 buffer at address 0xA00)
release of0_cons_lock_0 (lock 5, Release +1)
```

**Consumer (bb21, on tile N+1):**
```mlir
acquire of0_cons_lock_0 (West neighbor lock 5, AcquireGreaterEqual 1)
load from of0_buff_0
release of0_prod_lock_0 (West neighbor lock 4, Release +1)
```

The consumer on tile(1,2) reads tile(0,2)'s lock 5 and releases tile(0,2)'s
lock 4, creating a cross-tile ping-pong.

**CRITICAL**: The LLVM IR (`segment_0_core_0_2.ll`) only has the PRODUCER side:
```
Line 173: call void @llvm.aie2.acquire(i32 52, i32 -1)  ; lock 4
Line 180: store i32 %77, ptr @of0_buff_0
Line 181: call void @llvm.aie2.release(i32 53, i32 1)   ; lock 5
```

The consumer side (acquire lock 5, release lock 4) was removed during
MLIR->LLVM lowering because the cascade `put_mcd` was inlined as
`vmov MCD, bml0` / `vmov MCD, bmh0` directly -- the intermediate
objectFifo buffer load became dead code. However, the CONSUMER side
DOES exist on the NEXT tile (tile N+1 has the consumer for tile N's
objectFifo). This is confirmed by the debug log showing successful
cross-tile lock operations (raw=21 = West lock 5, raw=20 = West lock 4).

## ELF Disassembly Key Points

Core_0_2 main loop (`core_0_2` at 0xe0):
```
0xf0:  jl #0x4f0 (acquire), r0=0x30(48), r1=-1  -> ACQUIRE lock 0 (data A)
0x110: jl #0x4f0 (acquire), r0=0x32(50), r1=-1  -> ACQUIRE lock 2 (data B)
0x130: jl #0x1690                                -> event_0()
0x140: jl #0x1760                                -> zero C buffer
0x160: jl #0x16a0                                -> event_1()

[inner computation loop with 4 unrolled iterations, each containing:]
  jl #0x4f0 (acquire), r0=0x34(52), r1=-1       -> ACQUIRE lock 4 (objFifo)
  [matmul computation + store result]
  jl #0x510 (release), r0=0x35(53), r1=1        -> RELEASE lock 5 (objFifo)

[3 nested jnzd loops control the 4x4x4x1 iteration space]

0x496: jl #0x510 (release), r0=0x31(49), r1=1   -> RELEASE lock 1 (data A)
0x4b0: jl #0x510 (release), r0=0x33(51), r1=1   -> RELEASE lock 3 (data B)
0x4d0: jl #0x1580                                -> flush_trace()
0x4e0: j  #0xf0                                  -> loop back to top
```

Helper functions:
```
0x4f0: llvm___aie2___acquire:  acq r0, r1; ret lr
0x510: llvm___aie2___release:  rel r0, r1; ret lr
```

The matmul function at 0x530 (`matmul_scalar_cascade_i32_i32<true,false,...>`)
contains `vmov MCD, bml0` and `vmov MCD, bmh0` for cascade output.

## What Actually Happens (Emulator)

### Cascade directions (confirmed via CDO debug log):
```
Tile (0,2): input=North, output=East
Tile (1,2): input=West,  output=East
Tile (2,2): input=West,  output=East
Tile (3,2): input=West,  output=South
```

### Observed behavior:
1. DMA delivers A/B data to all tiles successfully
2. Cores acquire data locks 0/2 (success after DMA releases them)
3. Cores enter matmul inner loop
4. Lock 4/5 objectFifo cycling works:
   - tile(0,2) acquires lock4 (own), stores, releases lock5 (own, deferred)
   - tile(1,2) acquires lock5 (west=tile0,2), loads, releases lock4 (west=tile0,2)
   - Lock4 returns to 1, cycle repeats
5. **271 total successful lock 4 acquires** across all tiles (~68 per tile)
6. **Zero cascade operations** (vmov MCD/SCD never executed)
7. **Zero data lock releases** (locks 1/3 never released)
8. Cores never finish the matmul function -> never reach data lock releases
9. DMA starves waiting for locks 1/3 -> no data output -> test times out

### Key numbers:
- Expected objectFifo puts per tile per matmul call: 4*4*1*4*4 = 256
- Actual per tile: ~68 (26% of one iteration)
- DMA repeat_count=1 (2 total BD executions needed)
- Total expected: 256 * 2 = 512 puts per tile
- cascade_flows test (simple, no matmul): PASSES with 14 cascade ops

## Disproven Hypotheses

### 1. "Lock 4 can never be recycled" -- DISPROVEN
Cross-tile lock recycling works. tile(N+1) acquires tile(N)'s lock 5 and
releases tile(N)'s lock 4. Confirmed by 271 successful lock 4 acquires.

### 2. "acq instruction is unconditional (no stall)" -- DISPROVEN
Patched `acquire_with_value` to always succeed (no stall check, allow negative
lock values). Test STILL timed out. Lock stalling is not the only issue.

### 3. "Trace lock issue" -- DISPROVEN
Lock 4/5 are NOT trace locks. They're objectFifo producer/consumer locks
added by mlir-aie's objectFifo lowering pass. The name `of0_prod_lock_0`
in the lowered MLIR confirms this.

### 4. "Cascade data path is broken" -- DISPROVEN
cascade_flows test passes with 14 cascade operations. The cascade routing
in `src/device/array/routing.rs` works correctly.

## Remaining Hypothesis: Matmul Loop Never Terminates

The cores execute ~68 objectFifo lock acquire/release pairs per tile but
never exit the matmul inner loop. The loop is controlled by 3 nested `jnzd`
instructions (at 0x430, 0x456, 0x482) that count down from 3.

**Possible causes:**
1. **Loop counter bug**: Our `jnzd` (jump-nonzero-decrement) implementation
   may not correctly decrement the counter, causing infinite iteration.
2. **Computation error**: The unrolled matmul with interleaved lock ops may
   produce wrong values that affect loop control flow.
3. **Instruction interleaving**: The 4 unrolled iterations per loop body
   each have acquire-lock4 + compute + release-lock5. If any step fails
   silently (e.g., store to wrong address), the loop counter may not advance.

## Next Steps

### Priority 1: Check jnzd loop behavior
The `jnzd` instruction decrements a register and jumps if non-zero. With
3 nested loops (counters initialized to 3, meaning 4 iterations each =
4*4 = 16 iterations of the inner body), we expect 16 iterations of the
inner body which has 4 unrolled sub-iterations = 64 objectFifo acquires
per outer matmul call. 68 acquires per tile is suspiciously close to 64+4.

Verify:
- Does `jnzd` correctly decrement its counter register?
- Are the loop counter registers (r21, r23, r24) being clobbered by the
  matmul computation?
- Add logging to jnzd to see the actual counter values.

### Priority 2: Trace a single core's instruction flow
Add PC logging (info level) for tile(0,2) to see if the core visits the
same instructions repeatedly or makes forward progress through the function.
This would immediately show if the loop is infinite or if the core gets
stuck at a specific point.

### Priority 3: aiesimulator as reference
We compiled with --aiesim (`/tmp/claude-1000/cascade_sim/`) but the ps.so
needs a ps_main function that feeds data to the simulation. NPU-style tests
with runtime_sequence need custom ps_main generation. This is complex but
would give cycle-accurate reference behavior.

## Coordinator Stepping Model

Current model (coordinator.rs):
```
Phase 1: Sync DMA start requests
Phase 2: Step each core (col-major: 0,2 -> 1,2 -> 2,2 -> 3,2)
  - Clone neighbor locks BEFORE stepping
  - Execute ONE instruction bundle per core
  - Own-tile lock releases are DEFERRED to arbiter
  - Cross-tile lock releases via writeback_locks -> defer_core_lock_release
Phase 3: step_data_movement
  - Resolve all tile arbiters (applies deferred lock releases)
  - Step DMA channels
  - Route streams
  - Route cascade (Phase 4.5)
```

This gives correct 1-cycle latency for lock visibility across tiles.
The sequential stepping within Phase 2 means core(0,2) is always stepped
before core(1,2) within the same cycle, which may create systematic timing
bias, but the lock recycling handles this correctly.

## Files Modified During Investigation (all reverted)

- `src/device/tile/locks.rs` -- temporarily removed stall check (reverted)
- `src/interpreter/execute/cascade.rs` -- upgraded log levels to info (KEEP)
- `src/device/array/routing.rs` -- upgraded cascade route log to info (KEEP)

## Logging Changes (to keep)

Cascade operation and routing logs upgraded from debug to info:
- `[CASCADE] Read stall/complete` in cascade.rs
- `[CASCADE] Write stall/complete` in cascade.rs
- `[CASCADE] Route (src) -> (dst)` in routing.rs

These use `log::info!` which is compiled into all builds per our
`release_max_level_info` setting. They will be visible with RUST_LOG=info.

## Build/Test Commands

```bash
# Build emulator FFI lib
cd /home/triple/npu-work/xdna-emu && cargo build -p xdna-emu-ffi

# Run the failing test
cd /home/triple/npu-work/mlir-aie/build/test/npu-xrt/matrix_multiplication_using_cascade/chess
XDNA_EMU=debug RUST_LOG=info timeout 15 ./test.exe -x aie2_buffer.xclbin -k MLIR_AIE -i insts2_buffer.txt

# Run the passing cascade test for comparison
cd /home/triple/npu-work/mlir-aie/build/test/npu-xrt/cascade_flows/chess
XDNA_EMU=debug RUST_LOG=info timeout 15 ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin

# Run on real hardware (passes)
cd /home/triple/npu-work/mlir-aie/build/test/npu-xrt/matrix_multiplication_using_cascade/chess
timeout 15 ./test.exe -x aie2_buffer.xclbin -k MLIR_AIE -i insts2_buffer.txt

# Disassemble core ELF
/home/triple/npu-work/llvm-aie/install/bin/llvm-objdump -d \
  chess/aie_bufferx4.mlir.prj/segment_0_core_0_2.elf
```

## Bridge Test Results (2026-04-11)

```
ISA:    4815/4815 (100%)
Unit:   2783/2783

Chess: 62/62 compiled, 57 bridge pass, 7 bridge fail, 1 timeout
  HW: 64 pass, 0 fail
Peano: 55/55 compiled, 44 bridge pass, 10 bridge fail, 2 timeout, 1 xfail
  HW: 51 pass, 3 fail, 1 xfail
```
