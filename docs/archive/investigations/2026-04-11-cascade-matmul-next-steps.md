# Cascade Matmul: Next Steps After JNZD Fix

**Date**: 2026-04-11
**Predecessor**: `2026-04-11-cascade-matmul-timeout.md`
**Status**: TIMEOUT resolved, data error remains

## What Was Fixed

The `jnzd` instruction with split dest/source registers (`jnzd r20, r24, p1`)
was decrementing the wrong value. Our emulator computed `dest = dest - 1` when
the correct AIE2 behavior is `dest = source - 1`. The JNZD encoding has
separate 5-bit fields for mRx (output) and mRx0 (input) in the ALU word.

The Chess compiler generates split-register JNZD on intermediate cascade tiles
(core_1_2, core_2_2), where the middle loop counter is stack-bridged: r20 is
saved at loop top, loaded into r24 at loop bottom, and JNZD writes
`r20 = r24 - 1`. Peano always ties the operands to the same register, so the
bug only manifested with Chess-compiled cascade tests.

Fix: `control.rs` NotZeroDecrement handler, one-line change.

## Corrected Understanding

Several assumptions from the timeout investigation were wrong:

1. **The "buffer" variant does NOT use cascade (vmov MCD/SCD).** The main loop
   is fully inlined scalar computation. The matmul function at 0x530 (which
   contains vmov MCD) is dead code -- compiled into the ELF but never called.

2. **The objectFifo protocol is producer-only on each tile's main loop.** Each
   tile acquires its own lock 4 (acq_ge 1, buffer free) and releases its own
   lock 5 (rel +1, data ready). The consumer side (acquire neighbor lock 5,
   release neighbor lock 4) runs on the NEXT tile.

3. **Register setup for lock acquire/release happens in jl delay slot 5.** The
   function call delay slots set r0/r1 for the target function. Our delay slot
   engine (5 slots, BRANCH_DELAY_INITIAL=6) handles this correctly.

## Current State

Test: `matrix_multiplication_using_cascade/buffer` (Chess)
- HW: PASS (1.4ms)
- EMU: FAIL (data error, completes in ~7.5s)
- 159 incorrect cells out of 256 (16x16 matrix)

The output shows a repeating 4-value pattern in rows 1-15:
```
-982918320, 1573445287, -1758825043, -381086091
```

Row 0 has partially correct values (some cells match, some don't). This
repeating pattern suggests the pipeline produces one partial result set but
fails to accumulate correctly across the 4-tile chain.

## What to Investigate Next

### Priority 1: ObjectFifo buffer data flow

The objectFifo buffer is at address 0x70a00 (per `movxm p1, #0x70a00`). Each
tile stores its partial matmul result to this address after each lock 4/5
cycle. The next tile loads from this buffer via the consumer objectFifo
protocol.

Check:
- Is address 0x70a00 correctly mapped in the emulator's memory model?
- Where does core_1_2 LOAD the previous tile's partial result? (It should
  acquire West lock 5, then load from tile(0,2)'s objectFifo buffer.)
- Is the address translation for West-neighbor memory correct?

### Priority 2: Cross-tile memory visibility

The objectFifo buffer lives in tile(N)'s memory. The consumer on tile(N+1)
accesses it via the West memory quadrant. For compute tiles at row 2:
- Own memory: 0x60000-0x6FFFF (East/Internal)
- West memory: 0x10000-0x1FFFF (col-1 neighbor)

If the consumer on core_1_2 reads from West address `0x10a00` (mapping
tile(0,2)'s 0x0a00), the emulator needs to correctly route this to
tile(0,2)'s data memory.

### Priority 3: Partial result accumulation

The matmul is distributed: each tile computes a partial sum and the results
should be accumulated. Check whether:
- The first tile's output is a full matmul or a partial result
- Intermediate tiles add their local computation to the incoming partial
- The final tile (core_3_2) outputs the complete result

### Priority 4: Other affected tests

The investigation notes cascade/plain and cascade/cascade also fail with data
errors. After fixing the buffer variant, check whether those share the same
root cause or have additional issues. The cascade/cascade variant DOES use
vmov MCD/SCD, so cascade data path correctness matters there.

## Key Disassembly References

All ELFs are at:
```
mlir-aie/build/test/npu-xrt/matrix_multiplication_using_cascade/chess/
    aie_bufferx4.mlir.prj/segment_0_core_{0,1,2,3}_2.elf
```

Core_0_2 (first tile, 31KB): producer only, same-register jnzd
Core_1_2 (intermediate, 33KB): consumer + producer, split-register jnzd
Core_2_2 (intermediate, 33KB): same as core_1_2
Core_3_2 (last tile, 32KB): consumer, outputs via DMA MM2S

Lowered MLIR: `chess/aie_bufferx4.mlir.prj/input_with_addresses.mlir`

## Build/Test Commands

```bash
# Build FFI lib
cargo build -p xdna-emu-ffi

# Run the test (EMU)
cd mlir-aie/build/test/npu-xrt/matrix_multiplication_using_cascade/chess
XDNA_EMU=debug RUST_LOG=warn timeout 30 ./test.exe -x aie2_buffer.xclbin -k MLIR_AIE -i insts2_buffer.txt

# Run on HW for comparison
timeout 15 ./test.exe -x aie2_buffer.xclbin -k MLIR_AIE -i insts2_buffer.txt

# Disassemble any core
llvm-objdump -d aie_bufferx4.mlir.prj/segment_0_core_1_2.elf
```
