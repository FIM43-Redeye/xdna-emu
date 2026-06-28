# Bug Report: VEXTRACT with Element Index 0 Permanently Stalls AIE2 Core

## Environment

- **Hardware**: AMD Ryzen 7 7840U (Phoenix, NPU1)
- **Architecture**: AIE2 (XDNA)
- **Firmware**: 1.5.5.391
- **Driver**: xdna-driver (commit 0c6249f, 2026-03-24)
- **XRT**: 2.23.0 (481583db9)
- **Kernel**: Linux 7.0.0-rc5 (custom build with out-of-tree amdxdna module)
- **Compiler**: Peano (llvm-aie, commit matching mlir-aie 2026-03-24 build)

## Summary

`vextract.d16`, `vextract.s16`, and `vextract.d32` instructions permanently
stall the AIE2 compute core when ALL of the following conditions are met:

1. The source vector register was written by `vlda`
2. The element index register contains value 0
3. The full vlda pipeline latency (7+ cycles) has elapsed

The core enters a permanent stall state (xrt-smi reports Status: 8 / timeout).
It does not recover. The TDR eventually fires and kills the context.

Indices 1 through 15 work correctly under identical conditions.

## Minimal Reproducer (HANGS)

```asm
.text
.globl test_kernel
test_kernel:
    // p0 = input buffer (64 bytes of test data)
    // p1 = output buffer (4 bytes)
    vlda wl0, [p0, #0x0]       // load low 256 bits into x0
    vlda wh0, [p0, #0x20]      // load high 256 bits into x0
    mov r16, #0                 // element index = 0
    nop
    nop
    nop
    nop
    nop
    nop
    nop                         // 7 nops -- exceeds vlda latency
    vextract.d16 r0, x0, r16   // *** HANGS HERE ***
    st.d16 r0, [p1], #0x2
    mova r16, #0
    done
```

## Working Variants

### Index 1 instead of 0 (PASSES)

```asm
    vlda wl0, [p0, #0x0]
    vlda wh0, [p0, #0x20]
    mov r16, #1                 // only change: index = 1
    nop x7
    vextract.d16 r0, x0, r16   // works fine
```

### No vlda -- uninitialized register (PASSES)

```asm
    mov r16, #0
    nop x7
    vextract.d16 r0, x0, r16   // works fine (x0 not loaded by vlda)
```

### Compiler-generated code for v[0] (PASSES)

Peano compiler output for:
```cpp
aie::vector<int16_t, 32> v = aie::load_v<32>(in);
out[0] = v[0];
```

The compiler produces a different instruction schedule that avoids the hang,
though no individual scheduling difference we tested accounts for it (see
Test Matrix below).

## Test Matrix

### Index sweep (constant index in MOV, vlda present)

| Index | Result |
|-------|--------|
| 0     | HANG   |
| 1-15  | PASS   |

### Variant sweep (index 0, different instruction combinations)

| Variant                                      | Result |
|----------------------------------------------|--------|
| vlda wl0 + vlda wh0 + vextract.d16 idx=0    | HANG   |
| vlda wl0 only + vextract.d16 idx=0           | HANG   |
| vlda wl0 + vldb wh0 + vextract.d16 idx=0    | HANG   |
| vlda wl0 + vlda wh0 + vextract.s16 idx=0    | HANG   |
| vlda wl0 + vlda wh0 + vextract.d32 idx=0    | HANG   |
| no vlda + vextract.d16 idx=0                 | PASS   |
| vlda wl0 + vlda wh0 + vextract.d16 idx=1    | PASS   |
| mova r16 instead of mov r16 + vextract idx=0 | HANG   |
| 20 nops between vlda and vextract             | HANG   |
| startup delay (20 nops before vlda)           | HANG   |
| store instruction between vlda and vextract   | HANG   |
| compiler-generated code with v[0]             | PASS   |

### Key observations

- **Not a pipeline hazard**: 20 nops (far exceeding the 7-cycle vlda latency)
  does not help.
- **Not slot-specific**: `mova` (LDA slot) vs `mov` (MV slot) for loading
  the index register makes no difference.
- **Not sign-specific**: Both `.d16` (dynamic sign) and `.s16` (signed) hang.
- **Not element-width-specific**: `.d16` and `.d32` both hang.
- **vlda is required**: Without vlda writing the source vector register,
  vextract at index 0 works fine.
- **Only index 0**: All indices 1-15 pass under identical conditions.

## Compiler Workaround

The Peano compiler's generated code for `v[0]` works, but we were unable to
isolate which specific scheduling difference prevents the hang. Testing each
individual difference (vldb instead of vlda, mova instead of mov, inserting
a store between load and extract) did NOT reproduce the compiler's success.

The Peano scheduling model enforces "no bypass between VLDA and VEXTRACT"
(see `llvm/test/CodeGen/AIE/aie2/schedule/vextract.mir`), which naturally
avoids the pattern, but this appears to be a coincidence rather than a
deliberate workaround -- the scheduling constraint exists for normal pipeline
latency reasons, not for this specific errata.

## Encoding

`vextract.d16 r0, x0, r16` encodes as `b9 06 04 18` (MV slot, 22 bits):

```
{vec_extract_dest_1[6:0], vec_extract_dest_0[1:0], sign, idx[1:0], s1[2:0], 0b1101, 0b01}

Where:
  idx = 2-bit register selector: r16=0b00, r17=0b01, r18=0b10, r19=0b11
  s1 = 3-bit source vector register
  sign = 0 (unsigned/dynamic) or 1 (signed)
```

The `idx` field selects which ERS4 register (r16-r19) holds the element
index. The actual index is the VALUE in that register, not the encoding.
When r16 contains 0, element 0 is extracted -- and the core hangs.

## How to Reproduce

Build environment: mlir-aie with Peano compiler, XRT 2.23.

1. Save the minimal reproducer assembly as `kernel.s`
2. Assemble: `llvm-mc --triple=aie2-none-unknown-elf --filetype=obj kernel.s -o kernel.o`
3. Wrap in MLIR (objectfifo in, objectfifo out, call @test_kernel)
4. Compile: `aiecc.py --no-xchesscc --no-xbridge --aie-generate-cdo --aie-generate-npu-insts --npu-insts-name=insts.bin aie_wrapper.mlir`
5. Run the resulting test.exe on Phoenix NPU hardware

The test will hang until TDR timeout (~10 seconds). Changing `mov r16, #0`
to `mov r16, #1` makes it pass immediately.

Complete reproducer files (assembly kernels, MLIR wrappers, build scripts)
are available on request.

## Impact

Compiler-generated code appears to be unaffected (both Chess and Peano avoid
the pattern through scheduling). Hand-written assembly, inline asm, and
direct intrinsic use could trigger this. The ISA test harness for our NPU
emulator project hit it during systematic instruction validation.
