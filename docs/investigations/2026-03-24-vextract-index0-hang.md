# VEXTRACT Index-0 Hardware Hang Investigation

**Date**: 2026-03-24
**Status**: Open -- suspected silicon errata, not yet confirmed externally
**Affected Hardware**: AMD Phoenix (NPU1), AIE2 architecture
**Firmware**: 1.5.5.391
**Driver**: xdna-driver 0c6249f (2026-03-24 build)
**XRT**: 2.23.0 (481583db9)

## Summary

`vextract.d16` (and `.s16`, `.d32`) instructions hang the AIE2 compute core
when the element index register contains value 0 AND the source vector
register was previously written by `vlda`. The hang is a permanent core
stall (Status: 8 / timeout), not a crash. The core never completes.

The Peano compiler's generated code avoids this issue through different
instruction scheduling, but the exact mechanism is not yet understood.

## Reproduction

### Minimal reproducer (HANGS)

```asm
vlda wl0, [p0, #0x0]      // load low 256 bits of x0
mov r16, #0                // index = 0
nop x7                     // 7-cycle pipeline delay (vlda latency)
vextract.d16 r0, x0, r16  // HANGS HERE
```

### Working variant (index 1 instead of 0)

```asm
vlda wl0, [p0, #0x0]
mov r16, #1                // index = 1 instead of 0
nop x7
vextract.d16 r0, x0, r16  // works fine
```

### Working variant (no vlda -- uninitialized register)

```asm
mov r16, #0
nop x7
vextract.d16 r0, x0, r16  // works fine (x0 not loaded by vlda)
```

### Working variant (compiler-generated)

Peano compiler output for `aie::vector<int16_t, 32> v = aie::load_v<32>(in); out[0] = v[0];`:

```asm
nopa;  nopx;  mov p2, p0
vlda wl0, [p0, #0x0]; paddb [p2], #32
vldb wh0, [p2, #0x0]
nop
nop
st.s16 r0, [p1], #0x2     // store (unrelated) between load and extract
nop
nop
mova r16, #0x0; mov r1, r16  // VLIW bundle
vextract.s16 r0, x0, r16     // WORKS
```

## Test Matrix

### Index sweep (constant index, vlda present)

All tests use: `vlda wl0, [p0, #0]` + `vlda wh0, [p0, #0x20]` + 7 nops +
`mov r16, #N` + `vextract.d16 r0, x0, r16`.

| Index | Result |
|-------|--------|
| 0     | HANG   |
| 1-15  | PASS   |

With variable indices from PRNG (before masking fix), a periodic pattern
of hangs appeared that correlated with the index value modulo 16.

### Variant sweep (index 0, different instruction combinations)

| Variant | Result |
|---------|--------|
| vlda wl0 + vlda wh0 + vextract.d16 idx=0 | HANG |
| vlda wl0 only + vextract.d16 idx=0 | HANG |
| vlda wl0 + vldb wh0 + vextract.d16 idx=0 | HANG |
| vlda wl0 + vlda wh0 + vextract.s16 idx=0 | HANG |
| vlda wl0 + vlda wh0 + vextract.d32 idx=0 | HANG |
| no vlda + vextract.d16 idx=0 | PASS |
| vlda wl0 + vlda wh0 + vextract.d16 idx=1 | PASS |
| mova r16 instead of mov r16 + vextract idx=0 | HANG |
| 20 nops instead of 7 between vlda and vextract | HANG |
| startup delay (20 nops before vlda) | HANG |
| store instruction between vlda and vextract | HANG |
| compiler-generated code with v[0] | PASS |

### Seed sweep (variable PRNG data, constant index)

With `vextract.d16 r0, x0, r16` where r16 is loaded from PRNG input and
masked to 0-15:

| Seeds | Index range | Result |
|-------|-------------|--------|
| 0-4   | 13,14,15,0,1 | HANG (indices 0 and 13-15 hang) |
| 5-8   | 2,3,4,5 | PASS |
| 9-12  | 6,7,8,9 | HANG |
| 13-16 | 10,11,12,13 | PASS |

**Note**: The PRNG-based results showed more indices hanging than the
constant-index test. This discrepancy is not yet explained -- the AND
masking should produce identical index values. The additional hangs at
indices 6-9 and 13-15 in PRNG mode may indicate the full 32-bit register
value matters, not just the low bits used as the index.

## What We Know

1. **Index 0 is the trigger**: Constant-index tests show only index 0 hangs.
   PRNG tests show additional indices hanging, suggesting upper register
   bits may also matter.

2. **vlda is required**: Without a prior vlda to the source vector register,
   vextract at index 0 works fine.

3. **Not a pipeline hazard**: Increasing nop delay from 7 to 20 cycles
   doesn't help. The vlda has a 7-cycle latency (per AIE2Schedule.td),
   and we exceed it.

4. **Not slot-specific**: Changing `mov` to `mova` (different VLIW slot)
   doesn't help. Both .d16 and .s16 (different sign bit) hang.

5. **Compiler avoids it**: Peano's generated code for `v[0]` works. The
   differences are:
   - Uses `vlda` + `vldb` (different load slots) instead of `vlda` + `vlda`
   - Uses `mova` in a VLIW bundle with `mov r1, r16`
   - Has a `st.s16` instruction between the loads and the extract
   - Scheduled differently overall (VLIW bundles, different nop placement)

   However, individually testing each of these differences (vldb, mova,
   store-between) does NOT reproduce the compiler's success. The exact
   combination or some other aspect of the compiler's scheduling is key.

6. **Permanent stall**: The core enters CASCADE_STALL_MCD-like state
   (Status: 8) and never recovers. The TDR (Timeout Detection and
   Recovery) eventually fires and kills the context.

## Encoding Details

`vextract.d16 r0, x0, r16` encodes as `b9 06 04 18`:

```
MV slot (22 bits):
  {vec_extract_dest_1, vec_extract_dest_0, sign, idx, s1, 0b1101, 0b01}

Where:
  vec_extract_dest_1 = 7-bit destination (r0)
  vec_extract_dest_0 = 2-bit destination extension
  sign = 0 (unsigned/d16) or 1 (signed/s16)
  idx = 2-bit register selector (r16=0, r17=1, r18=2, r19=3)
  s1 = 3-bit source vector (x0=0)
```

The `idx` field selects which ERS4 register (r16-r19) holds the element
index. The actual element index is the VALUE in that register. When
idx=0b00 (selecting r16) and r16 contains 0, element 0 is extracted.

## Open Questions

1. **Why does the compiler output work?** We tested each individual
   difference and none alone fixes the hang. Is it a specific VLIW bundle
   combination? A linker-level alignment? A crt0/startup sequence
   difference?

2. **Is this a known errata?** AMD may have documented this in internal
   errata sheets. The Peano compiler's scheduler may contain a workaround
   without public documentation.

3. **PRNG vs constant index discrepancy**: Why do additional indices
   (6-9, 13-15) hang with PRNG-loaded values but not with constant MOV?
   Upper register bits? Pipeline state from the AND instruction?

4. **Does this affect real workloads?** The compiler appears to avoid
   the issue naturally. But hand-written assembly (intrinsics, inline asm)
   could trigger it.

## Workaround for ISA Test Harness

For the ISA test generator (`tools/isa-test-gen.py`):
- Mask vextract indices to start from 1 instead of 0, OR
- Use the compiler (C++ with aie_api) to generate vextract test kernels
  instead of hand-written assembly, OR
- Skip vextract index-0 test points with a documented note.

## Related Fixes (same investigation session)

### Cascade write hang (batch_62) -- FIXED

`vmov MCD, x0` (cascade write) was bin-packed into single-tile batches.
Without a downstream consumer tile, writing to MCD permanently stalls the
core. Fixed by rejecting cascade writes from `CascadeStrategy.can_test()`.

### vextbcst index masking -- FIXED

`vextbcst` instructions used `scalar` register kind for the index operand,
but the masking code only checked for `ERS4` kind. Extended the check to
`("ERS4", "scalar")`. Also fixed suffix matching (`endswith("64")` didn't
match names like `VEXTBCST_64_mRm` -- changed to `"_64" in name`).

## Files

Reproducer kernels: `build/repro-batch34/`
ISA test generator fixes: `tools/isa-test-gen.py` (cascade write, index masking)
