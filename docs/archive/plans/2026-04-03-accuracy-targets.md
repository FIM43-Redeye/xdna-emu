# ISA Accuracy Remaining Targets

**Current: ~87.2% (~4216/4833)** as of 2026-04-03 session 7.

## Session 7 fix: VMAXDIFF_LT signed overflow (+12 pts)

Root cause: signed `wrapping_sub().max(0)` gives wrong results when `a - b`
overflows the narrow type. Example for S16: `20000 - (-15000) = 35000` wraps
to -30536 as i16, then `max(0)` returns 0 instead of 35000.

Fix: widen to next integer size before subtracting (i8->i16, i16->i32,
i32->i64), then `max(0)` and truncate. Now 76/76 VMAXDIFF_LT pass.

The earlier hypothesis about "operand ordering / production decoder" was
wrong. The decoder correctly returns operands in TableGen order via LLVM FFI.
VMAXDIFF_LT IS in llvm-aie (the summary from session 6 was incorrect).

## Non-sparse failures (~258 pts)

### Tier 1: Tractable with known root cause

| Instruction | Fail/Total | Root Cause | Effort |
|-------------|-----------|------------|--------|
| VFLOOR_S32_BF16 | 16/16 | 0% -- quarter-accumulator source (amll/amlh/amhl/amhh) not handled. Execute reads full half-accum (8 lanes) instead of quarter (4 lanes). Also the packed BF16 data layout from raw accumulator bytes vs lane values needs investigation. Two variants: AM (accum quarter source) and W (w-register source). | Medium |
| VCONV_BF16_FP32 | 7/7 | Reverse direction (fp32 accum -> bf16 vector). Contraction path in execute_half writes via `write_vector_dest`, needs to handle AccumReg source correctly. Type inference now correct after dedup fix. | Small |
| MOVX_mvx_scl | 7/10 | Partial (3/10 pass). `ControlReg` now in `can_be_dest()`. Remaining 7 failures may be crSat/crRnd value masking (e.g., crSat only stores 2 bits but test writes wider values). | Small |
| DIVS | 19/25 | 2x algorithm mystery in dstep. First ~8 combos have exactly 2.00 ratio (EMU=2*HW). Likely shift count or initialization off-by-one. Needs aiesimulator comparison. | Medium |

### Tier 2: Float precision (needs aietools reference)

| Instruction | Fail/Total | Root Cause |
|-------------|-----------|------------|
| VMAC_F/VMSC_F bm dense | ~12/20 | Internal precision (29-bit MAC), not NaN. |
| VMUL_F bm dense | ~7/10 | Same precision issue. |
| VNEGMUL_F bm dense | ~9/10 | Same precision issue. |
| VADDMAC/VADDMSC bm dense | ~20/80 | 75% pass rate, remaining are precision edge cases. |
| VSUB_F/VNEGSUB_F | 2/10 | NaN mantissa propagation (HW keeps upper bits). |

### Tier 3: Structural / harder

| Instruction | Fail/Total | Root Cause |
|-------------|-----------|------------|
| VLDB_4x (16/32/64, HI/LO) | ~33/42 | Semantic confusion: may be gather load or broadcast. aietools ISS stubs return zeros. |
| Fused vldb.unpack | 4/4 | Fused load+unpack pipeline, completely wrong output. |
| Fused vst.pack | 3/4 | Fused store+pack pipeline, completely wrong output. |
| vlda.ups.s64.s32 | 1/1 | Single fused load+ups failure. |

## Sparse failures (~360 pts) -- DEFERRED

All sparse MAC/MSC/MUL/NEGMUL narrow+wide are 0%. Crossbar routing is
cleanroom-verified. Shared x0 ruled out as blocker. Root cause is
algorithmic divergence in the sparse MAC pipeline itself. This is the
big bucket to tackle after everything else is clean.

## Next actions (priority order)

1. **VFLOOR** (16 pts): Fix quarter-accumulator read + packed BF16 data layout.
2. **VCONV_BF16_FP32** (7 pts): Fix contraction path to read from accum.
3. **MOVX_mvx_scl** (7 pts): Debug remaining 7 failures (crSat masking?).
4. **DIVS** (19 pts): Debug dstep algorithm with aiesimulator.
5. **Float precision** (~50 pts): Needs aietools python_model reference.
6. **Sparse** (~360 pts): Full investigation after above is clean.
