# Session State - 2026-04-02 (Updated)

## Current Accuracy

**83.5% (3842/4603)** -- aligned HW baselines with fixed decoder.

Previous 79.5% (3843/4833) had correct baselines but broken sparse decoder.
The mQQXw misclassification meant all sparse MAC tests ran through the
dense path (is_sparse was never true). Test count changed because ISA JSON
regenerated with correct operand types, changing batch layout.

HW baselines: `build/isa-test-results/20260402b/`

## What Was Done This Session

### 1. Fused instruction handlers (commit ebcbb4c, prior session)

All 5 fused instruction types working. Standalone SRS routing fix (47f6e2f)
recovered 703 points (65% -> 79.5%).

### 2. Sparse MAC decoder fixes (commit c209422)

**Three decoder bugs found and fixed:**

a) **mQQXw -> SparseQx** (THE sparse MAC blocker): The `mQQXw` register
   class (qxs2 operand) starts with `mQQ`, matching the accumulator weight
   queue pattern before the `mQX` (SparseQx) check. Result: `is_sparse`
   was never true, ALL sparse MACs ran through the dense path with wrong
   operand reads. Fix: check `mQQX` before `mQQ` in both build_helpers
   and runtime resolver.

b) **eYs (Vector1024) missing from build_helpers**: The runtime resolver
   had `eY` -> `Vector1024`, but build_helpers never got this pattern.
   Result: `ys1` in sparse wide was `Unknown`, `is_quad_vector` was false,
   A operand read only 64/128 bytes. Fix: added eY -> Vector1024.

c) **ql/qh mask register clobbering**: `ql0-ql3` and `qh0-qh3` (64-bit
   halves) shared ControlReg IDs with full `q0-q3`. A load to `ql0`
   would clobber the high half. Fix: distinct IDs (28-31, 32-35) with
   partial-width commit logic. Currently dead code (no ISA encodings
   target ql/qh directly), but correct for future use.

### 3. ISA test infrastructure improvements

- Removed `--no-hw`/`--no-emu` flags from isa-test.sh (stale baseline
  footgun -- caused false regression during this session)
- Parallelized single-tile packaging via xargs -P (was serial)
- Added `vector1024` alias to isa-test-gen.py so sparse_wide tests
  generate (245 new test points, 14 batches)
- Total test points: 4856 (up from 4603)

### 4. Shared x0 register investigation

Investigated whether shared registers (xs1=x0, qxs2=qx0 both reading x0)
cause sparse MAC failures. **Conclusion: NOT the blocker.** The test
generator loads B data last (overwriting A in x0), but this happens
identically on HW and EMU. Both read the same data from x0. The HW vs
EMU comparison should still match regardless. Wide sparse (ys1=y2, qxs2=qx0)
uses independent registers, confirming the algorithm itself diverges.

### 5. Dependency graph for sparse MAC

Built a full dependency graph from test generation through result
comparison. Identified mask register loads, shared registers, and decoder
classification as the three critical risk nodes. Mask loads confirmed
working (ControlReg 16-19 deferred write path). Decoder classification
was the actual blocker.

## Remaining Failure Characterization (761 failing / 4603 total = 16.5%)

### Tier 1: Sparse MAC -- ~460 pts (estimate with wide tests)
- All sparse narrow: 0% (algorithm divergence, not decoder)
- Sparse wide: not yet tested (batches now generate)
- The crossbar routing formula is cleanroom-verified correct
- The shared x0 register is NOT the cause
- Root cause: unknown algorithmic divergence in sparse MAC core

### Tier 2: Float operations -- ~50 pts
- VADD_F, VSUB_F, VNEG_F, VNEGADD_F, VNEGSUB_F: all 0%
- VCONV BF16/FP32: 0%
- VFLOOR: 0%
- VMAC_F/VMSC_F bm_core: partial (20-40%)

### Tier 3: Vector features -- ~120 pts
- VINSERT: 1/47 (2%)
- VBCSTSHFL: 1/39 (3%)
- VLDB_4x: 14-43%
- VSHUFFLE: 1/15 (7%)
- VSHIFT_ALIGN: 1/17 (6%)
- 64-bit extract/broadcast: 0%

### Tier 4: SRS remaining -- ~18 pts
- VSRSM S32 variants: 0%

### Tier 5: Pack/Unpack sub-byte -- ~24 pts
- VPACK 4-bit: 0%, VUNPACK signed/4-bit: 0%

### Tier 6: Scalar/misc -- ~50 pts
- DIVS: 0/25
- MOVX_mvx_scl: 0/10
- ST idx: ~0-8%
- VEQZ: partial
- SBC: 14/25 (56%)

### Perfect categories (0 failures):
scalar-load (474), scalar-shift (25), scalar-bits (34), scalar-compare (184),
scalar-extend (68), pointer (133), vector-bitwise (46), vector-push (94),
vector-ups (96), branch (10)

## Priority for Next Session

1. **Sparse MAC algorithm** (~460 pts) -- decoder is fixed, sparse path
   is now active. Need to debug why the cleanroom crossbar routing
   produces wrong results in the ISA test context.
2. **Multi-tile mode by default** -- 23 cascade/stream pair batches skipped
3. **Float operations** (~50 pts) -- VADD_F, VCONV, VFLOOR
4. **VINSERT/VBCSTSHFL** (~85 pts)

## Key Files Changed

```
build_helpers/semantics.rs           -- eYs->Vector1024, mQQX->SparseQx ordering
src/tablegen/resolver.rs             -- mQQX before mQQ, ys1/qxs2 field name fallback
src/tablegen/decoder_ffi.rs          -- ql/qh distinct ControlReg IDs
src/interpreter/state/context.rs     -- ql/qh partial-width deferred writes
src/interpreter/execute/memory.rs    -- ql/qh store data read
src/interpreter/execute/semantic.rs  -- ql/qh scalar write paths
scripts/isa-test.sh                  -- removed --no-hw/--no-emu, parallel packaging
tools/isa-test-gen.py                -- vector1024 alias for sparse_wide generation
tools/aie2-isa.json                  -- regenerated with correct operand types
```

## Git State

Branch: `dev`
Latest commits:
- `c209422` fix(decode): sparse MAC operand classification, ql/qh mask registers
- `47f6e2f` fix(execute): guard fused handlers with has_memory_operand check
- `ebcbb4c` feat(execute): fused instruction handlers, dead code cleanup
