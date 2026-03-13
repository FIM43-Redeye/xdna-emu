# DMA Engine -- Divergence Catalog

Audited: 2026-03-12
Agent: F (DMA subsystem)

---

## [DMA-D1] Missing MemTile BD-Channel Validity Check

- **Severity**: LOW
- **Our behavior**: Any BD (0-47) can be assigned to any MemTile DMA channel.
  No validation is performed at BD parse time or channel start time.
- **aie-rt behavior**: `_XAieMl_MemTileDmaCheckBdChValidity()`
  (xaie_dma_aieml.c:1320-1331) enforces that BD 0-23 are valid only for
  even channels (0, 2, 4) and BD 24-47 are valid only for odd channels
  (1, 3, 5). Returns `XAIE_INVALID_ARGS` on mismatch.
- **Impact**: No test failures expected. The compiler (mlir-aie) generates
  correct BD-channel assignments that satisfy this constraint. An incorrect
  user-generated CDO could bypass this check and produce undefined behavior
  that the emulator would silently execute instead of rejecting.
- **Suggested fix**: Add a validation check in `DmaEngine::start_channel()`
  for MemTile that rejects invalid BD-channel combinations. Log a warning
  rather than erroring, since the emulator should be permissive.
- **Fixed in-place**: no (validation-only, no behavioral impact)

---

## [DMA-D2] Missing MemTile Zero-Padding Validation

- **Severity**: LOW
- **Our behavior**: Zero-padding fields are parsed and applied without
  validating the consistency rules between wrap and padding.
- **aie-rt behavior**: `_XAieMl_DmaMemTileCheckPaddingConfig()`
  (xaie_dma_aieml.c:218-266) enforces:
  - If D{N}_wrap == 0, then D{N}_after must be 0
  - If D{N}_wrap == 0, then all higher-dimension before/after must be 0
  - D0 padding max: 6 bits (63), D1 max: 5 bits (31), D2 max: 4 bits (15)
- **Impact**: No test failures expected. The compiler generates valid
  configurations. Incorrect CDOs would produce undefined padding behavior
  in the emulator.
- **Suggested fix**: Add `validate_padding()` to `BufferDescriptor` and
  call it from `to_bd_config()` or the engine's BD setup phase. Emit a
  warning on invalid combinations rather than hard-failing.
- **Fixed in-place**: no (validation-only, no behavioral impact)

---

## [DMA-D3] Shim BD Missing D2_Wrap Field

- **Severity**: LOW (informational)
- **Our behavior**: Shim BD `d2_wrap` is always 0 (bd.rs:376). This is
  correct -- the shim BD register layout does not have a D2_Wrap field.
- **aie-rt behavior**: `_XAieMl_ShimDmaReadBd()` never reads a D2_Wrap
  field from shim BD registers. The shim BD layout in xaiemlgbl_params.h
  (NOC_MODULE_DMA_BD0_*) does not define D2_Wrap.
- **Impact**: None. This is not a divergence -- both aie-rt and our emulator
  correctly recognize that shim BDs have no D2_Wrap.
- **Suggested fix**: None needed. Documented for completeness.
- **Fixed in-place**: N/A

---

## [DMA-D4] Shim BD Missing Compression Enable Field

- **Severity**: LOW (informational)
- **Our behavior**: Shim BD `compression_enable` is always `false`
  (bd.rs:377). The shim BD parser does not extract this field.
- **aie-rt behavior**: `_XAieMl_ShimDmaWriteBd()` and
  `_XAieMl_ShimDmaReadBd()` never read/write an Enable_Compression field
  for shim BDs. The NOC_MODULE BD register layout does not include this
  field.
- **Impact**: None. Shim DMA operates on DDR data and does not support
  sparsity compression. Both codebases correctly omit this field.
- **Suggested fix**: None needed.
- **Fixed in-place**: N/A

---

## [DMA-D5] DMA Init Default StepSize Not Explicitly Validated

- **Severity**: LOW
- **Our behavior**: When a BD has a stepsize of 0 in the register (which
  decodes to actual=1 after +1), we use it normally. There is no explicit
  check that stepsize >= 1.
- **aie-rt behavior**: `_XAieMl_DmaSetMultiDim()` (xaie_dma_aieml.c:174)
  rejects `StepSize == 0` as invalid. The init functions
  (`_XAieMl_TileDmaInit()` etc.) set default stepsize to 1 (actual).
- **Impact**: Minimal. Since stepsizes are stored as actual-1, a register
  value of 0 maps to actual=1, which is the valid minimum. The only way
  to get actual=0 would be a register value of -1 (impossible for unsigned
  fields). Our +1 conversion inherently prevents this case.
- **Suggested fix**: None needed. The encoding prevents the invalid state.
- **Fixed in-place**: N/A

---

## Summary

| ID | Severity | Type | Impact |
|----|----------|------|--------|
| DMA-D1 | LOW | Missing validation | No behavioral impact |
| DMA-D2 | LOW | Missing validation | No behavioral impact |
| DMA-D3 | LOW | Informational | Not a divergence |
| DMA-D4 | LOW | Informational | Not a divergence |
| DMA-D5 | LOW | Informational | Not a divergence |

**No CRITICAL or HIGH severity divergences found.**

The DMA engine implementation is accurate. All divergences are
validation-layer gaps (checks that aie-rt performs on the programming
API side) rather than behavioral differences in how the DMA operates.
The compiler toolchain ensures valid configurations, so these gaps do
not affect test results.
