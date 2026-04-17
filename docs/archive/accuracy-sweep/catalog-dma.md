# DMA Engine -- Divergence Catalog

Audited: 2026-03-12 (re-audited on dev branch)
Agent: F (DMA subsystem)

---

## [DMA-D1] MemTile BD-Channel Validity Check -- RESOLVED

- **Severity**: LOW (resolved)
- **Status**: RESOLVED. `check_memtile_bd_channel_validity()` in engine.rs
  implements the exact same logic as aie-rt
  `_XAieMl_MemTileDmaCheckBdChValidity()`. Called from
  `start_channel_with_repeat()` (engine.rs:758). Logs a warning on invalid
  combinations rather than returning an error, matching the emulator's
  permissive design.
- **aie-rt behavior**: BD 0-23 valid for even per-direction channels
  (0, 2, 4); BD 24-47 valid for odd channels (1, 3, 5). Returns
  `XAIE_INVALID_ARGS` on mismatch.
- **Our behavior**: Same check, logs warning instead of erroring.
- **Residual risk**: None. Behavioral equivalence confirmed.

---

## [DMA-D2] MemTile Zero-Padding Validation -- RESOLVED

- **Severity**: LOW (resolved)
- **Status**: RESOLVED. `ZeroPadConfig::validate_padding()` in
  addressing.rs (lines 208-278) implements the full validation from
  aie-rt `_XAieMl_DmaMemTileCheckPaddingConfig()`. It is called from
  `BufferDescriptor::to_bd_config()` (bd.rs:498-501).
- **aie-rt behavior**: Checks field width limits (D0 max 63, D1 max 31,
  D2 max 15) and wrap-zero propagation rules. Returns error on violation.
- **Our behavior**: Same checks, logs warnings instead of erroring.
  Extensively tested (addressing.rs:888-1005, 12 dedicated tests).
- **Residual risk**: None.

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

---

## [DMA-D5] DMA Init Default StepSize Not Explicitly Validated

- **Severity**: LOW (informational)
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

---

## Summary

| ID | Severity | Type | Status |
|----|----------|------|--------|
| DMA-D1 | LOW | Validation | RESOLVED |
| DMA-D2 | LOW | Validation | RESOLVED |
| DMA-D3 | LOW | Informational | Not a divergence |
| DMA-D4 | LOW | Informational | Not a divergence |
| DMA-D5 | LOW | Informational | Not a divergence |

**No CRITICAL or HIGH severity divergences found.**
**No OPEN divergences remain.**

The DMA engine implementation is accurate. All previously-identified
validation gaps (DMA-D1 and DMA-D2) have been resolved in the current
codebase. The remaining catalog entries (D3-D5) are informational notes
documenting intentional design decisions that match hardware behavior.
