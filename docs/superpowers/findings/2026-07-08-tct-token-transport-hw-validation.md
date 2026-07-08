# TCT token-transport thin slice -- HW validation (Phase 4)

**Date:** 2026-07-08
**Branch:** `feat/array-tct-completion`
**Change under test:** `44ebc1e0` -- per-sync stream-transit charge made emergent
(`transit = sync.row * INTER_TILE_HOP_LATENCY`), replacing the flat
`STREAM_FLUSH_CYCLES = 4` in `executor.rs` `BlockedOnSync`->satisfied.
**Design record:** `docs/arch/tct-completion-model.md`.

## Question

Maya's crux for the slice: *is 0 transit for a shim-local token more correct than
the old flat 4, or did the 4 absorb something real?* HW is the oracle.

## What was run

Bridge trace comparison (Peano, HW+EMU) on the three row-diverse kernels found by
surveying real sync rows (`NPU Sync: blocking on (col,row)` logs):

- `add_one_using_dma` -- 1 sync on **(1,0)** shim S2MM (transit 4->0)
- `memtile_dmas/dma_configure_task_{lock,token}` -- syncs on **(1,1)** mem tile
  (4->2) plus **(1,0)** shim (4->0)
- `core_dmas/dma_configure_task_{lock,token}` -- sync on **(1,2)** compute
  (4->4, unchanged) plus **(1,0)** shim

Real syncs are overwhelmingly shim (row 0): the host `WAIT_TCTS` waits for data
landing in / leaving DDR (the shim DMA); intermediate tile-to-tile transfers
synchronize via locks, not host TCT waits. So the change is, in practice, mostly
"drop the per-sync flush 4->0 for shim syncs," with occasional 4->2 (mem).

## Result

- **Correctness: no regression.** All 5 cases PASS on both HW and EMU.
- **The change is unmeasurable against existing shim-timing divergence.** The
  transit shift is <= 4 cy/sync. The shim DMA edge events already diverge HW-vs-EMU
  by **hundreds to thousands of cycles** (`trace-compare`, aligned on the first
  shared edge):

  | Kernel | event | HW-EMU dt |
  |--------|-------|-----------|
  | add_one_using_dma | DMA_S2MM_0_FINISHED_TASK | +412 |
  | add_one_using_dma | DMA_MM2S_0_FINISHED_TASK | -492 |
  | memtile_dmas (lock) | DMA_S2MM_0_FINISHED_TASK | +2773 |
  | memtile_dmas (lock) | DMA_MM2S_0_START_TASK | +3513 |
  | core_dmas (lock) | DMA_MM2S_0_FINISHED_TASK | +3672 |

  The <=4 cy transit sits 2-3 orders of magnitude below this. HW can neither
  confirm nor refute the exact shim transit value at this resolution.

## Conclusion

The transit slice is **structurally faithful and derived-from-toolchain** (it
removes a hardcode: the old 4 was a row-2 tile's 2 hops x 2 cy frozen). But the
question "0 vs 4 for a shim token" is **empirically undecidable** right now: it is
drowned by a +-hundreds-to-thousands-cycle disagreement in the shim DMA-completion
timing that dominates the region the transit lives in.

**Where accuracy is actually lost is the shim DMA-completion / mailbox timing**, not
the token transit. That is the firmware-cold-start residual (the retained `8000`) and
the DMA cycle model -- the same array/firmware wall the whole TCT/firmware arc keeps
hitting ([[project_firmware_emulation_dream]]). Chasing the exact shim transit value
is premature until that dominant term is modeled.

**Disposition:** land the slice (done, merged into the arc branch) as the correct
structural baseline; do NOT tune it against HW (there is no signal to tune to). The
next real accuracy lever is the DMA-completion/mailbox timing, gated on the firmware
loop.
