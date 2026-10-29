//
// Minimal untraced AIE design for mlir-trace-inject round-trip tests.
//
// Syntax notes vs. the original plan template:
//   - "aie.runtime_sequence" not "aiex.runtime_sequence": RuntimeSequenceOp
//     lives in the aie dialect; aiex imports it from _aie_ops_gen.  The text
//     syntax always uses the "aie." prefix.
//   - The npu DMA ops (aiex.npu.dma_memcpy_nd / aiex.npu.dma_wait) require a
//     matching @inbound symbol in the device body to validate.  Those ops are
//     omitted here to keep the fixture self-contained.  They will appear in a
//     later fixture once the object-fifo plumbing is in place.
//   - The MLIR printer normalises some syntax (drops explicit SSA names,
//     wraps core results, etc.) -- the round-trip test compares op counts,
//     not raw text, to stay robust across printer versions.
//
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<16xi32>
    aie.core(%tile_0_2) {
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
    }
  }
}
