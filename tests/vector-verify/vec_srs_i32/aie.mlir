//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B SRS capture design: single compute tile (0,2) reads a 48xi32 batch of
// accumulator values from DDR via shim DMA, runs the SRS kernel, writes the
// 48xi16 shift-round-saturated result back to DDR. Direct shim<->core
// objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @srs_i32(memref<48xi32>, memref<48xi16>)
        attributes {link_with = "srs.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<48xi16>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<48xi32>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<48xi32>> -> memref<48xi32>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<48xi16>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<48xi16>> -> memref<48xi16>

      func.call @srs_i32(%a, %o) : (memref<48xi32>, memref<48xi16>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<48xi32>, %c: memref<48xi16>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 48][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<48xi32>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 48][0, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<48xi16>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
