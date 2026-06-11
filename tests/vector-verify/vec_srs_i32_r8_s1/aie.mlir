//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B vec_srs_i32_r8_s1 capture design (GENERATED). Single compute tile (0,2) reads a
// 64-element batch from DDR via shim DMA, runs the vec_srs_i32_r8_s1 kernel, writes the
// result back. Direct shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @srs_i32(memref<64xi32>, memref<64xi16>)
        attributes {link_with = "srs.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<64xi16>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<64xi16>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<64xi16>> -> memref<64xi16>

      func.call @srs_i32(%a, %o) : (memref<64xi32>, memref<64xi16>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<64xi32>, %c: memref<64xi16>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<64xi16>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
