//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B vec_pack_i16_s3 capture design (GENERATED). Single compute tile (0,2) reads a
// 96-element batch from DDR via shim DMA, runs the vec_pack_i16_s3 kernel, writes the
// result back. Direct shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @pack_i16(memref<96xi16>, memref<96xi8>)
        attributes {link_with = "pack.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<96xi16>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<96xi8>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<96xi16>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<96xi16>> -> memref<96xi16>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<96xi8>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<96xi8>> -> memref<96xi8>

      func.call @pack_i16(%a, %o) : (memref<96xi16>, memref<96xi8>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<96xi16>, %c: memref<96xi8>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 96][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<96xi16>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 96][0, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<96xi8>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
