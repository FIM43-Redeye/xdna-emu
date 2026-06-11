//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B vec_ups_i32_s0 capture design (GENERATED). Single compute tile (0,2) reads a
// 48-element batch from DDR via shim DMA, runs the vec_ups_i32_s0 kernel, writes the
// result back. Direct shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @ups_i32(memref<48xi16>, memref<48xi32>)
        attributes {link_with = "ups.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<48xi16>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<48xi16>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<48xi16>> -> memref<48xi16>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<48xi32>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<48xi32>> -> memref<48xi32>

      func.call @ups_i32(%a, %o) : (memref<48xi16>, memref<48xi32>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<48xi16>, %c: memref<48xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 48][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<48xi16>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 48][0, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<48xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
