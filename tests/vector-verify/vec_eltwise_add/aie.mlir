//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B element-wise vector-add design: single compute tile (0,2) reads two
// 8xi32 inputs from DDR via shim DMA, runs the vector add kernel, writes the
// 8xi32 result back to DDR. Direct shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @eltwise_add(memref<8xi32>, memref<8xi32>, memref<8xi32>)
        attributes {link_with = "eltwise.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @inB(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
      %sb = aie.objectfifo.acquire @inB(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
      %b = aie.objectfifo.subview.access %sb[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>

      func.call @eltwise_add(%a, %b, %o) : (memref<8xi32>, memref<8xi32>, memref<8xi32>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @inB(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<8xi32>, %b: memref<8xi32>, %c: memref<8xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 1 : i64, metadata = @inB} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 2 : i64, metadata = @outC} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
