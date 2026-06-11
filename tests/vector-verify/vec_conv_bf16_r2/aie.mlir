//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B vec_conv_bf16_r2 capture design (GENERATED). Single compute tile (0,2) reads a
// 448-element batch from DDR via shim DMA, runs the vec_conv_bf16_r2 kernel, writes the
// result back. Direct shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @conv_bf16(memref<448xf32>, memref<448xbf16>)
        attributes {link_with = "conv.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<448xf32>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<448xbf16>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<448xf32>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<448xf32>> -> memref<448xf32>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<448xbf16>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<448xbf16>> -> memref<448xbf16>

      func.call @conv_bf16(%a, %o) : (memref<448xf32>, memref<448xbf16>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<448xf32>, %c: memref<448xbf16>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 448][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<448xf32>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 448][0, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<448xbf16>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
