//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B vec_vfloor_bf16_denorm capture design (GENERATED). Single compute tile (0,2) reads a
// 256-element batch from DDR via shim DMA, runs the vec_vfloor_bf16_denorm kernel, writes the
// result back. Direct shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @vfloor_bf16(memref<256xbf16>, memref<256xi32>)
        attributes {link_with = "vfloor.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<256xbf16>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<256xi32>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

      func.call @vfloor_bf16(%a, %o) : (memref<256xbf16>, memref<256xi32>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<256xbf16>, %c: memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<256xbf16>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<256xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
