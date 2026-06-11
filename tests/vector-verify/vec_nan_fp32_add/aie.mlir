//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// NaN/Inf sweep design (GENERATED). Single compute tile (0,2) reads two
// 256-element f32 operand matrices from DDR via shim DMA, runs the
// add_fp32 kernel, writes the 256-element f32 result back. Direct
// shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @add_fp32(memref<256xf32>, memref<256xf32>, memref<256xf32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<256xf32>>
    aie.objectfifo @inB(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<256xf32>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xf32>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<256xf32>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %sb = aie.objectfifo.acquire @inB(Consume, 1) : !aie.objectfifosubview<memref<256xf32>>
      %b = aie.objectfifo.subview.access %sb[0] : !aie.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<256xf32>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<256xf32>> -> memref<256xf32>

      func.call @add_fp32(%a, %b, %o) : (memref<256xf32>, memref<256xf32>, memref<256xf32>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @inB(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<256xf32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @inB} : memref<256xf32>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @outC} : memref<256xf32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
