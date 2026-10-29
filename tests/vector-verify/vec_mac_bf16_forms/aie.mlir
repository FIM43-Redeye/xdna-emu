//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_mac_bf16_forms capture design (GENERATED). One compute tile (0,2) consumes
// 21 chunks of 16 tiles (A, B, C0, C1 packed into one bf16-typed buffer of
// 2048 elements) and produces 23 forms x 16 lanes per tile.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @mac_forms(memref<2048xbf16>, memref<5888xf32>)
        attributes {link_with = "forms.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<5888xf32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 21 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<2048xbf16>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<5888xf32>
        func.call @mac_forms(%a, %o) : (memref<2048xbf16>, memref<5888xf32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<43008xbf16>, %o: memref<123648xf32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][21, 1, 1, 2048][2048, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<43008xbf16>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][21, 1, 1, 5888][5888, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<123648xf32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
