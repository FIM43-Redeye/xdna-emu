//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_pack_crtiming capture design (GENERATED). One compute tile (0,2) consumes
// 6 chunks of 1104 words and produces 2048 words per chunk.
// 64 sign words, 16 control words, then 64 blocks per chunk; 32 result words per block.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @pack_crt(memref<1104xi32>, memref<2048xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<1104xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<2048xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 6 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<1104xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<2048xi32>
        func.call @pack_crt(%a, %o) : (memref<1104xi32>, memref<2048xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<6624xi32>, %o: memref<12288xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][6, 1, 1, 1104][1104, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<6624xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][6, 1, 1, 2048][2048, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<12288xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
