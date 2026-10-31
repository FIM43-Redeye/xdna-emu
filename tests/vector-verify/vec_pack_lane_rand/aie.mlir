//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_pack_lane_rand capture design (GENERATED). One compute tile (0,2) consumes
// 32 chunks of 160 words and produces 4096 words per chunk.
// 8 blocks of 64 bytes per chunk; 512 result words per block.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @pack_lane(memref<160xi32>, memref<4096xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<160xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<4096xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 32 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<160xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<4096xi32>
        func.call @pack_lane(%a, %o) : (memref<160xi32>, memref<4096xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<5120xi32>, %o: memref<131072xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][32, 1, 1, 160][160, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<5120xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][32, 1, 1, 4096][4096, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<131072xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
