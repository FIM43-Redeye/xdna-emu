//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_unpack_lane_probe capture design (GENERATED). One compute tile (0,2) consumes
// 23 chunks of 144 words and produces 6144 words per chunk.
// 16 blocks of 32 bytes per chunk; 384 result words per block.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @unpack_lane(memref<144xi32>, memref<6144xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<144xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<6144xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 23 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<144xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<6144xi32>
        func.call @unpack_lane(%a, %o) : (memref<144xi32>, memref<6144xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<3312xi32>, %o: memref<141312xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][23, 1, 1, 144][144, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<3312xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][23, 1, 1, 6144][6144, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<141312xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
