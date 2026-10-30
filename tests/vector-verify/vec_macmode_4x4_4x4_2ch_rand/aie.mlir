//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_macmode_4x4_4x4_2ch_rand capture design (GENERATED). One compute tile (0,2) consumes
// 7 chunks of 1536 words and produces 6144 words per chunk.
// mode (0, 2, 1) 4x4_4x4_2ch; 4 signedness flavours x 3 stage(s) x 32 words per tile.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @mac_mode(memref<1536xi32>, memref<6144xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<1536xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<6144xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 7 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<1536xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<6144xi32>
        func.call @mac_mode(%a, %o) : (memref<1536xi32>, memref<6144xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<10752xi32>, %o: memref<43008xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][7, 1, 1, 1536][1536, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<10752xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][7, 1, 1, 6144][6144, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<43008xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
