//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_pack_addr capture design (GENERATED). One compute tile (0,2) consumes
// 4 chunks of 160 words and produces 4608 words per chunk.
// 8 blocks of 64 bytes per chunk; 576 result words per block.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @pack_addr(memref<160xi32>, memref<4608xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<160xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<4608xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<160xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<4608xi32>
        func.call @pack_addr(%a, %o) : (memref<160xi32>, memref<4608xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<640xi32>, %o: memref<18432xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][4, 1, 1, 160][160, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<640xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][4, 1, 1, 4608][4608, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<18432xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
