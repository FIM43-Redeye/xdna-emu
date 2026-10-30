//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_unpack_addr capture design (GENERATED). One compute tile (0,2) consumes
// 8 chunks of 648 words and produces 5664 words per chunk.
// one chunk per (crUnpackSign, form); 236 records of 24 words.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @unpack_addr(memref<648xi32>, memref<5664xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<648xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<5664xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 8 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<648xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<5664xi32>
        func.call @unpack_addr(%a, %o) : (memref<648xi32>, memref<5664xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<5184xi32>, %o: memref<45312xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][8, 1, 1, 648][648, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<5184xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][8, 1, 1, 5664][5664, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<45312xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
