//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_sparseload_ring_w16 capture design (GENERATED). One compute tile (0,2) consumes
// 35 chunks of 256 words and produces 3648 words per chunk.
// vldb.sparse width 16; 4 scripts, 32 words per step.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @sparse_load(memref<256xi32>, memref<3648xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<3648xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 35 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<256xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<3648xi32>
        func.call @sparse_load(%a, %o) : (memref<256xi32>, memref<3648xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<8960xi32>, %o: memref<127680xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][35, 1, 1, 256][256, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<8960xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][35, 1, 1, 3648][3648, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<127680xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
