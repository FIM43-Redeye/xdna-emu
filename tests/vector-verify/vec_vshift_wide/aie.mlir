//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_vshift_wide capture design (GENERATED). One compute tile (0,2) consumes
// 38 chunks of 162 words and produces 1024 words per chunk.
// Chunk = A, B, mode, 64 (step, shift) entries; 64 x 64-byte results.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @vshift_wide(memref<162xi32>, memref<1024xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<162xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 38 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<162xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<1024xi32>
        func.call @vshift_wide(%a, %o) : (memref<162xi32>, memref<1024xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<6156xi32>, %o: memref<38912xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][38, 1, 1, 162][162, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<6156xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][38, 1, 1, 1024][1024, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<38912xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
