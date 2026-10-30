//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_agu_vload capture design (GENERATED). One compute tile (0,2) consumes
// 10 chunks of 2160 words and produces 5504 words per chunk.
// one chunk per (family, dimension); records of 24 words, then the store image.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @agu_vload(memref<2160xi32>, memref<5504xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<2160xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<5504xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 10 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<2160xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<5504xi32>
        func.call @agu_vload(%a, %o) : (memref<2160xi32>, memref<5504xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<21600xi32>, %o: memref<55040xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][10, 1, 1, 2160][2160, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<21600xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][10, 1, 1, 5504][5504, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<55040xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
