//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_acc_addsub_f32 capture design (GENERATED). One compute tile (0,2) consumes
// 44 chunks of 512 words and produces 5120 words per chunk.
// 20 forms x 16 lanes per tile; chunk = A[256] then B[256] fp32.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @acc_addsub(memref<512xf32>, memref<5120xf32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<512xf32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<5120xf32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 44 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<512xf32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<5120xf32>
        func.call @acc_addsub(%a, %o) : (memref<512xf32>, memref<5120xf32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<22528xf32>, %o: memref<225280xf32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][44, 1, 1, 512][512, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<22528xf32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][44, 1, 1, 5120][5120, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<225280xf32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
