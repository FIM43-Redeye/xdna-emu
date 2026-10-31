//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_macsparse_2x8_8x8_rand capture design (GENERATED). One compute tile (0,2) consumes
// 14 chunks of 864 words and produces 3072 words per chunk.
// sparse mode (1, 3, 5) 2x8_8x8; 4 flavour(s) x 3 stage(s) x 32 words per tile.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @mac_sparse(memref<864xi32>, memref<3072xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<864xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<3072xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 14 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<864xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<3072xi32>
        func.call @mac_sparse(%a, %o) : (memref<864xi32>, memref<3072xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<12096xi32>, %o: memref<43008xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][14, 1, 1, 864][864, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<12096xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][14, 1, 1, 3072][3072, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<43008xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
