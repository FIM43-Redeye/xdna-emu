//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_macsparse_4x16_16x4_conf capture design (GENERATED). One compute tile (0,2) consumes
// 3 chunks of 2496 words and produces 2560 words per chunk.
// sparse mode (2, 3, 2) 4x16_16x4; 1 flavour(s) x 10 stage(s) x 16 words per tile.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @mac_sparse(memref<2496xi32>, memref<2560xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<2496xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<2560xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 3 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<2496xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<2560xi32>
        func.call @mac_sparse(%a, %o) : (memref<2496xi32>, memref<2560xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<7488xi32>, %o: memref<7680xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][3, 1, 1, 2496][2496, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<7488xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][3, 1, 1, 2560][2560, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<7680xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
