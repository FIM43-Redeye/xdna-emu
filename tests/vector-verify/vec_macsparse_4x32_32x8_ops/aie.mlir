//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_macsparse_4x32_32x8_ops capture design (GENERATED). One compute tile (0,2) consumes
// 20 chunks of 312 words and produces 2816 words per chunk.
// sparse mode (0, 0, 1) 4x32_32x8; 4 flavour(s) x 11 stage(s) x 32 words per tile.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @mac_sparse(memref<312xi32>, memref<2816xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<312xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<2816xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 20 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<312xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<2816xi32>
        func.call @mac_sparse(%a, %o) : (memref<312xi32>, memref<2816xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<6240xi32>, %o: memref<56320xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][20, 1, 1, 312][312, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<6240xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][20, 1, 1, 2816][2816, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<56320xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
