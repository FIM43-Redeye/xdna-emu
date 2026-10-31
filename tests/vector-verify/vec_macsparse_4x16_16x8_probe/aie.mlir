//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_macsparse_4x16_16x8_probe capture design (GENERATED). One compute tile (0,2) consumes
// 16 chunks of 1152 words and produces 4096 words per chunk.
// sparse mode (0, 1, 5) 4x16_16x8; 4 flavour(s) x 1 stage(s) x 32 words per tile.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @mac_sparse(memref<1152xi32>, memref<4096xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<1152xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<4096xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 16 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<1152xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<4096xi32>
        func.call @mac_sparse(%a, %o) : (memref<1152xi32>, memref<4096xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<18432xi32>, %o: memref<65536xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][16, 1, 1, 1152][1152, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<18432xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][16, 1, 1, 4096][4096, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<65536xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
