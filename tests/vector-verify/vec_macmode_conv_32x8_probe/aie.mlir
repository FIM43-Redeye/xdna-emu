//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_macmode_conv_32x8_probe capture design (GENERATED). One compute tile (0,2) consumes
// 8 chunks of 1024 words and produces 4096 words per chunk.
// mode (0, 1, 4) conv_32x8; 4 signedness flavours x 1 stage(s) x 32 words per tile.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @mac_mode(memref<1024xi32>, memref<4096xi32>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @outC(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<4096xi32>>

    %core_0_2 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 8 : index
      scf.for %i = %c0 to %cn step %c1 {
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<1024xi32>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<4096xi32>
        func.call @mac_mode(%a, %o) : (memref<1024xi32>, memref<4096xi32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<8192xi32>, %o: memref<32768xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][8, 1, 1, 1024][1024, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<8192xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][8, 1, 1, 4096][4096, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<32768xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
