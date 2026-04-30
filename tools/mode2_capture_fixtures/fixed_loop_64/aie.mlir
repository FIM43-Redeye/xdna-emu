// Mode-2 fixture aie.mlir: fixed_loop_64.
// Mirrors runtime_loop's plumbing exactly; the difference is the kernel
// uses a compile-time-known trip count (N=64), so Peano emits a vectorized
// ZOL with LC = 15 (16 vector iterations).

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    func.func private @k_fixed_loop_64(memref<64xi32>, memref<64xi32>) attributes {link_with = "kernel.o"}

    aie.core(%tile_0_2) {
      %sub_in   = aie.objectfifo.acquire @of_in(Consume, 1)  : !aie.objectfifosubview<memref<64xi32>>
      %elem_in  = aie.objectfifo.subview.access %sub_in[0]   : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
      %sub_out  = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
      %elem_out = aie.objectfifo.subview.access %sub_out[0]  : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

      func.call @k_fixed_loop_64(%elem_in, %elem_out) : (memref<64xi32>, memref<64xi32>) -> ()

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%in : memref<64xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0  = arith.constant 0 : i64
      %c1  = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in [%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_in,  id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
