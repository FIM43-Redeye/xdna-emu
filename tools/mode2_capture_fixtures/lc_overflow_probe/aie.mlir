// Mode-2 fixture aie.mlir: lc_overflow_probe.
//
// Mirrors heavy_zol's plumbing but with a runtime-driven wrapper count.
// The aie.core acquires once and calls k_pass `passes` times. Each
// k_pass runs one ZOL with trip count N. Both N and passes come from
// the input buffer:
//
//   in[0] = N      -- ZOL trip count
//   in[1] = passes -- outer wrapper count (1..16 in practice)
//
// At low N (<= ~2^24) we want passes=4 for trace volume + redundancy.
// At high N (>= 2^28) we drop passes=1 to stay under the XRT command
// timeout (~3s per kernel before HW TDR fires).
//
// k_pass is called via an scf.while loop because scf.for needs constant
// bounds at this lowering stage; while + counter handles the runtime
// `passes` value. NPUDEVICE is sed-replaced by build_fixture.sh.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    func.func private @k_pass(memref<64xi32>, memref<64xi32>) attributes {link_with = "kernel.o"}

    aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1_idx = arith.constant 1 : index

      %sub_in   = aie.objectfifo.acquire @of_in(Consume, 1)  : !aie.objectfifosubview<memref<64xi32>>
      %elem_in  = aie.objectfifo.subview.access %sub_in[0]   : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
      %sub_out  = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
      %elem_out = aie.objectfifo.subview.access %sub_out[0]  : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

      // passes = in[1]
      %passes = memref.load %elem_in[%c1_idx] : memref<64xi32>

      // for (i32 j = 0; j < passes; j++) k_pass(in, out);
      %final = scf.while (%j = %c0_i32) : (i32) -> (i32) {
        %cond = arith.cmpi slt, %j, %passes : i32
        scf.condition(%cond) %j : i32
      } do {
      ^bb0(%j: i32):
        func.call @k_pass(%elem_in, %elem_out) : (memref<64xi32>, memref<64xi32>) -> ()
        %j_next = arith.addi %j, %c1_i32 : i32
        scf.yield %j_next : i32
      }

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
