module {
  aie.device(npu1_1col) {
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "objFifo_in1_cons_buff_0"} : memref<8xi32> 
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "objFifo_in1_cons_buff_1"} : memref<8xi32> 
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "objFifo_out1_buff_0"} : memref<8xi32> 
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "objFifo_out1_buff_1"} : memref<8xi32> 
    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock"}
    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    %core_0_2 = aie.core(%tile_0_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c8 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c8 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %4 = memref.load %objFifo_in1_cons_buff_0[%2] : memref<8xi32>
      %5 = arith.addi %4, %c1_i32 : i32
      memref.store %5, %objFifo_out1_buff_0[%2] : memref<8xi32>
      %6 = arith.addi %2, %c1 : index
      cf.br ^bb3(%6 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
      aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
      aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%7: index):  // 2 preds: ^bb5, ^bb7
      %8 = arith.cmpi slt, %7, %c8 : index
      cf.cond_br %8, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %9 = memref.load %objFifo_in1_cons_buff_1[%7] : memref<8xi32>
      %10 = arith.addi %9, %c1_i32 : i32
      memref.store %10, %objFifo_out1_buff_1[%7] : memref<8xi32>
      %11 = arith.addi %7, %c1 : index
      cf.br ^bb6(%11 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
      aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
      %12 = arith.addi %0, %c2 : index
      cf.br ^bb1(%12 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    }
    aie.shim_dma_allocation @objFifo_in0(%shim_noc_tile_0_0, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>, %arg3: memref<262144xi32>) {
      aiex.npu.write32 {address = 606416 : ui32, column = 0 : i32, row = 1 : i32, value = 2627534848 : ui32}
      aiex.npu.write32 {address = 606420 : ui32, column = 0 : i32, row = 1 : i32, value = 12317 : ui32}
      aiex.npu.write32 {address = 606432 : ui32, column = 0 : i32, row = 1 : i32, value = 1549292624 : ui32}
      aiex.npu.write32 {address = 606436 : ui32, column = 0 : i32, row = 1 : i32, value = 1818780768 : ui32}
      aiex.npu.write32 {address = 724736 : ui32, column = 0 : i32, row = 1 : i32, value = 16780832 : ui32}
      aiex.npu.write32 {address = 724740 : ui32, column = 0 : i32, row = 1 : i32, value = 84148994 : ui32}
      aiex.npu.write32 {address = 606208 : ui32, column = 0 : i32, row = 1 : i32, value = 7424 : ui32}
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 30 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 1260724769 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 439168079 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.write32 {address = 82128 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 82132 : ui32, column = 0 : i32, row = 2 : i32, value = 4127 : ui32}
      aiex.npu.write32 {address = 82144 : ui32, column = 0 : i32, row = 2 : i32, value = 1313674515 : ui32}
      aiex.npu.write32 {address = 82148 : ui32, column = 0 : i32, row = 2 : i32, value = 202068047 : ui32}
      aiex.npu.write32 {address = 81920 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 262144 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119304 : ui32, column = 0 : i32, mask = 7936 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483663 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c64_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @objFifo_in0} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c64_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @objFifo_out0}
      aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %objFifo_in0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "objFifo_in0_cons_buff_0"} : memref<16xi32> 
      %objFifo_in0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "objFifo_in0_cons_buff_1"} : memref<16xi32> 
      %objFifo_out0_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "objFifo_out0_buff_0"} : memref<16xi32> 
      %objFifo_out0_buff_1 = aie.buffer(%mem_tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "objFifo_out0_buff_1"} : memref<16xi32> 
      %objFifo_in0_cons_prod_lock = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
      %objFifo_in0_cons_cons_lock = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
      %objFifo_out0_prod_lock = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "objFifo_out0_prod_lock"}
      %objFifo_out0_cons_lock = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    aie.shim_dma_allocation @objFifo_out0(%shim_noc_tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<8xi32>, 0, 8) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in1_cons_buff_1 : memref<8xi32>, 0, 8) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out1_buff_0 : memref<8xi32>, 0, 8) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out1_buff_1 : memref<8xi32>, 0, 8) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.packet_flow(29) {
      aie.packet_source<%mem_tile_0_1, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(30) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(31) {
      aie.packet_source<%tile_0_2, Trace : 1>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
