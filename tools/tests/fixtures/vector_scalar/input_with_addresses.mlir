module {
  aie.device(npu1_1col) {
    %c4096_i64 = arith.constant 4096 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    func.func private @scale_int32(memref<1024xi32>, memref<1024xi32>) attributes {link_with = "scale.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "in_cons_buff_0"} : memref<1024xi32> 
    %in_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "in_cons_buff_1"} : memref<1024xi32> 
    %out_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out_buff_0"} : memref<1024xi32> 
    %out_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out_buff_1"} : memref<1024xi32> 
    %in_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock"}
    %in_cons_lock = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock"}
    %in_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in_cons_prod_lock"}
    %in_prod_lock = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "in_prod_lock"}
    %out_cons_cons_lock = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
    %out_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_cons_lock"}
    %out_cons_prod_lock = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "out_cons_prod_lock"}
    %out_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out_prod_lock"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c4 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, %c1_i32)
      func.call @scale_int32(%in_cons_buff_0, %out_buff_0) : (memref<1024xi32>, memref<1024xi32>) -> ()
      aie.use_lock(%in_cons_prod_lock, Release, %c1_i32)
      aie.use_lock(%out_cons_lock, Release, %c1_i32)
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, %c1_i32)
      func.call @scale_int32(%in_cons_buff_1, %out_buff_1) : (memref<1024xi32>, memref<1024xi32>) -> ()
      aie.use_lock(%in_cons_prod_lock, Release, %c1_i32)
      aie.use_lock(%out_cons_lock, Release, %c1_i32)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["scale.o"]}
    aie.shim_dma_allocation @in(%shim_noc_tile_0_0, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %rt_c213200_i32 = arith.constant 213200 : i32
      %rt_c2038038528_i32 = arith.constant 2038038528 : i32
      %rt_c213204_i32 = arith.constant 213204 : i32
      %rt_c1_i32 = arith.constant 1 : i32
      %rt_c213216_i32 = arith.constant 213216 : i32
      %rt_c2236709_i32 = arith.constant 2236709 : i32
      %rt_c212992_i32 = arith.constant 212992 : i32
      %rt_c31232_i32 = arith.constant 31232 : i32
      %rt_c0_i32 = arith.constant 0 : i32
      %rt_c119304_i32 = arith.constant 119304 : i32
      %rt_c3840_i32 = arith.constant 3840 : i32
      %rt_c7936_i32 = arith.constant 7936 : i32
      %rt_c119308_i32 = arith.constant 119308 : i32
      %rt_c2147483663_i32 = arith.constant 2147483663 : i32
      %rt_c32512_i32 = arith.constant 32512 : i32
      %rt_c213068_i32 = arith.constant 213068 : i32
      %rt_c127_i32 = arith.constant 127 : i32
      %rt_c213000_i32 = arith.constant 213000 : i32
      %rt_c213064_i32 = arith.constant 213064 : i32
      %rt_c126_i32 = arith.constant 126 : i32
      aiex.npu.write32(%rt_c213200_i32, %rt_c2038038528_i32) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.write32(%rt_c213204_i32, %rt_c1_i32) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.write32(%rt_c213216_i32, %rt_c2236709_i32) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.write32(%rt_c212992_i32, %rt_c31232_i32) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 2048 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch(%rt_c0_i32 : i32) buffer %arg2 : memref<4096xi32> {addr = 119268 : ui32}
      aiex.npu.maskwrite32(%rt_c119304_i32, %rt_c3840_i32, %rt_c7936_i32) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      aiex.npu.write32(%rt_c119308_i32, %rt_c2147483663_i32) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.write32(%rt_c212992_i32, %rt_c32512_i32) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.write32(%rt_c213068_i32, %rt_c127_i32) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.write32(%rt_c213000_i32, %rt_c127_i32) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c4096_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, issue_token = true, metadata = @out} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c4096_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @in} : memref<4096xi32>
      aiex.npu.dma_wait {symbol = @out}
      aiex.npu.write32(%rt_c213064_i32, %rt_c126_i32) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.write32(%rt_c213000_i32, %rt_c126_i32) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
    aie.shim_dma_allocation @out(%shim_noc_tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in_cons_prod_lock, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%in_cons_buff_0 : memref<1024xi32> offset = 0 len = 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in_cons_cons_lock, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in_cons_prod_lock, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%in_cons_buff_1 : memref<1024xi32> offset = 0 len = 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in_cons_cons_lock, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%out_buff_0 : memref<1024xi32> offset = 0 len = 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out_prod_lock, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%out_buff_1 : memref<1024xi32> offset = 0 len = 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out_prod_lock, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.trace.config @trace_t0_2_config(%tile_0_2) packet_type = core {
      aie.trace.reg register = "Trace_Control0" value = 2038038528 mask = 2139029507 comment = "trace mode + start event + stop event"
      aie.trace.reg register = "Trace_Control1" value = 1 mask = 28703 comment = "packet ID + packet type"
      aie.trace.reg register = "Trace_Event0" value = 2236709 mask = 8355711 comment = "INSTR_VECTOR + INSTR_EVENT_0 + INSTR_EVENT_1"
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
