//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// xdna-emu CROSS-CHANNEL dispatch-gate probe (#140 SP-4a). Forked from
// _diag_shim_chain_sweep/k1. Dispatches ONE shim MM2S 0 BD and ONE shim
// S2MM 0 BD BACK-TO-BACK (no dma_wait between), to measure the gap between
// their START_TASK events on HW vs EMU.
//
// The EMU models a SINGLE shared controller dispatch gate
// (`controller_next_taskq_cycle`): the MM2S 0 dispatch arms it to
// npu+dispatch_mm2s(0)=npu+1086, and the S2MM 0 dispatch (next instruction)
// STALLS behind it -> EMU predicts a ~1086cy MM2S0_START -> S2MM0_START gap.
// If HW runs the two independent channels concurrently (gap ~0), the shared
// cross-channel gate is the SP-4a cold-start over-fill bug. If HW also shows
// ~1086, the gate is faithful and the SP-4a offset has another cause.
//
// MM2S 0 and S2MM 0 are DATA channels (their STARTs fire after trace setup,
// so both are captured); S2MM 1 is left free for the trace-data drain (the
// trace-drain channel's own START is unmeasurable -- it starts before the
// trace timer is alive).
//
// Data path: ddr -> shim MM2S 0 -> memtile S2MM 0 -> loopback_buf ->
// memtile MM2S 0 -> shim S2MM 0 -> ddr. S2MM 0 starves briefly after its
// (early) dispatch until the loopback delivers data; START_TASK fires at
// dispatch regardless. Verification is skipped (calibration, not correctness).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)

    %loopback_buf = aie.buffer(%tile_0_1) {sym_name = "loopback_buf"} : memref<64xi32>

    %prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "cons_lock"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    aie.shim_dma_allocation @in (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out (%tile_0_0, S2MM, 0)

    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c64_i64 = arith.constant 64 : i64
      // Back-to-back dispatch: MM2S 0 then S2MM 0, no dma_wait between.
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c64_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @in, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c64_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @out, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in}
      aiex.npu.dma_wait {symbol = @out}
    }

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // S2MM 0 receive loop
      aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%loopback_buf : memref<64xi32>, 0, 64)
      aie.use_lock(%cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // MM2S 0 send-back loop
      aie.use_lock(%cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%loopback_buf : memref<64xi32>, 0, 64)
      aie.use_lock(%prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}
