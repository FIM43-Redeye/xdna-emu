//===- producer_probe collide-READ (aiesim / xcve2802) ---------*- MLIR -*-===//
// TASK 2 forcing variant: the core densely READS dma_buf (the SAME buffer the
// MM2S is draining) during the drain window -> guaranteed same-bank READ-vs-READ
// contention in the same cycles. Control (collide_sim.mlir) had the core WRITE a
// same-bank buffer and saw zero conflict; if read-vs-read fires conflict, the ISS
// separates the write port from the read port.
//
// Also: a hi_buf @ 0x8000 gives a task-1 datapoint for the high bank bits
// (emulator: (0x8000>>14)&3 = 2 -> bank-pair {b4,b5}).
//===----------------------------------------------------------------------===//
module {
  aie.device(xcve2802) {
    %shim_7_0 = aie.tile(7, 0)
    %core_7_3 = aie.tile(7, 3)

    %dma_buf = aie.buffer(%core_7_3) {sym_name = "dma_buf", address = 9216 : i32} : memref<256xi32>   // 0x2400
    %hi_buf  = aie.buffer(%core_7_3) {sym_name = "hi_buf",  address = 32768 : i32} : memref<256xi32>   // 0x8000
    %acc_buf = aie.buffer(%core_7_3) {sym_name = "acc_buf", address = 40960 : i32} : memref<8xi32>     // 0xA000, sink

    %lk_empty = aie.lock(%core_7_3, 0) {init = 1 : i32, sym_name = "lk_empty"}
    %lk_full  = aie.lock(%core_7_3, 1) {init = 0 : i32, sym_name = "lk_full"}

    %out_buf   = aie.external_buffer {sym_name = "output_buffer"} : memref<768xi32>
    %out_write = aie.lock(%shim_7_0, 1) {sym_name = "output_lock_write", init = 1 : i32}
    %out_read  = aie.lock(%shim_7_0, 2) {sym_name = "output_lock_read"}

    aie.flow(%core_7_3, DMA : 0, %shim_7_0, DMA : 0)

    %core = aie.core(%core_7_3) {
      %c0    = arith.constant 0 : index
      %c1    = arith.constant 1 : index
      %cREPS = arith.constant 3 : index
      %cOBJ  = arith.constant 256 : index
      %cRD   = arith.constant 6 : index      // read-sweeps per rep (dense, outlasts drain)

      scf.for %r = %c0 to %cREPS step %c1 {
        aie.use_lock(%lk_empty, AcquireGreaterEqual, 1)
        scf.for %i = %c0 to %cOBJ step %c1 {
          %iv = arith.index_cast %i : index to i32
          memref.store %iv, %dma_buf[%i] : memref<256xi32>
        }
        aie.use_lock(%lk_full, Release, 1)
        // CONCURRENT DENSE READS of dma_buf -- same buffer/bank the MM2S drains.
        %sum0 = arith.constant 0 : i32
        %acc = scf.for %s = %c0 to %cRD step %c1 iter_args(%a0 = %sum0) -> (i32) {
          %ai = scf.for %i = %c0 to %cOBJ step %c1 iter_args(%a1 = %a0) -> (i32) {
            %v = memref.load %dma_buf[%i] : memref<256xi32>
            %n = arith.addi %a1, %v : i32
            scf.yield %n : i32
          }
          scf.yield %ai : i32
        }
        memref.store %acc, %acc_buf[%c0] : memref<8xi32>
      }
      // touch hi_buf once so its bank shows on a port (task-1 datapoint)
      %hv = memref.load %hi_buf[%c0] : memref<256xi32>
      memref.store %hv, %acc_buf[%c1] : memref<8xi32>
      aie.end
    } {stack_size = 1024 : i32}

    %mem = aie.mem(%core_7_3) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lk_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%dma_buf : memref<256xi32>, 0, 256) {bd_id = 0 : i32}
      aie.use_lock(%lk_empty, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    %shimdma = aie.shim_dma(%shim_7_0) {
      aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%out_write, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buf : memref<768xi32>, 0, 768)
      aie.use_lock(%out_read, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}
