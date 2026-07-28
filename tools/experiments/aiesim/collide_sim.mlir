//===- producer_probe collide (aiesim / xcve2802 variant) ------*- MLIR -*-===//
// Compute tile (7,3) logic IDENTICAL to producer_probe.py collide, reduced
// (REPS=3, MARCH_N=400), with an explicit shim S2MM drain to a DDR
// external_buffer so the aiesim host flow (no NPU controller) has a real
// stream sink (else the compute-tile MM2S backpressure-stalls and confounds
// the core-vs-DMA bank-arbitration measurement).
//
// Device is xcve2802 (Versal AIE2): the aiesim + XAie-host flow is validated
// for it, and its compute-tile memory-bank arbitration ISS is architecturally
// identical to Phoenix AIE2 (npu1). npu1_2col rejects a shim at col 0 in the
// raw XAie model ("Invalid Tile Type"); the arbitration mechanism under study
// is device-independent, so this is faithful.
//
// march_buf @ 0x0400 (logical bank 0), dma_buf @ 0x2400 (logical bank 0).
// logical = (addr>>14)&3;  physical = 2*logical + ((addr>>4)&1).
//===----------------------------------------------------------------------===//
module {
  aie.device(xcve2802) {
    %shim_7_0 = aie.tile(7, 0)
    %core_7_3 = aie.tile(7, 3)

    %march_buf = aie.buffer(%core_7_3) {sym_name = "march_buf", address = 1024 : i32} : memref<2048xi32>
    %dma_buf   = aie.buffer(%core_7_3) {sym_name = "dma_buf",   address = 9216 : i32} : memref<256xi32>

    %lk_empty = aie.lock(%core_7_3, 0) {init = 1 : i32, sym_name = "lk_empty"}
    %lk_full  = aie.lock(%core_7_3, 1) {init = 0 : i32, sym_name = "lk_full"}

    // DDR sink for the drained stream (counting-semaphore locks per 04_shim_dma).
    %out_buf   = aie.external_buffer {sym_name = "output_buffer"} : memref<768xi32>
    %out_write = aie.lock(%shim_7_0, 1) {sym_name = "output_lock_write", init = 1 : i32}
    %out_read  = aie.lock(%shim_7_0, 2) {sym_name = "output_lock_read"}

    aie.flow(%core_7_3, DMA : 0, %shim_7_0, DMA : 0)

    %core = aie.core(%core_7_3) {
      %c0     = arith.constant 0 : index
      %c1     = arith.constant 1 : index
      %cREPS  = arith.constant 3 : index
      %cOBJ   = arith.constant 256 : index
      %cMARCH = arith.constant 400 : index

      scf.for %r = %c0 to %cREPS step %c1 {
        aie.use_lock(%lk_empty, AcquireGreaterEqual, 1)
        scf.for %i = %c0 to %cOBJ step %c1 {
          %iv = arith.index_cast %i : index to i32
          memref.store %iv, %dma_buf[%i] : memref<256xi32>
        }
        aie.use_lock(%lk_full, Release, 1)
        // DENSE MARCH-STORE: one store per cycle, marching contiguously,
        // alternating the two physical banks of logical bank 0 every 4 words.
        scf.for %hi = %c0 to %cMARCH step %c1 {
          %hv = arith.index_cast %hi : index to i32
          memref.store %hv, %march_buf[%hi] : memref<2048xi32>
        }
      }
      aie.end
    } {stack_size = 1024 : i32}

    // Self-looping single-BD MM2S: fires once per lk_full release.
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

    // Shim S2MM drains the whole 3*256-word stream into DDR in one BD.
    // Signals output_lock_read on completion; host waits on it.
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
