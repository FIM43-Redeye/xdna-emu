//===- test.cpp - aiesim host for producer_probe collide -------*- C++ -*-===//
//
// Host PS program for AMD's cycle-accurate aiesimulator. Configures the
// compute tile (0,2) -- a dense march-store core + a self-looping MM2S draining
// dma_buf -- and a shim S2MM that drains the streamed result into DDR. The core
// and its tile MM2S both self-run on locks; the host only configures, arms the
// shim, starts the core, and waits (polling a lock so sim time advances) for
// the drain to finish, then verifies the DDR sink received the streamed data.
//
// Adapted from mlir-aie chess_compiler_tests_aie2/04_shim_dma_kernel/test.cpp.
// This is original code; it only calls the generated XAie configuration API.
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <xaiengine.h>

#include "memory_allocator.h"
#include "test_library.h"

#include "aie_inc.cpp"

#define DMA_WORDS 768 // 3 reps * 256 words

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie); // compute-tile (0,2) MM2S

  printf("after DMA config\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  int errors = 0;

  // DDR sink buffer.
  ext_mem_model_t buf_out;
  int *ddr_ptr_out = mlir_aie_mem_alloc(_xaie, buf_out, DMA_WORDS);
  for (int i = 0; i < DMA_WORDS; i++)
    ddr_ptr_out[i] = -1;
  mlir_aie_sync_mem_dev(buf_out);

  mlir_aie_external_set_addr_output_buffer(_xaie, (u64)ddr_ptr_out);
  mlir_aie_configure_shimdma_70(_xaie); // shim (0,0) S2MM -> DDR

  printf("before core start\n");
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  // Wait for the shim S2MM to finish draining all DMA_WORDS. The acquire polls
  // the lock register, which advances sim time; loop so the ~2000-cycle compute
  // + drain completes before the PS returns (PS return ends the simulation).
  printf("Waiting for drain to complete...\n");
  int done = 0;
  for (int attempt = 0; attempt < 200; attempt++) {
    if (mlir_aie_acquire_output_lock_read(_xaie, 1, 1000) == XAIE_OK) {
      done = 1;
      break;
    }
  }
  if (!done) {
    printf("TIMEOUT waiting for output_lock_read\n");
    errors++;
  }

  printf("after drain\n");
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  // Verify the DDR sink received the streamed pattern: each 256-word transfer
  // carries dma_buf = 0,1,...,255 (core stores %iv = i). 3 reps => that pattern
  // three times.
  mlir_aie_sync_mem_cpu(buf_out);
  int nonzero = 0;
  for (int i = 0; i < DMA_WORDS; i++)
    if (ddr_ptr_out[i] != -1)
      nonzero++;
  printf("DDR sink: %d/%d words written\n", nonzero, DMA_WORDS);
  for (int i = 0; i < 8; i++)
    printf("ddr_ptr_out[%d] = %d\n", i, ddr_ptr_out[i]);
  for (int rep = 0; rep < 3; rep++) {
    mlir_aie_check("DDR out[0]", ddr_ptr_out[rep * 256 + 0], 0, errors);
    mlir_aie_check("DDR out[1]", ddr_ptr_out[rep * 256 + 1], 1, errors);
    mlir_aie_check("DDR out[255]", ddr_ptr_out[rep * 256 + 255], 255, errors);
  }

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }
  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.\n");
  return res;
}
