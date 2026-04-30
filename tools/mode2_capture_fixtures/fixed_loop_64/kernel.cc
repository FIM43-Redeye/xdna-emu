// Mode-2 fixture: ZOL with fixed trip count = 64.
//
// Per the loop-shape probe (tools/mode2_loop_probes/probes.cc), Peano emits
// a vectorized ZOL for this shape: the loop body becomes width-4 SIMD stores
// and LC counts vector iterations, so the actual hardware LC value is 15
// (16 vector iterations -> LC = N-1 = 15).
//
// This fixture establishes the baseline trace pattern for a deterministic
// fixed-count ZOL. Comparing it against runtime_loop with n=64 should show
// identical mode-2 byte streams (or at least identical PC/LC frame counts),
// confirming that compile-time-known trip counts produce the same trace
// shape as runtime-known ones.

#include <stdint.h>

extern "C" {

__attribute__((noinline))
void k_fixed_loop_64(const int32_t *__restrict in, int32_t *__restrict out) {
  int32_t acc = 0;
  for (int32_t i = 0; i < 64; i++)
    acc += in[i];
  out[0] = acc;
}

} // extern "C"
