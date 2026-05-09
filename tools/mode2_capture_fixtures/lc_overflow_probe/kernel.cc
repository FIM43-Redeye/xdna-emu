// Mode-2 fixture: LC bit-28 overflow probe with heavy_zol-style wrapper.
//
// Probes what happens to the LC frame's bit-28 flag and 28-bit count field
// when the trip count straddles the 28-bit boundary at 2^28.
// The aie.core wrapper calls k_pass 4 times per invocation; each k_pass
// executes one ZOL of trip count N (read from in[0]). Phase 0 found the
// trace shim DMA only flushes after enough atom volume; the 4-pass
// wrapper guarantees that even at small N we see a real trace, and at
// high N each pass is one ZOL invocation -> exactly 4 LC frames per
// capture (= built-in redundancy for noise rejection).
//
// Body matches runtime_loop's exact shape (acc += in[i+1]) -- this is the
// only shape Phase 0 / runtime_loop confirmed Chess actually compiles into
// `mov lc, r25 (= N)` instead of folding to a small constant. See the
// in-tree disassembly notes if you change the body and the LC stops
// scaling with N.

#include <stdint.h>

extern "C" {

__attribute__((noinline))
void k_pass(const int32_t *__restrict in, int32_t *__restrict out) {
  int32_t n = in[0];
  int32_t acc = 0;
  for (int32_t i = 0; i < n; i++)
    acc += in[i + 1];
  // Write back so the call can't be DCE'd.
  out[0] = acc;
}

} // extern "C"
