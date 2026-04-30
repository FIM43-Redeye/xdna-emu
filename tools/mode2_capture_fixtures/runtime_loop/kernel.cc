// Mode-2 fixture: ZOL with runtime trip count.
//
// in[0] holds the trip count N (set by the host before each run). The kernel
// reads N at runtime so Peano cannot fold it to a constant; the resulting
// loop is a true ZOL with `add.nc lc, <reg>, #-0x1` initialization, where
// the LC register decrements from N-1 to 0.
//
// The Phase 0 hypothesis under test is that the LC frame's bit-28 flag is
// set (=1) on the LC frame whose final atom corresponds to the iteration
// where lc reaches 0 - i.e. the last iteration of the loop. The host runs
// this kernel with N = 1, 2, 4, 8 so we can compare flag emission across
// loop lengths.
//
// The accumulation does real work so the loop body cannot be optimized away.

#include <stdint.h>

extern "C" {

__attribute__((noinline))
void k_runtime_loop(const int32_t *__restrict in, int32_t *__restrict out) {
  int32_t n = in[0];
  int32_t acc = 0;
  for (int32_t i = 0; i < n; i++)
    acc += in[i + 1];
  out[0] = acc;
}

} // extern "C"
