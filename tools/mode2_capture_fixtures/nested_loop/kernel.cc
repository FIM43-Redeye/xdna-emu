// Mode-2 fixture: nested loops (inner ZOL, outer software).
//
// LC is a single hardware register, so Peano can only put one loop in ZOL
// at a time. From the probe disassembly we know it picks the innermost
// loop for ZOL (movxm ls/le/add.nc lc) and uses a `add r,r,#-1; jnz`
// software loop for the outer.
//
// in[0] = outer trip count, in[1] = inner trip count. Host runs this with
// outer=4, inner=8.
//
// Trace expectations under the placeholder hypothesis:
//   - 4 entries into the inner ZOL (one per outer iteration)
//   - Each inner-loop activation produces its own LC frame sequence
//   - The last LC frame of each activation should carry bit-28 = 1
//   - The outer software loop touches no LC and should not appear in any
//     LC frame; its iterations should manifest only as PC progression
//     through the outer-loop body
//
// This fixture is the cleanest way to ask "does the bit-28 flag fire once
// per ZOL completion, or once per LC=0 boundary regardless of context?"
// since the outer software loop reactivates the same inner ZOL repeatedly.

#include <stdint.h>

extern "C" {

__attribute__((noinline))
void k_nested_loop(const int32_t *__restrict in, int32_t *__restrict out) {
  int32_t outer = in[0];
  int32_t inner = in[1];
  int32_t acc = 0;
  for (int32_t i = 0; i < outer; i++) {
    for (int32_t j = 0; j < inner; j++)
      acc += in[2 + i * inner + j];
  }
  out[0] = acc;
}

} // extern "C"
