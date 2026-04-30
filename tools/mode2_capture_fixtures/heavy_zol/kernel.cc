// Mode-2 fixture: heavy ZOL activity for clean LC-frame analysis.
//
// The aie.core wrapper (in aie.mlir) calls k_pass repeatedly. Each k_pass
// runs a runtime-count ZOL loop with N read from in[0]. With 64 outer
// passes and N=8 inner iterations per pass, we get 64 ZOL completions in
// a single trace window - enough mode-2 activity to be sure we're seeing
// the LC frame's bit-28 behavior across many invocations rather than a
// single isolated event.
//
// Single-tile by construction: no memtile, no multi-source packet
// interleave in the trace BO. This makes the decoded LC frames trivial
// to attribute and lets us test the placeholder hypothesis directly:
// bit-28 == 1 iff lc_after == 0 (the iteration where loop count
// reaches zero). With 64 passes, we expect 64 instances of "last LC
// frame in pass" and (64 * (N-1)) instances of "intermediate LC frame".

#include <stdint.h>

extern "C" {

__attribute__((noinline))
void k_pass(const int32_t *__restrict in, int32_t *__restrict out) {
  int32_t n = in[0];
  int32_t acc = 0;
  for (int32_t i = 0; i < n; i++)
    acc += in[i + 1];
  // Write back so the call cannot be DCE'd.
  out[0] = acc;
}

} // extern "C"
