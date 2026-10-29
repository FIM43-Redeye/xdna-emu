//===- shuffle.cc -----------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). VSHUFFLE of two 512-bit operands under a table of runtime mode numbers (named controls, then 44..63). Output block k is the shuffle under mode k of the table.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define SHUF_MODES 25

extern "C" {

void shuffle_mode(int32_t *restrict in, int32_t *restrict out) {
  event0();
  static const unsigned modes[SHUF_MODES] = {INTLV_lo_32o64, INTLV_hi_32o64, DINTLV_lo_32o64, DINTLV_hi_32o64, T32_4x4, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  aie::vector<int32_t, 16> a = aie::load_v<16>(in);
  aie::vector<int32_t, 16> b = aie::load_v<16>(in + 16);
  for (int k = 0; k < SHUF_MODES; k++) {
    v16int32 r = ::shuffle((v16int32)a, (v16int32)b, modes[k]);
    aie::store_v(out + 16 * k, (aie::vector<int32_t, 16>)r);
  }
  event1();
}

} // extern "C"
