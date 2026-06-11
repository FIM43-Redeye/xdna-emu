//===- srs.cc ---------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). SRS (shift-round-saturate), int32 accumulator -> int16, shift=4. Mode-sweep point: rnd=12(conv_even), sat=3(symmetric).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define SRS_N 64
#define SRS_SHIFT 4

extern "C" {

void srs_i32(int32_t *restrict in, int16_t *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::symmetric);

  for (int i = 0; i < SRS_N; i += 16) {
    aie::vector<int32_t, 16> v = aie::load_v<16>(in + i);
    aie::accum<acc64, 16> acc;
    acc.from_vector(v, 0);
    aie::vector<int16_t, 16> o = acc.to_vector<int16_t>(SRS_SHIFT);
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
