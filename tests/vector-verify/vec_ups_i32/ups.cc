//===- ups.cc ---------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). UPS (unpack-shift widen), int16 -> int32 accumulator. Config: signed, sat=none, shift=4 -- the matching slice of the Half-A `ups` golden.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define UPS_N 48
#define UPS_SHIFT 4

extern "C" {

void ups_i32(int16_t *restrict in, int32_t *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::floor);
  ::aie::set_saturation(aie::saturation_mode::none);

  for (int i = 0; i < UPS_N; i += 16) {
    aie::vector<int16_t, 16> v = aie::load_v<16>(in + i);
    aie::accum<acc32, 16> acc;
    acc.from_vector(v, UPS_SHIFT);
    aie::vector<int32_t, 16> o = acc.to_vector<int32_t>(0);
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
