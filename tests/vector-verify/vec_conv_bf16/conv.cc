//===- conv.cc --------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). Convert (round-narrow), f32 -> bf16 through an accfloat accumulator. Config: rounding=conv_even (round-to-nearest, ties to even), normal finite inputs -- the matching slice of the Half-A `bf16_srs` golden (rnd=12). HW: crRnd governs to_v16bfloat16, so set_rounding selects the mode; the emulator's convert path honors it.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define CONV_N 448

extern "C" {

void conv_bf16(float *restrict in, bfloat16 *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int i = 0; i < CONV_N; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(in + i);
    aie::accum<accfloat, 16> acc(v);
    aie::vector<bfloat16, 16> o = acc.to_vector<bfloat16>();
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
