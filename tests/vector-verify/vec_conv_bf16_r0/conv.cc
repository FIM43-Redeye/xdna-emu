//===- conv.cc --------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). Convert (round-narrow), f32 -> bf16 through an accfloat accumulator, normal-finite inputs. Mode-sweep point: rnd=0(floor).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define CONV_N 448

extern "C" {

void conv_bf16(float *restrict in, bfloat16 *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::floor);

  for (int i = 0; i < CONV_N; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(in + i);
    aie::accum<accfloat, 16> acc(v);
    aie::vector<bfloat16, 16> o = acc.to_vector<bfloat16>();
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
