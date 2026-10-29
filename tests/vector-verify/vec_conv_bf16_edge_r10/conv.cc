//===- conv.cc --------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). Convert edge (round-narrow), f32 -> bf16, denormal/NaN/Inf inputs. Mode-sweep point: rnd=10(symmetric_zero).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define CONV_N 96

extern "C" {

void conv_bf16(float *restrict in, bfloat16 *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::symmetric_zero);

  for (int i = 0; i < CONV_N; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(in + i);
    aie::accum<accfloat, 16> acc(v);
    aie::vector<bfloat16, 16> o = acc.to_vector<bfloat16>();
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
