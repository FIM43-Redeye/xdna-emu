//===- convexp.cc -----------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). bf16->f32 expand (fused VLDA.CONV.fp32.bf16, no FTZ), exhaustive bf16 denormal inputs. EXP = HW-captured silicon.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define CONV_N 256

extern "C" {

void convexp_bf16(bfloat16 *restrict in, float *restrict out) {
  event0();
  for (int i = 0; i < CONV_N; i += 16) {
    aie::vector<bfloat16, 16> v = aie::load_v<16>(in + i);
    v16accfloat raw = ::ups_to_v16accfloat((v16bfloat16)v);
    aie::accum<accfloat, 16> acc(raw);
    aie::vector<float, 16> o = acc.to_vector<float>();
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
