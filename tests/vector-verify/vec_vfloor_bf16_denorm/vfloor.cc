//===- vfloor.cc ------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). bf16->int32 floor (standalone VFLOOR.s32.bf16), exhaustive bf16 denormal inputs. Routes through vector_convert's FTZ (vector_convert.rs:343): neg denormal floors to -1 without FTZ vs 0 with. EXP = HW-captured silicon.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define VFL_N 256

extern "C" {

void vfloor_bf16(bfloat16 *restrict in, int32_t *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::floor);
  for (int i = 0; i < VFL_N; i += 16) {
    aie::vector<bfloat16, 16> v = aie::load_v<16>(in + i);
    v16int32 o = ::bfloat16_to_int((v16bfloat16)v, 0);
    aie::store_v(out + i, (aie::vector<int32, 16>)o);
  }
  event1();
}

} // extern "C"
