//===- mac.cc ---------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). MatMul bf16 overflow tiles (4x8x4): result overflows to Inf/NaN. EXP is HW-captured silicon (model canonical NaN mantissa 0x7F vs silicon 1). Host stages bf16/fp32 bit patterns.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define MAC_BATCH 54

extern "C" {

void mac_bf16(bfloat16 *restrict inA, bfloat16 *restrict inB, float *restrict out) {
  event0();
  using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16, accauto>;
  for (int n = 0; n < MAC_BATCH; n++) {
    aie::vector<bfloat16, MMUL::size_A> a = aie::load_v<MMUL::size_A>(inA + n * MMUL::size_A);
    aie::vector<bfloat16, MMUL::size_B> b = aie::load_v<MMUL::size_B>(inB + n * MMUL::size_B);
    MMUL m;
    m.mul(a, b);
    aie::store_v(out + n * MMUL::size_C, m.to_vector<float>());
  }
  event1();
}

} // extern "C"
