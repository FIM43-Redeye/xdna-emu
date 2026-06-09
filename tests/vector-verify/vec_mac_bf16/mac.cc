//===- mac.cc ---------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). MatMul (native mmul tile), bf16 x bf16 -> fp32, 4x8x4. A batch of independent row-major tiles from the Half-A `matmul` golden (BFloat16/BFloat16, subtract=false), all-finite expected; each C = A.B in fp32. Host stages bf16/fp32 bit patterns; the kernel reads bfloat16/float.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define MAC_BATCH 24

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
