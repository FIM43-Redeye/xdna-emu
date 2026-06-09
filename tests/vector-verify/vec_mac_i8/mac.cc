//===- mac.cc ---------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). MatMul (native mmul tile), int8 x int8 -> int32, 4x8x8. A batch of independent row-major tiles from the Half-A `matmul` golden (Int8/Int8, subtract=false); each C = A.B exactly (integer sum of products).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define MAC_BATCH 64

extern "C" {

void mac_i8(int8_t *restrict inA, int8_t *restrict inB, int32_t *restrict out) {
  event0();
  using MMUL = aie::mmul<4, 8, 8, int8, int8, accauto>;
  for (int n = 0; n < MAC_BATCH; n++) {
    aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(inA + n * MMUL::size_A);
    aie::vector<int8, MMUL::size_B> b = aie::load_v<MMUL::size_B>(inB + n * MMUL::size_B);
    MMUL m;
    m.mul(a, b);
    aie::store_v(out + n * MMUL::size_C, m.to_vector<int32>());
  }
  event1();
}

} // extern "C"
