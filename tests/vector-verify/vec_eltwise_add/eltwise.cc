//===- eltwise.cc -----------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel: element-wise int32 vector add.
// Exercises the AIE2 vector add datapath (aie::add on v8int32) end-to-end
// (shim DMA -> core vector op -> shim DMA), so the bridge harness can compare
// real-NPU output against the emulator. Maps to the Half-A `vadd_Int32` golden
// (8 lanes of int32 = one 256-bit vector register).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void eltwise_add(int32_t *restrict a, int32_t *restrict b, int32_t *restrict c) {
  event0();
  aie::vector<int32_t, 8> va = aie::load_v<8>(a);
  aie::vector<int32_t, 8> vb = aie::load_v<8>(b);
  aie::vector<int32_t, 8> vc = aie::add(va, vb);
  aie::store_v(c, vc);
  event1();
}

} // extern "C"
