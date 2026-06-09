//===- pack.cc --------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). Pack (native VPACK), int16 -> int8 truncating narrow. Native pack takes the low 8 bits (it does NOT saturate, regardless of the saturation mode), so this matches the Half-A `pack` golden slice (bits_i=16, bits_o=8, signed, sat=false).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define PACK_N 32

extern "C" {

void pack_i16(int16_t *restrict in, int8_t *restrict out) {
  event0();
  ::aie::set_saturation(aie::saturation_mode::none);

  for (int i = 0; i < PACK_N; i += 32) {
    aie::vector<int16_t, 32> v = aie::load_v<32>(in + i);
    aie::vector<int8_t, 32> o = v.pack<int8_t>();
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
