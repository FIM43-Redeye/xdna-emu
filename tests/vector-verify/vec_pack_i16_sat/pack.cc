//===- pack.cc --------------------------------------------------*- C++ -*-===//
//
// Half-B verification kernel (saturating pack). Companion to vec_pack_i16
// (truncating); identical except set_saturation(saturate). Confirmed against
// real NPU1 silicon that AIE2 narrowing pack honors crSat (VPACK/vst.pack
// `Uses = [crSat]`): HW saturates int16->int8 to [-128,127], it does not take
// the low byte. Locks the emulator's crSat-derived pack mode -- fixed a gap
// where the fused vst.pack path (memory/mod.rs) hardcoded truncation.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define PACK_N 32

extern "C" {

void pack_i16(int16_t *restrict in, int8_t *restrict out) {
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);

  for (int i = 0; i < PACK_N; i += 32) {
    aie::vector<int16_t, 32> v = aie::load_v<32>(in + i);
    aie::vector<int8_t, 32> o = v.pack<int8_t>();
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
