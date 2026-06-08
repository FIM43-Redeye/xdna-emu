//===- srs.cc ---------------------------------------------------*- C++ -*-===//
//
// Half-B vector-compute capture kernel: SRS (shift-round-saturate),
// accumulator -> narrower vector. Loads int32 accumulator values, builds an
// acc64 accumulator (the inverse UPS at shift 0 is a plain widen), then applies
// the SRS datapath to int16 with a fixed config: rounding = floor, saturation =
// saturate, user-shift = 4. This is the exact (rnd=0 FLOOR, sat, sym_sat=false,
// bits_o=16, shift=4) slice of the Half-A `srs` golden.
//
// The SRS hardware left-shifts by SRS_SHIFT_BIAS before the user shift to
// provide guard/round precision; `to_vector<int16>(4)` therefore round-floors
// with that bias, exactly matching the emulator's srs_lane. A passing HW==EMU
// bridge is the silicon evidence that the BIAS + floor + saturate path is real.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

// Process the batch in full 256-bit-vector chunks (16x int32). N % 16 == 0.
//
// Note: Chess vectorizes int32->int16 SRS identically regardless of this lane
// grouping -- it emits `vlda.ups.s64.s32 bmlN` (8-lane low-half loads),
// `vsrs.s16.s64 wX, cmN` (full-cm SRS), and 128-bit `q`-register stores
// (`vmov qN, wX; st qN`). That q-register-as-vector-data path is what the
// emulator must model for this kernel to execute (the SRS arithmetic itself is
// Half-A verified); the HW leg compares against the baked model golden.
#ifndef SRS_N
#define SRS_N 48
#endif
#define SRS_SHIFT 4

extern "C" {

void srs_i32(int32_t *restrict in, int16_t *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::floor);
  ::aie::set_saturation(aie::saturation_mode::saturate);

  for (int i = 0; i < SRS_N; i += 16) {
    aie::vector<int32_t, 16> v = aie::load_v<16>(in + i);
    aie::accum<acc64, 16> acc;
    acc.from_vector(v, 0);
    aie::vector<int16_t, 16> o = acc.to_vector<int16_t>(SRS_SHIFT);
    aie::store_v(out + i, o);
  }
  event1();
}

} // extern "C"
