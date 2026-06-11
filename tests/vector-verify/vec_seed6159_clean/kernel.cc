// Generated vector fuzz chain -- seed 6159, target elem_ins/Bf16x32/m0. DO NOT EDIT.
#include <stdint.h>
#include <aie_api/aie.hpp>
using V = aie::vector<bfloat16, 32>;
__attribute__((noinline)) static V h0(V a) { return aie::select(a, aie::broadcast<bfloat16, 32>(a.get(31)), aie::mask<32>::from_uint32(1u << 0)); }
__attribute__((noinline)) static V h1(V a, V b) { return aie::select(a, b, aie::lt(a, b)); }
__attribute__((noinline)) static V h2(V a) { return [&]{ auto t = a; t.insert(1, a.extract<16>(0)); return t; }(); }
__attribute__((noinline)) static V h3(V a, V b) { return aie::add(a, b); }
__attribute__((noinline)) static V h4(V a, V b) { return aie::add(a, b); }
__attribute__((noinline)) static V h5(V a, V b) { return aie::max(a, b); }
__attribute__((noinline)) static V h6(V a, V b) { return [&]{ aie::mmul<4, 8, 4, bfloat16, bfloat16, accauto> m; m.mul(a, b); m.mac(a, b); return aie::concat(m.to_vector<bfloat16>(), b.extract<16>(0)); }(); }
__attribute__((noinline)) static V h7(V a, V b) { return aie::select(a, b, aie::ge(a, b)); }
extern "C" void fuzz_kernel(int32_t* __restrict in, int32_t* __restrict out) {
  aie::vector<bfloat16, 32> v0_in = aie::load_v<32>((bfloat16*)(in + 0));
  auto v0 = h0(v0_in);
  aie::store_v((bfloat16*)(out + 0), v0);
  aie::vector<bfloat16, 32> p1 = aie::load_v<32>((bfloat16*)(in + 16));
  auto v1 = h1(v0, p1);
  aie::store_v((bfloat16*)(out + 16), v1);
  auto v2 = h2(v1);
  aie::store_v((bfloat16*)(out + 32), v2);
  aie::vector<bfloat16, 32> p2 = aie::load_v<32>((bfloat16*)(in + 32));
  auto v3 = h3(v2, p2);
  aie::store_v((bfloat16*)(out + 48), v3);
  aie::vector<bfloat16, 32> p3 = aie::load_v<32>((bfloat16*)(in + 48));
  auto v4 = h4(v3, p3);
  aie::store_v((bfloat16*)(out + 64), v4);
  aie::vector<bfloat16, 32> p4 = aie::load_v<32>((bfloat16*)(in + 64));
  auto v5 = h5(v4, p4);
  aie::store_v((bfloat16*)(out + 80), v5);
  aie::vector<bfloat16, 32> p5 = aie::load_v<32>((bfloat16*)(in + 80));
  auto v6 = h6(v5, p5);
  aie::store_v((bfloat16*)(out + 96), v6);
  aie::vector<bfloat16, 32> p6 = aie::load_v<32>((bfloat16*)(in + 96));
  auto v7 = h7(v6, p6);
  aie::store_v((bfloat16*)(out + 112), v7);
}
