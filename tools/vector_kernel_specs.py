"""Registry of Half-B vector-compute capture-kernel specs.

Each entry is a `KernelSpec` the generator (gen_vector_kernel.py) turns into the
four files a bridge kernel needs. The `body` field is the genuine intrinsic IP,
hand-written once per class; everything else -- scaffold, MLIR design, host
harness, golden arrays -- is derived. To add a kernel, append a spec here and
run `python3 tools/gen_vector_kernel.py <name>`.

The golden slice in each spec must name a config that exists in
tools/golden/vector_ops.json (the Half-A corpus); the generator bakes the
input/expected arrays from it so no expected value is ever transcribed by hand.
"""

from gen_vector_kernel import Buf, KernelSpec

SPECS = {}


def _reg(spec):
    SPECS[spec.name] = spec
    return spec


# --- SRS: accumulator (acc64) -> narrower vector (int16), shift-round-saturate.
# This is the committed vec_srs_i32 kernel expressed as a spec; the generator
# regenerates it bit-for-bit on the golden arrays (validation anchor).
_reg(KernelSpec(
    name="vec_srs_i32",
    func="srs_i32",
    doc="SRS (shift-round-saturate), int32 accumulator -> int16. Config: "
        "rnd=FLOOR, sat=saturate, sym_sat=false, shift=4 -- the matching slice "
        "of the Half-A `srs` golden.",
    inputs=[Buf("in", "int32_t", "i32")],
    output=Buf("out", "int16_t", "i16"),
    n=48,
    golden={
        "class": "srs",
        "filt": {"bits_o": 16, "signed": True, "sat": True,
                 "sym_sat": False, "rnd": 0, "shift": 4},
        "value_range": (-(2 ** 31), 2 ** 31 - 1),
    },
    defines=[("SRS_N", 48), ("SRS_SHIFT", 4)],
    body="""  event0();
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
""",
))


# --- UPS: narrow vector (int16) -> accumulator (acc32) with left-shift, the
# inverse widen of SRS. `from_vector(v, shift)` is the UPS intrinsic; the acc32
# lanes are extracted to int32 via to_vector<int32>(0) (an SRS-0 reinterpret).
# Config: signed, sat=false, shift=4 -- expected = value<<4, all int32-fitting.
_reg(KernelSpec(
    name="vec_ups_i32",
    func="ups_i32",
    doc="UPS (unpack-shift widen), int16 -> int32 accumulator. Config: signed, "
        "sat=none, shift=4 -- the matching slice of the Half-A `ups` golden.",
    inputs=[Buf("in", "int16_t", "i16")],
    output=Buf("out", "int32_t", "i32"),
    n=48,
    golden={
        "class": "ups",
        "filt": {"bits_in": 16, "bits_out": 32, "signed": True,
                 "sat": False, "shift": 4},
        "value_range": (-(2 ** 15), 2 ** 15 - 1),
    },
    defines=[("UPS_N", 48), ("UPS_SHIFT", 4)],
    body="""  event0();
  ::aie::set_rounding(aie::rounding_mode::floor);
  ::aie::set_saturation(aie::saturation_mode::none);

  for (int i = 0; i < UPS_N; i += 16) {
    aie::vector<int16_t, 16> v = aie::load_v<16>(in + i);
    aie::accum<acc32, 16> acc;
    acc.from_vector(v, UPS_SHIFT);
    aie::vector<int32_t, 16> o = acc.to_vector<int32_t>(0);
    aie::store_v(out + i, o);
  }
  event1();
""",
))
