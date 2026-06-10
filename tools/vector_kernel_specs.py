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

from gen_vector_kernel import Buf, KernelSpec, Matmul, SweepSpec

SPECS = {}

# Mode-sweep families (Half-B phase A). Each SweepSpec.expand()s into one
# capture kernel per reachable crRnd/crSat point so the silicon check is
# mode-exhaustive, not one-representative-per-class. Keyed by sweep name; the
# generator resolves a sweep name (or 'all-sweeps') to its expanded KernelSpecs.
SWEEPS = {}


def _reg(spec):
    SPECS[spec.name] = spec
    return spec


def _reg_sweep(sweep):
    SWEEPS[sweep.prefix + "_sweep"] = sweep
    return sweep


# Reachable mode indices (verified against the toolchain; see gen_vector_kernel
# ROUNDING_ENUM / SAT_ENUM). crRnd valid indices are 0-3 and 8-13; crSat is
# none(0)/saturate(1)/symmetric(3).
_ALL_RND = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13]
_ALL_SAT = [0, 1, 3]


def _is_normal_f32(rec):
    """The record's f32 bit pattern (in `value`) is a normal finite number.

    True iff the biased exponent is in 1..254 -- i.e. not a denormal (exp=0)
    and not Inf/NaN (exp=255). Those edges are deliberately excluded from the
    conv capture kernel: the bf16_srs model does NOT flush denormal inputs to
    zero (e.g. 0x10000 -> 1) whereas the execute path's input FTZ would, and
    NaN canonicalization differs between the model and NPU1 silicon (the Half-A
    bf16 NaN finding). Restricting to normals isolates the rounding datapath --
    the actual subject of the kernel -- from those separate HW-gated questions.
    """
    exp = (rec["value"] >> 23) & 0xFF
    return 1 <= exp <= 254


def _all_expected_finite_f32(rec):
    """Every lane of the record's `expected` (fp32 bit patterns) is finite.

    True iff no expected lane is NaN or Inf (biased exponent != 255). Used by
    the bf16 matmul capture to exclude overflow tiles: an Inf result is bit-
    comparable but a NaN's bit pattern differs between the aietools model and
    NPU1 silicon (the Half-A bf16 NaN finding), which would break the exact
    bit-compare. Restricting to all-finite tiles isolates the mmul accumulate/
    round datapath -- the subject of the kernel -- from the overflow edge.
    """
    return all(((bits >> 23) & 0xFF) != 0xFF for bits in rec["expected"])


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


# --- Pack: native VPACK, int16 -> int8 (halving narrow). int32->int16 pack()
# falls back to the SRS accumulator path (would re-test srs), so this targets
# the int16->int8 native pack -- a distinct datapath. Config: signed, sat -- the
# matching slice of the Half-A `pack` golden (bits_i=16, bits_o=8).
_reg(KernelSpec(
    name="vec_pack_i16",
    func="pack_i16",
    doc="Pack (native VPACK), int16 -> int8 truncating narrow. Native pack "
        "takes the low 8 bits (it does NOT saturate, regardless of the "
        "saturation mode), so this matches the Half-A `pack` golden slice "
        "(bits_i=16, bits_o=8, signed, sat=false).",
    inputs=[Buf("in", "int16_t", "i16")],
    output=Buf("out", "int8_t", "i8"),
    n=32,
    golden={
        "class": "pack",
        "filt": {"bits_i": 16, "bits_o": 8, "signed": True,
                 "sat": False, "symsat": False},
        "value_range": (-(2 ** 15), 2 ** 15 - 1),
    },
    defines=[("PACK_N", 32)],
    body="""  event0();
  ::aie::set_saturation(aie::saturation_mode::none);

  for (int i = 0; i < PACK_N; i += 32) {
    aie::vector<int16_t, 32> v = aie::load_v<32>(in + i);
    aie::vector<int8_t, 32> o = v.pack<int8_t>();
    aie::store_v(out + i, o);
  }
  event1();
""",
))


# --- Convert: round-narrow f32 -> bf16 via an accfloat accumulator. The host
# stages exact f32/bf16 bit patterns (uint32/uint16); the kernel reads them as
# float/bfloat16. `accum<accfloat>(v).to_vector<bfloat16>()` is the SRS-style
# round-narrow (Chess `to_v16bfloat16`), governed by the configured rounding
# mode (crRnd). We pick conv_even (round-to-nearest, ties to even): unlike
# Floor it rounds positive values up, so a positive batch already distinguishes
# rounding from bit-truncation. Slice = all normal-finite inputs of the Half-A
# `bf16_srs` golden at rnd=12 (both signs), padded to a 16-multiple.
_reg(KernelSpec(
    name="vec_conv_bf16",
    func="conv_bf16",
    doc="Convert (round-narrow), f32 -> bf16 through an accfloat accumulator. "
        "Config: rounding=conv_even (round-to-nearest, ties to even), normal "
        "finite inputs -- the matching slice of the Half-A `bf16_srs` golden "
        "(rnd=12). HW: crRnd governs to_v16bfloat16, so set_rounding selects "
        "the mode; the emulator's convert path honors it.",
    inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
    output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
    n=448,
    golden={
        "class": "bf16_srs",
        "filt": {"rnd": 12},
        "predicate": _is_normal_f32,
    },
    defines=[("CONV_N", 448)],
    body="""  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int i = 0; i < CONV_N; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(in + i);
    aie::accum<accfloat, 16> acc(v);
    aie::vector<bfloat16, 16> o = acc.to_vector<bfloat16>();
    aie::store_v(out + i, o);
  }
  event1();
""",
))


# --- MatMul: native AIE2 mmul tile, int8 x int8 -> int32 accumulator, 4x8x8.
# First of the MAC tier and the first COMPILED matmul to run through the
# decode->execute interpreter (Half-A verified mmul arithmetic via hand-built
# SlotOps only). A batch of independent row-major tiles is unpacked from the
# Half-A `matmul` golden; each C = A.B exactly (integer sum of products, i32
# wrap), so the comparison is exact. The aie::mmul class handles the register-
# level A/B packing internally -- the kernel just DMAs plain row-major matrices.
_reg(KernelSpec(
    name="vec_mac_i8",
    func="mac_i8",
    doc="MatMul (native mmul tile), int8 x int8 -> int32, 4x8x8. A batch of "
        "independent row-major tiles from the Half-A `matmul` golden (Int8/Int8, "
        "subtract=false); each C = A.B exactly (integer sum of products).",
    inputs=[Buf("inA", "int8_t", "i8"), Buf("inB", "int8_t", "i8")],
    output=Buf("out", "int32_t", "i32"),
    n=0,
    golden={"class": "matmul",
            "filt": {"a_type": "Int8", "b_type": "Int8", "rows": 4,
                     "inner": 8, "cols": 8, "subtract": False,
                     "x_signed": True, "y_signed": True}},
    matmul=Matmul(M=4, K=8, N=8, a_bytes=1, b_bytes=1, batch=48),
    defines=[("MAC_BATCH", 48)],
    body="""  event0();
  using MMUL = aie::mmul<4, 8, 8, int8, int8, accauto>;
  for (int n = 0; n < MAC_BATCH; n++) {
    aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(inA + n * MMUL::size_A);
    aie::vector<int8, MMUL::size_B> b = aie::load_v<MMUL::size_B>(inB + n * MMUL::size_B);
    MMUL m;
    m.mul(a, b);
    aie::store_v(out + n * MMUL::size_C, m.to_vector<int32>());
  }
  event1();
""",
))


# --- MatMul: native AIE2 mmul tile, int16 x int16 -> int32 accumulator, 4x2x8.
# Same compiled-mmul datapath as vec_mac_i8 with a different element width and
# tile shape (K=2, N=8). The acc32 accumulator wraps at int32 (no saturation),
# so the golden `expected` -- taken verbatim -- already carries the wrapped
# lanes the kernel produces; the comparison stays exact.
_reg(KernelSpec(
    name="vec_mac_i16",
    func="mac_i16",
    doc="MatMul (native mmul tile), int16 x int16 -> int32, 4x2x8. A batch of "
        "independent row-major tiles from the Half-A `matmul` golden (Int16/"
        "Int16, subtract=false); each C = A.B with int32 accumulator wrap.",
    inputs=[Buf("inA", "int16_t", "i16"), Buf("inB", "int16_t", "i16")],
    output=Buf("out", "int32_t", "i32"),
    n=0,
    golden={"class": "matmul",
            "filt": {"a_type": "Int16", "b_type": "Int16", "rows": 4,
                     "inner": 2, "cols": 8, "subtract": False,
                     "x_signed": True, "y_signed": True}},
    matmul=Matmul(M=4, K=2, N=8, a_bytes=2, b_bytes=2, batch=48),
    defines=[("MAC_BATCH", 48)],
    body="""  event0();
  using MMUL = aie::mmul<4, 2, 8, int16, int16, accauto>;
  for (int n = 0; n < MAC_BATCH; n++) {
    aie::vector<int16, MMUL::size_A> a = aie::load_v<MMUL::size_A>(inA + n * MMUL::size_A);
    aie::vector<int16, MMUL::size_B> b = aie::load_v<MMUL::size_B>(inB + n * MMUL::size_B);
    MMUL m;
    m.mul(a, b);
    aie::store_v(out + n * MMUL::size_C, m.to_vector<int32>());
  }
  event1();
""",
))


# --- MatMul: native AIE2 mmul tile, bf16 x bf16 -> fp32 accumulator, 4x8x4.
# The float MAC tier: bf16 inputs, fp32 (accfloat) accumulator, exact fp32-bit
# output compare. The host stages bf16/fp32 bit patterns (uint16/uint32); the
# kernel reads bfloat16/float. Restricted to all-finite-expected tiles
# (_all_expected_finite_f32): a NaN result's bits differ between the model and
# NPU1 silicon (Half-A bf16 NaN finding), so overflow tiles are excluded to keep
# the bit-compare exact and isolate the mmul accumulate/round datapath.
_reg(KernelSpec(
    name="vec_mac_bf16",
    func="mac_bf16",
    doc="MatMul (native mmul tile), bf16 x bf16 -> fp32, 4x8x4. A batch of "
        "independent row-major tiles from the Half-A `matmul` golden (BFloat16/"
        "BFloat16, subtract=false), all-finite expected; each C = A.B in fp32. "
        "Host stages bf16/fp32 bit patterns; the kernel reads bfloat16/float.",
    inputs=[Buf("inA", "uint16_t", "bf16", ktype="bfloat16"),
            Buf("inB", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "uint32_t", "f32", ktype="float"),
    n=0,
    golden={"class": "matmul",
            "filt": {"a_type": "BFloat16", "b_type": "BFloat16", "rows": 4,
                     "inner": 8, "cols": 4, "subtract": False},
            "predicate": _all_expected_finite_f32},
    matmul=Matmul(M=4, K=8, N=4, a_bytes=2, b_bytes=2, batch=24, bfloat=True),
    defines=[("MAC_BATCH", 24)],
    body="""  event0();
  using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16, accauto>;
  for (int n = 0; n < MAC_BATCH; n++) {
    aie::vector<bfloat16, MMUL::size_A> a = aie::load_v<MMUL::size_A>(inA + n * MMUL::size_A);
    aie::vector<bfloat16, MMUL::size_B> b = aie::load_v<MMUL::size_B>(inB + n * MMUL::size_B);
    MMUL m;
    m.mul(a, b);
    aie::store_v(out + n * MMUL::size_C, m.to_vector<float>());
  }
  event1();
""",
))


# === Mode-sweep families (Half-B phase A: mode-exhaustive on silicon) =========
#
# Each sweep mirrors a single-mode anchor above but parametrizes the swept axes.
# The anchor stays as an independent generator-reproduction check; the sweep is
# the HW campaign covering every reachable mode-point of its class. Buffer sizes
# are set to the largest mode slice (padded points are 0 -> 0 under every mode):
# srs 50, pack 66, convert 438-normals, ups 45.

# --- SRS sweep: crRnd x crSat over the acc64 -> int16 shift-round-saturate
# datapath. 10 rounding modes x 3 saturation modes = 30 points -- the full
# discrete mode space the integer SRS pipeline can be configured into.
_reg_sweep(SweepSpec(
    prefix="vec_srs_i32",
    func="srs_i32",
    doc="SRS (shift-round-saturate), int32 accumulator -> int16, shift=4.",
    inputs=[Buf("in", "int32_t", "i32")],
    output=Buf("out", "int16_t", "i16"),
    n=64,
    gclass="srs",
    base_filt={"bits_o": 16, "signed": True, "shift": 4},
    rnds=_ALL_RND,
    sats=_ALL_SAT,
    sat_field="sat",
    symsat_field="sym_sat",
    value_range=(-(2 ** 31), 2 ** 31 - 1),
    defines=[("SRS_N", 64), ("SRS_SHIFT", 4)],
    body_template="""  event0();
$mode

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


# --- Pack sweep: crSat over native VPACK int16 -> int8. 3 saturation modes.
# Pack does not round (no crRnd dependence), so only the saturation axis varies;
# this is the exact axis the pack-saturation gap hid behind (model truncated,
# silicon saturates -- crSat=1/3).
_reg_sweep(SweepSpec(
    prefix="vec_pack_i16",
    func="pack_i16",
    doc="Pack (native VPACK), int16 -> int8 narrowing.",
    inputs=[Buf("in", "int16_t", "i16")],
    output=Buf("out", "int8_t", "i8"),
    n=96,
    gclass="pack",
    base_filt={"bits_i": 16, "bits_o": 8, "signed": True},
    sats=_ALL_SAT,
    sat_field="sat",
    symsat_field="symsat",
    value_range=(-(2 ** 15), 2 ** 15 - 1),
    defines=[("PACK_N", 96)],
    body_template="""  event0();
$mode

  for (int i = 0; i < PACK_N; i += 32) {
    aie::vector<int16_t, 32> v = aie::load_v<32>(in + i);
    aie::vector<int8_t, 32> o = v.pack<int8_t>();
    aie::store_v(out + i, o);
  }
  event1();
""",
))


# --- Convert sweep: crRnd over f32 -> bf16 round-narrow (accfloat). 10 rounding
# modes. No saturation axis. Normal-finite inputs only (denormal FTZ + NaN
# canonicalization are HW-gated edges deferred to phase B).
_reg_sweep(SweepSpec(
    prefix="vec_conv_bf16",
    func="conv_bf16",
    doc="Convert (round-narrow), f32 -> bf16 through an accfloat accumulator, "
        "normal-finite inputs.",
    inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
    output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
    n=448,
    gclass="bf16_srs",
    base_filt={},
    rnds=_ALL_RND,
    predicate=_is_normal_f32,
    defines=[("CONV_N", 448)],
    body_template="""  event0();
$mode

  for (int i = 0; i < CONV_N; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(in + i);
    aie::accum<accfloat, 16> acc(v);
    aie::vector<bfloat16, 16> o = acc.to_vector<bfloat16>();
    aie::store_v(out + i, o);
  }
  event1();
""",
))


# --- UPS sweep: crSat over int16 -> int32 widen (acc32). 2 saturation modes
# (the ups corpus has no symmetric column). UPS widens, so a shift<=4 never
# overflows int32 -- none and saturate produce identical output; the sweep
# confirms silicon plumbs the saturation control without diverging.
_reg_sweep(SweepSpec(
    prefix="vec_ups_i32",
    func="ups_i32",
    doc="UPS (unpack-shift widen), int16 -> int32 accumulator, shift=4.",
    inputs=[Buf("in", "int16_t", "i16")],
    output=Buf("out", "int32_t", "i32"),
    n=48,
    gclass="ups",
    base_filt={"bits_in": 16, "bits_out": 32, "signed": True, "shift": 4},
    sats=[0, 1],
    sat_field="sat",
    symsat_field=None,
    value_range=(-(2 ** 15), 2 ** 15 - 1),
    defines=[("UPS_N", 48), ("UPS_SHIFT", 4)],
    body_template="""  event0();
$mode

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
