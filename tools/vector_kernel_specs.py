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

from gen_vector_kernel import (
    Buf, DirectIO, KernelSpec, Matmul, SweepSpec, _silicon_path
)

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


_DENORM_N = 256  # all 254 bf16 denormals + 0x0000 + 0x8000


def _bf16_denorm_inputs():
    """All 254 bf16 denormals (exp=0, mantissa 1..127, both signs) + +/-0."""
    pos = [m for m in range(1, 128)]              # 0x0001..0x007F
    neg = [0x8000 | m for m in range(1, 128)]     # 0x8001..0x807F
    vals = pos + neg + [0x0000, 0x8000]
    assert len(vals) == _DENORM_N
    return tuple(vals)


def _bf16_to_f32_bits(b):
    """Exact bf16->f32 widen: bf16 is the high 16 bits of the f32 word."""
    return (b & 0xFFFF) << 16


def _bf16_floor_reference():
    """No-FTZ floor(bf16) per lane as SIGNED int32 values.

    The vfloor kernel's output buffer is int32_t (signed), so the reference and
    the captured silicon are stored as signed ints: floor of a tiny negative
    denormal is -1 (not the unsigned 0xFFFFFFFF, which is a C++11 narrowing
    error in the generated int32_t EXP array). This matches what silicon dumps
    to out.txt -- `(int64_t)(int32_t)-1` is -1 -- so bootstrap, capture, and
    comparison all share one signed representation.
    """
    import math
    import struct
    out = []
    for b in _bf16_denorm_inputs():
        f = struct.unpack("<f", struct.pack("<I", _bf16_to_f32_bits(b)))[0]
        out.append(int(math.floor(f)))
    return tuple(out)


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


def _is_edge_f32(rec):
    """The record's f32 bit pattern is NOT a normal finite number.

    Exact complement of `_is_normal_f32`: zero/denormal (exp=0) and Inf/NaN
    (exp=255) -- the inputs the phase-A conv kernel excluded. Where the execute
    path's input FTZ and NaN handling can diverge from the aietools model.
    """
    return not _is_normal_f32(rec)


def _has_overflow_expected(rec):
    """At least one lane of `expected` (fp32 bit patterns) is Inf/NaN.

    Complement of `_all_expected_finite_f32`: the bf16 matmul tiles whose result
    overflows. The overflow lane's canonical NaN is mantissa=1 on silicon+emu but
    mantissa=0x7F in the model, so these tiles need a silicon oracle.
    """
    return not _all_expected_finite_f32(rec)


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


# === Phase-B edge specs (silicon golden required) =============================
#
# These specs exercise the inputs that the phase-A kernels deliberately excluded
# (denormal/NaN/Inf f32, bf16 overflow) because the aietools model diverges from
# real silicon on those edges. EXP = HW-captured silicon (via silicon_golden).
# Until the silicon JSON is captured (bootstrap mode), the model golden is baked
# as a placeholder; the alignment assert in _bake_io prevents a stale capture
# from silently baking mismatched EXP once the JSON is present.

EDGE_N = 96            # ceil(94 non-normal bf16_srs inputs/mode / 16) * 16
MAC_OVF_BATCH = 54     # all bf16 matmul overflow tiles in the regrown corpus

# --- Convert edge sweep: denormal + NaN + Inf f32 -> bf16 across all 10 rounding
# modes. The model ROUNDS denormals (mode-dependent) while the execute path FTZs;
# rnd=0 and rnd=12 alone miss it, so the full mode space is swept. EXP = silicon.
_reg_sweep(SweepSpec(
    prefix="vec_conv_bf16_edge",
    func="conv_bf16",
    doc="Convert edge (round-narrow), f32 -> bf16, denormal/NaN/Inf inputs.",
    inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
    output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
    n=EDGE_N,
    gclass="bf16_srs",
    base_filt={},
    rnds=_ALL_RND,
    predicate=_is_edge_f32,
    silicon=True,
    defines=[("CONV_N", EDGE_N)],
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


# --- MAC bf16 overflow: the bf16 matmul tiles whose result overflows to Inf/NaN.
# The overflow lane's canonical NaN is mantissa=1 on silicon+emu vs 0x7F in the
# model, so EXP = silicon. Same 4x8x4 mmul datapath as vec_mac_bf16.
_reg(KernelSpec(
    name="vec_mac_bf16_ovf",
    func="mac_bf16",
    doc="MatMul bf16 overflow tiles (4x8x4): result overflows to Inf/NaN. EXP is "
        "HW-captured silicon (model canonical NaN mantissa 0x7F vs silicon 1). "
        "Host stages bf16/fp32 bit patterns.",
    inputs=[Buf("inA", "uint16_t", "bf16", ktype="bfloat16"),
            Buf("inB", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "uint32_t", "f32", ktype="float"),
    n=0,
    golden={"class": "matmul",
            "filt": {"a_type": "BFloat16", "b_type": "BFloat16", "rows": 4,
                     "inner": 8, "cols": 4, "subtract": False},
            "predicate": _has_overflow_expected},
    matmul=Matmul(M=4, K=8, N=4, a_bytes=2, b_bytes=2, batch=MAC_OVF_BATCH, bfloat=True),
    silicon_golden=_silicon_path("vec_mac_bf16_ovf"),
    defines=[("MAC_BATCH", MAC_OVF_BATCH)],
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


# === Convert FTZ-audit specs (Half-B phase C, silicon golden required) ========
#
# These two kernels audit whether the bf16 convert datapath flushes denormals to
# zero (FTZ). Both feed the exhaustive bf16-denormal space (all 254 denormals +
# +/-0) via DirectIO -- an input range no corpus class produces. EXP = HW-captured
# silicon (via silicon_golden); the bootstrap reference is the no-FTZ widen/floor
# computed here, which doubles as the divergence baseline. As with the phase-B
# edge specs, the alignment assert in _bake_io guards against a stale capture
# baking mismatched EXP once the silicon JSON is present.
#
# vec_vfloor_bf16_denorm is THE audit test. It emits a standalone VFLOOR.s32.bf16
# (decodes to SemanticOp::Convert), which routes through vector_convert's
# bf16->int32 branch (vector_convert.rs:343) where fp32_flush_to_zero is applied.
# A negative bf16 denormal floors to -1 without FTZ vs 0 with it -- a clean -1-vs-0
# discriminator.
#
# vec_convexp_bf16_denorm probes the bf16->f32 expand direction (new coverage;
# phase B only narrowed f32->bf16). The expand CANNOT be isolated as a standalone
# VCONV: Chess fuses it into VLDA.CONV.fp32.bf16, a load-convert. The emulator
# special-cases that opcode in execute_fused_load_convert (memory/mod.rs:891-900)
# with a pure bf16<<16 widen that already does NOT flush (it never calls
# vector_convert). So this kernel does not exercise vector_convert's FTZ; it
# silicon-validates the fused-load expand path and the new expand direction.
_reg(KernelSpec(
    name="vec_convexp_bf16_denorm", func="convexp_bf16", stem="convexp",
    doc="bf16->f32 expand (fused VLDA.CONV.fp32.bf16, no FTZ), exhaustive bf16 "
        "denormal inputs. EXP = HW-captured silicon.",
    silicon_golden=_silicon_path("vec_convexp_bf16_denorm"),
    inputs=[Buf("in", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "uint32_t", "f32", ktype="float"),
    n=_DENORM_N,
    golden={"class": "direct",
            "direct": DirectIO(inputs=_bf16_denorm_inputs(),
                               reference=tuple(_bf16_to_f32_bits(v) for v in _bf16_denorm_inputs()))},
    defines=[("CONV_N", _DENORM_N)],
    body="""  event0();
  for (int i = 0; i < CONV_N; i += 16) {
    aie::vector<bfloat16, 16> v = aie::load_v<16>(in + i);
    v16accfloat raw = ::ups_to_v16accfloat((v16bfloat16)v);
    aie::accum<accfloat, 16> acc(raw);
    aie::vector<float, 16> o = acc.to_vector<float>();
    aie::store_v(out + i, o);
  }
  event1();
""",
))

_reg(KernelSpec(
    name="vec_vfloor_bf16_denorm", func="vfloor_bf16", stem="vfloor",
    doc="bf16->int32 floor (standalone VFLOOR.s32.bf16), exhaustive bf16 denormal "
        "inputs. Routes through vector_convert's FTZ (vector_convert.rs:343): neg "
        "denormal floors to -1 without FTZ vs 0 with. EXP = HW-captured silicon.",
    silicon_golden=_silicon_path("vec_vfloor_bf16_denorm"),
    inputs=[Buf("in", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "int32_t", "i32"),
    n=_DENORM_N,
    golden={"class": "direct",
            "direct": DirectIO(inputs=_bf16_denorm_inputs(),
                               reference=_bf16_floor_reference())},
    defines=[("VFL_N", _DENORM_N)],
    body="""  event0();
  ::aie::set_rounding(aie::rounding_mode::floor);
  for (int i = 0; i < VFL_N; i += 16) {
    aie::vector<bfloat16, 16> v = aie::load_v<16>(in + i);
    v16int32 o = ::bfloat16_to_int((v16bfloat16)v, 0);
    aie::store_v(out + i, (aie::vector<int32, 16>)o);
  }
  event1();
""",
))
