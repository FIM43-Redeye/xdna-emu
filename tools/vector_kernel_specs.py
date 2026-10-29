"""Registry of vector-compute capture-kernel specs.

Each entry is a `KernelSpec` the generator (gen_vector_kernel.py) turns into the
four files a bridge kernel needs. The `body` field is the intrinsic IP,
hand-written once per class; everything else -- scaffold, MLIR design, host
harness, arrays -- is derived. Inputs are designed in tools/vector_inputs.py;
expected values are captured from NPU1 silicon (tools/capture_silicon.py ->
tools/golden/silicon/<name>.json). To add a kernel, append a spec here, generate
it (`python3 tools/gen_vector_kernel.py <name>`), run it once on hardware and
capture the output.
"""

import string

import vector_inputs as vi
from gen_vector_kernel import (Buf, DirectIO, KernelSpec, Matmul, ROUNDING_ENUM, SAT_ENUM,
                               SweepSpec, UNDEFINED_RND, mode_lines)

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
    SWEEPS[sweep.key or sweep.prefix + "_sweep"] = sweep
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


# --- SRS: accumulator (acc64) -> narrower vector (int16), shift-round-saturate.
# Single-mode anchor: floor rounding, saturate, shift 4.
_reg(KernelSpec(
    name="vec_srs_i32",
    func="srs_i32",
    doc="SRS (shift-round-saturate), int32 accumulator -> int16. Config: "
        "rnd=FLOOR, sat=saturate, shift=4.",
    inputs=[Buf("in", "int32_t", "i32")],
    output=Buf("out", "int16_t", "i16"),
    n=48,
    io_class="srs",
    direct=DirectIO(inputs=(vi.srs_inputs(48, shift=4),)),
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
        "sat=none, shift=4.",
    inputs=[Buf("in", "int16_t", "i16")],
    output=Buf("out", "int32_t", "i32"),
    n=48,
    io_class="ups",
    direct=DirectIO(inputs=(vi.ups_inputs(48),)),
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
# the int16->int8 native pack -- a distinct datapath. This anchor runs with
# crSat=none; vec_pack_i16_sat below runs the same body with crSat=saturate.
_reg(KernelSpec(
    name="vec_pack_i16",
    func="pack_i16",
    doc="Pack (native VPACK), int16 -> int8 narrow with crSat=none.",
    inputs=[Buf("in", "int16_t", "i16")],
    output=Buf("out", "int8_t", "i8"),
    n=32,
    io_class="pack",
    direct=DirectIO(inputs=(vi.pack_inputs(32),)),
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


# Same body as vec_pack_i16 with crSat=saturate (int16 -> int8 clamp).
_reg(KernelSpec(
    name="vec_pack_i16_sat",
    func="pack_i16",
    doc="Pack (native VPACK), int16 -> int8 narrow with crSat=saturate.",
    inputs=[Buf("in", "int16_t", "i16")],
    output=Buf("out", "int8_t", "i8"),
    n=32,
    io_class="pack",
    direct=DirectIO(inputs=(vi.pack_inputs(32),)),
    defines=[("PACK_N", 32)],
    stem="pack",
    body="""  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);

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
# rounding from bit-truncation. Inputs: normal finite f32 only (both signs);
# denormal/NaN/Inf are covered by the vec_conv_bf16_edge_* kernels.
_reg(KernelSpec(
    name="vec_conv_bf16",
    func="conv_bf16",
    doc="Convert (round-narrow), f32 -> bf16 through an accfloat accumulator. "
        "Config: rounding=conv_even (round-to-nearest, ties to even), normal "
        "finite inputs. crRnd governs to_v16bfloat16, so set_rounding selects "
        "the mode.",
    inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
    output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
    n=448,
    io_class="convert",
    direct=DirectIO(inputs=(vi.conv_inputs(448),)),
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
# A batch of independent row-major tiles; each C = A.B (integer sum of
# products, int32 accumulator), compared exactly against silicon. The
# aie::mmul class handles the register-level A/B packing internally -- the
# kernel just DMAs plain row-major matrices.
_reg(KernelSpec(
    name="vec_mac_i8",
    func="mac_i8",
    doc="MatMul (native mmul tile), int8 x int8 -> int32, 4x8x8. A batch of "
        "independent row-major tiles (full int8 range plus constant extreme "
        "tiles); each C = A.B (integer sum of products).",
    inputs=[Buf("inA", "int8_t", "i8"), Buf("inB", "int8_t", "i8")],
    output=Buf("out", "int32_t", "i32"),
    n=0,
    io_class="matmul",
    direct=DirectIO(inputs=vi.matmul_int_inputs(48, 4, 8, 8, bits=8, seed=0x18)),
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
# so the all-minimum tile (2 * 2^30) probes the int32 wrap/saturate boundary.
_reg(KernelSpec(
    name="vec_mac_i16",
    func="mac_i16",
    doc="MatMul (native mmul tile), int16 x int16 -> int32, 4x2x8. A batch of "
        "independent row-major tiles (full int16 range plus constant extreme "
        "tiles); each C = A.B in the int32 accumulator.",
    inputs=[Buf("inA", "int16_t", "i16"), Buf("inB", "int16_t", "i16")],
    output=Buf("out", "int32_t", "i32"),
    n=0,
    io_class="matmul",
    direct=DirectIO(inputs=vi.matmul_int_inputs(48, 4, 2, 8, bits=16, seed=0x116)),
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
# kernel reads bfloat16/float. Inputs are finite with mixed exponents (no
# Inf/NaN reachable) plus exact-cancellation tiles, isolating the mmul
# accumulate/round datapath from the overflow edge (vec_mac_bf16_ovf).
_reg(KernelSpec(
    name="vec_mac_bf16",
    func="mac_bf16",
    doc="MatMul (native mmul tile), bf16 x bf16 -> fp32, 4x8x4. A batch of "
        "independent row-major tiles (finite, mixed exponents, exact-cancellation "
        "tiles); each C = A.B in fp32. "
        "Host stages bf16/fp32 bit patterns; the kernel reads bfloat16/float.",
    inputs=[Buf("inA", "uint16_t", "bf16", ktype="bfloat16"),
            Buf("inB", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "uint32_t", "f32", ktype="float"),
    n=0,
    io_class="matmul",
    direct=DirectIO(inputs=vi.matmul_bf16_inputs(24, 4, 8, 4)),
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


# === Mode-sweep families (mode-exhaustive on silicon) ========================
#
# Each sweep mirrors a single-mode anchor above but parametrizes the swept axes
# (the `_r<rnd>` / `_s<crSat>` name suffixes; the SRS/UPS shift stays 4). Every
# point of a sweep is fed the same designed inputs, so mode-to-mode differences
# are attributable to the mode alone.

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
    io_class="srs",
    direct=DirectIO(inputs=(vi.srs_inputs(64, shift=4),)),
    rnds=_ALL_RND,
    sats=_ALL_SAT,
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
# crSat is the axis that selects truncation versus saturation on silicon.
_reg_sweep(SweepSpec(
    prefix="vec_pack_i16",
    func="pack_i16",
    doc="Pack (native VPACK), int16 -> int8 narrowing.",
    inputs=[Buf("in", "int16_t", "i16")],
    output=Buf("out", "int8_t", "i8"),
    n=96,
    io_class="pack",
    direct=DirectIO(inputs=(vi.pack_inputs(96),)),
    sats=_ALL_SAT,
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
# modes. No saturation axis. Normal-finite inputs only (denormal/NaN/Inf are
# the vec_conv_bf16_edge_* kernels).
_reg_sweep(SweepSpec(
    prefix="vec_conv_bf16",
    func="conv_bf16",
    doc="Convert (round-narrow), f32 -> bf16 through an accfloat accumulator, "
        "normal-finite inputs.",
    inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
    output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
    n=448,
    io_class="convert",
    direct=DirectIO(inputs=(vi.conv_inputs(448),)),
    rnds=_ALL_RND,
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
# (none, saturate). UPS widens, so a shift<=4 never overflows int32; the sweep
# records whether silicon's saturation control changes anything.
_reg_sweep(SweepSpec(
    prefix="vec_ups_i32",
    func="ups_i32",
    doc="UPS (unpack-shift widen), int16 -> int32 accumulator, shift=4.",
    inputs=[Buf("in", "int16_t", "i16")],
    output=Buf("out", "int32_t", "i32"),
    n=48,
    io_class="ups",
    direct=DirectIO(inputs=(vi.ups_inputs(48),)),
    sats=[0, 1],
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


# === Edge specs ================================================================
#
# These specs exercise the inputs the normal-finite kernels deliberately exclude
# (denormal/NaN/Inf f32, bf16 overflow and cancellation). Inputs are designed in
# vector_inputs.py; EXP is HW-captured silicon. The two denorm kernels below use
# the exhaustive bf16-denormal space.

EDGE_N = 96            
MAC_OVF_BATCH = 54     # bf16 matmul tiles whose result overflows

# --- Convert edge sweep: denormal + NaN + Inf f32 -> bf16 across all 10 rounding
# modes. Whether denormals are rounded or flushed can depend on the mode, so the
# full mode space is swept.
_reg_sweep(SweepSpec(
    prefix="vec_conv_bf16_edge",
    func="conv_bf16",
    doc="Convert edge (round-narrow), f32 -> bf16, denormal/NaN/Inf inputs.",
    inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
    output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
    n=EDGE_N,
    io_class="convert",
    direct=DirectIO(inputs=(vi.conv_edge_inputs(EDGE_N),)),
    rnds=_ALL_RND,
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
# Silicon decides the canonical NaN encoding of an overflow lane. Same 4x8x4
# mmul datapath as vec_mac_bf16.
_reg(KernelSpec(
    name="vec_mac_bf16_ovf",
    func="mac_bf16",
    doc="MatMul bf16 overflow tiles (4x8x4): result overflows to Inf/NaN. EXP is "
        "HW-captured silicon. Host stages bf16/fp32 bit patterns.",
    inputs=[Buf("inA", "uint16_t", "bf16", ktype="bfloat16"),
            Buf("inB", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "uint32_t", "f32", ktype="float"),
    n=0,
    io_class="matmul",
    direct=DirectIO(inputs=vi.matmul_bf16_edge_inputs(MAC_OVF_BATCH, 4, 8, 4)),
    matmul=Matmul(M=4, K=8, N=4, a_bytes=2, b_bytes=2, batch=MAC_OVF_BATCH, bfloat=True),
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


# === Convert FTZ-audit specs ===================================================
#
# These two kernels audit whether the bf16 convert datapath flushes denormals to
# zero (FTZ). Both feed the exhaustive bf16-denormal space (all 254 denormals +
# +/-0) via DirectIO. EXP = HW-captured silicon; the generator's staleness check
# fails if the capture's inputs no longer match.
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
    inputs=[Buf("in", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "uint32_t", "f32", ktype="float"),
    n=_DENORM_N,
    io_class="expand",
    direct=DirectIO(inputs=(_bf16_denorm_inputs(),)),
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
        "inputs (a negative denormal floors to -1 unless flushed to zero). "
        "EXP = HW-captured silicon.",
    inputs=[Buf("in", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "int32_t", "i32"),
    n=_DENORM_N,
    io_class="floor",
    direct=DirectIO(inputs=(_bf16_denorm_inputs(),)),
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


# === SRS shift sweep ===========================================================
#
# int32 -> acc64 -> int16 SRS with the shift as the parameter (a compile-time
# define in each kernel), at floor(0) / symmetric_ceil(3) / conv_even(12) with
# saturation on, plus conv_even with saturation off. Inputs are designed per
# shift (vector_inputs.srs_inputs). Names: vec_srs_i32_sh<shift>_r<rnd>[_s0].
SRS_SHIFTS = [0, 1, 2, 3, 5, 8, 12, 16, 24, 31]
SRS_SHIFT_RND = [0, 3, 12]
SRS_SHIFT_BODY = """  event0();
$mode

  for (int i = 0; i < SRS_N; i += 16) {
    aie::vector<int32_t, 16> v = aie::load_v<16>(in + i);
    aie::accum<acc64, 16> acc;
    acc.from_vector(v, 0);
    aie::vector<int16_t, 16> o = acc.to_vector<int16_t>(SRS_SHIFT);
    aie::store_v(out + i, o);
  }
  event1();
"""


def _reg_srs_shift_point(shift, rnd, sat):
    suffix = f"_sh{shift}_r{rnd}" + ("" if sat == 1 else f"_s{sat}")
    _reg(KernelSpec(
        name="vec_srs_i32" + suffix,
        func="srs_i32",
        doc=f"SRS (shift-round-saturate), int32 accumulator -> int16, shift={shift}, "
            f"rnd={rnd}({ROUNDING_ENUM[rnd]}), sat={sat}({SAT_ENUM[sat]}).",
        inputs=[Buf("in", "int32_t", "i32")],
        output=Buf("out", "int16_t", "i16"),
        n=64,
        io_class="srs",
        direct=DirectIO(inputs=(vi.srs_inputs(64, shift=shift),)),
        defines=[("SRS_N", 64), ("SRS_SHIFT", shift)],
        body=string.Template(SRS_SHIFT_BODY).substitute(mode=mode_lines(rnd, sat)),
    ))


for _sh in SRS_SHIFTS:
    for _rnd in SRS_SHIFT_RND:
        _reg_srs_shift_point(_sh, _rnd, 1)
    _reg_srs_shift_point(_sh, 12, 0)


# --- Undefined crRnd values (llvm-aie leaves 4-7, 14, 15 undefined): drive the raw
# index and record what silicon does. SRS at shift 4 with saturation on, and the
# f32 -> bf16 convert on the normal-finite input set.
_reg_sweep(SweepSpec(
    prefix="vec_srs_i32",
    key="vec_srs_i32_undef_rnd_sweep",
    func="srs_i32",
    doc="SRS (shift-round-saturate), int32 accumulator -> int16, shift=4, "
        "undefined crRnd index.",
    inputs=[Buf("in", "int32_t", "i32")],
    output=Buf("out", "int16_t", "i16"),
    n=64,
    io_class="srs",
    direct=DirectIO(inputs=(vi.srs_inputs(64, shift=4),)),
    rnds=list(UNDEFINED_RND),
    sats=[1],
    defines=[("SRS_N", 64), ("SRS_SHIFT", 4)],
    body_template=SRS_SHIFT_BODY,
))

_reg_sweep(SweepSpec(
    prefix="vec_conv_bf16",
    key="vec_conv_bf16_undef_rnd_sweep",
    func="conv_bf16",
    doc="Convert (round-narrow), f32 -> bf16 through an accfloat accumulator, "
        "normal-finite inputs, undefined crRnd index.",
    inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
    output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
    n=448,
    io_class="convert",
    direct=DirectIO(inputs=(vi.conv_inputs(448),)),
    rnds=list(UNDEFINED_RND),
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


# === Shuffle mode sweep =========================================================
#
# VSHUFFLE on two 512-bit operands (A = bytes 1..64, B = bytes 65..128, every byte
# distinct and nonzero) under a table of mode numbers: five named controls, then 44..63, which
# spans the top of the defined range and the modes >= 48. The mode comes from a
# table in data memory, so it reaches the instruction through a scalar register
# rather than as an immediate. Each mode writes one 16-word output block; the
# input buffer is padded to the output size.
SHUFFLE_CONTROLS = ["INTLV_lo_32o64", "INTLV_hi_32o64", "DINTLV_lo_32o64",
                    "DINTLV_hi_32o64", "T32_4x4"]
SHUFFLE_NUMERIC = list(range(44, 64))
SHUFFLE_MODES = SHUFFLE_CONTROLS + [str(m) for m in SHUFFLE_NUMERIC]
SHUFFLE_N = 16 * len(SHUFFLE_MODES)

_reg(KernelSpec(
    name="vec_shuffle_mode",
    func="shuffle_mode",
    stem="shuffle",
    doc="VSHUFFLE of two 512-bit operands under a table of runtime mode numbers "
        "(named controls, then 44..63). Output block k is the shuffle under "
        "mode k of the table.",
    inputs=[Buf("in", "int32_t", "i32")],
    output=Buf("out", "int32_t", "i32"),
    n=SHUFFLE_N,
    io_class="shuffle",
    direct=DirectIO(inputs=(vi.shuffle_inputs(SHUFFLE_N),)),
    defines=[("SHUF_MODES", len(SHUFFLE_MODES))],
    body="""  event0();
  static const unsigned modes[SHUF_MODES] = {%s};
  aie::vector<int32_t, 16> a = aie::load_v<16>(in);
  aie::vector<int32_t, 16> b = aie::load_v<16>(in + 16);
  for (int k = 0; k < SHUF_MODES; k++) {
    v16int32 r = ::shuffle((v16int32)a, (v16int32)b, modes[k]);
    aie::store_v(out + 16 * k, (aie::vector<int32_t, 16>)r);
  }
  event1();
""" % ", ".join(SHUFFLE_MODES),
))
