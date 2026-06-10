#!/usr/bin/env python3
"""Generate golden test data for vector compute validation.

Generates deterministic test vectors for SRS, UPS, and element-wise vector
operations. SRS and UPS use the **genuine aietools Python model** as the oracle
(driven out-of-repo, never copied in); element-wise integer ops use
straightforward wrap-around arithmetic (which *is* the spec for them -- no
rounding or saturation subtlety).

Usage:
    VECTOR_ORACLE_MODEL=/path/to/ported/model \
        python3 tools/gen_vector_golden.py

Output:
    tools/golden/vector_ops.json

The generated file is checked in and loaded by Rust tests at cargo test time,
so there is NO aietools dependency at test time -- only at regeneration time.

Oracle provenance (de-circularization)
--------------------------------------
Earlier this generator *re-implemented* srs_lane/ups_lane/trnc in Python "from
srs.py / ups.py". That was circular: a misread of the reference would corrupt
the emulator and the golden identically, hiding the bug. We now drive the real
aietools model functions directly, so the golden is provenanced to the genuine
silicon reference, not to a hand-port.

The model is Python 2 and needs a py2->py3 port (print statements, integer
division, long literals). Per the licensing policy, aietools code stays
OUT-OF-REPO: the ported model lives in an out-of-repo working copy (default
$VECTOR_ORACLE_MODEL below; see experiments/vector-oracle/). Only the derived
golden JSON is committed -- matching the aiesim oracle posture and the existing
golden precedent. Regeneration therefore requires the licensed tool, which is
correct: a genuine-aietools-derived golden should not be reproducible without
aietools.

Source functions (read-only reference, driven not copied):
  aietools/data/aie_ml/lib/python_model/model/srs.py:srs_lane
  aietools/data/aie_ml/lib/python_model/model/ups.py:ups_lane
  aietools/data/aie_ml/lib/python_model/model/helpers.py:trnc
"""

import json
import os
import random
import sys

# ---------------------------------------------------------------------------
# Genuine aietools oracle (out-of-repo, py2->py3 ported working copy)
# ---------------------------------------------------------------------------

ORACLE_MODEL = os.environ.get(
    "VECTOR_ORACLE_MODEL",
    "/home/triple/npu-work/experiments/vector-oracle/model",
)


def load_oracle():
    """Import the genuine aietools model fns from the out-of-repo working copy.

    Returns (srs_module, ups_module, trnc_fn). Fails loud if the oracle is not
    present -- regeneration requires the licensed tool by design.
    """
    if not os.path.isdir(ORACLE_MODEL):
        sys.exit(
            f"genuine aietools oracle model not found at:\n  {ORACLE_MODEL}\n"
            "Set VECTOR_ORACLE_MODEL to the py2->py3 ported aietools model dir.\n"
            "(The committed golden JSON is the test-time artifact; the oracle is\n"
            "only needed to REGENERATE it. See tools/gen_vector_golden.py header.)"
        )
    import builtins
    builtins.long = int  # py2 long() shim; model files are not edited for this
    sys.path.insert(0, ORACLE_MODEL)
    import srs
    import ups
    import pack
    import bfloat16
    from helpers import trnc
    return srs, ups, pack, bfloat16, trnc


SRS, UPS, PACK, BF16, trnc = load_oracle()

BIAS = 4  # SRS rounding-precision bias; caller passes user_shift + BIAS.

# ---------------------------------------------------------------------------
# Reproducible randomness
# ---------------------------------------------------------------------------

SEED = 20260301
rng = random.Random(SEED)

# ---------------------------------------------------------------------------
# SRS golden data (Tier 1)
# ---------------------------------------------------------------------------

# The Rust srs_lane() signature:
#   srs_lane(value: i64, shift: u32, signed_output: bool,
#            output_bits: u32, saturate: bool, symmetric_saturate: bool,
#            mode: RoundingMode) -> i64
#
# The genuine model srs_lane() signature:
#   srs_lane(a_in, shft, bits_i, bits_o, sgn_o, sat, symsat, rnd)
#   where shft = user_shift + BIAS (BIAS=4); bits_i is unused (arbitrary prec).
#
# Our Rust function adds BIAS internally, so we record user_shift in JSON
# and pass user_shift + BIAS to the oracle.

# Rounding mode constants (from constants.py).
RND_FLOOR = 0
RND_CEIL = 1
RND_SYM_FLOOR = 2
RND_SYM_CEIL = 3
RND_NEG_INF = 8
RND_POS_INF = 9
RND_SYM_ZERO = 10
RND_SYM_INF = 11
RND_CONV_EVEN = 12
RND_CONV_ODD = 13

RND_MODES = (RND_FLOOR, RND_CEIL, RND_SYM_FLOOR, RND_SYM_CEIL,
             RND_NEG_INF, RND_POS_INF, RND_SYM_ZERO, RND_SYM_INF,
             RND_CONV_EVEN, RND_CONV_ODD)

# Output type configurations: (output_bits, signed_output)
OUTPUT_CONFIGS = [
    (8, True),   # signed 8-bit
    (8, False),  # unsigned 8-bit
    (16, True),  # signed 16-bit
    (16, False), # unsigned 16-bit
    (32, True),  # signed 32-bit
    (32, False), # unsigned 32-bit
]

# Saturation configs: (saturate, sym_sat)
SAT_CONFIGS = [
    (True, False),   # normal saturation
    (True, True),    # symmetric saturation
    (False, False),  # no saturation
]


def srs_boundary_values(bits_o, signed):
    """Generate boundary test values for a given output configuration."""
    # Values that stress the shift/round/saturate pipeline.
    vals = [0, 1, -1, 16, -16, 255, -255, 256, -256]

    # Output range boundaries (pre-shift, so multiply by powers of 2).
    if signed:
        vmax = (1 << (bits_o - 1)) - 1
        vmin = -(1 << (bits_o - 1))
    else:
        vmax = (1 << bits_o) - 1
        vmin = 0

    # Values near the output range boundaries, shifted by typical amounts.
    for shift in [0, 4, 8]:
        scale = 1 << (shift + BIAS)
        vals.extend([vmax * scale, vmax * scale + 1, vmax * scale - 1])
        vals.extend([vmin * scale, vmin * scale + 1, vmin * scale - 1])
        # Exact halfway points for rounding tests.
        vals.append(vmax * scale + scale // 2)
        vals.append(vmin * scale + scale // 2)

    # Powers of 2.
    for p in range(0, 40, 4):
        vals.extend([1 << p, -(1 << p)])

    return list(set(vals))


def gen_srs_golden():
    """Generate SRS golden test cases against the genuine aietools model."""
    cases = []
    count_by_config = {}

    for bits_o, signed in OUTPUT_CONFIGS:
        for sat, sym_sat in SAT_CONFIGS:
            boundary_vals = srs_boundary_values(bits_o, signed)
            for rnd_mode in RND_MODES:
                # Boundary values with common shifts.
                for shift in [0, 4, 8, 12]:
                    for val in boundary_vals:
                        total_shift = shift + BIAS
                        py_result, _ = SRS.srs_lane(
                            val, total_shift, 64, bits_o, signed,
                            sat, sym_sat, rnd_mode,
                        )
                        cases.append({
                            "value": int(val),
                            "shift": shift,
                            "signed": signed,
                            "bits_o": bits_o,
                            "sat": sat,
                            "sym_sat": sym_sat,
                            "rnd": rnd_mode,
                            "expected": int(py_result),
                        })

                # Random values (seeded).
                for _ in range(10):
                    val = rng.randint(-(1 << 48), (1 << 48) - 1)
                    shift = rng.randint(0, 16)
                    total_shift = shift + BIAS
                    py_result, _ = SRS.srs_lane(
                        val, total_shift, 64, bits_o, signed,
                        sat, sym_sat, rnd_mode,
                    )
                    cases.append({
                        "value": int(val),
                        "shift": shift,
                        "signed": signed,
                        "bits_o": bits_o,
                        "sat": sat,
                        "sym_sat": sym_sat,
                        "rnd": rnd_mode,
                        "expected": int(py_result),
                    })

                key = (bits_o, signed, sat, sym_sat, rnd_mode)
                count_by_config[key] = count_by_config.get(key, 0) + len(boundary_vals) * 4 + 10

    print(f"  SRS: {len(cases)} cases across {len(count_by_config)} configurations")
    return cases


# ---------------------------------------------------------------------------
# UPS golden data (Tier 2)
# ---------------------------------------------------------------------------

# The Rust ups_lane() signature:
#   ups_lane(value: i64, shift: u32, bits_in: u32, bits_out: u32, saturate: bool) -> i64
#
# The genuine model ups_lane() signature:
#   ups_lane(a, shft, bits_o, sat) -> int
#   It takes the already-sign-extended input value (genuine trnc applied here).

UPS_MODES = [
    # (bits_in, bits_out, lanes)
    (8, 32, 32),
    (16, 32, 32),
    (16, 64, 16),
    (32, 64, 16),
]


def ups_boundary_values(bits_in, signed):
    """Generate boundary values for UPS testing."""
    if signed:
        vmax = (1 << (bits_in - 1)) - 1
        vmin = -(1 << (bits_in - 1))
    else:
        vmax = (1 << bits_in) - 1
        vmin = 0

    vals = [0, 1, -1, vmax, vmin, vmax - 1, vmin + 1]
    # Powers of 2 within range.
    for p in range(0, bits_in - 1):
        vals.extend([1 << p, -(1 << p)])

    return list(set(v for v in vals if vmin <= v <= vmax))


def gen_ups_golden():
    """Generate UPS golden test cases against the genuine aietools model."""
    cases = []

    for bits_in, bits_out, _lanes in UPS_MODES:
        for signed in [True, False]:
            for sat in [True, False]:
                boundary_vals = ups_boundary_values(bits_in, signed)

                for shift in [0, 1, 4, 8, 12]:
                    for val in boundary_vals:
                        # Sign-extend the input value to bits_in width,
                        # matching what the Rust code does (genuine trnc).
                        truncated = int(trnc(val, signed, bits_in))
                        py_result = int(UPS.ups_lane(
                            truncated, shift, bits_out, sat,
                        ))
                        cases.append({
                            "value": int(val),
                            "shift": shift,
                            "bits_in": bits_in,
                            "bits_out": bits_out,
                            "signed": signed,
                            "sat": sat,
                            "expected": int(py_result),
                        })

                # Random values.
                for shift in [0, 4, 8]:
                    for _ in range(10):
                        if signed:
                            val = rng.randint(-(1 << (bits_in - 1)),
                                              (1 << (bits_in - 1)) - 1)
                        else:
                            val = rng.randint(0, (1 << bits_in) - 1)
                        truncated = int(trnc(val, signed, bits_in))
                        py_result = int(UPS.ups_lane(
                            truncated, shift, bits_out, sat,
                        ))
                        cases.append({
                            "value": int(val),
                            "shift": shift,
                            "bits_in": bits_in,
                            "bits_out": bits_out,
                            "signed": signed,
                            "sat": sat,
                            "expected": int(py_result),
                        })

    print(f"  UPS: {len(cases)} cases across {len(UPS_MODES)} mode pairs")
    return cases


# ---------------------------------------------------------------------------
# Pack (narrowing) golden data (Tier 2b)
# ---------------------------------------------------------------------------

# The Rust pack_lane() signature:
#   pack_lane(value: i64, bits_i: u32, bits_o: u32, signed: bool,
#             mode: PackMode) -> i64
#   where PackMode in {Truncate, Saturate, SymmetricSaturate}.
#
# The genuine model pack_lane() signature:
#   pack_lane(a_in, bits_i, bits_o, sgn, sat, symsat) -> int
#   (sat=False => Truncate; sat=True,symsat=False => Saturate;
#    sat=True,symsat=True => SymmetricSaturate.)
#
# bits_i is unused by the lane computation (arbitrary-precision input); it is
# recorded so the Rust seam receives the same signature.

# AIE2 pack width pairs (bits_in -> bits_out). 4-bit outputs (D4) are real on
# AIE2 (VPACK_D4_*), so they are exercised too.
PACK_WIDTHS = [
    (32, 16),
    (32, 8),
    (16, 8),
    (16, 4),
    (8, 4),
]

# (sat, symsat) -> Truncate / Saturate / SymmetricSaturate.
PACK_SAT_CONFIGS = [
    (False, False),  # truncate
    (True, False),   # saturate
    (True, True),    # symmetric saturate
]


def pack_boundary_values(bits_o, signed):
    """Inputs that stress the saturate/truncate boundaries for a given output."""
    if signed:
        vmax = (1 << (bits_o - 1)) - 1
        vmin = -(1 << (bits_o - 1))
    else:
        vmax = (1 << bits_o) - 1
        vmin = 0

    vals = [0, 1, -1, vmax, vmin, vmax + 1, vmin - 1, vmax - 1, vmin + 1]
    # Symmetric-saturation min boundary (signed): -(2^(n-1)-1).
    if signed:
        vals.extend([-vmax, -vmax - 1, -vmax + 1])
    # Far out of range, both directions -- exercises truncate-wrap vs clamp.
    vals.extend([vmax * 4, vmin * 4 - 1, (1 << 30), -(1 << 30), (1 << 40), -(1 << 40)])
    # Powers of two spanning the input width.
    for p in range(0, 34, 2):
        vals.extend([1 << p, -(1 << p)])

    return list(set(vals))


def gen_pack_golden():
    """Generate pack (narrowing) golden cases against the genuine model."""
    cases = []

    for bits_i, bits_o in PACK_WIDTHS:
        for signed in [True, False]:
            for sat, symsat in PACK_SAT_CONFIGS:
                boundary_vals = pack_boundary_values(bits_o, signed)
                for val in boundary_vals:
                    r = int(PACK.pack_lane(val, bits_i, bits_o, signed, sat, symsat))
                    cases.append({
                        "value": int(val),
                        "bits_i": bits_i,
                        "bits_o": bits_o,
                        "signed": signed,
                        "sat": sat,
                        "symsat": symsat,
                        "expected": r,
                    })

                # Random values spanning a wide accumulator range.
                for _ in range(20):
                    val = rng.randint(-(1 << 40), (1 << 40) - 1)
                    r = int(PACK.pack_lane(val, bits_i, bits_o, signed, sat, symsat))
                    cases.append({
                        "value": int(val),
                        "bits_i": bits_i,
                        "bits_o": bits_o,
                        "signed": signed,
                        "sat": sat,
                        "symsat": symsat,
                        "expected": r,
                    })

    print(f"  Pack: {len(cases)} cases across {len(PACK_WIDTHS)} width pairs")
    return cases


# ---------------------------------------------------------------------------
# BF16 conversion golden data (Tier 2c): f32 -> bf16 with rounding
# ---------------------------------------------------------------------------

# The Rust seam: vector_float::f32_to_bf16(f32::from_bits(value), mode) -> u16.
# The genuine model: srs.srs_bf_lane(a_fp32_bits, rnd, flags) -> bf16 bits.
# (srs_bf_lane uses sgn_mag=True rounding; flags out-param is ignored here --
# we only compare the converted bf16 bit pattern.)


def fp32_bits(sgn, exp, man):
    return ((int(bool(sgn)) << 31) | ((exp & 0xFF) << 23) | (man & 0x7FFFFF)) & 0xFFFFFFFF


def bf16_srs_input_patterns():
    """fp32 bit patterns that stress the f32->bf16 rounding boundary.

    Rounding reads mantissa bits: guard=bit15, sticky=bits14:0, lsb=bit16.
    So the low 17 mantissa bits are what matter; sweep them against a spread
    of exponents and both signs, plus inf/NaN/denorm edges.
    """
    pats = set()

    # Mantissa patterns exercising guard/sticky/lsb and overflow-on-roundup.
    mans = [
        0x000000,            # exact (no discarded bits)
        0x008000,            # halfway (guard only)
        0x008001,            # past halfway (guard + sticky)
        0x007FFF,            # just below halfway
        0x010000,            # lsb=1, no guard
        0x018000,            # lsb=1 + halfway (ties-to-even pivot)
        0x004000, 0x004001,  # below halfway, with/without sticky
        0x7F0000, 0x7F8000, 0x7F8001,
        0x7FFFFF,            # all ones -- roundup overflows mantissa -> exp++
        0x7F7FFF, 0x400000, 0x3F8000,
    ]
    exps = [0, 1, 64, 126, 127, 128, 200, 253, 254, 255]
    for sgn in (0, 1):
        for exp in exps:
            for man in mans:
                pats.add(fp32_bits(sgn, exp, man))

    # Explicit NaN cases where truncation would drop NaN-ness (low-16-only man).
    for sgn in (0, 1):
        pats.add(fp32_bits(sgn, 255, 0x000001))   # NaN, man only in low bits
        pats.add(fp32_bits(sgn, 255, 0x008000))   # NaN, guard bit
        pats.add(fp32_bits(sgn, 255, 0x400000))   # NaN, top mantissa bit
        pats.add(fp32_bits(sgn, 255, 0x000000))   # inf (exp=255, man=0)

    # Dense denormal sweep (exp=0) straddling the guard (bit15) and lsb (bit16)
    # boundaries -- the FTZ diagnostic region. The model ROUNDS these (mode-
    # dependent: round-up modes give a nonzero bf16, round-down give 0), while the
    # execute path FTZs every denormal to signed zero. Silicon adjudicates per mode.
    denorm_mans = [0x000001, 0x002000, 0x004000, 0x006000, 0x007FFF,
                   0x008000, 0x008001, 0x00A000, 0x00C000, 0x00FFFF,
                   0x010000, 0x014000, 0x018000, 0x01C000, 0x020000,
                   0x030000, 0x040000, 0x07FFFF]
    for sgn in (0, 1):
        for man in denorm_mans:
            pats.add(fp32_bits(sgn, 0, man))   # exp=0 -> denormal

    # Richer NaN payloads (exp=255, man!=0), both signs: probe whether silicon
    # preserves the payload (like f32_to_bf16) or canonicalizes it.
    nan_mans = [0x000001, 0x000040, 0x004000, 0x008000, 0x080000,
                0x200000, 0x400000, 0x7FFFFF]
    for sgn in (0, 1):
        for man in nan_mans:
            pats.add(fp32_bits(sgn, 255, man))

    # Random fp32 bit patterns (seeded).
    for _ in range(200):
        pats.add(rng.randint(0, (1 << 32) - 1))

    return sorted(pats)


def gen_bf16_srs_golden():
    """Generate f32->bf16 conversion golden cases against the genuine model."""
    cases = []
    inputs = bf16_srs_input_patterns()
    for a in inputs:
        for rnd_mode in RND_MODES:
            flags = [False] * 5
            r = int(SRS.srs_bf_lane(a, rnd_mode, flags)) & 0xFFFF
            cases.append({
                "value": int(a),
                "rnd": rnd_mode,
                "expected": r,
            })
    print(f"  BF16-SRS: {len(cases)} cases ({len(inputs)} inputs x {len(RND_MODES)} modes)")
    return cases


# ---------------------------------------------------------------------------
# MatMul golden data (Tier 4): value differential vs INDEPENDENT oracles
# ---------------------------------------------------------------------------
#
# The live seam is matmul_config_driven (generic row-major tiling). mulmac.py
# (the gate-level model) cannot be driven -- it imports kernels.helpers
# (absent from aietools) and needs the full constants.py geometry tables. So
# the value-level oracles here are:
#   - integer paths: INDEPENDENT integer multiply-accumulate (plain Python),
#     wrapped to the accumulator width. No aietools dependency.
#   - bf16 path: the GENUINE bfloat16.py:bf16_mac_hw hardware MAC reference.
# Both oracles are geometry-free (flat element lists), so they don't touch the
# suspect constants.py port. The geometry is encoded here from the documented
# AIE2 tile shapes; the Rust side derives geometry from MatMulConfig::from_types
# and cross-checks it against these fields (a mismatch fails the test loudly).

# (a_type, b_type, rows, inner, cols, bits_x, bits_y, acc_width, bfloat)
# Tile shapes are the DEFAULT (first-matching) entries of the AIE2
# DENSE_GEOMETRY_TABLE that MatMulConfig::from_types selects per (bits_x,
# bits_y, bfloat). The Rust side derives geometry from from_types and asserts
# it equals these fields -- a mismatch fails the test loudly, cross-checking
# the geometry itself.
MATMUL_GEOMS = [
    ("Int8",     "Int8",     4, 8, 8, 8,  8,  "Acc32", False),  # mmode 1
    ("Int16",    "Int16",    4, 2, 8, 16, 16, "Acc32", False),  # mmode 3
    ("Int32",    "Int16",    4, 2, 4, 32, 16, "Acc64", False),  # mmode 7
    ("BFloat16", "BFloat16", 4, 8, 4, 16, 16, "Acc32", True),   # mmode 6
]


def _to_i32(v):
    return ((v + (1 << 31)) % (1 << 32)) - (1 << 31)


def _pack_elem(buf, byte_off, val, nbytes):
    v = val & ((1 << (8 * nbytes)) - 1)
    for i in range(nbytes):
        buf[byte_off + i] = (v >> (8 * i)) & 0xFF


def _buf_to_vec512(buf):
    return [buf[i * 4] | (buf[i * 4 + 1] << 8) | (buf[i * 4 + 2] << 16) | (buf[i * 4 + 3] << 24)
            for i in range(16)]


def _int_range(bits, signed):
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def _sample_int_matrix(n, bits, signed, kind, vmin, vmax):
    if kind == "zeros":
        return [0] * n
    if kind == "max":
        return [vmax] * n
    if kind == "min":
        return [vmin] * n
    if kind == "alt":
        return [vmax if i % 2 == 0 else vmin for i in range(n)]
    return [rng.randint(vmin, vmax) for _ in range(n)]


def _nice_bf16_values():
    # Small representable bf16 patterns (exact integers/halves) + a few randoms.
    vals = [0x0000, 0x3F80, 0xBF80, 0x4000, 0xC000, 0x3F00, 0xBF00,  # 0,1,-1,2,-2,.5,-.5
            0x4040, 0x4080, 0x40A0, 0x3E80, 0x4120]                  # 3,4,5,.25,10
    return vals


def gen_matmul_golden():
    """MatMul value differential: int vs independent arithmetic, bf16 vs the
    genuine bf16_mac_hw."""
    cases = []
    for (at, bt, rows, inner, cols, bx, by, accw, bfloat) in MATMUL_GEOMS:
        na = rows * inner
        nb = inner * cols
        bytes_x = bx // 8
        bytes_y = by // 8

        if bfloat:
            # bf16: sweep nice value pairs + random bf16-bit matrices.
            nice = _nice_bf16_values()
            mats = []
            mats.append(([0x3F80] * na, [0x3F80] * nb))            # all ones
            mats.append(([nice[i % len(nice)] for i in range(na)],
                         [nice[(i * 3) % len(nice)] for i in range(nb)]))
            for _ in range(24):
                mats.append(([rng.choice(nice) for _ in range(na)],
                             [rng.choice(nice) for _ in range(nb)]))
            for _ in range(12):
                mats.append(([rng.randint(0, 0xFFFF) for _ in range(na)],
                             [rng.randint(0, 0xFFFF) for _ in range(nb)]))
            for (av, bv) in mats:
                abuf = bytearray(64)
                bbuf = bytearray(64)
                for idx, v in enumerate(av):
                    _pack_elem(abuf, idx * 2, v, 2)
                for idx, v in enumerate(bv):
                    _pack_elem(bbuf, idx * 2, v, 2)
                # Independent per-lane oracle: genuine bf16_mac_hw(q=0, A row, B col).
                exp = []
                for r in range(rows):
                    for c in range(cols):
                        a_elems = [av[r * inner + k] for k in range(inner)]
                        b_elems = [bv[k * cols + c] for k in range(inner)]
                        exp.append(int(BF16.bf16_mac_hw(0, a_elems, b_elems)) & 0xFFFFFFFF)
                cases.append({
                    "a_type": at, "b_type": bt, "rows": rows, "inner": inner, "cols": cols,
                    "bits_x": bx, "bits_y": by, "acc_width": accw, "bfloat": True,
                    "x_signed": True, "y_signed": True, "subtract": False,
                    "a": _buf_to_vec512(abuf), "b": _buf_to_vec512(bbuf), "expected": exp,
                })
            continue

        # Integer paths: sweep sign combos, subtract, and stress matrices.
        for x_signed in (True, False):
            for y_signed in (True, False):
                for subtract in (False, True):
                    vminx, vmaxx = _int_range(bx, x_signed)
                    vminy, vmaxy = _int_range(by, y_signed)
                    kinds = ["zeros", "max", "min", "alt", "rand", "rand", "rand",
                             "rand", "rand", "rand"]
                    for ak in kinds:
                        for bk in kinds[:5]:
                            av = _sample_int_matrix(na, bx, x_signed, ak, vminx, vmaxx)
                            bv = _sample_int_matrix(nb, by, y_signed, bk, vminy, vmaxy)
                            abuf = bytearray(64)
                            bbuf = bytearray(64)
                            for idx, v in enumerate(av):
                                _pack_elem(abuf, idx * bytes_x, v, bytes_x)
                            for idx, v in enumerate(bv):
                                _pack_elem(bbuf, idx * bytes_y, v, bytes_y)
                            exp = []
                            for r in range(rows):
                                for c in range(cols):
                                    s = sum(av[r * inner + k] * bv[k * cols + c]
                                            for k in range(inner))
                                    if subtract:
                                        s = -s
                                    exp.append(_to_i32(s) if accw == "Acc32" else s)
                            cases.append({
                                "a_type": at, "b_type": bt, "rows": rows, "inner": inner,
                                "cols": cols, "bits_x": bx, "bits_y": by, "acc_width": accw,
                                "bfloat": False, "x_signed": x_signed, "y_signed": y_signed,
                                "subtract": subtract,
                                "a": _buf_to_vec512(abuf), "b": _buf_to_vec512(bbuf),
                                "expected": exp,
                            })
    print(f"  MatMul: {len(cases)} cases across {len(MATMUL_GEOMS)} geometries")
    return cases


# ---------------------------------------------------------------------------
# Element-wise vector ops golden data (Tier 3)
# ---------------------------------------------------------------------------

def to_signed(val, bits):
    """Interpret unsigned value as signed with given bit width."""
    mask = (1 << bits) - 1
    val = val & mask
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def to_unsigned(val, bits):
    """Wrap signed value to unsigned with given bit width."""
    return val & ((1 << bits) - 1)


def gen_elementwise_vectors(bits, count=20):
    """Generate pairs of vectors as lists of lane values."""
    lanes = 256 // bits
    pairs = []

    # All zeros.
    pairs.append(([0] * lanes, [0] * lanes))

    # All max.
    vmax = (1 << bits) - 1
    pairs.append(([vmax] * lanes, [vmax] * lanes))

    # Sequential vs constant.
    pairs.append(([i % (1 << bits) for i in range(lanes)],
                  [1] * lanes))

    # Alternating patterns.
    pairs.append(([0 if i % 2 == 0 else vmax for i in range(lanes)],
                  [vmax if i % 2 == 0 else 0 for i in range(lanes)]))

    # Random pairs.
    for _ in range(count):
        a = [rng.randint(0, vmax) for _ in range(lanes)]
        b = [rng.randint(0, vmax) for _ in range(lanes)]
        pairs.append((a, b))

    return pairs


def pack_lanes_to_u32x8(lanes, bits):
    """Pack a list of lane values into [u32; 8] representation."""
    result = [0] * 8
    lanes_per_word = 32 // bits
    for i, val in enumerate(lanes):
        word_idx = (i * bits) // 32
        bit_offset = (i * bits) % 32
        result[word_idx] |= (val & ((1 << bits) - 1)) << bit_offset
    # Ensure all values are proper u32.
    result = [v & 0xFFFFFFFF for v in result]
    return result


def elementwise_op(op_name, a_lanes, b_lanes, bits, signed):
    """Compute element-wise operation on lane values."""
    mask = (1 << bits) - 1
    result = []

    for av, bv in zip(a_lanes, b_lanes):
        if signed:
            sa = to_signed(av, bits)
            sb = to_signed(bv, bits)
        else:
            sa = av & mask
            sb = bv & mask

        if op_name == "add":
            r = (sa + sb) & mask
        elif op_name == "sub":
            r = (sa - sb) & mask
        elif op_name == "mul":
            r = (sa * sb) & mask
        elif op_name == "min":
            r = to_unsigned(min(sa, sb), bits)
        elif op_name == "max":
            r = to_unsigned(max(sa, sb), bits)
        else:
            raise ValueError(f"Unknown op: {op_name}")

        result.append(r & mask)

    return result


# Map of (op_name, bits, signed) -> Operation variant name.
# Only integer types -- float needs special handling.
ELEMENTWISE_CONFIGS = []
for op_name in ["add", "sub", "mul", "min", "max"]:
    for bits, signed, type_name in [
        (32, True, "Int32"),
        (32, False, "UInt32"),
        (16, True, "Int16"),
        (16, False, "UInt16"),
        (8, True, "Int8"),
        (8, False, "UInt8"),
    ]:
        ELEMENTWISE_CONFIGS.append((op_name, bits, signed, type_name))


def gen_elementwise_golden():
    """Generate element-wise operation golden data."""
    result = {}
    total = 0

    for op_name, bits, signed, type_name in ELEMENTWISE_CONFIGS:
        key = f"v{op_name}_{type_name}"
        cases = []
        pairs = gen_elementwise_vectors(bits)

        for a_lanes, b_lanes in pairs:
            expected_lanes = elementwise_op(op_name, a_lanes, b_lanes, bits, signed)
            a_packed = pack_lanes_to_u32x8(a_lanes, bits)
            b_packed = pack_lanes_to_u32x8(b_lanes, bits)
            expected_packed = pack_lanes_to_u32x8(expected_lanes, bits)

            cases.append({
                "a": a_packed,
                "b": b_packed,
                "expected": expected_packed,
            })

        result[key] = cases
        total += len(cases)

    print(f"  Element-wise: {total} cases across {len(ELEMENTWISE_CONFIGS)} type/op combos")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating vector compute golden data...")
    print(f"  oracle: {ORACLE_MODEL}")

    golden = {}
    golden["srs"] = gen_srs_golden()
    golden["ups"] = gen_ups_golden()
    golden["pack"] = gen_pack_golden()
    golden["bf16_srs"] = gen_bf16_srs_golden()
    golden["matmul"] = gen_matmul_golden()
    golden.update(gen_elementwise_golden())

    out_path = os.path.join(os.path.dirname(__file__), "golden", "vector_ops.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(golden, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")

    # Print summary.
    print(f"\nSummary:")
    print(f"  SRS:  {len(golden['srs']):>6} cases")
    print(f"  UPS:  {len(golden['ups']):>6} cases")
    print(f"  Pack: {len(golden['pack']):>6} cases")
    print(f"  BF16: {len(golden['bf16_srs']):>6} cases")
    print(f"  MatMul:{len(golden['matmul']):>5} cases")
    elem_total = sum(len(v) for k, v in golden.items() if k.startswith("v"))
    print(f"  Elem: {elem_total:>6} cases")
    grand = (len(golden['srs']) + len(golden['ups']) + len(golden['pack'])
             + len(golden['bf16_srs']) + len(golden['matmul']) + elem_total)
    print(f"  Total:{grand:>6} cases")


if __name__ == "__main__":
    main()
