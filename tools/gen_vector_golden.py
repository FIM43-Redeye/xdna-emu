#!/usr/bin/env python3
"""Generate golden test data for vector compute validation.

Generates deterministic test vectors for SRS, UPS, and element-wise vector
operations. SRS and UPS use the aietools Python model as the oracle; element-
wise ops use straightforward Python arithmetic.

Usage:
    cd tools && python3 gen_vector_golden.py

Output:
    tools/golden/vector_ops.json

The generated file is checked in and loaded by Rust tests at cargo test time,
so there is no aietools dependency at test time.
"""

import json
import os
import random
import sys

# ---------------------------------------------------------------------------
# SRS / UPS oracle functions
# ---------------------------------------------------------------------------

# The aietools model files are Python 2 and require extensive fixups for
# Python 3 (print statements, integer division throughout). Rather than
# importing them, we reimplement the small subset we need: srs_lane(),
# ups_lane(), and trnc(). These are direct translations of the aietools
# Python model, verified against the same algorithm our Rust code implements.
#
# Source functions (read-only reference, not copied):
#   aietools/data/aie_ml/lib/python_model/model/srs.py:srs_lane
#   aietools/data/aie_ml/lib/python_model/model/ups.py:ups_lane
#   aietools/data/aie_ml/lib/python_model/model/helpers.py:trnc


def trnc(a, sgn, bits):
    """Truncate to `bits` width with optional sign extension (from helpers.py)."""
    if bits == 0:
        return 0
    msk = (1 << bits) - 1
    a = int(a) & msk
    s = (a >> (bits - 1)) & 1
    if s == 1 and sgn:
        a = a | ~msk
    return a


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


def srs_round(rnd, sgn, lsb, grd, stk):
    """SRS rounding decision (from srs.py:srs_round, sgn_mag=False)."""
    rnd_halfway = rnd in (RND_NEG_INF, RND_POS_INF, RND_SYM_INF,
                          RND_SYM_ZERO, RND_CONV_EVEN, RND_CONV_ODD)
    symmetric = rnd in (RND_SYM_FLOOR, RND_SYM_CEIL, RND_SYM_INF, RND_SYM_ZERO)
    otherdir = rnd in (RND_CEIL, RND_SYM_CEIL, RND_POS_INF, RND_SYM_INF, RND_CONV_ODD)
    convergent = rnd in (RND_CONV_EVEN, RND_CONV_ODD)

    if convergent:
        det = lsb
    elif symmetric:
        det = sgn
    else:
        det = False
    if otherdir:
        det = not det

    if rnd_halfway:
        return grd and (det or stk)
    else:
        return det and (grd or stk)


def py_srs_lane(a_in, shft, bits_i, bits_o, sgn_o, sat, symsat, rnd):
    """SRS per-lane computation (from srs.py:srs_lane).

    Arguments match the aietools Python model:
      a_in:    input accumulator value
      shft:    total shift amount (user_shift + BIAS)
      bits_i:  input width (unused, we work on arbitrary-precision int)
      bits_o:  output bit width
      sgn_o:   True if output is signed
      sat:     True to clamp to output range
      symsat:  True for symmetric saturation (signed min = -(2^(n-1)-1))
      rnd:     rounding mode constant (0-3, 8-13)

    Returns (result, overflow_flag).
    """
    BIAS = 4
    a = int(a_in) << BIAS
    a_sft = a >> shft

    # Rounding signals.
    if shft > 0:
        stk = trnc((a << 1), False, shft) != 0
        grd = bool(((a << 1) >> shft) & 0x1)
    else:
        stk = False
        grd = False
    lsb = bool(((a << 1) >> (shft + 1)) & 0x1)
    sgn = a < 0

    p1 = srs_round(rnd, sgn, lsb, grd, stk)
    a_rnd = a_sft + 1 if p1 else a_sft

    # Saturation bounds.
    if sgn_o:
        vmax = 2 ** (bits_o - 1) - 1
        vmin_t = -(2 ** (bits_o - 1))
        vmin = vmin_t + (1 if symsat else 0)
    else:
        vmax = 2 ** bits_o - 1
        vmin = 0
        vmin_t = vmin

    a_sft_flt = (int(a_in) << BIAS) / 2.0 ** shft

    of = False
    a_sat = a_rnd
    if a_rnd > vmax:
        of = True
        if sat:
            a_sat = vmax
    elif a_rnd < vmin:
        if sat:
            a_sat = vmin

    of = of or a_sft_flt < vmin
    r = trnc(a_sat, sgn_o, bits_o)
    return (r, of)


def py_ups_lane(a, shft, bits_o, sat):
    """UPS per-lane computation (from ups.py:ups_lane)."""
    u = int(a) << shft

    if sat:
        vmax = 2 ** (bits_o - 1) - 1
        vmin = -(2 ** (bits_o - 1))
        u = max(min(u, vmax), vmin)

    r = trnc(u, True, bits_o)
    return r

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
# The Python srs_lane() signature:
#   srs_lane(a_in, shft, bits_i, bits_o, sgn_o, sat, symsat, rnd)
#   where shft = user_shift + BIAS (BIAS=4).
#
# Our Rust function adds BIAS internally, so we record user_shift in JSON
# and pass user_shift + BIAS to the oracle.

BIAS = 4
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
    """Generate SRS golden test cases."""
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
                        py_result, _ = py_srs_lane(
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
                    py_result, _ = py_srs_lane(
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
# The Python ups_lane() signature:
#   ups_lane(a, shft, bits_o, sat) -> int
#   It takes the already-sign-extended input value.

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
    """Generate UPS golden test cases."""
    cases = []

    for bits_in, bits_out, _lanes in UPS_MODES:
        for signed in [True, False]:
            for sat in [True, False]:
                boundary_vals = ups_boundary_values(bits_in, signed)

                for shift in [0, 1, 4, 8, 12]:
                    for val in boundary_vals:
                        # Sign-extend the input value to bits_in width,
                        # matching what the Rust code does.
                        truncated = int(trnc(val, signed, bits_in))
                        py_result = int(py_ups_lane(
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
                        py_result = int(py_ups_lane(
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

    golden = {}
    golden["srs"] = gen_srs_golden()
    golden["ups"] = gen_ups_golden()
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
    elem_total = sum(len(v) for k, v in golden.items() if k.startswith("v"))
    print(f"  Elem: {elem_total:>6} cases")
    print(f"  Total:{len(golden['srs']) + len(golden['ups']) + elem_total:>6} cases")


if __name__ == "__main__":
    main()
