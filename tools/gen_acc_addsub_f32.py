#!/usr/bin/env python3
"""Generate the fp32 accumulator add/sub capture kernel (vec_acc_addsub_f32).

Why this exists
---------------
The accumulator-to-accumulator float instructions vadd.f, vsub.f, vnegadd.f and
vnegsub.f, and the float negate vneg.f, operate on a 512-bit fp32 accumulator
(bm, 16 lanes). Nothing in the repository pinned their rounding, alignment,
signed-zero, denormal, overflow or NaN behaviour, so the emulator refuses them.
This kernel runs every form over one designed batch of lane pairs so a single
silicon capture settles all of it.

One tile is (A[16], B[16]): two fp32 accumulators whose 16 lanes are independent
(a, b) operand pairs. Every tile runs through every form in FORMS; form f of
tile t writes 16 fp32 words at out[(t * NF + f) * 16 ...]. The two identity forms
(id_a, id_b) store the operands straight back, so any input-path transformation
(canonicalisation, flushing) shows up as a mismatch between the golden inputs
and the identity outputs rather than being misread as an adder effect.

The *_conf forms drive the config word (aiev2 vector config, see
AIE2InstrPatterns.td VecConf): bit 0 dynZeroAccum, bit 12 dynAcc0Neg (sub_acc1),
bit 13 dynAcc1Neg (sub_acc2); the intrinsics supply the fp32 mode bits.

The inputs are DESIGNED (sections below), seeded and reproducible; the expected
output is whatever NPU1 silicon produces (`capture` writes the golden JSON the
emulator tests read). No value is computed by a model.

Usage
-----
  python3 tools/gen_acc_addsub_f32.py generate          # run.lit, aie.mlir, kernel .cc
  python3 tools/stage_vector_kernel.py vec_acc_addsub_f32   # no exp.bin yet
  # run the bridge on HW once (an uncaptured kernel fails by design; out.txt is
  # written first), then:
  python3 tools/gen_acc_addsub_f32.py capture <build>/out.txt --date YYYY-MM-DD
  python3 tools/stage_vector_kernel.py vec_acc_addsub_f32   # now stages exp.bin = silicon
"""

import argparse
import os
import random
from dataclasses import dataclass

import capture_kernel_render as render

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = "vec_acc_addsub_f32"
OUT_DIR = os.path.join(REPO, "tests", "vector-verify", KERNEL)
GOLDEN = os.path.join(REPO, "tools", "golden", "silicon", KERNEL + ".json")

T = 16            # tiles per kernel iteration (one objectfifo buffer)
LANES = 16        # fp32 lanes per accumulator
MAX_CHUNKS = 64   # shim DMA repeat-count limit


@dataclass(frozen=True)
class Form:
    """One accumulator form: what the opcode implies plus the config controls."""
    name: str
    expr: str
    instr: str           # id, vadd, vsub, vnegadd, vnegsub, vneg
    zero_acc1: int = 0
    sub_acc1: int = 0
    sub_acc2: int = 0

    def as_json(self):
        return {k: v for k, v in self.__dict__.items() if k != "expr"}


def _binary(name, fn, instr, z=0, s1=0, s2=0):
    if (z, s1, s2) == (0, 0, 0):
        return Form(name, f"::{fn}(a, b)", instr)
    return Form(name, f"::{fn}_conf(a, b, {z}, {s1}, {s2})", instr, z, s1, s2)


FORMS = [
    Form("id_a", "a", "id"),
    Form("id_b", "b", "id"),
    _binary("add", "add", "vadd"),
    _binary("sub", "sub", "vsub"),
    _binary("negadd", "negadd", "vnegadd"),
    _binary("negsub", "negsub", "vnegsub"),
    Form("neg", "::neg(a)", "vneg"),
    _binary("add_z1", "add", "vadd", z=1),
    _binary("add_s1", "add", "vadd", s1=1),
    _binary("add_s2", "add", "vadd", s2=1),
    _binary("add_s12", "add", "vadd", s1=1, s2=1),
    _binary("sub_z1", "sub", "vsub", z=1),
    _binary("sub_s1", "sub", "vsub", s1=1),
    _binary("sub_s2", "sub", "vsub", s2=1),
    _binary("negadd_s1", "negadd", "vnegadd", s1=1),
    _binary("negadd_s2", "negadd", "vnegadd", s2=1),
    _binary("negsub_s1", "negsub", "vnegsub", s1=1),
    _binary("negsub_s2", "negsub", "vnegsub", s2=1),
    Form("neg_z1", "::neg_conf(a, 1, 0)", "vneg", zero_acc1=1),
    Form("neg_s1", "::neg_conf(a, 0, 1)", "vneg", sub_acc1=1),
]
NF = len(FORMS)


# --- Designed inputs -----------------------------------------------------------

def f32(sign, exp, mant):
    return (sign << 31) | (exp << 23) | mant


INF = f32(0, 255, 0)
QNAN = 0x7FC00000
SNAN = 0x7F800001
MAX = f32(0, 254, 0x7FFFFF)
MIN_NORMAL = f32(0, 1, 0)


def neg(x):
    return x ^ 0x80000000


def sign_variants(a, b):
    """The four sign combinations of a lane pair given as magnitudes."""
    return [(a, b), (neg(a), b), (a, neg(b)), (neg(a), neg(b))]


def section_gap(rng):
    """Every exponent difference 0..40 plus huge ones, with mantissa patterns.

    The big operand has leading bit at E; the small one is E-gap with a power
    of two, all ones, half-way and sparse mantissa; both signs, either slot.
    """
    gaps = list(range(0, 41)) + [48, 60, 80, 100, 126, 150, 200, 253]
    big_m = (0, 0x7FFFFF, 0x2AAAAA)
    small_m = (0, 0x7FFFFF, 0x400000, 0x155555)
    pairs = []
    for gap in gaps:
        e_big = min(254, max(gap + 1, 100 + (gap * 13) % 100))
        for bm in big_m:
            for sm in small_m:
                big = f32(0, e_big, bm)
                small = f32(0, e_big - gap, sm)
                for a, b in sign_variants(big, small):
                    pairs.append((a, b))
                    pairs.append((b, a))
    return pairs


def section_cancel(rng):
    """Exact and near cancellation, results in the denormal range, 2^-126."""
    pairs = []
    # exact cancellation, same and opposite signs, many exponents
    for _ in range(48):
        x = f32(rng.randrange(2), rng.randrange(1, 255), rng.randrange(1 << 23))
        pairs += [(x, x), (x, neg(x)), (neg(x), x)]
    # cancellation with one-to-three ulp residue, same binade and across a binade
    for _ in range(96):
        e = rng.randrange(2, 254)
        m = rng.randrange(1 << 23)
        x = f32(0, e, m)
        d = rng.choice((1, 2, 3, 5, 0x100, 0x7FFFFF))
        y = (x - d) if m >= d else f32(0, e - 1, (m - d) & 0x7FFFFF)
        pairs += [(x, neg(y)), (neg(x), y)]
    # leading-one just straddling: 1.0 against 0.99999994 and neighbours
    for e in (2, 30, 127, 200, 253):
        for m in (0, 1, 0x400000, 0x7FFFFE, 0x7FFFFF):
            pairs.append((f32(0, e, 0), neg(f32(0, e - 1, m))))
    # normalisation shifts of every distance: 1.m - 0.5.m' with a close tail
    for k in range(0, 24):
        pairs.append((f32(0, 127, 0), neg(f32(0, 126, 0x7FFFFF - ((1 << k) - 1)))))
        pairs.append((f32(0, 127, (1 << k) - 1), neg(f32(0, 127, 0))))
    # results around and below the smallest normal 2^-126
    mants = (0, 1, 2, 0x400000, 0x7FFFFF, 0x7FFFFE, 0x155555)
    for ea, eb in ((1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (3, 3), (4, 3), (4, 2), (5, 4)):
        for ma in mants:
            for mb in mants:
                pairs.append((f32(0, ea, ma), f32(1, eb, mb)))
    # the exact 2^-126 landings
    pairs += [(f32(0, 2, 0), f32(1, 1, 0)), (f32(1, 2, 0), f32(0, 1, 0)),
              (f32(0, 1, 0x400000), f32(1, 1, 0x000000)), (f32(0, 1, 0), f32(0, 0, 0x400000)),
              (f32(0, 1, 0), f32(1, 0, 0x400000)), (f32(0, 1, 0), f32(1, 0, 0x7FFFFF)),
              (f32(0, 1, 0), f32(1, 0, 1)), (f32(0, 0, 0x7FFFFF), f32(0, 0, 1)),
              (f32(0, 0, 0x400000), f32(0, 0, 0x400000)), (f32(0, 1, 1), f32(1, 1, 0))]
    return pairs


def section_ties(rng):
    """Rounding ties and near-ties after alignment: guard and sticky patterns."""
    pairs = []
    small_m = (0, 1, 2, 0x200000, 0x3FFFFF, 0x400000, 0x7FFFFF)
    for gap in range(20, 31):
        for lsb in (0, 1):
            for _rep in range(2):
                e = rng.randrange(60, 200)
                bm = (rng.randrange(1 << 22) << 1) | lsb
                big = f32(0, e, bm)
                for sm in small_m:
                    small = f32(0, e - gap, sm)
                    for a, b in sign_variants(big, small):
                        pairs.append((a, b))
    return pairs


def section_overflow():
    """Sums reaching and passing the largest finite value; subtracts that do not."""
    pairs = []
    for ea in (254, 253, 252):
        for ma in (0, 0x400000, 0x7FFFFE, 0x7FFFFF):
            for eb in (254, 253, 252):
                for mb in (0, 0x400000, 0x7FFFFE, 0x7FFFFF):
                    for a, b in sign_variants(f32(0, ea, ma), f32(0, eb, mb)):
                        pairs.append((a, b))
    # just under and at the half-ulp boundary of the overflow rounding
    for gap in (22, 23, 24, 25, 26):
        for sm in (0, 1, 0x7FFFFF):
            pairs.append((MAX, f32(0, 254 - gap, sm)))
            pairs.append((neg(MAX), f32(1, 254 - gap, sm)))
    return pairs


def section_denormals(rng):
    """Denormal operands against denormals and against normals of many exponents."""
    dm = (1, 2, 0x3FFFFF, 0x400000, 0x7FFFFF, 0x123456)
    normals = [f32(0, e, m) for e in (1, 2, 24, 25, 127, 254) for m in (0, 0x7FFFFF, 5)]
    others = [f32(0, 0, m) for m in dm] + normals
    pairs = []
    for m in dm:
        d = f32(0, 0, m)
        for o in others:
            for a, b in sign_variants(d, o):
                pairs.append((a, b))
                pairs.append((b, a))
    return pairs


def section_zeros():
    """Signed zeros against every class: the sign of an exact-zero result."""
    cls = [f32(0, 0, 0), f32(1, 0, 0), f32(0, 0, 1), f32(1, 0, 0x7FFFFF), f32(0, 127, 0),
           f32(1, 127, 0), MIN_NORMAL, MAX, INF, neg(INF), QNAN, 0x3F800001]
    return [(a, b) for a in cls for b in cls]


def section_specials(rng):
    """Inf, NaN payloads and signs in every operand position."""
    alpha = [f32(0, 0, 0), f32(1, 0, 0), f32(0, 127, 0), f32(1, 127, 0), INF, neg(INF),
             QNAN, neg(QNAN), SNAN, neg(SNAN), 0x7FA55555, 0xFFC12345, 0x7FFFFFFF, 0xFF800001,
             f32(0, 0, 1), MAX]
    pairs = [(a, b) for a in alpha for b in alpha]
    for _ in range(192):
        nan = f32(rng.randrange(2), 255, rng.randrange(1, 1 << 23))
        other = rng.choice((nan, f32(rng.randrange(2), 255, rng.randrange(1, 1 << 23)),
                            INF, neg(INF), f32(rng.randrange(2), rng.randrange(0, 255),
                                              rng.randrange(1 << 23))))
        pairs.append((nan, other) if rng.random() < 0.5 else (other, nan))
    for _ in range(64):
        inf = f32(rng.randrange(2), 255, 0)
        fin = f32(rng.randrange(2), rng.randrange(0, 255), rng.randrange(1 << 23))
        pairs.append((inf, fin) if rng.random() < 0.5 else (fin, inf))
    return pairs


def section_random(rng, count):
    """Seeded fill: near-exponent pairs, wide-spread pairs, zero/denormal mixes."""
    pairs = []
    for i in range(count):
        kind = i % 4
        if kind in (0, 1):
            e = rng.randrange(30, 225)
            ea = e
            eb = max(1, min(254, e + rng.randrange(-30, 31)))
        elif kind == 2:
            ea, eb = rng.randrange(1, 255), rng.randrange(1, 255)
        else:
            ea, eb = rng.choice((0, 0, 1, 2, rng.randrange(1, 255))), rng.choice((0, 1, 2, rng.randrange(1, 255)))
        pairs.append((f32(rng.randrange(2), ea, rng.randrange(1 << 23)),
                      f32(rng.randrange(2), eb, rng.randrange(1 << 23))))
    return pairs


def design():
    """The designed batch as tiles of (A[16], B[16]), padded to whole chunks."""
    rng = random.Random(0xACC5F32)
    pairs = []
    pairs += section_gap(rng)
    pairs += section_cancel(rng)
    pairs += section_ties(rng)
    pairs += section_overflow()
    pairs += section_denormals(rng)
    pairs += section_zeros()
    pairs += section_specials(rng)
    pairs += section_random(rng, 2048)
    while len(pairs) % (T * LANES):
        pairs.append((0, 0))
    assert len(pairs) // (T * LANES) <= MAX_CHUNKS, len(pairs)
    assert all(0 <= x < 1 << 32 for p in pairs for x in p)
    return [([p[0] for p in pairs[i:i + LANES]], [p[1] for p in pairs[i:i + LANES]])
            for i in range(0, len(pairs), LANES)]


# --- Rendering -----------------------------------------------------------------

def render_kernel():
    body = "".join(
        "    { aie::accum<accfloat, 16> r = %s; aie::store_v(o + %d * 16, r.to_vector<float>()); }\n"
        % (f.expr, i)
        for i, f in enumerate(FORMS))
    return f"""//===- acc_addsub.cc --------------------------------------------*- C++ -*-===//
//
// fp32 accumulator add/sub capture kernel (GENERATED by tools/gen_acc_addsub_f32.py).
// Runs every form in FORMS over each of T tiles of one chunk; the chunk layout is
// A[T*16] fp32 followed by B[T*16] fp32.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define T {T}
#define NFORMS {NF}

extern "C" {{

void acc_addsub(float *restrict in, float *restrict out) {{
  float *inA = in;
  float *inB = in + T * 16;
  event0();
  for (int n = 0; n < T; n++) {{
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(inA + n * 16));
    aie::accum<accfloat, 16> b;
    b.from_vector(aie::load_v<16>(inB + n * 16));
    float *o = out + n * NFORMS * 16;
{body}  }}
  event1();
}}

}} // extern "C"
"""


def pack_input(tiles):
    """Per chunk: A[T*16] then B[T*16], as raw 32-bit words."""
    words = []
    for ch in range(len(tiles) // T):
        chunk = tiles[ch * T:(ch + 1) * T]
        words += [w for t in chunk for w in t[0]]
        words += [w for t in chunk for w in t[1]]
    return words


def expected_words():
    """The silicon golden's output words, or None before the first capture."""
    golden = render.load_golden(GOLDEN)
    return None if golden is None else golden["silicon"]


def kernel_data():
    """Raw input and expected buffers (see vv_data.KernelData) for staging."""
    tiles = design()
    golden = render.load_golden(GOLDEN)
    if golden is not None:
        a = [w for t in tiles for w in t[0]]
        b = [w for t in tiles for w in t[1]]
        if golden["inputs"] != [a, b]:
            raise SystemExit("golden inputs are stale: the design changed; recapture on silicon")
    expected = expected_words()
    assert expected is None or len(expected) == len(tiles) * NF * LANES
    return render.kernel_data(pack_input(tiles), expected)


def generate(out_dir=OUT_DIR):
    tiles = design()
    n_out = len(tiles) * NF * LANES
    kernel_data()   # fails on a stale golden before anything is written
    chunks = len(tiles) // T
    nin = T * LANES * 2
    nout = T * NF * LANES
    files = {
        "acc_addsub.cc": render_kernel(),
        "aie.mlir": render.render_mlir(
            KERNEL, "acc_addsub", chunks, nin, nout, "f32", "f32",
            f"{NF} forms x 16 lanes per tile; chunk = A[{T * LANES}] then B[{T * LANES}] fp32."),
        "run.lit": render.render_run_lit("acc_addsub.cc", n_out),
    }
    render.write_files(out_dir, files)
    return files


def capture(out_txt, date, golden_path=GOLDEN):
    tiles = design()
    silicon = render.read_out_txt(out_txt)
    n_out = len(tiles) * NF * LANES
    assert len(silicon) == n_out, f"{len(silicon)} != {n_out}"
    assert all(0 <= v < 1 << 32 for v in silicon)
    rec = {
        "kernel": KERNEL, "class": "acc_addsub_f32", "n": n_out,
        "tile": {"lanes": LANES, "tiles": len(tiles)},
        "forms": [f.as_json() for f in FORMS],
        "inputs": [[w for t in tiles for w in t[0]], [w for t in tiles for w in t[1]]],
        "silicon": silicon,
        "provenance": f"HW-observed: NPU1 Phoenix silicon, captured {date} via tools/gen_acc_addsub_f32.py",
    }
    return render.write_golden(golden_path, rec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    cap = sub.add_parser("capture")
    cap.add_argument("out_txt")
    cap.add_argument("--date", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        files = generate()
        print(f"wrote {len(files)} files to {OUT_DIR} ({len(design())} tiles, {NF} forms)")
    else:
        print("wrote", capture(args.out_txt, args.date))


if __name__ == "__main__":
    main()
