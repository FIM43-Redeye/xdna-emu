#!/usr/bin/env python3
"""Generate the bf16 matrix-multiply "forms" capture kernel (vec_mac_bf16_forms).

Why this exists
---------------
`vec_mac_bf16{,_ovf}` pinned only the fresh multiply (VMUL.f) and left its fp32
accumulation unexplained. This kernel runs every bf16 multiply-accumulate form
the toolchain can emit over one designed batch of tiles, so a single silicon
capture pins the whole float datapath: alignment of the products to a common
exponent, per-term rounding, the accumulator inputs (acc1, acc2) joining the same
sum, sign controls, flush-to-zero, and the NaN/Inf encoding.

One tile is (A[32], B[32], C0[16], C1[16]): bf16 A and B, fp32 accumulators C0
(acc1) and C1 (acc2). Every tile is run through every form in FORMS; form f of
tile t writes 16 fp32 words at out[(t * len(FORMS) + f) * 16 ...]. Two lane
geometries are read off the same A/B:

  mm  the native 4x8x4 mmul: C[i][j] = sum_k A[i*8+k] * B[k*4+j]  (K = 8)
  el  the 16-lane elementwise form: lane l = A[l]*B[l] + A[l+16]*B[l+16]  (K = 2)

The inputs are DESIGNED (sections below), seeded and reproducible; the expected
output is whatever NPU1 silicon produces (`capture` writes the golden JSON the
emulator tests read). No value is computed by a model.

Usage
-----
  python3 tools/gen_mac_bf16_forms.py generate          # run.lit, aie.mlir, forms.cc
  python3 tools/stage_vector_kernel.py vec_mac_bf16_forms   # no exp.bin yet
  # run the bridge on HW once (an uncaptured kernel fails by design; out.txt is
  # written first), then:
  python3 tools/gen_mac_bf16_forms.py capture <build>/out.txt --date YYYY-MM-DD
  python3 tools/stage_vector_kernel.py vec_mac_bf16_forms   # now stages exp.bin = silicon
"""

import argparse
import json
import os
import random
from dataclasses import dataclass

import vv_data

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = "vec_mac_bf16_forms"
OUT_DIR = os.path.join(REPO, "tests", "vector-verify", KERNEL)
GOLDEN = os.path.join(REPO, "tools", "golden", "silicon", KERNEL + ".json")

T = 16            # tiles per kernel iteration (one objectfifo buffer)
LANES = 16        # fp32 outputs per tile per form
MAX_CHUNKS = 64   # shim DMA repeat-count limit


@dataclass(frozen=True)
class Form:
    """One bf16 multiply-accumulate instruction form.

    `instr` is the instruction family (what the opcode alone implies); the
    remaining fields are the config-word controls (aiev2_compute_control:
    sub_mul, sub_acc1, sub_acc2, zero_acc1) the intrinsic sets.
    """
    name: str
    expr: str
    geometry: str                 # "mm" (4x8x4) or "el" (16 x 1x2)
    instr: str                    # mul, negmul, mac, msc, addmac, addmsc
    sub_mul: int = 0
    sub_acc1: int = 0
    sub_acc2: int = 0
    zero_acc1: int = 0

    def as_json(self):
        return {k: v for k, v in self.__dict__.items() if k != "expr"}


def _form(name, expr, geometry, instr, **conf):
    return Form(name, expr, geometry, instr, **conf)


FORMS = [
    _form("mul", "::mul_4x8_8x4(a, b)", "mm", "mul"),
    _form("negmul", "::negmul_4x8_8x4(a, b)", "mm", "negmul"),
    _form("mac", "::mac_4x8_8x4(a, b, c0)", "mm", "mac"),
    _form("msc", "::msc_4x8_8x4(a, b, c0)", "mm", "msc"),
    _form("mac_sa1", "::mac_4x8_8x4_conf(a, b, c0, 0, 0, 1)", "mm", "mac", sub_acc1=1),
    _form("mac_sm", "::mac_4x8_8x4_conf(a, b, c0, 0, 1, 0)", "mm", "mac", sub_mul=1),
    _form("mac_sm_sa1", "::mac_4x8_8x4_conf(a, b, c0, 0, 1, 1)", "mm", "mac", sub_mul=1, sub_acc1=1),
    _form("msc_sm", "::msc_4x8_8x4_conf(a, b, c0, 0, 1, 0)", "mm", "msc", sub_mul=1),
    _form("msc_sa1", "::msc_4x8_8x4_conf(a, b, c0, 0, 0, 1)", "mm", "msc", sub_acc1=1),
    _form("msc_sm_sa1", "::msc_4x8_8x4_conf(a, b, c0, 0, 1, 1)", "mm", "msc", sub_mul=1, sub_acc1=1),
    _form("mul_sm", "::mul_4x8_8x4_conf(a, b, 1)", "mm", "mul", sub_mul=1),
    _form("negmul_sm", "::negmul_4x8_8x4_conf(a, b, 1)", "mm", "negmul", sub_mul=1),
    _form("mac_z1", "::mac_4x8_8x4_conf(a, b, c0, 1, 0, 0)", "mm", "mac", zero_acc1=1),
    _form("addmac", "::addmac_4x8_8x4(a, b, c0, c1)", "mm", "addmac"),
    _form("addmsc", "::addmsc_4x8_8x4(a, b, c0, c1)", "mm", "addmsc"),
    _form("addmac_sa2", "::addmac_4x8_8x4_conf(a, b, c0, c1, 0, 0, 0, 1)", "mm", "addmac", sub_acc2=1),
    _form("addmac_sa1", "::addmac_4x8_8x4_conf(a, b, c0, c1, 0, 0, 1, 0)", "mm", "addmac", sub_acc1=1),
    _form("addmac_sm", "::addmac_4x8_8x4_conf(a, b, c0, c1, 0, 1, 0, 0)", "mm", "addmac", sub_mul=1),
    _form("addmsc_sa2", "::addmsc_4x8_8x4_conf(a, b, c0, c1, 0, 0, 0, 1)", "mm", "addmsc", sub_acc2=1),
    _form("e_mul", "::mul_elem_16_2(a, b)", "el", "mul"),
    _form("e_negmul", "::negmul_elem_16_2(a, b)", "el", "negmul"),
    _form("e_mac", "::mac_elem_16_2(a, b, c0)", "el", "mac"),
    _form("e_msc", "::msc_elem_16_2(a, b, c0)", "el", "msc"),
]
NF = len(FORMS)


# --- Designed inputs -----------------------------------------------------------

def bf(sign, exp, mant):
    return (sign << 15) | (exp << 7) | mant


def f32(sign, exp, mant):
    return (sign << 31) | (exp << 23) | mant


ONE = bf(0, 127, 0)
NAN_PAYLOADS = (0x7FC0, 0x7F81, 0xFFC1, 0x7FA5, 0x7FFF, 0xFF90)


def prod(exp, ma=0, mb=0, sign=0):
    """(a, b) bf16 pair whose exact product is (1.ma)(1.mb) * 2^exp, signed.

    The exponent is split between the factors so both stay inside the bf16
    normal range (exp in -252..254).
    """
    ea = exp // 2
    return bf(sign, 127 + ea, ma), bf(0, 127 + exp - ea, mb)


ZERO_PAIR = (0, 0)


def uniform_tile(terms, c0=0, c1=0):
    """A tile whose 16 mm lanes all compute the same 8-term sum."""
    a, b = [0] * 32, [0] * 32
    for i in range(4):
        for k in range(8):
            a[i * 8 + k] = terms[k][0]
    for k in range(8):
        for j in range(4):
            b[k * 4 + j] = terms[k][1]
    return a, b, [c0] * 16, [c1] * 16


def el_tile(lanes):
    """A tile of 16 independent elementwise lanes.

    Each lane is (p0, p1, c0, c1) with p0/p1 (a, b) pairs.
    """
    a, b, c0, c1 = [0] * 32, [0] * 32, [0] * 16, [0] * 16
    for l, (p0, p1, x0, x1) in enumerate(lanes):
        a[l], b[l] = p0
        a[l + 16], b[l + 16] = p1
        c0[l], c1[l] = x0, x1
    return a, b, c0, c1


def section_gap():
    """Gap sweep: one large product and one small product, exponent gap 0..47.

    Large product sig is 1.0 (leading bit at E) or ~3.97 (leading bit at E+1);
    the small product's mantissa pattern is a power of two, all ones, a
    half-way tie pattern or a sparse pattern; small has the same or opposite sign
    and sits in either slot.
    """
    tiles = []
    bigs = ((0, 0), (0x7F, 0x7F))
    smalls = ((0, 0), (0x7F, 0x7F), (0x40, 0), (0x55, 0x2A))
    for bi, (bma, bmb) in enumerate(bigs):
        for si, (sma, smb) in enumerate(smalls):
            for sign in (0, 1):
                for part in range(3):
                    lanes = []
                    for g in range(part * 16, part * 16 + 16):
                        big = prod(0, bma, bmb)
                        small = prod(-g, sma, smb, sign)
                        if (bi + si + part) % 2:
                            big, small = small, big
                        lanes.append((big, small, 0, 0))
                    tiles.append(el_tile(lanes))
    return tiles


def section_position(rng):
    """Same multiset of 8 products in many K positions (summation order)."""
    multisets = [
        # cancelling large pair, small survivors
        [prod(20, 64), prod(20, 64, 0, 1), prod(0, 5, 9), prod(-3, 100),
         prod(-12, 0x7F, 0x7F), prod(-22, 0, 0, 1), prod(-5, 33, 77), prod(-30, 1, 1)],
        # all the same exponent, mixed signs
        [prod(0, rng.randrange(128), rng.randrange(128), rng.randrange(2)) for _ in range(8)],
        # geometric ladder, every term rounds on the grid
        [prod(-3 * k, rng.randrange(128), rng.randrange(128), rng.randrange(2)) for k in range(8)],
        # one large, seven near the rounding threshold
        [prod(0, 0x3C, 0x11)] + [prod(-(20 + k), rng.randrange(128), rng.randrange(128), rng.randrange(2))
                                 for k in range(7)],
    ]
    tiles = []
    for ms in multisets:
        for _ in range(12):
            terms = ms[:]
            rng.shuffle(terms)
            tiles.append(uniform_tile(terms, f32(0, 120, 0x123456), f32(1, 118, 0x654321)))
    return tiles


def section_accumulator(rng):
    """Accumulators at controlled exponent offsets from the products."""
    tiles = []
    mants = (0, 1, 0x7FFFFF, 0x400000, 0x3FFFFF)
    for t in range(16):
        lanes = []
        for l in range(16):
            d = -34 + 4 * l + (t % 4)        # acc exponent relative to the product exponent
            e0 = 127 + d
            m0 = mants[(t + l) % len(mants)] if t < 10 else rng.randrange(1 << 23)
            m1 = rng.randrange(1 << 23)
            p0 = prod(0, rng.randrange(128), rng.randrange(128), rng.randrange(2))
            p1 = prod(-rng.randrange(0, 6), rng.randrange(128), rng.randrange(128), rng.randrange(2))
            c0 = f32(rng.randrange(2), min(254, max(1, e0)), m0)
            c1 = f32(rng.randrange(2), min(254, max(1, e0 + rng.randrange(-8, 9))), m1)
            lanes.append((p0, p1, c0, c1))
        tiles.append(el_tile(lanes))
    return tiles


def section_edges(rng):
    """Zeros, denormals, flush-to-zero boundary, overflow, carry, signed zeros."""
    Z = ZERO_PAIR
    cases = []
    # zero products must not set the reference exponent
    for ebz in (254, 160):
        cases.append(([prod(-250, 5, 3), (0, bf(0, ebz, 0))] + [Z] * 6, 0, 0))
    # denormal bf16 inputs
    for m in (1, 2, 0x40, 0x7F):
        for eb in (127, 200, 254, 1):
            cases.append(([(bf(0, 0, m), bf(0, eb, 0))] + [Z] * 7, 0, 0))
            cases.append(([(bf(1, 0, m), bf(0, eb, 5)), prod(0, 3)] + [Z] * 6, 0, 0))
    cases.append(([(bf(0, 0, 0x40), bf(0, 0, 0x40))] + [Z] * 7, 0, 0))
    # results at and below the smallest normal: flush happens before rounding
    cases.append(([prod(-127), prod(-127), prod(-150, 0, 0, 1)] + [Z] * 5, 0, 0))
    cases.append(([prod(-127), prod(-127), prod(-150, 64, 0, 1)] + [Z] * 5, 0, 0))
    cases.append(([prod(-127), prod(-127), prod(-149, 0, 0, 1)] + [Z] * 5, 0, 0))
    cases.append(([prod(-127), prod(-127)] + [Z] * 6, 0, 0))
    cases.append(([prod(-127, 0, 0, 1), prod(-127, 0, 0, 1), prod(-150)] + [Z] * 5, 0, 0))
    cases.append(([prod(-127)] + [Z] * 7, 0, 0))
    cases.append(([prod(-127, 0, 0, 1)] + [Z] * 7, 0, 0))
    cases.append(([prod(-127, 0x7F, 0x7F)] + [Z] * 7, 0, 0))
    cases.append(([prod(-130, 0, 0, 1), prod(-140)] + [Z] * 6, 0, 0))
    # denormal accumulator, with and without products around it
    cases.append(([prod(-140, 7, 9)] + [Z] * 7, f32(0, 0, 0x1234), 0))
    cases.append(([prod(-130, 7, 9, 1)] + [Z] * 7, f32(1, 0, 0x7FFFFF), f32(0, 0, 5)))
    cases.append(([Z] * 8, f32(1, 0, 0x400000), 0))
    cases.append(([Z] * 8, f32(1, 0, 0), f32(1, 0, 0)))
    # signed zeros and exact cancellation
    cases.append(([prod(0), prod(0, 0, 0, 1)] + [Z] * 6, 0, 0))
    cases.append(([(bf(1, 0, 0), ONE)] * 8, 0, 0))
    cases.append(([(bf(1, 0, 0), ONE), (0, ONE)] + [Z] * 6, f32(1, 0, 0), 0))
    # overflow and carry
    for ea, eb in ((254, 254), (254, 127), (254, 128), (254, 126), (200, 182), (250, 131)):
        for m in (0, 0x7F):
            cases.append(([(bf(0, ea, m), bf(0, eb, m))] + [Z] * 7, 0, 0))
    for m in (0, 0x7F):
        cases.append(([(bf(0, 200, m), bf(0, 127, m))] * 8, 0, 0))
        cases.append(([(bf(0, 254, m), bf(0, 127, m))] * 8, 0, 0))
        cases.append(([(bf(0, 254, m), bf(0, 127, m))] * 4 + [(bf(1, 254, m), bf(0, 127, m))] * 4, 0, 0))
        cases.append(([(bf(0, 254, m), bf(0, 127, m))] * 2 + [(bf(1, 254, m), bf(0, 127, m))] + [Z] * 5, 0, 0))
    cases.append(([prod(100), prod(101)] + [Z] * 6, f32(0, 254, 0x7FFFFF), 0))
    cases.append(([prod(100)] + [Z] * 7, f32(1, 254, 0x7FFFFF), f32(0, 254, 0x7FFFFF)))
    return [uniform_tile(t, c0, c1) for t, c0, c1 in cases]


def _special_bf(rng):
    r = rng.random()
    if r < 0.18:
        return rng.choice((0x7F80, 0xFF80))
    if r < 0.40:
        return rng.choice(NAN_PAYLOADS)
    if r < 0.55:
        return rng.choice((0x0000, 0x8000))
    if r < 0.60:
        return rng.choice((0x3F80, 0xBF80))
    return bf(rng.randrange(2), rng.randrange(100, 160), rng.randrange(128))


def _special_f32(rng):
    r = rng.random()
    if r < 0.2:
        return f32(rng.randrange(2), 255, 0)
    if r < 0.4:
        return f32(rng.randrange(2), 255, rng.randrange(1, 1 << 23))
    if r < 0.5:
        return f32(rng.randrange(2), 0, rng.choice((0, 5)))
    return f32(rng.randrange(2), rng.randrange(100, 160), rng.randrange(1 << 23))


def section_specials(rng):
    """Inf, NaN (several payloads), invalid 0*Inf, in every operand position."""
    tiles = []
    # elementwise: the whole product-pair / accumulator space, sampled
    for _ in range(40):
        lanes = []
        for _l in range(16):
            lanes.append(((_special_bf(rng), _special_bf(rng)), (_special_bf(rng), _special_bf(rng)),
                          _special_f32(rng), _special_f32(rng)))
        tiles.append(el_tile(lanes))
    # exhaustive small alphabet
    alpha = (ONE, bf(1, 127, 0), 0x0000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0xFFC1)
    pairs = [(a, b) for a in alpha for b in alpha]
    lanes = []
    for p0 in pairs:
        for p1 in ((ONE, ONE), (0x7F80, 0x0000), (0xFF80, ONE), (0xFFC1, ONE)):
            lanes.append((p0, p1, f32(0, 127, 0), 0))
    for k in range(0, len(lanes), 16):
        tiles.append(el_tile(lanes[k:k + 16]))
    # matmul view: sparse specials among finite terms
    for _ in range(16):
        a = [bf(rng.randrange(2), rng.randrange(110, 145), rng.randrange(128)) for _ in range(32)]
        b = [bf(rng.randrange(2), rng.randrange(110, 145), rng.randrange(128)) for _ in range(32)]
        for _n in range(rng.randrange(1, 4)):
            a[rng.randrange(32)] = _special_bf(rng)
            b[rng.randrange(32)] = _special_bf(rng)
        tiles.append((a, b, [_special_f32(rng) if rng.random() < 0.15 else f32(0, 127, rng.randrange(1 << 23))
                             for _ in range(16)],
                      [f32(rng.randrange(2), 126, rng.randrange(1 << 23)) for _ in range(16)]))
    return tiles


def section_random(rng):
    """Random K=8 / K=2 tiles with exponent spreads from 2 to 60."""
    tiles = []
    for t in range(64):
        spread = (2, 6, 12, 20, 30, 40, 50, 60)[t % 8]
        base = rng.randrange(70 + spread // 2, 185 - spread // 2)
        zero_p = (0, 0, 0.1, 0.3)[(t // 8) % 4]

        def rb(lo, hi, zp=0.0):
            if rng.random() < zp:
                return bf(rng.randrange(2), 0, rng.choice((0, 0, rng.randrange(128))))
            return bf(rng.randrange(2), rng.randrange(lo, hi + 1), rng.randrange(128))

        a = [rb(base - spread // 2, base + spread // 2, zero_p) for _ in range(32)]
        b = [rb(127 - spread // 4, 127 + spread // 4) for _ in range(32)]
        ab = base + rng.randrange(-30, 30)
        c0 = [f32(rng.randrange(2), min(254, max(1, rng.randrange(ab - spread // 2, ab + spread // 2 + 1))),
                  rng.randrange(1 << 23)) for _ in range(16)]
        c1 = [f32(rng.randrange(2), min(254, max(1, rng.randrange(ab - spread // 2 - 5, ab + spread // 2 + 6))),
                  rng.randrange(1 << 23)) for _ in range(16)]
        tiles.append((a, b, c0, c1))
    return tiles


def design():
    """The designed tile batch, padded with zero tiles to a whole number of chunks."""
    rng = random.Random(0xD6B1)
    tiles = []
    tiles += section_gap()
    tiles += section_position(rng)
    tiles += section_accumulator(rng)
    tiles += section_edges(rng)
    tiles += section_specials(rng)
    tiles += section_random(rng)
    while len(tiles) % T:
        tiles.append(([0] * 32, [0] * 32, [0] * 16, [0] * 16))
    assert len(tiles) // T <= MAX_CHUNKS, len(tiles)
    for a, b, c0, c1 in tiles:
        assert (len(a), len(b), len(c0), len(c1)) == (32, 32, 16, 16)
        assert all(0 <= v < 1 << 16 for v in a + b) and all(0 <= v < 1 << 32 for v in c0 + c1)
    return tiles


def flat_inputs(tiles):
    return ([x for t in tiles for x in t[0]], [x for t in tiles for x in t[1]],
            [x for t in tiles for x in t[2]], [x for t in tiles for x in t[3]])


# --- Rendering -----------------------------------------------------------------

def render_kernel(n_tiles):
    body = "".join(
        "    { aie::accum<accfloat, 16> r = %s; aie::store_v(o + %d * 16, r.to_vector<float>()); }\n"
        % (f.expr, i) for i, f in enumerate(FORMS))
    return f"""//===- forms.cc -------------------------------------------------*- C++ -*-===//
//
// bf16 MAC forms capture kernel (GENERATED by tools/gen_mac_bf16_forms.py).
// Runs every form in FORMS over each of T tiles of one chunk; the layout of the
// chunk is A[T*32] bf16, B[T*32] bf16, C0[T*16] fp32, C1[T*16] fp32.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define T {T}
#define NFORMS {NF}

extern "C" {{

void mac_forms(bfloat16 *restrict in, float *restrict out) {{
  bfloat16 *inA = in;
  bfloat16 *inB = in + T * 32;
  float *inC0 = (float *)(in + 2 * T * 32);
  float *inC1 = inC0 + T * 16;
  event0();
  for (int n = 0; n < T; n++) {{
    aie::vector<bfloat16, 32> a = aie::load_v<32>(inA + n * 32);
    aie::vector<bfloat16, 32> b = aie::load_v<32>(inB + n * 32);
    aie::accum<accfloat, 16> c0;
    c0.from_vector(aie::load_v<16>(inC0 + n * 16));
    aie::accum<accfloat, 16> c1;
    c1.from_vector(aie::load_v<16>(inC1 + n * 16));
    float *o = out + n * NFORMS * 16;
{body}  }}
  event1();
}}

}} // extern "C"
"""


def render_mlir(n_tiles):
    chunks = n_tiles // T
    nin = T * (32 + 32 + 32 + 32) // 2 * 1   # bf16 elements per chunk: A + B + C0 + C1 (fp32 = 2 bf16)
    nin = T * 32 * 2 + T * 16 * 2 * 2
    nout = T * NF * LANES
    return f"""//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// vec_mac_bf16_forms capture design (GENERATED). One compute tile (0,2) consumes
// {chunks} chunks of {T} tiles (A, B, C0, C1 packed into one bf16-typed buffer of
// {nin} elements) and produces {NF} forms x 16 lanes per tile.
//
//===----------------------------------------------------------------------===//

module {{
  aie.device(npu1_1col) {{
    func.func private @mac_forms(memref<{nin}xbf16>, memref<{nout}xf32>)
        attributes {{link_with = "forms.o"}}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {{%core}}, 1 : i32) : !aie.objectfifo<memref<{nin}xbf16>>
    aie.objectfifo @outC(%core, {{%shim}}, 1 : i32) : !aie.objectfifo<memref<{nout}xf32>>

    %core_0_2 = aie.core(%core) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant {chunks} : index
      scf.for %i = %c0 to %cn step %c1 {{
        %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<{nin}xbf16>
        %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<{nout}xf32>
        func.call @mac_forms(%a, %o) : (memref<{nin}xbf16>, memref<{nout}xf32>) -> ()
        aie.objectfifo.release @inA(Consume, 1)
        aie.objectfifo.release @outC(Produce, 1)
      }}
      aie.end
    }}

    aie.runtime_sequence @sequence(%a: memref<{chunks * nin}xbf16>, %o: memref<{chunks * nout}xf32>) {{
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][{chunks}, 1, 1, {nin}][{nin}, 0, 0, 1]) {{id = 0 : i64, metadata = @inA}} : memref<{chunks * nin}xbf16>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][{chunks}, 1, 1, {nout}][{nout}, 0, 0, 1]) {{id = 1 : i64, metadata = @outC}} : memref<{chunks * nout}xf32>
      aiex.npu.dma_wait {{symbol = @outC}}
    }}
  }}
}}
"""


def render_run_lit(n_out):
    """Build/run recipe; the host is the shared vv_host.cpp (1 input, `n_out` u32 out)."""
    return f"""// (c) Copyright 2026 -- xdna-emu vector-compute verification.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: ryzen_ai_npu1, chess
//
// RUN: xchesscc_wrapper aie2 -I %aietools/include -c %S/forms.cc -o ./forms.o
// RUN: %aiecc --xchesscc --xbridge --get-xclbin --get-npu-insts --xclbin-name=aie.xclbin --npu-insts-name=insts.bin %S/aie.mlir
// RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags {vv_data.host_flags(1, "uint32_t", n_out)}
// RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
"""


def load_golden():
    if not os.path.exists(GOLDEN):
        return None
    with open(GOLDEN) as f:
        return json.load(f)


def pack_input(tiles):
    """The input buffer as uint16 elements, chunk by chunk.

    Per chunk: A[T*32] u16, B[T*32] u16, C0[T*16] u32, C1[T*16] u32, where each
    fp32 accumulator word is staged as its (low, high) u16 halves.
    """
    a, b, c0, c1 = flat_inputs(tiles)
    packed = []
    for ch in range(len(tiles) // T):
        s = ch * T
        packed += a[s * 32:(s + T) * 32]
        packed += b[s * 32:(s + T) * 32]
        for arr in (c0, c1):
            for w in arr[s * 16:(s + T) * 16]:
                packed += [w & 0xFFFF, w >> 16]
    return packed


def expected_words():
    """The silicon golden's output words, or None before the first capture."""
    golden = load_golden()
    return None if golden is None else golden["silicon"]


def kernel_data():
    """Raw input and expected buffers (see vv_data.KernelData) for staging."""
    tiles = design()
    golden = load_golden()
    if golden is not None:
        a, b, c0, c1 = flat_inputs(tiles)
        if golden["inputs"] != [a, b, c0, c1]:
            raise SystemExit("golden inputs are stale: the design changed; recapture on silicon")
    expected = expected_words()
    assert expected is None or len(expected) == len(tiles) * NF * LANES
    return vv_data.KernelData(
        inputs=[vv_data.pack(pack_input(tiles), "uint16_t")],
        expected=None if expected is None else vv_data.pack(expected, "uint32_t"),
    )


def generate(out_dir=OUT_DIR):
    tiles = design()
    n_out = len(tiles) * NF * LANES
    kernel_data()   # fails on a stale golden before anything is written
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "forms.cc": render_kernel(len(tiles)),
        "aie.mlir": render_mlir(len(tiles)),
        "run.lit": render_run_lit(n_out),
    }
    for name, text in files.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text)
    return files


def capture(out_txt, date, golden_path=GOLDEN):
    tiles = design()
    with open(out_txt) as f:
        silicon = [int(line) for line in f if line.strip()]
    n_out = len(tiles) * NF * LANES
    assert len(silicon) == n_out, f"{len(silicon)} != {n_out}"
    assert all(0 <= v < 1 << 32 for v in silicon)
    a, b, c0, c1 = flat_inputs(tiles)
    rec = {
        "kernel": KERNEL, "class": "matmul_bf16_forms", "n": n_out,
        "tile": {"a": 32, "b": 32, "c0": 16, "c1": 16, "lanes": LANES, "tiles": len(tiles)},
        "forms": [f.as_json() for f in FORMS],
        "inputs": [a, b, c0, c1], "silicon": silicon,
        "provenance": f"HW-observed: NPU1 Phoenix silicon, captured {date} via tools/gen_mac_bf16_forms.py",
    }
    os.makedirs(os.path.dirname(golden_path), exist_ok=True)
    with open(golden_path, "w") as f:
        f.write("{\n")
        items = list(rec.items())
        for i, (k, v) in enumerate(items):
            f.write(json.dumps(k) + ": " + json.dumps(v, separators=(",", ":")))
            f.write(",\n" if i + 1 < len(items) else "\n")
        f.write("}\n")
    return golden_path


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
