#!/usr/bin/env python3
"""Generate the MAC configuration-word probe suites (Peano C++ kernels with a runtime conf word).

The MAC instructions (vmul, vmac, vmsc, vnegmul, ..., vsubmsc) take a 32-bit configuration
register (llvm-aie `VecConf`, aiev2_compute_control):

  b0 zero_acc  b2:1 amode  b4:3 bmode  b7:5 cmode(variant)  b8 sgn_y  b9 sgn_x  b10 shift16
  b11 sub_mul  b12 sub_acc1  b13 sub_acc2  b15:14 reserved  b23:16 term_neg  b31:24 reserved

Every kernel here loops over TILES, each carrying its own operands and conf word, so a case
sweeps one instruction over many conf words with one compile. A tile is 448 bytes of input

    A (64) | B (64) | acc1 (128) | acc2 (128) | conf (4) | pad (60)

and 128 bytes of output (the whole 1024-bit result accumulator, raw). Generators only write
suites; compile and run with `python3 -m harness suite build|hw|emu|compare <dir> ...`.

Plans (each a suite of its own, `--suite-dir`):
  flags   every instruction kind x {zero, shift16, sub_mul, sub_acc1, sub_acc2, sgn_x, sgn_y}
          on a fixed geometry (random operands)
  codes   every (amode, bmode, cmode) code, random operands, a few sign settings
  upper   single reserved / term_neg bits on top of a base conf, per instruction kind
  random  seeded fully random conf words per kind (what the Peano `*_conf` cases feed)

  python3 tools/gen_mac_conf_probe.py flags|codes|upper|random --suite-dir <dir> [--seed N]
"""
from __future__ import annotations

import argparse
import itertools
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cxx_suite import CxxKernel, write_cxx_suite  # noqa: E402

TILE_IN = 448
TILE_OUT = 128
TILES_PER_CASE = 48

# kind -> (builtin, takes acc1, takes acc2)
KINDS = {
    "mul": ("I512_I512_ACC1024_acc32_mul_conf", False, False),
    "negmul": ("I512_I512_ACC1024_acc32_negmul_conf", False, False),
    "mac": ("I512_I512_ACC1024_acc32_mac_conf", True, False),
    "msc": ("I512_I512_ACC1024_acc32_msc_conf", True, False),
    "negmac": ("I512_I512_ACC1024_acc32_negmac_conf", True, False),
    "negmsc": ("I512_I512_ACC1024_acc32_negmsc_conf", True, False),
    "addmac": ("I512_I512_ACC1024_ACC1024_acc32_addmac_conf", True, True),
    "addmsc": ("I512_I512_ACC1024_ACC1024_acc32_addmsc_conf", True, True),
    "submac": ("I512_I512_ACC1024_ACC1024_acc32_submac_conf", True, True),
    "submsc": ("I512_I512_ACC1024_ACC1024_acc32_submsc_conf", True, True),
}


def conf_word(zero=0, amode=0, bmode=0, cmode=0, sgn_y=1, sgn_x=1, shift16=0, sub_mul=0,
              sub_acc1=0, sub_acc2=0, reserved1=0, term_neg=0, reserved2=0) -> int:
    return (zero | amode << 1 | bmode << 3 | cmode << 5 | sgn_y << 8 | sgn_x << 9 | shift16 << 10
            | sub_mul << 11 | sub_acc1 << 12 | sub_acc2 << 13 | reserved1 << 14
            | term_neg << 16 | reserved2 << 24)


def kernel_source(kind: str, tiles: int) -> str:
    builtin, acc1, acc2 = KINDS[kind]
    args = "a, b" + (", c1" if acc1 else "") + (", c2" if acc2 else "") + ", conf"
    return f"""// Auto-generated (gen_mac_conf_probe.py): {kind} over {tiles} tiles with a per-tile conf word
#define NOCPP
#include <stdint.h>

extern "C" {{
void test_kernel(const int32_t *restrict in, int32_t *restrict out) {{
    for (int t = 0; t < {tiles}; t++) {{
        const int32_t *p = in + t * {TILE_IN // 4};
        v64int8 a = *(const v64int8 *)(p);
        v16int32 b = *(const v16int32 *)(p + 16);
        v16acc64 c1 = *(const v16acc64 *)(p + 32);
        v16acc64 c2 = *(const v16acc64 *)(p + 64);
        int32_t conf = p[96];
        v16acc64 r = __builtin_aiev2_{builtin}({args});
        *(v16acc64 *)(out + t * {TILE_OUT // 4}) = r;
    }}
}}
}} // extern "C"
"""


def tile_bytes(rng: random.Random, conf: int, *, a=None, b=None, c1=None, c2=None) -> bytes:
    rb = lambda n: bytes(rng.getrandbits(8) for _ in range(n))  # noqa: E731
    out = (a or rb(64)) + (b or rb(64)) + (c1 or rb(128)) + (c2 or rb(128))
    out += struct.pack("<I", conf & 0xFFFFFFFF)
    return out.ljust(TILE_IN, b"\0")


def cases_for(kind: str, plan: str, confs: list[int], rng: random.Random,
              per_case: int = TILES_PER_CASE, operands=None) -> list[CxxKernel]:
    """`operands`, if given, is a function (tile index) -> dict(a=, b=, c1=, c2=) of raw operand bytes."""
    kernels = []
    for k, start in enumerate(range(0, len(confs), per_case)):
        chunk = confs[start:start + per_case]
        data = b"".join(tile_bytes(rng, c, **(operands(start + i) if operands else {}))
                        for i, c in enumerate(chunk))
        kernels.append(CxxKernel(
            name=f"{plan}__{kind}__{k:02d}", source=kernel_source(kind, len(chunk)),
            in_size=len(data), out_size=TILE_OUT * len(chunk), input_bytes=data,
            meta={"plan": plan, "kind": kind, "confs": chunk, "tile_in": TILE_IN, "tile_out": TILE_OUT}))
    return kernels


# -- plans ---------------------------------------------------------------------------------

# Two reference geometries with a known lane map: int8 x int8 -> acc32 (4x8x8) and
# int16 x int16 -> acc64 (4x4x4).
GEOMETRIES = {"i8": dict(amode=0, bmode=1, cmode=0), "i16": dict(amode=1, bmode=3, cmode=1)}


def plan_flags(rng):
    out = []
    for kind in KINDS:
        for gname, g in GEOMETRIES.items():
            confs = [conf_word(zero=z, shift16=sh, sub_mul=m, sub_acc1=a1, sub_acc2=a2,
                               sgn_x=sx, sgn_y=sy, **g)
                     for z, sh, m, a1, a2, sx, sy in itertools.product((0, 1), repeat=7)]
            out += [k for k in cases_for(kind, f"flags_{gname}", confs, rng)]
    return out


def plan_codes(rng):
    out = []
    for kind in ("mul", "mac"):
        confs = []
        for code in range(128):
            for sx, sy in ((1, 1), (0, 0), (1, 0)):
                for _ in range(2):
                    confs.append(code << 1 | sy << 8 | sx << 9)
        out += cases_for(kind, "codes", confs, rng)
    return out


def plan_same(rng):
    """Every code on the same four operand sets, so equal outputs identify equal behaviour."""
    sets = [dict(a=bytes(rng.getrandbits(8) for _ in range(64)), b=bytes(rng.getrandbits(8) for _ in range(64)),
                 c1=bytes(rng.getrandbits(8) for _ in range(128)), c2=bytes(rng.getrandbits(8) for _ in range(128)))
            for _ in range(4)]
    confs, which = [], []
    for code in range(128):
        for i, (sx, sy) in enumerate(((1, 1), (0, 0), (1, 0), (0, 1))):
            confs.append(code << 1 | sy << 8 | sx << 9)
            which.append(i)
    return cases_for("mul", "same", confs, rng, operands=lambda i: sets[which[i]])


ONEHOT_CODES = [4, 21, 2, 18, 17, 69]   # two known geometries (calibration) and the four undecoded ones


def plan_onehot(rng, codes=None):
    """Bilinearity probes: one A byte (or one B byte) is 0x01, the other operand is all 0x01 bytes.

    Unsigned, so an output lane is (A value) * (B value) * (number of partner elements): the
    partners of every operand byte fall out for any element width. `codes` are conf>>1 & 127.
    """
    ones = bytes([1] * 64)
    out = []
    for code in codes or ONEHOT_CODES:
        confs, ops = [], []
        for i in range(64):
            confs.append(code << 1)
            ops.append(dict(a=bytes(1 if x == i else 0 for x in range(64)), b=ones))
        for j in range(64):
            confs.append(code << 1)
            ops.append(dict(a=ones, b=bytes(1 if x == j else 0 for x in range(64))))
        out += cases_for("mul", f"onehot{code:03d}", confs, rng, per_case=64, operands=lambda i, o=ops: o[i])
    return out


FPROBE_VALUES = [0x3F80, 0x4000, 0xBF80, 0x3FC0, 0x0001, 0x0080, 0x007F, 0x4040, 0x3F00, 0x4100, 0x7F80, 0x0000]


def plan_fprobe(rng):
    """The float conf (amode 2) on the integer multiply: one nonzero pair (A[0], B[0]) of bf16 values."""
    out = []
    for code in (2, 18):
        confs, ops = [], []
        for x in FPROBE_VALUES:
            for y in FPROBE_VALUES:
                for sx, sy in ((1, 1), (0, 0)):
                    confs.append(code << 1 | sy << 8 | sx << 9)
                    ops.append(dict(a=struct.pack("<H", x).ljust(64, b"\0"), b=struct.pack("<H", y).ljust(64, b"\0")))
        out += cases_for("mul", f"fprobe{code:03d}", confs, rng, operands=lambda i, o=ops: o[i])
    return out


def plan_fsingle(rng, code=18, n=240):
    """Float conf on the integer multiply, one product term per tile: A[0], B[0] random bf16 values."""
    confs, ops = [], []
    for _ in range(n):
        a, b = rng.getrandbits(16), rng.getrandbits(16)
        confs.append(code << 1 | 3 << 8)
        ops.append(dict(a=struct.pack("<H", a).ljust(64, b"\0"), b=struct.pack("<H", b).ljust(64, b"\0")))
    return cases_for("mul", f"fsingle{code:03d}", confs, rng, operands=lambda i: ops[i])


def valid_codes():
    """conf>>1 & 127 of every code that produces a nonzero product on a dense integer multiply:
    the header table's dense integer modes, the amode-1 bmode-1 aliases, the two complex modes and
    the two float-conf modes (amode 2)."""
    codes = []
    for amode, bmode, cmode in [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 0), (0, 2, 1),
                                (0, 3, 0), (0, 3, 1), (1, 0, 0), (1, 0, 1), (1, 2, 0), (1, 2, 1), (1, 3, 0), (1, 3, 1),
                                (1, 3, 2), (1, 3, 3), (1, 3, 4), (2, 0, 0), (2, 0, 1)]:
        codes.append(amode | bmode << 2 | cmode << 4)
    return codes


def plan_termneg(rng):
    """Each term_neg bit alone, on every valid code, all on one shared operand set per code
    (so out(base) - out(bit) is twice the negated part)."""
    out = []
    for code in valid_codes():
        a = bytes(rng.getrandbits(8) for _ in range(64))
        b = bytes(rng.getrandbits(8) for _ in range(64))
        confs = [code << 1 | 3 << 8]
        for bit in range(8):
            confs.append(code << 1 | 3 << 8 | 1 << (16 + bit))
        out += cases_for("mul", f"termneg{code:03d}", confs, rng, operands=lambda i, a=a, b=b: dict(a=a, b=b))
    return out


MODES_CODES = [2, 3, 18, 19, 17, 77, 69, 16, 32, 33, 34, 66, 98, 127, 84]   # float conf, complex, idle (am0/am1/am2/am3)


def plan_modes(rng):
    """The flags of the post-adder on the modes the flags plan does not cover: every kind, random flags."""
    out = []
    for kind in KINDS:
        confs = []
        for code in MODES_CODES:
            for _ in range(24):
                confs.append(code << 1 | rng.getrandbits(1) << 0 | rng.getrandbits(2) << 8 | rng.getrandbits(4) << 10
                             | rng.getrandbits(8) << 16)
        out += cases_for(kind, "modes", confs, rng)
    return out


def plan_repro(rng, suite_dir=None):
    """Replays the operands of one `modes` tile (the single EMU != HW tile) under flag variations."""
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_mac_conf_probe import tile_data
    src = Path(REPRO_SUITE)
    meta, tiles = tile_data(src, "r1", REPRO_CASE)
    t = next(t for t in tiles if t["conf"] == REPRO_CONF)
    ops = dict(a=t["a"], b=t["b"], c1=t["acc1"], c2=t["acc2"])
    out = []
    for kind in ("addmsc", "mul", "msc", "addmac"):
        confs = []
        for tn in (REPRO_CONF >> 16 & 0xFF, 0):
            for flags in range(16):
                confs.append((REPRO_CONF & 0xFF00FFFF & ~(0xF << 10)) | flags << 10 | tn << 16)
        out += cases_for(kind, "repro", confs, rng, operands=lambda i: ops)
    return out


# The `modes` probe run whose single mismatching tile the repro replays (experiment output, gitignored).
REPRO_SUITE = Path(__file__).resolve().parents[1] / "build/experiments/harness/macconf/modes"
REPRO_CASE = "modes__addmsc__00"
REPRO_CONF = 0x622706


# Sparse term_neg probe (Chess intrinsics: conf = base | sgn_y << 8 with sgn_x = 0, so any bit of the
# conf above bit 7 can be set through sgn_y). (intrinsic, A type, A bytes/elem, rows, inner, B sparse type,
# B data byte pattern, mask bits per unit)
SPARSE_GEOMS = {
    "2x8_8x8": ("mul_2x8_8x8", "v32int16", 2, 2, 8, "v64int16_sparse", 0x01, 2),
    "2x16_16x8": ("mul_2x16_16x8", "v32int16", 2, 2, 16, "v128int8_sparse", 0x01, 1),
    "4x16_16x8": ("mul_4x16_16x8", "v64int8", 1, 4, 16, "v128int8_sparse", 0x01, 1),
}
SPARSE_TILE_IN = 192


def sparse_kernel(name, tiles):
    intr, atype, _, _, _, btype, _, _ = SPARSE_GEOMS[name]
    return f"""// Auto-generated (gen_mac_conf_probe.py sparse): {intr}, conf above bit 7 through sgn_y
#define NOCPP
#include <stdint.h>
#include <string.h>

extern "C" {{
void test_kernel(const int32_t *restrict in, int32_t *restrict out) {{
    for (int t = 0; t < {tiles}; t++) {{
        const char *p = (const char *)in + t * {SPARSE_TILE_IN};
        {atype} a;
        memcpy(&a, p, sizeof(a));
        {btype} b;
        memcpy(&b, p + 64, sizeof(b));
        int sgn_y;
        memcpy(&sgn_y, p + 144, sizeof(int));
        auto r = {intr}(a, 0, b, sgn_y);
        memcpy((char *)out + t * 128, &r, 128);
    }}
}}
}} // extern "C"
"""


def plan_sparse(rng):
    out = []
    for name, (intr, atype, abytes, rows, inner, btype, bpat, mbpu) in SPARSE_GEOMS.items():
        nelem = 64 // abytes
        data = bytes([bpat, 0] * 32) if abytes == 2 and name == "2x8_8x8" else bytes([bpat] * 64)
        tiles = []
        for pat in (0, 1):
            if mbpu == 2:
                mask = bytes([0x0F if pat == 0 else 0xF0] * 16)
            else:
                mask = bytes([0x33 if pat == 0 else 0xCC] * 16)
            for i in range(nelem):
                a = bytearray(64)
                a[i * abytes] = 1
                for bit in [None] + list(range(8)):
                    conf_hi = 0 if bit is None else (1 << (16 + bit)) >> 8
                    tiles.append((bytes(a), data + mask, conf_hi))
        per = 48
        for k, start in enumerate(range(0, len(tiles), per)):
            chunk = tiles[start:start + per]
            buf = b"".join((a + bm + struct.pack("<I", ch)).ljust(SPARSE_TILE_IN, b"\0") for a, bm, ch in chunk)
            out.append(CxxKernel(name=f"sparse_{name}__{k:02d}", source=sparse_kernel(name, len(chunk)),
                                 in_size=len(buf), out_size=128 * len(chunk), input_bytes=buf,
                                 meta={"plan": "sparse", "geom": name, "tiles": [[i, ch] for _, _, ch in chunk]}))
    return out


ACC_KINDS = {
    "add_acc": ("add_acc", 2), "sub_acc": ("sub_acc", 2), "negadd_acc": ("negadd_acc", 2),
    "negsub_acc": ("negsub_acc", 2), "neg_acc32": ("ACC1024_acc32_neg_conf", 1), "neg_acc64": ("ACC1024_acc64_neg_conf", 1),
}


def acc_kernel(kind: str, tiles: int) -> str:
    builtin, n = ACC_KINDS[kind]
    call = f"__builtin_aiev2_{builtin}(c1, c2, conf)" if n == 2 else f"__builtin_aiev2_{builtin}(c1, conf)"
    return f"""// Auto-generated (gen_mac_conf_probe.py accops): {kind} over {tiles} tiles with a per-tile conf word
#define NOCPP
#include <stdint.h>

extern "C" {{
void test_kernel(const int32_t *restrict in, int32_t *restrict out) {{
    for (int t = 0; t < {tiles}; t++) {{
        const int32_t *p = in + t * {TILE_IN // 4};
        v16acc64 c1 = *(const v16acc64 *)(p + 32);
        v16acc64 c2 = *(const v16acc64 *)(p + 64);
        int32_t conf = p[96];
        v16acc64 r = {call};
        *(v16acc64 *)(out + t * {TILE_OUT // 4}) = r;
    }}
}}
}} // extern "C"
"""


def plan_accops(rng):
    """The accumulator-only integer instructions (vadd, vsub, vnegadd, vnegsub, vneg) under every conf field."""
    out = []
    for kind in ACC_KINDS:
        confs = [am << 1 | z | sh << 10 | m << 11 | a1 << 12 | a2 << 13
                 for am in range(4) for z in (0, 1) for sh in (0, 1) for m in (0, 1) for a1 in (0, 1) for a2 in (0, 1)]
        confs += [rng.getrandbits(32) for _ in range(64)]
        for k, start in enumerate(range(0, len(confs), TILES_PER_CASE)):
            chunk = confs[start:start + TILES_PER_CASE]
            data = b"".join(tile_bytes(rng, c) for c in chunk)
            out.append(CxxKernel(name=f"accops__{kind}__{k:02d}", source=acc_kernel(kind, len(chunk)),
                                 in_size=len(data), out_size=TILE_OUT * len(chunk), input_bytes=data,
                                 meta={"plan": "accops", "kind": kind, "confs": chunk}))
    return out


def plan_upper(rng):
    out = []
    bits = [14, 15] + list(range(16, 32))
    for kind in KINDS:
        for gname, g in GEOMETRIES.items():
            confs = []
            for bit in bits:
                for rep in range(2):
                    confs.append(conf_word(**g) | 1 << bit)
            confs.append(conf_word(**g))
            out += cases_for(kind, f"upper_{gname}", confs, rng)
    return out


def plan_random(rng):
    out = []
    for kind in KINDS:
        confs = [rng.getrandbits(32) for _ in range(TILES_PER_CASE * 4)]
        out += cases_for(kind, "random", confs, rng)
    return out


PLANS = {"accops": plan_accops, "sparse": plan_sparse, "repro": plan_repro, "modes": plan_modes, "termneg": plan_termneg, "fsingle": plan_fsingle, "fprobe": plan_fprobe, "onehot": plan_onehot, "same": plan_same, "flags": plan_flags, "codes": plan_codes, "upper": plan_upper, "random": plan_random}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan", choices=sorted(PLANS))
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    kernels = PLANS[args.plan](rng)
    write_cxx_suite(args.suite_dir, f"macconf-{args.plan}", "tools/gen_mac_conf_probe.py",
                    "chess" if args.plan == "sparse" else "peano",
                    kernels, meta={"plan": args.plan, "seed": args.seed})
    print(f"{len(kernels)} cases -> {args.suite_dir}")


if __name__ == "__main__":
    main()
