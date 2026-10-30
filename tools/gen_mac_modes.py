#!/usr/bin/env python3
"""Generate the capture kernels for the integer MAC config modes with no matmul tile.

Why this exists
---------------
aiev2_vmult.h builds eight integer MAC config modes that are not matmul tiles:
elementwise (`elem_32_2`, `elem_32`, `elem_16_2`), convolution (`conv_4x4_8ch`,
`conv_8x8_4ch`, `conv_32x8`, `conv_16x4`) and two-channel (`4x4_4x4_2ch`). Which A
and B elements feed each accumulator lane is not stated anywhere, so each mode is
captured on silicon twice:

  * a PROBE kernel (`vec_macmode_<stem>_probe`) that determines the lane map. The
    multiply is bilinear, so with A one-hot (A = e_i) the output of lane l is the sum
    of B[j] over the partners j of i in that lane. Against B = all ones that is the
    partner count; against the ramp B[j] = j + 1 it is the partner's index plus one
    when the count is one. Both sides are probed (one-hot B against A ones and A
    ramp), in all four signedness flavours, with small positive data.
  * a HELD-OUT kernel (`vec_macmode_<stem>_rand`) of seeded random data over the
    full value range plus extreme tiles, through a mul -> mac -> msc chain, in all
    four signedness flavours. It validates the lane map; it is never used to derive it.

The expected output of both is whatever NPU1 silicon produces (`capture` writes the
golden JSON the emulator tests read). No value is computed by a model.

Accumulator readout: acc32 modes store the 32 lanes raw (srs shift 0 of a 32-bit lane is
the identity). acc64 modes store each lane as two words: the low 32 bits (srs shift 0) then
bits 63:32 (srs shift 32), with crSat = none so the shift only truncates.

Usage
-----
  python3 tools/gen_mac_modes.py generate [<stem> ...]   # run.lit, aie.mlir, kernel .cc
  python3 tools/stage_vector_kernel.py <kernel>          # no exp.bin yet
  # run the bridge on HW once (an uncaptured kernel fails by design; out.txt is written
  # first), then:
  python3 tools/gen_mac_modes.py capture <kernel> <build>/out.txt --date YYYY-MM-DD
  python3 tools/stage_vector_kernel.py <kernel>          # now stages exp.bin = silicon
"""

import argparse
import os
import random
import struct
from dataclasses import dataclass

import capture_kernel_render as render

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY = os.path.join(REPO, "tests", "vector-verify")
GOLDEN_DIR = os.path.join(REPO, "tools", "golden", "silicon")
MAX_CHUNKS = 64   # shim DMA repeat-count limit

# Signedness flavours, as (name, x_signed, y_signed): x is the A operand.
SIGNS = [("ss", 1, 1), ("su", 1, 0), ("us", 0, 1), ("uu", 0, 0)]


@dataclass(frozen=True)
class Mode:
    """One config mode: the header intrinsic stem, operand shapes and accumulator."""
    key: tuple          # (amode, bmode, variant)
    stem: str           # header family name: mul_<stem>, mac_<stem>, msc_<stem>
    a_bits: int
    a_lanes: int
    b_bits: int
    b_lanes: int
    acc_bits: int       # 32 or 64
    acc_lanes: int

    @property
    def a_bytes(self):
        return self.a_bits * self.a_lanes // 8

    @property
    def b_bytes(self):
        return self.b_bits * self.b_lanes // 8

    @property
    def out_words(self):
        """Output words per accumulator readout: one per acc32 lane, lo+hi per acc64 lane."""
        return self.acc_lanes * (self.acc_bits // 32)

    @property
    def kernel_stem(self):
        return "vec_macmode_" + self.stem

    def vtype(self, side, signed):
        bits, lanes = (self.a_bits, self.a_lanes) if side == "a" else (self.b_bits, self.b_lanes)
        return f"v{lanes}{'int' if signed else 'uint'}{bits}"

    @property
    def acc_type(self):
        return f"v{self.acc_lanes}acc{self.acc_bits}"


MODES = [
    Mode((0, 1, 1), "elem_32_2", 8, 64, 8, 64, 32, 32),
    Mode((0, 1, 2), "conv_4x4_8ch", 8, 64, 8, 64, 32, 32),
    Mode((0, 1, 3), "conv_8x8_4ch", 8, 64, 8, 64, 32, 32),
    Mode((0, 1, 4), "conv_32x8", 8, 64, 8, 64, 32, 32),
    Mode((0, 2, 1), "4x4_4x4_2ch", 16, 32, 8, 64, 32, 32),
    Mode((0, 3, 1), "elem_32", 16, 32, 16, 32, 32, 32),
    Mode((1, 3, 2), "elem_16_2", 16, 32, 16, 32, 64, 16),
    Mode((1, 3, 3), "conv_16x4", 16, 32, 16, 32, 64, 16),
]
BY_STEM = {m.stem: m for m in MODES}


# --- Designed inputs -----------------------------------------------------------

def probe_tiles(mode):
    """Probe tiles as (A, B) element lists, small positive values only.

    For each A index i: A = e_i against B = ones, then against the ramp B[j] = j + 1.
    For each B index j: the mirror image. Tile order is part of the decode and fixed here;
    the ramp tops out at 64, inside every signed and unsigned element range used.
    """
    na, nb = mode.a_lanes, mode.b_lanes
    ramp_a = list(range(1, na + 1))
    ramp_b = list(range(1, nb + 1))
    tiles = []
    for i in range(na):
        one_hot = [int(k == i) for k in range(na)]
        tiles.append((one_hot, [1] * nb))
        tiles.append((one_hot, list(ramp_b)))
    for j in range(nb):
        one_hot = [int(k == j) for k in range(nb)]
        tiles.append(([1] * na, one_hot))
        tiles.append((list(ramp_a), one_hot))
    return tiles


def extremes(bits):
    """Boundary patterns of an element width: 0, 1, max positive, min negative, -1."""
    top = 1 << (bits - 1)
    return [0, 1, top - 1, top, (1 << bits) - 1]


def rand_operand_sets(mode, n_tiles, rng):
    """Held-out tiles: six operand vectors each (A0 B0 A1 B1 A2 B2).

    25 constant tiles (every extreme pair, all three stages), 32 tiles of per-element extremes,
    the rest uniform over the full element range.
    """
    ea, eb = extremes(mode.a_bits), extremes(mode.b_bits)
    tiles = []
    for xa in ea:
        for xb in eb:
            tiles.append([[xa] * mode.a_lanes, [xb] * mode.b_lanes] * 3)
    for _ in range(32):
        tiles.append([[rng.choice(ea) for _ in range(mode.a_lanes)] if s % 2 == 0
                      else [rng.choice(eb) for _ in range(mode.b_lanes)] for s in range(6)])
    while len(tiles) < n_tiles:
        tiles.append([[rng.randrange(1 << mode.a_bits) for _ in range(mode.a_lanes)] if s % 2 == 0
                      else [rng.randrange(1 << mode.b_bits) for _ in range(mode.b_lanes)]
                      for s in range(6)])
    return tiles[:n_tiles]


# --- Kernel shapes -------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """How a kernel kind lays out its chunks."""
    suffix: str
    t: int              # tiles per chunk
    stages: int         # accumulator readouts per sign flavour per tile (1 = mul only)
    operands: int       # operand vectors per tile (2 per stage)


PROBE = Shape("probe", 32, 1, 2)
RAND = Shape("rand", 16, 3, 6)
SHAPES = {s.suffix: s for s in (PROBE, RAND)}
RAND_TILES = 112


def tiles_of(mode, kind):
    """The designed tile list for a kernel kind: each tile is a list of operand vectors."""
    if kind == "probe":
        tiles = [[a, b] for a, b in probe_tiles(mode)]
        zero = [[0] * mode.a_lanes, [0] * mode.b_lanes]
    else:
        tiles = rand_operand_sets(mode, RAND_TILES, random.Random(0x3AC0 + mode.key[0] * 64 + mode.key[1] * 8 + mode.key[2]))
        zero = [[0] * mode.a_lanes, [0] * mode.b_lanes] * 3
    shape = SHAPES[kind]
    while len(tiles) % shape.t:
        tiles.append([list(v) for v in zero])
    assert len(tiles) // shape.t <= MAX_CHUNKS, len(tiles)
    return tiles


def pack_elements(values, bits):
    """Little-endian packed bytes of unsigned element values."""
    fmt = {8: "B", 16: "H"}[bits]
    return struct.pack("<%d%s" % (len(values), fmt), *values)


def pack_input(mode, kind, tiles):
    """Per chunk: for each operand slot, T tiles of that vector, as 32-bit words."""
    shape = SHAPES[kind]
    words = []
    for ch in range(len(tiles) // shape.t):
        chunk = tiles[ch * shape.t:(ch + 1) * shape.t]
        data = b""
        for slot in range(shape.operands):
            bits = mode.a_bits if slot % 2 == 0 else mode.b_bits
            for tile in chunk:
                data += pack_elements(tile[slot], bits)
        words += list(struct.unpack("<%dI" % (len(data) // 4), data))
    return words


def out_words_per_tile(mode, shape):
    return len(SIGNS) * shape.stages * mode.out_words


# --- Rendering -----------------------------------------------------------------

def render_kernel(mode, kind):
    shape = SHAPES[kind]
    rt = mode.acc_type
    lines = []
    emit = lines.append
    for ci, (sname, xs, ys) in enumerate(SIGNS):
        ta, tb = mode.vtype("a", xs), mode.vtype("b", ys)
        emit(f"    {{ // {sname}")
        for slot in range(shape.operands):
            side = "a" if slot % 2 == 0 else "b"
            ty = ta if side == "a" else tb
            nbytes = mode.a_bytes if side == "a" else mode.b_bytes
            emit(f"      {ty} v{slot} = *({ty} *)(in{slot} + n * {nbytes});")
        emit(f"      {rt} r0 = mul_{mode.stem}(v0, v1);")
        if shape.stages > 1:
            emit(f"      {rt} r1 = mac_{mode.stem}(v2, v3, r0);")
            emit(f"      {rt} r2 = msc_{mode.stem}(v4, v5, r1);")
        for st in range(shape.stages):
            emit(f"      STORE(r{st}, o + {(ci * shape.stages + st) * mode.out_words});")
        emit("    }")
    body = "\n".join(lines)
    slots = []
    off = "0"
    for slot in range(shape.operands):
        nbytes = mode.a_bytes if slot % 2 == 0 else mode.b_bytes
        slots.append(f"  uint8_t *in{slot} = in + {off};")
        off += f" + T * {nbytes}"
    slot_decl = "\n".join(slots)
    if mode.acc_bits == 32:
        store = (f"#define STORE(r, dst) \\\n  do {{ aie::accum<acc32, {mode.acc_lanes}> w(r); "
                 f"aie::store_v((dst), w.to_vector<int32>(0)); }} while (0)")
        setup = ""
    else:
        store = (f"#define STORE(r, dst) \\\n  do {{ aie::accum<acc64, {mode.acc_lanes}> w(r); "
                 f"aie::store_v((dst), w.to_vector<int32>(0)); "
                 f"aie::store_v((dst) + {mode.acc_lanes}, w.to_vector<int32>(32)); }} while (0)")
        setup = ("  // Read the 64-bit lanes as two 32-bit halves: srs by 0 and by 32 with no saturation.\n"
                 "  aie::set_saturation(aie::saturation_mode::none);\n"
                 "  aie::set_rounding(aie::rounding_mode::floor);\n")
    return f"""//===- mac_mode.cc ----------------------------------------------*- C++ -*-===//
//
// {kind} capture kernel for MAC config mode {mode.key} `{mode.stem}`
// (GENERATED by tools/gen_mac_modes.py). Per tile and signedness flavour (ss, su, us,
// uu; first letter is A) it runs mul_{mode.stem}{', then mac and msc chained on the result' if shape.stages > 1 else ''}.
// Chunk layout: {shape.operands} operand blocks of T tiles ({'A0 B0 A1 B1 A2 B2' if shape.operands == 6 else 'A, B'}).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define T {shape.t}
#define OUT_WORDS_PER_TILE {out_words_per_tile(mode, shape)}

{store}

extern "C" {{

void mac_mode(uint8_t *restrict in, int32_t *restrict out) {{
{slot_decl}
{setup}  event0();
  for (int n = 0; n < T; n++) {{
    int32_t *o = out + n * OUT_WORDS_PER_TILE;
{body}
  }}
  event1();
}}

}} // extern "C"
"""


def kernel_name(mode, kind):
    return f"{mode.kernel_stem}_{kind}"


def golden_path(mode, kind):
    return os.path.join(GOLDEN_DIR, kernel_name(mode, kind) + ".json")


def expected_words(mode, kind):
    """The silicon golden's output words, or None before the first capture."""
    golden = render.load_golden(golden_path(mode, kind))
    return None if golden is None else golden["silicon"]


def flat_columns(tiles):
    return [[x for t in tiles for x in t[slot]] for slot in range(len(tiles[0]))]


def kernel_data(mode, kind):
    """Raw input and expected buffers (see vv_data.KernelData) for staging."""
    shape = SHAPES[kind]
    tiles = tiles_of(mode, kind)
    golden = render.load_golden(golden_path(mode, kind))
    if golden is not None and golden["inputs"] != flat_columns(tiles):
        raise SystemExit(f"{kernel_name(mode, kind)}: golden inputs are stale; recapture on silicon")
    expected = expected_words(mode, kind)
    assert expected is None or len(expected) == len(tiles) * out_words_per_tile(mode, shape)
    return render.kernel_data(pack_input(mode, kind, tiles), expected)


def generate_one(mode, kind):
    shape = SHAPES[kind]
    tiles = tiles_of(mode, kind)
    n_out = len(tiles) * out_words_per_tile(mode, shape)
    kernel_data(mode, kind)   # fails on a stale golden before anything is written
    chunks = len(tiles) // shape.t
    in_words = pack_input(mode, kind, tiles)
    nin = len(in_words) // chunks
    nout = shape.t * out_words_per_tile(mode, shape)
    name = kernel_name(mode, kind)
    files = {
        "mac_mode.cc": render_kernel(mode, kind),
        "aie.mlir": render.render_mlir(
            name, "mac_mode", chunks, nin, nout, "i32", "i32",
            f"mode {mode.key} {mode.stem}; {len(SIGNS)} signedness flavours x {shape.stages} stage(s) "
            f"x {mode.out_words} words per tile."),
        "run.lit": render.render_run_lit("mac_mode.cc", n_out),
    }
    render.write_files(os.path.join(VERIFY, name), files)
    return name, len(tiles)


def capture(mode, kind, out_txt, date):
    shape = SHAPES[kind]
    tiles = tiles_of(mode, kind)
    silicon = render.read_out_txt(out_txt)
    n_out = len(tiles) * out_words_per_tile(mode, shape)
    assert len(silicon) == n_out, f"{len(silicon)} != {n_out}"
    assert all(0 <= v < 1 << 32 for v in silicon)
    forms = []
    for sname, xs, ys in SIGNS:
        for st, instr in enumerate(("mul", "mac", "msc")[:shape.stages]):
            forms.append({"name": f"{instr}_{sname}", "instr": instr, "stage": st,
                          "sgn_x": xs, "sgn_y": ys})
    rec = {
        "kernel": kernel_name(mode, kind), "class": "mac_mode_" + kind, "n": n_out,
        "mode": {"amode": mode.key[0], "bmode": mode.key[1], "variant": mode.key[2],
                 "stem": mode.stem, "a_bits": mode.a_bits, "a_lanes": mode.a_lanes,
                 "b_bits": mode.b_bits, "b_lanes": mode.b_lanes, "acc_bits": mode.acc_bits,
                 "acc_lanes": mode.acc_lanes},
        "tile": {"tiles": len(tiles), "out_words": mode.out_words, "stages": shape.stages},
        "forms": forms,
        "inputs": flat_columns(tiles),
        "silicon": silicon,
        "provenance": f"HW-observed: NPU1 Phoenix silicon, captured {date} via tools/gen_mac_modes.py",
    }
    return render.write_golden(golden_path(mode, kind), rec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("stems", nargs="*", help="mode stems (default: all)")
    cap = sub.add_parser("capture")
    cap.add_argument("kernel")
    cap.add_argument("out_txt")
    cap.add_argument("--date", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        for mode in [BY_STEM[s] for s in args.stems] if args.stems else MODES:
            for kind in SHAPES:
                name, n = generate_one(mode, kind)
                print(f"wrote {name} ({n} tiles)")
    else:
        for mode in MODES:
            for kind in SHAPES:
                if kernel_name(mode, kind) == args.kernel:
                    print("wrote", capture(mode, kind, args.out_txt, args.date))
                    return
        raise SystemExit(f"unknown kernel {args.kernel}")


if __name__ == "__main__":
    main()
