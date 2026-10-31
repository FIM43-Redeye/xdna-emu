#!/usr/bin/env python3
"""Generate the capture kernels for the AIE2 sparse MAC modes.

Why this exists
---------------
aiev2_vmult.h builds five MAC config modes whose B operand is a sparse register
(`qx`: 512 bits of stored data plus a 128-bit mask). Which logical B element each
stored element stands for, and how the mask decides, is stated nowhere we may read,
so each mode is captured on NPU1 silicon:

  * a PROBE kernel (`vec_macsparse_<stem>_probe`) of designed tiles that determine the
    rule. The multiply is bilinear in (A, stored data) for a fixed mask, so a one-hot
    stored element against an A ramp reads out, per accumulator lane, which A element
    met it. Masks are zero, all ones, single bits, structured patterns and random.
  * an OPS kernel (`..._ops`): every MAC-family op of the mode over random tiles, with
    accumulator chains, in all four signedness flavours.
  * a HELD-OUT kernel (`..._rand`): seeded random tiles through mul -> mac -> msc. It
    validates the rule; it is never used to derive it.

The expected output of all of them is whatever NPU1 silicon produces (`capture` writes
the golden JSON the emulator tests read). No value is computed by a model.

The qx register is built in the kernel from three memory blocks (A, stored data, mask)
with the update() intrinsics, so the sparse loads (vldb.sparse.*) are not involved here;
they have their own capture (phase 2).

Accumulator readout: acc32 modes store the 32 lanes raw; acc64 modes store each lane as two
words (srs shift 0 then 32, crSat none); bf16 stores the 16 fp32 lanes.

Usage
-----
  python3 tools/gen_mac_sparse.py generate [<stem> ...]
  python3 tools/stage_vector_kernel.py <kernel>
  # run the bridge on HW once (an uncaptured kernel fails by design; out.txt is written
  # first), then:
  python3 tools/gen_mac_sparse.py capture <kernel> <build>/out.txt --date YYYY-MM-DD
  python3 tools/stage_vector_kernel.py <kernel>          # now stages exp.bin = silicon
"""

import argparse
import os
import random
import re
import struct
from dataclasses import dataclass

import capture_kernel_render as render
import gen_agu_2d3d as agu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY = os.path.join(REPO, "tests", "vector-verify")
GOLDEN_DIR = os.path.join(REPO, "tools", "golden", "silicon")
MAX_CHUNKS = 64         # shim DMA repeat-count limit
MAX_CHUNK_WORDS = 4096  # per input or output chunk (L1 budget)
MASK_BITS = 128
DATA_BYTES = 64         # stored data of one qx: 512 bits
MASK_BYTES = MASK_BITS // 8

# Signedness flavours, as (name, x_signed, y_signed): x is the A operand.
SIGNS = [("ss", 1, 1), ("su", 1, 0), ("us", 0, 1), ("uu", 0, 0)]
BF16_SIGNS = [("bf", 0, 0)]


@dataclass(frozen=True)
class Mode:
    """One sparse config mode: header intrinsic stem, operand shapes, accumulator."""
    key: tuple          # (amode, bmode, variant)
    stem: str           # header family name: mul_<stem>, mac_<stem>, ...
    a_bits: int
    a_lanes: int
    b_bits: int
    b_logical: int      # logical B elements (the sparse type's lane count)
    acc_bits: int       # 32 or 64 (bf16: 32, float)
    acc_lanes: int
    bf16: bool = False
    peano: bool = False     # Chess has no intrinsic for the mode; build the kernels with Peano

    @property
    def tile(self):
        """(rows, inner, cols) of the matrix tile the stem names: `<R>x<K>_<K>x<C>`."""
        left, right = self.stem.split("_")
        rows, inner = (int(x) for x in left.split("x"))
        _, cols = (int(x) for x in right.split("x"))
        return rows, inner, cols

    @property
    def b_stored(self):
        return self.b_logical // 2

    @property
    def a_bytes(self):
        return self.a_bits * self.a_lanes // 8

    @property
    def wide(self):
        """The A operand is a 1024-bit register (the `_sparse_wide` encoding)."""
        return self.a_bytes == 128

    @property
    def out_words(self):
        return self.acc_lanes * (self.acc_bits // 32)

    @property
    def kernel_stem(self):
        return "vec_macsparse_" + self.stem

    @property
    def signs(self):
        return BF16_SIGNS if self.bf16 else SIGNS

    def a_type(self, signed):
        if self.bf16:
            return f"v{self.a_lanes}bfloat16"
        return f"v{self.a_lanes}{'int' if signed else 'uint'}{self.a_bits}"

    def d_type(self, signed):
        if self.bf16:
            return f"v{self.b_stored}bfloat16"
        return f"v{self.b_stored}{'int' if signed else 'uint'}{self.b_bits}"

    def b_type(self, signed):
        if self.bf16:
            return f"v{self.b_logical}bfloat16_sparse"
        return f"v{self.b_logical}{'int' if signed else 'uint'}{self.b_bits}_sparse"

    @property
    def acc_type(self):
        return f"v{self.acc_lanes}accfloat" if self.bf16 else f"v{self.acc_lanes}acc{self.acc_bits}"


MODES = [
    Mode((0, 0, 1), "4x32_32x8", 8, 128, 4, 256, 32, 32, peano=True),
    Mode((0, 1, 5), "4x16_16x8", 8, 64, 8, 128, 32, 32),
    Mode((1, 2, 2), "2x16_16x8", 16, 32, 8, 128, 64, 16),
    Mode((1, 3, 5), "2x8_8x8", 16, 32, 16, 64, 64, 16),
    Mode((2, 3, 2), "4x16_16x4", 16, 64, 16, 64, 32, 16, bf16=True),
]
BY_STEM = {m.stem: m for m in MODES}

# Instruction sequences. Each op is (name, accumulator inputs from earlier results).
INT_OPS = ["mul", "negmul", "mac", "msc", "negmac", "negmsc", "addmac", "addmsc", "submac", "submsc"]
BF16_OPS = ["mul", "negmul", "mac", "msc", "addmac", "addmsc"]


def bf16_bits(x):
    """The bfloat16 bit pattern (truncated float32) of a number."""
    return struct.unpack("<I", struct.pack("<f", float(x)))[0] >> 16


# --- Designed inputs -----------------------------------------------------------

@dataclass(frozen=True)
class Set:
    """One qx operand set: A elements, stored data elements (raw bit patterns), 128-bit mask."""
    a: tuple
    d: tuple
    m: int


def element(mode, side, value):
    """Raw element bit pattern of a small non-negative designed value."""
    if mode.bf16:
        return bf16_bits(value)
    bits = mode.a_bits if side == "a" else mode.b_bits
    return value & ((1 << bits) - 1)


def ones(mode, side):
    n = mode.a_lanes if side == "a" else mode.b_stored
    return tuple([element(mode, side, 1)] * n)


def ramp(mode, side):
    """A ramp 1, 2, 3, ... clamped inside the unsigned element range (wrapped for 4 bits)."""
    n = mode.a_lanes if side == "a" else mode.b_stored
    bits = mode.a_bits if side == "a" else mode.b_bits
    span = (1 << bits) - 1 if bits < 16 else 1 << 16
    return tuple(element(mode, side, (i % min(span, 128 if not mode.bf16 else 128)) + 1) for i in range(n))


def zeros(mode, side):
    n = mode.a_lanes if side == "a" else mode.b_stored
    return tuple([0] * n)


def a_matrix(mode, fn):
    """A elements (row-major rows x inner) from fn(row, k) -> small non-negative value."""
    rows, inner, _ = mode.tile
    used = [element(mode, "a", fn(r, k)) for r in range(rows) for k in range(inner)]
    return tuple(used + [0] * (mode.a_lanes - len(used)))     # the register may be wider than the tile


def position_a(mode):
    """A whose product against all-ones stored data reads out the active K positions per column.

    Every A element is a distinct power of two within its lane's sum (8-bit A: row r carries
    k in [8r, 8r+8), so the 4 rows cover inner <= 32; wider A: every row carries 1 << k).
    """
    if mode.a_bits == 8 and not mode.bf16:
        return a_matrix(mode, lambda r, k: (1 << (k % 8)) if k // 8 == r else 0)
    return a_matrix(mode, lambda r, k: 1 << k)


def identity_a(mode, group):
    """A with one 1 per row: row r selects k = group * rows + r. D ramp then reads B[k][c] out."""
    rows, _, _ = mode.tile
    return a_matrix(mode, lambda r, k: int(k == group * rows + r))


def mask_patterns(seed=0x5A):
    """Designed 128-bit masks: empty, single bits, pairs, runs, periodic, nibble codes, random."""
    full = (1 << MASK_BITS) - 1
    masks = [0]
    masks += [1 << b for b in (0, 1, 2, 3, 15, 16, 17, 63, 64, 127)]
    masks += [(1 << a) | (1 << b) for a, b in
              ((0, 1), (0, 2), (0, 3), (3, 5), (0, 16), (15, 16), (0, 127), (64, 127), (7, 8), (126, 127))]
    masks += [(1 << n) - 1 for n in (2, 3, 4, 5, 8, 16, 32, 48, 64, 65, 96, 128)]
    for period in (2, 4, 8):                      # alternating and block patterns
        pat = sum(1 << b for b in range(MASK_BITS) if (b // (period // 2 if period > 1 else 1)) % 2 == 0)
        masks += [pat, full ^ pat]
    for v in range(1, 16):                        # one nibble code repeated
        masks.append(int(f"{v:x}" * (MASK_BITS // 4), 16))
    # Four bits 32 apart: the 32 masks together set every one of the 128 bits.
    masks += [sum(1 << (j + 32 * i) for i in range(4)) for j in range(32)]
    rng = random.Random(seed)
    masks += [rng.getrandbits(MASK_BITS) for _ in range(8)]
    two_of_four = [3, 5, 6, 9, 10, 12]
    masks += [int("".join(f"{rng.choice(two_of_four):x}" for _ in range(MASK_BITS // 4)), 16) for _ in range(8)]
    masks += [rng.getrandbits(MASK_BITS) & rng.getrandbits(MASK_BITS) & rng.getrandbits(MASK_BITS) for _ in range(6)]
    seen, out = set(), []
    for m in masks:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def identity_ramps(mode):
    """Stored-data vectors that name their own index: one ramp, or lo/hi digits for 4-bit data.

    A 4-bit element holds 0..15, fewer than the 128 stored elements, so the index is split in
    two base-16 digits read from two tiles.
    """
    if mode.b_bits != 4:
        return [ramp(mode, "b")]
    n = mode.b_stored
    return [tuple(s & 15 for s in range(n)), tuple(s >> 4 for s in range(n))]


def probe_plan(mode):
    """Per mask: a position tile (A powers of two, data ones) then identity tiles (data ramps)."""
    rows, inner, _ = mode.tile
    tiles = []
    for m in mask_patterns():
        tiles.append([Set(position_a(mode), ones(mode, "b"), m)])
        for g_ in range(inner // rows):
            for d in identity_ramps(mode):
                tiles.append([Set(identity_a(mode, g_), d, m)])
    return tiles


def extremes(bits):
    top = 1 << (bits - 1)
    return [0, 1, top - 1, top, (1 << bits) - 1]


def structured_mask(mode, rng):
    """A mask whose every group of four elements selects 0, 1 or 2 elements (the common case).

    Which elements a mask selects is designed from the *probe-derived* element layout only in
    the sense of group size and mask bits per element; the capture, not this function, says what
    the hardware does with the result.
    """
    _, inner, cols = mode.tile
    bpe = MASK_BITS // (inner * cols)
    mask = 0
    if bpe == 0:      # 4-bit B: fewer mask bits than elements; design at nibble level
        for nib in range(MASK_BITS // 4):
            for bit in rng.sample(range(4), rng.choices([0, 1, 2], [1, 3, 6])[0]):
                mask |= 1 << (4 * nib + bit)
        return mask
    for c in range(cols):
        for j in range(inner // 4):
            count = rng.choices([0, 1, 2], [1, 3, 6])[0]
            for p in rng.sample(range(4), count):
                n = c * inner + 4 * j + p
                mask |= rng.randrange(1, 1 << bpe) << (n * bpe)
    return mask


def random_mask(mode, rng):
    style = rng.randrange(4)
    if style < 2:
        return structured_mask(mode, rng)
    if style == 2:
        return rng.getrandbits(MASK_BITS)
    return rng.getrandbits(MASK_BITS) & rng.getrandbits(MASK_BITS) & rng.getrandbits(MASK_BITS)


def rand_sets(mode, n_tiles, rng, stages, a_lanes=None):
    """Seeded random tiles: full-range elements, structured and random masks."""
    tiles = []
    for _ in range(n_tiles):
        sets = []
        for _ in range(stages):
            if mode.bf16:
                a = tuple(bf16_bits(rng.uniform(-8, 8)) for _ in range(a_lanes or mode.a_lanes))
                d = tuple(bf16_bits(rng.uniform(-8, 8)) for _ in range(mode.b_stored))
            else:
                a = tuple(rng.randrange(1 << mode.a_bits) for _ in range(a_lanes or mode.a_lanes))
                d = tuple(rng.randrange(1 << mode.b_bits) for _ in range(mode.b_stored))
            sets.append(Set(a, d, random_mask(mode, rng)))
        tiles.append(sets)
    return tiles


# --- Kernel kinds ----------------------------------------------------------------

# Stages of the config-word variants (`*_conf` intrinsics): zero_acc1, shift16 (integer modes
# only), sub_mul, sub_acc1, sub_acc2. A row is (stage name, intrinsic, accumulator inputs,
# trailing integer arguments).
CONF_INT = [
    ("mulc_sm", "mul", 0, (1,)),
    ("mac_z", "mac", 1, (1, 0, 0, 0)), ("mac_sh", "mac", 1, (0, 1, 0, 0)),
    ("mac_sm", "mac", 1, (0, 0, 1, 0)), ("mac_sa", "mac", 1, (0, 0, 0, 1)),
    ("mac_sm_sa", "mac", 1, (0, 0, 1, 1)),
    ("addmac_z", "addmac", 2, (1, 0, 0, 0, 0)), ("addmac_sa2", "addmac", 2, (0, 0, 0, 0, 1)),
    ("addmac_all", "addmac", 2, (0, 0, 1, 1, 1)),
]
CONF_BF16 = [
    ("mulc_sm", "mul", 0, (1,)),
    ("mac_z", "mac", 1, (1, 0, 0)), ("mac_sm", "mac", 1, (0, 1, 0)), ("mac_sa", "mac", 1, (0, 0, 1)),
    ("mac_sm_sa", "mac", 1, (0, 1, 1)),
    ("addmac_z", "addmac", 2, (1, 0, 0, 0)), ("addmac_sa2", "addmac", 2, (0, 0, 0, 1)),
    ("addmac_all", "addmac", 2, (0, 1, 1, 1)),
]


def conf_rows(mode):
    return CONF_BF16 if mode.bf16 else CONF_INT


@dataclass(frozen=True)
class Kind:
    suffix: str
    sets: int           # operand sets per tile

    def ops(self, mode):
        """Output stages per signedness flavour, in order."""
        if self.suffix == "alt" and mode.bf16:
            # The narrow bf16 builtins other than mul are declared with an extra vector operand
            # in this Peano, so only the multiply is driven through the narrow encoding.
            return ["mul", "mul", "mul"]
        if self.suffix in ("ops", "alt"):
            return (["mul", "mul", "negmul"] + [o for o in (BF16_OPS if mode.bf16 else INT_OPS)
                                                  if o not in ("mul", "negmul")])
        if self.suffix == "conf":
            return ["mul", "mul"] + [row[0] for row in conf_rows(mode)]
        if self.suffix == "rand":
            return ["mul", "mac", "msc"]
        return ["mul"]

    def peano(self, mode):
        """Whether the kernel is built with Peano (alt: it calls the builtins directly)."""
        return mode.peano or self.suffix == "alt"


PROBE = Kind("probe", 1)
OPS = Kind("ops", 3)
RAND = Kind("rand", 3)
CONF = Kind("conf", 3)
ALT = Kind("alt", 3)
KINDS = {k.suffix: k for k in (PROBE, OPS, RAND, CONF, ALT)}
OPS_TILES = 40
RAND_TILES = 112


def alt_a_bytes(mode):
    """A bytes of the alternative encoding: the other register width (wide <-> narrow)."""
    return 64 if mode.wide else 128


def a_bytes_of(mode, kind):
    return alt_a_bytes(mode) if kind == "alt" else mode.a_bytes


def a_lanes_of(mode, kind):
    return a_bytes_of(mode, kind) * 8 // (16 if mode.bf16 else mode.a_bits)


def seed_of(mode, kind):
    return 0x5BA0 + mode.key[0] * 64 + mode.key[1] * 8 + mode.key[2] + 4096 * list(KINDS).index(kind)


def plan_of(mode, kind):
    if kind == "probe":
        return probe_plan(mode)
    n = RAND_TILES if kind == "rand" else OPS_TILES
    return rand_sets(mode, n, random.Random(seed_of(mode, kind)), KINDS[kind].sets, a_lanes_of(mode, kind))


def words_per_tile_in(mode, kind):
    return KINDS[kind].sets * (a_bytes_of(mode, kind) + DATA_BYTES + MASK_BYTES) // 4


def words_per_tile_out(mode, kind):
    return len(mode.signs) * len(KINDS[kind].ops(mode)) * mode.out_words


def tiles_per_chunk(mode, kind):
    """Largest power of two that keeps both chunk buffers inside the L1 budget."""
    t = 1
    while (2 * t * words_per_tile_out(mode, kind) <= MAX_CHUNK_WORDS
           and 2 * t * words_per_tile_in(mode, kind) <= MAX_CHUNK_WORDS):
        t *= 2
    return t


def tiles_of(mode, kind):
    """The designed tile list padded with zero tiles to a whole number of chunks."""
    tiles = plan_of(mode, kind)
    t = tiles_per_chunk(mode, kind)
    zero = [Set(tuple([0] * a_lanes_of(mode, kind)), zeros(mode, "b"), 0)] * KINDS[kind].sets
    while len(tiles) % t:
        tiles.append(list(zero))
    assert len(tiles) // t <= MAX_CHUNKS, (mode.stem, kind, len(tiles), t)
    return tiles


# --- Packing -----------------------------------------------------------------

def pack_elements(values, bits):
    """Little-endian packed bytes of unsigned element values (4-bit: low nibble first)."""
    if bits == 4:
        assert len(values) % 2 == 0
        return bytes(values[i] | (values[i + 1] << 4) for i in range(0, len(values), 2))
    fmt = {8: "B", 16: "H"}[bits]
    return struct.pack("<%d%s" % (len(values), fmt), *values)


def slot_bytes(mode, tile_sets, slot):
    """Bytes of operand slot `slot` of a tile: slots go A0 D0 M0 A1 D1 M1 ..."""
    s = tile_sets[slot // 3]
    which = slot % 3
    if which == 0:
        return pack_elements(s.a, mode.a_bits)
    if which == 1:
        return pack_elements(s.d, mode.b_bits)
    return s.m.to_bytes(MASK_BYTES, "little")


def slot_size(mode, kind, slot):
    return (a_bytes_of(mode, kind), DATA_BYTES, MASK_BYTES)[slot % 3]


def pack_input(mode, kind, tiles):
    """Per chunk: for each operand slot, T tiles of that block, as 32-bit words."""
    t = tiles_per_chunk(mode, kind)
    words = []
    for ch in range(len(tiles) // t):
        chunk = tiles[ch * t:(ch + 1) * t]
        data = b""
        for slot in range(3 * KINDS[kind].sets):
            for tile in chunk:
                data += slot_bytes(mode, tile, slot)
        words += list(struct.unpack("<%dI" % (len(data) // 4), data))
    return words


# --- Rendering -----------------------------------------------------------------

def op_call(mode, op, a, b, accs):
    """The header call for one op on operands a, b with accumulator variables `accs`."""
    return f"{op}_{mode.stem}({', '.join([a, b] + list(accs))})"


# Accumulator inputs each op takes, as indices into the results so far.
def op_accs(op):
    return {"mul": 0, "negmul": 0, "mac": 1, "msc": 1, "negmac": 1, "negmsc": 1,
            "addmac": 2, "addmsc": 2, "submac": 2, "submsc": 2}[op]


def conf_call(mode, name, op, nacc, args, sets):
    """One `*_conf` stage: operands a<sets>/b<sets>, accumulators r0 (and r1)."""
    accs = ["r0", "r1"][:nacc] if nacc else []
    a, b = f"a{sets}", f"b{sets}"
    return f"{op}_{mode.stem}_conf({', '.join([a, b] + accs + [str(x) for x in args])})"


def stage_lines(mode, kind, sname, xs, ys):
    """Source lines of one signedness flavour for one tile (header intrinsics)."""
    k = KINDS[kind]
    ta, td, tb = mode.a_type(xs), mode.d_type(ys), mode.b_type(ys)
    lines = [f"    {{ // {sname}"]
    for si in range(k.sets):
        sl = 3 * si
        lines.append(f"      {ta} a{si} = *({ta} *)(in{sl} + n * {a_bytes_of(mode, kind)});")
        lines.append(f"      {td} d{si} = *({td} *)(in{sl + 1} + n * {DATA_BYTES});")
        lines.append(f"      sparsity_t m{si} = *(sparsity_t *)(in{sl + 2} + n * {MASK_BYTES});")
        lines.append(f"      {tb} b{si};")
        if mode.peano:     # Peano leaves update() undefined for the sparse types; fill the fields
            lines.append(f"      b{si}.data = d{si};")
            lines.append(f"      b{si}.mask = m{si};")
        else:
            lines.append(f"      b{si} = update(b{si}, d{si});")
            lines.append(f"      b{si} = update(b{si}, m{si});")
    rt = mode.acc_type
    ops = k.ops(mode)
    if k.suffix in ("ops", "conf"):
        # x0 = mul(s0), x1 = mul(s1); the rest combine s2 with x0 (and x1).
        lines.append(f"      {rt} r0 = mul_{mode.stem}(a0, b0);")
        lines.append(f"      {rt} r1 = mul_{mode.stem}(a1, b1);")
        if k.suffix == "ops":
            for idx, op in enumerate(ops[2:], start=2):
                accs = {0: [], 1: ["r0"], 2: ["r0", "r1"]}[op_accs(op)]
                lines.append(f"      {rt} r{idx} = {op_call(mode, op, 'a2', 'b2', accs)};")
        else:
            for idx, (name, op, nacc, args) in enumerate(conf_rows(mode), start=2):
                lines.append(f"      {rt} r{idx} = {conf_call(mode, name, op, nacc, args, 2)};")
    else:
        lines.append(f"      {rt} r0 = mul_{mode.stem}(a0, b0);")
        if k.suffix == "rand":
            lines.append(f"      {rt} r1 = mac_{mode.stem}(a1, b1, r0);")
            lines.append(f"      {rt} r2 = msc_{mode.stem}(a2, b2, r1);")
    return lines


def alt_types(mode):
    """(A type, data type, builtin prefix, accumulator token) of the alternative encoding."""
    wide_alt = alt_a_bytes(mode) == 128
    if mode.bf16:
        return ("v64bfloat16" if wide_alt else "v32bfloat16", "v32bfloat16",
                "I1024" if wide_alt else "I512", "accfloat_bf")
    if wide_alt:
        return "v128c_t", "v16int32", "I1024", "acc32"
    return ("v64c_t" if mode.acc_bits == 32 else "v32s_t"), "v64c_t", "I512", f"acc{mode.acc_bits}"


def alt_stage_lines(mode, kind, sname, xs, ys):
    """Source lines of one flavour for the alternative encoding (builtins called directly)."""
    k = KINDS[kind]
    ta, td, pre, acct = alt_types(mode)
    amode, bmode, var = mode.key
    conf = f"(({xs} << 9) | ({ys} << 8) | ({amode} << 1) | ({bmode} << 3) | ({var} << 5))"
    lines = [f"    {{ // {sname}", f"      const int conf = {conf};"]
    for si in range(k.sets):
        sl = 3 * si
        lines.append(f"      {ta} a{si} = *({ta} *)(in{sl} + n * {a_bytes_of(mode, kind)});")
        lines.append(f"      {td} d{si} = *({td} *)(in{sl + 1} + n * {DATA_BYTES});")
        lines.append(f"      sparsity_t m{si} = *(sparsity_t *)(in{sl + 2} + n * {MASK_BYTES});")

    def call(op, a, accs):
        two = len(accs) == 2
        name = f"__builtin_aiev2_{pre}_I1024_ACC1024_{'ACC1024_' if two else ''}{acct}_{op}_conf"
        return f"{name}({', '.join([f'a{a}', f'd{a}', f'm{a}'] + accs + ['conf'])})"

    lines.append(f"      auto r0 = {call('mul', 0, [])};")
    lines.append(f"      auto r1 = {call('mul', 1, [])};")
    for idx, op in enumerate(k.ops(mode)[2:], start=2):
        accs = {0: [], 1: ["r0"], 2: ["r0", "r1"]}[op_accs(op)]
        lines.append(f"      auto r{idx} = {call(op, 2, accs)};")
    return lines


def store_macro(mode):
    if mode.bf16:
        return ("#define STORE(r, dst) \\\n  do { aie::accum<accfloat, 16> w(r); "
                "aie::store_v((float *)(dst), w.to_vector<float>()); } while (0)"), ""
    if mode.acc_bits == 32:
        return (f"#define STORE(r, dst) \\\n  do {{ aie::accum<acc32, {mode.acc_lanes}> w(r); "
                f"aie::store_v((dst), w.to_vector<int32>(0)); }} while (0)"), ""
    return (f"#define STORE(r, dst) \\\n  do {{ aie::accum<acc64, {mode.acc_lanes}> w(r); "
            f"aie::store_v((dst), w.to_vector<int32>(0)); "
            f"aie::store_v((dst) + {mode.acc_lanes}, w.to_vector<int32>(32)); }} while (0)",
            "  // Read the 64-bit lanes as two 32-bit halves: srs by 0 and by 32 with no saturation.\n"
            "  aie::set_saturation(aie::saturation_mode::none);\n"
            "  aie::set_rounding(aie::rounding_mode::floor);\n")


def render_kernel(mode, kind):
    k = KINDS[kind]
    ops = k.ops(mode)
    t = tiles_per_chunk(mode, kind)
    lines = []
    for ci, (sname, xs, ys) in enumerate(mode.signs):
        lines += (alt_stage_lines if kind == "alt" else stage_lines)(mode, kind, sname, xs, ys)
        for st in range(len(ops)):
            lines.append(f"      STORE(r{st}, o + {(ci * len(ops) + st) * mode.out_words});")
        lines.append("    }")
    body = "\n".join(lines)
    slots, off = [], "0"
    for slot in range(3 * k.sets):
        slots.append(f"  uint8_t *in{slot} = in + {off};")
        off += f" + T * {slot_size(mode, kind, slot)}"
    slot_decl = "\n".join(slots)
    store, setup = store_macro(mode)
    how = ("calls the " + ("wide" if alt_a_bytes(mode) == 128 else "narrow") +
           "-encoding builtin directly (the header intrinsic of this mode uses the other)"
           if kind == "alt" else "builds the sparse B operand with update() from a stored-data block and a mask block")
    typedefs = ""
    if kind == "alt":
        typedefs = ("typedef char v128c_t __attribute__((vector_size(128)));\n"
                    "typedef char v64c_t __attribute__((vector_size(64)));\n"
                    "typedef short v32s_t __attribute__((vector_size(64)));\n")
    return f"""//===- mac_sparse.cc --------------------------------------------*- C++ -*-===//
//
// {kind} capture kernel for sparse MAC config mode {mode.key} `{mode.stem}`
// (GENERATED by tools/gen_mac_sparse.py). Per tile and signedness flavour it {how},
// then runs {', '.join(ops)}.
// Chunk layout: {3 * k.sets} operand blocks of T tiles ({' '.join(f'A{i} D{i} M{i}' for i in range(k.sets))}).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

// Peano builds see no Chess-side define; the stream/ADF half of aie_api is not needed.
#ifndef __AIE_API_AIE_ADF_HPP__
#define __AIE_API_AIE_ADF_HPP__
#endif
#include <aie_api/aie.hpp>

{typedefs}#define T {t}
#define OUT_WORDS_PER_TILE {words_per_tile_out(mode, kind)}

{store}

extern "C" {{

void mac_sparse(uint8_t *restrict in, int32_t *restrict out) {{
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


def flat_columns(mode, tiles):
    """Golden `inputs`: one flat list per operand slot (A0 D0 M0 ...); a mask is 32 hex digits (JSON has no 128-bit integers)."""
    cols = []
    for slot in range(3 * len(tiles[0])):
        which, si = slot % 3, slot // 3
        col = []
        for tile in tiles:
            s = tile[si]
            col += list(s.a) if which == 0 else list(s.d) if which == 1 else [f"{s.m:032x}"]
        cols.append(col)
    return cols


def kernel_data(mode, kind):
    """Raw input and expected buffers (see vv_data.KernelData) for staging."""
    tiles = tiles_of(mode, kind)
    golden = render.load_golden(golden_path(mode, kind))
    if golden is not None and golden["inputs"] != flat_columns(mode, tiles):
        raise SystemExit(f"{kernel_name(mode, kind)}: golden inputs are stale; recapture on silicon")
    expected = None if golden is None else golden["silicon"]
    assert expected is None or len(expected) == len(tiles) * words_per_tile_out(mode, kind)
    return render.kernel_data(pack_input(mode, kind, tiles), expected)


def render_run_lit(mode, kind, n_out):
    """Chess recipe; a Peano-built kernel gets the same recipe and is built with --peano-only."""
    text = render.render_run_lit("mac_sparse.cc", n_out)
    if KINDS[kind].peano(mode):
        text = text.replace("REQUIRES: ryzen_ai_npu1, chess",
                            "REQUIRES: ryzen_ai_npu1, chess, peano")
        text = text.replace("//\n// RUN: xchesscc_wrapper",
                            "// Built with Peano (Chess has no intrinsic for this form): run the bridge with\n"
                            "// --peano-only; the Chess leg below is only the recipe the bridge rewrites.\n"
                            "//\n// RUN: xchesscc_wrapper", 1)
    return text


def generate_one(mode, kind):
    tiles = tiles_of(mode, kind)
    kernel_data(mode, kind)   # fails on a stale golden before anything is written
    t = tiles_per_chunk(mode, kind)
    chunks = len(tiles) // t
    in_words = pack_input(mode, kind, tiles)
    nin = len(in_words) // chunks
    nout = t * words_per_tile_out(mode, kind)
    name = kernel_name(mode, kind)
    ops = KINDS[kind].ops(mode)
    files = {
        "mac_sparse.cc": render_kernel(mode, kind),
        "aie.mlir": render.render_mlir(
            name, "mac_sparse", chunks, nin, nout, "i32", "i32",
            f"sparse mode {mode.key} {mode.stem}; {len(mode.signs)} flavour(s) x {len(ops)} stage(s) "
            f"x {mode.out_words} words per tile."),
        "run.lit": render_run_lit(mode, kind, len(tiles) * words_per_tile_out(mode, kind)),
    }
    render.write_files(os.path.join(VERIFY, name), files)
    return name, len(tiles)


def capture(mode, kind, out_txt, date, elf=None):
    tiles = tiles_of(mode, kind)
    ops = KINDS[kind].ops(mode)
    silicon = render.read_out_txt(out_txt)
    n_out = len(tiles) * words_per_tile_out(mode, kind)
    assert len(silicon) == n_out, f"{len(silicon)} != {n_out}"
    assert all(0 <= v < 1 << 32 for v in silicon)
    forms = []
    conf = {name: (op, args) for name, op, _, args in conf_rows(mode)} if kind == "conf" else {}
    names = ("zero_acc1", "shift16", "sub_mul", "sub_acc1", "sub_acc2")
    for sname, xs, ys in mode.signs:
        for st, instr in enumerate(ops):
            form = {"name": f"{instr}_{sname}", "instr": instr, "stage": st, "sgn_x": xs, "sgn_y": ys}
            if instr in conf:       # the config-word controls of a `*_conf` stage
                base, args = conf[instr]
                if base == "mul":
                    wanted = dict(sub_mul=args[0])
                else:
                    order = ["zero_acc1"] + (["shift16"] if not mode.bf16 else []) + ["sub_mul", "sub_acc1"]
                    order += ["sub_acc2"] if base == "addmac" else []
                    wanted = dict(zip(order, args))
                form.update({n: wanted.get(n, 0) for n in names})
                form["instr"] = base
            forms.append(form)
    rec = {
        "kernel": kernel_name(mode, kind), "class": "mac_sparse_" + kind, "n": n_out,
        "mode": {"amode": mode.key[0], "bmode": mode.key[1], "variant": mode.key[2],
                 "stem": mode.stem, "a_bits": mode.a_bits, "a_lanes": mode.a_lanes,
                 "b_bits": mode.b_bits, "b_logical": mode.b_logical, "b_stored": mode.b_stored,
                 "acc_bits": mode.acc_bits, "acc_lanes": mode.acc_lanes, "bf16": mode.bf16,
                 "wide": mode.wide},
        "tile": {"tiles": len(tiles), "out_words": mode.out_words, "stages": len(ops),
                 "a_lanes": a_lanes_of(mode, kind),
                 "sets": KINDS[kind].sets, "ops": ops},
        "forms": forms,
        "encodings": [{"mnemonic": mn, "encoding": enc, "count": e["count"], "bytes": e["bytes"],
                       "bundle": e["bundle"]} for (mn, enc), e in sorted(proof(elf).items())] if elf else [],
        "inputs": flat_columns(mode, tiles),
        "silicon": silicon,
        "provenance": f"HW-observed: NPU1 Phoenix silicon, captured {date} via tools/gen_mac_sparse.py",
    }
    return render.write_golden(golden_path(mode, kind), rec)


# --- disassembly proof ------------------------------------------------------

SPARSE_MNEMONIC = re.compile(r"^v(?:neg)?(?:mul|mac|msc)(?:\.f)?$|^v(?:add|sub)(?:mac|msc)(?:\.f)?$")


def proof(elf):
    """Per (mnemonic, A register class): the first bundle with a sparse (qx) MAC form and a count.

    A form is sparse when one of its operands is a qx register; its `_sparse_narrow` or
    `_sparse_wide` encoding shows in the A register: x (512-bit) is narrow, y (1024-bit) wide.
    """
    found = {}
    listing, skipped = agu.disassemble(elf)
    if skipped:
        print(f"note: llvm-objdump cannot print the bundles at {[hex(a) for a in skipped]}; skipped")
    for raw in listing.splitlines():
        m = agu.LINE.match(raw)
        if not m:
            continue
        _, hexbytes, text = m.groups()
        for slot in (sl.strip() for sl in text.split(";")):
            parts = slot.replace(",", " ").split()
            if not parts or not SPARSE_MNEMONIC.match(parts[0]) or not any(p.startswith("qx") for p in parts):
                continue
            a_reg = next((p for p in parts[1:] if p[0] in "xy" and p[1:].isdigit()), "?")
            key = (parts[0], "narrow" if a_reg[0] == "x" else "wide")
            found.setdefault(key, {"count": 0, "bytes": " ".join(hexbytes.split()), "bundle": text.strip()})
            found[key]["count"] += 1
    return found


def kernel_names():
    return [kernel_name(m, k) for m in MODES for k in KINDS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("stems", nargs="*", help="mode stems (default: all)")
    cap = sub.add_parser("capture")
    cap.add_argument("kernel")
    cap.add_argument("out_txt")
    cap.add_argument("--date", required=True)
    cap.add_argument("--elf")
    prf = sub.add_parser("proof")
    prf.add_argument("elf")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        for mode in [BY_STEM[s] for s in args.stems] if args.stems else MODES:
            for kind in KINDS:
                name, n = generate_one(mode, kind)
                print(f"wrote {name} ({n} tiles)")
    elif args.cmd == "proof":
        for (mn, enc), e in sorted(proof(args.elf).items()):
            print(f"{mn:10s} {enc:7s} x{e['count']:<3d} {e['bytes']:34s} {e['bundle']}")
    else:
        for mode in MODES:
            for kind in KINDS:
                if kernel_name(mode, kind) == args.kernel:
                    print("wrote", capture(mode, kind, args.out_txt, args.date, args.elf))
                    return
        raise SystemExit(f"unknown kernel {args.kernel}")


if __name__ == "__main__":
    main()
