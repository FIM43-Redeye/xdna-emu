#!/usr/bin/env python3
"""Generate the capture kernels for the AIE2 pack instructions.

Why this exists
---------------
AIE2 narrows packed integers two ways: the fused store `vst.pack.*` (a 512-bit x register
narrowed into 256 bits of memory) and the standalone `vpack.*` (a 512-bit x register
narrowed into a 256-bit w register). Each has four forms: `s8.s16`, `s4.s8` (the `s` forms)
and `d8.d16`, `d4.d8` (the `d` forms). TableGen says every form reads the saturation
register `crSat` and the `d` forms also read `crPackSign`. What the sign register means, what
each of the four `crSat` values does, the exact clamp bounds and the lane order of the result
are not stated anywhere we may read, so they are captured from NPU1 silicon. Only
`vpack.s8.s16` under crSat 0, 1 and 3 had been checked (`vec_pack_i16*`).

Kernels
-------
  vec_pack_lane_probe   designed blocks: one-hot int16 lanes and bytes, ramps, clamp-boundary
                        values rotated through every lane position, every byte value at every
                        byte position, a dense run around zero, constant fills.
  vec_pack_lane_rand    a seeded random held-out set; never used to derive a rule.
  vec_pack_addr         fused-store addressing: runtime index offsets (`[p, dj]`) and runtime
                        post-modify steps (`[p], m`), with the data landing in reverse and
                        shifted slots so a wrong address shows.
  vec_pack_crtiming     the sign comes from a run-time table inside a software-pipelined
                        loop, where Chess writes crPackSign in the same bundle as (or just
                        before) the `d` form it controls.

The lane kernels run every block through all four forms, fused and standalone, under every
crSat value (0, 1, 2, 3) and both crPackSign values (0, 1). The `s` forms are run under both
sign values too, so the output shows whether the register is ignored there.

Source view of a block (64 bytes): 32 little-endian int16 lanes for the 16-to-8 forms and
64 int8 lanes for the 8-to-4 forms. The result is always 256 bits (8 words): 32 bytes or 64
nibbles.

Input chunk (32-bit words)
--------------------------
  header[0], header[1] = 0, 1    the two crPackSign values, read at run time so the compiler
                                 must emit the register-driven `d` forms
  header[2]                      unused
  header[3] = -32                post-modify step in bytes (vec_pack_addr; runtime so `[p], m`)
  header[4..8)  = 0, 1, 2, 3     the four crSat values
  header[8..12) = -64,-32,0,32   index offsets in bytes (vec_pack_addr; runtime so `[p, dj]`)
  header[16..24) = 0             a zero vector added to the standalone results so the compiler
                                 cannot fuse the pack into the store
Then BLOCKS_PER_CHUNK blocks of 16 words (64 bytes). The crSat value is written by
`set_saturation` with a run-time argument (crSat 2 has no aie_api name; a run-time value
reaches it all the same).

Output per block: 8 passes, pass = 2 * crSat + crPackSign, each pass 64 words (8 vectors of
8 words): the fused group then the standalone group, forms in the order s8.s16, d8.d16,
s4.s8, d4.d8. The addr kernel has the same shape with the index group and the post-modify
group in place of fused/standalone.

The group names are the source pattern; the compiler picks the final addressing mode, so
the disassembly of the core ELF, not the group name, says which instruction ran. `proof`
prints it; `capture --elf` records it in the golden.

Usage
-----
  python3 tools/gen_pack.py generate
  python3 tools/stage_vector_kernel.py <kernel>          # no exp.bin yet
  # run the bridge on HW once (an uncaptured kernel fails by design; out.txt is
  # written first), then:
  python3 tools/gen_pack.py capture <kernel> <build>/out.txt --date YYYY-MM-DD --elf <core.elf>
  python3 tools/stage_vector_kernel.py <kernel>          # now stages exp.bin = silicon
  python3 tools/gen_pack.py proof <core.elf>             # which pack instructions the ELF holds
"""

import argparse
import os
import random
import re

import capture_kernel_render as render
import gen_agu_2d3d as agu
from gen_vldb_unpack import words_of

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY = os.path.join(REPO, "tests", "vector-verify")
GOLDEN_DIR = os.path.join(REPO, "tools", "golden", "silicon")

BLOCK_BYTES = 64
BLOCK_WORDS = BLOCK_BYTES // 4
BLOCKS_PER_CHUNK = 8
HEADER_WORDS = 32
SIGN_VALUES = (0, 1)
SAT_VALUES = (0, 1, 2, 3)
STEP_BYTES = -32
IDX_OFFSETS = (-64, -32, 0, 32)
M32 = 0xFFFFFFFF

# (name, family, packed input bits, narrowed output bits). Output order of the vectors.
FORMS = [
    ("s8.s16", "s", 16, 8),
    ("d8.d16", "d", 16, 8),
    ("s4.s8", "s", 8, 4),
    ("d4.d8", "d", 8, 4),
]
WORDS_PER_VECTOR = 8
LANE_GROUPS = ["fused", "standalone"]
ADDR_GROUPS = ["fused_idx", "fused_postmod"]
PASSES = [(sat, sign) for sat in SAT_VALUES for sign in SIGN_VALUES]
WORDS_PER_PASS = 2 * len(FORMS) * WORDS_PER_VECTOR
WORDS_PER_BLOCK = len(PASSES) * WORDS_PER_PASS
# The addr kernel appends one record of 8 words per pass: word 0 is the post-modified pointer
# after the four-store chain, relative to the pass start (in bytes), kept live so that every
# store in the chain really is a post-modify store rather than a plain one.
ADDR_WORDS_PER_PASS = WORDS_PER_PASS + WORDS_PER_VECTOR
ADDR_WORDS_PER_BLOCK = len(PASSES) * ADDR_WORDS_PER_PASS
NIN = HEADER_WORDS + BLOCKS_PER_CHUNK * BLOCK_WORDS
NOUT = BLOCKS_PER_CHUNK * WORDS_PER_BLOCK


# ---------------------------------------------------------------------------
# Designed inputs. A block is 64 bytes.
# ---------------------------------------------------------------------------

def bytes_of_int16(lanes):
    """The 64 bytes of 32 int16 lanes, little-endian."""
    assert len(lanes) == 32
    out = []
    for v in lanes:
        out += [v & 0xFF, (v >> 8) & 0xFF]
    return out


RAND_BLOCKS = 256
RAND_SEED = 0x9ACC51


def one_hot_int16():
    """32 blocks: int16 lane l = 0x0051 (81: inside every 8-bit range), the rest zero."""
    return [bytes_of_int16([0x51 if l == i else 0 for l in range(32)]) for i in range(32)]


def one_hot_bytes():
    """64 blocks: byte p = 0x05 (inside both 4-bit ranges), the rest zero."""
    return [[0x05 if p == i else 0 for p in range(BLOCK_BYTES)] for i in range(BLOCK_BYTES)]


def ramps():
    return [
        bytes_of_int16([l + 1 for l in range(32)]),
        bytes_of_int16([0x80 + l for l in range(32)]),
        bytes_of_int16([-1 - l for l in range(32)]),
        bytes_of_int16([0x100 * (l + 1) + (l + 1) for l in range(32)]),
        [p + 1 for p in range(BLOCK_BYTES)],
        [0x80 + p for p in range(BLOCK_BYTES)],
    ]


def clamp_values_16():
    """int16 patterns around every clamp bound of the 16-to-8 narrowing.

    Signed int8 [-128,127] and symmetric [-127,127], unsigned [0,255], each with the values
    just inside, on and just outside the bound; the int16 extremes; and bit patterns that read
    differently as signed and unsigned (high bit set).
    """
    vals = [0, 1, -1, 2, -2, 7, 8, -7, -8, 15, 16, -15, -16]
    vals += list(range(125, 131)) + list(range(-131, -124))
    vals += list(range(253, 259)) + list(range(-259, -252))
    vals += [0x0100, 0x0101, 0x01FF, 0x0200, 0x7F00, 0x7F7F, 0x7F80, 0x7FFE, 0x7FFF]
    vals += [-32768, -32767, -32766, -0x7F01, -0x7F00, -0x100, -0x101, -0x4000, 0x4000]
    vals += [0x8080, 0xFF00, 0xFF7F, 0xFF80, 0xFF81, 0xFFFE, 0xC000, 0x00FF, 0xFE00, 0xFEFF]
    out, seen = [], set()
    for v in vals:
        m = v & 0xFFFF
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def clamp_values_8():
    """int8 patterns around every clamp bound of the 8-to-4 narrowing."""
    vals = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65, 126, 127]
    vals += [-1, -2, -3, -5, -6, -7, -8, -9, -10, -13, -14, -15, -16, -17, -31, -32, -33, -64, -65]
    vals += [-126, -127, -128]
    vals += [0x80, 0x81, 0xF0, 0xF7, 0xF8, 0xF9, 0xFF, 0x7F, 0x88, 0x77, 0x55, 0xAA]
    out, seen = [], set()
    for v in vals:
        m = v & 0xFF
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def rotated(values, lanes, to_block):
    """len(values) blocks in which every value sits at every lane position."""
    n = len(values)
    return [to_block([values[(k + l) % n] for l in range(lanes)]) for k in range(n)]


def boundary_blocks():
    return (rotated(clamp_values_16(), 32, bytes_of_int16)
            + rotated(clamp_values_8(), BLOCK_BYTES, lambda lanes: lanes))


def coverage():
    """256 blocks: byte p of block b is (b + 37 p) mod 256, so every byte value (and so every
    8-bit narrowing input) appears at every byte position."""
    return [[(b + 37 * p) & 0xFF for p in range(BLOCK_BYTES)] for b in range(256)]


def dense_int16():
    """20 blocks that together hold every int16 from -320 to 319, covering both byte clamp
    bounds on both sides densely."""
    vals = list(range(-320, 320))
    return [bytes_of_int16([v & 0xFFFF for v in vals[k * 32:(k + 1) * 32]]) for k in range(len(vals) // 32)]


def fills():
    byte_fills = [0x00, 0xFF, 0x80, 0x7F, 0x88, 0x77, 0xF0, 0x0F, 0x55, 0xAA, 0x01, 0xFE, 0x10, 0xEF]
    blocks = [[v] * BLOCK_BYTES for v in byte_fills]
    for v in (0x8000, 0x7FFF, 0x00FF, 0xFF00):
        blocks.append(bytes_of_int16([v] * 32))
    return blocks


def probe_blocks():
    blocks = (one_hot_int16() + one_hot_bytes() + ramps() + boundary_blocks() + coverage()
              + dense_int16() + fills())
    # Fill the last chunk with extra random blocks (distinct seed from the held-out set).
    rng = random.Random(RAND_SEED ^ 0xFFFF)
    while len(blocks) % BLOCKS_PER_CHUNK:
        blocks.append([rng.randrange(256) for _ in range(BLOCK_BYTES)])
    return blocks




def rand_blocks():
    rng = random.Random(RAND_SEED)
    return [[rng.randrange(256) for _ in range(BLOCK_BYTES)] for _ in range(RAND_BLOCKS)]


ADDR_SEED = 0xADD9AC
ADDR_BLOCKS = 32


def addr_blocks():
    """A small mixed set: boundary blocks, ramps and a few random ones."""
    rng = random.Random(ADDR_SEED)
    chosen = ramps() + boundary_blocks()[::9]
    chosen += [[rng.randrange(256) for _ in range(BLOCK_BYTES)] for _ in range(ADDR_BLOCKS - len(chosen))]
    return chosen[:ADDR_BLOCKS]


LANE_KERNELS = {
    "vec_pack_lane_probe": probe_blocks,
    "vec_pack_lane_rand": rand_blocks,
}
ADDR_KERNEL = "vec_pack_addr"
CRT_KERNEL = "vec_pack_crtiming"


def header_words():
    hdr = [0] * HEADER_WORDS
    hdr[0], hdr[1] = SIGN_VALUES
    hdr[3] = STEP_BYTES & M32
    hdr[4:8] = SAT_VALUES
    hdr[8:12] = [v & M32 for v in IDX_OFFSETS]
    return hdr


def lane_input_words(blocks):
    assert len(blocks) % BLOCKS_PER_CHUNK == 0, len(blocks)
    words = []
    for c in range(0, len(blocks), BLOCKS_PER_CHUNK):
        words += header_words()
        for blk in blocks[c:c + BLOCKS_PER_CHUNK]:
            assert len(blk) == BLOCK_BYTES
            words += words_of(blk)
    return words


# ---------------------------------------------------------------------------
# Kernel source
# ---------------------------------------------------------------------------

def source_common(words_per_block, words_per_pass):
    return f"""#include <stdint.h>

#include <aie_api/aie.hpp>

#define BLOCKS {BLOCKS_PER_CHUNK}
#define HEADER {HEADER_WORDS}
#define BLOCK_BYTES {BLOCK_BYTES}
#define WORDS_PER_BLOCK {words_per_block}
#define WORDS_PER_PASS {words_per_pass}
"""


LANE_SOURCE_COMMON = source_common(WORDS_PER_BLOCK, WORDS_PER_PASS)
ADDR_SOURCE_COMMON = source_common(ADDR_WORDS_PER_BLOCK, ADDR_WORDS_PER_PASS)


def render_lane_kernel():
    return f"""//===- pack_lane.cc ---------------------------------------------*- C++ -*-===//
//
// Pack lane-map capture kernel (GENERATED by tools/gen_pack.py).
// Every block runs through the fused vst.pack forms and the standalone vpack forms, under every
// crSat value and both crPackSign values. The sat and sign values come from the input header
// so the compiler cannot fold them: pack(v, sign) with a run-time sign is the register-driven
// `d` form, pack(v) the `s` form. Each noinline function holds the crSat write and every user.
//
//===----------------------------------------------------------------------===//

{LANE_SOURCE_COMMON}
__attribute__((noinline)) static void fused(int8_t *restrict bp, int sat, int sgn, int32_t *restrict out) {{
  ::aie::set_saturation(static_cast<aie::saturation_mode>(sat));
  v32int16 a = *(v32int16 *)bp;
  v64int8 b = *(v64int8 *)bp;
  *(v32int8 *)(out + 0) = pack(a);
  *(v32int8 *)(out + 8) = pack(a, sgn);
  *(v64int4 *)(out + 16) = pack(b);
  *(v64int4 *)(out + 24) = pack(b, sgn);
}}

// The zero vector z is added to every result, so the pack feeds a vector add instead of a
// store: the compiler keeps a standalone vpack into a w register.
__attribute__((noinline)) static void standalone(int8_t *restrict bp, int32_t *restrict hdr, int sat, int sgn,
                                                 int32_t *restrict out) {{
  ::aie::set_saturation(static_cast<aie::saturation_mode>(sat));
  v32int16 a = *(v32int16 *)bp;
  v64int8 b = *(v64int8 *)bp;
  aie::vector<int8_t, 32> z = aie::load_v<32>((int8_t *)(hdr + 16));
  aie::store_v((int8_t *)(out + 0), aie::add(aie::vector<int8_t, 32>(pack(a)), z));
  aie::store_v((int8_t *)(out + 8), aie::add(aie::vector<int8_t, 32>(pack(a, sgn)), z));
  aie::store_v((int8_t *)(out + 16), aie::add(aie::vector<int8_t, 32>((v32int8)pack(b)), z));
  aie::store_v((int8_t *)(out + 24), aie::add(aie::vector<int8_t, 32>((v32int8)pack(b, sgn)), z));
}}

extern "C" {{

void pack_lane(int32_t *restrict in, int32_t *restrict out) {{
  int8_t *blocks = (int8_t *)(in + HEADER);
  event0();
  for (int b = 0; b < BLOCKS; b++) {{
    for (int k = 0; k < 4; k++) {{
      for (int s = 0; s < 2; s++) {{
        int sat = in[4 + k];
        int sgn = in[s];
        int32_t *o = out + b * WORDS_PER_BLOCK + (k * 2 + s) * WORDS_PER_PASS;
        fused(blocks + b * BLOCK_BYTES, sat, sgn, o);
        standalone(blocks + b * BLOCK_BYTES, in, sat, sgn, o + 32);
      }}
    }}
  }}
  event1();
}}

}} // extern "C"
"""


def render_addr_kernel():
    return f"""//===- pack_addr.cc ---------------------------------------------*- C++ -*-===//
//
// Fused pack-store addressing capture kernel (GENERATED by tools/gen_pack.py).
// The index group stores with run-time offsets (`[p, dj]`) that put each form in its own slot
// of a 4-slot group; the post-modify group stores four times through a run-time negative step
// (`[p], m`) starting at the last slot, so the forms land in reverse order. A wrong offset or a
// wrong post-modify moves the data and shows. The post-modified pointer is stored too (relative
// to the pass), which also keeps the last store of the chain a post-modify store.
//
//===----------------------------------------------------------------------===//

{ADDR_SOURCE_COMMON}
__attribute__((noinline)) static void fused_idx(int8_t *restrict bp, int32_t *restrict hdr, int sat, int sgn,
                                                int32_t *restrict out) {{
  ::aie::set_saturation(static_cast<aie::saturation_mode>(sat));
  v32int16 a = *(v32int16 *)bp;
  v64int8 b = *(v64int8 *)bp;
  int8_t *q = (int8_t *)(out + 16);
  *(v32int8 *)(q + hdr[8]) = pack(a);
  *(v32int8 *)(q + hdr[9]) = pack(a, sgn);
  *(v64int4 *)(q + hdr[10]) = pack(b);
  *(v64int4 *)(q + hdr[11]) = pack(b, sgn);
}}

__attribute__((noinline)) static void fused_postmod(int8_t *restrict bp, int32_t *restrict hdr, int sat, int sgn,
                                                    int32_t *restrict out) {{
  ::aie::set_saturation(static_cast<aie::saturation_mode>(sat));
  v32int16 a = *(v32int16 *)bp;
  v64int8 b = *(v64int8 *)bp;
  int step = hdr[3];
  int8_t *start = (int8_t *)(out + 24);
  int8_t *p = start;
  *(v32int8 *)p = pack(a);
  p += step;
  *(v32int8 *)p = pack(a, sgn);
  p += step;
  *(v64int4 *)p = pack(b);
  p += step;
  *(v64int4 *)p = pack(b, sgn);
  p += step;
  out[32] = (int32_t)(p - (int8_t *)out);
  for (int i = 1; i < 8; i++) out[32 + i] = 0;
}}

extern "C" {{

void pack_addr(int32_t *restrict in, int32_t *restrict out) {{
  int8_t *blocks = (int8_t *)(in + HEADER);
  event0();
  for (int b = 0; b < BLOCKS; b++) {{
    for (int k = 0; k < 4; k++) {{
      for (int s = 0; s < 2; s++) {{
        int sat = in[4 + k];
        int sgn = in[s];
        int32_t *o = out + b * WORDS_PER_BLOCK + (k * 2 + s) * WORDS_PER_PASS;
        fused_idx(blocks + b * BLOCK_BYTES, in, sat, sgn, o);
        fused_postmod(blocks + b * BLOCK_BYTES, in, sat, sgn, o + 32);
      }}
    }}
  }}
  event1();
}}

}} // extern "C"
"""


# ---------------------------------------------------------------------------
# crPackSign timing kernel
# ---------------------------------------------------------------------------

CRT_BLOCKS = 64
CRT_SEED = 0xC9AC41
# fused d8.d16, fused d4.d8, standalone d8.d16, standalone d4.d8
CRT_WORDS_PER_BLOCK = 4 * WORDS_PER_VECTOR
CRT_CONTROL_WORDS = 16   # [1] = crSat, then 8 zero words at [8, 16): the vector added to the standalone results
CRT_NIN = CRT_BLOCKS + CRT_CONTROL_WORDS + CRT_BLOCKS * BLOCK_WORDS
CRT_NOUT = CRT_BLOCKS * CRT_WORDS_PER_BLOCK


def crt_sign_patterns():
    """Per-chunk sign sequences: random, alternating, runs of four, constant 1, constant 0, runs of two."""
    rng = random.Random(CRT_SEED)
    return [
        [rng.randrange(2) for _ in range(CRT_BLOCKS)],
        [b % 2 for b in range(CRT_BLOCKS)],
        [(b // 4) % 2 for b in range(CRT_BLOCKS)],
        [1] * CRT_BLOCKS,
        [0] * CRT_BLOCKS,
        [(b // 2) % 2 for b in range(CRT_BLOCKS)],
    ]


def crt_blocks():
    """Blocks that read differently signed and unsigned in both views: int16 lanes and bytes
    drawn from the clamp-boundary values, plus random bytes."""
    rng = random.Random(CRT_SEED + 1)
    pool16, pool8 = clamp_values_16(), clamp_values_8()
    blocks = []
    for _ in range(len(crt_sign_patterns()) * CRT_BLOCKS):
        if rng.random() < 0.5:
            blocks.append(bytes_of_int16([rng.choice(pool16) for _ in range(32)]))
        elif rng.random() < 0.5:
            blocks.append([rng.choice(pool8) for _ in range(BLOCK_BYTES)])
        else:
            blocks.append([rng.randrange(256) for _ in range(BLOCK_BYTES)])
    return blocks


def crt_input_words():
    blocks = crt_blocks()
    words = []
    for c, signs in enumerate(crt_sign_patterns()):
        words += signs
        words += [0, 1] + [0] * (CRT_CONTROL_WORDS - 2)   # [CRT_BLOCKS + 1] = 1: the saturating crSat
        for blk in blocks[c * CRT_BLOCKS:(c + 1) * CRT_BLOCKS]:
            words += words_of(blk)
    return words


def render_crt_kernel():
    return f"""//===- pack_crt.cc ----------------------------------------------*- C++ -*-===//
//
// crPackSign timing capture kernel (GENERATED by tools/gen_pack.py).
// The sign of each block comes from a run-time table, so the compiler rewrites crPackSign in a
// software-pipelined loop, in the same bundle as or just before the register-driven d form it
// controls, and overlaps neighbouring iterations. crSat is saturate throughout. The result shows
// which sign each instruction actually used on silicon.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define NB {CRT_BLOCKS}

extern "C" {{

void pack_crt(int32_t *restrict in, int32_t *restrict out) {{
  int8_t *blocks = (int8_t *)(in + NB + 16);
  event0();
  ::aie::set_saturation(static_cast<aie::saturation_mode>(in[NB + 1]));
  for (int b = 0; b < NB; b++) chess_prepare_for_pipelining chess_loop_range(NB, NB) {{
    int s = in[b];
    int8_t *bp = blocks + b * {BLOCK_BYTES};
    v32int16 a = *(v32int16 *)bp;
    v64int8 c = *(v64int8 *)bp;
    *(v32int8 *)(out + b * {CRT_WORDS_PER_BLOCK}) = pack(a, s);
    *(v64int4 *)(out + b * {CRT_WORDS_PER_BLOCK} + 8) = pack(c, s);
  }}
  aie::vector<int8_t, 32> zz = aie::load_v<32>((int8_t *)(in + NB + 8));
  for (int b = 0; b < NB; b++) chess_prepare_for_pipelining chess_loop_range(NB, NB) {{
    int s = in[b];
    int8_t *bp = blocks + b * {BLOCK_BYTES};
    v32int16 a = *(v32int16 *)bp;
    v64int8 c = *(v64int8 *)bp;
    aie::store_v((int8_t *)(out + b * {CRT_WORDS_PER_BLOCK} + 16),
                 aie::add(aie::vector<int8_t, 32>(pack(a, s)), zz));
    aie::store_v((int8_t *)(out + b * {CRT_WORDS_PER_BLOCK} + 24),
                 aie::add(aie::vector<int8_t, 32>((v32int8)pack(c, s)), zz));
  }}
  event1();
}}

}} // extern "C"
"""


# ---------------------------------------------------------------------------
# Generate / capture
# ---------------------------------------------------------------------------

def golden_path(kernel):
    return os.path.join(GOLDEN_DIR, kernel + ".json")


def kernel_names():
    return list(LANE_KERNELS) + [ADDR_KERNEL, CRT_KERNEL]


def blocks_of(kernel):
    if kernel == ADDR_KERNEL:
        return addr_blocks()
    return LANE_KERNELS[kernel]()


def input_words(kernel):
    if kernel == CRT_KERNEL:
        return crt_input_words()
    return lane_input_words(blocks_of(kernel))


def n_chunks(kernel):
    if kernel == CRT_KERNEL:
        return len(crt_sign_patterns())
    return len(blocks_of(kernel)) // BLOCKS_PER_CHUNK


def words_per_block(kernel):
    return ADDR_WORDS_PER_BLOCK if kernel == ADDR_KERNEL else WORDS_PER_BLOCK


def n_out(kernel):
    if kernel == CRT_KERNEL:
        return n_chunks(kernel) * CRT_NOUT
    return len(blocks_of(kernel)) * words_per_block(kernel)


def kernel_data(kernel):
    """Raw input and expected buffers (see vv_data.KernelData) for staging."""
    in_words = input_words(kernel)
    golden = render.load_golden(golden_path(kernel))
    if golden is not None and golden["inputs"] != [in_words]:
        raise SystemExit(f"{kernel}: golden inputs are stale; recapture on silicon")
    expected = None if golden is None else golden["silicon"]
    assert expected is None or len(expected) == n_out(kernel)
    return render.kernel_data(in_words, expected)


def generate_one(kernel):
    kernel_data(kernel)   # fails on a stale golden before anything is written
    if kernel == CRT_KERNEL:
        files = {
            "pack_crt.cc": render_crt_kernel(),
            "aie.mlir": render.render_mlir(
                kernel, "pack_crt", n_chunks(kernel), CRT_NIN, CRT_NOUT, "i32", "i32",
                f"{CRT_BLOCKS} sign words, {CRT_CONTROL_WORDS} control words, then {CRT_BLOCKS} blocks per chunk; "
                f"{CRT_WORDS_PER_BLOCK} result words per block."),
            "run.lit": render.render_run_lit("pack_crt.cc", n_out(kernel)),
        }
    else:
        stem, src = (("pack_addr", render_addr_kernel()) if kernel == ADDR_KERNEL
                     else ("pack_lane", render_lane_kernel()))
        files = {
            f"{stem}.cc": src,
            "aie.mlir": render.render_mlir(
                kernel, stem, n_chunks(kernel), NIN, BLOCKS_PER_CHUNK * words_per_block(kernel), "i32", "i32",
                f"{BLOCKS_PER_CHUNK} blocks of {BLOCK_BYTES} bytes per chunk; "
                f"{words_per_block(kernel)} result words per block."),
            "run.lit": render.render_run_lit(f"{stem}.cc", n_out(kernel)),
        }
    render.write_files(os.path.join(VERIFY, kernel), files)
    return n_chunks(kernel)


def vector_table(kernel):
    """The per-vector description stored in the golden, in output order within one block.

    `slot` is the 8-word position of the vector inside its pass. The post-modify group of the
    addr kernel starts at its last slot and steps backwards, so its forms are in reverse order.
    """
    groups = ADDR_GROUPS if kernel == ADDR_KERNEL else LANE_GROUPS
    table = []
    for sat, sign in PASSES:
        for gi, group in enumerate(groups):
            order = FORMS[::-1] if group == "fused_postmod" else FORMS
            for si, (name, family, bits_in, bits_out) in enumerate(order):
                table.append({"crSat": sat, "crPackSign": sign, "group": group, "form": name,
                              "family": family, "bits_in": bits_in, "bits_out": bits_out,
                              "slot": gi * len(FORMS) + si})
    return table


# -- disassembly proof ------------------------------------------------------

PACK_MNEMONIC = re.compile(r"^(?:vpack\.\S+|vst(?:\.[23]d)?\.pack\.\S+)$")


def addressing_class(slot):
    """`[p1, #0x40]` idx_imm, `[p1, dj0]` idx, `[p1], #0x20` pstm_imm, `[p1], m0` pstm; `reg` for vpack."""
    m = re.search(r"\[(\w+)(,\s*([^\]]+))?\](,\s*(\S+))?", slot)
    if not m:
        return "reg"
    inside, post = m.group(3), m.group(5)
    if post is not None:
        return "pstm_imm" if post.startswith("#") else "pstm"
    if inside is None:
        return "idx_imm"
    return "idx_imm" if inside.strip().startswith("#") else "idx"


def proof(elf):
    """Per (mnemonic, addressing): the first bundle containing it (bytes and text) and a count."""
    found = {}
    listing, skipped = agu.disassemble(elf)
    if skipped:
        print(f"note: llvm-objdump cannot print the bundles at {[hex(a) for a in skipped]}; skipped")
    for raw in listing.splitlines():
        m = agu.LINE.match(raw)
        if not m:
            continue
        addr, hexbytes, text = m.groups()
        for slot in (s.strip() for s in text.split(";")):
            head = slot.split()[0] if slot.split() else ""
            if not PACK_MNEMONIC.match(head):
                continue
            key = (head, addressing_class(slot))
            found.setdefault(key, {"count": 0, "bytes": " ".join(hexbytes.split()), "bundle": text.strip()})
            found[key]["count"] += 1
    return found


def description(kernel):
    if kernel == CRT_KERNEL:
        return {"blocks": CRT_BLOCKS, "chunks": n_chunks(kernel), "sign_words": CRT_BLOCKS,
                "control_words": CRT_CONTROL_WORDS, "words_per_block": CRT_WORDS_PER_BLOCK,
                "vectors": ["fused d8.d16", "fused d4.d8", "standalone d8.d16", "standalone d4.d8"],
                "signs": crt_sign_patterns()}
    return {"blocks": len(blocks_of(kernel)), "block_bytes": BLOCK_BYTES,
            "words_per_vector": WORDS_PER_VECTOR, "words_per_block": words_per_block(kernel),
            "words_per_pass": ADDR_WORDS_PER_PASS if kernel == ADDR_KERNEL else WORDS_PER_PASS,
            "blocks_per_chunk": BLOCKS_PER_CHUNK, "header_words": HEADER_WORDS,
            "header": header_words(), "step_bytes": STEP_BYTES, "idx_offsets": list(IDX_OFFSETS)}


def capture(kernel, out_txt, date, elf=None):
    silicon = [v & M32 for v in render.read_out_txt(out_txt)]
    assert len(silicon) == n_out(kernel), f"{len(silicon)} != {n_out(kernel)}"
    rec = {"kernel": kernel, "n": len(silicon)}
    rec["class"] = {ADDR_KERNEL: "pack_addr", CRT_KERNEL: "pack_crtiming"}.get(kernel, "pack_lane")
    rec["layout"] = description(kernel)
    if kernel != CRT_KERNEL:
        rec["vectors"] = vector_table(kernel)
    if elf:
        rec["encodings"] = [
            {"mnemonic": mn, "addressing": ad, "count": e["count"], "bytes": e["bytes"], "bundle": e["bundle"]}
            for (mn, ad), e in sorted(proof(elf).items())
        ]
    rec["inputs"] = [input_words(kernel)]
    rec["silicon"] = silicon
    rec["provenance"] = f"HW-observed: NPU1 Phoenix silicon, captured {date} via tools/gen_pack.py"
    return render.write_golden(golden_path(kernel), rec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    cap = sub.add_parser("capture")
    cap.add_argument("kernel")
    cap.add_argument("out_txt")
    cap.add_argument("--date", required=True)
    cap.add_argument("--elf")
    prf = sub.add_parser("proof")
    prf.add_argument("elf")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        for k in kernel_names():
            print(f"wrote {k} ({generate_one(k)} chunks)")
    elif args.cmd == "capture":
        if args.kernel not in kernel_names():
            raise SystemExit(f"unknown kernel {args.kernel}")
        print("wrote", capture(args.kernel, args.out_txt, args.date, args.elf))
    else:
        for (mn, ad), e in sorted(proof(args.elf).items()):
            print(f"{mn:28s} {ad:9s} x{e['count']:<3d} {e['bytes']:34s} {e['bundle']}")


if __name__ == "__main__":
    main()
