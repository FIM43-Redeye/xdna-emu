#!/usr/bin/env python3
"""Generate the capture kernels for the AIE2 unpack instructions.

Why this exists
---------------
AIE2 widens packed integers two ways: the fused load `vldb.unpack.*` (a 256-bit
memory load unpacked into a 512-bit x register) and the standalone `vunpack.*`
(a 256-bit w register unpacked into a 512-bit x register). Each has four forms:
`s8.s4`, `s16.s8` (sign fixed to signed in the encoding) and `d8.d4`, `d16.d8`
(signedness read from control register `crUnpackSign` at run time). The lane
order of the 512-bit result, the signedness rule, and what the post-modify of
the fused load adds to the pointer are not stated anywhere we may read, so they
are captured from NPU1 silicon.

Kernels
-------
  vec_unpack_lane_probe   designed blocks that read the lane map off silicon: one-hot
                          nibbles, one-hot bytes, ramps, and 256 blocks in which every
                          byte value (and so every nibble value) sits at every byte
                          position.
  vec_unpack_lane_rand    a seeded random held-out set; never used to derive the map.
  vec_unpack_crtiming     the sign comes from a run-time table inside a software-pipelined
                          loop, where the compiler writes crUnpackSign AFTER the d-form it
                          controls: shows the register is read late (pipeline stage 7).
  vec_unpack_addr         fused-load addressing: runtime index offsets, post-modify
                          register values, and 2D / 3D descriptor sequences, with the
                          pointer after each access recorded.

The lane kernels run every block through all four forms of both families, once with
`crUnpackSign` = 0 and once with 1. The `s` forms are included in both passes so the
output shows whether the register is ignored there.

Lane kernel layout (32-bit words)
---------------------------------
Chunk input: 16 header words, then BLOCKS_PER_CHUNK blocks of 8 words (32 bytes).
  header[0] = 0, header[1] = 1   the two crUnpackSign values, read at run time so the
                                 compiler must emit the register-driven `d` forms
  header[2] = 0                  index-form byte offset (runtime, so `[p, dj]` is used)
  header[3] = 0                  post-modify register value, bytes (zero: the lane
                                 kernels leave the pointer alone; vec_unpack_addr sweeps it)
Per block, per sign pass: 12 vectors of 16 words:
  fused index  : s16.s8  d16.d8  s8.s4  d8.d4
  fused postmod: s16.s8  d16.d8  s8.s4  d8.d4
  standalone   : s16.s8  d16.d8  s8.s4  d8.d4
The group names are the source pattern; the compiler may pick another addressing mode for a
form (the 4-bit "postmod" loads come out as index loads), so the disassembly of the kernel,
not the group name, says which instruction ran. vec_unpack_addr covers every form in every
addressing mode and checks the data against the same lane map.
Output per block: sign pass 0 then sign pass 1, 384 words.

Usage
-----
  python3 tools/gen_vldb_unpack.py generate
  python3 tools/stage_vector_kernel.py <kernel>          # no exp.bin yet
  # run the bridge on HW once (an uncaptured kernel fails by design; out.txt is
  # written first), then:
  python3 tools/gen_vldb_unpack.py capture <kernel> <build>/out.txt --date YYYY-MM-DD
  python3 tools/stage_vector_kernel.py <kernel>          # now stages exp.bin = silicon
"""

import argparse
import os
import random

import capture_kernel_render as render

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY = os.path.join(REPO, "tests", "vector-verify")
GOLDEN_DIR = os.path.join(REPO, "tools", "golden", "silicon")

BLOCK_BYTES = 32
BLOCKS_PER_CHUNK = 16
HEADER_WORDS = 16
SIGN_PASSES = (0, 1)
M32 = 0xFFFFFFFF

# (name, family, sign source in the encoding, packed bits, widened bits). The order is the
# order of the vectors in the output.
FORMS = [
    ("s16.s8", "s", 8, 16),
    ("d16.d8", "d", 8, 16),
    ("s8.s4", "s", 4, 8),
    ("d8.d4", "d", 4, 8),
]
GROUPS = ["fused_idx", "fused_postmod", "standalone"]
VECTORS_PER_PASS = len(GROUPS) * len(FORMS)
WORDS_PER_VECTOR = 16
WORDS_PER_BLOCK = len(SIGN_PASSES) * VECTORS_PER_PASS * WORDS_PER_VECTOR
NIN = HEADER_WORDS + BLOCKS_PER_CHUNK * BLOCK_BYTES // 4
NOUT = BLOCKS_PER_CHUNK * WORDS_PER_BLOCK


# ---------------------------------------------------------------------------
# Designed inputs
# ---------------------------------------------------------------------------

def nibble_one_hot():
    """64 blocks: nibble q = 0x9 (top bit set, so sign handling shows), the rest zero."""
    blocks = []
    for q in range(64):
        b = [0] * BLOCK_BYTES
        b[q // 2] = 0x9 << (4 * (q % 2))
        blocks.append(b)
    return blocks


def byte_one_hot():
    """32 blocks: byte p = 0xC3 (negative as int8; distinct nibbles), the rest zero."""
    blocks = []
    for p in range(BLOCK_BYTES):
        b = [0] * BLOCK_BYTES
        b[p] = 0xC3
        blocks.append(b)
    return blocks


def ramps():
    return [[p + 1 for p in range(BLOCK_BYTES)], [0x80 + p for p in range(BLOCK_BYTES)]]


def coverage():
    """256 blocks: byte p of block b is (b + 37 p) mod 256, so every byte value, and so
    every nibble value in both nibble positions, appears at every byte position."""
    return [[(b + 37 * p) & 0xFF for p in range(BLOCK_BYTES)] for b in range(256)]


def extremes():
    pats = [0x00, 0xFF, 0x80, 0x7F, 0x88, 0x77, 0xF0, 0x0F, 0x55, 0xAA, 0x01, 0xFE, 0x10, 0xEF]
    return [[v] * BLOCK_BYTES for v in pats]


def probe_blocks():
    return nibble_one_hot() + byte_one_hot() + ramps() + coverage() + extremes()


RAND_BLOCKS = 256
RAND_SEED = 0x5E7A11


def rand_blocks():
    rng = random.Random(RAND_SEED)
    return [[rng.randrange(256) for _ in range(BLOCK_BYTES)] for _ in range(RAND_BLOCKS)]


LANE_KERNELS = {
    "vec_unpack_lane_probe": probe_blocks,
    "vec_unpack_lane_rand": rand_blocks,
}


def words_of(byte_list):
    return [byte_list[i] | byte_list[i + 1] << 8 | byte_list[i + 2] << 16 | byte_list[i + 3] << 24
            for i in range(0, len(byte_list), 4)]


def lane_input_words(blocks):
    assert len(blocks) % BLOCKS_PER_CHUNK == 0, len(blocks)
    words = []
    for c in range(0, len(blocks), BLOCKS_PER_CHUNK):
        header = [SIGN_PASSES[0], SIGN_PASSES[1], 0, 0] + [0] * (HEADER_WORDS - 4)
        words += header
        for blk in blocks[c:c + BLOCKS_PER_CHUNK]:
            words += words_of(blk)
    return words


# ---------------------------------------------------------------------------
# Kernel source
# ---------------------------------------------------------------------------

def render_lane_kernel():
    return f"""//===- unpack_lane.cc -------------------------------------------*- C++ -*-===//
//
// Unpack lane-map capture kernel (GENERATED by tools/gen_vldb_unpack.py).
// Every block runs through the fused vldb.unpack forms (index and post-modify
// addressing) and the standalone vunpack forms, with crUnpackSign 0 and 1. The sign
// values and the offsets come from the input header so the compiler cannot fold them:
// unpack(v, sign) with a run-time sign is the register-driven `d` form.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define BLOCKS {BLOCKS_PER_CHUNK}
#define HEADER {HEADER_WORDS}
#define WORDS_PER_PASS {VECTORS_PER_PASS * WORDS_PER_VECTOR}

// noinline keeps the crUnpackSign write and its users in one scheduling region per call.
__attribute__((noinline)) static void fused(int8_t *restrict bp, int32_t *restrict hdr, bool sgn,
                                            int32_t *restrict out) {{
  int off = hdr[2];
  int stp = hdr[3];
  *(v32int16 *)(out + 0) = unpack(*(v32int8 *)(bp + off));
  *(v32int16 *)(out + 16) = unpack(*(v32int8 *)(bp + off), sgn);
  *(v64int8 *)(out + 32) = unpack(*(v64int4 *)(bp + off));
  *(v64int8 *)(out + 48) = unpack(*(v64int4 *)(bp + off), sgn);
  int8_t *p = bp;
  *(v32int16 *)(out + 64) = unpack(*(v32int8 *)p);
  p += stp;
  *(v32int16 *)(out + 80) = unpack(*(v32int8 *)p, sgn);
  p += stp;
  *(v64int8 *)(out + 96) = unpack(*(v64int4 *)p);
  p += stp;
  *(v64int8 *)(out + 112) = unpack(*(v64int4 *)p, sgn);
}}

// The source is also stored, so the load cannot fuse with the unpack: the compiler keeps
// a plain vldb into a w register and a standalone vunpack.
__attribute__((noinline)) static void standalone(int8_t *restrict bp, bool sgn, int32_t *restrict out,
                                                 int32_t *restrict scratch) {{
  v32int8 a = *(v32int8 *)bp;
  v64int4 c = *(v64int4 *)bp;
  *(v32int8 *)scratch = a;
  *(v64int4 *)(scratch + 8) = c;
  *(v32int16 *)(out + 0) = unpack(a);
  *(v32int16 *)(out + 16) = unpack(a, sgn);
  *(v64int8 *)(out + 32) = unpack(c);
  *(v64int8 *)(out + 48) = unpack(c, sgn);
}}

extern "C" {{

void unpack_lane(int32_t *restrict in, int32_t *restrict out) {{
  int8_t *blocks = (int8_t *)(in + HEADER);
  int32_t scratch[16];
  event0();
  for (int b = 0; b < BLOCKS; b++) {{
    for (int s = 0; s < 2; s++) {{
      bool sgn = in[s];
      int32_t *o = out + b * 2 * WORDS_PER_PASS + s * WORDS_PER_PASS;
      fused(blocks + b * {BLOCK_BYTES}, in, sgn, o);
      standalone(blocks + b * {BLOCK_BYTES}, sgn, o + 128, scratch);
    }}
  }}
  event1();
}}

}} // extern "C"
"""



# ---------------------------------------------------------------------------
# Addressing kernel: index offsets, post-modify values, 2D and 3D descriptors
# ---------------------------------------------------------------------------

ADDR_KERNEL = "vec_unpack_addr"
REGION_BYTES = 2048          # random data the loads walk over
REGION_BASE = REGION_BYTES // 2   # the loads start in the middle so offsets can be negative
# 24 words = 96 bytes keeps every record's 64-byte vector store 32-byte aligned (an 80-byte
# record halves the alignment and the vector store silently misplaces its data).
REC_WORDS = 24               # data(16), pointer after, count 1, count 2, base address, 4 zero pad
ITER_2D = 10
ITER_3D = 14
ADDR_SEED = 0xADD2E55

# Index form: byte offset added to the base, no pointer update.
IDX_OFFSETS = [0, 32, 64, 96, 128, 256, 512, -32, -64, -128, -256, -512]
# Post-modify register values, bytes. The load reads at the base; the pointer afterwards is
# base + value.
POSTMOD_VALUES = [0, 32, 64, 96, 128, 256, 512, 1024, -32, -64, -128, -512]
# 2D sets: (off, size1, inc1, initial count1) of add_2d_ptr(p, off, size1, count1, inc1).
SETS_2D = [(96, 4, 32, 0), (64, 2, 32, 0), (0, 3, 64, 0), (-64, 5, 32, 1),
           (32, 1, 32, 0), (128, 4, -32, 2), (32, 6, 32, 3), (-32, 3, 96, 0),
           (32, 2, 32, 5), (64, 3, 32, 7)]   # the last two start with count1 > size1
# 3D sets: (off, size1, inc1, size2, inc2, initial count1, initial count2).
SETS_3D = [(32, 3, 32, 2, 160, 0, 0), (0, 2, 64, 3, 32, 0, 0), (-32, 4, 32, 2, 96, 1, 0),
           (64, 3, 32, 3, 128, 2, 1), (32, 1, 32, 2, 64, 0, 0), (96, 2, 32, 2, 32, 0, 1),
           (32, 2, 32, 2, 64, 0, 4), (32, 2, 32, 2, 64, 3, 0)]   # counts beyond their sizes
ADDR_FORMS = [f[0] for f in FORMS]
ADDR_HDR_WORDS = 16
ADDR_TABLE_WORDS = (len(IDX_OFFSETS) + len(POSTMOD_VALUES) + 4 * len(SETS_2D) + 7 * len(SETS_3D))
ADDR_REGION_WORD = (ADDR_HDR_WORDS + ADDR_TABLE_WORDS + 7) // 8 * 8   # 32-byte aligned
ADDR_NIN = ADDR_REGION_WORD + REGION_BYTES // 4
ADDR_RECORDS = len(IDX_OFFSETS) + len(POSTMOD_VALUES) + len(SETS_2D) * ITER_2D + len(SETS_3D) * ITER_3D
ADDR_NOUT = ADDR_RECORDS * REC_WORDS
# One chunk per (crUnpackSign, form); the header carries both.
ADDR_CHUNKS = [(sign, form) for sign in SIGN_PASSES for form in range(len(FORMS))]


def addr_region():
    rng = random.Random(ADDR_SEED)
    return [rng.randrange(256) for _ in range(REGION_BYTES)]


def addr_input_words():
    region = words_of(addr_region())
    words = []
    for sign, form in ADDR_CHUNKS:
        chunk = [sign, form] + [0] * (ADDR_HDR_WORDS - 2)
        chunk += [v & M32 for v in IDX_OFFSETS]
        chunk += [v & M32 for v in POSTMOD_VALUES]
        for st in SETS_2D:
            chunk += [v & M32 for v in st]
        for st in SETS_3D:
            chunk += [v & M32 for v in st]
        chunk += [0] * (ADDR_REGION_WORD - len(chunk))
        chunk += region
        assert len(chunk) == ADDR_NIN
        words += chunk
    return words


# Per form: how the vector is loaded and stored (the C++ type of the result).
FORM_CODE = {
    "s16.s8": ("unpack(*(v32int8 *)({p}))", "v32int16"),
    "d16.d8": ("unpack(*(v32int8 *)({p}), sgn)", "v32int16"),
    "s8.s4": ("unpack(*(v64int4 *)({p}))", "v64int8"),
    "d8.d4": ("unpack(*(v64int4 *)({p}), sgn)", "v64int8"),
}


def render_addr_form(index, name):
    load_t, vt = FORM_CODE[name]
    fn = f"form{index}"

    def load(p):
        return load_t.format(p=p)

    return f"""
// {name}
__attribute__((noinline)) static void {fn}(int8_t *restrict bp, int32_t *restrict tab, bool sgn,
                                           int32_t *restrict out) {{
  int32_t *idx = tab;
  int32_t *pst = idx + {len(IDX_OFFSETS)};
  int32_t *t2 = pst + {len(POSTMOD_VALUES)};
  int32_t *t3 = t2 + {4 * len(SETS_2D)};
  const int32_t base = (int32_t)(uintptr_t)bp;
  for (int i = 0; i < {len(IDX_OFFSETS)}; i++) {{
    int off = idx[i];
    *({vt} *)(out) = {load("bp + off")};
    out[16] = base + off; out[17] = 0; out[18] = 0; out[19] = base;
    out[20] = out[21] = out[22] = out[23] = 0;
    out += REC;
  }}
  for (int i = 0; i < {len(POSTMOD_VALUES)}; i++) {{
    int8_t *p = bp;
    int step = pst[i];
    *({vt} *)(out) = {load("p")};
    p += step;
    out[16] = (int32_t)(uintptr_t)p; out[17] = 0; out[18] = 0; out[19] = base;
    out[20] = out[21] = out[22] = out[23] = 0;
    out += REC;
  }}
  for (int s = 0; s < {len(SETS_2D)}; s++) {{
    int8_t *p = bp;
    int off = t2[4 * s], size1 = t2[4 * s + 1], inc1 = t2[4 * s + 2];
    addr_t c1 = t2[4 * s + 3];
    for (int i = 0; i < {ITER_2D}; i++) {{
      *({vt} *)(out) = {load("p")};
      p = add_2d_ptr(p, off, size1, c1, inc1);
      out[16] = (int32_t)(uintptr_t)p; out[17] = c1; out[18] = 0; out[19] = base;
    out[20] = out[21] = out[22] = out[23] = 0;
      out += REC;
    }}
  }}
  for (int s = 0; s < {len(SETS_3D)}; s++) {{
    int8_t *p = bp;
    int off = t3[7 * s], size1 = t3[7 * s + 1], inc1 = t3[7 * s + 2];
    int size2 = t3[7 * s + 3], inc2 = t3[7 * s + 4];
    addr_t c1 = t3[7 * s + 5], c2 = t3[7 * s + 6];
    for (int i = 0; i < {ITER_3D}; i++) {{
      *({vt} *)(out) = {load("p")};
      p = add_3d_ptr(p, off, size1, c1, inc1, size2, c2, inc2);
      out[16] = (int32_t)(uintptr_t)p; out[17] = c1; out[18] = c2; out[19] = base;
    out[20] = out[21] = out[22] = out[23] = 0;
      out += REC;
    }}
  }}
}}
"""


def render_addr_kernel():
    forms = "".join(render_addr_form(i, name) for i, (name, *_rest) in enumerate(FORMS))
    return f"""//===- unpack_addr.cc -------------------------------------------*- C++ -*-===//
//
// Fused unpack-load addressing capture kernel (GENERATED by tools/gen_vldb_unpack.py).
// Each chunk runs ONE vldb.unpack form (header[1]) with crUnpackSign = header[0]: index
// offsets, post-modify register values, then 2D and 3D descriptor sequences. After each
// load the pointer (and the descriptor counts) are stored, so the post-modify the
// hardware applied is read from silicon, not from the compiler's model.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define REC {REC_WORDS}
#define HEADER {ADDR_HDR_WORDS}
#define REGION_WORD {ADDR_REGION_WORD}
{forms}
extern "C" {{

void unpack_addr(int32_t *restrict in, int32_t *restrict out) {{
  int8_t *bp = (int8_t *)(in + REGION_WORD) + {REGION_BASE};
  bool sgn = in[0];
  int form = in[1];
  event0();
  if (form == 0)
    form0(bp, in + HEADER, sgn, out);
  else if (form == 1)
    form1(bp, in + HEADER, sgn, out);
  else if (form == 2)
    form2(bp, in + HEADER, sgn, out);
  else
    form3(bp, in + HEADER, sgn, out);
  event1();
}}

}} // extern "C"
"""



# ---------------------------------------------------------------------------
# crUnpackSign timing kernel
# ---------------------------------------------------------------------------

CRT_KERNEL = "vec_unpack_crtiming"
CRT_BLOCKS = 64
CRT_SEED = 0xC27711
# fused d16.d8, fused d8.d4, standalone d16.d8, standalone d8.d4, then the two raw source
# vectors the standalone forms read (stored to out so the compiler cannot fuse their loads)
CRT_WORDS_PER_BLOCK = 80
CRT_NIN = CRT_BLOCKS + CRT_BLOCKS * BLOCK_BYTES // 4
CRT_NOUT = CRT_BLOCKS * CRT_WORDS_PER_BLOCK


def crt_sign_patterns():
    """Per-chunk sign sequences: random, alternating, runs of four, constant 1, constant 0."""
    rng = random.Random(CRT_SEED)
    return [
        [rng.randrange(2) for _ in range(CRT_BLOCKS)],
        [b % 2 for b in range(CRT_BLOCKS)],
        [(b // 4) % 2 for b in range(CRT_BLOCKS)],
        [1] * CRT_BLOCKS,
        [0] * CRT_BLOCKS,
        [(b // 2) % 2 for b in range(CRT_BLOCKS)],
    ]


def crt_input_words():
    rng = random.Random(CRT_SEED + 1)
    words = []
    for signs in crt_sign_patterns():
        words += signs
        for _ in range(CRT_BLOCKS):
            words += words_of([rng.randrange(256) for _ in range(BLOCK_BYTES)])
    return words


def render_crt_kernel():
    return f"""//===- unpack_crt.cc --------------------------------------------*- C++ -*-===//
//
// crUnpackSign timing capture kernel (GENERATED by tools/gen_vldb_unpack.py).
// The sign of each block comes from a run-time table, so the compiler writes
// crUnpackSign inside a software-pipelined loop, and schedules that write AFTER the
// register-driven vldb.unpack.d* it controls (the register is read at pipeline stage 7).
// The result shows which sign each instruction actually used on silicon.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define NB {CRT_BLOCKS}

extern "C" {{

void unpack_crt(int32_t *restrict in, int32_t *restrict out) {{
  int8_t *blocks = (int8_t *)(in + NB);
  event0();
  for (int b = 0; b < NB; b++) chess_prepare_for_pipelining chess_loop_range(NB, NB) {{
    bool s = in[b];
    int8_t *bp = blocks + b * {BLOCK_BYTES};
    v32int16 a = unpack(*(v32int8 *)bp, s);
    v64int8 c = unpack(*(v64int4 *)bp, s);
    *(v32int16 *)(out + b * {CRT_WORDS_PER_BLOCK}) = a;
    *(v64int8 *)(out + b * {CRT_WORDS_PER_BLOCK} + 16) = c;
  }}
  for (int b = 0; b < NB; b++) chess_prepare_for_pipelining chess_loop_range(NB, NB) {{
    bool s = in[b];
    int8_t *bp = blocks + b * {BLOCK_BYTES};
    v32int8 x = *(v32int8 *)bp;
    v64int4 y = *(v64int4 *)bp;
    *(v32int8 *)(out + b * {CRT_WORDS_PER_BLOCK} + 64) = x;
    *(v64int4 *)(out + b * {CRT_WORDS_PER_BLOCK} + 72) = y;
    *(v32int16 *)(out + b * {CRT_WORDS_PER_BLOCK} + 32) = unpack(x, s);
    *(v64int8 *)(out + b * {CRT_WORDS_PER_BLOCK} + 48) = unpack(y, s);
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
    return LANE_KERNELS[kernel]()


def input_words(kernel):
    if kernel == ADDR_KERNEL:
        return addr_input_words()
    if kernel == CRT_KERNEL:
        return crt_input_words()
    return lane_input_words(blocks_of(kernel))


def n_chunks(kernel):
    if kernel == ADDR_KERNEL:
        return len(ADDR_CHUNKS)
    if kernel == CRT_KERNEL:
        return len(crt_sign_patterns())
    return len(blocks_of(kernel)) // BLOCKS_PER_CHUNK


def n_out(kernel):
    if kernel == ADDR_KERNEL:
        return n_chunks(kernel) * ADDR_NOUT
    if kernel == CRT_KERNEL:
        return n_chunks(kernel) * CRT_NOUT
    return len(blocks_of(kernel)) * WORDS_PER_BLOCK


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
    if kernel == ADDR_KERNEL:
        files = {
            "unpack_addr.cc": render_addr_kernel(),
            "aie.mlir": render.render_mlir(
                kernel, "unpack_addr", n_chunks(kernel), ADDR_NIN, ADDR_NOUT, "i32", "i32",
                f"one chunk per (crUnpackSign, form); {ADDR_RECORDS} records of {REC_WORDS} words."),
            "run.lit": render.render_run_lit("unpack_addr.cc", n_out(kernel)),
        }
    elif kernel == CRT_KERNEL:
        files = {
            "unpack_crt.cc": render_crt_kernel(),
            "aie.mlir": render.render_mlir(
                kernel, "unpack_crt", n_chunks(kernel), CRT_NIN, CRT_NOUT, "i32", "i32",
                f"{CRT_BLOCKS} sign words then {CRT_BLOCKS} blocks per chunk; {CRT_WORDS_PER_BLOCK} result words per block."),
            "run.lit": render.render_run_lit("unpack_crt.cc", n_out(kernel)),
        }
    else:
        files = {
            "unpack_lane.cc": render_lane_kernel(),
            "aie.mlir": render.render_mlir(
                kernel, "unpack_lane", n_chunks(kernel), NIN, NOUT, "i32", "i32",
                f"{BLOCKS_PER_CHUNK} blocks of 32 bytes per chunk; {WORDS_PER_BLOCK} result words per block."),
            "run.lit": render.render_run_lit("unpack_lane.cc", n_out(kernel)),
        }
    render.write_files(os.path.join(VERIFY, kernel), files)
    return n_chunks(kernel)


def form_table():
    """The per-vector description stored in the golden, in output order."""
    table = []
    for sign in SIGN_PASSES:
        for group in GROUPS:
            for name, family, bits_in, bits_out in FORMS:
                table.append({"crUnpackSign": sign, "group": group, "form": name,
                              "family": family, "bits_in": bits_in, "bits_out": bits_out})
    return table


def addr_description():
    return {
        "region_bytes": REGION_BYTES, "region_base": REGION_BASE, "record_words": REC_WORDS,
        "record_fields": ["data[16]", "pointer_after", "count1", "count2", "base_address", "zero[4]"],
        "idx_offsets": IDX_OFFSETS, "postmod_values": POSTMOD_VALUES,
        "sets_2d": [list(t) for t in SETS_2D], "iterations_2d": ITER_2D,
        "sets_3d": [list(t) for t in SETS_3D], "iterations_3d": ITER_3D,
        "chunks": [{"crUnpackSign": sg, "form": FORMS[f][0]} for sg, f in ADDR_CHUNKS],
        "region": addr_region(),
    }


def capture(kernel, out_txt, date):
    silicon = [v & M32 for v in render.read_out_txt(out_txt)]
    assert len(silicon) == n_out(kernel), f"{len(silicon)} != {n_out(kernel)}"
    rec = {"kernel": kernel, "n": len(silicon)}
    if kernel == ADDR_KERNEL:
        rec["class"] = "unpack_addr"
        rec["layout"] = addr_description()
    elif kernel == CRT_KERNEL:
        rec["class"] = "unpack_crtiming"
        rec["layout"] = {"blocks": CRT_BLOCKS, "chunks": n_chunks(kernel), "sign_words": CRT_BLOCKS,
                         "words_per_block": CRT_WORDS_PER_BLOCK,
                         "vectors": ["fused d16.d8", "fused d8.d4", "standalone d16.d8", "standalone d8.d4",
                                     "raw x (8 words)", "raw y (8 words)"]}
    else:
        rec["class"] = "unpack_lane"
        rec["layout"] = {"blocks": len(blocks_of(kernel)), "block_bytes": BLOCK_BYTES,
                         "words_per_vector": WORDS_PER_VECTOR, "words_per_block": WORDS_PER_BLOCK,
                         "blocks_per_chunk": BLOCKS_PER_CHUNK, "header_words": HEADER_WORDS}
        rec["vectors"] = form_table()
    rec["inputs"] = [input_words(kernel)]
    rec["silicon"] = silicon
    rec["provenance"] = f"HW-observed: NPU1 Phoenix silicon, captured {date} via tools/gen_vldb_unpack.py"
    return render.write_golden(golden_path(kernel), rec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    cap = sub.add_parser("capture")
    cap.add_argument("kernel")
    cap.add_argument("out_txt")
    cap.add_argument("--date", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        for k in kernel_names():
            print(f"wrote {k} ({generate_one(k)} chunks)")
    else:
        if args.kernel not in kernel_names():
            raise SystemExit(f"unknown kernel {args.kernel}")
        print("wrote", capture(args.kernel, args.out_txt, args.date))


if __name__ == "__main__":
    main()
