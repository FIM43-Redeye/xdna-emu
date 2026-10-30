#!/usr/bin/env python3
"""Generate the silicon capture kernels for the AIE2 2D / 3D addressing modes.

Why this exists
---------------
AIE2 memory instructions come in a post-modify form `[p], d` where `d` names a
descriptor (the registers m, dn, dj, dc of the same index; a 3D access names a pair
of descriptors). The pointer and the counts are updated after the access. The rule
was pinned on NPU1 silicon through the fused `vldb.unpack.*` loads
(`tools/gen_vldb_unpack.py`, `vec_unpack_addr`). TableGen gives 92 instructions the
same addressing class (`ag_pstm_2d` / `ag_pstm_3d`): scalar and vector loads and
stores, and the fused ups / conv loads and srs / pack / conv stores. These kernels
check that every family follows the same rule, and that every data word moves as the
instruction says.

Kernels
-------
  vec_agu_scalar   lda.2d/3d (word, s8, u8, s16, u16) and st.2d/3d (word, s8, s16)
  vec_agu_vector   vlda/vldb.2d/3d (256 bit, 128 bit, accumulator half) and the
                   matching vst.2d/3d
  vec_agu_fused    vlda.2d/3d.ups.*, vlda.2d/3d.conv.*, vst.2d/3d.srs.*,
                   vst.2d/3d.pack.*, vst.2d/3d.conv.*

One chunk is one (family, dimension) pair. A chunk runs a few concurrent walkers (so the
compiler uses several descriptor registers at once), each with its own descriptor
parameters, and after every access records the pointer and the counts. Load chunks
record the loaded data. Store chunks store known data and dump the stored memory image
after the walk.

Record layout (24 words: 96 bytes keeps a 256-bit store aligned in every record)
  data[16], pointer after the access, count 1, count 2, base address, 4 zero words.

Output per chunk: records in (group, iteration, walker) order, then the memory image of
each walker's store region (all zero for a load chunk).

Usage
-----
  python3 tools/gen_agu_2d3d.py generate
  python3 tools/stage_vector_kernel.py <kernel>          # no exp.bin yet
  # run the bridge on HW once (an uncaptured kernel fails by design; out.txt is
  # written first), then:
  python3 tools/gen_agu_2d3d.py capture <kernel> <build>/out.txt --date YYYY-MM-DD [--elf core.elf]
  python3 tools/stage_vector_kernel.py <kernel>          # now stages exp.bin = silicon
  python3 tools/gen_agu_2d3d.py proof <kernel> <core.elf>   # disassembly proof per family
"""

import argparse
import os
import random
import re
import subprocess

import capture_kernel_render as render
from gen_vldb_unpack import words_of

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY = os.path.join(REPO, "tests", "vector-verify")
GOLDEN_DIR = os.path.join(REPO, "tools", "golden", "silicon")
M32 = 0xFFFFFFFF

WALKERS = 4
GROUPS = 3
ITER_2D = 8
ITER_3D = 12
REC_WORDS = 24
HDR_WORDS = 16
TABLE_WORDS = 96            # 3D: GROUPS * WALKERS * 7 = 84, padded; 2D uses 48
REGION_BYTES = 2048         # each source region; walkers start in the middle
REGION_BASE = REGION_BYTES // 2
EXTENT = REGION_BASE - 64   # a walk must stay within +-EXTENT of its start (access <= 64 bytes)
IMG_WORDS = REGION_BYTES // 4   # image dumped per walker
REGION_WORD = HDR_WORDS + TABLE_WORDS   # 112: 32-byte aligned
NIN = REGION_WORD + 4 * REGION_BYTES // 4
MAX_RECS = GROUPS * ITER_3D * WALKERS
NOUT = MAX_RECS * REC_WORDS + WALKERS * IMG_WORDS
assert REGION_WORD % 8 == 0 and REC_WORDS * 4 % 32 == 0

SEED = 0xA6D20D3


# ---------------------------------------------------------------------------
# Descriptor sets. Byte units. Two pools per kind of access: vector accesses move in
# multiples of 32 bytes, scalar ones in multiples of 4.
# ---------------------------------------------------------------------------

# 2D: (modifier m, size, stride, initial count). The first ten are the sets pinned through
# vldb.unpack; the rest add a negative modifier and size 0.
POOL_2D_VECTOR = [
    (96, 4, 32, 0), (64, 2, 32, 0), (0, 3, 64, 0), (-64, 5, 32, 1),
    (32, 1, 32, 0), (128, 4, -32, 2), (32, 6, 32, 3), (-32, 3, 96, 0),
    (32, 2, 32, 5), (64, 3, 32, 7), (-96, 3, -32, 1), (32, 0, 32, 0),
]
POOL_2D_SCALAR = [
    (12, 4, 4, 0), (-20, 3, 8, 1), (4, 2, -4, 0), (36, 5, 12, 2),
    (0, 3, 4, 0), (-8, 1, 16, 0), (20, 6, -12, 3), (8, 2, 4, 5),
    (96, 4, 32, 0), (-64, 5, 32, 1), (32, 2, 32, 5), (16, 0, 4, 0),
]
# 3D: (m, size1, stride1, size2, stride2, count1, count2).
POOL_3D_VECTOR = [
    (32, 3, 32, 2, 160, 0, 0), (0, 2, 64, 3, 32, 0, 0), (-32, 4, 32, 2, 96, 1, 0),
    (64, 3, 32, 3, 128, 2, 1), (32, 1, 32, 2, 64, 0, 0), (96, 2, 32, 2, 32, 0, 1),
    (32, 2, 32, 2, 64, 0, 4), (32, 2, 32, 2, 64, 3, 0), (-64, 2, -32, 2, -96, 0, 0),
    (32, 0, 32, 2, 64, 0, 0), (32, 2, 32, 0, 64, 0, 0), (0, 1, 32, 1, 64, 1, 1),
]
POOL_3D_SCALAR = [
    (12, 3, 4, 2, 20, 0, 0), (-4, 2, 8, 3, 16, 0, 0), (8, 4, -4, 2, 12, 1, 0),
    (4, 3, 4, 3, 36, 2, 1), (0, 1, 12, 2, 8, 0, 0), (20, 2, 4, 2, 4, 0, 1),
    (32, 3, 32, 2, 160, 0, 0), (-32, 4, 32, 2, 96, 1, 0), (32, 2, 32, 2, 64, 0, 4),
    (4, 0, 4, 2, 8, 0, 0), (8, 2, 4, 0, 8, 0, 0), (-12, 2, -8, 2, -24, 0, 0),
]


def rule_walk(dim, params, iterations):
    """Pointer offsets visited by a walk, from the rule in agu.rs. Used only to keep a walk
    inside its region; the silicon capture is what checks the rule."""
    ptr, seen = 0, []
    if dim == 2:
        m, size, stride, count = params
        for _ in range(iterations):
            seen.append(ptr)
            if count < size:
                ptr, count = ptr + stride, count + 1
            else:
                ptr, count = ptr + m, 0
    else:
        m, size1, stride1, size2, stride2, count1, count2 = params
        for _ in range(iterations):
            seen.append(ptr)
            if count1 < size1:
                ptr, count1 = ptr + stride1, count1 + 1
            elif count2 < size2:
                ptr, count1, count2 = ptr + stride2, 0, count2 + 1
            else:
                ptr, count1, count2 = ptr + m, 0, 0
    return seen


def chunk_sets(dim, scalar):
    """The descriptor sets of a chunk: GROUPS groups of WALKERS sets."""
    pool = {(2, False): POOL_2D_VECTOR, (2, True): POOL_2D_SCALAR,
            (3, False): POOL_3D_VECTOR, (3, True): POOL_3D_SCALAR}[(dim, scalar)]
    iterations = ITER_2D if dim == 2 else ITER_3D
    sets = [pool[i % len(pool)] for i in range(GROUPS * WALKERS)]
    for s in sets:
        pos = rule_walk(dim, s, iterations)
        assert max(pos) <= EXTENT and min(pos) >= -EXTENT, (dim, s, min(pos), max(pos))
    return sets


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

class Family:
    """One TableGen family. `access(p, k, rec)` returns the C++ that makes one access
    through pointer expression `p`; `k` is the access index expression (stores take their
    source data from it) and `rec` the record pointer expression (loads write their data)."""

    def __init__(self, name, kind, mnemonic, access, data_words=0, source="R", scalar=False, dims=(2, 3)):
        self.name, self.kind, self.mnemonic = name, kind, mnemonic
        self.dims = dims
        self.access, self.data_words, self.source, self.scalar = access, data_words, source, scalar


def _zero_tail(rec, used):
    return "".join(f" {rec}[{i}] = 0;" for i in range(used, 16))


def load_family(name, mnemonic, body, data_words, scalar=False, source="R", dims=(2, 3)):
    def access(p, k, rec):
        return _fill(body).format(p=p, rec=rec) + _zero_tail(rec, data_words)

    return Family(name, "ld", mnemonic, access, data_words, source, scalar, dims)


def store_family(name, mnemonic, body, source="R", scalar=False, dims=(2, 3)):
    def access(p, k, rec):
        return _fill(body).format(p=p, k=k, s="src")

    return Family(name, "st", mnemonic, access, 0, source, scalar, dims)


# The accumulator widening / narrowing shifts are fixed and small; values are chosen so
# that nothing saturates (crSat is none, crRnd floor).
UPS_SHIFT = 4
SRS_SHIFT = 2


def _fill(body):
    return body.replace("@UPS@", str(UPS_SHIFT)).replace("@SRS@", str(SRS_SHIFT))


SCALAR_FAMILIES = [
    load_family("lda", "lda.2d", "{rec}[0] = *(int32_t *)({p});", 1, scalar=True),
    load_family("lda_s8", "lda.2d.s8", "{rec}[0] = (int32_t)*(int8_t *)({p});", 1, scalar=True),
    load_family("lda_u8", "lda.2d.u8", "{rec}[0] = (int32_t)*(uint8_t *)({p});", 1, scalar=True),
    load_family("lda_s16", "lda.2d.s16", "{rec}[0] = (int32_t)*(int16_t *)({p});", 1, scalar=True),
    load_family("lda_u16", "lda.2d.u16", "{rec}[0] = (int32_t)*(uint16_t *)({p});", 1, scalar=True),
    store_family("st", "st.2d", "*(int32_t *)({p}) = *(int32_t *)({s} + 4 * ({k}));", scalar=True),
    store_family("st_s8", "st.2d.s8", "*(int8_t *)({p}) = *(int8_t *)({s} + ({k}));", scalar=True),
    store_family("st_s16", "st.2d.s16", "*(int16_t *)({p}) = *(int16_t *)({s} + 2 * ({k}));", scalar=True),
]

# The mask-register (q) loads and stores are what a plain 128-bit vector access compiles to;
# the w-register 128-bit forms need aie_api load_v / store_v of a half vector.
VECTOR_FAMILIES = [
    load_family("ldq", "lda.2d", "*(v4int32 *){rec} = *(v4int32 *)({p});", 4),
    load_family("vld256", "vldb.2d", "*(v8int32 *){rec} = *(v8int32 *)({p});", 8),
    load_family("vld128", "vldb.2d.128",
                "{{ v8int32 t = set_v8int32(0, *(v4int32 *)({p})); *(v8int32 *){rec} = t; }}", 8),
    load_family("vld_am32", "vlda.2d", "*(v8acc32 *){rec} = *(v8acc32 *)({p});", 8),
    load_family("vld_am64", "vlda.2d", "*(v4acc64 *){rec} = *(v4acc64 *)({p});", 8),
    # The 3D form of this store compiles to vst.3d.128 (the w-register form below), so only the
    # 2D one reaches the q-register store: st.3d q is not reachable from C++.
    store_family("stq", "st.2d", "*(v4int32 *)({p}) = *(v4int32 *)({s} + 16 * ({k}));", dims=(2,)),
    store_family("vst256", "vst.2d", "*(v8int32 *)({p}) = *(v8int32 *)({s} + 32 * ({k}));"),
    store_family("vst128", "vst.2d.128",
                 "{{ aie::vector<int32_t, 8> v = aie::load_v<8>((int32_t *)({s} + 32 * ({k}))); "
                 "aie::store_v((int32_t *)({p}), v.extract<4>(0)); }}"),
    store_family("vst_am32", "vst.2d", "*(v8acc32 *)({p}) = *(v8acc32 *)({s} + 32 * ({k}));"),
    store_family("vst_am64", "vst.2d", "*(v4acc64 *)({p}) = *(v4acc64 *)({s} + 32 * ({k}));"),
]


def ups_family(suffix, acc, lanes, vec, vtype, rec_words, extract):
    name = f"ups_{suffix}"
    body = (f"{{{{ aie::accum<{acc}, {lanes}> a; a.from_vector(aie::vector<{vec}, {lanes}>(*({vtype} *)({{p}})), "
            f"@UPS@); *({'v16int32' if rec_words == 16 else 'v8int32'} *){{rec}} = "
            f"a{extract}.to_vector<int32_t>(0); }}}}")
    return name, body, rec_words


UPS_FAMILIES = []
for sign, vec_s, vec_u in (("s", "int", "int"), ("d", "uint", "uint")):
    for suffix, acc, lanes, el, rec_words, extract in (
            ("s32_16", "acc32", 16, 16, 16, ""), ("s32_8", "acc32", 32, 8, 16, ".extract<16>(0)"),
            ("s64_32", "acc64", 8, 32, 8, ""), ("s64_16", "acc64", 16, 16, 16, "")):
        vec = f"{vec_s}{el}_t"
        vtype = f"v{lanes}{vec_s}{el}"
        name, body, words = ups_family(f"{sign}_{suffix}", acc, lanes, vec, vtype, rec_words, extract)
        acc_bits, src_bits = suffix.split("_")[0][1:], suffix.split("_")[1]
        UPS_FAMILIES.append(load_family(
            name, f"vlda.2d.ups.s{acc_bits}.{sign}{src_bits}", body, words))
UPS_FAMILIES.append(load_family(
    "conv_ld", "vlda.2d.conv.fp32.bf16",
    "{{ aie::accum<accfloat, 16> a; a.from_vector(aie::vector<bfloat16, 16>(*(v16bfloat16 *)({p}))); "
    "*(v16float *){rec} = a.to_vector<float>(); }}", 16, source="B"))


SRS_FAMILIES = []
for sign in ("s", "d"):
    for acc, lanes, src_t, src_bytes, out_t, out_bits in (
            # The d8 form widens a byte vector: with 16-bit sources Chess spills an accumulator to
            # [sp, #0], an encoding the LLVM decoder (and so the emulator) cannot decode.
            ("acc32", 16, "int16_t", 32, "int16", 16),
            ("acc32", 32, "int8_t" if sign == "d" else "int16_t", 32 if sign == "d" else 64, "int8", 8),
            ("acc64", 16, "int16_t", 32, "int16", 16), ("acc64", 8, "int32_t", 32, "int32", 32)):
        acc_bits = acc[3:]
        otype = ("u" if sign == "d" else "") + out_t
        body = (f"{{{{ aie::accum<{acc}, {lanes}> a; a.from_vector(aie::vector<{src_t}, {lanes}>"
                f"(*(v{lanes}{src_t[:-2]} *)({{s}} + {src_bytes} * ({{k}}))), 0); "
                f"*(v{lanes}{otype} *)({{p}}) = a.to_vector<{otype}_t>(@SRS@); }}}}")
        SRS_FAMILIES.append(store_family(
            f"srs_{sign}{out_bits}_s{acc_bits}", f"vst.2d.srs.{sign}{out_bits}.s{acc_bits}", body, source="V"))

PACK_FAMILIES = [
    store_family("pack_s8_s16", "vst.2d.pack.s8.s16",
                 "*(v32int8 *)({p}) = aie::vector<int16_t, 32>(*(v32int16 *)({s} + 64 * ({k}))).pack<int8_t>();",
                 source="V"),
    store_family("pack_d8_d16", "vst.2d.pack.d8.d16",
                 "*(v32uint8 *)({p}) = aie::vector<uint16_t, 32>(*(v32uint16 *)({s} + 64 * ({k}))).pack<uint8_t>();",
                 source="V"),
    store_family("pack_s4_s8", "vst.2d.pack.s4.s8",
                 "*(v64int4 *)({p}) = aie::vector<int8_t, 64>(*(v64int8 *)({s} + 64 * ({k}))).pack<int4>();",
                 source="N"),
    store_family("pack_d4_d8", "vst.2d.pack.d4.d8",
                 "*(v64uint4 *)({p}) = aie::vector<uint8_t, 64>(*(v64uint8 *)({s} + 64 * ({k}))).pack<uint4>();",
                 source="N"),
    store_family(
        "conv_st", "vst.2d.conv.bf16.fp32",
        "{{ aie::accum<accfloat, 16> a; a.from_vector(aie::vector<bfloat16, 16>(*(v16bfloat16 *)({s} + 32 * ({k})))); "
        "*(v16bfloat16 *)({p}) = a.to_vector<bfloat16>(); }}", source="B"),
]

def _pick(families, *names):
    by_name = {f.name: f for f in families}
    return [by_name[n] for n in names]


# Each kernel is limited by the 16 KiB of core program memory (one function per chunk, about
# 800 bytes each), hence nine kernels of at most ten chunks.
KERNELS = {
    "vec_agu_ldscalar": ("agu_ldscalar", SCALAR_FAMILIES[:5]),
    "vec_agu_stscalar": ("agu_stscalar", SCALAR_FAMILIES[5:]),
    "vec_agu_vload": ("agu_vload", VECTOR_FAMILIES[:5]),
    "vec_agu_vstore": ("agu_vstore", VECTOR_FAMILIES[5:]),
    "vec_agu_ups_s": ("agu_ups_s", [f for f in UPS_FAMILIES if f.name.startswith("ups_s_")] + UPS_FAMILIES[-1:]),
    "vec_agu_ups_d": ("agu_ups_d", [f for f in UPS_FAMILIES if f.name.startswith("ups_d_")]),
    "vec_agu_srs_s": ("agu_srs_s", [f for f in SRS_FAMILIES if f.name.startswith("srs_s")]),
    "vec_agu_srs_d": ("agu_srs_d", [f for f in SRS_FAMILIES if f.name.startswith("srs_d")]),
    "vec_agu_pack": ("agu_pack", PACK_FAMILIES),
}


def chunks_of(kernel):
    """(family, dimension) pairs, in chunk order."""
    return [(fam, dim) for fam in KERNELS[kernel][1] for dim in fam.dims]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def region_random():
    rng = random.Random(SEED)
    return [rng.randrange(256) for _ in range(REGION_BYTES)]


def region_small():
    """Non-negative int16 values below 100: widening, narrowing and packing them is exact
    whatever the rounding and saturation modes are."""
    rng = random.Random(SEED + 1)
    out = []
    for _ in range(REGION_BYTES // 2):
        v = rng.randrange(100)
        out += [v & 0xFF, v >> 8]
    return out


def region_nibble():
    """Bytes in 0..7: packing them to four bits cannot saturate."""
    rng = random.Random(SEED + 3)
    return [rng.randrange(8) for _ in range(REGION_BYTES)]


def region_bf16():
    """Finite normal bfloat16 values: exponent field 110..140, random sign and mantissa."""
    rng = random.Random(SEED + 2)
    out = []
    for _ in range(REGION_BYTES // 2):
        exp, sign, mant = rng.randrange(110, 141), rng.randrange(2), rng.randrange(128)
        out += [((exp & 1) << 7) | mant, (sign << 7) | (exp >> 1)]
    return out


def chunk_words(kernel, index):
    fam, dim = chunks_of(kernel)[index]
    table = []
    for s in chunk_sets(dim, fam.scalar):
        table += [v & M32 for v in s]
    header = [index] + [0] * (HDR_WORDS - 1)
    words = header + table + [0] * (TABLE_WORDS - len(table))
    for region in (region_random(), region_small(), region_bf16(), region_nibble()):
        words += words_of(region)
    assert len(words) == NIN
    return words


def n_chunks(kernel):
    return len(chunks_of(kernel))


def input_words(kernel):
    words = []
    for i in range(n_chunks(kernel)):
        words += chunk_words(kernel, i)
    return words


def n_out(kernel):
    return n_chunks(kernel) * NOUT


def records_in(dim):
    return GROUPS * WALKERS * (ITER_2D if dim == 2 else ITER_3D)


# ---------------------------------------------------------------------------
# Kernel source
# ---------------------------------------------------------------------------

def walker_decls(dim):
    lines = []
    for w in range(WALKERS):
        if dim == 2:
            lines.append(
                f"    int off{w} = tab[{4 * w}], size{w} = tab[{4 * w + 1}], inc{w} = tab[{4 * w + 2}];"
                f" addr_t c{w} = tab[{4 * w + 3}];")
        else:
            lines.append(
                f"    int off{w} = tab[{7 * w}], size{w} = tab[{7 * w + 1}], inc{w} = tab[{7 * w + 2}],"
                f" sizeb{w} = tab[{7 * w + 3}], incb{w} = tab[{7 * w + 4}];"
                f" addr_t c{w} = tab[{7 * w + 5}], cb{w} = tab[{7 * w + 6}];")
    return "\n".join(lines)


def step_call(dim, w):
    if dim == 2:
        return f"p{w} = add_2d_ptr(p{w}, off{w}, size{w}, c{w}, inc{w});"
    return f"p{w} = add_3d_ptr(p{w}, off{w}, size{w}, c{w}, inc{w}, sizeb{w}, cb{w}, incb{w});"


def render_family(fn, fam, dim):
    iters = ITER_2D if dim == 2 else ITER_3D
    stride = 4 * WALKERS if dim == 2 else 7 * WALKERS
    inits = " ".join(f"p{w} = bp{w};" for w in range(WALKERS))
    bodies = []
    for w in range(WALKERS):
        rec = f"r{w}"
        k = f"((i * {WALKERS} + {w}) & 31)"
        c2 = f"cb{w}" if dim == 3 else "0"
        bodies.append(f"""      {{
        int32_t *{rec} = out + {w} * REC;
        {fam.access(f"p{w}", k, rec)}
        {step_call(dim, w)}
        {rec}[16] = (int32_t)(uintptr_t)p{w}; {rec}[17] = c{w}; {rec}[18] = {c2}; {rec}[19] = (int32_t)(uintptr_t)bp0;
        {rec}[20] = {rec}[21] = {rec}[22] = {rec}[23] = 0;
      }}""")
    return f"""
// {fam.mnemonic} ({dim}D), family {fam.name}
__attribute__((noinline)) static void {fn}(int8_t *restrict bp0, int8_t *restrict bp1, int8_t *restrict bp2,
                                          int8_t *restrict bp3, int8_t *restrict src, int32_t *restrict tab,
                                          int32_t *restrict out) {{
  for (int g = 0; g < GROUPS; g++) {{
    int8_t *p0, *p1, *p2, *p3;
    {inits}
{walker_decls(dim)}
    tab += {stride};
    for (int i = 0; i < {iters}; i++) {{
{chr(10).join(bodies)}
      out += {WALKERS} * REC;
    }}
  }}
}}
"""


def render_kernel_source(kernel):
    symbol, _ = KERNELS[kernel]
    chunks = chunks_of(kernel)
    funcs = "".join(render_family(f"chunk{i}", fam, dim) for i, (fam, dim) in enumerate(chunks))
    regions = {"R": "region_r", "V": "region_v", "B": "region_b", "N": "region_n"}
    mids = {"R": "mid_r", "B": "mid_b", "V": "mid_v", "N": "mid_n"}
    cases = []
    for i, (fam, dim) in enumerate(chunks):
        if fam.kind == "ld":
            m = mids[fam.source]
            bases = f"{m}, {m}, {m}, {m}"
        else:
            bases = f"scr, scr + {REGION_BYTES}, scr + 2 * {REGION_BYTES}, scr + 3 * {REGION_BYTES}"
        cases.append(f"""    case {i}:
      chunk{i}({bases}, {regions[fam.source]}, tab, out);
      break;""")
    return f"""//===- {symbol}.cc ----------------------------------------------*- C++ -*-===//
//
// 2D / 3D addressing capture kernel (GENERATED by tools/gen_agu_2d3d.py).
// Each chunk runs ONE (family, dimension) pair (header[0] is the chunk number) with
// {WALKERS} concurrent walkers over {GROUPS} groups of descriptor sets. After every access the
// pointer and the descriptor counts are stored, so the update the hardware applied is
// read from silicon, not from the compiler's model. Load chunks record the data loaded;
// store chunks store known data and dump the memory image after the walk.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define REC {REC_WORDS}
#define GROUPS {GROUPS}
#define REGION {REGION_BYTES}
#define REGION_WORD {REGION_WORD}
#define NOUT {NOUT}
#define MAX_RECS {MAX_RECS}
#define IMG {IMG_WORDS}

static int32_t scratch[{WALKERS} * REGION / 4] __attribute__((aligned(32)));
{funcs}
extern "C" {{

void {symbol}(int32_t *restrict in, int32_t *restrict out) {{
  int chunk = in[0];
  int32_t *tab = in + {HDR_WORDS};
  int8_t *region_r = (int8_t *)(in + REGION_WORD);
  int8_t *region_v = region_r + REGION;
  int8_t *region_b = region_v + REGION;
  int8_t *region_n = region_b + REGION;
  int8_t *mid_r = region_r + {REGION_BASE};
  int8_t *mid_v = region_v + {REGION_BASE};
  int8_t *mid_b = region_b + {REGION_BASE};
  int8_t *mid_n = region_n + {REGION_BASE};
  int8_t *scr = (int8_t *)scratch + {REGION_BASE};
  event0();
  // Stale output of the previous chunk and stale scratch must not leak into this one.
  v8int32 zero = aie::zeros<int32_t, 8>();
  for (int i = 0; i < NOUT; i += 8) chess_prepare_for_pipelining chess_loop_range(8, ) *(v8int32 *)(out + i) = zero;
  for (int i = 0; i < {WALKERS} * REGION / 4; i += 8) chess_prepare_for_pipelining chess_loop_range(8, ) *(v8int32 *)(scratch + i) = zero;
  ::aie::set_rounding(aie::rounding_mode::floor);
  ::aie::set_saturation(aie::saturation_mode::none);
  switch (chunk) {{
{chr(10).join(cases)}
  }}
  // Memory image of every walker's store region (zero for a load chunk).
  int32_t *img = out + MAX_RECS * REC;
  for (int i = 0; i < {WALKERS} * IMG; i += 8) chess_prepare_for_pipelining chess_loop_range(8, )
    *(v8int32 *)(img + i) = *(v8int32 *)(scratch + i);
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
    return list(KERNELS)


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
    symbol, _ = KERNELS[kernel]
    files = {
        f"{symbol}.cc": render_kernel_source(kernel),
        "aie.mlir": render.render_mlir(
            kernel, symbol, n_chunks(kernel), NIN, NOUT, "i32", "i32",
            f"one chunk per (family, dimension); records of {REC_WORDS} words, then the store image."),
        "run.lit": render.render_run_lit(f"{symbol}.cc", n_out(kernel)),
    }
    render.write_files(os.path.join(VERIFY, kernel), files)
    return n_chunks(kernel)


def description(kernel):
    return {
        "walkers": WALKERS, "groups": GROUPS, "iterations_2d": ITER_2D, "iterations_3d": ITER_3D,
        "record_words": REC_WORDS, "record_fields": ["data[16]", "pointer_after", "count1", "count2",
                                                      "base_address", "zero[4]"],
        "header_words": HDR_WORDS, "table_words": TABLE_WORDS, "region_word": REGION_WORD,
        "region_bytes": REGION_BYTES, "region_base": REGION_BASE,
        "image_words": IMG_WORDS, "max_records": MAX_RECS, "words_per_chunk": NOUT,
        "regions": {"R": "random bytes", "V": "small non-negative int16", "B": "finite bf16", "N": "bytes 0..7"},
        "chunks": [
            {"family": fam.name, "kind": fam.kind, "mnemonic": fam.mnemonic, "dim": dim,
             "data_words": fam.data_words, "source": fam.source,
             "records": records_in(dim), "sets": [list(s) for s in chunk_sets(dim, fam.scalar)]}
            for fam, dim in chunks_of(kernel)
        ],
    }


# -- disassembly proof ------------------------------------------------------

def _objdump(elf, start):
    objdump = os.path.join(os.environ["PEANO_INSTALL_DIR"], "bin", "llvm-objdump")
    return subprocess.run([objdump, "-d", f"--start-address={start}", elf],
                          check=False, capture_output=True, text=True)


def disassemble(elf):
    """Disassembly text of a core ELF, and the addresses of bundles the printer could not print.

    llvm-objdump asserts on the few bundles it has no asm string for. Instead of giving up,
    resume after the unprintable bundle: its length is not known, so try the following even
    offsets until the printer produces sensible instructions again.
    """
    text, skipped, start = [], [], 0
    while True:
        run = _objdump(elf, start)
        lines = run.stdout.splitlines()
        text += lines
        if run.returncode == 0:
            return "\n".join(text), skipped
        last = None
        for raw in lines:
            m = LINE.match(raw)
            if m:
                last = (int(m.group(1), 16), len(m.group(2).split()))
            elif re.match(r"^[0-9a-f]{8} <", raw):
                last = (int(raw.split()[0], 16), 0)
        if last is None:
            raise SystemExit(f"llvm-objdump failed on {elf} before printing anything")
        bad = last[0] + last[1]
        skipped.append(bad)
        for length in range(2, 18, 2):
            probe = _objdump(elf, bad + length)
            head = [l for l in probe.stdout.splitlines() if LINE.match(l)][:3]
            if head and not any("<unknown>" in l for l in head):
                start = bad + length
                break
        else:
            raise SystemExit(f"cannot resume llvm-objdump after the bundle at {bad:#x}")


# `  addr: b0 b1 ...  <tab> text`. The byte column is padded with spaces and the printer
# wraps a long bundle's bytes into the same column, so take every pair left of the tab.
LINE = re.compile(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+)*[0-9a-f]{2})\s*\t(.*)$")
MEM_2D3D = re.compile(r"^(?:v?ld[ab]|v?st)\.[23]d\S*\s")


def operand_class(slot):
    """Register class of the first operand of an instruction text (`wl0` -> `wl`, `r24` -> `r`)."""
    rest = slot.split(None, 1)
    if len(rest) < 2:
        return ""
    return re.sub(r"\d+$", "", rest[1].split(",")[0].strip())


def proof(elf):
    """Per 2D/3D mnemonic, descriptor and data-operand register class: the first bundle that
    contains it (bytes and text) and how often it occurs. The same mnemonic serves several
    register files (`vst.2d` stores w, x and accumulator registers), hence the class."""
    found = {}
    listing, skipped = disassemble(elf)
    if skipped:
        print(f"note: llvm-objdump cannot print the bundles at {[hex(a) for a in skipped]}; skipped")
    for raw in listing.splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        addr, hexbytes, text = m.groups()
        for slot in (s.strip() for s in text.split(";")):
            if not MEM_2D3D.match(slot + " "):
                continue
            mnemonic = slot.split()[0]
            desc = re.search(r"\bd(\d)(_3d)?\s*$", slot)
            key = (mnemonic, desc.group(0).strip() if desc else "?", operand_class(slot))
            found.setdefault(key, {"count": 0, "bytes": " ".join(hexbytes.split()), "bundle": text.strip()})
            found[key]["count"] += 1
    return found


def capture(kernel, out_txt, date, elf=None):
    silicon = [v & M32 for v in render.read_out_txt(out_txt)]
    assert len(silicon) == n_out(kernel), f"{len(silicon)} != {n_out(kernel)}"
    rec = {"kernel": kernel, "class": "agu_2d3d", "n": len(silicon), "layout": description(kernel)}
    if elf:
        rec["encodings"] = [
            {"mnemonic": mn, "descriptor": d, "operand": oc, "count": e["count"], "bytes": e["bytes"],
             "bundle": e["bundle"]}
            for (mn, d, oc), e in sorted(proof(elf).items())
        ]
    rec["inputs"] = [input_words(kernel)]
    rec["silicon"] = silicon
    rec["provenance"] = f"HW-observed: NPU1 Phoenix silicon, captured {date} via tools/gen_agu_2d3d.py"
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
    prf.add_argument("kernel")
    prf.add_argument("elf")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        for k in kernel_names():
            print(f"wrote {k} ({generate_one(k)} chunks)")
    elif args.cmd == "capture":
        if args.kernel not in KERNELS:
            raise SystemExit(f"unknown kernel {args.kernel}")
        print("wrote", capture(args.kernel, args.out_txt, args.date, args.elf))
    else:
        for (mn, d, oc), e in sorted(proof(args.elf).items()):
            print(f"{mn:32s} {d:5s} {oc:5s} x{e['count']:<3d} {e['bytes']:30s} {e['bundle']}")


if __name__ == "__main__":
    main()
