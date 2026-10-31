#!/usr/bin/env python3
"""Generate the capture kernels for the AIE2 sparse loads `vldb.sparse.{reset,fill,peek,pop}.{4,8,16}`.

Why this exists
---------------
The sparse load reads a compressed stream from memory and fills half of a `qx` register (256
bits of stored data and 64 bits of mask per `pop`/`peek`). It keeps a hidden decompression
state `DP`, advances the pointer by a data-dependent amount, and sets the status flags
`srSparse_of` and `srCompr_uf`. None of that is stated anywhere we may read, so it is captured
from NPU1 silicon.

Each kernel (one per element width 4, 8, 16 and script group `proto`/`ring`: program memory holds
about 120 recorded steps) takes one designed 1 KiB stream per chunk, first writes the whole ring
(reset and nine fills, unrecorded) so that nothing depends on what ran before, and then runs every
SCRIPT of its group on the stream. A script is a list of steps; after each step a 32-word record is
stored:

    word 0        DP, raw (the value the last `dp` step read; bit 8 of word 3 says this step read it)
    word 1        srSparse_of, raw, read after the step
    word 2        srCompr_uf, raw, read after the step
    word 3        step id (the op code) | 1 << 8 when this step read DP
    words 8..11   q: the 128 mask bits of the sparse register `v` the step left
    words 16..23  wl: its low 256 data bits
    words 24..31  wh: its high 256 data bits

Steps: `reset`, `fill`; `pop_lo`, `peek_lo` (a fresh value: only the low half is defined);
`pop_hi`, `peek_hi` (insert the high half into the value the last lo step left); `dp` (a peek
through a throwaway copy of the pointer, `peek_and_get_pointer`, whose result is DP); `clr`
(clear both flags). `v` starts every script as marker bytes (data 0xA5, mask 0x5A) so a half a
step leaves alone is visible. The high halves a lo step leaves undefined are zeroed when a
capture is stored (`normalize`), so two captures of the same kernel are bit-identical.

The behaviour these captures pin down is stated and replayed in `docs/fidelity-gaps/vector-compute.md`
and `src/interpreter/execute/sparse_load.rs`; `scratch` analysis lives in the sparse evidence dir.
Every stream sits inside the kernel's own input buffer with at least a quarter of it left over
after the longest script, so no pop or fetch reads past allocated memory.

Usage
-----
  python3 tools/gen_vldb_sparse.py generate [--group proto|ring]
  python3 tools/stage_vector_kernel.py <kernel>          # no exp.bin yet
  # run the bridge on HW once (an uncaptured kernel fails by design; out.txt is written first),
  # then:
  python3 tools/gen_vldb_sparse.py capture <kernel> <build>/out.txt --date YYYY-MM-DD [--elf <core elf>]
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

STREAM_BYTES = 1024
RING_LINES = 10    # the 320-byte ring behind the sparse loads, see the model in the docs
STREAM_WORDS = STREAM_BYTES // 4
REC = 32
MAX_CHUNKS = 64
WIDTHS = (4, 8, 16)

# Step names and their ids in the record.
STEPS = {"clr": 0, "reset": 1, "fill": 2, "peek_lo": 3, "peek_hi": 4, "pop_lo": 5, "pop_hi": 6, "dp": 7}

ALL_SCRIPTS = [
    ("protocol", "reset fill fill fill pop_lo pop_hi dp pop_lo pop_hi dp pop_lo pop_hi dp pop_lo pop_hi fill "
                 "pop_lo pop_hi dp"),
    ("peek", "reset fill fill fill peek_lo peek_hi dp peek_lo pop_lo pop_hi dp peek_lo peek_hi dp"),
    ("underfilled", "reset fill pop_lo pop_hi dp pop_lo pop_hi dp"),
    ("unprimed", "reset pop_lo pop_hi dp fill pop_lo pop_hi dp"),
    ("carry", "fill pop_lo pop_hi dp pop_lo pop_hi dp"),
    ("resets", "reset reset fill fill fill pop_lo pop_hi dp reset pop_lo pop_hi dp"),
    ("fills", "reset fill fill fill fill fill fill fill fill dp"),
    ("flags", "clr reset clr fill clr fill clr fill clr peek_lo clr pop_lo clr pop_hi clr"),
    ("ring0", "clr reset " + "clr pop_lo clr pop_hi " * 6 + "dp"),
    ("ring1", "clr reset fill " + "clr pop_lo clr pop_hi " * 6 + "dp"),
    ("ring2", "clr reset fill fill " + "clr pop_lo clr pop_hi " * 6 + "dp"),
    ("ring3", "clr reset fill fill fill " + "clr pop_lo clr pop_hi " * 6 + "dp"),
    ("many", "clr reset fill fill fill pop_lo pop_hi pop_lo pop_hi pop_lo pop_hi pop_lo pop_hi fill pop_lo "
             "pop_hi pop_lo pop_hi pop_lo pop_hi pop_lo pop_hi dp"),
]

# Program memory holds about 120 recorded steps, so the scripts go into one kernel per group.
GROUPS = {
    "proto": ["protocol", "peek", "underfilled", "unprimed", "carry", "resets", "fills", "flags", "many"],
    "ring": ["ring0", "ring1", "ring2", "ring3"],
}
SCRIPTS = list(ALL_SCRIPTS)
GROUP = "proto"


def set_group(group):
    global GROUP
    GROUP = group
    select_scripts(GROUPS[group])


def select_scripts(names):
    """Keep only the named scripts (exploration runs a few at a time; the capture runs all)."""
    global SCRIPTS
    keep = [sc for sc in ALL_SCRIPTS if sc[0] in names]
    assert keep, f"no script named {names}"
    SCRIPTS = keep


def steps_of(script):
    return script.split()


def max_steps():
    return max(len(steps_of(s)) for _, s in SCRIPTS)


# --- Designed streams ----------------------------------------------------------

def streams():
    """The designed 1 KiB streams, as lists of byte values."""
    rng = random.Random(0x5A9E)
    out = []
    out.append(("ramp", [(i + 1) & 0xFF for i in range(STREAM_BYTES)]))
    out.append(("ones", [0xFF] * STREAM_BYTES))
    out.append(("zeros", [0] * STREAM_BYTES))
    out.append(("alt_aa55", [0xAA if i % 2 == 0 else 0x55 for i in range(STREAM_BYTES)]))
    for p in (0, 1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64):
        b = [0] * STREAM_BYTES
        b[p] = 0x81
        out.append((f"onehot_{p}", b))
    for _ in range(6):
        out.append((f"rand_{len(out)}", [rng.randrange(256) for _ in range(STREAM_BYTES)]))
    # Position-coded: every byte is 00, 0F, F0 or FF, so no mask nibble ever selects data (0 and F
    # take none), and no two 8-byte windows are equal. A mask read back names where it came from.
    for name in ("dbseq_a", "dbseq_b"):
        while True:
            b = [rng.choice((0x00, 0x0F, 0xF0, 0xFF)) for _ in range(STREAM_BYTES)]
            if len({tuple(b[i:i + 8]) for i in range(STREAM_BYTES - 7)}) == STREAM_BYTES - 7:
                break
        out.append((name, b))
    # Every pop is 8 bytes of 0x55 (nibble 5: two selected, all 16 groups full) then 32 data bytes
    # numbered by stream offset, so a returned data byte names the offset it was read from.
    dense = []
    while len(dense) < STREAM_BYTES:
        dense += [0x55] * 8 + [(len(dense) + 8 + i) % 255 + 1 for i in range(32)]
    out.append(("dense40", dense[:STREAM_BYTES]))
    # A 16-byte head with a chosen number of set bits, then a ramp (candidate mask + data).
    for pop in (0, 1, 2, 8, 32, 64, 96, 128):
        head = [0] * 16
        bits = rng.sample(range(128), pop)
        for bit in bits:
            head[bit // 8] |= 1 << (bit % 8)
        out.append((f"head_pop{pop}", head + [(i + 1) & 0xFF for i in range(STREAM_BYTES - 16)]))
    return out


def stream_words(byte_values):
    return list(struct.unpack("<%dI" % STREAM_WORDS, bytes(byte_values)))


# --- Rendering -------------------------------------------------------------------

def types_of(width):
    """(sparse value type, compressed pointer type, data type) of an element width."""
    return {4: ("v256int4_sparse", "v256int4_sparse_compress", "v128int4"),
            8: ("v128int8_sparse", "v128int8_sparse_compress", "v64int8"),
            16: ("v64int16_sparse", "v64int16_sparse_compress", "v32int16")}[width]


STEP_CODE = {
    "clr": "clr_sparse_of(); clr_compr_uf();",
    "reset": "sparse_reset(P);",
    "fill": "sparse_fill(P);",
    "peek_lo": "v = sparse_peek_set_lo(P);",
    "peek_hi": "v = sparse_peek_insert_hi(v, P);",
    "pop_lo": "v = sparse_pop_set_lo(P);",
    "pop_hi": "v = sparse_pop_insert_hi(v, P);",
    # The _and_get_pointer forms overwrite their pointer argument with DP, so DP is read through a
    # throwaway copy and the pointer the other steps use stays the stream base.
    "dp": "{ pt *Q = P; (void)sparse_peek_and_get_pointer(Q); last_dp = (int32_t)(uintptr_t)Q; }",
}
READS_DP = {"dp"}


def render_kernel(width):
    vt, pt, dt = types_of(width)
    body = []
    for si, (name, script) in enumerate(SCRIPTS):
        body.append(f"  // script {si}: {name}: {script}")
        body.append("  {")
        body.append(f"    {pt} *P = ({pt} *)in;")
        body.append(f"    {vt} v;")
        body.append("    v = update(v, *(volatile DT *)marker_data);")
        body.append("    v = update(v, *(volatile sparsity_t *)marker_mask);")
        body.append("    int32_t last_dp = 0;")
        for step in steps_of(script):
            body.append(f"    {STEP_CODE[step].replace('pt *', pt + ' *')}")
            body.append(f"    RECORD(o, last_dp, v, {STEPS[step]}, {int(step in READS_DP)});")
            body.append("    o += REC;")
        body.append("  }")
    # The ring keeps whatever it held, so every chunk first writes all ten of its lines: the
    # records then do not depend on what ran on the core before.
    prologue = (f"  {{\n    {pt} *P = ({pt} *)in;\n    sparse_reset(P);\n"
                + "    sparse_fill(P);\n" * (RING_LINES - 1) + "  }")
    body_text = prologue + "\n" + "\n".join(body)
    data_init = ", ".join(["0xA5"] * 64)
    mask_init = ", ".join(["0x5A"] * 16)
    return f"""//===- sparse_load.cc -------------------------------------------*- C++ -*-===//
//
// Sparse-load capture kernel, element width {width} (GENERATED by tools/gen_vldb_sparse.py).
// One designed {STREAM_BYTES}-byte stream per chunk; every script runs on it with the pointer at
// the stream base, and each step leaves a {REC}-word record (see the generator's docstring).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define REC {REC}
#define DT {dt}
#define VT {vt}

// The sparse register is split with the data/mask extract intrinsics. Vector stores ignore the low
// address bits, so the records are 128 bytes with the mask at +32 B and the data at +64 B. The status registers and DP are stored raw: they are wider than one-bit flags.
#define RECORD(o, dp, v, id, fresh)                                              \\
  do {{                                                                           \\
    (o)[0] = (dp);                                                               \\
    (o)[1] = (int32_t)get_sparse_of();                                           \\
    (o)[2] = (int32_t)get_compr_uf();                                            \\
    (o)[3] = (id) | ((fresh) << 8);                                              \\
    *(DT *)((o) + 16) = extract_sparse_data(v);                                  \\
    *(sparsity_t *)((o) + 8) = extract_sparsity(v);                              \\
  }} while (0)

// Marker sources read from memory: Chess cannot materialize a 128-bit constant.
static uint8_t marker_data[64] __attribute__((aligned(64))) = {{{data_init}}};
static uint8_t marker_mask[16] __attribute__((aligned(16))) = {{{mask_init}}};

extern "C" {{

void sparse_load(uint8_t *restrict in, int32_t *restrict out) {{
  int32_t *o = out;
  event0();
{body_text}
  event1();
}}

}} // extern "C"
"""


def out_words_per_chunk():
    return sum(len(steps_of(s)) for _, s in SCRIPTS) * REC


def kernel_name(width, group=None):
    return f"vec_sparseload_{group or GROUP}_w{width}"


def kernel_names():
    return [kernel_name(w, grp) for grp in GROUPS for w in WIDTHS]


def golden_path(width, group=None):
    return os.path.join(GOLDEN_DIR, kernel_name(width, group) + ".json")


def parse_kernel(name):
    """(group, width) of a kernel name."""
    _, group, w = name.rsplit("_", 2)
    set_group(group)
    return group, int(w[1:])


def input_words():
    words = []
    for _, b in streams():
        words += stream_words(b)
    return words


def kernel_data(name):
    _, width = parse_kernel(name)
    golden = render.load_golden(golden_path(width))
    words = input_words()
    if golden is not None and golden["inputs"] != [words]:
        raise SystemExit(f"{name}: golden inputs are stale; recapture on silicon")
    expected = None if golden is None else golden["silicon"]
    assert expected is None or len(expected) == len(streams()) * out_words_per_chunk()
    return render.kernel_data(words, expected)


def generate_one(width):
    name = kernel_name(width)
    chunks = len(streams())
    assert chunks <= MAX_CHUNKS
    kernel_data(name)
    n_out = chunks * out_words_per_chunk()
    files = {
        "sparse_load.cc": render_kernel(width),
        "aie.mlir": render.render_mlir(
            name, "sparse_load", chunks, STREAM_WORDS, out_words_per_chunk(), "i32", "i32",
            f"vldb.sparse width {width}; {len(SCRIPTS)} scripts, {REC} words per step."),
        "run.lit": render.render_run_lit("sparse_load.cc", n_out),
    }
    render.write_files(os.path.join(VERIFY, name), files)
    return chunks


# --- disassembly proof ------------------------------------------------------

SPARSE_MNEMONIC = re.compile(r"^vldb\.sparse\.(?:reset|fill|peek|pop)\.(?:4|8|16)$")


def proof(elf):
    """Per mnemonic: the first bundle containing it (bytes and text) and a count."""
    found = {}
    listing, skipped = agu.disassemble(elf)
    if skipped:
        print(f"note: llvm-objdump cannot print the bundles at {[hex(a) for a in skipped]}; skipped")
    for raw in listing.splitlines():
        m = agu.LINE.match(raw)
        if not m:
            continue
        _, hexbytes, text = m.groups()
        for slot in (s.strip() for s in text.split(";")):
            head = slot.split()[0] if slot.split() else ""
            if SPARSE_MNEMONIC.match(head):
                found.setdefault(head, {"count": 0, "bytes": " ".join(hexbytes.split()), "bundle": text.strip()})
                found[head]["count"] += 1
    return found


HI_HALF_WORDS = (10, 11, *range(24, 32))    # q high 64 bits, wh


def normalize(silicon):
    """Zero what a lo step leaves undefined.

    `sparse_pop_set_lo` and `sparse_peek_set_lo` return a fresh register of which only the low half
    is written; the high halves hold whatever the register held before, which differs from run to
    run. They stay undefined in every later record until a hi step fills them in (or a script
    starts over from the marker value). Everything else in a record is defined and left as captured.
    """
    out = list(silicon)
    base = 0
    while base < len(out):
        for _, script in SCRIPTS:
            undefined = False
            for step in steps_of(script):
                if step in ("peek_lo", "pop_lo"):
                    undefined = True
                elif step in ("peek_hi", "pop_hi"):
                    undefined = False
                if undefined:
                    for w in HI_HALF_WORDS:
                        out[base + w] = 0
                base += REC
    return out


def capture(width, out_txt, date, elf=None):
    silicon = normalize([v & 0xFFFFFFFF for v in render.read_out_txt(out_txt)])
    n = len(streams()) * out_words_per_chunk()
    assert len(silicon) == n, f"{len(silicon)} != {n}"
    rec = {
        "kernel": kernel_name(width), "class": "sparse_load", "n": n,
        "layout": {"width": width, "stream_bytes": STREAM_BYTES, "record_words": REC,
                   "streams": [name for name, _ in streams()],
                   "scripts": [{"name": nm, "steps": steps_of(sc)} for nm, sc in SCRIPTS],
                   "step_ids": STEPS},
        "encodings": [{"mnemonic": mn, "count": e["count"], "bytes": e["bytes"], "bundle": e["bundle"]}
                      for mn, e in sorted(proof(elf).items())] if elf else [],
        "inputs": [input_words()],
        "silicon": silicon,
        "provenance": f"HW-observed: NPU1 Phoenix silicon, captured {date} via tools/gen_vldb_sparse.py",
    }
    return render.write_golden(golden_path(width), rec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--group", choices=sorted(GROUPS), help="only this script group")
    gen.add_argument("--scripts", help="comma-separated script names (exploration; default: all)")
    cap = sub.add_parser("capture")
    cap.add_argument("kernel")
    cap.add_argument("out_txt")
    cap.add_argument("--date", required=True)
    cap.add_argument("--elf")
    prf = sub.add_parser("proof")
    prf.add_argument("elf")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        for grp in ([args.group] if args.group else GROUPS):
            set_group(grp)
            if args.scripts:
                select_scripts(args.scripts.split(","))
            for w in WIDTHS:
                print(f"wrote {kernel_name(w)} ({generate_one(w)} chunks)")
    elif args.cmd == "capture":
        if args.kernel not in kernel_names():
            raise SystemExit(f"unknown kernel {args.kernel}")
        _, width = parse_kernel(args.kernel)
        print("wrote", capture(width, args.out_txt, args.date, args.elf))
    else:
        for mn, e in sorted(proof(args.elf).items()):
            print(f"{mn:24s} x{e['count']:<3d} {e['bytes']:34s} {e['bundle']}")


if __name__ == "__main__":
    main()
