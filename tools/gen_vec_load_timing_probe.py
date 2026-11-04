#!/usr/bin/env python3
"""Hand-scheduled vector-register load timing probe for AIE2: when a load's data lands, and
what a load in flight sees across a taken branch.

Why this exists
---------------
The accumulator timing probe (gen_acc_timing_probe.py) found that `vlda am` loads follow the
accumulator file's write rule, not a wall-clock load commit. The vector-register loads (`vlda`,
`vldb` into `wl`/`wh`, their post-modify, 2D and 128-bit forms, the fused `vldb.unpack` and the
`vldb.4x` gather) were never probed the same way, and neither was a load whose data is still in
flight when a taken branch redirects the pipeline. This generator writes two families of
raw-assembly kernels; both classify every output word against in-kernel references, so no
value is computed by a model.

Family 1, `generate`: producer/consumer and write-after-write points
  The contested register is R = x4 = {wl4, wh4}. A producer (a vector load) and a consumer (an
  instruction that reads R) sit `d` bundles apart (RAW: load first; WAR: consumer first, d = 0
  shares a bundle), exactly as in gen_acc_timing_probe.py, whose 128-byte record layout and
  point metadata these kernels reuse, so `tools/analyze_acc_timing.py` analyses them unchanged.
  Write-after-write points (`--waw A:B`) issue a load A `s` bundles after another writer B of R
  and store R; the references s = -14 / +14 give the all-OLD / all-NEW records.

Family 2, `branch`: loads across a taken branch
  A load (scalar `lda r4`, vector `vlda wl4`, accumulator `vlda amll0`) issues at position i
  relative to a branch bundle (i < 0 before it, 0 in it, 1..5 in its delay slots); the branch
  target holds eight readers of the loaded register in consecutive bundles (target + j,
  j = 0..7), each storing what it saw into its own 32-byte slot. The issue distance of reader j
  is d = 6 + j - i in every kind: `none` (the branch bundle is a nop and the target follows the
  delay slots directly), `j` (unconditional), `jnz` taken, `jnz` not taken. Two references per
  kernel: the load issued long before the branch (every reader NEW) and no load (every reader
  OLD). The analysis (`analyze-branch`) reports per (family, kind, i) the first reader that sees
  NEW, as the distance d.

Usage
-----
  python3 tools/gen_vec_load_timing_probe.py generate <dir> [--pairs P:C ..] [--waw A:B ..]
  python3 tools/gen_vec_load_timing_probe.py branch <dir>
  python3 -m harness suite build|hw|emu|compare <dir> ...        (PYTHONPATH=tools)
  python3 tools/analyze_acc_timing.py <dir> <tag>                  (family 1)
  python3 tools/gen_vec_load_timing_probe.py analyze-branch <dir> <tag> [--compare TAG]
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import gen_acc_timing_probe as base  # noqa: E402
from harness import suite  # noqa: E402

isa_gen = base.isa_gen
nops = base.nops
LOAD_NOPS = base.LOAD_NOPS
RECORD_BYTES = base.RECORD_BYTES
REF_DISTANCE = base.REF_DISTANCE
SETTLE_BEFORE = base.SETTLE_BEFORE
SETTLE_AFTER = base.SETTLE_AFTER

# --------------------------------------------------------------------------
# Input table: the accumulator probe's table (offsets unchanged) plus the loaded data and the
# gather address vectors, appended after it
# --------------------------------------------------------------------------

EXTRA_ITEMS = [
    # what the vector loads bring in: 32 int16 / 32 bf16 elements, distinct from every source
    ("LDV_i", base._u16s([0x700 + 3 * k for k in range(32)])),
    ("LDV_f", base._u16s([0x4080 + k for k in range(32)])),
    # eight copies each of one absolute address (a W register: the `lo` gathers use words 0..3,
    # the `hi` ones 4..7), written by the kernel prologue: gathers read OLD_i or LDV_i through them
    ("ADDR_OLD", bytes(32)),
    ("ADDR_NEW", bytes(32)),
]


def build_table() -> tuple[bytes, dict[str, int]]:
    buf = bytearray(base.TABLE_BYTES)
    offsets = dict(base.TABLE_OFFSETS)
    for name, data in EXTRA_ITEMS:
        while len(buf) % 32:
            buf.append(0)
        offsets[name] = len(buf)
        buf += data
    while len(buf) % 32:
        buf.append(0)
    return bytes(buf), offsets


TABLE_BYTES, TABLE_OFFSETS = build_table()
IN_SIZE = len(TABLE_BYTES)
P3_BASE = base.P3_BASE


def table_ref(name: str, extra: int = 0) -> tuple[str, int]:
    off = TABLE_OFFSETS[name] + extra
    if off <= isa_gen.MAX_VECTOR_OFFSET:
        return "p0", off
    return "p3", off - P3_BASE


def load_vec(reg: str, name: str, half: int = 0) -> list[str]:
    ptr, off = table_ref(name, 32 * half)
    return [f"  vlda {reg}, [{ptr}, #{off}]"]


def point_to(preg: str, name: str) -> list[str]:
    return [f"  mov {preg}, p0", f"  padda [{preg}], #{TABLE_OFFSETS[name]}"]


# --------------------------------------------------------------------------
# Family 1 catalogue
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Load:
    name: str
    asm: str
    halves: frozenset          # which halves of R it writes: {"wl"}, {"wh"} or both
    dom: Optional[str] = None


@dataclass(frozen=True)
class Reader:
    name: str
    asm: str
    halves: frozenset          # which halves of R it reads
    dom: Optional[str]
    out: str                   # mem | cm6 | bml6 | x8 | wl8 | r9
    gather: bool = False       # reads R as an address vector (OLD/NEW = ADDR_OLD/ADDR_NEW)


WL, WH, X = frozenset({"wl"}), frozenset({"wh"}), frozenset({"wl", "wh"})

# p5 addresses the item a load brings in (LDV_<dom>, or ADDR_NEW for a gather reader); wl9 holds
# ADDR_NEW for the gather load. Every point resets p5, so post-modify forms start from the item.
LOADS = [
    Load("vlda_wl", "vlda wl4, [p5, #0]", WL),
    Load("vlda_wh", "vlda wh4, [p5, #32]", WH),
    Load("vldb_wl", "vldb wl4, [p5, #0]", WL),
    Load("vldb_wh", "vldb wh4, [p5, #32]", WH),
    Load("vlda_post", "vlda wl4, [p5], #32", WL),
    Load("vldb_2d", "vldb.2d wl4, [p5], d0", WL),
    Load("vlda_128", "vlda.128 wl4, [p5]", WL),
    Load("vldb_unpack", "vldb.unpack.s16.s8 x4, [p5, dj0]", X),
    Load("vldb_4x", "vldb.4x32.lo wl4, wl9", WL),
]

READERS = [
    Reader("vst_wl", "vst wl4, [p1, #0]", WL, None, "mem"),
    Reader("vst_wh", "vst wh4, [p1, #0]", WH, None, "mem"),
    Reader("vadd_32", "vadd.32 x8, x4, x1", X, "i", "x8"),
    Reader("vmov_x", "vmov x8, x4", X, None, "x8"),
    Reader("vmov_wl", "vmov wl8, wl4", WL, None, "wl8"),
    Reader("vmov_wh", "vmov wl8, wh4", WH, None, "wl8"),
    Reader("vmul_x", "vmul cm6, x4, x1, r1", X, "i", "cm6"),
    Reader("vmul_y", "vmul cm6, x1, x4, r1", X, "i", "cm6"),
    Reader("vmul_f", "vmul.f bml6, x4, x3, r3", X, "f", "bml6"),
    Reader("vups", "vups.s32.s16 bml6, wl4, s3", WL, None, "bml6"),
    Reader("vmov_bm_x", "vmov bml6, x4", X, None, "bml6"),
    Reader("vconv_fp32_bf16", "vconv.fp32.bf16 bml6, wl4", WL, "f", "bml6"),
    Reader("vpack", "vpack.s8.s16 wl8, x4", X, "i", "wl8"),
    # r16 = 2: index 0 hangs silicon (docs/investigations/2026-03-24-vextract-index0-hang.md)
    Reader("vextract", "vextract.s32 r9, x4, r16", X, "i", "r9"),
    Reader("vldb_4x_src", "vldb.4x32.lo wl8, wl4", WL, "i", "wl8", gather=True),
]
LOAD_BY_NAME = {p.name: p for p in LOADS}
READER_BY_NAME = {c.name: c for c in READERS}

# Gather readers turn R into addresses: a load whose NEW data is not an address vector (the
# widening unpack, a gather result) would send them anywhere in the data memory.
GATHER_SAFE_LOADS = ("vlda_wl", "vldb_wl", "vlda_post", "vldb_2d", "vlda_128")


def pair_dom(p: Load, c: Reader) -> Optional[str]:
    if p.dom and c.dom and p.dom != c.dom:
        return None
    if not (p.halves & c.halves):
        return None
    if c.gather and p.name not in GATHER_SAFE_LOADS:
        return None
    return p.dom or c.dom or "i"


def valid_pairs() -> list[tuple[str, str]]:
    return [(p.name, c.name) for p in LOADS for c in READERS if pair_dom(p, c)]


def parse_pairs(specs: list[str]) -> list[tuple[str, str]]:
    valid = valid_pairs()
    out: list[tuple[str, str]] = []
    for spec in specs:
        pn, cn = spec.split(":")
        for pair in valid:
            if pn in ("*", pair[0]) and cn in ("*", pair[1]) and pair not in out:
                out.append(pair)
    return out


# Write-after-write partners: each writes R (a half or all of it) without reading it.
@dataclass(frozen=True)
class Writer:
    name: str
    asm: str
    halves: frozenset
    dom: Optional[str] = None


WRITERS = [
    Writer("vadd_32_w", "vadd.32 x4, x0, x1", X, "i"),
    Writer("vmov_w_w", "vmov wl4, wl0", WL),
    Writer("vmov_x_x", "vmov x4, x0", X),
    Writer("vbcst_32", "vbcst.32 x4, r12", X),
    Writer("vpack_w", "vpack.s8.s16 wl4, x0", WL, "i"),
    Writer("vsrs_w", "vsrs.s16.s32 wl4, bml2, s1", WL),
    Writer("vconv_w", "vconv.bf16.fp32 wl4, bml2", WL, "f"),
    Writer("vmov_x_bm", "vmov x4, bml2", X),
    Writer("vlda_alt", "vlda wl4, [p0, #96]", WL),       # X1, low half
    Writer("vldb_alt_wh", "vldb wh4, [p0, #96]", WH),    # X1 (vldb reaches [-128, 127])
]
WRITER_BY_NAME = {w.name: w for w in WRITERS}


def waw_dom(a: Load, b: Writer) -> Optional[str]:
    if a.dom and b.dom and a.dom != b.dom:
        return None
    if not (a.halves & b.halves):
        return None
    return a.dom or b.dom or "i"


def valid_waw_pairs() -> list[tuple[str, str]]:
    return [(a.name, b.name) for a in LOADS for b in WRITERS if waw_dom(a, b)]


def parse_waw(specs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for spec in specs:
        an, bn = spec.split(":")
        for pair in valid_waw_pairs():
            if an in ("*", pair[0]) and bn in ("*", pair[1]) and pair not in out:
                out.append(pair)
    return out


# --------------------------------------------------------------------------
# Family 1 kernel text
# --------------------------------------------------------------------------

def address_vectors() -> list[str]:
    """Write ADDR_OLD / ADDR_NEW: eight copies each of the absolute address of OLD_i / LDV_i (the
    gather reduces each lane address to its element granule and bank, so equal lanes are fine)."""
    lines = []
    for item, target in (("ADDR_OLD", "OLD_i"), ("ADDR_NEW", "LDV_i")):
        # (llvm-mc silently truncates an `add` immediate above its 6-bit field: go through padda)
        lines += point_to("p2", target) + nops(2) + ["  mov r14, p2"]
        lines += point_to("p2", item) + nops(2)
        lines += [f"  st r14, [p2, #{4 * k}]" for k in range(8)]
    return lines + nops(4)


def prologue() -> list[str]:
    lines = base.prologue()
    lines += ["  // ==== vector-load probe prologue ===="]
    lines += address_vectors()
    # 2D descriptor d0 all zero (the load stays on its item); vbcst source; vextract index
    lines += ["  mov m0, #0", "  mov dn0, #0", "  mov dj0, #0", "  mov dc0, #0",
              "  movxm r12, #0x5a5a1234", "  mov r16, #2"]
    lines += load_vec("wl9", "ADDR_NEW", 0)
    lines += nops(LOAD_NOPS)
    return lines


def new_item(c: Optional[Reader], dom: str) -> str:
    return "ADDR_NEW" if c is not None and c.gather else f"LDV_{dom}"


def old_item(c: Optional[Reader], dom: str) -> str:
    return "ADDR_OLD" if c is not None and c.gather else f"OLD_{dom}"


def reset_r(c: Optional[Reader], dom: str) -> list[str]:
    """R back to OLD, p5 back to the loaded item; settled before the pair."""
    item = old_item(c, dom)
    lines = load_vec("wl4", item, 0)
    lines += load_vec("wh4", item, 1) if item != "ADDR_OLD" else load_vec("wh4", "OLD_i", 1)
    lines += point_to("p5", new_item(c, dom))
    lines += nops(SETTLE_BEFORE)
    return lines


def observation(out: str) -> list[str]:
    if out == "mem":
        return []
    if out == "cm6":
        return [f"  vst {q}6, [p1, #{32 * k}]" for k, q in enumerate(("amll", "amlh", "amhl", "amhh"))]
    if out == "bml6":
        return ["  vst amll6, [p1, #0]", "  vst amlh6, [p1, #32]"]
    if out == "x8":
        return ["  vst wl8, [p1, #0]", "  vst wh8, [p1, #32]"]
    if out == "wl8":
        return ["  vst wl8, [p1, #0]"]
    if out == "r9":
        return ["  st r9, [p1, #0]"]
    raise ValueError(out)


def advance_record() -> list[str]:
    return nops(4) + [f"  padda [p1], #{RECORD_BYTES}"] + nops(4)


def point_lines(p: Load, c: Reader, direction: str, d: int, dom: str) -> list[str]:
    lines = [f"  // ---- point: {p.name} -> {c.name} {direction} d={d} ----"]
    lines += reset_r(c, dom)
    p_asm, c_asm = "  " + p.asm, "  " + c.asm
    if direction == "RAW":
        if d < 1:
            raise ValueError("RAW needs d >= 1")
        lines += [p_asm] + nops(d - 1) + [c_asm]
    elif direction == "WAR":
        lines += [f"  {c.asm} ; {p.asm}"] if d == 0 else [c_asm] + nops(d - 1) + [p_asm]
    else:
        raise ValueError(direction)
    lines += nops(SETTLE_AFTER) + observation(c.out) + advance_record()
    return lines


RAW_DISTANCES = tuple(range(1, 10))
WAR_DISTANCES = (0, 1, 2)       # a reader issued before a load can only see OLD; controls


def sweep_points() -> list[tuple[str, int]]:
    pts = [("WAR", REF_DISTANCE), ("RAW", REF_DISTANCE)]
    pts += [("RAW", d) for d in RAW_DISTANCES]
    pts += [("WAR", d) for d in WAR_DISTANCES]
    return pts


def pair_points(p: Load, c: Reader) -> list[tuple[str, int]]:
    return [(dr, d) for dr, d in sweep_points()
            if not (dr == "WAR" and d == 0 and not base.can_share_bundle(c.asm, p.asm))]


def pair_preamble(dom: str) -> list[str]:
    return base.pair_preamble(dom)


def kernel_text(pairs: list[tuple[str, str]]) -> tuple[str, list[dict]]:
    points: list[dict] = []
    test_points = ["\n".join(prologue())]
    for pn, cn in pairs:
        p, c = LOAD_BY_NAME[pn], READER_BY_NAME[cn]
        dom = pair_dom(p, c)
        assert dom, (pn, cn)
        test_points.append("\n".join([f"  // ==== pair {pn} -> {cn} (dom {dom}) ===="]
                                     + pair_preamble(dom)))
        for direction, d in pair_points(p, c):
            test_points.append("\n".join(point_lines(p, c, direction, d, dom)))
            points.append({"producer": pn, "consumer": cn, "dir": direction, "d": d,
                           "record": len(points)})
    return isa_gen.build_mega_program(test_points, max(4, len(points) * RECORD_BYTES)), points


WAW_DISTANCES = tuple(range(-9, 10))


def waw_sweep() -> list[int]:
    return [-REF_DISTANCE, REF_DISTANCE] + list(WAW_DISTANCES)


def waw_point_lines(a: Load, b: Writer, s: int, dom: str) -> list[str]:
    lines = [f"  // ---- point: {a.name} after {b.name} s={s} ----"]
    lines += reset_r(None, dom)
    first, second = (b, a) if s >= 0 else (a, b)
    if s == 0:
        lines.append(f"  {b.asm} ; {a.asm}")
    else:
        lines += ["  " + first.asm] + nops(abs(s) - 1) + ["  " + second.asm]
    lines += nops(SETTLE_AFTER)
    lines += ["  vst wl4, [p1, #0]", "  vst wh4, [p1, #32]"]
    return lines + advance_record()


def waw_kernel_text(pairs: list[tuple[str, str]]) -> tuple[str, list[dict]]:
    points: list[dict] = []
    test_points = ["\n".join(prologue())]
    for an, bn in pairs:
        a, b = LOAD_BY_NAME[an], WRITER_BY_NAME[bn]
        dom = waw_dom(a, b)
        assert dom, (an, bn)
        test_points.append("\n".join([f"  // ==== WAW {an} vs {bn} (dom {dom}) ===="]
                                     + pair_preamble(dom)))
        for s in waw_sweep():
            if s == 0 and not base.can_share_bundle(b.asm, a.asm):
                continue
            test_points.append("\n".join(waw_point_lines(a, b, s, dom)))
            points.append({"producer": an, "consumer": bn, "dir": "WAW", "d": s,
                           "record": len(points)})
    return isa_gen.build_mega_program(test_points, max(4, len(points) * RECORD_BYTES)), points


def chunk(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def generate(out_dir: str, pairs, waw_pairs, pairs_per_kernel: int = 3) -> dict:
    """Producer/consumer kernels `k<n>`, write-after-write kernels `w<n>`."""
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(f"k{k:03d}", kernel_text, g) for k, g in enumerate(chunk(pairs, pairs_per_kernel))]
    jobs += [(f"w{k:03d}", waw_kernel_text, g)
             for k, g in enumerate(chunk(waw_pairs, pairs_per_kernel))]
    kernels = []
    for name, text_of, group in jobs:
        text, points = text_of(group)
        entry = {"name": name, "points": points, "in_size": IN_SIZE,
                 "out_size": max(4, len(points) * RECORD_BYTES)}
        suite.write_single_core_case(out_dir, name, text, input_bytes=TABLE_BYTES,
                                     out_bytes=entry["out_size"], meta=entry)
        kernels.append(entry)
    meta = {"in_size": IN_SIZE, "record_bytes": RECORD_BYTES}
    suite.write_suite_json(out_dir, "vec-load-timing", "gen_vec_load_timing_probe.py",
                           [k["name"] for k in kernels], meta=meta)
    return {**meta, "kernels": kernels}


# --------------------------------------------------------------------------
# Family 3: two instructions in one bundle that TableGen keeps apart (same-bundle hazards)
# --------------------------------------------------------------------------
#
# The family-1 sweep met two: a `vldb.4x` gather issued with a reader of R (its address read
# and the reader's read share a read port), and a gather issued with a `vlda` (both use load
# unit A). Each point here issues a first instruction F (writing wl5, reading its own source)
# and a second S (reading or writing R) in one bundle, next to F alone and S alone, and stores
# S's output, wl5 and R. Record: S output [0, 128) | wl5 [128, 160) | wl4 [160, 192) |
# wh4 [192, 224).

HAZARD_RECORD_BYTES = 256

HAZARD_FIRSTS = [
    ("g4x_lo", "vldb.4x32.lo wl5, wl9"),
    ("g4x_hi", "vldb.4x32.hi wl5, wl9"),
    ("g4x16_lo", "vldb.4x16.lo wl5, wl9"),
    ("vpack_x0", "vpack.s8.s16 wl5, x0"),
    ("vldb_plain", "vldb wl5, [p0, #64]"),
]
# Second instructions: every family-1 reader of R plus two loads into R.
HAZARD_SECONDS = [(c.name, c.asm, c.out) for c in READERS if not c.gather] + [
    ("vlda_alt", "vlda wl4, [p0, #96]", "none"),
    ("vldb_alt", "vldb wl4, [p0, #96]", "none"),
    # other holders of load unit A in the gather's cycle
    ("lda_r9", "lda r9, [p0, #96]", "r9"),
    ("vlda_128", "vlda.128 wl4, [p0, #96]", "none"),
    ("vlda_amll", "vlda amll6, [p0, #96]", "bml6"),
    ("vlda_ups", "vlda.ups.s32.s16 bml6, s3, [p0, #96]", "bml6"),
]


def hazard_pairs() -> list[tuple[str, str]]:
    return [(f, s) for f, fa in HAZARD_FIRSTS for s, sa, _ in HAZARD_SECONDS
            if base.can_share_bundle(fa, sa)]


def hazard_point_lines(first: Optional[str], second: Optional[tuple], label: str) -> list[str]:
    lines = [f"  // ---- point: {label} ----"]
    lines += load_vec("wl4", "OLD_i", 0) + load_vec("wh4", "OLD_i", 1)
    lines += load_vec("wl5", "S_i", 0)
    lines += nops(SETTLE_BEFORE)
    ops = ([first] if first else []) + ([second[1]] if second else [])
    lines.append("  " + " ; ".join(ops))
    lines += nops(SETTLE_AFTER)
    out = second[2] if second else "none"
    lines += [] if out == "none" else observation(out)
    lines += ["  vst wl5, [p1, #128]", "  vst wl4, [p1, #160]", "  vst wh4, [p1, #192]"]
    return lines + nops(4) + [f"  padda [p1], #{HAZARD_RECORD_BYTES}"] + nops(4)


def hazard_kernel_text(pairs: list[tuple[str, str]]) -> tuple[str, list[dict]]:
    firsts = dict(HAZARD_FIRSTS)
    seconds = {s[0]: s for s in HAZARD_SECONDS}
    points: list[dict] = []
    test_points = ["\n".join(prologue() + pair_preamble("i"))]
    for fn, sn in pairs:
        for mode in ("first", "second", "both"):
            f = firsts[fn] if mode in ("first", "both") else None
            sec = seconds[sn] if mode in ("second", "both") else None
            test_points.append("\n".join(hazard_point_lines(f, sec, f"{fn} {sn} {mode}")))
            points.append({"first": fn, "second": sn, "mode": mode, "record": len(points)})
    return isa_gen.build_mega_program(test_points, len(points) * HAZARD_RECORD_BYTES), points


def generate_hazard(out_dir: str, pairs_per_kernel: int = 12) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    kernels = []
    for k, group in enumerate(chunk(hazard_pairs(), pairs_per_kernel)):
        name = f"h{k:03d}"
        text, points = hazard_kernel_text(group)
        entry = {"name": name, "points": points, "in_size": IN_SIZE,
                 "out_size": len(points) * HAZARD_RECORD_BYTES}
        suite.write_single_core_case(out_dir, name, text, input_bytes=TABLE_BYTES,
                                     out_bytes=entry["out_size"], meta=entry)
        kernels.append(entry)
    meta = {"in_size": IN_SIZE, "record_bytes": HAZARD_RECORD_BYTES}
    suite.write_suite_json(out_dir, "vec-load-hazard", "gen_vec_load_timing_probe.py",
                           [k["name"] for k in kernels], meta=meta)
    return {**meta, "kernels": kernels}


# --------------------------------------------------------------------------
# Family 4: a load and another vector-file write landing in one cycle, any register
# --------------------------------------------------------------------------
#
# The write-after-write sweep only pairs writers of R. Here a load L (bundle 0) and a writer W
# (issued so that it lands with L) target the same or different registers and banks; the
# record holds wl4, wh4, wl8, wh8 after the tie, next to L alone and W alone.

TIE_LOADS = [
    ("vldb_wl", "vldb wl4, [p5, #0]"),
    ("vldb_wh", "vldb wh4, [p5, #32]"),
    ("vldb_unpack", "vldb.unpack.s16.s8 x4, [p5, dj0]"),
    ("vldb_4x", "vldb.4x32.lo wl4, wl9"),
    ("vlda_wl", "vlda wl4, [p5, #0]"),
]
# (name, asm, result cycle)
TIE_WRITERS = [
    ("vmov_wl8", "vmov wl8, wl0", 2),
    ("vmov_wh8", "vmov wh8, wl0", 2),
    ("vadd_x8", "vadd.32 x8, x0, x1", 2),
    ("vpack_wl8", "vpack.s8.s16 wl8, x0", 2),
    ("vmov_wl4", "vmov wl4, wl0", 2),
    ("vmov_wh4", "vmov wh4, wl0", 2),
    ("vadd_x4", "vadd.32 x4, x0, x1", 2),
    ("vsrs_wl8", "vsrs.s16.s32 wl8, bml2, s1", 4),
]
TIE_LOAD_CYCLE = 7


def tie_point_lines(load: Optional[str], writer: Optional[tuple], label: str) -> list[str]:
    lines = [f"  // ---- point: {label} ----"]
    lines += load_vec("wl4", "OLD_i", 0) + load_vec("wh4", "OLD_i", 1)
    lines += load_vec("wl8", "S_i", 0) + load_vec("wh8", "S_i", 1)
    lines += point_to("p5", "LDV_i") + nops(SETTLE_BEFORE)
    gap = TIE_LOAD_CYCLE - writer[2] if writer else 0
    for b in range(gap + 1):
        ops = ([load] if load and b == 0 else []) + ([writer[1]] if writer and b == gap else [])
        lines.append("  " + (" ; ".join(ops) if ops else "nop"))
    lines += nops(SETTLE_AFTER)
    lines += ["  vst wl4, [p1, #0]", "  vst wh4, [p1, #32]", "  vst wl8, [p1, #64]", "  vst wh8, [p1, #96]"]
    return lines + advance_record()


def tie_kernel_text(pairs) -> tuple[str, list[dict]]:
    loads, writers = dict(TIE_LOADS), {w[0]: w for w in TIE_WRITERS}
    points: list[dict] = []
    test_points = ["\n".join(prologue() + pair_preamble("i"))]
    for ln, wn in pairs:
        for mode in ("load", "writer", "both"):
            ld = loads[ln] if mode in ("load", "both") else None
            wr = writers[wn] if mode in ("writer", "both") else None
            test_points.append("\n".join(tie_point_lines(ld, wr, f"{ln} {wn} {mode}")))
            points.append({"load": ln, "writer": wn, "mode": mode, "record": len(points)})
    return isa_gen.build_mega_program(test_points, len(points) * RECORD_BYTES), points


def generate_tie(out_dir: str, pairs_per_kernel: int = 10) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    pairs = [(ln, wn) for ln, _ in TIE_LOADS for wn, _, _ in TIE_WRITERS]
    kernels = []
    for k, group in enumerate(chunk(pairs, pairs_per_kernel)):
        name = f"t{k:03d}"
        text, points = tie_kernel_text(group)
        entry = {"name": name, "points": points, "in_size": IN_SIZE,
                 "out_size": len(points) * RECORD_BYTES}
        suite.write_single_core_case(out_dir, name, text, input_bytes=TABLE_BYTES,
                                     out_bytes=entry["out_size"], meta=entry)
        kernels.append(entry)
    meta = {"in_size": IN_SIZE, "record_bytes": RECORD_BYTES}
    suite.write_suite_json(out_dir, "vec-load-tie", "gen_vec_load_timing_probe.py",
                           [k["name"] for k in kernels], meta=meta)
    return {**meta, "kernels": kernels}


# --------------------------------------------------------------------------
# Family 2: loads across a taken branch
# --------------------------------------------------------------------------

BRANCH_SLOT_BYTES = 32
READERS_PER_POINT = 8
BRANCH_RECORD_BYTES = BRANCH_SLOT_BYTES * READERS_PER_POINT
DELAY_SLOTS = 5
LOAD_POSITIONS = tuple(range(-3, DELAY_SLOTS + 1))     # relative to the branch bundle
REF_NEW_POSITION = -16                                  # far enough that every reader sees NEW
KINDS = ("none", "j", "jnz_t", "jnz_n", "jl", "ret")
# p2 = p1 + 128: a quarter `vst` reaches [-128, 127], so the eight slots sit at p2 - 128 + 32j.
SLOT_BASE = -128


@dataclass(frozen=True)
class BranchFamily:
    name: str
    load: str                  # the load into the observed register
    reset: tuple               # lines that put OLD into it (settled afterwards)
    item: str                  # what p5 addresses
    reader: str                # reader j, `{J}` = its index, `{S}` = its slot offset from p2
    store: str = ""            # for indirect readers: stores the copy of reader j afterwards


def _acc_reset() -> tuple:
    lines = base.load_acc_pair("0", "OLD_f", "s3") + base.load_acc_pair("4", "T_f", "s3")
    return tuple(ln.strip() for ln in lines)


_ACC_DESTS = ("bml1", "bmh1", "bml3", "bmh3", "bml5", "bmh5", "bml7", "bmh7")
_ACC_QUARTERS = ("amll1", "amhl1", "amll3", "amhl3", "amll5", "amhl5", "amll7", "amhl7")
_W_DESTS = ("wl0", "wh0", "wl1", "wh1", "wl2", "wh2", "wl3", "wh3")

BRANCH_FAMILIES = [
    BranchFamily("scalar_st", "lda r4, [p5, #0]", ("movxm r4, #0x1111",), "LD_i",
                 "st r4, [p2, #{S}]"),
    BranchFamily("scalar_mov", "lda r4, [p5, #0]", ("movxm r4, #0x1111",), "LD_i",
                 "mov r{R}, r4", "st r{R}, [p2, #{S}]"),
    BranchFamily("vec_st", "vlda wl4, [p5, #0]",
                 tuple(ln.strip() for ln in load_vec("wl4", "OLD_i")), "LDV_i",
                 "vst wl4, [p2, #{S}]"),
    BranchFamily("vec_mov", "vlda wl4, [p5, #0]",
                 tuple(ln.strip() for ln in load_vec("wl4", "OLD_i")), "LDV_i",
                 "vmov {W}, wl4", "vst {W}, [p2, #{S}]"),
    BranchFamily("vecb_st", "vldb wl4, [p5, #0]",
                 tuple(ln.strip() for ln in load_vec("wl4", "OLD_i")), "LDV_i",
                 "vst wl4, [p2, #{S}]"),
    BranchFamily("acc_st", "vlda amll0, [p5, #0]", _acc_reset(), "LD_f",
                 "vst amll0, [p2, #{S}]"),
    BranchFamily("acc_vadd", "vlda amll0, [p5, #0]", _acc_reset(), "LD_f",
                 "vadd.f {A}, bml0, bml4, r3", "vst {Q}, [p2, #{S}]"),
    # controls with result latency 1: the emulator delayed pointer writes across a taken branch
    BranchFamily("ptr_lda", "lda p4, [p5, #0]", ("movxm p4, #0x1111",), "LD_i",
                 "mov r{R}, p4", "st r{R}, [p2, #{S}]"),
    BranchFamily("ptr_mov", "movxm p4, #0x2222", ("movxm p4, #0x1111",), "LD_i",
                 "mov r{R}, p4", "st r{R}, [p2, #{S}]"),
    BranchFamily("ptr_padda", "padda [p4], #16", ("movxm p4, #0x1111",), "LD_i",
                 "mov r{R}, p4", "st r{R}, [p2, #{S}]"),
    BranchFamily("scalar_alu", "add r4, r4, #7", ("movxm r4, #0x1111",), "LD_i",
                 "mov r{R}, r4", "st r{R}, [p2, #{S}]"),
]
BRANCH_FAMILY_BY_NAME = {f.name: f for f in BRANCH_FAMILIES}


def _fill(template: str, j: int) -> str:
    return template.format(J=j, S=SLOT_BASE + BRANCH_SLOT_BYTES * j, R=16 + j, W=_W_DESTS[j],
                           A=_ACC_DESTS[j], Q=_ACC_QUARTERS[j])


def branch_bundle(kind: str, label: str) -> str:
    return {"none": "nop", "j": f"j #{label}", "jnz_t": f"jnz r10, #{label}",
            "jnz_n": f"jnz r11, #{label}", "jl": f"jl #{label}", "ret": "ret lr"}[kind]


def branch_point_lines(fam: BranchFamily, kind: str, pos: Optional[int], label: str) -> list[str]:
    """One point: the load at `pos` (None: no load) around a `kind` branch to `label`."""
    lines = [f"  // ---- point: {fam.name} {kind} pos={pos} ----"]
    lines += ["  " + ln for ln in fam.reset]
    lines += point_to("p5", fam.item)
    lines += ["  mov p2, p1", "  padda [p2], #128"]
    if kind == "ret":
        lines.append(f"  movxm lr, #{label}")
    lines += nops(SETTLE_BEFORE)
    # bundles relative to the branch bundle: REF_NEW_POSITION .. -1, 0, 1..DELAY_SLOTS
    first = min(pos, -3) if pos is not None else -3
    for b in range(first, DELAY_SLOTS + 1):
        ops = []
        if b == pos:
            ops.append(fam.load)
        if b == 0:
            ops.append(branch_bundle(kind, label))
        ops = [o for o in ops if o != "nop"]
        lines.append("  " + (" ; ".join(ops) if ops else "nop"))
    if kind in ("j", "jnz_t", "jl", "ret"):
        # never executed: the target must sit on a 16-byte boundary (silicon ignores the low
        # four bits of a branch target)
        lines.append("  .p2align 4")
    lines.append(f"{label}:")
    lines += ["  " + _fill(fam.reader, j) for j in range(READERS_PER_POINT)]
    lines += nops(SETTLE_AFTER)
    if fam.store:
        lines += ["  " + _fill(fam.store, j) for j in range(READERS_PER_POINT)]
    lines += nops(4) + [f"  padda [p1], #{BRANCH_RECORD_BYTES}"] + nops(4)
    return lines


def branch_points(fam: BranchFamily, kind: str) -> list[Optional[int]]:
    """References first (no load: all OLD; the load long before: all NEW), then positions; the
    branch bundle itself only when the load can share it."""
    shares = kind == "none" or base.can_share_bundle(fam.load, branch_bundle(kind, "0"))
    return [None, REF_NEW_POSITION] + [p for p in LOAD_POSITIONS if p != 0 or shares]


def branch_prologue() -> list[str]:
    lines = ["  // ==== branch probe prologue ====", "  mov p3, p0", f"  padda [p3], #{P3_BASE}"]
    lines += [f"  movxm r3, #{base.config_word('r3')}", "  mov r10, #1", "  mov r11, #0",
              "  mov r14, #16", "  mov s3, r14"]
    # `jl` overwrites lr, which the kernel's own `ret` needs: kept in r15, restored at the end
    lines += ["  mov r15, lr"]
    return lines + nops(LOAD_NOPS)


def branch_epilogue() -> list[str]:
    return ["  // ==== restore lr ====", "  mov lr, r15"] + nops(8)


def branch_kernel_text(fam: BranchFamily, kind: str, prefix: str) -> tuple[str, list[dict]]:
    points = []
    test_points = ["\n".join(branch_prologue())]
    for k, pos in enumerate(branch_points(fam, kind)):
        label = f".L{prefix}_{k}"
        test_points.append("\n".join(branch_point_lines(fam, kind, pos, label)))
        points.append({"family": fam.name, "kind": kind, "pos": pos, "record": len(points)})
    test_points.append("\n".join(branch_epilogue()))
    text = isa_gen.build_mega_program(test_points, len(points) * BRANCH_RECORD_BYTES)
    # the target alignment is relative to the section: align the section itself
    text = text.replace(".text\n", ".text\n.p2align 4\n", 1)
    return text, points


def generate_branch(out_dir: str, families=None, kinds=KINDS) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    kernels = []
    for fam in families or BRANCH_FAMILIES:
        for kind in kinds:
            name = f"b_{fam.name}_{kind}"
            text, points = branch_kernel_text(fam, kind, name)
            entry = {"name": name, "points": points, "in_size": IN_SIZE,
                     "out_size": len(points) * BRANCH_RECORD_BYTES}
            suite.write_single_core_case(out_dir, name, text, input_bytes=TABLE_BYTES,
                                         out_bytes=entry["out_size"], meta=entry)
            kernels.append(entry)
    meta = {"in_size": IN_SIZE, "record_bytes": BRANCH_RECORD_BYTES,
            "readers_per_point": READERS_PER_POINT, "delay_slots": DELAY_SLOTS}
    suite.write_suite_json(out_dir, "branch-load-timing", "gen_vec_load_timing_probe.py",
                           [k["name"] for k in kernels], meta=meta)
    return {**meta, "kernels": kernels}


# --------------------------------------------------------------------------
# Family 2 analysis
# --------------------------------------------------------------------------

def reader_distance(pos: int, j: int) -> int:
    """Issue distance from the load to reader j: the target is DELAY_SLOTS + 1 bundles after the
    branch bundle in issue order."""
    return DELAY_SLOTS + 1 + j - pos


def classify_slots(rec: bytes, old: bytes, new: bytes) -> str:
    """One letter per reader slot: O, N, X (neither) or . (OLD and NEW agree: no information)."""
    out = []
    for j in range(READERS_PER_POINT):
        sl = slice(j * BRANCH_SLOT_BYTES, (j + 1) * BRANCH_SLOT_BYTES)
        o, n, w = old[sl], new[sl], rec[sl]
        out.append("." if o == n else "O" if w == o else "N" if w == n else "X")
    return "".join(out)


def analyze_kernel(entry: dict, data: bytes) -> list[dict]:
    recs = {pt["pos"]: data[pt["record"] * BRANCH_RECORD_BYTES:(pt["record"] + 1) * BRANCH_RECORD_BYTES]
            for pt in entry["points"]}
    old, new = recs[None], recs[REF_NEW_POSITION]
    rows = []
    for pos in (p for p in LOAD_POSITIONS if p in recs):
        cls = classify_slots(recs[pos], old, new)
        # threshold: the smallest distance from which every reader sees NEW
        t = None
        for j in reversed(range(READERS_PER_POINT)):
            if cls[j] != "N":
                break
            t = reader_distance(pos, j)
        ok = t is not None and all(c == "O" for j, c in enumerate(cls) if reader_distance(pos, j) < t)
        rows.append({"pos": pos, "cls": cls, "t": t, "ok": ok})
    return rows


def analyze_branch(d: str, tag: str) -> dict:
    s = suite.load_suite(d)
    out = {}
    for entry in s.case_entries():
        out[entry["name"]] = analyze_kernel(entry, s.output(tag, entry["name"]).read_bytes())
    return out


def print_branch(res: dict) -> None:
    for name, rows in res.items():
        desc = "  ".join(f"{r['pos']:+d}:{r['cls']}:t={r['t']}{'' if r['ok'] else '!'}" for r in rows)
        print(f"{name:24s} {desc}")


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("out_dir")
    g.add_argument("--pairs", nargs="*", default=["*:*"])
    g.add_argument("--waw", nargs="*", default=[], metavar="A:B")
    g.add_argument("--pairs-per-kernel", type=int, default=3)
    b = sub.add_parser("branch")
    b.add_argument("out_dir")
    b.add_argument("--families", nargs="*", default=None)
    b.add_argument("--kinds", nargs="*", default=list(KINDS))
    hz = sub.add_parser("hazard")
    hz.add_argument("out_dir")
    ti = sub.add_parser("tie")
    ti.add_argument("out_dir")
    a = sub.add_parser("analyze-branch")
    a.add_argument("dir")
    a.add_argument("tag")
    a.add_argument("--compare", metavar="TAG")
    sub.add_parser("list")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for p, c in valid_pairs():
            print(p, c)
        for a_, b_ in valid_waw_pairs():
            print("WAW", a_, b_)
        return 0
    if args.cmd == "generate":
        m = generate(args.out_dir, parse_pairs(args.pairs), parse_waw(args.waw), args.pairs_per_kernel)
    elif args.cmd == "hazard":
        m = generate_hazard(args.out_dir)
    elif args.cmd == "tie":
        m = generate_tie(args.out_dir)
    elif args.cmd == "branch":
        fams = [BRANCH_FAMILY_BY_NAME[f] for f in args.families] if args.families else None
        m = generate_branch(args.out_dir, fams, tuple(args.kinds))
    else:
        res = analyze_branch(os.path.abspath(args.dir), args.tag)
        print_branch(res)
        if args.compare:
            other = analyze_branch(os.path.abspath(args.dir), args.compare)
            diff = [(n, r["pos"], r["cls"], o["cls"]) for n in res for r, o in zip(res[n], other[n])
                    if r["cls"] != o["cls"]]
            for n, pos, x, y in diff:
                print(f"DIFF {n:24s} pos={pos:+d} {args.tag}={x} {args.compare}={y}")
            print(f"points differing: {len(diff)}")
            return 1 if diff else 0
        return 0
    print(f"{len(m['kernels'])} kernels, {sum(len(k['points']) for k in m['kernels'])} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
