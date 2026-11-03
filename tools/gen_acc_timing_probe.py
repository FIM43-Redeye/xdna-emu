#!/usr/bin/env python3
"""Hand-scheduled accumulator read/write timing probe for the AIE2 vector unit.

Why this exists
---------------
Compiled kernels reschedule, so they pin an accumulator read or write cycle only
where one particular schedule happens to depend on it. This probe is raw assembly
with explicit nop padding: for a producer P (an instruction that writes the
contested accumulator register R) and a consumer C (an instruction that reads R),
it issues the two a controlled number of bundles apart and records which value C
saw, the OLD one that R held before P, or the NEW one P wrote.

  RAW point (d >= 1): P in bundle 0, C in bundle d.
  WAR point (d >= 0): C in bundle 0, P in bundle d (d = 0 shares one bundle).

With the write landing cycle w_P and the read cycle r_C of the two instructions
(both counted from their own issue), C sees NEW exactly when
issue(C) + r_C >= issue(P) + w_P, so the OLD/NEW threshold of a pair is
t = w_P - r_C regardless of direction. Every pair is swept over d in both
directions and two reference points per pair (d = REF_DISTANCE) give the
all-OLD (WAR) and all-NEW (RAW) records, so no value is computed by a model:
each output word is classified by comparing it with those two records.

The input table (operands, masks, OLD/NEW source patterns) is the same for every
kernel. A kernel packs several pairs; each point writes one RECORD_BYTES record.

Usage
-----
  python3 tools/gen_acc_timing_probe.py generate <out-dir> [--pairs SPEC ...]
  python3 -m harness suite build|hw|emu|compare <out-dir> ...   (PYTHONPATH=tools)
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from harness import suite                                   # noqa: E402

isa_gen = importlib.import_module("isa-test-gen")
instr_gen = importlib.import_module("instr-test-gen")

RECORD_BYTES = 128          # one point: up to a full 1024-bit accumulator
RECORD_WORDS = RECORD_BYTES // 4
REF_DISTANCE = 14           # far enough that both directions are settled
RAW_DISTANCES = tuple(range(1, 10))
WAR_DISTANCES = tuple(range(0, 10))
SETTLE_BEFORE = 9           # bundles between the last setup op and the pair
SETTLE_AFTER = 14           # bundles between the pair and the observation
LOAD_NOPS = isa_gen.LOAD_LATENCY

# MAC config words (aiev2_vmult.h layout, see isa-test-gen._make_vmac_config).
# (amode, bmode, variant): dense i16xi16 acc64, sparse i16xi16 acc64, dense bf16, sparse bf16.
CONFIGS = {
    "r1": (1, 3, 0),
    "r2": (1, 3, 5),
    "r3": (2, 3, 0),
    "r4": (2, 3, 2),
}


def config_word(key: str) -> int:
    amode, bmode, variant = CONFIGS[key]
    return isa_gen._make_vmac_config(amode, bmode, variant, sgn_x=1, sgn_y=1, zero_acc=0)


# --------------------------------------------------------------------------
# Input table
# --------------------------------------------------------------------------

def _u16s(values) -> bytes:
    return b"".join(struct.pack("<H", v & 0xFFFF) for v in values)


def _u32s(values) -> bytes:
    return b"".join(struct.pack("<I", v & 0xFFFFFFFF) for v in values)


def _int_pattern(base: int) -> bytes:
    """32 int16 elements: even positions are small distinct ints, odd positions 0, so that the
    same bits read as 16 acc32 lanes or as 16 acc64 lanes (value base+k, high word 0)."""
    return _u16s([(base + k // 2) if k % 2 == 0 else 0 for k in range(32)])


def _float_pattern(base: int) -> bytes:
    """32 bf16 bit patterns; vups shifts them left by 16 into finite fp32 lanes."""
    return _u16s([base + k for k in range(32)])


# name -> (alignment, bytes). Patterns feed `vups` (16-bit elements become 32-bit lanes).
TABLE_ITEMS = [
    ("X0", _u16s([1 + (k % 7) for k in range(32)])),
    ("X1", _u16s([2 + (k % 5) for k in range(32)])),
    ("X2", _u16s([0x3F80 + (k % 8) for k in range(32)])),
    ("X3", _u16s([0x3F80 + 2 * (k % 4) for k in range(32)])),
    ("Y3", _u16s([0x3F80 + (k % 16) for k in range(64)])),
    ("OLD_i", _int_pattern(100)),
    ("S_i", _int_pattern(1000)),
    ("T_i", _int_pattern(5000)),
    ("U_i", _int_pattern(20000)),
    ("OLD_f", _float_pattern(0x4300)),
    ("S_f", _float_pattern(0x3F80)),
    ("T_f", _float_pattern(0x3FA0)),
    ("U_f", _float_pattern(0x4200)),
    ("NEWX_i", _u32s([0x200 + k if k % 2 == 0 else 0 for k in range(16)])),
    ("NEWX_f", _u32s([(0x3FC0 + k) << 16 for k in range(16)])),
]
MASK_ITEMS = [
    ("Q1", bytes((0x5A + 37 * k) & 0xFF for k in range(16))),
    ("Q3", bytes((0xA5 + 29 * k) & 0xFF for k in range(16))),
]


def build_table() -> tuple[bytes, dict[str, int]]:
    """Return (input bytes, name -> byte offset). Masks first (lda q scale 16 reaches +496)."""
    offsets: dict[str, int] = {}
    buf = bytearray()
    for name, data in MASK_ITEMS:
        offsets[name] = len(buf)
        buf += data
    while len(buf) % 32:
        buf.append(0)
    for name, data in TABLE_ITEMS:
        offsets[name] = len(buf)
        buf += data
        while len(buf) % 32:
            buf.append(0)
    return bytes(buf), offsets


TABLE_BYTES, TABLE_OFFSETS = build_table()
IN_SIZE = (len(TABLE_BYTES) + 31) // 32 * 32
# Offsets above the vlda immediate range go through p3 = p0 + P3_BASE.
P3_BASE = 512


def table_ref(name: str, extra: int = 0) -> tuple[str, int]:
    off = TABLE_OFFSETS[name] + extra
    if off <= isa_gen.MAX_VECTOR_OFFSET:
        return "p0", off
    return "p3", off - P3_BASE


# --------------------------------------------------------------------------
# Instruction catalogue
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Producer:
    name: str
    asm: str
    dom: Optional[str]      # 'i' integer data, 'f' float data, None = either


@dataclass(frozen=True)
class Consumer:
    name: str
    asm: str
    dom: Optional[str]
    out: str                # what to store afterwards: cm6 bml6 x8 wl8 mem R
    dest_reinit: Optional[str] = None   # a tied destination that must start from U each point
    width_f: Optional[str] = None       # for out == "R": the accumulator R as float (bml0) or int (cm0)


PRODUCERS = [
    Producer("vmul_i_dense", "vmul cm0, x0, x1, r1", "i"),
    Producer("vmul_i_sparse", "vmul cm0, x0, qx1, r2", "i"),
    Producer("vmul_f_dense", "vmul.f bml0, x2, x3, r3", "f"),
    Producer("vmul_f_sparse", "vmul.f bml0, y3, qx3, r4", "f"),
    Producer("vmac_i_dense", "vmac cm0, cm2, x0, x1, r1", "i"),
    Producer("vmac_i_sparse", "vmac cm0, cm2, x0, qx1, r2", "i"),
    Producer("vmac_f_dense", "vmac.f bml0, bml2, x2, x3, r3", "f"),
    Producer("vmac_f_sparse", "vmac.f bml0, bml2, y3, qx3, r4", "f"),
    Producer("vaddmac_i_dense", "vaddmac cm0, cm0, cm4, x0, x1, r1", "i"),
    Producer("vaddmac_i_sparse", "vaddmac cm0, cm0, cm4, x0, qx1, r2", "i"),
    Producer("vaddmac_f_dense", "vaddmac.f bml0, bml0, bml4, x2, x3, r3", "f"),
    Producer("vaddmac_f_sparse", "vaddmac.f bml0, bml0, bml4, y3, qx3, r4", "f"),
    Producer("vaddmsc_f_dense", "vaddmsc.f bml0, bml0, bml4, x2, x3, r3", "f"),
    Producer("vaddmsc_f_sparse", "vaddmsc.f bml0, bml0, bml4, y3, qx3, r4", "f"),
    Producer("vmov_cm_cm", "vmov cm0, cm2", None),
    Producer("vmov_bm_bm", "vmov bml0, bml2", None),
    Producer("vmov_bm_x", "vmov bml0, x10", None),
    Producer("vclr_cm", "vclr cm0", None),
    Producer("vadd_f", "vadd.f bml0, bml2, bml4, r3", "f"),
    Producer("vadd_i", "vadd cm0, cm2, cm4, r1", "i"),
    Producer("vups", "vups.s32.s16 bml0, wl10, s3", None),
    Producer("vlda_ups", "vlda.ups.s32.s16 bml0, s3, [p4, #0]", None),
]

CONSUMERS = [
    Consumer("vmac_i_dense_acc1", "vmac cm6, cm0, x0, x1, r1", "i", "cm6"),
    Consumer("vmac_i_sparse_acc1", "vmac cm6, cm0, x0, qx1, r2", "i", "cm6"),
    Consumer("vmac_f_dense_acc1", "vmac.f bml6, bml0, x2, x3, r3", "f", "bml6"),
    Consumer("vmac_f_sparse_acc1", "vmac.f bml6, bml0, y3, qx3, r4", "f", "bml6"),
    Consumer("vaddmac_i_dense_acc2", "vaddmac cm6, cm6, cm0, x0, x1, r1", "i", "cm6", "cm6"),
    Consumer("vaddmac_i_sparse_acc2", "vaddmac cm6, cm6, cm0, x0, qx1, r2", "i", "cm6", "cm6"),
    Consumer("vaddmac_f_dense_acc2", "vaddmac.f bml6, bml6, bml0, x2, x3, r3", "f", "bml6", "bml6"),
    Consumer("vaddmac_f_sparse_acc2", "vaddmac.f bml6, bml6, bml0, y3, qx3, r4", "f", "bml6", "bml6"),
    # The tied acc1 read: the instruction overwrites R itself, so only RAW points carry information.
    Consumer("vaddmac_i_dense_acc1", "vaddmac cm0, cm0, cm4, x0, x1, r1", "i", "R", None, "i"),
    Consumer("vaddmac_i_sparse_acc1", "vaddmac cm0, cm0, cm4, x0, qx1, r2", "i", "R", None, "i"),
    Consumer("vaddmac_f_dense_acc1", "vaddmac.f bml0, bml0, bml4, x2, x3, r3", "f", "R", None, "f"),
    Consumer("vaddmac_f_sparse_acc1", "vaddmac.f bml0, bml0, bml4, y3, qx3, r4", "f", "R", None, "f"),
    Consumer("vsrs_s16_s32", "vsrs.s16.s32 wl8, bml0, s1", None, "wl8"),
    Consumer("vsrs_s32_s64", "vsrs.s32.s64 x8, cm0, s1", "i", "x8"),
    Consumer("vmov_x_bm", "vmov x8, bml0", None, "x8"),
    Consumer("vmov_cm_cm", "vmov cm6, cm0", None, "cm6"),
    Consumer("vmov_bm_bm", "vmov bml6, bml0", None, "bml6"),
    Consumer("vst_amll", "vst amll0, [p1, #0]", None, "mem"),
    Consumer("vst_amlh", "vst amlh0, [p1, #0]", None, "mem"),
    Consumer("vst_amhl", "vst amhl0, [p1, #0]", None, "mem"),
    Consumer("vst_amhh", "vst amhh0, [p1, #0]", None, "mem"),
    Consumer("vadd_f_acc1", "vadd.f bml6, bml0, bml4, r3", "f", "bml6"),
    Consumer("vadd_f_acc2", "vadd.f bml6, bml4, bml0, r3", "f", "bml6"),
    Consumer("vadd_i_acc1", "vadd cm6, cm0, cm4, r1", "i", "cm6"),
    Consumer("vadd_i_acc2", "vadd cm6, cm4, cm0, r1", "i", "cm6"),
    Consumer("vconv_bf16_fp32", "vconv.bf16.fp32 wl8, bml0", "f", "wl8"),
    Consumer("vsrsm_s16_s32", "vsrsm.s16.s32 wl8, bml0, s1", None, "wl8"),
    Consumer("vsrs_d16_s32", "vsrs.d16.s32 wl8, bml0, s1", None, "wl8"),
    Consumer("vst_srs_s16_s32", "vst.srs.s16.s32 bml0, s1, [p1, #0]", None, "mem"),
]
PRODUCER_BY_NAME = {p.name: p for p in PRODUCERS}
CONSUMER_BY_NAME = {c.name: c for c in CONSUMERS}


def pair_dom(p: Producer, c: Consumer) -> Optional[str]:
    """The data domain of a pair, or None if the two instructions need different ones."""
    if p.dom and c.dom and p.dom != c.dom:
        return None
    return p.dom or c.dom or "i"


def valid_pairs() -> list[tuple[str, str]]:
    return [(p.name, c.name) for p in PRODUCERS for c in CONSUMERS if pair_dom(p, c)]


# --------------------------------------------------------------------------
# Kernel text
# --------------------------------------------------------------------------

def nops(n: int) -> list[str]:
    return ["  nop"] * n


def load_vec(reg: str, name: str, half: int = 0) -> list[str]:
    ptr, off = table_ref(name, 32 * half)
    return [f"  vlda {reg}, [{ptr}, #{off}]"]


def load_acc_pair(first: str, name: str, shift_reg: str) -> list[str]:
    """Load a table pattern into the pair {bml, bmh} of index `first` (cm<first>) through vups."""
    lines = []
    lines += load_vec("wl11", name, 0)
    lines += load_vec("wh11", name, 1)
    lines += nops(LOAD_NOPS)
    lines.append(f"  vups.s32.s16 bml{first}, wl11, {shift_reg}")
    lines.append(f"  vups.s32.s16 bmh{first}, wh11, {shift_reg}")
    lines += nops(6)
    return lines


def operand_load_lines() -> list[str]:
    lines = []
    for name, reg in (("X0", "x0"), ("X1", "x1"), ("X2", "x2"), ("X3", "x3")):
        idx = reg[1:]
        lines += load_vec(f"wl{idx}", name, 0)
        lines += load_vec(f"wh{idx}", name, 1)
    for k in range(4):                      # y3 = {x6, x7}
        reg = ("wl6", "wh6", "wl7", "wh7")[k]
        lines += load_vec(reg, "Y3", k)
    for name, q in (("Q1", "q1"), ("Q3", "q3")):
        lines.append(f"  lda {q}, [p0, #{TABLE_OFFSETS[name]}]")
    lines += nops(LOAD_NOPS)
    return lines


def prologue() -> list[str]:
    lines = ["  // ==== probe prologue ====", "  mov p3, p0", f"  padda [p3], #{P3_BASE}"]
    for reg in CONFIGS:
        lines.append(f"  movxm {reg}, #{config_word(reg)}")
    lines += nops(3)
    lines += operand_load_lines()
    return lines


def pair_preamble(dom: str) -> list[str]:
    """Per pair: shifts, the constant accumulators cm2 (S) and cm4 (T), and x10 (NEWX)."""
    load_shift = 0 if dom == "i" else 16
    srs_shift = 0 if dom == "i" else 16
    lines = [
        f"  mov r14, #{load_shift}", "  mov s3, r14",
        f"  mov r14, #{srs_shift}", "  mov s1, r14",
        "  nop", "  nop",
    ]
    lines += load_acc_pair("2", f"S_{dom}", "s3")
    lines += load_acc_pair("4", f"T_{dom}", "s3")
    lines += load_vec("wl10", f"NEWX_{dom}", 0)
    lines += load_vec("wh10", f"NEWX_{dom}", 1)
    # p4 addresses the NEWX item for the fused load-and-ups producer.
    lines += ["  mov p4, p0", f"  padda [p4], #{TABLE_OFFSETS[f'NEWX_{dom}']}"]
    lines += nops(LOAD_NOPS)
    return lines


def point_setup(c: Consumer, dom: str) -> list[str]:
    shift = "s3"
    lines = load_acc_pair("0", f"OLD_{dom}", shift)
    if c.dest_reinit:
        lines += load_acc_pair("6", f"U_{dom}", shift)
    lines += nops(SETTLE_BEFORE)
    return lines


def observation(c: Consumer, dom: str) -> list[str]:
    """Store the consumer's output into the current record."""
    lines = []
    out = c.out
    if out == "mem":
        return lines
    if out in ("cm6", "R"):
        base = "6" if out == "cm6" else "0"
        quarters = ["amll", "amlh", "amhl", "amhh"]
        if out == "R" and c.width_f == "f":
            quarters = quarters[:2]
        for k, q in enumerate(quarters):
            lines.append(f"  vst {q}{base}, [p1, #{32 * k}]")
    elif out == "bml6":
        lines.append("  vst amll6, [p1, #0]")
        lines.append("  vst amlh6, [p1, #32]")
    elif out == "x8":
        lines.append("  vst wl8, [p1, #0]")
        lines.append("  vst wh8, [p1, #32]")
    elif out == "wl8":
        lines.append("  vst wl8, [p1, #0]")
    else:
        raise ValueError(out)
    return lines


def point_lines(p: Producer, c: Consumer, direction: str, d: int, dom: str) -> list[str]:
    lines = [f"  // ---- point: {p.name} -> {c.name} {direction} d={d} ----"]
    lines += point_setup(c, dom)
    p_asm, c_asm = "  " + p.asm, "  " + c.asm
    if direction == "CONLY":
        lines.append(c_asm)
    elif direction == "RAW":
        if d < 1:
            raise ValueError("RAW needs d >= 1")
        lines += [p_asm] + nops(d - 1) + [c_asm]
    elif direction == "WAR":
        if d == 0:
            lines.append(f"  {c.asm} ; {p.asm}")
        else:
            lines += [c_asm] + nops(d - 1) + [p_asm]
    else:
        raise ValueError(direction)
    lines += nops(SETTLE_AFTER)
    lines += observation(c, dom)
    lines += nops(4)
    lines += ["  padda [p1], #%d" % RECORD_BYTES] + nops(4)
    return lines


def sweep_points() -> list[tuple[str, int]]:
    """References first (all-OLD then all-NEW), then both directions in distance order."""
    pts = [("WAR", REF_DISTANCE), ("RAW", REF_DISTANCE), ("CONLY", 0)]
    pts += [("RAW", d) for d in RAW_DISTANCES]
    pts += [("WAR", d) for d in WAR_DISTANCES]
    return pts


LLVM_MC = os.path.join(os.environ.get("NPU_WORK_DIR", os.path.expanduser("~/npu-work")),
                       "llvm-aie", "build", "bin", "llvm-mc")
_BUNDLE_CACHE: dict[tuple[str, str], bool] = {}


def can_share_bundle(c_asm: str, p_asm: str) -> bool:
    """Whether the assembler can pack the two instructions into one VLIW bundle."""
    key = (c_asm, p_asm)
    if key not in _BUNDLE_CACHE:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "b.s")
            with open(src, "w") as f:
                f.write(f".text\n  {c_asm} ; {p_asm}\n")
            r = subprocess.run([LLVM_MC, "--triple=aie2", "--filetype=obj", "-o",
                                os.path.join(tmp, "b.o"), src], capture_output=True)
            _BUNDLE_CACHE[key] = r.returncode == 0
    return _BUNDLE_CACHE[key]


def pair_points(p: Producer, c: Consumer, sweep=None) -> list[tuple[str, int]]:
    pts = sweep if sweep is not None else sweep_points()
    out = []
    for direction, d in pts:
        if c.out == "R" and direction == "WAR":
            continue        # the tied consumer's own write confounds the observation
        if c.out != "R" and direction == "CONLY":
            continue        # only a tied consumer lacks an all-OLD WAR reference
        if direction == "WAR" and d == 0 and not can_share_bundle(c.asm, p.asm):
            continue        # the two instructions need the same VLIW slot
        out.append((direction, d))
    return out


def kernel_text(pairs: list[tuple[str, str]], sweep=None) -> tuple[str, list[dict]]:
    """Assembly for one kernel and its point list (record order)."""
    points: list[dict] = []
    test_points: list[str] = [("\n".join(prologue()))]
    for pname, cname in pairs:
        p, c = PRODUCER_BY_NAME[pname], CONSUMER_BY_NAME[cname]
        dom = pair_dom(p, c)
        assert dom, (pname, cname)
        test_points.append("\n".join([f"  // ==== pair {pname} -> {cname} (dom {dom}) ===="]
                                     + pair_preamble(dom)))
        for direction, d in pair_points(p, c, sweep):
            test_points.append("\n".join(point_lines(p, c, direction, d, dom)))
            points.append({"producer": pname, "consumer": cname, "dir": direction, "d": d,
                           "record": len(points)})
    out_size = max(4, len(points) * RECORD_BYTES)
    return isa_gen.build_mega_program(test_points, out_size), points


def kernel_geometry(points: list[dict]) -> dict:
    return {"in_size": IN_SIZE, "out_size": max(4, len(points) * RECORD_BYTES)}


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def parse_pairs(specs: list[str]) -> list[tuple[str, str]]:
    """`P:C` pairs; `P:*` all consumers of P; `*:C` all producers of C; `*:*` everything valid."""
    valid = valid_pairs()
    out: list[tuple[str, str]] = []
    for spec in specs:
        pn, cn = spec.split(":")
        for pair in valid:
            if pn in ("*", pair[0]) and cn in ("*", pair[1]) and pair not in out:
                out.append(pair)
    return out


def chunk(pairs, size):
    return [pairs[i:i + size] for i in range(0, len(pairs), size)]


def generate(out_dir: str, pairs: list[tuple[str, str]], pairs_per_kernel: int) -> dict:
    """Write a suite (harness/suite.py): one case per kernel, `case.json` = {name, points, sizes}."""
    os.makedirs(out_dir, exist_ok=True)
    table = TABLE_BYTES.ljust(IN_SIZE, b"\0")
    kernels = []
    for k, group in enumerate(chunk(pairs, pairs_per_kernel)):
        text, points = kernel_text(group)
        entry = {"name": f"k{k:03d}", "points": points, **kernel_geometry(points)}
        suite.write_single_core_case(out_dir, entry["name"], text, input_bytes=table,
                                     out_bytes=entry["out_size"], meta=entry)
        kernels.append(entry)
    meta = {"in_size": IN_SIZE, "record_bytes": RECORD_BYTES}
    suite.write_suite_json(out_dir, "acc-timing", "gen_acc_timing_probe.py",
                           [k["name"] for k in kernels], meta=meta)
    return {**meta, "kernels": kernels}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("out_dir")
    g.add_argument("--pairs", nargs="+", default=["*:*"])
    g.add_argument("--pairs-per-kernel", type=int, default=3)
    sub.add_parser("list")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for p, c in valid_pairs():
            print(p, c)
        return 0
    m = generate(args.out_dir, parse_pairs(args.pairs), args.pairs_per_kernel)
    print(f"{len(m['kernels'])} kernels, "
          f"{sum(len(k['points']) for k in m['kernels'])} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
