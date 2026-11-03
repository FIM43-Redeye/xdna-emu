#!/usr/bin/env python3
"""Per-port collision probe for the AIE2 accumulator datapath.

Why this exists
---------------
The accumulator timing probe (gen_acc_timing_probe.py) found 24 points where two instructions
reserved one TableGen resource in the same cycle and silicon returned neither the OLD nor the NEW
value. Compilers never emit such schedules, so nothing else constrains the behaviour. This probe
issues every pairing the reservation tables allow to collide, on purpose, and records what each
instruction produced.

Derivation of the design space
------------------------------
Nothing about ports or stage numbers is written down here. `collision-check ops` assembles-and-
decodes each catalogue instruction and reports its reservations (unit, cycle offset) from the
register-aware resolved TableGen itineraries. For two instructions A and B that hold one unit at
offsets a and b, issuing A at bundle 0 and B at bundle g = a - b >= 0 puts both on the unit in one
cycle. Every such (A, B, g) over the accumulator-side units is a collision point; the points at
g - 1 and g + 1 (and B-before-A when g is small) are the controls that show the boundary.

Point layout
------------
  setup     reload the destination registers of A and B (every other register keeps the pattern
            it got in the kernel prologue, so each accumulator register k holds PAT_k and cm8
            holds zeros), reset the witness register
  pair      A and B at their bundles, a witness load C (vldb) in a free bundle
  witness   two `vst` of the witness register, one cycle before and at the cycle the load lands.
            A pipeline stall anywhere between C and the first witness shifts the first one onto
            the landing cycle, so "lo reads NEW" means something after the collision ran late.
  observe   store A's and B's destination registers; memory-writing instructions store
            themselves.
Record (RECORD_BYTES, from the record base): A output [0,128) | B output [128,256) | witness lo
[256,288) | hi [288,320); p1 points at base + 128 while a point runs.

Reference points (instruction alone, sources swapped, sources zeroed, destinations untouched)
are generated next to the collision points so an observed word can be matched against the value
each hypothesis predicts without any model.

  python3 tools/gen_collision_probe.py generate <out-dir> [--limit N]
  python3 tools/gen_collision_probe.py calibrate <out-dir>
Writes a suite (tools/harness/suite.py): run it with
`python3 -m harness suite build|hw|emu|compare <out-dir> ...` (PYTHONPATH=tools); analyse
with tools/analyze_collision_probe.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
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
CHECKER = os.path.join(REPO, "target", "debug", "collision-check")
LLVM_MC = base.LLVM_MC

RECORD_BYTES = 320
# Offsets are relative to p1, which sits REC_BIAS bytes into the record: `vst` of an accumulator
# quarter only reaches [-128, 127], so A's output is stored below p1 and B's above it.
REC_BIAS = 128
A_OFF, B_OFF, WLO_OFF, WHI_OFF = -128, 0, 128, 160
NUM_PAT = 8                 # cm0..cm7 hold PAT_0..PAT_7; cm8 holds zeros
ZERO_REG = 8
SETTLE_BEFORE = 9
SETTLE_AFTER = 14
PROGRAM_BYTES = 16384
# The output buffer is double-buffered in the 64 KiB tile data memory next to the input table.
MAX_POINTS = 20000 // RECORD_BYTES
# Spacing (bundles) from the witness load to the first `vst` that reads the new value; measured
# by `calibrate` (see CALIBRATION in the evidence directory), not derived.
WITNESS_FLIP = 7
WITNESS_REG = 5             # wl5 / wh5
WITNESS_C = "vldb wl5, [p5, #0]"      # p5 addresses WNEW

# Units of the accumulator datapath the design space is derived from.
ACC_UNITS = ("CM_RM_PORT", "CM_WA_PORT", "CM_WM_PORT", "UPS_UNIT", "UPPER_SRS",
             "W_WA_PORT", "W_WM_PORT")
READ_UNITS = ("CM_RM_PORT", "UPPER_SRS")
# The two ports that write the accumulator register file (names: CM = accumulator, W = write).
# Two instructions that write one register in one cycle collide whichever port each uses.
ACC_WRITE_UNITS = tuple(u for u in ACC_UNITS if re.match(r"^CM_W", u))
# The same holds for the two ports that write the W register file.
W_WRITE_UNITS = tuple(u for u in ACC_UNITS if re.match(r"^W_W", u))
# Instructions whose accumulator source read the tables do not reserve: their fitted use cycle is
# 3 (FIT.md), i.e. a read in cycle 2, like the CM_RM_PORT readers that do reserve it. They are
# paired with the CM_RM_PORT cycle-0 readers to see whether the tables miss a collision.
UNRESERVED_READS = {"vmac_i": 2, "vmac_f": 2, "vmsc_f": 2, "vmac_i_sparse": 2, "vmac_f_sparse": 2,
                    "vneg_i": 2, "vneg_f": 2}

# --------------------------------------------------------------------------
# Input table: base items plus eight int and eight float patterns, zeros and a destination init
# --------------------------------------------------------------------------


# COLLISION_BITPAT=1 swaps the consecutive-integer patterns for one-hot ones (lane j of register
# k is (1 << k) | (j << 8) plus a half marker, 0x1000 for the low 512 bits of the register and
# 0x2000 for the high 512), so that an OR, a sum, an XOR or an AND of two registers' lanes are
# different, readable words and a lane from the low half is told from one from the high half.
# Used by the phase-2b UPPER_SRS follow-up probe.
BITPAT = os.environ.get("COLLISION_BITPAT") in ("1", "2")
# COLLISION_BITPAT=2 additionally makes the odd lanes nonzero (flag 0x4000): only for 32-bit
# accumulator lanes, since for 64-bit lanes the odd words are the high words and would saturate.
BITPAT_ODD = os.environ.get("COLLISION_BITPAT") == "2"
# COLLISION_KEYS=a,b,c restricts the collision list to pairs of these catalogue instructions.
KEYS_ONLY = set(filter(None, os.environ.get("COLLISION_KEYS", "").split(",")))
# COLLISION_ANY_KEYS=a,b,c keeps the pairs in which at least one of the two is one of these.
ANY_KEYS = set(filter(None, os.environ.get("COLLISION_ANY_KEYS", "").split(",")))
# COLLISION_SOLO=a,b,c adds the alone / initial references of these instructions even when they
# cannot collide with anything (a check of one instruction against silicon).
# COLLISION_SRS_SHIFT=n sets the shift register of the srs family (s1) to n while the pattern
# loads keep s3 = 0, to tell a converted result from raw data bits.
SRS_SHIFT = int(os.environ.get("COLLISION_SRS_SHIFT", "0"))
CASCADE_WAIT_ITERS = 1500       # ~15000 cycles for the producer to fill the cascade
SOLO_KEYS = [k for k in os.environ.get("COLLISION_SOLO", "").split(",") if k]


def _pat_i(k):
    if BITPAT:
        return base._u16s([((1 << k) | ((i // 2) << 8) | (0x1000 if i < 16 else 0x2000)
                            | (0x4000 if i % 2 else 0))
                           if (i % 2 == 0 or BITPAT_ODD) else 0 for i in range(32)])
    return base._int_pattern(100 + 700 * k)


def _pat_f(k):
    return base._float_pattern(0x3F00 + 0x30 * k)


def build_table():
    items = [(n, d) for n, d in base.TABLE_ITEMS if n not in ("OLD_i", "S_i", "T_i", "U_i",
                                                            "OLD_f", "S_f", "T_f", "U_f")]
    for k in range(NUM_PAT):
        items.append((f"PAT_i{k}", _pat_i(k)))
    items.append(("ZERO", bytes(64)))
    for k in range(NUM_PAT):
        items.append((f"PAT_f{k}", _pat_f(k)))
    # 64-bit accumulator lanes pair two 32-bit words: A = 0x00000000_FFFFFFFF, B = 1
    items.append(("CARRY_A", base._u16s([0xFFFF if k % 2 == 0 else 0 for k in range(32)])))
    items.append(("CARRY_B", base._u16s([1 if k % 2 == 0 else 0 for k in range(32)])))
    items.append(("WINIT", base._u16s([0x7000 + k for k in range(32)])))
    items.append(("WOLD", base._u16s([0x1111 + k for k in range(16)]) * 2))
    items.append(("WNEW", base._u16s([0x2222 + k for k in range(16)]) * 2))
    offsets, buf = {}, bytearray()
    for name, data in base.MASK_ITEMS:
        offsets[name] = len(buf)
        buf += data
    while len(buf) % 32:
        buf.append(0)
    for name, data in items:
        offsets[name] = len(buf)
        buf += data
        while len(buf) % 32:
            buf.append(0)
    return bytes(buf), offsets


TABLE_BYTES, TABLE_OFFSETS = build_table()
IN_SIZE = (len(TABLE_BYTES) + 31) // 32 * 32
# vlda immediates reach [-1024, 992] from a base pointer; p3 and p2 sit 512 and 1024 above p0.
BASES = (("p0", 0), ("p3", 512), ("p2", 1024))


def table_ref(name, extra=0):
    off = TABLE_OFFSETS[name] + extra
    for ptr, b in BASES:
        if 0 <= off - b <= isa_gen.MAX_VECTOR_OFFSET:
            return ptr, off - b
    raise ValueError(f"{name}+{extra} out of reach")


def load_vec(reg, name, half=0):
    ptr, off = table_ref(name, 32 * half)
    return [f"  vlda {reg}, [{ptr}, #{off}]"]


def load_acc(k, dom, name=None):
    """cm<k> := its pattern (zeros for the zero register), via vups through wl11/wh11."""
    name = name or ("ZERO" if k == ZERO_REG else f"PAT_{dom}{k}")
    lines = load_vec("wl11", name, 0) + load_vec("wh11", name, 1) + nops(LOAD_NOPS)
    lines += [f"  vups.s32.s16 bml{k}, wl11, s3", f"  vups.s32.s16 bmh{k}, wh11, s3"]
    return lines + nops(4)


def load_w(reg_lo, reg_hi, name):
    return load_vec(reg_lo, name, 0) + load_vec(reg_hi, name, 1)


def prologue(dom):
    lines = ["  // ==== probe prologue ====", f"  padda [p1], #{REC_BIAS}", "  mov p3, p0", f"  padda [p3], #512",
             "  mov p2, p0", "  padda [p2], #1024", "  mov p5, p0", f"  padda [p5], #{TABLE_OFFSETS['WNEW']}"]
    for reg in base.CONFIGS:
        lines.append(f"  movxm {reg}, #{base.config_word(reg)}")
    lines += nops(3)
    for name, reg in (("X0", "x0"), ("X1", "x1"), ("X2", "x2"), ("X3", "x3")):
        idx = reg[1:]
        lines += load_vec(f"wl{idx}", name, 0) + load_vec(f"wh{idx}", name, 1)
    for k, reg in enumerate(("wl6", "wh6", "wl7", "wh7")):          # y3 = {x6, x7}
        lines += load_vec(reg, "Y3", k)
    for name, q in (("Q1", "q1"), ("Q3", "q3")):
        lines.append(f"  lda {q}, [p0, #{TABLE_OFFSETS[name]}]")
    lines += nops(LOAD_NOPS)
    shift = 0 if dom == "i" else 16
    lines += [f"  mov r14, #{shift}", "  mov s3, r14", "  mov s1, r14", "  movxm r12, #9320",
              "  mov r29, #0", "  nop", "  nop"]
    if SRS_SHIFT:
        lines += [f"  mov r15, #{SRS_SHIFT}", "  nop", "  mov s1, r15", "  nop", "  nop"]
    lines += load_vec("wl10", f"NEWX_{dom}", 0) + load_vec("wh10", f"NEWX_{dom}", 1)
    lines += ["  mov p4, p0", f"  padda [p4], #{TABLE_OFFSETS[f'NEWX_{dom}']}"]
    lines += nops(LOAD_NOPS)
    for k in range(NUM_PAT + 1):
        lines += load_acc(k, dom)
    lines += nops(6)
    if CASCADE:
        lines += cascade_prologue()
    return lines


CASCADE_ALIGN_MARK = "  // CASCADE_ALIGN"


def cascade_prologue() -> list[str]:
    """Enable the cascade input and wait for the producer core to fill it.

    The two cores do not start in lockstep (the producer starts ~1000 cycles before or after
    the consumer), so the consumer waits a counted loop and finds the cascade full. Silicon
    ignores the low 4 bits of a branch target (gen_cascade_probe.delay_loop), so the label must be
    16-byte aligned; the padding is written as explicit nops (kernel_text fills in the count at
    the marker) rather than `.p2align`, so that one assembly line is still one bundle for the
    checkers that map bundles back to lines."""
    import gen_cascade_probe as cp
    return (["  // ==== cascade prologue ===="] + cp._enable("crSCDEn")
            + [f"  movxm r0, #{CASCADE_WAIT_ITERS}"] + nops(2) + [CASCADE_ALIGN_MARK, ".Lcascwait:",
               "  add r0, r0, #-1"] + nops(2) + ["  jnz r0, #.Lcascwait"] + nops(5) + nops(4))


def label_offset(head: str) -> int:
    """Byte offset in .text at which the line after `head` would start (the assembler rounds the
    whole section up to 16 bytes, so the section size is no measure; a label is)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, obj = os.path.join(tmp, "k.s"), os.path.join(tmp, "k.o")
        with open(src, "w") as f:
            f.write(head + "\n.globl alignprobe\nalignprobe:\n  nop\n")
        r = subprocess.run([LLVM_MC, "--triple=aie2", "--filetype=obj", "-o", obj, src],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[:2000])
        dump = subprocess.run([os.path.join(os.path.dirname(LLVM_MC), "llvm-objdump"), "-d", obj],
                              capture_output=True, text=True).stdout
    return int(re.search(r"^([0-9a-f]+) <alignprobe>:", dump, re.M).group(1), 16)


def align_cascade_label(text: str) -> str:
    """Replace the alignment marker by the nops that put the next line on a 16-byte boundary."""
    if CASCADE_ALIGN_MARK not in text:
        return text
    # `.p2align 4` at the top raises the section alignment to 16 so the linker places the kernel
    # on a 16-byte boundary; without it the offsets computed here are not the final addresses.
    text = text.replace(".text\n", ".text\n.p2align 4\n", 1)
    head, tail = text.split(CASCADE_ALIGN_MARK + "\n", 1)
    pad = (-label_offset(head)) % 16
    assert pad % 2 == 0, pad
    return head + "  nop\n" * (pad // 2) + tail


# --------------------------------------------------------------------------
# Instruction catalogue
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Ins:
    key: str
    asm: str                    # {S1} {S2} {D} {DW} {M} are filled in
    dom: Optional[str]          # 'i', 'f' or None (either)
    srcs: tuple = ()            # kinds of the accumulator sources S1, S2: cm | bml | amll | amhh
    dst: Optional[str] = None   # cm | bml | x | wl | mem
    tied: bool = False          # the destination is also read as an accumulator (D appears twice)


CATALOGUE = [
    Ins("vmov_cm", "vmov cm{D}, cm{S1}", None, ("cm",), "cm"),
    Ins("vmov_bm", "vmov bml{D}, bml{S1}", None, ("bml",), "bml"),
    Ins("vmov_x_bm", "vmov {DW}, bml{S1}", None, ("bml",), "x"),
    Ins("vsrs_s16_s32", "vsrs.s16.s32 {DW}, bml{S1}, s1", None, ("bml",), "wl"),
    Ins("vsrs_s32_s64", "vsrs.s32.s64 {DW}, cm{S1}, s1", "i", ("cm",), "x"),
    Ins("vsrsm_s16_s32", "vsrsm.s16.s32 {DW}, bml{S1}, s1", None, ("bml",), "wl"),
    Ins("vsrs_d16_s32", "vsrs.d16.s32 {DW}, bml{S1}, s1", None, ("bml",), "wl"),
    Ins("vconv_bf16_fp32", "vconv.bf16.fp32 {DW}, bml{S1}", "f", ("bml",), "wl"),
    Ins("vst_amll", "vst amll{S1}, [p1, #{M}]", None, ("bml",), "mem"),
    Ins("vst_amhh", "vst amhh{S1}, [p1, #{M}]", None, ("cm",), "mem"),
    Ins("vst_srs_s16_s32", "vst.srs.s16.s32 bml{S1}, s1, [p1, #{M}]", None, ("bml",), "mem"),
    Ins("vst_conv_bf16_fp32", "vst.conv.bf16.fp32 bml{S1}, [p1, #{M}]", "f", ("bml",), "mem"),
    Ins("vadd_i", "vadd cm{D}, cm{S1}, cm{S2}, r1", "i", ("cm", "cm"), "cm"),
    Ins("vadd_f", "vadd.f bml{D}, bml{S1}, bml{S2}, r3", "f", ("bml", "bml"), "bml"),
    Ins("vaddmac_i", "vaddmac cm{D}, cm{D}, cm{S1}, x0, x1, r1", "i", ("cm",), "cm", True),
    Ins("vaddmac_f", "vaddmac.f bml{D}, bml{D}, bml{S1}, x2, x3, r3", "f", ("bml",), "bml", True),
    Ins("vaddmsc_f", "vaddmsc.f bml{D}, bml{D}, bml{S1}, x2, x3, r3", "f", ("bml",), "bml", True),
    Ins("vaddmac_i_sparse", "vaddmac cm{D}, cm{D}, cm{S1}, x0, qx1, r2", "i", ("cm",), "cm", True),
    Ins("vaddmac_f_sparse", "vaddmac.f bml{D}, bml{D}, bml{S1}, y3, qx3, r4", "f", ("bml",), "bml", True),
    Ins("vmul_i", "vmul cm{D}, x0, x1, r1", "i", (), "cm"),
    Ins("vmul_f", "vmul.f bml{D}, x2, x3, r3", "f", (), "bml"),
    Ins("vmac_i", "vmac cm{D}, cm{S1}, x0, x1, r1", "i", ("cm",), "cm"),
    Ins("vmac_f", "vmac.f bml{D}, bml{S1}, x2, x3, r3", "f", ("bml",), "bml"),
    Ins("vmac_i_sparse", "vmac cm{D}, cm{S1}, x0, qx1, r2", "i", ("cm",), "cm"),
    Ins("vmac_f_sparse", "vmac.f bml{D}, bml{S1}, y3, qx3, r4", "f", ("bml",), "bml"),
    Ins("vmsc_f", "vmsc.f bml{D}, bml{S1}, x2, x3, r3", "f", ("bml",), "bml"),
    Ins("vclr_cm", "vclr cm{D}", None, (), "cm"),
    Ins("vclr_bm", "vclr bml{D}", None, (), "bml"),
    Ins("vneg_i", "vneg cm{D}, cm{S1}, r1", "i", ("cm",), "cm"),
    Ins("vneg_f", "vneg.f bml{D}, bml{S1}, r3", "f", ("bml",), "bml"),
    Ins("vups", "vups.s32.s16 bml{D}, wl10, s3", None, (), "bml"),
    Ins("vlda_ups", "vlda.ups.s32.s16 bml{D}, s3, [p4, #0]", None, (), "bml"),
    Ins("vlda_conv", "vlda.conv.fp32.bf16 bml{D}, [p4, #0]", "f", (), "bml"),
    Ins("vconv_fp32_bf16", "vconv.fp32.bf16 bml{D}, wl10", "f", (), "bml"),
    Ins("vmov_bm_x", "vmov bml{D}, x10", None, (), "bml"),
]
# Phase 2b: the accumulator-side classes and shapes probe2 did not cover. Sources that are not
# accumulators (x0/x1 vectors, the scalar r12, the q1 mask register, memory) are fixed in the text.
CATALOGUE_2B = [
    # UPPER_SRS holders: every vsrs / vsrsm / vst.srs shape (they differ in source and output width)
    Ins("vsrs_s8_s32", "vsrs.s8.s32 {DW}, cm{S1}, s1", None, ("cm",), "wl"),
    Ins("vsrs_d32_s64", "vsrs.d32.s64 {DW}, cm{S1}, s1", "i", ("cm",), "x"),
    Ins("vsrs_s16_s64", "vsrs.s16.s64 {DW}, cm{S1}, s1", "i", ("cm",), "wl"),
    Ins("vsrs_d16_s64", "vsrs.d16.s64 {DW}, cm{S1}, s1", "i", ("cm",), "wl"),
    Ins("vsrsm_d16_s32", "vsrsm.d16.s32 {DW}, bml{S1}, s1", None, ("bml",), "wl"),
    Ins("vsrsm_s32_s64", "vsrsm.s32.s64 {DW}, bml{S1}, s1", "i", ("bml",), "wl"),
    Ins("vsrsm_d32_s64", "vsrsm.d32.s64 {DW}, bml{S1}, s1", "i", ("bml",), "wl"),
    Ins("vst_srs_d16_s32", "vst.srs.d16.s32 bml{S1}, s1, [p1, #{M}]", None, ("bml",), "mem"),
    Ins("vst_srs_s8_s32", "vst.srs.s8.s32 cm{S1}, s1, [p1, #{M}]", None, ("cm",), "mem"),
    Ins("vst_srs_s16_s64", "vst.srs.s16.s64 cm{S1}, s1, [p1, #{M}]", "i", ("cm",), "mem"),
    Ins("vst_srs_d16_s64", "vst.srs.d16.s64 cm{S1}, s1, [p1, #{M}]", "i", ("cm",), "mem"),
    Ins("vst_srs_s32_s64", "vst.srs.s32.s64 bml{S1}, s1, [p1, #{M}]", "i", ("bml",), "mem"),
    Ins("vst_srs_d32_s64", "vst.srs.d32.s64 bml{S1}, s1, [p1, #{M}]", "i", ("bml",), "mem"),
    # W_WA_PORT holders
    Ins("vpack_s8_s16", "vpack.s8.s16 {DW}, x0", None, (), "wl"),
    Ins("vpack_d8_d16", "vpack.d8.d16 {DW}, x0", None, (), "wl"),
    Ins("vpack_s4_s8", "vpack.s4.s8 {DW}, x0", None, (), "wl"),
    Ins("vlda_w", "vlda {DW}, [p0, #32]", None, (), "wl"),
    Ins("vlda_128", "vlda.128 {DW}, [p0]", None, (), "wl"),
    Ins("vfloor_am", "vfloor.s32.bf16 {DW}, amll{S1}, s1", "f", ("bml",), "x"),
    Ins("vfloor_w", "vfloor.s32.bf16 {DW}, wl0, s1", "f", (), "x"),
    # W_WM_PORT holders (vector ALU writes)
    Ins("vband", "vband {DW}, x0, x1", None, (), "x"),
    Ins("vbor", "vbor {DW}, x0, x1", None, (), "x"),
    Ins("vadd_32", "vadd.32 {DW}, x0, x1", None, (), "x"),
    Ins("vsub_32", "vsub.32 {DW}, x0, x1", None, (), "x"),
    Ins("vshift_align", "vshift.align {DW}, x0, s0, x1, r1", None, (), "x"),
    Ins("vbcst_32", "vbcst.32 {DW}, r12", None, (), "x"),
    Ins("vmov_w_w", "vmov {DW}, wl0", None, (), "wl"),
    Ins("vmov_x_x", "vmov {DW}, x0", None, (), "x"),
    # CM_WM_PORT holders that hold CM_WM and W_WM both
    Ins("vshuffle_x", "vshuffle {DW}, x0, x1, r1", None, (), "x"),
    Ins("vshuffle_bm", "vshuffle bml{D}, x0, x1, r1", None, (), "bml"),
    Ins("vbcstshfl_x", "vbcstshfl.32 {DW}, r12, r29", None, (), "x"),
    Ins("vbcstshfl_bm", "vbcstshfl.32 bml{D}, r12, r29", None, (), "bml"),
    # the static VMOV_W class: an accumulator quarter and a q register
    Ins("vmov_am_q", "vmov amll{D}, q1", None, (), "am"),
    Ins("vmov_q_am", "vmov {DQ}, amll{S1}", None, ("bml",), "q"),
]
# Writers that target the high half of a W register, to tell a write lane fixed by the port from
# one chosen by the destination half.
CATALOGUE_2C = [
    Ins("vlda_wh", "vlda {DW}, [p0, #32]", None, (), "wh"),
    Ins("vpack_s8_s16_wh", "vpack.s8.s16 {DW}, x0", None, (), "wh"),
    Ins("vsrs_s16_s32_wh", "vsrs.s16.s32 {DW}, bml{S1}, s1", None, ("bml",), "wh"),
    Ins("vsrs_s16_s64_wh", "vsrs.s16.s64 {DW}, cm{S1}, s1", "i", ("cm",), "wh"),
    Ins("vconv_bf16_fp32_wh", "vconv.bf16.fp32 {DW}, bml{S1}", "f", ("bml",), "wh"),
    Ins("vmov_w_wh", "vmov {DW}, wl0", None, (), "wh"),
    Ins("vsrsm_s16_s32_wh", "vsrsm.s16.s32 {DW}, bml{S1}, s1", None, ("bml",), "wh"),
]
# Siblings of the three instructions the phase-2b alone references showed the emulator getting
# wrong (vmov am,q ; vshuffle bm ; vbcstshfl bm): the other am quarters, the high bm half, the
# other broadcast widths and their x forms.
CATALOGUE_2D = [
    Ins("vmov_amlh_q", "vmov amlh{D}, q1", None, (), "amlh"),
    Ins("vmov_amhl_q", "vmov amhl{D}, q1", None, (), "amhl"),
    Ins("vmov_amhh_q", "vmov amhh{D}, q1", None, (), "amhh"),
    Ins("vshuffle_bmh", "vshuffle bmh{D}, x0, x1, r1", None, (), "bmh"),
    Ins("vbcstshfl_bmh", "vbcstshfl.32 bmh{D}, r12, r29", None, (), "bmh"),
    Ins("vbcstshfl8_bm", "vbcstshfl.8 bml{D}, r12, r29", None, (), "bml"),
    Ins("vbcstshfl16_bm", "vbcstshfl.16 bml{D}, r12, r29", None, (), "bml"),
    Ins("vbcstshfl8_x", "vbcstshfl.8 {DW}, r12, r29", None, (), "x"),
    Ins("vbcstshfl16_x", "vbcstshfl.16 {DW}, r12, r29", None, (), "x"),
]
# Cascade reads (step 7): the SCD word is the source, so these need a producer core feeding the
# cascade (tools/gen_cascade_collision_probe.py). They join the generating catalogue only after
# enable_cascade(), so the single-tile probes keep theirs; BY_KEY always knows them, so the record
# tools read cascade probe dirs too.
CASCADE = False
CATALOGUE_CASCADE = [
    Ins("vmov_x_scd", "vmov {DW}, SCD", None, (), "x"),
    Ins("vmov_bm_scd", "vmov bml{D}, SCD", None, (), "bml"),
    Ins("vmov_bmh_scd", "vmov bmh{D}, SCD", None, (), "bmh"),
]
NEW_KEYS = {i.key for i in CATALOGUE_2B}
CATALOGUE = CATALOGUE + CATALOGUE_2B + CATALOGUE_2C + CATALOGUE_2D
BY_KEY = {i.key: i for i in CATALOGUE + CATALOGUE_CASCADE}


def enable_cascade() -> None:
    """Generate two-core cascade probes: the cascade reads join the catalogue and every kernel
    gets the cascade prologue. Idempotent. A switch made at import time (it was an environment
    variable) depended on which caller imported this module first."""
    global CASCADE, CATALOGUE
    if not CASCADE:
        CASCADE = True
        CATALOGUE = CATALOGUE + CATALOGUE_CASCADE
W_DEST = {"A": ("x8", "wl8", "wh8"), "B": ("x4", "wl4", "wh4")}     # (x, wl, wh name) per role
W_FAMILY = ("x", "wl", "wh")        # destinations that live in the W register file
Q_DEST = {"A": "q0", "B": "q2"}                          # q1 and q3 hold the masks


@dataclass(frozen=True)
class Regs:
    """Register assignment of one instruction: accumulator indices and the W destination."""
    s: tuple = ()
    d: Optional[int] = None
    role: str = "A"             # picks the memory record half of a `mem` instruction
    wsel: str = "A"             # picks the W destination register (x8/wl8 for A, x4/wl4 for B)


def render(ins: Ins, r: Regs) -> str:
    kw = {"D": r.d, "M": A_OFF if r.role == "A" else B_OFF}
    for i, s in enumerate(r.s):
        kw[f"S{i + 1}"] = s
    if ins.dst == "q":
        kw["DQ"] = Q_DEST[r.wsel]
    if ins.dst in W_FAMILY:
        kw["DW"] = W_DEST[r.wsel][W_FAMILY.index(ins.dst)]
    return ins.asm.format(**kw)


# --------------------------------------------------------------------------
# Reservations of every catalogue instruction, from the decoder
# --------------------------------------------------------------------------

# destination kinds that name an accumulator register; each is observed through its own quarters
ACC_QUARTERS = {"cm": ("amll", "amlh", "amhl", "amhh"), "bml": ("amll", "amlh"),
                "bmh": ("amhl", "amhh"), "am": ("amll",), "amlh": ("amlh",), "amhl": ("amhl",),
                "amhh": ("amhh",)}
ACC_DST = tuple(ACC_QUARTERS)


def sample_regs(ins: Ins, role: str = "A") -> Regs:
    return Regs(s=tuple(range(len(ins.srcs))), d=len(ins.srcs) if ins.dst in ACC_DST else None,
                role=role)


def reservations_of_catalogue() -> dict[str, list[tuple[str, int]]]:
    """key -> [(unit, offset)] for the catalogue, from `collision-check ops`."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, obj = os.path.join(tmp, "c.s"), os.path.join(tmp, "c.o")
        with open(src, "w") as f:
            f.write(".text\n")
            for ins in CATALOGUE:
                f.write("  " + render(ins, sample_regs(ins)) + "\n")
        r = subprocess.run([LLVM_MC, "--triple=aie2", "--filetype=obj", "-o", obj, src],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        out = subprocess.run([CHECKER, "ops", obj], capture_output=True, text=True, check=True).stdout
    held: dict[int, list[tuple[str, int]]] = {}
    for line in out.splitlines():
        m = re.match(r"\s+OP (\d+) (\S+) ?(.*)", line)
        if m:
            held[int(m.group(1))] = [(u, int(o.rstrip("r"))) for u, o in
                                     (x.split(":") for x in m.group(3).split(",") if x)]
    # the assembler may append a padding bundle
    assert len(held) >= len(CATALOGUE), (len(held), len(CATALOGUE))
    out = {ins.key: held[i] for i, ins in enumerate(CATALOGUE)}
    for key, off in UNRESERVED_READS.items():
        out[key] = out[key] + [("CM_RM_PORT*", off)]       # * = not in the tables
    return out


def pair_dom(a: Ins, b: Ins) -> str:
    """Data domain of a pair. A float instruction needs float patterns; an integer instruction
    only needs distinct bit patterns, which float patterns also are, so mixing is allowed."""
    return "f" if "f" in (a.dom, b.dom) else "i"


def same_resource(ua: str, ub: str) -> bool:
    """Whether holding ua and ub in one cycle is a collision: the same unit, a starred unit
    (an extension the tables do not make) against its unstarred name, or two accumulator write
    ports."""
    if ua.rstrip("*") == ub.rstrip("*"):
        return True
    return ((ua in ACC_WRITE_UNITS and ub in ACC_WRITE_UNITS)
            or (ua in W_WRITE_UNITS and ub in W_WRITE_UNITS))


def derive_collisions(res, only_new: bool = False, unit: Optional[str] = None) -> list[dict]:
    """Every (A, B, g) where A holds a unit at cycle a and B holds the same resource at b, with
    g = a - b >= 0 (A issued g bundles before B), merged over the units that collide there."""
    merged: dict[tuple, dict] = {}
    for ia, a in enumerate(CATALOGUE):
        for ib, b in enumerate(CATALOGUE):
            for ua, oa in res[a.key]:
                if ua.rstrip("*") not in ACC_UNITS:
                    continue
                for ub, ob in res[b.key]:
                    g = oa - ob
                    if not same_resource(ua, ub) or not (g > 0 or (g == 0 and ia <= ib)):
                        continue
                    if ua.endswith("*") and ub.endswith("*"):
                        continue
                    if only_new and a.key not in NEW_KEYS and b.key not in NEW_KEYS:
                        continue
                    if KEYS_ONLY and not (a.key in KEYS_ONLY and b.key in KEYS_ONLY):
                        continue
                    if ANY_KEYS and not (a.key in ANY_KEYS or b.key in ANY_KEYS):
                        continue
                    if unit and unit not in (ua.rstrip("*"), ub.rstrip("*")):
                        continue
                    k = (a.key, b.key, g)
                    m = merged.setdefault(k, {"a": a.key, "b": b.key, "g": g, "units": [],
                                              "cycles": []})
                    name = ua if ua == ub else f"{ua}~{ub}"
                    if name not in m["units"]:
                        m["units"].append(name)
                        m["cycles"].append(oa)
    return list(merged.values())


# --------------------------------------------------------------------------
# Register allocation and configurations
# --------------------------------------------------------------------------

def allocate(a: Ins, b: Ins, cfg: str) -> Optional[tuple[Regs, Regs]]:
    """Accumulator indices for A and B under a configuration, or None if it does not apply."""
    ra = Regs(s=tuple(range(len(a.srcs))), d=2 if a.dst in ACC_DST else None, role="A")
    sb = [3 + i for i in range(len(b.srcs))]
    db = 5 if b.dst in ACC_DST else None
    wsel = "B"
    if cfg == "srcshare":
        if not a.srcs or not b.srcs:
            return None
        sb[0] = ra.s[0]
    elif cfg == "dstshare":
        if a.dst in ACC_DST and b.dst in ACC_DST:
            db = ra.d
        elif (a.dst == b.dst and a.dst == "q") or (a.dst in W_FAMILY and b.dst in W_FAMILY):
            wsel = "A"          # one W register (or its halves) written by both
        else:
            return None
    elif cfg != "sep":
        raise ValueError(cfg)
    return ra, Regs(s=tuple(sb), d=db, role="B", wsel=wsel)


def configs_for(coll: dict, a: Ins, b: Ins) -> list[str]:
    cfgs = ["sep"]
    if any(u.rstrip("*") in READ_UNITS for u in coll["units"]) and a.srcs and b.srcs:
        cfgs.append("srcshare")
    if any(u.split("~")[0] in ("CM_WA_PORT", "CM_WM_PORT", "W_WA_PORT", "W_WM_PORT", "UPS_UNIT")
           for u in coll["units"]):
        if allocate(a, b, "dstshare"):
            cfgs.append("dstshare")
    return cfgs


# --------------------------------------------------------------------------
# Points
# --------------------------------------------------------------------------

def dst_init_lines(ins: Ins, r: Regs, dom: str) -> list[str]:
    if ins.dst in ACC_DST:
        return load_acc(r.d, dom)
    if ins.dst == "q":
        return [f"  lda {Q_DEST[r.wsel]}, [p0, #{TABLE_OFFSETS['Q3']}]"]
    if ins.dst == "x":
        n = W_DEST[r.wsel][0][1:]
        return load_w(f"wl{n}", f"wh{n}", "WINIT")
    if ins.dst == "wl":
        return load_vec(W_DEST[r.wsel][1], "WINIT", 0)
    if ins.dst == "wh":
        return load_vec(W_DEST[r.wsel][2], "WINIT", 1)
    return []


def observe_lines(ins: Ins, r: Regs, off: int) -> list[str]:
    out = []
    if ins.dst in ACC_QUARTERS:
        for j, q in enumerate(ACC_QUARTERS[ins.dst]):
            out.append(f"  vst {q}{r.d}, [p1, #{off + 32 * j}]")
    elif ins.dst == "q":
        out.append(f"  st {Q_DEST[r.wsel]}, [p1, #{off}]")
    elif ins.dst == "x":
        x = W_DEST[r.wsel][0][1:]
        out += [f"  vst wl{x}, [p1, #{off}]", f"  vst wh{x}, [p1, #{off + 32}]"]
    elif ins.dst in ("wl", "wh"):
        out.append(f"  vst {W_DEST[r.wsel][W_FAMILY.index(ins.dst)]}, [p1, #{off}]")
    return out


def can_share(x: str, y: str) -> bool:
    return base.can_share_bundle(x, y)


def witness_window(pa: int, pb: int, cycles: list[int]):
    """Bundle of the witness load C and of the first witness read, or None if no slot fits.

    A stall at absolute cycle T delays every bundle issued after T, so C must issue at or before
    T and the first witness read (WITNESS_FLIP - 1 after C) strictly after it. Neither may share a
    bundle with the pair, and the reads must follow the pair's last bundle.
    """
    T = max(cycles) + pa
    last = max(pa, pb)
    for c0 in range(T, T - WITNESS_FLIP + 1, -1):
        L = c0 + WITNESS_FLIP - 1
        if c0 in (pa, pb) or L <= last or L <= T:
            continue
        return c0, L
    return None


def timeline_lines(events: dict[int, list[str]]) -> list[str]:
    lo, hi = min(events), max(events)
    lines = []
    for b in range(lo, hi + 1):
        ev = events.get(b, [])
        lines.append("  " + " ; ".join(ev) if ev else "  nop")
    return lines


def point_lines(point: dict, dom: str) -> list[str]:
    a, b = BY_KEY[point["a"]] if point["a"] else None, BY_KEY[point["b"]] if point["b"] else None
    ra = Regs(**{**point["ra"], "s": tuple(point["ra"]["s"])}) if point["a"] else None
    rb = Regs(**{**point["rb"], "s": tuple(point["rb"]["s"])}) if point["b"] else None
    lines = [f"  // ---- point {point['id']}: {point['kind']} A={point['a']} B={point['b']} "
             f"cfg={point.get('cfg')} g={point.get('g')} delta={point.get('delta')} ----"]
    reinit = []
    for ins, r in ((a, ra), (b, rb)):
        if ins:
            reinit += dst_init_lines(ins, r, dom)
    reinit += load_vec("wl5", "WOLD", 0)
    lines += reinit + nops(SETTLE_BEFORE)
    events: dict[int, list[str]] = {}
    if a and point["kind"] != "ref_none":
        events.setdefault(point["pa"], []).append(render(a, ra))
    if b:
        events.setdefault(point["pb"], []).append(render(b, rb))
    if point.get("c0") is not None:
        events.setdefault(point["c0"], []).append(WITNESS_C)
        events.setdefault(point["wl"], []).append(f"vst wl5, [p1, #{WLO_OFF}]")
        events.setdefault(point["wl"] + 1, []).append(f"vst wl5, [p1, #{WHI_OFF}]")
    lines += timeline_lines(events) if events else nops(1)
    lines += nops(SETTLE_AFTER)
    for ins, r, off in ((a, ra, A_OFF), (b, rb, B_OFF)):
        if ins:
            lines += observe_lines(ins, r, off)
    lines += nops(4) + [f"  padda [p1], #{RECORD_BYTES}"] + nops(4)
    return lines


def make_points(res, limit: Optional[int] = None, only_new: bool = False,
                unit: Optional[str] = None) -> list[dict]:
    """All collision, control and reference points, in generation order."""
    points: list[dict] = []
    colls = derive_collisions(res, only_new, unit)
    if limit:
        colls = colls[:limit]
    ref_specs = []
    for coll in colls:
        a, b = BY_KEY[coll["a"]], BY_KEY[coll["b"]]
        dom = pair_dom(a, b)
        for cfg in configs_for(coll, a, b):
            alloc = allocate(a, b, cfg)
            if not alloc:
                continue
            ra, rb = alloc
            g = coll["g"]
            group = []
            for delta in (-1, 0, 1):
                rel = g + delta          # B's bundle minus A's
                if rel == 0 and not can_share(render(a, ra), render(b, rb)):
                    continue
                pa, pb = (0, rel) if rel >= 0 else (-rel, 0)
                win = witness_window(pa, pb, coll["cycles"])
                group.append({"kind": "pair" if delta == 0 else "control", "a": a.key, "b": b.key,
                              "ra": ra.__dict__, "rb": rb.__dict__, "cfg": cfg, "g": g,
                              "delta": delta, "units": coll["units"], "cycles": coll["cycles"],
                              "pa": pa, "pb": pb, "dom": dom, "c0": win[0] if win else None,
                              "wl": win[1] if win else None})
            if not any(q["kind"] == "pair" for q in group):
                continue                 # the two cannot issue in one bundle (same slot)
            points += group
            ref_specs.append((a, ra, b, rb, coll, dom))
    for key in SOLO_KEYS:
        ins = BY_KEY[key]
        r = Regs(s=tuple(3 + i for i in range(len(ins.srcs))), d=5 if ins.dst in ACC_DST else None,
                 role="B", wsel="B")
        ref_specs.append((ins, r, ins, r, {"units": []}, "f" if ins.dom == "f" else "i"))
    # references
    seen = set()
    for a, ra, b, rb, coll, dom in ref_specs:
        read_pair = any(u.rstrip("*") in READ_UNITS for u in coll["units"])
        want = [("alone", a, ra, dom), ("alone", b, rb, dom), ("none", a, ra, dom), ("none", b, rb, dom)]
        for ins, r in ((a, ra), (b, rb)):
            for j in range(len(r.s)):
                s = list(r.s)
                s[j] = ZERO_REG
                want.append(("alone", ins, Regs(tuple(s), r.d, r.role, r.wsel), dom))
        if read_pair:
            for ins, r, other_ins, other in ((a, ra, b, rb), (b, rb, a, ra)):
                for j in range(len(r.s)):
                    for src in other.s:
                        s = list(r.s)
                        s[j] = src
                        want.append(("alone", ins, Regs(tuple(s), r.d, r.role, r.wsel), dom))
        for kind, ins, r, d in want:
            key = (kind, ins.key, r.s, r.d, r.role, r.wsel, d)
            if key in seen:
                continue
            seen.add(key)
            points.append({"kind": "ref_" + kind, "a": ins.key, "b": None, "ra": r.__dict__,
                           "rb": None, "pa": 0, "pb": 0, "c0": None, "dom": d, "units": [],
                           "cycles": []})
    for i, p in enumerate(points):
        p["id"] = i
    return points


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

def kernel_text(points: list[dict], dom: str) -> str:
    chunks = ["\n".join(prologue(dom))]
    for p in points:
        chunks.append("\n".join(point_lines(p, dom)))
    return align_cascade_label(isa_gen.build_mega_program(chunks, max(4, len(points) * RECORD_BYTES)))


def text_size(text: str) -> int:
    """Bytes of .text the program assembles to."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, obj = os.path.join(tmp, "k.s"), os.path.join(tmp, "k.o")
        with open(src, "w") as f:
            f.write(text)
        r = subprocess.run([LLVM_MC, "--triple=aie2", "--filetype=obj", "-o", obj, src],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[:2000])
        sz = subprocess.run([os.path.join(os.path.dirname(LLVM_MC), "llvm-size"), obj],
                            capture_output=True, text=True).stdout.split("\n")[1].split()[0]
        return int(sz)


def write_probe_suite(out_dir: str, name: str, kernels: list[tuple[dict, str]], meta: dict) -> dict:
    """Write one case per (entry, asm) kernel plus `suite.json`; returns the old manifest shape
    ({in_size, record_bytes, kernels, ...meta}) for callers that count points."""
    table = TABLE_BYTES.ljust(IN_SIZE, b"\0")
    for entry, asm in kernels:
        suite.write_single_core_case(out_dir, entry["name"], asm, input_bytes=table,
                                     out_bytes=entry["out_size"], meta=entry)
    meta = {"in_size": IN_SIZE, "record_bytes": RECORD_BYTES, **meta}
    suite.write_suite_json(out_dir, name, "gen_collision_probe.py",
                           [e["name"] for e, _ in kernels], meta=meta)
    return {**meta, "kernels": [e for e, _ in kernels]}


def generate(out_dir: str, limit: Optional[int] = None, only_new: bool = False,
             unit: Optional[str] = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    res = reservations_of_catalogue()
    with open(os.path.join(out_dir, "reservations.json"), "w") as f:
        json.dump(res, f, indent=1)
    points = make_points(res, limit, only_new, unit)
    kernels: list[tuple[dict, str]] = []
    for dom in ("i", "f"):
        dp = [p for p in points if p["dom"] == dom]
        k = 0
        i = 0
        while i < len(dp):
            n = 16
            cap = min(MAX_POINTS, len(dp) - i)
            while True:                               # grow the chunk to the program-memory budget
                sz = text_size(kernel_text(dp[i:i + n], dom))
                if sz > PROGRAM_BYTES - 512 or n >= cap:
                    break
                n = min(int(n * 1.3) + 1, cap)
            n = min(n, cap)
            while n > 1 and text_size(kernel_text(dp[i:i + n], dom)) > PROGRAM_BYTES - 512:
                n = int(n * 0.85)
            chunk = dp[i:i + n]
            stem = f"k{len(kernels):03d}"
            kernels.append(({"name": stem, "dom": dom, "points": chunk, "in_size": IN_SIZE,
                             "out_size": max(4, len(chunk) * RECORD_BYTES)},
                            kernel_text(chunk, dom)))
            i += n
            k += 1
    return write_probe_suite(out_dir, "collisions", kernels, {"catalogue": {i.key: i.asm for i in CATALOGUE}})


def generate_calibration(out_dir: str) -> dict:
    """Witness calibration: C then a witness read d bundles later, d = 1..12, domain i."""
    os.makedirs(out_dir, exist_ok=True)
    chunks = ["\n".join(prologue("i"))]
    pts = []
    for d in range(1, 13):
        ev = {0: [WITNESS_C], d: [f"vst wl5, [p1, #{256 - REC_BIAS}]"]}
        lines = [f"  // ---- point calib d={d} ----"] + load_vec("wl5", "WOLD", 0) + nops(SETTLE_BEFORE)
        lines += timeline_lines(ev) + nops(SETTLE_AFTER) + nops(4)
        lines += [f"  padda [p1], #{RECORD_BYTES}"] + nops(4)
        chunks.append("\n".join(lines))
        pts.append({"kind": "calib", "d": d, "id": d - 1})
    entry = {"name": "k000", "dom": "i", "points": pts, "in_size": IN_SIZE,
             "out_size": len(pts) * RECORD_BYTES}
    return write_probe_suite(out_dir, "collisions-calibration",
                             [(entry, isa_gen.build_mega_program(chunks, len(pts) * RECORD_BYTES))], {})


ADDSUB_OPS = [("vneg", "vneg cm5, cm3, {c}"), ("vadd", "vadd cm5, cm3, cm4, {c}"),
              ("vsub", "vsub cm5, cm3, cm4, {c}"), ("vnegadd", "vnegadd cm5, cm3, cm4, {c}"),
              ("vnegsub", "vnegsub cm5, cm3, cm4, {c}")]
MAC_OPS = [("vnegmul", "vnegmul cm5, x0, x1, {c}"), ("vnegmac", "vnegmac cm5, cm3, x0, x1, {c}"),
           ("vnegmsc", "vnegmsc cm5, cm3, x0, x1, {c}"), ("vmul", "vmul cm5, x0, x1, {c}"),
           ("vmac", "vmac cm5, cm3, x0, x1, {c}"), ("vmsc", "vmsc cm5, cm3, x0, x1, {c}")]


def generate_negate_probe(out_dir: str) -> dict:
    """Integer accumulator negate/add/MAC family, one instruction alone per point (cm3, cm4
    sources, cm5 destination). Add/sub/neg run under an acc64 config (r1) and an acc32-style one
    (r3), on ordinary patterns and on carry operands, with and without the shift16 bit (a
    per-point config word in r5). MACs run under the dense integer config r1."""
    os.makedirs(out_dir, exist_ok=True)
    kernels: list[tuple[dict, str]] = []
    specs = []
    for name, tmpl in ADDSUB_OPS:
        specs.append((name, [(tmpl, cfg, data, sh) for cfg in ("r1", "r3") for data in ("pat", "carry")
                             for sh in (0, 1)]))
    for name, tmpl in MAC_OPS:
        specs.append((name, [(tmpl, "r1", "pat", 0)]))
    for ki, (name, cases) in enumerate(specs):       # one kernel per op: an unsupported one fails alone
        chunks = ["\n".join(prologue("i"))]
        pts = []
        for tmpl, cfg, data, sh in cases:
            reg = cfg
            lines = []
            if sh:
                word = base.config_word(cfg) | (1 << 10)
                lines += [f"  movxm r5, #{word}"] + nops(2)
                reg = "r5"
            asm = tmpl.format(c=reg)
            lines += [f"  // ---- point {len(pts)}: {asm} cfg={cfg} data={data} shift16={sh} ----"]
            n3, n4 = ("CARRY_A", "CARRY_B") if data == "carry" else (None, None)
            lines += load_acc(3, "i", n3) + load_acc(4, "i", n4) + load_acc(5, "i") + nops(SETTLE_BEFORE)
            lines += [f"  {asm}"] + nops(SETTLE_AFTER)
            lines += observe_lines(BY_KEY["vmov_cm"], Regs(s=(0,), d=5, role="A"), A_OFF)
            lines += nops(4) + [f"  padda [p1], #{RECORD_BYTES}"] + nops(4)
            chunks.append("\n".join(lines))
            pts.append({"kind": "neg", "id": len(pts), "op": name, "cfg": cfg, "data": data,
                        "shift16": sh, "asm": asm})
        kernels.append(({"name": f"k{ki:03d}", "dom": "i", "points": pts, "in_size": IN_SIZE,
                         "out_size": len(pts) * RECORD_BYTES},
                        isa_gen.build_mega_program(chunks, len(pts) * RECORD_BYTES)))
    return write_probe_suite(out_dir, "collisions-negate", kernels, {})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("out_dir")
    g.add_argument("--limit", type=int)
    g.add_argument("--only-new", action="store_true",
                   help="only pairs with an instruction from CATALOGUE_2B (phase 2b)")
    g.add_argument("--unit", help="only collisions on this unit (e.g. UPPER_SRS)")
    n = sub.add_parser("negate")
    n.add_argument("out_dir")
    c = sub.add_parser("calibrate")
    c.add_argument("out_dir")
    sub.add_parser("list")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        res = reservations_of_catalogue()
        colls = derive_collisions(res)
        print(f"{len(CATALOGUE)} instructions, {len(colls)} (A, B, g) collisions")
        for c_ in colls:
            print(c_["a"], c_["b"], c_["g"], ",".join(c_["units"]))
        return 0
    if args.cmd == "negate":
        generate_negate_probe(args.out_dir)
        return 0
    if args.cmd == "calibrate":
        generate_calibration(args.out_dir)
        return 0
    m = generate(args.out_dir, args.limit, args.only_new, args.unit)
    pts = [p for k in m["kernels"] for p in k["points"]]
    print(f"{len(m['kernels'])} kernels, {len(pts)} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
