#!/usr/bin/env python3
"""Generate the floating-point status-flag probe suite: srFPFlags, srF2IFlags, srF2FFlags.

TableGen (`AIE2GenFixupInstrInfo.td`) lists
  `Defs = [srFPFlags]`  on vadd.f, vsub.f, vnegadd.f, vnegsub.f, vneg.f, vclr (bm), vmul.f,
                        vnegmul.f, vmac.f, vmsc.f, vaddmac.f, vaddmsc.f (Uses crFPMask);
  `Defs = [srF2FFlags]` on vconv.bf16.fp32 and vst[.2d|.3d].conv.bf16.fp32 (Uses crF2FMask, crRnd);
  `Defs = [srF2IFlags]` on vfloor.s32.bf16 (accumulator and vector source; Uses crF2IMask).
All six registers keep 5 bits on silicon. This suite asks silicon which bit means what, when a
bit is raised, whether it is sticky and when it becomes visible.

Each case is one hand-written assembly kernel probing one form over a loop of RECORDS. A record
carries two operand sets S1 and S2 (each X, Y: 32 bf16; C1, C2: 16 fp32), the conf word (MAC and
accumulator forms), the shift (vfloor), crRnd (conversions) and an initial flag value. Per
record the kernel writes the initial value into the form's flag register (the two others are
cleared), runs the form on S1, reads the flag in each of the READS bundles after it, runs it on
S2, reads it in each of the READS bundles after that, and stores both results, the flag words,
the final value of all three flag registers and the three mask registers.

The kernels are assembly so every bundle offset is fixed. Generators only write suites; build
and run with `python3 -m harness suite build|hw|emu|compare <dir> ...`, analyze with `--analyze`.

  python3 tools/gen_fp_flags_probe.py --suite-dir <dir> [--seed N] [--only FORM ...]
  python3 tools/gen_fp_flags_probe.py --analyze <dir> <tag> [--verbose]
"""
from __future__ import annotations

import argparse
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import suite as S  # noqa: E402

GENERATOR = "tools/gen_fp_flags_probe.py"

# Record layout. Input: S1 and S2 (X 64 | Y 64 | C1 64 | C2 64 bytes each), then the parameter
# words. Output: S1 result, S2 result (64 bytes each), then the flag words.
SET_BYTES = 256
PARAM_OFF = 2 * SET_BYTES
PARAMS = ("shift", "conf", "rnd", "init", "mask")
REC_IN = PARAM_OFF + 32
RES_BYTES = 64
FLAG_OFF = 2 * RES_BYTES
READS = 12                      # flag reads after each operation, at bundle offsets 1..READS
F_INIT = 0
F_A = 1                         # F_A + k - 1 = read at offset k after S1
F_B = F_A + READS
F_FINAL = F_B + READS           # final value of the form's own flag
F_OTHERS = F_FINAL + 1          # final srFPFlags, srF2IFlags, srF2FFlags (all three)
F_MASKS = F_OTHERS + 3          # crFPMask, crF2IMask, crF2FMask
FLAG_WORDS = F_MASKS + 3
REC_OUT = FLAG_OFF + 4 * 48
CASE_BYTES = 24 * 1024

FLAGS = ("srFPFlags", "srF2IFlags", "srF2FFlags")
MASKS = ("crFPMask", "crF2IMask", "crF2FMask")

# bf16 MAC conf words (aiev2_vmult.h: amode 2, bmode 3): dense 4x8x4 and 16-lane elementwise.
CONF_DENSE = 2 << 1 | 3 << 3
CONF_ELEM = CONF_DENSE | 1 << 5


@dataclass(frozen=True)
class Form:
    name: str
    family: str         # "fp", "f2f", "f2i"
    op: str             # assembly template (see op_line)
    kind: str           # fp: acc2, acc1, clr, mul, mac, addmac; f2f: reg, store; f2i: am, w
    addressing: str = ""

    @property
    def flag(self) -> str:
        return {"fp": "srFPFlags", "f2f": "srF2FFlags", "f2i": "srF2IFlags"}[self.family]


def all_forms() -> list[Form]:
    forms = [Form(m.replace(".", "_"), "fp", f"{m} {{dst}}, {{c1}}, {{c2}}, r13", "acc2")
             for m in ("vadd.f", "vsub.f", "vnegadd.f", "vnegsub.f")]
    forms.append(Form("vneg_f", "fp", "vneg.f {dst}, {c1}, r13", "acc1"))
    forms.append(Form("vclr_bf", "fp", "vclr {dst}", "clr"))
    forms += [Form(m.replace(".", "_"), "fp", f"{m} {{dst}}, {{x}}, {{y}}, r13", "mul")
              for m in ("vmul.f", "vnegmul.f")]
    forms += [Form(m.replace(".", "_"), "fp", f"{m} {{dst}}, {{c1}}, {{x}}, {{y}}, r13", "mac")
              for m in ("vmac.f", "vmsc.f")]
    forms += [Form(m.replace(".", "_"), "fp", f"{m} {{c1}}, {{c1}}, {{c2}}, {{x}}, {{y}}, r13", "addmac")
              for m in ("vaddmac.f", "vaddmsc.f")]
    forms.append(Form("vconv_bf16_fp32", "f2f", "vconv.bf16.fp32 {w}, {c1}", "reg"))
    for adr in ("idx_imm", "idx", "pstm", "pstm_imm", "2d", "3d"):
        pre = {"2d": "vst.2d.conv", "3d": "vst.3d.conv"}.get(adr, "vst.conv")
        forms.append(Form(f"{pre.replace('.', '_')}_bf16_fp32_{adr}", "f2f",
                          f"{pre}.bf16.fp32 {{c1}}, {{mem}}", "store", adr))
    forms.append(Form("vfloor_s32_bf16_am", "f2i", "vfloor.s32.bf16 {xo}, {am}, s0", "am"))
    forms.append(Form("vfloor_s32_bf16_w", "f2i", "vfloor.s32.bf16 {xo}, {wsrc}, s0", "w"))
    return forms


# -- values -----------------------------------------------------------------------------------

def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf(x: float) -> int:
    """bf16 bits of `x` (x must be exactly representable)."""
    b = f32(x)
    assert b & 0xFFFF == 0, x
    return b >> 16


def pow2_f32(e: int, sign: int = 0) -> int:
    """fp32 bits of +-2^e, denormal below 2^-126."""
    mag = (e + 127) << 23 if e >= -126 else 1 << (e + 149)
    return sign << 31 | mag


def pow2_bf(e: int, sign: int = 0) -> int:
    assert -126 <= e <= 127, e
    return sign << 15 | (e + 127) << 7


BF_ONE, BF_TWO, BF_NAN, BF_SNAN, BF_INF, BF_NINF = 0x3F80, 0x4000, 0x7FC0, 0x7F81, 0x7F80, 0xFF80
F_ONE, F_NAN, F_SNAN, F_INF, F_NINF, F_MAX = 0x3F800000, 0x7FC00000, 0x7F800001, 0x7F800000, 0xFF800000, 0x7F7FFFFF


@dataclass
class OpSet:
    x: list[int]        # 32 bf16
    y: list[int]        # 32 bf16
    c1: list[int]       # 16 fp32
    c2: list[int]       # 16 fp32

    @staticmethod
    def zero() -> "OpSet":
        return OpSet([0] * 32, [0] * 32, [0] * 16, [0] * 16)

    def to_bytes(self) -> bytes:
        return (struct.pack("<32H", *self.x) + struct.pack("<32H", *self.y)
                + struct.pack("<16I", *self.c1) + struct.pack("<16I", *self.c2))


def place(s: OpSet, elementwise: bool, lane: int, terms: list[tuple[int, int]]) -> None:
    """Put product terms (a, b) into accumulator `lane` and nowhere else (the other lanes'
    products stay zero). Dense 4x8x4: lane (i, j) sums A[8i+k] * B[4k+j]; elementwise: lane L
    sums A[L] * B[L] + A[L+16] * B[L+16]."""
    if elementwise:
        assert len(terms) <= 2
        for k, (a, b) in enumerate(terms):
            s.x[lane + 16 * k], s.y[lane + 16 * k] = a, b
    else:
        i, j = divmod(lane, 4)
        for k, (a, b) in enumerate(terms):
            s.x[8 * i + k], s.y[4 * k + j] = a, b


# FP kinds: each a function (set, elementwise, lane, rng) filling one lane's terms and
# accumulators with a condition. The accumulator forms read c1 (and c2); the MAC forms the
# products, plus c1 (mac) and c2 (addmac).
def fp_kind(s: OpSet, kind: str, elem: bool, lane: int, rng: random.Random) -> None:
    def acc(v1: int, v2: int = 0) -> None:
        s.c1[lane], s.c2[lane] = v1, v2

    if kind == "none":
        place(s, elem, lane, [(BF_ONE, BF_TWO)])
        acc(F_ONE, f32(3.0))
    elif kind == "inexact":         # a term below the grid of the largest one vanishes
        place(s, elem, lane, [(BF_ONE, BF_ONE), (pow2_bf(-30), BF_ONE)])
        acc(F_ONE, 0)
    elif kind == "inexact_acc":     # the accumulator pair alone is inexact (1 + 2^-30)
        acc(F_ONE, pow2_f32(-30))
    elif kind == "overflow":        # finite terms whose sum exceeds the largest finite value
        place(s, elem, lane, [(pow2_bf(127), pow2_bf(1))])
        acc(F_MAX, F_MAX)
    elif kind == "underflow":       # a nonzero sum below 2^-126 (flushed to zero)
        place(s, elem, lane, [(pow2_bf(-70), pow2_bf(-70))])
        acc(pow2_f32(-130), pow2_f32(-131))
    elif kind == "tiny_exact":      # 2^-126 exactly: the smallest normal, no flush
        place(s, elem, lane, [(pow2_bf(-63), pow2_bf(-63))])
        acc(pow2_f32(-126), 0)
    elif kind == "denorm_in":       # a denormal bf16 input times zero, exact normal result
        place(s, elem, lane, [(0x0040, 0), (BF_ONE, BF_ONE)])
        acc(F_ONE, 1 << 31)
    elif kind == "denorm_acc":      # denormal accumulators summing to exactly 2^-126
        place(s, elem, lane, [(pow2_bf(-64), pow2_bf(-63))])
        acc(0x00400000, 0x00400000)
    elif kind == "nan_in":
        place(s, elem, lane, [(BF_NAN, BF_ONE)])
        acc(F_NAN, F_ONE)
    elif kind == "snan_in":
        place(s, elem, lane, [(BF_SNAN, BF_ONE)])
        acc(F_SNAN, F_ONE)
    elif kind == "inf_in":
        place(s, elem, lane, [(BF_INF, BF_ONE)])
        acc(F_INF, F_ONE)
    elif kind == "inf_inf":         # Inf - Inf
        place(s, elem, lane, [(BF_INF, BF_ONE), (BF_NINF, BF_ONE)])
        acc(F_INF, F_NINF)
    elif kind == "zero_inf":        # 0 * Inf (MAC); accumulators carry nothing special
        place(s, elem, lane, [(0, BF_INF)])
        acc(F_ONE, F_ONE)
    elif kind == "cancel":          # an exact zero sum
        place(s, elem, lane, [(BF_ONE, BF_ONE), (BF_ONE, 0xBF80)])
        acc(F_ONE, F_ONE | 1 << 31)
    elif kind == "cancel_lost":     # exact zero sum whose two small terms vanish on the grid
        if elem:                    # (elementwise: the accumulators carry the large pair)
            place(s, elem, lane, [(pow2_bf(-30), BF_ONE), (pow2_bf(-30, 1), BF_ONE)])
            acc(F_ONE, F_ONE | 1 << 31)
        else:
            place(s, elem, lane, [(BF_ONE, BF_ONE), (BF_ONE, 0xBF80), (pow2_bf(-30), BF_ONE),
                                  (pow2_bf(-30, 1), BF_ONE)])
            acc(0, 0)
    elif kind == "inf_and_ovf":     # an Inf term next to a finite product that overflows
        place(s, elem, lane, [(BF_INF, BF_ONE), (pow2_bf(127), pow2_bf(1))])
        acc(F_INF, F_MAX)
    elif kind == "nan_and_ovf":
        place(s, elem, lane, [(BF_NAN, BF_ONE), (pow2_bf(127), pow2_bf(1))])
        acc(F_NAN, F_MAX)
    elif kind == "rand":            # moderate random normals (inexact almost surely)
        place(s, elem, lane, [(rng.randrange(0x3C00, 0x4300) | rng.getrandbits(1) << 15,
                               rng.randrange(0x3C00, 0x4300)) for _ in range(2 if elem else 8)])
        acc(rng.randrange(0x3C000000, 0x43000000), rng.randrange(0x3C000000, 0x43000000))
    else:
        raise ValueError(kind)


FP_KINDS = ("none", "inexact", "inexact_acc", "overflow", "underflow", "tiny_exact", "denorm_in", "denorm_acc",
            "nan_in", "snan_in", "inf_in", "inf_inf", "zero_inf", "cancel", "rand")

# F2F (fp32 -> bf16) source lanes per kind; the other lanes hold exact small values.
F2F_KINDS = {
    "none": F_ONE, "inexact": F_ONE | 0x0400, "tie": F_ONE | 0x8000, "tie_odd": F_ONE | 0x18000,
    "overflow": F_MAX, "max_exact": 0x7F7F0000, "nan": F_NAN, "snan": F_SNAN, "nan_lowbits": 0x7FC00001,
    "inf": F_INF, "ninf": F_NINF, "denorm": 0x00000001, "denorm_exact": 0x00010000,
    "denorm_big": 0x007FFFFF, "min_normal": 0x00800000, "below_min": 0x00800001, "neg_zero": 0x80000000,
    "neg_inexact": 0xBF800400,
}

# F2I (bf16 -> s32 at shift s) source lanes per kind.
F2I_KINDS = {
    "none": BF_TWO, "frac": bf(1.5), "neg_frac": bf(-1.5), "small_frac": bf(0.25),
    "neg_small": bf(-0.25), "big": pow2_bf(30), "ovf": pow2_bf(31), "neg_min": pow2_bf(31, 1),
    "neg_ovf": pow2_bf(32, 1), "huge": pow2_bf(100), "nan": BF_NAN, "inf": BF_INF, "ninf": BF_NINF,
    "denorm": 0x0001, "neg_denorm": 0x8001, "neg_zero": 0x8000,
}


def make_set(form: Form, kind: str, lane: int, elem: bool, rng: random.Random) -> OpSet:
    s = OpSet.zero()
    if form.family == "fp":
        # Every lane computes an exact small value; `lane` carries the condition.
        for L in range(16):
            fp_kind(s, "none", elem, L, rng)
        fp_kind(s, kind, elem, lane, rng)
        return s
    if form.family == "f2f":
        s.c1 = [f32(float(L + 1)) for L in range(16)]
        if kind == "all_inexact":
            s.c1 = [F_ONE | 0x0400 | L for L in range(16)]
        elif kind.startswith("z_"):         # every other lane +0
            s.c1 = [0] * 16
            s.c1[lane] = F2F_KINDS[kind[2:]]
        else:
            s.c1[lane] = F2F_KINDS[kind]
        return s
    # f2i: the source is X's low 256 bits (vector form) or C1's low 256 bits (accumulator form).
    src = [bf(float(L + 1)) for L in range(16)]
    src[lane] = F2I_KINDS[kind]
    s.x[:16] = src
    s.c1[:8] = [src[2 * i] | src[2 * i + 1] << 16 for i in range(8)]
    return s


@dataclass(frozen=True)
class Record:
    a: str
    b: str
    lane_a: int = 0
    lane_b: int = 0
    shift: int = 0
    conf: int = 0
    rnd: int = 0
    init: int = 0
    mask: int = 0x1F            # written to all three mask registers

    def as_list(self) -> list:
        return [self.a, self.b, self.lane_a, self.lane_b, self.shift, self.conf, self.rnd, self.init,
                self.mask]


def records_for(form: Form) -> list[Record]:
    recs: list[Record] = []
    if form.family == "fp":
        if form.kind == "clr":
            return [Record("none", "none", init=i) for i in (0, 1, 2, 4, 8, 16, 31)]
        confs = (CONF_DENSE, CONF_ELEM) if form.kind in ("mul", "mac", "addmac") else (0,)
        kinds = FP_KINDS if form.kind != "acc1" else tuple(k for k in FP_KINDS if k not in ("inf_inf",))
        for conf in confs:
            recs += [Record(k, "none", 0, 0, conf=conf) for k in kinds]
            recs += [Record(k, "none", 15, 0, conf=conf) for k in ("inexact", "overflow", "nan_in")]
        conf = confs[0]
        # Stickiness and accumulation: two conditions in a row, from a nonzero initial value.
        recs += [Record(a, b, conf=conf, init=i)
                 for a, b in (("overflow", "none"), ("none", "nan_in"), ("inexact", "inf_inf"),
                              ("underflow", "overflow"), ("none", "none"))
                 for i in (0, 31)]
        recs += [Record("none", "none", conf=conf, init=i) for i in (1, 2, 4, 8, 16)]
        # The conf word's sign bits on the accumulator / product terms (zero_acc, sub_*).
        recs += [Record("overflow", "inf_inf", conf=conf | m) for m in (1, 1 << 11, 1 << 12, 1 << 13)]
        return recs
    if form.family == "f2f":
        recs = [Record(k, "none", 0) for k in F2F_KINDS] + [Record("all_inexact", "none")]
        recs += [Record(k, "none", 15) for k in ("inexact", "overflow", "nan")]
        recs += [Record(k, "none", 0, rnd=r) for k in ("inexact", "tie", "overflow", "neg_inexact")
                 for r in range(16) if r != 0]
        recs += [Record(a, b, init=i) for a, b in (("inexact", "none"), ("none", "overflow"),
                                                     ("nan", "inexact"), ("none", "none"))
                 for i in (0, 31)]
        recs += [Record("none", "none", init=i) for i in (1, 2, 4, 8, 16)]
        return recs
    recs = [Record(k, "none", 0) for k in F2I_KINDS]
    recs += [Record(k, "none", 15) for k in ("frac", "ovf", "nan")]
    recs += [Record(k, "none", 0, shift=sh) for k in ("none", "frac", "big", "small_frac", "neg_frac")
             for sh in (1, 2, 4, 8, -1, -2, 31, 32, 63)]
    recs += [Record(a, b, init=i) for a, b in (("frac", "none"), ("none", "ovf"), ("nan", "frac"),
                                                 ("none", "none"))
             for i in (0, 31)]
    recs += [Record("none", "none", init=i) for i in (1, 2, 4, 8, 16)]
    return recs


def record_bytes(form: Form, rec: Record, rng: random.Random) -> bytes:
    elem = bool(rec.conf >> 5 & 1)
    s1 = make_set(form, rec.a, rec.lane_a, elem, rng)
    s2 = make_set(form, rec.b, rec.lane_b, elem, rng)
    params = struct.pack("<5I", rec.shift & 0xFFFFFFFF, rec.conf, rec.rnd, rec.init, rec.mask)
    return (s1.to_bytes() + s2.to_bytes() + params).ljust(REC_IN, b"\0")


# -- kernel text ------------------------------------------------------------------------------

def nops(n: int) -> list[str]:
    return ["  nop"] * n


# Only caller-saved registers (AIE2CallingConv.td CSR_AIE2 keeps lr, r16..r23, p6, p7): r0..r11
# flag reads (r0 also scratch), r12..r14 parameters, r15 the record count, r24..r30 the caller's
# control and status values (silicon keeps them across kernels).
READ_REGS = [f"r{k}" for k in range(READS)]
SAVED = ("crRnd", "srFPFlags", "srF2IFlags", "srF2FFlags", "crFPMask", "crF2IMask", "crF2FMask")
SAVE_REGS = ("r24", "r25", "r26", "r27", "r28", "r29", "r30")

# Per operand set: X, Y, C1, C2, destination accumulator, conversion destinations.
SETS = [dict(x="x0", y="x2", c1="bml1", c2="bml2", dst="bml0", w="wl8", xo="x8", wsrc="wl0", am="amll1"),
        dict(x="x4", y="x6", c1="bml3", c2="bml4", dst="bml5", w="wl10", xo="x10", wsrc="wl4", am="amll3")]


def prologue(n: int) -> list[str]:
    lines = ["  // ---- prologue: keep the caller's control and status registers ----"]
    lines += [f"  mov {r}, {reg}" for reg, r in zip(SAVED, SAVE_REGS)]
    lines += [f"  movxm r15, #{n}",
              f"  movxm r12, #{REC_IN}", "  mov m6, r12",
              f"  movxm r12, #{REC_OUT}", "  mov m7, r12",
              "  mov r12, #0"]
    for i in (0, 1):
        lines += [f"  mov m{i}, r12", f"  mov dn{i}, r12", f"  mov dj{i}, r12", f"  mov dc{i}, r12"]
    lines += nops(4)
    return lines


def epilogue() -> list[str]:
    lines = ["  // ---- epilogue ----"]
    for reg, r in zip(SAVED, SAVE_REGS):
        lines.append(f"  movx {reg}, {r}" if reg.startswith("cr") else f"  mov {reg}, {r}")
    lines += nops(4) + ["  ret lr"] + nops(5)
    return lines


def operand_loads(form: Form) -> list[str]:
    lines = []
    for si, regs in enumerate(SETS):
        base = si * SET_BYTES
        x, y = int(regs["x"][1:]), int(regs["y"][1:])
        c1, c2 = int(regs["c1"][3:]), int(regs["c2"][3:])
        loads = [(f"wl{x}", 0), (f"wh{x}", 32), (f"wl{y}", 64), (f"wh{y}", 96),
                 (f"amll{c1}", 128), (f"amlh{c1}", 160), (f"amll{c2}", 192), (f"amlh{c2}", 224)]
        lines += [f"  vlda {r}, [p0, #{base + off}]" for r, off in loads]
    return lines


def pointer_setup(form: Form, si: int) -> list[str]:
    if form.kind != "store":
        return []
    lines = ["  mov p2, p1"]
    if si:
        lines += [f"  movxm r0, #{RES_BYTES}", "  mov m5, r0", "  padda [p2], m5"]
    return lines + nops(2)


def op_line(form: Form, si: int) -> list[str]:
    regs = dict(SETS[si])
    mem = {"idx_imm": "[p2, #0]", "idx": "[p2, dj0]", "pstm": "[p2], m0", "pstm_imm": "[p2], #0",
           "2d": "[p2], d0", "3d": "[p2], d0"}.get(form.addressing, "")
    return ["  " + form.op.format(mem=mem, **regs)]


def result_stores(form: Form) -> list[str]:
    lines = []
    for si, regs in enumerate(SETS):
        off = si * RES_BYTES
        if form.family == "fp":
            r = regs["c1"] if form.kind == "addmac" else regs["dst"]
            n = int(r[3:])
            lines += [f"  vst amll{n}, [p1, #{off}]", f"  vst amlh{n}, [p1, #{off + 32}]"]
        elif form.family == "f2f":
            if form.kind == "reg":
                lines.append(f"  vst {regs['w']}, [p1, #{off}]")
        else:
            n = int(regs["xo"][1:])
            lines += [f"  vst wl{n}, [p1, #{off}]", f"  vst wh{n}, [p1, #{off + 32}]"]
    return lines


def flag_store(reg: str, word: int) -> str:
    # A scalar store's immediate is 8 bits signed: p3 addresses words 0..31, p5 words 32..
    return f"  st {reg}, [p3, #{4 * word}]" if word < 32 else f"  st {reg}, [p5, #{4 * (word - 32)}]"


def record_body(form: Form, clear_at: int = 0, mask_at: int = 0) -> list[str]:
    p = {name: 4 * i for i, name in enumerate(PARAMS)}
    lines = ["  // ---- one record ----",
             "  mov p4, p0", f"  movxm r0, #{PARAM_OFF}", "  mov m5, r0", "  padda [p4], m5",
             "  mov p3, p1", f"  movxm r0, #{FLAG_OFF}", "  mov m5, r0", "  padda [p3], m5",
             "  mov p5, p1", f"  movxm r0, #{FLAG_OFF + 128}", "  mov m5, r0", "  padda [p5], m5"]
    lines += nops(2)
    lines += [f"  lda r12, [p4, #{p['shift']}]",
              f"  lda r13, [p4, #{p['conf']}]",
              f"  lda r14, [p4, #{p['rnd']}]",
              f"  lda r1, [p4, #{p['mask']}]",
              f"  lda r2, [p4, #{p['init']}]"]
    lines += operand_loads(form)
    lines += nops(8)
    lines += ["  mov s0, r12", "  movx crRnd, r14", "  mov r0, #0"]
    lines += [f"  movx {m}, r1" for m in MASKS]
    lines += [f"  mov {f}, r0" for f in FLAGS if f != form.flag]
    lines += [f"  mov {form.flag}, r2"]
    lines += nops(6)
    lines += [f"  mov r0, {form.flag}"] + nops(6) + [flag_store("r0", F_INIT)]
    for si, base in ((0, F_A), (1, F_B)):
        lines += pointer_setup(form, si)
        lines += nops(8)
        if si == 0 and (clear_at or mask_at):
            # The read slot the write replaces would keep a stale register: zero it.
            lines.append(f"  mov {READ_REGS[(clear_at or mask_at) - 1]}, #0")
        lines += op_line(form, si)
        # `clear_at` k: the read at offset k after S1 is replaced by `mov flag, r14` (the record's
        # rnd word, 0 in the clear plan): how a write races the operation's flag.
        # `mask_at` k: the same slot writes the form's mask register from r14 instead (when the
        # mask is sampled).
        mask = MASKS[FLAGS.index(form.flag)]
        lines += [f"  mov {form.flag}, r14" if si == 0 and k + 1 == clear_at
                  else f"  movx {mask}, r14" if si == 0 and k + 1 == mask_at
                  else f"  mov {r}, {form.flag}"
                  for k, r in enumerate(READ_REGS)]
        lines += nops(2)
        lines += [flag_store(r, base + k) for k, r in enumerate(READ_REGS)]
    lines += nops(8)
    lines += [f"  mov r0, {form.flag}"]
    lines += [f"  mov r{1 + i}, {f}" for i, f in enumerate(FLAGS)]
    lines += [f"  mov r{4 + i}, {m}" for i, m in enumerate(MASKS)]
    lines += nops(6) + [flag_store("r0", F_FINAL)]
    lines += [flag_store(f"r{1 + i}", F_OTHERS + i) for i in range(3)]
    lines += [flag_store(f"r{4 + i}", F_MASKS + i) for i in range(3)]
    lines += result_stores(form)
    lines += nops(2)
    lines += ["  padda [p0], m6", "  padda [p1], m7"]
    return lines


def kernel_text(form: Form, n: int, clear_at: int = 0, mask_at: int = 0) -> str:
    lines = [f"// Auto-generated ({GENERATOR}): {form.name} over {n} records",
             ".text", ".globl test_kernel", "test_kernel:"]
    lines += prologue(n)
    lines += [".p2align 4", ".Lrec:"]
    lines += record_body(form, clear_at, mask_at)
    lines += ["  add r15, r15, #-1"] + nops(2) + ["  jnz r15, #.Lrec"] + nops(5)
    lines += epilogue()
    return "\n".join(lines) + "\n"


def cases_for(form: Form, rng: random.Random) -> list[tuple[str, str, bytes, dict]]:
    recs = records_for(form)
    per_case = max(1, CASE_BYTES // (REC_IN + REC_OUT))
    out = []
    for k, start in enumerate(range(0, len(recs), per_case)):
        chunk = recs[start:start + per_case]
        data = b"".join(record_bytes(form, r, rng) for r in chunk)
        name = form.name if len(recs) <= per_case else f"{form.name}__{k:02d}"
        meta = {"form": form.name, "family": form.family, "op": form.op, "kind": form.kind,
                "addressing": form.addressing, "flag": form.flag,
                "records": [r.as_list() for r in chunk], "rec_in": REC_IN, "rec_out": REC_OUT}
        out.append((name, kernel_text(form, len(chunk)), data, meta))
    return out


MASKS_SWEEP = (0, 1, 2, 4, 8, 16, 0x0F, 0x15)


def extra_cases(rng: random.Random) -> list[tuple[str, str, bytes, dict]]:
    """The `extra` plan: rounding-lost cancellations and specials next to an overflow, the masks
    one bit at a time, and a write to the flag register racing the operation (`clear_at`)."""
    forms = {f.name: f for f in all_forms()}
    groups: list[tuple[str, Form, list[Record], int, int]] = []
    for fname in ("vmac_f", "vaddmac_f", "vmul_f"):
        recs = [Record(k, "none", conf=c) for c in (CONF_DENSE, CONF_ELEM)
                for k in ("cancel_lost", "inf_and_ovf", "nan_and_ovf")]
        groups.append((f"{fname}_lost", forms[fname], recs, 0, 0))
    mask_pairs = {"vmac_f": [("underflow", "overflow"), ("nan_in", "inf_in")],
                  "vconv_bf16_fp32": [("denorm", "nan"), ("inf", "neg_zero")],
                  "vfloor_s32_bf16_w": [("small_frac", "ovf"), ("nan", "neg_zero")]}
    for fname, pairs in mask_pairs.items():
        conf = CONF_DENSE if forms[fname].family == "fp" else 0
        recs = [Record(a, b, conf=conf, mask=m) for m in MASKS_SWEEP for a, b in pairs]
        groups.append((f"{fname}_masks", forms[fname], recs, 0, 0))
    for fname, kind, offsets in (("vmac_f", "overflow", range(1, 13)), ("vadd_f", "overflow", range(5, 10)),
                                 ("vconv_bf16_fp32", "nan", range(1, 4)),
                                 ("vfloor_s32_bf16_w", "ovf", range(1, 5))):
        conf = CONF_DENSE if fname == "vmac_f" else 0
        for k in offsets:
            recs = [Record(kind, "none", conf=conf), Record(kind, "none", conf=conf, init=0x1F)]
            groups.append((f"{fname}_clear{k:02d}", forms[fname], recs, k, 0))
    # The mask starts at 0 and is set to 0x1F (the record's rnd word) at offset k after S1.
    for fname, kind, offsets in (("vmac_f", "overflow", range(1, 9)), ("vconv_bf16_fp32", "nan", range(1, 3)),
                                 ("vfloor_s32_bf16_w", "ovf", range(1, 4))):
        conf = CONF_DENSE if fname == "vmac_f" else 0
        for k in offsets:
            recs = [Record(kind, "none", conf=conf, rnd=0x1F, mask=0)]
            groups.append((f"{fname}_maskat{k:02d}", forms[fname], recs, 0, k))
    out = []
    for name, form, recs, clear_at, mask_at in groups:
        data = b"".join(record_bytes(form, r, rng) for r in recs)
        meta = {"form": form.name, "family": form.family, "op": form.op, "kind": form.kind,
                "addressing": form.addressing, "flag": form.flag, "clear_at": clear_at, "mask_at": mask_at,
                "records": [r.as_list() for r in recs], "rec_in": REC_IN, "rec_out": REC_OUT}
        out.append((name, kernel_text(form, len(recs), clear_at, mask_at), data, meta))
    return out


def generate_extra(suite_dir: str | Path, seed: int = 1) -> list[str]:
    names = []
    for name, text, data, meta in extra_cases(random.Random(seed)):
        S.write_single_core_case(suite_dir, name, text, input_bytes=data,
                                 out_bytes=REC_OUT * len(meta["records"]), meta=meta)
        names.append(name)
    S.write_suite_json(suite_dir, "fpflags-extra", GENERATOR, names, meta={"seed": seed, "plan": "extra"},
                       hw_checks=["aligned_labels"])
    return names


def generate(suite_dir: str | Path, seed: int = 1, only: list[str] | None = None) -> list[str]:
    rng = random.Random(seed)
    names = []
    for form in all_forms():
        cases = cases_for(form, rng)        # always draw, so --only keeps the same data
        if only and form.name not in only:
            continue
        for name, text, data, meta in cases:
            S.write_single_core_case(suite_dir, name, text, input_bytes=data,
                                     out_bytes=REC_OUT * len(meta["records"]), meta=meta)
            names.append(name)
    S.write_suite_json(suite_dir, "fpflags", GENERATOR, names, meta={"seed": seed},
                       hw_checks=["aligned_labels"])
    return names


# -- analysis ---------------------------------------------------------------------------------

def flag_words(out: bytes, i: int) -> list[int]:
    return list(struct.unpack_from(f"<{FLAG_WORDS}I", out, i * REC_OUT + FLAG_OFF))


def describe(words: list[int]) -> str:
    a = words[F_A:F_A + READS]
    b = words[F_B:F_B + READS]

    def runs(seq: list[int]) -> str:
        out, prev = [], None
        for k, v in enumerate(seq, 1):
            if v != prev:
                out.append(f"{k}:{v:#x}")
                prev = v
        return ",".join(out)

    return (f"init {words[F_INIT]:#x} A[{runs(a)}] B[{runs(b)}] final {words[F_FINAL]:#x} "
            f"all {[hex(w) for w in words[F_OTHERS:F_OTHERS + 3]]} masks {[hex(w) for w in words[F_MASKS:F_MASKS + 3]]}")


def dump(suite_dir: str, tag: str) -> None:
    suite = S.load_suite(suite_dir)
    for case in suite.cases:
        meta = suite.case_meta(case)
        out_path = suite.output(tag, case)
        if not out_path.is_file():
            print(f"{case}: no output")
            continue
        out = out_path.read_bytes()
        for i, r in enumerate(meta["records"]):
            print(f"{case} {r}: {describe(flag_words(out, i))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite-dir")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="form names")
    ap.add_argument("--plan", choices=("main", "extra"), default="main")
    ap.add_argument("--dump", nargs=2, metavar=("DIR", "TAG"))
    args = ap.parse_args()
    if args.dump:
        dump(*args.dump)
        return
    if not args.suite_dir:
        ap.error("--suite-dir or --dump")
    names = (generate_extra(args.suite_dir, args.seed) if args.plan == "extra"
             else generate(args.suite_dir, args.seed, args.only))
    print(f"{len(names)} cases -> {args.suite_dir}")


if __name__ == "__main__":
    main()
