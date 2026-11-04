#!/usr/bin/env python3
"""Generate the saturation-flag probe suite: when srSRS_of / srUPS_of are set, and how.

TableGen (`AIE2GenFixupInstrInfo.td`) lists `Defs = [srSRS_of]` on every SRS form (`vsrs`,
`vsrsm`, `vst[.2d|.3d].srs`) and `Defs = [srUPS_of]` on every UPS form (`vups`,
`vlda[.2d|.3d].ups`); the itinerary (`AIE2Schedule.td`) puts the def at operand cycle 3
(vsrs, vsrsm, vst.srs), 1 (vups) and 8 (vlda.ups). This suite asks silicon what the flag
holds and when.

Each case is one hand-written assembly kernel probing one form over a loop of RECORDS. A
record carries two operands A and B, the shift, the sign word (crSRSSign / crUPSSign), crSat,
crRnd and an initial flag value. Per record the kernel writes the flag, converts A, reads the
flag in each of the 12 bundles after A, converts B, reads it in each of the 12 bundles after B,
and stores both results and the 26 flag words. The records cross operand kinds (no lane out of
range, every lane, one lane at the start / middle / end, a lane only one signedness overflows,
edge values, random) with every crSat, both signs and two shifts, then pair a saturating and a
non-saturating conversion both ways round from either initial flag value (sticky or
overwritten), and for SRS sweep crRnd over values a rounding carry pushes out of range.

The kernels are assembly so every bundle offset is fixed: no compiler reorders the flag reads.
Generators only write suites; build and run with
`python3 -m harness suite build|hw|emu|compare <dir> ...`, analyze with `--analyze`.

  python3 tools/gen_sat_flags_probe.py --suite-dir <dir> [--seed N] [--only FORM ...]
  python3 tools/gen_sat_flags_probe.py --analyze <dir> <tag>
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
from gen_dynamic_sign_probe import pack_lanes  # noqa: E402

GENERATOR = "tools/gen_sat_flags_probe.py"

# Record layout. Input: A operand, B operand (128 bytes each, the widest accumulator), then the
# parameter words. Output: A result, B result (128 bytes each), then the flag words.
OPND_BYTES = 128
PARAM_OFF = 2 * OPND_BYTES
PARAMS = ("shift", "sign", "sat", "init", "rnd")
REC_IN = PARAM_OFF + 32
FLAG_OFF = 2 * OPND_BYTES
READS = 12                      # flag reads after each conversion, at bundle offsets 1..READS
FLAG_WORDS = 2 + 2 * READS      # init read-back, A reads, B reads, final read
REC_OUT = FLAG_OFF + 4 * 32     # flag area padded to 32 words
# Input plus output bytes per case: well inside one tile's data memory.
CASE_BYTES = 24 * 1024

# Flag word index within a record's flag area.
F_INIT = 0
F_A = 1                         # F_A + k - 1 = read at offset k after A
F_B = F_A + READS
F_FINAL = F_B + READS


@dataclass(frozen=True)
class Form:
    """One probed instruction form."""
    name: str           # case-name stem
    family: str         # "srs" (srSRS_of, crSRSSign) or "ups" (srUPS_of, crUPSSign)
    op: str             # assembly template; see `op_lines`
    kind: str           # "reg", "store", "load"
    acc_bits: int       # accumulator lane bits
    vec_bits: int       # vector (narrow) lane bits
    lanes: int
    dform: bool         # sign from the sign register (d) or the encoding (s)
    addressing: str = ""    # store / load forms: idx_imm, idx, pstm, pstm_imm, 2d, 3d

    @property
    def flag(self) -> str:
        return "srSRS_of" if self.family == "srs" else "srUPS_of"

    @property
    def sign_reg(self) -> str:
        return "crSRSSign" if self.family == "srs" else "crUPSSign"

    @property
    def acc_reg_bits(self) -> int:
        return self.acc_bits * self.lanes

    @property
    def vec_reg_bits(self) -> int:
        return self.vec_bits * self.lanes


def _acc(form: Form, idx: int) -> str:
    return {512: f"bml{idx}", 1024: f"cm{idx}"}[form.acc_reg_bits]


def _vec(form: Form, idx: int) -> str:
    return {256: f"wl{idx}", 512: f"x{idx}"}[form.vec_reg_bits]


def srs_forms() -> list[Form]:
    # (out, acc, lanes): the accumulator register is bml (512) or cm (1024); output wl.
    shapes = [(16, 32, 16), (8, 32, 32), (16, 64, 16), (32, 64, 8)]
    forms = []
    for out, acc, lanes in shapes:
        for d in (True, False):
            t = f"{'d' if d else 's'}{out}.s{acc}"
            forms.append(Form(f"vsrs_{t.replace('.', '_')}", "srs", f"vsrs.{t} {{dst}}, {{src}}, s0",
                              "reg", acc, out, lanes, d))
            forms.append(Form(f"vst_srs_{t.replace('.', '_')}", "srs",
                              f"vst.srs.{t} {{src}}, s0, {{mem}}", "store", acc, out, lanes, d,
                              "idx_imm"))
        # 2D / 3D stores and vsrsm: d forms only (the residual gap), plus one s form each.
    for out, acc, lanes in shapes:
        t = f"d{out}.s{acc}"
        for adr in ("2d", "3d"):
            forms.append(Form(f"vst_{adr}_srs_{t.replace('.', '_')}", "srs",
                              f"vst.{adr}.srs.{t} {{src}}, s0, {{mem}}", "store", acc, out, lanes,
                              True, adr))
    for out, acc, lanes in ((16, 32, 16), (32, 64, 8)):
        for d in (True, False):
            t = f"{'d' if d else 's'}{out}.s{acc}"
            forms.append(Form(f"vsrsm_{t.replace('.', '_')}", "srs", f"vsrsm.{t} {{dst}}, {{src}}, s0",
                              "reg", acc, out, lanes, d))
    return forms


def ups_forms() -> list[Form]:
    # (acc, src, lanes): w2b / w2c by the accumulator size; x2c reads a 512-bit x.
    shapes = [(32, 16, 16), (32, 8, 32), (64, 16, 16), (64, 32, 8)]
    forms = []
    for acc, src, lanes in shapes:
        for d in (True, False):
            t = f"s{acc}.{'d' if d else 's'}{src}"
            forms.append(Form(f"vups_{t.replace('.', '_')}", "ups", f"vups.{t} {{dst}}, {{src}}, s0",
                              "reg", acc, src, lanes, d))
            forms.append(Form(f"vlda_ups_{t.replace('.', '_')}", "ups",
                              f"vlda.ups.{t} {{dst}}, s0, {{mem}}", "load", acc, src, lanes, d,
                              "idx_imm"))
    for d in (True, False):
        t = f"s32.{'d' if d else 's'}16"
        forms.append(Form(f"vups_x2c_{t.replace('.', '_')}", "ups", f"vups.{t} {{dst}}, {{src}}, s0",
                          "reg", 32, 16, 32, d))
    # The other addressing modes of the fused load, d forms (the residual gap and the
    # pstm form whose TableGen record was questioned) and their s twins for pstm.
    for acc, src, lanes in shapes:
        for adr in ("idx", "pstm", "pstm_imm", "2d", "3d"):
            for d in ((True, False) if adr == "pstm" else (True,)):
                t = f"s{acc}.{'d' if d else 's'}{src}"
                pre = "vlda.2d.ups" if adr == "2d" else "vlda.3d.ups" if adr == "3d" else "vlda.ups"
                forms.append(Form(f"{pre.replace('.', '_')}_{t.replace('.', '_')}_{adr}", "ups",
                                  f"{pre}.{t} {{dst}}, s0, {{mem}}", "load", acc, src, lanes, d, adr))
    return forms


def all_forms() -> list[Form]:
    return srs_forms() + ups_forms()


# -- operand kinds ----------------------------------------------------------------------------

KINDS = ("none", "all", "lane_first", "lane_mid", "lane_last", "neg", "uhi", "smin", "edge", "rand")


def srs_lanes(form: Form, kind: str, shift: int, rng: random.Random) -> list[int]:
    """Accumulator lanes (signed ints) of an SRS operand of `kind` for `shift`.

    `none` keeps every lane inside both the signed and unsigned output range; `all` puts every
    lane out of both; `lane_*` puts one lane out of both; `neg` one lane at -1 (out of range
    unsigned only); `uhi` one lane at the unsigned maximum (out of range signed only); `smin` one
    lane at the signed minimum (inside, but outside the symmetric range of crSat 2/3)."""
    n, ob, ab = form.lanes, form.vec_bits, form.acc_bits
    umax, smax, smin = (1 << ob) - 1, (1 << (ob - 1)) - 1, -(1 << (ob - 1))
    amax, amin = (1 << (ab - 1)) - 1, -(1 << (ab - 1))

    def fit(v: int) -> int:
        return max(amin, min(amax, v))

    def small() -> int:
        return rng.randrange(0, smax // 2 + 1) << shift

    vals = [small() for _ in range(n)]
    one = {"lane_first": 0, "lane_mid": n // 2, "lane_last": n - 1, "neg": 1, "uhi": 2, "smin": 3}
    if kind == "none":
        pass
    elif kind == "all":
        vals = [fit((umax + 1 + rng.randrange(0, 1 << 8)) << shift) if i % 2 == 0
                else fit((smin - 1 - rng.randrange(0, 1 << 8)) << shift) for i in range(n)]
    elif kind in ("lane_first", "lane_mid", "lane_last"):
        vals[one[kind]] = fit((umax + 1) << shift)
    elif kind == "neg":
        vals[one[kind]] = fit(-1 << shift)
    elif kind == "uhi":
        vals[one[kind]] = fit(umax << shift)
    elif kind == "smin":
        vals[one[kind]] = fit(smin << shift)
    elif kind == "edge":
        picks = [umax, umax + 1, smax, smax + 1, smin, smin - 1, -1, 0, 1, -umax, 3, -3]
        vals = [fit(picks[(i + rng.randrange(len(picks))) % len(picks)] << shift) for i in range(n)]
    elif kind == "rand":
        vals = [rng.randrange(amin, amax + 1) for _ in range(n)]
    elif kind == "rnd":
        # Just below the signed and unsigned maxima: a rounding carry pushes the lane over.
        half = 1 << (shift - 1) if shift else 0
        base = [smax, umax, smin]
        vals = [fit((base[i % 3] << shift) + [half - 1, half, half + 1, (1 << shift) - 1][(i // 3) % 4]
                    if shift else base[i % 3]) for i in range(n)]
    else:
        raise ValueError(kind)
    return vals


def ups_lanes(form: Form, kind: str, rng: random.Random) -> list[int]:
    """Source lanes (unsigned bit patterns) of a UPS operand of `kind`. Whether a lane overflows
    depends on the shift and sign of the record: `none` holds small positive values, `all`
    the largest magnitudes of both signs, `neg` -1, `uhi` the unsigned maximum, `smin` the
    signed minimum (exactly representable signed at shift acc-src)."""
    n, b = form.lanes, form.vec_bits
    m, smax, smin = (1 << b) - 1, (1 << (b - 1)) - 1, 1 << (b - 1)

    def small() -> int:
        return rng.randrange(0, max(2, 1 << (b - 3)))

    vals = [small() for _ in range(n)]
    one = {"lane_first": 0, "lane_mid": n // 2, "lane_last": n - 1, "neg": 1, "uhi": 2, "smin": 3}
    if kind == "none":
        pass
    elif kind == "all":
        vals = [smax if i % 2 == 0 else smin for i in range(n)]
    elif kind in ("lane_first", "lane_mid", "lane_last"):
        vals[one[kind]] = smax
    elif kind == "neg":
        vals[one[kind]] = m
    elif kind == "uhi":
        vals[one[kind]] = m
        vals[(one[kind] + 1) % n] = smin + 1
    elif kind == "smin":
        vals[one[kind]] = smin
    elif kind == "edge":
        picks = [smin, smax, m, 0, 1, smin + 1, smax - 1, m - 1, 2]
        vals = [picks[(i + rng.randrange(len(picks))) % len(picks)] for i in range(n)]
    elif kind == "rand":
        vals = [rng.getrandbits(b) for _ in range(n)]
    else:
        raise ValueError(kind)
    return vals


def srs_shifts(form: Form) -> tuple[int, ...]:
    return (5,)


def ups_shifts(form: Form) -> tuple[int, ...]:
    fit = form.acc_bits - form.vec_bits
    return (fit, fit + 1)


@dataclass(frozen=True)
class Record:
    a: str
    b: str
    shift: int
    sign: int
    sat: int
    init: int
    rnd: int = 0

    def as_list(self) -> list:
        return [self.a, self.b, self.shift, self.sign, self.sat, self.init, self.rnd]


# The full sweep runs on the register forms and the immediate-offset fused load; the other
# addressing modes and the fused store share their datapath and take a lighter sweep.
LIGHT_KINDS = ("none", "all", "lane_mid", "neg", "uhi", "rand")


def is_primary(form: Form) -> bool:
    return (form.kind == "reg" and not form.name.startswith("vsrsm")) or (
        form.kind == "load" and form.addressing == "idx_imm")


def records_for(form: Form) -> list[Record]:
    signs = (0, 1) if form.dform else (0,)
    shifts = srs_shifts(form) if form.family == "srs" else ups_shifts(form)
    if is_primary(form):
        recs = [Record(k, "none", sh, sg, sat, 0)
                for sh in shifts for sat in (0, 1, 2, 3) for sg in signs for k in KINDS]
    else:
        recs = [Record(k, "none", shifts[-1], sg, sat, 0)
                for sat in (0, 1, 3) for sg in signs for k in LIGHT_KINDS]
    # Sticky or overwritten: a saturating and a non-saturating conversion both ways round,
    # from either initial flag value. The last shift overflows `all` for UPS.
    sh = shifts[-1]
    recs += [Record(a, b, sh, signs[-1], sat, init)
             for a, b in (("all", "none"), ("none", "all"), ("none", "none"), ("all", "all"))
             for sat in (0, 1) for init in (0, 1)]
    if form.family == "srs" and is_primary(form):
        recs += [Record("rnd", "none", 5, sg, 1, 0, rnd) for sg in signs for rnd in range(16)]
    return recs


def operand_bytes(form: Form, kind: str, shift: int, rng: random.Random) -> bytes:
    if form.family == "srs":
        vals = srs_lanes(form, kind, shift, rng)
        data = pack_lanes(form.acc_bits, vals)
    else:
        data = pack_lanes(form.vec_bits, ups_lanes(form, kind, rng))
    return data.ljust(OPND_BYTES, b"\0")


def record_bytes(form: Form, rec: Record, rng: random.Random) -> bytes:
    params = struct.pack("<5I", rec.shift, rec.sign, rec.sat, rec.init, rec.rnd)
    out = (operand_bytes(form, rec.a, rec.shift, rng) + operand_bytes(form, rec.b, rec.shift, rng)
           + params)
    return out.ljust(REC_IN, b"\0")


# -- kernel text ------------------------------------------------------------------------------

def nops(n: int) -> list[str]:
    return ["  nop"] * n


# Scalar registers: r0..r11 flag reads, r12..r14 + r24..r26 parameters, r15 the record count,
# r27..r31 the caller's control and status values (silicon keeps them across kernels).
SAVED = ("crSat", "crRnd", "crSRSSign", "crUPSSign", "srSRS_of", "srUPS_of")
SAVE_REGS = ("r27", "r28", "r29", "r30", "r31", "r26")


def prologue(n: int) -> list[str]:
    lines = ["  // ---- prologue: keep the caller's control and status registers ----"]
    lines += [f"  mov {r}, {reg}" for reg, r in zip(SAVED, SAVE_REGS)]
    lines += [f"  movxm r15, #{n}",
              f"  movxm r12, #{REC_IN}", "  mov m6, r12",
              f"  movxm r12, #{REC_OUT}", "  mov m7, r12",
              # 2D / 3D modifiers all zero: the pointer stays put (it is recomputed per op).
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


def op_line(form: Form, which: str) -> list[str]:
    """The conversion of operand A or B (`which`), with the pointer set-up it needs before it."""
    idx = 0 if which == "A" else 2
    in_off = 0 if which == "A" else OPND_BYTES
    out_off = 0 if which == "A" else OPND_BYTES
    if form.kind == "reg":
        if form.family == "srs":
            return ["  " + form.op.format(dst=f"wl{idx}", src=_acc(form, idx))]
        return ["  " + form.op.format(dst=_acc(form, idx), src=_vec(form, idx + 4))]
    adr = form.addressing
    ptr = "p2"
    mem = {"idx_imm": f"[{ptr}, #0]", "idx": f"[{ptr}, dj0]", "pstm": f"[{ptr}], m0",
           "pstm_imm": f"[{ptr}], #0", "2d": f"[{ptr}], d0", "3d": f"[{ptr}], d0"}[adr]
    if form.kind == "store":
        return ["  " + form.op.format(src=_acc(form, idx), mem=mem)]
    return ["  " + form.op.format(dst=_acc(form, idx), mem=mem)]


def pointer_setup(form: Form, which: str) -> list[str]:
    """p2 = the operand (load forms) or result (store forms) address of `which`."""
    if form.kind == "reg":
        return []
    base = "p0" if form.kind == "load" else "p1"
    off = 0 if which == "A" else OPND_BYTES
    lines = [f"  mov p2, {base}"]
    if off:
        lines += [f"  movxm r12, #{off}", "  mov m5, r12", "  padda [p2], m5"]
    return lines + nops(2)


def operand_loads(form: Form) -> list[str]:
    """Register operands of both conversions, loaded from the record."""
    lines = []
    if form.kind == "load":
        return lines
    for which, idx, off in (("A", 0, 0), ("B", 2, OPND_BYTES)):
        if form.family == "srs":
            if form.acc_reg_bits == 512:
                quarters = [f"amll{idx}", f"amlh{idx}"]
            else:
                quarters = [f"amll{idx}", f"amlh{idx}", f"amhl{idx}", f"amhh{idx}"]
        else:
            r = idx + 4
            quarters = [f"wl{r}"] if form.vec_reg_bits == 256 else [f"wl{r}", f"wh{r}"]
        for k, q in enumerate(quarters):
            lines.append(f"  vlda {q}, [p0, #{off + 32 * k}]")
    return lines


def result_stores(form: Form) -> list[str]:
    lines = []
    if form.kind == "store":
        return lines
    for idx, off in ((0, 0), (2, OPND_BYTES)):
        if form.family == "srs":
            lines.append(f"  vst wl{idx}, [p1, #{off}]")
        else:
            quarters = (["amll", "amlh"] if form.acc_reg_bits == 512
                        else ["amll", "amlh", "amhl", "amhh"])
            lines += [f"  vst {q}{idx}, [p1, #{off + 32 * k}]" for k, q in enumerate(quarters)]
    return lines


def reads_after(form: Form) -> list[str]:
    """The flag read into r0..r11 at bundle offsets 1..READS after the conversion."""
    return [f"  mov r{k}, {form.flag}" for k in range(READS)]


def store_reads(base_word: int) -> list[str]:
    """st r0..r11 into flag words base_word.. of the record (p3 = flag area)."""
    return [f"  st r{k}, [p3, #{4 * (base_word + k)}]" for k in range(READS)]


def record_body(form: Form) -> list[str]:
    p = {name: 4 * i for i, name in enumerate(PARAMS)}
    # A scalar load's immediate is 8 bits signed: p4 addresses the parameter words.
    lines = ["  // ---- one record ----",
             "  mov p4, p0", f"  movxm r0, #{PARAM_OFF}", "  mov m5, r0", "  padda [p4], m5",
             "  mov p3, p1", f"  movxm r0, #{FLAG_OFF}", "  mov m5, r0", "  padda [p3], m5"]
    lines += nops(2)
    lines += [f"  lda r12, [p4, #{p['shift']}]",
              f"  lda r13, [p4, #{p['sign']}]",
              f"  lda r14, [p4, #{p['sat']}]",
              f"  lda r24, [p4, #{p['init']}]",
              f"  lda r25, [p4, #{p['rnd']}]"]
    lines += operand_loads(form)
    lines += nops(8)
    lines += ["  mov s0, r12", "  movx crSat, r14", f"  movx {form.sign_reg}, r13", "  movx crRnd, r25",
              f"  mov {form.flag}, r24"]
    lines += nops(6)
    lines += [f"  mov r0, {form.flag}"] + nops(6) + [f"  st r0, [p3, #{4 * F_INIT}]"]
    for which, base in (("A", F_A), ("B", F_B)):
        lines += pointer_setup(form, which)
        lines += nops(6)
        lines += op_line(form, which)
        lines += reads_after(form)
        lines += nops(2)
        lines += store_reads(base)
    lines += nops(8)
    lines += [f"  mov r0, {form.flag}"] + nops(6) + [f"  st r0, [p3, #{4 * F_FINAL}]"]
    lines += result_stores(form)
    lines += nops(2)
    lines += ["  padda [p0], m6", "  padda [p1], m7"]
    return lines


def kernel_text(form: Form, n: int) -> str:
    lines = [f"// Auto-generated ({GENERATOR}): {form.name} over {n} records",
             ".text", ".globl test_kernel", "test_kernel:"]
    lines += prologue(n)
    lines += [".p2align 4", ".Lrec:"]
    lines += record_body(form)
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
                "addressing": form.addressing, "acc_bits": form.acc_bits,
                "vec_bits": form.vec_bits, "lanes": form.lanes, "dform": form.dform,
                "records": [r.as_list() for r in chunk], "rec_in": REC_IN, "rec_out": REC_OUT}
        out.append((name, kernel_text(form, len(chunk)), data, meta))
    return out


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
    S.write_suite_json(suite_dir, "satflags", GENERATOR, names, meta={"seed": seed},
                       hw_checks=["aligned_labels"])
    return names


# -- analysis ---------------------------------------------------------------------------------

def flag_words(out: bytes, i: int) -> list[int]:
    base = i * REC_OUT + FLAG_OFF
    return list(struct.unpack_from(f"<{FLAG_WORDS}I", out, base))


def _round_up(value: int, shift: int, rnd: int) -> bool:
    """crRnd modes 0..15 (as `RoundingMode` in xdna-archspec: 4/6 and 5/7 alias, 12+ as 8+)."""
    if shift == 0:
        return False
    floor, dropped, half = value >> shift, value & ((1 << shift) - 1), 1 << (shift - 1)
    inexact, above, tie, neg, odd = dropped != 0, dropped > half, dropped == half, value < 0, floor & 1
    mode = rnd if rnd < 8 else 8 + (rnd & 3) if rnd < 12 else 12 + (rnd & 1)
    return {0: False, 1: inexact, 2: inexact and neg, 3: inexact and not neg,
            4: inexact and odd, 6: inexact and odd, 5: inexact and not odd, 7: inexact and not odd,
            8: above, 9: above or tie, 10: above or (tie and neg), 11: above or (tie and not neg),
            12: above or (tie and odd), 13: above or (tie and not odd)}[mode]


def unpack_lanes(bits: int, data: bytes, n: int, signed: bool) -> list[int]:
    out = []
    for i in range(n):
        v = int.from_bytes(data[i * bits // 8:(i + 1) * bits // 8], "little")
        if signed and v >> (bits - 1):
            v -= 1 << bits
        out.append(v)
    return out


def lanes_overflow(meta: dict, opnd: bytes, rec: list) -> list[bool]:
    """Per lane: is the converted value outside the output lane's range (before any clamp)?"""
    _, _, shift, sign, sat, _init, rnd = rec
    signed = (sign & 1) if meta["dform"] else 1
    n = meta["lanes"]
    if meta["family"] == "srs":
        # The range of the symmetric modes (crSat 2, 3) excludes the signed minimum.
        ob = meta["vec_bits"]
        lo, hi = (-(1 << (ob - 1)), (1 << (ob - 1)) - 1) if signed else (0, (1 << ob) - 1)
        lo += 1 if signed and sat & 2 else 0
        vals = unpack_lanes(meta["acc_bits"], opnd, n, True)
        res = [(v >> shift) + _round_up(v, shift, rnd) for v in vals]
    else:
        ab = meta["acc_bits"]
        lo, hi = -(1 << (ab - 1)), (1 << (ab - 1)) - 1
        res = [v << shift for v in unpack_lanes(meta["vec_bits"], opnd, n, bool(signed))]
    return [not lo <= r <= hi for r in res]


# Bundle offset from which a `mov rX, srY_of` sees the flag the conversion set (silicon).
VISIBLE_AT = {"srs": 3, "ups_reg": 3, "ups_load": 9}


def visible_at(meta: dict) -> int:
    if meta["family"] == "srs":
        return VISIBLE_AT["srs"]
    return VISIBLE_AT["ups_load" if meta["kind"] == "load" else "ups_reg"]


def predicted_words(meta: dict, rec_in: bytes, rec: list) -> list[int]:
    """Flag words under the model: sticky, set by any lane out of range (whatever crSat; the
    range of the symmetric SRS modes excludes the signed minimum),
    visible `visible_at` bundles after the conversion issues."""
    init = rec[5] & 1
    a = any(lanes_overflow(meta, rec_in[:OPND_BYTES], rec))
    b = any(lanes_overflow(meta, rec_in[OPND_BYTES:2 * OPND_BYTES], rec))
    d = visible_at(meta)
    after_a = init | a
    after_b = after_a | b
    words = [init]
    words += [after_a if k >= d else init for k in range(1, READS + 1)]
    words += [after_b if k >= d else after_a for k in range(1, READS + 1)]
    return words + [after_b]


def analyze(suite_dir: str, tag: str, verbose: bool = False) -> int:
    suite = S.load_suite(suite_dir)
    total = bad = 0
    for case in suite.cases:
        meta = suite.case_meta(case)
        out_path = suite.output(tag, case)
        if not out_path.is_file():
            print(f"{case}: no output")
            continue
        out = out_path.read_bytes()
        data = suite.topology(case).cores[0].args[0].data
        for i, r in enumerate(meta["records"]):
            got = flag_words(out, i)
            want = predicted_words(meta, data[i * REC_IN:(i + 1) * REC_IN], r)
            total += 1
            if got != want or verbose:
                bad += got != want
                fmt = lambda w: f"{w[0]} A={''.join(map(str, w[1:1 + READS]))} B={''.join(map(str, w[1 + READS:1 + 2 * READS]))} {w[-1]}"
                print(f"{case} {r}: hw {fmt(got)}  model {fmt(want)}")
    print(f"{total} records, {total - bad} match the model, {bad} differ")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite-dir")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="form names")
    ap.add_argument("--analyze", nargs=2, metavar=("DIR", "TAG"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        sys.exit(1 if analyze(*args.analyze, verbose=args.verbose) else 0)
    if not args.suite_dir:
        ap.error("--suite-dir or --analyze")
    names = generate(args.suite_dir, args.seed, args.only)
    print(f"{len(names)} cases -> {args.suite_dir}")


if __name__ == "__main__":
    main()
