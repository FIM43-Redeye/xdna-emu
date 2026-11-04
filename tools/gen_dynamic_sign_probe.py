#!/usr/bin/env python3
"""Generate the dynamic-sign probe suite: vector instructions whose signedness is a run-time value.

AIE2 has `d` forms of several vector instructions (`vups.s32.d16`, `vsrs.d16.s32`,
`vabs_gtz.d8`, `vlt.d16`, `vextract.d32`, ...). The encoding leaves the sign open; the
instruction reads it from a control register that its TableGen record lists in `Uses`:

  crUPSSign   vups.*.dN, vlda[.2d|.3d].ups.*.dN        (source lanes signed or unsigned)
  crSRSSign   vsrs.dN.*, vsrsm.dN.*, vst[.2d|.3d].srs.dN.*   (output lanes, saturation range)
  crVaddSign  vabs_gtz, vlt, vge, vmax_lt, vmin_ge, vmaxdiff_lt, vsub_lt, vsub_ge, vextract .dN

Each kernel loops over RECORDS, each carrying its own operands, shift, sign word and crSat,
so one compile sweeps one form over every sign value, shift and saturation mode. The sign
word sweeps values whose bit 0 differs and values whose higher bits differ (only bit 0
should count). Operand lanes mix the edge values of the lane width (min, max, -1, 0, 1, ...)
with random ones.

Peano emits the register forms (and, inside a loop, the fused `vlda.ups` / `vst.srs` with the
sign written in the load's shadow); Chess emits the fused `vlda.ups` from its `ups_to_*`
intrinsics. Generators only write suites; build and run with
`python3 -m harness suite build|hw|emu|compare <dir> ...`.

  python3 tools/gen_dynamic_sign_probe.py --suite-dir <dir> [--seed N]
"""
from __future__ import annotations

import argparse
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cxx_suite import CxxKernel, write_cxx_suite  # noqa: E402

# Sign words: bit 0 decides on silicon for the other sign registers; the high-bit values
# check that nothing else does here.
SIGNS = (0, 1, 2, 3, 0xFFFF_FFFE, 0xFFFF_FFFF)
# crSat (2 bits): every value, 2 included (the symmetric bit without the saturate bit).
SATS = (0, 1, 2, 3)
# Budget per case: input plus output bytes stay well inside one tile's data memory.
CASE_BYTES = 24 * 1024


def lane_edges(bits: int) -> list[int]:
    """Edge values of a `bits`-wide lane, as unsigned bit patterns."""
    m = (1 << bits) - 1
    smin, smax = 1 << (bits - 1), (1 << (bits - 1)) - 1
    return sorted({smin, smax, m, 0, 1, smin + 1, smax - 1, m - 1, 2, 0x5555_5555_5555_5555 & m,
                   0xAAAA_AAAA_AAAA_AAAA & m})


def lanes_bytes(bits: int, n: int, rng: random.Random, rotate: int) -> bytes:
    """n lanes of `bits`: edge values (rotated per record) interleaved with random lanes."""
    edges = lane_edges(bits)
    vals = []
    for i in range(n):
        if i % 2 == 0:
            vals.append(edges[(i // 2 + rotate) % len(edges)])
        else:
            vals.append(rng.getrandbits(bits))
    return pack_lanes(bits, vals)


def pack_lanes(bits: int, vals: list[int]) -> bytes:
    if bits == 4:
        return bytes((vals[i] & 0xF) | (vals[i + 1] & 0xF) << 4 for i in range(0, len(vals), 2))
    return b"".join((v & ((1 << bits) - 1)).to_bytes(bits // 8, "little") for v in vals)


def acc_lanes_bytes(acc_bits: int, out_bits: int, n: int, shift: int, rng: random.Random,
                    rotate: int) -> bytes:
    """Accumulator lanes around the saturation edges of an `out_bits` output after `shift`."""
    m = (1 << acc_bits) - 1
    umax, smax, smin = (1 << out_bits) - 1, (1 << (out_bits - 1)) - 1, -(1 << (out_bits - 1))
    picks = [umax, umax + 1, smax, smax + 1, smin, smin - 1, -1, 0, 1, -umax, 3, -3]
    vals = []
    for i in range(n):
        if i % 2 == 0:
            v = (picks[(i // 2 + rotate) % len(picks)] << shift) & m
        else:
            v = rng.getrandbits(acc_bits)
        vals.append(v)
    return pack_lanes(acc_bits, vals)


@dataclass(frozen=True)
class Form:
    """One probed form: a kernel body run once per record, and its record layout."""
    name: str               # case-name stem
    compiler: str           # "peano" or "chess"
    rec_in: int             # input bytes per record (multiple of 64)
    rec_out: int            # output bytes per record (multiple of 64)
    body: str               # C++ statements; `p` = record input, `o` = record output (int32_t *)
    records: tuple          # tuple of record-builder arguments
    build: Callable[..., bytes]  # (rng, *record) -> bytes of length rec_in


def kernel_source(form: Form, n: int) -> str:
    return f"""// Auto-generated (gen_dynamic_sign_probe.py): {form.name} over {n} records
#define NOCPP
#include <stdint.h>

extern "C" {{
void test_kernel(const int32_t *restrict in, int32_t *restrict out) {{
    for (int t = 0; t < {n}; t++) {{
        const int32_t *p = in + t * {form.rec_in // 4};
        int32_t *o = out + t * {form.rec_out // 4};
{form.body}
    }}
}}
}} // extern "C"
"""


def scalars(*words: int) -> bytes:
    return b"".join(struct.pack("<I", w & 0xFFFF_FFFF) for w in words)


# -- UPS (crUPSSign) ------------------------------------------------------------------------

# builtin, source type, acc type, source lane bits, source lanes, acc lane bits. Peano, in a
# loop over records, fuses the vector load into `vlda.ups` (the sign written in the load's
# shadow); the `_reg` variants route the source through a vshift by a run-time zero so the
# conversion reads a register (`vups`, including the x2c forms).
UPS_PEANO = {
    "ups_s32_d16_bm": ("acc32_v16_I256_ups", "v16int16", "v16acc32", 16, 16, 32),
    "ups_s32_d8_cm": ("acc32_v32_I256_ups", "v32int8", "v32acc32", 8, 32, 32),
    "ups_s32_d16_x": ("acc32_v32_I512_ups", "v32int16", "v32acc32", 16, 32, 32),
    "ups_s64_d16_cm": ("acc64_v16_I256_ups", "v16int16", "v16acc64", 16, 16, 64),
    "ups_s64_d32_x": ("acc64_v16_I512_ups", "v16int32", "v16acc64", 32, 16, 64),
    "ups_s64_d32_bm": ("acc64_v8_I256_ups", "v8int32", "v8acc64", 32, 8, 64),
}
# Chess `ups_to_*` inside the record loop: register `vups` (w2b / w2c).
UPS_CHESS = {
    "chess_ups_s32_d16": ("ups_to_v16acc32", "v16int16", "v16acc32", 16, 16, 32),
    "chess_ups_s32_d8": ("ups_to_v32acc32", "v32int8", "v32acc32", 8, 32, 32),
    "chess_ups_s64_d16": ("ups_to_v16acc64", "v16int16", "v16acc64", 16, 16, 64),
    "chess_ups_s64_d32": ("ups_to_v8acc64", "v8int32", "v8acc64", 32, 8, 64),
}


def ups_shifts(acc_bits: int) -> tuple[int, ...]:
    # The shift register is 6 bits: past the lane width every nonzero lane overflows.
    return (0, 1, 7, 8, 16, 24, 31, 40) if acc_bits == 32 else (0, 1, 16, 31, 32, 48, 56, 63)


def ups_form(name: str, spec, compiler: str, register: bool = False) -> Form:
    fn, vty, aty, bits, lanes, acc_bits = spec
    vbytes = bits * lanes // 8
    soff = vbytes // 4
    rec_in = -(-(vbytes + 16) // 64) * 64
    rec_out = acc_bits * lanes // 8
    call = f"__builtin_aiev2_{fn}" if compiler == "peano" else fn
    set_sat = (f"__builtin_aiev2_set_ctrl_reg(9, p[{soff + 2}]);" if compiler == "peano"
               else f"set_satmode(p[{soff + 2}]);")
    load = f"        {vty} v = *(const {vty} *)p;\n"
    if register:
        zero = f"p[{soff + 3}]"
        if vbytes == 64:
            load = (f"        v16int32 w = *(const v16int32 *)p;\n"
                    f"        {vty} v = ({vty})__builtin_aiev2_vshift_I512_I512(w, w, 0, {zero});\n")
        else:
            load = (f"        v8int32 n = *(const v8int32 *)p;\n"
                    f"        v16int32 w = __builtin_aiev2_concat_I512_I256(n, n);\n"
                    f"        {vty} v = ({vty})__builtin_aiev2_ext_I256_I512("
                    f"__builtin_aiev2_vshift_I512_I512(w, w, 0, {zero}), 0);\n")
    body = (f"        {set_sat}\n" + load +
            f"        *({aty} *)o = {call}(v, p[{soff}], p[{soff + 1}]);")
    records = tuple((s, sh, sat) for sat in SATS for sh in ups_shifts(acc_bits) for s in SIGNS)

    def build(rng, sign, shift, sat, k):
        return (lanes_bytes(bits, lanes, rng, k) + scalars(shift, sign, sat, 0)).ljust(rec_in, b"\0")

    return Form(name + ("_reg" if register else ""), compiler, rec_in, rec_out, body, records, build)


# -- SRS (crSRSSign) ------------------------------------------------------------------------

# builtin, acc type, acc lane bits, lanes, result type, out lane bits, register-form result is 512-bit
SRS_PEANO = {
    "srs_d16_s32_bm": ("I256_v16_acc32_srs", "v16acc32", 32, 16, "v16int16", 16, False),
    "srs_d16_s64_cm": ("I256_v16_acc64_srs", "v16acc64", 64, 16, "v16int16", 16, False),
    "srs_d8_s32_cm": ("I256_v32_acc32_srs", "v32acc32", 32, 32, "v32int8", 8, False),
    "srs_d32_s64_bm": ("I256_v8_acc64_srs", "v8acc64", 64, 8, "v8int32", 32, False),
    "srs_d32_s64_x": ("I512_v16_acc64_srs", "v16acc64", 64, 16, "v16int32", 32, True),
    "srs_d16_s32_x": ("I512_v32_acc32_srs", "v32acc32", 32, 32, "v32int16", 16, True),
}
SRS_SHIFTS = (0, 1, 4, 8, 15)


def srs_form(name: str, spec, register: bool) -> Form:
    fn, aty, acc_bits, lanes, rty, out_bits, wide = spec
    abytes = acc_bits * lanes // 8
    soff = abytes // 4
    rec_in = -(-(abytes + 16) // 64) * 64
    rec_out = 64
    call = f"__builtin_aiev2_{fn}(a, p[{soff}], p[{soff + 1}])"
    head = (f"        __builtin_aiev2_set_ctrl_reg(9, p[{soff + 2}]);\n"
            f"        {aty} a = *(const {aty} *)p;\n")
    if not register:
        # A store of the result: Peano fuses it into vst.srs.
        body = head + f"        *({rty} *)o = {call};"
    else:
        # Route the result through a vshift by a run-time zero (p[soff+3] == 0) so the SRS
        # writes a vector register (vsrs) instead of fusing into the store.
        widen = (f"(v16int32){call}" if wide
                 else f"(v16int32)__builtin_aiev2_concat_I512_I256({call}, {call})")
        if not wide:
            body = (head + f"        {rty} r = {call};\n"
                    f"        v16int32 w = (v16int32)__builtin_aiev2_concat_I512_I256(r, r);\n")
        else:
            body = head + f"        v16int32 w = {widen};\n"
        body += f"        *(v16int32 *)o = __builtin_aiev2_vshift_I512_I512(w, w, 0, p[{soff + 3}]);"
    records = tuple((s, sh, sat) for sat in SATS for sh in SRS_SHIFTS for s in SIGNS)

    def build(rng, sign, shift, sat, k):
        acc = acc_lanes_bytes(acc_bits, out_bits, lanes, shift, rng, k)
        return (acc + scalars(shift, sign, sat, 0)).ljust(rec_in, b"\0")

    return Form(name + ("_reg" if register else "_st"), "peano", rec_in, rec_out, body, records, build)


# -- crVaddSign family ----------------------------------------------------------------------

LANE = {8: ("v64int8", "unsigned long long"), 16: ("v32int16", "unsigned int"), 32: ("v16int32", "unsigned int")}
# op -> number of vector operands, returns a vector (else only a mask)
VADD_OPS = {
    "vabs_gtz": (1, True),
    "vlt": (2, False),
    "vge": (2, False),
    "vmax_lt": (2, True),
    "vmin_ge": (2, True),
    "vmaxdiff_lt": (2, True),
    "vsub_lt": (2, True),
    "vsub_ge": (2, True),
}
VADD_PATTERNS = 8


def vadd_form(op: str, bits: int) -> Form:
    nops, returns_vec = VADD_OPS[op]
    vty, mty = LANE[bits]
    rec_in = 64 * nops + 64
    rec_out = 128
    s = 16 * nops
    fn = f"__builtin_aiev2_{op}{bits}"
    load = "".join(f"        {vty} a{i} = *(const {vty} *)(p + {16 * i});\n" for i in range(nops))
    args = ", ".join(f"a{i}" for i in range(nops))
    if returns_vec:
        body = (load + f"        {mty} m;\n"
                f"        *({vty} *)o = {fn}({args}, p[{s}], m);\n"
                f"        *({mty} *)(o + 16) = m;")
    elif bits == 8:
        body = load + f"        *(v2int32 *)(o + 16) = {fn}({args}, p[{s}]);"
    else:
        body = load + f"        *({mty} *)(o + 16) = {fn}({args}, p[{s}]);"
    records = tuple((sg, k) for k in range(VADD_PATTERNS) for sg in SIGNS)
    lanes = 512 // bits

    def build(rng, sign, pattern, k):
        ops = b"".join(lanes_bytes(bits, lanes, rng, pattern * 3 + i * 5) for i in range(nops))
        return (ops + scalars(sign)).ljust(rec_in, b"\0")

    return Form(f"{op}_d{bits}", "peano", rec_in, rec_out, body, records, build)


EXTRACT = {8: ("vextract_elem8_I512", "v64int8", 64), 16: ("vextract_elem16_I512", "v32int16", 32),
           32: ("vextract_elem32_I512", "v16int32", 16), 64: ("vextract_elem64_I512", "v16int32", 8)}


def extract_form(bits: int) -> Form:
    fn, vty, lanes = EXTRACT[bits]
    call = f"__builtin_aiev2_{fn}(v, p[16], p[17])"
    store = f"*(v2int32 *)o = {call};" if bits == 64 else f"o[0] = {call};"
    body = f"        {vty} v = *(const {vty} *)p;\n        {store}"
    idxs = sorted({0, 1, lanes // 2 - 1, lanes // 2, lanes - 2, lanes - 1})
    records = tuple((s, i) for i in idxs for s in SIGNS)
    lane_bits = 32 if bits == 64 else bits

    def build(rng, sign, idx, k):
        return (lanes_bytes(lane_bits, 512 // lane_bits, rng, k) + scalars(idx, sign)).ljust(128, b"\0")

    return Form(f"vextract_d{bits}", "peano", 128, 64, body, records, build)


# -- plan -----------------------------------------------------------------------------------

def all_forms() -> list[Form]:
    forms = [ups_form(n, s, "peano", reg) for n, s in UPS_PEANO.items() for reg in (False, True)]
    forms += [ups_form(n, s, "chess") for n, s in UPS_CHESS.items()]
    forms += [srs_form(n, s, reg) for n, s in SRS_PEANO.items() for reg in (False, True)]
    forms += [vadd_form(op, b) for op in VADD_OPS for b in (8, 16, 32)]
    forms += [extract_form(b) for b in (8, 16, 32, 64)]
    return forms


def kernels_for(form: Form, rng: random.Random) -> list[CxxKernel]:
    per_case = max(1, CASE_BYTES // (form.rec_in + form.rec_out))
    out = []
    for k, start in enumerate(range(0, len(form.records), per_case)):
        chunk = form.records[start:start + per_case]
        data = b"".join(form.build(rng, *rec, start + i) for i, rec in enumerate(chunk))
        name = form.name if len(form.records) <= per_case else f"{form.name}__{k:02d}"
        out.append(CxxKernel(name=name, source=kernel_source(form, len(chunk)), in_size=len(data),
                             out_size=form.rec_out * len(chunk), input_bytes=data,
                             meta={"form": form.name, "records": [list(r) for r in chunk],
                                   "rec_in": form.rec_in, "rec_out": form.rec_out}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite-dir", required=True, help="root; writes <dir>/peano and <dir>/chess")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    by_compiler: dict[str, list[CxxKernel]] = {"peano": [], "chess": []}
    for form in all_forms():
        by_compiler[form.compiler] += kernels_for(form, rng)
    for compiler, kernels in by_compiler.items():
        write_cxx_suite(Path(args.suite_dir) / compiler, f"dynsign-{compiler}",
                        "tools/gen_dynamic_sign_probe.py", compiler, kernels, meta={"seed": args.seed})
        print(f"{len(kernels)} {compiler} cases -> {Path(args.suite_dir) / compiler}")


if __name__ == "__main__":
    main()
