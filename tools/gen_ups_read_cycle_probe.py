#!/usr/bin/env python3
"""Generate the vlda.ups read-cycle probe: in which bundle after a fused `vlda.ups` does a
write of crUPSSign, crSat or the shift register still change the conversion?

The itinerary (`AIE2Schedule.td`, II_VLDA_UPS and its post-increment, 2D and 3D twins) reads
the shift and crUPSSign at operand cycle 7 and crSat at 8; compilers write them in the load's
shadow. Each record issues the load with one register at its old value and writes the new
value N bundles later (N = 0: in the load's own bundle), then stores the accumulator. The last
N whose write still shows in the result pins the read bundle.

  sign: crUPSSign 0 -> 1, lanes with the top bit set (zero- vs sign-extended), no shift
  sat:  crSat 0 -> 1, signed, a shift that overflows the lane (wrapped vs clamped)
  shift: s0 0 -> 4 (unshifted vs shifted)

Unrolled assembly, one case per form.

  python3 tools/gen_ups_read_cycle_probe.py --suite-dir <dir>
  python3 tools/gen_ups_read_cycle_probe.py --analyze <dir> <tag>
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import suite as S  # noqa: E402

GENERATOR = "tools/gen_ups_read_cycle_probe.py"
OFFSETS = tuple(range(0, 11))
PARAMS = ("sign", "sat", "shift")
REC_OUT = 128
SETTLE = 14                     # bundles after the load before the accumulator is stored


@dataclass(frozen=True)
class Form:
    name: str
    mnemonic: str               # e.g. vlda.ups.s64.d32
    mem: str                    # operand text with p2 (and m0 / d0, all zero)
    acc_bits: int
    src_bits: int

    @property
    def lanes(self) -> int:
        return 256 // self.src_bits

    @property
    def cm(self) -> bool:
        return self.acc_bits * self.lanes == 1024


FORMS = tuple(
    [Form(f"vlda_ups_s{a}_d{s}", f"vlda.ups.s{a}.d{s}", "[p2, #0]", a, s)
     for a, s in ((32, 16), (64, 32), (32, 8), (64, 16))]
    + [Form("vlda_ups_s64_d32_pstm", "vlda.ups.s64.d32", "[p2], m0", 64, 32),
       Form("vlda_2d_ups_s64_d32", "vlda.2d.ups.s64.d32", "[p2], d0", 64, 32),
       Form("vlda_3d_ups_s32_d16", "vlda.3d.ups.s32.d16", "[p2], d0", 32, 16)])


def source_bytes(form: Form) -> bytes:
    """Lanes alternate the signed maximum and a top-bit-set value: the sign probe sees the
    latter extend, the saturation probe sees the former overflow."""
    b = form.src_bits
    vals = [(1 << (b - 1)) - 1 if i % 2 else (1 << (b - 1)) + 1 for i in range(form.lanes)]
    return b"".join(v.to_bytes(b // 8, "little") for v in vals)


def record(form: Form, param: str, n: int) -> tuple[dict, dict]:
    """(initial register values, the write issued n bundles after the load)."""
    over = form.acc_bits - form.src_bits + 1
    if param == "sign":
        return {"crUPSSign": 0, "crSat": 0, "s0": 0}, {"crUPSSign": 1}
    if param == "sat":
        return {"crUPSSign": 1, "crSat": 0, "s0": over}, {"crSat": 1}
    return {"crUPSSign": 1, "crSat": 0, "s0": 0}, {"s0": 4}


def _write(reg: str, value: int) -> list[str]:
    """r6 is loaded beforehand; the write itself is one bundle."""
    return [f"  movx {reg}, r6" if reg.startswith("cr") else f"  mov {reg}, r6"]


def record_lines(form: Form, param: str, n: int) -> list[str]:
    init, late = record(form, param, n)
    lines = [f"  // ---- {param} written {n} bundles after the load ----"]
    for reg, v in init.items():
        lines += [f"  movxm r7, #{v}"] + (["  movx " + reg + ", r7"] if reg.startswith("cr")
                                         else [f"  mov {reg}, r7"])
    (reg, v), = late.items()
    lines += [f"  movxm r6, #{v}", "  mov p2, p0"] + ["  nop"] * 8
    load = f"  {form.mnemonic} {'cm0' if form.cm else 'bml0'}, s0, {form.mem}"
    write = _write(reg, v)[0].strip()
    if n == 0:
        lines.append(f"{load} ; {write}")
        lines += ["  nop"] * SETTLE
    else:
        lines += [load] + ["  nop"] * (n - 1) + ["  " + write] + ["  nop"] * (SETTLE - n)
    quarters = ["amll0", "amlh0", "amhl0", "amhh0"] if form.cm else ["amll0", "amlh0"]
    lines += [f"  vst {q}, [p1, #{32 * k}]" for k, q in enumerate(quarters)]
    lines += ["  nop", f"  padda [p1], #{REC_OUT}"]
    return lines


SAVED = ("crSat", "crUPSSign")


def kernel_text(form: Form) -> str:
    lines = [f"// Auto-generated ({GENERATOR}): {form.name}", ".text", ".globl test_kernel",
             "test_kernel:", "  mov r27, crSat", "  mov r28, crUPSSign", "  mov r29, s0",
             "  mov r12, #0", "  mov m0, r12", "  mov dn0, r12", "  mov dj0, r12", "  mov dc0, r12",
             "  mov m1, r12", "  mov dn1, r12", "  mov dj1, r12", "  mov dc1, r12"]
    for param in PARAMS:
        for n in OFFSETS:
            lines += record_lines(form, param, n)
    lines += ["  nop"] * 4 + ["  movx crSat, r27", "  movx crUPSSign, r28", "  mov s0, r29"]
    lines += ["  nop"] * 4 + ["  ret lr"] + ["  nop"] * 5
    return "\n".join(lines) + "\n"


def records_of() -> list[list]:
    return [[p, n] for p in PARAMS for n in OFFSETS]


def generate(suite_dir: str | Path) -> list[str]:
    names = []
    for form in FORMS:
        recs = records_of()
        S.write_single_core_case(suite_dir, form.name, kernel_text(form),
                                 input_bytes=source_bytes(form), out_bytes=REC_OUT * len(recs),
                                 meta={"records": recs, "acc_bits": form.acc_bits,
                                       "src_bits": form.src_bits, "mnemonic": form.mnemonic})
        names.append(form.name)
    S.write_suite_json(suite_dir, "ups-read-cycle", GENERATOR, names)
    return names


def analyze(suite_dir: str, tag: str) -> None:
    """Per form and register: the bundle offsets whose late write still changed the result."""
    suite = S.load_suite(suite_dir)
    for case in suite.cases:
        meta = suite.case_meta(case)
        out = suite.output(tag, case).read_bytes()
        recs = meta["records"]
        for param in PARAMS:
            rows = [(n, out[i * REC_OUT:(i + 1) * REC_OUT]) for i, (p, n) in enumerate(recs)
                    if p == param]
            new = rows[0][1]                    # n = 0: the write lands before any read
            old = rows[-1][1]                   # n = 10: after every read
            seen = "".join("N" if r == new else "o" if r == old else "?" for _, r in rows)
            print(f"{case:24s} {param:6s} offsets 0..10: {seen}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite-dir")
    ap.add_argument("--analyze", nargs=2, metavar=("DIR", "TAG"))
    args = ap.parse_args()
    if args.analyze:
        analyze(*args.analyze)
        return
    if not args.suite_dir:
        ap.error("--suite-dir or --analyze")
    print(f"{len(generate(args.suite_dir))} cases -> {args.suite_dir}")


if __name__ == "__main__":
    main()
