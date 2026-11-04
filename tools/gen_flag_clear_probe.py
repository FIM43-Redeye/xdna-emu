#!/usr/bin/env python3
"""Generate the flag-clear race probe: a `mov srX_of, r` that clears the overflow flag N
bundles after a saturating conversion, before or after the flag becomes visible.

The saturation-flag probe (`gen_sat_flags_probe.py`) pins when a reader sees the flag (3
bundles after an SRS, at the accumulator result cycle of a UPS). This asks what a write to the
register in that window does: if the conversion's flag lands after the clear, the flag reads 1
afterwards; if the clear lands last, 0. Each record runs one saturating conversion (every lane
out of range) with the flag clear, writes 0 to the flag N = 0..11 bundles later (0: in the
conversion's own bundle where the slots allow), and reads the flag 16 bundles after the
conversion.

  python3 tools/gen_flag_clear_probe.py --suite-dir <dir>
  python3 tools/gen_flag_clear_probe.py --analyze <dir> <tag>
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import suite as S  # noqa: E402

GENERATOR = "tools/gen_flag_clear_probe.py"
OFFSETS = tuple(range(0, 12))
READ_AT = 16
SCRATCH = 64                    # 32-byte aligned (a vector store ignores the low address bits)


@dataclass(frozen=True)
class Form:
    name: str
    flag: str
    op: str             # the conversion (sources loaded beforehand)
    pairs: bool         # can share a bundle with `mov srX_of, r0`


FORMS = (
    Form("vsrs", "srSRS_of", "vsrs.s16.s32 wl0, bml0, s0", True),
    Form("vst_srs", "srSRS_of", "vst.srs.s16.s32 bml0, s0, [p3, #0]", True),
    Form("vups", "srUPS_of", "vups.s32.s16 bml0, wl4, s0", False),
    Form("vlda_ups", "srUPS_of", "vlda.ups.s32.s16 bml0, s0, [p0, #0]", True),
)


def input_bytes() -> bytes:
    """64 bytes: as int32 lanes 0x7fffffff (out of int16 range at shift 0); as int16 lanes
    0x7fff (out of int32 range at shift 17)."""
    return struct.pack("<16I", *([0x7FFF_FFFF] * 16))


def record_lines(form: Form, n: int) -> list[str]:
    shift = 0 if form.flag == "srSRS_of" else 17
    lines = [f"  // ---- clear {n} bundles after {form.name} ----",
             f"  movxm r1, #{shift}", "  mov s0, r1", "  mov r0, #0", f"  mov {form.flag}, r0",
             "  vlda amll0, [p0, #0]", "  vlda amlh0, [p0, #32]", "  vlda wl4, [p0, #0]"]
    lines += ["  nop"] * 9
    clear = f"mov {form.flag}, r0"
    if n == 0 and form.pairs:
        lines += [f"  {form.op} ; {clear}"] + ["  nop"] * (READ_AT - 1)
    elif n == 0:
        return []
    else:
        lines += [f"  {form.op}"] + ["  nop"] * (n - 1) + [f"  {clear}"] + ["  nop"] * (READ_AT - n - 1)
    lines += [f"  mov r2, {form.flag}"] + ["  nop"] * 2 + ["  st r2, [p1, #0]", "  nop",
                                                          "  padda [p1], #4"]
    return lines


def kernel_text(form: Form) -> tuple[str, list[int]]:
    lines = [f"// Auto-generated ({GENERATOR}): {form.name}", ".text", ".globl test_kernel",
             "test_kernel:", "  mov r27, crSat", f"  mov r28, {form.flag}", "  mov r29, s0",
             "  mov r1, #1", "  movx crSat, r1",
             # vst.srs results go to a scratch area past the flag words.
             "  mov p3, p1", f"  padda [p3], #{SCRATCH}"]
    offs = []
    for n in OFFSETS:
        rec = record_lines(form, n)
        if rec:
            lines += rec
            offs.append(n)
    lines += ["  nop"] * 4 + ["  movx crSat, r27", f"  mov {form.flag}, r28", "  mov s0, r29"]
    lines += ["  nop"] * 4 + ["  ret lr"] + ["  nop"] * 5
    return "\n".join(lines) + "\n", offs


def generate(suite_dir: str | Path) -> list[str]:
    names = []
    for form in FORMS:
        text, offs = kernel_text(form)
        S.write_single_core_case(suite_dir, form.name, text, input_bytes=input_bytes(),
                                 out_bytes=SCRATCH + 32,
                                 meta={"offsets": offs, "flag": form.flag, "op": form.op})
        names.append(form.name)
    S.write_suite_json(suite_dir, "flag-clear", GENERATOR, names)
    return names


def analyze(suite_dir: str, tag: str) -> None:
    suite = S.load_suite(suite_dir)
    for case in suite.cases:
        meta = suite.case_meta(case)
        out = suite.output(tag, case).read_bytes()
        offs = meta["offsets"]
        flags = struct.unpack_from(f"<{len(offs)}I", out, 0)
        print(f"{case:10s} clear at " + " ".join(f"{n}:{f}" for n, f in zip(offs, flags)))


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
