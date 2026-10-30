#!/usr/bin/env python3
"""Report the MAC config word each vector multiply of a compiled capture kernel loads.

Input is the `.lst` listing xchesscc writes beside the object file
(`xchesscc_wrapper aie2 -c mac_mode.cc -o k.o` makes `k.o.lst`). For every
VMUL / VMAC / VMSC the last operand is the scalar register holding the config word; the
word is the immediate that register was loaded with (`MOVA|MOVX|MOV r24, #808`). The
fields decode as in aiev2_compute_control(): bit 0 zero_acc, bits 2:1 amode, 4:3 bmode,
7:5 variant, bit 8 sgn_y, bit 9 sgn_x, bits 13:11 sub controls.

  python3 tools/mac_mode_config_words.py <kernel.o.lst> [--expect amode,bmode,variant]
"""

import argparse
import re
import sys

MOV = re.compile(r"\b(?:MOVA|MOVX|MOV)\s+(r\d+),\s*#(-?\d+)")
MUL = re.compile(r"\b(VMUL|VMAC|VMSC|VNEGMUL|VNEGMAC|VNEGMSC|VADDMAC|VSUBMAC)\s+([^;\n]*)")


def decode(word):
    return {
        "word": word, "zero_acc": word & 1, "amode": (word >> 1) & 3, "bmode": (word >> 3) & 3,
        "variant": (word >> 5) & 7, "sgn_y": (word >> 8) & 1, "sgn_x": (word >> 9) & 1,
        "sub0": (word >> 11) & 1, "sub1": (word >> 12) & 1, "sub2": (word >> 13) & 1,
        "high_bits": word >> 14,
    }


def config_words(listing):
    """(mnemonic, register, [word]) for each multiply in the listing, in program order.

    A register can be reused (r24 also carries the crRnd setup value), so each use takes
    the most recent immediate loaded into it; a use with no earlier load has no word.
    """
    current = {}
    uses = []
    for line in listing.splitlines():
        for op, operands in MUL.findall(line):
            reg = operands.split(",")[-1].strip()
            uses.append((op, reg, [current[reg]] if reg in current else []))
        for reg, imm in MOV.findall(line):
            current[reg] = int(imm)
    return uses


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("listing")
    ap.add_argument("--expect", help="amode,bmode,variant every word must decode to")
    args = ap.parse_args()
    with open(args.listing) as f:
        uses = config_words(f.read())
    if not uses:
        sys.exit("no vector multiplies found")
    words = set()
    for op, reg, cands in uses:
        if len(cands) != 1:
            sys.exit(f"{op} reads {reg}, which holds no immediate config word at that point")
        words.add(cands[0])
    seen = {}
    for w in sorted(words):
        d = decode(w)
        seen[w] = d
        print("0x%04x (%d): amode=%d bmode=%d variant=%d sgn_x=%d sgn_y=%d zero_acc=%d sub=%d%d%d%s"
              % (w, w, d["amode"], d["bmode"], d["variant"], d["sgn_x"], d["sgn_y"], d["zero_acc"],
                 d["sub0"], d["sub1"], d["sub2"], " HIGH=%d" % d["high_bits"] if d["high_bits"] else ""))
    ops = sorted({op for op, _, _ in uses})
    print("instructions:", " ".join(ops), f"({len(uses)} issued)")
    if args.expect:
        want = tuple(int(x) for x in args.expect.split(","))
        for w, d in seen.items():
            if (d["amode"], d["bmode"], d["variant"]) != want:
                sys.exit(f"word {w} decodes to {(d['amode'], d['bmode'], d['variant'])}, expected {want}")
        signs = {(d["sgn_x"], d["sgn_y"]) for d in seen.values()}
        if signs != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            sys.exit(f"sign flavours seen: {sorted(signs)}")
        print("OK: all four sign flavours, expected mode")


if __name__ == "__main__":
    main()
