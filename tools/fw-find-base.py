#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# fw-find-base.py -- recover the load base address of the Xtensa NPU
# firmware by correlating its literal-pool words against known anchors.
#
# Xtensa code loads constants and addresses through PC-relative l32r
# literal pools.  A literal that holds a pointer encodes (base + target),
# where target is a function entry or a string -- both at known *body
# offsets*.  So for every aligned 32-bit word W and every anchor offset A,
# (W - A) is a vote for the load base.  Function-pointer and string-pointer
# literals all vote for the same base, producing a sharp histogram spike;
# noise spreads thin.
#
# Anchors come from the Ghidra Xtensa analysis (functions.tsv) plus the
# firmware's plaintext strings -- a far cleaner anchor set than a raw
# printable-run scan.
#
# Output: ranked base candidates, and (for the winner) the literal sites
# that resolve to functions/strings -- these are the pointer tables.

import csv
import struct
import sys
from collections import Counter

BODY = "/home/triple/npu-work/ghidra-projects/npu-fw/npu-fw-body.bin"
FUNCS = "/home/triple/npu-work/ghidra-projects/npu-fw/analysis-xtensa/functions.tsv"
STRS = "/home/triple/npu-work/ghidra-projects/npu-fw/analysis-xtensa/strings.tsv"

PTR_LO, PTR_HI = 0x1000, 0x20000000   # plausible absolute-address range


def load_offsets(path, col=0):
    out = set()
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)  # header
        for row in r:
            if not row:
                continue
            try:
                out.add(int(row[col], 16))
            except (ValueError, IndexError):
                pass
    return out


def main():
    with open(BODY, "rb") as f:
        data = f.read()
    n = len(data)

    func_off = load_offsets(FUNCS)
    str_off = load_offsets(STRS)
    anchors = func_off | str_off
    print(f"body {n} bytes; anchors: {len(func_off)} functions + "
          f"{len(str_off)} strings = {len(anchors)} total")

    # Aligned 32-bit words -- literal pools are 4-byte aligned.
    words = []
    for off in range(0, n - 3, 4):
        w = struct.unpack_from("<I", data, off)[0]
        if PTR_LO <= w <= PTR_HI:
            words.append((off, w))
    print(f"aligned pointer-range words: {len(words)}")

    # Vote base = W - A for anchors A within reach of W.
    votes = Counter()
    anchor_sorted = sorted(anchors)
    amin, amax = anchor_sorted[0], anchor_sorted[-1]
    for _, w in words:
        lo = w - amax
        hi = w - amin
        if hi < 0:
            continue
        for a in anchors:
            b = w - a
            if 0 <= b <= PTR_HI:
                votes[b] += 1

    print()
    print("=== top 15 base candidates ===")
    for base, cnt in votes.most_common(15):
        # how many distinct anchor *types* hit -> a real base hits both
        # functions and strings
        fn = sum(1 for o, w in words if (w - base) in func_off)
        st = sum(1 for o, w in words if (w - base) in str_off)
        print(f"  base 0x{base:08x}  {cnt:4d} votes   "
              f"({fn} func-ptr, {st} str-ptr)")

    best = votes.most_common(1)[0][0]
    print()
    print(f"=== literal sites resolving under base 0x{best:08x} ===")
    sites = []
    for off, w in words:
        tgt = w - best
        if tgt in func_off:
            sites.append((off, w, tgt, "FUNC"))
        elif tgt in str_off:
            txt = data[tgt:tgt + 48].split(b"\x00")[0].decode("ascii", "replace")
            sites.append((off, w, tgt, f"STR {txt!r}"))
    print(f"  {len(sites)} resolved literal sites")
    for off, w, tgt, kind in sites[:50]:
        print(f"  body+0x{off:05x}: 0x{w:08x} -> 0x{tgt:05x}  {kind}")
    if len(sites) > 50:
        print(f"  ... +{len(sites) - 50} more")


if __name__ == "__main__":
    sys.exit(main())
