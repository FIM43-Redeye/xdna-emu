#!/usr/bin/env python3
"""Decode an r4d3 output buffer: OLD-tag count and the O/N pattern string."""
import struct
import sys

OLD_TAG, NEW_TAG = 0x0A, 0x0B


def decode(path):
    d = open(path, "rb").read()
    w = struct.unpack("<%dI" % (len(d) // 4), d)
    pat = "".join("O" if x >> 24 == OLD_TAG else "N" if x >> 24 == NEW_TAG else "?" for x in w)
    # index check: word i must carry index i when not '?'
    ok_idx = all((x & 0xFFFFFF) == i for i, x in enumerate(w) if x >> 24 in (OLD_TAG, NEW_TAG))
    stale = len(pat) - len(pat.lstrip("O"))
    monotone = pat == "O" * stale + "N" * (len(pat) - stale)
    return {"n": len(w), "stale": stale, "monotone": monotone, "idx_ok": ok_idx, "pattern": pat}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = decode(p)
        print(p, {k: v for k, v in r.items() if k != "pattern"})
        print(r["pattern"])
