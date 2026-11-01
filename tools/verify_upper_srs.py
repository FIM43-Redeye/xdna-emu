#!/usr/bin/env python3
"""Check the UPPER_SRS wired-OR rule on a bit-pattern probe (COLLISION_BITPAT=1).

Registers carry one-hot tags (bit k of the low byte = register k, bit 12 = low half of the
register, bit 13 = high half). The rule under test: when a vsrsm (reads the low 512 bits)
and a cm-source vsrs / vst.srs (reads 1024 bits) issue together, the vsrsm result lane L is
OR(own low-half lane L, other's high-half lane L), and the other's high-half result lanes
carry the same OR. This prints, per co-issuable class pair, whether the tags show it.
"""
import collections
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import analyze_collision_probe as an  # noqa: E402
import gen_collision_probe as gen     # noqa: E402


def elem_size(key):
    return int(re.search(r"_[sd](\d+)_", key).group(1)) // 8


def tags(out, size):
    res = []
    for i in range(0, len(out), size):
        v = int.from_bytes(out[i:i + size], "little")
        res.append((frozenset(k for k in range(8) if v >> k & 1),
                    frozenset(h for h, bit in (("l", 0x1000), ("h", 0x2000)) if v & bit)))
    return res


def main(d):
    _, hw = an.load_points(d, "r1")
    verdict = collections.defaultdict(list)
    for pid, (p, rec, dom) in sorted(hw.items()):
        if p["kind"] != "pair":
            continue
        vkey = p["a"] if p["a"].startswith("vsrsm") else p["b"]
        vslot = "A" if p["a"].startswith("vsrsm") else "B"
        vr = p["ra"] if vslot == "A" else p["rb"]
        out = an.own_output(p, gen.BY_KEY[vkey], vr, rec, vslot)
        t = tags(out, elem_size(vkey))
        nz = [x for x in t if x[0]]
        both_halves = bool(nz) and all("l" in m and "h" in m for _, m in nz)
        want = {0, 3} if p["cfg"] == "sep" else {0}
        regs_ok = bool(nz) and all(r == want for r, _ in nz)
        other = p["b"] if vslot == "A" else p["a"]
        verdict[(p["cfg"], vkey, other)] = ["OR" if both_halves and regs_ok else "clean"]
    summary = collections.Counter((cfg, other_kind(o), v[0]) for (cfg, _, o), v in verdict.items())
    for k, n in sorted(summary.items()):
        print(k, n)


def other_kind(key):
    return key


if __name__ == "__main__":
    main(sys.argv[1])
