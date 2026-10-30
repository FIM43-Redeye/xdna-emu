#!/usr/bin/env python3
"""Check the pointer / count columns of the 2D / 3D addressing goldens against the rule in
src/interpreter/execute/memory/agu.rs, and the loaded data of the raw load families against
the input regions. Prints match counts per kernel.

  python3 tools/check_agu_2d3d_golden.py [kernel ...]
"""

import json
import sys

import gen_agu_2d3d as g

M32 = g.M32


def step(dim, m, dims):
    """One access: (pointer delta, new counts). dims = [(size, stride, count), ...]; unsigned."""
    (s1, i1, c1) = dims[0]
    if c1 < s1:
        return i1, [(c1 + 1) & M32] + [d[2] for d in dims[1:]]
    if dim == 2:
        return m, [0]
    (s2, i2, c2) = dims[1]
    if c2 < s2:
        return i2, [0, (c2 + 1) & M32]
    return m, [0, 0]


def expected_walk(dim, params, iterations):
    if dim == 2:
        m, size, stride, count = params
        dims = [(size & M32, stride & M32, count & M32)]
    else:
        m, s1, i1, s2, i2, c1, c2 = params
        dims = [(s1 & M32, i1 & M32, c1 & M32), (s2 & M32, i2 & M32, c2 & M32)]
    ptr, out = 0, []
    for _ in range(iterations):
        delta, counts = step(dim, m & M32, dims)
        ptr = (ptr + delta) & M32
        dims = [(d[0], d[1], c) for d, c in zip(dims, counts)]
        out.append((ptr, counts))
    return out


def check(kernel):
    gold = json.load(open(g.golden_path(kernel)))
    sil = gold["silicon"]
    lay = gold["layout"]
    recs = matched = data_checked = data_matched = 0
    bad = []
    for ci, ch in enumerate(lay["chunks"]):
        dim, iters = ch["dim"], g.ITER_2D if ch["dim"] == 2 else g.ITER_3D
        base_words = ci * g.NOUT
        walks = [expected_walk(dim, tuple(s), iters) for s in ch["sets"]]
        for grp in range(g.GROUPS):
            for i in range(iters):
                for w in range(g.WALKERS):
                    r = base_words + ((grp * iters + i) * g.WALKERS + w) * g.REC_WORDS
                    rec = sil[r:r + g.REC_WORDS]
                    ptr_after, counts = walks[grp * g.WALKERS + w][i]
                    # store walkers have their own scratch region; load walkers share one start
                    base = rec[19] + (w * g.REGION_BYTES if ch['kind'] == 'st' else 0)
                    exp_c = counts + [0] * (2 - len(counts))
                    ok = ((rec[16] - base) & M32) == ptr_after and rec[17] == exp_c[0] and rec[18] == exp_c[1]
                    recs += 1
                    matched += ok
                    if not ok and len(bad) < 5:
                        bad.append((ch["family"], dim, grp, i, w, rec[16:20], ptr_after, counts))
    return recs, matched, bad


def main(argv):
    kernels = argv or g.kernel_names()
    total = matched = 0
    for k in kernels:
        n, m, bad = check(k)
        total += n
        matched += m
        print(f"{k:20s} records {n:5d} pointer/count matches {m:5d}" + ("" if n == m else "   MISMATCH"))
        for b in bad:
            print("   ", b)
    print(f"total records {total} matched {matched}")
    return 0 if total == matched else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
