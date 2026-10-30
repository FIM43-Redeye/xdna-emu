#!/usr/bin/env python3
"""Decode a MAC-mode probe golden into the (lane, A index, B index) pairs silicon sums.

The probe tiles (gen_mac_modes.probe_tiles) put A = e_i against B = ones and against the
ramp B[j] = j + 1, and the mirror image. Per lane l and A index i, the ones tile counts the
pairs (i, *) in lane l and, when that count is one, the ramp tile reads out j + 1; the B side
does the same from the other end. The two decodings must agree, and a count above one (which
this decode cannot resolve) is an error.

Usage: analyze_mac_mode_probe.py <probe golden.json> [--lanes]   (prints the relation)
Library: relation(golden, sign_index) -> {lane: sorted [(i, j, coefficient)]}
"""

import argparse
import json
import sys


def lanes_of(golden, tile, sign):
    """Accumulator lane values (signed) of one tile and signedness flavour."""
    mode, words = golden["mode"], golden["tile"]["out_words"]
    n_signs = len(golden["forms"]) // golden["tile"]["stages"]
    base = (tile * n_signs + sign) * words
    chunk = golden["silicon"][base:base + words]
    lanes = mode["acc_lanes"]
    if mode["acc_bits"] == 32:
        return [w - (1 << 32) if w >> 31 else w for w in chunk]
    out = []
    for l in range(lanes):
        v = chunk[l] | (chunk[lanes + l] << 32)
        out.append(v - (1 << 64) if v >> 63 else v)
    return out


def relation(golden, sign):
    mode = golden["mode"]
    na, nb = mode["a_lanes"], mode["b_lanes"]
    n_lanes = mode["acc_lanes"]
    from_a = {}   # (lane, i) -> j
    for i in range(na):
        ones = lanes_of(golden, 2 * i, sign)
        ramp = lanes_of(golden, 2 * i + 1, sign)
        for l in range(n_lanes):
            if ones[l] == 0:
                if ramp[l]:
                    raise SystemExit(f"lane {l} A[{i}]: ramp nonzero without a pair")
                continue
            if ones[l] != 1:
                raise SystemExit(f"lane {l} A[{i}] has {ones[l]} partners: ramp decode is not unique")
            from_a[(l, i)] = ramp[l] - 1
    from_b = {}   # (lane, j) -> i
    base = 2 * na
    for j in range(nb):
        ones = lanes_of(golden, base + 2 * j, sign)
        ramp = lanes_of(golden, base + 2 * j + 1, sign)
        for l in range(n_lanes):
            if ones[l] == 0:
                if ramp[l]:
                    raise SystemExit(f"lane {l} B[{j}]: ramp nonzero without a pair")
                continue
            if ones[l] != 1:
                raise SystemExit(f"lane {l} B[{j}] has {ones[l]} partners: ramp decode is not unique")
            from_b[(l, j)] = ramp[l] - 1
    pairs_a = {(l, i, j) for (l, i), j in from_a.items()}
    pairs_b = {(l, i, j) for (l, j), i in from_b.items()}
    if pairs_a != pairs_b:
        diff = pairs_a ^ pairs_b
        raise SystemExit(f"A-side and B-side decodings disagree on {len(diff)} pairs, e.g. {sorted(diff)[:6]}")
    rel = {l: [] for l in range(n_lanes)}
    for (l, i, j) in sorted(pairs_a):
        rel[l].append((i, j, 1))
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("golden")
    ap.add_argument("--lanes", action="store_true", help="print every lane's pairs")
    args = ap.parse_args()
    with open(args.golden) as f:
        golden = json.load(f)
    rels = [relation(golden, s) for s in range(4)]
    for s in range(1, 4):
        if rels[s] != rels[0]:
            print(f"sign flavour {s} differs from ss")
    print(f"mode {golden['mode']['stem']}: {sum(len(v) for v in rels[0].values())} pairs over "
          f"{len(rels[0])} lanes, signedness flavours "
          f"{'identical' if all(r == rels[0] for r in rels) else 'DIFFER'}")
    if args.lanes:
        for l, pairs in rels[0].items():
            print(l, pairs)


if __name__ == "__main__":
    main()
