#!/usr/bin/env python3
"""Reference model of the sparse MAC rule read off the probe captures, and its check.

The rule was derived from the `vec_macsparse_<stem>_probe` goldens (see
docs/fidelity-gaps/vector-compute.md); this module states it in Python so that every capture
(probe, ops, rand) can be compared with it lane by lane:

    python3 tools/analyze_mac_sparse.py check [<stem> ...]   # every golden vs the model

Rule (all four non-int4 modes; B is `inner x cols`, column-major in the mask):
  * logical element n = c * inner + k; it owns `bits_per_element` mask bits (128 / (inner*cols)),
    and is *selected* when any of them is set.
  * A group is 4 consecutive elements of one column (k = 4j .. 4j+3). With more than two selected
    elements the whole group contributes nothing.
  * Otherwise the group's two stored slots are rows q = 2j (u = 0) and q = 2j+1 (u = 1) of the
    stored `inner/2 x cols` row-major matrix. Two selected elements take u = 0 then u = 1 in
    ascending position; a lone selected element takes u = 0 at position 0 or 1, u = 1 at 2 or 3.
  * logical B[k][c] is the stored element q * cols + c, or zero where not selected.
"""

import argparse
import json
import os
import struct
import sys

import gen_mac_sparse as g

GROUP = 4


def unit_grid(mode):
    """(inner, unit columns, mask bits per unit): the grid the mask is laid over.

    The unit is the B element, except for 4-bit data where it is the byte holding two
    neighbouring columns (the mask has one bit per byte of logical B).
    """
    _, inner, cols = mode.tile
    ucols = cols // 2 if mode.b_bits == 4 else cols
    return inner, ucols, g.MASK_BITS // (inner * ucols)


def selected(mode, mask, k, c):
    inner, ucols, bpe = unit_grid(mode)
    cu = c // 2 if mode.b_bits == 4 else c
    n = cu * inner + k
    return (mask >> (n * bpe)) & ((1 << bpe) - 1) != 0


def stored_row(mode, mask, k, c):
    """Stored row q feeding B[k][c], or None when the element is zero."""
    if not selected(mode, mask, k, c):
        return None
    j, p = divmod(k, GROUP)
    sel = [i for i in range(GROUP) if selected(mode, mask, GROUP * j + i, c)]
    if len(sel) > 2:
        return None
    if len(sel) == 2:
        u = sel.index(p)
    else:
        u = 0 if p < 2 else 1
    return 2 * j + u


def expand_b(mode, data, mask):
    """B[k][c] raw element patterns (0 where unselected) as {(k, c): value}."""
    _, inner, cols = mode.tile
    out = {}
    for c in range(cols):
        for k in range(inner):
            q = stored_row(mode, mask, k, c)
            if q is None:
                out[(k, c)] = 0
            elif mode.b_bits == 4:
                # Stored data is a row-major `inner/2 x cols/2` matrix of bytes, two nibbles each.
                byte = data[2 * (q * (cols // 2) + c // 2)] | (data[2 * (q * (cols // 2) + c // 2) + 1] << 4)
                out[(k, c)] = (byte >> (4 * (c % 2))) & 15
            else:
                out[(k, c)] = data[q * cols + c]
    return out


def to_value(mode, bits, raw, signed):
    """Numeric value of a raw element pattern."""
    if mode.bf16:
        return struct.unpack("<f", struct.pack("<I", raw << 16))[0]
    if signed and raw >> (bits - 1):
        return raw - (1 << bits)
    return raw


def product(mode, a, data, mask, xs, ys):
    """Per-lane products (row-major rows x cols) of one operand set."""
    rows, inner, cols = mode.tile
    b = expand_b(mode, data, mask)
    lanes = []
    for r in range(rows):
        for c in range(cols):
            acc = 0
            for k in range(inner):
                av = to_value(mode, mode.a_bits, a[r * inner + k], xs)
                bv = to_value(mode, mode.b_bits, b[(k, c)], ys)
                acc += av * bv
            lanes.append(acc)
    return lanes


def wrap(mode, v):
    bits = 64 if mode.acc_bits == 64 else 32
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def golden(kernel):
    path = os.path.join(g.GOLDEN_DIR, kernel + ".json")
    with open(path) as f:
        return json.load(f)


def silicon_lanes(mode, gold, tile, form):
    """Signed (or float) lanes of one tile and form index (flavour-major, stage-minor)."""
    words = gold["tile"]["out_words"]
    nforms = len(gold["forms"])
    base = (tile * nforms + form) * words
    w = gold["silicon"][base:base + words]
    if mode.bf16:
        return [struct.unpack("<f", struct.pack("<I", x))[0] for x in w]
    if mode.acc_bits == 32:
        return [x - (1 << 32) if x >> 31 else x for x in w]
    n = mode.acc_lanes
    return [wrap(mode, w[l] | (w[n + l] << 32)) for l in range(n)]


def tile_sets(mode, gold, tile):
    """The operand sets of a tile from the golden's flat input columns."""
    cols = gold["inputs"]
    a_lanes = gold["tile"]["a_lanes"]
    sets = []
    for si in range(gold["tile"]["sets"]):
        a = cols[3 * si][tile * a_lanes:(tile + 1) * a_lanes]
        d = cols[3 * si + 1][tile * mode.b_stored:(tile + 1) * mode.b_stored]
        m = int(cols[3 * si + 2][tile], 16)
        sets.append((a, d, m))
    return sets


# What each op does to the product P of its own operands and the accumulators (r0, r1) it reads,
# as read off the ops/conf captures. `sub_acc1`/`shift16` etc. are config-word controls.
OP_RULES = {
    "mul": lambda p, r0, r1: p,
    "negmul": lambda p, r0, r1: -p,
    "mac": lambda p, r0, r1: r0 + p,
    "msc": lambda p, r0, r1: r0 - p,
    "negmac": lambda p, r0, r1: -r0 - p,
    "negmsc": lambda p, r0, r1: -r0 + p,
    "addmac": lambda p, r0, r1: r0 + r1 + p,
    "addmsc": lambda p, r0, r1: r0 + r1 - p,
    "submac": lambda p, r0, r1: r0 - r1 + p,
    "submsc": lambda p, r0, r1: r0 - r1 - p,
    # config-word variants (zero_acc1 / shift16 / sub_mul / sub_acc1 / sub_acc2)
    "mulc_sm": lambda p, r0, r1: -p,
    "mac_z": lambda p, r0, r1: p,
    "mac_sh": lambda p, r0, r1: (r0 << 16) + p,
    "mac_sm": lambda p, r0, r1: r0 - p,
    "mac_sa": lambda p, r0, r1: -r0 + p,
    "mac_sm_sa": lambda p, r0, r1: -r0 - p,
    "addmac_z": lambda p, r0, r1: r1 + p,
    "addmac_sa2": lambda p, r0, r1: r0 - r1 + p,
    "addmac_all": lambda p, r0, r1: -r0 - r1 - p,
}
FLOAT_TOLERANCE = 2e-3   # bf16 sums round per the datapath; the emulator golden pins it bit for bit


def lane_ok(mode, want, got):
    if mode.bf16:
        return abs(want - got) <= FLOAT_TOLERANCE * max(1.0, abs(want))
    return wrap(mode, want) == got


def padded_a(mode, a):
    """A elements padded with zeros to the whole tile.

    The narrow encoding of a mode whose A is a 1024-bit operand supplies only the low 512 bits;
    the capture shows the rows beyond them contribute zero.
    """
    need = mode.tile[0] * mode.tile[1]
    return list(a) + [0] * (need - len(a))


def check_capture(mode, gold):
    """Every lane of every tile, flavour and stage against the model; returns (checked, bad, first).

    probe: one set, mul. ops/conf/alt: x0 = mul(s0), x1 = mul(s1), then each op on s2 with x0, x1.
    rand: mul(s0), mac(s1, r0), msc(s2, r1). The accumulator inputs of a stage are taken from
    silicon's earlier stages so one wrong stage does not cascade.
    """
    kind = gold["class"].removeprefix("mac_sparse_")
    ops = gold["tile"]["ops"]
    checked, bad, first = 0, 0, []
    for t in range(gold["tile"]["tiles"]):
        sets = tile_sets(mode, gold, t)
        for fi, (_, xs, ys) in enumerate(mode.signs):
            r = [silicon_lanes(mode, gold, t, fi * len(ops) + s) for s in range(len(ops))]
            prods = [product(mode, padded_a(mode, st[0]), st[1], st[2], xs, ys) for st in sets]
            for si, op in enumerate(ops):
                if kind == "rand":
                    p = prods[si]
                    acc = [0] * len(p) if si == 0 else r[si - 1]
                    want = [(OP_RULES[op](p[l], acc[l], 0)) for l in range(len(p))]
                elif si < 2 and kind != "probe":
                    want = prods[si]
                elif kind == "probe":
                    want = prods[0]
                else:
                    p = prods[2]
                    want = [OP_RULES[op](p[l], r[0][l], r[1][l]) for l in range(len(p))]
                for lane, w in enumerate(want):
                    checked += 1
                    if not lane_ok(mode, w, r[si][lane]):
                        bad += 1
                        if len(first) < 5:
                            first.append((t, fi, op, lane, w, r[si][lane]))
    return checked, bad, first


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    chk = sub.add_parser("check")
    chk.add_argument("stems", nargs="*")
    args = ap.parse_args(argv)
    for mode in [g.BY_STEM[s] for s in args.stems] if args.stems else g.MODES:
        for kind in g.kinds_of(mode):
            kernel = g.kernel_name(mode, kind)
            if not os.path.exists(os.path.join(g.GOLDEN_DIR, kernel + ".json")):
                print(f"{kernel}: no golden")
                continue
            checked, bad, first = check_capture(mode, golden(kernel))
            print(f"{kernel}: {checked} lanes, {bad} differ {first}")


if __name__ == "__main__":
    main()
