"""The integer MAC geometry table (llvm-aie aiev2_vmult.h, as extracted by xdna-archspec) in python.

Reads the build-generated `gen_mac_geometry.rs` (newest under target/*/build/xdna-archspec-*/out), so the
analysis tools use the same table as the emulator. `lane_products` evaluates a geometry on raw operands.
"""
import glob
import os
import re
import struct

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_table():
    files = glob.glob(os.path.join(REPO, "target/*/build/xdna-archspec-*/out/gen_mac_geometry.rs"))
    text = open(max(files, key=os.path.getmtime)).read()
    table = {}
    pat = re.compile(r"\((\d), (\d), (\d), GeometryEntry \{ bits_x: (\d+), bits_y: (\d+), shape: MacShape::(\w+) \{([^}]*)\}, "
                     r"acc_cmb: (\d), bfloat: (\w+), sparse: (\w+)")
    for m in pat.finditer(text):
        am, bm, cm, bx, by, kind, params, cmb, bf, sp = m.groups()
        p = {k.strip(): int(v) for k, v in (kv.split(":") for kv in params.split(","))}
        table[(int(am), int(bm), int(cm))] = dict(bits_x=int(bx), bits_y=int(by), kind=kind, p=p,
                                                  acc_cmb=int(cmb), bfloat=bf == "true", sparse=sp == "true")
    return table


def lanes_terms(g):
    p, k = g["p"], g["kind"]
    if k == "Matmul":
        return p["rows"] * p["cols"], p["inner"]
    if k == "Elementwise":
        return p["lanes"], p["k"]
    if k == "Conv":
        return p["outputs"] * p["channels"], p["taps"]
    return p["rows"] * p["cols"] * p["channels"], p["inner"]


def operand_indices(g, lane, term):
    p, k = g["p"], g["kind"]
    if k == "Matmul":
        r, c = divmod(lane, p["cols"])
        return r * p["inner"] + term, term * p["cols"] + c
    if k == "Elementwise":
        i = lane + term * p["lanes"]
        return i, i
    if k == "Conv":
        pp, c = divmod(lane, p["channels"])
        return (pp + term) * p["channels"] + c, term * p["channels"] + c
    ch = lane % p["channels"]
    tl = lane // p["channels"]
    r, c = divmod(tl, p["cols"])
    return (r * p["inner"] + term) * p["channels"] + ch, (term * p["cols"] + c) * p["channels"] + ch


def elements(raw, bits, signed):
    """Elements of a little-endian raw vector, `bits` wide each."""
    n = len(raw) * 8 // bits
    out = []
    big = int.from_bytes(raw, "little")
    for i in range(n):
        v = (big >> (i * bits)) & ((1 << bits) - 1)
        if signed and v >> (bits - 1):
            v -= 1 << bits
        out.append(v)
    return out


def wrap(x, bits):
    x &= (1 << bits) - 1
    return x - (1 << bits) if x >> (bits - 1) else x


def lane_products(g, a, b, sx, sy):
    """products[lane][term] for raw A and B."""
    A = elements(a, g["bits_x"], sx)
    B = elements(b, g["bits_y"], sy)
    lanes, terms = lanes_terms(g)
    return [[A[operand_indices(g, l, t)[0]] * B[operand_indices(g, l, t)[1]] for t in range(terms)] for l in range(lanes)]


def pack_lanes(vals, cmb):
    w = 32 * cmb
    return b"".join((v & ((1 << w) - 1)).to_bytes(w // 8, "little") for v in vals)
