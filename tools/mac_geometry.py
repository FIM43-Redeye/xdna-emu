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


# -- silicon-only geometries (not in the header table) ----------------------------------------
# kind "ComplexElem": 8 complex lanes (acc64 lane 2n+p); term = 2t+q; A index 16t+2n+q, B index 16t+2n+(q^p).
# kind "FloatConf": amode >= 2 on the integer multiply: low byte of 16-bit elements, each product wrapped to 14 bits signed.
def silicon_geometry(code):
    am, bm, cm = code & 3, (code >> 2) & 3, code >> 4
    if am == 1 and bm == 1:
        bm = 3
    if (am, bm, cm) == (1, 0, 1):
        return dict(bits_x=32, bits_y=16, kind="ComplexElem", p=dict(sets=1), acc_cmb=2, bfloat=False, sparse=False)
    if (am, bm, cm) == (1, 3, 4):
        return dict(bits_x=16, bits_y=16, kind="ComplexElem", p=dict(sets=2), acc_cmb=2, bfloat=False, sparse=False)
    if am >= 2 and cm == 0:
        return dict(bits_x=16, bits_y=16, kind="FloatConf", p=dict(rows=4, inner=8, cols=4), acc_cmb=1, bfloat=False, sparse=False)
    if am >= 2 and cm == 1:
        return dict(bits_x=16, bits_y=16, kind="FloatConf", p=dict(lanes=16, k=2), acc_cmb=1, bfloat=False, sparse=False)
    g = load_table().get((am, bm, cm))
    return g if g and not g["bfloat"] and not g["sparse"] else None


def terms_of(g, a, b, sx, sy):
    """products[lane][term] for any silicon_geometry, as the multiplier array forms them."""
    k = g["kind"]
    if k == "ComplexElem":
        A = elements(a, g["bits_x"], sx)
        B = elements(b, g["bits_y"], sy)
        sets = g["p"]["sets"]
        out = []
        for lane in range(16):
            n, pp = divmod(lane, 2)
            out.append([A[16 * (t // 2) + 2 * n + t % 2] * B[16 * (t // 2) + 2 * n + ((t % 2) ^ pp)] for t in range(2 * sets)])
        return out
    if k == "FloatConf":
        A = list(a[0::2])
        B = list(b[0::2])
        p = g["p"]
        w14 = lambda x: wrap(x, 14)
        if "rows" in p:
            return [[w14(A[(l // 4) * 8 + t] * B[t * 4 + l % 4]) for t in range(8)] for l in range(16)] + [[0] for _ in range(16)]
        return [[w14(A[l + 16 * t] * B[l + 16 * t]) for t in range(2)] for l in range(16)] + [[0] for _ in range(16)]
    return lane_products(g, a, b, sx, sy)


def term_operands(g, a, b, sx, sy):
    """operands[lane][term] = (A element, B element) for any silicon_geometry except FloatConf."""
    k = g["kind"]
    A = elements(a, g["bits_x"], sx)
    B = elements(b, g["bits_y"], sy)
    if k == "ComplexElem":
        sets = g["p"]["sets"]
        out = []
        for lane in range(16):
            n, pp = divmod(lane, 2)
            out.append([(A[16 * (t // 2) + 2 * n + t % 2], B[16 * (t // 2) + 2 * n + ((t % 2) ^ pp)]) for t in range(2 * sets)])
        return out
    lanes, terms = lanes_terms(g)
    return [[(A[operand_indices(g, l, t)[0]], B[operand_indices(g, l, t)[1]]) for t in range(terms)] for l in range(lanes)]


def split_parts(g, a_el, b_el):
    """The partial products one term's product is formed from. A 32-bit A element is two 16-bit
    halves (low unsigned, high signed/unsigned as the element); everything else is one part."""
    if g["bits_x"] == 32:
        lo = a_el & 0xFFFF
        hi = a_el >> 16
        return [lo * b_el, (hi * b_el) << 16]
    return [a_el * b_el]
