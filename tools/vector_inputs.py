"""Designed inputs for the vector-verify capture kernels.

Every kernel's input arrays come from this module, never from an external model
or corpus. Each array is `edge cases + seeded PRNG fill`:

  * The edge set is derived from the operation's definition (where a rounding
    tie, a saturation bound, or a sign boundary sits for the configured shift
    and element widths). It is what real silicon must settle, so it comes first
    and is always kept whole.
  * The remainder up to the kernel's buffer size is filled from a seeded PRNG
    (`random.Random.getrandbits`, stable across Python versions), so the arrays
    are reproducible byte-for-byte from the seed.

The expected values are NOT computed here: they are captured from NPU1 silicon
(tools/capture_silicon.py). These functions only choose what to feed it.
"""

import random

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1
INT16_MIN, INT16_MAX = -(2 ** 15), 2 ** 15 - 1
INT8_MIN, INT8_MAX = -(2 ** 7), 2 ** 7 - 1


class Prng:
    """Seeded generator of raw bit fields (reproducible across Python versions)."""

    def __init__(self, seed):
        self._r = random.Random(seed)

    def bits(self, width):
        return self._r.getrandbits(width)

    def signed(self, width):
        """Uniform signed integer over the full `width`-bit two's-complement range."""
        v = self.bits(width)
        return v - (1 << width) if v >> (width - 1) else v


def _dedupe(values):
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def fill(edges, n, draw):
    """`edges` (deduplicated, order kept) followed by `draw()` values up to `n`."""
    out = _dedupe(edges)
    if len(out) > n:
        raise ValueError(f"{len(out)} edge values do not fit a buffer of {n}")
    while len(out) < n:
        out.append(draw())
    return tuple(out)


def _to_unsigned(v, width):
    return v & ((1 << width) - 1)


# --- SRS: int32 -> acc64 -> (>> shift, round, saturate) -> int16 ---------------

def srs_inputs(n, shift=4, seed=0x5251):
    """int32 accumulator inputs for a right-shift-round-saturate to int16.

    A value is `kept << shift | low`. The discarded bits `low` decide rounding
    and `kept` decides the output; so the grid is kept x low:
      * kept in {0, 1, 2, -1, -2}: both signs, both parities of the kept LSB;
        low in {half-1, half, half+1}: just below / exactly at (tie) / just above.
      * kept = +-1 with low 0 and all-ones: exact and maximal fractions.
      * kept at and beyond the int16 output bounds (32767, 32766, -32768,
        -32767, 32768, -32769) with low in {0, half, half+1}: saturation, with
        and without a rounding carry into the overflow.
      * 0, +-1, int32 min/max (+-1 off): the accumulator-width extremes.
    Candidates that do not fit int32 (e.g. saturation bounds at large shifts,
    which an int32 input cannot reach) are dropped. Shift 0 discards nothing, so only the kept-value cases remain.
    The remainder is PRNG: half full-range int32, half sized so the kept value
    lands inside the int16 range (the interesting part of the rounding space).
    """
    full = (1 << shift) - 1
    half = 1 << (shift - 1) if shift > 0 else 0
    lows = (half - 1, half, half + 1)
    edges = [0, 1, -1, INT32_MAX, INT32_MIN, INT32_MAX - 1, INT32_MIN + 1]
    for kept in (0, 1, 2, -1, -2):
        for low in lows:
            edges.append((kept << shift) | low)
    for kept in (1, -1):
        for low in (0, full):
            edges.append((kept << shift) | low)
    for kept in (32767, 32766, -32768, -32767, 32768, -32769):
        for low in (0, half, half + 1):
            edges.append((kept << shift) | low)
    if shift == 0:
        edges += [127, -128, 255, 256, 65535, -65536, 0x40000000, -0x40000000]
    edges = [v for v in edges if INT32_MIN <= v <= INT32_MAX]
    width = min(32, shift + 16)
    prng = Prng(seed)
    flip = [False]

    def draw():
        flip[0] = not flip[0]
        return prng.signed(32) if flip[0] else prng.signed(width)

    return fill(edges, n, draw)


# --- Convert: f32 bit pattern -> bf16 (round-narrow of the low 16 bits) --------

# bf16 high halves: even/odd LSB pairs at several exponents, the largest finite
# bf16 neighbourhood (rounding up overflows to Inf) and the smallest normals.
_CONV_HI = (0x3F80, 0x3F81,      # 1.0 (even LSB), next bf16 (odd LSB)
            0x4048, 0x4049,      # ~3.1 region, even/odd
            0x7F7E, 0x7F7F,      # largest finite bf16 (odd) and its even neighbour
            0x0080, 0x0081,      # smallest normals, even/odd
            0x3A00, 0x3A01,      # small mid-range exponent
            0x5F12, 0x5F13)      # large mid-range exponent
# Low 16 bits: zero, just above zero, just below half, exactly half (tie),
# just above half, all ones.
_CONV_LOW = (0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF)


def conv_inputs(n, seed=0xC0DE):
    """Normal finite f32 bit patterns for f32 -> bf16 rounding.

    Grid: every hi in _CONV_HI x every low in _CONV_LOW x both signs, so each
    tie / near-tie appears with an even and an odd bf16 LSB, at both signs
    (sign matters for the directional and symmetric rounding modes), near the
    bf16 overflow edge and at the smallest normals. Plus +-0. The remainder is
    PRNG normal finite floats (random sign, exponent 1..254, random mantissa).
    """
    edges = [0x00000000, 0x80000000]
    for hi in _CONV_HI:
        for low in _CONV_LOW:
            edges.append((hi << 16) | low)
            edges.append(0x80000000 | (hi << 16) | low)
    prng = Prng(seed)

    def draw():
        exp = 1 + prng.bits(8) % 254
        return (prng.bits(1) << 31) | (exp << 23) | prng.bits(23)

    return fill(edges, n, draw)


def conv_edge_inputs(n, seed=0xED6E):
    """Non-normal f32 bit patterns for f32 -> bf16: zero, denormal, Inf, NaN.

    Denormals (exponent 0): the smallest, the largest, powers of two across the
    mantissa, and values whose low 16 bits sit at a rounding tie / just below /
    just above half (0x8000/0x7FFF/0x8001) with an even and odd bf16 LSB, so a
    rounding mode that acts on denormals is distinguishable from flush-to-zero;
    the value 0x007FFFFF rounds up into the smallest normal. Both signs.
    Inf, and NaNs whose payload lives only in the low 16 bits (a plain
    truncation would turn those into Inf), only in the high bits, quiet and
    signalling, all-ones. The remainder is PRNG draws from the same classes.
    """
    dn = [0x00000001, 0x0000007F, 0x00004000, 0x00007FFF, 0x00008000, 0x00008001,
          0x0000FFFF, 0x00010000, 0x00017FFF, 0x00018000, 0x00018001, 0x0001FFFF,
          0x00020000, 0x00028000, 0x003FFFFF, 0x00400000, 0x00408000, 0x007F7FFF,
          0x007F8000, 0x007F8001, 0x007FFFFF, 0x00800000 - 0x10000]
    edges = [0x00000000, 0x80000000]
    for v in dn:
        edges += [v, 0x80000000 | v]
    nan = [0x7F800001, 0x7F80FFFF, 0x7F808000, 0x7FBFFFFF, 0x7FC00000, 0x7FC00001,
           0x7FC08000, 0x7FFFFFFF, 0x7FA00000, 0x7F810000]
    edges += [0x7F800000, 0xFF800000]
    for v in nan:
        edges += [v, 0x80000000 | v]
    prng = Prng(seed)

    def draw():
        sign = prng.bits(1) << 31
        kind = prng.bits(2)
        if kind < 2:                                   # denormal
            return sign | (1 + prng.bits(23) % 0x7FFFFF)
        if kind == 2:                                  # NaN
            return sign | 0x7F800000 | (1 + prng.bits(23) % 0x7FFFFF)
        return sign | (prng.bits(7) << 16) | 0x8000 * prng.bits(1)   # tie-ish denormal

    return fill(edges, n, draw)


# --- Pack: int16 -> int8 (truncate or saturate) --------------------------------

def pack_inputs(n, seed=0x9ACC):
    """int16 inputs around the int8 range and sign boundaries.

    +-126..129 around the int8 bounds, 255/256/257 and the negatives (low byte
    versus saturated value differ most around the byte boundaries), the int16
    extremes, and values whose low byte is a valid int8 while the high byte is
    not (0x017F, 0x0180, 0x7F7F, 0x7F80, 0x8080). PRNG remainder: alternately
    full-range int16 and values within +-600 of zero.
    """
    edges = [0, 1, -1]
    edges += [126, 127, 128, 129, -126, -127, -128, -129]
    edges += [254, 255, 256, 257, -254, -255, -256, -257]
    edges += [383, 384, 0x7F7F, 0x7F80, -0x7F80, -0x7F7F, 0x4000, -0x4000]
    edges += [INT16_MAX, INT16_MIN, INT16_MAX - 1, INT16_MIN + 1]
    prng = Prng(seed)
    flip = [False]

    def draw():
        flip[0] = not flip[0]
        return prng.signed(16) if flip[0] else prng.signed(10) + prng.signed(8) // 8

    return tuple(max(INT16_MIN, min(INT16_MAX, v)) for v in fill(edges, n, draw))


# --- UPS: int16 -> acc32 (<< shift) --------------------------------------------

def ups_inputs(n, seed=0x0B5):
    """int16 inputs for a widening left shift.

    Sign-extension edges: values straddling each byte/nibble boundary where bit
    15 or the sign extension changes (0x7F/0x80/0xFF/0x100 and negatives),
    int16 min/max and neighbours, 0 and +-1. PRNG remainder is full-range.
    """
    edges = [0, 1, -1, 2, -2, 0x7F, 0x80, 0xFF, 0x100, -0x7F, -0x80, -0x81, -0x100,
             0x0FFF, 0x1000, -0x1000, 0x3FFF, 0x4000, -0x4000, -0x4001,
             INT16_MAX, INT16_MIN, INT16_MAX - 1, INT16_MIN + 1]
    prng = Prng(seed)
    return fill(edges, n, lambda: prng.signed(16))


# --- Matmul --------------------------------------------------------------------
# Inputs are flat row-major tile batches: A is batch x (M*K), B is batch x (K*N).

def matmul_int_inputs(batch, m, k, nn, bits, seed):
    """(A, B) integer tile batches covering the full `bits`-bit signed range.

    Tile 0 is all-minimum x all-minimum (largest-magnitude positive products,
    exercises accumulator wrap for wide types), tile 1 all-maximum, tile 2
    minimum x maximum (negative products), tile 3 zeros, tile 4 -1 x -1; the
    remaining tiles are full-range PRNG.
    """
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    prng = Prng(seed)
    a, b = [], []
    constants = [(lo, lo), (hi, hi), (lo, hi), (0, 0), (-1, -1)]
    for t in range(batch):
        if t < len(constants):
            av, bv = constants[t]
            a += [av] * (m * k)
            b += [bv] * (k * nn)
        else:
            a += [prng.signed(bits) for _ in range(m * k)]
            b += [prng.signed(bits) for _ in range(k * nn)]
    return tuple(a), tuple(b)


def _bf16(sign, exp, mant):
    return (sign << 15) | (exp << 7) | mant


def matmul_bf16_inputs(batch, m, k, nn, seed=0xBF16):
    """(A, B) bf16 bit-pattern tile batches: finite values, mixed exponents.

    Tile 0: all ones. Tile 1: all zeros (mixed +-0). Tile 2: one large
    (2^8) and the rest small (2^-8) values so accumulation order and rounding
    of small addends show. Tiles 3-5: exact-cancellation rows (A row holds
    x, -x pairs against equal B column pairs, so every output is a sum of
    exactly-cancelling products). Remaining tiles: PRNG, random sign, exponent
    within +-8 of the bias (no Inf/NaN reachable), random mantissa.
    """
    prng = Prng(seed)
    one, neg_one = _bf16(0, 127, 0), _bf16(1, 127, 0)
    pos_zero, neg_zero = 0x0000, 0x8000

    def rnd_val():
        return _bf16(prng.bits(1), 119 + prng.bits(4) % 17, prng.bits(7))

    a, b = [], []
    for t in range(batch):
        ta, tb = [], []
        if t == 0:
            ta, tb = [one] * (m * k), [one] * (k * nn)
        elif t == 1:
            ta = [pos_zero if i % 2 else neg_zero for i in range(m * k)]
            tb = [neg_zero if i % 3 else pos_zero for i in range(k * nn)]
        elif t == 2:
            big, small = _bf16(0, 127 + 8, 0), _bf16(0, 127 - 8, 0)
            ta = [big if i % k == 0 else small for i in range(m * k)]
            tb = [small if i // nn == 0 else one for i in range(k * nn)]
        elif t in (3, 4, 5):
            # Row i of A: x0, -x0, x1, -x1, ...; column j of B: y0, y0, y1, y1, ...
            for _ in range(m):
                xs = [rnd_val() for _ in range(k // 2)]
                ta += [v for x in xs for v in (x, x ^ 0x8000)]
            for _ in range(k // 2):
                ys = [rnd_val() for _ in range(nn)]
                tb += ys + ys
            if t == 5:
                ta = [v ^ 0x8000 if v == neg_one else v for v in ta]
        else:
            ta = [rnd_val() for _ in range(m * k)]
            tb = [rnd_val() for _ in range(k * nn)]
        a += ta
        b += tb
    return tuple(a), tuple(b)


def matmul_bf16_edge_inputs(batch, m, k, nn, seed=0xB0F):
    """(A, B) bf16 tile batches aimed at overflow, cancellation, signed zeros and
    accumulation order (4x8x4 tiles, K=8).

    Designed tiles (rest: PRNG with exponents drawn from the whole finite range,
    so many results overflow):
      0-2   max finite x ones / x max / -max x max: sums overflowing to +-Inf
      3     +max*max in the first half of K, -max*max in the second: Inf - Inf
      4     rows holding 1..4 copies of 2^127 against ones: 2^128 first overflows
            at the second copy (row-dependent overflow threshold)
      5-8   (2^30, 1, -2^30) at different K positions: the 1 survives or is
            absorbed depending on summation order
      9-10  x, -x(1+ulp) near-cancellation and exact x,-x pairs
      11-13 signed zeros: -0 * +1, +0 * -1, mixed -0/+0 products
      14    Inf * 0 and Inf * finite
      15    subnormal-scale products (tiny exponents) summed with large values
    """
    prng = Prng(seed)
    one, neg_one = _bf16(0, 127, 0), _bf16(1, 127, 0)
    mx, nmx = _bf16(0, 254, 127), _bf16(1, 254, 127)
    p127, p30, n30 = _bf16(0, 254, 0), _bf16(0, 127 + 30, 0), _bf16(1, 127 + 30, 0)
    pinf, zero, nzero = _bf16(0, 255, 0), 0x0000, 0x8000

    def rows(f):
        return [x for i in range(m) for x in f(i)]

    def cols(f):
        return [x for kk in range(k) for x in f(kk)]

    def pad(lst, size):
        return (lst + [zero] * size)[:size]

    tiles = []
    ones_b = [one] * (k * nn)
    tiles.append(([mx] * (m * k), ones_b))
    tiles.append(([mx] * (m * k), [mx] * (k * nn)))
    tiles.append(([nmx] * (m * k), [mx] * (k * nn)))
    tiles.append((rows(lambda i: [mx] * (k // 2) + [nmx] * (k // 2)), [mx] * (k * nn)))
    tiles.append((rows(lambda i: pad([p127] * (i + 1), k)), ones_b))
    for pos in ((0, 1, 2), (2, 1, 0), (1, 0, 7), (3, 7, 5)):
        a = rows(lambda i, pos=pos: [p30 if kk == pos[0] else (one if kk == pos[1] else
                                    (n30 if kk == pos[2] else zero)) for kk in range(k)])
        tiles.append((a, ones_b))
    xs = [_bf16(0, 120 + i, 17 * i + 5) for i in range(4)]
    tiles.append((rows(lambda i: [v for x in xs for v in (x, (x + 1) ^ 0x8000 if i % 2 else x ^ 0x8000)]),
                  [one] * (k * nn)))
    ys = [[_bf16(0, 122 + j, 3 * j + kk) for j in range(nn)] for kk in range(k // 2)]
    tiles.append((rows(lambda i: [v for x in xs for v in (x, x ^ 0x8000)]),
                  [v for y in ys for v in y + y]))
    tiles.append(([nzero] * (m * k), ones_b))
    tiles.append(([zero] * (m * k), [neg_one] * (k * nn)))
    tiles.append((rows(lambda i: [nzero if (i + kk) % 2 else zero for kk in range(k)]),
                  [neg_one if kk % 2 else one for kk in range(k) for _ in range(nn)]))
    tiles.append((rows(lambda i: [pinf if kk == i else one for kk in range(k)]),
                  [zero if kk % 2 else one for kk in range(k) for _ in range(nn)]))
    tiles.append((rows(lambda i: [_bf16(0, 1 + kk, 3) if kk else mx for kk in range(k)]),
                  [_bf16(0, 1, 5)] * (k * nn)))

    a_out, b_out = [], []
    for t in range(batch):
        if t < len(tiles):
            ta, tb = tiles[t]
        else:
            ta = [_bf16(prng.bits(1), 1 + prng.bits(8) % 254, prng.bits(7)) for _ in range(m * k)]
            tb = [_bf16(prng.bits(1), 1 + prng.bits(8) % 254, prng.bits(7)) for _ in range(k * nn)]
        assert len(ta) == m * k and len(tb) == k * nn, t
        a_out += ta
        b_out += tb
    return tuple(a_out), tuple(b_out)


# --- Shuffle -------------------------------------------------------------------

def shuffle_inputs(n):
    """Operands for a two-input 512-bit byte shuffle, as int32 words.

    Words 0..15 are operand A with byte i = 1 + i (1..64); words 16..31 are
    operand B with byte i = 65 + i (65..128), little-endian. Every byte is
    distinct and nonzero, so each output byte names the source byte it came
    from and a zeroed output byte cannot be mistaken for a copied one. The rest
    of the buffer (the kernel's output-sized padding) is zero.
    """
    def word(base):
        return [sum(((base + 4 * w + j) & 0xFF) << (8 * j) for j in range(4))
                for w in range(16)]
    words = [w - (1 << 32) if w >> 31 else w for w in word(1) + word(65)]
    assert len(words) <= n
    return tuple(words + [0] * (n - len(words)))
