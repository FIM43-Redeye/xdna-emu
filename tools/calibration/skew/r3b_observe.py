"""R3b observation bridge (#140 SP-5b): control-packet readback buffer +
geometry.json -> per-tile design-row coefficients for extract_r3b. Shared by the
PC and TM R3b kernels (same {dn_h, dn_v, r} contract).

The R3b interval model is isotropic within each axis:
    r_X = const + dn_h*d_h + dn_v*d_v
where dn_h/dn_v are the change in *absolute* hop count between the two floods,
    dn_h = |X.col - s2.col| - |X.col - s1.col|,  dn_v likewise for rows.
A two-source two-flood interval cannot identify within-axis direction anisotropy
(d_hE vs d_hW, d_vN vs d_vS): a_hE - a_hW = s1.col - s2.col is constant for every
tile at any placement, so the East/West (and North/South) columns collapse. See
docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md and
the rank-2 guard in test_skew_r3b_identifiability.py. Hence {d_h, d_v} only.
"""
import struct


def _hops(src, tile):
    """Monotone rectilinear hop counts (east, west, north, south) from src to tile."""
    return (max(tile["col"] - src["col"], 0), max(src["col"] - tile["col"], 0),
            max(tile["row"] - src["row"], 0), max(src["row"] - tile["row"], 0))


def observe_r3b(readback_bytes, geometry):
    """readback_bytes: little-endian u32 Performance_Counter0 values, one per
    tile at counter_index*4. geometry: parsed geometry.json. Returns a list of
    {dn_h, dn_v, r} for extract_r3b. Fails loud on short buffer."""
    s1 = geometry["sources"]["s1"]
    s2 = geometry["sources"]["s2"]
    tiles = geometry["tiles"]
    need = max(t["counter_index"] for t in tiles) + 1
    if len(readback_bytes) < need * 4:
        raise ValueError(f"readback buffer too short: {len(readback_bytes)} bytes, "
                         f"need >= {need * 4}")
    out = []
    for t in tiles:
        e1, w1, n1, s1h = _hops(s1, t)
        e2, w2, n2, s2h = _hops(s2, t)
        (r,) = struct.unpack_from("<I", readback_bytes, t["counter_index"] * 4)
        out.append({
            "dn_h": float((e2 + w2) - (e1 + w1)),
            "dn_v": float((n2 + s2h) - (n1 + s1h)),
            "r": float(r),
        })
    return out
