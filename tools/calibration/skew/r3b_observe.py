"""R3b observation bridge (#140 SP-5b, rev3): control-packet readback buffer +
geometry.json -> per-tile design-row coefficients for extract_r3b. Shared by the
PC and TM R3b kernels (same {a_*, r} contract). See kernel spec rev3 Sec.5.2."""
import struct


def _hops(src, tile):
    """Monotone rectilinear hop counts (east, west, north, south) from src to tile."""
    return (max(tile["col"] - src["col"], 0), max(src["col"] - tile["col"], 0),
            max(tile["row"] - src["row"], 0), max(src["row"] - tile["row"], 0))


def observe_r3b(readback_bytes, geometry):
    """readback_bytes: little-endian u32 Performance_Counter0 values, one per
    tile at counter_index*4. geometry: parsed geometry.json. Returns a list of
    {a_hE, a_hW, a_vN, a_vS, a_turn, r} for extract_r3b. Fails loud on short buffer."""
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
            "a_hE": float(e2 - e1), "a_hW": float(w2 - w1),
            "a_vN": float(n2 - n1), "a_vS": float(s2h - s1h),
            "a_turn": float((e2 + w2) * (n2 + s2h) - (e1 + w1) * (n1 + s1h)),
            "r": float(r),
        })
    return out
