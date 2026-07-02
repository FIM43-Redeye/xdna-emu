"""R3b observation bridge (#140 SP-5b/SP-5c): control-packet readback buffer +
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

SP-5c Phase 2 (docs/superpowers/plans/2026-07-02-sp5c-phase2-dh-capture.md Sec.1b)
adds a second, decoupled routing: a block-mask-replicated flood that only
propagates horizontally on the shim row before climbing a tile's column (both
floods share that climb, so dn_v collapses to a constant -- see
reset_routed_coeffs and the rank-1 guard
test_block_routed_capture_is_rank_deficient_for_dh_dv). geometry["routing"]
selects which coefficient model applies to a capture; an unrecognized value
fails loud rather than silently falling back to the free-flood (Manhattan)
model, because fitting a block-routed capture with the wrong coefficients is
exactly the "clean-but-wrong" failure this design guards against (design Sec.8,
"mis-tagged routing").
"""
import struct

_FREE_FLOOD = "free_flood"
_BLOCK_SHIM_ROW = "block_shim_row"


def _hops(src, tile):
    """Monotone rectilinear hop counts (east, west, north, south) from src to tile."""
    return (max(tile["col"] - src["col"], 0), max(src["col"] - tile["col"], 0),
            max(tile["row"] - src["row"], 0), max(src["row"] - tile["row"], 0))


def reset_routed_coeffs(s1, s2, tile):
    """Interval coefficients (dn_h, dn_v) under the AIE2 timer-reset routing:
    horizontal only on the shim row, then a vertical climb up the tile's column
    (design Sec.1). Both floods share the tile's climb, so the vertical term is
    the source-row difference -- constant across tiles, i.e. d_v is NOT identifiable
    here. Horizontal is the shim-row distance difference, so d_h IS identifiable.
    s1/s2/tile are dicts with 'col'/'row'."""
    dn_h = abs(s2["col"] - tile["col"]) - abs(s1["col"] - tile["col"])
    dn_v = s2["row"] - s1["row"]
    return (float(dn_h), float(dn_v))


def _free_flood_coeffs(s1, s2, tile):
    """Interval coefficients (dn_h, dn_v) under unblocked (free-flood) Manhattan
    routing: absolute hop-count difference on each axis independently."""
    e1, w1, n1, s1h = _hops(s1, tile)
    e2, w2, n2, s2h = _hops(s2, tile)
    return (float((e2 + w2) - (e1 + w1)), float((n2 + s2h) - (n1 + s1h)))


def observe_r3b(readback_bytes, geometry):
    """readback_bytes: little-endian u32 Performance_Counter0 values, one per
    tile at counter_index*4. geometry: parsed geometry.json. Returns a list of
    {dn_h, dn_v, r} for extract_r3b/extract_r3b_dh. Fails loud on short buffer
    and on an unrecognized geometry["routing"] (no silent free-flood fallback --
    see module docstring / plan Sec.1b)."""
    routing = geometry.get("routing")
    if routing is None or routing == _FREE_FLOOD:
        coeffs = _free_flood_coeffs
    elif routing == _BLOCK_SHIM_ROW:
        coeffs = reset_routed_coeffs
    else:
        raise ValueError(f"unrecognized geometry routing: {routing!r}")

    s1 = geometry["sources"]["s1"]
    s2 = geometry["sources"]["s2"]
    tiles = geometry["tiles"]
    need = max(t["counter_index"] for t in tiles) + 1
    if len(readback_bytes) < need * 4:
        raise ValueError(f"readback buffer too short: {len(readback_bytes)} bytes, "
                         f"need >= {need * 4}")
    out = []
    for t in tiles:
        dn_h, dn_v = coeffs(s1, s2, t)
        (r,) = struct.unpack_from("<I", readback_bytes, t["counter_index"] * 4)
        out.append({"dn_h": dn_h, "dn_v": dn_v, "r": float(r)})
    return out
