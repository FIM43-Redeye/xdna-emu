"""R3b observation-bridge tests (#140 SP-5b/SP-5c): readback buffer + geometry ->
{dn_h, dn_v, r} design rows, against a frozen fixture, plus an observe->extract
round-trip that recovers a known {d_h, d_v}.

The routing-aware tests (SP-5c Phase 2, plan Sec.1b) cover the
geometry["routing"] branch: "free_flood"/absent must stay byte-for-byte
back-compat with the original Manhattan coefficients (the shipped
sp5_skew_r3b_pc/geometry.json has no routing key), "block_shim_row" must match
reset_routed_coeffs, and anything else must fail loud (no silent free-flood
fallback -- see r3b_observe.py's module docstring)."""
import math
import struct
import pytest
from calibration.skew.r3b_observe import observe_r3b, reset_routed_coeffs
from calibration.skew.r3b_extract import extract_r3b

# s1 at (0,0), s2 at (2,4) [opposite corner of a 3-col x 5-row virtual frame].
GEOM = {
    "sources": {"s1": {"col": 0, "row": 0}, "s2": {"col": 2, "row": 4}},
    "tiles": [
        {"col": 1, "row": 2, "counter_index": 0},  # equidistant both axes
        {"col": 2, "row": 2, "counter_index": 1},  # east edge
        {"col": 0, "row": 2, "counter_index": 2},  # west edge
        {"col": 1, "row": 3, "counter_index": 3},  # off the equidistant row
    ],
}

# Plan Sec.1b's worked example: s1 = shim(0,0), s2 = core(2,5) [the bring-up
# corner placement], horizontal spine (0,3),(1,3),(2,3) -> dn_h = [2,0,-2].
GEOM_BLOCK = {
    "routing": "block_shim_row",
    "sources": {"s1": {"col": 0, "row": 0}, "s2": {"col": 2, "row": 5}},
    "tiles": [
        {"col": 0, "row": 3, "counter_index": 0},
        {"col": 1, "row": 3, "counter_index": 1},
        {"col": 2, "row": 3, "counter_index": 2},
    ],
}


def _buf(values):
    return b"".join(struct.pack("<I", v) for v in values)


def test_coefficients_match_hand_computation():
    obs = observe_r3b(_buf([1000, 1010, 1020, 1030]), GEOM)
    # dn_h = |col-s2.col| - |col-s1.col|;  dn_v = |row-s2.row| - |row-s1.row|.
    # tile (1,2): |1-2|-|1-0| = 0;  |2-4|-|2-0| = 0
    # tile (2,2): |2-2|-|2-0| = -2; |2-4|-|2-0| = 0
    # tile (0,2): |0-2|-|0-0| = 2;  |2-4|-|2-0| = 0
    # tile (1,3): |1-2|-|1-0| = 0;  |3-4|-|3-0| = -2
    got = [(o["dn_h"], o["dn_v"]) for o in obs]
    assert got == [(0.0, 0.0), (-2.0, 0.0), (2.0, 0.0), (0.0, -2.0)]
    assert obs[0]["r"] == 1000.0


def test_short_buffer_fails_loud():
    with pytest.raises(ValueError):
        observe_r3b(_buf([1000, 1010, 1020]), GEOM)  # 4 tiles, 3 words


def test_reads_counter_by_index():
    obs = observe_r3b(_buf([7, 8, 9, 10]), GEOM)
    assert [o["r"] for o in obs] == [7.0, 8.0, 9.0, 10.0]


def test_free_flood_routing_key_matches_absent_key():
    # Back-compat: an explicit "routing": "free_flood" must produce identical
    # output to the (default) absent-key path -- the shipped
    # sp5_skew_r3b_pc/geometry.json has no routing key and must keep working
    # unchanged (plan Sec.1b).
    buf = _buf([1000, 1010, 1020, 1030])
    obs_absent = observe_r3b(buf, GEOM)
    obs_explicit = observe_r3b(buf, {**GEOM, "routing": "free_flood"})
    assert obs_absent == obs_explicit


def test_block_shim_row_routing_matches_reset_routed_coeffs():
    obs = observe_r3b(_buf([100, 200, 300]), GEOM_BLOCK)
    s1, s2 = GEOM_BLOCK["sources"]["s1"], GEOM_BLOCK["sources"]["s2"]
    expected = [reset_routed_coeffs(s1, s2, t) for t in GEOM_BLOCK["tiles"]]
    got = [(o["dn_h"], o["dn_v"]) for o in obs]
    assert got == expected
    # Worked example from the plan: dn_h = [2, 0, -2], dn_v constant.
    assert [dn_h for dn_h, _ in got] == [2.0, 0.0, -2.0]
    assert len({dn_v for _, dn_v in got}) == 1


def test_unknown_routing_fails_loud():
    with pytest.raises(ValueError):
        observe_r3b(_buf([1, 2, 3, 4]), {**GEOM, "routing": "some_future_routing"})


def test_observe_then_extract_round_trip():
    # Synthesize counters from a known {d_h, d_v}; observe -> extract recovers it.
    d_h, d_v, const = 2.0, 3.0, 500.0
    hops = {  # (dn_h, dn_v) per tile, matching GEOM order
        0: (0.0, 0.0), 1: (-2.0, 0.0), 2: (2.0, 0.0), 3: (0.0, -2.0),
    }
    vals = [int(round(const + dn_h * d_h + dn_v * d_v)) for dn_h, dn_v in
            (hops[i] for i in range(4))]
    obs = observe_r3b(_buf(vals), GEOM)
    r = extract_r3b(obs)
    assert math.isclose(r["d_h"], d_h, abs_tol=1e-6)
    assert math.isclose(r["d_v"], d_v, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6
