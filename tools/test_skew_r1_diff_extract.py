import math
import pytest
from calibration.skew.r1_diff_extract import extract_r1_diff
from calibration.skew._solve import RankDeficientError


def _pair(dn_a, kind_a, dn_b, kind_b, d_v, contrast):
    # contrast = core_off - mem_off; core_ind = 1 for core/shim else 0.
    ci = {"core": 1, "shim": 1, "mem": 0, "memtile": 0}
    md = lambda dn, k: dn * d_v + (contrast if ci[k] else 0)  # mem_off gauge = 0
    return {"a": {"dn_v": dn_a, "kind": kind_a},
            "b": {"dn_v": dn_b, "kind": kind_b},
            "skew": float(md(dn_b, kind_b) - md(dn_a, kind_a))}


def test_recovers_injected_dv_and_contrast():
    d_v, contrast = 3.0, -2.0
    pairs = [
        _pair(2, "core", 3, "core", d_v, contrast),   # d_v hop
        _pair(3, "core", 4, "core", d_v, contrast),   # d_v hop
        _pair(2, "core", 2, "mem", d_v, contrast),    # contrast (same tile)
    ]
    r = extract_r1_diff(pairs)
    assert math.isclose(r["d_v"], d_v, abs_tol=1e-9)
    assert math.isclose(r["intra_contrast"], contrast, abs_tol=1e-9)
    assert r["fit_residual"] < 1e-9


def test_fit_residual_grows_on_nonuniform_hops():
    # 3 collinear core points, but the 2->3 hop differs from 3->4 (non-uniform d_v).
    pairs = [
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 3, "kind": "core"}, "skew": 3.0},
        {"a": {"dn_v": 3, "kind": "core"}, "b": {"dn_v": 4, "kind": "core"}, "skew": 5.0},
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 4, "kind": "core"}, "skew": 8.0},
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 2, "kind": "mem"}, "skew": -2.0},
    ]
    r = extract_r1_diff(pairs)
    assert r["fit_residual"] > 1.0  # linear model cannot fit non-uniform hops


def test_two_points_per_axis_cannot_falsify():
    # Only 2 distinct dn_v (one hop) + one contrast pair: fits with ~0 residual.
    pairs = [
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 3, "kind": "core"}, "skew": 3.0},
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 2, "kind": "mem"}, "skew": -2.0},
    ]
    r = extract_r1_diff(pairs)
    assert r["fit_residual"] < 1e-9  # 2 points fit any line -- no falsification power


def test_rank_deficient_all_core_raises():
    # All-core pairs: contrast column is all-zero -> rank 1 < min_rank 2.
    pairs = [
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 3, "kind": "core"}, "skew": 3.0},
        {"a": {"dn_v": 3, "kind": "core"}, "b": {"dn_v": 4, "kind": "core"}, "skew": 3.0},
    ]
    with pytest.raises(RankDeficientError):
        extract_r1_diff(pairs)


def test_unknown_kind_raises():
    pairs = [{"a": {"dn_v": 2, "kind": "core"},
              "b": {"dn_v": 3, "kind": "bogus"}, "skew": 1.0}]
    with pytest.raises(ValueError):
        extract_r1_diff(pairs)


def test_sign_pin_contrast_negative():
    # core is EARLIER than mem (core_off < mem_off) -> contrast negative.
    # same tile: skew = md(mem) - md(core) = mem_off - core_off = -contrast.
    pairs = [
        _pair(2, "core", 3, "core", 3.0, -2.0),
        _pair(3, "core", 4, "core", 3.0, -2.0),
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 2, "kind": "mem"}, "skew": 2.0},
    ]
    r = extract_r1_diff(pairs)
    assert r["intra_contrast"] < 0
