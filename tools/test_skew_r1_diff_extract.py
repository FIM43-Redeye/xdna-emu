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


# --- Two-sided vertical-anisotropy falsification (mid-column R1 spine) ---
# R3b (two-source interval) cannot see within-axis direction anisotropy at all
# (test_skew_r3b_identifiability.py). R1 can, IF the source sits mid-column with
# cores on BOTH sides: North tiles then have dn_v>0, South tiles dn_v<0, so a
# genuine d_vN != d_vS becomes a slope kink at the source that a single-d_v fit
# cannot absorb -- fit_residual grows. See the resolved decision (Option 1 +
# R1 reallocation) in NEXT-STEPS.md / the identifiability finding.

def _spine_pair(dn_a, kind_a, dn_b, kind_b, d_vN, d_vS, contrast):
    ci = {"core": 1, "shim": 1, "mem": 0, "memtile": 0}
    def md(dn, k):
        slope = d_vN if dn >= 0 else d_vS  # per-direction hop cost
        return dn * slope + (contrast if ci[k] else 0)
    return {"a": {"dn_v": dn_a, "kind": kind_a},
            "b": {"dn_v": dn_b, "kind": kind_b},
            "skew": float(md(dn_b, kind_b) - md(dn_a, kind_a))}


def test_two_sided_spine_falsifies_vertical_anisotropy():
    # d_vN=3 != d_vS=5, cores both sides of the source: no single slope fits.
    pairs = [
        _spine_pair(1, "core", 2, "core", 3.0, 5.0, -2.0),   # North slope
        _spine_pair(-1, "core", -2, "core", 3.0, 5.0, -2.0),  # South slope
        _spine_pair(1, "core", -1, "core", 3.0, 5.0, -2.0),   # crosses source
        _spine_pair(1, "core", 1, "mem", 3.0, 5.0, -2.0),     # contrast
    ]
    r = extract_r1_diff(pairs)
    assert r["fit_residual"] > 0.5  # anisotropy detected


def test_two_sided_spine_isotropic_zero_residual():
    # Same geometry, d_vN == d_vS: a single slope fits both sides exactly.
    pairs = [
        _spine_pair(1, "core", 2, "core", 4.0, 4.0, -2.0),
        _spine_pair(-1, "core", -2, "core", 4.0, 4.0, -2.0),
        _spine_pair(1, "core", -1, "core", 4.0, 4.0, -2.0),
        _spine_pair(1, "core", 1, "mem", 4.0, 4.0, -2.0),
    ]
    r = extract_r1_diff(pairs)
    assert r["fit_residual"] < 1e-9


def test_one_sided_spine_cannot_falsify_anisotropy():
    # All cores North of the source: d_vS is never sampled, so d_vN != d_vS is
    # invisible (residual 0). This is exactly why the current one-sided R1 spine
    # cannot test vertical isotropy and a mid-column source is required.
    pairs = [
        _spine_pair(1, "core", 2, "core", 3.0, 5.0, -2.0),
        _spine_pair(2, "core", 3, "core", 3.0, 5.0, -2.0),
        _spine_pair(1, "core", 1, "mem", 3.0, 5.0, -2.0),
    ]
    r = extract_r1_diff(pairs)
    assert r["fit_residual"] < 1e-9
