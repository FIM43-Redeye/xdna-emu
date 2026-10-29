"""R3b interval-difference skew extraction tests (#140 SP-5b/SP-5c).

The extract_r3b_dh tests (SP-5c Phase 2, plan Sec.1c) exercise the
block_shim_row capture path: dn_v is constant across a block-routed spine (see
reset_routed_coeffs / test_block_routed_capture_is_rank_deficient_for_dh_dv in
test_skew_r3b_identifiability.py), so the rank-2 extract_r3b must keep
raising RankDeficientError on that data -- that is a deliberate guard, not a
bug -- while the new single-column extract_r3b_dh recovers d_h cleanly."""
import math
import pytest
from calibration.skew.r3b_extract import extract_r3b, extract_r3b_dh
from calibration.skew._solve import RankDeficientError


def _r(dn_h, dn_v, d_h, d_v, const, extra=0.0):
    return {"dn_h": dn_h, "dn_v": dn_v, "r": const + dn_h * d_h + dn_v * d_v + extra}


def test_recovers_dh_dv():
    d_h, d_v, const = 2.0, 3.0, 100.0
    obs = [_r(0, 0, d_h, d_v, const), _r(1, 0, d_h, d_v, const), _r(2, 0, d_h, d_v, const),
           _r(0, 1, d_h, d_v, const), _r(0, 2, d_h, d_v, const)]
    r = extract_r3b(obs)
    assert math.isclose(r["d_h"], d_h, abs_tol=1e-6)
    assert math.isclose(r["d_v"], d_v, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_const_cancels():
    d_h, d_v = 2.0, 3.0
    obs_a = [_r(0, 0, d_h, d_v, 100.0), _r(1, 0, d_h, d_v, 100.0), _r(0, 1, d_h, d_v, 100.0)]
    obs_b = [_r(0, 0, d_h, d_v, 999.0), _r(1, 0, d_h, d_v, 999.0), _r(0, 1, d_h, d_v, 999.0)]
    ra, rb = extract_r3b(obs_a), extract_r3b(obs_b)
    assert math.isclose(ra["d_h"], rb["d_h"], abs_tol=1e-6)
    assert math.isclose(ra["d_v"], rb["d_v"], abs_tol=1e-6)


def test_fit_residual_grows_on_nonuniform_with_three_points():
    d_h, d_v, const = 2.0, 3.0, 0.0
    obs = [_r(0, 0, d_h, d_v, const), _r(1, 0, d_h, d_v, const), _r(2, 0, d_h, d_v, const, extra=5.0),
           _r(0, 1, d_h, d_v, const), _r(0, 2, d_h, d_v, const)]
    r = extract_r3b(obs)
    assert r["fit_residual"] > 1.0


def test_rank_deficient_fails_loud():
    # All tiles on the vertical axis only -> d_h unidentifiable.
    obs = [{"dn_h": 0, "dn_v": n, "r": float(n)} for n in range(3)]
    with pytest.raises(RankDeficientError):
        extract_r3b(obs)


def _block_routed_obs(d_h, const, extra=None):
    # dn_h = [2, 0, -2] (plan Sec.1b worked example), dn_v constant = 5 (s2.row -
    # s1.row for s1=shim(0,0)/s2=core(2,5)) -- collapses to zero signal after
    # reference-differencing, matching reset_routed_coeffs.
    dn_hs = [2.0, 0.0, -2.0]
    dn_v = 5.0
    extra = extra or [0.0, 0.0, 0.0]
    return [{"dn_h": dn_h_i, "dn_v": dn_v, "r": const + dn_h_i * d_h + e}
            for dn_h_i, e in zip(dn_hs, extra)]


def test_extract_r3b_dh_recovers_known_dh():
    d_h_true, const = 1.5, 100.0
    obs = _block_routed_obs(d_h_true, const)
    r = extract_r3b_dh(obs)
    assert math.isclose(r["d_h"], d_h_true, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_extract_r3b_dh_ignores_constant_dn_v_offset():
    # Changing const (which folds in whatever d_v contributes on a block-routed
    # capture, since dn_v is constant) must not perturb the recovered d_h.
    d_h_true = -0.75
    r_a = extract_r3b_dh(_block_routed_obs(d_h_true, const=10.0))
    r_b = extract_r3b_dh(_block_routed_obs(d_h_true, const=-500.0))
    assert math.isclose(r_a["d_h"], r_b["d_h"], abs_tol=1e-6)
    assert math.isclose(r_a["d_h"], d_h_true, abs_tol=1e-6)


def test_extract_r3b_rank2_rejects_block_routed_capture():
    # The rank-2 {d_h, d_v} fit must keep raising RankDeficientError on a
    # block-routed capture: dn_v is constant, so after reference-differencing
    # the dn_v column is all-zero and rank(A) == 1 < min_rank=2. This is the
    # deliberate guard extract_r3b_dh exists to route around correctly, not a
    # defect to "fix" by loosening extract_r3b.
    obs = _block_routed_obs(d_h=2.0, const=100.0)
    with pytest.raises(RankDeficientError):
        extract_r3b(obs)


def test_two_points_per_axis_cannot_falsify():
    # ref + one h-axis point + one v-axis point: after differencing against ref,
    # 2 rows exactly determine (d_h, d_v), so a corrupted reading is absorbed
    # with ~0 residual. Documents why >=3 collinear tiles per axis are needed to
    # falsify per-hop non-uniformity (the const-differencing removes one DOF).
    d_h, d_v, const = 2.0, 3.0, 0.0
    obs = [_r(0, 0, d_h, d_v, const),
           _r(1, 0, d_h, d_v, const, extra=5.0),
           _r(0, 1, d_h, d_v, const)]
    r = extract_r3b(obs)
    assert r["fit_residual"] < 1e-6, "2 difference rows cannot detect non-uniformity"
