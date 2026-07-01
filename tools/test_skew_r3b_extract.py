"""R3b interval-difference skew extraction tests (#140 SP-5b)."""
import math
import pytest
from calibration.skew.r3b_extract import extract_r3b
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
