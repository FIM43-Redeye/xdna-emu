"""R3b enriched interval-difference skew extraction tests (#140 SP-5b, rev3).

Enriched model (per kernel spec rev3 Sec.5.1/5.2): the interval decomposes into
signed per-direction hop coefficients so anisotropy (d_hE!=d_hW, d_vN!=d_vS) and
a turn term (d_turn) are identifiable — the falsification apparatus the soundness
audit requires.
"""
import math
import pytest
from calibration.skew.r3b_extract import extract_r3b
from calibration.skew._solve import RankDeficientError

PARAMS5 = ("d_hE", "d_hW", "d_vN", "d_vS", "d_turn")
PARAMS4 = ("d_hE", "d_hW", "d_vN", "d_vS")


def _obs(a_hE, a_hW, a_vN, a_vS, a_turn, truth, const):
    """Synthesize a reading from coefficients and a truth param dict."""
    r = const + (a_hE * truth["d_hE"] + a_hW * truth["d_hW"]
                 + a_vN * truth["d_vN"] + a_vS * truth["d_vS"]
                 + a_turn * truth["d_turn"])
    return {"a_hE": a_hE, "a_hW": a_hW, "a_vN": a_vN, "a_vS": a_vS,
            "a_turn": a_turn, "r": r}


def _two_sided_isotropic_geometry(truth, const):
    """Ref + 2 East + 2 West + 2 North + 2 South + 1 diagonal. Populates all
    five columns with rank 5 after ref-differencing."""
    return [
        _obs(0, 0, 0, 0, 0, truth, const),   # reference
        _obs(1, 0, 0, 0, 0, truth, const),   # East 1
        _obs(2, 0, 0, 0, 0, truth, const),   # East 2
        _obs(0, 1, 0, 0, 0, truth, const),   # West 1
        _obs(0, 2, 0, 0, 0, truth, const),   # West 2
        _obs(0, 0, 1, 0, 0, truth, const),   # North 1
        _obs(0, 0, 2, 0, 0, truth, const),   # North 2
        _obs(0, 0, 0, 1, 0, truth, const),   # South 1
        _obs(0, 0, 0, 2, 0, truth, const),   # South 2
        _obs(1, 0, 1, 0, 1, truth, const),   # diagonal (E1,N1, turn=1)
    ]


def test_recovers_all_five_params():
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    obs = _two_sided_isotropic_geometry(truth, const=100.0)
    r = extract_r3b(obs, params=PARAMS5)
    for k in PARAMS5:
        assert math.isclose(r[k], truth[k], abs_tol=1e-6), k
    assert math.isclose(r["d_h"], 2.0, abs_tol=1e-6)
    assert math.isclose(r["d_v"], 3.0, abs_tol=1e-6)
    assert math.isclose(r["aniso_h"], 0.0, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_const_cancels():
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    ra = extract_r3b(_two_sided_isotropic_geometry(truth, const=100.0), params=PARAMS5)
    rb = extract_r3b(_two_sided_isotropic_geometry(truth, const=999.0), params=PARAMS5)
    for k in PARAMS5:
        assert math.isclose(ra[k], rb[k], abs_tol=1e-6), k


def test_exposes_horizontal_anisotropy():
    # Truth has d_hE != d_hW. The enriched fit RECOVERS the split (aniso_h != 0)
    # at zero residual — the apparatus can SEE anisotropy the old scalar hid.
    truth = {"d_hE": 2.0, "d_hW": 5.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    r = extract_r3b(_two_sided_isotropic_geometry(truth, const=0.0), params=PARAMS5)
    assert math.isclose(r["aniso_h"], -3.0, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_reduced_isotropic_fit_residual_grows_under_anisotropy():
    # If a caller fits the ASSUMED isotropic-additive shape (collapse E/W and N/S,
    # drop turn) against an anisotropic truth, the residual fires — the
    # falsification the audit requires.
    truth = {"d_hE": 2.0, "d_hW": 5.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    obs = _two_sided_isotropic_geometry(truth, const=0.0)
    # Collapse to an isotropic design: a_h = a_hE - a_hW, a_v = a_vN - a_vS.
    iso = [{"a_hE": o["a_hE"] - o["a_hW"], "a_hW": 0.0,
            "a_vN": o["a_vN"] - o["a_vS"], "a_vS": 0.0, "a_turn": 0.0,
            "r": o["r"]} for o in obs]
    r = extract_r3b(iso, params=("d_hE", "d_vN"))
    assert r["fit_residual"] > 1.0


def test_exposes_turn_term():
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 7.0}
    r = extract_r3b(_two_sided_isotropic_geometry(truth, const=0.0), params=PARAMS5)
    assert math.isclose(r["d_turn"], 7.0, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_one_sided_geometry_fails_loud():
    # All tiles East/North of the sole source -> d_hW, d_vS columns all-zero ->
    # rank-deficient -> fail loud. One-sided geometry cannot see anisotropy.
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    obs = [_obs(0, 0, 0, 0, 0, truth, 0.0),
           _obs(1, 0, 0, 0, 0, truth, 0.0),
           _obs(2, 0, 0, 0, 0, truth, 0.0),
           _obs(0, 0, 1, 0, 0, truth, 0.0),
           _obs(0, 0, 2, 0, 0, truth, 0.0)]
    with pytest.raises(RankDeficientError):
        extract_r3b(obs, params=PARAMS5)


def test_no_diagonal_cannot_request_turn():
    # Axis-collinear only -> a_turn all-zero -> requesting d_turn fails loud.
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    obs = _two_sided_isotropic_geometry(truth, const=0.0)[:-1]  # drop diagonal
    with pytest.raises(RankDeficientError):
        extract_r3b(obs, params=PARAMS5)
    # But the 4-param fit (no turn) succeeds on the same geometry.
    r = extract_r3b(obs, params=PARAMS4)
    assert r["fit_residual"] < 1e-6
