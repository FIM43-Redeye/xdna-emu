"""R1 within-column skew extraction tests (#140 SP-5b)."""
import math
import pytest
from calibration.skew.r1_extract import extract_r1
from calibration.skew._solve import RankDeficientError


# Emulator model (effects.rs:527-532): the intra offset applied to a module's
# origin_D is core_off for {core, shim} and mem_off for {mem, memtile}.
def _obs(dn_v, kind, d_v, intra_core, intra_mem, reflected=True, extra=0.0):
    off = intra_core if kind in ("core", "shim") else intra_mem  # {mem, memtile}
    origin_d = dn_v * d_v + off + extra
    return {"dn_v": dn_v, "kind": kind, "origin": (-origin_d if reflected else origin_d)}


def test_recovers_dv_and_intra():
    d_v, ic, im = 3.0, 2.0, 4.0
    obs = [_obs(n, "core", d_v, ic, im) for n in (2, 3, 4, 5)] + [_obs(1, "mem", d_v, ic, im)]
    r = extract_r1(obs)
    assert math.isclose(r["d_v"], d_v, abs_tol=1e-6)
    assert math.isclose(r["intra_core"], ic, abs_tol=1e-6)
    assert math.isclose(r["intra_mem"], im, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_fit_residual_grows_on_nonuniform_with_three_points():
    d_v, ic, im = 3.0, 2.0, 4.0
    obs = [_obs(2, "core", d_v, ic, im), _obs(3, "core", d_v, ic, im),
           _obs(4, "core", d_v, ic, im, extra=5.0), _obs(1, "mem", d_v, ic, im)]
    r = extract_r1(obs)
    assert r["fit_residual"] > 1.0, "non-uniform per-hop must produce a large residual"


def test_two_points_cannot_falsify():
    # Only 2 core points + 1 mem point: core sub-system is exactly determined,
    # so a non-uniform input still fits with ~0 residual. Documents why >=3.
    d_v, ic, im = 3.0, 2.0, 4.0
    obs = [_obs(2, "core", d_v, ic, im), _obs(4, "core", d_v, ic, im, extra=5.0),
           _obs(1, "mem", d_v, ic, im)]
    r = extract_r1(obs)
    assert r["fit_residual"] < 1e-6, "2 collinear points cannot detect non-uniformity"


def test_sign_convention_reflected_gives_positive_dv():
    # Reflected origins are negative for positive origin_D; extractor must flip.
    obs = [{"dn_v": n, "kind": "core", "origin": -(n * 3.0)} for n in (1, 2, 3)]
    r = extract_r1(obs, reflected=True)
    assert math.isclose(r["d_v"], 3.0, abs_tol=1e-6)


def test_rank_deficient_fails_loud():
    obs = [{"dn_v": 0, "kind": "core", "origin": 0.0} for _ in range(3)]
    with pytest.raises(RankDeficientError):
        extract_r1(obs)


def test_schema_round_trip(tmp_path):
    from calibration.skew.schema import write_constants, read_constants, empty_constants
    assert empty_constants()["intra"] == {"core": None, "mem": None}
    p = tmp_path / "skew_constants.json"
    written = write_constants(p, d_h=1.0, d_v=3.0, intra_core=2.0, intra_mem=4.0,
                              fit_residual=0.0, source_route="r1", provenance="measured-silicon")
    assert read_constants(p) == written
    assert written["intra"] == {"core": 2.0, "mem": 4.0}
