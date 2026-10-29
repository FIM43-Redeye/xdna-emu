import pytest
from calibration.skew.r1_observe import observe_r1


def _ev(col, row, pkt, name, soc):
    return {"col": col, "row": row, "pkt_type": pkt, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


_GEOM = {
    "pairs": [
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 2},
         "b": {"col": 0, "row": 3, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 3}},
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 2},
         "b": {"col": 0, "row": 2, "pkt_type": 1, "name": "PORT_RUNNING_0", "dn_v": 2}},
    ]
}


def test_delta_wall_subtraction_isolates_skew():
    # dwall run: pure execution timing (skew=0). measured: dwall + injected skew.
    dwall = [_ev(0, 2, 0, "LOCK_STALL", 100), _ev(0, 3, 0, "LOCK_STALL", 90),
             _ev(0, 2, 1, "PORT_RUNNING_0", 200)]
    # inject: core(0,3) later by 3 vs core(0,2); mem(0,2) later by 2 vs core(0,2).
    measured = [_ev(0, 2, 0, "LOCK_STALL", 100), _ev(0, 3, 0, "LOCK_STALL", 93),
                _ev(0, 2, 1, "PORT_RUNNING_0", 202)]
    obs = observe_r1(measured, dwall, _GEOM)
    # pair 0: skew = (100-93) - (100-90) = -3  (md(b=core@3) - md(a=core@2))
    assert obs[0]["skew"] == pytest.approx(-3.0)
    assert obs[0]["a"]["kind"] == "core" and obs[0]["b"]["kind"] == "core"
    # pair 1: skew = (100-202) - (100-200) = -2  (md(mem@2) - md(core@2))
    assert obs[1]["skew"] == pytest.approx(-2.0)
    assert obs[1]["b"]["kind"] == "mem"


def test_uses_first_occurrence():
    dwall = [_ev(0, 2, 0, "LOCK_STALL", 100), _ev(0, 3, 0, "LOCK_STALL", 100),
             _ev(0, 2, 1, "PORT_RUNNING_0", 100)]
    measured = [_ev(0, 2, 0, "LOCK_STALL", 50), _ev(0, 2, 0, "LOCK_STALL", 999),
                _ev(0, 3, 0, "LOCK_STALL", 50), _ev(0, 2, 1, "PORT_RUNNING_0", 50)]
    obs = observe_r1(measured, dwall, _GEOM)
    assert obs[0]["skew"] == pytest.approx(0.0)  # first-occurrence 50, not 999


def test_missing_anchor_raises():
    dwall = [_ev(0, 2, 0, "LOCK_STALL", 100)]
    measured = [_ev(0, 2, 0, "LOCK_STALL", 100)]
    with pytest.raises(KeyError):
        observe_r1(measured, dwall, _GEOM)  # core(0,3) anchor absent


def test_unknown_pkt_type_raises():
    geom = {"pairs": [
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "E", "dn_v": 2},
         "b": {"col": 0, "row": 2, "pkt_type": 9, "name": "E", "dn_v": 2}}]}
    ev = [_ev(0, 2, 0, "E", 10), _ev(0, 2, 9, "E", 10)]
    with pytest.raises(ValueError):
        observe_r1(ev, ev, geom)
