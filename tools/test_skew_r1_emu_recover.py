import json, math
from calibration.skew.r1_emu_recover import recover_and_check


def _ev(col, row, pkt, name, soc):
    return {"col": col, "row": row, "pkt_type": pkt, "name": name, "soc": soc}


def test_recover_matches_injected(tmp_path):
    geom = {"pairs": [
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "L", "dn_v": 2},
         "b": {"col": 0, "row": 3, "pkt_type": 0, "name": "L", "dn_v": 3}},
        {"a": {"col": 0, "row": 3, "pkt_type": 0, "name": "L", "dn_v": 3},
         "b": {"col": 0, "row": 4, "pkt_type": 0, "name": "L", "dn_v": 4}},
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "L", "dn_v": 2},
         "b": {"col": 0, "row": 2, "pkt_type": 1, "name": "P", "dn_v": 2}}]}
    dwall = [_ev(0, 2, 0, "L", 100), _ev(0, 3, 0, "L", 100),
             _ev(0, 4, 0, "L", 100), _ev(0, 2, 1, "P", 100)]
    d_v, contrast = 3, -2
    # Build measured so observe skew = md(b)-md(a): core pairs -> d_v per hop,
    # the same-tile core->mem pair -> -contrast (= mem_off - core_off).
    # core@r soc = 100 - (r-2)*d_v ; mem@2 soc = 100 + contrast.
    meas = [_ev(0, 2, 0, "L", 100), _ev(0, 3, 0, "L", 100 - 1 * d_v),
            _ev(0, 4, 0, "L", 100 - 2 * d_v), _ev(0, 2, 1, "P", 100 + contrast)]
    ok, got = recover_and_check(meas, dwall, geom,
                                expect_d_v=d_v, expect_contrast=contrast)
    assert ok, got


def test_mismatch_flags(tmp_path):
    geom = {"pairs": [
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "L", "dn_v": 2},
         "b": {"col": 0, "row": 3, "pkt_type": 0, "name": "L", "dn_v": 3}},
        {"a": {"col": 0, "row": 3, "pkt_type": 0, "name": "L", "dn_v": 3},
         "b": {"col": 0, "row": 4, "pkt_type": 0, "name": "L", "dn_v": 4}},
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "L", "dn_v": 2},
         "b": {"col": 0, "row": 2, "pkt_type": 1, "name": "P", "dn_v": 2}}]}
    ev = [_ev(0, 2, 0, "L", 100), _ev(0, 3, 0, "L", 100),
          _ev(0, 4, 0, "L", 100), _ev(0, 2, 1, "P", 100)]
    ok, _ = recover_and_check(ev, ev, geom, expect_d_v=3, expect_contrast=-2)
    assert not ok  # recovered d_v=0,contrast=0 != injected
