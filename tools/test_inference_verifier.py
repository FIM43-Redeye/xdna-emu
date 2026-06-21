import json
from inference.verifier import (correlates, deterministic, coincident,
                                verify_offset_stable, RejectedRule)


def _ev(col, row, name, soc, slot=0, pkt_type=0, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": slot,
            "name": name, "ts": soc, "soc": soc, "mode": mode}


def _make_runs(tmp_path, per_run_events):
    """per_run_events: list over runs of {event_name(at 1|0|0): anchored_offset}."""
    dirs = []
    for i, off in enumerate(per_run_events):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for name, delta in off.items():
            evs.append(_ev(1, 0, name, 1000 + delta))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def test_correlates_constant_offset(tmp_path):
    # A = S + 50 in every run -> std(A-S) ~ 0 -> correlates, offset 50
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 150}, {"S": 200, "A": 250},
                                 {"S": 300, "A": 350}])
    off = correlates(dirs, "1|0|0|A", "1|0|0|S")
    assert off == 50


def test_correlates_none_when_offset_unstable(tmp_path):
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 150}, {"S": 200, "A": 400},
                                 {"S": 300, "A": 360}])
    assert correlates(dirs, "1|0|0|A", "1|0|0|S") is None


def test_deterministic_true_for_fixed_anchored_ts(tmp_path):
    dirs = _make_runs(tmp_path, [{"D": 40}, {"D": 41}, {"D": 40}])
    assert deterministic(dirs, "1|0|0|D") is True


def test_deterministic_false_for_jittery_event(tmp_path):
    dirs = _make_runs(tmp_path, [{"J": 40}, {"J": 90}, {"J": 140}])
    assert deterministic(dirs, "1|0|0|J") is False


def test_coincident_when_two_events_share_anchored_ts(tmp_path):
    dirs = _make_runs(tmp_path, [{"A": 40, "B": 40}, {"A": 41, "B": 41}])
    assert coincident(dirs, "1|0|0|A", "1|0|0|B") is True


def test_verify_offset_stable_rejects_eps_boundary(tmp_path):
    # stable for 2 runs, breaks on the 3rd -> rejected finding
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 150}, {"S": 200, "A": 250},
                                 {"S": 300, "A": 999}])
    ok, finding = verify_offset_stable(dirs, "1|0|0|A", "1|0|0|S")
    assert ok is False
    assert isinstance(finding, RejectedRule)
    assert "1|0|0|A" in finding.evidence["pair"]
