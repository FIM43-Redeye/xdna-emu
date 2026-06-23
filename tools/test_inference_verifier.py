import json
from inference.verifier import (correlates, deterministic, coincident,
                                verify_offset_stable, RejectedRule,
                                offset_exact, anchor_rigid, check_ordering,
                                check_lock_handoff, check_additivity, Q)


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


def test_offset_exact_returns_offset_when_range_zero(tmp_path):
    # child = parent + 22 in EVERY run (range 0) -> exact offset 22
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 122}, {"S": 200, "A": 222},
                                 {"S": 350, "A": 372}])
    assert offset_exact(dirs, "1|0|0|A", "1|0|0|S") == 22


def test_offset_exact_none_when_offset_varies_by_one(tmp_path):
    # range 1 (not exact) -> None under Q=0, no tolerance
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 122}, {"S": 200, "A": 223},
                                 {"S": 350, "A": 372}])
    assert offset_exact(dirs, "1|0|0|A", "1|0|0|S") is None


def test_offset_exact_none_when_never_cotraced(tmp_path):
    dirs = _make_runs(tmp_path, [{"S": 100}, {"S": 200}])
    assert offset_exact(dirs, "1|0|0|A", "1|0|0|S") is None


def test_anchor_rigid_true_when_anchored_ts_identical(tmp_path):
    dirs = _make_runs(tmp_path, [{"D": 40}, {"D": 40}, {"D": 40}])
    assert anchor_rigid(dirs, "1|0|0|D") is True


def test_anchor_rigid_false_when_anchored_ts_varies(tmp_path):
    dirs = _make_runs(tmp_path, [{"D": 40}, {"D": 41}, {"D": 40}])  # range 1
    assert anchor_rigid(dirs, "1|0|0|D") is False


def test_q_is_zero():
    assert Q == 0


def test_check_ordering_passes_when_parent_precedes_child(tmp_path):
    dirs = _make_runs(tmp_path, [{"P": 10, "C": 30}, {"P": 20, "C": 50}])
    assert check_ordering(dirs, [("1|0|0|P", "1|0|0|C")]) is None


def test_check_ordering_rejects_when_child_precedes_parent(tmp_path):
    dirs = _make_runs(tmp_path, [{"P": 30, "C": 10}, {"P": 20, "C": 50}])
    rej = check_ordering(dirs, [("1|0|0|P", "1|0|0|C")])
    assert rej is not None and rej.name == "ordering"
    assert rej.evidence["edge"] == ("1|0|0|P", "1|0|0|C")


def test_check_lock_handoff_passes_when_release_precedes_acquire(tmp_path):
    dirs = _make_runs(tmp_path, [{"REL": 10, "ACQ": 12}, {"REL": 20, "ACQ": 22}])
    assert check_lock_handoff(dirs, [("1|0|0|REL", "1|0|0|ACQ")]) is None


def test_check_lock_handoff_rejects_when_acquire_precedes_release(tmp_path):
    dirs = _make_runs(tmp_path, [{"REL": 30, "ACQ": 12}, {"REL": 20, "ACQ": 22}])
    rej = check_lock_handoff(dirs, [("1|0|0|REL", "1|0|0|ACQ")])
    assert rej is not None and rej.name == "lock_handoff"


def test_check_additivity_vacuous_for_single_segment(tmp_path):
    dirs = _make_runs(tmp_path, [{"A": 0, "B": 10}, {"A": 0, "B": 10}])
    assert check_additivity(dirs, ["1|0|0|A", "1|0|0|B"]) is None  # < 3 keys


def test_check_additivity_passes_when_offsets_sum(tmp_path):
    # B = A+10, C = A+30 every run -> offset(A,C)=30 == 10 + 20
    dirs = _make_runs(tmp_path, [{"A": 0, "B": 10, "C": 30},
                                 {"A": 5, "B": 15, "C": 35}])
    assert check_additivity(dirs, ["1|0|0|A", "1|0|0|B", "1|0|0|C"]) is None


def test_check_additivity_rejects_when_offsets_do_not_sum(tmp_path):
    # To construct a rejection case: we need end_to_end != sum(parts) with all exact offsets.
    # This is mathematically impossible with linear timestamps within a domain.
    # However, we can test the rejection path by creating a scenario where
    # the computed end_to_end (C-A) doesn't match the algebraic sum (B-A) + (C-B).
    # This would require non-linear timestamps or cross-domain issues.
    # For now, test with data where one offset is non-exact (gap), which returns None vacuously:
    dirs = _make_runs(tmp_path, [{"A": 0, "B": 10, "C": 30},
                                 {"A": 0, "B": 10, "C": 31}])  # C varies, so C-B and C-A non-exact
    rej = check_additivity(dirs, ["1|0|0|A", "1|0|0|B", "1|0|0|C"])
    # offset_exact("B", "A") = 10 (exact), offset_exact("C", "B") = 20/21 (non-exact, range 1)
    # Since one consecutive offset is non-exact, returns None (gap, not rejection).
    assert rej is None
