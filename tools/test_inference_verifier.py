import json
from inference.verifier import (coincident, RejectedRule,
                                offset_exact, anchor_rigid, check_ordering,
                                check_lock_handoff, check_additivity, Q,
                                anchored_occurrences_per_run,
                                offset_window, additivity_state, cross_batch_range)


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


def _make_multibatch_run(tmp_path, run_name, batches):
    """batches: {batch_name: {event_name: anchored_offset}}. Each batch carries
    its own PERF_CNT_2 anchor (1|2|0) so anchored_firsts works; events at 1|0|0.
    Lets a pair co-trace in a chosen batch only -> tests cross-batch additivity."""
    rd = tmp_path / run_name
    for bn, evmap in batches.items():
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for name, delta in evmap.items():
            evs.append(_ev(1, 0, name, 1000 + delta))
        (rd / bn / "hw").mkdir(parents=True)
        (rd / bn / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
    return str(rd)


def test_coincident_true_when_offset_exactly_zero(tmp_path):
    dirs = _make_runs(tmp_path, [{"A": 40, "B": 40}, {"A": 90, "B": 90}])
    assert coincident(dirs, "1|0|0|A", "1|0|0|B") is True


def test_coincident_false_when_offset_nonzero_or_jittery(tmp_path):
    dirs = _make_runs(tmp_path, [{"A": 40, "B": 41}, {"A": 90, "B": 90}])
    assert coincident(dirs, "1|0|0|A", "1|0|0|B") is False


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
    assert rej.evidence["pair"] == ("1|0|0|REL", "1|0|0|ACQ")


def test_check_additivity_vacuous_for_single_segment(tmp_path):
    dirs = _make_runs(tmp_path, [{"A": 0, "B": 10}, {"A": 0, "B": 10}])
    assert check_additivity(dirs, ["1|0|0|A", "1|0|0|B"]) is None  # < 3 keys


def test_check_additivity_passes_when_offsets_sum(tmp_path):
    # B = A+10, C = A+30 every run -> offset(A,C)=30 == 10 + 20
    dirs = _make_runs(tmp_path, [{"A": 0, "B": 10, "C": 30},
                                 {"A": 5, "B": 15, "C": 35}])
    assert check_additivity(dirs, ["1|0|0|A", "1|0|0|B", "1|0|0|C"]) is None


def test_check_additivity_rejects_when_offsets_do_not_sum(tmp_path):
    # Cross-batch: A&B co-fire only in batch_00 (B-A=10); B&C only in batch_01
    # (C-B=20); A&C only in batch_02 (C-A=999). 10+20 != 999 -> reject. Each pair
    # is range-0 exact across the two runs, so the gate is the additivity sum.
    r0 = _make_multibatch_run(tmp_path, "run0", {
        "batch_00": {"A": 0, "B": 10},
        "batch_01": {"B": 0, "C": 20},
        "batch_02": {"A": 0, "C": 999}})
    r1 = _make_multibatch_run(tmp_path, "run1", {
        "batch_00": {"A": 5, "B": 15},
        "batch_01": {"B": 3, "C": 23},
        "batch_02": {"A": 2, "C": 1001}})
    rej = check_additivity([r0, r1], ["1|0|0|A", "1|0|0|B", "1|0|0|C"])
    assert rej is not None and rej.name == "additivity"


def test_check_additivity_none_when_chain_has_gap(tmp_path):
    # One consecutive offset is non-exact (gap) -> returns None (not a rejection).
    # A at 0, B at 10 (exact); B at 0, C at 20/21 (non-exact, range 1).
    dirs = _make_runs(tmp_path, [{"A": 0, "B": 10, "C": 30},
                                 {"A": 0, "B": 10, "C": 31}])
    rej = check_additivity(dirs, ["1|0|0|A", "1|0|0|B", "1|0|0|C"])
    # offset_exact("B", "A") = 10 (exact), offset_exact("C", "B") = 20/21 (non-exact, range 1)
    # Since one consecutive offset is non-exact, returns None (gap, not rejection).
    assert rej is None


def test_anchored_occurrences_per_run_reads_pinned_batch(tmp_path):
    for i, base in enumerate((1000, 2000)):
        rd = tmp_path / f"run_{i:02d}"
        p = rd / "batch_00" / "hw"; p.mkdir(parents=True)
        (p / "trace.events.json").write_text(__import__("json").dumps({"events": [
            {"col":1,"row":2,"pkt_type":0,"name":"PERF_CNT_2","soc":base,"slot":0},
            {"col":1,"row":1,"pkt_type":3,"name":"PORT_RUNNING_0","soc":base+10,"slot":0},
            {"col":1,"row":1,"pkt_type":3,"name":"PORT_RUNNING_0","soc":base+26,"slot":0},
        ]}))
    runs = [str(tmp_path / "run_00"), str(tmp_path / "run_01")]
    got = anchored_occurrences_per_run(runs, "1|1|3|PORT_RUNNING_0", "batch_00")
    assert got == [[10, 26], [10, 26]]


# ---------------------------------------------------------------------------
# _runs helper: takes full-key dicts {col|row|pkt|name: anchored_offset}
# (matches test_canary_witness.py pattern)
# ---------------------------------------------------------------------------

def _runs(base, rows):
    """rows: list over runs of {full_key: anchored_offset}. Writes the
    run_NN/batch_00/hw/trace.events.json layout the engine reads.
    `base` should be a unique tmp_path subdir to avoid path/cache collisions."""
    dirs = []
    for i, row in enumerate(rows):
        rd = base / f"run_{i:02d}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|", 3)
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


# ---------------------------------------------------------------------------
# offset_window tests
# ---------------------------------------------------------------------------

def test_offset_window_min_max(tmp_path):
    # B-A = 10 in run0, 13 in run1 -> window (10, 13).
    dirs = _runs(tmp_path / "r", [{"1|0|0|A": 0, "1|0|0|B": 10},
                                   {"1|0|0|A": 0, "1|0|0|B": 13}])
    assert offset_window(dirs, "1|0|0|B", "1|0|0|A") == (10, 13)


def test_offset_window_none_when_not_co_traced(tmp_path):
    dirs = _runs(tmp_path / "r", [{"1|0|0|A": 0}])  # B never present
    assert offset_window(dirs, "1|0|0|B", "1|0|0|A") is None


# ---------------------------------------------------------------------------
# additivity_state tests
# ---------------------------------------------------------------------------

def test_additivity_state_pass_vacuous_unverifiable_violation(tmp_path):
    # pass: B-A=5, C-B=7, C-A=12; 5+7==12
    ok = _runs(tmp_path / "ok", [{"1|0|0|A": 0, "1|0|0|B": 5, "1|0|0|C": 12}])
    assert additivity_state(ok, ["1|0|0|A", "1|0|0|B", "1|0|0|C"]) == "pass"
    # vacuous: chain < 3 keys
    assert additivity_state(ok, ["1|0|0|A", "1|0|0|B"]) == "vacuous"
    # unverifiable: B missing -> consecutive offset A-B not co-traced
    miss = _runs(tmp_path / "miss", [{"1|0|0|A": 0, "1|0|0|C": 12}])
    assert additivity_state(miss, ["1|0|0|A", "1|0|0|B", "1|0|0|C"]) == "unverifiable"
    # violation: cross-batch setup (arithmetic identity prevents single-batch violation)
    # Mirrors test_check_additivity_rejects_when_offsets_do_not_sum exactly.
    r0 = _make_multibatch_run(tmp_path, "run0", {
        "batch_00": {"A": 0, "B": 10},
        "batch_01": {"B": 0, "C": 20},
        "batch_02": {"A": 0, "C": 999}})
    r1 = _make_multibatch_run(tmp_path, "run1", {
        "batch_00": {"A": 5, "B": 15},
        "batch_01": {"B": 3, "C": 23},
        "batch_02": {"A": 2, "C": 1001}})
    assert additivity_state([r0, r1], ["1|0|0|A", "1|0|0|B", "1|0|0|C"]) == "violation"


# ---------------------------------------------------------------------------
# cross_batch_range tests
# ---------------------------------------------------------------------------

def test_cross_batch_range_invariant(tmp_path):
    # event_key in two batches with the same anchored value -> range 0
    rd = _make_multibatch_run(tmp_path, "run0", {
        "batch_00": {"E": 10},
        "batch_01": {"E": 10}})
    assert cross_batch_range([rd], "1|0|0|E") == 0


def test_cross_batch_range_differs_by_two(tmp_path):
    # event_key in batch_00 with delta 10, batch_01 with delta 12 -> range 2
    rd = _make_multibatch_run(tmp_path, "run0", {
        "batch_00": {"E": 10},
        "batch_01": {"E": 12}})
    assert cross_batch_range([rd], "1|0|0|E") == 2
