# tools/test_inference_engine.py
import json
from inference.engine import run_engine


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    dirs = []
    for i, off in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for (tile, name), delta in off.items():
            c, r, p = tile.split("|")
            evs.append(_ev(int(c), int(r), name, 1000 + delta, pkt_type=int(p)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def _ledger(tmp_path, entries):
    p = tmp_path / "led.json"
    p.write_text(json.dumps({"entries": entries}))
    return str(p)


def test_engine_reconstructs_placement(tmp_path):
    dirs = _runs(tmp_path, [
        {("1|0|0", "S"): 100, ("1|0|0", "C"): 130},
        {("1|0|0", "S"): 240, ("1|0|0", "C"): 270},
        {("1|0|0", "S"): 310, ("1|0|0", "C"): 340}])
    led = _ledger(tmp_path, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    rep = run_engine(dirs, led, [("1|0|0|C", "1|0|0|S")])
    assert rep["provenance_ok"] is True
    assert rep["replication_violations"] == []
    # Task 5 changed derives to 4-tuples; the segment is in the new `segments` list.
    assert ("1|0|0|C", "1|0|0|S", 30) in rep["segments"]
    assert "1|0|0|S" in rep["stochastic_roots"]
    assert rep["classification"]["1|0|0|C"] == "derived"


# test_engine_reports_irreducible_cycle was REMOVED (Task 6).
#
# Rationale: Task 5 introduced the ordering falsifier, which rejects any
# backward-in-time derives edge (requires parent.ts <= child.ts). Because
# first-occurrence timestamps are totally ordered across runs, try_derives can
# no longer produce a cycle -- the derives graph is a DAG by construction.
# The original test built a cycle by having try_derives admit BOTH A->B and
# B->A; the falsifier now rejects whichever direction has the later parent,
# so the cycle never forms. Removing the test is correct: it was testing a
# state (cyclic derives) that the new invariant rules out.
#
# condense() coverage is preserved by:
#   test_inference_degeneracy.py::test_condense_collapses_cycle_to_one_group
# which builds a cyclic KB by hand (bypassing the falsifier) and verifies
# condense collapses it correctly.


def test_engine_reports_segment_and_gap(tmp_path):
    # within-domain exact (1|0|0 S->C offset 30) -> segment;
    # cross-domain (shim 1|0|2 MM2S -> core 1|2|0 CORE) -> gap.
    dirs = []
    for i, row in enumerate([
        {"1|0|0|S": 100, "1|0|0|C": 130, "1|0|2|MM2S": 0, "1|2|0|CORE": 40},
        {"1|0|0|S": 200, "1|0|0|C": 230, "1|0|2|MM2S": 9, "1|2|0|CORE": 55}
    ]):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|")
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    led = tmp_path / "led.json"
    led.write_text(json.dumps({"entries": [
        {"cite": "route#1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"},
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}]}))
    rep = run_engine(dirs, str(led),
                     [("1|0|0|C", "1|0|0|S"), ("1|2|0|CORE", "1|0|2|MM2S")])
    assert ("1|0|0|C", "1|0|0|S", 30) in rep["segments"]
    assert ("1|2|0|CORE", "1|0|2|MM2S", None, "cross_domain") in rep["gaps"]
    assert isinstance(rep["rejected_rules"], list)
    assert rep["provenance_ok"] is True


def test_engine_gap_carries_reproduction_offset(tmp_path):
    # cross-domain pair (shim 1|0|2 MM2S -> core 1|2|0 CORE) with an EXACT raw
    # offset (40 every run) -> gap annotated with reproduction_offset=40.
    dirs = []
    for i, row in enumerate([
        {"1|0|2|MM2S": 0, "1|2|0|CORE": 40},
        {"1|0|2|MM2S": 9, "1|2|0|CORE": 49}
    ]):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|")
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    led = tmp_path / "led.json"
    led.write_text(json.dumps({"entries": [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}]}))
    rep = run_engine(dirs, str(led), [("1|2|0|CORE", "1|0|2|MM2S")])
    assert ("1|2|0|CORE", "1|0|2|MM2S", 40, "cross_domain") in rep["gaps"]


def test_engine_warns_on_unaccounted_within_domain_gap(tmp_path):
    # within-domain pair (1|0|0 S->C) whose offset RANGES (30 then 31): it should
    # be cycle-exact but isn't. Unaccounted -> surfaced in `warnings`, NOT silently
    # swallowed as a gap. This is the load-contamination canary.
    dirs = _runs(tmp_path, [
        {("1|0|0", "S"): 100, ("1|0|0", "C"): 130},
        {("1|0|0", "S"): 240, ("1|0|0", "C"): 271}])  # offset 30 then 31 -> range 1
    led = _ledger(tmp_path, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    rep = run_engine(dirs, led, [("1|0|0|C", "1|0|0|S")])
    assert rep["warnings"] == [
        {"child": "1|0|0|C", "parent": "1|0|0|S", "reason": "within_domain_nonexact"}]
    assert ("1|0|0|C", "1|0|0|S", None, "within_domain_nonexact") in rep["gaps"]


def test_engine_does_not_warn_on_accounted_cross_domain_gap(tmp_path):
    # cross-domain gap is structurally accounted: noted in `gaps` with its reason,
    # never warned on.
    dirs = _runs(tmp_path, [
        {("1|0|2", "MM2S"): 0, ("1|2|0", "CORE"): 40},
        {("1|0|2", "MM2S"): 9, ("1|2|0", "CORE"): 49}])
    led = _ledger(tmp_path, [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}])
    rep = run_engine(dirs, led, [("1|2|0|CORE", "1|0|2|MM2S")])
    assert rep["warnings"] == []
    assert ("1|2|0|CORE", "1|0|2|MM2S", 40, "cross_domain") in rep["gaps"]
