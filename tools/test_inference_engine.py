# tools/test_inference_engine.py
import json
from inference.engine import run_engine
from inference.timeline import CrossTrackEdge, IntegratedTimeline
from config_extract.dump_model import ConfigDump, RouteGraph, RouteEdge, PortRef


def _pr(col, row, dir="out"):
    return PortRef(col=col, row=row, port=0, dir=dir, kind="x")


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


def test_engine_report_includes_timeline(tmp_path):
    # Same fixture as test_engine_reconstructs_placement; asserts that the engine
    # report now carries a "timeline" key holding a fully-assembled IntegratedTimeline.
    dirs = _runs(tmp_path, [
        {("1|0|0", "S"): 100, ("1|0|0", "C"): 130},
        {("1|0|0", "S"): 240, ("1|0|0", "C"): 270},
        {("1|0|0", "S"): 310, ("1|0|0", "C"): 340}])
    led = _ledger(tmp_path, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    rep = run_engine(dirs, led, [("1|0|0|C", "1|0|0|S")])
    assert "timeline" in rep
    assert isinstance(rep["timeline"], IntegratedTimeline)


def test_engine_timeline_has_cross_track_edge_for_cross_domain_pair(tmp_path):
    # Exercises the cross-domain filter path: run_engine -> assemble_timeline -> weave.
    #
    # The candidate pair ("1|2|0|CORE", "1|0|2|MM2S") is cross-domain because
    # same_domain checks the col|row|pkt_type prefix and "1|2|0" != "1|0|2".
    # With the correct filter (`if not same_domain`), the pair reaches weave() and
    # ground_edge() returns a Gap(reason="cross_domain"), which becomes a CrossTrackEdge.
    #
    # An INVERTED filter (`if same_domain` instead of `if not same_domain`) would
    # drop the cross-domain pair before weave(), so no CrossTrackEdge would form and
    # the assertion below would fail -- catching the inversion.
    dirs = _runs(tmp_path, [
        {("1|0|2", "MM2S"): 0,  ("1|2|0", "CORE"): 40},
        {("1|0|2", "MM2S"): 9,  ("1|2|0", "CORE"): 49}])
    led = _ledger(tmp_path, [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}])
    rep = run_engine(dirs, led, [("1|2|0|CORE", "1|0|2|MM2S")])
    tl = rep["timeline"]
    assert isinstance(tl, IntegratedTimeline)
    # Exactly one CrossTrackEdge for our pair must exist.
    cross_edges = [e for e in tl.cross_track_edges
                   if e.child == "1|2|0|CORE" and e.parent == "1|0|2|MM2S"]
    assert cross_edges, (
        f"Expected a CrossTrackEdge for the cross-domain pair; "
        f"got cross_track_edges={tl.cross_track_edges}")
    assert isinstance(cross_edges[0], CrossTrackEdge)
    assert cross_edges[0].reason == "cross_domain"


def _cross_domain_dirs(tmp_path):
    return _runs(tmp_path, [
        {("1|0|2", "MM2S"): 0, ("1|2|0", "CORE"): 40},
        {("1|0|2", "MM2S"): 9, ("1|2|0", "CORE"): 49}])


def test_engine_dump_connectivity_oracle_no_defect(tmp_path):
    # With a dump threaded in, the connectivity oracle runs in PROD (not just
    # tests). Here the route edge couples tiles (abs) 1|0 and 1|2 -- exactly the
    # pair the cross-domain weave (MM2S<-CORE) connects -> NO connectivity defect.
    dirs = _cross_domain_dirs(tmp_path)
    led = _ledger(tmp_path, [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}])
    rg = RouteGraph(edges=(RouteEdge(_pr(0, 0), _pr(0, 2, "in"), "inter_tile"),))
    dump = ConfigDump(device="npu1", route_graph=rg, tiles=())
    rep = run_engine(dirs, led, [("1|2|0|CORE", "1|0|2|MM2S")],
                     dump=dump, start_col=1)
    tl = rep["timeline"]
    assert isinstance(tl, IntegratedTimeline)
    assert not any(str(f).startswith("connectivity_defect") for f in tl.flags)


def test_engine_dump_connectivity_oracle_flags_defect(tmp_path):
    # Same weave (connects 1|0~1|2), but the dump's route edge couples tiles
    # 1|1 and 1|3 -- which the weave does NOT connect -> the oracle reports a
    # connectivity defect as a timeline flag.
    dirs = _cross_domain_dirs(tmp_path)
    led = _ledger(tmp_path, [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}])
    rg = RouteGraph(edges=(RouteEdge(_pr(0, 1), _pr(0, 3, "in"), "inter_tile"),))
    dump = ConfigDump(device="npu1", route_graph=rg, tiles=())
    rep = run_engine(dirs, led, [("1|2|0|CORE", "1|0|2|MM2S")],
                     dump=dump, start_col=1)
    tl = rep["timeline"]
    assert "connectivity_defect:1|1~1|3" in tl.flags


def test_engine_backward_compat_no_dump(tmp_path):
    # Existing callers that don't pass a dump still work; dump=None means the
    # count-truncation ceiling isn't derivable -> count_ceiling_unknown remains.
    dirs = _cross_domain_dirs(tmp_path)
    led = _ledger(tmp_path, [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}])
    rep = run_engine(dirs, led, [("1|2|0|CORE", "1|0|2|MM2S")])
    tl = rep["timeline"]
    assert "count_ceiling_unknown" in tl.flags
    assert not any(str(f).startswith("connectivity_defect") for f in tl.flags)
