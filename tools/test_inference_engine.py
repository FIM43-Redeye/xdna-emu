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
    assert ("1|0|0|C", "1|0|0|S", 30) in rep["derives"]
    assert "1|0|0|S" in rep["stochastic_roots"]
    assert rep["classification"]["1|0|0|C"] == "derived"


def test_engine_reports_irreducible_cycle(tmp_path):
    # A and B each derive the other (lock round-trip) -> one irreducible group
    dirs = _runs(tmp_path, [
        {("1|0|0", "A"): 100, ("1|0|0", "B"): 110},
        {("1|0|0", "A"): 200, ("1|0|0", "B"): 210}])
    led = _ledger(tmp_path, [
        {"cite": "r1", "a": "1|0|0|A", "b": "1|0|0|B", "kind": "route"},
        {"cite": "r2", "a": "1|0|0|B", "b": "1|0|0|A", "kind": "route"}])
    rep = run_engine(dirs, led, [("1|0|0|B", "1|0|0|A"), ("1|0|0|A", "1|0|0|B")])
    multi = [g for g in rep["irreducible_groups"] if len(g) > 1]
    assert frozenset({"1|0|0|A", "1|0|0|B"}) in multi
