import json
from inference.facts import KB, provenance_ok
from inference.ledger import load_ledger, install_ledger
from inference.loader import load_fired
from inference.chainer import chain, classify_events, chaining_sound


def _ev(col, row, name, soc):
    return {"col": col, "row": row, "pkt_type": 0, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    dirs = []
    for i, off in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for name, delta in off.items():
            evs.append(_ev(1, 0, name, 1000 + delta))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def _kb(tmp_path, dirs, entries):
    p = tmp_path / "led.json"
    p.write_text(json.dumps({"entries": entries}))
    kb = KB.empty()
    install_ledger(kb, load_ledger(str(p)))
    for f in load_fired(dirs):
        kb.add(f)
    return kb


def test_chain_places_child_and_marks_root(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 230, "C": 260},
                            {"S": 300, "C": 330}])
    kb = _kb(tmp_path, dirs, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    kb = chain(dirs, kb, [("1|0|0|C", "1|0|0|S")])
    cls = classify_events(kb, ["1|0|0|S", "1|0|0|C", "1|2|0|PERF_CNT_2"])
    assert cls["1|0|0|C"] == "derived"
    assert cls["1|0|0|S"] == "stochastic_root"
    assert cls["1|2|0|PERF_CNT_2"] == "deterministic"


def test_chain_reaches_fixpoint_idempotent(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 230, "C": 260}])
    kb = _kb(tmp_path, dirs, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    kb = chain(dirs, kb, [("1|0|0|C", "1|0|0|S")])
    n1 = len(kb.facts)
    kb = chain(dirs, kb, [("1|0|0|C", "1|0|0|S")])
    assert len(kb.facts) == n1


def test_chaining_soundness_property_holds(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 230, "C": 260}])
    kb = _kb(tmp_path, dirs, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    kb = chain(dirs, kb, [("1|0|0|C", "1|0|0|S")])
    assert chaining_sound(kb) is True
    assert provenance_ok(kb) is True
