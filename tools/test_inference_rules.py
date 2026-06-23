import json
from inference.facts import KB
from inference.ledger import install_ledger
from inference.rules import (try_derives, try_same_source, is_stochastic_root,
                             mark_determinism)
from inference.loader import load_fired


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    """rows: list over runs of {name: anchored_offset at 1|0|0}."""
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


def _kb_with_ledger(tmp_path, dirs, entries):
    p = tmp_path / "led.json"
    p.write_text(json.dumps({"entries": entries}))
    from inference.ledger import load_ledger
    kb = KB.empty()
    install_ledger(kb, load_ledger(str(p)))
    from inference.loader import load_fired
    for f in load_fired(dirs):
        kb.add(f)
    return kb


def test_derives_segment_with_stochastic_parent_within_domain(tmp_path):
    # parent S jitters; child C = S + 30 EVERY run (range 0); same domain (1|0|0);
    # config routes S -> C -> a SEGMENT derive with exact offset 30.
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230},
                            {"S": 350, "C": 380}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    f = try_derives(dirs, kb, "1|0|0|C", "1|0|0|S")
    from inference.facts import derive_kind, derive_offset, provenance_ok
    assert f is not None and f.predicate == "derives"
    assert f.args[:2] == ("1|0|0|C", "1|0|0|S")
    assert derive_kind(f) == "segment" and derive_offset(f) == 30
    kb.add(f)
    assert provenance_ok(kb) is True


def test_derives_gap_when_within_domain_offset_nonexact(tmp_path):
    # offset C-S varies (30, 30, 31) -> not exact -> GAP derive (placed, no offset)
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230},
                            {"S": 350, "C": 381}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    f = try_derives(dirs, kb, "1|0|0|C", "1|0|0|S")
    from inference.facts import derive_kind, derive_offset
    assert f is not None and derive_kind(f) == "gap" and derive_offset(f) is None


def test_derives_rejected_when_parent_rigid(tmp_path):
    # parent D is rigid (range 0 anchored) -> not a stochastic source -> None
    dirs = _runs(tmp_path, [{"D": 40, "C": 70}, {"D": 40, "C": 70},
                            {"D": 40, "C": 70}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#1", "a": "1|0|0|D", "b": "1|0|0|C", "kind": "route"}])
    assert try_derives(dirs, kb, "1|0|0|C", "1|0|0|D") is None


def test_derives_rejected_without_orientation(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230}])
    kb = KB.empty()  # empty ledger -> no config_path/program_path
    assert try_derives(dirs, kb, "1|0|0|C", "1|0|0|S") is None


def test_derives_ordering_violation_is_rejected_and_recorded(tmp_path):
    # config routes S -> C but C fires BEFORE S every run -> ordering falsifier
    dirs = _runs(tmp_path, [{"S": 130, "C": 100}, {"S": 230, "C": 200}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    assert try_derives(dirs, kb, "1|0|0|C", "1|0|0|S") is None
    assert any(r.name == "ordering" for r in kb.rejected_rules)


def test_derives_places_backpressure_event_without_causal_claim(tmp_path):
    dirs = _runs(tmp_path, [{"P": 100, "SS": 130}, {"P": 220, "SS": 250},
                            {"P": 300, "SS": 330}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#9", "a": "1|0|0|P", "b": "1|0|0|SS", "kind": "route"}])
    f = try_derives(dirs, kb, "1|0|0|SS", "1|0|0|P")
    assert f is not None and f.predicate == "derives"
    assert "caus" not in f.support.rule.lower()


def test_try_derives_consumes_program_path(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230},
                            {"S": 350, "C": 380}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "program:1|0|0|S--via-core-->1|0|0|C", "a": "1|0|0|S",
         "b": "1|0|0|C", "kind": "program"}])
    f = try_derives(dirs, kb, "1|0|0|C", "1|0|0|S")
    from inference.facts import derive_kind
    assert f is not None and derive_kind(f) == "segment"
    assert kb.by_predicate("program_path") != []
    assert kb.by_predicate("config_path") == []


def test_same_source_requires_identity_and_coincidence(tmp_path):
    dirs = _runs(tmp_path, [{"A": 40, "A2": 40}, {"A": 41, "A2": 41}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "id#1", "a": "1|0|0|A", "b": "1|0|0|A2", "kind": "identity"}])
    f = try_same_source(dirs, kb, "1|0|0|A", "1|0|0|A2")
    assert f is not None and f.predicate == "same_source"


def test_stochastic_root_when_jittery_and_underived(tmp_path):
    dirs = _runs(tmp_path, [{"R": 40}, {"R": 90}, {"R": 140}])
    kb = KB.empty()
    for f in load_fired(dirs):
        kb.add(f)
    mark_determinism(dirs, kb, ["1|0|0|R"])
    assert is_stochastic_root(kb, "1|0|0|R") is True
