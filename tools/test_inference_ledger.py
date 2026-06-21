import json
import pytest
from inference.facts import KB
from inference.ledger import (load_ledger, ledger_facts, install_ledger,
                              citation_resolves)


def _write(tmp_path, entries):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"entries": entries}))
    return str(p)


def test_load_ledger_keys_by_cite(tmp_path):
    p = _write(tmp_path, [
        {"cite": "route#7", "a": "1|0|0|DMA", "b": "1|1|3|PORT", "kind": "route"}])
    led = load_ledger(p)
    assert led["route#7"]["kind"] == "route"


def test_load_ledger_rejects_duplicate_cite(tmp_path):
    p = _write(tmp_path, [
        {"cite": "x", "a": "A", "b": "B", "kind": "route"},
        {"cite": "x", "a": "C", "b": "D", "kind": "lock"}])
    with pytest.raises(ValueError):
        load_ledger(p)


def test_ledger_facts_emit_config_path_and_identity(tmp_path):
    p = _write(tmp_path, [
        {"cite": "route#7", "a": "A", "b": "B", "kind": "route"},
        {"cite": "id#1", "a": "A", "b": "A2", "kind": "identity"}])
    led = load_ledger(p)
    facts = ledger_facts(led)
    preds = {(f.predicate, f.args) for f in facts}
    assert ("config_path", ("A", "B", "route#7")) in preds
    assert ("identity", ("A", "A2", "id#1")) in preds
    for f in facts:
        assert type(f.support).__name__ == "Structural"


def test_install_ledger_makes_provenance_ok_hold(tmp_path):
    from inference.facts import provenance_ok
    p = _write(tmp_path, [
        {"cite": "route#7", "a": "A", "b": "B", "kind": "route"}])
    led = load_ledger(p)
    kb = KB.empty()
    install_ledger(kb, led)
    assert citation_resolves(led, "route#7") is True
    assert citation_resolves(led, "ghost") is False
    assert provenance_ok(kb) is True
