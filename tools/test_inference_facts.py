# tools/test_inference_facts.py
from inference.facts import (Measured, Structural, Derived, Fact, KB,
                             leaves, provenance_ok)


def test_fact_is_hashable_and_keyed():
    f = Fact("fired", ("1|0|2|X", 0, 5), Measured())
    assert f.key() == ("fired", ("1|0|2|X", 0, 5))
    assert f in {f}  # hashable


def test_leaves_of_measured_is_itself():
    f = Fact("fired", ("X", 0, 5), Measured())
    assert leaves(f) == frozenset({f})


def test_leaves_walk_to_measured_and_structural():
    m = Fact("correlates", ("A", "B", 3), Measured())
    s = Fact("config_path", ("B", "A", "cite:route#7"), Structural("cite:route#7"))
    d = Fact("derives", ("A", "B", 3), Derived("derives_rule", (m, s)))
    assert leaves(d) == frozenset({m, s})


def test_provenance_ok_requires_structural_leaf_in_ledger():
    kb = KB.empty()
    kb.ledger["cite:route#7"] = {"a": "B", "b": "A", "kind": "route"}
    m = Fact("correlates", ("A", "B", 3), Measured())
    s = Fact("config_path", ("B", "A", "cite:route#7"), Structural("cite:route#7"))
    d = kb.add(Fact("derives", ("A", "B", 3), Derived("derives_rule", (m, s))))
    kb.add(m); kb.add(s)
    assert provenance_ok(kb) is True


def test_provenance_ok_fails_when_citation_missing_from_ledger():
    kb = KB.empty()  # ledger empty -> the cite is unaudited
    s = Fact("config_path", ("B", "A", "cite:ghost"), Structural("cite:ghost"))
    kb.add(s)
    assert provenance_ok(kb) is False
