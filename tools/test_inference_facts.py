# tools/test_inference_facts.py
from inference.facts import (Measured, Structural, Derived, Fact, KB,
                             leaves, provenance_ok, derive_kind, derive_offset,
                             derive_reproduction_offset)


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


def test_provenance_ok_rejects_hanging_derived_node():
    # A Derived fact with empty premises traces to no leaves -> must be rejected,
    # not vacuously accepted.
    kb = KB.empty()
    kb.add(Fact("correlates", ("A", "B", 3), Derived("correlates_rule", ())))
    assert provenance_ok(kb) is False


def _leaf(pred, args):
    return Fact(pred, args, Measured())


def test_derive_kind_and_offset_for_segment():
    prem = _leaf("fired", ("1|2|0|REL",))
    f = Fact("derives", ("1|2|0|REL", "1|2|0|ACQ", 22, "segment"),
             Derived("derives_rule_placement", (prem,)))
    assert derive_kind(f) == "segment"
    assert derive_offset(f) == 22


def test_derive_kind_and_offset_for_gap():
    prem = _leaf("fired", ("1|0|2|S2MM",))
    f = Fact("derives", ("1|0|2|S2MM", "1|0|2|MM2S", None, "gap"),
             Derived("derives_rule_placement", (prem,)))
    assert derive_kind(f) == "gap"
    assert derive_offset(f) is None


def test_derive_kind_legacy_three_arg_reads_as_segment():
    prem = _leaf("fired", ("1|0|0|C",))
    f = Fact("derives", ("1|0|0|C", "1|0|0|S", 30),
             Derived("derives_rule_placement", (prem,)))
    assert derive_kind(f) == "segment"
    assert derive_offset(f) == 30


def test_derive_reproduction_offset_for_cross_domain_gap():
    prem = _leaf("fired", ("1|2|0|CORE",))
    f = Fact("derives", ("1|2|0|CORE", "1|0|2|MM2S", None, "gap", 30),
             Derived("derives_rule_placement", (prem,)))
    assert derive_reproduction_offset(f) == 30


def test_derive_reproduction_offset_none_for_plain_gap():
    prem = _leaf("fired", ("1|0|2|S2MM",))
    f = Fact("derives", ("1|0|2|S2MM", "1|0|2|MM2S", None, "gap", None),
             Derived("derives_rule_placement", (prem,)))
    assert derive_reproduction_offset(f) is None


def test_derive_reproduction_offset_none_for_segment():
    prem = _leaf("fired", ("1|2|0|REL",))
    f = Fact("derives", ("1|2|0|REL", "1|2|0|ACQ", 22, "segment"),
             Derived("derives_rule_placement", (prem,)))
    assert derive_reproduction_offset(f) is None


def test_derive_reproduction_offset_legacy_four_arg_gap_is_none():
    prem = _leaf("fired", ("1|0|2|S2MM",))
    f = Fact("derives", ("1|0|2|S2MM", "1|0|2|MM2S", None, "gap"),
             Derived("derives_rule_placement", (prem,)))
    assert derive_reproduction_offset(f) is None
