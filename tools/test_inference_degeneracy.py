from inference.facts import KB, Fact, Derived
from inference.degeneracy import IdentityClasses, condense


def _same_source(a, b):
    return Fact("same_source", (a, b), Derived("same_source_rule", ()))


def _derives(child, parent, off):
    return Fact("derives", (child, parent, off), Derived("derives_rule_placement", ()))


def test_identity_closure_unions_same_source_chain():
    kb = KB.empty()
    kb.add(_same_source("A", "B"))
    kb.add(_same_source("B", "C"))
    ic = IdentityClasses.from_kb(kb)
    assert ic.same_class("A", "C") is True
    assert ic.same_class("A", "D") is False
    parts = {frozenset(c) for c in ic.classes()}
    assert frozenset({"A", "B", "C"}) in parts


def test_derives_edges_excluded_from_identity():
    # nonzero offset proves NOT identical -> never same_class
    kb = KB.empty()
    kb.add(_derives("C", "S", 30))
    ic = IdentityClasses.from_kb(kb)
    assert ic.same_class("C", "S") is False


def test_condense_collapses_cycle_to_one_group():
    # lock round-trip: A -> B -> A is one irreducible group
    kb = KB.empty()
    kb.add(_derives("B", "A", 10))
    kb.add(_derives("A", "B", -10))
    kb.add(_derives("D", "C", 5))   # acyclic edge stays singletons
    comp, groups = condense(kb)
    multi = [g for g in groups if len(g) > 1]
    assert multi == [frozenset({"A", "B"})]
    assert comp["A"] == comp["B"] and comp["C"] != comp["D"]
