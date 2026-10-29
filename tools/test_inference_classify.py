from inference.facts import KB, Fact, Derived
from inference.degeneracy import IdentityClasses
from inference.reachability import Constraint, ReachabilityModel
from inference.classify import classify_pair, Verdict


def _ic_with(*edges):
    kb = KB.empty()
    for a, b in edges:
        kb.add(Fact("same_source", (a, b), Derived("same_source_rule", ())))
    return IdentityClasses.from_kb(kb)


def test_structural_candidate_when_same_class_and_confirmable():
    ic = _ic_with(("A", "B"))
    m = ReachabilityModel()
    m.add_constraint(Constraint("sep_AB", "can_separate", ("A", "B"),
                                provenance_batch="batch_03"))
    v = classify_pair("A", "B", ic, m)
    assert v.kind == "structural-candidate"


def test_unconfirmable_structural_when_no_experiment():
    ic = _ic_with(("A", "B"))
    m = ReachabilityModel()  # nothing known about separability of A,B
    v = classify_pair("A", "B", ic, m)
    assert v.kind == "unconfirmable-structural"


def test_blocked_when_undischarged_constraint():
    ic = _ic_with()  # not same class
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod", "cannot_cotrace", ("A", "B"),
                                provenance_batch=None))
    assert classify_pair("A", "B", ic, m).kind == "blocked"


def test_irreducible_when_discharged_cannot_separate():
    ic = _ic_with()
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod", "cannot_cotrace", ("A", "B"),
                                provenance_batch="batch_07"))
    assert classify_pair("A", "B", ic, m).kind == "irreducible-by-instrument"


def test_separable_when_reachable():
    ic = _ic_with()
    m = ReachabilityModel()
    m.add_constraint(Constraint("sep", "can_separate", ("A", "B"),
                                provenance_batch="batch_02"))
    assert classify_pair("A", "B", ic, m).kind == "separable"
