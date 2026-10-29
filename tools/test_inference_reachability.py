from inference.reachability import (Constraint, ReachabilityModel,
                                    observational_blocked)


def test_discharged_constraint_has_provenance():
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod_row2", "cannot_cotrace",
                                ("1|2|0|X", "1|2|1|Y"), provenance_batch="batch_07"))
    assert m.is_discharged("memmod_row2") is True


def test_undischarged_constraint_blocks_observational_verdict():
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod_row2", "cannot_cotrace",
                                ("1|2|0|X", "1|2|1|Y"), provenance_batch=None))
    assert m.is_discharged("memmod_row2") is False
    assert observational_blocked(m, "1|2|0|X", "1|2|1|Y") is True


def test_discharged_constraint_does_not_block():
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod_row2", "cannot_cotrace",
                                ("1|2|0|X", "1|2|1|Y"), provenance_batch="batch_07"))
    assert observational_blocked(m, "1|2|0|X", "1|2|1|Y") is False


def test_can_separate_unknown_when_no_constraint():
    m = ReachabilityModel()
    assert m.can_separate("a", "b") is None


def test_constraint_sharing_one_endpoint_is_not_relevant():
    # A discharged cannot_cotrace(A, X) must NOT drive the verdict for the
    # DIFFERENT pair (A, B) -- it only shares endpoint A. Over-matching here would
    # produce a false irreducible-by-instrument (the self-sealing error).
    m = ReachabilityModel()
    m.add_constraint(Constraint("ax", "cannot_cotrace", ("A", "X"),
                                provenance_batch="batch_01"))
    assert m.can_separate("A", "B") is None          # not relevant -> unknown
    assert observational_blocked(m, "A", "B") is False
