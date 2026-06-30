# tools/test_inference_model_provenance.py
from inference.facts import (Fact, Measured, ModelDerived, Derived, KB,
                             provenance_ok, leaves)


def _kb_with_model_source(cite="origin_d.json"):
    kb = KB.empty()
    kb.model = {cite: {"calibrated": True}}
    return kb


def test_model_derived_leaf_accepted_when_cited():
    kb = _kb_with_model_source()
    kb.add(Fact("origin_d", ("1|2|0", 0), ModelDerived("origin_d.json")))
    assert provenance_ok(kb) is True


def test_model_derived_leaf_rejected_when_uncited():
    kb = KB.empty()  # empty kb.model
    kb.add(Fact("origin_d", ("1|2|0", 0), ModelDerived("origin_d.json")))
    assert provenance_ok(kb) is False


def test_model_derived_surfaces_in_leaves():
    md = Fact("origin_d", ("1|2|0", 0), ModelDerived("origin_d.json"))
    measured = Fact("fired", ("x", 0, 0), Measured())
    causal = Fact("causal", ("c", "p", 5), Derived("causal_decomp_rule", (measured, md)))
    leaf_supports = {type(f.support).__name__ for f in leaves(causal)}
    assert "ModelDerived" in leaf_supports
