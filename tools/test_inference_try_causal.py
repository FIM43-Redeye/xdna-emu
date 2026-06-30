from inference.facts import Fact, Measured, ModelDerived, Derived, KB, provenance_ok, leaves
from inference.rules import try_causal


def _seed_cross_domain_gap(kb, child, parent, raw):
    # Minimal stand-in for what try_derives leaves in the KB: a cross-domain gap
    # derive with reproduction_offset = raw, plus the measured fired leaves.
    fired_c = kb.add(Fact("fired", (child, 0, 0), Measured()))
    fired_p = kb.add(Fact("fired", (parent, 0, 0), Measured()))
    grd = Fact("gap", (child, parent), Derived("grounding_rule", (fired_c, fired_p)))
    cp = Fact("config_path", (parent, child, "c0"), __import__("inference.facts", fromlist=["Structural"]).Structural("c0"))
    kb.ledger["c0"] = {"a": parent, "b": child, "kind": "route", "cite": "c0"}
    kb.add(cp)
    return kb.add(Fact("derives", (child, parent, None, "gap", raw, "cross_domain"),
                       Derived("derives_rule_placement", (cp, grd))))


def _install_model(kb):
    kb.model["origin_d.json"] = {"calibrated": True, "flood_source": "0|0",
                                 "modules": {"1|2|0": 10, "0|0|2": 4}}
    kb.add(Fact("skew_calibrated", (True,), ModelDerived("origin_d.json")))
    kb.add(Fact("origin_d", ("1|2|0", 10), ModelDerived("origin_d.json")))
    kb.add(Fact("origin_d", ("0|0|2", 4), ModelDerived("origin_d.json")))


def test_try_causal_emits_model_grounded_fact():
    kb = KB.empty()
    child, parent = "1|2|0|X", "0|0|2|Y"
    _seed_cross_domain_gap(kb, child, parent, raw=1)
    _install_model(kb)
    f = try_causal([], kb, child, parent)
    assert f is not None
    assert f.predicate == "causal" and f.args == (child, parent, 7)
    leaf_supports = {type(x.support).__name__ for x in leaves(f)}
    assert "ModelDerived" in leaf_supports and "Measured" in leaf_supports
    kb.add(f)
    assert provenance_ok(kb) is True


def test_try_causal_none_when_uncalibrated():
    kb = KB.empty()
    child, parent = "1|2|0|X", "0|0|2|Y"
    _seed_cross_domain_gap(kb, child, parent, raw=1)
    kb.model["origin_d.json"] = {"calibrated": False, "flood_source": "0|0",
                                 "modules": {"1|2|0": 10, "0|0|2": 4}}
    assert try_causal([], kb, child, parent) is None
