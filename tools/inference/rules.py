"""Verified derivation rules: derives, same_source, stochastic_root.

Orientation and identity are DERIVED from structural config premises by these stated,
verified rules -- not primitive assertions. `derives` is placement (dataflow-upstream
+ stable offset), explicitly NOT timing-causation: the rule never inspects timing
direction, so a backpressure event (timing-cause opposite to dataflow) is still
placed correctly and is never labeled causal.
"""
from __future__ import annotations
from typing import List, Optional
from inference.facts import (Fact, Derived, KB, derive_kind, derive_gap_reason,
                             derive_reproduction_offset)
from inference.verifier import anchor_rigid, check_ordering, coincident, ANCHOR
from inference.grounding import (ground_edge, Segment, causal_offset, domain_of,
                                 GAP_CROSS_DOMAIN)


def _measured_premises(kb: KB, *event_keys: str) -> tuple:
    """The measured `fired` facts for the given event keys, as provenance leaves."""
    keys = set(event_keys)
    return tuple(f for f in kb.by_predicate("fired") if f.args[0] in keys)


def mark_determinism(run_dirs: List[str], kb: KB, event_keys: List[str],
                     anchor_key: str = ANCHOR) -> None:
    for ek in event_keys:
        is_det = anchor_rigid(run_dirs, ek, anchor_key)
        pred = "deterministic" if is_det else "stochastic"
        premises = tuple(f for f in kb.by_predicate("fired") if f.args[0] == ek)
        kb.add(Fact(pred, (ek,), Derived("determinism_rule", premises)))


def try_derives(run_dirs: List[str], kb: KB, child: str, parent: str,
                anchor_key: str = ANCHOR) -> Optional[Fact]:
    # (1) orientation by verified rule: config_path OR program_path
    cp = next((f for f in (kb.by_predicate("config_path") + kb.by_predicate("program_path"))
               if f.args[0] == parent and f.args[1] == child), None)
    if cp is None:
        return None
    # (2) parent must be a stochastic source -- a rigid parent transmits no jitter
    if anchor_rigid(run_dirs, parent, anchor_key):
        return None
    # (3) falsifier: ordering must hold on this edge in every run
    rej = check_ordering(run_dirs, [(parent, child)], anchor_key)
    if rej is not None:
        kb.rejected_rules.append(rej)
        return None
    # (4) grounding: exact within-domain segment, else a named gap. Both are
    # PLACED (existence + orientation); only the segment carries a cycle offset.
    g = ground_edge(run_dirs, child, parent, anchor_key)
    if isinstance(g, Segment):
        grd = Fact("segment", (child, parent, g.offset),
                   Derived("grounding_rule", _measured_premises(kb, child, parent)))
        return Fact("derives", (child, parent, g.offset, "segment"),
                    Derived("derives_rule_placement", (cp, grd)))
    grd = Fact("gap", (child, parent),
               Derived("grounding_rule", _measured_premises(kb, child, parent)))
    return Fact("derives",
                (child, parent, None, "gap", g.reproduction_offset, g.reason),
                Derived("derives_rule_placement", (cp, grd)))


def try_same_source(run_dirs: List[str], kb: KB, a: str, b: str,
                    anchor_key: str = ANCHOR) -> Optional[Fact]:
    ident = next((f for f in kb.by_predicate("identity")
                  if {f.args[0], f.args[1]} == {a, b}), None)
    if ident is None:
        return None
    if not coincident(run_dirs, a, b, anchor_key):
        return None
    coin = Fact("coincident", (a, b),
                Derived("coincident_rule", _measured_premises(kb, a, b)))
    return Fact("same_source", (a, b), Derived("same_source_rule", (ident, coin)))


def try_causal(run_dirs: List[str], kb: KB, child: str, parent: str,
               anchor_key: str = ANCHOR) -> Optional[Fact]:
    """Emit a model-grounded causal fact for a placed cross-domain gap, when the
    broadcast model is calibrated. Its premises explicitly cite the origin_D
    ModelDerived facts, so leaves() surfaces the model dependency -- the causal
    cycle never launders into a measured claim (design Sec.5a)."""
    model = next(iter(kb.model.values()), None)
    if model is None or not model.get("calibrated", False):
        return None
    d = next((f for f in kb.by_predicate("derives")
              if f.args[0] == child and f.args[1] == parent
              and derive_kind(f) == "gap"
              and derive_gap_reason(f) == GAP_CROSS_DOMAIN), None)
    if d is None:
        return None
    raw = derive_reproduction_offset(d)
    co = causal_offset(model, child, parent, raw)  # raises on missing domain
    if co is None:
        return None
    od = model["modules"]
    cd, pd = domain_of(child), domain_of(parent)
    od_child = kb.get("origin_d", (cd, od[cd]))
    od_parent = kb.get("origin_d", (pd, od[pd]))
    cal = kb.get("skew_calibrated", (True,))
    if od_child is None or od_parent is None or cal is None:
        raise RuntimeError(
            f"causal fact for {child}<-{parent}: model is calibrated but the "
            f"origin_d/skew_calibrated ModelDerived facts are missing from the KB "
            f"(install_model invariant violated)")
    premises = (d, od_child, od_parent, cal)
    return Fact("causal", (child, parent, co),
                Derived("causal_decomp_rule", premises))


def is_stochastic_root(kb: KB, event_key: str) -> bool:
    if kb.has("deterministic", (event_key,)):
        return False
    if not kb.has("stochastic", (event_key,)):
        return False
    has_parent = any(f.args[0] == event_key for f in kb.by_predicate("derives"))
    return not has_parent
