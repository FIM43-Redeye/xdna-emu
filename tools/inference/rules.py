"""Verified derivation rules: derives, same_source, stochastic_root.

Orientation and identity are DERIVED from structural config premises by these stated,
verified rules -- not primitive assertions. `derives` is placement (dataflow-upstream
+ stable offset), explicitly NOT timing-causation: the rule never inspects timing
direction, so a backpressure event (timing-cause opposite to dataflow) is still
placed correctly and is never labeled causal.
"""
from __future__ import annotations
from typing import List, Optional
from inference.facts import Fact, Derived, KB
from inference.verifier import anchor_rigid, check_ordering, coincident, ANCHOR
from inference.grounding import ground_edge, Segment


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
    return Fact("derives", (child, parent, None, "gap"),
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


def is_stochastic_root(kb: KB, event_key: str) -> bool:
    if kb.has("deterministic", (event_key,)):
        return False
    if not kb.has("stochastic", (event_key,)):
        return False
    has_parent = any(f.args[0] == event_key for f in kb.by_predicate("derives"))
    return not has_parent
