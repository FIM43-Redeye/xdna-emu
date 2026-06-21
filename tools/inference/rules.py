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
from inference.verifier import correlates, deterministic, coincident, ANCHOR, EPS


def mark_determinism(run_dirs: List[str], kb: KB, event_keys: List[str],
                     anchor_key: str = ANCHOR, eps: float = EPS) -> None:
    for ek in event_keys:
        is_det = deterministic(run_dirs, ek, anchor_key, eps)
        pred = "deterministic" if is_det else "stochastic"
        premises = tuple(f for f in kb.by_predicate("fired") if f.args[0] == ek)
        kb.add(Fact(pred, (ek,), Derived("determinism_rule", premises)))


def try_derives(run_dirs: List[str], kb: KB, child: str, parent: str,
                anchor_key: str = ANCHOR, eps: float = EPS) -> Optional[Fact]:
    # (1) measured stable offset
    offset = correlates(run_dirs, child, parent, anchor_key, eps)
    if offset is None:
        return None
    # (2) parent is stochastic -- a deterministic parent transmits no jitter
    if deterministic(run_dirs, parent, anchor_key, eps):
        return None
    # (3) config_path(parent, child) gives orientation by verified rule
    cp = next((f for f in kb.by_predicate("config_path")
               if f.args[0] == parent and f.args[1] == child), None)
    if cp is None:
        return None
    corr = Fact("correlates", (child, parent, offset), Derived("correlates_rule", ()))
    return Fact("derives", (child, parent, offset),
                Derived("derives_rule_placement", (cp, corr)))


def try_same_source(run_dirs: List[str], kb: KB, a: str, b: str,
                    anchor_key: str = ANCHOR, eps: float = EPS) -> Optional[Fact]:
    ident = next((f for f in kb.by_predicate("identity")
                  if {f.args[0], f.args[1]} == {a, b}), None)
    if ident is None:
        return None
    if not coincident(run_dirs, a, b, anchor_key, eps):
        return None
    coin = Fact("coincident", (a, b), Derived("coincident_rule", ()))
    return Fact("same_source", (a, b), Derived("same_source_rule", (ident, coin)))


def is_stochastic_root(kb: KB, event_key: str) -> bool:
    if kb.has("deterministic", (event_key,)):
        return False
    if not kb.has("stochastic", (event_key,)):
        return False
    has_parent = any(f.args[0] == event_key for f in kb.by_predicate("derives"))
    return not has_parent
