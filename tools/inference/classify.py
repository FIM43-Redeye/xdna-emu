"""The degeneracy trichotomy: structural / observational / separable.

structurally degenerate -> structural-candidate (provisional until its falsifiable
non-separation prediction is run and confirmed) or unconfirmable-structural (the
confirmation batch is itself unreachable -- a distinct honest finding, never a
silently-trusted collapse). observationally degenerate -> irreducible-by-instrument
(finite enumeration over the verified self-model found no reachable separating batch).
separable -> MEASURE-NEXT. An observational verdict is blocked until its reachability
constraints are discharged.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from inference.degeneracy import IdentityClasses
from inference.reachability import ReachabilityModel, observational_blocked


@dataclass
class Verdict:
    kind: str
    pair: Tuple[str, str]
    detail: dict


def classify_pair(a: str, b: str, identity: IdentityClasses,
                  model: ReachabilityModel) -> Verdict:
    pair = (a, b)
    if identity.same_class(a, b):
        sep = model.can_separate(a, b)
        if sep is None and not _has_discharged_nonsep(model, a, b):
            return Verdict("unconfirmable-structural", pair,
                           {"why": "non-separation prediction unrunnable"})
        return Verdict("structural-candidate", pair,
                       {"why": "identity collapse over audited same_source edges",
                        "gate": "run non-separation prediction to confirm"})
    if observational_blocked(model, a, b):
        return Verdict("blocked", pair,
                       {"why": "undischarged reachability constraint",
                        "constraints": [c.name for c in
                                        model.blocking_constraints(a, b)]})
    if model.can_separate(a, b) is False:
        return Verdict("irreducible-by-instrument", pair,
                       {"why": "no reachable batch separates them"})
    return Verdict("separable", pair, {"why": "reachable distinguishing batch exists"})


def _has_discharged_nonsep(model: ReachabilityModel, a: str, b: str) -> bool:
    return model.can_separate(a, b) is False
