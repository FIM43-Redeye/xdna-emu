"""The reachability self-model -- the one place the keystone could leak.

The observational degeneracy branch stops measuring based on "no reachable batch
separates them". If the model were incomplete, a separable pair would be misclassified
as irreducible and the error would be self-sealing. So every constraint is a
first-class verified artifact: it carries the batch that DEMONSTRATED the limit
(measured provenance), and an observational verdict is BLOCKED until every constraint
it relies on is discharged. The memmod row-2 confound is the first such constraint.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Constraint:
    name: str
    predicate: str            # e.g. "cannot_cotrace"
    args: Tuple
    provenance_batch: Optional[str]   # the batch dir that demonstrated the limit


class ReachabilityModel:
    def __init__(self):
        self._constraints: List[Constraint] = []

    def add_constraint(self, c: Constraint) -> None:
        self._constraints.append(c)

    def is_discharged(self, name: str) -> bool:
        return any(c.name == name and c.provenance_batch is not None
                   for c in self._constraints)

    def _relevant(self, a: str, b: str) -> List[Constraint]:
        pair = {a, b}
        return [c for c in self._constraints if pair <= set(c.args)]

    def can_separate(self, a: str, b: str) -> Optional[bool]:
        rel = self._relevant(a, b)
        if not rel:
            return None
        # a discharged "cannot_cotrace"/"cannot_separate" constraint -> cannot separate
        for c in rel:
            if c.predicate.startswith("cannot") and c.provenance_batch is not None:
                return False
        return True

    def blocking_constraints(self, a: str, b: str) -> List[Constraint]:
        return [c for c in self._relevant(a, b)
                if c.predicate.startswith("cannot") and c.provenance_batch is None]


def observational_blocked(model: ReachabilityModel, a: str, b: str) -> bool:
    return len(model.blocking_constraints(a, b)) > 0
