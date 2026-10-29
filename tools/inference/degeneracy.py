"""Degeneracy substrate: union-find identity closure + SCC condensation.

v1 model is pure equalities, so union-find (congruence closure over same_source) is
"obviously correct by reading it". The two-method interface (same_class / classes) is
the swap point for Z3, which arrives in Plan 3 with group disjunctions. The placement
graph (derives edges) may contain cycles (circular BD chains, lock round-trips,
ping-pong); condense() collapses each strongly-connected component into one
irreducible group so the reduction runs over the acyclic condensation.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, FrozenSet
from inference.facts import KB


class IdentityClasses:
    """Union-find over same_source edges only. derives edges are excluded:
    a non-zero offset is proof two events are not the same physical event."""

    def __init__(self):
        self._parent: Dict[str, str] = {}

    def _find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[ra] = rb

    def same_class(self, a: str, b: str) -> bool:
        if a not in self._parent or b not in self._parent:
            return a == b
        return self._find(a) == self._find(b)

    def classes(self) -> List[FrozenSet[str]]:
        buckets: Dict[str, set] = {}
        for x in self._parent:
            buckets.setdefault(self._find(x), set()).add(x)
        return [frozenset(s) for s in buckets.values()]

    @classmethod
    def from_kb(cls, kb: KB) -> "IdentityClasses":
        ic = cls()
        for f in kb.by_predicate("same_source"):
            ic.union(f.args[0], f.args[1])
        return ic


def condense(kb: KB) -> Tuple[Dict[str, int], List[FrozenSet[str]]]:
    """Tarjan SCC over directed derives edges (parent -> child)."""
    adj: Dict[str, List[str]] = {}
    nodes: set = set()
    for f in kb.by_predicate("derives"):
        child, parent = f.args[0], f.args[1]
        nodes.add(child); nodes.add(parent)
        adj.setdefault(parent, []).append(child)
        adj.setdefault(child, adj.get(child, []))

    index = {}
    low = {}
    on_stack = {}
    stack: List[str] = []
    counter = [0]
    comp_of: Dict[str, int] = {}
    groups: List[FrozenSet[str]] = []

    def strongconnect(v: str):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v); on_stack[v] = True
        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif on_stack.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            members = set()
            while True:
                w = stack.pop(); on_stack[w] = False
                members.add(w)
                if w == v:
                    break
            cid = len(groups)
            for m in members:
                comp_of[m] = cid
            groups.append(frozenset(members))

    for v in sorted(nodes):
        if v not in index:
            strongconnect(v)
    return comp_of, groups
