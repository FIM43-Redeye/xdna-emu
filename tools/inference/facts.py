"""Facts, support types, and the knowledge base.

A fact is an immutable record (predicate, args, support). Support is exactly one of
measured (straight from capture data), structural(cite) (a quote of the loaded
configuration, ledgered to its location), or derived(rule, premises). The keystone
property `provenance_ok` enforces that every fact bottoms out only in measured or
ledgered-structural leaves -- no unaudited axioms.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Union, FrozenSet, Optional, List, Dict


@dataclass(frozen=True)
class Measured:
    """Support: read straight from replicated capture data."""


@dataclass(frozen=True)
class Structural:
    """Support: a quote of the loaded configuration, cited to its location."""
    cite: str


@dataclass(frozen=True)
class Derived:
    """Support: produced by an admitted rule from existing facts."""
    rule: str
    premises: Tuple["Fact", ...]


Support = Union[Measured, Structural, Derived]


@dataclass(frozen=True)
class Fact:
    predicate: str
    args: Tuple
    support: Support

    def key(self) -> Tuple[str, Tuple]:
        return (self.predicate, self.args)


def leaves(fact: Fact) -> FrozenSet[Fact]:
    """The measured/structural leaves of a fact's provenance DAG."""
    s = fact.support
    if isinstance(s, (Measured, Structural)):
        return frozenset({fact})
    out: set = set()
    for p in s.premises:
        out |= leaves(p)
    return frozenset(out)


def derive_kind(fact: Fact) -> str:
    """segment | gap for a `derives` fact. Legacy 3-arg facts (offset only)
    read as segments -- backward-compatible."""
    return fact.args[3] if len(fact.args) >= 4 else "segment"


def derive_offset(fact: Fact) -> Optional[int]:
    """The exact cycle offset for a segment derive; None for a gap."""
    return fact.args[2]


def derive_reproduction_offset(fact: Fact) -> Optional[int]:
    """The exact raw cross-domain reproduction-target offset for a gap derive.
    None for a segment, a gap with no deterministic target, an async-CDC gap, or a
    legacy <5-arg derives fact -- backward-compatible. This is NOT a causal offset;
    `derive_offset` (args[2]) stays None for gaps."""
    return fact.args[4] if len(fact.args) >= 5 else None


def derive_gap_reason(fact: Fact) -> Optional[str]:
    """The typed reason a derive is a gap (async_cdc | cross_domain |
    within_domain_nonexact), per inference.grounding. None for a segment or a
    legacy <6-arg derives fact -- backward-compatible. The engine NOTES accounted
    reasons and WARNS on the unaccounted (within_domain_nonexact) one."""
    return fact.args[5] if len(fact.args) >= 6 else None


@dataclass
class KB:
    facts: Dict[Tuple, Fact]
    admitted_rules: List = field(default_factory=list)
    rejected_rules: List = field(default_factory=list)
    ledger: Dict[str, dict] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "KB":
        return cls(facts={}, admitted_rules=[], rejected_rules=[], ledger={})

    def add(self, fact: Fact) -> Fact:
        self.facts[fact.key()] = fact
        return fact

    def get(self, predicate: str, args: Tuple) -> Optional[Fact]:
        return self.facts.get((predicate, args))

    def has(self, predicate: str, args: Tuple) -> bool:
        return (predicate, args) in self.facts

    def by_predicate(self, predicate: str) -> List[Fact]:
        return [f for (p, _), f in self.facts.items() if p == predicate]


def provenance_ok(kb: KB) -> bool:
    """Keystone: every leaf is measured, or structural with a ledgered citation;
    and no Derived node is hanging. A zero-premise Derived node traces to no leaves
    and would satisfy the leaf check vacuously -- it is itself a provenance defect."""
    for f in kb.facts.values():
        if not _well_founded(f):
            return False
        for leaf in leaves(f):
            s = leaf.support
            if isinstance(s, Measured):
                continue
            if isinstance(s, Structural):
                if s.cite not in kb.ledger:
                    return False
            else:  # a Derived fact can never be a leaf
                return False
    return True


def _well_founded(fact: Fact) -> bool:
    """A Derived fact must have at least one premise, recursively. A zero-premise
    Derived node traces to no leaves and would escape the leaf check entirely."""
    s = fact.support
    if isinstance(s, Derived):
        if not s.premises:
            return False
        return all(_well_founded(p) for p in s.premises)
    return True
