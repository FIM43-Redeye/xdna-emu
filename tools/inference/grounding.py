"""Explicit grounding: classify a static causal edge as a cycle-exact within-
domain Segment or a named Gap, and assemble a chain into a timeline.

The deterministic, cycle-accurate unit is a segment bounded by milestone events
WITHIN one per-module timer domain whose per-run offset agrees EXACTLY (range
<= Q == 0). Everything else -- cross-domain offsets, and within-domain offsets
that bundle a delivery wait (non-exact) -- is a Gap: existence + orientation
only, no cycle count. A through-core span is therefore reported as
gap + (exact segment) + gap, never as one deterministic number.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union
from inference.verifier import offset_exact, ANCHOR, Q  # noqa: F401 (Q documents the floor)


def same_domain(a: str, b: str) -> bool:
    """a and b share a per-module timer domain iff their col|row|pkt_type prefix
    matches. The trace timer resets per (pkt_type, row, col) (C1), so two events
    on the same tile but different modules are CROSS-domain."""
    return a.rsplit("|", 1)[0] == b.rsplit("|", 1)[0]


@dataclass(frozen=True)
class Segment:
    parent: str
    child: str
    offset: int


@dataclass(frozen=True)
class Gap:
    parent: str
    child: str


Grounding = Union[Segment, Gap]


def ground_edge(run_dirs: List[str], child: str, parent: str,
                anchor_key: str = ANCHOR) -> Grounding:
    """Segment iff parent and child share a timer domain AND their cross-run
    offset is exact (range <= Q); otherwise a named Gap."""
    if same_domain(child, parent):
        off = offset_exact(run_dirs, child, parent, anchor_key)
        if off is not None:
            return Segment(parent=parent, child=child, offset=off)
    return Gap(parent=parent, child=child)


@dataclass
class Timeline:
    items: List[Grounding]


def assemble(run_dirs: List[str], edges: List[Tuple[str, str]],
             anchor_key: str = ANCHOR) -> Timeline:
    """edges: ordered [(parent, child)] forming a static causal chain. Returns a
    Timeline of per-edge groundings (exact segments interleaved with named
    gaps), in chain order."""
    return Timeline([ground_edge(run_dirs, child, parent, anchor_key)
                     for parent, child in edges])
