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
from typing import List, Optional, Tuple, Union
from inference.verifier import offset_exact, ANCHOR, Q  # noqa: F401 (Q documents the floor)


def same_domain(a: str, b: str) -> bool:
    """a and b share a per-module timer domain iff their col|row|pkt_type prefix
    matches. The trace timer resets per (pkt_type, row, col) (C1), so two events
    on the same tile but different modules are CROSS-domain."""
    return a.rsplit("|", 1)[0] == b.rsplit("|", 1)[0]


def is_async_cdc(event_key: str) -> bool:
    """True for shim NoC-egress DMA completion events. Their timing crosses the
    async 1 GHz<->960 MHz NoC FIFO to DDR (AM020 CDC) and is non-deterministic --
    never a cycle-deterministic causal fact. Derived from event semantics: a
    shim-row (row 0, AIE2 topology) DMA_*_FINISHED_TASK event. Gap-only: never a
    Segment, never a reproduction target."""
    parts = event_key.split("|")
    if len(parts) != 4:
        return False
    _col, row, _pkt, name = parts
    return row == "0" and name.startswith("DMA_") and name.endswith("_FINISHED_TASK")


@dataclass(frozen=True)
class Segment:
    parent: str
    child: str
    offset: int


@dataclass(frozen=True)
class Gap:
    parent: str
    child: str
    reproduction_offset: Optional[int] = None


Grounding = Union[Segment, Gap]


def ground_edge(run_dirs: List[str], child: str, parent: str,
                anchor_key: str = ANCHOR) -> Grounding:
    """Within-domain exact offset -> Segment (cycle-accurate causal latency).
    Otherwise a named Gap. A cross-domain Gap carries the exact raw offset as a
    `reproduction_offset` when it agrees across runs (range <= Q), else None.
    Async-CDC events (shim NoC-egress DMA completion) are gap-only by semantics:
    never a Segment, never a reproduction target."""
    if is_async_cdc(child) or is_async_cdc(parent):
        return Gap(parent=parent, child=child)
    if same_domain(child, parent):
        off = offset_exact(run_dirs, child, parent, anchor_key)
        if off is not None:
            return Segment(parent=parent, child=child, offset=off)
        return Gap(parent=parent, child=child)
    raw = offset_exact(run_dirs, child, parent, anchor_key)
    return Gap(parent=parent, child=child, reproduction_offset=raw)


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
