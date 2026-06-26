"""Explicit grounding: classify a static causal edge as a cycle-exact within-
domain Segment or a named Gap, and assemble a chain into a timeline.

The deterministic, cycle-accurate unit is a segment bounded by milestone events
WITHIN one per-module timer domain whose per-run offset agrees EXACTLY (range
<= Q == 0). Everything else -- cross-domain offsets, and within-domain offsets
that bundle a delivery wait (non-exact) -- is a Gap: existence + orientation
only, no cycle count. A through-core span is therefore reported as
gap + (exact segment) + gap, never as one deterministic number.

Every Gap carries a typed `reason`. Two reasons are *accounted-for* -- the
non-exactness is structurally expected, so the engine NOTES it: cross-domain
(different timer domains) and async-CDC (shim NoC egress). The third,
within-domain-nonexact, is *unaccounted* -- a span that should be cycle-exact
but ranges -- and the engine WARNS on it instead of silently swallowing it as a
gap. That is the load-contamination canary (docs/trace/capture-load-sensitivity.md):
a within-domain range > 0 is either genuine HW nondeterminism or host-load
capture contamination, and which one is verified MANUALLY by re-capturing on a
quiet host -- never auto-classified, never tolerated by a statistical threshold.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from inference.verifier import offset_exact, ANCHOR, Q  # noqa: F401 (Q documents the floor)


# Gap reasons -- WHY a causal edge is a named gap rather than a cycle-exact
# segment. The first two are *accounted-for* (we understand the non-exactness and
# NOTE it); the last is *unaccounted* (a within-domain span that should be exact
# but isn't -- the engine WARNS on it, never silently swallows it as a gap). The
# unaccounted case is the load-contamination canary: see
# docs/trace/capture-load-sensitivity.md.
GAP_ASYNC_CDC = "async_cdc"            # shim NoC-egress DMA completion (AM020 CDC)
GAP_CROSS_DOMAIN = "cross_domain"      # endpoints in different per-module timer domains
GAP_WITHIN_DOMAIN_NONEXACT = "within_domain_nonexact"  # should be exact -- anomaly

_ACCOUNTED_GAP_REASONS = frozenset({GAP_ASYNC_CDC, GAP_CROSS_DOMAIN})


def gap_accounted(reason: Optional[str]) -> bool:
    """True if a gap's non-exactness is structurally accounted-for (NOTE it).
    False for the unaccounted within-domain anomaly -- and for an unknown/None
    reason, which fails loud (WARN) rather than silently passing."""
    return reason in _ACCOUNTED_GAP_REASONS


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
    reason: str
    reproduction_offset: Optional[int] = None

    @property
    def accounted(self) -> bool:
        """Whether this gap's non-exactness is structurally accounted-for (NOTE)
        vs an unaccounted anomaly to WARN on."""
        return gap_accounted(self.reason)


Grounding = Union[Segment, Gap]


def ground_edge(run_dirs: List[str], child: str, parent: str,
                anchor_key: str = ANCHOR) -> Grounding:
    """Within-domain exact offset -> Segment (cycle-accurate causal latency).
    Otherwise a named Gap. A cross-domain Gap carries the exact raw offset as a
    `reproduction_offset` when it agrees across runs (range <= Q), else None.
    Async-CDC events (shim NoC-egress DMA completion) are gap-only by semantics:
    never a Segment, never a reproduction target."""
    if is_async_cdc(child) or is_async_cdc(parent):
        return Gap(parent=parent, child=child, reason=GAP_ASYNC_CDC)
    if same_domain(child, parent):
        off = offset_exact(run_dirs, child, parent, anchor_key)
        if off is not None:
            return Segment(parent=parent, child=child, offset=off)
        return Gap(parent=parent, child=child, reason=GAP_WITHIN_DOMAIN_NONEXACT)
    raw = offset_exact(run_dirs, child, parent, anchor_key)
    return Gap(parent=parent, child=child, reason=GAP_CROSS_DOMAIN,
               reproduction_offset=raw)


