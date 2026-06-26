"""Empirical verifiers: offset_exact / anchor_rigid / coincident / falsifier triad.

All primitives are exact (range <= Q where Q=0). No std-based tolerance survives
here. A rule is a hypothesis paired with a verifier; a failed rule is never used
and is itself a finding (RejectedRule).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import trace_join as tj
import trace_variance as tv

ANCHOR = "1|2|0|PERF_CNT_2"

# Q -- the measurement floor for exact-agreement grounding. Within-domain
# edge-event offsets agree EXACTLY across runs (range 0 over 20 HW runs, spike
# findings). Q is a measured toolchain property, NOT a tuned tolerance: if a
# future kernel exposes a genuine discrete trace-frame quantum it is documented
# as that quantum, never a value chosen to pass a test.
Q = 0


@dataclass
class Rule:
    name: str
    verify: Callable


@dataclass
class RejectedRule:
    name: str
    reason: str
    evidence: dict


def _anchored_per_run(run_dirs: List[str], event_key: str,
                      anchor_key: str) -> List[Dict[str, int]]:
    per_run: List[Dict[str, int]] = []
    for rd in run_dirs:
        for bn in tj._batch_names(rd):
            f = tj.batch_firsts(rd, bn, anchor_key)
            if event_key in f:
                per_run.append({event_key: f[event_key]})
                break
    return per_run


def anchored_occurrences_per_run(run_dirs: List[str], event_key: str,
                                 pinned_batch: str,
                                 anchor_key: str = ANCHOR) -> List[List[int]]:
    """Per run dir, the occurrence list of event_key from pinned_batch (anchored,
    sorted). Empty list for a run where the batch/anchor/event is absent."""
    return [tj.batch_occurrences(rd, pinned_batch, anchor_key).get(event_key, [])
            for rd in run_dirs]


def offset_exact(run_dirs: List[str], a: str, b: str,
                 anchor_key: str = ANCHOR) -> Optional[int]:
    """The exact within-execution offset (a - b) iff it agrees across all
    co-traced runs (cross-run range <= Q). None if never co-traced or non-exact.
    Replaces `correlates`: equality, not std <= eps."""
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    if st is None or st.range > Q:
        return None
    return int(st.mean)  # range <= Q == 0 -> min == max == mean, exact


def offset_window(run_dirs: List[str], a: str, b: str,
                  anchor_key: str = ANCHOR) -> Optional[Tuple[int, int]]:
    """(min, max) of (a - b) over the first co-tracing batch per run; None if
    never co-traced. offset_exact is the range-0 special case (min == max)."""
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    if st is None:
        return None
    return (int(st.min), int(st.max))


def anchor_rigid(run_dirs: List[str], event_key: str,
                 anchor_key: str = ANCHOR) -> bool:
    """e's anchored first-occurrence agrees exactly across runs (range <= Q).
    Replaces std-based `deterministic`."""
    per_run = _anchored_per_run(run_dirs, event_key, anchor_key)
    stats = tv.aggregate(per_run).get(event_key)
    return stats is not None and stats.range <= Q


def check_ordering(run_dirs: List[str], edges: List[Tuple[str, str]],
                   anchor_key: str = ANCHOR) -> Optional[RejectedRule]:
    """Every (parent, child) edge: parent fires no later than child, every run
    (first co-tracing batch per run). RejectedRule on the first violation."""
    for parent, child in edges:
        for rd in run_dirs:
            for bn in tj._batch_names(rd):
                f = tj.batch_firsts(rd, bn, anchor_key)
                if parent in f and child in f:
                    if f[parent] > f[child]:
                        return RejectedRule(
                            "ordering",
                            f"{parent} ({f[parent]}) > {child} ({f[child]})",
                            {"edge": (parent, child), "run": rd, "batch": bn})
                    break  # first co-tracing batch per run
    return None


def check_lock_handoff(run_dirs: List[str], lock_pairs: List[Tuple[str, str]],
                       anchor_key: str = ANCHOR) -> Optional[RejectedRule]:
    """Every (release, acquire) lock pair: release fires no later than the
    matching acquire, every run (first co-tracing batch per run)."""
    for rel, acq in lock_pairs:
        for rd in run_dirs:
            for bn in tj._batch_names(rd):
                f = tj.batch_firsts(rd, bn, anchor_key)
                if rel in f and acq in f:
                    if f[rel] > f[acq]:
                        return RejectedRule(
                            "lock_handoff",
                            f"release {rel} ({f[rel]}) > acquire {acq} ({f[acq]})",
                            {"pair": (rel, acq), "run": rd, "batch": bn})
                    break  # first co-tracing batch per run
    return None


def check_additivity(run_dirs: List[str], chain: List[str],
                     anchor_key: str = ANCHOR) -> Optional[RejectedRule]:
    """offset(chain[0] -> chain[-1]) must equal the sum of consecutive exact
    offsets. Vacuous (None) for a chain of < 3 keys (single segment). A
    non-exact consecutive offset is a gap, not an additivity violation -> None."""
    if len(chain) < 3:
        return None
    parts = []
    for i in range(len(chain) - 1):
        o = offset_exact(run_dirs, chain[i + 1], chain[i], anchor_key)
        if o is None:
            return None  # a gap in the chain: additivity does not apply
        parts.append(o)
    end_to_end = offset_exact(run_dirs, chain[-1], chain[0], anchor_key)
    if end_to_end is None:
        return None
    if end_to_end != sum(parts):
        return RejectedRule(
            "additivity",
            f"offset({chain[0]} -> {chain[-1]}) = {end_to_end} != sum {sum(parts)}",
            {"chain": chain, "parts": parts})
    return None


def additivity_state(run_dirs: List[str], chain: List[str],
                     anchor_key: str = ANCHOR) -> str:
    """Tri-state additivity cross-check (does NOT replace check_additivity).
    "vacuous" (<3 keys), "unverifiable" (a consecutive/end offset not co-traced),
    "violation" (end != sum), else "pass"."""
    if len(chain) < 3:
        return "vacuous"
    parts = []
    for i in range(len(chain) - 1):
        o = offset_exact(run_dirs, chain[i + 1], chain[i], anchor_key)
        if o is None:
            return "unverifiable"
        parts.append(o)
    end = offset_exact(run_dirs, chain[-1], chain[0], anchor_key)
    if end is None:
        return "unverifiable"
    return "pass" if end == sum(parts) else "violation"


def cross_batch_range(run_dirs: List[str], event_key: str,
                      anchor_key: str = ANCHOR) -> int:
    """Max over runs of (max-min) of event_key's anchored value across the batches
    that trace it in that run. 0 => batch-invariant (cross-batch membership safe)."""
    worst = 0
    for rd in run_dirs:
        vals = [tj.batch_firsts(rd, bn, anchor_key)[event_key]
                for bn in tj._batch_names(rd)
                if event_key in tj.batch_firsts(rd, bn, anchor_key)]
        if len(vals) > 1:
            worst = max(worst, max(vals) - min(vals))
    return worst


def coincident(run_dirs: List[str], a: str, b: str,
               anchor_key: str = ANCHOR) -> bool:
    """a and b fire at the SAME anchored ts in every co-traced run -- the exact
    offset between them is 0. Identity-coincidence, equality not statistics."""
    return offset_exact(run_dirs, a, b, anchor_key) == 0
