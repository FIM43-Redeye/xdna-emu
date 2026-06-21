"""Empirical verifiers: correlates / deterministic / coincident.

Seeded by the existing cross-run std machinery -- correlates wraps
trace_join.pair_derivability (std of a.ts - b.ts), deterministic wraps
trace_variance.aggregate on the anchored ts. A rule is a hypothesis paired with a
verifier; for these numeric rules the rule body and the verifier coincide, so
self-verification falls out. A failed rule is never used and is itself a finding
(RejectedRule).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import trace_join as tj
import trace_variance as tv

ANCHOR = "1|2|0|PERF_CNT_2"
EPS = 2.0


@dataclass
class Rule:
    name: str
    verify: Callable


@dataclass
class RejectedRule:
    name: str
    reason: str
    evidence: dict


def correlates(run_dirs: List[str], a: str, b: str,
               anchor_key: str = ANCHOR, eps: float = EPS) -> Optional[int]:
    """Offset int(round(mean(a-b))) if std(a-b) <= eps across co-traced runs."""
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    if st is None or st.std > eps:
        return None
    return int(round(st.mean))


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


def deterministic(run_dirs: List[str], event_key: str,
                  anchor_key: str = ANCHOR, eps: float = EPS) -> bool:
    per_run = _anchored_per_run(run_dirs, event_key, anchor_key)
    stats = tv.aggregate(per_run).get(event_key)
    return stats is not None and stats.std <= eps


def coincident(run_dirs: List[str], a: str, b: str,
               anchor_key: str = ANCHOR, eps: float = EPS) -> bool:
    """a and b fire at identical anchored ts (within eps) in every co-traced run."""
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    return st is not None and st.std <= eps and abs(st.mean) <= eps


def verify_offset_stable(run_dirs: List[str], a: str, b: str,
                         anchor_key: str = ANCHOR,
                         eps: float = EPS) -> Tuple[bool, Optional[RejectedRule]]:
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    if st is not None and st.std <= eps:
        return True, None
    reason = "never co-traced" if st is None else f"std {st.std:.2f} > eps {eps}"
    return False, RejectedRule(
        name="offset_stable",
        reason=reason,
        evidence={"pair": (a, b), "stats": None if st is None else st._asdict()})
