"""Measured leaves: load `fired` facts from captured trace.events.json on disk.

Reuses trace_join's batch loading + anchoring (the per-run `fired` atoms survive on
disk in batch_NN/hw/trace.events.json even though build_derivability_graph discards
them). One `fired(event_key, run_idx, anchored_ts)` per (event, run), from each
event's first co-traced batch -- matching pair_derivability's selection so the
verifier (Task 4) sees a consistent value per (event, run).
"""
from __future__ import annotations
from typing import List, Dict
import trace_join as tj
from inference.facts import Fact, Measured

ANCHOR = "1|2|0|PERF_CNT_2"


def _first_firsts(run_dir: str, anchor_key: str) -> Dict[str, int]:
    """event_key -> anchored_ts, from the first batch that traces it in this run."""
    seen: Dict[str, int] = {}
    for bn in tj._batch_names(run_dir):
        f = tj.batch_firsts(run_dir, bn, anchor_key)
        for ekey, ats in f.items():
            seen.setdefault(ekey, ats)
    return seen


def load_fired(run_dirs: List[str], anchor_key: str = ANCHOR) -> List[Fact]:
    facts: List[Fact] = []
    for run_idx, rd in enumerate(run_dirs):
        for ekey, ats in _first_firsts(rd, anchor_key).items():
            facts.append(Fact("fired", (ekey, run_idx, ats), Measured()))
    return facts


def replication_violations(run_dirs: List[str], anchor_key: str = ANCHOR,
                           eps: float = 2.0) -> List[dict]:
    """Same (event_key, run) across multiple batches must agree within eps."""
    out: List[dict] = []
    for run_idx, rd in enumerate(run_dirs):
        per_batch: Dict[str, List[int]] = {}
        for bn in tj._batch_names(rd):
            f = tj.batch_firsts(rd, bn, anchor_key)
            for ekey, ats in f.items():
                per_batch.setdefault(ekey, []).append(ats)
        for ekey, vals in per_batch.items():
            if len(vals) > 1 and (max(vals) - min(vals)) > eps:
                out.append({"event_key": ekey, "run": run_idx,
                            "values": sorted(vals),
                            "spread": max(vals) - min(vals)})
    return out
