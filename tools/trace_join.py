#!/usr/bin/env python3
"""Derivability-driven cross-batch trace join for #140.

Merges a kernel sweep's per-batch HW traces into one complete every-event
trace. Deterministic/derivable events are placed exactly; stochastic DMA
milestones are placed as a real observed sample carrying a measured band.
HW-only: reads decoded events.json artifacts, never runs the emulator.

See docs/superpowers/specs/2026-06-16-cross-batch-trace-join-design.md.
"""
import collections
import functools
import glob as _glob
import json
from pathlib import Path
from typing import Dict, List, Optional
import trace_variance as tv


def _key(col, row, name) -> str:
    return f"{col}|{row}|{name}"


def _tile(col, row) -> str:
    return f"{col}|{row}"


def load_active_events(run_dir: str) -> Dict[str, set]:
    """{"col|row": {event_name,...}} — fired events per tile, unioned over batches."""
    out: Dict[str, set] = collections.defaultdict(set)
    for p in sorted(_glob.glob(str(Path(run_dir) / "batch_*" / "hw" / "trace.events.json"))):
        for e in json.loads(Path(p).read_text()).get("events", []):
            out[_tile(e["col"], e["row"])].add(e["name"])
    return dict(out)


def anchored_firsts(events: List[dict], anchor_key: str = "1|2|PERF_CNT_2") -> Dict[str, int]:
    """First-occurrence (soc - anchor_soc) per "col|row|name" for one batch.

    Returns {} if the anchor event never fired in this batch.
    """
    firsts: Dict[str, int] = {}
    for e in events:
        k = _key(e["col"], e["row"], e["name"])
        if k not in firsts or e["soc"] < firsts[k]:
            firsts[k] = e["soc"]
    if anchor_key not in firsts:
        return {}
    anchor = firsts[anchor_key]
    return {k: v - anchor for k, v in firsts.items()}


@functools.lru_cache(maxsize=None)
def batch_firsts(run_dir: str, batch_name: str,
                 anchor_key: str = "1|2|PERF_CNT_2") -> Dict[str, int]:
    # Memoized: the O(nodes^2) graph build calls this repeatedly for the same
    # batch. Callers treat the returned dict as read-only. Files do not change
    # mid-run, so caching is sound.
    p = Path(run_dir) / batch_name / "hw" / "trace.events.json"
    if not p.exists():
        return {}
    return anchored_firsts(json.loads(p.read_text()).get("events", []), anchor_key)


def _batch_names(run_dir: str) -> List[str]:
    return sorted(Path(p).parent.parent.name
                  for p in _glob.glob(str(Path(run_dir) / "batch_*" / "hw" / "trace.events.json")))


def pair_derivability(run_dirs: List[str], key_x: str, key_s: str,
                      anchor_key: str = "1|2|PERF_CNT_2") -> Optional[tv.Stats]:
    """Stats of (X - S) within-execution across runs; None if never co-traced."""
    diffs: List[Dict[str, int]] = []
    for rd in run_dirs:
        for bn in _batch_names(rd):
            f = batch_firsts(rd, bn, anchor_key)
            if key_x in f and key_s in f:
                diffs.append({"d": f[key_x] - f[key_s]})
                break  # first co-tracing batch in this run
    if not diffs:
        return None
    return tv.aggregate(diffs)["d"]
