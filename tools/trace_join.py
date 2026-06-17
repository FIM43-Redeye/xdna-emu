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
import math
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


def event_bands(run_dirs: List[str], keys, anchor_key: str = "1|2|PERF_CNT_2") -> Dict[str, dict]:
    """Per key, the Stats (as dict) of its anchored first-occurrence across runs."""
    per_run: List[Dict[str, int]] = []
    for rd in run_dirs:
        merged: Dict[str, int] = {}
        for bn in _batch_names(rd):
            for k, v in batch_firsts(rd, bn, anchor_key).items():
                merged.setdefault(k, v)   # first batch that observed k
        per_run.append({k: merged[k] for k in keys if k in merged})
    stats = tv.aggregate(per_run)
    return {k: s._asdict() for k, s in stats.items()}


def build_derivability_graph(run_dirs: List[str],
                             anchor_key: str = "1|2|PERF_CNT_2",
                             eps: float = 2.0) -> dict:
    """Build derivability graph: nodes, edges (root->derivable), roots, stochastic_roots.

    Anchor-centric algorithm: deterministic nodes (own anchored std <= eps) get no
    edge. Stochastic nodes are greedily assigned to an existing stochastic root
    (if rigidly linked, std <= eps) or promoted to new roots.
    """
    nodes = set()
    for rd in run_dirs:
        for tile, names in load_active_events(rd).items():
            col, row = tile.split("|")
            for n in names:
                nodes.add(f"{col}|{row}|{n}")
    nodes = sorted(nodes)
    bands = event_bands(run_dirs, nodes, anchor_key)

    # Deterministic = fixed offset from the anchor (own anchored std <= eps).
    stochastic = [n for n in nodes
                  if bands.get(n, {}).get("std", 0.0) > eps]

    # Greedily assign each stochastic node to an existing root, else promote it.
    stochastic_roots: List[str] = []
    edges = []
    for x in stochastic:   # already sorted (nodes is sorted)
        attached = False
        for r in stochastic_roots:
            st = pair_derivability(run_dirs, x, r, anchor_key)
            if st is not None and st.std <= eps:
                edges.append({"from": r, "to": x,
                              "offset": int(round(st.mean)), "std": st.std})
                attached = True
                break
        if not attached:
            stochastic_roots.append(x)

    derivable = {e["to"] for e in edges}
    roots = [n for n in nodes if n not in derivable]
    return {"anchor": anchor_key, "eps": eps, "nodes": nodes, "edges": edges,
            "roots": roots, "stochastic_roots": stochastic_roots, "bands": bands}


class PlannerError(Exception):
    pass


def _split_key(k):
    col, row, name = k.split("|", 2)
    return f"{col}|{row}", name


def synthesize_plan(graph: dict, slot_capacity: int = 8) -> dict:
    always_keys = [graph["anchor"]] + list(graph["stochastic_roots"])
    always_on: Dict[str, List[str]] = collections.defaultdict(list)
    for k in always_keys:
        tile, name = _split_key(k)
        if name not in always_on[tile]:
            always_on[tile].append(name)

    payload: Dict[str, List[str]] = collections.defaultdict(list)
    always_set = set(always_keys)
    for k in graph["nodes"]:
        if k in always_set:
            continue
        tile, name = _split_key(k)
        payload[tile].append(name)

    n_batches = 1
    for tile, on in always_on.items():
        if len(on) > slot_capacity:
            raise PlannerError(
                f"always-on set for tile {tile} needs {len(on)} slots "
                f"({on}) but capacity is {slot_capacity}: overage "
                f"{len(on) - slot_capacity}")
    for tile in set(list(always_on) + list(payload)):
        free = slot_capacity - len(always_on.get(tile, []))
        if payload.get(tile) and free <= 0:
            raise PlannerError(
                f"tile {tile} has no free slots for payload after always-on "
                f"({always_on.get(tile)}); capacity {slot_capacity}")
        if payload.get(tile):
            n_batches = max(n_batches, math.ceil(len(payload[tile]) / free))

    batches = []
    for i in range(n_batches):
        batch: Dict[str, List[str]] = {}
        for tile in set(list(always_on) + list(payload)):
            free = slot_capacity - len(always_on.get(tile, []))
            sl = payload.get(tile, [])[i * free:(i + 1) * free] if free > 0 else []
            batch[tile] = list(always_on.get(tile, [])) + sl
        batches.append(batch)

    return {"slot_capacity": slot_capacity, "anchor": graph["anchor"],
            "always_on": dict(always_on), "batches": batches, "n_batches": n_batches}
