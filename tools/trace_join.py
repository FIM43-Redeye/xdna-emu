#!/usr/bin/env python3
"""Derivability-driven cross-batch trace join for #140.

Merges a kernel sweep's per-batch HW traces into one complete every-event
trace. Deterministic/derivable events are placed exactly; stochastic DMA
milestones are placed as a real observed sample carrying a measured band.
HW-only: reads decoded events.json artifacts, never runs the emulator.

See docs/superpowers/specs/2026-06-16-cross-batch-trace-join-design.md.
"""
import argparse
import collections
import functools
import glob as _glob
import json
import math
import statistics as _st
import sys
from pathlib import Path
from typing import Dict, List, Optional
import trace_variance as tv


def _key(col, row, pkt_type, name) -> str:
    return f"{col}|{row}|{pkt_type}|{name}"


def _tile(col, row, pkt_type) -> str:
    return f"{col}|{row}|{pkt_type}"


def load_active_events(run_dir: str) -> Dict[str, set]:
    """{"col|row|pkt_type": {event_name,...}} — fired events per tile-module."""
    out: Dict[str, set] = collections.defaultdict(set)
    for p in sorted(_glob.glob(str(Path(run_dir) / "batch_*" / "hw" / "trace.events.json"))):
        for e in json.loads(Path(p).read_text()).get("events", []):
            out[_tile(e["col"], e["row"], e["pkt_type"])].add(e["name"])
    return dict(out)


def anchored_firsts(events: List[dict], anchor_key: str = "1|2|0|PERF_CNT_2") -> Dict[str, int]:
    """First-occurrence (soc - anchor_soc) per "col|row|pkt_type|name" for one batch.

    Returns {} if the anchor event never fired in this batch.
    """
    firsts: Dict[str, int] = {}
    for e in events:
        k = _key(e["col"], e["row"], e["pkt_type"], e["name"])
        if k not in firsts or e["soc"] < firsts[k]:
            firsts[k] = e["soc"]
    if anchor_key not in firsts:
        return {}
    anchor = firsts[anchor_key]
    return {k: v - anchor for k, v in firsts.items()}


@functools.lru_cache(maxsize=None)
def batch_firsts(run_dir: str, batch_name: str,
                 anchor_key: str = "1|2|0|PERF_CNT_2") -> Dict[str, int]:
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
                      anchor_key: str = "1|2|0|PERF_CNT_2") -> Optional[tv.Stats]:
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


def event_bands(run_dirs: List[str], keys, anchor_key: str = "1|2|0|PERF_CNT_2") -> Dict[str, dict]:
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
                             anchor_key: str = "1|2|0|PERF_CNT_2",
                             eps: float = 2.0) -> dict:
    """Build derivability graph: nodes, edges (root->derivable), roots, stochastic_roots.

    Anchor-centric algorithm: deterministic nodes (own anchored std <= eps) get no
    edge. Stochastic nodes are greedily assigned to an existing stochastic root
    (if rigidly linked, std <= eps) or promoted to new roots.
    """
    nodes = set()
    for rd in run_dirs:
        for tile, names in load_active_events(rd).items():   # tile == "col|row|pkt"
            for n in names:
                nodes.add(f"{tile}|{n}")
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


class JoinError(Exception):
    pass


def _split_key(k):
    col, row, pkt, name = k.split("|", 3)
    return f"{col}|{row}|{pkt}", name   # (tile-module key, name)


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


def join_run(run_dir: str, graph: dict, eps: float = 2.0) -> List[dict]:
    """Merge per-batch traces into one sorted record list with placement gates.

    For each batch under run_dir, anchor every event; classify each
    col|row|pkt_type|name as "stochastic" (a stochastic_root), "derivable"
    (has an incoming edge), else "deterministic". Attach band for stochastic,
    predictor for derivable.

    Reconcile multi-batch observations: deterministic/derivable keys must agree
    within eps across batches (raises JoinError on spread > eps); stochastic
    keys keep each batch's sample as a separate record tagged source_batch.

    Returns records sorted by ts_anchored.
    """
    anchor_key = graph["anchor"]
    stochastic = set(graph["stochastic_roots"])
    incoming = {e["to"]: e for e in graph["edges"]}
    bands = graph.get("bands", {})

    # gather per-key observations:
    # key -> list of (batch_idx, ts_anchored, slot, col, row, pkt_type, name)
    obs: Dict[str, List[tuple]] = collections.defaultdict(list)
    for p in sorted(_glob.glob(str(Path(run_dir) / "batch_*" / "hw" / "trace.events.json"))):
        batch_idx = int(Path(p).parent.parent.name.split("_")[1])
        events = json.loads(Path(p).read_text()).get("events", [])
        firsts = anchored_firsts(events, anchor_key)
        if not firsts:
            continue
        slot_of: Dict[str, int] = {}
        for e in events:
            k = _key(e["col"], e["row"], e["pkt_type"], e["name"])
            slot_of.setdefault(k, e.get("slot"))
        for k, ts in firsts.items():
            col, row, pkt, name = k.split("|", 3)
            obs[k].append((batch_idx, ts, slot_of.get(k), int(col), int(row), int(pkt), name))

    records: List[dict] = []
    for k, samples in obs.items():
        c, r, pt, nm = samples[0][3], samples[0][4], samples[0][5], samples[0][6]
        if k in stochastic:
            cls, pred, band = "stochastic", None, bands.get(k)
        elif k in incoming:
            cls = "derivable"
            pred = {"name": incoming[k]["from"], "offset": incoming[k]["offset"]}
            band = None
        else:
            cls, pred, band = "deterministic", None, None

        if cls == "stochastic":
            for (bi, ts, slot, c2, r2, pt2, nm2) in samples:
                records.append({"col": c2, "row": r2, "pkt_type": pt2, "name": nm2,
                                "slot": slot,
                                "ts_anchored": ts, "source_batch": bi,
                                "class": cls, "predictor": pred, "band": band})
        else:
            ts_vals = [s[1] for s in samples]
            if max(ts_vals) - min(ts_vals) > eps:
                raise JoinError(
                    f"{cls} event {k} spread {max(ts_vals) - min(ts_vals)} > eps "
                    f"{eps} across batches {[s[0] for s in samples]}")
            bi, _, slot = samples[0][0], None, samples[0][2]
            records.append({"col": c, "row": r, "pkt_type": pt, "name": nm,
                            "slot": slot,
                            "ts_anchored": int(_st.median(ts_vals)),
                            "source_batch": bi, "class": cls,
                            "predictor": pred, "band": band})

    records.sort(key=lambda r: (r["ts_anchored"], r["col"], r["row"], r["pkt_type"], r["name"]))
    return records


def to_perfetto(records: List[dict]) -> dict:
    """Emit records as Perfetto traceEvents JSON.

    One instant event ("ph":"i") per record at ts_anchored, with pid=col*100+row,
    name, and args carrying class/source_batch/band/predictor.
    """
    evs = []
    for r in records:
        evs.append({"ph": "i", "ts": r["ts_anchored"], "name": r["name"],
                    "pid": r["col"] * 100 + r["row"], "tid": r.get("slot") or 0,
                    "s": "t",
                    "args": {"class": r["class"], "source_batch": r["source_batch"],
                             "band": r.get("band"), "predictor": r.get("predictor")}})
    return {"traceEvents": evs}


def main(argv=None) -> int:
    """CLI for cross-batch trace join.

    Builds derivability graph from all matched runs, synthesizes plan, joins
    --join-run (default: first matched), writes four artifacts.
    Returns 1 with stderr message if no runs match glob.
    """
    ap = argparse.ArgumentParser(description="Cross-batch trace join (#140)")
    ap.add_argument("--runs-glob", required=True, help="glob of sweep run dirs")
    ap.add_argument("--join-run", default=None, help="run dir to merge (default: first match)")
    ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--slot-capacity", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    runs = sorted(d for d in _glob.glob(args.runs_glob) if Path(d).is_dir())
    if not runs:
        print(f"no run dirs matched {args.runs_glob}", file=sys.stderr)
        return 1

    graph = build_derivability_graph(runs, eps=args.eps)
    try:
        plan = synthesize_plan(graph, slot_capacity=args.slot_capacity)
    except PlannerError as e:
        print(f"planner panic: {e}", file=sys.stderr)
        return 1

    join_run_dir = args.join_run or runs[0]
    records = join_run(join_run_dir, graph, eps=args.eps)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "derivability-graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    (args.out / "batch-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (args.out / "merged.events.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "merged.perfetto.json").write_text(json.dumps(to_perfetto(records)) + "\n")
    print(f"wrote {args.out}: {len(records)} events, {len(graph['stochastic_roots'])} stochastic roots, "
          f"plan n_batches={plan['n_batches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
