#!/usr/bin/env python3
"""Characterize HW trace nondeterminism for #140.

Measures, across N repeats of one kernel, how much each trace event's timing
varies run-to-run, classifies events deterministic vs stochastic from the
variance itself, and decomposes the result. HW-only: this tool reads decoded
artifacts; it never runs the emulator.

Two representations, picked by the metric constraint (2026-06-16 finding):
  - Held-level events (PORT_RUNNING*, PORT_STALLED*) are measured as SPANS from
    the perfetto B/E json -- never by counting frame-records in events.json.
  - Milestone events (DMA START/FINISHED, STREAM_STARVATION, LOCK_STALL, ...)
    are measured as re-anchored SoC point timestamps from events.json.
Both are re-anchored within-tile only.
"""
import argparse
import collections
import json
import statistics as st
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

LEVEL_FAMILIES = ("PORT_RUNNING", "PORT_STALLED")


def is_level(name: str) -> bool:
    return any(name.startswith(f) for f in LEVEL_FAMILIES)


def load_milestone_events(events_path: str) -> Dict[Tuple, List[int]]:
    """events.json -> {(col,row,name): [re-anchored soc, ...]}, milestones only.

    Anchor is per-tile: the minimum soc over that tile's milestone events.
    """
    doc = json.loads(Path(events_path).read_text())
    by_tile_raw: Dict[Tuple[int, int], List[dict]] = collections.defaultdict(list)
    for e in doc.get("events", []):
        if is_level(e["name"]):
            continue
        by_tile_raw[(e["col"], e["row"])].append(e)
    out: Dict[Tuple, List[int]] = collections.defaultdict(list)
    for (col, row), evs in by_tile_raw.items():
        anchor = min(e["soc"] for e in evs)
        for e in evs:
            out[(col, row, e["name"])].append(e["soc"] - anchor)
    for k in out:
        out[k].sort()
    return dict(out)


Stats = namedtuple("Stats", "n mean std min max range")


def aggregate(per_run: List[Dict]) -> Dict:
    """[{key: scalar}, ...] -> {key: Stats} over the runs where the key appears."""
    samples: Dict = collections.defaultdict(list)
    for run in per_run:
        for key, val in run.items():
            samples[key].append(val)
    out: Dict = {}
    for key, vals in samples.items():
        mean = st.mean(vals)
        std = st.pstdev(vals) if len(vals) > 1 else 0.0
        out[key] = Stats(n=len(vals), mean=mean, std=std,
                         min=min(vals), max=max(vals), range=max(vals) - min(vals))
    return out


def classify(s: Stats, eps: float = 2.0) -> str:
    return "deterministic" if s.std <= eps else "stochastic"


def _group_spans(pairs: List[Tuple[int, int]], idle_gap: int = 2) -> List[int]:
    """[(begin,end), ...] (sorted) -> merged span durations; merge across gaps <= idle_gap."""
    pairs = sorted(pairs)
    spans: List[int] = []
    cur_b, cur_e = None, None
    for b, e in pairs:
        if cur_b is None:
            cur_b, cur_e = b, e
        elif b - cur_e <= idle_gap:
            cur_e = max(cur_e, e)
        else:
            spans.append(cur_e - cur_b)
            cur_b, cur_e = b, e
    if cur_b is not None:
        spans.append(cur_e - cur_b)
    return spans


def load_spans_from_events(evs: List[dict], name_map: Dict[Tuple, str],
                           idle_gap: int = 2) -> Dict[str, List[int]]:
    """Perfetto B/E event list -> {name: [span_duration,...]} per mapped lane."""
    stacks: Dict[Tuple, List[int]] = collections.defaultdict(list)
    pairs: Dict[Tuple, List[Tuple[int, int]]] = collections.defaultdict(list)
    for e in sorted(evs, key=lambda x: (x.get("ts", 0), 0 if x.get("ph") == "E" else 1)):
        ph = e.get("ph")
        if ph not in ("B", "E"):
            continue
        lane = (e.get("pid"), e.get("tid"))
        if ph == "B":
            stacks[lane].append(e.get("ts"))
        elif stacks[lane]:
            b = stacks[lane].pop()
            pairs[lane].append((b, e.get("ts")))
    out: Dict[str, List[int]] = {}
    for lane, ps in pairs.items():
        name = name_map.get(lane)
        if name is None:
            continue
        out.setdefault(name, [])
        out[name].extend(_group_spans(ps, idle_gap))
    return out


import re

# pkt_type codes, matching tools/parse-trace.py _PT_NAME_TO_CODE.
_PT_NAME_TO_CODE = {"core": 0, "mem": 1, "shim": 2, "memtile": 3}


def build_lane_name_map(perfetto_events: List[dict], events_path: str) -> Dict[Tuple, str]:
    """{(pid,tid): event_name} for perfetto lanes.

    Authoritative recipe (derived from tools/parse-trace.py, not reverse-
    engineered): perfetto names are empty, so naming is recovered from
      (1) pid -> pkt_type via the perfetto `process_name` metadata: the leading
          alphabetic token of args.name (robust to both "shim(0,1)" and
          "shim_trace for tile0,1"); 'memtile' wins over 'mem' (greedy regex).
      (2) (pkt_type, slot) -> name from the run's events.json records.
    Then lane (pid, tid) -> name where pkt_type = pid_to_pkt[pid], slot = tid.
    """
    pid_to_pkt: Dict[int, int] = {}
    for e in perfetto_events:
        if e.get("ph") != "M" or e.get("name") != "process_name":
            continue
        nm = (e.get("args", {}) or {}).get("name", "").strip()
        m = re.match(r"[a-z]+", nm)
        if m and m.group(0) in _PT_NAME_TO_CODE:
            pid_to_pkt[e.get("pid")] = _PT_NAME_TO_CODE[m.group(0)]
    doc = json.loads(Path(events_path).read_text())
    pktslot_to_name = {(ev["pkt_type"], ev["slot"]): ev["name"]
                       for ev in doc.get("events", [])}
    out: Dict[Tuple, str] = {}
    for pid, pkt in pid_to_pkt.items():
        for (pk, slot), name in pktslot_to_name.items():
            if pk == pkt:
                out[(pid, slot)] = name
    return out


def load_spans(perfetto_path: str, events_path: str,
               idle_gap: int = 2) -> Dict[str, List[int]]:
    doc = json.loads(Path(perfetto_path).read_text())
    evs = doc["traceEvents"] if isinstance(doc, dict) and "traceEvents" in doc else doc
    name_map = build_lane_name_map(evs, events_path)
    return load_spans_from_events(evs, name_map, idle_gap)


def check_span_law(spans: Dict[str, List[int]], words: int = 64) -> Dict[str, Tuple[int, bool]]:
    return {name: (sum(durs), sum(durs) == words) for name, durs in spans.items()}


def decompose(classified: Dict, law: Dict = None) -> Dict:
    """Split classified events into deterministic vs stochastic buckets.

    Args:
        classified: {key: (Stats, classification_string)}
        law: optional {name: (sum, ok)} span-sum law dict from check_span_law

    Returns:
        {
            "n_deterministic": count,
            "n_stochastic": count,
            "stochastic_keys": [key, ...],
            "deterministic_keys": [key, ...],
            "law_violations": [name, ...],
        }
    """
    law = law or {}
    det = [k for k, (_, c) in classified.items() if c == "deterministic"]
    sto = [k for k, (_, c) in classified.items() if c == "stochastic"]
    violations = [name for name, (s, ok) in law.items() if not ok]
    return {
        "n_deterministic": len(det),
        "n_stochastic": len(sto),
        "stochastic_keys": sto,
        "deterministic_keys": det,
        "law_violations": violations,
    }


def format_report(decomp: Dict, classified: Dict, law: Dict) -> str:
    """Format a human-readable markdown report.

    Args:
        decomp: output of decompose()
        classified: {key: (Stats, classification_string)}
        law: {name: (sum, ok)} span-sum law dict

    Returns:
        markdown string
    """
    lines = ["# DMA nondeterminism characterization — add_one_using_dma", ""]
    lines.append(f"- deterministic events: {decomp['n_deterministic']}")
    lines.append(f"- stochastic events:    {decomp['n_stochastic']}")
    lines.append("")
    lines.append("## span-sum word law (held-level ports)")
    for name in sorted(law):
        s, ok = law[name]
        flag = "OK" if ok else "VIOLATION"
        lines.append(f"- {name}: sum={s} {flag}")
    if decomp["law_violations"]:
        lines.append("")
        lines.append(f"**Law violations (real bug, not noise): {decomp['law_violations']}**")
    lines.append("")
    lines.append("## events by variance (descending std)")
    for key, (s, c) in sorted(classified.items(), key=lambda kv: -kv[1][0].std):
        lines.append(f"- {key} [{c}] n={s.n} mean={s.mean:.0f} std={s.std:.1f} "
                     f"min={s.min} max={s.max} range={s.range}")
    return "\n".join(lines) + "\n"


def report_json(decomp: Dict, classified: Dict, law: Dict) -> Dict:
    """Format a machine-readable JSON report.

    Args:
        decomp: output of decompose()
        classified: {key: (Stats, classification_string)}
        law: {name: (sum, ok)} span-sum law dict

    Returns:
        dict (serializable to JSON)
    """
    return {
        "decomposition": decomp,
        "law": {k: {"sum": s, "ok": ok} for k, (s, ok) in law.items()},
        "events": {"|".join(map(str, k)): {**s._asdict(), "class": c}
                   for k, (s, c) in classified.items()},
    }
