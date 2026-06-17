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
