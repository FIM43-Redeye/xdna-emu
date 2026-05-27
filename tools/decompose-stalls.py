#!/usr/bin/env python3
"""
Stall-decomposition aggregator over multirun-trace-campaign event data.

Reads events.json files per (K, run), pairs DMA_*_START_TASK with
DMA_*_FINISHED_TASK on the same channel to get per-transfer durations and
inter-transfer gaps, then overlays level-event intervals (STREAM_STARVATION,
PORT_RUNNING, etc.) to decompose each transfer's duration into active vs
stall cycles.

Currently captured per K-sweep run on shim (col=*, row=0):
  Slots 0-5: DMA_S2MM_0/S2MM_1/MM2S_0 START/FINISHED (edge events)
  Slot 6:   DMA_S2MM_0_STREAM_STARVATION (level, useful)
  Slot 7:   DMA_S2MM_1_STREAM_STARVATION (level, NOISE -- unused channel)

And on memtile (col=*, row=1):
  Slots 0-7: PORT_RUNNING_0..7 (level)

Output: per-K table of transfer counts, transfer durations (median + range),
inter-transfer gaps, and starvation cycles within transfer windows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def events_to_intervals(cycles: list[int]) -> list[tuple[int, int]]:
    """Group cycles where delta <= 1 into (start, end) intervals.

    Matches src/trace/compare.rs:events_to_intervals().
    """
    if not cycles:
        return []
    cycles = sorted(cycles)
    intervals = []
    start = cycles[0]
    prev = cycles[0]
    for c in cycles[1:]:
        if c - prev > 1:
            intervals.append((start, prev))
            start = c
        prev = c
    intervals.append((start, prev))
    return intervals


def overlap_cycles(window: tuple[int, int], intervals: list[tuple[int, int]]) -> int:
    """Total cycle count where intervals overlap with the [start, end] window."""
    w_start, w_end = window
    total = 0
    for i_start, i_end in intervals:
        lo = max(w_start, i_start)
        hi = min(w_end, i_end)
        if hi >= lo:
            total += hi - lo + 1
    return total


def collect_events_by_name(events: list[dict]) -> dict[str, list[int]]:
    """Bucket events by name, returning sorted soc cycle lists."""
    by_name = defaultdict(list)
    for e in events:
        by_name[e["name"]].append(e["soc"])
    return {n: sorted(cs) for n, cs in by_name.items()}


def decompose_one_run(events_path: Path) -> dict:
    """Decompose one run's events.json into per-channel stall data."""
    data = json.loads(events_path.read_text())
    by_name = collect_events_by_name(data["events"])

    # Per-channel transfer windows: pair START[i] with FINISHED[i].
    channels = [
        ("MM2S_0", "DMA_MM2S_0_START_TASK", "DMA_MM2S_0_FINISHED_TASK",
         "DMA_MM2S_0_STREAM_BACKPRESSURE"),
        ("S2MM_0", "DMA_S2MM_0_START_TASK", "DMA_S2MM_0_FINISHED_TASK",
         "DMA_S2MM_0_STREAM_STARVATION"),
        ("S2MM_1", "DMA_S2MM_1_START_TASK", "DMA_S2MM_1_FINISHED_TASK",
         "DMA_S2MM_1_STREAM_STARVATION"),
    ]
    out = {}
    for cname, start_evt, fin_evt, stall_evt in channels:
        starts = by_name.get(start_evt, [])
        fins = by_name.get(fin_evt, [])
        stall_intervals = events_to_intervals(by_name.get(stall_evt, []))
        n_tasks = min(len(starts), len(fins))
        if n_tasks == 0:
            continue
        per_task = []
        for i in range(n_tasks):
            t_start = starts[i]
            t_fin = fins[i]
            t_dur = t_fin - t_start
            stall_in_window = overlap_cycles((t_start, t_fin), stall_intervals)
            per_task.append({
                "i": i,
                "start_cyc": t_start,
                "fin_cyc": t_fin,
                "duration": t_dur,
                "stall_in_xfer": stall_in_window,
                "active_in_xfer": max(0, t_dur - stall_in_window),
            })
        # Inter-task gaps (FINISHED[i] -> START[i+1]).
        gaps = []
        for i in range(n_tasks - 1):
            gap = starts[i + 1] - fins[i]
            stall_in_gap = overlap_cycles((fins[i], starts[i + 1]), stall_intervals)
            gaps.append({
                "i": i,
                "gap": gap,
                "stall_in_gap": stall_in_gap,
            })
        out[cname] = {
            "tasks": per_task,
            "gaps": gaps,
            "stall_intervals_total": sum(e - s + 1 for s, e in stall_intervals),
            "n_stall_intervals": len(stall_intervals),
        }
    return out


def aggregate_session(session_dir: Path) -> dict:
    """Walk session/runNNN/k*/events.json and decompose."""
    manifest = json.loads((session_dir / "manifest.json").read_text())
    ks = manifest["ks"]
    by_k = defaultdict(list)
    for run_dir in sorted(session_dir.glob("run*")):
        for k in ks:
            ev_path = run_dir / f"k{k}" / "events.json"
            if not ev_path.exists():
                continue
            try:
                by_k[k].append(decompose_one_run(ev_path))
            except Exception as exc:
                print(f"warn: failed {ev_path}: {exc}", file=sys.stderr)
    return {"session": session_dir.name, "ks": ks, "by_k": dict(by_k)}


def median_or_none(xs: list[float]) -> float | None:
    if not xs:
        return None
    return statistics.median(xs)


def print_table(agg: dict) -> None:
    """Print a per-K table summarizing the decomposition."""
    print(f"== stall decomposition: {agg['session']} ==")
    for k in agg["ks"]:
        runs = agg["by_k"].get(k, [])
        if not runs:
            continue
        print(f"\nK={k}  (N runs={len(runs)})")
        # Aggregate across runs: each run gives per-channel per-task lists.
        # We pool per (channel, task_index) across runs and take median.
        # Channels in stable order.
        channels_seen = set()
        for r in runs:
            channels_seen.update(r.keys())
        for ch in ("MM2S_0", "S2MM_0", "S2MM_1"):
            if ch not in channels_seen:
                continue
            print(f"  {ch}:")
            n_tasks = max(len(r.get(ch, {}).get("tasks", [])) for r in runs)
            # Per task: median duration, median stall_in_xfer
            print(f"    {'idx':>3} {'dur_med':>8} {'stall_med':>10} {'active_med':>10}  (cyc)")
            for i in range(n_tasks):
                durs = [r[ch]["tasks"][i]["duration"]
                        for r in runs if ch in r and i < len(r[ch]["tasks"])]
                stalls = [r[ch]["tasks"][i]["stall_in_xfer"]
                          for r in runs if ch in r and i < len(r[ch]["tasks"])]
                actives = [r[ch]["tasks"][i]["active_in_xfer"]
                           for r in runs if ch in r and i < len(r[ch]["tasks"])]
                print(f"    {i:>3} {median_or_none(durs):>8} {median_or_none(stalls):>10} {median_or_none(actives):>10}")
            # Inter-task gaps
            n_gaps = max((len(r.get(ch, {}).get("gaps", [])) for r in runs), default=0)
            if n_gaps:
                print(f"    inter-task gaps:")
                print(f"    {'idx':>3} {'gap_med':>8} {'stall_med':>10}  (cyc)")
                for i in range(n_gaps):
                    gs = [r[ch]["gaps"][i]["gap"]
                          for r in runs if ch in r and i < len(r[ch]["gaps"])]
                    sg = [r[ch]["gaps"][i]["stall_in_gap"]
                          for r in runs if ch in r and i < len(r[ch]["gaps"])]
                    print(f"    {i:>3} {median_or_none(gs):>8} {median_or_none(sg):>10}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path, help="multirun campaign session directory")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of human-readable table")
    args = ap.parse_args()

    agg = aggregate_session(args.session)
    if args.json:
        # Strip per-task detail for terseness; emit medians.
        json.dump(agg, sys.stdout, indent=2, default=str)
        print()
    else:
        print_table(agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
