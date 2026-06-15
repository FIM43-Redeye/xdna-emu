#!/usr/bin/env python3
"""
port-cadence-baseline.py -- Aggregate a multirun-trace-campaign --kernels
session into a per-kernel, per-port PORT_RUNNING / STREAM_STARVATION cadence
baseline for the #140 DMA-delivery calibration.

For each kernel and each traced tile port (PORT_RUNNING_*, *_STREAM_STARVATION),
it extracts the pulse timeline from every run's events.json, groups pulses into
sub-bursts (consecutive within --gap cycles), and reports:

  - interval (sub-burst) count across runs: mean / std / distinct values
    -> a std of 0 with one distinct value == deterministic cadence (the
       PORT_RUNNING family); nonzero == stochastic (the shim STARVATION axis).
  - the modal sub-burst gap pattern (the calibration target shape).

This is the *target* side; compare an EMU capture's cadence against it.

Usage:
  ./tools/port-cadence-baseline.py <session-dir>
  ./tools/port-cadence-baseline.py <session-dir> --gap 3 --json out.json
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_events(p: Path):
    d = json.load(open(p))
    return d.get("events", [])


def pulses_for(events, col, row, name):
    """Sorted timestamps for one (col,row,event-name)."""
    return sorted(e["ts"] for e in events
                  if e["col"] == col and e["row"] == row and e["name"] == name)


def subbursts(ts, gap):
    """Group raw pulse timestamps into sub-bursts: a new burst starts when the
    inter-pulse delta exceeds `gap`. Returns (n_bursts, gaps_between_bursts)
    relative to the first pulse."""
    if not ts:
        return 0, []
    t0 = ts[0]
    rel = [t - t0 for t in ts]
    bursts = []
    s = p = rel[0]
    for t in rel[1:]:
        if t - p > gap:
            bursts.append((s, p))
            s = t
        p = t
    bursts.append((s, p))
    gaps = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
    return len(bursts), gaps


PORT_KINDS = ("PORT_RUNNING_", "STREAM_STARVATION", "PORT_STALLED_", "PORT_IDLE_")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("session", type=Path, help="campaign session dir")
    ap.add_argument("--gap", type=int, default=2,
                    help="max inter-pulse delta within a sub-burst (default 2)")
    ap.add_argument("--json", type=Path, default=None,
                    help="write full per-(kernel,tile,port) stats as JSON")
    ap.add_argument("--tiles", default=None,
                    help="restrict to comma-separated col:row tiles "
                         "(default: all traced tiles)")
    args = ap.parse_args()

    tile_filter = None
    if args.tiles:
        tile_filter = {tuple(int(x) for x in t.split(":"))
                       for t in args.tiles.split(",")}

    # kernel-label -> list of events.json paths (one per run)
    runs_by_kernel = defaultdict(list)
    for run_dir in sorted(args.session.glob("run*")):
        for kdir in sorted(run_dir.iterdir()):
            ej = kdir / "events.json"
            meta = kdir / "meta.json"
            if not ej.exists():
                continue
            if meta.exists() and not json.load(open(meta)).get("ok"):
                continue
            runs_by_kernel[kdir.name].append(ej)

    if not runs_by_kernel:
        print(f"no usable runs under {args.session}")
        return 1

    out = {}
    for kernel in sorted(runs_by_kernel):
        paths = runs_by_kernel[kernel]
        # discover (col,row,name) port events from the first run
        first = load_events(paths[0])
        ports = sorted({(e["col"], e["row"], e["name"]) for e in first
                        if any(k in e["name"] for k in PORT_KINDS)})
        print(f"\n{'='*78}\n{kernel}   ({len(paths)} runs)\n{'='*78}")
        out[kernel] = {"n_runs": len(paths), "ports": {}}
        for (col, row, name) in ports:
            if tile_filter and (col, row) not in tile_filter:
                continue
            counts = []
            gap_patterns = []
            for ej in paths:
                ev = load_events(ej)
                n, gaps = subbursts(pulses_for(ev, col, row, name), args.gap)
                counts.append(n)
                gap_patterns.append(tuple(gaps))
            uniq = sorted(set(counts))
            std = statistics.pstdev(counts) if len(counts) > 1 else 0.0
            modal_gap = Counter(gap_patterns).most_common(1)[0]
            verdict = "DETERMINISTIC" if std == 0 and len(uniq) == 1 else "stochastic"
            print(f"  ({col},{row}) {name:<28} "
                  f"intervals mean={statistics.mean(counts):.2f} std={std:.2f} "
                  f"distinct={uniq}  [{verdict}]")
            print(f"        modal gaps ({modal_gap[1]}/{len(paths)} runs): "
                  f"{list(modal_gap[0])}")
            out[kernel]["ports"][f"{col},{row},{name}"] = {
                "interval_counts": counts,
                "mean": statistics.mean(counts),
                "pstdev": std,
                "distinct": uniq,
                "verdict": verdict,
                "modal_gap_pattern": list(modal_gap[0]),
                "modal_gap_support": f"{modal_gap[1]}/{len(paths)}",
            }

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
