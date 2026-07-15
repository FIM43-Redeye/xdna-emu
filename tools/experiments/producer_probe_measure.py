#!/usr/bin/env python3
"""Measure the producer-probe: per-transfer MM2S duration + core MEMORY_STALL +
bank-conflict AREA, steady-state, for idle/apart/collide.

Reuses bankdisc_measure's slot-based interval decode (tid -> event name via
trace_config.json) and its transfer bracket: transfer r = [falling edge of the
STALLED_LOCK interval preceding FINISHED_BD[r], FINISHED_BD[r]).

collide - apart isolates the producer core-vs-MM2S stall rate per transfer.
Level events -> INTERVAL AREA (never record counts); ts, never soc.
"""
import argparse
import statistics
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from bankdisc_measure import load_intervals, area  # noqa: E402


def measure(build_dir: Path, rep: int):
    iv = load_intervals(build_dir / f"perfetto_r{rep}.json",
                        build_dir / "trace_config.json")
    finished = sorted(s for s, _ in iv[("mem", "DMA_MM2S_0_FINISHED_BD")])
    stalls = sorted(iv[("mem", "DMA_MM2S_0_STALLED_LOCK")])
    durations = []
    for f in finished:
        prior = [end for _, end in stalls if end <= f]
        if prior:
            durations.append(f - max(prior))
    res = {
        "n_finished": len(finished),
        "n_bracketed": len(durations),
        "median_dur": statistics.median(durations) if durations else None,
        "core_MEMORY_STALL": area(iv[("core", "MEMORY_STALL")]),
        "conflict_b0": area(iv[("mem", "CONFLICT_DM_BANK_0")]),
        "conflict_b1": area(iv[("mem", "CONFLICT_DM_BANK_1")]),
        "mm2s_starv": area(iv[("mem", "DMA_MM2S_0_MEMORY_STARVATION")]),
    }
    res["conflict"] = res["conflict_b0"] + res["conflict_b1"]
    # Per-transfer normalization uses ALL transfers (every FINISHED_BD), NOT the
    # bracketed subset. The STALLED_LOCK->FINISHED_BD bracket is for DURATION only
    # (a transfer needs both a start and end edge); the MEMORY_STALL / CONFLICT
    # numerators are whole-run areas, so they must divide by all transfers. An
    # earlier version divided the whole-run stall area by n_bracketed, which
    # inflated the rate and falsely implied a boundary/steady-state split -- the
    # stalls are not bracket-localized (they occur during active MM2S streaming).
    n = res["n_finished"] or 1
    res["stall_per_xfer"] = res["core_MEMORY_STALL"] / n
    res["conflict_per_xfer"] = res["conflict"] / n
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="build/experiments/producer-probe")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    root = Path(args.root)

    print(f"{'variant':9} {'run':>3} {'nBD':>3} {'medDur':>6} {'coreSTALL':>9} "
          f"{'conflict':>8} {'stall/xfer':>10} {'conf/xfer':>9} {'starv':>5}")
    print("-" * 74)
    pooled = {}
    for v in ("idle", "apart", "collide"):
        d = root / f"pp_{v}"
        pv = []
        for r in range(1, args.reps + 1):
            if not (d / f"perfetto_r{r}.json").exists():
                continue
            m = measure(d, r)
            pv.append(m)
            print(f"{v:9} {r:>3} {m['n_bracketed']:>3} {str(m['median_dur']):>6} "
                  f"{m['core_MEMORY_STALL']:>9} {m['conflict']:>8} "
                  f"{m['stall_per_xfer']:>10.2f} {m['conflict_per_xfer']:>9.2f} "
                  f"{m['mm2s_starv']:>5}")
        if pv:
            pooled[v] = statistics.median(x["stall_per_xfer"] for x in pv)
        print()

    if {"idle", "apart", "collide"} <= pooled.keys():
        print(f"stall/xfer  idle={pooled['idle']:.2f}  apart={pooled['apart']:.2f}  "
              f"collide={pooled['collide']:.2f}")
        print(f"PRODUCER core-vs-MM2S stall rate = collide - apart = "
              f"{pooled['collide'] - pooled['apart']:.2f} cycles/transfer")
