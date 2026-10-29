#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Histogram + percentile summary of rw-access-probe CSV output.

Consumes CSVs emitted by `rw-access-probe --csv <path>`.  Columns:
  index, timestamp_ns, roundtrip_us, value

Reports roundtrip distribution percentiles and (if requested) writes
a PNG histogram.  Also reports the apparent Timer_Low rate using the
wall-clock duration the CSV spans.
"""
import argparse
import csv
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="rw-access-probe CSV path")
    ap.add_argument("--out", default=None, help="PNG histogram output path (optional)")
    args = ap.parse_args()

    indices, ts_ns, rt_us, vals = [], [], [], []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(int(row["index"]))
            ts_ns.append(int(row["timestamp_ns"]))
            rt_us.append(float(row["roundtrip_us"]))
            vals.append(int(row["value"]))

    rt = np.array(rt_us)
    ts = np.array(ts_ns)
    v = np.array(vals, dtype=np.uint32)

    # Compute Timer_Low deltas using uint32 arithmetic to handle wrap.
    deltas = (v[1:].astype(np.int64) - v[:-1].astype(np.int64)) & 0xFFFFFFFF

    wall_ns = ts[-1] - ts[0] if len(ts) >= 2 else 0
    wall_s = wall_ns / 1e9
    # Timer_Low total advance (handles single-wrap correctly; if >1 wrap, breaks).
    total_ticks = int(deltas.sum())

    print(f"== rw-access-probe analysis: {args.csv} ==")
    print(f"N                  = {len(rt)}")
    print(f"")
    print(f"-- roundtrip (us) --")
    print(f"min                = {rt.min():.1f}")
    print(f"p50                = {np.percentile(rt, 50):.1f}")
    print(f"p90                = {np.percentile(rt, 90):.1f}")
    print(f"p99                = {np.percentile(rt, 99):.1f}")
    print(f"p99.9              = {np.percentile(rt, 99.9):.1f}")
    print(f"max                = {rt.max():.1f}")
    print(f"mean               = {rt.mean():.1f}  stddev={rt.std():.1f}")
    print(f"")
    print(f"-- wall-clock --")
    print(f"total duration     = {wall_s:.3f} s")
    if len(rt) >= 2:
        print(f"effective rate     = {len(rt) / wall_s:.0f} reads/s")
    print(f"")
    print(f"-- Timer_Low --")
    print(f"first value        = 0x{v[0]:08x}")
    print(f"last  value        = 0x{v[-1]:08x}")
    print(f"total advance      = {total_ticks:,} ticks")
    if wall_s > 0:
        rate_hz = total_ticks / wall_s
        print(f"apparent rate      = {rate_hz/1e6:.2f} MHz  ({rate_hz:.0f} Hz)")
    if len(deltas) > 0:
        nonzero = deltas[deltas > 0]
        print(f"per-read delta:")
        print(f"  zero-deltas      = {int((deltas == 0).sum())}  ({100*(deltas == 0).mean():.1f}%)")
        if len(nonzero) > 0:
            print(f"  min nonzero      = {int(nonzero.min())}")
            print(f"  p50 nonzero      = {int(np.percentile(nonzero, 50))}")
            print(f"  p99 nonzero      = {int(np.percentile(nonzero, 99))}")
            print(f"  max              = {int(nonzero.max())}")

    if args.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"matplotlib unavailable; skipping --out", file=sys.stderr)
            return
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        # Top: roundtrip histogram
        upper = np.percentile(rt, 99.5)
        axes[0].hist(rt, bins=200, range=(0, upper))
        axes[0].set_xlabel("roundtrip (us)")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"read_aie_reg roundtrip distribution (N={len(rt)}, 0 to p99.5)")
        axes[0].grid(alpha=0.3)
        # Bottom: Timer_Low over time
        axes[1].plot(ts / 1e9, v, marker=".", markersize=1, linestyle="-", linewidth=0.5)
        axes[1].set_xlabel("wall-clock (s)")
        axes[1].set_ylabel("Timer_Low value")
        axes[1].set_title("Timer_Low vs wall-clock")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out, dpi=100, bbox_inches="tight")
        print(f"")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
