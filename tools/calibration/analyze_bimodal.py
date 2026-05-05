#!/usr/bin/env python3
"""Characterize the bimodal cycle distribution from a high-rep sweep.

For each N, group cycle measurements into clusters (by gap detection) and
report each cluster's center, width, and probability. Useful for isolating
"fast mode" vs "slow mode" runs when a stochastic artifact creates two
distinct populations of cycle counts.

Usage:
  python3 analyze_bimodal.py path/to/measurements.json
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def cluster_by_gap(values: list, min_gap: int = 100) -> list:
    """Group sorted values into clusters separated by gaps >= min_gap.

    Returns list of clusters, each a list of values.
    """
    if not values:
        return []
    vs = sorted(values)
    clusters = [[vs[0]]]
    for v in vs[1:]:
        if v - clusters[-1][-1] >= min_gap:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return clusters


def predict_period2(n: int, P: int = 201, offset_even: int = 87) -> int:
    """Period-2 baseline prediction.
    cost(N) = P * ceil(N/2) + (offset_even if N even else 0).
    """
    return P * ((n + 1) // 2) + (offset_even if n % 2 == 0 else 0)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    data = json.loads(Path(sys.argv[1]).read_text())
    by_n = defaultdict(list)
    for m in data["measurements"]:
        if m.get("hw_cycles") is not None:
            by_n[m["count"]].append(m["hw_cycles"])

    print(f"{'N':>4s}  {'reps':>4s}  {'predict':>7s}  cluster (count × mean ± width)  "
          f"resid_from_pred")
    print("-" * 80)
    for n in sorted(by_n):
        vals = by_n[n]
        clusters = cluster_by_gap(vals, min_gap=500)
        pred = predict_period2(n)
        cluster_strs = []
        for c in clusters:
            mean = statistics.mean(c)
            spread = max(c) - min(c)
            cluster_strs.append(
                f"{len(c):>2d} × {mean:7.0f} ±{spread:>2d}"
            )
        # Per-cluster residual from period-2 baseline.
        resid_strs = [f"{statistics.mean(c) - pred:+.0f}" for c in clusters]
        print(f"{n:>4d}  {len(vals):>4d}  {pred:>7d}  "
              f"{'  |  '.join(cluster_strs):<50s}  "
              f"{'  |  '.join(resid_strs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
