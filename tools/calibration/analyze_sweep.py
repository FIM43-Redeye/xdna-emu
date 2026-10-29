#!/usr/bin/env python3
"""Analyze a calibration sweep: extract per-packet slopes, payload decomposition,
and (for grid sweeps) per-tile / per-distance breakdown.

Reads measurements JSON from run_sweep.py and prints:
  1. Per-(kind, target, payload) slope: cycles per packet at each configuration.
  2. For kinds swept across payload (blockwrite): a second-order fit
     slope_pkt(payload) = base + per_word * payload.
  3. For grid sweeps that vary target_col/target_row: per-tile slopes plus a
     fit against Manhattan distance from the anchor tile.

Usage:
  python3 analyze_sweep.py path/to/measurements.json
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def least_squares(xs: list, ys: list) -> tuple:
    """Return (slope, intercept) from a simple linear regression."""
    n = len(xs)
    if n < 2:
        return None, None
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None, None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def r_squared(xs: list, ys: list, slope: float, intercept: float) -> float:
    if slope is None:
        return float("nan")
    mean_y = sum(ys) / len(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def measurement_target_key(m: dict) -> tuple:
    """Return a tuple identifying the target tile, preferring explicit
    (target_col, target_row) if present, else the legacy --target shorthand."""
    if "target_col" in m or "target_row" in m:
        return ("xy", m.get("target_col", 0), m.get("target_row", 2))
    return ("named", m.get("target", "compute"))


def measurement_anchor_key(m: dict) -> tuple:
    if "anchor_col" in m or "anchor_row" in m:
        return (m.get("anchor_col", 0), m.get("anchor_row", 2))
    return (0, 2)


def fmt_target(key: tuple) -> str:
    if key[0] == "xy":
        return f"({key[1]},{key[2]})"
    return key[1]


def slopes_per_config(measurements: list) -> dict:
    """Group measurements and fit per-(kind, target, anchor, payload) slopes."""
    groups = defaultdict(lambda: defaultdict(list))
    for m in measurements:
        if m.get("hw_cycles") is None:
            continue
        key = (m["kind"], measurement_target_key(m),
               measurement_anchor_key(m), m["payload"])
        groups[key][m["count"]].append(m["hw_cycles"])

    fits = {}
    for key, counts in groups.items():
        pairs = [(n, statistics.mean(c)) for n, c in sorted(counts.items())]
        if len(pairs) < 2:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        slope, intercept = least_squares(xs, ys)
        fits[key] = {
            "slope": slope,
            "intercept": intercept,
            "r2": r_squared(xs, ys, slope, intercept),
            "n_points": len(pairs),
            "n_reps": min(len(c) for c in counts.values()),
            "raw_counts": dict(counts),
        }
    return fits


def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    data = json.loads(Path(sys.argv[1]).read_text())
    fits = slopes_per_config(data["measurements"])

    # ===== Section 1: per-config slopes =====
    print("=" * 80)
    print("Per-(kind, target, anchor, payload) slope: marginal cycles per packet")
    print("=" * 80)
    for (kind, target_key, anchor_key, payload), fit in sorted(fits.items()):
        target = fmt_target(target_key)
        anchor = f"({anchor_key[0]},{anchor_key[1]})"
        label = f"{kind} target={target} anchor={anchor}"
        if payload:
            label += f" payload={payload}"
        print(f"  {label:50s}  slope={fit['slope']:8.2f}  "
              f"intercept={fit['intercept']:+9.1f}  R^2={fit['r2']:.5f}  "
              f"n={fit['n_points']}  reps={fit['n_reps']}")

    # ===== Section 2: per-word decomposition =====
    by_kind = defaultdict(list)
    for (kind, target_key, anchor_key, payload), fit in fits.items():
        if payload:
            by_kind[kind].append((payload, fit["slope"]))

    if any(len({p for p, _ in v}) >= 2 for v in by_kind.values()):
        print()
        print("=" * 80)
        print("Per-word cost decomposition (kinds swept across payload):")
        print("=" * 80)
        for kind, entries in sorted(by_kind.items()):
            unique_payloads = sorted({p for p, _ in entries})
            if len(unique_payloads) < 2:
                continue
            # Average slopes across (target, anchor) at each payload.
            by_payload = defaultdict(list)
            for p, s in entries:
                by_payload[p].append(s)
            xs = sorted(by_payload)
            ys = [statistics.mean(by_payload[p]) for p in xs]
            per_word, base = least_squares(xs, ys)
            r2 = r_squared(xs, ys, per_word, base)
            print(f"  {kind:12s}  base={base:7.2f} cyc/pkt  "
                  f"per_word={per_word:6.3f} cyc/word  R^2={r2:.5f}")
            for p, s in zip(xs, ys):
                pred = per_word * p + base
                print(f"      payload={p:3d}  measured_slope={s:8.2f}  "
                      f"fit={pred:7.2f}  resid={s-pred:+6.2f}")

    # ===== Section 3: per-tile / distance breakdown =====
    # Find configs where target_col/target_row varies for a fixed kind/anchor.
    tile_groups = defaultdict(dict)  # (kind, anchor, payload) -> (col,row) -> fit
    for (kind, target_key, anchor_key, payload), fit in fits.items():
        if target_key[0] == "xy":
            tile_groups[(kind, anchor_key, payload)][(target_key[1], target_key[2])] = fit

    grid_groups = {k: v for k, v in tile_groups.items() if len(v) >= 2}
    if grid_groups:
        print()
        print("=" * 80)
        print("Per-tile slopes and distance fit (Manhattan dist from anchor):")
        print("=" * 80)
        for (kind, anchor_key, payload), tiles in sorted(grid_groups.items()):
            label = f"{kind} anchor=({anchor_key[0]},{anchor_key[1]})"
            if payload:
                label += f" payload={payload}"
            print(f"  ---- {label} ----")
            print(f"  {'tile':>8s}  {'dist':>4s}  {'slope':>8s}  "
                  f"{'intercept':>10s}  {'R^2':>7s}")
            rows = []
            for tile in sorted(tiles):
                fit = tiles[tile]
                d = manhattan(tile, anchor_key)
                rows.append((tile, d, fit["slope"], fit["intercept"], fit["r2"]))
                print(f"  ({tile[0]},{tile[1]})  {d:4d}  "
                      f"{fit['slope']:8.2f}  {fit['intercept']:+10.1f}  "
                      f"{fit['r2']:7.5f}")
            # Fit slope vs distance.
            if len({d for _, d, _, _, _ in rows}) >= 2:
                xs = [r[1] for r in rows]
                ys = [r[2] for r in rows]
                per_hop, base = least_squares(xs, ys)
                r2 = r_squared(xs, ys, per_hop, base)
                print(f"  fit:  base={base:6.2f} cyc/pkt  "
                      f"per_hop={per_hop:+6.3f} cyc/hop  R^2={r2:.5f}  "
                      f"distinct_distances={sorted(set(xs))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
