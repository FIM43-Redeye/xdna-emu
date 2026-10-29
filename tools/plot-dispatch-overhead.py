#!/usr/bin/env python3
"""
plot-dispatch-overhead.py -- Visualize the multi-run dispatch-overhead
aggregation produced by aggregate-dispatch-overhead.py.

Two plot styles:
  - Per-(K, direction) overall gap distribution histograms (one PNG per
    direction)
  - Per-gap-index distribution boxplots (gap[0], gap[1], gap[2], ...)
    one PNG per direction

Inputs:
  <session-dir>/aggregated.json  (run aggregate-dispatch-overhead first)

Outputs:
  <session-dir>/plots/
    histogram-MM2S.png
    histogram-S2MM.png
    by-index-MM2S.png
    by-index-S2MM.png

Usage:
  ./tools/plot-dispatch-overhead.py <session-dir>
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: <session>/plots)")
    args = ap.parse_args()

    agg_path = args.session_dir / "aggregated.json"
    if not agg_path.exists():
        print(f"error: {agg_path} missing -- run aggregate-dispatch-overhead first",
              file=sys.stderr)
        return 1
    agg = json.loads(agg_path.read_text())
    raw = agg.get("raw_per_run", {})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"error: matplotlib/numpy not importable ({e}); "
              f"source the ironenv: source mlir-aie/ironenv/bin/activate",
              file=sys.stderr)
        return 1

    out_dir = args.out_dir or (args.session_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (k, direction) keys
    ks_per_dir = {"MM2S": set(), "S2MM": set()}
    for key in agg["summary"]:
        if "." not in key:
            continue
        kstr, direction = key.split(".", 1)
        try:
            k = int(kstr.removeprefix("k"))
        except ValueError:
            continue
        if direction in ks_per_dir:
            ks_per_dir[direction].add(k)

    # ---- Plot 1: histograms of all_gaps per K, per direction ----
    for direction in ("MM2S", "S2MM"):
        ks = sorted(ks_per_dir[direction])
        if not ks:
            continue
        # One subplot per K
        ncols = len(ks)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4),
                                 sharey=False, squeeze=False)
        for ax, k in zip(axes[0], ks):
            key = f"k{k}.{direction}"
            gaps = []
            for run_rec in raw.get(key, []):
                gaps.extend(run_rec["gaps"])
            if not gaps:
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"K={k}")
                continue
            arr = np.array(gaps)
            ax.hist(arr, bins=40, alpha=0.85, edgecolor="black",
                    linewidth=0.4)
            med = np.median(arr)
            ax.axvline(med, color="red", linestyle="--", lw=1.5,
                       label=f"median {med:.0f}")
            ax.axvline(0, color="gray", linestyle=":", lw=1)
            ax.set_title(f"K={k} {direction} (n={len(arr)})")
            ax.set_xlabel("inter-task gap (cyc)")
            ax.set_ylabel("count")
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(alpha=0.3)
        fig.suptitle(
            f"Inter-task gap distribution: {direction} "
            f"(session: {agg['session']}, all gaps including first)",
        )
        fig.tight_layout()
        out = out_dir / f"histogram-{direction}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    # ---- Plot 2: per-gap-index boxplots per K, per direction ----
    for direction in ("MM2S", "S2MM"):
        ks = sorted(ks_per_dir[direction])
        if not ks:
            continue
        ncols = len(ks)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
        for ax, k in zip(axes[0], ks):
            key = f"k{k}.{direction}"
            # Build per-gap-index lists
            n_gaps = k - 1 if k >= 2 else 0
            data = []
            labels = []
            for i in range(n_gaps):
                vals = [run_rec["gaps"][i] for run_rec in raw.get(key, [])
                        if i < len(run_rec["gaps"])]
                if vals:
                    data.append(vals)
                    labels.append(f"gap[{i}]")
            if not data:
                ax.text(0.5, 0.5, "K=1 (no gaps)" if k == 1 else "no data",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"K={k}")
                continue
            bp = ax.boxplot(data, tick_labels=labels, showmeans=True,
                            meanprops=dict(marker="D", markerfacecolor="red",
                                           markeredgecolor="red", markersize=5))
            ax.axhline(0, color="gray", linestyle=":", lw=1)
            ax.set_title(f"K={k} {direction}")
            ax.set_ylabel("gap (cyc)")
            ax.grid(alpha=0.3)
        fig.suptitle(
            f"Per-gap-index distribution: {direction} "
            f"(red diamond = mean; session: {agg['session']})",
        )
        fig.tight_layout()
        out = out_dir / f"by-index-{direction}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
