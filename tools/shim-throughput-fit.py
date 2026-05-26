#!/usr/bin/env python3
"""Fit shim DMA throughput as cold_start + words/rate from calibration sweep traces.

Reads HW trace artifacts produced by the bridge harness for the
_diag_shim_throughput_sweep/n{N} family of calibration kernels.
Computes per-BD-size shim DMA durations using the soc field of
events.json, then fits a linear regression of duration vs. BD size
for MM2S (push) and S2MM (pull) directions separately.

Usage:
  ./tools/shim-throughput-fit.py --results-dir build/bridge-test-results/latest
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SIZE_PAT = re.compile(r"_diag_shim_throughput_sweep_n(\d+)\.")


@dataclass
class Row:
    n: int
    compiler: str
    side: str  # "hw" or "emu"
    mm2s_dur: Optional[int]
    s2mm_dur: Optional[int]


def extract_durations(events_path: Path) -> tuple[Optional[int], Optional[int]]:
    """Return (mm2s_duration, s2mm_duration) from events.json soc fields."""
    with events_path.open() as f:
        data = json.load(f)
    placement = data.get("placement") or {}
    origin_col = placement.get("origin_col")
    if origin_col is None:
        return None, None
    shim = (origin_col, 0)
    events = [e for e in data.get("events", []) if (e["col"], e["row"]) == shim]
    soc = {}
    for e in events:
        name = e["name"]
        if name not in soc:
            soc[name] = e["soc"]
    mm2s_dur = None
    if "DMA_MM2S_0_START_TASK" in soc and "DMA_MM2S_0_FINISHED_TASK" in soc:
        mm2s_dur = soc["DMA_MM2S_0_FINISHED_TASK"] - soc["DMA_MM2S_0_START_TASK"]
    s2mm_dur = None
    if "DMA_S2MM_0_START_TASK" in soc and "DMA_S2MM_0_FINISHED_TASK" in soc:
        s2mm_dur = soc["DMA_S2MM_0_FINISHED_TASK"] - soc["DMA_S2MM_0_START_TASK"]
    return mm2s_dur, s2mm_dur


def collect_rows(results_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for dent in sorted(results_dir.iterdir()):
        if not dent.is_dir():
            continue
        m = SIZE_PAT.match(dent.name)
        if not m:
            continue
        n = int(m.group(1))
        # Name layout: _diag_shim_throughput_sweep_n{N}.<compiler>.<side>
        parts = dent.name.split(".")
        if len(parts) < 3:
            continue
        compiler = parts[1]
        side = parts[2]
        events_path = dent / "events.json"
        if not events_path.is_file():
            continue
        mm2s, s2mm = extract_durations(events_path)
        rows.append(Row(n=n, compiler=compiler, side=side, mm2s_dur=mm2s, s2mm_dur=s2mm))
    return sorted(rows, key=lambda r: (r.compiler, r.side, r.n))


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares fit ys = intercept + slope * xs.

    Returns (intercept, slope, r_squared).
    """
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den if den else float("nan")
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return intercept, slope, r_squared


def fit_direction(rows: list[Row], direction: str) -> dict:
    """Fit duration vs N for one direction ('mm2s' or 's2mm')."""
    points = [(r.n, getattr(r, f"{direction}_dur")) for r in rows]
    points = [(n, d) for n, d in points if d is not None]
    if len(points) < 2:
        return {"direction": direction, "n_points": len(points)}
    xs = [float(n) for n, _ in points]
    ys = [float(d) for _, d in points]
    intercept, slope, r2 = linear_fit(xs, ys)
    return {
        "direction": direction,
        "n_points": len(points),
        "cold_start_cyc": intercept,
        "cyc_per_word": slope,
        "words_per_cyc": (1.0 / slope) if slope else float("nan"),
        "r_squared": r2,
        "points": points,
    }


def print_table(rows: list[Row], label: str) -> None:
    print(f"=== Per-BD-size shim DMA durations ({label}) ===")
    print(f"{'N':>6} {'MM2S (push)':>14} {'S2MM (pull)':>14}")
    for r in rows:
        mm = f"{r.mm2s_dur:>14}" if r.mm2s_dur is not None else f"{'—':>14}"
        ss = f"{r.s2mm_dur:>14}" if r.s2mm_dur is not None else f"{'—':>14}"
        print(f"{r.n:>6} {mm} {ss}")
    print()


def print_fits(rows: list[Row], label: str) -> None:
    fits = [fit_direction(rows, d) for d in ("mm2s", "s2mm")]
    print(f"=== Linear fits ({label}): duration = cold_start + N / rate ===")
    for f in fits:
        if f.get("n_points", 0) < 2:
            print(f"  {f['direction']:>5}: insufficient data ({f.get('n_points', 0)} points)")
            continue
        print(
            f"  {f['direction']:>5}: cold_start={f['cold_start_cyc']:.1f} cyc, "
            f"slope={f['cyc_per_word']:.4f} cyc/word "
            f"({f['words_per_cyc']:.3f} words/cyc), R^2={f['r_squared']:.4f}, "
            f"n_points={f['n_points']}"
        )
    print()


def print_compare(hw_rows: list[Row], emu_rows: list[Row]) -> None:
    hw_by_n = {r.n: r for r in hw_rows}
    emu_by_n = {r.n: r for r in emu_rows}
    all_n = sorted(set(hw_by_n.keys()) | set(emu_by_n.keys()))
    print("=== HW vs EMU side-by-side ===")
    print(f"{'N':>6}  {'MM2S HW':>10}  {'MM2S EMU':>10}  {'MM2S Δ':>10}    {'S2MM HW':>10}  {'S2MM EMU':>10}  {'S2MM Δ':>10}")
    for n in all_n:
        hw = hw_by_n.get(n)
        emu = emu_by_n.get(n)
        hw_mm = hw.mm2s_dur if hw else None
        emu_mm = emu.mm2s_dur if emu else None
        hw_ss = hw.s2mm_dur if hw else None
        emu_ss = emu.s2mm_dur if emu else None
        def fmt(v):
            return f"{v:>10}" if v is not None else f"{'—':>10}"
        def delta(a, b):
            if a is None or b is None:
                return f"{'—':>10}"
            d = b - a
            return f"{d:>+10}"
        print(f"{n:>6}  {fmt(hw_mm)}  {fmt(emu_mm)}  {delta(hw_mm, emu_mm)}    {fmt(hw_ss)}  {fmt(emu_ss)}  {delta(hw_ss, emu_ss)}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--csv-out", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--compiler", default="chess")
    ap.add_argument("--side", default="hw", help="Single-side mode; ignored if --compare is set")
    ap.add_argument("--compare", action="store_true", help="Show HW vs EMU side-by-side")
    args = ap.parse_args()

    all_rows = collect_rows(args.results_dir)
    all_rows = [r for r in all_rows if r.compiler == args.compiler]
    if not all_rows:
        print(f"no rows found in {args.results_dir} for compiler={args.compiler}", file=sys.stderr)
        return 1

    if args.compare:
        hw_rows = [r for r in all_rows if r.side == "hw"]
        emu_rows = [r for r in all_rows if r.side == "emu"]
        print_table(hw_rows, f"{args.compiler}/hw")
        print_table(emu_rows, f"{args.compiler}/emu")
        print_compare(hw_rows, emu_rows)
        print_fits(hw_rows, f"{args.compiler}/hw")
        print_fits(emu_rows, f"{args.compiler}/emu")
        rows = hw_rows  # for CSV/JSON output
    else:
        rows = [r for r in all_rows if r.side == args.side]
        if not rows:
            print(f"no rows for side={args.side}", file=sys.stderr)
            return 1
        print_table(rows, f"{args.compiler}/{args.side}")
        print_fits(rows, f"{args.compiler}/{args.side}")
    fits = [fit_direction(rows, d) for d in ("mm2s", "s2mm")]

    if args.csv_out:
        with args.csv_out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["n_words", "compiler", "side", "mm2s_duration_cyc", "s2mm_duration_cyc"])
            for r in rows:
                w.writerow([r.n, r.compiler, r.side, r.mm2s_dur, r.s2mm_dur])
        print(f"\ncsv: {args.csv_out}")

    if args.json_out:
        out = {
            "results_dir": str(args.results_dir),
            "compiler": args.compiler,
            "side": args.side,
            "rows": [
                {
                    "n": r.n,
                    "mm2s_dur": r.mm2s_dur,
                    "s2mm_dur": r.s2mm_dur,
                }
                for r in rows
            ],
            "fits": [{k: v for k, v in f.items() if k != "points"} for f in fits],
        }
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
