#!/usr/bin/env python3
"""Fit per-BD-dispatch cold-start amortization from chain-sweep traces.

Companion to shim-throughput-fit.py. Reads HW (and optionally EMU) trace
artifacts produced by the bridge harness for the _diag_shim_chain_sweep/k{K}
family of calibration kernels. For each K, computes:

  - Per-task durations: list of FINISHED_i - START_i, i in 0..K
  - Inter-task gaps:   list of START_(i+1) - FINISHED_i, i in 0..K-1
  - Total span:        last FINISHED - first START

Then across K, fits total_span(K) = cold_start + K * per_task_overhead
+ K * N * slope. If per_task_overhead > 0, HW pays per-task setup that
EMU should model. If per_task_overhead ~= 0, cold-start amortizes across
chained dispatches and EMU's current per-task cold-start model overshoots
multi-task chains.

Usage:
  ./tools/shim-chain-fit.py --results-dir build/bridge-test-results/latest
  ./tools/shim-chain-fit.py --results-dir build/bridge-test-results/latest --compare
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

K_PAT = re.compile(r"_diag_shim_chain_sweep_k(\d+)\.")
N_DEFAULT = 64  # matches the generator


@dataclass
class Run:
    k: int
    compiler: str
    side: str  # "hw" or "emu"
    mm2s_starts: list[int] = field(default_factory=list)
    mm2s_finishes: list[int] = field(default_factory=list)
    s2mm_starts: list[int] = field(default_factory=list)
    s2mm_finishes: list[int] = field(default_factory=list)

    def mm2s_durations(self) -> list[int]:
        return [f - s for s, f in zip(self.mm2s_starts, self.mm2s_finishes)]

    def s2mm_durations(self) -> list[int]:
        return [f - s for s, f in zip(self.s2mm_starts, self.s2mm_finishes)]

    def mm2s_gaps(self) -> list[int]:
        return [
            self.mm2s_starts[i + 1] - self.mm2s_finishes[i]
            for i in range(len(self.mm2s_starts) - 1)
        ]

    def s2mm_gaps(self) -> list[int]:
        return [
            self.s2mm_starts[i + 1] - self.s2mm_finishes[i]
            for i in range(len(self.s2mm_starts) - 1)
        ]

    def mm2s_span(self) -> int | None:
        if not self.mm2s_starts or not self.mm2s_finishes:
            return None
        return self.mm2s_finishes[-1] - self.mm2s_starts[0]

    def s2mm_span(self) -> int | None:
        if not self.s2mm_starts or not self.s2mm_finishes:
            return None
        return self.s2mm_finishes[-1] - self.s2mm_starts[0]


def extract_events(events_path: Path) -> Run | None:
    with events_path.open() as f:
        data = json.load(f)
    placement = data.get("placement") or {}
    origin_col = placement.get("origin_col")
    if origin_col is None:
        return None
    shim = (origin_col, 0)
    events = [e for e in data.get("events", []) if (e["col"], e["row"]) == shim]
    # Sort by soc time to get correct ordering of K dispatches
    events.sort(key=lambda e: e["soc"])
    run = Run(k=0, compiler="", side="")
    for e in events:
        name = e["name"]
        if name == "DMA_MM2S_0_START_TASK":
            run.mm2s_starts.append(e["soc"])
        elif name == "DMA_MM2S_0_FINISHED_TASK":
            run.mm2s_finishes.append(e["soc"])
        elif name == "DMA_S2MM_0_START_TASK":
            run.s2mm_starts.append(e["soc"])
        elif name == "DMA_S2MM_0_FINISHED_TASK":
            run.s2mm_finishes.append(e["soc"])
    return run


def collect_runs(results_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for dent in sorted(results_dir.iterdir()):
        if not dent.is_dir():
            continue
        m = K_PAT.match(dent.name)
        if not m:
            continue
        k = int(m.group(1))
        parts = dent.name.split(".")
        if len(parts) < 3:
            continue
        compiler = parts[1]
        side = parts[2]
        events_path = dent / "events.json"
        if not events_path.is_file():
            continue
        run = extract_events(events_path)
        if run is None:
            continue
        run.k = k
        run.compiler = compiler
        run.side = side
        runs.append(run)
    return sorted(runs, key=lambda r: (r.compiler, r.side, r.k))


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
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


def print_per_task_table(runs: list[Run], label: str) -> None:
    print(f"=== Per-task durations ({label}) ===")
    print(f"{'K':>4}  {'direction':>10}  {'per-task durations (cyc)'}")
    for r in runs:
        for direction, durs in [
            ("MM2S", r.mm2s_durations()),
            ("S2MM", r.s2mm_durations()),
        ]:
            if not durs:
                continue
            dur_str = ", ".join(f"{d:>5}" for d in durs)
            print(f"{r.k:>4}  {direction:>10}  [{dur_str}]")
    print()


def print_gap_table(runs: list[Run], label: str) -> None:
    print(f"=== Inter-task gaps ({label}) ===")
    print(f"{'K':>4}  {'direction':>10}  {'gaps (cyc, START_{i+1} - FINISHED_i)'}")
    for r in runs:
        for direction, gaps in [
            ("MM2S", r.mm2s_gaps()),
            ("S2MM", r.s2mm_gaps()),
        ]:
            if not gaps:
                continue
            gap_str = ", ".join(f"{g:>+5}" for g in gaps)
            print(f"{r.k:>4}  {direction:>10}  [{gap_str}]")
    print()


def print_span_table(runs: list[Run], label: str) -> None:
    print(f"=== Total spans by K ({label}) ===")
    print(f"{'K':>4}  {'MM2S span (cyc)':>16}  {'S2MM span (cyc)':>16}")
    for r in runs:
        mm = r.mm2s_span()
        ss = r.s2mm_span()
        mm_s = f"{mm:>16}" if mm is not None else f"{'—':>16}"
        ss_s = f"{ss:>16}" if ss is not None else f"{'—':>16}"
        print(f"{r.k:>4}  {mm_s}  {ss_s}")
    print()


def fit_total_span(runs: list[Run], direction: str, n: int) -> dict:
    """Fit total_span(K) = cold + K*per_task + K*N*slope.

    Treated as a 2-D linear regression of duration vs (1, K), with the
    K*N*slope term absorbed into the slope on K (since N is fixed in
    this sweep). The fit yields cold_start (intercept) and a combined
    "per_K_growth" (slope on K). per_K_growth = per_task_overhead + N*slope.
    To extract per_task_overhead specifically, we subtract N*slope_throughput
    using the throughput slope from the N-sweep (1.0 cyc/word for shim).
    """
    points = []
    for r in runs:
        span = r.mm2s_span() if direction == "mm2s" else r.s2mm_span()
        if span is None:
            continue
        points.append((r.k, span))
    if len(points) < 2:
        return {"direction": direction, "n_points": len(points)}
    xs = [float(k) for k, _ in points]
    ys = [float(span) for _, span in points]
    intercept, per_k_growth, r2 = linear_fit(xs, ys)
    # Throughput slope is 1.0 cyc/word from the N-sweep calibration.
    # per_K_growth = per_task_overhead + N * throughput_slope
    throughput_slope_cyc_per_word = 1.0
    per_task_overhead = per_k_growth - n * throughput_slope_cyc_per_word
    return {
        "direction": direction,
        "n_points": len(points),
        "cold_start_cyc": intercept,
        "per_k_growth_cyc": per_k_growth,
        "per_task_overhead_cyc": per_task_overhead,
        "n_words_per_bd": n,
        "throughput_slope_assumed": throughput_slope_cyc_per_word,
        "r_squared": r2,
        "points": points,
    }


def print_fits(runs: list[Run], label: str, n: int) -> None:
    fits = [fit_total_span(runs, d, n) for d in ("mm2s", "s2mm")]
    print(
        f"=== Total-span fit ({label}): "
        f"total_span(K) = cold + K * (per_task_overhead + N * slope) ==="
    )
    print(f"(N = {n} words/BD, throughput slope assumed = 1.0 cyc/word)")
    for f in fits:
        if f.get("n_points", 0) < 2:
            print(f"  {f['direction']:>5}: insufficient data ({f.get('n_points', 0)} points)")
            continue
        print(
            f"  {f['direction']:>5}: cold_start={f['cold_start_cyc']:.1f} cyc, "
            f"per_K_growth={f['per_k_growth_cyc']:.2f} cyc/K, "
            f"per_task_overhead={f['per_task_overhead_cyc']:.2f} cyc/task, "
            f"R^2={f['r_squared']:.4f}, n_points={f['n_points']}"
        )
    print()


def print_compare(hw_runs: list[Run], emu_runs: list[Run]) -> None:
    hw_by_k = {r.k: r for r in hw_runs}
    emu_by_k = {r.k: r for r in emu_runs}
    all_k = sorted(set(hw_by_k.keys()) | set(emu_by_k.keys()))
    print("=== HW vs EMU side-by-side (total spans) ===")
    print(
        f"{'K':>4}  "
        f"{'MM2S HW':>10}  {'MM2S EMU':>10}  {'MM2S Δ':>10}    "
        f"{'S2MM HW':>10}  {'S2MM EMU':>10}  {'S2MM Δ':>10}"
    )

    def fmt(v):
        return f"{v:>10}" if v is not None else f"{'—':>10}"

    def delta(a, b):
        if a is None or b is None:
            return f"{'—':>10}"
        return f"{(b - a):>+10}"

    for k in all_k:
        hw = hw_by_k.get(k)
        emu = emu_by_k.get(k)
        hw_mm = hw.mm2s_span() if hw else None
        emu_mm = emu.mm2s_span() if emu else None
        hw_ss = hw.s2mm_span() if hw else None
        emu_ss = emu.s2mm_span() if emu else None
        print(
            f"{k:>4}  "
            f"{fmt(hw_mm)}  {fmt(emu_mm)}  {delta(hw_mm, emu_mm)}    "
            f"{fmt(hw_ss)}  {fmt(emu_ss)}  {delta(hw_ss, emu_ss)}"
        )
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--compiler", default="chess")
    ap.add_argument("--side", default="hw", help="Single-side mode; ignored if --compare")
    ap.add_argument("--compare", action="store_true", help="HW vs EMU side-by-side")
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="Words per BD")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    all_runs = collect_runs(args.results_dir)
    all_runs = [r for r in all_runs if r.compiler == args.compiler]
    if not all_runs:
        print(
            f"no runs found in {args.results_dir} for compiler={args.compiler}",
            file=sys.stderr,
        )
        return 1

    if args.compare:
        hw_runs = [r for r in all_runs if r.side == "hw"]
        emu_runs = [r for r in all_runs if r.side == "emu"]
        print_span_table(hw_runs, f"{args.compiler}/hw")
        print_span_table(emu_runs, f"{args.compiler}/emu")
        print_per_task_table(hw_runs, f"{args.compiler}/hw")
        print_per_task_table(emu_runs, f"{args.compiler}/emu")
        print_gap_table(hw_runs, f"{args.compiler}/hw")
        print_gap_table(emu_runs, f"{args.compiler}/emu")
        print_compare(hw_runs, emu_runs)
        print_fits(hw_runs, f"{args.compiler}/hw", args.n)
        print_fits(emu_runs, f"{args.compiler}/emu", args.n)
        out_runs = hw_runs
    else:
        runs = [r for r in all_runs if r.side == args.side]
        if not runs:
            print(f"no runs for side={args.side}", file=sys.stderr)
            return 1
        print_span_table(runs, f"{args.compiler}/{args.side}")
        print_per_task_table(runs, f"{args.compiler}/{args.side}")
        print_gap_table(runs, f"{args.compiler}/{args.side}")
        print_fits(runs, f"{args.compiler}/{args.side}", args.n)
        out_runs = runs

    if args.json_out:
        out = {
            "results_dir": str(args.results_dir),
            "compiler": args.compiler,
            "n_words_per_bd": args.n,
            "runs": [
                {
                    "k": r.k,
                    "side": r.side,
                    "mm2s_starts": r.mm2s_starts,
                    "mm2s_finishes": r.mm2s_finishes,
                    "s2mm_starts": r.s2mm_starts,
                    "s2mm_finishes": r.s2mm_finishes,
                    "mm2s_durations": r.mm2s_durations(),
                    "s2mm_durations": r.s2mm_durations(),
                    "mm2s_gaps": r.mm2s_gaps(),
                    "s2mm_gaps": r.s2mm_gaps(),
                    "mm2s_span": r.mm2s_span(),
                    "s2mm_span": r.s2mm_span(),
                }
                for r in out_runs
            ],
            "fits": [
                {k: v for k, v in fit_total_span(out_runs, d, args.n).items() if k != "points"}
                for d in ("mm2s", "s2mm")
            ],
        }
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
