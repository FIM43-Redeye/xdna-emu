#!/usr/bin/env python3
"""Tables from op-cost probe runs: per-case interval statistics, per-op costs, two-point split.

  python3 tools/fw_op_cost_report.py <cases_dir> <runs_dir>

Intervals are in array cycles (tile counter). A stall is an interval more than
STALL_EXTRA_CYCLES above the run's median interval (about one per run, see the evidence notes);
it is counted apart from the op cost. Means are over the remaining intervals: the poll loop's
15-cycle dither on single intervals cancels in the sum.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import gen_fw_op_cost_probe as g
from fw_op_cost_run import EXPECTED

STALL_EXTRA_CYCLES = 1500   # a stall adds 2.1k-2.8k cycles to an interval; ordinary jitter stays far below
REPEATS = ("r1", "r2")


def split_stalls(intervals: list[int]) -> tuple[list[int], list[int]]:
    limit = statistics.median(intervals) + STALL_EXTRA_CYCLES
    return ([v for v in intervals if v <= limit], [v for v in intervals if v > limit])


def mean_interval(intervals: list[int]) -> float:
    return sum(intervals) / len(intervals)


def span_mean(stamps: list[int]) -> float:
    return ((stamps[-1] - stamps[0]) & 0xFFFFFFFF) / (len(stamps) - 1)


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares y = a + b x; returns (a, b)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def to_ns(cycles: float, array_mhz: float) -> float:
    return cycles * 1000.0 / array_mhz


def two_point_split(ns_at_600: float, ns_at_400: float) -> tuple[float, float]:
    """t = fw_cycles / f_fw + fixed_ns with f_fw = 600, 400 MHz: (fw_cycles, fixed_ns).
    Valid only if the cost has one firmware-clock part and one clock-independent part."""
    fw_cycles = (ns_at_400 - ns_at_600) / (1 / 400 - 1 / 600) / 1000.0
    return fw_cycles, ns_at_600 - fw_cycles * 1000.0 / 600


def load(runs: Path, tag: str, dpm: int, case: g.Case):
    d = runs / tag / f"dpm{dpm}" / case.name
    return g.analyze(case, (d / "out.bin").read_bytes()), json.loads((d / "run.json").read_text())


def case_stats(c: g.Case, runs: Path, dpm: int) -> dict:
    """Per repeat: stall-free interval summary and span mean; plus validity."""
    out = {}
    for tag in REPEATS:
        r, rec = load(runs, tag, dpm, c)
        normal, stalls = split_stalls(r.intervals)
        out[tag] = {"ok": r.ok and rec["valid"], "n": len(normal), "stalls": stalls,
                    "summary": g.summarize(normal) if normal else None,
                    "mean": mean_interval(normal) if normal else None}
    return out


def pooled_mean(runs: Path, names: list[str], dpm: int) -> tuple[float, int]:
    """Mean stall-free interval over both repeats of the named cases, and how many went in."""
    cases = {c.name: c for c in g.all_cases()}
    vals: list[int] = []
    for n in names:
        for tag in REPEATS:
            vals += split_stalls(load(runs, tag, dpm, cases[n])[0].intervals)[0]
    return mean_interval(vals), len(vals)


def derived(runs: Path) -> list[str]:
    """Per-op costs at both clock points and the two-point split (fw cycles + fixed ns)."""
    mhz = {d: EXPECTED[d][1] for d in (1, 0)}
    fw = {1: EXPECTED[1][0], 0: EXPECTED[0][0]}
    lines = ["op                               dpm1 cyc  dpm0 cyc   dpm1 ns  dpm0 ns   fw cycles  fixed ns"]

    def row(label: str, c1: float, c0: float, per: float = 1.0) -> None:
        n1, n0 = to_ns(c1 / per, mhz[1]), to_ns(c0 / per, mhz[0])
        f, fixed = two_point_split(n1, n0)
        lines.append(f"{label:32}{c1 / per:9.1f} {c0 / per:9.1f} {n1:9.1f} {n0:8.1f} {f:11.1f} {fixed:9.1f}")

    groups = {"write32 (w32_n8..64)": [f"w32_n{n}" for n in (8, 16, 32, 64)],
              "write32 after 64 prefill ops": ["pre_f64_w32_n64"],
              "maskwrite32 (mw32_n8..64)": [f"mw32_n{n}" for n in (8, 16, 32, 64)]}
    for label, names in groups.items():
        row(label, pooled_mean(runs, names, 1)[0], pooled_mean(runs, names, 0)[0])
    for tag, name in (("memtile", "mt"), ("column1", "c1")):
        for k in (1, 4):
            c1, c0 = (pooled_mean(runs, [f"il_{name}_k{k}"], d)[0] for d in (1, 0))
            row(f"filler write32 -> {tag}, k={k}", c1, c0, per=k + 1)
    blocks = {1: "bw_l1", 4: "bw_l4_last", 16: "bw_l16_last", 64: "bw_l64_last"}
    per_block = {d: {l: pooled_mean(runs, [n], d)[0] for l, n in blocks.items()} for d in (1, 0)}
    for l, n in blocks.items():
        row(f"blockwrite L={l} (per op)", per_block[1][l], per_block[0][l])
    fits = {d: fit_line(list(blocks), [per_block[d][l] for l in blocks]) for d in (1, 0)}
    row("blockwrite fixed part (fit)", fits[1][0], fits[0][0])
    row("blockwrite per word (fit)", fits[1][1], fits[0][1])
    return lines


def main(argv=None) -> int:
    cases_dir, runs = Path(argv[0]), Path(argv[1])
    cases = {c.name: c for c in g.all_cases()}
    print(f"{'case':18}{'dpm':>4} {'valid':>5} {'n':>4} {'min':>6} {'med':>7} {'max':>6} "
          f"{'mean r1':>8} {'mean r2':>8} {'stalls r1/r2':>14}")
    for name, c in cases.items():
        if c.kind == "spin":
            continue
        for dpm in (1, 0):
            s = case_stats(c, runs, dpm)
            a, b = s["r1"], s["r2"]
            if not a["summary"]:
                print(f"{name:18}{dpm:>4} {str(a['ok'] and b['ok']):>5}  (no stall-free interval)")
                continue
            sm = a["summary"]
            print(f"{name:18}{dpm:>4} {str(a['ok'] and b['ok']):>5} {sm['n']:>4} {sm['min']:>6} "
                  f"{sm['median']:>7} {sm['max']:>6} {a['mean']:>8.1f} {b['mean']:>8.1f} "
                  f"{str(a['stalls']) + '/' + str(b['stalls']):>14}")
    print()
    print("\n".join(derived(runs)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
