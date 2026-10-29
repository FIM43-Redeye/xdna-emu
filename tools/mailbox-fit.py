#!/usr/bin/env python3
"""mailbox-fit.py -- fit per-sync vs per-batch mailbox latency from a
bridge sweep with --with-hw-cycles enabled.

For each test, reads:
  - bridge-test-results/<run>/<test>.<compiler>.cycles.HW.txt  (HW cycles)
  - bridge-test-results/<run>/<test>.<compiler>.bridge.log     (EMU
    "halt_reason=completed cycles=N" line, plus dma_wait sync count)

Computes per-kernel gap = HW - EMU and fits two models:
  - Per-batch (constant):  gap = c
  - Per-sync (linear):     gap = a*N

Reports R^2 for both and the calibration value implied by each.

Usage:
  python3 tools/mailbox-fit.py --results-dir build/bridge-test-results/20260510 \\
                               --compiler peano
"""

import argparse
import re
from pathlib import Path

HALT_RE = re.compile(r"halt_reason=completed cycles=(\d+)")
SYNC_RE = re.compile(r"NPU Sync #\d+ satisfied")


def parse_bridge_log(path: Path) -> tuple[int | None, int]:
    """Return (emu_cycles, sync_count) for a bridge log."""
    emu_cycles = None
    sync_count = 0
    try:
        with open(path) as f:
            for line in f:
                m = HALT_RE.search(line)
                if m:
                    emu_cycles = int(m.group(1))
                if SYNC_RE.search(line):
                    sync_count += 1
    except FileNotFoundError:
        return None, 0
    return emu_cycles, sync_count


def parse_hw_cycles(path: Path) -> int | None:
    """Read scalar HW cycle count from cycles.HW.txt."""
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def fit_constant(gaps: list[int]) -> tuple[float, float]:
    """Return (mean, R^2-against-mean-as-model) for constant fit."""
    if not gaps:
        return 0.0, 0.0
    mean = sum(gaps) / len(gaps)
    ss_tot = sum((g - mean) ** 2 for g in gaps)
    # R^2 of "predict mean" model is by construction 0 (no variance explained).
    # We compare to a true zero-predictor baseline to be fair to the model:
    # ss_res for constant model = ss_tot (residuals = deviation from mean).
    # R^2 = 1 - ss_res / ss_tot_around_zero
    ss_tot_zero = sum(g * g for g in gaps)
    if ss_tot_zero == 0:
        return mean, 1.0
    r2 = 1.0 - ss_tot / ss_tot_zero
    return mean, r2


def fit_linear_through_origin(ns: list[int], gaps: list[int]) -> tuple[float, float]:
    """Least-squares fit gap = a*N. Returns (a, R^2)."""
    if not ns:
        return 0.0, 0.0
    sum_n2 = sum(n * n for n in ns)
    sum_ng = sum(n * g for n, g in zip(ns, gaps))
    if sum_n2 == 0:
        return 0.0, 0.0
    a = sum_ng / sum_n2
    ss_res = sum((g - a * n) ** 2 for n, g in zip(ns, gaps))
    ss_tot = sum(g * g for g in gaps)
    if ss_tot == 0:
        return a, 1.0
    return a, 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--compiler", default="peano", choices=["peano", "chess"])
    args = ap.parse_args()

    rd: Path = args.results_dir
    if not rd.exists():
        raise SystemExit(f"results dir not found: {rd}")

    # Discover tests by their cycles.HW.txt files.
    suffix = f".{args.compiler}.cycles.HW.txt"
    rows = []
    for hw_file in sorted(rd.glob(f"*{suffix}")):
        test_safe = hw_file.name[: -len(suffix)]
        # Bridge logs are named with the same safe-name.
        bridge_log = rd / f"{test_safe}.{args.compiler}.bridge.log"
        hw_cycles = parse_hw_cycles(hw_file)
        emu_cycles, sync_count = parse_bridge_log(bridge_log)
        if hw_cycles is None or emu_cycles is None or sync_count == 0:
            continue
        gap = hw_cycles - emu_cycles
        rows.append((test_safe, sync_count, hw_cycles, emu_cycles, gap))

    if not rows:
        raise SystemExit("no usable HW+EMU cycle pairs found")

    print(f"{'test':<45} {'syncs':>7} {'HW':>10} {'EMU':>10} {'gap':>10} {'gap/sync':>10}")
    print("-" * 96)
    for name, n, hw, emu, gap in sorted(rows, key=lambda r: r[1]):
        per = gap / n if n else 0
        print(f"{name:<45} {n:>7} {hw:>10} {emu:>10} {gap:>10} {per:>10.1f}")

    ns = [r[1] for r in rows]
    gaps = [r[4] for r in rows]
    c_mean, c_r2 = fit_constant(gaps)
    lin_a, lin_r2 = fit_linear_through_origin(ns, gaps)
    print()
    print(f"Constant fit (per-batch):  gap = {c_mean:>8.1f}     R^2 vs zero = {c_r2:>6.3f}")
    print(f"Linear fit (per-sync):     gap = {lin_a:>8.1f} * N  R^2 vs zero = {lin_r2:>6.3f}")

    if c_r2 > lin_r2:
        print(f"\nBetter fit: PER-BATCH (constant {c_mean:.0f} cyc, independent of sync count).")
    elif lin_r2 > c_r2:
        print(f"\nBetter fit: PER-SYNC (linear {lin_a:.0f} cyc/sync).")
    else:
        print("\nTie -- inspect by hand.")


if __name__ == "__main__":
    main()
