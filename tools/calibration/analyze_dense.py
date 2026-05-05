#!/usr/bin/env python3
"""Analyze a dense-N calibration sweep: expose modular structure and
fit an integer-coefficient model.

Reads measurements JSON from run_sweep.py. Expects a sweep with contiguous
counts (1, 2, 3, ..., N) so first/second differences are meaningful.

Output sections:
  1. Reproducibility check: rep-to-rep variance per count.
  2. Raw cycles vs N table (using min-of-reps to reject stochastic
     additive artifacts).
  3. First differences delta[N] = cycles[N] - cycles[N-1].
  4. Period-K search: variance of delta within each residue class N mod K
     for K in 2..32. Lowest within-class variance => most likely period.
  5. Best-fit integer model: cost(N) = a*N + b + extra(N mod p).

Usage:
  python3 analyze_dense.py path/to/measurements.json [--max-n N]

  --max-n N      Cap analysis at N <= max-n (default: all data).
                 Useful when an artifact contaminates higher counts.
"""

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def group_key(m: dict) -> tuple:
    return (
        m["kind"],
        m.get("target_col", 0), m.get("target_row", 2),
        m.get("anchor_col", 0), m.get("anchor_row", 2),
        m.get("payload", 0),
    )


def fmt_key(k: tuple) -> str:
    kind, tc, tr, ac, ar, pl = k
    s = f"{kind} target=({tc},{tr}) anchor=({ac},{ar})"
    if pl:
        s += f" payload={pl}"
    return s


def collect(measurements: list) -> dict:
    """Group by (kind, target, anchor, payload) -> count -> [cycles per rep]."""
    out = defaultdict(lambda: defaultdict(list))
    for m in measurements:
        if m.get("hw_cycles") is None:
            continue
        out[group_key(m)][m["count"]].append(m["hw_cycles"])
    return out


def reproducibility(per_count: dict) -> tuple:
    """Return (max_spread, mean_spread, count_with_spread) across reps."""
    spreads = []
    for _, vals in per_count.items():
        if len(vals) >= 2:
            spreads.append(max(vals) - min(vals))
    if not spreads:
        return (0, 0.0, 0)
    return (max(spreads), sum(spreads) / len(spreads),
            sum(1 for s in spreads if s > 0))


def reduce_per_count(per_count: dict) -> dict:
    """Pick a single cycle value per count. Uses min-of-reps to reject
    stochastic additive artifacts (rep-specific +N cycle penalties).

    Justification: real cycle costs are deterministic; any *extra* cycles
    must come from contention or buffer effects. Min across reps therefore
    estimates the unperturbed cost.
    """
    return {n: min(vals) for n, vals in per_count.items()}


def first_diffs(cycles: dict) -> dict:
    """Return {n: cycles[n] - cycles[n-1]} for contiguous n."""
    ns = sorted(cycles)
    return {ns[i]: cycles[ns[i]] - cycles[ns[i - 1]]
            for i in range(1, len(ns))
            if ns[i] - ns[i - 1] == 1}


def period_search(deltas: dict, max_period: int = 64) -> list:
    """For each candidate period p, score how well 'mean delta within each
    residue class N mod p' explains the data. Use BIC (Bayesian Information
    Criterion) to penalise larger periods so the simplest period that fits
    wins, rather than overfitting with finer-grained periods.

    Returns list of (p, bic, within_var, residue_means) sorted by BIC ascending.

    BIC = p * log(n) + n * log(RSS/n)
      where p = number of parameters (one mean per residue bucket),
            n = number of data points,
            RSS = sum of squared residuals within each bucket.
    """
    n = len(deltas)
    if n < 4:
        return []
    results = []
    for p in range(1, max_period + 1):
        if p > n:
            break
        buckets = defaultdict(list)
        for k, d in deltas.items():
            buckets[k % p].append(d)
        # Skip periods that leave any residue empty (degenerate fit).
        if len(buckets) < p:
            continue
        # Require at least 2 points per bucket so within-bucket variance is
        # meaningful. Otherwise BIC log(RSS/n) -> -inf when RSS hits 0,
        # which makes p=n always "win" trivially.
        if min(len(b) for b in buckets.values()) < 2:
            continue
        rss = 0.0
        for b in buckets.values():
            m = sum(b) / len(b)
            rss += sum((x - m) ** 2 for x in b)
        within_var = rss / n
        if rss <= 0:
            rss = 1e-9
        bic = p * math.log(n) + n * math.log(rss / n)
        means = {r: statistics.mean(buckets[r]) for r in buckets}
        results.append((p, bic, within_var, means))
    results.sort(key=lambda x: x[1])
    return results


def fit_integer_model(cycles: dict, period: int) -> dict:
    """Given cycles[N] for contiguous N and a candidate period, fit a model
    using per-period integer cost as the building block:

        cost(N) = (N // period) * P + offset(N mod period)

    Where:
      - P is the integer cycle cost of one full period (sum of per-residue
        mean deltas, rounded to nearest integer).
      - offset(r) is the integer cycle cost for the first `r` packets of
        a period; offset(0) = 0 by definition, offset(period) = P.

    This formulation keeps everything integer-valued, avoiding the
    misleading "fractional cycles per packet" you get from rounding the
    per-packet average when period > 1.

    Returns {"P": ..., "offset": {r: ...}, "max_residual": ...,
            "per_packet_avg": P/period}.
    """
    deltas = first_diffs(cycles)
    if not deltas:
        return None
    # Median delta per residue class. Median is robust to occasional +N
    # cycle artifacts (which a mean would absorb). For low-noise dense
    # cycle data, median lands on a true integer naturally.
    by_residue = defaultdict(list)
    for n, d in deltas.items():
        by_residue[n % period].append(d)
    # delta_int[r] = integer-rounded median delta at residue r.
    delta_int = {r: round(statistics.median(v)) for r, v in by_residue.items()}
    P = sum(delta_int.values())
    # offset(r) = sum of delta_int at residues 1, 2, ..., r (mod period).
    # That is, offset(0) = 0, offset(1) = delta_int[1], offset(2) =
    # delta_int[1] + delta_int[2], ..., offset(p-1) = P - delta_int[0].
    offset = {0: 0}
    cumul = 0
    for r in range(1, period):
        cumul += delta_int.get(r, 0)
        offset[r] = cumul
    # Pick the global b such that cost(N=1) matches actual.
    # cost(1) = (1 // p) * P + offset(1 mod p) + b
    # => b = cycles[1] - (1//p)*P - offset(1 mod p)
    ns = sorted(cycles)
    n0 = ns[0]
    b = cycles[n0] - (n0 // period) * P - offset[n0 % period]
    all_residuals = []
    for n in ns:
        pred = (n // period) * P + offset[n % period] + b
        all_residuals.append(cycles[n] - pred)
    return {
        "period": period,
        "P": P,
        "delta_int": delta_int,
        "offset": offset,
        "intercept": b,
        "per_packet_avg": P / period,
        "max_residual": max(abs(r) for r in all_residuals),
        "all_residuals": all_residuals,
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--max-n", type=int, default=None,
                    help="Cap analysis at N <= max-n")
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    measurements = data["measurements"]
    if args.max_n is not None:
        measurements = [m for m in measurements
                        if m.get("count", 0) <= args.max_n]
    grouped = collect(measurements)

    for key, per_count in sorted(grouped.items()):
        print("=" * 80)
        print(fmt_key(key))
        print("=" * 80)

        max_spr, mean_spr, n_var = reproducibility(per_count)
        print(f"Reproducibility: max rep-spread={max_spr} cyc, "
              f"mean={mean_spr:.2f} cyc, "
              f"{n_var}/{len(per_count)} counts had any variance")

        cycles = reduce_per_count(per_count)
        ns = sorted(cycles)

        # Section: raw + deltas.
        print()
        print(f"{'N':>4s}  {'cycles':>8s}  {'delta':>7s}  {'2nd_diff':>9s}")
        deltas = first_diffs(cycles)
        prev_d = None
        for n in ns:
            c = cycles[n]
            d = deltas.get(n)
            d_str = f"{d:>7.0f}" if d is not None else "       "
            if d is not None and prev_d is not None:
                d2 = d - prev_d
                d2_str = f"{d2:+9.0f}"
            else:
                d2_str = "         "
            prev_d = d
            print(f"{n:>4d}  {c:>8.0f}  {d_str}  {d2_str}")

        # Section: period search.
        print()
        print("Period search (BIC; lower is better, penalises overfitting):")
        results = period_search(deltas, max_period=64)
        if results:
            print(f"  {'period':>6s}  {'BIC':>10s}  {'within-var':>11s}")
            # Show period=1 for reference, then top BIC periods.
            top_p = {p for p, _, _, _ in results[:8]}
            for p, bic, var, _ in sorted(results, key=lambda x: x[0])[:8]:
                marker = " *" if p == results[0][0] else ""
                print(f"  {p:>6d}  {bic:>10.2f}  {var:>11.3f}{marker}")
            print(f"  ... best by BIC: p={results[0][0]} "
                  f"(BIC={results[0][1]:.2f})")
            # ΔBIC < 4 = no strong evidence to prefer the more complex model
            # (Kass & Raftery 1995). Pick the smallest period within that
            # band of the BIC minimum -- the most parsimonious explanation.
            min_bic = results[0][1]
            within_band = [r for r in results
                           if r[0] >= 2 and r[1] - min_bic <= 4.0]
            best = min(within_band, key=lambda x: x[0]) if within_band else None
            if best and best[0] != results[0][0]:
                print(f"  ... parsimonious choice (ΔBIC≤4): p={best[0]} "
                      f"(BIC={best[1]:.2f})")
            if best:
                best_period, _, _, residue_means = best
                print(f"\nResidue means at p={best_period}:")
                for r in sorted(residue_means):
                    print(f"  N mod {best_period} == {r}: mean delta = "
                          f"{residue_means[r]:.2f}")

                # Section: integer-fit model.
                fit = fit_integer_model(cycles, best_period)
                if fit:
                    p = fit["period"]
                    print(f"\nInteger-fit model (period {p}):")
                    print(f"  cost(N) = (N // {p}) * {fit['P']} + "
                          f"offset(N mod {p}) + {fit['intercept']}")
                    print(f"  per-period total P:  {fit['P']} cyc "
                          f"(per-packet avg: {fit['per_packet_avg']:.3f})")
                    print(f"  intercept:           {fit['intercept']} cyc "
                          f"(matches cost at N={min(cycles)})")
                    print(f"  per-residue delta (cyc to step into residue r):")
                    for r in sorted(fit['delta_int']):
                        print(f"    delta[{r}] = {fit['delta_int'][r]}")
                    print(f"  cumulative offset within period:")
                    for r in sorted(fit['offset']):
                        print(f"    offset[{r}] = {fit['offset'][r]}")
                    print(f"  max residual:        {fit['max_residual']:.1f} cyc")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
