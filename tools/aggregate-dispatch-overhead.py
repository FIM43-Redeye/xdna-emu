#!/usr/bin/env python3
"""
aggregate-dispatch-overhead.py -- Walk a multi-run campaign output tree
and extract per-K, per-direction inter-task gap distributions for the
dispatch_overhead noise-reduction analysis.

Input shape (per multirun-trace-campaign.py):
    <session-dir>/run{NNN}/k{K}/events.json
    <session-dir>/manifest.json

Per (run, K) we expect K each of:
    DMA_MM2S_0_START_TASK / DMA_MM2S_0_FINISHED_TASK
    DMA_S2MM_0_START_TASK / DMA_S2MM_0_FINISHED_TASK
(channel-0 events; channel-1 events are spurious HW emissions, ignored.)

Per-run logic:
  1. Filter to row=0 (shim), channel=0, in {MM2S, S2MM}.
  2. Dedup adjacent ts within 2 cyc (known trace decoder artifact).
  3. Sort by ts, pair k-th START with k-th FINISHED.
  4. Sanity-check: duration = FINISHED - START in [0, MAX_PLAUSIBLE_DURATION].
  5. Inter-task gap = START[i+1] - FINISHED[i] for i in 0..len-1.

Aggregation per (K, direction):
  - first_gaps   : gap[0]              first inter-task gap (often higher
                   due to ramp-up / fill latency)
  - steady_gaps  : gap[1:]             second-and-after inter-task gaps
  - durations    : per-task duration   useful sanity check
  Distribution stats: n, mean, std, median, MAD, p5/p25/p50/p75/p95, IQR,
  min, max. Outlier filter: drop values > 5*MAD from median.

Usage:
  ./tools/aggregate-dispatch-overhead.py <session-dir>
  ./tools/aggregate-dispatch-overhead.py <session-dir> --pretty
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MAX_PLAUSIBLE_DURATION = 5000   # cyc; larger means mispaired
DEDUP_WINDOW = 2                # adjacent ts within this delta merged
# Outlier filter disabled by default: inter-task gaps are bimodal (pipelined
# back-to-back tasks give near-zero or negative gaps; serialized tasks give
# ~3000-cyc gaps). A MAD-based filter destroys the bimodal structure that
# is exactly what we want to characterize.

TASK_EVENT_RE = {
    # NAME -> (direction, channel, kind)
    "DMA_MM2S_0_START_TASK":     ("MM2S", 0, "start"),
    "DMA_MM2S_0_FINISHED_TASK":  ("MM2S", 0, "finish"),
    "DMA_S2MM_0_START_TASK":     ("S2MM", 0, "start"),
    "DMA_S2MM_0_FINISHED_TASK":  ("S2MM", 0, "finish"),
}


def dedup_ts(ts_list: list[int]) -> list[int]:
    """Collapse adjacent timestamps within DEDUP_WINDOW; keep earlier."""
    if not ts_list:
        return []
    out = [ts_list[0]]
    for t in ts_list[1:]:
        if t - out[-1] > DEDUP_WINDOW:
            out.append(t)
    return out


def extract_run_gaps(events_path: Path) -> dict:
    """Return per-direction gap data for one (run, K) events.json.

    {
        "by_dir": {
            "MM2S": { "ok": bool, "reason": str | None,
                      "n_starts": int, "n_finishes": int,
                      "durations": [...], "gaps": [...] },
            "S2MM": { ... },
        }
    }
    """
    with open(events_path) as f:
        data = json.load(f)
    events = data.get("events", [])

    # Bucket starts/finishes per direction (channel-0 only)
    starts = defaultdict(list)
    finishes = defaultdict(list)
    for e in events:
        if e.get("row") != 0:
            continue
        info = TASK_EVENT_RE.get(e.get("name", ""))
        if info is None:
            continue
        direction, _channel, kind = info  # channel always 0 in our filter
        if kind == "start":
            starts[direction].append(e["ts"])
        else:
            finishes[direction].append(e["ts"])

    by_dir = {}
    for direction in ("MM2S", "S2MM"):
        s = sorted(dedup_ts(sorted(starts[direction])))
        f = sorted(dedup_ts(sorted(finishes[direction])))
        rec = {
            "n_starts": len(s),
            "n_finishes": len(f),
            "ok": False,
            "reason": None,
            "durations": [],
            "gaps": [],
        }

        if not s or not f:
            rec["reason"] = "no_starts_or_finishes"
        elif len(s) != len(f):
            # Realign: a leading finish without a preceding start (or
            # a trailing start without a following finish) likely indicates
            # a partial run. Try to drop.
            while f and s and f[0] < s[0]:
                f = f[1:]
            while s and f and s[-1] > f[-1]:
                s = s[:-1]
            if len(s) != len(f):
                rec["reason"] = f"len_mismatch s={len(s)} f={len(f)}"
        if rec["reason"] is None:
            if len(s) < 2:
                rec["reason"] = "fewer_than_2_tasks"
            else:
                durs = [fi - si for si, fi in zip(s, f)]
                if any(d < 0 for d in durs):
                    rec["reason"] = f"negative_durations {durs[:3]}"
                elif any(d > MAX_PLAUSIBLE_DURATION for d in durs):
                    rec["reason"] = f"implausible_durations {sorted(durs)[-3:]}"
                else:
                    rec["durations"] = durs
                    rec["gaps"] = [s[i + 1] - f[i] for i in range(len(s) - 1)]
                    rec["ok"] = True
        by_dir[direction] = rec
    return {"by_dir": by_dir}


def summarize(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    vs = sorted(values)
    n = len(vs)
    median = statistics.median(vs)
    mad = statistics.median(abs(v - median) for v in vs)

    def p(q):
        idx = max(0, min(n - 1, int(round((q / 100.0) * (n - 1)))))
        return vs[idx]

    return {
        "n": n,
        "mean": round(statistics.mean(vs), 1),
        "stdev": round(statistics.stdev(vs) if n > 1 else 0.0, 1),
        "median": median,
        "mad": mad,
        "p5": p(5),
        "p25": p(25),
        "p50": p(50),
        "p75": p(75),
        "p95": p(95),
        "iqr": p(75) - p(25),
        "min": vs[0],
        "max": vs[-1],
    }


def walk_session(session_dir: Path) -> dict:
    manifest = json.load(open(session_dir / "manifest.json"))

    # buckets[(K, direction)] = {
    #     "first_gaps":[...], "steady_gaps":[...], "durations":[...], "all_gaps":[...],
    #     "gaps_by_index": {0: [...], 1: [...], 2: [...], ...},
    #     "durations_by_index": {0: [...], 1: [...], ...},
    # }
    buckets = defaultdict(lambda: {
        "first_gaps": [], "steady_gaps": [], "durations": [], "all_gaps": [],
        "gaps_by_index": defaultdict(list),
        "durations_by_index": defaultdict(list),
        "raw_per_run": [],  # one list-of-gaps per run, for visualization
    })
    run_diag = []
    fail_diag = []

    for run_dir in sorted(session_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        run_idx = int(run_dir.name.removeprefix("run"))
        for k_dir in sorted(run_dir.glob("k*")):
            if not k_dir.is_dir():
                continue
            try:
                k = int(k_dir.name.removeprefix("k"))
            except ValueError:
                continue
            meta_path = k_dir / "meta.json"
            events_path = k_dir / "events.json"
            if meta_path.exists():
                meta = json.load(open(meta_path))
                if not meta.get("ok"):
                    fail_diag.append({
                        "run": run_idx, "k": k,
                        "reason": f"runner_rc={meta.get('runner_rc')} parser_rc={meta.get('parser_rc')}"
                    })
                    continue
            if not events_path.exists():
                fail_diag.append({"run": run_idx, "k": k, "reason": "no_events_json"})
                continue

            res = extract_run_gaps(events_path)
            for direction in ("MM2S", "S2MM"):
                rec = res["by_dir"].get(direction, {})
                if rec.get("ok"):
                    gaps = rec["gaps"]
                    durs = rec["durations"]
                    b = buckets[(k, direction)]
                    if gaps:
                        b["first_gaps"].append(gaps[0])
                        b["steady_gaps"].extend(gaps[1:])
                        b["all_gaps"].extend(gaps)
                        for i, g in enumerate(gaps):
                            b["gaps_by_index"][i].append(g)
                        b["raw_per_run"].append({
                            "run": run_idx, "gaps": gaps, "durations": durs,
                        })
                    b["durations"].extend(durs)
                    for i, d in enumerate(durs):
                        b["durations_by_index"][i].append(d)
                else:
                    fail_diag.append({
                        "run": run_idx, "k": k, "direction": direction,
                        "reason": rec.get("reason"),
                        "n_starts": rec.get("n_starts"),
                        "n_finishes": rec.get("n_finishes"),
                    })
            run_diag.append({"run": run_idx, "k": k, "ok": True})

    summary = {}
    raw = {}
    for (k, direction), data in sorted(buckets.items()):
        key = f"k{k}.{direction}"
        summary[key] = {
            "k": k,
            "direction": direction,
            "all_gaps": summarize(data["all_gaps"]),
            "first_gaps": summarize(data["first_gaps"]),
            "steady_gaps": summarize(data["steady_gaps"]),
            "durations": summarize(data["durations"]),
            "gaps_by_index": {
                str(i): summarize(data["gaps_by_index"][i])
                for i in sorted(data["gaps_by_index"])
            },
            "durations_by_index": {
                str(i): summarize(data["durations_by_index"][i])
                for i in sorted(data["durations_by_index"])
            },
        }
        raw[key] = data["raw_per_run"]

    return {
        "session": manifest.get("session"),
        "manifest": manifest,
        "summary": summary,
        "raw_per_run": raw,
        "run_diag_count": len(run_diag),
        "fail_diag_count": len(fail_diag),
        "fail_diag_sample": fail_diag[:30],
    }


def pretty_print(agg: dict) -> None:
    print(f"\n== Multirun dispatch-overhead aggregation ==")
    print(f"  session    : {agg['session']}")
    m = agg["manifest"]
    print(f"  iterations : {m.get('n_iterations')}")
    print(f"  ok / fail  : {m.get('n_ok')} / {m.get('n_fail')}")
    print(f"  elapsed    : {m.get('total_elapsed_s')}s")
    print(f"  successful : {agg['run_diag_count']}, failures: {agg['fail_diag_count']}")
    print()
    print(f"  {'bucket':<32} {'n':>5} {'med':>6} {'MAD':>5} {'IQR':>5} "
          f"{'mean':>7} {'std':>5} {'p5':>7} {'p95':>7} {'min':>7} {'max':>7}")

    def row(label, d):
        if d.get("n", 0) > 0:
            print(f"  {label:<32} {d['n']:>5} {d['median']:>6} {d['mad']:>5} "
                  f"{d['iqr']:>5} {d['mean']:>7} {d['stdev']:>5} "
                  f"{d['p5']:>7} {d['p95']:>7} {d['min']:>7} {d['max']:>7}")

    for key in sorted(agg["summary"]):
        s = agg["summary"][key]
        for kind in ("durations", "all_gaps", "first_gaps", "steady_gaps"):
            row(f"{key}.{kind}", s[kind])
        for i in sorted(s["gaps_by_index"], key=int):
            row(f"{key}.gap[{i}]", s["gaps_by_index"][i])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    if not args.session_dir.is_dir():
        print(f"error: {args.session_dir} is not a directory", file=sys.stderr)
        return 1

    agg = walk_session(args.session_dir)
    out_path = args.out or (args.session_dir / "aggregated.json")
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"wrote {out_path}")

    if args.pretty:
        pretty_print(agg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
