#!/usr/bin/env python3
"""
port-span-cadence.py -- SPAN-based PORT_RUNNING cadence metric (#140/#141).

Supersedes port-cadence-baseline.py's *frame-record* counting, which measured
the wrong thing. The old tool counted every trace frame that NAMES a port slot,
including the re-checkpoint frames the held-level encoder must emit whenever a
*concurrent* signal toggles during a held level (upstream parse_trace
deactivates all active events on any cycles>0 frame, so held levels are kept
alive by being re-named in cycles==0 frames at every foreign edge -- HW does
this too). That inflates a continuously-held port into many apparent
"sub-bursts", cross-contaminated by every other traced signal's edge timing,
AND manufactures run-to-run *variance* that does not exist on silicon.

The correct unit is the **span**: a held PORT_RUNNING level persists through
re-checkpoints, so one continuous run of the port being active is ONE B/E span
regardless of how many foreign toggles land inside it. This tool consumes the
oracle's Perfetto B/E output (parse-trace.py --out-perfetto), reconstructs
per-port spans, and groups them into sub-bursts by **idle gap** (prev end ->
next begin > --gap), the physically meaningful definition.

Measured this way, NPU1 add_one_using_dma PORT_RUNNING is DETERMINISTIC across
15 HW runs (slot0=1, slot1=5, slot4=3, slot5=4; std=0). See
docs/superpowers/findings/2026-06-16-port-cadence-metric-was-frame-records.md.

Usage:
  # campaign session (generates oracle perfetto per run, names from events.json):
  ./tools/port-span-cadence.py --session build/experiments/.../<session> [--limit N]
  # pre-decoded perfetto JSON files:
  ./tools/port-span-cadence.py file1.perfetto.json file2.perfetto.json
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLIR_AIE = REPO.parent / "mlir-aie"
PORT_KINDS = ("PORT_RUNNING_", "PORT_STALLED_", "PORT_IDLE_", "STREAM_STARVATION")


def gen_perfetto(trace_bin: Path, mlir: Path) -> Path | None:
    """Decode trace_raw.bin -> Perfetto B/E JSON via the oracle (parse-trace.py).
    Cached next to the bin. Returns the path, or None on failure."""
    out = trace_bin.with_suffix(".perfetto.json")
    if out.exists():
        return out
    env = dict(os.environ)
    env["PYTHONPATH"] = str(MLIR_AIE / "install" / "python")
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "parse-trace.py"),
         "--trace-bin", str(trace_bin), "--xclbin-mlir", str(mlir),
         "--trace-mode", "auto", "--out-perfetto", str(out)],
        env=env, capture_output=True)
    return out if r.returncode == 0 and out.exists() else None


def spans_from_perfetto(pf: Path, events_json: Path | None):
    """Return {(col,row,name): [(begin,end), ...]} of PORT spans.

    Perfetto carries spans by pid=tile / tid=slot but with blank thread names;
    the (tile,slot)->name map is recovered from the co-located events.json.
    """
    d = json.load(open(pf))
    evs = d["traceEvents"] if isinstance(d, dict) and "traceEvents" in d else d
    pid2tile = {}
    for e in evs:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            nm = e["args"]["name"]  # e.g. "memtile(1,1)" / "shim(0,1)" / "mem(2,1)"
            if "(" in nm:
                c, r = nm.split("(")[1].rstrip(")").split(",")
                pid2tile[e["pid"]] = (int(c), int(r))
    name_map = {}
    if events_json and events_json.exists():
        for x in json.load(open(events_json)).get("events", []):
            name_map[(x["col"], x["row"], x["slot"])] = x["name"]
    stack = defaultdict(list)
    spans = defaultdict(list)
    for e in evs:
        pid = e.get("pid")
        if pid not in pid2tile:
            continue
        col, row = pid2tile[pid]
        key = (pid, e.get("tid"))
        if e.get("ph") == "B":
            stack[key].append(e["ts"])
        elif e.get("ph") == "E" and stack[key]:
            b = stack[key].pop()
            nm = name_map.get((col, row, e["tid"]), f"slot{e['tid']}")
            if any(k in nm for k in PORT_KINDS):
                spans[(col, row, nm)].append((b, max(e["ts"], b)))
    return {k: sorted(v) for k, v in spans.items()}


def subbursts(spans, gap):
    """Merge spans whose idle gap (prev end -> next begin) <= gap."""
    if not spans:
        return 0, []
    groups = [list(spans[0])]
    gaps = []
    for b, e in spans[1:]:
        idle = b - groups[-1][1]
        if idle > gap:
            gaps.append(idle)
            groups.append([b, e])
        else:
            groups[-1][1] = max(groups[-1][1], e)
    return len(groups), gaps


def collect_runs(session: Path, limit):
    """Yield (kernel_label, [(trace_bin, events_json), ...]) for a campaign dir."""
    by_kernel = defaultdict(list)
    for run_dir in sorted(session.glob("run*")):
        for kdir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            tb, ej = kdir / "trace_raw.bin", kdir / "events.json"
            meta = kdir / "meta.json"
            if not (tb.exists() and ej.exists()):
                continue
            if meta.exists() and not json.load(open(meta)).get("ok", True):
                continue
            by_kernel[kdir.name].append((tb, ej))
    for k in by_kernel:
        by_kernel[k] = by_kernel[k][:limit] if limit else by_kernel[k]
    return by_kernel


def report(label, per_run_spans, gap, out):
    ports = sorted({k for run in per_run_spans for k in run})
    print(f"\n{'='*72}\n{label}   ({len(per_run_spans)} runs, span-based, idle-gap>{gap})\n{'='*72}")
    rep = {}
    for port in ports:
        counts, patterns = [], []
        for run in per_run_spans:
            n, gaps = subbursts(run.get(port, []), gap)
            counts.append(n)
            patterns.append(tuple(gaps))
        uniq = sorted(set(counts))
        std = statistics.pstdev(counts) if len(counts) > 1 else 0.0
        modal = Counter(patterns).most_common(1)[0]
        verdict = "DET" if std == 0 and len(uniq) == 1 else "stoch"
        col, row, name = port
        print(f"  ({col},{row}) {name:<26} sub-bursts mean={statistics.mean(counts):.2f} "
              f"std={std:.2f} distinct={uniq} [{verdict}]")
        print(f"        modal idle-gaps ({modal[1]}/{len(per_run_spans)}): {list(modal[0])}")
        rep[f"{col},{row},{name}"] = {
            "counts": counts, "mean": statistics.mean(counts), "pstdev": std,
            "distinct": uniq, "verdict": verdict, "modal_gaps": list(modal[0])}
    out[label] = rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("perfetto", type=Path, nargs="*",
                    help="pre-decoded Perfetto JSON file(s)")
    ap.add_argument("--session", type=Path, default=None,
                    help="campaign session dir (decodes trace_raw.bin per run)")
    ap.add_argument("--mlir", type=Path, default=None,
                    help="input_with_addresses.mlir for the kernel "
                         "(default: derive from kernel name under mlir-aie build)")
    ap.add_argument("--limit", type=int, default=None, help="cap runs per kernel")
    ap.add_argument("--gap", type=int, default=2,
                    help="max idle gap merged within a sub-burst (default 2)")
    ap.add_argument("--label", default="span cadence")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out = {}
    if args.session:
        for kernel, runs in collect_runs(args.session, args.limit).items():
            mlir = args.mlir or (MLIR_AIE / "build" / "test" / "npu-xrt" / kernel /
                                 "chess" / "aie_arch.mlir.prj" / "input_with_addresses.mlir")
            per_run = []
            for tb, ej in runs:
                pf = gen_perfetto(tb, mlir)
                if pf:
                    per_run.append(spans_from_perfetto(pf, ej))
            if per_run:
                report(f"{args.label}: {kernel}", per_run, args.gap, out)
    else:
        per_run = [spans_from_perfetto(p, p.with_suffix("").with_suffix(".json")
                                       if False else None) for p in args.perfetto]
        report(args.label, per_run, args.gap, out)

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
