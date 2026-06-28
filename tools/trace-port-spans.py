#!/usr/bin/env python3
"""Decode PORT_RUNNING span run-lengths per tile from a bridge trace dir.

Reconstructs PORT_RUNNING level spans (begin/end durations in cycles),
labeled by tile and slot, from a bridge-test result directory. This is the
run-length view used for S2MM/MM2S ingress-depth measurement (#140 device-model
audit): e.g. a clean [16,16,16,16] means the recv port staged a full 16-word BD
per transfer, while [8,8,2,6,...] means it backpressured after the FIFO depth.

Spans come from parse-trace.py --out-perfetto (mode-0 PORT_RUNNING spans), so
the traced tiles must be in event_time mode. Force core tiles to mode 0 with
XDNA_TRACE_MODE=event_time at capture time (memtile/shim/memmod are always
mode 0). Slot->event labels come from each side's trace_config.json.

Usage:
  tools/trace-port-spans.py <results_dir> <test_name> <compiler> [side ...]
  # default: results_dir=build/bridge-test-results/latest, chess, hw + emu

Example capture (both tiles, event_time) then decode:
  XDNA_TRACE_MODE=event_time \\
  XDNA_TRACE_CORE_EVENTS="PORT_RUNNING_0,PORT_RUNNING_1" \\
  XDNA_TRACE_MEMTILE_EVENTS="PORT_RUNNING_0,PORT_RUNNING_4" \\
    ./scripts/emu-bridge-test.sh --chess-only --trace -v add_one_using_dma
  tools/trace-port-spans.py build/bridge-test-results/latest add_one_using_dma
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MLIR_AIE_PY = "/home/triple/npu-work/mlir-aie/install/python"


def _find_mlir(name, comp):
    base = f"/home/triple/npu-work/mlir-aie/build/test/npu-xrt/{name}/{comp}"
    r = subprocess.run(["find", base, "-name", "input_with_addresses.mlir",
                        "-print", "-quit"], capture_output=True, text=True)
    return r.stdout.strip()


def spans_for(results_dir, name, comp, side, tmp):
    tdir = Path(results_dir) / f"{name}.{comp}.{side}"
    pf = Path(tmp) / f"perfetto_{side}.json"
    subprocess.run(
        ["python3", str(REPO / "tools" / "parse-trace.py"),
         "--trace-bin", str(tdir / "trace_raw.bin"),
         "--xclbin-mlir", _find_mlir(name, comp),
         "--trace-mode", "auto", "--out-perfetto", str(pf)],
        env={**os.environ, "PYTHONPATH": MLIR_AIE_PY}, capture_output=True)
    cfg = json.load(open(tdir / "trace_config.json"))
    tiles = {(t["row"], t.get("module")): t["events"] for t in cfg["tiles_traced"]}
    d = json.load(open(pf))
    evs = d["traceEvents"] if isinstance(d, dict) and "traceEvents" in d else d
    pname = {e["pid"]: e["args"]["name"] for e in evs
             if e.get("ph") == "M" and e["name"] == "process_name"}
    stack, spans = defaultdict(list), defaultdict(list)
    for e in evs:
        ph = e.get("ph")
        if ph not in ("B", "E"):
            continue  # skip metadata (M) events, which carry no tid
        key = (e["pid"], e["tid"])
        if ph == "B":
            stack[key].append(e["ts"])
        elif stack[key]:
            spans[key].append(e["ts"] - stack[key].pop())
    out = []
    for (pid, tid), durs in sorted(spans.items()):
        m = re.match(r"(\w+)\((\d+),(\d+)\)", pname.get(pid, ""))
        if not m:
            continue
        nm, row = m.group(1), int(m.group(2))
        mod = {"core": "core", "mem": "mem"}.get(nm)
        evlist = tiles.get((row, mod))
        ev = evlist[tid] if evlist and tid < len(evlist) else f"slot{tid}"
        if "PORT_RUNNING" in str(ev):
            out.append((pname[pid], ev, durs))
    return out


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else "build/bridge-test-results/latest"
    name = sys.argv[2] if len(sys.argv) > 2 else "add_one_using_dma"
    comp = sys.argv[3] if len(sys.argv) > 3 else "chess"
    sides = sys.argv[4:] or ["hw", "emu"]
    with tempfile.TemporaryDirectory() as tmp:
        for side in sides:
            print(f"\n############ {side.upper()} ############")
            for pn, ev, durs in spans_for(results, name, comp, side, tmp):
                print(f"  {pn:14s} {ev:16s} durs={durs}")


if __name__ == "__main__":
    main()
