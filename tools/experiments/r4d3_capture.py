#!/usr/bin/env python3
"""r4d3 driver: build one egress-race configuration, run it N times on HW (or
the emulator), decode the stale-word count of every run and append one JSON
line per run to <root>/results.jsonl. Evidence (aie.mlir, xclbin, insts.bin,
raw out_r*.bin) stays under <root>/<name>/.

  r4d3_capture.py --hops 1 --d1 200 --d2 20 --reps 5
  r4d3_capture.py --hops 1 --reps 3 --emu     # emulator run, same xclbin
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from r4d3_decode import decode  # noqa: E402
from r4d3_egress_race import build_mlir  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "build/experiments/r4-d3"
RUNNER = REPO / "bridge-runner/build/bridge-trace-runner"


def sh(cmd, cwd=None, env=None, timeout=120):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        raise RuntimeError(f"rc={r.returncode}: {' '.join(map(str, cmd))}")
    return r


def kernlog_tail():
    r = subprocess.run(["journalctl", "-k", "-b", "--no-pager", "-g", "amdxdna|npu|xdna", "-n", "5"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def build(name, hops, d1, d2, poll, source="dma", free_sink=False, via_fifo=False, align=0):
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "aie.mlir").write_text(build_mlir(hops, d1, d2, poll, source=source, free_sink=free_sink,
                                       via_fifo=via_fifo, align=align))
    if not (d / "aie.xclbin").exists():
        sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--get-npu-insts",
            "--xclbin-name=aie.xclbin", "--npu-insts-name=insts.bin", "aie.mlir"], cwd=d)
    return d


def run(d, reps, emu, tag):
    env = os.environ.copy()
    if emu:
        env["XDNA_EMU"] = "1"
        env["XDNA_EMU_RUNTIME"] = "release"
    else:
        env.pop("XDNA_EMU", None)
        env["XDNA_EMU_RUNTIME"] = "release"
    rows = []
    for rep in range(1, reps + 1):
        out = d / f"out_{tag}_r{rep}.bin"
        t0 = time.time()
        r = subprocess.run([str(RUNNER), "--xclbin", "aie.xclbin", "--instr", "insts.bin",
                            "--trace-out", str(d / "trace_unused.bin"), "--trace-size", "16",
                            "--trace-buf-idx", "1", "--output", str(out)],
                           cwd=d, env=env, capture_output=True, text=True, timeout=120)
        row = {"name": d.name, "backend": "emu" if emu else "hw", "rep": rep,
               "rc": r.returncode, "wall_s": round(time.time() - t0, 2)}
        if r.returncode == 0 and out.exists():
            row.update({k: v for k, v in decode(out).items() if k != "pattern"})
            row["pattern"] = decode(out)["pattern"]
        else:
            row["stderr"] = r.stderr[-400:]
        rows.append(row)
        with open(ROOT / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(d.name, row.get("backend"), "rep", rep, "stale", row.get("stale"),
              "mono", row.get("monotone"), "rc", row["rc"], flush=True)
        if r.returncode != 0:
            print("STOP: run failed; kernel log:\n" + kernlog_tail(), file=sys.stderr)
            break
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--d1", type=int, default=200)
    ap.add_argument("--d2", type=int, default=20)
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--emu", action="store_true")
    ap.add_argument("--source", choices=("dma", "core"), default="dma")
    ap.add_argument("--free-sink", action="store_true",
                    help="control: sink never held, overwrite races a free transfer")
    ap.add_argument("--via-fifo", action="store_true",
                    help="route through the source tile's 16-deep stream-switch FIFO")
    ap.add_argument("--align", type=int, default=0, choices=range(4),
                    help="BD start offset in words (128-bit granule phase)")
    ap.add_argument("--name")
    a = ap.parse_args()
    name = a.name or f"{'c' if a.source == 'core' else 'h'}{a.hops}_d1{a.d1}_d2{a.d2}{'_poll' if a.poll else ''}{'_free' if a.free_sink else ''}{'_fifo' if a.via_fifo else ''}{'_a%d' % a.align if a.align else ''}"
    d = build(name, a.hops, a.d1, a.d2, a.poll, a.source, a.free_sink, a.via_fifo, a.align)
    run(d, a.reps, a.emu, "emu" if a.emu else "hw")
