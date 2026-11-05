#!/usr/bin/env python3
"""Run the op-cost probe cases on silicon at the two clock points and record raw evidence.

  python3 tools/fw_op_cost_run.py <cases_dir> <out_dir> --tag r1 --dpm 1 0 [--only CASE ..]

Per run it waits for the shared NPU, counts TDR lines in the kernel log before and after, starts
`fw_op_cost_host` (which holds its hardware context alive across the launch), samples the clocks
with `npu-clocks` once at READY (context alive, before the launch) and once at DONE (after it),
and writes `<out_dir>/<tag>/dpm<d>/<case>/{out.bin,run.json,host.log}`. A run is valid only when
both samples equal the expected pair for the DPM point (DPM 1: default, 600/1028 MHz; DPM 0:
context QoS gops=1, fps=1000, 400/800 MHz). Recorded evidence is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

EXPECTED = {1: (600, 1028), 0: (400, 800)}
QOS = {1: [], 0: ["--gops", "1", "--fps", "1000"]}
HW_JOBS = re.compile(r"emu-bridge-test|harness suite hw|isa-test")
_CLOCK = re.compile(r"MP-NPU Clock (\d+) MHz, H Clock (\d+) MHz")


def parse_clocks(line: str):
    m = _CLOCK.search(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def clocks_valid(dpm: int, before, after) -> bool:
    return before == EXPECTED[dpm] and after == EXPECTED[dpm]


def host_qos_args(dpm: int) -> list[str]:
    return list(QOS[dpm])


def count_tdr(kernel_log: str) -> int:
    return sum(1 for l in kernel_log.splitlines() if "tdr" in l.lower())


def hw_busy(ps_args: str) -> bool:
    """True if another hardware job is listed (the shared-NPU rule's grep, minus shell wrappers)."""
    return any(HW_JOBS.search(l) and not re.search(r"grep|zsh -c|bash -c", l)
               for l in ps_args.splitlines())


def wait_for_npu(poll_s: float = 10) -> None:
    while hw_busy(subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout):
        time.sleep(poll_s)


def tdr_total() -> int:
    r = subprocess.run(["journalctl", "-k", "-b", "--no-pager"], capture_output=True, text=True)
    return count_tdr(r.stdout)


def sample_clocks(tool: Path):
    r = subprocess.run([str(tool), "/dev/accel/accel0", "0", "1"], capture_output=True, text=True)
    return parse_clocks(r.stdout)


def run_case(case_dir: Path, out_dir: Path, dpm: int, host: Path, tool: Path,
             timeout_s: float = 60) -> dict:
    meta = json.loads((case_dir / "case.json").read_text())
    if (out_dir / "run.json").exists():
        raise RuntimeError(f"{out_dir} already holds a run")
    out_dir.mkdir(parents=True, exist_ok=True)
    wait_for_npu()
    tdr0 = tdr_total()
    cmd = [str(host), "-x", str(case_dir / "aie.xclbin"), "-i", str(case_dir / "insts.bin"),
           "-k", "MLIR_AIE", "--out-size", str(4 * meta["results_words"]),
           "--out-file", str(out_dir / "out.bin"), *host_qos_args(dpm)]
    before = after = None
    wall_us = None
    log = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t_end = time.monotonic() + timeout_s
    try:
        for line in p.stdout:
            log.append(line)
            if line.startswith("READY"):
                before = sample_clocks(tool)
            elif line.startswith("DONE"):
                after = sample_clocks(tool)
                wall_us = int(line.split("wall_us=")[1])
            if time.monotonic() > t_end:
                p.kill()
                break
        rc = p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = -9
    (out_dir / "host.log").write_text("".join(log))
    rec = {"case": meta["name"], "dpm": dpm, "rc": rc, "wall_us": wall_us,
           "clocks_before": before, "clocks_after": after,
           "valid": rc == 0 and clocks_valid(dpm, before, after),
           "tdr_before": tdr0, "tdr_after": tdr_total()}
    (out_dir / "run.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dpm", type=int, nargs="+", default=[1, 0], choices=sorted(EXPECTED))
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--host", default=None)
    ap.add_argument("--clocks", default=None)
    a = ap.parse_args(argv)
    root = Path(a.cases)
    here = root.resolve().parent
    host = Path(a.host or here / "fw_op_cost_host")
    tool = Path(a.clocks or here / "npu-clocks")
    bad = 0
    for dpm in a.dpm:
        for d in sorted(p for p in root.iterdir() if (p / "insts.bin").is_file()):
            if a.only and d.name not in a.only:
                continue
            rec = run_case(d, Path(a.out) / a.tag / f"dpm{dpm}" / d.name, dpm, host, tool)
            flag = "ok" if rec["valid"] else "INVALID"
            bad += not rec["valid"]
            print(f"{a.tag} dpm{dpm} {d.name}: {flag} rc={rec['rc']} clocks={rec['clocks_before']}"
                  f"/{rec['clocks_after']} tdr={rec['tdr_after'] - rec['tdr_before']}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
