#!/usr/bin/env python3
"""Run a calibration sweep on real NPU hardware.

For each (kind, target, count) configuration in the sweep, this script:
  1. Generates a calibration kernel via gen_kernel.py
  2. Compiles it via aiecc.py (Peano)
  3. Runs it on the real NPU via bridge-trace-runner
  4. Parses the trace via parse-trace.py
  5. Records anchor cycle delta in the measurements JSON

Compile is parallelized; HW runs are serial (NPU is single-tenant);
trace parsing is parallel.

Usage:
  python3 run_sweep.py --out build/calib/sweep-001.json [--reps 3]
  python3 run_sweep.py --config calib_config.json --out sweep.json

Default sweep covers write32 -> compute, write32 -> mem, write32 -> shim
at N in {16, 64, 256, 1024, 4096} with 3 reps each (45 measurements).

Output JSON format:
  {
    "schema_version": 1,
    "device": "npu1",
    "compiler": "peano",
    "measurements": [
      {"kind": "write32", "target": "compute", "count": 1024,
       "payload": 0, "rep": 0, "hw_cycles": 116397, "anchor_a_ts": ...,
       "anchor_b_ts": ..., "core_ticks": 11981}
    ]
  }
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GEN_KERNEL = REPO_ROOT / "tools" / "calibration" / "gen_kernel.py"
PARSE_TRACE = REPO_ROOT / "tools" / "parse-trace.py"
BRIDGE_RUNNER = REPO_ROOT / "bridge-runner" / "build" / "bridge-trace-runner"

DEFAULT_SWEEP = [
    {"kind": "write32", "target": "compute", "counts": [16, 64, 256, 1024, 4096]},
    {"kind": "write32", "target": "mem",     "counts": [16, 64, 256, 1024, 4096]},
    {"kind": "write32", "target": "shim",    "counts": [16, 64, 256, 1024, 4096]},
]


def run(cmd: list, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def gen_one(work_dir: Path, kind: str, target: str, count: int, payload: int) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, str(GEN_KERNEL),
        "--kind", kind, "--target", target,
        "--count", str(count), "--payload", str(payload),
        "--out", str(work_dir),
    ])


def compile_one(work_dir: Path) -> bool:
    """Compile the kernel in `work_dir`. Returns True on success."""
    # Create dummy input buffers (4 unused kernargs).
    for i in range(4):
        (work_dir / f"dummy{i}.bin").write_bytes(b"\0" * 64)

    log_path = work_dir / "compile.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run([
            "nice", "-n", "19",
            "aiecc.py",
            "--no-aiesim",
            "--no-xchesscc", "--no-xbridge",
            "--aie-generate-xclbin",
            "--aie-generate-npu-insts",
            "--no-compile-host",
            "--xclbin-name=final.xclbin",
            "--npu-insts-name=insts.bin",
            "aie.mlir",
        ], cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode == 0 and (work_dir / "final.xclbin").exists()


def run_on_hw(work_dir: Path, trace_size: int = 65536) -> bool:
    """Run the kernel on real NPU. Returns True on success."""
    log_path = work_dir / "run.log"
    cmd = [
        str(BRIDGE_RUNNER),
        "--xclbin", "final.xclbin",
        "--instr", "insts.bin",
        "--kernel", "MLIR_AIE",
        "--input", "dummy0.bin", "--input", "dummy1.bin",
        "--input", "dummy2.bin", "--input", "dummy3.bin",
        "--trace-out", "trace.bin",
        "--trace-size", str(trace_size),
    ]
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode == 0 and (work_dir / "trace.bin").exists()


def parse_one(work_dir: Path) -> dict:
    """Parse the trace into events.json and extract anchor cycle delta.

    Returns a dict with keys:
      - hw_cycles: B - A delta, or None if anchors missing
      - anchor_a_ts, anchor_b_ts
      - core_ticks: count of INSTR_EVENT_0 firings
    """
    log_path = work_dir / "parse.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run([
            sys.executable, str(PARSE_TRACE),
            "--trace-bin", "trace.bin",
            "--xclbin-mlir", "aie.mlir.prj/input_with_addresses.mlir",
            "--trace-mode", "event_time",
            "--decoder", "ours",
            "--out-events", "events.json",
        ], cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        return {"hw_cycles": None, "error": "parse-trace failed"}

    try:
        events = json.loads((work_dir / "events.json").read_text())["events"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        return {"hw_cycles": None, "error": f"events.json read: {e}"}

    # Slot 0 = INSTR_EVENT_0, slot 1 = USER_EVENT_0, slot 2 = USER_EVENT_1
    # (order matches the aie.trace.event<...> declarations in gen_kernel.py).
    anchors_a = [e for e in events if e["slot"] == 1]
    anchors_b = [e for e in events if e["slot"] == 2]
    ticks = [e for e in events if e["slot"] == 0]

    if not anchors_a or not anchors_b:
        return {
            "hw_cycles": None,
            "error": f"missing anchors: a={len(anchors_a)} b={len(anchors_b)}",
        }
    return {
        "hw_cycles": anchors_b[0]["ts"] - anchors_a[0]["ts"],
        "anchor_a_ts": anchors_a[0]["ts"],
        "anchor_b_ts": anchors_b[0]["ts"],
        "core_ticks": len(ticks),
    }


def expand_sweep(sweep_groups: list, reps: int) -> list:
    """Expand sweep groups into individual (kind, target, count, rep) tasks."""
    tasks = []
    for grp in sweep_groups:
        kind = grp["kind"]
        target = grp["target"]
        payload = grp.get("payload", 0)
        for count in grp["counts"]:
            for rep in range(reps):
                tasks.append({
                    "kind": kind, "target": target, "count": count,
                    "payload": payload, "rep": rep,
                })
    return tasks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="Output measurements JSON path")
    p.add_argument("--workdir", type=Path,
                   default=REPO_ROOT / "build" / "calibration" / "sweep",
                   help="Per-task working directory root")
    p.add_argument("--reps", type=int, default=3,
                   help="Repetitions per (kind, target, count)")
    p.add_argument("--compile-jobs", type=int, default=4,
                   help="Parallel compile workers")
    p.add_argument("--config", type=Path,
                   help="Optional sweep config JSON; overrides default sweep")
    args = p.parse_args()

    sweep_groups = DEFAULT_SWEEP
    if args.config:
        sweep_groups = json.loads(args.config.read_text())

    tasks = expand_sweep(sweep_groups, args.reps)
    print(f"Sweep: {len(tasks)} tasks across "
          f"{sum(len(g['counts']) for g in sweep_groups)} configurations.",
          file=sys.stderr)

    args.workdir.mkdir(parents=True, exist_ok=True)

    # Phase 1: generate + compile in parallel.
    print("[phase 1/3] generating + compiling kernels...", file=sys.stderr)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.compile_jobs) as pool:
        futs = {}
        for t in tasks:
            # rep=0 generates+compiles; reps 1..N reuse rep=0's xclbin.
            if t["rep"] != 0:
                continue
            tag = f"{t['kind']}_{t['target']}_n{t['count']}_p{t['payload']}"
            wd = args.workdir / tag
            t["work_dir"] = wd
            futs[pool.submit(_gen_compile, wd, t)] = tag
        compile_failures = []
        for fut in as_completed(futs):
            tag = futs[fut]
            ok = fut.result()
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {tag}", file=sys.stderr)
            if not ok:
                compile_failures.append(tag)
    print(f"[phase 1/3] done in {time.time()-t0:.1f}s, "
          f"{len(compile_failures)} compile failures", file=sys.stderr)

    # Wire up rep > 0 tasks to share rep=0's work_dir.
    rep0_dirs = {(t["kind"], t["target"], t["count"], t["payload"]): t["work_dir"]
                 for t in tasks if t["rep"] == 0}
    for t in tasks:
        if t["rep"] != 0:
            t["work_dir"] = rep0_dirs[(t["kind"], t["target"], t["count"], t["payload"])]

    # Phase 2: HW runs (SERIAL -- NPU is single tenant).
    print("[phase 2/3] running on NPU (serial)...", file=sys.stderr)
    t0 = time.time()
    for i, t in enumerate(tasks):
        wd = t["work_dir"]
        if not (wd / "final.xclbin").exists():
            t["hw_status"] = "compile_skipped"
            continue
        # Each rep gets its own trace.bin; bridge-runner overwrites by default
        # so we run, then immediately parse, then move trace aside.
        ok = run_on_hw(wd)
        t["hw_status"] = "ok" if ok else "fail"
        # Snapshot trace + events files per rep.
        if ok:
            (wd / f"trace_rep{t['rep']}.bin").write_bytes((wd / "trace.bin").read_bytes())
        if (i + 1) % 10 == 0 or i + 1 == len(tasks):
            print(f"  [{i+1}/{len(tasks)}] elapsed {time.time()-t0:.1f}s",
                  file=sys.stderr)
    print(f"[phase 2/3] done in {time.time()-t0:.1f}s", file=sys.stderr)

    # Phase 3: parse traces in parallel (parse-trace.py reads trace.bin in
    # the work dir, so we need to swap each rep's trace.bin in turn -- do
    # this serially per work-dir but parallel across work-dirs).
    print("[phase 3/3] parsing traces...", file=sys.stderr)
    t0 = time.time()
    measurements = []
    for t in tasks:
        wd = t["work_dir"]
        if t.get("hw_status") != "ok":
            measurements.append({
                **{k: t[k] for k in ("kind", "target", "count", "payload", "rep")},
                "hw_cycles": None,
                "error": t.get("hw_status", "unknown"),
            })
            continue
        # Swap in this rep's trace and parse.
        rep_trace = wd / f"trace_rep{t['rep']}.bin"
        if rep_trace.exists():
            (wd / "trace.bin").write_bytes(rep_trace.read_bytes())
        result = parse_one(wd)
        measurements.append({
            **{k: t[k] for k in ("kind", "target", "count", "payload", "rep")},
            **result,
        })
    print(f"[phase 3/3] done in {time.time()-t0:.1f}s", file=sys.stderr)

    # Write the measurements JSON.
    out_doc = {
        "schema_version": 1,
        "device": "npu1",
        "compiler": "peano",
        "sweep_groups": sweep_groups,
        "reps": args.reps,
        "measurements": measurements,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2) + "\n")

    n_ok = sum(1 for m in measurements if m.get("hw_cycles") is not None)
    print(f"\nWrote {args.out} ({n_ok}/{len(measurements)} successful measurements)",
          file=sys.stderr)
    return 0 if n_ok > 0 else 1


def _gen_compile(work_dir: Path, task: dict) -> bool:
    try:
        gen_one(work_dir, task["kind"], task["target"], task["count"], task["payload"])
        return compile_one(work_dir)
    except subprocess.CalledProcessError:
        return False


if __name__ == "__main__":
    sys.exit(main())
