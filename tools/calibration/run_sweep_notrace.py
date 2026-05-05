#!/usr/bin/env python3
"""Run a no-trace calibration sweep on real NPU hardware.

Like run_sweep.py but generates kernels with `--no-trace` and measures
wall-clock kernel execution time (microseconds) via the bridge-runner's
batch-stdin mode. The trace controller is not configured at all, so any
trace-related artifacts cannot affect measurements.

Pairs cleanly with run_sweep.py results: same kernel structure, same
control packets, but no trace overhead.

Usage:
  python3 run_sweep_notrace.py --config sweep.json --out result.json [--reps 50]

Output JSON format:
  {
    "schema_version": 1,
    "trace": "off",
    "measurements": [
      {"kind": "write32", "count": 100, "rep": 0, "elapsed_us": 412, ...}
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
BRIDGE_RUNNER = REPO_ROOT / "bridge-runner" / "build" / "bridge-trace-runner"


def gen_one(work_dir: Path, task: dict) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(GEN_KERNEL),
        "--kind", task["kind"],
        "--count", str(task["count"]),
        "--payload", str(task["payload"]),
        "--out", str(work_dir),
        "--no-trace",
    ]
    if "target_col" in task or "target_row" in task:
        cmd += ["--target-col", str(task.get("target_col", 0))]
        cmd += ["--target-row", str(task.get("target_row", 2))]
    if "anchor_col" in task:
        cmd += ["--anchor-col", str(task["anchor_col"])]
    if "anchor_row" in task:
        cmd += ["--anchor-row", str(task["anchor_row"])]
    if "device" in task:
        cmd += ["--device", task["device"]]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def compile_one(work_dir: Path) -> bool:
    for i in range(4):
        (work_dir / f"dummy{i}.bin").write_bytes(b"\0" * 64)
    log_path = work_dir / "compile.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run([
            "nice", "-n", "19",
            "aiecc.py",
            "--no-aiesim", "--no-xchesscc", "--no-xbridge",
            "--aie-generate-xclbin",
            "--aie-generate-npu-insts",
            "--no-compile-host",
            "--xclbin-name=final.xclbin",
            "--npu-insts-name=insts.bin",
            "aie.mlir",
        ], cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode == 0 and (work_dir / "final.xclbin").exists()


def task_tag(task: dict) -> str:
    target_part = f"c{task.get('target_col', 0)}r{task.get('target_row', 2)}"
    return f"{task['kind']}_{target_part}_n{task['count']}_p{task['payload']}"


def expand_sweep(sweep_groups: list, reps: int) -> list:
    tasks = []
    for grp in sweep_groups:
        base = {k: v for k, v in grp.items() if k != "counts"}
        base.setdefault("payload", 0)
        for count in grp["counts"]:
            for rep in range(reps):
                t = dict(base)
                t["count"] = count
                t["rep"] = rep
                tasks.append(t)
    return tasks


def _gen_compile(work_dir: Path, task: dict) -> bool:
    try:
        gen_one(work_dir, task)
        return compile_one(work_dir)
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workdir", type=Path,
                   default=REPO_ROOT / "build" / "calibration" / "sweep_notrace")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--compile-jobs", type=int, default=4)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()

    sweep_groups = json.loads(args.config.read_text())
    tasks = expand_sweep(sweep_groups, args.reps)
    print(f"No-trace sweep: {len(tasks)} runs across "
          f"{sum(len(g['counts']) for g in sweep_groups)} configurations.",
          file=sys.stderr)

    args.workdir.mkdir(parents=True, exist_ok=True)

    # Compile per unique config (rep=0 only).
    print("[1/2] generating + compiling kernels...", file=sys.stderr)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.compile_jobs) as pool:
        futs = {}
        for t in tasks:
            if t["rep"] != 0:
                continue
            tag = task_tag(t)
            wd = args.workdir / tag
            t["work_dir"] = wd
            futs[pool.submit(_gen_compile, wd, t)] = tag
        compile_failures = []
        for fut in as_completed(futs):
            tag = futs[fut]
            ok = fut.result()
            if not ok:
                compile_failures.append(tag)
        if compile_failures:
            print(f"  {len(compile_failures)} compile failures",
                  file=sys.stderr)
    print(f"[1/2] done in {time.time()-t0:.1f}s", file=sys.stderr)

    rep0_dirs = {task_tag(t): t["work_dir"] for t in tasks if t["rep"] == 0}
    for t in tasks:
        if t["rep"] != 0:
            t["work_dir"] = rep0_dirs[task_tag(t)]

    # Phase 2: HW runs via batch-stdin mode (single bridge-runner process).
    # We feed one run per line, parse JSON, record elapsed_us.
    print("[2/2] running on NPU via batch-stdin...", file=sys.stderr)
    t0 = time.time()
    measurements = []

    proc = subprocess.Popen(
        [str(BRIDGE_RUNNER), "--batch-stdin",
         "--kernel", "MLIR_AIE"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # avoid pipe-buffer deadlock
        text=True, bufsize=1)

    # Feed all run requests in one go; reading replies happens after.
    # To avoid stdout buffer deadlock, feed in chunks and drain.
    keys = ("kind", "count", "payload", "rep",
            "target_col", "target_row", "anchor_col", "anchor_row")
    for i, t in enumerate(tasks):
        wd = t["work_dir"]
        if not (wd / "final.xclbin").exists():
            measurements.append({k: t[k] for k in keys if k in t} | {
                "elapsed_us": None, "error": "compile_skipped",
            })
            continue
        line = (
            f"--xclbin {wd}/final.xclbin "
            f"--instr {wd}/insts.bin "
            f"--input {wd}/dummy0.bin --input {wd}/dummy1.bin "
            f"--input {wd}/dummy2.bin --input {wd}/dummy3.bin "
            f"--trace-out {wd}/trace.bin "
            f"--trace-size 65536\n"
        )
        proc.stdin.write(line)
        proc.stdin.flush()
        reply = proc.stdout.readline().strip()
        try:
            r = json.loads(reply)
        except json.JSONDecodeError:
            measurements.append({k: t[k] for k in keys if k in t} | {
                "elapsed_us": None, "error": f"json_parse: {reply[:80]}",
            })
            continue
        rec = {k: t[k] for k in keys if k in t}
        if r.get("ok"):
            rec["elapsed_us"] = r.get("elapsed_us")
            rec["elapsed_ms"] = r.get("elapsed_ms")
            rec["kernel_us"] = r.get("kernel_us")
        else:
            rec["elapsed_us"] = None
            rec["kernel_us"] = None
            rec["error"] = r.get("error", "unknown")
        measurements.append(rec)
        if (i + 1) % 100 == 0 or i + 1 == len(tasks):
            print(f"  [{i+1}/{len(tasks)}] elapsed {time.time()-t0:.1f}s",
                  file=sys.stderr)

    proc.stdin.close()
    proc.wait(timeout=10)
    print(f"[2/2] done in {time.time()-t0:.1f}s", file=sys.stderr)

    out_doc = {
        "schema_version": 1,
        "trace": "off",
        "device": "npu1",
        "compiler": "peano",
        "sweep_groups": sweep_groups,
        "reps": args.reps,
        "measurements": measurements,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2) + "\n")

    n_ok = sum(1 for m in measurements if m.get("elapsed_us") is not None)
    print(f"Wrote {args.out} ({n_ok}/{len(measurements)} successful)",
          file=sys.stderr)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
