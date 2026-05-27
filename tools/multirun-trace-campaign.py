#!/usr/bin/env python3
"""
multirun-trace-campaign.py -- Run the K-sweep trace pipeline N times against
HW to enable run-to-run noise characterization for the dispatch_overhead
calibration.

Per iteration we drive the same pipeline that produced the original
2026-05-25 dispatch_overhead numbers:
    bridge-trace-runner --xclbin <K>/chess/aie.xclbin
                        --instr   <K>/chess/insts.bin
                        --trace-out trace_raw.bin
                        --trace-size 1048576
    parse-trace.py      --trace-bin trace_raw.bin
                        --xclbin-mlir <K>/chess/aie_arch.mlir.prj/input_with_addresses.mlir
                        --trace-mode auto
                        --out-events events.json

The K-sweep xclbins are already trace-injected (aie.trace ops live in
the existing aie_arch.mlir and are baked into the compiled xclbin) so
no recompilation is needed.

Randomized (run_idx, K) ordering decorrelates K from run-order effects
(thermal, FW-state drift). K=16 excluded by default -- wedges per the
2026-05-25 dispatch-overhead finding.

Outputs:
  build/experiments/dispatch-overhead-multirun/<session>/
    manifest.json          -- session metadata (FW + driver + git versions)
    schedule.json          -- the (run_idx, K) ordering executed
    run{NNN}/
      k{K}/
        trace_raw.bin      -- raw trace bytes from bridge-trace-runner
        events.json        -- decoded events list (consumed by aggregator)
        meta.json          -- per-run metadata (timestamps, returncodes)
        runner.log         -- bridge-trace-runner stdout/stderr
        parser.log         -- parse-trace.py stdout/stderr

Usage:
  ./tools/multirun-trace-campaign.py --n-runs 50
  ./tools/multirun-trace-campaign.py --n-runs 50 --ks 1,2,4,8
  ./tools/multirun-trace-campaign.py --n-runs 5 --dry-run  # just schedule
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER_REPO = Path("/home/triple/npu-work/xdna-driver")
DEFAULT_OUT_ROOT = REPO / "build" / "experiments" / "dispatch-overhead-multirun"
MLIR_AIE_ROOT = Path("/home/triple/npu-work/mlir-aie")
RUNNER = REPO / "bridge-runner" / "build" / "bridge-trace-runner"
PARSE_TRACE = REPO / "tools" / "parse-trace.py"
TRACE_SIZE = 1048576

# parse-trace.py needs mlir-aie's Python install on PYTHONPATH for slot-name
# lookup from the lowered MLIR.
MLIR_AIE_PY = str(MLIR_AIE_ROOT / "install" / "python")


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def git_head(repo: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception as e:
        return f"error:{e}"


def fw_state_from_xrt_smi() -> dict:
    """One-shot capture of FW + driver versions via xrt-smi.

    Safe BEFORE the campaign (no HW dispatch active). xrt-smi segfaults
    if run concurrently with HW work on this devbox, so we never call it
    again until the session ends.
    """
    out = {"fw_version": "", "amdxdna_version": "", "xrt_version": ""}
    env = dict(os.environ)
    env.pop("XDNA_EMU", None)
    env.pop("XDNA_EMU_RUNTIME", None)
    try:
        r = subprocess.run(
            ["/opt/xilinx/xrt/bin/xrt-smi", "examine"],
            capture_output=True, text=True, timeout=15, env=env,
        )
        if r.returncode != 0:
            return out
        for line in r.stdout.splitlines():
            s = line.strip()
            if s.startswith("NPU Firmware Version"):
                out["fw_version"] = s.split(":", 1)[1].strip()
            elif s.startswith("amdxdna Version"):
                out["amdxdna_version"] = s.split(":", 1)[1].strip()
            elif s.startswith("Version") and "xrt" not in out["xrt_version"].lower():
                out["xrt_version"] = s.split(":", 1)[1].strip()
    except Exception:
        pass
    return out


def kbuild_paths(k: int) -> dict:
    """Compute artifact paths for a given K. Raises if any are missing."""
    base = MLIR_AIE_ROOT / "build" / "test" / "npu-xrt" / "_diag_shim_chain_sweep" / f"k{k}" / "chess"
    paths = {
        "xclbin": base / "aie.xclbin",
        "instr":  base / "insts.bin",
        "mlir":   base / "aie_arch.mlir.prj" / "input_with_addresses.mlir",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"K={k}: missing {name} at {p}")
    return paths


def make_schedule(ks: list[int], n_runs: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    items = [(r, k) for r in range(1, n_runs + 1) for k in ks]
    rng.shuffle(items)
    return items


def env_no_emu() -> dict:
    """Env with XDNA_EMU* removed so XRT routes to real HW."""
    env = dict(os.environ)
    env.pop("XDNA_EMU", None)
    env.pop("XDNA_EMU_RUNTIME", None)
    return env


def run_one(run_idx: int, k: int, paths: dict, out_root: Path,
            verbose: bool) -> dict:
    """One iteration: bridge-trace-runner + parse-trace.py."""
    run_dir = out_root / f"run{run_idx:03d}" / f"k{k}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_bin = run_dir / "trace_raw.bin"
    events_json = run_dir / "events.json"
    runner_log = run_dir / "runner.log"
    parser_log = run_dir / "parser.log"

    t_start = time.monotonic()
    t_start_wall = datetime.now(timezone.utc).isoformat()

    # 1) bridge-trace-runner
    runner_cmd = [
        str(RUNNER),
        "--xclbin", str(paths["xclbin"]),
        "--instr", str(paths["instr"]),
        "--trace-out", str(trace_bin),
        "--trace-size", str(TRACE_SIZE),
    ]
    t_run_start = time.monotonic()
    with open(runner_log, "w") as logf:
        rr = subprocess.run(runner_cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=env_no_emu())
    run_us = (time.monotonic() - t_run_start) * 1e6

    # 2) parse-trace.py (skip if runner failed)
    pr_rc = None
    parse_us = 0.0
    if rr.returncode == 0:
        env_p = env_no_emu()
        env_p["PYTHONPATH"] = MLIR_AIE_PY
        parser_cmd = [
            sys.executable, str(PARSE_TRACE),
            "--trace-bin", str(trace_bin),
            "--xclbin-mlir", str(paths["mlir"]),
            "--trace-mode", "auto",
            "--out-events", str(events_json),
        ]
        t_parse_start = time.monotonic()
        with open(parser_log, "w") as logf:
            pr = subprocess.run(parser_cmd, stdout=logf, stderr=subprocess.STDOUT,
                                env=env_p)
        pr_rc = pr.returncode
        parse_us = (time.monotonic() - t_parse_start) * 1e6

    t_end = time.monotonic()
    t_end_wall = datetime.now(timezone.utc).isoformat()
    elapsed = t_end - t_start

    meta = {
        "run_idx": run_idx,
        "k": k,
        "start_utc": t_start_wall,
        "end_utc": t_end_wall,
        "elapsed_s": round(elapsed, 3),
        "runner_rc": rr.returncode,
        "parser_rc": pr_rc,
        "runner_us": round(run_us, 1),
        "parser_us": round(parse_us, 1),
        "ok": rr.returncode == 0 and pr_rc == 0,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        status = "OK " if meta["ok"] else "FAIL"
        print(f"  [{status}] run{run_idx:03d}/k{k}: {elapsed*1000:.0f}ms "
              f"(run={run_us/1000:.0f}ms parse={parse_us/1000:.0f}ms "
              f"rc=r{rr.returncode}/p{pr_rc})")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--n-runs", type=int, default=50,
                    help="number of full (K) sweeps per K value (default 50)")
    ap.add_argument("--ks", default="1,2,4,8",
                    help="comma-separated K values (default 1,2,4,8; "
                         "k16 wedges)")
    ap.add_argument("--seed", type=int, default=42,
                    help="schedule shuffle seed (default 42)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help=f"output root (default {DEFAULT_OUT_ROOT})")
    ap.add_argument("--session", default=None,
                    help="session subdir name (default: timestamp)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print schedule and manifest; don't execute")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress per-iteration progress lines")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    if 16 in ks:
        print("WARN: K=16 wedges per 2026-05-25 dispatch-overhead finding "
              "-- proceed anyway; this run will likely hang at K=16",
              file=sys.stderr)

    # Verify prerequisites before launching.
    if not RUNNER.is_file() or not os.access(RUNNER, os.X_OK):
        print(f"error: bridge-trace-runner missing or not executable at {RUNNER}",
              file=sys.stderr)
        return 1
    if not PARSE_TRACE.is_file():
        print(f"error: parse-trace.py missing at {PARSE_TRACE}",
              file=sys.stderr)
        return 1

    # Resolve per-K artifacts once; any error aborts before the campaign.
    paths_by_k = {}
    for k in ks:
        paths_by_k[k] = kbuild_paths(k)

    session = args.session or utc_now_str()
    out_root = args.out_root / session
    out_root.mkdir(parents=True, exist_ok=True)

    schedule = make_schedule(ks, args.n_runs, args.seed)

    manifest = {
        "session": session,
        "session_start_utc": datetime.now(timezone.utc).isoformat(),
        "ks": ks,
        "n_runs": args.n_runs,
        "n_iterations": len(schedule),
        "seed": args.seed,
        "xdna_emu_git": git_head(REPO),
        "xdna_driver_git": git_head(DRIVER_REPO),
        "fw_state": fw_state_from_xrt_smi(),
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "runner_path": str(RUNNER),
        "parse_trace_path": str(PARSE_TRACE),
        "trace_size_bytes": TRACE_SIZE,
        "out_root": str(out_root),
        "kbuild_paths": {str(k): {n: str(p) for n, p in v.items()}
                         for k, v in paths_by_k.items()},
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_root / "schedule.json").write_text(json.dumps(
        [{"order": i, "run_idx": r, "k": k} for i, (r, k) in enumerate(schedule)],
        indent=2,
    ))

    print(f"== multirun-trace-campaign ==")
    print(f"  session     : {session}")
    print(f"  ks          : {ks}")
    print(f"  n_runs      : {args.n_runs}")
    print(f"  iterations  : {len(schedule)}")
    print(f"  out_root    : {out_root}")
    print(f"  xdna-emu    : {manifest['xdna_emu_git']}")
    print(f"  xdna-driver : {manifest['xdna_driver_git']}")
    fw = manifest["fw_state"]
    if fw.get("fw_version"):
        print(f"  fw_version  : {fw['fw_version']}")
    if fw.get("amdxdna_version"):
        print(f"  amdxdna     : {fw['amdxdna_version']}")
    if args.dry_run:
        print("DRY-RUN: schedule and manifest written; not executing")
        return 0

    t0 = time.monotonic()
    n_ok = 0
    n_fail = 0
    for i, (run_idx, k) in enumerate(schedule, 1):
        if not args.quiet:
            pct = 100 * (i - 1) / len(schedule)
            print(f"[{i}/{len(schedule)}] {pct:5.1f}% run={run_idx} k={k}",
                  flush=True)
        meta = run_one(run_idx, k, paths_by_k[k], out_root,
                       verbose=not args.quiet)
        if meta["ok"]:
            n_ok += 1
        else:
            n_fail += 1
        if n_fail >= 5 and n_fail > n_ok:
            print(f"ABORT: {n_fail} failures, only {n_ok} OK; "
                  f"investigate before continuing", file=sys.stderr)
            break
    elapsed = time.monotonic() - t0

    manifest_final = dict(manifest)
    manifest_final.update({
        "session_end_utc": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_s": round(elapsed, 3),
        "n_ok": n_ok,
        "n_fail": n_fail,
    })
    (out_root / "manifest.json").write_text(json.dumps(manifest_final, indent=2))

    print(f"\n== done ==")
    print(f"  ok          : {n_ok}/{len(schedule)}")
    print(f"  failed      : {n_fail}/{len(schedule)}")
    print(f"  elapsed     : {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  manifest    : {out_root / 'manifest.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
