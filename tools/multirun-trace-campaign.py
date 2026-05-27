#!/usr/bin/env python3
"""
multirun-trace-campaign.py -- Run the K-sweep trace pipeline N times against
HW, with randomized (run_idx, K) ordering, to enable run-to-run noise
characterization for the dispatch_overhead calibration.

Wraps `tools/trace-sweep.py` (HW-only, shim-tile-only) in a driver loop.
Each iteration:
  1. Picks the next (run_idx, K) from a shuffled schedule.
  2. Calls trace-sweep.py against the pre-built k{K}/chess xclbin.
  3. Writes outputs into <session>/run{NNN}/k{K}/.
  4. Records per-iteration metadata (start/end timestamps, returncode).

At session start we record FW/driver/xdna-emu git state to a manifest
JSON. We don't probe FW state mid-run (xrt-smi segfaults concurrently
with HW dispatch on this devbox).

Skips K=16 by default -- wedges per the 2026-05-25 dispatch-overhead
finding, separate state-inspection effort in Phase 2b.

Outputs:
  build/experiments/dispatch-overhead-multirun/<session>/
    manifest.json          -- session metadata
    schedule.json          -- the (run_idx, K) ordering we executed
    run{NNN}/
      k{K}/
        _diag_shim_chain_sweep/
          k{K}.chess.shim_c0r0.json
          k{K}.chess.shim_c0r0.merged.json
          k{K}.chess.multitile.work/
            b*.hw.events.json    -- raw per-batch timestamped events
            b*.trace_hw.bin
            ... (trace-sweep working files)
      meta.json              -- per-run metadata (timestamps, returncodes)

Usage:
  ./tools/multirun-trace-campaign.py --n-runs 50
  ./tools/multirun-trace-campaign.py --n-runs 50 --ks 1,2,4,8
  ./tools/multirun-trace-campaign.py --n-runs 5 --dry-run  # just print schedule
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRACE_SWEEP = REPO / "tools" / "trace-sweep.py"
DRIVER_REPO = Path("/home/triple/npu-work/xdna-driver")
DEFAULT_OUT_ROOT = REPO / "build" / "experiments" / "dispatch-overhead-multirun"


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
                # First "Version" line under "XRT" block
                out["xrt_version"] = s.split(":", 1)[1].strip()
    except Exception:
        pass
    return out


def make_schedule(ks: list[int], n_runs: int, seed: int) -> list[tuple[int, int]]:
    """Build (run_idx, K) schedule, randomized to decorrelate K from run order."""
    rng = random.Random(seed)
    items = [(r, k) for r in range(1, n_runs + 1) for k in ks]
    rng.shuffle(items)
    return items


def run_one(run_idx: int, k: int, out_root: Path, verbose: bool) -> dict:
    """Execute one trace-sweep.py invocation; return per-run metadata."""
    run_dir = out_root / f"run{run_idx:03d}" / f"k{k}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "trace-sweep.log"

    cmd = [
        sys.executable, str(TRACE_SWEEP),
        "--test", f"_diag_shim_chain_sweep/k{k}",
        "--compiler", "chess",
        "--no-emu",
        "--tiles", "0:0:shim",
        "--out-dir", str(run_dir),
    ]

    t_start = time.monotonic()
    t_start_wall = datetime.now(timezone.utc).isoformat()
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    t_end = time.monotonic()
    t_end_wall = datetime.now(timezone.utc).isoformat()
    elapsed = t_end - t_start

    meta = {
        "run_idx": run_idx,
        "k": k,
        "start_utc": t_start_wall,
        "end_utc": t_end_wall,
        "elapsed_s": round(elapsed, 3),
        "returncode": r.returncode,
        "ok": r.returncode == 0,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        status = "OK " if meta["ok"] else "FAIL"
        print(f"  [{status}] run{run_idx:03d}/k{k}: {elapsed:.2f}s "
              f"(rc={r.returncode})")
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
        "trace_sweep_path": str(TRACE_SWEEP),
        "out_root": str(out_root),
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
    results = []
    n_ok = 0
    n_fail = 0
    for i, (run_idx, k) in enumerate(schedule, 1):
        if not args.quiet:
            pct = 100 * (i - 1) / len(schedule)
            print(f"[{i}/{len(schedule)}] {pct:5.1f}% run={run_idx} k={k}", flush=True)
        meta = run_one(run_idx, k, out_root, verbose=not args.quiet)
        results.append(meta)
        if meta["ok"]:
            n_ok += 1
        else:
            n_fail += 1
        # Bail out aggressively if many failures in a row -- something's wrong.
        if n_fail >= 5 and n_fail > n_ok:
            print(f"ABORT: {n_fail} failures, only {n_ok} OK; investigate before continuing",
                  file=sys.stderr)
            break
    elapsed = time.monotonic() - t0

    # Final manifest update with session totals.
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
