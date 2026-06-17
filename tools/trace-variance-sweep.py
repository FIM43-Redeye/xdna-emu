#!/usr/bin/env python3
"""Repeat the full-event HW trace sweep of one kernel N times for #140.

Thin orchestration over tools/trace-sweep.py: runs the existing coverage sweep
HW-only (--no-emu) N times with run-indexed output, so trace_variance.py can
measure per-event run-to-run variance. No coverage logic here -- trace-sweep.py
owns tile discovery, 8-slot batching, insts.bin patching, and decode.

Contention is timing-neutral (2026-06-16 control pass), so --hw-jobs may pack
the repeats; default 1 for a clean serial baseline.
"""
import argparse
import subprocess
from pathlib import Path

DEFAULT_TILES = "0:0:shim,0:1:memtile,0:2:core,0:2:memmod"
ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "tools" / "trace-sweep.py"
DEFAULT_OUT = ROOT / "build/experiments/gap140/nondeterminism"


def build_sweep_cmd(test: str, tiles: str, out_dir: Path, jobs: int) -> list:
    return [
        "python3", str(SWEEP),
        "--test", test,
        "--tiles", tiles,
        "--no-emu",            # HW-only: EMU is the expensive thing under test
        "--jobs", str(jobs),
        "--out-dir", str(out_dir),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Repeat HW trace sweep N times (#140)")
    ap.add_argument("--test", default="add_one_using_dma")
    ap.add_argument("--tiles", default=DEFAULT_TILES)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--hw-jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    base = args.out or (DEFAULT_OUT / args.test)
    for r in range(args.repeat):
        run_dir = base / f"run_{r:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_sweep_cmd(args.test, args.tiles, run_dir, args.hw_jobs)
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"run {r}: sweep returned {res.returncode} (continuing)")
    print(f"completed {args.repeat} repeats under {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
