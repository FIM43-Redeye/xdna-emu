#!/usr/bin/env python3
"""Experiment B driver: capture MM2S egress-FIFO-depth traces on HW.

Wires the mm2s_egress_depth.py generator through the standard 6-stage trace
pipeline (emit -> trace-prepare -> aiecc -> bridge-trace-runner -> parse-trace
-> measure), one build dir per variant under
build/experiments/mm2s-egress-depth/build_mm2s_egress_depth_<variant>/, the
path mm2s_egress_depth_measure.py expects.

Self-contained (does not import the session-tmp pp_recheck.py). HW runs use
`env -u XDNA_EMU XDNA_EMU_RUNTIME=release`; EMU cross-checks use XDNA_EMU=1.

Usage:
  b_egress_capture.py --variant stream_backpressure --compile-only   # routability smoke test
  b_egress_capture.py --variant fill_stall --reps 3                  # full HW capture
  b_egress_capture.py --all --reps 3                                 # every variant
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mm2s_egress_depth import VARIANTS  # noqa: E402
from mm2s_egress_depth_measure import measure  # noqa: E402

XDNA_EMU = Path("/home/triple/npu-work/xdna-emu")
GEN = XDNA_EMU / "tools/experiments/mm2s_egress_depth.py"
PREPARE = XDNA_EMU / "tools/trace-prepare.py"
PARSE = XDNA_EMU / "tools/parse-trace.py"
RUNNER = XDNA_EMU / "bridge-runner/build/bridge-trace-runner"
ROOT = XDNA_EMU / "build/experiments/mm2s-egress-depth"

# Compute-tile memory-module sweep events: the MM2S egress trio (stall onset ->
# starvation onset = occupancy) plus FINISHED_BD for reference; STREAM_BACKPRESSURE
# is the validity gate (a starvation while the stream itself is blocked is not an
# empty FIFO). Grounding core anchors mirror the producer_probe lineage.
MEMMOD_EVENTS = ("DMA_MM2S_0_FINISHED_BD,DMA_MM2S_0_STALLED_LOCK,"
                 "DMA_MM2S_0_MEMORY_STARVATION,DMA_MM2S_0_STREAM_BACKPRESSURE")
CORE_EVENTS = "MEMORY_STALL,LOCK_STALL,PORT_RUNNING_1,STREAM_STALL"


def run(cmd, cwd=None, env=None):
    print(f"+ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], file=sys.stderr)
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"FAILED (rc={r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


def gen(variant: str, src_dir: Path):
    src_dir.mkdir(parents=True, exist_ok=True)
    r = run([sys.executable, str(GEN), "--variant", variant])
    (src_dir / "aie.mlir").write_text(r.stdout)


def prepare(src_dir: Path, build_dir: Path):
    if build_dir.exists():
        shutil.rmtree(build_dir)
    run([sys.executable, str(PREPARE), str(src_dir),
         "-o", str(build_dir), "--trace-size", "16384",
         "--trace-mode", "event_time",
         "--memmod-sweep-events", MEMMOD_EVENTS,
         "--core-sweep-events", CORE_EVENTS])
    status = (build_dir / "prepare-status.txt").read_text().strip()
    if status != "OK":
        raise RuntimeError(f"prepare status not OK: {status}")


def build(build_dir: Path, chess: bool = True):
    flags = (["--xchesscc", "--xbridge"] if chess
             else ["--no-xchesscc", "--no-xbridge"])
    run(["aiecc.py", *flags, "--no-aiesim",
         "--aie-generate-xclbin", "--aie-generate-npu-insts",
         "--no-compile-host", "--alloc-scheme=basic-sequential",
         "--xclbin-name=aie.xclbin", "--npu-insts-name=insts.bin",
         "aie_traced.mlir"], cwd=build_dir)
    assert (build_dir / "aie.xclbin").exists() and (build_dir / "insts.bin").exists()


def run_hw(build_dir: Path, reps: int):
    env = os.environ.copy()
    env.pop("XDNA_EMU", None)
    env["XDNA_EMU_RUNTIME"] = "release"
    for rep in range(1, reps + 1):
        trace_out = build_dir / f"trace_r{rep}.bin"
        run([str(RUNNER), "--xclbin", str(build_dir / "aie.xclbin"),
             "--instr", str(build_dir / "insts.bin"),
             "--trace-out", str(trace_out), "--trace-size", "16384",
             "--output", str(build_dir / f"out_r{rep}.bin"), "-v"], env=env)
        assert trace_out.exists() and trace_out.stat().st_size > 0, \
            f"empty trace_out for {build_dir.name} rep {rep}"


def decode(build_dir: Path, reps: int):
    for rep in range(1, reps + 1):
        run([sys.executable, str(PARSE),
             "--trace-bin", str(build_dir / f"trace_r{rep}.bin"),
             "--xclbin-mlir", str(build_dir / "aie_traced.mlir"),
             "--out-perfetto", str(build_dir / f"perfetto_r{rep}.json"),
             "--trace-mode", "event_time"])


def go(variant: str, reps: int, compile_only: bool, chess: bool):
    src_dir = ROOT / f"src_{variant}"
    build_dir = ROOT / f"build_mm2s_egress_depth_{variant}"
    print(f"=== {variant}: gen+prepare+build ({'chess' if chess else 'peano'}) ===",
          file=sys.stderr)
    gen(variant, src_dir)
    prepare(src_dir, build_dir)
    build(build_dir, chess=chess)
    print(f"=== {variant}: COMPILED OK (routes) -> {build_dir} ===", file=sys.stderr)
    if compile_only:
        return None
    run_hw(build_dir, reps)
    decode(build_dir, reps)
    rows = [measure(build_dir, r) for r in range(1, reps + 1)]
    for r, m in enumerate(rows, 1):
        print(f"  {variant} rep{r}: windows={m['n_windows']} valid={m['n_valid']} "
              f"depth={m['depth_estimate']}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--all", action="store_true", help="run every variant")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--compile-only", action="store_true",
                    help="gen+prepare+build only (routability smoke test)")
    ap.add_argument("--peano", action="store_true", help="use Peano instead of Chess")
    args = ap.parse_args()
    if not args.variant and not args.all:
        ap.error("pass --variant <name> or --all")
    variants = VARIANTS if args.all else [args.variant]
    for v in variants:
        go(v, args.reps, args.compile_only, chess=not args.peano)
