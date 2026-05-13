#!/usr/bin/env python3
"""coldstart-residual.py -- measure HW vs EMU cold-start residual.

Two anchor strategies:

  --anchor=shim   shim DMA_MM2S_0_START_TASK -> shim DMA_MM2S_0_FINISHED_TASK
                  (single-tile, available on all default-trace kernels, but
                  contaminated by downstream backpressure on the same DMA)

  --anchor=memtile (default)
                  shim DMA_MM2S_0_START_TASK -> memtile DMA_S2MM_SEL0_FINISHED_BD
                  (cross-tile, requires memtile FINISHED_BD events, but
                  cleanly isolates DDR cold-start + stream-switch transit
                  to memtile SRAM; matches Phase C methodology)

For each kernel, prints HW gap, EMU gap, and residual = HW - EMU.
Aggregates with mean/median/stdev across kernels.
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PARSE_TRACE = REPO / "tools" / "parse-trace.py"


def parse_one(trace_bin: Path, mlir: Path, cache_dir: Path) -> list[dict]:
    """Decode trace_bin into events list, with on-disk caching."""
    key = f"{trace_bin.parent.name}__{trace_bin.name}.json"
    cached = cache_dir / key
    if cached.exists() and cached.stat().st_size > 0:
        return json.loads(cached.read_text()).get("events", [])
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["python3", str(PARSE_TRACE),
         "--trace-bin", str(trace_bin),
         "--xclbin-mlir", str(mlir),
         "--out-events", str(cached)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    return json.loads(cached.read_text()).get("events", [])


def first_soc(events: list[dict], row: int, name: str) -> int | None:
    matches = [e.get("soc", e.get("ts")) for e in events
               if e.get("row") == row and e.get("name") == name]
    return min(matches) if matches else None


def measure(events: list[dict], anchor: str) -> tuple[int | None, int | None, int | None]:
    """Return (start_soc, end_soc, gap)."""
    start = first_soc(events, 0, "DMA_MM2S_0_START_TASK")
    if anchor == "shim":
        end = first_soc(events, 0, "DMA_MM2S_0_FINISHED_TASK")
    elif anchor == "memtile":
        end = first_soc(events, 1, "DMA_S2MM_SEL0_FINISHED_BD")
    else:
        raise ValueError(f"unknown anchor: {anchor}")
    if start is None or end is None or end < start:
        return start, end, None
    return start, end, end - start


def resolve_mlir(name: str, variant: str) -> Path | None:
    """Find the post-lowering MLIR for a kernel; try several layouts."""
    base = Path(f"/home/triple/npu-work/mlir-aie/build/test/npu-xrt/{name}")
    candidates = [
        base / variant / "aie_arch.mlir.prj" / "input_with_addresses.mlir",
        base / "chess" / "aie_arch.mlir.prj" / "input_with_addresses.mlir",
        base / "aie2.mlir.prj" / "input_with_addresses.mlir",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw-dir", required=True, type=Path)
    ap.add_argument("--emu-dir", required=True, type=Path)
    ap.add_argument("--variant", default="chess")
    ap.add_argument("--anchor", choices=["shim", "memtile"], default="memtile")
    ap.add_argument("--kernels", nargs="*", default=None)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/tmp/claude-1000/coldstart-cache"))
    args = ap.parse_args()

    hw_kernels = {
        p.name.removesuffix(f".{args.variant}.hw")
        for p in args.hw_dir.iterdir()
        if p.name.endswith(f".{args.variant}.hw") and (p / "trace_raw.bin").exists()
    }
    emu_kernels = {
        p.name.removesuffix(f".{args.variant}.emu")
        for p in args.emu_dir.iterdir()
        if p.name.endswith(f".{args.variant}.emu") and (p / "trace_raw.bin").exists()
    }
    kernels = sorted(hw_kernels & emu_kernels)
    if args.kernels:
        kernels = [k for k in kernels if k in args.kernels]

    print(f"# Cold-start residual (anchor={args.anchor}) across {len(kernels)} kernels")
    print(f"# HW: {args.hw_dir}")
    print(f"# EMU: {args.emu_dir}")
    print(f"#")
    print(f"{'kernel':<48} {'HW_gap':>9} {'EMU_gap':>9} {'residual':>10}")

    residuals = []
    for k in kernels:
        mlir = resolve_mlir(k, args.variant)
        if not mlir:
            print(f"{k:<48}  -- no MLIR")
            continue
        hw = parse_one(args.hw_dir / f"{k}.{args.variant}.hw" / "trace_raw.bin",
                       mlir, args.cache_dir / "hw")
        emu = parse_one(args.emu_dir / f"{k}.{args.variant}.emu" / "trace_raw.bin",
                        mlir, args.cache_dir / "emu")
        _, _, hg = measure(hw, args.anchor)
        _, _, eg = measure(emu, args.anchor)
        hg_s = f"{hg:>9d}" if hg is not None else f"{'--':>9}"
        eg_s = f"{eg:>9d}" if eg is not None else f"{'--':>9}"
        if hg is not None and eg is not None:
            res = hg - eg
            residuals.append((k, res))
            print(f"{k:<48} {hg_s} {eg_s} {res:>+10d}")
        else:
            print(f"{k:<48} {hg_s} {eg_s} {'--':>10}")

    if residuals:
        rs = [r for _, r in residuals]
        sd = statistics.stdev(rs) if len(rs) > 1 else 0.0
        print(f"#")
        print(f"# n={len(rs)}  mean={statistics.mean(rs):+.0f}  "
              f"median={statistics.median(rs):+.0f}  stdev={sd:.0f}  "
              f"min={min(rs):+d}  max={max(rs):+d}  range={max(rs)-min(rs)}")


if __name__ == "__main__":
    main()
