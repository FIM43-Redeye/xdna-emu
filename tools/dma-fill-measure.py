#!/usr/bin/env python3
"""dma-fill-measure.py -- Extract DMA pipeline-fill metrics from bridge-test traces.

Walks a bridge-test results directory (build/bridge-test-results/<date>/)
and, for every (test, compiler) pair that has BOTH `<test>.<compiler>.hw/
trace_raw.bin` and `<test>.<compiler>.emu/trace_raw.bin`, decodes each side
into per-tile event timelines via `tools/parse-trace.py` and emits a CSV row
per (test, compiler, side, tile) with pipeline-fill metrics.

The numbers feed task #359 (#355a): compare HW vs EMU pipeline-fill across
the corpus to drive DMA cycle-cost calibration.

REQUIRES event_time (mode 0) traces.  The default bridge-test sweep produces
event_pc (mode 1) traces -- in mode 1 the ``ts`` field carries a PC value,
not a cycle count, and pipeline-fill arithmetic on PCs is meaningless.  Re-
run the sweep with ``--trace-mode event_time`` (or its equivalent) before
running this tool.

Metrics per tile (best-effort -- absent events leave the column empty):

  total_cycles            max(ts) - min(ts) across the tile's events
  t_first_dma_start       earliest DMA_*_START_TASK event ts
  t_first_dma_finished    earliest DMA_*_FINISHED_TASK event ts
  dma_roundtrip           t_first_dma_finished - t_first_dma_start
                          (single-BD DMA latency from BD launch to
                          completion -- the value calibration targets)
  t_first_acq_req         earliest INSTR_LOCK_ACQUIRE_REQ on core
  t_first_lock_stall      earliest LOCK_STALL on core
  acq_to_finish           t_first_dma_finished - t_first_acq_req
                          (kernel-visible DMA fill: time core waits
                          after issuing the first acquire)

Two passes over the events:
  - Per-tile: events grouped by (col, row, pkt_type), pick first by name.
  - Co-located: a core (pkt_type=0) at (col, row) and the matching
    mem-module (pkt_type=1) at (col, row) share a tile; metrics that
    cross modules pair them up.

Usage:
  dma-fill-measure.py \\
      --results-dir build/bridge-test-results/latest \\
      --build-base /home/triple/npu-work/mlir-aie/build/test/npu-xrt \\
      --out build/dma-fill.csv

  dma-fill-measure.py --results-dir <dir> --tests add_one_using_dma,passthrough
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
PARSE_TRACE = REPO / "tools" / "parse-trace.py"
DEFAULT_BUILD_BASE = Path("/home/triple/npu-work/mlir-aie/build/test/npu-xrt")
DEFAULT_PYTHONPATH = "/home/triple/npu-work/mlir-aie/install/python"

# --- Event name groups used in metric extraction ---

DMA_START_PREFIX = "DMA_"
DMA_START_NAMES = (
    "DMA_S2MM_0_START_TASK",
    "DMA_S2MM_1_START_TASK",
    "DMA_MM2S_0_START_TASK",
    "DMA_MM2S_1_START_TASK",
)
DMA_FINISHED_NAMES = (
    "DMA_S2MM_0_FINISHED_TASK",
    "DMA_S2MM_1_FINISHED_TASK",
    "DMA_MM2S_0_FINISHED_TASK",
    "DMA_MM2S_1_FINISHED_TASK",
)
ACQ_REQ_NAME = "INSTR_LOCK_ACQUIRE_REQ"
LOCK_STALL_NAME = "LOCK_STALL"

PKT_CORE = 0
PKT_MEMMOD = 1
PKT_SHIM = 2
PKT_MEMTILE = 3


# ----------------------------------------------------------------------------
# Pure metric extraction (testable in isolation)
# ----------------------------------------------------------------------------

@dataclass
class TileMetrics:
    col: int
    row: int
    pkt_type: int
    event_count: int = 0
    total_cycles: Optional[int] = None
    t_first_dma_start: Optional[int] = None
    t_first_dma_finished: Optional[int] = None
    t_first_acq_req: Optional[int] = None
    t_first_lock_stall: Optional[int] = None

    @property
    def dma_roundtrip(self) -> Optional[int]:
        if self.t_first_dma_start is None or self.t_first_dma_finished is None:
            return None
        # Negative roundtrip means the trace's first FINISHED predates its
        # first START -- typically a stale BD completing from a previous
        # iteration. Mark as None so calibration ignores it instead of
        # treating it as a real measurement.
        delta = self.t_first_dma_finished - self.t_first_dma_start
        return delta if delta >= 0 else None


@dataclass
class PairMetrics:
    """A core (pkt_type=0) plus its co-located mem module (pkt_type=1).

    Only populated when both pkt_types are present at (col, row); otherwise
    the cross-module fields stay None.
    """
    col: int
    row: int
    core: Optional[TileMetrics] = None
    mem: Optional[TileMetrics] = None

    @property
    def acq_to_finish(self) -> Optional[int]:
        if self.core is None or self.mem is None:
            return None
        if self.core.t_first_acq_req is None or self.mem.t_first_dma_finished is None:
            return None
        return self.mem.t_first_dma_finished - self.core.t_first_acq_req


def extract_tile_metrics(events: Iterable[dict]) -> dict[tuple[int, int, int], TileMetrics]:
    """Collapse an events-JSON event list into per-(col, row, pkt_type) metrics.

    Each input event must have ``col``, ``row``, ``pkt_type``, ``name``, ``ts``
    fields (the schema produced by ``parse-trace.py --out-events``).
    """
    by_tile: dict[tuple[int, int, int], TileMetrics] = {}
    for e in events:
        col = int(e["col"])
        row = int(e["row"])
        pkt = int(e["pkt_type"])
        name = e.get("name", "") or ""
        ts = int(e["ts"])

        key = (col, row, pkt)
        m = by_tile.get(key)
        if m is None:
            m = TileMetrics(col=col, row=row, pkt_type=pkt)
            by_tile[key] = m
        m.event_count += 1

        # Track total span first, regardless of name.
        if m.total_cycles is None:
            m.total_cycles = 0
            m._min_ts = ts  # type: ignore[attr-defined]
            m._max_ts = ts  # type: ignore[attr-defined]
        else:
            if ts < m._min_ts:  # type: ignore[attr-defined]
                m._min_ts = ts  # type: ignore[attr-defined]
            if ts > m._max_ts:  # type: ignore[attr-defined]
                m._max_ts = ts  # type: ignore[attr-defined]

        if name in DMA_START_NAMES:
            if m.t_first_dma_start is None or ts < m.t_first_dma_start:
                m.t_first_dma_start = ts
        if name in DMA_FINISHED_NAMES:
            if m.t_first_dma_finished is None or ts < m.t_first_dma_finished:
                m.t_first_dma_finished = ts
        if name == ACQ_REQ_NAME:
            if m.t_first_acq_req is None or ts < m.t_first_acq_req:
                m.t_first_acq_req = ts
        if name == LOCK_STALL_NAME:
            if m.t_first_lock_stall is None or ts < m.t_first_lock_stall:
                m.t_first_lock_stall = ts

    for m in by_tile.values():
        if m.total_cycles is not None:
            m.total_cycles = m._max_ts - m._min_ts  # type: ignore[attr-defined]
            del m._min_ts  # type: ignore[attr-defined]
            del m._max_ts  # type: ignore[attr-defined]
    return by_tile


def pair_core_with_memmod(
    by_tile: dict[tuple[int, int, int], TileMetrics],
) -> dict[tuple[int, int], PairMetrics]:
    """Group co-located core (pkt=0) and mem module (pkt=1) at each (col, row).

    Cross-module metrics (e.g. acq_to_finish) require both halves; tiles with
    only one side present still appear with the missing half left None.
    """
    pairs: dict[tuple[int, int], PairMetrics] = {}
    for (col, row, pkt), m in by_tile.items():
        if pkt not in (PKT_CORE, PKT_MEMMOD):
            continue
        p = pairs.get((col, row))
        if p is None:
            p = PairMetrics(col=col, row=row)
            pairs[(col, row)] = p
        if pkt == PKT_CORE:
            p.core = m
        elif pkt == PKT_MEMMOD:
            p.mem = m
    return pairs


# ----------------------------------------------------------------------------
# I/O: result-dir walk + parse-trace.py invocation
# ----------------------------------------------------------------------------

# Match `<test>.<compiler>.<side>` directory names.  side is "hw" or "emu";
# `<test>` may contain dots, so anchor the compiler+side suffix.
_DIR_RE = re.compile(r"^(?P<name>.+?)\.(?P<compiler>chess|peano)\.(?P<side>hw|emu)$")


@dataclass
class TestPair:
    """One (test, compiler) entry with whatever sides are present on disk."""
    name: str
    compiler: str
    hw_bin: Optional[Path] = None
    emu_bin: Optional[Path] = None
    mlir: Optional[Path] = None


def discover_pairs(results_dir: Path, build_base: Path) -> list[TestPair]:
    """Scan the results directory for HW/EMU trace pairs.

    Returns a list sorted by (name, compiler) for deterministic CSV output.
    """
    pairs: dict[tuple[str, str], TestPair] = {}
    for entry in results_dir.iterdir():
        if not entry.is_dir():
            continue
        m = _DIR_RE.match(entry.name)
        if not m:
            continue
        name = m.group("name")
        compiler = m.group("compiler")
        side = m.group("side")
        bin_path = entry / "trace_raw.bin"
        if not bin_path.exists():
            continue

        key = (name, compiler)
        p = pairs.get(key)
        if p is None:
            p = TestPair(name=name, compiler=compiler)
            pairs[key] = p
        if side == "hw":
            p.hw_bin = bin_path
        else:
            p.emu_bin = bin_path

    # MLIR discovery uses the standard mlir-aie build layout.
    for p in pairs.values():
        candidate = build_base / p.name / p.compiler / "aie_arch.mlir.prj" / "input_with_addresses.mlir"
        if candidate.exists():
            p.mlir = candidate

    return sorted(pairs.values(), key=lambda p: (p.name, p.compiler))


def decode_events(trace_bin: Path, mlir: Path, out_events: Path,
                  trace_mode: str = "event_time") -> bool:
    """Invoke parse-trace.py to convert a trace bin into events JSON.

    ``trace_mode`` is forwarded to parse-trace.py.  Pipeline-fill metrics
    only make sense for mode 0 (event_time); other modes are accepted for
    diagnostics but the resulting CSV will be misleading.

    Returns True on success, False on parser failure (caller should skip
    this side rather than abort the whole sweep).
    """
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", DEFAULT_PYTHONPATH)
    cmd = [
        sys.executable, str(PARSE_TRACE),
        "--trace-bin", str(trace_bin),
        "--xclbin-mlir", str(mlir),
        "--trace-mode", trace_mode,
        "--out-events", str(out_events),
    ]
    try:
        subprocess.run(cmd, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return out_events.exists()


def load_events(events_json: Path) -> list[dict]:
    with events_json.open() as f:
        d = json.load(f)
    return d.get("events", [])


# ----------------------------------------------------------------------------
# CSV emission
# ----------------------------------------------------------------------------

CSV_FIELDS = [
    "test", "compiler", "side", "col", "row", "pkt_type",
    "event_count", "total_cycles",
    "t_first_dma_start", "t_first_dma_finished", "dma_roundtrip",
    "t_first_acq_req", "t_first_lock_stall", "acq_to_finish",
]


def metrics_to_rows(
    test: str, compiler: str, side: str,
    by_tile: dict[tuple[int, int, int], TileMetrics],
    pairs: dict[tuple[int, int], PairMetrics],
) -> list[dict]:
    """Flatten per-tile and per-pair metrics into CSV rows.

    One row per (col, row, pkt_type).  ``acq_to_finish`` is a
    cross-module metric and gets attached to the core row (pkt=0) of
    each (col, row) pair.
    """
    rows = []
    for (col, row, pkt), m in sorted(by_tile.items()):
        row_d = {
            "test": test, "compiler": compiler, "side": side,
            "col": col, "row": row, "pkt_type": pkt,
            "event_count": m.event_count,
            "total_cycles": m.total_cycles,
            "t_first_dma_start": m.t_first_dma_start,
            "t_first_dma_finished": m.t_first_dma_finished,
            "dma_roundtrip": m.dma_roundtrip,
            "t_first_acq_req": m.t_first_acq_req,
            "t_first_lock_stall": m.t_first_lock_stall,
            "acq_to_finish": "",
        }
        if pkt == PKT_CORE:
            pair = pairs.get((col, row))
            if pair is not None:
                row_d["acq_to_finish"] = pair.acq_to_finish
        rows.append(row_d)
    return rows


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--results-dir", required=True, type=Path,
                   help="bridge-test results directory (e.g. build/"
                        "bridge-test-results/latest)")
    p.add_argument("--build-base", type=Path, default=DEFAULT_BUILD_BASE,
                   help=f"mlir-aie build base for input_with_addresses.mlir lookup "
                        f"(default: {DEFAULT_BUILD_BASE})")
    p.add_argument("--out", type=Path, required=True,
                   help="output CSV path")
    p.add_argument("--tests", default="",
                   help="comma-separated test name filter (default: all)")
    p.add_argument("--compilers", default="chess,peano",
                   help="comma-separated compiler list (default: chess,peano)")
    p.add_argument("--require-both-sides", action="store_true",
                   help="skip (test, compiler) pairs that lack either HW or EMU "
                        "trace; default is to emit whichever side is present")
    p.add_argument("--trace-mode", default="event_time",
                   choices=("event_time", "event_pc", "inst_exec", "auto"),
                   help="trace mode forwarded to parse-trace.py. Pipeline-fill "
                        "metrics only make sense for event_time (mode 0); other "
                        "modes are accepted for diagnostics. (default: event_time)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    results_dir: Path = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"error: results-dir not found: {results_dir}", file=sys.stderr)
        return 2

    test_filter = {t for t in args.tests.split(",") if t}
    compiler_filter = {c for c in args.compilers.split(",") if c}

    pairs = discover_pairs(results_dir, args.build_base)
    if test_filter:
        pairs = [p for p in pairs if p.name in test_filter]
    pairs = [p for p in pairs if p.compiler in compiler_filter]

    if not pairs:
        print(f"warning: no (test, compiler) pairs matched in {results_dir}",
              file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for pair in pairs:
            sides_to_run = []
            if pair.hw_bin:
                sides_to_run.append(("hw", pair.hw_bin))
            if pair.emu_bin:
                sides_to_run.append(("emu", pair.emu_bin))

            if args.require_both_sides and (
                pair.hw_bin is None or pair.emu_bin is None
            ):
                if args.verbose:
                    print(f"[skip] {pair.name}.{pair.compiler}: missing "
                          f"{'HW' if pair.hw_bin is None else 'EMU'} side",
                          file=sys.stderr)
                continue

            if pair.mlir is None:
                if args.verbose:
                    print(f"[skip] {pair.name}.{pair.compiler}: no "
                          f"input_with_addresses.mlir under {args.build_base}",
                          file=sys.stderr)
                continue

            with tempfile.TemporaryDirectory(prefix="dma-fill-") as tmp:
                tmp_path = Path(tmp)
                for side, bin_path in sides_to_run:
                    out_events = tmp_path / f"{side}.events.json"
                    ok = decode_events(bin_path, pair.mlir, out_events,
                                       trace_mode=args.trace_mode)
                    if not ok:
                        if args.verbose:
                            print(f"[warn] {pair.name}.{pair.compiler}/{side}: "
                                  f"parse-trace failed", file=sys.stderr)
                        continue
                    events = load_events(out_events)
                    by_tile = extract_tile_metrics(events)
                    pair_map = pair_core_with_memmod(by_tile)
                    for row in metrics_to_rows(
                        pair.name, pair.compiler, side, by_tile, pair_map
                    ):
                        writer.writerow(row)
                    if args.verbose:
                        print(f"[ok] {pair.name}.{pair.compiler}/{side}: "
                              f"{len(events)} events, {len(by_tile)} tiles",
                              file=sys.stderr)

    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
