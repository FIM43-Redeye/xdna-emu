# tools/inference/hw_instrument.py
"""The real-HW adapter: drive the verified loop against Phoenix.

Conforms to the loop's instrument interface (ledger_entries + capture) and
reuses the HW-validated capture plumbing (build_active_plan / capture /
HwRunner). The planner emits batches in ABSOLUTE (decoder) col; the patcher
consumes RELATIVE col -- we subtract start_col on the way in. HW imports are
lazy so this module loads clean offline (tests monkeypatch `capture`/`HwRunner`).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

from config_extract.generator import generate_ledger
from inference.planner import Batch
# Imported at module scope so tests can monkeypatch these names directly.
from trace_capture import build_active_plan, capture, HwRunner, _discover_xclbin_insts


class HwInstrument:
    def __init__(self, test, dump, configured_events: List[str], *,
                 start_col: int, anchor_tile_abs: str, anchor_event: str,
                 traced_col: int, n_runs: int, out_root: str,
                 compiler: str = "chess"):
        self._test = test
        self._dump = dump
        self._configured = list(configured_events)
        self._start_col = start_col
        self._anchor_tile_abs = anchor_tile_abs
        self._anchor_event = anchor_event
        self._traced_col = traced_col
        self._n_runs = n_runs
        self._out_root = Path(out_root)
        self._compiler = compiler
        self._iter = 0

    def ledger_entries(self) -> List[dict]:
        # Generate over the full configured set so every orientable pair is
        # present regardless of which events a given batch traced.
        return generate_ledger(self._dump, self._configured,
                               start_col=self._start_col)["entries"]

    def _abs_to_rel(self, tile_key: str) -> str:
        col, row, pkt = tile_key.split("|")
        return f"{int(col) - self._start_col}|{row}|{pkt}"

    def capture(self, batch: Batch) -> List[str]:
        # Convert the planner's ABS-col batch tiles to REL col for the patcher.
        active: Dict[str, set] = {}
        for tile_abs, names in batch.tiles.items():
            active.setdefault(self._abs_to_rel(tile_abs), set()).update(names)
        anchor_tile_rel = self._abs_to_rel(self._anchor_tile_abs)
        # build_active_plan splits to <=8 slots and rides the anchor in slot 0
        # of the anchor tile in every batch.
        plan = build_active_plan(active, anchor=self._anchor_event,
                                 anchor_tile=anchor_tile_rel)
        xclbin, insts = _discover_xclbin_insts(self._test, self._compiler)
        run_dirs: List[str] = []
        base = self._out_root / f"capture_{self._iter:02d}"
        for i in range(self._n_runs):
            rd = base / f"run_{i:02d}"
            rd.mkdir(parents=True, exist_ok=True)
            runner = HwRunner(xclbin, stderr_log=rd / "hw.runner.log")
            try:
                capture(plan, runner, test=self._test, out_dir=rd,
                        traced_col=self._traced_col, instr=insts)
            finally:
                runner.close()
            run_dirs.append(str(rd))
        self._iter += 1
        return run_dirs
