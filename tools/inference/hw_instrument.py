# tools/inference/hw_instrument.py
"""The real-HW adapter: drive the verified loop against Phoenix.

Conforms to the loop's instrument interface (ledger_entries + capture) and
reuses the HW-validated capture plumbing (build_active_plan / capture /
HwRunner). The planner emits batches in ABSOLUTE (decoder) col space.
configure_batch is the single rel<->abs reconcile point: it receives ABSOLUTE
col tile keys and emits RELATIVE col in patch_spec (abs - start_col). The
probe_slot_capacity gate reads insts.bin (a RELATIVE col artifact), so we
compute col_rel = abs - start_col only for the probe; the plan and anchor stay
ABSOLUTE. HW-boundary names are imported at module scope so tests can
monkeypatch them; the module still imports clean offline.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
from pathlib import Path
from typing import Dict, List

from config_extract.generator import generate_ledger
from inference.planner import Batch
# Imported at module scope so tests can monkeypatch these names directly.
from trace_capture import build_active_plan, capture, HwRunner, _discover_xclbin_insts
from trace_capture import PKT_TO_TILE_TYPE

# Bind probe_slot_capacity from trace-patch-events.py at module scope so tests
# can monkeypatch this name (inference.hw_instrument.probe_slot_capacity) to
# stub out filesystem reads in offline test runs.
_spec = _ilu.spec_from_file_location(
    "trace_patch_events", _P(__file__).resolve().parent.parent / "trace-patch-events.py")
_tpe = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tpe)
probe_slot_capacity = _tpe.probe_slot_capacity


class HwInstrument:
    def __init__(self, test, dump, configured_events: List[str], *,
                 start_col: int, anchor_tile_abs: str, anchor_event: str,
                 n_runs: int, out_root: str,
                 compiler: str = "chess"):
        self._test = test
        self._dump = dump
        self._configured = list(configured_events)
        self._start_col = start_col
        self._anchor_tile_abs = anchor_tile_abs
        self._anchor_event = anchor_event
        self._n_runs = n_runs
        self._out_root = Path(out_root)
        self._compiler = compiler
        self._iter = 0

    def ledger_entries(self) -> List[dict]:
        # Generate over the full configured set so every orientable pair is
        # present regardless of which events a given batch traced.
        return generate_ledger(self._dump, self._configured,
                               start_col=self._start_col)["entries"]

    def capture(self, batch: Batch) -> List[str]:
        xclbin, insts = _discover_xclbin_insts(self._test, self._compiler)
        insts_bytes = Path(insts).read_bytes()

        # Drop tiles the xclbin compiled WITHOUT trace. probe_slot_capacity reads
        # insts.bin, which is in RELATIVE col space -> probe with abs - start_col.
        # The plan/anchor stay in ABSOLUTE col (configure_batch does the abs->rel
        # for the patcher). Never feed absolute col to probe_slot_capacity.
        traceable_abs: Dict[str, set] = {}
        for tile_abs, names in batch.tiles.items():
            col_a, row_s, pkt_s = tile_abs.split("|")
            col_rel = int(col_a) - self._start_col
            tile_type = PKT_TO_TILE_TYPE[int(pkt_s)]
            if probe_slot_capacity(insts_bytes, col_rel, int(row_s), tile_type) > 0:
                traceable_abs.setdefault(tile_abs, set()).update(names)
            # else: silently drop -- un-traceable tile flows to never_fired

        # build_active_plan splits to <=8 slots and rides the anchor in slot 0
        # of the anchor tile in every batch. Both active tiles and anchor are
        # ABSOLUTE col; configure_batch converts to RELATIVE for the patcher.
        plan = build_active_plan(traceable_abs, anchor=self._anchor_event,
                                 anchor_tile=self._anchor_tile_abs)
        run_dirs: List[str] = []
        base = self._out_root / f"capture_{self._iter:02d}"
        for i in range(self._n_runs):
            rd = base / f"run_{i:02d}"
            rd.mkdir(parents=True, exist_ok=True)
            runner = HwRunner(xclbin, stderr_log=rd / "hw.runner.log")
            try:
                capture(plan, runner, test=self._test, out_dir=rd,
                        start_col=self._start_col, instr=insts)
            finally:
                runner.close()
            run_dirs.append(str(rd))
        self._iter += 1
        return run_dirs

