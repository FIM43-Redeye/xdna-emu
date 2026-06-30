#!/usr/bin/env python3
"""Sweep every trace event for a compiled test.

Compiles once (via an already-built xclbin + insts.bin), then walks every
event ID for the requested tile type, patching the insts.bin for each
8-event batch, running the pipeline on HW and EMU, and recording per-event
cycle counts and fire counts into a single JSON matrix.

Goal: produce a full per-(tile, event) visibility map for both HW and EMU
without any recompilation cost. Caller is responsible for providing a
trace-injected xclbin with the target tile already in the trace flow (we
patch the event-selection registers; we do not re-route anything).

Output JSON schema (one entry per event):
  {
    "test": "<test_name>",
    "compiler": "<chess|peano>",
    "tile": {"col": C, "row": R, "type": "core|memmod|memtile|shim"},
    "events": [
      {
        "id": 37,
        "name": "INSTR_VECTOR",
        "slot": 2,
        "batch": 0,
        "hw": {"cycles": 41181, "events": 24},
        "emu": {"cycles": 20612, "events": 10}
      },
      ...
    ]
  }

Event enumeration
-----------------
Events live in mlir-aie/build/include/xaienginecdo_static/xaiengine/xaie_events_aieml.h
as preprocessor defines `XAIEML_EVENTS_<MOD>_<NAME> <id>U`. That header is
the same one consumed by aie-rt and by AIEInsertTraceFlows (indirectly via
the register database), so keeping this enumeration sourced from it
guarantees our sweep stays aligned with both tools.

Runs
----
Each batch invocation is independent. Runner failures (TDR, state=8, EMU
hang) are recorded as failure markers in the output rather than aborting
the sweep; that way a single bad event doesn't wipe a 50-event run.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Shared default for --perfcnt-period.  See tools/perfcnt_defaults.py.
sys.path.insert(0, str(Path(__file__).parent))
from perfcnt_defaults import DEFAULT_PERFCNT_PERIOD  # noqa: E402

from trace_runner import (
    REPO_ROOT, MLIR_AIE_ROOT, RUNNER, PATCH_TOOL, PARSE_TOOL,
    _MOD_TO_TILE_TYPE, _TILE_TYPE_TO_MOD, _MODE_INT, _GROUNDING_BY_TILE_TYPE,
    RunResult, RunnerSession, ParseSession,
    _run_patch, _run_patch_multi, _relabel_events, _parse_trace_bin, _run_one_side,
    _discover_trace_buf_idx, _find_cdo_preambles, _find_post_lowering_mlir,
)

EVENTS_HEADER = (
    MLIR_AIE_ROOT / "build" / "include" / "xaienginecdo_static"
    / "xaiengine" / "xaie_events_aieml.h"
)


# ---------------------------------------------------------------------------
# Event enumeration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventDef:
    name: str       # e.g. "INSTR_VECTOR"
    id: int         # event ID as seen by the trace unit


def load_events(tile_type: str) -> List[EventDef]:
    """Parse aie-rt's event header and return all events for the given tile
    type, sorted by ID. We source from the build-tree header (not the
    installed one) because it's what the linked libaiengine sees; if the
    two disagree, the one next to the running library wins at runtime.
    """
    if not EVENTS_HEADER.exists():
        raise FileNotFoundError(f"missing events header: {EVENTS_HEADER}")
    try:
        mod_prefix = _TILE_TYPE_TO_MOD[tile_type]
    except KeyError:
        raise ValueError(
            f"unknown tile_type {tile_type!r}; want one of "
            f"{sorted(_TILE_TYPE_TO_MOD)}"
        )
    # Match longest prefix first so MEM_TILE doesn't get caught by MEM.
    prefix = f"XAIEML_EVENTS_{mod_prefix}_"
    pattern = re.compile(
        rf"^#define\s+{re.escape(prefix)}(\S+)\s+(\d+)U?\s*$"
    )
    seen: Dict[int, str] = {}
    for line in EVENTS_HEADER.read_text().splitlines():
        m = pattern.match(line)
        if not m:
            continue
        name, id_str = m.group(1), m.group(2)
        eid = int(id_str)
        # Later-defined aliases occasionally reuse IDs; keep the first so the
        # output is stable regardless of header ordering.
        seen.setdefault(eid, name)
    return [EventDef(name=n, id=i) for i, n in sorted(seen.items())]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def batch_events(events: List[EventDef], slots: int = 8) -> List[List[EventDef]]:
    """Group events into fixed-size batches. One batch = one trace config =
    one run. With 8 slots the overhead is ceil(N/8) runs per tile."""
    return [events[i:i + slots] for i in range(0, len(events), slots)]


def _build_batches(
    events: List[EventDef],
    slots: int = 8,
    ground_event: Optional[EventDef] = None,
) -> List[List[EventDef]]:
    """Group sweep events into 8-slot batches.

    Without grounding: straightforward slice into fixed-size batches.

    With grounding: slot 0 of every batch is reserved for the grounding
    event, so each batch carries at most (slots-1) swept events. The
    grounding event is filtered out of the sweep list if the caller
    included it, so it never double-counts in the matrix.

    An empty sweep list + non-None ground_event still produces one batch
    carrying just the grounding event -- useful for tile/event discovery
    without a full sweep.
    """
    if ground_event is None:
        return [events[i:i + slots] for i in range(0, len(events), slots)]
    sweep = [e for e in events if e.id != ground_event.id]
    per_batch = slots - 1
    batches: List[List[EventDef]] = []
    for i in range(0, len(sweep), per_batch):
        batches.append([ground_event] + sweep[i:i + per_batch])
    if not batches:
        batches.append([ground_event])
    return batches


# ---------------------------------------------------------------------------
# Anchor / merge
# ---------------------------------------------------------------------------

def _anchor_events(
    events: List[Dict], ground_slot: int = 0,
) -> Tuple[Optional[int], List[Dict]]:
    """Subtract the grounding event's timestamp from every other event's
    timestamp in a batch.

    Returns (anchor_ts, anchored_events):
    - anchor_ts is the `ts` of the *first* event recorded on ground_slot,
      or None if nothing fired there (grounding failure -- caller should
      flag and not trust anchored_ts values).
    - anchored_events excludes the grounding event itself (it's
      instrumentation, not swept data) and every remaining event carries a
      new `ts_anchored` field. If anchor_ts is None, ts_anchored is also
      None on every event.

    All events on the grounding slot are dropped from the output, not
    just the first -- the grounding slot is pure instrumentation, and if
    the event fires twice in a batch that's a diagnostic to surface via
    the anchor (the anchor is always the *first* firing), not something
    to pollute the swept-event timeline with.
    """
    anchor: Optional[int] = None
    for ev in events:
        if ev.get("slot") == ground_slot:
            anchor = ev.get("ts")
            break
    out: List[Dict] = []
    for ev in events:
        if ev.get("slot") == ground_slot:
            continue
        ts = ev.get("ts")
        copy = dict(ev)
        if anchor is not None and ts is not None:
            copy["ts_anchored"] = ts - anchor
        else:
            copy["ts_anchored"] = None
        out.append(copy)
    return anchor, out


def _merge_anchored(
    per_batch: List[Tuple[int, List[Dict]]],
) -> List[Dict]:
    """Flatten per-batch anchored event lists into one sorted timeline.

    Sort key: (is_unanchored, ts_anchored). Unanchored events (batches
    where the grounding event never fired) sink to the end so the
    deterministic prefix reads cleanly. Each event gets a `source_batch`
    tag so post-hoc attribution stays possible.

    Stable within the same ts_anchored value so tied events preserve
    batch order.
    """
    items: List[Dict] = []
    for batch_idx, events in per_batch:
        for ev in events:
            c = dict(ev)
            c["source_batch"] = batch_idx
            items.append(c)
    items.sort(key=lambda e: (
        e.get("ts_anchored") is None,
        e.get("ts_anchored") if e.get("ts_anchored") is not None else 0,
    ))
    return items


# ---------------------------------------------------------------------------
# Lockstep multi-tile batching (A.2 mode-1 sweep)
# ---------------------------------------------------------------------------

def _build_lockstep_batches(cursors: dict) -> list:
    """Generate per-batch event assignments across all tile cursors.

    Each cursor (one per (tile, module-type)) holds a sweep list and a
    remaining_slots count (= 8 - len(grounding)). Per batch, each cursor
    consumes its next remaining_slots events. Total batch count =
    max(ceil(per-cursor sweep length / per-cursor remaining_slots)).
    A cursor with an empty sweep still contributes one batch (so the
    caller gets at least one grounding-only batch). Cursors that exhaust
    early emit empty assignments (grounding-only) for the remaining batches.
    """
    if not cursors:
        return []
    n_batches = max(
        (1 if not c["sweep"]
         else (len(c["sweep"]) + c["remaining_slots"] - 1) // c["remaining_slots"])
        for c in cursors.values()
    )
    batches: list = []
    for batch_idx in range(n_batches):
        batch: dict = {}
        for key, c in cursors.items():
            start = batch_idx * c["remaining_slots"]
            end = start + c["remaining_slots"]
            batch[key] = list(c["sweep"][start:end])
        batches.append(batch)
    return batches


# ---------------------------------------------------------------------------
# Grounding-PC consistency check (A.2b cross-batch invariance)
# ---------------------------------------------------------------------------

def _check_grounding_pc_invariance(batches_dir: Path,
                                   grounding_events: List[str]) -> dict:
    """Verify that HW grounding-event PCs are batch-invariant.

    Reads hw/trace.events.json from each batch_NNN subdirectory of
    batches_dir. For each grounding event name, checks that the set of
    `ts` values observed in the first batch matches every subsequent batch.
    A mismatch means the kernel's entry PC shifted between batches (e.g.
    the kernel was recompiled or the patch altered something it shouldn't
    have), making cross-batch PC join unreliable.

    Returns a dict with keys:
      - "unsafe_for_pc_join": bool
      - "reason": str | None  (human-readable explanation on failure)
      - "per_batch_grounding_pcs": {batch_idx: {event_name: [sorted ts list]}}
    """
    per_batch: Dict[int, Dict[str, set]] = {}
    for batch_dir in sorted(batches_dir.glob("batch_*")):
        events_json = batch_dir / "hw" / "trace.events.json"
        if not events_json.exists():
            continue
        try:
            bidx = int(batch_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        per_batch[bidx] = {}
        try:
            records = json.loads(events_json.read_text())
        except Exception:
            continue
        events_list = (
            records.get("events", records)
            if isinstance(records, dict) else records
        )
        for rec in events_list:
            name = rec.get("name", "")
            if name in grounding_events:
                per_batch[bidx].setdefault(name, set()).add(rec["ts"])

    for ev in grounding_events:
        seen_pcs: Optional[set] = None
        for bidx, by_ev in sorted(per_batch.items()):
            if ev not in by_ev:
                continue
            if seen_pcs is None:
                seen_pcs = by_ev[ev]
            elif by_ev[ev] != seen_pcs:
                return {
                    "unsafe_for_pc_join": True,
                    "reason": (
                        f"grounding event {ev} PC drifted: "
                        f"first_batch={sorted(seen_pcs)} "
                        f"batch_{bidx}={sorted(by_ev[ev])}"
                    ),
                    "per_batch_grounding_pcs": {
                        k: {n: sorted(s) for n, s in v.items()}
                        for k, v in per_batch.items()
                    },
                }

    return {
        "unsafe_for_pc_join": False,
        "reason": None,
        "per_batch_grounding_pcs": {
            k: {n: sorted(s) for n, s in v.items()}
            for k, v in per_batch.items()
        },
    }



# ---------------------------------------------------------------------------
# Test-config discovery
# ---------------------------------------------------------------------------
#
# Tests produce different runtime-sequence filenames depending on what
# their run.lit tells aiecc.py to emit. Examples:
#
#   default test               -> insts.bin
#   ctrl_packet_reconfig       -> aie_run_seq.bin + ctrlpkt.bin (as --input)
#   ctrl_packet_reconfig_elf   -> same, plus a pre-built aie.elf
#
# Hardcoding "insts.bin" made the sweep silently skip entire families of
# tests. Parsing run.lit authoritatively fixes that: we read the exact
# --npu-insts-name / --ctrlpkt-name flags the test compiles with and use
# those. Mirrors scripts/emu-bridge-test.sh's _discover_aiecc_name helpers
# (kept in shell because bridge-test.sh is shell; kept here in Python to
# avoid inter-process calls for every sweep combo).

_AIECC_FLAG_RE = re.compile(r"--(npu-insts-name|ctrlpkt-name)=(\S+)")


@dataclass(frozen=True)
class TestConfig:
    """What a test actually produces and needs at runtime."""
    insts_name: str                 # filename in build_dir (patch target)
    ctrlpkt_name: Optional[str]     # None if test doesn't generate one


def _find_run_lit(test_src: Path) -> Optional[Path]:
    """Locate run.lit for a test. Most tests name it exactly; a handful
    use variant names like test.lit."""
    for name in ("run.lit", "test.lit"):
        cand = test_src / name
        if cand.is_file():
            return cand
    return None


def discover_test_config(test_name: str) -> TestConfig:
    """Parse a test's run.lit to learn which runtime-sequence file aiecc
    was told to emit.

    Lookup precedence:
      1. The last occurrence of --npu-insts-name=X wins (run.lit can
         mention it multiple times for different build variants).
      2. Fall back to 'insts.bin' when the flag is absent.
      3. --ctrlpkt-name=Y sets ctrlpkt_name; absence leaves it None.

    Does not check whether the files exist in the build tree -- that's
    the sweep driver's job. This function only surfaces the *names* from
    the source-of-truth run.lit.
    """
    test_src = MLIR_AIE_ROOT / "test" / "npu-xrt" / test_name
    lit = _find_run_lit(test_src)
    insts_name = "insts.bin"
    ctrlpkt_name: Optional[str] = None
    if lit is not None:
        for m in _AIECC_FLAG_RE.finditer(lit.read_text()):
            flag, value = m.group(1), m.group(2)
            if flag == "npu-insts-name":
                insts_name = value
            elif flag == "ctrlpkt-name":
                ctrlpkt_name = value
    return TestConfig(insts_name=insts_name, ctrlpkt_name=ctrlpkt_name)


def _resolve_ground_event(
    tile_type: str, name: Optional[str],
) -> Optional[EventDef]:
    """Resolve a grounding event name against the tile's event table.

    Returns None if the caller passed no grounding event. Raises
    ValueError if the name is unknown for this tile type -- better to
    surface the typo early than to have every batch silently run without
    anchoring.
    """
    if not name:
        return None
    for ev in load_events(tile_type):
        if ev.name == name:
            return ev
    raise ValueError(
        f"grounding event {name!r} not defined for tile_type {tile_type!r}; "
        f"check mlir-aie/build/include/.../xaie_events_aieml.h"
    )


@dataclass(frozen=True)
class TileSpec:
    """A tile to include in a multi-tile sweep.

    tile_type must be one of the four keys in trace-patch-events.py's
    _TRACE_EVENT_REGS. For multi-tile sweeps we currently require every
    TileSpec to share the same tile_type (different types have disjoint
    event lists, so a single sweep can't meaningfully cover both in the
    same batches).
    """
    col: int
    row: int
    tile_type: str

    @property
    def label(self) -> str:
        return f"{self.tile_type}_c{self.col}r{self.row}"


def _tile_output_path(
    out_dir: Path, test_name: str, compiler: str, tile: TileSpec,
) -> Path:
    return out_dir / f"{test_name}.{compiler}.{tile.label}.json"


def sweep_multi(
    test_name: str,
    compiler: str,
    tiles: List[TileSpec],
    build_dir: Path,
    out_dir: Path,
    events_filter: Optional[List[str]] = None,
    run_hw: bool = True,
    run_emu: bool = True,
    ctrlpkt: Optional[Path] = None,
    ground_event_name: Optional[str] = None,
    reuse_ctx: bool = False,
    jobs: int = 1,
) -> List[dict]:
    """Sweep every trace event across a set of tiles in one kernel run.

    The xclbin's trace routing is already set up at compile time for
    every tile in the traced MLIR; all tiles' trace packets converge to
    the same shim DMA output. So a single kernel invocation with every
    tile's Trace_Event registers programmed captures all of them
    simultaneously, and we split the resulting packet stream per-tile by
    (row, pkt_type) during relabelling.

    Returns one summary dict per tile (same schema as the old single-tile
    sweep). Each summary is also written to out_dir as
    `<test>.<compiler>.<tile.label>.json`.

    Batch size is min(slot_capacity) across the tiles -- if one tile has
    4 slots (Trace_Event0-only compile) and another has 8, we batch at
    4 so every tile has valid data in every batch. Lockstep batching
    across tiles keeps the per-tile result set aligned with the batch
    index for free.
    """
    if not tiles:
        raise ValueError("sweep_multi requires at least one tile")
    tile_types = {t.tile_type for t in tiles}
    if len(tile_types) > 1:
        raise ValueError(
            f"sweep_multi requires all tiles to share tile_type; got {tile_types}. "
            f"Different tile types have disjoint event ID spaces, so a shared "
            f"sweep can't cover both -- invoke once per tile_type."
        )
    tile_type = tiles[0].tile_type

    cfg = discover_test_config(test_name)
    xclbin = build_dir / "aie.xclbin"
    insts = build_dir / cfg.insts_name
    if not xclbin.exists():
        raise FileNotFoundError(
            f"xclbin missing in {build_dir}; compile with "
            f"--with-hw-cycles first so trace routing is present"
        )
    if not insts.exists():
        raise FileNotFoundError(
            f"runtime-sequence file {cfg.insts_name!r} missing in "
            f"{build_dir} (discovered from {test_name}/run.lit)"
        )
    if ctrlpkt is None and cfg.ctrlpkt_name:
        cand = build_dir / cfg.ctrlpkt_name
        if cand.is_file():
            ctrlpkt = cand
    mlir = _find_post_lowering_mlir(build_dir)
    if mlir is None:
        raise FileNotFoundError(
            f"no input_with_addresses.mlir under {build_dir}/*.prj/; "
            f"parse-trace cannot decode without it"
        )

    all_events = load_events(tile_type)
    if events_filter:
        wanted = set(events_filter)
        all_events = [e for e in all_events if e.name in wanted]
    ground_event = _resolve_ground_event(tile_type, ground_event_name)

    import importlib.util as _imputil
    _pspec = _imputil.spec_from_file_location(
        "trace_patch_events", REPO_ROOT / "tools" / "trace-patch-events.py",
    )
    _pmod = _imputil.module_from_spec(_pspec)
    sys.modules["trace_patch_events"] = _pmod
    _pspec.loader.exec_module(_pmod)
    insts_bytes = insts.read_bytes()
    per_tile_capacity: Dict[TileSpec, int] = {}
    for tile in tiles:
        cap = _pmod.probe_slot_capacity(
            insts_bytes, tile.col, tile.row, tile.tile_type,
        )
        if cap == 0:
            raise FileNotFoundError(
                f"xclbin at {build_dir} has no Trace_Event writes for tile "
                f"{tile.label}; re-compile with trace injection enabled "
                f"or drop this tile from the sweep."
            )
        per_tile_capacity[tile] = cap
    min_cap = min(per_tile_capacity.values())

    batches = _build_batches(
        all_events, slots=min_cap, ground_event=ground_event,
    )
    first_sweep_slot = 1 if ground_event is not None else 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-tile accumulators. Each tile carries its own result list + anchors.
    per_tile_results: Dict[TileSpec, List[dict]] = {t: [] for t in tiles}
    per_tile_hw_anchored: Dict[TileSpec, List[Tuple[int, List[Dict]]]] = {t: [] for t in tiles}
    per_tile_emu_anchored: Dict[TileSpec, List[Tuple[int, List[Dict]]]] = {t: [] for t in tiles}
    per_tile_hw_anchors: Dict[TileSpec, List[Dict]] = {t: [] for t in tiles}
    per_tile_emu_anchors: Dict[TileSpec, List[Dict]] = {t: [] for t in tiles}

    # Single work-dir shared across all tiles of this invocation. The
    # per-batch trace artifacts are one run -- splitting per-tile is a
    # post-processing filter, not a separate invocation.
    work_dir = out_dir / f"{test_name}.{compiler}.multitile.work"
    work_dir.mkdir(parents=True, exist_ok=True)
    patched = work_dir / "insts.patched.bin"

    # Open long-lived runner sessions for HW and EMU up front. Each
    # session holds a single xrt::device + xclbin across every batch,
    # so the ~228 ms per-process startup cost is paid once per sweep
    # instead of once per batch.
    hw_runner_env: Dict[str, str] = {}
    emu_runner_env: Dict[str, str] = {
        "XDNA_EMU": os.environ.get("XDNA_EMU", "debug"),
        "XDNA_EMU_LOG_LEVEL": os.environ.get("XDNA_EMU_LOG_LEVEL", "info"),
        "XRT_DEVICE_BDF": "ffff:ff:1f.0",
        # SP-4b: the origin_D sidecar (per-module broadcast timer-reset
        # arrival) the engine writes alongside its trace flush. One value
        # for the whole session is correct here, not a per-batch guess:
        # origin_D depends only on device topology + this test's single
        # flood source, not on which trace events a given batch patched
        # in, so every batch of this test produces byte-identical sidecar
        # content -- harmless idempotent re-writes, not a race.
        "XDNA_EMU_ORIGIN_D_OUT": str(work_dir / "origin_d.json"),
    }
    # Trace-BO discovery and CDO preamble paths are only used in the
    # --reuse-ctx flow. Baseline trace-sweep keeps the runner's legacy
    # behavior untouched (last-buffer-kernarg = trace, no preamble
    # injection), so opting *out* of --reuse-ctx is risk-free.
    trace_buf_idx: Optional[int] = None
    cdo_preambles: List[Path] = []
    if reuse_ctx:
        trace_buf_idx = _discover_trace_buf_idx(insts)
        if trace_buf_idx is None:
            raise FileNotFoundError(
                f"--reuse-ctx requested but trace_buf_idx could not be "
                f"discovered from {insts}; the runner's legacy "
                f"last-buffer heuristic isn't safe under hwctx reuse "
                f"because the wrong BO would be re-bound on every run"
            )
        cdo_preambles = _find_cdo_preambles(mlir)
        if not cdo_preambles:
            raise FileNotFoundError(
                f"--reuse-ctx requested but main_aie_cdo_init.bin / "
                f"main_aie_cdo_enable.bin are missing under {mlir.parent}"
            )

    hw_session: Optional[RunnerSession] = None
    emu_session: Optional[RunnerSession] = None
    if run_hw:
        hw_session = RunnerSession(
            xclbin=xclbin, runner_env=hw_runner_env, side="HW",
            stderr_log=work_dir / "hw.runner.log",
            cdo_preambles=cdo_preambles, trace_buf_idx=trace_buf_idx,
            reuse_ctx=reuse_ctx,
        )
    if run_emu:
        # The emulator side is already deterministic and doesn't have
        # the hwctx-reuse limitation, but we still pass the same args
        # so HW/EMU stay symmetric -- both sides get the same insts.bin
        # preamble and identify the trace BO the same way.
        emu_session = RunnerSession(
            xclbin=xclbin, runner_env=emu_runner_env, side="EMU",
            stderr_log=work_dir / "emu.runner.log",
            cdo_preambles=cdo_preambles, trace_buf_idx=trace_buf_idx,
            reuse_ctx=reuse_ctx,
        )

    # One persistent decoder process per side. The mlir-aie + numpy
    # imports cost ~620 ms per fresh interpreter, which dominates the
    # sweep wall clock; spawning them once and routing all batch
    # decodes through the long-lived processes drops per-decode cost
    # to ~100 ms.
    hw_parser: Optional[ParseSession] = None
    emu_parser: Optional[ParseSession] = None
    if run_hw:
        hw_parser = ParseSession(
            side="HW", stderr_log=work_dir / "hw.parser.log",
        )
    if run_emu:
        emu_parser = ParseSession(
            side="EMU", stderr_log=work_dir / "emu.parser.log",
        )

    t_start = time.time()

    # Two-phase sweep: HW first (serial -- one-job-in-flight rule on the
    # NPU), then EMU (parallel if jobs > 1, since EMU sessions are
    # independent subprocesses with no shared device). Per-tile
    # aggregation runs last in b_idx order so results are deterministic
    # regardless of EMU completion order.
    #
    # Per-batch patched.bin files (was a single shared file before): the
    # HW phase patches each, EMU phase reuses the same files on disk.
    # Negligible disk cost; required for parallel EMU correctness.
    skipped_hw = RunResult(ok=False, cycles=None, events_count=None,
                           error="hw skipped")
    skipped_emu = RunResult(ok=False, cycles=None, events_count=None,
                            error="emu skipped")
    batch_patched = [work_dir / f"insts.patched.b{b:02d}.bin"
                     for b in range(len(batches))]
    batch_hw_events_out = [work_dir / f"b{b}.hw.events.json"
                           for b in range(len(batches))]
    batch_emu_events_out = [work_dir / f"b{b}.emu.events.json"
                            for b in range(len(batches))]
    batch_hw_res: List[Optional[RunResult]] = [None] * len(batches)
    batch_emu_res: List[Optional[RunResult]] = [None] * len(batches)

    def _patch_batch(b_idx: int) -> None:
        event_ids = [e.id for e in batches[b_idx]]
        # Program the same event set on every tile in lockstep. Different
        # tile types would need different event IDs, but we enforced
        # same-tile-type above; same type => same IDs work across tiles.
        patches = [
            (t.col, t.row, t.tile_type, event_ids) for t in tiles
        ]
        _run_patch_multi(insts, batch_patched[b_idx], patches)

    # ---- Phase A: HW sweep (serial; required by one-job-in-flight) ----
    if run_hw:
        for b_idx in range(len(batches)):
            _patch_batch(b_idx)
            # Reset before every batch so the shim DMA BD write counter
            # is zeroed (each batch's events land at trace BO offset 0
            # instead of the cumulative offset where the prior batch
            # stopped). Cheap on the first batch (empty cache) and
            # ~50ms thereafter (partition realloc).
            hw_session.reset()
            batch_hw_res[b_idx] = _run_one_side(
                side="HW",
                session=hw_session,
                runner_env=hw_runner_env,
                instr=batch_patched[b_idx],
                trace_bin=work_dir / f"b{b_idx}.trace_hw.bin",
                mlir=mlir,
                events_out=batch_hw_events_out[b_idx],
                cycles_out=work_dir / f"b{b_idx}.hw.cycles.txt",
                parse_log=work_dir / f"b{b_idx}.hw.parse.log",
                ctrlpkt=ctrlpkt,
                parser_session=hw_parser,
            )
            print(f"[sweep-multi] HW batch {b_idx + 1}/{len(batches)}: "
                  f"hw_cyc={batch_hw_res[b_idx].cycles}", flush=True)
    else:
        for b_idx in range(len(batches)):
            batch_hw_res[b_idx] = skipped_hw

    # ---- Phase B: EMU sweep (parallel if jobs > 1) --------------------
    if run_emu:
        # If HW didn't run, patched.bin files don't exist yet; patch now.
        if not run_hw:
            for b_idx in range(len(batches)):
                _patch_batch(b_idx)

        effective_jobs = max(1, min(jobs, len(batches)))
        if effective_jobs > 1:
            # Pool of (RunnerSession, ParseSession) pairs. We already
            # created one of each above (emu_session, emu_parser); spawn
            # the remaining effective_jobs-1 here. Each worker borrows
            # one pair from a queue, returns it when done.
            extra_emu_sessions = [
                RunnerSession(
                    xclbin=xclbin, runner_env=emu_runner_env,
                    side=f"EMU#{i + 2}",
                    stderr_log=work_dir / f"emu.runner.{i + 2}.log",
                    cdo_preambles=cdo_preambles,
                    trace_buf_idx=trace_buf_idx,
                    reuse_ctx=reuse_ctx,
                )
                for i in range(effective_jobs - 1)
            ]
            extra_emu_parsers = [
                ParseSession(
                    side=f"EMU#{i + 2}",
                    stderr_log=work_dir / f"emu.parser.{i + 2}.log",
                )
                for i in range(effective_jobs - 1)
            ]
            all_emu_sessions = [emu_session] + extra_emu_sessions
            all_emu_parsers = [emu_parser] + extra_emu_parsers
            pair_q: "queue.Queue[Tuple[RunnerSession, ParseSession]]" = queue.Queue()
            for pair in zip(all_emu_sessions, all_emu_parsers):
                pair_q.put(pair)

            def _run_emu_batch(b_idx: int) -> Tuple[int, RunResult]:
                s, p = pair_q.get()
                try:
                    # Per-batch hw_context reset so the pool of sessions
                    # produces identical results regardless of which
                    # session services a given batch (otherwise the trace
                    # shim DMA's cumulative offset leaks across the pool).
                    s.reset()
                    res = _run_one_side(
                        side="EMU",
                        session=s,
                        runner_env=emu_runner_env,
                        instr=batch_patched[b_idx],
                        trace_bin=work_dir / f"b{b_idx}.trace_emu.bin",
                        mlir=mlir,
                        events_out=batch_emu_events_out[b_idx],
                        cycles_out=work_dir / f"b{b_idx}.emu.cycles.txt",
                        parse_log=work_dir / f"b{b_idx}.emu.parse.log",
                        ctrlpkt=ctrlpkt,
                        parser_session=p,
                    )
                finally:
                    pair_q.put((s, p))
                return b_idx, res

            with ThreadPoolExecutor(max_workers=effective_jobs) as ex:
                futures = [ex.submit(_run_emu_batch, b)
                           for b in range(len(batches))]
                for f in as_completed(futures):
                    b_idx, res = f.result()
                    batch_emu_res[b_idx] = res
                    print(f"[sweep-multi] EMU batch {b_idx + 1}/{len(batches)} "
                          f"done (parallel -j{effective_jobs}): "
                          f"emu_cyc={res.cycles}", flush=True)

            for s in extra_emu_sessions:
                s.close()
            for p in extra_emu_parsers:
                p.close()
        else:
            for b_idx in range(len(batches)):
                # Per-batch reset so j=1 matches the parallel path's
                # per-batch isolation. See _run_emu_batch above.
                emu_session.reset()
                batch_emu_res[b_idx] = _run_one_side(
                    side="EMU",
                    session=emu_session,
                    runner_env=emu_runner_env,
                    instr=batch_patched[b_idx],
                    trace_bin=work_dir / f"b{b_idx}.trace_emu.bin",
                    mlir=mlir,
                    events_out=batch_emu_events_out[b_idx],
                    cycles_out=work_dir / f"b{b_idx}.emu.cycles.txt",
                    parse_log=work_dir / f"b{b_idx}.emu.parse.log",
                    ctrlpkt=ctrlpkt,
                    parser_session=emu_parser,
                )
                print(f"[sweep-multi] EMU batch {b_idx + 1}/{len(batches)}: "
                      f"emu_cyc={batch_emu_res[b_idx].cycles}", flush=True)
    else:
        for b_idx in range(len(batches)):
            batch_emu_res[b_idx] = skipped_emu

    # ---- Phase C: per-tile aggregation (serial, in b_idx order) -------
    for b_idx, batch in enumerate(batches):
        hw_res = batch_hw_res[b_idx]
        emu_res = batch_emu_res[b_idx]
        hw_events_out = batch_hw_events_out[b_idx]
        emu_events_out = batch_emu_events_out[b_idx]

        # Split the one parsed events.json per-tile, relabel each tile's
        # filtered subset with this batch's event names. _relabel_events
        # is idempotent across tiles because its filter key (row,
        # pkt_type) is different for each tile, so rewriting the names
        # field on the subset it claims doesn't corrupt other tiles'
        # rows.
        for tile in tiles:
            hw_tile_events: List[Dict] = []
            emu_tile_events: List[Dict] = []
            hw_per_event: Dict[str, int] = {}
            emu_per_event: Dict[str, int] = {}
            if hw_res.ok:
                hw_per_event, hw_tile_events = _relabel_events(
                    hw_events_out, tile.col, tile.row, tile.tile_type, batch)
            if emu_res.ok:
                emu_per_event, emu_tile_events = _relabel_events(
                    emu_events_out, tile.col, tile.row, tile.tile_type, batch)

            if ground_event is not None:
                hw_anchor, hw_anchored = _anchor_events(hw_tile_events, ground_slot=0)
                emu_anchor, emu_anchored = _anchor_events(emu_tile_events, ground_slot=0)
            else:
                hw_anchor, hw_anchored = None, [dict(e, ts_anchored=None) for e in hw_tile_events]
                emu_anchor, emu_anchored = None, [dict(e, ts_anchored=None) for e in emu_tile_events]
            per_tile_hw_anchored[tile].append((b_idx, hw_anchored))
            per_tile_emu_anchored[tile].append((b_idx, emu_anchored))
            per_tile_hw_anchors[tile].append({"batch": b_idx, "anchor_ts": hw_anchor})
            per_tile_emu_anchors[tile].append({"batch": b_idx, "anchor_ts": emu_anchor})

            for slot, ev in enumerate(batch):
                if slot < first_sweep_slot:
                    continue
                per_tile_results[tile].append({
                    "id": ev.id,
                    "name": ev.name,
                    "slot": slot,
                    "batch": b_idx,
                    "hw": {
                        "cycles": hw_res.cycles,
                        "events_total": hw_res.events_count,
                        "fired": hw_per_event.get(ev.name, 0) if hw_res.ok else None,
                        "error": hw_res.error,
                    },
                    "emu": {
                        "cycles": emu_res.cycles,
                        "events_total": emu_res.events_count,
                        "fired": emu_per_event.get(ev.name, 0) if emu_res.ok else None,
                        "error": emu_res.error,
                    },
                })

    # Close sessions here rather than via try/finally so the cleanup
    # cost stays measurable (visible in elapsed_sec) without altering
    # the sweep result shape. On interrupt the parent process exit
    # takes the children with it; no state is lost that wouldn't
    # already be lost by interrupt.
    if hw_session is not None:
        hw_session.close()
    if emu_session is not None:
        emu_session.close()
    if hw_parser is not None:
        hw_parser.close()
    if emu_parser is not None:
        emu_parser.close()

    summaries: List[dict] = []
    elapsed = round(time.time() - t_start, 2)
    for tile in tiles:
        out_json = _tile_output_path(out_dir, test_name, compiler, tile)
        summary = {
            "test": test_name,
            "compiler": compiler,
            "tile": {"col": tile.col, "row": tile.row, "type": tile.tile_type},
            "grounding_event": ground_event.name if ground_event else None,
            "events": per_tile_results[tile],
            "elapsed_sec": elapsed,
        }
        out_json.write_text(json.dumps(summary, indent=2) + "\n")
        merged_path = out_json.with_suffix(".merged.json")
        merged = {
            "test": test_name,
            "compiler": compiler,
            "tile": {"col": tile.col, "row": tile.row, "type": tile.tile_type},
            "grounding_event": ground_event.name if ground_event else None,
            "batches_merged": len(batches),
            "hw": {
                "anchors": per_tile_hw_anchors[tile],
                "events": _merge_anchored(per_tile_hw_anchored[tile]),
            },
            "emu": {
                "anchors": per_tile_emu_anchors[tile],
                "events": _merge_anchored(per_tile_emu_anchored[tile]),
            },
        }
        merged_path.write_text(json.dumps(merged, indent=2) + "\n")
        summaries.append(summary)
    return summaries

def _build_lockstep_patch_spec(
    tiles: "List[TileSpec]",
    batch_assignment: dict,
    grounding_by_type: "Dict[str, List[str]]",
    all_events_by_type: "Dict[str, List[EventDef]]",
    mode: str,
) -> list:
    """Build the JSON spec list for trace-patch-events.py --multi-tile.

    For each tile in the batch, combines grounding event IDs (fixed slots,
    always present) with this batch's sweep event IDs. Core tiles get the
    requested mode; all other tile types get mode 0 (event-time) since only
    cores support mode-1 PC recording.

    Returns a list of dicts suitable for JSON serialization.
    """
    spec = []
    for tile in tiles:
        cursor_key = f"{tile.tile_type}_{tile.col}_{tile.row}"
        sweep_names = batch_assignment.get(cursor_key, [])
        grounding_names = grounding_by_type.get(tile.tile_type, [])
        ev_by_name = {e.name: e.id for e in all_events_by_type.get(tile.tile_type, [])}
        # Grounding events occupy the first N slots; sweep events follow.
        event_ids = []
        for name in grounding_names:
            eid = ev_by_name.get(name)
            if eid is not None:
                event_ids.append(eid)
        for name in sweep_names:
            eid = ev_by_name.get(name)
            if eid is not None:
                event_ids.append(eid)
        # Pad to 8 slots with 0 (trace unit ignores slot 0 = event-ID 0)
        event_ids = (event_ids + [0] * 8)[:8]
        tile_mode = _MODE_INT.get(mode, 0) if tile.tile_type == "core" else 0
        entry: dict = {
            "col": tile.col,
            "row": tile.row,
            "tile_type": tile.tile_type,
            "events": event_ids,
            "mode": tile_mode,
        }
        spec.append(entry)
    return spec


def sweep_lockstep(
    test_name: str,
    compiler: str,
    tiles: "List[TileSpec]",
    build_dir: Path,
    out_dir: Path,
    run_hw: bool = True,
    run_emu: bool = True,
    ctrlpkt: Optional[Path] = None,
    mode: str = "event_pc",
    core_grounding: Optional[List[str]] = None,
    memmod_grounding: Optional[List[str]] = None,
    memtile_grounding: Optional[List[str]] = None,
    shim_grounding: Optional[List[str]] = None,
    core_sweep: Optional[List[str]] = None,
    memmod_sweep: Optional[List[str]] = None,
    memtile_sweep: Optional[List[str]] = None,
    shim_sweep: Optional[List[str]] = None,
    perfcnt_period: int = DEFAULT_PERFCNT_PERIOD,
    with_mode2_baseline: bool = True,
    reuse_ctx: bool = False,
    jobs: int = 1,
) -> None:
    """Mode-1 lockstep sweep across mixed-type tiles (A.2 path).

    Unlike sweep_multi, tiles may have different tile_types (cores,
    memmods, memtiles, shims). Per-type grounding events are reserved
    in fixed slots. Per-type sweep event lists advance in lockstep
    using _build_lockstep_batches.

    Each batch applies all tile patches in a single --multi-tile
    invocation, then runs HW + EMU together. After the sweep, an
    optional mode-2 (inst_exec) baseline batch is captured for whichever
    sides are enabled (HW under mode2-baseline/hw/, EMU under
    mode2-baseline/emu/).
    A sweep-manifest.json is written to out_dir with per-batch grounding
    PC sets and the cross-batch invariance check result.
    """
    if not tiles:
        raise ValueError("sweep_lockstep requires at least one tile")

    cfg = discover_test_config(test_name)
    xclbin = build_dir / "aie.xclbin"
    insts = build_dir / cfg.insts_name
    for path, label in [(xclbin, "xclbin"), (insts, cfg.insts_name)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} missing in {build_dir}")
    if ctrlpkt is None and cfg.ctrlpkt_name:
        cand = build_dir / cfg.ctrlpkt_name
        if cand.is_file():
            ctrlpkt = cand
    mlir = _find_post_lowering_mlir(build_dir)
    if mlir is None:
        raise FileNotFoundError(
            f"no input_with_addresses.mlir under {build_dir}/*.prj/; "
            f"parse-trace cannot decode without it"
        )

    # Defaults for grounding event lists.
    grounding_by_type: Dict[str, List[str]] = {
        "core":    (core_grounding or
                    [s.strip() for s in _GROUNDING_BY_TILE_TYPE["core"].split(",")]),
        "memmod":  (memmod_grounding or
                    [s.strip() for s in _GROUNDING_BY_TILE_TYPE["memmod"].split(",")]),
        "memtile": (memtile_grounding or
                    [s.strip() for s in _GROUNDING_BY_TILE_TYPE["memtile"].split(",")]),
        "shim":    (shim_grounding or
                    [s.strip() for s in _GROUNDING_BY_TILE_TYPE["shim"].split(",")]),
    }
    sweep_filter_by_type: Dict[str, Optional[List[str]]] = {
        "core":    core_sweep,
        "memmod":  memmod_sweep,
        "memtile": memtile_sweep,
        "shim":    shim_sweep,
    }

    # Load event tables once per tile type present.
    tile_types_present = {t.tile_type for t in tiles}
    all_events_by_type: Dict[str, List[EventDef]] = {}
    for tt in tile_types_present:
        all_events_by_type[tt] = load_events(tt)

    # Build cursors: one per tile (col,row,tile_type).
    # remaining_slots = 8 - len(grounding events for this tile_type).
    cursors: dict = {}
    for tile in tiles:
        tt = tile.tile_type
        grounding = grounding_by_type.get(tt, [])
        remaining = max(1, 8 - len(grounding))
        sweep_filter = sweep_filter_by_type.get(tt)
        ev_names_all = [e.name for e in all_events_by_type.get(tt, [])]
        # Always exclude grounding events from the sweep list. Grounding
        # events occupy fixed leading slots in every batch; allowing them
        # to also land in the sweep tail would emit them twice in
        # _build_lockstep_patch_spec, and the `(event_ids + [0]*8)[:8]`
        # truncation would silently drop sweep slots off the end.
        if sweep_filter is not None:
            sweep_names = [n for n in ev_names_all
                           if n in sweep_filter and n not in grounding]
        else:
            sweep_names = [n for n in ev_names_all if n not in grounding]
        cursor_key = f"{tt}_{tile.col}_{tile.row}"
        cursors[cursor_key] = {
            "sweep": sweep_names,
            "remaining_slots": remaining,
        }

    batches = _build_lockstep_batches(cursors)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / f"{test_name}.{compiler}.lockstep.work"
    work_dir.mkdir(parents=True, exist_ok=True)
    patched = work_dir / "insts.patched.bin"

    hw_runner_env: Dict[str, str] = {}
    emu_runner_env: Dict[str, str] = {
        "XDNA_EMU": os.environ.get("XDNA_EMU", "debug"),
        "XDNA_EMU_LOG_LEVEL": os.environ.get("XDNA_EMU_LOG_LEVEL", "info"),
        "XRT_DEVICE_BDF": "ffff:ff:1f.0",
        # SP-4b: see the matching comment in sweep_multi -- one sidecar path
        # per session is correct since origin_D is batch-invariant for a
        # given test (same flood source across all of this test's batches).
        "XDNA_EMU_ORIGIN_D_OUT": str(work_dir / "origin_d.json"),
    }
    trace_buf_idx: Optional[int] = None
    cdo_preambles: List[Path] = []
    if reuse_ctx:
        trace_buf_idx = _discover_trace_buf_idx(insts)
        if trace_buf_idx is None:
            raise FileNotFoundError(
                f"--reuse-ctx requested but trace_buf_idx could not be "
                f"discovered from {insts}"
            )
        cdo_preambles = _find_cdo_preambles(mlir)
        if not cdo_preambles:
            raise FileNotFoundError(
                f"--reuse-ctx requested but CDO preambles missing under {mlir.parent}"
            )

    hw_session: Optional[RunnerSession] = None
    emu_session: Optional[RunnerSession] = None
    if run_hw:
        hw_session = RunnerSession(
            xclbin=xclbin, runner_env=hw_runner_env, side="HW",
            stderr_log=work_dir / "hw.runner.log",
            cdo_preambles=cdo_preambles, trace_buf_idx=trace_buf_idx,
            reuse_ctx=reuse_ctx,
        )
    if run_emu:
        emu_session = RunnerSession(
            xclbin=xclbin, runner_env=emu_runner_env, side="EMU",
            stderr_log=work_dir / "emu.runner.log",
            cdo_preambles=cdo_preambles, trace_buf_idx=trace_buf_idx,
            reuse_ctx=reuse_ctx,
        )
    hw_parser: Optional[ParseSession] = None
    emu_parser: Optional[ParseSession] = None
    if run_hw:
        hw_parser = ParseSession(side="HW", stderr_log=work_dir / "hw.parser.log")
    if run_emu:
        emu_parser = ParseSession(side="EMU", stderr_log=work_dir / "emu.parser.log")

    t_start = time.time()
    # Counters tracked at outer scope so the manifest can describe the
    # state we reached even if the sweep loop or mode-2 batch raises.
    mode2_succeeded = False
    completed_batches = 0
    sweep_error: Optional[str] = None
    sweep_exc_obj: Optional[BaseException] = None

    try:
        # Two-phase sweep: HW first (serial -- one-job-in-flight rule
        # on the NPU), then EMU (parallel if jobs > 1). See sweep_multi
        # for the same shape.
        #
        # Per-batch patched.bin files (one shared file before): HW phase
        # writes each, EMU phase reuses on disk.
        batch_dirs = []
        batch_patched_paths = []
        batch_hw_dirs = []
        batch_emu_dirs = []
        for b_idx in range(len(batches)):
            bd = out_dir / f"batch_{b_idx:02d}"
            hd = bd / "hw"
            ed = bd / "emu"
            hd.mkdir(parents=True, exist_ok=True)
            ed.mkdir(parents=True, exist_ok=True)
            batch_dirs.append(bd)
            batch_hw_dirs.append(hd)
            batch_emu_dirs.append(ed)
            batch_patched_paths.append(work_dir / f"insts.patched.b{b_idx:02d}.bin")

        def _patch_batch_lockstep(b_idx: int) -> None:
            patch_spec = _build_lockstep_patch_spec(
                tiles, batches[b_idx], grounding_by_type,
                all_events_by_type, mode,
            )
            spec_path = work_dir / f"b{b_idx:02d}.patch.json"
            spec_path.write_text(json.dumps(patch_spec, indent=2))
            subprocess.run([
                sys.executable, str(PATCH_TOOL), str(insts),
                "--multi-tile", str(spec_path),
                "--output", str(batch_patched_paths[b_idx]),
            ], check=True, capture_output=True)

        batch_hw_res: List[Optional[RunResult]] = [None] * len(batches)
        batch_emu_res: List[Optional[RunResult]] = [None] * len(batches)
        skipped_hw = RunResult(ok=False, cycles=None, events_count=None,
                               error="hw skipped")
        skipped_emu = RunResult(ok=False, cycles=None, events_count=None,
                                error="emu skipped")

        # ---- Phase A: HW sweep (serial) -------------------------------
        if run_hw:
            for b_idx in range(len(batches)):
                _patch_batch_lockstep(b_idx)
                # Reset before every batch -- see sweep_multi for rationale.
                hw_session.reset()
                batch_hw_res[b_idx] = _run_one_side(
                    side="HW",
                    session=hw_session,
                    runner_env=hw_runner_env,
                    instr=batch_patched_paths[b_idx],
                    trace_bin=batch_hw_dirs[b_idx] / "trace.bin",
                    mlir=mlir,
                    events_out=batch_hw_dirs[b_idx] / "trace.events.json",
                    cycles_out=batch_hw_dirs[b_idx] / "cycles.txt",
                    parse_log=batch_hw_dirs[b_idx] / "parse.log",
                    ctrlpkt=ctrlpkt,
                    parser_session=hw_parser,
                    trace_mode=mode,
                )
                print(
                    f"[sweep-lockstep] HW batch {b_idx + 1}/{len(batches)}: "
                    f"hw_ok={batch_hw_res[b_idx].ok} "
                    f"hw_cyc={batch_hw_res[b_idx].cycles}",
                    flush=True,
                )
        else:
            for b_idx in range(len(batches)):
                batch_hw_res[b_idx] = skipped_hw

        # ---- Phase B: EMU sweep (parallel if jobs > 1) ----------------
        if run_emu:
            # If HW didn't run, patched.bin files don't exist yet.
            if not run_hw:
                for b_idx in range(len(batches)):
                    _patch_batch_lockstep(b_idx)

            effective_jobs = max(1, min(jobs, len(batches)))
            if effective_jobs > 1:
                extra_emu_sessions = [
                    RunnerSession(
                        xclbin=xclbin, runner_env=emu_runner_env,
                        side=f"EMU#{i + 2}",
                        stderr_log=work_dir / f"emu.runner.{i + 2}.log",
                        cdo_preambles=cdo_preambles,
                        trace_buf_idx=trace_buf_idx,
                        reuse_ctx=reuse_ctx,
                    )
                    for i in range(effective_jobs - 1)
                ]
                extra_emu_parsers = [
                    ParseSession(
                        side=f"EMU#{i + 2}",
                        stderr_log=work_dir / f"emu.parser.{i + 2}.log",
                    )
                    for i in range(effective_jobs - 1)
                ]
                all_emu_sessions = [emu_session] + extra_emu_sessions
                all_emu_parsers = [emu_parser] + extra_emu_parsers
                pair_q: "queue.Queue[Tuple[RunnerSession, ParseSession]]" = queue.Queue()
                for pair in zip(all_emu_sessions, all_emu_parsers):
                    pair_q.put(pair)

                def _run_emu_lockstep(b_idx: int) -> Tuple[int, RunResult]:
                    s, p = pair_q.get()
                    try:
                        # Per-batch reset -- see sweep_multi for rationale.
                        s.reset()
                        res = _run_one_side(
                            side="EMU",
                            session=s,
                            runner_env=emu_runner_env,
                            instr=batch_patched_paths[b_idx],
                            trace_bin=batch_emu_dirs[b_idx] / "trace.bin",
                            mlir=mlir,
                            events_out=batch_emu_dirs[b_idx] / "trace.events.json",
                            cycles_out=batch_emu_dirs[b_idx] / "cycles.txt",
                            parse_log=batch_emu_dirs[b_idx] / "parse.log",
                            ctrlpkt=ctrlpkt,
                            parser_session=p,
                            trace_mode=mode,
                        )
                    finally:
                        pair_q.put((s, p))
                    return b_idx, res

                with ThreadPoolExecutor(max_workers=effective_jobs) as ex:
                    futures = [ex.submit(_run_emu_lockstep, b)
                               for b in range(len(batches))]
                    for f in as_completed(futures):
                        b_idx, res = f.result()
                        batch_emu_res[b_idx] = res
                        print(
                            f"[sweep-lockstep] EMU batch {b_idx + 1}/{len(batches)} "
                            f"done (parallel -j{effective_jobs}): "
                            f"emu_ok={res.ok} emu_cyc={res.cycles}",
                            flush=True,
                        )

                for s in extra_emu_sessions:
                    s.close()
                for p in extra_emu_parsers:
                    p.close()
            else:
                for b_idx in range(len(batches)):
                    # Per-batch reset -- see sweep_multi for rationale.
                    emu_session.reset()
                    batch_emu_res[b_idx] = _run_one_side(
                        side="EMU",
                        session=emu_session,
                        runner_env=emu_runner_env,
                        instr=batch_patched_paths[b_idx],
                        trace_bin=batch_emu_dirs[b_idx] / "trace.bin",
                        mlir=mlir,
                        events_out=batch_emu_dirs[b_idx] / "trace.events.json",
                        cycles_out=batch_emu_dirs[b_idx] / "cycles.txt",
                        parse_log=batch_emu_dirs[b_idx] / "parse.log",
                        ctrlpkt=ctrlpkt,
                        parser_session=emu_parser,
                        trace_mode=mode,
                    )
                    print(
                        f"[sweep-lockstep] EMU batch {b_idx + 1}/{len(batches)}: "
                        f"emu_ok={batch_emu_res[b_idx].ok} "
                        f"emu_cyc={batch_emu_res[b_idx].cycles}",
                        flush=True,
                    )
        else:
            for b_idx in range(len(batches)):
                batch_emu_res[b_idx] = skipped_emu

        # All batches finished both phases (whether ok or with recorded
        # errors). Earlier in-flight exceptions are caught below.
        completed_batches = len(batches)

        # ----------------------------------------------------------------
        # Mode-2 finishing batch (HW only, inst_exec).
        # ----------------------------------------------------------------
        # Mode-2 baseline runs in dedicated RunnerSessions (separate from
        # the sweep sessions), so the gating only checks the run_hw /
        # run_emu flags. The hw_session / emu_session existence checks
        # were correct before the dedicated-session refactor; they are
        # now redundant but harmless.
        if with_mode2_baseline and (run_hw or run_emu):
            compute_tiles = [t for t in tiles if t.tile_type == "core"]
            if compute_tiles:
                # No "events" key on purpose: trace-patch-events.py's
                # multi-tile mode treats absent "events" as "skip the
                # event-slot patch for this entry," and mode-2
                # (inst_exec) ignores Trace_Event registers anyway --
                # the per-instruction stream comes from the core, not
                # from the event-slot configuration. Leaving the slots
                # at whatever the last mode-1 batch wrote keeps the
                # patcher diff minimal.
                mode2_spec = [
                    {"col": t.col, "row": t.row, "tile_type": "core", "mode": 2}
                    for t in compute_tiles
                ]
                mode2_spec_path = out_dir / "mode2_patch.json"
                mode2_spec_path.write_text(json.dumps(mode2_spec, indent=2))
                mode2_insts = work_dir / "mode2_insts.bin"
                subprocess.run([
                    sys.executable, str(PATCH_TOOL), str(insts),
                    "--multi-tile", str(mode2_spec_path),
                    "--output", str(mode2_insts),
                ], check=True, capture_output=True)

                mode2_dir = out_dir / "mode2-baseline"

                hw_ok: Optional[bool] = None
                emu_ok: Optional[bool] = None

                # Use FRESH RunnerSessions for the mode-2 baseline. The
                # sweep batches accumulate trace data in the BO via a
                # shim DMA BD whose internal write counter persists
                # across runs in --batch-stdin mode (see
                # bridge-trace-runner.cpp "Note on cumulative offsets
                # across batches"). Reusing the sweep session for
                # mode-2 means the new mode-2 trace lands at the
                # cumulative offset, leaving the leading bytes populated
                # by the most recent mode-1 batch's data. The mode-2
                # parser then sees the leftover mode-1 marker (0xF1) at
                # offset 0 instead of mode-2's 0xF2, decodes it as
                # garbage, and reports zero PCs. Spinning up dedicated
                # sessions for the baseline isolates the state and gives
                # a clean BO + BD counter at offset 0. (The sweep loops
                # also call RunnerSession.reset() between batches now,
                # so a future cleanup could replace these fresh sessions
                # with a reset on the existing one -- but separate
                # processes are still the strongest isolation.)
                if run_hw:
                    mode2_hw_dir = mode2_dir / "hw"
                    mode2_hw_dir.mkdir(parents=True, exist_ok=True)
                    mode2_hw_session = RunnerSession(
                        xclbin=xclbin, runner_env=hw_runner_env, side="HW-mode2",
                        stderr_log=work_dir / "hw.mode2.runner.log",
                        cdo_preambles=cdo_preambles,
                        trace_buf_idx=trace_buf_idx, reuse_ctx=reuse_ctx,
                    )
                    try:
                        mode2_hw_res = _run_one_side(
                            side="HW",
                            session=mode2_hw_session,
                            runner_env=hw_runner_env,
                            instr=mode2_insts,
                            trace_bin=mode2_hw_dir / "trace.bin",
                            mlir=mlir,
                            events_out=mode2_hw_dir / "trace.events.json",
                            cycles_out=mode2_hw_dir / "cycles.txt",
                            parse_log=mode2_hw_dir / "parse.log",
                            ctrlpkt=ctrlpkt,
                            parser_session=hw_parser,
                            trace_mode="inst_exec",
                        )
                        hw_ok = mode2_hw_res.ok
                    finally:
                        mode2_hw_session.close()

                if run_emu:
                    mode2_emu_dir = mode2_dir / "emu"
                    mode2_emu_dir.mkdir(parents=True, exist_ok=True)
                    mode2_emu_session = RunnerSession(
                        xclbin=xclbin, runner_env=emu_runner_env, side="EMU-mode2",
                        stderr_log=work_dir / "emu.mode2.runner.log",
                        cdo_preambles=cdo_preambles,
                        trace_buf_idx=trace_buf_idx, reuse_ctx=reuse_ctx,
                    )
                    try:
                        mode2_emu_res = _run_one_side(
                            side="EMU",
                            session=mode2_emu_session,
                            runner_env=emu_runner_env,
                            instr=mode2_insts,
                            trace_bin=mode2_emu_dir / "trace.bin",
                            mlir=mlir,
                            events_out=mode2_emu_dir / "trace.events.json",
                            cycles_out=mode2_emu_dir / "cycles.txt",
                            parse_log=mode2_emu_dir / "parse.log",
                            ctrlpkt=ctrlpkt,
                            parser_session=emu_parser,
                            trace_mode="inst_exec",
                        )
                        emu_ok = mode2_emu_res.ok
                    finally:
                        mode2_emu_session.close()

                # Manifest captured-flag is true if either side produced
                # a baseline -- the mode-2 comparator's job is to handle
                # one-sided baselines gracefully (it currently SKIPs).
                mode2_succeeded = bool(hw_ok) or bool(emu_ok)
                print(
                    f"[sweep-lockstep] mode-2 baseline: hw_ok={hw_ok} emu_ok={emu_ok}",
                    flush=True,
                )

    except Exception as e:
        # Sweep raised partway through. Record the failure but do NOT
        # re-raise here -- we need the cleanup + manifest write to land
        # before propagating. The exception is captured in sweep_error
        # for the manifest, and the original is preserved in
        # sweep_exc_obj so the trailing block can `raise ... from` it
        # and keep the original traceback discoverable.
        sweep_error = f"{type(e).__name__}: {e}"
        sweep_exc_obj = e
        print(f"[sweep-lockstep] sweep raised: {sweep_error}", flush=True)
    finally:
        if hw_session is not None:
            hw_session.close()
        if emu_session is not None:
            emu_session.close()
        if hw_parser is not None:
            hw_parser.close()
        if emu_parser is not None:
            emu_parser.close()

    # ----------------------------------------------------------------
    # Grounding-PC invariance check + manifest.
    #
    # This block sits OUTSIDE the try/finally above on purpose: the
    # spec mandates the manifest lands "even on partial failures." If
    # the invariance check itself raises, swallow it into the manifest
    # so an unrelated decode error in one batch doesn't take the whole
    # manifest down with it.
    # ----------------------------------------------------------------
    core_grounding_names = grounding_by_type.get("core", [])
    try:
        invariance = _check_grounding_pc_invariance(out_dir, core_grounding_names)
    except Exception as e:
        invariance = {
            "unsafe_for_pc_join": True,
            "reason": f"invariance check failed: {type(e).__name__}: {e}",
            "per_batch_grounding_pcs": {},
        }

    elapsed = round(time.time() - t_start, 2)
    manifest = {
        "test_name": test_name,
        "compiler": compiler,
        "mode": mode,
        "perfcnt_period": perfcnt_period,
        "tiles": [
            {"col": t.col, "row": t.row, "type": t.tile_type}
            for t in tiles
        ],
        "n_batches": len(batches),
        "n_batches_completed": completed_batches,
        "grounding": {
            "core":    grounding_by_type.get("core", []),
            "memmod":  grounding_by_type.get("memmod", []),
            "memtile": grounding_by_type.get("memtile", []),
            "shim":    grounding_by_type.get("shim", []),
        },
        "mode2_baseline_captured": (with_mode2_baseline and mode2_succeeded),
        "elapsed_sec": elapsed,
        "sweep_error": sweep_error,
        **invariance,
    }
    manifest_path = out_dir / "sweep-manifest.json"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[sweep-lockstep] manifest written to {manifest_path}", flush=True)
    except OSError as e:
        # Last-ditch: if even the manifest write fails (disk full, etc),
        # surface it loudly. Don't mask the original sweep_error.
        print(f"[sweep-lockstep] manifest WRITE FAILED: {e}", flush=True)

    # If the sweep itself raised, propagate now so the caller still sees
    # the failure even though the manifest landed. Use `raise ... from`
    # so the original traceback is preserved and the failure site (the
    # subprocess call deep in batch N) stays visible to debuggers and
    # pytest -v output.
    if sweep_error is not None:
        raise RuntimeError(
            f"sweep_lockstep failed: {sweep_error}"
        ) from sweep_exc_obj


def sweep(
    test_name: str,
    compiler: str,
    col: int,
    row: int,
    tile_type: str,
    build_dir: Path,
    out_json: Path,
    events_filter: Optional[List[str]] = None,
    run_hw: bool = True,
    run_emu: bool = True,
    ctrlpkt: Optional[Path] = None,
    ground_event_name: Optional[str] = None,
    reuse_ctx: bool = False,
) -> dict:
    """Backward-compatible single-tile entry point.

    Delegates to sweep_multi with a single TileSpec. Output is written to
    the caller-specified out_json (not to out_dir/<label>.json) so
    existing callers keep working unchanged.
    """
    tile = TileSpec(col=col, row=row, tile_type=tile_type)
    summaries = sweep_multi(
        test_name=test_name,
        compiler=compiler,
        tiles=[tile],
        build_dir=build_dir,
        out_dir=out_json.parent,
        events_filter=events_filter,
        run_hw=run_hw,
        run_emu=run_emu,
        ctrlpkt=ctrlpkt,
        ground_event_name=ground_event_name,
        reuse_ctx=reuse_ctx,
    )
    # sweep_multi wrote to out_dir/<test>.<compiler>.<tile.label>.json.
    # Rename onto the caller's requested path if they differ, and also
    # move the sibling .merged.json.
    canonical = _tile_output_path(out_json.parent, test_name, compiler, tile)
    if canonical != out_json and canonical.exists():
        if out_json.exists():
            out_json.unlink()
        canonical.rename(out_json)
        canonical_merged = canonical.with_suffix(".merged.json")
        if canonical_merged.exists():
            target_merged = out_json.with_suffix(".merged.json")
            if target_merged.exists():
                target_merged.unlink()
            canonical_merged.rename(target_merged)
    return summaries[0]



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--test", required=True, help="test name (matches npu-xrt dir)")
    ap.add_argument("--compiler", default="chess", choices=["chess", "peano"])
    # Two mutually-exclusive invocation modes:
    #   single-tile: --col/--row/--tile-type/--out FILE  (legacy)
    #   multi-tile:  --tiles "c:r:t,..." --out-dir DIR   (preferred)
    ap.add_argument("--col", type=int)
    ap.add_argument("--row", type=int)
    ap.add_argument("--tile-type",
                    choices=["core", "memmod", "memtile", "shim"])
    ap.add_argument("--out", type=Path,
                    help="single-tile output JSON (single-tile mode only)")
    ap.add_argument("--tiles",
                    help="multi-tile spec: comma list of col:row:type, all "
                         "same tile_type (e.g. 0:3:core,1:3:core,1:2:core). "
                         "Writes one JSON per tile into --out-dir.")
    ap.add_argument("--out-dir", type=Path,
                    help="multi-tile output directory (multi-tile mode only)")
    ap.add_argument("--build-dir", type=Path,
                    help="override build directory "
                         "(default: <mlir-aie>/build/test/npu-xrt/<test>/<compiler>)")
    ap.add_argument("--events",
                    help="comma-separated event name whitelist "
                         "(default: every event in the tile type)")
    ap.add_argument("--ctrlpkt", type=Path, help="optional control-packet blob to pass as --input")
    ap.add_argument("--no-hw", action="store_true", help="skip HW runs")
    ap.add_argument("--no-emu", action="store_true", help="skip EMU runs")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="parallel EMU batch workers (default 1 = serial). "
                         "When both HW and EMU run, Phase A (HW) is always "
                         "serial -- one job in flight on the NPU -- and "
                         "Phase B (EMU) then runs the same batches with -j "
                         "workers. Each batch calls RunnerSession.reset() "
                         "before dispatch, so per-batch results are "
                         "independent of which session services the batch "
                         "and identical across j values. Each worker holds "
                         "its own RunnerSession + ParseSession subprocess "
                         "pair (memory cost is N x (~50MB runner + ~200MB "
                         "parser)).")
    ap.add_argument("--ground-event",
                    help="event name to reserve in slot 0 of every batch; "
                         "its timestamp anchors other events in a merged "
                         "timeline (e.g. USER_EVENT_1). Max sweep events per "
                         "batch drops from 8 to 7 when set.")
    ap.add_argument("--reuse-ctx", action="store_true",
                    help="have the runner reuse a single hw_context across "
                         "all batches by injecting the test's init+enable "
                         "CDO blobs as an in-band preamble. Cuts per-batch "
                         "latency from ~90ms to ~10ms on Phoenix; requires "
                         "main_aie_cdo_init.bin and main_aie_cdo_enable.bin "
                         "alongside the lowered MLIR.")
    # A.2 mode-1 lockstep sweep flags. When any of these are set alongside
    # --tiles, sweep_lockstep is invoked instead of the legacy sweep_multi.
    ap.add_argument("--mode", choices=("event_time", "event_pc"),
                    default="event_time",
                    help="trace mode for compute-core trace units; "
                         "matches mlir-trace-inject's --trace-mode")
    ap.add_argument("--core-grounding",
                    default="PERF_CNT_2,INSTR_EVENT_0,INSTR_EVENT_1",
                    help="grounding events reserved in fixed slots per batch "
                         "(cores). Comma-separated event names.")
    ap.add_argument("--memmod-grounding", default="PERF_CNT_2",
                    help="grounding events for memmod trace units")
    ap.add_argument("--memtile-grounding", default="PERF_CNT_2",
                    help="grounding events for memtile trace units")
    ap.add_argument("--shim-grounding", default="PERF_CNT_2",
                    help="grounding events for shim trace units")
    ap.add_argument("--core-sweep", default="all",
                    help="comma-separated event names to sweep on cores; "
                         "'all' enumerates from the event header")
    ap.add_argument("--memmod-sweep", default="all",
                    help="comma-separated event names to sweep on memmods; "
                         "'all' enumerates from the event header")
    ap.add_argument("--memtile-sweep", default="all",
                    help="comma-separated event names to sweep on memtiles; "
                         "'all' enumerates from the event header")
    ap.add_argument("--shim-sweep", default="all",
                    help="comma-separated event names to sweep on shims; "
                         "'all' enumerates from the event header")
    ap.add_argument("--perfcnt-period", type=int, default=DEFAULT_PERFCNT_PERIOD,
                    help="reserved for sanity-checking against the xclbin's "
                         "baked-in perfcnt period value")
    ap.add_argument("--with-mode2-baseline", action="store_true", default=True,
                    help="capture one HW-only mode-2 (inst_exec) batch per "
                         "test after the mode-1 sweep (for A.2b)")
    ap.add_argument("--no-mode2-baseline", action="store_false",
                    dest="with_mode2_baseline")
    args = ap.parse_args()

    build_dir = args.build_dir or (
        MLIR_AIE_ROOT / "build" / "test" / "npu-xrt" / args.test / args.compiler
    )
    events_filter = [s.strip() for s in args.events.split(",")] if args.events else None

    if args.tiles:
        if not args.out_dir:
            ap.error("--tiles requires --out-dir")
        tiles: List[TileSpec] = []
        for spec in args.tiles.split(","):
            parts = spec.strip().split(":")
            if len(parts) != 3:
                ap.error(f"bad --tiles entry {spec!r}; want col:row:tile_type")
            try:
                tiles.append(TileSpec(
                    col=int(parts[0]), row=int(parts[1]),
                    tile_type=parts[2],
                ))
            except ValueError as e:
                ap.error(f"bad --tiles entry {spec!r}: {e}")
        # Route to sweep_lockstep when tiles are mixed-type (sweep_multi
        # explicitly rejects those) or when the caller opts in via --mode
        # event_pc (the A.2 mode-1 sweep path).
        tile_types_seen = {t.tile_type for t in tiles}
        use_lockstep = (args.mode == "event_pc") or (len(tile_types_seen) > 1)
        if use_lockstep:
            # Warn once if a legacy --ground-event was passed alongside the
            # lockstep path. The lockstep flow uses --core-grounding /
            # --memmod-grounding / etc instead, so --ground-event would
            # silently do nothing.
            if args.ground_event:
                print(
                    f"[sweep-lockstep] warning: --ground-event "
                    f"{args.ground_event!r} ignored in lockstep mode; "
                    f"use --core-grounding / --memmod-grounding / "
                    f"--memtile-grounding / --shim-grounding instead",
                    file=sys.stderr,
                )
            sweep_lockstep(
                test_name=args.test, compiler=args.compiler,
                tiles=tiles, build_dir=build_dir, out_dir=args.out_dir,
                run_hw=not args.no_hw, run_emu=not args.no_emu,
                ctrlpkt=args.ctrlpkt,
                mode=args.mode,
                core_grounding=[s.strip() for s in args.core_grounding.split(",")],
                memmod_grounding=[s.strip() for s in args.memmod_grounding.split(",")],
                memtile_grounding=[s.strip() for s in args.memtile_grounding.split(",")],
                shim_grounding=[s.strip() for s in args.shim_grounding.split(",")],
                core_sweep=None if args.core_sweep == "all" else [s.strip() for s in args.core_sweep.split(",")],
                memmod_sweep=None if args.memmod_sweep == "all" else [s.strip() for s in args.memmod_sweep.split(",")],
                memtile_sweep=None if args.memtile_sweep == "all" else [s.strip() for s in args.memtile_sweep.split(",")],
                shim_sweep=None if args.shim_sweep == "all" else [s.strip() for s in args.shim_sweep.split(",")],
                perfcnt_period=args.perfcnt_period,
                with_mode2_baseline=args.with_mode2_baseline,
                reuse_ctx=args.reuse_ctx,
                jobs=args.jobs,
            )
            print(f"[sweep-lockstep] manifest: {args.out_dir}/sweep-manifest.json")
        else:
            # Symmetric to the lockstep path's --ground-event warning:
            # if a user passes lockstep-only flags (--core-grounding,
            # --core-sweep, etc.) but the routing landed on sweep_multi
            # because --mode is event_time and tiles are same-type, the
            # flags are silently ignored. Warn once so the typo is caught.
            _lockstep_only_set = (
                args.core_grounding != "PERF_CNT_2,INSTR_EVENT_0,INSTR_EVENT_1"
                or args.memmod_grounding != "PERF_CNT_2"
                or args.memtile_grounding != "PERF_CNT_2"
                or args.shim_grounding != "PERF_CNT_2"
                or args.core_sweep != "all"
                or args.memmod_sweep != "all"
                or args.memtile_sweep != "all"
                or args.shim_sweep != "all"
            )
            if _lockstep_only_set:
                print(
                    "[sweep-multi] warning: lockstep-only flags "
                    "(--core-grounding / --memmod-grounding / "
                    "--memtile-grounding / --shim-grounding / "
                    "--core-sweep / --memmod-sweep / --memtile-sweep / "
                    "--shim-sweep) ignored on the sweep_multi path; "
                    "pass --mode event_pc or use mixed-type --tiles to "
                    "route to the lockstep sweep, or use --ground-event "
                    "for sweep_multi grounding",
                    file=sys.stderr,
                )
            summaries = sweep_multi(
                test_name=args.test, compiler=args.compiler,
                tiles=tiles, build_dir=build_dir, out_dir=args.out_dir,
                events_filter=events_filter,
                run_hw=not args.no_hw, run_emu=not args.no_emu,
                ctrlpkt=args.ctrlpkt,
                ground_event_name=args.ground_event,
                reuse_ctx=args.reuse_ctx,
                jobs=args.jobs,
            )
            for s in summaries:
                tile = s["tile"]
                print(f"[sweep-multi] wrote {args.out_dir}/"
                      f"{args.test}.{args.compiler}.{tile['type']}_c{tile['col']}r{tile['row']}.json "
                      f"({len(s['events'])} events, {s['elapsed_sec']}s)")
        return 0

    # Legacy single-tile path
    if args.col is None or args.row is None or args.tile_type is None or args.out is None:
        ap.error("single-tile mode requires --col, --row, --tile-type, --out "
                 "(or use --tiles/--out-dir)")
    summary = sweep(
        test_name=args.test,
        compiler=args.compiler,
        col=args.col,
        row=args.row,
        tile_type=args.tile_type,
        build_dir=build_dir,
        out_json=args.out,
        events_filter=events_filter,
        run_hw=not args.no_hw,
        run_emu=not args.no_emu,
        ctrlpkt=args.ctrlpkt,
        ground_event_name=args.ground_event,
        reuse_ctx=args.reuse_ctx,
    )
    print(f"[sweep] wrote {args.out} ({len(summary['events'])} events, "
          f"{summary['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
