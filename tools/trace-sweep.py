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
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MLIR_AIE_ROOT = REPO_ROOT.parent / "mlir-aie"
EVENTS_HEADER = (
    MLIR_AIE_ROOT / "build" / "include" / "xaienginecdo_static"
    / "xaiengine" / "xaie_events_aieml.h"
)
RUNNER = REPO_ROOT / "bridge-runner" / "build" / "bridge-trace-runner"
PATCH_TOOL = REPO_ROOT / "tools" / "trace-patch-events.py"
PARSE_TOOL = REPO_ROOT / "tools" / "parse-trace.py"

# xaie_events_aieml.h uses module prefixes that map to our tile types.
_MOD_TO_TILE_TYPE = {
    "CORE": "core",
    "MEM": "memmod",
    "MEM_TILE": "memtile",
    "PL": "shim",
}
_TILE_TYPE_TO_MOD = {v: k for k, v in _MOD_TO_TILE_TYPE.items()}


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
# Runner wrappers
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    ok: bool
    cycles: Optional[int]
    events_count: Optional[int]
    per_event_count: Dict[str, int] = None  # populated by _relabel_events
    error: Optional[str] = None

    def __post_init__(self):
        if self.per_event_count is None:
            self.per_event_count = {}


def _run_patch(
    original_insts: Path,
    patched_insts: Path,
    col: int,
    row: int,
    tile_type: str,
    event_ids: List[int],
) -> None:
    spec = ",".join(str(e) for e in event_ids)
    cmd = [
        sys.executable, str(PATCH_TOOL), str(original_insts),
        "--col", str(col), "--row", str(row), "--tile-type", tile_type,
        "--events", spec, "--output", str(patched_insts),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _run_patch_multi(
    original_insts: Path,
    patched_insts: Path,
    patches: List[Tuple[int, int, str, List[int]]],
) -> None:
    """Apply multiple tile patches to one insts.bin in sequence.

    Each entry is (col, row, tile_type, event_ids). The patches are
    independent (they target disjoint Trace_Event registers because the
    NPU address encoding includes (col, row)), so chaining them -- read
    original, patch tile A, patch tile B over the intermediate, write
    final -- produces the same bytes as a single atomic multi-tile
    patch would.
    """
    if not patches:
        # Copy through unchanged -- downstream runner still needs the file.
        patched_insts.write_bytes(original_insts.read_bytes())
        return
    # Start from the original; chain patches through a scratch file so we
    # can invoke the existing single-tile patcher subprocess unchanged.
    scratch = patched_insts.parent / f"{patched_insts.name}.scratch"
    src = original_insts
    for i, (col, row, tile_type, event_ids) in enumerate(patches):
        dst = patched_insts if i == len(patches) - 1 else scratch
        _run_patch(src, dst, col, row, tile_type, event_ids)
        src = dst
    if scratch.exists():
        scratch.unlink()


def _relabel_events(
    events_json: Path,
    col: int, row: int,
    tile_type: str,
    batch: List["EventDef"],
) -> Tuple[Dict[str, int], List[Dict]]:
    """Rewrite slot_names and per-event `name` fields in an events.json so
    they reflect what the patched insts.bin actually programmed, and
    return (per-event fire counts, filtered event list for this tile).

    parse-trace.py reads slot names from the original (pre-patch) MLIR --
    the trace unit itself only records slot indices, so the raw data is
    already correct. This function fixes only the labelling layer so
    downstream consumers see the right event names.

    The returned filtered list is the subset of events whose (row,
    pkt_type) match this tile, with the `name` field overwritten to the
    current batch's label. It's ready to feed into _anchor_events.
    """
    if not events_json.exists():
        return {}, []
    try:
        doc = json.loads(events_json.read_text())
    except Exception:
        return {}, []
    # tile_type -> key in slot_names dict. parse-trace uses "mem" for
    # memmod; everything else matches our naming.
    slot_key = "mem" if tile_type == "memmod" else tile_type
    names = [e.name for e in batch] + [""] * (8 - len(batch))
    if "slot_names" in doc:
        doc["slot_names"].setdefault(slot_key, [""] * 8)
        doc["slot_names"][slot_key] = names

    # Column-axis caveat: HW trace records absolute columns (the runtime
    # allocator picks a start_col at launch), while EMU uses relative
    # columns starting at 0. Filtering strictly by col would drop every HW
    # event when start_col != 0. Since the sweep only enables one tile at
    # a time, we filter by (row, expected packet type) and let the slot
    # index carry identity. This works as long as stray events from
    # adjacent tiles don't share the same (row, slot) -- which they
    # wouldn't, because adjacent tiles aren't routed into this trace BO.
    pkt_for_tile = 1 if tile_type == "memmod" else 0
    per_slot: Dict[int, int] = {}
    filtered: List[Dict] = []
    for ev in doc.get("events", []):
        if ev.get("row") != row:
            continue
        if ev.get("pkt_type") != pkt_for_tile:
            continue
        slot = ev.get("slot")
        if slot is None or slot >= len(names) or not names[slot]:
            continue
        ev["name"] = names[slot]
        per_slot[slot] = per_slot.get(slot, 0) + 1
        filtered.append(ev)

    events_json.write_text(json.dumps(doc, indent=2) + "\n")
    # Map slot -> event name
    counts = {names[s]: n for s, n in per_slot.items() if names[s]}
    return counts, filtered


def _run_one_side(
    side: str,                   # "HW" or "EMU"
    runner_env: Dict[str, str],
    xclbin: Path,
    instr: Path,
    trace_bin: Path,
    mlir: Path,
    events_out: Path,
    cycles_out: Path,
    runner_log: Path,
    parse_log: Path,
    ctrlpkt: Optional[Path],
) -> RunResult:
    """One (patch → run → parse) cycle. Returns a RunResult; failures are
    recorded, not raised -- a single-event failure must not kill the sweep.
    """
    cmd = [
        str(RUNNER),
        "--xclbin", str(xclbin),
        "--instr", str(instr),
        "--trace-out", str(trace_bin),
        # 1 MiB trace BO. The 8 KiB default silently truncated lock-heavy
        # kernels halfway through, making EMU undercount events that fired
        # fine but didn't fit. See trace-sweep-v2 validation notes / #138.
        "--trace-size", "1048576",
    ]
    if ctrlpkt and ctrlpkt.is_file():
        cmd += ["--input", str(ctrlpkt)]

    env = os.environ.copy()
    env.update(runner_env)
    with runner_log.open("w") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} runner exit {rc}")

    parse_cmd = [
        sys.executable, str(PARSE_TOOL),
        "--trace-bin", str(trace_bin),
        "--xclbin-mlir", str(mlir),
        "--out-events", str(events_out),
        "--out-cycles", str(cycles_out),
    ]
    parse_env = env.copy()
    parse_env["PYTHONPATH"] = str(MLIR_AIE_ROOT / "install" / "python")
    with parse_log.open("w") as lf:
        rc = subprocess.run(parse_cmd, env=parse_env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        # Multiple parse-trace error shapes all mean "kernel ran but the
        # selected event set never fired" -- not a failure, just empty
        # data. Record it as ok with zero events so the sweep correctly
        # attributes "0 fires" to every event in the batch.
        #   "no timestamped events"            -- trace had bytes but no
        #                                         event records
        #   "empty or all zeros" (from mlir-aie) -- trace BO entirely zero
        log_text = parse_log.read_text(errors="replace") if parse_log.exists() else ""
        empty_markers = ("no timestamped events", "empty or all zeros")
        if any(m in log_text for m in empty_markers):
            events_out.write_text('{"schema_version":1,"events":[],"slot_names":{}}\n')
            cycles_out.write_text("0\n")
            return RunResult(ok=True, cycles=0, events_count=0)
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} parse-trace exit {rc}")

    try:
        cycles = int(cycles_out.read_text().strip() or "0")
    except ValueError:
        cycles = 0
    events_count = 0
    if events_out.exists():
        try:
            events_count = len(json.loads(events_out.read_text()))
        except Exception:
            events_count = 0
    return RunResult(ok=True, cycles=cycles, events_count=events_count)


# ---------------------------------------------------------------------------
# Top-level sweep
# ---------------------------------------------------------------------------

def _find_post_lowering_mlir(build_dir: Path) -> Optional[Path]:
    """Locate the aiecc-lowered MLIR (input_with_addresses.mlir) inside
    build_dir/*.prj/. Same discovery shape as emu-bridge-test.sh so parse
    behaves identically.
    """
    for prj in build_dir.glob("*.mlir.prj"):
        cand = prj / "input_with_addresses.mlir"
        if cand.exists():
            return cand
    # Fallback: one-level-deep search for robustness against future naming.
    for cand in build_dir.glob("*/input_with_addresses.mlir"):
        return cand
    return None


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

    t_start = time.time()
    for b_idx, batch in enumerate(batches):
        event_ids = [e.id for e in batch]
        # Program the same event set on every tile in lockstep. Different
        # tile types would need different event IDs, but we enforced
        # same-tile-type above; same type => same IDs work across tiles.
        patches = [
            (t.col, t.row, t.tile_type, event_ids) for t in tiles
        ]
        _run_patch_multi(insts, patched, patches)

        hw_events_out = work_dir / f"b{b_idx}.hw.events.json"
        emu_events_out = work_dir / f"b{b_idx}.emu.events.json"

        hw_res = RunResult(ok=False, cycles=None, events_count=None,
                           error="hw skipped") if not run_hw else _run_one_side(
            side="HW",
            runner_env={},
            xclbin=xclbin,
            instr=patched,
            trace_bin=work_dir / f"b{b_idx}.trace_hw.bin",
            mlir=mlir,
            events_out=hw_events_out,
            cycles_out=work_dir / f"b{b_idx}.hw.cycles.txt",
            runner_log=work_dir / f"b{b_idx}.hw.runner.log",
            parse_log=work_dir / f"b{b_idx}.hw.parse.log",
            ctrlpkt=ctrlpkt,
        )

        emu_res = RunResult(ok=False, cycles=None, events_count=None,
                            error="emu skipped") if not run_emu else _run_one_side(
            side="EMU",
            runner_env={
                "XDNA_EMU": os.environ.get("XDNA_EMU", "debug"),
                "XDNA_EMU_LOG_LEVEL": os.environ.get("XDNA_EMU_LOG_LEVEL", "info"),
                "XRT_DEVICE_BDF": "ffff:ff:1f.0",
            },
            xclbin=xclbin,
            instr=patched,
            trace_bin=work_dir / f"b{b_idx}.trace_emu.bin",
            mlir=mlir,
            events_out=emu_events_out,
            cycles_out=work_dir / f"b{b_idx}.emu.cycles.txt",
            runner_log=work_dir / f"b{b_idx}.emu.runner.log",
            parse_log=work_dir / f"b{b_idx}.emu.parse.log",
            ctrlpkt=ctrlpkt,
        )

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
        print(f"[sweep-multi] batch {b_idx + 1}/{len(batches)} on "
              f"{len(tiles)} tile(s): events "
              f"{[e.name for e in batch]} "
              f"hw_cyc={hw_res.cycles} emu_cyc={emu_res.cycles}",
              flush=True)

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
    ap.add_argument("--ground-event",
                    help="event name to reserve in slot 0 of every batch; "
                         "its timestamp anchors other events in a merged "
                         "timeline (e.g. USER_EVENT_1). Max sweep events per "
                         "batch drops from 8 to 7 when set.")
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
        summaries = sweep_multi(
            test_name=args.test, compiler=args.compiler,
            tiles=tiles, build_dir=build_dir, out_dir=args.out_dir,
            events_filter=events_filter,
            run_hw=not args.no_hw, run_emu=not args.no_emu,
            ctrlpkt=args.ctrlpkt,
            ground_event_name=args.ground_event,
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
    )
    print(f"[sweep] wrote {args.out} ({len(summary['events'])} events, "
          f"{summary['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
