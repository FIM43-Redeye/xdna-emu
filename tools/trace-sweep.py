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


class RunnerSession:
    """Long-lived bridge-trace-runner process in --batch-stdin mode.

    Holds a single XRT device + xclbin across many patched-instr runs
    so per-launch overhead drops from ~228 ms (fresh process) to
    ~90 ms (shared device, fresh hw_context per run). The hw_context
    is rebuilt per launch inside the runner -- see the runner's
    ``reuse_context_across_runs`` docstring for the reason the outer
    process can't safely hold a single hw_context across runs yet.

    Use as a context manager or call close() explicitly. If the
    subprocess dies or blocks, run_one() raises RuntimeError so the
    caller can restart the session rather than silently hang.
    """

    def __init__(self, xclbin: Path, runner_env: Dict[str, str],
                 side: str, stderr_log: Path, verbose: bool = False,
                 cdo_preambles: Optional[List[Path]] = None,
                 trace_buf_idx: Optional[int] = None,
                 reuse_ctx: bool = False):
        self.side = side
        self.stderr_log = stderr_log
        self._stderr_fh = stderr_log.open("w")
        # Per-call CLI fragments common to every run on this session.
        # Folded into run_one() rather than the outer cmd because the
        # runner's batch-stdin protocol re-parses every line through the
        # same parser, so they must live on the per-line side.
        self._cdo_preambles = list(cdo_preambles or [])
        self._trace_buf_idx = trace_buf_idx
        cmd = [str(RUNNER), "--batch-stdin", "--xclbin", str(xclbin)]
        if verbose:
            cmd.append("-v")
        env = os.environ.copy()
        env.update(runner_env)
        # Reuse-ctx mode enables BRIDGE_RUNNER_REUSE_CONTEXT in the
        # subprocess. Combined with --cdo-preamble that re-applies the
        # init/enable CDOs each launch, this lets the runner skip
        # hw_context teardown between runs (saving ~90 ms per launch on
        # Phoenix). Without the preamble, this mode hits the alternating
        # state=8/state=6 timeout pattern.
        if reuse_ctx:
            env["BRIDGE_RUNNER_REUSE_CONTEXT"] = "1"
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            env=env,
            text=True,
            bufsize=1,
        )
        ready_line = self.proc.stdout.readline().strip()
        if not ready_line:
            self.close()
            raise RuntimeError(f"{side} runner died before ready")
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as e:
            self.close()
            raise RuntimeError(
                f"{side} runner first line not JSON: {ready_line!r}") from e
        if ready.get("event") != "ready":
            self.close()
            raise RuntimeError(
                f"{side} runner first line not 'ready': {ready_line!r}")

    def run_one(
        self,
        instr: Path,
        trace_out: Path,
        inputs: Optional[List[Path]] = None,
        outputs: Optional[List[Path]] = None,
        ctrlpkts: Optional[List[Path]] = None,
        trace_size: int = 1 << 20,
    ) -> dict:
        """Dispatch one run. Returns the parsed JSON status dict.

        The status dict has keys {"ok", "trace_out", "elapsed_ms"} on
        success and an additional "error" string on failure; parse
        errors (invalid CLI line) surface as ok=false with a "parse:"
        error prefix.
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"{self.side} runner has exited")
        parts = [
            "--instr", str(instr),
            "--trace-out", str(trace_out),
            "--trace-size", str(trace_size),
        ]
        for p in (inputs or []):
            parts += ["--input", str(p)]
        for p in (outputs or []):
            parts += ["--output", str(p)]
        for p in (ctrlpkts or []):
            parts += ["--ctrlpkt", str(p)]
        for p in self._cdo_preambles:
            parts += ["--cdo-preamble", str(p)]
        if self._trace_buf_idx is not None:
            parts += ["--trace-buf-idx", str(self._trace_buf_idx)]
        # Our argument values never contain spaces in this codebase,
        # but quote paths defensively so a future path with spaces
        # doesn't silently corrupt tokenisation on the C++ side.
        def quote(s: str) -> str:
            return f'"{s}"' if (" " in s or "\t" in s) else s
        line = " ".join(quote(p) for p in parts)
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError(f"{self.side} runner stdin closed") from e
        resp = self.proc.stdout.readline()
        if not resp:
            raise RuntimeError(f"{self.side} runner produced no response")
        try:
            return json.loads(resp)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{self.side} runner non-JSON response: {resp!r}") from e

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if self._stderr_fh is not None and not self._stderr_fh.closed:
            self._stderr_fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class ParseSession:
    """Long-lived parse-trace.py decode server.

    Why this exists: parse-trace.py imports mlir-aie + numpy, which
    costs ~620 ms per Python startup. A 32-batch sweep that decodes
    once per side per batch pays that cost 32-64 times -- about 20-40
    seconds purely on imports, or 75% of the total sweep wall clock.
    Spawning one decoder process per sweep amortizes the import to a
    single ~430 ms startup, dropping subsequent decodes to ~100 ms each
    (~6x per-decode speedup).

    Protocol mirrors RunnerSession: the subprocess prints a "ready"
    event on startup, then accepts one JSON request per stdin line and
    emits one JSON response per stdout line.
    """

    def __init__(self, side: str, stderr_log: Path,
                 env_for_parse: Optional[Dict[str, str]] = None):
        self.side = side
        self.stderr_log = stderr_log
        self._stderr_fh = stderr_log.open("w")
        env = os.environ.copy()
        if env_for_parse:
            env.update(env_for_parse)
        # Always inject the mlir-aie install path -- the same way
        # _parse_trace_bin does in fallback mode -- so we don't depend
        # on the caller having activated ironenv.
        env["PYTHONPATH"] = (
            str(MLIR_AIE_ROOT / "install" / "python")
            + os.pathsep + env.get("PYTHONPATH", "")
        ).rstrip(os.pathsep)
        cmd = [sys.executable, str(PARSE_TOOL), "--server"]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            env=env,
            text=True,
            bufsize=1,
        )
        ready_line = self.proc.stdout.readline().strip()
        if not ready_line:
            self.close()
            raise RuntimeError(f"{side} parser died before ready "
                               f"(see {stderr_log})")
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as e:
            self.close()
            raise RuntimeError(
                f"{side} parser first line not JSON: {ready_line!r}") from e
        if ready.get("event") != "ready":
            self.close()
            raise RuntimeError(
                f"{side} parser first line not 'ready': {ready_line!r}")

    def parse_one(
        self,
        trace_bin: Path,
        xclbin_mlir: Path,
        out_events: Optional[Path] = None,
        out_cycles: Optional[Path] = None,
        out_perfetto: Optional[Path] = None,
        out_commands: Optional[Path] = None,
    ) -> dict:
        """Send one decode request, return parsed response dict.

        The response shape is what parse-trace.py's server_loop emits:
            {"ok": True, "events_count": N, "cycles": <span or None>,
             "empty": <bool>, "elapsed_ms": M}
          or {"ok": False, "error": "..."}.
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"{self.side} parser has exited")
        req = {"trace_bin": str(trace_bin), "xclbin_mlir": str(xclbin_mlir)}
        if out_events:   req["out_events"]   = str(out_events)
        if out_cycles:   req["out_cycles"]   = str(out_cycles)
        if out_perfetto: req["out_perfetto"] = str(out_perfetto)
        if out_commands: req["out_commands"] = str(out_commands)
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, IOError) as e:
            raise RuntimeError(f"{self.side} parser write failed: {e}") from e
        resp = self.proc.stdout.readline().strip()
        if not resp:
            raise RuntimeError(f"{self.side} parser closed before response")
        try:
            return json.loads(resp)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{self.side} parser response not JSON: {resp!r}") from e

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if self._stderr_fh is not None and not self._stderr_fh.closed:
            self._stderr_fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _parse_trace_bin(
    trace_bin: Path,
    mlir: Path,
    events_out: Path,
    cycles_out: Path,
    parse_log: Path,
    env_for_parse: Dict[str, str],
    parser_session: Optional[ParseSession] = None,
) -> Tuple[bool, Optional[str], Optional[int], Optional[int]]:
    """Run parse-trace on a trace binary. Returns (ok, error,
    cycles, events_count). ok=True with cycles=0/events_count=0 means
    the trace parsed as empty (kernel ran, no events fired); that's
    not a sweep failure.

    When ``parser_session`` is provided, the decode is dispatched
    through the long-lived parse-trace --server subprocess, avoiding
    a fresh Python startup (~620 ms -> ~100 ms per call). When None,
    falls back to the old subprocess.run() path so this helper still
    works outside a session-managed sweep.
    """
    if parser_session is not None:
        try:
            resp = parser_session.parse_one(
                trace_bin=trace_bin,
                xclbin_mlir=mlir,
                out_events=events_out,
                out_cycles=cycles_out,
            )
        except RuntimeError as e:
            return False, f"parse-trace session: {e}", None, None
        if not resp.get("ok"):
            err = resp.get("error", "unknown")
            # The server reports "empty" via flags rather than a
            # special error message, but errno-style failures still
            # arrive as ok=false. Treat empty-trace exactly the same
            # way the fallback path does.
            return False, f"parse-trace: {err}", None, None
        if resp.get("empty"):
            # Server still wrote whatever output files were requested
            # (with cycles=0); make doubly sure the events file is
            # well-formed for downstream tools that didn't pass
            # out_events to the server.
            if not events_out.exists():
                events_out.write_text(
                    '{"schema_version":1,"events":[],"slot_names":{}}\n')
            return True, None, 0, 0
        return True, None, int(resp.get("cycles") or 0), \
               int(resp.get("events_count") or 0)

    # Fallback path: spawn a fresh interpreter per call. Slow but kept
    # for callers that don't want to manage a ParseSession.
    parse_cmd = [
        sys.executable, str(PARSE_TOOL),
        "--trace-bin", str(trace_bin),
        "--xclbin-mlir", str(mlir),
        "--out-events", str(events_out),
        "--out-cycles", str(cycles_out),
    ]
    parse_env = env_for_parse.copy()
    parse_env["PYTHONPATH"] = str(MLIR_AIE_ROOT / "install" / "python")
    with parse_log.open("w") as lf:
        rc = subprocess.run(parse_cmd, env=parse_env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        log_text = parse_log.read_text(errors="replace") if parse_log.exists() else ""
        empty_markers = ("no timestamped events", "empty or all zeros")
        if any(m in log_text for m in empty_markers):
            events_out.write_text('{"schema_version":1,"events":[],"slot_names":{}}\n')
            cycles_out.write_text("0\n")
            return True, None, 0, 0
        return False, f"parse-trace exit {rc}", None, None

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
    return True, None, cycles, events_count


def _run_one_side(
    side: str,                   # "HW" or "EMU"
    session: Optional["RunnerSession"],
    runner_env: Dict[str, str],  # used to build parse_env (not for the runner)
    instr: Path,
    trace_bin: Path,
    mlir: Path,
    events_out: Path,
    cycles_out: Path,
    parse_log: Path,
    ctrlpkt: Optional[Path],
    parser_session: Optional["ParseSession"] = None,
) -> RunResult:
    """One (run → parse) cycle.

    The runner session is shared across all batches of a sweep; this
    function just dispatches a single run_one() and then parses. The
    trace_bin is written by the runner and read back by parse-trace.

    Failures are recorded in the RunResult, not raised -- a single
    batch failure must not kill the sweep.
    """
    if session is None:
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} session is not open")
    try:
        status = session.run_one(
            instr=instr,
            trace_out=trace_bin,
            ctrlpkts=[ctrlpkt] if (ctrlpkt and ctrlpkt.is_file()) else None,
            trace_size=1 << 20,
        )
    except RuntimeError as e:
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} runner session: {e}")
    if not status.get("ok", False):
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} runner: {status.get('error', 'unknown')}")

    env_for_parse = os.environ.copy()
    env_for_parse.update(runner_env)
    ok, err, cycles, events_count = _parse_trace_bin(
        trace_bin, mlir, events_out, cycles_out, parse_log, env_for_parse,
        parser_session=parser_session,
    )
    if not ok:
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} {err}")
    return RunResult(ok=True, cycles=cycles, events_count=events_count)


# ---------------------------------------------------------------------------
# Top-level sweep
# ---------------------------------------------------------------------------

_INSTS_HEADER_BYTES = 16
_INSTS_OPCODE_WRITE32   = 0x00
_INSTS_OPCODE_BLOCKWRITE = 0x01
_INSTS_OPCODE_MASKWRITE = 0x03
_INSTS_OPCODE_DDR_PATCH = 0x81


def _discover_trace_buf_idx(insts: Path) -> Optional[int]:
    """Compute the trace BO's 0-indexed position among data buffers.

    Method: walk insts.bin's DdrPatch ops and take the max ``arg_idx``.
    DdrPatch ops fill in BO addresses for shim-DMA BDs at runtime, with
    one patch per data buffer the kernel uses. Trace is added last by
    every build flow we support (mlir-aie's --with-hw-cycles puts it at
    arg_idx=N+1 where N is the user-buffer count; the trace-inject flow
    in xdna-emu's traced-tests adds it as the next runtime_sequence
    arg). In both cases the trace BO ends up with the highest arg_idx
    among DdrPatch targets, so taking the max is robust.

    Returns None if the file isn't a recognisable insts.bin or has no
    DdrPatch ops -- in that case the caller falls back to the runner's
    legacy "last buffer kernarg = trace" heuristic.

    DdrPatch layout (from src/npu/parser.rs lines 313-358):
      48-byte instruction; payload offsets 16/24/32 hold reg_addr,
      arg_idx (one byte at offset 24), and arg_plus respectively.
    """
    try:
        data = insts.read_bytes()
    except OSError:
        return None
    if len(data) < _INSTS_HEADER_BYTES:
        return None
    magic = int.from_bytes(data[0:4], "little")
    if magic != 0x06030100:
        return None
    total_size = int.from_bytes(data[12:16], "little")
    end = min(len(data), total_size)
    off = _INSTS_HEADER_BYTES
    max_arg_idx: Optional[int] = None
    while off + 4 <= end:
        opcode = data[off] & 0xFF
        if opcode == _INSTS_OPCODE_WRITE32:
            size = 24
        elif opcode == _INSTS_OPCODE_BLOCKWRITE:
            if off + 16 > end:
                break
            size = int.from_bytes(data[off + 12:off + 16], "little")
        elif opcode == _INSTS_OPCODE_MASKWRITE:
            size = 28
        elif opcode == _INSTS_OPCODE_DDR_PATCH:
            size = 48
            if off + 36 <= end:
                arg_idx = data[off + 8 + 24]   # payload[24] inside the op
                if max_arg_idx is None or arg_idx > max_arg_idx:
                    max_arg_idx = arg_idx
        else:
            # Unknown opcode -- stop walking; any answer we'd derive
            # past this point is unreliable.
            break
        off += size
    return max_arg_idx


def _find_cdo_preambles(mlir: Path) -> List[Path]:
    """Locate main_aie_cdo_init.bin + main_aie_cdo_enable.bin alongside
    the lowered MLIR. They live in the same .mlir.prj/ directory the
    aiecc.py compile produced. Returns [] if either is missing -- the
    caller treats that as "no preamble injection available."
    """
    prj = mlir.parent
    init = prj / "main_aie_cdo_init.bin"
    enable = prj / "main_aie_cdo_enable.bin"
    if init.is_file() and enable.is_file():
        return [init, enable]
    return []


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
    reuse_ctx: bool = False,
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
            session=hw_session,
            runner_env=hw_runner_env,
            instr=patched,
            trace_bin=work_dir / f"b{b_idx}.trace_hw.bin",
            mlir=mlir,
            events_out=hw_events_out,
            cycles_out=work_dir / f"b{b_idx}.hw.cycles.txt",
            parse_log=work_dir / f"b{b_idx}.hw.parse.log",
            ctrlpkt=ctrlpkt,
            parser_session=hw_parser,
        )

        emu_res = RunResult(ok=False, cycles=None, events_count=None,
                            error="emu skipped") if not run_emu else _run_one_side(
            side="EMU",
            session=emu_session,
            runner_env=emu_runner_env,
            instr=patched,
            trace_bin=work_dir / f"b{b_idx}.trace_emu.bin",
            mlir=mlir,
            events_out=emu_events_out,
            cycles_out=work_dir / f"b{b_idx}.emu.cycles.txt",
            parse_log=work_dir / f"b{b_idx}.emu.parse.log",
            ctrlpkt=ctrlpkt,
            parser_session=emu_parser,
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


_GROUNDING_BY_TILE_TYPE = {
    "core":    "PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1",
    "memmod":  "PERF_CNT_0",
    "memtile": "PERF_CNT_0",
    "shim":    "PERF_CNT_0",
}

_MODE_INT = {
    "event_time": 0,
    "event_pc":   1,
    "inst_exec":  2,
}


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
    perfcnt_period: int = 1024,
    with_mode2_baseline: bool = True,
    reuse_ctx: bool = False,
) -> None:
    """Mode-1 lockstep sweep across mixed-type tiles (A.2 path).

    Unlike sweep_multi, tiles may have different tile_types (cores,
    memmods, memtiles, shims). Per-type grounding events are reserved
    in fixed slots. Per-type sweep event lists advance in lockstep
    using _build_lockstep_batches.

    Each batch applies all tile patches in a single --multi-tile
    invocation, then runs HW + EMU together. After the sweep, an
    optional HW-only mode-2 (inst_exec) baseline batch is captured.
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
        if sweep_filter is not None:
            sweep_names = [n for n in ev_names_all if n in sweep_filter]
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
    mode2_succeeded = False

    try:
        for b_idx, batch_assignment in enumerate(batches):
            batch_dir = out_dir / f"batch_{b_idx:02d}"
            hw_dir = batch_dir / "hw"
            emu_dir = batch_dir / "emu"
            hw_dir.mkdir(parents=True, exist_ok=True)
            emu_dir.mkdir(parents=True, exist_ok=True)

            # Build multi-tile patch spec and apply it in one subprocess call.
            patch_spec = _build_lockstep_patch_spec(
                tiles, batch_assignment, grounding_by_type,
                all_events_by_type, mode,
            )
            spec_path = work_dir / f"b{b_idx:02d}.patch.json"
            spec_path.write_text(json.dumps(patch_spec, indent=2))
            subprocess.run([
                sys.executable, str(PATCH_TOOL), str(insts),
                "--multi-tile", str(spec_path),
                "--output", str(patched),
            ], check=True, capture_output=True)

            hw_events_out = hw_dir / "trace.events.json"
            emu_events_out = emu_dir / "trace.events.json"

            hw_res = (
                RunResult(ok=False, cycles=None, events_count=None, error="hw skipped")
                if not run_hw else _run_one_side(
                    side="HW",
                    session=hw_session,
                    runner_env=hw_runner_env,
                    instr=patched,
                    trace_bin=hw_dir / "trace.bin",
                    mlir=mlir,
                    events_out=hw_events_out,
                    cycles_out=hw_dir / "cycles.txt",
                    parse_log=hw_dir / "parse.log",
                    ctrlpkt=ctrlpkt,
                    parser_session=hw_parser,
                )
            )
            emu_res = (
                RunResult(ok=False, cycles=None, events_count=None, error="emu skipped")
                if not run_emu else _run_one_side(
                    side="EMU",
                    session=emu_session,
                    runner_env=emu_runner_env,
                    instr=patched,
                    trace_bin=emu_dir / "trace.bin",
                    mlir=mlir,
                    events_out=emu_events_out,
                    cycles_out=emu_dir / "cycles.txt",
                    parse_log=emu_dir / "parse.log",
                    ctrlpkt=ctrlpkt,
                    parser_session=emu_parser,
                )
            )
            print(
                f"[sweep-lockstep] batch {b_idx + 1}/{len(batches)}: "
                f"hw_ok={hw_res.ok} emu_ok={emu_res.ok} "
                f"hw_cyc={hw_res.cycles} emu_cyc={emu_res.cycles}",
                flush=True,
            )

        # ----------------------------------------------------------------
        # Mode-2 finishing batch (HW only, inst_exec).
        # ----------------------------------------------------------------
        if with_mode2_baseline and run_hw and hw_session is not None:
            compute_tiles = [t for t in tiles if t.tile_type == "core"]
            if compute_tiles:
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
                mode2_hw_dir = mode2_dir / "hw"
                mode2_hw_dir.mkdir(parents=True, exist_ok=True)
                mode2_trace_bin = mode2_hw_dir / "trace.bin"
                mode2_events_out = mode2_hw_dir / "trace.events.json"
                mode2_cycles_out = mode2_hw_dir / "cycles.txt"

                mode2_res = _run_one_side(
                    side="HW",
                    session=hw_session,
                    runner_env=hw_runner_env,
                    instr=mode2_insts,
                    trace_bin=mode2_trace_bin,
                    mlir=mlir,
                    events_out=mode2_events_out,
                    cycles_out=mode2_cycles_out,
                    parse_log=mode2_hw_dir / "parse.log",
                    ctrlpkt=ctrlpkt,
                    parser_session=hw_parser,
                )
                mode2_succeeded = mode2_res.ok
                print(
                    f"[sweep-lockstep] mode-2 baseline: ok={mode2_res.ok} "
                    f"cyc={mode2_res.cycles}",
                    flush=True,
                )

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
    # ----------------------------------------------------------------
    core_grounding_names = grounding_by_type.get("core", [])
    invariance = _check_grounding_pc_invariance(out_dir, core_grounding_names)

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
        "grounding": {
            "core":    grounding_by_type.get("core", []),
            "memmod":  grounding_by_type.get("memmod", []),
            "memtile": grounding_by_type.get("memtile", []),
            "shim":    grounding_by_type.get("shim", []),
        },
        "mode2_baseline_captured": (with_mode2_baseline and mode2_succeeded),
        "elapsed_sec": elapsed,
        **invariance,
    }
    manifest_path = out_dir / "sweep-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[sweep-lockstep] manifest written to {manifest_path}", flush=True)


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
                    default="PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1",
                    help="grounding events reserved in fixed slots per batch "
                         "(cores). Comma-separated event names.")
    ap.add_argument("--memmod-grounding", default="PERF_CNT_0",
                    help="grounding events for memmod trace units")
    ap.add_argument("--memtile-grounding", default="PERF_CNT_0",
                    help="grounding events for memtile trace units")
    ap.add_argument("--shim-grounding", default="PERF_CNT_0",
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
    ap.add_argument("--perfcnt-period", type=int, default=1024,
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
            )
            print(f"[sweep-lockstep] manifest: {args.out_dir}/sweep-manifest.json")
        else:
            summaries = sweep_multi(
                test_name=args.test, compiler=args.compiler,
                tiles=tiles, build_dir=build_dir, out_dir=args.out_dir,
                events_filter=events_filter,
                run_hw=not args.no_hw, run_emu=not args.no_emu,
                ctrlpkt=args.ctrlpkt,
                ground_event_name=args.ground_event,
                reuse_ctx=args.reuse_ctx,
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
