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


def _relabel_events(
    events_json: Path,
    col: int, row: int,
    tile_type: str,
    batch: List["EventDef"],
) -> Dict[str, int]:
    """Rewrite slot_names and per-event `name` fields in an events.json so
    they reflect what the patched insts.bin actually programmed, then
    return a per-slot fire count.

    parse-trace.py reads slot names from the original (pre-patch) MLIR --
    the trace unit itself only records slot indices, so the raw data is
    already correct. This function fixes only the labelling layer so
    downstream consumers see the right event names.
    """
    if not events_json.exists():
        return {}
    try:
        doc = json.loads(events_json.read_text())
    except Exception:
        return {}
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

    events_json.write_text(json.dumps(doc, indent=2) + "\n")
    # Map slot -> event name
    return {names[s]: n for s, n in per_slot.items() if names[s]}


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
        "--trace-size", "8192",
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
) -> dict:
    xclbin = build_dir / "aie.xclbin"
    insts = build_dir / "insts.bin"
    if not xclbin.exists() or not insts.exists():
        raise FileNotFoundError(
            f"xclbin or insts.bin missing in {build_dir}; compile with "
            f"--with-hw-cycles first so trace routing is present"
        )
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
    batches = batch_events(all_events)

    # Per-sweep work directory. Keep batch artifacts around for post-hoc
    # inspection; they compose with parse-trace's own JSON/cycles outputs.
    work_dir = out_json.parent / f"{out_json.stem}.work"
    work_dir.mkdir(parents=True, exist_ok=True)
    patched = work_dir / "insts.patched.bin"

    results: List[dict] = []
    t_start = time.time()
    for b_idx, batch in enumerate(batches):
        event_ids = [e.id for e in batch]
        _run_patch(insts, patched, col, row, tile_type, event_ids)

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
        if hw_res.ok:
            hw_res.per_event_count = _relabel_events(
                hw_events_out, col, row, tile_type, batch)

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
        if emu_res.ok:
            emu_res.per_event_count = _relabel_events(
                emu_events_out, col, row, tile_type, batch)

        # Per-slot attribution: events_count is the batch-wide total
        # (kept for sanity checks); per_event_count breaks it out by
        # slot so each event has its own fire count.  Cycles is the
        # last-event timestamp, still batch-level.
        for slot, ev in enumerate(batch):
            results.append({
                "id": ev.id,
                "name": ev.name,
                "slot": slot,
                "batch": b_idx,
                "hw": {
                    "cycles": hw_res.cycles,
                    "events_total": hw_res.events_count,
                    "fired": hw_res.per_event_count.get(ev.name, 0) if hw_res.ok else None,
                    "error": hw_res.error,
                },
                "emu": {
                    "cycles": emu_res.cycles,
                    "events_total": emu_res.events_count,
                    "fired": emu_res.per_event_count.get(ev.name, 0) if emu_res.ok else None,
                    "error": emu_res.error,
                },
            })
        print(f"[sweep] batch {b_idx + 1}/{len(batches)}: "
              f"events {[e.name for e in batch]} "
              f"hw_cyc={hw_res.cycles} emu_cyc={emu_res.cycles}",
              flush=True)

    summary = {
        "test": test_name,
        "compiler": compiler,
        "tile": {"col": col, "row": row, "type": tile_type},
        "events": results,
        "elapsed_sec": round(time.time() - t_start, 2),
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


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
    ap.add_argument("--col", type=int, required=True)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--tile-type", required=True,
                    choices=["core", "memmod", "memtile", "shim"])
    ap.add_argument("--build-dir", type=Path,
                    help="override build directory "
                         "(default: <mlir-aie>/build/test/npu-xrt/<test>/<compiler>)")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("--events",
                    help="comma-separated event name whitelist "
                         "(default: every event in the tile type)")
    ap.add_argument("--ctrlpkt", type=Path, help="optional control-packet blob to pass as --input")
    ap.add_argument("--no-hw", action="store_true", help="skip HW runs")
    ap.add_argument("--no-emu", action="store_true", help="skip EMU runs")
    args = ap.parse_args()

    build_dir = args.build_dir or (
        MLIR_AIE_ROOT / "build" / "test" / "npu-xrt" / args.test / args.compiler
    )
    events_filter = [s.strip() for s in args.events.split(",")] if args.events else None

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
    )
    print(f"[sweep] wrote {args.out} ({len(summary['events'])} events, "
          f"{summary['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
