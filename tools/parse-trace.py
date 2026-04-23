#!/usr/bin/env python3
"""
parse-trace.py -- Unified wrapper around mlir-aie's trace decoder.

Exposes every decoder layer the xdna-emu pipeline cares about, all fed by the
same mlir-aie parse, so downstream tools never disagree about what the trace
says:

  --out-events <path>    Flat per-tile event list (primary format).  Each
                         record: {col, row, pkt_type, slot, name, ts}.
                         Consumed by trace-compare.

  --out-cycles <path>    Scalar integer: max(ts) - min(ts) across all events.
                         Kernel-duration proxy; drop-in for the old
                         trace-to-cycles.py output.

  --out-perfetto <path>  Raw Perfetto JSON from aie.utils.trace.parse_trace
                         (begin/end/metadata events).  For debugging and
                         ui.perfetto.dev viewing.

  --out-commands <path>  Decoded command stream per (trace_type, row, col),
                         with Start tokens preserved.  For debugging decoder
                         edge cases (Start-token semantics, Event_Sync, etc.)
                         without touching mlir-aie.

At least one --out-* option is required.  Parsing happens once; each output is
derived from shared intermediate state.

Exit codes:
  0 -- success
  1 -- bad input, parse failure, or mlir-aie not importable
"""
import argparse
import json
import re
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--trace-bin", required=True,
                   help="raw trace bytes (uint32 words) from the runtime")
    p.add_argument("--xclbin-mlir", required=True,
                   help="post-lowering MLIR (input_with_addresses.mlir) used "
                        "to build the xclbin; names the trace events")
    p.add_argument("--out-events")
    p.add_argument("--out-cycles")
    p.add_argument("--out-perfetto")
    p.add_argument("--out-commands")
    args = p.parse_args()
    if not any([args.out_events, args.out_cycles, args.out_perfetto, args.out_commands]):
        p.error("at least one --out-* option is required")
    return args


# Matches mlir-aie's process_name_metadata args.name format:
# "{pt_name}_trace for tile{row},{col}" where pt_name is core/mem/shim/memtile.
_TILE_RE = re.compile(r"^(core|mem|shim|memtile)_trace for tile(\d+),(\d+)$")
_PT_NAME_TO_CODE = {"core": 0, "mem": 1, "shim": 2, "memtile": 3}


def build_pid_map(perfetto_events):
    """Recover pid -> (pkt_type, row, col) from Perfetto metadata events.

    mlir-aie's parse_trace emits one process_name metadata event per tile; its
    args.name encodes the tile coords. We re-read that so downstream tools can
    key on coords rather than opaque pid integers.
    """
    pid_map = {}
    for e in perfetto_events:
        if not isinstance(e, dict):
            continue
        if e.get("ph") != "M" or e.get("name") != "process_name":
            continue
        pid = e.get("pid")
        args_name = (e.get("args") or {}).get("name", "")
        m = _TILE_RE.match(args_name)
        if not m or pid is None:
            continue
        pt_name, row, col = m.group(1), int(m.group(2)), int(m.group(3))
        pid_map[pid] = (_PT_NAME_TO_CODE[pt_name], row, col)
    return pid_map


def perfetto_to_events(perfetto_events):
    """Flatten Perfetto events into our (col, row, pkt_type, slot, name, ts) form.

    Only B-phase events are kept. Perfetto emits B/E pairs for each trace
    event; the B timestamp is the occurrence time. This matches what the
    former Rust decoder produced (one record per trace-unit emission).
    """
    pid_map = build_pid_map(perfetto_events)
    flat = []
    for e in perfetto_events:
        if not isinstance(e, dict):
            continue
        if e.get("ph") != "B":
            continue
        pid = e.get("pid")
        tile = pid_map.get(pid)
        if tile is None:
            continue
        pkt_type, row, col = tile
        flat.append({
            "col": col,
            "row": row,
            "pkt_type": pkt_type,
            "slot": int(e.get("tid", 0)),
            "name": e.get("name", ""),
            "ts": int(e.get("ts", 0)),
        })
    flat.sort(key=lambda r: (r["col"], r["row"], r["pkt_type"], r["ts"]))
    return flat


def perfetto_to_slot_names(perfetto_events):
    """Recover slot->name tables from Perfetto thread_name metadata.

    mlir-aie emits one thread_name metadata event per (pid, tid) pair, where
    pid identifies the tile and tid is the slot index. args.name is the
    hardware event name. We aggregate across all tiles of each pkt_type; all
    tiles of the same type should share the same slot layout (the trace
    routing uses a single event config per tile type).

    Returns: {"core": [name|"", ...], "mem": [...], "shim": [...], "memtile": [...]}
    where each list has length 8 (the trace unit has 8 slots). Missing slots
    are empty strings.
    """
    pid_map = build_pid_map(perfetto_events)
    pt_code_to_key = {0: "core", 1: "mem", 2: "shim", 3: "memtile"}
    names = {k: [""] * 8 for k in pt_code_to_key.values()}
    for e in perfetto_events:
        if not isinstance(e, dict):
            continue
        if e.get("ph") != "M" or e.get("name") != "thread_name":
            continue
        pid = e.get("pid")
        tid = e.get("tid")
        if pid is None or tid is None or pid not in pid_map:
            continue
        pt_code, _row, _col = pid_map[pid]
        key = pt_code_to_key.get(pt_code)
        if key is None:
            continue
        slot = int(tid)
        if 0 <= slot < 8:
            name = (e.get("args") or {}).get("name", "")
            # First non-empty wins; later tiles of same type should agree.
            if name and not names[key][slot]:
                names[key][slot] = name
    return names


def commands_per_tile(trace_buffer, mlir_text):
    """Produce the per-tile decoded command stream (Start/Single/Multiple/
    Repeat/Event_Sync). Start tokens preserve their 56-bit timer value.

    Structure:
      {
        "trace_types": [
          {  # index = PacketType value (0=core, 1=mem, 2=shim, 3=memtile)
            "row,col": [ {type: "Start", timer_value: 268901}, ... ]
          },
          ...
        ]
      }
    """
    from aie.utils.trace.utils import (
        split_trace_segments,
        trim_trace_pkts,
        trace_pkts_de_interleave,
        convert_to_byte_stream,
        convert_to_commands,
    )
    from aie.utils.trace.parse import check_for_valid_trace
    from aie.utils.trace.events import NUM_TRACE_TYPES

    trace_pkts = [f"{int(w):08x}" for w in trace_buffer]
    segments = split_trace_segments(trace_pkts)
    if not segments:
        raise ValueError("invalid trace data: empty or all zeros")

    merged = [dict() for _ in range(NUM_TRACE_TYPES)]
    for segment in segments:
        if not check_for_valid_trace("<parse-trace>", segment):
            continue
        trimmed = trim_trace_pkts(segment)
        sorted_pkts = trace_pkts_de_interleave(trimmed)
        for t in range(NUM_TRACE_TYPES):
            for loc, data in sorted_pkts[t].items():
                merged[t].setdefault(loc, []).extend(data)

    byte_streams = convert_to_byte_stream(merged)
    # zero=False preserves Start-token timer values (defaults would zero them).
    commands = convert_to_commands(byte_streams, False)
    return {"trace_types": commands}


def main():
    args = parse_args()

    try:
        import numpy as np
        from aie.utils.trace.parse import parse_trace
    except ImportError as e:
        print(
            f"error: mlir-aie trace module not importable: {e}\n"
            "  ensure PYTHONPATH includes mlir-aie/install/python and the "
            "ironenv Python is active",
            file=sys.stderr,
        )
        return 1

    raw = np.fromfile(args.trace_bin, dtype=np.uint32)
    mlir_text = Path(args.xclbin_mlir).read_text()

    perfetto_events = parse_trace(raw, mlir_text)

    if args.out_perfetto:
        Path(args.out_perfetto).write_text(json.dumps(perfetto_events, indent=2))

    if args.out_commands:
        cmds = commands_per_tile(raw, mlir_text)
        Path(args.out_commands).write_text(json.dumps(cmds, indent=2))

    # Derive flat events from the Perfetto list once; used by both --out-events
    # and --out-cycles to guarantee they agree.
    flat = perfetto_to_events(perfetto_events)

    if args.out_events:
        slot_names = perfetto_to_slot_names(perfetto_events)
        Path(args.out_events).write_text(
            json.dumps({
                "schema_version": 1,
                "events": flat,
                "slot_names": slot_names,
            }, indent=2)
        )

    if args.out_cycles:
        if not flat:
            print("error: trace has no timestamped events", file=sys.stderr)
            return 1
        span = max(e["ts"] for e in flat) - min(e["ts"] for e in flat)
        Path(args.out_cycles).write_text(f"{span}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
