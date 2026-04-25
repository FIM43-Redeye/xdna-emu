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
    p.add_argument("--trace-bin",
                   help="raw trace bytes (uint32 words) from the runtime")
    p.add_argument("--xclbin-mlir",
                   help="post-lowering MLIR (input_with_addresses.mlir) used "
                        "to build the xclbin; names the trace events")
    p.add_argument("--out-events")
    p.add_argument("--out-cycles")
    p.add_argument("--out-perfetto")
    p.add_argument("--out-commands")
    p.add_argument("--server", action="store_true",
                   help="run as a long-lived decode server. Reads one JSON "
                        "request per stdin line {trace_bin, xclbin_mlir, "
                        "out_events?, out_cycles?, out_perfetto?, "
                        "out_commands?} and writes one JSON response per "
                        "line. Heavy mlir-aie/numpy imports happen once at "
                        "startup; subsequent decodes are ~50x faster than "
                        "spawning a fresh interpreter per call.")
    args = p.parse_args()
    if args.server:
        # Server mode reads request paths from stdin -- one-shot CLI args
        # would be ignored, so flag the misuse early.
        if any([args.trace_bin, args.xclbin_mlir, args.out_events,
                args.out_cycles, args.out_perfetto, args.out_commands]):
            p.error("--server is mutually exclusive with --trace-bin / "
                    "--out-* flags; pass those in stdin requests instead")
    else:
        if not args.trace_bin or not args.xclbin_mlir:
            p.error("--trace-bin and --xclbin-mlir are required (or use --server)")
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


def _import_decoder():
    """Import the heavy mlir-aie trace decoder. Returns (numpy, parse_trace)
    or raises ImportError with a hint if the env isn't set up. Pulled out
    so server mode can amortize the cost across many requests.
    """
    import numpy as np
    from aie.utils.trace.parse import parse_trace
    return np, parse_trace


def decode_one(np_mod, parse_trace_fn,
               trace_bin: str, xclbin_mlir: str,
               out_events: str = None, out_cycles: str = None,
               out_perfetto: str = None, out_commands: str = None) -> dict:
    """Decode a single trace and emit any of the four output formats.

    Returns a result dict that the server mode forwards to its caller:
      {"ok": True, "events_count": N, "cycles": <span or None>}
    or
      {"ok": True, "events_count": 0, "cycles": 0, "empty": True}
        when the trace had no timestamped events. (One-shot CLI mode
        treats empty as a hard failure to preserve the historic exit
        code; server mode lets callers handle it.)
    """
    raw = np_mod.fromfile(trace_bin, dtype=np_mod.uint32)
    mlir_text = Path(xclbin_mlir).read_text()
    try:
        perfetto_events = parse_trace_fn(raw, mlir_text)
    except ValueError as e:
        # mlir-aie's parse_trace raises ValueError("Invalid trace data:
        # empty or all zeros") when the kernel ran but no events fired.
        # That's a normal sweep outcome (e.g. an event that never
        # triggers in this kernel), not a parse failure -- emit empty
        # outputs and report empty=True so the caller can distinguish
        # "kernel ran, nothing to see" from "decoder broke."
        if "empty or all zeros" in str(e):
            if out_events:
                Path(out_events).write_text(
                    '{"schema_version":1,"events":[],"slot_names":{}}\n')
            if out_cycles:
                Path(out_cycles).write_text("0\n")
            if out_perfetto:
                Path(out_perfetto).write_text("[]\n")
            if out_commands:
                Path(out_commands).write_text('{"trace_types":{}}\n')
            return {"ok": True, "events_count": 0, "cycles": 0, "empty": True}
        raise

    if out_perfetto:
        Path(out_perfetto).write_text(json.dumps(perfetto_events, indent=2))
    if out_commands:
        cmds = commands_per_tile(raw, mlir_text)
        Path(out_commands).write_text(json.dumps(cmds, indent=2))

    flat = perfetto_to_events(perfetto_events)
    cycles = None
    if flat:
        cycles = max(e["ts"] for e in flat) - min(e["ts"] for e in flat)

    if out_events:
        slot_names = perfetto_to_slot_names(perfetto_events)
        Path(out_events).write_text(
            json.dumps({
                "schema_version": 1,
                "events": flat,
                "slot_names": slot_names,
            }, indent=2)
        )
    if out_cycles:
        # Empty trace -> 0 cycles. CLI main() turns that into a failure
        # for backward compat; server mode reports it as ok+empty.
        Path(out_cycles).write_text(f"{cycles or 0}\n")

    return {
        "ok": True,
        "events_count": len(flat),
        "cycles": cycles,
        "empty": not flat,
    }


def server_loop(np_mod, parse_trace_fn) -> int:
    """Long-lived decode loop. One JSON request per stdin line; one JSON
    response per stdout line. Stdin EOF exits cleanly.

    Why this exists: the import cost above (~620 ms of mlir-aie + numpy
    initialization) dominates a per-batch sweep; running 32 batches as
    32 fresh interpreters wastes ~20 s on imports alone. With the server
    mode, the sweep spawns one decoder process and reuses it.
    """
    import time as _t
    print(json.dumps({"event": "ready"}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "exit":
            break
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"bad request json: {e}"}),
                  flush=True)
            continue
        t0 = _t.monotonic()
        try:
            result = decode_one(
                np_mod, parse_trace_fn,
                trace_bin=req["trace_bin"],
                xclbin_mlir=req["xclbin_mlir"],
                out_events=req.get("out_events"),
                out_cycles=req.get("out_cycles"),
                out_perfetto=req.get("out_perfetto"),
                out_commands=req.get("out_commands"),
            )
        except KeyError as e:
            print(json.dumps({"ok": False,
                              "error": f"missing request key: {e}"}),
                  flush=True)
            continue
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)
            continue
        result["elapsed_ms"] = int((_t.monotonic() - t0) * 1000)
        print(json.dumps(result), flush=True)
    return 0


def main():
    args = parse_args()
    try:
        np_mod, parse_trace_fn = _import_decoder()
    except ImportError as e:
        print(
            f"error: mlir-aie trace module not importable: {e}\n"
            "  ensure PYTHONPATH includes mlir-aie/install/python and the "
            "ironenv Python is active",
            file=sys.stderr,
        )
        return 1

    if args.server:
        return server_loop(np_mod, parse_trace_fn)

    result = decode_one(
        np_mod, parse_trace_fn,
        trace_bin=args.trace_bin,
        xclbin_mlir=args.xclbin_mlir,
        out_events=args.out_events,
        out_cycles=args.out_cycles,
        out_perfetto=args.out_perfetto,
        out_commands=args.out_commands,
    )
    if result.get("empty") and args.out_cycles:
        # CLI mode preserves the old hard-failure behavior on empty
        # trace + --out-cycles; tools rely on the exit code today.
        print("error: trace has no timestamped events", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
