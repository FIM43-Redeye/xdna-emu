#!/usr/bin/env python3
"""
parse-trace.py -- Unified wrapper around the AIE trace decoders.

Exposes every decoder layer the xdna-emu pipeline cares about so downstream
tools never disagree about what the trace says.  Two decoder backends are
selectable via --decoder:

  --decoder ours       (default) routes through tools/trace_decoder/, the
                       in-tree MIT-licensed decoder.  Authoritative for the
                       xdna-emu validation pipeline today because mlir-aie's
                       parse_trace covers only mode 0 (EVENT_TIME); modes 1
                       (EVENT_PC) and 2 (INST_EXEC) are decode-only here.
  --decoder mlir-aie   routes through aie.utils.trace.parse_trace.  Useful
                       for cross-validation (we maintain bit-perfect mode-0
                       parity with this oracle) and as the swap-back path
                       once mlir-aie covers all three modes upstream.

Output formats:

  --out-events <path>    Flat per-tile event list (primary format).  Each
                         record: {col, row, pkt_type, slot, name, ts}.
                         Consumed by trace-compare.

  --out-cycles <path>    Scalar integer: max(ts) - min(ts) across all events.
                         Kernel-duration proxy; drop-in for the old
                         trace-to-cycles.py output.

  --out-perfetto <path>  Perfetto JSON timeline (begin/end/metadata events)
                         viewable at ui.perfetto.dev.  Both decoders emit
                         B/E pairs for mode 0; the in-tree decoder rejects
                         modes 1/2 here pending a B/E convention for those.

  --out-commands <path>  Decoded command stream per (trace_type, row, col),
                         with Start tokens preserved.  For debugging decoder
                         edge cases (Start-token semantics, Event_Sync, etc.)
                         without touching mlir-aie.

At least one --out-* option is required.  Parsing happens once; each output is
derived from shared intermediate state.

Exit codes:
  0 -- success
  1 -- bad input, parse failure, or selected decoder not importable
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
    p.add_argument("--decoder", choices=("mlir-aie", "ours"), default="ours",
                   help="byte-stream decoder backend. 'ours' (default) uses "
                        "the in-tree tools/trace_decoder package, which "
                        "covers EVENT_TIME (mode 0), EVENT_PC (mode 1), and "
                        "INST_EXEC (mode 2). 'mlir-aie' routes through "
                        "aie.utils.trace.parse_trace (mode 0 only); use it "
                        "for cross-validation against the upstream oracle. "
                        "--out-perfetto is supported on both backends for "
                        "mode 0; 'ours' rejects --out-perfetto for modes "
                        "1/2 pending a B/E convention.")
    p.add_argument("--trace-mode", choices=("event_time", "event_pc", "inst_exec"),
                   default="event_time",
                   help="trace mode selector for --decoder=ours. 'event_time' "
                        "(default) decodes mode-0 traces with cycle-delta "
                        "events. 'event_pc' decodes mode-1 traces where the "
                        "encoded quantity is the PC at each event fire. "
                        "'inst_exec' decodes mode-2 instruction-execution "
                        "traces (E/N_atom cycle records, New_PC branches, "
                        "LC loop counts).")
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
        if not args.trace_bin:
            p.error("--trace-bin is required (or use --server)")
        # Mode 2 emits PC/atom records and never needs the MLIR for
        # slot-name lookup, so --xclbin-mlir is optional there.  Modes
        # 0 and 1 still require it.
        if args.trace_mode != "inst_exec" and not args.xclbin_mlir:
            p.error("--xclbin-mlir is required for trace_mode={!r} (or use "
                    "--server)".format(args.trace_mode))
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


def _import_numpy():
    """Import numpy. Both decoder backends need it (raw trace words are
    loaded via ``np.fromfile``)."""
    import numpy as np
    return np


def _load_trace_words(np_mod, trace_bin: str):
    """Read trace_bin as uint32 dwords and trim the leading and trailing
    zero-padding runs. Returns a trimmed numpy array; callers needing a
    Python list can call .tolist() on the result.

    Why both ends: the bridge runner allocates trace_size bytes for the
    BO (typically 1 MB) and the kernel only fills the slice it actually
    used. Trailing zeros are uninitialized BO past the trace tail.
    Leading zeros happen because the trace channel can append into the
    BO at an offset (mode-1 sweeps reuse the BO across batches and each
    batch's frames land where the previous one's ended), and the
    leading-zero region is simply "this batch wrote bytes [N..M); the
    rest stayed zero from prior runs / from BO init".

    Inner zeros are preserved -- mode-2 streams legitimately encode
    N_atom frames as 0x00 nibbles, and a long N_atom run inside the
    real trace must not be conflated with padding. Empty BOs (no
    non-zero dwords anywhere) return a single zero dword so downstream
    "empty trace" branches still fire.
    """
    raw = np_mod.fromfile(trace_bin, dtype=np_mod.uint32)
    if raw.size == 0:
        return raw
    non_zero = np_mod.flatnonzero(raw)
    if non_zero.size == 0:
        return raw[:1]
    first = int(non_zero[0])
    last = int(non_zero[-1])
    return raw[first:last + 1]


def _import_mlir_aie_parse_trace():
    """Import the heavy mlir-aie trace decoder.  Returns ``parse_trace`` or
    raises ImportError with a hint if the env isn't set up.  Lazy because
    --decoder=ours (the default) doesn't need it; pulling it in
    unconditionally would force the ironenv Python on every caller."""
    from aie.utils.trace.parse import parse_trace
    return parse_trace


def _import_ours():
    """Import the in-tree trace_decoder package.

    Lives next to this script (tools/trace_decoder/), so we add the
    parent directory to sys.path on first call.  Returns the package
    module so callers can use trace_decoder.decode_words /
    trace_decoder.parse_trace directly.
    """
    import importlib
    import os
    import sys as _sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in _sys.path:
        _sys.path.insert(0, here)
    return importlib.import_module("trace_decoder")


def _slot_names_from_mlir(mlir_text):
    """Extract per-tile slot-name tables by delegating to mlir-aie.

    Returns a dict ``{pkt_type_int: {"row,col": [name0..name7]}}`` that
    matches the shape ``trace_decoder.parse_trace`` expects.

    We use mlir-aie's ``parse_mlir_trace_events`` because MLIR parsing
    is squarely in the mlir-aie toolchain's domain (and mlir-aie is
    open source, so this is not the dependency we are trying to avoid
    -- aietools is).  Slot codes are translated to event names via
    ``lookup_event_name_by_type`` for the same reason.
    """
    from aie.utils.trace.parse import (
        parse_mlir_trace_events,
        lookup_event_name_by_type,
    )
    pid_events, events_module = parse_mlir_trace_events(mlir_text, None)
    slot_names = {}
    for pkt_type, by_loc in enumerate(pid_events):
        if not by_loc:
            continue
        slot_names[pkt_type] = {}
        for loc, codes in by_loc.items():
            slot_names[pkt_type][loc] = [
                lookup_event_name_by_type(pkt_type, code, events_module)
                for code in codes[:8]
            ]
    return slot_names


def decode_one_ours(np_mod, td_mod,
                    trace_bin: str, xclbin_mlir: str,
                    out_events: str = None, out_cycles: str = None,
                    out_perfetto: str = None, out_commands: str = None,
                    trace_mode: str = "event_time") -> dict:
    """Decode a single trace using the in-tree decoder backend.

    Mirrors the public surface of ``decode_one`` but routes through
    ``trace_decoder``.  Output formats supported:

      --out-commands   per-tile typed-command JSON (Start/Single/Multiple/
                       Repeat/Event_Sync), shape-compatible with the
                       mlir-aie path.
      --out-events     flat event list ({col, row, pkt_type, slot, name,
                       ts}), bit-equivalent to the mlir-aie path on mode 0.
                       In mode 1 (EVENT_PC) the ``ts`` field carries the
                       PC instead of a cycle count.
      --out-cycles     scalar max(ts) - min(ts) over emitted events.

    ``trace_mode`` selects the per-tile decoder; supported values are
    ``"event_time"`` (mode 0, default), ``"event_pc"`` (mode 1), and
    ``"inst_exec"`` (mode 2).  Mode 2 has its own native output shape
    (E/N_atom + New_PC + LC + Repeat + Start/Stop/Sync); see
    ``_decode_mode2_to_outputs`` for details.

    --out-perfetto is supported for ``trace_mode='event_time'`` (mode 0)
    only -- modes 1 (PC anchors instead of cycle anchors) and 2 (per-
    cycle E_atom/N_atom records) don't have an established B/E pair
    convention yet, so we reject them rather than emit a malformed
    timeline.

    --out-cycles is rejected for mode 2 because mode-2 frames don't
    carry cycle deltas in a form ``max(ts) - min(ts)`` would summarize
    meaningfully -- the cycle count is the number of E/N_atom frames,
    which downstream tools can derive from --out-events.
    """
    mode_lookup = {
        "event_time": td_mod.TraceMode.EVENT_TIME,
        "event_pc": td_mod.TraceMode.EVENT_PC,
        "inst_exec": td_mod.TraceMode.INST_EXEC,
    }
    if trace_mode not in mode_lookup:
        raise ValueError(
            f"--decoder=ours: unsupported trace_mode {trace_mode!r}; "
            f"expected one of {sorted(mode_lookup)}"
        )

    if out_perfetto and trace_mode != "event_time":
        raise ValueError(
            f"--decoder=ours --out-perfetto: trace_mode={trace_mode!r} "
            "not supported; only event_time (mode 0) emits Perfetto B/E "
            "pairs in this iteration"
        )

    if trace_mode == "inst_exec":
        # Mode 2 has a different command vocabulary (atoms, New_PC, LC)
        # that doesn't fit the cycles/timeline model of modes 0/1, so it
        # gets its own decode/emit path entirely.
        return _decode_mode2_to_outputs(
            np_mod, td_mod,
            trace_bin=trace_bin,
            out_events=out_events,
            out_cycles=out_cycles,
            out_commands=out_commands,
        )

    mode_enum = mode_lookup[trace_mode]

    raw = _load_trace_words(np_mod, trace_bin)
    words = raw.tolist()

    # Compute slot_names once if anything downstream needs them.
    slot_names = {}
    if (out_events or out_cycles or out_perfetto) and xclbin_mlir:
        mlir_text = Path(xclbin_mlir).read_text()
        slot_names = _slot_names_from_mlir(mlir_text)

    if out_commands:
        commands_per_tile = td_mod.decode_words(words, mode=mode_enum)
        # Reshape into the mlir-aie-compatible output: trace_types[pkt_type][f"{row},{col}"] = [...]
        max_pt = max((pt for (pt, _, _) in commands_per_tile), default=-1)
        trace_types = [dict() for _ in range(max(4, max_pt + 1))]
        for (pt, r, c), cmds in commands_per_tile.items():
            trace_types[pt][f"{r},{c}"] = [_cmd_to_oracle_dict(cmd) for cmd in cmds]
        Path(out_commands).write_text(json.dumps({"trace_types": trace_types}, indent=2))

    if out_perfetto:
        commands_per_tile = td_mod.decode_words(words, mode=mode_enum)
        perfetto_events = td_mod.rebuild_perfetto_mode0(commands_per_tile, slot_names)
        Path(out_perfetto).write_text(json.dumps(perfetto_events, indent=2))

    flat = []
    cycles = None
    if out_events or out_cycles:
        events = td_mod.parse_trace(words, slot_names=slot_names,
                                    mode=mode_enum)
        flat = [
            {
                "col": e.col,
                "row": e.row,
                "pkt_type": e.pkt_type,
                "slot": e.slot,
                "name": e.name,
                "ts": e.ts,
            }
            for e in events
        ]
        flat.sort(key=lambda r: (r["col"], r["row"], r["pkt_type"], r["ts"]))
        if flat:
            cycles = max(e["ts"] for e in flat) - min(e["ts"] for e in flat)

    if out_events:
        # The mlir-aie path emits a slot_names structure here too; we
        # reuse the same one we used internally so downstream consumers
        # can render slot indices to names without re-reading the MLIR.
        slot_names_named = _slot_names_for_output(slot_names if (out_events and xclbin_mlir) else {})
        # Per-side observed placement: smallest (col, row) corner that
        # actually produced trace events. trace-compare uses this to
        # normalize HW vs EMU when one side runs at start_col != 0,
        # without relying on the dense-remap heuristic.
        payload = {
            "schema_version": 1,
            "events": flat,
            "slot_names": slot_names_named,
        }
        if flat:
            payload["placement"] = {
                "origin_col": min(e["col"] for e in flat),
                "origin_row": min(e["row"] for e in flat),
            }
        Path(out_events).write_text(json.dumps(payload, indent=2))
    if out_cycles:
        Path(out_cycles).write_text(f"{cycles or 0}\n")

    return {
        "ok": True,
        "events_count": len(flat),
        "cycles": cycles,
        "empty": (out_events or out_cycles) and not flat,
    }


def _mode2_cmd_to_dict(cmd) -> dict:
    """Serialize one mode-2 ``TraceCommand`` to its fixture-shape dict.

    Schema is the one frozen by
    ``tools/trace_decoder/fixtures/mode2_mixed_r0_core_expected.json``
    and used by ``test_trace_decoder.py::_mode2_cmd_to_dict`` -- the
    canonical mode-2 emission format for this project.  Keeping the two
    in lockstep means parse-trace.py output can be diffed directly
    against fixtures without an adapter layer.
    """
    # Imported lazily inside the function so callers that never touch
    # mode 2 don't pay for the modes/mode2 import.
    from trace_decoder.modes.mode2 import CycleCmd, LoopCountCmd
    from trace_decoder.frame import (
        EventCmd, RepeatCmd, StartCmd, StopCmd, SyncCmd,
    )

    if isinstance(cmd, StartCmd):
        return {"type": "Start", "anchor_pc": cmd.timer_value}
    if isinstance(cmd, CycleCmd):
        return {"type": "N_atom" if cmd.stalled else "E_atom"}
    if isinstance(cmd, EventCmd):
        # In mode 2 EventCmd is reused for New_PC; event_bits is always
        # 0 and ``cycles`` carries the 14-bit absolute PC.
        return {"type": "New_PC", "pc": cmd.cycles}
    if isinstance(cmd, RepeatCmd):
        return {"type": "Repeat", "count": cmd.count}
    if isinstance(cmd, LoopCountCmd):
        return {"type": "LC", "flag": cmd.flag, "count": cmd.count}
    if isinstance(cmd, SyncCmd):
        return {"type": "Sync"}
    if isinstance(cmd, StopCmd):
        return {"type": "Stop"}
    raise AssertionError(f"unhandled mode-2 cmd: {cmd!r}")


def _decode_mode2_to_outputs(np_mod, td_mod, *,
                             trace_bin: str,
                             out_events: str = None,
                             out_cycles: str = None,
                             out_commands: str = None) -> dict:
    """Decode a mode-2 (INST_EXEC) trace and write per-tile event lists.

    Output JSON shape (matches the frozen fixture schema):

      {
        "schema_version": 1,
        "trace_mode": "inst_exec",
        "tiles": {
          "<pkt_type>,<row>,<col>": [
            {"type": "Start", "anchor_pc": 816},
            {"type": "New_PC", "pc": 368},
            {"type": "E_atom"},
            {"type": "N_atom"},
            {"type": "LC", "flag": 1, "count": 8},
            {"type": "Repeat", "count": 5},
            {"type": "Sync"},
            {"type": "Stop"}
          ],
          ...
        }
      }

    Each per-tile list has the same shape as
    ``mode2_mixed_r0_core_expected.json``, so single-tile fixtures can
    be diffed via ``tiles["0,2,1"]`` directly.

    --out-cycles is rejected (no cycle scalar in mode 2 -- callers can
    count atoms in --out-events if they need a length proxy).
    --out-commands is accepted as an alias for --out-events in mode 2;
    the per-tile list is the typed command stream (no separate
    Single/Multiple reshape happens).
    """
    if out_cycles:
        raise ValueError(
            "--decoder=ours --trace-mode=inst_exec: --out-cycles not "
            "supported; mode 2 has no cycle delta to summarize. Count "
            "atoms in --out-events for a length proxy."
        )

    raw = _load_trace_words(np_mod, trace_bin)
    words = raw.tolist()

    commands_per_tile = td_mod.decode_words(
        words, mode=td_mod.TraceMode.INST_EXEC
    )

    tiles = {}
    total_cmds = 0
    for (pt, row, col), cmds in commands_per_tile.items():
        key = f"{int(pt)},{row},{col}"
        tiles[key] = [_mode2_cmd_to_dict(c) for c in cmds]
        total_cmds += len(cmds)

    payload = {
        "schema_version": 1,
        "trace_mode": "inst_exec",
        "tiles": tiles,
    }
    if out_events:
        Path(out_events).write_text(json.dumps(payload, indent=2))
    if out_commands:
        # Mode 2's typed-command output is already the per-tile list
        # we just wrote; --out-commands is functionally an alias for
        # --out-events here, but we honour it so callers can ask for
        # both files without special-casing.
        Path(out_commands).write_text(json.dumps(payload, indent=2))

    return {
        "ok": True,
        "events_count": total_cmds,
        "cycles": None,
        "empty": (out_events or out_commands) and total_cmds == 0,
    }


def _cmd_to_oracle_dict(cmd) -> dict:
    """Convert a typed TraceCommand to the mlir-aie convert_to_commands
    output shape, so --decoder=ours --out-commands stays drop-in.

    The size-variant distinction (Single0 vs Single1 vs Single2, etc.)
    is encoder-side and not preserved by our schema; we use the variant
    suffix that minimally encodes the cycles field, matching what
    mlir-aie's encoder would have chosen for the same byte stream.
    """
    type_name = type(cmd).__name__
    if type_name == "StartCmd":
        return {"type": "Start", "timer_value": cmd.timer_value}
    if type_name == "SyncCmd":
        return {"type": "Event_Sync"}
    if type_name == "RepeatCmd":
        suffix = "0" if cmd.count < 16 else "1"
        return {"type": f"Repeat{suffix}", "repeats": cmd.count}
    if type_name == "EventCmd":
        bits_set = bin(cmd.event_bits).count("1")
        if bits_set == 1:
            slot = (cmd.event_bits & -cmd.event_bits).bit_length() - 1
            if cmd.cycles < 16:
                suffix = "0"
            elif cmd.cycles < 1024:
                suffix = "1"
            else:
                suffix = "2"
            return {"type": f"Single{suffix}", "event": slot, "cycles": cmd.cycles}
        if cmd.cycles < 16:
            suffix = "0"
        elif cmd.cycles < 1024:
            suffix = "1"
        else:
            suffix = "2"
        out = {"type": f"Multiple{suffix}", "cycles": cmd.cycles}
        for i in range(8):
            if cmd.event_bits & (1 << i):
                out[f"event{i}"] = i
        return out
    raise AssertionError(f"unknown command type: {type_name}")


def _slot_names_for_output(slot_names_internal):
    """Reshape internal slot_names ({pkt_type: {loc: [names]}}) into the
    output shape mlir-aie's parse-trace.py emits ({"core"/"mem"/"shim"/
    "memtile": [name0..name7]}), aggregating across tiles of the same
    type.  First non-empty wins per slot.
    """
    pt_code_to_key = {0: "core", 1: "mem", 2: "shim", 3: "memtile"}
    result = {k: [""] * 8 for k in pt_code_to_key.values()}
    for pkt_type, by_loc in slot_names_internal.items():
        key = pt_code_to_key.get(pkt_type)
        if key is None:
            continue
        for _loc, names in by_loc.items():
            for slot, name in enumerate(names[:8]):
                if name and not result[key][slot]:
                    result[key][slot] = name
    return result


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
    raw = _load_trace_words(np_mod, trace_bin)
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


def server_loop(np_mod, parse_trace_fn, td_mod=None,
                default_decoder="ours") -> int:
    """Long-lived decode loop. One JSON request per stdin line; one JSON
    response per stdout line. Stdin EOF exits cleanly.

    Why this exists: the import cost above (~620 ms of mlir-aie + numpy
    initialization) dominates a per-batch sweep; running 32 batches as
    32 fresh interpreters wastes ~20 s on imports alone. With the server
    mode, the sweep spawns one decoder process and reuses it.

    Per-request override: include ``"decoder": "ours"`` or
    ``"decoder": "mlir-aie"`` in the JSON request to override the
    server's default backend on a per-call basis (defaults to whatever
    was passed to ``--decoder`` at server startup).

    Both backends are lazy-loaded: pass ``parse_trace_fn=None`` and/or
    ``td_mod=None`` and the loop will import on first use.  Lets a
    --decoder=ours server avoid the mlir-aie import entirely when no
    request needs it.
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
        decoder = req.get("decoder", default_decoder)
        if decoder == "ours" and td_mod is None:
            try:
                td_mod = _import_ours()
            except ImportError as e:
                print(json.dumps({"ok": False,
                                  "error": f"trace_decoder import failed: {e}"}),
                      flush=True)
                continue
        if decoder == "mlir-aie" and parse_trace_fn is None:
            try:
                parse_trace_fn = _import_mlir_aie_parse_trace()
            except ImportError as e:
                print(json.dumps({"ok": False,
                                  "error": f"mlir-aie parse_trace import "
                                           f"failed: {e}"}),
                      flush=True)
                continue
        try:
            if decoder == "ours":
                result = decode_one_ours(
                    np_mod, td_mod,
                    trace_bin=req["trace_bin"],
                    xclbin_mlir=req["xclbin_mlir"],
                    out_events=req.get("out_events"),
                    out_cycles=req.get("out_cycles"),
                    out_perfetto=req.get("out_perfetto"),
                    out_commands=req.get("out_commands"),
                    trace_mode=req.get("trace_mode", "event_time"),
                )
            else:
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
        np_mod = _import_numpy()
    except ImportError as e:
        print(f"error: numpy not importable: {e}", file=sys.stderr)
        return 1

    # Both decoder imports are lazy: only pull in what the selected backend
    # actually needs.  --decoder=ours (the default) skips mlir-aie entirely
    # so callers don't need ironenv on PATH for in-tree decoding.
    td_mod = None
    parse_trace_fn = None
    if args.decoder == "ours" or args.server:
        try:
            td_mod = _import_ours()
        except ImportError as e:
            print(f"error: in-tree trace_decoder not importable: {e}",
                  file=sys.stderr)
            return 1
    if args.decoder == "mlir-aie":
        try:
            parse_trace_fn = _import_mlir_aie_parse_trace()
        except ImportError as e:
            print(
                f"error: mlir-aie trace module not importable: {e}\n"
                "  ensure PYTHONPATH includes mlir-aie/install/python and the "
                "ironenv Python is active",
                file=sys.stderr,
            )
            return 1

    if args.server:
        return server_loop(np_mod, parse_trace_fn, td_mod=td_mod,
                           default_decoder=args.decoder)

    if args.decoder == "ours":
        try:
            result = decode_one_ours(
                np_mod, td_mod,
                trace_bin=args.trace_bin,
                xclbin_mlir=args.xclbin_mlir,
                out_events=args.out_events,
                out_cycles=args.out_cycles,
                out_perfetto=args.out_perfetto,
                out_commands=args.out_commands,
                trace_mode=args.trace_mode,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
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
