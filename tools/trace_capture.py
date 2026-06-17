#!/usr/bin/env python3
"""Self-owned NPU trace-capture engine for #140.

Takes a batch plan and produces correctly-labeled events.json per batch per run
on real hardware, reusing three audited primitives (register patcher, XRT
runner, in-tree decoder) and owning column-free exact labeling + N-run coverage.

See docs/superpowers/specs/2026-06-17-trace-capture-engine-design.md.
"""
import json
from pathlib import Path
from shlex import quote
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
_EVENTS_HEADER = (_REPO.parent / "mlir-aie/build/include/xaienginecdo_static/"
                  "xaiengine/xaie_events_aieml.h")
_MOD_PREFIX = {"core": "CORE", "memmod": "MEM", "memtile": "MEM_TILE", "shim": "PL"}


def load_event_ids(tile_type: str) -> Dict[str, int]:
    """{event_name: numeric_id} for a tile-type, from the aie-rt events header."""
    full = f"XAIEML_EVENTS_{_MOD_PREFIX[tile_type]}_"
    exclude = "XAIEML_EVENTS_MEM_TILE_" if tile_type == "memmod" else None
    out: Dict[str, int] = {}
    for line in _EVENTS_HEADER.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith(full):
            if exclude and parts[1].startswith(exclude):
                continue
            name = parts[1][len(full):]
            val = parts[2].rstrip("U")
            if val.isdigit():
                out.setdefault(name, int(val))   # first definition wins (stable)
    return out


PKT_TO_TILE_TYPE = {0: "core", 1: "memmod", 2: "shim", 3: "memtile"}


class CaptureError(Exception):
    pass


def _get(ev, attr):
    return ev[attr] if isinstance(ev, dict) else getattr(ev, attr)


def label_events(raw_events, label_map, traced_col: int) -> List[dict]:
    """Apply label_map to raw decoded events, with hard-error guards.

    Each raw event (dict or object with col,row,pkt_type,slot,ts,soc,mode)
    becomes a record {col,row,pkt_type,name,slot,ts,soc,mode}.

    Raises CaptureError if:
    - (pkt_type,row,slot) not in label_map (unconfigured slot)
    - col != traced_col (foreign column / start_col mismatch)
    """
    out = []
    for ev in raw_events:
        col = _get(ev, "col"); row = _get(ev, "row"); pkt = _get(ev, "pkt_type")
        slot = _get(ev, "slot")
        if col != traced_col:
            raise CaptureError(
                f"foreign column {col} (traced {traced_col}); start_col mismatch")
        key = (pkt, row, slot)
        if key not in label_map:
            raise CaptureError(f"event at unconfigured (pkt,row,slot)={key}")
        out.append({"col": col, "row": row, "pkt_type": pkt,
                    "name": label_map[key], "slot": slot,
                    "ts": _get(ev, "ts"), "soc": _get(ev, "soc"),
                    "mode": _get(ev, "mode")})
    return out


def configure_batch(batch: Dict[str, List[str]], anchor: str = "PERF_CNT_2"):
    """batch {"col|row|pkt": [names]} -> (patch_spec, label_map).

    label_map is keyed (pkt_type, row, slot) -- column-free by design.
    """
    patch_spec = []
    label_map: Dict[tuple, str] = {}
    for tile_key, names in batch.items():
        col, row, pkt = (int(x) for x in tile_key.split("|"))
        tile_type = PKT_TO_TILE_TYPE[pkt]
        # anchor first (slot 0) if present, then the rest in plan order
        ordered = ([anchor] if anchor in names else []) + [n for n in names if n != anchor]
        if len(ordered) > 8:
            raise ValueError(f"tile {tile_key} has {len(ordered)} events > 8 slots")
        ids = load_event_ids(tile_type)
        event_ids = []
        for slot, name in enumerate(ordered):
            if name not in ids:
                raise ValueError(f"event {name!r} not in {tile_type} table")
            event_ids.append(ids[name])
            label_map[(pkt, row, slot)] = name
        patch_spec.append({"col": col, "row": row, "tile_type": tile_type,
                           "events": event_ids})
    return patch_spec, label_map


def coverage_report(configured: Dict[tuple, str], observed_per_run: List[set]) -> dict:
    """Union observed slots across N runs and report gaps.

    Args:
        configured: {(pkt_type, row, slot): name}
        observed_per_run: list (one per run) of sets of (pkt_type, row, slot) that fired

    Returns:
        {
            "n_configured": total configured slots,
            "n_covered": slots observed in at least one run,
            "gaps": [{"pkt_type", "row", "slot", "name"}, ...] sorted by (pkt, row, slot)
        }
    """
    seen = set().union(*observed_per_run) if observed_per_run else set()
    gaps = [{"pkt_type": p, "row": r, "slot": s, "name": configured[(p, r, s)]}
            for (p, r, s) in sorted(configured) if (p, r, s) not in seen]
    return {"n_configured": len(configured),
            "n_covered": len(configured) - len(gaps), "gaps": gaps}


TRACE_SIZE_DEFAULT = 1 << 21


def write_patch_spec(patch_spec, path) -> Path:
    """Write patch spec JSON to file.

    Args:
        patch_spec: list of {col, row, tile_type, events}
        path: output file path

    Returns:
        Path object pointing to the written file
    """
    path = Path(path)
    path.write_text(json.dumps(patch_spec))
    return path


def runner_command(instr, trace_out, trace_size, inputs, outputs) -> str:
    """Build the stdin command line for bridge-trace-runner.

    Args:
        instr: path to instruction binary
        trace_out: output trace file path
        trace_size: maximum trace size in bytes
        inputs: list of input file paths
        outputs: list of output file paths

    Returns:
        Space-joined command line with shell-safe quoting
    """
    parts = ["--instr", str(instr), "--trace-out", str(trace_out),
             "--trace-size", str(trace_size)]
    for p in inputs:
        parts += ["--input", str(p)]
    for p in outputs:
        parts += ["--output", str(p)]
    return " ".join(quote(p) for p in parts)
