#!/usr/bin/env python3
"""Self-owned NPU trace-capture engine for #140.

Takes a batch plan and produces correctly-labeled events.json per batch per run
on real hardware, reusing three audited primitives (register patcher, XRT
runner, in-tree decoder) and owning column-free exact labeling + N-run coverage.

See docs/superpowers/specs/2026-06-17-trace-capture-engine-design.md.
"""
from pathlib import Path
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
