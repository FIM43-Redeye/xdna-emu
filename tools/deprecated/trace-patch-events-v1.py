#!/usr/bin/env python3
"""Patch trace event slots in an already-compiled insts.bin.

Avoids recompilation when sweeping through different event configurations.
The stream switch routing (packet flows) is set up by aiecc.py and does not
change between event configurations -- only the Trace_Event0/Event1 register
writes need to be patched.

Usage:
    trace-patch-events.py <insts.bin> --manifest <manifest.json> \
        --events-json <events.json> --output <patched_insts.bin>

The events.json format matches trace-inject.py / trace-sweep.py:
    {
        "core_events": ["TRUE", "LOCK_STALL", ...],   // 8 slots
        "mem_events":  ["TRUE", "DMA_S2MM_0_START_TASK", ...]  // 8 slots
    }

Register layout:
    Core  module: Trace_Event0 at 0x340E0, Trace_Event1 at 0x340E4
    Mem   module: Trace_Event0 at 0x140E0, Trace_Event1 at 0x140E4
    NPU address:  (col << 25) | (row << 20) | tile_offset
"""

import argparse
import json
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# NPU instruction format: Write32
# ---------------------------------------------------------------------------
# Layout (24 bytes):
#   [0]:    opcode (0x00 = Write32)
#   [1-7]:  padding (zeros)
#   [8-15]: register address as u64 LE
#   [16-19]: value as u32 LE
#   [20-23]: size as u32 LE

def find_write32_offsets(insts: bytes, target_addr: int) -> list[int]:
    """Find all Write32 instructions targeting a given NPU address.

    Returns a list of byte offsets into insts pointing to the value field
    (the 4 bytes at offset +16 within the instruction).
    """
    results = []
    i = 0
    while i + 24 <= len(insts):
        if insts[i] == 0x00 and all(b == 0 for b in insts[i + 1:i + 8]):
            addr_lo = struct.unpack_from("<I", insts, i + 8)[0]
            addr_hi = struct.unpack_from("<I", insts, i + 12)[0]
            if addr_lo == target_addr and addr_hi == 0:
                results.append(i + 16)
        i += 4
    return results


def npu_address(col: int, row: int, tile_offset: int) -> int:
    """Encode a tile register address in NPU format."""
    return (col << 25) | (row << 20) | tile_offset


def pack_event_slots(event_ids: list[int]) -> tuple[int, int]:
    """Pack 8 event IDs into Trace_Event0 and Trace_Event1 register values.

    Trace_Event0 holds slots 0-3, Trace_Event1 holds slots 4-7.
    Each slot is one byte, little-endian order.
    """
    if len(event_ids) != 8:
        raise ValueError(f"Expected 8 event IDs, got {len(event_ids)}")
    event0 = int.from_bytes(bytes(event_ids[0:4]), "little")
    event1 = int.from_bytes(bytes(event_ids[4:8]), "little")
    return event0, event1


# ---------------------------------------------------------------------------
# Event name -> hardware ID resolution
# ---------------------------------------------------------------------------

# Lazy-loaded event enums from mlir-aie
_event_enums = {}


def get_event_enum(module_type: str):
    """Get the event enum for a given trace module type."""
    if module_type not in _event_enums:
        from aie.utils.trace.events import (  # type: ignore
            CoreEvent, MemEvent, MemTileEvent, ShimTileEvent,
        )
        _event_enums["core"] = CoreEvent
        _event_enums["mem"] = MemEvent
        _event_enums["memtile"] = MemTileEvent
        _event_enums["shim"] = ShimTileEvent
    return _event_enums[module_type]


def resolve_event_ids(names: list[str], module_type: str) -> list[int]:
    """Resolve event name strings to hardware IDs."""
    enum = get_event_enum(module_type)
    ids = []
    for name in names:
        name = name.strip().upper()
        if not name or name == "NONE":
            ids.append(0)
        else:
            try:
                ids.append(enum[name].value)
            except KeyError:
                raise ValueError(
                    f"Unknown {module_type} event '{name}'. "
                    f"Available: {', '.join(e.name for e in enum)}"
                )
    # Pad to 8 slots with NONE (0)
    while len(ids) < 8:
        ids.append(0)
    return ids[:8]


# ---------------------------------------------------------------------------
# Trace register offsets per module type
# ---------------------------------------------------------------------------

TRACE_OFFSETS = {
    "core": (0x340E0, 0x340E4),     # Core module Trace_Event0, Trace_Event1
    "mem": (0x140E0, 0x140E4),      # Memory module Trace_Event0, Trace_Event1
    "memtile": (0x340E0, 0x340E4),  # MemTile uses core trace offset
    "shim": (0x340E0, 0x340E4),     # Shim PL uses core trace offset
}


def patch_insts(
    insts: bytes,
    tiles: list[dict],
    events_config: dict,
) -> bytes:
    """Patch insts.bin with new event configurations for all traced tiles.

    Args:
        insts: Original insts.bin contents
        tiles: List of tile dicts from manifest, each with:
            - col, row: tile coordinates
            - trace_modules: list of module types being traced ("core", "mem")
        events_config: Dict mapping module type to event name lists:
            {"core_events": [...], "mem_events": [...], ...}

    Returns:
        Patched insts.bin bytes
    """
    patched = bytearray(insts)
    patch_count = 0

    for tile in tiles:
        col = tile["col"]
        row = tile["row"]
        modules = tile.get("trace_modules", ["core", "mem"])

        for module in modules:
            # Get event names for this module type
            events_key = f"{module}_events"
            if events_key not in events_config:
                continue

            event_ids = resolve_event_ids(events_config[events_key], module)
            event0_val, event1_val = pack_event_slots(event_ids)

            # Find and patch Trace_Event0
            offset0, offset1 = TRACE_OFFSETS[module]
            addr0 = npu_address(col, row, offset0)
            addr1 = npu_address(col, row, offset1)

            offsets0 = find_write32_offsets(insts, addr0)
            offsets1 = find_write32_offsets(insts, addr1)

            for off in offsets0:
                struct.pack_into("<I", patched, off, event0_val)
                patch_count += 1

            for off in offsets1:
                struct.pack_into("<I", patched, off, event1_val)
                patch_count += 1

    return bytes(patched), patch_count


def extract_tiles_from_manifest(manifest: dict) -> list[dict]:
    """Extract traced tile info from a trace-inject.py manifest.

    The manifest's 'tiles_traced' field lists tiles with coordinates and
    tile_type.  We map tile_type to trace module names for patching.
    """
    tiles = []
    for t in manifest.get("tiles_traced", []):
        col = t["col"]
        row = t["row"]
        tile_type = t.get("tile_type", "")
        if tile_type == "core" or row >= 2:
            modules = ["core", "mem"]
        elif tile_type == "memtile" or row == 1:
            modules = ["memtile"]
        elif tile_type == "shim" or row == 0:
            modules = ["shim"]
        else:
            modules = ["core", "mem"]
        tiles.append({"col": col, "row": row, "trace_modules": modules})
    return tiles


def main():
    parser = argparse.ArgumentParser(
        description="Patch trace event slots in compiled insts.bin",
    )
    parser.add_argument(
        "insts_bin", type=Path,
        help="Path to insts.bin to patch",
    )
    parser.add_argument(
        "--manifest", "-m", type=Path, required=True,
        help="Path to manifest.json from trace-inject.py",
    )
    parser.add_argument(
        "--events-json", "-e", type=Path, required=True,
        help="Path to events configuration JSON",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path for patched insts.bin (default: overwrite input)",
    )
    args = parser.parse_args()

    # Load inputs
    insts = args.insts_bin.read_bytes()
    manifest = json.loads(args.manifest.read_text())
    events_config = json.loads(args.events_json.read_text())

    # Extract tile info
    tiles = extract_tiles_from_manifest(manifest)
    if not tiles:
        print("Warning: no traced tiles found in manifest", file=sys.stderr)

    # Patch
    patched, count = patch_insts(insts, tiles, events_config)

    # Write output
    out_path = args.output or args.insts_bin
    out_path.write_bytes(patched)
    print(f"Patched {count} register writes across {len(tiles)} tiles -> {out_path}")


if __name__ == "__main__":
    main()
