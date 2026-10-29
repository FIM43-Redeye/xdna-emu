"""The planner: proven-gain MEASURE-NEXT batches.

Turns an ambiguity into a batch ONLY after first proving the batch adds a
separating/co-tracing gain -- never emit-then-discover. A fully-measured exact-offset
pair (range <= Q across all runs) returns NO_GAIN and goes straight to observational
degeneracy without burning a batch. Carries per-tile mode on the write side (cores can
use EVENT_PC mode 1; memmod/memtile/shim are always mode 0) and a Phase-0 seed sweep
over the static configured-event set.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from inference.facts import KB
from inference.verifier import offset_exact, ANCHOR
from inference.reachability import ReachabilityModel

MEASURE_NEXT = object()
NO_GAIN = object()


@dataclass
class Batch:
    tiles: Dict[str, List[str]] = field(default_factory=dict)
    mode_by_tile: Dict[str, int] = field(default_factory=dict)


def _tile_of(event_key: str) -> Tuple[str, str]:
    col, row, pkt, name = event_key.split("|")
    return f"{col}|{row}|{pkt}", name


def _default_mode(tile_key: str) -> int:
    # v1 always returns 0 (EVENT_TIME). Per-tile mode (cores can use EVENT_PC
    # mode 1) comes from the caller's mode_by_tile override, not from here.
    return 0


def plan_cotrace(a: str, b: str,
                 mode_by_tile: Optional[Dict[str, int]] = None) -> Batch:
    mode_by_tile = mode_by_tile or {}
    batch = Batch()
    for ek in (a, b):
        tile, name = _tile_of(ek)
        batch.tiles.setdefault(tile, [])
        if name not in batch.tiles[tile]:
            batch.tiles[tile].append(name)
        batch.mode_by_tile[tile] = mode_by_tile.get(tile, _default_mode(tile))
    return batch


def _co_traced(run_dirs: List[str], a: str, b: str, anchor_key: str) -> bool:
    return offset_exact(run_dirs, a, b, anchor_key) is not None


def propose_next(kb: KB, run_dirs: List[str], pair: Tuple[str, str],
                 model: ReachabilityModel, anchor_key: str = ANCHOR):
    a, b = pair
    if not _co_traced(run_dirs, a, b, anchor_key):
        # never co-traced: genuine co-trace gain, but only if reachable
        if model.can_separate(a, b) is False:
            return NO_GAIN
        return plan_cotrace(a, b)
    # fully measured, stable offset, but unoriented -> no batch can add orientation
    # (orientation comes from config, not measurement) -> observational degeneracy
    return NO_GAIN


def seed_plan(configured_events: List[str]) -> Batch:
    batch = Batch()
    for ek in configured_events:
        tile, name = _tile_of(ek)
        batch.tiles.setdefault(tile, [])
        if name not in batch.tiles[tile]:
            batch.tiles[tile].append(name)
        batch.mode_by_tile[tile] = _default_mode(tile)
    return batch
