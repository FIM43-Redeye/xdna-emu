"""The static reachability self-model: legality + gain inputs from the config.

enumerate_configured_events() is the termination domain (a generous per-tile-type
dataflow-event menu on active tiles -- a deliberate over-approximation pruned
empirically by the loop's never-fired constraints). candidate_pairs_from_dump()
reuses the generator's route-graph reachability as the orientation oracle, so
"gain" is proven from config (never emit-then-discover). legal_batch() defers to
configure_batch's real <=8-slot / valid-name checks.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from config_extract.generator import generate_ledger
from inference.planner import Batch

# Per-tile-type trace-event menu (pkt_type -> event names). Starter superset of
# the dataflow events; row decides tile-type on NPU1 (row0 shim, row1 memtile,
# row>=2 core+memmod). Full per-device event tables are follow-up.
_MENU: Dict[int, List[str]] = {
    # Shim: 9 entries -- deliberately exceeds the 8-slot hardware limit.
    # build_active_plan splits across batches; legal_batch / never_fired prune.
    2: ["DMA_MM2S_0_START_TASK", "DMA_MM2S_0_FINISHED_TASK",       # shim
        "DMA_S2MM_0_START_TASK", "DMA_S2MM_0_FINISHED_TASK",
        "DMA_S2MM_0_STREAM_STARVATION", "DMA_MM2S_1_START_TASK",
        "DMA_MM2S_1_FINISHED_TASK", "DMA_S2MM_1_START_TASK",
        "DMA_S2MM_1_FINISHED_TASK"],
    3: [f"PORT_RUNNING_{i}" for i in range(8)],                    # memtile
    0: ["PERF_CNT_2", "INSTR_VECTOR", "LOCK_STALL",               # core
        "MEMORY_STALL", "STREAM_STALL"],
    1: ["DMA_MM2S_0_START_TASK", "DMA_S2MM_0_START_TASK",         # memmod
        "EDGE_DETECTION_EVENT_0"],
}


def _pkts_for_row(row: int) -> List[int]:
    if row == 0:
        return [2]
    if row == 1:
        return [3]
    return [0, 1]


def _active_tiles(dump) -> set:
    """(rel_col, row) tiles referenced by the route graph (dump is relative-col)."""
    tiles = set()
    for e in dump.route_graph.edges:
        for node in (e.src, e.dst):
            tiles.add((node.col, node.row))
    return tiles


def enumerate_configured_events(dump, start_col: int) -> List[str]:
    # The dump is RELATIVE-col (generate_ledger subtracts start_col for tile
    # lookup); engine/decoder event keys are ABSOLUTE -- so add start_col here.
    out: List[str] = []
    for (col, row) in sorted(_active_tiles(dump)):
        for pkt in _pkts_for_row(row):
            for name in _MENU.get(pkt, []):
                out.append(f"{col + start_col}|{row}|{pkt}|{name}")
    # Deduplicate preserving order.
    seen, deduped = set(), []
    for k in out:
        if k not in seen:
            seen.add(k); deduped.append(k)
    return deduped


def candidate_pairs_from_dump(dump, configured_events: List[str],
                              start_col: int) -> List[Tuple[str, str]]:
    # Pass the full enumerated set as the candidate set (deliberate
    # over-approximation); the orientation oracle is config-derived,
    # so never_fired constraints prune what the loop can't ground.
    led = generate_ledger(dump, configured_events, start_col=start_col)
    # Ledger stores a=parent, b=child; the engine's candidate order is (child, parent).
    pairs = [(e["b"], e["a"]) for e in led["entries"]]
    seen, deduped = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p); deduped.append(p)
    return deduped


def legal_batch(batch: Batch) -> bool:
    from trace_capture import configure_batch
    try:
        configure_batch(batch.tiles)
    except (ValueError, KeyError):
        return False
    return True
