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

# The trace-event menu is derived COMPLETE from the toolchain table
# (trace_capture.load_event_ids) -- the engine knows every event the hardware can
# trace. Two views:
#   complete_menu() -- the universe: all events minus NONE (the disable/pad sentinel).
#   swept_menu()    -- what the sweep enumerates: measurement events first, infra
#                      events last, the always-on "stateful" flood-risk tier excluded
#                      (still in the universe, just kept out of co-traced batches so
#                      capture() never truncates).
# Tiers are toolchain-category classifications; see docs and _event_tier below.

_INFRA_PREFIXES = ("BROADCAST_", "USER_EVENT_", "INSTR_EVENT_", "GROUP_")
_STATEFUL_EXACT = {"TRUE", "ACTIVE", "DISABLED", "DEBUG_HALTED"}


def _event_tier(name: str) -> str:
    """Classify a (prefix-stripped) event name: measurement | infra | stateful.

    stateful = always-on level signals that flood the 2 MB trace buffer (TRUE,
    core enable/halt state, PORT_IDLE_*). Conservative prior; the HW seed-smoke
    promotes any further event that actually truncates. PORT_RUNNING/STALLED/TLAST
    are discrete measurement events, NOT stateful.
    """
    if name in _STATEFUL_EXACT or name.startswith("PORT_IDLE_"):
        return "stateful"
    if name == "TIMER_SYNC" or name.startswith(_INFRA_PREFIXES):
        return "infra"
    # Defensive: no RSVD/RESERVED entries exist for aieml today, but guard other
    # devices -- treat reserved ids as infra (low priority), never measurement.
    if name.upper().startswith("RSVD") or name.upper().startswith("RESERVED"):
        return "infra"
    return "measurement"


_COMPLETE_CACHE: "Dict[int, List[str]] | None" = None
_SWEPT_CACHE: "Dict[int, List[str]] | None" = None


def complete_menu() -> Dict[int, List[str]]:
    """The complete per-packet-type traceable event universe, toolchain-derived.
    All events minus NONE, ordered by event id. The engine's completeness claim."""
    global _COMPLETE_CACHE
    if _COMPLETE_CACHE is not None:
        return _COMPLETE_CACHE
    from trace_capture import load_event_ids, PKT_TO_TILE_TYPE  # lazy: header on demand
    menu: Dict[int, List[str]] = {}
    for pkt, tile_type in PKT_TO_TILE_TYPE.items():
        ids = load_event_ids(tile_type)
        menu[pkt] = sorted((n for n in ids if n != "NONE"), key=lambda n: ids[n])
    _COMPLETE_CACHE = menu
    return menu


def swept_menu() -> Dict[int, List[str]]:
    """The flood-safe default sweep menu: per packet type, measurement-tier names
    (ordered by id) then infra-tier names (ordered by id); the stateful tier is
    excluded (reachable via complete_menu, kept out of co-traced batches). This is
    what enumerate_configured_events draws from."""
    global _SWEPT_CACHE
    if _SWEPT_CACHE is not None:
        return _SWEPT_CACHE
    out: Dict[int, List[str]] = {}
    for pkt, names in complete_menu().items():
        meas = [n for n in names if _event_tier(n) == "measurement"]
        infra = [n for n in names if _event_tier(n) == "infra"]
        out[pkt] = meas + infra   # within each tier already id-ordered from complete_menu
    _SWEPT_CACHE = out
    return out


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
            for name in swept_menu().get(pkt, []):
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
