"""Frozen dataclasses mirroring the xdna-emu config dump JSON schema.

The JSON is produced by the Rust Tier-A/B config extractor (Task A1-B2).
Schema source: tools/config_extract/fixtures/add_one_using_dma.config.json.

Design notes:
- All dataclasses are frozen=True for immutability.
- PortRef is also frozen and hashable; it is used as dict keys in
  Reachability and as set elements in reaches_any.
- Optional fields (dma_channel on Port, shim_mux on TileDump) default to
  None; load_dump handles absent JSON keys with .get().
- event_port_selection entries may be null (JSON null -> Python None);
  load_dump produces a list of EventPortSelection | None.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Leaf types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PortRef:
    """A stream-switch port reference: uniquely identifies a port in the array.

    Physical identity for BFS: (col, row, port, dir).
    kind is descriptive metadata -- the same physical port may have a different
    kind string depending on which edge producer emitted the reference
    (inter_tile_dest uses direction labels; intra_tile_edges use port_type
    labels).  Do NOT key on kind for reachability; key on (col, row, port, dir).
    """
    col: int
    row: int
    port: int
    dir: str
    kind: str


@dataclass(frozen=True)
class RouteEdge:
    """A directed edge in the route graph between two PortRefs."""
    src: PortRef
    dst: PortRef
    kind: str  # "circuit" | "packet" | "inter_tile"


@dataclass(frozen=True)
class Port:
    """A stream-switch port on a tile (from the tile's port list)."""
    index: int
    dir: str
    kind: str
    packet: bool
    dma_channel: Optional[int] = None


@dataclass(frozen=True)
class EventPortSelection:
    """One slot of the event port selection register (non-null)."""
    slot: int
    port: int
    is_master: bool


@dataclass(frozen=True)
class DmaChannel:
    """A DMA channel descriptor on a tile."""
    index: int
    dir: str    # "mm2s" | "s2mm"
    start_bd: int


@dataclass(frozen=True)
class Bd:
    """A buffer descriptor on a tile."""
    id: int
    valid: bool
    use_next_bd: bool
    next_bd: int
    lock_acq_id: int
    lock_acq_value: int
    lock_rel_id: int
    lock_rel_value: int
    # base_addr (bytes) and length (bytes) were added in Tier E.  Older fixtures
    # lack them; default to 0 for backward compatibility (see _load_bd).
    base_addr: int = 0
    length: int = 0


@dataclass(frozen=True)
class Lock:
    """A lock with its initial value on a tile."""
    id: int
    value: int


@dataclass(frozen=True)
class ShimMux:
    """Shim-tile MUX configuration (present only on shim tiles)."""
    mm2s_slaves: tuple[Optional[int], ...]
    s2mm_masters: tuple[Optional[int], ...]


# ---------------------------------------------------------------------------
# Aggregate types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileDump:
    """Complete dump of a single tile's configuration."""
    col: int
    row: int
    kind: str  # "shim" | "memtile" | "compute"
    ports: tuple[Port, ...]
    event_port_selection: tuple[Optional[EventPortSelection], ...]
    dma_channels: tuple[DmaChannel, ...]
    bds: tuple[Bd, ...]
    locks: tuple[Lock, ...]
    shim_mux: Optional[ShimMux] = None


@dataclass(frozen=True)
class RouteGraph:
    """The directed route graph for the whole device."""
    edges: tuple[RouteEdge, ...]


@dataclass(frozen=True)
class ConfigDump:
    """Top-level config dump produced by the Rust extractor."""
    device: str
    route_graph: RouteGraph
    tiles: tuple[TileDump, ...]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_portref(d: dict) -> PortRef:
    return PortRef(
        col=d["col"],
        row=d["row"],
        port=d["port"],
        dir=d["dir"],
        kind=d["kind"],
    )


def _load_route_edge(d: dict) -> RouteEdge:
    return RouteEdge(
        src=_load_portref(d["src"]),
        dst=_load_portref(d["dst"]),
        kind=d["kind"],
    )


def _load_port(d: dict) -> Port:
    return Port(
        index=d["index"],
        dir=d["dir"],
        kind=d["kind"],
        packet=d["packet"],
        dma_channel=d.get("dma_channel"),
    )


def _load_eps(d: Optional[dict]) -> Optional[EventPortSelection]:
    if d is None:
        return None
    return EventPortSelection(
        slot=d["slot"],
        port=d["port"],
        is_master=d["is_master"],
    )


def _load_dma_channel(d: dict) -> DmaChannel:
    return DmaChannel(
        index=d["index"],
        dir=d["dir"],
        start_bd=d["start_bd"],
    )


def _load_bd(d: dict) -> Bd:
    return Bd(
        id=d["id"],
        valid=d["valid"],
        use_next_bd=d["use_next_bd"],
        next_bd=d["next_bd"],
        lock_acq_id=d["lock_acq_id"],
        lock_acq_value=d["lock_acq_value"],
        lock_rel_id=d["lock_rel_id"],
        lock_rel_value=d["lock_rel_value"],
        # Tolerate fixtures predating Tier E (base_addr/length absent -> 0).
        base_addr=d.get("base_addr", 0),
        length=d.get("length", 0),
    )


def _load_lock(d: dict) -> Lock:
    return Lock(id=d["id"], value=d["value"])


def _load_shim_mux(d: Optional[dict]) -> Optional[ShimMux]:
    if d is None:
        return None
    return ShimMux(
        mm2s_slaves=tuple(d["mm2s_slaves"]),
        s2mm_masters=tuple(d["s2mm_masters"]),
    )


def _load_tile(d: dict) -> TileDump:
    return TileDump(
        col=d["col"],
        row=d["row"],
        kind=d["kind"],
        ports=tuple(_load_port(p) for p in d["ports"]),
        event_port_selection=tuple(_load_eps(e) for e in d["event_port_selection"]),
        dma_channels=tuple(_load_dma_channel(c) for c in d["dma_channels"]),
        bds=tuple(_load_bd(b) for b in d["bds"]),
        locks=tuple(_load_lock(l) for l in d["locks"]),
        shim_mux=_load_shim_mux(d.get("shim_mux")),
    )


def load_dump(path: Path | str) -> ConfigDump:
    """Load a config dump JSON file produced by the Rust extractor.

    Tolerates null event_port_selection slots (JSON null -> None) and
    absent shim_mux (only present on shim tiles).
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    route_graph = RouteGraph(
        edges=tuple(_load_route_edge(e) for e in raw["route_graph"]["edges"]),
    )
    tiles = tuple(_load_tile(t) for t in raw["tiles"])
    return ConfigDump(
        device=raw["device"],
        route_graph=route_graph,
        tiles=tiles,
    )
