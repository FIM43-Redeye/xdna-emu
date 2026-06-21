"""Map a trace event name to its physical route-graph node (PortRef).

This is C2 in the inference-engine Plan 2 (#140).  The job is purely
positional: given an event name and the tile it was observed on, return
the PortRef (col, row, port, dir, kind) that the event is physically
monitoring.  Orientation (which node is the upstream source vs. downstream
sink in a chain) is NOT determined here -- that is C3's job via
reachability.

API
---
resolve_event_port(tile, event_name, dump) -> PortRef | None
    Map one trace event to its PortRef.  Returns None for events that do
    not correspond to a routable stream-switch port (LOCK_STALL, PERF_CNT_*,
    CONFLICT_*, etc.).

event_key(col, row, pkt, name) -> str
    Format a trace event into the canonical lookup key "col|row|pkt|name".
    Mirrors trace_join._key for use by callers that build the event index.

Event family rules
------------------
PORT_RUNNING_N
    N indexes tile.event_port_selection[N].  The EPS slot gives (port,
    is_master).  dir = "master" if is_master else "slave".  The kind is
    resolved from tile.ports -- the port at that (index, dir); falls back
    to the placeholder "port" if the ports list does not contain a matching
    entry.  If the EPS slot is None (not configured), returns None.

DMA_MM2S_{ch}_* on a shim tile
    Shim DMAs are mux-routed and do NOT have a kind="dma" SS port.  The
    mux maps each MM2S channel to a specific SS slave port (data enters the
    SS there = the source node).  Use tile.shim_mux.mm2s_slaves[ch] for the
    port index.  dir = "slave" (source in stream-switch terms).  Returns
    None if shim_mux is absent, the channel index is out of range, or the
    slot is None (channel not configured in this kernel).

DMA_S2MM_{ch}_* on a shim tile
    Same scheme: s2mm_masters[ch] gives the SS master port index (data exits
    SS to the DMA = the sink node).  dir = "master".

DMA_MM2S_{ch}_* on a memtile or compute tile
    These tiles have kind="dma" ports in their SS.  Find the Port with
    kind=="dma", dma_channel==ch, dir=="master" (MM2S = memory-to-stream =
    data leaves memory into the SS = master).

DMA_S2MM_{ch}_* on a memtile or compute tile
    kind=="dma", dma_channel==ch, dir=="slave" (S2MM = stream-to-memory =
    data enters memory from the SS = slave).

All other event names -> None.
"""

from __future__ import annotations

import re
from typing import Optional

from config_extract.dump_model import ConfigDump, Port, PortRef, TileDump

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# PORT_RUNNING_N (N = decimal slot index)
_RE_PORT_RUNNING = re.compile(r"^PORT_RUNNING_(\d+)$")

# DMA_(MM2S|S2MM)_{channel}_ ... rest of event name varies
_RE_DMA = re.compile(r"^DMA_(MM2S|S2MM)_(\d+)_")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kind_for(tile: TileDump, port_index: int, port_dir: str) -> str:
    """Return the kind string from tile.ports for (port_index, port_dir).

    Falls back to the placeholder "port" if no matching entry is found
    (reachability ignores kind so the placeholder never breaks routing).
    """
    for p in tile.ports:
        if p.index == port_index and p.dir == port_dir:
            return p.kind
    return "port"


def _resolve_shim_dma(
    tile: TileDump,
    direction: str,
    ch: int,
) -> Optional[PortRef]:
    """Resolve a shim DMA event to its stream-switch port via shim_mux.

    MM2S (memory-to-stream):
        data enters the SS at a *slave* port -> dir="slave".
        port index = shim_mux.mm2s_slaves[ch].

    S2MM (stream-to-memory):
        data exits the SS at a *master* port -> dir="master".
        port index = shim_mux.s2mm_masters[ch].

    Returns None if shim_mux is absent, ch is out of range, or the mux slot
    is None (channel not wired in this kernel).
    """
    if tile.shim_mux is None:
        return None

    if direction == "MM2S":
        slots = tile.shim_mux.mm2s_slaves
        port_dir = "slave"
    else:  # S2MM
        slots = tile.shim_mux.s2mm_masters
        port_dir = "master"

    if ch >= len(slots):
        return None
    port_index = slots[ch]
    if port_index is None:
        return None

    kind = _kind_for(tile, port_index, port_dir)
    return PortRef(col=tile.col, row=tile.row, port=port_index, dir=port_dir, kind=kind)


def _resolve_tile_dma(
    tile: TileDump,
    direction: str,
    ch: int,
) -> Optional[PortRef]:
    """Resolve a memtile/compute DMA event to its kind="dma" SS port.

    MM2S = memory-to-stream = data leaves memory into the SS = master port.
    S2MM = stream-to-memory = data enters memory from the SS = slave port.
    """
    port_dir = "master" if direction == "MM2S" else "slave"
    for p in tile.ports:
        if p.kind == "dma" and p.dma_channel == ch and p.dir == port_dir:
            return PortRef(
                col=tile.col,
                row=tile.row,
                port=p.index,
                dir=port_dir,
                kind="dma",
            )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def event_key(col: int, row: int, pkt: int, name: str) -> str:
    """Format a trace event as the canonical lookup key "col|row|pkt|name".

    Mirrors the trace_join._key helper.  Used by callers that build the
    trace-event index for C3 inference.
    """
    return f"{col}|{row}|{pkt}|{name}"


def resolve_event_port(
    tile: TileDump,
    event_name: str,
    dump: ConfigDump,
) -> Optional[PortRef]:
    """Map a trace event name to the PortRef it physically observes.

    Parameters
    ----------
    tile:       The tile the event was recorded on.
    event_name: The raw event name string from the trace (e.g. "PORT_RUNNING_0",
                "DMA_MM2S_0_START_TASK", "LOCK_STALL").
    dump:       The full ConfigDump (unused currently; reserved for future
                cross-tile lookups and kept for API stability).

    Returns
    -------
    A PortRef identifying the physical stream-switch port the event monitors,
    or None if the event does not correspond to a routable SS port.
    """
    # --- PORT_RUNNING_N ---
    m = _RE_PORT_RUNNING.match(event_name)
    if m:
        slot = int(m.group(1))
        if slot >= len(tile.event_port_selection):
            return None
        eps = tile.event_port_selection[slot]
        if eps is None:
            return None
        port_dir = "master" if eps.is_master else "slave"
        kind = _kind_for(tile, eps.port, port_dir)
        return PortRef(
            col=tile.col,
            row=tile.row,
            port=eps.port,
            dir=port_dir,
            kind=kind,
        )

    # --- DMA_MM2S_{ch}_* or DMA_S2MM_{ch}_* ---
    m = _RE_DMA.match(event_name)
    if m:
        direction = m.group(1)  # "MM2S" or "S2MM"
        ch = int(m.group(2))
        if tile.kind == "shim":
            return _resolve_shim_dma(tile, direction, ch)
        else:
            # memtile and compute: use kind="dma" ports from tile.ports
            return _resolve_tile_dma(tile, direction, ch)

    # --- Everything else (LOCK_STALL, PERF_CNT_*, CONFLICT_*, ...) ---
    return None
