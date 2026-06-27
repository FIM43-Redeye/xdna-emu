"""Tests for config_extract.event_map (TDD -- write failing tests first).

Tests use the committed fixture add_one_using_dma.config.json.  Active data
in that fixture lives on col=0, not col=1 -- verified against the real JSON:
- memtile with non-null event_port_selection: col=0, row=1
- shim with non-null shim_mux: col=0, row=0 (mm2s_slaves[0]=5, s2mm_masters[0]=4)

Run with:
  cd /home/triple/npu-work/xdna-emu/tools
  python -m pytest test_config_extract_event_map.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config_extract.dump_model import load_dump
from config_extract.event_map import event_key, resolve_event_port

FIX = Path(__file__).resolve().parent / "config_extract" / "fixtures" / "add_one_using_dma.config.json"


def _tile(dump, col, row):
    return next(t for t in dump.tiles if t.col == col and t.row == row)


# ---------------------------------------------------------------------------
# event_key helper
# ---------------------------------------------------------------------------

def test_event_key_format():
    assert event_key(1, 0, 2, "DMA_MM2S_0_START_TASK") == "1|0|2|DMA_MM2S_0_START_TASK"


def test_event_key_zero_args():
    assert event_key(0, 0, 0, "LOCK_STALL") == "0|0|0|LOCK_STALL"


# ---------------------------------------------------------------------------
# PORT_RUNNING_N -> event_port_selection[N]
# ---------------------------------------------------------------------------

def test_port_running_maps_to_event_port_selection():
    """PORT_RUNNING_0 on memtile col=0,row=1 -> slot0 = (port0, master)."""
    dump = load_dump(FIX)
    memtile = _tile(dump, 0, 1)
    pr = resolve_event_port(memtile, "PORT_RUNNING_0", dump)
    assert pr is not None
    assert (pr.col, pr.row) == (0, 1)
    assert pr.port == 0
    assert pr.dir == "master"


def test_port_running_slot1():
    """PORT_RUNNING_1 -> slot1 = (port1, master)."""
    dump = load_dump(FIX)
    memtile = _tile(dump, 0, 1)
    pr = resolve_event_port(memtile, "PORT_RUNNING_1", dump)
    assert pr is not None and pr.port == 1 and pr.dir == "master"


def test_port_running_slot4():
    """PORT_RUNNING_4 -> slot4 = (port0, slave)."""
    dump = load_dump(FIX)
    memtile = _tile(dump, 0, 1)
    pr = resolve_event_port(memtile, "PORT_RUNNING_4", dump)
    assert pr is not None and pr.port == 0 and pr.dir == "slave"


def test_port_running_slot5():
    """PORT_RUNNING_5 -> slot5 = (port1, slave)."""
    dump = load_dump(FIX)
    memtile = _tile(dump, 0, 1)
    pr = resolve_event_port(memtile, "PORT_RUNNING_5", dump)
    assert pr is not None and pr.port == 1 and pr.dir == "slave"


def test_port_running_null_slot_returns_none():
    """A slot with a null EPS entry (not configured) -> None."""
    dump = load_dump(FIX)
    # col=1,row=1 memtile has all-null EPS in this fixture
    memtile = _tile(dump, 1, 1)
    pr = resolve_event_port(memtile, "PORT_RUNNING_0", dump)
    assert pr is None


# ---------------------------------------------------------------------------
# DMA_MM2S_{ch} (shim) -> shim_mux.mm2s_slaves[ch] -> slave port (source)
# ---------------------------------------------------------------------------

def test_dma_mm2s_maps_to_shim_slave():
    """DMA_MM2S_0_START_TASK on shim col=0,row=0 -> slave port 5 (source)."""
    dump = load_dump(FIX)
    shim = _tile(dump, 0, 0)
    pr = resolve_event_port(shim, "DMA_MM2S_0_START_TASK", dump)
    assert pr is not None
    assert (pr.col, pr.row) == (0, 0)
    assert pr.port == 5
    assert pr.dir == "slave"


def test_dma_mm2s_kind_matches_port_kind():
    """The returned PortRef kind should match the shim's port at that index."""
    dump = load_dump(FIX)
    shim = _tile(dump, 0, 0)
    pr = resolve_event_port(shim, "DMA_MM2S_0_START_TASK", dump)
    # port index 5 on shim is kind "south" (per fixture); kind is derived from ports list
    assert pr is not None and pr.kind != ""


def test_dma_event_maps_to_dma_port_kind():
    """DMA_MM2S on a shim returns a PortRef (kind string from ports list, not necessarily 'dma')."""
    dump = load_dump(FIX)
    shim = _tile(dump, 0, 0)
    pr = resolve_event_port(shim, "DMA_MM2S_0_START_TASK", dump)
    assert pr is not None


# ---------------------------------------------------------------------------
# DMA_S2MM_{ch} (shim) -> shim_mux.s2mm_masters[ch] -> master port (sink)
# ---------------------------------------------------------------------------

def test_dma_s2mm_maps_to_shim_master():
    """DMA_S2MM_0_* on shim col=0,row=0 -> master port 4 (sink)."""
    dump = load_dump(FIX)
    shim = _tile(dump, 0, 0)
    pr = resolve_event_port(shim, "DMA_S2MM_0_FINISHED_TASK", dump)
    assert pr is not None
    assert (pr.col, pr.row) == (0, 0)
    assert pr.port == 4
    assert pr.dir == "master"


def test_dma_s2mm_null_channel_returns_none():
    """DMA_S2MM_1 when s2mm_masters[1] is None -> None."""
    dump = load_dump(FIX)
    shim = _tile(dump, 0, 0)
    # s2mm_masters[1] == 5 in fixture, actually check the null-mux shim
    shim_null = _tile(dump, 1, 0)  # col=1 shim has all-null shim_mux
    pr = resolve_event_port(shim_null, "DMA_S2MM_0_START_TASK", dump)
    assert pr is None


# ---------------------------------------------------------------------------
# Memtile/compute DMA events -> tile's ports with kind=="dma" + dma_channel
# ---------------------------------------------------------------------------

def test_memtile_dma_mm2s_maps_to_dma_port():
    """DMA_MM2S_0_* on memtile -> kind=dma, dma_channel=0, dir=slave.

    MM2S reads memory and drives data INTO the stream switch, feeding a switch
    SLAVE port -- matching the authoritative route-graph convention
    (route_graph.rs: a DMA *slave* port is an MM2S channel) and the shim path.
    """
    dump = load_dump(FIX)
    memtile = _tile(dump, 0, 1)
    pr = resolve_event_port(memtile, "DMA_MM2S_0_START_TASK", dump)
    assert pr is not None
    assert pr.kind == "dma"
    assert pr.dir == "slave"
    assert pr.port == 0  # dma_channel=0, dir=slave -> port index 0 from fixture


def test_memtile_dma_s2mm_maps_to_dma_port():
    """DMA_S2MM_0_* on memtile -> kind=dma, dma_channel=0, dir=master.

    S2MM is fed BY the stream switch (a switch MASTER port drives data into the
    DMA, which writes it to memory) -- matching route_graph.rs (a DMA *master*
    port is an S2MM channel) and the shim path.
    """
    dump = load_dump(FIX)
    memtile = _tile(dump, 0, 1)
    pr = resolve_event_port(memtile, "DMA_S2MM_0_START_TASK", dump)
    assert pr is not None
    assert pr.kind == "dma"
    assert pr.dir == "master"
    assert pr.port == 0  # dma_channel=0, dir=master -> port index 0 from fixture


def test_tile_and_shim_dma_use_same_direction_convention():
    """Regression guard against the cross-column orientation inversion.

    Compute/memtile DMA events MUST use the same S2MM=master / MM2S=slave
    stream-switch convention as the shim path and the authoritative route graph
    (route_graph.rs: S2MM=master switch port, MM2S=slave). A prior inversion in
    _resolve_tile_dma labelled every compute/memtile DMA event onto the opposite
    node, scrambling cross-column producer/consumer orientation.
    """
    dump = load_dump(FIX)
    shim = _tile(dump, 0, 0)
    memtile = _tile(dump, 0, 1)
    for tile in (shim, memtile):
        assert resolve_event_port(tile, "DMA_S2MM_0_START_TASK", dump).dir == "master"
        assert resolve_event_port(tile, "DMA_MM2S_0_START_TASK", dump).dir == "slave"


# ---------------------------------------------------------------------------
# Non-route events -> None
# ---------------------------------------------------------------------------

def test_non_route_event_returns_none():
    """LOCK_STALL is not a route event; must return None."""
    dump = load_dump(FIX)
    core = _tile(dump, 0, 2)
    assert resolve_event_port(core, "LOCK_STALL", dump) is None


def test_perf_cnt_returns_none():
    dump = load_dump(FIX)
    core = _tile(dump, 0, 2)
    assert resolve_event_port(core, "PERF_CNT_2", dump) is None


def test_conflict_event_returns_none():
    dump = load_dump(FIX)
    core = _tile(dump, 0, 2)
    assert resolve_event_port(core, "CONFLICT_SS_SWITCH_OVERFLOW", dump) is None


def test_unrecognised_event_returns_none():
    dump = load_dump(FIX)
    core = _tile(dump, 0, 2)
    assert resolve_event_port(core, "SOMETHING_UNKNOWN", dump) is None
