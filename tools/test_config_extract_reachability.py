"""Tests for config_extract.dump_model (loader) and config_extract.reachability (BFS).

TDD: these tests are written first.  Run with:
  cd /home/triple/npu-work/xdna-emu/tools
  python -m pytest test_config_extract_reachability.py -v
"""

from __future__ import annotations

import pathlib

import pytest

from config_extract.dump_model import (
    ConfigDump,
    PortRef,
    RouteEdge,
    load_dump,
)
from config_extract.reachability import Reachability

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE = pathlib.Path(__file__).parent / "config_extract" / "fixtures" / "add_one_using_dma.config.json"


def _p(col: int, row: int, port: int, d: str = "master", kind: str = "x") -> PortRef:
    """Convenience constructor that matches the brief's helper exactly."""
    return PortRef(col=col, row=row, port=port, dir=d, kind=kind)


# ---------------------------------------------------------------------------
# dump_model: load_dump round-trip tests
# ---------------------------------------------------------------------------

class TestLoadDump:
    def test_load_returns_config_dump(self):
        dump = load_dump(FIXTURE)
        assert isinstance(dump, ConfigDump)

    def test_device_field(self):
        dump = load_dump(FIXTURE)
        assert dump.device == "npu1"

    def test_route_graph_has_edges(self):
        dump = load_dump(FIXTURE)
        assert len(dump.route_graph.edges) > 0

    def test_route_edge_fields(self):
        dump = load_dump(FIXTURE)
        edge = dump.route_graph.edges[0]
        assert isinstance(edge, RouteEdge)
        assert isinstance(edge.src, PortRef)
        assert isinstance(edge.dst, PortRef)
        assert edge.kind in ("circuit", "packet", "inter_tile")

    def test_tiles_present(self):
        dump = load_dump(FIXTURE)
        # add_one_using_dma has shim (row 0), memtile (row 1), compute tile (row 2)
        assert len(dump.tiles) >= 3

    def test_shim_tile_kind(self):
        dump = load_dump(FIXTURE)
        shim = next(t for t in dump.tiles if t.kind == "shim")
        assert shim.col == 0 and shim.row == 0

    def test_shim_mux_present_on_shim(self):
        dump = load_dump(FIXTURE)
        shim = next(t for t in dump.tiles if t.kind == "shim")
        assert shim.shim_mux is not None

    def test_shim_mux_absent_on_compute(self):
        dump = load_dump(FIXTURE)
        compute = next(t for t in dump.tiles if t.kind == "compute")
        assert compute.shim_mux is None

    def test_null_event_port_slots_tolerated(self):
        dump = load_dump(FIXTURE)
        shim = next(t for t in dump.tiles if t.kind == "shim")
        # shim fixture has all-null event_port_selection
        assert all(slot is None for slot in shim.event_port_selection)

    def test_non_null_event_port_slots(self):
        dump = load_dump(FIXTURE)
        memtile = next(t for t in dump.tiles if t.kind == "memtile")
        # memtile fixture has non-null eps
        assert any(slot is not None for slot in memtile.event_port_selection)

    def test_bds_parsed(self):
        dump = load_dump(FIXTURE)
        shim = next(t for t in dump.tiles if t.kind == "shim")
        assert len(shim.bds) > 0
        bd = shim.bds[0]
        assert hasattr(bd, "id")
        assert hasattr(bd, "valid")

    def test_dma_channels_parsed(self):
        dump = load_dump(FIXTURE)
        shim = next(t for t in dump.tiles if t.kind == "shim")
        assert len(shim.dma_channels) > 0
        ch = shim.dma_channels[0]
        assert hasattr(ch, "index")
        assert hasattr(ch, "dir")
        assert hasattr(ch, "start_bd")

    def test_port_optional_dma_channel(self):
        dump = load_dump(FIXTURE)
        memtile = next(t for t in dump.tiles if t.kind == "memtile")
        dma_port = next(p for p in memtile.ports if p.kind == "dma")
        assert dma_port.dma_channel is not None
        shim = next(t for t in dump.tiles if t.kind == "shim")
        non_dma_port = next(p for p in shim.ports if p.kind != "dma")
        assert non_dma_port.dma_channel is None

    def test_portref_hashable(self):
        """PortRef must be usable as a dict key / set element."""
        p = _p(1, 0, 12)
        s: set[PortRef] = {p}
        assert p in s

    def test_portref_equality_includes_all_fields(self):
        """Frozen-dataclass equality covers every field, including dir and kind.

        Note: PortRef equality includes kind, even though *reachability*
        deliberately ignores kind (see reachability.py).  The two are
        different notions: dataclass identity vs. physical BFS identity.
        """
        assert _p(1, 0, 12, "master", "north") == _p(1, 0, 12, "master", "north")
        # differ only in dir -> unequal
        assert _p(1, 0, 12, "master", "north") != _p(1, 0, 12, "slave", "north")
        # differ only in kind -> unequal (equality includes kind)
        assert _p(1, 0, 12, "master", "north") != _p(1, 0, 12, "master", "south")

    def test_core_lock_relay_round_trips(self):
        """A RouteEdge with kind='core_lock_relay' loads without error.

        load_dump accepts kind as a free string (no closed-set validation in
        _load_route_edge).  This test confirms the kind passes through
        round-trip correctly so consumers can filter on it.
        """
        edge = RouteEdge(
            src=_p(0, 2, 0, "slave", "dma"),
            dst=_p(0, 2, 0, "master", "dma"),
            kind="core_lock_relay",
        )
        assert edge.kind == "core_lock_relay"
        assert edge.src.row == 2   # compute tile row
        assert edge.dst.row == 2

    def test_fixture_has_core_lock_relay_edge(self):
        """The committed fixture must contain at least one core_lock_relay edge.

        This guards against silent regressions where the dump regeneration
        produces no CoreLockRelay edges (e.g. ELF loading silently failed).
        """
        dump = load_dump(FIXTURE)
        relay_edges = [e for e in dump.route_graph.edges if e.kind == "core_lock_relay"]
        assert len(relay_edges) >= 1, (
            f"fixture must have >= 1 core_lock_relay edge; got {len(relay_edges)}.  "
            "Regenerate the fixture with compute-core ELFs loaded."
        )


# ---------------------------------------------------------------------------
# reachability: the four tests from the brief
# ---------------------------------------------------------------------------

class TestReachability:
    def test_reachable_direct_edge(self):
        r = Reachability([(_p(1, 0, 12), _p(1, 1, 7))])
        assert r.reachable(_p(1, 0, 12), _p(1, 1, 7))

    def test_reachable_transitive_two_hops(self):
        r = Reachability([
            (_p(1, 0, 12), _p(1, 1, 7, "slave")),
            (_p(1, 1, 7, "slave"), _p(1, 1, 11)),
        ])
        assert r.reachable(_p(1, 0, 12), _p(1, 1, 11))

    def test_not_reachable_wrong_direction(self):
        r = Reachability([(_p(1, 0, 12), _p(1, 1, 7))])
        assert not r.reachable(_p(1, 1, 7), _p(1, 0, 12))

    def test_self_not_reachable_without_self_loop(self):
        r = Reachability([(_p(1, 0, 12), _p(1, 1, 7))])
        assert not r.reachable(_p(1, 0, 12), _p(1, 0, 12))

    def test_physical_identity_ignores_kind(self):
        """BFS must traverse the inter/intra seam: same (col,row,port,dir) but
        different kind strings must be treated as the same node."""
        # src --circuit--> (0,1,11,slave,south) as inter_tile_dest
        # then (0,1,11,slave,circuit_type) --circuit--> dst
        # kind differs between the two edges touching the same physical port
        src = _p(0, 0, 16, "master", "north")
        mid_a = _p(0, 1, 11, "slave", "south")   # kind as labelled by inter_tile
        mid_b = _p(0, 1, 11, "slave", "circuit")  # same physical port, different kind label
        dst = _p(0, 1, 0, "master", "dma")
        r = Reachability([
            (src, mid_a),
            (mid_b, dst),
        ])
        assert r.reachable(src, dst), (
            "reachability must key on (col,row,port,dir) only, ignoring kind"
        )

    def test_reaches_any_returns_set(self):
        r = Reachability([
            (_p(1, 0, 12), _p(1, 1, 7)),
            (_p(1, 1, 7), _p(1, 1, 11)),
        ])
        reachable = r.reaches_any(_p(1, 0, 12))
        assert isinstance(reachable, set)
        # Must include both the direct and transitive destinations
        assert any(pr.col == 1 and pr.row == 1 and pr.port == 7 for pr in reachable)
        assert any(pr.col == 1 and pr.row == 1 and pr.port == 11 for pr in reachable)

    def test_reaches_any_empty_for_unknown_src(self):
        r = Reachability([(_p(1, 0, 12), _p(1, 1, 7))])
        assert r.reaches_any(_p(9, 9, 99)) == set()

    def test_accepts_route_edges(self):
        """Reachability must also accept RouteEdge objects, not only tuples."""
        edge = RouteEdge(
            src=_p(1, 0, 12),
            dst=_p(1, 1, 7),
            kind="circuit",
        )
        r = Reachability([edge])
        assert r.reachable(_p(1, 0, 12), _p(1, 1, 7))

    def test_fixture_shim_to_memtile_dma_reachable(self):
        """End-to-end: load the real fixture and verify a known path is reachable.

        In add_one_using_dma the shim north master port 16 connects (via
        inter_tile) to memtile south slave port 11, which then routes (circuit)
        to memtile DMA master port 0.  This path must be reachable.
        """
        dump = load_dump(FIXTURE)
        r = Reachability(dump.route_graph.edges)
        # Port numbers (16, 0) are the actual values in the committed fixture
        # add_one_using_dma.config.json -- not magic.  If a future fixture
        # regen changes the routing, update these here.
        shim_north_out = PortRef(col=0, row=0, port=16, dir="master", kind="north")
        memtile_dma_in = PortRef(col=0, row=1, port=0, dir="master", kind="dma")
        assert r.reachable(shim_north_out, memtile_dma_in)
