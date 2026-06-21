"""Tests for config_extract.generator (TDD -- write failing tests first).

Soundness framing (2026-06-21, Maya decision B): the SS-only route graph is
the truth predicate.  We do NOT assert any hand-authored edge count; we assert
the SOUNDNESS PROPERTIES:

1. Every emitted edge corresponds to a genuinely route-reachable
   (parent_node reaches child_node) fired-event pair -- and ONLY those.
2. Co-firing pairs that are NOT reachable are declined (no spurious edges).
3. Every emitted cite starts with "route:" (provenance is sound for C4).

Run with:
  cd /home/triple/npu-work/xdna-emu/tools
  python -m pytest test_config_extract_generator.py -v

Col-offset note: the 11 fired event keys use col=1 (decoder / kernel space,
where the kernel was loaded at col offset 1 on the NPU).  The fixture tiles
live at col=0.  The generator must accept a start_col parameter (default=1
for this fixture) and subtract it before looking up tiles.  See the
col-reconciliation section in the module docstring of generator.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config_extract.dump_model import load_dump
from config_extract.generator import generate_ledger
from config_extract.reachability import Reachability
from config_extract.event_map import resolve_event_port

FIX = Path(__file__).resolve().parent / "config_extract" / "fixtures" / "add_one_using_dma.config.json"

# The 11 fired event keys from the add_one_using_dma Plan-1 HW run.
# Keys are in col=1 (decoder space); dump tiles are at col=0 (start_col=1).
FIRED = [
    "1|0|2|DMA_MM2S_0_FINISHED_TASK",
    "1|0|2|DMA_MM2S_0_START_TASK",
    "1|0|2|DMA_S2MM_0_FINISHED_TASK",
    "1|0|2|DMA_S2MM_0_START_TASK",
    "1|0|2|DMA_S2MM_0_STREAM_STARVATION",
    "1|1|3|PORT_RUNNING_0",
    "1|1|3|PORT_RUNNING_1",
    "1|1|3|PORT_RUNNING_4",
    "1|1|3|PORT_RUNNING_5",
    "1|2|0|LOCK_STALL",
    "1|2|0|PERF_CNT_2",
]

# start_col=1: decoder-space col minus this offset = dump-tile col
START_COL = 1


# ---------------------------------------------------------------------------
# Helper: build reachability from the fixture dump (reference oracle)
# ---------------------------------------------------------------------------

def _load_reach(dump):
    return Reachability(dump.route_graph.edges)


def _parse_key(key: str):
    """Parse 'col|row|pkt|name' -> (col, row, pkt, name)."""
    parts = key.split("|", 3)
    return int(parts[0]), int(parts[1]), int(parts[2]), parts[3]


def _resolve(dump, key: str, start_col: int = START_COL):
    """Resolve an event key to its PortRef using the dump, with col offset."""
    fcol, frow, _fpkt, name = _parse_key(key)
    dump_col = fcol - start_col
    tile_idx = {(t.col, t.row): t for t in dump.tiles}
    tile = tile_idx.get((dump_col, frow))
    if tile is None:
        return None
    return resolve_event_port(tile, name, dump)


# ---------------------------------------------------------------------------
# Soundness test 1: every emitted edge is route-reachable
# ---------------------------------------------------------------------------

class TestGeneratorSoundness:
    def test_all_emitted_edges_are_route_reachable(self):
        """Every (a=child, b=parent) in the ledger must satisfy
        reachable(parent_node, child_node) in the SS route graph.

        This is the primary soundness invariant: the generator must NEVER emit
        an edge that the route graph does not justify.
        """
        dump = load_dump(FIX)
        reach = _load_reach(dump)
        led = generate_ledger(dump, FIRED, start_col=START_COL)

        route_entries = [e for e in led["entries"] if e["kind"] == "route"]

        for e in route_entries:
            child_key = e["a"]
            parent_key = e["b"]
            parent_node = _resolve(dump, parent_key)
            child_node = _resolve(dump, child_key)
            assert parent_node is not None, (
                f"Emitted edge has unresolvable parent {parent_key!r}"
            )
            assert child_node is not None, (
                f"Emitted edge has unresolvable child {child_key!r}"
            )
            assert reach.reachable(parent_node, child_node), (
                f"Emitted edge ({child_key!r}, {parent_key!r}) is NOT "
                f"route-reachable -- this is a spurious (false) edge"
            )

    def test_only_resolvable_fired_pairs_can_be_emitted(self):
        """Events that do not resolve to a PortRef (LOCK_STALL, PERF_CNT_2)
        must never appear in any emitted edge.
        """
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)

        non_route_events = {"LOCK_STALL", "PERF_CNT_2"}

        for e in led["entries"]:
            child_name = e["a"].split("|", 3)[-1]
            parent_name = e["b"].split("|", 3)[-1]
            assert child_name not in non_route_events, (
                f"Non-route event {child_name!r} appeared as child in edge {e}"
            )
            assert parent_name not in non_route_events, (
                f"Non-route event {parent_name!r} appeared as parent in edge {e}"
            )

    def test_emitted_edges_only_from_fired_events(self):
        """The generator must not emit edges for events outside fired_event_keys."""
        dump = load_dump(FIX)
        fired_set = set(FIRED)
        led = generate_ledger(dump, FIRED, start_col=START_COL)

        for e in led["entries"]:
            assert e["a"] in fired_set, (
                f"Edge child {e['a']!r} is not in fired_event_keys"
            )
            assert e["b"] in fired_set, (
                f"Edge parent {e['b']!r} is not in fired_event_keys"
            )

    def test_no_self_edges(self):
        """An event cannot be its own parent (no self-loops)."""
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)

        for e in led["entries"]:
            assert e["a"] != e["b"], f"Self-edge emitted for {e['a']!r}"


# ---------------------------------------------------------------------------
# Soundness test 2: co-firing pairs that are NOT reachable are declined
# ---------------------------------------------------------------------------

class TestGeneratorDeclines:
    def test_declines_cofire_pr1_parent_pr0(self):
        """PORT_RUNNING_1 co-fires with PORT_RUNNING_0 (broadcast), but their
        SS nodes are not reachable from one another.  The generator must NOT
        emit config_path(child=PR_1, parent=PR_0).
        """
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        edges = {(e["a"], e["b"]) for e in led["entries"] if e["kind"] == "route"}
        assert ("1|1|3|PORT_RUNNING_1", "1|1|3|PORT_RUNNING_0") not in edges, (
            "Generator emitted a co-firing edge PR_1->PR_0 that is not "
            "justified by SS route reachability"
        )

    def test_declines_cofire_pr4_parent_pr0(self):
        """PORT_RUNNING_4 is a slave port on the memtile; PR_0 is a master port.
        No SS path goes master-0 -> slave-0 (that would be a U-turn inside the
        same tile, which the SS does not form).  Decline.
        """
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        edges = {(e["a"], e["b"]) for e in led["entries"] if e["kind"] == "route"}
        assert ("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0") not in edges, (
            "Generator emitted a co-firing edge PR_4->PR_0 that is not "
            "justified by SS route reachability"
        )

    def test_declines_cofire_pr5_parent_pr0(self):
        """Same rationale as PR_4/PR_0: slave-1 is not reachable from master-0."""
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        edges = {(e["a"], e["b"]) for e in led["entries"] if e["kind"] == "route"}
        assert ("1|1|3|PORT_RUNNING_5", "1|1|3|PORT_RUNNING_0") not in edges, (
            "Generator emitted a co-firing edge PR_5->PR_0 that is not "
            "justified by SS route reachability"
        )

    def test_declines_reverse_of_emitted_edges(self):
        """For each genuinely emitted edge (a, b), the reverse (b, a) must NOT
        also be emitted (no symmetric back-edges in a directed acyclic dataflow).

        On a cyclic route graph this could happen, but the add_one path is
        feedforward -- shim→memtile→compute.  Verify no reversal.
        """
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        edges = {(e["a"], e["b"]) for e in led["entries"] if e["kind"] == "route"}
        for a, b in list(edges):
            assert (b, a) not in edges, (
                f"Both ({a!r},{b!r}) and its reverse ({b!r},{a!r}) are emitted"
            )


# ---------------------------------------------------------------------------
# Soundness test 3: cite provenance
# ---------------------------------------------------------------------------

class TestGeneratorCite:
    def test_all_cites_start_with_route_prefix(self):
        """Every route entry's cite must begin with 'route:' so C4 can
        distinguish generator-produced cites from hand-authored ones.
        """
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        for e in led["entries"]:
            if e["kind"] == "route":
                assert e["cite"].startswith("route:"), (
                    f"cite {e['cite']!r} does not start with 'route:'"
                )

    def test_cites_are_unique(self):
        """Each cite must be distinct -- the ledger uses cite as a primary key."""
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        cites = [e["cite"] for e in led["entries"]]
        assert len(cites) == len(set(cites)), (
            f"Duplicate cites found: {[c for c in cites if cites.count(c) > 1]}"
        )

    def test_ledger_has_comment(self):
        """The top-level dict must have a '_comment' key."""
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        assert "_comment" in led
        assert isinstance(led["_comment"], str) and len(led["_comment"]) > 0

    def test_ledger_has_entries_list(self):
        """The top-level dict must have an 'entries' key that is a list."""
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        assert "entries" in led
        assert isinstance(led["entries"], list)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestGeneratorSchema:
    def test_entries_have_required_fields(self):
        """Each entry must have cite, a, b, kind."""
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        for e in led["entries"]:
            for field in ("cite", "a", "b", "kind"):
                assert field in e, f"Entry missing {field!r}: {e}"

    def test_entries_kind_is_route(self):
        """All generated entries have kind='route' (generator only emits route)."""
        dump = load_dump(FIX)
        led = generate_ledger(dump, FIRED, start_col=START_COL)
        for e in led["entries"]:
            assert e["kind"] == "route", f"Unexpected kind {e['kind']!r} in {e}"


# ---------------------------------------------------------------------------
# Empty-input edge case
# ---------------------------------------------------------------------------

class TestGeneratorEdgeCases:
    def test_empty_fired_keys_returns_empty_ledger(self):
        dump = load_dump(FIX)
        led = generate_ledger(dump, [], start_col=START_COL)
        assert led["entries"] == []

    def test_all_non_route_fired_keys_returns_empty(self):
        """If all fired events are non-route (LOCK_STALL etc.) -> no edges."""
        dump = load_dump(FIX)
        led = generate_ledger(dump, ["1|2|0|LOCK_STALL", "1|2|0|PERF_CNT_2"],
                              start_col=START_COL)
        assert led["entries"] == []

    def test_single_event_no_pairs(self):
        """One event -> no ordered pairs -> empty ledger."""
        dump = load_dump(FIX)
        led = generate_ledger(dump, ["1|0|2|DMA_MM2S_0_START_TASK"],
                              start_col=START_COL)
        assert led["entries"] == []
