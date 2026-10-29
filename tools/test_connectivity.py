"""Unit tests for the pure logical-connectivity classifier."""
from types import SimpleNamespace

from inference.connectivity import (
    classify_connectivity, GROUNDED, OBSERVED_UNGROUNDED, UNOBSERVED, _tile)


def _edge(child, parent):
    return SimpleNamespace(child=child, parent=parent)


def test_tile_extracts_col_row():
    assert _tile("2|4|1|DMA_MM2S_0_START_TASK") == "2|4"
    assert _tile("1|1|3") == "1|1"


def test_grounded_when_both_fired_and_edge_present():
    pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    fired = {"2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6"}
    edges = [_edge("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    out = classify_connectivity(pairs, fired, edges)
    assert out == {("1|1", "2|4"): GROUNDED}


def test_observed_but_ungrounded_when_both_fired_no_edge():
    pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    fired = {"2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6"}
    out = classify_connectivity(pairs, fired, edges=[])
    assert out == {("1|1", "2|4"): OBSERVED_UNGROUNDED}


def test_unobserved_when_an_endpoint_did_not_fire():
    pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    fired = {"1|1|3|PORT_RUNNING_6"}          # col-2 DMA endpoint never fired
    out = classify_connectivity(pairs, fired, edges=[])
    assert out == {("1|1", "2|4"): UNOBSERVED}


def test_same_tile_pairs_are_skipped():
    # Two events on the same tile (different module) are NOT a cross-tile
    # conversation -> not in the connectivity report.
    pairs = [("1|2|1|DMA_MM2S_0_START_TASK", "1|2|0|INSTR_VECTOR")]
    out = classify_connectivity(pairs, fired={"1|2|1|DMA_MM2S_0_START_TASK",
                                              "1|2|0|INSTR_VECTOR"}, edges=[])
    assert out == {}


def test_grounded_wins_when_pair_seen_both_ways():
    # One candidate pair for a coupling grounds; another for the same tile pair
    # has an unfired endpoint. The coupling is grounded (at least one grounded).
    pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6"),
             ("2|4|1|DMA_S2MM_0_START_TASK", "1|1|3|PORT_RUNNING_7")]
    fired = {"2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6"}
    edges = [_edge("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    out = classify_connectivity(pairs, fired, edges)
    assert out == {("1|1", "2|4"): GROUNDED}


def test_same_tile_edge_does_not_leak_as_grounded():
    # A weave edge between two events on the SAME tile (cross-module, e.g. core
    # pkt0 -> memmod pkt1) is cross-domain, so weave can emit an edge for it. Its
    # tile projection is ("1|2","1|2") -- a same-tile pair the module must NOT
    # report. Guards the `all_pairs |= grounded` fold against same-tile leakage.
    edges = [_edge("1|2|1|DMA_MM2S_0_START_TASK", "1|2|0|INSTR_VECTOR")]
    out = classify_connectivity([], fired=set(), edges=edges)
    assert out == {}
