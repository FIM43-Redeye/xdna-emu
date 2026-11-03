"""Tests for tools/gen_cascade_collision_probe.py (producer sizing, word, alignment guard)."""
import struct

import pytest

import gen_cascade_collision_probe as g


def test_scd_reads_counts_instructions_not_comments():
    asm = "  vmov x8, SCD\n  nop // SCD in a comment\n  vmov bml2, SCD ; vmov x0, x1\n"
    assert g.scd_reads(asm) == 2


def test_producer_writes_exactly_the_requested_words():
    text = g.producer_asm(5)
    assert text.count("vmov MCD, x0") == 5
    assert "crMCDEn" in text


def test_cascade_word_has_distinct_lanes():
    w = struct.unpack("<16I", g.cascade_word())
    assert len(set(w)) == 16 and w[0] == 0x5C0D0000


def test_cascade_catalogue_present_and_pairs_derive():
    assert g.CASCADE_KEYS == ["vmov_x_scd", "vmov_bm_scd", "vmov_bmh_scd"]
    for k in g.CASCADE_KEYS:
        assert k in g.cp.BY_KEY


def test_cascade_reads_join_the_catalogue_only_when_enabled(monkeypatch):
    # Start single-tile and restore the module afterwards, whichever test imported it first.
    monkeypatch.setattr(g.cp, "CASCADE", False)
    monkeypatch.setattr(g.cp, "CATALOGUE", [i for i in g.cp.CATALOGUE if i.key not in g.CASCADE_KEYS])
    single = len(g.cp.CATALOGUE)
    g.cp.enable_cascade()
    g.cp.enable_cascade()
    assert g.cp.CASCADE
    assert [i.key for i in g.cp.CATALOGUE[single:]] == g.CASCADE_KEYS


def test_case_is_a_two_core_cascade_topology_with_a_matching_producer():
    asm = "  vmov x8, SCD\n  vmov x9, SCD\n"
    kernel = {"in_size": 32, "out_size": 256}
    topo, prod = g.case_topology("k007", kernel, asm, bytes(32))
    assert topo.name == "cascol_k007"
    assert [(c.name, c.tile.col, c.tile.row) for c in topo.cores] == [("producer", 0, 3), ("consumer", 0, 2)]
    assert [str(c.source) for c in topo.cores] == ["producer.s", "consumer.s"]
    assert prod.count("vmov MCD, x0") == 2
    assert [(l.src, l.dst) for l in topo.links] == [("producer", "consumer")]
    consumer = topo.cores[1]
    assert [b.size_bytes for b in consumer.args] == [32, 256]
