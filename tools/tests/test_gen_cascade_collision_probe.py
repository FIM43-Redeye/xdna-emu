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


def test_alignment_guard_refuses_an_unaligned_label(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "label_addresses", lambda pkg: {".Lcascwait": 0x48E})
    with pytest.raises(SystemExit):
        g.assert_aligned(tmp_path)
    monkeypatch.setattr(g, "label_addresses", lambda pkg: {".Lcascwait": 0x490})
    g.assert_aligned(tmp_path)
