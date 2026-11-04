"""Tests for the vector-load timing probe generator (all four families) and its branch analysis."""

import gen_acc_timing_probe as base
import gen_vec_load_timing_probe as gen


class TestTable:
    def test_extends_the_accumulator_table_without_moving_it(self):
        assert gen.TABLE_BYTES[:len(base.TABLE_BYTES)] == base.TABLE_BYTES
        for name, off in base.TABLE_OFFSETS.items():
            assert gen.TABLE_OFFSETS[name] == off

    def test_every_vector_item_is_reachable(self):
        for name in ("OLD_i", "LDV_i", "LDV_f", "ADDR_OLD", "ADDR_NEW", "S_i"):
            for half in (0, 1):
                ptr, off = gen.table_ref(name, 32 * half)
                assert -1024 <= off <= base.isa_gen.MAX_VECTOR_OFFSET and off % 32 == 0, name


class TestFamily1:
    def test_pairs_overlap_in_the_halves_they_touch(self):
        for p, c in gen.valid_pairs():
            assert gen.LOAD_BY_NAME[p].halves & gen.READER_BY_NAME[c].halves

    def test_gather_readers_only_meet_loads_of_address_vectors(self):
        for p, c in gen.valid_pairs():
            if gen.READER_BY_NAME[c].gather:
                assert p in gen.GATHER_SAFE_LOADS

    def test_raw_point_places_load_and_reader_d_bundles_apart(self):
        p, c = gen.LOAD_BY_NAME["vldb_wl"], gen.READER_BY_NAME["vst_wl"]
        text = "\n".join(gen.point_lines(p, c, "RAW", 5, "i"))
        body = text[text.index(p.asm):text.index(c.asm)]
        assert body.count("nop") == 4

    def test_waw_partners_write_r_without_reading_it(self):
        for w in gen.WRITERS:
            dest, *srcs = w.asm.split(None, 1)[1].split(",")
            assert dest.strip() in ("x4", "wl4", "wh4"), w.asm
            assert not any(s.strip() in ("x4", "wl4", "wh4") for s in srcs), w.asm

    def test_records_fit_the_accumulator_analyzer(self):
        assert gen.RECORD_BYTES == base.RECORD_BYTES and gen.REF_DISTANCE == base.REF_DISTANCE


class TestBranch:
    def test_reader_distance_counts_the_delay_slots(self):
        # a load in the last delay slot and the target's first reader are one bundle apart
        assert gen.reader_distance(gen.DELAY_SLOTS, 0) == 1
        assert gen.reader_distance(0, 0) == gen.DELAY_SLOTS + 1

    def test_branching_points_align_the_target_and_fallthrough_ones_do_not(self):
        fam = gen.BRANCH_FAMILY_BY_NAME["vec_st"]
        taken = "\n".join(gen.branch_point_lines(fam, "j", 2, ".Lx"))
        assert "j #.Lx" in taken and ".p2align 4\n.Lx:" in taken
        none = "\n".join(gen.branch_point_lines(fam, "none", 2, ".Ly"))
        assert ".p2align" not in none and "j #" not in none

    def test_classification_and_threshold(self):
        slot = gen.BRANCH_SLOT_BYTES
        old = bytes(slot * gen.READERS_PER_POINT)
        new = bytes([1]) * len(old)
        # readers 0..1 OLD, 2..7 NEW, load at position 1: reader 2 is at distance 7
        rec = old[:2 * slot] + new[2 * slot:]
        entry = {"points": [{"pos": None, "record": 0}, {"pos": gen.REF_NEW_POSITION, "record": 1},
                            {"pos": 1, "record": 2}]}
        rows = gen.analyze_kernel(entry, old + new + rec)
        assert rows == [{"pos": 1, "cls": "OONNNNNN", "t": 7, "ok": True}]


class TestHazardAndTie:
    def test_hazard_points_come_in_threes(self):
        text, points = gen.hazard_kernel_text(gen.hazard_pairs()[:2])
        assert [p["mode"] for p in points] == ["first", "second", "both"] * 2

    def test_tie_writer_lands_with_the_load(self):
        load = dict(gen.TIE_LOADS)["vldb_wl"]
        writer = next(w for w in gen.TIE_WRITERS if w[0] == "vmov_wl4")
        lines = gen.tie_point_lines(load, writer, "x")
        i, j = lines.index("  " + load), lines.index("  " + writer[1])
        assert j - i == gen.TIE_LOAD_CYCLE - writer[2]
