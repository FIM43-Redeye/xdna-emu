"""Tests for the collision probe generator and its analysis."""

import os

import pytest

import analyze_collision_probe as an
import gen_collision_probe as gen

needs_checker = pytest.mark.skipif(
    not os.path.exists(gen.CHECKER) or not os.path.exists(gen.LLVM_MC),
    reason="needs target/debug/collision-check and llvm-mc")


class TestLayout:
    def test_accumulator_stores_stay_inside_the_vst_immediate_range(self):
        for off in (gen.A_OFF, gen.B_OFF):
            assert -128 <= off and off + 96 <= 127

    def test_record_regions_do_not_overlap(self):
        spans = [(gen.A_OFF + gen.REC_BIAS, 128), (gen.B_OFF + gen.REC_BIAS, 128),
                 (gen.WLO_OFF + gen.REC_BIAS - gen.REC_BIAS, 32)]
        assert gen.A_OFF + gen.REC_BIAS == 0 and gen.B_OFF + gen.REC_BIAS == 128
        assert gen.WLO_OFF + gen.REC_BIAS == 256 and gen.WHI_OFF + gen.REC_BIAS == 288
        assert 288 + 32 == gen.RECORD_BYTES

    def test_every_table_item_is_reachable(self):
        for name in gen.TABLE_OFFSETS:
            if name.startswith("Q"):
                continue
            for half in (0, 1):
                ptr, off = gen.table_ref(name, 32 * half)
                assert -1024 <= off <= gen.isa_gen.MAX_VECTOR_OFFSET and off % 32 == 0


class TestAllocation:
    def test_separate_registers_never_alias(self):
        for a in gen.CATALOGUE:
            for b in gen.CATALOGUE:
                ra, rb = gen.allocate(a, b, "sep")
                used = list(ra.s) + list(rb.s) + [x for x in (ra.d, rb.d) if x is not None]
                assert len(used) == len(set(used)), (a.key, b.key)
                assert all(0 <= x < gen.NUM_PAT for x in used)

    def test_srcshare_reads_one_register_twice(self):
        a, b = gen.BY_KEY["vadd_i"], gen.BY_KEY["vmov_cm"]
        ra, rb = gen.allocate(a, b, "srcshare")
        assert rb.s[0] == ra.s[0]

    def test_dstshare_targets_one_register(self):
        ra, rb = gen.allocate(gen.BY_KEY["vmul_i"], gen.BY_KEY["vclr_cm"], "dstshare")
        assert ra.d == rb.d
        assert gen.allocate(gen.BY_KEY["vst_amll"], gen.BY_KEY["vmov_cm"], "dstshare") is None

    def test_zero_register_is_outside_the_pattern_pool(self):
        assert gen.ZERO_REG >= gen.NUM_PAT


class TestWitness:
    def test_window_brackets_the_collision_and_clears_the_pair(self):
        for g in range(0, 9):
            for cyc in range(0, 9):
                w = gen.witness_window(0, g, [cyc])
                if w is None:
                    continue
                c0, lo = w
                assert c0 not in (0, g)
                assert c0 <= cyc < lo          # a stall at the collision falls between C and the read
                assert lo > g                  # the reads follow the pair
                assert lo - c0 == gen.WITNESS_FLIP - 1


class TestDerivation:
    RES = {
        "a": [("CM_RM_PORT", 2)], "b": [("CM_RM_PORT", 0)], "c": [("CM_WA_PORT", 5)],
        "d": [("CM_WM_PORT", 5)],
    }

    def test_known_shapes_are_derived_from_reservations(self, monkeypatch):
        cat = [gen.Ins(k, "nop", None) for k in "abcd"]
        monkeypatch.setattr(gen, "CATALOGUE", cat)
        found = {(c["a"], c["b"], c["g"]): c["units"] for c in gen.derive_collisions(self.RES)}
        assert found[("a", "b", 2)] == ["CM_RM_PORT"]
        assert found[("c", "d", 0)] == ["CM_WA_PORT~CM_WM_PORT"]     # two write ports, one cycle
        assert ("b", "a", 2) not in found                            # B-first would be g = -2

    @needs_checker
    def test_catalogue_matches_the_itineraries(self):
        res = gen.reservations_of_catalogue()
        assert ("CM_RM_PORT", 0) in res["vmov_cm"] and ("CM_RM_PORT", 2) in res["vadd_i"]
        assert ("CM_WA_PORT", 4) in res["vclr_cm"] and ("CM_WA_PORT", 5) in res["vaddmac_f"]
        colls = {(c["a"], c["b"], c["g"]) for c in gen.derive_collisions(res)}
        assert ("vadd_i", "vmov_cm", 2) in colls                     # CM_RM_PORT read collision
        assert ("vaddmac_f", "vclr_cm", 1) in colls                  # CM_WA_PORT write collision
        assert ("vlda_ups", "vaddmac_f", 3) in colls                 # same register, two ports


class TestDescribe:
    def test_exact_candidate_wins_over_a_partial_one(self):
        out = bytes(range(8))
        assert an.describe(out, {"alone": out, "init": bytes(8)}) == "alone"

    def test_partial_matches_are_tallied_by_word(self):
        out = bytes([1, 0, 0, 0, 2, 0, 0, 0])
        cands = {"alone": bytes([1, 0, 0, 0, 9, 0, 0, 0]), "init": bytes([7, 0, 0, 0, 2, 0, 0, 0])}
        assert an.describe(out, cands) == "mixed(alone:1/2,init:1/2)"

    def test_unexplained_when_nothing_matches(self):
        assert an.describe(bytes(4), {"alone": bytes([1, 0, 0, 0])}) == "unexplained[1w]"
