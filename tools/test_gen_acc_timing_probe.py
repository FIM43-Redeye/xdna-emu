"""Tests for the accumulator timing probe: generator, record analysis and TableGen fit."""

import os
import subprocess
import tempfile

import pytest

import analyze_acc_timing as an
import fit_acc_timing as fit
import gen_acc_timing_probe as gen


class TestCatalogue:
    def test_every_table_offset_is_reachable_by_a_vlda_immediate(self):
        for name in gen.TABLE_OFFSETS:
            if name.startswith("Q"):
                continue
            for half in (0, 1):
                ptr, off = gen.table_ref(name, 32 * half)
                assert -1024 <= off <= gen.isa_gen.MAX_VECTOR_OFFSET, (name, ptr, off)
                assert off % 32 == 0

    def test_masks_are_reachable_by_lda_q(self):
        for name in ("Q1", "Q3"):
            assert 0 <= gen.TABLE_OFFSETS[name] <= 496

    def test_pair_domains_never_mix(self):
        for p, c in gen.valid_pairs():
            dp, dc = gen.PRODUCER_BY_NAME[p].dom, gen.CONSUMER_BY_NAME[c].dom
            assert dp is None or dc is None or dp == dc

    def test_pair_spec_expansion(self):
        pairs = gen.parse_pairs(["vmul_i_dense:vst_amll", "vclr_cm:*"])
        assert ("vmul_i_dense", "vst_amll") in pairs
        assert all(p == "vclr_cm" or (p, c) == ("vmul_i_dense", "vst_amll") for p, c in pairs)


class TestSweep:
    def test_references_come_first_and_distances_are_covered(self):
        pts = gen.sweep_points()
        assert pts[0] == ("WAR", gen.REF_DISTANCE) and pts[1] == ("RAW", gen.REF_DISTANCE)
        assert {d for dr, d in pts if dr == "RAW"} >= set(gen.RAW_DISTANCES)
        assert {d for dr, d in pts if dr == "WAR"} >= set(gen.WAR_DISTANCES)

    def test_tied_consumers_have_no_war_points_but_an_old_reference(self):
        p, c = gen.PRODUCER_BY_NAME["vmul_i_dense"], gen.CONSUMER_BY_NAME["vaddmac_i_dense_acc1"]
        pts = gen.pair_points(p, c)
        assert not any(dr == "WAR" for dr, _ in pts)
        assert ("CONLY", 0) in pts

    def test_point_places_the_two_instructions_d_bundles_apart(self):
        p, c = gen.PRODUCER_BY_NAME["vmul_i_dense"], gen.CONSUMER_BY_NAME["vst_amll"]
        text = "\n".join(gen.point_lines(p, c, "RAW", 4, "i"))
        body = text[text.index(p.asm):text.index(c.asm)]
        assert body.count("nop") == 3
        war = "\n".join(gen.point_lines(p, c, "WAR", 0, "i"))
        assert f"{c.asm} ; {p.asm}" in war


@pytest.mark.skipif(not os.path.exists(gen.LLVM_MC), reason="llvm-mc not built")
class TestAssembles:
    def test_a_kernel_with_every_producer_and_consumer_assembles(self):
        pairs = [(p.name, "vst_amll") for p in gen.PRODUCERS if gen.pair_dom(p, gen.CONSUMER_BY_NAME["vst_amll"])]
        pairs += [("vmul_i_dense", c.name) for c in gen.CONSUMERS if gen.pair_dom(gen.PRODUCER_BY_NAME["vmul_i_dense"], c)]
        pairs += [("vmul_f_dense", c.name) for c in gen.CONSUMERS if gen.pair_dom(gen.PRODUCER_BY_NAME["vmul_f_dense"], c)]
        # one kernel per producer/consumer keeps the program memory bound irrelevant here
        for p, c in sorted(set(pairs)):
            text, _ = gen.kernel_text([(p, c)])
            with tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "k.s")
                open(src, "w").write(text)
                r = subprocess.run([gen.LLVM_MC, "--triple=aie2", "--filetype=obj", "-o",
                                    os.path.join(tmp, "k.o"), src], capture_output=True, text=True)
                assert r.returncode == 0, (p, c, r.stderr[:400])


class TestAnalysis:
    OLD = tuple(range(100, 100 + gen.RECORD_WORDS))
    NEW = tuple(range(200, 200 + gen.RECORD_WORDS))

    def pt(self, direction, d):
        return {"dir": direction, "d": d}

    def sweep(self, threshold, anomalies=()):
        """Points whose records see NEW from signed distance `threshold` on."""
        pts = [(self.pt("WAR", gen.REF_DISTANCE), self.OLD), (self.pt("RAW", gen.REF_DISTANCE), self.NEW)]
        for s in range(-9, 10):
            if s == 0:
                direction, d = "WAR", 0
            else:
                direction, d = ("RAW", s) if s > 0 else ("WAR", -s)
            rec = self.NEW if s >= threshold else self.OLD
            if s in anomalies:
                rec = tuple(0xDEAD for _ in rec)
            pts.append((self.pt(direction, d), rec))
        return pts

    def test_threshold_is_the_first_signed_distance_that_sees_new(self):
        for t in (-1, 0, 2, 5):
            res = an.analyze_pair(self.sweep(t))
            assert {w["t"] for w in res["words"].values()} == {t}
            assert all(w["ok"] for w in res["words"].values())

    def test_an_isolated_anomaly_below_the_threshold_is_reported_not_folded_in(self):
        res = an.analyze_pair(self.sweep(3, anomalies=(-2,)))
        w = next(iter(res["words"].values()))
        assert w["t"] == 3 and not w["ok"] and w["anomalies"] == [(-2, "X")]

    def test_words_where_old_equals_new_carry_no_information(self):
        old = list(self.OLD)
        old[1] = self.NEW[1]
        pts = [(p, tuple(old) if r is self.OLD else r) for p, r in self.sweep(2)]
        res = an.analyze_pair(pts)
        assert 1 not in res["words"]

    def test_conly_reference_replaces_the_war_reference(self):
        pts = [(self.pt("CONLY", 0), self.OLD), (self.pt("RAW", gen.REF_DISTANCE), self.NEW)]
        pts += [(self.pt("RAW", d), self.NEW if d >= 3 else self.OLD) for d in range(1, 10)]
        res = an.analyze_pair(pts)
        assert {w["t"] for w in res["words"].values()} == {3}


class TestFit:
    ITINS = {
        "II_P": ([5], ["VEC_Bypass"]),
        "II_PF": ([6], []),
        "II_C": ([5, 3, 1], ["VEC_Bypass", "VEC_Bypass", "NoBypass"]),
        "II_S": ([1, 1], ["NoBypass", "NoBypass"]),
    }

    def test_prediction_rule(self, monkeypatch):
        monkeypatch.setitem(fit.PRODUCER_CLASS, "p", "II_P")
        monkeypatch.setitem(fit.PRODUCER_CLASS, "pf", "II_PF")
        monkeypatch.setitem(fit.CONSUMER_CLASS, "mac", ("II_C", 1))
        monkeypatch.setitem(fit.CONSUMER_CLASS, "st", ("II_S", 0))
        assert fit.predict(self.ITINS, "p", "mac")[0] == 2        # 5 - 3 + 1 - forwarded
        assert fit.predict(self.ITINS, "p", "st")[0] == 5         # 5 - 1 + 1
        assert fit.predict(self.ITINS, "pf", "mac")[0] == 4       # 6 - 3 + 1, no forwarding

    def test_schedule_td_parses_the_mac_itineraries(self):
        if not os.path.exists(fit.SCHEDULE_TD):
            pytest.skip("llvm-aie not present")
        itins = fit.parse_itineraries()
        assert itins["II_VMAC"][0][:2] == [5, 3]
        assert itins["II_VMACf"][0][:2] == [6, 3]
        assert itins["II_VMOV_CM"][0][:2] == [2, 1]
