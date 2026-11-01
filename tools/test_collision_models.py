"""Tests for the write-lane model and the rule normaliser, using silicon observations as fixtures."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collision_rules as cr   # noqa: E402
import verify_w_lane_model as vm  # noqa: E402

I0, I1 = "init0", "init1"
INIT = {("A", 0): I0, ("A", 1): I1, ("B", 0): I0, ("B", 1): I1}


def pred(a_dst, b_dst, a_data, b_data, shared=False):
    return vm.predict(a_dst, b_dst, a_data, b_data, INIT, shared)


class LaneModel(unittest.TestCase):
    def test_same_half_drops_the_earlier_writer(self):
        # probe6 sep, wl A vs wl B: A keeps its initial value, B lands
        st = pred("wl", "wl", ["a0"], ["b0"])
        self.assertEqual(st[("A", 0)], I0)
        self.assertEqual(st[("B", 0)], "b0")

    def test_other_half_still_receives_the_later_data(self):
        # probe6 point 70: wl A vs wh B: A's wl takes B's data, B's wh lands
        st = pred("wl", "wh", ["a0"], ["b0"])
        self.assertEqual(st[("A", 0)], "b0")
        self.assertEqual(st[("B", 1)], "b0")

    def test_wide_writer_loses_only_the_contested_bank(self):
        # point 166: x A (64 bytes) vs wh B: A.lo takes B's data, A.hi is dropped,
        # and B's wh lands A's high lane because only a 512-bit writer drives lane 1
        st = pred("x", "wh", ["a0", "a1"], ["b0"])
        self.assertEqual(st[("A", 0)], "b0")
        self.assertEqual(st[("A", 1)], I1)
        self.assertEqual(st[("B", 1)], "a1")

    def test_x_after_x_drops_both_banks(self):
        st = pred("x", "x", ["a0", "a1"], ["b0", "b1"])
        self.assertEqual((st[("A", 0)], st[("A", 1)]), (I0, I1))
        self.assertEqual((st[("B", 0)], st[("B", 1)]), ("b0", "b1"))

    def test_shared_register_ends_with_the_later_value(self):
        st = pred("wl", "wl", ["a0"], ["b0"], shared=True)
        self.assertEqual(st[("R", 0)], "b0")

    def test_cm_banks_beyond_the_later_writers_width_survive(self):
        # cm A (four banks) then bml B (two): A keeps banks 2 and 3
        init = {(w, h): f"i{h}" for w in "AB" for h in range(4)}
        st = vm.predict("cm", "bml", ["a0", "a1", "a2", "a3"], ["b0", "b1"], init, False)
        self.assertEqual([st[("A", h)] for h in range(4)], ["i0", "i1", "a2", "a3"])
        self.assertEqual([st[("B", h)] for h in range(2)], ["b0", "b1"])


class CrossPort(unittest.TestCase):
    def test_separate_registers_do_not_collide(self):
        st = vm.predict_cross("cm", "cm", ["a"] * 4, ["b"] * 4,
                              {(w, h): "i" for w in "AB" for h in range(4)}, False, "B")
        self.assertEqual(st[("A", 0)], "a")
        self.assertEqual(st[("B", 0)], "b")

    def test_winner_takes_a_shared_bank(self):
        init = {(w, h): "i" for w in "AB" for h in range(4)}
        st = vm.predict_cross("cm", "cm", ["a"] * 4, ["b"] * 4, init, True, "A")
        self.assertEqual(st[("R", 0)], "a")


class Normalise(unittest.TestCase):
    def test_mover_names_collapse(self):
        self.assertEqual(cr.normalise("src2.lo512<-vmov_bm.s1"), "src2.lo512<-MOVER")

    def test_detail_is_dropped(self):
        self.assertEqual(cr.normalise("mixed(alone:8/16)"), "mixed(...)")


if __name__ == "__main__":
    unittest.main()
