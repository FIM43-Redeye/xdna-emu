"""Analysis helpers of the op-cost probe: stall split, per-op means, line fit, two-point split."""
import math

import fw_op_cost_report as rep


def test_split_stalls_separates_intervals_over_the_threshold():
    normal, stalls = rep.split_stalls([100, 105, 2400, 98, 3000])
    assert normal == [100, 105, 98] and stalls == [2400, 3000]


def test_split_stalls_is_relative_to_the_typical_interval():
    # a 64-word block costs ~1100 cycles per interval; only the 3800 one is a stall
    normal, stalls = rep.split_stalls([1126, 1212, 1141, 3837, 1137])
    assert stalls == [3837] and len(normal) == 4


def test_mean_interval_is_exact_for_constant_input():
    assert rep.mean_interval([100] * 10) == 100


def test_span_mean_ignores_the_quantisation_of_single_intervals():
    # stamps dithered by +-15 around a true 102-cycle period: the span mean stays close
    stamps = [0, 117, 204, 321, 408, 525, 612]
    assert abs(rep.span_mean(stamps) - 102) < 3


def test_fit_line_recovers_intercept_and_slope():
    a, b = rep.fit_line([1, 4, 16, 64], [106, 147, 312, 987])
    assert math.isclose(b, (987 - 106) / 63, rel_tol=0.05) and 85 < a < 100


def test_two_point_split_of_a_pure_firmware_cost():
    # 60 firmware cycles at 600 MHz = 100 ns, at 400 MHz = 150 ns
    fw_cycles, fixed_ns = rep.two_point_split(ns_at_600=100.0, ns_at_400=150.0)
    assert math.isclose(fw_cycles, 60.0) and abs(fixed_ns) < 1e-9


def test_two_point_split_of_a_fixed_cost():
    fw_cycles, fixed_ns = rep.two_point_split(ns_at_600=80.0, ns_at_400=80.0)
    assert abs(fw_cycles) < 1e-9 and math.isclose(fixed_ns, 80.0)


def test_cycles_to_ns_uses_the_sampled_array_clock():
    assert math.isclose(rep.to_ns(1028, 1028), 1000.0)
