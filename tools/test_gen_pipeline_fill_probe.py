import gen_pipeline_fill_probe as g


def test_a_start_phase_suffix_keeps_the_case_route_length_and_arming():
    assert g.race_phase("race_prod_shimq_p2") == 2
    assert g.race_words("race_prod_shimq_p2") == g.race_words("race_prod_shimq") == 2 * g.RACE_WORDS
    assert g.race_route_channel("race_prod_shim0_p3") == g.race_route_channel("race_prod_shim0") == 0
    assert g.race_route_channel("race_prod_shimq_p1") is None
    assert g.RACE_CASES["race_prod_shim0_p1"] == g.RACE_CASES["race_prod_shim0"]


def test_the_source_bd_starts_phase_words_into_its_buffer():
    steps = g.race_steps("prod", "shim", phase=3)
    bd = next(line for s in steps if isinstance(s, g.cp.Raw) for line in s.lines if "bd_id = 1" in line)
    assert "offset = 3" in bd
