import gen_path_onset_probe as g


def test_a_length_suffix_sets_the_bd_length_and_keeps_the_route():
    assert g.case_spec("mem_own_n23") == ("mem", "mem", 23)
    assert g.case_spec("prod_mem") == ("prod", "mem", g.N_WORDS)
    text = g.design_mlir("prod", "prod", 23)
    assert "len = 23" in text and f"len = {g.N_WORDS}" not in text
