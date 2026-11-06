"""Runner pieces of the op-cost probe that need no hardware: clock parsing, validity, TDR counting."""
import fw_op_cost_run as r


def test_parse_clock_line():
    line = "1793904395.128 MP-NPU Clock 400 MHz, H Clock 800 MHz"
    assert r.parse_clocks(line) == (400, 800)


def test_parse_clock_line_rejects_garbage():
    assert r.parse_clocks("nothing here") is None


def test_run_is_valid_only_when_both_samples_match_the_dpm_point():
    assert r.clocks_valid(1, (600, 1028), (600, 1028))
    assert r.clocks_valid(0, (400, 800), (400, 800))
    assert not r.clocks_valid(0, (400, 800), (600, 1028))
    assert not r.clocks_valid(1, None, (600, 1028))


def test_qos_args_for_each_dpm_point():
    assert r.host_qos_args(1) == []
    assert r.host_qos_args(0) == ["--gops", "1", "--fps", "1000"]


def test_tdr_count_matches_tdr_lines_case_insensitively():
    log = "a\nTDR detected\nxx tdr yy\nnothing\n"
    assert r.count_tdr(log) == 2


def test_hw_busy_detects_other_hardware_jobs():
    ps = "python3 -m harness suite hw x r1\nzsh -c grep emu-bridge-test\nvim a\n"
    assert r.hw_busy(ps)
    assert not r.hw_busy("vim a\nzsh -c emu-bridge-test\nbash -c isa-test\n")


def test_host_command_carries_extra_host_args_after_the_qos_args():
    import fw_op_cost_run as r
    cmd = r.host_command("h", "case", {"results_words": 4}, "out", 0, ["--instr-bo-bytes", "65536"])
    assert cmd[-6:] == ["--gops", "1", "--fps", "1000", "--instr-bo-bytes", "65536"]
    assert "--out-size" in cmd and cmd[cmd.index("--out-size") + 1] == "16"
