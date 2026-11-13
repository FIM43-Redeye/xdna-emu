"""The lock LOCK_STALL trace probe: case table, kernel text, and the trace reductions."""
import re

import gen_lock_pulse_probe as g


def test_every_case_keeps_the_aiecc_mains_live_registers_and_brackets_the_window():
    for name, case in g.cases().items():
        asm = case.consumer_asm()
        assert "st r16, [sp, #-4]" in asm and "lda r16, [sp, #-4]" in asm, name
        assert "st r17, [sp, #-8]" in asm and "lda r17, [sp, #-8]" in asm, name
        assert asm.index("event #0") < asm.index("event #1"), name
        if name == "cnt_event":      # the cycle accounting case carries markers of its own
            continue
        # only the window bundles sit between the markers, plus the closing gap
        between = asm[asm.index("event #0"):asm.index("event #1")].splitlines()[1:]
        between = [line for line in between if line.strip()]
        assert len(between) == len(case.window) + case.end_gap, name


def test_a_producer_exists_exactly_for_the_cases_with_a_refused_acquire():
    for name, case in g.cases().items():
        assert (case.producer_asm() is not None) == (case.producer_release is not None), name
    assert g.cases()["acq_refused"].producer_release == g.A


def test_bundles_places_slots_at_their_offsets_and_pads_with_nops():
    assert g.bundles({0: "acq #62, r3", 2: "rel #63, r2"}, 3) == ["  acq #62, r3", g.N, "  rel #63, r2", g.N]


def test_pair_cases_put_the_second_request_d_bundles_after_the_first():
    window = g.cases()["relacq_d3"].window
    assert [i for i, b in enumerate(window) if b != g.N] == [0, 3]
    assert "rel" in window[0] and "acq" in window[3]


def test_conditional_requests_carry_the_condition_register():
    window = g.cases()["cond_acq_skipped"].window
    assert any(re.search(r"acq\.cond #\d+, r3, r26", b) for b in window)


def test_levels_merge_consecutive_cycles_and_split_gaps():
    ev = [{"name": "LOCK_STALL", "ts": t, "soc": t} for t in (5, 6, 9, 20, 21, 22)]
    assert g.levels(ev, "LOCK_STALL", "ts") == [(5, 6), (9, 9), (20, 22)]


def test_both_backends_are_read_in_ts():
    # a frame's cycles field is the idle cycles since the previous frame on silicon and in the
    # emulator, so `ts` is the cycle for both
    assert g.TRUE_CYCLE == {"hw": "ts", "emu": "ts"}


def test_gather_cases_set_up_the_lane_addresses_before_the_window():
    case = g.cases()["gatacq_d3"]
    asm = case.consumer_asm()
    assert asm.index("vbcst.32 x8, r7") < asm.index("event #0") < asm.index(g.GATHER)
    assert [i for i, b in enumerate(case.window) if b != g.N] == [0, 3]
    assert "vbcst" not in g.cases()["acq_free"].consumer_asm()


def test_runner_extra_args_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(g.RUNNER_EXTRA_ARGS_ENV, raising=False)
    assert g.runner_extra_args() == []
    monkeypatch.setenv(g.RUNNER_EXTRA_ARGS_ENV, "-v --post-completion-us 5")
    assert g.runner_extra_args() == ["-v", "--post-completion-us", "5"]
    seen = {}

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return Done()

    monkeypatch.setattr(g.subprocess, "run", fake_run)
    (tmp_path / "c").mkdir()
    assert g.run_case_raw(tmp_path, "c", "t", "emu", output=False)
    assert seen["cmd"][-3:] == ["-v", "--post-completion-us", "5"]
