"""CLI wiring for `suite hw --repeat` and `suite repeats` (the runner itself is faked)."""
import pytest

from harness import __main__ as cli
from harness import suite as S


class Args:
    pass


@pytest.fixture
def fake_suite(monkeypatch):
    monkeypatch.setattr(S, "load_suite", lambda d: "suite")
    return monkeypatch


def test_hw_passes_repeat_through_and_exits_1_when_a_case_disagrees(fake_suite, capsys):
    seen = {}

    def fake_run_hw(suite, tag, **kw):
        seen.update(kw)
        return [S.CaseOutcome("a", "ok", agree=True), S.CaseOutcome("b", "ok", agree=False)]

    fake_suite.setattr(S, "run_hw", fake_run_hw)
    assert cli.main(["suite", "hw", "d", "r1", "--repeat", "5"]) == 1
    assert seen["repeat"] == 5 and "b: repeats DIFFER" in capsys.readouterr().out


def test_hw_without_repeat_is_unchanged(fake_suite):
    seen = {}
    fake_suite.setattr(S, "run_hw", lambda suite, tag, **kw: seen.update(kw) or [S.CaseOutcome("a", "ok")])
    assert cli.main(["suite", "hw", "d", "r1"]) == 0 and seen["repeat"] == 1


def test_repeats_subcommand_prints_the_report_and_exits_by_agreement(fake_suite, capsys):
    rows = [S.CaseRepeats("a", 3, (), False),
            S.CaseRepeats("b", 3, (S.ByteDiff(2, "core.out.bin", 4, 1, 2),), True)]
    fake_suite.setattr(S, "repeat_report", lambda suite, tag, only=None: rows)
    assert cli.main(["suite", "repeats", "d", "r1"]) == 1
    assert "b: NONDETERMINISTIC" in capsys.readouterr().out
    fake_suite.setattr(S, "repeat_report", lambda suite, tag, only=None: rows[:1])
    assert cli.main(["suite", "repeats", "d", "r1"]) == 0
