"""Tests for show-sweep-matrix.py classification / diff logic.

Pure-function tests only. The main output path is visual and exercised
by hand.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "show_sweep_matrix", Path(__file__).parent / "show-sweep-matrix.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["show_sweep_matrix"] = _mod
_spec.loader.exec_module(_mod)

classify_row = _mod.classify_row
ComboMatrix = _mod.ComboMatrix
show_diff = _mod.show_diff


def _row(name, hf, ef, hw_err=None, emu_err=None):
    return {
        "name": name, "slot": 0, "batch": 0,
        "hw":  {"fired": hf, "error": hw_err},
        "emu": {"fired": ef, "error": emu_err},
    }


def test_classify_match():
    status, hf, ef = classify_row(_row("E", 5, 5))
    assert status == "MATCH(5)"
    assert (hf, ef) == (5, 5)


def test_classify_zero():
    status, hf, ef = classify_row(_row("E", 0, 0))
    assert status == "ZERO"
    assert (hf, ef) == (0, 0)


def test_classify_drift():
    status, hf, ef = classify_row(_row("E", 17, 1))
    assert status == "DRIFT(H17/E1)"
    assert (hf, ef) == (17, 1)


def test_classify_emu_only():
    status, hf, ef = classify_row(_row("E", 0, 4))
    assert status == "EMU-ONLY(4)"


def test_classify_hw_only():
    status, hf, ef = classify_row(_row("E", 8, 0))
    assert status == "HW-ONLY(8)"


def test_classify_hw_err_only():
    status, hf, ef = classify_row(_row("E", None, 3, hw_err="TDR"))
    assert status == "HW-ERR"


def test_classify_both_err():
    status, hf, ef = classify_row(
        _row("E", None, None, hw_err="TDR", emu_err="crash"))
    assert status == "BOTH-ERR"


def test_diff_flags_regressed_event(capsys, tmp_path):
    # Build baseline and new dirs with one combo each
    def mk_dir(d, fired_vec):
        d.mkdir()
        doc = {
            "test": "demo", "compiler": "chess",
            "tile": {"col": 0, "row": 2, "type": "core"},
            "events": [
                {"name": n, "slot": i, "batch": 0,
                 "hw":  {"fired": hw, "error": None},
                 "emu": {"fired": emu, "error": None}}
                for i, (n, hw, emu) in enumerate(fired_vec)
            ],
        }
        (d / "demo.chess.core_c0r2.json").write_text(json.dumps(doc))

    baseline = tmp_path / "base"
    newer = tmp_path / "new"
    # Same event list; INSTR_VECTOR regresses from MATCH(4) to DRIFT(H4/E1)
    mk_dir(baseline, [("INSTR_VECTOR", 4, 4), ("LOCK_STALL", 17, 1)])
    mk_dir(newer,    [("INSTR_VECTOR", 4, 1), ("LOCK_STALL", 17, 1)])

    from show_sweep_matrix import load_dir
    diffs = show_diff(load_dir(baseline), load_dir(newer))
    out = capsys.readouterr().out
    assert diffs == 1
    assert "INSTR_VECTOR" in out
    assert "MATCH(4)" in out
    assert "DRIFT(H4/E1)" in out
    # LOCK_STALL was unchanged (already DRIFT in both) -- should not appear
    # as a diff cell.
    assert "LOCK_STALL" not in out


def test_diff_suppresses_zero_zero_equivalence(tmp_path):
    # Both sides report "ZERO" on the same row -- not a diff.
    def mk_dir(d):
        d.mkdir()
        doc = {
            "test": "demo", "compiler": "chess",
            "tile": {"col": 0, "row": 2, "type": "core"},
            "events": [
                {"name": "NEVER_FIRES", "slot": 0, "batch": 0,
                 "hw":  {"fired": 0, "error": None},
                 "emu": {"fired": 0, "error": None}},
            ],
        }
        (d / "demo.chess.core_c0r2.json").write_text(json.dumps(doc))

    base = tmp_path / "b"
    new = tmp_path / "n"
    mk_dir(base)
    mk_dir(new)

    from show_sweep_matrix import load_dir
    diffs = show_diff(load_dir(base), load_dir(new))
    assert diffs == 0
