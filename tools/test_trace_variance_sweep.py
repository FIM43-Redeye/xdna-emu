import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_spec = importlib.util.spec_from_file_location(
    "trace_variance_sweep", Path(__file__).parent / "trace-variance-sweep.py",
)
tvs = importlib.util.module_from_spec(_spec)
sys.modules["trace_variance_sweep"] = tvs
_spec.loader.exec_module(tvs)


def test_build_sweep_cmd_is_hw_only_with_tiles_and_outdir():
    cmd = tvs.build_sweep_cmd("add_one_using_dma",
                              "0:0:shim,0:1:memtile,0:2:core,0:2:memmod",
                              Path("/tmp/out/run_03"), jobs=5)
    assert "--no-emu" in cmd
    assert "--test" in cmd and "add_one_using_dma" in cmd
    i = cmd.index("--tiles")
    assert "memtile" in cmd[i + 1]
    j = cmd.index("--out-dir")
    assert cmd[j + 1].endswith("run_03")


def test_main_invokes_sweep_once_per_repeat(tmp_path):
    calls = []
    with patch.object(tvs.subprocess, "run",
                      side_effect=lambda cmd, **kw: calls.append(cmd) or _ok()):
        rc = tvs.main(["--test", "add_one_using_dma", "--repeat", "4",
                       "--out", str(tmp_path)])
    assert rc == 0
    assert len(calls) == 4
    outdirs = [c[c.index("--out-dir") + 1] for c in calls]
    assert sorted(Path(o).name for o in outdirs) == ["run_00", "run_01", "run_02", "run_03"]


def _ok():
    class R:  # minimal CompletedProcess stand-in
        returncode = 0
    return R()
