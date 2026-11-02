"""The tools that read aiecc's input_with_addresses.mlir must find it where
`aiecc --get-input-with-addresses` writes it, and fail naming that flag when
it is absent (aiecc no longer writes it by default)."""
import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import aiecc_artifacts as aa  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FLAG = "--get-input-with-addresses"


def test_coldstart_resolve_mlir_finds_flag_location(tmp_path, monkeypatch):
    mod = _load("coldstart_residual", "tools/coldstart-residual.py")
    mlir_aie = tmp_path / "mlir-aie"
    sentinel = mlir_aie / "lib/Dialect/AIE/Util/aie_registers_aie2.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("{}")
    monkeypatch.setenv("MLIR_AIE_PATH", str(mlir_aie))
    build = mlir_aie / "build/test/npu-xrt/k/chess"
    build.mkdir(parents=True)
    with pytest.raises(aa.MissingInputWithAddresses, match=FLAG):
        mod.resolve_mlir("k", "chess")
    (build / aa.FILENAME).write_text("x")
    assert mod.resolve_mlir("k", "chess") == build / aa.FILENAME


@pytest.mark.parametrize("fn,arg", [("kbuild_paths", 8),
                                    ("kernel_build_paths", "kern")])
def test_multirun_paths_name_the_flag_when_mlir_missing(tmp_path, monkeypatch, fn, arg):
    mod = _load("multirun_trace_campaign", "tools/multirun-trace-campaign.py")
    monkeypatch.setattr(mod, "MLIR_AIE_ROOT", tmp_path)
    sub = "_diag_shim_chain_sweep/k8" if fn == "kbuild_paths" else "kern"
    base = tmp_path / "build/test/npu-xrt" / sub / "chess"
    base.mkdir(parents=True)
    (base / "aie.xclbin").write_text("x")
    (base / "insts.bin").write_text("x")
    with pytest.raises(FileNotFoundError, match=FLAG):
        getattr(mod, fn)(arg)
    (base / aa.FILENAME).write_text("x")
    assert getattr(mod, fn)(arg)["mlir"] == base / aa.FILENAME


def test_port_span_cadence_requires_the_file_with_the_flag_named(tmp_path):
    # The tool resolves it through require_input_with_addresses.
    with pytest.raises(aa.MissingInputWithAddresses, match=FLAG):
        aa.require_input_with_addresses(tmp_path)
    src = (REPO / "tools/port-span-cadence.py").read_text()
    assert "require_input_with_addresses" in src


def test_run_sweep_compile_asks_aiecc_for_the_file(tmp_path):
    mod = _load("run_sweep", "tools/calibration/run_sweep.py")
    with mock.patch.object(mod.subprocess, "run") as run:
        run.return_value = mock.Mock(returncode=0)
        mod.compile_one(tmp_path, make_dummies=False)
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("aiecc") + 1:].count(FLAG) == 1


def test_run_sweep_parse_reads_the_file_aiecc_wrote(tmp_path):
    mod = _load("run_sweep", "tools/calibration/run_sweep.py")
    (tmp_path / aa.FILENAME).write_text("x")
    with mock.patch.object(mod.subprocess, "run") as run:
        run.return_value = mock.Mock(returncode=1)
        try:
            mod.parse_one(tmp_path, None)
        except Exception:
            pass  # only the command line matters here
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("--xclbin-mlir") + 1] == aa.FILENAME


def test_run_sweep_parse_names_the_flag_when_file_missing(tmp_path):
    mod = _load("run_sweep", "tools/calibration/run_sweep.py")
    with pytest.raises(aa.MissingInputWithAddresses, match=FLAG):
        mod.parse_one(tmp_path, None)
