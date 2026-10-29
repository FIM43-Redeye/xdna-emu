import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


BRIDGE_PATH = Path(__file__).with_name("mlir-aie-bridge.py")
SPEC = importlib.util.spec_from_file_location("mlir_aie_bridge", BRIDGE_PATH)
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def write_fake_mlir_aie(root, instr_vector):
    python_root = root / "build" / "python"
    files = {
        "aie/__init__.py": "",
        "aie/_mlir_libs/__init__.py": "",
        "aie/_mlir_libs/_aie.py": "def get_target_model(_device):\n    return None\n",
        "aie/utils/__init__.py": "",
        "aie/utils/trace/__init__.py": "",
        "aie/utils/trace/events/__init__.py": (
            "from enum import Enum\n"
            f"class CoreEvent(Enum):\n    INSTR_VECTOR = {instr_vector}\n"
            "class MemEvent(Enum):\n    DMA_S2MM_0_START_TASK = 1\n"
            "class MemTileEvent(Enum):\n    DMA_S2MM_0_START_TASK = 2\n"
            "class ShimTileEvent(Enum):\n    DMA_S2MM_0_START_TASK = 3\n"
        ),
    }
    for relative, contents in files.items():
        path = python_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    return python_root


def run_trace_events(explicit_root, ambient_python):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ambient_python)
    return subprocess.run(
        [
            sys.executable,
            str(BRIDGE_PATH),
            "--mlir-aie-path",
            str(explicit_root),
            "trace-events",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_physical_column_start_comes_from_mapped_driver_device(tmp_path):
    regs = tmp_path / "drivers" / "accel" / "amdxdna"
    regs.mkdir(parents=True)
    (regs / "npu1_regs.c").write_text(
        "const struct amdxdna_dev_info dev_npu1_info = {\n"
        "\t.first_col = 1,\n"
        "};\n"
    )
    (regs / "npu4_regs.c").write_text(
        "const struct amdxdna_dev_info dev_npu4_info = {\n"
        "\t.first_col = 0,\n"
        "};\n"
    )

    assert BRIDGE.driver_physical_column_start("npu1_2col", tmp_path) == 1
    assert BRIDGE.driver_physical_column_start("npu2_7col", tmp_path) == 0


def test_explicit_mlir_aie_root_wins_over_ambient_package(tmp_path):
    explicit = tmp_path / "explicit"
    ambient = tmp_path / "ambient"
    write_fake_mlir_aie(explicit, instr_vector=73)
    ambient_python = write_fake_mlir_aie(ambient, instr_vector=199)

    result = run_trace_events(explicit, ambient_python)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["enums"]["CoreEvent"]["INSTR_VECTOR"] == 73


def test_broken_explicit_mlir_aie_root_does_not_fall_back_to_ambient(tmp_path):
    broken = tmp_path / "broken"
    (broken / "build" / "python").mkdir(parents=True)
    ambient_python = write_fake_mlir_aie(tmp_path / "ambient", instr_vector=199)

    result = run_trace_events(broken, ambient_python)

    assert result.returncode != 0


def test_trace_events_reports_binding_from_explicit_root(tmp_path):
    explicit = tmp_path / "explicit"
    ambient = tmp_path / "ambient"
    write_fake_mlir_aie(explicit, instr_vector=73)
    ambient_python = write_fake_mlir_aie(ambient, instr_vector=199)

    result = run_trace_events(explicit, ambient_python)

    assert result.returncode == 0, result.stderr
    binding = Path(json.loads(result.stdout)["mlir_aie_binding_path"]).resolve()
    assert binding.is_relative_to(explicit.resolve())
