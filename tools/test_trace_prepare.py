"""Tests for trace-prepare.py.

Run with:
    cd /home/triple/npu-work/xdna-emu
    PYTHONPATH=tools python3 -m pytest tools/test_trace_prepare.py -v -k "not integration"
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from trace_config import load as trace_config_load  # noqa: E402

# The script under test.
SCRIPT = Path(__file__).parent / "trace-prepare.py"
REPO_ROOT = SCRIPT.parent.parent


def run_prepare(*args, env_extra=None):
    """Run trace-prepare.py as a subprocess and return the CompletedProcess."""
    env = os.environ.copy()
    # Ensure tools/ is on PYTHONPATH so trace_inject and cpp_trace_patch import.
    tools_dir = str(SCRIPT.parent)
    env["PYTHONPATH"] = tools_dir + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + list(args),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# 1. CLI help
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_help(self):
        """--help runs without error."""
        result = run_prepare("--help")
        assert result.returncode == 0
        assert "trace-prepare" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_missing_test_dir(self, tmp_path):
        """Exits nonzero for nonexistent test directory."""
        result = run_prepare(
            str(tmp_path / "does_not_exist"),
            "--output", str(tmp_path / "out"),
        )
        assert result.returncode != 0
        assert "not a directory" in result.stderr.lower() or "does not exist" in result.stderr.lower()


# ---------------------------------------------------------------------------
# 2. Quarantine checks
# ---------------------------------------------------------------------------

class TestQuarantine:
    def _make_test_dir(self, tmp_path, name="quarantined_test"):
        """Create a minimal test source directory with aie.mlir and test.cpp."""
        test_dir = tmp_path / name
        test_dir.mkdir()
        (test_dir / "aie.mlir").write_text(
            'module { aie.device(NPUDEVICE) { aie.tile(0, 2) } }\n'
        )
        (test_dir / "test.cpp").write_text(
            '#include <xrt/xrt_device.h>\n'
            'int main() {\n'
            '  unsigned int device_index = 0;\n'
            '  auto device = xrt::device(device_index);\n'
            '  return 0;\n'
            '}\n'
        )
        return test_dir

    def test_test_quarantined(self, tmp_path):
        """Exit 0 with SKIP status when test-quarantined."""
        test_dir = self._make_test_dir(tmp_path, "bad_test")
        output_dir = tmp_path / "out"

        # Create a test quarantine file.
        quarantine = tmp_path / "test-quarantine.txt"
        quarantine.write_text("# Comment line\nbad_test\n")

        result = run_prepare(
            str(test_dir),
            "--output", str(output_dir),
            "--test-quarantine", str(quarantine),
        )
        assert result.returncode == 0

        status = (output_dir / "prepare-status.txt").read_text().strip()
        assert status.startswith("SKIP")
        assert "test-quarantined" in status

    def test_trace_quarantined(self, tmp_path):
        """Exit 0 with SKIP status when trace-quarantined."""
        test_dir = self._make_test_dir(tmp_path, "traced_bad")
        output_dir = tmp_path / "out"

        # Create a trace quarantine file.
        quarantine = tmp_path / "trace-quarantine.txt"
        quarantine.write_text("traced_bad\n")

        result = run_prepare(
            str(test_dir),
            "--output", str(output_dir),
            "--trace-quarantine", str(quarantine),
        )
        assert result.returncode == 0

        status = (output_dir / "prepare-status.txt").read_text().strip()
        assert status.startswith("SKIP")
        assert "trace-quarantined" in status

    def test_quarantine_comment_ignored(self, tmp_path):
        """Quarantine file comments and blank lines are ignored."""
        test_dir = self._make_test_dir(tmp_path, "good_test")
        output_dir = tmp_path / "out"

        quarantine = tmp_path / "test-quarantine.txt"
        quarantine.write_text(
            "# This is a comment\n"
            "\n"
            "some_other_test\n"
        )

        # good_test is NOT in the quarantine -- should proceed (and fail
        # because mlir-aie is not available, but NOT with SKIP status).
        result = run_prepare(
            str(test_dir),
            "--output", str(output_dir),
            "--test-quarantine", str(quarantine),
        )
        # It will fail (no mlir-aie), but should NOT be SKIP.
        if output_dir.exists() and (output_dir / "prepare-status.txt").exists():
            status = (output_dir / "prepare-status.txt").read_text().strip()
            assert not status.startswith("SKIP")

    def test_missing_quarantine_file(self, tmp_path):
        """Missing quarantine files are handled gracefully (no quarantine)."""
        test_dir = self._make_test_dir(tmp_path, "any_test")
        output_dir = tmp_path / "out"

        # Point to nonexistent quarantine files -- should not crash.
        result = run_prepare(
            str(test_dir),
            "--output", str(output_dir),
            "--test-quarantine", str(tmp_path / "nonexistent.txt"),
            "--trace-quarantine", str(tmp_path / "also_nonexistent.txt"),
        )
        # Should proceed past quarantine checks (and likely fail on mlir-aie).
        # The key check: it did NOT crash with FileNotFoundError.
        assert "FileNotFoundError" not in result.stderr


# ---------------------------------------------------------------------------
# 3. --skip-mlir mode
# ---------------------------------------------------------------------------

class TestSkipMlir:
    def test_skip_mlir_produces_test_traced_cpp(self, tmp_path):
        """--skip-mlir applies BDF patch but skips trace BO injection.

        Trace BO injection requires trace_config.json (single source of
        truth for kernel_arg_slot); --skip-mlir bypasses the injector that
        writes it, so trace patching is skipped too.
        """
        test_dir = tmp_path / "skip_test"
        test_dir.mkdir()
        (test_dir / "aie.mlir").write_text("module {}\n")
        (test_dir / "test.cpp").write_text(
            '#include <xrt/xrt_device.h>\n'
            '#include <xrt/xrt_bo.h>\n'
            '#include <xrt/xrt_kernel.h>\n'
            'int main() {\n'
            '  unsigned int device_index = 0;\n'
            '  auto device = xrt::device(device_index);\n'
            '  auto bo_instr = xrt::bo(device, 4096, XRT_BO_FLAGS_CACHEABLE,\n'
            '                           kernel.group_id(1));\n'
            '  auto bo_in = xrt::bo(device, 1024, XRT_BO_FLAGS_HOST_ONLY,\n'
            '                       kernel.group_id(3));\n'
            '  auto run = kernel(0, bo_instr, 0, bo_in);\n'
            '  ert_cmd_state r = run.wait();\n'
            '  return 0;\n'
            '}\n'
        )
        output_dir = tmp_path / "out"

        result = run_prepare(
            str(test_dir),
            "--output", str(output_dir),
            "--skip-mlir",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # test_traced.cpp should exist with BDF patch but NO trace BO.
        traced_cpp = output_dir / "test_traced.cpp"
        assert traced_cpp.exists(), "test_traced.cpp not created"
        content = traced_cpp.read_text()
        assert "XRT_DEVICE_BDF" in content, "BDF patch not applied"
        assert "bo_trace" not in content, (
            "Trace buffer injected without trace_config.json"
        )

        # No MLIR or trace_config artifacts should exist.
        assert not (output_dir / "aie_traced.mlir").exists()
        assert not (output_dir / "trace_config.json").exists()

        # Status should be OK.
        status = (output_dir / "prepare-status.txt").read_text().strip()
        assert status == "OK"

    def test_skip_mlir_no_test_cpp(self, tmp_path):
        """--skip-mlir without test.cpp still succeeds (nothing to patch)."""
        test_dir = tmp_path / "no_cpp"
        test_dir.mkdir()
        (test_dir / "aie.mlir").write_text("module {}\n")
        output_dir = tmp_path / "out"

        result = run_prepare(
            str(test_dir),
            "--output", str(output_dir),
            "--skip-mlir",
        )
        assert result.returncode == 0
        assert not (output_dir / "test_traced.cpp").exists()
        status = (output_dir / "prepare-status.txt").read_text().strip()
        assert status == "OK"


# ---------------------------------------------------------------------------
# 4. BDF patch
# ---------------------------------------------------------------------------

class TestBDFPatch:
    def test_bdf_patch_applied(self, tmp_path):
        """BDF environment variable patch is applied to test.cpp."""
        test_dir = tmp_path / "bdf_test"
        test_dir.mkdir()
        (test_dir / "aie.mlir").write_text("module {}\n")
        (test_dir / "test.cpp").write_text(
            '#include <xrt/xrt_device.h>\n'
            '#include <xrt/xrt_bo.h>\n'
            '#include <xrt/xrt_kernel.h>\n'
            'int main() {\n'
            '  unsigned int device_index = 0;\n'
            '  auto device = xrt::device(device_index);\n'
            '  auto bo_in = xrt::bo(device, 1024, XRT_BO_FLAGS_HOST_ONLY,\n'
            '                       kernel.group_id(3));\n'
            '  auto run = kernel(0, bo_in);\n'
            '  run.wait();\n'
            '  return 0;\n'
            '}\n'
        )
        output_dir = tmp_path / "out"

        result = run_prepare(
            str(test_dir),
            "--output", str(output_dir),
            "--skip-mlir",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        content = (output_dir / "test_traced.cpp").read_text()
        # BDF patch replaces the two lines.
        assert 'std::getenv("XRT_DEVICE_BDF")' in content
        assert "unsigned int device_index = 0;" not in content
        assert "xrt::device(device_index)" not in content


# ---------------------------------------------------------------------------
# 5. No aie.mlir or test.cpp
# ---------------------------------------------------------------------------

class TestMissingFiles:
    def test_no_aie_mlir(self, tmp_path):
        """Error when test dir has no aie.mlir or aie2.py, with status file."""
        test_dir = tmp_path / "empty_test"
        test_dir.mkdir()
        (test_dir / "test.cpp").write_text("int main() { return 0; }\n")
        output_dir = tmp_path / "out"

        result = run_prepare(
            str(test_dir),
            "--output", str(output_dir),
        )
        assert result.returncode != 0
        # prepare-status.txt must be written even on failure
        status_file = output_dir / "prepare-status.txt"
        assert status_file.exists(), "prepare-status.txt not written on failure"
        assert status_file.read_text().startswith("FAIL")


# ---------------------------------------------------------------------------
# 6. Integration test (requires mlir-aie Python API)
# ---------------------------------------------------------------------------

# Check if mlir-aie Python API is available.
_HAS_MLIR_AIE = False
try:
    # Attempt an import that would only work with mlir-aie installed.
    import importlib
    importlib.import_module("aie.ir")
    _HAS_MLIR_AIE = True
except (ImportError, ModuleNotFoundError):
    pass

# Also check that the npu-xrt test directory exists.
_ADD_ONE_DIR = Path(
    "/home/triple/npu-work/mlir-aie/test/npu-xrt/add_one_using_dma"
)


@pytest.mark.skipif(
    not _HAS_MLIR_AIE,
    reason="mlir-aie Python API not available",
)
@pytest.mark.skipif(
    not _ADD_ONE_DIR.is_dir(),
    reason="add_one_using_dma test source not found",
)
class TestIntegration:
    def test_integration_add_one(self, tmp_path):
        """Full integration: inject trace into add_one_using_dma."""
        output_dir = tmp_path / "traced"

        result = run_prepare(
            str(_ADD_ONE_DIR),
            "--output", str(output_dir),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Check outputs.
        assert (output_dir / "aie_traced.mlir").exists()
        assert (output_dir / "test_traced.cpp").exists()
        assert (output_dir / "trace_config.json").exists()

        status = (output_dir / "prepare-status.txt").read_text().strip()
        assert status == "OK"

        # trace_config.json: schema-validated load + sanity-check fields.
        cfg = trace_config_load(output_dir / "trace_config.json")
        assert cfg["schema_version"] == 1
        assert cfg["test_name"] == "add_one_using_dma"
        assert cfg["buffer"]["kernel_arg_slot"] >= 4  # past the 3 fixed slots
        assert len(cfg["tiles_traced"]) >= 1

        # test_traced.cpp should have both BDF and trace patches.
        cpp_content = (output_dir / "test_traced.cpp").read_text()
        assert "XRT_DEVICE_BDF" in cpp_content
        assert "bo_trace" in cpp_content

        # aie_traced.mlir should have trace packet flows.
        mlir_content = (output_dir / "aie_traced.mlir").read_text()
        assert "Trace" in mlir_content
