#!/usr/bin/env python3

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("emu-bridge-test.sh")


class BridgePathTests(unittest.TestCase):
    def run_list(self, work_root: Path, test_filter: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["NPU_WORK_DIR"] = str(work_root)
        env["BRIDGE_TEST_RESULTS"] = str(work_root / "results")
        return subprocess.run(
            [str(SCRIPT), "--list", test_filter],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_list_resolves_mlir_aie_from_npu_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_root = Path(tmp)
            test_dir = work_root / "mlir-aie/test/npu-xrt/add_one_using_dma"
            test_dir.mkdir(parents=True)
            (test_dir / "test.cpp").write_text("", encoding="utf-8")
            (test_dir / "aie.mlir").write_text("", encoding="utf-8")

            result = self.run_list(work_root, "^add_one_using_dma$")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(">>> Available tests (1):\n  add_one_using_dma\n", result.stdout)

    def test_list_rejects_an_empty_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_root = Path(tmp)
            (work_root / "mlir-aie/test/npu-xrt").mkdir(parents=True)

            result = self.run_list(work_root, "does-not-exist")

            self.assertEqual(result.returncode, 1)
            self.assertIn("No tests found matching 'does-not-exist'", result.stderr)


if __name__ == "__main__":
    unittest.main()
