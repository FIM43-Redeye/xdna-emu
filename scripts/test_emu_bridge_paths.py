#!/usr/bin/env python3

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("emu-bridge-test.sh")


class BridgePathTests(unittest.TestCase):
    def run_bridge(self, work_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["NPU_WORK_DIR"] = str(work_root)
        env["BRIDGE_TEST_RESULTS"] = str(work_root / "results")
        return subprocess.run(
            [str(SCRIPT), *args],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_list(self, work_root: Path, test_filter: str) -> subprocess.CompletedProcess[str]:
        return self.run_bridge(work_root, "--list", test_filter)

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

    def test_compile_expands_current_lit_host_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_root = Path(tmp)
            test_dir = work_root / "mlir-aie/test/npu-xrt/sample"
            test_dir.mkdir(parents=True)
            (test_dir / "test.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
            (test_dir / "aie.mlir").write_text("// cached fixture\n", encoding="utf-8")
            (test_dir / "run.lit").write_text(
                "// RUN: %host_clang %S/test.cpp -o test %host_link_flags\n",
                encoding="utf-8",
            )
            for compiler in ("chess", "peano"):
                build_dir = work_root / f"mlir-aie/build/test/npu-xrt/sample/{compiler}"
                build_dir.mkdir(parents=True)
                (build_dir / "aie_arch.mlir").write_text("// cached fixture\n", encoding="utf-8")
                (build_dir / "aie.xclbin").write_bytes(b"cached")

            result = self.run_bridge(
                work_root,
                "--no-trace",
                "--no-hw",
                "--no-emu",
                "^sample$",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(
                (work_root / "mlir-aie/build/test/npu-xrt/sample/test.exe").is_file(),
                result.stdout,
            )
            self.assertNotIn("FAIL (test.exe)", result.stdout)

    def test_failed_multistep_compile_is_not_served_from_cache_next_run(self) -> None:
        # A multi-step build (ctrl_packet_reconfig_elf: xclbin step, then a
        # ctrlpkt/elf step) leaves the first step's xclbin behind when a later
        # step fails. The next run must not mistake that for a finished build.
        with tempfile.TemporaryDirectory() as tmp:
            work_root = Path(tmp)
            test_dir = work_root / "mlir-aie/test/npu-xrt/twostep"
            test_dir.mkdir(parents=True)
            (test_dir / "test.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
            (test_dir / "aie.mlir").write_text("// fixture\n", encoding="utf-8")
            (test_dir / "run.lit").write_text(
                "// RUN: touch aie.xclbin\n// RUN: false\n", encoding="utf-8"
            )
            build_dir = work_root / "mlir-aie/build/test/npu-xrt/twostep/peano"
            args = ("--no-trace", "--no-hw", "--no-emu", "^twostep$")

            first = self.run_bridge(work_root, *args)
            self.assertIn("COMPILE twostep (peano): FAIL", first.stdout, first.stdout)
            self.assertFalse(
                list(build_dir.glob("*.xclbin")),
                "a failed build must not leave an xclbin that reads as cached",
            )

            second = self.run_bridge(work_root, *args)
            self.assertNotIn("twostep (peano): cached", second.stdout, second.stdout)
            self.assertIn("COMPILE twostep (peano): FAIL", second.stdout, second.stdout)

    def test_a_toolchain_change_invalidates_cached_builds(self) -> None:
        # A rebuilt aiecc must not reuse binaries the previous toolchain built:
        # an unchanged aie_arch.mlir says nothing about which compiler built them.
        with tempfile.TemporaryDirectory() as tmp:
            work_root = Path(tmp)
            test_dir = work_root / "mlir-aie/test/npu-xrt/stamped"
            test_dir.mkdir(parents=True)
            (test_dir / "test.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
            (test_dir / "aie.mlir").write_text("// fixture\n", encoding="utf-8")
            (test_dir / "run.lit").write_text("// RUN: touch aie.xclbin test.exe\n", encoding="utf-8")
            aiecc = work_root / "mlir-aie/build/bin/aiecc"
            aiecc.parent.mkdir(parents=True)
            aiecc.write_text("old\n", encoding="utf-8")
            args = ("--no-trace", "--no-hw", "--no-emu", "^stamped$")

            first = self.run_bridge(work_root, *args)
            self.assertIn("COMPILE stamped (peano): OK", first.stdout, first.stdout)
            second = self.run_bridge(work_root, *args)
            self.assertIn("COMPILE stamped (peano): cached", second.stdout, second.stdout)

            aiecc.write_text("rebuilt\n", encoding="utf-8")
            third = self.run_bridge(work_root, *args)
            self.assertNotIn("stamped (peano): cached", third.stdout, third.stdout)
            self.assertIn("COMPILE stamped (peano): OK", third.stdout, third.stdout)


if __name__ == "__main__":
    unittest.main()
