#!/usr/bin/env python3

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run-native-pm-fault-guard.sh")
TEST_NAME = (
    "firmware::boot_tests::guards::"
    "m2c_chained_pm_fault_publishes_native_core_error"
)


class NativePmFaultGuardRunnerTests(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[dict[str, str], Path]:
        firmware = root / "npu.dev.sbin"
        firmware.write_bytes(b"firmware")

        mlir_aie = root / "mlir-aie"
        regdb = mlir_aie / "lib/Dialect/AIE/Util/aie_registers_aie2.json"
        regdb.parent.mkdir(parents=True)
        regdb.write_text("{}\n", encoding="utf-8")

        mlir_build = root / "mlir-build"
        fixture = mlir_build / "test/npu-xrt/add_one_using_dma/chess"
        fixture.mkdir(parents=True)
        (fixture / "aie.xclbin").write_bytes(b"xclbin")
        (fixture / "insts.bin").write_bytes(b"instructions")

        pdi = root / "native-pm-fault.pdi"
        pdi.write_bytes(b"pdi")

        receipt = root / "cargo.receipt"
        fake_bin = root / "bin"
        fake_bin.mkdir()
        cargo = fake_bin / "cargo"
        cargo.write_text(
            "#!/bin/sh\n"
            "{\n"
            "  printf '%s\\n' \"$@\"\n"
            "  printf 'XDNA_FIRMWARE=%s\\n' \"$XDNA_FIRMWARE\"\n"
            "  printf 'MLIR_AIE_PATH=%s\\n' \"$MLIR_AIE_PATH\"\n"
            "  printf 'MLIR_AIE_BUILD=%s\\n' \"$MLIR_AIE_BUILD\"\n"
            "  printf 'XDNA_PM_ERROR_PDI=%s\\n' \"$XDNA_PM_ERROR_PDI\"\n"
            "} >\"$RECEIPT\"\n",
            encoding="utf-8",
        )
        cargo.chmod(0o755)

        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                "RECEIPT": str(receipt),
                "XDNA_FIRMWARE": str(firmware),
                "MLIR_AIE_PATH": str(mlir_aie),
                "MLIR_AIE_BUILD": str(mlir_build),
                "XDNA_PM_ERROR_PDI": str(pdi),
                "NICE_LEVEL": "19",
            }
        )
        return env, receipt

    def run_guard(self, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        self.assertTrue(SCRIPT.is_file(), "native PM-fault guard runner is missing")
        return subprocess.run(
            [str(SCRIPT)],
            cwd=SCRIPT.parents[1],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_invokes_only_the_explicit_ignored_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, receipt = self.make_inputs(Path(tmp))

            result = self.run_guard(env)

            self.assertEqual(result.returncode, 0, result.stderr)
            lines = receipt.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                lines[:9],
                [
                    "test",
                    "-p",
                    "xdna-emu",
                    "--lib",
                    TEST_NAME,
                    "--",
                    "--ignored",
                    "--exact",
                    "--nocapture",
                ],
            )
            self.assertIn(f"MLIR_AIE_BUILD={env['MLIR_AIE_BUILD']}", lines)
            self.assertIn(f"XDNA_PM_ERROR_PDI={env['XDNA_PM_ERROR_PDI']}", lines)

    def test_missing_fault_pdi_fails_before_cargo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, receipt = self.make_inputs(Path(tmp))
            env.pop("XDNA_PM_ERROR_PDI")

            result = self.run_guard(env)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("XDNA_PM_ERROR_PDI must name the native PM-fault PDI", result.stderr)
            self.assertFalse(receipt.exists(), "cargo ran despite failed preflight")

    def test_blank_explicit_build_root_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, receipt = self.make_inputs(Path(tmp))
            env["MLIR_AIE_BUILD"] = ""

            result = self.run_guard(env)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MLIR_AIE_BUILD is set but blank", result.stderr)
            self.assertFalse(receipt.exists(), "cargo ran despite failed preflight")


if __name__ == "__main__":
    unittest.main()
