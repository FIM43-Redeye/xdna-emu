"""End-to-end test for the AIE2 PM-address fault ELF patcher."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PEANO = Path(
    os.environ.get("PEANO_INSTALL_DIR", REPO_ROOT.parent / "llvm-aie" / "install")
)
CLANG = PEANO / "bin" / "clang"
OBJDUMP = PEANO / "bin" / "llvm-objdump"
PATCHER = REPO_ROOT / "tools" / "patch-aie2-pm-address-fault.py"

pytestmark = pytest.mark.skipif(
    not CLANG.is_file() or not OBJDUMP.is_file(), reason="Peano is not installed"
)


def test_patches_only_terminal_done_sequence(tmp_path: Path) -> None:
    """A missing/wrong patch must not silently produce a different core ELF."""
    source = tmp_path / "core.s"
    original = tmp_path / "core.o"
    fault = tmp_path / "core-fault.o"
    source.write_text(
        ".text\n"
        ".globl _start\n"
        "_start:\n"
        "  done\n"
        "  nop\n"
        "  nop\n"
        "  nop\n"
        "  nop\n"
        "  nop\n"
        "  nop\n"
        "  j #0x10\n"
    )
    subprocess.run(
        [str(CLANG), "--target=aie2-none-unknown-elf", "-c", source, "-o", original],
        check=True,
    )
    original_bytes = original.read_bytes()

    patched = subprocess.run(
        [
            sys.executable,
            str(PATCHER),
            "--peano",
            str(PEANO),
            str(original),
            str(fault),
        ],
        capture_output=True,
        text=True,
    )

    assert patched.returncode == 0, patched.stderr
    assert original.read_bytes() == original_bytes
    assert fault.read_bytes() != original_bytes
    disassembly = subprocess.run(
        [str(OBJDUMP), "-d", "--disassemble-zeroes", str(fault)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "j\t#0x4000" in disassembly
    assert "6: 01 00" in disassembly
    assert "nop" in disassembly
