#!/usr/bin/env python3
"""Replace one terminal AIE2 ``done`` with ``j #0x4000``.

The replacement encoding and ELF layout come from the selected Peano tools.
The input is never modified, and any unexpected instruction layout is fatal.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


class PatchError(Exception):
    """The ELF or resolved toolchain does not satisfy the patch contract."""


_INSTRUCTION = re.compile(
    r"^\s*([0-9a-fA-F]+):\s*((?:[0-9a-fA-F]{2}\s+)+)\s*(.+)$"
)
_TEXT_SECTION = re.compile(
    r"^\s*\[\s*\d+\]\s+\.text\s+\S+\s+"
    r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\b"
)


def _run(command: list[object]) -> str:
    try:
        return subprocess.run(
            [str(part) for part in command],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "") or ""
        raise PatchError(f"command failed: {' '.join(map(str, command))}: {stderr.strip()}") from error


def _instructions(disassembly: str) -> list[tuple[int, bytes, str]]:
    result = []
    for line in disassembly.splitlines():
        match = _INSTRUCTION.match(line)
        if match:
            result.append(
                (
                    int(match.group(1), 16),
                    bytes.fromhex(match.group(2)),
                    match.group(3).strip(),
                )
            )
    return result


def _mnemonic(assembly: str) -> str:
    return assembly.split(";", 1)[0].split()[0]


def _terminal_done(disassembly: str) -> tuple[int, bytes]:
    instructions = _instructions(disassembly)
    done = [instruction for instruction in instructions if _mnemonic(instruction[2]) == "done"]
    if len(done) != 1:
        raise PatchError(f"expected one done instruction, found {len(done)}")
    pc, encoding, _ = done[0]
    if len(encoding) != 4:
        raise PatchError(f"done at 0x{pc:x} is {len(encoding)} bytes, expected 4")

    by_pc = {instruction[0]: instruction for instruction in instructions}
    for index in range(6):
        nop_pc = pc + 4 + index * 2
        instruction = by_pc.get(nop_pc)
        if instruction is None or len(instruction[1]) != 2 or _mnemonic(instruction[2]) != "nop":
            raise PatchError(
                f"expected a two-byte nop at 0x{nop_pc:x} after terminal done"
            )
    return pc, encoding + by_pc[pc + 4][1]


def _text_layout(section_headers: str) -> tuple[int, int, int]:
    matches = [
        match for line in section_headers.splitlines()
        if (match := _TEXT_SECTION.match(line))
    ]
    if len(matches) != 1:
        raise PatchError(f"expected one .text section, found {len(matches)}")
    address, offset, size = (int(value, 16) for value in matches[0].groups())
    return address, offset, size


def _jump_encoding(clang: Path, objdump: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="aie2-pm-address-jump-") as directory:
        directory = Path(directory)
        source = directory / "jump.s"
        obj = directory / "jump.o"
        source.write_text(".text\n.globl _start\n_start:\n  j #0x4000\n")
        _run([clang, "--target=aie2-none-unknown-elf", "-c", source, "-o", obj])
        instructions = _instructions(
            _run([objdump, "-d", "--disassemble-zeroes", obj])
        )
    candidates = [
        encoding
        for pc, encoding, assembly in instructions
        if pc == 0 and _mnemonic(assembly) == "j" and "#0x4000" in assembly
    ]
    if len(candidates) != 1 or len(candidates[0]) != 6:
        raise PatchError("Peano did not derive one six-byte j #0x4000 encoding")
    return candidates[0]


def patch(input_elf: Path, output_elf: Path, peano: Path) -> int:
    if not input_elf.is_file():
        raise PatchError(f"input ELF does not exist: {input_elf}")
    if input_elf.resolve() == output_elf.resolve():
        raise PatchError("input and output ELF paths must differ")
    if output_elf.exists():
        raise PatchError(f"refusing to overwrite output ELF: {output_elf}")

    clang = peano / "bin" / "clang"
    objdump = peano / "bin" / "llvm-objdump"
    readelf = peano / "bin" / "llvm-readelf"
    for tool in (clang, objdump, readelf):
        if not tool.is_file():
            raise PatchError(f"missing Peano tool: {tool}")

    disassembly = _run([objdump, "-d", "--disassemble-zeroes", input_elf])
    done_pc, expected = _terminal_done(disassembly)
    section_address, section_offset, section_size = _text_layout(
        _run([readelf, "-S", "--wide", input_elf])
    )
    relative_pc = done_pc - section_address
    if relative_pc < 0 or relative_pc + len(expected) > section_size:
        raise PatchError(f"done PC 0x{done_pc:x} is outside .text")

    replacement = _jump_encoding(clang, objdump)
    data = input_elf.read_bytes()
    patch_offset = section_offset + relative_pc
    if data[patch_offset : patch_offset + len(expected)] != expected:
        raise PatchError("ELF bytes disagree with Peano disassembly at terminal done")

    patched = bytearray(data)
    patched[patch_offset : patch_offset + len(replacement)] = replacement
    output_elf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_elf.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(patched)
    try:
        shutil.copymode(input_elf, temporary_path)
        check = _run([objdump, "-d", "--disassemble-zeroes", temporary_path])
        instructions = {pc: (encoding, assembly) for pc, encoding, assembly in _instructions(check)}
        encoding, assembly = instructions.get(done_pc, (b"", ""))
        if encoding != replacement or _mnemonic(assembly) != "j" or "#0x4000" not in assembly:
            raise PatchError("patched ELF did not disassemble as j #0x4000")
        for index in range(5):
            nop_pc = done_pc + 6 + index * 2
            encoding, assembly = instructions.get(nop_pc, (b"", ""))
            if len(encoding) != 2 or _mnemonic(assembly) != "nop":
                raise PatchError(f"patched ELF lost delay-slot nop at 0x{nop_pc:x}")
        os.replace(temporary_path, output_elf)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"patched terminal done at PC 0x{done_pc:x} to j #0x4000")
    return done_pc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peano", required=True, type=Path)
    parser.add_argument("input_elf", type=Path)
    parser.add_argument("output_elf", type=Path)
    args = parser.parse_args(argv)
    try:
        patch(args.input_elf, args.output_elf, args.peano)
    except PatchError as error:
        print(f"patch-aie2-pm-address-fault: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
