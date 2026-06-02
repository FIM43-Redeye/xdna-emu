#!/usr/bin/env python3
"""Clear the PF_X (executable) bit on an ELF's PT_GNU_STACK segment, in place.

Equivalent to `execstack -c` (not packaged on this box). ELF64 little-endian.

Used by build-aiesim-bridge.sh to sanitize a local copy of aietools'
libsystemc.so: its RWE stack marker is a spurious missing-.note.GNU-stack
artifact of the QuickThreads assembly, not a real runtime need (proven: SystemC
coroutines run fine without it). Clearing it lets the bridge .so be dlopened
into a normal (non-exec-stack) host -- see
docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md.
"""
import struct
import sys

PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def clear(path: str) -> int:
    with open(path, "r+b") as f:
        data = bytearray(f.read())
        if data[:4] != b"\x7fELF":
            raise SystemExit(f"{path}: not an ELF")
        if data[4] != 2 or data[5] != 1:
            raise SystemExit(f"{path}: expected ELF64 little-endian")
        e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
        e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
        e_phnum = struct.unpack_from("<H", data, 0x38)[0]
        cleared = 0
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            p_type = struct.unpack_from("<I", data, off)[0]
            if p_type == PT_GNU_STACK:
                p_flags = struct.unpack_from("<I", data, off + 4)[0]
                if p_flags & PF_X:
                    struct.pack_into("<I", data, off + 4, p_flags & ~PF_X)
                    cleared += 1
        f.seek(0)
        f.write(data)
        return cleared


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: clear-execstack.py <elf>")
    n = clear(sys.argv[1])
    print(f"cleared PF_X on {n} PT_GNU_STACK segment(s) in {sys.argv[1]}")
