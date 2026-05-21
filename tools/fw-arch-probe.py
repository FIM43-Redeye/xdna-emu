#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# fw-arch-probe.py -- brute-force the instruction-set architecture of a raw
# firmware blob by disassembling sample regions with every candidate
# objdump target and scoring how cleanly each decodes.
#
# A region that is genuine code for architecture X decodes, under X, with
# almost no ".word"/".byte"/"undefined"/"(bad)" filler and with branch
# targets that land inside the sampled window.  Wrong arches produce a high
# filler ratio.  The highest-scoring (arch, region) pairs name the ISA.

import re
import subprocess
import sys

BODY = "/home/triple/npu-work/ghidra-projects/npu-fw/npu-fw-body.bin"

MB_BE = "/home/triple/npu-work/amd-unified-software/gnu/microblaze/lin/bin/microblaze-xilinx-elf-objdump"
MB_LE = ("/home/triple/npu-work/amd-unified-software/gnu/microblaze/"
         "linux_toolchain/lin64_le/bin/microblazeel-xilinx-linux-gnu-objdump")
OD = "/usr/bin/objdump"

# (label, objdump-binary, extra args)
TARGETS = [
    ("microblaze-BE", MB_BE, ["-m", "MicroBlaze", "-EB"]),
    ("microblaze-LE", MB_LE, ["-m", "MicroBlaze", "-EL"]),
    ("arm-LE",        OD,    ["-m", "arm", "-EL"]),
    ("arm-BE",        OD,    ["-m", "arm", "-EB"]),
    ("thumb-LE",      OD,    ["-m", "arm", "-EL", "-M", "force-thumb"]),
    ("aarch64-LE",    OD,    ["-m", "aarch64", "-EL"]),
    ("riscv32-LE",    OD,    ["-m", "riscv:rv32", "-EL"]),
    ("ppc-LE",        OD,    ["-m", "powerpc:common", "-EL"]),
    ("sh-LE",         OD,    ["-m", "sh", "-EL"]),
]

# Candidate code regions from the entropy map (body offsets).
REGIONS = [0x3000, 0x4000, 0x5800, 0x8800, 0xb000, 0xd000,
           0x31000, 0x33000, 0x37800, 0x3b000]

WIN = 0x100  # bytes per sample

BAD_RE = re.compile(r"\.word|\.byte|\.short|\.long|\bbad\b|undefined|illegal|"
                    r"unknown|\.inst|\.4byte|\.2byte", re.I)


def disasm(objdump, args, start, stop):
    cmd = [objdump, "-D", "-b", "binary", *args,
           f"--start-address={start}", f"--stop-address={stop}", BODY]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=20).stdout
    except Exception as e:
        return None, str(e)
    return out, None


def score(out):
    """Return (insn_count, good_count, good_ratio) for a disassembly dump."""
    insn = 0
    good = 0
    for line in out.splitlines():
        # objdump insn lines look like:  "   6000:\t<hex>\t<mnemonic ...>"
        m = re.match(r"\s*[0-9a-f]+:\s+([0-9a-f ]+)\t(.*)$", line)
        if not m:
            continue
        insn += 1
        mnem = m.group(2).strip()
        if not mnem:               # empty mnemonic -> failed decode
            continue
        if BAD_RE.search(mnem):    # filler / undefined
            continue
        good += 1
    ratio = good / insn if insn else 0.0
    return insn, good, ratio


def main():
    print(f"{'region':>10}  " + "  ".join(f"{t[0]:>14}" for t in TARGETS))
    best_overall = (0.0, None, None)
    for region in REGIONS:
        cells = []
        for label, objdump, args in TARGETS:
            out, err = disasm(objdump, args, region, region + WIN)
            if err or out is None:
                cells.append("   ERR")
                continue
            insn, good, ratio = score(out)
            cells.append(f"{ratio*100:5.0f}% ({insn:2d})")
            if ratio > best_overall[0]:
                best_overall = (ratio, label, region)
        print(f"  0x{region:06x}  " + "  ".join(f"{c:>14}" for c in cells))
    print()
    r, lbl, reg = best_overall
    if lbl:
        print(f"BEST: {lbl} at body 0x{reg:06x}  ({r*100:.0f}% clean decode)")
    else:
        print("BEST: no target produced a clean decode -- arch still unknown")


if __name__ == "__main__":
    sys.exit(main())
