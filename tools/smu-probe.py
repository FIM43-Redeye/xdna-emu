#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# smu-probe.py -- read and (optionally) exec commands on the Phoenix NPU SMU
# via direct MMIO writes to /sys/bus/pci/devices/<bdf>/resource0.
#
# The xdna driver only wraps SMU commands 0x3-0x8 (POWER_ON/OFF,
# SET_*CLK_FREQ, SET_*_DPMLEVEL).  This tool exposes the rest of the
# command space so we can discover whether the NPU SMU accepts
# additional commands beyond what the driver uses.
#
# Requires root (pkexec).  Default register offsets are for NPU1
# (Phoenix); they match xdna-driver/src/driver/amdxdna/npu1_regs.c.
#
# Subcommands:
#   read                     -- snapshot the current SMU register set
#   exec CMD [ARG]           -- send an SMU command (CMD in hex/dec),
#                               wait up to 1s for a response, print
#                               cmd/arg/resp/out and a decoded meaning
#
# WARNING: writing unknown command numbers is undefined behavior.  AMD
# documents only 0x3-0x8 publicly.  An unknown command could in
# principle trigger a destructive action -- SoC reset, hang, etc.
# Don't run exec mode on a healthy device.  This tool exists so we can
# probe a wedged device where we have nothing to lose.

import argparse
import errno
import mmap
import os
import struct
import sys
import time

BDF_DEFAULT = "0000:c6:00.1"
BAR0_SIZE = 0x80000  # 512 KiB, from sysfs `resource` line on this dev box

# Offsets within BAR0 (== MPNPU_PUB_SCRATCH_* - MPNPU_APERTURE0_BASE).
# Source: xdna-driver/src/driver/amdxdna/npu1_regs.c.
SMU_OFFSETS = {
    "CMD":  0x100AC,   # MPNPU_PUB_SCRATCH5
    "RESP": 0x100B0,   # MPNPU_PUB_SCRATCH6
    "ARG":  0x100B4,   # MPNPU_PUB_SCRATCH7 (also OUT)
    "INTR": 0x10094,   # MPNPU_PUB_PWRMGMT_INTR
}

# PSP registers (informational; read only in this tool).
PSP_OFFSETS = {
    "CMD":   0x100A0,  # MPNPU_PUB_SCRATCH2
    "ARG0":  0x100A4,  # MPNPU_PUB_SCRATCH3
    "ARG1":  0x100A8,  # MPNPU_PUB_SCRATCH4
}

# Response code -> (name, description).  Modern PPSMC convention; see
# amdgpu's drivers/gpu/drm/amd/pm/swsmu/smu_cmn.c:77-83 for the canonical
# enumeration.  The xdna driver only knows SMU_RESULT_OK = 0x01.
RESP_CODES = {
    0x00: ("NONE",         "SMU never wrote a response (poll target -- SMU dead if persistent)"),
    0x01: ("OK",           "Success"),
    0xFB: ("DEBUG_END",    "Debug command terminus"),
    0xFC: ("BUSY_OTHER",   "SMU busy with another command (transient -- retry may help)"),
    0xFD: ("BAD_PREREQ",   "Prerequisites not met (state machine in wrong state)"),
    0xFE: ("UNKNOWN_CMD",  "SMU does not recognize this command number"),
    0xFF: ("CMD_FAIL",     "Generic 'I tried and failed'"),
}

# Known driver-defined SMU commands -- for informational output.
CMD_KNOWN = {
    0x3: "POWER_ON",
    0x4: "POWER_OFF",
    0x5: "SET_MPNPUCLK_FREQ",
    0x6: "SET_HCLK_FREQ",
    0x7: "SET_SOFT_DPMLEVEL",
    0x8: "SET_HARD_DPMLEVEL",
}


def decode_resp(resp):
    name, desc = RESP_CODES.get(resp, ("?", f"undocumented response code 0x{resp:02X}"))
    return f"0x{resp:08X}  [{name}] {desc}"


def open_bar(bdf):
    path = f"/sys/bus/pci/devices/{bdf}/resource0"
    if not os.path.exists(path):
        sys.exit(f"FATAL: BAR0 missing at {path}; is the device PCI-present?")
    try:
        f = open(path, "r+b")
    except PermissionError:
        sys.exit("FATAL: BAR0 read needs root; run under pkexec.")
    return f, mmap.mmap(f.fileno(), BAR0_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)


def read32(mem, off):
    return struct.unpack("<I", mem[off:off + 4])[0]


def write32(mem, off, val):
    mem[off:off + 4] = struct.pack("<I", val)


def cmd_read(args):
    f, mem = open_bar(args.bdf)
    try:
        print(f"BAR0  bdf={args.bdf}  size={BAR0_SIZE} bytes\n")
        print("== SMU registers (current state) ==")
        smu_cmd = read32(mem, SMU_OFFSETS["CMD"])
        smu_resp = read32(mem, SMU_OFFSETS["RESP"])
        smu_arg = read32(mem, SMU_OFFSETS["ARG"])
        smu_intr = read32(mem, SMU_OFFSETS["INTR"])
        cmd_name = CMD_KNOWN.get(smu_cmd, "?")
        print(f"  SMU_CMD   (0x{SMU_OFFSETS['CMD']:05X}) = 0x{smu_cmd:08X}   [{cmd_name}]")
        print(f"  SMU_RESP  (0x{SMU_OFFSETS['RESP']:05X}) = {decode_resp(smu_resp & 0xFF)}")
        print(f"  SMU_ARG   (0x{SMU_OFFSETS['ARG']:05X}) = 0x{smu_arg:08X}   (also SMU_OUT)")
        print(f"  SMU_INTR  (0x{SMU_OFFSETS['INTR']:05X}) = 0x{smu_intr:08X}")
        print()
        print("== PSP registers (informational) ==")
        for name, off in PSP_OFFSETS.items():
            val = read32(mem, off)
            print(f"  PSP_{name:5s} (0x{off:05X}) = 0x{val:08X}")
    finally:
        mem.close()
        f.close()


def cmd_exec(args):
    cmd = int(args.cmd, 0)
    arg = int(args.arg, 0) if args.arg else 0
    if not args.yes_destructive and cmd not in CMD_KNOWN and cmd not in (0x1, 0x2):
        sys.exit(f"refusing to send undocumented cmd 0x{cmd:X} without --yes-destructive; "
                 "only 0x1, 0x2, and driver-known cmds (0x3-0x8) are allowed by default.")
    f, mem = open_bar(args.bdf)
    try:
        cmd_name = CMD_KNOWN.get(cmd, "?")
        print(f"[exec] cmd=0x{cmd:X} [{cmd_name}] arg=0x{arg:X}")
        # Mirror aie_smu_exec() in xdna-driver/src/driver/amdxdna/aie_smu.c:
        #   clear RESP, write ARG, write CMD, kick INTR (0 then 1), poll RESP.
        write32(mem, SMU_OFFSETS["RESP"], 0)
        write32(mem, SMU_OFFSETS["ARG"],  arg)
        write32(mem, SMU_OFFSETS["CMD"],  cmd)
        write32(mem, SMU_OFFSETS["INTR"], 0)
        write32(mem, SMU_OFFSETS["INTR"], 1)
        t0 = time.monotonic()
        timeout_s = 1.0
        interval_s = 0.020
        resp = 0
        while time.monotonic() - t0 < timeout_s:
            resp = read32(mem, SMU_OFFSETS["RESP"])
            if resp != 0:
                break
            time.sleep(interval_s)
        elapsed_ms = (time.monotonic() - t0) * 1000
        out_val = read32(mem, SMU_OFFSETS["ARG"])
        if resp == 0:
            print(f"  ! TIMEOUT after {elapsed_ms:.1f} ms -- SMU did not respond")
            print(f"  SMU_RESP = 0x00000000 [NONE] (SMU appears dead or command silently dropped)")
            sys.exit(2)
        print(f"  SMU_RESP = {decode_resp(resp & 0xFF)}  (after {elapsed_ms:.1f} ms)")
        print(f"  SMU_OUT  = 0x{out_val:08X}")
    finally:
        mem.close()
        f.close()


def main():
    parser = argparse.ArgumentParser(
        description="Read and (optionally) exec commands on the Phoenix NPU SMU via direct MMIO.")
    parser.add_argument("--bdf", default=BDF_DEFAULT, help=f"PCI BDF (default {BDF_DEFAULT})")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("read", help="snapshot current SMU register state")
    ex = sub.add_parser("exec", help="send an SMU command and decode the response")
    ex.add_argument("cmd", help="command number (hex/decimal)")
    ex.add_argument("arg", nargs="?", default="0", help="argument value (hex/decimal, default 0)")
    ex.add_argument("--yes-destructive", action="store_true",
                    help="allow sending command numbers outside the safe set (0x1, 0x2, 0x3-0x8)")
    args = parser.parse_args()
    if args.mode == "read":
        cmd_read(args)
    elif args.mode == "exec":
        cmd_exec(args)


if __name__ == "__main__":
    main()
