#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# fw-alive-probe.py -- read the NPU FW alive watchdog (and optionally
# the mgmt_mbox_chann_info struct it points at) directly via BAR2 MMIO.
# Lets us inspect FW liveness when the driver is unloaded or has failed
# to probe -- conditions under which the usual driver paths are dead.
#
# Layout (Phoenix, from xdna-driver/drivers/accel/amdxdna/{aie2_pci,npu1_regs}.[ch]):
#   BAR2 = SRAM aperture, base = MPNPU_APERTURE1_BASE (0x3080000)
#   FW_ALIVE_OFF = MPNPU_SRAM_I2X_MAILBOX_15 = 0x30BF000
#       -> BAR2-relative offset = 0x3F000
#
# FW writes a non-zero device-address into FW_ALIVE_OFF when ready;
# that value points at a mgmt_mbox_chann_info struct (also in SRAM)
# whose .magic field is MGMT_MBOX_MAGIC = 0x55504e5f ("_NPU").  The
# driver reads the struct, then zeros FW_ALIVE_OFF before normal
# operation, so a zero value means either (a) FW never came up
# this boot or (b) the driver already consumed the handshake.

import argparse
import mmap
import os
import struct
import sys
import time

BDF_DEFAULT = "0000:c6:00.1"
BAR2_SIZE_DEFAULT = 0x40000  # 256 KiB on Phoenix; sysfs `resource` reports it

# BAR2-relative offsets.  SRAM aperture base = 0x3080000.
FW_ALIVE_OFF = 0x30BF000 - 0x3080000  # 0x3F000
SRAM_BASE_DEV_ADDR = 0x3080000

MGMT_MBOX_MAGIC = 0x55504E5F  # "_NPU" little-endian

# mgmt_mbox_chann_info layout (16 u32s).
INFO_FIELDS = [
    "x2i_tail", "x2i_head", "x2i_buf", "x2i_buf_sz",
    "i2x_tail", "i2x_head", "i2x_buf", "i2x_buf_sz",
    "magic", "msi_id", "prot_major", "prot_minor",
    "rsvd0", "rsvd1", "rsvd2", "rsvd3",
]
INFO_SIZE = 4 * len(INFO_FIELDS)


def open_bar(bdf, bar_index, size):
    path = f"/sys/bus/pci/devices/{bdf}/resource{bar_index}"
    if not os.path.exists(path):
        sys.exit(f"FATAL: BAR{bar_index} missing at {path}; is the device PCI-present?")
    try:
        f = open(path, "r+b")
    except PermissionError:
        sys.exit(f"FATAL: BAR{bar_index} read needs root; run under pkexec.")
    return f, mmap.mmap(f.fileno(), size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)


def read32(mem, off):
    return struct.unpack("<I", mem[off:off + 4])[0]


def dev_addr_to_bar_off(addr):
    """Convert SRAM device address (e.g. 0x30BF000) to BAR2-relative offset."""
    return addr - SRAM_BASE_DEV_ADDR


def cmd_read(args):
    f, mem = open_bar(args.bdf, 2, args.bar_size)
    try:
        n = max(1, args.repeats)
        print(f"BAR2  bdf={args.bdf}  size=0x{args.bar_size:X} bytes")
        print(f"FW_ALIVE_OFF (BAR2+0x{FW_ALIVE_OFF:05X}, dev=0x{SRAM_BASE_DEV_ADDR + FW_ALIVE_OFF:08X})\n")

        samples = []
        for i in range(n):
            v = read32(mem, FW_ALIVE_OFF)
            t = time.monotonic()
            samples.append((t, v))
            if n > 1:
                print(f"  [t={t:.3f}s] FW_ALIVE = 0x{v:08X}")
            if i < n - 1:
                time.sleep(args.interval)
        if n == 1:
            t, v = samples[-1]
            print(f"  FW_ALIVE = 0x{v:08X}")

        v_last = samples[-1][1]
        changed = any(samples[i][1] != samples[0][1] for i in range(1, n))
        if changed:
            print(f"  Observed change across {n} samples -- FW is writing here actively.")
        elif v_last == 0:
            print("  Verdict: zero -- FW never came up this boot, OR driver already consumed handshake.")
        else:
            print(f"  Verdict: non-zero, stable -- FW handshake present, awaiting driver consumption.")

        # If non-zero, dereference and dump the mgmt_mbox_chann_info struct.
        if v_last != 0 and not args.no_deref:
            info_off = dev_addr_to_bar_off(v_last)
            print(f"\n== mgmt_mbox_chann_info @ dev=0x{v_last:08X} (BAR2+0x{info_off:05X}) ==")
            if info_off < 0 or info_off + INFO_SIZE > args.bar_size:
                print(f"  WARN: dereference offset 0x{info_off:X} outside BAR2 [0, 0x{args.bar_size:X}); skipping.")
                return
            info = {}
            for i, name in enumerate(INFO_FIELDS):
                info[name] = read32(mem, info_off + i * 4)
                print(f"  {name:12s} = 0x{info[name]:08X}")
            print()
            if info["magic"] == MGMT_MBOX_MAGIC:
                print(f"  magic OK ({MGMT_MBOX_MAGIC:#010x} = '_NPU') -- FW handshake intact.")
            else:
                print(f"  magic MISMATCH (got 0x{info['magic']:08X}, want 0x{MGMT_MBOX_MAGIC:08X}) -- struct likely garbage.")
            print(f"  protocol: {info['prot_major']}.{info['prot_minor']}")
    finally:
        mem.close()
        f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    ap.add_argument("--bdf", default=BDF_DEFAULT, help=f"PCI BDF (default {BDF_DEFAULT})")
    ap.add_argument("--bar-size", type=lambda s: int(s, 0), default=BAR2_SIZE_DEFAULT,
                    help=f"BAR2 size in bytes (default 0x{BAR2_SIZE_DEFAULT:X})")
    ap.add_argument("--repeats", type=int, default=5,
                    help="number of samples to take (default 5)")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="seconds between samples (default 0.5)")
    ap.add_argument("--no-deref", action="store_true",
                    help="don't follow non-zero FW_ALIVE pointer into the mbox info struct")
    args = ap.parse_args()
    cmd_read(args)


if __name__ == "__main__":
    main()
