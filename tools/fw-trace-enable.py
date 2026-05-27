#!/usr/bin/env python3
"""Enable/disable amdxdna firmware-trace via DRM_AMDXDNA_SET_FW_TRACE_STATE.

The ioctl is gated by CAP_SYS_ADMIN (DRM_ROOT_ONLY) -- run via pkexec.

Usage:
  pkexec python3 tools/fw-trace-enable.py enable  [--config 0x1]
  pkexec python3 tools/fw-trace-enable.py disable

Once enabled, the kernel hex-dumps each trace entry payload to dmesg
prefixed by '[FW TRACE]:'. Capture with: dmesg -w -T | grep 'FW TRACE'.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import os
import struct
import sys


# struct amdxdna_drm_set_state { u32 param; u32 buffer_size; u64 buffer; }
SET_STATE_FMT = "IIQ"
SET_STATE_SIZE = struct.calcsize(SET_STATE_FMT)  # 16

# struct amdxdna_drm_set_dpt_state { u32 action; u32 config; u64 pad; }
SET_DPT_STATE_FMT = "IIQ"
SET_DPT_STATE_SIZE = struct.calcsize(SET_DPT_STATE_FMT)  # 16

# enum amdxdna_drm_set_param -> DRM_AMDXDNA_SET_FW_TRACE_STATE = 6
DRM_AMDXDNA_SET_FW_TRACE_STATE = 6

# Linux _IOC encoding: dir(2) | size(14) | type(8) | nr(8)
# _IOC(dir, type, nr, size)
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(d: int, t: int, nr: int, size: int) -> int:
    return (d << _IOC_DIRSHIFT) | (t << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)


def _IOWR(t: int, nr: int, size: int) -> int:
    return _IOC(_IOC_READ | _IOC_WRITE, t, nr, size)


# DRM ioctl: type='d' (0x64), nr = DRM_COMMAND_BASE + DRM_AMDXDNA_SET_STATE
DRM_COMMAND_BASE = 0x40
DRM_AMDXDNA_SET_STATE = 8

DRM_IOCTL_AMDXDNA_SET_STATE = _IOWR(0x64, DRM_COMMAND_BASE + DRM_AMDXDNA_SET_STATE, SET_STATE_SIZE)


def set_fw_trace_state(device_path: str, action: int, config: int) -> None:
    """action: 1=enable, 0=disable. config: category bitmask (non-zero on enable)."""
    fd = os.open(device_path, os.O_RDWR)
    try:
        # Build the inner DPT state struct
        dpt_buf = struct.pack(SET_DPT_STATE_FMT, action, config, 0)
        dpt_ctypes = ctypes.create_string_buffer(dpt_buf)
        dpt_addr = ctypes.addressof(dpt_ctypes)

        # Build the outer set_state struct
        outer = struct.pack(SET_STATE_FMT,
                            DRM_AMDXDNA_SET_FW_TRACE_STATE,  # param
                            SET_DPT_STATE_SIZE,              # buffer_size
                            dpt_addr)                        # buffer (pointer)
        outer_buf = ctypes.create_string_buffer(outer)

        fcntl.ioctl(fd, DRM_IOCTL_AMDXDNA_SET_STATE, outer_buf, True)
        print(f"OK: action={action} config=0x{config:x}", file=sys.stderr)
    finally:
        os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["enable", "disable"])
    ap.add_argument("--device", default="/dev/accel/accel0")
    ap.add_argument("--config", default="0xffffffff",
                    help="Category bitmask (hex). Default: all categories (0xffffffff). "
                         "Used only on enable. 0 returns EINVAL.")
    args = ap.parse_args()

    if args.action == "enable":
        config = int(args.config, 0)
        if config == 0:
            print("error: --config must be non-zero on enable", file=sys.stderr)
            return 1
        set_fw_trace_state(args.device, action=1, config=config)
    else:
        set_fw_trace_state(args.device, action=0, config=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
