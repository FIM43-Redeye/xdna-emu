#!/usr/bin/env python3
"""Patch Trace_Control0 mode bits in an insts.bin for all 4 trace modes.

Finds WRITE ops targeting core Trace_Control0 (offset 0x340D0) and patches
bits [1:0] to the requested mode. Produces one output file per mode.
"""
import struct
import sys
import os

TRACE_CONTROL0_OFFSET = 0x340D0
MODE_MASK = 0x3

MODE_NAMES = {
    0: "event_time",
    1: "event_pc",
    2: "execution",
    3: "reserved_11",
}


def find_trace_control0_writes(data):
    """Find all WRITE ops targeting core Trace_Control0 in the transaction."""
    numops = struct.unpack_from("<I", data, 8)[0]
    pos = 16
    matches = []
    for op_num in range(numops):
        if pos + 3 > len(data):
            break
        op = data[pos]
        if op == 0:  # WRITE (24 bytes)
            if pos + 24 > len(data):
                break
            regoff = struct.unpack_from("<Q", data, pos + 8)[0]
            value = struct.unpack_from("<I", data, pos + 16)[0]
            tile_offset = regoff & 0xFFFFF
            row = (regoff >> 20) & 0x1F
            col = regoff >> 25
            if tile_offset == TRACE_CONTROL0_OFFSET and row >= 2:
                matches.append({
                    "op_num": op_num, "pos": pos, "value_offset": pos + 16,
                    "col": col, "row": row, "value": value,
                    "current_mode": value & MODE_MASK,
                })
            pos += 24
        elif op == 3:  # MASKWRITE (28 bytes)
            pos += 28
        elif op == 1:  # BLOCKWRITE
            if pos + 16 > len(data):
                break
            size = struct.unpack_from("<I", data, pos + 12)[0]
            pos += size
        elif op == 4:  # MASKPOLL (28 bytes)
            pos += 28
        elif op >= 128:  # Custom
            if pos + 8 > len(data):
                break
            size = struct.unpack_from("<I", data, pos + 4)[0]
            pos += size
        else:
            break
    return matches


def patch_mode(data, matches, mode):
    patched = bytearray(data)
    for m in matches:
        new_val = (m["value"] & ~MODE_MASK) | (mode & MODE_MASK)
        struct.pack_into("<I", patched, m["value_offset"], new_val)
    return patched


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <insts.bin> [output_dir]")
        sys.exit(1)

    inpath = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(inpath) or "."

    with open(inpath, "rb") as f:
        data = f.read()

    matches = find_trace_control0_writes(data)
    if not matches:
        print("No core Trace_Control0 writes found.")
        sys.exit(1)

    print(f"Found {len(matches)} core Trace_Control0 write(s):")
    for m in matches:
        print(f"  op[{m['op_num']}] col={m['col']} row={m['row']} "
              f"val=0x{m['value']:08x} mode={m['current_mode']}")

    os.makedirs(outdir, exist_ok=True)
    for mode in range(4):
        patched = patch_mode(data, matches, mode)
        outpath = os.path.join(outdir, f"insts_mode{mode}_{MODE_NAMES[mode]}.bin")
        with open(outpath, "wb") as f:
            f.write(patched)
        print(f"  Mode {mode} ({MODE_NAMES[mode]:12s}) -> {outpath}")


if __name__ == "__main__":
    main()
