#!/usr/bin/env python3
"""Analyze output from the sparse mask characterization test.

The test kernel constructs A=all-0x01 and B=all-0x01 internally, then runs
vmac with each of 16 mask patterns (0x0..0xF in bits 0-3 of q0).

Config: i8xi8 sparse (4x16x8), zero_acc=1, signed.

With A=0x10 everywhere and B=0x10 before masking:
  C[r][c] = sum(k=0..15) 0x10 * B_effective[k][c]

After SRS with shift=0 (hardware BIAS=4): output = C >> 4.
Each active inner position contributes 0x10*0x10 >> 4 = 16 to the output.
So the output value = 16 * (number of active inner positions for that column).

Output: 16 patterns x 64 bytes = 1024 bytes.
Each 64 bytes = 32 x int16 (bml0 lanes 0-15 + bmh0 lanes 16-31).
Acc layout: lane = row*8 + col, so lanes 0-7 = row 0, lanes 8-15 = row 1, etc.
"""

import argparse
import struct
import sys


def analyze(path: str, label: str):
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 1024:
        print(f"ERROR: expected >= 1024 bytes, got {len(data)}")
        sys.exit(1)

    print(f"\n=== Sparse Mask Characterization: {label} ===")
    print(f"Config: i8xi8 sparse 4x16x8, A=B=all-0x10, zero_acc, signed")
    print(f"Each output lane = sum of column c across active inner positions")
    print()

    for p in range(16):
        offset = p * 64
        # 32 x int16 values
        vals = struct.unpack_from("<32h", data, offset)

        mask_str = f"{p:04b}"
        print(f"Pattern {p:2d} (0b{mask_str}):")
        for r in range(4):
            row_vals = vals[r*8:(r+1)*8]
            row_str = " ".join(f"{v:4d}" for v in row_vals)
            print(f"  row {r}: [{row_str}]")
        print()

    # Summary table: just row 0 for each pattern
    print("=== Summary (row 0 only) ===")
    print(f"{'Pattern':>8s}  {'c0':>4s} {'c1':>4s} {'c2':>4s} {'c3':>4s} {'c4':>4s} {'c5':>4s} {'c6':>4s} {'c7':>4s}")
    for p in range(16):
        offset = p * 64
        vals = struct.unpack_from("<32h", data, offset)
        row0 = vals[:8]
        mask_str = f"0b{p:04b}"
        cols = " ".join(f"{v:4d}" for v in row0)
        print(f"{mask_str:>8s}  {cols}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse mask characterization")
    parser.add_argument("--gen-input", metavar="PATH",
                        help="(No longer needed -- kernel is self-contained)")
    parser.add_argument("--analyze-hw", metavar="PATH", help="Analyze HW output")
    parser.add_argument("--analyze-emu", metavar="PATH", help="Analyze EMU output")
    args = parser.parse_args()

    if args.gen_input:
        print("Input generation no longer needed (kernel constructs data internally)")
    if args.analyze_hw:
        analyze(args.analyze_hw, "Hardware")
    if args.analyze_emu:
        analyze(args.analyze_emu, "Emulator")
    if not any([args.gen_input, args.analyze_hw, args.analyze_emu]):
        parser.print_help()
