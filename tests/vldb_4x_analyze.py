#!/usr/bin/env python3
"""Analyze VLDB_4x characterizer output.

Usage:
    python3 tests/vldb_4x_analyze.py <hw_output.bin> [emu_output.bin]

The characterizer (vldb_4x_characterizer.s) writes 8 test results to the
output buffer. This script reads the output and compares against predicted
gather-load results to determine the address decoding behavior.

Input buffer: 256 bytes of PRNG data (seed 42).
Output layout:
    [0:32]    Test 1: vldb.4x32.lo  (addrs from lo half, 8-byte aligned)
    [32:64]   Test 2: vldb.4x32.hi  (addrs from hi half, 8-byte aligned)
    [64:96]   Test 3: vldb.4x16.lo  (addrs from lo half, 4-byte aligned)
    [96:128]  Test 4: vldb.4x16.hi  (addrs from hi half, 4-byte aligned)
    [128:160] Test 5: vldb.4x64.lo  (addrs from lo half, 16-byte aligned)
    [160:192] Test 6: vldb.4x64.hi  (addrs from hi half, 16-byte aligned)
    [192:224] Test 7: address vector (for verification)
    [224:256] Test 8: vldb.4x32.lo with all 4 addrs = p0+0x00
"""

import struct
import sys


def fill_prng(n: int, seed: int = 42) -> bytes:
    """Deterministic PRNG matching test_host.cpp."""
    state = seed
    buf = bytearray(n)
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        buf[i] = (state >> 16) & 0xFF
    return bytes(buf)


def read_u32s(data: bytes, offset: int, count: int) -> list[int]:
    return [struct.unpack_from("<I", data, offset + i * 4)[0] for i in range(count)]


def hex_words(words: list[int]) -> str:
    return " ".join(f"{w:08x}" for w in words)


def predict_gather(input_data: bytes, base_addr: int, addrs: list[int],
                   align_mask: int) -> list[int]:
    """Predict gather-load result: read 64 bits from each aligned address."""
    result = []
    for addr in addrs:
        aligned = addr & align_mask
        # Offset within input buffer
        offset = aligned - base_addr
        if 0 <= offset <= len(input_data) - 8:
            lo = struct.unpack_from("<I", input_data, offset)[0]
            hi = struct.unpack_from("<I", input_data, offset + 4)[0]
            result.extend([lo, hi])
        else:
            result.extend([0, 0])  # out-of-range
    return result


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hw_output.bin> [emu_output.bin]")
        sys.exit(1)

    hw_path = sys.argv[1]
    emu_path = sys.argv[2] if len(sys.argv) > 2 else None

    hw = open(hw_path, "rb").read()
    emu = open(emu_path, "rb").read() if emu_path else None
    input_data = fill_prng(256)

    # Test 7: address vector -- tells us the actual p0 base address
    addr_vec = read_u32s(hw, 192, 8)
    base_addr = addr_vec[0]  # p0 + 0x00
    print(f"Address vector (test 7): {hex_words(addr_vec)}")
    print(f"Input buffer base (p0): 0x{base_addr:08x}")
    print(f"Expected offsets: +0x00, +0x20, +0x40, +0x60, +0x80, +0xA0, +0xC0, +0xE0")
    expected_addrs = [base_addr + off for off in [0, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0]]
    print(f"Expected addrs:  {hex_words(expected_addrs)}")
    addr_match = addr_vec == expected_addrs
    print(f"Address vector correct: {addr_match}")
    print()

    tests = [
        ("vldb.4x32.lo", 0, 0xFFFFFFF8, addr_vec[0:4]),
        ("vldb.4x32.hi", 32, 0xFFFFFFF8, addr_vec[4:8]),
        ("vldb.4x16.lo", 64, 0xFFFFFFFC, addr_vec[0:4]),
        ("vldb.4x16.hi", 96, 0xFFFFFFFC, addr_vec[4:8]),
        ("vldb.4x64.lo", 128, 0xFFFFFFF0, addr_vec[0:4]),
        ("vldb.4x64.hi", 160, 0xFFFFFFF0, addr_vec[4:8]),
    ]

    for name, out_off, align_mask, addrs in tests:
        hw_result = read_u32s(hw, out_off, 8)
        predicted = predict_gather(input_data, base_addr, addrs, align_mask)
        match = hw_result == predicted

        print(f"--- {name} (align mask 0x{align_mask:08x}) ---")
        print(f"  Addresses:  {hex_words(addrs)}")
        aligned = [a & align_mask for a in addrs]
        offsets = [a - base_addr for a in aligned]
        print(f"  Aligned:    {hex_words(aligned)}")
        print(f"  Offsets:    {offsets}")
        print(f"  HW result:  {hex_words(hw_result)}")
        print(f"  Predicted:  {hex_words(predicted)}")
        print(f"  Match: {match}")

        if not match:
            # Try to identify where each 64-bit chunk came from
            for i in range(4):
                chunk = struct.pack("<II", hw_result[i*2], hw_result[i*2+1])
                # Search input for this chunk
                found_at = None
                for off in range(0, len(input_data) - 8):
                    if input_data[off:off+8] == chunk:
                        found_at = off
                        break
                if chunk == b'\x00' * 8:
                    print(f"    chunk[{i}]: zeros")
                elif found_at is not None:
                    print(f"    chunk[{i}]: found at input offset 0x{found_at:02x}")
                else:
                    print(f"    chunk[{i}]: {chunk.hex()} -- NOT found in input")

        if emu:
            emu_result = read_u32s(emu, out_off, 8)
            emu_match = emu_result == hw_result
            print(f"  EMU result: {hex_words(emu_result)}")
            print(f"  EMU==HW: {emu_match}")

        print()

    # Test 8: same address in all 4 slots
    print("--- Test 8: vldb.4x32.lo with all addrs = p0+0x00 ---")
    hw8 = read_u32s(hw, 224, 8)
    predicted8 = predict_gather(input_data, base_addr,
                                [base_addr] * 4, 0xFFFFFFF8)
    match8 = hw8 == predicted8
    print(f"  HW result:  {hex_words(hw8)}")
    print(f"  Predicted:  {hex_words(predicted8)}")
    print(f"  Match: {match8}")
    if emu:
        emu8 = read_u32s(emu, 224, 8)
        print(f"  EMU result: {hex_words(emu8)}")
        print(f"  EMU==HW: {emu8 == hw8}")


if __name__ == "__main__":
    main()
