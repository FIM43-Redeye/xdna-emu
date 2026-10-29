#!/usr/bin/env python3
"""Analyze bf16 sub-precision characterizer results.

Reconstructs full fp32 from dual SRS outputs and compares HW vs EMU
at the bit level, focusing on sub-bf16 mantissa bits [15:0].
"""

import struct
import sys

def reconstruct_fp32(hi16, lo16):
    """Reconstruct fp32 from SRS shift=16 (upper) and shift=0 (lower)."""
    return ((hi16 & 0xFFFF) << 16) | (lo16 & 0xFFFF)

def fp32_parts(bits):
    """Split fp32 bits into (sign, exponent, mantissa)."""
    return (bits >> 31, (bits >> 23) & 0xFF, bits & 0x7FFFFF)

def analyze_test(test_num, input_data, hw_data, emu_data, config_name):
    """Analyze one test's results."""
    # Parse input: 64 bytes x0 + 64 bytes x2
    x0 = struct.unpack('<32H', input_data[:64])
    x2 = struct.unpack('<32H', input_data[64:128])

    # Parse output: 32 bytes SRS shift=16 + 32 bytes SRS shift=0
    hw_hi = struct.unpack('<16H', hw_data[:32])
    hw_lo = struct.unpack('<16H', hw_data[32:64])
    emu_hi = struct.unpack('<16H', emu_data[:32])
    emu_lo = struct.unpack('<16H', emu_data[32:64])

    print(f"\n{'='*70}")
    print(f"TEST {test_num}: {config_name}")
    print(f"{'='*70}")

    diffs = 0
    for lane in range(8):  # bml0 has 8 fp32 lanes
        hw_fp32 = reconstruct_fp32(hw_hi[lane], hw_lo[lane])
        emu_fp32 = reconstruct_fp32(emu_hi[lane], emu_lo[lane])

        s_hw, e_hw, m_hw = fp32_parts(hw_fp32)
        s_emu, e_emu, m_emu = fp32_parts(emu_fp32)

        match = "MATCH" if hw_fp32 == emu_fp32 else "DIFF"
        if hw_fp32 != emu_fp32:
            diffs += 1

            # Categorize the difference
            bf16_match = (hw_fp32 >> 16) == (emu_fp32 >> 16)
            lo16_hw = hw_fp32 & 0xFFFF
            lo16_emu = emu_fp32 & 0xFFFF

            print(f"  Lane {lane}: {match}")
            print(f"    HW:  0x{hw_fp32:08X}  (sign={s_hw} exp={e_hw:3d} man=0x{m_hw:06X})")
            print(f"    EMU: 0x{emu_fp32:08X}  (sign={s_emu} exp={e_emu:3d} man=0x{m_emu:06X})")
            print(f"    Upper bf16: {'MATCH' if bf16_match else 'DIFF!'}")
            print(f"    Lower 16b:  HW=0x{lo16_hw:04X}  EMU=0x{lo16_emu:04X}  diff={lo16_emu - lo16_hw:+d}")

            # Show mantissa bits breakdown
            man_diff = m_emu - m_hw
            print(f"    Mantissa diff: {man_diff:+d} (0x{abs(man_diff):06X})")

            # Check if the difference is in specific bit ranges
            for name, mask in [("bits[22:16] bf16", 0x7F0000),
                               ("bits[15:7]", 0x00FF80),
                               ("bits[6:0] lsb7", 0x00007F)]:
                hw_part = m_hw & mask
                emu_part = m_emu & mask
                if hw_part != emu_part:
                    print(f"    {name}: HW=0x{hw_part:06X} EMU=0x{emu_part:06X}")
        else:
            # Still print for completeness
            f_val = struct.unpack('f', struct.pack('I', hw_fp32))[0]
            print(f"  Lane {lane}: MATCH  0x{hw_fp32:08X} ({f_val:.6g})")

    print(f"\n  Summary: {diffs} lanes differ out of 8")
    return diffs

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} input.bin hw_output.bin emu_output.bin")
        sys.exit(1)

    with open(sys.argv[1], 'rb') as f:
        input_data = f.read()
    with open(sys.argv[2], 'rb') as f:
        hw_output = f.read()
    with open(sys.argv[3], 'rb') as f:
        emu_output = f.read()

    print(f"Input:  {len(input_data)} bytes")
    print(f"HW out: {len(hw_output)} bytes")
    print(f"EMU out: {len(emu_output)} bytes")

    num_tests = len(input_data) // 128
    test_names = [
        "Dense 4x8x4 (config=0x1d) -- ISA-test failing input",
        "Element-wise 16x2x1 (config=0x3d) -- same input",
        "Dense 4x8x4 -- rounding boundary inputs",
        "Dense 4x8x4 -- mixed-sign cancellation",
    ]

    total_diffs = 0
    for i in range(min(num_tests, len(hw_output) // 64)):
        in_off = (i // 2) * 128 if i < 2 else (i - 1) * 128  # Tests 1&2 share input
        hw_off = i * 64
        emu_off = i * 64

        config_name = test_names[i] if i < len(test_names) else f"Test {i+1}"

        in_slice = input_data[in_off:in_off+128]
        hw_slice = hw_output[hw_off:hw_off+64]
        emu_slice = emu_output[emu_off:emu_off+64]

        if len(hw_slice) < 64 or len(emu_slice) < 64:
            break

        total_diffs += analyze_test(i + 1, in_slice, hw_slice, emu_slice, config_name)

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_diffs} differing lanes across all tests")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
