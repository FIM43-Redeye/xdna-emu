#!/usr/bin/env python3
"""Analyze bf16 element-wise characterizer output.

Usage:
    python3 tests/bf16_elemwise_analyze.py <hw_output.bin> [emu_output.bin]

Decodes the output from bf16_elemwise_characterizer.s to determine
the A-side and B-side element permutation for variant=1 (config=0x3d)
bf16 multiply with 16x2x1 geometry.

Uses a Sidon set (all pairwise sums are unique) as input values.
Full fp32 accumulator values are reconstructed from two SRS passes
(shift=16 for high 16 bits, shift=0 for low 16 bits).

Output layout (192 bytes):
    [0:32]    Test 1 high (A=Sidon, B=ones)
    [32:64]   Test 1 low
    [64:96]   Test 2 high (A=ones, B=Sidon)
    [96:128]  Test 2 low
    [128:160] Test 3 high (A=B=Sidon)
    [160:192] Test 3 low
"""

import struct
import sys


# Sidon set: all pairwise sums are unique, all values are bf16-exact.
SIDON = [1, 2, 4, 8, 13, 21, 31, 45, 66, 81, 97, 123, 148, 182,
         204, 252, 290, 364, 410, 482, 536, 636, 788, 876, 916,
         1080, 1288, 1456, 1640, 1896, 2032, 2400]

assert len(SIDON) == 32


def reconstruct_fp32_lanes(data: bytes, hi_offset: int, lo_offset: int, count: int = 16) -> list[float]:
    """Reconstruct fp32 values from two SRS passes (shift=16 and shift=0).

    SRS s16.s32 with shift=N extracts: (s32_lane >> N) as s16, then sign-extends to 16 bits.
    With shift=16: output = (fp32_bits >> 16) as s16 = high 16 bits (sign-extended)
    With shift=0:  output = fp32_bits as s16 = low 16 bits (sign-extended)

    Reconstruction: fp32_bits = (hi << 16) | (lo & 0xFFFF)
    """
    results = []
    for i in range(count):
        hi_s16 = struct.unpack_from("<h", data, hi_offset + i * 2)[0]  # signed
        lo_s16 = struct.unpack_from("<h", data, lo_offset + i * 2)[0]  # signed
        # Reconstruct: high 16 bits from hi, low 16 bits from lo
        fp32_bits = ((hi_s16 & 0xFFFF) << 16) | (lo_s16 & 0xFFFF)
        results.append(struct.unpack("<f", struct.pack("<I", fp32_bits & 0xFFFFFFFF))[0])
    return results


def read_raw_u16s(data: bytes, offset: int, count: int = 16) -> list[int]:
    """Read raw unsigned 16-bit values."""
    return [struct.unpack_from("<H", data, offset + i * 2)[0] for i in range(count)]


def find_sidon_pair(target: float) -> list[tuple[int, int]]:
    """Find all unordered pairs (i, j) from SIDON where SIDON[i] + SIDON[j] == target.

    Since SIDON is a Sidon set, there should be at most ONE unordered pair.
    We return ordered pairs (both orderings) for completeness.
    """
    results = []
    target_int = round(target)
    if abs(target - target_int) > 0.5:
        return results  # not an integer sum
    for i in range(32):
        for j in range(32):
            if SIDON[i] + SIDON[j] == target_int:
                results.append((i, j))
    return results


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hw_output.bin> [emu_output.bin]")
        sys.exit(1)

    hw = open(sys.argv[1], "rb").read()
    emu = open(sys.argv[2], "rb").read() if len(sys.argv) > 2 else None

    print("=" * 72)
    print("bf16 Element-Wise Characterizer Analysis")
    print("=" * 72)
    print(f"\nSidon set (32 bf16 elements, indices 0-31):")
    for i in range(0, 32, 8):
        chunk = SIDON[i:i+8]
        idxs = list(range(i, i+8))
        print(f"  [{i:2d}-{i+7:2d}]: {chunk}")

    # Test 1: A=Sidon, B=ones -> A-side mapping
    print("\n" + "-" * 72)
    print("Test 1: A=Sidon, B=ones  (reveals A-side indices)")
    print("-" * 72)
    t1_vals = reconstruct_fp32_lanes(hw, 0, 32)
    t1_hi = read_raw_u16s(hw, 0)
    t1_lo = read_raw_u16s(hw, 32)

    a_indices = []
    for lane in range(16):
        val = t1_vals[lane]
        pairs = find_sidon_pair(val)
        if val == 0.0:
            print(f"  Lane {lane:2d}: {val:10.1f} -> ZERO (inactive lane)")
            a_indices.append(None)
        elif len(pairs) == 2:
            # Sidon set guarantees unique unordered pair; 2 ordered pairs = (i,j) and (j,i)
            i, j = pairs[0]
            print(f"  Lane {lane:2d}: {val:10.1f} = S[{i}]({SIDON[i]}) + S[{j}]({SIDON[j]})")
            a_indices.append(pairs[0])
        elif len(pairs) == 1:
            # i == j case (SIDON[i] + SIDON[i])
            i, j = pairs[0]
            print(f"  Lane {lane:2d}: {val:10.1f} = S[{i}]({SIDON[i]}) + S[{j}]({SIDON[j]})")
            a_indices.append(pairs[0])
        else:
            print(f"  Lane {lane:2d}: {val:10.1f} (hi=0x{t1_hi[lane]:04x} lo=0x{t1_lo[lane]:04x}) -> "
                  f"{'NO MATCH' if not pairs else f'PAIRS: {pairs}'}")
            a_indices.append(None)

    # Test 2: A=ones, B=Sidon -> B-side mapping
    print("\n" + "-" * 72)
    print("Test 2: A=ones, B=Sidon  (reveals B-side indices)")
    print("-" * 72)
    t2_vals = reconstruct_fp32_lanes(hw, 64, 96)
    t2_hi = read_raw_u16s(hw, 64)
    t2_lo = read_raw_u16s(hw, 96)

    b_indices = []
    for lane in range(16):
        val = t2_vals[lane]
        pairs = find_sidon_pair(val)
        if val == 0.0:
            print(f"  Lane {lane:2d}: {val:10.1f} -> ZERO (inactive lane)")
            b_indices.append(None)
        elif len(pairs) == 2:
            i, j = pairs[0]
            print(f"  Lane {lane:2d}: {val:10.1f} = S[{i}]({SIDON[i]}) + S[{j}]({SIDON[j]})")
            b_indices.append(pairs[0])
        elif len(pairs) == 1:
            i, j = pairs[0]
            print(f"  Lane {lane:2d}: {val:10.1f} = S[{i}]({SIDON[i]}) + S[{j}]({SIDON[j]})")
            b_indices.append(pairs[0])
        else:
            print(f"  Lane {lane:2d}: {val:10.1f} (hi=0x{t2_hi[lane]:04x} lo=0x{t2_lo[lane]:04x}) -> "
                  f"{'NO MATCH' if not pairs else f'PAIRS: {pairs}'}")
            b_indices.append(None)

    # Test 3: A=B=Sidon -> cross-check
    print("\n" + "-" * 72)
    print("Test 3: A=B=Sidon  (cross-check)")
    print("-" * 72)
    t3_vals = reconstruct_fp32_lanes(hw, 128, 160)

    for lane in range(16):
        val = t3_vals[lane]
        if val == 0.0:
            print(f"  Lane {lane:2d}: ZERO")
            continue

        a = a_indices[lane]
        b = b_indices[lane]
        if a and b:
            a0, a1 = a
            b0, b1 = b
            expected = float(SIDON[a0] * SIDON[b0] + SIDON[a1] * SIDON[b1])
            match = abs(val - expected) < 0.5
            print(f"  Lane {lane:2d}: {val:12.1f}  expected={expected:12.1f} "
                  f"(S[{a0}]*S[{b0}] + S[{a1}]*S[{b1}] = "
                  f"{SIDON[a0]}*{SIDON[b0]} + {SIDON[a1]}*{SIDON[b1]}) "
                  f"[{'MATCH' if match else 'MISMATCH'}]")
        else:
            print(f"  Lane {lane:2d}: {val:12.1f}  (A or B indices unknown)")

    # Summary: element mapping table
    print("\n" + "=" * 72)
    print("ELEMENT MAPPING SUMMARY")
    print("=" * 72)
    print(f"  Config 0x3d: variant=1, geometry 16x2x1")
    print(f"  16 output lanes, 2 inner products each, 1 column")
    print(f"  x register = 32 bf16 elements (indices 0-31)")
    print()
    print(f"{'Lane':>4} | {'A[k=0]':>8} {'A[k=1]':>8} | {'B[k=0]':>8} {'B[k=1]':>8} | Cross-check")
    print("-" * 72)
    all_match = True
    for lane in range(16):
        a = a_indices[lane]
        b = b_indices[lane]
        if a and b:
            a0, a1 = a
            b0, b1 = b
            expected = float(SIDON[a0] * SIDON[b0] + SIDON[a1] * SIDON[b1])
            actual = t3_vals[lane]
            match = abs(actual - expected) < 0.5
            if not match:
                all_match = False
            print(f"{lane:4d} | {a0:8d} {a1:8d} | {b0:8d} {b1:8d} | "
                  f"{'OK' if match else f'FAIL (got {actual:.0f}, expected {expected:.0f})'}")
        else:
            all_match = False
            print(f"{lane:4d} | {'???':>8} {'???':>8} | {'???':>8} {'???':>8} | "
                  f"UNKNOWN")

    print()
    if all_match:
        print("ALL LANES VERIFIED -- mapping is correct!")
    else:
        print("SOME LANES FAILED -- check raw values above.")

    # Pattern analysis: look for formulas
    print("\n" + "=" * 72)
    print("PATTERN ANALYSIS")
    print("=" * 72)
    if all(a is not None for a in a_indices):
        print("\nA-side pattern:")
        for lane in range(16):
            a0, a1 = a_indices[lane]
            print(f"  Lane {lane:2d}: A[{a0:2d}], A[{a1:2d}]  "
                  f"(delta_0={a0-lane*2 if a0 is not None else '?':>3}, "
                  f"delta_1={a1-lane*2-1 if a1 is not None else '?':>3})")

    if all(b is not None for b in b_indices):
        print("\nB-side pattern:")
        for lane in range(16):
            b0, b1 = b_indices[lane]
            print(f"  Lane {lane:2d}: B[{b0:2d}], B[{b1:2d}]  "
                  f"(delta_0={b0-lane*2 if b0 is not None else '?':>3}, "
                  f"delta_1={b1-lane*2-1 if b1 is not None else '?':>3})")

    # EMU comparison
    if emu:
        print("\n" + "=" * 72)
        print("EMULATOR COMPARISON")
        print("=" * 72)
        for name, hi_off, lo_off in [
            ("Test 1 (A=Sidon, B=ones)", 0, 32),
            ("Test 2 (A=ones, B=Sidon)", 64, 96),
            ("Test 3 (A=B=Sidon)", 128, 160),
        ]:
            hw_vals = reconstruct_fp32_lanes(hw, hi_off, lo_off)
            emu_vals = reconstruct_fp32_lanes(emu, hi_off, lo_off)
            match = all(abs(h - e) < 0.5 for h, e in zip(hw_vals, emu_vals))
            print(f"\n  {name}: {'MATCH' if match else 'MISMATCH'}")
            if not match:
                for i in range(16):
                    if abs(hw_vals[i] - emu_vals[i]) >= 0.5:
                        print(f"    Lane {i}: HW={hw_vals[i]:.1f}  EMU={emu_vals[i]:.1f}")


if __name__ == "__main__":
    main()
