#!/usr/bin/env python3
"""Generate input data for the bf16 sub-precision characterizer.

Creates 3 test vectors (128 bytes each = 384 bytes total):
  Test 1: Known-failing input from ISA tests (produces model=0x007F, HW=0x0001)
  Test 2: Inputs designed to hit rounding boundaries (near half-ULP)
  Test 3: Inputs with maximum cancellation (mixed signs, similar magnitudes)

Each test vector = 64 bytes (x0 register) + 64 bytes (x2 register).
"""

import struct
import sys

def bf16(val):
    """Convert float to bf16 (truncate, no rounding)."""
    bits = struct.unpack('I', struct.pack('f', val))[0]
    return (bits >> 16) & 0xFFFF

def bf16_from_parts(sign, exp, man):
    """Build bf16 from components."""
    return ((int(sign) & 1) << 15) | ((exp & 0xFF) << 7) | (man & 0x7F)

def pack_bf16_register(vals):
    """Pack up to 32 bf16 values into 64 bytes (512 bits)."""
    buf = bytearray(64)
    for i, v in enumerate(vals[:32]):
        struct.pack_into('<H', buf, i * 2, v)
    return bytes(buf)

# =========================================================================
# Test 1: Random inputs from ISA test that produced model/HW discrepancy
# These are the PRNG-generated values for a known-failing batch.
# =========================================================================
# Use the input pattern that gave model=0xD298007F for dense 8-product MAC
test1_x0_vals = [0xbd9f, 0xe837, 0x6a80, 0xd0d3, 0x511c, 0x0848, 0x922a, 0xc850,
                 0x3c3e, 0x21b1, 0xa798, 0x8290, 0x08af, 0x1092, 0x8b3c, 0x0331,
                 # Fill remaining 16 slots with zeros (not used in dense 4x8x4 row 0)
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# For vmul.f bml0, x0, x2, r0: A comes from x0, B from x2
# Dense 4x8x4: lane (r=0,c=0) uses A[0..7] and B[0,4,8,12,16,20,24,28]
# To get exact ISA test behavior with s1=s2=x0, set x2=x0
test1_x2_vals = test1_x0_vals[:]

# =========================================================================
# Test 2: Engineered rounding boundary inputs
# Products that sum to a value with the guard bit exactly at the halfway point.
# Use bf16 values where a*b products have specific mantissa patterns.
# =========================================================================
# Strategy: use pairs where the product mantissa sum falls on an RNE boundary.
# Two large same-sign products + several small products to push past halfway.
#
# bf16 1.0 = 0x3F80 (exp=127, man=0)
# bf16 1.5 = 0x3FC0 (exp=127, man=64)
# bf16 1.25 = 0x3FA0 (exp=127, man=32)
# bf16 1.125 = 0x3F90 (exp=127, man=16)
# bf16 1.0625 = 0x3F88 (exp=127, man=8)
# bf16 1.03125 = 0x3F84 (exp=127, man=4)
# bf16 1.015625 = 0x3F82 (exp=127, man=2)
# bf16 1.0078125 = 0x3F81 (exp=127, man=1)
#
# Product of 1.0078125 * 1.0078125 = 1.015625061... which has sub-bf16 precision bits.
test2_x0_vals = [
    bf16(1.0078125),  # 0x3F81
    bf16(1.5),        # 0x3FC0
    bf16(1.25),       # 0x3FA0
    bf16(2.0),        # 0x4000
    bf16(0.5),        # 0x3F00
    bf16(3.0),        # 0x4040
    bf16(0.25),       # 0x3E80
    bf16(1.0078125),  # 0x3F81
    # Remaining for other rows
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
]
test2_x2_vals = [
    bf16(1.0078125),  # Same as A for squaring
    bf16(1.5),
    bf16(1.25),
    bf16(2.0),
    bf16(0.5),
    bf16(3.0),
    bf16(0.25),
    bf16(1.0078125),
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
]

# =========================================================================
# Test 3: Mixed-sign cancellation -- large positive + large negative products
# This exercises the signed accumulation path and can produce small results
# with many significant sub-bf16 bits.
# =========================================================================
test3_x0_vals = [
    bf16(100.0),    # 0x42C8
    bf16(-99.5),    # 0xC2C7 (close to -100, partial cancellation)
    bf16(50.0),     # 0x4248
    bf16(-50.25),   # 0xC249
    bf16(10.0),     # 0x4120
    bf16(-9.875),   # 0xC11E
    bf16(1.0),      # 0x3F80
    bf16(-0.9375),  # 0xBF70
    # Remaining
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
]
# B = all ones for simple products (A * 1.0 = A)
test3_x2_vals = [
    bf16(1.0), bf16(1.0), bf16(1.0), bf16(1.0),
    bf16(1.0), bf16(1.0), bf16(1.0), bf16(1.0),
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
]

# =========================================================================
# Build the output
# =========================================================================
output = bytearray()
for test_x0, test_x2 in [
    (test1_x0_vals, test1_x2_vals),
    (test2_x0_vals, test2_x2_vals),
    (test3_x0_vals, test3_x2_vals),
]:
    output += pack_bf16_register(test_x0)
    output += pack_bf16_register(test_x2)

out_path = sys.argv[1] if len(sys.argv) > 1 else 'input.bin'
with open(out_path, 'wb') as f:
    f.write(output)

print(f"Generated {len(output)} bytes ({len(output)//128} test vectors) -> {out_path}")

# Print summary
for i, (x0, x2) in enumerate([
    (test1_x0_vals, test1_x2_vals),
    (test2_x0_vals, test2_x2_vals),
    (test3_x0_vals, test3_x2_vals),
]):
    print(f"\nTest {i+1}:")
    print(f"  x0[0:7] = {[f'0x{v:04X}' for v in x0[:8]]}")
    print(f"  x2[0:7] = {[f'0x{v:04X}' for v in x2[:8]]}")
    # Compute expected dense lane (0,0): A[0..7] dot B[0,4,8,...,28]
    # But for B, dense indexing: B[k*4+c] for k=0..7, c=0
    # With x2: B[0], B[4], B[8], B[12], B[16], B[20], B[24], B[28]
    b_indices = [k * 4 for k in range(8)]
    a_vals = x0[:8]
    b_vals = [x2[j] if j < len(x2) else 0 for j in b_indices]
    print(f"  Lane(0,0) A: {[f'0x{v:04X}' for v in a_vals]}")
    print(f"  Lane(0,0) B: {[f'0x{v:04X}' for v in b_vals]}")
