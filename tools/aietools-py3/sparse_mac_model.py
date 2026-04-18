#!/usr/bin/env python3
"""
End-to-end sparse MAC model using the parsed crossbar routing table.
Validates against actual hardware output from ISA test batch 83.

Key insight: x0 is used for BOTH xs1 and qxs2 in the test, so the same
64 bytes serve as both the i16 dense operand (Y side of hardware) and
the i8 sparse operand (X side, routed by crossbar).

With acc_cmb=2, each output is 64-bit, stored in two adjacent i32 accumulator
lanes. SRS.s16.s32 with shift=0 extracts the lower 16 bits of each i32 lane,
producing the (lo_half, hi_half) pattern seen in HW output.
"""

import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fill_prng(n, seed=42):
    """Generate n bytes of PRNG data matching the C++ LCG."""
    buf = bytearray(n)
    state = seed
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        buf[i] = (state >> 16) & 0xFF
    return bytes(buf)


def load_batch_83_sparse_data(seed=42):
    """Reconstruct input data for the first sparse test in batch 83.

    From batch_083.s: first sparse test at config=0x353.
    x0 is loaded last with qxs2 data, overwriting xs1 data.
    """
    buf = fill_prng(4096, seed)

    # Accumulator cm0: loaded as i16, then UPS to i32
    acc_lo_bytes = buf[2112:2112+32]  # 16 i16 -> bml0
    acc_hi_bytes = buf[2144:2144+32]  # 16 i16 -> bmh0

    # x0 final content = qxs2 data (overwrites xs1)
    x0_data = buf[2240:2240+32] + buf[2272:2272+32]  # 64 bytes

    # q0 mask
    q0_bytes = buf[2304:2304+16]
    mask128 = int.from_bytes(q0_bytes, 'little')

    # UPS: sign-extend i16 to i32 with shift=0
    acc_lo = [struct.unpack_from('<h', acc_lo_bytes, i*2)[0] for i in range(16)]
    acc_hi = [struct.unpack_from('<h', acc_hi_bytes, i*2)[0] for i in range(16)]

    return acc_lo, acc_hi, x0_data, mask128


def compute_sparse_mac(x0_data, mask128, acc_lo, acc_hi, config):
    """Compute sparse MAC using crossbar routing + Y permutation.

    For each output lane, sum element products: A_i16 * B_i8
    where A comes from Y-side perm (dense i16) and B from X-side crossbar (sparse i8).

    With acc_cmb=2: lanes 0,1 combine into one 64-bit output,
    lanes 2,3 into another, etc. Total 16 effective outputs.

    Returns: (result_lo, result_hi) each with 16 i32 values,
    representing the SRS output from bml0 and bmh0.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location('pc',
        os.path.join(os.path.dirname(__file__), 'parse_crossbar.py'))
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    import constants as C_module
    C = C_module.C

    # Parse crossbar and evaluate routing
    routing = pc.consolidate_routing(pc.parse_function())
    m_val = pc.build_m_word(23, mask128)
    m_bits = pc.m_to_bit_set(m_val)
    x_routing = pc.evaluate_routing(routing, m_bits)

    # Extend x0 to 128 bytes for crossbar
    x_full = bytearray(128)
    x_full[:64] = x0_data

    # Y permutation table and multiplier indices
    yt = C.permute_y[23]
    pm = C.perm_modes[23]
    mm = C.mult_modes[pm.mmode]
    mpy_idx = C.mpy_indices[(pm.mmode, pm.complex_x, pm.complex_y)]
    mults_per_lane = C.mult_num // C.acc_num * mm.acc_cmb  # 32

    # Config parsing
    sgn_x = (config >> 7) & 1  # 0 = unsigned for bits_x (i16 side)
    sgn_y = (config >> 8) & 1  # 1 = signed for bits_y (i8 side)
    zero_acc1 = (config >> 9) & 1

    print(f"Config 0x{config:x}: sgn_x={sgn_x}, sgn_y={sgn_y}, zero_acc1={zero_acc1}")
    print(f"Mask: 0x{mask128:032x}")

    # Identify element products per lane.
    # mpy_idx[j][0] = inner dimension value.
    # Group mults by inner value to find element products.
    # For acc_cmb=2, each effective output spans 2 accumulator lanes.

    # Initialize accumulator (32 lanes of i32)
    if zero_acc1:
        acc = [0] * 32
    else:
        acc = list(acc_lo) + list(acc_hi)

    # For each accumulator lane pair (acc_cmb=2):
    for row in range(pm.rows):
        for col in range(pm.cols):
            lane = pm.idx_o(row, col, 0)  # 0..15
            # This output uses acc lanes: lane*2 and lane*2+1 (Acc32 packing)
            # Actually, with acc_cmb=2 and 32 mults per combined pair,
            # the mult_base = lane * mults_per_lane
            mult_base = lane * mults_per_lane

            # Find unique inner values and their representative mults
            seen_inner = {}
            for j in range(mults_per_lane):
                inner_val = mpy_idx[j][0]
                if inner_val not in seen_inner:
                    seen_inner[inner_val] = j

            # Compute element products
            product_sum = 0
            for inner_val, first_j in sorted(seen_inner.items()):
                mult_idx = mult_base + first_j

                # B value (i8, from crossbar X routing)
                b_byte_idx = x_routing.get(mult_idx)
                if b_byte_idx is None or isinstance(b_byte_idx, list):
                    continue
                b_byte = x_full[b_byte_idx]
                if sgn_y:  # bits_y = i8, sgn_y controls i8 signedness
                    b_val = b_byte if b_byte < 128 else b_byte - 256
                else:
                    b_val = b_byte

                # A value (i16, from Y permutation)
                y_nib = yt[mult_idx]
                if y_nib < 0:
                    continue
                # Y nibble index to byte index: byte = nib // 2
                # i16 element starts at this byte (lo byte)
                y_byte_idx = y_nib // 2
                # Read full i16 from x0_data (which is BOTH A and B source)
                if y_byte_idx + 1 < len(x0_data):
                    if sgn_x:  # bits_x = i16, sgn_x controls i16 signedness
                        a_val = struct.unpack_from('<h', x0_data, y_byte_idx)[0]
                    else:
                        a_val = struct.unpack_from('<H', x0_data, y_byte_idx)[0]
                else:
                    continue

                product = a_val * b_val
                product_sum += product

            # Store as 64-bit result split into two i32 lanes
            # acc[lane*2] = lo 32 bits, acc[lane*2+1] = hi 32 bits
            result_64 = product_sum
            if zero_acc1:
                pass  # result_64 = product_sum (no accumulator input)
            else:
                # Add existing accumulator value (64-bit from two i32 lanes)
                acc_64 = (acc[lane*2] & 0xFFFFFFFF) | ((acc[lane*2+1] & 0xFFFFFFFF) << 32)
                # Sign extend if needed
                if acc_64 >= (1 << 63):
                    acc_64 -= (1 << 64)
                result_64 += acc_64

            # Pack back into two i32 lanes
            r64 = result_64 & 0xFFFFFFFFFFFFFFFF  # unsigned 64-bit
            acc[lane*2] = r64 & 0xFFFFFFFF
            if acc[lane*2] >= 0x80000000:
                acc[lane*2] -= 0x100000000
            acc[lane*2+1] = (r64 >> 32) & 0xFFFFFFFF
            if acc[lane*2+1] >= 0x80000000:
                acc[lane*2+1] -= 0x100000000

    return acc[:16], acc[16:]


def srs_s16_s32(acc_values):
    """SRS.s16.s32 with shift=0: truncate i32 to lower 16 bits."""
    result = []
    for val in acc_values:
        t = val & 0xFFFF
        if t >= 0x8000:
            t -= 0x10000
        result.append(t)
    return result


def main():
    acc_lo, acc_hi, x0_data, mask128 = load_batch_83_sparse_data()

    print(f"Acc lo (first 4): {acc_lo[:4]}")
    print(f"Acc hi (first 4): {acc_hi[:4]}")

    result_lo_32, result_hi_32 = compute_sparse_mac(
        x0_data, mask128, acc_lo, acc_hi, 0x353
    )

    # SRS: extract i16 from i32
    result_lo = srs_s16_s32(result_lo_32)
    result_hi = srs_s16_s32(result_hi_32)

    # Pack to bytes
    result_bytes = b''
    for v in result_lo:
        result_bytes += struct.pack('<h', v)
    for v in result_hi:
        result_bytes += struct.pack('<h', v)

    # Compare with hardware output
    results_dir = '/home/triple/npu-work/xdna-emu/build/isa-test-results/20260330'
    hw_data = open(f'{results_dir}/batch_83_hw.bin', 'rb').read()
    hw_out = hw_data[704:704+64]

    print(f"\nComparison (i16 values):")
    print(f"{'Idx':>3} {'HW':>8} {'Model':>8} {'Match':>6}")
    print("-" * 30)
    match = 0
    for i in range(32):
        hw_val = struct.unpack_from('<h', hw_out, i * 2)[0]
        model_val = struct.unpack_from('<h', result_bytes, i * 2)[0]
        m = "OK" if hw_val == model_val else ""
        if hw_val == model_val:
            match += 1
        print(f"{i:3d} {hw_val:8d} {model_val:8d} {m:>6}")

    print(f"\nMatch: {match}/32")


if __name__ == "__main__":
    main()
