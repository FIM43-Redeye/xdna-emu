#!/usr/bin/env python3
"""Analyze aietools Y-side permutation tables for all sparse modes.

Determines the correct B data layout after decompression for each sparse
MAC geometry. This is the ground truth for xdna-emu's sparse model.
"""
import sys
sys.path.insert(0, '/home/triple/npu-work/xdna-emu/tools/aietools-py3')
import constants as C_module

C = C_module.C

def analyze_sparse_y(pmode_num):
    pm = C.perm_modes[pmode_num]
    mm = C.mult_modes[pm.mmode]
    yt = C.permute_y[pmode_num]
    mpy_idx = C.mpy_indices[(pm.mmode, pm.complex_x, pm.complex_y)]
    mults_per_lane = C.mult_num // C.acc_num * mm.acc_cmb

    print("=" * 70)
    print(f"pmode {pmode_num}: {mm.bits_x}x{mm.bits_y} sparse "
          f"rows={pm.rows} inner={pm.inner} cols={pm.cols} acc_cmb={mm.acc_cmb}")
    print(f"mults_per_lane={mults_per_lane}, "
          f"mult_gran_y={C.mult_gran_y}")
    vals = [v for v in yt if v >= 0]
    print(f"Y range: {min(vals)} to {max(vals)}, active: {len(vals)}/{len(yt)}")

    nibs_per_elem = mm.bits_y // C.mult_gran_y  # 2 for 8-bit, 4 for 16-bit
    inner_half = pm.inner // 2  # dense inner dimension

    # B data after decompression:
    # For sparse, sz_y /= 2 -> 256 bits input, which is 32 bytes for i8, 16 elements for i16
    # But the Y range goes up to 127 (64 bytes in nibble units for i8)
    # This means the perm table addresses into the DECOMPRESSED space
    decompressed_bytes = max(vals) // nibs_per_elem + 1
    print(f"Decompressed B space: {decompressed_bytes} elements of {mm.bits_y} bits")
    print(f"inner_half (dense inner) = {inner_half}")

    # Show per-lane B element mapping for first 2 rows, first 3 cols
    print(f"\nPer-lane B element mapping:")
    for row in range(min(pm.rows, 2)):
        for col in range(min(pm.cols, 4)):
            lane = pm.idx_o(row, col, 0)
            start = lane * mults_per_lane
            elems = []
            for j in range(0, mults_per_lane, nibs_per_elem):
                nib_val = yt[start + j]
                if nib_val >= 0:
                    elem_idx = nib_val // nibs_per_elem
                else:
                    elem_idx = -1
                inn_dense = mpy_idx[j][0]
                elems.append((inn_dense, elem_idx))
            print(f"  ({row},{col}) lane {lane:2d}: {elems}")

    # Test hypothesis 1: ROW-MAJOR B[k][c] = k * cols + c
    print(f"\nHypothesis 1: B[k][c] = k * {pm.cols} + c (row-major)")
    h1_ok = True
    h1_mismatches = 0
    for row in range(pm.rows):
        for col in range(pm.cols):
            lane = pm.idx_o(row, col, 0)
            start = lane * mults_per_lane
            for j in range(0, mults_per_lane, nibs_per_elem):
                nib_val = yt[start + j]
                if nib_val < 0:
                    continue
                inn_dense = mpy_idx[j][0]
                actual = nib_val // nibs_per_elem
                expected = inn_dense * pm.cols + col
                if actual != expected:
                    h1_ok = False
                    h1_mismatches += 1
                    if h1_mismatches <= 3:
                        print(f"  MISMATCH ({row},{col}) k={inn_dense}: "
                              f"expected {expected}, got {actual}")
    if h1_ok:
        print(f"  CONFIRMED: B[k][c] = k * {pm.cols} + c")
    else:
        print(f"  FAILED: {h1_mismatches} mismatches")

    # Test hypothesis 2: COLUMN-MAJOR B[k][c] = c * inner_half + k
    print(f"\nHypothesis 2: B[k][c] = c * {inner_half} + k (column-major)")
    h2_ok = True
    h2_mismatches = 0
    for row in range(pm.rows):
        for col in range(pm.cols):
            lane = pm.idx_o(row, col, 0)
            start = lane * mults_per_lane
            for j in range(0, mults_per_lane, nibs_per_elem):
                nib_val = yt[start + j]
                if nib_val < 0:
                    continue
                inn_dense = mpy_idx[j][0]
                actual = nib_val // nibs_per_elem
                expected = col * inner_half + inn_dense
                if actual != expected:
                    h2_ok = False
                    h2_mismatches += 1
                    if h2_mismatches <= 3:
                        print(f"  MISMATCH ({row},{col}) k={inn_dense}: "
                              f"expected {expected}, got {actual}")
    if h2_ok:
        print(f"  CONFIRMED: B[k][c] = c * {inner_half} + k")
    else:
        print(f"  FAILED: {h2_mismatches} mismatches")

    # Test hypothesis 3: same as dense non-sparse indexing
    # Dense non-sparse pmode for same mmode:
    dense_pmode = None
    for dp, dpm in enumerate(C.perm_modes):
        if dpm.mmode == pm.mmode and not dpm.sparse and dpm.channels == 1:
            dense_pmode = dp
            break
    if dense_pmode is not None:
        dpm = C.perm_modes[dense_pmode]
        print(f"\nCorresponding dense mode: pmode {dense_pmode} "
              f"(rows={dpm.rows} inner={dpm.inner} cols={dpm.cols})")
        print(f"Dense idx_y formula (inn, col):")
        for inn in range(min(dpm.inner, 4)):
            vals = [dpm.idx_y(inn, col, 0) for col in range(min(dpm.cols, 4))]
            print(f"  inn={inn}: {vals}")

    print()

for sp in [22, 23, 24, 25]:
    analyze_sparse_y(sp)
