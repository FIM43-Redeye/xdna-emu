//! AIE2 matrix multiply geometry tables.
//!
//! The hardware permute mode tables for the AIE2 vector multiply array.
//! These tables encode the valid matrix tile geometries (rows, inner, cols)
//! and accumulator modes for each element type pair.
//!
//! ## Source
//!
//! Geometry tables and mode assignments derived from hardware behavior
//! as described by the multiplier array architecture (constants.py
//! `__make_mult_modes` and `__make_perm_modes` tables). Config word bit
//! layout for sign/accumulate derived from instruction encoding.
//!
//! ## Hardware Background
//!
//! The AIE2 multiply array has 512 physical multiplier cells operating at
//! 8-bit x 4-bit granularity. Input vectors are permuted and reinterpreted
//! as 2D matrix tiles, and the products are summed through a post-add tree
//! (PSA) into accumulator lanes. The tile dimensions vary by element type
//! pair because wider elements consume more multiplier cells per operation.

/// A single valid matmul geometry entry from the hardware permute mode table.
///
/// Each entry corresponds to one or more pmode indices that an instruction can
/// select. Used to look up tile dimensions for a given element type pair.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // All fields are used in table entries; sparse distinguishes tables.
pub struct GeometryEntry {
    /// Element width of A input in bits.
    pub bits_x: u32,
    /// Element width of B input in bits.
    pub bits_y: u32,
    /// Tile rows (M).
    pub rows: u32,
    /// Inner dimension (K). Reduction dimension summed over.
    pub inner: u32,
    /// Tile columns (N).
    pub cols: u32,
    /// Accumulator combining factor (1 = 32-bit lanes, 2 = 64-bit lanes).
    pub acc_cmb: u32,
    /// Whether this is a bfloat16 mode.
    pub bfloat: bool,
    /// Whether this is a sparse mode.
    pub sparse: bool,
}

/// Complete table of dense matrix multiply geometries supported by AIE2.
///
/// Derived from the hardware permute mode table (constants.py
/// `__make_perm_modes`). Only includes the basic matrix multiply modes
/// (channels=1, no convolution, no complex, not sparse, default strides
/// and ordering).
///
/// Each entry: (bits_x, bits_y, acc_cmb, bfloat, rows, inner, cols)
pub const DENSE_GEOMETRY_TABLE: &[GeometryEntry] = &[
    // int8 x int4 -> int32 (mmode 0: 8x4 acc_cmb=1)
    GeometryEntry {
        bits_x: 8,
        bits_y: 4,
        rows: 4,
        inner: 16,
        cols: 8,
        acc_cmb: 1,
        bfloat: false,
        sparse: false,
    },
    // int8 x int8 -> int32 (mmode 1: 8x8 acc_cmb=1)
    GeometryEntry {
        bits_x: 8,
        bits_y: 8,
        rows: 4,
        inner: 8,
        cols: 8,
        acc_cmb: 1,
        bfloat: false,
        sparse: false,
    },
    // int16 x int8 -> int32 (mmode 2: 16x8 acc_cmb=1)
    GeometryEntry {
        bits_x: 16,
        bits_y: 8,
        rows: 4,
        inner: 4,
        cols: 8,
        acc_cmb: 1,
        bfloat: false,
        sparse: false,
    },
    // int16 x int16 -> int32 (mmode 3: 16x16 acc_cmb=1)
    GeometryEntry {
        bits_x: 16,
        bits_y: 16,
        rows: 4,
        inner: 2,
        cols: 8,
        acc_cmb: 1,
        bfloat: false,
        sparse: false,
    },
    // int16 x int8 -> int64 (mmode 4: 16x8 acc_cmb=2) -- two variants
    GeometryEntry {
        bits_x: 16,
        bits_y: 8,
        rows: 2,
        inner: 8,
        cols: 8,
        acc_cmb: 2,
        bfloat: false,
        sparse: false,
    },
    GeometryEntry {
        bits_x: 16,
        bits_y: 8,
        rows: 4,
        inner: 8,
        cols: 4,
        acc_cmb: 2,
        bfloat: false,
        sparse: false,
    },
    // int16 x int16 -> int64 (mmode 5: 16x16 acc_cmb=2) -- two variants
    GeometryEntry {
        bits_x: 16,
        bits_y: 16,
        rows: 2,
        inner: 4,
        cols: 8,
        acc_cmb: 2,
        bfloat: false,
        sparse: false,
    },
    GeometryEntry {
        bits_x: 16,
        bits_y: 16,
        rows: 4,
        inner: 4,
        cols: 4,
        acc_cmb: 2,
        bfloat: false,
        sparse: false,
    },
    // bf16 x bf16 -> fp32 (mmode 6: 16x16 bfloat acc_cmb=1, acc_num=16)
    GeometryEntry {
        bits_x: 16,
        bits_y: 16,
        rows: 4,
        inner: 8,
        cols: 4,
        acc_cmb: 1,
        bfloat: true,
        sparse: false,
    },
    // int32 x int16 -> int64 (mmode 7: 32x16 acc_cmb=2)
    GeometryEntry {
        bits_x: 32,
        bits_y: 16,
        rows: 4,
        inner: 2,
        cols: 4,
        acc_cmb: 2,
        bfloat: false,
        sparse: false,
    },
];

/// Sparse matrix multiply geometries.
///
/// Sparse modes use the same multiplier array but with a 2x wider X-side
/// permute window and control signals selecting which elements are non-zero.
pub const SPARSE_GEOMETRY_TABLE: &[GeometryEntry] = &[
    // sparse int8 x int4 (inner doubles vs dense)
    GeometryEntry {
        bits_x: 8,
        bits_y: 4,
        rows: 4,
        inner: 32,
        cols: 8,
        acc_cmb: 1,
        bfloat: false,
        sparse: true,
    },
    // sparse int8 x int8
    GeometryEntry {
        bits_x: 8,
        bits_y: 8,
        rows: 4,
        inner: 16,
        cols: 8,
        acc_cmb: 1,
        bfloat: false,
        sparse: true,
    },
    // sparse int16 x int8
    GeometryEntry {
        bits_x: 16,
        bits_y: 8,
        rows: 2,
        inner: 16,
        cols: 8,
        acc_cmb: 2,
        bfloat: false,
        sparse: true,
    },
    // sparse int16 x int16
    GeometryEntry {
        bits_x: 16,
        bits_y: 16,
        rows: 2,
        inner: 8,
        cols: 8,
        acc_cmb: 2,
        bfloat: false,
        sparse: true,
    },
    // sparse bf16 x bf16
    GeometryEntry {
        bits_x: 16,
        bits_y: 16,
        rows: 4,
        inner: 16,
        cols: 4,
        acc_cmb: 1,
        bfloat: true,
        sparse: true,
    },
];

/// Hardware config word (amode, bmode, variant) to geometry lookup table.
///
/// Derived from the `aiev2_compute_control()` function in aiev2_vmult.h
/// and the geometry function names (e.g., `mac_4x8_8x8` = rows=4, inner=8, cols=8).
///
/// Each entry: (amode, bmode, variant) -> GeometryEntry
pub const CONFIG_GEOMETRY_TABLE: &[(u32, u32, u32, GeometryEntry)] = &[
    // amode=0: acc_cmb=1 (32-bit accumulator lanes)
    // i8 x i4 -> acc32
    (
        0,
        0,
        0,
        GeometryEntry {
            bits_x: 8,
            bits_y: 4,
            rows: 4,
            inner: 16,
            cols: 8,
            acc_cmb: 1,
            bfloat: false,
            sparse: false,
        },
    ),
    // i8 x i4 sparse -> acc32
    (
        0,
        0,
        1,
        GeometryEntry {
            bits_x: 8,
            bits_y: 4,
            rows: 4,
            inner: 32,
            cols: 8,
            acc_cmb: 1,
            bfloat: false,
            sparse: true,
        },
    ),
    // i8 x i8 -> acc32
    (
        0,
        1,
        0,
        GeometryEntry {
            bits_x: 8,
            bits_y: 8,
            rows: 4,
            inner: 8,
            cols: 8,
            acc_cmb: 1,
            bfloat: false,
            sparse: false,
        },
    ),
    // i8 x i8 sparse -> acc32
    (
        0,
        1,
        5,
        GeometryEntry {
            bits_x: 8,
            bits_y: 8,
            rows: 4,
            inner: 16,
            cols: 8,
            acc_cmb: 1,
            bfloat: false,
            sparse: true,
        },
    ),
    // i16 x i8 -> acc32
    (
        0,
        2,
        0,
        GeometryEntry {
            bits_x: 16,
            bits_y: 8,
            rows: 4,
            inner: 4,
            cols: 8,
            acc_cmb: 1,
            bfloat: false,
            sparse: false,
        },
    ),
    // i16 x i16 -> acc32
    (
        0,
        3,
        0,
        GeometryEntry {
            bits_x: 16,
            bits_y: 16,
            rows: 4,
            inner: 2,
            cols: 8,
            acc_cmb: 1,
            bfloat: false,
            sparse: false,
        },
    ),
    // amode=1: acc_cmb=2 (64-bit accumulator lanes)
    // i32 x i16 -> acc64
    (
        1,
        0,
        0,
        GeometryEntry {
            bits_x: 32,
            bits_y: 16,
            rows: 4,
            inner: 2,
            cols: 4,
            acc_cmb: 2,
            bfloat: false,
            sparse: false,
        },
    ),
    // i16 x i8 -> acc64 (variant 0: 2x8x8)
    (
        1,
        2,
        0,
        GeometryEntry {
            bits_x: 16,
            bits_y: 8,
            rows: 2,
            inner: 8,
            cols: 8,
            acc_cmb: 2,
            bfloat: false,
            sparse: false,
        },
    ),
    // i16 x i8 -> acc64 (variant 1: 4x8x4)
    (
        1,
        2,
        1,
        GeometryEntry {
            bits_x: 16,
            bits_y: 8,
            rows: 4,
            inner: 8,
            cols: 4,
            acc_cmb: 2,
            bfloat: false,
            sparse: false,
        },
    ),
    // i16 x i8 sparse -> acc64
    (
        1,
        2,
        2,
        GeometryEntry {
            bits_x: 16,
            bits_y: 8,
            rows: 2,
            inner: 16,
            cols: 8,
            acc_cmb: 2,
            bfloat: false,
            sparse: true,
        },
    ),
    // i16 x i16 -> acc64 (variant 0: 2x4x8)
    (
        1,
        3,
        0,
        GeometryEntry {
            bits_x: 16,
            bits_y: 16,
            rows: 2,
            inner: 4,
            cols: 8,
            acc_cmb: 2,
            bfloat: false,
            sparse: false,
        },
    ),
    // i16 x i16 -> acc64 (variant 1: 4x4x4)
    (
        1,
        3,
        1,
        GeometryEntry {
            bits_x: 16,
            bits_y: 16,
            rows: 4,
            inner: 4,
            cols: 4,
            acc_cmb: 2,
            bfloat: false,
            sparse: false,
        },
    ),
    // i16 x i16 sparse -> acc64
    (
        1,
        3,
        5,
        GeometryEntry {
            bits_x: 16,
            bits_y: 16,
            rows: 2,
            inner: 8,
            cols: 8,
            acc_cmb: 2,
            bfloat: false,
            sparse: true,
        },
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Drift-detection test: locks the dense geometry table entry count and
    /// selected key values so any accidental modification is caught immediately.
    ///
    /// If the hardware table changes, update this test alongside the data.
    #[test]
    fn dense_geometry_table_entry_count_and_spot_check() {
        assert_eq!(DENSE_GEOMETRY_TABLE.len(), 10, "dense table entry count changed");

        // Entry 0: i8 x i4 -> acc32 (4x16x8)
        let e0 = &DENSE_GEOMETRY_TABLE[0];
        assert_eq!(e0.bits_x, 8);
        assert_eq!(e0.bits_y, 4);
        assert_eq!(e0.rows, 4);
        assert_eq!(e0.inner, 16);
        assert_eq!(e0.cols, 8);
        assert_eq!(e0.acc_cmb, 1);
        assert!(!e0.bfloat);
        assert!(!e0.sparse);

        // Entry 8: bf16 x bf16 -> fp32 (4x8x4)
        let e8 = &DENSE_GEOMETRY_TABLE[8];
        assert_eq!(e8.bits_x, 16);
        assert_eq!(e8.bits_y, 16);
        assert_eq!(e8.rows, 4);
        assert_eq!(e8.inner, 8);
        assert_eq!(e8.cols, 4);
        assert_eq!(e8.acc_cmb, 1);
        assert!(e8.bfloat);
        assert!(!e8.sparse);

        // Entry 9: i32 x i16 -> acc64 (4x2x4)
        let e9 = &DENSE_GEOMETRY_TABLE[9];
        assert_eq!(e9.bits_x, 32);
        assert_eq!(e9.bits_y, 16);
        assert_eq!(e9.rows, 4);
        assert_eq!(e9.inner, 2);
        assert_eq!(e9.cols, 4);
        assert_eq!(e9.acc_cmb, 2);
    }

    /// Drift-detection test: locks the sparse geometry table entry count and
    /// key values.
    #[test]
    fn sparse_geometry_table_entry_count_and_spot_check() {
        assert_eq!(SPARSE_GEOMETRY_TABLE.len(), 5, "sparse table entry count changed");

        // Entry 0: sparse i8 x i4 (inner=32, doubled vs dense inner=16)
        let e0 = &SPARSE_GEOMETRY_TABLE[0];
        assert_eq!(e0.bits_x, 8);
        assert_eq!(e0.bits_y, 4);
        assert_eq!(e0.inner, 32);
        assert!(e0.sparse);

        // Entry 4: sparse bf16 x bf16 (4x16x4)
        let e4 = &SPARSE_GEOMETRY_TABLE[4];
        assert_eq!(e4.bits_x, 16);
        assert_eq!(e4.bits_y, 16);
        assert_eq!(e4.inner, 16);
        assert!(e4.sparse);
        assert!(e4.bfloat);
    }

    /// Drift-detection test: config geometry table entry count and spot-check.
    #[test]
    fn config_geometry_table_entry_count_and_spot_check() {
        assert_eq!(CONFIG_GEOMETRY_TABLE.len(), 13, "config table entry count changed");

        // First entry: amode=0, bmode=0, variant=0 -> i8 x i4 dense acc32
        let (a, b, v, ref e) = CONFIG_GEOMETRY_TABLE[0];
        assert_eq!(a, 0);
        assert_eq!(b, 0);
        assert_eq!(v, 0);
        assert_eq!(e.bits_x, 8);
        assert_eq!(e.bits_y, 4);
        assert!(!e.sparse);

        // Entry for i16 x i8 sparse acc64: amode=1, bmode=2, variant=2
        let sparse_entry = CONFIG_GEOMETRY_TABLE.iter().find(|(a, b, v, _)| *a == 1 && *b == 2 && *v == 2);
        assert!(sparse_entry.is_some(), "sparse i16xi8 acc64 entry must exist");
        let (_, _, _, ref se) = sparse_entry.unwrap();
        assert!(se.sparse);
        assert_eq!(se.acc_cmb, 2);
    }
}
