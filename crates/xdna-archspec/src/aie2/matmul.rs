//! AIE2 matrix multiply geometry table.
//!
//! The valid matrix tile geometries (rows, inner, cols) and accumulator modes
//! of the AIE2 vector multiply array, keyed by the (amode, bmode, variant)
//! fields of the MAC config word.
//!
//! ## Source
//!
//! Generated at build time from llvm-aie's `aiev2_vmult.h`
//! (`clang/lib/Headers/aiev2/`): each `<op>_<rows>x<inner>_<inner>x<cols>`
//! intrinsic names a tile shape, builds its config word with
//! `aiev2_compute_control(sgn_x, sgn_y, amode, bmode, variant, ...)`, and
//! types its operands and result with the element and accumulator widths.
//! See `mac_geometry_extract` for the parser and `build.rs` for the wiring.
//!
//! ## Hardware Background
//!
//! The AIE2 multiply array has 512 physical multiplier cells operating at
//! 8-bit x 4-bit granularity. Input vectors are permuted and reinterpreted
//! as 2D matrix tiles, and the products are summed through a post-add tree
//! (PSA) into accumulator lanes. The tile dimensions vary by element type
//! pair because wider elements consume more multiplier cells per operation.

/// One valid matmul geometry: the tile shape and element types that a
/// (amode, bmode, variant) config word selects.
#[derive(Debug, Clone, Copy)]
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

include!(concat!(env!("OUT_DIR"), "/gen_mac_geometry.rs"));

/// Dense (non-sparse) geometries, in config-key order.
pub fn dense_geometries() -> impl Iterator<Item = &'static GeometryEntry> {
    CONFIG_GEOMETRY_TABLE
        .iter()
        .map(|(_, _, _, entry)| entry)
        .filter(|entry| !entry.sparse)
}

/// Sparse geometries, in config-key order.
///
/// Sparse modes read a compressed B operand with control bits selecting the
/// non-zero elements, so the inner dimension doubles relative to the dense
/// mode with the same element types.
pub fn sparse_geometries() -> impl Iterator<Item = &'static GeometryEntry> {
    CONFIG_GEOMETRY_TABLE
        .iter()
        .map(|(_, _, _, entry)| entry)
        .filter(|entry| entry.sparse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(amode: u32, bmode: u32, variant: u32) -> &'static GeometryEntry {
        CONFIG_GEOMETRY_TABLE
            .iter()
            .find(|(a, b, v, _)| (*a, *b, *v) == (amode, bmode, variant))
            .map(|(_, _, _, e)| e)
            .unwrap_or_else(|| panic!("no geometry for ({amode}, {bmode}, {variant})"))
    }

    #[test]
    fn config_keys_are_unique_and_sorted() {
        let keys: Vec<_> = CONFIG_GEOMETRY_TABLE.iter().map(|(a, b, v, _)| (*a, *b, *v)).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn dense_and_sparse_partition_the_table() {
        assert_eq!(dense_geometries().count() + sparse_geometries().count(), CONFIG_GEOMETRY_TABLE.len());
        assert!(dense_geometries().all(|e| !e.sparse));
        assert!(sparse_geometries().all(|e| e.sparse));
        assert!(dense_geometries().count() > 0 && sparse_geometries().count() > 0);
    }

    /// Spot checks of shapes named by intrinsics in aiev2_vmult.h
    /// (`mac_4x16_16x8`, `mac_4x8_8x4` with bfloat16, ...).
    #[test]
    fn known_intrinsic_shapes_are_extracted() {
        let e = entry(0, 0, 0);
        assert_eq!((e.bits_x, e.bits_y, e.rows, e.inner, e.cols, e.acc_cmb), (8, 4, 4, 16, 8, 1));
        assert!(!e.sparse && !e.bfloat);

        let e = entry(1, 2, 1);
        assert_eq!((e.bits_x, e.bits_y, e.rows, e.inner, e.cols, e.acc_cmb), (16, 8, 4, 8, 4, 2));

        let e = entry(2, 3, 0);
        assert_eq!((e.bits_x, e.bits_y, e.rows, e.inner, e.cols, e.acc_cmb), (16, 16, 4, 8, 4, 1));
        assert!(e.bfloat && !e.sparse);
    }

    #[test]
    fn sparse_modes_are_flagged() {
        let e = entry(0, 0, 1);
        assert_eq!((e.bits_x, e.bits_y, e.rows, e.inner, e.cols), (8, 4, 4, 32, 8));
        assert!(e.sparse);

        let e = entry(2, 3, 2);
        assert_eq!((e.rows, e.inner, e.cols), (4, 16, 4));
        assert!(e.sparse && e.bfloat);
    }

    #[test]
    fn every_sparse_inner_exceeds_its_dense_counterpart() {
        for sparse in sparse_geometries() {
            let dense_inner = dense_geometries()
                .filter(|d| {
                    (d.bits_x, d.bits_y, d.bfloat, d.acc_cmb, d.cols)
                        == (sparse.bits_x, sparse.bits_y, sparse.bfloat, sparse.acc_cmb, sparse.cols)
                })
                .map(|d| d.inner)
                .min();
            if let Some(inner) = dense_inner {
                assert!(sparse.inner > inner, "{sparse:?} vs dense inner {inner}");
            }
        }
    }
}
