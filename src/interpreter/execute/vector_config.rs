//! AIE2 matrix multiply configuration word parser.
//!
//! The AIE2 vector unit's matrix multiply instructions encode their tile
//! geometry, accumulator mode, and sign/subtract configuration in a "config
//! word" (called `pmode` in the aietools Python model). This module parses
//! that config word into a structured representation.
//!
//! The config word selects a permute mode (pmode), which in turn selects:
//! - A mult mode (mmode) controlling element widths and accumulator combining
//! - Tile geometry: rows (M), inner (K), cols (N)
//! - Whether accumulation adds to or replaces the existing value
//! - Sign extension for X and Y input buffers
//!
//! ## Hardware Background
//!
//! The AIE2 multiply array has 512 physical multiplier cells operating at
//! 8-bit x 4-bit granularity. Input vectors are permuted and reinterpreted
//! as 2D matrix tiles, and the products are summed through a post-add tree
//! (PSA) into accumulator lanes. The tile dimensions vary by element type
//! pair because wider elements consume more multiplier cells per operation.
//!
//! ## Source
//!
//! Geometry tables and mode assignments derived from hardware behavior
//! as described by the multiplier array architecture (constants.py
//! `__make_mult_modes` and `__make_perm_modes` tables). Config word bit
//! layout for sign/accumulate derived from instruction encoding.

use crate::interpreter::bundle::ElementType;
use xdna_archspec::aie2::matmul::{
    GeometryEntry, DENSE_GEOMETRY_TABLE, SPARSE_GEOMETRY_TABLE, CONFIG_GEOMETRY_TABLE,
};

/// Accumulator width mode, determined by `acc_cmb` in the mult mode.
///
/// acc_cmb=1 means each accumulator lane is 32 bits (two per u64 lane).
/// acc_cmb=2 means each accumulator lane is 64 bits (one per u64 lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccWidth {
    /// 32-bit accumulator lanes (acc_cmb=1). 16 outputs in 8 u64 lanes.
    Acc32,
    /// 64-bit accumulator lanes (acc_cmb=2). 8 outputs in 8 u64 lanes.
    Acc64,
}

/// Parsed matrix multiply configuration from instruction bits.
///
/// Encapsulates the tile geometry and operational modes extracted from the
/// config word that accompanies each matrix multiply instruction. This
/// replaces the hardcoded geometry tables in vector_matmul.rs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulConfig {
    /// Tile rows (M dimension). How many rows of A contribute to the output.
    pub rows: u32,
    /// Inner dimension (K). Reduction dimension summed over.
    pub inner: u32,
    /// Tile columns (N dimension). How many columns of B contribute.
    pub cols: u32,
    /// Whether to accumulate (add to existing acc) or replace.
    /// When true: acc += A * B. When false: acc = A * B.
    pub accumulate: bool,
    /// Sign extension mode for X (A) buffer. True = signed, false = unsigned.
    pub x_signed: bool,
    /// Sign extension mode for Y (B) buffer. True = signed, false = unsigned.
    pub y_signed: bool,
    /// Element type for input A.
    pub a_type: ElementType,
    /// Element type for input B.
    pub b_type: ElementType,
    /// Accumulator width mode.
    pub acc_width: AccWidth,
    /// Whether this is a bfloat16 mode (fp32 accumulator).
    pub bfloat: bool,
    /// Whether to subtract (negate the product) rather than add.
    pub subtract: bool,
    /// Whether this is a sparse mode (B operand has a sparsity mask).
    ///
    /// Sparse modes have doubled inner dimension compared to their dense
    /// equivalents. The B operand comes from a composite qx register
    /// (vector data + 64-bit mask). Elements at masked-out positions are
    /// treated as zero during the multiply.
    pub sparse: bool,
    /// Raw bit width of A (X) elements from the geometry table.
    ///
    /// Usually matches `a_type.bits()`, but for 4-bit modes the a_type
    /// field uses Int8 (no Int4 variant exists). Use this field instead
    /// of `a_type.bits()` in computation paths that need the true width.
    pub bits_x: u32,
    /// Raw bit width of B (Y) elements from the geometry table.
    ///
    /// See `bits_x` for rationale. Critical for i8xi4 mode where
    /// `b_type` is Int8 but the actual element width is 4 bits.
    pub bits_y: u32,
}

impl MatMulConfig {
    /// Parse a matmul configuration from element types and mode flags.
    ///
    /// This is the primary constructor. Given the element types of the two
    /// operands and the operational flags (accumulate, sign, subtract), it
    /// looks up the correct tile geometry from the hardware's geometry table.
    ///
    /// The geometry is uniquely determined by the (bits_x, bits_y, bfloat,
    /// acc_cmb) tuple for the default (non-variant) modes. When multiple
    /// geometries exist for the same type pair (the acc_cmb=2 modes), the
    /// first variant (smaller rows, larger cols) is selected by default.
    /// Use `from_geometry` to select a specific variant.
    ///
    /// # Arguments
    ///
    /// * `a_type` - Element type of the A (X) input buffer
    /// * `b_type` - Element type of the B (Y) input buffer
    /// * `accumulate` - If true, add product to existing accumulator value
    /// * `x_signed` - Sign extension for X buffer
    /// * `y_signed` - Sign extension for Y buffer
    /// * `subtract` - If true, subtract the product instead of adding
    pub fn from_types(
        a_type: ElementType,
        b_type: ElementType,
        accumulate: bool,
        x_signed: bool,
        y_signed: bool,
        subtract: bool,
    ) -> Option<Self> {
        let bits_x = a_type.bits() as u32;
        let bits_y = b_type.bits() as u32;
        let bfloat = matches!(a_type, ElementType::BFloat16) && matches!(b_type, ElementType::BFloat16);

        // Find the first matching geometry entry.
        let entry = DENSE_GEOMETRY_TABLE
            .iter()
            .find(|e| e.bits_x == bits_x && e.bits_y == bits_y && e.bfloat == bfloat)?;

        Some(Self {
            rows: entry.rows,
            inner: entry.inner,
            cols: entry.cols,
            accumulate,
            x_signed,
            y_signed,
            a_type,
            b_type,
            acc_width: if entry.acc_cmb == 2 {
                AccWidth::Acc64
            } else {
                AccWidth::Acc32
            },
            bfloat: entry.bfloat,
            subtract,
            sparse: false,
            bits_x: entry.bits_x,
            bits_y: entry.bits_y,
        })
    }

    /// Look up a matmul configuration by explicit geometry dimensions.
    ///
    /// Allows selecting a specific variant when multiple geometries exist
    /// for the same element type pair (e.g., int16 x int8 with acc_cmb=2
    /// has both 2x8x8 and 4x8x4 variants).
    ///
    /// Returns None if no matching geometry exists in the hardware table.
    pub fn from_geometry(
        a_type: ElementType,
        b_type: ElementType,
        rows: u32,
        inner: u32,
        cols: u32,
        accumulate: bool,
        x_signed: bool,
        y_signed: bool,
        subtract: bool,
    ) -> Option<Self> {
        let bits_x = a_type.bits() as u32;
        let bits_y = b_type.bits() as u32;
        let bfloat = matches!(a_type, ElementType::BFloat16) && matches!(b_type, ElementType::BFloat16);

        let entry = DENSE_GEOMETRY_TABLE.iter().find(|e| {
            e.bits_x == bits_x
                && e.bits_y == bits_y
                && e.bfloat == bfloat
                && e.rows == rows
                && e.inner == inner
                && e.cols == cols
        })?;

        Some(Self {
            rows: entry.rows,
            inner: entry.inner,
            cols: entry.cols,
            accumulate,
            x_signed,
            y_signed,
            a_type,
            b_type,
            acc_width: if entry.acc_cmb == 2 {
                AccWidth::Acc64
            } else {
                AccWidth::Acc32
            },
            bfloat: entry.bfloat,
            subtract,
            sparse: false,
            bits_x: entry.bits_x,
            bits_y: entry.bits_y,
        })
    }

    /// Parse a matmul configuration from a pmode index.
    ///
    /// The pmode is the permute mode index as defined by the hardware. It
    /// indexes into the complete permute mode table (which includes not just
    /// matrix multiply modes but also element-wise, convolution, FFT, and
    /// sparse modes). This method only handles the dense matrix multiply
    /// subset (pmode 0..=9 in the default constants.py ordering).
    ///
    /// The sign and accumulate fields are not part of pmode -- they come
    /// from separate instruction bits.
    ///
    /// # Arguments
    ///
    /// * `pmode` - Permute mode index (0-based)
    /// * `x_signed` - Sign extension for X buffer
    /// * `y_signed` - Sign extension for Y buffer
    /// * `accumulate` - Whether to accumulate
    /// * `subtract` - Whether to subtract
    pub fn from_pmode(
        pmode: u32,
        x_signed: bool,
        y_signed: bool,
        accumulate: bool,
        subtract: bool,
    ) -> Option<Self> {
        // The pmode maps directly to entries in the dense geometry table
        // for pmode 0..9 (the basic matrix multiply permute modes).
        let entry = DENSE_GEOMETRY_TABLE.get(pmode as usize)?;

        let (a_type, b_type) = element_types_from_entry(entry);

        Some(Self {
            rows: entry.rows,
            inner: entry.inner,
            cols: entry.cols,
            accumulate,
            x_signed,
            y_signed,
            a_type,
            b_type,
            acc_width: if entry.acc_cmb == 2 {
                AccWidth::Acc64
            } else {
                AccWidth::Acc32
            },
            bfloat: entry.bfloat,
            subtract,
            sparse: false,
            bits_x: entry.bits_x,
            bits_y: entry.bits_y,
        })
    }

    /// Get the total number of output elements produced by this multiply.
    ///
    /// For acc_cmb=1 (32-bit acc): rows * cols outputs, each 32 bits.
    /// For acc_cmb=2 (64-bit acc): rows * cols outputs, each 64 bits.
    /// The bfloat mode produces acc_num=16 outputs (4x4, fp32 each).
    pub fn output_count(&self) -> u32 {
        self.rows * self.cols
    }

    /// Get the number of multiply-accumulate operations per output element.
    ///
    /// Each output element is the dot product of one row of A with one
    /// column of B, summing over the inner dimension.
    pub fn macs_per_output(&self) -> u32 {
        self.inner
    }

    /// Get all valid dense geometry entries for a given element type pair.
    ///
    /// Some type pairs have multiple valid geometries (e.g., int16 x int8
    /// with 64-bit accumulator can be 2x8x8 or 4x8x4). This returns all
    /// of them.
    pub fn valid_geometries(a_type: ElementType, b_type: ElementType) -> Vec<(u32, u32, u32)> {
        let bits_x = a_type.bits() as u32;
        let bits_y = b_type.bits() as u32;
        let bfloat = matches!(a_type, ElementType::BFloat16) && matches!(b_type, ElementType::BFloat16);

        DENSE_GEOMETRY_TABLE
            .iter()
            .filter(|e| e.bits_x == bits_x && e.bits_y == bits_y && e.bfloat == bfloat)
            .map(|e| (e.rows, e.inner, e.cols))
            .collect()
    }

    /// Get all valid sparse geometry entries for a given element type pair.
    pub fn valid_sparse_geometries(a_type: ElementType, b_type: ElementType) -> Vec<(u32, u32, u32)> {
        let bits_x = a_type.bits() as u32;
        let bits_y = b_type.bits() as u32;
        let bfloat = matches!(a_type, ElementType::BFloat16) && matches!(b_type, ElementType::BFloat16);

        SPARSE_GEOMETRY_TABLE
            .iter()
            .filter(|e| e.bits_x == bits_x && e.bits_y == bits_y && e.bfloat == bfloat)
            .map(|e| (e.rows, e.inner, e.cols))
            .collect()
    }

    /// Parse a matmul configuration from a hardware config word.
    ///
    /// The config word is passed in a scalar register and encodes the
    /// geometry (amode, bmode, variant) plus operational flags (signedness,
    /// zero_acc, subtract). Bit layout from `aiev2_compute_control()`:
    ///
    /// ```text
    /// Bit  0:     zero_acc  (zero accumulator before operation)
    /// Bits 1-2:   amode     (accumulator combining mode)
    /// Bits 3-4:   bmode     (B element width selector)
    /// Bits 5-7:   variant   (geometry variant)
    /// Bit  8:     sgn_y     (Y/B signedness: 1=signed, 0=unsigned)
    /// Bit  9:     sgn_x     (X/A signedness: 1=signed, 0=unsigned)
    /// Bit  10:    shift16   (shift output by 16 bits)
    /// Bit  11:    sub0      (subtract control 0)
    /// Bit  12:    sub1      (subtract control 1)
    /// Bit  13:    sub2      (subtract control 2)
    /// Bits 16+:   sub_mask  (per-element subtract mask)
    /// ```
    ///
    /// Returns None if the (amode, bmode, variant) triple does not
    /// correspond to a known geometry.
    pub fn from_config_word(conf: u32, is_bf16: bool) -> Option<Self> {
        let zero_acc = (conf & 1) != 0;
        let amode = (conf >> 1) & 3;
        let bmode = (conf >> 3) & 3;
        let variant = (conf >> 5) & 7;
        let sgn_y = ((conf >> 8) & 1) != 0;
        let sgn_x = ((conf >> 9) & 1) != 0;
        let _shift16 = ((conf >> 10) & 1) != 0;
        let sub0 = ((conf >> 11) & 1) != 0;
        let _sub1 = ((conf >> 12) & 1) != 0;
        let _sub2 = ((conf >> 13) & 1) != 0;

        let entry = if is_bf16 {
            lookup_bf16_geometry(variant)
        } else {
            lookup_integer_geometry(amode, bmode, variant)
        }?;

        let (a_type, b_type) = element_types_from_entry(entry);

        Some(Self {
            rows: entry.rows,
            inner: entry.inner,
            cols: entry.cols,
            accumulate: !zero_acc,
            x_signed: sgn_x,
            y_signed: sgn_y,
            a_type,
            b_type,
            acc_width: if entry.acc_cmb == 2 {
                AccWidth::Acc64
            } else {
                AccWidth::Acc32
            },
            bfloat: entry.bfloat,
            bits_x: entry.bits_x,
            bits_y: entry.bits_y,
            subtract: sub0,
            sparse: entry.sparse,
        })
    }
}

/// Look up geometry for integer MAC from config word fields.
fn lookup_integer_geometry(amode: u32, bmode: u32, variant: u32) -> Option<&'static GeometryEntry> {
    CONFIG_GEOMETRY_TABLE
        .iter()
        .find(|(a, b, v, _)| *a == amode && *b == bmode && *v == variant)
        .map(|(_, _, _, entry)| entry)
}

/// Look up geometry for bf16 MAC from config word variant field.
///
/// BFloat16 MAC uses a separate instruction format (vmac.f) with only
/// the variant field selecting geometry.
fn lookup_bf16_geometry(variant: u32) -> Option<&'static GeometryEntry> {
    match variant {
        // Dense 4x8x4: 16 outputs in 32-bit accumulator lanes
        0 => Some(&GeometryEntry {
            bits_x: 16,
            bits_y: 16,
            rows: 4,
            inner: 8,
            cols: 4,
            acc_cmb: 1,
            bfloat: true,
            sparse: false,
        }),
        // Element-wise: 16 channels, each a 1x2 dot product.
        // Lane L computes A[L]*B[L] + A[L+16]*B[L+16] -- stride-16
        // symmetric mapping for both A and B sides. Handled in
        // matmul_config_driven's is_elemwise path.
        1 => Some(&GeometryEntry {
            bits_x: 16,
            bits_y: 16,
            rows: 16,
            inner: 2,
            cols: 1,
            acc_cmb: 1,
            bfloat: true,
            sparse: false,
        }),
        // Sparse: doubled inner dimension (4x16x4)
        2 => Some(&GeometryEntry {
            bits_x: 16,
            bits_y: 16,
            rows: 4,
            inner: 16,
            cols: 4,
            acc_cmb: 1,
            bfloat: true,
            sparse: true,
        }),
        _ => None,
    }
}

/// Map a geometry entry back to ElementType pairs.
///
/// Uses the bit width and bfloat flag to determine the canonical element
/// types. For integer types, defaults to the signed variant (the sign
/// extension is a separate concern controlled by instruction bits, not
/// the geometry).
fn element_types_from_entry(entry: &GeometryEntry) -> (ElementType, ElementType) {
    let a_type = if entry.bfloat && entry.bits_x == 16 {
        ElementType::BFloat16
    } else {
        match entry.bits_x {
            8 => ElementType::Int8,
            16 => ElementType::Int16,
            32 => ElementType::Int32,
            other => panic!("element_types_from_entry: unexpected bits_x={} in geometry {:?}", other, entry,),
        }
    };

    let b_type = if entry.bfloat && entry.bits_y == 16 {
        ElementType::BFloat16
    } else {
        match entry.bits_y {
            4 => ElementType::Int8, // int4 represented as Int8 (no Int4 variant)
            8 => ElementType::Int8,
            16 => ElementType::Int16,
            other => panic!("element_types_from_entry: unexpected bits_y={} in geometry {:?}", other, entry,),
        }
    };

    (a_type, b_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // int8 x int8 geometry
    // -------------------------------------------------------------------

    #[test]
    fn test_i8xi8_geometry() {
        let config = MatMulConfig::from_types(ElementType::Int8, ElementType::Int8, true, true, true, false)
            .expect("i8xi8 should have a valid geometry");

        // Hardware: mmode 1 (8x8 acc_cmb=1), rows=4, inner=8, cols=8
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 8);
        assert_eq!(config.cols, 8);
        assert_eq!(config.acc_width, AccWidth::Acc32);
        assert!(!config.bfloat);
        assert_eq!(config.output_count(), 32);
    }

    #[test]
    fn test_i8xi8_sign_and_accumulate() {
        let config =
            MatMulConfig::from_types(ElementType::Int8, ElementType::Int8, false, false, true, false)
                .unwrap();

        assert!(!config.accumulate);
        assert!(!config.x_signed);
        assert!(config.y_signed);
        assert!(!config.subtract);
    }

    #[test]
    fn test_u8xu8_same_geometry_as_i8xi8() {
        // Unsigned int8 uses the same geometry; sign is a separate concern.
        let config =
            MatMulConfig::from_types(ElementType::UInt8, ElementType::UInt8, true, false, false, false)
                .unwrap();

        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 8);
        assert_eq!(config.cols, 8);
        assert_eq!(config.acc_width, AccWidth::Acc32);
    }

    // -------------------------------------------------------------------
    // int16 x int16 geometry (32-bit accumulator)
    // -------------------------------------------------------------------

    #[test]
    fn test_i16xi16_geometry_acc32() {
        let config =
            MatMulConfig::from_types(ElementType::Int16, ElementType::Int16, true, true, true, false)
                .unwrap();

        // Hardware: mmode 3 (16x16 acc_cmb=1), rows=4, inner=2, cols=8
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 2);
        assert_eq!(config.cols, 8);
        assert_eq!(config.acc_width, AccWidth::Acc32);
        assert!(!config.bfloat);
        assert_eq!(config.output_count(), 32);
    }

    // -------------------------------------------------------------------
    // int16 x int8 geometry
    // -------------------------------------------------------------------

    #[test]
    fn test_i16xi8_geometry_acc32() {
        let config =
            MatMulConfig::from_types(ElementType::Int16, ElementType::Int8, true, true, true, false).unwrap();

        // Hardware: mmode 2 (16x8 acc_cmb=1), rows=4, inner=4, cols=8
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 4);
        assert_eq!(config.cols, 8);
        assert_eq!(config.acc_width, AccWidth::Acc32);
    }

    // -------------------------------------------------------------------
    // bf16 x bf16 geometry
    // -------------------------------------------------------------------

    #[test]
    fn test_bf16xbf16_geometry() {
        let config =
            MatMulConfig::from_types(ElementType::BFloat16, ElementType::BFloat16, true, true, true, false)
                .unwrap();

        // Hardware: mmode 6 (16x16 bfloat acc_cmb=1 acc_num=16),
        // rows=4, inner=8, cols=4
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 8);
        assert_eq!(config.cols, 4);
        assert_eq!(config.acc_width, AccWidth::Acc32);
        assert!(config.bfloat);
        assert_eq!(config.output_count(), 16);
    }

    // -------------------------------------------------------------------
    // int32 x int16 geometry (64-bit accumulator)
    // -------------------------------------------------------------------

    #[test]
    fn test_i32xi16_geometry() {
        let config =
            MatMulConfig::from_types(ElementType::Int32, ElementType::Int16, true, true, true, false)
                .unwrap();

        // Hardware: mmode 7 (32x16 acc_cmb=2), rows=4, inner=2, cols=4
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 2);
        assert_eq!(config.cols, 4);
        assert_eq!(config.acc_width, AccWidth::Acc64);
        assert_eq!(config.output_count(), 16);
    }

    // -------------------------------------------------------------------
    // Variant geometry selection (acc_cmb=2 modes with multiple options)
    // -------------------------------------------------------------------

    #[test]
    fn test_i16xi8_acc64_variant_2x8x8() {
        let config = MatMulConfig::from_geometry(
            ElementType::Int16,
            ElementType::Int8,
            2,
            8,
            8,
            true,
            true,
            true,
            false,
        )
        .unwrap();

        assert_eq!(config.rows, 2);
        assert_eq!(config.inner, 8);
        assert_eq!(config.cols, 8);
        assert_eq!(config.acc_width, AccWidth::Acc64);
    }

    #[test]
    fn test_i16xi8_acc64_variant_4x8x4() {
        let config = MatMulConfig::from_geometry(
            ElementType::Int16,
            ElementType::Int8,
            4,
            8,
            4,
            true,
            true,
            true,
            false,
        )
        .unwrap();

        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 8);
        assert_eq!(config.cols, 4);
        assert_eq!(config.acc_width, AccWidth::Acc64);
    }

    #[test]
    fn test_i16xi16_acc64_variant_2x4x8() {
        let config = MatMulConfig::from_geometry(
            ElementType::Int16,
            ElementType::Int16,
            2,
            4,
            8,
            true,
            true,
            true,
            false,
        )
        .unwrap();

        assert_eq!(config.rows, 2);
        assert_eq!(config.inner, 4);
        assert_eq!(config.cols, 8);
        assert_eq!(config.acc_width, AccWidth::Acc64);
    }

    #[test]
    fn test_i16xi16_acc64_variant_4x4x4() {
        let config = MatMulConfig::from_geometry(
            ElementType::Int16,
            ElementType::Int16,
            4,
            4,
            4,
            true,
            true,
            true,
            false,
        )
        .unwrap();

        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 4);
        assert_eq!(config.cols, 4);
        assert_eq!(config.acc_width, AccWidth::Acc64);
    }

    // -------------------------------------------------------------------
    // Invalid geometry rejection
    // -------------------------------------------------------------------

    #[test]
    fn test_invalid_geometry_rejected() {
        // 4x4x4 is not a valid geometry for int8 x int8.
        let config = MatMulConfig::from_geometry(
            ElementType::Int8,
            ElementType::Int8,
            4,
            4,
            4,
            true,
            true,
            true,
            false,
        );
        assert!(config.is_none());
    }

    #[test]
    fn test_float32_not_a_matmul_type() {
        // Float32 is not a valid direct matmul input type (bf16 is).
        // Float32 as a_type has bits=32, which matches int32 geometry when
        // paired with int16, but Float32 x Float32 has no entry.
        let config =
            MatMulConfig::from_types(ElementType::Float32, ElementType::Float32, true, true, true, false);
        assert!(config.is_none());
    }

    // -------------------------------------------------------------------
    // pmode-based construction
    // -------------------------------------------------------------------

    #[test]
    fn test_pmode_0_i8xi4() {
        let config = MatMulConfig::from_pmode(0, true, true, true, false).unwrap();
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 16);
        assert_eq!(config.cols, 8);
        assert_eq!(config.a_type, ElementType::Int8);
    }

    #[test]
    fn test_pmode_1_i8xi8() {
        let config = MatMulConfig::from_pmode(1, true, true, true, false).unwrap();
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 8);
        assert_eq!(config.cols, 8);
    }

    #[test]
    fn test_pmode_2_i16xi8() {
        let config = MatMulConfig::from_pmode(2, true, true, true, false).unwrap();
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 4);
        assert_eq!(config.cols, 8);
    }

    #[test]
    fn test_pmode_3_i16xi16_acc32() {
        let config = MatMulConfig::from_pmode(3, true, true, true, false).unwrap();
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 2);
        assert_eq!(config.cols, 8);
        assert_eq!(config.acc_width, AccWidth::Acc32);
    }

    #[test]
    fn test_pmode_8_bf16() {
        let config = MatMulConfig::from_pmode(8, true, true, true, false).unwrap();
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 8);
        assert_eq!(config.cols, 4);
        assert!(config.bfloat);
        assert_eq!(config.a_type, ElementType::BFloat16);
        assert_eq!(config.b_type, ElementType::BFloat16);
    }

    #[test]
    fn test_pmode_9_i32xi16() {
        let config = MatMulConfig::from_pmode(9, true, true, true, false).unwrap();
        assert_eq!(config.rows, 4);
        assert_eq!(config.inner, 2);
        assert_eq!(config.cols, 4);
        assert_eq!(config.acc_width, AccWidth::Acc64);
    }

    #[test]
    fn test_pmode_out_of_range() {
        assert!(MatMulConfig::from_pmode(99, true, true, true, false).is_none());
    }

    // -------------------------------------------------------------------
    // Accumulate mode bit
    // -------------------------------------------------------------------

    #[test]
    fn test_accumulate_true() {
        let config =
            MatMulConfig::from_types(ElementType::Int8, ElementType::Int8, true, true, true, false).unwrap();
        assert!(config.accumulate);
    }

    #[test]
    fn test_accumulate_false() {
        let config =
            MatMulConfig::from_types(ElementType::Int8, ElementType::Int8, false, true, true, false).unwrap();
        assert!(!config.accumulate);
    }

    // -------------------------------------------------------------------
    // Sign extension bits
    // -------------------------------------------------------------------

    #[test]
    fn test_signed_signed() {
        let config =
            MatMulConfig::from_types(ElementType::Int8, ElementType::Int8, true, true, true, false).unwrap();
        assert!(config.x_signed);
        assert!(config.y_signed);
    }

    #[test]
    fn test_unsigned_unsigned() {
        let config =
            MatMulConfig::from_types(ElementType::UInt8, ElementType::UInt8, true, false, false, false)
                .unwrap();
        assert!(!config.x_signed);
        assert!(!config.y_signed);
    }

    #[test]
    fn test_signed_unsigned() {
        let config =
            MatMulConfig::from_types(ElementType::Int8, ElementType::UInt8, true, true, false, false)
                .unwrap();
        assert!(config.x_signed);
        assert!(!config.y_signed);
    }

    // -------------------------------------------------------------------
    // Subtract mode
    // -------------------------------------------------------------------

    #[test]
    fn test_subtract_true() {
        let config =
            MatMulConfig::from_types(ElementType::Int16, ElementType::Int16, true, true, true, true).unwrap();
        assert!(config.subtract);
    }

    #[test]
    fn test_subtract_false() {
        let config =
            MatMulConfig::from_types(ElementType::Int16, ElementType::Int16, true, true, true, false)
                .unwrap();
        assert!(!config.subtract);
    }

    // -------------------------------------------------------------------
    // valid_geometries enumeration
    // -------------------------------------------------------------------

    #[test]
    fn test_valid_geometries_i8xi8() {
        let geoms = MatMulConfig::valid_geometries(ElementType::Int8, ElementType::Int8);
        assert_eq!(geoms.len(), 1);
        assert_eq!(geoms[0], (4, 8, 8));
    }

    #[test]
    fn test_valid_geometries_i16xi8_has_three() {
        // int16 x int8 has two modes: acc_cmb=1 (4x4x8) and
        // acc_cmb=2 (2x8x8, 4x8x4).
        let geoms = MatMulConfig::valid_geometries(ElementType::Int16, ElementType::Int8);
        assert_eq!(geoms.len(), 3);
        assert!(geoms.contains(&(4, 4, 8))); // acc_cmb=1
        assert!(geoms.contains(&(2, 8, 8))); // acc_cmb=2 variant 1
        assert!(geoms.contains(&(4, 8, 4))); // acc_cmb=2 variant 2
    }

    #[test]
    fn test_valid_geometries_i16xi16_has_three() {
        let geoms = MatMulConfig::valid_geometries(ElementType::Int16, ElementType::Int16);
        assert_eq!(geoms.len(), 3);
        assert!(geoms.contains(&(4, 2, 8))); // acc_cmb=1
        assert!(geoms.contains(&(2, 4, 8))); // acc_cmb=2 variant 1
        assert!(geoms.contains(&(4, 4, 4))); // acc_cmb=2 variant 2
    }

    #[test]
    fn test_valid_geometries_bf16() {
        let geoms = MatMulConfig::valid_geometries(ElementType::BFloat16, ElementType::BFloat16);
        assert_eq!(geoms.len(), 1);
        assert_eq!(geoms[0], (4, 8, 4));
    }

    #[test]
    fn test_valid_geometries_i32xi16() {
        let geoms = MatMulConfig::valid_geometries(ElementType::Int32, ElementType::Int16);
        assert_eq!(geoms.len(), 1);
        assert_eq!(geoms[0], (4, 2, 4));
    }

    // -------------------------------------------------------------------
    // Sparse geometry enumeration
    // -------------------------------------------------------------------

    #[test]
    fn test_sparse_geometries_i8xi8() {
        let geoms = MatMulConfig::valid_sparse_geometries(ElementType::Int8, ElementType::Int8);
        assert_eq!(geoms.len(), 1);
        assert_eq!(geoms[0], (4, 16, 8));
    }

    #[test]
    fn test_sparse_geometries_bf16() {
        let geoms = MatMulConfig::valid_sparse_geometries(ElementType::BFloat16, ElementType::BFloat16);
        assert_eq!(geoms.len(), 1);
        assert_eq!(geoms[0], (4, 16, 4));
    }

    // -------------------------------------------------------------------
    // Derived quantities
    // -------------------------------------------------------------------

    #[test]
    fn test_macs_per_output_i8xi8() {
        let config =
            MatMulConfig::from_types(ElementType::Int8, ElementType::Int8, true, true, true, false).unwrap();
        assert_eq!(config.macs_per_output(), 8);
    }

    #[test]
    fn test_macs_per_output_bf16() {
        let config =
            MatMulConfig::from_types(ElementType::BFloat16, ElementType::BFloat16, true, true, true, false)
                .unwrap();
        assert_eq!(config.macs_per_output(), 8);
    }

    #[test]
    fn test_output_count_i16xi16_acc32() {
        let config =
            MatMulConfig::from_types(ElementType::Int16, ElementType::Int16, true, true, true, false)
                .unwrap();
        // 4 rows x 8 cols = 32
        assert_eq!(config.output_count(), 32);
    }

    #[test]
    fn test_output_count_bf16() {
        let config =
            MatMulConfig::from_types(ElementType::BFloat16, ElementType::BFloat16, true, true, true, false)
                .unwrap();
        // 4 rows x 4 cols = 16
        assert_eq!(config.output_count(), 16);
    }

    // -------------------------------------------------------------------
    // All 6+ type combinations produce valid geometry (comprehensive)
    // -------------------------------------------------------------------

    #[test]
    fn test_all_type_combinations_have_geometry() {
        // These are all the valid A x B type pairs that the hardware supports.
        let valid_pairs: &[(ElementType, ElementType)] = &[
            (ElementType::Int8, ElementType::Int8),         // 8x8 dense
            (ElementType::UInt8, ElementType::UInt8),       // same hw, unsigned
            (ElementType::Int16, ElementType::Int8),        // 16x8
            (ElementType::Int16, ElementType::Int16),       // 16x16
            (ElementType::BFloat16, ElementType::BFloat16), // bf16
            (ElementType::Int32, ElementType::Int16),       // 32x16
        ];

        for (a, b) in valid_pairs {
            let config = MatMulConfig::from_types(*a, *b, true, true, true, false);
            assert!(config.is_some(), "Expected valid geometry for {:?} x {:?}", a, b);
            let c = config.unwrap();
            // Sanity: output count must be positive and fit the acc register.
            assert!(c.output_count() > 0);
            assert!(c.output_count() <= 32);
        }
    }

    // -------------------------------------------------------------------
    // Consistency: from_types and from_geometry agree
    // -------------------------------------------------------------------

    #[test]
    fn test_from_types_matches_from_geometry_i8xi8() {
        let by_type =
            MatMulConfig::from_types(ElementType::Int8, ElementType::Int8, true, true, true, false).unwrap();

        let by_geom = MatMulConfig::from_geometry(
            ElementType::Int8,
            ElementType::Int8,
            by_type.rows,
            by_type.inner,
            by_type.cols,
            true,
            true,
            true,
            false,
        )
        .unwrap();

        assert_eq!(by_type, by_geom);
    }

    #[test]
    fn test_from_types_matches_from_geometry_bf16() {
        let by_type =
            MatMulConfig::from_types(ElementType::BFloat16, ElementType::BFloat16, false, true, true, true)
                .unwrap();

        let by_geom = MatMulConfig::from_geometry(
            ElementType::BFloat16,
            ElementType::BFloat16,
            by_type.rows,
            by_type.inner,
            by_type.cols,
            false,
            true,
            true,
            true,
        )
        .unwrap();

        assert_eq!(by_type, by_geom);
    }
}
