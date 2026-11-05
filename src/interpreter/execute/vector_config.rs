//! AIE2 matrix multiply configuration word parser.
//!
//! The AIE2 vector unit's matrix multiply instructions encode their tile
//! geometry, accumulator mode, and sign/subtract configuration in a "config
//! word" (built by `aiev2_compute_control()` in llvm-aie's `aiev2_vmult.h`).
//! This module parses that config word into a structured representation.
//!
//! The config word's (amode, bmode, variant) fields select:
//! - Element widths and accumulator combining
//! - The lane map ([`MacShape`]): a matmul tile (rows, inner, cols), an
//!   elementwise product, a convolution or a multi-channel tile
//! - Dense or sparse operation
//!
//! and its remaining bits give:
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
//! Geometries are extracted at build time from llvm-aie's `aiev2_vmult.h`
//! (see `xdna_archspec::aie2::matmul`); the config word bit layout is
//! `aiev2_compute_control()` from the same header.

use crate::interpreter::bundle::ElementType;
use xdna_archspec::aie2::matmul::{
    canonical_key, dense_geometries, dense_integer_geometry, float_conf_term_neg_rule, geometry_for,
    sparse_geometries, term_neg_rule, GeometryEntry, MacShape, TermNegRule,
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
    /// Which A and B elements feed each accumulator lane.
    pub shape: MacShape,
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
    /// Config-word `sub1`: negate the first accumulator input (bf16 forms only;
    /// the integer path does not read it).
    pub sub_acc1: bool,
    /// Config-word `sub2`: negate the second accumulator input (bf16 add* forms).
    pub sub_acc2: bool,
    /// Config-word `shift16` (bf16 forms; the integer path reads the raw word): acc1's 32 bits
    /// are shifted left by 16 before they join the float sum (silicon, `bfupper` probe).
    pub shift16: bool,
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
    /// Config-word `term_neg` field (bits 23:16): sign controls of the post-adder's term slots.
    pub term_neg: u8,
    /// The float-conf multiplier negates every product before it is wrapped to 14 bits (the
    /// instruction's own sign, `msc` / `negmul` / ..., folded with `sub_mul` by the caller).
    pub negate_products: bool,
    /// Which bit negates which partial product, for the geometry (`None`: no rule, the field has
    /// no effect on this mode).
    pub term_neg_rule: Option<TermNegRule>,
    /// What produces the products.
    pub multiplier: Multiplier,
}

/// What the multiplier array does under a config word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Multiplier {
    /// The integer array with the lane map of `shape`.
    Array,
    /// `amode` 2 or 3 on an integer instruction (the float multiplier mode): the lane map of
    /// `shape` over the low byte of 16-bit elements, each product wrapped to 14 bits (signed).
    /// The result fills 16 32-bit lanes; the rest of the accumulator is zero. Silicon-derived
    /// (`docs/fidelity-gaps/mac-conf.md`).
    FloatConf,
    /// A key no mode uses (or a sparse mode on a dense instruction): every product is zero.
    Idle,
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

        // Find the first matching matmul tile.
        let entry = dense_geometries().find(|e| {
            e.bits_x == bits_x && e.bits_y == bits_y && e.bfloat == bfloat && e.shape.matmul_dims().is_some()
        })?;

        Some(Self {
            shape: entry.shape,
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
            sub_acc1: false,
            sub_acc2: false,
            shift16: false,
            sparse: false,
            bits_x: entry.bits_x,
            bits_y: entry.bits_y,
            term_neg: 0,
            negate_products: false,
            term_neg_rule: None,
            multiplier: Multiplier::Array,
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

        let entry = dense_geometries().find(|e| {
            e.bits_x == bits_x
                && e.bits_y == bits_y
                && e.bfloat == bfloat
                && e.shape.matmul_dims() == Some((rows, inner, cols))
        })?;

        Some(Self {
            shape: entry.shape,
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
            sub_acc1: false,
            sub_acc2: false,
            shift16: false,
            sparse: false,
            bits_x: entry.bits_x,
            bits_y: entry.bits_y,
            term_neg: 0,
            negate_products: false,
            term_neg_rule: None,
            multiplier: Multiplier::Array,
        })
    }

    /// Get the total number of output elements produced by this multiply.
    ///
    /// For acc_cmb=1 (32-bit acc) each is 32 bits, for acc_cmb=2 (64-bit acc)
    /// 64 bits. The bfloat tile produces 16 fp32 outputs.
    pub fn output_count(&self) -> u32 {
        self.shape.lanes()
    }

    /// Get the number of multiply-accumulate operations per output element:
    /// the products summed into each lane (the inner dimension of a tile).
    pub fn macs_per_output(&self) -> u32 {
        self.shape.terms()
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

        dense_geometries()
            .filter(|e| e.bits_x == bits_x && e.bits_y == bits_y && e.bfloat == bfloat)
            .filter_map(|e| e.shape.matmul_dims())
            .collect()
    }

    /// Get all valid sparse geometry entries for a given element type pair.
    pub fn valid_sparse_geometries(a_type: ElementType, b_type: ElementType) -> Vec<(u32, u32, u32)> {
        let bits_x = a_type.bits() as u32;
        let bits_y = b_type.bits() as u32;
        let bfloat = matches!(a_type, ElementType::BFloat16) && matches!(b_type, ElementType::BFloat16);

        sparse_geometries()
            .filter(|e| e.bits_x == bits_x && e.bits_y == bits_y && e.bfloat == bfloat)
            .filter_map(|e| e.shape.matmul_dims())
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
    /// correspond to a geometry in the table (the caller must then report the
    /// op unhandled, not drop it).
    pub fn from_config_word(conf: u32, is_bf16: bool) -> Option<Self> {
        let zero_acc = (conf & 1) != 0;
        let amode = (conf >> 1) & 3;
        let bmode = (conf >> 3) & 3;
        let variant = (conf >> 5) & 7;
        let sgn_y = ((conf >> 8) & 1) != 0;
        let sgn_x = ((conf >> 9) & 1) != 0;
        let sub0 = ((conf >> 11) & 1) != 0;
        let sub1 = ((conf >> 12) & 1) != 0;
        let sub2 = ((conf >> 13) & 1) != 0;
        let term_neg = (conf >> 16) as u8;

        let mut config = if is_bf16 {
            // Silicon (bfcodes probe): on a .f instruction amode 2 and 3 are the float multiplier
            // and bmode is ignored; cmode 0 is the 4x8x4 tile, 1 the 16-lane pair form, 2 the
            // sparse tile, every other cmode a zero product (the accumulators still pass through
            // the float adder). term_neg follows the float-conf rule. amode 0 and 1 (an integer
            // key) do something else on a .f instruction that is not modelled.
            if amode < 2 {
                return None;
            }
            match lookup_bf16_geometry(variant) {
                Some(entry) => {
                    let mut c = Self::from_entry(entry, Multiplier::Array);
                    c.term_neg_rule = float_conf_term_neg_rule(variant);
                    c
                }
                None => Self::bf16_idle(),
            }
        } else if amode >= 2 {
            // The float multiplier mode on an integer instruction; bmode is ignored.
            // cmode 2 is the sparse bf16 tile's layout under the float-conf products (a sparse
            // instruction only; silicon, `sparsekeys` probe); its term_neg is not probed.
            let (shape, rule, sparse) = match variant {
                0 => (MacShape::Matmul { rows: 4, inner: 8, cols: 4 }, float_conf_term_neg_rule(0), false),
                1 => (MacShape::Elementwise { lanes: 16, k: 2 }, float_conf_term_neg_rule(1), false),
                2 => (lookup_bf16_geometry(2)?.shape, None, true),
                _ => return Some(Self::idle(amode)),
            };
            let mut c = Self::from_entry(
                &GeometryEntry { bits_x: 16, bits_y: 16, shape, acc_cmb: 1, bfloat: false, sparse },
                Multiplier::FloatConf,
            );
            c.term_neg_rule = rule;
            c
        } else {
            // The header's modes (dense and sparse; a sparse one on a dense instruction is
            // demoted to `Idle` by the caller) and the silicon-only complex modes.
            let key = canonical_key(amode, bmode, variant);
            match geometry_for(key.0, key.1, key.2)
                .filter(|e| !e.bfloat)
                .or_else(|| dense_integer_geometry(amode, bmode, variant))
            {
                Some(entry) => {
                    let mut c = Self::from_entry(entry, Multiplier::Array);
                    c.term_neg_rule = term_neg_rule(amode, bmode, variant);
                    c
                }
                None => Self::idle(amode),
            }
        };
        config.accumulate = !zero_acc;
        config.x_signed = sgn_x;
        config.y_signed = sgn_y;
        config.subtract = sub0;
        config.sub_acc1 = sub1;
        config.sub_acc2 = sub2;
        config.shift16 = (conf >> 10) & 1 != 0;
        config.term_neg = term_neg;
        Some(config)
    }

    /// Accumulator lanes the instruction writes: the multiplier's lanes, except in the float
    /// conf mode, where the other 16 lanes see the accumulator inputs and a zero product.
    pub fn accumulator_lanes(&self) -> u32 {
        if self.multiplier == Multiplier::FloatConf {
            32
        } else {
            self.shape.lanes()
        }
    }

    fn from_entry(entry: &GeometryEntry, multiplier: Multiplier) -> Self {
        let (a_type, b_type) = element_types_from_entry(entry);
        Self {
            shape: entry.shape,
            accumulate: true,
            x_signed: false,
            y_signed: false,
            a_type,
            b_type,
            acc_width: if entry.acc_cmb == 2 {
                AccWidth::Acc64
            } else {
                AccWidth::Acc32
            },
            bfloat: entry.bfloat,
            subtract: false,
            sub_acc1: false,
            sub_acc2: false,
            shift16: false,
            sparse: entry.sparse,
            bits_x: entry.bits_x,
            bits_y: entry.bits_y,
            term_neg: 0,
            negate_products: false,
            term_neg_rule: None,
            multiplier,
        }
    }

    /// A bf16 key that selects no multiplier mode: no product, the 16 fp32 accumulator lanes go
    /// through the float adder.
    pub fn bf16_idle() -> Self {
        let entry = lookup_bf16_geometry(0).expect("the bf16 4x8x4 geometry");
        Self::from_entry(entry, Multiplier::Idle)
    }

    /// A key that selects no multiplier mode: every product is zero. The accumulator lanes are
    /// 64-bit under `amode` 1 and 32-bit otherwise.
    pub fn idle(amode: u32) -> Self {
        let lanes = if amode == 1 { 16 } else { 32 };
        let entry = GeometryEntry {
            bits_x: 8,
            bits_y: 8,
            shape: MacShape::Elementwise { lanes, k: 1 },
            acc_cmb: if amode == 1 { 2 } else { 1 },
            bfloat: false,
            sparse: false,
        };
        Self::from_entry(&entry, Multiplier::Idle)
    }
}

/// aiev2_vmult.h gives every bfloat16 intrinsic amode 2 and bmode 3.
const BF16_AMODE: u32 = 2;
const BF16_BMODE: u32 = 3;

/// Look up geometry for bf16 MAC from config word variant field.
///
/// BFloat16 MAC uses a separate instruction format (vmac.f) with only
/// the variant field selecting geometry: dense 4x8x4 (variant 0), elementwise
/// 16 lanes of two products (variant 1) and sparse 4x16x4 (variant 2).
fn lookup_bf16_geometry(variant: u32) -> Option<&'static GeometryEntry> {
    geometry_for(BF16_AMODE, BF16_BMODE, variant).filter(|entry| entry.bfloat)
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

    /// Every mode the toolchain header builds has a geometry, so the config
    /// word of every one of them parses, to the shape the table gives.
    #[test]
    fn every_header_mode_parses_to_its_table_geometry() {
        use xdna_archspec::aie2::matmul::MAC_CONFIG_MODES;

        for mode in MAC_CONFIG_MODES {
            let conf = (mode.amode << 1) | (mode.bmode << 3) | (mode.variant << 5);
            let parsed = MatMulConfig::from_config_word(conf, mode.bfloat).unwrap_or_else(|| {
                panic!("mode ({}, {}, {}) `{}` must parse", mode.amode, mode.bmode, mode.variant, mode.family)
            });
            let entry = geometry_for(mode.amode, mode.bmode, mode.variant).unwrap();
            assert_eq!(parsed.shape, entry.shape, "{}", mode.family);
            assert_eq!(
                (parsed.bits_x, parsed.bits_y, parsed.bfloat),
                (entry.bits_x, entry.bits_y, entry.bfloat)
            );
            assert_eq!(parsed.sparse, entry.sparse);
        }
    }

    /// bf16 configs still reject a variant no float mode uses; every integer word parses.
    #[test]
    /// bfcodes probe: on a .f instruction amode 2 and 3 select the float multiplier whatever
    /// bmode; cmode 3..7 are a zero product; an integer key (amode 0, 1) is not modelled.
    fn bf16_keys_follow_silicon() {
        for amode in [2, 3] {
            for bmode in 0..4 {
                let k = amode << 1 | bmode << 3;
                assert_eq!(MatMulConfig::from_config_word(k, true).unwrap().shape.lanes(), 16);
                let e = MatMulConfig::from_config_word(k | 1 << 5, true).unwrap();
                assert_eq!(e.shape, MacShape::Elementwise { lanes: 16, k: 2 });
                for cmode in 3..8 {
                    let c = MatMulConfig::from_config_word(k | cmode << 5, true).unwrap();
                    assert_eq!(c.multiplier, Multiplier::Idle);
                    assert!(c.bfloat);
                }
            }
        }
        assert!(MatMulConfig::from_config_word(0, true).is_none());
        assert!(MatMulConfig::from_config_word(1 << 1 | 1 << 5, true).is_none());
    }

    fn word(amode: u32, bmode: u32, cmode: u32) -> u32 {
        (amode << 1) | (bmode << 3) | (cmode << 5)
    }

    /// A key no mode uses selects the idle multiplier (zero products), 64-bit lanes under amode 1.
    #[test]
    fn an_unused_key_selects_the_idle_multiplier() {
        for (amode, bmode, cmode, lanes) in [(0, 0, 7, 32), (0, 3, 2, 32), (1, 0, 3, 16), (1, 2, 7, 16)] {
            let c = MatMulConfig::from_config_word(word(amode, bmode, cmode), false).unwrap();
            assert_eq!(c.multiplier, Multiplier::Idle, "({amode}, {bmode}, {cmode})");
            assert_eq!(c.shape.lanes(), lanes);
        }
    }

    /// With a 64-bit accumulator, bmode 1 is the same multiplier mode as bmode 3.
    #[test]
    fn amode_one_bmode_one_is_bmode_three() {
        for cmode in 0..8 {
            let a = MatMulConfig::from_config_word(word(1, 1, cmode), false).unwrap();
            let b = MatMulConfig::from_config_word(word(1, 3, cmode), false).unwrap();
            assert_eq!(
                (a.shape, a.bits_x, a.bits_y, a.multiplier),
                (b.shape, b.bits_x, b.bits_y, b.multiplier)
            );
        }
        // ... but not with a 32-bit accumulator (8x8 against 16x16).
        let a = MatMulConfig::from_config_word(word(0, 1, 0), false).unwrap();
        let b = MatMulConfig::from_config_word(word(0, 3, 0), false).unwrap();
        assert_ne!((a.bits_x, a.bits_y), (b.bits_x, b.bits_y));
    }

    /// amode 2 and 3 select the float multiplier on an integer instruction, whatever bmode is;
    /// cmode 0 is the 4x8x4 tile, cmode 1 the 16-lane pair form, the others are idle.
    #[test]
    fn float_amodes_select_the_float_conf_multiplier() {
        for amode in [2, 3] {
            for bmode in 0..4 {
                let c0 = MatMulConfig::from_config_word(word(amode, bmode, 0), false).unwrap();
                assert_eq!(
                    (c0.multiplier, c0.shape),
                    (Multiplier::FloatConf, MacShape::Matmul { rows: 4, inner: 8, cols: 4 })
                );
                let c1 = MatMulConfig::from_config_word(word(amode, bmode, 1), false).unwrap();
                assert_eq!(
                    (c1.multiplier, c1.shape),
                    (Multiplier::FloatConf, MacShape::Elementwise { lanes: 16, k: 2 })
                );
                assert_eq!(c1.accumulator_lanes(), 32);
                // cmode 2: the sparse bf16 tile under the float-conf products (a sparse
                // instruction; a dense one demotes it to a zero product).
                let c2 = MatMulConfig::from_config_word(word(amode, bmode, 2), false).unwrap();
                assert_eq!((c2.multiplier, c2.sparse, c2.bfloat), (Multiplier::FloatConf, true, false));
                assert_eq!(
                    MatMulConfig::from_config_word(word(amode, bmode, 3), false).unwrap().multiplier,
                    Multiplier::Idle
                );
            }
        }
    }

    /// The two complex modes the header does not build.
    #[test]
    fn complex_modes_parse() {
        let c = MatMulConfig::from_config_word(word(1, 0, 1), false).unwrap();
        assert_eq!(
            (c.shape, c.bits_x, c.bits_y, c.acc_width),
            (MacShape::ComplexElem { sets: 1 }, 32, 16, AccWidth::Acc64)
        );
        for bmode in [1, 3] {
            let c = MatMulConfig::from_config_word(word(1, bmode, 4), false).unwrap();
            assert_eq!((c.shape, c.bits_x, c.bits_y), (MacShape::ComplexElem { sets: 2 }, 16, 16));
        }
    }

    /// Every field of the word lands where `VecConf` puts it; the reserved bits (15:14, 31:24) read nothing.
    #[test]
    fn every_conf_field_is_decoded_from_its_bit() {
        let base = word(0, 1, 0);
        let c = MatMulConfig::from_config_word(
            base | 1 | 1 << 8 | 1 << 11 | 1 << 12 | 1 << 13 | 0xA5 << 16,
            false,
        )
        .unwrap();
        assert!(!c.accumulate && c.y_signed && !c.x_signed && c.subtract && c.sub_acc1 && c.sub_acc2);
        assert_eq!(c.term_neg, 0xA5);
        let c = MatMulConfig::from_config_word(base | 1 << 9, false).unwrap();
        assert!(c.accumulate && c.x_signed && !c.y_signed && c.term_neg == 0);
        let reserved = MatMulConfig::from_config_word(base | 3 << 14 | 0xFF << 24, false).unwrap();
        let plain = MatMulConfig::from_config_word(base, false).unwrap();
        assert_eq!(
            (reserved.term_neg, reserved.subtract, reserved.sub_acc1, reserved.sub_acc2, reserved.accumulate),
            (plain.term_neg, plain.subtract, plain.sub_acc1, plain.sub_acc2, plain.accumulate)
        );
    }

    /// Integer elementwise, convolution and 2-channel modes parse, with the
    /// signedness flags straight from the config word (bit 9 is X, bit 8 Y).
    #[test]
    fn non_matmul_integer_modes_parse_with_their_kind_and_signs() {
        let word =
            |a: u32, b: u32, v: u32, sx: u32, sy: u32| (a << 1) | (b << 3) | (v << 5) | (sy << 8) | (sx << 9);
        let c = MatMulConfig::from_config_word(word(0, 1, 1, 1, 0), false).unwrap();
        assert_eq!(c.shape, MacShape::Elementwise { lanes: 32, k: 2 });
        assert_eq!(
            (c.x_signed, c.y_signed, c.acc_width, c.bits_x, c.bits_y),
            (true, false, AccWidth::Acc32, 8, 8)
        );
        assert_eq!(c.output_count(), 32);
        assert_eq!(c.macs_per_output(), 2);

        let c = MatMulConfig::from_config_word(word(1, 3, 3, 0, 1), false).unwrap();
        assert_eq!(c.shape, MacShape::Conv { outputs: 16, taps: 4, channels: 1 });
        assert_eq!(
            (c.x_signed, c.y_signed, c.acc_width, c.bits_x, c.bits_y),
            (false, true, AccWidth::Acc64, 16, 16)
        );

        let c = MatMulConfig::from_config_word(word(0, 2, 1, 1, 1), false).unwrap();
        assert_eq!(c.shape, MacShape::MultiChannel { rows: 4, inner: 4, cols: 4, channels: 2 });
        assert_eq!((c.bits_x, c.bits_y), (16, 8));
        assert_eq!((c.a_type, c.b_type), (ElementType::Int16, ElementType::Int8));
    }

    /// The bf16 elementwise geometry is the same structured kind as the
    /// integer elementwise modes: it is read from the table, not special-cased.
    #[test]
    fn bf16_elementwise_is_the_shared_elementwise_kind() {
        let c = MatMulConfig::from_config_word(BF16_AMODE << 1 | BF16_BMODE << 3 | 1 << 5, true).unwrap();
        assert_eq!(c.shape, MacShape::Elementwise { lanes: 16, k: 2 });
        assert!(c.bfloat);
        assert_eq!((c.a_type, c.b_type), (ElementType::BFloat16, ElementType::BFloat16));
        assert_eq!(c.output_count(), 16);
        assert_eq!(c.macs_per_output(), 2);
        // Same lookup path as every other mode.
        assert_eq!(geometry_for(2, 3, 1).unwrap().shape, c.shape);
    }

    /// The matmul-only constructors never hand back a non-tile shape, even
    /// for operand widths that elementwise or convolution modes share.
    #[test]
    fn matmul_constructors_only_select_tiles() {
        for (a, b) in [
            (ElementType::Int8, ElementType::Int8),
            (ElementType::Int16, ElementType::Int16),
            (ElementType::Int16, ElementType::Int8),
            (ElementType::BFloat16, ElementType::BFloat16),
        ] {
            let c = MatMulConfig::from_types(a, b, true, true, true, false).unwrap();
            assert!(c.shape.matmul_dims().is_some(), "{a:?}x{b:?}");
            for (r, i, n) in MatMulConfig::valid_geometries(a, b) {
                assert!(MatMulConfig::from_geometry(a, b, r, i, n, true, true, true, false).is_some());
            }
        }
    }

    // -------------------------------------------------------------------
    // int8 x int8 geometry
    // -------------------------------------------------------------------

    #[test]
    fn test_i8xi8_geometry() {
        let config = MatMulConfig::from_types(ElementType::Int8, ElementType::Int8, true, true, true, false)
            .expect("i8xi8 should have a valid geometry");

        // Hardware: mmode 1 (8x8 acc_cmb=1), rows=4, inner=8, cols=8
        assert_eq!(config.shape.matmul_dims(), Some((4, 8, 8)));
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

        assert_eq!(config.shape.matmul_dims(), Some((4, 8, 8)));
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
        assert_eq!(config.shape.matmul_dims(), Some((4, 2, 8)));
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
        assert_eq!(config.shape.matmul_dims(), Some((4, 4, 8)));
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
        assert_eq!(config.shape.matmul_dims(), Some((4, 8, 4)));
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
        assert_eq!(config.shape.matmul_dims(), Some((4, 2, 4)));
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

        assert_eq!(config.shape.matmul_dims(), Some((2, 8, 8)));
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

        assert_eq!(config.shape.matmul_dims(), Some((4, 8, 4)));
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

        assert_eq!(config.shape.matmul_dims(), Some((2, 4, 8)));
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

        assert_eq!(config.shape.matmul_dims(), Some((4, 4, 4)));
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
            by_type.shape.matmul_dims().unwrap().0,
            by_type.shape.matmul_dims().unwrap().1,
            by_type.shape.matmul_dims().unwrap().2,
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
            by_type.shape.matmul_dims().unwrap().0,
            by_type.shape.matmul_dims().unwrap().1,
            by_type.shape.matmul_dims().unwrap().2,
            false,
            true,
            true,
            true,
        )
        .unwrap();

        assert_eq!(by_type, by_geom);
    }
}
