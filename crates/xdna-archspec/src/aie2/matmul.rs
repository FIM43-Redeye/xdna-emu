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

/// What a MAC-array config mode does with its operands, taken from the
/// `__builtin_aiev2_*` call each header intrinsic returns.
///
/// The multiplying kinds are the vector MAC family; `Add`, `Sub`, `NegAdd`,
/// `NegSub` and `Neg` combine accumulators without multiplying but are built
/// with the same config word, so they appear in the mode table too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacOpKind {
    Mul,
    NegMul,
    Mac,
    Msc,
    NegMac,
    NegMsc,
    AddMac,
    AddMsc,
    SubMac,
    SubMsc,
    Add,
    Sub,
    NegAdd,
    NegSub,
    Neg,
}

impl MacOpKind {
    /// The TableGen mnemonic stem of the instruction that executes this kind
    /// (`vmac`; bfloat16 modes append `.f`, integer modes carry no suffix for
    /// the accumulator forms). Follows the llvm-aie naming convention, which
    /// the mnemonic-existence test in this module checks against the decoder
    /// tables.
    pub const fn mnemonic_stem(self) -> &'static str {
        match self {
            MacOpKind::Mul => "vmul",
            MacOpKind::NegMul => "vnegmul",
            MacOpKind::Mac => "vmac",
            MacOpKind::Msc => "vmsc",
            MacOpKind::NegMac => "vnegmac",
            MacOpKind::NegMsc => "vnegmsc",
            MacOpKind::AddMac => "vaddmac",
            MacOpKind::AddMsc => "vaddmsc",
            MacOpKind::SubMac => "vsubmac",
            MacOpKind::SubMsc => "vsubmsc",
            MacOpKind::Add => "vadd",
            MacOpKind::Sub => "vsub",
            MacOpKind::NegAdd => "vnegadd",
            MacOpKind::NegSub => "vnegsub",
            MacOpKind::Neg => "vneg",
        }
    }

    /// Whether the kind multiplies operands (as opposed to only combining
    /// accumulators).
    pub const fn is_multiply(self) -> bool {
        !matches!(
            self,
            MacOpKind::Add | MacOpKind::Sub | MacOpKind::NegAdd | MacOpKind::NegSub | MacOpKind::Neg
        )
    }
}

/// One (amode, bmode, variant) value of the MAC config word, with what the
/// header does with it. Unlike [`GeometryEntry`] this covers every mode the
/// header builds, including elementwise, convolution, multi-channel and
/// accumulator-only ones that have no matmul tile shape.
///
/// Generated at build time from llvm-aie's `aiev2_vmult.h`: every intrinsic
/// that calls `aiev2_compute_control(sgn_x, sgn_y, amode, bmode, variant, ...)`
/// contributes its config key, its signedness arguments (`__SIGN_*` macros
/// read from `aiev2intrin.h`, literals, or a runtime parameter that can be
/// either), the builtin it returns (the operation kind), and the operand
/// types (bfloat16 or not). See `mac_geometry_extract::parse_vmult_modes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacConfigMode {
    pub amode: u32,
    pub bmode: u32,
    pub variant: u32,
    /// Distinct `(sgn_x, sgn_y)` pairs the header can produce (1 = signed,
    /// 0 = unsigned, per the `__SIGN_*` macros), sorted.
    pub signs: &'static [(u8, u8)],
    /// Whether the multiplying intrinsics take bfloat16 operands.
    pub bfloat: bool,
    /// Intrinsic family name for human docs (`4x16_16x8`, `elem_16_2`,
    /// `conv_4x4_8ch`): the multiplying intrinsics' name without operation
    /// prefix or `_conf` suffix. Not a key: operand-type overloads reuse it.
    pub family: &'static str,
    /// Operation kinds the header emits for this mode, in `MacOpKind` order.
    pub ops: &'static [MacOpKind],
    /// Whether [`CONFIG_GEOMETRY_TABLE`] has a matmul geometry for this mode.
    pub has_geometry: bool,
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

    // ---- MAC_CONFIG_MODES ----

    fn mode(amode: u32, bmode: u32, variant: u32) -> &'static MacConfigMode {
        MAC_CONFIG_MODES
            .iter()
            .find(|m| (m.amode, m.bmode, m.variant) == (amode, bmode, variant))
            .unwrap_or_else(|| panic!("no config mode ({amode}, {bmode}, {variant})"))
    }

    const ALL_SIGNS: &[(u8, u8)] = &[(0, 0), (0, 1), (1, 0), (1, 1)];

    /// The header builds 26 distinct keys textually, but two of them (the
    /// complex-int `elem_8` at (1, 0, 1) and `elem_8_2` at (1, 3, 4)) sit in
    /// `#if 0 // Complex Type` regions, so 24 are real.
    #[test]
    fn mode_table_has_every_active_config_key_once_and_sorted() {
        let keys: Vec<_> = MAC_CONFIG_MODES.iter().map(|m| (m.amode, m.bmode, m.variant)).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(keys, sorted);
        assert_eq!(keys.len(), 24);
        assert!(!keys.contains(&(1, 0, 1)) && !keys.contains(&(1, 3, 4)));
    }

    #[test]
    fn every_matmul_geometry_key_is_a_config_mode_with_matching_flags() {
        for (a, b, v, geometry) in CONFIG_GEOMETRY_TABLE {
            let m = mode(*a, *b, *v);
            assert!(m.has_geometry, "({a}, {b}, {v})");
            assert_eq!(m.bfloat, geometry.bfloat, "({a}, {b}, {v})");
        }
        let with_geometry = MAC_CONFIG_MODES.iter().filter(|m| m.has_geometry).count();
        assert_eq!(with_geometry, CONFIG_GEOMETRY_TABLE.len());
    }

    #[test]
    fn modes_without_a_matmul_shape_are_the_elementwise_conv_and_multichannel_ones() {
        let without: Vec<_> = MAC_CONFIG_MODES
            .iter()
            .filter(|m| !m.has_geometry)
            .map(|m| ((m.amode, m.bmode, m.variant), m.family))
            .collect();
        assert_eq!(
            without,
            vec![
                ((0, 1, 1), "elem_32_2"),
                ((0, 1, 2), "conv_4x4_8ch"),
                ((0, 1, 3), "conv_8x8_4ch"),
                ((0, 1, 4), "conv_32x8"),
                ((0, 2, 1), "4x4_4x4_2ch"),
                ((0, 3, 1), "elem_32"),
                ((1, 3, 2), "elem_16_2"),
                ((1, 3, 3), "conv_16x4"),
                ((2, 3, 1), "elem_16_2"),
            ]
        );
    }

    #[test]
    fn signedness_sets_follow_the_header_arguments() {
        // Integer modes take sgn_x/sgn_y as parameters or via the four
        // __SIGN_* flavours: every combination.
        assert_eq!(mode(0, 0, 0).signs, ALL_SIGNS);
        assert_eq!(mode(0, 1, 1).signs, ALL_SIGNS);
        assert_eq!(mode(1, 3, 2).signs, ALL_SIGNS);
        // bfloat16 modes pass literal 0, 0.
        for m in MAC_CONFIG_MODES.iter().filter(|m| m.bfloat) {
            assert_eq!(m.signs, &[(0, 0)], "({}, {}, {})", m.amode, m.bmode, m.variant);
        }
        assert!(mode(2, 3, 0).bfloat && mode(2, 3, 1).bfloat && mode(2, 3, 2).bfloat);
        assert_eq!(MAC_CONFIG_MODES.iter().filter(|m| m.bfloat).count(), 3);
        // Every integer mode's pairs are within the four possible ones, sorted.
        for m in MAC_CONFIG_MODES {
            assert!(m.signs.windows(2).all(|w| w[0] < w[1]), "{m:?}");
            assert!(m.signs.iter().all(|s| ALL_SIGNS.contains(s)), "{m:?}");
        }
    }

    #[test]
    fn family_names_are_the_intrinsic_names_without_operation_or_conf() {
        assert_eq!(mode(0, 0, 0).family, "4x16_16x8");
        assert_eq!(mode(1, 3, 2).family, "elem_16_2");
        assert_eq!(mode(2, 3, 1).family, "elem_16_2");
        assert_eq!(mode(0, 1, 2).family, "conv_4x4_8ch");
        assert_eq!(mode(0, 2, 1).family, "4x4_4x4_2ch");
        assert!(MAC_CONFIG_MODES
            .iter()
            .all(|m| !m.family.is_empty() && !m.family.ends_with("_conf")));
    }

    #[test]
    fn op_kinds_per_mode_match_the_builtins() {
        use MacOpKind::*;
        // Integer matmul: the ten multiplying kinds, plus acc32 add/sub/negadd/neg at (0, 0, 0).
        assert_eq!(
            mode(0, 1, 0).ops,
            &[Mul, NegMul, Mac, Msc, NegMac, NegMsc, AddMac, AddMsc, SubMac, SubMsc]
        );
        assert_eq!(
            mode(0, 0, 0).ops,
            &[Mul, NegMul, Mac, Msc, NegMac, NegMsc, AddMac, AddMsc, SubMac, SubMsc, Add, Sub, NegAdd, Neg]
        );
        // bfloat16 has no negmac/negmsc/submac/submsc builtins.
        assert_eq!(mode(2, 3, 1).ops, &[Mul, NegMul, Mac, Msc, AddMac, AddMsc]);
        // The bf16 accumulator-only builtins are the accfloat forms.
        assert_eq!(mode(2, 3, 0).ops, &[Mul, NegMul, Mac, Msc, AddMac, AddMsc, Add, Sub, NegSub, Neg]);
        for m in MAC_CONFIG_MODES {
            assert!(m.ops.iter().any(|k| k.is_multiply()), "{m:?}");
            assert!(m.ops.windows(2).all(|w| (w[0] as usize) < (w[1] as usize)), "{m:?}");
        }
    }

    #[test]
    fn mnemonic_stems_are_distinct_and_classified() {
        use MacOpKind::*;
        let all = [
            Mul, NegMul, Mac, Msc, NegMac, NegMsc, AddMac, AddMsc, SubMac, SubMsc, Add, Sub, NegAdd, NegSub,
            Neg,
        ];
        let mut stems: Vec<_> = all.iter().map(|k| k.mnemonic_stem()).collect();
        stems.sort();
        stems.dedup();
        assert_eq!(stems.len(), all.len());
        assert_eq!(all.iter().filter(|k| k.is_multiply()).count(), 10);
        assert_eq!(Mac.mnemonic_stem(), "vmac");
        assert_eq!(NegSub.mnemonic_stem(), "vnegsub");
    }

    /// Every operation kind the header emits for a mode must exist as a
    /// TableGen mnemonic: the bare stem for integer modes, `<stem>.f` for
    /// bfloat16 modes. This ties the hand-written stem names to the ISA.
    #[test]
    fn every_mode_op_kind_has_a_tablegen_mnemonic() {
        let isa = crate::aie2::isa::load_from_generated();
        let mnemonics: std::collections::HashSet<&str> =
            isa.encodings_by_slot.values().flatten().map(|e| e.mnemonic.as_str()).collect();
        for m in MAC_CONFIG_MODES {
            for kind in m.ops {
                let expected = if m.bfloat {
                    format!("{}.f", kind.mnemonic_stem())
                } else {
                    kind.mnemonic_stem().to_string()
                };
                assert!(
                    mnemonics.contains(expected.as_str()),
                    "mode ({}, {}, {}) kind {kind:?}: no TableGen mnemonic `{expected}`",
                    m.amode,
                    m.bmode,
                    m.variant
                );
            }
        }
    }
}
