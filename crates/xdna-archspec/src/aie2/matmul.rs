//! AIE2 MAC-array geometry table.
//!
//! The lane map of every config mode of the AIE2 vector multiply array
//! (which A and B elements are multiplied and summed into each accumulator
//! lane), keyed by the (amode, bmode, variant) fields of the MAC config word.
//! A geometry is one of four structured kinds ([`MacShape`]): a matrix tile,
//! an elementwise product, a sliding-window convolution, or a multi-channel
//! tile.
//!
//! ## Source
//!
//! Generated at build time from llvm-aie's `aiev2_vmult.h`
//! (`clang/lib/Headers/aiev2/`): each multiplying intrinsic's family name
//! (`4x16_16x8`, `elem_32_2`, `conv_4x4_8ch`, `4x4_4x4_2ch`, ...) names the
//! kind and its parameters, its `aiev2_compute_control(sgn_x, sgn_y, amode,
//! bmode, variant, ...)` call gives the config key, and its operand and
//! result types give the element and accumulator widths (and are
//! cross-checked against the name). See `mac_geometry_extract` for the parser
//! and `build.rs` for the wiring.
//!
//! What the header does not give is the operand index map, i.e. how a
//! lane's A and B indices follow from the parameters ([`MacShape::operand_indices`]).
//! That was observed on NPU1 silicon with one-hot probes and validated with
//! held-out random captures; `docs/fidelity-gaps/vector-compute.md` records
//! the evidence and the alternatives that silicon ruled out.
//!
//! ## Hardware Background
//!
//! The AIE2 multiply array has 512 physical multiplier cells operating at
//! 8-bit x 4-bit granularity. Input vectors are permuted and reinterpreted
//! as 2D matrix tiles, and the products are summed through a post-add tree
//! (PSA) into accumulator lanes. The tile dimensions vary by element type
//! pair because wider elements consume more multiplier cells per operation.

/// The lane map of a MAC config mode: how many accumulator lanes it fills,
/// how many products each lane sums, and which A and B element feeds each
/// product. Parameters come from the intrinsic family name in the toolchain
/// header; the index map of each kind was observed on silicon.
///
/// Indices are element indices into the A and B vectors (`v64int8` has 64
/// elements), not byte offsets. Lanes are numbered as the accumulator stores
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacShape {
    /// A `rows x inner` row-major tile of A times an `inner x cols` row-major
    /// tile of B into a `rows x cols` row-major tile of lanes
    /// (`4x16_16x8`): lane `r * cols + c` sums `A[r][k] * B[k][c]` over `k`.
    Matmul { rows: u32, inner: u32, cols: u32 },
    /// `lanes` independent products, each lane summing `k` strided pairs
    /// (`elem_32_2`): lane `l` sums `A[l + t * lanes] * B[l + t * lanes]`
    /// over `t < k`.
    Elementwise { lanes: u32, k: u32 },
    /// A one-dimensional sliding window over channel-interleaved data
    /// (`conv_4x4_8ch`, `conv_32x8`): lane `p * channels + c` sums
    /// `A[(p + t) * channels + c] * B[t * channels + c]` over `t < taps`.
    /// A is the signal (`outputs + taps - 1` positions are read), B the
    /// filter taps.
    Conv { outputs: u32, taps: u32, channels: u32 },
    /// `channels` independent matmul tiles with the channel as the innermost
    /// index of every operand and of the result (`4x4_4x4_2ch`): lane
    /// `(r * cols + c) * channels + ch` sums `A[(r * inner + k) * channels + ch]
    /// * B[(k * cols + c) * channels + ch]` over `k`.
    MultiChannel {
        rows: u32,
        inner: u32,
        cols: u32,
        channels: u32,
    },
}

impl MacShape {
    /// Accumulator lanes the mode fills.
    pub const fn lanes(&self) -> u32 {
        match *self {
            MacShape::Matmul { rows, cols, .. } => rows * cols,
            MacShape::Elementwise { lanes, .. } => lanes,
            MacShape::Conv { outputs, channels, .. } => outputs * channels,
            MacShape::MultiChannel { rows, cols, channels, .. } => rows * cols * channels,
        }
    }

    /// Products summed into each lane.
    pub const fn terms(&self) -> u32 {
        match *self {
            MacShape::Matmul { inner, .. } | MacShape::MultiChannel { inner, .. } => inner,
            MacShape::Elementwise { k, .. } => k,
            MacShape::Conv { taps, .. } => taps,
        }
    }

    /// The (A element, B element) pair that product `term` of `lane` multiplies.
    ///
    /// `lane < lanes()` and `term < terms()`; out-of-range arguments are a
    /// caller bug and give an out-of-range index or a division by zero for
    /// degenerate shapes.
    pub const fn operand_indices(&self, lane: u32, term: u32) -> (u32, u32) {
        match *self {
            MacShape::Matmul { rows: _, inner, cols } => {
                let (r, c) = (lane / cols, lane % cols);
                (r * inner + term, term * cols + c)
            }
            MacShape::Elementwise { lanes, .. } => {
                let index = lane + term * lanes;
                (index, index)
            }
            MacShape::Conv { channels, .. } => {
                let (p, c) = (lane / channels, lane % channels);
                ((p + term) * channels + c, term * channels + c)
            }
            MacShape::MultiChannel { rows: _, inner, cols, channels } => {
                let (ch, tile_lane) = (lane % channels, lane / channels);
                let (r, c) = (tile_lane / cols, tile_lane % cols);
                ((r * inner + term) * channels + ch, (term * cols + c) * channels + ch)
            }
        }
    }

    /// The `(rows, inner, cols)` of a matrix tile, `None` for the other kinds.
    pub const fn matmul_dims(&self) -> Option<(u32, u32, u32)> {
        match *self {
            MacShape::Matmul { rows, inner, cols } => Some((rows, inner, cols)),
            _ => None,
        }
    }

    /// How many elements of A and of B the mode reads (the highest index
    /// plus one), for sizing against the operand vectors.
    pub const fn operand_extent(&self) -> (u32, u32) {
        match *self {
            MacShape::Matmul { rows, inner, cols } => (rows * inner, inner * cols),
            MacShape::Elementwise { lanes, k } => (lanes * k, lanes * k),
            MacShape::Conv { outputs, taps, channels } => ((outputs + taps - 1) * channels, taps * channels),
            MacShape::MultiChannel { rows, inner, cols, channels } => {
                (rows * inner * channels, inner * cols * channels)
            }
        }
    }
}

/// One valid geometry: the lane map and element types that a
/// (amode, bmode, variant) config word selects.
#[derive(Debug, Clone, Copy)]
pub struct GeometryEntry {
    /// Element width of A input in bits.
    pub bits_x: u32,
    /// Element width of B input in bits.
    pub bits_y: u32,
    /// Which A and B elements feed each accumulator lane.
    pub shape: MacShape,
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
/// header does with it. Unlike [`GeometryEntry`] it also carries the
/// signedness pairs and operation kinds the header emits for the key; the
/// two tables list the same keys (checked by test).
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
}

include!(concat!(env!("OUT_DIR"), "/gen_mac_geometry.rs"));

/// The geometry a config key selects.
pub fn geometry_for(amode: u32, bmode: u32, variant: u32) -> Option<&'static GeometryEntry> {
    CONFIG_GEOMETRY_TABLE
        .iter()
        .find(|(a, b, v, _)| (*a, *b, *v) == (amode, bmode, variant))
        .map(|(_, _, _, entry)| entry)
}

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
        geometry_for(amode, bmode, variant)
            .unwrap_or_else(|| panic!("no geometry for ({amode}, {bmode}, {variant})"))
    }

    fn matmul(rows: u32, inner: u32, cols: u32) -> MacShape {
        MacShape::Matmul { rows, inner, cols }
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
        assert_eq!((e.bits_x, e.bits_y, e.shape, e.acc_cmb), (8, 4, matmul(4, 16, 8), 1));
        assert!(!e.sparse && !e.bfloat);

        let e = entry(1, 2, 1);
        assert_eq!((e.bits_x, e.bits_y, e.shape, e.acc_cmb), (16, 8, matmul(4, 8, 4), 2));

        let e = entry(2, 3, 0);
        assert_eq!((e.bits_x, e.bits_y, e.shape, e.acc_cmb), (16, 16, matmul(4, 8, 4), 1));
        assert!(e.bfloat && !e.sparse);
    }

    #[test]
    fn sparse_modes_are_flagged() {
        let e = entry(0, 0, 1);
        assert_eq!((e.bits_x, e.bits_y, e.shape), (8, 4, matmul(4, 32, 8)));
        assert!(e.sparse);

        let e = entry(2, 3, 2);
        assert_eq!(e.shape, matmul(4, 16, 4));
        assert!(e.sparse && e.bfloat);
    }

    #[test]
    fn every_sparse_inner_exceeds_its_dense_counterpart() {
        for sparse in sparse_geometries() {
            let (_, sparse_inner, sparse_cols) = sparse.shape.matmul_dims().expect("sparse modes are tiles");
            let dense_inner = dense_geometries()
                .filter_map(|d| d.shape.matmul_dims().map(|(_, inner, cols)| (d, inner, cols)))
                .filter(|(d, _, cols)| {
                    (d.bits_x, d.bits_y, d.bfloat, d.acc_cmb, *cols)
                        == (sparse.bits_x, sparse.bits_y, sparse.bfloat, sparse.acc_cmb, sparse_cols)
                })
                .map(|(_, inner, _)| inner)
                .min();
            if let Some(inner) = dense_inner {
                assert!(sparse_inner > inner, "{sparse:?} vs dense inner {inner}");
            }
        }
    }

    // ---- MacShape: the structured kinds ----

    /// Lane counts, products per lane and operand extents of each kind,
    /// worked out by hand from the definitions.
    #[test]
    fn shape_arithmetic_per_kind() {
        let m = matmul(4, 16, 8);
        assert_eq!(
            (m.lanes(), m.terms(), m.operand_extent(), m.matmul_dims()),
            (32, 16, (64, 128), Some((4, 16, 8)))
        );

        let e = MacShape::Elementwise { lanes: 32, k: 2 };
        assert_eq!((e.lanes(), e.terms(), e.operand_extent(), e.matmul_dims()), (32, 2, (64, 64), None));

        let c = MacShape::Conv { outputs: 4, taps: 4, channels: 8 };
        assert_eq!((c.lanes(), c.terms(), c.operand_extent()), (32, 4, (56, 32)));
        let c = MacShape::Conv { outputs: 16, taps: 4, channels: 1 };
        assert_eq!((c.lanes(), c.terms(), c.operand_extent()), (16, 4, (19, 4)));

        let mc = MacShape::MultiChannel { rows: 4, inner: 4, cols: 4, channels: 2 };
        assert_eq!((mc.lanes(), mc.terms(), mc.operand_extent()), (32, 4, (32, 32)));
    }

    /// Index maps, pinned at hand-picked lanes. The expected pairs are what
    /// the one-hot silicon probes decoded to (vec_macmode_*_probe goldens;
    /// `shapes_reproduce_the_silicon_probes` replays them in full).
    #[test]
    fn operand_indices_at_known_lanes() {
        // Matmul: lane r*cols+c sums A[r][k] * B[k][c].
        let m = matmul(4, 8, 4);
        assert_eq!(m.operand_indices(0, 0), (0, 0));
        assert_eq!(m.operand_indices(5, 3), (8 + 3, 3 * 4 + 1));

        // Elementwise: stride is the lane count (silicon: lane 3 of elem_32_2 pairs 3 and 35).
        let e = MacShape::Elementwise { lanes: 32, k: 2 };
        assert_eq!((e.operand_indices(3, 0), e.operand_indices(3, 1)), ((3, 3), (35, 35)));

        // Conv, one channel (conv_16x4): lane l reads A[l..l+4] against B[0..4].
        let c = MacShape::Conv { outputs: 16, taps: 4, channels: 1 };
        assert_eq!(
            (0..4).map(|t| c.operand_indices(5, t)).collect::<Vec<_>>(),
            [(5, 0), (6, 1), (7, 2), (8, 3)]
        );

        // Conv, channel-interleaved (conv_4x4_8ch): lane 8 is position 1, channel 0.
        let c = MacShape::Conv { outputs: 4, taps: 4, channels: 8 };
        assert_eq!(
            (0..4).map(|t| c.operand_indices(8, t)).collect::<Vec<_>>(),
            [(8, 0), (16, 8), (24, 16), (32, 24)]
        );
        // Lane 9 is position 1, channel 1.
        assert_eq!(c.operand_indices(9, 1), (17, 9));

        // Multi-channel (4x4_4x4_2ch): lane 1 is row 0, col 0, channel 1; lane 9 is row 1, col 0, channel 1.
        let mc = MacShape::MultiChannel { rows: 4, inner: 4, cols: 4, channels: 2 };
        assert_eq!(
            (0..4).map(|k| mc.operand_indices(1, k)).collect::<Vec<_>>(),
            [(1, 1), (3, 9), (5, 17), (7, 25)]
        );
        assert_eq!(
            (0..4).map(|k| mc.operand_indices(9, k)).collect::<Vec<_>>(),
            [(9, 1), (11, 9), (13, 17), (15, 25)]
        );
        // Lane 2 is row 0, col 1, channel 0.
        assert_eq!(
            (0..4).map(|k| mc.operand_indices(2, k)).collect::<Vec<_>>(),
            [(0, 2), (2, 10), (4, 18), (6, 26)]
        );
    }

    /// For every geometry in the table: indices stay inside the operand
    /// extent, no lane sums the same pair twice, and every operand element is
    /// read by some lane (a map that ignores part of an operand is wrong).
    #[test]
    fn every_geometry_index_map_is_in_range_and_covers_its_operands() {
        for (a, b, v, g) in CONFIG_GEOMETRY_TABLE {
            let (extent_a, extent_b) = g.shape.operand_extent();
            let mut seen_a = vec![false; extent_a as usize];
            let mut seen_b = vec![false; extent_b as usize];
            for lane in 0..g.shape.lanes() {
                let mut pairs = Vec::new();
                for term in 0..g.shape.terms() {
                    let (ia, ib) = g.shape.operand_indices(lane, term);
                    assert!(ia < extent_a && ib < extent_b, "({a},{b},{v}) lane {lane} term {term}");
                    seen_a[ia as usize] = true;
                    seen_b[ib as usize] = true;
                    pairs.push((ia, ib));
                }
                let distinct: std::collections::BTreeSet<_> = pairs.iter().collect();
                assert_eq!(distinct.len(), pairs.len(), "({a},{b},{v}) lane {lane} repeats a pair");
            }
            assert!(
                seen_a.iter().all(|&x| x) && seen_b.iter().all(|&x| x),
                "({a},{b},{v}) leaves operand elements unread"
            );
        }
    }

    /// The structured kinds each header family name yields, with the
    /// parameters the name gives. Matmul shapes keep their old values.
    #[test]
    fn non_matmul_families_are_extracted_as_their_kind() {
        let kind = |a, b, v| entry(a, b, v).shape;
        assert_eq!(kind(0, 1, 1), MacShape::Elementwise { lanes: 32, k: 2 });
        assert_eq!(kind(0, 3, 1), MacShape::Elementwise { lanes: 32, k: 1 });
        assert_eq!(kind(1, 3, 2), MacShape::Elementwise { lanes: 16, k: 2 });
        assert_eq!(kind(2, 3, 1), MacShape::Elementwise { lanes: 16, k: 2 });
        assert_eq!(kind(0, 1, 2), MacShape::Conv { outputs: 4, taps: 4, channels: 8 });
        assert_eq!(kind(0, 1, 3), MacShape::Conv { outputs: 8, taps: 8, channels: 4 });
        assert_eq!(kind(0, 1, 4), MacShape::Conv { outputs: 32, taps: 8, channels: 1 });
        assert_eq!(kind(1, 3, 3), MacShape::Conv { outputs: 16, taps: 4, channels: 1 });
        assert_eq!(kind(0, 2, 1), MacShape::MultiChannel { rows: 4, inner: 4, cols: 4, channels: 2 });
        // Operand widths and accumulator combining come from the vector types.
        let e = entry(0, 2, 1);
        assert_eq!((e.bits_x, e.bits_y, e.acc_cmb, e.bfloat, e.sparse), (16, 8, 1, false, false));
        let e = entry(1, 3, 3);
        assert_eq!((e.bits_x, e.bits_y, e.acc_cmb, e.bfloat, e.sparse), (16, 16, 2, false, false));
        let e = entry(2, 3, 1);
        assert_eq!((e.bits_x, e.bits_y, e.acc_cmb, e.bfloat, e.sparse), (16, 16, 1, true, false));
    }

    // ---- Silicon goldens (NPU1) ----

    const GOLDEN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/golden/silicon");

    struct Capture {
        a_bits: u32,
        b_bits: u32,
        a_lanes: usize,
        b_lanes: usize,
        acc_bits: u32,
        lanes: usize,
        tiles: usize,
        stages: usize,
        out_words: usize,
        inputs: Vec<Vec<u64>>,
        silicon: Vec<u64>,
        stem: String,
    }

    fn load_capture(kernel: &str) -> Capture {
        let path = format!("{GOLDEN_DIR}/{kernel}.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let num = |x: &serde_json::Value| x.as_u64().unwrap();
        let list = |x: &serde_json::Value| -> Vec<u64> { x.as_array().unwrap().iter().map(num).collect() };
        Capture {
            a_bits: num(&v["mode"]["a_bits"]) as u32,
            b_bits: num(&v["mode"]["b_bits"]) as u32,
            a_lanes: num(&v["mode"]["a_lanes"]) as usize,
            b_lanes: num(&v["mode"]["b_lanes"]) as usize,
            acc_bits: num(&v["mode"]["acc_bits"]) as u32,
            lanes: num(&v["mode"]["acc_lanes"]) as usize,
            tiles: num(&v["tile"]["tiles"]) as usize,
            stages: num(&v["tile"]["stages"]) as usize,
            out_words: num(&v["tile"]["out_words"]) as usize,
            inputs: v["inputs"].as_array().unwrap().iter().map(list).collect(),
            silicon: list(&v["silicon"]),
            stem: v["mode"]["stem"].as_str().unwrap().to_string(),
        }
    }

    /// The four signedness flavours a capture runs, as (x signed, y signed),
    /// in capture order (ss, su, us, uu).
    const SIGNS: [(bool, bool); 4] = [(true, true), (true, false), (false, true), (false, false)];

    fn extend(value: u64, bits: u32, signed: bool) -> i128 {
        let v = value as i128;
        if signed && value >> (bits - 1) & 1 == 1 {
            v - (1i128 << bits)
        } else {
            v
        }
    }

    /// Lanes of one product block, from the geometry's index map alone.
    fn product_lanes(shape: MacShape, a: &[i128], b: &[i128]) -> Vec<i128> {
        (0..shape.lanes())
            .map(|lane| {
                (0..shape.terms())
                    .map(|t| {
                        let (ia, ib) = shape.operand_indices(lane, t);
                        a[ia as usize] * b[ib as usize]
                    })
                    .sum()
            })
            .collect()
    }

    /// The silicon lanes of one readout (accumulator lane values as u64 bit patterns).
    fn silicon_lanes(c: &Capture, tile: usize, sign: usize, stage: usize) -> Vec<u64> {
        let base = ((tile * 4 + sign) * c.stages + stage) * c.out_words;
        let words = &c.silicon[base..base + c.out_words];
        if c.acc_bits == 32 {
            words.to_vec()
        } else {
            (0..c.lanes).map(|l| words[l] | (words[c.lanes + l] << 32)).collect()
        }
    }

    fn wrap(value: i128, acc_bits: u32) -> u64 {
        (value as u128 & ((1u128 << acc_bits) - 1)) as u64
    }

    /// Replay a capture against a shape: tile t of operand slot s is
    /// `inputs[s][t * lanes ..]`. Stage 0 is mul, stage 1 mac and stage 2 msc,
    /// each chained on the previous readout. Returns (lanes checked, mismatches).
    fn replay(kernel: &str, shape: MacShape) -> (usize, Vec<String>) {
        let c = load_capture(kernel);
        assert_eq!(shape.lanes() as usize, c.lanes, "{kernel}: lane count");
        let (ea, eb) = shape.operand_extent();
        assert!(ea as usize <= c.a_lanes && eb as usize <= c.b_lanes, "{kernel}: extents");
        let (mut checked, mut bad) = (0, Vec::new());
        for t in 0..c.tiles {
            for (si, &(xs, ys)) in SIGNS.iter().enumerate() {
                let mut acc = vec![0i128; c.lanes];
                for stage in 0..c.stages {
                    let a: Vec<i128> = c.inputs[2 * stage][t * c.a_lanes..(t + 1) * c.a_lanes]
                        .iter()
                        .map(|&x| extend(x, c.a_bits, xs))
                        .collect();
                    let b: Vec<i128> = c.inputs[2 * stage + 1][t * c.b_lanes..(t + 1) * c.b_lanes]
                        .iter()
                        .map(|&x| extend(x, c.b_bits, ys))
                        .collect();
                    for (lane, p) in product_lanes(shape, &a, &b).into_iter().enumerate() {
                        acc[lane] = if stage == 2 { acc[lane] - p } else { acc[lane] + p };
                    }
                    let want = silicon_lanes(&c, t, si, stage);
                    for lane in 0..c.lanes {
                        checked += 1;
                        let got = wrap(acc[lane], c.acc_bits);
                        if got != want[lane] && bad.len() < 5 {
                            bad.push(format!(
                                "tile {t} sign {si} stage {stage} lane {lane}: {got:#x} != {:#x}",
                                want[lane]
                            ));
                        } else if got != want[lane] {
                            bad.push(String::new());
                        }
                    }
                }
            }
        }
        (checked, bad)
    }

    /// The eight modes silicon was captured for: (key, stem).
    const CAPTURED: [((u32, u32, u32), &str); 8] = [
        ((0, 1, 1), "elem_32_2"),
        ((0, 1, 2), "conv_4x4_8ch"),
        ((0, 1, 3), "conv_8x8_4ch"),
        ((0, 1, 4), "conv_32x8"),
        ((0, 2, 1), "4x4_4x4_2ch"),
        ((0, 3, 1), "elem_32"),
        ((1, 3, 2), "elem_16_2"),
        ((1, 3, 3), "conv_16x4"),
    ];

    /// Every one-hot probe tile (A one-hot against B ones and B bit planes,
    /// and the mirror image), all four signedness flavours, every lane: the
    /// table's shape predicts the silicon output exactly.
    #[test]
    fn shapes_reproduce_the_silicon_probes() {
        for ((a, b, v), stem) in CAPTURED {
            let kernel = format!("vec_macmode_{stem}_probe");
            let (checked, bad) = replay(&kernel, entry(a, b, v).shape);
            assert!(checked >= 128 * 4 * 16, "{kernel}: only {checked} lanes");
            assert!(
                bad.is_empty(),
                "{kernel}: {} of {checked} lanes differ, first: {:?}",
                bad.len(),
                &bad[..bad.len().min(5)]
            );
            assert_eq!(load_capture(&kernel).stem, stem);
        }
    }

    /// The held-out captures (seeded random data, extremes, mul -> mac -> msc
    /// chains, all four signedness flavours) were not used to derive the maps.
    #[test]
    fn shapes_predict_the_held_out_random_captures() {
        for ((a, b, v), stem) in CAPTURED {
            let kernel = format!("vec_macmode_{stem}_rand");
            let (checked, bad) = replay(&kernel, entry(a, b, v).shape);
            assert!(checked >= 112 * 4 * 3 * 16, "{kernel}: only {checked} lanes");
            assert!(
                bad.is_empty(),
                "{kernel}: {} of {checked} lanes differ, first: {:?}",
                bad.len(),
                &bad[..bad.len().min(5)]
            );
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

    /// A header family the geometry extractor does not understand would drop
    /// out of `CONFIG_GEOMETRY_TABLE` silently; this makes it loud.
    #[test]
    fn every_config_mode_has_a_geometry() {
        for m in MAC_CONFIG_MODES {
            assert!(
                geometry_for(m.amode, m.bmode, m.variant).is_some(),
                "mode ({}, {}, {}) `{}` has no geometry",
                m.amode,
                m.bmode,
                m.variant,
                m.family
            );
        }
        assert_eq!(MAC_CONFIG_MODES.len(), CONFIG_GEOMETRY_TABLE.len());
    }

    #[test]
    fn each_family_name_agrees_with_its_geometry_kind() {
        for m in MAC_CONFIG_MODES {
            let g = entry(m.amode, m.bmode, m.variant);
            let is_elem = m.family.starts_with("elem_");
            let is_conv = m.family.starts_with("conv_");
            let is_multi = m.family.ends_with("ch") && !is_conv;
            assert_eq!(matches!(g.shape, MacShape::Elementwise { .. }), is_elem, "{}", m.family);
            assert_eq!(matches!(g.shape, MacShape::Conv { .. }), is_conv, "{}", m.family);
            assert_eq!(matches!(g.shape, MacShape::MultiChannel { .. }), is_multi, "{}", m.family);
            assert_eq!(
                matches!(g.shape, MacShape::Matmul { .. }),
                !(is_elem || is_conv || is_multi),
                "{}",
                m.family
            );
            assert_eq!(m.bfloat, g.bfloat, "{}", m.family);
        }
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
