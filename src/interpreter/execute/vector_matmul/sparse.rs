//! Sparse MAC: the operand decode and the integer term combine.
//!
//! A sparse MAC instruction takes A from an x register (`_sparse_narrow`) or a y register
//! (`_sparse_wide`) and B from a `qx` register: 512 bits of stored data in `x_n` and a 128-bit
//! mask in `q_n`. The stored data is the 2:4-compressed form of B; the mask says which logical
//! B units are present (rule and silicon evidence: `xdna_archspec::aie2::matmul::SparseLayout`).
//! This module expands the pair to the logical B matrix the dense multiplier reads
//! (`B[k][c]` at element index `k * cols + c`), so the multiply itself is the dense one.
//!
//! The integer result is `acc1 + acc2 + P` with the signs of [`MacOpKind::term_signs`] flipped
//! by the config word's `sub_mul` / `sub_acc1` / `sub_acc2`; `shift16` shifts acc1 left by 16
//! and `zero_acc1` drops it. The float modes keep the bf16 datapath's own combine.

use super::helpers::{read_acc_wide, write_acc_wide};
use crate::interpreter::execute::vector_config::AccWidth;
use crate::interpreter::state::{Acc1024, Vec512};
use xdna_archspec::aie2::matmul::{geometry_for, AccTerm, MacOpKind, MacTermSigns, SparseLayout};

/// A vector operand image, 1024 bits (a dense operand fills the low half).
pub(super) type Image = [u32; 32];

/// Config-word control bits the sparse integer combine reads (the TableGen `VecConf` layout:
/// dynZeroAccum, accShift, dynMulNeg, dynAcc0Neg, dynAcc1Neg; pinned by a test below).
const ZERO_ACC1: u32 = 1 << 0;
const SHIFT16: u32 = 1 << 10;
const SUB_MUL: u32 = 1 << 11;
const SUB_ACC1: u32 = 1 << 12;
const SUB_ACC2: u32 = 1 << 13;

/// What a sparse integer MAC needs besides its operands.
#[derive(Debug, Clone, Copy)]
pub(super) struct SparseCombine {
    pub signs: MacTermSigns,
    /// The raw config word (its control bits are sampled at issue).
    pub config_word: u32,
}

/// The sparse operand layout of the geometry a config word selects. A bfloat16 mode names only
/// its variant (its amode and bmode are fixed by the header).
pub(super) fn layout_of(config_word: u32, is_bf16: bool) -> Option<SparseLayout> {
    let (amode, bmode, variant) = if is_bf16 {
        (2, 3, (config_word >> 5) & 7)
    } else {
        ((config_word >> 1) & 3, (config_word >> 3) & 3, (config_word >> 5) & 7)
    };
    geometry_for(amode, bmode, variant)?.sparse_layout()
}

/// The op kind of a sparse MAC record name (`VNEGMSC_vmac_cm_core_sparse_wide`,
/// `VMUL_F_vmac_bm_core_sparse_narrow`): the stem before the first `_`.
pub(super) fn op_kind_of(encoding_name: &str) -> Option<MacOpKind> {
    let stem = encoding_name.split('_').next()?.to_ascii_lowercase();
    MacOpKind::from_stem(&stem)
}

/// Write `bits`-wide element `index` of an image.
fn set_element(image: &mut Image, index: usize, bits: u32, value: u32) {
    let per_word = (32 / bits) as usize;
    let shift = bits as usize * (index % per_word);
    let mask = ((1u64 << bits) - 1) as u32;
    let word = &mut image[index / per_word];
    *word = (*word & !(mask << shift)) | ((value & mask) << shift);
}

/// The `bits`-wide unit `index` of a 512-bit stored-data vector.
fn stored_unit_value(data: &Vec512, index: u32, bits: u32) -> u32 {
    let bit = index * bits;
    let word = data[(bit / 32) as usize];
    (word >> (bit % 32)) & (((1u64 << bits) - 1) as u32)
}

/// Expand the stored data and mask of a `qx` register to the logical `inner x cols` B image,
/// elements of `bits_y` bits, row-major.
pub(super) fn logical_b(layout: &SparseLayout, bits_y: u32, data: &Vec512, mask: u128) -> Image {
    let cols = layout.unit_cols * layout.elements_per_unit;
    let mut image = [0u32; 32];
    for k in 0..layout.inner {
        for c in 0..cols {
            let unit_col = c / layout.elements_per_unit;
            let Some(stored) = layout.stored_unit(mask, k, unit_col) else {
                continue;
            };
            let unit = stored_unit_value(data, stored, layout.unit_bits);
            // A 4-bit unit is a byte holding two neighbouring columns, low nibble first.
            let element = if layout.elements_per_unit == 2 {
                (unit >> (4 * (c % layout.elements_per_unit))) & 0xF
            } else {
                unit
            };
            set_element(&mut image, (k * cols + c) as usize, bits_y, element);
        }
    }
    image
}

/// The logical B elements the mask keeps, bit `k * cols + c` set for each: the other elements
/// are absent from the multiplier (see [`super::matmul_bf16`]), which is not the same as
/// holding a zero.
pub(super) fn present_elements(layout: &SparseLayout, mask: u128) -> u128 {
    let cols = layout.unit_cols * layout.elements_per_unit;
    let mut present = 0u128;
    for k in 0..layout.inner {
        for c in 0..cols {
            if layout.stored_unit(mask, k, c / layout.elements_per_unit).is_some() {
                present |= 1 << (k * cols + c);
            }
        }
    }
    present
}

/// Combine the product sum `product` (a zero-accumulator multiply) with the accumulator inputs
/// into the result, lane by lane, wrapping at the accumulator width.
pub(super) fn combine_integer(
    acc1: &Acc1024,
    acc2: Option<&Acc1024>,
    product: &Acc1024,
    lanes: usize,
    width: AccWidth,
    c: &SparseCombine,
) -> Acc1024 {
    let word = c.config_word;
    let flip = |bit: u32| word & bit != 0;
    let product_negated = c.signs.product_negated ^ flip(SUB_MUL);
    let acc1_negated = (c.signs.acc1 == AccTerm::Subtracted) ^ flip(SUB_ACC1);
    let acc2_negated = (c.signs.acc2 == AccTerm::Subtracted) ^ flip(SUB_ACC2);
    let use_acc1 = c.signs.acc1 != AccTerm::Absent && !flip(ZERO_ACC1);

    let mut out = [0u64; 16];
    for lane in 0..lanes {
        let mut sum = read_acc_wide(product, lane, width);
        if product_negated {
            sum = sum.wrapping_neg();
        }
        if use_acc1 {
            let mut v = read_acc_wide(acc1, lane, width);
            if flip(SHIFT16) {
                v = v.wrapping_shl(16);
            }
            sum = sum.wrapping_add(if acc1_negated { v.wrapping_neg() } else { v });
        }
        if let (true, Some(acc2)) = (c.signs.acc2 != AccTerm::Absent, acc2) {
            let v = read_acc_wide(acc2, lane, width);
            sum = sum.wrapping_add(if acc2_negated { v.wrapping_neg() } else { v });
        }
        write_acc_wide(&mut out, lane, sum, width);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use xdna_archspec::toolchain_paths::ToolchainPaths;

    /// The control-bit positions above are the `VecConf` layout of `AIE2InstrPatterns.td`
    /// (`all = {..., dynAcc1Neg, dynAcc0Neg, dynMulNeg, accShift, signX, signY, cmode, bmode,
    /// amode, dynZeroAccum}`, listed MSB first). archspec does not extract it, so this test
    /// derives each position from the TableGen source and compares. Skipped without llvm-aie.
    #[test]
    fn control_bits_match_the_tablegen_vecconf_layout() {
        let Ok(paths) = ToolchainPaths::resolve(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))) else {
            eprintln!("llvm-aie not found; skipping");
            return;
        };
        let td = paths.llvm_aie.join("llvm/lib/Target/AIE/aie2/AIE2InstrPatterns.td");
        let Ok(text) = std::fs::read_to_string(&td) else {
            eprintln!("{} unreadable; skipping", td.display());
            return;
        };
        let (class_body, _) = text
            .split_once("class VecConf")
            .map(|(_, b)| b)
            .unwrap()
            .split_once("ConfBits")
            .unwrap();
        let widths: std::collections::HashMap<&str, u32> = class_body
            .lines()
            .filter_map(|l| {
                let l = l.trim().strip_prefix("bits<")?;
                let (w, rest) = l.split_once('>')?;
                let name = rest.split_whitespace().next()?;
                Some((name, w.parse().ok()?))
            })
            .collect();
        let all = class_body
            .lines()
            .find_map(|l| l.trim().strip_prefix("bits<32> all = {"))
            .unwrap();
        let fields: Vec<&str> = all.split('}').next().unwrap().split(',').map(str::trim).collect();
        let mut bit = 0;
        let mut position = std::collections::HashMap::new();
        for name in fields.iter().rev() {
            position.insert(*name, bit);
            bit += widths[name];
        }
        assert_eq!(ZERO_ACC1, 1 << position["dynZeroAccum"]);
        assert_eq!(SHIFT16, 1 << position["accShift"]);
        assert_eq!(SUB_MUL, 1 << position["dynMulNeg"]);
        assert_eq!(SUB_ACC1, 1 << position["dynAcc0Neg"]);
        assert_eq!(SUB_ACC2, 1 << position["dynAcc1Neg"]);
    }
}
