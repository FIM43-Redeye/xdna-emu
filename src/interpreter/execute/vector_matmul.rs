//! AIE2 matrix multiply unit with proper tile geometry.
//!
//! The AIE2 vector unit contains a systolic-style multiplier array with 512
//! multiply units operating at 8-bit x 4-bit granularity. Input vectors are
//! permuted and reinterpreted as 2D matrix tiles. The tile dimensions depend
//! on element types:
//!
//! | A type | B type | Acc type | Rows | Inner | Cols | Output elements |
//! |--------|--------|----------|------|-------|------|-----------------|
//! | int8   | int8   | int32    | 4    | 8     | 8    | 32 (4x8)        |
//! | int16  | int16  | int32    | 4    | 2     | 8    | 32 (4x8)        |
//! | int16  | int16  | int64    | 4    | 4     | 4    | 16 (4x4)        |
//! | bf16   | bf16   | fp32     | 4    | 8     | 4    | 16 (4x4)        |
//! | int8   | int4   | int32    | 4    | 16    | 8    | 32 (4x8)        |
//! | int16  | int8   | int32    | 4    | 4     | 8    | 32 (4x8)        |
//!
//! The multiply is: acc[r][c] += sum(k=0..inner-1) { A[r][k] * B[k][c] }
//!
//! Hardware reference: mulmac.py (read for understanding, original implementation).

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::execute::vector_config::{AccWidth, MatMulConfig};
use crate::interpreter::state::{ExecutionContext, Vec512, Acc1024};
use crate::tablegen::SemanticOp;

// ---------------------------------------------------------------------------
// Entry point: execute_matmul
// ---------------------------------------------------------------------------

/// Execute a MAC-family instruction using config-driven matrix multiply.
///
/// This is the single entry point for ALL MAC-family instructions (Mac,
/// MatMul, MatMulSub, NegMul, NegMatMul, AddMac, SubMac). It reads the
/// config register, parses tile geometry, reads 512-bit vector inputs and
/// 1024-bit accumulator, performs the multiply, and writes back.
///
/// Returns `true` if this function handled the instruction, `false` if the
/// semantic is not a MAC-family operation (caller should use fallback).
pub fn execute_matmul(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    // Only handle MAC-family semantics.
    let semantic = match op.semantic {
        Some(
            s @ (SemanticOp::Mac
            | SemanticOp::MatMul
            | SemanticOp::MatMulSub
            | SemanticOp::NegMul
            | SemanticOp::NegMatMul
            | SemanticOp::AddMac
            | SemanticOp::SubMac),
        ) => s,
        _ => return false,
    };

    // Read config register (last ScalarReg in sources).
    let conf_val = match get_config_reg(op, ctx) {
        Some(v) => v,
        None => {
            log::error!(
                "[MATMUL] no config register found in sources for {:?}",
                op.encoding_name
            );
            return true; // Handled (as error), don't fall through.
        }
    };

    // Detect bf16 mode from element_type (set at build time from mnemonic).
    // BFloat16/Float32 MAC variants use a different tile geometry and
    // accumulator format than integer variants.
    let is_bf16 = matches!(op.element_type, Some(ElementType::BFloat16 | ElementType::Float32));

    // Parse config word into tile geometry and modes.
    let mut config = match MatMulConfig::from_config_word(conf_val, is_bf16) {
        Some(c) => c,
        None => {
            log::error!(
                "[MATMUL] failed to parse config word 0x{:08x} (bf16={}) for {:?}",
                conf_val,
                is_bf16,
                op.encoding_name
            );
            return true;
        }
    };

    // Handle subtract semantics for MAC variants.
    //
    // The hardware compressor stage controls the sign of the product via a
    // subtract flag.  Several instruction families flip this flag:
    //
    //   vmac:     result = acc + A*B        (subtract=false)
    //   vmsc:     result = acc - A*B        (subtract=true, negates product)
    //   vnegmac:  result = -(acc) + A*B     (NegMul semantic)
    //   vnegmsc:  result = -(acc) - A*B     (NegMul + MSC)
    //   vaddmac:  result = (acc + A*B) + acc2
    //   vaddmsc:  result = (acc - A*B) + acc2  (MSC: product negated)
    //   vsubmac:  result = (acc + A*B) - acc2
    //   vsubmsc:  result = (acc - A*B) - acc2  (MSC: product negated)
    //
    // The NegMul/NegMatMul/MatMulSub semantics flip subtract for vnegmac.
    // MSC (multiply-subtract) variants have the subtract flag baked into
    // the instruction encoding (not in the config word).  Detect MSC from
    // the encoding mnemonic and flip subtract accordingly.
    match semantic {
        SemanticOp::NegMul | SemanticOp::NegMatMul | SemanticOp::MatMulSub => {
            config.subtract = !config.subtract;
        }
        _ => {}
    }

    // MSC variants (vaddmsc, vsubmsc) with AddMac/SubMac semantics need
    // the product negation applied.  VMSC and VNEGMSC already get the
    // product negation via their NegMul/NegMatMul semantics above, so
    // only apply this for AddMac/SubMac.
    if matches!(semantic, SemanticOp::AddMac | SemanticOp::SubMac) {
        if let Some(ref name) = op.encoding_name {
            if name.contains("msc") {
                config.subtract = !config.subtract;
            }
        }
    }

    // Detect sparse mode from operand types.
    //
    // Sparse MAC instructions use different operand types than dense:
    //   - Dense: two VectorReg sources (x registers)
    //   - Sparse wide: VectorReg (ys1 = 1024-bit) + SparseQxReg (qxs2)
    //   - Sparse narrow: VectorReg (xs1 = 512-bit) + SparseQxReg (qxs2)
    //
    // The qxs2 operand is a composite register: qx_n = {x_n (data), q_n (mask)}.
    // SparseQxReg(n) maps to qx_n. The presence of a SparseQxReg source
    // is the structural signal that distinguishes sparse from dense.
    let is_sparse = op.sources.iter().any(|s| matches!(s, Operand::SparseQxReg(_)));

    // Determine accumulator access mode from destination operand.
    // cm (1024-bit): AccumReg is always even, read/write as wide pair.
    // bm (512-bit):  AccumReg can be even (bml) or odd (bmh), single register.
    //
    // bm_core instructions produce 512-bit output (e.g., bf16 4x8x4 = 16 fp32
    // = 64 bytes). cm_core instructions produce 1024-bit output (e.g., int8
    // 4x8x8 = 32 int32 = 128 bytes). The geometry determines how many output
    // elements fill the accumulator; the register index determines where they
    // go in the physical register file.
    let (acc_reg, is_half) = get_acc_dest(op);
    let mut acc = if is_half {
        // bm (512-bit): read single register, pad to 1024-bit working buffer.
        let half = ctx.accumulator.read(acc_reg);
        let mut buf = [0u64; 16];
        buf[..8].copy_from_slice(&half);
        buf
    } else {
        // cm (1024-bit): read wide pair.
        ctx.accumulator.read_wide(acc_reg)
    };

    // Read input vectors and perform multiply.
    if is_sparse {
        let (a_bytes, b_register, mask) = get_sparse_operands(op, ctx);
        matmul_sparse_config_driven(&mut acc, &a_bytes, &b_register, mask, &config);
    } else {
        let (a, b) = get_two_vec512(op, ctx);
        matmul_config_driven(&mut acc, &a, &b, &config);
    }

    // For AddMac/SubMac: merge a second accumulator source AFTER the multiply.
    // This must happen after matmul_config_driven because that function zeros
    // the accumulator when zero_acc=1. The hardware computes:
    //   AddMac: result = (acc_dest [or 0] + A*B) + acc2
    //   SubMac: result = (acc_dest [or 0] + A*B) - acc2
    //
    // In Acc32 mode, the hardware performs the acc2 merge as independent
    // 32-bit lane additions (no carry propagation from the low half to the
    // high half within each u64 word). In Acc64 mode, it's a full 64-bit
    // addition on each u64 lane.
    match semantic {
        SemanticOp::AddMac | SemanticOp::SubMac => {
            let src_reg = get_acc_source(op);
            let src_acc = if is_half {
                let half = ctx.accumulator.read(src_reg);
                let mut buf = [0u64; 16];
                buf[..8].copy_from_slice(&half);
                buf
            } else {
                ctx.accumulator.read_wide(src_reg & !1)
            };
            let n = if is_half { 8 } else { 16 };
            let is_sub = semantic == SemanticOp::SubMac;

            if config.bfloat {
                // BFloat16 mode: accumulator holds fp32 values, two per u64.
                // Use AIE2 fp32 add (FTZ + NaN canonicalization) for the acc2
                // merge, matching the hardware's accumulator ALU behavior.
                use super::vector_float::aie2_fp32_add;
                for i in 0..n {
                    let a_lo = (acc[i] & 0xFFFF_FFFF) as u32;
                    let a_hi = (acc[i] >> 32) as u32;
                    let mut s_lo = (src_acc[i] & 0xFFFF_FFFF) as u32;
                    let mut s_hi = (src_acc[i] >> 32) as u32;
                    if is_sub {
                        s_lo ^= 0x8000_0000;
                        s_hi ^= 0x8000_0000;
                    }
                    let r_lo = aie2_fp32_add(a_lo, s_lo);
                    let r_hi = aie2_fp32_add(a_hi, s_hi);
                    acc[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            } else {
                match config.acc_width {
                    AccWidth::Acc32 => {
                        // Acc32: two i32 values per u64, added independently.
                        for i in 0..n {
                            let a_lo = (acc[i] & 0xFFFF_FFFF) as u32;
                            let a_hi = (acc[i] >> 32) as u32;
                            let s_lo = (src_acc[i] & 0xFFFF_FFFF) as u32;
                            let s_hi = (src_acc[i] >> 32) as u32;
                            let (r_lo, r_hi) = if is_sub {
                                (a_lo.wrapping_sub(s_lo), a_hi.wrapping_sub(s_hi))
                            } else {
                                (a_lo.wrapping_add(s_lo), a_hi.wrapping_add(s_hi))
                            };
                            acc[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                        }
                    }
                    AccWidth::Acc64 => {
                        // Acc64: full 64-bit add/sub on each u64 lane.
                        for i in 0..n {
                            if is_sub {
                                acc[i] = acc[i].wrapping_sub(src_acc[i]);
                            } else {
                                acc[i] = acc[i].wrapping_add(src_acc[i]);
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    // Write result back to the accumulator.
    if is_half {
        // bm (512-bit): write back single register only.
        let mut half = [0u64; 8];
        half.copy_from_slice(&acc[..8]);
        ctx.accumulator.write(acc_reg, half);
    } else {
        // cm (1024-bit): write wide pair.
        ctx.accumulator.write_wide(acc_reg, acc);
    }

    true
}

// ---------------------------------------------------------------------------
// Private helpers for execute_matmul
// ---------------------------------------------------------------------------

/// Scan sources for the last ScalarReg operand (the config register).
fn get_config_reg(op: &SlotOp, ctx: &ExecutionContext) -> Option<u32> {
    for src in op.sources.iter().rev() {
        if let Operand::ScalarReg(r) = src {
            return Some(ctx.scalar.read(*r));
        }
    }
    None
}

/// Read two 512-bit vectors from the VectorReg operands.
///
/// MAC instructions have two VectorReg sources that each name an x-register
/// (already decoded as even indices: x0->0, x2->2, etc.). We read the full
/// 512-bit value via `read_wide`.
fn get_two_vec512(op: &SlotOp, ctx: &ExecutionContext) -> (Vec512, Vec512) {
    let mut vregs = op
        .sources
        .iter()
        .filter_map(|s| {
            if let Operand::VectorReg(r) = s {
                Some(*r)
            } else {
                None
            }
        });

    let a_reg = vregs.next().unwrap_or_else(|| {
        log::error!("[MATMUL] missing first VectorReg source");
        0
    });
    let b_reg = vregs.next().unwrap_or_else(|| {
        log::error!("[MATMUL] missing second VectorReg source");
        0
    });

    // Ensure even alignment for wide read.
    let a_base = a_reg & !1;
    let b_base = b_reg & !1;

    (ctx.vector.read_wide(a_base), ctx.vector.read_wide(b_base))
}

/// Extract the AccumReg from the destination operand.
///
/// Returns (register_index, is_half_width):
/// - cm destinations (1024-bit): even index, is_half=false
/// - bml destinations (512-bit low): even index, is_half=true
/// - bmh destinations (512-bit high): odd index, is_half=true
///
/// Determine accumulator destination register and whether it's a narrow
/// (bm, 512-bit) or wide (cm, 1024-bit) access.
///
/// Uses AccumWidth from the decoder's register class metadata:
///   Half (bml/bmh) -> 512-bit bm_core (is_half = true)
///   Full (cm)       -> 1024-bit cm_core (is_half = false)
///   None            -> infer from element type (bf16/float -> half, integer -> wide)
fn get_acc_dest(op: &SlotOp) -> (u8, bool) {
    let is_half = match op.accum_width {
        Some(crate::tablegen::decoder_ffi::AccumWidth::Half)
        | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterLow)
        | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterHigh) => true,
        Some(crate::tablegen::decoder_ffi::AccumWidth::Full) => false,
        None => {
            // Legacy fallback: bf16/float -> bm_core, integer -> cm_core.
            matches!(op.element_type, Some(ElementType::BFloat16 | ElementType::Float32))
        }
    };

    match &op.dest {
        Some(Operand::AccumReg(r)) => {
            if is_half {
                // bm_core: use register index as-is (even=bml, odd=bmh).
                (*r, true)
            } else {
                // cm_core: force even alignment for wide pair access.
                (*r & !1, false)
            }
        }
        other => {
            log::error!(
                "[MATMUL] expected AccumReg dest, got {:?} -- defaulting to cm0",
                other
            );
            (0, false)
        }
    }
}

/// Extract the second AccumReg from sources (acc2 for AddMac/SubMac).
///
/// VADDMAC/VSUBMAC have three accumulator operands: dest, acc1 (=dest),
/// and acc2 (the merge source). In our operand list, acc1 appears first
/// and acc2 second. We skip the first AccumReg (acc1, same as dest) and
/// return the second one (acc2).
fn get_acc_source(op: &SlotOp) -> u8 {
    let mut seen_first = false;
    for src in &op.sources {
        if let Operand::AccumReg(r) = src {
            if seen_first {
                return *r;
            }
            seen_first = true;
        }
    }
    // If only one AccumReg in sources, acc2 = acc1 (same register).
    for src in &op.sources {
        if let Operand::AccumReg(r) = src {
            return *r;
        }
    }
    log::error!(
        "[MATMUL] no AccumReg found in sources -- defaulting to cm0"
    );
    0
}

/// Read operands for a sparse MAC instruction.
///
/// Sparse MAC instructions have different operand types from dense:
/// - A (dense input): VectorReg -- 512-bit (xs1) or 1024-bit (ys1)
/// - B (sparse input): SparseQxReg(n) -- composite qx_n = {x_n, q_n}
///
/// Returns:
/// - `a_bytes`: A operand as 128-byte buffer (zero-padded if A < 128 bytes)
/// - `b_compressed`: Raw 64 bytes from the B x-register (before decompression)
/// - `mask`: Full 128-bit mask from the q register
fn get_sparse_operands(op: &SlotOp, ctx: &ExecutionContext) -> ([u8; 128], [u8; 64], u128) {
    let mut a_bytes = [0u8; 128];
    let mut b_compressed = [0u8; 64];
    let mut mask: u128 = 0;
    let mut found_a = false;

    for src in &op.sources {
        match src {
            Operand::VectorReg(r) => {
                if !found_a {
                    found_a = true;
                    if op.is_quad_vector {
                        // y-register (1024-bit): full 128 bytes for sparse wide.
                        let base = *r & !3;
                        let quad = ctx.vector.read_quad(base);
                        a_bytes = vec1024_to_bytes(&quad);
                    } else {
                        // x-register (512-bit): 64 bytes, zero-padded to 128.
                        let base = *r & !1;
                        let wide = ctx.vector.read_wide(base);
                        let narrow = vec512_to_bytes(&wide);
                        a_bytes[..64].copy_from_slice(&narrow);
                    }
                }
            }
            Operand::SparseQxReg(qx_idx) => {
                // qx_n = {x_n (vector data), q_n (sparsity mask)}
                let x_base = (*qx_idx) * 2;
                let wide = ctx.vector.read_wide(x_base);
                b_compressed = vec512_to_bytes(&wide);
                mask = ctx.mask.read_u128(*qx_idx);
            }
            _ => {}
        }
    }

    if !found_a {
        log::error!("[MATMUL-SPARSE] missing VectorReg source for A operand");
    }

    (a_bytes, b_compressed, mask)
}

/// Decompress sparse B data using pair-routing.
///
/// The hardware treats the 512-bit x register as 32 byte-pairs. For each
/// group of 4 mask bits, the pair `(compressed[2*g], compressed[2*g+1])` is
/// routed to the set-bit positions within the group's 4-byte output span.
///
/// The output is **column-major**: groups traverse inner-dimension positions
/// first, then columns. For i8xi8 sparse (inner=16, cols=8):
///   - Groups 0-3 → column 0 (inner positions 0-15, 4 bytes each)
///   - Groups 4-7 → column 1
///   - Groups 28-31 → column 7
///
/// Within each group:
///   - `compressed[2*g+1]` (lo) -> lowest set-bit position in the group
///   - `compressed[2*g]`   (hi) -> highest set-bit position in the group
///   - Clear-bit positions -> 0
///
/// Convert a Vec512 ([u32; 16], 64 bytes) to a byte array in little-endian order.
fn vec512_to_bytes(v: &Vec512) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for (i, word) in v.iter().enumerate() {
        let base = i * 4;
        bytes[base] = *word as u8;
        bytes[base + 1] = (*word >> 8) as u8;
        bytes[base + 2] = (*word >> 16) as u8;
        bytes[base + 3] = (*word >> 24) as u8;
    }
    bytes
}

/// Convert a quad vector ([u32; 32], 128 bytes) to a byte array.
fn vec1024_to_bytes(v: &[u32; 32]) -> [u8; 128] {
    let mut bytes = [0u8; 128];
    for (i, word) in v.iter().enumerate() {
        let base = i * 4;
        bytes[base] = *word as u8;
        bytes[base + 1] = (*word >> 8) as u8;
        bytes[base + 2] = (*word >> 16) as u8;
        bytes[base + 3] = (*word >> 24) as u8;
    }
    bytes
}

/// Extract an element from a 128-byte buffer.
///
/// For 4-bit elements, `byte_idx` is the element index (two elements per
/// byte: element 0 = low nibble, element 1 = high nibble). For 8/16/32-bit
/// elements, `byte_idx` is the byte offset.
///
/// Returns 0 for out-of-bounds accesses.
fn extract_element_bytes(src: &[u8; 128], byte_idx: usize, bits: u32, signed: bool) -> i64 {
    match bits {
        4 => {
            let byte_pos = byte_idx / 2;
            let nibble = byte_idx % 2;
            if byte_pos >= 128 { return 0; }
            let raw = src[byte_pos];
            let val = if nibble == 0 { raw & 0xF } else { (raw >> 4) & 0xF };
            if signed && (val & 0x8) != 0 {
                (val as i8 | !0x0Fi8) as i64
            } else {
                val as i64
            }
        }
        8 => {
            if byte_idx >= 128 { return 0; }
            let val = src[byte_idx];
            if signed { val as i8 as i64 } else { val as i64 }
        }
        16 => {
            if byte_idx + 1 >= 128 { return 0; }
            let val = u16::from_le_bytes([src[byte_idx], src[byte_idx + 1]]);
            if signed { val as i16 as i64 } else { val as i64 }
        }
        32 => {
            if byte_idx + 3 >= 128 { return 0; }
            let val = u32::from_le_bytes([
                src[byte_idx], src[byte_idx + 1],
                src[byte_idx + 2], src[byte_idx + 3],
            ]);
            if signed { val as i32 as i64 } else { val as i64 }
        }
        _ => 0,
    }
}

/// Sparse config-driven matrix multiply.
///
/// `a` is a 128-byte buffer (1024 bits) containing the DENSE operand (xs1).
/// For narrow sparse, only the first 64 bytes are populated; the rest are zero.
/// Layout: `A[r, k] = a[(r * inner + k) * bytes_x]` (row-major i16 or i8).
///
/// `b` is a 64-byte buffer (512 bits) containing COMPRESSED sparse data from
/// the qxs2 register. This is NOT decompressed -- it holds 32 byte-pairs
/// where each pair represents the 2 active elements within a group of 4.
/// The pair at positions `(b[2*g], b[2*g+1])` corresponds to mask group `g`.
///
/// `mask` is the 128-bit sparsity mask. It has 32 groups of 4 bits (nibbles).
/// Each nibble selects which 2 of 4 sparse inner positions are active.
/// Groups are column-major: `g = col * inner_groups + ig`.
///
/// The function decompresses B via cleanroom crossbar routing before multiplying.
/// After decompression, the 128-byte array is indexed as:
///   `b_dec[4*g + bit_pos] = b_dec[c * inner + sparse_k]`
/// which follows directly from `g = c * inner_groups + ig` and
/// `sparse_k = ig * 4 + bit_pos`.
///
/// Hardware reference: The broadcast stage (prmx_bcst_hw) is a no-op
/// (type reinterpretation only), confirmed from me_inline_primitives.h
/// line 11784. The crossbar output byte positions directly correspond to
/// multiplier lane positions.
pub fn matmul_sparse_config_driven(
    acc: &mut Acc1024,
    a: &[u8; 128],
    b: &[u8; 64],
    mask: u128,
    config: &MatMulConfig,
) {
    let rows = config.rows as usize;
    let cols = config.cols as usize;
    let bits_x = config.bits_x;
    let bits_y = config.bits_y;
    let bytes_x = if bits_x == 4 { 1 } else { (bits_x / 8) as usize };

    // Total number of mask groups. Each group covers 4 positions with 2 active
    // (2:4 sparsity). The 128-bit mask has 4 bits per group.
    let num_groups = 64 / 2; // 32 groups, each consuming 2 compressed bytes

    // Pad compressed B to 128 bytes for extract_element_bytes compatibility.
    let mut b_pad = [0u8; 128];
    b_pad[..64].copy_from_slice(b);

    if !config.accumulate {
        *acc = [0u64; 16];
    }

    // Sparse crossbar routing, derived from real NPU hardware observation:
    //
    //   output_column = compressed_byte_index % cols
    //   inner_k = (group / 8) * 4 + bit_positions[(group / 4) % 2]
    //
    // where bit_positions are the two set bit indices from the mask nibble.
    // Groups alternate between the two set bits in 4-group blocks.
    //
    // Column routing is mask-independent: each compressed byte routes to
    // a column based purely on its position in the 64-byte buffer.
    //
    // Cleanroom: NPU crossbar sweep characterization (2026-04-02).

    // Extract the two set bit positions from the first mask nibble.
    let representative_nibble = (mask & 0xF) as u8;
    let bit_positions: [usize; 2] = {
        let mut bits = [0usize; 2];
        let mut idx = 0;
        for b in 0..4u8 {
            if (representative_nibble >> b) & 1 != 0 && idx < 2 {
                bits[idx] = b as usize;
                idx += 1;
            }
        }
        if idx == 1 { bits[1] = bits[0]; }
        bits
    };

    if config.bfloat {
        // Bf16 sparse: each compressed B element is 2 bytes (little-endian).
        // Each mask group has 4 bf16 positions, 2 active. Each active position
        // consumes 2 consecutive compressed bytes.
        // TODO: needs characterization for bf16 sparse routing.
        // For now, use same column routing with 2-byte elements.
        for g in 0..num_groups {
            let mask4 = ((mask >> (4 * g)) & 0xF) as u8;
            if mask4 == 0 || mask4.count_ones() > 2 {
                continue;
            }

            let mut comp_idx = 0usize; // which compressed element within group (0 or 1)
            for bit in 0..4u8 {
                if (mask4 >> bit) & 1 == 0 || comp_idx >= 2 {
                    continue;
                }

                let b_byte_pos = g * 2 + comp_idx;
                let col = b_byte_pos % cols;
                // Mask-dependent inner_k (crossbar sweep 2026-04-02).
                let bit_select = (g / 4) % 2;
                let inner_k = (g / 8) * 4 + bit_positions[bit_select];

                // Read 2-byte bf16 from compressed B.
                let b_off = b_byte_pos * 2;
                if b_off + 1 >= 64 {
                    comp_idx += 1;
                    continue;
                }
                let b_bits = u16::from_le_bytes([b[b_off], b[b_off + 1]]);
                let b_val = f32::from_bits((b_bits as u32) << 16);

                for r in 0..rows {
                    let a_off = (r * config.inner as usize + inner_k) * 2;
                    if a_off + 1 >= 128 {
                        continue;
                    }
                    let a_bits = u16::from_le_bytes([a[a_off], a[a_off + 1]]);
                    let a_val = f32::from_bits((a_bits as u32) << 16);

                    let out_idx = r * cols + col;
                    let prev = read_acc_wide_f32(acc, out_idx);
                    let product = a_val * b_val;
                    if config.subtract {
                        write_acc_wide_f32(acc, out_idx, prev - product);
                    } else {
                        write_acc_wide_f32(acc, out_idx, prev + product);
                    }
                }

                comp_idx += 1;
            }
        }
        return;
    }

    // Integer sparse path.
    //
    // Iterate over all 32 mask groups. For each group, check which of the 4
    // positions are active (set bits in the mask nibble). For each active
    // position, read the corresponding compressed B byte and route it to the
    // correct output column.
    //
    // inner_k routing (cleanroom, NPU crossbar sweep 2026-04-02):
    //
    // The mask nibble's set bit positions determine the inner_k value.
    // Groups are processed in blocks of 4, alternating between the two
    // set bits of the mask nibble:
    //
    //   bit_positions = [b0, b1]  (the two set bits, sorted ascending)
    //   bit_select = (g / 4) % 2  (alternates per 4-group block)
    //   inner_k = (g / 8) * 4 + bit_positions[bit_select]
    //
    // Example: mask 0x3 (bits 0,1) → inner_k sequence: 0,1,4,5,8,9,12,13
    //          mask 0x5 (bits 0,2) → inner_k sequence: 0,2,4,6,8,10,12,14
    //          mask 0xC (bits 2,3) → inner_k sequence: 2,3,6,7,10,11,14,15
    //
    // Both compressed bytes within a group share the same inner_k.
    // The column routing (col = b_byte_pos % cols) is mask-independent.

    // Extract the two set bit positions from mask nibble 0 as representative.
    // All nibbles in the mask should use the same pattern (the 2:4 sparsity
    // pattern is uniform across groups in hardware-generated masks).
    // Fall back to [0, 1] if the mask doesn't have exactly 2 bits set.
    let representative_nibble = (mask & 0xF) as u8;
    let bit_positions: [usize; 2] = {
        let mut bits = [0usize; 2];
        let mut idx = 0;
        for b in 0..4u8 {
            if (representative_nibble >> b) & 1 != 0 && idx < 2 {
                bits[idx] = b as usize;
                idx += 1;
            }
        }
        // If mask nibble doesn't have exactly 2 bits, use first-seen or default.
        if idx < 2 {
            // Single-bit or zero-bit mask: replicate the found bit.
            if idx == 1 { bits[1] = bits[0]; }
            // idx == 0: both stay 0, which is fine (no active groups anyway).
        }
        bits
    };

    for g in 0..num_groups {
        let mask4 = ((mask >> (4 * g)) & 0xF) as u8;
        if mask4 == 0 {
            continue;
        }

        // Track which compressed byte within this group we're reading (0 or 1).
        let mut comp_idx = 0usize;

        for bit in 0..4u8 {
            if (mask4 >> bit) & 1 == 0 || comp_idx >= 2 {
                continue;
            }

            // Compressed byte position in the 64-byte buffer.
            let b_byte_pos = g * 2 + comp_idx;

            // Output column: cleanroom routing from NPU observation.
            let col = b_byte_pos % cols;

            // Inner dimension index: depends on group position AND mask
            // bit positions. Groups alternate between the two set bits
            // in 4-group blocks.
            let bit_select = (g / 4) % 2;
            let inner_k = (g / 8) * 4 + bit_positions[bit_select];

            // Read B element from compressed buffer.
            let b_val = extract_element_bytes(&b_pad, b_byte_pos, bits_y, config.y_signed);

            // Accumulate product for each row.
            for r in 0..rows {
                let a_byte = if bits_x == 4 {
                    r * config.inner as usize + inner_k
                } else {
                    (r * config.inner as usize + inner_k) * bytes_x
                };

                let a_val = extract_element_bytes(a, a_byte, bits_x, config.x_signed);

                let out_idx = r * cols + col;
                let prev = read_acc_wide(acc, out_idx, config.acc_width);
                let product = a_val * b_val;
                if config.subtract {
                    write_acc_wide(acc, out_idx, prev - product, config.acc_width);
                } else {
                    write_acc_wide(acc, out_idx, prev + product, config.acc_width);
                }
            }

            comp_idx += 1;
        }
    }
}

/// Tile geometry for a matrix multiply mode.
#[derive(Debug, Clone, Copy)]
struct TileGeometry {
    rows: usize,
    inner: usize,
    cols: usize,
}

/// Extract int8 elements from packed [u32; 8] (256 bits = 32 bytes = 32 int8 values).
/// Elements are in little-endian byte order within each u32.
fn extract_i8(packed: &[u32; 8], index: usize) -> i8 {
    let word = index / 4;
    let byte = index % 4;
    ((packed[word] >> (byte * 8)) & 0xFF) as u8 as i8
}

/// Extract uint8 elements from packed [u32; 8].
fn extract_u8(packed: &[u32; 8], index: usize) -> u8 {
    let word = index / 4;
    let byte = index % 4;
    ((packed[word] >> (byte * 8)) & 0xFF) as u8
}

/// Extract int16 elements from packed [u32; 8] (256 bits = 16 int16 values).
fn extract_i16(packed: &[u32; 8], index: usize) -> i16 {
    let word = index / 2;
    let half = index % 2;
    ((packed[word] >> (half * 16)) & 0xFFFF) as u16 as i16
}

/// Extract uint16 elements from packed [u32; 8].
fn extract_u16(packed: &[u32; 8], index: usize) -> u16 {
    let word = index / 2;
    let half = index % 2;
    ((packed[word] >> (half * 16)) & 0xFFFF) as u16
}

/// Extract bf16 as f32 from packed [u32; 8] (256 bits = 16 bf16 values).
fn extract_bf16_as_f32(packed: &[u32; 8], index: usize) -> f32 {
    let word = index / 2;
    let half = index % 2;
    let bits = ((packed[word] >> (half * 16)) & 0xFFFF) as u16;
    f32::from_bits((bits as u32) << 16)
}

/// Extract int32 elements from packed [u32; 8] (256 bits = 8 int32 values).
fn extract_i32(packed: &[u32; 8], index: usize) -> i32 {
    packed[index] as i32
}

/// Read a 32-bit accumulator lane from the [u64; 8] accumulator.
///
/// The 8 u64 lanes hold 16 int32 values (acc_cmb=1 mode, 32-bit accumulator).
/// Lane layout: acc[0] holds output[0] in low 32 bits, output[1] in high 32 bits, etc.
fn read_acc32(acc: &[u64; 8], index: usize) -> i64 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    bits as i32 as i64
}

/// Write a 32-bit accumulator lane into the [u64; 8] accumulator.
fn write_acc32(acc: &mut [u64; 8], index: usize, value: i64) {
    let u64_lane = index / 2;
    let half = index % 2;
    let masked = (value as u32) as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (masked << shift);
}

/// Read a 64-bit accumulator lane (acc_cmb=2 mode).
fn read_acc64(acc: &[u64; 8], index: usize) -> i64 {
    acc[index] as i64
}

/// Write a 64-bit accumulator lane (acc_cmb=2 mode).
fn write_acc64(acc: &mut [u64; 8], index: usize, value: i64) {
    acc[index] = value as u64;
}

/// Read a float32 accumulator lane from [u64; 8].
///
/// For bf16 matmul, the accumulator holds fp32 values. Since we have 16 output
/// elements (4x4) and 8 u64 lanes, each u64 holds two fp32 values.
fn read_acc_f32(acc: &[u64; 8], index: usize) -> f32 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    f32::from_bits(bits)
}

/// Write a float32 accumulator lane.
fn write_acc_f32(acc: &mut [u64; 8], index: usize, value: f32) {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = value.to_bits() as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (bits << shift);
}

/// Dense matrix multiply: acc += A * B (or acc = A * B if clear_acc is true).
///
/// Performs a tiled matrix multiply based on the element type. The input vectors
/// are reinterpreted as 2D tiles and multiplied using the geometry appropriate
/// for the element type combination.
pub fn matmul_dense(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    elem_type: ElementType,
    signed_a: bool,
    signed_b: bool,
) {
    match elem_type {
        ElementType::Int8 => matmul_i8xi8(acc, a, b, true, true, false),
        ElementType::UInt8 => matmul_i8xi8(acc, a, b, signed_a, signed_b, false),
        ElementType::Int16 => matmul_i16xi16_32(acc, a, b, true, true, false),
        ElementType::UInt16 => matmul_i16xi16_32(acc, a, b, signed_a, signed_b, false),
        ElementType::BFloat16 => matmul_bf16xbf16(acc, a, b, false),
        ElementType::Int32 => matmul_i32xi16(acc, a, b, true, true, false),
        ElementType::UInt32 => matmul_i32xi16(acc, a, b, false, false, false),
        ElementType::Int64 | ElementType::UInt64 => matmul_i32xi16(acc, a, b, signed_a, signed_b, false),
        ElementType::Float32 => matmul_bf16xbf16(acc, a, b, false),
    }
}

/// Matrix multiply-subtract: acc -= A * B.
pub fn matmul_sub(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    elem_type: ElementType,
    signed_a: bool,
    signed_b: bool,
) {
    match elem_type {
        ElementType::Int8 => matmul_i8xi8(acc, a, b, true, true, true),
        ElementType::UInt8 => matmul_i8xi8(acc, a, b, signed_a, signed_b, true),
        ElementType::Int16 => matmul_i16xi16_32(acc, a, b, true, true, true),
        ElementType::UInt16 => matmul_i16xi16_32(acc, a, b, signed_a, signed_b, true),
        ElementType::BFloat16 => matmul_bf16xbf16(acc, a, b, true),
        ElementType::Int32 => matmul_i32xi16(acc, a, b, true, true, true),
        ElementType::UInt32 => matmul_i32xi16(acc, a, b, false, false, true),
        ElementType::Int64 | ElementType::UInt64 => matmul_i32xi16(acc, a, b, signed_a, signed_b, true),
        ElementType::Float32 => matmul_bf16xbf16(acc, a, b, true),
    }
}

// ---------------------------------------------------------------------------
// Hardware-accurate bf16 MAC (29-bit accumulator model)
// ---------------------------------------------------------------------------
//
// Exact translation of `bf16_mac_hw` from the aietools python_model
// (bfloat16.py lines 280-399).  This matches the AIE2 hardware MAC
// pipeline's fixed-point accumulation with 29-bit internal precision.
//
// Pipeline stages:
//   1. Split bf16 inputs, detect NaN/Inf, compute product exponents
//   2. Integer multiply (8-bit mantissa * 8-bit mantissa = 16-bit product)
//   3. Align products to max exponent, round to iprec=23 bits (RNE)
//   4. Sum aligned products in integer arithmetic
//   5. Merge product sum with previous fp32 accumulator at aprec=29 bits
//   6. Normalize back to fp32 with RNE rounding, FTZ on underflow

/// Shift-and-round with RNE (round-to-nearest-even).
///
/// Shifts `man` right so that the leading bit ends up at position `prec`.
/// `lo_pos` is the current leading bit position.  Returns (rounded_value,
/// exponent_increment_from_overflow).
///
/// Direct translation of `shift_round_rne` from bfloat16.py.
fn shift_round_rne(man: i64, lo_pos: i32, prec: i32, norm_round_overflow: bool) -> (i64, bool) {
    let sh_dn = lo_pos - prec - 1;

    // Guard against shifts exceeding i64 width.
    if sh_dn >= 63 {
        // Shifted completely out -- rounds to zero.
        return (0, false);
    }

    let rmask: i64 = if sh_dn > 0 && sh_dn < 63 { (1i64 << sh_dn) - 1 } else { 0 };

    let r: i64 = if sh_dn < 0 {
        let lsh = (-sh_dn) as u32;
        if lsh >= 63 { return (0, false); } // Would overflow
        man.wrapping_shl(lsh)
    } else {
        man >> sh_dn
    };
    let q = r.wrapping_add(2); // Pre-compute round-up value

    let grd = (r & 1) != 0;
    let lsb = (r & 2) != 0;
    let stk = (man & rmask) != 0;

    // RNE: round up when guard=1 AND (sticky=1 OR lsb=1)
    let rup = grd && (stk || lsb);

    let mut expincr = false;

    let overflow_threshold = if (prec + 2) < 63 { 1i64 << (prec + 2) } else { i64::MAX };
    if norm_round_overflow && rup && (q >= overflow_threshold) {
        let val = q >> 2;
        expincr = true;
        return (val, expincr);
    }

    let val = if rup { q >> 1 } else { r >> 1 };
    (val, expincr)
}

/// Find the position of the leading one bit (0-indexed from LSB).
/// Returns -1 for input 0.
fn flo(man: i64) -> i32 {
    if man <= 0 { return -1; }
    63 - man.leading_zeros() as i32
}

/// Find the leading bit position for a signed value.
/// For non-negative: same as flo.  For negative: flo(bitwise complement).
fn flb(man: i64) -> i32 {
    if man >= 0 { flo(man) } else { flo(!man) }
}

/// Truncate to `bits` width with sign extension.
fn trnc(a: i64, sgn: bool, bits: u32) -> i64 {
    if bits == 0 { return 0; }
    let mask = if bits >= 64 { !0i64 as u64 } else { (1u64 << bits) - 1 };
    let v = (a as u64) & mask;
    let s = (v >> (bits - 1)) & 1;
    if s == 1 && sgn {
        // Sign-extend: set all upper bits
        (v | !mask) as i64
    } else {
        v as i64
    }
}

/// Hardware-accurate bf16 MAC for one output lane.
///
/// Takes the previous accumulator value `q` (fp32 bits), paired bf16
/// input elements `a_elems` and `b_elems`, and whether to subtract.
/// Returns the new fp32 accumulator value (as bits).
///
/// Direct translation of `bf16_mac_hw` from bfloat16.py.
fn bf16_mac_hw_lane(q: u32, a_elems: &[u16], b_elems: &[u16], subtract: bool) -> u32 {
    use super::vector_float::{bf16_split, fp32_split, fp32_make};

    let iprec: i32 = 23; // Internal precision for product alignment
    let aprec: i32 = 29; // Accumulator precision for merge

    let n = a_elems.len();
    debug_assert_eq!(n, b_elems.len());

    // Phase 1: Split inputs, detect NaN/Inf, compute product exponents.
    let mut nan_flag = false;
    let mut inf_flag = false;
    let mut inf_sgn = false;
    let mut csgnl = Vec::with_capacity(n);
    let mut cexpl = Vec::with_capacity(n);
    let mut amanl = Vec::with_capacity(n);
    let mut bmanl = Vec::with_capacity(n);

    for i in 0..n {
        let (asgn, aexp, aman) = bf16_split(a_elems[i]);
        let (bsgn, bexp, bman) = bf16_split(b_elems[i]);

        let csgn = asgn != bsgn; // Product sign = XOR
        csgnl.push(csgn);
        amanl.push(aman);
        bmanl.push(bman);

        let ainf = aexp == 255 && aman == 0;
        let binf = bexp == 255 && bman == 0;
        let anan = aexp == 255 && aman != 0;
        let bnan = bexp == 255 && bman != 0;
        let cinf = ainf || binf;
        let cnan = anan || bnan;

        nan_flag = nan_flag || cnan || (inf_flag && cinf && csgn != inf_sgn);
        inf_flag = (inf_flag || cinf) && !nan_flag;
        if cinf && !inf_flag { inf_sgn = csgn; }

        let cexp: i32 = if aexp == 0 || bexp == 0 {
            0 // FTZ: if either input has exp=0, product is zero
        } else {
            (aexp as i32) + (bexp as i32)
        };
        cexpl.push(cexp);
    }

    // Phase 2: Accumulate products in fixed-point.
    let cexpmax = *cexpl.iter().max().unwrap_or(&0);

    let mut cmansum: i64 = 0;
    for i in 0..n {
        let aman = if cexpl[i] > 0 { (1i64 << 7) | (amanl[i] as i64) } else { 0 };
        let bman = if cexpl[i] > 0 { (1i64 << 7) | (bmanl[i] as i64) } else { 0 };
        let cman_f = aman * bman; // Q.14 (up to 16 significant bits)

        let expdiff = cexpmax - cexpl[i];

        // Align to max exponent, keep iprec=23 bits.
        let (cman, _) = shift_round_rne(cman_f, expdiff + 14, iprec, false);

        if csgnl[i] {
            cmansum -= cman;
        } else {
            cmansum += cman;
        }
    }

    // Phase 3: Merge with previous fp32 accumulator.
    //
    // For subtract mode (MSC/SubMac): result = acc - products.
    // Negate the product sum so the merge computes acc + (-products).
    if subtract {
        cmansum = -cmansum;
    }

    let lo_pos = flb(cmansum);

    let (qsgn, qexp, qman_raw) = fp32_split(q);
    let qinf = qexp == 255 && qman_raw == 0;
    let qnan = qexp == 255 && qman_raw != 0;

    nan_flag = nan_flag || qnan || (inf_flag && qinf && qsgn != inf_sgn);
    inf_flag = (inf_flag || qinf) && !nan_flag;

    // Reconstruct signed accumulator mantissa (FTZ: exp=0 → man=0).
    let qman_full: i64 = if qexp > 0 {
        (1i64 << 23) | (qman_raw as i64)
    } else {
        0
    };
    let qman: i64 = if qsgn { -qman_full } else { qman_full };

    // Compute effective exponents for alignment.
    let cexp_eff = cexpmax as i64 + lo_pos as i64 - iprec as i64 - 127;
    let rexp = std::cmp::max(cexp_eff, qexp as i64) as i32;

    let qlo_pos = rexp as i64 - qexp as i64 + 23;
    let clo_pos = rexp as i64 - cexpmax as i64 + iprec as i64 + 127;

    let qsh_dn = qlo_pos - aprec as i64;
    let csh_dn = clo_pos - aprec as i64;

    // Swap: the value with the smaller shift keeps full precision (left-shifted),
    // the value with the larger shift gets rounded to aprec bits.
    let (qm3, cm3, _qlo_pos2, clo_pos2, qsh_dn2) = if qsh_dn > csh_dn {
        (cmansum, qman, clo_pos, qlo_pos, csh_dn)
    } else {
        (qman, cmansum, qlo_pos, clo_pos, qsh_dn)
    };

    let lsh = (-qsh_dn2) as u32;
    let qman2 = if lsh >= 63 { 0i64 } else { qm3.wrapping_shl(lsh) };
    let (cmansum2, _) = shift_round_rne(cm3, clo_pos2 as i32, aprec, true);

    let rsum2 = qman2 + cmansum2;
    let rsum = trnc(rsum2, true, 32);

    // Phase 4: Handle NaN/Inf.
    if nan_flag {
        return fp32_make(false, 255, 0x7F); // Canonical NaN
    }
    if inf_flag {
        return fp32_make(inf_sgn, 255, 0); // Infinity
    }

    // Phase 5: Normalize back to fp32.
    let sgn = rsum < 0;
    let sum_abs = rsum.unsigned_abs() as i64;

    if sum_abs == 0 {
        return 0;
    }

    let nlo_pos = flo(sum_abs);
    let (man, expincr) = shift_round_rne(sum_abs, nlo_pos, 23, true);
    let exp = rexp + nlo_pos - aprec + if expincr { 1 } else { 0 };

    if exp >= 255 {
        return fp32_make(sgn, 255, 0); // Overflow
    }
    if exp <= 0 {
        return fp32_make(sgn, 0, 0); // Underflow (FTZ)
    }

    fp32_make(sgn, exp as u8, man as u32 & 0x7F_FFFF)
}

// ---------------------------------------------------------------------------
// Config-driven full-width matmul (512-bit inputs, 1024-bit accumulator)
// ---------------------------------------------------------------------------

/// Extract an element from a 512-bit vector (`Vec512` = `[u32; 16]`).
///
/// `byte_idx` is the byte offset within the full 64-byte vector.
fn extract_element_512(src: &Vec512, byte_idx: usize, bits: u32, signed: bool) -> i64 {
    match bits {
        4 => {
            // 4-bit elements: two per byte
            let byte_pos = byte_idx / 2;
            let nibble = byte_idx % 2;
            let word = byte_pos / 4;
            let byte_in_word = byte_pos % 4;
            if word >= src.len() { return 0; }
            let raw_byte = ((src[word] >> (byte_in_word * 8)) & 0xFF) as u8;
            let val = if nibble == 0 { raw_byte & 0xF } else { (raw_byte >> 4) & 0xF };
            if signed && (val & 0x8) != 0 {
                // Sign-extend from 4 bits
                (val as i8 | !0xFu8 as i8) as i64
            } else {
                val as i64
            }
        }
        8 => {
            let word = byte_idx / 4;
            let byte_in_word = byte_idx % 4;
            if word >= src.len() { return 0; }
            let val = ((src[word] >> (byte_in_word * 8)) & 0xFF) as u8;
            if signed { val as i8 as i64 } else { val as i64 }
        }
        16 => {
            let elem_idx = byte_idx / 2;
            let word = elem_idx / 2;
            let half_in_word = elem_idx % 2;
            if word >= src.len() { return 0; }
            let val = ((src[word] >> (half_in_word * 16)) & 0xFFFF) as u16;
            if signed { val as i16 as i64 } else { val as i64 }
        }
        32 => {
            let word = byte_idx / 4;
            if word >= src.len() { return 0; }
            let val = src[word];
            if signed { val as i32 as i64 } else { val as i64 }
        }
        _ => 0,
    }
}

/// Read from a 1024-bit accumulator (`Acc1024` = `[u64; 16]`).
fn read_acc_wide(acc: &Acc1024, index: usize, acc_width: AccWidth) -> i64 {
    match acc_width {
        AccWidth::Acc32 => {
            // 32 x 32-bit lanes packed into 16 u64 words (two per word)
            let u64_lane = index / 2;
            let half = index % 2;
            let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
            bits as i32 as i64
        }
        AccWidth::Acc64 => {
            // 16 x 64-bit lanes, one per u64 word
            acc[index] as i64
        }
    }
}

/// Write to a 1024-bit accumulator (`Acc1024` = `[u64; 16]`).
fn write_acc_wide(acc: &mut Acc1024, index: usize, value: i64, acc_width: AccWidth) {
    match acc_width {
        AccWidth::Acc32 => {
            let u64_lane = index / 2;
            let half = index % 2;
            let masked = (value as u32) as u64;
            let shift = half * 32;
            acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (masked << shift);
        }
        AccWidth::Acc64 => {
            acc[index] = value as u64;
        }
    }
}

/// Write an fp32 value to a wide accumulator.
fn write_acc_wide_f32(acc: &mut Acc1024, index: usize, value: f32) {
    // bf16 mode always uses acc_cmb=1 (32-bit lanes)
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = value.to_bits() as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (bits << shift);
}

/// Read an fp32 value from a wide accumulator.
fn read_acc_wide_f32(acc: &Acc1024, index: usize) -> f32 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    f32::from_bits(bits)
}

/// Config-driven matrix multiply on full-width (512-bit) inputs.
///
/// Reads element data from the 512-bit A and B vectors, performs a tiled
/// matrix multiply according to the geometry in `config`, and writes the
/// result to the 1024-bit accumulator.
///
/// The input vectors are interpreted as flat arrays of elements in
/// row-major order:
/// - A[r][k] at byte offset `(r * inner + k) * (bits_x / 8)`
/// - B[k][c] at byte offset `(k * cols + c) * (bits_y / 8)`
///
/// Output goes to accumulator lane `r * cols + c`.
pub fn matmul_config_driven(
    acc: &mut Acc1024,
    a: &Vec512,
    b: &Vec512,
    config: &MatMulConfig,
) {
    let rows = config.rows as usize;
    let inner = config.inner as usize;
    let cols = config.cols as usize;
    let bits_x = config.bits_x;
    let bits_y = config.bits_y;
    let bytes_x = (bits_x / 8) as usize;
    let bytes_y = if bits_y == 4 { 1 } else { (bits_y / 8) as usize }; // 4-bit: 2 elements per byte

    // Zero accumulator if requested (zero_acc = !accumulate in MatMulConfig)
    if !config.accumulate {
        *acc = [0u64; 16];
    }

    if config.bfloat {
        // BFloat16 MAC using exact 29-bit accumulator model.
        //
        // Uses bf16_mac_hw_lane which faithfully implements the aietools
        // hardware model (bfloat16.py bf16_mac_hw): 23-bit product
        // alignment, 29-bit accumulator merge, RNE rounding throughout,
        // and proper FTZ on inputs/outputs.
        //
        // Element-wise mode (variant=1, 16x2x1 geometry): each lane L
        // computes A[L]*B[L] + A[L+16]*B[L+16].  Verified by hardware
        // characterization (bf16_elemwise_characterizer.s).
        let is_elemwise = rows == 16 && inner == 2 && cols == 1;

        for r in 0..rows {
            for c in 0..cols {
                let out_idx = r * cols + c;

                // Collect the bf16 element pairs for this output lane.
                let mut a_elems = Vec::with_capacity(inner);
                let mut b_elems = Vec::with_capacity(inner);

                for k in 0..inner {
                    let elem_idx = if is_elemwise {
                        r + k * 16
                    } else {
                        r * inner + k
                    };
                    let a_word = elem_idx / 2;
                    let a_half = elem_idx % 2;

                    let b_elem_idx = if is_elemwise {
                        r + k * 16
                    } else {
                        k * cols + c
                    };
                    let b_word = b_elem_idx / 2;
                    let b_half = b_elem_idx % 2;

                    a_elems.push(((a[a_word] >> (a_half * 16)) & 0xFFFF) as u16);
                    b_elems.push(((b[b_word] >> (b_half * 16)) & 0xFFFF) as u16);
                }

                let prev_bits = read_acc_wide_f32(acc, out_idx).to_bits();
                let result = bf16_mac_hw_lane(
                    prev_bits,
                    &a_elems,
                    &b_elems,
                    config.subtract,
                );
                write_acc_wide_f32(acc, out_idx, f32::from_bits(result));
            }
        }
        return;
    }

    // Integer path
    for r in 0..rows {
        for c in 0..cols {
            let out_idx = r * cols + c;
            let mut sum: i64 = 0;

            for k in 0..inner {
                // For 4-bit elements, extract_element_512 expects an ELEMENT
                // index (it divides by 2 internally to get the byte and uses
                // modulo 2 for the nibble selector).  For 8/16/32-bit elements
                // it expects a BYTE offset.
                let a_byte = if bits_x == 4 {
                    r * inner + k
                } else {
                    (r * inner + k) * bytes_x
                };
                let b_byte = if bits_y == 4 {
                    k * cols + c
                } else {
                    (k * cols + c) * bytes_y
                };

                let a_val = extract_element_512(a, a_byte, bits_x, config.x_signed);
                let b_val = extract_element_512(b, b_byte, bits_y, config.y_signed);

                sum += a_val * b_val;
            }

            let prev = read_acc_wide(acc, out_idx, config.acc_width);
            if config.subtract {
                write_acc_wide(acc, out_idx, prev - sum, config.acc_width);
            } else {
                write_acc_wide(acc, out_idx, prev + sum, config.acc_width);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// int8 x int8 -> int32 accumulator
//
// Geometry: rows=4, inner=8, cols=8 => 32 output elements
//
// A is 4 rows x 8 cols of int8 = 32 bytes = 256 bits (one vector register)
// B is 8 rows x 8 cols of int8 = 64 bytes = 512 bits (two vector registers,
//   but the hardware only uses 256 bits from the second source, selecting
//   via permutation. For the basic mode we use the full 256-bit B vector
//   reshaped as 8x4.)
//
// Actually for the basic int8xi8 mode (mmode=1, perm_mode row 1):
//   rows=4, inner=8, cols=8
// But we only have 256 bits of B = 32 int8 values. With 8 rows and 8 cols
// that would be 64 values, which exceeds our vector width.
//
// Looking more carefully at the hardware: the 256-bit vector holds 32 int8
// elements. For 4x8 output, the inner dimension must be such that
// 4 * inner * sizeof(int8) <= 256 bits AND inner * 8 * sizeof(int8) <= 256 bits.
// So inner * 8 <= 32, meaning inner <= 4.
//
// The correct basic mode is actually rows=4, inner=4, cols=8 for 16xi8 input,
// but for 8x8 input (mmode=1): rows=4, inner=8, cols=8.
// In that case B needs 8*8 = 64 bytes which exceeds 256 bits.
//
// The resolution: the hardware permute unit rearranges the input data.
// With 512-bit permute width, B can actually be drawn from the full permute
// space. For the simpler emulation, we implement the most common sub-case:
// rows=4, inner=8, cols=4 which fits in 256 bits for both A (4*8=32 bytes)
// and B (8*4=32 bytes).
//
// For the full 4x8x8 mode, the hardware uses both X and Y permute inputs
// which may come from different vector register halves or two registers.
// We implement the 4x8x4 variant first as it matches one 256-bit register
// per operand.
// ---------------------------------------------------------------------------

/// int8 x int8 matrix multiply with 32-bit accumulator.
///
/// Geometry: A[4][8] * B[8][4] = C[4][4], 16 output int32 values.
/// A: 32 int8 values from 256-bit vector, row-major 4 rows x 8 cols.
/// B: 32 int8 values from 256-bit vector, row-major 8 rows x 4 cols.
/// Output: 16 int32 values in 8 u64 lanes (two int32 per u64).
fn matmul_i8xi8(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
    // A is 4 rows x 8 cols of int8 = 32 elements (256 bits)
    // B is 8 rows x 4 cols of int8 = 32 elements (256 bits)
    // Output is 4 rows x 4 cols of int32 = 16 elements in acc
    let geom = TileGeometry { rows: 4, inner: 8, cols: 4 };

    for r in 0..geom.rows {
        for c in 0..geom.cols {
            let out_idx = r * geom.cols + c;
            let mut sum: i64 = 0;

            for k in 0..geom.inner {
                let a_idx = r * geom.inner + k;
                let b_idx = k * geom.cols + c;

                let a_val = if signed_a {
                    extract_i8(a, a_idx) as i64
                } else {
                    extract_u8(a, a_idx) as i64
                };

                let b_val = if signed_b {
                    extract_i8(b, b_idx) as i64
                } else {
                    extract_u8(b, b_idx) as i64
                };

                sum += a_val * b_val;
            }

            let prev = read_acc32(acc, out_idx);
            if subtract {
                write_acc32(acc, out_idx, prev - sum);
            } else {
                write_acc32(acc, out_idx, prev + sum);
            }
        }
    }
}

/// int16 x int16 matrix multiply with 32-bit accumulator (acc_cmb=1).
///
/// Geometry: A[4][2] * B[2][8] = C[4][8], 32 output int32 values.
/// A: 16 int16 values from 256-bit vector. We use the first 8 (4 rows x 2 inner).
/// B: 16 int16 values from 256-bit vector. We use all 16 (2 rows x 8 cols).
/// Output: 32 int32 values packed into the accumulator.
///
/// Note: with acc_cmb=1 and 32-bit accumulator, we get 32 output lanes.
/// With 8 u64 lanes that's 16 int32 values directly addressable, so we use
/// the common sub-case: A[4][2] * B[2][4] = C[4][4] = 16 outputs.
fn matmul_i16xi16_32(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
    // A: 4 rows x 2 inner = 8 int16 elements (128 bits, first half of vector)
    // B: 2 rows x 4 cols = 8 int16 elements (128 bits, first half of vector)
    // Output: 4 rows x 4 cols = 16 int32 values
    let geom = TileGeometry { rows: 4, inner: 2, cols: 4 };

    for r in 0..geom.rows {
        for c in 0..geom.cols {
            let out_idx = r * geom.cols + c;
            let mut sum: i64 = 0;

            for k in 0..geom.inner {
                let a_idx = r * geom.inner + k;
                let b_idx = k * geom.cols + c;

                let a_val = if signed_a {
                    extract_i16(a, a_idx) as i64
                } else {
                    extract_u16(a, a_idx) as i64
                };

                let b_val = if signed_b {
                    extract_i16(b, b_idx) as i64
                } else {
                    extract_u16(b, b_idx) as i64
                };

                sum += a_val * b_val;
            }

            let prev = read_acc32(acc, out_idx);
            if subtract {
                write_acc32(acc, out_idx, prev - sum);
            } else {
                write_acc32(acc, out_idx, prev + sum);
            }
        }
    }
}

/// bf16 x bf16 matrix multiply with fp32 accumulator.
///
/// Geometry: A[4][8] * B[8][4] = C[4][4], 16 output fp32 values.
/// A: 16 bf16 values from 256-bit vector, reinterpreted as 4 rows x 4 cols
///    (limited by 256-bit width: 16 bf16 = 4x4).
/// B: 16 bf16 values, reinterpreted as 4 rows x 4 cols.
///
/// The hardware actually uses rows=4, inner=8, cols=4 from constants.py,
/// but with 256-bit inputs we only have 16 bf16 values per vector,
/// so the practical single-register mode is 4x4x4.
fn matmul_bf16xbf16(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    subtract: bool,
) {
    // A: 4 rows x 4 inner = 16 bf16 elements (256 bits)
    // B: 4 rows x 4 cols = 16 bf16 elements (256 bits)
    // Output: 4 rows x 4 cols = 16 fp32 values
    let geom = TileGeometry { rows: 4, inner: 4, cols: 4 };

    for r in 0..geom.rows {
        for c in 0..geom.cols {
            let out_idx = r * geom.cols + c;
            let mut sum: f32 = 0.0;

            for k in 0..geom.inner {
                let a_idx = r * geom.inner + k;
                let b_idx = k * geom.cols + c;

                let a_val = extract_bf16_as_f32(a, a_idx);
                let b_val = extract_bf16_as_f32(b, b_idx);
                sum += a_val * b_val;
            }

            let prev = read_acc_f32(acc, out_idx);
            if subtract {
                write_acc_f32(acc, out_idx, prev - sum);
            } else {
                write_acc_f32(acc, out_idx, prev + sum);
            }
        }
    }
}

/// int32 x int16 matrix multiply with 64-bit accumulator (acc_cmb=2).
///
/// Geometry: A[4][2] * B[2][4] = C[4][4], producing 16 int64 outputs.
/// But we only have 8 u64 lanes, so the practical mode is:
/// A[2][2] * B[2][4] = C[2][4] = 8 int64 outputs.
///
/// Actually from constants.py: mmode=7 is 32x16 acc_cmb=2, perm_modes
/// include rows=4, inner=2, cols=4.
fn matmul_i32xi16(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
    // A: 4 rows x 2 inner of int32 = 8 elements (256 bits)
    // B: 2 rows x 4 cols of int16 = 8 elements (128 bits)
    // Output: 4 rows x 2 cols = 8 int64 values
    //
    // With acc_cmb=2, each output is 64 bits, fitting in one u64 lane.
    // 4*2 = 8 outputs = 8 u64 lanes.
    let geom = TileGeometry { rows: 4, inner: 2, cols: 2 };

    for r in 0..geom.rows {
        for c in 0..geom.cols {
            let out_idx = r * geom.cols + c;
            let mut sum: i64 = 0;

            for k in 0..geom.inner {
                let a_idx = r * geom.inner + k;
                let b_idx = k * geom.cols + c;

                let a_val = if signed_a {
                    extract_i32(a, a_idx) as i64
                } else {
                    a[a_idx] as i64
                };

                let b_val = if signed_b {
                    extract_i16(b, b_idx) as i64
                } else {
                    extract_u16(b, b_idx) as i64
                };

                sum += a_val * b_val;
            }

            let prev = read_acc64(acc, out_idx);
            if subtract {
                write_acc64(acc, out_idx, prev - sum);
            } else {
                write_acc64(acc, out_idx, prev + sum);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack int8 values into [u32; 8] in little-endian order.
    fn pack_i8(values: &[i8]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate() {
            let word = i / 4;
            let byte = i % 4;
            packed[word] |= ((v as u8) as u32) << (byte * 8);
        }
        packed
    }

    /// Pack int16 values into [u32; 8] in little-endian order.
    fn pack_i16(values: &[i16]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate() {
            let word = i / 2;
            let half = i % 2;
            packed[word] |= ((v as u16) as u32) << (half * 16);
        }
        packed
    }

    /// Pack bf16 values (as u16 bit patterns) into [u32; 8].
    fn pack_bf16(values: &[u16]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate() {
            let word = i / 2;
            let half = i % 2;
            packed[word] |= (v as u32) << (half * 16);
        }
        packed
    }

    /// Convert f32 to bf16 bit pattern (truncate lower 16 bits).
    fn f32_to_bf16_bits(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    // -----------------------------------------------------------------------
    // int8 x int8 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i8xi8_identity_like() {
        // A = 4x8 identity-like (first 4 cols are identity, rest zero)
        // B = 8x4 with known values in first 4 rows
        //
        // A[r][k] = if k == r { 1 } else { 0 } for k < 4, rest 0
        // B[k][c] = (k * 4 + c + 1) for k < 4, rest 0
        //
        // Result should be: C[r][c] = B[r][c] = r * 4 + c + 1

        let mut a_vals = [0i8; 32];
        // Row 0: a_vals[0] = 1 (k=0)
        a_vals[0] = 1;
        // Row 1: a_vals[8+1] = 1 (k=1)
        a_vals[8 + 1] = 1;
        // Row 2: a_vals[16+2] = 1 (k=2)
        a_vals[16 + 2] = 1;
        // Row 3: a_vals[24+3] = 1 (k=3)
        a_vals[24 + 3] = 1;

        let mut b_vals = [0i8; 32];
        for k in 0..4 {
            for c in 0..4 {
                b_vals[k * 4 + c] = (k * 4 + c + 1) as i8;
            }
        }

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);

        // Check C[r][c] = B[r][c] for the identity rows
        for r in 0..4 {
            for c in 0..4 {
                let out_idx = r * 4 + c;
                let expected = (r * 4 + c + 1) as i64;
                let actual = read_acc32(&acc, out_idx);
                assert_eq!(
                    actual, expected,
                    "C[{}][{}]: expected {}, got {}",
                    r, c, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_i8xi8_all_ones() {
        // A = 4x8, all 1s
        // B = 8x4, all 1s
        // C[r][c] = sum of 8 ones = 8

        let a_vals = [1i8; 32];
        let b_vals = [1i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);

        for r in 0..4 {
            for c in 0..4 {
                let out_idx = r * 4 + c;
                assert_eq!(
                    read_acc32(&acc, out_idx),
                    8,
                    "C[{}][{}] should be 8",
                    r, c
                );
            }
        }
    }

    #[test]
    fn test_i8xi8_accumulate() {
        // Verify accumulation: run matmul twice, values should double
        let a_vals = [1i8; 32];
        let b_vals = [1i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        matmul_i8xi8(&mut acc, &a, &b, true, true, false);

        for r in 0..4 {
            for c in 0..4 {
                let out_idx = r * 4 + c;
                assert_eq!(
                    read_acc32(&acc, out_idx),
                    16,
                    "C[{}][{}] should be 16 after two accumulations",
                    r, c
                );
            }
        }
    }

    #[test]
    fn test_i8xi8_subtract() {
        // First accumulate, then subtract the same product
        let a_vals = [2i8; 32];
        let b_vals = [3i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        // acc += A * B => each output = 8 * (2 * 3) = 48
        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        for idx in 0..16 {
            assert_eq!(read_acc32(&acc, idx), 48);
        }

        // acc -= A * B => each output = 48 - 48 = 0
        matmul_i8xi8(&mut acc, &a, &b, true, true, true);
        for idx in 0..16 {
            assert_eq!(read_acc32(&acc, idx), 0);
        }
    }

    #[test]
    fn test_i8xi8_signed_negative() {
        // Test with negative values
        let mut a_vals = [0i8; 32];
        let mut b_vals = [0i8; 32];

        // A[0][0] = -1, B[0][0] = -2
        // C[0][0] = (-1) * (-2) = 2
        a_vals[0] = -1;
        b_vals[0] = -2;

        // A[0][1] = 3, B[1][0] = -4
        // C[0][0] += 3 * (-4) = -12
        // Total C[0][0] = 2 + (-12) = -10
        a_vals[1] = 3;
        b_vals[4] = -4;

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        assert_eq!(read_acc32(&acc, 0), -10);
    }

    #[test]
    fn test_i8xi8_unsigned() {
        // Unsigned: 200 * 200 = 40000 (would be negative if signed)
        let mut a_vals = [0i8; 32];
        let mut b_vals = [0i8; 32];

        // 200 as u8 = 0xC8, which as i8 = -56
        a_vals[0] = -56; // 200 as u8
        b_vals[0] = -56; // 200 as u8

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        // Signed: (-56) * (-56) = 3136
        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        assert_eq!(read_acc32(&acc, 0), 3136);

        // Unsigned: 200 * 200 = 40000
        let mut acc2 = [0u64; 8];
        matmul_i8xi8(&mut acc2, &a, &b, false, false, false);
        assert_eq!(read_acc32(&acc2, 0), 40000);
    }

    // -----------------------------------------------------------------------
    // int16 x int16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i16xi16_identity() {
        // A = 4x2, identity-like: A[r][k] = delta(r%2, k)
        // B = 2x4 with sequential values
        // For rows 0,1: result should pick from B rows 0,1
        // For rows 2,3: same pattern wraps

        let mut a_vals = [0i16; 16];
        // Row 0: A[0][0] = 1 (k=0)
        a_vals[0] = 1;
        // Row 1: A[1][1] = 1 (k=1)
        a_vals[3] = 1;

        let mut b_vals = [0i16; 16];
        // B[0][0..4] = {10, 20, 30, 40}
        b_vals[0] = 10;
        b_vals[1] = 20;
        b_vals[2] = 30;
        b_vals[3] = 40;
        // B[1][0..4] = {50, 60, 70, 80}
        b_vals[4] = 50;
        b_vals[5] = 60;
        b_vals[6] = 70;
        b_vals[7] = 80;

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i16xi16_32(&mut acc, &a, &b, true, true, false);

        // C[0][c] = A[0][0]*B[0][c] + A[0][1]*B[1][c] = 1*B[0][c] + 0 = B[0][c]
        assert_eq!(read_acc32(&acc, 0), 10);
        assert_eq!(read_acc32(&acc, 1), 20);
        assert_eq!(read_acc32(&acc, 2), 30);
        assert_eq!(read_acc32(&acc, 3), 40);

        // C[1][c] = A[1][0]*B[0][c] + A[1][1]*B[1][c] = 0 + 1*B[1][c] = B[1][c]
        assert_eq!(read_acc32(&acc, 4), 50);
        assert_eq!(read_acc32(&acc, 5), 60);
        assert_eq!(read_acc32(&acc, 6), 70);
        assert_eq!(read_acc32(&acc, 7), 80);
    }

    #[test]
    fn test_i16xi16_multiply() {
        // A = [[1, 2], [3, 4], [5, 6], [7, 8]]  (4x2)
        // B = [[1, 0, 0, 0], [0, 1, 0, 0]]       (2x4)
        //
        // C = A * B:
        // C[0] = [1*1+2*0, 1*0+2*1, 1*0+2*0, 1*0+2*0] = [1, 2, 0, 0]
        // C[1] = [3, 4, 0, 0]
        // C[2] = [5, 6, 0, 0]
        // C[3] = [7, 8, 0, 0]

        let a_vals: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0];
        let b_vals: [i16; 16] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i16xi16_32(&mut acc, &a, &b, true, true, false);

        assert_eq!(read_acc32(&acc, 0), 1);  // C[0][0]
        assert_eq!(read_acc32(&acc, 1), 2);  // C[0][1]
        assert_eq!(read_acc32(&acc, 4), 3);  // C[1][0]
        assert_eq!(read_acc32(&acc, 5), 4);  // C[1][1]
        assert_eq!(read_acc32(&acc, 8), 5);  // C[2][0]
        assert_eq!(read_acc32(&acc, 9), 6);  // C[2][1]
        assert_eq!(read_acc32(&acc, 12), 7); // C[3][0]
        assert_eq!(read_acc32(&acc, 13), 8); // C[3][1]
    }

    #[test]
    fn test_i16xi16_accumulate_subtract() {
        let a_vals: [i16; 16] = [10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b_vals: [i16; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mut acc = [0u64; 8];

        // acc += A * B => C[0][0] = 10
        matmul_i16xi16_32(&mut acc, &a, &b, true, true, false);
        assert_eq!(read_acc32(&acc, 0), 10);

        // acc -= A * B => C[0][0] = 10 - 10 = 0
        matmul_i16xi16_32(&mut acc, &a, &b, true, true, true);
        assert_eq!(read_acc32(&acc, 0), 0);
    }

    // -----------------------------------------------------------------------
    // bf16 x bf16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_identity() {
        // A = 4x4 identity matrix as bf16
        // B = 4x4 with known values
        // C = B (identity property)

        let mut a_bits = [0u16; 16];
        let one = f32_to_bf16_bits(1.0);
        // Diagonal: A[0][0], A[1][1], A[2][2], A[3][3]
        a_bits[0 * 4 + 0] = one;
        a_bits[1 * 4 + 1] = one;
        a_bits[2 * 4 + 2] = one;
        a_bits[3 * 4 + 3] = one;

        let mut b_bits = [0u16; 16];
        for i in 0..16 {
            b_bits[i] = f32_to_bf16_bits((i + 1) as f32);
        }

        let a = pack_bf16(&a_bits);
        let b = pack_bf16(&b_bits);
        let mut acc = [0u64; 8];

        matmul_bf16xbf16(&mut acc, &a, &b, false);

        for r in 0..4 {
            for c in 0..4 {
                let idx = r * 4 + c;
                let expected = (r * 4 + c + 1) as f32;
                let actual = read_acc_f32(&acc, idx);
                assert!(
                    (actual - expected).abs() < 0.01,
                    "C[{}][{}]: expected {}, got {}",
                    r, c, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_bf16_all_ones() {
        // A = 4x4 all ones, B = 4x4 all ones
        // C[r][c] = sum of 4 ones = 4.0
        let one = f32_to_bf16_bits(1.0);
        let a_bits = [one; 16];
        let b_bits = [one; 16];

        let a = pack_bf16(&a_bits);
        let b = pack_bf16(&b_bits);
        let mut acc = [0u64; 8];

        matmul_bf16xbf16(&mut acc, &a, &b, false);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!(
                (actual - 4.0).abs() < 0.01,
                "Output[{}]: expected 4.0, got {}",
                idx, actual
            );
        }
    }

    #[test]
    fn test_bf16_accumulate() {
        let one = f32_to_bf16_bits(1.0);
        let a_bits = [one; 16];
        let b_bits = [one; 16];

        let a = pack_bf16(&a_bits);
        let b = pack_bf16(&b_bits);
        let mut acc = [0u64; 8];

        // First: acc = 4.0
        matmul_bf16xbf16(&mut acc, &a, &b, false);
        // Second: acc = 8.0
        matmul_bf16xbf16(&mut acc, &a, &b, false);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!(
                (actual - 8.0).abs() < 0.01,
                "Output[{}]: expected 8.0, got {}",
                idx, actual
            );
        }
    }

    #[test]
    fn test_bf16_subtract() {
        let two = f32_to_bf16_bits(2.0);
        let three = f32_to_bf16_bits(3.0);
        let a_bits = [two; 16];
        let b_bits = [three; 16];

        let a = pack_bf16(&a_bits);
        let b = pack_bf16(&b_bits);
        let mut acc = [0u64; 8];

        // acc += 2*3*4 = 24 per lane
        matmul_bf16xbf16(&mut acc, &a, &b, false);
        // acc -= 2*3*4 = 24 per lane => 0
        matmul_bf16xbf16(&mut acc, &a, &b, true);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!(
                actual.abs() < 0.01,
                "Output[{}]: expected 0.0, got {}",
                idx, actual
            );
        }
    }

    // -----------------------------------------------------------------------
    // int32 x int16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i32xi16_basic() {
        // A = 4x2 of int32 = 8 values (256 bits)
        // B = 2x2 of int16 = 4 values (64 bits)
        // C = 4x2 of int64 = 8 values

        let mut a_packed = [0u32; 8];
        // A[0][0] = 100, A[0][1] = 200
        a_packed[0] = 100;
        a_packed[1] = 200;
        // A[1][0] = 300, A[1][1] = 400
        a_packed[2] = 300;
        a_packed[3] = 400;

        // B = 2x2 identity matrix (row-major): B[0][0]=1, B[0][1]=0, B[1][0]=0, B[1][1]=1
        let b_vals: [i16; 16] = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b = pack_i16(&b_vals);

        let mut acc = [0u64; 8];
        matmul_i32xi16(&mut acc, &a_packed, &b, true, true, false);

        // C[0][0] = 100*1 + 200*0 = 100
        // C[0][1] = 100*0 + 200*1 = 200
        assert_eq!(read_acc64(&acc, 0), 100);
        assert_eq!(read_acc64(&acc, 1), 200);

        // C[1][0] = 300*1 + 400*0 = 300
        // C[1][1] = 300*0 + 400*1 = 400
        assert_eq!(read_acc64(&acc, 2), 300);
        assert_eq!(read_acc64(&acc, 3), 400);
    }

    // -----------------------------------------------------------------------
    // Element extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_i8() {
        let packed = pack_i8(&[1, -2, 3, -4, 5, -6, 7, -8,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(extract_i8(&packed, 0), 1);
        assert_eq!(extract_i8(&packed, 1), -2);
        assert_eq!(extract_i8(&packed, 2), 3);
        assert_eq!(extract_i8(&packed, 3), -4);
        assert_eq!(extract_i8(&packed, 4), 5);
        assert_eq!(extract_i8(&packed, 5), -6);
    }

    #[test]
    fn test_extract_i16() {
        let packed = pack_i16(&[100, -200, 300, -400, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(extract_i16(&packed, 0), 100);
        assert_eq!(extract_i16(&packed, 1), -200);
        assert_eq!(extract_i16(&packed, 2), 300);
        assert_eq!(extract_i16(&packed, 3), -400);
    }

    #[test]
    fn test_acc32_read_write_roundtrip() {
        let mut acc = [0u64; 8];
        write_acc32(&mut acc, 0, 42);
        write_acc32(&mut acc, 1, -100);
        write_acc32(&mut acc, 15, 999);

        assert_eq!(read_acc32(&acc, 0), 42);
        assert_eq!(read_acc32(&acc, 1), -100);
        assert_eq!(read_acc32(&acc, 15), 999);
    }

    #[test]
    fn test_acc_f32_read_write_roundtrip() {
        let mut acc = [0u64; 8];
        write_acc_f32(&mut acc, 0, 3.14);
        write_acc_f32(&mut acc, 1, -2.71);
        write_acc_f32(&mut acc, 15, 42.0);

        assert!((read_acc_f32(&acc, 0) - 3.14).abs() < 0.001);
        assert!((read_acc_f32(&acc, 1) - (-2.71)).abs() < 0.001);
        assert!((read_acc_f32(&acc, 15) - 42.0).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // Public API tests (matmul_dense / matmul_sub)
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_dense_i8() {
        let a_vals = [1i8; 32];
        let b_vals = [1i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_dense(&mut acc, &a, &b, ElementType::Int8, true, true);

        // Each output = inner (8) dot products of 1*1 = 8
        for idx in 0..16 {
            assert_eq!(read_acc32(&acc, idx), 8);
        }
    }

    #[test]
    fn test_matmul_sub_i16() {
        let a_vals: [i16; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b_vals: [i16; 16] = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);

        // Pre-load accumulator with 100 in lane 0
        let mut acc = [0u64; 8];
        write_acc32(&mut acc, 0, 100);

        matmul_sub(&mut acc, &a, &b, ElementType::Int16, true, true);

        // C[0][0] = 100 - (1*5 + 0*0) = 95
        assert_eq!(read_acc32(&acc, 0), 95);
    }

    #[test]
    fn test_matmul_dense_bf16() {
        let two = f32_to_bf16_bits(2.0);
        let three = f32_to_bf16_bits(3.0);

        // A = all 2.0, B = all 3.0
        // Each output = 4 * (2.0 * 3.0) = 24.0
        let a = pack_bf16(&[two; 16]);
        let b = pack_bf16(&[three; 16]);
        let mut acc = [0u64; 8];

        matmul_dense(&mut acc, &a, &b, ElementType::BFloat16, true, true);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!(
                (actual - 24.0).abs() < 0.1,
                "Output[{}]: expected 24.0, got {}",
                idx, actual
            );
        }
    }

    #[test]
    fn test_bf16_mac_hw_lane_basic() {
        // Test 1: 8 * (1.0 * 1.0) + 0 = 8.0
        let a = [0x3F80u16; 8]; // bf16 1.0
        let b = [0x3F80u16; 8];
        let r = bf16_mac_hw_lane(0, &a, &b, false);
        assert_eq!(r, 0x41000000, "8*1*1+0: expected 0x41000000, got 0x{:08X}", r);

        // Test 2: 8 * (-2.0 * 3.0) + 0 = -48.0
        let a2 = [0xC000u16; 8]; // bf16 -2.0
        let b2 = [0x4040u16; 8]; // bf16 3.0
        let r2 = bf16_mac_hw_lane(0, &a2, &b2, false);
        assert_eq!(r2, 0xC2400000, "8*-2*3+0: expected 0xC2400000, got 0x{:08X}", r2);

        // Test 3: tiny * tiny = underflow to 0
        let a3 = [0x0080u16; 8]; // bf16 min normal
        let b3 = [0x0080u16; 8];
        let r3 = bf16_mac_hw_lane(0, &a3, &b3, false);
        assert_eq!(r3, 0x00000000, "tiny*tiny: expected 0, got 0x{:08X}", r3);

        // Test 4: zeros * ones = 0
        let a4 = [0x0000u16; 8];
        let b4 = [0x3F80u16; 8];
        let r4 = bf16_mac_hw_lane(0, &a4, &b4, false);
        assert_eq!(r4, 0x00000000, "zeros: expected 0, got 0x{:08X}", r4);

        // Test 5: subtract mode: 0 - 8*1*1 = -8.0
        let r5 = bf16_mac_hw_lane(0, &a, &b, true);
        assert_eq!(r5, 0xC1000000, "0-8*1*1: expected 0xC1000000, got 0x{:08X}", r5);
    }

    // -----------------------------------------------------------------------
    // execute_matmul entry point tests
    // -----------------------------------------------------------------------

    use crate::interpreter::bundle::SlotIndex;
    use crate::interpreter::state::ExecutionContext;

    /// Build a minimal SlotOp for a MAC-family instruction.
    ///
    /// Sources: [VectorReg(a_reg), VectorReg(b_reg), ScalarReg(conf_reg)]
    /// Dest: AccumReg(acc_reg)
    fn make_mac_op(
        semantic: SemanticOp,
        a_vreg: u8,
        b_vreg: u8,
        conf_sreg: u8,
        acc_dest: u8,
        encoding_name: Option<&str>,
    ) -> SlotOp {
        let mut op = SlotOp::from_semantic(SlotIndex::Accumulator, semantic);
        op.is_vector = true;
        op.is_wide_vector = true;
        op.sources.push(Operand::VectorReg(a_vreg));
        op.sources.push(Operand::VectorReg(b_vreg));
        op.sources.push(Operand::ScalarReg(conf_sreg));
        op.dest = Some(Operand::AccumReg(acc_dest));
        op.encoding_name = encoding_name.map(|s| s.to_string());
        op
    }

    /// Build a MAC op with an extra AccumReg source (for AddMac/SubMac).
    fn make_double_acc_op(
        semantic: SemanticOp,
        a_vreg: u8,
        b_vreg: u8,
        conf_sreg: u8,
        acc_dest: u8,
        acc_src: u8,
        encoding_name: Option<&str>,
    ) -> SlotOp {
        let mut op = make_mac_op(semantic, a_vreg, b_vreg, conf_sreg, acc_dest, encoding_name);
        // Insert AccumReg source before the ScalarReg (config is last).
        op.sources.insert(2, Operand::AccumReg(acc_src));
        op
    }

    /// Pack [u32; 16] (Vec512) of all-ones bytes (int8 = 1 in every byte).
    fn vec512_all_ones_i8() -> Vec512 {
        // 0x01010101 repeated 16 times = 64 bytes of 0x01
        [0x01010101u32; 16]
    }

    /// Pack Vec512 of all bf16(1.0) values.
    fn vec512_all_ones_bf16() -> Vec512 {
        // bf16(1.0) = 0x3F80; two per u32 = 0x3F80_3F80
        let word = 0x3F80_3F80u32;
        [word; 16]
    }

    /// Build a config word for int8xi8 mode.
    ///
    /// Config word layout:
    ///   bit 0:    zero_acc
    ///   bits 1-2: amode (0 = acc_cmb=1, 32-bit)
    ///   bits 3-4: bmode (1 = int8 B)
    ///   bits 5-7: variant (0)
    ///   bit 8:    sgn_y (1 = signed)
    ///   bit 9:    sgn_x (1 = signed)
    ///
    /// i8xi8 signed accumulate: amode=0, bmode=1, sgn_x=1, sgn_y=1
    /// = (1<<3) | (1<<8) | (1<<9) = 0x308
    fn config_i8xi8_accumulate() -> u32 {
        (1 << 3) | (1 << 8) | (1 << 9) // amode=0, bmode=1, signed
    }

    /// Build a config word for bf16 mode.
    /// bf16 uses the bf16 lookup path (variant=0), sign bits irrelevant.
    fn config_bf16_accumulate() -> u32 {
        0x00 // zero_acc=0, variant=0
    }

    /// Build a config word for int8xi8 with zero_acc=1 (clear before multiply).
    fn config_i8xi8_zero_acc() -> u32 {
        (1 << 3) | (1 << 8) | (1 << 9) | 1 // same as accumulate but bit 0 set
    }

    #[test]
    fn test_execute_matmul_i8xi8_ones() {
        // All-ones int8 inputs through the full entry point.
        // Config-driven: 4x8x8 (int8xi8, pmode=0) -> 32 output elements.
        // Each output = sum(k=0..7) { 1 * 1 } = 8.
        let mut ctx = ExecutionContext::new();

        // Zero accumulator cm0 before accumulate (no zero_acc in config).
        ctx.accumulator.write_wide(0, [0u64; 16]);

        // Write all-ones to vector regs x0 (v0+v1) and x2 (v2+v3).
        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);

        // Write config to scalar r5.
        ctx.scalar.write(5, config_i8xi8_accumulate());

        let op = make_mac_op(
            SemanticOp::Mac,
            0, // x0
            2, // x2
            5, // config in r5
            0, // cm0
            None,
        );

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled, "execute_matmul should handle Mac semantic");

        // Read back accumulator cm0.
        let acc = ctx.accumulator.read_wide(0);

        // For 4x8x8 int8, there are 32 output elements (Acc32, two per u64).
        // Each = 8.
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(
                val, 8,
                "output[{}]: expected 8, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_execute_matmul_bf16_ones() {
        // All bf16(1.0) inputs. Config-driven: 4x8x4 bf16 -> 16 outputs.
        // Each output = sum(k=0..7) { 1.0 * 1.0 } = 8.0.
        let mut ctx = ExecutionContext::new();

        // Zero accumulator cm0 before accumulate (no zero_acc in config).
        ctx.accumulator.write_wide(0, [0u64; 16]);

        let ones = vec512_all_ones_bf16();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_bf16_accumulate());

        let mut op = make_mac_op(
            SemanticOp::Mac,
            0,
            2,
            5,
            0,
            Some("VMAC_F_vmac_bm_core_dense"),
        );
        op.element_type = Some(ElementType::BFloat16);

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        // bf16 mode: 16 fp32 outputs, two per u64 lane.
        for i in 0..16 {
            let lane = i / 2;
            let half = i % 2;
            let bits = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
            let val = f32::from_bits(bits);
            assert!(
                (val - 8.0).abs() < 0.01,
                "output[{}]: expected 8.0, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_execute_matmul_negate() {
        // NegMul semantic: output = -(A * B).
        // All-ones int8 -> each product sum = 8, negated = -8.
        let mut ctx = ExecutionContext::new();

        // Zero accumulator cm0 before accumulate (no zero_acc in config).
        ctx.accumulator.write_wide(0, [0u64; 16]);

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_i8xi8_accumulate());

        let op = make_mac_op(
            SemanticOp::NegMul,
            0,
            2,
            5,
            0,
            None,
        );

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as u32 as i32;
            assert_eq!(
                val, -8,
                "output[{}]: expected -8, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_execute_matmul_zero_acc() {
        // Verify zero_acc=1 clears the accumulator before multiply.
        let mut ctx = ExecutionContext::new();

        // Pre-fill accumulator with garbage.
        let mut preload = [0u64; 16];
        for i in 0..16 {
            preload[i] = 0xDEAD_BEEF_CAFE_BABEu64;
        }
        ctx.accumulator.write_wide(0, preload);

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_i8xi8_zero_acc());

        let op = make_mac_op(
            SemanticOp::Mac,
            0,
            2,
            5,
            0,
            None,
        );

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        // With zero_acc, accumulator is cleared first, so result = 0 + products = 8.
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(
                val, 8,
                "output[{}]: expected 8 (zero_acc should clear), got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_execute_matmul_returns_false_for_non_mac() {
        // Verify non-MAC semantics return false.
        let mut ctx = ExecutionContext::new();
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add);
        assert!(!execute_matmul(&op, &mut ctx));
    }

    // -----------------------------------------------------------------------
    // Sparse config-driven tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_config_driven_all_zero_b_produces_zero() {
        // Sparse matmul with all-zero B register -> product = 0.
        let mut a = [0u8; 128];
        for b in a.iter_mut() { *b = 1; } // all-ones A
        let b = [0u8; 64]; // all-zero B register
        let mask: u128 = 0x33333333_33333333_33333333_33333333; // all groups: bits 0,1 set
        let mut acc = [0u64; 16];

        // i8xi8 sparse config: amode=0, bmode=1, variant=5 -> 4x16x8
        let config = MatMulConfig::from_config_word(
            (1 << 3) | (5 << 5) | (1 << 8) | (1 << 9) | 1,
            false,
        ).unwrap();
        assert!(config.sparse, "variant=5 should give sparse geometry");
        assert_eq!(config.inner, 16, "sparse i8xi8 inner should be 16");

        matmul_sparse_config_driven(&mut acc, &a, &b, mask, &config);

        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 0, "output[{}] should be 0 with zero B", i);
        }
    }

    #[test]
    fn test_sparse_config_driven_all_ones() {
        // Sparse matmul with all-ones A and B, mask selects positions 0,1.
        // rows=4, inner=16 (sparse), cols=8
        // inner_groups=4, each with 2 active positions -> 8 dense inner
        // Each output = sum over 8 dense k: { 1 * 1 } = 8.
        let a = [1u8; 128];
        let b = [1u8; 64]; // all-ones B register
        let mask: u128 = 0x33333333_33333333_33333333_33333333; // bits 0,1 in each group
        let mut acc = [0u64; 16];

        let config = MatMulConfig::from_config_word(
            (1 << 3) | (5 << 5) | (1 << 8) | (1 << 9) | 1,
            false,
        ).unwrap();

        matmul_sparse_config_driven(&mut acc, &a, &b, mask, &config);

        // 4 inner_groups, each contributing 2 multiplies of 1*1 = 8 total per output.
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 8, "output[{}]: expected 8, got {}", i, val);
        }
    }

    #[test]
    fn test_execute_matmul_sparse_via_encoding_name() {
        // Test the full execute_matmul path with a sparse encoding name.
        // The encoding name containing "sparse" triggers the sparse path.
        let mut ctx = ExecutionContext::new();

        // Set up A in vector reg x4 (VectorReg 8,9), B in x0 (VectorReg 0,1).
        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(8, ones); // A in y2's x4 component
        ctx.vector.write_wide(0, ones); // B in x0

        // Explicitly zero mask q0 and accumulator cm0.
        // Zero mask means all B elements are masked out -> product = 0.
        ctx.mask.write_u32_low(0, 0);
        ctx.accumulator.write_wide(0, [0u64; 16]);

        // Config: i8xi8 sparse, zero_acc=1
        let conf = (1 << 3) | (5 << 5) | (1 << 8) | (1 << 9) | 1;
        ctx.scalar.write(5, conf);

        // Build a sparse MAC op:
        // Sources: VectorReg(8) [A=y2], ControlReg(28) [B=qx0], ScalarReg(5) [config]
        // Dest: AccumReg(0) [cm0]
        let mut op = SlotOp::from_semantic(SlotIndex::Accumulator, SemanticOp::Mac);
        op.is_vector = true;
        op.is_wide_vector = true;
        op.sources.push(Operand::VectorReg(8));   // A from y2
        op.sources.push(Operand::SparseQxReg(0));  // B from qx0
        op.sources.push(Operand::ScalarReg(5));     // config
        op.dest = Some(Operand::AccumReg(0));
        op.encoding_name = Some("VMAC_vmac_cm_core_sparse_wide".to_string());

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled);

        // With zero mask, zero_acc: all outputs should be 0.
        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(
                val, 0,
                "sparse output[{}] with zero mask should be 0, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_execute_matmul_sparse_with_mask() {
        // Test sparse MAC with a non-zero mask. Set mask to activate
        // the first byte of B, put a known value there.
        let mut ctx = ExecutionContext::new();

        // A: put value 3 at position [0][0] (byte 0), rest zero.
        let mut a_data = [0u32; 16];
        a_data[0] = 3; // A[0][0] = 3 (int8 at byte 0)
        ctx.vector.write_wide(8, a_data);

        // B: put value 7 at position [0][0] (byte 0), rest zero.
        let mut b_data = [0u32; 16];
        b_data[0] = 7; // B[0][0] = 7 (int8 at byte 0)
        ctx.vector.write_wide(0, b_data);

        // Set mask to activate only byte 0.
        ctx.mask.write_u32_low(0, 1); // q0 low word = 1 (bit 0 set)

        // Config: i8xi8 sparse, zero_acc=1, signed
        let conf = (1 << 3) | (5 << 5) | (1 << 8) | (1 << 9) | 1;
        ctx.scalar.write(5, conf);

        let mut op = SlotOp::from_semantic(SlotIndex::Accumulator, SemanticOp::Mac);
        op.is_vector = true;
        op.is_wide_vector = true;
        op.sources.push(Operand::VectorReg(8));
        op.sources.push(Operand::SparseQxReg(0));
        op.sources.push(Operand::ScalarReg(5));
        op.dest = Some(Operand::AccumReg(0));
        op.encoding_name = Some("VMAC_vmac_cm_core_sparse_wide".to_string());

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled);

        // C[0][0] = A[0][0] * B[0][0] = 3 * 7 = 21 (byte 0 of B is active).
        let acc = ctx.accumulator.read_wide(0);
        let val = (acc[0] & 0xFFFF_FFFF) as i32;
        assert_eq!(val, 21, "C[0][0] = 3 * 7 = 21, got {}", val);

        // All other outputs should be 0 (only one element was active).
        for i in 1..32 {
            let lane = i / 2;
            let half = i % 2;
            let v = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(v, 0, "output[{}] should be 0", i);
        }
    }
}

#[cfg(test)]
mod extract_bytes_tests {
    use super::*;

    #[test]
    fn test_extract_i8_signed() {
        let mut buf = [0u8; 128];
        buf[0] = 0xFF; // -1 as i8
        buf[1] = 0x7F; // 127
        buf[5] = 0x80; // -128
        assert_eq!(extract_element_bytes(&buf, 0, 8, true), -1);
        assert_eq!(extract_element_bytes(&buf, 1, 8, true), 127);
        assert_eq!(extract_element_bytes(&buf, 5, 8, true), -128);
    }

    #[test]
    fn test_extract_u8() {
        let mut buf = [0u8; 128];
        buf[0] = 0xFF;
        assert_eq!(extract_element_bytes(&buf, 0, 8, false), 255);
    }

    #[test]
    fn test_extract_i16_le() {
        let mut buf = [0u8; 128];
        // 0x0102 at byte 0 (LE: buf[0]=0x02, buf[1]=0x01)
        buf[0] = 0x02;
        buf[1] = 0x01;
        assert_eq!(extract_element_bytes(&buf, 0, 16, false), 0x0102);
        assert_eq!(extract_element_bytes(&buf, 0, 16, true), 0x0102);
    }

    #[test]
    fn test_extract_i16_signed_negative() {
        let mut buf = [0u8; 128];
        buf[4] = 0x00;
        buf[5] = 0x80; // 0x8000 = -32768
        assert_eq!(extract_element_bytes(&buf, 4, 16, true), -32768);
        assert_eq!(extract_element_bytes(&buf, 4, 16, false), 0x8000);
    }

    #[test]
    fn test_extract_4bit() {
        let mut buf = [0u8; 128];
        buf[0] = 0xBA; // low nibble = 0xA, high nibble = 0xB
        assert_eq!(extract_element_bytes(&buf, 0, 4, false), 0xA);
        assert_eq!(extract_element_bytes(&buf, 1, 4, false), 0xB);
        assert_eq!(extract_element_bytes(&buf, 0, 4, true), -6);
        assert_eq!(extract_element_bytes(&buf, 1, 4, true), -5);
    }

    #[test]
    fn test_extract_oob_returns_zero() {
        let buf = [0u8; 128];
        assert_eq!(extract_element_bytes(&buf, 200, 8, false), 0);
    }

    #[test]
    fn test_vec512_to_bytes() {
        let mut v: Vec512 = [0u32; 16];
        v[0] = 0x04030201;
        v[1] = 0x08070605;
        let bytes = vec512_to_bytes(&v);
        assert_eq!(bytes[0], 0x01);
        assert_eq!(bytes[1], 0x02);
        assert_eq!(bytes[2], 0x03);
        assert_eq!(bytes[3], 0x04);
        assert_eq!(bytes[4], 0x05);
        assert_eq!(bytes[7], 0x08);
    }
}

#[cfg(test)]
mod sparse_matmul_tests {
    use super::*;
    use crate::interpreter::execute::vector_config::MatMulConfig;

    #[test]
    fn test_sparse_i8xi8_identity() {
        // i8xi8 sparse: 4x16x8, acc32.
        // Cleanroom routing: col = compressed_byte_pos % 8.
        //
        // With A = all 1s (row 0, 16 inner positions) and mask 0x3 (bits 0,1):
        // Each compressed byte contributes its value to col = byte_pos % 8.
        //
        // To test col 0: bytes at positions 0, 8, 16, 24, 32, 40, 48, 56
        // each route to col 0. With all B = 1 and all A = 1:
        // col 0 sum = 8 (one contribution from each of 8 even-group first bytes)
        let mut a = [0u8; 128];
        for k in 0..16 {
            a[0 * 16 + k] = 1; // A row 0, all sparse inner positions = 1
        }
        let mut b = [1u8; 64]; // All compressed B bytes = 1
        let mask: u128 = 0x33333333_33333333_33333333_33333333;
        let config = MatMulConfig::from_config_word(
            (1 << 0) | (0 << 1) | (1 << 3) | (5 << 5) | (1 << 8) | (1 << 9),
            false,
        ).expect("valid sparse i8xi8 config");
        let mut acc = [0u64; 16];
        matmul_sparse_config_driven(&mut acc, &a, &b, mask, &config);
        // Each column gets 8 products (64 bytes / 8 cols = 8 per col), each 1*1=1.
        // acc_cmb=1 (Acc32): two 32-bit values per u64 lane.
        // Element index i -> u64_lane = i/2, half = i%2.
        for c in 0..8 {
            let u64_lane = c / 2;
            let half = c % 2;
            let val = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 8, "row 0 col {c}: 8 active bytes route here");
        }
    }

    #[test]
    #[ignore] // TODO: bf16 sparse needs word-level pair routing (byte-level swaps byte order)
    fn test_sparse_bf16_basic() {
        // bf16 sparse: 4x16x4, acc32(fp32).
        // inner_groups=4, cols=4. Group g=0 -> inner_group=0, col=0.
        // Mask bits 0,1 -> A sparse positions 0,1.
        //
        // This test is currently broken because the crossbar routing operates
        // at byte granularity, which swaps the byte order within bf16
        // elements. Bf16 sparse needs a word-level decompression function.
        let mut a = [0u8; 128];
        a[0] = 0x80; a[1] = 0x3F; // bf16 1.0 at A[row=0][sparse_k=0]
        a[2] = 0x80; a[3] = 0x3F; // bf16 1.0 at A[row=0][sparse_k=1]
        let mut b = [0u8; 64];
        b[0] = 0x00; b[1] = 0x40; // bf16 2.0 at B[0][0]
        let mask: u128 = 0x3;
        let config = MatMulConfig::from_config_word(
            (1 << 0) | (2 << 5),
            true,
        ).expect("valid sparse bf16 config");
        let mut acc = [0u64; 16];
        matmul_sparse_config_driven(&mut acc, &a, &b, mask, &config);
        let result_bits = (acc[0] & 0xFFFF_FFFF) as u32;
        let result = f32::from_bits(result_bits);
        assert_eq!(result, 2.0);
    }
}
