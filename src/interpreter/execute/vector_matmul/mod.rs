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

mod bf16_pipeline;
mod helpers;

// Re-export the bf16 pipeline for vmac_hw.
pub(crate) use bf16_pipeline::bf16_mac_hw_lane;

// Re-export legacy API used elsewhere.
pub use helpers::{matmul_dense, matmul_sub};

use helpers::*;

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::execute::vector_config::{AccWidth, MatMulConfig};
use crate::interpreter::state::{ExecutionContext, Vec512, Acc1024};
use xdna_archspec::aie2::isa::SemanticOp;

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
            log::error!("[MATMUL] no config register found in sources for {:?}", op.encoding_name);
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

    // Fresh-multiply forms zero the accumulator regardless of the config word.
    //
    // VMUL (-> MatMul) and VNEGMUL (-> NegMul) encode acc1 = 0b1111, i.e. NO
    // accumulator input -- the products land in a freshly-zeroed accumulator.
    // This is structural to the instruction, distinct from the config word's
    // dynZeroAccum bit (which only dynamically zeroes the VMAC-family). Chess
    // emits the fresh VMUL with a config word that has zero_acc=0 (accumulate),
    // so honoring only the config word would leak the destination accumulator's
    // stale/poison contents into the result. Mac/MatMulSub/NegMatMul/AddMac/
    // SubMac (the VMAC family) keep the config-driven accumulate behavior.
    if matches!(semantic, SemanticOp::MatMul | SemanticOp::NegMul) {
        config.accumulate = false;
    }

    // Handle subtract semantics for MAC variants.
    //
    // The hardware compressor stage controls the sign of the product via a
    // subtract flag.  Several instruction families flip this flag:
    //
    //   vmac:     result = acc + A*B        (subtract=false)
    //   vmsc:     result = acc - A*B        (subtract=true, negates product)
    //   vnegmac:  result = -(acc + A*B)     (NegMul semantic)
    //   vnegmsc:  result = acc - A*B        (same sign as VMSC per HW observation)
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

    // MSC name-based correction for the dense path's config.subtract:
    //
    // VNEGMSC (NegMul + MSC): NegMul flipped subtract above, but the MSC
    // and NEG product negations cancel. Un-flip for VNEGMSC.
    //
    // VADDMSC/VSUBMSC (AddMac/SubMac + MSC): the subtract flag wasn't
    // flipped above (AddMac/SubMac not in the match), but MSC needs it.
    if let Some(ref name) = op.encoding_name {
        if name.contains("msc")
            && matches!(
                semantic,
                SemanticOp::NegMul | SemanticOp::NegMatMul | SemanticOp::AddMac | SemanticOp::SubMac
            )
        {
            config.subtract = !config.subtract;
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
    let (acc_dest, is_half) = get_acc_dest(op);

    // The initial accumulator value is read from the FIRST AccumReg source
    // (acc1), NOT from the destination register.  VMAC_F/VMSC_F etc. have
    // separate $dst and $acc1 fields -- when they differ, the hardware reads
    // from acc1 and writes to dst.  For tied instructions (dst == acc1), the
    // first AccumReg source IS the same register as dest.
    let acc_src = op
        .sources
        .iter()
        .find_map(|s| {
            if let Operand::AccumReg(r) = s {
                Some(*r)
            } else {
                None
            }
        })
        .unwrap_or(acc_dest);

    let mut acc = if is_half {
        // bm (512-bit): read single register, pad to 1024-bit working buffer.
        let half = ctx.accumulator.read(acc_src);
        let mut buf = [0u64; 16];
        buf[..8].copy_from_slice(&half);
        buf
    } else {
        // cm (1024-bit): read wide pair.
        ctx.accumulator.read_wide(acc_src & !1)
    };

    // For AddMac/SubMac: read the second accumulator source.
    //
    // In bfloat mode, acc2 participates in the 68-bit PSA adder BEFORE
    // normalization (hardware: psal_hw merges products + acc1 + acc2, then
    // bfnorm_hw normalizes). We pass acc2 into matmul_config_driven so
    // bf16_mac_hw_lane can include it in the pre-normalization sum.
    //
    // In integer mode, acc2 is a simple post-multiply add/sub.
    let (acc2_for_bfloat, is_sub_acc2) = match semantic {
        SemanticOp::AddMac | SemanticOp::SubMac if config.bfloat => {
            let src_reg = get_acc_source(op);
            let src_acc = if is_half {
                let half = ctx.accumulator.read(src_reg);
                let mut buf = [0u64; 16];
                buf[..8].copy_from_slice(&half);
                buf
            } else {
                ctx.accumulator.read_wide(src_reg & !1)
            };
            (Some(src_acc), semantic == SemanticOp::SubMac)
        }
        _ => (None, false),
    };

    // Read input vectors and perform multiply.
    if is_sparse {
        let (a_bytes, b_register, mask) = get_sparse_operands(op, ctx);

        // Determine sub0/sub1/sub2 flags from instruction encoding bits.
        //
        // The AIE2 vec slot encodes MAC variants via two bits (mdm, md1):
        //   mdm=0, md1=0 -> VMAC  (acc + products)
        //   mdm=1, md1=0 -> VMSC  (acc - products)
        //   mdm=1, md1=1 -> VNEGMAC (-(acc + products))
        //   mdm=0, md1=1 -> VNEGMSC (-(acc) - products)
        //
        // In the C++ ISS vec_control function:
        //   sub0 -> XOR into subtract_mul (product negation)
        //   sub1 -> XOR into subtract_acc (accumulator negation)
        //   sub2 -> XOR into subtract_acc bit 1 (scd negation)
        //
        // The sub0/sub1/sub2 flags are derived from the SemanticOp, which
        // is resolved at build time from instruction names:
        //   VMAC     -> Mac        -> sub0=0, sub1=0
        //   VMSC     -> MatMulSub  -> sub0=1, sub1=0  (product negate)
        //   VNEGMAC  -> NegMatMul  -> sub0=1, sub1=1  (product + acc negate)
        //   VNEGMSC  -> MatMulSub  -> sub0=1, sub1=0  (product negate, same as VMSC)
        //
        // is_msc XOR is_neg gives the correct mdm bit for all variants.
        // VNEGMSC gets MatMulSub (not NegMul) because HW testing shows
        // it produces acc - products (same sign as VMSC).
        let is_msc = op.encoding_name.as_ref().map(|n| n.contains("msc")).unwrap_or(false);
        let is_neg = matches!(semantic, SemanticOp::NegMul | SemanticOp::NegMatMul);
        let sub0 = is_msc ^ is_neg;
        let sub1 = is_neg;
        let sub2 = matches!(semantic, SemanticOp::SubMac);

        // For AddMac/SubMac with bfloat, acc2 participates in the PSA
        // adder. Pass it as scd to sparse_vmac.
        let scd = match (semantic, config.bfloat) {
            (SemanticOp::AddMac | SemanticOp::SubMac, true) => {
                let src_reg = get_acc_source(op);
                if is_half {
                    let half = ctx.accumulator.read(src_reg);
                    let mut buf = [0u64; 16];
                    buf[..8].copy_from_slice(&half);
                    buf
                } else {
                    ctx.accumulator.read_wide(src_reg & !1)
                }
            }
            _ => [0u64; 16],
        };

        // Use the hardware-faithful vmac pipeline.
        acc =
            super::vmac_hw::sparse_vmac(&a_bytes, &b_register, mask, &acc, &scd, conf_val, sub0, sub1, sub2);
    } else {
        let (a, b) = get_two_vec512(op, ctx);
        matmul_config_driven(&mut acc, &a, &b, &config, acc2_for_bfloat.as_ref(), is_sub_acc2);
    }

    // For integer AddMac/SubMac: merge acc2 AFTER the multiply.
    // (Bfloat AddMac/SubMac is handled inside bf16_mac_hw_lane above.)
    match semantic {
        SemanticOp::AddMac | SemanticOp::SubMac if !config.bfloat => {
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
        _ => {}
    }

    // Write the result back to the DESTINATION accumulator with the MAC result
    // latency (LATENCY_VECTOR_MAC = 5, from llvm-aie II_VMAC/II_VMUL
    // operand_cycles[0]=5). Deferring the write models the AIE2 MAC pipeline: in
    // a software-pipelined batch loop, the stores draining a tile read the
    // previous tile's accumulator while the next tile's VMUL is still in flight.
    // Writing immediately makes the result visible too early and shifts every
    // tile one ahead. `is_half` selects the 512-bit bm-half vs 1024-bit cm-pair
    // write inside the deferred apply.
    // Float (bf16/fp32) MAC has a longer result latency than integer (6 vs 5)
    // for the normalization stage; using the integer latency on a bf16
    // software-pipelined loop shifts every stored tile by one (II_VMACf /
    // II_VMULf operand_cycles[0]=6 vs II_VMAC/II_VMUL=5).
    let latency = if config.bfloat {
        crate::interpreter::timing::latency::LATENCY_VECTOR_MAC_F
    } else {
        crate::interpreter::timing::latency::LATENCY_VECTOR_MAC
    } as u64;
    ctx.queue_matmul_accum_write(Operand::AccumReg(acc_dest), acc, is_half, latency);

    true
}

// ---------------------------------------------------------------------------
// Private helpers for execute_matmul
// ---------------------------------------------------------------------------

/// Scan sources for the last ScalarReg operand (the config register).
fn get_config_reg(op: &SlotOp, ctx: &ExecutionContext) -> Option<u32> {
    for src in op.sources.iter().rev() {
        if let Operand::ScalarReg(r) = src {
            return Some(ctx.scalar_read(*r));
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
    let mut vregs = op.sources.iter().filter_map(|s| {
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
        Some(crate::interpreter::decode::register_map::AccumWidth::Half)
        | Some(crate::interpreter::decode::register_map::AccumWidth::QuarterLow)
        | Some(crate::interpreter::decode::register_map::AccumWidth::QuarterHigh) => true,
        Some(crate::interpreter::decode::register_map::AccumWidth::Full) => false,
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
            log::error!("[MATMUL] expected AccumReg dest, got {:?} -- defaulting to cm0", other);
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
    log::error!("[MATMUL] no AccumReg found in sources -- defaulting to cm0");
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

/// Sparse config-driven matrix multiply.
///
/// `a` is a 128-byte buffer (1024 bits) containing the DENSE operand (xs1).
/// For narrow sparse, only the first 64 bytes are populated; the rest are zero.
/// Layout: `A[r, k] = a[(r * inner + k) * bytes_x]` (row-major i16 or i8).
///
/// `b` is a 64-byte buffer (512 bits) containing COMPRESSED sparse data from
/// the qxs2 register. The buffer has 32 groups of 2 bytes each. Group `g`
/// contains bytes `b[2*g]` and `b[2*g+1]`. Both bytes in a group route to
/// the SAME inner_k but different output columns.
///
/// `mask` is the 128-bit sparsity mask. It has 32 nibbles (4 bits each),
/// indexed sequentially: nibble `g` = `(mask >> (4*g)) & 0xF`. Each nibble
/// selects which 2 of 4 inner positions within a group are active.
///
/// Groups are organized in blocks of 4, and blocks alternate between the
/// two active bits of the mask nibble:
///
///   block = g / 4
///   ig = block / 2              (inner_group, 0..inner_groups-1)
///   active_bit_idx = block % 2  (0 = first active bit, 1 = second)
///   inner_k = ig * 4 + nth_set_bit(nibble_g, active_bit_idx)
///
/// Column routing is positional: `col = (g*2 + comp_idx) % cols`.
///
/// Cleanroom: NPU crossbar sweep characterization (2026-04-02).
/// Hardware broadcast stage (prmx_bcst_hw) is a no-op (type reinterpretation),
/// confirmed from me_inline_primitives.h line 11784.
pub fn matmul_sparse_config_driven(
    acc: &mut Acc1024,
    a: &[u8; 128],
    b: &[u8; 64],
    mask: u128,
    config: &MatMulConfig,
) {
    let rows = config.rows as usize;
    let cols = config.cols as usize;
    let inner = config.inner as usize;
    let bits_x = config.bits_x;
    let bits_y = config.bits_y;
    let bytes_x = if bits_x == 4 { 1 } else { (bits_x / 8) as usize };

    // 32 groups, each consuming 2 compressed bytes.
    let num_groups: usize = 32;

    // Pad compressed B to 128 bytes for extract_element_bytes compatibility.
    let mut b_pad = [0u8; 128];
    b_pad[..64].copy_from_slice(b);

    if !config.accumulate {
        *acc = [0u64; 16];
    }

    // Helper: get the n-th set bit position (0-indexed) from a 4-bit nibble.
    #[inline]
    fn nth_set_bit(nibble: u8, n: usize) -> Option<usize> {
        let mut count = 0;
        for b in 0..4u8 {
            if (nibble >> b) & 1 != 0 {
                if count == n {
                    return Some(b as usize);
                }
                count += 1;
            }
        }
        None
    }

    if config.bfloat {
        // Bf16 sparse: 2 bytes per B element. Groups consume 4 bytes each
        // (2 bf16 elements). The inner_k derivation is the same as integer.
        for g in 0..num_groups {
            let mask4 = ((mask >> (4 * g)) & 0xF) as u8;
            if mask4.count_ones() > 2 {
                continue;
            }

            let block = g / 4;
            let ig = block / 2;
            let active_bit_idx = block % 2;

            let bit_pos = match nth_set_bit(mask4, active_bit_idx) {
                Some(b) => b,
                None => continue,
            };
            let inner_k = ig * 4 + bit_pos;

            // Each group has 2 compressed bf16 elements (4 bytes total).
            for comp_idx in 0..2usize {
                let b_elem = g * 2 + comp_idx;
                let b_off = b_elem * 2;
                if b_off + 1 >= 64 {
                    continue;
                }
                let col = b_elem % cols;

                let b_bits = u16::from_le_bytes([b[b_off], b[b_off + 1]]);
                let b_val = f32::from_bits((b_bits as u32) << 16);

                for r in 0..rows {
                    let a_off = (r * inner + inner_k) * 2;
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
            }
        }
        return;
    }

    // Integer sparse path.
    //
    // Iterate over 32 groups. Each group's mask nibble determines the
    // inner_k via the block-based formula. Both compressed bytes in the
    // group share the same inner_k but route to different output columns.
    for g in 0..num_groups {
        let mask4 = ((mask >> (4 * g)) & 0xF) as u8;
        // Hardware decode_mask: >2 set bits maps to no-route.
        if mask4.count_ones() > 2 {
            continue;
        }

        // Block-based inner_k derivation (cleanroom, 2026-04-02).
        let block = g / 4;
        let ig = block / 2;
        let active_bit_idx = block % 2;

        let bit_pos = match nth_set_bit(mask4, active_bit_idx) {
            Some(b) => b,
            None => continue,
        };
        let inner_k = ig * 4 + bit_pos;

        // Both compressed bytes in this group share the same inner_k.
        for comp_idx in 0..2usize {
            let b_byte_pos = g * 2 + comp_idx;
            let col = b_byte_pos % cols;

            let b_byte = if bits_y == 4 {
                b_byte_pos // nibble-packed: extract_element_bytes handles lo/hi
            } else {
                b_byte_pos * (bits_y as usize / 8)
            };
            let b_val = extract_element_bytes(&b_pad, b_byte, bits_y, config.y_signed);

            for r in 0..rows {
                let a_byte = if bits_x == 4 {
                    r * inner + inner_k
                } else {
                    (r * inner + inner_k) * bytes_x
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
        }
    }
}

/// Config-driven full-width matrix multiply.
///
/// `acc2_data` is an optional second accumulator for AddMac/SubMac bfloat
/// operations. When present, each lane's acc2 value is merged into the
/// 68-bit PSA adder BEFORE normalization (matching hardware pipeline).
/// `sub_acc2` controls whether acc2 is subtracted (SubMac) or added (AddMac).
pub fn matmul_config_driven(
    acc: &mut Acc1024,
    a: &Vec512,
    b: &Vec512,
    config: &MatMulConfig,
    acc2_data: Option<&Acc1024>,
    sub_acc2: bool,
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
                    let elem_idx = if is_elemwise { r + k * 16 } else { r * inner + k };
                    let a_word = elem_idx / 2;
                    let a_half = elem_idx % 2;

                    let b_elem_idx = if is_elemwise { r + k * 16 } else { k * cols + c };
                    let b_word = b_elem_idx / 2;
                    let b_half = b_elem_idx % 2;

                    a_elems.push(((a[a_word] >> (a_half * 16)) & 0xFFFF) as u16);
                    b_elems.push(((b[b_word] >> (b_half * 16)) & 0xFFFF) as u16);
                }

                let prev_bits = read_acc_wide_f32(acc, out_idx).to_bits();
                let lane_acc2 = acc2_data.map(|a2| read_acc_wide_f32(a2, out_idx).to_bits());
                let result =
                    bf16_mac_hw_lane(prev_bits, &a_elems, &b_elems, config.subtract, lane_acc2, sub_acc2);
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
// Legacy type-specific matmul functions
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
/// Geometry: A[4][2] * B[2][4] = C[4][4], 16 output int32 values.
fn matmul_i16xi16_32(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
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
/// Geometry: A[4][4] * B[4][4] = C[4][4], 16 output fp32 values.
fn matmul_bf16xbf16(acc: &mut [u64; 8], a: &[u32; 8], b: &[u32; 8], subtract: bool) {
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
/// Geometry: A[4][2] * B[2][2] = C[4][2], 8 output int64 values.
fn matmul_i32xi16(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
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

    // -----------------------------------------------------------------------
    // MatMul value differential vs independent oracles
    //
    // Golden from tools/gen_vector_golden.py: integer lanes computed by
    // independent integer multiply-accumulate (wrapped to acc width); bf16
    // lanes computed by the genuine aietools bfloat16.py:bf16_mac_hw. The live
    // seam (matmul_config_driven) is driven via MatMulConfig::from_types, and
    // the geometry from_types yields is cross-checked against the golden's
    // geometry fields. No aietools dependency at test time.
    // -----------------------------------------------------------------------

    #[derive(serde::Deserialize)]
    struct MatmulCase {
        a_type: String,
        b_type: String,
        rows: u32,
        inner: u32,
        cols: u32,
        bits_x: u32,
        bits_y: u32,
        acc_width: String,
        bfloat: bool,
        x_signed: bool,
        y_signed: bool,
        subtract: bool,
        a: Vec<u32>,
        b: Vec<u32>,
        /// Per-lane expected: int lanes as i64; bf16 lanes as fp32 bit pattern.
        expected: Vec<i64>,
    }

    fn parse_elem_type(s: &str) -> ElementType {
        match s {
            "Int8" => ElementType::Int8,
            "Int16" => ElementType::Int16,
            "Int32" => ElementType::Int32,
            "BFloat16" => ElementType::BFloat16,
            other => panic!("unhandled matmul element type {other}"),
        }
    }

    /// True if an fp32 bit pattern is NaN (exp all-ones, mantissa nonzero).
    fn fp32_is_nan_bits(bits: u32) -> bool {
        (bits >> 23) & 0xFF == 0xFF && (bits & 0x7F_FFFF) != 0
    }

    fn load_matmul_golden() -> Vec<MatmulCase> {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tools");
        path.push("golden");
        path.push("vector_ops.json");
        let data = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read golden {}: {}", path.display(), e));
        let v: serde_json::Value = serde_json::from_str(&data).expect("parse golden json");
        let arr = v.get("matmul").cloned().unwrap_or(serde_json::Value::Null);
        if arr.is_null() {
            return Vec::new();
        }
        serde_json::from_value(arr).expect("deserialize matmul cases")
    }

    #[test]
    fn validate_matmul_golden() {
        let cases = load_matmul_golden();
        assert!(
            !cases.is_empty(),
            "matmul golden data missing -- regenerate tools/golden/vector_ops.json \
             (VECTOR_ORACLE_MODEL=... python3 tools/gen_vector_golden.py)"
        );

        let mut pass = 0usize;
        let mut fail = 0usize;
        let mut first_failures = Vec::new();

        for (ci, case) in cases.iter().enumerate() {
            let a_type = parse_elem_type(&case.a_type);
            let b_type = parse_elem_type(&case.b_type);
            let config = MatMulConfig::from_types(
                a_type,
                b_type,
                /*accumulate=*/ false,
                case.x_signed,
                case.y_signed,
                case.subtract,
            )
            .unwrap_or_else(|| panic!("from_types returned None for {}x{}", case.a_type, case.b_type));

            // Cross-check: the geometry from_types selects must match the
            // independently-encoded golden geometry.
            let want_aw = if case.acc_width == "Acc64" {
                AccWidth::Acc64
            } else {
                AccWidth::Acc32
            };
            assert_eq!(
                (
                    config.rows,
                    config.inner,
                    config.cols,
                    config.bits_x,
                    config.bits_y,
                    config.acc_width,
                    config.bfloat
                ),
                (case.rows, case.inner, case.cols, case.bits_x, case.bits_y, want_aw, case.bfloat),
                "geometry mismatch for {}x{} (case {ci})",
                case.a_type,
                case.b_type,
            );

            let mut a = [0u32; 16];
            let mut b = [0u32; 16];
            a.copy_from_slice(&case.a);
            b.copy_from_slice(&case.b);
            let mut acc: Acc1024 = [0u64; 16];
            matmul_config_driven(&mut acc, &a, &b, &config, None, false);

            let lanes = (case.rows * case.cols) as usize;
            for out_idx in 0..lanes {
                let expected = case.expected[out_idx];
                let matched = if case.bfloat {
                    // bf16 lanes are fp32 bit patterns. The emulator matches the
                    // genuine model bit-exact on all finite + inf results, but
                    // diverges on NaN canonicalization: the aietools model uses
                    // NaN mantissa 0x7F, while real NPU1 silicon (and thus the
                    // emulator) uses mantissa = 1. That is a documented
                    // model-vs-silicon gap where the emulator is HW-correct, so
                    // NaN-vs-NaN counts as a match (payload divergence ignored).
                    let act = read_acc_wide_f32(&acc, out_idx).to_bits();
                    let exp = expected as u32;
                    if fp32_is_nan_bits(exp) {
                        fp32_is_nan_bits(act)
                    } else {
                        act == exp
                    }
                } else {
                    read_acc_wide(&acc, out_idx, config.acc_width) == expected
                };
                if matched {
                    pass += 1;
                } else {
                    fail += 1;
                    if first_failures.len() < 10 {
                        let actual: i64 = if case.bfloat {
                            read_acc_wide_f32(&acc, out_idx).to_bits() as i64
                        } else {
                            read_acc_wide(&acc, out_idx, config.acc_width)
                        };
                        first_failures.push(format!(
                            "MATMUL MISMATCH {}x{} (case {ci}) lane {out_idx}: expected {expected}, actual {actual}",
                            case.a_type, case.b_type,
                        ));
                    }
                }
            }
        }

        for msg in &first_failures {
            eprintln!("{msg}");
        }
        eprintln!("MatMul: {pass} pass, {fail} fail across {} cases", cases.len());
        assert_eq!(fail, 0, "MatMul validation: {fail} lane(s) failed");
    }

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
        let mut a_vals = [0i8; 32];
        a_vals[0] = 1;
        a_vals[8 + 1] = 1;
        a_vals[16 + 2] = 1;
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

        for r in 0..4 {
            for c in 0..4 {
                let out_idx = r * 4 + c;
                let expected = (r * 4 + c + 1) as i64;
                let actual = read_acc32(&acc, out_idx);
                assert_eq!(actual, expected, "C[{}][{}]: expected {}, got {}", r, c, expected, actual);
            }
        }
    }

    #[test]
    fn test_i8xi8_all_ones() {
        let a_vals = [1i8; 32];
        let b_vals = [1i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);

        for r in 0..4 {
            for c in 0..4 {
                let out_idx = r * 4 + c;
                assert_eq!(read_acc32(&acc, out_idx), 8, "C[{}][{}] should be 8", r, c);
            }
        }
    }

    #[test]
    fn test_i8xi8_accumulate() {
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
                    r,
                    c
                );
            }
        }
    }

    #[test]
    fn test_i8xi8_subtract() {
        let a_vals = [2i8; 32];
        let b_vals = [3i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        for idx in 0..16 {
            assert_eq!(read_acc32(&acc, idx), 48);
        }

        matmul_i8xi8(&mut acc, &a, &b, true, true, true);
        for idx in 0..16 {
            assert_eq!(read_acc32(&acc, idx), 0);
        }
    }

    #[test]
    fn test_i8xi8_signed_negative() {
        let mut a_vals = [0i8; 32];
        let mut b_vals = [0i8; 32];

        a_vals[0] = -1;
        b_vals[0] = -2;
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
        let mut a_vals = [0i8; 32];
        let mut b_vals = [0i8; 32];

        a_vals[0] = -56; // 200 as u8
        b_vals[0] = -56; // 200 as u8

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        assert_eq!(read_acc32(&acc, 0), 3136);

        let mut acc2 = [0u64; 8];
        matmul_i8xi8(&mut acc2, &a, &b, false, false, false);
        assert_eq!(read_acc32(&acc2, 0), 40000);
    }

    // -----------------------------------------------------------------------
    // int16 x int16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i16xi16_identity() {
        let mut a_vals = [0i16; 16];
        a_vals[0] = 1;
        a_vals[3] = 1;

        let mut b_vals = [0i16; 16];
        b_vals[0] = 10;
        b_vals[1] = 20;
        b_vals[2] = 30;
        b_vals[3] = 40;
        b_vals[4] = 50;
        b_vals[5] = 60;
        b_vals[6] = 70;
        b_vals[7] = 80;

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i16xi16_32(&mut acc, &a, &b, true, true, false);

        assert_eq!(read_acc32(&acc, 0), 10);
        assert_eq!(read_acc32(&acc, 1), 20);
        assert_eq!(read_acc32(&acc, 2), 30);
        assert_eq!(read_acc32(&acc, 3), 40);
        assert_eq!(read_acc32(&acc, 4), 50);
        assert_eq!(read_acc32(&acc, 5), 60);
        assert_eq!(read_acc32(&acc, 6), 70);
        assert_eq!(read_acc32(&acc, 7), 80);
    }

    #[test]
    fn test_i16xi16_multiply() {
        let a_vals: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0];
        let b_vals: [i16; 16] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i16xi16_32(&mut acc, &a, &b, true, true, false);

        assert_eq!(read_acc32(&acc, 0), 1);
        assert_eq!(read_acc32(&acc, 1), 2);
        assert_eq!(read_acc32(&acc, 4), 3);
        assert_eq!(read_acc32(&acc, 5), 4);
        assert_eq!(read_acc32(&acc, 8), 5);
        assert_eq!(read_acc32(&acc, 9), 6);
        assert_eq!(read_acc32(&acc, 12), 7);
        assert_eq!(read_acc32(&acc, 13), 8);
    }

    #[test]
    fn test_i16xi16_accumulate_subtract() {
        let a_vals: [i16; 16] = [10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b_vals: [i16; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i16xi16_32(&mut acc, &a, &b, true, true, false);
        assert_eq!(read_acc32(&acc, 0), 10);

        matmul_i16xi16_32(&mut acc, &a, &b, true, true, true);
        assert_eq!(read_acc32(&acc, 0), 0);
    }

    // -----------------------------------------------------------------------
    // bf16 x bf16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_identity() {
        let mut a_bits = [0u16; 16];
        let one = f32_to_bf16_bits(1.0);
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
                    r,
                    c,
                    expected,
                    actual
                );
            }
        }
    }

    #[test]
    fn test_bf16_all_ones() {
        let one = f32_to_bf16_bits(1.0);
        let a_bits = [one; 16];
        let b_bits = [one; 16];

        let a = pack_bf16(&a_bits);
        let b = pack_bf16(&b_bits);
        let mut acc = [0u64; 8];

        matmul_bf16xbf16(&mut acc, &a, &b, false);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!((actual - 4.0).abs() < 0.01, "Output[{}]: expected 4.0, got {}", idx, actual);
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

        matmul_bf16xbf16(&mut acc, &a, &b, false);
        matmul_bf16xbf16(&mut acc, &a, &b, false);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!((actual - 8.0).abs() < 0.01, "Output[{}]: expected 8.0, got {}", idx, actual);
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

        matmul_bf16xbf16(&mut acc, &a, &b, false);
        matmul_bf16xbf16(&mut acc, &a, &b, true);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!(actual.abs() < 0.01, "Output[{}]: expected 0.0, got {}", idx, actual);
        }
    }

    // -----------------------------------------------------------------------
    // int32 x int16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i32xi16_basic() {
        let mut a_packed = [0u32; 8];
        a_packed[0] = 100;
        a_packed[1] = 200;
        a_packed[2] = 300;
        a_packed[3] = 400;

        let b_vals: [i16; 16] = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b = pack_i16(&b_vals);

        let mut acc = [0u64; 8];
        matmul_i32xi16(&mut acc, &a_packed, &b, true, true, false);

        assert_eq!(read_acc64(&acc, 0), 100);
        assert_eq!(read_acc64(&acc, 1), 200);
        assert_eq!(read_acc64(&acc, 2), 300);
        assert_eq!(read_acc64(&acc, 3), 400);
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

        let mut acc = [0u64; 8];
        write_acc32(&mut acc, 0, 100);

        matmul_sub(&mut acc, &a, &b, ElementType::Int16, true, true);

        assert_eq!(read_acc32(&acc, 0), 95);
    }

    #[test]
    fn test_matmul_dense_bf16() {
        let two = f32_to_bf16_bits(2.0);
        let three = f32_to_bf16_bits(3.0);

        let a = pack_bf16(&[two; 16]);
        let b = pack_bf16(&[three; 16]);
        let mut acc = [0u64; 8];

        matmul_dense(&mut acc, &a, &b, ElementType::BFloat16, true, true);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!((actual - 24.0).abs() < 0.1, "Output[{}]: expected 24.0, got {}", idx, actual);
        }
    }

    // -----------------------------------------------------------------------
    // execute_matmul entry point tests
    // -----------------------------------------------------------------------

    use crate::interpreter::bundle::SlotIndex;
    use crate::interpreter::state::ExecutionContext;

    /// Build a minimal SlotOp for a MAC-family instruction.
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

    /// Pack [u32; 16] (Vec512) of all-ones bytes (int8 = 1 in every byte).
    fn vec512_all_ones_i8() -> Vec512 {
        [0x01010101u32; 16]
    }

    /// Pack Vec512 of all bf16(1.0) values.
    fn vec512_all_ones_bf16() -> Vec512 {
        let word = 0x3F80_3F80u32;
        [word; 16]
    }

    /// Build a config word for int8xi8 mode.
    fn config_i8xi8_accumulate() -> u32 {
        (1 << 3) | (1 << 8) | (1 << 9)
    }

    /// Build a config word for bf16 mode.
    fn config_bf16_accumulate() -> u32 {
        0x00
    }

    /// Build a config word for int8xi8 with zero_acc=1.
    fn config_i8xi8_zero_acc() -> u32 {
        (1 << 3) | (1 << 8) | (1 << 9) | 1
    }

    #[test]
    fn test_execute_matmul_i8xi8_ones() {
        let mut ctx = ExecutionContext::new();

        ctx.accumulator.write_wide(0, [0u64; 16]);

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);

        ctx.scalar.write(5, config_i8xi8_accumulate());

        let op = make_mac_op(SemanticOp::Mac, 0, 2, 5, 0, None);

        let handled = execute_matmul(&op, &mut ctx);
        ctx.force_commit_all_pending(); // commit the deferred MAC-latency write
        assert!(handled, "execute_matmul should handle Mac semantic");

        let acc = ctx.accumulator.read_wide(0);

        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 8, "output[{}]: expected 8, got {}", i, val);
        }
    }

    #[test]
    fn test_matmul_fresh_zeroes_dirty_accumulator() {
        // VMUL (mnemonic "vmul", MatMul semantic) is a FRESH matrix multiply:
        // acc1 is hardcoded 0b1111 (no accumulator input), so the destination
        // accumulator must be zeroed before the multiply regardless of the
        // config word's zero_acc bit. The compiled VMUL config word has
        // zero_acc=0 (accumulate), so the fresh-zero must come from the
        // semantic, not the config. Reproduces the MAC-tier poison bug: a dirty
        // accumulator must NOT leak into the result.
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write_wide(0, [0xDEAD_BEEF_DEAD_BEEFu64; 16]);

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        // Config word says ACCUMULATE (bit 0 = 0), exactly like Chess emits.
        ctx.scalar.write(5, config_i8xi8_accumulate());

        let op = make_mac_op(SemanticOp::MatMul, 0, 2, 5, 0, Some("VMUL_vmac_cm_core_dense"));
        let handled = execute_matmul(&op, &mut ctx);
        ctx.force_commit_all_pending(); // commit the deferred MAC-latency write
        assert!(handled, "execute_matmul should handle MatMul semantic");

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 8, "output[{}]: fresh A.B must be 8, not poison+A.B (got {})", i, val);
        }
    }

    #[test]
    fn test_matmul_mac_still_accumulates_after_fresh_fix() {
        // Regression guard: the fresh-zero must apply ONLY to MatMul/NegMul, not
        // to Mac. With a pre-loaded accumulator (10 per lane) and accumulate
        // config, VMAC must add: 10 + 8 = 18.
        let mut ctx = ExecutionContext::new();
        let mut pre = [0u64; 16];
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            pre[lane] |= (10u64 & 0xFFFF_FFFF) << (half * 32);
        }
        ctx.accumulator.write_wide(0, pre);

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_i8xi8_accumulate());

        let op = make_mac_op(SemanticOp::Mac, 0, 2, 5, 0, Some("VMAC_vmac_cm_core_dense"));
        execute_matmul(&op, &mut ctx);
        ctx.force_commit_all_pending(); // commit the deferred MAC-latency write

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 18, "Mac must accumulate 10 + 8 = 18, got {}", val);
        }
    }

    #[test]
    fn test_execute_matmul_bf16_ones() {
        let mut ctx = ExecutionContext::new();

        ctx.accumulator.write_wide(0, [0u64; 16]);

        let ones = vec512_all_ones_bf16();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_bf16_accumulate());

        let mut op = make_mac_op(SemanticOp::Mac, 0, 2, 5, 0, Some("VMAC_F_vmac_bm_core_dense"));
        op.element_type = Some(ElementType::BFloat16);

        let handled = execute_matmul(&op, &mut ctx);
        ctx.force_commit_all_pending(); // commit the deferred MAC-latency write
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..16 {
            let lane = i / 2;
            let half = i % 2;
            let bits = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
            let val = f32::from_bits(bits);
            assert!((val - 8.0).abs() < 0.01, "output[{}]: expected 8.0, got {}", i, val);
        }
    }

    #[test]
    fn test_matmul_bf16_uses_float_latency_six() {
        // Float (bf16) vector MAC has result latency 6 (llvm-aie II_VMULf /
        // II_VMACf operand_cycles[0]=6), one cycle longer than integer (5)
        // because of the extra float-normalization pipeline stage
        // (EmptyCycles<4> vs <3> in AIE2Schedule.td). The deferred accumulator
        // write must use the FLOAT latency: Chess software-pipelines bf16 batch
        // loops one stage deeper than integer (a 3-deep VMUL.f prologue vs
        // 2-deep), so a write committed at issue+5 makes the next tile's result
        // visible too early and shifts every stored tile by one.
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write_wide(0, [0u64; 16]);

        let ones = vec512_all_ones_bf16();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_bf16_accumulate());

        let mut op = make_mac_op(SemanticOp::MatMul, 0, 2, 5, 0, Some("VMUL_F_vmac_bm_core_dense"));
        op.element_type = Some(ElementType::Float32);

        ctx.cycles = 100;
        assert!(execute_matmul(&op, &mut ctx));

        // At issue+5 the float result must NOT yet be visible (integer latency
        // would have committed here; float latency 6 must not).
        ctx.cycles = 105;
        ctx.commit_pending_writes();
        let bits5 = (ctx.accumulator.read_wide(0)[0] & 0xFFFF_FFFF) as u32;
        assert_eq!(bits5, 0, "bf16 matmul result must NOT be visible at issue+5 (float latency is 6)");

        // At issue+6 it commits.
        ctx.cycles = 106;
        ctx.commit_pending_writes();
        let bits6 = (ctx.accumulator.read_wide(0)[0] & 0xFFFF_FFFF) as u32;
        let v = f32::from_bits(bits6);
        assert!((v - 8.0).abs() < 0.01, "bf16 result must be visible at issue+6, got {}", v);
    }

    #[test]
    fn test_execute_matmul_negate() {
        let mut ctx = ExecutionContext::new();

        ctx.accumulator.write_wide(0, [0u64; 16]);

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_i8xi8_accumulate());

        let op = make_mac_op(SemanticOp::NegMul, 0, 2, 5, 0, None);

        let handled = execute_matmul(&op, &mut ctx);
        ctx.force_commit_all_pending(); // commit the deferred MAC-latency write
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as u32 as i32;
            assert_eq!(val, -8, "output[{}]: expected -8, got {}", i, val);
        }
    }

    #[test]
    fn test_execute_matmul_zero_acc() {
        let mut ctx = ExecutionContext::new();

        let mut preload = [0u64; 16];
        for i in 0..16 {
            preload[i] = 0xDEAD_BEEF_CAFE_BABEu64;
        }
        ctx.accumulator.write_wide(0, preload);

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_i8xi8_zero_acc());

        let op = make_mac_op(SemanticOp::Mac, 0, 2, 5, 0, None);

        let handled = execute_matmul(&op, &mut ctx);
        ctx.force_commit_all_pending(); // commit the deferred MAC-latency write
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 8, "output[{}]: expected 8 (zero_acc should clear), got {}", i, val);
        }
    }

    #[test]
    fn test_execute_matmul_returns_false_for_non_mac() {
        let mut ctx = ExecutionContext::new();
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add);
        assert!(!execute_matmul(&op, &mut ctx));
    }

    // -----------------------------------------------------------------------
    // Sparse config-driven tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_config_driven_all_zero_b_produces_zero() {
        let mut a = [0u8; 128];
        for b in a.iter_mut() {
            *b = 1;
        }
        let b = [0u8; 64];
        let mask: u128 = 0x33333333_33333333_33333333_33333333;
        let mut acc = [0u64; 16];

        let config =
            MatMulConfig::from_config_word((1 << 3) | (5 << 5) | (1 << 8) | (1 << 9) | 1, false).unwrap();
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
        let a = [1u8; 128];
        let b = [1u8; 64];
        let mask: u128 = 0x33333333_33333333_33333333_33333333;
        let mut acc = [0u64; 16];

        let config =
            MatMulConfig::from_config_word((1 << 3) | (5 << 5) | (1 << 8) | (1 << 9) | 1, false).unwrap();

        matmul_sparse_config_driven(&mut acc, &a, &b, mask, &config);

        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 8, "output[{}]: expected 8, got {}", i, val);
        }
    }

    #[test]
    fn test_execute_matmul_sparse_via_encoding_name() {
        let mut ctx = ExecutionContext::new();

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(8, ones);
        ctx.vector.write_wide(0, ones);

        ctx.mask.write_u32_low(0, 0);
        ctx.accumulator.write_wide(0, [0u64; 16]);

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
        ctx.force_commit_all_pending(); // commit the deferred MAC-latency write
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 0, "sparse output[{}] with zero mask should be 0, got {}", i, val);
        }
    }

    #[test]
    #[ignore = "superseded by vmac_hw oracle tests; old formula model assumptions don't match hardware routing"]
    fn test_execute_matmul_sparse_with_mask() {
        let mut ctx = ExecutionContext::new();

        let mut a_data = [0u32; 16];
        a_data[0] = 3;
        ctx.vector.write_wide(8, a_data);

        let mut b_data = [0u32; 16];
        b_data[0] = 7;
        ctx.vector.write_wide(0, b_data);

        ctx.mask.write_u32_low(0, 1);

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
        ctx.force_commit_all_pending(); // commit the deferred MAC-latency write
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        let val = (acc[0] & 0xFFFF_FFFF) as i32;
        assert_eq!(val, 21, "C[0][0] = 3 * 7 = 21, got {}", val);

        for i in 1..32 {
            let lane = i / 2;
            let half = i % 2;
            let v = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(v, 0, "output[{}] should be 0", i);
        }
    }
}

#[cfg(test)]
mod sparse_matmul_tests {
    use super::*;
    use crate::interpreter::execute::vector_config::MatMulConfig;

    #[test]
    fn test_sparse_i8xi8_identity() {
        let mut a = [0u8; 128];
        for k in 0..16 {
            a[0 * 16 + k] = 1;
        }
        let b = [1u8; 64];
        let mask: u128 = 0x33333333_33333333_33333333_33333333;
        let config = MatMulConfig::from_config_word(
            (1 << 0) | (0 << 1) | (1 << 3) | (5 << 5) | (1 << 8) | (1 << 9),
            false,
        )
        .expect("valid sparse i8xi8 config");
        let mut acc = [0u64; 16];
        matmul_sparse_config_driven(&mut acc, &a, &b, mask, &config);
        for c in 0..8 {
            let u64_lane = c / 2;
            let half = c % 2;
            let val = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 8, "row 0 col {c}: 8 active bytes route here");
        }
    }

    #[test]
    fn test_sparse_bf16_basic() {
        let mut a = [0u8; 128];
        a[0] = 0x80;
        a[1] = 0x3F;
        a[2] = 0x80;
        a[3] = 0x3F;
        let mut b = [0u8; 64];
        b[0] = 0x00;
        b[1] = 0x40;
        let mask: u128 = 0x3;
        let config =
            MatMulConfig::from_config_word((1 << 0) | (2 << 5), true).expect("valid sparse bf16 config");
        let mut acc = [0u64; 16];
        matmul_sparse_config_driven(&mut acc, &a, &b, mask, &config);
        let result_bits = (acc[0] & 0xFFFF_FFFF) as u32;
        let result = f32::from_bits(result_bits);
        assert_eq!(result, 2.0);
    }
}
