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
//! Geometries and config-word layout: llvm-aie `aiev2_vmult.h` (extracted at
//! build time, see `xdna_archspec::aie2::matmul`); tile shapes also appear in
//! mlir-aie `third_party/aie_api` (`mmul_*.hpp`).

mod helpers;

// Re-export legacy API used elsewhere.
pub use helpers::{matmul_dense, matmul_sub};

use helpers::*;

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::execute::vector_config::{AccWidth, MatMulConfig};
use crate::interpreter::state::{ExecutionContext, Vec512, Acc1024};
use xdna_archspec::aie2::isa::SemanticOp;

/// VMAC accumulator-source read latency in issued bundles.
///
/// II_VMAC / II_VMACf (AIE2Schedule.td) read their accumulator source at
/// operand cycle 3 ([5,3,1,1,1] / [6,3,1,1,1]): vector sources at issue,
/// acc source three cycles later. Peano chains VMUL -> VMAC at distance 2
/// (seed 5039 silicon: `vmul cm0` then `vmac cm0, cm0` two bundles later),
/// so the VMAC's acc read at +3 lands exactly on the VMUL's stage-5 write --
/// hardware forwards the fresh value. Sampling at issue read the poison
/// accumulator. Process AFTER commit_pending_writes (the producer's write
/// commits first), mirroring the forwarding behavior.
const VMAC_ACC_SRC_READ_LATENCY: u64 = 3;

/// A MAC-family instruction awaiting its stage-3 accumulator source read.
///
/// Vector sources, config word, and all control flags are sampled at issue
/// (operand cycle 1); only the accumulator sources are read at +3. The result
/// then carries the remaining write latency so the destination still commits
/// at the full MAC latency from issue.
#[derive(Debug, Clone)]
pub struct PendingMatmul {
    /// Dense A/B vector sources sampled at issue.
    dense: (Vec512, Vec512),
    config: MatMulConfig,
    /// Second accumulator merge: None, Some(false)=AddMac, Some(true)=SubMac.
    acc2_sub: Option<bool>,
    acc_src: u8,
    acc2_reg: u8,
    acc_dest: u8,
    is_half: bool,
    /// Remaining destination write latency (cycles) after the source read.
    write_latency: u64,
    /// Issued-bundle count at which the accumulator sources are read.
    pub ready_bundle: u64,
}

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

    // Sparse MAC (a SparseQxReg source is the structural signal that
    // distinguishes sparse from dense): not modeled.
    if op.sources.iter().any(|s| matches!(s, Operand::SparseQxReg(_))) {
        log::warn!("[NOT MODELED] sparse MAC ({:?}): awaiting clean-room derivation", op.encoding_name);
        return false;
    }

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

    // bf16/fp32 MAC (float accumulator): only the fresh multiply (vmul.f) is
    // modeled, because it is the form with silicon captures
    // (vec_mac_bf16{,_ovf}). The accumulating forms still need their own
    // silicon evidence for the fp32 accumulate step.
    if config.bfloat && semantic != SemanticOp::MatMul {
        log::warn!("[NOT MODELED] bf16 MAC ({:?}): awaiting clean-room derivation", op.encoding_name);
        return false;
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

    // Second accumulator merge mode (AddMac/SubMac). The acc2 REGISTER is
    // resolved at issue; its value is read with the deferred acc sources.
    let acc2_sub = match semantic {
        SemanticOp::AddMac => Some(false),
        SemanticOp::SubMac => Some(true),
        _ => None,
    };

    // Sample the vector sources and control flags at issue (operand cycle 1);
    // defer the accumulator source read and the multiply to +3.
    let dense = get_two_vec512(op, ctx);

    // Total result latency from issue: LATENCY_VECTOR_MAC = 5 (llvm-aie
    // II_VMAC/II_VMUL operand_cycles[0]=5). Deferring the write models the
    // AIE2 MAC pipeline: in a software-pipelined batch loop, the stores
    // draining a tile read the previous tile's accumulator while the next
    // tile's VMUL is still in flight. The acc sources are read at +3
    // (VMAC_ACC_SRC_READ_LATENCY); the write carries the remaining latency so
    // it still lands at issue+5.
    // Float MAC/MUL carries one extra float-normalization stage:
    // II_VMULf/II_VMACf operand_cycles[0] = 6 (LATENCY_VECTOR_MAC_F).
    let latency = if config.bfloat {
        crate::interpreter::timing::latency::LATENCY_VECTOR_MAC_F
    } else {
        crate::interpreter::timing::latency::LATENCY_VECTOR_MAC
    } as u64;

    ctx.queue_pending_matmul(PendingMatmul {
        dense,
        config,
        acc2_sub,
        acc_src,
        // acc2 register only exists for AddMac/SubMac; resolving it on other
        // forms would log a spurious missing-AccumReg error (VMUL has none).
        acc2_reg: if acc2_sub.is_some() { get_acc_source(op) } else { 0 },
        acc_dest,
        is_half,
        write_latency: latency.saturating_sub(VMAC_ACC_SRC_READ_LATENCY),
        ready_bundle: ctx.bundle_seq + VMAC_ACC_SRC_READ_LATENCY,
    });

    true
}

/// Read a full 1024-bit working accumulator from `reg` (bm half padded high).
fn read_acc_full(ctx: &ExecutionContext, reg: u8, is_half: bool) -> Acc1024 {
    if is_half {
        let half = ctx.accumulator.read(reg);
        let mut buf = [0u64; 16];
        buf[..8].copy_from_slice(&half);
        buf
    } else {
        ctx.accumulator.read_wide(reg & !1)
    }
}

/// Complete a deferred MAC-family instruction: read its accumulator sources
/// (operand cycle 3), perform the multiply, and queue the destination write
/// with the remaining latency.
fn complete_matmul(ctx: &mut ExecutionContext, p: &PendingMatmul) {
    let mut acc = read_acc_full(ctx, p.acc_src, p.is_half);

    // For AddMac/SubMac: read the second accumulator source. Integer acc2 is a
    // post-multiply add/sub.
    let acc2 = p.acc2_sub.map(|_| read_acc_full(ctx, p.acc2_reg, p.is_half));

    let (a, b) = &p.dense;
    matmul_config_driven(&mut acc, a, b, &p.config);

    // For integer AddMac/SubMac: merge acc2 AFTER the multiply.
    if let (Some(is_sub), Some(src_acc)) = (p.acc2_sub, acc2) {
        let n = if p.is_half { 8 } else { 16 };
        match p.config.acc_width {
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
                    acc[i] = if is_sub {
                        acc[i].wrapping_sub(src_acc[i])
                    } else {
                        acc[i].wrapping_add(src_acc[i])
                    };
                }
            }
        }
    }

    ctx.queue_matmul_accum_write(Operand::AccumReg(p.acc_dest), acc, p.is_half, p.write_latency);
}

/// Complete deferred MAC-family instructions whose stage-3 accumulator read
/// bundle arrived. Call once per issued bundle, after `commit_pending_writes`
/// (so a chained VMUL's write landing this cycle is visible -- hardware
/// forwarding) and before slot execution.
pub fn process_pending_matmuls(ctx: &mut ExecutionContext) {
    let ready = ctx.take_ready_pending_matmuls();
    for p in ready {
        complete_matmul(ctx, &p);
    }
}

/// Force-complete ALL deferred MAC-family instructions regardless of bundle
/// (used by force_commit_all_pending; the destination writes are queued and
/// drained by the caller's pending-write flush).
pub fn drain_pending_matmuls(ctx: &mut ExecutionContext) {
    let all = ctx.take_all_pending_matmuls();
    for p in all {
        complete_matmul(ctx, &p);
    }
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
/// The reads respect the op's per-operand forwarding context: II_VMAC reads
/// its vector sources at operand cycle 1 with NoBypass (AIE2Schedule.td
/// [5,3,1,1,1] / [VEC,VEC,No,No]), so a producer like `vmov wl0, wh0`
/// (latency 2) issued one bundle before the VMUL must still show the OLD
/// half -- Peano chains `vmov wl; vmul; vmac` expecting the mul to see the
/// first halves and the mac the second (seed 5038 silicon).
fn get_two_vec512(op: &SlotOp, ctx: &ExecutionContext) -> (Vec512, Vec512) {
    let mut vregs = op.sources.iter().enumerate().filter_map(|(pos, s)| {
        if let Operand::VectorReg(r) = s {
            Some((pos, *r))
        } else {
            None
        }
    });

    let (a_pos, a_reg) = vregs.next().unwrap_or_else(|| {
        log::error!("[MATMUL] missing first VectorReg source");
        (0, 0)
    });
    let (b_pos, b_reg) = vregs.next().unwrap_or_else(|| {
        log::error!("[MATMUL] missing second VectorReg source");
        (0, 0)
    });

    let read = |pos: usize, reg: u8| {
        // Ensure even alignment for wide read.
        let base = reg & !1;
        match op.source_forward.get(pos) {
            Some(&(uc, ub)) => ctx.vector.read_wide_with(base, uc, ub),
            None => ctx.vector.read_wide(base),
        }
    };
    (read(a_pos, a_reg), read(b_pos, b_reg))
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

/// Config-driven full-width integer matrix multiply.
///
/// Float (`config.bfloat`) configs are not modeled; callers gate them out
/// before reaching here.
pub fn matmul_config_driven(acc: &mut Acc1024, a: &Vec512, b: &Vec512, config: &MatMulConfig) {
    if config.bfloat {
        matmul_bf16_fresh(acc, a, b, config);
        return;
    }
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

/// Fresh bf16 x bf16 -> fp32 matrix multiply (`vmul.f`), `rows x inner x cols`
/// from the config (4x8x4 for the dense geometry), A row-major, B row-major.
///
/// Each output is one dot product of `inner` bf16 products. A bf16 x bf16
/// product is exact in f64 (16 significant bits), so the dot product is
/// accumulated in f64 and rounded once to fp32 (round to nearest even). Silicon
/// (vec_mac_bf16_ovf) shows the hardware does not round per product or per
/// partial sum in fp32: products that overflow fp32 on their own still cancel
/// to the finite true sum. Known gap: for tiles with wide exponent spread
/// silicon differs from the correctly rounded sum by a few ulps in some lanes
/// (see docs/known-fidelity-gaps.md); the exact internal alignment/truncation
/// is not yet derived. The destination is overwritten (no accumulator input).
fn matmul_bf16_fresh(acc: &mut Acc1024, a: &Vec512, b: &Vec512, config: &MatMulConfig) {
    let bf16 = |v: &Vec512, i: usize| -> f64 {
        let bits = (v[i / 2] >> (16 * (i % 2))) as u16;
        f32::from_bits((bits as u32) << 16) as f64
    };
    let (rows, inner, cols) = (config.rows as usize, config.inner as usize, config.cols as usize);
    *acc = [0u64; 16];
    for r in 0..rows {
        for c in 0..cols {
            // Fold from +0.0, not `sum()` (which starts at -0.0): on NPU1 silicon
            // a lane whose products are all -0 reads back +0 (vec_mac_bf16_ovf).
            let dot: f64 = (0..inner)
                .map(|k| bf16(a, r * inner + k) * bf16(b, k * cols + c))
                .fold(0.0, |s, p| s + p);
            let out = r * cols + c;
            let shift = 32 * (out % 2);
            acc[out / 2] |= ((dot as f32).to_bits() as u64) << shift;
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
    // Golden from xdna-emu-priv: tools/gen_vector_golden.py. Integer lanes computed by
    // independent integer multiply-accumulate (wrapped to acc width); bf16
    // cases are skipped (not modeled). The live
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

    fn load_matmul_golden() -> Option<Vec<MatmulCase>> {
        let path = crate::interpreter::execute::vector_validate::corpus_path("validate_matmul_golden")?;
        let data = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read golden {}: {}", path.display(), e));
        let v: serde_json::Value = serde_json::from_str(&data).expect("parse golden json");
        let arr = v.get("matmul").cloned().unwrap_or(serde_json::Value::Null);
        if arr.is_null() {
            return Some(Vec::new());
        }
        Some(serde_json::from_value(arr).expect("deserialize matmul cases"))
    }

    #[test]
    fn validate_matmul_golden() {
        let Some(cases) = load_matmul_golden() else { return };
        assert!(
            !cases.is_empty(),
            "matmul golden data missing -- regenerate it in xdna-emu-priv \
             (VECTOR_ORACLE_MODEL=... python3 tools/gen_vector_golden.py)"
        );

        let mut pass = 0usize;
        let mut fail = 0usize;
        let mut first_failures = Vec::new();

        for (ci, case) in cases.iter().enumerate() {
            // bf16 MAC golden lanes are skipped until bf16 MAC is modeled.
            if case.bfloat {
                continue;
            }
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
            matmul_config_driven(&mut acc, &a, &b, &config);

            let lanes = (case.rows * case.cols) as usize;
            for out_idx in 0..lanes {
                let expected = case.expected[out_idx];
                let matched = read_acc_wide(&acc, out_idx, config.acc_width) == expected;
                if matched {
                    pass += 1;
                } else {
                    fail += 1;
                    if first_failures.len() < 10 {
                        let actual: i64 = read_acc_wide(&acc, out_idx, config.acc_width);
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
    fn test_execute_matmul_bf16_not_modeled() {
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write_wide(0, [0u64; 16]);

        let ones = vec512_all_ones_bf16();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_bf16_accumulate());

        let mut op = make_mac_op(SemanticOp::Mac, 0, 2, 5, 0, Some("VMAC_F_vmac_bm_core_dense"));
        op.element_type = Some(ElementType::BFloat16);

        // Not handled: falls through to the interpreter's unhandled-instruction error.
        assert!(!execute_matmul(&op, &mut ctx));
        ctx.force_commit_all_pending();
        assert_eq!(ctx.accumulator.read_wide(0), [0u64; 16], "not-modeled op must not write the accumulator");
    }

    /// Pack bf16 bit patterns into a Vec512 (32 elements, lane i at bits 16*i).
    fn vec512_from_bf16(values: &[u16]) -> Vec512 {
        let mut v = [0u32; 16];
        for (i, &x) in values.iter().enumerate() {
            v[i / 2] |= (x as u32) << (16 * (i % 2));
        }
        v
    }

    fn fp32_lane(acc: &[u64; 8], i: usize) -> f32 {
        f32::from_bits((acc[i / 2] >> (32 * (i % 2))) as u32)
    }

    /// Fresh bf16 VMUL as the decoder labels it: `VMUL_F_vmac_bm_core_dense`,
    /// MatMul semantic, Float32 element type, bm destination.
    fn make_vmul_f_op() -> SlotOp {
        let mut op = make_mac_op(SemanticOp::MatMul, 0, 2, 5, 0, Some("VMUL_F_vmac_bm_core_dense"));
        op.element_type = Some(ElementType::Float32);
        op
    }

    #[test]
    fn test_vmul_f_fresh_bf16_tile_ignores_dirty_accumulator() {
        // 4x8x4 tile: A[r][k] = r+1, B[k][c] = c+1, so C[r][c] = 8(r+1)(c+1).
        // The destination starts as poison: VMUL.F has no accumulator input.
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write(0, [0xDEAD_BEEF_DEAD_BEEFu64; 8]);

        let bf16 = |x: f32| (x.to_bits() >> 16) as u16;
        let a: Vec<u16> = (0..32).map(|i| bf16((i / 8 + 1) as f32)).collect();
        let b: Vec<u16> = (0..32).map(|i| bf16((i % 4 + 1) as f32)).collect();
        ctx.vector.write_wide(0, vec512_from_bf16(&a));
        ctx.vector.write_wide(2, vec512_from_bf16(&b));
        ctx.scalar.write(5, config_bf16_accumulate());

        assert!(execute_matmul(&make_vmul_f_op(), &mut ctx), "vmul.f must dispatch to the matmul unit");
        ctx.force_commit_all_pending();

        let acc = ctx.accumulator.read(0);
        for r in 0..4 {
            for c in 0..4 {
                let expected = (8 * (r + 1) * (c + 1)) as f32;
                assert_eq!(fp32_lane(&acc, r * 4 + c), expected, "C[{r}][{c}]");
            }
        }
    }

    #[test]
    fn test_vmul_f_overflowing_products_cancel_exactly() {
        // bf16(2^70)^2 = 2^140 exceeds fp32 range as a standalone product, but
        // the +/- pair cancels inside the datapath and the true sum is 2^0 = 1.
        // Silicon (vec_mac_bf16_ovf) does not behave like per-product fp32
        // rounding followed by a sequential fp32 sum, which would give NaN.
        let mut ctx = ExecutionContext::new();
        let bf16 = |x: f32| (x.to_bits() >> 16) as u16;
        let big = bf16(2f32.powi(70));
        let neg_big = bf16(-(2f32.powi(70)));
        let one = bf16(1.0);
        let mut a = vec![0u16; 32];
        let mut b = vec![0u16; 32];
        // C[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0]
        a[0] = big;
        a[1] = neg_big;
        a[2] = one;
        b[0] = big;
        b[4] = big;
        b[8] = one;
        ctx.vector.write_wide(0, vec512_from_bf16(&a));
        ctx.vector.write_wide(2, vec512_from_bf16(&b));
        ctx.scalar.write(5, config_bf16_accumulate());

        assert!(execute_matmul(&make_vmul_f_op(), &mut ctx));
        ctx.force_commit_all_pending();
        assert_eq!(fp32_lane(&ctx.accumulator.read(0), 0), 1.0);
    }

    #[test]
    fn test_vmul_f_result_latency_is_float_mac_latency() {
        // Float vector MAC/MUL results land one cycle later than integer ones
        // (II_VMULf operand_cycles[0] = 6 vs 5).
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write(0, [0u64; 8]);
        ctx.vector.write_wide(0, vec512_all_ones_bf16());
        ctx.vector.write_wide(2, vec512_all_ones_bf16());
        ctx.scalar.write(5, config_bf16_accumulate());
        assert!(execute_matmul(&make_vmul_f_op(), &mut ctx));
        for _ in 0..5 {
            step_bundle(&mut ctx);
        }
        assert_eq!(fp32_lane(&ctx.accumulator.read(0), 0), 0.0, "not yet visible at +5");
        step_bundle(&mut ctx);
        assert_eq!(fp32_lane(&ctx.accumulator.read(0), 0), 8.0, "visible at +6");
    }

    /// Run the silicon-captured vec_mac_bf16 kernels through the VMUL.F path
    /// and compare lanes against NPU1 silicon (tools/golden/silicon).
    ///
    /// Returns (lanes compared, lanes bit-exact, worst bit-pattern distance,
    /// tiles with a mismatch). Lanes with a non-finite input are skipped:
    /// silicon's NaN sign/payload encoding is not modeled.
    fn bf16_silicon_lanes(kernel: &str) -> (usize, usize, u32, Vec<usize>) {
        let path = format!("{}/tools/golden/silicon/{kernel}.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let col = |i: usize| -> Vec<u64> {
            v["inputs"][i].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
        };
        let (a_all, b_all) = (col(0), col(1));
        let silicon: Vec<u32> = v["silicon"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let finite = |x: u64| (x >> 7) & 0xff != 0xff;

        let (mut compared, mut exact, mut worst, mut inexact_tiles) = (0, 0, 0u32, Vec::new());
        for tile in 0..silicon.len() / 16 {
            let a: Vec<u16> = a_all[tile * 32..tile * 32 + 32].iter().map(|&x| x as u16).collect();
            let b: Vec<u16> = b_all[tile * 32..tile * 32 + 32].iter().map(|&x| x as u16).collect();
            let mut ctx = ExecutionContext::new();
            ctx.vector.write_wide(0, vec512_from_bf16(&a));
            ctx.vector.write_wide(2, vec512_from_bf16(&b));
            ctx.scalar.write(5, config_bf16_accumulate());
            assert!(execute_matmul(&make_vmul_f_op(), &mut ctx));
            ctx.force_commit_all_pending();
            let acc = ctx.accumulator.read(0);
            for lane in 0..16 {
                let (r, c) = (lane / 4, lane % 4);
                let inputs_finite =
                    (0..8).all(|k| finite(a[r * 8 + k] as u64) && finite(b[k * 4 + c] as u64));
                if !inputs_finite {
                    continue;
                }
                let got = fp32_lane(&acc, lane).to_bits();
                let want = silicon[tile * 16 + lane];
                compared += 1;
                if got == want {
                    exact += 1;
                } else {
                    worst = worst.max(got.abs_diff(want));
                    inexact_tiles.push(tile);
                }
            }
        }
        inexact_tiles.dedup();
        (compared, exact, worst, inexact_tiles)
    }

    #[test]
    fn test_vmul_f_matches_silicon_on_narrow_exponent_tiles() {
        // Tiles 0-5 of vec_mac_bf16 (uniform, identity-like, narrow spread)
        // are bit-exact against NPU1 silicon.
        let (compared, exact, _, inexact_tiles) = bf16_silicon_lanes("vec_mac_bf16");
        assert!(compared > 0);
        assert!(
            inexact_tiles.iter().all(|&t| t >= 6),
            "narrow-spread tiles must match silicon: {inexact_tiles:?}"
        );
        assert!(exact >= 96);
    }

    #[test]
    fn test_vmul_f_known_gap_wide_exponent_tiles_within_few_ulps() {
        // KNOWN FIDELITY GAP (docs/known-fidelity-gaps.md): on tiles with wide
        // exponent spread silicon differs from the correctly rounded dot
        // product, almost always by 1 ulp (79 of 85 lanes in vec_mac_bf16),
        // never more than 8 ulps in these captures. Pin that envelope so a
        // regression is visible; tighten it when the datapath is derived.
        // vec_mac_bf16_ovf is not pinned here: its designed summation-order and
        // cancellation tiles (e.g. 2^30 + 1 - 2^30) expose the underived
        // alignment/truncation outright and are the input to that derivation.
        for kernel in ["vec_mac_bf16"] {
            let (compared, exact, worst, _) = bf16_silicon_lanes(kernel);
            assert!(worst <= 8, "{kernel}: worst ulp distance {worst}");
            assert!(exact * 100 >= compared * 75, "{kernel}: only {exact}/{compared} lanes exact");
        }
    }

    /// Step the context the way the cycle-accurate loop does: advance both
    /// clocks, commit pending writes, then complete ready deferred matmuls.
    fn step_bundle(ctx: &mut ExecutionContext) {
        ctx.cycles += 1;
        ctx.bundle_seq += 1;
        ctx.commit_pending_writes();
        process_pending_matmuls(ctx);
    }

    #[test]
    fn test_vmac_acc_source_read_at_stage3_sees_chained_vmul() {
        // Peano chains `vmul cm0, x0, x2` then `vmac cm0, cm0, x0, x2` at
        // distance 2 (seed 5039 silicon, fuzz banked 2026-06-10). II_VMAC
        // reads its accumulator source at operand cycle 3, exactly when the
        // VMUL's stage-5 write lands -- hardware forwards the fresh value.
        // Reading at issue saw the poison accumulator (all banked mmac seeds
        // showed 0xDEADBEEF + mac contribution on EMU vs mul+mac on silicon).
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write_wide(0, [0xDEAD_BEEF_DEAD_BEEFu64; 16]);

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_i8xi8_accumulate());

        // Bundle 0: vmul (fresh multiply, zeroes acc).
        let mul = make_mac_op(SemanticOp::MatMul, 0, 2, 5, 0, Some("VMUL_vmac_cm_core_dense"));
        assert!(execute_matmul(&mul, &mut ctx));

        step_bundle(&mut ctx);
        step_bundle(&mut ctx);

        // Bundle 2: vmac chained on the same accumulator.
        let mac = make_mac_op(SemanticOp::Mac, 0, 2, 5, 0, Some("VMAC_vmac_cm_core_dense"));
        assert!(execute_matmul(&mac, &mut ctx));

        for _ in 0..8 {
            step_bundle(&mut ctx);
        }

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 16, "output[{i}]: chained vmul+vmac must be 8+8=16, got {val}");
        }
    }

    #[test]
    fn test_vmul_vec_sources_read_at_cycle1_nobypass() {
        // II_VMAC reads its vector sources at operand cycle 1 with NoBypass.
        // Peano puts `vmov wl0, wh0` (latency 2) one bundle before the VMUL:
        // hardware's VMUL still sees the OLD low half; only the VMAC two
        // bundles later sees the new one (seed 5038 silicon). The default
        // compute-read (use_cycle 2, Mov) makes the in-flight vmov visible a
        // bundle early and feeds the wrong matrix into the mul.
        use crate::interpreter::state::Bypass;
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write_wide(0, [0u64; 16]);
        ctx.vector.write_wide(0, vec512_all_ones_i8());
        ctx.vector.write_wide(2, vec512_all_ones_i8());
        ctx.scalar.write(5, config_i8xi8_accumulate());

        // Bundle 1: vmov-like in-flight write of 2s to wl0 (l_def 2, NoBypass).
        ctx.bundle_seq = 1;
        ctx.vector.advance_bundle(1);
        ctx.vector.queue_write(0, [0x02020202u32; 8], 2, Bypass::No);

        // Bundle 2: vmul issues, reading its vec sources at (1, NoBypass).
        ctx.cycles = 2;
        ctx.bundle_seq = 2;
        ctx.vector.advance_bundle(2);
        let mut op = make_mac_op(SemanticOp::MatMul, 0, 2, 5, 0, Some("VMUL_vmac_cm_core_dense"));
        op.source_forward.push((1, Bypass::No));
        op.source_forward.push((1, Bypass::No));
        op.source_forward.push((1, Bypass::No));
        assert!(execute_matmul(&op, &mut ctx));
        ctx.force_commit_all_pending();

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(val, 8, "output[{i}]: vmul must read OLD wl0 (all ones -> 8), got {val}");
        }
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
    fn test_execute_matmul_sparse_not_modeled() {
        let mut ctx = ExecutionContext::new();
        ctx.vector.write_wide(8, vec512_all_ones_i8());
        ctx.accumulator.write_wide(0, [0u64; 16]);
        ctx.scalar.write(5, (1 << 3) | (5 << 5) | (1 << 8) | (1 << 9) | 1);

        let mut op = SlotOp::from_semantic(SlotIndex::Accumulator, SemanticOp::Mac);
        op.is_vector = true;
        op.is_wide_vector = true;
        op.sources.push(Operand::VectorReg(8));
        op.sources.push(Operand::SparseQxReg(0));
        op.sources.push(Operand::ScalarReg(5));
        op.dest = Some(Operand::AccumReg(0));
        op.encoding_name = Some("VMAC_vmac_cm_core_sparse_wide".to_string());

        // Not handled: falls through to the interpreter's unhandled-instruction error.
        assert!(!execute_matmul(&op, &mut ctx));
        ctx.force_commit_all_pending();
        assert_eq!(ctx.accumulator.read_wide(0), [0u64; 16], "not-modeled op must not write the accumulator");
    }
}
