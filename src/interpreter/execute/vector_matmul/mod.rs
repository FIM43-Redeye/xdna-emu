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
#[cfg(test)]
mod mode_goldens;
mod sparse;
#[cfg(test)]
mod sparse_goldens;
pub mod bf16;

// Re-export legacy API used elsewhere.
pub use helpers::{matmul_dense, matmul_sub};

use helpers::*;
use sparse::{Image, SparseCombine};

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::execute::vector_config::{AccWidth, MatMulConfig, Multiplier};
use crate::interpreter::state::{ExecutionContext, Vec512, Acc1024};
use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::aie2::matmul::MacOpKind;

/// A MAC-family instruction awaiting its accumulator source read.
///
/// Vector sources, config word, and all control flags are sampled at issue
/// (operand cycle 1); the accumulator sources are read at their itinerary use cycle
/// (II_VMAC/II_VMACf/II_VADDMAC `operand_cycles[1..]`, 3 for the whole family). The result is
/// timed from the issue: it carries the issue bundle, the itinerary result operand cycle and the
/// result bypass class, and `AccumulatorRegisterFile` decides per reader when it is visible.
#[derive(Debug, Clone)]
pub struct PendingMatmul {
    /// A and B vector images sampled at issue (a dense operand fills the low 512 bits; a
    /// sparse B is the logical matrix expanded from its stored data and mask).
    operands: (Image, Image),
    /// Present for sparse integer MACs, whose terms are combined by [`sparse::combine_integer`].
    sparse_integer: Option<SparseCombine>,
    /// For a sparse float MAC, the B elements the mask keeps (bit `k * cols + c`); the products
    /// of the others do not take part at all.
    b_present: Option<u128>,
    config: MatMulConfig,
    /// Second accumulator merge: None, Some(false)=AddMac, Some(true)=SubMac.
    acc2_sub: Option<bool>,
    /// Dense integer vnegmac/vnegmsc: the whole result is negated, so the accumulator input
    /// enters negated (silicon: -(acc + A*B) and -(acc - A*B)).
    negate_acc: bool,
    acc_src: u8,
    acc2_reg: u8,
    acc_dest: u8,
    is_half: bool,
    /// Bundle the instruction issued in, its result operand cycle and its result bypass class.
    issue_bundle: u64,
    l_def: u8,
    def_bypass: crate::interpreter::state::Bypass,
    /// Bypass class of the accumulator sources (acc1, acc2).
    acc_use_bypass: [crate::interpreter::state::Bypass; 2],
    /// Issued-bundle count at which the accumulator sources are read.
    pub ready_bundle: u64,
    /// Write tag of the instruction, restored when the destination is queued.
    tag: Option<crate::interpreter::state::write_collision::WriteTag>,
    /// Accumulator read port of the instruction, for the late-read steal of acc2.
    acc_read: Option<crate::interpreter::bundle::AccReadPort>,
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

    // Parse config word into its lane map and modes. A word whose
    // (amode, bmode, variant) names no geometry is reported unhandled; claiming
    // the op handled would leave its destination silently unwritten.
    let mut config = match MatMulConfig::from_config_word(conf_val, is_bf16) {
        Some(c) => c,
        None => {
            log::warn!(
                "[NOT MODELED] MAC config word 0x{:08x} (bf16={}) for {:?}: no geometry for its \
                 (amode, bmode, variant)",
                conf_val,
                is_bf16,
                op.encoding_name
            );
            return false;
        }
    };

    // A `qx` source and a sparse geometry go together: the encoding supplies the stored data
    // and mask, the config word the layout. Either without the other cannot be decoded, so the
    // op is reported unhandled rather than computed on a guess.
    let qx = op.sources.iter().enumerate().find_map(|(pos, s)| match s {
        Operand::SparseQxReg(n) => Some((pos, *n)),
        _ => None,
    });
    // A sparse geometry on a dense instruction has no sparse operand to read: the multiplier
    // outputs zero products (silicon, `docs/fidelity-gaps/mac-conf.md`).
    if qx.is_none() && config.sparse {
        config = MatMulConfig { sparse: false, ..MatMulConfig::idle((conf_val >> 1) & 3) };
        config.accumulate = (conf_val & 1) == 0;
        config.x_signed = (conf_val >> 9) & 1 != 0;
        config.y_signed = (conf_val >> 8) & 1 != 0;
    }
    if qx.is_some() != config.sparse {
        log::warn!(
            "[NOT MODELED] {:?}: sparse operand {} but config word 0x{:08x} selects a {} geometry",
            op.encoding_name,
            if qx.is_some() { "present" } else { "absent" },
            conf_val,
            if config.sparse { "sparse" } else { "dense" }
        );
        return false;
    }

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
    //
    // vnegmac (NegMatMul) and vnegmsc (NegMul) do take an accumulator: dense integer silicon gives
    // -(acc + A*B) and -(acc - A*B) (collisions/neg), so the accumulator input is negated and only
    // the fresh vnegmul zeroes it. The float and sparse forms keep the old rule.
    let negate_acc = matches!(semantic, SemanticOp::NegMul | SemanticOp::NegMatMul)
        && qx.is_none()
        && !config.bfloat
        && op.encoding_name.as_deref().is_some_and(|n| {
            let n = n.to_ascii_lowercase();
            n.contains("negmac") || n.contains("negmsc")
        });
    if matches!(semantic, SemanticOp::MatMul | SemanticOp::NegMul) && !negate_acc {
        config.accumulate = false;
    }

    // bf16 MAC (float accumulator): the forms with silicon evidence
    // (vec_mac_bf16_forms: vmul/vnegmul/vmac/vmsc/vaddmac/vaddmsc, dense and
    // elementwise). vnegmac.f, vsubmac.f and vsubmsc.f have no capture.
    if config.bfloat
        && !matches!(
            semantic,
            SemanticOp::MatMul
                | SemanticOp::NegMul
                | SemanticOp::Mac
                | SemanticOp::MatMulSub
                | SemanticOp::AddMac
        )
    {
        log::warn!("[NOT MODELED] bf16 MAC ({:?}): no silicon evidence for this form", op.encoding_name);
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
        // Encoding names are upper-case TableGen names (VADDMSC_...).
        if name.to_ascii_lowercase().contains("msc")
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
    let (operands, sparse_integer, b_present) = match qx {
        None => {
            let (a, b) = get_two_vec512(op, ctx);
            // Every integer MAC, dense or sparse, is a zero-accumulator multiply followed by the
            // same post-adder (`sparse::combine_integer`).
            let combine = if config.bfloat {
                None
            } else {
                match integer_combine(op, semantic, conf_val) {
                    Some(c) => Some(c),
                    None => {
                        log::warn!("[NOT MODELED] {:?}: integer MAC with no op kind", op.encoding_name);
                        return false;
                    }
                }
            };
            ((widen(&a), widen(&b)), combine, None)
        }
        Some((qx_pos, qx_index)) => {
            match sample_sparse(op, ctx, &config, conf_val, is_bf16, qx_pos, qx_index) {
                Some(sampled) => sampled,
                None => return false,
            }
        }
    };

    // Timing (docs/fidelity-gaps/vector-compute.md, "Accumulator read/write timing"): the
    // result is visible from issue + def + 1 (def: the itinerary result operand cycle, 5 for the
    // integer MAC family and 6 for the float one), one bundle earlier to a reader on the same
    // bypass class; the accumulator sources are read at their itinerary use cycle. Dense and
    // sparse forms share their itineraries and so their timing. An op without itinerary data
    // (hand-built) falls back to the AM020 MAC latencies.
    let l_def = if ctx.result_latency > 0 {
        ctx.result_latency
    } else if config.bfloat {
        crate::interpreter::timing::latency::LATENCY_VECTOR_MAC_F
    } else {
        crate::interpreter::timing::latency::LATENCY_VECTOR_MAC
    };
    let (use1, bypass1) = op.acc_source_timing(0);
    let (use2, bypass2) = if acc2_sub.is_some() {
        op.acc_source_timing(1)
    } else {
        (use1, bypass1)
    };

    let pending = PendingMatmul {
        operands,
        sparse_integer,
        b_present,
        config,
        acc2_sub,
        negate_acc,
        acc_src,
        // acc2 register only exists for AddMac/SubMac; resolving it on other
        // forms would log a spurious missing-AccumReg error (VMUL has none).
        acc2_reg: if acc2_sub.is_some() { get_acc_source(op) } else { 0 },
        acc_dest,
        is_half,
        issue_bundle: ctx.bundle_seq,
        l_def,
        def_bypass: ctx.result_bypass,
        acc_use_bypass: [bypass1, bypass2],
        ready_bundle: ctx.bundle_seq + use1.max(use2) as u64,
        tag: ctx.write_tag(),
        acc_read: ctx.slot_acc_read(),
    };
    ctx.queue_pending_matmul(pending);

    true
}

/// Read a full 1024-bit working accumulator from `reg` (bm half padded high).
fn read_acc_full(
    ctx: &ExecutionContext,
    reg: u8,
    is_half: bool,
    bypass: crate::interpreter::state::Bypass,
) -> Acc1024 {
    // The use cycle has elapsed since issue: read at the current bundle.
    if is_half {
        let half = ctx.accumulator.read_with(reg, 0, bypass);
        let mut buf = [0u64; 16];
        buf[..8].copy_from_slice(&half);
        buf
    } else {
        ctx.accumulator.read_wide_with(reg & !1, 0, bypass)
    }
}

/// Read the second accumulator source of a deferred AddMac/SubMac. It goes through the
/// accumulator read port, which a mover reading in the same cycle takes over.
fn read_acc2(ctx: &ExecutionContext, p: &PendingMatmul) -> Acc1024 {
    let mut acc2 = read_acc_full(ctx, p.acc2_reg, p.is_half, p.acc_use_bypass[1]);
    if let Some(port) = p.acc_read {
        ctx.steal_acc_read(port, p.issue_bundle, p.acc2_reg, !p.is_half, &mut acc2);
    }
    acc2
}

/// Queue the destination write of a completed MAC, timed from its issue.
fn queue_result(ctx: &mut ExecutionContext, p: &PendingMatmul, value: Acc1024) {
    ctx.set_write_tag(p.tag);
    ctx.queue_acc_write_timed(
        p.issue_bundle,
        Operand::AccumReg(p.acc_dest),
        value,
        p.is_half,
        p.l_def,
        p.def_bypass,
    );
    ctx.set_write_tag(None);
}

/// Complete a deferred MAC-family instruction: read its accumulator sources
/// (at their use cycle), perform the multiply, and queue the destination write
/// timed from the issue.
fn complete_matmul(ctx: &mut ExecutionContext, p: &PendingMatmul) {
    let mut acc = read_acc_full(ctx, p.acc_src, p.is_half, p.acc_use_bypass[0]);

    // For AddMac/SubMac: read the second accumulator source. Integer acc2 is a
    // post-multiply add/sub.
    let acc2 = p.acc2_sub.map(|_| read_acc2(ctx, p));

    let (a, b) = &p.operands;
    if let Some(combine) = &p.sparse_integer {
        // Sparse integer: the multiply is the dense one on a zero accumulator, the terms are
        // combined by the sparse rule (the product sign, shift16 and both accumulator signs).
        let mut product = [0u64; 16];
        let mut product_config = p.config;
        product_config.accumulate = false;
        product_config.subtract = false;
        matmul_config_driven(&mut product, &a[..], &b[..], &product_config);
        let acc1 = read_acc_full(ctx, p.acc_src, p.is_half, p.acc_use_bypass[0]);
        let acc2 = p.acc2_sub.map(|_| read_acc2(ctx, p));
        let result = sparse::combine_integer(
            &acc1,
            acc2.as_ref(),
            &product,
            p.config.shape.lanes() as usize,
            p.config.acc_width,
            combine,
        );
        queue_result(ctx, p, result);
        return;
    }
    if p.config.bfloat {
        // The float accumulators join the product sum (one aligned sum, one
        // rounding), so acc2 is an input here rather than a post-add.
        matmul_bf16(&mut acc, acc2.as_ref(), &a[..], &b[..], &p.config, p.b_present);
        queue_result(ctx, p, acc);
        return;
    }
    if p.negate_acc {
        match p.config.acc_width {
            AccWidth::Acc32 => acc.iter_mut().for_each(|w| {
                *w = ((*w as u32 as i32).wrapping_neg() as u32 as u64)
                    | (((*w >> 32) as u32 as i32).wrapping_neg() as u32 as u64) << 32
            }),
            AccWidth::Acc64 => acc.iter_mut().for_each(|w| *w = (*w as i64).wrapping_neg() as u64),
        }
    }
    matmul_config_driven(&mut acc, &a[..], &b[..], &p.config);

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

    queue_result(ctx, p, acc);
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

/// A 512-bit dense operand as the low half of an image.
fn widen(v: &Vec512) -> Image {
    let mut image = [0u32; 32];
    image[..16].copy_from_slice(v);
    image
}

/// Sample the operands of a sparse MAC at issue: A from an x register (narrow) or a y register
/// (wide), B expanded from the `qx` register's stored data and mask.
///
/// Also returns, for integer modes, the term combine. `None` when the geometry has no sparse
/// layout or the op has no MAC kind (the caller reports the op unhandled).
fn sample_sparse(
    op: &SlotOp,
    ctx: &ExecutionContext,
    config: &MatMulConfig,
    conf_val: u32,
    is_bf16: bool,
    qx_pos: usize,
    qx_index: u8,
) -> Option<((Image, Image), Option<SparseCombine>, Option<u128>)> {
    let layout = sparse::layout_of(conf_val, is_bf16)?;
    let name = op.encoding_name.as_deref().unwrap_or("");
    let wide = name.ends_with("sparse_wide") || (op.is_quad_vector && !name.ends_with("sparse_narrow"));
    let (a_pos, a_reg) = op.sources.iter().enumerate().find_map(|(pos, s)| match s {
        Operand::VectorReg(r) => Some((pos, *r)),
        _ => None,
    })?;
    let read = |pos: usize, base: u8| match op.source_forward.get(pos) {
        Some(&(uc, ub)) => ctx.vector.read_wide_with(base, uc, ub),
        None => ctx.vector.read_wide(base),
    };
    let mut a = [0u32; 32];
    if wide {
        let base = a_reg & !3;
        a[..16].copy_from_slice(&read(a_pos, base));
        a[16..].copy_from_slice(&read(a_pos, base + 2));
    } else {
        a[..16].copy_from_slice(&read(a_pos, a_reg & !1));
    }
    // qx_n = {x_n, q_n}: the stored data is the x register, the mask the q register.
    let data = read(qx_pos, 2 * qx_index);
    let mask = ctx.mask.read_u128(qx_index);
    let b = sparse::logical_b(&layout, config.bits_y, &data, mask);

    let present = config.bfloat.then(|| sparse::present_elements(&layout, mask));
    let combine = if config.bfloat {
        None
    } else {
        let kind = sparse::op_kind_of(name)?;
        Some(SparseCombine { signs: kind.term_signs()?, config_word: conf_val })
    };
    Some(((a, b), combine, present))
}

/// The post-adder description of an integer MAC: the op kind from the record name (the
/// mnemonic stem), or from the semantic when the op carries no name.
fn integer_combine(op: &SlotOp, semantic: SemanticOp, conf_val: u32) -> Option<SparseCombine> {
    let kind = match op.encoding_name.as_deref().and_then(sparse::op_kind_of) {
        Some(k) => k,
        None => match semantic {
            SemanticOp::MatMul => MacOpKind::Mul,
            SemanticOp::NegMul => MacOpKind::NegMul,
            SemanticOp::Mac => MacOpKind::Mac,
            SemanticOp::MatMulSub => MacOpKind::Msc,
            SemanticOp::NegMatMul => MacOpKind::NegMac,
            SemanticOp::AddMac => MacOpKind::AddMac,
            SemanticOp::SubMac => MacOpKind::SubMac,
            _ => return None,
        },
    };
    Some(SparseCombine { signs: kind.term_signs()?, config_word: conf_val })
}

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

/// Element-unit index into `extract_element_512`'s addressing: a byte offset
/// for 8/16/32-bit elements, the element index itself for 4-bit ones (which
/// the extractor splits into byte and nibble).
fn element_offset(index: u32, bits: u32) -> usize {
    if bits == 4 {
        index as usize
    } else {
        (index * (bits / 8)) as usize
    }
}

/// The signed value of the low `bits` bits of `v`.
fn sign_extend(v: i64, bits: u32) -> i64 {
    (v << (64 - bits)) >> (64 - bits)
}

/// Config-driven full-width integer multiply: every accumulator lane sums the
/// products its geometry names (`MacShape::operand_indices`), whatever the
/// kind (matrix tile, elementwise, convolution, multi-channel), with the config
/// word's `term_neg` bits negating the partial products of their slots
/// (`xdna_archspec::aie2::matmul::TermNegRule`).
///
/// Float (`config.bfloat`) configs take the bf16 datapath.
pub fn matmul_config_driven(acc: &mut Acc1024, a: &[u32], b: &[u32], config: &MatMulConfig) {
    if config.bfloat {
        matmul_bf16(acc, None, a, b, config, None);
        return;
    }
    let shape = config.shape;
    let (bits_x, bits_y) = (config.bits_x, config.bits_y);

    // Zero accumulator if requested (zero_acc = !accumulate in MatMulConfig)
    if !config.accumulate {
        *acc = [0u64; 16];
    }
    if config.multiplier == Multiplier::Idle {
        return;
    }
    let negated = |lane: u32, term: u32, half: u32| {
        config
            .term_neg_rule
            .is_some_and(|rule| config.term_neg >> rule.bit(lane, term, half) & 1 != 0)
    };

    for lane in 0..shape.lanes() {
        let mut sum: i64 = 0;
        for term in 0..shape.terms() {
            let (ia, ib) = shape.operand_indices(lane, term);
            if config.multiplier == Multiplier::FloatConf {
                // The low bytes of 16-bit elements, the product kept to 14 signed bits.
                let a_val = extract_element_512(a, element_offset(ia, 16), 16, false) & 0xFF;
                let b_val = extract_element_512(b, element_offset(ib, 16), 16, false) & 0xFF;
                let product = sign_extend(a_val * b_val, 14);
                sum += if negated(lane, term, 0) { -product } else { product };
                continue;
            }
            let a_val = extract_element_512(a, element_offset(ia, bits_x), bits_x, config.x_signed);
            let b_val = extract_element_512(b, element_offset(ib, bits_y), bits_y, config.y_signed);
            if bits_x == 32 {
                // A 32-bit element is multiplied as two 16-bit halves, each with its own sign slot.
                let (lo, hi) = (a_val & 0xFFFF, a_val >> 16);
                let lo_product = lo * b_val;
                let hi_product = (hi * b_val) << 16;
                sum += if negated(lane, term, 0) {
                    -lo_product
                } else {
                    lo_product
                };
                sum += if negated(lane, term, 1) {
                    -hi_product
                } else {
                    hi_product
                };
            } else {
                let product = a_val * b_val;
                sum += if negated(lane, term, 0) { -product } else { product };
            }
        }

        let out_idx = lane as usize;
        let prev = read_acc_wide(acc, out_idx, config.acc_width);
        if config.subtract {
            write_acc_wide(acc, out_idx, prev - sum, config.acc_width);
        } else {
            write_acc_wide(acc, out_idx, prev + sum, config.acc_width);
        }
    }
}

/// bf16 x bf16 -> fp32 multiply-accumulate (see `bf16::mac_lane` for the
/// derived datapath). Geometry from the config: the dense 4x8x4 tile (A and B
/// row-major) or the elementwise form (16 lanes, lane L = A[L]*B[L] +
/// A[L+16]*B[L+16]), both through `MacShape::operand_indices`. `acc` holds
/// acc1 on entry when the config accumulates and is overwritten with the result; `acc2` is the second accumulator of the add*
/// forms. `config.subtract` is the effective product negation (instruction
/// family folded in by the caller); `sub_acc1`/`sub_acc2` negate the accumulators.
fn matmul_bf16(
    acc: &mut Acc1024,
    acc2: Option<&Acc1024>,
    a: &[u32],
    b: &[u32],
    config: &MatMulConfig,
    b_present: Option<u128>,
) {
    let bf16 = |v: &[u32], i: usize| (v[i / 2] >> (16 * (i % 2))) as u16;
    let lane32 = |v: &Acc1024, i: usize| (v[i / 2] >> (32 * (i % 2))) as u32;
    let shape = config.shape;
    let mut out: Acc1024 = [0u64; 16];
    for lane in 0..shape.lanes() {
        // A B element the sparse mask drops is absent from the multiplier: its product is not a
        // zero term (which would turn an Inf or NaN in A into a NaN) but not a term at all.
        let products: Vec<(u16, u16)> = (0..shape.terms())
            .filter_map(|term| {
                let (ia, ib) = shape.operand_indices(lane, term);
                let kept = b_present.map_or(true, |p| p >> ib & 1 != 0);
                kept.then(|| (bf16(a, ia as usize), bf16(b, ib as usize)))
            })
            .collect();
        let lane = lane as usize;
        let mut accs = Vec::with_capacity(2);
        if config.accumulate {
            accs.push((lane32(acc, lane), config.sub_acc1));
        }
        if let Some(a2) = acc2 {
            accs.push((lane32(a2, lane), config.sub_acc2));
        }
        let bits = bf16::mac_lane(&products, &accs, config.subtract);
        out[lane / 2] |= (bits as u64) << (32 * (lane % 2));
    }
    *acc = out;
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
            let (rows, inner, cols) = config.shape.matmul_dims().expect("from_types selects a matmul tile");
            assert_eq!(
                (rows, inner, cols, config.bits_x, config.bits_y, config.acc_width, config.bfloat),
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

        // vnegmac.f / vsubmac.f have no silicon capture: they stay gated.
        let mut op = make_mac_op(SemanticOp::NegMatMul, 0, 2, 5, 0, Some("VNEGMAC_F_vmac_bm_core_dense"));
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
    fn test_vmul_f_overflowing_products_cancel_and_absorb_the_small_term() {
        // bf16(2^70)^2 = 2^140 exceeds fp32 range as a standalone product, but
        // the +/- pair cancels inside the datapath (no overflow to NaN). The
        // third product, 1.0, is 2^140 below the largest term and rounds away
        // onto the 2^(140-23) grid, so the lane is +0 -- not the true sum 1
        // (vec_mac_bf16_ovf, 2^30 / 1 / -2^30 tiles).
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
        assert_eq!(fp32_lane(&ctx.accumulator.read(0), 0), 0.0);
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

    #[test]
    fn test_vaddmac_f_result_is_visible_with_the_float_latency() {
        // II_VADDMACf has the float result operand cycle 6 like II_VMULf: no extra cycle for the
        // add forms (the accumulator timing probe, tools/gen_acc_timing_probe.py). A vmsc.f
        // issued three bundles later still reads the old value, since it reads at cycle 3.
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write(0, [0u64; 8]);
        ctx.accumulator.write(2, [0u64; 8]);
        ctx.accumulator.write(4, [0u64; 8]);
        ctx.vector.write_wide(0, vec512_all_ones_bf16());
        ctx.vector.write_wide(2, vec512_all_ones_bf16());
        ctx.scalar.write(5, config_bf16_accumulate());
        let mut op = make_mac_op(SemanticOp::AddMac, 0, 2, 5, 4, Some("VADDMAC_F_vmac_bm_core_dense"));
        op.element_type = Some(ElementType::Float32);
        op.sources.push(Operand::AccumReg(0));
        op.sources.push(Operand::AccumReg(2));
        assert!(execute_matmul(&op, &mut ctx));
        for _ in 0..5 {
            step_bundle(&mut ctx);
        }
        assert_eq!(fp32_lane(&ctx.accumulator.read(4), 0), 0.0, "not yet visible at +5");
        step_bundle(&mut ctx);
        assert_eq!(fp32_lane(&ctx.accumulator.read(4), 0), 8.0, "visible at +6");
    }

    /// Execute one bf16 form through `execute_matmul` for a whole silicon tile and
    /// return its 16 fp32 lanes. `f` is the form's JSON descriptor from
    /// `vec_mac_bf16_forms.json`; the semantic, mnemonic and config word are
    /// built the way the decoder and Chess produce them.
    fn run_bf16_form(f: &serde_json::Value, a: &[u16], b: &[u16], c0: &[u32], c1: &[u32]) -> Vec<u32> {
        let flag = |k: &str| f[k].as_u64().unwrap() as u32;
        let instr = f["instr"].as_str().unwrap();
        let (semantic, name) = match instr {
            "mul" => (SemanticOp::MatMul, "VMUL_F_vmac_bm_core_dense"),
            "negmul" => (SemanticOp::NegMul, "VNEGMUL_F_vmac_bm_core_dense"),
            "mac" => (SemanticOp::Mac, "VMAC_F_vmac_bm_core_dense"),
            "msc" => (SemanticOp::MatMulSub, "VMSC_F_vmac_bm_core_dense"),
            "addmac" => (SemanticOp::AddMac, "VADDMAC_F_vmac_bm_core_dense"),
            "addmsc" => (SemanticOp::AddMac, "VADDMSC_F_vmac_bm_core_dense"),
            other => panic!("unknown form {other}"),
        };
        let variant = if f["geometry"].as_str().unwrap() == "el" {
            1
        } else {
            0
        };
        let conf = flag("zero_acc1")
            | (variant << 5)
            | (flag("sub_mul") << 11)
            | (flag("sub_acc1") << 12)
            | (flag("sub_acc2") << 13);
        let mut ctx = ExecutionContext::new();
        let fp32_acc = |w: &[u32]| {
            let mut r = [0u64; 8];
            for (i, &x) in w.iter().enumerate() {
                r[i / 2] |= (x as u64) << (32 * (i % 2));
            }
            r
        };
        ctx.accumulator.write(0, fp32_acc(c0));
        ctx.accumulator.write(2, fp32_acc(c1));
        ctx.vector.write_wide(0, vec512_from_bf16(a));
        ctx.vector.write_wide(2, vec512_from_bf16(b));
        ctx.scalar.write(5, conf);
        let mut op = make_mac_op(semantic, 0, 2, 5, 4, Some(name));
        op.element_type = Some(ElementType::Float32);
        if instr != "mul" && instr != "negmul" {
            op.sources.push(Operand::AccumReg(0));
        }
        if instr.starts_with("add") {
            op.sources.push(Operand::AccumReg(2));
        }
        assert!(execute_matmul(&op, &mut ctx), "{name} must dispatch to the matmul unit");
        ctx.force_commit_all_pending();
        let acc = ctx.accumulator.read(4);
        (0..16).map(|i| (acc[i / 2] >> (32 * (i % 2))) as u32).collect()
    }

    /// Every form of the forms capture, run end to end through the instruction
    /// handler (config word, semantic, accumulator plumbing), is bit-exact
    /// against NPU1 silicon.
    #[test]
    fn test_bf16_forms_match_silicon_through_execute_matmul() {
        let path = format!("{}/tools/golden/silicon/vec_mac_bf16_forms.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let col = |i: usize| -> Vec<u64> {
            v["inputs"][i].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
        };
        let (a, b, c0, c1) = (col(0), col(1), col(2), col(3));
        let silicon: Vec<u32> = v["silicon"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let forms = v["forms"].as_array().unwrap();
        let (mut checked, mut bad) = (0, Vec::new());
        for t in 0..v["tile"]["tiles"].as_u64().unwrap() as usize {
            let ta: Vec<u16> = a[t * 32..t * 32 + 32].iter().map(|&x| x as u16).collect();
            let tb: Vec<u16> = b[t * 32..t * 32 + 32].iter().map(|&x| x as u16).collect();
            let tc0: Vec<u32> = c0[t * 16..t * 16 + 16].iter().map(|&x| x as u32).collect();
            let tc1: Vec<u32> = c1[t * 16..t * 16 + 16].iter().map(|&x| x as u32).collect();
            for (fi, f) in forms.iter().enumerate() {
                let got = run_bf16_form(f, &ta, &tb, &tc0, &tc1);
                let want = &silicon[(t * forms.len() + fi) * 16..(t * forms.len() + fi) * 16 + 16];
                checked += 16;
                for lane in 0..16 {
                    if got[lane] != want[lane] {
                        bad.push((t, f["name"].as_str().unwrap().to_string(), lane, got[lane], want[lane]));
                    }
                }
            }
        }
        assert!(checked > 100_000);
        assert!(
            bad.is_empty(),
            "{} of {checked} lanes differ, first: {:?}",
            bad.len(),
            &bad[..bad.len().min(5)]
        );
    }

    /// The fresh-multiply captures that were off by 1-8 ulp on wide-exponent
    /// tiles (and unmatched on NaN/Inf tiles) are now bit-exact on every lane.
    #[test]
    fn test_vmul_f_matches_silicon_exactly_on_older_captures() {
        for kernel in ["vec_mac_bf16", "vec_mac_bf16_ovf"] {
            let path = format!("{}/tools/golden/silicon/{kernel}.json", env!("CARGO_MANIFEST_DIR"));
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
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
            let mut bad = Vec::new();
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
                    let got = fp32_lane(&acc, lane).to_bits();
                    if got != silicon[tile * 16 + lane] {
                        bad.push((tile, lane, got, silicon[tile * 16 + lane]));
                    }
                }
            }
            assert!(
                bad.is_empty(),
                "{kernel}: {} lanes differ, first {:?}",
                bad.len(),
                &bad[..bad.len().min(5)]
            );
        }
    }

    /// Step the context the way the cycle-accurate loop does: advance both
    /// clocks, commit pending writes and land the register files' in-flight writes, then
    /// complete ready deferred matmuls.
    fn step_bundle(ctx: &mut ExecutionContext) {
        ctx.cycles += 1;
        ctx.bundle_seq += 1;
        ctx.commit_pending_writes();
        ctx.advance_vector_bundle();
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

        // The itinerary data a decoded op carries: II_VMUL/II_VMAC results are 5 bundles, VEC_Bypass;
        // II_VMAC reads its accumulator source at cycle 3 on the same bypass class.
        use crate::interpreter::state::Bypass;
        ctx.result_latency = 5;
        ctx.result_bypass = Bypass::Vec;

        // Bundle 0: vmul (fresh multiply, zeroes acc).
        let mul = make_mac_op(SemanticOp::MatMul, 0, 2, 5, 0, Some("VMUL_vmac_cm_core_dense"));
        assert!(execute_matmul(&mul, &mut ctx));

        step_bundle(&mut ctx);
        step_bundle(&mut ctx);

        // Bundle 2: vmac chained on the same accumulator.
        let mut mac = make_mac_op(SemanticOp::Mac, 0, 2, 5, 0, Some("VMAC_vmac_cm_core_dense"));
        mac.sources.push(Operand::AccumReg(0));
        mac.source_forward =
            smallvec::smallvec![(1, Bypass::No), (1, Bypass::No), (1, Bypass::No), (3, Bypass::Vec)];
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
    fn test_sparse_integer_mac_without_a_known_op_is_unhandled() {
        // The sparse integer combine needs the op kind (its term signs) from the record name; a
        // name that gives none is reported unhandled and writes nothing.
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write_wide(0, [0x55u64; 16]);
        ctx.scalar.write(5, (1 << 3) | (5 << 5) | (1 << 8) | (1 << 9) | 1);

        let mut op = SlotOp::from_semantic(SlotIndex::Accumulator, SemanticOp::Mac);
        op.is_vector = true;
        op.is_wide_vector = true;
        op.sources.push(Operand::VectorReg(0));
        op.sources.push(Operand::SparseQxReg(1));
        op.sources.push(Operand::ScalarReg(5));
        op.dest = Some(Operand::AccumReg(0));
        op.encoding_name = Some("VUNKNOWN_vmac_cm_core_sparse_narrow".to_string());

        assert!(!execute_matmul(&op, &mut ctx));
        ctx.force_commit_all_pending();
        assert_eq!(ctx.accumulator.read_wide(0), [0x55u64; 16]);
    }

    /// Dense integer MACs against silicon (`gen_collision_probe.py negate`, NPU1, r1 == r2):
    /// x0 = 1 + k % 7, x1 = 2 + k % 5 (int16), acc = 2200 + j in 64-bit lanes, config 0x31a.
    /// vnegmac is -(acc + A*B) and vnegmsc is -(acc - A*B): the whole result is negated, not
    /// just the product.
    #[test]
    fn dense_int_mac_family_matches_silicon() {
        let x0: Vec<u32> = [
            131073, 262147, 393221, 65543, 196610, 327684, 458758, 131073, 262147, 393221, 65543, 196610,
            327684, 458758, 131073, 262147,
        ]
        .to_vec();
        let x1: Vec<u32> = [
            196610, 327684, 131078, 262147, 393221, 196610, 327684, 131078, 262147, 393221, 196610, 327684,
            131078, 262147, 393221, 196610,
        ]
        .to_vec();
        let rows: &[(SemanticOp, &str, [u32; 8])] = &[
            (SemanticOp::Mac, "VMAC_vmac_cm_core_dense", [0x8c5, 0x0, 0x8bc, 0x0, 0x8bd, 0x0, 0x8c8, 0x0]), // vmac
            (
                SemanticOp::MatMulSub,
                "VMSC_vmac_cm_core_dense",
                [0x86b, 0x0, 0x876, 0x0, 0x877, 0x0, 0x86e, 0x0],
            ), // vmsc
            (
                SemanticOp::NegMul,
                "VNEGMUL_vmac_cm_core_dense",
                [
                    0xffffffd3, 0xffffffff, 0xffffffdd, 0xffffffff, 0xffffffdd, 0xffffffff, 0xffffffd3,
                    0xffffffff,
                ],
            ), // vnegmul
            (
                SemanticOp::NegMul,
                "VNEGMAC_vmac_cm_core_dense",
                [
                    0xfffff73b, 0xffffffff, 0xfffff744, 0xffffffff, 0xfffff743, 0xffffffff, 0xfffff738,
                    0xffffffff,
                ],
            ), // vnegmac
            (
                SemanticOp::NegMul,
                "VNEGMSC_vmac_cm_core_dense",
                [
                    0xfffff795, 0xffffffff, 0xfffff78a, 0xffffffff, 0xfffff789, 0xffffffff, 0xfffff792,
                    0xffffffff,
                ],
            ), // vnegmsc
        ];
        let mut bad = Vec::new();
        for (sem, enc, want) in rows {
            let mut ctx = ExecutionContext::new();
            ctx.accumulator.write_wide(0, std::array::from_fn(|j| 2200 + j as u64));
            ctx.vector.write_wide(0, x0.clone().try_into().unwrap());
            ctx.vector.write_wide(2, x1.clone().try_into().unwrap());
            ctx.scalar.write(5, 0x31a);
            let op = make_mac_op(*sem, 0, 2, 5, 0, Some(enc));
            assert!(execute_matmul(&op, &mut ctx), "{enc}");
            ctx.force_commit_all_pending();
            let acc = ctx.accumulator.read_wide(0);
            let got: [u32; 8] = std::array::from_fn(|i| (acc[i / 2] >> (32 * (i % 2))) as u32);
            if got != *want {
                bad.push(format!("{enc}: silicon {want:x?}, emu {got:x?}"));
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }
}
