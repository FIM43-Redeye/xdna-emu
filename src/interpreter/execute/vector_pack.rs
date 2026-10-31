//! Vector pack/unpack operations for AIE2.
//!
//! Pack narrows vector lanes by half (int16 to int8, int8 to int4); unpack widens them. Both
//! operate on 512-bit vector registers (the narrow side is 256 bits).
//!
//! [`pack_lanes`] is the one lane map behind both the standalone `vpack` and the fused
//! `vst.pack`; the per-form facts (widths, sign source) come from `xdna_archspec::aie2::pack`
//! and the live `crSat` / `crPackSign` registers. Observed on NPU1 silicon for all four forms
//! (`s8.s16`, `d8.d16`, `s4.s8`, `d4.d8`), fused and standalone (`vec_pack_lane_probe`,
//! `vec_pack_lane_rand`, `vec_pack_addr`, `vec_pack_crtiming`):
//!
//! - lane `i` of the result is lane `i` of the source; the two 256-bit halves of the source
//!   continue the order, and 4-bit results are packed low nibble first;
//! - a signed lane is read two's complement and clamps to the signed output range; an
//!   unsigned lane is read as an unsigned number and clamps to `[0, 2^bits_out - 1]`;
//! - the `s` forms are signed; a `d` form is signed when `crPackSign` is 1, unsigned when 0;
//! - `crSat` selects truncation or saturation per [`PackMode::for_pack`].
//!
//! Unpack is always a widening operation: sign-extend (signed) or zero-extend (unsigned) from
//! the source width to the destination width. [`unpack_lanes`] is the one lane map behind
//! both the standalone `vunpack` and the fused `vldb.unpack`; the per-form facts (widths,
//! sign source) come from `xdna_archspec::aie2::unpack`.

/// Pack mode controlling how wider values are narrowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackMode {
    /// Truncate: mask to output width, no range checking.
    Truncate,
    /// Saturate: clamp to destination range, then truncate.
    /// Signed destinations use [-2^(n-1), 2^(n-1)-1].
    /// Unsigned destinations use [0, 2^n - 1].
    Saturate,
    /// Symmetric saturate: like Saturate, but for signed destinations
    /// the minimum is -(2^(n-1) - 1) instead of -2^(n-1).
    /// This ensures the negative and positive ranges are mirror images.
    SymmetricSaturate,
}

impl PackMode {
    /// Select the narrow mode from saturation flags: bit 0 = saturate, bit 1 = symmetric.
    ///
    /// This is the flag reading the model-validation harness uses; it is NOT what the pack
    /// instructions do (see [`PackMode::for_pack`]): it reads crSat 2 as truncation for
    /// both signednesses.
    pub fn from_sat_flags(saturate: bool, symmetric: bool) -> PackMode {
        match (saturate, symmetric) {
            (false, _) => PackMode::Truncate,
            (true, false) => PackMode::Saturate,
            (true, true) => PackMode::SymmetricSaturate,
        }
    }

    /// The narrowing the pack instructions (`vpack`, `vst.pack`) perform for a `crSat` value,
    /// as observed on NPU1 silicon for every form and both signednesses:
    ///
    /// | crSat | signed lanes                  | unsigned lanes          |
    /// |-------|-------------------------------|-------------------------|
    /// | 0     | truncate                      | truncate                |
    /// | 1     | saturate to [-2^(n-1), 2^(n-1)-1] | saturate to [0, 2^n-1] |
    /// | 2     | truncate                      | saturate to [0, 2^n-1]  |
    /// | 3     | symmetric: [-(2^(n-1)-1), 2^(n-1)-1] | saturate to [0, 2^n-1] |
    ///
    /// Only the low two bits of the register matter (Core_CR [1:0]). The unsigned column
    /// saturates at every non-zero value, including 2, which the signed column truncates.
    pub fn for_pack(crsat: u8, signed: bool) -> PackMode {
        match (crsat & 0x3, signed) {
            (0, _) | (2, true) => PackMode::Truncate,
            (1, _) | (_, false) => PackMode::Saturate,
            (3, true) => PackMode::SymmetricSaturate,
            _ => unreachable!("crsat is masked to two bits"),
        }
    }
}

/// Truncate a value to `bits` width, interpreting as signed or unsigned.
///
/// If `signed`: sign-extend from bit position (bits-1).
/// If `unsigned`: zero-extend (mask only).
///
/// This matches the hardware truncation behavior: mask to width, then
/// optionally sign-extend the result for signed interpretation.
fn truncate(value: i64, signed: bool, bits: u32) -> i64 {
    if bits == 0 {
        return 0;
    }
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let masked = (value as u64) & mask;

    if signed && bits < 64 {
        // Sign-extend: if the top bit of the masked value is set, extend it.
        let sign_bit = 1u64 << (bits - 1);
        if masked & sign_bit != 0 {
            // Set all bits above the field to 1
            (masked | !mask) as i64
        } else {
            masked as i64
        }
    } else {
        masked as i64
    }
}

/// Pack a single lane: narrow from `bits_i` to `bits_o`.
///
/// The `signed` parameter controls whether saturation uses signed or unsigned
/// bounds. `mode` selects truncation, saturation, or symmetric saturation.
pub fn pack_lane(value: i64, _bits_i: u32, bits_o: u32, signed: bool, mode: PackMode) -> i64 {
    let clamped = match mode {
        PackMode::Truncate => value,
        PackMode::Saturate => {
            let (vmin, vmax) = if signed {
                let vmax = (1i64 << (bits_o - 1)) - 1;
                let vmin = -(1i64 << (bits_o - 1));
                (vmin, vmax)
            } else {
                let vmax = (1i64 << bits_o) - 1;
                (0i64, vmax)
            };
            value.clamp(vmin, vmax)
        }
        PackMode::SymmetricSaturate => {
            let (vmin, vmax) = if signed {
                let vmax = (1i64 << (bits_o - 1)) - 1;
                // Symmetric: min = -(max) = -(2^(n-1) - 1)
                let vmin = -vmax;
                (vmin, vmax)
            } else {
                // Unsigned symmetric saturation is same as regular saturation
                let vmax = (1i64 << bits_o) - 1;
                (0i64, vmax)
            };
            value.clamp(vmin, vmax)
        }
    };

    truncate(clamped, signed, bits_o)
}

/// Narrow a 512-bit source register into 256 bits.
///
/// Lane `i` of the result is source lane `i`, `bits_in` wide with lane 0 in the
/// least-significant bits, read as signed or unsigned and narrowed to `bits_out` per `mode`.
/// There are `512 / bits_in` lanes and the result holds all of them, `bits_out` wide.
///
/// This is the one lane map of every pack form, standalone and fused alike; callers resolve
/// `signed` from the form's sign source and the live `crPackSign`, and `mode` from the live
/// `crSat` through [`PackMode::for_pack`].
pub fn pack_lanes(src: &[u32; 16], bits_in: u32, bits_out: u32, signed: bool, mode: PackMode) -> [u32; 8] {
    assert!(
        matches!(bits_in, 8 | 16) && bits_in == 2 * bits_out,
        "pack halves the lane width from 8 or 16 bits"
    );
    let lanes = (512 / bits_in) as usize;
    debug_assert_eq!(lanes * bits_out as usize, 256);
    let narrowed: Vec<i64> = (0..lanes)
        .map(|i| pack_lane(extract_lane(src, i, bits_in, signed), bits_in, bits_out, signed, mode))
        .collect();
    let mut result = [0u32; 8];
    insert_lanes(&mut result, &narrowed, bits_out);
    result
}

/// Narrow a source register as the pack form `form` does with the live control registers.
///
/// `read_control` returns a control register's raw value by hardware id (the sign register the
/// form names); `crsat` is the live `crSat` value. Both pack registers are read at issue (the
/// itineraries read them at pipeline stage 1 and 2, no deferred read): the caller passes the
/// values as of the issuing bundle.
pub fn pack_register(
    src: &[u32; 16],
    form: &PackForm,
    crsat: u8,
    read_control: impl FnOnce(u8) -> u32,
) -> [u32; 8] {
    let signed = form.sign.is_signed(read_control);
    pack_lanes(src, form.bits_in as u32, form.bits_out as u32, signed, PackMode::for_pack(crsat, signed))
}

/// Widen a 256-bit packed source into the 512-bit result of an unpack.
///
/// Lane `i` of the result is source element `i`, `bits_in` wide with element 0 in the
/// least-significant bits, sign-extended or zero-extended to `bits_out`. There are
/// `256 / bits_in` source elements and `512 / bits_out` result lanes, and for every unpack
/// form these are equal, so the whole source is consumed and the whole result written.
///
/// Observed on NPU1 silicon for all four forms (`s8.s4`, `s16.s8`, `d8.d4`, `d16.d8`), in
/// both the fused load and the standalone instruction (`vec_unpack_lane_probe` and
/// `vec_unpack_lane_rand`): the high half of the result continues the order, with no
/// interleaving or per-half restart.
pub fn unpack_lanes(src: &[u32; 8], bits_in: u32, bits_out: u32, signed: bool) -> [u32; 16] {
    assert!(matches!(bits_in, 4 | 8) && bits_out == 2 * bits_in, "unpack widens by 2x from 4 or 8 bits");
    let lanes = (512 / bits_out) as usize;
    debug_assert_eq!(lanes, (256 / bits_in) as usize);
    let in_mask = (1u32 << bits_in) - 1;
    let out_mask = (1u32 << bits_out) - 1;
    let mut out = [0u32; 16];
    for lane in 0..lanes {
        let in_bit = lane as u32 * bits_in;
        let raw = (src[(in_bit / 32) as usize] >> (in_bit % 32)) & in_mask;
        let extended = if signed && raw >> (bits_in - 1) != 0 {
            raw | (out_mask & !in_mask)
        } else {
            raw
        };
        let out_bit = lane as u32 * bits_out;
        out[(out_bit / 32) as usize] |= extended << (out_bit % 32);
    }
    out
}

/// Extract lane `idx` from a register (a slice of u32 words) at `bits` width.
///
/// Handles arbitrary bit widths including those that cross u32 boundaries.
fn extract_lane(reg: &[u32], idx: usize, bits: u32, signed: bool) -> i64 {
    let bit_offset = idx as u32 * bits;
    let word_idx = (bit_offset / 32) as usize;
    let bit_idx = bit_offset % 32;

    // Gather enough bits -- may span two u32 words.
    if word_idx >= reg.len() {
        return 0; // Lane extends beyond the register
    }
    let mut raw: u64 = reg[word_idx] as u64 >> bit_idx;
    let bits_from_first = 32 - bit_idx;
    if bits > bits_from_first && word_idx + 1 < reg.len() {
        raw |= (reg[word_idx + 1] as u64) << bits_from_first;
    }

    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let masked = raw & mask;

    if signed && bits > 0 && bits < 64 {
        let sign_bit = 1u64 << (bits - 1);
        if masked & sign_bit != 0 {
            (masked | !mask) as i64
        } else {
            masked as i64
        }
    } else {
        masked as i64
    }
}

/// Insert `lanes` values at `bits` width into a 256-bit (8 x u32) register.
///
/// Each lane value is masked to `bits` width and packed at the appropriate
/// bit position.
fn insert_lanes(reg: &mut [u32; 8], lanes: &[i64], bits: u32) {
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };

    for (idx, &val) in lanes.iter().enumerate() {
        let bit_offset = idx as u32 * bits;
        let word_idx = (bit_offset / 32) as usize;
        let bit_idx = bit_offset % 32;
        let masked = (val as u64) & mask;

        if word_idx < 8 {
            reg[word_idx] |= (masked << bit_idx) as u32;
        }

        // Handle overflow into next word
        let bits_in_first = 32 - bit_idx;
        if bits > bits_in_first && word_idx + 1 < 8 {
            reg[word_idx + 1] |= (masked >> bits_in_first) as u32;
        }
    }
}

// ========== VectorAlu dispatch for Pack/Unpack ==========

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::{ExecutionContext, UnpackSource};
use xdna_archspec::aie2::pack::{pack_form, PackForm};
use xdna_archspec::aie2::unpack::unpack_form;

use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    /// Standalone `vpack`: narrow a 512-bit x-register source into a 256-bit w register.
    ///
    /// One lane map ([`pack_register`]) driven by the form table and the live `crSat` and
    /// `crPackSign`; the fused `vst.pack` goes through the same function. II_VPACK reads
    /// `crSat` and `crPackSign` at pipeline stage 1 and the source at stage 2, so unlike
    /// `vunpack` nothing is deferred: the narrowing happens at issue. A control register
    /// written in the same bundle is not yet visible (silicon, `vec_pack_crtiming`).
    pub(super) fn execute_pack(op: &SlotOp, ctx: &mut ExecutionContext, _et: ElementType) -> bool {
        let Some(form) = op.encoding_name.as_deref().and_then(pack_form) else {
            log::error!("[VECTOR] execute_pack: {:?} is not a pack mnemonic", op.encoding_name);
            return true;
        };
        if !op.is_wide_vector {
            // Every VPACK record's source is an x register (`mXs`), which the decoder marks
            // wide; a pack that is not has no AIE2 meaning.
            log::error!("[VECTOR] execute_pack: {} without a 512-bit source", form.mnemonic);
            return true;
        }
        let src = Self::get_wide_vec_source(op, ctx, 0);
        let result = pack_register(&src, form, ctx.srs_config.saturation_mode, |id| ctx.control_reg_raw(id));
        // The destination is a 256-bit w register (NOT write_wide_vec_dest).
        Self::write_vector_dest(op, ctx, result);
        true
    }

    /// Standalone `vunpack`: widen a 256-bit w-register source into a 512-bit x register.
    ///
    /// II_VUNPACK reads its SOURCE (and `crUnpackSign` for the register-driven forms) at
    /// pipeline stage 7 (operand cycles `[7,7,7]` in AIE2Schedule.td), not at issue. The
    /// compiler exploits this: a preceding vldb's data is naturally visible (load latency
    /// 7), and producers of the source can be scheduled AFTER the vunpack (silicon-verified,
    /// fuzzer seed 7 where a vsel 4 bundles later supplies the unpack source). So the read
    /// and the widening are deferred to issue+6 via the pending-unpack queue; only the
    /// operand registers are captured here.
    pub(super) fn execute_unpack(op: &SlotOp, ctx: &mut ExecutionContext, _et: ElementType) -> bool {
        let Some(form) = op.encoding_name.as_deref().and_then(unpack_form) else {
            log::error!("[VECTOR] execute_unpack: {:?} is not an unpack mnemonic", op.encoding_name);
            return true;
        };
        let src_reg = match op.sources.first() {
            Some(Operand::VectorReg(r)) => *r,
            other => {
                log::error!("[VECTOR] execute_unpack: expected VectorReg source, got {:?}", other);
                return true;
            }
        };
        let dest_reg = match &op.dest {
            Some(Operand::VectorReg(r)) => *r,
            other => {
                log::error!("[VECTOR] execute_unpack: expected VectorReg dest, got {:?}", other);
                return true;
            }
        };
        ctx.queue_pending_unpack(UnpackSource::Register(src_reg), dest_reg, form);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{Operand, SlotIndex};
    use xdna_archspec::aie2::isa::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    /// VPACK_S8_S16 `Uses = [crSat]`: a saturating crSat must make the wide
    /// int16->int8 pack saturate (clamp), not truncate. Verified against real
    /// NPU1 silicon (vec_pack_i16_sat: HW saturates 200->127, matches the
    /// saturating golden; the emulator previously hardcoded truncation).
    #[test]
    fn execute_pack_honors_crsat_saturation() {
        let mut ctx = make_ctx();
        // lane 0 = 200 (out of int8 range): truncate -> -56, saturate -> 127.
        let mut src = [0u32; 16];
        src[0] = 200;
        ctx.vector.write_wide(0, src);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Pack)
            .as_vector(ElementType::Int8)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0));
        op.is_wide_vector = true;
        op.encoding_name = Some("vpack.s8.s16".to_string());

        // crSat = saturate (bit 0 set): clamp 200 -> 127.
        ctx.srs_config.saturation_mode = 1;
        VectorAlu::execute(&op, &mut ctx);
        assert_eq!((ctx.vector.read(2)[0] & 0xff) as u8 as i8, 127, "crSat=saturate must clamp 200 -> 127");

        // crSat = none (0): truncate low 8 bits -> -56.
        ctx.vector.write_wide(0, src);
        ctx.srs_config.saturation_mode = 0;
        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(
            (ctx.vector.read(2)[0] & 0xff) as u8 as i8,
            200u32 as u8 as i8,
            "crSat=none must truncate 200 -> -56"
        );
    }

    /// II_VUNPACK reads its w-reg source at pipeline stage 7 (operand cycles
    /// [7,7,7], AIE2Schedule.td), so a producer scheduled AFTER the vunpack
    /// supplies the data. Silicon-verified: vector fuzzer seed 7 has Peano
    /// emit `vunpack.s16.s8 x4, wl2` four bundles BEFORE the `vsel.8 x2, ...`
    /// whose result the unpack consumes; NPU1 unpacks the vsel result.
    /// Sampling at issue read the stale pre-vsel register.
    #[test]
    fn execute_unpack_samples_source_at_stage7_not_issue() {
        let mut ctx = make_ctx();
        // Committed (old) value: bytes 0x11. New value (written by a producer
        // 4 bundles after the unpack issues): bytes 0xAA -> -86 sign-extended.
        ctx.vector.write(2, [0x1111_1111; 8]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Unpack)
            .as_vector(ElementType::Int8)
            .with_dest(Operand::VectorReg(4))
            .with_source(Operand::VectorReg(2));
        op.is_wide_vector = true;
        op.encoding_name = Some("vunpack.s16.s8".to_string());

        // Issue the unpack at bundle 0.
        ctx.vector.advance_bundle(0);
        ctx.bundle_seq = 0;
        VectorAlu::execute(&op, &mut ctx);

        let step = |ctx: &mut ExecutionContext, b: u64| {
            ctx.bundle_seq = b;
            ctx.vector.advance_bundle(b);
            ctx.process_pending_unpacks();
        };

        for b in 1..=3 {
            step(&mut ctx, b);
        }
        // Producer (vsel, latency 2) writes the source register at bundle 4:
        // architecturally visible at bundle 6 -- exactly the unpack read bundle.
        ctx.vector
            .queue_write(2, [0xAAAA_AAAA; 8], 2, crate::interpreter::state::Bypass::No);
        for b in 4..=7 {
            step(&mut ctx, b);
        }

        // Dest lands at issue+7: every i16 lane is sign-extend(0xAA) = 0xFFAA.
        let dest = ctx.vector.read_wide(4);
        assert_eq!(dest, [0xFFAA_FFAA; 16], "unpack must widen the post-producer source value");
    }

    // -- truncate --

    #[test]
    fn truncate_unsigned_masks_bits() {
        // 0x1234 truncated to 8 bits unsigned = 0x34
        assert_eq!(truncate(0x1234, false, 8), 0x34);
    }

    #[test]
    fn truncate_signed_positive() {
        // 127 fits in signed 8-bit -> 127
        assert_eq!(truncate(127, true, 8), 127);
    }

    #[test]
    fn truncate_signed_negative() {
        // 0xFF in 8-bit signed = -1
        assert_eq!(truncate(0xFF, true, 8), -1);
    }

    #[test]
    fn truncate_signed_from_negative() {
        // -1 as i64 -> 0xFF mask -> sign extend -> -1
        assert_eq!(truncate(-1, true, 8), -1);
    }

    #[test]
    fn truncate_zero_bits() {
        assert_eq!(truncate(0xABCD, true, 0), 0);
        assert_eq!(truncate(0xABCD, false, 0), 0);
    }

    #[test]
    fn truncate_16_bit_signed() {
        // 0x8000 in 16-bit signed = -32768
        assert_eq!(truncate(0x8000, true, 16), -32768);
        // 0x7FFF in 16-bit signed = 32767
        assert_eq!(truncate(0x7FFF, true, 16), 32767);
    }

    // -- pack_lane --

    #[test]
    fn pack_lane_truncate_just_masks() {
        // 0x12345 truncated to 16 bits = 0x2345
        let r = pack_lane(0x12345, 32, 16, false, PackMode::Truncate);
        assert_eq!(r, 0x2345);
    }

    #[test]
    fn pack_lane_saturate_signed_clamps_positive() {
        // 50000 saturated to signed 16-bit -> 32767
        let r = pack_lane(50000, 32, 16, true, PackMode::Saturate);
        assert_eq!(r, 32767);
    }

    #[test]
    fn pack_lane_saturate_signed_clamps_negative() {
        // -50000 saturated to signed 16-bit -> -32768
        let r = pack_lane(-50000, 32, 16, true, PackMode::Saturate);
        assert_eq!(r, -32768);
    }

    #[test]
    fn pack_lane_saturate_unsigned_clamps() {
        // 70000 saturated to unsigned 16-bit -> 65535
        let r = pack_lane(70000, 32, 16, false, PackMode::Saturate);
        assert_eq!(r, 65535);
    }

    #[test]
    fn pack_lane_saturate_unsigned_clamps_negative() {
        // -5 saturated to unsigned 16-bit -> 0
        let r = pack_lane(-5, 32, 16, false, PackMode::Saturate);
        assert_eq!(r, 0);
    }

    #[test]
    fn pack_lane_symmetric_saturate_signed() {
        // Symmetric signed 16-bit: range is [-32767, 32767]
        // -50000 -> -32767 (not -32768)
        let r = pack_lane(-50000, 32, 16, true, PackMode::SymmetricSaturate);
        assert_eq!(r, -32767);
        // 50000 -> 32767
        let r = pack_lane(50000, 32, 16, true, PackMode::SymmetricSaturate);
        assert_eq!(r, 32767);
    }

    #[test]
    fn pack_lane_value_in_range_passes_through() {
        // 100 is within signed 16-bit range -> unchanged
        let r = pack_lane(100, 32, 16, true, PackMode::Saturate);
        assert_eq!(r, 100);
        // -100 within range
        let r = pack_lane(-100, 32, 16, true, PackMode::Saturate);
        assert_eq!(r, -100);
    }

    #[test]
    fn pack_lane_8bit_signed_saturate() {
        // Saturate to signed 8-bit: [-128, 127]
        assert_eq!(pack_lane(200, 16, 8, true, PackMode::Saturate), 127);
        assert_eq!(pack_lane(-200, 16, 8, true, PackMode::Saturate), -128);
    }

    #[test]
    fn pack_lane_8bit_unsigned_saturate() {
        // Saturate to unsigned 8-bit: [0, 255]
        assert_eq!(pack_lane(300, 16, 8, false, PackMode::Saturate), 255);
        assert_eq!(pack_lane(-1, 16, 8, false, PackMode::Saturate), 0);
    }

    // -- unpack_lanes --

    #[test]
    fn unpack_lanes_8_to_16_continues_the_order_into_the_high_half() {
        // Source byte i = i + 1: result lane i is i + 1, for all 32 lanes (silicon: no
        // interleave, no per-half restart).
        let mut src = [0u32; 8];
        insert_lanes(&mut src, &(1..=32).collect::<Vec<i64>>(), 8);
        let out = unpack_lanes(&src, 8, 16, false);
        for lane in 0..32 {
            assert_eq!(extract_lane16(&out, lane), lane as u16 + 1, "lane {lane}");
        }
    }

    #[test]
    fn unpack_lanes_4_to_8_sign_extends_each_nibble() {
        // Nibbles 0x9 0x7 0xF 0x8 ...: low nibble of byte 0 is lane 0.
        let mut src = [0u32; 8];
        src[0] = 0x8F79;
        let signed = unpack_lanes(&src, 4, 8, true);
        let bytes: Vec<u8> = signed.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(&bytes[..4], &[0xF9, 0x07, 0xFF, 0xF8]);
        let unsigned = unpack_lanes(&src, 4, 8, false);
        let bytes: Vec<u8> = unsigned.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(&bytes[..4], &[0x09, 0x07, 0x0F, 0x08]);
        assert!(bytes[4..].iter().all(|b| *b == 0), "zero nibbles stay zero, 64 lanes in all");
    }

    fn extract_lane16(v: &[u32; 16], lane: usize) -> u16 {
        (v[lane / 2] >> (16 * (lane % 2))) as u16
    }

    // -- extract_lane / insert_lanes --

    #[test]
    fn extract_insert_roundtrip_16bit() {
        // 16 lanes of 16-bit values in 256 bits (8 x u32)
        let values: Vec<i64> = vec![1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16];
        let mut reg = [0u32; 8];
        insert_lanes(&mut reg, &values, 16);

        for (i, &expected) in values.iter().enumerate() {
            let got = extract_lane(&reg, i, 16, true);
            assert_eq!(got, expected, "lane {} mismatch", i);
        }
    }

    #[test]
    fn extract_insert_roundtrip_8bit() {
        // 32 lanes of 8-bit values in 256 bits (8 x u32)
        let values: Vec<i64> = (0..32).map(|i| (i % 256) as i64).collect();
        let mut reg = [0u32; 8];
        insert_lanes(&mut reg, &values, 8);

        for (i, &expected) in values.iter().enumerate() {
            let got = extract_lane(&reg, i, 8, false);
            assert_eq!(got, expected, "lane {} mismatch", i);
        }
    }

    #[test]
    fn extract_insert_roundtrip_32bit() {
        // 16 lanes of 32-bit values (uses full 512 bits)
        // But our register is only 8 x u32 = 256 bits, so 8 lanes.
        let values: Vec<i64> = vec![100, -200, 300, -400, 500, -600, 700, -800];
        let mut reg = [0u32; 8];
        insert_lanes(&mut reg, &values, 32);

        for (i, &expected) in values.iter().enumerate() {
            let got = extract_lane(&reg, i, 32, true);
            assert_eq!(got, expected, "lane {} mismatch", i);
        }
    }

    // -- truncate edge cases --

    #[test]
    fn truncate_64bit_passthrough() {
        // bits >= 64: no masking, value passes through unchanged
        assert_eq!(truncate(i64::MAX, true, 64), i64::MAX);
        assert_eq!(truncate(i64::MIN, true, 64), i64::MIN);
        assert_eq!(truncate(i64::MAX, false, 64), i64::MAX);
    }

    #[test]
    fn truncate_unsigned_high_bits_stripped() {
        // 0xFFFF_FFFF in unsigned 16-bit = 0xFFFF (65535)
        assert_eq!(truncate(0xFFFF_FFFF, false, 16), 0xFFFF);
        // Same value in signed 16-bit = -1
        assert_eq!(truncate(0xFFFF_FFFF, true, 16), -1);
    }

    #[test]
    fn truncate_4bit() {
        // 4-bit unsigned: 0..15
        assert_eq!(truncate(0xAB, false, 4), 0xB);
        // 4-bit signed: -8..7
        assert_eq!(truncate(0xF, true, 4), -1);
        assert_eq!(truncate(0x8, true, 4), -8);
        assert_eq!(truncate(0x7, true, 4), 7);
    }

    // -- pack_lane 4-bit --

    #[test]
    fn pack_lane_8_to_4_truncate() {
        assert_eq!(pack_lane(0xAB, 8, 4, false, PackMode::Truncate), 0xB);
    }

    #[test]
    fn pack_lane_8_to_4_signed_saturate() {
        // Signed 4-bit range: [-8, 7]
        assert_eq!(pack_lane(10, 8, 4, true, PackMode::Saturate), 7);
        assert_eq!(pack_lane(-10, 8, 4, true, PackMode::Saturate), -8);
        assert_eq!(pack_lane(5, 8, 4, true, PackMode::Saturate), 5);
    }

    #[test]
    fn pack_lane_8_to_4_symmetric_saturate() {
        // Symmetric signed 4-bit: [-7, 7] (not -8)
        assert_eq!(pack_lane(-10, 8, 4, true, PackMode::SymmetricSaturate), -7);
        assert_eq!(pack_lane(-8, 8, 4, true, PackMode::SymmetricSaturate), -7);
        assert_eq!(pack_lane(-7, 8, 4, true, PackMode::SymmetricSaturate), -7);
    }

    #[test]
    fn pack_lane_symmetric_unsigned_same_as_regular() {
        // Unsigned symmetric saturate == regular saturate
        assert_eq!(
            pack_lane(300, 16, 8, false, PackMode::SymmetricSaturate),
            pack_lane(300, 16, 8, false, PackMode::Saturate),
        );
        assert_eq!(
            pack_lane(-1, 16, 8, false, PackMode::SymmetricSaturate),
            pack_lane(-1, 16, 8, false, PackMode::Saturate),
        );
    }

    // -- extract_lane cross-boundary --

    #[test]
    fn extract_lane_crosses_u32_boundary() {
        // 24-bit lanes: lane 1 starts at bit 24, crossing words 0 and 1.
        // Needs 24 bits from bit 24: 8 from word 0, 16 from word 1.
        let mut reg = [0u32; 8];
        reg[0] = 0xAB_00_00_00; // bits [24..32) = 0xAB
        reg[1] = 0x0000_CDEF; // bits [32..48) = 0xCDEF -> combined [24..48) = 0xCDEFAB
        let val = extract_lane(&reg, 1, 24, false);
        // word 0 >> 24 = 0xAB, bits_from_first = 8
        // word 1 << 8 = 0x00CDEF00, OR'd -> raw = 0x00CDEFAB
        // mask = 0xFFFFFF -> 0xCDEFAB
        assert_eq!(val, 0xCDEFAB);
    }

    #[test]
    fn extract_lane_out_of_bounds_returns_zero() {
        let reg = [0xFFFF_FFFFu32; 8];
        // Lane 100 at 32-bit -> bit_offset = 3200, word_idx = 100, >= 8
        assert_eq!(extract_lane(&reg, 100, 32, false), 0);
    }

    // -- insert_lanes cross-boundary --

    #[test]
    fn insert_lanes_crosses_u32_boundary() {
        // Insert a 24-bit value at lane 1 (bit_offset=24), crossing words 0-1.
        let mut reg = [0u32; 8];
        let lanes = vec![0i64, 0xABCDEF]; // lane 0 = 0, lane 1 = 0xABCDEF
        insert_lanes(&mut reg, &lanes, 24);

        // Lane 1 at bit 24: low 8 bits in word 0 [24..32), high 16 bits in word 1 [0..16)
        assert_eq!(reg[0] & 0xFF00_0000, 0xEF00_0000);
        assert_eq!(reg[1] & 0x0000_FFFF, 0x0000_ABCD);
    }

    // -- pack: one lane map for every form --

    fn lanes16(vals: &[u16]) -> [u32; 16] {
        let mut src = [0u32; 16];
        for (i, v) in vals.iter().enumerate() {
            src[i / 2] |= (*v as u32) << (16 * (i % 2));
        }
        src
    }

    fn result_bytes(r: &[u32; 8]) -> Vec<u8> {
        r.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn form(mnemonic: &str) -> &'static PackForm {
        pack_form(mnemonic).unwrap_or_else(|| panic!("{mnemonic}"))
    }

    const CR_PACK_SIGN: u8 = 5;

    #[test]
    fn for_pack_follows_the_silicon_table() {
        use PackMode::*;
        // (crSat, signed) -> mode, as captured: crSat 2 saturates unsigned lanes only.
        for (crsat, signed, want) in [
            (0, true, Truncate),
            (1, true, Saturate),
            (2, true, Truncate),
            (3, true, SymmetricSaturate),
            (0, false, Truncate),
            (1, false, Saturate),
            (2, false, Saturate),
            (3, false, Saturate),
            (4 + 3, true, SymmetricSaturate), // only the low two bits matter
        ] {
            assert_eq!(PackMode::for_pack(crsat, signed), want, "crSat {crsat} signed {signed}");
        }
    }

    #[test]
    fn crpacksign_decides_how_a_d_form_reads_the_input() {
        // int16 -1 under d8.d16 with saturation: unsigned reads 65535 and clamps to 255,
        // signed reads -1 and keeps it.
        let src = lanes16(&[0xFFFF, 300, 0x8000]);
        let f = form("vpack.d8.d16");
        let run = |sign: u32| {
            result_bytes(&pack_register(&src, f, 1, |id| {
                assert_eq!(id, CR_PACK_SIGN);
                sign
            }))
        };
        assert_eq!(&run(0)[..3], &[255, 255, 255]);
        assert_eq!(&run(1)[..3], &[0xFF, 127, 0x80]);
    }

    #[test]
    fn s_forms_do_not_read_the_sign_register() {
        let src = lanes16(&[0xFFFF, 200]);
        let f = form("vst.pack.s8.s16");
        let r = pack_register(&src, f, 1, |_| panic!("an s form must not read crPackSign"));
        assert_eq!(&result_bytes(&r)[..2], &[0xFF, 127]);
    }

    #[test]
    fn four_bit_results_are_packed_low_nibble_first() {
        let mut src = [0u32; 16];
        src[0] = 0x0000_0201; // byte lanes 0, 1 = 1, 2
        let r = pack_register(&src, form("vpack.s4.s8"), 0, |_| 0);
        assert_eq!(r[0] & 0xFF, 0x21);
    }
}
