//! Vector pack/unpack operations for AIE2.
//!
//! Pack narrows vector lanes from a wider type to a narrower type (e.g. 32-bit
//! to 16-bit). Unpack widens lanes (e.g. 16-bit to 32-bit). Both operate on
//! 512-bit vector registers.
//!
//! Pack supports three modes:
//! - **Truncate**: simply discard upper bits (no saturation).
//! - **Saturate**: clamp to the destination type range before truncating.
//!   For signed destinations: [-2^(n-1), 2^(n-1)-1].
//!   For unsigned destinations: [0, 2^n - 1].
//! - **SymmetricSaturate**: signed saturation with symmetric bounds:
//!   [-(2^(n-1) - 1), 2^(n-1) - 1].
//!
//! Unpack is always a widening operation: sign-extend (signed) or zero-extend
//! (unsigned) from the source width to the destination width. [`unpack_lanes`] is the
//! one lane map behind both the standalone `vunpack` and the fused `vldb.unpack`; the
//! per-form facts (widths, sign source) come from `xdna_archspec::aie2::unpack`.

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
    /// Select the narrow mode from the live crSat saturation flags
    /// (Core_CR [1:0]): bit 0 = saturate, bit 1 = symmetric saturate.
    /// The hardware narrowing pack reads crSat (`VPACK_* Uses = [crSat]`, llvm-aie
    /// TableGen) and SRS does the same, so the emulator derives the mode from the
    /// live saturation register rather than assuming truncation. Confirmed on real
    /// NPU1 silicon: a saturating crSat clamps int16->int8 (200 -> 127), it does
    /// not take the low byte.
    ///
    /// 0 -> Truncate, 1 -> Saturate, 3 -> SymmetricSaturate.
    pub fn from_sat_flags(saturate: bool, symmetric: bool) -> PackMode {
        match (saturate, symmetric) {
            (false, _) => PackMode::Truncate,
            (true, false) => PackMode::Saturate,
            (true, true) => PackMode::SymmetricSaturate,
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

/// Pack a full 512-bit vector: narrow each lane from `bits_i` to `bits_o`.
///
/// The input is treated as 512/bits_i lanes of `bits_i`-bit values packed
/// into the 256-bit (8 x u32) register representation. The output contains
/// 512/bits_i lanes of `bits_o`-bit values, repacked into 8 x u32.
pub fn pack_vector(src: &[u32; 8], bits_i: u32, bits_o: u32, signed: bool, mode: PackMode) -> [u32; 8] {
    let lanes = (512 / bits_i) as usize;
    let mut narrowed = vec![0i64; lanes];

    // Extract each lane at input width, pack it.
    for i in 0..lanes {
        let val = extract_lane(src, i, bits_i, signed);
        narrowed[i] = pack_lane(val, bits_i, bits_o, signed, mode);
    }

    // Repack narrowed lanes at output width into the result register.
    let mut result = [0u32; 8];
    insert_lanes(&mut result, &narrowed, bits_o);
    result
}

/// Check if a token is a type token (D4, S8, D16, S32, etc.).
fn is_type_token(token: &str) -> bool {
    (token.starts_with('D') || token.starts_with('S'))
        && token.len() >= 2
        && token[1..].parse::<u32>().is_ok()
}

/// Find consecutive type token pair in parts, returning their indices.
fn find_type_pair(parts: &[&str]) -> Option<(usize, usize)> {
    for i in 0..parts.len().saturating_sub(1) {
        if is_type_token(parts[i]) && is_type_token(parts[i + 1]) {
            return Some((i, i + 1));
        }
    }
    None
}

/// Parse VPACK bit widths from encoding name.
///
/// Handles standalone `VPACK_{OUT}_{IN}` and fused `VST_PACK_{OUT}_{IN}_*`.
/// Type tokens are {D|S}{bits} (e.g., D4, S8, D16, S32).
/// Returns (bits_in, bits_out, signed).
pub fn pack_widths_from_name(name: &str) -> (u32, u32, bool) {
    let upper = name.to_uppercase();
    // Split on both '_' and '.' to handle encoding names ("VPACK_D4_D8")
    // and mnemonics ("vpack.d4.d8").
    let parts: Vec<&str> = upper.split(|c| c == '_' || c == '.').collect();
    if let Some((out_idx, in_idx)) = find_type_pair(&parts) {
        let signed = parts[out_idx].starts_with('S') || parts[in_idx].starts_with('S');
        let out_bits: u32 = parts[out_idx][1..].parse().unwrap_or(8);
        let in_bits: u32 = parts[in_idx][1..].parse().unwrap_or(16);
        (in_bits, out_bits, signed)
    } else {
        (16, 8, false)
    }
}

/// Pack a 256-bit half: narrow 256/bits_i lanes from bits_i to bits_o.
///
/// Processes one 256-bit register, treating it as 256/bits_i lanes of
/// bits_i-width values, and packs each down to bits_o width.
pub fn pack_half(src: &[u32; 8], bits_i: u32, bits_o: u32, signed: bool, mode: PackMode) -> [u32; 8] {
    let lanes = (256 / bits_i) as usize;
    let mut narrowed = vec![0i64; lanes];
    for i in 0..lanes {
        let val = extract_lane(src, i, bits_i, signed);
        narrowed[i] = pack_lane(val, bits_i, bits_o, signed, mode);
    }
    let mut result = [0u32; 8];
    insert_lanes(&mut result, &narrowed, bits_o);
    result
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

/// Extract lane `idx` from a 256-bit (8 x u32) register at `bits` width.
///
/// Handles arbitrary bit widths including those that cross u32 boundaries.
fn extract_lane(reg: &[u32; 8], idx: usize, bits: u32, signed: bool) -> i64 {
    let bit_offset = idx as u32 * bits;
    let word_idx = (bit_offset / 32) as usize;
    let bit_idx = bit_offset % 32;

    // Gather enough bits -- may span two u32 words.
    if word_idx >= 8 {
        return 0; // Lane extends beyond the 256-bit register
    }
    let mut raw: u64 = reg[word_idx] as u64 >> bit_idx;
    let bits_from_first = 32 - bit_idx;
    if bits > bits_from_first && word_idx + 1 < 8 {
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
use xdna_archspec::aie2::unpack::unpack_form;

use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    /// Pack: narrow vector lanes.
    ///
    /// Narrow path: 2-source pack (legacy vector_pack helper).
    /// Wide path: 512->256, split source halves, pack each, combine.
    pub(super) fn execute_pack(op: &SlotOp, ctx: &mut ExecutionContext, _et: ElementType) -> bool {
        if op.is_wide_vector {
            // VPACK: 512-bit x-reg source -> 256-bit w-reg dest.
            // Each 256-bit half is packed independently, then the two
            // packed halves are concatenated into one 256-bit result.
            let name = op.encoding_name.as_deref().unwrap_or("");
            let (bits_i, bits_o, signed) = pack_widths_from_name(name);
            let src = Self::get_wide_vec_source(op, ctx, 0);
            let src_lo: [u32; 8] = src[..8].try_into().unwrap();
            let src_hi: [u32; 8] = src[8..].try_into().unwrap();

            // VPACK reads crSat (Uses = [crSat]); derive the narrow mode from
            // the live saturation register instead of assuming truncation.
            let mode =
                PackMode::from_sat_flags(ctx.srs_config.saturate(), ctx.srs_config.symmetric_saturate());
            let packed_lo = pack_half(&src_lo, bits_i, bits_o, signed, mode);
            let packed_hi = pack_half(&src_hi, bits_i, bits_o, signed, mode);

            // Each half produces (256/bits_i * bits_o) bits of packed data.
            // Concatenate the two halves into a single 256-bit w-register.
            let words_per_half = ((256 / bits_i) * bits_o / 32) as usize;
            let mut result = [0u32; 8];
            result[..words_per_half].copy_from_slice(&packed_lo[..words_per_half]);
            result[words_per_half..words_per_half * 2].copy_from_slice(&packed_hi[..words_per_half]);

            // Write to 256-bit w-register dest (NOT write_wide_vec_dest).
            Self::write_vector_dest(op, ctx, result);
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = Self::vector_pack(&a, &b);
            Self::write_vector_dest(op, ctx, result);
        }
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

    // -- pack_vector --

    #[test]
    fn pack_vector_32_to_16_truncate() {
        // 8 lanes of 32-bit -> 8 lanes of 16-bit (lower half of result).
        // Input: [0x10001, 0x20002, 0x30003, 0x40004, 0x50005, 0x60006, 0x70007, 0x80008]
        let mut src = [0u32; 8];
        let vals: Vec<i64> = vec![0x10001, 0x20002, 0x30003, 0x40004, 0x50005, 0x60006, 0x70007, 0x80008];
        insert_lanes(&mut src, &vals, 32);

        let result = pack_vector(&src, 32, 16, false, PackMode::Truncate);

        // Verify: each lane should be the low 16 bits
        for (i, &v) in vals.iter().enumerate() {
            let got = extract_lane(&result, i, 16, false);
            assert_eq!(got, v & 0xFFFF, "lane {} mismatch", i);
        }
    }

    #[test]
    fn pack_vector_16_to_8_saturate_signed() {
        // 16 lanes of 16-bit -> 16 lanes of 8-bit, signed saturate.
        // 256-bit register holds 16 lanes at 16-bit.
        let mut src = [0u32; 8];
        let vals: Vec<i64> = vec![
            50, -50, 127, -128, // in range for s8
            200, -200, 0, 1, // overflow positive and negative
            -1, 100, -100, 64, -64, 32, -32, 16,
        ];
        insert_lanes(&mut src, &vals, 16);

        let result = pack_vector(&src, 16, 8, true, PackMode::Saturate);

        let expected: Vec<i64> = vals.iter().map(|&v| v.clamp(-128, 127)).collect();
        for (i, &exp) in expected.iter().enumerate() {
            let got = extract_lane(&result, i, 8, true);
            assert_eq!(got, exp, "lane {} mismatch: expected {}, got {}", i, exp, got);
        }
    }

    // -- pack then unpack roundtrip --

    #[test]
    fn test_pack_widths_from_name() {
        // Standalone: VPACK_{OUT}_{IN}
        assert_eq!(pack_widths_from_name("VPACK_D4_D8"), (8, 4, false));
        assert_eq!(pack_widths_from_name("VPACK_S8_S16"), (16, 8, true));
        assert_eq!(pack_widths_from_name("VPACK_D8_D16"), (16, 8, false));
        // Fused: VST_PACK_{OUT}_{IN}_*
        assert_eq!(pack_widths_from_name("VST_PACK_D4_D8_ag_idx"), (8, 4, false));
        assert_eq!(pack_widths_from_name("VST_PACK_S4_S8_ag_pstm_nrm"), (8, 4, true));
        assert_eq!(pack_widths_from_name("VST_2D_PACK_S8_S16"), (16, 8, true));
        // Fallback for unrecognized format
        assert_eq!(pack_widths_from_name("VPACK"), (16, 8, false));
    }

    #[test]
    fn test_pack_half_d16_to_d8() {
        // 16 lanes of 16-bit values in a 256-bit register, pack to 8-bit.
        // Output: 16 lanes of 8-bit = 128 bits = 4 words.
        let mut src = [0u32; 8];
        let vals: Vec<i64> = (0..16).map(|i| (i * 10) as i64).collect();
        insert_lanes(&mut src, &vals, 16);

        let result = pack_half(&src, 16, 8, false, PackMode::Truncate);

        // Verify each output lane has the low 8 bits of the input.
        for (i, &expected) in vals.iter().enumerate() {
            let got = extract_lane(&result, i, 8, false);
            assert_eq!(got, expected & 0xFF, "lane {} mismatch", i);
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

    // -- is_type_token --

    #[test]
    fn is_type_token_valid() {
        assert!(is_type_token("D4"));
        assert!(is_type_token("D8"));
        assert!(is_type_token("D16"));
        assert!(is_type_token("D32"));
        assert!(is_type_token("S8"));
        assert!(is_type_token("S16"));
        assert!(is_type_token("S32"));
    }

    #[test]
    fn is_type_token_invalid() {
        assert!(!is_type_token("D")); // too short
        assert!(!is_type_token("VPACK")); // wrong prefix
        assert!(!is_type_token("DX")); // not a number
        assert!(!is_type_token("8")); // no prefix
        assert!(!is_type_token("")); // empty
        assert!(!is_type_token("AG")); // wrong prefix
    }

    // -- find_type_pair --

    #[test]
    fn find_type_pair_finds_consecutive() {
        let parts = vec!["VPACK", "D8", "D16", "ag"];
        assert_eq!(find_type_pair(&parts), Some((1, 2)));
    }

    #[test]
    fn find_type_pair_none_when_no_consecutive() {
        let parts = vec!["VPACK", "D8", "ag", "D16"];
        assert_eq!(find_type_pair(&parts), None);
    }

    #[test]
    fn find_type_pair_empty() {
        let parts: Vec<&str> = vec![];
        assert_eq!(find_type_pair(&parts), None);
    }

    #[test]
    fn find_type_pair_single_element() {
        let parts = vec!["D8"];
        assert_eq!(find_type_pair(&parts), None);
    }

    // -- name parsing: dot-separated mnemonics --

    #[test]
    fn pack_widths_from_name_dot_separated() {
        // Mnemonic format: vpack.d4.d8
        assert_eq!(pack_widths_from_name("vpack.d4.d8"), (8, 4, false));
        assert_eq!(pack_widths_from_name("vpack.s8.s16"), (16, 8, true));
    }

    // -- pack_half with saturation --

    #[test]
    fn pack_half_saturate_signed() {
        // 8 lanes of 32-bit, pack to 16-bit with signed saturation.
        // 256-bit / 32 = 8 input lanes.
        let mut src = [0u32; 8];
        let vals: Vec<i64> = vec![50000, -50000, 100, -100, 32767, -32768, 0, 1];
        insert_lanes(&mut src, &vals, 32);

        let result = pack_half(&src, 32, 16, true, PackMode::Saturate);

        let expected: Vec<i64> = vals.iter().map(|&v| v.clamp(-32768, 32767)).collect();
        for (i, &exp) in expected.iter().enumerate() {
            let got = extract_lane(&result, i, 16, true);
            assert_eq!(got, exp, "lane {} mismatch", i);
        }
    }

    // -- pack_vector 4-bit --

    #[test]
    fn pack_vector_8_to_4_truncate() {
        // 512/8 = 64 lanes at 8-bit, but our 256-bit reg holds 32 lanes.
        // pack_vector uses 512/bits_i lanes -> this is a 512-bit semantic.
        // With [u32; 8] = 256 bits, 256/8 = 32 lanes. pack_vector divides
        // 512/bits_i which for 8-bit = 64 lanes, exceeding the 8-word register.
        // So pack_vector is for conceptual 512-bit. Let's use pack_half instead.
        let mut src = [0u32; 8];
        let vals: Vec<i64> = (0..32).map(|i| (i * 3 % 256) as i64).collect();
        insert_lanes(&mut src, &vals, 8);

        let result = pack_half(&src, 8, 4, false, PackMode::Truncate);

        for (i, &v) in vals.iter().enumerate() {
            let got = extract_lane(&result, i, 4, false);
            assert_eq!(got, v & 0xF, "lane {} mismatch", i);
        }
    }
}
