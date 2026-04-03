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
//! (unsigned) from the source width to the destination width.

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
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
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

/// Unpack a single lane: widen from `bits_i` to `bits_o`.
///
/// If `signed`, sign-extends from `bits_i` to `bits_o`.
/// If unsigned, zero-extends (upper bits are already zero after truncation).
pub fn unpack_lane(value: i64, bits_i: u32, bits_o: u32, signed: bool) -> i64 {
    // First normalize the input to bits_i width with correct sign/zero extension.
    // This ensures the sign bit at position (bits_i - 1) is propagated correctly.
    // Then truncate to bits_o, which is a no-op widening since bits_o >= bits_i.
    let normalized = truncate(value, signed, bits_i);
    truncate(normalized, signed, bits_o)
}

/// Pack a full 512-bit vector: narrow each lane from `bits_i` to `bits_o`.
///
/// The input is treated as 512/bits_i lanes of `bits_i`-bit values packed
/// into the 256-bit (8 x u32) register representation. The output contains
/// 512/bits_i lanes of `bits_o`-bit values, repacked into 8 x u32.
pub fn pack_vector(
    src: &[u32; 8],
    bits_i: u32,
    bits_o: u32,
    signed: bool,
    mode: PackMode,
) -> [u32; 8] {
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

/// Unpack a full 512-bit vector: widen each lane from `bits_i` to `bits_o`.
///
/// The input is treated as 512/bits_o lanes of `bits_i`-bit values. Each
/// lane is widened to `bits_o` bits. The result contains 512/bits_o lanes.
pub fn unpack_vector(
    src: &[u32; 8],
    bits_i: u32,
    bits_o: u32,
    signed: bool,
) -> [u32; 8] {
    // Number of output lanes that fit in a 512-bit register
    let lanes = (512 / bits_o) as usize;
    let mut widened = vec![0i64; lanes];

    // Extract each lane at input width, widen it.
    for i in 0..lanes {
        let val = extract_lane(src, i, bits_i, signed);
        widened[i] = unpack_lane(val, bits_i, bits_o, signed);
    }

    // Repack widened lanes at output width into the result register.
    let mut result = [0u32; 8];
    insert_lanes(&mut result, &widened, bits_o);
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

/// Parse VUNPACK bit widths from encoding name.
///
/// Handles standalone `VUNPACK_{OUT}_{IN}` and fused `VLDB_UNPACK_{OUT}_{IN}_*`.
/// Type tokens are {D|S}{bits} (e.g., D8, S16, D32).
/// Returns (bits_in, bits_out, signed).
pub fn unpack_widths_from_name(name: &str) -> (u32, u32, bool) {
    let upper = name.to_uppercase();
    // Split on both '_' and '.' to handle encoding names ("VUNPACK_S16_S8")
    // and mnemonics ("vunpack.s16.s8").
    let parts: Vec<&str> = upper.split(|c| c == '_' || c == '.').collect();
    if let Some((out_idx, in_idx)) = find_type_pair(&parts) {
        let signed = parts[out_idx].starts_with('S') || parts[in_idx].starts_with('S');
        let out_bits: u32 = parts[out_idx][1..].parse().unwrap_or(16);
        let in_bits: u32 = parts[in_idx][1..].parse().unwrap_or(8);
        (in_bits, out_bits, signed)
    } else {
        (8, 16, false)
    }
}

/// Pack a 256-bit half: narrow 256/bits_i lanes from bits_i to bits_o.
///
/// Processes one 256-bit register, treating it as 256/bits_i lanes of
/// bits_i-width values, and packs each down to bits_o width.
pub fn pack_half(
    src: &[u32; 8],
    bits_i: u32,
    bits_o: u32,
    signed: bool,
    mode: PackMode,
) -> [u32; 8] {
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

/// Unpack lanes from a 256-bit source starting at lane_start, widening to bits_o.
///
/// Extracts lanes at bits_i width starting from lane_start, widens each to
/// bits_o, and packs the results into a 256-bit output register. The number
/// of output lanes is 256/bits_o.
pub fn unpack_half(
    src: &[u32; 8],
    lane_start: usize,
    bits_i: u32,
    bits_o: u32,
    signed: bool,
) -> [u32; 8] {
    let out_lanes = (256 / bits_o) as usize;
    let mut widened = vec![0i64; out_lanes];
    for i in 0..out_lanes {
        let val = extract_lane(src, lane_start + i, bits_i, signed);
        widened[i] = unpack_lane(val, bits_i, bits_o, signed);
    }
    let mut result = [0u32; 8];
    insert_lanes(&mut result, &widened, bits_o);
    result
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

#[cfg(test)]
mod tests {
    use super::*;

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

    // -- unpack_lane --

    #[test]
    fn unpack_lane_signed_extends() {
        // -1 in 8-bit signed, extended to 32-bit = -1
        let r = unpack_lane(-1, 8, 32, true);
        assert_eq!(r, -1);
    }

    #[test]
    fn unpack_lane_unsigned_zero_extends() {
        // 0xFF (255) in 8-bit unsigned, extended to 32-bit = 255
        let r = unpack_lane(255, 8, 32, false);
        assert_eq!(r, 255);
    }

    #[test]
    fn unpack_lane_positive_value() {
        // 42 in 8-bit signed, extended to 16-bit = 42
        let r = unpack_lane(42, 8, 16, true);
        assert_eq!(r, 42);
    }

    // -- extract_lane / insert_lanes --

    #[test]
    fn extract_insert_roundtrip_16bit() {
        // 16 lanes of 16-bit values in 256 bits (8 x u32)
        let values: Vec<i64> = vec![1, -2, 3, -4, 5, -6, 7, -8,
                                     9, -10, 11, -12, 13, -14, 15, -16];
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
        let vals: Vec<i64> = vec![0x10001, 0x20002, 0x30003, 0x40004,
                                   0x50005, 0x60006, 0x70007, 0x80008];
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
            50, -50, 127, -128,     // in range for s8
            200, -200, 0, 1,        // overflow positive and negative
            -1, 100, -100, 64,
            -64, 32, -32, 16,
        ];
        insert_lanes(&mut src, &vals, 16);

        let result = pack_vector(&src, 16, 8, true, PackMode::Saturate);

        let expected: Vec<i64> = vals.iter().map(|&v| v.clamp(-128, 127)).collect();
        for (i, &exp) in expected.iter().enumerate() {
            let got = extract_lane(&result, i, 8, true);
            assert_eq!(got, exp, "lane {} mismatch: expected {}, got {}", i, exp, got);
        }
    }

    // -- unpack_vector --

    #[test]
    fn unpack_vector_16_to_32_signed() {
        // 8 lanes of 16-bit signed -> 8 lanes of 32-bit.
        let mut src = [0u32; 8];
        let vals: Vec<i64> = vec![1, -1, 32767, -32768, 0, 100, -100, 42];
        insert_lanes(&mut src, &vals, 16);

        let result = unpack_vector(&src, 16, 32, true);

        for (i, &expected) in vals.iter().enumerate() {
            let got = extract_lane(&result, i, 32, true);
            assert_eq!(got, expected, "lane {} mismatch", i);
        }
    }

    #[test]
    fn unpack_vector_8_to_16_unsigned() {
        // 256-bit register: 32 lanes at 8-bit, but only 16 lanes fit at 16-bit output.
        let mut src = [0u32; 8];
        let vals: Vec<i64> = (0..32).map(|i| (i * 8) as i64).collect();
        insert_lanes(&mut src, &vals, 8);

        let result = unpack_vector(&src, 8, 16, false);
        let output_lanes = 256 / 16; // 16 lanes fit in 256-bit output

        for i in 0..output_lanes {
            let got = extract_lane(&result, i, 16, false);
            assert_eq!(got, vals[i], "lane {} mismatch", i);
        }
    }

    #[test]
    fn unpack_vector_8_to_16_signed_negative() {
        // 256-bit register: 32 lanes at 8-bit, only 16 fit at 16-bit output.
        let mut src = [0u32; 8];
        let mut vals: Vec<i64> = Vec::new();
        for i in 0..32 {
            let v = if i % 2 == 0 { i as i64 } else { -(i as i64) };
            vals.push(truncate(v, true, 8)); // Ensure 8-bit range
        }
        insert_lanes(&mut src, &vals, 8);

        let result = unpack_vector(&src, 8, 16, true);
        let output_lanes = 256 / 16;

        for i in 0..output_lanes {
            let got = extract_lane(&result, i, 16, true);
            assert_eq!(got, vals[i], "lane {} mismatch", i);
        }
    }

    // -- pack then unpack roundtrip --

    // -- pack_widths_from_name / unpack_widths_from_name --

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
    fn test_unpack_widths_from_name() {
        // Standalone: VUNPACK_{OUT}_{IN}
        assert_eq!(unpack_widths_from_name("VUNPACK_D16_D8"), (8, 16, false));
        assert_eq!(unpack_widths_from_name("VUNPACK_S32_S16"), (16, 32, true));
        // Fused: VLDB_UNPACK_{OUT}_{IN}_*
        assert_eq!(unpack_widths_from_name("VLDB_UNPACK_D8_D4_ag_idx"), (4, 8, false));
        assert_eq!(unpack_widths_from_name("VLDB_UNPACK_S16_S8_ag_pstm_nrm"), (8, 16, true));
        // Fallback for unrecognized format
        assert_eq!(unpack_widths_from_name("VUNPACK"), (8, 16, false));
    }

    // -- pack_half / unpack_half --

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

    #[test]
    fn test_unpack_half_d8_to_d16() {
        // 32 lanes of 8-bit values in a 256-bit register.
        // Unpack lanes 0..15 to 16-bit (unsigned).
        let mut src = [0u32; 8];
        let vals: Vec<i64> = (0..32).map(|i| (i * 7) as i64).collect();
        insert_lanes(&mut src, &vals, 8);

        let result = unpack_half(&src, 0, 8, 16, false);

        // Output: 16 lanes of 16-bit, each matching the original 8-bit value.
        for i in 0..16 {
            let got = extract_lane(&result, i, 16, false);
            assert_eq!(got, vals[i], "lane {} mismatch", i);
        }
    }

    #[test]
    fn test_unpack_half_d8_to_d16_upper() {
        // Unpack lanes 16..31 from a 256-bit register of 8-bit values.
        let mut src = [0u32; 8];
        let vals: Vec<i64> = (0..32).map(|i| (i * 3) as i64).collect();
        insert_lanes(&mut src, &vals, 8);

        let result = unpack_half(&src, 16, 8, 16, false);

        for i in 0..16 {
            let got = extract_lane(&result, i, 16, false);
            assert_eq!(got, vals[16 + i], "lane {} mismatch", i);
        }
    }

    // -- pack then unpack roundtrip --

    #[test]
    fn pack_unpack_roundtrip_no_overflow() {
        // Values that fit in 16-bit signed: pack 32->16, unpack 16->32 = identity.
        let mut src = [0u32; 8];
        let vals: Vec<i64> = vec![1, -1, 100, -100, 0, 32767, -32768, 42];
        insert_lanes(&mut src, &vals, 32);

        let packed = pack_vector(&src, 32, 16, true, PackMode::Truncate);
        let unpacked = unpack_vector(&packed, 16, 32, true);

        for (i, &expected) in vals.iter().enumerate() {
            let got = extract_lane(&unpacked, i, 32, true);
            assert_eq!(got, expected, "roundtrip lane {} mismatch", i);
        }
    }
}
