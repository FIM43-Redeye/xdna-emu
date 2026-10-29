//! AIE2 vector permutation and shuffle engine.
//!
//! The AIE2 has two distinct data-rearrangement subsystems:
//!
//! 1. **Shuffle unit** (`vshuffle` instruction): operates on a 1024-bit pair
//!    (two 512-bit vectors concatenated) and performs transposition,
//!    interleaving, and deinterleaving. Controlled by a 6-bit mode number
//!    (0..47) passed as an immediate operand.
//!
//! 2. **MAC permutation engine**: routes input data to the 512 multiplier
//!    array before multiply-accumulate operations. The tile geometry (rows,
//!    cols, inner dimension) and element types come from the MAC config word;
//!    see `vector_config`.
//!
//! # Shuffle Modes
//!
//! Mode numbers follow llvm-aie's `aiev2_enums.h`; the byte routing behind
//! each mode is hardware-probed (`SHUFFLE_ROUTING`). The naming convention is `T{bits}_{rows}x{cols}_{lo|hi}`:
//!
//! - `T{bits}`: element width in bits (8, 16, 32, 64, 128, 256, 512)
//! - `{rows}x{cols}`: matrix dimensions being transposed
//! - `_lo` / `_hi`: which 512-bit half of the 1024-bit result to extract
//!
//! The shuffle unit conceptually arranges the 1024-bit input as a matrix
//! of elements, transposes it, then returns either the low or high 512-bit
//! half of the result.
//!
//! # Implementation
//!
//! All permutations operate at the byte level on 64-byte (512-bit) vectors.
//! Two input vectors are concatenated into a 128-byte workspace, the
//! transpose is applied, and 64 bytes are extracted from the result.

/// Number of bytes in a vector register (from archspec processor model).
const VEC_BYTES: usize = xdna_archspec::aie2::processor::VECTOR_REGISTER_BYTES;

/// Number of bytes in a vector pair (shuffle unit input, from archspec).
const PAIR_BYTES: usize = xdna_archspec::aie2::processor::VECTOR_PAIR_BYTES;

// Per-arch permute data (enums + tables + config lookups) lives in archspec.
// Only the execution algorithms that consume this data remain in xdna-emu.
use xdna_archspec::aie2::permute::{ShuffleMode, SHUFFLE_ROUTING};

// ============================================================================
// Shuffle execution
// ============================================================================

/// Execute a shuffle operation on two 512-bit vectors.
///
/// The two input vectors (`lo` and `hi`) are concatenated to form a 1024-bit
/// workspace. The specified mode transposes the data and returns 512 bits.
///
/// Vectors are represented as 64-byte arrays (little-endian byte order).
pub fn shuffle_vectors(lo: &[u8; VEC_BYTES], hi: &[u8; VEC_BYTES], mode: ShuffleMode) -> [u8; VEC_BYTES] {
    // Concatenate lo and hi into 128-byte input space.
    let mut input = [0u8; PAIR_BYTES];
    input[..VEC_BYTES].copy_from_slice(lo);
    input[VEC_BYTES..].copy_from_slice(hi);

    // Apply the hardware-verified routing table.
    // Each entry in SHUFFLE_ROUTING[mode][i] tells us which input byte
    // (0-127) appears at output position i.
    let route = &SHUFFLE_ROUTING[mode.as_u8() as usize];
    let mut out = [0u8; VEC_BYTES];
    for i in 0..VEC_BYTES {
        out[i] = input[route[i] as usize];
    }
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Shuffle mode tests
    // ========================================================================

    // Helper: build a 64-byte vector from a pattern function.
    fn make_vec<F: Fn(usize) -> u8>(f: F) -> [u8; VEC_BYTES] {
        let mut v = [0u8; VEC_BYTES];
        for i in 0..VEC_BYTES {
            v[i] = f(i);
        }
        v
    }

    // Helper: build an identity vector where byte i = i.
    fn identity_lo() -> [u8; VEC_BYTES] {
        make_vec(|i| i as u8)
    }

    fn identity_hi() -> [u8; VEC_BYTES] {
        make_vec(|i| (i + VEC_BYTES) as u8)
    }

    #[test]
    fn test_mode_roundtrip() {
        for i in 0..=47u8 {
            let mode = ShuffleMode::from_mode(i).unwrap();
            assert_eq!(mode.as_u8(), i);
        }
        assert!(ShuffleMode::from_mode(48).is_none());
    }

    #[test]
    fn test_bypass_lo() {
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T512_1x2Lo);
        assert_eq!(result, lo);
    }

    #[test]
    fn test_bypass_hi() {
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T512_1x2Hi);
        assert_eq!(result, hi);
    }

    #[test]
    fn test_t256_2x2_lo() {
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T256_2x2Lo);

        let mut expected = [0u8; VEC_BYTES];
        expected[..32].copy_from_slice(&lo[..32]);
        expected[32..64].copy_from_slice(&hi[..32]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_t256_2x2_hi() {
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T256_2x2Hi);

        let mut expected = [0u8; VEC_BYTES];
        expected[..32].copy_from_slice(&lo[32..64]);
        expected[32..64].copy_from_slice(&hi[32..64]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_t32_16x2_lo_deinterleave() {
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T32_16x2Lo);

        let mut input = [0u8; PAIR_BYTES];
        input[..VEC_BYTES].copy_from_slice(&lo);
        input[VEC_BYTES..].copy_from_slice(&hi);

        for i in 0..16 {
            let src_elem = i * 2;
            let src_off = src_elem * 4;
            let dst_off = i * 4;
            assert_eq!(
                &result[dst_off..dst_off + 4],
                &input[src_off..src_off + 4],
                "mismatch at output element {i}"
            );
        }
    }

    #[test]
    fn test_t32_16x2_hi_deinterleave() {
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T32_16x2Hi);

        let mut input = [0u8; PAIR_BYTES];
        input[..VEC_BYTES].copy_from_slice(&lo);
        input[VEC_BYTES..].copy_from_slice(&hi);

        for i in 0..16 {
            let src_elem = i * 2 + 1;
            let src_off = src_elem * 4;
            let dst_off = i * 4;
            assert_eq!(
                &result[dst_off..dst_off + 4],
                &input[src_off..src_off + 4],
                "mismatch at output element {i}"
            );
        }
    }

    #[test]
    fn test_t8_64x2_deinterleave() {
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T8_64x2Lo);

        let mut input = [0u8; PAIR_BYTES];
        input[..VEC_BYTES].copy_from_slice(&lo);
        input[VEC_BYTES..].copy_from_slice(&hi);

        for i in 0..64 {
            assert_eq!(
                result[i],
                input[i * 2],
                "mismatch at byte {i}: expected input[{}]={}, got {}",
                i * 2,
                input[i * 2],
                result[i]
            );
        }
    }

    #[test]
    fn test_t8_2x64_interleave() {
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T8_2x64Lo);

        let mut input = [0u8; PAIR_BYTES];
        input[..VEC_BYTES].copy_from_slice(&lo);
        input[VEC_BYTES..].copy_from_slice(&hi);

        for c in 0..32 {
            for r in 0..2 {
                let dst_byte = c * 2 + r;
                let src_byte = r * 64 + c;
                assert_eq!(
                    result[dst_byte], input[src_byte],
                    "mismatch at byte {dst_byte}: expected input[{src_byte}]"
                );
            }
        }
    }

    #[test]
    fn test_t16_1x2_flip() {
        let mut lo = [0u8; VEC_BYTES];
        for i in 0..32 {
            let base = i * 2;
            lo[base] = (i * 2) as u8;
            lo[base + 1] = (i * 2 + 1) as u8;
        }
        let hi = [0u8; VEC_BYTES];
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T16_1x2Flip);

        for i in 0..16 {
            let base = i * 4;
            assert_eq!(result[base], lo[base + 2]);
            assert_eq!(result[base + 1], lo[base + 3]);
            assert_eq!(result[base + 2], lo[base]);
            assert_eq!(result[base + 3], lo[base + 1]);
        }
    }

    #[test]
    fn test_t32_4x4_transpose() {
        let lo = identity_lo();
        let hi = [0u8; VEC_BYTES];
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T32_4x4);

        for r in 0..4 {
            for c in 0..4 {
                let src_off = (r * 4 + c) * 4;
                let dst_off = (c * 4 + r) * 4;
                assert_eq!(&result[dst_off..dst_off + 4], &lo[src_off..src_off + 4], "mismatch at ({r},{c})");
            }
        }
    }

    #[test]
    fn test_t8_8x8_transpose() {
        let lo = identity_lo();
        let hi = [0u8; VEC_BYTES];
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T8_8x8);

        for r in 0..8 {
            for c in 0..8 {
                let src = r * 8 + c;
                let dst = c * 8 + r;
                assert_eq!(
                    result[dst], lo[src],
                    "mismatch at ({r},{c}): result[{dst}]={} expected lo[{src}]={}",
                    result[dst], lo[src]
                );
            }
        }
    }

    #[test]
    fn test_t16_4x4_exact() {
        let lo = identity_lo();
        let hi = [0u8; VEC_BYTES];
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T16_4x4);

        for r in 0..4 {
            for c in 0..4 {
                let src_off = (r * 4 + c) * 2;
                let dst_off = (c * 4 + r) * 2;
                assert_eq!(&result[dst_off..dst_off + 2], &lo[src_off..src_off + 2], "mismatch at ({r},{c})");
            }
        }
    }

    // Verify a known common shuffle pattern: 32-bit deinterleave.
    #[test]
    fn test_deinterleave_32bit_pattern() {
        let mut lo = [0u8; VEC_BYTES];
        let mut hi = [0u8; VEC_BYTES];
        for i in 0..16u32 {
            lo[i as usize * 4..i as usize * 4 + 4].copy_from_slice(&i.to_le_bytes());
        }
        for i in 0..16u32 {
            hi[i as usize * 4..i as usize * 4 + 4].copy_from_slice(&(i + 16).to_le_bytes());
        }

        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T32_16x2Lo);

        for i in 0..16 {
            let val =
                u32::from_le_bytes([result[i * 4], result[i * 4 + 1], result[i * 4 + 2], result[i * 4 + 3]]);
            let expected = (i * 2) as u32;
            assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
        }
    }

    // Verify interleave is the inverse of deinterleave.
    #[test]
    fn test_interleave_deinterleave_roundtrip_16bit() {
        let lo = identity_lo();
        let hi = identity_hi();

        let evens = shuffle_vectors(&lo, &hi, ShuffleMode::T16_32x2Lo);
        let odds = shuffle_vectors(&lo, &hi, ShuffleMode::T16_32x2Hi);

        let reconstructed_lo = shuffle_vectors(&evens, &odds, ShuffleMode::T16_2x32Lo);
        let reconstructed_hi = shuffle_vectors(&evens, &odds, ShuffleMode::T16_2x32Hi);

        assert_eq!(reconstructed_lo, lo, "lo mismatch after roundtrip");
        assert_eq!(reconstructed_hi, hi, "hi mismatch after roundtrip");
    }

    // ========================================================================
    // Additional shuffle edge case tests
    // ========================================================================

    /// Verify that all 48 shuffle modes are accessible.
    #[test]
    fn test_all_shuffle_modes_execute() {
        let lo = identity_lo();
        let hi = identity_hi();

        for i in 0..=47u8 {
            let mode = ShuffleMode::from_mode(i).unwrap();
            let result = shuffle_vectors(&lo, &hi, mode);
            // Just verify it produces 64 bytes without panicking
            assert_eq!(result.len(), VEC_BYTES, "mode {i}: result length mismatch");
        }
    }

    /// Verify all deinterleave+interleave roundtrips (8, 16, 32, 64, 128
    /// bit element sizes).
    #[test]
    fn test_all_deinterleave_roundtrips() {
        let lo = identity_lo();
        let hi = identity_hi();

        // 8-bit: T8_64x2 deinterleave, T8_2x64 interleave
        let evens = shuffle_vectors(&lo, &hi, ShuffleMode::T8_64x2Lo);
        let odds = shuffle_vectors(&lo, &hi, ShuffleMode::T8_64x2Hi);
        let rlo = shuffle_vectors(&evens, &odds, ShuffleMode::T8_2x64Lo);
        let rhi = shuffle_vectors(&evens, &odds, ShuffleMode::T8_2x64Hi);
        assert_eq!(rlo, lo, "8-bit roundtrip lo mismatch");
        assert_eq!(rhi, hi, "8-bit roundtrip hi mismatch");

        // 32-bit: T32_16x2 deinterleave, T32_2x16 interleave
        let evens = shuffle_vectors(&lo, &hi, ShuffleMode::T32_16x2Lo);
        let odds = shuffle_vectors(&lo, &hi, ShuffleMode::T32_16x2Hi);
        let rlo = shuffle_vectors(&evens, &odds, ShuffleMode::T32_2x16Lo);
        let rhi = shuffle_vectors(&evens, &odds, ShuffleMode::T32_2x16Hi);
        assert_eq!(rlo, lo, "32-bit roundtrip lo mismatch");
        assert_eq!(rhi, hi, "32-bit roundtrip hi mismatch");

        // 64-bit: T64_8x2 deinterleave, T64_2x8 interleave
        let evens = shuffle_vectors(&lo, &hi, ShuffleMode::T64_8x2Lo);
        let odds = shuffle_vectors(&lo, &hi, ShuffleMode::T64_8x2Hi);
        let rlo = shuffle_vectors(&evens, &odds, ShuffleMode::T64_2x8Lo);
        let rhi = shuffle_vectors(&evens, &odds, ShuffleMode::T64_2x8Hi);
        assert_eq!(rlo, lo, "64-bit roundtrip lo mismatch");
        assert_eq!(rhi, hi, "64-bit roundtrip hi mismatch");

        // 128-bit: T128_4x2 deinterleave, T128_2x4 interleave
        let evens = shuffle_vectors(&lo, &hi, ShuffleMode::T128_4x2Lo);
        let odds = shuffle_vectors(&lo, &hi, ShuffleMode::T128_4x2Hi);
        let rlo = shuffle_vectors(&evens, &odds, ShuffleMode::T128_2x4Lo);
        let rhi = shuffle_vectors(&evens, &odds, ShuffleMode::T128_2x4Hi);
        assert_eq!(rlo, lo, "128-bit roundtrip lo mismatch");
        assert_eq!(rhi, hi, "128-bit roundtrip hi mismatch");

        // 256-bit: T256_2x2
        let a = shuffle_vectors(&lo, &hi, ShuffleMode::T256_2x2Lo);
        let b = shuffle_vectors(&lo, &hi, ShuffleMode::T256_2x2Hi);
        let rlo = shuffle_vectors(&a, &b, ShuffleMode::T256_2x2Lo);
        let rhi = shuffle_vectors(&a, &b, ShuffleMode::T256_2x2Hi);
        assert_eq!(rlo, lo, "256-bit roundtrip lo mismatch");
        assert_eq!(rhi, hi, "256-bit roundtrip hi mismatch");
    }

    /// Verify that zero-filled inputs produce zero-filled outputs for
    /// all shuffle modes.
    #[test]
    fn test_shuffle_zero_input() {
        let zero = [0u8; VEC_BYTES];

        for i in 0..=47u8 {
            let mode = ShuffleMode::from_mode(i).unwrap();
            let result = shuffle_vectors(&zero, &zero, mode);
            assert_eq!(result, zero, "mode {i}: zero input should produce zero output");
        }
    }

    /// Verify that shuffle mode 26 (T16_4x16Lo) with (broadcast, zeros) input
    /// matches hardware VBCSTSHFL output for the r29!=0 case.
    ///
    /// On NPU1, VBCSTSHFL behaves as a shuffle of (broadcast, zeros) with
    /// mode r29.
    ///
    /// For r29=0, our shuffle mode 0 doesn't match hardware (the intlv_hw
    /// combinational circuit differs from our transpose_extract model).
    /// We use an empirical transpose model for r29=0 instead.
    ///
    /// For r29!=0, shuffle_vectors(broadcast, zeros, mode) works correctly,
    /// verified against NPU1 hardware with mode 26 (T16_4x16Lo).
    #[test]
    fn test_vbcstshfl_mode26_matches_hw() {
        let zero = [0u8; VEC_BYTES];

        // VBCSTSHFL_64 combo 7 from batch 30: r29=0x45bac25a, mode=26
        // Broadcast u16 components: a1d6, ea99, c25a, 45ba
        let comp: [u8; 8] = [0xd6, 0xa1, 0x99, 0xea, 0x5a, 0xc2, 0xba, 0x45];
        let mut bcast = [0u8; VEC_BYTES];
        for i in 0..8 {
            bcast[i * 8..i * 8 + 8].copy_from_slice(&comp);
        }

        // HW output: [A A 0 0 B B 0 0 C C 0 0 D D 0 0] * 2
        let mut hw = [0u8; VEC_BYTES];
        let comps: [[u8; 2]; 4] = [[0xd6, 0xa1], [0x99, 0xea], [0x5a, 0xc2], [0xba, 0x45]];
        for half in 0..2 {
            let base = half * 32;
            for (c, comp) in comps.iter().enumerate() {
                hw[base + c * 8] = comp[0];
                hw[base + c * 8 + 1] = comp[1];
                hw[base + c * 8 + 2] = comp[0];
                hw[base + c * 8 + 3] = comp[1];
            }
        }

        let result = shuffle_vectors(&bcast, &zero, ShuffleMode::T16_4x16Lo);
        assert_eq!(
            result, hw,
            "shuffle(broadcast.64, zeros, T16_4x16Lo) must match NPU1 VBCSTSHFL_64 output"
        );
    }
}
