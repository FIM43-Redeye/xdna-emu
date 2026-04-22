//! AIE2 vector permutation and shuffle engine.
//!
//! The AIE2 has two distinct data-rearrangement subsystems:
//!
//! 1. **Shuffle unit** (`vshuffle` instruction): operates on a 1024-bit pair
//!    (two 512-bit vectors concatenated) and performs transposition,
//!    interleaving, and deinterleaving. Controlled by a 6-bit mode number
//!    (0..47) passed as an immediate operand.
//!
//! 2. **MAC permutation engine** (`PMODE_*`): routes input data to the 512
//!    multiplier array before multiply-accumulate operations. Controlled by
//!    a permute mode index that determines rows, cols, inner dimension,
//!    strides, and element types.
//!
//! # Shuffle Modes
//!
//! Mode numbers and semantics are derived from me_enums.h in the aietools
//! reference. The naming convention is `T{bits}_{rows}x{cols}_{lo|hi}`:
//!
//! - `T{bits}`: element width in bits (8, 16, 32, 64, 128, 256, 512)
//! - `{rows}x{cols}`: matrix dimensions being transposed
//! - `_lo` / `_hi`: which 512-bit half of the 1024-bit result to extract
//!
//! The shuffle unit conceptually arranges the 1024-bit input as a matrix
//! of elements, transposes it, then returns either the low or high 512-bit
//! half of the result.
//!
//! # MAC Permutation Modes
//!
//! All 26 MAC permutation modes from the aietools constants.py are
//! implemented. Cross-reference audit (2026-03-12):
//!
//! The aietools `Constants.__make_perm_modes()` generates exactly 26 modes
//! from 26 base configuration tuples (lines 268-300 of constants.py). The
//! stride/ordering iteration produces only one permutation per base config
//! because:
//! - `order_o == COL_MAJOR` and `order_y == COL_MAJOR` are always skipped
//! - `order_x == COL_MAJOR` is always skipped (commented-out allow list)
//! - Any stride > 1 is always skipped
//!
//! This yields modes 0-25 with all strides = 1 and all orderings = ROW_MAJOR.
//! Every mode in our `MacPermuteMode` enum has been verified to match its
//! corresponding aietools base configuration tuple field-by-field.
//!
//! # Implementation
//!
//! All permutations operate at the byte level on 64-byte (512-bit) vectors.
//! Two input vectors are concatenated into a 128-byte workspace, the
//! transpose is applied, and 64 bytes are extracted from the result.
//!
//! # MAC Permutation
//!
//! The MAC permutation engine is more complex and is used during matrix
//! multiply operations. It controls how X-buffer and Y-buffer elements are
//! routed to the multiplier array. Each mode defines a multi-dimensional
//! index function (`rc2i`) that computes source positions from (row, col,
//! inner, channel) coordinates.

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
// Multi-dimensional index computation (rc2i)
// ============================================================================

/// Row-major vs column-major ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    RowMajor,
    ColMajor,
}

/// Compute a linear index from (row, col, channel) coordinates.
///
/// This is the fundamental address function used by the MAC permutation
/// engine. It computes:
///
/// - Row-major: `((row_stride * row * cols + col) * col_stride * channels + channel) * channel_stride`
/// - Col-major: `((col_stride * col * rows + row) * row_stride * channels + channel) * channel_stride`
///
/// Derived from the hardware's multi-dimensional indexing behavior.
pub fn rc2i(
    row: u32,
    col: u32,
    channel: u32,
    rows: u32,
    cols: u32,
    _channels: u32,
    row_stride: u32,
    col_stride: u32,
    channel_stride: u32,
    order: Ordering,
) -> u32 {
    match order {
        Ordering::RowMajor => {
            ((row_stride * row * cols + col) * col_stride * _channels + channel) * channel_stride
        }
        Ordering::ColMajor => {
            ((col_stride * col * rows + row) * row_stride * _channels + channel) * channel_stride
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use xdna_archspec::aie2::permute::{MacPermuteMode, mac_permute_config};

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
                i * 2, input[i * 2], result[i]
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
                    result[dst_byte],
                    input[src_byte],
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
                assert_eq!(
                    &result[dst_off..dst_off + 4],
                    &lo[src_off..src_off + 4],
                    "mismatch at ({r},{c})"
                );
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
                assert_eq!(
                    &result[dst_off..dst_off + 2],
                    &lo[src_off..src_off + 2],
                    "mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn test_mac_permute_mode_roundtrip() {
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            assert_eq!(mode.as_u8(), i);
        }
        assert!(MacPermuteMode::from_mode(26).is_none());
    }

    #[test]
    fn test_mac_permute_config_consistency() {
        // Verify that all modes have consistent dimensions:
        // For non-sparse, non-convolve: X buffer needs bits_x * channels * rows * inner elements
        // For non-sparse, non-convolve: Y buffer needs bits_y * channels * inner * cols elements
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            // Verify basic consistency
            assert!(cfg.rows > 0, "mode {i}: rows must be > 0");
            assert!(cfg.cols > 0, "mode {i}: cols must be > 0");
            assert!(cfg.inner > 0, "mode {i}: inner must be > 0");
            assert!(cfg.channels > 0, "mode {i}: channels must be > 0");
            assert!(cfg.bits_x > 0, "mode {i}: bits_x must be > 0");
            assert!(cfg.bits_y > 0, "mode {i}: bits_y must be > 0");

            if !cfg.convolve_x && !cfg.sparse {
                // X buffer size = bits_x * channels * rows * inner bits
                let x_bits = cfg.bits_x * cfg.channels * cfg.rows * cfg.inner;
                assert!(
                    x_bits <= 512,
                    "mode {i}: X buffer too large: {x_bits} bits"
                );
            }

            if !cfg.sparse {
                // Y buffer size = bits_y * channels * inner * cols bits
                let y_bits = cfg.bits_y * cfg.channels * cfg.inner * cfg.cols;
                assert!(
                    y_bits <= 512,
                    "mode {i}: Y buffer too large: {y_bits} bits"
                );
            }
        }
    }

    #[test]
    fn test_rc2i_row_major() {
        // Simple 4x8 matrix, no channels, unit strides
        // Element (2, 3) should be at index 2*8 + 3 = 19
        let idx = rc2i(2, 3, 0, 4, 8, 1, 1, 1, 1, Ordering::RowMajor);
        assert_eq!(idx, 19);
    }

    #[test]
    fn test_rc2i_col_major() {
        // 4x8 matrix, col-major: element (2, 3) at index 3*4 + 2 = 14
        let idx = rc2i(2, 3, 0, 4, 8, 1, 1, 1, 1, Ordering::ColMajor);
        assert_eq!(idx, 14);
    }

    #[test]
    fn test_rc2i_with_channels() {
        // 4x8 matrix with 2 channels, row-major
        // Element (1, 2, channel=1): ((1*8 + 2) * 2 + 1) = 21
        let idx = rc2i(1, 2, 1, 4, 8, 2, 1, 1, 1, Ordering::RowMajor);
        assert_eq!(idx, 21);
    }

    #[test]
    fn test_rc2i_with_strides() {
        // 4x8 matrix, row_stride=2, col_stride=1, channel_stride=1
        // Element (1, 2, 0): ((2*1*8 + 2) * 1 * 1 + 0) * 1 = 18
        let idx = rc2i(1, 2, 0, 4, 8, 1, 2, 1, 1, Ordering::RowMajor);
        assert_eq!(idx, 18);
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
            let val = u32::from_le_bytes([
                result[i * 4],
                result[i * 4 + 1],
                result[i * 4 + 2],
                result[i * 4 + 3],
            ]);
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
    // Comprehensive MAC permute mode verification against aietools constants
    // ========================================================================
    //
    // These tests verify every field of every MAC permute mode against the
    // exact values from aietools constants.py __make_perm_modes() (lines
    // 268-300). The aietools tuple format is:
    //   (bits_x, bits_y, acc_cmb, channels, rows, inner, cols,
    //    complex_x, complex_y, bfloat, convolve_x, convolve_y, sparse)
    //
    // Note: convolve_y is always False for all modes.

    /// Verify a single MAC permute mode against expected aietools values.
    fn assert_mac_mode(
        mode_idx: u8,
        expected_bits_x: u32,
        expected_bits_y: u32,
        expected_acc_combine: u32,
        expected_bfloat: bool,
        expected_rows: u32,
        expected_inner: u32,
        expected_cols: u32,
        expected_channels: u32,
        expected_complex_x: bool,
        expected_complex_y: bool,
        expected_convolve_x: bool,
        expected_sparse: bool,
    ) {
        let mode = MacPermuteMode::from_mode(mode_idx)
            .unwrap_or_else(|| panic!("mode {mode_idx} should be valid"));
        let cfg = mac_permute_config(mode);

        assert_eq!(cfg.bits_x, expected_bits_x,
            "mode {mode_idx}: bits_x mismatch");
        assert_eq!(cfg.bits_y, expected_bits_y,
            "mode {mode_idx}: bits_y mismatch");
        assert_eq!(cfg.acc_combine, expected_acc_combine,
            "mode {mode_idx}: acc_combine mismatch");
        assert_eq!(cfg.bfloat, expected_bfloat,
            "mode {mode_idx}: bfloat mismatch");
        assert_eq!(cfg.rows, expected_rows,
            "mode {mode_idx}: rows mismatch");
        assert_eq!(cfg.inner, expected_inner,
            "mode {mode_idx}: inner mismatch");
        assert_eq!(cfg.cols, expected_cols,
            "mode {mode_idx}: cols mismatch");
        assert_eq!(cfg.channels, expected_channels,
            "mode {mode_idx}: channels mismatch");
        assert_eq!(cfg.complex_x, expected_complex_x,
            "mode {mode_idx}: complex_x mismatch");
        assert_eq!(cfg.complex_y, expected_complex_y,
            "mode {mode_idx}: complex_y mismatch");
        assert_eq!(cfg.convolve_x, expected_convolve_x,
            "mode {mode_idx}: convolve_x mismatch");
        assert_eq!(cfg.sparse, expected_sparse,
            "mode {mode_idx}: sparse mismatch");
    }

    // --- Matrix multiply modes (modes 0-8) ---

    #[test]
    fn test_aietools_mode_00_8x4_matmul() {
        // aietools: ( 8, 4, 1,  1,  4,16, 8,  False, False,  False, False, False, False)
        assert_mac_mode(0, 8, 4, 1, false, 4, 16, 8, 1, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_01_8x8_matmul() {
        // aietools: ( 8, 8, 1,  1,  4, 8, 8,  False, False,  False, False, False, False)
        assert_mac_mode(1, 8, 8, 1, false, 4, 8, 8, 1, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_02_16x8_matmul() {
        // aietools: (16, 8, 1,  1,  4, 4, 8,  False, False,  False, False, False, False)
        assert_mac_mode(2, 16, 8, 1, false, 4, 4, 8, 1, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_03_16x16_matmul() {
        // aietools: (16,16, 1,  1,  4, 2, 8,  False, False,  False, False, False, False)
        assert_mac_mode(3, 16, 16, 1, false, 4, 2, 8, 1, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_04_16x8_2acc_matmul() {
        // aietools: (16, 8, 2,  1,  2, 8, 8,  False, False,  False, False, False, False)
        assert_mac_mode(4, 16, 8, 2, false, 2, 8, 8, 1, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_05_16x8_2acc_matmul_alt() {
        // aietools: (16, 8, 2,  1,  4, 8, 4,  False, False,  False, False, False, False)
        assert_mac_mode(5, 16, 8, 2, false, 4, 8, 4, 1, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_06_16x16_2acc_matmul() {
        // aietools: (16,16, 2,  1,  2, 4, 8,  False, False,  False, False, False, False)
        assert_mac_mode(6, 16, 16, 2, false, 2, 4, 8, 1, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_07_16x16_2acc_matmul_alt() {
        // aietools: (16,16, 2,  1,  4, 4, 4,  False, False,  False, False, False, False)
        assert_mac_mode(7, 16, 16, 2, false, 4, 4, 4, 1, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_08_bf16_matmul() {
        // aietools: (16,16, 1,  1,  4, 8, 4,  False, False,  True , False, False, False)
        assert_mac_mode(8, 16, 16, 1, true, 4, 8, 4, 1, false, false, false, false);
    }

    // --- Element-wise modes (modes 9-12) ---

    #[test]
    fn test_aietools_mode_09_16x8_2ch_elemwise() {
        // aietools: (16, 8, 1,  2,  4, 4, 4,  False, False,  False, False, False, False)
        assert_mac_mode(9, 16, 8, 1, false, 4, 4, 4, 2, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_10_8x8_32ch_elemwise() {
        // aietools: ( 8, 8, 1, 32,  1, 2, 1,  False, False,  False, False, False, False)
        assert_mac_mode(10, 8, 8, 1, false, 1, 2, 1, 32, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_11_16x16_32ch_elemwise() {
        // aietools: (16,16, 1, 32,  1, 1, 1,  False, False,  False, False, False, False)
        assert_mac_mode(11, 16, 16, 1, false, 1, 1, 1, 32, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_12_16x16_16ch_2acc_elemwise() {
        // aietools: (16,16, 2, 16,  1, 2, 1,  False, False,  False, False, False, False)
        assert_mac_mode(12, 16, 16, 2, false, 1, 2, 1, 16, false, false, false, false);
    }

    // --- Convolution modes (modes 13-16) ---

    #[test]
    fn test_aietools_mode_13_8x8_conv_8ch() {
        // aietools: ( 8, 8, 1,  8,  4, 4, 1,  False, False,  False, True,  False, False)
        assert_mac_mode(13, 8, 8, 1, false, 4, 4, 1, 8, false, false, true, false);
    }

    #[test]
    fn test_aietools_mode_14_8x8_conv_4ch() {
        // aietools: ( 8, 8, 1,  4,  8, 8, 1,  False, False,  False, True,  False, False)
        assert_mac_mode(14, 8, 8, 1, false, 8, 8, 1, 4, false, false, true, false);
    }

    #[test]
    fn test_aietools_mode_15_8x8_conv_1ch_2d() {
        // aietools: ( 8, 8, 1,  1, 32, 8, 1,  False, False,  False, True,  False, False)
        assert_mac_mode(15, 8, 8, 1, false, 32, 8, 1, 1, false, false, true, false);
    }

    #[test]
    fn test_aietools_mode_16_16x16_conv_2acc() {
        // aietools: (16,16, 2,  1, 16, 4, 1,  False, False,  False, True,  False, False)
        assert_mac_mode(16, 16, 16, 2, false, 16, 4, 1, 1, false, false, true, false);
    }

    // --- BFloat16 element-wise (mode 17) ---

    #[test]
    fn test_aietools_mode_17_bf16_16ch_elemwise() {
        // aietools: (16,16, 1, 16,  1, 2, 1,  False, False,  True,  False, False, False)
        assert_mac_mode(17, 16, 16, 1, true, 1, 2, 1, 16, false, false, false, false);
    }

    // --- FFT modes (modes 18-20) ---

    #[test]
    fn test_aietools_mode_18_32x16_matmul() {
        // aietools: (32,16, 2,  1,  4, 2, 4,  False, False,  False, False, False, False)
        assert_mac_mode(18, 32, 16, 2, false, 4, 2, 4, 1, false, false, false, false);
    }

    #[test]
    fn test_aietools_mode_19_32x16_cplx_elemwise() {
        // aietools: (32,16, 2,  8,  1, 1, 1,  True,  True,   False, False, False, False)
        assert_mac_mode(19, 32, 16, 2, false, 1, 1, 1, 8, true, true, false, false);
    }

    #[test]
    fn test_aietools_mode_20_16x16_cplx_elemwise() {
        // aietools: (16,16, 2,  8,  1, 2, 1,  True,  True,   False, False, False, False)
        assert_mac_mode(20, 16, 16, 2, false, 1, 2, 1, 8, true, true, false, false);
    }

    // --- Sparse modes (modes 21-25) ---

    #[test]
    fn test_aietools_mode_21_8x4_sparse() {
        // aietools: ( 8, 4, 1,  1,  4,32, 8,  False, False,  False, False, False, True )
        assert_mac_mode(21, 8, 4, 1, false, 4, 32, 8, 1, false, false, false, true);
    }

    #[test]
    fn test_aietools_mode_22_8x8_sparse() {
        // aietools: ( 8, 8, 1,  1,  4,16, 8,  False, False,  False, False, False, True )
        assert_mac_mode(22, 8, 8, 1, false, 4, 16, 8, 1, false, false, false, true);
    }

    #[test]
    fn test_aietools_mode_23_16x8_sparse() {
        // aietools: (16, 8, 2,  1,  2,16, 8,  False, False,  False, False, False, True )
        assert_mac_mode(23, 16, 8, 2, false, 2, 16, 8, 1, false, false, false, true);
    }

    #[test]
    fn test_aietools_mode_24_16x16_sparse() {
        // aietools: (16,16, 2,  1,  2, 8, 8,  False, False,  False, False, False, True )
        assert_mac_mode(24, 16, 16, 2, false, 2, 8, 8, 1, false, false, false, true);
    }

    #[test]
    fn test_aietools_mode_25_bf16_sparse() {
        // aietools: (16,16, 1,  1,  4,16, 4,  False, False,  True , False, False, True )
        assert_mac_mode(25, 16, 16, 1, true, 4, 16, 4, 1, false, false, false, true);
    }

    // ========================================================================
    // Cross-cutting MAC permute mode property tests
    // ========================================================================

    /// Verify that the total number of MAC permute modes is exactly 26,
    /// matching aietools constants.py.
    #[test]
    fn test_mac_mode_count_matches_aietools() {
        let mut count = 0u8;
        while MacPermuteMode::from_mode(count).is_some() {
            count += 1;
        }
        assert_eq!(count, 26,
            "Expected exactly 26 MAC permute modes (aietools constants.py generates 26)");
    }

    /// Verify that all modes produce the expected number of output lanes.
    ///
    /// Hardware constraint: the AIE2 has acc_num accumulators (32 for
    /// integer, 16 for bfloat). The number of output lanes is
    /// acc_num / acc_combine. Output elements (rows * cols * channels)
    /// must equal this lane count.
    ///
    /// Per aietools:
    ///   - acc_num = 32 for non-bfloat modes, 16 for bfloat
    ///   - lanes_num = acc_num / acc_combine
    #[test]
    fn test_output_lanes_constraint() {
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            let output_elements = cfg.rows * cfg.cols * cfg.channels;
            if cfg.complex_x || cfg.complex_y {
                // Complex modes double the output due to real+imag parts
                continue;
            }

            let acc_num: u32 = if cfg.bfloat { 16 } else { 32 };
            let expected_lanes = acc_num / cfg.acc_combine;

            assert_eq!(output_elements, expected_lanes,
                "mode {i}: output elements ({output_elements}) != expected lanes ({expected_lanes}) \
                 [acc_num={acc_num}, acc_combine={}]", cfg.acc_combine);
        }
    }

    /// Verify that bfloat modes always use 16x16 element sizes.
    ///
    /// BFloat16 is encoded as 16-bit in the MAC pipeline.
    #[test]
    fn test_bfloat_modes_are_16x16() {
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            if cfg.bfloat {
                assert_eq!(cfg.bits_x, 16,
                    "mode {i}: bfloat mode should have bits_x=16");
                assert_eq!(cfg.bits_y, 16,
                    "mode {i}: bfloat mode should have bits_y=16");
            }
        }
    }

    /// Verify that sparse modes double the inner dimension compared to
    /// their non-sparse counterparts.
    ///
    /// Sparse modes use 50% sparsity, so the X buffer is twice as wide
    /// to hold the same number of non-zero elements.
    #[test]
    fn test_sparse_modes_double_inner() {
        // Sparse mode 21 (8x4) vs mode 0 (8x4): inner 32 vs 16
        let dense = mac_permute_config(MacPermuteMode::Mode_8x4_4x16_16x8);
        let sparse = mac_permute_config(MacPermuteMode::Mode_8x4_Sparse);
        assert_eq!(sparse.inner, dense.inner * 2,
            "sparse 8x4: inner should be 2x dense");
        assert_eq!(sparse.rows, dense.rows);
        assert_eq!(sparse.cols, dense.cols);

        // Sparse mode 22 (8x8) vs mode 1 (8x8): inner 16 vs 8
        let dense = mac_permute_config(MacPermuteMode::Mode_8x8_4x8_8x8);
        let sparse = mac_permute_config(MacPermuteMode::Mode_8x8_Sparse);
        assert_eq!(sparse.inner, dense.inner * 2,
            "sparse 8x8: inner should be 2x dense");

        // Sparse mode 23 (16x8) vs mode 4 (16x8): inner 16 vs 8
        let dense = mac_permute_config(MacPermuteMode::Mode_16x8_2x8_8x8);
        let sparse = mac_permute_config(MacPermuteMode::Mode_16x8_Sparse);
        assert_eq!(sparse.inner, dense.inner * 2,
            "sparse 16x8: inner should be 2x dense");

        // Sparse mode 24 (16x16) vs mode 6 (16x16): inner 8 vs 4
        let dense = mac_permute_config(MacPermuteMode::Mode_16x16_2x4_4x8);
        let sparse = mac_permute_config(MacPermuteMode::Mode_16x16_Sparse);
        assert_eq!(sparse.inner, dense.inner * 2,
            "sparse 16x16: inner should be 2x dense");

        // Sparse mode 25 (bf16) vs mode 8 (bf16): inner 16 vs 8
        let dense = mac_permute_config(MacPermuteMode::Mode_BF16_4x8_8x4);
        let sparse = mac_permute_config(MacPermuteMode::Mode_BF16_Sparse);
        assert_eq!(sparse.inner, dense.inner * 2,
            "sparse bf16: inner should be 2x dense");
    }

    /// Verify that all sparse modes have channels=1 (no multi-channel
    /// sparse modes exist in hardware).
    #[test]
    fn test_sparse_modes_single_channel() {
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            if cfg.sparse {
                assert_eq!(cfg.channels, 1,
                    "mode {i}: sparse modes must have channels=1");
            }
        }
    }

    /// Verify that convolution modes always have cols=1 (1D output in
    /// the column dimension).
    #[test]
    fn test_convolution_modes_cols_1() {
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            if cfg.convolve_x {
                assert_eq!(cfg.cols, 1,
                    "mode {i}: convolution modes must have cols=1");
            }
        }
    }

    /// Verify that complex modes always have complex_x AND complex_y set
    /// (the hardware does not support mixed real/complex operands).
    #[test]
    fn test_complex_modes_both_xy() {
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            if cfg.complex_x || cfg.complex_y {
                assert!(cfg.complex_x && cfg.complex_y,
                    "mode {i}: complex modes must have both complex_x and complex_y");
            }
        }
    }

    /// Verify that each lane gets the correct number of multiplier operations.
    ///
    /// The AIE2 has 512 multipliers feeding acc_num accumulators (32 for
    /// integer, 16 for bfloat). With acc_combine, adjacent accumulators
    /// merge, yielding `output_lanes = acc_num / acc_combine` outputs.
    ///
    /// Each output lane gets `mult_per_lane = (mult_num / acc_num) * acc_combine`
    /// multiplier operations. These decompose into:
    ///   - `mul_per_op = (bx / 8) * (by / 4)` multipliers per element pair
    ///   - `dup = mult_per_lane / (mul_per_op * complex_factor)` post-addition depth
    ///   - `inner * mul_per_op` actual unique multiply-index combinations
    ///   - `dup` repeats of the inner loop for post-addition accumulation
    ///
    /// The useful constraint: `inner * mul_per_op * dup * complex_factor == mult_per_lane`
    /// and `mult_per_lane * output_lanes == 512`.
    #[test]
    fn test_multiplier_lane_constraint() {
        const MULT_NUM: u32 = 512;
        const MULT_GRAN_X: u32 = 8;
        const MULT_GRAN_Y: u32 = 4;

        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            if cfg.sparse || cfg.convolve_x || cfg.bfloat {
                continue;
            }

            let acc_num: u32 = 32;
            let output_lanes = acc_num / cfg.acc_combine;
            let mult_per_lane = (MULT_NUM / acc_num) * cfg.acc_combine;

            // Verify the fundamental identity
            assert_eq!(output_lanes * mult_per_lane, MULT_NUM,
                "mode {i}: output_lanes * mult_per_lane != 512");

            // Verify output_lanes matches mode dimensions
            let complex_factor = if cfg.complex_x || cfg.complex_y { 2 } else { 1 };
            let computed_lanes = cfg.rows * cfg.cols * cfg.channels;
            // Complex modes produce complex outputs (real+imag per lane)
            if complex_factor == 1 {
                assert_eq!(computed_lanes, output_lanes,
                    "mode {i}: dimension product doesn't match output_lanes");
            }

            // Verify mul_per_op makes sense
            let bx = cfg.bits_x;
            let by = cfg.bits_y;
            let mul_per_op = (bx / MULT_GRAN_X) * (by / MULT_GRAN_Y);
            assert!(mul_per_op > 0,
                "mode {i}: mul_per_op should be > 0");

            // Verify dup (post-addition depth) is a whole number
            let inner_muls = cfg.inner * mul_per_op * complex_factor;
            assert!(mult_per_lane % inner_muls == 0,
                "mode {i}: mult_per_lane ({mult_per_lane}) not divisible by \
                 inner_muls ({inner_muls})");

            let dup = mult_per_lane / inner_muls;
            assert!(dup >= 1,
                "mode {i}: dup={dup} should be >= 1");
        }
    }

    /// Verify the X-buffer fits within the 512-bit permutation width for
    /// all non-sparse, non-convolve modes.
    ///
    /// aietools uses perm_width_x = 512 bits. For sparse modes, the
    /// effective width doubles to 1024. For convolution modes, the
    /// addressing is sliding-window and the formula is different.
    #[test]
    fn test_x_buffer_fits_perm_width() {
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            if cfg.convolve_x {
                // Convolution X size: (rows-1)*row_stride + 1 + inner*inner_stride - 1
                // With all strides = 1: rows + inner - 1
                let x_elements = cfg.rows + cfg.inner - 1;
                let x_bits = cfg.bits_x * cfg.channels * x_elements;
                assert!(x_bits <= 512,
                    "mode {i} (conv): X buffer {x_bits} bits exceeds 512");
            } else if cfg.sparse {
                // Sparse modes: X permutation width doubles to 1024
                let x_bits = cfg.bits_x * cfg.channels * cfg.rows * cfg.inner;
                assert!(x_bits <= 1024,
                    "mode {i} (sparse): X buffer {x_bits} bits exceeds 1024");
            } else {
                let x_bits = cfg.bits_x * cfg.channels * cfg.rows * cfg.inner;
                assert!(x_bits <= 512,
                    "mode {i}: X buffer {x_bits} bits exceeds 512");
            }
        }
    }

    /// Verify the Y-buffer fits within the 512-bit permutation width.
    ///
    /// For sparse modes, Y buffer size is halved (sparsity mask selects
    /// which elements to use).
    #[test]
    fn test_y_buffer_fits_perm_width() {
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            let y_bits = cfg.bits_y * cfg.channels * cfg.inner * cfg.cols;
            if cfg.sparse {
                assert!(y_bits / 2 <= 512,
                    "mode {i} (sparse): Y buffer {y_bits}/2 bits exceeds 512");
            } else {
                assert!(y_bits <= 512,
                    "mode {i}: Y buffer {y_bits} bits exceeds 512");
            }
        }
    }

    /// Verify that bits_y is always 4, 8, or 16 and bits_x is always
    /// 8, 16, or 32 (hardware granularity constraint).
    #[test]
    fn test_element_width_constraints() {
        for i in 0..=25u8 {
            let mode = MacPermuteMode::from_mode(i).unwrap();
            let cfg = mac_permute_config(mode);

            assert!(
                cfg.bits_x == 8 || cfg.bits_x == 16 || cfg.bits_x == 32,
                "mode {i}: bits_x={} not in {{8, 16, 32}}",
                cfg.bits_x
            );
            assert!(
                cfg.bits_y == 4 || cfg.bits_y == 8 || cfg.bits_y == 16,
                "mode {i}: bits_y={} not in {{4, 8, 16}}",
                cfg.bits_y
            );
            assert!(cfg.bits_x >= cfg.bits_y,
                "mode {i}: bits_x={} < bits_y={} violates hardware constraint",
                cfg.bits_x, cfg.bits_y);
        }
    }

    /// Verify the mult_mode index that aietools would assign to each
    /// permute mode.
    ///
    /// The mult_modes table (from constants.py lines 213-222) maps
    /// (bits_x, bits_y, acc_cmb, bfloat) to a mult_mode index.
    #[test]
    fn test_mult_mode_assignment() {
        // aietools mult_modes table:
        //   0: MultMode(8,  4, 1, false, 32)
        //   1: MultMode(8,  8, 1, false, 32)
        //   2: MultMode(16, 8, 1, false, 32)
        //   3: MultMode(16, 16, 1, false, 32)
        //   4: MultMode(16, 8, 2, false, 32)
        //   5: MultMode(16, 16, 2, false, 32)
        //   6: MultMode(16, 16, 1, true, 16)
        //   7: MultMode(32, 16, 2, false, 32)

        let expected_mmodes: [(u32, u32, u32, bool); 26] = [
            // mode 0-8: matrix multiply
            ( 8,  4, 1, false), // mmode 0
            ( 8,  8, 1, false), // mmode 1
            (16,  8, 1, false), // mmode 2
            (16, 16, 1, false), // mmode 3
            (16,  8, 2, false), // mmode 4
            (16,  8, 2, false), // mmode 4
            (16, 16, 2, false), // mmode 5
            (16, 16, 2, false), // mmode 5
            (16, 16, 1, true),  // mmode 6
            // mode 9-12: element-wise
            (16,  8, 1, false), // mmode 2
            ( 8,  8, 1, false), // mmode 1
            (16, 16, 1, false), // mmode 3
            (16, 16, 2, false), // mmode 5
            // mode 13-16: convolution
            ( 8,  8, 1, false), // mmode 1
            ( 8,  8, 1, false), // mmode 1
            ( 8,  8, 1, false), // mmode 1
            (16, 16, 2, false), // mmode 5
            // mode 17: bf16 element-wise
            (16, 16, 1, true),  // mmode 6
            // mode 18-20: FFT
            (32, 16, 2, false), // mmode 7
            (32, 16, 2, false), // mmode 7
            (16, 16, 2, false), // mmode 5
            // mode 21-25: sparse
            ( 8,  4, 1, false), // mmode 0
            ( 8,  8, 1, false), // mmode 1
            (16,  8, 2, false), // mmode 4
            (16, 16, 2, false), // mmode 5
            (16, 16, 1, true),  // mmode 6
        ];

        for (i, &(bx, by, ac, bf)) in expected_mmodes.iter().enumerate() {
            let mode = MacPermuteMode::from_mode(i as u8).unwrap();
            let cfg = mac_permute_config(mode);

            assert_eq!(cfg.bits_x, bx,
                "mode {i}: bits_x mismatch for mult_mode lookup");
            assert_eq!(cfg.bits_y, by,
                "mode {i}: bits_y mismatch for mult_mode lookup");
            assert_eq!(cfg.acc_combine, ac,
                "mode {i}: acc_combine mismatch for mult_mode lookup");
            assert_eq!(cfg.bfloat, bf,
                "mode {i}: bfloat mismatch for mult_mode lookup");
        }
    }

    /// Verify the rc2i function with identity (all ones) strides produces
    /// simple row-major and col-major linear indices.
    #[test]
    fn test_rc2i_identity_strides() {
        // 4x4 matrix with 1 channel, all unit strides, row-major
        for r in 0..4 {
            for c in 0..4 {
                let idx = rc2i(r, c, 0, 4, 4, 1, 1, 1, 1, Ordering::RowMajor);
                assert_eq!(idx, r * 4 + c,
                    "rc2i row-major ({r},{c}): expected {}, got {idx}", r * 4 + c);
            }
        }

        // Same but col-major
        for r in 0..4 {
            for c in 0..4 {
                let idx = rc2i(r, c, 0, 4, 4, 1, 1, 1, 1, Ordering::ColMajor);
                assert_eq!(idx, c * 4 + r,
                    "rc2i col-major ({r},{c}): expected {}, got {idx}", c * 4 + r);
            }
        }
    }

    /// Verify rc2i with multi-channel stride patterns.
    #[test]
    fn test_rc2i_multi_channel() {
        // 2x2 matrix with 4 channels, row-major
        // Element (row=1, col=0, ch=2): ((1*2 + 0) * 4 + 2) * 1 = 10
        let idx = rc2i(1, 0, 2, 2, 2, 4, 1, 1, 1, Ordering::RowMajor);
        assert_eq!(idx, 10);

        // Element (row=0, col=1, ch=3): ((0*2 + 1) * 4 + 3) * 1 = 7
        let idx = rc2i(0, 1, 3, 2, 2, 4, 1, 1, 1, Ordering::RowMajor);
        assert_eq!(idx, 7);
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
            assert_eq!(result.len(), VEC_BYTES,
                "mode {i}: result length mismatch");
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
            assert_eq!(result, zero,
                "mode {i}: zero input should produce zero output");
        }
    }

    /// Verify that shuffle mode 26 (T16_4x16Lo) with (broadcast, zeros) input
    /// matches hardware VBCSTSHFL output for the r29!=0 case.
    ///
    /// Per aietools ISG (me_inline_primitives.h lines 2795-2831):
    /// VBCSTSHFL = s2v_interleave_sw(broadcast, ZEROS, mode=r29)
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
        for i in 0..8 { bcast[i*8..i*8+8].copy_from_slice(&comp); }

        // HW output: [A A 0 0 B B 0 0 C C 0 0 D D 0 0] * 2
        let mut hw = [0u8; VEC_BYTES];
        let comps: [[u8; 2]; 4] = [[0xd6, 0xa1], [0x99, 0xea], [0x5a, 0xc2], [0xba, 0x45]];
        for half in 0..2 {
            let base = half * 32;
            for (c, comp) in comps.iter().enumerate() {
                hw[base + c*8] = comp[0]; hw[base + c*8+1] = comp[1];
                hw[base + c*8+2] = comp[0]; hw[base + c*8+3] = comp[1];
            }
        }

        let result = shuffle_vectors(&bcast, &zero, ShuffleMode::T16_4x16Lo);
        assert_eq!(result, hw,
            "shuffle(broadcast.64, zeros, T16_4x16Lo) must match NPU1 VBCSTSHFL_64 output");
    }
}

