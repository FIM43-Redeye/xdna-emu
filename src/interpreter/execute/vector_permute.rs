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
//!
//! TODO: MAC permutation modes are not yet implemented. They will be needed
//! when the emulator implements cycle-accurate MAC pipeline behavior.

/// Number of bytes in a 512-bit vector.
const VEC_BYTES: usize = 64;

/// Number of bytes in a 1024-bit vector pair (shuffle unit input).
const PAIR_BYTES: usize = 128;

// ============================================================================
// Shuffle modes (from me_enums.h)
// ============================================================================

/// All 48 hardware shuffle modes.
///
/// Each mode transposes a matrix of elements within a 1024-bit vector pair,
/// then extracts either the low or high 512-bit half.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShuffleMode {
    // Deinterleave modes (NxM -> MxN, extract half)
    T8_64x2Lo    =  0,
    T8_64x2Hi    =  1,
    T16_32x2Lo   =  2,
    T16_32x2Hi   =  3,
    T32_16x2Lo   =  4,
    T32_16x2Hi   =  5,
    T64_8x2Lo    =  6,
    T64_8x2Hi    =  7,
    T128_4x2Lo   =  8,
    T128_4x2Hi   =  9,
    T256_2x2Lo   = 10,
    T256_2x2Hi   = 11,
    // Interleave modes (2xN -> Nx2, extract half)
    T128_2x4Lo   = 12,
    T128_2x4Hi   = 13,
    T64_2x8Lo    = 14,
    T64_2x8Hi    = 15,
    T32_2x16Lo   = 16,
    T32_2x16Hi   = 17,
    T16_2x32Lo   = 18,
    T16_2x32Hi   = 19,
    T8_2x64Lo    = 20,
    T8_2x64Hi    = 21,
    // Bypass modes (512-bit halves)
    T512_1x2Lo   = 22,
    T512_1x2Hi   = 23,
    // NL (non-linear) modes
    T16_16x4Lo   = 24,
    T16_16x4Hi   = 25,
    T16_4x16Lo   = 26,
    T16_4x16Hi   = 27,
    // FFT and sparsity modes
    T16_8x4      = 28,
    T16_4x8      = 29,
    T32_8x4Lo    = 30,
    T32_8x4Hi    = 31,
    T32_4x8Lo    = 32,
    T32_4x8Hi    = 33,
    T32_4x4      = 34,
    T8_8x8       = 35,
    T8_16x4      = 36,
    T8_4x16      = 37,
    T16_1x2Flip  = 38,
    // Permute reduction modes
    T16_4x4      = 39,
    T16_4x2      = 40,
    T16_2x4      = 41,
    T16_8x2      = 42,
    T16_2x8      = 43,
    T16_16x2     = 44,
    T16_2x16     = 45,
    T8_8x4       = 46,
    T8_4x8       = 47,
}

impl ShuffleMode {
    /// Try to convert from a raw mode number.
    pub fn from_mode(mode: u8) -> Option<Self> {
        if mode > 47 {
            return None;
        }
        // Safety: all values 0..=47 are valid enum variants with matching repr.
        Some(unsafe { std::mem::transmute(mode) })
    }

    /// Return the raw mode number.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

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
    // Build the 128-byte concatenated input.
    let mut input = [0u8; PAIR_BYTES];
    input[..VEC_BYTES].copy_from_slice(lo);
    input[VEC_BYTES..].copy_from_slice(hi);

    match mode {
        // ---- Bypass ----
        ShuffleMode::T512_1x2Lo => {
            let mut out = [0u8; VEC_BYTES];
            out.copy_from_slice(lo);
            out
        }
        ShuffleMode::T512_1x2Hi => {
            let mut out = [0u8; VEC_BYTES];
            out.copy_from_slice(hi);
            out
        }

        // ---- 2x2 transpose of 256-bit elements ----
        ShuffleMode::T256_2x2Lo => transpose_extract(&input, 32, 2, 2, false),
        ShuffleMode::T256_2x2Hi => transpose_extract(&input, 32, 2, 2, true),

        // ---- Deinterleave: NxM -> MxN, extract half ----
        // T{bits}_{N}x{M}: treat 1024 bits as N rows of M elements of {bits} size
        // Transpose to MxN, extract lo or hi 512 bits
        ShuffleMode::T8_64x2Lo  => transpose_extract(&input, 1, 64, 2, false),
        ShuffleMode::T8_64x2Hi  => transpose_extract(&input, 1, 64, 2, true),
        ShuffleMode::T16_32x2Lo => transpose_extract(&input, 2, 32, 2, false),
        ShuffleMode::T16_32x2Hi => transpose_extract(&input, 2, 32, 2, true),
        ShuffleMode::T32_16x2Lo => transpose_extract(&input, 4, 16, 2, false),
        ShuffleMode::T32_16x2Hi => transpose_extract(&input, 4, 16, 2, true),
        ShuffleMode::T64_8x2Lo  => transpose_extract(&input, 8,  8, 2, false),
        ShuffleMode::T64_8x2Hi  => transpose_extract(&input, 8,  8, 2, true),
        ShuffleMode::T128_4x2Lo => transpose_extract(&input, 16, 4, 2, false),
        ShuffleMode::T128_4x2Hi => transpose_extract(&input, 16, 4, 2, true),

        // ---- Interleave: 2xN -> Nx2, extract half ----
        ShuffleMode::T128_2x4Lo => transpose_extract(&input, 16, 2, 4, false),
        ShuffleMode::T128_2x4Hi => transpose_extract(&input, 16, 2, 4, true),
        ShuffleMode::T64_2x8Lo  => transpose_extract(&input, 8,  2, 8, false),
        ShuffleMode::T64_2x8Hi  => transpose_extract(&input, 8,  2, 8, true),
        ShuffleMode::T32_2x16Lo => transpose_extract(&input, 4,  2, 16, false),
        ShuffleMode::T32_2x16Hi => transpose_extract(&input, 4,  2, 16, true),
        ShuffleMode::T16_2x32Lo => transpose_extract(&input, 2,  2, 32, false),
        ShuffleMode::T16_2x32Hi => transpose_extract(&input, 2,  2, 32, true),
        ShuffleMode::T8_2x64Lo  => transpose_extract(&input, 1,  2, 64, false),
        ShuffleMode::T8_2x64Hi  => transpose_extract(&input, 1,  2, 64, true),

        // ---- NL modes: 16-bit 16x4 / 4x16 ----
        ShuffleMode::T16_16x4Lo => transpose_extract(&input, 2, 16, 4, false),
        ShuffleMode::T16_16x4Hi => transpose_extract(&input, 2, 16, 4, true),
        ShuffleMode::T16_4x16Lo => transpose_extract(&input, 2, 4, 16, false),
        ShuffleMode::T16_4x16Hi => transpose_extract(&input, 2, 4, 16, true),

        // ---- FFT/sparsity modes: exact 512-bit result ----
        ShuffleMode::T16_8x4 => transpose_exact(&input, 2, 8, 4),
        ShuffleMode::T16_4x8 => transpose_exact(&input, 2, 4, 8),

        ShuffleMode::T32_8x4Lo => transpose_extract(&input, 4, 8, 4, false),
        ShuffleMode::T32_8x4Hi => transpose_extract(&input, 4, 8, 4, true),
        ShuffleMode::T32_4x8Lo => transpose_extract(&input, 4, 4, 8, false),
        ShuffleMode::T32_4x8Hi => transpose_extract(&input, 4, 4, 8, true),

        ShuffleMode::T32_4x4 => transpose_exact(&input, 4, 4, 4),

        ShuffleMode::T8_8x8  => transpose_exact(&input, 1, 8, 8),
        ShuffleMode::T8_16x4 => transpose_exact(&input, 1, 16, 4),
        ShuffleMode::T8_4x16 => transpose_exact(&input, 1, 4, 16),

        // ---- Flip mode: swap pairs of 16-bit values ----
        ShuffleMode::T16_1x2Flip => {
            let mut out = [0u8; VEC_BYTES];
            // Swap adjacent 16-bit elements: [A, B, C, D, ...] -> [B, A, D, C, ...]
            for i in 0..(VEC_BYTES / 4) {
                let base = i * 4;
                // Swap two 16-bit values (2 bytes each)
                out[base]     = input[base + 2];
                out[base + 1] = input[base + 3];
                out[base + 2] = input[base];
                out[base + 3] = input[base + 1];
            }
            out
        }

        // ---- Permute reduction modes ----
        ShuffleMode::T16_4x4  => transpose_exact(&input, 2, 4, 4),
        ShuffleMode::T16_4x2  => transpose_exact(&input, 2, 4, 2),
        ShuffleMode::T16_2x4  => transpose_exact(&input, 2, 2, 4),
        ShuffleMode::T16_8x2  => transpose_exact(&input, 2, 8, 2),
        ShuffleMode::T16_2x8  => transpose_exact(&input, 2, 2, 8),
        ShuffleMode::T16_16x2 => transpose_exact(&input, 2, 16, 2),
        ShuffleMode::T16_2x16 => transpose_exact(&input, 2, 2, 16),
        ShuffleMode::T8_8x4   => transpose_exact(&input, 1, 8, 4),
        ShuffleMode::T8_4x8   => transpose_exact(&input, 1, 4, 8),
    }
}

/// Generic matrix transpose within the 128-byte input buffer.
///
/// Interprets the input as a `rows x cols` matrix of elements, each
/// `elem_bytes` wide. Transposes to `cols x rows`, then returns either
/// the low or high 64-byte half.
fn transpose_extract(
    input: &[u8; PAIR_BYTES],
    elem_bytes: usize,
    rows: usize,
    cols: usize,
    high: bool,
) -> [u8; VEC_BYTES] {
    debug_assert_eq!(rows * cols * elem_bytes, PAIR_BYTES);

    let mut transposed = [0u8; PAIR_BYTES];

    // Transpose: element at (r, c) moves to (c, r).
    for r in 0..rows {
        for c in 0..cols {
            let src_off = (r * cols + c) * elem_bytes;
            let dst_off = (c * rows + r) * elem_bytes;
            transposed[dst_off..dst_off + elem_bytes]
                .copy_from_slice(&input[src_off..src_off + elem_bytes]);
        }
    }

    let mut out = [0u8; VEC_BYTES];
    let start = if high { VEC_BYTES } else { 0 };
    out.copy_from_slice(&transposed[start..start + VEC_BYTES]);
    out
}

/// Transpose that produces exactly 512 bits of result (no lo/hi split).
///
/// The matrix (`rows x cols` of `elem_bytes`-wide elements) may fit in
/// exactly 512 bits or in a smaller region. When smaller than 512 bits,
/// the hardware tiles the transpose across multiple independent blocks
/// within the 512-bit vector.
///
/// For example, T16_4x4 (4x4 of 16-bit = 32 bytes) transposes two
/// independent 4x4 blocks to fill the 64-byte output.
fn transpose_exact(
    input: &[u8; PAIR_BYTES],
    elem_bytes: usize,
    rows: usize,
    cols: usize,
) -> [u8; VEC_BYTES] {
    let block_bytes = rows * cols * elem_bytes;
    debug_assert!(block_bytes > 0);
    debug_assert!(block_bytes <= VEC_BYTES);

    let mut out = [0u8; VEC_BYTES];
    let num_blocks = VEC_BYTES / block_bytes;

    for blk in 0..num_blocks {
        let blk_off = blk * block_bytes;
        for r in 0..rows {
            for c in 0..cols {
                let src_off = blk_off + (r * cols + c) * elem_bytes;
                let dst_off = blk_off + (c * rows + r) * elem_bytes;
                out[dst_off..dst_off + elem_bytes]
                    .copy_from_slice(&input[src_off..src_off + elem_bytes]);
            }
        }
    }
    out
}

// ============================================================================
// MAC permutation modes (PMODE_*)
// ============================================================================

/// MAC permutation mode index.
///
/// These modes control how input data from the X and Y buffers is routed
/// to the 512 multiplier array during matrix multiply-accumulate operations.
///
/// Each mode defines a specific data layout: element types (bits_x, bits_y),
/// matrix dimensions (rows, cols, inner), number of channels, and whether
/// the operation involves convolution, complex arithmetic, or sparse data.
///
/// The mode index is passed as part of the MAC instruction encoding.
///
/// TODO: Full MAC permutation implementation is deferred until the emulator
/// implements cycle-accurate MAC pipeline behavior. The mode definitions
/// below document the hardware's capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum MacPermuteMode {
    // Matrix multiply modes
    Mode_8x4_4x16_16x8            =  0,  // 8b x 4b,  1 ch, 4x16x8
    Mode_8x8_4x8_8x8              =  1,  // 8b x 8b,  1 ch, 4x8x8
    Mode_16x8_4x4_4x8             =  2,  // 16b x 8b, 1 ch, 4x4x8
    Mode_16x16_4x2_2x8            =  3,  // 16b x16b, 1 ch, 4x2x8
    Mode_16x8_2x8_8x8             =  4,  // 16b x 8b, 2acc, 2x8x8
    Mode_16x8_4x8_8x4             =  5,  // 16b x 8b, 2acc, 4x8x4
    Mode_16x16_2x4_4x8            =  6,  // 16b x16b, 2acc, 2x4x8
    Mode_16x16_4x4_4x4            =  7,  // 16b x16b, 2acc, 4x4x4
    Mode_BF16_4x8_8x4             =  8,  // bf16xbf16,1 ch, 4x8x4
    // Element-wise modes
    Mode_16x8_4x4_2ch             =  9,  // 16b x 8b, 2 ch, 4x4x4
    Mode_8x8_Elem2                = 10,  // 8b x 8b, 32 ch, 1x2x1
    Mode_16x16_Elem               = 11,  // 16bx16b, 32 ch, 1x1x1
    Mode_16x16_Elem2              = 12,  // 16bx16b, 16 ch, 1x2x1 (2acc)
    // Convolution modes
    Mode_8x8_Conv_4x4_8ch         = 13,  // 8b x 8b,  8 ch, 4x4x1 conv
    Mode_8x8_Conv_8x8_4ch         = 14,  // 8b x 8b,  4 ch, 8x8x1 conv
    Mode_8x8_Conv_32x8            = 15,  // 8b x 8b,  1 ch, 32x8x1 conv
    Mode_16x16_Conv_16x4          = 16,  // 16bx16b,  1 ch, 16x4x1 conv (2acc)
    // BFloat16 element-wise
    Mode_BF16_Elem2               = 17,  // bf16, 16 ch, 1x2x1
    // FFT modes
    Mode_32x16_4x2_2x4            = 18,  // 32bx16b, 1 ch, 4x2x4 (2acc)
    Mode_32x16_Elem_Cplx          = 19,  // 32bx16b, 8 ch, cplx elem
    Mode_16x16_Elem2_Cplx         = 20,  // 16bx16b, 8 ch, cplx elem (2acc)
    // Sparse modes
    Mode_8x4_Sparse               = 21,  // sparse variant of mode 0
    Mode_8x8_Sparse               = 22,  // sparse variant of mode 1
    Mode_16x8_Sparse              = 23,  // sparse variant of mode 4
    Mode_16x16_Sparse             = 24,  // sparse variant of mode 6
    Mode_BF16_Sparse              = 25,  // sparse variant of mode 8
}

impl MacPermuteMode {
    /// Try to convert from a raw mode number.
    pub fn from_mode(mode: u8) -> Option<Self> {
        if mode > 25 {
            return None;
        }
        // Safety: all values 0..=25 are valid enum variants with matching repr.
        Some(unsafe { std::mem::transmute(mode) })
    }

    /// Return the raw mode number.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// MAC permute mode configuration.
///
/// Captures the multi-dimensional data routing parameters for a given MAC
/// permute mode. These parameters define how elements from the X and Y
/// input buffers are mapped to multiplier lanes.
///
/// Derived from hardware behavior. The index function is:
///   `index = ((stride_outer * outer * inner + inner_pos) * stride_inner * channels + channel) * stride_channel`
/// where the ordering (row-major vs col-major) swaps outer/inner roles.
#[derive(Debug, Clone)]
pub struct MacPermuteConfig {
    /// MAC permute mode index.
    pub mode: MacPermuteMode,
    /// X operand element width in bits.
    pub bits_x: u32,
    /// Y operand element width in bits.
    pub bits_y: u32,
    /// Accumulator combine factor (1 = 32-bit accumulators, 2 = 64-bit).
    pub acc_combine: u32,
    /// Whether this is a bfloat16 mode.
    pub bfloat: bool,
    /// Number of output rows.
    pub rows: u32,
    /// Number of output columns.
    pub cols: u32,
    /// Inner (reduction) dimension.
    pub inner: u32,
    /// Number of independent channels.
    pub channels: u32,
    /// Whether X input uses convolution addressing.
    pub convolve_x: bool,
    /// Whether complex arithmetic is used on X.
    pub complex_x: bool,
    /// Whether complex arithmetic is used on Y.
    pub complex_y: bool,
    /// Whether this is a sparse mode.
    pub sparse: bool,
}

/// Look up the configuration for a MAC permute mode.
///
/// Returns the multi-dimensional parameters that define how data is routed
/// to the multiplier array for the given mode.
pub fn mac_permute_config(mode: MacPermuteMode) -> MacPermuteConfig {
    use MacPermuteMode::*;

    // (bits_x, bits_y, acc_cmb, bfloat, rows, inner, cols, channels, complex_x, complex_y, convolve_x, sparse)
    let (bx, by, ac, bf, r, i, c, ch, cx, cy, cv, sp) = match mode {
        Mode_8x4_4x16_16x8            => ( 8,  4, 1, false,  4, 16,  8,  1, false, false, false, false),
        Mode_8x8_4x8_8x8              => ( 8,  8, 1, false,  4,  8,  8,  1, false, false, false, false),
        Mode_16x8_4x4_4x8             => (16,  8, 1, false,  4,  4,  8,  1, false, false, false, false),
        Mode_16x16_4x2_2x8            => (16, 16, 1, false,  4,  2,  8,  1, false, false, false, false),
        Mode_16x8_2x8_8x8             => (16,  8, 2, false,  2,  8,  8,  1, false, false, false, false),
        Mode_16x8_4x8_8x4             => (16,  8, 2, false,  4,  8,  4,  1, false, false, false, false),
        Mode_16x16_2x4_4x8            => (16, 16, 2, false,  2,  4,  8,  1, false, false, false, false),
        Mode_16x16_4x4_4x4            => (16, 16, 2, false,  4,  4,  4,  1, false, false, false, false),
        Mode_BF16_4x8_8x4             => (16, 16, 1, true,   4,  8,  4,  1, false, false, false, false),
        Mode_16x8_4x4_2ch             => (16,  8, 1, false,  4,  4,  4,  2, false, false, false, false),
        Mode_8x8_Elem2                => ( 8,  8, 1, false,  1,  2,  1, 32, false, false, false, false),
        Mode_16x16_Elem               => (16, 16, 1, false,  1,  1,  1, 32, false, false, false, false),
        Mode_16x16_Elem2              => (16, 16, 2, false,  1,  2,  1, 16, false, false, false, false),
        Mode_8x8_Conv_4x4_8ch         => ( 8,  8, 1, false,  4,  4,  1,  8, false, false, true,  false),
        Mode_8x8_Conv_8x8_4ch         => ( 8,  8, 1, false,  8,  8,  1,  4, false, false, true,  false),
        Mode_8x8_Conv_32x8            => ( 8,  8, 1, false, 32,  8,  1,  1, false, false, true,  false),
        Mode_16x16_Conv_16x4          => (16, 16, 2, false, 16,  4,  1,  1, false, false, true,  false),
        Mode_BF16_Elem2               => (16, 16, 1, true,   1,  2,  1, 16, false, false, false, false),
        Mode_32x16_4x2_2x4            => (32, 16, 2, false,  4,  2,  4,  1, false, false, false, false),
        Mode_32x16_Elem_Cplx          => (32, 16, 2, false,  1,  1,  1,  8, true,  true,  false, false),
        Mode_16x16_Elem2_Cplx         => (16, 16, 2, false,  1,  2,  1,  8, true,  true,  false, false),
        Mode_8x4_Sparse               => ( 8,  4, 1, false,  4, 32,  8,  1, false, false, false, true),
        Mode_8x8_Sparse               => ( 8,  8, 1, false,  4, 16,  8,  1, false, false, false, true),
        Mode_16x8_Sparse              => (16,  8, 2, false,  2, 16,  8,  1, false, false, false, true),
        Mode_16x16_Sparse             => (16, 16, 2, false,  2,  8,  8,  1, false, false, false, true),
        Mode_BF16_Sparse              => (16, 16, 1, true,   4, 16,  4,  1, false, false, false, true),
    };

    MacPermuteConfig {
        mode,
        bits_x: bx,
        bits_y: by,
        acc_combine: ac,
        bfloat: bf,
        rows: r,
        cols: c,
        inner: i,
        channels: ch,
        convolve_x: cv,
        complex_x: cx,
        complex_y: cy,
        sparse: sp,
    }
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
        // 1024 bits = 2 elements of 256 bits (32 bytes) arranged as 2x2 matrix.
        // Input layout: [A0(32B), A1(32B), A2(32B), A3(32B)]
        // As 2x2: [[A0, A1], [A2, A3]]
        // Transposed: [[A0, A2], [A1, A3]]
        // Lo 512 bits: [A0, A2]
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T256_2x2Lo);

        // Element 0 stays (bytes 0..32 from lo), element 1 is from hi bytes 0..32
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

        // Transposed hi: [A1, A3]
        let mut expected = [0u8; VEC_BYTES];
        expected[..32].copy_from_slice(&lo[32..64]);
        expected[32..64].copy_from_slice(&hi[32..64]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_t32_16x2_lo_deinterleave() {
        // 1024 bits = 16x2 matrix of 32-bit (4-byte) elements.
        // Input: [e0, e1, e2, e3, ..., e31] (each 4 bytes)
        //   As 16x2: [[e0,e1], [e2,e3], ..., [e30,e31]]
        // Transposed (2x16): [[e0,e2,...,e30], [e1,e3,...,e31]]
        // Lo: even-indexed elements [e0, e2, e4, ..., e30]
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T32_16x2Lo);

        // Build 128-byte input
        let mut input = [0u8; PAIR_BYTES];
        input[..VEC_BYTES].copy_from_slice(&lo);
        input[VEC_BYTES..].copy_from_slice(&hi);

        // Check: result should contain elements at even column indices
        for i in 0..16 {
            let src_elem = i * 2; // even elements
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

        // Hi: odd-indexed elements [e1, e3, e5, ..., e31]
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
        // 64x2 matrix of 8-bit elements -> deinterleave bytes
        // Lo result should contain even-indexed bytes
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
        // 2x64 matrix of 8-bit elements -> interleave bytes
        // Transposed to 64x2: pairs of (lo[i], hi[i])
        // Lo result: bytes 0..63 of interleaved data
        let lo = identity_lo();
        let hi = identity_hi();
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T8_2x64Lo);

        let mut input = [0u8; PAIR_BYTES];
        input[..VEC_BYTES].copy_from_slice(&lo);
        input[VEC_BYTES..].copy_from_slice(&hi);

        // Input as 2x64: row 0 = lo[0..64], row 1 = hi[0..64]
        // Transposed to 64x2: element (c, r) at position c*2+r
        // Lo result = first 64 bytes = elements (0,0),(0,1),(1,0),(1,1),...,(31,0),(31,1)
        for c in 0..32 {
            for r in 0..2 {
                let dst_byte = c * 2 + r;
                // Source was at (r, c) in the 2x64 layout
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
        // Swap adjacent 16-bit pairs
        let mut lo = [0u8; VEC_BYTES];
        for i in 0..32 {
            let base = i * 2;
            lo[base] = (i * 2) as u8;
            lo[base + 1] = (i * 2 + 1) as u8;
        }
        let hi = [0u8; VEC_BYTES];
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T16_1x2Flip);

        // Each pair of 16-bit values should be swapped
        for i in 0..16 {
            let base = i * 4;
            // Original: [A_lo, A_hi, B_lo, B_hi]
            // Flipped:  [B_lo, B_hi, A_lo, A_hi]
            assert_eq!(result[base], lo[base + 2]);
            assert_eq!(result[base + 1], lo[base + 3]);
            assert_eq!(result[base + 2], lo[base]);
            assert_eq!(result[base + 3], lo[base + 1]);
        }
    }

    #[test]
    fn test_t32_4x4_transpose() {
        // 4x4 matrix of 32-bit (4-byte) elements = 64 bytes = exactly 512 bits
        // Transpose within a single vector
        let lo = identity_lo();
        let hi = [0u8; VEC_BYTES]; // unused for exact modes
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T32_4x4);

        // Element (r, c) at byte offset (r*4+c)*4 moves to (c*4+r)*4
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
        // 8x8 matrix of 8-bit elements = 64 bytes = exactly 512 bits
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
        // 4x4 matrix of 16-bit (2-byte) elements = 32 bytes < 64 bytes
        // Only uses first 32 bytes of input
        let lo = identity_lo();
        let hi = [0u8; VEC_BYTES];
        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T16_4x4);

        // Check that elements within the 4x4 block are transposed
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
    // This is how the AIE API implements filter_even for 32-bit types.
    #[test]
    fn test_deinterleave_32bit_pattern() {
        // Put sequential 32-bit values across lo and hi:
        // lo = [0, 1, 2, ..., 15], hi = [16, 17, ..., 31]
        let mut lo = [0u8; VEC_BYTES];
        let mut hi = [0u8; VEC_BYTES];
        for i in 0..16u32 {
            lo[i as usize * 4..i as usize * 4 + 4].copy_from_slice(&i.to_le_bytes());
        }
        for i in 0..16u32 {
            hi[i as usize * 4..i as usize * 4 + 4].copy_from_slice(&(i + 16).to_le_bytes());
        }

        let result = shuffle_vectors(&lo, &hi, ShuffleMode::T32_16x2Lo);

        // 128 bytes = 32 elements of 4 bytes, as 16x2 matrix:
        //   Row 0: [elem_0, elem_1], Row 1: [elem_2, elem_3], ...
        // Transposed to 2x16:
        //   Row 0 (lo result): [elem_0, elem_2, elem_4, ..., elem_30] = even indices
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

        // Deinterleave 16-bit: T16_32x2 separates even/odd 16-bit elements
        let evens = shuffle_vectors(&lo, &hi, ShuffleMode::T16_32x2Lo);
        let odds = shuffle_vectors(&lo, &hi, ShuffleMode::T16_32x2Hi);

        // Interleave them back: T16_2x32 should reconstruct the original
        let reconstructed_lo = shuffle_vectors(&evens, &odds, ShuffleMode::T16_2x32Lo);
        let reconstructed_hi = shuffle_vectors(&evens, &odds, ShuffleMode::T16_2x32Hi);

        assert_eq!(reconstructed_lo, lo, "lo mismatch after roundtrip");
        assert_eq!(reconstructed_hi, hi, "hi mismatch after roundtrip");
    }
}
