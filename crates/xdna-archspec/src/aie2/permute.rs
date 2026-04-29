//! AIE2 permute/shuffle data: ShuffleMode + MacPermuteMode enums,
//! SHUFFLE_ROUTING lookup table, and MAC permute configuration.
//!
//! Moved from src/interpreter/execute/vector_permute.rs as part of
//! Subsystem 7 Part B (audit item 2). Pure data -- no algorithmic
//! content. The shuffle routing table is hardware-probed; the two
//! enums (`ShuffleMode` with 48 variants, `MacPermuteMode` with 26
//! variants) mirror hardware mode sets exactly.
//!
//! Consumers in xdna-emu (`shuffle_vectors`, `rc2i`, test suites)
//! import these types via `use xdna_archspec::aie2::permute::*;`.

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
    T8_64x2Lo = 0,
    T8_64x2Hi = 1,
    T16_32x2Lo = 2,
    T16_32x2Hi = 3,
    T32_16x2Lo = 4,
    T32_16x2Hi = 5,
    T64_8x2Lo = 6,
    T64_8x2Hi = 7,
    T128_4x2Lo = 8,
    T128_4x2Hi = 9,
    T256_2x2Lo = 10,
    T256_2x2Hi = 11,
    // Interleave modes (2xN -> Nx2, extract half)
    T128_2x4Lo = 12,
    T128_2x4Hi = 13,
    T64_2x8Lo = 14,
    T64_2x8Hi = 15,
    T32_2x16Lo = 16,
    T32_2x16Hi = 17,
    T16_2x32Lo = 18,
    T16_2x32Hi = 19,
    T8_2x64Lo = 20,
    T8_2x64Hi = 21,
    // Bypass modes (512-bit halves)
    T512_1x2Lo = 22,
    T512_1x2Hi = 23,
    // NL (non-linear) modes
    T16_16x4Lo = 24,
    T16_16x4Hi = 25,
    T16_4x16Lo = 26,
    T16_4x16Hi = 27,
    // FFT and sparsity modes
    T16_8x4 = 28,
    T16_4x8 = 29,
    T32_8x4Lo = 30,
    T32_8x4Hi = 31,
    T32_4x8Lo = 32,
    T32_4x8Hi = 33,
    T32_4x4 = 34,
    T8_8x8 = 35,
    T8_16x4 = 36,
    T8_4x16 = 37,
    T16_1x2Flip = 38,
    // Permute reduction modes
    T16_4x4 = 39,
    T16_4x2 = 40,
    T16_2x4 = 41,
    T16_8x2 = 42,
    T16_2x8 = 43,
    T16_16x2 = 44,
    T16_2x16 = 45,
    T8_8x4 = 46,
    T8_4x8 = 47,
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
/// All 26 modes (0-25) correspond 1:1 with the 26 base configuration tuples
/// in aietools constants.py `__make_perm_modes()`. This has been verified
/// by cross-referencing each field (bits_x, bits_y, acc_cmb, bfloat, rows,
/// inner, cols, channels, complex_x, complex_y, convolve_x, sparse) against
/// the aietools source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum MacPermuteMode {
    // Matrix multiply modes (modes 0-8)
    /// 8b x 4b, 1 channel, 4x16x8 matrix multiply.
    Mode_8x4_4x16_16x8 = 0,
    /// 8b x 8b, 1 channel, 4x8x8 matrix multiply.
    Mode_8x8_4x8_8x8 = 1,
    /// 16b x 8b, 1 channel, 4x4x8 matrix multiply (red9).
    Mode_16x8_4x4_4x8 = 2,
    /// 16b x 16b, 1 channel, 4x2x8 matrix multiply.
    Mode_16x16_4x2_2x8 = 3,
    /// 16b x 8b, 2-accumulator, 2x8x8 matrix multiply.
    Mode_16x8_2x8_8x8 = 4,
    /// 16b x 8b, 2-accumulator, 4x8x4 matrix multiply.
    Mode_16x8_4x8_8x4 = 5,
    /// 16b x 16b, 2-accumulator, 2x4x8 matrix multiply.
    Mode_16x16_2x4_4x8 = 6,
    /// 16b x 16b, 2-accumulator, 4x4x4 matrix multiply.
    Mode_16x16_4x4_4x4 = 7,
    /// BFloat16 x BFloat16, 1 channel, 4x8x4 matrix multiply.
    Mode_BF16_4x8_8x4 = 8,

    // Element-wise modes (modes 9-12)
    /// 16b x 8b, 2 channels, 4x4x4 element-wise (red9).
    Mode_16x8_4x4_2ch = 9,
    /// 8b x 8b, 32 channels, 1x2x1 element-wise.
    Mode_8x8_Elem2 = 10,
    /// 16b x 16b, 32 channels, 1x1x1 element-wise.
    Mode_16x16_Elem = 11,
    /// 16b x 16b, 16 channels, 1x2x1 element-wise (2-accumulator).
    Mode_16x16_Elem2 = 12,

    // Convolution modes (modes 13-16)
    /// 8b x 8b, 8 channels, 4x4x1 depth-wise convolution (red10).
    Mode_8x8_Conv_4x4_8ch = 13,
    /// 8b x 8b, 4 channels, 8x8x1 depth-wise convolution (red10).
    Mode_8x8_Conv_8x8_4ch = 14,
    /// 8b x 8b, 1 channel, 32x8x1 2D filter convolution.
    Mode_8x8_Conv_32x8 = 15,
    /// 16b x 16b, 1 channel, 16x4x1 2D filter convolution (2-accumulator).
    Mode_16x16_Conv_16x4 = 16,

    // Additional modes (mode 17)
    /// BFloat16, 16 channels, 1x2x1 element-wise.
    Mode_BF16_Elem2 = 17,

    // FFT modes (modes 18-20)
    /// 32b x 16b, 1 channel, 4x2x4 matrix multiply (2-accumulator).
    Mode_32x16_4x2_2x4 = 18,
    /// 32b x 16b, 8 channels, complex element-wise (2-accumulator).
    Mode_32x16_Elem_Cplx = 19,
    /// 16b x 16b, 8 channels, complex element-wise (2-accumulator).
    Mode_16x16_Elem2_Cplx = 20,

    // Sparse modes (modes 21-25)
    /// Sparse: 8b x 4b, 1 channel, 4x32x8 (sparse variant of mode 0).
    Mode_8x4_Sparse = 21,
    /// Sparse: 8b x 8b, 1 channel, 4x16x8 (sparse variant of mode 1).
    Mode_8x8_Sparse = 22,
    /// Sparse: 16b x 8b, 2-accumulator, 2x16x8 (sparse variant of mode 4).
    Mode_16x8_Sparse = 23,
    /// Sparse: 16b x 16b, 2-accumulator, 2x8x8 (sparse variant of mode 6).
    Mode_16x16_Sparse = 24,
    /// Sparse: BFloat16, 1 channel, 4x16x4 (sparse variant of mode 8).
    Mode_BF16_Sparse = 25,
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
///
/// Every entry has been verified against the corresponding base configuration
/// tuple in aietools `constants.py` `__make_perm_modes()` (lines 268-300).
pub fn mac_permute_config(mode: MacPermuteMode) -> MacPermuteConfig {
    use MacPermuteMode::*;

    // Fields: (bits_x, bits_y, acc_cmb, bfloat, rows, inner, cols, channels,
    //          complex_x, complex_y, convolve_x, sparse)
    //
    // Cross-reference key (aietools constants.py tuple order):
    //   (bits_x, bits_y, acc_cmb, channels, rows, inner, cols,
    //    complex_x, complex_y, bfloat, convolve_x, convolve_y, sparse)
    // Note: convolve_y is always False in all 26 modes.
    let (bx, by, ac, bf, r, i, c, ch, cx, cy, cv, sp) = match mode {
        // Matrix multiply modes
        Mode_8x4_4x16_16x8 => (8, 4, 1, false, 4, 16, 8, 1, false, false, false, false),
        Mode_8x8_4x8_8x8 => (8, 8, 1, false, 4, 8, 8, 1, false, false, false, false),
        Mode_16x8_4x4_4x8 => (16, 8, 1, false, 4, 4, 8, 1, false, false, false, false),
        Mode_16x16_4x2_2x8 => (16, 16, 1, false, 4, 2, 8, 1, false, false, false, false),
        Mode_16x8_2x8_8x8 => (16, 8, 2, false, 2, 8, 8, 1, false, false, false, false),
        Mode_16x8_4x8_8x4 => (16, 8, 2, false, 4, 8, 4, 1, false, false, false, false),
        Mode_16x16_2x4_4x8 => (16, 16, 2, false, 2, 4, 8, 1, false, false, false, false),
        Mode_16x16_4x4_4x4 => (16, 16, 2, false, 4, 4, 4, 1, false, false, false, false),
        Mode_BF16_4x8_8x4 => (16, 16, 1, true, 4, 8, 4, 1, false, false, false, false),
        // Element-wise modes
        Mode_16x8_4x4_2ch => (16, 8, 1, false, 4, 4, 4, 2, false, false, false, false),
        Mode_8x8_Elem2 => (8, 8, 1, false, 1, 2, 1, 32, false, false, false, false),
        Mode_16x16_Elem => (16, 16, 1, false, 1, 1, 1, 32, false, false, false, false),
        Mode_16x16_Elem2 => (16, 16, 2, false, 1, 2, 1, 16, false, false, false, false),
        // Convolution modes
        Mode_8x8_Conv_4x4_8ch => (8, 8, 1, false, 4, 4, 1, 8, false, false, true, false),
        Mode_8x8_Conv_8x8_4ch => (8, 8, 1, false, 8, 8, 1, 4, false, false, true, false),
        Mode_8x8_Conv_32x8 => (8, 8, 1, false, 32, 8, 1, 1, false, false, true, false),
        Mode_16x16_Conv_16x4 => (16, 16, 2, false, 16, 4, 1, 1, false, false, true, false),
        // BFloat16 element-wise
        Mode_BF16_Elem2 => (16, 16, 1, true, 1, 2, 1, 16, false, false, false, false),
        // FFT modes
        Mode_32x16_4x2_2x4 => (32, 16, 2, false, 4, 2, 4, 1, false, false, false, false),
        Mode_32x16_Elem_Cplx => (32, 16, 2, false, 1, 1, 1, 8, true, true, false, false),
        Mode_16x16_Elem2_Cplx => (16, 16, 2, false, 1, 2, 1, 8, true, true, false, false),
        // Sparse modes
        Mode_8x4_Sparse => (8, 4, 1, false, 4, 32, 8, 1, false, false, false, true),
        Mode_8x8_Sparse => (8, 8, 1, false, 4, 16, 8, 1, false, false, false, true),
        Mode_16x8_Sparse => (16, 8, 2, false, 2, 16, 8, 1, false, false, false, true),
        Mode_16x16_Sparse => (16, 16, 2, false, 2, 8, 8, 1, false, false, false, true),
        Mode_BF16_Sparse => (16, 16, 1, true, 4, 16, 4, 1, false, false, false, true),
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
// Hardware-verified shuffle routing table
// ============================================================================

/// Byte-level routing table for all 48 shuffle modes.
///
/// SHUFFLE_ROUTING[mode][i] = which input byte (0-127) appears at output
/// position i. Input bytes 0-63 come from the lo vector, 64-127 from hi.
///
/// Cleanroom source: NPU hardware observation (tests/shuffle-sweep/).
/// Generated by running identity patterns through the VSHUFFLE instruction
/// on real AIE2 silicon and recording the output.
pub static SHUFFLE_ROUTING: [[u8; 64]; 48] = [
    // Mode 0
    [
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
        52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100,
        102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
    ],
    // Mode 1
    [
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51,
        53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101,
        103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127,
    ],
    // Mode 2
    [
        0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52,
        53, 56, 57, 60, 61, 64, 65, 68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97, 100, 101,
        104, 105, 108, 109, 112, 113, 116, 117, 120, 121, 124, 125,
    ],
    // Mode 3
    [
        2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51,
        54, 55, 58, 59, 62, 63, 66, 67, 70, 71, 74, 75, 78, 79, 82, 83, 86, 87, 90, 91, 94, 95, 98, 99, 102,
        103, 106, 107, 110, 111, 114, 115, 118, 119, 122, 123, 126, 127,
    ],
    // Mode 4
    [
        0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50,
        51, 56, 57, 58, 59, 64, 65, 66, 67, 72, 73, 74, 75, 80, 81, 82, 83, 88, 89, 90, 91, 96, 97, 98, 99,
        104, 105, 106, 107, 112, 113, 114, 115, 120, 121, 122, 123,
    ],
    // Mode 5
    [
        4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53,
        54, 55, 60, 61, 62, 63, 68, 69, 70, 71, 76, 77, 78, 79, 84, 85, 86, 87, 92, 93, 94, 95, 100, 101,
        102, 103, 108, 109, 110, 111, 116, 117, 118, 119, 124, 125, 126, 127,
    ],
    // Mode 6
    [
        0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50,
        51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71, 80, 81, 82, 83, 84, 85, 86, 87, 96, 97, 98, 99,
        100, 101, 102, 103, 112, 113, 114, 115, 116, 117, 118, 119,
    ],
    // Mode 7
    [
        8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57,
        58, 59, 60, 61, 62, 63, 72, 73, 74, 75, 76, 77, 78, 79, 88, 89, 90, 91, 92, 93, 94, 95, 104, 105,
        106, 107, 108, 109, 110, 111, 120, 121, 122, 123, 124, 125, 126, 127,
    ],
    // Mode 8
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 96, 97, 98, 99, 100,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    ],
    // Mode 9
    [
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 112, 113,
        114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    ],
    // Mode 10
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    ],
    // Mode 11
    [
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    ],
    // Mode 12
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
        76, 77, 78, 79, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 84,
        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    ],
    // Mode 13
    [
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 96, 97, 98, 99, 100, 101, 102, 103,
        104, 105, 106, 107, 108, 109, 110, 111, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
        63, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    ],
    // Mode 14
    [
        0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67, 68, 69, 70, 71, 8, 9, 10, 11, 12, 13, 14, 15, 72, 73, 74, 75,
        76, 77, 78, 79, 16, 17, 18, 19, 20, 21, 22, 23, 80, 81, 82, 83, 84, 85, 86, 87, 24, 25, 26, 27, 28,
        29, 30, 31, 88, 89, 90, 91, 92, 93, 94, 95,
    ],
    // Mode 15
    [
        32, 33, 34, 35, 36, 37, 38, 39, 96, 97, 98, 99, 100, 101, 102, 103, 40, 41, 42, 43, 44, 45, 46, 47,
        104, 105, 106, 107, 108, 109, 110, 111, 48, 49, 50, 51, 52, 53, 54, 55, 112, 113, 114, 115, 116, 117,
        118, 119, 56, 57, 58, 59, 60, 61, 62, 63, 120, 121, 122, 123, 124, 125, 126, 127,
    ],
    // Mode 16
    [
        0, 1, 2, 3, 64, 65, 66, 67, 4, 5, 6, 7, 68, 69, 70, 71, 8, 9, 10, 11, 72, 73, 74, 75, 12, 13, 14, 15,
        76, 77, 78, 79, 16, 17, 18, 19, 80, 81, 82, 83, 20, 21, 22, 23, 84, 85, 86, 87, 24, 25, 26, 27, 88,
        89, 90, 91, 28, 29, 30, 31, 92, 93, 94, 95,
    ],
    // Mode 17
    [
        32, 33, 34, 35, 96, 97, 98, 99, 36, 37, 38, 39, 100, 101, 102, 103, 40, 41, 42, 43, 104, 105, 106,
        107, 44, 45, 46, 47, 108, 109, 110, 111, 48, 49, 50, 51, 112, 113, 114, 115, 52, 53, 54, 55, 116,
        117, 118, 119, 56, 57, 58, 59, 120, 121, 122, 123, 60, 61, 62, 63, 124, 125, 126, 127,
    ],
    // Mode 18
    [
        0, 1, 64, 65, 2, 3, 66, 67, 4, 5, 68, 69, 6, 7, 70, 71, 8, 9, 72, 73, 10, 11, 74, 75, 12, 13, 76, 77,
        14, 15, 78, 79, 16, 17, 80, 81, 18, 19, 82, 83, 20, 21, 84, 85, 22, 23, 86, 87, 24, 25, 88, 89, 26,
        27, 90, 91, 28, 29, 92, 93, 30, 31, 94, 95,
    ],
    // Mode 19
    [
        32, 33, 96, 97, 34, 35, 98, 99, 36, 37, 100, 101, 38, 39, 102, 103, 40, 41, 104, 105, 42, 43, 106,
        107, 44, 45, 108, 109, 46, 47, 110, 111, 48, 49, 112, 113, 50, 51, 114, 115, 52, 53, 116, 117, 54,
        55, 118, 119, 56, 57, 120, 121, 58, 59, 122, 123, 60, 61, 124, 125, 62, 63, 126, 127,
    ],
    // Mode 20
    [
        0, 64, 1, 65, 2, 66, 3, 67, 4, 68, 5, 69, 6, 70, 7, 71, 8, 72, 9, 73, 10, 74, 11, 75, 12, 76, 13, 77,
        14, 78, 15, 79, 16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87, 24, 88, 25, 89, 26,
        90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95,
    ],
    // Mode 21
    [
        32, 96, 33, 97, 34, 98, 35, 99, 36, 100, 37, 101, 38, 102, 39, 103, 40, 104, 41, 105, 42, 106, 43,
        107, 44, 108, 45, 109, 46, 110, 47, 111, 48, 112, 49, 113, 50, 114, 51, 115, 52, 116, 53, 117, 54,
        118, 55, 119, 56, 120, 57, 121, 58, 122, 59, 123, 60, 124, 61, 125, 62, 126, 63, 127,
    ],
    // Mode 22
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ],
    // Mode 23
    [
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    ],
    // Mode 24
    [
        0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64, 65, 72, 73, 80, 81, 88, 89, 96, 97,
        104, 105, 112, 113, 120, 121, 2, 3, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 50, 51, 58, 59, 66, 67,
        74, 75, 82, 83, 90, 91, 98, 99, 106, 107, 114, 115, 122, 123,
    ],
    // Mode 25
    [
        4, 5, 12, 13, 20, 21, 28, 29, 36, 37, 44, 45, 52, 53, 60, 61, 68, 69, 76, 77, 84, 85, 92, 93, 100,
        101, 108, 109, 116, 117, 124, 125, 6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55, 62, 63, 70,
        71, 78, 79, 86, 87, 94, 95, 102, 103, 110, 111, 118, 119, 126, 127,
    ],
    // Mode 26
    [
        0, 1, 32, 33, 64, 65, 96, 97, 2, 3, 34, 35, 66, 67, 98, 99, 4, 5, 36, 37, 68, 69, 100, 101, 6, 7, 38,
        39, 70, 71, 102, 103, 8, 9, 40, 41, 72, 73, 104, 105, 10, 11, 42, 43, 74, 75, 106, 107, 12, 13, 44,
        45, 76, 77, 108, 109, 14, 15, 46, 47, 78, 79, 110, 111,
    ],
    // Mode 27
    [
        16, 17, 48, 49, 80, 81, 112, 113, 18, 19, 50, 51, 82, 83, 114, 115, 20, 21, 52, 53, 84, 85, 116, 117,
        22, 23, 54, 55, 86, 87, 118, 119, 24, 25, 56, 57, 88, 89, 120, 121, 26, 27, 58, 59, 90, 91, 122, 123,
        28, 29, 60, 61, 92, 93, 124, 125, 30, 31, 62, 63, 94, 95, 126, 127,
    ],
    // Mode 28
    [
        0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 2, 3, 10, 11, 18, 19, 26, 27, 34, 35, 42,
        43, 50, 51, 58, 59, 4, 5, 12, 13, 20, 21, 28, 29, 36, 37, 44, 45, 52, 53, 60, 61, 6, 7, 14, 15, 22,
        23, 30, 31, 38, 39, 46, 47, 54, 55, 62, 63,
    ],
    // Mode 29
    [
        0, 1, 16, 17, 32, 33, 48, 49, 2, 3, 18, 19, 34, 35, 50, 51, 4, 5, 20, 21, 36, 37, 52, 53, 6, 7, 22,
        23, 38, 39, 54, 55, 8, 9, 24, 25, 40, 41, 56, 57, 10, 11, 26, 27, 42, 43, 58, 59, 12, 13, 28, 29, 44,
        45, 60, 61, 14, 15, 30, 31, 46, 47, 62, 63,
    ],
    // Mode 30
    [
        0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67, 80, 81, 82, 83, 96, 97,
        98, 99, 112, 113, 114, 115, 4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55, 68, 69, 70,
        71, 84, 85, 86, 87, 100, 101, 102, 103, 116, 117, 118, 119,
    ],
    // Mode 31
    [
        8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59, 72, 73, 74, 75, 88, 89, 90, 91, 104,
        105, 106, 107, 120, 121, 122, 123, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63,
        76, 77, 78, 79, 92, 93, 94, 95, 108, 109, 110, 111, 124, 125, 126, 127,
    ],
    // Mode 32
    [
        0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99, 4, 5, 6, 7, 36, 37, 38, 39, 68, 69, 70,
        71, 100, 101, 102, 103, 8, 9, 10, 11, 40, 41, 42, 43, 72, 73, 74, 75, 104, 105, 106, 107, 12, 13, 14,
        15, 44, 45, 46, 47, 76, 77, 78, 79, 108, 109, 110, 111,
    ],
    // Mode 33
    [
        16, 17, 18, 19, 48, 49, 50, 51, 80, 81, 82, 83, 112, 113, 114, 115, 20, 21, 22, 23, 52, 53, 54, 55,
        84, 85, 86, 87, 116, 117, 118, 119, 24, 25, 26, 27, 56, 57, 58, 59, 88, 89, 90, 91, 120, 121, 122,
        123, 28, 29, 30, 31, 60, 61, 62, 63, 92, 93, 94, 95, 124, 125, 126, 127,
    ],
    // Mode 34
    [
        0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38,
        39, 52, 53, 54, 55, 8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59, 12, 13, 14, 15, 28,
        29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63,
    ],
    // Mode 35
    [
        0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19,
        27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38,
        46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
    ],
    // Mode 36
    [
        0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41,
        45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19,
        23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
    ],
    // Mode 37
    [
        0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38,
        54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13,
        29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63,
    ],
    // Mode 38
    [
        2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25,
        30, 31, 28, 29, 34, 35, 32, 33, 38, 39, 36, 37, 42, 43, 40, 41, 46, 47, 44, 45, 50, 51, 48, 49, 54,
        55, 52, 53, 58, 59, 56, 57, 62, 63, 60, 61,
    ],
    // Mode 39
    [
        0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15,
        22, 23, 30, 31, 32, 33, 40, 41, 48, 49, 56, 57, 34, 35, 42, 43, 50, 51, 58, 59, 36, 37, 44, 45, 52,
        53, 60, 61, 38, 39, 46, 47, 54, 55, 62, 63,
    ],
    // Mode 40
    [
        0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23,
        26, 27, 30, 31, 32, 33, 36, 37, 40, 41, 44, 45, 34, 35, 38, 39, 42, 43, 46, 47, 48, 49, 52, 53, 56,
        57, 60, 61, 50, 51, 54, 55, 58, 59, 62, 63,
    ],
    // Mode 41
    [
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 16, 17, 24, 25, 18, 19, 26, 27, 20, 21, 28, 29,
        22, 23, 30, 31, 32, 33, 40, 41, 34, 35, 42, 43, 36, 37, 44, 45, 38, 39, 46, 47, 48, 49, 56, 57, 50,
        51, 58, 59, 52, 53, 60, 61, 54, 55, 62, 63,
    ],
    // Mode 42
    [
        0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23,
        26, 27, 30, 31, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 34, 35, 38, 39, 42,
        43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63,
    ],
    // Mode 43
    [
        0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29,
        14, 15, 30, 31, 32, 33, 48, 49, 34, 35, 50, 51, 36, 37, 52, 53, 38, 39, 54, 55, 40, 41, 56, 57, 42,
        43, 58, 59, 44, 45, 60, 61, 46, 47, 62, 63,
    ],
    // Mode 44
    [
        0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52,
        53, 56, 57, 60, 61, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42,
        43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63,
    ],
    // Mode 45
    [
        0, 1, 32, 33, 2, 3, 34, 35, 4, 5, 36, 37, 6, 7, 38, 39, 8, 9, 40, 41, 10, 11, 42, 43, 12, 13, 44, 45,
        14, 15, 46, 47, 16, 17, 48, 49, 18, 19, 50, 51, 20, 21, 52, 53, 22, 23, 54, 55, 24, 25, 56, 57, 26,
        27, 58, 59, 28, 29, 60, 61, 30, 31, 62, 63,
    ],
    // Mode 46
    [
        0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15,
        19, 23, 27, 31, 32, 36, 40, 44, 48, 52, 56, 60, 33, 37, 41, 45, 49, 53, 57, 61, 34, 38, 42, 46, 50,
        54, 58, 62, 35, 39, 43, 47, 51, 55, 59, 63,
    ],
    // Mode 47
    [
        0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22,
        30, 7, 15, 23, 31, 32, 40, 48, 56, 33, 41, 49, 57, 34, 42, 50, 58, 35, 43, 51, 59, 36, 44, 52, 60,
        37, 45, 53, 61, 38, 46, 54, 62, 39, 47, 55, 63,
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_routing_table_shape() {
        assert_eq!(SHUFFLE_ROUTING.len(), 48, "48 shuffle modes");
        assert_eq!(SHUFFLE_ROUTING[0].len(), 64, "64-byte vector output");
    }

    #[test]
    fn shuffle_routing_mode_0_is_even_bytes_deinterleave() {
        // Mode 0 = T8_64x2Lo: deinterleave even bytes of a 128-byte input.
        // Output[i] = input[2i] for i in 0..64.
        for i in 0..64 {
            assert_eq!(
                SHUFFLE_ROUTING[0][i],
                (2 * i) as u8,
                "mode 0 byte {i} should route from input {}",
                2 * i
            );
        }
    }

    #[test]
    fn shuffle_routing_mode_1_is_odd_bytes_deinterleave() {
        // Mode 1 = T8_64x2Hi: deinterleave odd bytes.
        for i in 0..64 {
            assert_eq!(
                SHUFFLE_ROUTING[1][i],
                (2 * i + 1) as u8,
                "mode 1 byte {i} should route from input {}",
                2 * i + 1
            );
        }
    }

    #[test]
    fn shuffle_mode_from_mode_roundtrip() {
        for raw in 0u8..=47 {
            let mode = ShuffleMode::from_mode(raw).expect("0..=47 are valid");
            assert_eq!(mode.as_u8(), raw);
        }
        assert_eq!(ShuffleMode::from_mode(48), None);
        assert_eq!(ShuffleMode::from_mode(255), None);
    }

    #[test]
    fn mac_permute_mode_from_mode_roundtrip() {
        for raw in 0u8..=25 {
            let mode = MacPermuteMode::from_mode(raw).expect("0..=25 are valid");
            assert_eq!(mode.as_u8(), raw);
        }
        assert_eq!(MacPermuteMode::from_mode(26), None);
        assert_eq!(MacPermuteMode::from_mode(255), None);
    }

    #[test]
    fn mac_permute_config_mode_0_matrix_multiply() {
        // Mode 0 = 8b x 4b, 1 channel, 4x16x8 matrix multiply, no flags.
        let cfg = mac_permute_config(MacPermuteMode::Mode_8x4_4x16_16x8);
        assert_eq!(cfg.bits_x, 8);
        assert_eq!(cfg.bits_y, 4);
        assert_eq!(cfg.acc_combine, 1);
        assert!(!cfg.bfloat);
        assert_eq!(cfg.rows, 4);
        assert_eq!(cfg.inner, 16);
        assert_eq!(cfg.cols, 8);
        assert_eq!(cfg.channels, 1);
        assert!(!cfg.convolve_x);
        assert!(!cfg.complex_x);
        assert!(!cfg.complex_y);
        assert!(!cfg.sparse);
    }

    #[test]
    fn mac_permute_config_sparse_mode_flags_sparse() {
        let cfg = mac_permute_config(MacPermuteMode::Mode_BF16_Sparse);
        assert!(cfg.sparse);
        assert!(cfg.bfloat);
    }
}
