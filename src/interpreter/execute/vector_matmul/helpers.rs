//! Element extraction and accumulator I/O helpers for matrix multiply.
//!
//! These utility functions handle packing/unpacking elements from various
//! register widths (256-bit, 512-bit, 1024-bit) and accumulator read/write
//! in both 32-bit and 64-bit lane modes.

use crate::interpreter::state::{Vec512, Acc1024};
use crate::interpreter::execute::vector_config::AccWidth;
use crate::interpreter::bundle::ElementType;

/// Tile geometry for a matrix multiply mode.
#[derive(Debug, Clone, Copy)]
pub(super) struct TileGeometry {
    pub rows: usize,
    pub inner: usize,
    pub cols: usize,
}

// ---------------------------------------------------------------------------
// 256-bit element extraction (legacy [u32; 8] format)
// ---------------------------------------------------------------------------

/// Extract int8 elements from packed [u32; 8] (256 bits = 32 bytes = 32 int8 values).
/// Elements are in little-endian byte order within each u32.
pub(super) fn extract_i8(packed: &[u32; 8], index: usize) -> i8 {
    let word = index / 4;
    let byte = index % 4;
    ((packed[word] >> (byte * 8)) & 0xFF) as u8 as i8
}

/// Extract uint8 elements from packed [u32; 8].
pub(super) fn extract_u8(packed: &[u32; 8], index: usize) -> u8 {
    let word = index / 4;
    let byte = index % 4;
    ((packed[word] >> (byte * 8)) & 0xFF) as u8
}

/// Extract int16 elements from packed [u32; 8] (256 bits = 16 int16 values).
pub(super) fn extract_i16(packed: &[u32; 8], index: usize) -> i16 {
    let word = index / 2;
    let half = index % 2;
    ((packed[word] >> (half * 16)) & 0xFFFF) as u16 as i16
}

/// Extract uint16 elements from packed [u32; 8].
pub(super) fn extract_u16(packed: &[u32; 8], index: usize) -> u16 {
    let word = index / 2;
    let half = index % 2;
    ((packed[word] >> (half * 16)) & 0xFFFF) as u16
}

/// Extract bf16 as f32 from packed [u32; 8] (256 bits = 16 bf16 values).
pub(super) fn extract_bf16_as_f32(packed: &[u32; 8], index: usize) -> f32 {
    let word = index / 2;
    let half = index % 2;
    let bits = ((packed[word] >> (half * 16)) & 0xFFFF) as u16;
    f32::from_bits((bits as u32) << 16)
}

/// Extract int32 elements from packed [u32; 8] (256 bits = 8 int32 values).
pub(super) fn extract_i32(packed: &[u32; 8], index: usize) -> i32 {
    packed[index] as i32
}

// ---------------------------------------------------------------------------
// 512-bit (8 u64 lanes) accumulator I/O
// ---------------------------------------------------------------------------

/// Read a 32-bit accumulator lane from the [u64; 8] accumulator.
///
/// The 8 u64 lanes hold 16 int32 values (acc_cmb=1 mode, 32-bit accumulator).
/// Lane layout: acc[0] holds output[0] in low 32 bits, output[1] in high 32 bits, etc.
pub(super) fn read_acc32(acc: &[u64; 8], index: usize) -> i64 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    bits as i32 as i64
}

/// Write a 32-bit accumulator lane into the [u64; 8] accumulator.
pub(super) fn write_acc32(acc: &mut [u64; 8], index: usize, value: i64) {
    let u64_lane = index / 2;
    let half = index % 2;
    let masked = (value as u32) as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (masked << shift);
}

/// Read a 64-bit accumulator lane (acc_cmb=2 mode).
pub(super) fn read_acc64(acc: &[u64; 8], index: usize) -> i64 {
    acc[index] as i64
}

/// Write a 64-bit accumulator lane (acc_cmb=2 mode).
pub(super) fn write_acc64(acc: &mut [u64; 8], index: usize, value: i64) {
    acc[index] = value as u64;
}

/// Read a float32 accumulator lane from [u64; 8].
///
/// For bf16 matmul, the accumulator holds fp32 values. Since we have 16 output
/// elements (4x4) and 8 u64 lanes, each u64 holds two fp32 values.
pub(super) fn read_acc_f32(acc: &[u64; 8], index: usize) -> f32 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    f32::from_bits(bits)
}

/// Write a float32 accumulator lane.
pub(super) fn write_acc_f32(acc: &mut [u64; 8], index: usize, value: f32) {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = value.to_bits() as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (bits << shift);
}

// ---------------------------------------------------------------------------
// 1024-bit (16 u64 lanes) wide accumulator I/O
// ---------------------------------------------------------------------------

/// Extract an element from a 512-bit vector (`Vec512` = `[u32; 16]`).
///
/// `byte_idx` is the byte offset within the full 64-byte vector.
pub(super) fn extract_element_512(src: &Vec512, byte_idx: usize, bits: u32, signed: bool) -> i64 {
    match bits {
        4 => {
            // 4-bit elements: two per byte
            let byte_pos = byte_idx / 2;
            let nibble = byte_idx % 2;
            let word = byte_pos / 4;
            let byte_in_word = byte_pos % 4;
            if word >= src.len() {
                return 0;
            }
            let raw_byte = ((src[word] >> (byte_in_word * 8)) & 0xFF) as u8;
            let val = if nibble == 0 {
                raw_byte & 0xF
            } else {
                (raw_byte >> 4) & 0xF
            };
            if signed && (val & 0x8) != 0 {
                // Sign-extend from 4 bits
                (val as i8 | !0xFu8 as i8) as i64
            } else {
                val as i64
            }
        }
        8 => {
            let word = byte_idx / 4;
            let byte_in_word = byte_idx % 4;
            if word >= src.len() {
                return 0;
            }
            let val = ((src[word] >> (byte_in_word * 8)) & 0xFF) as u8;
            if signed {
                val as i8 as i64
            } else {
                val as i64
            }
        }
        16 => {
            let elem_idx = byte_idx / 2;
            let word = elem_idx / 2;
            let half_in_word = elem_idx % 2;
            if word >= src.len() {
                return 0;
            }
            let val = ((src[word] >> (half_in_word * 16)) & 0xFFFF) as u16;
            if signed {
                val as i16 as i64
            } else {
                val as i64
            }
        }
        32 => {
            let word = byte_idx / 4;
            if word >= src.len() {
                return 0;
            }
            let val = src[word];
            if signed {
                val as i32 as i64
            } else {
                val as i64
            }
        }
        _ => 0,
    }
}

/// Read from a 1024-bit accumulator (`Acc1024` = `[u64; 16]`).
pub(super) fn read_acc_wide(acc: &Acc1024, index: usize, acc_width: AccWidth) -> i64 {
    match acc_width {
        AccWidth::Acc32 => {
            // 32 x 32-bit lanes packed into 16 u64 words (two per word)
            let u64_lane = index / 2;
            let half = index % 2;
            let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
            bits as i32 as i64
        }
        AccWidth::Acc64 => {
            // 16 x 64-bit lanes, one per u64 word
            acc[index] as i64
        }
    }
}

/// Write to a 1024-bit accumulator (`Acc1024` = `[u64; 16]`).
pub(super) fn write_acc_wide(acc: &mut Acc1024, index: usize, value: i64, acc_width: AccWidth) {
    match acc_width {
        AccWidth::Acc32 => {
            let u64_lane = index / 2;
            let half = index % 2;
            let masked = (value as u32) as u64;
            let shift = half * 32;
            acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (masked << shift);
        }
        AccWidth::Acc64 => {
            acc[index] = value as u64;
        }
    }
}

/// Dense matrix multiply: acc += A * B (or acc = A * B if clear_acc is true).
///
/// Performs a tiled matrix multiply based on the element type. The input vectors
/// are reinterpreted as 2D tiles and multiplied using the geometry appropriate
/// for the element type combination.
pub fn matmul_dense(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    elem_type: ElementType,
    signed_a: bool,
    signed_b: bool,
) {
    match elem_type {
        ElementType::Int8 => super::matmul_i8xi8(acc, a, b, true, true, false),
        ElementType::UInt8 => super::matmul_i8xi8(acc, a, b, signed_a, signed_b, false),
        ElementType::Int16 => super::matmul_i16xi16_32(acc, a, b, true, true, false),
        ElementType::UInt16 => super::matmul_i16xi16_32(acc, a, b, signed_a, signed_b, false),
        ElementType::BFloat16 => super::matmul_bf16xbf16(acc, a, b, false),
        ElementType::Int32 => super::matmul_i32xi16(acc, a, b, true, true, false),
        ElementType::UInt32 => super::matmul_i32xi16(acc, a, b, false, false, false),
        ElementType::Int64 | ElementType::UInt64 => {
            super::matmul_i32xi16(acc, a, b, signed_a, signed_b, false)
        }
        ElementType::Float32 => super::matmul_bf16xbf16(acc, a, b, false),
    }
}

/// Matrix multiply-subtract: acc -= A * B.
pub fn matmul_sub(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    elem_type: ElementType,
    signed_a: bool,
    signed_b: bool,
) {
    match elem_type {
        ElementType::Int8 => super::matmul_i8xi8(acc, a, b, true, true, true),
        ElementType::UInt8 => super::matmul_i8xi8(acc, a, b, signed_a, signed_b, true),
        ElementType::Int16 => super::matmul_i16xi16_32(acc, a, b, true, true, true),
        ElementType::UInt16 => super::matmul_i16xi16_32(acc, a, b, signed_a, signed_b, true),
        ElementType::BFloat16 => super::matmul_bf16xbf16(acc, a, b, true),
        ElementType::Int32 => super::matmul_i32xi16(acc, a, b, true, true, true),
        ElementType::UInt32 => super::matmul_i32xi16(acc, a, b, false, false, true),
        ElementType::Int64 | ElementType::UInt64 => {
            super::matmul_i32xi16(acc, a, b, signed_a, signed_b, true)
        }
        ElementType::Float32 => super::matmul_bf16xbf16(acc, a, b, true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_i8() {
        let packed = pack_i8(&[
            1, -2, 3, -4, 5, -6, 7, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,
        ]);
        assert_eq!(extract_i8(&packed, 0), 1);
        assert_eq!(extract_i8(&packed, 1), -2);
        assert_eq!(extract_i8(&packed, 2), 3);
        assert_eq!(extract_i8(&packed, 3), -4);
        assert_eq!(extract_i8(&packed, 4), 5);
        assert_eq!(extract_i8(&packed, 5), -6);
    }

    #[test]
    fn test_extract_i16() {
        let packed = pack_i16(&[100, -200, 300, -400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(extract_i16(&packed, 0), 100);
        assert_eq!(extract_i16(&packed, 1), -200);
        assert_eq!(extract_i16(&packed, 2), 300);
        assert_eq!(extract_i16(&packed, 3), -400);
    }

    #[test]
    fn test_acc32_read_write_roundtrip() {
        let mut acc = [0u64; 8];
        write_acc32(&mut acc, 0, 42);
        write_acc32(&mut acc, 1, -100);
        write_acc32(&mut acc, 15, 999);

        assert_eq!(read_acc32(&acc, 0), 42);
        assert_eq!(read_acc32(&acc, 1), -100);
        assert_eq!(read_acc32(&acc, 15), 999);
    }

    #[test]
    fn test_acc_f32_read_write_roundtrip() {
        let mut acc = [0u64; 8];
        write_acc_f32(&mut acc, 0, 3.14);
        write_acc_f32(&mut acc, 1, -2.71);
        write_acc_f32(&mut acc, 15, 42.0);

        assert!((read_acc_f32(&acc, 0) - 3.14).abs() < 0.001);
        assert!((read_acc_f32(&acc, 1) - (-2.71)).abs() < 0.001);
        assert!((read_acc_f32(&acc, 15) - 42.0).abs() < 0.001);
    }

    // Test helper: pack int8 values into [u32; 8] in little-endian order.
    fn pack_i8(values: &[i8]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate() {
            let word = i / 4;
            let byte = i % 4;
            packed[word] |= ((v as u8) as u32) << (byte * 8);
        }
        packed
    }

    // Test helper: pack int16 values into [u32; 8] in little-endian order.
    fn pack_i16(values: &[i16]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate() {
            let word = i / 2;
            let half = i % 2;
            packed[word] |= ((v as u16) as u32) << (half * 16);
        }
        packed
    }
}
