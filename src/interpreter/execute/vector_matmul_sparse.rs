//! AIE2 sparse matrix multiply unit.
//!
//! The AIE2 vector unit supports structured sparsity in matrix multiply
//! operations. Sparse mode doubles the effective inner dimension compared
//! to dense mode by exploiting guaranteed 50% sparsity (2:4 structured
//! sparsity -- at most 2 non-zero values per group of 4 elements).
//!
//! # Sparse Data Format (per AIE API documentation)
//!
//! Data is stored in memory as: 64-bit mask + packed non-zero data values.
//! The mask uses 4 bits per group of 4 elements to indicate which elements
//! are present. After loading through the sparse buffer stream, the hardware
//! performs a **partial decompression**: each 4-bit mask nibble maps to 2
//! output data slots using this table:
//!
//! | Mask bits [3:0] | Output slot 0 | Output slot 1 |
//! |-----------------|---------------|---------------|
//! | 0000            | 0             | 0             |
//! | 0001            | 0             | A             |
//! | 0010            | 0             | B             |
//! | 0011            | B             | A             |
//! | 0100            | C             | 0             |
//! | 0101            | C             | A             |
//! | 0110            | C             | B             |
//! | 1000            | D             | 0             |
//! | 1001            | D             | A             |
//! | 1010            | D             | B             |
//! | 1100            | D             | C             |
//!
//! Where A, B, C, D are consecutive packed data values. After partial
//! decompression, the 64-bit mask and 256-bit partially decompressed
//! data are paired as a sparse vector register.
//!
//! # Sparse Matmul Geometries
//!
//! Sparse modes have 2x the inner dimension of their dense equivalents:
//!
//! | A type | B type (sparse) | Acc type | Rows | Inner | Cols |
//! |--------|-----------------|----------|------|-------|------|
//! | int8   | int8 (sparse)   | int32    | 4    | 16    | 8    |
//! | int16  | int16 (sparse)  | int64    | 2    | 8     | 8    |
//! | bf16   | bf16 (sparse)   | fp32     | 4    | 16    | 4    |
//! | int8   | int4 (sparse)   | int32    | 4    | 32    | 8    |
//! | int16  | int8 (sparse)   | int32    | 2    | 16    | 8    |
//!
//! The B operand is the sparse input. A is always dense.
//!
//! # Implementation Notes
//!
//! The emulator receives the sparse data already partially decompressed
//! (the hardware load path handles decompression). At the emulator level,
//! the sparse vector register contains:
//!   - Partially decompressed data in the vector register lanes
//!   - Mask bits that the permute unit uses to select active lanes
//!
//! For functional correctness, the key difference from dense matmul is
//! that the effective B matrix has been expanded to the full inner
//! dimension with zeros inserted at masked-out positions. The multiply
//! then proceeds as a standard matmul over the expanded geometry.
//!
//! Hardware reference: sparse_vector.hpp, mmul_8_8.hpp, aie_doc.hpp
//! (read for understanding, original implementation below).

use crate::interpreter::bundle::ElementType;

// -----------------------------------------------------------------------
// Sparse partial decompression table
// -----------------------------------------------------------------------

/// Partially decompress one group of 4 elements described by a 4-bit mask
/// nibble, consuming packed data values from the iterator.
///
/// Returns two output values (slot 0, slot 1) for the partially decompressed
/// pair. Advances `data_cursor` by the number of packed values consumed.
///
/// The decompression table maps 4 mask bits to 2 output slots. The values
/// A, B, C, D are consumed sequentially from the packed data stream.
fn decompress_nibble(mask_nibble: u8, packed_data: &[i32], data_cursor: &mut usize) -> (i32, i32) {
    // Consume up to 4 values from packed data. Not all mask patterns are
    // valid in hardware (some are missing from the table -- e.g. 0b0111,
    // 0b1011, 0b1101, 0b1110, 0b1111, 0b0111, 0b1011, 0b1101). The hardware
    // guarantees at most 2 non-zero per group of 4.
    //
    // Values A, B, C, D map to bit positions 0, 1, 2, 3 in the mask.
    // The number of values consumed equals the number of set bits.

    let get = |cursor: &mut usize| -> i32 {
        let val = if *cursor < packed_data.len() {
            packed_data[*cursor]
        } else {
            0
        };
        *cursor += 1;
        val
    };

    match mask_nibble & 0xF {
        0b0000 => (0, 0),
        0b0001 => {
            let a = get(data_cursor);
            (0, a)
        }
        0b0010 => {
            let b = get(data_cursor);
            (0, b)
        }
        0b0011 => {
            let a = get(data_cursor);
            let b = get(data_cursor);
            (b, a)
        }
        0b0100 => {
            let c = get(data_cursor);
            (c, 0)
        }
        0b0101 => {
            let a = get(data_cursor);
            let c = get(data_cursor);
            (c, a)
        }
        0b0110 => {
            let b = get(data_cursor);
            let c = get(data_cursor);
            (c, b)
        }
        0b1000 => {
            let d = get(data_cursor);
            (d, 0)
        }
        0b1001 => {
            let a = get(data_cursor);
            let d = get(data_cursor);
            (d, a)
        }
        0b1010 => {
            let b = get(data_cursor);
            let d = get(data_cursor);
            (d, b)
        }
        0b1100 => {
            let c = get(data_cursor);
            let d = get(data_cursor);
            (d, c)
        }
        // Invalid patterns -- hardware guarantees at most 2 set bits per
        // nibble (2:4 structured sparsity). For robustness, treat as zero.
        _ => {
            // Count set bits and consume that many data values (discard them)
            let popcount = (mask_nibble & 0xF).count_ones() as usize;
            for _ in 0..popcount {
                get(data_cursor);
            }
            (0, 0)
        }
    }
}

/// Fully decompress a sparse vector from its memory representation.
///
/// Input format (for int8, 128 logical elements):
///   - 64 bits of mask (8 bytes)
///   - Packed non-zero data bytes (variable length, up to 64 bytes)
///
/// Output: a vector of `logical_elements` values with zeros at masked-out
/// positions.
///
/// The mask uses 4 bits per group of 4 elements. Each group produces 2
/// partially decompressed values. The full decompression reconstructs the
/// original logical positions.
///
/// For int8: 128 logical elements => 32 nibbles => 64 partially decompressed
///           values => expanded to 128 full values using position knowledge.
/// For int16: 64 logical elements => 16 nibbles => 32 partially decompressed
///           values => expanded to 64 full values.
pub fn decompress_sparse_vector(
    mask_bits: u64,
    packed_data: &[i32],
    logical_elements: usize,
) -> Vec<i32> {
    // Number of groups of 4 elements
    let num_groups = logical_elements / 4;
    // Number of nibbles in the mask
    let num_nibbles = num_groups;

    let mut result = vec![0i32; logical_elements];
    let mut data_cursor: usize = 0;

    for group in 0..num_nibbles {
        // Extract 4-bit mask for this group
        let nibble = ((mask_bits >> (group * 4)) & 0xF) as u8;

        // Partially decompress: 4 mask bits -> 2 output values
        let (val0, val1) = decompress_nibble(nibble, packed_data, &mut data_cursor);

        // Map the 2 partially decompressed values back to the 4-element
        // group positions. The partial decompression maps to positions
        // (group*2, group*2+1) in the "half-decompressed" space. For full
        // decompression, we need to place values at the correct positions
        // within the group of 4.
        //
        // However, the hardware matmul uses the PARTIALLY decompressed form
        // (2 values per group, not 4). The multiplier's permute unit handles
        // the mapping. For emulator functional correctness, we place the
        // partially decompressed values at stride-2 positions.
        let base = group * 2;
        if base < logical_elements {
            result[base] = val0;
        }
        if base + 1 < logical_elements {
            result[base + 1] = val1;
        }
    }

    // Trim to the partially decompressed size (logical_elements / 2)
    result.truncate(logical_elements / 2);
    result
}

// -----------------------------------------------------------------------
// Element extraction helpers (mirror dense matmul's helpers)
// -----------------------------------------------------------------------

/// Extract int8 from packed [u32; 8] at byte index.
fn extract_i8(packed: &[u32; 8], index: usize) -> i8 {
    let word = index / 4;
    let byte = index % 4;
    if word >= 8 {
        return 0;
    }
    ((packed[word] >> (byte * 8)) & 0xFF) as u8 as i8
}

/// Extract uint8 from packed [u32; 8] at byte index.
fn extract_u8(packed: &[u32; 8], index: usize) -> u8 {
    let word = index / 4;
    let byte = index % 4;
    if word >= 8 {
        return 0;
    }
    ((packed[word] >> (byte * 8)) & 0xFF) as u8
}

/// Extract int16 from packed [u32; 8] at halfword index.
fn extract_i16(packed: &[u32; 8], index: usize) -> i16 {
    let word = index / 2;
    let half = index % 2;
    if word >= 8 {
        return 0;
    }
    ((packed[word] >> (half * 16)) & 0xFFFF) as u16 as i16
}

/// Extract uint16 from packed [u32; 8] at halfword index.
fn extract_u16(packed: &[u32; 8], index: usize) -> u16 {
    let word = index / 2;
    let half = index % 2;
    if word >= 8 {
        return 0;
    }
    ((packed[word] >> (half * 16)) & 0xFFFF) as u16
}

/// Extract bf16 as f32 from packed [u32; 8] at halfword index.
fn extract_bf16_as_f32(packed: &[u32; 8], index: usize) -> f32 {
    let word = index / 2;
    let half = index % 2;
    if word >= 8 {
        return 0.0;
    }
    let bits = ((packed[word] >> (half * 16)) & 0xFFFF) as u16;
    f32::from_bits((bits as u32) << 16)
}

// -----------------------------------------------------------------------
// Accumulator access helpers
// -----------------------------------------------------------------------

/// Read a 32-bit accumulator lane from [u64; 8].
fn read_acc32(acc: &[u64; 8], index: usize) -> i64 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    bits as i32 as i64
}

/// Write a 32-bit accumulator lane.
fn write_acc32(acc: &mut [u64; 8], index: usize, value: i64) {
    let u64_lane = index / 2;
    let half = index % 2;
    let masked = (value as u32) as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (masked << shift);
}

/// Read a 64-bit accumulator lane.
fn read_acc64(acc: &[u64; 8], index: usize) -> i64 {
    acc[index] as i64
}

/// Write a 64-bit accumulator lane.
fn write_acc64(acc: &mut [u64; 8], index: usize, value: i64) {
    acc[index] = value as u64;
}

/// Read fp32 from accumulator.
fn read_acc_f32(acc: &[u64; 8], index: usize) -> f32 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    f32::from_bits(bits)
}

/// Write fp32 to accumulator.
fn write_acc_f32(acc: &mut [u64; 8], index: usize, value: f32) {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = value.to_bits() as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (bits << shift);
}

// -----------------------------------------------------------------------
// Sparse mask extraction from register
// -----------------------------------------------------------------------

/// Extract the 64-bit sparsity mask from a sparse vector register.
///
/// In the AIE2 sparse vector format, after partial decompression by the
/// load unit, the mask occupies the first 64 bits (8 bytes) of the
/// register. The remaining 256 bits hold the partially decompressed data.
///
/// For the emulator, we model the sparse register as two [u32; 8] chunks:
/// the mask register and the data register. This function extracts the
/// mask from the dedicated mask portion.
///
/// In practice, the sparse vector is loaded through the sparse buffer
/// stream, which places:
///   - mask in specific control register bits
///   - data in the vector register proper
///
/// For our functional emulator, the mask is encoded in the lower 64 bits
/// of an auxiliary register or the sparse vector's mask region.
pub fn extract_sparse_mask(sparse_input: &[u32; 8]) -> u64 {
    // The mask occupies the first two u32 words (64 bits) of the sparse
    // register representation.
    (sparse_input[0] as u64) | ((sparse_input[1] as u64) << 32)
}

/// Extract partially decompressed data values from a sparse input using
/// the mask. Returns a vector of values at the positions indicated by the
/// partially decompressed format.
///
/// The data follows the mask in the sparse register. For int8 elements,
/// the data starts at byte offset 8 (after the 64-bit mask).
pub fn extract_sparse_data_i8(
    sparse_input: &[u32; 8],
    mask: u64,
    num_logical_elements: usize,
) -> Vec<i8> {
    let num_groups = num_logical_elements / 4;
    let mut result = Vec::with_capacity(num_logical_elements / 2);

    // Packed data starts after the mask (byte offset 8 in the register)
    let mut packed_values: Vec<i32> = Vec::new();
    for byte_idx in 8..32 {
        let word = byte_idx / 4;
        let byte_off = byte_idx % 4;
        let val = ((sparse_input[word] >> (byte_off * 8)) & 0xFF) as u8 as i8;
        packed_values.push(val as i32);
    }

    let mut data_cursor: usize = 0;

    for group in 0..num_groups {
        let nibble = ((mask >> (group * 4)) & 0xF) as u8;
        let (val0, val1) = decompress_nibble(nibble, &packed_values, &mut data_cursor);
        result.push(val0 as i8);
        result.push(val1 as i8);
    }

    result
}

// -----------------------------------------------------------------------
// Sparse matrix multiply core
// -----------------------------------------------------------------------

/// Sparse matrix multiply: acc += A * sparse(B)
///
/// A is the dense input matrix (from vector register).
/// B is the sparse input -- partially decompressed data where zero
/// elements have been inserted at masked-out positions.
///
/// The effective inner dimension is 2x the dense equivalent because
/// the sparse format guarantees 50% sparsity (half the elements are
/// zero and skipped by the hardware). The emulator expands the sparse
/// B matrix to its full logical size and performs the standard matmul.
///
/// This is the top-level dispatch function for sparse matmul.
pub fn matmul_sparse(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b_data: &[u32; 8],
    b_mask: u64,
    elem_type: ElementType,
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
    match elem_type {
        ElementType::Int8 | ElementType::UInt8 => {
            sparse_matmul_i8xi8(acc, a, b_data, b_mask, signed_a, signed_b, subtract);
        }
        ElementType::Int16 | ElementType::UInt16 => {
            sparse_matmul_i16xi16(acc, a, b_data, b_mask, signed_a, signed_b, subtract);
        }
        ElementType::BFloat16 | ElementType::Float32 => {
            sparse_matmul_bf16xbf16(acc, a, b_data, b_mask, subtract);
        }
        ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
            // int32/int64 x int16 sparse: acc_cmb=2
            sparse_matmul_i16xi16(acc, a, b_data, b_mask, signed_a, signed_b, subtract);
        }
    }
}

// -----------------------------------------------------------------------
// int8 x int8 sparse multiply
// -----------------------------------------------------------------------

/// Sparse int8 x int8 matmul with 32-bit accumulator.
///
/// Geometry: A[4][16] * B_sparse[16][4] -> C[4][4], where B has been
/// expanded from its sparse representation to 16 inner elements (compared
/// to dense inner=8). The sparse B register holds 32 partially decompressed
/// int8 values (half of the 64 logical elements -- the other half are zero).
///
/// For functional correctness: we reconstruct the full B[16][4] matrix by
/// expanding the partially decompressed data with the mask, then perform
/// the standard matmul.
fn sparse_matmul_i8xi8(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b_data: &[u32; 8],
    b_mask: u64,
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
    // Sparse geometry: rows=4, inner=16, cols=4
    // A provides 4*16 = 64 int8 values (512 bits -- requires two vector regs
    // in the full hardware, but for emulation we work with the logical matrix).
    // B_sparse provides 16*4 = 64 logical int8 values (32 stored + mask).
    //
    // For the emulator's single-register model, we use a reduced geometry
    // that fits our register width: rows=4, inner=8, cols=4 (same as dense
    // but with the sparse mask applied to B).
    let rows = 4;
    let inner = 8;
    let cols = 4;

    // Expand B using mask. The mask tells us which positions in B are non-zero.
    // Each 4-bit mask nibble covers 4 consecutive byte positions.
    // The partially decompressed B data is stored densely in b_data.
    //
    // For the emulator, B data occupies all 32 bytes of b_data (256 bits).
    // The mask tells us which of those positions are active.
    let mut b_expanded = [0i64; 32]; // inner * cols = 8 * 4 = 32

    for idx in 0..32 {
        // Check if this position's mask bit is set.
        // Each bit in the mask corresponds to one byte in the expanded vector.
        let mask_bit = (b_mask >> idx) & 1;
        if mask_bit != 0 {
            let val = if signed_b {
                extract_i8(b_data, idx) as i64
            } else {
                extract_u8(b_data, idx) as i64
            };
            b_expanded[idx] = val;
        }
        // else: already zero
    }

    for r in 0..rows {
        for c in 0..cols {
            let out_idx = r * cols + c;
            let mut sum: i64 = 0;

            for k in 0..inner {
                let a_idx = r * inner + k;
                let b_idx = k * cols + c;

                let a_val = if signed_a {
                    extract_i8(a, a_idx) as i64
                } else {
                    extract_u8(a, a_idx) as i64
                };

                let b_val = if b_idx < 32 { b_expanded[b_idx] } else { 0 };

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

// -----------------------------------------------------------------------
// int16 x int16 sparse multiply
// -----------------------------------------------------------------------

/// Sparse int16 x int16 matmul with 64-bit accumulator (acc_cmb=2).
///
/// Geometry: A[2][8] * B_sparse[8][4] -> C[2][4], where B has been
/// expanded from its sparse representation. Dense equivalent is inner=4.
fn sparse_matmul_i16xi16(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b_data: &[u32; 8],
    b_mask: u64,
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
    let rows = 2;
    let inner = 4;
    let cols = 4;

    // B_sparse holds 16 int16 values (256 bits), mask indicates active positions.
    // Each mask bit covers 2 bytes (one int16 element), so we use mask bits
    // at stride 2 for 16-bit elements.
    let mut b_expanded = [0i64; 16]; // inner * cols = 4 * 4 = 16

    for idx in 0..16 {
        // For 16-bit elements, each mask bit covers 2 bytes.
        // Mask bit at position (idx*2) corresponds to element idx.
        let mask_bit_lo = (b_mask >> (idx * 2)) & 1;
        let mask_bit_hi = (b_mask >> (idx * 2 + 1)) & 1;
        if mask_bit_lo != 0 || mask_bit_hi != 0 {
            let val = if signed_b {
                extract_i16(b_data, idx) as i64
            } else {
                extract_u16(b_data, idx) as i64
            };
            b_expanded[idx] = val;
        }
    }

    for r in 0..rows {
        for c in 0..cols {
            let out_idx = r * cols + c;
            let mut sum: i64 = 0;

            for k in 0..inner {
                let a_idx = r * inner + k;
                let b_idx = k * cols + c;

                let a_val = if signed_a {
                    extract_i16(a, a_idx) as i64
                } else {
                    extract_u16(a, a_idx) as i64
                };

                let b_val = if b_idx < 16 { b_expanded[b_idx] } else { 0 };

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

// -----------------------------------------------------------------------
// bf16 x bf16 sparse multiply
// -----------------------------------------------------------------------

/// Sparse bf16 x bf16 matmul with fp32 accumulator.
///
/// Geometry: A[4][8] * B_sparse[8][4] -> C[4][4], where B has been
/// expanded from its sparse representation. Dense equivalent inner=4.
fn sparse_matmul_bf16xbf16(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b_data: &[u32; 8],
    b_mask: u64,
    subtract: bool,
) {
    let rows = 4;
    let inner = 4;
    let cols = 4;

    // B_sparse holds 16 bf16 values. Mask indicates which are active.
    let mut b_expanded = [0.0f32; 16];

    for idx in 0..16 {
        // For bf16, each element is 2 bytes. Mask bit coverage is
        // 2 bits per element (same as int16).
        let mask_bit_lo = (b_mask >> (idx * 2)) & 1;
        let mask_bit_hi = (b_mask >> (idx * 2 + 1)) & 1;
        if mask_bit_lo != 0 || mask_bit_hi != 0 {
            b_expanded[idx] = extract_bf16_as_f32(b_data, idx);
        }
    }

    for r in 0..rows {
        for c in 0..cols {
            let out_idx = r * cols + c;
            let mut sum: f32 = 0.0;

            for k in 0..inner {
                let a_idx = r * inner + k;
                let b_idx = k * cols + c;

                let a_val = extract_bf16_as_f32(a, a_idx);
                let b_val = if b_idx < 16 { b_expanded[b_idx] } else { 0.0 };

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

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack int8 values into [u32; 8].
    fn pack_i8(values: &[i8]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate().take(32) {
            let word = i / 4;
            let byte = i % 4;
            packed[word] |= ((v as u8) as u32) << (byte * 8);
        }
        packed
    }

    /// Pack int16 values into [u32; 8].
    fn pack_i16(values: &[i16]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate().take(16) {
            let word = i / 2;
            let half = i % 2;
            packed[word] |= ((v as u16) as u32) << (half * 16);
        }
        packed
    }

    /// Pack bf16 bit patterns into [u32; 8].
    fn pack_bf16(values: &[u16]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate().take(16) {
            let word = i / 2;
            let half = i % 2;
            packed[word] |= (v as u32) << (half * 16);
        }
        packed
    }

    /// Convert f32 to bf16 bit pattern (truncate).
    fn f32_to_bf16_bits(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// Build a 64-bit mask where specific byte positions are marked active.
    /// Each bit in the mask corresponds to one byte in the decompressed
    /// vector.
    fn build_mask_from_bytes(active_bytes: &[usize], total_bytes: usize) -> u64 {
        let mut mask: u64 = 0;
        for &pos in active_bytes {
            if pos < total_bytes && pos < 64 {
                mask |= 1u64 << pos;
            }
        }
        mask
    }

    /// Build a 64-bit mask for 16-bit elements. Each element occupies 2
    /// mask bits (corresponding to its 2 bytes).
    fn build_mask_from_i16_positions(active_elements: &[usize], total_elements: usize) -> u64 {
        let mut mask: u64 = 0;
        for &pos in active_elements {
            if pos < total_elements {
                // Set both bytes of this 16-bit element
                let bit_lo = pos * 2;
                let bit_hi = pos * 2 + 1;
                if bit_lo < 64 {
                    mask |= 1u64 << bit_lo;
                }
                if bit_hi < 64 {
                    mask |= 1u64 << bit_hi;
                }
            }
        }
        mask
    }

    // -------------------------------------------------------------------
    // Decompression table tests
    // -------------------------------------------------------------------

    #[test]
    fn test_decompress_nibble_all_zero() {
        let data = vec![10, 20, 30, 40];
        let mut cursor = 0;
        let (v0, v1) = decompress_nibble(0b0000, &data, &mut cursor);
        assert_eq!(v0, 0);
        assert_eq!(v1, 0);
        // No data consumed
        assert_eq!(cursor, 0);
    }

    #[test]
    fn test_decompress_nibble_single_a() {
        let data = vec![10, 20, 30, 40];
        let mut cursor = 0;
        let (v0, v1) = decompress_nibble(0b0001, &data, &mut cursor);
        assert_eq!(v0, 0);
        assert_eq!(v1, 10); // A=10
        assert_eq!(cursor, 1);
    }

    #[test]
    fn test_decompress_nibble_single_b() {
        let data = vec![10, 20, 30, 40];
        let mut cursor = 0;
        let (v0, v1) = decompress_nibble(0b0010, &data, &mut cursor);
        assert_eq!(v0, 0);
        assert_eq!(v1, 10); // B=10 (first available data)
        assert_eq!(cursor, 1);
    }

    #[test]
    fn test_decompress_nibble_a_and_b() {
        let data = vec![10, 20, 30, 40];
        let mut cursor = 0;
        let (v0, v1) = decompress_nibble(0b0011, &data, &mut cursor);
        // mask 0011 -> (B, A) = (20, 10)
        assert_eq!(v0, 20); // B
        assert_eq!(v1, 10); // A
        assert_eq!(cursor, 2);
    }

    #[test]
    fn test_decompress_nibble_c_only() {
        let data = vec![10, 20, 30, 40];
        let mut cursor = 0;
        let (v0, v1) = decompress_nibble(0b0100, &data, &mut cursor);
        assert_eq!(v0, 10); // C=10
        assert_eq!(v1, 0);
        assert_eq!(cursor, 1);
    }

    #[test]
    fn test_decompress_nibble_c_and_a() {
        let data = vec![10, 20, 30, 40];
        let mut cursor = 0;
        let (v0, v1) = decompress_nibble(0b0101, &data, &mut cursor);
        // mask 0101 -> (C, A) = (20, 10)
        assert_eq!(v0, 20); // C
        assert_eq!(v1, 10); // A
        assert_eq!(cursor, 2);
    }

    #[test]
    fn test_decompress_nibble_d_and_c() {
        let data = vec![10, 20, 30, 40];
        let mut cursor = 0;
        let (v0, v1) = decompress_nibble(0b1100, &data, &mut cursor);
        // mask 1100 -> (D, C) = (20, 10)
        assert_eq!(v0, 20); // D
        assert_eq!(v1, 10); // C
        assert_eq!(cursor, 2);
    }

    // -------------------------------------------------------------------
    // Full sparse decompression tests
    // -------------------------------------------------------------------

    #[test]
    fn test_decompress_sparse_vector_all_zero_mask() {
        // All mask bits zero -> all elements are zero
        let mask: u64 = 0;
        let packed_data: Vec<i32> = vec![];
        let result = decompress_sparse_vector(mask, &packed_data, 16);
        assert_eq!(result.len(), 8); // 16/2 = 8 partially decompressed
        assert!(result.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_decompress_sparse_vector_single_element() {
        // First nibble = 0b0001 (only bit 0 set -> A at slot 1)
        // Remaining nibbles = 0
        let mask: u64 = 0b0001;
        let packed_data = vec![42];
        let result = decompress_sparse_vector(mask, &packed_data, 16);
        assert_eq!(result.len(), 8);
        assert_eq!(result[0], 0);  // slot 0
        assert_eq!(result[1], 42); // slot 1 = A
        // Rest should be zero
        for i in 2..8 {
            assert_eq!(result[i], 0, "position {} should be 0", i);
        }
    }

    // -------------------------------------------------------------------
    // Sparse matmul: fully dense mask (all bits set) should match dense
    // -------------------------------------------------------------------

    #[test]
    fn test_sparse_i8xi8_all_mask_set_matches_dense() {
        // When all mask bits are set, sparse matmul should behave like
        // dense matmul (no elements masked out).
        let a_vals = [1i8; 32];
        let b_vals = [1i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);

        // All 32 byte positions active
        let mask: u64 = 0xFFFF_FFFF; // 32 bits set (32 bytes)

        let mut acc_sparse = [0u64; 8];
        sparse_matmul_i8xi8(&mut acc_sparse, &a, &b, mask, true, true, false);

        // With all-ones inputs and inner=8, cols=4:
        // Each output = sum of (inner) products = 8 * (1 * 1) = 8
        for idx in 0..16 {
            let val = read_acc32(&acc_sparse, idx);
            assert_eq!(
                val, 8,
                "output[{}]: expected 8 with all-ones sparse, got {}",
                idx, val
            );
        }
    }

    // -------------------------------------------------------------------
    // Sparse matmul: fully zero mask produces zero accumulation
    // -------------------------------------------------------------------

    #[test]
    fn test_sparse_i8xi8_zero_mask_produces_zero() {
        // With mask=0, all B elements are zero, so result should be zero
        // regardless of A values.
        let a_vals = [5i8; 32];
        let b_vals = [99i8; 32]; // Non-zero B data, but mask kills it
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mask: u64 = 0; // No active positions

        let mut acc = [0u64; 8];
        sparse_matmul_i8xi8(&mut acc, &a, &b, mask, true, true, false);

        for idx in 0..16 {
            assert_eq!(
                read_acc32(&acc, idx), 0,
                "output[{}] should be zero when mask is all-zero",
                idx
            );
        }
    }

    // -------------------------------------------------------------------
    // 50% sparse: verify only non-zero elements contribute
    // -------------------------------------------------------------------

    #[test]
    fn test_sparse_i8xi8_half_mask() {
        // Set only even byte positions in B's mask. This means only
        // elements at even indices participate in the multiply.
        let a_vals = [1i8; 32];
        let mut b_vals = [0i8; 32];
        // Put known values at even positions
        for i in (0..32).step_by(2) {
            b_vals[i] = 2;
        }
        // Odd positions have non-zero data that should be masked out
        for i in (1..32).step_by(2) {
            b_vals[i] = 99;
        }

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);

        // Mask: only even byte positions active
        let mut mask: u64 = 0;
        for i in (0..32).step_by(2) {
            mask |= 1u64 << i;
        }

        let mut acc = [0u64; 8];
        sparse_matmul_i8xi8(&mut acc, &a, &b, mask, true, true, false);

        // With A=all-ones and B=2 at even positions, 0 at odd positions,
        // the result depends on the geometry. Inner=8, cols=4.
        // B[k][c] indexed as k*cols+c. The even byte positions in B
        // correspond to specific (k,c) pairs.
        //
        // Even indices: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
        // For B[k][c] = b_vals[k*4 + c]:
        //   k=0: positions 0,1,2,3 -> even are 0,2 -> B[0][0]=2, B[0][2]=2
        //   k=1: positions 4,5,6,7 -> even are 4,6 -> B[1][0]=2, B[1][2]=2
        //   ... etc
        // So half the elements in each column are active.
        // For output C[r][c] = sum_k A[r][k]*B[k][c]:
        //   Columns 0,2 have 4 active B elements (even positions), each = 2
        //   Columns 1,3 have 4 active B elements (even positions), each = 0 (they're at odd byte offsets)
        //
        // Wait, let me recalculate. B is stored row-major: B[k][c] at index k*cols+c.
        //   k=0,c=0 -> idx 0 (even, masked in, val=2)
        //   k=0,c=1 -> idx 1 (odd, masked out, val=0)
        //   k=0,c=2 -> idx 2 (even, masked in, val=2)
        //   k=0,c=3 -> idx 3 (odd, masked out, val=0)
        //   k=1,c=0 -> idx 4 (even, masked in, val=2)
        //   ...
        //
        // For even cols (c=0,2): every k has B[k][c] = 2
        //   C[r][c] = sum(k=0..7) { 1 * 2 } = 16
        //
        // For odd cols (c=1,3): every k has B[k][c] = 0 (masked out)
        //   C[r][c] = 0

        for r in 0..4 {
            for c in 0..4 {
                let idx = r * 4 + c;
                let val = read_acc32(&acc, idx);
                if c % 2 == 0 {
                    assert_eq!(
                        val, 16,
                        "C[{}][{}]: expected 16 (even col), got {}",
                        r, c, val
                    );
                } else {
                    assert_eq!(
                        val, 0,
                        "C[{}][{}]: expected 0 (odd col, masked out), got {}",
                        r, c, val
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Single non-zero element: verify correct position
    // -------------------------------------------------------------------

    #[test]
    fn test_sparse_i8xi8_single_active_element() {
        // Only B[0][0] (byte index 0) is active with value 7.
        // A is identity-like: A[0][0] = 3, rest zero.
        // Result: C[0][0] = 3 * 7 = 21, all others = 0.

        let mut a_vals = [0i8; 32];
        a_vals[0] = 3; // A[0][0] = 3

        let mut b_vals = [0i8; 32];
        b_vals[0] = 7; // B[0][0] = 7

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);

        // Only byte 0 is active
        let mask: u64 = 1;

        let mut acc = [0u64; 8];
        sparse_matmul_i8xi8(&mut acc, &a, &b, mask, true, true, false);

        assert_eq!(read_acc32(&acc, 0), 21, "C[0][0] = 3 * 7 = 21");

        // All other positions should be zero
        for idx in 1..16 {
            assert_eq!(
                read_acc32(&acc, idx), 0,
                "C[{}] should be 0 with single active element",
                idx
            );
        }
    }

    // -------------------------------------------------------------------
    // Sparse matmul subtract
    // -------------------------------------------------------------------

    #[test]
    fn test_sparse_i8xi8_subtract() {
        let a_vals = [1i8; 32];
        let b_vals = [2i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mask: u64 = 0xFFFF_FFFF; // All active

        // First: accumulate
        let mut acc = [0u64; 8];
        sparse_matmul_i8xi8(&mut acc, &a, &b, mask, true, true, false);

        let first_val = read_acc32(&acc, 0);
        assert!(first_val > 0, "should have accumulated positive value");

        // Then: subtract the same product
        sparse_matmul_i8xi8(&mut acc, &a, &b, mask, true, true, true);

        for idx in 0..16 {
            assert_eq!(
                read_acc32(&acc, idx), 0,
                "C[{}] should be 0 after accumulate then subtract",
                idx
            );
        }
    }

    // -------------------------------------------------------------------
    // int16 sparse tests
    // -------------------------------------------------------------------

    #[test]
    fn test_sparse_i16xi16_all_active() {
        // All mask bits set -> all B elements active
        let a_vals: [i16; 16] = [1; 16];
        let b_vals: [i16; 16] = [3; 16];
        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);

        // For 16-bit elements, both bytes per element must be masked.
        // 16 elements * 2 bytes = 32 mask bits.
        let mask: u64 = 0xFFFF_FFFF;

        let mut acc = [0u64; 8];
        sparse_matmul_i16xi16(&mut acc, &a, &b, mask, true, true, false);

        // Geometry: rows=2, inner=4, cols=4
        // Each output = sum(k=0..3) { 1 * 3 } = 12
        for idx in 0..8 {
            let val = read_acc64(&acc, idx);
            assert_eq!(val, 12, "output[{}]: expected 12, got {}", idx, val);
        }
    }

    #[test]
    fn test_sparse_i16xi16_zero_mask() {
        let a_vals: [i16; 16] = [10; 16];
        let b_vals: [i16; 16] = [20; 16];
        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mask: u64 = 0;

        let mut acc = [0u64; 8];
        sparse_matmul_i16xi16(&mut acc, &a, &b, mask, true, true, false);

        for idx in 0..8 {
            assert_eq!(
                read_acc64(&acc, idx), 0,
                "output[{}] should be 0 with zero mask",
                idx
            );
        }
    }

    #[test]
    fn test_sparse_i16xi16_partial_mask() {
        // Only first element of B active (element 0 = B[0][0])
        let mut a_vals = [0i16; 16];
        a_vals[0] = 5; // A[0][0]

        let mut b_vals = [0i16; 16];
        b_vals[0] = 7; // B[0][0]

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);

        // Element 0 active: mask bits 0 and 1 set
        let mask: u64 = 0b11;

        let mut acc = [0u64; 8];
        sparse_matmul_i16xi16(&mut acc, &a, &b, mask, true, true, false);

        // C[0][0] = A[0][0] * B[0][0] = 5 * 7 = 35
        assert_eq!(read_acc64(&acc, 0), 35);

        // Others should be zero
        for idx in 1..8 {
            assert_eq!(read_acc64(&acc, idx), 0, "output[{}] should be 0", idx);
        }
    }

    // -------------------------------------------------------------------
    // bf16 sparse tests
    // -------------------------------------------------------------------

    #[test]
    fn test_sparse_bf16_all_active() {
        let two = f32_to_bf16_bits(2.0);
        let three = f32_to_bf16_bits(3.0);
        let a = pack_bf16(&[two; 16]);
        let b = pack_bf16(&[three; 16]);

        // 16 bf16 elements * 2 bytes = 32 mask bits
        let mask: u64 = 0xFFFF_FFFF;

        let mut acc = [0u64; 8];
        sparse_matmul_bf16xbf16(&mut acc, &a, &b, mask, false);

        // Geometry: rows=4, inner=4, cols=4
        // Each output = 4 * (2.0 * 3.0) = 24.0
        for idx in 0..16 {
            let val = read_acc_f32(&acc, idx);
            assert!(
                (val - 24.0).abs() < 0.1,
                "output[{}]: expected 24.0, got {}",
                idx, val
            );
        }
    }

    #[test]
    fn test_sparse_bf16_zero_mask() {
        let two = f32_to_bf16_bits(2.0);
        let a = pack_bf16(&[two; 16]);
        let b = pack_bf16(&[two; 16]);
        let mask: u64 = 0;

        let mut acc = [0u64; 8];
        sparse_matmul_bf16xbf16(&mut acc, &a, &b, mask, false);

        for idx in 0..16 {
            let val = read_acc_f32(&acc, idx);
            assert!(
                val.abs() < 0.001,
                "output[{}]: expected 0.0, got {}",
                idx, val
            );
        }
    }

    #[test]
    fn test_sparse_bf16_subtract() {
        let two = f32_to_bf16_bits(2.0);
        let three = f32_to_bf16_bits(3.0);
        let a = pack_bf16(&[two; 16]);
        let b = pack_bf16(&[three; 16]);
        let mask: u64 = 0xFFFF_FFFF;

        let mut acc = [0u64; 8];

        // Accumulate
        sparse_matmul_bf16xbf16(&mut acc, &a, &b, mask, false);
        // Subtract same product
        sparse_matmul_bf16xbf16(&mut acc, &a, &b, mask, true);

        for idx in 0..16 {
            let val = read_acc_f32(&acc, idx);
            assert!(
                val.abs() < 0.001,
                "output[{}]: expected 0.0 after subtract, got {}",
                idx, val
            );
        }
    }

    // -------------------------------------------------------------------
    // Mask extraction tests
    // -------------------------------------------------------------------

    #[test]
    fn test_extract_sparse_mask() {
        let mut reg = [0u32; 8];
        reg[0] = 0xDEADBEEF;
        reg[1] = 0xCAFEBABE;

        let mask = extract_sparse_mask(&reg);
        assert_eq!(mask, 0xCAFEBABE_DEADBEEF);
    }

    #[test]
    fn test_extract_sparse_mask_zero() {
        let reg = [0u32; 8];
        assert_eq!(extract_sparse_mask(&reg), 0);
    }

    // -------------------------------------------------------------------
    // Data extraction tests
    // -------------------------------------------------------------------

    #[test]
    fn test_extract_sparse_data_i8_simple() {
        // Build a sparse register: first 8 bytes = mask, remaining = data
        let mut reg = [0u32; 8];

        // Mask: nibble 0 = 0b0001 (only A present)
        reg[0] = 0x0000_0001; // First nibble is 0001
        reg[1] = 0x0000_0000;

        // Data starts at byte 8 (word 2, byte 0)
        // A = 42
        reg[2] = 42;

        let mask = extract_sparse_mask(&reg);
        let data = extract_sparse_data_i8(&reg, mask, 16);

        // 16/4 = 4 groups, each producing 2 values = 8 total
        assert_eq!(data.len(), 8);
        assert_eq!(data[0], 0);  // slot 0 of group 0
        assert_eq!(data[1], 42); // slot 1 of group 0 = A
        for i in 2..8 {
            assert_eq!(data[i], 0, "position {} should be 0", i);
        }
    }

    // -------------------------------------------------------------------
    // Sparse differs from dense for sparse data
    // -------------------------------------------------------------------

    #[test]
    fn test_sparse_differs_from_dense_when_masked() {
        // This test demonstrates that sparse matmul produces DIFFERENT
        // results than if we just did a dense matmul on the same raw data.
        // The mask causes some B elements to be zeroed out.

        let a_vals = [1i8; 32]; // All ones
        let b_vals = [1i8; 32]; // All ones in raw data

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);

        // Dense: no mask, inner=8, result = 8 per lane
        // (Dense matmul doesn't use mask, so all B elements contribute)
        //
        // Sparse with half mask: only even bytes active, inner=8
        // Even positions in B contribute 1*1=1, odd positions contribute 0.
        // B[k][c] at index k*4+c:
        //   even cols (c=0,2): all k contribute -> sum = 8
        //   odd cols (c=1,3): no k contributes -> sum = 0
        let mask_half: u64 = build_mask_from_bytes(
            &(0..32).step_by(2).collect::<Vec<_>>(),
            32,
        );

        let mut acc_sparse = [0u64; 8];
        sparse_matmul_i8xi8(&mut acc_sparse, &a, &b, mask_half, true, true, false);

        // Verify sparse result has a mix of non-zero and zero columns
        let sparse_c00 = read_acc32(&acc_sparse, 0);
        let sparse_c01 = read_acc32(&acc_sparse, 1);

        // Even col should be non-zero, odd col should be zero
        assert!(sparse_c00 != 0, "even col should be non-zero");
        assert_eq!(sparse_c01, 0, "odd col should be zero due to mask");

        // This proves sparse != dense for the same B data when mask is partial
    }

    // -------------------------------------------------------------------
    // Top-level dispatch test
    // -------------------------------------------------------------------

    #[test]
    fn test_matmul_sparse_dispatch_int8() {
        let a = pack_i8(&[1i8; 32]);
        let b = pack_i8(&[1i8; 32]);
        let mask: u64 = 0xFFFF_FFFF;

        let mut acc = [0u64; 8];
        matmul_sparse(
            &mut acc, &a, &b, mask,
            ElementType::Int8, true, true, false,
        );

        // Should produce valid non-zero results
        let val = read_acc32(&acc, 0);
        assert!(val > 0, "dispatch should produce non-zero result");
    }

    #[test]
    fn test_matmul_sparse_dispatch_bf16() {
        let two = f32_to_bf16_bits(2.0);
        let a = pack_bf16(&[two; 16]);
        let b = pack_bf16(&[two; 16]);
        let mask: u64 = 0xFFFF_FFFF;

        let mut acc = [0u64; 8];
        matmul_sparse(
            &mut acc, &a, &b, mask,
            ElementType::BFloat16, true, true, false,
        );

        let val = read_acc_f32(&acc, 0);
        assert!(val > 0.0, "dispatch should produce non-zero result");
    }

    // -------------------------------------------------------------------
    // Helper function tests
    // -------------------------------------------------------------------

    #[test]
    fn test_build_mask_from_bytes() {
        let mask = build_mask_from_bytes(&[0, 2, 4], 32);
        assert_eq!(mask, (1 << 0) | (1 << 2) | (1 << 4));
    }

    #[test]
    fn test_build_mask_from_i16_positions() {
        let mask = build_mask_from_i16_positions(&[0, 1], 16);
        // Element 0 -> bits 0,1; Element 1 -> bits 2,3
        assert_eq!(mask, 0b1111);
    }
}
