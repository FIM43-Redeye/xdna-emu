//! Miscellaneous operations for the vector ALU.
//!
//! Extracted from vector.rs -- shuffle, broadcast, extract, insert,
//! align, bitwise, and mask expansion functions.

use crate::interpreter::bundle::{ElementType, ShufflePattern};

use super::vector::VectorAlu;

impl VectorAlu {
    /// Vector shuffle with pattern.
    #[allow(dead_code)]
    pub(super) fn vector_shuffle(src: &[u32; 8], pattern: ShufflePattern) -> [u32; 8] {
        match pattern {
            ShufflePattern::Identity => *src,

            ShufflePattern::Reverse => {
                let mut result = [0u32; 8];
                for i in 0..8 {
                    result[i] = src[7 - i];
                }
                result
            }

            ShufflePattern::Broadcast(lane) => {
                let val = src[(lane & 0x07) as usize];
                [val; 8]
            }

            ShufflePattern::InterleaveLow => {
                // Interleave low halves of two conceptual vectors
                // Here we just shuffle within single vector
                let mut result = [0u32; 8];
                for i in 0..4 {
                    result[i * 2] = src[i];
                    result[i * 2 + 1] = src[i + 4];
                }
                result
            }

            ShufflePattern::InterleaveHigh => {
                // Interleave high halves
                let mut result = [0u32; 8];
                for i in 0..4 {
                    result[i * 2] = src[i + 4];
                    result[i * 2 + 1] = src[i];
                }
                result
            }

            ShufflePattern::Custom(mask) => {
                // Each 3-bit field selects a source lane
                let mut result = [0u32; 8];
                for i in 0..8 {
                    let lane_sel = ((mask >> (i * 3)) & 0x7) as usize;
                    result[i] = src[lane_sel];
                }
                result
            }
        }
    }

    /// Broadcast a scalar value to all vector lanes.
    pub(super) fn vector_broadcast(value: u32, elem_type: ElementType) -> [u32; 8] {
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
                // Broadcast 32-bit value to all 8 lanes
                [value; 8]
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // Broadcast 16-bit value to all 16 lanes (replicate in each u32)
                let val16 = value & 0xFFFF;
                let packed = val16 | (val16 << 16);
                [packed; 8]
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // Broadcast 8-bit value to all 32 lanes (replicate in each u32)
                let val8 = value & 0xFF;
                let packed = val8 | (val8 << 8) | (val8 << 16) | (val8 << 24);
                [packed; 8]
            }
        }
    }

    /// Extract a single element from a vector.
    ///
    /// Returns the element at the given lane index, converted to a u32.
    pub(super) fn vector_extract(src: &[u32; 8], index: u32, elem_type: ElementType) -> u32 {
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
                // 8 lanes of 32-bit elements
                let lane = (index as usize) & 0x7;
                src[lane]
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 lanes of 16-bit elements (2 per u32)
                let word_idx = ((index as usize) >> 1) & 0x7;
                let sub_idx = (index as usize) & 0x1;
                let value = (src[word_idx] >> (sub_idx * 16)) & 0xFFFF;
                // Sign-extend for signed types
                if matches!(elem_type, ElementType::Int16) {
                    value as i16 as i32 as u32
                } else {
                    value
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 lanes of 8-bit elements (4 per u32)
                let word_idx = ((index as usize) >> 2) & 0x7;
                let sub_idx = (index as usize) & 0x3;
                let value = (src[word_idx] >> (sub_idx * 8)) & 0xFF;
                // Sign-extend for signed types
                if matches!(elem_type, ElementType::Int8) {
                    value as i8 as i32 as u32
                } else {
                    value
                }
            }
        }
    }

    /// Extract a single element from a 256-bit vector by element index.
    ///
    /// Returns the element value (zero-extended to u32).
    pub(super) fn extract_element_by_index(src: &[u32; 8], index: u32, et: ElementType) -> u32 {
        match et {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
                let idx = (index as usize) & 7;
                src[idx]
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 elements of 16-bit each in 256 bits
                let idx = (index as usize) & 15;
                let word = idx / 2;
                let half = idx % 2;
                (src[word] >> (half * 16)) & 0xFFFF
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 elements of 8-bit each in 256 bits
                let idx = (index as usize) & 31;
                let word = idx / 4;
                let byte_in_word = idx % 4;
                (src[word] >> (byte_in_word * 8)) & 0xFF
            }
        }
    }

    pub(super) fn vector_insert(dst: &mut [u32; 8], value: u32, index: u32, elem_type: ElementType) {
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
                // 8 lanes of 32-bit elements
                let lane = (index as usize) & 0x7;
                dst[lane] = value;
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 lanes of 16-bit elements (2 per u32)
                let word_idx = ((index as usize) >> 1) & 0x7;
                let sub_idx = (index as usize) & 0x1;
                let shift = sub_idx * 16;
                let mask = !(0xFFFFu32 << shift);
                dst[word_idx] = (dst[word_idx] & mask) | ((value & 0xFFFF) << shift);
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 lanes of 8-bit elements (4 per u32)
                let word_idx = ((index as usize) >> 2) & 0x7;
                let sub_idx = (index as usize) & 0x3;
                let shift = sub_idx * 8;
                let mask = !(0xFFu32 << shift);
                dst[word_idx] = (dst[word_idx] & mask) | ((value & 0xFF) << shift);
            }
        }
    }

    /// Vector align: concatenates two 256-bit vectors and extracts 256 bits at byte offset.
    /// Result = (src1 || src2) >> (byte_shift * 8), extracting lower 256 bits.
    pub(super) fn vector_align(src1: &[u32; 8], src2: &[u32; 8], byte_shift: u32) -> [u32; 8] {
        // Treat as 64 bytes (512 bits total), shift right by byte_shift bytes
        // and take lower 32 bytes (256 bits)
        let shift = (byte_shift & 0x3F) as usize; // Max 63 bytes
        let mut result = [0u32; 8];

        // Build concatenated 64-byte array: [src2 || src1] (src2 is high, src1 is low)
        // Then shift right and take lower 32 bytes
        for i in 0..8 {
            let byte_idx = i * 4 + shift;

            // Get value from concatenated vector
            let get_byte = |idx: usize| -> u8 {
                let w = idx / 4;
                let b = idx % 4;
                if w < 8 {
                    ((src1[w] >> (b * 8)) & 0xFF) as u8
                } else if w < 16 {
                    ((src2[w - 8] >> (b * 8)) & 0xFF) as u8
                } else {
                    0
                }
            };

            let b0 = get_byte(byte_idx) as u32;
            let b1 = get_byte(byte_idx + 1) as u32;
            let b2 = get_byte(byte_idx + 2) as u32;
            let b3 = get_byte(byte_idx + 3) as u32;
            result[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        result
    }

    /// Vector bitwise AND: dst = a & b
    pub(super) fn vector_bitwise_and(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = a[i] & b[i];
        }
        result
    }

    /// Vector bitwise OR: dst = a | b
    pub(super) fn vector_bitwise_or(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = a[i] | b[i];
        }
        result
    }

    /// Vector bitwise XOR: dst = a ^ b
    pub(super) fn vector_bitwise_xor(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = a[i] ^ b[i];
        }
        result
    }

    /// Vector bitwise NOT: dst = ~a
    pub(super) fn vector_bitwise_not(a: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = !a[i];
        }
        result
    }

    /// Expand a scalar select mask to a per-lane vector mask.
    ///
    /// VSEL uses a scalar register where each bit selects the corresponding
    /// element. For 32-bit mode, bits 0-7 select 8 elements. For 16-bit,
    /// bits 0-15 select 16 elements (2 per u32 lane). For 8-bit, bits 0-31
    /// select 32 elements (4 per u32 lane).
    pub(super) fn expand_select_mask(sel: u32, elem_type: ElementType) -> [u32; 8] {
        let mut mask = [0u32; 8];
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
                // 8 elements, 1 bit each
                for i in 0..8 {
                    mask[i] = if (sel >> i) & 1 != 0 { 1 } else { 0 };
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 elements (2 per u32), 1 bit each
                for i in 0..8 {
                    let lo = if (sel >> (i * 2)) & 1 != 0 { 0xFFFF } else { 0 };
                    let hi = if (sel >> (i * 2 + 1)) & 1 != 0 { 0xFFFF } else { 0 };
                    mask[i] = lo | (hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 elements (4 per u32), 1 bit each
                for i in 0..8 {
                    let mut m = 0u32;
                    for j in 0..4 {
                        if (sel >> (i * 4 + j)) & 1 != 0 {
                            m |= 0xFF << (j * 8);
                        }
                    }
                    mask[i] = m;
                }
            }
        }
        mask
    }
}
