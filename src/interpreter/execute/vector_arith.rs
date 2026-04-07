//! Arithmetic operations for the vector ALU.
//!
//! Extracted from vector.rs -- these are pure computational functions
//! with no internal dependencies on dispatch or helper code.

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::tablegen::SemanticOp;

use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    pub(super) fn vector_addsub(
        s1: &[u32; 8],
        s2: &[u32; 8],
        sel: &[u32; 8],
        elem_type: ElementType,
    ) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
                for i in 0..8 {
                    if sel[i] & 1 != 0 {
                        result[i] = s1[i].wrapping_sub(s2[i]);
                    } else {
                        result[i] = s1[i].wrapping_add(s2[i]);
                    }
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = (s1[i] & 0xFFFF) as u16;
                    let a_hi = ((s1[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (s2[i] & 0xFFFF) as u16;
                    let b_hi = ((s2[i] >> 16) & 0xFFFF) as u16;
                    let sel_lo = sel[i] & 1;
                    let sel_hi = (sel[i] >> 16) & 1;
                    let r_lo = if sel_lo != 0 { a_lo.wrapping_sub(b_lo) } else { a_lo.wrapping_add(b_lo) };
                    let r_hi = if sel_hi != 0 { a_hi.wrapping_sub(b_hi) } else { a_hi.wrapping_add(b_hi) };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut val = 0u32;
                    for j in 0..4u32 {
                        let a_byte = ((s1[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((s2[i] >> (j * 8)) & 0xFF) as u8;
                        let s_bit = (sel[i] >> (j * 8)) & 1;
                        let r = if s_bit != 0 {
                            a_byte.wrapping_sub(b_byte)
                        } else {
                            a_byte.wrapping_add(b_byte)
                        };
                        val |= (r as u32) << (j * 8);
                    }
                    result[i] = val;
                }
            }
        }

        result
    }

    pub(super) fn vector_add(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
                // 8 x 32-bit integer lanes
                for i in 0..8 {
                    result[i] = a[i].wrapping_add(b[i]);
                }
            }
            ElementType::Float32 => {
                // 8 x 32-bit IEEE 754 float lanes
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa + fb).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 16 x 16-bit integer lanes (2 per u32)
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = a_lo.wrapping_add(b_lo);
                    let r_hi = a_hi.wrapping_add(b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                // 16 x BFloat16 lanes (2 per u32)
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo + b_lo);
                    let r_hi = Self::f32_to_bf16(a_hi + b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 x 8-bit lanes (4 per u32)
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = a_byte.wrapping_add(b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector subtraction by element type.
    pub(super) fn vector_sub(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = a[i].wrapping_sub(b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa - fb).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = a_lo.wrapping_sub(b_lo);
                    let r_hi = a_hi.wrapping_sub(b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo - b_lo);
                    let r_hi = Self::f32_to_bf16(a_hi - b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = a_byte.wrapping_sub(b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector multiplication by element type.
    pub(super) fn vector_mul(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = a[i].wrapping_mul(b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa * fb).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = a_lo.wrapping_mul(b_lo);
                    let r_hi = a_hi.wrapping_mul(b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo * b_lo);
                    let r_hi = Self::f32_to_bf16(a_hi * b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = a_byte.wrapping_mul(b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector minimum by element type.
    pub(super) fn vector_min(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = std::cmp::min(a[i] as i32, b[i] as i32) as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = std::cmp::min(a[i], b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = fa.min(fb).to_bits();
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo.min(b_lo));
                    let r_hi = Self::f32_to_bf16(a_hi.min(b_hi));
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = std::cmp::min(a_lo, b_lo) as u16;
                    let r_hi = std::cmp::min(a_hi, b_hi) as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = std::cmp::min(a_lo, b_lo);
                    let r_hi = std::cmp::min(a_hi, b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = std::cmp::min(a_byte, b_byte) as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = std::cmp::min(a_byte, b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector maximum by element type.
    pub(super) fn vector_max(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = std::cmp::max(a[i] as i32, b[i] as i32) as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = std::cmp::max(a[i], b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = fa.max(fb).to_bits();
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo.max(b_lo));
                    let r_hi = Self::f32_to_bf16(a_hi.max(b_hi));
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = std::cmp::max(a_lo, b_lo) as u16;
                    let r_hi = std::cmp::max(a_hi, b_hi) as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = std::cmp::max(a_lo, b_lo);
                    let r_hi = std::cmp::max(a_hi, b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = std::cmp::max(a_byte, b_byte) as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = std::cmp::max(a_byte, b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector logical left shift: each lane is shifted left by the corresponding shift amount.
    pub(super) fn vector_shift_left(src: &[u32; 8], shift: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
                for i in 0..8 {
                    let sh = shift[i] & 0x1F;
                    result[i] = src[i].wrapping_shl(sh);
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let src_lo = src[i] & 0xFFFF;
                    let src_hi = (src[i] >> 16) & 0xFFFF;
                    let sh_lo = shift[i] & 0xF;
                    let sh_hi = (shift[i] >> 16) & 0xF;
                    let r_lo = (src_lo << sh_lo) & 0xFFFF;
                    let r_hi = (src_hi << sh_hi) & 0xFFFF;
                    result[i] = r_lo | (r_hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let s = (src[i] >> (j * 8)) & 0xFF;
                        let sh = (shift[i] >> (j * 8)) & 0x7;
                        let v = (s << sh) & 0xFF;
                        r |= v << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }
        result
    }

    /// Vector logical right shift: unsigned shift right.
    pub(super) fn vector_shift_right_logical(src: &[u32; 8], shift: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
                for i in 0..8 {
                    let sh = shift[i] & 0x1F;
                    result[i] = src[i].wrapping_shr(sh);
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let src_lo = src[i] & 0xFFFF;
                    let src_hi = (src[i] >> 16) & 0xFFFF;
                    let sh_lo = shift[i] & 0xF;
                    let sh_hi = (shift[i] >> 16) & 0xF;
                    let r_lo = src_lo >> sh_lo;
                    let r_hi = src_hi >> sh_hi;
                    result[i] = r_lo | (r_hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let s = (src[i] >> (j * 8)) & 0xFF;
                        let sh = (shift[i] >> (j * 8)) & 0x7;
                        let v = s >> sh;
                        r |= v << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }
        result
    }

    /// Vector arithmetic right shift: signed shift right (preserves sign bit).
    pub(super) fn vector_shift_right_arith(src: &[u32; 8], shift: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];
        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 | ElementType::Float32 => {
                for i in 0..8 {
                    let sh = (shift[i] & 0x1F) as u32;
                    result[i] = ((src[i] as i32).wrapping_shr(sh)) as u32;
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let src_lo = (src[i] & 0xFFFF) as i16;
                    let src_hi = ((src[i] >> 16) & 0xFFFF) as i16;
                    let sh_lo = (shift[i] & 0xF) as u32;
                    let sh_hi = ((shift[i] >> 16) & 0xF) as u32;
                    let r_lo = ((src_lo >> sh_lo) as u16) as u32;
                    let r_hi = ((src_hi >> sh_hi) as u16) as u32;
                    result[i] = r_lo | (r_hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let s = ((src[i] >> (j * 8)) & 0xFF) as i8;
                        let sh = (shift[i] >> (j * 8)) & 0x7;
                        let v = ((s >> sh) as u8) as u32;
                        r |= v << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }
        result
    }

    /// Absolute value.  The GTZ suffix describes comparison flags written to
    /// the `cmp` register, NOT a condition on the primary result.  The primary
    /// output is always abs(s).
    pub(super) fn vector_abs_gtz(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    let val = src[i] as i32;
                    result[i] = val.wrapping_abs() as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                // Unsigned: abs is identity.
                result = *src;
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
                    result[i] = f.abs().to_bits();
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as i16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = lo.wrapping_abs() as u16;
                    let r_hi = hi.wrapping_abs() as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                result = *src;
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((src[i] & 0xFFFF) as u16);
                    let hi = Self::bf16_to_f32(((src[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(lo.abs());
                    let r_hi = Self::f32_to_bf16(hi.abs());
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = byte.wrapping_abs() as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                result = *src;
            }
        }

        result
    }

    /// Unconditional negate.  The GTZ suffix describes comparison flags written
    /// to the `cmp` register, NOT a condition on the primary result.  The
    /// primary output is always -s.
    pub(super) fn vector_neg_gtz(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    let val = src[i] as i32;
                    result[i] = val.wrapping_neg() as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = 0u32.wrapping_sub(src[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    // Flip sign bit.
                    result[i] = src[i] ^ 0x8000_0000;
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as i16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = lo.wrapping_neg() as u16;
                    let r_hi = hi.wrapping_neg() as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as u16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = 0u16.wrapping_sub(lo);
                    let r_hi = 0u16.wrapping_sub(hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    // Flip sign bits for bf16 elements.
                    result[i] = src[i] ^ 0x8000_8000;
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = byte.wrapping_neg() as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = 0u8.wrapping_sub(byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Negate if less than zero: dst[i] = (src[i] < 0) ? -src[i] : src[i]
    /// This is essentially abs() for signed types.
    /// Conditional negate: d[i] = (cmp[i] > 0) ? -s1[i] : s1[i]
    pub(super) fn vector_bneg_gtz(cmp: &[u32; 8], s1: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    let c = cmp[i] as i32;
                    let v = s1[i] as i32;
                    result[i] = if c > 0 { v.wrapping_neg() as u32 } else { s1[i] };
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    let c = cmp[i];
                    result[i] = if c > 0 { (s1[i] as i32).wrapping_neg() as u32 } else { s1[i] };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let c = f32::from_bits(cmp[i]);
                    let v = f32::from_bits(s1[i]);
                    result[i] = if c > 0.0 { (-v).to_bits() } else { s1[i] };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let c_lo = (cmp[i] & 0xFFFF) as i16;
                    let c_hi = ((cmp[i] >> 16) & 0xFFFF) as i16;
                    let v_lo = (s1[i] & 0xFFFF) as i16;
                    let v_hi = ((s1[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = if c_lo > 0 { v_lo.wrapping_neg() } else { v_lo };
                    let r_hi = if c_hi > 0 { v_hi.wrapping_neg() } else { v_hi };
                    result[i] = (r_lo as u16 as u32) | ((r_hi as u16 as u32) << 16);
                }
            }
            ElementType::UInt16 | ElementType::BFloat16 => {
                for i in 0..8 {
                    let c_lo = (cmp[i] & 0xFFFF) as u16;
                    let c_hi = ((cmp[i] >> 16) & 0xFFFF) as u16;
                    let v_lo = (s1[i] & 0xFFFF) as i16;
                    let v_hi = ((s1[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = if c_lo > 0 { v_lo.wrapping_neg() } else { v_lo };
                    let r_hi = if c_hi > 0 { v_hi.wrapping_neg() } else { v_hi };
                    result[i] = (r_lo as u16 as u32) | ((r_hi as u16 as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut val = 0u32;
                    for j in 0..4u32 {
                        let c = ((cmp[i] >> (j * 8)) & 0xFF) as i8;
                        let v = ((s1[i] >> (j * 8)) & 0xFF) as i8;
                        let r = if c > 0 { v.wrapping_neg() } else { v };
                        val |= (r as u8 as u32) << (j * 8);
                    }
                    result[i] = val;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut val = 0u32;
                    for j in 0..4u32 {
                        let c = ((cmp[i] >> (j * 8)) & 0xFF) as u8;
                        let v = ((s1[i] >> (j * 8)) & 0xFF) as i8;
                        let r = if c > 0 { v.wrapping_neg() } else { v };
                        val |= (r as u8 as u32) << (j * 8);
                    }
                    result[i] = val;
                }
            }
        }

        result
    }

    /// Bitwise NOT (VBNEG_LTZ).  The "B" prefix means "bitwise", matching
    /// VBAND/VBOR.  The LTZ suffix describes comparison flags written to
    /// the `cmp` register.  The primary output is always ~s.
    pub(super) fn vector_neg_ltz(src: &[u32; 8], _elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = !src[i];
        }
        result
    }

    /// Vector accumulate: acc += src (add to accumulator without multiply).
    pub(super) fn vector_accumulate(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        src: &[u32; 8],
        elem_type: ElementType,
    ) {
        let current = ctx.accumulator.read(acc_reg);
        let mut new_acc = current;

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
                for i in 0..8 {
                    new_acc[i] = current[i].wrapping_add(src[i] as u64);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + f).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as u64;
                    let hi = ((src[i] >> 16) & 0xFFFF) as u64;
                    new_acc[i] = current[i].wrapping_add(lo).wrapping_add(hi);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((src[i] & 0xFFFF) as u16) as f64;
                    let hi = Self::bf16_to_f32(((src[i] >> 16) & 0xFFFF) as u16) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + lo + hi).to_bits();
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut sum = 0u64;
                    for j in 0..4 {
                        sum += ((src[i] >> (j * 8)) & 0xFF) as u64;
                    }
                    new_acc[i] = current[i].wrapping_add(sum);
                }
            }
        }

        ctx.accumulator.write(acc_reg, new_acc);
    }

    /// Vector negate: dst = -src (per element negation).
    /// Hardware uses two's complement wrapping: -MIN = MIN.
    pub(super) fn vector_negate(src: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = (src[i] as i32).wrapping_neg() as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = 0u32.wrapping_sub(src[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
                    result[i] = (-f).to_bits();
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as i16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = lo.wrapping_neg() as u16;
                    let r_hi = hi.wrapping_neg() as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let lo = (src[i] & 0xFFFF) as u16;
                    let hi = ((src[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = 0u16.wrapping_sub(lo);
                    let r_hi = 0u16.wrapping_sub(hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((src[i] & 0xFFFF) as u16);
                    let hi = Self::bf16_to_f32(((src[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(-lo);
                    let r_hi = Self::f32_to_bf16(-hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = byte.wrapping_neg() as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((src[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = 0u8.wrapping_sub(byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector negate and add: dst = -src1 + src2.
    pub(super) fn vector_neg_add(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = ((-(a[i] as i32)) as i64 + (b[i] as i32) as i64) as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = b[i].wrapping_sub(a[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (-fa + fb).to_bits();
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo = (-a_lo).wrapping_add(b_lo) as u16;
                    let r_hi = (-a_hi).wrapping_add(b_hi) as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = b_lo.wrapping_sub(a_lo);
                    let r_hi = b_hi.wrapping_sub(a_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(-a_lo + b_lo);
                    let r_hi = Self::f32_to_bf16(-a_hi + b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte = (-a_byte).wrapping_add(b_byte) as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = b_byte.wrapping_sub(a_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Subtract if less-than: dst[i] = (a[i] < b[i]) ? a[i] - b[i] : a[i]
    /// Unconditional subtraction: d = a - b.  The LT suffix describes
    /// comparison flags written to the `cmp` register, NOT a condition on
    /// the subtraction.
    pub(super) fn vector_sub_lt(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        Self::vector_sub_unconditional(a, b, elem_type)
    }

    /// Unconditional subtraction: d = a - b.  The GE suffix describes
    /// comparison flags written to the `cmp` register, NOT a condition on
    /// the subtraction.
    pub(super) fn vector_sub_ge(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        Self::vector_sub_unconditional(a, b, elem_type)
    }

    /// Shared unconditional subtraction for vsub_lt / vsub_ge.
    fn vector_sub_unconditional(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32
            | ElementType::Int64 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = a[i].wrapping_sub(b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa - fb).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = a_lo.wrapping_sub(b_lo);
                    let r_hi = a_hi.wrapping_sub(b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16(a_lo - b_lo);
                    let r_hi = Self::f32_to_bf16(a_hi - b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = a_byte.wrapping_sub(b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Maximum difference if less-than: dst[i] = a[i] - b[i] (wrapping for signed,
    /// saturating at 0 for unsigned). The comparison flag output indicates a < b.
    pub(super) fn vector_maxdiff_lt(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                // Signed: widen to i64 to avoid overflow, then clamp to [0, u32::MAX].
                // wrapping_sub at i32 width loses the true sign of large differences.
                for i in 0..8 {
                    let diff = (a[i] as i32 as i64) - (b[i] as i32 as i64);
                    result[i] = diff.max(0) as u32;
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = a[i].saturating_sub(b[i]);
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa - fb).max(0.0).to_bits();
                }
            }
            ElementType::Int16 => {
                // Signed: widen to i32 to avoid overflow, then clamp to [0, u16::MAX].
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16 as i32;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16 as i32;
                    let b_lo = (b[i] & 0xFFFF) as i16 as i32;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16 as i32;
                    let r_lo = (a_lo - b_lo).max(0) as u16;
                    let r_hi = (a_hi - b_hi).max(0) as u16;
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = a_lo.saturating_sub(b_lo);
                    let r_hi = a_hi.saturating_sub(b_hi);
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16);
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                    let r_lo = Self::f32_to_bf16((a_lo - b_lo).max(0.0));
                    let r_hi = Self::f32_to_bf16((a_hi - b_hi).max(0.0));
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                // Signed: widen to i16 to avoid overflow, then clamp to [0, u8::MAX].
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8 as i16;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8 as i16;
                        let r_byte = (a_byte - b_byte).max(0) as u8;
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
            ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte = a_byte.saturating_sub(b_byte);
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// VFLOOR: BF16 -> S32 with shift scaling.
    ///
    /// Converts 8 packed BF16 values (from 4 u32 words in `src[0..4]`) to 8 i32:
    ///   result[i] = saturate_s32(floor(bf16_value * 2^shift))
    ///
    /// The shift parameter comes from a scalar register operand.
    /// Hardware uses the low 6 bits as a signed value (range -32 to +31).
    pub(super) fn vector_floor_bf16_to_s32(src: &[u32; 8], shift: i32) -> [u32; 8] {
        use super::vector_float::fp32_flush_to_zero;
        let mut result = [0u32; 8];
        // Hardware interprets shift as signed 6-bit (-32 to +31).
        let masked = (shift & 0x3F) as i8;
        let effective_shift = if masked >= 32 { masked - 64 } else { masked } as i32;
        let scale = (2.0f64).powi(effective_shift);
        for i in 0..8 {
            let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
            let f = f32::from_bits(fp32_flush_to_zero(Self::bf16_to_f32(bf16).to_bits()));
            // Hardware saturates NaN to INT_MAX (positive saturation).
            if f.is_nan() {
                result[i] = i32::MAX as u32;
                continue;
            }
            // Scale then floor, using f64 for intermediate precision.
            let scaled = (f as f64) * scale;
            let floored = scaled.floor();
            // Saturate to i32 range.
            let clamped = floored.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            result[i] = clamped as u32;
        }
        result
    }

    // ========== Dispatch-owning execute_* functions ==========
    //
    // These combine both the execute_half and execute_wide arms for each
    // operation, gated by op.is_wide_vector internally.

    /// Execute vector Add (includes VADDSUB detection).
    pub(super) fn execute_add(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        // Two variants:
        // 1. VADD_elem: simple vector add (2 sources)
        // 2. VADDSUB: conditional add/subtract (2 vector + 1 scalar source)
        //    vaddsub.8 $d, $s1, $s2, $sel
        //    d[i] = (sel[i] & 1) ? s1[i] - s2[i] : s1[i] + s2[i]
        //    sel is a SCALAR register (bitmask), not a vector.
        let has_scalar_sel = op.sources.iter()
            .any(|s| matches!(s, Operand::ScalarReg(_)))
            && op.sources.iter()
                .filter(|s| matches!(s, Operand::VectorReg(_)))
                .count() == 2;

        if op.is_wide_vector {
            if has_scalar_sel {
                // VADDSUB: read full 512-bit sources and scalar selector.
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let sel_scalar = Self::get_nth_scalar_source(op, ctx, 0);
                // Split wide vectors into lo/hi halves for per-half processing.
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let b_lo: [u32; 8] = b[..8].try_into().unwrap();
                let b_hi: [u32; 8] = b[8..].try_into().unwrap();
                // Expand selector bits: lo half uses lower bits, hi uses upper.
                // For 8-bit elements (32 per half), the selector is a 64-bit
                // register pair (eL class: l_n = {r(16+2n), r(16+2n+1)}).
                // LLVM decodes this as a single ScalarReg for the even register;
                // we read reg+1 for the upper 32 bits.
                let elems_per_half = 256 / et.bits() as u32;
                let sel_lo = Self::expand_select_mask(sel_scalar, et);
                let sel_hi_bits = if elems_per_half >= 32 {
                    // 8-bit: 64-bit selector, upper 32 bits in the next register
                    let sel_reg = op.sources.iter().find_map(|s| {
                        if let Operand::ScalarReg(r) = s { Some(*r) } else { None }
                    }).unwrap_or(0);
                    ctx.scalar.read(sel_reg + 1)
                } else {
                    sel_scalar >> elems_per_half
                };
                let sel_hi = Self::expand_select_mask(sel_hi_bits, et);
                let lo = Self::vector_addsub(&a_lo, &b_lo, &sel_lo, et);
                let hi = Self::vector_addsub(&a_hi, &b_hi, &sel_hi, et);
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&lo);
                result[8..].copy_from_slice(&hi);
                Self::write_wide_vec_dest(op, ctx, result);
                return true;
            }
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_add);
            Self::write_wide_vec_dest(op, ctx, result);
            true
        } else {
            if has_scalar_sel {
                let s1 = Self::get_vector_source(op, ctx, 0);
                let s2 = Self::get_vector_source(op, ctx, 1);
                // Read the scalar selector bitmask
                let sel_scalar = Self::get_nth_scalar_source(op, ctx, 0);
                let sel = Self::expand_select_mask(sel_scalar, et);
                let result = Self::vector_addsub(&s1, &s2, &sel, et);
                Self::write_vector_dest(op, ctx, result);
            } else {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_add(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
            }
            true
        }
    }

    /// Execute vector Neg (includes accumulator-only routing).
    pub(super) fn execute_neg(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        // Detect accumulator-only: AccumReg source with no VectorReg source.
        let is_accum_only = op.sources.iter().any(|s| matches!(s, Operand::AccumReg(_)))
            && !op.sources.iter().any(|s| matches!(s, Operand::VectorReg(_)));

        if is_accum_only {
            Self::execute_acc_negate(op, ctx);
        } else if op.is_wide_vector {
            let a = Self::get_wide_vec_source(op, ctx, 0);
            let result = Self::wide_element_wise_unary(&a, et, Self::vector_negate);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            // Vector negate: vneg $d, $s1 -- per-element negate.
            let src = Self::get_vector_source(op, ctx, 0);
            let result = Self::vector_negate(&src, et);
            Self::write_vector_dest(op, ctx, result);
        }
        true
    }

    /// Execute vector shift left.
    pub(super) fn execute_shl(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        Self::execute_binary_elementwise(op, ctx, et, Self::vector_shift_left)
    }

    /// Execute vector logical shift right.
    pub(super) fn execute_srl(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        Self::execute_binary_elementwise(op, ctx, et, Self::vector_shift_right_logical)
    }

    /// Execute vector arithmetic shift right.
    pub(super) fn execute_sra(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        Self::execute_binary_elementwise(op, ctx, et, Self::vector_shift_right_arith)
    }

    /// Execute AbsGtz: absolute value + comparison flags (s > 0).
    pub(super) fn execute_abs_gtz(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        if op.is_wide_vector {
            let a = Self::get_wide_vec_source(op, ctx, 0);
            let result = Self::wide_element_wise_unary(&a, et, Self::vector_abs_gtz);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let src = Self::get_vector_source(op, ctx, 0);
            let result = Self::vector_abs_gtz(&src, et);
            Self::write_vector_dest(op, ctx, result);
            // cmp = per-element (s > 0)
            let zero = [0u32; 8];
            let cmp = Self::vector_compare_lt(&zero, &src, et); // 0 < s  =>  s > 0
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
        }
        true
    }

    /// Execute NegGtz: negate + comparison flags (s > 0).
    pub(super) fn execute_neg_gtz(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        if op.is_wide_vector {
            // Two-vector-source variant (VNEG_GTZ with select operand) uses fallback.
            let vec_source_count = op.sources.iter()
                .filter(|s| matches!(s, Operand::VectorReg(_)))
                .count();
            if vec_source_count >= 2 {
                // Two-source wide VNEG_GTZ: not yet implemented.
                return false;
            }
            let a = Self::get_wide_vec_source(op, ctx, 0);
            let result = Self::wide_element_wise_unary(&a, et, Self::vector_neg_gtz);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let vec_source_count = op.sources.iter()
                .filter(|s| matches!(s, Operand::VectorReg(_)))
                .count();

            let src = if vec_source_count >= 2 {
                let cond = Self::get_vector_source(op, ctx, 0);
                let s1 = Self::get_vector_source(op, ctx, 1);
                let result = Self::vector_bneg_gtz(&cond, &s1, et);
                Self::write_vector_dest(op, ctx, result);
                s1
            } else {
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_neg_gtz(&src, et);
                Self::write_vector_dest(op, ctx, result);
                src
            };
            // cmp = per-element (s > 0)
            let zero = [0u32; 8];
            let cmp = Self::vector_compare_lt(&zero, &src, et);
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
        }
        true
    }

    /// Execute NegLtz: bitwise NOT + comparison flags (s < 0).
    pub(super) fn execute_neg_ltz(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        if op.is_wide_vector {
            let a = Self::get_wide_vec_source(op, ctx, 0);
            let result = Self::wide_element_wise_unary(&a, et, Self::vector_neg_ltz);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let src = Self::get_vector_source(op, ctx, 0);
            let result = Self::vector_neg_ltz(&src, et);
            Self::write_vector_dest(op, ctx, result);
            // cmp = per-element (s < 0)
            let zero = [0u32; 8];
            let cmp = Self::vector_compare_lt(&src, &zero, et);
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
        }
        true
    }

    /// Execute SubLt: subtraction + comparison flags (a < b).
    pub(super) fn execute_sub_lt(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        if op.is_wide_vector {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_sub_lt);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = Self::vector_sub_lt(&a, &b, et);
            Self::write_vector_dest(op, ctx, result);
            // cmp = per-element (a < b)
            let cmp = Self::vector_compare_lt(&a, &b, et);
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
        }
        true
    }

    /// Execute SubGe: subtraction + comparison flags (a >= b).
    pub(super) fn execute_sub_ge(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        if op.is_wide_vector {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_sub_ge);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = Self::vector_sub_ge(&a, &b, et);
            Self::write_vector_dest(op, ctx, result);
            // cmp = per-element (a >= b)
            let cmp = Self::vector_compare_ge(&a, &b, et);
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
        }
        true
    }

    /// Execute MaxDiffLt: max(a-b, 0) + comparison flags (a < b).
    pub(super) fn execute_maxdiff_lt(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        if op.is_wide_vector {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_maxdiff_lt);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = Self::vector_maxdiff_lt(&a, &b, et);
            Self::write_vector_dest(op, ctx, result);
            // cmp = per-element (a < b)
            let cmp = Self::vector_compare_lt(&a, &b, et);
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
        }
        true
    }

    /// Execute Accumulate/AccumSub/AccumNegAdd/AccumNegSub/NegAdd.
    ///
    /// Routes between accumulator-to-accumulator operations and regular
    /// vector operations based on source operand types.
    pub(super) fn execute_accumulate(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
        semantic: crate::tablegen::SemanticOp,
    ) -> bool {
        let has_acc_source = op.sources.iter().any(|s| matches!(s, Operand::AccumReg(_)));

        if has_acc_source {
            // VADD/VSUB/VNEGADD/VNEGSUB: accumulator-to-accumulator
            Self::execute_acc_add_sub(op, ctx);
        } else if matches!(semantic, crate::tablegen::SemanticOp::NegAdd) {
            // Regular vector negate-add (no accumulator involvement).
            if op.is_wide_vector {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_neg_add);
                Self::write_wide_vec_dest(op, ctx, result);
            } else {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_neg_add(&a, &b, et);
                Self::write_vector_dest(op, ctx, result);
            }
        } else if op.is_wide_vector {
            // Wide vector-into-accumulator: not yet implemented.
            return false;
        } else {
            // Legacy: accumulate a vector source into an accumulator.
            let src = Self::get_vector_source(op, ctx, 0);
            let acc_reg = Self::get_acc_dest(op);
            Self::vector_accumulate(ctx, acc_reg, &src, et);
        }
        true
    }

    /// Accumulator-to-accumulator add/subtract.
    ///
    /// Handles VADD, VSUB, VNEGADD, VNEGSUB and their .f (float) variants.
    /// These take two accumulator (cm-class, 1024-bit) sources and write
    /// a 1024-bit result.
    ///
    /// The config register (scalar operand) controls:
    /// - Bit  0: zero_acc1 (zero acc1 before operation)
    /// - Bit 10: shift16 (right-shift result by 16 bits)
    /// - Bit 11: sub_acc1 (negate acc1)
    /// - Bit 12: sub_acc2 (negate acc2)
    ///
    /// Base operations (before config modifiers):
    /// - vadd:    dst = acc1 + acc2
    /// - vsub:    dst = acc1 - acc2
    /// - vnegadd: dst = -acc1 + acc2
    /// - vnegsub: dst = -acc1 - acc2
    pub(super) fn execute_acc_add_sub(op: &SlotOp, ctx: &mut ExecutionContext) {
        // Collect the two AccumReg source indices.
        let mut acc_sources: Vec<u8> = Vec::new();
        for src in &op.sources {
            if let Operand::AccumReg(r) = src {
                acc_sources.push(*r);
            }
        }

        let acc1_reg = if !acc_sources.is_empty() { acc_sources[0] } else { 0 };
        let acc2_reg = if acc_sources.len() >= 2 { acc_sources[1] } else { acc1_reg };

        // Read config register.
        let conf = Self::get_config_register(op, ctx).unwrap_or(0);
        let zero_acc1 = (conf & 1) != 0;
        let shift16 = ((conf >> 10) & 1) != 0;
        let sub_acc1 = ((conf >> 11) & 1) != 0;
        let sub_acc2 = ((conf >> 12) & 1) != 0;

        // Determine md1/md2 encoding bits from SemanticOp.
        //
        // The hardware VACC ALU has two mode bits (md1, md2) that XOR with the
        // config register's sub_acc1/sub_acc2 bits to determine the effective
        // sign of each accumulator operand:
        //   negate_acc1 = md1 XOR sub_acc1
        //   negate_acc2 = md2 XOR sub_acc2
        //
        // Encoding:       md1  md2   Operation
        //   VADD           0    0    +acc1 + acc2
        //   VSUB           0    1    +acc1 - acc2
        //   VNEGSUB        1    0    -acc1 + acc2
        //   VNEGADD        1    1    -acc1 - acc2 = -(acc1 + acc2)
        //
        // Note: VNEGADD negates BOTH operands (the "add" refers to the
        // unsigned operation, "neg" flips both signs). VNEGSUB negates only
        // acc1 (the "sub" flips one sign, "neg" flips another, net one flip).
        let semantic = op.semantic.unwrap_or(SemanticOp::Accumulate);
        let enc_lower = op.encoding_name.as_deref().unwrap_or("").to_ascii_lowercase();
        // Map semantic -> (md1, md2).  Accept both AccumXxx (build-time) and
        // generic NegAdd (runtime mnemonic dispatch, no NegSub variant exists).
        let (md1, md2) = match semantic {
            SemanticOp::Accumulate => (false, false),                    // VADD
            SemanticOp::AccumSub => (false, true),                       // VSUB
            SemanticOp::AccumNegAdd => (true, true),                     // VNEGADD
            SemanticOp::AccumNegSub => (true, false),                    // VNEGSUB
            SemanticOp::NegAdd => {
                // Runtime mnemonic dispatch -- both vnegadd and vnegsub map to
                // NegAdd.  Distinguish via encoding name.
                if enc_lower.contains("negsub") {
                    (true, false)                                        // VNEGSUB
                } else {
                    (true, true)                                         // VNEGADD
                }
            }
            _ => (false, false),
        };
        // Float mode from element_type (set at build time from mnemonic ".f" suffix).
        let is_float = matches!(op.element_type, Some(ElementType::Float32) | Some(ElementType::BFloat16));

        // Wide (cm, 1024-bit) vs narrow (bm, 512-bit) accumulator.
        // Use AccumWidth metadata from decoder when available (structural signal).
        // Fallback: cm registers always use even base indices, so odd index implies bm.
        let dst_reg = Self::get_acc_dest(op);
        let is_wide = match op.accum_width {
            Some(crate::tablegen::decoder_ffi::AccumWidth::Full) => true,
            Some(crate::tablegen::decoder_ffi::AccumWidth::Half)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterLow)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterHigh) => false,
            None => {
                // Legacy fallback: if any index is odd, it must be a bm half.
                let any_odd = acc_sources.iter().any(|r| r % 2 != 0) || dst_reg % 2 != 0;
                !any_odd
            }
        };

        // Compute effective signs: md bits XOR config bits (mirrors hardware).
        let negate_acc1 = md1 ^ sub_acc1;
        let negate_acc2 = md2 ^ sub_acc2;

        if is_wide {
            // Wide path: 1024-bit cm registers (16 x 64-bit lanes).
            let a1 = ctx.accumulator.read_wide(acc1_reg);
            let a2 = ctx.accumulator.read_wide(acc2_reg);
            let mut result = [0u64; 16];

            if is_float {
                use super::vector_float::{fp32_flush_to_zero, aie2_acc_fp32_add};
                for i in 0..16 {
                    let mut a1_lo = if zero_acc1 { 0u32 } else { fp32_flush_to_zero(a1[i] as u32) };
                    let mut a1_hi = if zero_acc1 { 0u32 } else { fp32_flush_to_zero((a1[i] >> 32) as u32) };
                    let mut a2_lo = fp32_flush_to_zero(a2[i] as u32);
                    let mut a2_hi = fp32_flush_to_zero((a2[i] >> 32) as u32);

                    // Negate by flipping sign bit (works for zero, normal, inf, NaN).
                    if negate_acc1 { a1_lo ^= 0x8000_0000; a1_hi ^= 0x8000_0000; }
                    if negate_acc2 { a2_lo ^= 0x8000_0000; a2_hi ^= 0x8000_0000; }

                    // Use acc ALU add (no output FTZ): the accumulator
                    // register file preserves denormalized fp32 values.
                    let r_lo = aie2_acc_fp32_add(a1_lo, a2_lo);
                    let r_hi = aie2_acc_fp32_add(a1_hi, a2_hi);
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            } else {
                // Acc32 mode: each u64 holds two independent 32-bit accumulator
                // lanes.  The hardware ALU has no carry chain between the lo and
                // hi halves -- all arithmetic is per-lane 32-bit.
                for i in 0..16 {
                    let v1_lo = if zero_acc1 { 0i32 } else { a1[i] as i32 };
                    let v1_hi = if zero_acc1 { 0i32 } else { (a1[i] >> 32) as i32 };
                    let v2_lo = a2[i] as i32;
                    let v2_hi = (a2[i] >> 32) as i32;
                    let v1_lo = if negate_acc1 { v1_lo.wrapping_neg() } else { v1_lo };
                    let v1_hi = if negate_acc1 { v1_hi.wrapping_neg() } else { v1_hi };
                    let v2_lo = if negate_acc2 { v2_lo.wrapping_neg() } else { v2_lo };
                    let v2_hi = if negate_acc2 { v2_hi.wrapping_neg() } else { v2_hi };
                    let mut r_lo = v1_lo.wrapping_add(v2_lo);
                    let mut r_hi = v1_hi.wrapping_add(v2_hi);
                    if shift16 { r_lo >>= 16; r_hi >>= 16; }
                    result[i] = (r_lo as u32 as u64) | ((r_hi as u32 as u64) << 32);
                }
            }

            ctx.accumulator.write_wide(dst_reg, result);
        } else {
            // Narrow path: 512-bit bm registers (8 x 64-bit lanes).
            let a1 = ctx.accumulator.read(acc1_reg);
            let a2 = ctx.accumulator.read(acc2_reg);
            let mut result = [0u64; 8];

            if is_float {
                use super::vector_float::{fp32_flush_to_zero, aie2_acc_fp32_add};
                for i in 0..8 {
                    let mut a1_lo = if zero_acc1 { 0u32 } else { fp32_flush_to_zero(a1[i] as u32) };
                    let mut a1_hi = if zero_acc1 { 0u32 } else { fp32_flush_to_zero((a1[i] >> 32) as u32) };
                    let mut a2_lo = fp32_flush_to_zero(a2[i] as u32);
                    let mut a2_hi = fp32_flush_to_zero((a2[i] >> 32) as u32);

                    if negate_acc1 { a1_lo ^= 0x8000_0000; a1_hi ^= 0x8000_0000; }
                    if negate_acc2 { a2_lo ^= 0x8000_0000; a2_hi ^= 0x8000_0000; }

                    let r_lo = aie2_acc_fp32_add(a1_lo, a2_lo);
                    let r_hi = aie2_acc_fp32_add(a1_hi, a2_hi);
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            } else {
                // Acc32 mode: independent 32-bit lane operations (see wide path).
                for i in 0..8 {
                    let v1_lo = if zero_acc1 { 0i32 } else { a1[i] as i32 };
                    let v1_hi = if zero_acc1 { 0i32 } else { (a1[i] >> 32) as i32 };
                    let v2_lo = a2[i] as i32;
                    let v2_hi = (a2[i] >> 32) as i32;
                    let v1_lo = if negate_acc1 { v1_lo.wrapping_neg() } else { v1_lo };
                    let v1_hi = if negate_acc1 { v1_hi.wrapping_neg() } else { v1_hi };
                    let v2_lo = if negate_acc2 { v2_lo.wrapping_neg() } else { v2_lo };
                    let v2_hi = if negate_acc2 { v2_hi.wrapping_neg() } else { v2_hi };
                    let mut r_lo = v1_lo.wrapping_add(v2_lo);
                    let mut r_hi = v1_hi.wrapping_add(v2_hi);
                    if shift16 { r_lo >>= 16; r_hi >>= 16; }
                    result[i] = (r_lo as u32 as u64) | ((r_hi as u32 as u64) << 32);
                }
            }

            ctx.accumulator.write(dst_reg, result);
        }
    }

    /// Accumulator negate: dst = -src.
    ///
    /// Handles VNEG on both bm (512-bit) and cm (1024-bit) accumulators.
    /// The config word controls zero_acc (zero the source before negation,
    /// producing 0). Float mode is detected from element_type (Float32/BFloat16):
    /// each u64 lane holds two fp32 values (bits [31:0] and bits [63:32]).
    pub(super) fn execute_acc_negate(op: &SlotOp, ctx: &mut ExecutionContext) {
        let acc_reg = op.sources.iter().find_map(|s| match s {
            Operand::AccumReg(r) => Some(*r),
            _ => None,
        }).unwrap_or(0);
        let dst_reg = Self::get_acc_dest(op);

        let conf = Self::get_config_register(op, ctx).unwrap_or(0);
        let zero_acc = (conf & 1) != 0;
        // Float mode from element_type (set at build time from mnemonic ".f" suffix).
        let is_float = matches!(op.element_type, Some(ElementType::Float32) | Some(ElementType::BFloat16));

        // Wide (cm) vs narrow (bm) -- same detection as execute_acc_add_sub.
        // Use AccumWidth metadata from decoder when available (structural signal).
        let is_wide = match op.accum_width {
            Some(crate::tablegen::decoder_ffi::AccumWidth::Full) => true,
            Some(crate::tablegen::decoder_ffi::AccumWidth::Half)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterLow)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::QuarterHigh) => false,
            None => {
                let any_odd = acc_reg % 2 != 0 || dst_reg % 2 != 0;
                !any_odd
            }
        };

        if is_wide {
            let src = ctx.accumulator.read_wide(acc_reg);
            let mut result = [0u64; 16];
            if is_float {
                use super::vector_float::{fp32_flush_to_zero, fp32_is_nan, fp32_make_nan};
                for i in 0..16 {
                    let lo = fp32_flush_to_zero(src[i] as u32);
                    let hi = fp32_flush_to_zero((src[i] >> 32) as u32);
                    let r_lo = if zero_acc { 0u32 }
                        else if fp32_is_nan(lo) { fp32_make_nan(false) }
                        else { lo ^ 0x8000_0000 };
                    let r_hi = if zero_acc { 0u32 }
                        else if fp32_is_nan(hi) { fp32_make_nan(false) }
                        else { hi ^ 0x8000_0000 };
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            } else {
                // Acc32 mode: independent 32-bit negation per lane half.
                for i in 0..16 {
                    let lo = if zero_acc { 0i32 } else { src[i] as i32 };
                    let hi = if zero_acc { 0i32 } else { (src[i] >> 32) as i32 };
                    let r_lo = lo.wrapping_neg() as u32;
                    let r_hi = hi.wrapping_neg() as u32;
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            }
            ctx.accumulator.write_wide(dst_reg, result);
        } else {
            let src = ctx.accumulator.read(acc_reg);
            let mut result = [0u64; 8];
            if is_float {
                use super::vector_float::{fp32_flush_to_zero, fp32_is_nan, fp32_make_nan};
                for i in 0..8 {
                    let lo = fp32_flush_to_zero(src[i] as u32);
                    let hi = fp32_flush_to_zero((src[i] >> 32) as u32);
                    let r_lo = if zero_acc { 0u32 }
                        else if fp32_is_nan(lo) { fp32_make_nan(false) }
                        else { lo ^ 0x8000_0000 };
                    let r_hi = if zero_acc { 0u32 }
                        else if fp32_is_nan(hi) { fp32_make_nan(false) }
                        else { hi ^ 0x8000_0000 };
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            } else {
                // Acc32 mode: independent 32-bit negation per lane half.
                for i in 0..8 {
                    let lo = if zero_acc { 0i32 } else { src[i] as i32 };
                    let hi = if zero_acc { 0i32 } else { (src[i] >> 32) as i32 };
                    let r_lo = lo.wrapping_neg() as u32;
                    let r_hi = hi.wrapping_neg() as u32;
                    result[i] = (r_lo as u64) | ((r_hi as u64) << 32);
                }
            }
            ctx.accumulator.write(dst_reg, result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{ElementType, Operand, SlotIndex, SlotOp};
    use crate::interpreter::state::ExecutionContext;
    use crate::tablegen::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    #[test]
    fn test_vector_add_i32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        assert!(VectorAlu::execute(&op, &mut ctx));
        assert_eq!(ctx.vector.read(2), [11, 22, 33, 44, 55, 66, 77, 88]);
    }

    #[test]
    fn test_vector_add_i16() {
        let mut ctx = make_ctx();
        // Pack 16-bit values: [1, 2] in lane 0, etc.
        ctx.vector.write(0, [0x0002_0001, 0, 0, 0, 0, 0, 0, 0]);
        ctx.vector.write(1, [0x0020_0010, 0, 0, 0, 0, 0, 0, 0]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Int16)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2)[0], 0x0022_0011);
    }

    #[test]
    fn test_vector_sub() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [100, 200, 300, 400, 500, 600, 700, 800]);
        ctx.vector.write(1, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Sub)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [90, 180, 270, 360, 450, 540, 630, 720]);
    }

    #[test]
    fn test_vector_mul() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [2, 3, 4, 5, 6, 7, 8, 9]);
        ctx.vector.write(1, [10, 10, 10, 10, 10, 10, 10, 10]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Mul)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    fn test_vector_min_max() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [5, 10, 15, 20, 25, 30, 35, 40]);
        ctx.vector.write(1, [10, 5, 20, 15, 30, 25, 40, 35]);

        // Min
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Min)
            .as_vector(ElementType::UInt32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [5, 5, 15, 15, 25, 25, 35, 35]);

        // Max
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Max)
            .as_vector(ElementType::UInt32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [10, 10, 20, 20, 30, 30, 40, 40]);
    }

    #[test]
    fn test_vector_add_f32() {
        let mut ctx = make_ctx();

        // Create float vectors: [1.0, 2.0, 3.0, ...] and [0.5, 0.5, ...]
        let a: [u32; 8] = [
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            4.0f32.to_bits(),
            5.0f32.to_bits(),
            6.0f32.to_bits(),
            7.0f32.to_bits(),
            8.0f32.to_bits(),
        ];
        let b: [u32; 8] = [0.5f32.to_bits(); 8];

        ctx.vector.write(0, a);
        ctx.vector.write(1, b);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);

        // Check results: 1.5, 2.5, 3.5, ...
        assert_eq!(f32::from_bits(result[0]), 1.5);
        assert_eq!(f32::from_bits(result[1]), 2.5);
        assert_eq!(f32::from_bits(result[2]), 3.5);
        assert_eq!(f32::from_bits(result[7]), 8.5);
    }

    #[test]
    fn test_vector_mul_f32() {
        let mut ctx = make_ctx();

        // [2.0, 3.0, 4.0, ...] * [0.5, 0.5, ...]
        let a: [u32; 8] = [
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            4.0f32.to_bits(),
            5.0f32.to_bits(),
            6.0f32.to_bits(),
            7.0f32.to_bits(),
            8.0f32.to_bits(),
            9.0f32.to_bits(),
        ];
        let b: [u32; 8] = [0.5f32.to_bits(); 8];

        ctx.vector.write(0, a);
        ctx.vector.write(1, b);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Mul)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);

        // Check results: 1.0, 1.5, 2.0, ...
        assert_eq!(f32::from_bits(result[0]), 1.0);
        assert_eq!(f32::from_bits(result[1]), 1.5);
        assert_eq!(f32::from_bits(result[2]), 2.0);
    }

    #[test]
    fn test_vector_min_max_f32() {
        let mut ctx = make_ctx();

        // Test with positive and negative floats
        let a: [u32; 8] = [
            1.0f32.to_bits(),
            (-2.0f32).to_bits(),
            3.0f32.to_bits(),
            (-4.0f32).to_bits(),
            5.0f32.to_bits(),
            6.0f32.to_bits(),
            7.0f32.to_bits(),
            8.0f32.to_bits(),
        ];
        let b: [u32; 8] = [
            0.5f32.to_bits(),
            (-1.0f32).to_bits(),
            2.0f32.to_bits(),
            (-5.0f32).to_bits(),
            4.0f32.to_bits(),
            7.0f32.to_bits(),
            6.0f32.to_bits(),
            9.0f32.to_bits(),
        ];

        ctx.vector.write(0, a);
        ctx.vector.write(1, b);

        // Min
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Min)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);
        assert_eq!(f32::from_bits(result[0]), 0.5);  // min(1.0, 0.5)
        assert_eq!(f32::from_bits(result[1]), -2.0); // min(-2.0, -1.0)
        assert_eq!(f32::from_bits(result[3]), -5.0); // min(-4.0, -5.0)

        // Max
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Max)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);
        assert_eq!(f32::from_bits(result[0]), 1.0);  // max(1.0, 0.5)
        assert_eq!(f32::from_bits(result[1]), -1.0); // max(-2.0, -1.0)
        assert_eq!(f32::from_bits(result[3]), -4.0); // max(-4.0, -5.0)
    }

    #[test]
    fn test_vector_add_bf16() {
        let mut ctx = make_ctx();

        // BFloat16 is upper 16 bits of f32
        // Pack two bf16 values per u32: low half and high half
        fn pack_bf16(lo: f32, hi: f32) -> u32 {
            let lo_bits = (lo.to_bits() >> 16) as u16;
            let hi_bits = (hi.to_bits() >> 16) as u16;
            (lo_bits as u32) | ((hi_bits as u32) << 16)
        }

        // Create vectors with bf16 pairs
        let a: [u32; 8] = [
            pack_bf16(1.0, 2.0),
            pack_bf16(3.0, 4.0),
            0, 0, 0, 0, 0, 0,
        ];
        let b: [u32; 8] = [
            pack_bf16(0.5, 0.5),
            pack_bf16(0.5, 0.5),
            0, 0, 0, 0, 0, 0,
        ];

        ctx.vector.write(0, a);
        ctx.vector.write(1, b);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::BFloat16)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);

        // Extract and check bf16 values
        let lo0 = VectorAlu::bf16_to_f32((result[0] & 0xFFFF) as u16);
        let hi0 = VectorAlu::bf16_to_f32(((result[0] >> 16) & 0xFFFF) as u16);

        // BFloat16 has limited precision, so check approximate equality
        assert!((lo0 - 1.5).abs() < 0.1, "Expected ~1.5, got {}", lo0);
        assert!((hi0 - 2.5).abs() < 0.1, "Expected ~2.5, got {}", hi0);
    }

    // -----------------------------------------------------------------------
    // Pure function tests (no dispatch, no ExecutionContext where possible)
    // -----------------------------------------------------------------------

    // -- vector_add: i8 --

    #[test]
    fn vector_add_i8_wrapping() {
        let a = [0x01_FF_80_7F_u32, 0, 0, 0, 0, 0, 0, 0]; // bytes: 0x7F, 0x80, 0xFF, 0x01
        let b = [0x01_01_01_01_u32, 0, 0, 0, 0, 0, 0, 0]; // all 1
        let r = VectorAlu::vector_add(&a, &b, ElementType::Int8);
        // 0x7F+1=0x80, 0x80+1=0x81, 0xFF+1=0x00(wrap), 0x01+1=0x02
        assert_eq!(r[0], 0x02_00_81_80);
    }

    // -- vector_sub: i8, i16 --

    #[test]
    fn vector_sub_i8() {
        let a = [0x00_80_FF_10_u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x01_01_01_01_u32, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_sub(&a, &b, ElementType::Int8);
        // 0x10-1=0x0F, 0xFF-1=0xFE, 0x80-1=0x7F, 0x00-1=0xFF(wrap)
        assert_eq!(r[0], 0xFF_7F_FE_0F);
    }

    #[test]
    fn vector_sub_i16() {
        let a = [0x0000_8000_u32, 0, 0, 0, 0, 0, 0, 0]; // lo=0x8000(-32768), hi=0
        let b = [0x0001_0001_u32, 0, 0, 0, 0, 0, 0, 0]; // lo=1, hi=1
        let r = VectorAlu::vector_sub(&a, &b, ElementType::Int16);
        // lo: 0x8000-1=0x7FFF, hi: 0-1=0xFFFF(wrap)
        assert_eq!(r[0], 0xFFFF_7FFF);
    }

    // -- vector_mul: i16 --

    #[test]
    fn vector_mul_i16() {
        let a = [0x0003_0002_u32, 0, 0, 0, 0, 0, 0, 0]; // lo=2, hi=3
        let b = [0x0005_0004_u32, 0, 0, 0, 0, 0, 0, 0]; // lo=4, hi=5
        let r = VectorAlu::vector_mul(&a, &b, ElementType::Int16);
        // lo: 2*4=8, hi: 3*5=15
        assert_eq!(r[0], 0x000F_0008);
    }

    // -- vector_min/max: signed i32, i16, i8 --

    #[test]
    fn vector_min_signed_i32() {
        let a = [(-5i32) as u32, 10, 0, 0, 0, 0, 0, 0];
        let b = [(-3i32) as u32, 5, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_min(&a, &b, ElementType::Int32);
        assert_eq!(r[0] as i32, -5);
        assert_eq!(r[1], 5);
    }

    #[test]
    fn vector_max_signed_i32() {
        let a = [(-5i32) as u32, 10, 0, 0, 0, 0, 0, 0];
        let b = [(-3i32) as u32, 5, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_max(&a, &b, ElementType::Int32);
        assert_eq!(r[0] as i32, -3);
        assert_eq!(r[1], 10);
    }

    #[test]
    fn vector_min_i8_signed() {
        // bytes: 0x80(-128), 0x7F(127), 0xFF(-1), 0x01(1)
        let a = [0x01_FF_7F_80_u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x02_FE_00_81_u32, 0, 0, 0, 0, 0, 0, 0]; // -127, 0, -2, 2
        let r = VectorAlu::vector_min(&a, &b, ElementType::Int8);
        // min(-128,-127)=-128, min(127,0)=0, min(-1,-2)=-2, min(1,2)=1
        assert_eq!(r[0], 0x01_FE_00_80);
    }

    // -- vector_shift_left --

    #[test]
    fn shift_left_i32() {
        let src = [1, 0, 0, 0, 0, 0, 0, 0];
        let shift = [4, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_shift_left(&src, &shift, ElementType::Int32);
        assert_eq!(r[0], 16); // 1 << 4
    }

    #[test]
    fn shift_left_i16() {
        // lo=1, hi=1; shift lo by 2, hi by 4
        let src = [0x0001_0001_u32, 0, 0, 0, 0, 0, 0, 0];
        let shift = [0x0004_0002_u32, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_shift_left(&src, &shift, ElementType::Int16);
        // lo: 1<<2=4, hi: 1<<4=16
        assert_eq!(r[0], 0x0010_0004);
    }

    #[test]
    fn shift_left_i32_wraps_at_5_bits() {
        // Shift amount masked to 5 bits: 32 & 0x1F = 0
        let src = [0xFF, 0, 0, 0, 0, 0, 0, 0];
        let shift = [32, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_shift_left(&src, &shift, ElementType::Int32);
        assert_eq!(r[0], 0xFF); // shift by 0 (32 & 0x1F)
    }

    // -- vector_shift_right_logical --

    #[test]
    fn shift_right_logical_i32() {
        let src = [0x8000_0000, 0, 0, 0, 0, 0, 0, 0]; // MSB set
        let shift = [1, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_shift_right_logical(&src, &shift, ElementType::Int32);
        assert_eq!(r[0], 0x4000_0000); // zero-filled from left
    }

    // -- vector_shift_right_arith --

    #[test]
    fn shift_right_arith_i32_sign_extends() {
        let src = [0x8000_0000, 0, 0, 0, 0, 0, 0, 0]; // -2147483648
        let shift = [1, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_shift_right_arith(&src, &shift, ElementType::Int32);
        assert_eq!(r[0], 0xC000_0000); // sign bit propagated
    }

    #[test]
    fn shift_right_arith_i16_sign_extends() {
        // lo=0x8000(-32768), hi=0x0001(1)
        let src = [0x0001_8000_u32, 0, 0, 0, 0, 0, 0, 0];
        let shift = [0x0001_0001_u32, 0, 0, 0, 0, 0, 0, 0]; // shift both by 1
        let r = VectorAlu::vector_shift_right_arith(&src, &shift, ElementType::Int16);
        // lo: -32768 >> 1 = -16384 = 0xC000, hi: 1 >> 1 = 0
        assert_eq!(r[0], 0x0000_C000);
    }

    // -- vector_abs_gtz --

    #[test]
    fn abs_gtz_i32() {
        let src = [(-5i32) as u32, 5, 0, i32::MIN as u32, 0, 0, 0, 0];
        let r = VectorAlu::vector_abs_gtz(&src, ElementType::Int32);
        assert_eq!(r[0], 5);
        assert_eq!(r[1], 5);
        assert_eq!(r[2], 0);
        // wrapping_abs of i32::MIN = i32::MIN (wraps)
        assert_eq!(r[3], i32::MIN as u32);
    }

    #[test]
    fn abs_gtz_f32() {
        let src = [(-3.0f32).to_bits(), 3.0f32.to_bits(), 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_abs_gtz(&src, ElementType::Float32);
        assert_eq!(f32::from_bits(r[0]), 3.0);
        assert_eq!(f32::from_bits(r[1]), 3.0);
    }

    #[test]
    fn abs_gtz_unsigned_is_identity() {
        let src = [42, 0, 0xFF, 0, 0, 0, 0, 0];
        assert_eq!(VectorAlu::vector_abs_gtz(&src, ElementType::UInt32), src);
    }

    // -- vector_neg_gtz --

    #[test]
    fn neg_gtz_i32() {
        let src = [5, (-5i32) as u32, 0, i32::MIN as u32, 0, 0, 0, 0];
        let r = VectorAlu::vector_neg_gtz(&src, ElementType::Int32);
        assert_eq!(r[0] as i32, -5);
        assert_eq!(r[1] as i32, 5);
        assert_eq!(r[2], 0);
        // wrapping_neg of i32::MIN = i32::MIN
        assert_eq!(r[3], i32::MIN as u32);
    }

    #[test]
    fn neg_gtz_f32_flips_sign_bit() {
        let src = [1.0f32.to_bits(), (-2.0f32).to_bits(), 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_neg_gtz(&src, ElementType::Float32);
        assert_eq!(f32::from_bits(r[0]), -1.0);
        assert_eq!(f32::from_bits(r[1]), 2.0);
    }

    // -- vector_neg_ltz (bitwise NOT) --

    #[test]
    fn neg_ltz_bitwise_not() {
        let src = [0u32, 0xFFFF_FFFF, 0x5555_5555, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_neg_ltz(&src, ElementType::Int32);
        assert_eq!(r[0], 0xFFFF_FFFF);
        assert_eq!(r[1], 0);
        assert_eq!(r[2], 0xAAAA_AAAA);
    }

    // -- vector_negate --

    #[test]
    fn negate_i16() {
        // lo=1, hi=0xFFFF(-1)
        let src = [0xFFFF_0001_u32, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_negate(&src, ElementType::Int16);
        // lo: -1 = 0xFFFF, hi: 1 = 0x0001
        assert_eq!(r[0], 0x0001_FFFF);
    }

    #[test]
    fn negate_bf16_flips_both_sign_bits() {
        // bf16 negate = XOR with 0x8000_8000
        let src = [0x3F80_3F80_u32, 0, 0, 0, 0, 0, 0, 0]; // two +1.0 bf16
        let r = VectorAlu::vector_negate(&src, ElementType::BFloat16);
        assert_eq!(r[0], 0xBF80_BF80); // both now negative
    }

    // -- vector_addsub --

    #[test]
    fn addsub_i32_mixed() {
        let s1 = [10, 20, 0, 0, 0, 0, 0, 0];
        let s2 = [3,  5, 0, 0, 0, 0, 0, 0];
        // sel bit 0: lane 0 subtracts, lane 1 adds
        let sel = [1, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_addsub(&s1, &s2, &sel, ElementType::Int32);
        assert_eq!(r[0], 7);  // 10 - 3
        assert_eq!(r[1], 25); // 20 + 5
    }

    // -- vector_bneg_gtz (conditional negate) --

    #[test]
    fn bneg_gtz_i32_conditional() {
        let cmp = [1, 0, (-1i32) as u32, 0, 0, 0, 0, 0]; // >0, =0, <0
        let val = [10, 20, 30, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_bneg_gtz(&cmp, &val, ElementType::Int32);
        assert_eq!(r[0] as i32, -10); // cmp > 0 -> negate
        assert_eq!(r[1], 20);         // cmp == 0 -> no change
        assert_eq!(r[2], 30);         // cmp < 0 -> no change
    }

    // -- vector_maxdiff_lt --

    #[test]
    fn maxdiff_lt_unsigned_saturates_at_zero() {
        let a = [5, 3, 10, 0, 0, 0, 0, 0];
        let b = [3, 5, 10, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_maxdiff_lt(&a, &b, ElementType::UInt32);
        assert_eq!(r[0], 2);  // 5 - 3
        assert_eq!(r[1], 0);  // 3 - 5 saturates to 0
        assert_eq!(r[2], 0);  // 10 - 10
    }

    #[test]
    fn maxdiff_lt_signed_clamps() {
        let a = [(-10i32) as u32, 10, 0, 0, 0, 0, 0, 0];
        let b = [5, (-5i32) as u32, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_maxdiff_lt(&a, &b, ElementType::Int32);
        assert_eq!(r[0], 0);  // -10 - 5 < 0 -> clamp to 0
        assert_eq!(r[1], 15); // 10 - (-5) = 15
    }

    // -- vector_neg_add --

    #[test]
    fn neg_add_i32() {
        let a = [10, (-5i32) as u32, 0, 0, 0, 0, 0, 0];
        let b = [3, 3, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_neg_add(&a, &b, ElementType::Int32);
        assert_eq!(r[0] as i32, -7); // -10 + 3
        assert_eq!(r[1] as i32, 8);  // 5 + 3
    }

    #[test]
    fn neg_add_u32_is_sub_reversed() {
        let a = [10, 3, 0, 0, 0, 0, 0, 0];
        let b = [3, 10, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_neg_add(&a, &b, ElementType::UInt32);
        assert_eq!(r[0], 3u32.wrapping_sub(10)); // b - a
        assert_eq!(r[1], 7); // 10 - 3
    }

    // -- vector_sub_lt / vector_sub_ge (both alias to unconditional sub) --

    #[test]
    fn sub_lt_and_sub_ge_same_result() {
        let a = [100, 50, 0, 0, 0, 0, 0, 0];
        let b = [30, 60, 0, 0, 0, 0, 0, 0];
        let r_lt = VectorAlu::vector_sub_lt(&a, &b, ElementType::Int32);
        let r_ge = VectorAlu::vector_sub_ge(&a, &b, ElementType::Int32);
        assert_eq!(r_lt, r_ge);
        assert_eq!(r_lt[0], 70);
        assert_eq!(r_lt[1] as i32, -10);
    }

    // -- vector_floor_bf16_to_s32 --

    #[test]
    fn floor_bf16_to_s32_positive() {
        // bf16 1.5 = 0x3FC0 (f32 = 0x3FC00000)
        // Pack into src: word 0 lo=0x3FC0, word 0 hi=0x4000 (2.0)
        let src = [0x4000_3FC0_u32, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_floor_bf16_to_s32(&src, 0);
        // floor(1.5 * 2^0) = 1, floor(2.0 * 2^0) = 2
        assert_eq!(r[0] as i32, 1);
        assert_eq!(r[1] as i32, 2);
    }

    #[test]
    fn floor_bf16_to_s32_with_shift() {
        // bf16 1.0 = 0x3F80
        let src = [0x0000_3F80_u32, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_floor_bf16_to_s32(&src, 3);
        // floor(1.0 * 2^3) = floor(8.0) = 8
        assert_eq!(r[0] as i32, 8);
    }

    #[test]
    fn floor_bf16_to_s32_negative() {
        // bf16 -1.5 = 0xBFC0
        let src = [0x0000_BFC0_u32, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_floor_bf16_to_s32(&src, 0);
        // floor(-1.5) = -2
        assert_eq!(r[0] as i32, -2);
    }

    #[test]
    fn floor_bf16_to_s32_nan_saturates_to_max() {
        // bf16 NaN = 0x7FC0 (or any value with exp=0xFF, mantissa!=0)
        let src = [0x0000_7FC0_u32, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_floor_bf16_to_s32(&src, 0);
        assert_eq!(r[0] as i32, i32::MAX);
    }

    /// VADD_F: NaN + NaN should produce canonical NaN with mantissa = 1.
    ///
    /// Negative s16 values sign-extended to s32 produce f32 NaN bit patterns
    /// (exponent = 0xFF, mantissa != 0).  The hardware canonical NaN is
    /// 0x7F800001 (mantissa = 1, sign = 0).  Verified against real NPU1 HW.
    #[test]
    fn test_vadd_f_nan_canonical() {
        use crate::tablegen::decoder_ffi::AccumWidth;
        let mut ctx = make_ctx();

        // Load NaN values: s16 = -1 -> s32 = 0xFFFFFFFF -> f32: NaN (exp=0xFF).
        // Pack two NaN values per u64 accumulator lane.
        let nan_s32 = 0xFFFFFFFFu32;
        let nan_u64 = (nan_s32 as u64) | ((nan_s32 as u64) << 32);
        let acc = [nan_u64; 8];
        ctx.accumulator.write(0, acc);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Accumulate)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(0));
        op.accum_width = Some(AccumWidth::Half);
        ctx.scalar.write(0, 0);

        VectorAlu::execute(&op, &mut ctx);

        let result = ctx.accumulator.read(0);
        // Each u64 lane should hold two copies of canonical NaN = 0x7F800001.
        let expected = 0x7F800001_7F800001u64;
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, expected,
                "acc lane {} should be canonical NaN pair, got {:#018x}", i, v);
        }
    }

    /// VADD_F: float accumulator add should flush denormals to zero.
    ///
    /// The test harness loads s16 data via `vups.s32.s16` then runs
    /// `vadd.f bml0, bml0, bml0, r0`.  Small positive s16 values
    /// (1..127) sign-extend to s32 = denormalized f32 bit patterns.
    /// FTZ should flush them to +0.0, so vadd.f(0, 0) = 0.
    /// SRS should then produce zero.
    #[test]
    fn test_vadd_f_denormal_ftz() {
        use crate::tablegen::decoder_ffi::AccumWidth;
        let mut ctx = make_ctx();

        // Simulate UPS: sign-extend 16 s16 values to s32, pack 2 per u64.
        // Values 0..15: all small positive -> denormalized f32 patterns.
        let src = [0x0002_0001u32, 0x0004_0003, 0x0006_0005, 0x0008_0007,
                    0x000A_0009, 0x000C_000B, 0x000E_000D, 0x0010_000F];
        let acc = super::super::vector_ups::ups_vector_to_acc(
            &src, 0, ElementType::Int16, ElementType::Int32,
        );
        ctx.accumulator.write(0, acc);

        // Build a vadd.f operation: bml0 = bml0 + bml0.
        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Accumulate)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(0)); // config = 0
        op.accum_width = Some(AccumWidth::Half);

        // Config register r0 = 0.
        ctx.scalar.write(0, 0);

        VectorAlu::execute(&op, &mut ctx);

        // Read back via SRS s16.s32 with shift=0.
        let result_acc = ctx.accumulator.read(0);
        // All values were denormalized f32 -> FTZ to 0 -> add(0,0) = 0.
        for (i, &v) in result_acc.iter().enumerate() {
            assert_eq!(v, 0, "acc lane {} should be 0 after FTZ, got {:#018x}", i, v);
        }
    }
}
