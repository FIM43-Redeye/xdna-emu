//! Arithmetic operations for the vector ALU.
//!
//! Extracted from vector.rs -- these are pure computational functions
//! with no internal dependencies on dispatch or helper code.

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use xdna_archspec::aie2::isa::SemanticOp;

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
            ElementType::Int32
            | ElementType::UInt32
            | ElementType::Int64
            | ElementType::UInt64
            | ElementType::Float32 => {
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
                    let r_lo = if sel_lo != 0 {
                        a_lo.wrapping_sub(b_lo)
                    } else {
                        a_lo.wrapping_add(b_lo)
                    };
                    let r_hi = if sel_hi != 0 {
                        a_hi.wrapping_sub(b_hi)
                    } else {
                        a_hi.wrapping_add(b_hi)
                    };
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
            }
        }

        result
    }

    /// Float (fp32 / bf16) add/sub on x/w vector registers is not modeled: the
    /// executors that reach `vector_add` / `vector_sub` / `vector_sub_{lt,ge}`
    /// call this first and report the op unhandled when it returns true.
    /// No AIE2 encoding can reach it (float add/sub exists only on accumulators,
    /// `vadd.f` and friends, which `execute_accumulate` models from silicon; see
    /// `float_add_sub_is_reachable_only_through_the_accumulator_family`), so this
    /// is a guard against a future decode change, not a known gap.
    pub(super) fn float_addsub_not_modeled(op: &SlotOp, et: ElementType) -> bool {
        if matches!(et, ElementType::Float32 | ElementType::BFloat16) {
            log::warn!(
                "[NOT MODELED] {:?} add/sub ({:?}): awaiting clean-room derivation",
                et,
                op.encoding_name
            );
            true
        } else {
            false
        }
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
            ElementType::Float32 | ElementType::BFloat16 => {
                unreachable!("float add/sub is not modeled; executors gate it via float_addsub_not_modeled")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
            ElementType::Float32 | ElementType::BFloat16 => {
                unreachable!("float add/sub is not modeled; executors gate it via float_addsub_not_modeled")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
                // AIE2 NaN semantics: when either operand is NaN, always return s1.
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = if fa.is_nan() || fb.is_nan() {
                        a[i]
                    } else {
                        fa.min(fb).to_bits()
                    };
                }
            }
            ElementType::BFloat16 => {
                // VMIN_GE mux: select s2 when (s1 >= s2) under the silicon
                // sign-magnitude ordering, else s1. NaN makes the compare
                // false, propagating s1 (NOT IEEE 754 minNum); -0 sorts
                // strictly below +0, so min(-0, +0) = -0 (silicon-verified,
                // vector fuzzer seed 1433 max counterpart).
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = if Self::bf16_cmp_ge(a_lo, b_lo) { b_lo } else { a_lo };
                    let r_hi = if Self::bf16_cmp_ge(a_hi, b_hi) { b_hi } else { a_hi };
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
                // AIE2 NaN semantics: when either operand is NaN, always return s1.
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = if fa.is_nan() || fb.is_nan() {
                        a[i]
                    } else {
                        fa.max(fb).to_bits()
                    };
                }
            }
            ElementType::BFloat16 => {
                // VMAX_LT mux: select s2 when (s1 < s2) under the silicon
                // sign-magnitude ordering, else s1. NaN makes the compare
                // false, propagating s1 (NOT IEEE 754 maxNum); -0 sorts
                // strictly below +0, so max(-0, +0) = +0 (silicon-verified,
                // vector fuzzer seed 1433: numeric f32 max kept -0).
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo = if Self::bf16_cmp_lt(a_lo, b_lo) { b_lo } else { a_lo };
                    let r_hi = if Self::bf16_cmp_lt(a_hi, b_hi) { b_hi } else { a_hi };
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
            }
        }

        result
    }

    /// Vector logical left shift: each lane is shifted left by the corresponding shift amount.
    pub(super) fn vector_shift_left(src: &[u32; 8], shift: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];
        match elem_type {
            ElementType::Int32
            | ElementType::UInt32
            | ElementType::Int64
            | ElementType::UInt64
            | ElementType::Float32 => {
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
            }
        }
        result
    }

    /// Vector logical right shift: unsigned shift right.
    pub(super) fn vector_shift_right_logical(
        src: &[u32; 8],
        shift: &[u32; 8],
        elem_type: ElementType,
    ) -> [u32; 8] {
        let mut result = [0u32; 8];
        match elem_type {
            ElementType::Int32
            | ElementType::UInt32
            | ElementType::Int64
            | ElementType::UInt64
            | ElementType::Float32 => {
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
            }
        }
        result
    }

    /// Vector arithmetic right shift: signed shift right (preserves sign bit).
    pub(super) fn vector_shift_right_arith(
        src: &[u32; 8],
        shift: &[u32; 8],
        elem_type: ElementType,
    ) -> [u32; 8] {
        let mut result = [0u32; 8];
        match elem_type {
            ElementType::Int32
            | ElementType::UInt32
            | ElementType::Int64
            | ElementType::UInt64
            | ElementType::Float32 => {
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
                    result[i] = if c > 0 {
                        (s1[i] as i32).wrapping_neg() as u32
                    } else {
                        s1[i]
                    };
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = a[i].wrapping_sub(b[i]);
                }
            }
            ElementType::Float32 | ElementType::BFloat16 => {
                unreachable!("float add/sub is not modeled; executors gate it via float_addsub_not_modeled")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
            ElementType::Int4 | ElementType::UInt4 => {
                unreachable!("4-bit element types occur only as pack/unpack operands, never in vector_arith")
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
        let mut result = [0u32; 8];
        // Hardware interprets shift as signed 6-bit (-32 to +31).
        let masked = (shift & 0x3F) as i8;
        let effective_shift = if masked >= 32 { masked - 64 } else { masked } as i32;
        let scale = (2.0f64).powi(effective_shift);
        for i in 0..8 {
            let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
            // Silicon (NPU1 Phoenix, vec_vfloor_bf16_denorm, 2026-06-10) does NOT
            // flush bf16 denormals before floor: a tiny negative denormal floors to
            // -1, not 0. This is the datapath the compiled VFLOOR kernel uses (wide
            // path, quarter-acc source) -- distinct from vector_convert's branch.
            let f = Self::bf16_to_f32(bf16);
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
    pub(super) fn execute_add(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        // Two variants:
        // 1. VADD_elem: simple vector add (2 sources)
        // 2. VADDSUB: conditional add/subtract (2 vector + 1 scalar source)
        //    vaddsub.8 $d, $s1, $s2, $sel
        //    d[i] = (sel[i] & 1) ? s1[i] - s2[i] : s1[i] + s2[i]
        //    sel is a SCALAR register (bitmask), not a vector.
        let has_scalar_sel = op.sources.iter().any(|s| matches!(s, Operand::ScalarReg(_)))
            && op.sources.iter().filter(|s| matches!(s, Operand::VectorReg(_))).count() == 2;

        if !has_scalar_sel && Self::float_addsub_not_modeled(op, et) {
            return false;
        }

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
                    let sel_reg = op
                        .sources
                        .iter()
                        .find_map(|s| {
                            if let Operand::ScalarReg(r) = s {
                                Some(*r)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    ctx.scalar_read(sel_reg + 1)
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
    pub(super) fn execute_neg(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
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
    pub(super) fn execute_shl(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        Self::execute_binary_elementwise(op, ctx, et, Self::vector_shift_left)
    }

    /// Execute vector logical shift right.
    pub(super) fn execute_srl(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        Self::execute_binary_elementwise(op, ctx, et, Self::vector_shift_right_logical)
    }

    /// Execute vector arithmetic shift right.
    pub(super) fn execute_sra(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        Self::execute_binary_elementwise(op, ctx, et, Self::vector_shift_right_arith)
    }

    /// Execute AbsGtz: absolute value + comparison flags (s > 0).
    pub(super) fn execute_abs_gtz(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
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
    pub(super) fn execute_neg_gtz(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if op.is_wide_vector {
            // Two-vector-source variant (VNEG_GTZ with select operand) uses fallback.
            let vec_source_count = op.sources.iter().filter(|s| matches!(s, Operand::VectorReg(_))).count();
            if vec_source_count >= 2 {
                // Two-source wide VNEG_GTZ: not yet implemented.
                return false;
            }
            let a = Self::get_wide_vec_source(op, ctx, 0);
            let result = Self::wide_element_wise_unary(&a, et, Self::vector_neg_gtz);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let vec_source_count = op.sources.iter().filter(|s| matches!(s, Operand::VectorReg(_))).count();

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
    pub(super) fn execute_neg_ltz(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
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

    /// Execute Sub: elementwise subtraction (float is not modeled).
    pub(super) fn execute_sub(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if Self::float_addsub_not_modeled(op, et) {
            return false;
        }
        Self::execute_binary_elementwise(op, ctx, et, Self::vector_sub)
    }

    /// Execute SubLt: subtraction + comparison flags (a < b).
    pub(super) fn execute_sub_lt(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if Self::float_addsub_not_modeled(op, et) {
            return false;
        }
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
    pub(super) fn execute_sub_ge(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if Self::float_addsub_not_modeled(op, et) {
            return false;
        }
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
    pub(super) fn execute_maxdiff_lt(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
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
        semantic: xdna_archspec::aie2::isa::SemanticOp,
    ) -> bool {
        let has_acc_source = op.sources.iter().any(|s| matches!(s, Operand::AccumReg(_)));

        if has_acc_source {
            // VADD/VSUB/VNEGADD/VNEGSUB (and their .f float forms):
            // accumulator-to-accumulator.
            Self::execute_acc_add_sub(op, ctx, matches!(et, ElementType::Float32 | ElementType::BFloat16));
        } else if matches!(semantic, xdna_archspec::aie2::isa::SemanticOp::NegAdd) {
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
    /// Handles integer VADD, VSUB, VNEGADD, VNEGSUB (two accumulator sources,
    /// cm-class 1024-bit or bm-class 512-bit) and the fp32 `.f` variants
    /// (`is_float`; bm only, 16 fp32 lanes, see `vector_matmul::bf16::acc_add_lane`).
    ///
    /// The config register (scalar operand) controls:
    /// - Bit  0: zero_acc1 (zero acc1 before operation)
    /// - Bit 10: shift16 (right-shift result by 16 bits; integer only)
    /// - Integer: bit 11 sub_acc1, bit 12 sub_acc2 (as decoded here historically;
    ///   TableGen's VecConf and the Chess-emitted constants put dynAcc0Neg and
    ///   dynAcc1Neg at bits 12 and 13, which the float decode uses -- the
    ///   integer decode is unverified, see docs/fidelity-gaps/vector-compute.md)
    /// - Float: bit 12 sub_acc1, bit 13 sub_acc2 (silicon-verified)
    ///
    /// Base operations (before config modifiers):
    /// - vadd:    dst = acc1 + acc2
    /// - vsub:    dst = acc1 - acc2
    /// - vnegadd: dst = -acc1 + acc2
    /// - vnegsub: dst = -acc1 - acc2
    pub(super) fn execute_acc_add_sub(op: &SlotOp, ctx: &mut ExecutionContext, is_float: bool) {
        // Collect the two AccumReg source indices.
        let mut acc_sources: Vec<u8> = Vec::new();
        for src in &op.sources {
            if let Operand::AccumReg(r) = src {
                acc_sources.push(*r);
            }
        }

        let acc1_reg = if !acc_sources.is_empty() {
            acc_sources[0]
        } else {
            0
        };
        let acc2_reg = if acc_sources.len() >= 2 {
            acc_sources[1]
        } else {
            acc1_reg
        };

        // Read config register.
        let conf = Self::get_config_register(op, ctx).unwrap_or(0);
        let zero_acc1 = (conf & 1) != 0;
        let shift16 = !is_float && ((conf >> 10) & 1) != 0;
        // One VecConf layout for integer and float (AIE2InstrPatterns.td):
        // bit 11 is dynMulNeg, 12 dynAcc0Neg, 13 dynAcc1Neg. Silicon confirms
        // 12/13 on the float forms (vec_acc_addsub_f32).
        let sub_acc1 = ((conf >> 12) & 1) != 0;
        let sub_acc2 = ((conf >> 13) & 1) != 0;

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
            SemanticOp::Accumulate => (false, false), // VADD
            SemanticOp::AccumSub => (false, true),    // VSUB
            SemanticOp::AccumNegAdd => (true, true),  // VNEGADD
            SemanticOp::AccumNegSub => (true, false), // VNEGSUB
            SemanticOp::NegAdd => {
                // Runtime mnemonic dispatch -- both vnegadd and vnegsub map to
                // NegAdd.  Distinguish via encoding name.
                if enc_lower.contains("negsub") {
                    (true, false) // VNEGSUB
                } else {
                    (true, true) // VNEGADD
                }
            }
            _ => (false, false),
        };
        // Wide (cm, 1024-bit) vs narrow (bm, 512-bit) accumulator.
        // Use AccumWidth metadata from decoder when available (structural signal).
        // Fallback: cm registers always use even base indices, so odd index implies bm.
        let dst_reg = Self::get_acc_dest(op);
        let is_wide = match op.accum_width {
            Some(crate::interpreter::decode::register_map::AccumWidth::Full) => true,
            Some(crate::interpreter::decode::register_map::AccumWidth::Half)
            | Some(crate::interpreter::decode::register_map::AccumWidth::QuarterLow)
            | Some(crate::interpreter::decode::register_map::AccumWidth::QuarterHigh) => false,
            None => {
                // Legacy fallback: if any index is odd, it must be a bm half.
                let any_odd = acc_sources.iter().any(|r| r % 2 != 0) || dst_reg % 2 != 0;
                !any_odd
            }
        };

        // Compute effective signs: md bits XOR config bits (mirrors hardware).
        let negate_acc1 = md1 ^ sub_acc1;
        let negate_acc2 = md2 ^ sub_acc2;

        // II_VADDMAC / II_VADDMACf read their accumulator sources at operand
        // cycle 3, not at issue (AIE2Schedule.td); Peano bundles the vconv
        // producing a source WITH the vadd that consumes it. Defer the source
        // read+compute to the pending-acc-add queue; all control bits are
        // resolved here at issue. The 32-bit lane math lives in
        // `ExecutionContext::acc_add_sub_lane_pair`.
        ctx.queue_pending_acc_add(
            acc1_reg,
            acc2_reg,
            dst_reg,
            negate_acc1,
            negate_acc2,
            zero_acc1,
            shift16,
            is_wide,
            is_float,
            false,
        );
    }

    /// Accumulator negate: dst = -src.
    ///
    /// Handles VNEG on both bm (512-bit) and cm (1024-bit) accumulators.
    /// The config word controls zero_acc (zero the source before negation,
    /// producing 0). Float mode is detected from element_type (Float32/BFloat16):
    /// each u64 lane holds two fp32 values (bits [31:0] and bits [63:32]) and the
    /// op goes through the deferred float adder, not the integer negate.
    pub(super) fn execute_acc_negate(op: &SlotOp, ctx: &mut ExecutionContext) {
        let acc_reg = op
            .sources
            .iter()
            .find_map(|s| match s {
                Operand::AccumReg(r) => Some(*r),
                _ => None,
            })
            .unwrap_or(0);
        let dst_reg = Self::get_acc_dest(op);

        let conf = Self::get_config_register(op, ctx).unwrap_or(0);
        let zero_acc = (conf & 1) != 0;
        // Float mode from element_type (set at build time from mnemonic ".f" suffix).
        let is_float = matches!(op.element_type, Some(ElementType::Float32) | Some(ElementType::BFloat16));

        // Wide (cm) vs narrow (bm) -- same detection as execute_acc_add_sub.
        // Use AccumWidth metadata from decoder when available (structural signal).
        let is_wide = match op.accum_width {
            Some(crate::interpreter::decode::register_map::AccumWidth::Full) => true,
            Some(crate::interpreter::decode::register_map::AccumWidth::Half)
            | Some(crate::interpreter::decode::register_map::AccumWidth::QuarterLow)
            | Some(crate::interpreter::decode::register_map::AccumWidth::QuarterHigh) => false,
            None => {
                let any_odd = acc_reg % 2 != 0 || dst_reg % 2 != 0;
                !any_odd
            }
        };

        if is_float {
            // vneg.f is the fp32 adder with one operand (see `acc_add_lane`); like
            // the adds it reads its source at operand cycle 3 and lands its
            // result at 6. Config bit 12 (dynAcc0Neg) un-negates it.
            let sub_acc1 = (conf >> 12) & 1 != 0;
            ctx.queue_pending_acc_add(
                acc_reg, acc_reg, dst_reg, !sub_acc1, false, zero_acc, false, is_wide, true, true,
            );
            return;
        }

        // Integer acc32 mode: independent 32-bit negation per lane half.
        let neg_pair = |word: u64| -> u64 {
            let lo = if zero_acc { 0i32 } else { word as i32 };
            let hi = if zero_acc { 0i32 } else { (word >> 32) as i32 };
            (lo.wrapping_neg() as u32 as u64) | ((hi.wrapping_neg() as u32 as u64) << 32)
        };
        if is_wide {
            let mut result = ctx.accumulator.read_wide(acc_reg);
            result.iter_mut().for_each(|w| *w = neg_pair(*w));
            ctx.accumulator.write_wide(dst_reg, result);
        } else {
            let mut result = ctx.accumulator.read(acc_reg);
            result.iter_mut().for_each(|w| *w = neg_pair(*w));
            ctx.accumulator.write(dst_reg, result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{ElementType, Operand, SlotIndex, SlotOp};
    use crate::interpreter::state::ExecutionContext;
    use xdna_archspec::aie2::isa::SemanticOp;

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
        assert_eq!(f32::from_bits(result[0]), 0.5); // min(1.0, 0.5)
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
        assert_eq!(f32::from_bits(result[0]), 1.0); // max(1.0, 0.5)
        assert_eq!(f32::from_bits(result[1]), -1.0); // max(-2.0, -1.0)
        assert_eq!(f32::from_bits(result[3]), -4.0); // max(-4.0, -5.0)
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

    /// bf16 min/max mux semantics on signed zero and NaN (silicon-verified,
    /// vector fuzzer seed 1433): -0 sorts strictly below +0, so
    /// max(-0,+0)=+0 and min(-0,+0)=-0 regardless of operand order; a NaN
    /// operand makes the compare false, propagating s1. Numeric f32 min/max
    /// collapses the zeros and got the seed-1433 lane wrong.
    #[test]
    fn vector_max_min_bf16_signed_zero_and_nan() {
        let nz = 0x8000u32; // -0.0 bf16
        let pz = 0x0000u32; // +0.0 bf16
        let nan = 0x7FC0u32;
        let one = 0x3F80u32;
        // lane lo of word 0; hi half unused (zeros compare equal -> s1).
        let a = [nz, pz, nan, one, 0, 0, 0, 0];
        let b = [pz, nz, one, nan, 0, 0, 0, 0];
        let max = VectorAlu::vector_max(&a, &b, ElementType::BFloat16);
        assert_eq!(max[0] & 0xFFFF, pz, "max(-0,+0)=+0");
        assert_eq!(max[1] & 0xFFFF, pz, "max(+0,-0)=+0");
        assert_eq!(max[2] & 0xFFFF, nan, "max(NaN,1) propagates s1");
        assert_eq!(max[3] & 0xFFFF, one, "max(1,NaN) propagates s1");
        let min = VectorAlu::vector_min(&a, &b, ElementType::BFloat16);
        assert_eq!(min[0] & 0xFFFF, nz, "min(-0,+0)=-0");
        assert_eq!(min[1] & 0xFFFF, nz, "min(+0,-0)=-0");
        assert_eq!(min[2] & 0xFFFF, nan, "min(NaN,1) propagates s1");
        assert_eq!(min[3] & 0xFFFF, one, "min(1,NaN) propagates s1");
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
        let s2 = [3, 5, 0, 0, 0, 0, 0, 0];
        // sel bit 0: lane 0 subtracts, lane 1 adds
        let sel = [1, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_addsub(&s1, &s2, &sel, ElementType::Int32);
        assert_eq!(r[0], 7); // 10 - 3
        assert_eq!(r[1], 25); // 20 + 5
    }

    // -- vector_bneg_gtz (conditional negate) --

    #[test]
    fn bneg_gtz_i32_conditional() {
        let cmp = [1, 0, (-1i32) as u32, 0, 0, 0, 0, 0]; // >0, =0, <0
        let val = [10, 20, 30, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_bneg_gtz(&cmp, &val, ElementType::Int32);
        assert_eq!(r[0] as i32, -10); // cmp > 0 -> negate
        assert_eq!(r[1], 20); // cmp == 0 -> no change
        assert_eq!(r[2], 30); // cmp < 0 -> no change
    }

    // -- vector_maxdiff_lt --

    #[test]
    fn maxdiff_lt_unsigned_saturates_at_zero() {
        let a = [5, 3, 10, 0, 0, 0, 0, 0];
        let b = [3, 5, 10, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_maxdiff_lt(&a, &b, ElementType::UInt32);
        assert_eq!(r[0], 2); // 5 - 3
        assert_eq!(r[1], 0); // 3 - 5 saturates to 0
        assert_eq!(r[2], 0); // 10 - 10
    }

    #[test]
    fn maxdiff_lt_signed_clamps() {
        let a = [(-10i32) as u32, 10, 0, 0, 0, 0, 0, 0];
        let b = [5, (-5i32) as u32, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_maxdiff_lt(&a, &b, ElementType::Int32);
        assert_eq!(r[0], 0); // -10 - 5 < 0 -> clamp to 0
        assert_eq!(r[1], 15); // 10 - (-5) = 15
    }

    // -- vector_neg_add --

    #[test]
    fn neg_add_i32() {
        let a = [10, (-5i32) as u32, 0, 0, 0, 0, 0, 0];
        let b = [3, 3, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_neg_add(&a, &b, ElementType::Int32);
        assert_eq!(r[0] as i32, -7); // -10 + 3
        assert_eq!(r[1] as i32, 8); // 5 + 3
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

    // -- Float add/sub is not modeled: the executors report the op unhandled --

    #[test]
    fn test_float_add_sub_not_modeled() {
        for (sem, et) in [
            (SemanticOp::Add, ElementType::Float32),
            (SemanticOp::Add, ElementType::BFloat16),
            (SemanticOp::Sub, ElementType::Float32),
            (SemanticOp::Sub, ElementType::BFloat16),
        ] {
            let mut ctx = make_ctx();
            ctx.vector.write(0, [0x3F80_0000; 8]);
            ctx.vector.write(1, [0x3F80_0000; 8]);
            ctx.vector.write(2, [0xDEAD_BEEF; 8]);
            let op = SlotOp::from_semantic(SlotIndex::Vector, sem)
                .as_vector(et)
                .with_dest(Operand::VectorReg(2))
                .with_source(Operand::VectorReg(0))
                .with_source(Operand::VectorReg(1));
            assert!(!VectorAlu::execute(&op, &mut ctx), "{sem:?} {et:?} must be unhandled");
            assert_eq!(ctx.vector.read(2), [0xDEAD_BEEF; 8], "{sem:?} {et:?} must not write");
        }
    }

    /// No TableGen encoding pairs a float element type with the vector-register
    /// Add/Sub/SubLt/SubGe semantics (AIE2 has no float VADD/VSUB on x/w
    /// registers; float add/sub is the accumulator family), so
    /// `float_addsub_not_modeled` is unreachable from decoded code. The float
    /// accumulator family is exactly the five mnemonics the silicon capture covers.
    #[test]
    fn float_add_sub_is_reachable_only_through_the_accumulator_family() {
        let out = xdna_archspec::aie2::isa::load_from_generated();
        let is_float = |e: &xdna_archspec::aie2::isa::InstrEncoding| {
            matches!(e.element_type, Some(ElementType::Float32) | Some(ElementType::BFloat16))
        };
        let mut vector_reg_hits = Vec::new();
        let mut acc_family = Vec::new();
        for encs in out.encodings_by_slot.values() {
            for e in encs.iter().filter(|e| is_float(e)) {
                match e.semantic {
                    Some(SemanticOp::Add | SemanticOp::Sub | SemanticOp::SubLt | SemanticOp::SubGe) => {
                        vector_reg_hits.push(e.mnemonic.clone())
                    }
                    Some(
                        SemanticOp::Accumulate
                        | SemanticOp::AccumSub
                        | SemanticOp::AccumNegAdd
                        | SemanticOp::AccumNegSub
                        | SemanticOp::Neg,
                    ) => acc_family.push(e.mnemonic.clone()),
                    _ => {}
                }
            }
        }
        acc_family.sort();
        assert!(vector_reg_hits.is_empty(), "float vector-register add/sub encodings: {vector_reg_hits:?}");
        assert_eq!(acc_family, ["vadd.f", "vneg.f", "vnegadd.f", "vnegsub.f", "vsub.f"]);
    }

    /// The integer adder reads the same VecConf layout as the float one
    /// (AIE2InstrPatterns.td): bit 12 negates acc1, bit 13 negates acc2, and
    /// bit 11 (dynMulNeg) does not touch an add.
    #[test]
    fn int_acc_add_negation_bits_follow_vecconf() {
        use crate::interpreter::decode::register_map::AccumWidth;
        let run = |conf: u32| -> i32 {
            let mut ctx = make_ctx();
            let pack = |v: u32| -> [u64; 8] { [v as u64 | (v as u64) << 32; 8] };
            ctx.accumulator.write(2, pack(5));
            ctx.accumulator.write(4, pack(3));
            ctx.scalar.write(0, conf);
            let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Accumulate)
                .as_vector(ElementType::Int32)
                .with_dest(Operand::AccumReg(6))
                .with_source(Operand::AccumReg(2))
                .with_source(Operand::AccumReg(4))
                .with_source(Operand::ScalarReg(0));
            op.accum_width = Some(AccumWidth::Half);
            assert!(VectorAlu::execute(&op, &mut ctx));
            ctx.bundle_seq += 4;
            ctx.process_pending_acc_adds();
            ctx.cycles += 10;
            ctx.commit_pending_writes();
            ctx.accumulator.read(6)[0] as u32 as i32
        };
        assert_eq!(run(0), 8);
        assert_eq!(run(1 << 11), 8);
        assert_eq!(run(1 << 12), -2);
        assert_eq!(run(1 << 13), 2);
    }

    /// Issue a float accumulator op, advance to its source-read bundle, and
    /// return the destination bm register (16 fp32 lanes, two per u64 word).
    fn run_float_acc_op(sem: SemanticOp, conf: u32, acc1: [u32; 16], acc2: [u32; 16]) -> [u32; 16] {
        use crate::interpreter::decode::register_map::AccumWidth;
        let pack = |l: [u32; 16]| -> [u64; 8] {
            std::array::from_fn(|j| l[2 * j] as u64 | (l[2 * j + 1] as u64) << 32)
        };
        let mut ctx = make_ctx();
        ctx.accumulator.write(2, pack(acc1)); // bml1
        ctx.accumulator.write(4, pack(acc2)); // bml2
        ctx.scalar.write(0, conf);
        let mut op = SlotOp::from_semantic(SlotIndex::Vector, sem)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::AccumReg(6))
            .with_source(Operand::AccumReg(2))
            .with_source(Operand::AccumReg(4))
            .with_source(Operand::ScalarReg(0));
        op.accum_width = Some(AccumWidth::Half);
        let before = ctx.accumulator.read(6);
        assert!(VectorAlu::execute(&op, &mut ctx));
        ctx.bundle_seq += 4;
        ctx.process_pending_acc_adds();
        // The result lands six cycles after issue, not at the source read.
        assert_eq!(ctx.accumulator.read(6), before, "result must not be visible at the source read");
        ctx.cycles += 10;
        ctx.commit_pending_writes();
        let out = ctx.accumulator.read(6);
        std::array::from_fn(|i| (out[i / 2] >> (32 * (i % 2))) as u32)
    }

    const FP32_CONF: u32 = 0x1c;

    #[test]
    fn float_acc_forms_follow_the_silicon_signs() {
        let a = [0x3f80_0000u32; 16]; // 1.0
        let b = [0x4000_0000u32; 16]; // 2.0
        let f = |sem, conf| f32::from_bits(run_float_acc_op(sem, conf, a, b)[3]);
        assert_eq!(f(SemanticOp::Accumulate, FP32_CONF), 3.0);
        assert_eq!(f(SemanticOp::AccumSub, FP32_CONF), -1.0);
        assert_eq!(f(SemanticOp::AccumNegAdd, FP32_CONF), -3.0);
        assert_eq!(f(SemanticOp::AccumNegSub, FP32_CONF), 1.0);
        // Config bits 12 / 13 / 0 negate acc1 / negate acc2 / zero acc1.
        assert_eq!(f(SemanticOp::Accumulate, FP32_CONF | 1 << 12), 1.0);
        assert_eq!(f(SemanticOp::Accumulate, FP32_CONF | 1 << 13), -1.0);
        assert_eq!(f(SemanticOp::Accumulate, FP32_CONF | 1), 2.0);
    }

    #[test]
    fn float_acc_add_uses_both_lanes_of_each_word() {
        let mut a = [0x3f80_0000u32; 16];
        let mut b = [0x3f80_0000u32; 16];
        a[4] = 0x4000_0000;
        b[5] = 0x4040_0000;
        let r = run_float_acc_op(SemanticOp::Accumulate, FP32_CONF, a, b);
        assert_eq!(f32::from_bits(r[4]), 3.0);
        assert_eq!(f32::from_bits(r[5]), 4.0);
        assert_eq!(f32::from_bits(r[0]), 2.0);
    }

    #[test]
    fn float_vneg_is_not_a_bare_sign_flip() {
        use crate::interpreter::decode::register_map::AccumWidth;
        let mut ctx = make_ctx();
        let lanes: [u32; 4] = [0x007f_ffff, 0, 0x3f80_0000, 0x7fc0_0000];
        let mut acc = [0u64; 8];
        acc[0] = lanes[0] as u64 | (lanes[1] as u64) << 32;
        acc[1] = lanes[2] as u64 | (lanes[3] as u64) << 32;
        ctx.accumulator.write(2, acc);
        ctx.scalar.write(0, FP32_CONF);
        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Neg)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::AccumReg(4))
            .with_source(Operand::AccumReg(2))
            .with_source(Operand::ScalarReg(0));
        op.accum_width = Some(AccumWidth::Half);
        assert!(VectorAlu::execute(&op, &mut ctx));
        ctx.bundle_seq += 4;
        ctx.process_pending_acc_adds();
        ctx.cycles += 10;
        ctx.commit_pending_writes();
        let out = ctx.accumulator.read(4);
        let lane = |i: usize| (out[i / 2] >> (32 * (i % 2))) as u32;
        assert_eq!(lane(0), 0x8000_0000, "a denormal flushes to -0");
        assert_eq!(lane(1), 0, "+0 stays +0");
        assert_eq!(lane(2), 0xbf80_0000);
        assert_eq!(lane(3), 0xff80_0001, "NaN is re-encoded with its sign flipped");
    }
}
