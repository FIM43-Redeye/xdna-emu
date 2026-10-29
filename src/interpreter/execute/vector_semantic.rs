//! Vector semantic operations -- element-wise arithmetic, comparison, and bitwise.
//!
//! This module handles the "simple" vector operations that map directly to
//! per-element semantics: add, sub, mul, min, max, cmp, neg, and/or/xor/not,
//! shifts, and conditional compound ops. These are the operations that the
//! SemanticOp enum in tablegen/types.rs can represent.
//!
//! Complex vector operations (MAC, MatMul, SRS/UPS, Shuffle, Pack/Unpack)
//! remain in their respective modules.
//!
//! # Design
//!
//! A single public entry point [`execute_vector_semantic`] dispatches on
//! `SemanticOp` variant (from `op.semantic`). If the operation is handled,
//! it returns `true`; otherwise `false` so the caller can fall through to
//! the next handler.

use super::vector_dispatch::VectorAlu;
use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::{Bypass, ExecutionContext};
use xdna_archspec::aie2::isa::SemanticOp;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Execute a vector semantic operation.
///
/// Returns `true` if the operation was handled, `false` if it is not a
/// vector semantic op (caller should try other handlers).
pub fn execute_vector_semantic(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let Some(semantic) = op.semantic else {
        return false;
    };

    // Only handle vector operations
    if !op.is_vector {
        return false;
    }

    let et = op.element_type.unwrap_or(ElementType::Int32);

    match semantic {
        // -- Arithmetic --
        SemanticOp::Add => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_add(&a, &b, et);
            write_vector_dest(op, ctx, result);
            true
        }
        SemanticOp::Sub => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_sub(&a, &b, et);
            write_vector_dest(op, ctx, result);
            true
        }
        SemanticOp::Mul => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_mul(&a, &b, et);
            write_vector_dest(op, ctx, result);
            true
        }
        SemanticOp::Neg => {
            let src = get_vector_source(op, ctx, 0);
            let result = vector_neg(&src, et);
            write_vector_dest(op, ctx, result);
            true
        }

        // -- Comparison --
        SemanticOp::Cmp => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_cmp_eq(&a, &b, et);
            write_vector_dest(op, ctx, result);
            true
        }

        // -- Min / Max --
        SemanticOp::Min => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_min(&a, &b, et);
            write_vector_dest(op, ctx, result);
            true
        }
        SemanticOp::Max => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_max(&a, &b, et);
            write_vector_dest(op, ctx, result);
            true
        }

        // -- Bitwise --
        SemanticOp::And => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_and(&a, &b);
            write_vector_dest(op, ctx, result);
            true
        }
        SemanticOp::Or => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_or(&a, &b);
            write_vector_dest(op, ctx, result);
            true
        }
        SemanticOp::Xor => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_xor(&a, &b);
            write_vector_dest(op, ctx, result);
            true
        }
        SemanticOp::Not => {
            let src = get_vector_source(op, ctx, 0);
            let result = vector_not(&src);
            write_vector_dest(op, ctx, result);
            true
        }

        // -- Shifts --
        SemanticOp::Shl => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_shl(&a, &b, et);
            write_vector_dest(op, ctx, result);
            true
        }
        SemanticOp::Srl => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_srl(&a, &b, et);
            write_vector_dest(op, ctx, result);
            true
        }
        SemanticOp::Sra => {
            let (a, b) = get_two_vector_sources(op, ctx);
            let result = vector_sra(&a, &b, et);
            write_vector_dest(op, ctx, result);
            true
        }

        _ => false, // Not handled here -- fall through to VectorAlu
    }
}

// ---------------------------------------------------------------------------
// Operand helpers (moved from VectorAlu)
// ---------------------------------------------------------------------------

/// Read two vector source operands from a SlotOp.
pub(crate) fn get_two_vector_sources(op: &SlotOp, ctx: &ExecutionContext) -> ([u32; 8], [u32; 8]) {
    let a = get_vector_source(op, ctx, 0);
    let b = get_vector_source(op, ctx, 1);
    (a, b)
}

/// Read a single vector source operand by index, applying per-operand forwarding.
///
/// `idx` is the position in `op.sources` (same index space as `source_forward`),
/// so the resolved `(use_cycle, use_bypass)` pair aligns directly. Falls back to
/// the compute default (`ctx.vector.read`) when the op carries no itinerary data.
pub(crate) fn get_vector_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> [u32; 8] {
    let fwd = op.source_forward.get(idx).copied();
    op.sources.get(idx).map_or([0; 8], |src| read_vector_operand(src, ctx, fwd))
}

/// Interpret an operand as a vector register read with optional forwarding context.
///
/// When `fwd` is `Some((use_cycle, use_bypass))`, the read honors the resolved
/// bypass-network window via `read_with`; otherwise it uses the compute default.
pub(crate) fn read_vector_operand(
    operand: &Operand,
    ctx: &ExecutionContext,
    fwd: Option<(u8, Bypass)>,
) -> [u32; 8] {
    match operand {
        Operand::VectorReg(r) => match fwd {
            Some((uc, ub)) => ctx.vector.read_with(*r, uc, ub),
            None => ctx.vector.read(*r),
        },
        _ => [0; 8],
    }
}

/// Write a 256-bit result to the destination vector register.
pub(crate) fn write_vector_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: [u32; 8]) {
    if let Some(Operand::VectorReg(r)) = &op.dest {
        ctx.vector.write(*r, value);
    }
}

// ---------------------------------------------------------------------------
// BFloat16 conversion helpers
// ---------------------------------------------------------------------------

/// Convert BFloat16 (upper 16 bits of f32) to f32.
#[inline]
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to BFloat16 (truncate lower 16 bits).
#[inline]
pub(crate) fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

// ---------------------------------------------------------------------------
// Element-wise operation implementations
// ---------------------------------------------------------------------------

/// Vector addition by element type.
pub(crate) fn vector_add(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
    let mut result = [0u32; 8];

    match elem_type {
        ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
            for i in 0..8 {
                result[i] = a[i].wrapping_add(b[i]);
            }
        }
        ElementType::Float32 => {
            for i in 0..8 {
                let fa = f32::from_bits(a[i]);
                let fb = f32::from_bits(b[i]);
                result[i] = (fa + fb).to_bits();
            }
        }
        ElementType::Int16 | ElementType::UInt16 => {
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
            for i in 0..8 {
                let a_lo = bf16_to_f32((a[i] & 0xFFFF) as u16);
                let a_hi = bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                let b_lo = bf16_to_f32((b[i] & 0xFFFF) as u16);
                let b_hi = bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                let r_lo = f32_to_bf16(a_lo + b_lo);
                let r_hi = f32_to_bf16(a_hi + b_hi);
                result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
            }
        }
        ElementType::Int8 | ElementType::UInt8 => {
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
pub(crate) fn vector_sub(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
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
                let a_lo = bf16_to_f32((a[i] & 0xFFFF) as u16);
                let a_hi = bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                let b_lo = bf16_to_f32((b[i] & 0xFFFF) as u16);
                let b_hi = bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                let r_lo = f32_to_bf16(a_lo - b_lo);
                let r_hi = f32_to_bf16(a_hi - b_hi);
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
pub(crate) fn vector_mul(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
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
                let a_lo = bf16_to_f32((a[i] & 0xFFFF) as u16);
                let a_hi = bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                let b_lo = bf16_to_f32((b[i] & 0xFFFF) as u16);
                let b_hi = bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16);
                let r_lo = f32_to_bf16(a_lo * b_lo);
                let r_hi = f32_to_bf16(a_hi * b_hi);
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

/// Vector negation by element type.
///
/// Computes the two's complement (integer) or IEEE negation (float) of
/// each element.
pub(crate) fn vector_neg(a: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
    let mut result = [0u32; 8];

    match elem_type {
        ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
            for i in 0..8 {
                result[i] = (a[i] as i32).wrapping_neg() as u32;
            }
        }
        ElementType::Float32 => {
            for i in 0..8 {
                // Flip the sign bit
                result[i] = a[i] ^ 0x8000_0000;
            }
        }
        ElementType::Int16 | ElementType::UInt16 => {
            for i in 0..8 {
                let a_lo = (a[i] & 0xFFFF) as i16;
                let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                let r_lo = a_lo.wrapping_neg() as u16;
                let r_hi = a_hi.wrapping_neg() as u16;
                result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
            }
        }
        ElementType::BFloat16 => {
            for i in 0..8 {
                // Flip sign bit of each bf16 lane
                let lo = (a[i] & 0xFFFF) ^ 0x8000;
                let hi = ((a[i] >> 16) & 0xFFFF) ^ 0x8000;
                result[i] = lo | (hi << 16);
            }
        }
        ElementType::Int8 | ElementType::UInt8 => {
            for i in 0..8 {
                let mut r = 0u32;
                for j in 0..4 {
                    let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                    let r_byte = a_byte.wrapping_neg() as u8;
                    r |= (r_byte as u32) << (j * 8);
                }
                result[i] = r;
            }
        }
    }

    result
}

/// Vector equality comparison -- returns all-ones mask for equal lanes.
pub(crate) fn vector_cmp_eq(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
    let mut result = [0u32; 8];

    match elem_type {
        ElementType::Int32
        | ElementType::UInt32
        | ElementType::Int64
        | ElementType::UInt64
        | ElementType::Float32 => {
            for i in 0..8 {
                result[i] = if a[i] == b[i] { 0xFFFF_FFFF } else { 0 };
            }
        }
        ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
            for i in 0..8 {
                let a_lo = a[i] & 0xFFFF;
                let a_hi = (a[i] >> 16) & 0xFFFF;
                let b_lo = b[i] & 0xFFFF;
                let b_hi = (b[i] >> 16) & 0xFFFF;
                let r_lo = if a_lo == b_lo { 0xFFFF } else { 0 };
                let r_hi = if a_hi == b_hi { 0xFFFF } else { 0 };
                result[i] = r_lo | (r_hi << 16);
            }
        }
        ElementType::Int8 | ElementType::UInt8 => {
            for i in 0..8 {
                let mut r = 0u32;
                for j in 0..4 {
                    let a_byte = (a[i] >> (j * 8)) & 0xFF;
                    let b_byte = (b[i] >> (j * 8)) & 0xFF;
                    let mask = if a_byte == b_byte { 0xFF } else { 0 };
                    r |= mask << (j * 8);
                }
                result[i] = r;
            }
        }
    }

    result
}

/// Vector minimum by element type.
pub(crate) fn vector_min(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
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
            // VMIN_GE mux semantics: s2 when (s1 >= s2) sign-magnitude,
            // else s1 (NaN -> s1; -0 < +0). Mirrors VectorAlu::vector_min.
            for i in 0..8 {
                let a_lo = (a[i] & 0xFFFF) as u16;
                let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                let b_lo = (b[i] & 0xFFFF) as u16;
                let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                let r_lo = if VectorAlu::bf16_cmp_ge(a_lo, b_lo) {
                    b_lo
                } else {
                    a_lo
                };
                let r_hi = if VectorAlu::bf16_cmp_ge(a_hi, b_hi) {
                    b_hi
                } else {
                    a_hi
                };
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
pub(crate) fn vector_max(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
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
            // VMAX_LT mux semantics: s2 when (s1 < s2) sign-magnitude,
            // else s1 (NaN -> s1; -0 < +0). Mirrors VectorAlu::vector_max.
            for i in 0..8 {
                let a_lo = (a[i] & 0xFFFF) as u16;
                let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                let b_lo = (b[i] & 0xFFFF) as u16;
                let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                let r_lo = if VectorAlu::bf16_cmp_lt(a_lo, b_lo) {
                    b_lo
                } else {
                    a_lo
                };
                let r_hi = if VectorAlu::bf16_cmp_lt(a_hi, b_hi) {
                    b_hi
                } else {
                    a_hi
                };
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

/// Bitwise AND of two vectors (type-agnostic, operates on raw bits).
pub(crate) fn vector_and(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
    let mut result = [0u32; 8];
    for i in 0..8 {
        result[i] = a[i] & b[i];
    }
    result
}

/// Bitwise OR of two vectors.
pub(crate) fn vector_or(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
    let mut result = [0u32; 8];
    for i in 0..8 {
        result[i] = a[i] | b[i];
    }
    result
}

/// Bitwise XOR of two vectors.
pub(crate) fn vector_xor(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
    let mut result = [0u32; 8];
    for i in 0..8 {
        result[i] = a[i] ^ b[i];
    }
    result
}

/// Bitwise NOT of a vector.
pub(crate) fn vector_not(a: &[u32; 8]) -> [u32; 8] {
    let mut result = [0u32; 8];
    for i in 0..8 {
        result[i] = !a[i];
    }
    result
}

/// Vector left shift by element type.
///
/// Each element in `a` is shifted left by the corresponding element in `b`.
/// Shift amount is masked to the element width.
pub(crate) fn vector_shl(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
    let mut result = [0u32; 8];

    match elem_type {
        ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
            for i in 0..8 {
                let shift = (b[i] & 0x1F) as u32;
                result[i] = a[i].wrapping_shl(shift);
            }
        }
        ElementType::Float32 => {
            // Shift on floats is meaningless; treat as integer shift on bits
            for i in 0..8 {
                let shift = (b[i] & 0x1F) as u32;
                result[i] = a[i].wrapping_shl(shift);
            }
        }
        ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
            for i in 0..8 {
                let a_lo = (a[i] & 0xFFFF) as u16;
                let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                let s_lo = (b[i] & 0xF) as u32;
                let s_hi = ((b[i] >> 16) & 0xF) as u32;
                let r_lo = a_lo.wrapping_shl(s_lo);
                let r_hi = a_hi.wrapping_shl(s_hi);
                result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
            }
        }
        ElementType::Int8 | ElementType::UInt8 => {
            for i in 0..8 {
                let mut r = 0u32;
                for j in 0..4 {
                    let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                    let s = ((b[i] >> (j * 8)) & 0x7) as u32;
                    let r_byte = a_byte.wrapping_shl(s);
                    r |= (r_byte as u32) << (j * 8);
                }
                result[i] = r;
            }
        }
    }

    result
}

/// Vector logical right shift by element type.
pub(crate) fn vector_srl(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
    let mut result = [0u32; 8];

    match elem_type {
        ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
            for i in 0..8 {
                let shift = (b[i] & 0x1F) as u32;
                result[i] = a[i].wrapping_shr(shift);
            }
        }
        ElementType::Float32 => {
            for i in 0..8 {
                let shift = (b[i] & 0x1F) as u32;
                result[i] = a[i].wrapping_shr(shift);
            }
        }
        ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
            for i in 0..8 {
                let a_lo = (a[i] & 0xFFFF) as u16;
                let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                let s_lo = (b[i] & 0xF) as u32;
                let s_hi = ((b[i] >> 16) & 0xF) as u32;
                let r_lo = a_lo.wrapping_shr(s_lo);
                let r_hi = a_hi.wrapping_shr(s_hi);
                result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
            }
        }
        ElementType::Int8 | ElementType::UInt8 => {
            for i in 0..8 {
                let mut r = 0u32;
                for j in 0..4 {
                    let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                    let s = ((b[i] >> (j * 8)) & 0x7) as u32;
                    let r_byte = a_byte.wrapping_shr(s);
                    r |= (r_byte as u32) << (j * 8);
                }
                result[i] = r;
            }
        }
    }

    result
}

/// Vector arithmetic right shift by element type (sign-extending).
pub(crate) fn vector_sra(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
    let mut result = [0u32; 8];

    match elem_type {
        ElementType::Int32 | ElementType::Int64 => {
            for i in 0..8 {
                let shift = (b[i] & 0x1F) as u32;
                result[i] = (a[i] as i32).wrapping_shr(shift) as u32;
            }
        }
        ElementType::UInt32 | ElementType::UInt64 => {
            // Arithmetic shift on unsigned is same as logical shift
            for i in 0..8 {
                let shift = (b[i] & 0x1F) as u32;
                result[i] = a[i].wrapping_shr(shift);
            }
        }
        ElementType::Float32 => {
            for i in 0..8 {
                let shift = (b[i] & 0x1F) as u32;
                result[i] = (a[i] as i32).wrapping_shr(shift) as u32;
            }
        }
        ElementType::Int16 => {
            for i in 0..8 {
                let a_lo = (a[i] & 0xFFFF) as i16;
                let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                let s_lo = (b[i] & 0xF) as u32;
                let s_hi = ((b[i] >> 16) & 0xF) as u32;
                let r_lo = a_lo.wrapping_shr(s_lo) as u16;
                let r_hi = a_hi.wrapping_shr(s_hi) as u16;
                result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
            }
        }
        ElementType::UInt16 | ElementType::BFloat16 => {
            for i in 0..8 {
                let a_lo = (a[i] & 0xFFFF) as u16;
                let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                let s_lo = (b[i] & 0xF) as u32;
                let s_hi = ((b[i] >> 16) & 0xF) as u32;
                let r_lo = a_lo.wrapping_shr(s_lo);
                let r_hi = a_hi.wrapping_shr(s_hi);
                result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
            }
        }
        ElementType::Int8 => {
            for i in 0..8 {
                let mut r = 0u32;
                for j in 0..4 {
                    let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                    let s = ((b[i] >> (j * 8)) & 0x7) as u32;
                    let r_byte = a_byte.wrapping_shr(s) as u8;
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
                    let s = ((b[i] >> (j * 8)) & 0x7) as u32;
                    let r_byte = a_byte.wrapping_shr(s);
                    r |= (r_byte as u32) << (j * 8);
                }
                result[i] = r;
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;
    use smallvec::smallvec;
    use xdna_archspec::aie2::isa::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    // ======================================================================
    // source_forward threading (mirrors vector_helpers Task 5 test)
    // ======================================================================

    #[test]
    fn test_semantic_get_vector_source_honors_source_forward() {
        let mut ctx = make_ctx();
        // Known committed (old) value in reg 3 before any in-flight writes.
        ctx.vector.write(3, [0xAAAA_BBBB; 8]);
        // Queue an in-flight write to reg 3: l_def 2, No-def, issued at bundle 0.
        ctx.vector.advance_bundle(0);
        ctx.vector.queue_write(3, [0x1234_5678; 8], 2, Bypass::No);
        ctx.vector.advance_bundle(1); // consumer reads at issue+1

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add).as_vector(ElementType::Int32);
        op.sources = smallvec![Operand::VectorReg(3)];

        // Store-like consumer (uc 1, No): eff = 2-1+1 = 2 -> NOT visible at issue+1.
        op.source_forward = smallvec![(1u8, Bypass::No)];
        assert_eq!(get_vector_source(&op, &ctx, 0), [0xAAAA_BBBB; 8], "uc1: old value at issue+1");

        // Compute consumer (uc 2, No): eff = 2-2+1 = 1 -> visible at issue+1.
        op.source_forward = smallvec![(2u8, Bypass::No)];
        assert_eq!(get_vector_source(&op, &ctx, 0), [0x1234_5678; 8], "uc2: new value visible at issue+1");

        // No source_forward: falls back to compute default (uc 2, Mov) -> visible at +1.
        op.source_forward = smallvec![];
        assert_eq!(get_vector_source(&op, &ctx, 0), [0x1234_5678; 8], "fallback: compute default visible");
    }

    #[test]
    fn test_semantic_add_end_to_end_honors_source_forward() {
        // Exercise the full execute_vector_semantic path: an Add whose two
        // source reads carry store-like forwarding (uc 1, No) must read the OLD
        // values of an in-flight write, not the forwarded-new compute default.
        let mut ctx = make_ctx();
        ctx.vector.write(0, [10u32; 8]); // committed old
        ctx.vector.write(1, [20u32; 8]); // committed old
        ctx.vector.advance_bundle(0);
        // In-flight writes that would change v0/v1 if visible.
        ctx.vector.queue_write(0, [1000u32; 8], 2, Bypass::No);
        ctx.vector.queue_write(1, [2000u32; 8], 2, Bypass::No);
        ctx.vector.advance_bundle(1); // reads happen at issue+1

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));
        // Store-like (uc 1, No): in-flight writes NOT visible at issue+1.
        op.source_forward = smallvec![(1u8, Bypass::No), (1u8, Bypass::No)];

        assert!(execute_vector_semantic(&op, &mut ctx));
        // 10 + 20 = 30 (old values), NOT 1000 + 2000.
        assert_eq!(ctx.vector.read(2), [30u32; 8], "uc1: Add reads old in-flight values");
    }

    // -- Helper to build a vector binop --
    fn make_binop(semantic: SemanticOp, et: ElementType, src0: u8, src1: u8, dst: u8) -> SlotOp {
        SlotOp::from_semantic(SlotIndex::Vector, semantic)
            .as_vector(et)
            .with_dest(Operand::VectorReg(dst))
            .with_source(Operand::VectorReg(src0))
            .with_source(Operand::VectorReg(src1))
    }

    // ======================================================================
    // Addition tests
    // ======================================================================

    #[test]
    fn test_vector_add_i32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = make_binop(SemanticOp::Add, ElementType::Int32, 0, 1, 2);
        assert!(execute_vector_semantic(&op, &mut ctx));
        assert_eq!(ctx.vector.read(2), [11, 22, 33, 44, 55, 66, 77, 88]);
    }

    #[test]
    fn test_vector_add_i32_wrapping() {
        let a = [0xFFFF_FFFF, 0, 0, 0, 0, 0, 0, 0];
        let b = [1, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_add(&a, &b, ElementType::Int32);
        assert_eq!(result[0], 0); // Wraps around
    }

    #[test]
    fn test_vector_add_i16() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [0x0002_0001, 0, 0, 0, 0, 0, 0, 0]);
        ctx.vector.write(1, [0x0020_0010, 0, 0, 0, 0, 0, 0, 0]);

        let op = make_binop(SemanticOp::Add, ElementType::Int16, 0, 1, 2);
        execute_vector_semantic(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2)[0], 0x0022_0011);
    }

    #[test]
    fn test_vector_add_i8() {
        let a = [0x01020304, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x10101010, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_add(&a, &b, ElementType::Int8);
        assert_eq!(result[0], 0x11121314);
    }

    #[test]
    fn test_vector_add_f32() {
        let mut ctx = make_ctx();
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

        let op = make_binop(SemanticOp::Add, ElementType::Float32, 0, 1, 2);
        execute_vector_semantic(&op, &mut ctx);
        let result = ctx.vector.read(2);
        assert_eq!(f32::from_bits(result[0]), 1.5);
        assert_eq!(f32::from_bits(result[7]), 8.5);
    }

    #[test]
    fn test_vector_add_bf16() {
        fn pack_bf16(lo: f32, hi: f32) -> u32 {
            let lo_bits = (lo.to_bits() >> 16) as u16;
            let hi_bits = (hi.to_bits() >> 16) as u16;
            (lo_bits as u32) | ((hi_bits as u32) << 16)
        }

        let a: [u32; 8] = [pack_bf16(1.0, 2.0), pack_bf16(3.0, 4.0), 0, 0, 0, 0, 0, 0];
        let b: [u32; 8] = [pack_bf16(0.5, 0.5), pack_bf16(0.5, 0.5), 0, 0, 0, 0, 0, 0];
        let result = vector_add(&a, &b, ElementType::BFloat16);

        let lo0 = bf16_to_f32((result[0] & 0xFFFF) as u16);
        let hi0 = bf16_to_f32(((result[0] >> 16) & 0xFFFF) as u16);
        assert!((lo0 - 1.5).abs() < 0.1, "Expected ~1.5, got {}", lo0);
        assert!((hi0 - 2.5).abs() < 0.1, "Expected ~2.5, got {}", hi0);
    }

    // ======================================================================
    // Subtraction tests
    // ======================================================================

    #[test]
    fn test_vector_sub_i32() {
        let a = [100, 200, 300, 400, 500, 600, 700, 800];
        let b = [10, 20, 30, 40, 50, 60, 70, 80];
        let result = vector_sub(&a, &b, ElementType::Int32);
        assert_eq!(result, [90, 180, 270, 360, 450, 540, 630, 720]);
    }

    #[test]
    fn test_vector_sub_f32() {
        let a = [3.0f32.to_bits(), 5.0f32.to_bits(), 0, 0, 0, 0, 0, 0];
        let b = [1.0f32.to_bits(), 2.0f32.to_bits(), 0, 0, 0, 0, 0, 0];
        let result = vector_sub(&a, &b, ElementType::Float32);
        assert_eq!(f32::from_bits(result[0]), 2.0);
        assert_eq!(f32::from_bits(result[1]), 3.0);
    }

    #[test]
    fn test_vector_sub_i16() {
        // 0x000A_0014 = [20, 10] in 16-bit lanes
        let a = [0x000A_0014u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x0003_0005u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_sub(&a, &b, ElementType::Int16);
        assert_eq!(result[0], 0x0007_000F); // [15, 7]
    }

    #[test]
    fn test_vector_sub_i8() {
        let a = [0x10203040u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x01020304u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_sub(&a, &b, ElementType::Int8);
        assert_eq!(result[0], 0x0F1E2D3C);
    }

    // ======================================================================
    // Multiplication tests
    // ======================================================================

    #[test]
    fn test_vector_mul_i32() {
        let a = [2, 3, 4, 5, 6, 7, 8, 9];
        let b = [10; 8];
        let result = vector_mul(&a, &b, ElementType::Int32);
        assert_eq!(result, [20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    fn test_vector_mul_f32() {
        let a = [2.0f32.to_bits(), 3.0f32.to_bits(), 0, 0, 0, 0, 0, 0];
        let b = [0.5f32.to_bits(); 8];
        let result = vector_mul(&a, &b, ElementType::Float32);
        assert_eq!(f32::from_bits(result[0]), 1.0);
        assert_eq!(f32::from_bits(result[1]), 1.5);
    }

    #[test]
    fn test_vector_mul_i16() {
        // lo=3, hi=2 in lane 0
        let a = [0x0002_0003u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x0004_0005u32, 0, 0, 0, 0, 0, 0, 0]; // lo=5, hi=4
        let result = vector_mul(&a, &b, ElementType::Int16);
        let r_lo = result[0] & 0xFFFF; // 3 * 5 = 15
        let r_hi = (result[0] >> 16) & 0xFFFF; // 2 * 4 = 8
        assert_eq!(r_lo, 15);
        assert_eq!(r_hi, 8);
    }

    #[test]
    fn test_vector_mul_i8() {
        let a = [0x02030405u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x02020202u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_mul(&a, &b, ElementType::Int8);
        // bytes: 5*2=10, 4*2=8, 3*2=6, 2*2=4
        assert_eq!(result[0], 0x0406080A);
    }

    // ======================================================================
    // Negation tests
    // ======================================================================

    #[test]
    fn test_vector_neg_i32() {
        let a = [1, 0xFFFFFFFF, 0, 42, 0, 0, 0, 0]; // 1, -1, 0, 42
        let result = vector_neg(&a, ElementType::Int32);
        assert_eq!(result[0] as i32, -1);
        assert_eq!(result[1] as i32, 1);
        assert_eq!(result[2] as i32, 0);
        assert_eq!(result[3] as i32, -42);
    }

    #[test]
    fn test_vector_neg_f32() {
        let a = [1.0f32.to_bits(), (-3.0f32).to_bits(), 0.0f32.to_bits(), 0, 0, 0, 0, 0];
        let result = vector_neg(&a, ElementType::Float32);
        assert_eq!(f32::from_bits(result[0]), -1.0);
        assert_eq!(f32::from_bits(result[1]), 3.0);
        // -0.0 is a valid IEEE value
        assert_eq!(result[2], 0x8000_0000); // -0.0
    }

    #[test]
    fn test_vector_neg_i16() {
        // lo=1, hi=2
        let a = [0x0002_0001u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_neg(&a, ElementType::Int16);
        let r_lo = (result[0] & 0xFFFF) as i16;
        let r_hi = ((result[0] >> 16) & 0xFFFF) as i16;
        assert_eq!(r_lo, -1);
        assert_eq!(r_hi, -2);
    }

    #[test]
    fn test_vector_neg_i8() {
        // bytes: 1, 2, 3, 4
        let a = [0x04030201u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_neg(&a, ElementType::Int8);
        let b0 = (result[0] & 0xFF) as i8;
        let b1 = ((result[0] >> 8) & 0xFF) as i8;
        let b2 = ((result[0] >> 16) & 0xFF) as i8;
        let b3 = ((result[0] >> 24) & 0xFF) as i8;
        assert_eq!(b0, -1);
        assert_eq!(b1, -2);
        assert_eq!(b2, -3);
        assert_eq!(b3, -4);
    }

    // ======================================================================
    // Comparison tests
    // ======================================================================

    #[test]
    fn test_vector_cmp_eq_i32() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [1, 0, 3, 0, 5, 0, 7, 0];
        let result = vector_cmp_eq(&a, &b, ElementType::Int32);
        assert_eq!(result[0], 0xFFFF_FFFF);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 0xFFFF_FFFF);
        assert_eq!(result[3], 0);
    }

    #[test]
    fn test_vector_cmp_eq_i16() {
        // lo matches, hi doesn't
        let a = [0x0002_0001u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x0003_0001u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_cmp_eq(&a, &b, ElementType::Int16);
        assert_eq!(result[0], 0x0000_FFFF); // lo match, hi no match
    }

    #[test]
    fn test_vector_cmp_eq_i8() {
        // bytes: [1,2,3,4] vs [1,0,3,0]
        let a = [0x04030201u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x00030001u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_cmp_eq(&a, &b, ElementType::Int8);
        assert_eq!(result[0], 0x00FF00FF); // bytes 0,2 match
    }

    // ======================================================================
    // Min/Max tests
    // ======================================================================

    #[test]
    fn test_vector_min_uint32() {
        let a = [5, 10, 15, 20, 25, 30, 35, 40];
        let b = [10, 5, 20, 15, 30, 25, 40, 35];
        let result = vector_min(&a, &b, ElementType::UInt32);
        assert_eq!(result, [5, 5, 15, 15, 25, 25, 35, 35]);
    }

    #[test]
    fn test_vector_max_uint32() {
        let a = [5, 10, 15, 20, 25, 30, 35, 40];
        let b = [10, 5, 20, 15, 30, 25, 40, 35];
        let result = vector_max(&a, &b, ElementType::UInt32);
        assert_eq!(result, [10, 10, 20, 20, 30, 30, 40, 40]);
    }

    #[test]
    fn test_vector_min_int32_signed() {
        // Test with negative numbers
        let a = [(-5i32) as u32, 10, 0, 0, 0, 0, 0, 0];
        let b = [3, (-2i32) as u32, 0, 0, 0, 0, 0, 0];
        let result = vector_min(&a, &b, ElementType::Int32);
        assert_eq!(result[0] as i32, -5); // min(-5, 3)
        assert_eq!(result[1] as i32, -2); // min(10, -2)
    }

    #[test]
    fn test_vector_max_int32_signed() {
        let a = [(-5i32) as u32, 10, 0, 0, 0, 0, 0, 0];
        let b = [3, (-2i32) as u32, 0, 0, 0, 0, 0, 0];
        let result = vector_max(&a, &b, ElementType::Int32);
        assert_eq!(result[0] as i32, 3); // max(-5, 3)
        assert_eq!(result[1] as i32, 10); // max(10, -2)
    }

    #[test]
    fn test_vector_min_max_f32() {
        let a = [1.0f32.to_bits(), (-2.0f32).to_bits(), 0, 0, 0, 0, 0, 0];
        let b = [0.5f32.to_bits(), (-1.0f32).to_bits(), 0, 0, 0, 0, 0, 0];
        let min_r = vector_min(&a, &b, ElementType::Float32);
        let max_r = vector_max(&a, &b, ElementType::Float32);
        assert_eq!(f32::from_bits(min_r[0]), 0.5);
        assert_eq!(f32::from_bits(min_r[1]), -2.0);
        assert_eq!(f32::from_bits(max_r[0]), 1.0);
        assert_eq!(f32::from_bits(max_r[1]), -1.0);
    }

    #[test]
    fn test_vector_min_int16() {
        // lo=10, hi=20 vs lo=5, hi=30
        let a = [0x0014_000Au32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x001E_0005u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_min(&a, &b, ElementType::Int16);
        let r_lo = (result[0] & 0xFFFF) as i16;
        let r_hi = ((result[0] >> 16) & 0xFFFF) as i16;
        assert_eq!(r_lo, 5); // min(10, 5)
        assert_eq!(r_hi, 20); // min(20, 30)
    }

    #[test]
    fn test_vector_min_max_int8() {
        // bytes: [10, 5, 20, 15]
        let a = [0x0F140510u32, 0, 0, 0, 0, 0, 0, 0];
        // bytes: [5, 10, 15, 20]
        let b = [0x140F0A05u32, 0, 0, 0, 0, 0, 0, 0];
        let min_r = vector_min(&a, &b, ElementType::UInt8);
        let max_r = vector_max(&a, &b, ElementType::UInt8);

        // min: [min(0x10,0x05), min(0x05,0x0A), min(0x14,0x0F), min(0x0F,0x14)]
        //    = [0x05, 0x05, 0x0F, 0x0F]
        assert_eq!(min_r[0], 0x0F0F0505);
        // max: [max(0x10,0x05), max(0x05,0x0A), max(0x14,0x0F), max(0x0F,0x14)]
        //    = [0x10, 0x0A, 0x14, 0x14]
        assert_eq!(max_r[0], 0x14140A10);
    }

    // ======================================================================
    // Bitwise operation tests
    // ======================================================================

    #[test]
    fn test_vector_and() {
        let a = [0xFF00FF00, 0xAAAA5555, 0, 0, 0, 0, 0, 0];
        let b = [0x0F0F0F0F, 0xFFFF0000, 0, 0, 0, 0, 0, 0];
        let result = vector_and(&a, &b);
        assert_eq!(result[0], 0x0F000F00);
        assert_eq!(result[1], 0xAAAA0000);
    }

    #[test]
    fn test_vector_or() {
        let a = [0xFF00FF00, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x00FF00FF, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_or(&a, &b);
        assert_eq!(result[0], 0xFFFFFFFF);
    }

    #[test]
    fn test_vector_xor() {
        let a = [0xFF00FF00, 0xAAAA5555, 0, 0, 0, 0, 0, 0];
        let b = [0xFF00FF00, 0x55AA55AA, 0, 0, 0, 0, 0, 0];
        let result = vector_xor(&a, &b);
        assert_eq!(result[0], 0); // XOR with self = 0
        assert_eq!(result[1], 0xFF0000FF);
    }

    #[test]
    fn test_vector_not() {
        let a = [0xFF00FF00, 0, 0xFFFFFFFF, 0, 0, 0, 0, 0];
        let result = vector_not(&a);
        assert_eq!(result[0], 0x00FF00FF);
        assert_eq!(result[1], 0xFFFFFFFF);
        assert_eq!(result[2], 0);
    }

    // ======================================================================
    // Shift operation tests
    // ======================================================================

    #[test]
    fn test_vector_shl_i32() {
        let a = [1, 0xFF, 0, 0, 0, 0, 0, 0];
        let b = [4, 8, 0, 0, 0, 0, 0, 0];
        let result = vector_shl(&a, &b, ElementType::Int32);
        assert_eq!(result[0], 16); // 1 << 4
        assert_eq!(result[1], 0xFF00); // 0xFF << 8
    }

    #[test]
    fn test_vector_srl_i32() {
        let a = [0x80000000, 0xFF00, 0, 0, 0, 0, 0, 0];
        let b = [1, 8, 0, 0, 0, 0, 0, 0];
        let result = vector_srl(&a, &b, ElementType::Int32);
        assert_eq!(result[0], 0x40000000); // Logical: no sign extension
        assert_eq!(result[1], 0xFF);
    }

    #[test]
    fn test_vector_sra_i32() {
        let a = [0x80000000, 0xFF00, 0, 0, 0, 0, 0, 0]; // -2^31, 0xFF00
        let b = [1, 8, 0, 0, 0, 0, 0, 0];
        let result = vector_sra(&a, &b, ElementType::Int32);
        assert_eq!(result[0], 0xC0000000); // Arithmetic: sign extends
        assert_eq!(result[1], 0xFF); // Positive, same as logical
    }

    #[test]
    fn test_vector_shl_i16() {
        // lo=0x0001, hi=0x00FF -> shift lo by 4, hi by 2
        let a = [0x00FF_0001u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x0002_0004u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_shl(&a, &b, ElementType::Int16);
        let r_lo = result[0] & 0xFFFF;
        let r_hi = (result[0] >> 16) & 0xFFFF;
        assert_eq!(r_lo, 0x0010); // 1 << 4
        assert_eq!(r_hi, 0x03FC); // 0xFF << 2
    }

    #[test]
    fn test_vector_shl_i8() {
        // byte 0 = 1, shift by 2
        let a = [0x00000001u32, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x00000002u32, 0, 0, 0, 0, 0, 0, 0];
        let result = vector_shl(&a, &b, ElementType::Int8);
        assert_eq!(result[0] & 0xFF, 4); // 1 << 2
    }

    // ======================================================================
    // Absolute value tests
    // ======================================================================

    // ======================================================================
    // Integration: dispatch through execute_vector_semantic
    // ======================================================================

    #[test]
    fn test_dispatch_returns_false_for_non_vector_ops() {
        let mut ctx = make_ctx();
        // Scalar add (is_vector = false) should not be handled
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(2));
        assert!(!execute_vector_semantic(&op, &mut ctx));
    }

    #[test]
    fn test_dispatch_add_via_entry_point() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1; 8]);
        ctx.vector.write(1, [2; 8]);

        let op = make_binop(SemanticOp::Add, ElementType::UInt32, 0, 1, 2);
        assert!(execute_vector_semantic(&op, &mut ctx));
        assert_eq!(ctx.vector.read(2), [3; 8]);
    }

    #[test]
    fn test_dispatch_sub_via_entry_point() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [10; 8]);
        ctx.vector.write(1, [3; 8]);

        let op = make_binop(SemanticOp::Sub, ElementType::UInt32, 0, 1, 2);
        assert!(execute_vector_semantic(&op, &mut ctx));
        assert_eq!(ctx.vector.read(2), [7; 8]);
    }

    #[test]
    fn test_dispatch_mul_via_entry_point() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [5; 8]);
        ctx.vector.write(1, [3; 8]);

        let op = make_binop(SemanticOp::Mul, ElementType::UInt32, 0, 1, 2);
        assert!(execute_vector_semantic(&op, &mut ctx));
        assert_eq!(ctx.vector.read(2), [15; 8]);
    }

    #[test]
    fn test_dispatch_cmp_via_entry_point() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [1, 0, 3, 0, 5, 0, 7, 0]);

        let op = make_binop(SemanticOp::Cmp, ElementType::Int32, 0, 1, 2);
        assert!(execute_vector_semantic(&op, &mut ctx));
        let result = ctx.vector.read(2);
        assert_eq!(result[0], 0xFFFF_FFFF);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 0xFFFF_FFFF);
        assert_eq!(result[3], 0);
    }

    #[test]
    fn test_dispatch_min_via_entry_point() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [5, 10, 15, 20, 25, 30, 35, 40]);
        ctx.vector.write(1, [10, 5, 20, 15, 30, 25, 40, 35]);

        let op = make_binop(SemanticOp::Min, ElementType::UInt32, 0, 1, 2);
        assert!(execute_vector_semantic(&op, &mut ctx));
        assert_eq!(ctx.vector.read(2), [5, 5, 15, 15, 25, 25, 35, 35]);
    }

    #[test]
    fn test_dispatch_max_via_entry_point() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [5, 10, 15, 20, 25, 30, 35, 40]);
        ctx.vector.write(1, [10, 5, 20, 15, 30, 25, 40, 35]);

        let op = make_binop(SemanticOp::Max, ElementType::UInt32, 0, 1, 2);
        assert!(execute_vector_semantic(&op, &mut ctx));
        assert_eq!(ctx.vector.read(2), [10, 10, 20, 20, 30, 30, 40, 40]);
    }

    // ======================================================================
    // Edge cases
    // ======================================================================

    #[test]
    fn test_vector_add_no_dest() {
        // If no dest register is set, the function still returns true
        // (handled) but writes nothing.
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1; 8]);
        ctx.vector.write(1, [2; 8]);
        // Explicitly zero v2 so we can verify no write happened.
        ctx.vector.write(2, [0; 8]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Int32)
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));
        // No dest set
        assert!(execute_vector_semantic(&op, &mut ctx));
        // v2 should still be zero (no write happened)
        assert_eq!(ctx.vector.read(2), [0; 8]);
    }

    #[test]
    fn test_vector_add_missing_source() {
        // Missing source defaults to zero vector
        let mut ctx = make_ctx();
        ctx.vector.write(0, [42; 8]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0));
        // Only one source, second defaults to [0; 8]
        assert!(execute_vector_semantic(&op, &mut ctx));
        assert_eq!(ctx.vector.read(2), [42; 8]); // 42 + 0 = 42
    }
}
