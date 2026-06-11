//! Comparison operations for the vector ALU.
//!
//! Contains both pure computational functions (vector_cmp_eq, vector_compare_ge,
//! etc.) and dispatched execute_* entry points that handle narrow/wide routing.

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;

use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    /// Sign-magnitude ordering key for bf16 comparisons (None = NaN).
    ///
    /// The bf16 vector comparators (VGE/VLT) are unordered on NaN -- any
    /// NaN operand makes the comparison FALSE in both directions -- and
    /// order all other lanes by their bit pattern, sign-magnitude, so -0
    /// sorts strictly BELOW +0. Verified against NPU1 silicon (vector
    /// fuzzer seeds 4/41/47: vge.bf16(-0, +0) is FALSE; lt/ge with a NaN
    /// operand always FALSE; a numeric f32 compare flips the vsel lane).
    #[inline]
    pub(super) fn bf16_order_key(bits: u16) -> Option<i32> {
        if bits & 0x7FFF > 0x7F80 {
            return None; // NaN: unordered
        }
        let mag = (bits & 0x7FFF) as i32;
        Some(if bits & 0x8000 != 0 { -mag - 1 } else { mag })
    }

    /// bf16 a >= b under the silicon ordering (false when either is NaN).
    #[inline]
    fn bf16_cmp_ge(a: u16, b: u16) -> bool {
        matches!(
            (Self::bf16_order_key(a), Self::bf16_order_key(b)),
            (Some(ka), Some(kb)) if ka >= kb
        )
    }

    /// bf16 a < b under the silicon ordering (false when either is NaN).
    #[inline]
    fn bf16_cmp_lt(a: u16, b: u16) -> bool {
        matches!(
            (Self::bf16_order_key(a), Self::bf16_order_key(b)),
            (Some(ka), Some(kb)) if ka < kb
        )
    }

    /// Vector equality comparison (returns mask).
    pub(super) fn vector_cmp_eq(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
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

    /// Vector compare greater-or-equal: dst[i] = (a[i] >= b[i]) ? ~0 : 0
    pub(super) fn vector_compare_ge(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = if (a[i] as i32) >= (b[i] as i32) { !0 } else { 0 };
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = if a[i] >= b[i] { !0 } else { 0 };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = if fa >= fb { !0 } else { 0 };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo: u16 = if a_lo >= b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi >= b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo: u16 = if a_lo >= b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi >= b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                // Sign-magnitude ordering, NaN unordered (see bf16_order_key).
                for i in 0..8 {
                    let r_lo: u16 = if Self::bf16_cmp_ge((a[i] & 0xFFFF) as u16, (b[i] & 0xFFFF) as u16) {
                        !0
                    } else {
                        0
                    };
                    let r_hi: u16 = if Self::bf16_cmp_ge((a[i] >> 16) as u16, (b[i] >> 16) as u16) {
                        !0
                    } else {
                        0
                    };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte: u8 = if a_byte >= b_byte { !0 } else { 0 };
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
                        let r_byte: u8 = if a_byte >= b_byte { !0 } else { 0 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector compare less-than: dst[i] = (a[i] < b[i]) ? ~0 : 0
    pub(super) fn vector_compare_lt(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::Int64 => {
                for i in 0..8 {
                    result[i] = if (a[i] as i32) < (b[i] as i32) { !0 } else { 0 };
                }
            }
            ElementType::UInt32 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = if a[i] < b[i] { !0 } else { 0 };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = if fa < fb { !0 } else { 0 };
                }
            }
            ElementType::Int16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16;
                    let b_lo = (b[i] & 0xFFFF) as i16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16;
                    let r_lo: u16 = if a_lo < b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi < b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::UInt16 => {
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as u16;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let b_lo = (b[i] & 0xFFFF) as u16;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as u16;
                    let r_lo: u16 = if a_lo < b_lo { !0 } else { 0 };
                    let r_hi: u16 = if a_hi < b_hi { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                // Sign-magnitude ordering, NaN unordered (see bf16_order_key).
                for i in 0..8 {
                    let r_lo: u16 = if Self::bf16_cmp_lt((a[i] & 0xFFFF) as u16, (b[i] & 0xFFFF) as u16) {
                        !0
                    } else {
                        0
                    };
                    let r_hi: u16 = if Self::bf16_cmp_lt((a[i] >> 16) as u16, (b[i] >> 16) as u16) {
                        !0
                    } else {
                        0
                    };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as i8;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as i8;
                        let r_byte: u8 = if a_byte < b_byte { !0 } else { 0 };
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
                        let r_byte: u8 = if a_byte < b_byte { !0 } else { 0 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    /// Vector compare equal-to-zero: dst[i] = (a[i] == 0) ? ~0 : 0
    pub(super) fn vector_compare_eqz(a: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Int64 | ElementType::UInt64 => {
                for i in 0..8 {
                    result[i] = if a[i] == 0 { !0 } else { 0 };
                }
            }
            ElementType::Float32 => {
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    result[i] = if fa == 0.0 { !0 } else { 0 };
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                for i in 0..8 {
                    let lo = (a[i] & 0xFFFF) as u16;
                    let hi = ((a[i] >> 16) & 0xFFFF) as u16;
                    let r_lo: u16 = if lo == 0 { !0 } else { 0 };
                    let r_hi: u16 = if hi == 0 { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                for i in 0..8 {
                    let lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16);
                    let hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16);
                    let r_lo: u16 = if lo == 0.0 { !0 } else { 0 };
                    let r_hi: u16 = if hi == 0.0 { !0 } else { 0 };
                    result[i] = (r_lo as u32) | ((r_hi as u32) << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let byte = ((a[i] >> (j * 8)) & 0xFF) as u8;
                        let r_byte: u8 = if byte == 0 { !0 } else { 0 };
                        r |= (r_byte as u32) << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }

    // ========== Dispatched entry points ==========

    /// Vector compare equal (Cmp).
    pub(super) fn execute_cmp(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if op.is_wide_vector {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_cmp_eq);
            Self::write_wide_vec_dest(op, ctx, result);
            true
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = Self::vector_cmp_eq(&a, &b, et);
            Self::write_vector_dest(op, ctx, result);
            true
        }
    }

    /// Set if greater-or-equal (SetGe).
    pub(super) fn execute_setge(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        // Accumulator sources (Half or Full) need the wide path.
        let has_wide_acc_source = matches!(
            op.accum_width,
            Some(crate::interpreter::decode::register_map::AccumWidth::Full)
                | Some(crate::interpreter::decode::register_map::AccumWidth::Half)
        );
        if op.is_wide_vector || has_wide_acc_source {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_compare_ge);
            if matches!(op.dest, Some(Operand::ScalarReg(_))) {
                // VGE: condense per-lane mask into scalar bitmask
                let lo: [u32; 8] = result[..8].try_into().unwrap();
                let hi: [u32; 8] = result[8..].try_into().unwrap();
                let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                let lanes_per_half = 256 / et.bits() as u32;
                let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
                // For 8-bit elements, the mask needs 64 bits (register pair).
                if lanes_per_half >= 32 {
                    // Write to primary dest (VGE/VLT: cmp is the main dest).
                    Self::write_scalar_dest_wide(op, ctx, full_mask);
                    // Also write to extra_dests (dual-result ops like VSUB_LT).
                    Self::write_cmp_dest_wide(op, ctx, full_mask);
                } else {
                    Self::write_scalar_dest(op, ctx, full_mask as u32);
                    // Also write to extra_dests (cmp register) for 16/32-bit.
                    Self::write_cmp_dest(op, ctx, full_mask as u32);
                }
            } else {
                Self::write_wide_vec_dest(op, ctx, result);
                let lo: [u32; 8] = result[..8].try_into().unwrap();
                let hi: [u32; 8] = result[8..].try_into().unwrap();
                let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                let lanes_per_half = 256 / et.bits() as u32;
                let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
                if lanes_per_half >= 32 {
                    Self::write_cmp_dest_wide(op, ctx, full_mask);
                } else {
                    Self::write_cmp_dest(op, ctx, full_mask as u32);
                }
            }
            true
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = Self::vector_compare_ge(&a, &b, et);
            Self::write_vector_dest(op, ctx, result);
            // cmp = packed bitmask of the same comparison
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&result, et));
            true
        }
    }

    /// Set if less-than (SetLt).
    pub(super) fn execute_setlt(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        // Accumulator sources (Half or Full) need the wide path.
        let has_wide_acc_source = matches!(
            op.accum_width,
            Some(crate::interpreter::decode::register_map::AccumWidth::Full)
                | Some(crate::interpreter::decode::register_map::AccumWidth::Half)
        );
        if op.is_wide_vector || has_wide_acc_source {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_compare_lt);
            if matches!(op.dest, Some(Operand::ScalarReg(_))) {
                let lo: [u32; 8] = result[..8].try_into().unwrap();
                let hi: [u32; 8] = result[8..].try_into().unwrap();
                let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                let lanes_per_half = 256 / et.bits() as u32;
                let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
                if lanes_per_half >= 32 {
                    Self::write_scalar_dest_wide(op, ctx, full_mask);
                    Self::write_cmp_dest_wide(op, ctx, full_mask);
                } else {
                    Self::write_scalar_dest(op, ctx, full_mask as u32);
                    Self::write_cmp_dest(op, ctx, full_mask as u32);
                }
            } else {
                Self::write_wide_vec_dest(op, ctx, result);
                let lo: [u32; 8] = result[..8].try_into().unwrap();
                let hi: [u32; 8] = result[8..].try_into().unwrap();
                let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                let lanes_per_half = 256 / et.bits() as u32;
                let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
                if lanes_per_half >= 32 {
                    Self::write_cmp_dest_wide(op, ctx, full_mask);
                } else {
                    Self::write_cmp_dest(op, ctx, full_mask as u32);
                }
            }
            true
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = Self::vector_compare_lt(&a, &b, et);
            Self::write_vector_dest(op, ctx, result);
            // cmp = packed bitmask of the same comparison
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&result, et));
            true
        }
    }

    /// Set if equal to zero (SetEq / VEQZ).
    pub(super) fn execute_seteq(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if op.is_wide_vector {
            // VEQZ: compare each element to zero, produce scalar comparison bitmask.
            // VEQZ has the cmp register as primary dest (not extra_dests), so
            // write to dest via write_scalar_dest. Also write to extra_dests
            // for dual-result ops that might share this path.
            let a = Self::get_wide_vec_source(op, ctx, 0);
            let result = Self::wide_element_wise_unary(&a, et, Self::vector_compare_eqz);
            let lo: [u32; 8] = result[..8].try_into().unwrap();
            let hi: [u32; 8] = result[8..].try_into().unwrap();
            let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
            let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
            let lanes_per_half = 256 / et.bits() as u32;
            let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
            if lanes_per_half >= 32 {
                Self::write_scalar_dest_wide(op, ctx, full_mask);
                Self::write_cmp_dest_wide(op, ctx, full_mask);
            } else {
                Self::write_scalar_dest(op, ctx, full_mask as u32);
                Self::write_cmp_dest(op, ctx, full_mask as u32);
            }
            true
        } else {
            // Vector equal-to-zero (VEQZ): produces a comparison bitmask
            // written to scalar cmp register(s), not a vector result.
            let src = Self::get_vector_source(op, ctx, 0);
            let result = Self::vector_compare_eqz(&src, et);
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&result, et));
            true
        }
    }

    /// Vector max with LT comparison flag (MaxLt).
    pub(super) fn execute_maxlt(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if op.is_wide_vector {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let a_lo: [u32; 8] = a[..8].try_into().unwrap();
            let a_hi: [u32; 8] = a[8..].try_into().unwrap();
            let b_lo: [u32; 8] = b[..8].try_into().unwrap();
            let b_hi: [u32; 8] = b[8..].try_into().unwrap();
            let result_lo = Self::vector_max(&a_lo, &b_lo, et);
            let result_hi = Self::vector_max(&a_hi, &b_hi, et);
            let mut result = [0u32; 16];
            result[..8].copy_from_slice(&result_lo);
            result[8..].copy_from_slice(&result_hi);
            Self::write_wide_vec_dest(op, ctx, result);
            // cmp = per-element (a < b)
            let cmp_lo = Self::vector_compare_lt(&a_lo, &b_lo, et);
            let cmp_hi = Self::vector_compare_lt(&a_hi, &b_hi, et);
            let mask_lo = Self::condense_comparison_mask(&cmp_lo, op.element_type);
            let mask_hi = Self::condense_comparison_mask(&cmp_hi, op.element_type);
            let lanes_per_half = 256 / et.bits() as u32;
            let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
            if lanes_per_half >= 32 {
                Self::write_cmp_dest_wide(op, ctx, full_mask);
            } else {
                Self::write_cmp_dest(op, ctx, full_mask as u32);
            }
            true
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = Self::vector_max(&a, &b, et);
            Self::write_vector_dest(op, ctx, result);
            // cmp = per-element (a < b)
            let cmp = Self::vector_compare_lt(&a, &b, et);
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
            true
        }
    }

    /// Vector min with GE comparison flag (MinGe).
    pub(super) fn execute_minge(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if op.is_wide_vector {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let a_lo: [u32; 8] = a[..8].try_into().unwrap();
            let a_hi: [u32; 8] = a[8..].try_into().unwrap();
            let b_lo: [u32; 8] = b[..8].try_into().unwrap();
            let b_hi: [u32; 8] = b[8..].try_into().unwrap();
            let result_lo = Self::vector_min(&a_lo, &b_lo, et);
            let result_hi = Self::vector_min(&a_hi, &b_hi, et);
            let mut result = [0u32; 16];
            result[..8].copy_from_slice(&result_lo);
            result[8..].copy_from_slice(&result_hi);
            Self::write_wide_vec_dest(op, ctx, result);
            // cmp = per-element (a >= b)
            let cmp_lo = Self::vector_compare_ge(&a_lo, &b_lo, et);
            let cmp_hi = Self::vector_compare_ge(&a_hi, &b_hi, et);
            let mask_lo = Self::condense_comparison_mask(&cmp_lo, op.element_type);
            let mask_hi = Self::condense_comparison_mask(&cmp_hi, op.element_type);
            let lanes_per_half = 256 / et.bits() as u32;
            let full_mask = (mask_lo as u64) | ((mask_hi as u64) << lanes_per_half);
            if lanes_per_half >= 32 {
                Self::write_cmp_dest_wide(op, ctx, full_mask);
            } else {
                Self::write_cmp_dest(op, ctx, full_mask as u32);
            }
            true
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = Self::vector_min(&a, &b, et);
            Self::write_vector_dest(op, ctx, result);
            // cmp = per-element (a >= b)
            let cmp = Self::vector_compare_ge(&a, &b, et);
            Self::write_cmp_dest(op, ctx, Self::pack_comparison_flags(&cmp, et));
            true
        }
    }

    /// Per-element vector select based on scalar bitmask (VectorSelect).
    pub(super) fn execute_select(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if op.is_wide_vector {
            // VSEL: 512-bit select with scalar bitmask.
            // Must split the selector across lo/hi halves.
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let sel_scalar = op
                .sources
                .iter()
                .find_map(|s| match s {
                    Operand::ScalarReg(r) => Some(ctx.scalar_read(*r)),
                    _ => None,
                })
                .unwrap_or(0);
            let a_lo: [u32; 8] = a[..8].try_into().unwrap();
            let a_hi: [u32; 8] = a[8..].try_into().unwrap();
            let b_lo: [u32; 8] = b[..8].try_into().unwrap();
            let b_hi: [u32; 8] = b[8..].try_into().unwrap();
            let elems_per_half = 256 / et.bits() as u32;
            let sel_lo = Self::expand_select_mask(sel_scalar, et);
            let sel_hi_bits = if elems_per_half >= 32 {
                // 8-bit elements: 64-bit mask from register pair.
                // The operand decodes as a single ScalarReg (low reg
                // of the pair). Read r+1 for the high 32 bits.
                op.sources
                    .iter()
                    .find_map(|s| match s {
                        Operand::ScalarReg(r) => Some(ctx.scalar_read(r + 1)),
                        _ => None,
                    })
                    .unwrap_or(0)
            } else {
                sel_scalar >> elems_per_half
            };
            let sel_hi = Self::expand_select_mask(sel_hi_bits, et);
            // Hardware: bit=0 -> s1, bit=1 -> s2
            let lo = Self::vector_select(&sel_lo, &b_lo, &a_lo, et);
            let hi = Self::vector_select(&sel_hi, &b_hi, &a_hi, et);
            let mut result = [0u32; 16];
            result[..8].copy_from_slice(&lo);
            result[8..].copy_from_slice(&hi);
            Self::write_wide_vec_dest(op, ctx, result);
            true
        } else {
            // VSEL.32 d, s1, s2, sel: per-lane conditional select.
            // input_order is [s1, s2, sel] where sel is a scalar register
            // whose bits control per-element selection.
            //
            // Hardware semantics (aietools aie_api select documentation):
            //   d[i] = (sel >> i) & 1 == 0 ? s1[i] : s2[i]
            //
            // When sel bit is 0: take from s1 (first vector source).
            // When sel bit is 1: take from s2 (second vector source).
            let s1 = Self::get_vector_source(op, ctx, 0);
            let s2 = Self::get_vector_source(op, ctx, 1);
            // Read the scalar select mask from the last source operand
            let sel_scalar = op
                .sources
                .get(2)
                .map(|s| match s {
                    Operand::ScalarReg(r) => ctx.scalar_read(*r),
                    Operand::Immediate(v) => *v as u32,
                    _ => Self::get_scalar_source(op, ctx),
                })
                .unwrap_or(0);
            // Expand scalar mask to per-lane vector mask.
            let sel = Self::expand_select_mask(sel_scalar, et);
            // vector_select does: mask != 0 ? arg1 : arg2
            // Hardware wants: mask != 0 ? s2 : s1
            // So pass s2 as arg1 and s1 as arg2.
            let result = Self::vector_select(&sel, &s2, &s1, et);
            log::debug!(
                "[VSEL] sel_scalar=0x{:X} sel={:?} s1={:?} s2={:?} -> {:?} (sources={:?}, dest={:?})",
                sel_scalar,
                sel,
                s1,
                s2,
                result,
                op.sources,
                op.dest
            );
            Self::write_vector_dest(op, ctx, result);
            true
        }
    }

    /// Per-element select: dst[i] = mask[i] != 0 ? src1[i] : src2[i]
    pub(super) fn vector_select(
        mask: &[u32; 8],
        src1: &[u32; 8],
        src2: &[u32; 8],
        elem_type: ElementType,
    ) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32
            | ElementType::UInt32
            | ElementType::Int64
            | ElementType::UInt64
            | ElementType::Float32 => {
                // 8 lanes of 32-bit elements
                for i in 0..8 {
                    result[i] = if mask[i] != 0 { src1[i] } else { src2[i] };
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 lanes of 16-bit elements (2 per u32)
                for i in 0..8 {
                    let mask_lo = mask[i] & 0xFFFF;
                    let mask_hi = (mask[i] >> 16) & 0xFFFF;
                    let src1_lo = src1[i] & 0xFFFF;
                    let src1_hi = (src1[i] >> 16) & 0xFFFF;
                    let src2_lo = src2[i] & 0xFFFF;
                    let src2_hi = (src2[i] >> 16) & 0xFFFF;
                    let r_lo = if mask_lo != 0 { src1_lo } else { src2_lo };
                    let r_hi = if mask_hi != 0 { src1_hi } else { src2_hi };
                    result[i] = r_lo | (r_hi << 16);
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 lanes of 8-bit elements (4 per u32)
                for i in 0..8 {
                    let mut r = 0u32;
                    for j in 0..4 {
                        let m = (mask[i] >> (j * 8)) & 0xFF;
                        let s1 = (src1[i] >> (j * 8)) & 0xFF;
                        let s2 = (src2[i] >> (j * 8)) & 0xFF;
                        let val = if m != 0 { s1 } else { s2 };
                        r |= val << (j * 8);
                    }
                    result[i] = r;
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{SlotIndex, Operand, ElementType};
    use xdna_archspec::aie2::isa::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    // ========== vector_cmp_eq ==========

    #[test]
    fn test_cmp_eq_i32_equal() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [1, 2, 3, 4, 5, 6, 7, 8];
        let r = VectorAlu::vector_cmp_eq(&a, &b, ElementType::Int32);
        assert_eq!(r, [!0; 8]);
    }

    #[test]
    fn test_cmp_eq_i32_none_equal() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [9, 10, 11, 12, 13, 14, 15, 16];
        let r = VectorAlu::vector_cmp_eq(&a, &b, ElementType::Int32);
        assert_eq!(r, [0; 8]);
    }

    #[test]
    fn test_cmp_eq_i32_partial() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [1, 0, 3, 0, 5, 0, 7, 0];
        let r = VectorAlu::vector_cmp_eq(&a, &b, ElementType::Int32);
        assert_eq!(r[0], !0); // 1 == 1
        assert_eq!(r[1], 0); // 2 != 0
        assert_eq!(r[2], !0); // 3 == 3
        assert_eq!(r[3], 0); // 4 != 0
    }

    #[test]
    fn test_cmp_eq_i16() {
        // Word 0: [a_lo=5, a_hi=10], [b_lo=5, b_hi=99]
        let a = [0x000A_0005, 0, 0, 0, 0, 0, 0, 0]; // lo=5, hi=10
        let b = [0x0063_0005, 0, 0, 0, 0, 0, 0, 0]; // lo=5, hi=99
        let r = VectorAlu::vector_cmp_eq(&a, &b, ElementType::Int16);
        // lo half matches (5==5) -> 0xFFFF, hi doesn't (10!=99) -> 0x0000
        assert_eq!(r[0], 0x0000_FFFF);
    }

    #[test]
    fn test_cmp_eq_i8() {
        // Word 0: bytes [0x01, 0x02, 0x03, 0x04]
        let a = [0x04030201, 0, 0, 0, 0, 0, 0, 0];
        // Word 0: bytes [0x01, 0xFF, 0x03, 0xFF]
        let b = [0xFF03FF01, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_cmp_eq(&a, &b, ElementType::Int8);
        // byte 0: 0x01==0x01 -> 0xFF, byte 1: 0x02!=0xFF -> 0x00
        // byte 2: 0x03==0x03 -> 0xFF, byte 3: 0x04!=0xFF -> 0x00
        assert_eq!(r[0], 0x00FF_00FF);
    }

    // ========== vector_compare_ge ==========

    #[test]
    fn test_ge_i32_signed() {
        let a = [5, 0xFFFF_FFFF, 10, 0, 0, 0, 0, 0]; // [5, -1, 10, 0, ...]
        let b = [3, 0, 10, 1, 0, 0, 0, 0]; // [3,  0, 10, 1, ...]
        let r = VectorAlu::vector_compare_ge(&a, &b, ElementType::Int32);
        assert_eq!(r[0], !0); // 5 >= 3
        assert_eq!(r[1], 0); // -1 >= 0 is false
        assert_eq!(r[2], !0); // 10 >= 10
        assert_eq!(r[3], 0); // 0 >= 1 is false
    }

    #[test]
    fn test_ge_u32_unsigned() {
        let a = [0xFFFF_FFFF, 0, 0, 0, 0, 0, 0, 0];
        let b = [1, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_compare_ge(&a, &b, ElementType::UInt32);
        assert_eq!(r[0], !0); // 0xFFFFFFFF >= 1 (unsigned)
    }

    #[test]
    fn test_ge_f32() {
        let a = [f32::to_bits(1.5), f32::to_bits(-1.0), 0, 0, 0, 0, 0, 0];
        let b = [f32::to_bits(1.0), f32::to_bits(0.0), 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_compare_ge(&a, &b, ElementType::Float32);
        assert_eq!(r[0], !0); // 1.5 >= 1.0
        assert_eq!(r[1], 0); // -1.0 >= 0.0 is false
    }

    #[test]
    fn test_ge_i16_signed() {
        // lo=0x8000 (-32768), hi=0x0001 (1)  vs  lo=0x7FFF (32767), hi=0x0000 (0)
        let a = [0x0001_8000, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x0000_7FFF, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_compare_ge(&a, &b, ElementType::Int16);
        // lo: -32768 >= 32767 -> false (0x0000)
        // hi: 1 >= 0 -> true (0xFFFF)
        assert_eq!(r[0], 0xFFFF_0000);
    }

    #[test]
    fn test_ge_i8_signed() {
        // bytes: [127, -128, 0, 0] vs [0, 0, 0, 0]
        let a = [0x0000_807F, 0, 0, 0, 0, 0, 0, 0]; // [0x7F, 0x80, 0x00, 0x00]
        let b = [0x0000_0000, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_compare_ge(&a, &b, ElementType::Int8);
        // byte 0: 127 >= 0 -> true (0xFF)
        // byte 1: -128 >= 0 -> false (0x00)
        // byte 2: 0 >= 0 -> true (0xFF)
        // byte 3: 0 >= 0 -> true (0xFF)
        assert_eq!(r[0], 0xFFFF_00FF);
    }

    /// bf16 VGE/VLT silicon semantics (fuzzer seeds 4/41/47): sign-magnitude
    /// ordering (-0 strictly below +0), NaN unordered (both directions FALSE).
    #[test]
    fn test_bf16_compare_neg_zero_and_nan() {
        // lo = -0 vs +0, hi = +0 vs -0
        let a = [0x0000_8000u32, 0x7FC0_8006, 0, 0, 0, 0, 0, 0];
        let b = [0x8000_0000u32, 0x0000_7FC0, 0, 0, 0, 0, 0, 0];
        let ge = VectorAlu::vector_compare_ge(&a, &b, ElementType::BFloat16);
        let lt = VectorAlu::vector_compare_lt(&a, &b, ElementType::BFloat16);
        // word 0 lo: ge(-0, +0) FALSE, lt TRUE; hi: ge(+0, -0) TRUE, lt FALSE.
        assert_eq!(ge[0], 0xFFFF_0000);
        assert_eq!(lt[0], 0x0000_FFFF);
        // word 1 lo: -denorm vs +NaN -> both FALSE; hi: NaN vs 0 -> both FALSE.
        assert_eq!(ge[1], 0);
        assert_eq!(lt[1], 0);
    }

    // ========== vector_compare_lt ==========

    #[test]
    fn test_lt_is_complement_of_ge_i32() {
        let a = [5, 0xFFFF_FFFF, 10, 0, 100, 50, 0, 1];
        let b = [3, 0, 10, 1, 99, 51, 0, 0];
        let ge = VectorAlu::vector_compare_ge(&a, &b, ElementType::Int32);
        let lt = VectorAlu::vector_compare_lt(&a, &b, ElementType::Int32);
        // For every lane, exactly one of GE or LT should be all-ones
        for i in 0..8 {
            assert_eq!(ge[i] ^ lt[i], !0, "ge^lt should be all-ones at lane {}", i);
        }
    }

    #[test]
    fn test_lt_i16_complement() {
        let a = [0x0001_8000, 0x7FFF_0000, 0, 0, 0, 0, 0, 0];
        let b = [0x0000_7FFF, 0x8000_0001, 0, 0, 0, 0, 0, 0];
        let ge = VectorAlu::vector_compare_ge(&a, &b, ElementType::Int16);
        let lt = VectorAlu::vector_compare_lt(&a, &b, ElementType::Int16);
        for i in 0..8 {
            assert_eq!(ge[i] ^ lt[i], !0, "ge^lt should be all-ones at lane {}", i);
        }
    }

    // ========== vector_compare_eqz ==========

    #[test]
    fn test_eqz_i32() {
        let a = [0, 1, 0, 0xFFFF_FFFF, 0, 42, 0, 0];
        let r = VectorAlu::vector_compare_eqz(&a, ElementType::Int32);
        assert_eq!(r[0], !0); // 0 == 0
        assert_eq!(r[1], 0); // 1 != 0
        assert_eq!(r[2], !0); // 0 == 0
        assert_eq!(r[3], 0); // -1 != 0
        assert_eq!(r[4], !0); // 0 == 0
        assert_eq!(r[5], 0); // 42 != 0
    }

    #[test]
    fn test_eqz_i16() {
        // Word: lo=0x0000, hi=0x0001
        let a = [0x0001_0000, 0, 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_compare_eqz(&a, ElementType::Int16);
        // lo=0 -> true (0xFFFF), hi=1 -> false (0x0000)
        assert_eq!(r[0], 0x0000_FFFF);
    }

    #[test]
    fn test_eqz_f32_neg_zero() {
        // -0.0 should compare equal to zero
        let a = [f32::to_bits(-0.0), f32::to_bits(0.0), 0, 0, 0, 0, 0, 0];
        let r = VectorAlu::vector_compare_eqz(&a, ElementType::Float32);
        assert_eq!(r[0], !0); // -0.0 == 0.0
        assert_eq!(r[1], !0); // 0.0 == 0.0
    }

    // ========== vector_select ==========

    #[test]
    fn test_select_i32_all_from_src1() {
        let mask = [!0u32; 8]; // all true -> src1
        let src1 = [10, 20, 30, 40, 50, 60, 70, 80];
        let src2 = [1, 2, 3, 4, 5, 6, 7, 8];
        let r = VectorAlu::vector_select(&mask, &src1, &src2, ElementType::Int32);
        assert_eq!(r, src1);
    }

    #[test]
    fn test_select_i32_all_from_src2() {
        let mask = [0u32; 8]; // all false -> src2
        let src1 = [10, 20, 30, 40, 50, 60, 70, 80];
        let src2 = [1, 2, 3, 4, 5, 6, 7, 8];
        let r = VectorAlu::vector_select(&mask, &src1, &src2, ElementType::Int32);
        assert_eq!(r, src2);
    }

    #[test]
    fn test_select_i32_alternating() {
        let mask = [!0, 0, !0, 0, !0, 0, !0, 0];
        let src1 = [10, 20, 30, 40, 50, 60, 70, 80];
        let src2 = [1, 2, 3, 4, 5, 6, 7, 8];
        let r = VectorAlu::vector_select(&mask, &src1, &src2, ElementType::Int32);
        assert_eq!(r, [10, 2, 30, 4, 50, 6, 70, 8]);
    }

    #[test]
    fn test_select_i16_per_lane() {
        // mask: lo=0xFFFF (true), hi=0x0000 (false)
        let mask = [0x0000_FFFF; 8];
        let src1 = [0xAAAA_BBBB; 8];
        let src2 = [0xCCCC_DDDD; 8];
        let r = VectorAlu::vector_select(&mask, &src1, &src2, ElementType::Int16);
        // lo from src1, hi from src2
        assert_eq!(r[0], 0xCCCC_BBBB);
    }

    #[test]
    fn test_select_i8_per_byte() {
        // mask: byte0=0xFF, byte1=0x00, byte2=0xFF, byte3=0x00
        let mask = [0x00FF_00FF; 8];
        let src1 = [0xAA_BB_CC_DD; 8];
        let src2 = [0x11_22_33_44; 8];
        let r = VectorAlu::vector_select(&mask, &src1, &src2, ElementType::Int8);
        // byte0: mask=FF -> src1=0xDD, byte1: mask=00 -> src2=0x33
        // byte2: mask=FF -> src1=0xBB, byte3: mask=00 -> src2=0x11
        assert_eq!(r[0], 0x11BB_33DD);
    }

    // ========== Integration tests (existing) ==========

    #[test]
    fn test_vector_cmp() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [1, 0, 3, 0, 5, 0, 7, 0]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Cmp)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);
        assert_eq!(result[0], 0xFFFF_FFFF); // 1 == 1
        assert_eq!(result[1], 0); // 2 != 0
        assert_eq!(result[2], 0xFFFF_FFFF); // 3 == 3
        assert_eq!(result[3], 0); // 4 != 0
    }

    /// VGE with wide (x-register) sources and scalar dest produces a bitmask.
    #[test]
    fn test_wide_setge_scalar_dest_i32() {
        let mut ctx = make_ctx();
        // x0 = v0:v1 (lo:hi), x2 = v2:v3
        // lo half (v0): [10, 5, 20, 3, 8, 8, 100, 0]
        // hi half (v1): [1, 2, 3, 4, 5, 6, 7, 8]
        ctx.vector.write(0, [10, 5, 20, 3, 8, 8, 100, 0]);
        ctx.vector.write(1, [1, 2, 3, 4, 5, 6, 7, 8]);
        // lo half (v2): [5, 10, 20, 4, 7, 8, 50, 1]
        // hi half (v3): [1, 3, 2, 4, 6, 5, 7, 9]
        ctx.vector.write(2, [5, 10, 20, 4, 7, 8, 50, 1]);
        ctx.vector.write(3, [1, 3, 2, 4, 6, 5, 7, 9]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::SetGe)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::ScalarReg(16))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(2));
        op.is_wide_vector = true;

        assert!(VectorAlu::execute(&op, &mut ctx));

        // lo half comparison (a >= b):
        //   10>=5=T, 5>=10=F, 20>=20=T, 3>=4=F, 8>=7=T, 8>=8=T, 100>=50=T, 0>=1=F
        //   mask_lo = 0b_0111_0101 = 0x75
        // hi half comparison:
        //   1>=1=T, 2>=3=F, 3>=2=T, 4>=4=T, 5>=6=F, 6>=5=T, 7>=7=T, 8>=9=F
        //   mask_hi = 0b_0110_1101 = 0x6D
        // full_mask = mask_lo | (mask_hi << 8) = 0x6D75
        // The scalar mask has pipeline latency 2 (II_VCMP); flush to observe it.
        ctx.flush_pending_writes();
        let scalar_result = ctx.scalar.read(16);
        assert_eq!(scalar_result, 0x6D75, "wide VGE scalar bitmask mismatch: got {:#06x}", scalar_result);
    }

    /// VLT with wide (x-register) sources and scalar dest produces a bitmask.
    #[test]
    fn test_wide_setlt_scalar_dest_i32() {
        let mut ctx = make_ctx();
        // Same data as above
        ctx.vector.write(0, [10, 5, 20, 3, 8, 8, 100, 0]);
        ctx.vector.write(1, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(2, [5, 10, 20, 4, 7, 8, 50, 1]);
        ctx.vector.write(3, [1, 3, 2, 4, 6, 5, 7, 9]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::SetLt)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::ScalarReg(16))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(2));
        op.is_wide_vector = true;

        assert!(VectorAlu::execute(&op, &mut ctx));

        // VLT is the complement of VGE (for signed integers, no NaN).
        // lo: F,T,F,T,F,F,F,T -> 0b_1000_1010 = 0x8A
        // hi: F,T,F,F,T,F,F,T -> 0b_1001_0010 = 0x92
        // full_mask = 0x928A
        // The scalar mask has pipeline latency 2 (II_VCMP); flush to observe it.
        ctx.flush_pending_writes();
        let scalar_result = ctx.scalar.read(16);
        assert_eq!(scalar_result, 0x928A, "wide VLT scalar bitmask mismatch: got {:#06x}", scalar_result);
    }

    /// Regression: the scalar mask of a vector compare is pipelined (latency
    /// 2, NoBypass per AIE2Schedule.td II_VMAX_LT/II_VCMP), so the next bundle
    /// must still read the PRE-compare value. Peano schedules callee-save
    /// copies of the mask register in that shadow (e.g. `vmax_lt x4,r16,..`
    /// then `or r1,r16,r16`); an immediate write corrupts the saved value,
    /// which fuzzer kernels then use as the lock-release delta (lock wedged
    /// at -64, output DMA stalls).
    #[test]
    fn test_cmp_scalar_mask_not_visible_to_next_bundle() {
        let mut ctx = make_ctx();
        ctx.scalar.write(16, 1); // caller value (lock-release amount)
        ctx.vector.write(0, [10, 5, 20, 3, 8, 8, 100, 0]);
        ctx.vector.write(1, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(2, [5, 10, 20, 4, 7, 8, 50, 1]);
        ctx.vector.write(3, [1, 3, 2, 4, 6, 5, 7, 9]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::SetGe)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::ScalarReg(16))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(2));
        op.is_wide_vector = true;

        assert!(VectorAlu::execute(&op, &mut ctx));

        // Latency shadow: a read in the next bundle sees the old value.
        assert_eq!(ctx.scalar.read(16), 1, "compare mask forwarded too early");

        // After the pipeline drains the mask becomes architectural.
        ctx.flush_pending_writes();
        assert_eq!(ctx.scalar.read(16), 0x6D75);
    }
}
