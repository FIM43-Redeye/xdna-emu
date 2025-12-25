//! Vector ALU execution unit.
//!
//! Handles SIMD operations on 256-bit vectors (8 × 32-bit, 16 × 16-bit, 32 × 8-bit).
//!
//! # Operations
//!
//! - **Arithmetic**: vadd, vsub, vmul
//! - **MAC**: vmac (multiply-accumulate to 512-bit accumulator)
//! - **Compare**: vcmp, vmin, vmax
//! - **Shuffle**: vshuffle, vpack, vunpack

use crate::interpreter::bundle::{ElementType, Operation, Operand, ShufflePattern, SlotOp};
use crate::interpreter::state::ExecutionContext;

/// Vector ALU execution unit.
pub struct VectorAlu;

impl VectorAlu {
    /// Execute a vector operation.
    ///
    /// Returns `true` if the operation was handled, `false` if not a vector op.
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
        match &op.op {
            Operation::VectorAdd { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_add(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorSub { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_sub(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMul { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_mul(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMac { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_mac(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            Operation::VectorMin { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_min(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMax { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_max(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorShuffle { pattern } => {
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_shuffle(&src, *pattern);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorPack => {
                // Pack two vectors into one (narrow)
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_pack(&a, &b);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorUnpack => {
                // Unpack one vector into two (widen) - writes low half
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_unpack_low(&src);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorCmp { element_type } => {
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let result = Self::vector_cmp_eq(&a, &b, *element_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            _ => false, // Not a vector operation
        }
    }

    /// Get two vector source operands.
    fn get_two_vector_sources(op: &SlotOp, ctx: &ExecutionContext) -> ([u32; 8], [u32; 8]) {
        let a = Self::get_vector_source(op, ctx, 0);
        let b = Self::get_vector_source(op, ctx, 1);
        (a, b)
    }

    /// Get a single vector source operand.
    fn get_vector_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> [u32; 8] {
        op.sources
            .get(idx)
            .map_or([0; 8], |src| Self::read_vector_operand(src, ctx))
    }

    /// Read a vector operand value.
    fn read_vector_operand(operand: &Operand, ctx: &ExecutionContext) -> [u32; 8] {
        match operand {
            Operand::VectorReg(r) => ctx.vector.read(*r),
            _ => [0; 8],
        }
    }

    /// Write result to vector destination.
    fn write_vector_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: [u32; 8]) {
        if let Some(Operand::VectorReg(r)) = &op.dest {
            ctx.vector.write(*r, value);
        }
    }

    /// Get accumulator destination register.
    fn get_acc_dest(op: &SlotOp) -> u8 {
        match &op.dest {
            Some(Operand::AccumReg(r)) => *r,
            _ => 0,
        }
    }

    /// Convert BFloat16 (upper 16 bits of f32) to f32.
    #[inline]
    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// Convert f32 to BFloat16 (truncate lower 16 bits).
    #[inline]
    fn f32_to_bf16(val: f32) -> u16 {
        (val.to_bits() >> 16) as u16
    }

    /// Vector addition by element type.
    fn vector_add(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
                // 8 × 32-bit integer lanes
                for i in 0..8 {
                    result[i] = a[i].wrapping_add(b[i]);
                }
            }
            ElementType::Float32 => {
                // 8 × 32-bit IEEE 754 float lanes
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]);
                    let fb = f32::from_bits(b[i]);
                    result[i] = (fa + fb).to_bits();
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 16 × 16-bit integer lanes (2 per u32)
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
                // 16 × BFloat16 lanes (2 per u32)
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
                // 32 × 8-bit lanes (4 per u32)
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
    fn vector_sub(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
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
    fn vector_mul(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
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

    /// Vector multiply-accumulate (adds to 512-bit accumulator).
    fn vector_mac(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        a: &[u32; 8],
        b: &[u32; 8],
        elem_type: ElementType,
    ) {
        let current = ctx.accumulator.read(acc_reg);

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 => {
                // 8 × 32-bit → 8 × 64-bit accumulator lanes
                let mut new_acc = [0u64; 8];
                for i in 0..8 {
                    let prod = (a[i] as u64) * (b[i] as u64);
                    new_acc[i] = current[i].wrapping_add(prod);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Float32 => {
                // 8 × f32 MAC: accumulator holds f64 for precision
                let mut new_acc = current;
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]) as f64;
                    let fb = f32::from_bits(b[i]) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + fa * fb).to_bits();
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 16 × 16-bit → 16 products, accumulated
                let mut new_acc = current;
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF);
                    let a_hi = ((a[i] >> 16) & 0xFFFF);
                    let b_lo = (b[i] & 0xFFFF);
                    let b_hi = ((b[i] >> 16) & 0xFFFF);
                    let prod_lo = a_lo * b_lo;
                    let prod_hi = a_hi * b_hi;
                    // Accumulate both products into single 64-bit lane
                    new_acc[i] = current[i]
                        .wrapping_add(prod_lo as u64)
                        .wrapping_add(prod_hi as u64);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::BFloat16 => {
                // 16 × bf16 MAC: convert to f32, accumulate to f64
                let mut new_acc = current;
                for i in 0..8 {
                    let a_lo = Self::bf16_to_f32((a[i] & 0xFFFF) as u16) as f64;
                    let a_hi = Self::bf16_to_f32(((a[i] >> 16) & 0xFFFF) as u16) as f64;
                    let b_lo = Self::bf16_to_f32((b[i] & 0xFFFF) as u16) as f64;
                    let b_hi = Self::bf16_to_f32(((b[i] >> 16) & 0xFFFF) as u16) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + a_lo * b_lo + a_hi * b_hi).to_bits();
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 × 8-bit → 32 products, accumulated
                let mut new_acc = current;
                for i in 0..8 {
                    let mut sum = 0u64;
                    for j in 0..4 {
                        let a_byte = ((a[i] >> (j * 8)) & 0xFF) as u64;
                        let b_byte = ((b[i] >> (j * 8)) & 0xFF) as u64;
                        sum += a_byte * b_byte;
                    }
                    new_acc[i] = current[i].wrapping_add(sum);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
        }
    }

    /// Vector minimum by element type.
    fn vector_min(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    result[i] = std::cmp::min(a[i] as i32, b[i] as i32) as u32;
                }
            }
            ElementType::UInt32 => {
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
    fn vector_max(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 => {
                for i in 0..8 {
                    result[i] = std::cmp::max(a[i] as i32, b[i] as i32) as u32;
                }
            }
            ElementType::UInt32 => {
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

    /// Vector shuffle with pattern.
    fn vector_shuffle(src: &[u32; 8], pattern: ShufflePattern) -> [u32; 8] {
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

    /// Pack two 32-bit vectors into one 16-bit vector.
    fn vector_pack(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];

        // Pack lower halves: take low 16 bits of each input lane
        for i in 0..4 {
            let a_lo = (a[i * 2] & 0xFFFF);
            let a_hi = (a[i * 2 + 1] & 0xFFFF);
            result[i] = a_lo | (a_hi << 16);
        }
        for i in 0..4 {
            let b_lo = (b[i * 2] & 0xFFFF);
            let b_hi = (b[i * 2 + 1] & 0xFFFF);
            result[i + 4] = b_lo | (b_hi << 16);
        }

        result
    }

    /// Unpack low half (sign-extend 16-bit to 32-bit).
    fn vector_unpack_low(src: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];

        for i in 0..4 {
            // Sign-extend low 16 bits
            let val = (src[i] & 0xFFFF) as i16 as i32 as u32;
            result[i * 2] = val;
            // Sign-extend high 16 bits
            let val = ((src[i] >> 16) & 0xFFFF) as i16 as i32 as u32;
            result[i * 2 + 1] = val;
        }

        result
    }

    /// Vector equality comparison (returns mask).
    fn vector_cmp_eq(a: &[u32; 8], b: &[u32; 8], elem_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match elem_type {
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    #[test]
    fn test_vector_add_i32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorAdd {
                element_type: ElementType::Int32,
            },
        )
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

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorAdd {
                element_type: ElementType::Int16,
            },
        )
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

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorSub {
                element_type: ElementType::Int32,
            },
        )
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

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMul {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    fn test_vector_mac() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [2, 2, 2, 2, 2, 2, 2, 2]);

        // acc0 = v0 * v1
        let op = SlotOp::new(
            SlotIndex::Accumulator,
            Operation::VectorMac {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::AccumReg(0))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let acc = ctx.accumulator.read(0);
        assert_eq!(acc, [2, 4, 6, 8, 10, 12, 14, 16]);

        // Accumulate again
        VectorAlu::execute(&op, &mut ctx);
        let acc = ctx.accumulator.read(0);
        assert_eq!(acc, [4, 8, 12, 16, 20, 24, 28, 32]);
    }

    #[test]
    fn test_vector_min_max() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [5, 10, 15, 20, 25, 30, 35, 40]);
        ctx.vector.write(1, [10, 5, 20, 15, 30, 25, 40, 35]);

        // Min
        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMin {
                element_type: ElementType::UInt32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [5, 5, 15, 15, 25, 25, 35, 35]);

        // Max
        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMax {
                element_type: ElementType::UInt32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(2), [10, 10, 20, 20, 30, 30, 40, 40]);
    }

    #[test]
    fn test_vector_shuffle_reverse() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorShuffle {
                pattern: ShufflePattern::Reverse,
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_vector_shuffle_broadcast() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorShuffle {
                pattern: ShufflePattern::Broadcast(2),
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [30, 30, 30, 30, 30, 30, 30, 30]);
    }

    #[test]
    fn test_vector_cmp() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [1, 0, 3, 0, 5, 0, 7, 0]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorCmp {
                element_type: ElementType::Int32,
            },
        )
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

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorAdd {
                element_type: ElementType::Float32,
            },
        )
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

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMul {
                element_type: ElementType::Float32,
            },
        )
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
        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMin {
                element_type: ElementType::Float32,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(2);
        assert_eq!(f32::from_bits(result[0]), 0.5);  // min(1.0, 0.5)
        assert_eq!(f32::from_bits(result[1]), -2.0); // min(-2.0, -1.0)
        assert_eq!(f32::from_bits(result[3]), -5.0); // min(-4.0, -5.0)

        // Max
        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMax {
                element_type: ElementType::Float32,
            },
        )
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

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorAdd {
                element_type: ElementType::BFloat16,
            },
        )
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
}
