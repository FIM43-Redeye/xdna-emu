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

            Operation::VectorMatMulDense { element_type } => {
                // Dense matrix multiply: acc += A * B
                // AIE2 processes 4x4 or 4x8 matrix tiles
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_matmul_dense(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            Operation::VectorMatMulSparse { element_type, .. } => {
                // Sparse matrix multiply: similar to dense but with sparse format
                // For now, treat as dense - sparse format optimization is a refinement
                let (a, b) = Self::get_two_vector_sources(op, ctx);
                let acc_reg = Self::get_acc_dest(op);
                Self::vector_matmul_dense(ctx, acc_reg, &a, &b, *element_type);
                true
            }

            Operation::VectorSRS { from_type, to_type } => {
                // Shift-Round-Saturate: convert accumulator to vector
                let acc_reg = Self::get_acc_source(op);
                let shift = Self::get_shift_amount(op, ctx);
                let result = Self::vector_srs(ctx, acc_reg, shift, *from_type, *to_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorConvert { from_type, to_type } => {
                // Type conversion (e.g., bf16 <-> f32)
                let src = Self::get_vector_source(op, ctx, 0);
                let result = Self::vector_convert(&src, *from_type, *to_type);
                Self::write_vector_dest(op, ctx, result);
                true
            }

            Operation::VectorMov { .. } => {
                // Vector register move
                let src = Self::get_vector_source(op, ctx, 0);
                Self::write_vector_dest(op, ctx, src);
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

    /// Get accumulator source register from operands.
    fn get_acc_source(op: &SlotOp) -> u8 {
        for src in &op.sources {
            if let Operand::AccumReg(r) = src {
                return *r;
            }
        }
        0
    }

    /// Get shift amount from operands (immediate or register).
    fn get_shift_amount(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // Look for immediate in sources
        for src in &op.sources {
            if let Operand::Immediate(imm) = src {
                return *imm as u32;
            }
            if let Operand::ScalarReg(r) = src {
                return ctx.scalar.read(*r);
            }
        }
        0 // Default: no shift
    }

    /// Dense matrix multiply: acc += A * B (operates on 4x4 or 4x8 tiles).
    ///
    /// For AIE2, this processes matrix tiles where:
    /// - A: activation matrix (from vector register)
    /// - B: weight matrix (from vector register)
    /// - Result accumulates to 512-bit accumulator
    fn vector_matmul_dense(
        ctx: &mut ExecutionContext,
        acc_reg: u8,
        a: &[u32; 8],
        b: &[u32; 8],
        elem_type: ElementType,
    ) {
        let current = ctx.accumulator.read(acc_reg);

        match elem_type {
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 x int8 elements: interpret as 4x8 * 8x4 = 4x4 output
                // Each u32 contains 4 int8 values
                // Simplified: element-wise multiply-accumulate for compatibility
                let mut new_acc = current;
                for i in 0..8 {
                    let mut acc = current[i];
                    for byte_idx in 0..4 {
                        let a_byte = ((a[i] >> (byte_idx * 8)) & 0xFF) as u8;
                        let b_byte = ((b[i] >> (byte_idx * 8)) & 0xFF) as u8;
                        let prod = if matches!(elem_type, ElementType::Int8) {
                            (a_byte as i8 as i64) * (b_byte as i8 as i64)
                        } else {
                            (a_byte as i64) * (b_byte as i64)
                        };
                        acc = acc.wrapping_add(prod as u64);
                    }
                    new_acc[i] = acc;
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 16 x int16 elements: interpret as matrix multiply
                let mut new_acc = current;
                for i in 0..8 {
                    let a_lo = (a[i] & 0xFFFF) as i16 as i32;
                    let a_hi = ((a[i] >> 16) & 0xFFFF) as i16 as i32;
                    let b_lo = (b[i] & 0xFFFF) as i16 as i32;
                    let b_hi = ((b[i] >> 16) & 0xFFFF) as i16 as i32;
                    // Accumulate products
                    new_acc[i] = current[i]
                        .wrapping_add((a_lo * b_lo) as u64)
                        .wrapping_add((a_hi * b_hi) as u64);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::BFloat16 => {
                // 16 x bf16 matrix multiply
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
            ElementType::Int32 | ElementType::UInt32 => {
                // 8 x int32 matrix multiply-accumulate
                let mut new_acc = current;
                for i in 0..8 {
                    let prod = (a[i] as u64) * (b[i] as u64);
                    new_acc[i] = current[i].wrapping_add(prod);
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
            ElementType::Float32 => {
                // 8 x f32 matrix multiply
                let mut new_acc = current;
                for i in 0..8 {
                    let fa = f32::from_bits(a[i]) as f64;
                    let fb = f32::from_bits(b[i]) as f64;
                    let current_f = f64::from_bits(current[i]);
                    new_acc[i] = (current_f + fa * fb).to_bits();
                }
                ctx.accumulator.write(acc_reg, new_acc);
            }
        }
    }

    /// Shift-Round-Saturate: convert 64-bit accumulator lanes to narrower output.
    ///
    /// This is used after MAC operations to normalize results back to vector format.
    fn vector_srs(
        ctx: &ExecutionContext,
        acc_reg: u8,
        shift: u32,
        _from_type: ElementType,
        to_type: ElementType,
    ) -> [u32; 8] {
        let acc = ctx.accumulator.read(acc_reg);
        let mut result = [0u32; 8];

        match to_type {
            ElementType::Int32 | ElementType::UInt32 => {
                // 64-bit -> 32-bit with shift and saturation
                for i in 0..8 {
                    let shifted = if shift > 0 {
                        // Round: add 0.5 ULP before shift
                        let round_bit = 1u64 << (shift - 1);
                        acc[i].wrapping_add(round_bit) >> shift
                    } else {
                        acc[i]
                    };
                    // Saturate to 32 bits
                    result[i] = shifted.min(u32::MAX as u64) as u32;
                }
            }
            ElementType::Int16 | ElementType::UInt16 => {
                // 64-bit -> 16-bit (pack 2 lanes into each u32)
                for i in 0..4 {
                    let val0 = {
                        let shifted = if shift > 0 {
                            let round_bit = 1u64 << (shift - 1);
                            acc[i * 2].wrapping_add(round_bit) >> shift
                        } else {
                            acc[i * 2]
                        };
                        shifted.min(u16::MAX as u64) as u16
                    };
                    let val1 = {
                        let shifted = if shift > 0 {
                            let round_bit = 1u64 << (shift - 1);
                            acc[i * 2 + 1].wrapping_add(round_bit) >> shift
                        } else {
                            acc[i * 2 + 1]
                        };
                        shifted.min(u16::MAX as u64) as u16
                    };
                    result[i] = (val0 as u32) | ((val1 as u32) << 16);
                }
            }
            ElementType::BFloat16 => {
                // 64-bit float -> bf16 (pack 2 per u32)
                for i in 0..4 {
                    let f0 = f64::from_bits(acc[i * 2]) as f32;
                    let f1 = f64::from_bits(acc[i * 2 + 1]) as f32;
                    let bf0 = Self::f32_to_bf16(f0);
                    let bf1 = Self::f32_to_bf16(f1);
                    result[i] = (bf0 as u32) | ((bf1 as u32) << 16);
                }
            }
            ElementType::Float32 => {
                // 64-bit float -> 32-bit float
                for i in 0..8 {
                    let f = f64::from_bits(acc[i]) as f32;
                    result[i] = f.to_bits();
                }
            }
            _ => {}
        }

        result
    }

    /// Vector type conversion.
    fn vector_convert(src: &[u32; 8], from_type: ElementType, to_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match (from_type, to_type) {
            // BFloat16 -> Float32 (expand: 16 bf16 -> 8 f32, use lower half)
            (ElementType::BFloat16, ElementType::Float32) => {
                for i in 0..8 {
                    // Take lower bf16 from each pair
                    let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
                    result[i] = Self::bf16_to_f32(bf16).to_bits();
                }
            }
            // Float32 -> BFloat16 (pack: 8 f32 -> 16 bf16, store in lower 4 words)
            (ElementType::Float32, ElementType::BFloat16) => {
                for i in 0..4 {
                    let f0 = f32::from_bits(src[i * 2]);
                    let f1 = f32::from_bits(src[i * 2 + 1]);
                    let bf0 = Self::f32_to_bf16(f0);
                    let bf1 = Self::f32_to_bf16(f1);
                    result[i] = (bf0 as u32) | ((bf1 as u32) << 16);
                }
            }
            // Int32 -> Float32
            (ElementType::Int32, ElementType::Float32) => {
                for i in 0..8 {
                    result[i] = (src[i] as i32 as f32).to_bits();
                }
            }
            // UInt32 -> Float32
            (ElementType::UInt32, ElementType::Float32) => {
                for i in 0..8 {
                    result[i] = (src[i] as f32).to_bits();
                }
            }
            // Float32 -> Int32
            (ElementType::Float32, ElementType::Int32) => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
                    result[i] = f as i32 as u32;
                }
            }
            // Float32 -> UInt32
            (ElementType::Float32, ElementType::UInt32) => {
                for i in 0..8 {
                    let f = f32::from_bits(src[i]);
                    result[i] = f.max(0.0) as u32;
                }
            }
            // Int16 -> Int32 (expand lower half)
            (ElementType::Int16, ElementType::Int32) => {
                for i in 0..8 {
                    let i16_val = (src[i / 2] >> ((i % 2) * 16)) as i16;
                    result[i] = i16_val as i32 as u32;
                }
            }
            // Same type: pass through
            _ if from_type == to_type => {
                result = *src;
            }
            _ => {
                // Unhandled conversion: pass through
                result = *src;
            }
        }

        result
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

    #[test]
    fn test_vector_matmul_dense_int32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        ctx.vector.write(1, [2, 2, 2, 2, 2, 2, 2, 2]);

        let op = SlotOp::new(
            SlotIndex::Accumulator,
            Operation::VectorMatMulDense {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::AccumReg(0))
        .with_source(Operand::VectorReg(0))
        .with_source(Operand::VectorReg(1));

        VectorAlu::execute(&op, &mut ctx);
        let acc = ctx.accumulator.read(0);
        // Each lane: a[i] * b[i] accumulated
        assert_eq!(acc, [2, 4, 6, 8, 10, 12, 14, 16]);

        // Accumulate again
        VectorAlu::execute(&op, &mut ctx);
        let acc = ctx.accumulator.read(0);
        assert_eq!(acc, [4, 8, 12, 16, 20, 24, 28, 32]);
    }

    #[test]
    fn test_vector_srs_int32() {
        let mut ctx = make_ctx();
        // Set up accumulator with values that need shifting
        ctx.accumulator.write(0, [256, 512, 768, 1024, 1280, 1536, 1792, 2048]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorSRS {
                from_type: ElementType::Int32,
                to_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::AccumReg(0))
        .with_source(Operand::Immediate(4)); // Shift right by 4

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(0);
        // 256 >> 4 = 16, 512 >> 4 = 32, etc (with rounding)
        assert_eq!(result[0], 16);
        assert_eq!(result[1], 32);
        assert_eq!(result[2], 48);
        assert_eq!(result[3], 64);
    }

    #[test]
    fn test_vector_convert_bf16_to_f32() {
        let mut ctx = make_ctx();
        // Create bf16 values: 1.0, 2.0, 3.0, 4.0 (packed 2 per u32)
        let bf16_1 = VectorAlu::f32_to_bf16(1.0);
        let bf16_2 = VectorAlu::f32_to_bf16(2.0);
        let bf16_3 = VectorAlu::f32_to_bf16(3.0);
        let bf16_4 = VectorAlu::f32_to_bf16(4.0);
        ctx.vector.write(
            0,
            [
                (bf16_1 as u32) | ((bf16_2 as u32) << 16),
                (bf16_3 as u32) | ((bf16_4 as u32) << 16),
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        );

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorConvert {
                from_type: ElementType::BFloat16,
                to_type: ElementType::Float32,
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(1);

        // First 4 lanes should be 1.0, 2.0, 3.0, 4.0
        let f0 = f32::from_bits(result[0]);
        let f1 = f32::from_bits(result[1]);
        let f2 = f32::from_bits(result[2]);
        let f3 = f32::from_bits(result[3]);
        assert!((f0 - 1.0).abs() < 0.01, "Expected 1.0, got {}", f0);
        assert!((f1 - 2.0).abs() < 0.01, "Expected 2.0, got {}", f1);
        assert!((f2 - 3.0).abs() < 0.01, "Expected 3.0, got {}", f2);
        assert!((f3 - 4.0).abs() < 0.01, "Expected 4.0, got {}", f3);
    }

    #[test]
    fn test_vector_convert_int32_to_f32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorConvert {
                from_type: ElementType::Int32,
                to_type: ElementType::Float32,
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(1);

        for i in 0..8 {
            let f = f32::from_bits(result[i]);
            assert_eq!(f, (i + 1) as f32);
        }
    }

    #[test]
    fn test_vector_mov() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMov {
                element_type: ElementType::Int32,
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [10, 20, 30, 40, 50, 60, 70, 80]);
    }
}
