//! Type conversion operations (VCONV, VFLOOR).
//!
//! Handles Convert dispatch for both narrow (256-bit) and wide (512-bit / 1024-bit)
//! paths. The wide path is the most complex single operation in the vector ALU,
//! with multiple sub-paths for accumulator sources, expansion, contraction, and
//! same-width conversions.

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;

use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    /// Dispatch entry point for Convert (VCONV/VFLOOR) operations.
    ///
    /// Handles both narrow and wide paths, including accumulator sources
    /// with AccumWidth detection (Quarter/Half/Full).
    pub(super) fn execute_convert(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let has_wide_acc_source = matches!(op.accum_width,
            Some(crate::interpreter::decode::register_map::AccumWidth::Full)
            | Some(crate::interpreter::decode::register_map::AccumWidth::Half));
        let is_quarter_acc = matches!(op.accum_width,
            Some(crate::interpreter::decode::register_map::AccumWidth::QuarterLow)
            | Some(crate::interpreter::decode::register_map::AccumWidth::QuarterHigh));

        if op.is_wide_vector || has_wide_acc_source || is_quarter_acc {
            Self::execute_convert_wide(op, ctx, et)
        } else {
            Self::execute_convert_narrow(op, ctx, et)
        }
    }

    /// Narrow Convert path (256-bit).
    ///
    /// Type conversion (e.g., bf16 <-> f32).
    /// Expansion (bf16->fp32) writes to accumulator;
    /// contraction (fp32->bf16) writes to vector register.
    fn execute_convert_narrow(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let src = Self::get_vector_source(op, ctx, 0);
        let from = op.from_type.unwrap_or(ElementType::Int32);
        let is_expansion = from.bits() < et.bits();
        if is_expansion && matches!(op.dest, Some(Operand::AccumReg(_))) {
            // BF16->FP32 expansion: 8 bf16 -> 8 fp32 packed into accum.
            let result = Self::vector_convert(&src, from, et);
            let dst_reg = Self::get_acc_dest(op);
            let mut acc = [0u64; 8];
            for i in 0..8 {
                acc[i] = result[i] as u64;
            }
            ctx.accumulator.write(dst_reg, acc);
        } else {
            let result = Self::vector_convert(&src, from, et);
            Self::write_vector_dest(op, ctx, result);
        }
        true
    }

    /// Wide Convert path (512-bit / 1024-bit).
    ///
    /// VCONV/VFLOOR: type conversion with multiple sub-paths:
    /// A. Accumulator source (VCONV_FP32_BF16, VCONV_BF16_FP32): read acc,
    ///    extract lower 32 bits per lane as the input vector.
    /// B. Narrow vector source (VFLOOR bf16->s32): 256-bit in, 512-bit out.
    /// C. Wide vector source: 512-bit in, 512-bit out.
    ///
    /// VFLOOR has a shift operand (scalar register) that scales:
    ///   result[i] = saturate_s32(floor(bf16_value * 2^shift))
    fn execute_convert_wide(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let from = op.from_type.unwrap_or(ElementType::Int32);

        // Extract shift from scalar source (VFLOOR only).
        let shift_val: Option<i32> = op.sources.iter().find_map(|s| {
            if let Operand::ScalarReg(r) = s {
                Some(ctx.scalar_read(*r) as i32)
            } else {
                None
            }
        });
        let is_vfloor = from == ElementType::BFloat16
            && et == ElementType::Int32
            && shift_val.is_some();
        let has_acc_source = op.sources.iter()
            .any(|s| matches!(s, Operand::AccumReg(_)));

        if has_acc_source {
            // Accumulator source: read acc data as raw bits, then convert.
            //
            // Accumulator access widths (from LLVM register class):
            //   QuarterLow  (amll/amhl): 256 bits = lanes 0-3 of a bm register
            //   QuarterHigh (amlh/amhh): 256 bits = lanes 4-7 of a bm register
            //   Half        (bml/bmh):   512 bits = 8 lanes
            //   Full        (cm):       1024 bits = 16 lanes (2 bm registers)
            //
            // The raw bits are reinterpreted according to `from` type -- e.g.,
            // VFLOOR reads a quarter as 16 packed BF16 values, not as lane values.
            let acc_reg = Self::get_acc_source(op);
            let is_quarter = matches!(op.accum_width,
                Some(crate::interpreter::decode::register_map::AccumWidth::QuarterLow) |
                Some(crate::interpreter::decode::register_map::AccumWidth::QuarterHigh));
            // Only Full (cm-class) is truly wide (1024-bit, 16 lanes).
            // Half (bml/bmh) is 512-bit = 8 lanes. When accum_width is
            // None, default to half -- the even-register heuristic was
            // wrong because bml registers ARE even-numbered.
            let is_wide = matches!(op.accum_width,
                Some(crate::interpreter::decode::register_map::AccumWidth::Full));

            let (src_lo, src_hi) = if is_quarter {
                // Quarter-accumulator: 256 bits = 4 u64 lanes.
                // Repack as 8 u32 words (raw byte reinterpretation).
                let acc = ctx.accumulator.read(acc_reg);
                let lane_start = match op.accum_width {
                    Some(crate::interpreter::decode::register_map::AccumWidth::QuarterHigh) => 4,
                    _ => 0,
                };
                let mut words = [0u32; 8];
                for i in 0..4 {
                    words[i * 2] = acc[lane_start + i] as u32;
                    words[i * 2 + 1] = (acc[lane_start + i] >> 32) as u32;
                }
                // Split into two halves for vector_convert (4 words each).
                let mut lo = [0u32; 8];
                let mut hi = [0u32; 8];
                lo[..4].copy_from_slice(&words[..4]);
                hi[..4].copy_from_slice(&words[4..]);
                (lo, hi)
            } else if is_wide {
                let acc = ctx.accumulator.read_wide(acc_reg);
                let mut lo = [0u32; 8];
                let mut hi = [0u32; 8];
                for i in 0..8 {
                    lo[i] = acc[i] as u32;
                    hi[i] = acc[i + 8] as u32;
                }
                (lo, hi)
            } else {
                // Half (bml/bmh): 8 u64 lanes.
                // Acc32 mode packs two 32-bit values per lane (consecutive):
                //   lane[i] = {val[2i+1], val[2i]}
                // Extract all 16 values in consecutive order, split into two
                // halves of 8 for vector_convert processing.
                let acc = ctx.accumulator.read(acc_reg);
                let mut all = [0u32; 16];
                for i in 0..8 {
                    all[i * 2] = acc[i] as u32;         // val[2i]
                    all[i * 2 + 1] = (acc[i] >> 32) as u32; // val[2i+1]
                }
                let mut lo = [0u32; 8];
                let mut hi = [0u32; 8];
                lo.copy_from_slice(&all[..8]);
                hi.copy_from_slice(&all[8..]);
                (lo, hi)
            };
            let (res_lo, res_hi) = if is_vfloor {
                let s = shift_val.unwrap();
                (Self::vector_floor_bf16_to_s32(&src_lo, s),
                 Self::vector_floor_bf16_to_s32(&src_hi, s))
            } else {
                (Self::vector_convert(&src_lo, from, et),
                 Self::vector_convert(&src_hi, from, et))
            };
            // VCONV may produce narrow output (e.g., f32->bf16 packs 8 f32
            // into 4 words of bf16). Write to the appropriate dest width.
            if is_quarter && from.bits() < et.bits() {
                // Quarter expansion (e.g., VFLOOR bf16->s32): 256 bits ->
                // 512 bits. Both halves produce full 8-word results.
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&res_lo);
                result[8..].copy_from_slice(&res_hi);
                Self::write_wide_vec_dest(op, ctx, result);
            } else if et.bits() < from.bits() {
                // Contraction: 512-bit acc -> 256-bit vector.
                // Each half contracts; pack results into one 256-bit output.
                let words_per_half = (8 * et.bits() as usize) / (from.bits() as usize);
                let mut result = [0u32; 8];
                result[..words_per_half].copy_from_slice(&res_lo[..words_per_half]);
                result[words_per_half..words_per_half * 2]
                    .copy_from_slice(&res_hi[..words_per_half]);
                Self::write_vector_dest(op, ctx, result);
            } else {
                // Same-width or expansion from half/full acc.
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&res_lo);
                result[8..].copy_from_slice(&res_hi);
                Self::write_wide_vec_dest(op, ctx, result);
            }
        } else {
            let is_expansion = from.bits() < et.bits();
            if is_expansion {
                // Narrow source (256-bit w-register) -> wide dest (512-bit x/acc).
                let src = Self::get_vector_source(op, ctx, 0);
                let mut lo_in = [0u32; 8];
                lo_in[..4].copy_from_slice(&src[..4]);
                let mut hi_in = [0u32; 8];
                hi_in[..4].copy_from_slice(&src[4..]);
                let (res_lo, res_hi) = if is_vfloor {
                    let s = shift_val.unwrap();
                    (Self::vector_floor_bf16_to_s32(&lo_in, s),
                     Self::vector_floor_bf16_to_s32(&hi_in, s))
                } else {
                    (Self::vector_convert(&lo_in, from, et),
                     Self::vector_convert(&hi_in, from, et))
                };
                if matches!(op.dest, Some(Operand::AccumReg(_))) {
                    // Accumulator dest: pack 16 x u32 into 8 x u64.
                    // Acc32 layout: consecutive pairs per lane:
                    //   lane[i] = {val[2i+1], val[2i]}
                    let dst_reg = Self::get_acc_dest(op);
                    let mut all = [0u32; 16];
                    all[..8].copy_from_slice(&res_lo);
                    all[8..].copy_from_slice(&res_hi);
                    let mut acc = [0u64; 8];
                    for i in 0..8 {
                        acc[i] = (all[i * 2] as u64) | ((all[i * 2 + 1] as u64) << 32);
                    }
                    ctx.accumulator.write(dst_reg, acc);
                } else {
                    let mut result = [0u32; 16];
                    result[..8].copy_from_slice(&res_lo);
                    result[8..].copy_from_slice(&res_hi);
                    Self::write_wide_vec_dest(op, ctx, result);
                }
            } else {
                // Same-width: 512-bit source, 512-bit dest.
                let src = Self::get_wide_vec_source(op, ctx, 0);
                let src_lo: [u32; 8] = src[..8].try_into().unwrap();
                let src_hi: [u32; 8] = src[8..].try_into().unwrap();
                let res_lo = Self::vector_convert(&src_lo, from, et);
                let res_hi = Self::vector_convert(&src_hi, from, et);
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&res_lo);
                result[8..].copy_from_slice(&res_hi);
                Self::write_wide_vec_dest(op, ctx, result);
            }
        }
        true
    }

    /// Vector type conversion.
    ///
    /// Used by standalone `VCONV` and fused `vlda.conv` / `vst.conv`.
    pub(crate) fn vector_convert(src: &[u32; 8], from_type: ElementType, to_type: ElementType) -> [u32; 8] {
        let mut result = [0u32; 8];

        match (from_type, to_type) {
            // BFloat16 -> Float32 (expand: 16 bf16 -> 8 f32, use lower half)
            (ElementType::BFloat16, ElementType::Float32) => {
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..8 {
                    // Take lower bf16 from each pair
                    let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
                    // AIE2 FTZ: bf16 denormals (exp=0) flush to signed zero.
                    result[i] = fp32_flush_to_zero(Self::bf16_to_f32(bf16).to_bits());
                }
            }
            // Float32 -> BFloat16 (pack: 8 f32 -> 16 bf16, store in lower 4 words)
            (ElementType::Float32, ElementType::BFloat16) => {
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..4 {
                    // AIE2 FTZ on inputs before conversion.
                    let f0 = f32::from_bits(fp32_flush_to_zero(src[i * 2]));
                    let f1 = f32::from_bits(fp32_flush_to_zero(src[i * 2 + 1]));
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
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..8 {
                    let f = f32::from_bits(fp32_flush_to_zero(src[i]));
                    result[i] = f as i32 as u32;
                }
            }
            // Float32 -> UInt32
            (ElementType::Float32, ElementType::UInt32) => {
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..8 {
                    let f = f32::from_bits(fp32_flush_to_zero(src[i]));
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
            // Int32 -> BFloat16 (VFLOOR_S32_BF16): 8 i32 -> 16 bf16 (pack into 4 words)
            (ElementType::Int32, ElementType::BFloat16) => {
                for i in 0..4 {
                    let f0 = src[i * 2] as i32 as f32;
                    let f1 = src[i * 2 + 1] as i32 as f32;
                    let bf0 = Self::f32_to_bf16(f0);
                    let bf1 = Self::f32_to_bf16(f1);
                    result[i] = (bf0 as u32) | ((bf1 as u32) << 16);
                }
            }
            // BFloat16 -> Int32: 16 bf16 -> 8 i32 (use lower half)
            // VFLOOR uses floor rounding (toward negative infinity).
            (ElementType::BFloat16, ElementType::Int32) => {
                use super::vector_float::fp32_flush_to_zero;
                for i in 0..8 {
                    let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
                    // AIE2 FTZ: bf16 denormals flush to signed zero before floor.
                    let f = f32::from_bits(fp32_flush_to_zero(Self::bf16_to_f32(bf16).to_bits()));
                    result[i] = f.floor() as i32 as u32;
                }
            }
            // Same type: pass through
            _ if from_type == to_type => {
                result = *src;
            }
            (from, to) => {
                panic!(
                    "vector_convert: unhandled conversion {:?} -> {:?} (encoding={:?})",
                    from, to, "vector_convert",
                );
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::interpreter::bundle::{ElementType, Operand, SlotIndex, SlotOp};
    use crate::interpreter::execute::vector_dispatch::VectorAlu;
    use crate::interpreter::state::ExecutionContext;
    use xdna_archspec::aie2::isa::SemanticOp;
    use crate::interpreter::decode::register_map::AccumWidth;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    #[test]
    fn test_vector_convert_bf16_to_f32() {
        let mut ctx = make_ctx();
        let bf16_1 = VectorAlu::f32_to_bf16(1.0);
        let bf16_2 = VectorAlu::f32_to_bf16(2.0);
        let bf16_3 = VectorAlu::f32_to_bf16(3.0);
        let bf16_4 = VectorAlu::f32_to_bf16(4.0);
        ctx.vector.write(
            0,
            [
                (bf16_1 as u32) | ((bf16_2 as u32) << 16),
                (bf16_3 as u32) | ((bf16_4 as u32) << 16),
                0, 0, 0, 0, 0, 0,
            ],
        );

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Convert)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::VectorReg(0));
        op.from_type = Some(ElementType::BFloat16);

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(1);

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
    fn test_vfloor_bf16_to_s32_shift0() {
        let result = VectorAlu::vector_floor_bf16_to_s32(
            &[
                ((VectorAlu::f32_to_bf16(1.5) as u32) | ((VectorAlu::f32_to_bf16(-2.5) as u32) << 16)),
                ((VectorAlu::f32_to_bf16(0.0) as u32) | ((VectorAlu::f32_to_bf16(100.0) as u32) << 16)),
                0, 0, 0, 0, 0, 0,
            ],
            0,
        );
        assert_eq!(result[0] as i32, 1, "floor(1.5) = 1");
        assert_eq!(result[1] as i32, -3, "floor(-2.5) = -3");
        assert_eq!(result[2] as i32, 0, "floor(0.0) = 0");
        assert_eq!(result[3] as i32, 100, "floor(100.0) = 100");
    }

    #[test]
    fn test_vfloor_signed_6bit_shift() {
        let bf16_10 = VectorAlu::f32_to_bf16(10.0);
        let src = [bf16_10 as u32, 0, 0, 0, 0, 0, 0, 0];

        assert_eq!(VectorAlu::vector_floor_bf16_to_s32(&src, 3)[0] as i32, 80);
        assert_eq!(VectorAlu::vector_floor_bf16_to_s32(&src, 0x23)[0] as i32, 0);
        assert_eq!(VectorAlu::vector_floor_bf16_to_s32(&src, -249)[0] as i32, 1280);
        assert_eq!(VectorAlu::vector_floor_bf16_to_s32(&src, 0x20)[0] as i32, 0);
    }

    #[test]
    fn test_vfloor_nan_saturates_to_int_max() {
        let src = [0xFFFF_FFFF, 0, 0, 0, 0, 0, 0, 0];
        let result = VectorAlu::vector_floor_bf16_to_s32(&src, 0);
        assert_eq!(result[0], i32::MAX as u32, "NaN -> INT_MAX");
        assert_eq!(result[1], i32::MAX as u32, "NaN -> INT_MAX");
    }

    #[test]
    fn test_vconv_bf16_fp32_acc_source() {
        let mut ctx = make_ctx();

        let mut acc = [0u64; 8];
        for i in 0..8 {
            let f_lo = ((i * 2 + 1) as f32).to_bits();
            let f_hi = ((i * 2 + 2) as f32).to_bits();
            acc[i] = (f_lo as u64) | ((f_hi as u64) << 32);
        }
        ctx.accumulator.write(0, acc);

        let mut op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Convert)
            .as_vector(ElementType::BFloat16)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::AccumReg(0));
        op.from_type = Some(ElementType::Float32);
        op.accum_width = Some(AccumWidth::Half);

        assert!(VectorAlu::execute(&op, &mut ctx));
        let result = ctx.vector.read(0);

        for i in 0..8 {
            let lo_bf16 = (result[i / 2] >> ((i % 2) * 16)) as u16;
            let expected = VectorAlu::f32_to_bf16((i + 1) as f32);
            assert_eq!(lo_bf16, expected,
                "element {}: expected bf16({}.0)=0x{:04X}, got 0x{:04X}",
                i, i + 1, expected, lo_bf16);
        }
    }

    #[test]
    fn test_vconv_fp32_bf16_acc_dest() {
        let mut ctx = make_ctx();

        let mut vec_data = [0u32; 8];
        for i in 0..8 {
            let bf_lo = VectorAlu::f32_to_bf16((i * 2 + 1) as f32);
            let bf_hi = VectorAlu::f32_to_bf16((i * 2 + 2) as f32);
            vec_data[i] = (bf_lo as u32) | ((bf_hi as u32) << 16);
        }
        ctx.vector.write(0, vec_data);

        let mut op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Convert)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::VectorReg(0));
        op.from_type = Some(ElementType::BFloat16);
        op.is_wide_vector = true;

        assert!(VectorAlu::execute(&op, &mut ctx));
        let acc = ctx.accumulator.read(0);

        for i in 0..8 {
            let lo = acc[i] as u32;
            let hi = (acc[i] >> 32) as u32;
            let f_lo = f32::from_bits(lo);
            let f_hi = f32::from_bits(hi);
            let expected_lo = (i * 2 + 1) as f32;
            let expected_hi = (i * 2 + 2) as f32;
            assert!((f_lo - expected_lo).abs() < 0.1,
                "lane {} lo: expected {}, got {}", i, expected_lo, f_lo);
            assert!((f_hi - expected_hi).abs() < 0.1,
                "lane {} hi: expected {}, got {}", i, expected_hi, f_hi);
        }
    }

    #[test]
    fn test_vector_convert_int32_to_f32() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Convert)
            .as_vector(ElementType::Float32)
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::VectorReg(0));
        op.from_type = Some(ElementType::Int32);

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read(1);

        for i in 0..8 {
            let f = f32::from_bits(result[i]);
            assert_eq!(f, (i + 1) as f32);
        }
    }
}
