//! AIE2 matrix multiply unit with proper tile geometry.
//!
//! The AIE2 vector unit contains a systolic-style multiplier array with 512
//! multiply units operating at 8-bit x 4-bit granularity. Input vectors are
//! permuted and reinterpreted as 2D matrix tiles. The tile dimensions depend
//! on element types:
//!
//! | A type | B type | Acc type | Rows | Inner | Cols | Output elements |
//! |--------|--------|----------|------|-------|------|-----------------|
//! | int8   | int8   | int32    | 4    | 8     | 8    | 32 (4x8)        |
//! | int16  | int16  | int32    | 4    | 2     | 8    | 32 (4x8)        |
//! | int16  | int16  | int64    | 4    | 4     | 4    | 16 (4x4)        |
//! | bf16   | bf16   | fp32     | 4    | 8     | 4    | 16 (4x4)        |
//! | int8   | int4   | int32    | 4    | 16    | 8    | 32 (4x8)        |
//! | int16  | int8   | int32    | 4    | 4     | 8    | 32 (4x8)        |
//!
//! The multiply is: acc[r][c] += sum(k=0..inner-1) { A[r][k] * B[k][c] }
//!
//! Hardware reference: mulmac.py (read for understanding, original implementation).

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::execute::vector_config::{AccWidth, MatMulConfig};
use crate::interpreter::state::{ExecutionContext, Vec512, Acc1024};
use crate::tablegen::SemanticOp;

// ---------------------------------------------------------------------------
// Entry point: execute_matmul
// ---------------------------------------------------------------------------

/// Execute a MAC-family instruction using config-driven matrix multiply.
///
/// This is the single entry point for ALL MAC-family instructions (Mac,
/// MatMul, MatMulSub, NegMul, NegMatMul, AddMac, SubMac). It reads the
/// config register, parses tile geometry, reads 512-bit vector inputs and
/// 1024-bit accumulator, performs the multiply, and writes back.
///
/// Returns `true` if this function handled the instruction, `false` if the
/// semantic is not a MAC-family operation (caller should use fallback).
pub fn execute_matmul(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    // Only handle MAC-family semantics.
    let semantic = match op.semantic {
        Some(
            s @ (SemanticOp::Mac
            | SemanticOp::MatMul
            | SemanticOp::MatMulSub
            | SemanticOp::NegMul
            | SemanticOp::NegMatMul
            | SemanticOp::AddMac
            | SemanticOp::SubMac),
        ) => s,
        _ => return false,
    };

    // Read config register (last ScalarReg in sources).
    let conf_val = match get_config_reg(op, ctx) {
        Some(v) => v,
        None => {
            log::error!(
                "[MATMUL] no config register found in sources for {:?}",
                op.encoding_name
            );
            return true; // Handled (as error), don't fall through.
        }
    };

    // Detect bf16 mode from encoding name.
    let is_bf16 = op
        .encoding_name
        .as_ref()
        .map(|n| n.contains("_F_") || n.ends_with("_F"))
        .unwrap_or(false);

    // Parse config word into tile geometry and modes.
    let mut config = match MatMulConfig::from_config_word(conf_val, is_bf16) {
        Some(c) => c,
        None => {
            log::error!(
                "[MATMUL] failed to parse config word 0x{:08x} (bf16={}) for {:?}",
                conf_val,
                is_bf16,
                op.encoding_name
            );
            return true;
        }
    };

    // Handle product negation for NegMul/NegMatMul/MatMulSub semantics.
    match semantic {
        SemanticOp::NegMul | SemanticOp::NegMatMul | SemanticOp::MatMulSub => {
            config.subtract = !config.subtract;
        }
        _ => {}
    }

    // Read the two 512-bit input vectors.
    let (a, b) = get_two_vec512(op, ctx);

    // Read the accumulator destination (1024-bit wide register pair).
    let acc_reg = get_acc_dest(op);
    let mut acc = ctx.accumulator.read_wide(acc_reg);

    // For AddMac/SubMac: merge a second accumulator source before the multiply.
    // AddMac: acc_dest = acc_dest + acc_src + A * B
    // SubMac: acc_dest = acc_dest - acc_src + A * B
    match semantic {
        SemanticOp::AddMac => {
            let src_reg = get_acc_source(op);
            let src_acc = ctx.accumulator.read_wide(src_reg);
            for i in 0..16 {
                acc[i] = acc[i].wrapping_add(src_acc[i]);
            }
        }
        SemanticOp::SubMac => {
            let src_reg = get_acc_source(op);
            let src_acc = ctx.accumulator.read_wide(src_reg);
            for i in 0..16 {
                acc[i] = acc[i].wrapping_sub(src_acc[i]);
            }
        }
        _ => {}
    }

    // Perform the config-driven matrix multiply.
    matmul_config_driven(&mut acc, &a, &b, &config);

    // Write result back to the accumulator.
    ctx.accumulator.write_wide(acc_reg, acc);

    true
}

// ---------------------------------------------------------------------------
// Private helpers for execute_matmul
// ---------------------------------------------------------------------------

/// Scan sources for the last ScalarReg operand (the config register).
fn get_config_reg(op: &SlotOp, ctx: &ExecutionContext) -> Option<u32> {
    for src in op.sources.iter().rev() {
        if let Operand::ScalarReg(r) = src {
            return Some(ctx.scalar.read(*r));
        }
    }
    None
}

/// Read two 512-bit vectors from the VectorReg operands.
///
/// MAC instructions have two VectorReg sources that each name an x-register
/// (already decoded as even indices: x0->0, x2->2, etc.). We read the full
/// 512-bit value via `read_wide`.
fn get_two_vec512(op: &SlotOp, ctx: &ExecutionContext) -> (Vec512, Vec512) {
    let mut vregs = op
        .sources
        .iter()
        .filter_map(|s| {
            if let Operand::VectorReg(r) = s {
                Some(*r)
            } else {
                None
            }
        });

    let a_reg = vregs.next().unwrap_or_else(|| {
        log::error!("[MATMUL] missing first VectorReg source");
        0
    });
    let b_reg = vregs.next().unwrap_or_else(|| {
        log::error!("[MATMUL] missing second VectorReg source");
        0
    });

    // Ensure even alignment for wide read.
    let a_base = a_reg & !1;
    let b_base = b_reg & !1;

    (ctx.vector.read_wide(a_base), ctx.vector.read_wide(b_base))
}

/// Extract the AccumReg from the destination operand.
fn get_acc_dest(op: &SlotOp) -> u8 {
    match &op.dest {
        Some(Operand::AccumReg(r)) => {
            // Ensure even alignment for wide access.
            *r & !1
        }
        other => {
            log::error!(
                "[MATMUL] expected AccumReg dest, got {:?} -- defaulting to cm0",
                other
            );
            0
        }
    }
}

/// Extract the first AccumReg from sources (for AddMac/SubMac).
fn get_acc_source(op: &SlotOp) -> u8 {
    for src in &op.sources {
        if let Operand::AccumReg(r) = src {
            return *r & !1;
        }
    }
    log::error!(
        "[MATMUL] no AccumReg found in sources -- defaulting to cm0"
    );
    0
}

/// Tile geometry for a matrix multiply mode.
#[derive(Debug, Clone, Copy)]
struct TileGeometry {
    rows: usize,
    inner: usize,
    cols: usize,
}

/// Extract int8 elements from packed [u32; 8] (256 bits = 32 bytes = 32 int8 values).
/// Elements are in little-endian byte order within each u32.
fn extract_i8(packed: &[u32; 8], index: usize) -> i8 {
    let word = index / 4;
    let byte = index % 4;
    ((packed[word] >> (byte * 8)) & 0xFF) as u8 as i8
}

/// Extract uint8 elements from packed [u32; 8].
fn extract_u8(packed: &[u32; 8], index: usize) -> u8 {
    let word = index / 4;
    let byte = index % 4;
    ((packed[word] >> (byte * 8)) & 0xFF) as u8
}

/// Extract int16 elements from packed [u32; 8] (256 bits = 16 int16 values).
fn extract_i16(packed: &[u32; 8], index: usize) -> i16 {
    let word = index / 2;
    let half = index % 2;
    ((packed[word] >> (half * 16)) & 0xFFFF) as u16 as i16
}

/// Extract uint16 elements from packed [u32; 8].
fn extract_u16(packed: &[u32; 8], index: usize) -> u16 {
    let word = index / 2;
    let half = index % 2;
    ((packed[word] >> (half * 16)) & 0xFFFF) as u16
}

/// Extract bf16 as f32 from packed [u32; 8] (256 bits = 16 bf16 values).
fn extract_bf16_as_f32(packed: &[u32; 8], index: usize) -> f32 {
    let word = index / 2;
    let half = index % 2;
    let bits = ((packed[word] >> (half * 16)) & 0xFFFF) as u16;
    f32::from_bits((bits as u32) << 16)
}

/// Extract int32 elements from packed [u32; 8] (256 bits = 8 int32 values).
fn extract_i32(packed: &[u32; 8], index: usize) -> i32 {
    packed[index] as i32
}

/// Read a 32-bit accumulator lane from the [u64; 8] accumulator.
///
/// The 8 u64 lanes hold 16 int32 values (acc_cmb=1 mode, 32-bit accumulator).
/// Lane layout: acc[0] holds output[0] in low 32 bits, output[1] in high 32 bits, etc.
fn read_acc32(acc: &[u64; 8], index: usize) -> i64 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    bits as i32 as i64
}

/// Write a 32-bit accumulator lane into the [u64; 8] accumulator.
fn write_acc32(acc: &mut [u64; 8], index: usize, value: i64) {
    let u64_lane = index / 2;
    let half = index % 2;
    let masked = (value as u32) as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (masked << shift);
}

/// Read a 64-bit accumulator lane (acc_cmb=2 mode).
fn read_acc64(acc: &[u64; 8], index: usize) -> i64 {
    acc[index] as i64
}

/// Write a 64-bit accumulator lane (acc_cmb=2 mode).
fn write_acc64(acc: &mut [u64; 8], index: usize, value: i64) {
    acc[index] = value as u64;
}

/// Read a float32 accumulator lane from [u64; 8].
///
/// For bf16 matmul, the accumulator holds fp32 values. Since we have 16 output
/// elements (4x4) and 8 u64 lanes, each u64 holds two fp32 values.
fn read_acc_f32(acc: &[u64; 8], index: usize) -> f32 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    f32::from_bits(bits)
}

/// Write a float32 accumulator lane.
fn write_acc_f32(acc: &mut [u64; 8], index: usize, value: f32) {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = value.to_bits() as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (bits << shift);
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
        ElementType::Int8 => matmul_i8xi8(acc, a, b, true, true, false),
        ElementType::UInt8 => matmul_i8xi8(acc, a, b, signed_a, signed_b, false),
        ElementType::Int16 => matmul_i16xi16_32(acc, a, b, true, true, false),
        ElementType::UInt16 => matmul_i16xi16_32(acc, a, b, signed_a, signed_b, false),
        ElementType::BFloat16 => matmul_bf16xbf16(acc, a, b, false),
        ElementType::Int32 => matmul_i32xi16(acc, a, b, true, true, false),
        ElementType::UInt32 => matmul_i32xi16(acc, a, b, false, false, false),
        ElementType::Int64 | ElementType::UInt64 => matmul_i32xi16(acc, a, b, signed_a, signed_b, false),
        ElementType::Float32 => matmul_bf16xbf16(acc, a, b, false),
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
        ElementType::Int8 => matmul_i8xi8(acc, a, b, true, true, true),
        ElementType::UInt8 => matmul_i8xi8(acc, a, b, signed_a, signed_b, true),
        ElementType::Int16 => matmul_i16xi16_32(acc, a, b, true, true, true),
        ElementType::UInt16 => matmul_i16xi16_32(acc, a, b, signed_a, signed_b, true),
        ElementType::BFloat16 => matmul_bf16xbf16(acc, a, b, true),
        ElementType::Int32 => matmul_i32xi16(acc, a, b, true, true, true),
        ElementType::UInt32 => matmul_i32xi16(acc, a, b, false, false, true),
        ElementType::Int64 | ElementType::UInt64 => matmul_i32xi16(acc, a, b, signed_a, signed_b, true),
        ElementType::Float32 => matmul_bf16xbf16(acc, a, b, true),
    }
}

// ---------------------------------------------------------------------------
// Config-driven full-width matmul (512-bit inputs, 1024-bit accumulator)
// ---------------------------------------------------------------------------

/// Extract an element from a 512-bit vector (`Vec512` = `[u32; 16]`).
///
/// `byte_idx` is the byte offset within the full 64-byte vector.
fn extract_element_512(src: &Vec512, byte_idx: usize, bits: u32, signed: bool) -> i64 {
    match bits {
        4 => {
            // 4-bit elements: two per byte
            let byte_pos = byte_idx / 2;
            let nibble = byte_idx % 2;
            let word = byte_pos / 4;
            let byte_in_word = byte_pos % 4;
            let raw_byte = ((src[word] >> (byte_in_word * 8)) & 0xFF) as u8;
            let val = if nibble == 0 { raw_byte & 0xF } else { (raw_byte >> 4) & 0xF };
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
            let val = ((src[word] >> (byte_in_word * 8)) & 0xFF) as u8;
            if signed { val as i8 as i64 } else { val as i64 }
        }
        16 => {
            let elem_idx = byte_idx / 2;
            let word = elem_idx / 2;
            let half_in_word = elem_idx % 2;
            let val = ((src[word] >> (half_in_word * 16)) & 0xFFFF) as u16;
            if signed { val as i16 as i64 } else { val as i64 }
        }
        32 => {
            let word = byte_idx / 4;
            let val = src[word];
            if signed { val as i32 as i64 } else { val as i64 }
        }
        _ => 0,
    }
}

/// Read from a 1024-bit accumulator (`Acc1024` = `[u64; 16]`).
fn read_acc_wide(acc: &Acc1024, index: usize, acc_width: AccWidth) -> i64 {
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
fn write_acc_wide(acc: &mut Acc1024, index: usize, value: i64, acc_width: AccWidth) {
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

/// Write an fp32 value to a wide accumulator.
fn write_acc_wide_f32(acc: &mut Acc1024, index: usize, value: f32) {
    // bf16 mode always uses acc_cmb=1 (32-bit lanes)
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = value.to_bits() as u64;
    let shift = half * 32;
    acc[u64_lane] = (acc[u64_lane] & !(0xFFFF_FFFF_u64 << shift)) | (bits << shift);
}

/// Read an fp32 value from a wide accumulator.
fn read_acc_wide_f32(acc: &Acc1024, index: usize) -> f32 {
    let u64_lane = index / 2;
    let half = index % 2;
    let bits = ((acc[u64_lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
    f32::from_bits(bits)
}

/// Config-driven matrix multiply on full-width (512-bit) inputs.
///
/// Reads element data from the 512-bit A and B vectors, performs a tiled
/// matrix multiply according to the geometry in `config`, and writes the
/// result to the 1024-bit accumulator.
///
/// The input vectors are interpreted as flat arrays of elements in
/// row-major order:
/// - A[r][k] at byte offset `(r * inner + k) * (bits_x / 8)`
/// - B[k][c] at byte offset `(k * cols + c) * (bits_y / 8)`
///
/// Output goes to accumulator lane `r * cols + c`.
pub fn matmul_config_driven(
    acc: &mut Acc1024,
    a: &Vec512,
    b: &Vec512,
    config: &MatMulConfig,
) {
    let rows = config.rows as usize;
    let inner = config.inner as usize;
    let cols = config.cols as usize;
    let bits_x = config.a_type.bits() as u32;
    let bits_y = config.b_type.bits() as u32;
    let bytes_x = (bits_x / 8) as usize;
    let bytes_y = if bits_y == 4 { 1 } else { (bits_y / 8) as usize }; // 4-bit: 2 elements per byte

    // Zero accumulator if requested (zero_acc = !accumulate in MatMulConfig)
    if !config.accumulate {
        *acc = [0u64; 16];
    }

    if config.bfloat {
        // BFloat16 path: extract bf16 as f32, accumulate in fp32
        for r in 0..rows {
            for c in 0..cols {
                let out_idx = r * cols + c;
                let mut sum: f32 = 0.0;

                for k in 0..inner {
                    let a_byte = (r * inner + k) * bytes_x;
                    let b_byte = (k * cols + c) * bytes_y;

                    // Extract bf16 elements: byte offset / 2 gives the bf16 index
                    let a_elem_idx = a_byte / 2;
                    let b_elem_idx = b_byte / 2;
                    // Vec512 has 16 u32 words = 32 bf16 elements; index directly
                    let a_word = a_elem_idx / 2;
                    let a_half = a_elem_idx % 2;
                    let b_word = b_elem_idx / 2;
                    let b_half = b_elem_idx % 2;
                    let a_bits = ((a[a_word] >> (a_half * 16)) & 0xFFFF) as u16;
                    let b_bits = ((b[b_word] >> (b_half * 16)) & 0xFFFF) as u16;
                    let a_val = f32::from_bits((a_bits as u32) << 16);
                    let b_val = f32::from_bits((b_bits as u32) << 16);

                    sum += a_val * b_val;
                }

                let prev = read_acc_wide_f32(acc, out_idx);
                if config.subtract {
                    write_acc_wide_f32(acc, out_idx, prev - sum);
                } else {
                    write_acc_wide_f32(acc, out_idx, prev + sum);
                }
            }
        }
        return;
    }

    // Integer path
    for r in 0..rows {
        for c in 0..cols {
            let out_idx = r * cols + c;
            let mut sum: i64 = 0;

            for k in 0..inner {
                let a_byte = if bits_x == 4 {
                    (r * inner + k) / 2
                } else {
                    (r * inner + k) * bytes_x
                };
                let b_byte = if bits_y == 4 {
                    (k * cols + c) / 2
                } else {
                    (k * cols + c) * bytes_y
                };

                let a_val = extract_element_512(a, a_byte, bits_x, config.x_signed);
                let b_val = extract_element_512(b, b_byte, bits_y, config.y_signed);

                sum += a_val * b_val;
            }

            let prev = read_acc_wide(acc, out_idx, config.acc_width);
            if config.subtract {
                write_acc_wide(acc, out_idx, prev - sum, config.acc_width);
            } else {
                write_acc_wide(acc, out_idx, prev + sum, config.acc_width);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// int8 x int8 -> int32 accumulator
//
// Geometry: rows=4, inner=8, cols=8 => 32 output elements
//
// A is 4 rows x 8 cols of int8 = 32 bytes = 256 bits (one vector register)
// B is 8 rows x 8 cols of int8 = 64 bytes = 512 bits (two vector registers,
//   but the hardware only uses 256 bits from the second source, selecting
//   via permutation. For the basic mode we use the full 256-bit B vector
//   reshaped as 8x4.)
//
// Actually for the basic int8xi8 mode (mmode=1, perm_mode row 1):
//   rows=4, inner=8, cols=8
// But we only have 256 bits of B = 32 int8 values. With 8 rows and 8 cols
// that would be 64 values, which exceeds our vector width.
//
// Looking more carefully at the hardware: the 256-bit vector holds 32 int8
// elements. For 4x8 output, the inner dimension must be such that
// 4 * inner * sizeof(int8) <= 256 bits AND inner * 8 * sizeof(int8) <= 256 bits.
// So inner * 8 <= 32, meaning inner <= 4.
//
// The correct basic mode is actually rows=4, inner=4, cols=8 for 16xi8 input,
// but for 8x8 input (mmode=1): rows=4, inner=8, cols=8.
// In that case B needs 8*8 = 64 bytes which exceeds 256 bits.
//
// The resolution: the hardware permute unit rearranges the input data.
// With 512-bit permute width, B can actually be drawn from the full permute
// space. For the simpler emulation, we implement the most common sub-case:
// rows=4, inner=8, cols=4 which fits in 256 bits for both A (4*8=32 bytes)
// and B (8*4=32 bytes).
//
// For the full 4x8x8 mode, the hardware uses both X and Y permute inputs
// which may come from different vector register halves or two registers.
// We implement the 4x8x4 variant first as it matches one 256-bit register
// per operand.
// ---------------------------------------------------------------------------

/// int8 x int8 matrix multiply with 32-bit accumulator.
///
/// Geometry: A[4][8] * B[8][4] = C[4][4], 16 output int32 values.
/// A: 32 int8 values from 256-bit vector, row-major 4 rows x 8 cols.
/// B: 32 int8 values from 256-bit vector, row-major 8 rows x 4 cols.
/// Output: 16 int32 values in 8 u64 lanes (two int32 per u64).
fn matmul_i8xi8(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
    // A is 4 rows x 8 cols of int8 = 32 elements (256 bits)
    // B is 8 rows x 4 cols of int8 = 32 elements (256 bits)
    // Output is 4 rows x 4 cols of int32 = 16 elements in acc
    let geom = TileGeometry { rows: 4, inner: 8, cols: 4 };

    for r in 0..geom.rows {
        for c in 0..geom.cols {
            let out_idx = r * geom.cols + c;
            let mut sum: i64 = 0;

            for k in 0..geom.inner {
                let a_idx = r * geom.inner + k;
                let b_idx = k * geom.cols + c;

                let a_val = if signed_a {
                    extract_i8(a, a_idx) as i64
                } else {
                    extract_u8(a, a_idx) as i64
                };

                let b_val = if signed_b {
                    extract_i8(b, b_idx) as i64
                } else {
                    extract_u8(b, b_idx) as i64
                };

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

/// int16 x int16 matrix multiply with 32-bit accumulator (acc_cmb=1).
///
/// Geometry: A[4][2] * B[2][8] = C[4][8], 32 output int32 values.
/// A: 16 int16 values from 256-bit vector. We use the first 8 (4 rows x 2 inner).
/// B: 16 int16 values from 256-bit vector. We use all 16 (2 rows x 8 cols).
/// Output: 32 int32 values packed into the accumulator.
///
/// Note: with acc_cmb=1 and 32-bit accumulator, we get 32 output lanes.
/// With 8 u64 lanes that's 16 int32 values directly addressable, so we use
/// the common sub-case: A[4][2] * B[2][4] = C[4][4] = 16 outputs.
fn matmul_i16xi16_32(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
    // A: 4 rows x 2 inner = 8 int16 elements (128 bits, first half of vector)
    // B: 2 rows x 4 cols = 8 int16 elements (128 bits, first half of vector)
    // Output: 4 rows x 4 cols = 16 int32 values
    let geom = TileGeometry { rows: 4, inner: 2, cols: 4 };

    for r in 0..geom.rows {
        for c in 0..geom.cols {
            let out_idx = r * geom.cols + c;
            let mut sum: i64 = 0;

            for k in 0..geom.inner {
                let a_idx = r * geom.inner + k;
                let b_idx = k * geom.cols + c;

                let a_val = if signed_a {
                    extract_i16(a, a_idx) as i64
                } else {
                    extract_u16(a, a_idx) as i64
                };

                let b_val = if signed_b {
                    extract_i16(b, b_idx) as i64
                } else {
                    extract_u16(b, b_idx) as i64
                };

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

/// bf16 x bf16 matrix multiply with fp32 accumulator.
///
/// Geometry: A[4][8] * B[8][4] = C[4][4], 16 output fp32 values.
/// A: 16 bf16 values from 256-bit vector, reinterpreted as 4 rows x 4 cols
///    (limited by 256-bit width: 16 bf16 = 4x4).
/// B: 16 bf16 values, reinterpreted as 4 rows x 4 cols.
///
/// The hardware actually uses rows=4, inner=8, cols=4 from constants.py,
/// but with 256-bit inputs we only have 16 bf16 values per vector,
/// so the practical single-register mode is 4x4x4.
fn matmul_bf16xbf16(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    subtract: bool,
) {
    // A: 4 rows x 4 inner = 16 bf16 elements (256 bits)
    // B: 4 rows x 4 cols = 16 bf16 elements (256 bits)
    // Output: 4 rows x 4 cols = 16 fp32 values
    let geom = TileGeometry { rows: 4, inner: 4, cols: 4 };

    for r in 0..geom.rows {
        for c in 0..geom.cols {
            let out_idx = r * geom.cols + c;
            let mut sum: f32 = 0.0;

            for k in 0..geom.inner {
                let a_idx = r * geom.inner + k;
                let b_idx = k * geom.cols + c;

                let a_val = extract_bf16_as_f32(a, a_idx);
                let b_val = extract_bf16_as_f32(b, b_idx);
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

/// int32 x int16 matrix multiply with 64-bit accumulator (acc_cmb=2).
///
/// Geometry: A[4][2] * B[2][4] = C[4][4], producing 16 int64 outputs.
/// But we only have 8 u64 lanes, so the practical mode is:
/// A[2][2] * B[2][4] = C[2][4] = 8 int64 outputs.
///
/// Actually from constants.py: mmode=7 is 32x16 acc_cmb=2, perm_modes
/// include rows=4, inner=2, cols=4.
fn matmul_i32xi16(
    acc: &mut [u64; 8],
    a: &[u32; 8],
    b: &[u32; 8],
    signed_a: bool,
    signed_b: bool,
    subtract: bool,
) {
    // A: 4 rows x 2 inner of int32 = 8 elements (256 bits)
    // B: 2 rows x 4 cols of int16 = 8 elements (128 bits)
    // Output: 4 rows x 2 cols = 8 int64 values
    //
    // With acc_cmb=2, each output is 64 bits, fitting in one u64 lane.
    // 4*2 = 8 outputs = 8 u64 lanes.
    let geom = TileGeometry { rows: 4, inner: 2, cols: 2 };

    for r in 0..geom.rows {
        for c in 0..geom.cols {
            let out_idx = r * geom.cols + c;
            let mut sum: i64 = 0;

            for k in 0..geom.inner {
                let a_idx = r * geom.inner + k;
                let b_idx = k * geom.cols + c;

                let a_val = if signed_a {
                    extract_i32(a, a_idx) as i64
                } else {
                    a[a_idx] as i64
                };

                let b_val = if signed_b {
                    extract_i16(b, b_idx) as i64
                } else {
                    extract_u16(b, b_idx) as i64
                };

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack int8 values into [u32; 8] in little-endian order.
    fn pack_i8(values: &[i8]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate() {
            let word = i / 4;
            let byte = i % 4;
            packed[word] |= ((v as u8) as u32) << (byte * 8);
        }
        packed
    }

    /// Pack int16 values into [u32; 8] in little-endian order.
    fn pack_i16(values: &[i16]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate() {
            let word = i / 2;
            let half = i % 2;
            packed[word] |= ((v as u16) as u32) << (half * 16);
        }
        packed
    }

    /// Pack bf16 values (as u16 bit patterns) into [u32; 8].
    fn pack_bf16(values: &[u16]) -> [u32; 8] {
        let mut packed = [0u32; 8];
        for (i, &v) in values.iter().enumerate() {
            let word = i / 2;
            let half = i % 2;
            packed[word] |= (v as u32) << (half * 16);
        }
        packed
    }

    /// Convert f32 to bf16 bit pattern (truncate lower 16 bits).
    fn f32_to_bf16_bits(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    // -----------------------------------------------------------------------
    // int8 x int8 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i8xi8_identity_like() {
        // A = 4x8 identity-like (first 4 cols are identity, rest zero)
        // B = 8x4 with known values in first 4 rows
        //
        // A[r][k] = if k == r { 1 } else { 0 } for k < 4, rest 0
        // B[k][c] = (k * 4 + c + 1) for k < 4, rest 0
        //
        // Result should be: C[r][c] = B[r][c] = r * 4 + c + 1

        let mut a_vals = [0i8; 32];
        // Row 0: a_vals[0] = 1 (k=0)
        a_vals[0] = 1;
        // Row 1: a_vals[8+1] = 1 (k=1)
        a_vals[8 + 1] = 1;
        // Row 2: a_vals[16+2] = 1 (k=2)
        a_vals[16 + 2] = 1;
        // Row 3: a_vals[24+3] = 1 (k=3)
        a_vals[24 + 3] = 1;

        let mut b_vals = [0i8; 32];
        for k in 0..4 {
            for c in 0..4 {
                b_vals[k * 4 + c] = (k * 4 + c + 1) as i8;
            }
        }

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);

        // Check C[r][c] = B[r][c] for the identity rows
        for r in 0..4 {
            for c in 0..4 {
                let out_idx = r * 4 + c;
                let expected = (r * 4 + c + 1) as i64;
                let actual = read_acc32(&acc, out_idx);
                assert_eq!(
                    actual, expected,
                    "C[{}][{}]: expected {}, got {}",
                    r, c, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_i8xi8_all_ones() {
        // A = 4x8, all 1s
        // B = 8x4, all 1s
        // C[r][c] = sum of 8 ones = 8

        let a_vals = [1i8; 32];
        let b_vals = [1i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);

        for r in 0..4 {
            for c in 0..4 {
                let out_idx = r * 4 + c;
                assert_eq!(
                    read_acc32(&acc, out_idx),
                    8,
                    "C[{}][{}] should be 8",
                    r, c
                );
            }
        }
    }

    #[test]
    fn test_i8xi8_accumulate() {
        // Verify accumulation: run matmul twice, values should double
        let a_vals = [1i8; 32];
        let b_vals = [1i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        matmul_i8xi8(&mut acc, &a, &b, true, true, false);

        for r in 0..4 {
            for c in 0..4 {
                let out_idx = r * 4 + c;
                assert_eq!(
                    read_acc32(&acc, out_idx),
                    16,
                    "C[{}][{}] should be 16 after two accumulations",
                    r, c
                );
            }
        }
    }

    #[test]
    fn test_i8xi8_subtract() {
        // First accumulate, then subtract the same product
        let a_vals = [2i8; 32];
        let b_vals = [3i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        // acc += A * B => each output = 8 * (2 * 3) = 48
        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        for idx in 0..16 {
            assert_eq!(read_acc32(&acc, idx), 48);
        }

        // acc -= A * B => each output = 48 - 48 = 0
        matmul_i8xi8(&mut acc, &a, &b, true, true, true);
        for idx in 0..16 {
            assert_eq!(read_acc32(&acc, idx), 0);
        }
    }

    #[test]
    fn test_i8xi8_signed_negative() {
        // Test with negative values
        let mut a_vals = [0i8; 32];
        let mut b_vals = [0i8; 32];

        // A[0][0] = -1, B[0][0] = -2
        // C[0][0] = (-1) * (-2) = 2
        a_vals[0] = -1;
        b_vals[0] = -2;

        // A[0][1] = 3, B[1][0] = -4
        // C[0][0] += 3 * (-4) = -12
        // Total C[0][0] = 2 + (-12) = -10
        a_vals[1] = 3;
        b_vals[4] = -4;

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        assert_eq!(read_acc32(&acc, 0), -10);
    }

    #[test]
    fn test_i8xi8_unsigned() {
        // Unsigned: 200 * 200 = 40000 (would be negative if signed)
        let mut a_vals = [0i8; 32];
        let mut b_vals = [0i8; 32];

        // 200 as u8 = 0xC8, which as i8 = -56
        a_vals[0] = -56; // 200 as u8
        b_vals[0] = -56; // 200 as u8

        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        // Signed: (-56) * (-56) = 3136
        matmul_i8xi8(&mut acc, &a, &b, true, true, false);
        assert_eq!(read_acc32(&acc, 0), 3136);

        // Unsigned: 200 * 200 = 40000
        let mut acc2 = [0u64; 8];
        matmul_i8xi8(&mut acc2, &a, &b, false, false, false);
        assert_eq!(read_acc32(&acc2, 0), 40000);
    }

    // -----------------------------------------------------------------------
    // int16 x int16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i16xi16_identity() {
        // A = 4x2, identity-like: A[r][k] = delta(r%2, k)
        // B = 2x4 with sequential values
        // For rows 0,1: result should pick from B rows 0,1
        // For rows 2,3: same pattern wraps

        let mut a_vals = [0i16; 16];
        // Row 0: A[0][0] = 1 (k=0)
        a_vals[0] = 1;
        // Row 1: A[1][1] = 1 (k=1)
        a_vals[3] = 1;

        let mut b_vals = [0i16; 16];
        // B[0][0..4] = {10, 20, 30, 40}
        b_vals[0] = 10;
        b_vals[1] = 20;
        b_vals[2] = 30;
        b_vals[3] = 40;
        // B[1][0..4] = {50, 60, 70, 80}
        b_vals[4] = 50;
        b_vals[5] = 60;
        b_vals[6] = 70;
        b_vals[7] = 80;

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i16xi16_32(&mut acc, &a, &b, true, true, false);

        // C[0][c] = A[0][0]*B[0][c] + A[0][1]*B[1][c] = 1*B[0][c] + 0 = B[0][c]
        assert_eq!(read_acc32(&acc, 0), 10);
        assert_eq!(read_acc32(&acc, 1), 20);
        assert_eq!(read_acc32(&acc, 2), 30);
        assert_eq!(read_acc32(&acc, 3), 40);

        // C[1][c] = A[1][0]*B[0][c] + A[1][1]*B[1][c] = 0 + 1*B[1][c] = B[1][c]
        assert_eq!(read_acc32(&acc, 4), 50);
        assert_eq!(read_acc32(&acc, 5), 60);
        assert_eq!(read_acc32(&acc, 6), 70);
        assert_eq!(read_acc32(&acc, 7), 80);
    }

    #[test]
    fn test_i16xi16_multiply() {
        // A = [[1, 2], [3, 4], [5, 6], [7, 8]]  (4x2)
        // B = [[1, 0, 0, 0], [0, 1, 0, 0]]       (2x4)
        //
        // C = A * B:
        // C[0] = [1*1+2*0, 1*0+2*1, 1*0+2*0, 1*0+2*0] = [1, 2, 0, 0]
        // C[1] = [3, 4, 0, 0]
        // C[2] = [5, 6, 0, 0]
        // C[3] = [7, 8, 0, 0]

        let a_vals: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0];
        let b_vals: [i16; 16] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mut acc = [0u64; 8];

        matmul_i16xi16_32(&mut acc, &a, &b, true, true, false);

        assert_eq!(read_acc32(&acc, 0), 1);  // C[0][0]
        assert_eq!(read_acc32(&acc, 1), 2);  // C[0][1]
        assert_eq!(read_acc32(&acc, 4), 3);  // C[1][0]
        assert_eq!(read_acc32(&acc, 5), 4);  // C[1][1]
        assert_eq!(read_acc32(&acc, 8), 5);  // C[2][0]
        assert_eq!(read_acc32(&acc, 9), 6);  // C[2][1]
        assert_eq!(read_acc32(&acc, 12), 7); // C[3][0]
        assert_eq!(read_acc32(&acc, 13), 8); // C[3][1]
    }

    #[test]
    fn test_i16xi16_accumulate_subtract() {
        let a_vals: [i16; 16] = [10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b_vals: [i16; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);
        let mut acc = [0u64; 8];

        // acc += A * B => C[0][0] = 10
        matmul_i16xi16_32(&mut acc, &a, &b, true, true, false);
        assert_eq!(read_acc32(&acc, 0), 10);

        // acc -= A * B => C[0][0] = 10 - 10 = 0
        matmul_i16xi16_32(&mut acc, &a, &b, true, true, true);
        assert_eq!(read_acc32(&acc, 0), 0);
    }

    // -----------------------------------------------------------------------
    // bf16 x bf16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_identity() {
        // A = 4x4 identity matrix as bf16
        // B = 4x4 with known values
        // C = B (identity property)

        let mut a_bits = [0u16; 16];
        let one = f32_to_bf16_bits(1.0);
        // Diagonal: A[0][0], A[1][1], A[2][2], A[3][3]
        a_bits[0 * 4 + 0] = one;
        a_bits[1 * 4 + 1] = one;
        a_bits[2 * 4 + 2] = one;
        a_bits[3 * 4 + 3] = one;

        let mut b_bits = [0u16; 16];
        for i in 0..16 {
            b_bits[i] = f32_to_bf16_bits((i + 1) as f32);
        }

        let a = pack_bf16(&a_bits);
        let b = pack_bf16(&b_bits);
        let mut acc = [0u64; 8];

        matmul_bf16xbf16(&mut acc, &a, &b, false);

        for r in 0..4 {
            for c in 0..4 {
                let idx = r * 4 + c;
                let expected = (r * 4 + c + 1) as f32;
                let actual = read_acc_f32(&acc, idx);
                assert!(
                    (actual - expected).abs() < 0.01,
                    "C[{}][{}]: expected {}, got {}",
                    r, c, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_bf16_all_ones() {
        // A = 4x4 all ones, B = 4x4 all ones
        // C[r][c] = sum of 4 ones = 4.0
        let one = f32_to_bf16_bits(1.0);
        let a_bits = [one; 16];
        let b_bits = [one; 16];

        let a = pack_bf16(&a_bits);
        let b = pack_bf16(&b_bits);
        let mut acc = [0u64; 8];

        matmul_bf16xbf16(&mut acc, &a, &b, false);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!(
                (actual - 4.0).abs() < 0.01,
                "Output[{}]: expected 4.0, got {}",
                idx, actual
            );
        }
    }

    #[test]
    fn test_bf16_accumulate() {
        let one = f32_to_bf16_bits(1.0);
        let a_bits = [one; 16];
        let b_bits = [one; 16];

        let a = pack_bf16(&a_bits);
        let b = pack_bf16(&b_bits);
        let mut acc = [0u64; 8];

        // First: acc = 4.0
        matmul_bf16xbf16(&mut acc, &a, &b, false);
        // Second: acc = 8.0
        matmul_bf16xbf16(&mut acc, &a, &b, false);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!(
                (actual - 8.0).abs() < 0.01,
                "Output[{}]: expected 8.0, got {}",
                idx, actual
            );
        }
    }

    #[test]
    fn test_bf16_subtract() {
        let two = f32_to_bf16_bits(2.0);
        let three = f32_to_bf16_bits(3.0);
        let a_bits = [two; 16];
        let b_bits = [three; 16];

        let a = pack_bf16(&a_bits);
        let b = pack_bf16(&b_bits);
        let mut acc = [0u64; 8];

        // acc += 2*3*4 = 24 per lane
        matmul_bf16xbf16(&mut acc, &a, &b, false);
        // acc -= 2*3*4 = 24 per lane => 0
        matmul_bf16xbf16(&mut acc, &a, &b, true);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!(
                actual.abs() < 0.01,
                "Output[{}]: expected 0.0, got {}",
                idx, actual
            );
        }
    }

    // -----------------------------------------------------------------------
    // int32 x int16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i32xi16_basic() {
        // A = 4x2 of int32 = 8 values (256 bits)
        // B = 2x2 of int16 = 4 values (64 bits)
        // C = 4x2 of int64 = 8 values

        let mut a_packed = [0u32; 8];
        // A[0][0] = 100, A[0][1] = 200
        a_packed[0] = 100;
        a_packed[1] = 200;
        // A[1][0] = 300, A[1][1] = 400
        a_packed[2] = 300;
        a_packed[3] = 400;

        // B = 2x2 identity matrix (row-major): B[0][0]=1, B[0][1]=0, B[1][0]=0, B[1][1]=1
        let b_vals: [i16; 16] = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b = pack_i16(&b_vals);

        let mut acc = [0u64; 8];
        matmul_i32xi16(&mut acc, &a_packed, &b, true, true, false);

        // C[0][0] = 100*1 + 200*0 = 100
        // C[0][1] = 100*0 + 200*1 = 200
        assert_eq!(read_acc64(&acc, 0), 100);
        assert_eq!(read_acc64(&acc, 1), 200);

        // C[1][0] = 300*1 + 400*0 = 300
        // C[1][1] = 300*0 + 400*1 = 400
        assert_eq!(read_acc64(&acc, 2), 300);
        assert_eq!(read_acc64(&acc, 3), 400);
    }

    // -----------------------------------------------------------------------
    // Element extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_i8() {
        let packed = pack_i8(&[1, -2, 3, -4, 5, -6, 7, -8,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(extract_i8(&packed, 0), 1);
        assert_eq!(extract_i8(&packed, 1), -2);
        assert_eq!(extract_i8(&packed, 2), 3);
        assert_eq!(extract_i8(&packed, 3), -4);
        assert_eq!(extract_i8(&packed, 4), 5);
        assert_eq!(extract_i8(&packed, 5), -6);
    }

    #[test]
    fn test_extract_i16() {
        let packed = pack_i16(&[100, -200, 300, -400, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0]);
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

    // -----------------------------------------------------------------------
    // Public API tests (matmul_dense / matmul_sub)
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_dense_i8() {
        let a_vals = [1i8; 32];
        let b_vals = [1i8; 32];
        let a = pack_i8(&a_vals);
        let b = pack_i8(&b_vals);
        let mut acc = [0u64; 8];

        matmul_dense(&mut acc, &a, &b, ElementType::Int8, true, true);

        // Each output = inner (8) dot products of 1*1 = 8
        for idx in 0..16 {
            assert_eq!(read_acc32(&acc, idx), 8);
        }
    }

    #[test]
    fn test_matmul_sub_i16() {
        let a_vals: [i16; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b_vals: [i16; 16] = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let a = pack_i16(&a_vals);
        let b = pack_i16(&b_vals);

        // Pre-load accumulator with 100 in lane 0
        let mut acc = [0u64; 8];
        write_acc32(&mut acc, 0, 100);

        matmul_sub(&mut acc, &a, &b, ElementType::Int16, true, true);

        // C[0][0] = 100 - (1*5 + 0*0) = 95
        assert_eq!(read_acc32(&acc, 0), 95);
    }

    #[test]
    fn test_matmul_dense_bf16() {
        let two = f32_to_bf16_bits(2.0);
        let three = f32_to_bf16_bits(3.0);

        // A = all 2.0, B = all 3.0
        // Each output = 4 * (2.0 * 3.0) = 24.0
        let a = pack_bf16(&[two; 16]);
        let b = pack_bf16(&[three; 16]);
        let mut acc = [0u64; 8];

        matmul_dense(&mut acc, &a, &b, ElementType::BFloat16, true, true);

        for idx in 0..16 {
            let actual = read_acc_f32(&acc, idx);
            assert!(
                (actual - 24.0).abs() < 0.1,
                "Output[{}]: expected 24.0, got {}",
                idx, actual
            );
        }
    }

    // -----------------------------------------------------------------------
    // execute_matmul entry point tests
    // -----------------------------------------------------------------------

    use crate::interpreter::bundle::SlotIndex;
    use crate::interpreter::state::ExecutionContext;

    /// Build a minimal SlotOp for a MAC-family instruction.
    ///
    /// Sources: [VectorReg(a_reg), VectorReg(b_reg), ScalarReg(conf_reg)]
    /// Dest: AccumReg(acc_reg)
    fn make_mac_op(
        semantic: SemanticOp,
        a_vreg: u8,
        b_vreg: u8,
        conf_sreg: u8,
        acc_dest: u8,
        encoding_name: Option<&str>,
    ) -> SlotOp {
        let mut op = SlotOp::from_semantic(SlotIndex::Accumulator, semantic);
        op.is_vector = true;
        op.is_wide_vector = true;
        op.sources.push(Operand::VectorReg(a_vreg));
        op.sources.push(Operand::VectorReg(b_vreg));
        op.sources.push(Operand::ScalarReg(conf_sreg));
        op.dest = Some(Operand::AccumReg(acc_dest));
        op.encoding_name = encoding_name.map(|s| s.to_string());
        op
    }

    /// Build a MAC op with an extra AccumReg source (for AddMac/SubMac).
    fn make_double_acc_op(
        semantic: SemanticOp,
        a_vreg: u8,
        b_vreg: u8,
        conf_sreg: u8,
        acc_dest: u8,
        acc_src: u8,
        encoding_name: Option<&str>,
    ) -> SlotOp {
        let mut op = make_mac_op(semantic, a_vreg, b_vreg, conf_sreg, acc_dest, encoding_name);
        // Insert AccumReg source before the ScalarReg (config is last).
        op.sources.insert(2, Operand::AccumReg(acc_src));
        op
    }

    /// Pack [u32; 16] (Vec512) of all-ones bytes (int8 = 1 in every byte).
    fn vec512_all_ones_i8() -> Vec512 {
        // 0x01010101 repeated 16 times = 64 bytes of 0x01
        [0x01010101u32; 16]
    }

    /// Pack Vec512 of all bf16(1.0) values.
    fn vec512_all_ones_bf16() -> Vec512 {
        // bf16(1.0) = 0x3F80; two per u32 = 0x3F80_3F80
        let word = 0x3F80_3F80u32;
        [word; 16]
    }

    /// Build a config word for int8xi8 mode.
    ///
    /// Config word layout:
    ///   bit 0:    zero_acc
    ///   bits 1-2: amode (0 = acc_cmb=1, 32-bit)
    ///   bits 3-4: bmode (1 = int8 B)
    ///   bits 5-7: variant (0)
    ///   bit 8:    sgn_y (1 = signed)
    ///   bit 9:    sgn_x (1 = signed)
    ///
    /// i8xi8 signed accumulate: amode=0, bmode=1, sgn_x=1, sgn_y=1
    /// = (1<<3) | (1<<8) | (1<<9) = 0x308
    fn config_i8xi8_accumulate() -> u32 {
        (1 << 3) | (1 << 8) | (1 << 9) // amode=0, bmode=1, signed
    }

    /// Build a config word for bf16 mode.
    /// bf16 uses the bf16 lookup path (variant=0), sign bits irrelevant.
    fn config_bf16_accumulate() -> u32 {
        0x00 // zero_acc=0, variant=0
    }

    /// Build a config word for int8xi8 with zero_acc=1 (clear before multiply).
    fn config_i8xi8_zero_acc() -> u32 {
        (1 << 3) | (1 << 8) | (1 << 9) | 1 // same as accumulate but bit 0 set
    }

    #[test]
    fn test_execute_matmul_i8xi8_ones() {
        // All-ones int8 inputs through the full entry point.
        // Config-driven: 4x8x8 (int8xi8, pmode=0) -> 32 output elements.
        // Each output = sum(k=0..7) { 1 * 1 } = 8.
        let mut ctx = ExecutionContext::new();

        // Write all-ones to vector regs x0 (v0+v1) and x2 (v2+v3).
        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);

        // Write config to scalar r5.
        ctx.scalar.write(5, config_i8xi8_accumulate());

        let op = make_mac_op(
            SemanticOp::Mac,
            0, // x0
            2, // x2
            5, // config in r5
            0, // cm0
            None,
        );

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled, "execute_matmul should handle Mac semantic");

        // Read back accumulator cm0.
        let acc = ctx.accumulator.read_wide(0);

        // For 4x8x8 int8, there are 32 output elements (Acc32, two per u64).
        // Each = 8.
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(
                val, 8,
                "output[{}]: expected 8, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_execute_matmul_bf16_ones() {
        // All bf16(1.0) inputs. Config-driven: 4x8x4 bf16 -> 16 outputs.
        // Each output = sum(k=0..7) { 1.0 * 1.0 } = 8.0.
        let mut ctx = ExecutionContext::new();

        let ones = vec512_all_ones_bf16();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_bf16_accumulate());

        let op = make_mac_op(
            SemanticOp::Mac,
            0,
            2,
            5,
            0,
            Some("VMAC_F_vmac_bm_core_dense"),
        );

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        // bf16 mode: 16 fp32 outputs, two per u64 lane.
        for i in 0..16 {
            let lane = i / 2;
            let half = i % 2;
            let bits = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as u32;
            let val = f32::from_bits(bits);
            assert!(
                (val - 8.0).abs() < 0.01,
                "output[{}]: expected 8.0, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_execute_matmul_negate() {
        // NegMul semantic: output = -(A * B).
        // All-ones int8 -> each product sum = 8, negated = -8.
        let mut ctx = ExecutionContext::new();

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_i8xi8_accumulate());

        let op = make_mac_op(
            SemanticOp::NegMul,
            0,
            2,
            5,
            0,
            None,
        );

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as u32 as i32;
            assert_eq!(
                val, -8,
                "output[{}]: expected -8, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_execute_matmul_zero_acc() {
        // Verify zero_acc=1 clears the accumulator before multiply.
        let mut ctx = ExecutionContext::new();

        // Pre-fill accumulator with garbage.
        let mut preload = [0u64; 16];
        for i in 0..16 {
            preload[i] = 0xDEAD_BEEF_CAFE_BABEu64;
        }
        ctx.accumulator.write_wide(0, preload);

        let ones = vec512_all_ones_i8();
        ctx.vector.write_wide(0, ones);
        ctx.vector.write_wide(2, ones);
        ctx.scalar.write(5, config_i8xi8_zero_acc());

        let op = make_mac_op(
            SemanticOp::Mac,
            0,
            2,
            5,
            0,
            None,
        );

        let handled = execute_matmul(&op, &mut ctx);
        assert!(handled);

        let acc = ctx.accumulator.read_wide(0);
        // With zero_acc, accumulator is cleared first, so result = 0 + products = 8.
        for i in 0..32 {
            let lane = i / 2;
            let half = i % 2;
            let val = ((acc[lane] >> (half * 32)) & 0xFFFF_FFFF) as i32;
            assert_eq!(
                val, 8,
                "output[{}]: expected 8 (zero_acc should clear), got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_execute_matmul_returns_false_for_non_mac() {
        // Verify non-MAC semantics return false.
        let mut ctx = ExecutionContext::new();
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add);
        assert!(!execute_matmul(&op, &mut ctx));
    }
}
