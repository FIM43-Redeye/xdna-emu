//! Miscellaneous operations for the vector ALU.
//!
//! Extracted from vector.rs -- shuffle, broadcast, extract, insert,
//! align, copy, clear, bitwise, and mask expansion functions.
//!
//! Contains both pure compute functions (vector_shuffle, vector_broadcast, etc.)
//! and execute_* dispatch functions that combine narrow/wide paths.

use crate::interpreter::bundle::{ElementType, Operand, ShufflePattern, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::tablegen::SemanticOp;

use super::vector_dispatch::VectorAlu;

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

    // ========== Dispatch functions (combine narrow + wide paths) ==========

    /// Shuffle: VSHUFFLE is a 512-bit-only operation.
    ///
    /// Narrow path returns false (shuffle is wide-only).
    /// Wide path handles both VSHUFFLE and VBCSTSHFL (broadcast + shuffle).
    pub(super) fn execute_shuffle(op: &SlotOp, ctx: &mut ExecutionContext, _et: ElementType) -> bool {
        if !op.is_wide_vector {
            // VSHUFFLE is a 512-bit operation handled in the wide path.
            // If we reach here via the narrow (256-bit) fallback path,
            // it means execute_wide didn't handle it -- should not happen
            // for real VSHUFFLE instructions. Return false to signal
            // unhandled (avoids panic from calling wide helpers in narrow context).
            return false;
        }

        // VSHUFFLE = s2v_interleave_sw(s1, s2, mode)
        // Two 512-bit vector inputs shuffled through the 48-mode
        // crossbar, with mode from the scalar `mod` operand.
        let (s1, s2) = Self::get_two_wide_vec_sources(op, ctx);
        let mode_val = Self::get_scalar_source(op, ctx);
        let mode_idx = (mode_val & 0x3F) as u8;

        let mut lo_bytes = [0u8; 64];
        let mut hi_bytes = [0u8; 64];
        for i in 0..16 {
            lo_bytes[i * 4..i * 4 + 4].copy_from_slice(&s1[i].to_le_bytes());
            hi_bytes[i * 4..i * 4 + 4].copy_from_slice(&s2[i].to_le_bytes());
        }

        let out_bytes = if let Some(mode) = super::vector_permute::ShuffleMode::from_mode(mode_idx) {
            super::vector_permute::shuffle_vectors(&lo_bytes, &hi_bytes, mode)
        } else {
            // mode >= 48: mask overflows to 0, crossbar passes
            // only byte 0 of lo input, rest is zero.
            let mut z = [0u8; 64];
            z[0] = lo_bytes[0];
            z
        };

        let mut result = [0u32; 16];
        for i in 0..16 {
            result[i] = u32::from_le_bytes([
                out_bytes[i * 4], out_bytes[i * 4 + 1],
                out_bytes[i * 4 + 2], out_bytes[i * 4 + 3],
            ]);
        }
        Self::write_wide_vec_dest(op, ctx, result);
        true
    }

    /// VectorBroadcast: broadcast scalar to vector.
    ///
    /// Three variants:
    /// 1. VBCST: broadcast scalar to all lanes
    /// 2. VBCST_64: broadcast 64-bit scalar pair to all lanes
    /// 3. VEXTBCST: extract element from vector, then broadcast
    ///
    /// Wide path additionally handles VBCSTSHFL (broadcast + shuffle).
    pub(super) fn execute_vector_broadcast(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let has_vector_source = op.sources.iter().any(|s| matches!(s, Operand::VectorReg(_)));

        if op.is_wide_vector {
            if has_vector_source {
                // VEXTBCST: extract element from 512-bit source, then broadcast
                let src = Self::get_wide_vec_source(op, ctx, 0);
                let index = Self::get_lane_index(op, ctx);
                let value = Self::extract_wide_element(&src, index, et);
                let mut result = [0u32; 16];
                if et.bits() >= 64 {
                    // 64-bit broadcast: replicate lo:hi pairs across 512 bits
                    let lo = value as u32;
                    let hi = (value >> 32) as u32;
                    for i in 0..8 {
                        result[i * 2] = lo;
                        result[i * 2 + 1] = hi;
                    }
                } else {
                    let narrow_result = Self::vector_broadcast(value as u32, et);
                    result[..8].copy_from_slice(&narrow_result);
                    result[8..].copy_from_slice(&narrow_result);
                }
                Self::write_wide_vec_dest(op, ctx, result);
            } else {
                // VBCST / VBCSTSHFL: broadcast scalar to 512-bit vector.
                // VBCSTSHFL additionally applies a 16-bit matrix transpose.

                // Step 1: broadcast
                let mut result = if matches!(et, ElementType::Int64 | ElementType::UInt64) {
                    // 64-bit: read register pair for full 64-bit value
                    let val64 = Self::get_scalar_source_64(op, ctx);
                    let lo = val64 as u32;
                    let hi = (val64 >> 32) as u32;
                    let mut r = [0u32; 16];
                    for i in 0..8 {
                        r[i * 2] = lo;
                        r[i * 2 + 1] = hi;
                    }
                    r
                } else {
                    let value = Self::get_scalar_source(op, ctx);
                    let narrow_result = Self::vector_broadcast(value, et);
                    let mut r = [0u32; 16];
                    r[..8].copy_from_slice(&narrow_result);
                    r[8..].copy_from_slice(&narrow_result);
                    r
                };

                // Step 2: VBCSTSHFL applies an implicit 16-bit matrix
                // transpose based on element size (observed on NPU1,
                // r29=0). The transpose groups all copies of each
                // 16-bit sub-component together:
                //   .8:  T32_2x16Lo(broadcast, zeros)
                //   .16: identity (1 component, no rearrangement)
                //   .32: 128-bit blocks (4 copies x 2 components)
                //   .64: 256-bit blocks (4 copies x 4 components)
                let is_shfl = op.encoding_name.as_deref()
                    .map_or(false, |n| n.contains("SHFL") || n.contains("shfl"));

                if is_shfl {
                    // Per aietools ISG (me_inline_primitives.h):
                    // VBCSTSHFL = s2v_interleave_sw(broadcast, ZEROS, r29)
                    // i.e., shuffle_vectors(broadcast, zeros, mode=r29)
                    //
                    // mode_decode: mask = u48(1) << (r29 & 0x3F)
                    // When mode >= 48, the shift exceeds the 48-bit field width
                    // and the hardware produces mask = 0 (no routing bits set).
                    // With mask = 0, the crossbar passes byte 0 through and
                    // zeros the remaining 63 bytes.
                    let r29 = ctx.scalar_read(29);
                    let mode_idx = (r29 & 0x3F) as u8;

                    if let Some(mode) = super::vector_permute::ShuffleMode::from_mode(mode_idx) {
                        let mut lo_bytes = [0u8; 64];
                        let hi_bytes = [0u8; 64];
                        for i in 0..16 {
                            lo_bytes[i * 4..i * 4 + 4].copy_from_slice(&result[i].to_le_bytes());
                        }
                        let shuffled = super::vector_permute::shuffle_vectors(&lo_bytes, &hi_bytes, mode);
                        for i in 0..16 {
                            result[i] = u32::from_le_bytes([
                                shuffled[i * 4], shuffled[i * 4 + 1],
                                shuffled[i * 4 + 2], shuffled[i * 4 + 3],
                            ]);
                        }
                    } else {
                        // mode >= 48: mode_decode overflows the 48-bit mask,
                        // producing mask = 0. The crossbar with no routing
                        // bits passes only byte 0 of the lo input; everything
                        // else is zero. Convert the broadcast to [byte0, 0..].
                        let byte0 = result[0] as u8;
                        result = [0u32; 16];
                        result[0] = byte0 as u32;
                    }
                }

                Self::write_wide_vec_dest(op, ctx, result);
            }
        } else {
            // Narrow path
            if matches!(et, ElementType::Int64 | ElementType::UInt64) && !has_vector_source {
                // 64-bit broadcast: read register pair (rN, rN+1) as lo:hi.
                let val64 = Self::get_scalar_source_64(op, ctx);
                let lo = val64 as u32;
                let hi = (val64 >> 32) as u32;
                let mut result = [0u32; 8];
                for i in 0..4 {
                    result[i * 2] = lo;
                    result[i * 2 + 1] = hi;
                }
                Self::write_vector_dest(op, ctx, result);
            } else {
                let value = if has_vector_source {
                    // VEXTBCST: extract element at index from vector source
                    let src = Self::get_vector_source(op, ctx, 0);
                    let index = Self::get_lane_index(op, ctx);
                    Self::extract_element_by_index(&src, index, et)
                } else {
                    // VBCST: broadcast scalar value
                    Self::get_scalar_source(op, ctx)
                };

                let result = Self::vector_broadcast(value, et);
                Self::write_vector_dest(op, ctx, result);
            }
        }
        true
    }

    /// VectorExtract: extract element from vector to scalar.
    ///
    /// Wide path operates on full 512-bit source.
    ///
    /// Writeback to scalar GPR is deferred by 2 cycles. The vec2scl functional
    /// unit ends at pipeline stage E2 in the chess scheduler model
    /// (`fu_vec2scl_E2` / `l_wm_copy0_v2s_scl_o_E2` in aie_ml/lib/isg/me_iss.isb),
    /// matching II_VEXTRACT def-latency=2 in AIE2Schedule.td. Without the
    /// deferral, kernels that overwrite a register holding a recent LDA value
    /// (e.g., cascade matmul get-only) see the new VEXTRACT value too early
    /// and break the consumer that was scheduled to read the LDA value first.
    pub(super) fn execute_vector_extract(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        const VEXTRACT_TO_GPR_LATENCY: u64 = 2;
        if op.is_wide_vector {
            // VEXTRACT operates on a full 512-bit source.
            // Cannot use fallback (execute_half twice) because the second
            // call overwrites the scalar dest with a result from the wrong half.
            let src = Self::get_wide_vec_source(op, ctx, 0);
            let index = Self::get_lane_index(op, ctx);
            let value = Self::extract_wide_element(&src, index, et);
            if et.bits() >= 64 {
                // 64-bit extract: write register pair (rN, rN+1)
                if let Some(dest) = op.dest.clone() {
                    ctx.queue_scalar_load(dest, value as u32, VEXTRACT_TO_GPR_LATENCY);
                    if let Some(Operand::ScalarReg(r)) = &op.dest {
                        ctx.queue_scalar_load(
                            Operand::ScalarReg(r + 1),
                            (value >> 32) as u32,
                            VEXTRACT_TO_GPR_LATENCY,
                        );
                    }
                }
            } else if let Some(dest) = op.dest.clone() {
                ctx.queue_scalar_load(dest, value as u32, VEXTRACT_TO_GPR_LATENCY);
            }
        } else {
            let src = Self::get_vector_source(op, ctx, 0);
            let index = Self::get_lane_index(op, ctx);
            let result = Self::vector_extract(&src, index, et);
            if let Some(dest) = op.dest.clone() {
                ctx.queue_scalar_load(dest, result, VEXTRACT_TO_GPR_LATENCY);
            }
        }
        true
    }

    /// VectorInsert: insert scalar into vector.
    ///
    /// Uses r29 as implicit index register. Wide path handles 64-bit inserts.
    pub(super) fn execute_vector_insert(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if op.is_wide_vector {
            // VINSERT.N dst, s1, idx, s0: copy s1 with s1[idx] = s0.
            // Cannot use fallback (execute_half twice) because it
            // inserts at the same index in BOTH halves.
            //
            // Decoded sources: [s1 (VectorReg), idx (ScalarReg r29), s0 (ScalarReg)].
            let base = Self::get_wide_vec_source(op, ctx, 0);
            let index = ctx.scalar_read(29);  // r29: implicit index register
            if matches!(et, ElementType::Int64 | ElementType::UInt64) {
                // 64-bit: s0 is a register pair (rN+1:rN). Read both halves.
                // get_nth_scalar_source returns the pair's base register value;
                // we need to find the actual register number to read rN+1.
                let mut s0_reg = None;
                let mut scalar_count = 0;
                for src in &op.sources {
                    if let Operand::ScalarReg(r) = src {
                        if scalar_count == 1 { s0_reg = Some(*r); break; }
                        scalar_count += 1;
                    }
                }
                let reg = s0_reg.unwrap_or(0);
                let lo = ctx.scalar_read(reg);
                let hi = ctx.scalar_read(reg + 1);
                let result = Self::insert_wide_element_64(&base, index, lo, hi);
                Self::write_wide_vec_dest(op, ctx, result);
            } else {
                let value = Self::get_nth_scalar_source(op, ctx, 1);  // s0 (skip idx)
                let result = Self::insert_wide_element(&base, index, value, et);
                Self::write_wide_vec_dest(op, ctx, result);
            }
        } else {
            // VINSERT.N dst, s1, idx, s0: copy s1 with s1[idx] replaced by s0.
            // The base vector is s1 (sources[0]), NOT the current dst value.
            // VPUSH (shift+insert) is handled in the execute_wide path.
            //
            // Decoded sources: [s1 (VectorReg), idx (ScalarReg r29), s0 (ScalarReg)].
            // Index is always r29 (implicit), value is the second scalar source.
            let mut base = Self::get_vector_source(op, ctx, 0);
            let index = ctx.scalar_read(29);  // r29: implicit index register
            let value = Self::get_nth_scalar_source(op, ctx, 1);  // s0 (skip idx)
            Self::vector_insert(&mut base, value, index, et);
            Self::write_vector_dest(op, ctx, base);
        }
        true
    }

    /// VectorPush / VectorPushHi: push scalar into vector, shifting elements.
    ///
    /// Wide-only operation (512-bit).
    pub(super) fn execute_vector_push(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        let src = Self::get_wide_vec_source(op, ctx, 0);
        // For 64-bit VPUSH, read a register pair (rN, rN+1).
        let value = if et.bits() >= 64 {
            Self::get_scalar_source_64(op, ctx)
        } else {
            Self::get_scalar_source(op, ctx) as u64
        };
        let is_hi = matches!(op.semantic, Some(SemanticOp::VectorPushHi));
        let result = Self::wide_vector_push(&src, value, is_hi, et);
        Self::write_wide_vec_dest(op, ctx, result);
        true
    }

    /// Align (VSHIFT): barrel shift / concatenate two vectors.
    ///
    /// Narrow path: simple byte-shift concatenation.
    /// Wide path: mask-based merge + barrel shift + optional pre-shift.
    pub(super) fn execute_align(op: &SlotOp, ctx: &mut ExecutionContext, _et: ElementType) -> bool {
        if op.is_wide_vector {
            // VSHIFT / VSHIFT_ALIGN barrel shifter.
            //
            // VSHIFT:       sources = [s1, s2, shift(Scalar)]
            //               step = 0 (no pre-shift stage)
            //
            // VSHIFT_ALIGN: sources = [s1, step(Scalar), s2, shift(Scalar)]
            //               step = s-register value (pre-shift selector)
            //
            // The hardware uses a mask-based merge + barrel shift +
            // optional pre-shift merge. See wide_vector_shift() docs.
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);

            let n_scalars = op.sources.iter().filter(|s| {
                matches!(s, Operand::ScalarReg(_) | Operand::Immediate(_))
            }).count();

            let (step, shift) = if n_scalars >= 2 {
                // VSHIFT_ALIGN: first scalar = step, second = shift
                let step = Self::get_nth_scalar_source(op, ctx, 0);
                let shift = Self::get_nth_scalar_source(op, ctx, 1);
                (step, shift)
            } else {
                // VSHIFT: single scalar = shift, step = 0
                (0, Self::get_nth_scalar_source(op, ctx, 0))
            };

            let result = Self::wide_vector_shift(&a, &b, step, shift);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            // Concatenate two vectors and shift
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let shift = Self::get_lane_index(op, ctx);
            let result = Self::vector_align(&a, &b, shift);
            Self::write_vector_dest(op, ctx, result);
        }
        true
    }

    /// Copy: vector/accumulator move.
    ///
    /// Handles all register-file move combinations:
    /// - VectorReg -> VectorReg (narrow 256-bit and wide 512-bit)
    /// - AccumReg -> AccumReg (bm 512-bit and cm 1024-bit)
    /// - VectorReg -> AccumReg (e.g., vmov bmh0, x1)
    /// - AccumReg -> VectorReg (e.g., vmov x0, bml0)
    pub(super) fn execute_copy(op: &SlotOp, ctx: &mut ExecutionContext, _et: ElementType) -> bool {
        let has_acc_source = op.sources.iter()
            .any(|s| matches!(s, Operand::AccumReg(_)));
        let has_acc_dest = matches!(&op.dest, Some(Operand::AccumReg(_)));

        if has_acc_source && has_acc_dest {
            // Accum -> Accum (bm or cm move).
            let src_reg = op.sources.iter().find_map(|s| match s {
                Operand::AccumReg(r) => Some(*r),
                _ => None,
            }).unwrap_or(0);
            let is_half = matches!(op.accum_width,
                Some(crate::tablegen::decoder_ffi::AccumWidth::Half));
            if !is_half {
                // Accumulator move: vmov cm_dst, cm_src
                let data = ctx.accumulator.read_wide(src_reg);
                Self::write_wide_acc_dest(op, ctx, data);
            } else {
                // Half-accum move: vmov bm_dst, bm_src
                let data = ctx.accumulator.read(src_reg);
                let dst = Self::get_acc_dest(op);
                ctx.accumulator.write(dst, data);
            }
        } else if has_acc_source || has_acc_dest {
            // Cross-register-file move (vector <-> accumulator).
            // get_vector_source handles AccumReg sources (truncates u64->u32).
            // write_vector_dest handles AccumReg destinations (zero-extends u32->u64).
            let src = Self::get_vector_source(op, ctx, 0);
            Self::write_vector_dest(op, ctx, src);
        } else if op.is_wide_vector {
            // Wide vector move: vmov x_dst_wide, x_src_wide (512-bit)
            let a = Self::get_wide_vec_source(op, ctx, 0);
            Self::write_wide_vec_dest(op, ctx, a);
        } else {
            // Narrow vector move: vmov x_dst, x_src (256-bit)
            let src = Self::get_vector_source(op, ctx, 0);
            Self::write_vector_dest(op, ctx, src);
        }
        true
    }

    /// VectorClear: zero a register.
    ///
    /// Handles both vector and accumulator clearing (narrow and wide).
    pub(super) fn execute_vector_clear(op: &SlotOp, ctx: &mut ExecutionContext, _et: ElementType) -> bool {
        // VCLR cm targets don't set is_wide_vector but need the wide path.
        let has_acc_dest = matches!(&op.dest, Some(Operand::AccumReg(_)));
        if op.is_wide_vector || has_acc_dest {
            // Handle both vector and accumulator clears.
            let has_acc_dest = matches!(&op.dest, Some(Operand::AccumReg(_)));
            if has_acc_dest {
                let is_half = matches!(op.accum_width,
                    Some(crate::tablegen::decoder_ffi::AccumWidth::Half));
                if !is_half {
                    Self::write_wide_acc_dest(op, ctx, [0u64; 16]);
                } else {
                    let dst = Self::get_acc_dest(op);
                    ctx.accumulator.clear(dst);
                }
            } else {
                Self::write_wide_vec_dest(op, ctx, [0u32; 16]);
            }
        } else {
            Self::write_vector_dest(op, ctx, [0u32; 8]);
        }
        true
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    #[test]
    fn test_vector_shuffle_mode0() {
        // VSHUFFLE with mode=0: verify the crossbar is invoked and
        // produces a non-trivial permutation of the input data.
        let mut ctx = make_ctx();
        let mut lo = [0u32; 16];
        let mut hi = [0u32; 16];
        for i in 0..16 {
            lo[i] = (i as u32) + 1;
            hi[i] = (i as u32) + 0x100;
        }
        ctx.vector.write_wide(0, lo);
        ctx.vector.write_wide(2, hi);
        ctx.scalar.write(5, 0);

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Shuffle)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(4))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(2))
            .with_source(Operand::ScalarReg(5));
        op.is_wide_vector = true;

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read_wide(4);
        // The crossbar should produce a permutation that differs from
        // both inputs -- not identity, not zeros.
        assert_ne!(result, lo, "result should not be identity of lo");
        assert_ne!(result, hi, "result should not be identity of hi");
        assert_ne!(result, [0u32; 16], "result should not be all zeros");
    }

    #[test]
    fn test_vector_shuffle_overflow_mode() {
        // mode >= 48: mask overflows, only byte 0 of lo passes through
        let mut ctx = make_ctx();
        ctx.vector.write_wide(0, [0xDEADBEEF, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        ctx.vector.write_wide(2, [0xFF; 16]);
        ctx.scalar.write(3, 50); // mode 50 >= 48

        let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Shuffle)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(4))
            .with_source(Operand::VectorReg(0))
            .with_source(Operand::VectorReg(2))
            .with_source(Operand::ScalarReg(3));
        op.is_wide_vector = true;

        VectorAlu::execute(&op, &mut ctx);
        let result = ctx.vector.read_wide(4);
        // Only byte 0 (0xEF) passes, everything else zero
        assert_eq!(result[0], 0xEF);
        for i in 1..16 {
            assert_eq!(result[i], 0, "lane {} should be zero", i);
        }
    }

    /// vpush.lo: scalar inserted at index 0, existing elements shift up.
    /// Verifies data crossing the 256-bit (word index 8) boundary.
    #[test]
    fn test_wide_vector_push_lo_32() {
        let mut src = [0u32; 16];
        for i in 0..16 { src[i] = (i as u32 + 1) * 100; }
        let result = VectorAlu::wide_vector_push(&src, 0xDEAD_BEEF_u64, false, ElementType::Int32);
        assert_eq!(result[0],  0xDEAD_BEEF, "inserted value at lo end");
        assert_eq!(result[1],  100,         "former element 0 shifted to 1");
        assert_eq!(result[8],  800,         "former element 7 crossed 256-bit boundary");
        assert_eq!(result[15], 1500,        "former element 14 at high end");
    }

    /// vpush.hi: scalar inserted at the highest position, existing elements
    /// shift down.  Verifies element that was in the high half appears in lo.
    #[test]
    fn test_wide_vector_push_hi_32() {
        let mut src = [0u32; 16];
        for i in 0..16 { src[i] = (i as u32 + 1) * 100; }
        let result = VectorAlu::wide_vector_push(&src, 0xCAFE_BABE_u64, true, ElementType::Int32);
        assert_eq!(result[0],  200,         "former element 1 shifted to 0");
        assert_eq!(result[7],  900,         "former element 8 crossed boundary to 7");
        assert_eq!(result[14], 1600,        "former element 15 at second-to-last");
        assert_eq!(result[15], 0xCAFE_BABE, "inserted value at hi end");
    }

    /// vpush.lo.64: 64-bit scalar inserted at the low end, consuming 8 bytes.
    #[test]
    fn test_wide_vector_push_lo_64() {
        let mut src = [0u32; 16];
        for i in 0..16 { src[i] = (i as u32 + 1) * 100; }
        let result = VectorAlu::wide_vector_push(&src, 0x1234_5678_ABCD_EF00_u64, false, ElementType::Int64);
        // Low 32 bits at word 0, high 32 bits at word 1.
        assert_eq!(result[0], 0xABCD_EF00, "64-bit value low word");
        assert_eq!(result[1], 0x1234_5678, "64-bit value high word");
        // Former element 0 (words 0-1) shifted to words 2-3.
        assert_eq!(result[2], 100, "former word 0 shifted to word 2");
        assert_eq!(result[3], 200, "former word 1 shifted to word 3");
    }

    /// vpush.hi.64: 64-bit scalar inserted at the high end, consuming 8 bytes.
    #[test]
    fn test_wide_vector_push_hi_64() {
        let mut src = [0u32; 16];
        for i in 0..16 { src[i] = (i as u32 + 1) * 100; }
        let result = VectorAlu::wide_vector_push(&src, 0xDEAD_BEEF_CAFE_0000_u64, true, ElementType::Int64);
        // High 8 bytes = inserted value.
        assert_eq!(result[14], 0xCAFE_0000, "64-bit value low word at hi end");
        assert_eq!(result[15], 0xDEAD_BEEF, "64-bit value high word at hi end");
        // Former words shifted down by 2.
        assert_eq!(result[0], 300, "former word 2 shifted to word 0");
        assert_eq!(result[1], 400, "former word 3 shifted to word 1");
    }

    /// Element 8 of i32 lives in word index 8 (the high 256-bit half).
    #[test]
    fn test_extract_wide_element_high_half() {
        let mut src = [0u32; 16];
        src[8] = 0x4242_4242;
        let val = VectorAlu::extract_wide_element(&src, 8, ElementType::Int32);
        assert_eq!(val, 0x4242_4242);
    }

    /// Element 17 of i16: bit offset = 17*16 = 272.  Word index = 272/32 = 8,
    /// bit-in-word = 272%32 = 16, so it is the high 16 bits of word 8.
    #[test]
    fn test_extract_wide_element_16bit() {
        let mut src = [0u32; 16];
        src[8] = 0xBEEF_DEAD; // lo16 = 0xDEAD, hi16 = 0xBEEF
        // UInt16: no sign extension, raw 16-bit value
        let val = VectorAlu::extract_wide_element(&src, 17, ElementType::UInt16);
        assert_eq!(val, 0xBEEF);
        // Int16: sign-extended (0xBEEF is negative as i16)
        let val_signed = VectorAlu::extract_wide_element(&src, 17, ElementType::Int16);
        assert_eq!(val_signed, 0xBEEF_u16 as i16 as i32 as u32 as u64);
    }

    /// Insert i32 at element 8 (high half, word index 8).
    #[test]
    fn test_insert_wide_element_high_half() {
        let src = [0u32; 16];
        let result = VectorAlu::insert_wide_element(&src, 8, 0xDEAD_BEEF, ElementType::Int32);
        assert_eq!(result[8], 0xDEAD_BEEF);
        // Other words remain zero.
        assert_eq!(result[0], 0);
        assert_eq!(result[7], 0);
        assert_eq!(result[9], 0);
    }

    /// Insert i16 at element 17 (high 16 bits of word 8).
    #[test]
    fn test_insert_wide_element_16bit() {
        let mut src = [0u32; 16];
        src[8] = 0x0000_DEAD; // low 16 bits should be preserved
        let result = VectorAlu::insert_wide_element(&src, 17, 0xBEEF, ElementType::Int16);
        assert_eq!(result[8], 0xBEEF_DEAD); // hi16=BEEF, lo16=DEAD (preserved)
    }

    /// Round-trip: insert then extract should return the inserted value.
    #[test]
    fn test_insert_extract_roundtrip() {
        let src = [0xFFFF_FFFFu32; 16];
        let after = VectorAlu::insert_wide_element(&src, 5, 0x42, ElementType::Int32);
        let val = VectorAlu::extract_wide_element(&after, 5, ElementType::Int32);
        assert_eq!(val, 0x42);
        // Neighbor should be untouched.
        let neighbor = VectorAlu::extract_wide_element(&after, 4, ElementType::Int32);
        assert_eq!(neighbor, 0xFFFF_FFFF);
    }

    /// Zero shift returns src1 unchanged.
    #[test]
    fn test_wide_vector_align_no_shift() {
        let src1 = [1u32; 16];
        let src2 = [2u32; 16];
        let result = VectorAlu::wide_vector_shift(&src1, &src2, 0, 0);
        assert_eq!(result, [1u32; 16]);
    }

    /// Shift by exactly 64 bytes skips all of src1 and returns src2 unchanged.
    #[test]
    fn test_wide_vector_align_full_shift() {
        let src1 = [1u32; 16];
        let src2 = [2u32; 16];
        let result = VectorAlu::wide_vector_shift(&src1, &src2, 0, 64);
        assert_eq!(result, [2u32; 16]);
    }

    /// Shift by 60 bytes: result[0] = last word of src1, result[1] = first
    /// word of src2.  This exercises the cross-boundary stitch path.
    #[test]
    fn test_wide_vector_align_cross_boundary() {
        let mut src1 = [0u32; 16];
        let mut src2 = [0u32; 16];
        src1[15] = 0xAAAA_AAAA; // last word of src1
        src2[0]  = 0xBBBB_BBBB; // first word of src2
        let result = VectorAlu::wide_vector_shift(&src1, &src2, 0, 60);
        assert_eq!(result[0], 0xAAAA_AAAA, "last word of src1 at result[0]");
        assert_eq!(result[1], 0xBBBB_BBBB, "first word of src2 at result[1]");
    }

    #[test]
    fn test_vector_mov() {
        let mut ctx = make_ctx();
        ctx.vector.write(0, [10, 20, 30, 40, 50, 60, 70, 80]);

        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Copy)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::VectorReg(0));

        VectorAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.vector.read(1), [10, 20, 30, 40, 50, 60, 70, 80]);
    }
}
