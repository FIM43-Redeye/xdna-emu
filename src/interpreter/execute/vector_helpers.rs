//! Shared helper methods for the VectorAlu unit.
//!
//! This module contains operand access, result writing, comparison mask
//! packing, scalar/lane helpers, wide vector manipulation, and generic
//! dispatch helpers. All methods are on `impl VectorAlu` (Rust allows
//! multiple impl blocks across files in the same crate).
//!
//! Extracted from vector.rs to separate shared infrastructure from
//! dispatch logic and operation implementations.

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::{ExecutionContext, Vec512, Acc1024};

use super::vector_dispatch::VectorAlu;
use super::vector_pack;

impl VectorAlu {
    // ========== Operand Access ==========

    /// Get two vector source operands.
    pub(super) fn get_two_vector_sources(op: &SlotOp, ctx: &ExecutionContext) -> ([u32; 8], [u32; 8]) {
        let a = Self::get_vector_source(op, ctx, 0);
        let b = Self::get_vector_source(op, ctx, 1);
        (a, b)
    }

    /// Get a single vector source operand.
    pub(super) fn get_vector_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> [u32; 8] {
        op.sources.get(idx).map_or([0; 8], |src| Self::read_vector_operand(src, ctx))
    }

    /// Read a vector operand value.
    ///
    /// Handles both VectorReg (native 256-bit) and AccumReg (truncated from
    /// 512-bit to 256-bit by taking the low 32 bits of each u64 lane).
    /// This truncation matches hardware behavior for VMOV x, bml/bmh where
    /// the 256-bit accumulator half maps directly to 8 x u32 lanes.
    pub(super) fn read_vector_operand(operand: &Operand, ctx: &ExecutionContext) -> [u32; 8] {
        match operand {
            Operand::VectorReg(r) => ctx.vector.read(*r),
            Operand::AccumReg(r) => {
                // Accumulator -> vector: take low 32 bits of each lane.
                let acc = ctx.accumulator.read(*r);
                let mut v = [0u32; 8];
                for i in 0..8 {
                    v[i] = acc[i] as u32;
                }
                v
            }
            Operand::Immediate(val) => {
                // Scalar broadcast into all lanes (for immediate vector operands).
                [*val as u32; 8]
            }
            other => {
                log::error!(
                    "[VECTOR] read_vector_operand: unexpected operand {:?} -- \
                     returning zeros, check decoder",
                    other
                );
                [0; 8]
            }
        }
    }

    // ========== Result Writing ==========

    /// Write result to vector destination.
    ///
    /// Handles both VectorReg (native 256-bit) and AccumReg (widened from
    /// 256-bit to 512-bit by zero-extending each u32 lane to u64).
    pub(super) fn write_vector_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: [u32; 8]) {
        match &op.dest {
            Some(Operand::VectorReg(r)) => {
                ctx.vector.write(*r, value);
            }
            Some(Operand::AccumReg(r)) => {
                // Vector -> accumulator: zero-extend each u32 to u64.
                let mut acc = [0u64; 8];
                for i in 0..8 {
                    acc[i] = value[i] as u64;
                }
                ctx.accumulator.write(*r, acc);
            }
            Some(Operand::ScalarReg(r)) => {
                // Vector comparison -> scalar mask register (r16-r23).
                // Condense per-lane all-ones/all-zeros into a packed bitmask.
                let mask = Self::condense_comparison_mask(&value, op.element_type);
                ctx.scalar.write(*r, mask);
            }
            Some(other) => {
                log::error!(
                    "[VECTOR] write_vector_dest: unexpected dest {:?} -- \
                     result discarded, check decoder",
                    other
                );
            }
            None => {
                // Some vector operations have no explicit destination
                // (e.g., comparison that sets status flags).
            }
        }
    }

    // ========== Comparison Masks ==========

    /// Condense a per-lane comparison mask ([u32; 8] of all-ones/zeros)
    /// into a packed scalar bitmask. One bit per element lane.
    ///
    /// - 32-bit elements: 8 lanes -> 8-bit mask (bits [7:0])
    /// - 16-bit elements: 16 lanes -> 16-bit mask (bits [15:0])
    /// - 8-bit elements: 32 lanes -> 32-bit mask (bits [31:0])
    pub(super) fn condense_comparison_mask(value: &[u32; 8], elem_type: Option<ElementType>) -> u32 {
        match elem_type {
            Some(ElementType::Int32) | Some(ElementType::UInt32) | Some(ElementType::Float32) | None => {
                // 8 x 32-bit lanes: one bit per u32 word
                let mut mask = 0u32;
                for i in 0..8 {
                    if value[i] != 0 {
                        mask |= 1 << i;
                    }
                }
                mask
            }
            Some(ElementType::Int16) | Some(ElementType::UInt16) | Some(ElementType::BFloat16) => {
                // 16 x 16-bit lanes: two lanes packed per u32 word
                let mut mask = 0u32;
                for i in 0..8 {
                    if value[i] & 0xFFFF != 0 {
                        mask |= 1 << (i * 2);
                    }
                    if value[i] & 0xFFFF_0000 != 0 {
                        mask |= 1 << (i * 2 + 1);
                    }
                }
                mask
            }
            Some(ElementType::Int8) | Some(ElementType::UInt8) => {
                // 32 x 8-bit lanes: four lanes packed per u32 word
                let mut mask = 0u32;
                for i in 0..8 {
                    for j in 0..4 {
                        if (value[i] >> (j * 8)) & 0xFF != 0 {
                            mask |= 1 << (i * 4 + j);
                        }
                    }
                }
                mask
            }
            other => {
                panic!("condense_comparison_mask: unhandled element type {:?}", other,);
            }
        }
    }

    /// Compute per-element comparison flags from a vector comparison result.
    ///
    /// Takes the expanded mask (all-ones or all-zeros per element) and packs
    /// it into a scalar bitmask: bit i = 1 if element i was non-zero.
    pub(super) fn pack_comparison_flags(mask: &[u32; 8], elem_type: ElementType) -> u32 {
        let mut flags: u32 = 0;
        match elem_type {
            ElementType::Int32
            | ElementType::UInt32
            | ElementType::Int64
            | ElementType::UInt64
            | ElementType::Float32 => {
                // 8 elements of 32-bit
                for i in 0..8 {
                    if mask[i] != 0 {
                        flags |= 1 << i;
                    }
                }
            }
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => {
                // 16 elements of 16-bit
                for i in 0..8 {
                    let lo = mask[i] & 0xFFFF;
                    let hi = (mask[i] >> 16) & 0xFFFF;
                    let bit = i * 2;
                    if lo != 0 {
                        flags |= 1 << bit;
                    }
                    if hi != 0 {
                        flags |= 1 << (bit + 1);
                    }
                }
            }
            ElementType::Int8 | ElementType::UInt8 => {
                // 32 elements of 8-bit
                for i in 0..8 {
                    for j in 0..4 {
                        let byte = (mask[i] >> (j * 8)) & 0xFF;
                        let bit = i * 4 + j;
                        if byte != 0 {
                            flags |= 1 << bit;
                        }
                    }
                }
            }
        }
        flags
    }

    // ========== Config / Accumulator Helpers ==========

    /// Get config register value from a scalar register in sources.
    ///
    /// The config register is the last scalar register source in the VMAC
    /// instruction (r0-r15 in `vmac cm0, cm0, x0, x0, r0`).
    /// Returns None if no scalar register is present (e.g., unit tests).
    pub(super) fn get_config_register(op: &SlotOp, ctx: &ExecutionContext) -> Option<u32> {
        // Scan sources in reverse order -- the config register is typically last.
        for src in op.sources.iter().rev() {
            if let Operand::ScalarReg(r) = src {
                return Some(ctx.scalar_read(*r));
            }
        }
        None
    }

    /// Get accumulator destination register.
    pub(super) fn get_acc_dest(op: &SlotOp) -> u8 {
        match &op.dest {
            Some(Operand::AccumReg(r)) => *r,
            other => {
                log::error!(
                    "[VECTOR] get_acc_dest: expected AccumReg, got {:?} -- defaulting to acc0",
                    other
                );
                0
            }
        }
    }

    /// Get accumulator source register from operands.
    pub(super) fn get_acc_source(op: &SlotOp) -> u8 {
        for src in &op.sources {
            if let Operand::AccumReg(r) = src {
                return *r;
            }
        }
        log::error!(
            "[VECTOR] get_acc_source: no AccumReg found in sources {:?} -- defaulting to acc0",
            op.sources
        );
        0
    }

    /// Get shift amount from operands (immediate or register).
    pub(super) fn get_shift_amount(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // Look for immediate in sources
        for src in &op.sources {
            if let Operand::Immediate(imm) = src {
                return *imm as u32;
            }
            if let Operand::ScalarReg(r) = src {
                return ctx.scalar_read(*r);
            }
        }
        0 // Default: no shift
    }

    // ========== Conversion Utilities ==========

    /// Convert BFloat16 (upper 16 bits of f32) to f32.
    #[inline]
    pub(super) fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// Convert f32 to BFloat16 (truncate lower 16 bits).
    #[inline]
    pub(super) fn f32_to_bf16(val: f32) -> u16 {
        (val.to_bits() >> 16) as u16
    }

    // ========== Pack / Unpack Wrappers ==========

    /// Pack: narrow 32-bit lanes to 16-bit (truncation mode).
    ///
    /// Delegates to the `vector_pack` module. Takes two 256-bit vectors
    /// of 32-bit elements and produces one 256-bit vector of 16-bit elements.
    /// Uses truncation mode (no saturation) since the SlotOp does not
    /// currently carry pack saturation information.
    pub(super) fn vector_pack(a: &[u32; 8], _b: &[u32; 8]) -> [u32; 8] {
        // The vector_pack module operates on a single register at a time,
        // narrowing from bits_i to bits_o. Pack the first source; the second
        // source would go into the upper half for a full 512-bit result,
        // but our register model is 256-bit so we pack just the first.
        vector_pack::pack_vector(a, 32, 16, false, vector_pack::PackMode::Truncate)
    }

    /// Unpack: widen 16-bit lanes to 32-bit (signed, sign-extend).
    ///
    /// Delegates to the `vector_pack` module. Takes a 256-bit vector of
    /// 16-bit elements and produces a 256-bit vector of 32-bit elements
    /// (lower half of the logical result).
    pub(super) fn vector_unpack_low(src: &[u32; 8]) -> [u32; 8] {
        vector_pack::unpack_vector(src, 16, 32, true)
    }

    // ========== Lane / Scalar Helpers ==========

    /// Get lane index from operands (immediate or register).
    pub(super) fn get_lane_index(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // Look for immediate in sources (typically the last source)
        for src in op.sources.iter().rev() {
            if let Operand::Immediate(imm) = src {
                return *imm as u32;
            }
            if let Operand::ScalarReg(r) = src {
                return ctx.scalar_read(*r);
            }
        }
        0 // Default: lane 0
    }

    /// Get scalar source value from operands.
    pub(super) fn get_scalar_source(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        for src in &op.sources {
            match src {
                Operand::ScalarReg(r) => return ctx.scalar_read(*r),
                Operand::Immediate(imm) => return *imm as u32,
                _ => {}
            }
        }
        0
    }

    /// Get 64-bit scalar source from a register pair (rN, rN+1).
    ///
    /// For 64-bit VPUSH, the scalar value spans two adjacent registers:
    /// rN holds the low 32 bits, rN+1 holds the high 32 bits.
    pub(super) fn get_scalar_source_64(op: &SlotOp, ctx: &ExecutionContext) -> u64 {
        for src in &op.sources {
            if let Operand::ScalarReg(r) = src {
                let lo = ctx.scalar_read(*r) as u64;
                let hi = ctx.scalar_read(r + 1) as u64;
                return lo | (hi << 32);
            }
        }
        0
    }

    /// Get the Nth scalar source operand (0-indexed among scalars/immediates).
    ///
    /// For instructions with multiple scalar operands (e.g., VINSERT has both
    /// idx and s0), type-scanning heuristics pick the wrong one. This function
    /// returns the Nth scalar in source order.
    pub(super) fn get_nth_scalar_source(op: &SlotOp, ctx: &ExecutionContext, n: usize) -> u32 {
        let mut count = 0;
        for src in &op.sources {
            match src {
                Operand::ScalarReg(r) => {
                    if count == n {
                        return ctx.scalar_read(*r);
                    }
                    count += 1;
                }
                Operand::Immediate(imm) => {
                    if count == n {
                        return *imm as u32;
                    }
                    count += 1;
                }
                _ => {}
            }
        }
        0
    }

    // ========== Scalar Destination Writers ==========

    /// Write scalar result to destination.
    pub(super) fn write_scalar_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: u32) {
        if let Some(Operand::ScalarReg(r)) = &op.dest {
            ctx.scalar.write(*r, value);
        }
    }

    /// Write a 64-bit scalar result to destination register pair (rN, rN+1).
    /// Used for 8-bit comparisons where the bitmask exceeds 32 bits.
    pub(super) fn write_scalar_dest_wide(op: &SlotOp, ctx: &mut ExecutionContext, value: u64) {
        if let Some(Operand::ScalarReg(r)) = &op.dest {
            ctx.scalar.write(*r, value as u32);
            ctx.scalar.write(*r + 1, (value >> 32) as u32);
        }
    }

    /// Write comparison flags to the secondary destination register (cmp) of
    /// dual-result instructions (VSUB_LT, VABS_GTZ, VNEG_GTZ, etc.).
    ///
    /// The cmp register receives a per-element bitmask: bit i is set when the
    /// comparison is true for element i.
    pub(super) fn write_cmp_dest(op: &SlotOp, ctx: &mut ExecutionContext, flags: u32) {
        if let Some(Operand::ScalarReg(r)) = op.extra_dests.first() {
            ctx.scalar.write(*r, flags);
        }
    }

    /// Write a 64-bit comparison bitmask to the cmp register pair (for 8-bit
    /// element comparisons that need 64 bits: 32 elements per half * 2 halves).
    pub(super) fn write_cmp_dest_wide(op: &SlotOp, ctx: &mut ExecutionContext, flags: u64) {
        if let Some(Operand::ScalarReg(r)) = op.extra_dests.first() {
            // Write low 32 bits to rN, high 32 bits to rN+1.
            ctx.scalar.write(*r, flags as u32);
            ctx.scalar.write(*r + 1, (flags >> 32) as u32);
        }
    }

    // ========== Wide (512-bit / 1024-bit) Helpers ==========

    /// Read the nth VectorReg source as a full 512-bit value.
    ///
    /// Unlike get_vector_source which reads a single 256-bit register,
    /// this reads a pair of consecutive registers (x-register = two w-registers).
    /// Skips non-VectorReg sources when counting, so idx=0 is the first
    /// VectorReg in sources, idx=1 is the second, etc.
    pub(super) fn get_wide_vec_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> Vec512 {
        let mut vec_count = 0;
        for src in &op.sources {
            if let Operand::VectorReg(r) = src {
                if vec_count == idx {
                    return ctx.vector.read_wide(*r);
                }
                vec_count += 1;
            }
        }
        [0u32; 16]
    }

    /// Read two wide vector sources.
    pub(super) fn get_two_wide_vec_sources(op: &SlotOp, ctx: &ExecutionContext) -> (Vec512, Vec512) {
        let a = Self::get_wide_vec_source(op, ctx, 0);
        let b = Self::get_wide_vec_source(op, ctx, 1);
        (a, b)
    }

    /// Write a 512-bit result to the vector destination.
    pub(super) fn write_wide_vec_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: Vec512) {
        if let Some(Operand::VectorReg(r)) = &op.dest {
            ctx.vector.write_wide(*r, value);
        } else {
            log::error!("[VECTOR_WIDE] write_wide_vec_dest: expected VectorReg dest, got {:?}", op.dest);
        }
    }

    /// Write a 1024-bit result to the accumulator destination.
    pub(super) fn write_wide_acc_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: Acc1024) {
        let reg = Self::get_acc_dest(op);
        ctx.accumulator.write_wide(reg, value);
    }

    // ========== Wide Dispatch Bridges ==========

    /// Bridge: apply a narrow element-wise function to a wide vector.
    ///
    /// Splits Vec512 into two [u32; 8] halves, applies the function to
    /// each half independently, and concatenates the results. Works for
    /// any operation where each output element depends only on
    /// corresponding input elements.
    pub(super) fn wide_element_wise_unary(
        a: &Vec512,
        et: ElementType,
        op_fn: fn(&[u32; 8], ElementType) -> [u32; 8],
    ) -> Vec512 {
        let a_lo: [u32; 8] = a[..8].try_into().unwrap();
        let a_hi: [u32; 8] = a[8..].try_into().unwrap();
        let r_lo = op_fn(&a_lo, et);
        let r_hi = op_fn(&a_hi, et);
        let mut result = [0u32; 16];
        result[..8].copy_from_slice(&r_lo);
        result[8..].copy_from_slice(&r_hi);
        result
    }

    /// Bridge: apply a narrow two-input element-wise function to wide vectors.
    pub(super) fn wide_element_wise_binary(
        a: &Vec512,
        b: &Vec512,
        et: ElementType,
        op_fn: fn(&[u32; 8], &[u32; 8], ElementType) -> [u32; 8],
    ) -> Vec512 {
        let a_lo: [u32; 8] = a[..8].try_into().unwrap();
        let a_hi: [u32; 8] = a[8..].try_into().unwrap();
        let b_lo: [u32; 8] = b[..8].try_into().unwrap();
        let b_hi: [u32; 8] = b[8..].try_into().unwrap();
        let r_lo = op_fn(&a_lo, &b_lo, et);
        let r_hi = op_fn(&a_hi, &b_hi, et);
        let mut result = [0u32; 16];
        result[..8].copy_from_slice(&r_lo);
        result[8..].copy_from_slice(&r_hi);
        result
    }

    // ========== Wide Element Manipulation ==========

    /// Push a scalar into a 512-bit vector, shifting existing elements.
    ///
    /// - `vpush.lo` (`is_hi=false`): shift elements toward high indices, insert
    ///   scalar at the lowest position (index 0).
    /// - `vpush.hi` (`is_hi=true`): shift elements toward low indices, insert
    ///   scalar at the highest position.
    ///
    /// The shift is element-size-aware: for i32, one push moves 4 bytes; for
    /// i16, 2 bytes; for i8, 1 byte.  The operation works on the full 64-byte
    /// (512-bit) vector, so elements cross the 256-bit lane boundary freely.
    pub(super) fn wide_vector_push(src: &Vec512, value: u64, is_hi: bool, et: ElementType) -> Vec512 {
        // Flatten to bytes for element-size-agnostic shifting.
        let mut bytes = [0u8; 64];
        for (i, word) in src.iter().enumerate() {
            let b = word.to_le_bytes();
            bytes[i * 4..i * 4 + 4].copy_from_slice(&b);
        }

        let elem_bytes = (et.bits() as usize / 8).max(1);
        let val_bytes = value.to_le_bytes();

        if is_hi {
            // Shift towards low indices, open a slot at the high end.
            bytes.copy_within(elem_bytes.., 0);
            let insert_pos = 64 - elem_bytes;
            for i in 0..elem_bytes {
                bytes[insert_pos + i] = val_bytes[i];
            }
        } else {
            // Shift towards high indices, open a slot at the low end.
            bytes.copy_within(..64 - elem_bytes, elem_bytes);
            for i in 0..elem_bytes {
                bytes[i] = val_bytes[i];
            }
        }

        let mut result = [0u32; 16];
        for (i, chunk) in bytes.chunks(4).enumerate() {
            result[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        result
    }

    /// Extract a single element from any position in a 512-bit vector.
    ///
    /// The element index is masked to the valid range for the given element
    /// type (e.g., 0-31 for i16, 0-15 for i32).  Supports sub-word types
    /// stored in packed little-endian order within each u32 word.
    pub(super) fn extract_wide_element(src: &Vec512, index: u32, et: ElementType) -> u64 {
        let bits = et.bits() as u32;
        if bits >= 64 {
            // 64-bit element: 8 elements in 512-bit register.
            let idx = (index as usize % 8) * 2;
            (src[idx] as u64) | ((src[idx + 1] as u64) << 32)
        } else {
            let max_elems = 512 / bits;
            let idx = index % max_elems;
            let bit_offset = idx * bits;
            let word_idx = (bit_offset / 32) as usize;
            let bit_in_word = bit_offset % 32;
            let mask = (1u64 << bits) - 1;
            let raw = ((src[word_idx] as u64 >> bit_in_word) & mask) as u32;
            // Sign-extend for signed types
            match et {
                ElementType::Int8 => (raw as u8 as i8 as i32 as u32) as u64,
                ElementType::Int16 => (raw as u16 as i16 as i32 as u32) as u64,
                _ => raw as u64,
            }
        }
    }

    /// Insert a 64-bit value (lo + hi words) at a specific 64-bit element position.
    pub(super) fn insert_wide_element_64(src: &Vec512, index: u32, lo: u32, hi: u32) -> Vec512 {
        let mut result = *src;
        // 512 bits / 64 bits = 8 elements
        let idx = (index as usize) % 8;
        result[idx * 2] = lo;
        result[idx * 2 + 1] = hi;
        result
    }

    /// Insert a scalar value at a specific element position in a 512-bit vector.
    ///
    /// Returns a copy of `src` with the element at `index` replaced by `value`.
    /// The index is masked to the valid range for the element type.
    pub(super) fn insert_wide_element(src: &Vec512, index: u32, value: u32, et: ElementType) -> Vec512 {
        let mut result = *src;
        let bits = et.bits() as u32;
        if bits >= 64 {
            // 64-bit element: only lower 32 bits available via this path.
            // Use insert_wide_element_64 for full 64-bit inserts.
            let idx = (index as usize % 8) * 2;
            result[idx] = value;
            return result;
        }
        let max_elems = 512 / bits;
        let idx = index % max_elems;
        let bit_offset = idx * bits;
        let word_idx = (bit_offset / 32) as usize;
        let bit_in_word = bit_offset % 32;
        let mask = (1u64 << bits) - 1;
        // Clear the target element bits and insert the new value.
        let word_val = result[word_idx] as u64;
        let cleared = word_val & !(mask << bit_in_word);
        let inserted = cleared | (((value as u64) & mask) << bit_in_word);
        result[word_idx] = inserted as u32;
        // Handle elements that span a word boundary (e.g., 64-bit at bit_in_word > 0).
        if bit_in_word + et.bits() as u32 > 32 && word_idx + 1 < 16 {
            let overflow_bits = bit_in_word + et.bits() as u32 - 32;
            let overflow_mask = (1u64 << overflow_bits) - 1;
            let hi_val = result[word_idx + 1] as u64;
            let hi_cleared = hi_val & !overflow_mask;
            let hi_inserted =
                hi_cleared | (((value as u64) >> (et.bits() as u32 - overflow_bits)) & overflow_mask);
            result[word_idx + 1] = hi_inserted as u32;
        }
        result
    }

    /// VSHIFT/VSHIFT_ALIGN barrel shifter.
    ///
    /// Implements the hardware barrel shift with optional pre-shift stage.
    /// Derived from aietools ISG reference model (`vshift_hw` in
    /// `me_inline_primitives.h`). The hardware uses a mask-based merge
    /// of vectors a and b, followed by a progressive barrel shift, then
    /// an optional pre-shifted copy of a merged into the high bytes.
    ///
    /// Parameters:
    ///   a, b: 512-bit source vectors
    ///   step: 3-bit pre-shift selector (from s-register & 0x7):
    ///         0 = no pre-shift (plain VSHIFT behavior)
    ///         1 = pre-shift a by 4 bytes (32 bits)
    ///         2 = pre-shift a by 8 bytes (64 bits)
    ///         3 = pre-shift a by 16 bytes (128 bits)
    ///         4 = pre-shift a by 32 bytes (256 bits)
    ///   shift: byte shift amount from r-register (full 32-bit value)
    pub(super) fn wide_vector_shift(a_orig: &Vec512, b: &Vec512, step: u32, shift: u32) -> Vec512 {
        // Convert to byte arrays for manipulation.
        let mut a_bytes = [0u8; 64];
        let mut b_bytes = [0u8; 64];
        for i in 0..16 {
            let aw = a_orig[i].to_le_bytes();
            let bw = b[i].to_le_bytes();
            a_bytes[i * 4..i * 4 + 4].copy_from_slice(&aw);
            b_bytes[i * 4..i * 4 + 4].copy_from_slice(&bw);
        }

        // Step 1: vshift_mask -- decompose shift register value.
        // Extract 6-bit shift amount and hi_shft flag.
        let shift_6 = (shift & 0x3F) as usize;
        let hi_shft = shift >= 64;

        // Build 64-bit mask: mask = (0xFFFF_FFFF_FFFF_FFFF << shift)[127:64]
        // This gives a mask where the high bits are set based on shift amount.
        let mask_128: u128 = 0xFFFF_FFFF_FFFF_FFFFu128 << (shift & 0x7F);
        let mask_64: u64 = (mask_128 >> 64) as u64;

        // Step 2: Compute pre-shifted version of a (for step != 0).
        let mut pre_bytes = [0u8; 64];
        let mut a_active = a_bytes;
        let step_val = step & 0x7;
        if step_val >= 1 && step_val <= 4 {
            // Pre-shift a right by 2^(step+1) bytes = {4, 8, 16, 32} bytes.
            let pre_shift_bits = match step_val {
                1 => 32,  // 4 bytes
                2 => 64,  // 8 bytes
                3 => 128, // 16 bytes
                4 => 256, // 32 bytes
                _ => 0,
            };
            // Right-shift a_orig by pre_shift_bits as a 512-bit value.
            let pre_shift_bytes = pre_shift_bits / 8;
            for i in 0..64 {
                let src_idx = i + pre_shift_bytes;
                pre_bytes[i] = if src_idx < 64 { a_bytes[src_idx] } else { 0 };
            }
            // When step != 0, a is zeroed for the main shift stage.
            a_active = [0u8; 64];
        }

        // Step 3: Build per-byte masks from the 64-bit mask.
        // maska: for vector a, maskb: for vector b.
        // If hi_shft: maska = 0 (all zero), else maska = ~mask (inverted).
        // maskb = mask.
        let maska_64: u64 = if hi_shft { 0 } else { !mask_64 };
        let maskb_64: u64 = mask_64;

        // Expand 1-bit-per-byte masks to full byte masks.
        let mut c_bytes = [0u8; 64];
        for i in 0..64 {
            let a_sel = (maska_64 >> i) & 1;
            let b_sel = (maskb_64 >> i) & 1;
            c_bytes[i] = if a_sel != 0 {
                a_active[i]
            } else if b_sel != 0 {
                b_bytes[i]
            } else {
                0
            };
        }

        // Step 4: Progressive barrel shift using shift[5:0] bit by bit.
        // Each bit rotates c by that power of 2 bytes.
        // Bit 5: rotate by 32 bytes
        // Bit 4: rotate by 16 bytes
        // Bit 3: rotate by 8 bytes
        // Bit 2: rotate by 4 bytes
        // Bit 1: rotate by 2 bytes
        // Bit 0: rotate by 1 byte
        // Each rotation: c = [c[0..N-1], c[N..63]] (low bytes move to top).
        for bit in (0..6).rev() {
            if (shift_6 >> bit) & 1 != 0 {
                let n = 1 << bit; // bytes to rotate
                let mut rotated = [0u8; 64];
                for i in 0..64 {
                    rotated[i] = c_bytes[(i + n) % 64];
                }
                c_bytes = rotated;
            }
        }

        // Step 5: Merge pre-shifted portion into result.
        // maskpre = bit-reversed maska (bit i of maskpre = bit (63-i) of maska).
        // result = c | (pre & maskpre)
        let mut maskpre_64: u64 = 0;
        for i in 0..64 {
            let bit = (maska_64 >> (63 - i)) & 1;
            maskpre_64 |= bit << i;
        }
        for i in 0..64 {
            if (maskpre_64 >> i) & 1 != 0 {
                c_bytes[i] |= pre_bytes[i];
            }
        }

        // Convert back to Vec512.
        let mut result = [0u32; 16];
        for i in 0..16 {
            result[i] = u32::from_le_bytes([
                c_bytes[i * 4],
                c_bytes[i * 4 + 1],
                c_bytes[i * 4 + 2],
                c_bytes[i * 4 + 3],
            ]);
        }
        result
    }

    // ========== Generic Dispatch Helpers ==========

    /// Generic dispatcher for binary element-wise operations.
    ///
    /// Handles the read-compute-write cycle for both narrow (256-bit) and
    /// wide (512-bit) paths. Eliminates per-op boilerplate for simple ops
    /// like Sub, Mul, Min, Max.
    pub(super) fn execute_binary_elementwise(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
        compute: fn(&[u32; 8], &[u32; 8], ElementType) -> [u32; 8],
    ) -> bool {
        if op.is_wide_vector {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let result = Self::wide_element_wise_binary(&a, &b, et, compute);
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = compute(&a, &b, et);
            Self::write_vector_dest(op, ctx, result);
        }
        true
    }

    /// Generic dispatcher for binary typeless operations (bitwise and/or/xor).
    /// Same as execute_binary_elementwise but compute fn has no ElementType param.
    pub(super) fn execute_binary_typeless(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        compute: fn(&[u32; 8], &[u32; 8]) -> [u32; 8],
    ) -> bool {
        if op.is_wide_vector {
            let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
            let a_lo: [u32; 8] = a[..8].try_into().unwrap();
            let a_hi: [u32; 8] = a[8..].try_into().unwrap();
            let b_lo: [u32; 8] = b[..8].try_into().unwrap();
            let b_hi: [u32; 8] = b[8..].try_into().unwrap();
            let mut result = [0u32; 16];
            result[..8].copy_from_slice(&compute(&a_lo, &b_lo));
            result[8..].copy_from_slice(&compute(&a_hi, &b_hi));
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let (a, b) = Self::get_two_vector_sources(op, ctx);
            let result = compute(&a, &b);
            Self::write_vector_dest(op, ctx, result);
        }
        true
    }

    /// Generic dispatcher for unary typeless operations (bitwise not).
    pub(super) fn execute_unary_typeless(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        compute: fn(&[u32; 8]) -> [u32; 8],
    ) -> bool {
        if op.is_wide_vector {
            let src = Self::get_wide_vec_source(op, ctx, 0);
            let src_lo: [u32; 8] = src[..8].try_into().unwrap();
            let src_hi: [u32; 8] = src[8..].try_into().unwrap();
            let mut result = [0u32; 16];
            result[..8].copy_from_slice(&compute(&src_lo));
            result[8..].copy_from_slice(&compute(&src_hi));
            Self::write_wide_vec_dest(op, ctx, result);
        } else {
            let src = Self::get_vector_source(op, ctx, 0);
            let result = compute(&src);
            Self::write_vector_dest(op, ctx, result);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{SlotIndex, SlotOp};
    use smallvec::smallvec;

    // ========== BFloat16 Conversion ==========

    #[test]
    fn test_bf16_to_f32_one() {
        // BF16 representation of 1.0: 0x3F80 (upper 16 bits of f32 1.0)
        assert_eq!(VectorAlu::bf16_to_f32(0x3F80), 1.0f32);
    }

    #[test]
    fn test_bf16_to_f32_negative() {
        // BF16 -1.0: 0xBF80
        assert_eq!(VectorAlu::bf16_to_f32(0xBF80), -1.0f32);
    }

    #[test]
    fn test_bf16_to_f32_zero() {
        assert_eq!(VectorAlu::bf16_to_f32(0x0000), 0.0f32);
    }

    #[test]
    fn test_bf16_to_f32_neg_zero() {
        // Negative zero: sign bit set, rest zero
        let val = VectorAlu::bf16_to_f32(0x8000);
        assert!(val.is_sign_negative());
        assert_eq!(val, -0.0f32);
    }

    #[test]
    fn test_bf16_to_f32_infinity() {
        // BF16 +inf: 0x7F80
        assert!(VectorAlu::bf16_to_f32(0x7F80).is_infinite());
        assert!(VectorAlu::bf16_to_f32(0x7F80).is_sign_positive());
    }

    #[test]
    fn test_bf16_to_f32_nan() {
        // BF16 NaN: 0x7FC0 (quiet NaN)
        assert!(VectorAlu::bf16_to_f32(0x7FC0).is_nan());
    }

    #[test]
    fn test_f32_to_bf16_truncation() {
        // f32 1.0 = 0x3F80_0000 -> bf16 = 0x3F80
        assert_eq!(VectorAlu::f32_to_bf16(1.0f32), 0x3F80);
    }

    #[test]
    fn test_bf16_roundtrip() {
        // Values that are exactly representable in bf16 should roundtrip
        for bits in [0x3F80u16, 0x4000, 0xBF80, 0x0000, 0x7F80, 0x4120] {
            let f = VectorAlu::bf16_to_f32(bits);
            let back = VectorAlu::f32_to_bf16(f);
            assert_eq!(back, bits, "roundtrip failed for 0x{:04X}", bits);
        }
    }

    #[test]
    fn test_f32_to_bf16_truncates_mantissa() {
        // f32 with low mantissa bits that get truncated (NOT rounded)
        // 1.0 + epsilon in low bits: 0x3F80_0001 -> still 0x3F80
        let val = f32::from_bits(0x3F80_0001);
        assert_eq!(VectorAlu::f32_to_bf16(val), 0x3F80);
    }

    // ========== Comparison Masks (32-bit elements) ==========

    #[test]
    fn test_condense_mask_32bit_all_set() {
        let value = [0xFFFF_FFFF; 8];
        assert_eq!(
            VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int32)),
            0xFF, // all 8 lanes set
        );
    }

    #[test]
    fn test_condense_mask_32bit_none_set() {
        let value = [0u32; 8];
        assert_eq!(VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int32)), 0x00,);
    }

    #[test]
    fn test_condense_mask_32bit_alternating() {
        // Lanes 0, 2, 4, 6 set
        let mut value = [0u32; 8];
        for i in (0..8).step_by(2) {
            value[i] = 0xFFFF_FFFF;
        }
        assert_eq!(VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int32)), 0b0101_0101,);
    }

    #[test]
    fn test_condense_mask_32bit_nonzero_means_set() {
        // Any nonzero value in a lane should count as "set"
        let value = [1, 0, 0x8000_0000, 0, 0, 0, 0, 42];
        assert_eq!(
            VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int32)),
            0b1000_0101, // lanes 0, 2, 7
        );
    }

    #[test]
    fn test_condense_mask_32bit_none_element_type() {
        // None element type should behave like 32-bit
        let value = [0xFFFF_FFFF; 8];
        assert_eq!(VectorAlu::condense_comparison_mask(&value, None), 0xFF,);
    }

    // ========== Comparison Masks (16-bit elements) ==========

    #[test]
    fn test_condense_mask_16bit_all_set() {
        let value = [0xFFFF_FFFF; 8]; // all 16 lanes set
        assert_eq!(VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int16)), 0xFFFF,);
    }

    #[test]
    fn test_condense_mask_16bit_low_only() {
        // Only low 16-bit halves set in each word
        let value = [0x0000_FFFF; 8];
        // Even-indexed lanes (0, 2, 4, ..., 14) should be set
        assert_eq!(
            VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int16)),
            0b0101_0101_0101_0101,
        );
    }

    #[test]
    fn test_condense_mask_16bit_high_only() {
        // Only high 16-bit halves set in each word
        let value = [0xFFFF_0000; 8];
        // Odd-indexed lanes (1, 3, 5, ..., 15) should be set
        assert_eq!(
            VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int16)),
            0b1010_1010_1010_1010,
        );
    }

    // ========== Comparison Masks (8-bit elements) ==========

    #[test]
    fn test_condense_mask_8bit_all_set() {
        let value = [0xFFFF_FFFF; 8]; // all 32 lanes set
        assert_eq!(VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int8)), 0xFFFF_FFFF,);
    }

    #[test]
    fn test_condense_mask_8bit_first_byte_each_word() {
        // Only byte 0 in each word is set
        let value = [0x0000_00FF; 8];
        // Lanes 0, 4, 8, 12, 16, 20, 24, 28
        assert_eq!(VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int8)), 0x1111_1111,);
    }

    // ========== pack_comparison_flags ==========

    #[test]
    fn test_pack_flags_32bit_matches_condense() {
        // pack_comparison_flags and condense_comparison_mask should agree
        // for the same inputs when element type is specified
        let value = [0xFFFF_FFFF, 0, 0xFFFF_FFFF, 0, 0, 0xFFFF_FFFF, 0, 0];
        let condense = VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int32));
        let pack = VectorAlu::pack_comparison_flags(&value, ElementType::Int32);
        assert_eq!(condense, pack);
    }

    #[test]
    fn test_pack_flags_16bit_matches_condense() {
        let value = [0xFFFF_0000, 0x0000_FFFF, 0, 0xFFFF_FFFF, 0, 0, 0, 0];
        let condense = VectorAlu::condense_comparison_mask(&value, Some(ElementType::Int16));
        let pack = VectorAlu::pack_comparison_flags(&value, ElementType::Int16);
        assert_eq!(condense, pack);
    }

    // ========== Wide Element Extract/Insert ==========

    #[test]
    fn test_extract_wide_element_32bit() {
        let mut src = [0u32; 16];
        src[0] = 0xDEAD_BEEF;
        src[5] = 0xCAFE_BABE;
        assert_eq!(VectorAlu::extract_wide_element(&src, 0, ElementType::Int32), 0xDEAD_BEEF,);
        assert_eq!(VectorAlu::extract_wide_element(&src, 5, ElementType::Int32), 0xCAFE_BABE,);
    }

    #[test]
    fn test_extract_wide_element_16bit() {
        let mut src = [0u32; 16];
        // Word 0 = 0x0002_0001 -> element 0 = 0x0001, element 1 = 0x0002
        src[0] = 0x0002_0001;
        assert_eq!(VectorAlu::extract_wide_element(&src, 0, ElementType::UInt16), 0x0001,);
        assert_eq!(VectorAlu::extract_wide_element(&src, 1, ElementType::UInt16), 0x0002,);
    }

    #[test]
    fn test_extract_wide_element_8bit_signed() {
        let mut src = [0u32; 16];
        // Word 0: bytes [0x01, 0xFF, 0x7F, 0x80]
        src[0] = 0x807F_FF01;
        // Element 0 = 0x01 (signed: 1)
        assert_eq!(VectorAlu::extract_wide_element(&src, 0, ElementType::Int8), 1,);
        // Element 1 = 0xFF (signed: -1, sign-extended to u64)
        assert_eq!(
            VectorAlu::extract_wide_element(&src, 1, ElementType::Int8) as u32,
            0xFFFF_FFFF, // -1 sign extended
        );
    }

    #[test]
    fn test_extract_wide_element_64bit() {
        let mut src = [0u32; 16];
        // Element 0 at words [0, 1]: lo=0xAAAA, hi=0xBBBB
        src[0] = 0xAAAA_AAAA;
        src[1] = 0xBBBB_BBBB;
        assert_eq!(VectorAlu::extract_wide_element(&src, 0, ElementType::Int64), 0xBBBB_BBBB_AAAA_AAAA,);
    }

    #[test]
    fn test_insert_wide_element_32bit() {
        let src = [0u32; 16];
        let result = VectorAlu::insert_wide_element(&src, 3, 0x12345678, ElementType::Int32);
        assert_eq!(result[3], 0x12345678);
        // Other words untouched
        assert_eq!(result[0], 0);
        assert_eq!(result[4], 0);
    }

    #[test]
    fn test_insert_wide_element_16bit() {
        let src = [0u32; 16];
        // Insert 0xABCD at element 1 (upper half of word 0)
        let result = VectorAlu::insert_wide_element(&src, 1, 0xABCD, ElementType::UInt16);
        assert_eq!(result[0], 0xABCD_0000);
    }

    #[test]
    fn test_insert_extract_roundtrip_32bit() {
        let src = [0u32; 16];
        for idx in 0..16u32 {
            let val = 0x1000_0000 + idx;
            let inserted = VectorAlu::insert_wide_element(&src, idx, val, ElementType::Int32);
            let extracted = VectorAlu::extract_wide_element(&inserted, idx, ElementType::Int32);
            assert_eq!(extracted as u32, val, "roundtrip failed at index {}", idx);
        }
    }

    #[test]
    fn test_insert_extract_roundtrip_16bit() {
        let src = [0u32; 16];
        for idx in 0..32u32 {
            let val = 0x1000 + idx;
            let inserted = VectorAlu::insert_wide_element(&src, idx, val, ElementType::UInt16);
            let extracted = VectorAlu::extract_wide_element(&inserted, idx, ElementType::UInt16);
            assert_eq!(extracted as u32, val, "roundtrip failed at index {}", idx);
        }
    }

    #[test]
    fn test_insert_wide_element_64bit() {
        let src = [0u32; 16];
        let result = VectorAlu::insert_wide_element_64(&src, 2, 0xAAAA_1111, 0xBBBB_2222);
        assert_eq!(result[4], 0xAAAA_1111); // element 2 -> words [4, 5]
        assert_eq!(result[5], 0xBBBB_2222);
    }

    // ========== Wide Vector Push ==========

    #[test]
    fn test_wide_push_lo_32bit() {
        // Push value at low end, shift existing toward high
        let mut src = [0u32; 16];
        src[0] = 0xAAAA_AAAA; // element 0
        let result = VectorAlu::wide_vector_push(&src, 0x42, false, ElementType::Int32);
        // New value at word 0
        assert_eq!(result[0], 0x42);
        // Old element 0 shifted to word 1
        assert_eq!(result[1], 0xAAAA_AAAA);
    }

    #[test]
    fn test_wide_push_hi_32bit() {
        // Push value at high end, shift existing toward low
        let mut src = [0u32; 16];
        src[15] = 0xBBBB_BBBB; // last word
        let result = VectorAlu::wide_vector_push(&src, 0x99, true, ElementType::Int32);
        // New value at last word
        assert_eq!(result[15], 0x99);
        // Old last word shifted to word 14
        assert_eq!(result[14], 0xBBBB_BBBB);
    }

    #[test]
    fn test_wide_push_lo_8bit() {
        // 8-bit push: shifts by 1 byte, inserts 1 byte at position 0
        let src = [0u32; 16];
        let result = VectorAlu::wide_vector_push(&src, 0xAB, false, ElementType::Int8);
        // Byte 0 of word 0 should be 0xAB
        assert_eq!(result[0] & 0xFF, 0xAB);
    }

    // ========== Wide Dispatch Bridges ==========

    #[test]
    fn test_wide_element_wise_unary() {
        // Simple identity function applied to both halves
        fn negate(v: &[u32; 8], _et: ElementType) -> [u32; 8] {
            let mut r = [0u32; 8];
            for i in 0..8 {
                r[i] = !v[i];
            }
            r
        }

        let mut src = [0u32; 16];
        src[0] = 0x0000_FFFF;
        src[8] = 0xFFFF_0000;

        let result = VectorAlu::wide_element_wise_unary(&src, ElementType::Int32, negate);
        assert_eq!(result[0], 0xFFFF_0000);
        assert_eq!(result[8], 0x0000_FFFF);
    }

    #[test]
    fn test_wide_element_wise_binary() {
        fn add_32(a: &[u32; 8], b: &[u32; 8], _et: ElementType) -> [u32; 8] {
            let mut r = [0u32; 8];
            for i in 0..8 {
                r[i] = a[i].wrapping_add(b[i]);
            }
            r
        }

        let mut a = [0u32; 16];
        let mut b = [0u32; 16];
        a[0] = 10;
        b[0] = 20;
        a[8] = 100;
        b[8] = 200;

        let result = VectorAlu::wide_element_wise_binary(&a, &b, ElementType::Int32, add_32);
        assert_eq!(result[0], 30);
        assert_eq!(result[8], 300);
    }

    // ========== Wide Vector Shift (barrel shifter) ==========

    #[test]
    fn test_wide_shift_zero() {
        // Shift by 0 with step 0 should return a (no b contribution)
        let mut a = [0u32; 16];
        a[0] = 0xDEAD_BEEF;
        let b = [0u32; 16];
        let result = VectorAlu::wide_vector_shift(&a, &b, 0, 0);
        assert_eq!(result[0], 0xDEAD_BEEF);
    }

    #[test]
    fn test_wide_shift_by_4_bytes() {
        // Shift by 4 bytes = rotate c by 4 bytes (1 word)
        let mut a = [0u32; 16];
        for i in 0..16 {
            a[i] = i as u32;
        }
        let b = [0u32; 16];
        let result = VectorAlu::wide_vector_shift(&a, &b, 0, 4);
        // After shifting a by 4 bytes: word[0] should contain what was word[1]
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
    }

    // ========== Operand Access (needs ExecutionContext) ==========

    #[test]
    fn test_read_vector_operand_vector_reg() {
        let mut ctx = ExecutionContext::new();
        let data = [1u32, 2, 3, 4, 5, 6, 7, 8];
        ctx.vector.write(3, data);
        let result = VectorAlu::read_vector_operand(&Operand::VectorReg(3), &ctx);
        assert_eq!(result, data);
    }

    #[test]
    fn test_read_vector_operand_accum_truncation() {
        let mut ctx = ExecutionContext::new();
        // Write accumulator with values that have high bits
        let acc = [0x1_DEAD_BEEFu64, 0x2_CAFE_BABEu64, 0, 0, 0, 0, 0, 0];
        ctx.accumulator.write(0, acc);
        let result = VectorAlu::read_vector_operand(&Operand::AccumReg(0), &ctx);
        // Should truncate to low 32 bits
        assert_eq!(result[0], 0xDEAD_BEEF);
        assert_eq!(result[1], 0xCAFE_BABE);
    }

    #[test]
    fn test_read_vector_operand_immediate_broadcast() {
        let result = VectorAlu::read_vector_operand(&Operand::Immediate(42), &ExecutionContext::new());
        assert_eq!(result, [42u32; 8]);
    }

    #[test]
    fn test_read_vector_operand_unexpected_returns_zeros() {
        let result = VectorAlu::read_vector_operand(&Operand::PointerReg(0), &ExecutionContext::new());
        assert_eq!(result, [0u32; 8]);
    }

    // ========== Write Vector Dest ==========

    #[test]
    fn test_write_vector_dest_to_vector_reg() {
        let mut ctx = ExecutionContext::new();
        let data = [10u32, 20, 30, 40, 50, 60, 70, 80];
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.dest = Some(Operand::VectorReg(5));
        VectorAlu::write_vector_dest(&op, &mut ctx, data);
        assert_eq!(ctx.vector.read(5), data);
    }

    #[test]
    fn test_write_vector_dest_to_accum_zero_extends() {
        let mut ctx = ExecutionContext::new();
        let data = [0xFFFF_FFFFu32, 0, 0, 0, 0, 0, 0, 0];
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.dest = Some(Operand::AccumReg(2));
        VectorAlu::write_vector_dest(&op, &mut ctx, data);
        let acc = ctx.accumulator.read(2);
        // Zero-extended: 0xFFFF_FFFF -> 0x0000_0000_FFFF_FFFF
        assert_eq!(acc[0], 0x0000_0000_FFFF_FFFF);
    }

    #[test]
    fn test_write_vector_dest_to_scalar_condenses_mask() {
        let mut ctx = ExecutionContext::new();
        // 32-bit comparison: lanes 0 and 7 true
        let data = [0xFFFF_FFFFu32, 0, 0, 0, 0, 0, 0, 0xFFFF_FFFF];
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.dest = Some(Operand::ScalarReg(16));
        op.element_type = Some(ElementType::Int32);
        VectorAlu::write_vector_dest(&op, &mut ctx, data);
        assert_eq!(ctx.scalar.read(16), 0b1000_0001);
    }

    // ========== Config / Accumulator Helpers ==========

    #[test]
    fn test_get_config_register_finds_last_scalar() {
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(5, 0xCC00_FF19);
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.sources = smallvec![Operand::VectorReg(0), Operand::VectorReg(1), Operand::ScalarReg(5),];
        assert_eq!(VectorAlu::get_config_register(&op, &ctx), Some(0xCC00_FF19));
    }

    #[test]
    fn test_get_config_register_no_scalar_returns_none() {
        let ctx = ExecutionContext::new();
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.sources = smallvec![Operand::VectorReg(0)];
        assert_eq!(VectorAlu::get_config_register(&op, &ctx), None);
    }

    #[test]
    fn test_get_acc_dest() {
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.dest = Some(Operand::AccumReg(3));
        assert_eq!(VectorAlu::get_acc_dest(&op), 3);
    }

    #[test]
    fn test_get_acc_dest_wrong_type_defaults_to_zero() {
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.dest = Some(Operand::VectorReg(5));
        assert_eq!(VectorAlu::get_acc_dest(&op), 0);
    }

    #[test]
    fn test_get_shift_amount_immediate() {
        let ctx = ExecutionContext::new();
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.sources = smallvec![Operand::Immediate(7)];
        assert_eq!(VectorAlu::get_shift_amount(&op, &ctx), 7);
    }

    #[test]
    fn test_get_shift_amount_register() {
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(3, 42);
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.sources = smallvec![Operand::ScalarReg(3)];
        assert_eq!(VectorAlu::get_shift_amount(&op, &ctx), 42);
    }

    #[test]
    fn test_get_shift_amount_no_source_defaults_zero() {
        let ctx = ExecutionContext::new();
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.sources = smallvec![Operand::VectorReg(0)];
        assert_eq!(VectorAlu::get_shift_amount(&op, &ctx), 0);
    }

    // ========== Scalar Source Helpers ==========

    #[test]
    fn test_get_nth_scalar_source() {
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(2, 100);
        ctx.scalar.write(4, 200);
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.sources = smallvec![Operand::VectorReg(0), Operand::ScalarReg(2), Operand::ScalarReg(4),];
        assert_eq!(VectorAlu::get_nth_scalar_source(&op, &ctx, 0), 100);
        assert_eq!(VectorAlu::get_nth_scalar_source(&op, &ctx, 1), 200);
    }

    #[test]
    fn test_get_scalar_source_64_register_pair() {
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(6, 0xAAAA_1111);
        ctx.scalar.write(7, 0xBBBB_2222);
        let mut op = SlotOp::nop(SlotIndex::Vector);
        op.sources = smallvec![Operand::ScalarReg(6)];
        let val = VectorAlu::get_scalar_source_64(&op, &ctx);
        assert_eq!(val as u32, 0xAAAA_1111);
        assert_eq!((val >> 32) as u32, 0xBBBB_2222);
    }
}
