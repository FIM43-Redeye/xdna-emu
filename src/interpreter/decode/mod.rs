//! Instruction decoder for AIE2.
//!
//! This module provides instruction decoding from raw bytes to [`VliwBundle`]
//! structures. The decoder uses TableGen definitions from llvm-aie for
//! accurate, O(1) instruction lookup.
//!
//! # AIE2 Instruction Format
//!
//! AIE2 uses a VLIW architecture with 8 functional slots:
//!
//! | Slot | Bits | Purpose |
//! |------|------|---------|
//! | lda | 21 | Load A channel |
//! | ldb | 16 | Load B channel |
//! | alu | 20 | Scalar ALU |
//! | mv | 22 | Move operations |
//! | st | 21 | Store |
//! | vec | 26 | Vector operations |
//! | lng | 42 | Long (48-bit) format |
//! | nop | 1 | NOP marker (artificial) |
//!
//! Instructions come in bundles of varying sizes:
//! - 2 bytes: NOP
//! - 4 bytes: Single slot operation
//! - 6 bytes: Long slot operation
//! - 8+ bytes: Multi-slot VLIW bundles
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::interpreter::decode::InstructionDecoder;
//! use xdna_emu::interpreter::traits::Decoder;
//!
//! let decoder = InstructionDecoder::load_default();
//! let bytes = &program_memory[pc..];
//! let bundle = decoder.decode(bytes, pc)?;
//! ```

mod decoder;

pub use decoder::{DecodedInstr, InstructionDecoder};

use crate::interpreter::bundle::{
    Operand, Operation, SlotIndex, SlotOp,
};

/// AIE2 slot types based on TableGen definitions.
///
/// These correspond to the functional units in the AIE2 pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aie2Slot {
    /// Load A channel (21 bits).
    Lda,
    /// Load B channel (16 bits).
    Ldb,
    /// Scalar ALU (20 bits).
    Alu,
    /// Move operations (22 bits).
    Mv,
    /// Store (21 bits).
    St,
    /// Vector operations (26 bits).
    Vec,
    /// Long format (42 bits, 6 bytes).
    Lng,
    /// NOP marker (artificial, 1 bit).
    Nop,
}

impl Aie2Slot {
    /// Get the bit width for this slot's encoding.
    pub fn bit_width(self) -> u8 {
        match self {
            Aie2Slot::Lda => 21,
            Aie2Slot::Ldb => 16,
            Aie2Slot::Alu => 20,
            Aie2Slot::Mv => 22,
            Aie2Slot::St => 21,
            Aie2Slot::Vec => 26,
            Aie2Slot::Lng => 42,
            Aie2Slot::Nop => 1,
        }
    }

    /// Map to the interpreter's SlotIndex.
    pub fn to_slot_index(self) -> SlotIndex {
        match self {
            Aie2Slot::Lda => SlotIndex::Load,
            Aie2Slot::Ldb => SlotIndex::Load,
            Aie2Slot::Alu => SlotIndex::Scalar0,
            Aie2Slot::Mv => SlotIndex::Scalar1,
            Aie2Slot::St => SlotIndex::Store,
            Aie2Slot::Vec => SlotIndex::Vector,
            Aie2Slot::Lng => SlotIndex::Vector, // Long operations are often vector
            Aie2Slot::Nop => SlotIndex::Control,
        }
    }
}

/// Common opcode patterns for quick classification.
///
/// These are derived from observing real instruction streams and
/// will be refined with TableGen data.
pub mod opcodes {
    /// Zero word - simplest NOP.
    pub const NOP_ZERO: u32 = 0x0000_0000;

    /// AIE canonical NOP pattern.
    pub const NOP_AIE: u32 = 0x1501_0040;

    /// High nibble patterns for instruction classification.
    pub mod high_nibble {
        /// Branch/call patterns.
        pub const BRANCH: u32 = 0x0;
        pub const CALL: u32 = 0x1;
        /// Load patterns.
        pub const LOAD: u32 = 0x2;
        /// Store patterns.
        pub const STORE: u32 = 0x3;
        /// Arithmetic patterns.
        pub const ARITH_0: u32 = 0x4;
        pub const ARITH_1: u32 = 0x5;
        pub const ARITH_2: u32 = 0x6;
        pub const ARITH_3: u32 = 0x7;
        /// Move patterns.
        pub const MOVE: u32 = 0x8;
        /// Lock patterns.
        pub const LOCK: u32 = 0x9;
        /// DMA patterns.
        pub const DMA: u32 = 0xA;
        /// Vector patterns.
        pub const VEC_0: u32 = 0xB;
        pub const VEC_1: u32 = 0xC;
        pub const VEC_2: u32 = 0xD;
        pub const VEC_3: u32 = 0xE;
        pub const VEC_4: u32 = 0xF;
    }
}

/// Result of decoding a single slot from an instruction word.
#[derive(Debug, Clone)]
pub struct SlotDecode {
    /// Which AIE2 slot this belongs to.
    pub slot: Aie2Slot,
    /// The decoded operation.
    pub op: Operation,
    /// Source operands.
    pub sources: Vec<Operand>,
    /// Destination operand.
    pub dest: Option<Operand>,
}

impl SlotDecode {
    /// Convert to a SlotOp for the bundle.
    pub fn to_slot_op(self) -> SlotOp {
        let mut op = SlotOp::new(self.slot.to_slot_index(), self.op);
        op.dest = self.dest;
        for src in self.sources {
            op.sources.push(src);
        }
        op
    }
}

/// Extract register fields from an instruction word.
#[inline]
pub fn extract_reg(word: u32, shift: u8, mask: u32) -> u8 {
    ((word >> shift) & mask) as u8
}

/// Extract an immediate value from an instruction word.
#[inline]
pub fn extract_imm(word: u32, shift: u8, bits: u8) -> i32 {
    let mask = (1u32 << bits) - 1;
    let raw = (word >> shift) & mask;

    // Sign extend if high bit is set
    if bits < 32 && (raw & (1 << (bits - 1))) != 0 {
        (raw | !mask) as i32
    } else {
        raw as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aie2_slot_bit_widths() {
        assert_eq!(Aie2Slot::Lda.bit_width(), 21);
        assert_eq!(Aie2Slot::Ldb.bit_width(), 16);
        assert_eq!(Aie2Slot::Alu.bit_width(), 20);
        assert_eq!(Aie2Slot::Vec.bit_width(), 26);
        assert_eq!(Aie2Slot::Lng.bit_width(), 42);
    }

    #[test]
    fn test_extract_reg() {
        let word = 0x12345678;
        assert_eq!(extract_reg(word, 0, 0xF), 0x8);
        assert_eq!(extract_reg(word, 4, 0xF), 0x7);
        assert_eq!(extract_reg(word, 8, 0xF), 0x6);
    }

    #[test]
    fn test_extract_imm_positive() {
        // 0x7FF with 12 bits = 2047 (positive, sign bit not set)
        let word = 0x000007FF;
        assert_eq!(extract_imm(word, 0, 12), 2047);
    }

    #[test]
    fn test_extract_imm_signed() {
        // 0xFFF with 12 bits = -1 when sign extended
        let word = 0x00000FFF;
        assert_eq!(extract_imm(word, 0, 12), -1);

        // 0x7FF with 12 bits = 2047 (positive)
        let word = 0x000007FF;
        assert_eq!(extract_imm(word, 0, 12), 2047);
    }
}
