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
mod loader;
mod operand_extraction;
mod slot_builder;
pub(crate) mod composite;
pub mod crossref;

pub use decoder::{DecodedInstr, InstructionDecoder};

use crate::interpreter::bundle::SlotIndex;

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
            Aie2Slot::Lda => SlotIndex::LoadA,
            Aie2Slot::Ldb => SlotIndex::LoadB,
            Aie2Slot::Alu => SlotIndex::Scalar0,
            Aie2Slot::Mv => SlotIndex::Scalar1,
            Aie2Slot::St => SlotIndex::Store,
            Aie2Slot::Vec => SlotIndex::Vector,
            // LNG is polymorphic: j/jl -> Control, movxm -> Scalar0.
            // decode_bundle() resolves via slot_op.natural_slot() after decoding.
            Aie2Slot::Lng => SlotIndex::Control,
            Aie2Slot::Nop => SlotIndex::Control,
        }
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
}
