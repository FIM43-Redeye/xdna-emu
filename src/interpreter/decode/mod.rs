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
pub mod ports;
#[cfg(test)]
mod ports_goldens;
#[cfg(test)]
mod scalar_ports_goldens;
mod slot_builder;
pub(crate) mod composite;
pub mod crossref;
pub mod probe;
pub mod register_map;

pub use decoder::{DecodedInstr, InstructionDecoder};
