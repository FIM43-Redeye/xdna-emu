//! AIE2 instruction set architecture: decoder tables, runtime model,
//! and LLVM MCDisassembler FFI.
//!
//! Content here is build-time extracted from llvm-aie (TableGen sources
//! + LLVM libraries). The runtime consumes generated constants via
//! `load_from_generated()` (added in Task 7).
//!
//! The interpreter-aware half of the old `decoder_ffi.rs` (MappedOperand,
//! RegisterMap, classify_reg_name) lives in xdna-emu's `tablegen::register_map`
//! (relocates to `interpreter::decode::register_map` in Part B), not here.

pub mod types;
pub mod resolver;
pub mod decoder_bytecode;
pub mod decoder_ffi;
pub mod element_type_logic;

// Re-exports matching xdna-emu's src/tablegen/mod.rs surface so consumers
// can `use xdna_archspec::aie2::isa::*;` to get the full tablegen API.
pub use types::*;
pub use resolver::{
    build_decoder_tables, AddressingMode, CompositeEncoder, DecoderIndex, InstrEncoding,
    InstrMemWidth, OperandField, OperandType, RegisterKind, ResolveError, Resolver, SlotIndex,
    classify_operand_type, detect_addressing_mode, detect_mem_width,
    infer_branch_condition, infer_dual_element_types, infer_element_type, infer_select_variant,
    refine_branch_semantic,
};
