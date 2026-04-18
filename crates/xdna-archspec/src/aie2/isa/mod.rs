//! AIE2 instruction set architecture: decoder tables, runtime model,
//! and LLVM MCDisassembler FFI.
//!
//! All content here is build-time extracted from llvm-aie (TableGen
//! sources + LLVM libraries). The runtime consumes generated constants
//! via `load_from_generated()`.
//!
//! Submodules populate across Subsystem 6's Part A relocation:
//! - `types` (arch-agnostic instruction/register/operand types)
//! - `resolver` (operand classification + semantic inference)
//! - `decoder_bytecode` (bytecode walker for instruction decode)
//! - `decoder_ffi` (LLVM MCDisassembler FFI, raw side only)
//! - `element_type_logic` (shared build+runtime element-type inference)
//!
//! The interpreter-aware half of the old `decoder_ffi.rs` (MappedOperand,
//! RegisterMap, classify_reg_name) lives in xdna-emu's
//! `interpreter::decode::register_map`, not here.
