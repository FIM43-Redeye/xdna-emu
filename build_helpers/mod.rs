//! Build-time TableGen extraction helpers.
//!
//! These modules extract instruction encodings, scheduling models, register
//! definitions, and decoder bytecode from llvm-aie at compile time, producing
//! a generated Rust source file (`gen_tablegen.rs`) that the main crate
//! includes via `include!()`. This eliminates the runtime LLVM dependency.

pub mod bytecode;
pub mod codegen;
pub mod cpp_switch;
pub mod extract;
pub mod records;
pub mod semantics;
