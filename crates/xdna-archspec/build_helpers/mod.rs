//! Build-time toolchain extraction helpers.
//!
//! These modules extract instruction encodings, scheduling models, register
//! and definitions from llvm-aie at compile time, producing
//! a generated Rust source file (`gen_tablegen.rs`) that the main crate
//! includes via `include!()`. This eliminates the runtime LLVM dependency.

pub mod aiert;
pub mod codegen;
pub mod cpp_switch;
/// The element-type token parser, shared with the crate (one copy, tested there).
#[path = "../src/aie2/isa/element_type_logic.rs"]
pub mod element_type_logic;
pub mod extract;
pub mod records;
pub mod schema_rules;
pub mod semantics;
