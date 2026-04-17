//! NPU Architecture Specification -- validated hardware model.
//!
//! Extracts hardware architecture from the open-source NPU toolchain
//! (aie-rt, AM025 JSON, device model) into a single typed Rust model.
//! Multi-architecture: each `ArchModel` represents one architecture
//! (AIE, AIE2, AIE2P) with all its tile types, registers, and
//! relationships.
//!
//! This crate is a workspace member of xdna-emu. Its own `build.rs`
//! performs all AIE2 code generation (under `src/aie2/`) and LLVM
//! MCDisassembler FFI compilation. Runtime users import from `runtime`
//! for `ArchConfig`/`ModelConfig` or from `aie2` for generated
//! const data.

pub mod device_model;
pub mod model_builder;
pub mod regdb;
pub mod regdb_extractor;
pub mod runtime;
pub mod tablegen;
pub mod types;

pub use model_builder::{build_arch_model, confirm_subsystem_ranges};
