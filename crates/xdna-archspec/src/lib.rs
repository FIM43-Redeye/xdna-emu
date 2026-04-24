//! NPU Architecture Specification -- validated hardware model.
//!
//! Extracts hardware architecture from the open-source NPU toolchain
//! (aie-rt, AM025 JSON, device model) into a single typed Rust model.
//! Multi-architecture: each `ArchModel` represents one architecture
//! (AIE, AIE2, AIE2P) with all its tile types, registers, and
//! relationships.
//!
//! This crate is a workspace member of xdna-emu, usable as both a
//! library dependency (for runtime queries) and a build dependency
//! (for compile-time code generation from the validated spec).
//! Runtime users import from `runtime` for `ArchConfig`/`ModelConfig`.
//! The `model_builder` module is factored out so a future `build.rs`
//! can `#[path]`-include it without pulling in `runtime`'s `Arc`/`LazyLock`.

pub mod aie2;
pub mod device_model;
pub mod dma;
pub mod elf;
pub mod isa_execute;
pub mod locks;
pub mod stream_switch;
pub mod model_builder;
pub mod regdb;
pub mod regdb_extractor;
pub mod runtime;
pub mod tablegen;
pub mod topology;
pub mod types;

pub use model_builder::{build_arch_model, confirm_subsystem_ranges};
