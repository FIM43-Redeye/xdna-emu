//! NPU Architecture Graph -- queryable hardware model.
//!
//! Extracts hardware architecture from the open-source NPU toolchain
//! (aie-rt, AM025 JSON, device model) into a single typed Rust model.
//! Multi-architecture: each `ArchModel` represents one architecture
//! (AIE, AIE2, AIE2P) with all its tile types, registers, and
//! relationships.
//!
//! This crate is a workspace member of xdna-emu, usable as both a
//! library dependency (for runtime queries) and a build dependency
//! (for compile-time code generation from the validated graph).

pub mod device_model;
pub mod regdb;
pub mod regdb_extractor;
pub mod types;
