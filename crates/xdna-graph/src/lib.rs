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

use std::path::Path;

/// Build a fully-populated ArchModel for a named device.
///
/// This is the primary entry point for build.rs. It:
/// 1. Extracts device topology from the device model JSON
/// 2. Enriches with register data from the AM025 register database
/// 3. Cross-validates via Confirmed<T> (panics on conflicts)
///
/// The result is the validated architecture graph, ready for code generation.
pub fn build_arch_model(
    device_model_path: &Path,
    regdb: &regdb::RegisterDb,
    device_name: &str,
) -> Result<types::ArchModel, String> {
    let mut model = device_model::extract_device_model(device_model_path, device_name)
        .map_err(|e| format!("Device model extraction failed: {}", e))?;
    regdb_extractor::populate_tile_modules(&mut model, regdb);
    Ok(model)
}
