//! Register database accessor.
//!
//! After Subsystem 4, DeviceRegLayout lives entirely in xdna_archspec.
//! This module retains only:
//!
//! - Re-exports of the archspec types (so every
//!   `crate::device::regdb::DeviceRegLayout` etc. keeps working).
//! - The `OnceLock`-backed global accessor `device_reg_layout()`.
//! - The shared-toolchain-aware `load_for_device()` loader.

#[cfg(test)]
mod tests;

pub use xdna_archspec::regdb::*;
pub use xdna_archspec::dma::DeviceRegLayout;
pub use xdna_archspec::dma::field_layouts::{
    BdFieldLayout, ChannelFieldLayout, StatusFieldLayout, MemTileBdFieldLayout, ShimBdFieldLayout,
    ShimMuxField, ShimMuxLayout, StreamSwitchLayout, ModuleEventLayout,
};

use std::path::Path;
use std::sync::OnceLock;

use xdna_archspec::toolchain_paths::ToolchainPaths;

static DEVICE_REG_LAYOUT: OnceLock<DeviceRegLayout> = OnceLock::new();

/// Get the global register layout, loading from JSON on first access.
///
/// # Panics
///
/// Panics if the register database JSON file cannot be loaded. This
/// requires mlir-aie to be installed and discoverable.
pub fn device_reg_layout() -> &'static DeviceRegLayout {
    DEVICE_REG_LAYOUT.get_or_init(|| {
        load_for_device("aie2").unwrap_or_else(|e| {
            panic!(
                "Failed to load register database: {}.\n\
                 The register database JSON (aie_registers_aie2.json) is required.\n\
                 Ensure mlir-aie is installed or configure its root explicitly.\n\
                 See AGENTS.md for environment setup instructions.",
                e
            )
        })
    })
}

/// Load a DeviceRegLayout from the resolved mlir-aie tree.
pub fn load_for_device(device: &str) -> Result<DeviceRegLayout, String> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mlir_aie = ToolchainPaths::require_mlir_aie(workspace_root)?;
    let json_path = mlir_aie.join(format!("lib/Dialect/AIE/Util/aie_registers_{}.json", device));
    let db = RegisterDb::from_file(&json_path)?;
    DeviceRegLayout::from_regdb(db)
}
