//! Register database accessor.
//!
//! After Subsystem 4, DeviceRegLayout lives entirely in xdna_archspec.
//! This module retains only:
//!
//! - Re-exports of the archspec types (so every
//!   `crate::device::regdb::DeviceRegLayout` etc. keeps working).
//! - The `OnceLock`-backed global accessor `device_reg_layout()`.
//! - The config-aware `load_for_device()` loader (xdna-emu owns
//!   `Config` for path resolution).

#[cfg(test)]
mod tests;

pub use xdna_archspec::regdb::*;
pub use xdna_archspec::dma::DeviceRegLayout;
pub use xdna_archspec::dma::field_layouts::{
    BdFieldLayout, ChannelFieldLayout, StatusFieldLayout,
    MemTileBdFieldLayout, ShimBdFieldLayout,
    ShimMuxField, ShimMuxLayout,
    StreamSwitchLayout, ModuleEventLayout,
};

use std::sync::OnceLock;

static DEVICE_REG_LAYOUT: OnceLock<DeviceRegLayout> = OnceLock::new();

/// Get the global register layout, loading from JSON on first access.
///
/// # Panics
///
/// Panics if the register database JSON file cannot be loaded. This
/// requires mlir-aie to be installed and `MLIR_AIE_PATH` configured.
pub fn device_reg_layout() -> &'static DeviceRegLayout {
    DEVICE_REG_LAYOUT.get_or_init(|| {
        load_for_device("aie2").unwrap_or_else(|e| {
            panic!(
                "Failed to load register database: {}.\n\
                 The register database JSON (aie_registers_aie2.json) is required.\n\
                 Ensure mlir-aie is installed and MLIR_AIE_PATH is set.\n\
                 See CLAUDE.md for environment setup instructions.",
                e
            )
        })
    })
}

/// Load a DeviceRegLayout from the mlir-aie install using the emulator's
/// `Config` for path resolution.
pub fn load_for_device(device: &str) -> Result<DeviceRegLayout, String> {
    let config = crate::config::Config::get();
    let json_path = config.mlir_aie_subpath(
        &format!("lib/Dialect/AIE/Util/aie_registers_{}.json", device)
    );
    let db = RegisterDb::from_file(&json_path)?;
    DeviceRegLayout::from_regdb(db)
}
