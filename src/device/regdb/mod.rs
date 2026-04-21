//! Register database wrapper.
//!
//! Subsystem 3 migrated the layout types (DeviceRegLayout + BdFieldLayout
//! family) into the `xdna_archspec` crate. This module now provides:
//!
//! - The xdna-emu-side wrapper struct `DeviceRegLayout` that extends the
//!   archspec `DeviceRegLayout` with lock-value-width fields and the
//!   `sign_extend_lock_value` helper (Subsystem 4 migrates these to
//!   `LockModel`).
//! - The `OnceLock` runtime cache + `device_reg_layout()` accessor.
//! - The `load_for_device()` loader that uses xdna-emu's `Config` for
//!   path resolution to the `mlir-aie/lib/Dialect/AIE/Util/aie_registers_*.json`
//!   files.

#[cfg(test)]
mod tests;

// Re-export the base types from the archspec crate, and re-export the
// migrated layouts so every `crate::device::regdb::BdFieldLayout` etc.
// keeps working without a mass-import-rewrite.
pub use xdna_archspec::regdb::*;
pub use xdna_archspec::dma::field_layouts::{
    BdFieldLayout, ChannelFieldLayout, StatusFieldLayout,
    MemTileBdFieldLayout, ShimBdFieldLayout,
    ShimMuxField, ShimMuxLayout,
    StreamSwitchLayout, ModuleEventLayout,
};
pub use xdna_archspec::dma::DeviceRegLayout as ArchDeviceRegLayout;

/// xdna-emu-side extension of `xdna_archspec::dma::DeviceRegLayout`.
///
/// Wraps the archspec layout and adds the lock-value-width fields
/// that live on the emulator side until Subsystem 4 migrates them to
/// `LockModel`.
#[derive(Debug, Clone)]
pub struct DeviceRegLayout {
    /// The archspec-side layout data (BD layouts, channel bases, etc.).
    ///
    /// Exposed publicly because Subsystem 4's `LockModel` migration will
    /// need to reach into this field when moving the lock-value-width
    /// metadata out of the xdna-emu wrapper.
    pub arch: ArchDeviceRegLayout,
    /// Width of the Lock_value field in bits (6 for AIE2).
    ///
    /// Subsystem 4 migrates this and the two fields below to `LockModel`.
    pub lock_value_width: u8,
    /// Mask for the Lock_value field: `(1 << width) - 1`.
    pub lock_value_mask: u32,
    /// Sign bit position within the Lock_value field: `width - 1`.
    pub lock_value_sign_bit: u8,
}

impl DeviceRegLayout {
    /// Extract and sign-extend a lock value from a raw register word.
    ///
    /// Uses the Lock_value field width from the register database (6 bits
    /// for AIE2). Subsystem 4 migrates this helper and its backing fields
    /// to `LockModel`; this signature is preserved for now so callers
    /// don't churn twice.
    #[inline]
    pub fn sign_extend_lock_value(&self, raw: u32) -> i8 {
        let masked = (raw & self.lock_value_mask) as u8;
        if masked & (1 << self.lock_value_sign_bit) != 0 {
            masked as i8 | !(self.lock_value_mask as i8)
        } else {
            masked as i8
        }
    }

    /// Build from a register database, resolving all field layouts.
    pub fn from_regdb(db: RegisterDb) -> Result<Self, String> {
        // Derive lock-value metadata (stays on xdna-emu side).
        let lock_value_field = db.module("memory")
            .and_then(|m| m.register("Lock0_value"))
            .and_then(|r| r.field("Lock_value"))
            .ok_or_else(|| "memory.Lock0_value.Lock_value field not found".to_string())?;
        let lock_value_width = lock_value_field.width;
        let lock_value_mask = lock_value_field.mask;
        let lock_value_sign_bit = lock_value_width - 1;

        // Delegate the bulk of the work to archspec.
        let arch = ArchDeviceRegLayout::from_regdb(db)?;

        Ok(Self { arch, lock_value_width, lock_value_mask, lock_value_sign_bit })
    }

    /// Load from the mlir-aie install for a given device.
    pub fn load_for_device(device: &str) -> Result<Self, String> {
        let db = load_for_device(device)?;
        Self::from_regdb(db)
    }
}

// Preserve source-level API compatibility: every existing call site like
// `device_reg_layout().memory_bd_base` compiled before this migration.
// Deref makes `layout.memory_bd_base` continue to work.
impl std::ops::Deref for DeviceRegLayout {
    type Target = ArchDeviceRegLayout;
    fn deref(&self) -> &Self::Target { &self.arch }
}

/// Load a register database from the mlir-aie install, using the
/// emulator's config system for path resolution.
pub fn load_for_device(device: &str) -> Result<RegisterDb, String> {
    let config = crate::config::Config::get();
    let json_path = config.mlir_aie_subpath(
        &format!("lib/Dialect/AIE/Util/aie_registers_{}.json", device)
    );
    RegisterDb::from_file(&json_path)
}

// ============================================================================
// Global accessor with lazy initialization (JSON required)
// ============================================================================

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
        DeviceRegLayout::load_for_device("aie2").unwrap_or_else(|e| {
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
