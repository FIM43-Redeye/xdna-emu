//! Tests for the xdna-emu register database wrapper.
//!
//! Layout derivation tests and BitField unit tests have been migrated to
//! `xdna_archspec::dma::layouts_tests` (Subsystem 3 migration). Only
//! xdna-emu-specific tests remain here:
//!
//! - Lock-value sign-extension (depends on xdna-emu wrapper fields)
//! - `load_for_device` integration (depends on `crate::config::Config`)

use super::*;

/// Helper to load the real database (skips if not available).
fn load_test_db() -> Option<RegisterDb> {
    super::load_for_device("aie2").ok()
}

#[test]
fn test_device_reg_layout_lock_value_extension() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found (set MLIR_AIE_PATH)");
        return;
    };

    let layout = DeviceRegLayout::from_regdb(db)
        .expect("Failed to build DeviceRegLayout");

    // Verify Lock_value field (data-driven from regdb, xdna-emu wrapper side)
    assert_eq!(layout.lock_value_width, 6, "Lock_value field width");
    assert_eq!(layout.lock_value_mask, 0x3F, "Lock_value field mask");
    assert_eq!(layout.lock_value_sign_bit, 5, "Lock_value sign bit");

    // Verify sign_extend_lock_value with known values
    assert_eq!(layout.sign_extend_lock_value(0), 0);
    assert_eq!(layout.sign_extend_lock_value(31), 31);    // max positive
    assert_eq!(layout.sign_extend_lock_value(0x20), -32); // min negative
    assert_eq!(layout.sign_extend_lock_value(0x3F), -1);  // all bits set
    assert_eq!(layout.sign_extend_lock_value(0xFF), -1);  // extra bits masked
}

#[test]
fn test_load_for_device_uses_config() {
    // Verify that load_for_device succeeds (requires mlir-aie install + config).
    // This tests the xdna-emu-side path resolution via crate::config::Config.
    let result = DeviceRegLayout::load_for_device("aie2");
    match result {
        Ok(layout) => {
            // If it loaded, do a quick sanity check on a well-known offset.
            assert_eq!(layout.memory_bd_base, 0x1D000,
                "Compute BD base should be 0x1D000 per AM025");
            // Verify the lock-value wrapper fields were populated.
            assert_eq!(layout.lock_value_width, 6);
        }
        Err(e) => {
            eprintln!("Skipping: load_for_device failed: {}", e);
        }
    }
}
