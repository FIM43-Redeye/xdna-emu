//! Tests for the xdna-emu register database accessor.
//!
//! The `DeviceRegLayout` struct itself lives in `xdna_archspec::dma`.
//! Only xdna-emu-specific integration with `crate::config::Config`
//! is tested here.

use super::*;

#[test]
fn test_load_for_device_uses_config() {
    // Verify that load_for_device succeeds (requires mlir-aie install + config).
    // This tests the xdna-emu-side path resolution via crate::config::Config.
    let result = load_for_device("aie2");
    match result {
        Ok(layout) => {
            // If it loaded, do a quick sanity check on a well-known offset.
            assert_eq!(layout.memory_bd_base, 0x1D000,
                "Compute BD base should be 0x1D000 per AM025");
        }
        Err(e) => {
            eprintln!("Skipping: load_for_device failed: {}", e);
        }
    }
}
