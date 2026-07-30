//! Tests for the xdna-emu register database accessor.
//!
//! The `DeviceRegLayout` struct itself lives in `xdna_archspec::dma`.
//! Only xdna-emu-specific integration with `crate::config::Config`
//! is tested here.

use super::*;

#[test]
fn test_load_for_device_uses_shared_toolchain_discovery() {
    let layout = load_for_device("aie2").expect("required AM025 register database was not resolved");

    assert_eq!(layout.memory_bd_base, 0x1D000, "Compute BD base should be 0x1D000 per AM025");
}
