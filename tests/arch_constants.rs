//! Tests for the generated `arch` constants module.
//!
//! These verify that build.rs produces correct constants from the ArchModel.
//! Values are known from the device model JSON and AM025 register database.
//!
//! Note: bank counts come from mlir-aie's getNumBanks(), which reports memory
//! groups (4 for compute, 8 for memtile). AM020 documents physical 128-bit
//! bank rows (8 for compute, 16 for memtile). Both are correct at different
//! abstraction levels. The `banking` module uses the AM020 physical count for
//! bank conflict detection; the `arch` module exposes the device model values.

use xdna_emu::arch;

// ============================================================================
// Memory topology (from device model)
// ============================================================================

#[test]
fn compute_tile_memory() {
    // 4 logical banks * 16KB = 64KB total (mlir-aie memory groups)
    assert_eq!(arch::compute::MEMORY_SIZE, 64 * 1024);
    assert_eq!(arch::compute::LOGICAL_BANKS, 4);
    assert_eq!(arch::compute::LOGICAL_BANK_SIZE, 16 * 1024);
    assert_eq!(arch::compute::PROGRAM_MEMORY_SIZE, 16 * 1024);
}

#[test]
fn memtile_memory() {
    // 8 logical banks * 64KB = 512KB total (mlir-aie memory groups)
    assert_eq!(arch::memtile::MEMORY_SIZE, 512 * 1024);
    assert_eq!(arch::memtile::LOGICAL_BANKS, 8);
    assert_eq!(arch::memtile::LOGICAL_BANK_SIZE, 64 * 1024);
}

// ============================================================================
// Resource counts (from device model, cross-validated with regdb)
// ============================================================================

#[test]
fn compute_tile_resources() {
    assert_eq!(arch::compute::NUM_LOCKS, 16);
    assert_eq!(arch::compute::NUM_BDS, 16);
    assert_eq!(arch::compute::NUM_DMA_CHANNELS, 2);
}

#[test]
fn memtile_resources() {
    assert_eq!(arch::memtile::NUM_LOCKS, 64);
    assert_eq!(arch::memtile::NUM_BDS, 48);
    assert_eq!(arch::memtile::NUM_DMA_CHANNELS, 6);
}

#[test]
fn shim_resources() {
    assert_eq!(arch::shim::NUM_LOCKS, 16);
    assert_eq!(arch::shim::NUM_BDS, 16);
    assert_eq!(arch::shim::NUM_DMA_CHANNELS, 2);
}

// ============================================================================
// Device-level constants
// ============================================================================

#[test]
fn device_constants() {
    assert_eq!(arch::MAX_LOCK_VALUE, 63);
    assert_eq!(arch::MIN_LOCK_VALUE, -64);
}

// ============================================================================
// Array topology
// ============================================================================

#[test]
fn array_dimensions() {
    // Device model reports 4 columns for npu1
    assert_eq!(arch::COLUMNS, 4);
    assert_eq!(arch::ROWS, 6);
}

// ============================================================================
// Core address space (from aie-rt AieMlCoreMod)
// ============================================================================

#[test]
fn core_address_map() {
    // AIE2 core data address space starts at 0x40000 (aie-rt DataMemAddr)
    assert_eq!(arch::compute::DATA_MEM_ADDR, 0x40000);
    // 64KB per quadrant = shift 16
    assert_eq!(arch::compute::DATA_MEM_SHIFT, 16);
    // AIE2 is NOT checkerboard (East is always local)
    assert_eq!(arch::compute::IS_CHECKERBOARD, false);
    // Program memory at 0x20000 in host/CDO view
    assert_eq!(arch::compute::PROGRAM_MEM_HOST_OFFSET, 0x20000);
}

#[test]
fn cardinal_directions() {
    // Cardinal direction = address / MEMORY_SIZE
    // Matches aie-rt _XAie_GetTargetTileLoc()
    assert_eq!(arch::cardinal::SOUTH, 4);
    assert_eq!(arch::cardinal::WEST, 5);
    assert_eq!(arch::cardinal::NORTH, 6);
    assert_eq!(arch::cardinal::EAST, 7);

    // East * MEMORY_SIZE = 0x70000 (the "AIE_DATA_MEMORY_BASE" value
    // that was previously mislabeled as a linker convention)
    assert_eq!(
        arch::cardinal::EAST as u32 * arch::compute::MEMORY_SIZE as u32,
        0x70000,
    );
}

#[test]
fn data_mem_host_offset() {
    // Data memory is always at offset 0 in the tile's host/CDO view
    assert_eq!(arch::DATA_MEM_HOST_OFFSET, 0);
}
