//! Derived memory-map constants for AIE2.
//!
//! These consts are computed from values in other `aie2` modules:
//! `compute::MEMORY_SIZE`, `compute::PROGRAM_MEM_HOST_OFFSET`,
//! `DATA_MEM_HOST_OFFSET`, `memtile::MEMORY_SIZE`, and `cardinal::EAST`.
//! They live here rather than in consumer crates so all AIE2 memory-map
//! derivations have a single home.

use super::{cardinal, compute, memtile, DATA_MEM_HOST_OFFSET};

/// AIE data memory base in the core's data address space.
///
/// This is the East cardinal direction (local memory for AIE2) base
/// address: `cardinal::EAST * MEMORY_SIZE = 7 * 0x10000 = 0x70000`.
///
/// ELF binaries place data at this address because it IS the hardware
/// address for the core's own data memory. The linker respects the
/// hardware memory map; this is NOT merely a linker convention.
///
/// Source: aie-rt `_XAie_GetTargetTileLoc()` --
/// `CardDir = Addr / DataMemSize`, where CardDir 7 = East = local tile
/// (for AIE2 with IsCheckerBoard=0).
pub const AIE_DATA_MEMORY_BASE: u32 =
    cardinal::EAST as u32 * compute::MEMORY_SIZE as u32;

/// Program memory base offset in host/CDO address space.
///
/// Source: AM025 CORE_MODULE_PROGRAM_MEMORY + aie-rt
/// XAIEMLGBL_CORE_MODULE_PROGRAM_MEMORY.
pub const PROGRAM_MEMORY_BASE: u32 = compute::PROGRAM_MEM_HOST_OFFSET;

/// Program memory end offset.
///
/// Note: only 16 KB is implemented (`compute::PROGRAM_MEMORY_SIZE`),
/// but the address window spans a full 64 KB region in the tile's host
/// address space.
pub const PROGRAM_MEMORY_END: u32 = PROGRAM_MEMORY_BASE + 0xFFFF;

/// Data memory base offset in host/CDO address space (always 0).
pub const DATA_MEMORY_BASE: u32 = DATA_MEM_HOST_OFFSET;

/// Data memory end offset for compute tile.
pub const COMPUTE_DATA_MEMORY_END: u32 = compute::MEMORY_SIZE as u32 - 1;

/// Data memory end offset for memory tile.
pub const MEM_TILE_DATA_MEMORY_END: u32 = memtile::MEMORY_SIZE as u32 - 1;
