//! AIE2 Register Address Specification
//!
//! Addresses derived from AMD AM025 (AIE-ML Register Reference).
//!
//! Most structural constants (BD base/stride, channel base/stride, lock
//! base/stride) and all bit field layouts are now derived from the register
//! database at runtime via [`super::regdb::device_reg_layout()`].
//!
//! Core module offsets and lock request constants are generated at build time
//! by `build.rs` from the same AM025 JSON. Data memory sizes come from the
//! `arch` module (generated from the validated ArchModel). See the
//! `include!()` directives below.
//!
//! This module retains derived constants and helpers:
//! - Program/data memory base addresses (now derived from `arch::*`)
//! - Core data memory base (`AIE_DATA_MEMORY_BASE`, derived from cardinal direction)
//! - Helper functions (`sign_extend_7bit`)
//!
//! Address space layout (tile encoding shifts/masks) and tile row indices
//! are now in `crate::arch` (generated from ArchModel).
//!
//! Reference: docs/xdna/am025-compact/

// ============================================================================
// Core data memory base (hardware addressing)
// ============================================================================

/// AIE data memory base in the core's data address space.
///
/// This is the East cardinal direction (local memory for AIE2) base address:
/// `cardinal::EAST * MEMORY_SIZE = 7 * 0x10000 = 0x70000`.
///
/// ELF binaries place data at this address because it IS the hardware address
/// for the core's own data memory. The linker respects the hardware memory map;
/// this is NOT merely a linker convention.
///
/// Source: aie-rt `_XAie_GetTargetTileLoc()` -- `CardDir = Addr / DataMemSize`,
/// where CardDir 7 = East = local tile (for AIE2 with IsCheckerBoard=0).
pub const AIE_DATA_MEMORY_BASE: u32 =
    crate::arch::cardinal::EAST as u32 * crate::arch::compute::MEMORY_SIZE as u32;

// ============================================================================
// Memory Module Registers (Compute Tiles)
// AM025: MEMORY_MODULE
// ============================================================================

pub mod memory_module {
    //! Memory module register addresses for compute tiles.
    //! AM025 Section: MEMORY_MODULE
    //!
    //! Lock_Request constants are generated at build time from the AM025 JSON.
    //! DMA BD base/stride/words, channel base/stride, lock base/stride, and
    //! BD field extraction are all derived from the register database at runtime
    //! via [`super::super::regdb::device_reg_layout()`].

    // Lock_Request register (AM025 memory_module/lock/misc.txt)
    // 16KB address space where address encodes lock operation parameters
    pub use xdna_archspec::aie2::registers::memory::*;

    // Lock overflow/underflow status register offsets are now data-driven
    // via `regdb::device_reg_layout().memory_locks_overflow` etc.

    // DMA channel status register fields are now data-driven via
    // `regdb::device_reg_layout().memory_status` (StatusFieldLayout).
    //
    // FoT mode values: `crate::arch::fot::*` (DISABLED, NO_COUNTS, etc.)

    // Stream switch base/end addresses are now data-driven via
    // `regdb::device_reg_layout().memory_stream_switch` (StreamSwitchLayout).
}

// ============================================================================
// Core Module Registers (Compute Tiles)
// AM025: CORE_MODULE
// ============================================================================

pub mod core_module {
    //! Core module register addresses.
    //! Generated from AM025 JSON at build time; forwarded from xdna_archspec.
    pub use xdna_archspec::aie2::registers::*;
}

// ============================================================================
// Memory Tile Module Registers
// AM025: MEMORY_TILE_MODULE
// ============================================================================

pub mod mem_tile_module {
    //! Memory tile module register addresses.
    //! AM025 Section: MEMORY_TILE_MODULE
    //!
    //! Lock_Request constants are generated at build time from the AM025 JSON.
    //! DMA BD base/stride, channel base/stride, lock base/stride, and BD
    //! field extraction are all derived from the register database at runtime
    //! via [`super::super::regdb::device_reg_layout()`].

    // Lock_Request register (AM025 memory_tile_module/lock/misc.txt)
    // 64KB address space where address encodes lock operation parameters
    pub use xdna_archspec::aie2::registers::mem_tile::*;

    // Lock overflow/underflow status register offsets are now data-driven
    // via `regdb::device_reg_layout().memtile_locks_overflow_0` etc.

    // Stream switch base/end addresses are now data-driven via
    // `regdb::device_reg_layout().memtile_stream_switch` (StreamSwitchLayout).
}

// ============================================================================
// Program Memory (Compute Tiles Only)
// ============================================================================

/// Program memory base offset in host/CDO address space (derived from arch model).
pub const PROGRAM_MEMORY_BASE: u32 = crate::arch::compute::PROGRAM_MEM_HOST_OFFSET;

/// Program memory end offset (base + 64KB window - 1).
/// Note: only 16KB is implemented (PROGRAM_MEMORY_SIZE), but the address
/// window spans a full 64KB region in the tile's host address space.
pub const PROGRAM_MEMORY_END: u32 = PROGRAM_MEMORY_BASE + 0xFFFF;

// ============================================================================
// Data Memory
// ============================================================================

/// Data memory base offset in host/CDO address space (derived from arch model).
pub const DATA_MEMORY_BASE: u32 = crate::arch::DATA_MEM_HOST_OFFSET;

/// Data memory end offset for compute tile (derived from arch model).
pub const COMPUTE_DATA_MEMORY_END: u32 = crate::arch::compute::MEMORY_SIZE as u32 - 1;

/// Data memory end offset for memory tile (derived from arch model).
pub const MEM_TILE_DATA_MEMORY_END: u32 = crate::arch::memtile::MEMORY_SIZE as u32 - 1;

// ============================================================================
// Helper Functions
// ============================================================================

/// Sign-extend a 7-bit value to i8
///
/// Used for lock acquire/release values which are 7-bit signed in the BD.
#[inline]
pub const fn sign_extend_7bit(val: u32) -> i8 {
    if val & 0x40 != 0 {
        // Bit 6 is set = negative in 7-bit two's complement
        (val | 0x80) as u8 as i8
    } else {
        val as i8
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_extend_7bit() {
        // Positive values (0-63)
        assert_eq!(sign_extend_7bit(0), 0);
        assert_eq!(sign_extend_7bit(1), 1);
        assert_eq!(sign_extend_7bit(63), 63);

        // Negative values (bit 6 set)
        assert_eq!(sign_extend_7bit(0x7F), -1);  // 127 -> -1
        assert_eq!(sign_extend_7bit(0x40), -64); // 64 -> -64
        assert_eq!(sign_extend_7bit(0x41), -63); // 65 -> -63
    }

    // Structural address validation tests moved to regdb::tests
    // (test_device_reg_layout_from_regdb validates all base/stride values).
}
