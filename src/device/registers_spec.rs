//! AIE2 Register Address Specification
//!
//! Addresses derived from AMD AM025 (AIE-ML Register Reference).
//!
//! Most structural constants (BD base/stride, channel base/stride, lock
//! base/stride) and all bit field layouts are now derived from the register
//! database at runtime via [`super::regdb::device_reg_layout()`].
//!
//! Core module offsets, lock request constants, and data memory sizes are
//! generated at build time by `build.rs` from the same AM025 JSON. See the
//! `include!()` directives below.
//!
//! This module retains only constants with **no machine-readable source**:
//! - Address space layout (tile encoding shifts/masks)
//! - Program/data memory base addresses
//! - Tile layout row indices
//! - Helper functions (`sign_extend_7bit`)
//!
//! Reference: docs/xdna/am025-compact/

// ============================================================================
// Address Space Layout (AM020 Ch2)
// ============================================================================

/// Column shift for tile address encoding (bits 31:25)
pub const TILE_COL_SHIFT: u32 = 25;

/// Row shift for tile address encoding (bits 24:20)
pub const TILE_ROW_SHIFT: u32 = 20;

/// Offset mask for tile-local addresses (bits 19:0)
pub const TILE_OFFSET_MASK: u32 = 0xFFFFF;

/// Row bits in tile address: 5 bits
pub const TILE_ROW_BITS: u32 = 5;

/// Column bits in tile address: 7 bits
pub const TILE_COL_BITS: u32 = 7;

/// AIE data memory base in linker address space
/// ELF binaries use addresses starting at 0x00070000 for data memory.
/// This is the linker convention, not a hardware address.
pub const AIE_DATA_MEMORY_BASE: u32 = 0x0007_0000;

// ============================================================================
// Tile Layout (AM020 Ch2)
// ============================================================================

/// Shim tile row index (row 0)
pub const SHIM_TILE_ROW: u8 = 0;

/// Memory tile row index (row 1)
pub const MEM_TILE_ROW: u8 = 1;

/// First compute tile row index (rows 2-5 for NPU1)
pub const COMPUTE_TILE_ROW_START: u8 = 2;

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
    include!(concat!(env!("OUT_DIR"), "/gen_memory_lock.rs"));

    // Lock overflow/underflow status register offsets are now data-driven
    // via `regdb::device_reg_layout().memory_locks_overflow` etc.

    // DMA channel status register fields are now data-driven via
    // `regdb::device_reg_layout().memory_status` (StatusFieldLayout).
    //
    // FoT mode enum values moved to `aie2_spec.rs`
    // (FOT_DISABLED, FOT_NO_COUNTS, FOT_COUNTS_WITH_TOKENS, FOT_COUNTS_FROM_REGISTER).

    // Stream switch base/end addresses are now data-driven via
    // `regdb::device_reg_layout().memory_stream_switch` (StreamSwitchLayout).
}

// ============================================================================
// Core Module Registers (Compute Tiles)
// AM025: CORE_MODULE
// ============================================================================

pub mod core_module {
    //! Core module register addresses.
    //! Generated from AM025 JSON at build time. See build.rs.
    include!(concat!(env!("OUT_DIR"), "/gen_core_module.rs"));
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
    include!(concat!(env!("OUT_DIR"), "/gen_memtile_lock.rs"));

    // Lock overflow/underflow status register offsets are now data-driven
    // via `regdb::device_reg_layout().memtile_locks_overflow_0` etc.

    // Stream switch base/end addresses are now data-driven via
    // `regdb::device_reg_layout().memtile_stream_switch` (StreamSwitchLayout).
}

// ============================================================================
// Program Memory (Compute Tiles Only)
// ============================================================================

/// Program memory base offset
pub const PROGRAM_MEMORY_BASE: u32 = 0x20000;

/// Program memory end offset
pub const PROGRAM_MEMORY_END: u32 = 0x2FFFF;

// ============================================================================
// Data Memory
// ============================================================================

/// Data memory base offset
pub const DATA_MEMORY_BASE: u32 = 0x00000;

// Compute and mem tile data memory end offsets are generated from the device
// model JSON at build time.
include!(concat!(env!("OUT_DIR"), "/gen_data_memory.rs"));

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
