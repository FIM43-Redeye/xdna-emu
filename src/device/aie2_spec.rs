//! AIE2 (AIE-ML) Architecture Specification Constants
//!
//! These values are derived from AMD documentation:
//! - AM020: Versal AI Engine ML (AIE-ML) Architecture Manual
//! - AM025: AIE-ML Register Reference
//!
//! References are noted as (AM020 ChN) or (AM025 Section).
//!
//! # Data-Driven vs. Hardcoded
//!
//! Many constants that were previously here have been replaced by data-driven
//! sources. See:
//! - **mlir-aie device model** (`model.rs`): data memory sizes, lock counts,
//!   BD counts, DMA channel counts, array dimensions
//! - **regdb** (`regdb.rs`): register addresses, BD field layouts, structural
//!   base/stride values
//! - **TableGen** (`tablegen/`): register file sizes, instruction latencies
//!   (via `latency.rs` with itinerary cross-validation)
//!
//! This module retains only constants with **no machine-readable source**:
//! AM020/AM025-only values for memory architecture, lock timing, DMA timing,
//! stream switch configuration, packet format, and routing latencies.

// ============================================================================
// Memory Architecture (AM020 Ch2, Ch4)
// ============================================================================

/// Program memory size per compute tile: 16 KB
/// "The program memory size on the AIE-ML is 16 KB, which allows storing
/// 1024 instructions of 128-bit each." (AM020 Ch4)
pub const PROGRAM_MEMORY_SIZE: usize = 16 * 1024;

/// Number of memory banks per compute tile: 8
/// "The AIE-ML data memory is 64 KB, organized as eight memory banks" (AM020 Ch2)
pub const COMPUTE_TILE_MEMORY_BANKS: usize = 8;

/// Size of each memory bank in compute tile: 8 KB (64 KB / 8 banks)
pub const COMPUTE_TILE_BANK_SIZE: usize = 8 * 1024;

/// Memory bank width: 128 bits (16 bytes)
/// "Each bank is 128 bits wide" (AM020 Ch5)
pub const MEMORY_BANK_WIDTH_BITS: usize = 128;

/// Memory bank width in bytes (128 bits = 16 bytes).
pub const MEMORY_BANK_WIDTH_BYTES: usize = MEMORY_BANK_WIDTH_BITS / 8;

/// Number of memory banks per MemTile: 16
/// 512KB / 32KB per bank = 16 banks (event IDs 0-15 in MemTileEvent)
pub const MEMTILE_MEMORY_BANKS: usize = 16;

/// Size of each memory bank in MemTile: 32 KB (512 KB / 16 banks)
pub const MEMTILE_BANK_SIZE: usize = 32 * 1024;

/// Compute the bank index for a local memory address.
///
/// AIE2 uses interleaved banking at 128-bit (16-byte) boundaries.
/// Consecutive 16-byte lines map to different banks, enabling parallel
/// access to sequential addresses from separate load/store units.
///
/// `addr` is the byte offset within tile data memory.
/// `num_banks` is 8 for compute tiles, 16 for MemTiles.
#[inline]
pub fn addr_to_bank(addr: u32, num_banks: usize) -> u8 {
    ((addr as usize >> 4) % num_banks) as u8
}

/// Compute a bitmask of all banks touched by a memory access.
///
/// A 32-byte (256-bit) vector access spans two 128-bit bank rows and may
/// touch two different banks. This function returns a u16 bitmask with one
/// bit set per bank touched.
#[inline]
pub fn banks_for_access(addr: u32, bytes: usize, num_banks: usize) -> u16 {
    if bytes == 0 {
        return 0;
    }
    let mut mask = 0u16;
    // Align down to bank row boundary
    let start = (addr & !0xF) as usize;
    let end = (addr as usize) + bytes;
    let mut a = start;
    while a < end {
        let bank = (a >> 4) % num_banks;
        mask |= 1 << bank;
        a += 16;
    }
    mask
}

// ============================================================================
// Lock Architecture (AM020 Ch2, Appendix A)
// ============================================================================

// LOCK_MAX_VALUE moved to Lock::MAX_VALUE (tile.rs), validated against model.

// ============================================================================
// Lock Timing (AM020 Ch2, AM025 Lock_Request register)
// ============================================================================

/// Lock acquire latency (uncontested): 1 cycle
/// The lock module can handle a new request every clock cycle (AM020 Ch2).
pub const LOCK_ACQUIRE_LATENCY: u8 = 1;

/// Lock release latency: 1 cycle
/// Release operations are non-blocking and complete in 1 cycle.
pub const LOCK_RELEASE_LATENCY: u8 = 1;

/// Lock contention retry interval: 1 cycle
/// When a lock acquire fails, the core stalls and retries each cycle.
pub const LOCK_RETRY_INTERVAL: u8 = 1;

// ============================================================================
// Instruction Latencies (AM020 Ch4)
//
// Most instruction latencies live in interpreter/timing/latency.rs, which
// cross-validates against TableGen itinerary data (AIE2Schedule.td). These
// two constants are used by modules that don't depend on the interpreter.
// ============================================================================

/// Data memory access latency: 5 cycles
/// "Load and store units manage the 5-cycle latency of data memory."
/// This is the memory pipeline depth (= ProcessorModel.LoadLatency).
pub const LATENCY_DATA_MEMORY: u8 = 5;

/// Branch penalty: cycles lost when a branch is taken
/// Based on AM020 Ch4, the fetch pipeline has ~3 stages that are discarded.
pub const BRANCH_PENALTY_CYCLES: u8 = 3;

// ============================================================================
// DMA Timing (AM020 Ch2, derived from architecture)
// ============================================================================

/// DMA BD setup latency: cycles to parse and configure a buffer descriptor
pub const DMA_BD_SETUP_CYCLES: u8 = 4;

/// DMA channel start latency: cycles from start trigger to first data
pub const DMA_CHANNEL_START_CYCLES: u8 = 2;

/// DMA words per cycle (throughput): 1 word (32-bit) per cycle per channel
pub const DMA_WORDS_PER_CYCLE: u8 = 1;

/// DMA memory access latency: uses same as data memory (5 cycles)
pub const DMA_MEMORY_LATENCY_CYCLES: u8 = 5;

/// DMA lock acquire latency: cycles to check and acquire a lock
pub const DMA_LOCK_ACQUIRE_CYCLES: u8 = 1;

/// DMA lock release latency: cycles to release a lock
pub const DMA_LOCK_RELEASE_CYCLES: u8 = 1;

/// DMA BD chain latency: cycles between finishing one BD and starting next
pub const DMA_BD_CHAIN_CYCLES: u8 = 2;

// ============================================================================
// Stream Switch Configuration (AM020 Ch2)
// ============================================================================

/// Stream FIFO depths for local ports.
/// "Local slave ports are 2-cycle latency and a 4-deep FIFO" (AM020 Ch2)
pub const STREAM_LOCAL_SLAVE_FIFO_DEPTH: u8 = 4;

/// "Local master ports have 1-cycle latency and a 2-deep FIFO" (AM020 Ch2)
pub const STREAM_LOCAL_MASTER_FIFO_DEPTH: u8 = 2;

/// Local slave to local master: 3-cycle latency
pub const STREAM_LOCAL_TO_LOCAL_LATENCY: u8 = 3;

/// Local slave to external master: 4-cycle latency
pub const STREAM_LOCAL_TO_EXTERNAL_LATENCY: u8 = 4;

/// External to external: 4-cycle latency
pub const STREAM_EXTERNAL_TO_EXTERNAL_LATENCY: u8 = 4;

// ============================================================================
// Packet Header Format (AM020 Ch2, Table 2)
// ============================================================================
//
// 32-bit packet header layout:
// | 31    | 30-28 | 27-21      | 20-16     | 15  | 14-12       | 11-5    | 4-0       |
// | Parity| Rsvd  | Src Column | Src Row   | Rsvd| Packet Type | Rsvd    | Stream ID |

/// Stream ID field: bits 4-0
pub const PACKET_STREAM_ID_MASK: u32 = 0x1F;

/// Packet type field: bits 14-12
pub const PACKET_TYPE_SHIFT: usize = 12;
pub const PACKET_TYPE_MASK: u32 = 0x7;

/// Source row field: bits 20-16
pub const PACKET_SRC_ROW_SHIFT: usize = 16;
pub const PACKET_SRC_ROW_MASK: u32 = 0x1F;

/// Source column field: bits 27-21
pub const PACKET_SRC_COL_SHIFT: usize = 21;
pub const PACKET_SRC_COL_MASK: u32 = 0x7F;

/// Parity bit: bit 31
pub const PACKET_PARITY_SHIFT: usize = 31;

/// Packet switch arbitration overhead: cycles per packet header
pub const PACKET_ARBITRATION_OVERHEAD_CYCLES: u8 = 1;

// ============================================================================
// Stream Routing Latency (AM020 Ch2)
// ============================================================================

/// Routing latency: local slave to local master (3 cycles)
pub const ROUTE_LATENCY_LOCAL_TO_LOCAL: u8 = 3;

/// Routing latency: local slave to external master (4 cycles)
pub const ROUTE_LATENCY_LOCAL_TO_EXTERNAL: u8 = 4;

/// Routing latency: external slave to local master (3 cycles)
pub const ROUTE_LATENCY_EXTERNAL_TO_LOCAL: u8 = 3;

/// Routing latency: external slave to external master (4 cycles)
pub const ROUTE_LATENCY_EXTERNAL_TO_EXTERNAL: u8 = 4;

/// Minimum routing latency per hop between tiles
pub const ROUTE_LATENCY_PER_HOP: u8 = ROUTE_LATENCY_EXTERNAL_TO_EXTERNAL;

// ============================================================================
// Control Packet Header Format (AM020 Table 3)
// ============================================================================
//
// Control packets are delivered via the TileControl stream master port
// to reprogram tile registers at runtime. The header word encodes the
// target register address, operation, and beat count.

/// Control packet header: Address field mask (bits 19:0)
pub const CTRL_PKT_ADDRESS_MASK: u32 = 0x000F_FFFF;

/// Control packet header: Length field shift (bits 21:20, value+1 = beats)
pub const CTRL_PKT_LENGTH_SHIFT: usize = 20;
/// Control packet header: Length field mask (2 bits)
pub const CTRL_PKT_LENGTH_MASK: u32 = 0x3;

/// Control packet header: Operation field shift (bits 23:22)
pub const CTRL_PKT_OPERATION_SHIFT: usize = 22;
/// Control packet header: Operation field mask (2 bits)
pub const CTRL_PKT_OPERATION_MASK: u32 = 0x3;

/// Control packet header: Response_ID field shift (bits 30:24)
pub const CTRL_PKT_RESPONSE_ID_SHIFT: usize = 24;
/// Control packet header: Response_ID field mask (7 bits)
pub const CTRL_PKT_RESPONSE_ID_MASK: u32 = 0x7F;

/// Control packet header: Parity bit position (bit 31)
pub const CTRL_PKT_PARITY_BIT: usize = 31;

/// Control packet operation: write data to register(s)
pub const CTRL_PKT_OP_WRITE: u8 = 0;
/// Control packet operation: read register (requires response routing)
pub const CTRL_PKT_OP_READ: u8 = 1;
/// Control packet operation: write with auto-increment address
pub const CTRL_PKT_OP_WRITE_INCR: u8 = 2;
/// Control packet operation: block write (same as write for our purposes)
pub const CTRL_PKT_OP_BLOCK_WRITE: u8 = 3;

// ============================================================================
// DMA FoT (Finish-on-TLAST) Mode Values (AM025 DMA_S2MM_x_Ctrl.FoT_Mode)
// ============================================================================

/// FoT disabled: channel runs until BD transfer count is exhausted
pub const FOT_DISABLED: u8 = 0;
/// FoT no counts: transfer finishes on TLAST regardless of count
pub const FOT_NO_COUNTS: u8 = 1;
/// FoT counts with tokens: finish on TLAST, issue task-complete token
pub const FOT_COUNTS_WITH_TOKENS: u8 = 2;
/// FoT counts from register: length comes from a separate count register
pub const FOT_COUNTS_FROM_REGISTER: u8 = 3;

// ============================================================================
// Stream Switch Port Layouts (AM025)
// ============================================================================
//
// These define the port types at each index for each tile type.
// The hardware has fixed port assignments documented in AM025.

/// Stream switch port type identifier.
pub mod port_type {
    pub const CORE: u8 = 0;
    pub const FIFO: u8 = 1;
    pub const TRACE: u8 = 2;
    pub const NORTH_BASE: u8 = 10;
    pub const SOUTH_BASE: u8 = 20;
    pub const EAST_BASE: u8 = 30;
    pub const WEST_BASE: u8 = 40;
    pub const DMA_BASE: u8 = 50;

    pub const fn north(n: u8) -> u8 {
        NORTH_BASE + n
    }
    pub const fn south(n: u8) -> u8 {
        SOUTH_BASE + n
    }
    pub const fn east(n: u8) -> u8 {
        EAST_BASE + n
    }
    pub const fn west(n: u8) -> u8 {
        WEST_BASE + n
    }
    pub const fn dma(n: u8) -> u8 {
        DMA_BASE + n
    }
}

// Port type arrays are generated at build time from AM025 Stream_Switch_*_Config
// register names, sorted by offset. See build.rs for the generation logic.
include!(concat!(env!("OUT_DIR"), "/gen_stream_ports.rs"));

// ============================================================================
// Stream Switch Port Ranges (derived from AM025 port layout arrays above)
// ============================================================================

pub mod stream_switch {
    // Port range constants and ENABLE_BIT are generated at build time by
    // scanning the generated port arrays above. SLAVE_SELECT_MASK stays
    // hardcoded (sub-field not individually specified in AM025 JSON).
    include!(concat!(env!("OUT_DIR"), "/gen_stream_ranges.rs"));
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_max_value() {
        // Lock max value is now defined in Lock::MAX_VALUE (tile.rs)
        // and validated against the mlir-aie model at test time.
        assert_eq!(crate::device::tile::Lock::MAX_VALUE, 63);
    }

    #[test]
    fn test_memory_bank_size() {
        assert_eq!(COMPUTE_TILE_BANK_SIZE, 8 * 1024);
    }

    #[test]
    fn test_memtile_bank_size() {
        assert_eq!(MEMTILE_BANK_SIZE, 32 * 1024);
        assert_eq!(MEMTILE_MEMORY_BANKS * MEMTILE_BANK_SIZE, 512 * 1024);
    }

    #[test]
    fn test_addr_to_bank_interleaved() {
        // Consecutive 16-byte rows map to consecutive banks
        assert_eq!(addr_to_bank(0x00, 8), 0);
        assert_eq!(addr_to_bank(0x10, 8), 1);
        assert_eq!(addr_to_bank(0x20, 8), 2);
        assert_eq!(addr_to_bank(0x70, 8), 7);
        // Wraps around after 8 banks
        assert_eq!(addr_to_bank(0x80, 8), 0);
        assert_eq!(addr_to_bank(0x90, 8), 1);
    }

    #[test]
    fn test_addr_to_bank_within_row() {
        // All bytes within a 16-byte bank row map to the same bank
        for offset in 0..16 {
            assert_eq!(addr_to_bank(0x30 + offset, 8), 3);
        }
    }

    #[test]
    fn test_addr_to_bank_memtile_16_banks() {
        assert_eq!(addr_to_bank(0x00, 16), 0);
        assert_eq!(addr_to_bank(0xF0, 16), 15);
        assert_eq!(addr_to_bank(0x100, 16), 0); // wraps at 16
    }

    #[test]
    fn test_banks_for_access_scalar() {
        // 4-byte scalar load within one bank row -> one bank
        let mask = banks_for_access(0x00, 4, 8);
        assert_eq!(mask, 0b0000_0001); // bank 0

        let mask = banks_for_access(0x14, 4, 8);
        assert_eq!(mask, 0b0000_0010); // bank 1
    }

    #[test]
    fn test_banks_for_access_vector() {
        // 32-byte vector access at 0x00: spans banks 0 and 1
        let mask = banks_for_access(0x00, 32, 8);
        assert_eq!(mask, 0b0000_0011); // banks 0,1

        // 32-byte vector access at 0x70: spans banks 7 and 0 (wraps)
        let mask = banks_for_access(0x70, 32, 8);
        assert_eq!(mask, 0b1000_0001); // banks 7,0
    }

    #[test]
    fn test_banks_for_access_zero_bytes() {
        assert_eq!(banks_for_access(0x00, 0, 8), 0);
    }
}
