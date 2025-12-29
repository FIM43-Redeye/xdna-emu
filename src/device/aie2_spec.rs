//! AIE2 (AIE-ML) Architecture Specification Constants
//!
//! These values are derived from AMD documentation:
//! - AM020: Versal AI Engine ML (AIE-ML) Architecture Manual
//! - AM025: AIE-ML Register Reference
//!
//! References are noted as (AM020 ChN) or (AM025 Section).

// ============================================================================
// Memory Architecture (AM020 Ch2, Ch4)
// ============================================================================

/// Program memory size per compute tile: 16 KB
/// "The program memory size on the AIE-ML is 16 KB, which allows storing
/// 1024 instructions of 128-bit each." (AM020 Ch4)
pub const PROGRAM_MEMORY_SIZE: usize = 16 * 1024;

/// Data memory size per compute tile: 64 KB
/// "An individual data memory block is 64 KB." (AM020 Ch4)
pub const COMPUTE_TILE_DATA_MEMORY_SIZE: usize = 64 * 1024;

/// Data memory size per memory tile: 512 KB
/// "Each AIE-ML memory tile has 512 KB of memory" (AM020 Ch5)
pub const MEM_TILE_DATA_MEMORY_SIZE: usize = 512 * 1024;

/// Number of memory banks per compute tile: 8
/// "The AIE-ML data memory is 64 KB, organized as eight memory banks" (AM020 Ch2)
pub const COMPUTE_TILE_MEMORY_BANKS: usize = 8;

/// Size of each memory bank in compute tile: 8 KB
/// 64 KB / 8 banks = 8 KB per bank
pub const COMPUTE_TILE_BANK_SIZE: usize = COMPUTE_TILE_DATA_MEMORY_SIZE / COMPUTE_TILE_MEMORY_BANKS;

/// Number of memory banks per memory tile: 16
/// "Each AIE-ML memory tile has 512 KB of memory as 16 banks of 32 KB" (AM020 Ch5)
pub const MEM_TILE_MEMORY_BANKS: usize = 16;

/// Size of each memory bank in memory tile: 32 KB
pub const MEM_TILE_BANK_SIZE: usize = MEM_TILE_DATA_MEMORY_SIZE / MEM_TILE_MEMORY_BANKS;

/// Memory bank width: 128 bits
/// "Each bank is 128 bits wide" (AM020 Ch5)
pub const MEMORY_BANK_WIDTH_BITS: usize = 128;

/// Addressable memory per core (4 neighboring tiles): 256 KB
/// "The AIE-ML accesses four 64 KB data memory blocks to create a 256 KB unit" (AM020 Ch4)
pub const ADDRESSABLE_MEMORY_SIZE: usize = 4 * COMPUTE_TILE_DATA_MEMORY_SIZE;

// ============================================================================
// Lock Architecture (AM020 Ch2, Appendix A)
// ============================================================================

/// Number of locks per compute tile: 16
/// "The AIE-ML features 16 semaphore locks" (AM020 Ch2)
pub const COMPUTE_TILE_NUM_LOCKS: usize = 16;

/// Number of locks per memory tile: 64
/// "there are 64 semaphore locks" (AM020 Ch5, Appendix A)
pub const MEM_TILE_NUM_LOCKS: usize = 64;

/// Lock state width: 6 bits (values 0-63)
/// "each lock state is 6-bit unsigned" (AM020 Ch2)
pub const LOCK_STATE_BITS: usize = 6;

/// Maximum lock value: 63
pub const LOCK_MAX_VALUE: u8 = (1 << LOCK_STATE_BITS) - 1;

// ============================================================================
// Lock Timing (AM020 Ch2, AM025 Lock_Request register)
// ============================================================================

/// Lock acquire latency (uncontested): 1 cycle
/// The lock module can handle a new request every clock cycle (AM020 Ch2).
/// Similar to memory bank arbitration which is round-robin.
pub const LOCK_ACQUIRE_LATENCY: u8 = 1;

/// Lock release latency: 1 cycle
/// Release operations are non-blocking and complete in 1 cycle.
pub const LOCK_RELEASE_LATENCY: u8 = 1;

/// Lock contention retry interval: 1 cycle
/// When a lock acquire fails, the core stalls and retries each cycle.
/// "Examples include: ... lock modules" in stall sources (AM020 Ch2).
pub const LOCK_RETRY_INTERVAL: u8 = 1;

/// Lock value change bits in Lock_Request register: 7 bits
/// "Change_Value [8:2]" (AM025) - allows delta from -64 to +63
pub const LOCK_CHANGE_VALUE_BITS: usize = 7;

// ============================================================================
// Register Architecture (AM020 Ch4, Table 8-11)
// ============================================================================

/// Number of scalar general-purpose registers (r0-r31): 32 x 32-bit
pub const NUM_SCALAR_GPR: usize = 32;
pub const SCALAR_GPR_WIDTH_BITS: usize = 32;

/// Number of pointer registers (p0-p7): 8 x 20-bit
pub const NUM_POINTER_REGS: usize = 8;
pub const POINTER_REG_WIDTH_BITS: usize = 20;

/// Number of modifier registers (m0-m7): 8 x 20-bit
pub const NUM_MODIFIER_REGS: usize = 8;
pub const MODIFIER_REG_WIDTH_BITS: usize = 20;

/// Number of dimension registers (dn0-dn7, dj0-dj7, dc0-dc7): 8 each x 20-bit
pub const NUM_DIMENSION_REGS: usize = 8;
pub const DIMENSION_REG_WIDTH_BITS: usize = 20;

/// Number of shift control registers (s0-s3): 4 x 6-bit
pub const NUM_SHIFT_CONTROL_REGS: usize = 4;
pub const SHIFT_CONTROL_REG_WIDTH_BITS: usize = 6;

/// Number of W (vector) registers: 24 x 256-bit
/// "There are 24 x 256-bit registers: wln and whn, n=0..11" (AM020 Ch4)
pub const NUM_VECTOR_W_REGS: usize = 24;
pub const VECTOR_W_REG_WIDTH_BITS: usize = 256;

/// Number of X (512-bit) registers: 12 (pairs of W)
/// "Two W registers can be grouped to form a 512-bit register prefixed with X"
pub const NUM_VECTOR_X_REGS: usize = 12;
pub const VECTOR_X_REG_WIDTH_BITS: usize = 512;

/// Number of Y (1024-bit) registers: 6 (pairs of X)
pub const NUM_VECTOR_Y_REGS: usize = 6;
pub const VECTOR_Y_REG_WIDTH_BITS: usize = 1024;

/// Number of mask registers (Q0-Q3): 4 x 128-bit (for sparsity)
/// "there are 4 x 128-bit mask registers (Q0 to Q3)" (AM020 Ch4)
pub const NUM_MASK_REGS: usize = 4;
pub const MASK_REG_WIDTH_BITS: usize = 128;

/// Number of am (256-bit) accumulator registers: 8
/// "256 bit wide, they can be viewed as eight lanes of 32-bit data" (AM020 Ch4)
pub const NUM_ACCUMULATOR_AM_REGS: usize = 8;
pub const ACCUMULATOR_AM_WIDTH_BITS: usize = 256;

/// Number of bm (512-bit) accumulator registers: 4 (pairs of am)
pub const NUM_ACCUMULATOR_BM_REGS: usize = 4;
pub const ACCUMULATOR_BM_WIDTH_BITS: usize = 512;

/// Number of cm (1024-bit) accumulator registers: 2 (pairs of bm)
pub const NUM_ACCUMULATOR_CM_REGS: usize = 2;
pub const ACCUMULATOR_CM_WIDTH_BITS: usize = 1024;

// ============================================================================
// Instruction Latencies (AM020 Ch4)
// ============================================================================

/// Scalar add/subtract latency: 1 cycle
/// "Integer addition and subtraction: 32 bits. The operation has a one cycle latency."
pub const LATENCY_SCALAR_ADD_SUB: u8 = 1;

/// Scalar logical operations (BAND, BOR, BXOR) latency: 1 cycle
pub const LATENCY_SCALAR_LOGIC: u8 = 1;

/// Scalar shift latency: 1 cycle
/// "The shift amount is passed through a general purpose register...one-cycle latency."
pub const LATENCY_SCALAR_SHIFT: u8 = 1;

/// Scalar compare latency: 1 cycle
pub const LATENCY_SCALAR_COMPARE: u8 = 1;

/// Scalar multiply latency: 2 cycles
/// "Integer multiplication: 32 x 32 bit...The operation has a two cycle latency."
pub const LATENCY_SCALAR_MUL: u8 = 2;

/// Data memory access latency: 5 cycles
/// "Load and store units manage the 5-cycle latency of data memory."
pub const LATENCY_DATA_MEMORY: u8 = 5;

/// Address generation unit (AGU) latency: 1 cycle
/// "The AGU has a one cycle latency."
pub const LATENCY_AGU: u8 = 1;

/// Maximum pipeline depth: 8 stages
/// "Different pipeline on each functional unit (eight stages maximum)."
pub const MAX_PIPELINE_STAGES: u8 = 8;

/// Branch penalty: cycles lost when a branch is taken
/// When a branch is taken, the fetch pipeline must be flushed.
/// Based on AM020 Ch4, the fetch unit has ~3 stages that are discarded.
/// This is a conservative estimate for taken branches.
pub const BRANCH_PENALTY_CYCLES: u8 = 3;

// ============================================================================
// Stream Switch Latencies (AM020 Ch2)
// ============================================================================

/// External port latency: 2 cycles, 4-deep FIFO
/// "External ports are 2-cycle latency and a 4-deep FIFO"
pub const STREAM_EXTERNAL_PORT_LATENCY: u8 = 2;
pub const STREAM_EXTERNAL_PORT_FIFO_DEPTH: u8 = 4;

/// Local slave port latency: 2 cycles, 4-deep FIFO
/// "Local slave ports are 2-cycle latency and a 4-deep FIFO"
pub const STREAM_LOCAL_SLAVE_LATENCY: u8 = 2;
pub const STREAM_LOCAL_SLAVE_FIFO_DEPTH: u8 = 4;

/// Local master port latency: 1 cycle, 2-deep FIFO
/// "Local master ports have one register slice with 1-cycle latency and a 2-deep FIFO"
pub const STREAM_LOCAL_MASTER_LATENCY: u8 = 1;
pub const STREAM_LOCAL_MASTER_FIFO_DEPTH: u8 = 2;

/// Local slave to local master: 3-cycle latency, 6-deep FIFO
pub const STREAM_LOCAL_TO_LOCAL_LATENCY: u8 = 3;
pub const STREAM_LOCAL_TO_LOCAL_FIFO_DEPTH: u8 = 6;

/// Local slave to external master: 4-cycle latency, 8-deep FIFO
pub const STREAM_LOCAL_TO_EXTERNAL_LATENCY: u8 = 4;
pub const STREAM_LOCAL_TO_EXTERNAL_FIFO_DEPTH: u8 = 8;

/// External slave to local master: 3-cycle latency, 6-deep FIFO
pub const STREAM_EXTERNAL_TO_LOCAL_LATENCY: u8 = 3;
pub const STREAM_EXTERNAL_TO_LOCAL_FIFO_DEPTH: u8 = 6;

/// External to external: 4-cycle latency, 8-deep FIFO
pub const STREAM_EXTERNAL_TO_EXTERNAL_LATENCY: u8 = 4;
pub const STREAM_EXTERNAL_TO_EXTERNAL_FIFO_DEPTH: u8 = 8;

// ============================================================================
// DMA Configuration (AM020 Ch2)
// ============================================================================

/// Number of DMA buffer descriptors per tile: 16
/// "the DMA controller has access to the 16 buffer descriptors" (AM020 Ch2)
pub const NUM_DMA_BUFFER_DESCRIPTORS: usize = 16;

/// Number of S2MM DMA channels per compute tile: 2
pub const COMPUTE_TILE_S2MM_CHANNELS: usize = 2;

/// Number of MM2S DMA channels per compute tile: 2
pub const COMPUTE_TILE_MM2S_CHANNELS: usize = 2;

/// Number of S2MM DMA channels per memory tile: 6
/// "Memory to stream DMA (MM2S) with six channels" (AM020 Appendix A)
pub const MEM_TILE_S2MM_CHANNELS: usize = 6;

/// Number of MM2S DMA channels per memory tile: 6
pub const MEM_TILE_MM2S_CHANNELS: usize = 6;

/// DMA data width: 32 bits
/// "two incoming and two outgoing streams...32-bit data" (AM020 Ch2)
pub const DMA_DATA_WIDTH_BITS: usize = 32;

// ============================================================================
// DMA Timing (AM020 Ch2, derived from architecture)
// ============================================================================

/// DMA BD setup latency: cycles to parse and configure a buffer descriptor
/// Estimated based on BD register reads and internal state machine setup
pub const DMA_BD_SETUP_CYCLES: u8 = 4;

/// DMA channel start latency: cycles from start trigger to first data
/// Includes arbitration and initial memory request
pub const DMA_CHANNEL_START_CYCLES: u8 = 2;

/// DMA words per cycle (throughput): 1 word (32-bit) per cycle per channel
/// At 1 GHz, this gives 4 GB/s per channel
pub const DMA_WORDS_PER_CYCLE: u8 = 1;

/// DMA memory access latency: uses same as data memory (5 cycles)
/// "Load and store units manage the 5-cycle latency" (AM020 Ch4)
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

/// Stream switch data width: 32 bits
/// "a fully programmable, 32-bit, AXI4-Stream crossbar" (AM020 Ch2)
pub const STREAM_SWITCH_DATA_WIDTH_BITS: usize = 32;

/// Number of slave ports: 25
/// "There are 25 slave ports and 23 master ports"
pub const STREAM_SWITCH_SLAVE_PORTS: usize = 25;

/// Number of master ports: 23
pub const STREAM_SWITCH_MASTER_PORTS: usize = 23;

/// Stream switch FIFO depth: 16 entries, 34 bits wide
/// "The switch has one FIFO that is 16-deep and 34 bit"
pub const STREAM_SWITCH_FIFO_DEPTH: usize = 16;
pub const STREAM_SWITCH_FIFO_WIDTH_BITS: usize = 34; // 32 data + 1 parity + 1 TLAST

/// Packet header stream ID width: 5 bits
/// "A packet-switched stream is identified by a 5-bit ID"
pub const PACKET_STREAM_ID_BITS: usize = 5;

// ============================================================================
// Packet Header Format (AM020 Ch2, Table 2)
// ============================================================================
//
// 32-bit packet header layout:
// | 31    | 30-28 | 27-21      | 20-16     | 15  | 14-12       | 11-5    | 4-0       |
// | Parity| Rsvd  | Src Column | Src Row   | Rsvd| Packet Type | Rsvd    | Stream ID |
// | 1 bit | 3 bits| 7 bits     | 5 bits    | 1bit| 3 bits      | 7 bits  | 5 bits    |
//

/// Packet header total width: 32 bits
pub const PACKET_HEADER_WIDTH_BITS: usize = 32;

/// Stream ID field: bits 4-0
pub const PACKET_STREAM_ID_SHIFT: usize = 0;
pub const PACKET_STREAM_ID_MASK: u32 = 0x1F; // 5 bits

/// Packet type field: bits 14-12
pub const PACKET_TYPE_SHIFT: usize = 12;
pub const PACKET_TYPE_MASK: u32 = 0x7; // 3 bits

/// Source row field: bits 20-16
pub const PACKET_SRC_ROW_SHIFT: usize = 16;
pub const PACKET_SRC_ROW_MASK: u32 = 0x1F; // 5 bits

/// Source column field: bits 27-21
pub const PACKET_SRC_COL_SHIFT: usize = 21;
pub const PACKET_SRC_COL_MASK: u32 = 0x7F; // 7 bits

/// Parity bit: bit 31
pub const PACKET_PARITY_SHIFT: usize = 31;
pub const PACKET_PARITY_MASK: u32 = 0x1;

/// Packet switch arbitration overhead: cycles per packet header
/// This is the additional latency for packet routing vs circuit routing.
/// Estimated based on header parsing and route lookup.
pub const PACKET_ARBITRATION_OVERHEAD_CYCLES: u8 = 1;

/// Maximum number of packet routes per tile
/// Each stream ID can map to multiple destinations.
pub const MAX_PACKET_ROUTES_PER_TILE: usize = 32;

// ============================================================================
// Stream Routing Latency (AM020 Ch2)
// ============================================================================
//
// Stream switch routing latency depends on the path type:
// - Local = within the same tile (core port, DMA port)
// - External = to neighboring tile (north/south/east/west)
//
// The latency is due to FIFO buffering at each switch:
// - 6-deep FIFO: used for local-to-local and external-to-local
// - 8-deep FIFO: used for local-to-external and external-to-external
//

/// Routing latency: local slave to local master (3 cycles)
/// "local slave → local master: 3 cycles (6-deep FIFO)" (AM020 Ch2)
pub const ROUTE_LATENCY_LOCAL_TO_LOCAL: u8 = 3;

/// Routing latency: local slave to external master (4 cycles)
/// "local slave → external master: 4 cycles (8-deep FIFO)" (AM020 Ch2)
pub const ROUTE_LATENCY_LOCAL_TO_EXTERNAL: u8 = 4;

/// Routing latency: external slave to local master (3 cycles)
/// "external slave → local master: 3 cycles (6-deep FIFO)" (AM020 Ch2)
pub const ROUTE_LATENCY_EXTERNAL_TO_LOCAL: u8 = 3;

/// Routing latency: external slave to external master (4 cycles)
/// "external → external: 4 cycles (8-deep FIFO)" (AM020 Ch2)
pub const ROUTE_LATENCY_EXTERNAL_TO_EXTERNAL: u8 = 4;

/// Minimum routing latency per hop between tiles
/// This is the base latency for crossing a tile boundary.
pub const ROUTE_LATENCY_PER_HOP: u8 = ROUTE_LATENCY_EXTERNAL_TO_EXTERNAL;

// ============================================================================
// Cascade Interface (AM020 Ch2, Ch4)
// ============================================================================

/// Cascade stream width: 512 bits
/// "The 512-bit accumulator data from one AIE-ML can be forwarded to another"
pub const CASCADE_STREAM_WIDTH_BITS: usize = 512;

/// Cascade FIFO depth: 2 entries (4 values total)
/// "a small two-deep 512-bit wide FIFO...allow storing up to four values"
pub const CASCADE_FIFO_DEPTH: usize = 2;

// ============================================================================
// Clock and Performance (AM020 Ch1)
// ============================================================================

/// Target AIE-ML clock frequency for -1L speed grade: 1 GHz
/// "The AIE-ML FMAX is 1 GHz for the -1L speed grade devices"
pub const TARGET_CLOCK_FREQ_HZ: u64 = 1_000_000_000;

/// AIE-ML array clock domain
pub const AIE_ML_CLOCK_DOMAIN: &str = "AIE-ML";

/// PL interface clock (half of AIE clock): 500 MHz
pub const PL_CLOCK_FREQ_HZ: u64 = 500_000_000;

// ============================================================================
// Vector Unit (AM020 Ch4, Table 7)
// ============================================================================

/// Vector register width: 256 bits
pub const VECTOR_REG_WIDTH_BITS: usize = 256;

/// Accumulator lanes for 32-bit mode: 32
pub const ACCUMULATOR_LANES_32BIT: usize = 32;

/// Accumulator lanes for 64-bit mode: 16
pub const ACCUMULATOR_LANES_64BIT: usize = 16;

/// BFloat16 MAC operations per cycle: 128
/// "Supports 128 bfloat 16 MAC operations with FP32 accumulation"
pub const BF16_MACS_PER_CYCLE: usize = 128;

/// Int8 multipliers: 256
/// "The number of int8 multipliers is 256"
pub const INT8_MULTIPLIERS: usize = 256;

// ============================================================================
// Memory Access and Alignment (AM020 Ch4)
// ============================================================================

/// Vector load/store alignment requirement: 32 bytes (256 bits)
/// "Two 256-bit load and one 256-bit store units with aligned addresses" (AM020 Ch4)
pub const VECTOR_ALIGNMENT_BYTES: usize = 32;

/// Word alignment: 4 bytes
pub const WORD_ALIGNMENT_BYTES: usize = 4;

/// Halfword alignment: 2 bytes
pub const HALFWORD_ALIGNMENT_BYTES: usize = 2;

/// Memory bank conflict penalty: 1 cycle stall
/// "When there are multiple requests in the same cycle to the same memory bank,
/// only one request per cycle is allowed...stalled for one cycle" (AM020 Ch2)
pub const BANK_CONFLICT_PENALTY_CYCLES: u8 = 1;

/// Bank interleaving: physical banks are paired into logical banks
/// "From a programmer's perspective, every two banks are interleaved to form one bank" (AM020 Ch2)
pub const BANKS_PER_LOGICAL_BANK: usize = 2;

/// Number of logical banks (programmer's view): 4
pub const LOGICAL_BANKS: usize = COMPUTE_TILE_MEMORY_BANKS / BANKS_PER_LOGICAL_BANK;

/// Logical bank size: 16 KB
pub const LOGICAL_BANK_SIZE: usize = COMPUTE_TILE_BANK_SIZE * BANKS_PER_LOGICAL_BANK;

// ============================================================================
// Address Space (AM020 Ch2, Ch4)
// ============================================================================

/// Tile address space: 20 bits (1 MB)
/// "The lower 20 bits represent the tile address range"
pub const TILE_ADDRESS_BITS: usize = 20;

/// Row bits in tile address: 5 bits
/// "followed by five bits that represent the row location"
pub const TILE_ROW_BITS: usize = 5;

/// Column bits in tile address: 7 bits
/// "and seven bits that represent the column location"
pub const TILE_COLUMN_BITS: usize = 7;

/// Data memory address range: 0x0000 to 0x3FFFF (256 KB)
/// "The AGU generates addresses for data memory access that span from
/// 0x0000 to 0x3FFFF (256 KB)."
pub const DATA_MEMORY_ADDRESS_MAX: u32 = 0x3FFFF;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_bank_sizes() {
        assert_eq!(COMPUTE_TILE_BANK_SIZE, 8 * 1024); // 8 KB
        assert_eq!(MEM_TILE_BANK_SIZE, 32 * 1024); // 32 KB
    }

    #[test]
    fn test_lock_max_value() {
        assert_eq!(LOCK_MAX_VALUE, 63);
    }

    #[test]
    fn test_vector_register_hierarchy() {
        // W -> X -> Y grouping
        assert_eq!(NUM_VECTOR_X_REGS, NUM_VECTOR_W_REGS / 2);
        assert_eq!(NUM_VECTOR_Y_REGS, NUM_VECTOR_X_REGS / 2);
    }

    #[test]
    fn test_accumulator_hierarchy() {
        // am -> bm -> cm grouping
        assert_eq!(NUM_ACCUMULATOR_BM_REGS, NUM_ACCUMULATOR_AM_REGS / 2);
        assert_eq!(NUM_ACCUMULATOR_CM_REGS, NUM_ACCUMULATOR_BM_REGS / 2);
    }

    #[test]
    fn test_stream_latencies() {
        // Local-to-local = slave + master latency
        assert_eq!(
            STREAM_LOCAL_TO_LOCAL_LATENCY,
            STREAM_LOCAL_SLAVE_LATENCY + STREAM_LOCAL_MASTER_LATENCY
        );
    }

    #[test]
    fn test_addressable_memory() {
        // 4 x 64KB = 256KB
        assert_eq!(ADDRESSABLE_MEMORY_SIZE, 256 * 1024);
    }
}
