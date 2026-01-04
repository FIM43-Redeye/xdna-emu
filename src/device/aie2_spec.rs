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

/// Number of DMA buffer descriptors per compute tile: 16
/// "the DMA controller has access to the 16 buffer descriptors" (AM020 Ch2)
pub const NUM_DMA_BUFFER_DESCRIPTORS: usize = 16;

/// Number of DMA buffer descriptors per memory tile: 48
/// MemTile has 24 S2MM BDs + 24 MM2S BDs (AM025 memory_tile_module/dma/bd.txt)
pub const MEMTILE_NUM_DMA_BUFFER_DESCRIPTORS: usize = 48;

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

// ============================================================================
// Stream Switch Port Layouts (AM025)
// ============================================================================
//
// These layouts define the port types at each index for each tile type.
// The hardware has fixed port assignments documented in AM025 register reference.
// AIE2P (Strix/Strix Halo) may have different layouts.

/// Stream switch port type identifier.
///
/// These map to the PortType enum in stream_switch.rs but are simple u8s
/// for use in const arrays. The mapping is:
/// - 0: Core/Tile_Ctrl
/// - 1: FIFO
/// - 2: Trace
/// - 10+n: North(n)
/// - 20+n: South(n)
/// - 30+n: East(n)
/// - 40+n: West(n)
/// - 50+n: DMA(n)
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

/// Shim tile stream switch master port layout.
///
/// Per AM025 PL_MODULE/STREAM_SWITCH/MASTER_CONFIG:
/// - Port 0: Tile_Ctrl
/// - Port 1: FIFO0
/// - Ports 2-7: South 0-5 (to NoC/DDR)
/// - Ports 8-11: West 0-3
/// - Ports 12-17: North 0-5 (to MemTile)
/// - Ports 18-21: East 0-3
pub const SHIM_MASTER_PORTS: &[u8] = &[
    port_type::CORE,      // 0: Tile_Ctrl
    port_type::FIFO,      // 1: FIFO0
    port_type::south(0),  // 2: South0
    port_type::south(1),  // 3: South1
    port_type::south(2),  // 4: South2
    port_type::south(3),  // 5: South3
    port_type::south(4),  // 6: South4
    port_type::south(5),  // 7: South5
    port_type::west(0),   // 8: West0
    port_type::west(1),   // 9: West1
    port_type::west(2),   // 10: West2
    port_type::west(3),   // 11: West3
    port_type::north(0),  // 12: North0
    port_type::north(1),  // 13: North1
    port_type::north(2),  // 14: North2
    port_type::north(3),  // 15: North3
    port_type::north(4),  // 16: North4
    port_type::north(5),  // 17: North5
    port_type::east(0),   // 18: East0
    port_type::east(1),   // 19: East1
    port_type::east(2),   // 20: East2
    port_type::east(3),   // 21: East3
];

/// Shim tile stream switch slave port layout.
///
/// Per AM025 PL_MODULE/STREAM_SWITCH/SLAVE_CONFIG:
/// - Port 0: Tile_Ctrl
/// - Port 1: FIFO0
/// - Ports 2-9: South 0-7 (from NoC/DDR - DMA data path)
/// - Ports 10-13: West 0-3
/// - Ports 14-17: North 0-3 (from MemTile)
/// - Ports 18-21: East 0-3
/// - Port 22: Trace
pub const SHIM_SLAVE_PORTS: &[u8] = &[
    port_type::CORE,      // 0: Tile_Ctrl
    port_type::FIFO,      // 1: FIFO0
    port_type::south(0),  // 2: South0
    port_type::south(1),  // 3: South1
    port_type::south(2),  // 4: South2
    port_type::south(3),  // 5: South3
    port_type::south(4),  // 6: South4
    port_type::south(5),  // 7: South5
    port_type::south(6),  // 8: South6
    port_type::south(7),  // 9: South7
    port_type::west(0),   // 10: West0
    port_type::west(1),   // 11: West1
    port_type::west(2),   // 12: West2
    port_type::west(3),   // 13: West3
    port_type::north(0),  // 14: North0
    port_type::north(1),  // 15: North1
    port_type::north(2),  // 16: North2
    port_type::north(3),  // 17: North3
    port_type::east(0),   // 18: East0
    port_type::east(1),   // 19: East1
    port_type::east(2),   // 20: East2
    port_type::east(3),   // 21: East3
    port_type::TRACE,     // 22: Trace
];

/// MemTile stream switch master port layout.
///
/// Per AM025 MEMORY_TILE_MODULE/STREAM_SWITCH/MASTER_CONFIG:
/// - Ports 0-5: DMA MM2S channels 0-5
/// - Port 6: Tile_Ctrl
/// - Ports 7-10: South 0-3 (to Shim)
/// - Ports 11-16: North 0-5 (to Compute tiles)
pub const MEMTILE_MASTER_PORTS: &[u8] = &[
    port_type::dma(0),    // 0: DMA0
    port_type::dma(1),    // 1: DMA1
    port_type::dma(2),    // 2: DMA2
    port_type::dma(3),    // 3: DMA3
    port_type::dma(4),    // 4: DMA4
    port_type::dma(5),    // 5: DMA5
    port_type::CORE,      // 6: Tile_Ctrl
    port_type::south(0),  // 7: South0
    port_type::south(1),  // 8: South1
    port_type::south(2),  // 9: South2
    port_type::south(3),  // 10: South3
    port_type::north(0),  // 11: North0
    port_type::north(1),  // 12: North1
    port_type::north(2),  // 13: North2
    port_type::north(3),  // 14: North3
    port_type::north(4),  // 15: North4
    port_type::north(5),  // 16: North5
];

/// MemTile stream switch slave port layout.
///
/// Per AM025 MEMORY_TILE_MODULE/STREAM_SWITCH/SLAVE_CONFIG:
/// - Ports 0-5: DMA S2MM channels 0-5
/// - Port 6: Tile_Ctrl
/// - Ports 7-12: South 0-5 (from Shim)
/// - Ports 13-16: North 0-3 (from Compute tiles)
/// - Port 17: Trace
///
/// Note the asymmetry: 6 South slaves but only 4 North slaves.
/// This matches the Shim having 6 North masters that feed the MemTile.
pub const MEMTILE_SLAVE_PORTS: &[u8] = &[
    port_type::dma(0),    // 0: DMA0
    port_type::dma(1),    // 1: DMA1
    port_type::dma(2),    // 2: DMA2
    port_type::dma(3),    // 3: DMA3
    port_type::dma(4),    // 4: DMA4
    port_type::dma(5),    // 5: DMA5
    port_type::CORE,      // 6: Tile_Ctrl
    port_type::south(0),  // 7: South0
    port_type::south(1),  // 8: South1
    port_type::south(2),  // 9: South2
    port_type::south(3),  // 10: South3
    port_type::south(4),  // 11: South4
    port_type::south(5),  // 12: South5
    port_type::north(0),  // 13: North0
    port_type::north(1),  // 14: North1
    port_type::north(2),  // 15: North2
    port_type::north(3),  // 16: North3
    port_type::TRACE,     // 17: Trace
];

/// Compute tile stream switch master port layout.
///
/// Per AM025 CORE_MODULE/STREAM_SWITCH/MASTER_CONFIG (verified against register addresses):
/// - Port 0: AIE_Core0 (AIE engine output)
/// - Ports 1-2: DMA MM2S channels 0-1
/// - Port 3: Tile_Ctrl
/// - Port 4: FIFO0
/// - Ports 5-10: South 0-5
/// - Ports 11-14: West 0-3
/// - Ports 15-18: North 0-3
/// - Ports 19-22: East 0-3
pub const COMPUTE_MASTER_PORTS: &[u8] = &[
    port_type::CORE,      // 0: AIE_Core0
    port_type::dma(0),    // 1: DMA0
    port_type::dma(1),    // 2: DMA1
    port_type::CORE,      // 3: Tile_Ctrl
    port_type::FIFO,      // 4: FIFO0
    port_type::south(0),  // 5: South0
    port_type::south(1),  // 6: South1
    port_type::south(2),  // 7: South2
    port_type::south(3),  // 8: South3
    port_type::south(4),  // 9: South4
    port_type::south(5),  // 10: South5
    port_type::west(0),   // 11: West0
    port_type::west(1),   // 12: West1
    port_type::west(2),   // 13: West2
    port_type::west(3),   // 14: West3
    port_type::north(0),  // 15: North0
    port_type::north(1),  // 16: North1
    port_type::north(2),  // 17: North2
    port_type::north(3),  // 18: North3
    port_type::east(0),   // 19: East0
    port_type::east(1),   // 20: East1
    port_type::east(2),   // 21: East2
    port_type::east(3),   // 22: East3
];

/// Compute tile stream switch slave port layout.
///
/// Per AM025 CORE_MODULE/STREAM_SWITCH/SLAVE_CONFIG (verified against register addresses):
/// - Port 0: AIE_Core0 (AIE engine input)
/// - Ports 1-2: DMA S2MM channels 0-1
/// - Port 3: Tile_Ctrl
/// - Port 4: FIFO0
/// - Ports 5-10: South 0-5
/// - Ports 11-14: West 0-3
/// - Ports 15-18: North 0-3
/// - Ports 19-22: East 0-3
/// - Port 23: AIE_Trace
/// - Port 24: Mem_Trace
pub const COMPUTE_SLAVE_PORTS: &[u8] = &[
    port_type::CORE,      // 0: AIE_Core0
    port_type::dma(0),    // 1: DMA0
    port_type::dma(1),    // 2: DMA1
    port_type::CORE,      // 3: Tile_Ctrl
    port_type::FIFO,      // 4: FIFO0
    port_type::south(0),  // 5: South0
    port_type::south(1),  // 6: South1
    port_type::south(2),  // 7: South2
    port_type::south(3),  // 8: South3
    port_type::south(4),  // 9: South4
    port_type::south(5),  // 10: South5
    port_type::west(0),   // 11: West0
    port_type::west(1),   // 12: West1
    port_type::west(2),   // 13: West2
    port_type::west(3),   // 14: West3
    port_type::north(0),  // 15: North0
    port_type::north(1),  // 16: North1
    port_type::north(2),  // 17: North2
    port_type::north(3),  // 18: North3
    port_type::east(0),   // 19: East0
    port_type::east(1),   // 20: East1
    port_type::east(2),   // 21: East2
    port_type::east(3),   // 22: East3
    port_type::TRACE,     // 23: AIE_Trace
    port_type::TRACE,     // 24: Mem_Trace
];

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

// ============================================================================
// Stream Switch Port Ranges (derived from AM025 port layout arrays above)
// ============================================================================
//
// These constants provide convenient ranges for port mapping operations.
// Values are derived from the SHIM/MEMTILE/COMPUTE_*_PORTS arrays above.

pub mod stream_switch {
    //! Stream switch port range constants for tile-to-tile routing.
    //!
    //! These define which master/slave port indices correspond to which
    //! direction (North/South) for each tile type.

    /// Shim tile port ranges (toward/from MemTile = North)
    pub mod shim {
        /// North-facing master ports: 12-17 (6 ports: North0-North5)
        /// These send data up to MemTile
        pub const NORTH_MASTER_START: u8 = 12;
        pub const NORTH_MASTER_END: u8 = 17;

        /// North-facing slave ports: 14-17 (4 ports: North0-North3)
        /// These receive data from MemTile
        pub const NORTH_SLAVE_START: u8 = 14;
        pub const NORTH_SLAVE_END: u8 = 17;
    }

    /// MemTile port ranges (South = toward Shim, North = toward Compute)
    pub mod mem_tile {
        /// South-facing master ports: 7-10 (4 ports: South0-South3)
        /// These send data down to Shim
        pub const SOUTH_MASTER_START: u8 = 7;
        pub const SOUTH_MASTER_END: u8 = 10;

        /// North-facing master ports: 11-16 (6 ports: North0-North5)
        /// These send data up to Compute tiles
        pub const NORTH_MASTER_START: u8 = 11;
        pub const NORTH_MASTER_END: u8 = 16;

        /// South-facing slave ports: 7-12 (6 ports: South0-South5)
        /// These receive data from Shim
        pub const SOUTH_SLAVE_START: u8 = 7;
        pub const SOUTH_SLAVE_END: u8 = 12;

        /// North-facing slave ports: 13-16 (4 ports: North0-North3)
        /// These receive data from Compute tiles
        pub const NORTH_SLAVE_START: u8 = 13;
        pub const NORTH_SLAVE_END: u8 = 16;
    }

    /// Compute tile port ranges (South = toward MemTile/Shim, North = toward upper tiles)
    pub mod compute {
        /// South-facing master ports: 5-10 (6 ports: South0-South5)
        /// These send data down to MemTile
        pub const SOUTH_MASTER_START: u8 = 5;
        pub const SOUTH_MASTER_END: u8 = 10;

        /// North-facing master ports: 15-18 (4 ports: North0-North3)
        /// These send data up to upper compute tiles
        pub const NORTH_MASTER_START: u8 = 15;
        pub const NORTH_MASTER_END: u8 = 18;

        /// South-facing slave ports: 5-10 (6 ports: South0-South5)
        /// These receive data from MemTile
        pub const SOUTH_SLAVE_START: u8 = 5;
        pub const SOUTH_SLAVE_END: u8 = 10;

        /// North-facing slave ports: 15-18 (4 ports: North0-North3)
        /// These receive data from upper compute tiles
        pub const NORTH_SLAVE_START: u8 = 15;
        pub const NORTH_SLAVE_END: u8 = 18;
    }

    /// Stream switch configuration bit constants
    pub const ENABLE_BIT: u32 = 31;
    pub const SLAVE_SELECT_MASK: u32 = 0x1F;
}

// ============================================================================
// Re-export Register Specifications
// ============================================================================
//
// For unified API access, re-export the registers_spec module.
// Users can access via `aie2_spec::registers::memory_module::LOCK_BASE` etc.

pub use super::registers_spec as registers;
