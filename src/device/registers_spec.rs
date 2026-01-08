//! AIE2 Register Address Specification
//!
//! All addresses and bit field layouts derived from AMD AM025 (AIE-ML Register Reference).
//! This module centralizes register addresses to eliminate magic numbers throughout the codebase.
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

    // Lock registers (AM025 memory_module/lock/value.txt)

    /// Lock value registers base address
    pub const LOCK_BASE: u32 = 0x1F000;

    /// Spacing between lock registers (16 bytes per lock)
    /// AM025: Lock0_value @ 0x1F000, Lock1_value @ 0x1F010, etc.
    pub const LOCK_STRIDE: u32 = 0x10;

    /// Number of locks per compute tile
    pub const LOCK_COUNT: usize = 16;

    /// Lock address range end (exclusive)
    pub const LOCK_END: u32 = LOCK_BASE + (LOCK_COUNT as u32 * LOCK_STRIDE);

    // Lock_Request register (AM025 memory_module/lock/misc.txt)
    // 16KB address space where address encodes lock operation parameters

    /// Lock_Request base address
    pub const LOCK_REQUEST_BASE: u32 = 0x40000;

    /// Lock_Request end address (exclusive)
    pub const LOCK_REQUEST_END: u32 = 0x44000; // 16KB range

    /// Lock_Request address field: Lock_Id [13:10] (4 bits)
    pub const LOCK_REQUEST_ID_SHIFT: u32 = 10;
    pub const LOCK_REQUEST_ID_MASK: u32 = 0xF; // 4 bits for 16 locks

    /// Lock_Request address field: Acq_Rel [9] (1=acquire, 0=release)
    pub const LOCK_REQUEST_ACQ_REL_BIT: u32 = 9;

    /// Lock_Request address field: Change_Value [8:2] (7 bits signed)
    pub const LOCK_REQUEST_VALUE_SHIFT: u32 = 2;
    pub const LOCK_REQUEST_VALUE_MASK: u32 = 0x7F; // 7 bits

    // Lock status registers (AM025 memory_module/lock/value.txt)

    /// Lock overflow status register (write-to-clear)
    pub const LOCKS_OVERFLOW: u32 = 0x1F120;

    /// Lock underflow status register (write-to-clear)
    pub const LOCKS_UNDERFLOW: u32 = 0x1F128;

    // DMA buffer descriptors (AM025 memory_module/dma/bd.txt)

    /// DMA buffer descriptor base address
    pub const DMA_BD_BASE: u32 = 0x1D000;

    /// Spacing between BDs (32 bytes = 8 words, though only 6 used)
    /// AM025: BD0 @ 0x1D000, BD1 @ 0x1D020, etc.
    pub const DMA_BD_STRIDE: u32 = 0x20;

    /// Number of BDs per compute tile
    pub const DMA_BD_COUNT: usize = 16;

    /// Words per BD (6 words used, 2 padding)
    pub const DMA_BD_WORDS: usize = 6;

    /// DMA BD address range end (exclusive)
    pub const DMA_BD_END: u32 = DMA_BD_BASE + (DMA_BD_COUNT as u32 * DMA_BD_STRIDE);

    // DMA channel control (AM025 memory_module/dma/mm2s.txt, s2mm.txt)

    /// DMA channel control base address
    pub const DMA_CHANNEL_BASE: u32 = 0x1DE00;

    /// Spacing between channel register pairs (ctrl + start_queue = 8 bytes)
    pub const DMA_CHANNEL_STRIDE: u32 = 0x08;

    /// Number of DMA channels (2 S2MM + 2 MM2S)
    pub const DMA_CHANNEL_COUNT: usize = 4;

    /// DMA channel status base
    pub const DMA_STATUS_BASE: u32 = 0x1DF00;

    /// Channel register field layouts (AM025 memory_module/dma/s2mm.txt, mm2s.txt)
    pub mod channel {
        // S2MM Control Register (DMA_S2MM_x_Ctrl @ 0x1DE00, 0x1DE08)
        // MM2S Control Register (DMA_MM2S_x_Ctrl @ 0x1DE10, 0x1DE18)

        /// FoT_Mode shift (bits 17:16) - S2MM only
        /// 00 = disabled, 01 = no_counts, 10 = counts_with_tokens, 11 = counts_from_register
        pub const CTRL_FOT_MODE_SHIFT: u32 = 16;
        /// FoT_Mode mask (2 bits)
        pub const CTRL_FOT_MODE_MASK: u32 = 0x3;

        /// Controller_ID shift (bits 15:8)
        pub const CTRL_CONTROLLER_ID_SHIFT: u32 = 8;
        /// Controller_ID mask (8 bits)
        pub const CTRL_CONTROLLER_ID_MASK: u32 = 0xFF;

        /// Compression/Decompression enable bit (bit 4)
        pub const CTRL_COMPRESSION_ENABLE_BIT: u32 = 4;

        /// Enable Out-of-Order bit (bit 3) - S2MM only
        pub const CTRL_ENABLE_OUT_OF_ORDER_BIT: u32 = 3;

        /// Reset bit (bit 1)
        pub const CTRL_RESET_BIT: u32 = 1;

        // Start_Queue Register (DMA_S2MM_x_Start_Queue, DMA_MM2S_x_Start_Queue)

        /// Enable_Token_Issue bit (bit 31)
        pub const START_QUEUE_ENABLE_TOKEN_ISSUE_BIT: u32 = 31;

        /// Repeat_Count shift (bits 23:16)
        pub const START_QUEUE_REPEAT_COUNT_SHIFT: u32 = 16;
        /// Repeat_Count mask (8 bits, actual - 1)
        pub const START_QUEUE_REPEAT_COUNT_MASK: u32 = 0xFF;

        /// Start_BD_ID mask (bits 3:0)
        pub const START_QUEUE_BD_ID_MASK: u32 = 0xF;

        // Status Register fields (DMA_S2MM_Status_x, DMA_MM2S_Status_x)

        /// Cur_BD shift (bits 27:24)
        pub const STATUS_CUR_BD_SHIFT: u32 = 24;
        /// Cur_BD mask (4 bits)
        pub const STATUS_CUR_BD_MASK: u32 = 0xF;

        /// Task_Queue_Size shift (bits 22:20)
        pub const STATUS_TASK_QUEUE_SIZE_SHIFT: u32 = 20;
        /// Task_Queue_Size mask (3 bits)
        pub const STATUS_TASK_QUEUE_SIZE_MASK: u32 = 0x7;

        /// Channel_Running bit (bit 19)
        pub const STATUS_CHANNEL_RUNNING_BIT: u32 = 19;

        /// Task_Queue_Overflow bit (bit 18)
        pub const STATUS_TASK_QUEUE_OVERFLOW_BIT: u32 = 18;

        /// Error_BD_Invalid bit (bit 11)
        pub const STATUS_ERROR_BD_INVALID_BIT: u32 = 11;

        /// Error_BD_Unavailable bit (bit 10) - S2MM out-of-order mode only
        pub const STATUS_ERROR_BD_UNAVAILABLE_BIT: u32 = 10;

        /// Stalled_TCT bit (bit 5)
        pub const STATUS_STALLED_TCT_BIT: u32 = 5;

        /// Stalled_Stream bit (bit 4)
        pub const STATUS_STALLED_STREAM_BIT: u32 = 4;

        /// Stalled_Lock_Rel bit (bit 3)
        pub const STATUS_STALLED_LOCK_REL_BIT: u32 = 3;

        /// Stalled_Lock_Acq bit (bit 2)
        pub const STATUS_STALLED_LOCK_ACQ_BIT: u32 = 2;

        /// Status mask (bits 1:0)
        /// 00=IDLE, 01=STARTING, 10=RUNNING
        pub const STATUS_STATE_MASK: u32 = 0x3;

        /// FoT Mode values
        pub const FOT_DISABLED: u8 = 0;
        pub const FOT_NO_COUNTS: u8 = 1;
        pub const FOT_COUNTS_WITH_TOKENS: u8 = 2;
        pub const FOT_COUNTS_FROM_REGISTER: u8 = 3;
    }

    // Stream switch (AM025 memory_module/../stream_switch/)

    /// Stream switch master config base
    pub const STREAM_SWITCH_MASTER_BASE: u32 = 0x3F000;

    /// Stream switch master config end
    pub const STREAM_SWITCH_MASTER_END: u32 = 0x3F058;

    /// Stream switch slave config base
    pub const STREAM_SWITCH_SLAVE_BASE: u32 = 0x3F100;

    /// Stream switch slave config end
    pub const STREAM_SWITCH_SLAVE_END: u32 = 0x3F180;

    /// BD field layouts (AM025 memory_module/dma/bd.txt)
    pub mod bd {
        // Word 0: Base_Address[27:14], Buffer_Length[13:0]

        /// Base address shift (bits 27:14)
        pub const WORD0_BASE_ADDR_SHIFT: u32 = 14;
        /// Base address mask (14 bits)
        pub const WORD0_BASE_ADDR_MASK: u32 = 0x3FFF;
        /// Buffer length mask (bits 13:0, 14 bits)
        pub const WORD0_BUFFER_LEN_MASK: u32 = 0x3FFF;

        // Word 1: Compression/Packet Control (MM2S only)
        // AM025: memory_module/dma/bd.txt Word 1

        /// Enable compression bit (bit 31)
        pub const WORD1_ENABLE_COMPRESSION_BIT: u32 = 31;
        /// Enable packet header bit (bit 30)
        pub const WORD1_ENABLE_PACKET_BIT: u32 = 30;
        /// Out-of-order BD ID shift (bits 29:24)
        pub const WORD1_OOO_BD_ID_SHIFT: u32 = 24;
        /// Out-of-order BD ID mask (6 bits)
        pub const WORD1_OOO_BD_ID_MASK: u32 = 0x3F;
        /// Packet ID shift (bits 23:19)
        pub const WORD1_PACKET_ID_SHIFT: u32 = 19;
        /// Packet ID mask (5 bits)
        pub const WORD1_PACKET_ID_MASK: u32 = 0x1F;
        /// Packet type shift (bits 18:16)
        pub const WORD1_PACKET_TYPE_SHIFT: u32 = 16;
        /// Packet type mask (3 bits)
        pub const WORD1_PACKET_TYPE_MASK: u32 = 0x7;

        // Word 2: Dimension Stepsizes
        // AM025: memory_module/dma/bd.txt Word 2

        /// D1 stepsize shift (bits 25:13)
        pub const WORD2_D1_STEPSIZE_SHIFT: u32 = 13;
        /// D1 stepsize mask (13 bits, actual stride - 1)
        pub const WORD2_D1_STEPSIZE_MASK: u32 = 0x1FFF;
        /// D0 stepsize mask (bits 12:0, 13 bits, actual stride - 1)
        pub const WORD2_D0_STEPSIZE_MASK: u32 = 0x1FFF;

        // Word 3: Wrap Counts and D2 Stepsize
        // AM025: memory_module/dma/bd.txt Word 3

        /// D1 wrap count shift (bits 28:21)
        pub const WORD3_D1_WRAP_SHIFT: u32 = 21;
        /// D1 wrap count mask (8 bits, 0 = no wrap)
        pub const WORD3_D1_WRAP_MASK: u32 = 0xFF;
        /// D0 wrap count shift (bits 20:13)
        pub const WORD3_D0_WRAP_SHIFT: u32 = 13;
        /// D0 wrap count mask (8 bits, 0 = no wrap)
        pub const WORD3_D0_WRAP_MASK: u32 = 0xFF;
        /// D2 stepsize mask (bits 12:0, 13 bits, actual stride - 1)
        pub const WORD3_D2_STEPSIZE_MASK: u32 = 0x1FFF;

        // Word 4: Iteration Control
        // AM025: memory_module/dma/bd.txt Word 4

        /// Iteration current shift (bits 24:19)
        pub const WORD4_ITERATION_CURRENT_SHIFT: u32 = 19;
        /// Iteration current mask (6 bits)
        pub const WORD4_ITERATION_CURRENT_MASK: u32 = 0x3F;
        /// Iteration wrap shift (bits 18:13)
        pub const WORD4_ITERATION_WRAP_SHIFT: u32 = 13;
        /// Iteration wrap mask (6 bits, actual - 1)
        pub const WORD4_ITERATION_WRAP_MASK: u32 = 0x3F;
        /// Iteration stepsize mask (bits 12:0, 13 bits, actual - 1)
        pub const WORD4_ITERATION_STEPSIZE_MASK: u32 = 0x1FFF;

        // Word 5: Lock and chaining fields

        /// TLAST suppress bit (bit 31)
        pub const WORD5_TLAST_SUPPRESS_BIT: u32 = 31;
        /// Next BD field shift (bits 30:27)
        pub const WORD5_NEXT_BD_SHIFT: u32 = 27;
        /// Next BD mask (4 bits)
        pub const WORD5_NEXT_BD_MASK: u32 = 0xF;
        /// Use next BD bit (bit 26)
        pub const WORD5_USE_NEXT_BD_BIT: u32 = 26;
        /// Valid BD bit (bit 25)
        pub const WORD5_VALID_BD_BIT: u32 = 25;

        /// Lock release value shift (bits 24:18)
        pub const WORD5_LOCK_REL_VALUE_SHIFT: u32 = 18;
        /// Lock release value mask (7 bits, signed)
        pub const WORD5_LOCK_REL_VALUE_MASK: u32 = 0x7F;
        /// Lock release ID shift (bits 16:13)
        pub const WORD5_LOCK_REL_ID_SHIFT: u32 = 13;
        /// Lock release ID mask (4 bits)
        pub const WORD5_LOCK_REL_ID_MASK: u32 = 0xF;

        /// Lock acquire enable bit (bit 12)
        pub const WORD5_LOCK_ACQ_ENABLE_BIT: u32 = 12;
        /// Lock acquire value shift (bits 11:5)
        pub const WORD5_LOCK_ACQ_VALUE_SHIFT: u32 = 5;
        /// Lock acquire value mask (7 bits, signed)
        pub const WORD5_LOCK_ACQ_VALUE_MASK: u32 = 0x7F;
        /// Lock acquire ID mask (bits 3:0, 4 bits)
        pub const WORD5_LOCK_ACQ_ID_MASK: u32 = 0xF;
    }
}

// ============================================================================
// Core Module Registers (Compute Tiles)
// AM025: CORE_MODULE
// ============================================================================

pub mod core_module {
    //! Core module register addresses.
    //! AM025 Section: CORE_MODULE

    /// Core control register
    pub const CORE_CONTROL: u32 = 0x32000;

    /// Core status register
    pub const CORE_STATUS: u32 = 0x32004;

    /// Core enable events
    pub const CORE_ENABLE_EVENTS: u32 = 0x32008;

    /// Core reset event
    pub const CORE_RESET_EVENT: u32 = 0x3200C;

    /// Core debug control 0
    pub const CORE_DEBUG_CONTROL0: u32 = 0x32400;

    /// Program counter
    pub const CORE_PC: u32 = 0x31100;

    /// Stack pointer
    pub const CORE_SP: u32 = 0x31120;

    /// Link register
    pub const CORE_LR: u32 = 0x31130;

    /// Tile control
    pub const TILE_CONTROL: u32 = 0x36030;

    /// Memory control
    pub const MEMORY_CONTROL: u32 = 0x36070;

    /// Core module offset range start
    pub const OFFSET_START: u32 = 0x30000;

    /// Core module offset range end
    pub const OFFSET_END: u32 = 0x3EFFF;
}

// ============================================================================
// Memory Tile Module Registers
// AM025: MEMORY_TILE_MODULE
// ============================================================================

pub mod mem_tile_module {
    //! Memory tile module register addresses.
    //! AM025 Section: MEMORY_TILE_MODULE

    // Lock registers

    /// Lock value registers base address
    pub const LOCK_BASE: u32 = 0xC0000;

    /// Spacing between lock registers (16 bytes per lock)
    pub const LOCK_STRIDE: u32 = 0x10;

    /// Number of locks per memory tile
    pub const LOCK_COUNT: usize = 64;

    /// Lock address range end
    pub const LOCK_END: u32 = LOCK_BASE + (LOCK_COUNT as u32 * LOCK_STRIDE);

    // Lock_Request register (AM025 memory_tile_module/lock/misc.txt)
    // 64KB address space where address encodes lock operation parameters

    /// Lock_Request base address
    pub const LOCK_REQUEST_BASE: u32 = 0xD0000;

    /// Lock_Request end address (exclusive)
    pub const LOCK_REQUEST_END: u32 = 0xE0000; // 64KB range

    /// Lock_Request address field: Lock_Id [15:10] (6 bits)
    pub const LOCK_REQUEST_ID_SHIFT: u32 = 10;
    pub const LOCK_REQUEST_ID_MASK: u32 = 0x3F; // 6 bits for 64 locks

    /// Lock_Request address field: Acq_Rel [9] (1=acquire, 0=release)
    pub const LOCK_REQUEST_ACQ_REL_BIT: u32 = 9;

    /// Lock_Request address field: Change_Value [8:2] (7 bits signed)
    pub const LOCK_REQUEST_VALUE_SHIFT: u32 = 2;
    pub const LOCK_REQUEST_VALUE_MASK: u32 = 0x7F; // 7 bits

    // Lock status registers (AM025 memory_tile_module/lock/value.txt)

    /// Lock overflow status register 0 (locks 0-31, write-to-clear)
    pub const LOCKS_OVERFLOW_0: u32 = 0xC0420;

    /// Lock overflow status register 1 (locks 32-63, write-to-clear)
    pub const LOCKS_OVERFLOW_1: u32 = 0xC0424;

    /// Lock underflow status register 0 (locks 0-31, write-to-clear)
    pub const LOCKS_UNDERFLOW_0: u32 = 0xC0428;

    /// Lock underflow status register 1 (locks 32-63, write-to-clear)
    pub const LOCKS_UNDERFLOW_1: u32 = 0xC042C;

    // DMA buffer descriptors

    /// DMA buffer descriptor base address
    pub const DMA_BD_BASE: u32 = 0xA0000;

    /// Spacing between BDs (32 bytes)
    pub const DMA_BD_STRIDE: u32 = 0x20;

    /// Number of BDs per memory tile (24 S2MM + 24 MM2S)
    pub const DMA_BD_COUNT: usize = 48;

    /// Words per BD (8 words for MemTile)
    pub const DMA_BD_WORDS: usize = 8;

    // DMA channel control

    /// S2MM channel control base
    pub const DMA_CHANNEL_S2MM_BASE: u32 = 0xA0600;

    /// MM2S channel control base
    pub const DMA_CHANNEL_MM2S_BASE: u32 = 0xA0630;

    /// Spacing between channel registers
    pub const DMA_CHANNEL_STRIDE: u32 = 0x08;

    /// Number of S2MM channels
    pub const S2MM_CHANNEL_COUNT: usize = 6;

    /// Number of MM2S channels
    pub const MM2S_CHANNEL_COUNT: usize = 6;

    // Stream switch

    /// Stream switch master config base
    pub const STREAM_SWITCH_MASTER_BASE: u32 = 0xB0000;

    /// Stream switch master config end
    pub const STREAM_SWITCH_MASTER_END: u32 = 0xB0100;

    /// Stream switch slave config base
    pub const STREAM_SWITCH_SLAVE_BASE: u32 = 0xB0100;

    /// Stream switch slave config end
    pub const STREAM_SWITCH_SLAVE_END: u32 = 0xB0200;

    /// BD field layouts for MemTile (AM025 memory_tile_module/dma/bd.txt)
    /// Note: MemTile BDs have different field layouts than compute tile BDs
    pub mod bd {
        // Word 0: Buffer_Length[16:0] (17 bits for MemTile)

        /// Buffer length mask (17 bits)
        pub const WORD0_BUFFER_LEN_MASK: u32 = 0x1FFFF;

        // Word 1: Base_Address[18:0], Use_Next_BD[19], Next_BD[25:20]

        /// Base address mask (19 bits)
        pub const WORD1_BASE_ADDR_MASK: u32 = 0x7FFFF;
        /// Use next BD bit (bit 19)
        pub const WORD1_USE_NEXT_BD_BIT: u32 = 19;
        /// Next BD shift (bits 25:20)
        pub const WORD1_NEXT_BD_SHIFT: u32 = 20;
        /// Next BD mask (6 bits for MemTile)
        pub const WORD1_NEXT_BD_MASK: u32 = 0x3F;

        // Word 7: Lock and valid fields

        /// Valid BD bit (bit 31)
        pub const WORD7_VALID_BD_BIT: u32 = 31;
        /// Lock release value shift (bits 30:24)
        pub const WORD7_LOCK_REL_VALUE_SHIFT: u32 = 24;
        /// Lock release value mask (7 bits)
        pub const WORD7_LOCK_REL_VALUE_MASK: u32 = 0x7F;
        /// Lock release ID shift (bits 23:16)
        pub const WORD7_LOCK_REL_ID_SHIFT: u32 = 16;
        /// Lock release ID mask (8 bits for MemTile)
        pub const WORD7_LOCK_REL_ID_MASK: u32 = 0xFF;
        /// Lock acquire enable bit (bit 15)
        pub const WORD7_LOCK_ACQ_ENABLE_BIT: u32 = 15;
        /// Lock acquire value shift (bits 14:8)
        pub const WORD7_LOCK_ACQ_VALUE_SHIFT: u32 = 8;
        /// Lock acquire value mask (7 bits)
        pub const WORD7_LOCK_ACQ_VALUE_MASK: u32 = 0x7F;
        /// Lock acquire ID mask (bits 7:0, 8 bits for MemTile)
        pub const WORD7_LOCK_ACQ_ID_MASK: u32 = 0xFF;
    }
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

/// Data memory end offset for compute tile (64 KB)
pub const COMPUTE_DATA_MEMORY_END: u32 = 0x0FFFF;

/// Data memory end offset for memory tile (512 KB)
pub const MEM_TILE_DATA_MEMORY_END: u32 = 0x7FFFF;

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

/// Compute lock register address from lock index
#[inline]
pub const fn compute_lock_addr(lock_idx: usize) -> u32 {
    memory_module::LOCK_BASE + (lock_idx as u32 * memory_module::LOCK_STRIDE)
}

/// Compute BD register address from BD index and word
#[inline]
pub const fn compute_bd_addr(bd_idx: usize, word: usize) -> u32 {
    memory_module::DMA_BD_BASE + (bd_idx as u32 * memory_module::DMA_BD_STRIDE) + (word as u32 * 4)
}

/// Compute MemTile lock register address from lock index
#[inline]
pub const fn mem_tile_lock_addr(lock_idx: usize) -> u32 {
    mem_tile_module::LOCK_BASE + (lock_idx as u32 * mem_tile_module::LOCK_STRIDE)
}

/// Compute MemTile BD register address from BD index and word
#[inline]
pub const fn mem_tile_bd_addr(bd_idx: usize, word: usize) -> u32 {
    mem_tile_module::DMA_BD_BASE + (bd_idx as u32 * mem_tile_module::DMA_BD_STRIDE) + (word as u32 * 4)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_addresses_match_am025() {
        // AM025: Lock0_value @ 0x1F000, Lock1_value @ 0x1F010, etc.
        assert_eq!(compute_lock_addr(0), 0x1F000);
        assert_eq!(compute_lock_addr(1), 0x1F010);
        assert_eq!(compute_lock_addr(15), 0x1F0F0);
    }

    #[test]
    fn test_bd_addresses_match_am025() {
        // AM025: DMA_BD0_0 @ 0x1D000, DMA_BD1_0 @ 0x1D020, etc.
        assert_eq!(compute_bd_addr(0, 0), 0x1D000);
        assert_eq!(compute_bd_addr(0, 5), 0x1D014);
        assert_eq!(compute_bd_addr(1, 0), 0x1D020);
        assert_eq!(compute_bd_addr(15, 0), 0x1D1E0);
    }

    #[test]
    fn test_memtile_lock_addresses() {
        assert_eq!(mem_tile_lock_addr(0), 0xC0000);
        assert_eq!(mem_tile_lock_addr(1), 0xC0010);
        assert_eq!(mem_tile_lock_addr(63), 0xC03F0);
    }

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

    #[test]
    fn test_lock_stride_is_16_bytes() {
        // This was the bug we fixed - locks are 16 bytes apart, not 4
        assert_eq!(memory_module::LOCK_STRIDE, 0x10);
        assert_eq!(mem_tile_module::LOCK_STRIDE, 0x10);
    }

    #[test]
    fn test_bd_stride_is_32_bytes() {
        assert_eq!(memory_module::DMA_BD_STRIDE, 0x20);
        assert_eq!(mem_tile_module::DMA_BD_STRIDE, 0x20);
    }
}
