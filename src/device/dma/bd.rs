//! Buffer Descriptor (BD) parsing and types for AIE-ML DMA.
//!
//! This module provides clean BD parsing from register layouts for all three tile types:
//! - Compute tile: 6 registers per BD, 16 BDs, 14-bit word address
//! - MemTile: 8 registers per BD, 48 BDs, 19-bit word address
//! - Shim tile: 8 registers per BD, 16 BDs, 46-bit word address
//!
//! All addresses and stepsizes are in 32-bit WORD units (hardware native).
//! Stepsizes in hardware are stored as (actual - 1); we convert to actual values.
//!
//! Reference: AMD AM020/AM025 and docs/dma-reference.md

use crate::device::tile::TileType;

/// BD base addresses per tile type (from AM025)
pub const BD_BASE_COMPUTE: u64 = 0x1D000;
pub const BD_BASE_MEMTILE: u64 = 0xA0000;
pub const BD_BASE_SHIM: u64 = 0x1D000;

/// BD spacing: 32 bytes (0x20) between BDs for all tile types
pub const BD_SPACING: u64 = 0x20;

/// Number of BDs per tile type
pub const BD_COUNT_COMPUTE: usize = 16;
pub const BD_COUNT_MEMTILE: usize = 48;
pub const BD_COUNT_SHIM: usize = 16;

/// Number of registers per BD
pub const BD_REGS_COMPUTE: usize = 6;
pub const BD_REGS_MEMTILE: usize = 8;
pub const BD_REGS_SHIM: usize = 8;

/// Parsed Buffer Descriptor fields (common to all tile types).
///
/// All addresses and stepsizes are in 32-bit WORD units.
/// Stepsizes are converted to actual values (stored + 1).
#[derive(Debug, Clone, Default)]
pub struct BufferDescriptor {
    /// BD is valid and can be used
    pub valid: bool,

    /// Base address in 32-bit word units
    pub base_addr_words: u64,

    /// Transfer length in 32-bit words
    pub length_words: u32,

    // Dimensional addressing (all stepsizes are actual values)
    /// Dimension 0 stepsize in words (actual, not stored-1)
    pub d0_stepsize: u32,
    /// Dimension 0 wrap count (0 = no wrap)
    pub d0_wrap: u16,

    /// Dimension 1 stepsize in words
    pub d1_stepsize: u32,
    /// Dimension 1 wrap count
    pub d1_wrap: u16,

    /// Dimension 2 stepsize in words
    pub d2_stepsize: u32,
    /// Dimension 2 wrap count
    pub d2_wrap: u16,

    /// Dimension 3 stepsize in words (MemTile only)
    pub d3_stepsize: u32,

    // Iteration (outermost loop)
    /// Iteration stepsize in words (actual)
    pub iteration_stepsize: u32,
    /// Iteration wrap (actual, 1-64)
    pub iteration_wrap: u8,
    /// Current iteration counter
    pub iteration_current: u8,

    // Lock configuration
    /// Enable lock acquire before transfer
    pub lock_acq_enable: bool,
    /// Lock ID to acquire (4-bit for Compute/Shim, 8-bit for MemTile)
    pub lock_acq_id: u8,
    /// Lock acquire value (signed: >=0 means exact match, <0 means GE)
    pub lock_acq_value: i8,
    /// Lock ID to release after transfer
    pub lock_rel_id: u8,
    /// Lock release delta (signed, 0 = no release)
    pub lock_rel_value: i8,

    // BD chaining
    /// Continue with next BD after completion
    pub use_next_bd: bool,
    /// Next BD ID to chain to
    pub next_bd: u8,

    // Packet mode (MM2S only)
    /// Enable packet header insertion
    pub enable_packet: bool,
    /// Packet ID (5 bits)
    pub packet_id: u8,
    /// Packet type (3 bits)
    pub packet_type: u8,
    /// Out-of-order BD ID (6 bits)
    pub ooo_bd_id: u8,
    /// Suppress TLAST at end of transfer
    pub tlast_suppress: bool,

    // Compression (optional)
    /// Enable compression (MM2S) or decompression (S2MM)
    pub compression_enable: bool,

    // MemTile-specific: zero padding (MM2S only)
    /// D0 zeros before
    pub d0_zero_before: u8,
    /// D0 zeros after
    pub d0_zero_after: u8,
    /// D1 zeros before
    pub d1_zero_before: u8,
    /// D1 zeros after
    pub d1_zero_after: u8,
    /// D2 zeros before
    pub d2_zero_before: u8,
    /// D2 zeros after
    pub d2_zero_after: u8,

    // Shim-specific: AXI parameters
    /// Burst length (0=64B, 1=128B, 2=256B)
    pub burst_length: u8,
    /// AXI SMID
    pub smid: u8,
    /// AXI cache attributes
    pub axcache: u8,
    /// AXI QoS
    pub axqos: u8,
    /// Secure access flag
    pub secure_access: bool,
}

impl BufferDescriptor {
    /// Create a new empty/invalid BD
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse BD from register words based on tile type.
    ///
    /// # Arguments
    /// * `words` - Slice of 6-8 u32 register values (tile-type dependent)
    /// * `tile_type` - Type of tile this BD belongs to
    ///
    /// # Panics
    /// Panics if `words` slice is too short for the tile type.
    pub fn from_registers(words: &[u32], tile_type: TileType) -> Self {
        match tile_type {
            TileType::Compute => Self::parse_compute(words),
            TileType::MemTile => Self::parse_memtile(words),
            TileType::Shim => Self::parse_shim(words),
        }
    }

    /// Parse Compute tile BD (6 registers per BD).
    ///
    /// Layout from AM025 MEMORY_MODULE/DMA/BD:
    /// - BD_0: Base_Address[27:14], Buffer_Length[13:0]
    /// - BD_1: Compression[31], Packet[30], OOO_ID[29:24], Packet_ID[23:19], Packet_Type[18:16]
    /// - BD_2: D1_Stepsize[25:13], D0_Stepsize[12:0]
    /// - BD_3: D1_Wrap[28:21], D0_Wrap[20:13], D2_Stepsize[12:0]
    /// - BD_4: Iteration_Current[24:19], Iteration_Wrap[18:13], Iteration_Stepsize[12:0]
    /// - BD_5: TLAST_Suppress[31], Next_BD[30:27], Use_Next_BD[26], Valid_BD[25],
    ///         Lock_Rel_Value[24:18], Lock_Rel_ID[16:13], Lock_Acq_Enable[12],
    ///         Lock_Acq_Value[11:5], Lock_Acq_ID[3:0]
    fn parse_compute(words: &[u32]) -> Self {
        assert!(words.len() >= BD_REGS_COMPUTE, "Compute BD needs 6 registers");

        let w0 = words[0];
        let w1 = words[1];
        let w2 = words[2];
        let w3 = words[3];
        let w4 = words[4];
        let w5 = words[5];

        Self {
            // BD_0
            base_addr_words: ((w0 >> 14) & 0x3FFF) as u64,  // 14 bits
            length_words: w0 & 0x3FFF,                       // 14 bits

            // BD_1
            compression_enable: (w1 >> 31) & 1 != 0,
            enable_packet: (w1 >> 30) & 1 != 0,
            ooo_bd_id: ((w1 >> 24) & 0x3F) as u8,
            packet_id: ((w1 >> 19) & 0x1F) as u8,
            packet_type: ((w1 >> 16) & 0x7) as u8,

            // BD_2 - stepsizes stored as (actual-1), convert to actual
            d1_stepsize: ((w2 >> 13) & 0x1FFF) + 1,  // 13 bits + 1
            d0_stepsize: (w2 & 0x1FFF) + 1,           // 13 bits + 1

            // BD_3
            d1_wrap: ((w3 >> 21) & 0xFF) as u16,     // 8 bits
            d0_wrap: ((w3 >> 13) & 0xFF) as u16,     // 8 bits
            d2_stepsize: (w3 & 0x1FFF) + 1,          // 13 bits + 1

            // BD_4
            iteration_current: ((w4 >> 19) & 0x3F) as u8,  // 6 bits
            iteration_wrap: (((w4 >> 13) & 0x3F) + 1) as u8, // 6 bits, stored as actual-1
            iteration_stepsize: (w4 & 0x1FFF) + 1,   // 13 bits + 1

            // BD_5
            tlast_suppress: (w5 >> 31) & 1 != 0,
            next_bd: ((w5 >> 27) & 0xF) as u8,       // 4 bits
            use_next_bd: (w5 >> 26) & 1 != 0,
            valid: (w5 >> 25) & 1 != 0,
            lock_rel_value: sign_extend_7bit(((w5 >> 18) & 0x7F) as u8),
            lock_rel_id: ((w5 >> 13) & 0xF) as u8,   // 4 bits
            lock_acq_enable: (w5 >> 12) & 1 != 0,
            lock_acq_value: sign_extend_7bit(((w5 >> 5) & 0x7F) as u8),
            lock_acq_id: (w5 & 0xF) as u8,           // 4 bits

            // Not used in compute tile
            d3_stepsize: 0,
            d2_wrap: 0,
            d0_zero_before: 0,
            d0_zero_after: 0,
            d1_zero_before: 0,
            d1_zero_after: 0,
            d2_zero_before: 0,
            d2_zero_after: 0,
            burst_length: 0,
            smid: 0,
            axcache: 0,
            axqos: 0,
            secure_access: false,
        }
    }

    /// Parse MemTile BD (8 registers per BD).
    ///
    /// Layout from AM025 MEMORY_TILE_MODULE/DMA/BD:
    /// - BD_0: Enable_Packet[31], Packet_Type[30:28], Packet_ID[27:23],
    ///         OOO_BD_ID[22:17], Buffer_Length[16:0]
    /// - BD_1: D0_Zero_Before[31:26], Next_BD[25:20], Use_Next_BD[19], Base_Address[18:0]
    /// - BD_2: TLAST_Suppress[31], D0_Wrap[26:17], D0_Stepsize[16:0]
    /// - BD_3: D1_Zero_Before[31:27], D1_Wrap[26:17], D1_Stepsize[16:0]
    /// - BD_4: Compression[31], D2_Zero_Before[30:27], D2_Wrap[26:17], D2_Stepsize[16:0]
    /// - BD_5: D2_Zero_After[31:28], D1_Zero_After[27:23], D0_Zero_After[22:17], D3_Stepsize[16:0]
    /// - BD_6: Iteration_Current[28:23], Iteration_Wrap[22:17], Iteration_Stepsize[16:0]
    /// - BD_7: Valid_BD[31], Lock_Rel_Value[30:24], Lock_Rel_ID[23:16],
    ///         Lock_Acq_Enable[15], Lock_Acq_Value[14:8], Lock_Acq_ID[7:0]
    fn parse_memtile(words: &[u32]) -> Self {
        assert!(words.len() >= BD_REGS_MEMTILE, "MemTile BD needs 8 registers");

        let w0 = words[0];
        let w1 = words[1];
        let w2 = words[2];
        let w3 = words[3];
        let w4 = words[4];
        let w5 = words[5];
        let w6 = words[6];
        let w7 = words[7];

        Self {
            // BD_0
            enable_packet: (w0 >> 31) & 1 != 0,
            packet_type: ((w0 >> 28) & 0x7) as u8,
            packet_id: ((w0 >> 23) & 0x1F) as u8,
            ooo_bd_id: ((w0 >> 17) & 0x3F) as u8,
            length_words: w0 & 0x1FFFF,              // 17 bits (up to 128K)

            // BD_1
            d0_zero_before: ((w1 >> 26) & 0x3F) as u8,
            next_bd: ((w1 >> 20) & 0x3F) as u8,      // 6 bits (0-47, but only 0-23 valid)
            use_next_bd: (w1 >> 19) & 1 != 0,
            base_addr_words: (w1 & 0x7FFFF) as u64,  // 19 bits (512KB)

            // BD_2
            tlast_suppress: (w2 >> 31) & 1 != 0,
            d0_wrap: ((w2 >> 17) & 0x3FF) as u16,    // 10 bits
            d0_stepsize: (w2 & 0x1FFFF) + 1,         // 17 bits + 1

            // BD_3
            d1_zero_before: ((w3 >> 27) & 0x1F) as u8,
            d1_wrap: ((w3 >> 17) & 0x3FF) as u16,
            d1_stepsize: (w3 & 0x1FFFF) + 1,

            // BD_4
            compression_enable: (w4 >> 31) & 1 != 0,
            d2_zero_before: ((w4 >> 27) & 0xF) as u8,
            d2_wrap: ((w4 >> 17) & 0x3FF) as u16,
            d2_stepsize: (w4 & 0x1FFFF) + 1,

            // BD_5
            d2_zero_after: ((w5 >> 28) & 0xF) as u8,
            d1_zero_after: ((w5 >> 23) & 0x1F) as u8,
            d0_zero_after: ((w5 >> 17) & 0x3F) as u8,
            d3_stepsize: (w5 & 0x1FFFF) + 1,         // MemTile has D3!

            // BD_6
            iteration_current: ((w6 >> 23) & 0x3F) as u8,
            iteration_wrap: (((w6 >> 17) & 0x3F) + 1) as u8,
            iteration_stepsize: (w6 & 0x1FFFF) + 1,

            // BD_7
            valid: (w7 >> 31) & 1 != 0,
            lock_rel_value: sign_extend_7bit(((w7 >> 24) & 0x7F) as u8),
            lock_rel_id: ((w7 >> 16) & 0xFF) as u8,  // 8 bits for MemTile!
            lock_acq_enable: (w7 >> 15) & 1 != 0,
            lock_acq_value: sign_extend_7bit(((w7 >> 8) & 0x7F) as u8),
            lock_acq_id: (w7 & 0xFF) as u8,          // 8 bits for MemTile!

            // Not used in MemTile
            burst_length: 0,
            smid: 0,
            axcache: 0,
            axqos: 0,
            secure_access: false,
        }
    }

    /// Parse Shim tile BD (8 registers per BD).
    ///
    /// Layout from AM025 NOC_MODULE/DMA/BD:
    /// - BD_0: Buffer_Length[31:0]
    /// - BD_1: Base_Address_Low[31:2], Reserved[1:0]
    /// - BD_2: Enable_Packet[30], OOO_ID[29:24], Packet_ID[23:19],
    ///         Packet_Type[18:16], Base_Address_High[15:0]
    /// - BD_3: Secure_Access[30], D0_Wrap[29:20], D0_Stepsize[19:0]
    /// - BD_4: Burst_Length[31:30], D1_Wrap[29:20], D1_Stepsize[19:0]
    /// - BD_5: SMID[31:28], AxCache[27:24], AxQoS[23:20], D2_Stepsize[19:0]
    /// - BD_6: Iteration_Current[31:26], Iteration_Wrap[25:20], Iteration_Stepsize[19:0]
    /// - BD_7: TLAST_Suppress[31], Next_BD[30:27], Use_Next_BD[26], Valid_BD[25],
    ///         Lock_Rel_Value[24:18], Lock_Rel_ID[16:13], Lock_Acq_Enable[12],
    ///         Lock_Acq_Value[11:5], Lock_Acq_ID[3:0]
    fn parse_shim(words: &[u32]) -> Self {
        assert!(words.len() >= BD_REGS_SHIM, "Shim BD needs 8 registers");

        let w0 = words[0];
        let w1 = words[1];
        let w2 = words[2];
        let w3 = words[3];
        let w4 = words[4];
        let w5 = words[5];
        let w6 = words[6];
        let w7 = words[7];

        // 46-bit word address from low and high parts
        let addr_low = (w1 >> 2) as u64;             // 30 bits
        let addr_high = (w2 & 0xFFFF) as u64;        // 16 bits
        let base_addr = addr_low | (addr_high << 30);

        Self {
            // BD_0
            length_words: w0,                         // Full 32 bits for DDR

            // BD_1 + BD_2 (address)
            base_addr_words: base_addr,

            // BD_2
            enable_packet: (w2 >> 30) & 1 != 0,
            ooo_bd_id: ((w2 >> 24) & 0x3F) as u8,
            packet_id: ((w2 >> 19) & 0x1F) as u8,
            packet_type: ((w2 >> 16) & 0x7) as u8,

            // BD_3
            secure_access: (w3 >> 30) & 1 != 0,
            d0_wrap: ((w3 >> 20) & 0x3FF) as u16,    // 10 bits
            d0_stepsize: (w3 & 0xFFFFF) + 1,         // 20 bits + 1 (up to 1M)

            // BD_4
            burst_length: ((w4 >> 30) & 0x3) as u8,
            d1_wrap: ((w4 >> 20) & 0x3FF) as u16,
            d1_stepsize: (w4 & 0xFFFFF) + 1,

            // BD_5
            smid: ((w5 >> 28) & 0xF) as u8,
            axcache: ((w5 >> 24) & 0xF) as u8,
            axqos: ((w5 >> 20) & 0xF) as u8,
            d2_stepsize: (w5 & 0xFFFFF) + 1,

            // BD_6
            iteration_current: ((w6 >> 26) & 0x3F) as u8,
            iteration_wrap: (((w6 >> 20) & 0x3F) + 1) as u8,
            iteration_stepsize: (w6 & 0xFFFFF) + 1,

            // BD_7
            tlast_suppress: (w7 >> 31) & 1 != 0,
            next_bd: ((w7 >> 27) & 0xF) as u8,
            use_next_bd: (w7 >> 26) & 1 != 0,
            valid: (w7 >> 25) & 1 != 0,
            lock_rel_value: sign_extend_7bit(((w7 >> 18) & 0x7F) as u8),
            lock_rel_id: ((w7 >> 13) & 0xF) as u8,
            lock_acq_enable: (w7 >> 12) & 1 != 0,
            lock_acq_value: sign_extend_7bit(((w7 >> 5) & 0x7F) as u8),
            lock_acq_id: (w7 & 0xF) as u8,

            // Not used in Shim
            d3_stepsize: 0,
            d2_wrap: 0,
            compression_enable: false,
            d0_zero_before: 0,
            d0_zero_after: 0,
            d1_zero_before: 0,
            d1_zero_after: 0,
            d2_zero_before: 0,
            d2_zero_after: 0,
        }
    }

    /// Parse BD from memory at the given BD slot.
    ///
    /// # Arguments
    /// * `memory` - Tile memory slice
    /// * `bd_id` - BD slot index
    /// * `tile_type` - Type of tile
    pub fn from_memory(memory: &[u8], bd_id: u8, tile_type: TileType) -> Self {
        let base = bd_base_address(tile_type);
        let offset = (base + bd_id as u64 * BD_SPACING) as usize;
        let reg_count = bd_register_count(tile_type);

        // Read register words from memory
        let mut words = Vec::with_capacity(reg_count);
        for i in 0..reg_count {
            let addr = offset + i * 4;
            if addr + 4 <= memory.len() {
                let word = u32::from_le_bytes([
                    memory[addr],
                    memory[addr + 1],
                    memory[addr + 2],
                    memory[addr + 3],
                ]);
                words.push(word);
            } else {
                words.push(0);
            }
        }

        Self::from_registers(&words, tile_type)
    }

    /// Convert byte address to word address
    pub fn byte_to_word_addr(byte_addr: u64) -> u64 {
        byte_addr / 4
    }

    /// Convert word address to byte address
    pub fn word_to_byte_addr(word_addr: u64) -> u64 {
        word_addr * 4
    }

    /// Get the byte address for this BD
    pub fn base_addr_bytes(&self) -> u64 {
        Self::word_to_byte_addr(self.base_addr_words)
    }

    /// Get the transfer length in bytes
    pub fn length_bytes(&self) -> u64 {
        self.length_words as u64 * 4
    }
}

/// Get BD base address for a tile type
pub fn bd_base_address(tile_type: TileType) -> u64 {
    match tile_type {
        TileType::Compute => BD_BASE_COMPUTE,
        TileType::MemTile => BD_BASE_MEMTILE,
        TileType::Shim => BD_BASE_SHIM,
    }
}

/// Get number of BDs for a tile type
pub fn bd_count(tile_type: TileType) -> usize {
    match tile_type {
        TileType::Compute => BD_COUNT_COMPUTE,
        TileType::MemTile => BD_COUNT_MEMTILE,
        TileType::Shim => BD_COUNT_SHIM,
    }
}

/// Get number of registers per BD for a tile type
pub fn bd_register_count(tile_type: TileType) -> usize {
    match tile_type {
        TileType::Compute => BD_REGS_COMPUTE,
        TileType::MemTile | TileType::Shim => BD_REGS_MEMTILE, // Both use 8
    }
}

/// Sign-extend a 7-bit value to i8
fn sign_extend_7bit(val: u8) -> i8 {
    if val & 0x40 != 0 {
        // Negative: extend sign
        (val | 0x80) as i8
    } else {
        val as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_extend_7bit() {
        assert_eq!(sign_extend_7bit(0), 0);
        assert_eq!(sign_extend_7bit(1), 1);
        assert_eq!(sign_extend_7bit(63), 63);
        assert_eq!(sign_extend_7bit(64), -64);  // 0x40 -> -64
        assert_eq!(sign_extend_7bit(127), -1);  // 0x7F -> -1
    }

    #[test]
    fn test_compute_bd_parsing() {
        // Sample BD registers for compute tile
        let words = [
            0x0004_1000,  // BD_0: base=0x10, length=0x1000
            0x4000_0000,  // BD_1: packet enabled
            0x0000_0001,  // BD_2: d0_step=2, d1_step=1
            0x0000_0000,  // BD_3: no wrap
            0x0000_0000,  // BD_4: no iteration
            0x0200_0000,  // BD_5: valid=1
        ];

        let bd = BufferDescriptor::from_registers(&words, TileType::Compute);

        assert!(bd.valid);
        assert_eq!(bd.base_addr_words, 0x10);
        assert_eq!(bd.length_words, 0x1000);
        assert!(bd.enable_packet);
        assert_eq!(bd.d0_stepsize, 2);  // stored 1 + 1 = 2
        assert_eq!(bd.d1_stepsize, 1);  // stored 0 + 1 = 1
    }

    #[test]
    fn test_memtile_bd_parsing() {
        // Sample BD registers for memtile
        // BD_1 layout: D0_Zero_Before[31:26], Next_BD[25:20], Use_Next_BD[19], Base_Address[18:0]
        // For next_bd=2: bit 21 (in Next_BD field) = 0x00200000
        // For use_next=1: bit 19 = 0x00080000
        // For addr=0x100: bits 18:0 = 0x00000100
        // Combined: 0x00280100
        let words = [
            0x8000_0400,  // BD_0: packet=1, length=0x400
            0x0028_0100,  // BD_1: next_bd=2, use_next=1, addr=0x100
            0x0000_0003,  // BD_2: d0_step=4
            0x0000_0000,  // BD_3
            0x0000_0000,  // BD_4
            0x0000_0000,  // BD_5
            0x0000_0000,  // BD_6
            0x8000_0000,  // BD_7: valid=1
        ];

        let bd = BufferDescriptor::from_registers(&words, TileType::MemTile);

        assert!(bd.valid);
        assert!(bd.enable_packet);
        assert_eq!(bd.length_words, 0x400);
        assert_eq!(bd.base_addr_words, 0x100);
        assert!(bd.use_next_bd);
        assert_eq!(bd.next_bd, 2);
        assert_eq!(bd.d0_stepsize, 4);
    }

    #[test]
    fn test_shim_bd_parsing() {
        // Sample BD registers for shim tile
        let words = [
            0x0000_1000,  // BD_0: length=0x1000
            0x0000_0400,  // BD_1: addr_low=0x100
            0x0000_0000,  // BD_2: addr_high=0
            0x0010_0003,  // BD_3: d0_wrap=1, d0_step=4
            0x0000_0000,  // BD_4
            0x0000_0000,  // BD_5
            0x0000_0000,  // BD_6
            0x0200_0000,  // BD_7: valid=1
        ];

        let bd = BufferDescriptor::from_registers(&words, TileType::Shim);

        assert!(bd.valid);
        assert_eq!(bd.length_words, 0x1000);
        assert_eq!(bd.base_addr_words, 0x100);
        assert_eq!(bd.d0_wrap, 1);
        assert_eq!(bd.d0_stepsize, 4);
    }

    #[test]
    fn test_bd_address_conversion() {
        assert_eq!(BufferDescriptor::byte_to_word_addr(0), 0);
        assert_eq!(BufferDescriptor::byte_to_word_addr(4), 1);
        assert_eq!(BufferDescriptor::byte_to_word_addr(1024), 256);

        assert_eq!(BufferDescriptor::word_to_byte_addr(0), 0);
        assert_eq!(BufferDescriptor::word_to_byte_addr(1), 4);
        assert_eq!(BufferDescriptor::word_to_byte_addr(256), 1024);
    }
}
