//! Stream I/O types and port mappings for AIE-ML DMA.
//!
//! This module defines:
//! - Stream word representation with TLAST signaling
//! - Packet header format per AM020 Table 2
//! - Stream switch port mappings for all tile types
//!
//! Reference: AMD AM020/AM025 and docs/dma-reference.md

/// A word on the stream with metadata.
///
/// This is the fundamental unit of data transfer between DMA and stream switch.
/// Each word carries 32 bits of payload plus control signals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamWord {
    /// 32-bit data payload
    pub data: u32,
    /// TLAST marker - true on last word of packet/transfer
    pub tlast: bool,
    /// Optional parity bit for error detection
    pub parity: bool,
}

impl StreamWord {
    /// Create a new stream word with data only
    pub fn new(data: u32) -> Self {
        Self {
            data,
            tlast: false,
            parity: false,
        }
    }

    /// Create a new stream word with TLAST set
    pub fn with_tlast(data: u32) -> Self {
        Self {
            data,
            tlast: true,
            parity: false,
        }
    }

    /// Set TLAST on this word
    pub fn set_tlast(mut self) -> Self {
        self.tlast = true;
        self
    }

    /// Compute odd parity for the data
    pub fn compute_parity(data: u32) -> bool {
        data.count_ones() % 2 == 1
    }

    /// Create with computed parity
    pub fn with_parity(data: u32) -> Self {
        Self {
            data,
            tlast: false,
            parity: Self::compute_parity(data),
        }
    }
}

// ============================================================================
// Stream Switch Port Mappings
// ============================================================================
//
// These constants define how DMA channels connect to stream switch ports.
// Master ports carry data OUT from the switch, Slave ports carry data IN.

/// Compute Tile port mappings
pub mod compute {
    /// Number of master ports (data OUT from switch)
    pub const MASTER_PORT_COUNT: usize = 23;
    /// Number of slave ports (data IN to switch)
    pub const SLAVE_PORT_COUNT: usize = 25;

    /// Master ports (switch outputs TO these destinations)
    pub mod master {
        pub const CORE: u8 = 0;
        pub const DMA_0: u8 = 1;       // S2MM ch0 receives FROM switch
        pub const DMA_1: u8 = 2;       // S2MM ch1 receives FROM switch
        pub const TILE_CTRL: u8 = 3;
        pub const FIFO_0: u8 = 4;
        pub const SOUTH_0: u8 = 5;
        pub const SOUTH_1: u8 = 6;
        pub const SOUTH_2: u8 = 7;
        pub const SOUTH_3: u8 = 8;
        pub const WEST_0: u8 = 9;
        pub const WEST_1: u8 = 10;
        pub const WEST_2: u8 = 11;
        pub const WEST_3: u8 = 12;
        pub const NORTH_0: u8 = 13;
        pub const NORTH_1: u8 = 14;
        pub const NORTH_2: u8 = 15;
        pub const NORTH_3: u8 = 16;
        pub const NORTH_4: u8 = 17;
        pub const NORTH_5: u8 = 18;
        pub const EAST_0: u8 = 19;
        pub const EAST_1: u8 = 20;
        pub const EAST_2: u8 = 21;
        pub const EAST_3: u8 = 22;
    }

    /// Slave ports (switch receives FROM these sources)
    pub mod slave {
        pub const CORE: u8 = 0;
        pub const DMA_0: u8 = 1;       // MM2S ch0 sends TO switch
        pub const DMA_1: u8 = 2;       // MM2S ch1 sends TO switch
        pub const TILE_CTRL: u8 = 3;
        pub const FIFO_0: u8 = 4;
        pub const SOUTH_0: u8 = 5;
        pub const SOUTH_1: u8 = 6;
        pub const SOUTH_2: u8 = 7;
        pub const SOUTH_3: u8 = 8;
        pub const SOUTH_4: u8 = 9;
        pub const SOUTH_5: u8 = 10;
        pub const WEST_0: u8 = 11;
        pub const WEST_1: u8 = 12;
        pub const WEST_2: u8 = 13;
        pub const WEST_3: u8 = 14;
        pub const NORTH_0: u8 = 15;
        pub const NORTH_1: u8 = 16;
        pub const NORTH_2: u8 = 17;
        pub const NORTH_3: u8 = 18;
        pub const EAST_0: u8 = 19;
        pub const EAST_1: u8 = 20;
        pub const EAST_2: u8 = 21;
        pub const EAST_3: u8 = 22;
        pub const AIE_TRACE: u8 = 23;
        pub const MEM_TRACE: u8 = 24;
    }

    /// Get master port for S2MM channel (receives data FROM switch)
    pub fn s2mm_master_port(channel: u8) -> u8 {
        match channel {
            0 => master::DMA_0,
            1 => master::DMA_1,
            _ => panic!("Invalid S2MM channel {} for compute tile", channel),
        }
    }

    /// Get slave port for MM2S channel (sends data TO switch)
    pub fn mm2s_slave_port(channel: u8) -> u8 {
        match channel {
            0 => slave::DMA_0,
            1 => slave::DMA_1,
            _ => panic!("Invalid MM2S channel {} for compute tile", channel),
        }
    }
}

/// MemTile port mappings
pub mod memtile {
    /// Number of master ports
    pub const MASTER_PORT_COUNT: usize = 17;
    /// Number of slave ports
    pub const SLAVE_PORT_COUNT: usize = 18;

    /// Master ports
    pub mod master {
        pub const DMA_0: u8 = 0;       // S2MM ch0
        pub const DMA_1: u8 = 1;       // S2MM ch1
        pub const DMA_2: u8 = 2;       // S2MM ch2
        pub const DMA_3: u8 = 3;       // S2MM ch3
        pub const DMA_4: u8 = 4;       // S2MM ch4
        pub const DMA_5: u8 = 5;       // S2MM ch5
        pub const TILE_CTRL: u8 = 6;
        pub const SOUTH_0: u8 = 7;
        pub const SOUTH_1: u8 = 8;
        pub const SOUTH_2: u8 = 9;
        pub const SOUTH_3: u8 = 10;
        pub const NORTH_0: u8 = 11;
        pub const NORTH_1: u8 = 12;
        pub const NORTH_2: u8 = 13;
        pub const NORTH_3: u8 = 14;
        pub const NORTH_4: u8 = 15;
        pub const NORTH_5: u8 = 16;
    }

    /// Slave ports
    pub mod slave {
        pub const DMA_0: u8 = 0;       // MM2S ch0
        pub const DMA_1: u8 = 1;       // MM2S ch1
        pub const DMA_2: u8 = 2;       // MM2S ch2
        pub const DMA_3: u8 = 3;       // MM2S ch3
        pub const DMA_4: u8 = 4;       // MM2S ch4
        pub const DMA_5: u8 = 5;       // MM2S ch5
        pub const TILE_CTRL: u8 = 6;
        pub const SOUTH_0: u8 = 7;
        pub const SOUTH_1: u8 = 8;
        pub const SOUTH_2: u8 = 9;
        pub const SOUTH_3: u8 = 10;
        pub const SOUTH_4: u8 = 11;
        pub const SOUTH_5: u8 = 12;
        pub const NORTH_0: u8 = 13;
        pub const NORTH_1: u8 = 14;
        pub const NORTH_2: u8 = 15;
        pub const NORTH_3: u8 = 16;
        pub const TRACE: u8 = 17;
    }

    /// Get master port for S2MM channel
    pub fn s2mm_master_port(channel: u8) -> u8 {
        assert!(channel < 6, "Invalid S2MM channel {} for memtile", channel);
        channel // DMA_0 through DMA_5 are ports 0-5
    }

    /// Get slave port for MM2S channel
    pub fn mm2s_slave_port(channel: u8) -> u8 {
        assert!(channel < 6, "Invalid MM2S channel {} for memtile", channel);
        channel // DMA_0 through DMA_5 are ports 0-5
    }
}

/// Shim Tile port mappings
pub mod shim {
    /// Master ports
    pub mod master {
        pub const FIFO: u8 = 0;
        pub const TILE_CTRL: u8 = 1;
        pub const SOUTH_0: u8 = 2;
        pub const SOUTH_1: u8 = 3;
        pub const SOUTH_2: u8 = 4;
        pub const SOUTH_3: u8 = 5;
        pub const SOUTH_4: u8 = 6;
        pub const SOUTH_5: u8 = 7;
        pub const WEST_0: u8 = 8;
        pub const WEST_1: u8 = 9;
        pub const WEST_2: u8 = 10;
        pub const WEST_3: u8 = 11;
        pub const NORTH_0: u8 = 12;
        pub const NORTH_1: u8 = 13;
        pub const NORTH_2: u8 = 14;
        pub const NORTH_3: u8 = 15;
        pub const NORTH_4: u8 = 16;
        pub const NORTH_5: u8 = 17;
        pub const EAST_0: u8 = 18;
        pub const EAST_1: u8 = 19;
        pub const EAST_2: u8 = 20;
        pub const EAST_3: u8 = 21;
    }

    /// Slave ports
    pub mod slave {
        pub const FIFO: u8 = 0;
        pub const TILE_CTRL: u8 = 1;
        pub const SOUTH_0: u8 = 2;
        pub const SOUTH_1: u8 = 3;
        pub const SOUTH_2: u8 = 4;
        pub const SOUTH_3: u8 = 5;
        pub const SOUTH_4: u8 = 6;
        pub const SOUTH_5: u8 = 7;
        pub const SOUTH_6: u8 = 8;
        pub const SOUTH_7: u8 = 9;
        pub const WEST_0: u8 = 10;
        pub const WEST_1: u8 = 11;
        pub const WEST_2: u8 = 12;
        pub const WEST_3: u8 = 13;
        pub const NORTH_0: u8 = 14;
        pub const NORTH_1: u8 = 15;
        pub const NORTH_2: u8 = 16;
        pub const NORTH_3: u8 = 17;
        pub const EAST_0: u8 = 18;
        pub const EAST_1: u8 = 19;
        pub const EAST_2: u8 = 20;
        pub const EAST_3: u8 = 21;
        pub const TRACE: u8 = 22;
    }
}

// ============================================================================
// Inter-Tile Connection Mappings
// ============================================================================

/// Inter-tile stream connections (which ports connect between adjacent tiles)
pub mod connections {
    /// Shim North masters [12-17] connect to MemTile South slaves [7-12]
    pub fn shim_north_to_memtile_south(shim_master: u8) -> Option<u8> {
        if (12..=17).contains(&shim_master) {
            Some(shim_master - 12 + 7) // 12->7, 13->8, ..., 17->12
        } else {
            None
        }
    }

    /// MemTile North masters [11-16] connect to Compute South slaves [5-10]
    pub fn memtile_north_to_compute_south(memtile_master: u8) -> Option<u8> {
        if (11..=16).contains(&memtile_master) {
            Some(memtile_master - 11 + 5) // 11->5, 12->6, ..., 16->10
        } else {
            None
        }
    }

    /// Compute South masters [5-8] connect to MemTile North slaves [13-16]
    pub fn compute_south_to_memtile_north(compute_master: u8) -> Option<u8> {
        if (5..=8).contains(&compute_master) {
            Some(compute_master - 5 + 13) // 5->13, 6->14, 7->15, 8->16
        } else {
            None
        }
    }

    /// MemTile South masters [7-10] connect to Shim North slaves [14-17]
    pub fn memtile_south_to_shim_north(memtile_master: u8) -> Option<u8> {
        if (7..=10).contains(&memtile_master) {
            Some(memtile_master - 7 + 14) // 7->14, 8->15, 9->16, 10->17
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_word() {
        let word = StreamWord::new(0x12345678);
        assert_eq!(word.data, 0x12345678);
        assert!(!word.tlast);

        let tlast_word = StreamWord::with_tlast(0xDEADBEEF);
        assert!(tlast_word.tlast);
    }

    #[test]
    fn test_compute_port_mappings() {
        // S2MM channels receive from master ports
        assert_eq!(compute::s2mm_master_port(0), compute::master::DMA_0);
        assert_eq!(compute::s2mm_master_port(1), compute::master::DMA_1);

        // MM2S channels send to slave ports
        assert_eq!(compute::mm2s_slave_port(0), compute::slave::DMA_0);
        assert_eq!(compute::mm2s_slave_port(1), compute::slave::DMA_1);
    }

    #[test]
    fn test_memtile_port_mappings() {
        // MemTile has 6 DMA channels each direction
        for ch in 0..6 {
            assert_eq!(memtile::s2mm_master_port(ch), ch);
            assert_eq!(memtile::mm2s_slave_port(ch), ch);
        }
    }

    #[test]
    fn test_inter_tile_connections() {
        // Shim North [12-17] -> MemTile South [7-12]
        assert_eq!(connections::shim_north_to_memtile_south(12), Some(7));
        assert_eq!(connections::shim_north_to_memtile_south(17), Some(12));
        assert_eq!(connections::shim_north_to_memtile_south(11), None);

        // MemTile North [11-16] -> Compute South [5-10]
        assert_eq!(connections::memtile_north_to_compute_south(11), Some(5));
        assert_eq!(connections::memtile_north_to_compute_south(16), Some(10));
    }
}
