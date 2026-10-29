//! Stream I/O types and port mappings for AIE-ML DMA.
//!
//! This module defines:
//! - Stream word representation with TLAST signaling
//! - DMA-to-stream-switch port mapping functions (derived from generated data)
//! - Inter-tile connection mappings (derived from generated range constants)
//!
//! Port layouts and ranges are generated at build time from AM025 register
//! definitions in `gen_stream_ports.rs` and `gen_stream_ranges.rs`, included
//! via `xdna_archspec::aie2`. This module provides DMA-specific convenience functions
//! on top of that generated data.
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
        Self { data, tlast: false, parity: false }
    }

    /// Create a new stream word with TLAST set
    pub fn with_tlast(data: u32) -> Self {
        Self { data, tlast: true, parity: false }
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
        Self { data, tlast: false, parity: Self::compute_parity(data) }
    }
}

// ============================================================================
// DMA Port Mapping Functions
// ============================================================================
//
// These map DMA channel indices to stream switch port indices. Port layouts
// are defined by the generated arrays in xdna_archspec::aie2 (from AM025).

/// Compute tile DMA port mappings (derived from gen_stream_ranges.rs).
pub mod compute {
    use xdna_archspec::aie2::stream_switch::compute as ranges;

    /// Port count derived from generated port arrays.
    pub const MASTER_PORT_COUNT: usize = xdna_archspec::aie2::COMPUTE_MASTER_PORTS.len();
    /// Port count derived from generated port arrays.
    pub const SLAVE_PORT_COUNT: usize = xdna_archspec::aie2::COMPUTE_SLAVE_PORTS.len();

    /// Get master port for S2MM channel (receives data FROM switch).
    /// S2MM channels map to DMA master ports starting at DMA_MASTER_START.
    pub fn s2mm_master_port(channel: u8) -> u8 {
        assert!(
            ranges::DMA_MASTER_START + channel <= ranges::DMA_MASTER_END,
            "Invalid S2MM channel {} for compute tile (DMA master range {}-{})",
            channel,
            ranges::DMA_MASTER_START,
            ranges::DMA_MASTER_END
        );
        ranges::DMA_MASTER_START + channel
    }

    /// Get slave port for MM2S channel (sends data TO switch).
    /// MM2S channels map to DMA slave ports starting at DMA_SLAVE_START.
    pub fn mm2s_slave_port(channel: u8) -> u8 {
        assert!(
            ranges::DMA_SLAVE_START + channel <= ranges::DMA_SLAVE_END,
            "Invalid MM2S channel {} for compute tile (DMA slave range {}-{})",
            channel,
            ranges::DMA_SLAVE_START,
            ranges::DMA_SLAVE_END
        );
        ranges::DMA_SLAVE_START + channel
    }
}

/// MemTile DMA port mappings (derived from gen_stream_ranges.rs).
pub mod memtile {
    use xdna_archspec::aie2::stream_switch::mem_tile as ranges;

    /// Port count derived from generated port arrays.
    pub const MASTER_PORT_COUNT: usize = xdna_archspec::aie2::MEMTILE_MASTER_PORTS.len();
    /// Port count derived from generated port arrays.
    pub const SLAVE_PORT_COUNT: usize = xdna_archspec::aie2::MEMTILE_SLAVE_PORTS.len();

    /// Get master port for S2MM channel.
    /// MemTile DMA master ports start at DMA_MASTER_START (0).
    pub fn s2mm_master_port(channel: u8) -> u8 {
        assert!(
            ranges::DMA_MASTER_START + channel <= ranges::DMA_MASTER_END,
            "Invalid S2MM channel {} for memtile (DMA master range {}-{})",
            channel,
            ranges::DMA_MASTER_START,
            ranges::DMA_MASTER_END
        );
        ranges::DMA_MASTER_START + channel
    }

    /// Get slave port for MM2S channel.
    /// MemTile DMA slave ports start at DMA_SLAVE_START (0).
    pub fn mm2s_slave_port(channel: u8) -> u8 {
        assert!(
            ranges::DMA_SLAVE_START + channel <= ranges::DMA_SLAVE_END,
            "Invalid MM2S channel {} for memtile (DMA slave range {}-{})",
            channel,
            ranges::DMA_SLAVE_START,
            ranges::DMA_SLAVE_END
        );
        ranges::DMA_SLAVE_START + channel
    }
}

/// Shim tile DMA port mappings.
///
/// Shim DMA ports come through the shim_mux (not the switchbox), so
/// gen_stream_ranges.rs does not have DMA_MASTER/SLAVE ranges for shim.
/// The DMA channels map to South-facing switchbox ports for NoC access.
pub mod shim {
    use xdna_archspec::aie2::stream_switch::shim as ranges;

    /// Get master port for S2MM channel (shim receives from NoC via South).
    pub fn s2mm_master_port(channel: u8) -> u8 {
        assert!(
            ranges::SOUTH_MASTER_START + channel <= ranges::SOUTH_MASTER_END,
            "Invalid S2MM channel {} for shim (South master range {}-{})",
            channel,
            ranges::SOUTH_MASTER_START,
            ranges::SOUTH_MASTER_END
        );
        ranges::SOUTH_MASTER_START + channel
    }

    /// Get slave port for MM2S channel (shim sends to NoC via South).
    pub fn mm2s_slave_port(channel: u8) -> u8 {
        assert!(
            ranges::SOUTH_SLAVE_START + channel <= ranges::SOUTH_SLAVE_END,
            "Invalid MM2S channel {} for shim (South slave range {}-{})",
            channel,
            ranges::SOUTH_SLAVE_START,
            ranges::SOUTH_SLAVE_END
        );
        ranges::SOUTH_SLAVE_START + channel
    }
}

// ============================================================================
// Inter-Tile Connection Mappings
// ============================================================================
//
// These define how ports on adjacent tiles connect. All range values come
// from gen_stream_ranges.rs (derived from AM025 port layout arrays).

/// Inter-tile stream connections (which ports connect between adjacent tiles).
pub mod connections {
    use xdna_archspec::aie2::stream_switch::{shim, mem_tile, compute};

    /// Shim North masters connect to MemTile South slaves.
    pub fn shim_north_to_memtile_south(shim_master: u8) -> Option<u8> {
        let range = shim::NORTH_MASTER_START..=shim::NORTH_MASTER_END;
        if range.contains(&shim_master) {
            Some(shim_master - shim::NORTH_MASTER_START + mem_tile::SOUTH_SLAVE_START)
        } else {
            None
        }
    }

    /// MemTile North masters connect to Compute South slaves.
    pub fn memtile_north_to_compute_south(memtile_master: u8) -> Option<u8> {
        let range = mem_tile::NORTH_MASTER_START..=mem_tile::NORTH_MASTER_END;
        if range.contains(&memtile_master) {
            Some(memtile_master - mem_tile::NORTH_MASTER_START + compute::SOUTH_SLAVE_START)
        } else {
            None
        }
    }

    /// Compute South masters connect to MemTile North slaves.
    pub fn compute_south_to_memtile_north(compute_master: u8) -> Option<u8> {
        let range = compute::SOUTH_MASTER_START..=compute::SOUTH_MASTER_END;
        if range.contains(&compute_master) {
            Some(compute_master - compute::SOUTH_MASTER_START + mem_tile::NORTH_SLAVE_START)
        } else {
            None
        }
    }

    /// MemTile South masters connect to Shim North slaves.
    pub fn memtile_south_to_shim_north(memtile_master: u8) -> Option<u8> {
        let range = mem_tile::SOUTH_MASTER_START..=mem_tile::SOUTH_MASTER_END;
        if range.contains(&memtile_master) {
            Some(memtile_master - mem_tile::SOUTH_MASTER_START + shim::NORTH_SLAVE_START)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xdna_archspec::aie2 as arch;

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
        // S2MM channels map to DMA master ports (1, 2)
        assert_eq!(compute::s2mm_master_port(0), 1);
        assert_eq!(compute::s2mm_master_port(1), 2);

        // MM2S channels map to DMA slave ports (1, 2)
        assert_eq!(compute::mm2s_slave_port(0), 1);
        assert_eq!(compute::mm2s_slave_port(1), 2);
    }

    #[test]
    fn test_compute_port_counts_match_generated() {
        assert_eq!(compute::MASTER_PORT_COUNT, arch::COMPUTE_MASTER_PORTS.len());
        assert_eq!(compute::SLAVE_PORT_COUNT, arch::COMPUTE_SLAVE_PORTS.len());
    }

    #[test]
    fn test_memtile_port_mappings() {
        // MemTile DMA ports start at 0
        for ch in 0..6 {
            assert_eq!(memtile::s2mm_master_port(ch), ch);
            assert_eq!(memtile::mm2s_slave_port(ch), ch);
        }
    }

    #[test]
    fn test_memtile_port_counts_match_generated() {
        assert_eq!(memtile::MASTER_PORT_COUNT, arch::MEMTILE_MASTER_PORTS.len());
        assert_eq!(memtile::SLAVE_PORT_COUNT, arch::MEMTILE_SLAVE_PORTS.len());
    }

    #[test]
    fn test_shim_port_mappings() {
        // Shim DMA ports map to South ports starting at SOUTH_MASTER/SLAVE_START
        use arch::stream_switch::shim as ranges;
        assert_eq!(shim::s2mm_master_port(0), ranges::SOUTH_MASTER_START);
        assert_eq!(shim::mm2s_slave_port(0), ranges::SOUTH_SLAVE_START);
    }

    #[test]
    fn test_inter_tile_connections() {
        use arch::stream_switch::{shim as sr, mem_tile as mr, compute as cr};

        // Shim North -> MemTile South
        assert_eq!(
            connections::shim_north_to_memtile_south(sr::NORTH_MASTER_START),
            Some(mr::SOUTH_SLAVE_START)
        );
        assert_eq!(connections::shim_north_to_memtile_south(sr::NORTH_MASTER_END), Some(mr::SOUTH_SLAVE_END));
        assert_eq!(connections::shim_north_to_memtile_south(sr::NORTH_MASTER_START - 1), None);

        // MemTile North -> Compute South
        assert_eq!(
            connections::memtile_north_to_compute_south(mr::NORTH_MASTER_START),
            Some(cr::SOUTH_SLAVE_START)
        );
        assert_eq!(
            connections::memtile_north_to_compute_south(mr::NORTH_MASTER_END),
            Some(cr::SOUTH_SLAVE_END)
        );
    }

    /// Validate that generated port ranges produce the same mappings as the
    /// previously hardcoded values. This catches any drift between AM025 data
    /// and the expected hardware behavior.
    #[test]
    fn test_backward_compatible_values() {
        // Compute: DMA master ports at 1,2; DMA slave ports at 1,2
        assert_eq!(compute::s2mm_master_port(0), 1);
        assert_eq!(compute::s2mm_master_port(1), 2);
        assert_eq!(compute::mm2s_slave_port(0), 1);
        assert_eq!(compute::mm2s_slave_port(1), 2);

        // MemTile: DMA at 0-5 for both directions
        assert_eq!(memtile::s2mm_master_port(0), 0);
        assert_eq!(memtile::s2mm_master_port(5), 5);

        // Shim: South ports at 2+
        assert_eq!(shim::s2mm_master_port(0), 2);
        assert_eq!(shim::mm2s_slave_port(0), 2);

        // Inter-tile: Shim North [12-17] -> MemTile South [7-12]
        assert_eq!(connections::shim_north_to_memtile_south(12), Some(7));
        assert_eq!(connections::shim_north_to_memtile_south(17), Some(12));

        // Inter-tile: MemTile North [11-16] -> Compute South [5-10]
        assert_eq!(connections::memtile_north_to_compute_south(11), Some(5));
        assert_eq!(connections::memtile_north_to_compute_south(16), Some(10));
    }
}
