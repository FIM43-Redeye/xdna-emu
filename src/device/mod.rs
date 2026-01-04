//! Device models and register definitions for AMD XDNA NPUs.
//!
//! This module provides:
//! - Architecture definitions for NPU variants (AIE2, AIE2P)
//! - Tile state representation (memory, locks, DMA, core)
//! - Tile array (the complete device)
//! - Register definitions for address decoding
//!
//! # Architecture Overview
//!
//! AMD XDNA NPUs use a tile-based architecture:
//!
//! ```text
//!     Col 0    Col 1    Col 2    Col 3    Col 4
//!   +--------+--------+--------+--------+--------+
//! 5 |Compute |Compute |Compute |Compute |Compute |  <- Row 5
//!   +--------+--------+--------+--------+--------+
//! 4 |Compute |Compute |Compute |Compute |Compute |
//!   +--------+--------+--------+--------+--------+
//! 3 |Compute |Compute |Compute |Compute |Compute |
//!   +--------+--------+--------+--------+--------+
//! 2 |Compute |Compute |Compute |Compute |Compute |
//!   +--------+--------+--------+--------+--------+
//! 1 |MemTile |MemTile |MemTile |MemTile |MemTile |  <- 512KB each
//!   +--------+--------+--------+--------+--------+
//! 0 | Shim   | Shim   | Shim   | Shim   | Shim   |  <- DDR interface
//!   +--------+--------+--------+--------+--------+
//! ```
//!
//! # Example
//!
//! ```
//! use xdna_emu::device::{TileArray, TileAddress, RegisterInfo};
//!
//! // Create NPU1 device
//! let mut array = TileArray::npu1();
//! assert_eq!(array.cols(), 5);
//! assert_eq!(array.rows(), 6);
//!
//! // Access a compute tile
//! let tile = array.tile_mut(1, 2);
//! tile.write_data(0x100, &[0xDE, 0xAD, 0xBE, 0xEF]);
//!
//! // Decode a CDO address
//! let addr = TileAddress::decode(0x02232000);
//! assert_eq!(addr.col, 1);
//! assert_eq!(addr.row, 2);
//! ```

pub mod aie2_spec;
pub mod registers;
pub mod registers_spec;
pub mod tile;
pub mod array;
pub mod state;
pub mod host_memory;
pub mod dma;
pub mod stream_switch;
pub mod stream_router;

pub use registers::{RegisterInfo, RegisterModule, TileAddress};
pub use tile::{Tile, TileType, Lock, LockResult, DmaBufferDescriptor, DmaChannel, CoreState};
pub use array::TileArray;
pub use state::{DeviceState, CdoStats};
pub use host_memory::{HostMemory, HostMemoryError, MemoryRegion, DataDirection};
pub use dma::{
    DmaEngine, BdConfig, ChannelState, ChannelId, ChannelType,
    DmaResult, DmaError, AddressGenerator, DimensionConfig,
    Transfer, TransferState, TransferDirection, TransferEndpoint,
    DmaTimingConfig, ChannelTimingState, TransferPhase, ChannelArbiter,
    StreamData,
};
pub use stream_switch::{StreamSwitch, StreamPort, StreamPacket, PortDirection, PortType};
pub use stream_router::{StreamRouter, StreamWord, StreamFifo, PortId, Route, RouterStats};

/// AIE architecture variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AieArch {
    /// AIE2 (AI Engine ML) - Phoenix/HawkPoint NPU
    Aie2,
    /// AIE2P (AI Engine ML+) - Strix/Krackan NPU
    Aie2P,
}

impl AieArch {
    /// Get the number of columns for this architecture
    pub fn columns(&self) -> u8 {
        match self {
            AieArch::Aie2 => 5,  // NPU1: cols 0-4 (col 0 is shim)
            AieArch::Aie2P => 5, // NPU2: same base, can be larger
        }
    }

    /// Get the number of rows for this architecture
    pub fn rows(&self) -> u8 {
        match self {
            AieArch::Aie2 => 6,  // rows 0-5 (row 0 is shim, row 1 is mem tile)
            AieArch::Aie2P => 6,
        }
    }

    /// Check if a tile position is valid
    pub fn is_valid_tile(&self, col: u8, row: u8) -> bool {
        col < self.columns() && row < self.rows()
    }

    /// Check if a tile is a compute tile (has AIE core)
    pub fn is_compute_tile(&self, _col: u8, row: u8) -> bool {
        // Rows 2-5 are compute tiles
        row >= 2
    }

    /// Check if a tile is a memory tile
    pub fn is_mem_tile(&self, _col: u8, row: u8) -> bool {
        // Row 1 is memory tile
        row == 1
    }

    /// Check if a tile is a shim tile
    pub fn is_shim_tile(&self, _col: u8, row: u8) -> bool {
        // Row 0 is shim tile
        row == 0
    }
}

impl std::fmt::Display for AieArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AieArch::Aie2 => write!(f, "AIE2 (NPU1)"),
            AieArch::Aie2P => write!(f, "AIE2P (NPU2+)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aie_arch_dimensions() {
        let aie2 = AieArch::Aie2;
        assert_eq!(aie2.columns(), 5);
        assert_eq!(aie2.rows(), 6);
    }

    #[test]
    fn test_tile_classification() {
        let arch = AieArch::Aie2;

        // Shim tiles (row 0)
        assert!(arch.is_shim_tile(0, 0));
        assert!(arch.is_shim_tile(1, 0));

        // Mem tiles (row 1)
        assert!(arch.is_mem_tile(0, 1));
        assert!(arch.is_mem_tile(1, 1));

        // Compute tiles (rows 2-5)
        assert!(arch.is_compute_tile(0, 2));
        assert!(arch.is_compute_tile(1, 5));

        // Not compute in lower rows
        assert!(!arch.is_compute_tile(0, 0));
        assert!(!arch.is_compute_tile(0, 1));
    }
}
