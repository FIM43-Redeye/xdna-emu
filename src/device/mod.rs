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

pub mod banking;
pub mod arch_config;
pub mod model;
pub mod regdb;
pub mod registers;
pub mod registers_spec;
pub mod tile;
pub mod array;
pub mod state;
pub mod host_memory;
pub mod dma;
pub mod stream_switch;
pub mod trace_unit;
pub mod aiert_validation;

pub use arch_config::{ArchConfig, ModelConfig, default_arch};
pub use registers::{RegisterInfo, TileAddress};
pub use tile::{Tile, TileType, Lock, LockResult, DmaBufferDescriptor, DmaChannel, CoreState};
pub use array::TileArray;
pub use state::{DeviceState, CdoStats};
pub use host_memory::{HostMemory, HostMemoryError, MemoryRegion, DataDirection};
pub use dma::{
    DmaEngine, BdConfig, ChannelState, ChannelId, ChannelType,
    DmaResult, DmaError, AddressGenerator, DimensionConfig,
    Transfer, TransferDirection, TransferEndpoint,
    DmaTimingConfig, StreamData,
};
pub use stream_switch::{StreamSwitch, StreamPort, StreamPacket, PortDirection, PortType};

// Architecture variant enum removed -- use ArchConfig trait or crate::arch::*
// constants for architecture-specific values.
