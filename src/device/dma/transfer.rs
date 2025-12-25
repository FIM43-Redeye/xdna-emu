//! DMA transfer state machine.
//!
//! A transfer represents an active DMA operation from a single buffer
//! descriptor. It tracks the current position, handles multi-dimensional
//! addressing, and manages the transfer lifecycle.
//!
//! # Transfer Lifecycle
//!
//! ```text
//! ┌─────────┐   lock    ┌───────────────┐   lock    ┌──────────┐
//! │ Created ├──acquire──► WaitingForLock├──acquired─► Active   │
//! └─────────┘           └───────────────┘           └────┬─────┘
//!                                                        │
//!                        ┌───────────────┐   all data   │
//!                        │   Complete    ◄──transferred─┘
//!                        └───────┬───────┘
//!                                │ lock release
//!                                ▼
//!                        ┌───────────────┐
//!                        │   Finished    │
//!                        └───────────────┘
//! ```
//!
//! # Data Flow
//!
//! For MM2S (Memory to Stream):
//! 1. Read data from tile memory at current address
//! 2. Send to stream switch
//! 3. Advance address generator
//!
//! For S2MM (Stream to Memory):
//! 1. Receive data from stream switch
//! 2. Write to tile memory at current address
//! 3. Advance address generator

use super::addressing::AddressGenerator;
use super::{BdConfig, DmaError};

/// Transfer state in the state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferState {
    /// Transfer created but not yet started
    Created,
    /// Waiting to acquire lock before transfer
    WaitingForLock(u8),
    /// Transfer is actively moving data
    Active,
    /// All data transferred, waiting to release lock
    ReleasingLock(u8),
    /// Transfer complete
    Complete,
    /// Transfer failed with error
    Error,
}

/// Direction of a DMA transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Stream to Memory: receive from stream, write to tile memory
    S2MM,
    /// Memory to Stream: read from tile memory, send to stream
    MM2S,
}

/// Endpoint for a transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferEndpoint {
    /// Tile local memory (compute tile: 64KB, mem tile: 512KB)
    TileMemory {
        col: u8,
        row: u8,
    },
    /// Host/DDR memory (via shim tile)
    HostMemory,
    /// Stream switch connection
    Stream {
        port: u8,
    },
}

/// An active DMA transfer.
#[derive(Debug)]
pub struct Transfer {
    /// Source of data
    pub source: TransferEndpoint,

    /// Destination of data
    pub dest: TransferEndpoint,

    /// Transfer direction
    pub direction: TransferDirection,

    /// Current state
    pub state: TransferState,

    /// Address generator for multi-dimensional addressing
    pub address_gen: AddressGenerator,

    /// BD index this transfer was created from
    pub bd_index: u8,

    /// Channel performing this transfer
    pub channel: u8,

    /// Lock to acquire before transfer (if any)
    pub acquire_lock: Option<u8>,

    /// Lock value to acquire
    pub acquire_value: i8,

    /// Lock to release after transfer (if any)
    pub release_lock: Option<u8>,

    /// Lock value after release
    pub release_value: i8,

    /// Next BD to chain to (if any)
    pub next_bd: Option<u8>,

    /// Bytes transferred so far
    pub bytes_transferred: u64,

    /// Total bytes to transfer
    pub total_bytes: u64,

    /// Cycle count for this transfer (for timing)
    pub cycles_elapsed: u64,

    /// Last error (if any)
    pub error: Option<DmaError>,
}

impl Transfer {
    /// Create a new transfer from a BD configuration.
    pub fn new(
        bd_config: &BdConfig,
        bd_index: u8,
        channel: u8,
        direction: TransferDirection,
        tile_col: u8,
        tile_row: u8,
    ) -> Result<Self, DmaError> {
        if !bd_config.valid {
            return Err(DmaError::BdNotValid(bd_index));
        }

        // Create address generator based on BD dimensions
        let address_gen = AddressGenerator::new(
            bd_config.base_addr,
            [
                bd_config.d0,
                bd_config.d1,
                bd_config.d2,
                bd_config.d3,
            ],
        );

        // Determine endpoints based on direction
        let (source, dest) = match direction {
            TransferDirection::S2MM => (
                TransferEndpoint::Stream { port: channel },
                TransferEndpoint::TileMemory { col: tile_col, row: tile_row },
            ),
            TransferDirection::MM2S => (
                TransferEndpoint::TileMemory { col: tile_col, row: tile_row },
                TransferEndpoint::Stream { port: channel },
            ),
        };

        let initial_state = if bd_config.acquire_lock.is_some() {
            TransferState::WaitingForLock(bd_config.acquire_lock.unwrap())
        } else {
            TransferState::Active
        };

        Ok(Self {
            source,
            dest,
            direction,
            state: initial_state,
            address_gen,
            bd_index,
            channel,
            acquire_lock: bd_config.acquire_lock,
            acquire_value: bd_config.acquire_value,
            release_lock: bd_config.release_lock,
            release_value: bd_config.release_value,
            next_bd: bd_config.next_bd,
            bytes_transferred: 0,
            total_bytes: bd_config.length as u64,
            cycles_elapsed: 0,
            error: None,
        })
    }

    /// Create a simple memory-to-memory transfer (for testing).
    pub fn new_mem_copy(
        source_col: u8,
        source_row: u8,
        dest_col: u8,
        dest_row: u8,
        base_addr: u64,
        length: u32,
    ) -> Self {
        let address_gen = AddressGenerator::new_1d(base_addr, length, 1);

        Self {
            source: TransferEndpoint::TileMemory { col: source_col, row: source_row },
            dest: TransferEndpoint::TileMemory { col: dest_col, row: dest_row },
            direction: TransferDirection::MM2S, // Arbitrary for mem-to-mem
            state: TransferState::Active,
            address_gen,
            bd_index: 0,
            channel: 0,
            acquire_lock: None,
            acquire_value: 0,
            release_lock: None,
            release_value: 0,
            next_bd: None,
            bytes_transferred: 0,
            total_bytes: length as u64,
            cycles_elapsed: 0,
            error: None,
        }
    }

    /// Create a host-to-tile transfer (for loading data).
    pub fn new_host_to_tile(
        tile_col: u8,
        tile_row: u8,
        _host_addr: u64,
        tile_addr: u64,
        length: u32,
    ) -> Self {
        // Address gen uses tile address since that's where we write
        let address_gen = AddressGenerator::new_1d(tile_addr, length, 1);

        Self {
            source: TransferEndpoint::HostMemory,
            dest: TransferEndpoint::TileMemory { col: tile_col, row: tile_row },
            direction: TransferDirection::S2MM,
            state: TransferState::Active,
            address_gen,
            bd_index: 0,
            channel: 0,
            acquire_lock: None,
            acquire_value: 0,
            release_lock: None,
            release_value: 0,
            next_bd: None,
            bytes_transferred: 0,
            total_bytes: length as u64,
            cycles_elapsed: 0,
            error: None,
        }
    }

    /// Create a tile-to-host transfer (for reading results).
    pub fn new_tile_to_host(
        tile_col: u8,
        tile_row: u8,
        tile_addr: u64,
        _host_addr: u64,
        length: u32,
    ) -> Self {
        // Address gen uses tile address since that's where we read
        let address_gen = AddressGenerator::new_1d(tile_addr, length, 1);

        Self {
            source: TransferEndpoint::TileMemory { col: tile_col, row: tile_row },
            dest: TransferEndpoint::HostMemory,
            direction: TransferDirection::MM2S,
            state: TransferState::Active,
            address_gen,
            bd_index: 0,
            channel: 0,
            acquire_lock: None,
            acquire_value: 0,
            release_lock: None,
            release_value: 0,
            next_bd: None,
            bytes_transferred: 0,
            total_bytes: length as u64,
            cycles_elapsed: 0,
            error: None,
        }
    }

    /// Check if transfer is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self.state, TransferState::Complete)
    }

    /// Check if transfer is actively moving data.
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self.state, TransferState::Active)
    }

    /// Check if transfer is waiting for a lock.
    #[inline]
    pub fn is_waiting_for_lock(&self) -> bool {
        matches!(self.state, TransferState::WaitingForLock(_))
    }

    /// Check if transfer has an error.
    #[inline]
    pub fn has_error(&self) -> bool {
        matches!(self.state, TransferState::Error)
    }

    /// Get current address for the transfer.
    #[inline]
    pub fn current_address(&self) -> u64 {
        self.address_gen.current()
    }

    /// Get progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_bytes == 0 {
            1.0
        } else {
            self.bytes_transferred as f32 / self.total_bytes as f32
        }
    }

    /// Get remaining bytes to transfer.
    #[inline]
    pub fn remaining_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.bytes_transferred)
    }

    /// Notify that lock was acquired.
    pub fn lock_acquired(&mut self) {
        if let TransferState::WaitingForLock(_) = self.state {
            self.state = TransferState::Active;
        }
    }

    /// Notify that data was transferred.
    pub fn data_transferred(&mut self, bytes: u64) {
        self.bytes_transferred += bytes;

        // Advance address generator
        for _ in 0..(bytes as usize) {
            if self.address_gen.next().is_none() {
                break;
            }
        }

        // Check if transfer is complete
        if self.bytes_transferred >= self.total_bytes {
            if let Some(lock) = self.release_lock {
                self.state = TransferState::ReleasingLock(lock);
            } else {
                self.state = TransferState::Complete;
            }
        }
    }

    /// Notify that lock was released.
    pub fn lock_released(&mut self) {
        if let TransferState::ReleasingLock(_) = self.state {
            self.state = TransferState::Complete;
        }
    }

    /// Mark transfer as failed.
    pub fn set_error(&mut self, error: DmaError) {
        self.error = Some(error);
        self.state = TransferState::Error;
    }

    /// Increment cycle counter.
    #[inline]
    pub fn tick(&mut self) {
        self.cycles_elapsed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::addressing::DimensionConfig;

    fn simple_bd() -> BdConfig {
        BdConfig {
            base_addr: 0x1000,
            length: 256,
            valid: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_transfer_creation() {
        let bd = simple_bd();
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2).unwrap();

        assert_eq!(transfer.bd_index, 0);
        assert_eq!(transfer.channel, 0);
        assert_eq!(transfer.total_bytes, 256);
        assert!(transfer.is_active());
    }

    #[test]
    fn test_transfer_invalid_bd() {
        let mut bd = simple_bd();
        bd.valid = false;

        let result = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2);
        assert!(matches!(result, Err(DmaError::BdNotValid(0))));
    }

    #[test]
    fn test_transfer_with_acquire_lock() {
        let mut bd = simple_bd();
        bd.acquire_lock = Some(5);
        bd.acquire_value = 1;

        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2).unwrap();
        assert!(transfer.is_waiting_for_lock());
        assert!(matches!(transfer.state, TransferState::WaitingForLock(5)));
    }

    #[test]
    fn test_transfer_lifecycle() {
        let mut bd = simple_bd();
        bd.acquire_lock = Some(5);
        bd.release_lock = Some(5);

        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2).unwrap();

        // Initially waiting for lock
        assert!(transfer.is_waiting_for_lock());

        // Acquire lock
        transfer.lock_acquired();
        assert!(transfer.is_active());

        // Transfer data
        transfer.data_transferred(256);
        assert!(matches!(transfer.state, TransferState::ReleasingLock(5)));

        // Release lock
        transfer.lock_released();
        assert!(transfer.is_complete());
    }

    #[test]
    fn test_transfer_progress() {
        let bd = simple_bd();
        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2).unwrap();

        assert_eq!(transfer.progress(), 0.0);

        transfer.data_transferred(128);
        assert_eq!(transfer.progress(), 0.5);

        transfer.data_transferred(128);
        assert_eq!(transfer.progress(), 1.0);
    }

    #[test]
    fn test_transfer_remaining() {
        let bd = simple_bd();
        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2).unwrap();

        assert_eq!(transfer.remaining_bytes(), 256);

        transfer.data_transferred(100);
        assert_eq!(transfer.remaining_bytes(), 156);
    }

    #[test]
    fn test_transfer_direction_s2mm() {
        let bd = simple_bd();
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::S2MM, 1, 2).unwrap();

        assert!(matches!(transfer.source, TransferEndpoint::Stream { .. }));
        assert!(matches!(transfer.dest, TransferEndpoint::TileMemory { .. }));
    }

    #[test]
    fn test_transfer_direction_mm2s() {
        let bd = simple_bd();
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2).unwrap();

        assert!(matches!(transfer.source, TransferEndpoint::TileMemory { .. }));
        assert!(matches!(transfer.dest, TransferEndpoint::Stream { .. }));
    }

    #[test]
    fn test_host_to_tile_transfer() {
        let transfer = Transfer::new_host_to_tile(1, 2, 0x8000_0000, 0x1000, 512);

        assert!(matches!(transfer.source, TransferEndpoint::HostMemory));
        assert!(matches!(transfer.dest, TransferEndpoint::TileMemory { col: 1, row: 2 }));
        assert_eq!(transfer.total_bytes, 512);
    }

    #[test]
    fn test_tile_to_host_transfer() {
        let transfer = Transfer::new_tile_to_host(1, 2, 0x1000, 0x8000_0000, 512);

        assert!(matches!(transfer.source, TransferEndpoint::TileMemory { col: 1, row: 2 }));
        assert!(matches!(transfer.dest, TransferEndpoint::HostMemory));
        assert_eq!(transfer.total_bytes, 512);
    }

    #[test]
    fn test_transfer_error() {
        let bd = simple_bd();
        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2).unwrap();

        transfer.set_error(DmaError::AddressOutOfBounds { address: 0xFFFF, limit: 0x1000 });

        assert!(transfer.has_error());
        assert!(matches!(transfer.state, TransferState::Error));
    }

    #[test]
    fn test_next_bd_chaining() {
        let bd = BdConfig::simple_1d(0x1000, 256).with_next(3);
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2).unwrap();

        assert_eq!(transfer.next_bd, Some(3));
    }
}
