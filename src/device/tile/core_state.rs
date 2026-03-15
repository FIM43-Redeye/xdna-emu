//! Core processor state, legacy stream port, tile type, and control packet action.

/// Core processor state.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct CoreState {
    /// Program counter
    pub pc: u32,
    /// Stack pointer
    pub sp: u32,
    /// Link register
    pub lr: u32,
    /// Status register
    pub status: u32,
    /// Control register
    pub control: u32,
    /// Core is enabled
    pub enabled: bool,
    /// Core is running (not halted)
    pub running: bool,
    /// Padding
    pub(super) _pad: [u8; 2],
}

impl CoreState {
    /// Reset the core to initial state
    pub fn reset(&mut self) {
        self.pc = 0;
        self.sp = 0x7_0000; // Default stack at start of data memory
        self.lr = 0;
        self.status = 0;
        self.control = 0;
        self.enabled = false;
        self.running = false;
    }
}

/// Legacy stream switch port configuration (kept for reference).
/// The actual stream switch functionality is now in FunctionalStreamSwitch.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct LegacyStreamPort {
    /// Port configuration register
    pub config: u32,
}

/// Tile type determines available resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileType {
    /// Shim tile (row 0) - interface to NoC/DDR
    Shim,
    /// Memory tile (row 1) - large memory, no core
    MemTile,
    /// Compute tile (rows 2-5) - core + local memory
    Compute,
}

impl TileType {
    /// Check if this is a shim tile.
    #[inline]
    pub fn is_shim(self) -> bool {
        self == TileType::Shim
    }

    /// Check if this is a memory tile.
    #[inline]
    pub fn is_mem_tile(self) -> bool {
        self == TileType::MemTile
    }

    /// Check if this is a compute tile.
    #[inline]
    pub fn is_compute(self) -> bool {
        self == TileType::Compute
    }
}

// Control packet state machine (ControlPacketState) has been moved to
// control_packets::StreamReassembler in array.rs.

/// An action produced by processing a control packet.
///
/// Control packets are register writes that arrive via the stream switch
/// network. Rather than writing directly within the tile (which misses the
/// full module dispatch in DeviceState), the tile returns actions that the
/// caller routes through `DeviceState::write_tile_register()`.
#[derive(Debug)]
pub enum CtrlPacketAction {
    /// Write a value to a tile-local register offset.
    WriteRegister { col: u8, row: u8, offset: u32, value: u32 },
    /// Read registers starting at offset (not yet implemented; logged).
    ReadRegisters { col: u8, row: u8, offset: u32, count: u8, response_id: u8 },
    /// An error occurred during control packet processing.
    Error(String),
}
