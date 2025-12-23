//! AIE tile state representation.
//!
//! Each tile contains:
//! - Data memory (64KB for compute tiles, 512KB for mem tiles)
//! - Program memory (64KB, compute tiles only)
//! - 64 locks for synchronization
//! - DMA engine with buffer descriptors and channels
//! - Core state (PC, registers, status)
//! - Stream switch configuration
//!
//! # Performance
//!
//! This module is designed for fast emulation:
//! - Fixed-size arrays (no heap allocation during emulation)
//! - Direct field access (no hash maps)
//! - Cache-friendly layout (related data together)

/// Size of data memory in compute tiles (64KB)
pub const COMPUTE_TILE_MEMORY_SIZE: usize = 64 * 1024;

/// Size of data memory in memory tiles (512KB)
pub const MEM_TILE_MEMORY_SIZE: usize = 512 * 1024;

/// Size of program memory (64KB = 16K instructions)
pub const PROGRAM_MEMORY_SIZE: usize = 64 * 1024;

/// Number of locks per tile
pub const NUM_LOCKS: usize = 64;

/// Number of DMA buffer descriptors per tile
pub const NUM_DMA_BDS: usize = 16;

/// Number of DMA channels (2 S2MM + 2 MM2S)
pub const NUM_DMA_CHANNELS: usize = 4;

/// Lock state.
///
/// AIE2 uses semaphore locks with acquire/release semantics.
/// Value is typically 0 (available) or 1+ (acquired N times).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Lock {
    /// Current lock value (semaphore count)
    pub value: u32,
}

impl Lock {
    /// Create a new lock with initial value
    #[inline]
    pub fn new(value: u32) -> Self {
        Self { value }
    }

    /// Acquire the lock (decrement if > 0)
    #[inline]
    pub fn acquire(&mut self) -> bool {
        if self.value > 0 {
            self.value -= 1;
            true
        } else {
            false
        }
    }

    /// Release the lock (increment)
    #[inline]
    pub fn release(&mut self) {
        self.value += 1;
    }
}

/// DMA buffer descriptor.
///
/// Describes a memory region for DMA transfer with multi-dimensional
/// addressing support.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DmaBufferDescriptor {
    /// Base address (low 32 bits)
    pub addr_low: u32,
    /// Base address (high 32 bits, for 64-bit addressing)
    pub addr_high: u32,
    /// Transfer length in bytes
    pub length: u32,
    /// Control register (valid, compression, etc.)
    pub control: u32,
    /// Dimension 1 configuration (stride, wrap)
    pub d0: u32,
    /// Dimension 2 configuration
    pub d1: u32,
}

impl DmaBufferDescriptor {
    /// Check if this BD is valid (enabled)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.control & 1 != 0
    }

    /// Get the base address as 64-bit
    #[inline]
    pub fn address(&self) -> u64 {
        ((self.addr_high as u64) << 32) | (self.addr_low as u64)
    }

    /// Get the next BD index (for chaining)
    #[inline]
    pub fn next_bd(&self) -> Option<u8> {
        let next = ((self.control >> 8) & 0xF) as u8;
        if self.control & 0x80 != 0 {
            // Use next BD bit set
            Some(next)
        } else {
            None
        }
    }
}

/// DMA channel state.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DmaChannel {
    /// Control register
    pub control: u32,
    /// Start queue (BD to start)
    pub start_queue: u32,
    /// Current BD being processed
    pub current_bd: u8,
    /// Channel is running
    pub running: bool,
    /// Padding for alignment
    _pad: [u8; 2],
}

impl DmaChannel {
    /// Check if channel is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.control & 1 != 0
    }

    /// Check if channel is paused
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.control & 2 != 0
    }
}

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
    _pad: [u8; 2],
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

/// Stream switch port configuration.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct StreamPort {
    /// Port configuration register
    pub config: u32,
}

/// Stream switch state.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct StreamSwitch {
    /// Master ports (typically 6)
    pub master: [StreamPort; 8],
    /// Slave ports (typically 8)
    pub slave: [StreamPort; 8],
    /// Control packet handler config
    pub ctrl_pkt: u32,
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

/// Complete state of a single AIE tile.
///
/// This struct is designed for cache-friendly access during emulation.
/// Hot data (core state, locks) is at the start; cold data (memory) is at the end.
#[derive(Debug)]
pub struct Tile {
    /// Tile type
    pub tile_type: TileType,

    /// Column index
    pub col: u8,

    /// Row index
    pub row: u8,

    // === Hot data (accessed every cycle) ===

    /// Core processor state (compute tiles only)
    pub core: CoreState,

    /// Lock states
    pub locks: [Lock; NUM_LOCKS],

    // === Warm data (accessed during DMA) ===

    /// DMA buffer descriptors
    pub dma_bds: [DmaBufferDescriptor; NUM_DMA_BDS],

    /// DMA channels (0-1: S2MM, 2-3: MM2S)
    pub dma_channels: [DmaChannel; NUM_DMA_CHANNELS],

    // === Cold data (routing configuration) ===

    /// Stream switch configuration
    pub stream_switch: StreamSwitch,

    // === Large data (memory) ===

    /// Data memory (64KB for compute, 512KB for mem tile)
    /// Boxed to avoid huge stack allocation
    data_memory: Box<[u8]>,

    /// Program memory (64KB, compute tiles only)
    /// None for shim and mem tiles
    program_memory: Option<Box<[u8; PROGRAM_MEMORY_SIZE]>>,
}

impl Tile {
    /// Create a new tile of the specified type.
    pub fn new(tile_type: TileType, col: u8, row: u8) -> Self {
        let data_memory_size = match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => MEM_TILE_MEMORY_SIZE,
            TileType::Compute => COMPUTE_TILE_MEMORY_SIZE,
        };

        let program_memory = match tile_type {
            TileType::Compute => Some(Box::new([0u8; PROGRAM_MEMORY_SIZE])),
            _ => None,
        };

        Self {
            tile_type,
            col,
            row,
            core: CoreState::default(),
            locks: [Lock::default(); NUM_LOCKS],
            dma_bds: [DmaBufferDescriptor::default(); NUM_DMA_BDS],
            dma_channels: [DmaChannel::default(); NUM_DMA_CHANNELS],
            stream_switch: StreamSwitch::default(),
            data_memory: vec![0u8; data_memory_size].into_boxed_slice(),
            program_memory,
        }
    }

    /// Create a compute tile.
    #[inline]
    pub fn compute(col: u8, row: u8) -> Self {
        Self::new(TileType::Compute, col, row)
    }

    /// Create a memory tile.
    #[inline]
    pub fn mem_tile(col: u8, row: u8) -> Self {
        Self::new(TileType::MemTile, col, row)
    }

    /// Create a shim tile.
    #[inline]
    pub fn shim(col: u8, row: u8) -> Self {
        Self::new(TileType::Shim, col, row)
    }

    /// Get data memory slice.
    #[inline]
    pub fn data_memory(&self) -> &[u8] {
        &self.data_memory
    }

    /// Get mutable data memory slice.
    #[inline]
    pub fn data_memory_mut(&mut self) -> &mut [u8] {
        &mut self.data_memory
    }

    /// Get program memory (compute tiles only).
    #[inline]
    pub fn program_memory(&self) -> Option<&[u8; PROGRAM_MEMORY_SIZE]> {
        self.program_memory.as_deref()
    }

    /// Get mutable program memory (compute tiles only).
    #[inline]
    pub fn program_memory_mut(&mut self) -> Option<&mut [u8; PROGRAM_MEMORY_SIZE]> {
        self.program_memory.as_deref_mut()
    }

    /// Write to data memory at offset.
    /// Returns false if offset + data would exceed memory bounds.
    #[inline]
    pub fn write_data(&mut self, offset: usize, data: &[u8]) -> bool {
        if offset + data.len() <= self.data_memory.len() {
            self.data_memory[offset..offset + data.len()].copy_from_slice(data);
            true
        } else {
            false
        }
    }

    /// Write to program memory at offset (compute tiles only).
    /// Returns false if not a compute tile or would exceed bounds.
    #[inline]
    pub fn write_program(&mut self, offset: usize, data: &[u8]) -> bool {
        if let Some(ref mut pm) = self.program_memory {
            if offset + data.len() <= PROGRAM_MEMORY_SIZE {
                pm[offset..offset + data.len()].copy_from_slice(data);
                return true;
            }
        }
        false
    }

    /// Read 32-bit word from data memory.
    #[inline]
    pub fn read_data_u32(&self, offset: usize) -> Option<u32> {
        if offset + 4 <= self.data_memory.len() {
            Some(u32::from_le_bytes([
                self.data_memory[offset],
                self.data_memory[offset + 1],
                self.data_memory[offset + 2],
                self.data_memory[offset + 3],
            ]))
        } else {
            None
        }
    }

    /// Write 32-bit word to data memory.
    #[inline]
    pub fn write_data_u32(&mut self, offset: usize, value: u32) -> bool {
        if offset + 4 <= self.data_memory.len() {
            self.data_memory[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            true
        } else {
            false
        }
    }

    /// Check if this is a compute tile.
    #[inline]
    pub fn is_compute(&self) -> bool {
        self.tile_type == TileType::Compute
    }

    /// Check if this is a memory tile.
    #[inline]
    pub fn is_mem_tile(&self) -> bool {
        self.tile_type == TileType::MemTile
    }

    /// Check if this is a shim tile.
    #[inline]
    pub fn is_shim(&self) -> bool {
        self.tile_type == TileType::Shim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_creation() {
        let tile = Tile::compute(1, 2);
        assert_eq!(tile.col, 1);
        assert_eq!(tile.row, 2);
        assert!(tile.is_compute());
        assert!(tile.program_memory().is_some());
        assert_eq!(tile.data_memory().len(), COMPUTE_TILE_MEMORY_SIZE);
    }

    #[test]
    fn test_mem_tile_creation() {
        let tile = Tile::mem_tile(0, 1);
        assert!(tile.is_mem_tile());
        assert!(tile.program_memory().is_none());
        assert_eq!(tile.data_memory().len(), MEM_TILE_MEMORY_SIZE);
    }

    #[test]
    fn test_shim_tile_creation() {
        let tile = Tile::shim(0, 0);
        assert!(tile.is_shim());
        assert!(tile.program_memory().is_none());
        assert_eq!(tile.data_memory().len(), 0);
    }

    #[test]
    fn test_data_memory_write() {
        let mut tile = Tile::compute(0, 2);
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        assert!(tile.write_data(0x100, &data));
        assert_eq!(&tile.data_memory()[0x100..0x104], &data);
    }

    #[test]
    fn test_program_memory_write() {
        let mut tile = Tile::compute(0, 2);
        let code = [0x15, 0x01, 0x00, 0x40]; // Sample AIE instruction
        assert!(tile.write_program(0, &code));
        assert_eq!(&tile.program_memory().unwrap()[0..4], &code);
    }

    #[test]
    fn test_data_u32_operations() {
        let mut tile = Tile::compute(0, 2);
        assert!(tile.write_data_u32(0x200, 0xCAFEBABE));
        assert_eq!(tile.read_data_u32(0x200), Some(0xCAFEBABE));
    }

    #[test]
    fn test_lock_operations() {
        let mut lock = Lock::new(2);
        assert!(lock.acquire()); // 2 -> 1
        assert!(lock.acquire()); // 1 -> 0
        assert!(!lock.acquire()); // 0 -> can't acquire
        lock.release(); // 0 -> 1
        assert!(lock.acquire()); // 1 -> 0
    }

    #[test]
    fn test_dma_bd_valid() {
        let mut bd = DmaBufferDescriptor::default();
        assert!(!bd.is_valid());
        bd.control = 1;
        assert!(bd.is_valid());
    }

    #[test]
    fn test_core_state_reset() {
        let mut core = CoreState {
            pc: 0x1000,
            sp: 0x8000,
            lr: 0x500,
            status: 0xFF,
            control: 0x3,
            enabled: true,
            running: true,
            _pad: [0; 2],
        };
        core.reset();
        assert_eq!(core.pc, 0);
        assert_eq!(core.sp, 0x7_0000);
        assert!(!core.enabled);
    }

    #[test]
    fn test_struct_sizes() {
        // Ensure structs are reasonably sized
        assert_eq!(std::mem::size_of::<Lock>(), 4);
        assert_eq!(std::mem::size_of::<DmaBufferDescriptor>(), 24);
        assert_eq!(std::mem::size_of::<DmaChannel>(), 12);
        assert_eq!(std::mem::size_of::<CoreState>(), 24);
    }
}
