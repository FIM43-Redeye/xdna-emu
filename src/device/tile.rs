//! AIE tile state representation.
//!
//! Each tile contains:
//! - Data memory (64KB for compute tiles, 512KB for mem tiles)
//! - Program memory (16KB, compute tiles only)
//! - Locks for synchronization (16 for compute tiles, 64 for mem tiles)
//! - DMA engine with buffer descriptors and channels
//! - Core state (PC, registers, status)
//! - Stream switch configuration
//!
//! # Architecture Constants
//!
//! All constants are derived from AMD AM020 (AIE-ML Architecture Manual).
//! See `aie2_spec` module for the authoritative values.
//!
//! # Performance
//!
//! This module is designed for fast emulation:
//! - Fixed-size arrays (no heap allocation during emulation)
//! - Direct field access (no hash maps)
//! - Cache-friendly layout (related data together)

use super::aie2_spec;
use super::stream_switch::StreamSwitch as FunctionalStreamSwitch;

/// Size of data memory in compute tiles (64KB)
/// See AM020 Ch4: "An individual data memory block is 64 KB"
pub const COMPUTE_TILE_MEMORY_SIZE: usize = aie2_spec::COMPUTE_TILE_DATA_MEMORY_SIZE;

/// Size of data memory in memory tiles (512KB)
/// See AM020 Ch5: "Each AIE-ML memory tile has 512 KB of memory"
pub const MEM_TILE_MEMORY_SIZE: usize = aie2_spec::MEM_TILE_DATA_MEMORY_SIZE;

/// Size of program memory (16KB = 1024 x 128-bit instructions)
/// See AM020 Ch4: "The program memory size on the AIE-ML is 16 KB"
pub const PROGRAM_MEMORY_SIZE: usize = aie2_spec::PROGRAM_MEMORY_SIZE;

/// Number of locks per compute tile (16)
/// See AM020 Ch2: "The AIE-ML features 16 semaphore locks"
pub const NUM_LOCKS_COMPUTE: usize = aie2_spec::COMPUTE_TILE_NUM_LOCKS;

/// Number of locks per memory tile (64)
/// See AM020 Ch5: "there are 64 semaphore locks"
pub const NUM_LOCKS_MEM_TILE: usize = aie2_spec::MEM_TILE_NUM_LOCKS;

/// Maximum number of locks (for array sizing - uses mem tile count)
pub const NUM_LOCKS: usize = NUM_LOCKS_MEM_TILE;

/// Number of DMA buffer descriptors per tile (16)
/// See AM020 Ch2: "the DMA controller has access to the 16 buffer descriptors"
pub const NUM_DMA_BDS: usize = aie2_spec::NUM_DMA_BUFFER_DESCRIPTORS;

/// Number of DMA channels for compute tiles (2 S2MM + 2 MM2S = 4)
pub const NUM_DMA_CHANNELS: usize = aie2_spec::COMPUTE_TILE_S2MM_CHANNELS
    + aie2_spec::COMPUTE_TILE_MM2S_CHANNELS;

/// Result of a lock operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockResult {
    /// Operation succeeded
    Success,
    /// Operation failed - would underflow (value would go negative)
    WouldUnderflow,
    /// Operation failed - would overflow (value would exceed 63)
    WouldOverflow,
}

/// Lock state.
///
/// AIE2 uses semaphore locks with acquire/release semantics.
/// Lock value is 6-bit unsigned (0-63). See AM020 Ch2:
/// "The semaphore lock has a larger state and no acquired bit;
/// each lock state is 6-bit unsigned."
///
/// # Semaphore Model (AM025)
///
/// Lock operations use a change_value parameter:
/// - Acquire: Waits until (value + change_value >= 0), then applies change
/// - Release: Applies change_value, saturating at MAX_VALUE (63)
///
/// The Lock_Request register format (AM025):
/// - Lock_Id [13:10]: Which lock (0-15 for compute, 0-63 for mem tile)
/// - Acq_Rel [9]: 1 = acquire (blocking), 0 = release (non-blocking)
/// - Change_Value [8:2]: Signed 7-bit delta (-64 to +63)
/// - Request_Result [0]: 0 = failed, 1 = succeeded
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Lock {
    /// Current lock value (semaphore count, 0-63)
    /// Stored as u8 for efficiency but only lower 6 bits are valid.
    pub value: u8,
    /// Overflow flag - set when release would exceed MAX_VALUE
    pub overflow: bool,
    /// Underflow flag - set when acquire would go negative
    pub underflow: bool,
}

impl Lock {
    /// Maximum lock value (6-bit: 0-63)
    pub const MAX_VALUE: u8 = aie2_spec::LOCK_MAX_VALUE;

    /// Create a new lock with initial value (clamped to 0-63)
    #[inline]
    pub fn new(value: u8) -> Self {
        Self {
            value: value.min(Self::MAX_VALUE),
            overflow: false,
            underflow: false,
        }
    }

    /// Acquire the lock (decrement if > 0).
    ///
    /// This is the simple form equivalent to `acquire_with_value(1, -1)`.
    #[inline]
    pub fn acquire(&mut self) -> bool {
        if self.value > 0 {
            self.value -= 1;
            true
        } else {
            false
        }
    }

    /// Release the lock (increment, saturating at MAX_VALUE).
    ///
    /// This is the simple form equivalent to `release_with_value(1)`.
    #[inline]
    pub fn release(&mut self) {
        if self.value < Self::MAX_VALUE {
            self.value += 1;
        }
    }

    /// Acquire with greater-or-equal check (acquire_ge mode).
    ///
    /// Checks if `value >= expected_value`, and if so, applies `delta` to the
    /// lock value. Returns `LockResult::Success` if the operation succeeded,
    /// or the appropriate error if it would underflow.
    ///
    /// This implements the AIE-ML acquire_ge semantics where a negative
    /// Lock_Acq_Value in the BD indicates waiting for lock >= |value|.
    ///
    /// # Arguments
    /// * `expected_value` - Minimum value required for acquire to succeed
    /// * `delta` - Change to apply (typically negative for acquire)
    ///
    /// # Example
    /// ```ignore
    /// // Wait for lock value >= 1, then decrement by 1
    /// lock.acquire_with_value(1, -1);
    ///
    /// // Wait for lock value >= 2, then decrement by 2
    /// lock.acquire_with_value(2, -2);
    /// ```
    #[inline]
    pub fn acquire_with_value(&mut self, expected_value: u8, delta: i8) -> LockResult {
        if self.value < expected_value {
            // Not enough value - operation would stall
            return LockResult::WouldUnderflow;
        }

        // Apply delta (convert to i16 for safe arithmetic)
        let new_value = (self.value as i16) + (delta as i16);

        if new_value < 0 {
            self.underflow = true;
            return LockResult::WouldUnderflow;
        }

        if new_value > Self::MAX_VALUE as i16 {
            // This shouldn't happen for acquire (negative delta), but handle it
            self.overflow = true;
            self.value = Self::MAX_VALUE;
            return LockResult::WouldOverflow;
        }

        self.value = new_value as u8;
        LockResult::Success
    }

    /// Acquire with exact-match check (acquire_eq mode).
    ///
    /// Checks if `value == expected_value`, and if so, applies `delta` to the
    /// lock value. Returns `LockResult::Success` if the operation succeeded.
    /// Returns `LockResult::WouldUnderflow` if the value doesn't match exactly.
    ///
    /// This implements the AIE-ML acquire_eq semantics where a non-negative
    /// Lock_Acq_Value in the BD indicates waiting for lock == value exactly.
    ///
    /// # Arguments
    /// * `expected_value` - Exact value required for acquire to succeed
    /// * `delta` - Change to apply (typically sets to 0 for acquire_eq)
    ///
    /// # Example
    /// ```ignore
    /// // Wait for lock value == 1, then set to 0
    /// lock.acquire_equal(1, -1);
    ///
    /// // Wait for lock value == 2, then set to 0
    /// lock.acquire_equal(2, -2);
    /// ```
    #[inline]
    pub fn acquire_equal(&mut self, expected_value: u8, delta: i8) -> LockResult {
        if self.value != expected_value {
            // Value doesn't match exactly - operation would stall
            return LockResult::WouldUnderflow;
        }

        // Apply delta (convert to i16 for safe arithmetic)
        let new_value = (self.value as i16) + (delta as i16);

        if new_value < 0 {
            self.underflow = true;
            return LockResult::WouldUnderflow;
        }

        if new_value > Self::MAX_VALUE as i16 {
            self.overflow = true;
            self.value = Self::MAX_VALUE;
            return LockResult::WouldOverflow;
        }

        self.value = new_value as u8;
        LockResult::Success
    }

    /// Release with specific delta.
    ///
    /// Adds `delta` to the lock value, saturating at MAX_VALUE.
    /// Sets overflow flag if saturation occurs.
    ///
    /// # Arguments
    /// * `delta` - Amount to add (typically positive for release)
    ///
    /// # Example
    /// ```ignore
    /// // Release: increment by 1
    /// lock.release_with_value(1);
    ///
    /// // Release: increment by 2
    /// lock.release_with_value(2);
    /// ```
    #[inline]
    pub fn release_with_value(&mut self, delta: i8) -> LockResult {
        let new_value = (self.value as i16) + (delta as i16);

        if new_value < 0 {
            self.underflow = true;
            self.value = 0;
            return LockResult::WouldUnderflow;
        }

        if new_value > Self::MAX_VALUE as i16 {
            self.overflow = true;
            self.value = Self::MAX_VALUE;
            return LockResult::WouldOverflow;
        }

        self.value = new_value as u8;
        LockResult::Success
    }

    /// Set the lock value directly (clamped to 0-63)
    #[inline]
    pub fn set(&mut self, value: u8) {
        self.value = value.min(Self::MAX_VALUE);
    }

    /// Clear the overflow and underflow flags.
    #[inline]
    pub fn clear_flags(&mut self) {
        self.overflow = false;
        self.underflow = false;
    }

    /// Check if the lock has any error flags set.
    #[inline]
    pub fn has_error(&self) -> bool {
        self.overflow || self.underflow
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
    /// Controller ID for task complete tokens (from control register bits 15:8)
    pub controller_id: u8,
    /// Finish-on-TLAST mode (S2MM only, from control register bits 17:16)
    pub fot_mode: u8,
    /// Enable token issue for current task (from start_queue bit 31)
    pub enable_token_issue: bool,
    /// Compression enable (MM2S only, from control register bit 4)
    pub compression_enable: bool,
    /// Decompression enable (S2MM only, from control register bit 4)
    pub decompression_enable: bool,
    /// Out-of-order mode enable (S2MM only, from control register bit 3)
    pub out_of_order_enable: bool,
    /// Status register (read-only bits updated during execution)
    pub status: u32,
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

    /// Check if channel is in reset
    #[inline]
    pub fn is_reset(&self) -> bool {
        (self.control >> 1) & 1 != 0
    }

    /// Get the controller ID for task complete tokens
    #[inline]
    pub fn get_controller_id(&self) -> u8 {
        self.controller_id
    }

    /// Get the FoT mode (S2MM only)
    #[inline]
    pub fn get_fot_mode(&self) -> u8 {
        self.fot_mode
    }

    /// Check if token issue is enabled for current task
    #[inline]
    pub fn should_issue_token(&self) -> bool {
        self.enable_token_issue
    }

    /// Update status register field: Cur_BD
    pub fn set_cur_bd(&mut self, bd: u8) {
        use super::registers_spec::memory_module::channel;
        self.status = (self.status & !(channel::STATUS_CUR_BD_MASK << channel::STATUS_CUR_BD_SHIFT))
            | (((bd as u32) & channel::STATUS_CUR_BD_MASK) << channel::STATUS_CUR_BD_SHIFT);
    }

    /// Update status register: Channel_Running
    pub fn set_channel_running(&mut self, running: bool) {
        use super::registers_spec::memory_module::channel;
        if running {
            self.status |= 1 << channel::STATUS_CHANNEL_RUNNING_BIT;
        } else {
            self.status &= !(1 << channel::STATUS_CHANNEL_RUNNING_BIT);
        }
    }

    /// Update status register: State bits (00=IDLE, 01=STARTING, 10=RUNNING)
    pub fn set_state(&mut self, state: u8) {
        use super::registers_spec::memory_module::channel;
        self.status = (self.status & !channel::STATUS_STATE_MASK)
            | ((state as u32) & channel::STATUS_STATE_MASK);
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

    /// Lock value snapshot for cycle-accurate arbitration.
    /// All lock operations within a cycle check against this snapshot,
    /// ensuring all requestors see the same values regardless of processing order.
    lock_snapshot: [u8; NUM_LOCKS],

    /// Pending lock deltas to apply at end of cycle.
    /// Releases add positive deltas, acquires add negative deltas.
    lock_deltas: [i8; NUM_LOCKS],

    // === Warm data (accessed during DMA) ===

    /// DMA buffer descriptors
    pub dma_bds: [DmaBufferDescriptor; NUM_DMA_BDS],

    /// DMA channels (0-1: S2MM, 2-3: MM2S)
    pub dma_channels: [DmaChannel; NUM_DMA_CHANNELS],

    // === Stream port buffers (for core direct stream access) ===

    /// Stream input buffer for core direct reads (StreamReadScalar)
    /// Maps port number to queue of incoming data
    pub stream_input: [std::collections::VecDeque<u32>; 8],

    /// Stream output buffer for core direct writes (StreamWriteScalar)
    pub stream_output: [std::collections::VecDeque<u32>; 8],

    // === Cold data (routing configuration) ===

    /// Stream switch configuration (full functional model with FIFOs and local routes)
    pub stream_switch: FunctionalStreamSwitch,

    // === Large data (memory) ===

    /// Data memory (64KB for compute, 512KB for mem tile)
    /// Boxed to avoid huge stack allocation
    data_memory: Box<[u8]>,

    /// Program memory (64KB, compute tiles only)
    /// None for shim and mem tiles
    program_memory: Option<Box<[u8; PROGRAM_MEMORY_SIZE]>>,

    /// Register store for shim tiles (NPU configuration registers).
    /// Shim tiles don't have data memory but need to store DMA/stream config.
    /// Stored as sparse map since most addresses won't be written.
    registers: std::collections::HashMap<u32, u32>,
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
            lock_snapshot: [0; NUM_LOCKS],
            lock_deltas: [0; NUM_LOCKS],
            dma_bds: [DmaBufferDescriptor::default(); NUM_DMA_BDS],
            dma_channels: [DmaChannel::default(); NUM_DMA_CHANNELS],
            stream_input: Default::default(),
            stream_output: Default::default(),
            stream_switch: match tile_type {
                TileType::Shim => FunctionalStreamSwitch::new_shim_tile(col),
                TileType::MemTile => FunctionalStreamSwitch::new_mem_tile(col, row),
                TileType::Compute => FunctionalStreamSwitch::new_compute_tile(col, row),
            },
            data_memory: vec![0u8; data_memory_size].into_boxed_slice(),
            program_memory,
            registers: std::collections::HashMap::new(),
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

    // === Lock Arbiter Methods ===
    //
    // These methods implement snapshot-based lock arbitration for cycle-accurate
    // emulation. Within a single cycle:
    // 1. `begin_lock_cycle()` - snapshot current lock values
    // 2. All lock operations (DMA, core) use `try_acquire_snapshot()` / `release_snapshot()`
    // 3. `end_lock_cycle()` - apply accumulated deltas to actual lock values
    //
    // This ensures all requestors see the same lock state regardless of processing
    // order within a cycle, matching hardware arbiter behavior.

    /// Begin a lock cycle by snapshotting current lock values.
    ///
    /// Call this at the start of each simulation cycle, before any lock operations.
    /// All subsequent `try_acquire_snapshot()` and `release_snapshot()` calls will
    /// check/modify based on this snapshot rather than live values.
    #[inline]
    pub fn begin_lock_cycle(&mut self) {
        for i in 0..NUM_LOCKS {
            self.lock_snapshot[i] = self.locks[i].value;
            self.lock_deltas[i] = 0;
        }
    }

    /// Try to acquire a lock using snapshot-based arbitration.
    ///
    /// Checks the lock's snapshot value against the expected threshold.
    /// If successful, records a negative delta to be applied at cycle end.
    ///
    /// # Arguments
    /// * `lock_id` - Lock index (0-63)
    /// * `expected` - Minimum value required (for acq_ge mode)
    /// * `delta` - Change to apply (typically negative for acquire)
    /// * `equal_mode` - If true, requires exact match (acq_eq mode)
    ///
    /// # Returns
    /// `LockResult::Success` if acquire would succeed, error otherwise.
    pub fn try_acquire_snapshot(&mut self, lock_id: usize, expected: u8, delta: i8, equal_mode: bool) -> LockResult {
        if lock_id >= NUM_LOCKS {
            return LockResult::WouldUnderflow;
        }

        // Calculate effective value: snapshot + any deltas already recorded this cycle
        let effective = (self.lock_snapshot[lock_id] as i16 + self.lock_deltas[lock_id] as i16) as u8;

        let success = if equal_mode {
            effective == expected
        } else {
            effective >= expected
        };

        if success {
            // Record the delta - will be applied at end of cycle
            self.lock_deltas[lock_id] = self.lock_deltas[lock_id].saturating_add(delta);
            LockResult::Success
        } else {
            LockResult::WouldUnderflow
        }
    }

    /// Release a lock using snapshot-based arbitration.
    ///
    /// Records a positive delta to be applied at cycle end.
    /// Release operations always succeed (no blocking).
    ///
    /// # Arguments
    /// * `lock_id` - Lock index (0-63)
    /// * `delta` - Change to apply (typically positive for release)
    #[inline]
    pub fn release_snapshot(&mut self, lock_id: usize, delta: i8) {
        if lock_id < NUM_LOCKS {
            self.lock_deltas[lock_id] = self.lock_deltas[lock_id].saturating_add(delta);
        }
    }

    /// End a lock cycle by applying accumulated deltas.
    ///
    /// Call this at the end of each simulation cycle, after all lock operations.
    /// Applies all recorded deltas to the actual lock values, with clamping.
    #[inline]
    pub fn end_lock_cycle(&mut self) {
        for i in 0..NUM_LOCKS {
            let new_value = (self.locks[i].value as i16) + (self.lock_deltas[i] as i16);
            if new_value < 0 {
                self.locks[i].underflow = true;
                self.locks[i].value = 0;
            } else if new_value > Lock::MAX_VALUE as i16 {
                self.locks[i].overflow = true;
                self.locks[i].value = Lock::MAX_VALUE;
            } else {
                self.locks[i].value = new_value as u8;
            }
        }
    }

    /// Get the snapshot value for a lock (for debugging/logging).
    #[inline]
    pub fn lock_snapshot_value(&self, lock_id: usize) -> u8 {
        if lock_id < NUM_LOCKS {
            self.lock_snapshot[lock_id]
        } else {
            0
        }
    }

    /// Get the pending delta for a lock (for debugging/logging).
    #[inline]
    pub fn lock_pending_delta(&self, lock_id: usize) -> i8 {
        if lock_id < NUM_LOCKS {
            self.lock_deltas[lock_id]
        } else {
            0
        }
    }

    // === Register Access (for NPU instruction execution) ===

    /// Write a 32-bit value to a register offset.
    ///
    /// For shim tiles, this stores to a sparse register map.
    /// For other tiles, this may route to DMA BDs, locks, or other subsystems
    /// based on the offset.
    ///
    /// # Register Offset Ranges (from AM020/AM025)
    ///
    /// - 0x14000-0x147FF: Lock registers
    /// - 0x1D000-0x1D3FF: DMA BD registers
    /// - 0x1D400-0x1D7FF: DMA channel control
    /// - 0x3F000-0x3FFFF: Stream switch configuration
    pub fn write_register(&mut self, offset: u32, value: u32) {
        // Always store in the register map for later retrieval
        self.registers.insert(offset, value);

        // For specific ranges, also update internal state
        // DMA BD range: 0x1D000-0x1D3FF (16 BDs × 32 bytes each)
        if (0x1D000..0x1D200).contains(&offset) {
            let bd_offset = offset - 0x1D000;
            let bd_index = (bd_offset / 0x20) as usize;
            let reg_in_bd = (bd_offset % 0x20) as usize / 4;

            if bd_index < NUM_DMA_BDS {
                let bd = &mut self.dma_bds[bd_index];
                match reg_in_bd {
                    0 => bd.addr_low = value,
                    1 => bd.addr_high = value,
                    2 => bd.length = value,
                    3 => bd.control = value,
                    4 => bd.d0 = value,
                    5 => bd.d1 = value,
                    _ => {}
                }
            }
        }

        // Lock registers: 0x14000-0x14XXX
        if (0x14000..0x15000).contains(&offset) {
            let lock_offset = offset - 0x14000;
            let lock_id = (lock_offset / 0x10) as usize;
            if lock_id < NUM_LOCKS {
                // Lock value register is at offset 0x0 within each lock block
                if lock_offset % 0x10 == 0 {
                    self.locks[lock_id].set((value & 0x3F) as u8);
                }
            }
        }

        // DMA channel control: 0x1D200-0x1D3FF
        if (0x1D200..0x1D400).contains(&offset) {
            let ch_offset = offset - 0x1D200;
            let ch_index = (ch_offset / 0x8) as usize;
            if ch_index < NUM_DMA_CHANNELS {
                let ch = &mut self.dma_channels[ch_index];
                if ch_offset % 0x8 == 0 {
                    ch.control = value;
                } else {
                    ch.start_queue = value;
                }
            }
        }

        // Lock status registers (write-to-clear)
        // Writing 1 to a bit clears that lock's overflow/underflow status
        use super::registers_spec::{memory_module as mm, mem_tile_module as mt};
        if self.is_mem_tile() {
            if offset == mt::LOCKS_OVERFLOW_0 {
                self.clear_lock_overflow_bits(0, 32, value);
            } else if offset == mt::LOCKS_OVERFLOW_1 {
                self.clear_lock_overflow_bits(32, 64, value);
            } else if offset == mt::LOCKS_UNDERFLOW_0 {
                self.clear_lock_underflow_bits(0, 32, value);
            } else if offset == mt::LOCKS_UNDERFLOW_1 {
                self.clear_lock_underflow_bits(32, 64, value);
            }
        } else if self.is_compute() {
            if offset == mm::LOCKS_OVERFLOW {
                self.clear_lock_overflow_bits(0, 16, value);
            } else if offset == mm::LOCKS_UNDERFLOW {
                self.clear_lock_underflow_bits(0, 16, value);
            }
        }
    }

    /// Read a 32-bit value from a register offset.
    ///
    /// Returns 0 for unwritten registers (default state).
    pub fn read_register(&mut self, offset: u32) -> u32 {
        use super::registers_spec::{memory_module as mm, mem_tile_module as mt};

        // Lock_Request register - address encodes operation parameters
        // Reading performs the lock operation and returns result
        if self.is_mem_tile() {
            if (mt::LOCK_REQUEST_BASE..mt::LOCK_REQUEST_END).contains(&offset) {
                return self.handle_lock_request(offset, true);
            }
            // Lock status registers
            if offset == mt::LOCKS_OVERFLOW_0 {
                return self.get_lock_overflow_bits(0, 32);
            }
            if offset == mt::LOCKS_OVERFLOW_1 {
                return self.get_lock_overflow_bits(32, 64);
            }
            if offset == mt::LOCKS_UNDERFLOW_0 {
                return self.get_lock_underflow_bits(0, 32);
            }
            if offset == mt::LOCKS_UNDERFLOW_1 {
                return self.get_lock_underflow_bits(32, 64);
            }
        } else if self.is_compute() {
            if (mm::LOCK_REQUEST_BASE..mm::LOCK_REQUEST_END).contains(&offset) {
                return self.handle_lock_request(offset, false);
            }
            // Lock status registers
            if offset == mm::LOCKS_OVERFLOW {
                return self.get_lock_overflow_bits(0, 16);
            }
            if offset == mm::LOCKS_UNDERFLOW {
                return self.get_lock_underflow_bits(0, 16);
            }
        }

        // Check specific subsystem state first
        // DMA BD range: 0x1D000-0x1D1FF
        if (0x1D000..0x1D200).contains(&offset) {
            let bd_offset = offset - 0x1D000;
            let bd_index = (bd_offset / 0x20) as usize;
            let reg_in_bd = (bd_offset % 0x20) as usize / 4;

            if bd_index < NUM_DMA_BDS {
                let bd = &self.dma_bds[bd_index];
                return match reg_in_bd {
                    0 => bd.addr_low,
                    1 => bd.addr_high,
                    2 => bd.length,
                    3 => bd.control,
                    4 => bd.d0,
                    5 => bd.d1,
                    _ => 0,
                };
            }
        }

        // Fall back to register map
        self.registers.get(&offset).copied().unwrap_or(0)
    }

    /// Get a reference to the raw register map.
    ///
    /// Useful for debugging and inspection.
    pub fn registers(&self) -> &std::collections::HashMap<u32, u32> {
        &self.registers
    }

    // === Stream Port Access (for core direct stream reads/writes) ===

    /// Push a word to the stream input buffer for a port.
    ///
    /// Called by the stream router when data arrives for this tile.
    pub fn push_stream_input(&mut self, port: u8, value: u32) {
        if (port as usize) < self.stream_input.len() {
            self.stream_input[port as usize].push_back(value);
        }
    }

    /// Pop a word from the stream input buffer for a port.
    ///
    /// Called by StreamReadScalar when the core reads from a stream port.
    /// Returns None if no data is available (should stall if blocking).
    pub fn pop_stream_input(&mut self, port: u8) -> Option<u32> {
        if (port as usize) < self.stream_input.len() {
            self.stream_input[port as usize].pop_front()
        } else {
            None
        }
    }

    /// Check if stream input has data for a port.
    pub fn has_stream_input(&self, port: u8) -> bool {
        if (port as usize) < self.stream_input.len() {
            !self.stream_input[port as usize].is_empty()
        } else {
            false
        }
    }

    /// Get stream input queue length for a port.
    pub fn stream_input_len(&self, port: u8) -> usize {
        if (port as usize) < self.stream_input.len() {
            self.stream_input[port as usize].len()
        } else {
            0
        }
    }

    /// Push a word to the stream output buffer for a port.
    ///
    /// Called by StreamWriteScalar when the core writes to a stream port.
    pub fn push_stream_output(&mut self, port: u8, value: u32) {
        if (port as usize) < self.stream_output.len() {
            self.stream_output[port as usize].push_back(value);
        }
    }

    /// Pop a word from the stream output buffer for a port.
    ///
    /// Called by the stream router to collect data from this tile.
    pub fn pop_stream_output(&mut self, port: u8) -> Option<u32> {
        if (port as usize) < self.stream_output.len() {
            self.stream_output[port as usize].pop_front()
        } else {
            None
        }
    }

    // === Lock_Request Register Handling ===

    /// Handle a Lock_Request register read.
    ///
    /// The address encodes the lock operation:
    /// - Lock_Id: bits [13:10] (compute) or [15:10] (memtile)
    /// - Acq_Rel: bit [9] (1=acquire, 0=release)
    /// - Change_Value: bits [8:2] (7-bit signed)
    ///
    /// Reading from this address performs the operation and returns:
    /// - Bit 0: 1 if operation succeeded, 0 if it would stall/fail
    fn handle_lock_request(&mut self, offset: u32, is_memtile: bool) -> u32 {
        use super::registers_spec::{memory_module as mm, mem_tile_module as mt};

        let base = if is_memtile { mt::LOCK_REQUEST_BASE } else { mm::LOCK_REQUEST_BASE };
        let addr = offset - base;

        // Extract fields from address
        let id_shift = if is_memtile { mt::LOCK_REQUEST_ID_SHIFT } else { mm::LOCK_REQUEST_ID_SHIFT };
        let id_mask = if is_memtile { mt::LOCK_REQUEST_ID_MASK } else { mm::LOCK_REQUEST_ID_MASK };

        let lock_id = ((addr >> id_shift) & id_mask) as usize;
        let is_acquire = (addr >> mm::LOCK_REQUEST_ACQ_REL_BIT) & 1 != 0;
        let change_raw = ((addr >> mm::LOCK_REQUEST_VALUE_SHIFT) & mm::LOCK_REQUEST_VALUE_MASK) as i8;

        // Sign-extend 7-bit value
        let change_value = if change_raw & 0x40 != 0 {
            change_raw | !0x7F_i8 // Sign extend
        } else {
            change_raw
        };

        // Bounds check
        let max_locks = if is_memtile { NUM_LOCKS_MEM_TILE } else { NUM_LOCKS_COMPUTE };
        if lock_id >= max_locks {
            return 0; // Invalid lock ID
        }

        // Perform the operation
        let result = if is_acquire {
            // Acquire: check if value >= abs(change), then apply delta
            // For acquire, change_value is typically negative (consuming tokens)
            self.locks[lock_id].acquire_with_value((-change_value).max(0) as u8, change_value)
        } else {
            // Release: just apply delta (typically positive, releasing tokens)
            self.locks[lock_id].release_with_value(change_value)
        };

        // Return success bit
        if matches!(result, LockResult::Success) { 1 } else { 0 }
    }

    /// Get lock overflow bits for a range of locks.
    ///
    /// Returns a bitmask where bit N is set if lock (start + N) has overflowed.
    fn get_lock_overflow_bits(&self, start: usize, end: usize) -> u32 {
        let mut bits = 0u32;
        for i in start..end.min(self.locks.len()) {
            if self.locks[i].overflow {
                bits |= 1 << (i - start);
            }
        }
        bits
    }

    /// Get lock underflow bits for a range of locks.
    ///
    /// Returns a bitmask where bit N is set if lock (start + N) has underflowed.
    fn get_lock_underflow_bits(&self, start: usize, end: usize) -> u32 {
        let mut bits = 0u32;
        for i in start..end.min(self.locks.len()) {
            if self.locks[i].underflow {
                bits |= 1 << (i - start);
            }
        }
        bits
    }

    /// Clear lock overflow bits for a range (write-to-clear behavior).
    fn clear_lock_overflow_bits(&mut self, start: usize, end: usize, bits: u32) {
        for i in start..end.min(self.locks.len()) {
            if bits & (1 << (i - start)) != 0 {
                self.locks[i].overflow = false;
            }
        }
    }

    /// Clear lock underflow bits for a range (write-to-clear behavior).
    fn clear_lock_underflow_bits(&mut self, start: usize, end: usize, bits: u32) {
        for i in start..end.min(self.locks.len()) {
            if bits & (1 << (i - start)) != 0 {
                self.locks[i].underflow = false;
            }
        }
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
    fn test_lock_max_value() {
        // Test clamping at creation
        let lock = Lock::new(100);
        assert_eq!(lock.value, Lock::MAX_VALUE); // Clamped to 63

        // Test saturation on release
        let mut lock = Lock::new(63);
        lock.release();
        assert_eq!(lock.value, 63); // Saturated at max

        // Test set
        let mut lock = Lock::new(0);
        lock.set(50);
        assert_eq!(lock.value, 50);
        lock.set(200);
        assert_eq!(lock.value, 63); // Clamped
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
        assert_eq!(std::mem::size_of::<Lock>(), 3); // u8 + 2 bools
        assert_eq!(std::mem::size_of::<DmaBufferDescriptor>(), 24);
        // DmaChannel: control u32 + start_queue u32 + 4x u8 + bool (3 padding) + status u32 = 20 bytes
        assert_eq!(std::mem::size_of::<DmaChannel>(), 20);
        assert_eq!(std::mem::size_of::<CoreState>(), 24);
    }

    #[test]
    fn test_program_memory_size() {
        // Verify program memory is 16KB per AM020
        assert_eq!(PROGRAM_MEMORY_SIZE, 16 * 1024);
    }

    #[test]
    fn test_lock_counts() {
        // Verify lock counts per AM020
        assert_eq!(NUM_LOCKS_COMPUTE, 16);
        assert_eq!(NUM_LOCKS_MEM_TILE, 64);
    }

    #[test]
    fn test_lock_acquire_with_value() {
        let mut lock = Lock::new(5);

        // Acquire with value >= 3, decrement by 2
        assert_eq!(lock.acquire_with_value(3, -2), LockResult::Success);
        assert_eq!(lock.value, 3);

        // Acquire with value >= 2, decrement by 1
        assert_eq!(lock.acquire_with_value(2, -1), LockResult::Success);
        assert_eq!(lock.value, 2);

        // Try to acquire with value >= 5 - should fail (only have 2)
        assert_eq!(lock.acquire_with_value(5, -3), LockResult::WouldUnderflow);
        assert_eq!(lock.value, 2); // Value unchanged

        // Acquire all remaining
        assert_eq!(lock.acquire_with_value(2, -2), LockResult::Success);
        assert_eq!(lock.value, 0);

        // Can't acquire when value is 0
        assert_eq!(lock.acquire_with_value(1, -1), LockResult::WouldUnderflow);
        assert_eq!(lock.value, 0);
    }

    #[test]
    fn test_lock_release_with_value() {
        let mut lock = Lock::new(0);

        // Release by 3
        assert_eq!(lock.release_with_value(3), LockResult::Success);
        assert_eq!(lock.value, 3);

        // Release by 10
        assert_eq!(lock.release_with_value(10), LockResult::Success);
        assert_eq!(lock.value, 13);

        // Release to max (60 + 13 = 73, saturates to 63)
        assert_eq!(lock.release_with_value(60), LockResult::WouldOverflow);
        assert_eq!(lock.value, 63);
        assert!(lock.overflow);
    }

    #[test]
    fn test_lock_release_negative_delta() {
        // Release with negative delta (unusual but supported)
        let mut lock = Lock::new(10);

        // "Release" with -3 is like an acquire
        assert_eq!(lock.release_with_value(-3), LockResult::Success);
        assert_eq!(lock.value, 7);

        // Try to underflow
        assert_eq!(lock.release_with_value(-10), LockResult::WouldUnderflow);
        assert_eq!(lock.value, 0);
        assert!(lock.underflow);
    }

    #[test]
    fn test_lock_flags_clear() {
        let mut lock = Lock::new(63);

        // Cause overflow
        lock.release_with_value(10);
        assert!(lock.overflow);
        assert!(!lock.underflow);
        assert!(lock.has_error());

        // Clear flags
        lock.clear_flags();
        assert!(!lock.overflow);
        assert!(!lock.has_error());
    }

    #[test]
    fn test_lock_acquire_equal() {
        // Test acquire_eq semantics (wait for exact match)
        let mut lock = Lock::new(2);

        // acquire_equal: wait for value == 1, should fail (value is 2)
        assert_eq!(lock.acquire_equal(1, -1), LockResult::WouldUnderflow);
        assert_eq!(lock.value, 2); // Unchanged

        // acquire_equal: wait for value == 2, should succeed
        assert_eq!(lock.acquire_equal(2, -2), LockResult::Success);
        assert_eq!(lock.value, 0); // Decremented to 0

        // Reset and test acquire_ge vs acquire_eq difference
        lock.set(3);

        // acquire_ge (acquire_with_value): wait for value >= 2, succeeds with 3
        assert_eq!(lock.acquire_with_value(2, -1), LockResult::Success);
        assert_eq!(lock.value, 2); // 3 - 1 = 2

        // acquire_eq: wait for value == 2, succeeds
        assert_eq!(lock.acquire_equal(2, -2), LockResult::Success);
        assert_eq!(lock.value, 0);

        // Reset to test exact-match requirement
        lock.set(5);

        // acquire_eq for value == 3 should fail (we have 5)
        assert_eq!(lock.acquire_equal(3, -3), LockResult::WouldUnderflow);
        assert_eq!(lock.value, 5); // Unchanged

        // acquire_ge for value >= 3 should succeed (we have 5)
        assert_eq!(lock.acquire_with_value(3, -2), LockResult::Success);
        assert_eq!(lock.value, 3); // 5 - 2 = 3
    }
}
