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
use super::trace_unit::TraceUnit;

/// Size of program memory (16 KB = 1024 x 128-bit instructions).
pub const PROGRAM_MEMORY_SIZE: usize = aie2_spec::PROGRAM_MEMORY_SIZE;

/// Parameters for constructing a Tile with correct per-tile-type sizing.
///
/// Production code derives these from `ArchConfig` (which reads mlir-aie
/// device models). Test convenience constructors use hardcoded defaults
/// matching NPU1/AIE2.
#[derive(Debug, Clone)]
pub struct TileParams {
    /// Data memory size in bytes (0 for shim, 64K for compute, 512K for mem tile).
    pub data_memory_size: usize,
    /// Number of locks (0 for shim, 16 for compute, 64 for mem tile).
    pub num_locks: usize,
    /// Number of DMA buffer descriptors (16 for compute, 48 for mem tile).
    pub num_bds: usize,
    /// Total DMA channels (4 for compute, 12 for mem tile).
    pub num_channels: usize,
}

impl TileParams {
    /// Default compute tile params (NPU1/AIE2).
    pub fn compute() -> Self {
        Self { data_memory_size: 64 * 1024, num_locks: 16, num_bds: 16, num_channels: 4 }
    }

    /// Default memory tile params (NPU1/AIE2).
    pub fn mem_tile() -> Self {
        Self { data_memory_size: 512 * 1024, num_locks: 64, num_bds: 48, num_channels: 12 }
    }

    /// Default shim tile params.
    pub fn shim() -> Self {
        Self { data_memory_size: 0, num_locks: 0, num_bds: 0, num_channels: 0 }
    }
}

/// Result of a lock operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockResult {
    /// Operation succeeded
    Success,
    /// Operation failed - would underflow (value would go below -64)
    WouldUnderflow,
    /// Operation failed - would overflow (value would exceed +63)
    WouldOverflow,
}

/// Lock state.
///
/// AIE2 uses semaphore locks with acquire/release semantics.
/// Lock value is 7-bit signed (-64 to +63). Per AM025, the
/// Lock_Value register field is bits [6:0] with sign extension.
///
/// # Semaphore Model (AM025)
///
/// Lock operations use a change_value parameter:
/// - Acquire: Waits until condition met, then applies change
/// - Release: Applies change_value, clamping to [MIN_VALUE, MAX_VALUE]
///
/// The Lock_Request register format (AM025):
/// - Lock_Id [13:10]: Which lock (0-15 for compute, 0-63 for mem tile)
/// - Acq_Rel [9]: 1 = acquire (blocking), 0 = release (non-blocking)
/// - Change_Value [8:2]: Signed 7-bit delta (-64 to +63)
/// - Request_Result [0]: 0 = failed, 1 = succeeded
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Lock {
    /// Current lock value (semaphore count, -64 to +63).
    /// 7-bit signed per AM025 Lock_Value register field.
    pub value: i8,
    /// Overflow flag - set when release would exceed MAX_VALUE
    pub overflow: bool,
    /// Underflow flag - set when acquire would go below MIN_VALUE
    pub underflow: bool,
}

impl Lock {
    /// Maximum lock value (7-bit signed: +63).
    ///
    /// This compile-time constant is validated against the mlir-aie device
    /// model at startup (see `model::validate_against_spec()`). It is kept
    /// as a const for hot-path efficiency in lock acquire/release.
    pub const MAX_VALUE: i8 = 63;

    /// Minimum lock value (7-bit signed: -64).
    pub const MIN_VALUE: i8 = -64;

    /// Create a new lock with initial value (clamped to -64..+63)
    #[inline]
    pub fn new(value: i8) -> Self {
        Self {
            value: value.clamp(Self::MIN_VALUE, Self::MAX_VALUE),
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

    /// Release the lock (increment, clamping at MAX_VALUE).
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
    pub fn acquire_with_value(&mut self, expected_value: i8, delta: i8) -> LockResult {
        if self.value < expected_value {
            // Not enough value - operation would stall
            return LockResult::WouldUnderflow;
        }

        // Apply delta (convert to i16 for safe arithmetic)
        let new_value = (self.value as i16) + (delta as i16);

        if new_value < Self::MIN_VALUE as i16 {
            self.underflow = true;
            return LockResult::WouldUnderflow;
        }

        if new_value > Self::MAX_VALUE as i16 {
            // This shouldn't happen for acquire (negative delta), but handle it
            self.overflow = true;
            self.value = Self::MAX_VALUE;
            return LockResult::WouldOverflow;
        }

        self.value = new_value as i8;
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
    pub fn acquire_equal(&mut self, expected_value: i8, delta: i8) -> LockResult {
        if self.value != expected_value {
            // Value doesn't match exactly - operation would stall
            return LockResult::WouldUnderflow;
        }

        // Apply delta (convert to i16 for safe arithmetic)
        let new_value = (self.value as i16) + (delta as i16);

        if new_value < Self::MIN_VALUE as i16 {
            self.underflow = true;
            return LockResult::WouldUnderflow;
        }

        if new_value > Self::MAX_VALUE as i16 {
            self.overflow = true;
            self.value = Self::MAX_VALUE;
            return LockResult::WouldOverflow;
        }

        self.value = new_value as i8;
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

        if new_value < Self::MIN_VALUE as i16 {
            self.underflow = true;
            self.value = Self::MIN_VALUE;
            return LockResult::WouldUnderflow;
        }

        if new_value > Self::MAX_VALUE as i16 {
            self.overflow = true;
            self.value = Self::MAX_VALUE;
            return LockResult::WouldOverflow;
        }

        self.value = new_value as i8;
        LockResult::Success
    }

    /// Set the lock value directly (clamped to -64..+63)
    #[inline]
    pub fn set(&mut self, value: i8) {
        self.value = value.clamp(Self::MIN_VALUE, Self::MAX_VALUE);
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
    ///
    /// Uses the compute tile status layout. The DmaEngine.get_channel_status()
    /// method selects the correct layout per tile type; this is a convenience
    /// for the DmaChannel struct which stores a copy of the status word.
    pub fn set_cur_bd(&mut self, bd: u8) {
        let layout = &super::regdb::device_reg_layout().memory_status;
        self.status = layout.cur_bd.insert(self.status, bd as u32);
    }

    /// Update status register: Channel_Running
    pub fn set_channel_running(&mut self, running: bool) {
        let layout = &super::regdb::device_reg_layout().memory_status;
        if running {
            self.status = layout.channel_running.set_bit(self.status);
        } else {
            self.status &= !(layout.channel_running.mask << layout.channel_running.shift);
        }
    }

    /// Update status register: State bits (00=IDLE, 01=STARTING, 10=RUNNING)
    pub fn set_state(&mut self, state: u8) {
        let layout = &super::regdb::device_reg_layout().memory_status;
        self.status = layout.status.insert(self.status, state as u32);
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
/// Control packet state machine for TileControl port.
///
/// The TileControl port receives control packets from the stream switch.
/// Each packet has a header word followed by 1-4 data words (beats).
///
/// Control packet header format (AM020 Table 3):
/// - Bits 19:0 = Address (tile-local register offset)
/// - Bits 21:20 = Length (00=1 beat, 01=2, 10=3, 11=4)
/// - Bits 23:22 = Operation (00=write, 01=read+return, 10=write_incr, 11=block_write)
/// - Bits 30:24 = Stream_ID (for response routing)
/// - Bit 31 = Parity
#[derive(Debug, Clone, Default)]
pub enum ControlPacketState {
    /// Waiting for stream header (when master port Drop_Header=false).
    /// The stream switch forwards the routing header to TileCtrl; we must
    /// consume and discard it before the actual control packet header.
    WaitingForStreamHeader,
    /// Waiting for control packet header (stream header already consumed
    /// or was dropped by the switch).
    #[default]
    Idle,
    /// Collecting data beats after header
    Collecting {
        /// Target register address (bits 19:0 of header)
        address: u32,
        /// Operation: 0=write, 1=read_return, 2=write_incr, 3=block_write
        operation: u8,
        /// Stream ID for response routing (bits 30:24)
        response_id: u8,
        /// Total beats expected (1-4)
        beats_total: u8,
        /// Accumulated data words
        data: Vec<u32>,
    },
}

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

    /// Lock states (sized from TileParams: 16 for compute, 64 for mem tile, 0 for shim)
    pub locks: Vec<Lock>,

    /// Lock value snapshot for cycle-accurate arbitration.
    /// All lock operations within a cycle check against this snapshot,
    /// ensuring all requestors see the same values regardless of processing order.
    lock_snapshot: Vec<i8>,

    /// Pending lock deltas to apply at end of cycle.
    /// Releases add positive deltas, acquires add negative deltas.
    lock_deltas: Vec<i8>,

    // === Warm data (accessed during DMA) ===

    /// DMA buffer descriptors (sized from TileParams: 16 for compute, 48 for mem tile)
    pub dma_bds: Vec<DmaBufferDescriptor>,

    /// DMA channels (sized from TileParams: 4 for compute, 12 for mem tile)
    pub dma_channels: Vec<DmaChannel>,

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

    /// Control packet state machine for TileCtrl port.
    pub ctrl_pkt_state: ControlPacketState,

    /// Whether the TileCtrl master port drops stream headers.
    /// Set from the master port's packet config during CDO processing.
    /// When false, the handler must consume and discard the stream header
    /// before parsing the control packet header.
    pub ctrl_pkt_drop_header: bool,

    /// Shim Mux: which switchbox South slave port each DMA MM2S channel feeds.
    /// Parsed from Mux_Config register (0x1F000). Index 0 = MM2S ch0, 1 = MM2S ch1.
    /// Value is the switchbox slave port index (e.g., 5 for South3).
    pub shim_mux_mm2s_slaves: [Option<usize>; 2],

    /// Shim Mux: which switchbox South master port feeds each DMA S2MM channel.
    /// Parsed from Demux_Config register (0x1F004). Index 0 = S2MM ch0, 1 = S2MM ch1.
    /// Value is the switchbox master port index (e.g., 2 for South0).
    pub shim_mux_s2mm_masters: [Option<usize>; 2],

    // === Trace Units ===

    /// Core module trace unit (compute tiles only).
    ///
    /// Configured by writes to offsets 0x340D0-0x340E4. Monitors core module
    /// events (instructions, stalls) and emits binary trace packets through
    /// the stream switch Trace slave port (slave[23] = AIE_TRACE).
    pub core_trace: TraceUnit,

    /// Memory module trace unit (compute and mem tiles).
    ///
    /// Configured by writes to offsets 0x140D0-0x140E4 (compute) or
    /// 0x940D0-0x940E4 (memtile). Monitors memory module events (DMA, locks)
    /// and emits through Trace slave port (slave[24] = MEM_TRACE for compute,
    /// slave[17] = TRACE for memtile).
    pub mem_trace: TraceUnit,

    /// Stream switch event port selection (8 logical event ports).
    ///
    /// Each entry maps a logical event port (0-7) to a physical stream switch
    /// port. `None` means the port is not configured. `Some((port_idx, is_master))`
    /// identifies the physical port to monitor for PORT_RUNNING/IDLE/STALLED events.
    ///
    /// Configured by Event Port Selection registers:
    /// - Compute/Shim: 0x3FF00 (ports 0-3), 0x3FF04 (ports 4-7)
    /// - MemTile: 0xB0F00 (ports 0-3), 0xB0F04 (ports 4-7)
    ///
    /// Register encoding per 8-bit slot: bit 5 = master (1) or slave (0),
    /// bits 4:0 = port index.
    pub event_port_selection: [Option<(u8, bool)>; 8],

    // === Memory Bank Conflict Detection ===

    /// Bitmask of memory banks accessed by DMA during this cycle.
    /// Bit N set = bank N was accessed. Supports up to 16 banks (MemTile).
    /// Reset at the start of each coordinator step.
    pub cycle_dma_banks: u16,

    // === Edge Detection ===

    /// Core module edge detectors (two independent circuits).
    /// Configured by Edge_Detection_event_control register at 0x34408 (compute).
    /// Monitor core module event signals for rising/falling transitions.
    pub core_edge_detectors: [EdgeDetector; 2],

    /// Memory module edge detectors (two independent circuits).
    /// Configured by Edge_Detection_event_control register at:
    /// - 0x14408 (compute tile memory module)
    /// - 0x94408 (MemTile)
    pub mem_edge_detectors: [EdgeDetector; 2],
}

/// Single edge detection circuit.
///
/// Monitors one event signal and generates an EDGE_DETECTION_EVENT when
/// the signal transitions (rising, falling, or both). Each module has two
/// independent edge detectors (SelectId 0 and 1).
///
/// Register layout (Edge_Detection_event_control):
/// - Event 0: bits [6:0] event select, bit 9 rising, bit 10 falling
/// - Event 1: bits [22:16] event select, bit 25 rising, bit 26 falling
/// - MemTile: 8-bit event fields (bits [7:0] and [23:16])
#[derive(Debug, Clone, Copy)]
pub struct EdgeDetector {
    /// Hardware event ID to monitor (0 = disabled).
    pub input_event: u8,
    /// Fire on 0->1 transition.
    pub trigger_rising: bool,
    /// Fire on 1->0 transition.
    pub trigger_falling: bool,
    /// Whether the monitored event was active last cycle.
    prev_active: bool,
    /// Whether the monitored event was active this cycle (accumulates
    /// during event notification, reset at end of cycle).
    curr_active: bool,
}

impl Default for EdgeDetector {
    fn default() -> Self {
        Self {
            input_event: 0,
            trigger_rising: false,
            trigger_falling: false,
            prev_active: false,
            curr_active: false,
        }
    }
}

impl Tile {
    /// Create a new tile of the specified type with explicit parameters.
    ///
    /// Production code should use the `ArchConfig`-derived params (via
    /// `TileArray::new()`). Test code can use `Tile::compute()` etc. for
    /// convenience with NPU1/AIE2 defaults.
    pub fn new(tile_type: TileType, col: u8, row: u8, params: &TileParams) -> Self {
        let program_memory = match tile_type {
            TileType::Compute => Some(Box::new([0u8; PROGRAM_MEMORY_SIZE])),
            _ => None,
        };

        Self {
            tile_type,
            col,
            row,
            core: CoreState::default(),
            locks: vec![Lock::default(); params.num_locks],
            lock_snapshot: vec![0; params.num_locks],
            lock_deltas: vec![0; params.num_locks],
            dma_bds: vec![DmaBufferDescriptor::default(); params.num_bds],
            dma_channels: vec![DmaChannel::default(); params.num_channels],
            stream_input: Default::default(),
            stream_output: Default::default(),
            stream_switch: match tile_type {
                TileType::Shim => FunctionalStreamSwitch::new_shim_tile(col),
                TileType::MemTile => FunctionalStreamSwitch::new_mem_tile(col, row),
                TileType::Compute => FunctionalStreamSwitch::new_compute_tile(col, row),
            },
            data_memory: vec![0u8; params.data_memory_size].into_boxed_slice(),
            program_memory,
            registers: std::collections::HashMap::new(),
            ctrl_pkt_state: ControlPacketState::Idle,
            ctrl_pkt_drop_header: true,
            shim_mux_mm2s_slaves: [None; 2],
            shim_mux_s2mm_masters: [None; 2],
            core_trace: TraceUnit::new(col, row),
            mem_trace: TraceUnit::new(col, row),
            event_port_selection: [None; 8],
            cycle_dma_banks: 0,
            core_edge_detectors: [EdgeDetector::default(); 2],
            mem_edge_detectors: [EdgeDetector::default(); 2],
        }
    }

    /// Create a compute tile with NPU1/AIE2 default parameters.
    ///
    /// Convenience constructor for tests. Production code should use
    /// `Tile::new()` with ArchConfig-derived params.
    #[inline]
    pub fn compute(col: u8, row: u8) -> Self {
        Self::new(TileType::Compute, col, row, &TileParams::compute())
    }

    /// Create a memory tile with NPU1/AIE2 default parameters.
    #[inline]
    pub fn mem_tile(col: u8, row: u8) -> Self {
        Self::new(TileType::MemTile, col, row, &TileParams::mem_tile())
    }

    /// Create a shim tile with NPU1/AIE2 default parameters.
    #[inline]
    pub fn shim(col: u8, row: u8) -> Self {
        Self::new(TileType::Shim, col, row, &TileParams::shim())
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

    // === Memory Bank Conflict Detection ===

    /// Number of memory banks for this tile type.
    pub fn num_banks(&self) -> usize {
        match self.tile_type {
            TileType::Compute => crate::device::aie2_spec::COMPUTE_TILE_MEMORY_BANKS,
            TileType::MemTile => crate::device::aie2_spec::MEMTILE_MEMORY_BANKS,
            TileType::Shim => 0,
        }
    }

    /// Record that DMA accessed the given memory address range this cycle.
    /// Call from DMA transfer methods during Phase 2.
    #[inline]
    pub fn record_dma_bank_access(&mut self, addr: u32, bytes: usize) {
        let nb = self.num_banks();
        if nb > 0 {
            self.cycle_dma_banks |= crate::device::aie2_spec::banks_for_access(addr, bytes, nb);
        }
    }

    /// Reset bank tracking for a new cycle. Call at the start of each step.
    #[inline]
    pub fn reset_bank_tracking(&mut self) {
        self.cycle_dma_banks = 0;
    }

    // === Edge Detection ===

    /// Notify a core module event for both tracing and edge detection.
    ///
    /// Forwards the event to `core_trace.notify_event()` and marks it as
    /// active for the core module edge detectors this cycle.
    #[inline]
    pub fn notify_core_trace_event(&mut self, hw_id: u8, cycle: u64) {
        self.core_trace.notify_event(hw_id, cycle);
        for det in &mut self.core_edge_detectors {
            if det.input_event == hw_id {
                det.curr_active = true;
            }
        }
    }

    /// Notify a memory module event for both tracing and edge detection.
    ///
    /// Forwards the event to `mem_trace.notify_event()` and marks it as
    /// active for the memory module edge detectors this cycle.
    #[inline]
    pub fn notify_mem_trace_event(&mut self, hw_id: u8, cycle: u64) {
        self.mem_trace.notify_event(hw_id, cycle);
        for det in &mut self.mem_edge_detectors {
            if det.input_event == hw_id {
                det.curr_active = true;
            }
        }
    }

    /// Evaluate edge detectors and fire generated events to trace units.
    ///
    /// Call once per cycle after all raw events have been notified.
    /// Compares current vs previous signal state and fires
    /// EDGE_DETECTION_EVENT_0/1 on detected transitions.
    pub fn evaluate_edge_detectors(&mut self, cycle: u64) {
        // Core module edge detectors -> core_trace
        for i in 0..2 {
            let det = &self.core_edge_detectors[i];
            let fire = (det.trigger_rising && det.curr_active && !det.prev_active)
                || (det.trigger_falling && !det.curr_active && det.prev_active);
            if fire {
                let hw_id = crate::trace::core_edge_detection_event_hw_id(i as u8);
                self.core_trace.notify_event(hw_id, cycle);
            }
        }
        // Memory module edge detectors -> mem_trace
        for i in 0..2 {
            let det = &self.mem_edge_detectors[i];
            let fire = (det.trigger_rising && det.curr_active && !det.prev_active)
                || (det.trigger_falling && !det.curr_active && det.prev_active);
            if fire {
                let hw_id = if self.is_mem_tile() {
                    crate::trace::memtile_edge_detection_event_hw_id(i as u8)
                } else {
                    crate::trace::mem_edge_detection_event_hw_id(i as u8)
                };
                self.mem_trace.notify_event(hw_id, cycle);
            }
        }
        // Advance state: current becomes previous, reset current
        for det in &mut self.core_edge_detectors {
            det.prev_active = det.curr_active;
            det.curr_active = false;
        }
        for det in &mut self.mem_edge_detectors {
            det.prev_active = det.curr_active;
            det.curr_active = false;
        }
    }

    /// Configure edge detectors from a register write.
    ///
    /// Parses the Edge_Detection_event_control register value and updates
    /// the specified detector pair. `is_memtile` controls whether event
    /// fields are 7-bit (compute/shim) or 8-bit (MemTile).
    fn configure_edge_detectors(detectors: &mut [EdgeDetector; 2], value: u32, is_memtile: bool) {
        // Event 0: bits [6:0] or [7:0], rising=bit 9, falling=bit 10
        let event_mask_0: u32 = if is_memtile { 0xFF } else { 0x7F };
        detectors[0].input_event = (value & event_mask_0) as u8;
        detectors[0].trigger_rising = (value & (1 << 9)) != 0;
        detectors[0].trigger_falling = (value & (1 << 10)) != 0;

        // Event 1: bits [22:16] or [23:16], rising=bit 25, falling=bit 26
        let event_mask_1: u32 = if is_memtile { 0xFF } else { 0x7F };
        detectors[1].input_event = ((value >> 16) & event_mask_1) as u8;
        detectors[1].trigger_rising = (value & (1 << 25)) != 0;
        detectors[1].trigger_falling = (value & (1 << 26)) != 0;

        log::debug!(
            "Edge detectors configured: det0(event={}, rise={}, fall={}), det1(event={}, rise={}, fall={})",
            detectors[0].input_event, detectors[0].trigger_rising, detectors[0].trigger_falling,
            detectors[1].input_event, detectors[1].trigger_rising, detectors[1].trigger_falling,
        );
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
        // Debug: log any non-zero deltas that are being cleared for Tile(0,1)
        if self.col == 0 && self.row == 1 {
            let non_zero: Vec<_> = self.lock_deltas.iter().enumerate()
                .filter(|(_, &d)| d != 0)
                .map(|(i, &d)| format!("{}:{}", i, d))
                .collect();
            if !non_zero.is_empty() {
                log::debug!("Tile(0,1) begin_lock_cycle: CLEARING non-zero deltas = [{}] self_ptr={:p}",
                    non_zero.join(", "), self);
            }
        }
        // Snapshot current values and clear deltas from previous cycle
        let num_locks = self.locks.len();
        for i in 0..num_locks {
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
    pub fn try_acquire_snapshot(&mut self, lock_id: usize, expected: i8, delta: i8, equal_mode: bool) -> LockResult {
        if lock_id >= self.locks.len() {
            return LockResult::WouldUnderflow;
        }

        // Calculate effective value: snapshot + any deltas already recorded this cycle
        let snapshot_val = self.lock_snapshot[lock_id];
        let delta_val = self.lock_deltas[lock_id];
        let effective = (snapshot_val as i16 + delta_val as i16) as i8;

        let success = if equal_mode {
            effective == expected
        } else {
            effective >= expected
        };

        // Debug: log acquire attempts for lock 1 on Tile(0,1)
        if self.col == 0 && self.row == 1 && lock_id == 1 {
            log::debug!("Tile(0,1) try_acquire lock 1: snapshot={} delta={} effective={} expected={} success={}",
                snapshot_val, delta_val, effective, expected, success);
        }

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
        if lock_id < self.locks.len() {
            let old = self.lock_deltas[lock_id];
            self.lock_deltas[lock_id] = self.lock_deltas[lock_id].saturating_add(delta);
            log::debug!("Tile({},{}) release_snapshot lock {} delta {} (old_delta={}, new_delta={}) self_ptr={:p}",
                self.col, self.row, lock_id, delta, old, self.lock_deltas[lock_id], self);
        }
    }

    /// End a lock cycle by applying accumulated deltas.
    ///
    /// Call this at the end of each simulation cycle, after all lock operations.
    /// Applies all recorded deltas to the actual lock values, with clamping.
    #[inline]
    pub fn end_lock_cycle(&mut self) {
        // Debug: log all non-zero deltas for Tile(0,1)
        if self.col == 0 && self.row == 1 {
            let non_zero: Vec<_> = self.lock_deltas.iter().enumerate()
                .filter(|(_, &d)| d != 0)
                .map(|(i, &d)| format!("{}:{}", i, d))
                .collect();
            if !non_zero.is_empty() {
                log::debug!("Tile(0,1) end_lock_cycle: non-zero deltas = [{}] self_ptr={:p}",
                    non_zero.join(", "), self);
            }
        }
        let num_locks = self.locks.len();
        for i in 0..num_locks {
            let delta = self.lock_deltas[i];
            if delta != 0 {
                let old_value = self.locks[i].value;
                let new_value = (old_value as i16) + (delta as i16);
                if new_value < Lock::MIN_VALUE as i16 {
                    self.locks[i].underflow = true;
                    self.locks[i].value = Lock::MIN_VALUE;
                    log::debug!("Tile({},{}) lock {} end_cycle: {}+({})=MIN (clamped, underflow)",
                        self.col, self.row, i, old_value, delta);
                } else if new_value > Lock::MAX_VALUE as i16 {
                    self.locks[i].overflow = true;
                    self.locks[i].value = Lock::MAX_VALUE;
                    log::debug!("Tile({},{}) lock {} end_cycle: {}+({})->MAX (overflow)",
                        self.col, self.row, i, old_value, delta);
                } else {
                    self.locks[i].value = new_value as i8;
                    log::debug!("Tile({},{}) lock {} end_cycle: {}+({})={}",
                        self.col, self.row, i, old_value, delta, new_value);
                }
            }
        }
    }

    /// Get the snapshot value for a lock (for debugging/logging).
    #[inline]
    pub fn lock_snapshot_value(&self, lock_id: usize) -> i8 {
        if lock_id < self.locks.len() {
            self.lock_snapshot[lock_id]
        } else {
            0
        }
    }

    /// Get the pending delta for a lock (for debugging/logging).
    #[inline]
    pub fn lock_pending_delta(&self, lock_id: usize) -> i8 {
        if lock_id < self.locks.len() {
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
    /// Get an immutable reference to the register map.
    ///
    /// Used by mask_write_register in state.rs to read current values without
    /// triggering side effects (unlike read_register which executes lock operations).
    pub fn registers_ref(&self) -> &std::collections::HashMap<u32, u32> {
        &self.registers
    }

    pub fn write_register(&mut self, offset: u32, value: u32) {
        // Always store in the register map for later retrieval
        self.registers.insert(offset, value);

        // For specific ranges, also update internal state
        // DMA BD range: 0x1D000-0x1D3FF (16 BDs × 32 bytes each)
        if (0x1D000..0x1D200).contains(&offset) {
            let bd_offset = offset - 0x1D000;
            let bd_index = (bd_offset / 0x20) as usize;
            let reg_in_bd = (bd_offset % 0x20) as usize / 4;

            if bd_index < self.dma_bds.len() {
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

        // Lock value registers (from AM025 regdb: Lock0_value @ 0x1F000, stride 0x10)
        let reg_layout = super::regdb::device_reg_layout();
        let lock_base = if self.tile_type == TileType::MemTile {
            reg_layout.memtile_lock_base
        } else {
            reg_layout.memory_lock_base
        };
        let lock_stride = if self.tile_type == TileType::MemTile {
            reg_layout.memtile_lock_stride
        } else {
            reg_layout.memory_lock_stride
        };
        let lock_end = lock_base + (self.locks.len() as u32) * lock_stride;
        if (lock_base..lock_end).contains(&offset) {
            let lock_id = ((offset - lock_base) / lock_stride) as usize;
            let sub_offset = (offset - lock_base) % lock_stride;
            if lock_id < self.locks.len() && sub_offset == 0 {
                // Lock_value field is bits 6:0 (7-bit signed, -64 to +63)
                let raw7 = (value & 0x7F) as u8;
                let signed = if raw7 & 0x40 != 0 {
                    raw7 as i8 | !0x7F_i8  // sign-extend bit 6
                } else {
                    raw7 as i8
                };
                self.locks[lock_id].set(signed);
                log::info!("Tile ({},{}) write_register: Lock{} = {} (raw=0x{:08X})",
                    self.col, self.row, lock_id, signed, value);
            }
        }

        // DMA channel control: 0x1D200-0x1D3FF
        if (0x1D200..0x1D400).contains(&offset) {
            let ch_offset = offset - 0x1D200;
            let ch_index = (ch_offset / 0x8) as usize;
            if ch_index < self.dma_channels.len() {
                let ch = &mut self.dma_channels[ch_index];
                if ch_offset % 0x8 == 0 {
                    ch.control = value;
                } else {
                    ch.start_queue = value;
                }
            }
        }

        // Shim Mux config registers (shim tiles only)
        // Mux_Config (0x1F000): selects source for switchbox South slave ports
        //   Each 2-bit field: 0=South/PL, 1=DMA, 2=NoC
        // Shim Mux/Demux config registers: select DMA/PL/NoC sources/dests
        // for switchbox South ports. Offsets and field layout are data-driven.
        let reg_layout = super::regdb::device_reg_layout();
        if self.tile_type == TileType::Shim {
            if offset == reg_layout.shim_mux.mux_offset {
                self.parse_shim_mux_config(value);
            } else if offset == reg_layout.shim_mux.demux_offset {
                self.parse_shim_demux_config(value);
            }
        }

        // Lock status registers (write-to-clear)
        // Writing 1 to a bit clears that lock's overflow/underflow status
        if self.is_mem_tile() {
            if offset == reg_layout.memtile_locks_overflow_0 {
                self.clear_lock_overflow_bits(0, 32, value);
            } else if offset == reg_layout.memtile_locks_overflow_1 {
                self.clear_lock_overflow_bits(32, 64, value);
            } else if offset == reg_layout.memtile_locks_underflow_0 {
                self.clear_lock_underflow_bits(0, 32, value);
            } else if offset == reg_layout.memtile_locks_underflow_1 {
                self.clear_lock_underflow_bits(32, 64, value);
            }
        } else if self.is_compute() {
            if offset == reg_layout.memory_locks_overflow {
                self.clear_lock_overflow_bits(0, 16, value);
            } else if offset == reg_layout.memory_locks_underflow {
                self.clear_lock_underflow_bits(0, 16, value);
            }
        }

        // Trace register routing.
        //
        // Core module trace registers (compute tiles):
        //   0x340D0 (Control0), 0x340D4 (Control1), 0x340E0 (Event0), 0x340E4 (Event1)
        //
        // Memory module trace registers (compute tiles):
        //   0x140D0 (Control0), 0x140D4 (Control1), 0x140E0 (Event0), 0x140E4 (Event1)
        //
        // MemTile trace registers:
        //   0x940D0 (Control0), 0x940D4 (Control1), 0x940E0 (Event0), 0x940E4 (Event1)
        if self.is_compute() {
            // Core module trace: base 0x340D0
            if offset >= 0x340D0 && offset <= 0x340E4 {
                let trace_offset = offset - 0x340D0;
                self.core_trace.write_register(trace_offset, value);
            }
            // Memory module trace: base 0x140D0
            if offset >= 0x140D0 && offset <= 0x140E4 {
                let trace_offset = offset - 0x140D0;
                self.mem_trace.write_register(trace_offset, value);
            }
        } else if self.is_mem_tile() {
            // MemTile trace: base 0x940D0
            if offset >= 0x940D0 && offset <= 0x940E4 {
                let trace_offset = offset - 0x940D0;
                self.mem_trace.write_register(trace_offset, value);
            }
        }

        // Edge detection event control registers.
        //
        // Each module has one register with two independent edge detectors.
        // Compute tile: 0x34408 (core module), 0x14408 (memory module)
        // MemTile: 0x94408
        // Shim: 0x34408 (PL module, handled in shim tracing)
        if self.is_compute() {
            if offset == 0x34408 {
                Self::configure_edge_detectors(&mut self.core_edge_detectors, value, false);
            }
            if offset == 0x14408 {
                Self::configure_edge_detectors(&mut self.mem_edge_detectors, value, false);
            }
        } else if self.is_mem_tile() {
            if offset == 0x94408 {
                Self::configure_edge_detectors(&mut self.mem_edge_detectors, value, true);
            }
        }

        // Event port selection registers.
        //
        // These configure which physical stream switch ports map to logical
        // event ports 0-7 for PORT_RUNNING/IDLE/STALLED trace events.
        //
        // Register layout: 4 x 8-bit entries per register.
        // Per entry: bit 5 = master (1) / slave (0), bits 4:0 = port index.
        //
        // Compute/Shim: 0x3FF00 (ports 0-3), 0x3FF04 (ports 4-7)
        // MemTile:      0xB0F00 (ports 0-3), 0xB0F04 (ports 4-7)
        let port_sel_base = match self.tile_type {
            TileType::Compute | TileType::Shim => Some((0x3FF00u32, 0x3FF04u32)),
            TileType::MemTile => Some((0xB0F00, 0xB0F04)),
        };
        if let Some((reg0, reg1)) = port_sel_base {
            if offset == reg0 || offset == reg1 {
                let base_slot = if offset == reg0 { 0 } else { 4 };
                for i in 0..4usize {
                    let byte = ((value >> (i * 8)) & 0xFF) as u8;
                    let port_idx = byte & 0x1F;
                    let is_master = (byte & 0x20) != 0;
                    self.event_port_selection[base_slot + i] = Some((port_idx, is_master));
                }
                log::debug!(
                    "Tile({},{}) event port sel @0x{:X}: {:?}",
                    self.col, self.row, offset, &self.event_port_selection[base_slot..base_slot+4]
                );
            }
        }
    }

    /// Read a 32-bit value from a register offset.
    ///
    /// Returns 0 for unwritten registers (default state).
    pub fn read_register(&mut self, offset: u32) -> u32 {
        use super::registers_spec::{memory_module as mm, mem_tile_module as mt};
        let reg_layout = super::regdb::device_reg_layout();

        // Lock_Request register - address encodes operation parameters
        // Reading performs the lock operation and returns result
        if self.is_mem_tile() {
            if (mt::LOCK_REQUEST_BASE..mt::LOCK_REQUEST_END).contains(&offset) {
                return self.handle_lock_request(offset, true);
            }
            // Lock status registers
            if offset == reg_layout.memtile_locks_overflow_0 {
                return self.get_lock_overflow_bits(0, 32);
            }
            if offset == reg_layout.memtile_locks_overflow_1 {
                return self.get_lock_overflow_bits(32, 64);
            }
            if offset == reg_layout.memtile_locks_underflow_0 {
                return self.get_lock_underflow_bits(0, 32);
            }
            if offset == reg_layout.memtile_locks_underflow_1 {
                return self.get_lock_underflow_bits(32, 64);
            }
        } else if self.is_compute() {
            if (mm::LOCK_REQUEST_BASE..mm::LOCK_REQUEST_END).contains(&offset) {
                return self.handle_lock_request(offset, false);
            }
            // Lock status registers
            if offset == reg_layout.memory_locks_overflow {
                return self.get_lock_overflow_bits(0, 16);
            }
            if offset == reg_layout.memory_locks_underflow {
                return self.get_lock_underflow_bits(0, 16);
            }
        }

        // Check specific subsystem state first
        // DMA BD range: 0x1D000-0x1D1FF
        if (0x1D000..0x1D200).contains(&offset) {
            let bd_offset = offset - 0x1D000;
            let bd_index = (bd_offset / 0x20) as usize;
            let reg_in_bd = (bd_offset % 0x20) as usize / 4;

            if bd_index < self.dma_bds.len() {
                let bd = &self.dma_bds[bd_index];
                // Legacy struct has 6 fields; words 6-7 (shim/memtile iteration
                // and lock/valid) fall through to the register HashMap below.
                match reg_in_bd {
                    0 => return bd.addr_low,
                    1 => return bd.addr_high,
                    2 => return bd.length,
                    3 => return bd.control,
                    4 => return bd.d0,
                    5 => return bd.d1,
                    _ => {} // Fall through to register map for words 6-7
                }
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

    // === Control Packet Handling ===

    /// Parse Shim Mux_Config register to find DMA MM2S South slave mapping.
    ///
    /// The Shim Mux selects which source (PL/DMA/NoC) feeds each switchbox South
    /// slave port. DMA MM2S output enters the switchbox through a South slave.
    ///
    /// Field layout and port mappings are derived from the AM025 register database.
    /// Select values: 0=South/PL, 1=DMA, 2=NoC
    fn parse_shim_mux_config(&mut self, value: u32) {
        let mux = &super::regdb::device_reg_layout().shim_mux;

        // Reset mapping (register may be rewritten with different config)
        self.shim_mux_mm2s_slaves = [None; 2];

        let mut dma_ch = 0usize;
        for mf in &mux.mux_fields {
            let select = mf.field.extract(value);
            if select == 1 && dma_ch < 2 {
                // DMA source -> this South slave gets MM2S output
                self.shim_mux_mm2s_slaves[dma_ch] = Some(mf.port_index);
                log::info!("Shim Mux ({},{}): MM2S ch{} -> slave[{}] ({})",
                    self.col, self.row, dma_ch, mf.port_index, mf.field.name);
                dma_ch += 1;
            }
        }
    }

    /// Parse Shim Demux_Config register to find DMA S2MM South master mapping.
    ///
    /// The Shim Demux selects which destination (PL/DMA/NoC) receives switchbox
    /// South master output. DMA S2MM input comes from a South master.
    ///
    /// Field layout and port mappings are derived from the AM025 register database.
    /// Select values: 0=South/PL, 1=DMA, 2=NoC
    fn parse_shim_demux_config(&mut self, value: u32) {
        let mux = &super::regdb::device_reg_layout().shim_mux;

        // Reset mapping
        self.shim_mux_s2mm_masters = [None; 2];

        let mut dma_ch = 0usize;
        for df in &mux.demux_fields {
            let select = df.field.extract(value);
            if select == 1 && dma_ch < 2 {
                self.shim_mux_s2mm_masters[dma_ch] = Some(df.port_index);
                log::info!("Shim Mux ({},{}): S2MM ch{} <- master[{}] ({})",
                    self.col, self.row, dma_ch, df.port_index, df.field.name);
                dma_ch += 1;
            }
        }
    }

    /// Process a data word arriving at the TileControl port.
    ///
    /// The TileControl master port delivers control packets that reprogram
    /// tile registers at runtime. Each packet consists of:
    /// 1. A control header word (address, operation, beat count)
    /// 2. One or more data words (the register values to write)
    ///
    /// Control packet header format (AM020 Table 3):
    /// - Bits 19:0 = Address (tile-local register offset)
    /// - Bits 21:20 = Length (00=1 beat, 01=2, 10=3, 11=4)
    /// - Bits 23:22 = Operation (00=write, 01=read, 10=write_incr, 11=block_write)
    /// - Bits 30:24 = Stream_ID (for response routing)
    /// - Bit 31 = Parity
    pub fn process_ctrl_packet_word(&mut self, word: u32, tlast: bool) {
        match std::mem::take(&mut self.ctrl_pkt_state) {
            ControlPacketState::WaitingForStreamHeader => {
                // Stream header forwarded because Drop_Header=false on master.
                // Consume it and transition to Idle for the actual ctrl header.
                let pkt_id = word & 0x1F;
                let pkt_type = (word >> 12) & 0x7;
                log::debug!("Tile ({},{}) ctrl_pkt: consuming stream header 0x{:08X} (pkt_id={}, pkt_type={})",
                    self.col, self.row, word, pkt_id, pkt_type);
                self.ctrl_pkt_state = ControlPacketState::Idle;
            }
            ControlPacketState::Idle => {
                // Parse control packet header (AM020 Table 3)
                use crate::device::aie2_spec::*;
                let address = word & CTRL_PKT_ADDRESS_MASK;
                let beats = ((word >> CTRL_PKT_LENGTH_SHIFT) & CTRL_PKT_LENGTH_MASK) as u8 + 1;
                let operation = ((word >> CTRL_PKT_OPERATION_SHIFT) & CTRL_PKT_OPERATION_MASK) as u8;
                let response_id = ((word >> CTRL_PKT_RESPONSE_ID_SHIFT) & CTRL_PKT_RESPONSE_ID_MASK) as u8;

                log::info!("Tile ({},{}) ctrl_pkt: header 0x{:08X} addr=0x{:05X} op={} beats={} resp_id={}",
                    self.col, self.row, word, address, operation, beats, response_id);

                self.ctrl_pkt_state = ControlPacketState::Collecting {
                    address,
                    operation,
                    response_id,
                    beats_total: beats,
                    data: Vec::with_capacity(beats as usize),
                };
            }
            ControlPacketState::Collecting {
                address,
                operation,
                response_id,
                beats_total,
                mut data,
            } => {
                data.push(word);
                log::debug!("Tile ({},{}) ctrl_pkt: data[{}] = 0x{:08X} ({}/{}){}",
                    self.col, self.row, data.len() - 1, word, data.len(), beats_total,
                    if tlast { " TLAST" } else { "" });

                if data.len() >= beats_total as usize {
                    // All beats received -- execute the operation
                    self.execute_ctrl_packet(address, operation, response_id, &data);
                    // After completion: if TLAST marks end of stream packet AND
                    // headers aren't dropped, expect a stream header next time.
                    // Otherwise stay Idle for the next ctrl packet within this
                    // same stream packet.
                    self.ctrl_pkt_state = if tlast && !self.ctrl_pkt_drop_header {
                        ControlPacketState::WaitingForStreamHeader
                    } else {
                        ControlPacketState::Idle
                    };
                } else {
                    // Still collecting
                    self.ctrl_pkt_state = ControlPacketState::Collecting {
                        address,
                        operation,
                        response_id,
                        beats_total,
                        data,
                    };
                }
            }
        }
    }

    /// Execute a complete control packet operation.
    ///
    /// Currently supports:
    /// - Operation 0 (write): Write data words to consecutive register addresses
    /// - Operation 1 (read): Not yet implemented (would need response routing)
    fn execute_ctrl_packet(&mut self, base_address: u32, operation: u8, _response_id: u8, data: &[u32]) {
        use crate::device::aie2_spec::*;
        match operation {
            CTRL_PKT_OP_WRITE | CTRL_PKT_OP_BLOCK_WRITE => {
                // Write / Block write: write data to consecutive 32-bit registers
                for (i, &value) in data.iter().enumerate() {
                    let addr = base_address + (i as u32) * 4;
                    log::info!("Tile ({},{}) ctrl_pkt WRITE: [0x{:05X}] = 0x{:08X}",
                        self.col, self.row, addr, value);
                    self.write_register(addr, value);
                }
            }
            CTRL_PKT_OP_READ => {
                // Read: would need to send response back via stream
                log::warn!("Tile ({},{}) ctrl_pkt: read operation not yet supported (addr=0x{:05X})",
                    self.col, self.row, base_address);
            }
            CTRL_PKT_OP_WRITE_INCR => {
                // Write with increment: write data to address, then address+4, etc.
                for (i, &value) in data.iter().enumerate() {
                    let addr = base_address + (i as u32) * 4;
                    log::info!("Tile ({},{}) ctrl_pkt WRITE_INCR: [0x{:05X}] = 0x{:08X}",
                        self.col, self.row, addr, value);
                    self.write_register(addr, value);
                }
            }
            _ => {
                log::warn!("Tile ({},{}) ctrl_pkt: unknown operation {} (addr=0x{:05X})",
                    self.col, self.row, operation, base_address);
            }
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

        // Bounds check against actual lock count for this tile
        if lock_id >= self.locks.len() {
            return 0; // Invalid lock ID
        }

        // Perform the operation
        let result = if is_acquire {
            // Acquire: check if value >= abs(change), then apply delta
            // For acquire, change_value is typically negative (consuming tokens)
            self.locks[lock_id].acquire_with_value((-change_value).max(0), change_value)
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
        assert_eq!(tile.data_memory().len(), 64 * 1024);
        assert_eq!(tile.locks.len(), 16);
        assert_eq!(tile.dma_bds.len(), 16);
        assert_eq!(tile.dma_channels.len(), 4);
    }

    #[test]
    fn test_mem_tile_creation() {
        let tile = Tile::mem_tile(0, 1);
        assert!(tile.is_mem_tile());
        assert!(tile.program_memory().is_none());
        assert_eq!(tile.data_memory().len(), 512 * 1024);
        assert_eq!(tile.locks.len(), 64);
        assert_eq!(tile.dma_bds.len(), 48);
        assert_eq!(tile.dma_channels.len(), 12);
    }

    #[test]
    fn test_shim_tile_creation() {
        let tile = Tile::shim(0, 0);
        assert!(tile.is_shim());
        assert!(tile.program_memory().is_none());
        assert_eq!(tile.data_memory().len(), 0);
        assert_eq!(tile.locks.len(), 0);
        assert_eq!(tile.dma_bds.len(), 0);
        assert_eq!(tile.dma_channels.len(), 0);
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
        // Test clamping at creation (positive overflow)
        let lock = Lock::new(100);
        assert_eq!(lock.value, Lock::MAX_VALUE); // Clamped to 63

        // Test clamping at creation (negative overflow)
        let lock = Lock::new(-100);
        assert_eq!(lock.value, Lock::MIN_VALUE); // Clamped to -64

        // Test saturation on release
        let mut lock = Lock::new(63);
        lock.release();
        assert_eq!(lock.value, 63); // Saturated at max

        // Test set
        let mut lock = Lock::new(0);
        lock.set(50);
        assert_eq!(lock.value, 50);
        lock.set(Lock::MAX_VALUE + 1); // would be 64, but i8 can't hold it; test boundary
        // i8 max is 127, so test with explicit value
        lock.set(100); // > 63
        assert_eq!(lock.value, 63); // Clamped
        lock.set(-100); // < -64
        assert_eq!(lock.value, -64); // Clamped
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
        // Verify lock counts per AM020 (via TileParams defaults)
        assert_eq!(TileParams::compute().num_locks, 16);
        assert_eq!(TileParams::mem_tile().num_locks, 64);
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

        // Large negative delta: goes into negative range (7 - 10 = -3, valid)
        assert_eq!(lock.release_with_value(-10), LockResult::Success);
        assert_eq!(lock.value, -3);

        // Push to underflow past MIN_VALUE (-3 - 62 = -65, beyond -64)
        assert_eq!(lock.release_with_value(-62), LockResult::WouldUnderflow);
        assert_eq!(lock.value, Lock::MIN_VALUE); // Clamped to -64
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

    // === Edge Detection Tests ===

    #[test]
    fn test_edge_detector_default() {
        let det = EdgeDetector::default();
        assert_eq!(det.input_event, 0);
        assert!(!det.trigger_rising);
        assert!(!det.trigger_falling);
        assert!(!det.prev_active);
        assert!(!det.curr_active);
    }

    #[test]
    fn test_configure_edge_detectors_compute() {
        let mut dets = [EdgeDetector::default(); 2];
        // Event 0: event=37 (0x25), rising=1, falling=0
        // Event 1: event=29 (0x1D), rising=0, falling=1
        // value = (1<<26) | (29<<16) | (1<<9) | 37
        let value = (1 << 26) | (29 << 16) | (1 << 9) | 37;
        Tile::configure_edge_detectors(&mut dets, value, false);

        assert_eq!(dets[0].input_event, 37);
        assert!(dets[0].trigger_rising);
        assert!(!dets[0].trigger_falling);

        assert_eq!(dets[1].input_event, 29);
        assert!(!dets[1].trigger_rising);
        assert!(dets[1].trigger_falling);
    }

    #[test]
    fn test_configure_edge_detectors_memtile_8bit() {
        let mut dets = [EdgeDetector::default(); 2];
        // MemTile uses 8-bit event fields: [7:0] and [23:16]
        // Event 0: event=200 (> 127, needs 8 bits), rising+falling
        // Event 1: event=150, rising only
        let value = (1 << 25) | (150 << 16) | (1 << 10) | (1 << 9) | 200;
        Tile::configure_edge_detectors(&mut dets, value, true);

        assert_eq!(dets[0].input_event, 200);
        assert!(dets[0].trigger_rising);
        assert!(dets[0].trigger_falling);

        assert_eq!(dets[1].input_event, 150);
        assert!(dets[1].trigger_rising);
        assert!(!dets[1].trigger_falling);
    }

    #[test]
    fn test_edge_detector_rising_edge() {
        let mut tile = Tile::compute(0, 2);
        // Configure core edge detector 0: monitor event 37, rising edge
        tile.core_edge_detectors[0].input_event = 37;
        tile.core_edge_detectors[0].trigger_rising = true;

        // Configure core trace to accept edge events (need start event)
        tile.core_trace.write_register(0x00, 0x01); // mode=EventTime
        tile.core_trace.write_register(0x10, 37); // event slot 0 = event 37
        // Also configure slot for edge detection event (ID 13)
        tile.core_trace.write_register(0x10, 37 | (13 << 8)); // slot 0=37, slot 1=13

        // Cycle 1: event 37 fires (0->1 = rising edge)
        tile.notify_core_trace_event(37, 100);
        tile.evaluate_edge_detectors(100);
        // The edge detector should have detected rising edge and fired event 13

        // Cycle 2: event 37 does not fire (1->0 = falling, not configured)
        tile.evaluate_edge_detectors(200);
        // No event should fire (falling not configured)

        // Cycle 3: event 37 fires again (0->1 = rising edge again)
        tile.notify_core_trace_event(37, 300);
        tile.evaluate_edge_detectors(300);
        // Rising edge detected again
    }

    #[test]
    fn test_edge_detector_falling_edge() {
        let mut tile = Tile::compute(0, 2);
        // Configure mem edge detector 1: monitor event 77, falling edge
        tile.mem_edge_detectors[1].input_event = 77;
        tile.mem_edge_detectors[1].trigger_falling = true;

        // Cycle 1: event fires (0->1), no trigger (falling only)
        tile.notify_mem_trace_event(77, 100);
        tile.evaluate_edge_detectors(100);

        // Cycle 2: event does NOT fire (1->0 = falling edge)
        tile.evaluate_edge_detectors(200);
        // Falling edge should fire EDGE_DETECTION_EVENT_1 (mem ID 12)
    }

    #[test]
    fn test_edge_detector_register_write() {
        let mut tile = Tile::compute(0, 2);
        // Write core module edge detection register (0x34408)
        // Event 0: event=42, rising=1; Event 1: event=50, falling=1
        let value = (1u32 << 26) | (50 << 16) | (1 << 9) | 42;
        tile.write_register(0x34408, value);

        assert_eq!(tile.core_edge_detectors[0].input_event, 42);
        assert!(tile.core_edge_detectors[0].trigger_rising);
        assert!(!tile.core_edge_detectors[0].trigger_falling);

        assert_eq!(tile.core_edge_detectors[1].input_event, 50);
        assert!(!tile.core_edge_detectors[1].trigger_rising);
        assert!(tile.core_edge_detectors[1].trigger_falling);
    }

    #[test]
    fn test_edge_detector_mem_module_register() {
        let mut tile = Tile::compute(0, 2);
        // Write memory module edge detection register (0x14408)
        let value = (1u32 << 25) | (30 << 16) | (1 << 10) | (1 << 9) | 20;
        tile.write_register(0x14408, value);

        assert_eq!(tile.mem_edge_detectors[0].input_event, 20);
        assert!(tile.mem_edge_detectors[0].trigger_rising);
        assert!(tile.mem_edge_detectors[0].trigger_falling);

        assert_eq!(tile.mem_edge_detectors[1].input_event, 30);
        assert!(tile.mem_edge_detectors[1].trigger_rising);
        assert!(!tile.mem_edge_detectors[1].trigger_falling);
    }

    #[test]
    fn test_edge_detector_memtile_register() {
        let mut tile = Tile::mem_tile(0, 1);
        // Write MemTile edge detection register (0x94408)
        // Use event > 127 to verify 8-bit field
        let value = (1u32 << 25) | (200 << 16) | (1 << 9) | 180;
        tile.write_register(0x94408, value);

        assert_eq!(tile.mem_edge_detectors[0].input_event, 180);
        assert!(tile.mem_edge_detectors[0].trigger_rising);

        assert_eq!(tile.mem_edge_detectors[1].input_event, 200);
        assert!(tile.mem_edge_detectors[1].trigger_rising);
    }

    #[test]
    fn test_edge_detector_no_trigger_when_unconfigured() {
        let mut tile = Tile::compute(0, 2);
        // Default: no edge detection configured (input_event=0, no triggers)
        // Notify event 37
        tile.notify_core_trace_event(37, 100);
        tile.evaluate_edge_detectors(100);
        // No edge events should fire (detectors not configured)
        // Just verify it doesn't panic
    }
}
