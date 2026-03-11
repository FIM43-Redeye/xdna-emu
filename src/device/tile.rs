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
//! See `crate::arch` module for the authoritative constants.
//!
//! # Performance
//!
//! This module is designed for fast emulation:
//! - Fixed-size arrays (no heap allocation during emulation)
//! - Direct field access (no hash maps)
//! - Cache-friendly layout (related data together)

use super::stream_switch::StreamSwitch as FunctionalStreamSwitch;
use super::trace_unit::TraceUnit;
use crate::interpreter::state::EventType;

/// Size of program memory (16 KB = 1024 x 128-bit instructions).
pub const PROGRAM_MEMORY_SIZE: usize = crate::arch::compute::PROGRAM_MEMORY_SIZE as usize;

/// Parameters for constructing a Tile with correct per-tile-type sizing.
///
/// Production code derives these from `ArchConfig` (which reads mlir-aie
/// device models). Convenience constructors also load from the device model
/// so there is a single source of truth.
#[derive(Debug, Clone)]
pub struct TileParams {
    /// Data memory size in bytes (0 for shim, 64K for compute, 512K for mem tile).
    pub data_memory_size: usize,
    /// Number of locks (16 for shim/compute, 64 for mem tile).
    pub num_locks: usize,
    /// Number of DMA buffer descriptors (16 for compute, 48 for mem tile).
    pub num_bds: usize,
    /// Total DMA channels (4 for compute, 12 for mem tile).
    pub num_channels: usize,
    /// S2MM (write) DMA channels (2 for compute/shim, 6 for mem tile).
    pub dma_s2mm_channels: usize,
    /// MM2S (read) DMA channels (2 for compute/shim, 6 for mem tile).
    pub dma_mm2s_channels: usize,
}

impl TileParams {
    /// NPU1/AIE2 compute tile params, from compile-time arch constants.
    pub fn compute() -> Self {
        use crate::arch;
        let ch = arch::compute::NUM_DMA_CHANNELS as usize;
        Self {
            data_memory_size: arch::compute::MEMORY_SIZE as usize,
            num_locks: arch::compute::NUM_LOCKS as usize,
            num_bds: arch::compute::NUM_BDS as usize,
            num_channels: ch * 2,
            dma_s2mm_channels: ch,
            dma_mm2s_channels: ch,
        }
    }

    /// NPU1/AIE2 memory tile params, from compile-time arch constants.
    pub fn mem_tile() -> Self {
        use crate::arch;
        let ch = arch::memtile::NUM_DMA_CHANNELS as usize;
        Self {
            data_memory_size: arch::memtile::MEMORY_SIZE as usize,
            num_locks: arch::memtile::NUM_LOCKS as usize,
            num_bds: arch::memtile::NUM_BDS as usize,
            num_channels: ch * 2,
            dma_s2mm_channels: ch,
            dma_mm2s_channels: ch,
        }
    }

    /// NPU1/AIE2 shim tile params, from compile-time arch constants.
    pub fn shim() -> Self {
        use crate::arch;
        let ch = arch::shim::NUM_DMA_CHANNELS as usize;
        Self {
            data_memory_size: 0,
            num_locks: arch::shim::NUM_LOCKS as usize,
            num_bds: arch::shim::NUM_BDS as usize,
            num_channels: ch * 2,
            dma_s2mm_channels: ch,
            dma_mm2s_channels: ch,
        }
    }
}

/// Result of a lock operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockResult {
    /// Operation succeeded
    Success,
    /// Operation failed - would underflow (value would go below -64)
    PreconditionNotMet,
    /// Operation failed - would overflow (value would exceed +63)
    WouldOverflow,
}

// ---------------------------------------------------------------------------
// Lock Arbiter -- round-robin arbitration per AM020
// ---------------------------------------------------------------------------

/// Identifies the source of a lock request.
///
/// The hardware lock arbiter processes requests from the core and each DMA
/// channel independently. Priority rotates among all requestors using
/// round-robin to ensure fairness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockRequestor {
    /// Core processor (lock acquire/release instructions)
    Core,
    /// DMA S2MM channel (stream-to-memory, index 0..n)
    DmaS2mm(u8),
    /// DMA MM2S channel (memory-to-stream, index 0..n)
    DmaMm2s(u8),
}

impl LockRequestor {
    /// Map requestor to a priority index for round-robin ordering.
    ///
    /// Layout: [Core, S2MM_0, S2MM_1, ..., MM2S_0, MM2S_1, ...]
    pub fn to_priority_index(&self, s2mm_count: u8) -> usize {
        match self {
            LockRequestor::Core => 0,
            LockRequestor::DmaS2mm(ch) => 1 + *ch as usize,
            LockRequestor::DmaMm2s(ch) => 1 + s2mm_count as usize + *ch as usize,
        }
    }
}

/// A pending lock request submitted to the arbiter.
#[derive(Debug, Clone)]
pub struct LockRequest {
    /// Who is requesting
    pub requestor: LockRequestor,
    /// Which lock
    pub lock_id: usize,
    /// True = acquire (blocking), false = release (non-blocking)
    pub is_acquire: bool,
    /// For acquire: value threshold (acq_ge: lock >= expected; acq_eq: lock == expected)
    pub expected: i8,
    /// Change to apply to the lock value
    pub delta: i8,
    /// For acquire: true = exact match (acq_eq), false = greater-or-equal (acq_ge)
    pub equal_mode: bool,
}

/// Per-lock contention statistics for debugging.
#[derive(Debug, Clone, Default)]
pub struct LockArbiterStats {
    /// Total grants (successful arbitrations)
    pub grants: u64,
    /// Contentions (multiple requestors wanted the same lock in the same cycle)
    pub contentions: u64,
    /// Stalls (request denied due to contention, not precondition failure)
    pub stalls: u64,
}

/// Round-robin lock arbiter for a tile's memory module.
///
/// Per AM020, the lock arbiter sits in each tile's memory module and handles
/// competing lock requests from multiple sources (core, DMA S2MM channels,
/// DMA MM2S channels). It uses round-robin arbitration and processes one
/// request per lock per clock cycle.
///
/// # Design
///
/// Requests are submitted during the cycle (by core execution and DMA steps).
/// At the end of the cycle, `resolve()` is called to arbitrate and apply
/// granted requests. Denied requests (due to contention) must be resubmitted
/// next cycle.
#[derive(Debug)]
pub struct LockArbiter {
    /// Pending requests for this cycle, grouped by lock_id
    pending: Vec<LockRequest>,

    /// Round-robin priority pointer (index into the requestor ordering).
    /// Rotates after each grant to ensure fairness.
    priority: usize,

    /// Number of S2MM channels (for priority index calculation)
    s2mm_count: u8,

    /// Total number of requestors (Core + S2MM + MM2S)
    num_requestors: usize,

    /// Per-lock statistics (indexed by lock_id)
    stats: Vec<LockArbiterStats>,

    /// Results of the last arbitration round.
    /// (requestor, lock_id, granted, is_acquire)
    results: Vec<(LockRequestor, usize, bool, bool)>,
}

impl LockArbiter {
    /// Create a new arbiter for a tile.
    pub fn new(num_locks: usize, s2mm_channels: u8, mm2s_channels: u8) -> Self {
        Self {
            pending: Vec::with_capacity(8),
            priority: 0,
            s2mm_count: s2mm_channels,
            num_requestors: 1 + s2mm_channels as usize + mm2s_channels as usize,
            stats: vec![LockArbiterStats::default(); num_locks],
            results: Vec::with_capacity(8),
        }
    }

    /// Submit a lock request to the arbiter.
    pub fn submit(&mut self, request: LockRequest) {
        self.pending.push(request);
    }

    /// Resolve all pending requests using round-robin arbitration.
    ///
    /// For each lock that has pending requests:
    /// - If only one requestor: grant if precondition met (fast path)
    /// - If multiple requestors: grant to the one closest to the current
    ///   priority pointer (round-robin), deny others
    ///
    /// Granted acquire requests are applied to the lock values. Releases
    /// always succeed when granted. The priority pointer rotates after
    /// each contended grant.
    pub fn resolve(&mut self, locks: &mut [Lock]) -> &[(LockRequestor, usize, bool, bool)] {
        self.results.clear();

        if self.pending.is_empty() {
            return &self.results;
        }

        // Group pending requests by lock_id using simple O(n^2) grouping.
        let mut processed = vec![false; self.pending.len()];

        for i in 0..self.pending.len() {
            if processed[i] {
                continue;
            }

            let lock_id = self.pending[i].lock_id;

            // Collect all requests for this lock_id
            let mut group: Vec<usize> = vec![i];
            for j in (i + 1)..self.pending.len() {
                if !processed[j] && self.pending[j].lock_id == lock_id {
                    group.push(j);
                    processed[j] = true;
                }
            }
            processed[i] = true;

            if group.len() == 1 {
                // Fast path: single requestor, no contention
                let req = &self.pending[group[0]];
                let granted = Self::try_apply(req, locks);
                if lock_id < self.stats.len() {
                    self.stats[lock_id].grants += granted as u64;
                }
                self.results.push((req.requestor, lock_id, granted, req.is_acquire));
            } else {
                // Contention: multiple requestors want the same lock
                if lock_id < self.stats.len() {
                    self.stats[lock_id].contentions += 1;
                }

                // Sort by round-robin distance from priority pointer
                let s2mm_count = self.s2mm_count;
                let priority = self.priority;
                let num_requestors = self.num_requestors;
                let pending = &self.pending;

                let mut group_with_dist: Vec<(usize, usize)> = group
                    .iter()
                    .map(|&idx| {
                        let pi = pending[idx].requestor.to_priority_index(s2mm_count);
                        let dist = (pi + num_requestors - priority) % num_requestors;
                        (idx, dist)
                    })
                    .collect();
                group_with_dist.sort_by_key(|&(_, dist)| dist);

                // Process releases first: they are non-blocking and always
                // succeed on real hardware. A release must not prevent a
                // same-cycle acquire from seeing the updated value.
                for &(idx, _) in &group_with_dist {
                    let req = &self.pending[idx];
                    if !req.is_acquire {
                        Self::try_apply(req, locks);
                        if lock_id < self.stats.len() {
                            self.stats[lock_id].grants += 1;
                        }
                        self.results.push((req.requestor, lock_id, true, false));
                    }
                }

                // Then process acquires with round-robin arbitration.
                // Only ONE acquire granted per lock per cycle.
                let mut any_acquire_granted = false;
                for &(idx, _) in &group_with_dist {
                    let req = &self.pending[idx];
                    if !req.is_acquire {
                        continue; // already handled above
                    }
                    if !any_acquire_granted {
                        let granted = Self::try_apply(req, locks);
                        if granted {
                            any_acquire_granted = true;
                            if lock_id < self.stats.len() {
                                self.stats[lock_id].grants += 1;
                            }
                            let winner_pi = req.requestor.to_priority_index(s2mm_count);
                            self.priority = (winner_pi + 1) % num_requestors;
                        }
                        self.results.push((req.requestor, lock_id, granted, true));
                    } else {
                        if lock_id < self.stats.len() {
                            self.stats[lock_id].stalls += 1;
                        }
                        self.results.push((req.requestor, lock_id, false, true));
                    }
                }
            }
        }

        self.pending.clear();
        &self.results
    }

    /// Try to apply a single lock request. Returns true if granted.
    fn try_apply(req: &LockRequest, locks: &mut [Lock]) -> bool {
        if req.lock_id >= locks.len() {
            return false;
        }
        let lock = &mut locks[req.lock_id];

        if req.is_acquire {
            // Check precondition based on mode
            let precondition_met = if req.equal_mode {
                lock.value == req.expected
            } else {
                lock.value >= req.expected
            };

            if !precondition_met {
                return false;
            }

            // Apply delta
            let new_value = (lock.value as i16) + (req.delta as i16);
            if new_value < Lock::MIN_VALUE as i16 {
                lock.underflow = true;
                return false;
            }
            if new_value > Lock::MAX_VALUE as i16 {
                lock.overflow = true;
                lock.value = Lock::MAX_VALUE;
            } else {
                lock.value = new_value as i8;
            }
            true
        } else {
            // Release: always succeeds (non-blocking), apply delta
            lock.release_with_value(req.delta);
            true
        }
    }

    /// Check if a specific requestor was granted in the last resolve.
    pub fn was_granted(&self, requestor: LockRequestor, lock_id: usize) -> bool {
        self.results
            .iter()
            .any(|&(ref r, lid, granted, _)| *r == requestor && lid == lock_id && granted)
    }

    /// Get per-lock statistics.
    pub fn lock_stats(&self, lock_id: usize) -> Option<&LockArbiterStats> {
        self.stats.get(lock_id)
    }

    /// Returns true if there are pending requests.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Reset the arbiter state (priority, stats, pending).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.results.clear();
        self.priority = 0;
        for s in &mut self.stats {
            *s = LockArbiterStats::default();
        }
    }
}

/// Lock state.
///
/// AIE2 uses semaphore locks with acquire/release semantics.
/// Lock value range is -64 to +63 (per aie-rt LockValLowerBound/UpperBound).
/// The Lock_Value register field is bits [5:0] (6-bit, mask 0x3F per
/// xaiemlgbl_params.h). Values outside the 6-bit range (-64 to -33) are
/// valid in the logical model but alias when read back from the register.
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
    /// Register field is 6-bit [5:0] (mask 0x3F), but aie-rt defines the
    /// logical range as LockValLowerBound=-64 to LockValUpperBound=63.
    pub value: i8,
    /// Overflow flag - set when release would exceed MAX_VALUE
    pub overflow: bool,
    /// Underflow flag - set when acquire would go below MIN_VALUE
    pub underflow: bool,
}

impl Lock {
    /// Maximum lock value (+63, per aie-rt LockValUpperBound).
    ///
    /// This compile-time constant is validated against the mlir-aie device
    /// model in tests (see `device::model::validate_against_spec()`). It is kept
    /// as a const for hot-path efficiency in lock acquire/release.
    pub const MAX_VALUE: i8 = 63;

    /// Minimum lock value (-64, per aie-rt LockValLowerBound).
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
            return LockResult::PreconditionNotMet;
        }

        // Apply delta (convert to i16 for safe arithmetic)
        let new_value = (self.value as i16) + (delta as i16);

        if new_value < Self::MIN_VALUE as i16 {
            self.underflow = true;
            return LockResult::PreconditionNotMet;
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
    /// Returns `LockResult::PreconditionNotMet` if the value doesn't match exactly.
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
            return LockResult::PreconditionNotMet;
        }

        // Apply delta (convert to i16 for safe arithmetic)
        let new_value = (self.value as i16) + (delta as i16);

        if new_value < Self::MIN_VALUE as i16 {
            self.underflow = true;
            return LockResult::PreconditionNotMet;
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
            return LockResult::PreconditionNotMet;
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
        /// Number of beats collected so far
        beats_collected: u8,
        /// Accumulated data words (max 4 beats per control packet)
        data: [u32; 4],
    },
}

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

    /// Round-robin lock arbiter for this tile's memory module.
    ///
    /// Per AM020, the lock arbiter serializes competing lock requests from
    /// the core and DMA channels. Requests are submitted during the cycle
    /// and resolved at end-of-cycle. Core releases submitted in Phase 2
    /// are resolved alongside DMA requests in Phase 3, providing the
    /// 1-cycle visibility delay that matches hardware.
    pub lock_arbiter: LockArbiter,

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
    pub(crate) registers: std::collections::HashMap<u32, u32>,

    /// Control packet state machine for TileCtrl port.
    pub ctrl_pkt_state: ControlPacketState,

    /// Whether the TileCtrl master port drops stream headers.
    /// Set from the master port's packet config during CDO processing.
    /// When false, the handler must consume and discard the stream header
    /// before parsing the control packet header.
    pub ctrl_pkt_drop_header: bool,

    /// Shim Mux: which switchbox South slave port each DMA MM2S channel feeds.
    /// Parsed from Mux_Config register (0x1F000). Index 0 = MM2S ch0, etc.
    /// Value is the switchbox slave port index (e.g., 5 for South3).
    pub shim_mux_mm2s_slaves: Vec<Option<usize>>,

    /// Shim Mux: which switchbox South master port feeds each DMA S2MM channel.
    /// Parsed from Demux_Config register (0x1F004). Index 0 = S2MM ch0, etc.
    /// Value is the switchbox master port index (e.g., 2 for South0).
    pub shim_mux_s2mm_masters: Vec<Option<usize>>,

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

    /// Pending memory-module trace events from all sources (DMA, locks, etc.).
    ///
    /// On real hardware, the mem trace unit monitors event wires from the
    /// entire memory module -- it doesn't distinguish DMA events from lock
    /// events. This buffer unifies all memory-module event sources. The
    /// coordinator drains it each cycle and routes events to the mem trace
    /// unit via `notify_mem_trace_event()`.
    pub mem_trace_pending: Vec<(u64, crate::interpreter::state::EventType)>,

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

    // === Cascade Stream (compute tiles only) ===

    /// Cascade input FIFO (SCD). 384-bit width, depth 1.
    /// Dedicated point-to-point link between adjacent compute tiles,
    /// entirely separate from the stream switch fabric.
    /// Source: aie-rt/driver/src/core/xaie_core.c:993-1046
    pub cascade_input: std::collections::VecDeque<[u64; 6]>,

    /// Cascade output FIFO (MCD). 384-bit width, depth 1.
    pub cascade_output: std::collections::VecDeque<[u64; 6]>,

    /// Cascade input direction: 0=North, 1=West.
    /// From accumulator control register at offset 0x36060 bit 0.
    pub cascade_input_dir: u8,

    /// Cascade output direction: 0=South, 1=East.
    /// From accumulator control register at offset 0x36060 bit 1.
    pub cascade_output_dir: u8,

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

    // === Event Broadcast ===

    /// Event broadcast channel mapping (16 channels).
    ///
    /// Each entry stores the local event ID that triggers that broadcast channel.
    /// When Event_Generate fires an event matching channel N's configured ID,
    /// BROADCAST_N (hw_id 107+N) is generated and propagated to the column.
    ///
    /// Configured by writes to Event_Broadcast registers:
    /// - Compute core module:  0x34010 + N*4 (N=0..15)
    /// - Compute memory module: 0x14010 + N*4
    /// - MemTile:              0x94010 + N*4
    /// - Shim (PL module):     0x34010 + N*4
    pub broadcast_channels: [u8; 16],

    /// Pending broadcast events to propagate to all tiles in this column.
    ///
    /// When Event_Generate fires an event that matches a broadcast channel,
    /// the BROADCAST_N hw_id (107+N) is pushed here. The caller (state.rs
    /// or NPU executor) drains this after each write and propagates to the
    /// column.
    pub pending_broadcasts: Vec<u8>,

    /// Pending control packet read response words.
    ///
    /// When an OP_READ control packet is processed, the response (stream
    /// header + data words) is queued here. Each routing cycle, words are
    /// drained into the TileCtrl slave port as FIFO space permits, matching
    /// the backpressure-aware pattern used by trace unit injection.
    ///
    /// Each entry is (word, tlast). The header is first, followed by data
    /// words, with TLAST on the final data word.
    pub pending_ctrl_response: std::collections::VecDeque<(u32, bool)>,
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
            lock_arbiter: LockArbiter::new(
                params.num_locks,
                params.dma_s2mm_channels as u8,
                params.dma_mm2s_channels as u8,
            ),
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
            shim_mux_mm2s_slaves: vec![None; params.dma_mm2s_channels],
            shim_mux_s2mm_masters: vec![None; params.dma_s2mm_channels],
            cascade_input: std::collections::VecDeque::new(),
            cascade_output: std::collections::VecDeque::new(),
            cascade_input_dir: 0,
            cascade_output_dir: 0,
            core_trace: TraceUnit::new(col, row),
            mem_trace: TraceUnit::new(col, row),
            mem_trace_pending: Vec::new(),
            event_port_selection: [None; 8],
            cycle_dma_banks: 0,
            core_edge_detectors: [EdgeDetector::default(); 2],
            mem_edge_detectors: [EdgeDetector::default(); 2],
            broadcast_channels: [0; 16],
            pending_broadcasts: Vec::new(),
            pending_ctrl_response: std::collections::VecDeque::new(),
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

    /// DMA BD base address and stride for this tile type (from register database).
    #[inline]
    fn bd_layout(&self, rl: &super::regdb::DeviceRegLayout) -> (u32, u32) {
        match self.tile_type {
            TileType::MemTile => (rl.memtile_bd_base, rl.memtile_bd_stride),
            TileType::Shim => (rl.shim_bd_base, rl.shim_bd_stride),
            TileType::Compute => (rl.memory_bd_base, rl.memory_bd_stride),
        }
    }

    /// DMA channel control base address and stride for this tile type.
    ///
    /// For memtiles, returns the S2MM base (channels are S2MM then MM2S,
    /// contiguous with the same stride).
    #[inline]
    fn channel_layout(&self, rl: &super::regdb::DeviceRegLayout) -> (u32, u32) {
        match self.tile_type {
            TileType::MemTile => (rl.memtile_channel_s2mm_base, rl.memtile_channel_stride),
            TileType::Shim => (rl.shim_channel_base, rl.shim_channel_stride),
            TileType::Compute => (rl.memory_channel_base, rl.memory_channel_stride),
        }
    }

    // === Event Broadcast ===

    /// Drain pending broadcast events generated by Event_Generate.
    ///
    /// Returns the hw_ids of broadcast events (e.g., 107 for BROADCAST_0)
    /// that should be propagated to all tiles in this column.
    pub fn drain_pending_broadcasts(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pending_broadcasts)
    }

    // === Memory Bank Conflict Detection ===

    /// Number of physical memory banks for this tile type (for conflict detection).
    pub fn num_banks(&self) -> usize {
        match self.tile_type {
            TileType::Compute => crate::arch::compute::PHYSICAL_BANKS as usize,
            TileType::MemTile => crate::arch::memtile::PHYSICAL_BANKS as usize,
            TileType::Shim => 0,
        }
    }

    /// Record that DMA accessed the given memory address range this cycle.
    /// Call from DMA transfer methods during Phase 2.
    #[inline]
    pub fn record_dma_bank_access(&mut self, addr: u32, bytes: usize) {
        let nb = self.num_banks();
        if nb > 0 {
            self.cycle_dma_banks |= crate::device::banking::banks_for_access(addr, bytes, nb);
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
        // Core module / PL module edge detectors -> core_trace
        for i in 0..2 {
            let det = &self.core_edge_detectors[i];
            let fire = (det.trigger_rising && det.curr_active && !det.prev_active)
                || (det.trigger_falling && !det.curr_active && det.prev_active);
            if fire {
                // Shim PL module: IDs 11-12; Core module: IDs 13-14
                let hw_id = if self.is_shim() {
                    crate::trace::shim_edge_detection_event_hw_id(i as u8)
                } else {
                    crate::trace::core_edge_detection_event_hw_id(i as u8)
                };
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
    pub(crate) fn configure_edge_detectors(detectors: &mut [EdgeDetector; 2], value: u32, is_memtile: bool) {
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

    // === Lock Arbiter Interface ===
    //
    // All lock operations go through the round-robin arbiter. Requests are
    // submitted during the cycle (core in Phase 2, DMA in Phase 3) and
    // resolved at the end of Phase 3. The arbiter serializes competing
    // requests: one grant per lock per cycle, with round-robin fairness.
    //
    // Core lock releases submitted in Phase 2 are resolved alongside DMA
    // requests in Phase 3, providing the 1-cycle visibility delay that
    // matches hardware (core release at cycle N visible to DMA at cycle N+1).

    /// Submit a lock request to the arbiter.
    ///
    /// Called by core (lock release instructions) and DMA (acquire/release).
    /// The request is queued until `resolve_lock_requests()` is called.
    #[inline]
    pub fn submit_lock_request(&mut self, request: LockRequest) {
        log::debug!("Tile({},{}) submit_lock_request: {:?} lock={} acquire={} expected={} delta={}",
            self.col, self.row, request.requestor, request.lock_id,
            request.is_acquire, request.expected, request.delta);
        self.lock_arbiter.submit(request);
    }

    /// Defer a core lock release through the arbiter.
    ///
    /// Core releases are deferred by 1 cycle: submitted during Phase 2
    /// (core stepping), resolved at end of Phase 3 (data movement).
    /// This matches hardware's lock arbiter pipeline latency.
    #[inline]
    pub fn defer_core_lock_release(&mut self, lock_id: usize, delta: i8) {
        if lock_id < self.locks.len() {
            log::debug!("Tile({},{}) defer_core_lock_release lock {} delta {}",
                self.col, self.row, lock_id, delta);
            self.lock_arbiter.submit(LockRequest {
                requestor: LockRequestor::Core,
                lock_id,
                is_acquire: false,
                expected: 0,
                delta,
                equal_mode: false,
            });
        }
    }

    /// Resolve all pending lock requests using round-robin arbitration.
    ///
    /// Call at end of Phase 3, after all requestors have submitted.
    /// Applies granted requests directly to lock values. Returns results
    /// for callers that need to check grant status (e.g., DMA engine).
    ///
    /// Granted lock operations emit trace events into `mem_trace_pending`,
    /// matching real hardware where the memory module trace unit monitors
    /// all lock state changes regardless of source.
    pub fn resolve_lock_requests(&mut self, cycle: u64) -> Vec<(LockRequestor, usize, bool, bool)> {
        let results = self.lock_arbiter.resolve(&mut self.locks);
        // Emit trace events for granted lock operations.
        for &(_, lock_id, granted, is_acquire) in results {
            if granted {
                let event = if is_acquire {
                    EventType::LockAcquire { lock_id: lock_id as u8 }
                } else {
                    EventType::LockRelease { lock_id: lock_id as u8 }
                };
                self.mem_trace_pending.push((cycle, event));
            }
        }
        results.to_vec()
    }

    /// Check if a specific requestor was granted a lock in the last resolve.
    #[inline]
    pub fn lock_was_granted(&self, requestor: LockRequestor, lock_id: usize) -> bool {
        self.lock_arbiter.was_granted(requestor, lock_id)
    }

    /// Get the current committed lock value.
    ///
    /// Returns the live lock value. Pending arbiter requests that have
    /// not yet been resolved are NOT reflected.
    #[inline]
    pub fn effective_lock_value(&self, lock_id: usize) -> i8 {
        if lock_id < self.locks.len() {
            self.locks[lock_id].value
        } else {
            0
        }
    }

    /// Get the current lock value (same as effective_lock_value).
    ///
    /// Compatibility shim -- previously returned the snapshot value.
    /// Now returns the live committed value.
    #[inline]
    pub fn lock_snapshot_value(&self, lock_id: usize) -> i8 {
        self.effective_lock_value(lock_id)
    }

    // === Register Access (for NPU instruction execution) ===

    /// Write a 32-bit value to a register offset.
    ///
    /// Get an immutable reference to the register map.
    ///
    /// Used by mask_write_register in state.rs to read current values without
    /// triggering side effects (unlike read_register which executes lock operations).
    pub fn registers_ref(&self) -> &std::collections::HashMap<u32, u32> {
        &self.registers
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

        // Check specific subsystem state first: DMA BD range.
        // Base and stride are per-tile-type from the register database.
        let (bd_base, bd_stride) = self.bd_layout(reg_layout);
        let bd_end = bd_base + (self.dma_bds.len() as u32) * bd_stride;
        if offset >= bd_base && offset < bd_end {
            let bd_offset = offset - bd_base;
            let bd_index = (bd_offset / bd_stride) as usize;
            let reg_in_bd = (bd_offset % bd_stride) as usize / 4;

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

    /// Read a register value without side effects.
    ///
    /// Unlike `read_register()`, this does NOT execute lock operations.
    /// Used for MMIO loads from the memory unit where mutable tile access
    /// is not available during instruction execution.
    pub fn read_register_pure(&self, offset: u32) -> u32 {
        let reg_layout = super::regdb::device_reg_layout();

        // DMA BD range (per-tile-type from register database)
        let (bd_base, bd_stride) = self.bd_layout(reg_layout);
        let bd_end = bd_base + (self.dma_bds.len() as u32) * bd_stride;
        if offset >= bd_base && offset < bd_end {
            let bd_offset = offset - bd_base;
            let bd_index = (bd_offset / bd_stride) as usize;
            let reg_in_bd = (bd_offset % bd_stride) as usize / 4;
            if bd_index < self.dma_bds.len() {
                let bd = &self.dma_bds[bd_index];
                return match reg_in_bd {
                    0 => bd.addr_low,
                    1 => bd.addr_high,
                    2 => bd.length,
                    3 => bd.control,
                    4 => bd.d0,
                    5 => bd.d1,
                    _ => self.registers.get(&offset).copied().unwrap_or(0),
                };
            }
        }

        // DMA channel control (per-tile-type from register database).
        // Compute tiles have a single channel base (S2MM and MM2S interleaved).
        // MemTiles have separate S2MM/MM2S bases. The stride is the same.
        let (ch_base, ch_stride) = self.channel_layout(reg_layout);
        let ch_end = ch_base + (self.dma_channels.len() as u32) * ch_stride;
        if offset >= ch_base && offset < ch_end {
            let ch_offset = offset - ch_base;
            let ch_index = (ch_offset / ch_stride) as usize;
            if ch_index < self.dma_channels.len() {
                return if ch_offset % ch_stride == 0 {
                    self.dma_channels[ch_index].control
                } else {
                    self.dma_channels[ch_index].start_queue
                };
            }
        }

        // Lock value registers (read-only, no acquire side effect)
        let lock_base = if self.is_mem_tile() {
            reg_layout.memtile_lock_base
        } else {
            reg_layout.memory_lock_base
        };
        let lock_stride = if self.is_mem_tile() {
            reg_layout.memtile_lock_stride
        } else {
            reg_layout.memory_lock_stride
        };
        let lock_end = lock_base + (self.locks.len() as u32) * lock_stride;
        if (lock_base..lock_end).contains(&offset) {
            let lock_id = ((offset - lock_base) / lock_stride) as usize;
            if lock_id < self.locks.len() {
                return self.locks[lock_id].value as u32 & reg_layout.lock_value_mask;
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

    // === Cascade Stream Helpers ===

    /// Push a 384-bit value into the cascade input FIFO (SCD).
    pub fn push_cascade_input(&mut self, data: [u64; 6]) {
        self.cascade_input.push_back(data);
    }

    /// Pop a 384-bit value from the cascade input FIFO (SCD).
    /// Returns None if the FIFO is empty (core should stall).
    pub fn pop_cascade_input(&mut self) -> Option<[u64; 6]> {
        self.cascade_input.pop_front()
    }

    /// Push a 384-bit value into the cascade output FIFO (MCD).
    pub fn push_cascade_output(&mut self, data: [u64; 6]) {
        self.cascade_output.push_back(data);
    }

    /// Pop a 384-bit value from the cascade output FIFO (MCD).
    pub fn pop_cascade_output(&mut self) -> Option<[u64; 6]> {
        self.cascade_output.pop_front()
    }

    /// Check if cascade input has data available.
    pub fn has_cascade_input(&self) -> bool {
        !self.cascade_input.is_empty()
    }

    /// Check if cascade output has data (for routing to neighbor).
    pub fn has_cascade_output(&self) -> bool {
        !self.cascade_output.is_empty()
    }

    // === Control Packet Handling ===

    /// Parse Shim Mux_Config register to find DMA MM2S South slave mapping.
    ///
    /// The Shim Mux selects which source (PL/DMA/NoC) feeds each switchbox South
    /// slave port. DMA MM2S output enters the switchbox through a South slave.
    ///
    /// Field layout and port mappings are derived from the AM025 register database.
    /// Select values: 0=South/PL, 1=DMA, 2=NoC
    pub(crate) fn parse_shim_mux_config(&mut self, value: u32) {
        let mux = &super::regdb::device_reg_layout().shim_mux;

        // Reset mapping (register may be rewritten with different config)
        self.shim_mux_mm2s_slaves.fill(None);

        let mut dma_ch = 0usize;
        for mf in &mux.mux_fields {
            let select = mf.field.extract(value);
            if select == 1 && dma_ch < self.shim_mux_mm2s_slaves.len() {
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
    pub(crate) fn parse_shim_demux_config(&mut self, value: u32) {
        let mux = &super::regdb::device_reg_layout().shim_mux;

        // Reset mapping
        self.shim_mux_s2mm_masters.fill(None);

        let mut dma_ch = 0usize;
        for df in &mux.demux_fields {
            let select = df.field.extract(value);
            if select == 1 && dma_ch < self.shim_mux_s2mm_masters.len() {
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
    pub fn process_ctrl_packet_word(&mut self, word: u32, tlast: bool) -> Vec<CtrlPacketAction> {
        match std::mem::take(&mut self.ctrl_pkt_state) {
            ControlPacketState::WaitingForStreamHeader => {
                // Stream header forwarded because Drop_Header=false on master.
                // Consume it and transition to Idle for the actual ctrl header.
                let pkt_id = word & 0x1F;
                let pkt_type = (word >> 12) & 0x7;
                log::debug!("Tile ({},{}) ctrl_pkt: consuming stream header 0x{:08X} (pkt_id={}, pkt_type={})",
                    self.col, self.row, word, pkt_id, pkt_type);
                self.ctrl_pkt_state = ControlPacketState::Idle;
                Vec::new()
            }
            ControlPacketState::Idle => {
                // Parse control packet header (AM020 Table 3)
                use crate::arch::ctrl_packet::*;
                let address = word & ADDRESS_MASK;
                let beats = ((word >> LENGTH_SHIFT) & LENGTH_MASK) as u8 + 1;
                let operation = ((word >> OPERATION_SHIFT) & OPERATION_MASK) as u8;
                let response_id = ((word >> RESPONSE_ID_SHIFT) & RESPONSE_ID_MASK) as u8;

                log::info!("Tile ({},{}) ctrl_pkt: header 0x{:08X} addr=0x{:05X} op={} beats={} resp_id={}",
                    self.col, self.row, word, address, operation, beats, response_id);

                // OP_READ has no data payload -- execute immediately
                if operation == OP_READ {
                    let actions = self.execute_ctrl_packet(
                        address, operation, response_id, beats, &[],
                    );
                    self.ctrl_pkt_state = if tlast && !self.ctrl_pkt_drop_header {
                        ControlPacketState::WaitingForStreamHeader
                    } else {
                        ControlPacketState::Idle
                    };
                    return actions;
                }

                self.ctrl_pkt_state = ControlPacketState::Collecting {
                    address,
                    operation,
                    response_id,
                    beats_total: beats,
                    beats_collected: 0,
                    data: [0; 4],
                };
                Vec::new()
            }
            ControlPacketState::Collecting {
                address,
                operation,
                response_id,
                beats_total,
                mut beats_collected,
                mut data,
            } => {
                data[beats_collected as usize] = word;
                beats_collected += 1;
                log::debug!("Tile ({},{}) ctrl_pkt: data[{}] = 0x{:08X} ({}/{}){}",
                    self.col, self.row, beats_collected - 1, word, beats_collected, beats_total,
                    if tlast { " TLAST" } else { "" });

                if beats_collected >= beats_total {
                    // All beats received -- execute the operation
                    let actions = self.execute_ctrl_packet(
                        address, operation, response_id, beats_total,
                        &data[..beats_collected as usize],
                    );
                    // After completion: if TLAST marks end of stream packet AND
                    // headers aren't dropped, expect a stream header next time.
                    // Otherwise stay Idle for the next ctrl packet within this
                    // same stream packet.
                    self.ctrl_pkt_state = if tlast && !self.ctrl_pkt_drop_header {
                        ControlPacketState::WaitingForStreamHeader
                    } else {
                        ControlPacketState::Idle
                    };
                    actions
                } else {
                    // Still collecting
                    self.ctrl_pkt_state = ControlPacketState::Collecting {
                        address,
                        operation,
                        response_id,
                        beats_total,
                        beats_collected,
                        data,
                    };
                    Vec::new()
                }
            }
        }
    }

    /// Execute a complete control packet operation.
    ///
    /// Returns a list of actions for the caller to dispatch through
    /// `DeviceState::write_tile_register()`, which provides the full module
    /// dispatch (MemTile DMA BDs, stream switch, etc.).
    ///
    /// Currently supports:
    /// - Operation 0 (write): Write data words to consecutive register addresses
    /// - Operation 1 (read): Logged, returns ReadRegisters action
    /// - Operation 2 (write_incr): Same as write
    /// - Operation 3 (block_write): Same as write
    fn execute_ctrl_packet(
        &self,
        base_address: u32,
        operation: u8,
        response_id: u8,
        beats_total: u8,
        data: &[u32],
    ) -> Vec<CtrlPacketAction> {
        use crate::arch::ctrl_packet::*;
        let mut actions = Vec::new();

        match operation {
            OP_WRITE | OP_BLOCK_WRITE => {
                for (i, &value) in data.iter().enumerate() {
                    let addr = base_address + (i as u32) * 4;
                    log::info!("Tile ({},{}) ctrl_pkt WRITE: [0x{:05X}] = 0x{:08X}",
                        self.col, self.row, addr, value);
                    actions.push(CtrlPacketAction::WriteRegister {
                        col: self.col,
                        row: self.row,
                        offset: addr,
                        value,
                    });
                }
            }
            OP_READ => {
                log::info!(
                    "Tile ({},{}) ctrl_pkt READ: addr=0x{:05X} beats={} resp_id={}",
                    self.col, self.row, base_address, beats_total, response_id,
                );
                actions.push(CtrlPacketAction::ReadRegisters {
                    col: self.col,
                    row: self.row,
                    offset: base_address,
                    count: beats_total,
                    response_id,
                });
            }
            OP_WRITE_INCR => {
                for (i, &value) in data.iter().enumerate() {
                    let addr = base_address + (i as u32) * 4;
                    log::info!("Tile ({},{}) ctrl_pkt WRITE_INCR: [0x{:05X}] = 0x{:08X}",
                        self.col, self.row, addr, value);
                    actions.push(CtrlPacketAction::WriteRegister {
                        col: self.col,
                        row: self.row,
                        offset: addr,
                        value,
                    });
                }
            }
            _ => {
                let msg = format!(
                    "Tile ({},{}) ctrl_pkt: unknown operation {} (addr=0x{:05X}) -- impossible on hardware",
                    self.col, self.row, operation, base_address,
                );
                log::error!("{}", msg);
                actions.push(CtrlPacketAction::Error(msg));
            }
        }

        actions
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

        // Perform the operation.
        //
        // AIE-ML lock semantics (matching DMA engine in dma/engine.rs):
        // - change_value < 0: acq_ge -- wait until lock >= |value|, then decrement
        // - change_value > 0: acq_eq -- wait until lock == value, then set to 0
        // - change_value == 0: simple acquire (decrement if > 0)
        let result = if is_acquire {
            if change_value < 0 {
                // acq_ge: wait until lock >= |value|, then decrement by |value|
                let expected = (-change_value) as i8;
                self.locks[lock_id].acquire_with_value(expected, change_value)
            } else if change_value > 0 {
                // acq_eq: wait until lock == value, then decrement to 0
                let expected = change_value as i8;
                let delta = -expected;
                self.locks[lock_id].acquire_equal(expected, delta)
            } else {
                // Simple acquire: decrement by 1 if > 0
                self.locks[lock_id].acquire_with_value(1, -1)
            }
        } else {
            // Release: apply delta (typically positive)
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
    pub(crate) fn clear_lock_overflow_bits(&mut self, start: usize, end: usize, bits: u32) {
        for i in start..end.min(self.locks.len()) {
            if bits & (1 << (i - start)) != 0 {
                self.locks[i].overflow = false;
            }
        }
    }

    /// Clear lock underflow bits for a range (write-to-clear behavior).
    pub(crate) fn clear_lock_underflow_bits(&mut self, start: usize, end: usize, bits: u32) {
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
        assert_eq!(tile.locks.len(), 16);
        assert_eq!(tile.dma_bds.len(), 16);
        assert_eq!(tile.dma_channels.len(), 4);
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
        assert_eq!(lock.acquire_with_value(5, -3), LockResult::PreconditionNotMet);
        assert_eq!(lock.value, 2); // Value unchanged

        // Acquire all remaining
        assert_eq!(lock.acquire_with_value(2, -2), LockResult::Success);
        assert_eq!(lock.value, 0);

        // Can't acquire when value is 0
        assert_eq!(lock.acquire_with_value(1, -1), LockResult::PreconditionNotMet);
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
        assert_eq!(lock.release_with_value(-62), LockResult::PreconditionNotMet);
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
        assert_eq!(lock.acquire_equal(1, -1), LockResult::PreconditionNotMet);
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
        assert_eq!(lock.acquire_equal(3, -3), LockResult::PreconditionNotMet);
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
        // Core module edge detection register (0x34408)
        // Event 0: event=42, rising=1; Event 1: event=50, falling=1
        let value = (1u32 << 26) | (50 << 16) | (1 << 9) | 42;
        Tile::configure_edge_detectors(&mut tile.core_edge_detectors, value, false);

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
        // Memory module edge detection register (0x14408)
        let value = (1u32 << 25) | (30 << 16) | (1 << 10) | (1 << 9) | 20;
        Tile::configure_edge_detectors(&mut tile.mem_edge_detectors, value, false);

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
        // MemTile edge detection register (0x94408)
        // Use event > 127 to verify 8-bit field (is_memtile=true)
        let value = (1u32 << 25) | (200 << 16) | (1 << 9) | 180;
        Tile::configure_edge_detectors(&mut tile.mem_edge_detectors, value, true);

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

    // === Shim Tile Tracing Tests ===

    #[test]
    fn test_shim_trace_register_write() {
        let mut device = super::super::state::DeviceState::new_npu1();
        // Write Trace_Control0 at 0x340D0 (same offset as core module)
        // start_event=1 (TRUE), stop_event=0 (NONE), mode=0 (event-time)
        let ctrl0 = (0u32 << 24) | (1 << 16) | 0;
        device.write_tile_register(0, 0, 0x340D0, ctrl0);
        // Trace unit should now be configured
        let tile = device.array.get(0, 0).unwrap();
        assert!(tile.core_trace.is_configured());
    }

    #[test]
    fn test_shim_edge_detection_register() {
        let mut tile = Tile::shim(0, 0);
        // Edge detection register at 0x34408 for PL module
        // Shim uses core_edge_detectors for its PL module
        let value = (1u32 << 25) | (14 << 16) | (1 << 9) | 22;
        Tile::configure_edge_detectors(&mut tile.core_edge_detectors, value, false);

        assert_eq!(tile.core_edge_detectors[0].input_event, 22);
        assert!(tile.core_edge_detectors[0].trigger_rising);

        assert_eq!(tile.core_edge_detectors[1].input_event, 14);
        assert!(tile.core_edge_detectors[1].trigger_rising);
    }

    #[test]
    fn test_shim_dma_event_notification() {
        let mut device = super::super::state::DeviceState::new_npu1();
        // Configure trace unit with start=TRUE(1)
        device.write_tile_register(0, 0, 0x340D0, (1 << 16) | 0); // start=1, mode=0

        // Shim DMA events go through core_trace (PL module)
        // DMA_S2MM_0_START_TASK = PL event 14
        let tile = device.array.get_mut(0, 0).unwrap();
        tile.notify_core_trace_event(14, 100);
        // Should not panic, trace unit accepts it
    }

    // === Cascade Stream Tests ===

    #[test]
    fn test_cascade_init_state() {
        let tile = Tile::compute(1, 2);
        assert!(tile.cascade_input.is_empty());
        assert!(tile.cascade_output.is_empty());
        assert_eq!(tile.cascade_input_dir, 0);
        assert_eq!(tile.cascade_output_dir, 0);
    }

    #[test]
    fn test_cascade_register_write() {
        let mut device = super::super::state::DeviceState::new_npu1();

        // Input=North(0), Output=South(0)
        device.write_tile_register(1, 2, 0x36060, 0b00);
        let tile = device.array.get(1, 2).unwrap();
        assert_eq!(tile.cascade_input_dir, 0);
        assert_eq!(tile.cascade_output_dir, 0);

        // Input=West(1), Output=East(1)
        device.write_tile_register(1, 2, 0x36060, 0b11);
        let tile = device.array.get(1, 2).unwrap();
        assert_eq!(tile.cascade_input_dir, 1);
        assert_eq!(tile.cascade_output_dir, 1);

        // Input=West(1), Output=South(0)
        device.write_tile_register(1, 2, 0x36060, 0b01);
        let tile = device.array.get(1, 2).unwrap();
        assert_eq!(tile.cascade_input_dir, 1);
        assert_eq!(tile.cascade_output_dir, 0);

        // Input=North(0), Output=East(1)
        device.write_tile_register(1, 2, 0x36060, 0b10);
        let tile = device.array.get(1, 2).unwrap();
        assert_eq!(tile.cascade_input_dir, 0);
        assert_eq!(tile.cascade_output_dir, 1);
    }

    #[test]
    fn test_cascade_register_ignored_for_non_compute() {
        let mut device = super::super::state::DeviceState::new_npu1();
        device.write_tile_register(1, 1, 0x36060, 0b11);
        // MemTile should not have cascade direction changed
        let tile = device.array.get(1, 1).unwrap();
        assert_eq!(tile.cascade_input_dir, 0);
        assert_eq!(tile.cascade_output_dir, 0);
    }

    #[test]
    fn test_cascade_fifo_push_pop() {
        let mut tile = Tile::compute(1, 2);
        let data: [u64; 6] = [1, 2, 3, 4, 5, 6];

        assert!(!tile.has_cascade_input());
        tile.push_cascade_input(data);
        assert!(tile.has_cascade_input());

        let result = tile.pop_cascade_input().unwrap();
        assert_eq!(result, data);
        assert!(!tile.has_cascade_input());
        assert!(tile.pop_cascade_input().is_none());
    }

    #[test]
    fn test_cascade_output_fifo() {
        let mut tile = Tile::compute(1, 2);
        let data: [u64; 6] = [10, 20, 30, 40, 50, 60];

        assert!(!tile.has_cascade_output());
        tile.push_cascade_output(data);
        assert!(tile.has_cascade_output());

        let result = tile.pop_cascade_output().unwrap();
        assert_eq!(result, data);
        assert!(!tile.has_cascade_output());
    }

    /// Proves that DeviceState::write_tile_register() correctly dispatches
    /// MemTile BD writes through the full module dispatch path. This is the
    /// unified path used by all register write sources (CDO, NPU executor,
    /// control packets).
    #[test]
    fn test_write_tile_register_updates_memtile_bd() {
        let reg_layout = super::super::regdb::device_reg_layout();
        let bd0_word2_offset = reg_layout.memtile_bd_base + 2 * 4; // BD0, word 2 (length)

        let mut device = super::super::state::DeviceState::new_npu1();

        // Verify BD starts zeroed on the MemTile (col=1, row=1)
        let tile = device.array.get(1, 1).expect("tile(1,1) should exist");
        assert!(tile.is_mem_tile(), "tile(1,1) should be a MemTile");
        assert_eq!(tile.dma_bds[0].length, 0, "BD0 length should start at 0");

        // Write via write_tile_register -- the unified register bus
        let test_length: u32 = 0x0000_1000;
        device.write_tile_register(1, 1, bd0_word2_offset, test_length);

        // Both the register HashMap AND the structured BD should be updated
        let tile = device.array.get(1, 1).unwrap();
        assert_eq!(
            *tile.registers_ref().get(&bd0_word2_offset).unwrap_or(&0),
            test_length,
            "Register HashMap should have the value"
        );
        assert_eq!(
            tile.dma_bds[0].length, test_length,
            "MemTile BD0 length should be updated via write_tile_register dispatch"
        );
    }

    /// OP_READ (operation=1) produces a ReadRegisters action immediately upon
    /// receiving the header, with no data payload. The action carries the
    /// offset, count (beats+1), and response_id from the header.
    #[test]
    fn test_ctrl_packet_op_read_produces_read_registers_action() {
        use crate::arch::ctrl_packet::*;

        let mut tile = Tile::compute(2, 3);

        // Pre-populate registers at 0x440, 0x444, 0x448, 0x44C with known values.
        // Direct register map insertion -- no side effects needed for these offsets.
        tile.registers.insert(0x440, 0xDEAD_0001);
        tile.registers.insert(0x444, 0xDEAD_0002);
        tile.registers.insert(0x448, 0xDEAD_0003);
        tile.registers.insert(0x44C, 0xDEAD_0004);

        // Build OP_READ control packet header:
        //   address   = 0x440 (bits 19:0)
        //   beats     = 3 (bits 21:20) -> actual count = 3 + 1 = 4
        //   operation = 1 (bits 23:22) = OP_READ
        //   response_id = 2 (bits 30:24)
        let address: u32 = 0x440;
        let beats_raw: u32 = 3; // means 4 words
        let operation: u32 = OP_READ as u32;
        let response_id: u32 = 2;

        let header = address
            | (beats_raw << LENGTH_SHIFT)
            | (operation << OPERATION_SHIFT)
            | (response_id << RESPONSE_ID_SHIFT);

        // Start from Idle state (skip stream header)
        tile.ctrl_pkt_state = ControlPacketState::Idle;

        // OP_READ has no data payload -- TLAST on the header itself
        let actions = tile.process_ctrl_packet_word(header, true);

        // Should produce exactly one ReadRegisters action
        assert_eq!(actions.len(), 1, "OP_READ should produce exactly one action");

        match &actions[0] {
            CtrlPacketAction::ReadRegisters { col, row, offset, count, response_id: rid } => {
                assert_eq!(*col, 2, "col should match tile col");
                assert_eq!(*row, 3, "row should match tile row");
                assert_eq!(*offset, 0x440, "offset should be the address from header");
                assert_eq!(*count, 4, "count should be beats+1 = 4");
                assert_eq!(*rid, 2, "response_id should match header");
            }
            other => panic!("Expected ReadRegisters, got {:?}", other),
        }

        // State machine should return to Idle (header was dropped)
        assert!(
            matches!(tile.ctrl_pkt_state, ControlPacketState::Idle),
            "State should be Idle after OP_READ with TLAST and drop_header=true"
        );
    }

    // === Lock Trace Event Pipeline Tests ===

    #[test]
    fn test_lock_event_reaches_trace_unit() {
        // Verify end-to-end: lock acquire -> mem_trace_pending -> trace unit capture.
        //
        // This tests the full pipeline that sweep batch 3 exercises:
        // 1. Lock acquire resolves -> pushes EventType::LockAcquire{lock_id:0}
        // 2. mem_event_to_hw_id maps to 45 (LOCK_SEL0_ACQ_GE)
        // 3. Trace unit with slot configured to 45 captures the event
        let mut tile = Tile::compute(0, 2);

        // Initialize lock 0 to value 1 so an acquire(>=1) will succeed.
        tile.locks[0] = Lock { value: 1, ..Default::default() };

        // Configure mem trace unit:
        //   Control0: mode=EventTime(0), start=1 (TRUE), stop=0
        //   Event0: slot0=1(TRUE), slot1=45(LOCK_SEL0_ACQ_GE), slot2=46(LOCK_0_REL)
        tile.mem_trace.write_register(0x00, 0 | (1 << 16) | (0 << 24)); // start=TRUE(1)
        tile.mem_trace.write_register(0x04, (1 << 12) | 1); // pkt_type=1, pkt_id=1
        tile.mem_trace.write_register(0x10, 1 | (45 << 8) | (46 << 16)); // slots 0-2

        // Start the trace unit by firing TRUE event
        tile.mem_trace.notify_event(1, 0); // TRUE at cycle 0
        assert!(tile.mem_trace.is_configured());

        // Submit and resolve a lock acquire on lock 0
        tile.submit_lock_request(LockRequest {
            requestor: LockRequestor::DmaS2mm(0),
            lock_id: 0,
            is_acquire: true,
            expected: 1,
            delta: -1,
            equal_mode: false,
        });
        let results = tile.resolve_lock_requests(100);

        // Lock should be granted (value was 1, needed >=1)
        assert_eq!(results.len(), 1, "Expected one lock result");
        assert!(results[0].2, "Lock acquire should be granted");
        assert!(results[0].3, "Should be marked as acquire");

        // mem_trace_pending should have the lock event
        assert_eq!(tile.mem_trace_pending.len(), 1, "Expected one pending trace event");
        let (cycle, ref event) = tile.mem_trace_pending[0];
        assert_eq!(cycle, 100);
        assert!(
            matches!(event, crate::interpreter::state::EventType::LockAcquire { lock_id: 0 }),
            "Expected LockAcquire{{lock_id:0}}, got {:?}", event
        );

        // Map through mem_event_to_hw_id -- should return 45
        let hw_id = crate::trace::mem_event_to_hw_id(event);
        assert_eq!(hw_id, Some(45), "LOCK_SEL0_ACQ_GE should be event ID 45");

        // Notify the trace unit and flush to check capture
        tile.mem_trace.notify_event(45, 100);
        tile.mem_trace.flush();
        assert!(
            tile.mem_trace.has_pending_packets(),
            "Trace unit should have recorded the lock event (packet pending after flush)"
        );
    }

    #[test]
    fn test_lock_release_event_reaches_trace_unit() {
        let mut tile = Tile::compute(0, 2);

        // Configure mem trace with LOCK_0_REL (46) in slot 1
        tile.mem_trace.write_register(0x00, 0 | (1 << 16)); // start=TRUE(1)
        tile.mem_trace.write_register(0x04, (1 << 12) | 1);
        tile.mem_trace.write_register(0x10, 1 | (46 << 8)); // slot0=TRUE, slot1=LOCK_0_REL
        tile.mem_trace.notify_event(1, 0); // start

        // Submit a lock release on lock 0
        tile.submit_lock_request(LockRequest {
            requestor: LockRequestor::Core,
            lock_id: 0,
            is_acquire: false,
            expected: 0,
            delta: 1,
            equal_mode: false,
        });
        let results = tile.resolve_lock_requests(50);

        assert_eq!(results.len(), 1);
        assert!(results[0].2, "Release should be granted");
        assert!(!results[0].3, "Should be marked as release");

        // Check pending event
        assert_eq!(tile.mem_trace_pending.len(), 1);
        let hw_id = crate::trace::mem_event_to_hw_id(&tile.mem_trace_pending[0].1);
        assert_eq!(hw_id, Some(46), "LOCK_0_REL should be event ID 46");

        // Notify, flush, and verify capture
        tile.mem_trace.notify_event(46, 50);
        tile.mem_trace.flush();
        assert!(
            tile.mem_trace.has_pending_packets(),
            "Trace unit should have recorded the lock release event"
        );
    }
}
