//! DMA timing model.
//!
//! This module provides cycle-accurate timing for DMA operations:
//! - BD setup latency (parsing buffer descriptor)
//! - Channel startup latency
//! - Data transfer bandwidth
//! - Lock acquire/release overhead
//! - BD chaining latency
//!
//! # Timing Model
//!
//! A complete DMA transfer has these phases:
//!
//! ```text
//! ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────┐
//! │ BD Setup │─►│ Lock    │─►│ Data         │─►│ Lock    │─►│ BD Chain │
//! │ (4 cyc)  │  │ Acquire │  │ Transfer     │  │ Release │  │ (2 cyc)  │
//! │          │  │ (1 cyc) │  │ (N/bandwidth)│  │ (1 cyc) │  │          │
//! └──────────┘  └─────────┘  └──────────────┘  └─────────┘  └──────────┘
//! ```
//!
//! # Channel Arbitration
//!
//! When multiple channels are active, they share memory bandwidth:
//! - Round-robin arbitration among active channels
//! - Each channel gets one word per arbitration slot
//! - Stalls when all channels are waiting for memory

use crate::device::aie2_spec;

/// DMA timing configuration.
#[derive(Debug, Clone, Copy)]
pub struct DmaTimingConfig {
    /// Cycles to parse and setup a buffer descriptor
    pub bd_setup_cycles: u8,

    /// Cycles from channel start to first data movement
    pub channel_start_cycles: u8,

    /// Words (32-bit) transferred per cycle per channel
    pub words_per_cycle: u8,

    /// Memory access latency in cycles
    pub memory_latency_cycles: u8,

    /// Cycles to acquire a lock
    pub lock_acquire_cycles: u8,

    /// Cycles to release a lock
    pub lock_release_cycles: u8,

    /// Cycles between BD completion and next BD start
    pub bd_chain_cycles: u8,
}

impl Default for DmaTimingConfig {
    fn default() -> Self {
        Self::from_aie2_spec()
    }
}

impl DmaTimingConfig {
    /// Create timing config from AIE2 specification constants.
    pub fn from_aie2_spec() -> Self {
        Self {
            bd_setup_cycles: aie2_spec::DMA_BD_SETUP_CYCLES,
            channel_start_cycles: aie2_spec::DMA_CHANNEL_START_CYCLES,
            words_per_cycle: aie2_spec::DMA_WORDS_PER_CYCLE,
            memory_latency_cycles: aie2_spec::DMA_MEMORY_LATENCY_CYCLES,
            lock_acquire_cycles: aie2_spec::DMA_LOCK_ACQUIRE_CYCLES,
            lock_release_cycles: aie2_spec::DMA_LOCK_RELEASE_CYCLES,
            bd_chain_cycles: aie2_spec::DMA_BD_CHAIN_CYCLES,
        }
    }

    /// Create instant timing (no delays) for fast simulation.
    pub fn instant() -> Self {
        Self {
            bd_setup_cycles: 0,
            channel_start_cycles: 0,
            words_per_cycle: 255, // Max throughput
            memory_latency_cycles: 0,
            lock_acquire_cycles: 0,
            lock_release_cycles: 0,
            bd_chain_cycles: 0,
        }
    }

    /// Calculate total cycles for a transfer of given size.
    pub fn transfer_cycles(&self, bytes: u64, has_acquire_lock: bool, has_release_lock: bool) -> u64 {
        let words = (bytes + 3) / 4; // Round up to words
        let data_cycles = (words + self.words_per_cycle as u64 - 1) / self.words_per_cycle as u64;

        let mut total = self.bd_setup_cycles as u64
            + self.channel_start_cycles as u64
            + self.memory_latency_cycles as u64 // Initial memory latency
            + data_cycles;

        if has_acquire_lock {
            total += self.lock_acquire_cycles as u64;
        }
        if has_release_lock {
            total += self.lock_release_cycles as u64;
        }

        total
    }

    /// Calculate cycles for BD chaining.
    pub fn chain_cycles(&self) -> u64 {
        self.bd_chain_cycles as u64 + self.bd_setup_cycles as u64
    }
}

/// Timing state for a single DMA channel.
#[derive(Debug, Clone, Default)]
pub struct ChannelTimingState {
    /// Current phase of the transfer
    pub phase: TransferPhase,

    /// Cycles remaining in current phase
    pub cycles_remaining: u64,

    /// Total cycles spent in this transfer
    pub total_cycles: u64,

    /// Words transferred so far
    pub words_transferred: u64,

    /// Total words to transfer
    pub total_words: u64,

    /// Memory pipeline: cycles until next word can be issued
    pub memory_pipeline_busy: u8,
}

/// Phase of a DMA transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferPhase {
    /// Channel is idle
    #[default]
    Idle,
    /// Parsing buffer descriptor
    BdSetup,
    /// Waiting to acquire lock
    LockAcquire,
    /// Initial memory latency
    MemoryLatency,
    /// Actively transferring data
    DataTransfer,
    /// Releasing lock
    LockRelease,
    /// Transitioning to next BD in chain
    BdChain,
    /// Transfer complete
    Complete,
}

impl ChannelTimingState {
    /// Create a new timing state for a transfer.
    pub fn new_transfer(bytes: u64, config: &DmaTimingConfig) -> Self {
        Self {
            phase: TransferPhase::BdSetup,
            cycles_remaining: config.bd_setup_cycles as u64,
            total_cycles: 0,
            words_transferred: 0,
            total_words: (bytes + 3) / 4,
            memory_pipeline_busy: 0,
        }
    }

    /// Advance timing by one cycle.
    ///
    /// Returns the new phase after this cycle.
    pub fn tick(&mut self, config: &DmaTimingConfig, has_lock: bool, lock_available: bool) -> TransferPhase {
        self.total_cycles += 1;

        // Decrement memory pipeline counter
        if self.memory_pipeline_busy > 0 {
            self.memory_pipeline_busy -= 1;
        }

        match self.phase {
            TransferPhase::Idle => {
                // Nothing to do
            }

            TransferPhase::BdSetup => {
                if self.cycles_remaining > 0 {
                    self.cycles_remaining -= 1;
                }
                if self.cycles_remaining == 0 {
                    if has_lock {
                        self.phase = TransferPhase::LockAcquire;
                        self.cycles_remaining = config.lock_acquire_cycles as u64;
                    } else {
                        self.phase = TransferPhase::MemoryLatency;
                        self.cycles_remaining = config.memory_latency_cycles as u64;
                    }
                }
            }

            TransferPhase::LockAcquire => {
                if lock_available {
                    if self.cycles_remaining > 0 {
                        self.cycles_remaining -= 1;
                    }
                    if self.cycles_remaining == 0 {
                        self.phase = TransferPhase::MemoryLatency;
                        self.cycles_remaining = config.memory_latency_cycles as u64;
                    }
                }
                // If lock not available, we stall (cycles_remaining doesn't decrement)
            }

            TransferPhase::MemoryLatency => {
                if self.cycles_remaining > 0 {
                    self.cycles_remaining -= 1;
                }
                if self.cycles_remaining == 0 {
                    self.phase = TransferPhase::DataTransfer;
                    self.cycles_remaining = self.total_words;
                }
            }

            TransferPhase::DataTransfer => {
                // Transfer words based on bandwidth
                let words_this_cycle = (config.words_per_cycle as u64).min(self.cycles_remaining);
                self.words_transferred += words_this_cycle;
                self.cycles_remaining -= words_this_cycle;

                if self.cycles_remaining == 0 {
                    if has_lock {
                        self.phase = TransferPhase::LockRelease;
                        self.cycles_remaining = config.lock_release_cycles as u64;
                    } else {
                        self.phase = TransferPhase::Complete;
                    }
                }
            }

            TransferPhase::LockRelease => {
                if self.cycles_remaining > 0 {
                    self.cycles_remaining -= 1;
                }
                if self.cycles_remaining == 0 {
                    self.phase = TransferPhase::Complete;
                }
            }

            TransferPhase::BdChain => {
                if self.cycles_remaining > 0 {
                    self.cycles_remaining -= 1;
                }
                if self.cycles_remaining == 0 {
                    // Ready for next BD
                    self.phase = TransferPhase::BdSetup;
                    self.cycles_remaining = config.bd_setup_cycles as u64;
                }
            }

            TransferPhase::Complete => {
                // Nothing to do
            }
        }

        self.phase
    }

    /// Start BD chaining to next descriptor.
    pub fn start_chain(&mut self, config: &DmaTimingConfig) {
        self.phase = TransferPhase::BdChain;
        self.cycles_remaining = config.bd_chain_cycles as u64;
        self.words_transferred = 0;
    }

    /// Check if transfer is complete.
    pub fn is_complete(&self) -> bool {
        self.phase == TransferPhase::Complete
    }

    /// Check if actively transferring data.
    pub fn is_transferring(&self) -> bool {
        self.phase == TransferPhase::DataTransfer
    }

    /// Check if waiting for lock.
    pub fn is_waiting_for_lock(&self) -> bool {
        self.phase == TransferPhase::LockAcquire
    }

    /// Get bytes transferred so far.
    pub fn bytes_transferred(&self) -> u64 {
        self.words_transferred * 4
    }
}

/// Channel arbitration state for multi-channel DMA.
#[derive(Debug, Clone)]
pub struct ChannelArbiter {
    /// Number of channels
    num_channels: usize,

    /// Current arbitration slot (round-robin)
    current_slot: usize,

    /// Channels that are active and need bandwidth
    active_mask: u16,

    /// Memory bandwidth: words available this cycle
    bandwidth_available: u8,

    /// Memory bandwidth: words per cycle total
    bandwidth_per_cycle: u8,
}

impl ChannelArbiter {
    /// Create a new arbiter for the given number of channels.
    pub fn new(num_channels: usize, bandwidth_per_cycle: u8) -> Self {
        Self {
            num_channels,
            current_slot: 0,
            active_mask: 0,
            bandwidth_available: bandwidth_per_cycle,
            bandwidth_per_cycle,
        }
    }

    /// Mark a channel as active (needs bandwidth).
    pub fn set_active(&mut self, channel: usize) {
        if channel < self.num_channels {
            self.active_mask |= 1 << channel;
        }
    }

    /// Mark a channel as inactive.
    pub fn set_inactive(&mut self, channel: usize) {
        if channel < self.num_channels {
            self.active_mask &= !(1 << channel);
        }
    }

    /// Check if a channel is active.
    pub fn is_active(&self, channel: usize) -> bool {
        channel < self.num_channels && (self.active_mask & (1 << channel)) != 0
    }

    /// Get number of active channels.
    pub fn active_count(&self) -> usize {
        self.active_mask.count_ones() as usize
    }

    /// Reset bandwidth for a new cycle.
    pub fn new_cycle(&mut self) {
        self.bandwidth_available = self.bandwidth_per_cycle;
    }

    /// Try to allocate bandwidth for a channel.
    ///
    /// Returns the number of words allocated (0 if no bandwidth available).
    pub fn allocate(&mut self, channel: usize, words_requested: u8) -> u8 {
        if !self.is_active(channel) {
            return 0;
        }

        if self.bandwidth_available == 0 {
            return 0;
        }

        // Round-robin: only allocate if it's this channel's turn
        // With only a few active channels, we give each a fair share
        let active = self.active_count();
        if active == 0 {
            return 0;
        }

        // Fair share of bandwidth
        let fair_share = (self.bandwidth_per_cycle as usize / active).max(1) as u8;
        let words = words_requested.min(fair_share).min(self.bandwidth_available);

        self.bandwidth_available -= words;
        words
    }

    /// Advance to next arbitration slot.
    pub fn advance_slot(&mut self) {
        self.current_slot = (self.current_slot + 1) % self.num_channels;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_config_default() {
        let config = DmaTimingConfig::default();
        assert_eq!(config.bd_setup_cycles, 4);
        assert_eq!(config.words_per_cycle, 1);
    }

    #[test]
    fn test_timing_config_instant() {
        let config = DmaTimingConfig::instant();
        assert_eq!(config.bd_setup_cycles, 0);
        assert_eq!(config.memory_latency_cycles, 0);
    }

    #[test]
    fn test_transfer_cycles_simple() {
        let config = DmaTimingConfig::default();

        // 16 bytes = 4 words, no locks
        let cycles = config.transfer_cycles(16, false, false);
        // BD setup (4) + channel start (2) + memory latency (5) + data (4) = 15
        assert_eq!(cycles, 15);
    }

    #[test]
    fn test_transfer_cycles_with_locks() {
        let config = DmaTimingConfig::default();

        // 16 bytes = 4 words, with locks
        let cycles = config.transfer_cycles(16, true, true);
        // BD setup (4) + channel start (2) + memory latency (5) + data (4) + acquire (1) + release (1) = 17
        assert_eq!(cycles, 17);
    }

    #[test]
    fn test_channel_timing_phases() {
        let config = DmaTimingConfig::default();
        let mut state = ChannelTimingState::new_transfer(16, &config);

        assert_eq!(state.phase, TransferPhase::BdSetup);

        // Tick through BD setup (4 cycles)
        for _ in 0..4 {
            state.tick(&config, false, true);
        }
        assert_eq!(state.phase, TransferPhase::MemoryLatency);

        // Tick through memory latency (5 cycles)
        for _ in 0..5 {
            state.tick(&config, false, true);
        }
        assert_eq!(state.phase, TransferPhase::DataTransfer);

        // Tick through data transfer (4 words)
        for _ in 0..4 {
            state.tick(&config, false, true);
        }
        assert_eq!(state.phase, TransferPhase::Complete);
    }

    #[test]
    fn test_channel_timing_with_lock_stall() {
        let config = DmaTimingConfig::default();
        let mut state = ChannelTimingState::new_transfer(16, &config);

        // Tick through BD setup
        for _ in 0..4 {
            state.tick(&config, true, true);
        }
        assert_eq!(state.phase, TransferPhase::LockAcquire);

        // Lock not available - should stall
        state.tick(&config, true, false);
        assert_eq!(state.phase, TransferPhase::LockAcquire);
        state.tick(&config, true, false);
        assert_eq!(state.phase, TransferPhase::LockAcquire);

        // Lock becomes available
        state.tick(&config, true, true);
        assert_eq!(state.phase, TransferPhase::MemoryLatency);
    }

    #[test]
    fn test_arbiter_round_robin() {
        let mut arbiter = ChannelArbiter::new(4, 4);

        // Activate channels 0 and 2
        arbiter.set_active(0);
        arbiter.set_active(2);

        assert_eq!(arbiter.active_count(), 2);

        // Each should get fair share (2 words each with bandwidth of 4)
        let alloc0 = arbiter.allocate(0, 4);
        assert_eq!(alloc0, 2);

        let alloc2 = arbiter.allocate(2, 4);
        assert_eq!(alloc2, 2);

        // No more bandwidth
        let alloc0_again = arbiter.allocate(0, 1);
        assert_eq!(alloc0_again, 0);

        // New cycle resets bandwidth
        arbiter.new_cycle();
        let alloc0_new = arbiter.allocate(0, 1);
        assert_eq!(alloc0_new, 1);
    }

    #[test]
    fn test_arbiter_inactive_channel() {
        let mut arbiter = ChannelArbiter::new(4, 4);
        arbiter.set_active(0);

        // Inactive channel gets nothing
        let alloc1 = arbiter.allocate(1, 4);
        assert_eq!(alloc1, 0);

        // Active channel gets full bandwidth
        let alloc0 = arbiter.allocate(0, 4);
        assert_eq!(alloc0, 4);
    }

    #[test]
    fn test_bd_chain_cycles() {
        let config = DmaTimingConfig::default();
        let mut state = ChannelTimingState::new_transfer(4, &config);

        // Complete a transfer quickly using instant config
        let instant = DmaTimingConfig::instant();
        while !state.is_complete() {
            state.tick(&instant, false, true);
        }

        // Start chaining
        state.start_chain(&config);
        assert_eq!(state.phase, TransferPhase::BdChain);

        // Should take bd_chain_cycles + bd_setup_cycles to get to next transfer
        for _ in 0..config.bd_chain_cycles {
            state.tick(&config, false, true);
        }
        assert_eq!(state.phase, TransferPhase::BdSetup);
    }
}
