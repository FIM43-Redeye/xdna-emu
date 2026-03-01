//! DMA timing configuration.
//!
//! Provides cycle counts for each phase of a DMA transfer. The unified
//! channel FSM (`ChannelFsm`) uses these values directly as countdown
//! timers in its state variants -- no separate timing state machine needed.
//!
//! # Timing Model
//!
//! ```text
//! ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────┐
//! │ BD Setup │->│ Lock    │->│ Data         │->│ Lock    │->│ BD Chain │
//! │ (4 cyc)  │  │ Acquire │  │ Transfer     │  │ Release │  │ (2 cyc)  │
//! │          │  │ (1 cyc) │  │ (N/bandwidth)│  │ (1 cyc) │  │          │
//! └──────────┘  └─────────┘  └──────────────┘  └─────────┘  └──────────┘
//! ```

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

    /// Calculate total cycles for a transfer of given size.
    pub fn transfer_cycles(&self, bytes: u64, has_acquire_lock: bool, has_release_lock: bool) -> u64 {
        let words = (bytes + 3) / 4; // Round up to words
        let data_cycles = (words + self.words_per_cycle as u64 - 1) / self.words_per_cycle as u64;

        let mut total = self.bd_setup_cycles as u64
            + self.channel_start_cycles as u64
            + self.memory_latency_cycles as u64
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
}
