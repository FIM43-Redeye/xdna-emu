//! Synchronization timing and contention tracking.
//!
//! This module provides cycle-accurate timing for lock operations:
//!
//! - **Lock acquire latency**: Cycles for successful acquisition
//! - **Lock release latency**: Cycles for release
//! - **Contention tracking**: Stall cycles when lock is busy
//! - **Lock state tracking**: Per-lock statistics
//!
//! # AIE2 Lock Model (AM020 Ch2, AM025)
//!
//! AIE2 uses semaphore locks with 6-bit unsigned values (0-63):
//! - Acquire: Check value against threshold, apply delta if successful
//! - Release: Apply delta (non-blocking), saturating at 63
//! - Contention: Core stalls and retries each cycle until successful
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::interpreter::timing::sync::{LockTimingState, SyncTimingConfig};
//!
//! let config = SyncTimingConfig::from_aie2_spec();
//! let mut state = LockTimingState::new(16); // 16 locks
//!
//! // Track a lock acquire attempt
//! state.track_acquire(5, false); // Lock 5, failed
//! state.track_acquire(5, false); // Still waiting
//! state.track_acquire(5, true);  // Succeeded after 2 stall cycles
//!
//! println!("Lock 5 contention cycles: {}", state.contention_cycles(5));
//! ```

use crate::device::aie2_spec;

/// Configuration for synchronization timing.
#[derive(Debug, Clone, Copy)]
pub struct SyncTimingConfig {
    /// Cycles for a successful lock acquire
    pub acquire_latency: u8,
    /// Cycles for lock release
    pub release_latency: u8,
    /// Cycles between retry attempts when contended
    pub retry_interval: u8,
}

impl Default for SyncTimingConfig {
    fn default() -> Self {
        Self::from_aie2_spec()
    }
}

impl SyncTimingConfig {
    /// Create configuration from AIE2 spec constants.
    pub fn from_aie2_spec() -> Self {
        Self {
            acquire_latency: aie2_spec::LOCK_ACQUIRE_LATENCY,
            release_latency: aie2_spec::LOCK_RELEASE_LATENCY,
            retry_interval: aie2_spec::LOCK_RETRY_INTERVAL,
        }
    }

    /// Create instant timing (no delays, for fast simulation).
    pub fn instant() -> Self {
        Self {
            acquire_latency: 0,
            release_latency: 0,
            retry_interval: 0,
        }
    }
}

/// Statistics for a single lock.
#[derive(Debug, Clone, Copy, Default)]
pub struct LockStats {
    /// Number of successful acquires
    pub acquires: u64,
    /// Number of successful releases
    pub releases: u64,
    /// Total cycles spent waiting for this lock (contention)
    pub contention_cycles: u64,
    /// Number of acquire attempts that had to retry
    pub contention_count: u64,
    /// Maximum consecutive stall cycles observed
    pub max_contention: u64,
    /// Current consecutive stall cycles (reset on success)
    current_stall: u64,
}

impl LockStats {
    /// Record a successful acquire.
    pub fn record_acquire(&mut self, stall_cycles: u64) {
        self.acquires += 1;
        self.contention_cycles += stall_cycles;
        if stall_cycles > 0 {
            self.contention_count += 1;
            self.max_contention = self.max_contention.max(stall_cycles);
        }
        self.current_stall = 0;
    }

    /// Record a stall cycle (failed acquire attempt).
    pub fn record_stall(&mut self) {
        self.current_stall += 1;
    }

    /// Record a successful release.
    pub fn record_release(&mut self) {
        self.releases += 1;
    }

    /// Get current consecutive stall cycles.
    pub fn current_stall(&self) -> u64 {
        self.current_stall
    }
}

/// Lock timing state for a tile.
///
/// Tracks timing and statistics for all locks in a tile.
#[derive(Debug)]
pub struct LockTimingState {
    /// Configuration for timing calculations
    config: SyncTimingConfig,
    /// Per-lock statistics
    stats: Vec<LockStats>,
}

impl LockTimingState {
    /// Create a new lock timing state for the given number of locks.
    pub fn new(num_locks: usize) -> Self {
        Self {
            config: SyncTimingConfig::from_aie2_spec(),
            stats: vec![LockStats::default(); num_locks],
        }
    }

    /// Create with custom configuration.
    pub fn with_config(num_locks: usize, config: SyncTimingConfig) -> Self {
        Self {
            config,
            stats: vec![LockStats::default(); num_locks],
        }
    }

    /// Get the timing configuration.
    pub fn config(&self) -> &SyncTimingConfig {
        &self.config
    }

    /// Track a lock acquire attempt.
    ///
    /// Returns the number of cycles for this operation.
    pub fn track_acquire(&mut self, lock_id: usize, success: bool) -> u64 {
        if lock_id >= self.stats.len() {
            return 0;
        }

        if success {
            let stall_cycles = self.stats[lock_id].current_stall();
            self.stats[lock_id].record_acquire(stall_cycles);
            self.config.acquire_latency as u64
        } else {
            self.stats[lock_id].record_stall();
            self.config.retry_interval as u64
        }
    }

    /// Track a lock release.
    ///
    /// Returns the number of cycles for this operation.
    pub fn track_release(&mut self, lock_id: usize) -> u64 {
        if lock_id < self.stats.len() {
            self.stats[lock_id].record_release();
        }
        self.config.release_latency as u64
    }

    /// Get statistics for a specific lock.
    pub fn stats(&self, lock_id: usize) -> Option<&LockStats> {
        self.stats.get(lock_id)
    }

    /// Get the total contention cycles for a lock.
    pub fn contention_cycles(&self, lock_id: usize) -> u64 {
        self.stats.get(lock_id).map_or(0, |s| s.contention_cycles)
    }

    /// Get the current stall cycles for a lock.
    pub fn current_stall(&self, lock_id: usize) -> u64 {
        self.stats.get(lock_id).map_or(0, |s| s.current_stall)
    }

    /// Get aggregate statistics for all locks.
    pub fn aggregate_stats(&self) -> AggregateStats {
        let mut agg = AggregateStats::default();
        for stat in &self.stats {
            agg.total_acquires += stat.acquires;
            agg.total_releases += stat.releases;
            agg.total_contention_cycles += stat.contention_cycles;
            agg.total_contention_count += stat.contention_count;
            agg.max_contention = agg.max_contention.max(stat.max_contention);
        }
        agg
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        for stat in &mut self.stats {
            *stat = LockStats::default();
        }
    }
}

/// Aggregate statistics across all locks.
#[derive(Debug, Clone, Default)]
pub struct AggregateStats {
    /// Total successful acquires
    pub total_acquires: u64,
    /// Total successful releases
    pub total_releases: u64,
    /// Total cycles spent in contention
    pub total_contention_cycles: u64,
    /// Number of times any lock was contended
    pub total_contention_count: u64,
    /// Maximum contention observed on any lock
    pub max_contention: u64,
}

impl AggregateStats {
    /// Calculate average contention per acquire.
    pub fn avg_contention(&self) -> f64 {
        if self.total_acquires == 0 {
            0.0
        } else {
            self.total_contention_cycles as f64 / self.total_acquires as f64
        }
    }

    /// Calculate contention rate (fraction of acquires that were contended).
    pub fn contention_rate(&self) -> f64 {
        if self.total_acquires == 0 {
            0.0
        } else {
            self.total_contention_count as f64 / self.total_acquires as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_timing_config() {
        let config = SyncTimingConfig::from_aie2_spec();
        assert_eq!(config.acquire_latency, 1);
        assert_eq!(config.release_latency, 1);
        assert_eq!(config.retry_interval, 1);
    }

    #[test]
    fn test_lock_stats_acquire() {
        let mut stats = LockStats::default();

        // First acquire - no stall
        stats.record_acquire(0);
        assert_eq!(stats.acquires, 1);
        assert_eq!(stats.contention_cycles, 0);
        assert_eq!(stats.contention_count, 0);

        // Second acquire - 3 stall cycles
        stats.record_acquire(3);
        assert_eq!(stats.acquires, 2);
        assert_eq!(stats.contention_cycles, 3);
        assert_eq!(stats.contention_count, 1);
        assert_eq!(stats.max_contention, 3);
    }

    #[test]
    fn test_lock_timing_state() {
        let mut state = LockTimingState::new(16);

        // Acquire lock 5 successfully
        let cycles = state.track_acquire(5, true);
        assert_eq!(cycles, 1); // acquire_latency

        // Stall on lock 3
        state.track_acquire(3, false);
        state.track_acquire(3, false);
        assert_eq!(state.current_stall(3), 2);

        // Acquire lock 3 after stall
        state.track_acquire(3, true);
        assert_eq!(state.contention_cycles(3), 2);
        assert_eq!(state.current_stall(3), 0); // Reset after success
    }

    #[test]
    fn test_lock_release() {
        let mut state = LockTimingState::new(16);

        let cycles = state.track_release(7);
        assert_eq!(cycles, 1); // release_latency
        assert_eq!(state.stats(7).unwrap().releases, 1);
    }

    #[test]
    fn test_aggregate_stats() {
        let mut state = LockTimingState::new(16);

        // Lock 0: 2 acquires, no contention
        state.track_acquire(0, true);
        state.track_acquire(0, true);

        // Lock 1: 1 acquire, 3 stall cycles
        state.track_acquire(1, false);
        state.track_acquire(1, false);
        state.track_acquire(1, false);
        state.track_acquire(1, true);

        // Lock 2: 2 releases
        state.track_release(2);
        state.track_release(2);

        let agg = state.aggregate_stats();
        assert_eq!(agg.total_acquires, 3);
        assert_eq!(agg.total_releases, 2);
        assert_eq!(agg.total_contention_cycles, 3);
        assert_eq!(agg.total_contention_count, 1);
        assert_eq!(agg.max_contention, 3);
    }

    #[test]
    fn test_instant_config() {
        let config = SyncTimingConfig::instant();
        assert_eq!(config.acquire_latency, 0);
        assert_eq!(config.release_latency, 0);
        assert_eq!(config.retry_interval, 0);
    }

    #[test]
    fn test_reset() {
        let mut state = LockTimingState::new(4);

        state.track_acquire(0, true);
        state.track_acquire(1, false);
        state.track_acquire(1, true);
        state.track_release(2);

        let agg = state.aggregate_stats();
        assert!(agg.total_acquires > 0);

        state.reset();
        let agg = state.aggregate_stats();
        assert_eq!(agg.total_acquires, 0);
        assert_eq!(agg.total_releases, 0);
    }
}
