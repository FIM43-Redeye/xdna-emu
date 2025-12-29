//! Barrier synchronization timing and tracking.
//!
//! This module provides cycle-accurate timing for barrier synchronization:
//!
//! - **Barrier state**: Track which cores have arrived at a barrier
//! - **Arrival timing**: Record when each core arrives
//! - **Wait cycles**: Calculate how long each core waits for others
//! - **Statistics**: Aggregate barrier performance metrics
//!
//! # AIE2 Barrier Model
//!
//! AIE2 implements barriers using semaphore locks. A typical pattern:
//!
//! 1. Initialize lock to N (number of participants)
//! 2. Each core decrements the lock on arrival (acquire with delta=-1)
//! 3. Each core waits for lock to reach 0 (all arrived)
//! 4. Last arrival releases everyone
//!
//! This module tracks the timing of this pattern without requiring
//! specific lock values - it tracks logical barrier participation.
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::interpreter::timing::barrier::{BarrierTracker, BarrierConfig};
//! use xdna_emu::interpreter::timing::deadlock::TileId;
//!
//! let mut tracker = BarrierTracker::new();
//!
//! // Define a barrier with 4 participants
//! tracker.create_barrier(0, BarrierConfig::new(4));
//!
//! // Cores arrive at different times
//! tracker.arrive(0, TileId::new(0, 2), 100);  // Core (0,2) at cycle 100
//! tracker.arrive(0, TileId::new(1, 2), 105);  // Core (1,2) at cycle 105
//! tracker.arrive(0, TileId::new(0, 3), 110);  // Core (0,3) at cycle 110
//! tracker.arrive(0, TileId::new(1, 3), 120);  // Core (1,3) at cycle 120 - all arrived!
//!
//! // Check if barrier is complete
//! assert!(tracker.is_complete(0));
//!
//! // Get wait cycles for each core
//! let wait = tracker.wait_cycles(0, TileId::new(0, 2));
//! assert_eq!(wait, Some(20)); // Waited from cycle 100 to 120
//! ```

use std::collections::{HashMap, HashSet};
use super::deadlock::TileId;

/// Barrier identifier.
pub type BarrierId = u16;

/// Configuration for a barrier.
#[derive(Debug, Clone, Copy)]
pub struct BarrierConfig {
    /// Number of participants required for barrier completion.
    pub participants: u8,
    /// Optional timeout in cycles (0 = no timeout).
    pub timeout_cycles: u64,
    /// Whether to automatically reset after completion.
    pub auto_reset: bool,
}

impl BarrierConfig {
    /// Create a new barrier configuration.
    pub fn new(participants: u8) -> Self {
        Self {
            participants,
            timeout_cycles: 0,
            auto_reset: true,
        }
    }

    /// Create with a timeout.
    pub fn with_timeout(participants: u8, timeout_cycles: u64) -> Self {
        Self {
            participants,
            timeout_cycles,
            auto_reset: true,
        }
    }

    /// Set auto-reset behavior.
    pub fn with_auto_reset(mut self, auto_reset: bool) -> Self {
        self.auto_reset = auto_reset;
        self
    }
}

impl Default for BarrierConfig {
    fn default() -> Self {
        Self::new(2)
    }
}

/// Current phase of a barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierPhase {
    /// Waiting for participants to arrive.
    Waiting,
    /// All participants have arrived (barrier complete).
    Complete,
    /// Barrier timed out before all participants arrived.
    Timeout,
}

/// State of a single barrier instance.
#[derive(Debug, Clone)]
pub struct BarrierState {
    /// Configuration for this barrier.
    config: BarrierConfig,
    /// Current phase.
    phase: BarrierPhase,
    /// Set of tiles that have arrived.
    arrived: HashSet<TileId>,
    /// Arrival time for each tile (cycle count).
    arrival_times: HashMap<TileId, u64>,
    /// Cycle when the first participant arrived.
    first_arrival: Option<u64>,
    /// Cycle when the last participant arrived (barrier completed).
    completion_cycle: Option<u64>,
    /// Number of times this barrier has completed.
    completion_count: u64,
}

impl BarrierState {
    /// Create a new barrier state.
    pub fn new(config: BarrierConfig) -> Self {
        Self {
            config,
            phase: BarrierPhase::Waiting,
            arrived: HashSet::new(),
            arrival_times: HashMap::new(),
            first_arrival: None,
            completion_cycle: None,
            completion_count: 0,
        }
    }

    /// Get the barrier configuration.
    pub fn config(&self) -> &BarrierConfig {
        &self.config
    }

    /// Get the current phase.
    pub fn phase(&self) -> BarrierPhase {
        self.phase
    }

    /// Check if barrier is complete.
    pub fn is_complete(&self) -> bool {
        self.phase == BarrierPhase::Complete
    }

    /// Check if barrier is waiting for more participants.
    pub fn is_waiting(&self) -> bool {
        self.phase == BarrierPhase::Waiting
    }

    /// Get the number of participants that have arrived.
    pub fn arrived_count(&self) -> usize {
        self.arrived.len()
    }

    /// Get the number of participants still expected.
    pub fn remaining_count(&self) -> usize {
        self.config.participants as usize - self.arrived.len()
    }

    /// Check if a specific tile has arrived.
    pub fn has_arrived(&self, tile: TileId) -> bool {
        self.arrived.contains(&tile)
    }

    /// Get the arrival time for a tile.
    pub fn arrival_time(&self, tile: TileId) -> Option<u64> {
        self.arrival_times.get(&tile).copied()
    }

    /// Get the completion cycle (when last participant arrived).
    pub fn completion_cycle(&self) -> Option<u64> {
        self.completion_cycle
    }

    /// Calculate wait cycles for a specific tile.
    ///
    /// Returns the number of cycles between this tile's arrival and barrier completion.
    pub fn wait_cycles(&self, tile: TileId) -> Option<u64> {
        let arrival = self.arrival_times.get(&tile)?;
        let completion = self.completion_cycle?;
        Some(completion.saturating_sub(*arrival))
    }

    /// Get synchronization delay (time between first and last arrival).
    pub fn sync_delay(&self) -> Option<u64> {
        let first = self.first_arrival?;
        let last = self.completion_cycle?;
        Some(last - first)
    }

    /// Record a tile's arrival at the barrier.
    ///
    /// Returns true if this arrival completed the barrier.
    pub fn arrive(&mut self, tile: TileId, cycle: u64) -> bool {
        // Already complete or timed out - no more arrivals
        if self.phase != BarrierPhase::Waiting {
            return false;
        }

        // Already arrived
        if self.arrived.contains(&tile) {
            return false;
        }

        // Record arrival
        self.arrived.insert(tile);
        self.arrival_times.insert(tile, cycle);

        // Track first arrival
        if self.first_arrival.is_none() {
            self.first_arrival = Some(cycle);
        }

        // Check if all participants have arrived
        if self.arrived.len() >= self.config.participants as usize {
            self.phase = BarrierPhase::Complete;
            self.completion_cycle = Some(cycle);
            self.completion_count += 1;
            return true;
        }

        false
    }

    /// Check for timeout at the given cycle.
    ///
    /// Returns true if the barrier timed out.
    pub fn check_timeout(&mut self, current_cycle: u64) -> bool {
        if self.phase != BarrierPhase::Waiting {
            return false;
        }

        if self.config.timeout_cycles == 0 {
            return false;
        }

        if let Some(first) = self.first_arrival {
            if current_cycle - first >= self.config.timeout_cycles {
                self.phase = BarrierPhase::Timeout;
                return true;
            }
        }

        false
    }

    /// Reset the barrier for reuse.
    pub fn reset(&mut self) {
        self.phase = BarrierPhase::Waiting;
        self.arrived.clear();
        self.arrival_times.clear();
        self.first_arrival = None;
        self.completion_cycle = None;
        // Note: completion_count is not reset
    }

    /// Get the set of tiles that have arrived.
    pub fn arrived_tiles(&self) -> &HashSet<TileId> {
        &self.arrived
    }

    /// Get the number of times this barrier has completed.
    pub fn completion_count(&self) -> u64 {
        self.completion_count
    }
}

/// Statistics for barrier timing.
#[derive(Debug, Clone, Default)]
pub struct BarrierStats {
    /// Number of barrier completions.
    pub completions: u64,
    /// Number of barrier timeouts.
    pub timeouts: u64,
    /// Total wait cycles across all participants.
    pub total_wait_cycles: u64,
    /// Maximum wait cycles for any single participant.
    pub max_wait_cycles: u64,
    /// Maximum sync delay (first to last arrival).
    pub max_sync_delay: u64,
    /// Sum of sync delays for average calculation.
    total_sync_delay: u64,
}

impl BarrierStats {
    /// Calculate average wait cycles per completion.
    pub fn avg_wait_cycles(&self) -> f64 {
        if self.completions == 0 {
            0.0
        } else {
            self.total_wait_cycles as f64 / self.completions as f64
        }
    }

    /// Calculate average sync delay.
    pub fn avg_sync_delay(&self) -> f64 {
        if self.completions == 0 {
            0.0
        } else {
            self.total_sync_delay as f64 / self.completions as f64
        }
    }

    /// Record a barrier completion.
    fn record_completion(&mut self, state: &BarrierState) {
        self.completions += 1;

        // Sum wait cycles for all participants
        let completion = state.completion_cycle.unwrap_or(0);
        for &arrival in state.arrival_times.values() {
            let wait = completion.saturating_sub(arrival);
            self.total_wait_cycles += wait;
            self.max_wait_cycles = self.max_wait_cycles.max(wait);
        }

        // Track sync delay
        if let Some(delay) = state.sync_delay() {
            self.total_sync_delay += delay;
            self.max_sync_delay = self.max_sync_delay.max(delay);
        }
    }

    /// Record a barrier timeout.
    fn record_timeout(&mut self) {
        self.timeouts += 1;
    }
}

/// Tracker for multiple barriers across the device.
#[derive(Debug, Default)]
pub struct BarrierTracker {
    /// Active barriers by ID.
    barriers: HashMap<BarrierId, BarrierState>,
    /// Per-barrier statistics.
    stats: HashMap<BarrierId, BarrierStats>,
    /// Whether tracking is enabled.
    enabled: bool,
}

impl BarrierTracker {
    /// Create a new barrier tracker.
    pub fn new() -> Self {
        Self {
            barriers: HashMap::new(),
            stats: HashMap::new(),
            enabled: true,
        }
    }

    /// Create a disabled tracker (for fast simulation).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Check if tracking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable tracking.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Create a new barrier with the given configuration.
    pub fn create_barrier(&mut self, id: BarrierId, config: BarrierConfig) {
        if !self.enabled {
            return;
        }
        self.barriers.insert(id, BarrierState::new(config));
        self.stats.entry(id).or_default();
    }

    /// Remove a barrier.
    pub fn remove_barrier(&mut self, id: BarrierId) {
        self.barriers.remove(&id);
    }

    /// Record a tile's arrival at a barrier.
    ///
    /// Returns true if this arrival completed the barrier.
    pub fn arrive(&mut self, barrier_id: BarrierId, tile: TileId, cycle: u64) -> bool {
        if !self.enabled {
            return false;
        }

        let Some(state) = self.barriers.get_mut(&barrier_id) else {
            return false;
        };

        let completed = state.arrive(tile, cycle);

        if completed {
            // Record statistics
            if let Some(stats) = self.stats.get_mut(&barrier_id) {
                stats.record_completion(state);
            }

            // Auto-reset if configured
            if state.config.auto_reset {
                state.reset();
            }
        }

        completed
    }

    /// Check for timeouts on all barriers.
    ///
    /// Returns IDs of barriers that timed out.
    pub fn check_timeouts(&mut self, current_cycle: u64) -> Vec<BarrierId> {
        if !self.enabled {
            return Vec::new();
        }

        let mut timed_out = Vec::new();

        for (&id, state) in &mut self.barriers {
            if state.check_timeout(current_cycle) {
                timed_out.push(id);
                if let Some(stats) = self.stats.get_mut(&id) {
                    stats.record_timeout();
                }
            }
        }

        timed_out
    }

    /// Check if a barrier is complete.
    pub fn is_complete(&self, barrier_id: BarrierId) -> bool {
        self.barriers
            .get(&barrier_id)
            .map_or(false, |s| s.is_complete())
    }

    /// Check if a barrier is waiting for more participants.
    pub fn is_waiting(&self, barrier_id: BarrierId) -> bool {
        self.barriers
            .get(&barrier_id)
            .map_or(false, |s| s.is_waiting())
    }

    /// Get wait cycles for a tile at a barrier.
    pub fn wait_cycles(&self, barrier_id: BarrierId, tile: TileId) -> Option<u64> {
        self.barriers.get(&barrier_id)?.wait_cycles(tile)
    }

    /// Get the barrier state.
    pub fn barrier(&self, id: BarrierId) -> Option<&BarrierState> {
        self.barriers.get(&id)
    }

    /// Get mutable barrier state.
    pub fn barrier_mut(&mut self, id: BarrierId) -> Option<&mut BarrierState> {
        self.barriers.get_mut(&id)
    }

    /// Get statistics for a barrier.
    pub fn stats(&self, id: BarrierId) -> Option<&BarrierStats> {
        self.stats.get(&id)
    }

    /// Get the number of active barriers.
    pub fn barrier_count(&self) -> usize {
        self.barriers.len()
    }

    /// Get the number of barriers currently waiting.
    pub fn waiting_count(&self) -> usize {
        self.barriers.values().filter(|s| s.is_waiting()).count()
    }

    /// Reset all barriers.
    pub fn reset(&mut self) {
        for state in self.barriers.values_mut() {
            state.reset();
        }
    }

    /// Clear all barriers and statistics.
    pub fn clear(&mut self) {
        self.barriers.clear();
        self.stats.clear();
    }

    /// Get aggregate statistics across all barriers.
    pub fn aggregate_stats(&self) -> AggregateBarrierStats {
        let mut agg = AggregateBarrierStats::default();
        for stats in self.stats.values() {
            agg.total_completions += stats.completions;
            agg.total_timeouts += stats.timeouts;
            agg.total_wait_cycles += stats.total_wait_cycles;
            agg.max_wait_cycles = agg.max_wait_cycles.max(stats.max_wait_cycles);
            agg.max_sync_delay = agg.max_sync_delay.max(stats.max_sync_delay);
        }
        agg.barrier_count = self.stats.len();
        agg
    }
}

/// Aggregate statistics across all barriers.
#[derive(Debug, Clone, Default)]
pub struct AggregateBarrierStats {
    /// Number of barriers tracked.
    pub barrier_count: usize,
    /// Total completions across all barriers.
    pub total_completions: u64,
    /// Total timeouts across all barriers.
    pub total_timeouts: u64,
    /// Total wait cycles across all participants.
    pub total_wait_cycles: u64,
    /// Maximum wait cycles for any participant.
    pub max_wait_cycles: u64,
    /// Maximum sync delay observed.
    pub max_sync_delay: u64,
}

impl AggregateBarrierStats {
    /// Calculate completion rate.
    pub fn completion_rate(&self) -> f64 {
        let total = self.total_completions + self.total_timeouts;
        if total == 0 {
            1.0
        } else {
            self.total_completions as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_config() {
        let config = BarrierConfig::new(4);
        assert_eq!(config.participants, 4);
        assert_eq!(config.timeout_cycles, 0);
        assert!(config.auto_reset);

        let config = BarrierConfig::with_timeout(8, 1000);
        assert_eq!(config.participants, 8);
        assert_eq!(config.timeout_cycles, 1000);
    }

    #[test]
    fn test_barrier_state_basic() {
        let state = BarrierState::new(BarrierConfig::new(3));

        assert!(state.is_waiting());
        assert!(!state.is_complete());
        assert_eq!(state.arrived_count(), 0);
        assert_eq!(state.remaining_count(), 3);
    }

    #[test]
    fn test_barrier_arrivals() {
        let mut state = BarrierState::new(BarrierConfig::new(3));

        let tile1 = TileId::new(0, 2);
        let tile2 = TileId::new(1, 2);
        let tile3 = TileId::new(0, 3);

        // First arrival
        assert!(!state.arrive(tile1, 100));
        assert!(state.is_waiting());
        assert_eq!(state.arrived_count(), 1);
        assert!(state.has_arrived(tile1));
        assert_eq!(state.arrival_time(tile1), Some(100));

        // Second arrival
        assert!(!state.arrive(tile2, 110));
        assert!(state.is_waiting());
        assert_eq!(state.arrived_count(), 2);

        // Third arrival - completes the barrier
        assert!(state.arrive(tile3, 120));
        assert!(state.is_complete());
        assert_eq!(state.arrived_count(), 3);
        assert_eq!(state.completion_cycle(), Some(120));
    }

    #[test]
    fn test_barrier_wait_cycles() {
        let mut state = BarrierState::new(BarrierConfig::new(3));

        let tile1 = TileId::new(0, 2);
        let tile2 = TileId::new(1, 2);
        let tile3 = TileId::new(0, 3);

        state.arrive(tile1, 100);
        state.arrive(tile2, 115);
        state.arrive(tile3, 130);

        // Wait cycles = completion - arrival
        assert_eq!(state.wait_cycles(tile1), Some(30)); // 130 - 100
        assert_eq!(state.wait_cycles(tile2), Some(15)); // 130 - 115
        assert_eq!(state.wait_cycles(tile3), Some(0));  // 130 - 130 (last arrival)

        // Sync delay = last - first
        assert_eq!(state.sync_delay(), Some(30)); // 130 - 100
    }

    #[test]
    fn test_barrier_duplicate_arrival() {
        let mut state = BarrierState::new(BarrierConfig::new(2));

        let tile = TileId::new(0, 2);

        state.arrive(tile, 100);
        assert_eq!(state.arrived_count(), 1);

        // Duplicate arrival is ignored
        state.arrive(tile, 110);
        assert_eq!(state.arrived_count(), 1);
        assert_eq!(state.arrival_time(tile), Some(100)); // Original time preserved
    }

    #[test]
    fn test_barrier_reset() {
        let mut state = BarrierState::new(BarrierConfig::new(2));

        state.arrive(TileId::new(0, 2), 100);
        state.arrive(TileId::new(1, 2), 110);
        assert!(state.is_complete());
        assert_eq!(state.completion_count(), 1);

        state.reset();

        assert!(state.is_waiting());
        assert_eq!(state.arrived_count(), 0);
        assert_eq!(state.completion_count(), 1); // Preserved
    }

    #[test]
    fn test_barrier_timeout() {
        let mut state = BarrierState::new(BarrierConfig::with_timeout(3, 100));

        state.arrive(TileId::new(0, 2), 50);
        state.arrive(TileId::new(1, 2), 60);

        // Not timed out yet
        assert!(!state.check_timeout(100));

        // Timed out at cycle 150 (first arrival at 50, timeout after 100 cycles)
        assert!(state.check_timeout(151));
        assert_eq!(state.phase(), BarrierPhase::Timeout);
    }

    #[test]
    fn test_barrier_tracker() {
        let mut tracker = BarrierTracker::new();

        tracker.create_barrier(0, BarrierConfig::new(4));

        let tile1 = TileId::new(0, 2);
        let tile2 = TileId::new(1, 2);
        let tile3 = TileId::new(0, 3);
        let tile4 = TileId::new(1, 3);

        assert!(!tracker.arrive(0, tile1, 100));
        assert!(!tracker.arrive(0, tile2, 105));
        assert!(!tracker.arrive(0, tile3, 110));
        assert!(tracker.arrive(0, tile4, 120)); // Completes

        // Auto-reset is enabled, so barrier should be waiting again
        assert!(tracker.is_waiting(0));

        // Check statistics
        let stats = tracker.stats(0).unwrap();
        assert_eq!(stats.completions, 1);
        assert_eq!(stats.max_wait_cycles, 20); // tile1 waited 20 cycles
    }

    #[test]
    fn test_barrier_tracker_disabled() {
        let mut tracker = BarrierTracker::disabled();
        assert!(!tracker.is_enabled());

        tracker.create_barrier(0, BarrierConfig::new(2));
        tracker.arrive(0, TileId::new(0, 2), 100);

        // Nothing should be tracked
        assert_eq!(tracker.barrier_count(), 0);
    }

    #[test]
    fn test_barrier_stats() {
        let mut stats = BarrierStats::default();

        // Create a completed barrier state
        let mut state = BarrierState::new(BarrierConfig::new(2));
        let _ = state.arrive(TileId::new(0, 2), 100);
        let _ = state.arrive(TileId::new(1, 2), 130);

        stats.record_completion(&state);

        assert_eq!(stats.completions, 1);
        assert_eq!(stats.max_wait_cycles, 30);
        assert_eq!(stats.max_sync_delay, 30);
        // Total wait = 30 (tile1) + 0 (tile2) = 30
        assert_eq!(stats.total_wait_cycles, 30);
    }

    #[test]
    fn test_aggregate_stats() {
        let mut tracker = BarrierTracker::new();

        // Create two barriers
        tracker.create_barrier(0, BarrierConfig::new(2));
        tracker.create_barrier(1, BarrierConfig::new(2));

        // Complete both
        tracker.arrive(0, TileId::new(0, 2), 100);
        tracker.arrive(0, TileId::new(1, 2), 110);
        tracker.arrive(1, TileId::new(0, 3), 200);
        tracker.arrive(1, TileId::new(1, 3), 250);

        let agg = tracker.aggregate_stats();
        assert_eq!(agg.barrier_count, 2);
        assert_eq!(agg.total_completions, 2);
        assert_eq!(agg.max_sync_delay, 50); // barrier 1 had 50 cycle delay
    }

    #[test]
    fn test_check_timeouts() {
        let mut tracker = BarrierTracker::new();

        tracker.create_barrier(0, BarrierConfig::with_timeout(2, 100));
        tracker.create_barrier(1, BarrierConfig::with_timeout(2, 200));

        // Partial arrivals
        tracker.arrive(0, TileId::new(0, 2), 50);
        tracker.arrive(1, TileId::new(0, 3), 50);

        // Check at cycle 160 - barrier 0 should timeout (50 + 100 = 150)
        let timeouts = tracker.check_timeouts(160);
        assert_eq!(timeouts.len(), 1);
        assert!(timeouts.contains(&0));

        // Check at cycle 260 - barrier 1 should timeout (50 + 200 = 250)
        let timeouts = tracker.check_timeouts(260);
        assert_eq!(timeouts.len(), 1);
        assert!(timeouts.contains(&1));
    }

    #[test]
    fn test_completion_rate() {
        let agg = AggregateBarrierStats {
            barrier_count: 2,
            total_completions: 8,
            total_timeouts: 2,
            total_wait_cycles: 100,
            max_wait_cycles: 50,
            max_sync_delay: 30,
        };

        assert!((agg.completion_rate() - 0.8).abs() < 0.001);
    }
}
