//! Deadlock detection for AIE2 lock synchronization.
//!
//! This module detects potential deadlocks in lock acquisition patterns:
//!
//! - **Wait-for graph**: Tracks which tiles are waiting for which locks
//! - **Hold tracking**: Tracks which tiles have acquired locks (potential releasers)
//! - **Cycle detection**: Finds circular wait dependencies
//!
//! # AIE2 Lock Deadlock Model
//!
//! Deadlock occurs when there's a circular wait:
//! - Tile A waits for Lock L (needs release from Tile B)
//! - Tile B waits for Lock M (needs release from Tile A)
//! - Neither can proceed
//!
//! # Semaphore Considerations
//!
//! AIE2 uses semaphore locks (0-63 values), not binary locks. This means:
//! - Multiple tiles can "hold" the same lock (if they've acquired but not released)
//! - A waiting tile may need releases from multiple holders
//! - Detection is conservative: we flag potential deadlocks when cycles exist
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::interpreter::timing::deadlock::{DeadlockDetector, TileId};
//!
//! let mut detector = DeadlockDetector::new();
//!
//! // Tile (0,2) starts waiting for lock 5
//! detector.start_wait(TileId::new(0, 2), 5);
//!
//! // Tile (0,3) holds lock 5 and starts waiting for lock 7
//! detector.record_acquire(TileId::new(0, 3), 5);
//! detector.start_wait(TileId::new(0, 3), 7);
//!
//! // Tile (0,2) holds lock 7 - this creates a cycle!
//! detector.record_acquire(TileId::new(0, 2), 7);
//!
//! // Check for deadlock
//! if let Some(cycle) = detector.detect_deadlock() {
//!     println!("Deadlock detected! Cycle: {:?}", cycle);
//! }
//! ```

use std::collections::{HashMap, HashSet};

/// Tile identifier (column, row).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    pub col: u8,
    pub row: u8,
}

impl TileId {
    /// Create a new tile identifier.
    #[inline]
    pub const fn new(col: u8, row: u8) -> Self {
        Self { col, row }
    }
}

impl std::fmt::Display for TileId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.col, self.row)
    }
}

/// Lock identifier (tile + lock index).
///
/// Locks are per-tile, so we need both the tile location and lock index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LockId {
    /// Tile containing the lock
    pub tile: TileId,
    /// Lock index within the tile (0-15 for compute, 0-63 for mem tile)
    pub index: u8,
}

impl LockId {
    /// Create a new lock identifier.
    #[inline]
    pub const fn new(tile: TileId, index: u8) -> Self {
        Self { tile, index }
    }

    /// Create from raw coordinates.
    #[inline]
    pub const fn from_coords(col: u8, row: u8, index: u8) -> Self {
        Self {
            tile: TileId::new(col, row),
            index,
        }
    }
}

impl std::fmt::Display for LockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lock[{}:{}]", self.tile, self.index)
    }
}

/// A detected deadlock cycle.
#[derive(Debug, Clone)]
pub struct DeadlockCycle {
    /// Tiles involved in the cycle (in dependency order)
    pub tiles: Vec<TileId>,
    /// Locks involved in the cycle (each tile waits for the corresponding lock)
    pub locks: Vec<LockId>,
}

impl DeadlockCycle {
    /// Check if this cycle involves the given tile.
    pub fn involves(&self, tile: TileId) -> bool {
        self.tiles.contains(&tile)
    }

    /// Check if this cycle involves the given lock.
    pub fn involves_lock(&self, lock: LockId) -> bool {
        self.locks.contains(&lock)
    }
}

impl std::fmt::Display for DeadlockCycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Deadlock: ")?;
        for (i, (tile, lock)) in self.tiles.iter().zip(self.locks.iter()).enumerate() {
            if i > 0 {
                write!(f, " -> ")?;
            }
            write!(f, "{} waits for {}", tile, lock)?;
        }
        if !self.tiles.is_empty() {
            write!(f, " -> {}", self.tiles[0])?;
        }
        Ok(())
    }
}

/// Deadlock detector using wait-for graph analysis.
///
/// Tracks lock acquisition and waiting patterns across tiles to detect
/// potential deadlock situations.
#[derive(Debug, Default)]
pub struct DeadlockDetector {
    /// Which lock each tile is currently waiting for (if any)
    waiting_for: HashMap<TileId, LockId>,

    /// Which tiles have acquired each lock (potential releasers)
    /// Maps lock -> set of tiles that hold it
    lock_holders: HashMap<LockId, HashSet<TileId>>,

    /// Detected deadlocks (cached for inspection)
    detected_deadlocks: Vec<DeadlockCycle>,

    /// Whether detection is enabled
    enabled: bool,
}

impl DeadlockDetector {
    /// Create a new deadlock detector.
    pub fn new() -> Self {
        Self {
            waiting_for: HashMap::new(),
            lock_holders: HashMap::new(),
            detected_deadlocks: Vec::new(),
            enabled: true,
        }
    }

    /// Create a disabled detector (for fast simulation).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Check if detection is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable detection.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Record that a tile has started waiting for a lock.
    ///
    /// This is called when an acquire attempt fails and the tile will stall.
    pub fn start_wait(&mut self, tile: TileId, lock: LockId) {
        if !self.enabled {
            return;
        }
        self.waiting_for.insert(tile, lock);
    }

    /// Record that a tile has stopped waiting (acquire succeeded or was cancelled).
    pub fn end_wait(&mut self, tile: TileId) {
        if !self.enabled {
            return;
        }
        self.waiting_for.remove(&tile);
    }

    /// Record that a tile has successfully acquired a lock.
    ///
    /// The tile is added to the set of potential releasers for this lock.
    pub fn record_acquire(&mut self, tile: TileId, lock: LockId) {
        if !self.enabled {
            return;
        }
        // Remove from waiting (if it was waiting)
        self.waiting_for.remove(&tile);

        // Add to holders
        self.lock_holders.entry(lock).or_default().insert(tile);
    }

    /// Record that a tile has released a lock.
    ///
    /// The tile is removed from the set of holders for this lock.
    pub fn record_release(&mut self, tile: TileId, lock: LockId) {
        if !self.enabled {
            return;
        }
        if let Some(holders) = self.lock_holders.get_mut(&lock) {
            holders.remove(&tile);
            if holders.is_empty() {
                self.lock_holders.remove(&lock);
            }
        }
    }

    /// Detect deadlocks in the current wait-for graph.
    ///
    /// Returns the first deadlock cycle found, or None if no deadlock exists.
    /// For semaphore locks, a deadlock is detected when:
    /// - Tile A waits for lock L
    /// - All holders of L are themselves waiting for locks
    /// - Those locks are held by tiles that (transitively) depend on A
    pub fn detect_deadlock(&mut self) -> Option<&DeadlockCycle> {
        if !self.enabled || self.waiting_for.is_empty() {
            return None;
        }

        self.detected_deadlocks.clear();

        // For each waiting tile, try to find a cycle
        for &start_tile in self.waiting_for.keys() {
            if let Some(cycle) = self.find_cycle_from(start_tile) {
                self.detected_deadlocks.push(cycle);
                return self.detected_deadlocks.last();
            }
        }

        None
    }

    /// Find a cycle starting from the given tile using DFS.
    fn find_cycle_from(&self, start: TileId) -> Option<DeadlockCycle> {
        let mut visited = HashSet::new();
        let mut path_tiles = Vec::new();
        let mut path_locks = Vec::new();

        self.dfs_cycle(start, start, &mut visited, &mut path_tiles, &mut path_locks)
    }

    /// DFS to find a cycle back to the start tile.
    fn dfs_cycle(
        &self,
        current: TileId,
        start: TileId,
        visited: &mut HashSet<TileId>,
        path_tiles: &mut Vec<TileId>,
        path_locks: &mut Vec<LockId>,
    ) -> Option<DeadlockCycle> {
        // What lock is this tile waiting for?
        let waiting_lock = self.waiting_for.get(&current)?;

        // Add current to path
        path_tiles.push(current);
        path_locks.push(*waiting_lock);

        // Who holds this lock?
        if let Some(holders) = self.lock_holders.get(waiting_lock) {
            for &holder in holders {
                // Found a cycle back to start!
                if holder == start && path_tiles.len() > 1 {
                    return Some(DeadlockCycle {
                        tiles: path_tiles.clone(),
                        locks: path_locks.clone(),
                    });
                }

                // Continue DFS if not visited and holder is also waiting
                if !visited.contains(&holder) && self.waiting_for.contains_key(&holder) {
                    visited.insert(holder);
                    if let Some(cycle) =
                        self.dfs_cycle(holder, start, visited, path_tiles, path_locks)
                    {
                        return Some(cycle);
                    }
                    visited.remove(&holder);
                }
            }
        }

        // Backtrack
        path_tiles.pop();
        path_locks.pop();

        None
    }

    /// Check if any tiles are currently waiting.
    pub fn has_waiting_tiles(&self) -> bool {
        !self.waiting_for.is_empty()
    }

    /// Get the number of currently waiting tiles.
    pub fn waiting_count(&self) -> usize {
        self.waiting_for.len()
    }

    /// Get which lock a tile is waiting for (if any).
    pub fn waiting_for(&self, tile: TileId) -> Option<LockId> {
        self.waiting_for.get(&tile).copied()
    }

    /// Get the tiles holding a specific lock.
    pub fn lock_holders(&self, lock: LockId) -> Option<&HashSet<TileId>> {
        self.lock_holders.get(&lock)
    }

    /// Get all previously detected deadlocks.
    pub fn detected_deadlocks(&self) -> &[DeadlockCycle] {
        &self.detected_deadlocks
    }

    /// Clear all tracking state (but keep enabled status).
    pub fn reset(&mut self) {
        self.waiting_for.clear();
        self.lock_holders.clear();
        self.detected_deadlocks.clear();
    }

    /// Get a summary of the current state for debugging.
    pub fn summary(&self) -> DeadlockSummary {
        DeadlockSummary {
            enabled: self.enabled,
            waiting_tiles: self.waiting_for.len(),
            held_locks: self.lock_holders.len(),
            detected_deadlocks: self.detected_deadlocks.len(),
        }
    }
}

/// Summary of deadlock detector state.
#[derive(Debug, Clone)]
pub struct DeadlockSummary {
    /// Whether detection is enabled
    pub enabled: bool,
    /// Number of tiles currently waiting
    pub waiting_tiles: usize,
    /// Number of locks currently held
    pub held_locks: usize,
    /// Number of deadlocks detected
    pub detected_deadlocks: usize,
}

/// Configuration for deadlock detection behavior.
#[derive(Debug, Clone, Copy)]
pub struct DeadlockConfig {
    /// Enable deadlock detection
    pub enabled: bool,
    /// Maximum cycle length to detect (0 = unlimited)
    pub max_cycle_length: usize,
    /// Check for deadlocks on every acquire attempt (expensive)
    pub check_on_acquire: bool,
}

impl Default for DeadlockConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cycle_length: 16, // Reasonable limit for AIE2 arrays
            check_on_acquire: false, // Only check on explicit request
        }
    }
}

impl DeadlockConfig {
    /// Create a disabled configuration for fast simulation.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            max_cycle_length: 0,
            check_on_acquire: false,
        }
    }

    /// Create an aggressive configuration that checks on every acquire.
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            max_cycle_length: 0, // Unlimited
            check_on_acquire: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_id() {
        let tile = TileId::new(2, 3);
        assert_eq!(tile.col, 2);
        assert_eq!(tile.row, 3);
        assert_eq!(format!("{}", tile), "(2,3)");
    }

    #[test]
    fn test_lock_id() {
        let lock = LockId::from_coords(1, 2, 5);
        assert_eq!(lock.tile, TileId::new(1, 2));
        assert_eq!(lock.index, 5);
        assert_eq!(format!("{}", lock), "Lock[(1,2):5]");
    }

    #[test]
    fn test_no_deadlock_single_waiter() {
        let mut detector = DeadlockDetector::new();

        // Tile (0,2) waits for lock 5
        let tile = TileId::new(0, 2);
        let lock = LockId::from_coords(0, 2, 5);

        detector.start_wait(tile, lock);

        // No deadlock - no holders, just a waiter
        assert!(detector.detect_deadlock().is_none());
    }

    #[test]
    fn test_no_deadlock_holder_not_waiting() {
        let mut detector = DeadlockDetector::new();

        let tile_a = TileId::new(0, 2);
        let tile_b = TileId::new(0, 3);
        let lock = LockId::from_coords(0, 2, 5);

        // Tile B holds lock 5
        detector.record_acquire(tile_b, lock);

        // Tile A waits for lock 5
        detector.start_wait(tile_a, lock);

        // No deadlock - B is not waiting for anything
        assert!(detector.detect_deadlock().is_none());
    }

    #[test]
    fn test_simple_deadlock() {
        let mut detector = DeadlockDetector::new();

        let tile_a = TileId::new(0, 2);
        let tile_b = TileId::new(0, 3);
        let lock_l = LockId::from_coords(0, 2, 5);
        let lock_m = LockId::from_coords(0, 2, 7);

        // Tile A holds lock M
        detector.record_acquire(tile_a, lock_m);

        // Tile B holds lock L
        detector.record_acquire(tile_b, lock_l);

        // Tile A waits for lock L (held by B)
        detector.start_wait(tile_a, lock_l);

        // Tile B waits for lock M (held by A) - DEADLOCK!
        detector.start_wait(tile_b, lock_m);

        let cycle = detector.detect_deadlock();
        assert!(cycle.is_some());

        let cycle = cycle.unwrap();
        assert_eq!(cycle.tiles.len(), 2);
        assert!(cycle.involves(tile_a));
        assert!(cycle.involves(tile_b));
    }

    #[test]
    fn test_three_way_deadlock() {
        let mut detector = DeadlockDetector::new();

        let tile_a = TileId::new(0, 2);
        let tile_b = TileId::new(0, 3);
        let tile_c = TileId::new(0, 4);
        let lock_l = LockId::from_coords(0, 2, 0);
        let lock_m = LockId::from_coords(0, 2, 1);
        let lock_n = LockId::from_coords(0, 2, 2);

        // A holds L, waits for M
        detector.record_acquire(tile_a, lock_l);
        detector.start_wait(tile_a, lock_m);

        // B holds M, waits for N
        detector.record_acquire(tile_b, lock_m);
        detector.start_wait(tile_b, lock_n);

        // C holds N, waits for L - DEADLOCK!
        detector.record_acquire(tile_c, lock_n);
        detector.start_wait(tile_c, lock_l);

        let cycle = detector.detect_deadlock();
        assert!(cycle.is_some());

        let cycle = cycle.unwrap();
        assert_eq!(cycle.tiles.len(), 3);
        assert!(cycle.involves(tile_a));
        assert!(cycle.involves(tile_b));
        assert!(cycle.involves(tile_c));
    }

    #[test]
    fn test_deadlock_resolved_by_release() {
        let mut detector = DeadlockDetector::new();

        let tile_a = TileId::new(0, 2);
        let tile_b = TileId::new(0, 3);
        let lock_l = LockId::from_coords(0, 2, 5);
        let lock_m = LockId::from_coords(0, 2, 7);

        // Create deadlock
        detector.record_acquire(tile_a, lock_m);
        detector.record_acquire(tile_b, lock_l);
        detector.start_wait(tile_a, lock_l);
        detector.start_wait(tile_b, lock_m);

        assert!(detector.detect_deadlock().is_some());

        // B releases lock L
        detector.record_release(tile_b, lock_l);
        detector.end_wait(tile_a); // A can now acquire L

        // No more deadlock
        assert!(detector.detect_deadlock().is_none());
    }

    #[test]
    fn test_disabled_detector() {
        let mut detector = DeadlockDetector::disabled();
        assert!(!detector.is_enabled());

        let tile = TileId::new(0, 2);
        let lock = LockId::from_coords(0, 2, 5);

        detector.start_wait(tile, lock);
        detector.record_acquire(tile, lock);

        // Nothing tracked when disabled
        assert_eq!(detector.waiting_count(), 0);
    }

    #[test]
    fn test_summary() {
        let mut detector = DeadlockDetector::new();

        let tile_a = TileId::new(0, 2);
        let lock = LockId::from_coords(0, 2, 5);

        detector.record_acquire(tile_a, lock);
        detector.start_wait(TileId::new(0, 3), lock);

        let summary = detector.summary();
        assert!(summary.enabled);
        assert_eq!(summary.waiting_tiles, 1);
        assert_eq!(summary.held_locks, 1);
    }

    #[test]
    fn test_reset() {
        let mut detector = DeadlockDetector::new();

        let tile = TileId::new(0, 2);
        let lock = LockId::from_coords(0, 2, 5);

        detector.record_acquire(tile, lock);
        detector.start_wait(tile, lock);

        detector.reset();

        assert_eq!(detector.waiting_count(), 0);
        assert!(detector.lock_holders(lock).is_none());
    }

    #[test]
    fn test_deadlock_cycle_display() {
        let cycle = DeadlockCycle {
            tiles: vec![TileId::new(0, 2), TileId::new(0, 3)],
            locks: vec![
                LockId::from_coords(0, 2, 5),
                LockId::from_coords(0, 2, 7),
            ],
        };

        let display = format!("{}", cycle);
        assert!(display.contains("Deadlock"));
        assert!(display.contains("(0,2)"));
        assert!(display.contains("(0,3)"));
    }
}
