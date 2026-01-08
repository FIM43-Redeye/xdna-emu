//! Memory tile arbitration for multi-core access.
//!
//! This module models contention when multiple compute tiles and DMA channels
//! access the same memory tile simultaneously.
//!
//! # AIE2 Memory Tile Access (AM020 Ch2)
//!
//! Memory tiles can be accessed by:
//! - Up to 4 compute tiles in the same column
//! - 6 S2MM + 6 MM2S DMA channels
//!
//! When multiple sources request access in the same cycle, round-robin
//! arbitration applies stall cycles to waiting requesters.
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::interpreter::timing::arbitration::{MemTileArbiter, ArbiterSource};
//!
//! let mut arbiter = MemTileArbiter::new(0); // Column 0
//!
//! // Two compute tiles request access in same cycle
//! let stall1 = arbiter.request(ArbiterSource::ComputeTile { row: 2 }, 100);
//! let stall2 = arbiter.request(ArbiterSource::ComputeTile { row: 3 }, 100);
//!
//! // First gets through immediately, second waits
//! assert_eq!(stall1, 0);
//! assert!(stall2 > 0);
//! ```

/// Source of a memory tile access request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbiterSource {
    /// Request from a compute tile (row 2-5 typically).
    ComputeTile {
        /// Row of the compute tile.
        row: u8,
    },
    /// Request from a DMA channel.
    DmaChannel {
        /// DMA channel number.
        channel: u8,
    },
}

impl ArbiterSource {
    /// Get a unique identifier for this source (for round-robin ordering).
    fn id(&self) -> u8 {
        match self {
            // Compute tiles: IDs 0-3 (rows 2-5)
            Self::ComputeTile { row } => row.saturating_sub(2).min(3),
            // DMA channels: IDs 4-15 (channels 0-11)
            Self::DmaChannel { channel } => 4 + (*channel).min(11),
        }
    }
}

/// Pending request in the arbitration queue.
#[derive(Debug, Clone, Copy)]
struct PendingRequest {
    /// Source of the request.
    source: ArbiterSource,
    /// Cycle when request was made (reserved for latency tracking).
    _request_cycle: u64,
}

/// Memory tile arbiter for a single column.
///
/// Implements round-robin arbitration when multiple sources request
/// access to the memory tile in the same cycle.
#[derive(Debug)]
pub struct MemTileArbiter {
    /// Column of the memory tile.
    col: u8,
    /// Pending requests for current cycle.
    pending: Vec<PendingRequest>,
    /// Last granted source ID (for round-robin fairness).
    last_granted: Option<u8>,
    /// Current arbitration cycle.
    current_cycle: u64,
    /// Total contention cycles accumulated.
    contention_cycles: u64,
    /// Number of contentions (multiple requests in same cycle).
    contention_count: u64,
}

impl MemTileArbiter {
    /// Create a new arbiter for the given column.
    pub fn new(col: u8) -> Self {
        Self {
            col,
            pending: Vec::with_capacity(16), // Max: 4 compute + 12 DMA
            last_granted: None,
            current_cycle: 0,
            contention_cycles: 0,
            contention_count: 0,
        }
    }

    /// Get the column this arbiter manages.
    pub fn col(&self) -> u8 {
        self.col
    }

    /// Request access to the memory tile.
    ///
    /// Returns the number of stall cycles before access is granted.
    /// A return of 0 means immediate access.
    pub fn request(&mut self, source: ArbiterSource, cycle: u64) -> u64 {
        // New cycle - clear pending requests from previous cycle
        if cycle > self.current_cycle {
            self.resolve_cycle();
            self.current_cycle = cycle;
        }

        // Add this request
        self.pending.push(PendingRequest {
            source,
            _request_cycle: cycle,
        });

        // If this is the only request, grant immediately
        if self.pending.len() == 1 {
            return 0;
        }

        // Multiple requests - contention!
        self.contention_count += 1;

        // Calculate stall based on round-robin position
        let source_id = source.id();
        let position = self.round_robin_position(source_id);

        // Each position adds 1 stall cycle
        let stall = position as u64;
        self.contention_cycles += stall;

        stall
    }

    /// Calculate round-robin position for a source.
    ///
    /// Position 0 = granted this cycle, position 1 = wait 1 cycle, etc.
    fn round_robin_position(&self, source_id: u8) -> u8 {
        // If no previous grant, use simple ordering
        let last = match self.last_granted {
            Some(l) => l,
            None => return self.pending.iter().position(|r| r.source.id() == source_id).unwrap_or(0) as u8,
        };

        // Sort pending by round-robin order starting after last_granted
        let mut sorted_ids: Vec<u8> = self.pending.iter().map(|r| r.source.id()).collect();
        sorted_ids.sort_by_key(|&id| {
            // Distance from last_granted in round-robin order (using wrapping arithmetic)
            if id > last {
                (id - last) as u16
            } else {
                (id as u16) + 16 - (last as u16)
            }
        });

        // Find position of this source
        sorted_ids.iter().position(|&id| id == source_id).unwrap_or(0) as u8
    }

    /// Resolve pending requests for the current cycle.
    fn resolve_cycle(&mut self) {
        if !self.pending.is_empty() {
            // Grant to the first in round-robin order
            let mut min_pos = u8::MAX;
            let mut granted_id = 0u8;

            for req in &self.pending {
                let pos = self.round_robin_position(req.source.id());
                if pos < min_pos {
                    min_pos = pos;
                    granted_id = req.source.id();
                }
            }

            self.last_granted = Some(granted_id);
            self.pending.clear();
        }
    }

    /// Advance to a new cycle.
    pub fn advance_to(&mut self, cycle: u64) {
        if cycle > self.current_cycle {
            self.resolve_cycle();
            self.current_cycle = cycle;
        }
    }

    /// Get total contention cycles accumulated.
    pub fn contention_cycles(&self) -> u64 {
        self.contention_cycles
    }

    /// Get number of contention events (cycles with multiple requests).
    pub fn contention_count(&self) -> u64 {
        self.contention_count
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.contention_cycles = 0;
        self.contention_count = 0;
    }
}

/// Aggregate statistics for all arbiters.
#[derive(Debug, Clone, Default)]
pub struct ArbiterStats {
    /// Total contention cycles across all columns.
    pub total_contention_cycles: u64,
    /// Total contention events.
    pub total_contention_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arbiter_creation() {
        let arbiter = MemTileArbiter::new(0);
        assert_eq!(arbiter.col(), 0);
        assert_eq!(arbiter.contention_cycles(), 0);
        assert_eq!(arbiter.contention_count(), 0);
    }

    #[test]
    fn test_single_request_no_contention() {
        let mut arbiter = MemTileArbiter::new(0);

        // Single request - immediate grant
        let stall = arbiter.request(ArbiterSource::ComputeTile { row: 2 }, 100);
        assert_eq!(stall, 0);
        assert_eq!(arbiter.contention_count(), 0);
    }

    #[test]
    fn test_multiple_requests_contention() {
        let mut arbiter = MemTileArbiter::new(0);

        // First request - immediate
        let stall1 = arbiter.request(ArbiterSource::ComputeTile { row: 2 }, 100);
        assert_eq!(stall1, 0);

        // Second request same cycle - contention
        let stall2 = arbiter.request(ArbiterSource::ComputeTile { row: 3 }, 100);
        assert!(stall2 > 0, "Second request should have stall");
        assert!(arbiter.contention_count() >= 1);
    }

    #[test]
    fn test_round_robin_fairness() {
        let mut arbiter = MemTileArbiter::new(0);

        // Cycle 100: Row 2 and 3 request
        let s1 = arbiter.request(ArbiterSource::ComputeTile { row: 2 }, 100);
        let s2 = arbiter.request(ArbiterSource::ComputeTile { row: 3 }, 100);

        // Resolve and advance
        arbiter.advance_to(101);

        // Cycle 101: Row 2 and 3 request again
        // Row 3 should now have priority (round-robin)
        let s3 = arbiter.request(ArbiterSource::ComputeTile { row: 2 }, 101);
        let s4 = arbiter.request(ArbiterSource::ComputeTile { row: 3 }, 101);

        // Verify fairness - row 3 should get lower stall in second round
        assert!(s1 <= s2, "First round: earlier request should win");
        assert!(s4 <= s3, "Second round: round-robin should favor row 3");
    }

    #[test]
    fn test_dma_channel_arbitration() {
        let mut arbiter = MemTileArbiter::new(0);

        // Compute tile vs DMA channel
        let stall1 = arbiter.request(ArbiterSource::ComputeTile { row: 2 }, 100);
        let stall2 = arbiter.request(ArbiterSource::DmaChannel { channel: 0 }, 100);

        // One gets through, one waits
        assert!(stall1 == 0 || stall2 == 0, "One should get immediate access");
        assert!(stall1 > 0 || stall2 > 0, "One should have to wait");
    }

    #[test]
    fn test_new_cycle_clears_pending() {
        let mut arbiter = MemTileArbiter::new(0);

        // Cycle 100
        arbiter.request(ArbiterSource::ComputeTile { row: 2 }, 100);
        arbiter.request(ArbiterSource::ComputeTile { row: 3 }, 100);

        // Advance to new cycle - pending should clear
        arbiter.advance_to(101);

        // New request should be immediate
        let stall = arbiter.request(ArbiterSource::ComputeTile { row: 4 }, 101);
        assert_eq!(stall, 0);
    }

    #[test]
    fn test_contention_stats() {
        let mut arbiter = MemTileArbiter::new(0);

        // Create several contentions
        for cycle in 0..5 {
            arbiter.request(ArbiterSource::ComputeTile { row: 2 }, cycle);
            arbiter.request(ArbiterSource::ComputeTile { row: 3 }, cycle);
            arbiter.advance_to(cycle + 1);
        }

        // Should have recorded contentions
        assert!(arbiter.contention_count() >= 5);
        assert!(arbiter.contention_cycles() > 0);
    }

    #[test]
    fn test_source_id() {
        // Compute tiles get IDs 0-3
        assert_eq!(ArbiterSource::ComputeTile { row: 2 }.id(), 0);
        assert_eq!(ArbiterSource::ComputeTile { row: 3 }.id(), 1);
        assert_eq!(ArbiterSource::ComputeTile { row: 4 }.id(), 2);
        assert_eq!(ArbiterSource::ComputeTile { row: 5 }.id(), 3);

        // DMA channels get IDs 4-15
        assert_eq!(ArbiterSource::DmaChannel { channel: 0 }.id(), 4);
        assert_eq!(ArbiterSource::DmaChannel { channel: 5 }.id(), 9);
        assert_eq!(ArbiterSource::DmaChannel { channel: 11 }.id(), 15);
    }

    #[test]
    fn test_reset_stats() {
        let mut arbiter = MemTileArbiter::new(0);

        // Create some contention
        arbiter.request(ArbiterSource::ComputeTile { row: 2 }, 0);
        arbiter.request(ArbiterSource::ComputeTile { row: 3 }, 0);

        assert!(arbiter.contention_count() > 0);

        // Reset
        arbiter.reset_stats();

        assert_eq!(arbiter.contention_cycles(), 0);
        assert_eq!(arbiter.contention_count(), 0);
    }
}
