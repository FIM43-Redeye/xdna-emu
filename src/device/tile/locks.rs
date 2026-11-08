//! Lock system: semaphore locks and round-robin arbiter.

/// Result of a lock operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockResult {
    /// Operation granted.
    Success,
    /// Acquire refused: the count does not meet the acquire's condition.
    PreconditionNotMet,
    /// Refused because the lock's overflow or underflow bit is set: a
    /// flagged lock grants nothing until the bit is cleared.
    Frozen,
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
    pub expected: u8,
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

                // Process releases first: they are non-blocking and succeed
                // unless the lock is frozen. A release must not prevent a
                // same-cycle acquire from seeing the updated value.
                for &(idx, _) in &group_with_dist {
                    let req = &self.pending[idx];
                    if !req.is_acquire {
                        // A release is refused only by a frozen lock.
                        let granted = Self::try_apply(req, locks);
                        if lock_id < self.stats.len() {
                            self.stats[lock_id].grants += granted as u64;
                        }
                        self.results.push((req.requestor, lock_id, granted, false));
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
        let Some(lock) = locks.get_mut(req.lock_id) else {
            return false;
        };
        let result = if !req.is_acquire {
            lock.release_with_value(req.delta)
        } else if req.equal_mode {
            lock.acquire_equal(req.expected, req.delta)
        } else {
            lock.acquire_with_value(req.expected, req.delta)
        };
        result == LockResult::Success
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
/// An AIE2 lock is a 6-bit unsigned count, 0..=63, plus sticky overflow and
/// underflow bits. Acquires and releases carry a 7-bit signed change value
/// (-64..=63). Silicon (lock request probe, 2026-11-08, compute and memtile
/// locks alike):
///
/// - An acquire with a negative change c is granted iff count >= -c, and
///   then subtracts -c. An acquire with c >= 0 is granted iff count == c
///   and leaves the count unchanged. A refused acquire changes nothing and
///   sets no bit.
/// - A release adds its change modulo 64 and is granted even when it wraps;
///   wrapping above 63 sets the overflow bit, below 0 the underflow bit.
/// - While either bit is set the lock refuses every acquire and release. A
///   Lock_value write still lands and leaves the bits alone; only a
///   write-1-to-clear of the bit frees the lock.
///
/// The acquire-with-c>=0 rule was measured on the host Lock_Request path;
/// the DMA and core paths still pass their own (expected, delta) pairs.
///
/// The Lock_Request register format (AM025):
/// - Lock_Id [13:10]: Which lock (0-15 for compute, 0-63 for mem tile)
/// - Acq_Rel [9]: 1 = acquire (blocking), 0 = release (non-blocking)
/// - Change_Value [8:2]: Signed 7-bit delta (-64 to +63)
/// - Request_Result [0]: 0 = failed, 1 = succeeded
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Lock {
    /// Current count, 0..=MAX_VALUE.
    pub value: u8,
    /// Overflow bit: a release wrapped the count past MAX_VALUE.
    pub overflow: bool,
    /// Underflow bit: a release wrapped the count below 0.
    pub underflow: bool,
}

impl Lock {
    /// Largest count (the 6-bit Lock_value field, mlir-aie
    /// `getMaxLockValue()`).
    ///
    /// Kept as a const for the hot path; a test checks it against the
    /// archspec value layout.
    pub const MAX_VALUE: u8 = 63;

    /// Counts are taken modulo this (the field is `MAX_VALUE + 1` wide).
    const MODULUS: i16 = Self::MAX_VALUE as i16 + 1;

    /// Create a lock holding `value`'s low 6 bits (what a Lock_value write
    /// stores).
    #[inline]
    pub fn new(value: u8) -> Self {
        Self { value: value & Self::MAX_VALUE, overflow: false, underflow: false }
    }

    /// Whether the lock refuses requests: either sticky bit is set.
    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.overflow || self.underflow
    }

    /// Acquire one (equivalent to `acquire_with_value(1, -1)`).
    #[inline]
    pub fn acquire(&mut self) -> bool {
        self.acquire_with_value(1, -1) == LockResult::Success
    }

    /// Release one (equivalent to `release_with_value(1)`).
    #[inline]
    pub fn release(&mut self) {
        self.release_with_value(1);
    }

    /// Greater-or-equal acquire: granted iff the count is at least
    /// `expected`, then `delta` (normally `-expected`) is applied.
    ///
    /// A delta that would take the count below 0 is refused like an unmet
    /// condition; the hardware encodings (one change value) never ask for
    /// one.
    #[inline]
    pub fn acquire_with_value(&mut self, expected: u8, delta: i8) -> LockResult {
        if self.is_frozen() {
            return LockResult::Frozen;
        }
        let new_value = self.value as i16 + delta as i16;
        if self.value < expected || !(0..Self::MODULUS).contains(&new_value) {
            return LockResult::PreconditionNotMet;
        }
        self.value = new_value as u8;
        LockResult::Success
    }

    /// Equal acquire: granted iff the count is exactly `expected`, then
    /// `delta` is applied (0 on the Lock_Request path, where silicon leaves
    /// the count unchanged).
    #[inline]
    pub fn acquire_equal(&mut self, expected: u8, delta: i8) -> LockResult {
        if self.is_frozen() {
            return LockResult::Frozen;
        }
        let new_value = self.value as i16 + delta as i16;
        if self.value != expected || !(0..Self::MODULUS).contains(&new_value) {
            return LockResult::PreconditionNotMet;
        }
        self.value = new_value as u8;
        LockResult::Success
    }

    /// Release: add `delta` modulo 64, setting the overflow bit when the sum
    /// passes MAX_VALUE and the underflow bit when it drops below 0. Granted
    /// unless the lock is frozen.
    #[inline]
    pub fn release_with_value(&mut self, delta: i8) -> LockResult {
        if self.is_frozen() {
            return LockResult::Frozen;
        }
        let sum = self.value as i16 + delta as i16;
        self.overflow = sum >= Self::MODULUS;
        self.underflow = sum < 0;
        self.value = sum.rem_euclid(Self::MODULUS) as u8;
        LockResult::Success
    }

    /// Write the count (a Lock_value register write): keeps the low 6 bits
    /// and leaves the sticky bits alone, as silicon does.
    #[inline]
    pub fn set(&mut self, value: u8) {
        self.value = value & Self::MAX_VALUE;
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
        self.is_frozen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_bounds_match_archspec() {
        use xdna_archspec::aie2::locks::AIE2_LOCK_VALUE_LAYOUT;
        assert_eq!(
            Lock::MAX_VALUE,
            AIE2_LOCK_VALUE_LAYOUT.max(),
            "Lock::MAX_VALUE must match the AIE2 Lock_value field"
        );
    }
}
