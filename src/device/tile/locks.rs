//! Lock system: semaphore locks and round-robin arbiter.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_bounds_match_archspec() {
        use xdna_archspec::aie2::locks::AIE2_LOCK_VALUE_LAYOUT;
        assert_eq!(Lock::MIN_VALUE, AIE2_LOCK_VALUE_LAYOUT.min,
                   "Lock::MIN_VALUE must match AIE2_LOCK_VALUE_LAYOUT.min; \
                    update the constant (or split per-arch) when AIE1 lands");
        assert_eq!(Lock::MAX_VALUE, AIE2_LOCK_VALUE_LAYOUT.max,
                   "Lock::MAX_VALUE must match AIE2_LOCK_VALUE_LAYOUT.max");
    }
}
