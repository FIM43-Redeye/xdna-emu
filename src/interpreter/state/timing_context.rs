//! Timing context for cycle-accurate AIE2 execution.
//!
//! Contains `TimingContext` (hazard detection, memory model, event tracing),
//! `PendingBranch` (delay slot modeling), and `SrsConfig` (vector control
//! register state for SRS/UPS operations).

use super::event_trace::{EventLog, EventType};
use crate::interpreter::timing::{HazardDetector, LatencyTable, MemoryModel};

// ============================================================================
// Timing Context
// ============================================================================

/// Timing context for cycle-accurate execution.
///
/// Bundles all timing-related state into one structure that can be
/// optionally attached to an `ExecutionContext` for accurate cycle counting.
#[derive(Clone)]
pub struct TimingContext {
    /// Register hazard detector (RAW, WAW, WAR).
    pub hazards: HazardDetector,

    /// Memory bank conflict detector.
    pub memory: MemoryModel,

    /// Operation latency lookup table.
    // TODO(subsys7-followup): Use arch_handle::latency_table() instead of owning a copy.
    // See NEXT-STEPS.md or docs/arch/subsys7-audit.md Completion section.
    pub latencies: LatencyTable,

    /// Total hazard stall cycles.
    pub hazard_stalls: u64,

    /// Total memory conflict stall cycles.
    pub memory_stalls: u64,

    /// Event log for profiling/tracing.
    pub events: EventLog,
}

impl TimingContext {
    /// Create a new timing context with AIE2 defaults.
    pub fn new() -> Self {
        Self {
            hazards: HazardDetector::new(),
            memory: MemoryModel::new(),
            latencies: LatencyTable::aie2(),
            hazard_stalls: 0,
            memory_stalls: 0,
            events: EventLog::new(),
        }
    }

    /// Advance all timing models to the given cycle.
    pub fn advance_to(&mut self, cycle: u64) {
        self.hazards.advance_to(cycle);
        self.memory.advance_to(cycle);
    }

    /// Reset timing state but keep latency table.
    pub fn reset(&mut self) {
        self.hazards.reset();
        self.memory.reset();
        self.hazard_stalls = 0;
        self.memory_stalls = 0;
        self.events.clear();
    }

    /// Enable event tracing.
    pub fn enable_tracing(&mut self) {
        self.events.enable();
    }

    /// Disable event tracing.
    pub fn disable_tracing(&mut self) {
        self.events.disable();
    }

    /// Record an event at the given cycle.
    #[inline]
    pub fn record_event(&mut self, cycle: u64, event: EventType) {
        self.events.record(cycle, event);
    }

    /// Get combined timing statistics.
    pub fn total_stall_cycles(&self) -> u64 {
        self.hazard_stalls + self.memory_stalls
    }
}

impl Default for TimingContext {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TimingContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimingContext")
            .field("hazard_stalls", &self.hazard_stalls)
            .field("memory_stalls", &self.memory_stalls)
            .finish()
    }
}

// ============================================================================
// Branch Delay Slots
// ============================================================================

/// Pending branch for delay slot handling.
///
/// AIE2 has 5-cycle branch delay slots - after a branch instruction,
/// the next 5 instructions still execute before the branch takes effect.
///
/// The delay_slots counter starts at 6 because `tick()` is called on the
/// same cycle as the branch instruction itself (in the interpreter loop).
/// The first tick brings it to 5, then 5 subsequent instruction cycles
/// bring it to 0, giving exactly 5 executed delay slot instructions.
///
/// For `jl` (call) instructions, `is_call` is set and LR is updated
/// WHEN the delay slots are exhausted (not immediately). This matches
/// hardware behavior where delay slot instructions see the pre-call LR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingBranch {
    /// The target address to branch to.
    pub target: u32,
    /// Number of delay slots remaining.
    /// Starts at BRANCH_DELAY_INITIAL: first tick (on branch cycle) -> BRANCH_DELAY_SLOTS,
    /// then BRANCH_DELAY_SLOTS DS instructions -> 0.
    pub delay_slots: u8,
    /// Whether this branch is a call (jl) that should update LR on completion.
    pub is_call: bool,
}

/// Branch delay slot count (pipeline depth, from archspec processor model).
const BRANCH_DELAY_SLOTS: u8 = xdna_archspec::aie2::processor::BRANCH_DELAY_SLOTS;

/// Initial counter value: BRANCH_DELAY_SLOTS + 1 because tick() is called on
/// the branch cycle itself (before the first delay-slot instruction executes).
const BRANCH_DELAY_INITIAL: u8 = BRANCH_DELAY_SLOTS + 1;

impl PendingBranch {
    /// Create a new pending branch with BRANCH_DELAY_SLOTS delay slots.
    ///
    /// Uses initial count of BRANCH_DELAY_INITIAL because tick() is called
    /// on the branch cycle.
    pub fn new(target: u32) -> Self {
        Self {
            target,
            delay_slots: BRANCH_DELAY_INITIAL,
            is_call: false,
        }
    }

    /// Create a new pending call (jl) with BRANCH_DELAY_SLOTS delay slots.
    ///
    /// When delay slots are exhausted, the caller should set LR = current PC.
    pub fn new_call(target: u32) -> Self {
        Self {
            target,
            delay_slots: BRANCH_DELAY_INITIAL,
            is_call: true,
        }
    }

    /// Decrement delay slot counter. Returns true if branch should now be taken.
    pub fn tick(&mut self) -> bool {
        if self.delay_slots > 0 {
            self.delay_slots -= 1;
        }
        self.delay_slots == 0
    }
}

// ============================================================================
// SRS/UPS Control Register State
// ============================================================================

/// Vector control register state for SRS/UPS operations.
///
/// Models the Core_CR MMIO register fields that control shift-round-saturate
/// behavior. In hardware, these are set by instructions like `set_satmode()`
/// and `set_rnd()` which compile to control register writes.
///
/// Hardware register layout (from aie-rt `xaiemlgbl_params.h`):
///   [1:0]  SATURATION_MODE  -- 0=none, 1=saturate, 2=symmetric (reserved), 3=symmetric saturate
///   [5:2]  ROUND_MODE       -- 0-3, 8-13 valid (see `RoundingMode`)
///   [17]   SRS_SIGN         -- 0=unsigned, 1=signed
///
/// Hardware reset defaults: all zero (no saturation, Floor rounding, unsigned).
/// But compiled code invariably sets these before SRS instructions -- typical
/// settings are PosInf rounding, saturation enabled, signed output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SrsConfig {
    /// Rounding mode (crRnd) -- hardware field [5:2] of Core_CR.
    pub rounding_mode: u8,

    /// Saturation mode (crSat) -- hardware field [1:0] of Core_CR.
    /// 0 = no saturation (truncate), 1 = saturate, 3 = symmetric saturate.
    pub saturation_mode: u8,

    /// SRS sign mode (crSRSSign) -- hardware bit [17] of Core_CR.
    /// 0 = unsigned output, 1 = signed output.
    pub srs_sign: bool,
}

impl Default for SrsConfig {
    /// Hardware reset defaults: all zero.
    ///
    /// Kernel preamble code configures crRnd/crSat/crSRSSign via control
    /// register write instructions before any SRS/UPS operations.
    fn default() -> Self {
        Self {
            rounding_mode: 0,   // Floor
            saturation_mode: 0, // No saturation
            srs_sign: false,    // Unsigned
        }
    }
}

impl SrsConfig {
    /// Whether saturation is enabled (mode bits [0] set).
    pub fn saturate(&self) -> bool {
        self.saturation_mode & 0x1 != 0
    }

    /// Whether symmetric saturation is enabled (mode bits [1] set).
    pub fn symmetric_saturate(&self) -> bool {
        self.saturation_mode & 0x2 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_context_reset() {
        let mut timing = TimingContext::new();
        timing.hazard_stalls = 10;
        timing.memory_stalls = 5;

        timing.reset();
        assert_eq!(timing.hazard_stalls, 0);
        assert_eq!(timing.memory_stalls, 0);
        assert_eq!(timing.total_stall_cycles(), 0);
    }

    #[test]
    fn test_timing_context_event_tracing() {
        let mut timing = TimingContext::new();

        // Events disabled by default
        timing.record_event(10, EventType::BranchTaken { from_pc: 0x100, to_pc: 0x200 });
        assert!(timing.events.is_empty());

        // Enable tracing
        timing.enable_tracing();
        timing.record_event(20, EventType::BranchTaken { from_pc: 0x200, to_pc: 0x300 });
        assert_eq!(timing.events.len(), 1);

        // Disable tracing
        timing.disable_tracing();
        timing.record_event(30, EventType::CoreDisabled);
        assert_eq!(timing.events.len(), 1); // No new event recorded
    }
}
