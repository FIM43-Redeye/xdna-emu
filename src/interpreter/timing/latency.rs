//! Instruction latency tables for AIE2.
//!
//! Latencies are derived from AMD AM020 (AIE-ML Architecture Manual) Chapter 4.
//! All values assume 1 GHz clock (1 cycle = 1 nanosecond).
//!
//! # Latency Categories
//!
//! | Category | Latency | Notes |
//! |----------|---------|-------|
//! | Scalar simple | 1 cycle | add, sub, and, or, xor, shift, compare |
//! | Scalar multiply | 2 cycles | 32x32 multiplication |
//! | Memory access | 5 cycles | Load/store to data memory |
//! | AGU | 1 cycle | Address generation unit |
//! | Vector simple | 1-2 cycles | Basic vector ops |
//! | Vector MAC | 4+ cycles | Multiply-accumulate |

use std::collections::HashMap;
use crate::interpreter::bundle::Operation;

/// Timing information for a single operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationTiming {
    /// Latency in cycles (time until result is available).
    pub latency: u8,

    /// Throughput: cycles until another op of same type can issue.
    /// 1 = can issue every cycle, 2 = every other cycle, etc.
    pub throughput: u8,

    /// Pipeline stage where result is produced (for forwarding).
    /// 0 = combinational, 1 = after 1 stage, etc.
    pub result_stage: u8,
}

impl OperationTiming {
    /// Create a simple timing with latency = throughput.
    pub const fn simple(latency: u8) -> Self {
        Self {
            latency,
            throughput: 1,  // Can issue every cycle
            result_stage: latency.saturating_sub(1),
        }
    }

    /// Create timing with explicit throughput.
    pub const fn with_throughput(latency: u8, throughput: u8) -> Self {
        Self {
            latency,
            throughput,
            result_stage: latency.saturating_sub(1),
        }
    }
}

impl Default for OperationTiming {
    fn default() -> Self {
        Self::simple(1)
    }
}

// ============================================================================
// AM020 Latency Constants
// ============================================================================

/// Scalar add/subtract latency: 1 cycle (AM020 Ch4)
pub const LATENCY_SCALAR_ADD: u8 = 1;

/// Scalar logical operations (AND, OR, XOR): 1 cycle
pub const LATENCY_SCALAR_LOGIC: u8 = 1;

/// Scalar shift operations: 1 cycle
pub const LATENCY_SCALAR_SHIFT: u8 = 1;

/// Scalar compare: 1 cycle
pub const LATENCY_SCALAR_CMP: u8 = 1;

/// Scalar multiply (32x32): 2 cycles (AM020 Ch4)
pub const LATENCY_SCALAR_MUL: u8 = 2;

/// Scalar move: 1 cycle
pub const LATENCY_SCALAR_MOV: u8 = 1;

/// Data memory access: 5 cycles (AM020 Ch4)
/// "Load and store units manage the 5-cycle latency of data memory."
pub const LATENCY_MEMORY: u8 = 5;

/// Address generation unit: 1 cycle (AM020 Ch4)
pub const LATENCY_AGU: u8 = 1;

/// NOP: 0 cycles (just advances PC)
pub const LATENCY_NOP: u8 = 0;

/// Branch (not taken): 1 cycle
pub const LATENCY_BRANCH_NOT_TAKEN: u8 = 1;

/// Branch (taken): 3 cycles (pipeline flush estimate)
/// AM020 doesn't specify exact penalty; this is an estimate.
pub const LATENCY_BRANCH_TAKEN: u8 = 3;

/// Call: same as branch taken
pub const LATENCY_CALL: u8 = 3;

/// Return: same as branch taken
pub const LATENCY_RETURN: u8 = 3;

/// Lock acquire (uncontested): 1 cycle
pub const LATENCY_LOCK_ACQUIRE: u8 = 1;

/// Lock release: 1 cycle
pub const LATENCY_LOCK_RELEASE: u8 = 1;

// Vector operation latencies (estimates based on pipeline depth)
// AM020 mentions "eight stages maximum" for the pipeline.

/// Vector simple ops (add, sub, compare): 2 cycles
pub const LATENCY_VECTOR_SIMPLE: u8 = 2;

/// Vector multiply: 3 cycles
pub const LATENCY_VECTOR_MUL: u8 = 3;

/// Vector MAC (multiply-accumulate): 4 cycles
/// "Supports 128 bfloat 16 MAC operations" (AM020 Ch4)
pub const LATENCY_VECTOR_MAC: u8 = 4;

/// Vector shuffle/permute: 2 cycles
pub const LATENCY_VECTOR_SHUFFLE: u8 = 2;

/// Vector pack/unpack: 2 cycles
pub const LATENCY_VECTOR_PACK: u8 = 2;

// ============================================================================
// Latency Table
// ============================================================================

/// Lookup table for operation latencies.
///
/// Pre-computed for fast access during execution.
#[derive(Debug, Clone)]
pub struct LatencyTable {
    /// Default timing for unknown operations.
    default_timing: OperationTiming,

    /// Cached timings for common operations.
    /// Using a small array for cache-friendly access.
    cache: [OperationTiming; 32],

    /// Full map for less common operations.
    extended: HashMap<OperationKey, OperationTiming>,
}

/// Key for operation lookup (simplified from Operation enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationKey {
    Nop,
    ScalarAdd,
    ScalarSub,
    ScalarMul,
    ScalarAnd,
    ScalarOr,
    ScalarXor,
    ScalarShl,
    ScalarShr,
    ScalarSra,
    ScalarMov,
    ScalarCmp,
    Load,
    Store,
    Branch,
    Call,
    Return,
    LockAcquire,
    LockRelease,
    VectorAdd,
    VectorSub,
    VectorMul,
    VectorMac,
    VectorShuffle,
    VectorPack,
    VectorCmp,
    DmaStart,
    DmaWait,
    Unknown,
}

impl LatencyTable {
    /// Create the AIE2 latency table from AM020 specifications.
    pub fn aie2() -> Self {
        let mut table = Self {
            default_timing: OperationTiming::simple(1),
            cache: [OperationTiming::simple(1); 32],
            extended: HashMap::new(),
        };

        // Populate cache with common operations
        table.set(OperationKey::Nop, OperationTiming::simple(LATENCY_NOP));
        table.set(OperationKey::ScalarAdd, OperationTiming::simple(LATENCY_SCALAR_ADD));
        table.set(OperationKey::ScalarSub, OperationTiming::simple(LATENCY_SCALAR_ADD));
        table.set(OperationKey::ScalarMul, OperationTiming::simple(LATENCY_SCALAR_MUL));
        table.set(OperationKey::ScalarAnd, OperationTiming::simple(LATENCY_SCALAR_LOGIC));
        table.set(OperationKey::ScalarOr, OperationTiming::simple(LATENCY_SCALAR_LOGIC));
        table.set(OperationKey::ScalarXor, OperationTiming::simple(LATENCY_SCALAR_LOGIC));
        table.set(OperationKey::ScalarShl, OperationTiming::simple(LATENCY_SCALAR_SHIFT));
        table.set(OperationKey::ScalarShr, OperationTiming::simple(LATENCY_SCALAR_SHIFT));
        table.set(OperationKey::ScalarSra, OperationTiming::simple(LATENCY_SCALAR_SHIFT));
        table.set(OperationKey::ScalarMov, OperationTiming::simple(LATENCY_SCALAR_MOV));
        table.set(OperationKey::ScalarCmp, OperationTiming::simple(LATENCY_SCALAR_CMP));

        // Memory operations
        table.set(OperationKey::Load, OperationTiming::simple(LATENCY_MEMORY));
        table.set(OperationKey::Store, OperationTiming::simple(LATENCY_MEMORY));

        // Control flow
        table.set(OperationKey::Branch, OperationTiming::simple(LATENCY_BRANCH_TAKEN));
        table.set(OperationKey::Call, OperationTiming::simple(LATENCY_CALL));
        table.set(OperationKey::Return, OperationTiming::simple(LATENCY_RETURN));

        // Locks
        table.set(OperationKey::LockAcquire, OperationTiming::simple(LATENCY_LOCK_ACQUIRE));
        table.set(OperationKey::LockRelease, OperationTiming::simple(LATENCY_LOCK_RELEASE));

        // Vector operations
        table.set(OperationKey::VectorAdd, OperationTiming::simple(LATENCY_VECTOR_SIMPLE));
        table.set(OperationKey::VectorSub, OperationTiming::simple(LATENCY_VECTOR_SIMPLE));
        table.set(OperationKey::VectorMul, OperationTiming::simple(LATENCY_VECTOR_MUL));
        table.set(OperationKey::VectorMac, OperationTiming::simple(LATENCY_VECTOR_MAC));
        table.set(OperationKey::VectorShuffle, OperationTiming::simple(LATENCY_VECTOR_SHUFFLE));
        table.set(OperationKey::VectorPack, OperationTiming::simple(LATENCY_VECTOR_PACK));
        table.set(OperationKey::VectorCmp, OperationTiming::simple(LATENCY_VECTOR_SIMPLE));

        // DMA (instant start, wait depends on transfer)
        table.set(OperationKey::DmaStart, OperationTiming::simple(1));
        table.set(OperationKey::DmaWait, OperationTiming::simple(1)); // Actual wait is modeled separately

        table
    }

    /// Set timing for an operation.
    fn set(&mut self, key: OperationKey, timing: OperationTiming) {
        let idx = key as usize;
        if idx < self.cache.len() {
            self.cache[idx] = timing;
        } else {
            self.extended.insert(key, timing);
        }
    }

    /// Get timing for an operation.
    #[inline]
    pub fn get(&self, key: OperationKey) -> OperationTiming {
        let idx = key as usize;
        if idx < self.cache.len() {
            self.cache[idx]
        } else {
            self.extended.get(&key).copied().unwrap_or(self.default_timing)
        }
    }

    /// Get latency (cycles) for an operation.
    #[inline]
    pub fn latency(&self, key: OperationKey) -> u8 {
        self.get(key).latency
    }

    /// Convert from Operation enum to OperationKey.
    pub fn key_from_operation(op: &Operation) -> OperationKey {
        match op {
            Operation::Nop => OperationKey::Nop,
            Operation::Halt => OperationKey::Unknown, // Treat halt as unknown for timing
            Operation::ScalarAdd => OperationKey::ScalarAdd,
            Operation::ScalarSub => OperationKey::ScalarSub,
            Operation::ScalarMul => OperationKey::ScalarMul,
            Operation::ScalarAnd => OperationKey::ScalarAnd,
            Operation::ScalarOr => OperationKey::ScalarOr,
            Operation::ScalarXor => OperationKey::ScalarXor,
            Operation::ScalarShl => OperationKey::ScalarShl,
            Operation::ScalarShr => OperationKey::ScalarShr,
            Operation::ScalarSra => OperationKey::ScalarSra,
            Operation::ScalarMov | Operation::ScalarMovi { .. } => OperationKey::ScalarMov,
            Operation::ScalarCmp => OperationKey::ScalarCmp,
            Operation::PointerAdd | Operation::PointerMov => OperationKey::Load, // Pointer ops have similar timing to loads
            Operation::Load { .. } => OperationKey::Load,
            Operation::Store { .. } => OperationKey::Store,
            Operation::Branch { .. } => OperationKey::Branch,
            Operation::Call => OperationKey::Call,
            Operation::Return => OperationKey::Return,
            Operation::LockAcquire => OperationKey::LockAcquire,
            Operation::LockRelease => OperationKey::LockRelease,
            Operation::VectorAdd { .. } => OperationKey::VectorAdd,
            Operation::VectorSub { .. } => OperationKey::VectorSub,
            Operation::VectorMul { .. } => OperationKey::VectorMul,
            Operation::VectorMac { .. } => OperationKey::VectorMac,
            Operation::VectorShuffle { .. } => OperationKey::VectorShuffle,
            Operation::VectorPack | Operation::VectorUnpack => OperationKey::VectorPack,
            Operation::VectorCmp { .. } | Operation::VectorMin { .. } | Operation::VectorMax { .. } => {
                OperationKey::VectorCmp
            }
            Operation::DmaStart => OperationKey::DmaStart,
            Operation::DmaWait => OperationKey::DmaWait,
            Operation::Unknown { .. } => OperationKey::Unknown,
        }
    }

    /// Get timing directly from an Operation.
    #[inline]
    pub fn timing_for(&self, op: &Operation) -> OperationTiming {
        self.get(Self::key_from_operation(op))
    }
}

impl Default for LatencyTable {
    fn default() -> Self {
        Self::aie2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_table_creation() {
        let table = LatencyTable::aie2();

        // Scalar ops
        assert_eq!(table.latency(OperationKey::ScalarAdd), 1);
        assert_eq!(table.latency(OperationKey::ScalarMul), 2);

        // Memory
        assert_eq!(table.latency(OperationKey::Load), 5);
        assert_eq!(table.latency(OperationKey::Store), 5);

        // Vector
        assert_eq!(table.latency(OperationKey::VectorMac), 4);
    }

    #[test]
    fn test_operation_timing() {
        let timing = OperationTiming::simple(3);
        assert_eq!(timing.latency, 3);
        assert_eq!(timing.throughput, 1);
        assert_eq!(timing.result_stage, 2);
    }

    #[test]
    fn test_key_from_operation() {
        assert_eq!(
            LatencyTable::key_from_operation(&Operation::ScalarAdd),
            OperationKey::ScalarAdd
        );
        assert_eq!(
            LatencyTable::key_from_operation(&Operation::Nop),
            OperationKey::Nop
        );
    }
}
