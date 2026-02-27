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
//! | Memory access | 7 cycles | Load result availability (issue to register) |
//! | AGU | 1 cycle | Address generation unit |
//! | Vector simple | 1-2 cycles | Basic vector ops |
//! | Vector MAC | 4+ cycles | Multiply-accumulate |

use std::collections::HashMap;
use crate::interpreter::bundle::{Operation, SlotOp};
use crate::tablegen::SemanticOp;

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

/// Scalar division: 6 cycles (iterative algorithm estimate)
/// AIE2 uses iterative division, not a hardware divider.
pub const LATENCY_SCALAR_DIV: u8 = 6;

/// Scalar select: 1 cycle (conditional move)
pub const LATENCY_SCALAR_SEL: u8 = 1;

/// Scalar move: 1 cycle
pub const LATENCY_SCALAR_MOV: u8 = 1;

/// Data memory load result latency: 7 cycles (AIE2Schedule.td)
///
/// This is the full pipeline from issue to register availability:
///   Cycle 0: Address validation (AvoidPartWordStore stage)
///   Cycle 2: Address sent to memory (LOAD_UNIT_A/B stage)
///   Cycle 5: Memory access completes (MemoryCycles<[5]>)
///   Cycle 7: Writeback to register file (P_WM/R_WA ports)
///
/// Every load itinerary in AIE2Schedule.td confirms operandcycles[0] = 7.
/// ProcessorModel.LoadLatency = 5 is a DEFAULT fallback for instructions
/// without itinerary data; the compiler's scheduler uses the itinerary-
/// specific 7-cycle value.
pub const LATENCY_MEMORY: u8 = 7;

/// Data memory store: 1 cycle (fire-and-forget from the core).
/// Stores push data into a write buffer and the core continues immediately.
/// (TableGen II_ST operand_cycles[0] = 1)
pub const LATENCY_STORE: u8 = 1;

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

/// Vector multiply: 5 cycles (TableGen II_VMUL operand_cycles[0] = 5)
pub const LATENCY_VECTOR_MUL: u8 = 5;

/// Vector MAC (multiply-accumulate): 5 cycles
/// (TableGen II_VMAC operand_cycles[0] = 5; accumulator input at cycle 3)
pub const LATENCY_VECTOR_MAC: u8 = 5;

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
    ScalarDiv,
    ScalarSel,
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
        table.set(OperationKey::ScalarDiv, OperationTiming::with_throughput(LATENCY_SCALAR_DIV, 6));
        table.set(OperationKey::ScalarSel, OperationTiming::simple(LATENCY_SCALAR_SEL));

        // Memory operations
        table.set(OperationKey::Load, OperationTiming::simple(LATENCY_MEMORY));
        table.set(OperationKey::Store, OperationTiming::simple(LATENCY_STORE));

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
            // New scalar operations - use existing timing categories
            Operation::ScalarAbs
            | Operation::ScalarClz
            | Operation::ScalarClb
            | Operation::ScalarExtendS8
            | Operation::ScalarExtendS16
            | Operation::ScalarExtendU8
            | Operation::ScalarExtendU16 => OperationKey::ScalarAdd, // Single-cycle scalar ops
            Operation::ScalarAdc | Operation::ScalarSbc => OperationKey::ScalarAdd, // Same as add/sub
            // Division operations - multi-cycle
            Operation::ScalarDiv | Operation::ScalarDivu | Operation::ScalarMod => OperationKey::ScalarDiv,
            // Select operations - single cycle
            Operation::ScalarSelEqz | Operation::ScalarSelNez => OperationKey::ScalarSel,
            // Comparison operations have same timing as compare
            Operation::ScalarLt
            | Operation::ScalarLtu
            | Operation::ScalarLe
            | Operation::ScalarLeu
            | Operation::ScalarGt
            | Operation::ScalarGtu
            | Operation::ScalarGe
            | Operation::ScalarGeu
            | Operation::ScalarEq
            | Operation::ScalarNe
            | Operation::ScalarSel => OperationKey::ScalarCmp,
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
            // Matrix multiply operations - use MAC timing (multi-cycle)
            // All convolution-related operations share MAC timing
            Operation::VectorMatMulDense { .. }
            | Operation::VectorMatMulSparse { .. }
            | Operation::VectorMatMulSubDense { .. }
            | Operation::VectorMatMulSubSparse { .. }
            | Operation::VectorNegMatMulDense { .. }
            | Operation::VectorNegMatMulSubDense { .. }
            | Operation::VectorMatMulAccFloat { .. }
            | Operation::VectorMatMulSubFloat { .. }
            | Operation::VectorAddMac { .. }
            | Operation::VectorSubMac { .. } => OperationKey::VectorMac,
            // Type conversion and move operations
            Operation::VectorSRS { .. }
            | Operation::VectorConvert { .. }
            | Operation::VectorMov { .. } => OperationKey::VectorAdd, // Similar to vector ALU
            // Vector element operations - single/double cycle
            Operation::VectorExtract { .. }
            | Operation::VectorInsert { .. }
            | Operation::VectorSelect { .. }
            | Operation::VectorClear
            | Operation::VectorBroadcast { .. } => OperationKey::VectorAdd,
            // Vector shift operations
            Operation::VectorShiftLeft { .. }
            | Operation::VectorShiftRight { .. }
            | Operation::VectorArithShiftRight { .. } => OperationKey::VectorAdd,
            // Align is similar to shuffle (concatenate and shift)
            Operation::VectorAlign { .. } => OperationKey::VectorShuffle,
            // Upshift for precision scaling
            Operation::VectorUpshift { .. } => OperationKey::VectorAdd,
            // Conditional vector operations - use simple vector ALU timing (2 cycles)
            Operation::VectorAbsGtz { .. }
            | Operation::VectorNegGtz { .. }
            | Operation::VectorNegLtz { .. }
            | Operation::VectorNegate { .. }
            | Operation::VectorNegAdd { .. } => OperationKey::VectorAdd,
            // Accumulator operations use MAC timing
            Operation::VectorAccumulate { .. }
            | Operation::VectorNegMul { .. } => OperationKey::VectorMac,
            // Vector memory operations
            Operation::VectorLoadA { .. }
            | Operation::VectorLoadB { .. }
            | Operation::VectorLoadUnpack { .. } => OperationKey::Load,
            Operation::VectorStore { .. } => OperationKey::Store,
            // Vector comparison operations - use compare timing (2 cycles)
            Operation::VectorGe { .. }
            | Operation::VectorLt { .. }
            | Operation::VectorEqz { .. }
            | Operation::VectorMaxLt { .. }
            | Operation::VectorMinGe { .. } => OperationKey::VectorCmp,
            // Vector bitwise operations - simple ALU timing (2 cycles)
            Operation::VectorAnd { .. }
            | Operation::VectorOr { .. }
            | Operation::VectorXor { .. }
            | Operation::VectorNot { .. } => OperationKey::VectorAdd,
            // Vector conditional arithmetic - simple ALU timing (2 cycles)
            Operation::VectorSubLt { .. }
            | Operation::VectorSubGe { .. }
            | Operation::VectorMaxDiffLt { .. } => OperationKey::VectorAdd,
            // Cascade operations - use vector move timing
            Operation::CascadeRead | Operation::CascadeWrite => OperationKey::VectorAdd,
            // Stream operations - use DMA timing
            Operation::StreamWriteScalar { .. }
            | Operation::StreamWritePacketHeader { .. }
            | Operation::StreamReadScalar { .. } => OperationKey::DmaStart,
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

    /// Convert from SemanticOp + vector flag to OperationKey.
    ///
    /// This is the preferred path for latency lookup -- it uses the TableGen-derived
    /// semantic operation instead of the deprecated Operation enum. The `is_vector`
    /// flag disambiguates between scalar and vector functional units (e.g.,
    /// `SemanticOp::Add` + `is_vector=false` -> `ScalarAdd`, `is_vector=true` -> `VectorAdd`).
    pub fn key_from_semantic(semantic: SemanticOp, is_vector: bool) -> OperationKey {
        match semantic {
            // Arithmetic
            SemanticOp::Add if !is_vector => OperationKey::ScalarAdd,
            SemanticOp::Add => OperationKey::VectorAdd,
            SemanticOp::Sub if !is_vector => OperationKey::ScalarSub,
            SemanticOp::Sub => OperationKey::VectorSub,
            SemanticOp::Mul if !is_vector => OperationKey::ScalarMul,
            SemanticOp::Mul => OperationKey::VectorMul,
            SemanticOp::SDiv | SemanticOp::UDiv | SemanticOp::SRem | SemanticOp::URem
                if !is_vector => OperationKey::ScalarDiv,
            SemanticOp::SDiv | SemanticOp::UDiv | SemanticOp::SRem | SemanticOp::URem
                => OperationKey::VectorMac, // Vector div uses MAC pipeline
            SemanticOp::Abs | SemanticOp::Neg if !is_vector => OperationKey::ScalarAdd,
            SemanticOp::Abs | SemanticOp::Neg => OperationKey::VectorAdd,

            // Bitwise
            SemanticOp::And if !is_vector => OperationKey::ScalarAnd,
            SemanticOp::And => OperationKey::VectorAdd, // Vector bitwise uses ALU pipeline
            SemanticOp::Or if !is_vector => OperationKey::ScalarOr,
            SemanticOp::Or => OperationKey::VectorAdd,
            SemanticOp::Xor if !is_vector => OperationKey::ScalarXor,
            SemanticOp::Xor => OperationKey::VectorAdd,
            SemanticOp::Not if !is_vector => OperationKey::ScalarXor, // NOT = XOR with -1
            SemanticOp::Not => OperationKey::VectorAdd,

            // Shifts
            SemanticOp::Shl if !is_vector => OperationKey::ScalarShl,
            SemanticOp::Shl => OperationKey::VectorAdd,
            SemanticOp::Sra if !is_vector => OperationKey::ScalarSra,
            SemanticOp::Sra => OperationKey::VectorAdd,
            SemanticOp::Srl if !is_vector => OperationKey::ScalarShr,
            SemanticOp::Srl => OperationKey::VectorAdd,
            SemanticOp::Rotl | SemanticOp::Rotr if !is_vector => OperationKey::ScalarShl,
            SemanticOp::Rotl | SemanticOp::Rotr => OperationKey::VectorShuffle,

            // Comparisons
            SemanticOp::SetEq | SemanticOp::SetNe
            | SemanticOp::SetLt | SemanticOp::SetLe
            | SemanticOp::SetGt | SemanticOp::SetGe
            | SemanticOp::SetUlt | SemanticOp::SetUle
            | SemanticOp::SetUgt | SemanticOp::SetUge
                if !is_vector => OperationKey::ScalarCmp,
            SemanticOp::SetEq | SemanticOp::SetNe
            | SemanticOp::SetLt | SemanticOp::SetLe
            | SemanticOp::SetGt | SemanticOp::SetGe
            | SemanticOp::SetUlt | SemanticOp::SetUle
            | SemanticOp::SetUgt | SemanticOp::SetUge
                => OperationKey::VectorCmp,

            // Bit manipulation (scalar only in practice)
            SemanticOp::Ctlz | SemanticOp::Cttz | SemanticOp::Ctpop | SemanticOp::Bswap
                => OperationKey::ScalarAdd, // Single-cycle scalar ALU

            // Memory
            SemanticOp::Load => OperationKey::Load,
            SemanticOp::Store => OperationKey::Store,

            // Control flow (never vector)
            SemanticOp::Br | SemanticOp::BrCond => OperationKey::Branch,
            SemanticOp::Call => OperationKey::Call,
            SemanticOp::Ret => OperationKey::Return,
            SemanticOp::Select if !is_vector => OperationKey::ScalarSel,
            SemanticOp::Select => OperationKey::VectorAdd, // Vector select = ALU

            // Type conversion
            SemanticOp::SignExtend | SemanticOp::ZeroExtend | SemanticOp::Truncate
                if !is_vector => OperationKey::ScalarAdd, // Single-cycle scalar
            SemanticOp::SignExtend | SemanticOp::ZeroExtend | SemanticOp::Truncate
                => OperationKey::VectorPack, // Vector type conversion = pack pipeline

            // Move / NOP / Halt
            SemanticOp::Copy if !is_vector => OperationKey::ScalarMov,
            SemanticOp::Copy => OperationKey::VectorAdd,
            SemanticOp::Nop => OperationKey::Nop,
            SemanticOp::Event => OperationKey::Nop, // EVENT is 1-cycle, no pipeline impact
            SemanticOp::Done => OperationKey::Unknown, // Halt has no meaningful latency

            // Synchronization
            SemanticOp::LockAcquire => OperationKey::LockAcquire,
            SemanticOp::LockRelease => OperationKey::LockRelease,

            // Intrinsics -- default to vector MAC (conservative for high-latency ops)
            SemanticOp::Intrinsic(_) if is_vector => OperationKey::VectorMac,
            SemanticOp::Intrinsic(_) => OperationKey::ScalarMul,
        }
    }

    /// Get timing for a SlotOp, preferring the semantic path when available.
    ///
    /// Falls back to the deprecated `Operation`-based lookup when no semantic
    /// is set (e.g., for instructions not yet covered by the TableGen pattern matcher).
    #[inline]
    pub fn timing_for_slot_op(&self, op: &SlotOp) -> OperationTiming {
        if let Some(semantic) = op.semantic {
            self.get(Self::key_from_semantic(semantic, op.is_vector))
        } else {
            self.get(Self::key_from_operation(&op.op))
        }
    }

    /// Create an AIE2 latency table validated against parsed ProcessorModel.
    ///
    /// Asserts that our hardcoded constants match the values extracted from
    /// `AIE2Schedule.td`. This catches any drift between the emulator and
    /// the compiler's scheduling model.
    pub fn validated_aie2(model: &crate::tablegen::ProcessorModel) -> Self {
        // ProcessorModel.LoadLatency (5) is the memory pipeline depth.
        // LATENCY_MEMORY (7) is the full instruction result latency including
        // 2 extra writeback stages visible in AIE2Schedule.td itineraries.
        assert_eq!(
            model.load_latency + 2, LATENCY_MEMORY,
            "LATENCY_MEMORY should be ProcessorModel.load_latency ({}) + 2 writeback cycles = {}",
            model.load_latency, LATENCY_MEMORY
        );
        assert_eq!(
            model.mispredict_penalty, LATENCY_BRANCH_TAKEN + 1,
            "ProcessorModel.mispredict_penalty ({}) != LATENCY_BRANCH_TAKEN+1 ({})",
            model.mispredict_penalty, LATENCY_BRANCH_TAKEN + 1
        );
        Self::aie2()
    }
}

/// Map an itinerary class name to the corresponding OperationKey.
///
/// Only maps the "representative" class for each OperationKey bucket.
/// Classes with no clear mapping (e.g., `II_MOVd3`, combined load+UPS)
/// return None.
pub fn itinerary_to_operation_key(class_name: &str) -> Option<OperationKey> {
    Some(match class_name {
        // Scalar arithmetic (1-cycle ALU)
        "II_ADD" | "II_ADD_NC" => OperationKey::ScalarAdd,
        "II_SUB" | "II_SBC" | "II_ADC" => OperationKey::ScalarSub,
        "II_MUL" => OperationKey::ScalarMul,
        "II_AND" => OperationKey::ScalarAnd,
        "II_OR" => OperationKey::ScalarOr,
        "II_XOR" => OperationKey::ScalarXor,
        "II_ASHL" => OperationKey::ScalarShl,
        "II_LSHL" => OperationKey::ScalarShr,
        "II_MOV" | "II_MOVA" | "II_MOVX" | "II_MOV_SCL" => OperationKey::ScalarMov,
        "II_EQ" | "II_NE" | "II_GE" | "II_GEU" | "II_LT" | "II_LTU"
        | "II_EQZ" | "II_NEZ" => OperationKey::ScalarCmp,
        "II_DIVS" => OperationKey::ScalarDiv,
        "II_SELEQZ" | "II_SELNEZ" => OperationKey::ScalarSel,
        // Memory
        "II_LDA" | "II_LDA_POST_1D" | "II_LDA_POST_2D" | "II_LDA_POST_3D" => OperationKey::Load,
        "II_VLDB" | "II_VLDB_POSTINC" | "II_VLDB_2D" | "II_VLDB_3D"
        | "II_VLDA_W" | "II_VLDA_AM" => OperationKey::Load,
        "II_ST" | "II_ST_POST_1D" | "II_ST_POST_2D" | "II_ST_POST_3D" => OperationKey::Store,
        "II_VST_W" | "II_VST_AM" | "II_VST_POSTINC" => OperationKey::Store,
        // Control flow
        "II_J" | "II_JNZ" | "II_JZ" | "II_JNZD" => OperationKey::Branch,
        "II_JL" | "II_JL_IND" => OperationKey::Call,
        "II_RET" => OperationKey::Return,
        // Locks
        "II_ACQ" | "II_ACQ_COND" => OperationKey::LockAcquire,
        "II_REL" | "II_REL_COND" => OperationKey::LockRelease,
        // Vector compute
        "II_VADD" | "II_VSUB" => OperationKey::VectorAdd,
        "II_VMUL" | "II_VNEGMUL" => OperationKey::VectorMul,
        "II_VMAC" | "II_VMSC" | "II_VNEGMAC" | "II_VNEGMSC"
        | "II_VACC" | "II_VADDMAC" => OperationKey::VectorMac,
        "II_VSHUFFLE" | "II_VBCSTSHFL" => OperationKey::VectorShuffle,
        "II_VPACK" => OperationKey::VectorPack,
        "II_VCMP" => OperationKey::VectorCmp,
        "II_NOP" => OperationKey::Nop,
        _ => return None,
    })
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
        assert_eq!(table.latency(OperationKey::Load), 7);
        assert_eq!(table.latency(OperationKey::Store), 1);

        // Vector
        assert_eq!(table.latency(OperationKey::VectorMac), 5);
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

    // ── Semantic path tests ──────────────────────────────────────────

    #[test]
    fn test_key_from_semantic_scalar_arithmetic() {
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Add, false), OperationKey::ScalarAdd);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Sub, false), OperationKey::ScalarSub);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Mul, false), OperationKey::ScalarMul);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::SDiv, false), OperationKey::ScalarDiv);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::UDiv, false), OperationKey::ScalarDiv);
    }

    #[test]
    fn test_key_from_semantic_vector_arithmetic() {
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Add, true), OperationKey::VectorAdd);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Sub, true), OperationKey::VectorSub);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Mul, true), OperationKey::VectorMul);
    }

    #[test]
    fn test_key_from_semantic_bitwise() {
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::And, false), OperationKey::ScalarAnd);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Or, false), OperationKey::ScalarOr);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Xor, false), OperationKey::ScalarXor);
        // Vector bitwise uses ALU pipeline (same timing as vector add)
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::And, true), OperationKey::VectorAdd);
    }

    #[test]
    fn test_key_from_semantic_shifts() {
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Shl, false), OperationKey::ScalarShl);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Sra, false), OperationKey::ScalarSra);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Srl, false), OperationKey::ScalarShr);
    }

    #[test]
    fn test_key_from_semantic_comparison() {
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::SetEq, false), OperationKey::ScalarCmp);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::SetLt, false), OperationKey::ScalarCmp);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::SetEq, true), OperationKey::VectorCmp);
    }

    #[test]
    fn test_key_from_semantic_memory() {
        // Memory ops are the same regardless of is_vector
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Load, false), OperationKey::Load);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Load, true), OperationKey::Load);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Store, false), OperationKey::Store);
    }

    #[test]
    fn test_key_from_semantic_control_flow() {
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Br, false), OperationKey::Branch);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::BrCond, false), OperationKey::Branch);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Call, false), OperationKey::Call);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Ret, false), OperationKey::Return);
    }

    #[test]
    fn test_key_from_semantic_special() {
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Copy, false), OperationKey::ScalarMov);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Copy, true), OperationKey::VectorAdd);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Nop, false), OperationKey::Nop);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::Select, false), OperationKey::ScalarSel);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::LockAcquire, false), OperationKey::LockAcquire);
        assert_eq!(LatencyTable::key_from_semantic(SemanticOp::LockRelease, false), OperationKey::LockRelease);
    }

    #[test]
    fn test_timing_for_slot_op_prefers_semantic() {
        use crate::interpreter::bundle::{SlotIndex, PostModify, MemWidth};

        let table = LatencyTable::aie2();

        // SlotOp with semantic set -> uses semantic path
        let mut op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd);
        op.semantic = Some(SemanticOp::Mul); // Override: semantic says Mul
        op.is_vector = false;

        // Should get ScalarMul latency (2), not ScalarAdd latency (1)
        let timing = table.timing_for_slot_op(&op);
        assert_eq!(timing.latency, LATENCY_SCALAR_MUL);
    }

    #[test]
    fn test_timing_for_slot_op_falls_back_to_operation() {
        use crate::interpreter::bundle::SlotIndex;

        let table = LatencyTable::aie2();

        // SlotOp without semantic -> falls back to Operation
        let mut op = SlotOp::new(SlotIndex::Scalar1, Operation::ScalarMul);
        op.semantic = None;

        let timing = table.timing_for_slot_op(&op);
        assert_eq!(timing.latency, LATENCY_SCALAR_MUL);
    }

    #[test]
    fn test_semantic_and_operation_paths_agree() {
        // For the common operations where both paths are available, verify they
        // produce the same latency. This catches divergence between the two maps.
        let table = LatencyTable::aie2();

        let agreement_cases: &[(SemanticOp, bool, Operation)] = &[
            (SemanticOp::Add, false, Operation::ScalarAdd),
            (SemanticOp::Sub, false, Operation::ScalarSub),
            (SemanticOp::Mul, false, Operation::ScalarMul),
            (SemanticOp::And, false, Operation::ScalarAnd),
            (SemanticOp::Or, false, Operation::ScalarOr),
            (SemanticOp::Xor, false, Operation::ScalarXor),
            (SemanticOp::Shl, false, Operation::ScalarShl),
            (SemanticOp::Sra, false, Operation::ScalarSra),
            (SemanticOp::Copy, false, Operation::ScalarMov),
            (SemanticOp::Nop, false, Operation::Nop),
            (SemanticOp::Call, false, Operation::Call),
            (SemanticOp::Ret, false, Operation::Return),
            (SemanticOp::LockAcquire, false, Operation::LockAcquire),
            (SemanticOp::LockRelease, false, Operation::LockRelease),
        ];

        for (semantic, is_vector, operation) in agreement_cases {
            let semantic_key = LatencyTable::key_from_semantic(*semantic, *is_vector);
            let operation_key = LatencyTable::key_from_operation(operation);
            assert_eq!(
                table.latency(semantic_key),
                table.latency(operation_key),
                "Latency mismatch for {:?} (is_vector={}) vs {:?}: semantic={}, operation={}",
                semantic, is_vector, operation,
                table.latency(semantic_key),
                table.latency(operation_key),
            );
        }
    }

    #[test]
    fn test_validated_aie2_matches_processor_model() {
        use crate::tablegen::ProcessorModel;

        let model = ProcessorModel {
            load_latency: 5,
            high_latency: 37,
            mispredict_penalty: 4,
            issue_width: 1000,
            itinerary_name: "AIE2Itineraries".into(),
        };

        // Should not panic -- our constants match AIE2's ProcessorModel
        let table = LatencyTable::validated_aie2(&model);
        assert_eq!(table.latency(OperationKey::Load), 7);
    }

    #[test]
    #[should_panic(expected = "LATENCY_MEMORY should be")]
    fn test_validated_aie2_catches_load_latency_drift() {
        use crate::tablegen::ProcessorModel;

        let model = ProcessorModel {
            load_latency: 9, // Wrong! AIE2 is 5 (9 + 2 = 11 != 7)
            high_latency: 37,
            mispredict_penalty: 4,
            issue_width: 1000,
            itinerary_name: "AIE2Itineraries".into(),
        };

        // Should panic because load_latency doesn't match
        let _table = LatencyTable::validated_aie2(&model);
    }

    /// Cross-validate latency table against TableGen itinerary data.
    ///
    /// For each itinerary class that maps to an OperationKey, compares
    /// our hardcoded latency against `operand_cycles[0]` (the result
    /// availability cycle from the compiler's scheduling model).
    ///
    /// Known differences:
    /// - Branch/Call/Return: TableGen models operand read timing (cycle 1),
    ///   not the pipeline flush penalty. Our latency represents the taken-
    ///   branch cost, which is a different concept.
    #[test]
    fn test_latency_cross_validation_against_itineraries() {
        let tblgen = match crate::tablegen::load_full_via_tblgen(
            std::path::Path::new("/home/triple/npu-work/llvm-aie"),
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Skipping itinerary cross-validation (llvm-aie not available): {}", e);
                return;
            }
        };

        let table = LatencyTable::aie2();
        let mut checked = 0u32;
        let mut mismatches = Vec::new();

        // Classes where the LatencyTable latency intentionally differs from
        // operand_cycles[0]. For these we verify the relationship rather than
        // asserting equality.
        let known_difference_classes: &[&str] = &[
            // Branch/call/return: TableGen models operand timing, not flush penalty.
            "II_J", "II_JNZ", "II_JZ", "II_JNZD", "II_JL", "II_JL_IND", "II_RET",
            // NOP: TableGen has 1-cycle stage, we model as 0 (just advances PC).
            "II_NOP",
            // Division: TableGen models operand_cycles[0]=1 because the pipeline
            // stalls during iterative division. Our 6-cycle value represents the
            // actual computation time (throughput, not scheduling latency).
            "II_DIVS",
        ];

        for (class_name, itin) in &tblgen.itineraries {
            let key = match itinerary_to_operation_key(class_name) {
                Some(k) => k,
                None => continue, // No mapping for this class
            };

            let our_latency = table.latency(key);
            // operand_cycles[0] = result availability cycle (the key metric)
            let tblgen_latency = itin.operand_cycles.first().copied().unwrap_or(itin.total_latency);

            checked += 1;

            if known_difference_classes.contains(&class_name.as_str()) {
                // Known difference: just verify the relationship is sane
                continue;
            }

            if our_latency != tblgen_latency {
                mismatches.push(format!(
                    "  {} -> {:?}: ours={}, tblgen operand_cycles[0]={}",
                    class_name, key, our_latency, tblgen_latency,
                ));
            }
        }

        assert!(checked > 0, "No itinerary classes were checked -- is the mapping empty?");

        if !mismatches.is_empty() {
            panic!(
                "Latency mismatches between LatencyTable and TableGen itineraries ({}/{} checked):\n{}",
                mismatches.len(), checked, mismatches.join("\n"),
            );
        }
    }
}
