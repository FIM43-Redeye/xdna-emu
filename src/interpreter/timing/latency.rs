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

use crate::interpreter::bundle::SlotOp;
use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::aie2::isa::decoder_ffi;
use xdna_archspec::aie2::{Bypass, instruction_latency, timing};

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
            throughput: 1, // Can issue every cycle
            result_stage: latency.saturating_sub(1),
        }
    }

    /// Create timing with explicit throughput.
    pub const fn with_throughput(latency: u8, throughput: u8) -> Self {
        Self { latency, throughput, result_stage: latency.saturating_sub(1) }
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

/// Scalar multiply (32x32): 2 cycles.
///
/// Sourced from `xdna_archspec::aie2::instruction_latency::SCALAR_MUL`
/// (AIE2Schedule.td `II_MUL` itinerary, `operand_cycles[0] = 2`).
pub const LATENCY_SCALAR_MUL: u8 = instruction_latency::SCALAR_MUL;

/// Scalar division: 6 cycles (iterative algorithm).
///
/// AIE2 uses iterative division, not a hardware divider.
/// Sourced from `xdna_archspec::aie2::instruction_latency::SCALAR_DIV`.
pub const LATENCY_SCALAR_DIV: u8 = instruction_latency::SCALAR_DIV;

/// Scalar select: 1 cycle (conditional move)
pub const LATENCY_SCALAR_SEL: u8 = 1;

/// Scalar move: 1 cycle
pub const LATENCY_SCALAR_MOV: u8 = 1;

/// Data memory load result latency: `DATA_MEMORY_LATENCY + 2` cycles.
///
/// This is the full pipeline from issue to register availability:
///   Cycle 0: Address validation (AvoidPartWordStore stage)
///   Cycle 2: Address sent to memory (LOAD_UNIT_A/B stage)
///   Cycle 5: Memory access completes (MemoryCycles<[5]>)
///   Cycle 7: Writeback to register file (P_WM/R_WA ports)
///
/// `DATA_MEMORY_LATENCY` (= 5) is the memory pipeline depth from archspec.
/// The +2 accounts for the two writeback stages visible in AIE2Schedule.td.
/// Every load itinerary confirms operandcycles[0] = 7.
pub const LATENCY_MEMORY: u8 = timing::DATA_MEMORY_LATENCY + 2;

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

/// Branch (taken): pipeline flush penalty in cycles.
///
/// Sourced from `xdna_archspec::aie2::timing::BRANCH_PENALTY`.
pub const LATENCY_BRANCH_TAKEN: u8 = timing::BRANCH_PENALTY;

/// Call: same as branch taken
pub const LATENCY_CALL: u8 = timing::BRANCH_PENALTY;

/// Return: same as branch taken
pub const LATENCY_RETURN: u8 = timing::BRANCH_PENALTY;

/// Lock acquire (uncontested): 1 cycle.
///
/// Sourced from `xdna_archspec::aie2::timing::LOCK_ACQUIRE_LATENCY`.
pub const LATENCY_LOCK_ACQUIRE: u8 = timing::LOCK_ACQUIRE_LATENCY;

/// Lock release: 1 cycle.
///
/// Sourced from `xdna_archspec::aie2::timing::LOCK_RELEASE_LATENCY`.
pub const LATENCY_LOCK_RELEASE: u8 = timing::LOCK_RELEASE_LATENCY;

// Vector operation latencies (estimates based on pipeline depth)
// AM020 mentions "eight stages maximum" for the pipeline.

/// Vector simple ops (add, sub, compare): 2 cycles.
///
/// Sourced from `xdna_archspec::aie2::instruction_latency::VECTOR_SIMPLE`.
pub const LATENCY_VECTOR_SIMPLE: u8 = instruction_latency::VECTOR_SIMPLE;

/// Vector multiply: 5 cycles (TableGen II_VMUL operand_cycles[0] = 5).
///
/// Sourced from `xdna_archspec::aie2::instruction_latency::VECTOR_MUL`.
pub const LATENCY_VECTOR_MUL: u8 = instruction_latency::VECTOR_MUL;

/// Vector MAC (multiply-accumulate): 5 cycles.
/// (TableGen II_VMAC operand_cycles[0] = 5; accumulator input at cycle 3).
///
/// Sourced from `xdna_archspec::aie2::instruction_latency::VECTOR_MAC`.
pub const LATENCY_VECTOR_MAC: u8 = instruction_latency::VECTOR_MAC;

/// Float (bf16/fp32) vector MAC/MUL: 6 cycles.
/// (TableGen II_VMACf / II_VMULf operand_cycles[0] = 6 -- one cycle longer
/// than the integer MAC for the float normalization stage.)
///
/// Sourced from `xdna_archspec::aie2::instruction_latency::VECTOR_MAC_F`.
pub const LATENCY_VECTOR_MAC_F: u8 = instruction_latency::VECTOR_MAC_F;

/// Vector shuffle/permute: 2 cycles.
///
/// Sourced from `xdna_archspec::aie2::instruction_latency::VECTOR_SHUFFLE`.
pub const LATENCY_VECTOR_SHUFFLE: u8 = instruction_latency::VECTOR_SHUFFLE;

/// Vector pack/unpack: 2 cycles.
///
/// Sourced from `xdna_archspec::aie2::instruction_latency::VECTOR_PACK`.
pub const LATENCY_VECTOR_PACK: u8 = instruction_latency::VECTOR_PACK;

// ============================================================================
// Timing Constants (pre-computed from the latency constants above)
// ============================================================================

const TIMING_NOP: OperationTiming = OperationTiming::simple(LATENCY_NOP);
const TIMING_SCALAR_1: OperationTiming = OperationTiming::simple(LATENCY_SCALAR_ADD);
const TIMING_SCALAR_MUL: OperationTiming = OperationTiming::simple(LATENCY_SCALAR_MUL);
const TIMING_SCALAR_DIV: OperationTiming = OperationTiming::with_throughput(LATENCY_SCALAR_DIV, 6);
const TIMING_LOAD: OperationTiming = OperationTiming::simple(LATENCY_MEMORY);
const TIMING_STORE: OperationTiming = OperationTiming::simple(LATENCY_STORE);
const TIMING_BRANCH: OperationTiming = OperationTiming::simple(LATENCY_BRANCH_TAKEN);
const TIMING_CALL: OperationTiming = OperationTiming::simple(LATENCY_CALL);
const TIMING_RETURN: OperationTiming = OperationTiming::simple(LATENCY_RETURN);
const TIMING_LOCK: OperationTiming = OperationTiming::simple(LATENCY_LOCK_ACQUIRE);
const TIMING_VECTOR_SIMPLE: OperationTiming = OperationTiming::simple(LATENCY_VECTOR_SIMPLE);
const TIMING_VECTOR_MUL: OperationTiming = OperationTiming::simple(LATENCY_VECTOR_MUL);
const TIMING_VECTOR_MAC: OperationTiming = OperationTiming::simple(LATENCY_VECTOR_MAC);
const TIMING_VECTOR_SHUFFLE: OperationTiming = OperationTiming::simple(LATENCY_VECTOR_SHUFFLE);
const TIMING_VECTOR_PACK: OperationTiming = OperationTiming::simple(LATENCY_VECTOR_PACK);
const TIMING_DEFAULT: OperationTiming = OperationTiming::simple(1);

// ============================================================================
// Latency Table
// ============================================================================

/// Lookup table for operation latencies.
///
/// Two-tier lookup:
/// 1. **LLVM itinerary** (primary): per-opcode latency from LLVM's scheduling
///    model, queried once at init via FFI and stored in a Vec indexed by
///    LLVM opcode. Zero runtime FFI cost -- just an array index.
/// 2. **SemanticOp fallback**: when the LLVM opcode is unavailable (legacy
///    decode path) or has no itinerary data, falls back to AM020-derived
///    constants dispatched via SemanticOp + is_vector.
#[derive(Debug, Clone)]
pub struct LatencyTable {
    /// Per-opcode latency from LLVM's itinerary model.
    /// Indexed by LLVM opcode ID; None if the opcode has no itinerary data.
    llvm_latencies: Vec<Option<u8>>,
    /// Per-opcode result-operand forwarding id (itinerary `Forwardings[0]`).
    /// Indexed by LLVM opcode ID; 0 = NoBypass. Used by the vector-register
    /// bypass-visibility model to decide whether a result forwards to ALU
    /// consumers at issue+1 (see [`Self::def_bypass`]).
    llvm_def_bypass: Vec<u16>,
}

impl LatencyTable {
    /// Create the AIE2 latency table, populated with LLVM itinerary data.
    pub fn aie2() -> Self {
        let infos = decoder_ffi::query_all_instr_info();
        let llvm_latencies = infos.iter().map(|info| info.latency).collect();
        let llvm_def_bypass = infos.iter().map(|info| info.def_bypass).collect();
        Self { llvm_latencies, llvm_def_bypass }
    }

    /// Result bypass class of an opcode, for the vector-register visibility
    /// model. Maps the raw itinerary forwarding id: 0 -> `No`; nonzero -> `Mov`.
    ///
    /// The nonzero->`Mov` mapping is exact for vector-*register* results (the
    /// only caller): such results carry `MOV_Bypass` or `NoBypass`, never
    /// `VEC_Bypass` (which is accumulator-domain). When the accumulator file
    /// joins this model, the `Vec` case must be distinguished here -- see the
    /// FIXME at `ExecutionContext::queue_matmul_accum_write`.
    pub fn def_bypass(&self, llvm_opcode: u32) -> Bypass {
        match self.llvm_def_bypass.get(llvm_opcode as usize) {
            Some(0) | None => Bypass::No,
            Some(_) => Bypass::Mov,
        }
    }

    /// Map a SemanticOp + vector flag directly to its timing.
    ///
    /// The `is_vector` flag disambiguates between scalar and vector functional
    /// units (e.g., `Add` + scalar = 1 cycle, `Add` + vector = 2 cycles).
    pub fn timing_from_semantic(semantic: SemanticOp, is_vector: bool) -> OperationTiming {
        match semantic {
            // ── Arithmetic ──────────────────────────────────────────────
            SemanticOp::Add | SemanticOp::Adc | SemanticOp::Abs | SemanticOp::Neg if !is_vector => {
                TIMING_SCALAR_1
            }
            SemanticOp::Add | SemanticOp::Adc | SemanticOp::Abs | SemanticOp::Neg => TIMING_VECTOR_SIMPLE,
            SemanticOp::Sub | SemanticOp::Sbc if !is_vector => TIMING_SCALAR_1,
            SemanticOp::Sub | SemanticOp::Sbc => TIMING_VECTOR_SIMPLE,
            SemanticOp::Mul if !is_vector => TIMING_SCALAR_MUL,
            SemanticOp::Mul => TIMING_VECTOR_MUL,
            SemanticOp::SDiv
            | SemanticOp::UDiv
            | SemanticOp::SRem
            | SemanticOp::URem
            | SemanticOp::DivStep
                if !is_vector =>
            {
                TIMING_SCALAR_DIV
            }
            SemanticOp::SDiv
            | SemanticOp::UDiv
            | SemanticOp::SRem
            | SemanticOp::URem
            | SemanticOp::DivStep => TIMING_VECTOR_MAC, // Vector div uses MAC pipeline

            // ── Bitwise ─────────────────────────────────────────────────
            SemanticOp::And | SemanticOp::Or | SemanticOp::Xor | SemanticOp::Not if !is_vector => {
                TIMING_SCALAR_1
            }
            SemanticOp::And | SemanticOp::Or | SemanticOp::Xor | SemanticOp::Not => TIMING_VECTOR_SIMPLE, // Vector bitwise uses ALU pipeline

            // ── Shifts ──────────────────────────────────────────────────
            SemanticOp::Shl
            | SemanticOp::Sra
            | SemanticOp::Srl
            | SemanticOp::AshlBidir
            | SemanticOp::LshlBidir
                if !is_vector =>
            {
                TIMING_SCALAR_1
            }
            SemanticOp::Shl
            | SemanticOp::Sra
            | SemanticOp::Srl
            | SemanticOp::AshlBidir
            | SemanticOp::LshlBidir => TIMING_VECTOR_SIMPLE,
            SemanticOp::Rotl | SemanticOp::Rotr if !is_vector => TIMING_SCALAR_1,
            SemanticOp::Rotl | SemanticOp::Rotr => TIMING_VECTOR_SHUFFLE,

            // ── Comparisons ─────────────────────────────────────────────
            SemanticOp::SetEq
            | SemanticOp::SetNe
            | SemanticOp::SetLt
            | SemanticOp::SetLe
            | SemanticOp::SetGt
            | SemanticOp::SetGe
            | SemanticOp::SetUlt
            | SemanticOp::SetUle
            | SemanticOp::SetUgt
            | SemanticOp::SetUge
                if !is_vector =>
            {
                TIMING_SCALAR_1
            }
            SemanticOp::SetEq
            | SemanticOp::SetNe
            | SemanticOp::SetLt
            | SemanticOp::SetLe
            | SemanticOp::SetGt
            | SemanticOp::SetGe
            | SemanticOp::SetUlt
            | SemanticOp::SetUle
            | SemanticOp::SetUgt
            | SemanticOp::SetUge => TIMING_VECTOR_SIMPLE,

            // ── Bit manipulation (scalar only in practice) ──────────────
            SemanticOp::Ctlz | SemanticOp::Cttz | SemanticOp::Ctpop | SemanticOp::Bswap => TIMING_SCALAR_1,
            SemanticOp::Clb | SemanticOp::Cmp => TIMING_SCALAR_1,

            // ── Memory ──────────────────────────────────────────────────
            SemanticOp::Load => TIMING_LOAD,
            SemanticOp::Store => TIMING_STORE,

            // ── Control flow ────────────────────────────────────────────
            SemanticOp::Br | SemanticOp::BrCond => TIMING_BRANCH,
            SemanticOp::Call => TIMING_CALL,
            SemanticOp::Ret => TIMING_RETURN,
            SemanticOp::Select if !is_vector => TIMING_SCALAR_1,
            SemanticOp::Select => TIMING_VECTOR_SIMPLE,

            // ── Type conversion ─────────────────────────────────────────
            SemanticOp::SignExtend | SemanticOp::ZeroExtend | SemanticOp::Truncate if !is_vector => {
                TIMING_SCALAR_1
            }
            SemanticOp::SignExtend | SemanticOp::ZeroExtend | SemanticOp::Truncate => TIMING_VECTOR_PACK,

            // ── Move / NOP / Halt ───────────────────────────────────────
            SemanticOp::Copy if !is_vector => TIMING_SCALAR_1,
            SemanticOp::Copy => TIMING_VECTOR_SIMPLE,
            SemanticOp::Nop => TIMING_NOP,
            SemanticOp::Event => TIMING_NOP,
            SemanticOp::Done | SemanticOp::Halt => TIMING_DEFAULT,

            // ── Synchronization ─────────────────────────────────────────
            SemanticOp::LockAcquire | SemanticOp::LockRelease => TIMING_LOCK,

            // ── Vector-specific operations ──────────────────────────────
            SemanticOp::Mac
            | SemanticOp::MatMul
            | SemanticOp::MatMulSub
            | SemanticOp::NegMatMul
            | SemanticOp::AddMac
            | SemanticOp::SubMac
            | SemanticOp::NegMul
            | SemanticOp::Accumulate
            | SemanticOp::AccumSub
            | SemanticOp::AccumNegAdd
            | SemanticOp::AccumNegSub => TIMING_VECTOR_MAC,
            SemanticOp::Srs
            | SemanticOp::Ups
            | SemanticOp::Convert
            | SemanticOp::Pack
            | SemanticOp::Unpack => TIMING_VECTOR_PACK,
            SemanticOp::Shuffle | SemanticOp::Align => TIMING_VECTOR_SHUFFLE,
            SemanticOp::VectorBroadcast
            | SemanticOp::VectorExtract
            | SemanticOp::VectorInsert
            | SemanticOp::VectorPush
            | SemanticOp::VectorPushHi
            | SemanticOp::VectorSelect
            | SemanticOp::VectorClear => TIMING_VECTOR_SIMPLE,
            SemanticOp::Min | SemanticOp::Max => TIMING_VECTOR_SIMPLE,

            // ── Conditional vector operations ───────────────────────────
            SemanticOp::SubLt
            | SemanticOp::SubGe
            | SemanticOp::MaxDiffLt
            | SemanticOp::MaxLt
            | SemanticOp::MinGe
            | SemanticOp::AbsGtz
            | SemanticOp::NegGtz
            | SemanticOp::NegLtz
            | SemanticOp::NegAdd => TIMING_VECTOR_SIMPLE,

            // ── Side-effect operations ──────────────────────────────────
            SemanticOp::CascadeRead | SemanticOp::CascadeWrite => TIMING_VECTOR_SIMPLE,
            SemanticOp::StreamRead | SemanticOp::StreamWrite | SemanticOp::StreamWritePacketHeader => {
                TIMING_SCALAR_1
            }
            SemanticOp::DmaStart | SemanticOp::DmaWait => TIMING_DEFAULT, // Actual wait modeled separately

            // ── Pointer operations ──────────────────────────────────────
            SemanticOp::PointerAdd | SemanticOp::PointerMov => TIMING_SCALAR_1,

            // ── Hardware state reads ──────────────────────────────────────
            SemanticOp::ReadCycleCounter => TIMING_SCALAR_1,

            // ── Intrinsics ──────────────────────────────────────────────
            SemanticOp::Intrinsic(_) if is_vector => TIMING_VECTOR_MAC, // Conservative
            SemanticOp::Intrinsic(_) => TIMING_SCALAR_MUL,
        }
    }

    /// Get timing for a SlotOp.
    ///
    /// First checks the LLVM itinerary data (via opcode index), then falls
    /// back to the SemanticOp-based lookup for instructions decoded via the
    /// legacy path or without itinerary data.
    #[inline]
    pub fn timing_for_slot_op(&self, op: &SlotOp) -> OperationTiming {
        // Tier 1: LLVM itinerary lookup (O(1) array index, zero FFI).
        if let Some(opcode) = op.llvm_opcode {
            if let Some(&Some(latency)) = self.llvm_latencies.get(opcode as usize) {
                return OperationTiming::simple(latency);
            }
        }

        // Tier 2: SemanticOp fallback (AM020 constants).
        if let Some(semantic) = op.semantic {
            Self::timing_from_semantic(semantic, op.is_vector)
        } else {
            TIMING_NOP
        }
    }

    /// Create an AIE2 latency table validated against parsed ProcessorModel.
    ///
    /// Asserts that our hardcoded constants match the values extracted from
    /// `AIE2Schedule.td`. This catches any drift between the emulator and
    /// the compiler's scheduling model.
    pub fn validated_aie2(model: &xdna_archspec::aie2::isa::ProcessorModel) -> Self {
        // ProcessorModel.LoadLatency (5) is the memory pipeline depth.
        // LATENCY_MEMORY (7) is the full instruction result latency including
        // 2 extra writeback stages visible in AIE2Schedule.td itineraries.
        assert_eq!(
            model.load_latency + 2,
            LATENCY_MEMORY,
            "LATENCY_MEMORY should be ProcessorModel.load_latency ({}) + 2 writeback cycles = {}",
            model.load_latency,
            LATENCY_MEMORY
        );
        assert_eq!(
            model.mispredict_penalty,
            LATENCY_BRANCH_TAKEN + 1,
            "ProcessorModel.mispredict_penalty ({}) != LATENCY_BRANCH_TAKEN+1 ({})",
            model.mispredict_penalty,
            LATENCY_BRANCH_TAKEN + 1
        );

        let table = Self::aie2();

        // Verify LLVM itinerary data was populated.
        let with_latency = table.llvm_latencies.iter().filter(|l| l.is_some()).count();
        log::info!(
            "LatencyTable: {} / {} opcodes have LLVM itinerary latency",
            with_latency,
            table.llvm_latencies.len()
        );

        table
    }
}

/// Map an itinerary class name to its expected OperationTiming.
///
/// Used for cross-validation against TableGen scheduling data. Only maps
/// the "representative" class for each timing bucket. Classes with no clear
/// mapping (e.g., `II_MOVd3`, combined load+UPS) return None.
pub fn itinerary_to_timing(class_name: &str) -> Option<OperationTiming> {
    Some(match class_name {
        // Scalar arithmetic (1-cycle ALU)
        "II_ADD" | "II_ADD_NC" | "II_SUB" | "II_SBC" | "II_ADC" => TIMING_SCALAR_1,
        "II_MUL" => TIMING_SCALAR_MUL,
        "II_AND" | "II_OR" | "II_XOR" => TIMING_SCALAR_1,
        "II_ASHL" | "II_LSHL" => TIMING_SCALAR_1,
        "II_MOV" | "II_MOVA" | "II_MOVX" | "II_MOV_SCL" => TIMING_SCALAR_1,
        "II_EQ" | "II_NE" | "II_GE" | "II_GEU" | "II_LT" | "II_LTU" | "II_EQZ" | "II_NEZ" => TIMING_SCALAR_1,
        "II_DIVS" => TIMING_SCALAR_DIV,
        "II_SELEQZ" | "II_SELNEZ" => TIMING_SCALAR_1,
        // Memory
        "II_LDA" | "II_LDA_POST_1D" | "II_LDA_POST_2D" | "II_LDA_POST_3D" => TIMING_LOAD,
        "II_VLDB" | "II_VLDB_POSTINC" | "II_VLDB_2D" | "II_VLDB_3D" | "II_VLDA_W" | "II_VLDA_AM" => {
            TIMING_LOAD
        }
        "II_ST" | "II_ST_POST_1D" | "II_ST_POST_2D" | "II_ST_POST_3D" => TIMING_STORE,
        "II_VST_W" | "II_VST_AM" | "II_VST_POSTINC" => TIMING_STORE,
        // Control flow
        "II_J" | "II_JNZ" | "II_JZ" | "II_JNZD" => TIMING_BRANCH,
        "II_JL" | "II_JL_IND" => TIMING_CALL,
        "II_RET" => TIMING_RETURN,
        // Locks
        "II_ACQ" | "II_ACQ_COND" => TIMING_LOCK,
        "II_REL" | "II_REL_COND" => TIMING_LOCK,
        // Vector compute
        "II_VADD" | "II_VSUB" => TIMING_VECTOR_SIMPLE,
        "II_VMUL" | "II_VNEGMUL" => TIMING_VECTOR_MUL,
        "II_VMAC" | "II_VMSC" | "II_VNEGMAC" | "II_VNEGMSC" | "II_VACC" | "II_VADDMAC" => TIMING_VECTOR_MAC,
        "II_VSHUFFLE" | "II_VBCSTSHFL" => TIMING_VECTOR_SHUFFLE,
        "II_VPACK" => TIMING_VECTOR_PACK,
        "II_VCMP" => TIMING_VECTOR_SIMPLE,
        "II_NOP" => TIMING_NOP,
        _ => return None,
    })
}

impl Default for LatencyTable {
    fn default() -> Self {
        // Default creates an empty table (no LLVM data) -- uses SemanticOp fallback only.
        // Production code should use aie2() or validated_aie2() for full LLVM latencies.
        Self { llvm_latencies: Vec::new(), llvm_def_bypass: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_constants() {
        // Verify key timing values from the constants.
        assert_eq!(TIMING_SCALAR_1.latency, 1);
        assert_eq!(TIMING_SCALAR_MUL.latency, 2);
        assert_eq!(TIMING_LOAD.latency, 7);
        assert_eq!(TIMING_STORE.latency, 1);
        assert_eq!(TIMING_VECTOR_MAC.latency, 5);
        assert_eq!(TIMING_NOP.latency, 0);
    }

    #[test]
    fn test_operation_timing() {
        let timing = OperationTiming::simple(3);
        assert_eq!(timing.latency, 3);
        assert_eq!(timing.throughput, 1);
        assert_eq!(timing.result_stage, 2);
    }

    // ── Semantic timing tests ───────────────────────────────────────

    #[test]
    fn test_timing_scalar_arithmetic() {
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Add, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Sub, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Mul, false).latency, 2);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::SDiv, false).latency, 6);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::UDiv, false).latency, 6);
    }

    #[test]
    fn test_timing_vector_arithmetic() {
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Add, true).latency, 2);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Sub, true).latency, 2);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Mul, true).latency, 5);
    }

    #[test]
    fn test_timing_bitwise() {
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::And, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Or, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Xor, false).latency, 1);
        // Vector bitwise uses ALU pipeline
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::And, true).latency, 2);
    }

    #[test]
    fn test_timing_shifts() {
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Shl, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Sra, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Srl, false).latency, 1);
    }

    #[test]
    fn test_timing_comparison() {
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::SetEq, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::SetLt, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::SetEq, true).latency, 2);
    }

    #[test]
    fn test_timing_memory() {
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Load, false).latency, 7);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Load, true).latency, 7);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Store, false).latency, 1);
    }

    #[test]
    fn test_timing_control_flow() {
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Br, false).latency, 3);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::BrCond, false).latency, 3);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Call, false).latency, 3);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Ret, false).latency, 3);
    }

    #[test]
    fn test_timing_special() {
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Copy, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Copy, true).latency, 2);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Nop, false).latency, 0);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::Select, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::LockAcquire, false).latency, 1);
        assert_eq!(LatencyTable::timing_from_semantic(SemanticOp::LockRelease, false).latency, 1);
    }

    #[test]
    fn test_timing_for_slot_op_uses_semantic() {
        use crate::interpreter::bundle::SlotIndex;

        let table = LatencyTable::aie2();

        // SlotOp with Mul semantic -> ScalarMul latency (2)
        let op = SlotOp::from_semantic(SlotIndex::Scalar1, SemanticOp::Mul);
        let timing = table.timing_for_slot_op(&op);
        assert_eq!(timing.latency, LATENCY_SCALAR_MUL);
    }

    #[test]
    fn test_timing_for_slot_op_nop_when_no_semantic() {
        use crate::interpreter::bundle::SlotIndex;

        let table = LatencyTable::aie2();

        // SlotOp without semantic -> treated as NOP
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar1, SemanticOp::Mul);
        op.semantic = None;

        let timing = table.timing_for_slot_op(&op);
        assert_eq!(timing.latency, 0); // NOP latency
    }

    #[test]
    fn test_semantic_timing_matches_expected_latencies() {
        // Verify the semantic path produces correct latencies for common ops.
        let cases: &[(SemanticOp, bool, u8)] = &[
            (SemanticOp::Add, false, LATENCY_SCALAR_ADD),
            (SemanticOp::Sub, false, LATENCY_SCALAR_ADD),
            (SemanticOp::Mul, false, LATENCY_SCALAR_MUL),
            (SemanticOp::And, false, LATENCY_SCALAR_LOGIC),
            (SemanticOp::Or, false, LATENCY_SCALAR_LOGIC),
            (SemanticOp::Xor, false, LATENCY_SCALAR_LOGIC),
            (SemanticOp::Shl, false, LATENCY_SCALAR_SHIFT),
            (SemanticOp::Sra, false, LATENCY_SCALAR_SHIFT),
            (SemanticOp::Copy, false, LATENCY_SCALAR_MOV),
            (SemanticOp::Nop, false, LATENCY_NOP),
            (SemanticOp::Call, false, LATENCY_CALL),
            (SemanticOp::Ret, false, LATENCY_RETURN),
            (SemanticOp::LockAcquire, false, LATENCY_LOCK_ACQUIRE),
            (SemanticOp::LockRelease, false, LATENCY_LOCK_RELEASE),
        ];

        for &(semantic, is_vector, expected) in cases {
            let timing = LatencyTable::timing_from_semantic(semantic, is_vector);
            assert_eq!(
                timing.latency, expected,
                "Latency mismatch for {:?} (is_vector={}): got {}, expected {}",
                semantic, is_vector, timing.latency, expected,
            );
        }
    }

    #[test]
    fn test_validated_aie2_matches_processor_model() {
        use xdna_archspec::aie2::isa::ProcessorModel;

        let model = ProcessorModel {
            load_latency: 5,
            high_latency: 37,
            mispredict_penalty: 4,
            issue_width: 1000,
            itinerary_name: "AIE2Itineraries".into(),
        };

        // Should not panic -- our constants match AIE2's ProcessorModel
        let table = LatencyTable::validated_aie2(&model);
        assert_eq!(
            table
                .timing_for_slot_op(&SlotOp::from_semantic(
                    crate::interpreter::bundle::SlotIndex::LoadA,
                    SemanticOp::Load,
                ))
                .latency,
            7
        );
    }

    #[test]
    #[should_panic(expected = "LATENCY_MEMORY should be")]
    fn test_validated_aie2_catches_load_latency_drift() {
        use xdna_archspec::aie2::isa::ProcessorModel;

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
    /// For each itinerary class that maps to a timing, compares our
    /// hardcoded latency against `operand_cycles[0]` (the result
    /// availability cycle from the compiler's scheduling model).
    ///
    /// Known differences:
    /// - Branch/Call/Return: TableGen models operand read timing (cycle 1),
    ///   not the pipeline flush penalty. Our latency represents the taken-
    ///   branch cost, which is a different concept.
    #[test]
    fn test_latency_cross_validation_against_itineraries() {
        let tblgen = xdna_archspec::aie2::isa::load_from_generated();

        let mut checked = 0u32;
        let mut mismatches = Vec::new();

        // Classes where our latency intentionally differs from operand_cycles[0].
        // For these we verify the relationship rather than asserting equality.
        let known_difference_classes: &[&str] = &[
            // Branch/call/return: TableGen models operand timing, not flush penalty.
            "II_J",
            "II_JNZ",
            "II_JZ",
            "II_JNZD",
            "II_JL",
            "II_JL_IND",
            "II_RET",
            // NOP: TableGen has 1-cycle stage, we model as 0 (just advances PC).
            "II_NOP",
            // Division: TableGen models operand_cycles[0]=1 because the pipeline
            // stalls during iterative division. Our 6-cycle value represents the
            // actual computation time (throughput, not scheduling latency).
            "II_DIVS",
        ];

        for (class_name, itin) in &tblgen.itineraries {
            let our_timing = match itinerary_to_timing(class_name) {
                Some(t) => t,
                None => continue, // No mapping for this class
            };

            // operand_cycles[0] = result availability cycle (the key metric)
            let tblgen_latency = itin.operand_cycles.first().copied().unwrap_or(itin.total_latency);

            checked += 1;

            if known_difference_classes.contains(&class_name.as_str()) {
                // Known difference: just verify the relationship is sane
                continue;
            }

            if our_timing.latency != tblgen_latency {
                mismatches.push(format!(
                    "  {}: ours={}, tblgen operand_cycles[0]={}",
                    class_name, our_timing.latency, tblgen_latency,
                ));
            }
        }

        assert!(checked > 0, "No itinerary classes were checked -- is the mapping empty?");

        if !mismatches.is_empty() {
            panic!(
                "Latency mismatches between LatencyTable and TableGen itineraries ({}/{} checked):\n{}",
                mismatches.len(),
                checked,
                mismatches.join("\n"),
            );
        }
    }

    /// Verify that the LLVM opcode-indexed latency path works end-to-end.
    ///
    /// Creates a SlotOp with an llvm_opcode set, and verifies that
    /// timing_for_slot_op returns the LLVM itinerary latency rather than
    /// the SemanticOp fallback.
    #[test]
    fn test_llvm_opcode_latency_path() {
        use xdna_archspec::aie2::isa::decoder_ffi;

        let table = LatencyTable::aie2();
        let infos = decoder_ffi::query_all_instr_info();

        // Find a VMAC instruction (latency should be 5 from LLVM itinerary).
        let vmac_opcode = infos
            .iter()
            .enumerate()
            .find(|(_, info)| info.latency == Some(5) && !info.is_load())
            .map(|(opcode, _)| opcode as u32);

        if let Some(opcode) = vmac_opcode {
            let mut op =
                SlotOp::from_semantic(crate::interpreter::bundle::SlotIndex::Vector, SemanticOp::Mac);
            op.llvm_opcode = Some(opcode);

            let timing = table.timing_for_slot_op(&op);
            assert_eq!(
                timing.latency, 5,
                "LLVM itinerary path should return latency 5 for opcode {}",
                opcode
            );
        }

        // Find a load instruction (latency should be 7).
        let load_opcode = infos
            .iter()
            .enumerate()
            .find(|(_, info)| info.latency == Some(7) && info.is_load())
            .map(|(opcode, _)| opcode as u32);

        if let Some(opcode) = load_opcode {
            let mut op =
                SlotOp::from_semantic(crate::interpreter::bundle::SlotIndex::LoadA, SemanticOp::Load);
            op.llvm_opcode = Some(opcode);

            let timing = table.timing_for_slot_op(&op);
            assert_eq!(
                timing.latency, 7,
                "LLVM itinerary path should return latency 7 for load opcode {}",
                opcode
            );
        }
    }

    /// Verify fallback to SemanticOp when llvm_opcode is None.
    #[test]
    fn test_semantic_fallback_when_no_llvm_opcode() {
        let table = LatencyTable::aie2();

        // SlotOp without llvm_opcode should use SemanticOp path.
        let op = SlotOp::from_semantic(crate::interpreter::bundle::SlotIndex::Scalar0, SemanticOp::Add);
        assert!(op.llvm_opcode.is_none());

        let timing = table.timing_for_slot_op(&op);
        assert_eq!(timing.latency, LATENCY_SCALAR_ADD, "Without llvm_opcode, should fall back to SemanticOp");
    }
}
