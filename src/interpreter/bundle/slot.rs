//! VLIW slot definitions.
//!
//! AIE2 uses a 7-slot VLIW architecture where multiple operations can execute
//! in parallel within a single cycle. Each slot has specific capabilities:
//!
//! | Slot | Name | Operations |
//! |------|------|------------|
//! | 0 | Scalar0 | Integer ALU, shifts, logic |
//! | 1 | Scalar1 | Integer ALU, multiply |
//! | 2 | Vector | Vector ALU, shuffle |
//! | 3 | Accumulator | MAC, accumulator ops |
//! | 4 | Load | Memory loads |
//! | 5 | Store | Memory stores |
//! | 6 | Control | Branches, calls, locks |
//!
//! Not all slot combinations are valid. The decoder validates slot compatibility.

use smallvec::SmallVec;

// TableGen-derived semantic information and shared types
use crate::tablegen::{ImplicitReg, SemanticOp};

// Re-export types defined in tablegen::types so existing code using
// crate::interpreter::bundle::{ElementType, BranchCondition, SelectVariant}
// continues to compile without changes.
pub use crate::tablegen::{BranchCondition, ElementType, SelectVariant};

/// Slot index within a VLIW bundle.
///
/// AIE2 has 8 execution slots that can potentially operate in parallel.
/// The LDA and LDB slots in 128-bit bundles are independent load ports
/// with separate bit fields (21-bit LDA, 16-bit LDB) that can both
/// issue in the same cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SlotIndex {
    /// Scalar ALU slot 0 - primary integer operations.
    Scalar0 = 0,
    /// Scalar ALU slot 1 - secondary integer operations, multiply.
    Scalar1 = 1,
    /// Vector ALU slot - SIMD operations on 256-bit vectors.
    Vector = 2,
    /// Accumulator slot - MAC operations with 512-bit accumulators.
    Accumulator = 3,
    /// Load A slot - primary memory read port (LDA in VLIW encoding).
    LoadA = 4,
    /// Store slot - memory write operations.
    Store = 5,
    /// Control slot - branches, calls, returns, lock operations.
    Control = 6,
    /// Load B slot - secondary memory read port (LDB in VLIW encoding).
    /// Only present in 128-bit bundles; shares the load execution unit
    /// with LoadA but has independent operands.
    LoadB = 7,
}

impl SlotIndex {
    /// Total number of slots in a VLIW bundle.
    pub const COUNT: usize = 8;

    /// Get all slot indices.
    pub fn all() -> [SlotIndex; 8] {
        [
            SlotIndex::Scalar0,
            SlotIndex::Scalar1,
            SlotIndex::Vector,
            SlotIndex::Accumulator,
            SlotIndex::LoadA,
            SlotIndex::Store,
            SlotIndex::Control,
            SlotIndex::LoadB,
        ]
    }

    /// Check if this is a scalar slot.
    #[inline]
    pub fn is_scalar(self) -> bool {
        matches!(self, SlotIndex::Scalar0 | SlotIndex::Scalar1)
    }

    /// Check if this is a memory slot.
    #[inline]
    pub fn is_memory(self) -> bool {
        matches!(self, SlotIndex::LoadA | SlotIndex::LoadB | SlotIndex::Store)
    }

    /// Check if this is a load slot (either port).
    #[inline]
    pub fn is_load(self) -> bool {
        matches!(self, SlotIndex::LoadA | SlotIndex::LoadB)
    }
}

/// Memory access width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemWidth {
    /// 8-bit byte access.
    Byte,
    /// 16-bit halfword access.
    HalfWord,
    /// 32-bit word access.
    Word,
    /// 64-bit doubleword access.
    DoubleWord,
    /// 128-bit quadword access.
    QuadWord,
    /// 256-bit vector access.
    Vector256,
}

impl MemWidth {
    /// Get the size in bytes.
    pub fn bytes(self) -> u8 {
        match self {
            MemWidth::Byte => 1,
            MemWidth::HalfWord => 2,
            MemWidth::Word => 4,
            MemWidth::DoubleWord => 8,
            MemWidth::QuadWord => 16,
            MemWidth::Vector256 => 32,
        }
    }
}

/// Post-modify mode for addressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PostModify {
    /// No modification to base register.
    #[default]
    None,
    /// Add immediate to base register after access.
    Immediate(i16),
    /// Add modifier register to base register after access.
    Register(u8),
}

/// Shuffle pattern for vector permute operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum ShufflePattern {
    /// No shuffle (identity).
    #[default]
    Identity,
    /// Reverse elements.
    Reverse,
    /// Interleave low halves.
    InterleaveLow,
    /// Interleave high halves.
    InterleaveHigh,
    /// Broadcast single element.
    Broadcast(u8),
    /// Custom pattern (encoded).
    Custom(u32),
}


/// Unified operation enum covering all AIE2 instruction types.
///
/// This enum represents the decoded operation from an instruction.
/// Each variant contains the operation-specific parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    // ========== Scalar Operations ==========
    /// Scalar addition: dst = src1 + src2
    ScalarAdd,
    /// Scalar subtraction: dst = src1 - src2
    ScalarSub,
    /// Scalar multiplication: dst = src1 * src2
    ScalarMul,
    /// Scalar bitwise AND: dst = src1 & src2
    ScalarAnd,
    /// Scalar bitwise OR: dst = src1 | src2
    ScalarOr,
    /// Scalar bitwise XOR: dst = src1 ^ src2
    ScalarXor,
    /// Scalar left shift: dst = src1 << src2
    ScalarShl,
    /// Scalar logical right shift: dst = src1 >> src2 (unsigned)
    ScalarShr,
    /// Scalar arithmetic right shift: dst = src1 >> src2 (signed)
    ScalarSra,
    /// Scalar move: dst = src
    ScalarMov,
    /// Scalar move immediate: dst = imm
    ScalarMovi {
        /// Immediate value.
        value: i32,
    },
    /// Scalar compare (sets flags): flags = cmp(src1, src2)
    ScalarCmp,

    /// Scalar absolute value: dst = |src|
    ScalarAbs,
    /// Count leading zeros: dst = clz(src)
    ScalarClz,
    /// Count leading bits (ones or zeros): dst = clb(src)
    ScalarClb,
    /// Add with carry: dst = src1 + src2 + C
    ScalarAdc,
    /// Subtract with borrow: dst = src1 - src2 - !C
    ScalarSbc,
    /// Sign extend from 8 bits: dst = sign_extend(src[7:0])
    ScalarExtendS8,
    /// Sign extend from 16 bits: dst = sign_extend(src[15:0])
    ScalarExtendS16,
    /// Zero extend from 8 bits: dst = zero_extend(src[7:0])
    ScalarExtendU8,
    /// Zero extend from 16 bits: dst = zero_extend(src[15:0])
    ScalarExtendU16,

    // ========== Comparison Operations (produce 0/1) ==========
    /// Signed less than: dst = (src1 < src2) ? 1 : 0
    ScalarLt,
    /// Unsigned less than: dst = (src1 < src2) ? 1 : 0 (unsigned)
    ScalarLtu,
    /// Signed less than or equal: dst = (src1 <= src2) ? 1 : 0
    ScalarLe,
    /// Unsigned less than or equal: dst = (src1 <= src2) ? 1 : 0 (unsigned)
    ScalarLeu,
    /// Signed greater than: dst = (src1 > src2) ? 1 : 0
    ScalarGt,
    /// Unsigned greater than: dst = (src1 > src2) ? 1 : 0 (unsigned)
    ScalarGtu,
    /// Signed greater than or equal: dst = (src1 >= src2) ? 1 : 0
    ScalarGe,
    /// Unsigned greater than or equal: dst = (src1 >= src2) ? 1 : 0 (unsigned)
    ScalarGeu,
    /// Equal: dst = (src1 == src2) ? 1 : 0
    ScalarEq,
    /// Not equal: dst = (src1 != src2) ? 1 : 0
    ScalarNe,

    /// Conditional select: dst = cond ? src1 : src2
    ScalarSel,
    /// Signed integer division: dst = src1 / src2
    ScalarDiv,
    /// Unsigned integer division: dst = src1 / src2
    ScalarDivu,
    /// Modulo: dst = src1 % src2
    ScalarMod,
    /// Select if equal zero: dst = (cond == 0) ? true_val : false_val
    ScalarSelEqz,
    /// Select if not equal zero: dst = (cond != 0) ? true_val : false_val
    ScalarSelNez,

    // ========== Vector Operations ==========
    /// Vector addition: vdst = vsrc1 + vsrc2
    VectorAdd {
        /// Element type for the operation.
        element_type: ElementType,
    },
    /// Vector subtraction: vdst = vsrc1 - vsrc2
    VectorSub {
        /// Element type for the operation.
        element_type: ElementType,
    },
    /// Vector multiplication: vdst = vsrc1 * vsrc2
    VectorMul {
        /// Element type for the operation.
        element_type: ElementType,
    },
    /// Vector multiply-accumulate: acc += vsrc1 * vsrc2
    VectorMac {
        /// Element type for the operation.
        element_type: ElementType,
    },
    /// Vector shuffle/permute: vdst = shuffle(vsrc, pattern)
    VectorShuffle {
        /// Shuffle pattern to apply.
        pattern: ShufflePattern,
    },
    /// Vector pack (narrow elements).
    VectorPack,
    /// Vector unpack (widen elements).
    VectorUnpack,
    /// Vector compare.
    VectorCmp {
        /// Element type for comparison.
        element_type: ElementType,
    },
    /// Vector minimum.
    VectorMin {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector maximum.
    VectorMax {
        /// Element type.
        element_type: ElementType,
    },

    // ========== Vector Comparison Operations ==========
    /// Vector greater-equal: dst[i] = (a[i] >= b[i]) ? 1 : 0
    VectorGe {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector less-than: dst[i] = (a[i] < b[i]) ? 1 : 0
    VectorLt {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector equal-to-zero: dst[i] = (a[i] == 0) ? 1 : 0
    VectorEqz {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector max with less-than: dst[i] = (a[i] < b[i]) ? max(a[i], c[i]) : a[i]
    VectorMaxLt {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector min with greater-equal: dst[i] = (a[i] >= b[i]) ? min(a[i], c[i]) : a[i]
    VectorMinGe {
        /// Element type.
        element_type: ElementType,
    },

    // ========== Vector Bitwise Operations ==========
    /// Vector bitwise AND: dst[i] = a[i] & b[i]
    VectorAnd {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector bitwise OR: dst[i] = a[i] | b[i]
    VectorOr {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector bitwise XOR: dst[i] = a[i] ^ b[i]
    VectorXor {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector bitwise NOT: dst[i] = ~a[i]
    VectorNot {
        /// Element type.
        element_type: ElementType,
    },

    // ========== Vector Conditional Arithmetic ==========
    /// Vector subtract if less-than: dst[i] = (a[i] < b[i]) ? a[i] - c[i] : a[i]
    VectorSubLt {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector subtract if greater-equal: dst[i] = (a[i] >= b[i]) ? a[i] - c[i] : a[i]
    VectorSubGe {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector max difference if less-than: dst[i] = (a[i] < b[i]) ? max(a[i]-c[i], 0) : a[i]
    VectorMaxDiffLt {
        /// Element type.
        element_type: ElementType,
    },

    // ========== Conditional Vector Operations ==========

    /// Absolute value if greater than zero: dst[i] = (src[i] > 0) ? abs(src[i]) : src[i]
    /// Used for ReLU-like activations.
    VectorAbsGtz {
        /// Element type.
        element_type: ElementType,
    },

    /// Negate if greater than zero: dst[i] = (src[i] > 0) ? -src[i] : src[i]
    VectorNegGtz {
        /// Element type.
        element_type: ElementType,
    },

    /// Negate if less than zero: dst[i] = (src[i] < 0) ? -src[i] : src[i]
    /// This is essentially abs().
    VectorNegLtz {
        /// Element type.
        element_type: ElementType,
    },

    /// Vector accumulate: acc += src (no multiply, just add to accumulator).
    VectorAccumulate {
        /// Element type.
        element_type: ElementType,
    },

    /// Vector negate: dst = -src (per element negation).
    VectorNegate {
        /// Element type.
        element_type: ElementType,
    },

    /// Vector negate and add: dst = -src1 + src2.
    VectorNegAdd {
        /// Element type.
        element_type: ElementType,
    },

    /// Vector negate multiply: acc += -(src1 * src2) but with single operand negation.
    VectorNegMul {
        /// Element type.
        element_type: ElementType,
    },

    // ========== Matrix/Accumulator Operations ==========
    //
    // AIE2 Convolution Support:
    // -------------------------
    // AIE2 does NOT have dedicated "conv2d" instructions. Instead, convolutions
    // are implemented using matrix multiply-accumulate (VMAC) instructions combined
    // with DMA n-dimensional addressing for data layout.
    //
    // The convolution workflow:
    // 1. DMA loads activation windows using stride patterns
    // 2. Weights are loaded (optionally with on-the-fly sparse decompression)
    // 3. VMAC/VMSC perform multiply-accumulate on matrix tiles
    // 4. VSRS converts accumulator results back to vector format
    //
    // Instruction variants:
    // - VMAC: acc += A * B (dense or sparse formats)
    // - VMSC: acc -= A * B (subtract variant)
    // - VNEGMAC: acc += -(A * B) (negated product)
    // - VNEGMSC: acc -= -(A * B) (negated subtract)
    // - VMAC.f / VMSC.f: BFloat16 floating-point variants

    /// Matrix multiply (dense): acc = A * B
    /// Used by VMUL_vmac_cm_core_dense instructions.
    VectorMatMulDense {
        /// Element type for the multiplication.
        element_type: ElementType,
    },
    /// Matrix multiply (sparse): acc = sparse(A) * B
    /// Used by VMUL_vmac_cm_core_sparse_* instructions.
    VectorMatMulSparse {
        /// Element type for the multiplication.
        element_type: ElementType,
        /// Wide or narrow sparse format.
        wide: bool,
    },
    /// Matrix multiply-subtract (dense): acc -= A * B
    /// Used by VMSC instructions for convolution with subtraction.
    VectorMatMulSubDense {
        /// Element type for the multiplication.
        element_type: ElementType,
    },
    /// Matrix multiply-subtract (sparse): acc -= sparse(A) * B
    /// Used by VMSC_vmac_cm_core_sparse_* instructions.
    VectorMatMulSubSparse {
        /// Element type for the multiplication.
        element_type: ElementType,
        /// Wide or narrow sparse format.
        wide: bool,
    },
    /// Negated matrix multiply (dense): acc += -(A * B)
    /// Used by VNEGMAC instructions.
    VectorNegMatMulDense {
        /// Element type for the multiplication.
        element_type: ElementType,
    },
    /// Negated matrix multiply-subtract (dense): acc -= -(A * B)
    /// Used by VNEGMSC instructions.
    VectorNegMatMulSubDense {
        /// Element type for the multiplication.
        element_type: ElementType,
    },
    /// BFloat16 matrix multiply-accumulate: acc += A * B (bf16 operands, fp32 accumulator)
    /// Used by VMAC.f instructions for CNN/ML workloads.
    VectorMatMulAccFloat {
        /// Always BFloat16 for inputs, Float32 for accumulator.
        element_type: ElementType,
    },
    /// BFloat16 matrix multiply-subtract: acc -= A * B (bf16 operands, fp32 accumulator)
    /// Used by VMSC.f instructions.
    VectorMatMulSubFloat {
        /// Always BFloat16 for inputs, Float32 for accumulator.
        element_type: ElementType,
    },
    /// Double accumulator matrix multiply-add: acc1 = acc1 + acc2 + A * B
    /// Used by VADDMAC instructions for fused operations.
    VectorAddMac {
        /// Element type for the multiplication.
        element_type: ElementType,
    },
    /// Double accumulator matrix multiply-subtract: acc1 = acc1 - acc2 + A * B
    /// Used by VSUBMAC instructions for fused operations.
    VectorSubMac {
        /// Element type for the multiplication.
        element_type: ElementType,
    },
    /// Shift-round-saturate: vdst = srs(acc, shift)
    /// Converts accumulator to vector with rounding.
    VectorSRS {
        /// Source accumulator element type.
        from_type: ElementType,
        /// Destination vector element type.
        to_type: ElementType,
    },
    /// Type conversion (e.g., bf16 <-> f32).
    VectorConvert {
        /// Source element type.
        from_type: ElementType,
        /// Destination element type.
        to_type: ElementType,
    },
    /// Vector move (register to register within vector file).
    VectorMov {
        /// Element type.
        element_type: ElementType,
    },

    // ========== Vector Element Operations ==========
    /// Extract single element from vector to scalar.
    VectorExtract {
        /// Element type.
        element_type: ElementType,
    },
    /// Insert scalar into vector lane.
    VectorInsert {
        /// Element type.
        element_type: ElementType,
    },
    /// Per-lane conditional select: dst[i] = sel[i] ? a[i] : b[i].
    VectorSelect {
        /// Element type.
        element_type: ElementType,
    },
    /// Clear vector register to zero.
    VectorClear,
    /// Broadcast scalar value to all vector lanes.
    VectorBroadcast {
        /// Element type.
        element_type: ElementType,
    },

    // ========== Vector Shift Operations ==========
    /// Vector logical left shift: dst[i] = a[i] << b[i].
    VectorShiftLeft {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector logical right shift: dst[i] = a[i] >> b[i].
    VectorShiftRight {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector arithmetic right shift: dst[i] = (signed)a[i] >> b[i].
    VectorArithShiftRight {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector align: concatenate and shift two vectors.
    VectorAlign {
        /// Element type.
        element_type: ElementType,
    },
    /// Vector upshift: shift left with rounding for precision scaling.
    VectorUpshift {
        /// Source element type.
        from_type: ElementType,
        /// Destination element type.
        to_type: ElementType,
    },

    // ========== Pointer Operations ==========
    /// Pointer add: ptr = ptr + offset (for address generation).
    /// Used by padda, paddb, padds instructions.
    PointerAdd,
    /// Pointer move: ptr = value.
    /// Used by mova, movb instructions.
    PointerMov,

    // ========== Memory Operations ==========
    /// Load from memory.
    Load {
        /// Width of the access.
        width: MemWidth,
        /// Post-modify behavior.
        post_modify: PostModify,
    },
    /// Store to memory.
    Store {
        /// Width of the access.
        width: MemWidth,
        /// Post-modify behavior.
        post_modify: PostModify,
    },
    /// Vector load A channel (VLDA).
    VectorLoadA {
        /// Post-modify behavior.
        post_modify: PostModify,
    },
    /// Vector load B channel (VLDB).
    VectorLoadB {
        /// Post-modify behavior.
        post_modify: PostModify,
    },
    /// Vector load with unpack (VLDB_UNPACK).
    VectorLoadUnpack {
        /// Source element type (packed).
        from_type: ElementType,
        /// Destination element type (unpacked).
        to_type: ElementType,
        /// Post-modify behavior.
        post_modify: PostModify,
    },
    /// Vector store (VST).
    VectorStore {
        /// Post-modify behavior.
        post_modify: PostModify,
    },

    // ========== Control Operations ==========
    /// Conditional branch.
    Branch {
        /// Branch condition.
        condition: BranchCondition,
    },
    /// Call subroutine.
    Call,
    /// Return from subroutine.
    Return,

    // ========== Synchronization Operations ==========
    /// Acquire lock (blocking if unavailable).
    LockAcquire,
    /// Release lock.
    LockRelease,
    /// DMA start.
    DmaStart,
    /// DMA wait for completion.
    DmaWait,

    // ========== Cascade Operations ==========
    /// Read from cascade input (SCD) to vector/accumulator register.
    /// Implements get_scd_v32int32() / get_scd() intrinsics.
    /// vmov dst, SCD -- reads 384-bit cascade link.
    CascadeRead,
    /// Write vector/accumulator register to cascade output (MCD).
    /// Implements put_mcd() intrinsic.
    /// vmov MCD, src -- writes 384-bit cascade link.
    CascadeWrite,

    // ========== Stream Operations ==========
    /// Write scalar to master stream (MOV_mv_scl2ms).
    StreamWriteScalar {
        /// Blocking or non-blocking.
        blocking: bool,
    },
    /// Write packet header to master stream (MOV_mv_ph2ms).
    StreamWritePacketHeader {
        /// Blocking or non-blocking.
        blocking: bool,
    },
    /// Read from slave stream to scalar (MOV_mv_ss2scl).
    StreamReadScalar {
        /// Blocking or non-blocking.
        blocking: bool,
    },

    // ========== Misc ==========
    /// No operation.
    Nop,
    /// Halt execution.
    Halt,
    /// Unknown/unrecognized instruction.
    Unknown {
        /// Raw opcode value.
        opcode: u32,
    },
}

impl Operation {
    /// Check if this operation is a control flow operation.
    #[inline]
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            Operation::Branch { .. } | Operation::Call | Operation::Return
        )
    }

    /// Check if this operation is a memory operation.
    #[inline]
    pub fn is_memory(&self) -> bool {
        matches!(self, Operation::Load { .. } | Operation::Store { .. })
    }

    /// Check if this operation is a synchronization operation.
    #[inline]
    pub fn is_sync(&self) -> bool {
        matches!(
            self,
            Operation::LockAcquire
                | Operation::LockRelease
                | Operation::DmaStart
                | Operation::DmaWait
        )
    }

    /// Check if this is a no-op.
    #[inline]
    pub fn is_nop(&self) -> bool {
        matches!(self, Operation::Nop)
    }

    /// Get the natural slot for this operation.
    pub fn natural_slot(&self) -> SlotIndex {
        match self {
            // Scalar ALU operations (Scalar0 slot)
            Operation::ScalarAdd
            | Operation::ScalarSub
            | Operation::ScalarAnd
            | Operation::ScalarOr
            | Operation::ScalarXor
            | Operation::ScalarShl
            | Operation::ScalarShr
            | Operation::ScalarSra
            | Operation::ScalarMov
            | Operation::ScalarMovi { .. }
            | Operation::ScalarCmp
            | Operation::ScalarAbs
            | Operation::ScalarClz
            | Operation::ScalarClb
            | Operation::ScalarAdc
            | Operation::ScalarSbc
            | Operation::ScalarExtendS8
            | Operation::ScalarExtendS16
            | Operation::ScalarExtendU8
            | Operation::ScalarExtendU16
            | Operation::ScalarLt
            | Operation::ScalarLtu
            | Operation::ScalarLe
            | Operation::ScalarLeu
            | Operation::ScalarGt
            | Operation::ScalarGtu
            | Operation::ScalarGe
            | Operation::ScalarGeu
            | Operation::ScalarEq
            | Operation::ScalarNe
            | Operation::ScalarSel
            | Operation::ScalarDiv
            | Operation::ScalarDivu
            | Operation::ScalarMod
            | Operation::ScalarSelEqz
            | Operation::ScalarSelNez => SlotIndex::Scalar0,

            // Scalar multiply (Scalar1 slot)
            Operation::ScalarMul => SlotIndex::Scalar1,

            // Vector ALU operations (Vector slot)
            Operation::VectorAdd { .. }
            | Operation::VectorSub { .. }
            | Operation::VectorMul { .. }
            | Operation::VectorShuffle { .. }
            | Operation::VectorPack
            | Operation::VectorUnpack
            | Operation::VectorCmp { .. }
            | Operation::VectorMin { .. }
            | Operation::VectorMax { .. }
            // Vector comparison operations
            | Operation::VectorGe { .. }
            | Operation::VectorLt { .. }
            | Operation::VectorEqz { .. }
            | Operation::VectorMaxLt { .. }
            | Operation::VectorMinGe { .. }
            // Vector bitwise operations
            | Operation::VectorAnd { .. }
            | Operation::VectorOr { .. }
            | Operation::VectorXor { .. }
            | Operation::VectorNot { .. }
            // Vector conditional arithmetic
            | Operation::VectorSubLt { .. }
            | Operation::VectorSubGe { .. }
            | Operation::VectorMaxDiffLt { .. }
            | Operation::VectorConvert { .. }
            | Operation::VectorMov { .. }
            | Operation::VectorSRS { .. }
            // Conditional vector operations
            | Operation::VectorAbsGtz { .. }
            | Operation::VectorNegGtz { .. }
            | Operation::VectorNegLtz { .. }
            | Operation::VectorNegate { .. }
            | Operation::VectorNegAdd { .. }
            | Operation::VectorNegMul { .. }
            // Vector element operations
            | Operation::VectorExtract { .. }
            | Operation::VectorInsert { .. }
            | Operation::VectorSelect { .. }
            | Operation::VectorClear
            | Operation::VectorBroadcast { .. }
            // Vector shift operations
            | Operation::VectorShiftLeft { .. }
            | Operation::VectorShiftRight { .. }
            | Operation::VectorArithShiftRight { .. }
            | Operation::VectorAlign { .. }
            | Operation::VectorUpshift { .. } => SlotIndex::Vector,

            // Accumulator operations (Accumulator slot)
            Operation::VectorAccumulate { .. } => SlotIndex::Accumulator,

            // Matrix/Accumulator operations (Accumulator slot)
            // All VMAC/VMSC variants used for convolutions
            Operation::VectorMac { .. }
            | Operation::VectorMatMulDense { .. }
            | Operation::VectorMatMulSparse { .. }
            | Operation::VectorMatMulSubDense { .. }
            | Operation::VectorMatMulSubSparse { .. }
            | Operation::VectorNegMatMulDense { .. }
            | Operation::VectorNegMatMulSubDense { .. }
            | Operation::VectorMatMulAccFloat { .. }
            | Operation::VectorMatMulSubFloat { .. }
            | Operation::VectorAddMac { .. }
            | Operation::VectorSubMac { .. } => SlotIndex::Accumulator,

            // Memory load operations (LoadA slot -- primary load port)
            Operation::PointerAdd
            | Operation::PointerMov
            | Operation::Load { .. }
            | Operation::VectorLoadA { .. }
            | Operation::VectorLoadUnpack { .. }
            // Cascade operations appear in the LDA slot in VLIW bundles
            | Operation::CascadeRead
            | Operation::CascadeWrite => SlotIndex::LoadA,

            // Memory load operations (LoadB slot -- secondary load port)
            Operation::VectorLoadB { .. } => SlotIndex::LoadB,

            // Memory store operations (Store slot)
            Operation::Store { .. }
            | Operation::VectorStore { .. } => SlotIndex::Store,

            // Control and synchronization operations (Control slot)
            Operation::Branch { .. }
            | Operation::Call
            | Operation::Return
            | Operation::LockAcquire
            | Operation::LockRelease
            | Operation::DmaStart
            | Operation::DmaWait
            | Operation::StreamWriteScalar { .. }
            | Operation::StreamWritePacketHeader { .. }
            | Operation::StreamReadScalar { .. }
            | Operation::Halt => SlotIndex::Control,

            Operation::Nop | Operation::Unknown { .. } => SlotIndex::Scalar0,
        }
    }
}

/// Operand specification.
///
/// Represents a source or destination operand for an operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// Scalar general purpose register (r0-r31).
    ScalarReg(u8),
    /// Vector register (v0-v31).
    VectorReg(u8),
    /// Accumulator register (acc0-acc7).
    AccumReg(u8),
    /// Pointer register (p0-p7) for addressing.
    PointerReg(u8),
    /// Modifier register (m0-m7) for post-modify addressing.
    ModifierReg(u8),
    /// Immediate value.
    Immediate(i32),
    /// Memory address with base register and offset.
    Memory {
        /// Base pointer register.
        base: u8,
        /// Signed offset from base.
        offset: i16,
    },
    /// Lock identifier.
    Lock(u8),
    /// DMA channel.
    DmaChannel(u8),
    /// Buffer descriptor.
    BufferDescriptor(u8),
    /// Control register (crRnd=6, crSat=9, crSRSSign=8, crVaddSign=0, etc.).
    /// The u8 is the 4-bit hardware register ID from the ISA encoding.
    ControlReg(u8),
}

/// Predicate for conditional execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Predicate {
    /// Predicate register (0-7).
    pub reg: u8,
    /// Invert the predicate.
    pub inverted: bool,
}

/// A single slot operation within a VLIW bundle.
///
/// Contains the decoded instruction, its operands, and metadata from
/// TableGen. The `semantic` field carries the operation kind (Add, Load, Br,
/// etc.) and the accompanying metadata fields (is_vector, element_type,
/// mem_width, post_modify, branch_condition, select_variant) carry the
/// disambiguation and addressing info.
///
/// # Execution Dispatch
///
/// Executors should match on `semantic` plus metadata guards:
/// ```ignore
/// match op.semantic {
///     Some(SemanticOp::Add) if !op.is_vector => { /* scalar add */ }
///     Some(SemanticOp::Add) => { /* vector add, use op.element_type */ }
///     Some(SemanticOp::Load) => { /* use op.mem_width, op.post_modify */ }
///     _ => { /* fallback to op.op for legacy operations */ }
/// }
/// ```
///
/// The `op` field is **deprecated** and retained only for legacy executor
/// paths and the latency table (Phase 4 will remove it).
#[derive(Debug, Clone)]
pub struct SlotOp {
    /// Which slot this operation occupies.
    pub slot: SlotIndex,
    /// **Deprecated**: Legacy operation enum. Prefer `semantic` + metadata.
    /// Retained for latency lookup until Phase 4 replaces it.
    pub op: Operation,
    /// TableGen-derived semantic operation (when available).
    ///
    /// This represents *what* the instruction computes (Add, Sub, Select, etc.)
    /// independent of slot, element type, or encoding details.
    pub semantic: Option<SemanticOp>,
    /// Implicit register uses/defs from TableGen.
    ///
    /// For example, `sel.eqz` uses r27 implicitly to read the test value.
    /// This is not encoded in the instruction bits -- TableGen tells us it's fixed.
    pub implicit_regs: SmallVec<[ImplicitReg; 2]>,

    // ── Instruction Metadata (from InstrEncoding) ──────────────────────
    //
    // These fields carry all the disambiguation and addressing info that
    // was previously baked into the 130-variant Operation enum. They are
    // populated from InstrEncoding at decode time.

    /// Whether this is a vector (SIMD) operation.
    pub is_vector: bool,
    /// Element type for vector operations (None for scalar).
    pub element_type: Option<ElementType>,
    /// Memory access width for load/store operations.
    pub mem_width: MemWidth,
    /// Post-modify mode for load/store addressing.
    /// Populated directly from the address generator field, not backpatched.
    pub post_modify: PostModify,
    /// Branch condition for conditional branches.
    pub branch_condition: Option<BranchCondition>,
    /// Select variant (generic, equal-zero, not-equal-zero).
    pub select_variant: Option<SelectVariant>,

    // ── Operands ───────────────────────────────────────────────────────

    /// Source operands (ordered per InstrDef.inputs when semantic is set).
    pub sources: SmallVec<[Operand; 4]>,
    /// Destination operand (if any).
    pub dest: Option<Operand>,
    /// Optional predicate for conditional execution.
    pub predicate: Option<Predicate>,
    /// Encoding mnemonic from TableGen (e.g., "jl", "mova", "vadd").
    /// Used by the crossref tool to compare against llvm-objdump output.
    pub encoding_name: Option<String>,
}

impl SlotOp {
    /// Create a new slot operation.
    pub fn new(slot: SlotIndex, op: Operation) -> Self {
        Self {
            slot,
            op,
            semantic: None,
            implicit_regs: SmallVec::new(),
            is_vector: false,
            element_type: None,
            mem_width: MemWidth::Word,
            post_modify: PostModify::None,
            branch_condition: None,
            select_variant: None,
            sources: SmallVec::new(),
            dest: None,
            predicate: None,
            encoding_name: None,
        }
    }

    /// Create a new slot operation with TableGen semantic info.
    pub fn with_semantic(slot: SlotIndex, op: Operation, semantic: SemanticOp) -> Self {
        Self {
            slot,
            op,
            semantic: Some(semantic),
            implicit_regs: SmallVec::new(),
            is_vector: false,
            element_type: None,
            mem_width: MemWidth::Word,
            post_modify: PostModify::None,
            branch_condition: None,
            select_variant: None,
            sources: SmallVec::new(),
            dest: None,
            predicate: None,
            encoding_name: None,
        }
    }

    /// Add implicit registers from TableGen.
    pub fn with_implicit_regs(mut self, regs: impl IntoIterator<Item = ImplicitReg>) -> Self {
        self.implicit_regs.extend(regs);
        self
    }

    /// Get an implicit register use by register number.
    ///
    /// Returns the register number if found and it's a use (not a def).
    pub fn get_implicit_use(&self, reg_num: u8) -> Option<u8> {
        self.implicit_regs
            .iter()
            .find(|ir| ir.reg_num == reg_num && ir.is_use)
            .map(|ir| ir.reg_num)
    }

    /// Add a source operand.
    pub fn with_source(mut self, src: Operand) -> Self {
        self.sources.push(src);
        self
    }

    /// Set the destination operand.
    pub fn with_dest(mut self, dst: Operand) -> Self {
        self.dest = Some(dst);
        self
    }

    /// Set the post-modify mode for load/store addressing.
    pub fn with_post_modify(mut self, pm: PostModify) -> Self {
        self.post_modify = pm;
        self
    }

    /// Set the memory access width.
    pub fn with_mem_width(mut self, width: MemWidth) -> Self {
        self.mem_width = width;
        self
    }

    /// Set the predicate.
    pub fn with_predicate(mut self, pred: Predicate) -> Self {
        self.predicate = Some(pred);
        self
    }

    /// Create a NOP for the given slot.
    pub fn nop(slot: SlotIndex) -> Self {
        Self::new(slot, Operation::Nop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_index_all() {
        let all = SlotIndex::all();
        assert_eq!(all.len(), 8);
        assert_eq!(all[0], SlotIndex::Scalar0);
        assert_eq!(all[6], SlotIndex::Control);
        assert_eq!(all[7], SlotIndex::LoadB);
    }

    #[test]
    fn test_slot_index_is_scalar() {
        assert!(SlotIndex::Scalar0.is_scalar());
        assert!(SlotIndex::Scalar1.is_scalar());
        assert!(!SlotIndex::Vector.is_scalar());
        assert!(!SlotIndex::Control.is_scalar());
    }

    #[test]
    fn test_element_type_bits() {
        assert_eq!(ElementType::Int8.bits(), 8);
        assert_eq!(ElementType::Int16.bits(), 16);
        assert_eq!(ElementType::Int32.bits(), 32);
        assert_eq!(ElementType::BFloat16.bits(), 16);
    }

    #[test]
    fn test_element_type_lanes() {
        assert_eq!(ElementType::Int8.lanes_256(), 32);
        assert_eq!(ElementType::Int16.lanes_256(), 16);
        assert_eq!(ElementType::Int32.lanes_256(), 8);
    }

    #[test]
    fn test_mem_width_bytes() {
        assert_eq!(MemWidth::Byte.bytes(), 1);
        assert_eq!(MemWidth::Word.bytes(), 4);
        assert_eq!(MemWidth::Vector256.bytes(), 32);
    }

    #[test]
    fn test_operation_natural_slot() {
        assert_eq!(Operation::ScalarAdd.natural_slot(), SlotIndex::Scalar0);
        assert_eq!(Operation::ScalarMul.natural_slot(), SlotIndex::Scalar1);
        assert_eq!(
            Operation::VectorAdd {
                element_type: ElementType::Int32
            }
            .natural_slot(),
            SlotIndex::Vector
        );
        assert_eq!(
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::None
            }
            .natural_slot(),
            SlotIndex::LoadA
        );
        assert_eq!(
            Operation::Branch {
                condition: BranchCondition::Always
            }
            .natural_slot(),
            SlotIndex::Control
        );
    }

    #[test]
    fn test_operation_classifications() {
        assert!(Operation::Branch {
            condition: BranchCondition::Equal
        }
        .is_control_flow());
        assert!(Operation::Call.is_control_flow());
        assert!(Operation::Return.is_control_flow());
        assert!(!Operation::ScalarAdd.is_control_flow());

        assert!(Operation::Load {
            width: MemWidth::Word,
            post_modify: PostModify::None
        }
        .is_memory());
        assert!(!Operation::ScalarAdd.is_memory());

        assert!(Operation::LockAcquire.is_sync());
        assert!(Operation::DmaStart.is_sync());
    }

    #[test]
    fn test_slot_op_builder() {
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(2));

        assert_eq!(op.slot, SlotIndex::Scalar0);
        assert_eq!(op.sources.len(), 2);
        assert!(op.dest.is_some());
        assert!(op.predicate.is_none());
    }
}
