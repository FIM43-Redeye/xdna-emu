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

    /// True for sub-word widths that use the read-modify-write pipeline.
    ///
    /// AIE2 partial-word stores (st.s8/st.u8/st.s16/st.u16) use a RMW
    /// pipeline where the data register is read 7 cycles after issue
    /// (II_STHB in AIE2Schedule.td). Full-word and vector stores read
    /// their data register at issue time.
    pub fn is_partial_word(self) -> bool {
        matches!(self, MemWidth::Byte | MemWidth::HalfWord)
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
/// Executors match on `semantic` plus metadata guards:
/// ```ignore
/// match op.semantic {
///     Some(SemanticOp::Add) if !op.is_vector => { /* scalar add */ }
///     Some(SemanticOp::Add) => { /* vector add, use op.element_type */ }
///     Some(SemanticOp::Load) => { /* use op.mem_width, op.post_modify */ }
///     None => { /* unknown instruction */ }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SlotOp {
    /// Which slot this operation occupies.
    pub slot: SlotIndex,
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
    // These fields carry disambiguation and addressing info, populated
    // from InstrEncoding at decode time.

    /// Whether this is a vector (SIMD) operation.
    pub is_vector: bool,
    /// Whether this operates on 512-bit (x) registers rather than 256-bit (w).
    /// When true, the VectorAlu processes both halves (reg and reg+1).
    pub is_wide_vector: bool,
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
    /// Whether this stream read/write is blocking.
    pub blocking: bool,
    /// Source element type for SRS/UPS/Convert (element_type is destination).
    pub from_type: Option<ElementType>,
    /// Shuffle pattern for shuffle operations.
    pub shuffle_pattern: ShufflePattern,

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
    /// Raw opcode bits for unrecognized instructions (diagnostics only).
    pub raw_opcode: Option<u32>,
}

// Core methods on SlotOp.
impl SlotOp {
    /// Get the natural execution slot for this operation based on SemanticOp + metadata.
    ///
    /// Used by the decoder to resolve LNG slot instructions (which can contain
    /// either control flow or scalar/vector operations).
    pub fn natural_slot(&self) -> SlotIndex {
        match self.semantic {
            None | Some(SemanticOp::Nop) => SlotIndex::Scalar0,

            // Scalar multiply uses Scalar1
            Some(SemanticOp::Mul) if !self.is_vector => SlotIndex::Scalar1,

            // Scalar ops use Scalar0
            Some(SemanticOp::Add | SemanticOp::Sub | SemanticOp::And | SemanticOp::Or
                | SemanticOp::Xor | SemanticOp::Shl | SemanticOp::Srl | SemanticOp::Sra
                | SemanticOp::Copy | SemanticOp::Cmp | SemanticOp::Abs | SemanticOp::Ctlz
                | SemanticOp::Clb | SemanticOp::Adc | SemanticOp::Sbc
                | SemanticOp::SignExtend | SemanticOp::ZeroExtend
                | SemanticOp::SetLt | SemanticOp::SetUlt | SemanticOp::SetLe
                | SemanticOp::SetUle | SemanticOp::SetGt | SemanticOp::SetUgt
                | SemanticOp::SetGe | SemanticOp::SetUge | SemanticOp::SetEq
                | SemanticOp::SetNe | SemanticOp::Select
                | SemanticOp::SDiv | SemanticOp::UDiv | SemanticOp::SRem
                | SemanticOp::Neg | SemanticOp::Truncate | SemanticOp::Event
                | SemanticOp::ReadCycleCounter
            ) if !self.is_vector => SlotIndex::Scalar0,

            // MAC/MatMul/Accumulate -> Accumulator slot
            Some(SemanticOp::Mac | SemanticOp::MatMul | SemanticOp::MatMulSub
                | SemanticOp::NegMatMul | SemanticOp::AddMac | SemanticOp::SubMac
                | SemanticOp::Accumulate) => SlotIndex::Accumulator,

            // Load/PointerAdd/Cascade -> LoadA
            Some(SemanticOp::Load | SemanticOp::PointerAdd | SemanticOp::PointerMov
                | SemanticOp::CascadeRead | SemanticOp::CascadeWrite) => SlotIndex::LoadA,

            // Store -> Store slot
            Some(SemanticOp::Store) => SlotIndex::Store,

            // Control flow and synchronization -> Control slot
            Some(SemanticOp::Br | SemanticOp::BrCond | SemanticOp::Call | SemanticOp::Ret
                | SemanticOp::LockAcquire | SemanticOp::LockRelease
                | SemanticOp::DmaStart | SemanticOp::DmaWait
                | SemanticOp::StreamRead | SemanticOp::StreamWrite
                | SemanticOp::StreamWritePacketHeader
                | SemanticOp::Halt | SemanticOp::Done) => SlotIndex::Control,

            // Vector ops -> Vector slot (catch-all for is_vector)
            _ if self.is_vector => SlotIndex::Vector,

            // Fallback
            _ => SlotIndex::Scalar0,
        }
    }

}

// Builder and query methods on SlotOp.
impl SlotOp {
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

    /// Mark as a vector operation with the given element type.
    pub fn as_vector(mut self, et: ElementType) -> Self {
        self.is_vector = true;
        self.element_type = Some(et);
        self
    }

    /// Set the branch condition.
    pub fn with_branch_condition(mut self, cond: BranchCondition) -> Self {
        self.branch_condition = Some(cond);
        self
    }

    /// Set the blocking flag (for stream read/write).
    pub fn with_blocking(mut self, blocking: bool) -> Self {
        self.blocking = blocking;
        self
    }

    /// Set the select variant.
    pub fn with_select_variant(mut self, variant: SelectVariant) -> Self {
        self.select_variant = Some(variant);
        self
    }

    /// Create a NOP for the given slot.
    pub fn nop(slot: SlotIndex) -> Self {
        Self {
            slot,
            semantic: Some(SemanticOp::Nop),
            implicit_regs: SmallVec::new(),
            is_vector: false,
            is_wide_vector: false,
            element_type: None,
            mem_width: MemWidth::Word,
            post_modify: PostModify::None,
            branch_condition: None,
            select_variant: None,
            blocking: false,
            from_type: None,
            shuffle_pattern: ShufflePattern::default(),
            sources: SmallVec::new(),
            dest: None,
            predicate: None,
            encoding_name: None,
            raw_opcode: None,
        }
    }

    /// Check if this slot operation is a NOP.
    #[inline]
    pub fn is_nop(&self) -> bool {
        matches!(self.semantic, Some(SemanticOp::Nop) | None)
    }

    /// Check if this slot operation is a control flow operation.
    #[inline]
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self.semantic,
            Some(SemanticOp::Br | SemanticOp::BrCond | SemanticOp::Call | SemanticOp::Ret)
        )
    }

    /// Check if this slot operation is a memory operation.
    #[inline]
    pub fn is_memory(&self) -> bool {
        matches!(
            self.semantic,
            Some(SemanticOp::Load | SemanticOp::Store)
        )
    }

    /// Check if this slot operation is a synchronization operation.
    #[inline]
    pub fn is_sync(&self) -> bool {
        matches!(
            self.semantic,
            Some(SemanticOp::LockAcquire | SemanticOp::LockRelease
                | SemanticOp::DmaStart | SemanticOp::DmaWait)
        )
    }

    /// Create a SlotOp from a SemanticOp.
    pub fn from_semantic(slot: SlotIndex, semantic: SemanticOp) -> Self {
        Self {
            slot,
            semantic: Some(semantic),
            implicit_regs: SmallVec::new(),
            is_vector: false,
            is_wide_vector: false,
            element_type: None,
            mem_width: MemWidth::Word,
            post_modify: PostModify::None,
            branch_condition: None,
            select_variant: None,
            blocking: false,
            from_type: None,
            shuffle_pattern: ShufflePattern::default(),
            sources: SmallVec::new(),
            dest: None,
            predicate: None,
            encoding_name: None,
            raw_opcode: None,
        }
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
    fn test_slot_op_builder() {
        use crate::tablegen::SemanticOp;

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(2));

        assert_eq!(op.slot, SlotIndex::Scalar0);
        assert_eq!(op.sources.len(), 2);
        assert!(op.dest.is_some());
        assert!(op.predicate.is_none());
        assert_eq!(op.semantic, Some(SemanticOp::Add));
    }
}
