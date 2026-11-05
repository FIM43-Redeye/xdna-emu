//! Types of the build-time TableGen model: semantic vocabulary (`SemanticOp`, `ElementType`,
//! ...), scheduling, registers, composite bundle formats, and [`TblgenOutput`], which bundles
//! them with the resolved encodings.

use std::collections::HashMap;

/// LLVM SelectionDAG node types that map to semantic operations.
///
/// These come from `Pat<>` patterns in AIE2InstrPatterns.td.
/// The SDNode name tells us what the instruction actually computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SemanticOp {
    // Arithmetic
    Add,
    Sub,
    Adc, // Add with carry (reads carry flag before computation)
    Sbc, // Subtract with borrow (reads carry flag before computation)
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    Abs,
    Neg,

    // Bitwise
    And,
    Or,
    Xor,
    Not,
    Shl, // Shift left
    Sra, // Shift right arithmetic (signed)
    Srl, // Shift right logical (unsigned)
    // Bidirectional shifts (AIE2 ASHL/LSHL hardware instructions).
    // Positive shift amount = left, negative = right.
    // ASHL: right-shifts are arithmetic (sign-preserving).
    // LSHL: right-shifts are logical (zero-filling).
    AshlBidir,
    LshlBidir,
    Rotl, // Rotate left
    Rotr, // Rotate right

    // Comparison (result is 0 or 1)
    SetEq,  // ==
    SetNe,  // !=
    SetLt,  // < (signed)
    SetLe,  // <= (signed)
    SetGt,  // > (signed)
    SetGe,  // >= (signed)
    SetUlt, // < (unsigned)
    SetUle, // <= (unsigned)
    SetUgt, // > (unsigned)
    SetUge, // >= (unsigned)

    // Bit manipulation
    Ctlz,  // Count leading zeros
    Cttz,  // Count trailing zeros
    Ctpop, // Population count (count ones)
    Bswap, // Byte swap

    // Memory
    Load,
    Store,

    // Control flow
    Br,     // Unconditional branch
    BrCond, // Conditional branch
    Call,   // Call subroutine (jl: sets LR after delay slots)
    Ret,    // Return from subroutine
    Select, // Conditional select (ternary)

    // Sign/zero extension
    SignExtend,
    ZeroExtend,
    Truncate,

    // Special
    Copy,    // Move/copy
    Nop,     // No operation
    Done,    // Core termination (halt)
    Event,   // Generate trace event (operand selects INSTR_EVENT_0/1)
    DivStep, // Iterative signed division step (dstep via r31)

    // Synchronization (AIE-specific)
    LockAcquire, // Acquire lock
    LockRelease, // Release lock

    // Bit manipulation (scalar-only)
    Clb, // Count leading bits (ones or zeros, != CLZ)
    Cmp, // Compare (flag-setting, no destination register)

    // Vector-specific operations
    Mac,             // Multiply-accumulate: acc += A * B
    MatMul,          // Matrix multiply (dense): acc = A * B
    MatMulSub,       // Matrix multiply-subtract: acc -= A * B
    NegMatMul,       // Negated matrix multiply: acc += -(A * B)
    AddMac,          // Double accumulator: acc1 = acc1 + acc2 + A * B
    SubMac,          // Double accumulator: acc1 = acc1 - acc2 + A * B
    Srs,             // Shift-round-saturate: acc -> vec
    Ups,             // Upshift: vec -> acc
    Shuffle,         // Vector lane permutation
    Pack,            // Pack two vectors (narrow)
    Unpack,          // Unpack vector (widen)
    Align,           // Concatenate and shift two vectors (vshift)
    VectorBroadcast, // Broadcast scalar to all lanes
    VectorExtract,   // Extract single element from vector
    VectorInsert,    // Insert scalar into vector lane
    VectorPush,      // Push scalar into vector low half (vpush.lo)
    VectorPushHi,    // Push scalar into vector high half (vpush.hi)
    VectorSelect,    // Per-lane conditional select
    VectorClear,     // Clear vector to zero
    Convert,         // Type conversion (vconv, vfloor, vceil, etc.)
    Min,             // Minimum (scalar or vector)
    Max,             // Maximum (scalar or vector)

    // Conditional vector operations (AIE2 compound ops)
    SubLt,       // dst[i] = (a < b) ? a - b : a
    SubGe,       // dst[i] = (a >= b) ? a - b : a
    MaxDiffLt,   // dst[i] = max(a - b, 0) when a < b
    MaxLt,       // dst = max(a, b) with less-than flag
    MinGe,       // dst = min(a, b) with greater-equal flag
    AbsGtz,      // dst[i] = (src > 0) ? abs(src) : src
    NegGtz,      // dst[i] = (src > 0) ? -src : src
    NegLtz,      // dst[i] = (src < 0) ? -src : src
    NegAdd,      // dst = -src1 + src2
    NegMul,      // acc += -(src1 * src2)
    Accumulate,  // acc += src (no multiply)
    AccumSub,    // acc1 - acc2 (vadd with md2=1 / vsub)
    AccumNegAdd, // -acc1 + acc2 (vnegadd)
    AccumNegSub, // -acc1 - acc2 (vnegsub)

    // Side-effect operations
    CascadeRead,             // Read from cascade input
    CascadeWrite,            // Write to cascade output
    StreamRead,              // Read from slave stream
    StreamWrite,             // Write to master stream
    StreamWritePacketHeader, // Write packet header to master stream
    DmaStart,                // Start DMA transfer
    DmaWait,                 // Wait for DMA completion
    Halt,                    // Core termination (halt)

    // Hardware state reads
    ReadCycleCounter, // MOV_CNTR: read per-core cycle counter

    // Pointer operations
    PointerAdd, // Pointer arithmetic: ptr + offset
    PointerMov, // Pointer move: ptr = value

    // Target-specific intrinsic (needs name lookup)
    Intrinsic(u32), // Index into intrinsic name table
}

impl SemanticOp {
    /// Returns true if this is a commutative operation.
    pub fn is_commutative(&self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::And | Self::Or | Self::Xor | Self::SetEq | Self::SetNe)
    }

    /// Returns true if this is a comparison operation.
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::SetEq
                | Self::SetNe
                | Self::SetLt
                | Self::SetLe
                | Self::SetGt
                | Self::SetGe
                | Self::SetUlt
                | Self::SetUle
                | Self::SetUgt
                | Self::SetUge
        )
    }
}

/// Element type for vector operations.
///
/// Defined here in tablegen::types so the resolver can populate it during
/// TableGen loading without depending on the interpreter module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum ElementType {
    /// 4-bit signed integer (pack/unpack narrow operand; packed two per byte).
    Int4,
    /// 4-bit unsigned integer (pack/unpack narrow operand; packed two per byte).
    UInt4,
    /// 8-bit signed integer.
    Int8,
    /// 8-bit unsigned integer.
    UInt8,
    /// 16-bit signed integer.
    Int16,
    /// 16-bit unsigned integer.
    UInt16,
    /// 32-bit signed integer.
    Int32,
    /// 32-bit unsigned integer.
    UInt32,
    /// 64-bit signed integer (accumulator lanes in S64 mode).
    Int64,
    /// 64-bit unsigned integer (accumulator lanes in D64 mode).
    UInt64,
    /// 16-bit brain floating point.
    BFloat16,
    /// 32-bit floating point.
    Float32,
}

impl ElementType {
    /// Get the size of this element type in bits.
    pub fn bits(self) -> u8 {
        match self {
            ElementType::Int4 | ElementType::UInt4 => 4,
            ElementType::Int8 | ElementType::UInt8 => 8,
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => 16,
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => 32,
            ElementType::Int64 | ElementType::UInt64 => 64,
        }
    }

    /// Get the number of elements that fit in a 256-bit vector.
    pub fn lanes_256(self) -> u8 {
        (256u16 / self.bits() as u16) as u8
    }

    /// Whether this element type is a signed integer. Floating-point types
    /// report `false` (they are neither signed nor unsigned integers).
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            ElementType::Int4
                | ElementType::Int8
                | ElementType::Int16
                | ElementType::Int32
                | ElementType::Int64
        )
    }

    /// The integer type of the same width with the given signedness; floats are returned
    /// unchanged. Used by the `d` forms, whose signedness is a run-time control-register value.
    pub fn with_signedness(self, signed: bool) -> Self {
        use ElementType::*;
        match (self, signed) {
            (Int4 | UInt4, true) => Int4,
            (Int4 | UInt4, false) => UInt4,
            (Int8 | UInt8, true) => Int8,
            (Int8 | UInt8, false) => UInt8,
            (Int16 | UInt16, true) => Int16,
            (Int16 | UInt16, false) => UInt16,
            (Int32 | UInt32, true) => Int32,
            (Int32 | UInt32, false) => UInt32,
            (Int64 | UInt64, true) => Int64,
            (Int64 | UInt64, false) => UInt64,
            (BFloat16 | Float32, _) => self,
        }
    }
}

/// Branch condition codes.
///
/// Defined here in tablegen::types so the resolver can populate it during
/// TableGen loading without depending on the interpreter module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BranchCondition {
    /// Always branch (unconditional).
    #[default]
    Always,
    /// Branch if equal (Z=1).
    Equal,
    /// Branch if not equal (Z=0).
    NotEqual,
    /// Branch if less than (signed: N!=V).
    Less,
    /// Branch if greater or equal (signed: N==V).
    GreaterEqual,
    /// Branch if less or equal (signed: Z=1 or N!=V).
    LessEqual,
    /// Branch if greater (signed: Z=0 and N==V).
    Greater,
    /// Branch if negative (N=1).
    Negative,
    /// Branch if positive or zero (N=0).
    PositiveOrZero,
    /// Branch if carry set (unsigned overflow).
    CarrySet,
    /// Branch if carry clear.
    CarryClear,
    /// Branch if overflow set.
    OverflowSet,
    /// Branch if overflow clear.
    OverflowClear,
    /// Branch if source register is zero (jz instruction).
    Zero,
    /// Branch if source register is not zero (jnz instruction).
    NotZero,
    /// Branch if source register is not zero AND decrement (jnzd instruction).
    /// Same as NotZero but also decrements the destination register.
    NotZeroDecrement,
}

/// Select instruction variant.
///
/// AIE2 has three select forms:
/// - Generic `sel` (3 operands: dst = cond ? a : b)
/// - `sel.eqz` (2 operands + implicit r27: dst = (r27 == 0) ? a : b)
/// - `sel.nez` (2 operands + implicit r27: dst = (r27 != 0) ? a : b)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectVariant {
    /// Standard 3-operand select.
    Generic,
    /// Select if equal zero (tests implicit r27).
    EqualZero,
    /// Select if not equal zero (tests implicit r27).
    NotEqualZero,
}

/// An implicit register use/def of an instruction (TableGen `Uses` / `Defs`), as the
/// interpreter consumes it today.
///
/// The generated tables fill it from `Defs` / `Uses` (`crSat`, `srSRS_of`, `SP`, ...); the
/// interpreter also builds some by hand for fixed operands (`eR27` for `sel.eqz`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitReg {
    /// Register name (generated) or single-register class (hand-built, e.g. "eR27").
    pub reg_class: String,
    /// Scalar GPR index: the hardware encoding when the register is a GPR (`r0`..`r31`), 0 for
    /// every other register. `get_implicit_use(op, ctx, n)` reads `r<n>` through it. Until
    /// 2026-11-04 it was parsed from the digits of the name (`crF2FMask` -> 2). The hardware
    /// encoding of every implicit register is in `EncodingSchema::implicit_regs`.
    pub reg_num: u8,
    /// Whether this is a use (read) or def (write)
    pub is_use: bool,
}

// ============================================================================
// Scheduling Model (from AIE2Schedule.td)
// ============================================================================

/// Processor-level scheduling parameters from AIE2SchedModel.
///
/// These are the top-level latency parameters that the compiler's scheduler
/// uses. Values are extracted from the `SchedMachineModel` record in
/// `AIE2Schedule.td`.
///
/// # Example (AIE2)
/// ```text
/// LoadLatency = 5       -- Memory load pipeline depth
/// HighLatency = 37      -- Matrix multiply / high-latency ops
/// MispredictPenalty = 4  -- Branch misprediction cost
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessorModel {
    /// Cycles for a memory load result to be available (default 5 for AIE2).
    pub load_latency: u8,
    /// Cycles for the longest "high latency" operations (e.g., matmul).
    pub high_latency: u8,
    /// Branch misprediction penalty in cycles.
    pub mispredict_penalty: u8,
    /// Instruction issue width (1000 = "unlimited" in LLVM conventions).
    pub issue_width: u16,
    /// Name of the itinerary model (e.g., "AIE2Itineraries").
    pub itinerary_name: String,
}

/// Per-instruction scheduling info from InstrItinData records.
///
/// Each instruction class (e.g., II_ADD, II_LDA, II_VMUL) has an itinerary
/// that describes its pipeline occupancy, operand latencies, and functional
/// unit requirements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItineraryInfo {
    /// Itinerary class name (e.g., "II_ADD", "II_LDA", "II_VMUL").
    pub class_name: String,
    /// Total pipeline latency: sum of stage cycles (result latency).
    pub total_latency: u8,
    /// Per-operand result latencies (index 0 = first result, etc.).
    pub operand_cycles: Vec<u8>,
    /// Pipeline stages with functional unit assignments.
    pub stages: Vec<PipelineStage>,
    /// Bypass classes (forwarding paths), parallel to operand_cycles.
    pub bypasses: Vec<String>,
}

/// A single pipeline stage from an InstrStage record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStage {
    /// Number of cycles this stage occupies.
    pub cycles: u8,
    /// Functional units that can execute this stage.
    pub units: Vec<String>,
    /// Time increment to next stage (-1 = blocking until done).
    pub time_inc: i8,
    /// TableGen `ReservationKind::Reserved`: the stage holds its unit for later instructions but
    /// does not itself require the unit to be free (LLVM's scoreboard hazard check skips it).
    pub reserved: bool,
}

// ============================================================================
// Register Model (from AIE2GenRegisterInfo.td)
// ============================================================================

/// A single register definition with its hardware encoding.
///
/// Extracted from register records in `--print-records` output. Each register
/// has a unique HWEncoding used by the assembler/disassembler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterDef {
    /// Register name as used in assembly (e.g., "r0", "p3", "lr", "m5").
    pub name: String,
    /// TableGen `AsmName`, the spelling the assembler accepts (`ls` for the def `LS`); the def
    /// name where no `AsmName` is given.
    pub asm_name: String,
    /// Hardware encoding value (from `bits<16> HWEncoding`).
    pub hw_encoding: u16,
    /// Parent classes from the def line (e.g., ["AIE2GPReg", "DwarfRegNum"]).
    pub parents: Vec<String>,
    /// TableGen `SubRegs`: the registers this one is made of (`x0` -> `[wl0, wh0]`,
    /// `l1` -> `[r18, r19]`); empty for a leaf register.
    pub sub_regs: Vec<String>,
    /// TableGen `SubRegIndices`, parallel to `sub_regs` (`[sub_256_lo, sub_256_hi]`).
    pub sub_reg_indices: Vec<String>,
}

/// A register class with its member list.
///
/// Register classes group registers that are interchangeable for a given
/// operand type. They are used to classify composite register encodings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterClassDef {
    /// Class name (e.g., "eR", "eP", "eM", "eSPL").
    pub name: String,
    /// Ordered list of member register names (e.g., ["r0", "r1", ..., "r31"]).
    pub members: Vec<String>,
    /// Register alignment in bits (e.g., 32, 256).
    pub alignment: u16,
    /// Width of one register of the class in bits (TableGen `RegInfo.RegSize`): 32 for the
    /// scalar classes, 20 for the p / m / dn / dj / dc classes and the 20-bit special registers.
    pub reg_size_bits: u16,
    /// Parent classes from the def line.
    pub parents: Vec<String>,
    /// The `AIE2*RegisterClass` family the class derives from.
    pub family: super::schema::RegClassFamily,
}

/// Complete register model extracted from TableGen.
#[derive(Debug, Clone, Default)]
pub struct RegisterModel {
    /// All individual register definitions, indexed by name.
    pub registers: HashMap<String, RegisterDef>,
    /// All register classes, indexed by class name.
    pub classes: HashMap<String, RegisterClassDef>,
}

impl RegisterModel {
    /// The registers of a class in member order, member classes expanded (`mWm` -> `wh0`, `wh2`,
    /// ..., `wl11`). Empty for an unknown class.
    pub fn leaves(&self, class: &str) -> Vec<&str> {
        let mut out = Vec::new();
        if let Some(c) = self.classes.get(class) {
            self.leaf_registers(c, &mut out, 0);
        }
        out
    }

    /// The hardware encodings of [`leaves`](Self::leaves), in the same order.
    pub fn hw_encodings(&self, class: &str) -> Vec<u16> {
        self.leaves(class)
            .iter()
            .filter_map(|r| self.registers.get(*r))
            .map(|r| r.hw_encoding)
            .collect()
    }

    /// The hardware encoding of the first register of a class: the register a field value of 0
    /// selects in a plain (non-composite) class (`eRS8` -> 16).
    pub fn base_hw_encoding(&self, class: &str) -> Option<u16> {
        self.hw_encodings(class).first().copied()
    }

    /// The leaf registers of a class: its members, with member classes expanded.
    fn leaf_registers<'a>(&'a self, class: &'a RegisterClassDef, out: &mut Vec<&'a str>, depth: u8) {
        for member in &class.members {
            match self.classes.get(member) {
                Some(inner) if depth < 8 => self.leaf_registers(inner, out, depth + 1),
                Some(_) => {}
                None => out.push(member),
            }
        }
    }

    /// The width in bits TableGen gives a register: the narrowest `RegInfo.RegSize` among the
    /// classes that contain it. A register sits in wider "as 32 bit" classes too (a pointer
    /// moves through a scalar class), so the narrowest is its own width. TableGen states a
    /// register class width, not a register width: a register only as narrow as its class
    /// (`s0` is in a 32-bit class; silicon keeps 6 bits).
    pub fn register_width_bits(&self, name: &str) -> Option<u16> {
        let mut leaves = Vec::new();
        self.classes
            .values()
            .filter(|c| c.reg_size_bits > 0)
            .filter(|c| {
                leaves.clear();
                self.leaf_registers(c, &mut leaves, 0);
                leaves.contains(&name)
            })
            .map(|c| c.reg_size_bits)
            .min()
    }
}

// ============================================================================
// Composite Bundle Formats (from AIE2CompositeFormats.td)
// ============================================================================

/// Bit-level extraction map for a single slot within a composite format.
///
/// Slot bits are always contiguous in AIE2 composite formats (verified across
/// all 78 defs), so `start_bit + width` fully describes the extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotBitMap {
    /// Slot name (e.g., "st", "mv", "alu", "lda", "ldb", "vec", "lng", "nop").
    pub slot_name: String,
    /// Slot bit width.
    pub width: u8,
    /// LSB position of this slot in the bundle word (little-endian bit numbering).
    pub start_bit: u8,
}

/// A composite instruction format (VLIW bundle layout).
///
/// Extracted from records with `isComposite = 1` in the `--print-records`
/// output. Each format defines the total bundle size and which VLIW slots
/// are present at what bit positions.
///
/// The `fixed_mask`/`fixed_value` pair enables format discrimination:
/// `(word & fixed_mask) == fixed_value` uniquely identifies this format
/// within its size group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeFormatDef {
    /// Record name (e.g., "I128_LDB_LDA_ST_ALU_MV_VEC").
    pub name: String,
    /// Total size in bytes (e.g., 16 for 128-bit).
    pub total_bytes: u8,
    /// Total size in bits.
    pub total_bits: u16,
    /// Slot fields present: (slot_name, bit_width).
    /// Example: [("ldb", 16), ("lda", 21), ("st", 21), ("alu", 20), ("mv", 22), ("vec", 26)]
    pub slots: Vec<(String, u16)>,
    /// Bitmask: bit N = 1 if that position in the Inst field is a fixed 0 or 1.
    pub fixed_mask: u128,
    /// Expected value at fixed positions: `(word & fixed_mask) == fixed_value`.
    pub fixed_value: u128,
    /// Per-slot extraction info derived from the Inst field.
    pub slot_maps: Vec<SlotBitMap>,
}

// ============================================================================
// Combined tblgen output
// ============================================================================

/// Complete output from parsing `llvm-tblgen --print-records`.
///
/// This bundles instruction encodings with scheduling, register, and format
/// metadata -- everything needed to build a fully data-driven emulator.
pub struct TblgenOutput {
    /// Instruction encodings grouped by slot name.
    pub encodings_by_slot: HashMap<String, Vec<super::resolver::InstrEncoding>>,
    /// Processor scheduling model (LoadLatency, MispredictPenalty, etc.).
    pub processor_model: Option<ProcessorModel>,
    /// Per-itinerary-class scheduling info, keyed by class name (e.g., "II_ADD").
    pub itineraries: HashMap<String, ItineraryInfo>,
    /// Functional units in processor-itinerary order; the decoder's resolved stages name their
    /// unit by index into this list.
    pub functional_units: Vec<String>,
    /// Distinct `(itinerary class, register class of the first output operand)` pairs: the
    /// register classes the instructions of one scheduling class write.
    pub itinerary_output_classes: Vec<(String, String)>,
    /// Register definitions and classes.
    pub register_model: RegisterModel,
    /// Composite VLIW bundle formats.
    pub composite_formats: Vec<CompositeFormatDef>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_op_properties() {
        assert!(SemanticOp::Add.is_commutative());
        assert!(SemanticOp::Mul.is_commutative());
        assert!(!SemanticOp::Sub.is_commutative());
        assert!(!SemanticOp::Shl.is_commutative());

        assert!(SemanticOp::SetEq.is_comparison());
        assert!(SemanticOp::SetLt.is_comparison());
        assert!(!SemanticOp::Add.is_comparison());
    }

    #[test]
    fn test_int64_bits() {
        assert_eq!(ElementType::Int4.bits(), 4);
        assert_eq!(ElementType::UInt4.bits(), 4);
        assert_eq!(ElementType::Int64.bits(), 64);
        assert_eq!(ElementType::UInt64.bits(), 64);
    }

    #[test]
    fn test_int64_lanes() {
        assert_eq!(ElementType::Int4.lanes_256(), 64);
        assert_eq!(ElementType::Int64.lanes_256(), 4);
        assert_eq!(ElementType::UInt64.lanes_256(), 4);
    }

    #[test]
    fn test_int64_signed() {
        assert!(ElementType::Int4.is_signed());
        assert!(!ElementType::UInt4.is_signed());
        assert!(ElementType::Int64.is_signed());
        assert!(!ElementType::UInt64.is_signed());
    }

    #[test]
    fn with_signedness_flips_integers_and_keeps_floats() {
        use ElementType::*;
        for (u, s) in [(UInt4, Int4), (UInt8, Int8), (UInt16, Int16), (UInt32, Int32), (UInt64, Int64)] {
            assert_eq!(u.with_signedness(true), s);
            assert_eq!(s.with_signedness(false), u);
            assert_eq!(u.with_signedness(false), u);
            assert_eq!(s.with_signedness(true), s);
        }
        assert_eq!(BFloat16.with_signedness(false), BFloat16);
        assert_eq!(Float32.with_signedness(true), Float32);
    }
}
