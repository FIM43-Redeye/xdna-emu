//! Type definitions for TableGen parsing.
//!
//! These types represent the parsed structure of LLVM TableGen files,
//! specifically the AIE2 instruction definitions from llvm-aie.
//!
//! ## Semantic Information
//!
//! Beyond encoding, we extract semantic info from TableGen:
//! - **Attributes**: `mayLoad`, `mayStore`, `hasSideEffects`, `Defs`, `Uses`
//! - **Patterns**: `Pat<(add ...), (ADD ...)>` tells us ADD performs addition
//!
//! This allows auto-generating most execution logic, not just decoding.

use std::collections::{HashMap, HashSet};

/// A VLIW slot definition from AIE2Slots.td.
///
/// Slots define the bit widths for different instruction categories:
/// - `lda`: Load A (21 bits)
/// - `ldb`: Load B (16 bits)
/// - `alu`: ALU operations (20 bits)
/// - `mv`: Move operations (22 bits)
/// - `st`: Store (21 bits)
/// - `vec`: Vector operations (26 bits)
/// - `lng`: Long instructions (42 bits)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotDef {
    /// Internal name (e.g., "lda_slot")
    pub name: String,
    /// Display name (e.g., "Lda")
    pub display_name: String,
    /// Bit width of this slot
    pub bits: u8,
    /// Field name to match in encoding (e.g., "lda")
    pub field: String,
    /// Whether this is an artificial slot (e.g., nop_slot)
    pub artificial: bool,
}

/// A part of an instruction encoding.
///
/// Encodings are specified in TableGen as concatenations like:
/// ```tablegen
/// let alu = {mRx0, mRx, c7s, 0b11, 0b0};
/// ```
///
/// Each part is either a field reference or a literal value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodingPart {
    /// Reference to an operand field, optionally with bit range.
    /// - `mRx` -> FieldRef { name: "mRx", high: None, low: None }
    /// - `i{10-6}` -> FieldRef { name: "i", high: Some(10), low: Some(6) }
    FieldRef {
        name: String,
        /// High bit of slice (inclusive), if specified
        high: Option<u8>,
        /// Low bit of slice (inclusive), if specified
        low: Option<u8>,
    },
    /// Literal bit pattern.
    /// - `0b11` -> Literal { value: 0b11, width: 2 }
    /// - `0b0` -> Literal { value: 0, width: 1 }
    Literal {
        value: u64,
        width: u8,
    },
    /// A dontcare field: `dontcare{N}` or `dontcare{high-low}`
    DontCare {
        width: u8,
    },
}

impl EncodingPart {
    /// Returns the bit width of this encoding part.
    pub fn width(&self, field_widths: &HashMap<String, u8>) -> Option<u8> {
        match self {
            EncodingPart::FieldRef { name, high, low } => {
                if let (Some(h), Some(l)) = (high, low) {
                    Some(h - l + 1)
                } else {
                    field_widths.get(name).copied()
                }
            }
            EncodingPart::Literal { width, .. } => Some(*width),
            EncodingPart::DontCare { width } => Some(*width),
        }
    }
}

/// A template parameter in a format class definition.
///
/// Example: `bits<4> op` becomes TemplateParam { name: "op", bits: 4 }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateParam {
    pub name: String,
    pub bits: u8,
}

/// An instruction format class from AIE2GenInstrFormats.td.
///
/// Format classes define the encoding layout for a family of instructions.
/// Example:
/// ```tablegen
/// class AIE2_alu_r_rr_inst_alu<bits<4> op, ...> : AIE2_inst_alu_instr32 {
///   bits<5> mRx, mRx0, mRy;
///   let alu = {mRx0, mRx, mRy, op, 0b1};
/// }
/// ```
///
/// Computed field assignments are also tracked:
/// ```tablegen
/// class AIE2_mLockId_imm : AIE2_mLockId {
///   bits<6> id;
///   let mLockId = {id, 0b0};  // mLockId is derived from id
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatClass {
    /// Class name (e.g., "AIE2_alu_r_rr_inst_alu")
    pub name: String,
    /// Parent class name (e.g., "AIE2_inst_alu_instr32")
    pub parent: Option<String>,
    /// Template parameters (e.g., [TemplateParam { name: "op", bits: 4 }])
    pub template_params: Vec<TemplateParam>,
    /// Field definitions with their bit widths (e.g., {"mRx": 5, "mRy": 5})
    pub fields: HashMap<String, u8>,
    /// Slot field name (e.g., "alu") - determines which slot this format uses
    pub slot_field: Option<String>,
    /// Encoding parts in MSB-first order
    pub encoding: Vec<EncodingPart>,
    /// Computed field sources: maps derived field to source operand fields.
    /// e.g., "mLockId" -> ["id"] when `let mLockId = {id, 0b0}` is parsed.
    /// This is essential for mapping encoding fields back to DAG operand names.
    pub field_sources: HashMap<String, Vec<String>>,
}

impl FormatClass {
    /// Returns the total bit width of the encoding.
    pub fn encoding_width(&self) -> Option<u8> {
        let mut total: u8 = 0;
        for part in &self.encoding {
            total = total.checked_add(part.width(&self.fields)?)?;
        }
        Some(total)
    }

    /// Checks if this format class inherits from a slot instruction class.
    pub fn slot_from_parent(&self) -> Option<&str> {
        self.parent.as_ref().and_then(|p| {
            if p.contains("_alu_") {
                Some("alu")
            } else if p.contains("_lda_") {
                Some("lda")
            } else if p.contains("_ldb_") {
                Some("ldb")
            } else if p.contains("_st_") {
                Some("st")
            } else if p.contains("_mv_") {
                Some("mv")
            } else if p.contains("_vec_") {
                Some("vec")
            } else if p.contains("_lng_") {
                Some("lng")
            } else {
                None
            }
        })
    }
}

/// A mixin class from AIE2GenInstrFormats.td.
///
/// Mixin classes are parameterless classes that provide field mappings
/// to instruction definitions via multiple inheritance. Unlike format classes,
/// they don't have template parameters or encoding - just field declarations
/// and computed field assignments.
///
/// Example:
/// ```tablegen
/// class AIE2_mLockId_imm : AIE2_mLockId {
///   bits<6> id;
///   let mLockId = {id, 0b0};  // mLockId is derived from id
/// }
/// ```
///
/// Instructions use mixins via multiple inheritance:
/// ```tablegen
/// def ACQ : FormatClass<...>, AIE2_mLockId_imm;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixinClass {
    /// Class name (e.g., "AIE2_mLockId_imm")
    pub name: String,
    /// Parent class name (e.g., "AIE2_mLockId")
    pub parent: Option<String>,
    /// Field definitions with their bit widths (e.g., {"id": 6})
    pub fields: HashMap<String, u8>,
    /// Computed field sources: maps derived field to source operand fields.
    /// e.g., "mLockId" -> ["id"] when `let mLockId = {id, 0b0}` is parsed.
    pub field_sources: HashMap<String, Vec<String>>,
}

/// An operand definition from an instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandDef {
    /// Whether this is an output (true) or input (false)
    pub is_output: bool,
    /// Register class (e.g., "eR", "eP", "eM")
    pub reg_class: String,
    /// Operand name from asm string (e.g., "mRx", "ptr")
    pub name: String,
}

/// Instruction attributes extracted from TableGen.
///
/// These come from `let` statements in instruction definitions:
/// ```tablegen
/// let hasSideEffects = false, mayLoad = false, mayStore = false in {
///   let Defs = [srCarry] in
///   def ADD : ...
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InstrAttributes {
    /// Whether this instruction may load from memory
    pub may_load: bool,
    /// Whether this instruction may store to memory
    pub may_store: bool,
    /// Whether this instruction has side effects beyond outputs
    pub has_side_effects: bool,
    /// Registers implicitly defined (written) by this instruction
    pub defs: HashSet<String>,
    /// Registers implicitly used (read) by this instruction
    pub uses: HashSet<String>,
}

/// LLVM SelectionDAG node types that map to semantic operations.
///
/// These come from `Pat<>` patterns in AIE2InstrPatterns.td.
/// The SDNode name tells us what the instruction actually computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticOp {
    // Arithmetic
    Add,
    Sub,
    Adc,  // Add with carry (reads carry flag before computation)
    Sbc,  // Subtract with borrow (reads carry flag before computation)
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
    Shl,   // Shift left
    Sra,   // Shift right arithmetic (signed)
    Srl,   // Shift right logical (unsigned)
    // Bidirectional shifts (AIE2 ASHL/LSHL hardware instructions).
    // Positive shift amount = left, negative = right.
    // ASHL: right-shifts are arithmetic (sign-preserving).
    // LSHL: right-shifts are logical (zero-filling).
    AshlBidir,
    LshlBidir,
    Rotl,  // Rotate left
    Rotr,  // Rotate right

    // Comparison (result is 0 or 1)
    SetEq,   // ==
    SetNe,   // !=
    SetLt,   // < (signed)
    SetLe,   // <= (signed)
    SetGt,   // > (signed)
    SetGe,   // >= (signed)
    SetUlt,  // < (unsigned)
    SetUle,  // <= (unsigned)
    SetUgt,  // > (unsigned)
    SetUge,  // >= (unsigned)

    // Bit manipulation
    Ctlz,    // Count leading zeros
    Cttz,    // Count trailing zeros
    Ctpop,   // Population count (count ones)
    Bswap,   // Byte swap

    // Memory
    Load,
    Store,

    // Control flow
    Br,      // Unconditional branch
    BrCond,  // Conditional branch
    Call,    // Call subroutine (jl: sets LR after delay slots)
    Ret,     // Return from subroutine
    Select,  // Conditional select (ternary)

    // Sign/zero extension
    SignExtend,
    ZeroExtend,
    Truncate,

    // Special
    Copy,    // Move/copy
    Nop,     // No operation
    Done,    // Core termination (halt)
    Event,   // Generate trace event (operand selects INSTR_EVENT_0/1)

    // Synchronization (AIE-specific)
    LockAcquire,  // Acquire lock
    LockRelease,  // Release lock

    // Bit manipulation (scalar-only)
    Clb,  // Count leading bits (ones or zeros, != CLZ)
    Cmp,  // Compare (flag-setting, no destination register)

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
    VectorSelect,    // Per-lane conditional select
    VectorClear,     // Clear vector to zero
    Convert,         // Type conversion (vconv, vfloor, vceil, etc.)
    Min,             // Minimum (scalar or vector)
    Max,             // Maximum (scalar or vector)

    // Conditional vector operations (AIE2 compound ops)
    SubLt,        // dst[i] = (a < b) ? a - b : a
    SubGe,        // dst[i] = (a >= b) ? a - b : a
    MaxDiffLt,    // dst[i] = max(a - b, 0) when a < b
    MaxLt,        // dst = max(a, b) with less-than flag
    MinGe,        // dst = min(a, b) with greater-equal flag
    AbsGtz,       // dst[i] = (src > 0) ? abs(src) : src
    NegGtz,       // dst[i] = (src > 0) ? -src : src
    NegLtz,       // dst[i] = (src < 0) ? -src : src
    NegAdd,       // dst = -src1 + src2
    NegMul,       // acc += -(src1 * src2)
    Accumulate,   // acc += src (no multiply)

    // Side-effect operations
    CascadeRead,            // Read from cascade input
    CascadeWrite,           // Write to cascade output
    StreamRead,             // Read from slave stream
    StreamWrite,            // Write to master stream
    StreamWritePacketHeader, // Write packet header to master stream
    DmaStart,               // Start DMA transfer
    DmaWait,                // Wait for DMA completion
    Halt,                   // Core termination (halt)

    // Hardware state reads
    ReadCycleCounter,       // MOV_CNTR: read per-core cycle counter

    // Pointer operations
    PointerAdd,  // Pointer arithmetic: ptr + offset
    PointerMov,  // Pointer move: ptr = value

    // Target-specific intrinsic (needs name lookup)
    Intrinsic(u32),  // Index into intrinsic name table
}

impl SemanticOp {
    /// Parse an LLVM SDNode name into a SemanticOp.
    pub fn from_sdnode(name: &str) -> Option<Self> {
        Some(match name {
            // Arithmetic
            "add" => Self::Add,
            "sub" => Self::Sub,
            "mul" => Self::Mul,
            "sdiv" => Self::SDiv,
            "udiv" => Self::UDiv,
            "srem" => Self::SRem,
            "urem" => Self::URem,
            "abs" => Self::Abs,
            "neg" | "ineg" => Self::Neg,

            // Bitwise
            "and" => Self::And,
            "or" => Self::Or,
            "xor" => Self::Xor,
            "not" => Self::Not,
            "shl" => Self::Shl,
            "sra" => Self::Sra,
            "srl" => Self::Srl,
            "rotl" => Self::Rotl,
            "rotr" => Self::Rotr,

            // Comparison
            "seteq" => Self::SetEq,
            "setne" => Self::SetNe,
            "setlt" => Self::SetLt,
            "setle" => Self::SetLe,
            "setgt" => Self::SetGt,
            "setge" => Self::SetGe,
            "setult" => Self::SetUlt,
            "setule" => Self::SetUle,
            "setugt" => Self::SetUgt,
            "setuge" => Self::SetUge,

            // Bit manipulation
            "ctlz" | "ctlz_zero_undef" => Self::Ctlz,
            "cttz" | "cttz_zero_undef" => Self::Cttz,
            "ctpop" => Self::Ctpop,
            "bswap" => Self::Bswap,

            // Floating-point arithmetic
            "fadd" => Self::Add,
            "fsub" => Self::Sub,
            "fmul" => Self::Mul,

            // Memory / pointer
            "load" => Self::Load,
            "store" => Self::Store,
            "ptradd" => Self::PointerAdd,

            // Control
            "br" => Self::Br,
            "brcond" => Self::BrCond,
            "select" => Self::Select,

            // Extensions
            "sext" | "sign_extend" | "sext_inreg" => Self::SignExtend,
            "zext" | "zero_extend" => Self::ZeroExtend,
            "trunc" | "truncate" => Self::Truncate,

            // Special
            "copy" | "CopyToReg" | "CopyFromReg" => Self::Copy,

            _ => return None,
        })
    }

    /// Parse an LLVM intrinsic name into a SemanticOp.
    ///
    /// Intrinsic names come from Pat<> patterns in AIE2InstrPatterns.td where
    /// the source operand is an intrinsic call (e.g., `int_aie2_vshuffle`)
    /// rather than a standard SDNode (e.g., `add`). These names are stable
    /// machine-generated identifiers from LLVM's intrinsic definitions.
    ///
    /// Strategy: strip `int_aie2_` prefix, then match suffixes first (for
    /// matmul/MAC variants), then prefixes (for vector ops), then exact
    /// matches (for simple ops).
    pub fn from_intrinsic(name: &str) -> Option<Self> {
        let stem = name.strip_prefix("int_aie2_")?;

        // Suffix-based matching for _conf family (matmul/MAC variants).
        // These end with `_OPTYPE[DIGITS]_conf` where DIGITS is optional
        // (e.g., `_mul_conf`, `_mul16_conf`, `_mac_conf`).
        // Strip `_conf` suffix, then strip trailing digits, then match.
        if let Some(before_conf) = stem.strip_suffix("_conf") {
            let base = before_conf.trim_end_matches(|c: char| c.is_ascii_digit());
            // Most specific first: negmac/negmsc before mac/msc
            if base.ends_with("_negmac") || base.ends_with("_negmsc") {
                return Some(Self::NegMul);
            }
            if base.ends_with("_negmul") {
                return Some(Self::NegMatMul);
            }
            if base.ends_with("_addmac") || base.ends_with("_addmsc") {
                return Some(Self::AddMac);
            }
            if base.ends_with("_submac") || base.ends_with("_submsc") {
                return Some(Self::SubMac);
            }
            if base.ends_with("_mac") || base.ends_with("_msc") {
                return Some(Self::Mac);
            }
            if base.ends_with("_mul") {
                return Some(Self::MatMul);
            }
            if base.ends_with("_neg") {
                return Some(Self::Neg);
            }
            // vaddsub_conf and similar compound ops ending in _conf
            if base.starts_with("vaddsub") {
                return Some(Self::Add);
            }
            // clr16f_conf and similar vector clear intrinsics
            if base.starts_with("clr") {
                return Some(Self::VectorClear);
            }
        }

        // Accumulate intrinsics: add_acc, sub_acc, negadd_acc, negsub_acc
        // These are accumulator operations without multiply (acc += src).
        if stem.starts_with("add_acc") || stem.starts_with("sub_acc")
            || stem.starts_with("negadd_acc") || stem.starts_with("negsub_acc")
        {
            return Some(Self::Accumulate);
        }

        // Concat intrinsics: concat_I512_I256, concat_bf1024_bf512, etc.
        if stem.starts_with("concat_") {
            return Some(Self::Shuffle);
        }

        // Suffix-based matching (non-_conf)
        if stem.ends_with("_ups") {
            return Some(Self::Ups);
        }
        if stem.ends_with("_srs") {
            return Some(Self::Srs);
        }

        // Prefix-based matching (vector operations)
        if stem.starts_with("vshuffle") || stem.starts_with("vbcst_shuffle") {
            return Some(Self::Shuffle);
        }
        if stem.starts_with("vshift") {
            return Some(Self::Align);
        }
        if stem.starts_with("vsel") {
            return Some(Self::VectorSelect);
        }
        if stem.starts_with("vbroadcast_zero") {
            return Some(Self::VectorClear);
        }
        if stem.starts_with("vbroadcast") {
            return Some(Self::VectorBroadcast);
        }
        if stem.starts_with("vextract_broadcast") || stem.starts_with("vextract_bcast") {
            return Some(Self::VectorBroadcast);
        }
        if stem.starts_with("vextract_elem") {
            return Some(Self::VectorExtract);
        }
        if stem.starts_with("vinsert") {
            return Some(Self::VectorInsert);
        }
        if stem.starts_with("vmaxdiff_lt") {
            return Some(Self::MaxDiffLt);
        }
        if stem.starts_with("vsub_lt") {
            return Some(Self::SubLt);
        }
        if stem.starts_with("vsub_ge") {
            return Some(Self::SubGe);
        }
        if stem.starts_with("vlt") || stem.starts_with("vmax_lt") {
            return Some(Self::MaxLt);
        }
        if stem.starts_with("vge") || stem.starts_with("vmin_ge") {
            return Some(Self::MinGe);
        }
        if stem.starts_with("veqz") {
            return Some(Self::SetEq);
        }
        if stem.starts_with("vabs_gtz") {
            return Some(Self::AbsGtz);
        }
        if stem.starts_with("vneg_gtz") {
            return Some(Self::NegGtz);
        }
        if stem.starts_with("vbneg_ltz") {
            return Some(Self::NegLtz);
        }
        // Conversion intrinsics: v*_to_v* (e.g., vconv, vfloor, vceil)
        if stem.contains("_to_v") {
            return Some(Self::Convert);
        }

        // Load/update/set/extract sub-vector intrinsics
        if stem.starts_with("load_4x") {
            return Some(Self::Load);
        }
        if stem.starts_with("upd_") || stem.starts_with("set_") {
            return Some(Self::Copy);
        }
        if stem.starts_with("ext_") {
            return Some(Self::Copy);
        }

        // Lock operations
        if stem.starts_with("acquire") {
            return Some(Self::LockAcquire);
        }
        if stem.starts_with("release") {
            return Some(Self::LockRelease);
        }

        // Exact matches
        match stem {
            "done" => Some(Self::Done),
            "event" | "event0" | "event1" => Some(Self::Event),
            "clb" => Some(Self::Clb),
            "divs" => Some(Self::SDiv),
            "sched_barrier" => Some(Self::Nop),
            _ => None,
        }
    }

    /// Returns true if this is a commutative operation.
    pub fn is_commutative(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::And | Self::Or | Self::Xor |
            Self::SetEq | Self::SetNe
        )
    }

    /// Returns true if this is a comparison operation.
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::SetEq | Self::SetNe |
            Self::SetLt | Self::SetLe | Self::SetGt | Self::SetGe |
            Self::SetUlt | Self::SetUle | Self::SetUgt | Self::SetUge
        )
    }
}

/// Element type for vector operations.
///
/// Defined here in tablegen::types so the resolver can populate it during
/// TableGen loading without depending on the interpreter module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
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
    /// 16-bit brain floating point.
    BFloat16,
    /// 32-bit floating point.
    Float32,
}

impl ElementType {
    /// Get the size of this element type in bits.
    pub fn bits(self) -> u8 {
        match self {
            ElementType::Int8 | ElementType::UInt8 => 8,
            ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => 16,
            ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => 32,
        }
    }

    /// Get the number of elements that fit in a 256-bit vector.
    pub fn lanes_256(self) -> u8 {
        (256u16 / self.bits() as u16) as u8
    }

    /// Whether this element type is signed (Int8, Int16, Int32).
    pub fn is_signed(self) -> bool {
        matches!(self, ElementType::Int8 | ElementType::Int16 | ElementType::Int32)
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

/// A semantic pattern from AIE2InstrPatterns.td.
///
/// Links an LLVM SDNode operation to a concrete instruction.
/// Example: `Pat<(add eR:$rs1, eR:$rs2), (ADD eR:$rs1, eR:$rs2)>`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticPattern {
    /// The semantic operation (what it computes)
    pub operation: SemanticOp,
    /// The instruction name that implements it
    pub instruction: String,
    /// Number of operands (helps disambiguate patterns)
    pub operand_count: u8,
    /// For intrinsics, the full intrinsic name
    pub intrinsic_name: Option<String>,
}

/// An implicit register use/def from an instruction.
///
/// Some instructions use fixed registers that aren't variable operands.
/// For example, `sel.eqz` always tests r27 via the `eR27` register class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitReg {
    /// Register class that constrains to a single register (e.g., "eR27")
    pub reg_class: String,
    /// The fixed register number (e.g., 27 for eR27)
    pub reg_num: u8,
    /// Whether this is a use (read) or def (write)
    pub is_use: bool,
}

/// A concrete instruction definition from AIE2GenInstrInfo.td.
///
/// Example:
/// ```tablegen
/// def ADD : AIE2_alu_r_rr_inst_alu<0b0000, (outs eR:$mRx), (ins eR:$mRx0, eR:$mRy), "add", "$mRx, $mRx0, $mRy">;
/// ```
///
/// For instructions with implicit registers (like sel.eqz which always uses r27):
/// ```tablegen
/// def SELEQZ : AIE2_select_r_rr_inst_alu<0b0, (outs eR:$mRx),
///     (ins eR:$mRx0, eR:$mRy, eR27:$s2), "sel.eqz", "$mRx, $mRx0, $mRy, r27">;
/// ```
/// The `eR27` register class constrains `$s2` to only r27, making it implicit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstrDef {
    /// Instruction name (e.g., "ADD")
    pub name: String,
    /// Format class this instantiates (e.g., "AIE2_alu_r_rr_inst_alu")
    pub format: String,
    /// Mixin classes providing field mappings (e.g., ["AIE2_mLockId_imm"])
    pub mixin_classes: Vec<String>,
    /// Template arguments (e.g., [0b0000] for the `op` parameter)
    pub template_args: Vec<u64>,
    /// Assembly mnemonic (e.g., "add")
    pub mnemonic: String,
    /// Assembly format string (e.g., "$mRx, $mRx0, $mRy")
    pub asm_string: String,
    /// Output operands (variable, extractable from encoding)
    pub outputs: Vec<OperandDef>,
    /// Input operands (variable, extractable from encoding)
    pub inputs: Vec<OperandDef>,
    /// Implicit register uses/defs (fixed registers like r27 for sel.eqz)
    pub implicit_regs: Vec<ImplicitReg>,
    /// Instruction attributes (mayLoad, Defs, etc.)
    pub attributes: InstrAttributes,
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
    /// Hardware encoding value (from `bits<16> HWEncoding`).
    pub hw_encoding: u16,
    /// Parent classes from the def line (e.g., ["AIE2GPReg", "DwarfRegNum"]).
    pub parents: Vec<String>,
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
    /// Parent classes from the def line.
    pub parents: Vec<String>,
}

/// Complete register model extracted from TableGen.
#[derive(Debug, Clone, Default)]
pub struct RegisterModel {
    /// All individual register definitions, indexed by name.
    pub registers: HashMap<String, RegisterDef>,
    /// All register classes, indexed by class name.
    pub classes: HashMap<String, RegisterClassDef>,
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
    /// Register definitions and classes.
    pub register_model: RegisterModel,
    /// Composite VLIW bundle formats.
    pub composite_formats: Vec<CompositeFormatDef>,
    /// LLVM decoder bytecode tables, keyed by slot name.
    /// Extracted from `llvm-tblgen -gen-disassembler`.
    /// These provide authoritative disambiguation for instruction decoding.
    pub decoder_tables: HashMap<String, super::decoder_bytecode::DecoderTable>,
}

/// Collection of all parsed TableGen definitions.
#[derive(Debug, Default)]
pub struct TableGenData {
    /// Slot definitions by name
    pub slots: HashMap<String, SlotDef>,
    /// Format classes by name (parameterized, with encodings)
    pub formats: HashMap<String, FormatClass>,
    /// Mixin classes by name (parameterless, provide field mappings)
    pub mixins: HashMap<String, MixinClass>,
    /// Instruction definitions by name
    pub instructions: HashMap<String, InstrDef>,
    /// Semantic patterns (instruction → what it computes)
    pub patterns: Vec<SemanticPattern>,
    /// Intrinsic name table (for SemanticOp::Intrinsic indices)
    pub intrinsic_names: Vec<String>,
}

impl TableGenData {
    pub fn new() -> Self {
        Self::default()
    }

    /// Looks up the slot definition for a format class.
    pub fn slot_for_format(&self, format_name: &str) -> Option<&SlotDef> {
        let format = self.formats.get(format_name)?;

        // First check explicit slot_field, then try to infer from parent
        let slot_field: &str = if let Some(ref field) = format.slot_field {
            field.as_str()
        } else if let Some(field) = format.slot_from_parent() {
            field
        } else {
            return None;
        };

        // Find slot by field name
        self.slots.values().find(|s| s.field == slot_field)
    }

    /// Get the semantic operation for an instruction, if known.
    pub fn semantic_for_instruction(&self, instr_name: &str) -> Option<&SemanticPattern> {
        self.patterns.iter().find(|p| p.instruction == instr_name)
    }

    /// Get all instructions that implement a given semantic operation.
    pub fn instructions_for_semantic(&self, op: SemanticOp) -> Vec<&SemanticPattern> {
        self.patterns.iter().filter(|p| p.operation == op).collect()
    }

    /// Look up an intrinsic name by index.
    pub fn intrinsic_name(&self, index: u32) -> Option<&str> {
        self.intrinsic_names.get(index as usize).map(|s| s.as_str())
    }

    /// Add an intrinsic and return its index.
    pub fn add_intrinsic(&mut self, name: String) -> u32 {
        // Check if already exists
        if let Some(idx) = self.intrinsic_names.iter().position(|n| n == &name) {
            return idx as u32;
        }
        let idx = self.intrinsic_names.len() as u32;
        self.intrinsic_names.push(name);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_def() {
        let slot = SlotDef {
            name: "alu_slot".to_string(),
            display_name: "Alu".to_string(),
            bits: 20,
            field: "alu".to_string(),
            artificial: false,
        };
        assert_eq!(slot.bits, 20);
        assert_eq!(slot.field, "alu");
    }

    #[test]
    fn test_encoding_part_width() {
        let mut fields = HashMap::new();
        fields.insert("mRx".to_string(), 5);

        // Field reference without slice
        let part = EncodingPart::FieldRef {
            name: "mRx".to_string(),
            high: None,
            low: None,
        };
        assert_eq!(part.width(&fields), Some(5));

        // Field reference with slice
        let part = EncodingPart::FieldRef {
            name: "i".to_string(),
            high: Some(10),
            low: Some(6),
        };
        assert_eq!(part.width(&fields), Some(5)); // 10-6+1 = 5

        // Literal
        let part = EncodingPart::Literal { value: 0b11, width: 2 };
        assert_eq!(part.width(&fields), Some(2));

        // DontCare
        let part = EncodingPart::DontCare { width: 3 };
        assert_eq!(part.width(&fields), Some(3));
    }

    #[test]
    fn test_format_class_encoding_width() {
        let mut fields = HashMap::new();
        fields.insert("mRx0".to_string(), 5);
        fields.insert("mRx".to_string(), 5);
        fields.insert("mRy".to_string(), 5);
        fields.insert("op".to_string(), 4);

        let format = FormatClass {
            name: "AIE2_alu_r_rr_inst_alu".to_string(),
            parent: Some("AIE2_inst_alu_instr32".to_string()),
            template_params: vec![TemplateParam {
                name: "op".to_string(),
                bits: 4,
            }],
            fields: fields.clone(),
            slot_field: Some("alu".to_string()),
            encoding: vec![
                EncodingPart::FieldRef {
                    name: "mRx0".to_string(),
                    high: None,
                    low: None,
                },
                EncodingPart::FieldRef {
                    name: "mRx".to_string(),
                    high: None,
                    low: None,
                },
                EncodingPart::FieldRef {
                    name: "mRy".to_string(),
                    high: None,
                    low: None,
                },
                EncodingPart::FieldRef {
                    name: "op".to_string(),
                    high: None,
                    low: None,
                },
                EncodingPart::Literal { value: 1, width: 1 },
            ],
            field_sources: HashMap::new(),
        };

        // 5 + 5 + 5 + 4 + 1 = 20 bits
        assert_eq!(format.encoding_width(), Some(20));
    }

    #[test]
    fn test_instr_def() {
        let mut defs = HashSet::new();
        defs.insert("srCarry".to_string());

        let instr = InstrDef {
            name: "ADD".to_string(),
            format: "AIE2_alu_r_rr_inst_alu".to_string(),
            mixin_classes: vec![],
            template_args: vec![0b0000],
            mnemonic: "add".to_string(),
            asm_string: "$mRx, $mRx0, $mRy".to_string(),
            outputs: vec![OperandDef {
                is_output: true,
                reg_class: "eR".to_string(),
                name: "mRx".to_string(),
            }],
            inputs: vec![
                OperandDef {
                    is_output: false,
                    reg_class: "eR".to_string(),
                    name: "mRx0".to_string(),
                },
                OperandDef {
                    is_output: false,
                    reg_class: "eR".to_string(),
                    name: "mRy".to_string(),
                },
            ],
            implicit_regs: vec![],
            attributes: InstrAttributes {
                may_load: false,
                may_store: false,
                has_side_effects: false,
                defs,
                uses: HashSet::new(),
            },
        };

        assert_eq!(instr.name, "ADD");
        assert_eq!(instr.template_args[0], 0b0000);
        assert_eq!(instr.outputs.len(), 1);
        assert_eq!(instr.inputs.len(), 2);
        assert!(instr.attributes.defs.contains("srCarry"));
    }

    #[test]
    fn test_semantic_op_from_sdnode() {
        assert_eq!(SemanticOp::from_sdnode("add"), Some(SemanticOp::Add));
        assert_eq!(SemanticOp::from_sdnode("sub"), Some(SemanticOp::Sub));
        assert_eq!(SemanticOp::from_sdnode("mul"), Some(SemanticOp::Mul));
        assert_eq!(SemanticOp::from_sdnode("and"), Some(SemanticOp::And));
        assert_eq!(SemanticOp::from_sdnode("or"), Some(SemanticOp::Or));
        assert_eq!(SemanticOp::from_sdnode("xor"), Some(SemanticOp::Xor));
        assert_eq!(SemanticOp::from_sdnode("shl"), Some(SemanticOp::Shl));
        assert_eq!(SemanticOp::from_sdnode("sra"), Some(SemanticOp::Sra));
        assert_eq!(SemanticOp::from_sdnode("ctlz"), Some(SemanticOp::Ctlz));
        assert_eq!(SemanticOp::from_sdnode("select"), Some(SemanticOp::Select));
        assert_eq!(SemanticOp::from_sdnode("ptradd"), Some(SemanticOp::PointerAdd));
        assert_eq!(SemanticOp::from_sdnode("fadd"), Some(SemanticOp::Add));
        assert_eq!(SemanticOp::from_sdnode("fsub"), Some(SemanticOp::Sub));
        assert_eq!(SemanticOp::from_sdnode("fmul"), Some(SemanticOp::Mul));
        assert_eq!(SemanticOp::from_sdnode("unknown_op"), None);
    }

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
    fn test_from_intrinsic_requires_prefix() {
        // Must start with int_aie2_
        assert_eq!(SemanticOp::from_intrinsic("vshuffle"), None);
        assert_eq!(SemanticOp::from_intrinsic("int_aie_vshuffle"), None);
    }

    #[test]
    fn test_from_intrinsic_matmul_family() {
        // _mul_conf -> MatMul
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_mul_conf"),
            Some(SemanticOp::MatMul)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc32_mul_conf"),
            Some(SemanticOp::MatMul)
        );
        // _mac_conf -> Mac
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_mac_conf"),
            Some(SemanticOp::Mac)
        );
        // _msc_conf -> Mac
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_msc_conf"),
            Some(SemanticOp::Mac)
        );
        // _negmul_conf -> NegMatMul
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_negmul_conf"),
            Some(SemanticOp::NegMatMul)
        );
        // _negmac_conf -> NegMul
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_negmac_conf"),
            Some(SemanticOp::NegMul)
        );
        // _negmsc_conf -> NegMul
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_negmsc_conf"),
            Some(SemanticOp::NegMul)
        );
        // _addmac_conf -> AddMac
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_addmac_conf"),
            Some(SemanticOp::AddMac)
        );
        // _addmsc_conf -> AddMac
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_addmsc_conf"),
            Some(SemanticOp::AddMac)
        );
        // _submac_conf -> SubMac
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_submac_conf"),
            Some(SemanticOp::SubMac)
        );
        // _submsc_conf -> SubMac
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_I512_acc64_submsc_conf"),
            Some(SemanticOp::SubMac)
        );
    }

    #[test]
    fn test_from_intrinsic_neg_conf() {
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I512_neg_conf"),
            Some(SemanticOp::Neg)
        );
    }

    #[test]
    fn test_from_intrinsic_ups_srs() {
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_acc32_v16_I256_ups"),
            Some(SemanticOp::Ups)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_acc64_v16_I256_ups"),
            Some(SemanticOp::Ups)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I256_v16_acc32_srs"),
            Some(SemanticOp::Srs)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I256_v16_acc64_srs"),
            Some(SemanticOp::Srs)
        );
    }

    #[test]
    fn test_from_intrinsic_vector_ops() {
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vshuffle"),
            Some(SemanticOp::Shuffle)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vshuffle_bf16"),
            Some(SemanticOp::Shuffle)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vbcst_shuffle_bf512"),
            Some(SemanticOp::Shuffle)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vshift_I512_I512"),
            Some(SemanticOp::Align)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vsel32"),
            Some(SemanticOp::VectorSelect)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vbroadcast_zero_acc1024"),
            Some(SemanticOp::VectorClear)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vbroadcast32"),
            Some(SemanticOp::VectorBroadcast)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vextract_broadcast_I512"),
            Some(SemanticOp::VectorBroadcast)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vextract_elem32_I512"),
            Some(SemanticOp::VectorExtract)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vinsert32"),
            Some(SemanticOp::VectorInsert)
        );
    }

    #[test]
    fn test_from_intrinsic_compound_ops() {
        // vaddsub ends with _conf, handled by the _conf suffix matcher
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vaddsub_conf"),
            Some(SemanticOp::Add)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vaddsub16_conf"),
            Some(SemanticOp::Add)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vmaxdiff_lt8"),
            Some(SemanticOp::MaxDiffLt)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vsub_lt16"),
            Some(SemanticOp::SubLt)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vsub_ge32"),
            Some(SemanticOp::SubGe)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vmax_lt8"),
            Some(SemanticOp::MaxLt)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vmin_ge16"),
            Some(SemanticOp::MinGe)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_veqz32"),
            Some(SemanticOp::SetEq)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vabs_gtz16"),
            Some(SemanticOp::AbsGtz)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vneg_gtz32"),
            Some(SemanticOp::NegGtz)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_vbneg_ltz16"),
            Some(SemanticOp::NegLtz)
        );
    }

    #[test]
    fn test_from_intrinsic_conversion() {
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_v16int32_to_v16float"),
            Some(SemanticOp::Convert)
        );
    }

    #[test]
    fn test_from_intrinsic_load_and_copy() {
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_load_4x16_lo"),
            Some(SemanticOp::Load)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_load_4x32_hi"),
            Some(SemanticOp::Load)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_upd_I512_I256"),
            Some(SemanticOp::Copy)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_set_I512_I256"),
            Some(SemanticOp::Copy)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_ext_I256_I512"),
            Some(SemanticOp::Copy)
        );
    }

    #[test]
    fn test_from_intrinsic_lock_ops() {
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_acquire"),
            Some(SemanticOp::LockAcquire)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_acquire_cond"),
            Some(SemanticOp::LockAcquire)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_release"),
            Some(SemanticOp::LockRelease)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_release_cond"),
            Some(SemanticOp::LockRelease)
        );
    }

    #[test]
    fn test_from_intrinsic_exact_matches() {
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_done"),
            Some(SemanticOp::Done)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_event"),
            Some(SemanticOp::Event)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_clb"),
            Some(SemanticOp::Clb)
        );
    }

    #[test]
    fn test_from_intrinsic_bf_mul() {
        // BFloat16 multiply uses PatInaccessibleMem in the .td file
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_bf_mul16_conf"),
            Some(SemanticOp::MatMul)
        );
        assert_eq!(
            SemanticOp::from_intrinsic("int_aie2_I1024_I1024_ACC1024_accfloat_bf_mul_conf"),
            Some(SemanticOp::MatMul)
        );
    }
}
