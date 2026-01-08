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
    Ret,     // Return from subroutine
    Select,  // Conditional select (ternary)

    // Sign/zero extension
    SignExtend,
    ZeroExtend,
    Truncate,

    // Special
    Copy,    // Move/copy
    Nop,     // No operation

    // Synchronization (AIE-specific)
    LockAcquire,  // Acquire lock
    LockRelease,  // Release lock

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

            // Memory
            "load" => Self::Load,
            "store" => Self::Store,

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
}
