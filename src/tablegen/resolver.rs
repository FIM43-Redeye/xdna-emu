//! Resolver for computing instruction encodings from TableGen data.
//!
//! The resolver takes parsed TableGen definitions (format classes and instructions)
//! and computes concrete bit masks and field positions needed for decoding.
//!
//! # Overview
//!
//! TableGen defines instruction encodings as templates:
//! ```tablegen
//! class AIE2_alu_r_rr_inst_alu<bits<4> op> {
//!   bits<5> mRx0, mRx, mRy;
//!   let alu = {mRx0, mRx, mRy, op, 0b1};
//! }
//! def ADD : AIE2_alu_r_rr_inst_alu<0b0000>;
//! ```
//!
//! The resolver:
//! 1. Substitutes template parameters (op = 0b0000)
//! 2. Computes fixed bit positions (the literal 0b1 and the resolved 0b0000)
//! 3. Computes operand field positions (mRx0 at bits 19:15, mRx at 14:10, etc.)
//!
//! # Output
//!
//! For each instruction, we produce an [`InstrEncoding`] containing:
//! - `fixed_mask`: Bits that must match for this instruction
//! - `fixed_bits`: Expected values for those bits
//! - `operand_fields`: Where to extract operand values

use std::collections::HashMap;

use super::types::{
    BranchCondition, ElementType, EncodingPart, FormatClass, ImplicitReg, InstrDef, MixinClass,
    SelectVariant, SemanticOp, SlotDef, TableGenData,
};

/// Addressing mode extracted from the TableGen instruction name.
///
/// The AIE2 instruction naming convention encodes the addressing mode:
/// - `_idx_imm` -> IndexedImmediate:   *(ptr + imm), no pointer update
/// - `_idx` (no _imm) -> IndexedRegister: *(ptr + mN), no pointer update
/// - `_pstm_*_imm` -> PostModifyImmediate: t = *ptr, ptr += imm
/// - `_pstm_*` (no _imm) -> PostModifyRegister: t = *ptr, ptr += mN
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressingMode {
    #[default]
    Unknown,
    /// *(ptr + imm), pointer unchanged
    IndexedImmediate,
    /// *(ptr + mN), pointer unchanged
    IndexedRegister,
    /// t = *ptr; ptr += imm
    PostModifyImmediate,
    /// t = *ptr; ptr += mN
    PostModifyRegister,
}

/// Memory access width extracted from the instruction mnemonic.
///
/// The mnemonic suffix determines the width:
/// - `lda.s8` / `lda.u8` -> Byte (8-bit)
/// - `lda.s16` / `lda.u16` -> HalfWord (16-bit)
/// - `lda` (no suffix) -> Word (32-bit)
/// - `vlda` / `vldb` / `vst` -> Vector256 (256-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InstrMemWidth {
    Byte,
    HalfWord,
    #[default]
    Word,
    Vector256,
}

/// Detect addressing mode from the TableGen instruction name.
///
/// Uses the deterministic naming convention from llvm-aie's class hierarchy:
/// format class names propagate into instruction names as suffixes.
pub fn detect_addressing_mode(instr_name: &str) -> AddressingMode {
    let lower = instr_name.to_lowercase();
    if lower.contains("_pstm_") {
        if lower.ends_with("_imm") || lower.contains("_imm_") {
            AddressingMode::PostModifyImmediate
        } else {
            AddressingMode::PostModifyRegister
        }
    } else if lower.contains("_idx") {
        if lower.ends_with("_imm") || lower.contains("_imm_") {
            AddressingMode::IndexedImmediate
        } else {
            AddressingMode::IndexedRegister
        }
    } else {
        AddressingMode::Unknown
    }
}

/// Detect memory access width from the instruction mnemonic.
///
/// Uses the assembly mnemonic suffix convention: `.s8`, `.u16`, etc.
/// Vector operations (`vlda`, `vldb`, `vst`) are always 256-bit.
pub fn detect_mem_width(mnemonic: &str) -> InstrMemWidth {
    let lower = mnemonic.to_lowercase();
    if lower.starts_with("vlda") || lower.starts_with("vldb")
        || lower.starts_with("vst")
    {
        InstrMemWidth::Vector256
    } else if lower.contains(".s8") || lower.contains(".u8") {
        InstrMemWidth::Byte
    } else if lower.contains(".s16") || lower.contains(".u16") {
        InstrMemWidth::HalfWord
    } else {
        InstrMemWidth::Word
    }
}

/// How a raw bit-field value should be decoded into an Operand.
///
/// Determined from the `OperandDef.reg_class` in the TableGen DAG pattern.
/// This replaces all hand-written heuristics in the decoder with a single
/// data-driven dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandType {
    /// Simple register: raw value IS the register index.
    Register(RegisterKind),
    /// Composite register: multiple register classes share one field,
    /// discriminated by low bits. Requires an inverse encoder function.
    CompositeRegister(CompositeEncoder),
    /// Immediate value with sign and scale.
    Immediate { signed: bool, scale: i32 },
    /// Lock ID.
    LockId,
    /// Unknown (safe fallback: treated as unsigned immediate).
    Unknown,
}

/// Which simple register file a field encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterKind {
    /// eR: scalar general-purpose registers (r0-r31)
    Scalar,
    /// eP: pointer registers (p0-p7)
    Pointer,
    /// eM: post-modify registers (m0-m7)
    ModifierM,
    /// eDN: dimension size registers (dn0-dn7)
    ModifierDN,
    /// eDJ: dimension stride/jump registers (dj0-dj7)
    ModifierDJ,
    /// eDC: dimension count registers (dc0-dc7)
    ModifierDC,
    /// eW*, mWm: 256-bit vector registers
    Vector256,
    /// mXm, eX*: 512-bit vector registers
    Vector512,
    /// eAM, eBM, eCM, mAMm, mBMm: accumulator registers
    Accumulator,
}

/// Which Peano composite encoder function was used.
///
/// Each variant corresponds to a `get<Name>OpValue` function in
/// `AIE2MCCodeEmitterRegOperandDef.h`. The decoder inverts these
/// to recover the original register from the encoded bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeEncoder {
    /// mLdaScl, mSclSt, mSclMS: load/store scalar composite
    LdaScl,
    /// mMvSclSrc, mMvSclDst, mMvSclDstCg: move scalar composite
    MvSclSrc,
    /// mLdaCg, mLdbCg: load destination composite group
    LdaCg,
    /// mAluCg: ALU destination composite group
    AluCg,
    /// mMvAMWQDst: accumulator/weight-queue destination
    MvAMWQDst,
    /// mMvAMWQSrc: accumulator/weight-queue source
    MvAMWQSrc,
    /// mMvBMXSrc: buffer/cross source
    MvBMXSrc,
    /// mMvBMXDst: buffer/cross destination
    MvBMXDst,
    /// eRS4: scalar register subset (r16-r31)
    ERS4,
    /// mShflDst: shuffle destination
    ShflDst,
    /// mWm_1: vector register with bit rearrangement
    Wm1,
    /// mQXHLb: cross high/low byte
    QXHLb,
}

/// Classify an operand's decode type from the OperandDef.reg_class.
///
/// Maps the `reg_class` string (from the TableGen DAG pattern) to the
/// appropriate `OperandType`. Falls back to field-name heuristics when
/// `reg_class` is empty (e.g., for immediates not captured in DAG patterns).
pub fn classify_operand_type(reg_class: &str, field_name: &str) -> OperandType {
    // 1. Composite register operands (OP_* prefix from TableGen operand classes)
    match reg_class {
        "OP_mLdaScl" | "OP_mSclSt" | "OP_mSclMS" | "OP_mLdbScl"
            => return OperandType::CompositeRegister(CompositeEncoder::LdaScl),
        "OP_mMvSclSrc" | "OP_mMvSclDst" | "OP_mMvSclDstCg"
            => return OperandType::CompositeRegister(CompositeEncoder::MvSclSrc),
        "OP_mLdaCg" | "OP_mLdbCg"
            => return OperandType::CompositeRegister(CompositeEncoder::LdaCg),
        "OP_mAluCg"
            => return OperandType::CompositeRegister(CompositeEncoder::AluCg),
        "OP_mMvAMWQDst"
            => return OperandType::CompositeRegister(CompositeEncoder::MvAMWQDst),
        "OP_mMvAMWQSrc"
            => return OperandType::CompositeRegister(CompositeEncoder::MvAMWQSrc),
        "OP_mMvBMXSrc"
            => return OperandType::CompositeRegister(CompositeEncoder::MvBMXSrc),
        "OP_mMvBMXDst"
            => return OperandType::CompositeRegister(CompositeEncoder::MvBMXDst),
        "OP_eRS4"
            => return OperandType::CompositeRegister(CompositeEncoder::ERS4),
        "OP_mShflDst"
            => return OperandType::CompositeRegister(CompositeEncoder::ShflDst),
        "OP_mWm_1"
            => return OperandType::CompositeRegister(CompositeEncoder::Wm1),
        "OP_mQXHLb"
            => return OperandType::CompositeRegister(CompositeEncoder::QXHLb),
        _ => {}
    }

    // 2. Simple register classes (exact match or prefix)
    match reg_class {
        "eR" => return OperandType::Register(RegisterKind::Scalar),
        "eP" => return OperandType::Register(RegisterKind::Pointer),
        "eM" => return OperandType::Register(RegisterKind::ModifierM),
        "eDN" => return OperandType::Register(RegisterKind::ModifierDN),
        "eDJ" => return OperandType::Register(RegisterKind::ModifierDJ),
        "eDC" => return OperandType::Register(RegisterKind::ModifierDC),
        _ => {}
    }
    if reg_class.starts_with("eW") || reg_class == "mWm" {
        return OperandType::Register(RegisterKind::Vector256);
    }
    if reg_class.starts_with("mXm") || reg_class.starts_with("eX") {
        return OperandType::Register(RegisterKind::Vector512);
    }
    if reg_class.starts_with("eAM") || reg_class.starts_with("mAMm")
        || reg_class.starts_with("eBM") || reg_class.starts_with("mBMm")
        || reg_class.starts_with("eCM") || reg_class.starts_with("mBMS")
    {
        return OperandType::Register(RegisterKind::Accumulator);
    }
    // mQQm (weight queue) -- treat as accumulator for emulator purposes
    if reg_class.starts_with("mQQ") || reg_class.starts_with("mQX") {
        return OperandType::Register(RegisterKind::Accumulator);
    }

    // 3. Immediate operands (parsed from reg_class name)
    if let Some(imm_type) = parse_immediate_type(reg_class) {
        return imm_type;
    }

    // 4. Field-name fallback (when reg_class is empty or unrecognized)
    if reg_class.is_empty() || reg_class == "?" {
        return classify_from_field_name(field_name);
    }

    // Unrecognized reg_class -- warn and fallback
    OperandType::Unknown
}

/// Parse immediate type from reg_class name (e.g., "simm7", "imm12x4").
fn parse_immediate_type(reg_class: &str) -> Option<OperandType> {
    if reg_class.starts_with("simm") {
        // Signed immediate, check for scale suffix
        let rest = &reg_class[4..]; // after "simm"
        let scale = extract_scale_suffix(rest);
        return Some(OperandType::Immediate { signed: true, scale });
    }
    if reg_class.starts_with("imm") {
        let rest = &reg_class[3..]; // after "imm"
        let scale = extract_scale_suffix(rest);
        return Some(OperandType::Immediate { signed: false, scale });
    }
    if reg_class.starts_with("addr") {
        return Some(OperandType::Immediate { signed: false, scale: 1 });
    }
    // Target-specific formats like "t5u" (5-bit unsigned)
    if reg_class.starts_with('t') && reg_class.len() >= 3 {
        let last = reg_class.as_bytes()[reg_class.len() - 1];
        if last == b'u' || last == b's' {
            let signed = last == b's';
            return Some(OperandType::Immediate { signed, scale: 1 });
        }
    }
    None
}

/// Extract scale factor from immediate suffix (e.g., "12x4" -> 4, "7" -> 1).
fn extract_scale_suffix(s: &str) -> i32 {
    if let Some(pos) = s.find('x') {
        s[pos + 1..].parse::<i32>().unwrap_or(1)
    } else {
        1
    }
}

/// Classify operand type purely from the field name (last resort fallback).
fn classify_from_field_name(field_name: &str) -> OperandType {
    if field_name == "id" || field_name == "mLockId" {
        return OperandType::LockId;
    }
    // Constant fields: c5s, c11s (signed), c5u, c6u (unsigned)
    if field_name.len() >= 3 && field_name.starts_with('c') {
        let bytes = field_name.as_bytes();
        if bytes[1].is_ascii_digit() {
            let last = bytes[field_name.len() - 1];
            if last == b's' {
                return OperandType::Immediate { signed: true, scale: 1 };
            }
            if last == b'u' {
                return OperandType::Immediate { signed: false, scale: 1 };
            }
        }
    }
    OperandType::Unknown
}

/// A fragment of a split operand field.
///
/// AIE2 VLIW encodings sometimes scatter operand bits across non-contiguous
/// positions in the instruction word. For example, MOV_mv_cg encodes a 10-bit
/// immediate as `{i{9-1}, ..fixed.., i{0}, ..fixed..}`. Each non-contiguous
/// piece is a FieldFragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldFragment {
    /// Bit position in the instruction word where this fragment starts
    pub inst_bit: u8,
    /// Width of this fragment in bits
    pub width: u8,
    /// Starting bit position in the logical operand value.
    /// For `i{9-1}`, target_bit = 1 (maps to bits 1..=9 of the value).
    /// For `i{0}`, target_bit = 0 (maps to bit 0 of the value).
    pub target_bit: u8,
}

/// A resolved operand field within an instruction encoding.
///
/// Specifies where an operand can be extracted from the instruction bits,
/// and how the raw extracted value should be interpreted.
///
/// For contiguous fields, `fragments` is empty and extraction uses the simple
/// `(word >> bit_position) & mask` path. For split fields (like MOV_mv_cg's
/// immediate), `fragments` records each piece and extraction reassembles them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandField {
    /// Field name from TableGen (e.g., "mRx", "imm")
    pub name: String,
    /// Bit position of LSB within the slot encoding (for contiguous fields)
    pub bit_position: u8,
    /// Total logical width in bits (sum of all fragments)
    pub width: u8,
    /// Whether this is a signed immediate (for sign extension)
    pub signed: bool,
    /// Data-driven operand type (determines how raw bits become an Operand)
    pub operand_type: OperandType,
    /// Non-contiguous fragments. Empty for contiguous fields.
    pub fragments: Vec<FieldFragment>,
}

impl OperandField {
    /// Create a new operand field with Unknown operand type.
    pub fn new(name: impl Into<String>, bit_position: u8, width: u8) -> Self {
        Self {
            name: name.into(),
            bit_position,
            width,
            signed: false,
            operand_type: OperandType::Unknown,
            fragments: Vec::new(),
        }
    }

    /// Mark this field as signed.
    pub fn signed(mut self) -> Self {
        self.signed = true;
        self
    }

    /// Extract this field's value from an instruction word.
    ///
    /// For contiguous fields (the common case), uses a simple shift+mask.
    /// For split fields, reassembles the value from scattered fragments.
    #[inline]
    pub fn extract(&self, word: u64) -> u64 {
        if self.fragments.is_empty() {
            // Contiguous field: simple extraction
            let mask = (1u64 << self.width) - 1;
            (word >> self.bit_position) & mask
        } else {
            // Split field: reassemble from fragments
            let mut value = 0u64;
            for frag in &self.fragments {
                let frag_mask = (1u64 << frag.width) - 1;
                let bits = (word >> frag.inst_bit) & frag_mask;
                value |= bits << frag.target_bit;
            }
            value
        }
    }

    /// Extract as signed value (sign-extend if needed).
    #[inline]
    pub fn extract_signed(&self, word: u64) -> i64 {
        let unsigned = self.extract(word);
        if self.signed && self.width < 64 {
            // Sign extend based on total logical width
            let sign_bit = 1u64 << (self.width - 1);
            if unsigned & sign_bit != 0 {
                let mask = !((1u64 << self.width) - 1);
                (unsigned | mask) as i64
            } else {
                unsigned as i64
            }
        } else {
            unsigned as i64
        }
    }
}

/// A fully resolved instruction encoding.
///
/// Contains all information needed to decode and identify an instruction.
#[derive(Debug, Clone)]
pub struct InstrEncoding {
    /// Instruction name (e.g., "ADD", "LDA_ri")
    pub name: String,

    /// Assembly mnemonic (e.g., "add", "lda")
    pub mnemonic: String,

    /// Slot this instruction belongs to (e.g., "alu", "lda")
    pub slot: String,

    /// Total bit width of the encoding
    pub width: u8,

    /// Mask of fixed bits (1 = this bit is part of the opcode)
    pub fixed_mask: u64,

    /// Expected values for fixed bits
    pub fixed_bits: u64,

    /// Operand fields in MSB-first order
    pub operand_fields: Vec<OperandField>,

    /// Semantic operation, if known from patterns
    pub semantic: Option<SemanticOp>,

    /// Whether this instruction may load from memory
    pub may_load: bool,

    /// Whether this instruction may store to memory
    pub may_store: bool,

    /// Input operand order from TableGen InstrDef.inputs.
    ///
    /// This is the canonical order for source operands. When building SlotOp,
    /// sources should be ordered according to this list, not field extraction order.
    /// Empty if InstrDef was not available.
    pub input_order: Vec<String>,

    /// Output operand order from TableGen InstrDef.outputs.
    ///
    /// For instructions with multiple outputs (rare), this defines their order.
    pub output_order: Vec<String>,

    /// Implicit register uses/defs from TableGen.
    ///
    /// For example, `sel.eqz` reads r27 implicitly (via `eR27:$s2` in TableGen).
    /// These registers are not encoded in instruction bits - they're fixed.
    pub implicit_regs: Vec<ImplicitReg>,

    /// Addressing mode detected from instruction name (e.g., `_pstm_nrm_imm`).
    /// Used by the decoder to correctly extract post-modify vs indexed operands.
    pub addressing_mode: AddressingMode,

    /// Memory access width detected from mnemonic (e.g., `.s8` -> Byte).
    /// Used by the decoder to set the correct MemWidth on Load/Store operations.
    pub mem_width: InstrMemWidth,

    /// Whether the encoding uniquely identifies this instruction.
    /// When false (from TableGen's `hasCompleteDecoder = 0`), the encoding
    /// may be ambiguous and this instruction should be deprioritized during
    /// disambiguation. Complete-decoder instructions are preferred when
    /// multiple encodings match the same bit pattern.
    pub has_complete_decoder: bool,

    // ── Pre-resolved metadata (populated once at TableGen load time) ──

    /// Element type inferred from the mnemonic (e.g., "vadd_8" -> Int8).
    /// None for instructions that don't have an element type suffix.
    pub element_type: Option<ElementType>,

    /// Branch condition inferred from the mnemonic (e.g., "jnz" -> NotZero).
    /// Only set for BrCond instructions.
    pub branch_condition: Option<BranchCondition>,

    /// Whether this is a vector instruction (mnemonic starts with 'v').
    pub is_vector: bool,

    /// Select variant inferred from the mnemonic (e.g., "sel.eqz" -> EqualZero).
    /// Only set for Select instructions.
    pub select_variant: Option<SelectVariant>,

    /// Whether this is a pointer arithmetic instruction (mnemonic starts with "padd").
    /// When true, address generator fields produce a destination pointer + source
    /// operand instead of a Memory operand.
    pub is_ptr_arithmetic: bool,

    /// Itinerary class from TableGen (e.g., "II_ADD", "II_LDA", "II_VMUL").
    /// Used for cross-validating latency values against the compiler's scheduling model.
    pub sched_class: Option<String>,
}

impl InstrEncoding {
    /// Check if an instruction word matches this encoding.
    #[inline]
    pub fn matches(&self, word: u64) -> bool {
        (word & self.fixed_mask) == self.fixed_bits
    }

    /// Get the number of set bits in the fixed mask (for matching priority).
    ///
    /// Higher specificity = more fixed bits = should match first.
    pub fn specificity(&self) -> u32 {
        self.fixed_mask.count_ones()
    }

    /// Sort key for disambiguation: (has_complete_decoder, fixed_ones, specificity).
    ///
    /// The sort order (all descending) is:
    /// 1. Complete-decoder instructions first (unambiguous encodings).
    /// 2. Among incomplete-decoder instructions, prefer those with more fixed
    ///    1-bits in their opcode. A fixed 1-bit is a positive assertion ("this
    ///    bit must be 1"), which is more distinctive than a fixed 0-bit. An
    ///    encoding with all-zero fixed bits matches a huge swath of encoding
    ///    space and is likely a false match when competing with an encoding
    ///    that has specific 1-bits.
    /// 3. Break ties with total fixed bit count (specificity).
    pub fn sort_key(&self) -> (bool, u32, u32) {
        (
            self.has_complete_decoder,
            self.fixed_bits.count_ones(),
            self.specificity(),
        )
    }
}

/// Errors that can occur during resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// Format class not found
    FormatNotFound(String),
    /// Slot not found
    SlotNotFound(String),
    /// Template parameter count mismatch
    TemplateArgsMismatch {
        expected: usize,
        got: usize,
    },
    /// Unknown field in encoding
    UnknownField(String),
    /// Encoding width mismatch
    WidthMismatch {
        expected: u8,
        computed: u8,
    },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FormatNotFound(name) => write!(f, "Format class not found: {}", name),
            Self::SlotNotFound(name) => write!(f, "Slot not found: {}", name),
            Self::TemplateArgsMismatch { expected, got } => {
                write!(f, "Template args mismatch: expected {}, got {}", expected, got)
            }
            Self::UnknownField(name) => write!(f, "Unknown field: {}", name),
            Self::WidthMismatch { expected, computed } => {
                write!(f, "Width mismatch: expected {}, computed {}", expected, computed)
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// Resolver for computing instruction encodings.
pub struct Resolver<'a> {
    data: &'a TableGenData,
}

impl<'a> Resolver<'a> {
    /// Create a new resolver from TableGen data.
    pub fn new(data: &'a TableGenData) -> Self {
        Self { data }
    }

    /// Merge field_sources from format class and all mixin classes.
    ///
    /// Mixin classes provide field mappings like `let mLockId = {id, 0b0}` that
    /// map encoding fields to DAG operand names. This function collects all
    /// such mappings from:
    /// 1. The format class itself
    /// 2. Each mixin class used by the instruction
    /// 3. Parent classes of the mixin classes (inheritance chain)
    fn merge_field_sources(
        &self,
        format: &FormatClass,
        instr: &InstrDef,
    ) -> HashMap<String, Vec<String>> {
        let mut merged = format.field_sources.clone();

        // Helper to collect field_sources from a mixin and its parent chain
        fn collect_mixin_sources(
            mixins: &HashMap<String, MixinClass>,
            mixin_name: &str,
            merged: &mut HashMap<String, Vec<String>>,
        ) {
            if let Some(mixin) = mixins.get(mixin_name) {
                // First collect from parent if present (parent sources are lower priority)
                if let Some(ref parent_name) = mixin.parent {
                    collect_mixin_sources(mixins, parent_name, merged);
                }

                // Then collect from this mixin (overrides parent)
                for (field, sources) in &mixin.field_sources {
                    merged.insert(field.clone(), sources.clone());
                }
            }
        }

        // Collect from each mixin class used by this instruction
        for mixin_name in &instr.mixin_classes {
            collect_mixin_sources(&self.data.mixins, mixin_name, &mut merged);
        }

        merged
    }

    /// Resolve a single instruction definition to its encoding.
    pub fn resolve_instruction(&self, instr: &InstrDef) -> Result<InstrEncoding, ResolveError> {
        // Find the format class
        let format = self.data.formats.get(&instr.format)
            .ok_or_else(|| ResolveError::FormatNotFound(instr.format.clone()))?;

        // Find the slot
        let slot = self.find_slot(format)?;

        // Check template argument count
        if format.template_params.len() != instr.template_args.len() {
            return Err(ResolveError::TemplateArgsMismatch {
                expected: format.template_params.len(),
                got: instr.template_args.len(),
            });
        }

        // Build template substitution map
        let mut template_values: HashMap<String, u64> = HashMap::new();
        for (param, value) in format.template_params.iter().zip(&instr.template_args) {
            template_values.insert(param.name.clone(), *value);
        }

        // Combine format fields with template params for width lookup
        let mut field_widths: HashMap<String, u8> = format.fields.clone();
        for param in &format.template_params {
            field_widths.insert(param.name.clone(), param.bits);
        }

        // Build merged field_sources from format class and all mixin classes
        // This is critical for mapping encoding fields (e.g., mLockId) to DAG operands (e.g., id)
        let merged_field_sources = self.merge_field_sources(format, instr);

        // Process encoding parts to compute masks and fields
        // Pass merged field_sources to trace derived fields back to source operands
        let (fixed_mask, fixed_bits, mut operand_fields) =
            self.process_encoding(
                &format.encoding,
                &field_widths,
                &template_values,
                &merged_field_sources,
            )?;

        // Populate operand_type on each field using OperandDef.reg_class
        let all_operand_defs: Vec<&super::types::OperandDef> = instr.outputs.iter()
            .chain(instr.inputs.iter())
            .collect();

        for field in &mut operand_fields {
            if let Some(opdef) = all_operand_defs.iter()
                .find(|od| od.name == field.name)
            {
                field.operand_type = classify_operand_type(&opdef.reg_class, &field.name);
            } else {
                // No matching OperandDef -- use field-name fallback
                field.operand_type = classify_operand_type("", &field.name);
            }
            // Sync signed flag from operand type
            if let OperandType::Immediate { signed: true, .. } = &field.operand_type {
                field.signed = true;
            }
        }

        // Three-tier semantic inference: pattern -> structural -> mnemonic.
        let defs_vec: Vec<String> = instr.attributes.defs.iter().cloned().collect();
        let uses_vec: Vec<String> = instr.attributes.uses.iter().cloned().collect();
        let semantic = self.data.semantic_for_instruction(&instr.name)
            .map(|p| p.operation)
            .or_else(|| infer_semantic_from_structure(
                &defs_vec, &uses_vec,
                instr.attributes.may_load, instr.attributes.may_store,
                false, // regex parser doesn't extract hasDelaySlot
                &[],   // regex parser doesn't have parent class chain
            ))
            .or_else(|| infer_semantic_from_mnemonic(&instr.mnemonic));

        // Extract operand ordering from InstrDef
        let input_order: Vec<String> = instr.inputs.iter().map(|o| o.name.clone()).collect();
        let output_order: Vec<String> = instr.outputs.iter().map(|o| o.name.clone()).collect();

        // Pre-resolve metadata from mnemonic (once per encoding, not per decode)
        let is_vector = instr.mnemonic.starts_with('v') || instr.mnemonic.starts_with('V');
        let is_ptr_arithmetic = instr.mnemonic.starts_with("padd");
        let element_type = infer_element_type(&instr.mnemonic);
        let branch_condition = infer_branch_condition(&instr.mnemonic, semantic);
        let select_variant = infer_select_variant(&instr.mnemonic, semantic);

        Ok(InstrEncoding {
            name: instr.name.clone(),
            mnemonic: instr.mnemonic.clone(),
            slot: slot.field.clone(),
            width: slot.bits,
            fixed_mask,
            fixed_bits,
            operand_fields,
            semantic,
            may_load: instr.attributes.may_load,
            may_store: instr.attributes.may_store,
            input_order,
            output_order,
            implicit_regs: instr.implicit_regs.clone(),
            addressing_mode: detect_addressing_mode(&instr.name),
            mem_width: detect_mem_width(&instr.mnemonic),
            // Regex parser doesn't parse hasCompleteDecoder; assume true (safe default)
            has_complete_decoder: true,
            element_type,
            branch_condition,
            is_vector,
            select_variant,
            is_ptr_arithmetic,
            sched_class: None, // Regex parser doesn't extract itinerary class
        })
    }

    /// Resolve all instructions in the TableGen data.
    pub fn resolve_all(&self) -> Vec<Result<InstrEncoding, ResolveError>> {
        self.data.instructions.values()
            .map(|instr| self.resolve_instruction(instr))
            .collect()
    }

    /// Resolve all instructions, filtering out errors.
    pub fn resolve_all_ok(&self) -> Vec<InstrEncoding> {
        self.resolve_all()
            .into_iter()
            .filter_map(|r| r.ok())
            .collect()
    }

    /// Resolve all instructions and group by slot.
    pub fn resolve_by_slot(&self) -> HashMap<String, Vec<InstrEncoding>> {
        let mut by_slot: HashMap<String, Vec<InstrEncoding>> = HashMap::new();

        for encoding in self.resolve_all_ok() {
            by_slot.entry(encoding.slot.clone())
                .or_default()
                .push(encoding);
        }

        // Sort each slot by disambiguation key
        for encodings in by_slot.values_mut() {
            encodings.sort_by(|a, b| b.sort_key().cmp(&a.sort_key()));
        }

        by_slot
    }

    /// Find the slot for a format class.
    fn find_slot(&self, format: &FormatClass) -> Result<&SlotDef, ResolveError> {
        // First try explicit slot_field
        if let Some(ref field) = format.slot_field {
            if let Some(slot) = self.data.slots.values().find(|s| &s.field == field) {
                return Ok(slot);
            }
        }

        // Try to infer from parent class name
        if let Some(field) = format.slot_from_parent() {
            if let Some(slot) = self.data.slots.values().find(|s| s.field == field) {
                return Ok(slot);
            }
        }

        Err(ResolveError::SlotNotFound(format.name.clone()))
    }

    /// Process encoding parts to compute fixed mask, fixed bits, and operand fields.
    ///
    /// Encoding parts are in MSB-first order, so we process from highest bit down.
    ///
    /// The `field_sources` map traces derived fields back to their source operands.
    /// For example, if `let mLockId = {id, 0b0}` was parsed, then
    /// `field_sources["mLockId"] = ["id"]`. When we encounter `mLockId` in the encoding,
    /// we create an OperandField named `id` (the source) instead of `mLockId` (the derived).
    fn process_encoding(
        &self,
        parts: &[EncodingPart],
        field_widths: &HashMap<String, u8>,
        template_values: &HashMap<String, u64>,
        field_sources: &HashMap<String, Vec<String>>,
    ) -> Result<(u64, u64, Vec<OperandField>), ResolveError> {
        // First, compute total width
        let mut total_width: u8 = 0;
        for part in parts {
            let width = part.width(field_widths)
                .ok_or_else(|| ResolveError::UnknownField(format!("{:?}", part)))?;
            total_width = total_width.saturating_add(width);
        }

        let mut fixed_mask: u64 = 0;
        let mut fixed_bits: u64 = 0;
        let mut operand_fields: Vec<OperandField> = Vec::new();

        // Process parts MSB-first, tracking current bit position
        let mut bit_pos = total_width;

        for part in parts {
            let width = part.width(field_widths).unwrap();
            bit_pos = bit_pos.saturating_sub(width);

            match part {
                EncodingPart::Literal { value, width } => {
                    // Fixed bits - add to mask and expected value
                    let mask = ((1u64 << width) - 1) << bit_pos;
                    fixed_mask |= mask;
                    fixed_bits |= (value << bit_pos) & mask;
                }

                EncodingPart::FieldRef { name, high, low } => {
                    // Check if this is a template parameter (fixed) or operand field (variable)
                    if let Some(value) = template_values.get(name) {
                        // Template parameter - treat as fixed bits
                        let w = if let (Some(h), Some(l)) = (high, low) {
                            h - l + 1
                        } else {
                            field_widths.get(name).copied().unwrap_or(0)
                        };
                        let mask = ((1u64 << w) - 1) << bit_pos;
                        fixed_mask |= mask;

                        // Extract the relevant bits if sliced
                        let extracted = if let (Some(_h), Some(l)) = (high, low) {
                            (value >> l) & ((1u64 << w) - 1)
                        } else {
                            *value
                        };
                        fixed_bits |= (extracted << bit_pos) & mask;
                    } else {
                        // Operand field - record position for extraction
                        let w = if let (Some(h), Some(l)) = (high, low) {
                            h - l + 1
                        } else {
                            field_widths.get(name).copied().unwrap_or(0)
                        };

                        // Trace derived fields to their source operands
                        // If `let mLockId = {id, 0b0}` was parsed, use "id" instead of "mLockId"
                        let operand_name = if let Some(sources) = field_sources.get(name.as_str()) {
                            // Use the first source operand name (there's usually just one)
                            sources.first().map(|s| s.as_str()).unwrap_or(name.as_str())
                        } else {
                            name.as_str()
                        };

                        // Determine which bits of the logical operand this fragment covers.
                        // For `i{9-1}`, target_bit = 1 (covers bits 1..=9 of the value).
                        // For `i` (no slice), target_bit = 0 (covers all bits).
                        let target_bit = low.unwrap_or(0);

                        // Check if we should merge with existing field of same name
                        // (for split fields like i{9-1} ... i{0})
                        if let Some(existing) = operand_fields.iter_mut().find(|f| f.name == operand_name) {
                            // Split field: convert to fragment-based extraction.
                            // On the first merge, we retroactively record the original
                            // contiguous piece as the first fragment.
                            if existing.fragments.is_empty() {
                                // First merge: convert the existing contiguous field
                                // into a fragment. The existing field was the MSB piece
                                // (parts are processed MSB-first), so its target_bit
                                // is the width of remaining pieces.
                                //
                                // We need to figure out the target_bit for the first
                                // fragment. The logical field width is in field_widths.
                                // First fragment covers the MSB, so target_bit =
                                // total_logical_width - existing.width.
                                let logical_width = field_widths.get(name)
                                    .copied()
                                    .unwrap_or(existing.width + w);
                                let first_target = logical_width.saturating_sub(existing.width);
                                existing.fragments.push(FieldFragment {
                                    inst_bit: existing.bit_position,
                                    width: existing.width,
                                    target_bit: first_target,
                                });
                            }
                            // Add the new fragment
                            existing.fragments.push(FieldFragment {
                                inst_bit: bit_pos,
                                width: w,
                                target_bit: target_bit,
                            });
                            // Update total logical width
                            existing.width = existing.width.saturating_add(w);
                        } else {
                            operand_fields.push(OperandField::new(operand_name, bit_pos, w));
                        }
                    }
                }

                EncodingPart::DontCare { .. } => {
                    // Don't care bits - not part of fixed mask, not an operand
                    // They're effectively wildcards during matching
                }
            }
        }

        Ok((fixed_mask, fixed_bits, operand_fields))
    }
}

/// Build a decoder table from resolved encodings.
///
/// Returns encodings grouped by slot, sorted by specificity (most specific first).
/// Includes synthetic encodings for LNG control flow instructions (JL, J) that
/// are not represented in TableGen.
pub fn build_decoder_tables(data: &TableGenData) -> HashMap<String, Vec<InstrEncoding>> {
    let mut by_slot = Resolver::new(data).resolve_by_slot();
    add_lng_control_flow_encodings(&mut by_slot);
    by_slot
}

/// Add synthetic InstrEncoding entries for LNG slot control flow instructions.
///
/// JL (jump-and-link) and J (unconditional jump) use the LNG slot but aren't
/// defined in the standard TableGen instruction definitions. Their bit layout
/// (from AIE2InstrFormats.td comments):
///
/// ```text
/// JL: lng = {00000, cpmaddr[19:0], 0, 0000, 0000, 00000, 100}
/// J:  lng = {00000, cpmaddr[19:0], 0, 0000, 0000, 00000, 010}
/// ```
///
/// 42-bit LNG slot layout (MSB to LSB):
/// - bits[41:37] = 00000 (reserved)
/// - bits[36:17] = cpmaddr[19:0]
/// - bits[16:3]  = 0 (reserved fields)
/// - bits[2:0]   = opcode (100 = JL, 010 = J)
///
/// For the opcode, movxm uses 001, so no collision with JL (100) or J (010).
fn add_lng_control_flow_encodings(by_slot: &mut HashMap<String, Vec<InstrEncoding>>) {
    let cpmaddr_field = OperandField {
        name: "cpmaddr".into(),
        bit_position: 17,
        width: 20,
        signed: false,
        operand_type: OperandType::Immediate { signed: false, scale: 1 },
        fragments: vec![],
    };

    // Common fixed bits: bits[41:37] = 0, bits[16:3] = 0
    // These are the "reserved zero" fields. We mask them for specificity.
    let reserved_mask: u64 =
        (0x1Fu64 << 37) |   // bits 41:37
        (0x3FFFu64 << 3);   // bits 16:3
    let opcode_mask: u64 = 0x7; // bits 2:0

    // JL: opcode = 0b100
    let jl_encoding = InstrEncoding {
        name: "JL_synthetic".into(),
        mnemonic: "jl".into(),
        slot: "lng".into(),
        width: 42,
        fixed_mask: reserved_mask | opcode_mask,
        fixed_bits: 0b100, // Only the opcode is non-zero
        operand_fields: vec![cpmaddr_field.clone()],
        semantic: Some(SemanticOp::Call),
        may_load: false,
        may_store: false,
        input_order: vec!["cpmaddr".into()],
        output_order: vec![],
        implicit_regs: vec![],
        addressing_mode: AddressingMode::Unknown,
        mem_width: InstrMemWidth::Word,
        has_complete_decoder: true,
        element_type: None,
        branch_condition: None,
        is_vector: false,
        select_variant: None,
        is_ptr_arithmetic: false,
        sched_class: Some("II_JL".into()),
    };

    // J: opcode = 0b010
    let j_encoding = InstrEncoding {
        name: "J_synthetic".into(),
        mnemonic: "j".into(),
        slot: "lng".into(),
        width: 42,
        fixed_mask: reserved_mask | opcode_mask,
        fixed_bits: 0b010, // Only the opcode is non-zero
        operand_fields: vec![cpmaddr_field],
        semantic: Some(SemanticOp::Br),
        may_load: false,
        may_store: false,
        input_order: vec!["cpmaddr".into()],
        output_order: vec![],
        implicit_regs: vec![],
        addressing_mode: AddressingMode::Unknown,
        mem_width: InstrMemWidth::Word,
        has_complete_decoder: true,
        element_type: None,
        branch_condition: Some(BranchCondition::Always),
        is_vector: false,
        select_variant: None,
        is_ptr_arithmetic: false,
        sched_class: Some("II_J".into()),
    };

    by_slot.entry("lng".into()).or_default().push(jl_encoding);
    by_slot.entry("lng".into()).or_default().push(j_encoding);
}

/// O(1) instruction lookup index for a single slot.
///
/// Uses a HashMap keyed on the common opcode bits to achieve constant-time
/// lookup for most instructions, with a small linear scan for disambiguation.
#[derive(Debug, Clone)]
pub struct SlotIndex {
    /// The slot name (e.g., "alu", "lda")
    pub slot_name: String,

    /// Intersection of all fixed_masks - the "common opcode bits".
    /// All instructions in this slot have these bits as part of their opcode.
    pub common_mask: u64,

    /// Primary lookup table: (word & common_mask) -> candidate encodings.
    /// Usually 1 entry per key, occasionally 2-3 for disambiguation.
    by_opcode: HashMap<u64, Vec<InstrEncoding>>,

    /// Fallback encodings for instructions with unusual masks.
    /// These have fixed_mask bits outside the common_mask (rare).
    fallback: Vec<InstrEncoding>,
}

impl SlotIndex {
    /// Build a slot index from a list of encodings.
    ///
    /// Computes the common opcode mask and organizes encodings for O(1) lookup.
    pub fn build(slot_name: impl Into<String>, mut encodings: Vec<InstrEncoding>) -> Self {
        let slot_name = slot_name.into();

        if encodings.is_empty() {
            return Self {
                slot_name,
                common_mask: 0,
                by_opcode: HashMap::new(),
                fallback: Vec::new(),
            };
        }

        // Sort by disambiguation key: complete-decoder first, then fixed 1-bits,
        // then total specificity. See InstrEncoding::sort_key() for rationale.
        encodings.sort_by(|a, b| b.sort_key().cmp(&a.sort_key()));

        // Compute common mask: intersection of all fixed_masks
        let common_mask = encodings
            .iter()
            .map(|e| e.fixed_mask)
            .fold(!0u64, |acc, mask| acc & mask);

        // Partition encodings into by_opcode vs fallback
        let mut by_opcode: HashMap<u64, Vec<InstrEncoding>> = HashMap::new();
        let mut fallback = Vec::new();

        for encoding in encodings {
            // Check if this encoding's mask is exactly the common mask
            // or a superset (has additional fixed bits beyond common)
            if (encoding.fixed_mask & common_mask) == common_mask {
                // Normal case: index by the common opcode bits
                let opcode = encoding.fixed_bits & common_mask;
                by_opcode.entry(opcode).or_default().push(encoding);
            } else {
                // Unusual: this encoding has a smaller mask than common
                // (shouldn't happen with proper TableGen, but be safe)
                fallback.push(encoding);
            }
        }

        // Sort each bucket by disambiguation key
        for bucket in by_opcode.values_mut() {
            bucket.sort_by(|a, b| b.sort_key().cmp(&a.sort_key()));
        }
        fallback.sort_by(|a, b| b.sort_key().cmp(&a.sort_key()));

        Self {
            slot_name,
            common_mask,
            by_opcode,
            fallback,
        }
    }

    /// Decode a word using O(1) lookup.
    ///
    /// Returns the matching encoding and extracted operand values.
    #[inline]
    pub fn decode(&self, word: u64) -> Option<(&InstrEncoding, HashMap<String, u64>)> {
        // Primary lookup: use common opcode bits
        let opcode = word & self.common_mask;

        if let Some(candidates) = self.by_opcode.get(&opcode) {
            // Usually 1 candidate, occasionally 2-3
            for encoding in candidates {
                if encoding.matches(word) {
                    let operands = self.extract_operands(encoding, word);
                    return Some((encoding, operands));
                }
            }
        }

        // Fallback: linear scan of unusual encodings
        for encoding in &self.fallback {
            if encoding.matches(word) {
                let operands = self.extract_operands(encoding, word);
                return Some((encoding, operands));
            }
        }

        None
    }

    /// Extract operand values from a matched instruction.
    #[inline]
    fn extract_operands(&self, encoding: &InstrEncoding, word: u64) -> HashMap<String, u64> {
        let mut operands = HashMap::new();
        for field in &encoding.operand_fields {
            let value = field.extract(word);
            operands.insert(field.name.clone(), value);
        }
        operands
    }

    /// Get the number of indexed encodings.
    pub fn encoding_count(&self) -> usize {
        self.by_opcode.values().map(|v| v.len()).sum::<usize>() + self.fallback.len()
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> SlotIndexStats {
        let bucket_sizes: Vec<usize> = self.by_opcode.values().map(|v| v.len()).collect();
        let max_bucket = bucket_sizes.iter().copied().max().unwrap_or(0);
        let avg_bucket = if bucket_sizes.is_empty() {
            0.0
        } else {
            bucket_sizes.iter().sum::<usize>() as f64 / bucket_sizes.len() as f64
        };

        SlotIndexStats {
            slot_name: self.slot_name.clone(),
            common_mask: self.common_mask,
            bucket_count: self.by_opcode.len(),
            fallback_count: self.fallback.len(),
            max_bucket_size: max_bucket,
            avg_bucket_size: avg_bucket,
        }
    }
}

/// Statistics about a SlotIndex.
#[derive(Debug, Clone)]
pub struct SlotIndexStats {
    pub slot_name: String,
    pub common_mask: u64,
    pub bucket_count: usize,
    pub fallback_count: usize,
    pub max_bucket_size: usize,
    pub avg_bucket_size: f64,
}

/// Complete decoder index for all slots.
///
/// Provides O(1) instruction decoding by slot type.
#[derive(Debug, Clone, Default)]
pub struct DecoderIndex {
    /// Per-slot indices
    slots: HashMap<String, SlotIndex>,
}

impl DecoderIndex {
    /// Build a decoder index from TableGen data.
    pub fn build(data: &TableGenData) -> Self {
        let by_slot = Resolver::new(data).resolve_by_slot();
        Self::from_slot_encodings(by_slot)
    }

    /// Build from pre-resolved slot encodings.
    pub fn from_slot_encodings(by_slot: HashMap<String, Vec<InstrEncoding>>) -> Self {
        let slots = by_slot
            .into_iter()
            .map(|(name, encodings)| {
                let index = SlotIndex::build(&name, encodings);
                (name, index)
            })
            .collect();

        Self { slots }
    }

    /// Decode slot bits for a specific slot type.
    #[inline]
    pub fn decode_slot(&self, slot_name: &str, bits: u64) -> Option<(&InstrEncoding, HashMap<String, u64>)> {
        self.slots.get(slot_name).and_then(|idx| idx.decode(bits))
    }

    /// Get the index for a specific slot.
    pub fn slot_index(&self, slot_name: &str) -> Option<&SlotIndex> {
        self.slots.get(slot_name)
    }

    /// Get all slot names.
    pub fn slot_names(&self) -> impl Iterator<Item = &str> {
        self.slots.keys().map(|s| s.as_str())
    }

    /// Get statistics for all slots.
    pub fn stats(&self) -> Vec<SlotIndexStats> {
        self.slots.values().map(|idx| idx.stats()).collect()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Get total encoding count across all slots.
    pub fn total_encodings(&self) -> usize {
        self.slots.values().map(|idx| idx.encoding_count()).sum()
    }
}

/// Infer semantic operation from structured TableGen data.
///
/// Uses register defs/uses, memory flags, delay slot, and parent class names
/// to classify instructions without parsing the mnemonic string. This is the
/// preferred inference path -- it relies on compiler-verified attributes from
/// llvm-aie rather than hand-maintained string matching tables.
///
/// Returns None for instructions that can't be classified structurally
/// (arithmetic, comparison, bitwise -- these need the mnemonic fallback).
pub fn infer_semantic_from_structure(
    defs: &[String],
    uses: &[String],
    may_load: bool,
    may_store: bool,
    has_delay_slot: bool,
    parents: &[String],
) -> Option<SemanticOp> {
    // Control flow: use Defs/Uses of lr and hasDelaySlot.
    // On AIE2 hardware encodings, isBranch/isCall/isReturn are only set on
    // Pseudo instructions, but these structural signals are reliable:
    //   JL/JL_IND: Defs=[lr], hasDelaySlot=1    -> Call
    //   RET:       Uses=[lr], hasDelaySlot=1     -> Ret
    //   J_jump_*:  hasDelaySlot=1, no lr         -> Br
    //   JNZD:      hasDelaySlot=1, no lr         -> BrCond (needs mnemonic for condition)
    let defs_lr = defs.iter().any(|r| r == "lr");
    let uses_lr = uses.iter().any(|r| r == "lr");

    if has_delay_slot {
        if defs_lr {
            return Some(SemanticOp::Call);
        }
        if uses_lr {
            return Some(SemanticOp::Ret);
        }
        // Other delay-slot instructions are branches; the mnemonic fallback
        // will disambiguate unconditional vs conditional + condition code.
        // We don't return here because we can't tell j vs jnz vs jz structurally.
    }

    // Memory operations: mayLoad/mayStore are authoritative flags from TableGen.
    // The slot (lda/ldb/st) and mnemonic (vlda/vst) refine the operation type,
    // but Load vs Store classification comes from these flags.
    if may_load && !may_store {
        return Some(SemanticOp::Load);
    }
    if may_store && !may_load {
        return Some(SemanticOp::Store);
    }

    // Parent class chain: format class names encode operation type.
    // These are compiler-internal names that follow a strict naming convention.
    for parent in parents {
        // Lock operations: AIE2_mLockId in parent chain
        if parent.contains("mLockId") || parent.contains("LockId") {
            // Can't distinguish acquire vs release from class alone;
            // leave for mnemonic fallback.
        }
        // Done (halt): AIE2_done_inst_alu
        if parent.contains("_done_") {
            return Some(SemanticOp::Done);
        }
    }

    None
}

/// Infer semantic operation from instruction mnemonic.
///
/// This is the final fallback for instructions not classified by patterns
/// or structural attributes. It handles arithmetic, comparison, bitwise,
/// and other operations where no structured TableGen flag exists.
///
/// Used by both the resolver and tblgen_records modules.
pub fn infer_semantic_from_mnemonic(mnemonic: &str) -> Option<SemanticOp> {
    let lower = mnemonic.to_lowercase();

    // Select operations
    if lower.starts_with("sel") {
        return Some(SemanticOp::Select);
    }

    // Arithmetic operations
    if lower == "add" || lower.starts_with("add.") {
        return Some(SemanticOp::Add);
    }
    if lower == "sub" || lower.starts_with("sub.") {
        return Some(SemanticOp::Sub);
    }
    if lower == "mul" || lower.starts_with("mul.") {
        return Some(SemanticOp::Mul);
    }
    if lower == "abs" || lower.starts_with("abs.") {
        return Some(SemanticOp::Abs);
    }
    if lower == "neg" || lower.starts_with("neg.") {
        return Some(SemanticOp::Neg);
    }

    // Comparison operations
    if lower == "lt" || lower.starts_with("lt.") {
        return Some(SemanticOp::SetLt);
    }
    if lower == "le" || lower.starts_with("le.") {
        return Some(SemanticOp::SetLe);
    }
    if lower == "gt" || lower.starts_with("gt.") {
        return Some(SemanticOp::SetGt);
    }
    if lower == "ge" || lower.starts_with("ge.") {
        return Some(SemanticOp::SetGe);
    }
    if lower == "eq" || lower.starts_with("eq.") {
        return Some(SemanticOp::SetEq);
    }
    if lower == "ne" || lower.starts_with("ne.") {
        return Some(SemanticOp::SetNe);
    }

    // Bitwise operations
    if lower == "and" || lower.starts_with("and.") {
        return Some(SemanticOp::And);
    }
    if lower == "or" || lower.starts_with("or.") {
        return Some(SemanticOp::Or);
    }
    if lower == "xor" || lower.starts_with("xor.") {
        return Some(SemanticOp::Xor);
    }
    if lower == "not" || lower.starts_with("not.") {
        return Some(SemanticOp::Not);
    }

    // Shift operations
    if lower.starts_with("lshl") || lower == "shl" {
        return Some(SemanticOp::Shl);
    }
    if lower.starts_with("ashr") || lower == "asr" {
        return Some(SemanticOp::Sra);
    }
    if lower.starts_with("lshr") || lower == "lsr" {
        return Some(SemanticOp::Srl);
    }

    // Move/copy (mov, mova, movx, movxm, mov.* variants)
    if lower.starts_with("mov") {
        return Some(SemanticOp::Copy);
    }

    // Memory operations
    // Vector load/store must come before scalar to avoid vlda matching "lda"
    if lower.starts_with("vlda") || lower.starts_with("vldb") {
        return Some(SemanticOp::Load);
    }
    if lower.starts_with("vst") {
        return Some(SemanticOp::Store);
    }
    // Scalar load B channel
    if lower.starts_with("ldb") {
        return Some(SemanticOp::Load);
    }
    if lower.starts_with("lda") || lower.starts_with("ld.") {
        return Some(SemanticOp::Load);
    }
    if lower.starts_with("st.") || lower == "st" || lower.starts_with("st ") {
        return Some(SemanticOp::Store);
    }

    // Nop
    if lower == "nop" || lower.starts_with("nop") {
        return Some(SemanticOp::Nop);
    }

    // Done (core termination/halt)
    if lower == "done" {
        return Some(SemanticOp::Done);
    }

    // Unsigned comparisons (ltu, leu, gtu, geu)
    if lower == "ltu" || lower.starts_with("ltu.") {
        return Some(SemanticOp::SetUlt);
    }
    if lower == "leu" || lower.starts_with("leu.") {
        return Some(SemanticOp::SetUle);
    }
    if lower == "gtu" || lower.starts_with("gtu.") {
        return Some(SemanticOp::SetUgt);
    }
    if lower == "geu" || lower.starts_with("geu.") {
        return Some(SemanticOp::SetUge);
    }

    // Pointer add (treat as add)
    if lower.starts_with("padd") {
        return Some(SemanticOp::Add);
    }

    // Call (jump-and-link) operations
    if lower == "jl" || lower.starts_with("call") {
        return Some(SemanticOp::Call);
    }
    // Unconditional jumps
    if lower == "j" {
        return Some(SemanticOp::Br);
    }
    // Conditional branches: register-based (jnz/jz) and flag-based (b*)
    if lower == "jnz" || lower == "jz" || lower == "jnzd"
        || lower.starts_with("b") && !lower.starts_with("bswap")
    {
        return Some(SemanticOp::BrCond);
    }

    // Lock operations
    if lower.starts_with("acq") {
        return Some(SemanticOp::LockAcquire);
    }
    if lower.starts_with("rel") {
        return Some(SemanticOp::LockRelease);
    }

    // Return
    if lower == "ret" || lower.starts_with("ret.") {
        return Some(SemanticOp::Ret);
    }

    // Vector operations (vadd, vsub, vmul, etc.)
    // Map to same SemanticOp as scalar - execution will handle width
    if lower.starts_with("vadd") {
        return Some(SemanticOp::Add);
    }
    if lower.starts_with("vsub") {
        return Some(SemanticOp::Sub);
    }
    if lower.starts_with("vmul") || lower.starts_with("vmac") || lower.starts_with("vmsc") {
        return Some(SemanticOp::Mul);
    }
    if lower.starts_with("vand") {
        return Some(SemanticOp::And);
    }
    if lower.starts_with("vor") {
        return Some(SemanticOp::Or);
    }
    if lower.starts_with("vxor") {
        return Some(SemanticOp::Xor);
    }
    if lower.starts_with("vnot") || lower.starts_with("vbnot") {
        return Some(SemanticOp::Not);
    }
    if lower.starts_with("vshl") || lower.starts_with("vlshl") {
        return Some(SemanticOp::Shl);
    }
    if lower.starts_with("vshr") || lower.starts_with("vlshr") {
        return Some(SemanticOp::Srl);
    }
    if lower.starts_with("vashr") {
        return Some(SemanticOp::Sra);
    }
    if lower.starts_with("vsel") {
        return Some(SemanticOp::Select);
    }
    if lower.starts_with("vmov") {
        return Some(SemanticOp::Copy);
    }

    // Vector comparison operations
    if lower.starts_with("vlt") && !lower.starts_with("vltu") {
        return Some(SemanticOp::SetLt);
    }
    if lower.starts_with("vltu") {
        return Some(SemanticOp::SetUlt);
    }
    if lower.starts_with("vle") && !lower.starts_with("vleu") {
        return Some(SemanticOp::SetLe);
    }
    if lower.starts_with("vleu") {
        return Some(SemanticOp::SetUle);
    }
    if lower.starts_with("vgt") && !lower.starts_with("vgtu") {
        return Some(SemanticOp::SetGt);
    }
    if lower.starts_with("vgtu") {
        return Some(SemanticOp::SetUgt);
    }
    if lower.starts_with("vge") && !lower.starts_with("vgeu") {
        return Some(SemanticOp::SetGe);
    }
    if lower.starts_with("vgeu") {
        return Some(SemanticOp::SetUge);
    }
    if lower.starts_with("veq") {
        return Some(SemanticOp::SetEq);
    }
    if lower.starts_with("vne") {
        return Some(SemanticOp::SetNe);
    }

    // Vector abs/neg
    if lower.starts_with("vabs") {
        return Some(SemanticOp::Abs);
    }
    if lower.starts_with("vneg") {
        return Some(SemanticOp::Neg);
    }

    // Bit manipulation (scalar)
    if lower == "clz" || lower.starts_with("clz.") {
        return Some(SemanticOp::Ctlz);
    }
    if lower == "clb" || lower.starts_with("clb.") {
        return Some(SemanticOp::Ctlz); // CLB (count leading bits) ~ CLZ variant
    }

    // Sign/zero extension (scalar)
    if lower.starts_with("extend.s") || lower.starts_with("ext.s") || lower.starts_with("sext") {
        return Some(SemanticOp::SignExtend);
    }
    if lower.starts_with("extend.u") || lower.starts_with("ext.u") || lower.starts_with("zext") {
        return Some(SemanticOp::ZeroExtend);
    }

    // Division (scalar)
    if lower == "div" || lower == "divs" || lower.starts_with("div.") {
        return Some(SemanticOp::SDiv);
    }
    if lower == "divu" || lower.starts_with("divu.") {
        return Some(SemanticOp::UDiv);
    }
    if lower == "mod" || lower.starts_with("mod.") {
        return Some(SemanticOp::SRem);
    }

    // Carry ops (scalar, map to Add/Sub)
    if lower == "adc" || lower.starts_with("adc.") {
        return Some(SemanticOp::Add);
    }
    if lower == "sbc" || lower.starts_with("sbc.") {
        return Some(SemanticOp::Sub);
    }

    // Shift (alternate naming: ashl = arithmetic shift left)
    if lower == "ashl" || lower.starts_with("ashl.") {
        return Some(SemanticOp::Shl);
    }

    // Zero/nonzero test (scalar comparison, produce 0/1)
    if lower == "eqz" || lower.starts_with("eqz.") {
        return Some(SemanticOp::SetEq);
    }
    if lower == "nez" || lower.starts_with("nez.") {
        return Some(SemanticOp::SetNe);
    }

    // Event (control, treat as nop for execution purposes)
    if lower == "event" || lower.starts_with("event.") {
        return Some(SemanticOp::Nop);
    }

    // "ret lr" is a return (alternate mnemonic form)
    if lower == "ret lr" {
        return Some(SemanticOp::Ret);
    }

    // Vector boolean/bitwise ops (vband, vbor = vector boolean and/or)
    if lower.starts_with("vband") {
        return Some(SemanticOp::And);
    }
    if lower.starts_with("vbor") {
        return Some(SemanticOp::Or);
    }
    if lower.starts_with("vbneg") {
        return Some(SemanticOp::Neg);
    }

    // Vector conditional min/max (vmax_lt, vmin_ge, vmaxdiff_lt)
    if lower.starts_with("vmax_lt") || lower.starts_with("vmaxdiff_lt") {
        return Some(SemanticOp::SetLt);
    }
    if lower.starts_with("vmin_ge") {
        return Some(SemanticOp::SetGe);
    }

    // Vector data movement (vshift, vshuffle, vconv, vunpack)
    if lower.starts_with("vshift") || lower.starts_with("vshuffle") {
        return Some(SemanticOp::Copy); // Data rearrangement
    }
    if lower.starts_with("vconv") {
        return Some(SemanticOp::Copy); // Type conversion
    }
    if lower.starts_with("vunpack") {
        return Some(SemanticOp::Load); // Unpack from load channel
    }

    None
}

/// Infer element type from a mnemonic suffix.
///
/// Resolved once per encoding during TableGen loading (not per decoded instruction).
/// Returns None for instructions without an element type suffix.
pub fn infer_element_type(mnemonic: &str) -> Option<ElementType> {
    if mnemonic.ends_with("8") || mnemonic.contains(".i8") || mnemonic.contains(".u8") {
        if mnemonic.contains(".u") {
            Some(ElementType::UInt8)
        } else {
            Some(ElementType::Int8)
        }
    } else if mnemonic.ends_with("16") || mnemonic.contains(".i16") || mnemonic.contains(".u16") {
        if mnemonic.contains(".u") {
            Some(ElementType::UInt16)
        } else {
            Some(ElementType::Int16)
        }
    } else if mnemonic.ends_with("32") || mnemonic.contains(".i32") || mnemonic.contains(".u32") {
        if mnemonic.contains(".u") {
            Some(ElementType::UInt32)
        } else {
            Some(ElementType::Int32)
        }
    } else if mnemonic.contains("bf16") {
        Some(ElementType::BFloat16)
    } else if mnemonic.contains("f32") || mnemonic.contains("float") {
        Some(ElementType::Float32)
    } else {
        None
    }
}

/// Infer branch condition from mnemonic and semantic operation.
///
/// Only meaningful for BrCond instructions. Returns None for other semantics.
pub fn infer_branch_condition(mnemonic: &str, semantic: Option<SemanticOp>) -> Option<BranchCondition> {
    if semantic != Some(SemanticOp::BrCond) {
        return None;
    }
    let mn = mnemonic.to_lowercase();
    Some(if mn == "jnz" || mn == "jnzd" {
        BranchCondition::NotZero
    } else if mn == "jz" {
        BranchCondition::Zero
    } else if mn.starts_with("beq") || mn == "bz" {
        BranchCondition::Equal
    } else if mn.starts_with("bne") || mn == "bnz" {
        BranchCondition::NotEqual
    } else if mn.starts_with("blt") {
        BranchCondition::Less
    } else if mn.starts_with("bge") {
        BranchCondition::GreaterEqual
    } else {
        BranchCondition::NotEqual // Fallback
    })
}

/// Infer select variant from mnemonic and semantic operation.
///
/// Only meaningful for Select instructions. Returns None for other semantics.
pub fn infer_select_variant(mnemonic: &str, semantic: Option<SemanticOp>) -> Option<SelectVariant> {
    if semantic != Some(SemanticOp::Select) {
        return None;
    }
    let mn = mnemonic.to_lowercase();
    Some(if mn.contains("eqz") {
        SelectVariant::EqualZero
    } else if mn.contains("nez") {
        SelectVariant::NotEqualZero
    } else {
        SelectVariant::Generic
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tablegen::types::{InstrAttributes, OperandDef, TemplateParam};

    fn make_test_data() -> TableGenData {
        let mut data = TableGenData::new();

        // Add ALU slot
        data.slots.insert("alu_slot".to_string(), SlotDef {
            name: "alu_slot".to_string(),
            display_name: "Alu".to_string(),
            bits: 20,
            field: "alu".to_string(),
            artificial: false,
        });

        // Add format class: ALU r,r,r format
        // Encoding: {mRx0[4:0], mRx[4:0], mRy[4:0], op[3:0], 0b1}
        // Total: 5 + 5 + 5 + 4 + 1 = 20 bits
        let mut fields = HashMap::new();
        fields.insert("mRx0".to_string(), 5);
        fields.insert("mRx".to_string(), 5);
        fields.insert("mRy".to_string(), 5);

        data.formats.insert("AIE2_alu_r_rr_inst_alu".to_string(), FormatClass {
            name: "AIE2_alu_r_rr_inst_alu".to_string(),
            parent: Some("AIE2_inst_alu_instr32".to_string()),
            template_params: vec![TemplateParam { name: "op".to_string(), bits: 4 }],
            fields,
            slot_field: Some("alu".to_string()),
            encoding: vec![
                EncodingPart::FieldRef { name: "mRx0".to_string(), high: None, low: None },
                EncodingPart::FieldRef { name: "mRx".to_string(), high: None, low: None },
                EncodingPart::FieldRef { name: "mRy".to_string(), high: None, low: None },
                EncodingPart::FieldRef { name: "op".to_string(), high: None, low: None },
                EncodingPart::Literal { value: 0b1, width: 1 },
            ],
            field_sources: HashMap::new(),
        });

        // Add ADD instruction: op = 0b0000
        data.instructions.insert("ADD".to_string(), InstrDef {
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
            attributes: InstrAttributes::default(),
        });

        // Add SUB instruction: op = 0b0001
        data.instructions.insert("SUB".to_string(), InstrDef {
            name: "SUB".to_string(),
            format: "AIE2_alu_r_rr_inst_alu".to_string(),
            mixin_classes: vec![],
            template_args: vec![0b0001],
            mnemonic: "sub".to_string(),
            asm_string: "$mRx, $mRx0, $mRy".to_string(),
            outputs: vec![],
            inputs: vec![],
            implicit_regs: vec![],
            attributes: InstrAttributes::default(),
        });

        data
    }

    #[test]
    fn test_resolve_add_instruction() {
        let data = make_test_data();
        let resolver = Resolver::new(&data);

        let add = data.instructions.get("ADD").unwrap();
        let encoding = resolver.resolve_instruction(add).unwrap();

        assert_eq!(encoding.name, "ADD");
        assert_eq!(encoding.mnemonic, "add");
        assert_eq!(encoding.slot, "alu");
        assert_eq!(encoding.width, 20);

        // Fixed bits: op=0b0000, literal=0b1
        // Position: bits 4:1 are op (0b0000), bit 0 is literal (0b1)
        // Fixed mask should be 0b1_1111 = 0x1F (bits 4:0)
        // Fixed bits should be 0b0_0001 = 0x01 (op=0, literal=1)
        assert_eq!(encoding.fixed_mask, 0b1_1111);
        assert_eq!(encoding.fixed_bits, 0b0_0001);

        // Should have 3 operand fields
        assert_eq!(encoding.operand_fields.len(), 3);

        // mRx0 at bits 19:15, mRx at 14:10, mRy at 9:5
        // Names match AIE TableGen register field names (not Rust convention)
        #[allow(non_snake_case)]
        let mRx0 = encoding.operand_fields.iter().find(|f| f.name == "mRx0").unwrap();
        assert_eq!(mRx0.bit_position, 15);
        assert_eq!(mRx0.width, 5);

        #[allow(non_snake_case)]
        let mRx = encoding.operand_fields.iter().find(|f| f.name == "mRx").unwrap();
        assert_eq!(mRx.bit_position, 10);
        assert_eq!(mRx.width, 5);

        #[allow(non_snake_case)]
        let mRy = encoding.operand_fields.iter().find(|f| f.name == "mRy").unwrap();
        assert_eq!(mRy.bit_position, 5);
        assert_eq!(mRy.width, 5);
    }

    #[test]
    fn test_resolve_sub_instruction() {
        let data = make_test_data();
        let resolver = Resolver::new(&data);

        let sub = data.instructions.get("SUB").unwrap();
        let encoding = resolver.resolve_instruction(sub).unwrap();

        assert_eq!(encoding.name, "SUB");

        // Fixed mask same as ADD (same format)
        assert_eq!(encoding.fixed_mask, 0b1_1111);
        // But different fixed bits: op=0b0001, literal=0b1 -> 0b0_0011 = 0x03
        assert_eq!(encoding.fixed_bits, 0b0_0011);
    }

    #[test]
    fn test_encoding_matches() {
        let data = make_test_data();
        let resolver = Resolver::new(&data);

        let add = resolver.resolve_instruction(data.instructions.get("ADD").unwrap()).unwrap();
        let sub = resolver.resolve_instruction(data.instructions.get("SUB").unwrap()).unwrap();

        // Construct test words:
        // ADD: r5 = r3 + r2 -> mRx0=3, mRx=5, mRy=2, op=0, lit=1
        // Encoding: 00011_00101_00010_0000_1 = 0x18A01
        let add_word = 0b00011_00101_00010_0000_1u64;
        assert!(add.matches(add_word));
        assert!(!sub.matches(add_word));

        // SUB: r5 = r3 - r2 -> mRx0=3, mRx=5, mRy=2, op=1, lit=1
        // Encoding: 00011_00101_00010_0001_1 = 0x18A03
        let sub_word = 0b00011_00101_00010_0001_1u64;
        assert!(!add.matches(sub_word));
        assert!(sub.matches(sub_word));
    }

    #[test]
    fn test_operand_extraction() {
        let field = OperandField::new("mRx0", 15, 5);

        // Word with mRx0 = 0b10101 (21) at bits 19:15
        let word = 0b10101_00000_00000_0000_0u64;
        assert_eq!(field.extract(word), 21);
    }

    #[test]
    fn test_signed_operand_extraction() {
        let field = OperandField::new("imm", 0, 8).signed();

        // Positive value: 127
        assert_eq!(field.extract_signed(127), 127);

        // Negative value: -1 (0xFF in 8 bits)
        assert_eq!(field.extract_signed(0xFF), -1);

        // Negative value: -128 (0x80 in 8 bits)
        assert_eq!(field.extract_signed(0x80), -128);
    }

    #[test]
    fn test_resolve_by_slot() {
        let data = make_test_data();
        let by_slot = build_decoder_tables(&data);

        assert!(by_slot.contains_key("alu"));
        let alu_instrs = &by_slot["alu"];
        assert_eq!(alu_instrs.len(), 2); // ADD and SUB

        // Check they're sorted by specificity (both have same, so order may vary)
        for enc in alu_instrs {
            assert_eq!(enc.specificity(), 5); // 5 fixed bits
        }
    }

    #[test]
    fn test_format_not_found_error() {
        let data = TableGenData::new();
        let resolver = Resolver::new(&data);

        let instr = InstrDef {
            name: "BAD".to_string(),
            format: "NonexistentFormat".to_string(),
            mixin_classes: vec![],
            template_args: vec![],
            mnemonic: "bad".to_string(),
            asm_string: "".to_string(),
            outputs: vec![],
            inputs: vec![],
            implicit_regs: vec![],
            attributes: InstrAttributes::default(),
        };

        let result = resolver.resolve_instruction(&instr);
        assert!(matches!(result, Err(ResolveError::FormatNotFound(_))));
    }

    #[test]
    fn test_template_args_mismatch_error() {
        let mut data = make_test_data();

        // Modify ADD to have wrong number of template args
        data.instructions.get_mut("ADD").unwrap().template_args = vec![0, 1, 2];

        let resolver = Resolver::new(&data);
        let result = resolver.resolve_instruction(data.instructions.get("ADD").unwrap());

        assert!(matches!(result, Err(ResolveError::TemplateArgsMismatch { .. })));
    }

    // === O(1) SlotIndex Tests ===

    #[test]
    fn test_slot_index_build() {
        let data = make_test_data();
        let by_slot = build_decoder_tables(&data);

        let alu_index = SlotIndex::build("alu", by_slot["alu"].clone());

        // Common mask should be 0x1F (bits 4:0 are the opcode)
        assert_eq!(alu_index.common_mask, 0b1_1111);

        // Should have 2 encodings
        assert_eq!(alu_index.encoding_count(), 2);

        // No fallback needed
        let stats = alu_index.stats();
        assert_eq!(stats.fallback_count, 0);
    }

    #[test]
    fn test_slot_index_o1_decode() {
        let data = make_test_data();
        let by_slot = build_decoder_tables(&data);
        let alu_index = SlotIndex::build("alu", by_slot["alu"].clone());

        // ADD: r5 = r3 + r2 -> mRx0=3, mRx=5, mRy=2, op=0, lit=1
        let add_word = 0b00011_00101_00010_0000_1u64;
        let (encoding, operands) = alu_index.decode(add_word).expect("Should decode ADD");
        assert_eq!(encoding.name, "ADD");
        assert_eq!(operands.get("mRx0"), Some(&3));
        assert_eq!(operands.get("mRx"), Some(&5));
        assert_eq!(operands.get("mRy"), Some(&2));

        // SUB: r5 = r3 - r2 -> mRx0=3, mRx=5, mRy=2, op=1, lit=1
        let sub_word = 0b00011_00101_00010_0001_1u64;
        let (encoding, operands) = alu_index.decode(sub_word).expect("Should decode SUB");
        assert_eq!(encoding.name, "SUB");
        assert_eq!(operands.get("mRx0"), Some(&3));
    }

    #[test]
    fn test_slot_index_unknown() {
        let data = make_test_data();
        let by_slot = build_decoder_tables(&data);
        let alu_index = SlotIndex::build("alu", by_slot["alu"].clone());

        // Invalid instruction (wrong opcode bits)
        let bad_word = 0b00011_00101_00010_1111_0u64; // Wrong trailing bit
        assert!(alu_index.decode(bad_word).is_none());
    }

    #[test]
    fn test_decoder_index_build() {
        let data = make_test_data();
        let index = DecoderIndex::build(&data);

        assert!(!index.is_empty());
        assert!(index.slot_index("alu").is_some());
        assert_eq!(index.total_encodings(), 2);
    }

    #[test]
    fn test_decoder_index_decode_slot() {
        let data = make_test_data();
        let index = DecoderIndex::build(&data);

        let add_word = 0b00011_00101_00010_0000_1u64;
        let (encoding, _) = index.decode_slot("alu", add_word).expect("Should decode");
        assert_eq!(encoding.name, "ADD");
    }

    #[test]
    fn test_slot_index_stats() {
        let data = make_test_data();
        let by_slot = build_decoder_tables(&data);
        let alu_index = SlotIndex::build("alu", by_slot["alu"].clone());

        let stats = alu_index.stats();
        assert_eq!(stats.slot_name, "alu");
        assert_eq!(stats.common_mask, 0b1_1111);
        assert_eq!(stats.bucket_count, 2); // ADD and SUB have different opcodes
        assert_eq!(stats.fallback_count, 0);
        assert_eq!(stats.max_bucket_size, 1); // Each opcode maps to 1 instruction
    }

    #[test]
    fn test_decoder_index_with_real_llvm_aie() {
        use std::path::Path;
        use crate::tablegen::load_from_llvm_aie;

        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }

        let data = load_from_llvm_aie(llvm_aie_path).expect("Failed to load llvm-aie");
        let index = DecoderIndex::build(&data);

        eprintln!("DecoderIndex built with {} total encodings", index.total_encodings());

        // Print stats for each slot
        for stat in index.stats() {
            eprintln!(
                "  {}: common_mask=0x{:X}, buckets={}, fallback={}, max_bucket={}, avg_bucket={:.2}",
                stat.slot_name,
                stat.common_mask,
                stat.bucket_count,
                stat.fallback_count,
                stat.max_bucket_size,
                stat.avg_bucket_size
            );
        }

        // Verify the index is usable
        assert!(!index.is_empty());
        assert!(index.total_encodings() > 0);

        // Verify we have expected slots
        assert!(index.slot_index("alu").is_some(), "Should have alu slot");
        assert!(index.slot_index("lda").is_some(), "Should have lda slot");
    }

    // === Operand Ordering Tests ===

    #[test]
    fn test_input_order_from_instrdef() {
        // Verify that InstrEncoding.input_order matches InstrDef.inputs order
        let data = make_test_data();
        let resolver = Resolver::new(&data);

        let add = data.instructions.get("ADD").unwrap();
        let encoding = resolver.resolve_instruction(add).unwrap();

        // ADD's InstrDef.inputs = [mRx0, mRy] (in that order)
        assert_eq!(encoding.input_order, vec!["mRx0", "mRy"]);
        // Output is mRx
        assert_eq!(encoding.output_order, vec!["mRx"]);
    }

    #[test]
    fn test_semantic_inference_from_mnemonic() {
        // Test that mnemonics get semantic ops inferred
        assert_eq!(infer_semantic_from_mnemonic("sel.eqz"), Some(SemanticOp::Select));
        assert_eq!(infer_semantic_from_mnemonic("sel.nez"), Some(SemanticOp::Select));
        assert_eq!(infer_semantic_from_mnemonic("add"), Some(SemanticOp::Add));
        assert_eq!(infer_semantic_from_mnemonic("add.nc"), Some(SemanticOp::Add));
        assert_eq!(infer_semantic_from_mnemonic("sub"), Some(SemanticOp::Sub));
        assert_eq!(infer_semantic_from_mnemonic("mov"), Some(SemanticOp::Copy));
        assert_eq!(infer_semantic_from_mnemonic("lt"), Some(SemanticOp::SetLt));
        assert_eq!(infer_semantic_from_mnemonic("ge.s32"), Some(SemanticOp::SetGe));
        assert_eq!(infer_semantic_from_mnemonic("and"), Some(SemanticOp::And));
        assert_eq!(infer_semantic_from_mnemonic("or"), Some(SemanticOp::Or));
        assert_eq!(infer_semantic_from_mnemonic("xor"), Some(SemanticOp::Xor));
        assert_eq!(infer_semantic_from_mnemonic("lshl"), Some(SemanticOp::Shl));
        assert_eq!(infer_semantic_from_mnemonic("ashr"), Some(SemanticOp::Sra));
        assert_eq!(infer_semantic_from_mnemonic("lshr"), Some(SemanticOp::Srl));

        // Unsigned comparisons
        assert_eq!(infer_semantic_from_mnemonic("ltu"), Some(SemanticOp::SetUlt));
        assert_eq!(infer_semantic_from_mnemonic("geu"), Some(SemanticOp::SetUge));

        // Pointer ops (map to add)
        assert_eq!(infer_semantic_from_mnemonic("paddb"), Some(SemanticOp::Add));

        // Branch/call operations
        assert_eq!(infer_semantic_from_mnemonic("jl"), Some(SemanticOp::Call));
        assert_eq!(infer_semantic_from_mnemonic("jnz"), Some(SemanticOp::BrCond));
        assert_eq!(infer_semantic_from_mnemonic("bz"), Some(SemanticOp::BrCond));

        // Lock operations
        assert_eq!(infer_semantic_from_mnemonic("acq"), Some(SemanticOp::LockAcquire));
        assert_eq!(infer_semantic_from_mnemonic("rel"), Some(SemanticOp::LockRelease));

        // Done (halt) operation
        assert_eq!(infer_semantic_from_mnemonic("done"), Some(SemanticOp::Done));

        // Unknown mnemonics return None
        assert_eq!(infer_semantic_from_mnemonic("unknown"), None);
    }

    #[test]
    fn test_semantic_inference_from_structure() {
        // Call: Defs=[lr] + hasDelaySlot (JL, JL_IND)
        assert_eq!(
            infer_semantic_from_structure(
                &["lr".into()], &[], false, false, true, &[],
            ),
            Some(SemanticOp::Call),
        );

        // Return: Uses=[lr] + hasDelaySlot (RET)
        assert_eq!(
            infer_semantic_from_structure(
                &[], &["lr".into()], false, false, true, &[],
            ),
            Some(SemanticOp::Ret),
        );

        // Load: mayLoad=true (VLDA, LDA, etc.)
        assert_eq!(
            infer_semantic_from_structure(
                &[], &[], true, false, false, &[],
            ),
            Some(SemanticOp::Load),
        );

        // Store: mayStore=true (VST, ST, etc.)
        assert_eq!(
            infer_semantic_from_structure(
                &[], &[], false, true, false, &[],
            ),
            Some(SemanticOp::Store),
        );

        // Done: parent class chain contains "_done_"
        assert_eq!(
            infer_semantic_from_structure(
                &[], &[], false, false, false,
                &["AIE2_done_inst_alu".into()],
            ),
            Some(SemanticOp::Done),
        );

        // hasDelaySlot alone (without lr) returns None -- needs mnemonic
        // to distinguish j vs jnz vs jz
        assert_eq!(
            infer_semantic_from_structure(
                &["srCarry".into()], &[], false, false, true, &[],
            ),
            None,
        );

        // Pure arithmetic: no structural signals -> None
        assert_eq!(
            infer_semantic_from_structure(
                &["srCarry".into()], &[], false, false, false, &[],
            ),
            None,
        );
    }

    #[test]
    fn test_semantic_attached_to_encoding() {
        // Verify semantic is attached when resolving instructions
        let data = make_test_data();

        // ADD has "add" mnemonic, resolver should infer SemanticOp::Add
        let resolver = Resolver::new(&data);
        let add = data.instructions.get("ADD").unwrap();
        let encoding = resolver.resolve_instruction(add).unwrap();

        assert_eq!(encoding.semantic, Some(SemanticOp::Add));
    }

    #[test]
    fn test_select_semantic_from_mnemonic() {
        let mut data = make_test_data();

        // Add a SELEQZ instruction
        data.instructions.insert("SELEQZ".to_string(), InstrDef {
            name: "SELEQZ".to_string(),
            format: "AIE2_alu_r_rr_inst_alu".to_string(),
            mixin_classes: vec![],
            template_args: vec![0b0100],
            mnemonic: "sel.eqz".to_string(),
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
            implicit_regs: vec![ImplicitReg {
                reg_class: "eR27".to_string(),
                reg_num: 27,
                is_use: true,
            }],
            attributes: InstrAttributes::default(),
        });

        let resolver = Resolver::new(&data);
        let seleqz = data.instructions.get("SELEQZ").unwrap();
        let encoding = resolver.resolve_instruction(seleqz).unwrap();

        // Should have Select semantic from mnemonic inference
        assert_eq!(encoding.semantic, Some(SemanticOp::Select));

        // Should have implicit register r27
        assert_eq!(encoding.implicit_regs.len(), 1);
        assert_eq!(encoding.implicit_regs[0].reg_num, 27);
        assert!(encoding.implicit_regs[0].is_use);
    }

    #[test]
    fn test_detect_addressing_mode() {
        // Post-modify immediate
        assert_eq!(
            detect_addressing_mode("LDA_dms_lda_pstm_nrm_imm"),
            AddressingMode::PostModifyImmediate
        );
        assert_eq!(
            detect_addressing_mode("ST_dms_st_pstm_nrm_imm"),
            AddressingMode::PostModifyImmediate
        );

        // Post-modify register
        assert_eq!(
            detect_addressing_mode("LDA_dms_lda_pstm_nrm"),
            AddressingMode::PostModifyRegister
        );

        // Indexed immediate
        assert_eq!(
            detect_addressing_mode("LDA_S8_ag_idx_imm"),
            AddressingMode::IndexedImmediate
        );
        assert_eq!(
            detect_addressing_mode("LDA_dms_lda_idx_imm"),
            AddressingMode::IndexedImmediate
        );

        // Indexed register
        assert_eq!(
            detect_addressing_mode("LDA_dms_lda_idx"),
            AddressingMode::IndexedRegister
        );

        // Unknown (non-memory instructions)
        assert_eq!(
            detect_addressing_mode("ADD_add_r_ri"),
            AddressingMode::Unknown
        );
    }

    #[test]
    fn test_detect_mem_width() {
        // Byte (8-bit)
        assert_eq!(detect_mem_width("lda.s8"), InstrMemWidth::Byte);
        assert_eq!(detect_mem_width("lda.u8"), InstrMemWidth::Byte);
        assert_eq!(detect_mem_width("st.s8"), InstrMemWidth::Byte);

        // HalfWord (16-bit)
        assert_eq!(detect_mem_width("lda.s16"), InstrMemWidth::HalfWord);
        assert_eq!(detect_mem_width("lda.u16"), InstrMemWidth::HalfWord);
        assert_eq!(detect_mem_width("st.u16"), InstrMemWidth::HalfWord);

        // Word (32-bit, default)
        assert_eq!(detect_mem_width("lda"), InstrMemWidth::Word);
        assert_eq!(detect_mem_width("st"), InstrMemWidth::Word);
        assert_eq!(detect_mem_width("add"), InstrMemWidth::Word);

        // Vector (256-bit)
        assert_eq!(detect_mem_width("vlda"), InstrMemWidth::Vector256);
        assert_eq!(detect_mem_width("vldb"), InstrMemWidth::Vector256);
        assert_eq!(detect_mem_width("vst"), InstrMemWidth::Vector256);
    }

    #[test]
    fn test_operand_ordering_with_real_llvm_aie() {
        use std::path::Path;
        use crate::tablegen::load_from_llvm_aie;

        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }

        let data = load_from_llvm_aie(llvm_aie_path).expect("Failed to load llvm-aie");
        let resolver = Resolver::new(&data);

        // Check some key instructions have correct operand ordering

        // SELEQZ should have Select semantic and r27 implicit
        if let Some(seleqz) = data.instructions.get("SELEQZ") {
            let encoding = resolver.resolve_instruction(seleqz).unwrap();
            eprintln!("SELEQZ: semantic={:?}, input_order={:?}, implicit_regs={:?}",
                encoding.semantic, encoding.input_order, encoding.implicit_regs);

            assert_eq!(encoding.semantic, Some(SemanticOp::Select),
                "SELEQZ should have Select semantic");

            // Check for r27 in implicit registers
            let has_r27 = encoding.implicit_regs.iter()
                .any(|ir| ir.reg_num == 27 && ir.is_use);
            assert!(has_r27, "SELEQZ should have r27 as implicit use");
        }

        // Count instructions with semantic info
        let with_semantic: Vec<_> = data.instructions.values()
            .filter_map(|instr| resolver.resolve_instruction(instr).ok())
            .filter(|enc| enc.semantic.is_some())
            .collect();

        let total = data.instructions.len();
        eprintln!("Instructions with semantic: {}/{} ({:.1}%)",
            with_semantic.len(), total,
            100.0 * with_semantic.len() as f64 / total as f64);

        // Should have at least some semantic coverage
        assert!(with_semantic.len() > 0, "Should have some instructions with semantic info");
    }

    // === OperandType / classify_operand_type Tests ===

    #[test]
    fn test_classify_simple_registers() {
        assert_eq!(
            classify_operand_type("eR", "mRx"),
            OperandType::Register(RegisterKind::Scalar)
        );
        assert_eq!(
            classify_operand_type("eP", "ptr"),
            OperandType::Register(RegisterKind::Pointer)
        );
        assert_eq!(
            classify_operand_type("eM", "mod"),
            OperandType::Register(RegisterKind::ModifierM)
        );
        assert_eq!(
            classify_operand_type("eDN", "dn"),
            OperandType::Register(RegisterKind::ModifierDN)
        );
        assert_eq!(
            classify_operand_type("eDJ", "dj"),
            OperandType::Register(RegisterKind::ModifierDJ)
        );
        assert_eq!(
            classify_operand_type("eDC", "dc"),
            OperandType::Register(RegisterKind::ModifierDC)
        );
    }

    #[test]
    fn test_classify_vector_accumulator_registers() {
        assert_eq!(
            classify_operand_type("eWLE", "mW"),
            OperandType::Register(RegisterKind::Vector256)
        );
        assert_eq!(
            classify_operand_type("mWm", "mWm"),
            OperandType::Register(RegisterKind::Vector256)
        );
        assert_eq!(
            classify_operand_type("mXm", "mXm"),
            OperandType::Register(RegisterKind::Vector512)
        );
        assert_eq!(
            classify_operand_type("eAM", "mAM"),
            OperandType::Register(RegisterKind::Accumulator)
        );
        assert_eq!(
            classify_operand_type("mAMm", "mAMm"),
            OperandType::Register(RegisterKind::Accumulator)
        );
        assert_eq!(
            classify_operand_type("mBMm", "mBMm"),
            OperandType::Register(RegisterKind::Accumulator)
        );
    }

    #[test]
    fn test_classify_composite_registers() {
        assert_eq!(
            classify_operand_type("OP_mLdaScl", "mLdaScl"),
            OperandType::CompositeRegister(CompositeEncoder::LdaScl)
        );
        assert_eq!(
            classify_operand_type("OP_mSclSt", "mSclSt"),
            OperandType::CompositeRegister(CompositeEncoder::LdaScl)
        );
        assert_eq!(
            classify_operand_type("OP_mMvSclSrc", "mMvSclSrc"),
            OperandType::CompositeRegister(CompositeEncoder::MvSclSrc)
        );
        assert_eq!(
            classify_operand_type("OP_mMvSclDstCg", "mMvSclDstCg"),
            OperandType::CompositeRegister(CompositeEncoder::MvSclSrc)
        );
        assert_eq!(
            classify_operand_type("OP_mLdaCg", "mLdaCg"),
            OperandType::CompositeRegister(CompositeEncoder::LdaCg)
        );
        assert_eq!(
            classify_operand_type("OP_mAluCg", "mAluCg"),
            OperandType::CompositeRegister(CompositeEncoder::AluCg)
        );
    }

    #[test]
    fn test_classify_immediates() {
        assert_eq!(
            classify_operand_type("simm7", "imm"),
            OperandType::Immediate { signed: true, scale: 1 }
        );
        assert_eq!(
            classify_operand_type("imm12x4", "imm"),
            OperandType::Immediate { signed: false, scale: 4 }
        );
        assert_eq!(
            classify_operand_type("imm6x32", "imm"),
            OperandType::Immediate { signed: false, scale: 32 }
        );
        assert_eq!(
            classify_operand_type("imm5", "imm"),
            OperandType::Immediate { signed: false, scale: 1 }
        );
        assert_eq!(
            classify_operand_type("addr20", "cpmaddr"),
            OperandType::Immediate { signed: false, scale: 1 }
        );
    }

    #[test]
    fn test_classify_field_name_fallback() {
        // Lock IDs
        assert_eq!(classify_operand_type("", "id"), OperandType::LockId);
        assert_eq!(classify_operand_type("", "mLockId"), OperandType::LockId);

        // Constant fields from field name
        assert_eq!(
            classify_operand_type("", "c11s"),
            OperandType::Immediate { signed: true, scale: 1 }
        );
        assert_eq!(
            classify_operand_type("", "c5u"),
            OperandType::Immediate { signed: false, scale: 1 }
        );

        // Unknown fallback
        assert_eq!(classify_operand_type("", "unknown"), OperandType::Unknown);
    }

    #[test]
    fn test_operand_type_populated_on_resolve() {
        // Verify that resolve_instruction populates operand_type on fields
        let data = make_test_data();
        let resolver = Resolver::new(&data);

        let add = data.instructions.get("ADD").unwrap();
        let encoding = resolver.resolve_instruction(add).unwrap();

        // ADD has eR reg_class for all operands
        for field in &encoding.operand_fields {
            assert_eq!(
                field.operand_type,
                OperandType::Register(RegisterKind::Scalar),
                "Field '{}' should be classified as Scalar register",
                field.name
            );
        }
    }
}
