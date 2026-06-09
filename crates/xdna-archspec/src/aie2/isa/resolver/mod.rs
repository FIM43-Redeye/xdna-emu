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

mod operand_classification;
mod semantic_inference;

pub use operand_classification::{
    AddressingMode, CompositeEncoder, InstrMemWidth, OperandType, RegisterKind, classify_operand_type,
    detect_addressing_mode, detect_mem_width, detect_mem_width_full,
};
pub use semantic_inference::{
    infer_branch_condition, infer_dual_element_types, infer_element_type, infer_select_variant,
    infer_semantic_from_structure, refine_branch_semantic, refine_fused_semantic, refine_matmul_semantic,
};

use std::collections::HashMap;

use super::types::{
    BranchCondition, ElementType, EncodingPart, FormatClass, ImplicitReg, InstrDef, MixinClass,
    SelectVariant, SemanticOp, SlotDef, TableGenData,
};

// Operand classification (AddressingMode, OperandType, etc.) -> operand_classification.rs
// Semantic inference (infer_*, refine_*) -> semantic_inference.rs

/// A fragment of a split operand field.
///
/// AIE2 VLIW encodings sometimes scatter operand bits across non-contiguous
/// positions in the instruction word. For example, MOV_mv_cg encodes a 10-bit
/// immediate as `{i{9-1}, ..fixed.., i{0}, ..fixed..}`. Each non-contiguous
/// piece is a FieldFragment.
///
/// See also [`operand_classification`] for operand type classification and
/// [`semantic_inference`] for semantic operation inference.
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
    /// Whether this operand is an output (destination). From TableGen (outs) vs (ins).
    pub is_output: bool,
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
            is_output: false,
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

    /// Assembly format string (e.g., "$mRx, $mRx0, $mRy")
    pub asm_string: String,

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

    /// Source element type for dual-type instructions (SRS/UPS).
    /// The output type goes in `element_type`; the input type goes here.
    /// None for single-type instructions.
    pub from_type: Option<ElementType>,

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

    /// Whether this instruction implicitly uses SP (e.g., spill/fill instructions).
    /// When true and no explicit Memory/PointerReg operand is found, the decoder
    /// converts the Immediate operand to Memory { base: 6 (SP), offset: imm }.
    pub is_sp_relative: bool,

    /// Itinerary class from TableGen (e.g., "II_ADD", "II_LDA", "II_VMUL").
    /// Used for cross-validating latency values against the compiler's scheduling model.
    pub sched_class: Option<String>,
}

impl InstrEncoding {}

/// Errors that can occur during resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// Format class not found
    FormatNotFound(String),
    /// Slot not found
    SlotNotFound(String),
    /// Template parameter count mismatch
    TemplateArgsMismatch { expected: usize, got: usize },
    /// Unknown field in encoding
    UnknownField(String),
    /// Encoding width mismatch
    WidthMismatch { expected: u8, computed: u8 },
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
    fn merge_field_sources(&self, format: &FormatClass, instr: &InstrDef) -> HashMap<String, Vec<String>> {
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
        let format = self
            .data
            .formats
            .get(&instr.format)
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
            self.process_encoding(&format.encoding, &field_widths, &template_values, &merged_field_sources)?;

        // Populate operand_type on each field using OperandDef.reg_class
        let all_operand_defs: Vec<&super::types::OperandDef> =
            instr.outputs.iter().chain(instr.inputs.iter()).collect();

        for field in &mut operand_fields {
            if let Some(opdef) = all_operand_defs.iter().find(|od| od.name == field.name) {
                field.operand_type = classify_operand_type(&opdef.reg_class, &field.name);
                field.is_output = opdef.is_output;
            } else {
                // No matching OperandDef -- use field-name fallback
                field.operand_type = classify_operand_type("", &field.name);
                // is_output stays false (safe default: unknowns are inputs)
            }
            // Sync signed flag from operand type
            if let OperandType::Immediate { signed: true, .. } = &field.operand_type {
                field.signed = true;
            }
        }

        // Three-tier semantic inference: pattern -> structural -> fused refinement.
        // Pattern-based semantics come from parsed Pat<> entries in TableGenData.
        // Structural inference uses TableGen attributes (mayLoad, Defs, etc.).
        // Fused refinement uses the instruction NAME to distinguish Load/Store
        // from fused ops (UPS/SRS/Pack/Unpack/Convert) that LLVM lacks Pat<>
        // entries for (they're selected in C++, not TableGen patterns).
        let defs_vec: Vec<String> = instr.attributes.defs.iter().cloned().collect();
        let uses_vec: Vec<String> = instr.attributes.uses.iter().cloned().collect();
        let semantic = self
            .data
            .semantic_for_instruction(&instr.name)
            .map(|p| p.operation)
            .or_else(|| {
                infer_semantic_from_structure(
                    &defs_vec,
                    &uses_vec,
                    instr.attributes.may_load,
                    instr.attributes.may_store,
                    false, // regex parser doesn't extract hasDelaySlot
                    &[],   // regex parser doesn't have parent class chain
                )
            });

        // Refine Br -> BrCond for conditional branches
        let semantic = refine_branch_semantic(&instr.mnemonic, semantic);

        // Refine Load/Store -> fused semantics (UPS/SRS/Pack/Unpack/Convert)
        // using the TableGen instruction name, not the runtime mnemonic.
        let semantic = refine_fused_semantic(&instr.name, semantic);

        // Refine the matrix-multiply VMUL (Mul -> MatMul): in AIE2 a `vmul`
        // writing a cm/bm accumulator IS a fresh matrix multiply, not the
        // elementwise Mul that Pat<> inference assigns.
        let semantic = refine_matmul_semantic(&instr.name, semantic);

        // Extract operand ordering from InstrDef
        let input_order: Vec<String> = instr.inputs.iter().map(|o| o.name.clone()).collect();
        let output_order: Vec<String> = instr.outputs.iter().map(|o| o.name.clone()).collect();

        // Pre-resolve metadata from mnemonic (once per encoding, not per decode)
        let is_vector = instr.mnemonic.starts_with('v') || instr.mnemonic.starts_with('V');
        let is_ptr_arithmetic = instr.mnemonic.starts_with("padd");
        // Spill/fill instructions implicitly use SP as the base address.
        // Detected from TableGen Uses = [SP] attribute.
        let is_sp_relative = instr.attributes.uses.iter().any(|u| u == "SP");
        let (dual_et, dual_ft) = infer_dual_element_types(&instr.name);
        let element_type = dual_et.or_else(|| infer_element_type(&instr.mnemonic));
        let from_type = dual_ft;
        let branch_condition = infer_branch_condition(&instr.mnemonic, semantic);
        let select_variant = infer_select_variant(&instr.mnemonic, semantic);

        Ok(InstrEncoding {
            name: instr.name.clone(),
            mnemonic: instr.mnemonic.clone(),
            asm_string: String::new(),
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
            mem_width: detect_mem_width_full(&instr.name, &instr.mnemonic),
            // Regex parser doesn't parse hasCompleteDecoder; assume true (safe default)
            has_complete_decoder: true,
            element_type,
            from_type,
            branch_condition,
            is_vector,
            select_variant,
            is_ptr_arithmetic,
            is_sp_relative,
            sched_class: None, // Regex parser doesn't extract itinerary class
        })
    }

    /// Resolve all instructions in the TableGen data.
    pub fn resolve_all(&self) -> Vec<Result<InstrEncoding, ResolveError>> {
        self.data
            .instructions
            .values()
            .map(|instr| self.resolve_instruction(instr))
            .collect()
    }

    /// Resolve all instructions, filtering out errors.
    pub fn resolve_all_ok(&self) -> Vec<InstrEncoding> {
        self.resolve_all().into_iter().filter_map(|r| r.ok()).collect()
    }

    /// Resolve all instructions and group by slot.
    pub fn resolve_by_slot(&self) -> HashMap<String, Vec<InstrEncoding>> {
        let mut by_slot: HashMap<String, Vec<InstrEncoding>> = HashMap::new();

        for encoding in self.resolve_all_ok() {
            by_slot.entry(encoding.slot.clone()).or_default().push(encoding);
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
            let width = part
                .width(field_widths)
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
                                let logical_width =
                                    field_widths.get(name).copied().unwrap_or(existing.width + w);
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
/// Returns encodings grouped by slot.
/// JL and J_jump_imm are included from TableGen (they have `isCodeGenOnly = 0`
/// and use the "lng" slot, which is in the parser's slot_names array).
pub fn build_decoder_tables(data: &TableGenData) -> HashMap<String, Vec<InstrEncoding>> {
    Resolver::new(data).resolve_by_slot()
}

/// O(1) instruction lookup index for a single slot.
///
/// Uses a HashMap keyed on the common opcode bits to achieve constant-time
/// lookup via LLVM decoder bytecode tables.
///
/// The bytecode table identifies the instruction name from raw bits,
/// which is then looked up in `by_name` to retrieve the full
/// `InstrEncoding` with all semantic metadata.
#[derive(Debug, Clone)]
pub struct SlotIndex {
    /// The slot name (e.g., "alu", "lda")
    pub slot_name: String,

    /// Name-based lookup for LLVM bytecode decoder integration.
    /// Maps instruction name (e.g., "MOV_mv_cg") to its encoding.
    by_name: HashMap<String, InstrEncoding>,

    /// LLVM decoder bytecode table for this slot.
    /// `None` for slots with no bytecode table (e.g. "nop").
    decoder_table: Option<super::decoder_bytecode::DecoderTable>,
}

impl SlotIndex {
    /// Build a slot index from encodings with an optional LLVM decoder bytecode table.
    ///
    /// The decoder table is the sole disambiguation mechanism. If `None`,
    /// all decodes for this slot will return `None` (unknown instruction).
    pub fn build(
        slot_name: impl Into<String>,
        encodings: Vec<InstrEncoding>,
        decoder_table: Option<super::decoder_bytecode::DecoderTable>,
    ) -> Self {
        let slot_name = slot_name.into();

        let by_name: HashMap<String, InstrEncoding> =
            encodings.into_iter().map(|e| (e.name.clone(), e)).collect();

        Self { slot_name, by_name, decoder_table }
    }

    /// Decode a word using LLVM's disassembler (via FFI) with bytecode fallback.
    ///
    /// The FFI path uses LLVM's MCDisassembler which includes full TRY_DECODE
    /// register class validation, giving perfect disambiguation.  Falls back
    /// to the bytecode interpreter if the FFI decoder is unavailable.
    #[inline]
    pub fn decode(&self, word: u64) -> Option<(&InstrEncoding, HashMap<String, u64>)> {
        // Try LLVM FFI decoder first (perfect TRY_DECODE disambiguation).
        if let Some(ffi_slot) = super::decoder_ffi::slot_from_name(&self.slot_name) {
            if let Some(name) = super::decoder_ffi::decode_slot_name(ffi_slot, word) {
                if let Some(encoding) = self.by_name.get(&name) {
                    let operands = self.extract_operands(encoding, word);
                    return Some((encoding, operands));
                }
            }
        }

        // Fall back to bytecode interpreter.
        let table = self.decoder_table.as_ref()?;
        let instr_name = table.decode(word)?;
        let encoding = self.by_name.get(instr_name)?;
        let operands = self.extract_operands(encoding, word);
        Some((encoding, operands))
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
    /// Build from pre-resolved slot encodings with LLVM decoder bytecode tables.
    ///
    /// Decoder tables are attached to the corresponding slot indices for
    /// authoritative disambiguation via LLVM bytecode.
    pub fn from_slot_encodings(
        by_slot: HashMap<String, Vec<InstrEncoding>>,
        mut decoder_tables: HashMap<String, super::decoder_bytecode::DecoderTable>,
    ) -> Self {
        let slots = by_slot
            .into_iter()
            .map(|(name, encodings)| {
                let decoder = decoder_tables.remove(&name);
                let index = SlotIndex::build(&name, encodings, decoder);
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

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Look up an InstrEncoding by instruction name within a slot.
    ///
    /// Used by the LLVM FFI decode path to retrieve metadata (semantic,
    /// element_type, etc.) after LLVM identifies the instruction name.
    pub fn encoding_by_name(&self, slot_name: &str, instr_name: &str) -> Option<&InstrEncoding> {
        self.slots.get(slot_name).and_then(|idx| idx.by_name.get(instr_name))
    }

    /// Find the most specific encoding matching `bits` in a slot.
    ///
    /// LLVM's MCDisassembler sometimes picks a general encoding (e.g. VMAC)
    /// when a more specific one (e.g. VADD_F) also matches the same bits.
    /// This happens because `hasCompleteDecoder = 0` means multiple encodings
    /// can match the same bit pattern.
    ///
    /// This method checks ALL encodings in the slot and returns the one with
    /// the most constrained fixed_mask (highest popcount) whose fixed_bits
    /// match. More constrained = more specific = better match.
    ///
    /// Returns `None` if no encoding matches, or the LLVM-selected encoding
    /// if it's already the most specific.
    pub fn refine_encoding<'a>(
        &'a self,
        slot_name: &str,
        llvm_name: &str,
        bits: u64,
    ) -> Option<&'a InstrEncoding> {
        let slot = self.slots.get(slot_name)?;
        let llvm_enc = slot.by_name.get(llvm_name)?;

        // Find the most specific encoding that matches these bits.
        let mut best: Option<&InstrEncoding> = None;
        let mut best_specificity: u32 = 0;

        for enc in slot.by_name.values() {
            // Skip if bits don't match this encoding's fixed pattern.
            if (bits & enc.fixed_mask) != enc.fixed_bits {
                continue;
            }
            let specificity = enc.fixed_mask.count_ones();
            if specificity > best_specificity {
                best_specificity = specificity;
                best = Some(enc);
            }
        }

        // If the best match is the same as what LLVM returned, no refinement needed.
        // If different, return the more specific one.
        let refined = best.unwrap_or(llvm_enc);
        if refined.name != llvm_enc.name {
            log::trace!(
                "[REFINE] {} -> {} (specificity {} > {})",
                llvm_enc.name,
                refined.name,
                refined.fixed_mask.count_ones(),
                llvm_enc.fixed_mask.count_ones(),
            );
        }
        Some(refined)
    }

    /// Iterate over all encodings across all slots.
    pub fn all_encodings(&self) -> impl Iterator<Item = &InstrEncoding> {
        self.slots.values().flat_map(|s| s.by_name.values())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::{InstrAttributes, OperandDef, TemplateParam};

    fn make_test_data() -> TableGenData {
        let mut data = TableGenData::new();

        // Add ALU slot
        data.slots.insert(
            "alu_slot".to_string(),
            SlotDef {
                name: "alu_slot".to_string(),
                display_name: "Alu".to_string(),
                bits: 20,
                field: "alu".to_string(),
                artificial: false,
            },
        );

        // Add format class: ALU r,r,r format
        // Encoding: {mRx0[4:0], mRx[4:0], mRy[4:0], op[3:0], 0b1}
        // Total: 5 + 5 + 5 + 4 + 1 = 20 bits
        let mut fields = HashMap::new();
        fields.insert("mRx0".to_string(), 5);
        fields.insert("mRx".to_string(), 5);
        fields.insert("mRy".to_string(), 5);

        data.formats.insert(
            "AIE2_alu_r_rr_inst_alu".to_string(),
            FormatClass {
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
            },
        );

        // Add ADD instruction: op = 0b0000
        data.instructions.insert(
            "ADD".to_string(),
            InstrDef {
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
                    OperandDef { is_output: false, reg_class: "eR".to_string(), name: "mRx0".to_string() },
                    OperandDef { is_output: false, reg_class: "eR".to_string(), name: "mRy".to_string() },
                ],
                implicit_regs: vec![],
                attributes: InstrAttributes::default(),
            },
        );

        // Add SUB instruction: op = 0b0001
        data.instructions.insert(
            "SUB".to_string(),
            InstrDef {
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
            },
        );

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

        // Both have 5 fixed bits in their encoding mask
        for enc in alu_instrs {
            assert_eq!(enc.fixed_mask.count_ones(), 5);
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

    // === SlotIndex Tests ===
    //
    // Note: SlotIndex now requires an LLVM decoder bytecode table to decode.
    // Tests using synthetic data without bytecode tables can only verify
    // construction (not decode). Decode tests use real llvm-aie data.

    #[test]
    fn test_slot_index_build_without_decoder() {
        let data = make_test_data();
        let by_slot = build_decoder_tables(&data);

        // Build without bytecode decoder table.
        let alu_index = SlotIndex::build("alu", by_slot["alu"].clone(), None);
        assert_eq!(alu_index.slot_name, "alu");
        // The decode() call itself requires LLVM FFI link (Task 8/9),
        // so we only verify construction here in archspec's standalone tests.
        // xdna-emu's test suite exercises the full decode path.
    }

    #[test]
    fn test_decoder_index_build() {
        let data = make_test_data();
        let by_slot = build_decoder_tables(&data);

        // Build without decoder tables -- construction succeeds, decodes return None
        let index = DecoderIndex::from_slot_encodings(by_slot, HashMap::new());
        assert!(!index.is_empty());
        assert!(index.slot_index("alu").is_some());
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
    fn test_operand_type_populated_on_resolve() {
        // Verify that resolve_instruction populates operand_type and is_output on fields
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

        // Verify is_output: mRx is the output, mRx0 and mRy are inputs
        for field in &encoding.operand_fields {
            let expected_output = field.name == "mRx";
            assert_eq!(
                field.is_output, expected_output,
                "Field '{}' is_output should be {}",
                field.name, expected_output,
            );
        }
    }
}
