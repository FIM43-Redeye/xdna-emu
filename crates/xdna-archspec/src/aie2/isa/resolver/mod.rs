//! Resolved instruction encodings and the decoder index over them.
//!
//! The build script (`build_helpers/`) resolves every llvm-aie instruction record into an
//! [`InstrEncoding`] (fixed bits, operand fields, semantic facts, and the TableGen
//! [`EncodingSchema`](super::schema::EncodingSchema)); the generated tables are loaded with
//! [`load_from_generated`](super::load_from_generated). [`DecoderIndex`] looks an encoding up
//! by the name the LLVM disassembler (FFI) reports.
//!
//! The regex-era runtime resolver (`Resolver` over hand-parsed `TableGenData`) and its
//! classification helpers were removed on 2026-11-04: nothing outside their own tests used them.

mod operand_types;

pub use operand_types::{AddressingMode, CompositeEncoder, InstrMemWidth, OperandType, RegisterKind};

use std::collections::HashMap;

use super::types::{BranchCondition, ElementType, ImplicitReg, SelectVariant, SemanticOp};

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
///
/// The fields before [`schema`](Self::schema) are the legacy, partly name-derived facts that the
/// interpreter consumes today; `schema` holds the TableGen-derived replacements, which consumers
/// adopt one at a time. `Default` exists for tests that build an encoding by hand.
#[derive(Debug, Clone, Default)]
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

    /// TableGen record facts: MC operands with their classes, format class and its template
    /// arguments, class lineage, address-generator form, pattern memory width, implicit
    /// registers with hardware encodings, stream mode. See [`EncodingSchema`].
    pub schema: super::schema::EncodingSchema,
}

/// O(1) instruction lookup index for a single slot.
///
/// LLVM's disassembler (via FFI) identifies the instruction name from raw
/// bits; that name is then looked up in `by_name` to retrieve the full
/// `InstrEncoding` with all semantic metadata.
#[derive(Debug, Clone)]
pub struct SlotIndex {
    /// The slot name (e.g., "alu", "lda")
    pub slot_name: String,

    /// Name-based lookup for the LLVM FFI decoder.
    /// Maps instruction name (e.g., "MOV_mv_cg") to its encoding.
    by_name: HashMap<String, InstrEncoding>,
}

impl SlotIndex {
    /// Build a slot index from the slot's encodings.
    ///
    /// Decoding a slot with no FFI counterpart (e.g. "nop") returns `None`.
    pub fn build(slot_name: impl Into<String>, encodings: Vec<InstrEncoding>) -> Self {
        let slot_name = slot_name.into();

        let by_name: HashMap<String, InstrEncoding> =
            encodings.into_iter().map(|e| (e.name.clone(), e)).collect();

        Self { slot_name, by_name }
    }

    /// Decode a word using LLVM's disassembler (via FFI).
    ///
    /// LLVM's MCDisassembler includes full TRY_DECODE register class
    /// validation, giving perfect disambiguation. This is the only decode
    /// path.
    #[inline]
    pub fn decode(&self, word: u64) -> Option<(&InstrEncoding, HashMap<String, u64>)> {
        let ffi_slot = super::decoder_ffi::slot_from_name(&self.slot_name)?;
        let name = super::decoder_ffi::decode_slot_name(ffi_slot, word)?;
        let encoding = self.by_name.get(&name)?;
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
        assert_eq!(field.extract_signed(127), 127);
        assert_eq!(field.extract_signed(0xFF), -1);
        assert_eq!(field.extract_signed(0x80), -128);
    }

    #[test]
    fn test_split_operand_extraction() {
        // i{9-1} at instruction bits 20..=28, i{0} at bit 3 (the MOV_mv_cg shape).
        let mut field = OperandField::new("i", 3, 10);
        field.fragments = vec![
            FieldFragment { inst_bit: 20, width: 9, target_bit: 1 },
            FieldFragment { inst_bit: 3, width: 1, target_bit: 0 },
        ];
        let word = (0b1_0000_0001u64 << 20) | (1 << 3);
        assert_eq!(field.extract(word), 0b10_0000_0011);
    }

    #[test]
    fn decoder_index_finds_encodings_by_slot_and_name() {
        let output = super::super::load_from_generated();
        let index = DecoderIndex::from_slot_encodings(output.encodings_by_slot);
        assert!(!index.is_empty());
        assert_eq!(index.slot_index("alu").map(|s| s.slot_name.as_str()), Some("alu"));
        let add = index.encoding_by_name("alu", "ADD").expect("ADD in the alu slot");
        assert_eq!(add.input_order, ["mRx0", "mRy"]);
        assert_eq!(add.output_order, ["mRx"]);
        assert!(index.encoding_by_name("alu", "NO_SUCH_INSTR").is_none());
    }
}
