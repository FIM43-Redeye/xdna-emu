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

use super::types::{EncodingPart, FormatClass, InstrDef, SemanticOp, SlotDef, TableGenData};

/// A resolved operand field within an instruction encoding.
///
/// Specifies where an operand can be extracted from the instruction bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandField {
    /// Field name from TableGen (e.g., "mRx", "imm")
    pub name: String,
    /// Bit position of LSB within the slot encoding
    pub bit_position: u8,
    /// Width in bits
    pub width: u8,
    /// Whether this is a signed immediate (for sign extension)
    pub signed: bool,
}

impl OperandField {
    /// Create a new operand field.
    pub fn new(name: impl Into<String>, bit_position: u8, width: u8) -> Self {
        Self {
            name: name.into(),
            bit_position,
            width,
            signed: false,
        }
    }

    /// Mark this field as signed.
    pub fn signed(mut self) -> Self {
        self.signed = true;
        self
    }

    /// Extract this field's value from an instruction word.
    #[inline]
    pub fn extract(&self, word: u64) -> u64 {
        let mask = (1u64 << self.width) - 1;
        (word >> self.bit_position) & mask
    }

    /// Extract as signed value (sign-extend if needed).
    #[inline]
    pub fn extract_signed(&self, word: u64) -> i64 {
        let unsigned = self.extract(word);
        if self.signed && self.width < 64 {
            // Sign extend
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

        // Process encoding parts to compute masks and fields
        let (fixed_mask, fixed_bits, operand_fields) =
            self.process_encoding(&format.encoding, &field_widths, &template_values)?;

        // Look up semantic operation
        let semantic = self.data.semantic_for_instruction(&instr.name)
            .map(|p| p.operation);

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

        // Sort each slot's instructions by specificity (most specific first)
        for encodings in by_slot.values_mut() {
            encodings.sort_by(|a, b| b.specificity().cmp(&a.specificity()));
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
    fn process_encoding(
        &self,
        parts: &[EncodingPart],
        field_widths: &HashMap<String, u8>,
        template_values: &HashMap<String, u64>,
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

                        // Check if we should merge with existing field of same name
                        // (for split fields like immediate{10-6} and immediate{5-0})
                        if let Some(existing) = operand_fields.iter_mut().find(|f| f.name == *name) {
                            // For now, just note that this is a split field
                            // Full handling would track all slices
                            // We'll use the first (MSB) slice's position
                            existing.width = existing.width.saturating_add(w);
                        } else {
                            operand_fields.push(OperandField::new(name, bit_pos, w));
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
pub fn build_decoder_tables(data: &TableGenData) -> HashMap<String, Vec<InstrEncoding>> {
    Resolver::new(data).resolve_by_slot()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tablegen::types::{InstrAttributes, OperandDef, TemplateParam};
    use std::collections::HashSet;

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
        });

        // Add ADD instruction: op = 0b0000
        data.instructions.insert("ADD".to_string(), InstrDef {
            name: "ADD".to_string(),
            format: "AIE2_alu_r_rr_inst_alu".to_string(),
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
            attributes: InstrAttributes::default(),
        });

        // Add SUB instruction: op = 0b0001
        data.instructions.insert("SUB".to_string(), InstrDef {
            name: "SUB".to_string(),
            format: "AIE2_alu_r_rr_inst_alu".to_string(),
            template_args: vec![0b0001],
            mnemonic: "sub".to_string(),
            asm_string: "$mRx, $mRx0, $mRy".to_string(),
            outputs: vec![],
            inputs: vec![],
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
        let mRx0 = encoding.operand_fields.iter().find(|f| f.name == "mRx0").unwrap();
        assert_eq!(mRx0.bit_position, 15);
        assert_eq!(mRx0.width, 5);

        let mRx = encoding.operand_fields.iter().find(|f| f.name == "mRx").unwrap();
        assert_eq!(mRx.bit_position, 10);
        assert_eq!(mRx.width, 5);

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
            template_args: vec![],
            mnemonic: "bad".to_string(),
            asm_string: "".to_string(),
            outputs: vec![],
            inputs: vec![],
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
}
