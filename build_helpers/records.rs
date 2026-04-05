//! Intermediate instruction record types for build-time extraction.
//!
//! These mirror `src/tablegen/tblgen_records.rs` but are independent: they
//! live in the build context and can't import main crate types. Fields that
//! reference enums in the main crate (SemanticOp, OperandType, etc.) use
//! `String` holding the Rust expression text (e.g., `"SemanticOp::Add"`).
//! The compiler validates these strings when compiling the generated code.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Encoding bits (mirrors EncodingBit)
// ---------------------------------------------------------------------------

/// A single bit in a slot encoding, used during build-time extraction.
#[derive(Debug, Clone)]
pub enum BuildEncodingBit {
    Zero,
    One,
    DontCare,
    FieldBit { field: String, bit: u8 },
}

// ---------------------------------------------------------------------------
// Slot encoding (mirrors SlotEncoding)
// ---------------------------------------------------------------------------

/// Parsed slot encoding from bits<N> field (build-time version).
#[derive(Debug, Clone)]
pub struct BuildSlotEncoding {
    pub slot: String,
    pub width: u8,
    pub parts: Vec<BuildEncodingBit>,
}

// ---------------------------------------------------------------------------
// Field fragments and operand fields (build-time versions)
// ---------------------------------------------------------------------------

/// A fragment of a split operand field (build-time version).
#[derive(Debug, Clone)]
pub struct BuildFieldFragment {
    pub inst_bit: u8,
    pub width: u8,
    pub target_bit: u8,
}

/// A resolved operand field (build-time version).
///
/// Where the main crate uses `OperandType`, we use a `String` holding the
/// Rust constructor expression (e.g., `"OperandType::Register(RegisterKind::Scalar)"`).
#[derive(Debug, Clone)]
pub struct BuildOperandField {
    pub name: String,
    pub bit_position: u8,
    pub width: u8,
    pub signed: bool,
    /// Rust expression string for the OperandType variant.
    pub operand_type: String,
    pub is_output: bool,
    pub fragments: Vec<BuildFieldFragment>,
}

// ---------------------------------------------------------------------------
// Implicit register (build-time version)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BuildImplicitReg {
    pub reg_class: String,
    pub reg_num: u32,
    pub is_use: bool,
}

// ---------------------------------------------------------------------------
// Instruction record (mirrors InstrRecord)
// ---------------------------------------------------------------------------

/// A parsed instruction record from TableGen (build-time version).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BuildInstrRecord {
    pub name: String,
    pub parents: Vec<String>,
    pub decoder_namespace: String,
    pub inputs: Vec<(String, String)>,
    pub outputs: Vec<(String, String)>,
    pub mnemonic: String,
    pub asm_string: String,
    pub slot_encoding: Option<BuildSlotEncoding>,
    pub defs: Vec<String>,
    pub uses: Vec<String>,
    pub may_load: bool,
    pub may_store: bool,
    pub has_side_effects: bool,
    pub has_complete_decoder: bool,
    pub has_delay_slot: bool,
    pub is_code_gen_only: bool,
    pub is_move_imm: bool,
    pub is_move_reg: bool,
    pub is_slot_nop: bool,
    pub is_branch: bool,
    pub is_call: bool,
    pub is_return: bool,
    pub is_select: bool,
    pub is_terminator: bool,
    pub is_compare: bool,
    pub is_composite: bool,
    pub itinerary_class: Option<String>,
}

// ---------------------------------------------------------------------------
// Fully resolved encoding (build-time version)
// ---------------------------------------------------------------------------

/// A fully resolved instruction encoding (build-time version).
///
/// All enum fields are stored as `String` holding the Rust expression text
/// that will be emitted into the generated code.
#[derive(Debug, Clone)]
pub struct BuildInstrEncoding {
    pub name: String,
    pub mnemonic: String,
    pub asm_string: String,
    pub slot: String,
    pub width: u8,
    pub fixed_mask: u64,
    pub fixed_bits: u64,
    pub operand_fields: Vec<BuildOperandField>,
    /// `None` = no semantic; `Some("SemanticOp::Add")` = known semantic.
    pub semantic: Option<String>,
    pub may_load: bool,
    pub may_store: bool,
    pub input_order: Vec<String>,
    pub output_order: Vec<String>,
    pub implicit_regs: Vec<BuildImplicitReg>,
    /// Rust expression string for AddressingMode variant.
    pub addressing_mode: String,
    /// Rust expression string for InstrMemWidth variant.
    pub mem_width: String,
    pub has_complete_decoder: bool,
    /// Rust expression string for ElementType variant, or None.
    pub element_type: Option<String>,
    /// Rust expression string for source ElementType (SRS/UPS dual-type), or None.
    pub from_type: Option<String>,
    /// Rust expression string for BranchCondition variant, or None.
    pub branch_condition: Option<String>,
    pub is_vector: bool,
    /// Rust expression string for SelectVariant, or None.
    pub select_variant: Option<String>,
    pub is_ptr_arithmetic: bool,
    pub is_sp_relative: bool,
    pub sched_class: Option<String>,
}

// ---------------------------------------------------------------------------
// SlotEncoding methods (mirrors compute_fixed_bits / extract_operand_fields)
// ---------------------------------------------------------------------------

impl BuildSlotEncoding {
    /// Compute fixed_mask and fixed_bits for instruction matching.
    pub fn compute_fixed_bits(&self) -> (u64, u64) {
        let mut mask: u64 = 0;
        let mut bits: u64 = 0;

        for (i, part) in self.parts.iter().enumerate() {
            let bit_pos = self.width as usize - 1 - i;
            match part {
                BuildEncodingBit::Zero => {
                    mask |= 1 << bit_pos;
                }
                BuildEncodingBit::One => {
                    mask |= 1 << bit_pos;
                    bits |= 1 << bit_pos;
                }
                BuildEncodingBit::DontCare | BuildEncodingBit::FieldBit { .. } => {}
            }
        }

        (mask, bits)
    }

    /// Extract operand fields from the encoding, handling split fields.
    pub fn extract_operand_fields(&self) -> Vec<BuildOperandField> {
        let mut field_bits: HashMap<String, Vec<(u8, u8)>> = HashMap::new();

        for (i, part) in self.parts.iter().enumerate() {
            if let BuildEncodingBit::FieldBit { field, bit } = part {
                let inst_bit = (self.width as usize - 1 - i) as u8;
                field_bits
                    .entry(field.clone())
                    .or_default()
                    .push((inst_bit, *bit));
            }
        }

        field_bits
            .into_iter()
            .map(|(name, mut bit_pairs)| {
                bit_pairs.sort_by_key(|&(_, target)| target);

                let logical_width = bit_pairs
                    .iter()
                    .map(|&(_, target)| target)
                    .max()
                    .unwrap_or(0)
                    + 1;
                let low_inst_bit = bit_pairs.iter().map(|&(inst, _)| inst).min().unwrap_or(0);

                let is_contiguous = bit_pairs.windows(2).all(|w| {
                    let (inst_a, tgt_a) = w[0];
                    let (inst_b, tgt_b) = w[1];
                    tgt_b == tgt_a + 1 && inst_b == inst_a + 1
                });

                if is_contiguous {
                    BuildOperandField {
                        name,
                        bit_position: low_inst_bit,
                        width: logical_width,
                        signed: false,
                        operand_type: "OperandType::Unknown".to_string(),
                        is_output: false,
                        fragments: Vec::new(),
                    }
                } else {
                    let fragments = build_fragments(&bit_pairs);
                    BuildOperandField {
                        name,
                        bit_position: low_inst_bit,
                        width: logical_width,
                        signed: false,
                        operand_type: "OperandType::Unknown".to_string(),
                        is_output: false,
                        fragments,
                    }
                }
            })
            .collect()
    }
}

/// Build FieldFragment entries from sorted (inst_bit, target_bit) pairs.
fn build_fragments(bit_pairs: &[(u8, u8)]) -> Vec<BuildFieldFragment> {
    if bit_pairs.is_empty() {
        return Vec::new();
    }

    let mut fragments = Vec::new();
    let mut run_start = 0;

    for i in 1..=bit_pairs.len() {
        let is_break = if i < bit_pairs.len() {
            let (inst_prev, tgt_prev) = bit_pairs[i - 1];
            let (inst_curr, tgt_curr) = bit_pairs[i];
            !(tgt_curr == tgt_prev + 1 && inst_curr == inst_prev + 1)
        } else {
            true
        };

        if is_break {
            let (inst_bit, target_bit) = bit_pairs[run_start];
            let width = (i - run_start) as u8;
            fragments.push(BuildFieldFragment {
                inst_bit,
                width,
                target_bit,
            });
            run_start = i;
        }
    }

    fragments
}
