//! Instruction record types and encoding conversion.
//!
//! Defines `InstrRecord`, `SlotEncoding`, and `EncodingBit` -- the intermediate
//! representation between TableGen extraction (native or text) and the decoder's
//! `InstrEncoding`. The key method is `InstrRecord::to_encoding()` which converts
//! parsed records into decoder-ready encodings with operand fields, fixed bits,
//! and structural semantic inference.
//!
//! Text-based `--print-records` parsers were removed after native TableGen
//! integration (Phase 2, commit 6d12812). Data is now extracted in-process
//! via the `tblgen` crate in `native.rs`.

use std::collections::HashMap;

use super::resolver::{FieldFragment, InstrEncoding, OperandField, OperandType, classify_operand_type, detect_addressing_mode, detect_mem_width, infer_branch_condition, infer_element_type, infer_select_variant, infer_semantic_from_structure, refine_branch_semantic};
use super::types::{ImplicitReg, SemanticOp};

/// A parsed instruction record from TableGen.
#[derive(Debug, Clone)]
pub struct InstrRecord {
    /// Instruction definition name (e.g., "ADD_add_r_ri")
    pub name: String,
    /// Parent classes for inheritance (e.g., ["AIE2Inst", "AIE2SlotInst", ...])
    pub parents: Vec<String>,
    /// Decoder namespace = slot (e.g., "Alu", "Lda", "Vec")
    pub decoder_namespace: String,
    /// Input operands: (type, name) pairs
    pub inputs: Vec<(String, String)>,
    /// Output operands: (type, name) pairs
    pub outputs: Vec<(String, String)>,
    /// Assembly mnemonic from AsmString (e.g., "add")
    pub mnemonic: String,
    /// Full assembly string (e.g., "add    $mRx, $mRx0, $c7s")
    pub asm_string: String,
    /// Slot encoding bits (e.g., alu, lda, vec)
    pub slot_encoding: Option<SlotEncoding>,
    /// Implicit register definitions
    pub defs: Vec<String>,
    /// Implicit register uses
    pub uses: Vec<String>,
    /// mayLoad flag
    pub may_load: bool,
    /// mayStore flag
    pub may_store: bool,
    /// hasSideEffects flag
    pub has_side_effects: bool,
    /// hasCompleteDecoder flag from TableGen.
    /// When false, the encoding may be ambiguous and needs post-decode validation.
    /// We deprioritize these during disambiguation.
    pub has_complete_decoder: bool,
    /// hasDelaySlot flag -- identifies control flow instructions (branches, calls, returns).
    /// On AIE2 hardware encodings this is the most reliable structural indicator of
    /// control flow, since isBranch/isCall/isReturn are only set on Pseudo variants.
    pub has_delay_slot: bool,
    /// isCodeGenOnly flag -- instruction exists only for the compiler's instruction
    /// selection (codegen), not for the assembler/disassembler. These should be
    /// excluded from decoder tables since they are aliases (e.g., MOV_OR is
    /// `or $rd, $rs, $rs` with only one input operand).
    pub is_code_gen_only: bool,
    /// isMoveImm flag -- instruction is an immediate-to-register move.
    pub is_move_imm: bool,
    /// isMoveReg flag -- instruction is a register-to-register move.
    pub is_move_reg: bool,
    /// isSlotNOP flag -- instruction is a slot-specific NOP (e.g., NOPA, NOPB).
    pub is_slot_nop: bool,
    /// isBranch flag -- instruction is a branch.
    pub is_branch: bool,
    /// isCall flag -- instruction is a call.
    pub is_call: bool,
    /// isReturn flag -- instruction is a return.
    pub is_return: bool,
    /// isSelect flag -- instruction is a select/conditional-move.
    pub is_select: bool,
    /// isTerminator flag -- instruction is a basic-block terminator.
    pub is_terminator: bool,
    /// isCompare flag -- instruction is a comparison.
    pub is_compare: bool,
    /// isComposite flag -- this record is a VLIW bundle envelope, not a real instruction.
    pub is_composite: bool,
    /// Itinerary class name from `InstrItinClass Itinerary = II_ADD;`.
    pub itinerary_class: Option<String>,
}

/// Parsed slot encoding from bits<N> field.
#[derive(Debug, Clone)]
pub struct SlotEncoding {
    /// Slot name (alu, lda, ldb, st, mv, vec, lng)
    pub slot: String,
    /// Bit width (20, 21, 16, etc.)
    pub width: u8,
    /// Encoding parts, MSB first
    pub parts: Vec<EncodingBit>,
}

/// A single bit or bit slice in an encoding.
#[derive(Debug, Clone)]
pub enum EncodingBit {
    /// Literal 0
    Zero,
    /// Literal 1
    One,
    /// Don't care (?)
    DontCare,
    /// Field bit reference (e.g., mRx0{4} -> field="mRx0", bit=4)
    FieldBit { field: String, bit: u8 },
}

impl SlotEncoding {
    /// Compute fixed_mask and fixed_bits for instruction matching.
    pub fn compute_fixed_bits(&self) -> (u64, u64) {
        let mut mask: u64 = 0;
        let mut bits: u64 = 0;

        for (i, part) in self.parts.iter().enumerate() {
            let bit_pos = self.width as usize - 1 - i;
            match part {
                EncodingBit::Zero => {
                    mask |= 1 << bit_pos;
                }
                EncodingBit::One => {
                    mask |= 1 << bit_pos;
                    bits |= 1 << bit_pos;
                }
                EncodingBit::DontCare | EncodingBit::FieldBit { .. } => {}
            }
        }

        (mask, bits)
    }

    /// Extract operand fields from the encoding, handling split fields.
    pub fn extract_operand_fields(&self) -> Vec<OperandField> {
        let mut field_bits: HashMap<String, Vec<(u8, u8)>> = HashMap::new();

        for (i, part) in self.parts.iter().enumerate() {
            if let EncodingBit::FieldBit { field, bit } = part {
                let inst_bit = (self.width as usize - 1 - i) as u8;
                field_bits.entry(field.clone())
                    .or_default()
                    .push((inst_bit, *bit));
            }
        }

        field_bits
            .into_iter()
            .map(|(name, mut bit_pairs)| {
                bit_pairs.sort_by_key(|&(_, target)| target);

                let logical_width = bit_pairs.iter()
                    .map(|&(_, target)| target)
                    .max().unwrap_or(0) + 1;
                let low_inst_bit = bit_pairs.iter()
                    .map(|&(inst, _)| inst)
                    .min().unwrap_or(0);

                let is_contiguous = bit_pairs.windows(2).all(|w| {
                    let (inst_a, tgt_a) = w[0];
                    let (inst_b, tgt_b) = w[1];
                    tgt_b == tgt_a + 1 && inst_b == inst_a + 1
                });

                if is_contiguous {
                    OperandField::new(name, low_inst_bit, logical_width)
                } else {
                    let fragments = build_fragments(&bit_pairs);
                    let mut field = OperandField::new(name, low_inst_bit, logical_width);
                    field.fragments = fragments;
                    field
                }
            })
            .collect()
    }
}

/// Build FieldFragment entries from sorted (inst_bit, target_bit) pairs.
fn build_fragments(bit_pairs: &[(u8, u8)]) -> Vec<FieldFragment> {
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
            fragments.push(FieldFragment {
                inst_bit,
                width,
                target_bit,
            });
            run_start = i;
        }
    }

    fragments
}

impl InstrRecord {
    /// Convert to InstrEncoding for the decoder.
    ///
    /// Returns None for `isCodeGenOnly` instructions and `isComposite` records.
    pub fn to_encoding(&self) -> Option<InstrEncoding> {
        if self.is_code_gen_only || self.is_composite {
            return None;
        }
        let enc = self.slot_encoding.as_ref()?;
        let (fixed_mask, fixed_bits) = enc.compute_fixed_bits();
        let mut operand_fields = enc.extract_operand_fields();

        // Populate operand_type using reg_class from inputs/outputs.
        let all_operands: Vec<(&str, &str)> = self.outputs.iter()
            .chain(self.inputs.iter())
            .map(|(cls, name)| (cls.as_str(), name.as_str()))
            .collect();

        for field in &mut operand_fields {
            if let Some((reg_class, _)) = all_operands.iter()
                .find(|(_, name)| *name == field.name)
            {
                field.operand_type = classify_operand_type(reg_class, &field.name);
                if let OperandType::Immediate { signed: true, .. } = &field.operand_type {
                    field.signed = true;
                }
            } else {
                field.operand_type = classify_operand_type("", &field.name);
            }
        }

        let slot = match enc.slot.as_str() {
            "alu" => "alu",
            "lda" => "lda",
            "ldb" => "ldb",
            "st" => "st",
            "mv" => "mv",
            "vec" => "vec",
            "lng" => "lng",
            _ => return None,
        };

        // Structural semantic inference from TableGen attributes.
        let semantic = if self.is_move_imm || self.is_move_reg {
            Some(SemanticOp::Copy)
        } else if self.is_slot_nop {
            Some(SemanticOp::Nop)
        } else {
            infer_semantic_from_structure(
                &self.defs, &self.uses, self.may_load, self.may_store,
                self.has_delay_slot, &self.parents,
            )
        };

        let semantic = refine_branch_semantic(&self.mnemonic, semantic);

        // Cross-validate flags vs inferred semantics (trace-level diagnostics)
        if self.is_call && semantic != Some(SemanticOp::Call) {
            log::trace!("[TBLGEN] {} has isCall=1 but inferred {:?}", self.name, semantic);
        }
        if self.is_return && semantic != Some(SemanticOp::Ret) {
            log::trace!("[TBLGEN] {} has isReturn=1 but inferred {:?}", self.name, semantic);
        }
        if self.is_branch && !matches!(semantic, Some(SemanticOp::Br) | Some(SemanticOp::BrCond)) {
            log::trace!("[TBLGEN] {} has isBranch=1 but inferred {:?}", self.name, semantic);
        }

        let input_order: Vec<String> = self.inputs.iter().map(|(_, n)| n.clone()).collect();
        let output_order: Vec<String> = self.outputs.iter().map(|(_, n)| n.clone()).collect();

        let implicit_regs: Vec<ImplicitReg> = self.defs.iter()
            .filter_map(|r| {
                let reg_num = r.chars().filter(|c| c.is_ascii_digit())
                    .collect::<String>().parse().unwrap_or(0);
                Some(ImplicitReg { reg_class: r.clone(), reg_num, is_use: false })
            })
            .chain(self.uses.iter().filter_map(|r| {
                let reg_num = r.chars().filter(|c| c.is_ascii_digit())
                    .collect::<String>().parse().unwrap_or(0);
                Some(ImplicitReg { reg_class: r.clone(), reg_num, is_use: true })
            }))
            .collect();

        let is_vector = self.mnemonic.starts_with('v') || self.mnemonic.starts_with('V');
        let is_ptr_arithmetic = semantic == Some(SemanticOp::PointerAdd);
        let is_sp_relative = self.uses.iter().any(|u| u == "SP");
        let element_type = infer_element_type(&self.mnemonic);
        let branch_condition = infer_branch_condition(&self.mnemonic, semantic);
        let select_variant = infer_select_variant(&self.mnemonic, semantic);

        Some(InstrEncoding {
            name: self.name.clone(),
            mnemonic: self.mnemonic.clone(),
            slot: slot.to_string(),
            width: enc.width,
            fixed_mask,
            fixed_bits,
            operand_fields,
            semantic,
            may_load: self.may_load,
            may_store: self.may_store,
            input_order,
            output_order,
            implicit_regs,
            addressing_mode: detect_addressing_mode(&self.name),
            mem_width: detect_mem_width(&self.mnemonic),
            has_complete_decoder: self.has_complete_decoder,
            element_type,
            branch_condition,
            is_vector,
            select_variant,
            is_ptr_arithmetic,
            is_sp_relative,
            sched_class: self.itinerary_class.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fixed_bits() {
        let enc = SlotEncoding {
            slot: "alu".to_string(),
            width: 5,
            parts: vec![
                EncodingBit::FieldBit { field: "f".to_string(), bit: 1 },
                EncodingBit::FieldBit { field: "f".to_string(), bit: 0 },
                EncodingBit::One,
                EncodingBit::Zero,
                EncodingBit::One,
            ],
        };

        let (mask, bits) = enc.compute_fixed_bits();
        assert_eq!(mask, 0b00111);
        assert_eq!(bits, 0b00101);
    }
}
