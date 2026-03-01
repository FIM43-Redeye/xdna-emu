//! Parser for llvm-tblgen --print-records output.
//!
//! This module parses the output of `llvm-tblgen --print-records` which gives
//! us fully resolved instruction definitions with all inheritance, template
//! substitution, and field assignments applied.
//!
//! # Why This Approach?
//!
//! Parsing raw .td files with regex is fragile - we miss mixin inheritance,
//! template parameters, and computed field assignments. Using tblgen directly
//! gives us the ground truth: exactly what bits encode what.
//!
//! # Format
//!
//! ```text
//! def ADD_add_r_ri {    // InstructionEncoding Instruction InstFormat ...
//!   string DecoderNamespace = "Alu";
//!   dag InOperandList = (ins eR:$mRx0, simm7:$c7s);
//!   dag OutOperandList = (outs eR:$mRx);
//!   string AsmString = "add    $mRx, $mRx0, $c7s";
//!   bits<20> alu = { mRx0{4}, mRx0{3}, ..., 1, 1, 0 };
//!   list<Register> Defs = [srCarry];
//!   list<Register> Uses = [];
//!   bit mayLoad = 0;
//!   bit mayStore = 0;
//! }
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use std::process::Command;
//!
//! let output = Command::new("llvm-tblgen")
//!     .arg("--print-records")
//!     .arg("AIE2.td")
//!     .output()?;
//!
//! let records = parse_tblgen_records(&String::from_utf8_lossy(&output.stdout));
//! ```

use std::collections::HashMap;

use super::resolver::{FieldFragment, InstrEncoding, OperandField, OperandType, classify_operand_type, detect_addressing_mode, detect_mem_width, infer_branch_condition, infer_element_type, infer_select_variant, infer_semantic_from_mnemonic, infer_semantic_from_structure};
use super::types::ImplicitReg;

/// A parsed instruction record from tblgen output.
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
    /// Itinerary class name from `InstrItinClass Itinerary = II_ADD;`.
    /// Links this instruction to its pipeline timing data in the scheduling model.
    /// None only if the field is absent; "NoItinerary" means explicitly unscheduled.
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
    ///
    /// Returns (fixed_mask, fixed_bits) where:
    /// - fixed_mask has 1s for literal bits (0 or 1)
    /// - fixed_bits has the actual values for those bits
    pub fn compute_fixed_bits(&self) -> (u64, u64) {
        let mut mask: u64 = 0;
        let mut bits: u64 = 0;

        // Parts are MSB-first, so part[0] is the highest bit
        for (i, part) in self.parts.iter().enumerate() {
            let bit_pos = self.width as usize - 1 - i;
            match part {
                EncodingBit::Zero => {
                    mask |= 1 << bit_pos;
                    // bits already 0
                }
                EncodingBit::One => {
                    mask |= 1 << bit_pos;
                    bits |= 1 << bit_pos;
                }
                EncodingBit::DontCare | EncodingBit::FieldBit { .. } => {
                    // Not fixed
                }
            }
        }

        (mask, bits)
    }

    /// Extract operand fields from the encoding, handling split fields.
    ///
    /// Each `FieldBit { field, bit }` in the encoding tells us exactly which
    /// logical bit of the operand maps to which instruction bit. For contiguous
    /// fields (all bits adjacent with sequential target_bit values), we create
    /// a simple OperandField. For split fields (like MOV_mv_cg's immediate
    /// where `i{9:1}` is at positions 14-6 and `i{0}` at position 3), we
    /// create FieldFragments for correct reassembly.
    pub fn extract_operand_fields(&self) -> Vec<OperandField> {
        // Collect all (inst_bit, target_bit) pairs for each field name
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
                // Sort by target_bit ascending for easier analysis
                bit_pairs.sort_by_key(|&(_, target)| target);

                let logical_width = bit_pairs.iter()
                    .map(|&(_, target)| target)
                    .max().unwrap_or(0) + 1;
                let low_inst_bit = bit_pairs.iter()
                    .map(|&(inst, _)| inst)
                    .min().unwrap_or(0);

                // Check if bits form a single contiguous range in instruction
                // space with sequential target bits. A contiguous field has:
                // - target bits 0, 1, 2, ... (sequential)
                // - inst bits that differ by exactly 1 per step
                let is_contiguous = bit_pairs.windows(2).all(|w| {
                    let (inst_a, tgt_a) = w[0];
                    let (inst_b, tgt_b) = w[1];
                    tgt_b == tgt_a + 1 && inst_b == inst_a + 1
                });

                if is_contiguous {
                    OperandField::new(name, low_inst_bit, logical_width)
                } else {
                    // Split field: build fragments from contiguous runs
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
///
/// Groups consecutive runs of bits (where both inst_bit and target_bit
/// increment by 1) into single fragments. For example:
/// - [(3, 0)] and [(6, 1), (7, 2), ..., (14, 9)] become two fragments:
///   Fragment { inst_bit: 3, width: 1, target_bit: 0 } and
///   Fragment { inst_bit: 6, width: 9, target_bit: 1 }
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
            true // end of sequence
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

/// Parse tblgen --print-records output.
///
/// Extracts all instruction definitions (those inheriting from AIE2Inst)
/// with their encodings and operands.
pub fn parse_tblgen_records(content: &str) -> Vec<InstrRecord> {
    let mut records = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        // Look for "def NAME {"
        if lines[i].starts_with("def ") && lines[i].contains('{') {
            if let Some(record) = parse_single_record(&lines, &mut i) {
                // Only include actual instructions (not classes, registers, etc.)
                if record.parents.iter().any(|p| p == "AIE2Inst" || p == "AIE2SlotInst") {
                    records.push(record);
                }
            }
        } else {
            i += 1;
        }
    }

    records
}

/// Parse a single def block.
fn parse_single_record(lines: &[&str], i: &mut usize) -> Option<InstrRecord> {
    let first_line = lines[*i];

    // Parse "def NAME {	// Parent1 Parent2 ..."
    let (name, parents) = parse_def_line(first_line)?;

    // Skip to next line
    *i += 1;

    let mut record = InstrRecord {
        name,
        parents,
        decoder_namespace: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        mnemonic: String::new(),
        asm_string: String::new(),
        slot_encoding: None,
        defs: Vec::new(),
        uses: Vec::new(),
        may_load: false,
        may_store: false,
        has_side_effects: false,
        has_complete_decoder: true, // Default to true; set false if field says 0
        has_delay_slot: false,
        is_code_gen_only: false,
        itinerary_class: None,
    };

    // Parse fields until closing brace
    while *i < lines.len() {
        let line = lines[*i].trim();
        *i += 1;

        if line == "}" || line.starts_with('}') {
            break;
        }

        // Parse various field types
        if let Some(ns) = parse_string_field(line, "DecoderNamespace") {
            record.decoder_namespace = ns;
        } else if let Some(asm) = parse_string_field(line, "AsmString") {
            record.asm_string = asm.clone();
            record.mnemonic = extract_mnemonic(&asm);
        } else if line.starts_with("dag InOperandList = ") {
            record.inputs = parse_dag_operands(line);
        } else if line.starts_with("dag OutOperandList = ") {
            record.outputs = parse_dag_operands(line);
        } else if line.starts_with("list<Register> Defs = ") {
            record.defs = parse_register_list(line);
        } else if line.starts_with("list<Register> Uses = ") {
            record.uses = parse_register_list(line);
        } else if line.starts_with("bit mayLoad = ") {
            record.may_load = line.contains("= 1");
        } else if line.starts_with("bit mayStore = ") {
            record.may_store = line.contains("= 1");
        } else if line.starts_with("bit hasSideEffects = ") {
            record.has_side_effects = line.contains("= 1");
        } else if line.starts_with("bit hasCompleteDecoder = ") {
            record.has_complete_decoder = line.contains("= 1");
        } else if line.starts_with("bit hasDelaySlot = ") {
            record.has_delay_slot = line.contains("= 1");
        } else if line.starts_with("bit isCodeGenOnly = ") {
            record.is_code_gen_only = line.contains("= 1");
        } else if line.starts_with("InstrItinClass Itinerary = ") {
            let val = line
                .strip_prefix("InstrItinClass Itinerary = ")
                .unwrap_or("")
                .trim_end_matches(';')
                .to_string();
            if val != "NoItinerary" {
                record.itinerary_class = Some(val);
            }
        } else if let Some(enc) = parse_slot_encoding(line) {
            record.slot_encoding = Some(enc);
        }
    }

    Some(record)
}

/// Parse "def NAME {    // Parent1 Parent2 ..."
fn parse_def_line(line: &str) -> Option<(String, Vec<String>)> {
    let line = line.strip_prefix("def ")?;
    let brace_pos = line.find('{')?;
    let name = line[..brace_pos].trim().to_string();

    let parents = if let Some(comment_pos) = line.find("//") {
        line[comment_pos + 2..]
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    } else {
        Vec::new()
    };

    Some((name, parents))
}

/// Parse string field like: string Name = "value";
fn parse_string_field(line: &str, field_name: &str) -> Option<String> {
    let prefix = format!("string {} = \"", field_name);
    if !line.starts_with(&prefix) {
        return None;
    }

    let rest = &line[prefix.len()..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract mnemonic from AsmString (e.g., "add    $mRx, ..." -> "add")
fn extract_mnemonic(asm: &str) -> String {
    // Mnemonic is everything before first whitespace or $
    let end = asm.find(|c: char| c.is_whitespace() || c == '$')
        .unwrap_or(asm.len());
    asm[..end].to_string()
}

/// Parse dag operand list: (ins/outs Type:$name, Type:$name, ...)
fn parse_dag_operands(line: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();

    // Find content between parens
    let start = line.find('(').map(|p| p + 1).unwrap_or(0);
    let end = line.rfind(')').unwrap_or(line.len());
    let content = &line[start..end];

    // Skip the "ins" or "outs" keyword
    let content = if content.starts_with("ins ") {
        &content[4..]
    } else if content.starts_with("outs ") {
        &content[5..]
    } else {
        content
    };

    // Parse each "Type:$name" pair
    for part in content.split(',') {
        let part = part.trim();
        if let Some(colon_pos) = part.find(':') {
            let typ = part[..colon_pos].trim().to_string();
            let name = part[colon_pos + 1..].trim().trim_start_matches('$').to_string();
            result.push((typ, name));
        }
    }

    result
}

/// Parse register list: list<Register> Defs = [Reg1, Reg2, ...];
fn parse_register_list(line: &str) -> Vec<String> {
    let start = line.find('[').map(|p| p + 1).unwrap_or(0);
    let end = line.find(']').unwrap_or(line.len());
    let content = &line[start..end];

    content
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse slot encoding: bits<20> alu = { mRx0{4}, mRx0{3}, ..., 1, 1, 0 };
fn parse_slot_encoding(line: &str) -> Option<SlotEncoding> {
    // Match "bits<N> SLOT = { ... };"
    let slot_names = ["alu", "lda", "ldb", "st", "mv", "vec", "lng"];

    for slot in &slot_names {
        let prefix = format!("bits<");
        if !line.starts_with(&prefix) {
            continue;
        }

        // Extract width
        let gt_pos = line.find('>')?;
        let width: u8 = line[5..gt_pos].parse().ok()?;

        // Check slot name
        let after_gt = line[gt_pos + 1..].trim();
        if !after_gt.starts_with(*slot) {
            continue;
        }

        // Find the braces content
        let open = line.find('{')?;
        let close = line.rfind('}')?;
        if close <= open {
            continue;
        }

        let bits_content = &line[open + 1..close];
        let parts = parse_encoding_bits(bits_content);

        return Some(SlotEncoding {
            slot: slot.to_string(),
            width,
            parts,
        });
    }

    None
}

/// Parse encoding bits content: "mRx0{4}, mRx0{3}, ..., 1, 1, 0"
fn parse_encoding_bits(content: &str) -> Vec<EncodingBit> {
    let mut parts = Vec::new();

    for part in content.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        let bit = if part == "0" {
            EncodingBit::Zero
        } else if part == "1" {
            EncodingBit::One
        } else if part == "?" || part.starts_with("dontcare") {
            EncodingBit::DontCare
        } else if let Some(brace_pos) = part.find('{') {
            // Field bit reference: mRx0{4}
            let field = part[..brace_pos].to_string();
            let bit_str = &part[brace_pos + 1..part.len() - 1];
            let bit: u8 = bit_str.parse().unwrap_or(0);
            EncodingBit::FieldBit { field, bit }
        } else {
            // Unknown - treat as field reference with bit 0
            EncodingBit::FieldBit { field: part.to_string(), bit: 0 }
        };

        parts.push(bit);
    }

    parts
}

impl InstrRecord {
    /// Convert to InstrEncoding for the decoder.
    ///
    /// Returns None for isCodeGenOnly instructions, which exist only for the
    /// compiler's instruction selection and must not participate in decoding.
    /// Including them causes ambiguous matches (e.g., MOV_OR vs OR have identical
    /// fixed bits but different operand counts, leading to lost source operands).
    pub fn to_encoding(&self) -> Option<InstrEncoding> {
        if self.is_code_gen_only {
            return None;
        }
        let enc = self.slot_encoding.as_ref()?;
        let (fixed_mask, fixed_bits) = enc.compute_fixed_bits();
        let mut operand_fields = enc.extract_operand_fields();

        // Populate operand_type using reg_class from inputs/outputs.
        // The encoding extraction creates fields with OperandType::Unknown;
        // we resolve them here using the same classifier as the resolver path.
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
                // No matching operand def -- use field-name fallback
                field.operand_type = classify_operand_type("", &field.name);
            }
        }

        // Map slot name to canonical form
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

        // Three-tier semantic inference: structure first, mnemonic fallback.
        // (Pattern-based semantics are only available on the regex-parser path.)
        let semantic = infer_semantic_from_structure(
                &self.defs, &self.uses, self.may_load, self.may_store,
                self.has_delay_slot, &self.parents,
            )
            .or_else(|| infer_semantic_from_mnemonic(&self.mnemonic));

        // Build input/output ordering
        let input_order: Vec<String> = self.inputs.iter().map(|(_, n)| n.clone()).collect();
        let output_order: Vec<String> = self.outputs.iter().map(|(_, n)| n.clone()).collect();

        // Convert implicit registers
        // Note: For now, we just use the register name as the class and extract number if possible
        let implicit_regs: Vec<ImplicitReg> = self.defs.iter()
            .filter_map(|r| {
                // Try to extract register number from names like "r27" or "srCarry"
                let reg_num = r.chars().filter(|c| c.is_ascii_digit())
                    .collect::<String>().parse().unwrap_or(0);
                Some(ImplicitReg {
                    reg_class: r.clone(),
                    reg_num,
                    is_use: false,  // defs are writes, not uses
                })
            })
            .chain(self.uses.iter().filter_map(|r| {
                let reg_num = r.chars().filter(|c| c.is_ascii_digit())
                    .collect::<String>().parse().unwrap_or(0);
                Some(ImplicitReg {
                    reg_class: r.clone(),
                    reg_num,
                    is_use: true,
                })
            }))
            .collect();

        // Pre-resolve metadata from mnemonic
        let is_vector = self.mnemonic.starts_with('v') || self.mnemonic.starts_with('V');
        let is_ptr_arithmetic = self.mnemonic.starts_with("padd");
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

// ============================================================================
// Phase 1: Extended Parsing for Scheduling, Register, and Format Data
// ============================================================================

use super::types::{
    CompositeFormatDef, ItineraryInfo, PipelineStage, ProcessorModel,
    RegisterClassDef, RegisterDef, RegisterModel,
};

/// Parse the AIE2SchedModel record from `--print-records` output.
///
/// Looks for a record whose parent chain includes "SchedMachineModel" and
/// extracts the key scheduling parameters (LoadLatency, MispredictPenalty, etc.).
pub fn parse_processor_model(content: &str) -> Option<ProcessorModel> {
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].starts_with("def ") && lines[i].contains("SchedMachineModel") {
            // Found a SchedMachineModel record
            i += 1;
            let mut load_latency = 5u8;
            let mut high_latency = 37u8;
            let mut mispredict_penalty = 4u8;
            let mut issue_width = 1000u16;
            let mut itinerary_name = String::new();

            while i < lines.len() {
                let line = lines[i].trim();
                if line == "}" || line.starts_with('}') {
                    break;
                }

                if let Some(v) = parse_int_field(line, "LoadLatency") {
                    load_latency = v as u8;
                } else if let Some(v) = parse_int_field(line, "HighLatency") {
                    high_latency = v as u8;
                } else if let Some(v) = parse_int_field(line, "MispredictPenalty") {
                    mispredict_penalty = v as u8;
                } else if let Some(v) = parse_int_field(line, "IssueWidth") {
                    issue_width = v as u16;
                } else if line.starts_with("ProcessorItineraries Itineraries = ") {
                    itinerary_name = line
                        .strip_prefix("ProcessorItineraries Itineraries = ")
                        .unwrap_or("")
                        .trim_end_matches(';')
                        .to_string();
                }
                i += 1;
            }

            return Some(ProcessorModel {
                load_latency,
                high_latency,
                mispredict_penalty,
                issue_width,
                itinerary_name,
            });
        }
        i += 1;
    }
    None
}

/// Parse all InstrItinData records from `--print-records` output.
///
/// InstrItinData records are anonymous (e.g., `def anonymous_8663`) and
/// reference InstrItinClass and InstrStage records. We resolve these
/// references to build a complete picture of each itinerary class.
///
/// Returns a map from itinerary class name (e.g., "II_ADD") to its info.
pub fn parse_itinerary_data(content: &str) -> HashMap<String, ItineraryInfo> {
    let lines: Vec<&str> = content.lines().collect();

    // First pass: collect InstrStage records (anonymous_XXXX -> stage data)
    let mut stages_map: HashMap<String, PipelineStage> = HashMap::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("def ") && lines[i].contains("InstrStage") {
            if let Some((name, _)) = parse_def_line(lines[i]) {
                let stage = parse_instr_stage_record(&lines, &mut i);
                stages_map.insert(name, stage);
                continue;
            }
        }
        i += 1;
    }

    // Second pass: collect InstrItinData records
    let mut result: HashMap<String, ItineraryInfo> = HashMap::new();
    i = 0;
    while i < lines.len() {
        if lines[i].starts_with("def ") && lines[i].contains("InstrItinData") {
            i += 1;
            let mut class_name = String::new();
            let mut stage_refs: Vec<String> = Vec::new();
            let mut operand_cycles: Vec<u8> = Vec::new();
            let mut bypasses: Vec<String> = Vec::new();

            while i < lines.len() {
                let line = lines[i].trim();
                if line == "}" || line.starts_with('}') {
                    break;
                }

                if line.starts_with("InstrItinClass TheClass = ") {
                    class_name = line
                        .strip_prefix("InstrItinClass TheClass = ")
                        .unwrap_or("")
                        .trim_end_matches(';')
                        .to_string();
                } else if line.starts_with("list<InstrStage> Stages = ") {
                    stage_refs = parse_reference_list(line);
                } else if line.starts_with("list<int> OperandCycles = ") {
                    operand_cycles = parse_int_list(line);
                } else if line.starts_with("list<Bypass> Bypasses = ") {
                    bypasses = parse_reference_list(line);
                }
                i += 1;
            }

            if !class_name.is_empty() {
                // Resolve stage references
                let stages: Vec<PipelineStage> = stage_refs
                    .iter()
                    .filter_map(|name| stages_map.get(name).cloned())
                    .collect();

                // Compute total latency from stage cycles
                let total_latency = stages.iter().map(|s| s.cycles).sum::<u8>();

                result.insert(class_name.clone(), ItineraryInfo {
                    class_name,
                    total_latency,
                    operand_cycles,
                    stages,
                    bypasses,
                });
            }
        }
        i += 1;
    }

    result
}

/// Parse a single InstrStage record body.
fn parse_instr_stage_record(lines: &[&str], i: &mut usize) -> PipelineStage {
    *i += 1;
    let mut cycles = 0u8;
    let mut units: Vec<String> = Vec::new();
    let mut time_inc = 1i8;

    while *i < lines.len() {
        let line = lines[*i].trim();
        if line == "}" || line.starts_with('}') {
            *i += 1;
            break;
        }

        if let Some(v) = parse_int_field(line, "Cycles") {
            cycles = v as u8;
        } else if line.starts_with("list<FuncUnit> Units = ") {
            units = parse_reference_list(line);
        } else if let Some(v) = parse_signed_int_field(line, "TimeInc") {
            time_inc = v as i8;
        }
        *i += 1;
    }

    PipelineStage { cycles, units, time_inc }
}

/// Parse all register definitions (records with HWEncoding field).
///
/// Identifies registers by checking for the `bits<16> HWEncoding` field
/// in any record within the "AIE2" namespace. Extracts the register name,
/// HWEncoding value, and parent class chain.
pub fn parse_register_defs(content: &str) -> Vec<RegisterDef> {
    let lines: Vec<&str> = content.lines().collect();
    let mut regs = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        // Look for register records (not register classes)
        if lines[i].starts_with("def ") && lines[i].contains("Register") && !lines[i].contains("RegisterClass") {
            if let Some((name, parents)) = parse_def_line(lines[i]) {
                i += 1;
                let mut namespace = String::new();
                let mut hw_encoding: Option<u16> = None;

                while i < lines.len() {
                    let line = lines[i].trim();
                    if line == "}" || line.starts_with('}') {
                        break;
                    }

                    if let Some(ns) = parse_string_field(line, "Namespace") {
                        namespace = ns;
                    } else if line.starts_with("bits<16> HWEncoding = ") {
                        hw_encoding = Some(parse_bits16(line));
                    }
                    i += 1;
                }

                // Only include AIE2 registers with HWEncoding
                if namespace == "AIE2" && hw_encoding.is_some() {
                    regs.push(RegisterDef {
                        name,
                        hw_encoding: hw_encoding.unwrap(),
                        parents,
                    });
                }
            }
        }
        i += 1;
    }

    regs
}

/// Parse `bits<16> HWEncoding = { bit15, bit14, ..., bit0 };`
///
/// The format uses MSB-first: `{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1 }`
/// means bits 5,2,1,0 are set = 0b100111 = 39.
fn parse_bits16(line: &str) -> u16 {
    let open = match line.find('{') {
        Some(p) => p + 1,
        None => return 0,
    };
    let close = match line.rfind('}') {
        Some(p) => p,
        None => return 0,
    };

    let bits_str = &line[open..close];
    let mut value: u16 = 0;

    for (i, part) in bits_str.split(',').enumerate() {
        let part = part.trim();
        if part == "1" {
            // MSB first: bit index 0 is bit 15, index 15 is bit 0
            value |= 1 << (15 - i);
        }
    }

    value
}

/// Parse all register class definitions.
///
/// Register classes are identified by "RegisterClass" in their parent chain
/// and have a `MemberList = (add reg1, reg2, ...)` field.
pub fn parse_register_classes(content: &str) -> Vec<RegisterClassDef> {
    let lines: Vec<&str> = content.lines().collect();
    let mut classes = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].starts_with("def ") && lines[i].contains("RegisterClass") {
            if let Some((name, parents)) = parse_def_line(lines[i]) {
                i += 1;
                let mut namespace = String::new();
                let mut members: Vec<String> = Vec::new();
                let mut alignment = 0u16;

                while i < lines.len() {
                    let line = lines[i].trim();
                    if line == "}" || line.starts_with('}') {
                        break;
                    }

                    if let Some(ns) = parse_string_field(line, "Namespace") {
                        namespace = ns;
                    } else if line.starts_with("dag MemberList = ") {
                        members = parse_dag_add_list(line);
                    } else if let Some(v) = parse_int_field(line, "Alignment") {
                        alignment = v as u16;
                    }
                    i += 1;
                }

                // Only include AIE2 register classes with members
                if namespace == "AIE2" && !members.is_empty() {
                    classes.push(RegisterClassDef {
                        name,
                        members,
                        alignment,
                        parents,
                    });
                }
            }
        }
        i += 1;
    }

    classes
}

/// Parse `dag MemberList = (add reg1, reg2, ..., regN);` to extract member names.
fn parse_dag_add_list(line: &str) -> Vec<String> {
    let start = match line.find("(add ") {
        Some(p) => p + 5,
        None => return Vec::new(),
    };
    let end = match line.rfind(')') {
        Some(p) => p,
        None => return Vec::new(),
    };

    line[start..end]
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Build a complete RegisterModel from the parsed data.
pub fn parse_register_model(content: &str) -> RegisterModel {
    let reg_defs = parse_register_defs(content);
    let class_defs = parse_register_classes(content);

    let registers: HashMap<String, RegisterDef> = reg_defs
        .into_iter()
        .map(|r| (r.name.clone(), r))
        .collect();

    let classes: HashMap<String, RegisterClassDef> = class_defs
        .into_iter()
        .map(|c| (c.name.clone(), c))
        .collect();

    RegisterModel { registers, classes }
}

/// Parse composite format definitions (records with `isComposite = 1`).
///
/// These define the VLIW bundle layouts: which slots are present and their
/// bit widths. Extracted from the `InOperandList` (slot fields) and `Size`.
pub fn parse_composite_formats(content: &str) -> Vec<CompositeFormatDef> {
    let lines: Vec<&str> = content.lines().collect();
    let mut formats = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].starts_with("def ") && lines[i].contains("AIE2CompositeInst") {
            if let Some((name, _)) = parse_def_line(lines[i]) {
                i += 1;
                let mut total_bytes = 0u8;
                let mut is_composite = false;
                let mut slots: Vec<(String, u16)> = Vec::new();

                while i < lines.len() {
                    let line = lines[i].trim();
                    if line == "}" || line.starts_with('}') {
                        break;
                    }

                    if let Some(v) = parse_int_field(line, "Size") {
                        total_bytes = v as u8;
                    } else if line == "bit isComposite = 1;" || line == "bit isComposite = 1" {
                        is_composite = true;
                    } else if line.starts_with("dag InOperandList = ") {
                        // Parse slot operands: (ins ldb_slot:$ldb, lda_slot:$lda, ...)
                        let operands = parse_dag_operands(line);
                        for (slot_type, slot_name) in &operands {
                            // Extract bit width from corresponding bits<N> field
                            // The slot_name should match a field defined in the record
                            let _ = slot_type; // We'll use the bits<N> field instead
                            slots.push((slot_name.clone(), 0)); // Width filled below
                        }
                    } else if line.starts_with("bits<") && !line.contains("Inst =")
                        && !line.contains("TSFlags") && !line.contains("HWEncoding")
                        && !line.contains("dontcare")
                    {
                        // Parse slot field widths: bits<16> ldb = { ... };
                        if let Some(gt_pos) = line.find('>') {
                            let width: u16 = line[5..gt_pos].parse().unwrap_or(0);
                            let after = line[gt_pos + 1..].trim();
                            if let Some(eq_pos) = after.find(" = ") {
                                let field_name = after[..eq_pos].trim();
                                // Update slot width if this field was in InOperandList
                                for slot in &mut slots {
                                    if slot.0 == field_name {
                                        slot.1 = width;
                                    }
                                }
                            }
                        }
                    }
                    i += 1;
                }

                if is_composite && total_bytes > 0 {
                    // Remove slots with zero width (intermediate fields like alu_mv)
                    slots.retain(|s| s.1 > 0);

                    formats.push(CompositeFormatDef {
                        name,
                        total_bytes,
                        total_bits: total_bytes as u16 * 8,
                        slots,
                    });
                }
            }
        }
        i += 1;
    }

    formats
}

// ============================================================================
// Field parsing helpers
// ============================================================================

/// Parse `int FieldName = VALUE;` to extract an integer value.
fn parse_int_field(line: &str, field_name: &str) -> Option<i64> {
    let prefix = format!("int {} = ", field_name);
    if !line.starts_with(&prefix) {
        return None;
    }
    let value_str = line[prefix.len()..].trim_end_matches(';').trim();
    value_str.parse().ok()
}

/// Parse `int FieldName = VALUE;` that may be negative.
fn parse_signed_int_field(line: &str, field_name: &str) -> Option<i64> {
    parse_int_field(line, field_name)
}

/// Parse a reference list: `list<Type> Field = [ref1, ref2, ...];`
fn parse_reference_list(line: &str) -> Vec<String> {
    let start = match line.find('[') {
        Some(p) => p + 1,
        None => return Vec::new(),
    };
    let end = match line.find(']') {
        Some(p) => p,
        None => return Vec::new(),
    };

    line[start..end]
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse an integer list: `list<int> Field = [1, 2, 3];`
fn parse_int_list(line: &str) -> Vec<u8> {
    let start = match line.find('[') {
        Some(p) => p + 1,
        None => return Vec::new(),
    };
    let end = match line.find(']') {
        Some(p) => p,
        None => return Vec::new(),
    };

    line[start..end]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_def_line() {
        let line = "def ADD_add_r_ri {\t// InstructionEncoding Instruction InstFormat AIEBaseInst AIE2Inst";
        let (name, parents) = parse_def_line(line).unwrap();
        assert_eq!(name, "ADD_add_r_ri");
        assert!(parents.contains(&"AIE2Inst".to_string()));
    }

    #[test]
    fn test_parse_dag_operands() {
        let line = "  dag InOperandList = (ins eR:$mRx0, simm7:$c7s);";
        let ops = parse_dag_operands(line);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0], ("eR".to_string(), "mRx0".to_string()));
        assert_eq!(ops[1], ("simm7".to_string(), "c7s".to_string()));
    }

    #[test]
    fn test_parse_encoding_bits() {
        let content = "mRx0{4}, mRx0{3}, mRx0{2}, 1, 0";
        let parts = parse_encoding_bits(content);
        assert_eq!(parts.len(), 5);
        assert!(matches!(parts[3], EncodingBit::One));
        assert!(matches!(parts[4], EncodingBit::Zero));
    }

    #[test]
    fn test_compute_fixed_bits() {
        // Simple 5-bit encoding: field{1}, field{0}, 1, 0, 1
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
        // Bits 2, 1, 0 are fixed (0b111 = 7)
        assert_eq!(mask, 0b00111);
        // Values: 1, 0, 1 = 0b101 = 5
        assert_eq!(bits, 0b00101);
    }
}
