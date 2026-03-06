//! Native TableGen integration via the `tblgen` crate.
//!
//! Parses .td files in-process using LLVM's own TableGen library (linked via
//! the `tblgen` Rust crate). This replaces the subprocess + text parsing
//! approach with direct access to fully-resolved LLVM records.
//!
//! # What this solves
//!
//! 1. **Composite format extraction**: Multi-level field hierarchies
//!    (Inst -> instr80 -> inst_lda_alu_vec -> {lda, alu, vec}) are fully
//!    resolved by LLVM. We read leaf-level VarBitInit references directly.
//!
//! 2. **Instruction coverage**: Access ALL instruction records (607+), not
//!    just the ~210 that LLVM's `-gen-disassembler` selects.
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::tablegen::native::{load_composite_formats, load_instruction_records};
//!
//! let td_file = Path::new("../llvm-aie/llvm/lib/Target/AIE/AIE2.td");
//! let includes = &[Path::new("../llvm-aie/llvm/include")];
//! let formats = load_composite_formats(td_file, includes)?;
//! let records = load_instruction_records(td_file, includes)?;
//! ```


use std::path::Path;


use tblgen::TableGenParser;


use super::tblgen_records::{EncodingBit, InstrRecord, SlotEncoding};

use super::types::{
    CompositeFormatDef, ItineraryInfo, PipelineStage, ProcessorModel,
    RegisterClassDef, RegisterDef, RegisterModel, SemanticOp, SlotBitMap,
};

use std::collections::HashMap;

/// Extract composite VLIW format definitions from TableGen records.
///
/// Reads all records that are subclasses of "AIE2CompositeInst" and extracts
/// their Inst field bit-by-bit. For each bit:
/// - VarBitInit (e.g., `lda{17}`) -> slot mapping entry
/// - BitInit (0 or 1) -> fixed discriminator bit
/// - Anything else -> don't-care bit
///
/// Returns fully populated CompositeFormatDef with correct slot_maps,
/// fixed_mask, and fixed_value.

pub fn load_composite_formats(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<Vec<CompositeFormatDef>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;

    let mut formats = Vec::new();

    // Iterate all records that derive from AIE2CompositeInst.
    // The class name varies by architecture; try common names.
    for class_name in &["AIE2CompositeInst"] {
        for record in keeper.all_derived_definitions(class_name) {
            if let Some(fmt) = extract_composite_format(&record) {
                formats.push(fmt);
            }
        }
    }

    log::info!(
        "Native TableGen: extracted {} composite format definitions",
        formats.len()
    );
    Ok(formats)
}

/// Extract instruction records from TableGen for all slot instructions.
///
/// Reads all records that are subclasses of "Instruction" and have encoding
/// fields (Inst bits, Size, DecoderNamespace). Produces InstrRecord values
/// compatible with the existing `to_encoding()` pipeline.

pub fn load_instruction_records(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<Vec<InstrRecord>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;

    let mut records = Vec::new();

    for record in keeper.all_derived_definitions("Instruction") {
        if let Some(instr) = extract_instruction_record(&record) {
            records.push(instr);
        }
    }

    log::info!(
        "Native TableGen: extracted {} instruction records",
        records.len()
    );
    Ok(records)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Parse a .td file and return the RecordKeeper.

fn parse_td_file<'a>(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<tblgen::RecordKeeper<'a>, String> {
    let td_str = td_file.to_str().ok_or("non-UTF8 .td path")?;

    let mut parser = TableGenParser::new().add_source_file(td_str);
    for inc in include_paths {
        let inc_str = inc.to_str().ok_or("non-UTF8 include path")?;
        parser = parser.add_include_directory(inc_str);
    }

    parser.parse().map_err(|e| format!("TableGen parse failed: {}", e))
}

/// Extract a CompositeFormatDef from a single TableGen record.

fn extract_composite_format(record: &tblgen::Record<'_>) -> Option<CompositeFormatDef> {
    let name = record.name().ok()?.to_string();

    // Get the Inst field (bits<N>)
    let inst_val = record.value("Inst").ok()?;
    let bits_init = inst_val.init.as_bits().ok()?;
    let total_bits = bits_init.num_bits();

    if total_bits == 0 {
        return None;
    }

    let total_bytes = (total_bits as u8 + 7) / 8;

    // Walk each bit in the Inst field
    let mut fixed_mask: u128 = 0;
    let mut fixed_value: u128 = 0;
    // slot_name -> Vec<(var_bit_index, word_bit_position)>
    let mut slot_bits: std::collections::HashMap<String, Vec<(usize, usize)>> =
        std::collections::HashMap::new();

    for i in 0..total_bits {
        let bit = match bits_init.bit(i) {
            Some(b) => b,
            None => continue,
        };

        if let Some((var_name, var_bit)) = bit.as_var_bit() {
            // Variable reference: record which slot field this bit belongs to
            slot_bits
                .entry(var_name.to_string())
                .or_default()
                .push((var_bit, i));
        } else if let Some(val) = bit.as_literal() {
            // Fixed 0 or 1: part of the discriminator
            fixed_mask |= 1u128 << i;
            if val {
                fixed_value |= 1u128 << i;
            }
        }
        // else: unresolved/don't-care bit, skip
    }

    // Build SlotBitMap entries from the collected bit positions.
    // Each slot's bits should form a contiguous range starting from some offset.
    let mut slot_maps = Vec::new();
    let mut slots = Vec::new();

    // Sort slot names for deterministic output
    let mut slot_names: Vec<String> = slot_bits.keys().cloned().collect();
    slot_names.sort();

    for slot_name in &slot_names {
        let bits = &slot_bits[slot_name];
        if bits.is_empty() {
            continue;
        }

        // Find the range of word positions this slot occupies
        let min_word_pos = bits.iter().map(|&(_, wp)| wp).min().unwrap();
        let max_word_pos = bits.iter().map(|&(_, wp)| wp).max().unwrap();
        let width = (max_word_pos - min_word_pos + 1) as u8;

        // Verify the mapping is contiguous and correctly ordered:
        // var_bit N should map to word_pos (min_word_pos + N)
        let contiguous = bits.iter().all(|&(vb, wp)| wp == min_word_pos + vb);

        if !contiguous {
            log::warn!(
                "Native TableGen: non-contiguous slot mapping for {} in {}, \
                 {} bits across positions {}-{}",
                slot_name, name, bits.len(), min_word_pos, max_word_pos
            );
        }

        slot_maps.push(SlotBitMap {
            slot_name: slot_name.clone(),
            width,
            start_bit: min_word_pos as u8,
        });
        slots.push((slot_name.clone(), width as u16));
    }

    Some(CompositeFormatDef {
        name,
        total_bytes,
        total_bits: total_bits as u16,
        slots,
        fixed_mask,
        fixed_value,
        slot_maps,
    })
}

/// Extract an InstrRecord from a single TableGen Instruction record.
///
/// Filters out composite instructions and codegen-only pseudo instructions.
/// Extracts the slot encoding from the Inst field using VarBitInit resolution.

fn extract_instruction_record(record: &tblgen::Record<'_>) -> Option<InstrRecord> {
    let name = record.name().ok()?.to_string();

    // Filter: skip composite instructions
    if record.subclass_of("AIE2CompositeInst") {
        return None;
    }

    // Get isCodeGenOnly field (bit type in TableGen)
    let is_code_gen_only = record
        .bit_value("isCodeGenOnly")
        .or_else(|_| record.int_value("isCodeGenOnly").map(|v| v != 0))
        .unwrap_or(false);

    // Get isComposite from the record's parent classes or fields
    let is_composite = record.subclass_of("AIE2CompositeInst");

    // Get DecoderNamespace (= slot name)
    let decoder_namespace = record
        .string_value("DecoderNamespace")
        .ok()?;
    let decoder_namespace = decoder_namespace.trim().to_string();

    if decoder_namespace.is_empty() {
        return None;
    }

    // Get AsmString for mnemonic extraction
    let asm_string = record
        .string_value("AsmString")
        .unwrap_or_default();
    let mnemonic = asm_string
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches('.')
        .to_string();

    // Get the Inst field (bits<N> encoding)
    let inst_val = record.value("Inst").ok()?;
    let bits_init = inst_val.init.as_bits().ok()?;
    let width = bits_init.num_bits();

    if width == 0 {
        return None;
    }

    // Extract encoding bits (MSB-first order for compatibility with existing pipeline)
    let mut parts = Vec::with_capacity(width);
    for i in (0..width).rev() {
        let bit = match bits_init.bit(i) {
            Some(b) => b,
            None => {
                parts.push(EncodingBit::DontCare);
                continue;
            }
        };

        if let Some((field_name, bit_idx)) = bit.as_var_bit() {
            parts.push(EncodingBit::FieldBit {
                field: field_name.to_string(),
                bit: bit_idx as u8,
            });
        } else if let Some(val) = bit.as_literal() {
            parts.push(if val { EncodingBit::One } else { EncodingBit::Zero });
        } else {
            parts.push(EncodingBit::DontCare);
        }
    }

    // Map DecoderNamespace to slot name
    let slot = match decoder_namespace.as_str() {
        "Alu" | "alu" => "alu",
        "Lda" | "lda" => "lda",
        "Ldb" | "ldb" => "ldb",
        "St" | "st" => "st",
        "Mv" | "mv" => "mv",
        "Vec" | "vec" => "vec",
        "Lng" | "lng" => "lng",
        _ => return None,
    };

    let slot_encoding = Some(SlotEncoding {
        slot: slot.to_string(),
        width: width as u8,
        parts,
    });

    // Extract operand info from InOperandList and OutOperandList (dag fields)
    let inputs = extract_dag_operands(record, "InOperandList");
    let outputs = extract_dag_operands(record, "OutOperandList");

    // Extract implicit defs/uses
    let defs = extract_def_list(record, "Defs");
    let uses = extract_def_list(record, "Uses");

    // Helper: read a boolean field (handles both `bit` and `int` types).
    // TableGen uses `bit` for most boolean flags, but some use `int`.
    let bool_field = |name: &str, default: bool| -> bool {
        // Try bit_value first (for `bit` fields), then int_value (for `int` fields)
        record
            .bit_value(name)
            .or_else(|_| record.int_value(name).map(|v| v != 0))
            .unwrap_or(default)
    };

    // Extract boolean flags
    let may_load = bool_field("mayLoad", false);
    let may_store = bool_field("mayStore", false);
    let has_side_effects = bool_field("hasSideEffects", false);
    let has_delay_slot = bool_field("hasDelaySlot", false);
    let has_complete_decoder = bool_field("hasCompleteDecoder", true);
    let is_move_imm = bool_field("isMoveImm", false);
    let is_move_reg = bool_field("isMoveReg", false);
    let is_branch = bool_field("isBranch", false);
    let is_call = bool_field("isCall", false);
    let is_return = bool_field("isReturn", false);
    let is_select = bool_field("isSelect", false);
    let is_terminator = bool_field("isTerminator", false);
    let is_compare = bool_field("isCompare", false);
    let is_slot_nop = mnemonic.starts_with("nop");

    // Extract itinerary class
    let itinerary_class = record
        .value("Itinerary")
        .ok()
        .and_then(|v| {
            let def = v.init.as_def().ok()?;
            let rec: tblgen::Record = def.into();
            Some(rec.name().ok()?.to_string())
        });

    // Extract parent class names for structural semantic inference
    let mut parents = Vec::new();
    for class in &[
        "AIE2Inst",
        "AIE2SlotInst",
        "AIE2_brcc_base",
        "AIE2_br_base",
        "AIE2_call_base",
    ] {
        if record.subclass_of(class) {
            parents.push(class.to_string());
        }
    }

    Some(InstrRecord {
        name,
        parents,
        decoder_namespace,
        inputs,
        outputs,
        mnemonic,
        asm_string,
        slot_encoding,
        defs,
        uses,
        may_load,
        may_store,
        has_side_effects,
        has_delay_slot,
        has_complete_decoder,
        is_code_gen_only,
        is_composite,
        is_move_imm,
        is_move_reg,
        is_slot_nop,
        is_branch,
        is_call,
        is_return,
        is_select,
        is_terminator,
        is_compare,
        itinerary_class,
    })
}

/// Extract (reg_class, operand_name) pairs from an InOperandList or OutOperandList dag.

fn extract_dag_operands(record: &tblgen::Record<'_>, field_name: &str) -> Vec<(String, String)> {
    let val = match record.value(field_name) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let dag: tblgen::init::DagInit = match val.init.as_dag() {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();
    for (arg_name, arg_init) in dag.args() {
        // Each arg is typically a DefInit pointing to a RegisterClass or Operand
        let reg_class = if let Ok(def_init) = arg_init.as_def() {
            let rec: tblgen::Record = def_init.into();
            rec.name().unwrap_or("").to_string()
        } else {
            String::new()
        };
        result.push((reg_class, arg_name.to_string()));
    }
    result
}

/// Extract a list of register names from a Defs or Uses field (list<Register>).

fn extract_def_list(record: &tblgen::Record<'_>, field_name: &str) -> Vec<String> {
    let val = match record.value(field_name) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let list: tblgen::init::ListInit = match val.init.as_list() {
        Ok(l) => l,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();
    for item in list.iter() {
        if let Ok(def_init) = item.as_def() {
            let rec: tblgen::Record = def_init.into();
            if let Ok(n) = rec.name() {
                result.push(n.to_string());
            }
        }
    }
    result
}

// ============================================================================
// Pattern records: Pat<> -> SemanticOp mapping
// ============================================================================

/// Extract Pat<> semantic mappings from TableGen Pattern records.
///
/// Walks all records derived from "Pattern" and extracts the SDNode/intrinsic
/// from PatternToMatch and the target instruction from ResultInstrs.
/// Applies the same "direct beats compound" selection logic as the text parser.

pub fn load_pattern_records(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<HashMap<String, SemanticOp>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;

    // Collect all candidates per instruction, then pick the best one.
    let mut candidates: HashMap<String, Vec<(SemanticOp, bool)>> = HashMap::new();

    for record in keeper.all_derived_definitions("Pattern") {
        // Extract the target instruction name from ResultInstrs (list<dag>)
        let result_instrs = match record.value("ResultInstrs") {
            Ok(v) => v,
            Err(_) => continue,
        };
        let result_list: tblgen::init::ListInit = match result_instrs.init.as_list() {
            Ok(l) => l,
            Err(_) => continue,
        };
        if result_list.len() == 0 {
            continue;
        }

        // Get first result dag: (INSTR_NAME operands...)
        let first_result = match result_list.iter().next() {
            Some(item) => item,
            None => continue,
        };
        let result_dag: tblgen::init::DagInit = match first_result.as_dag() {
            Ok(d) => d,
            Err(_) => continue,
        };

        // The operator of the result dag is the target instruction
        let instr_name = match result_dag.operator().name() {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };

        // Skip pseudo/meta instructions
        if instr_name == "REG_SEQUENCE"
            || instr_name == "COPY_TO_REGCLASS"
            || instr_name.starts_with("Pseudo")
            || instr_name == "EXTRACT_SUBREG"
            || instr_name == "INSERT_SUBREG"
            || instr_name == "IMPLICIT_DEF"
            || instr_name == "NegateImm"
        {
            continue;
        }

        // Extract semantic from PatternToMatch dag
        let pattern_val = match record.value("PatternToMatch") {
            Ok(v) => v,
            Err(_) => continue,
        };
        let pattern_dag: tblgen::init::DagInit = match pattern_val.init.as_dag() {
            Ok(d) => d,
            Err(_) => continue,
        };

        let semantic = match extract_semantic_from_dag_init(&pattern_dag) {
            Some(s) => s,
            None => continue,
        };

        // Determine if this is a "direct/simple" pattern.
        // Check if the record is a CmpSwapPat (swapped comparison operands).
        let is_swap = record.subclass_of("CmpSwapPat");
        // Check if result dag args contain nested instruction applications
        // (compound patterns like NegateImm, SUB wrapping).
        let has_compound_result = result_dag_has_compound_ops(&result_dag);
        let is_simple = !is_swap && !has_compound_result;

        candidates
            .entry(instr_name)
            .or_default()
            .push((semantic, is_simple));
    }

    // Pick the best candidate for each instruction.
    let mut result: HashMap<String, SemanticOp> = HashMap::new();
    for (instr_name, cands) in &candidates {
        if let Some((op, _)) = cands.iter().find(|(_, is_simple)| *is_simple) {
            result.insert(instr_name.clone(), *op);
        } else if let Some((op, _)) = cands.first() {
            result.insert(instr_name.clone(), *op);
        }
    }

    log::info!(
        "Native TableGen: extracted {} pattern semantics from {} Pattern records",
        result.len(),
        candidates.len(),
    );
    Ok(result)
}

/// Recursively extract a SemanticOp from a DAG init tree.
///
/// Walks the dag operator and nested sub-dags to find the first SDNode or
/// intrinsic name that maps to a SemanticOp.

fn extract_semantic_from_dag_init(dag: &tblgen::init::DagInit) -> Option<SemanticOp> {
    // Try the operator of this dag (operator() returns Record directly)
    let op_rec = dag.operator();
    if let Ok(name) = op_rec.name() {
        // Try as SDNode
        if let Some(op) = SemanticOp::from_sdnode(&name) {
            return Some(op);
        }
        // Try as intrinsic
        if let Some(op) = SemanticOp::from_intrinsic(&name) {
            return Some(op);
        }
    }

    // Recurse into arguments that are themselves dags.
    // Use indexed access since dag args may be unnamed (DagIter skips those).
    for i in 0..dag.num_args() {
        if let Some(arg_init) = dag.get(i) {
            if let Ok(sub_dag) = arg_init.as_dag() {
                if let Some(op) = extract_semantic_from_dag_init(&sub_dag) {
                    return Some(op);
                }
            }
        }
    }

    None
}

/// Check if a result DAG contains compound operations (NegateImm, SUB, ADD).

fn result_dag_has_compound_ops(dag: &tblgen::init::DagInit) -> bool {
    for i in 0..dag.num_args() {
        if let Some(arg_init) = dag.get(i) {
            if let Ok(sub_dag) = arg_init.as_dag() {
                let op_rec = sub_dag.operator();
                if let Ok(name) = op_rec.name() {
                    if name == "NegateImm" || name == "SUB" || name == "ADD" {
                        return true;
                    }
                }
            }
        }
    }
    false
}

// ============================================================================
// Pseudo expansion map: MultiSlot_Pseudo -> concrete instructions
// ============================================================================

/// Extract pseudo -> concrete instruction expansion maps.
///
/// Reads all records derived from "MultiSlot_Pseudo" and extracts the
/// `materializableInto` field (list<AIE2Inst>).

pub fn load_pseudo_expansion_map(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<HashMap<String, Vec<String>>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;
    let mut result: HashMap<String, Vec<String>> = HashMap::new();

    for record in keeper.all_derived_definitions("MultiSlot_Pseudo") {
        let name = match record.name() {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };

        let mat_val = match record.value("materializableInto") {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mat_list: tblgen::init::ListInit = match mat_val.init.as_list() {
            Ok(l) => l,
            Err(_) => continue,
        };

        let mut concretes = Vec::new();
        for item in mat_list.iter() {
            if let Ok(def_init) = item.as_def() {
                let rec: tblgen::Record = def_init.into();
                if let Ok(n) = rec.name() {
                    concretes.push(n.to_string());
                }
            }
        }

        if !concretes.is_empty() {
            result.insert(name, concretes);
        }
    }

    log::info!(
        "Native TableGen: extracted {} pseudo expansion maps",
        result.len(),
    );
    Ok(result)
}

// ============================================================================
// Processor model: SchedMachineModel
// ============================================================================

/// Extract the processor scheduling model.
///
/// Reads the single record derived from "SchedMachineModel" and extracts
/// latency/penalty/width parameters.

pub fn load_processor_model(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<Option<ProcessorModel>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;

    for record in keeper.all_derived_definitions("SchedMachineModel") {
        let int_field = |name: &str, default: i64| -> i64 {
            record.int_value(name).unwrap_or(default)
        };

        let itinerary_name = record
            .value("Itineraries")
            .ok()
            .and_then(|v| {
                let def = v.init.as_def().ok()?;
                let rec: tblgen::Record = def.into();
                Some(rec.name().ok()?.to_string())
            })
            .unwrap_or_default();

        return Ok(Some(ProcessorModel {
            load_latency: int_field("LoadLatency", 5) as u8,
            high_latency: int_field("HighLatency", 37) as u8,
            mispredict_penalty: int_field("MispredictPenalty", 4) as u8,
            issue_width: int_field("IssueWidth", 1000) as u16,
            itinerary_name,
        }));
    }

    Ok(None)
}

// ============================================================================
// Itinerary data: InstrItinData + InstrStage
// ============================================================================

/// Extract instruction itinerary data (per-class pipeline timing).
///
/// Two-pass extraction matching the text parser:
/// 1. Collect all InstrStage records (anonymous) into a name->stage map
/// 2. Collect all InstrItinData records, resolving stage references

pub fn load_itinerary_data(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<HashMap<String, ItineraryInfo>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;

    // Pass 1: collect InstrStage records
    let mut stages_map: HashMap<String, PipelineStage> = HashMap::new();
    for record in keeper.all_derived_definitions("InstrStage") {
        let name = match record.name() {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };

        let cycles = record.int_value("Cycles").unwrap_or(0) as u8;
        let time_inc = record.int_value("TimeInc").unwrap_or(1) as i8;

        let units_val = record.value("Units");
        let units = match units_val {
            Ok(v) => {
                match v.init.as_list() {
                    Ok(list) => {
                        let mut u = Vec::new();
                        for item in list.iter() {
                            if let Ok(def_init) = item.as_def() {
                                let rec: tblgen::Record = def_init.into();
                                if let Ok(n) = rec.name() {
                                    u.push(n.to_string());
                                }
                            }
                        }
                        u
                    }
                    Err(_) => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        };

        stages_map.insert(name, PipelineStage { cycles, units, time_inc });
    }

    // Pass 2: collect InstrItinData records
    let mut result: HashMap<String, ItineraryInfo> = HashMap::new();
    for record in keeper.all_derived_definitions("InstrItinData") {
        // Get the itinerary class name from TheClass field
        let class_name = match record.value("TheClass") {
            Ok(v) => {
                match v.init.as_def() {
                    Ok(def_init) => {
                        let rec: tblgen::Record = def_init.into();
                        match rec.name() {
                            Ok(n) => n.to_string(),
                            Err(_) => continue,
                        }
                    }
                    Err(_) => continue,
                }
            }
            Err(_) => continue,
        };

        // Get stage references
        let stages: Vec<PipelineStage> = match record.value("Stages") {
            Ok(v) => {
                match v.init.as_list() {
                    Ok(list) => {
                        let mut s = Vec::new();
                        for item in list.iter() {
                            if let Ok(def_init) = item.as_def() {
                                let rec: tblgen::Record = def_init.into();
                                if let Ok(n) = rec.name() {
                                    if let Some(stage) = stages_map.get(n) {
                                        s.push(stage.clone());
                                    }
                                }
                            }
                        }
                        s
                    }
                    Err(_) => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        };

        // Get operand cycles
        let operand_cycles: Vec<u8> = match record.value("OperandCycles") {
            Ok(v) => {
                match v.init.as_list() {
                    Ok(list) => {
                        list.iter()
                            .filter_map(|item| {
                                let int_init: tblgen::init::IntInit = item.as_int().ok()?;
                                let val: i64 = int_init.into();
                                Some(val as u8)
                            })
                            .collect()
                    }
                    Err(_) => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        };

        // Get bypasses
        let bypasses: Vec<String> = match record.value("Bypasses") {
            Ok(v) => {
                match v.init.as_list() {
                    Ok(list) => {
                        let mut b = Vec::new();
                        for item in list.iter() {
                            if let Ok(def_init) = item.as_def() {
                                let rec: tblgen::Record = def_init.into();
                                if let Ok(n) = rec.name() {
                                    b.push(n.to_string());
                                }
                            }
                        }
                        b
                    }
                    Err(_) => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        };

        let total_latency = stages.iter().map(|s| s.cycles).sum::<u8>();

        result.insert(
            class_name.clone(),
            ItineraryInfo {
                class_name,
                total_latency,
                operand_cycles,
                stages,
                bypasses,
            },
        );
    }

    log::info!(
        "Native TableGen: extracted {} itinerary classes, {} stage definitions",
        result.len(),
        stages_map.len(),
    );
    Ok(result)
}

// ============================================================================
// Register model: Register defs + RegisterClass
// ============================================================================

/// Recursively collect register names from a MemberList dag.
///
/// MemberList is typically `(add reg1, reg2, ...)` but may contain nested
/// `(add ...)` sub-dags for large register classes, or `(sequence ...)` nodes.
///
/// NOTE: We use indexed access (`dag.get(i)`) instead of `dag.args()` because
/// the DagIter requires both name AND value to be Some, but MemberList args
/// are unnamed positional arguments. Using `dag.args()` silently yields nothing.

fn collect_dag_register_refs(dag: &tblgen::init::DagInit, out: &mut Vec<String>) {
    for i in 0..dag.num_args() {
        let arg_init = match dag.get(i) {
            Some(a) => a,
            None => continue,
        };
        // Try as a register reference (DefInit -> Record)
        if let Ok(def_init) = arg_init.as_def() {
            let rec: tblgen::Record = def_init.into();
            if let Ok(n) = rec.name() {
                out.push(n.to_string());
            }
        }
        // Try as a nested dag (add ...)
        else if let Ok(sub_dag) = arg_init.as_dag() {
            collect_dag_register_refs(&sub_dag, out);
        }
    }
}

/// Extract the complete register model (registers + register classes).

pub fn load_register_model(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<RegisterModel, String> {
    let keeper = parse_td_file(td_file, include_paths)?;

    // Extract individual register definitions
    let mut registers: HashMap<String, RegisterDef> = HashMap::new();
    for record in keeper.all_derived_definitions("Register") {
        // Skip RegisterClass records
        if record.subclass_of("RegisterClass") {
            continue;
        }

        let name = match record.name() {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };

        // Only include AIE2 registers
        let namespace = record
            .string_value("Namespace")
            .unwrap_or_default();
        if namespace != "AIE2" {
            continue;
        }

        // Extract HWEncoding (bits<16>)
        let hw_encoding = match record.value("HWEncoding") {
            Ok(v) => {
                match v.init.as_bits() {
                    Ok(bits) => {
                        let mut val: u16 = 0;
                        for i in 0..bits.num_bits().min(16) {
                            if let Some(bit) = bits.bit(i) {
                                if let Some(true) = bit.as_literal() {
                                    val |= 1u16 << i;
                                }
                            }
                        }
                        val
                    }
                    Err(_) => continue,
                }
            }
            Err(_) => continue,
        };

        // Extract parent class names
        let mut parents = Vec::new();
        for class in &["AIE2GPReg", "DwarfRegNum", "AIE2Reg"] {
            if record.subclass_of(class) {
                parents.push(class.to_string());
            }
        }

        registers.insert(
            name.clone(),
            RegisterDef {
                name,
                hw_encoding,
                parents,
            },
        );
    }

    // Extract register classes.
    // Try multiple class names -- LLVM's RegisterClass may be stored under
    // different names depending on the TableGen version.
    let mut classes: HashMap<String, RegisterClassDef> = HashMap::new();
    let class_names = ["RegisterClass", "AIE2RegisterClass", "AIEBaseRegisterClass"];
    let class_iter = class_names.iter()
        .flat_map(|cn| keeper.all_derived_definitions(cn));
    for record in class_iter {
        let name = match record.name() {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };

        // Only include AIE2 register classes
        let namespace = record
            .string_value("Namespace")
            .unwrap_or_default();
        if namespace != "AIE2" {
            continue;
        }

        // Extract MemberList (dag with (add reg1, reg2, ...))
        // The MemberList may contain nested (add ...) dags for large classes,
        // so we recursively flatten all register references.
        let members: Vec<String> = match record.value("MemberList") {
            Ok(v) => {
                match v.init.as_dag() {
                    Ok(dag) => {
                        let mut m = Vec::new();
                        collect_dag_register_refs(&dag, &mut m);
                        m
                    }
                    Err(_) => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        };

        if members.is_empty() {
            continue;
        }

        let alignment = record.int_value("Alignment").unwrap_or(0) as u16;

        // Extract parent class names
        let mut parents = Vec::new();
        for class in &["RegisterClass", "AIE2RegisterClass"] {
            if record.subclass_of(class) {
                parents.push(class.to_string());
            }
        }

        classes.insert(
            name.clone(),
            RegisterClassDef {
                name,
                members,
                alignment,
                parents,
            },
        );
    }

    log::info!(
        "Native TableGen: extracted {} registers, {} register classes",
        registers.len(),
        classes.len(),
    );
    Ok(RegisterModel { registers, classes })
}

#[cfg(test)]

mod tests {
    use super::*;
    use std::path::PathBuf;

    fn llvm_aie_path() -> PathBuf {
        PathBuf::from("../llvm-aie")
    }

    fn td_file() -> PathBuf {
        llvm_aie_path().join("llvm/lib/Target/AIE/AIE2.td")
    }

    fn include_paths() -> Vec<PathBuf> {
        vec![
            llvm_aie_path().join("llvm/include"),
            llvm_aie_path().join("llvm/lib/Target/AIE"),
        ]
    }

    #[test]
    fn test_load_composite_formats() {
        let td = td_file();
        if !td.exists() {
            eprintln!("Skipping: llvm-aie not found at {:?}", td);
            return;
        }
        let inc_paths = include_paths();
        let incs: Vec<&Path> = inc_paths.iter().map(|p| p.as_path()).collect();
        let formats = load_composite_formats(&td, &incs).expect("should parse");

        // We expect ~78 composite formats
        assert!(
            formats.len() >= 50,
            "expected >= 50 composite formats, got {}",
            formats.len()
        );

        // Every format should have at least one slot
        for fmt in &formats {
            assert!(
                !fmt.slot_maps.is_empty(),
                "format {} has empty slot_maps",
                fmt.name
            );
            assert!(fmt.total_bits > 0, "format {} has 0 bits", fmt.name);
        }

        // Spot-check: I80_LDA_ALU_VEC should have lda, alu, vec slots
        if let Some(fmt80) = formats.iter().find(|f| f.name.contains("LDA_ALU_VEC")) {
            let slot_names: Vec<&str> = fmt80.slot_maps.iter().map(|s| s.slot_name.as_str()).collect();
            assert!(slot_names.contains(&"lda"), "missing lda slot in {:?}", slot_names);
            assert!(slot_names.contains(&"alu"), "missing alu slot in {:?}", slot_names);
            assert!(slot_names.contains(&"vec"), "missing vec slot in {:?}", slot_names);
            assert_eq!(fmt80.total_bytes, 10, "I80 should be 10 bytes");
        }
    }

    #[test]
    fn test_load_instruction_records() {
        let td = td_file();
        if !td.exists() {
            eprintln!("Skipping: llvm-aie not found at {:?}", td);
            return;
        }
        let inc_paths = include_paths();
        let incs: Vec<&Path> = inc_paths.iter().map(|p| p.as_path()).collect();
        let records = load_instruction_records(&td, &incs).expect("should parse");

        // We expect ~600+ instruction records
        assert!(
            records.len() >= 400,
            "expected >= 400 instruction records, got {}",
            records.len()
        );

        // Every record should have a non-empty slot encoding
        for rec in &records {
            assert!(
                rec.slot_encoding.is_some(),
                "record {} has no slot_encoding",
                rec.name
            );
        }

        // Spot-check: MOVX_alu_cg should be present (this was missing from
        // -gen-disassembler output, causing Peano decode failures)
        let has_movx_cg = records.iter().any(|r| r.name == "MOVX_alu_cg");
        assert!(
            has_movx_cg,
            "MOVX_alu_cg should be present in native records"
        );

        // Check slot distribution
        let mut by_slot = std::collections::HashMap::new();
        for rec in &records {
            if let Some(enc) = &rec.slot_encoding {
                *by_slot.entry(enc.slot.clone()).or_insert(0usize) += 1;
            }
        }
        log::info!("Instruction records by slot: {:?}", by_slot);
    }

    #[test]
    fn test_load_pattern_records() {
        let td = td_file();
        if !td.exists() {
            eprintln!("Skipping: llvm-aie not found at {:?}", td);
            return;
        }
        let inc_paths = include_paths();
        let incs: Vec<&Path> = inc_paths.iter().map(|p| p.as_path()).collect();
        let patterns = load_pattern_records(&td, &incs).expect("should parse");

        // We expect a reasonable number of pattern-derived semantics
        assert!(
            patterns.len() >= 30,
            "expected >= 30 pattern semantics, got {}",
            patterns.len(),
        );

        // Spot-check: ADD should map to SemanticOp::Add
        if let Some(op) = patterns.get("ADD_alu_r_ri") {
            assert_eq!(*op, super::super::types::SemanticOp::Add);
        }
    }

    #[test]
    fn test_load_pseudo_expansion_map() {
        let td = td_file();
        if !td.exists() {
            eprintln!("Skipping: llvm-aie not found at {:?}", td);
            return;
        }
        let inc_paths = include_paths();
        let incs: Vec<&Path> = inc_paths.iter().map(|p| p.as_path()).collect();
        let pseudo_map = load_pseudo_expansion_map(&td, &incs).expect("should parse");

        // We expect some pseudo expansion maps
        assert!(
            pseudo_map.len() >= 5,
            "expected >= 5 pseudo expansion maps, got {}",
            pseudo_map.len(),
        );

        // Each pseudo should expand to at least one concrete
        for (name, concretes) in &pseudo_map {
            assert!(
                !concretes.is_empty(),
                "pseudo {} has empty expansion list",
                name,
            );
        }
    }

    #[test]
    fn test_load_processor_model() {
        let td = td_file();
        if !td.exists() {
            eprintln!("Skipping: llvm-aie not found at {:?}", td);
            return;
        }
        let inc_paths = include_paths();
        let incs: Vec<&Path> = inc_paths.iter().map(|p| p.as_path()).collect();
        let model = load_processor_model(&td, &incs)
            .expect("should parse")
            .expect("should find a SchedMachineModel");

        assert!(model.load_latency > 0, "load_latency should be > 0");
        assert!(model.mispredict_penalty > 0, "mispredict_penalty should be > 0");
        assert!(!model.itinerary_name.is_empty(), "itinerary_name should not be empty");
    }

    #[test]
    fn test_load_itinerary_data() {
        let td = td_file();
        if !td.exists() {
            eprintln!("Skipping: llvm-aie not found at {:?}", td);
            return;
        }
        let inc_paths = include_paths();
        let incs: Vec<&Path> = inc_paths.iter().map(|p| p.as_path()).collect();
        let itineraries = load_itinerary_data(&td, &incs).expect("should parse");

        // We expect ~278 itinerary classes
        assert!(
            itineraries.len() >= 100,
            "expected >= 100 itinerary classes, got {}",
            itineraries.len(),
        );

        // Spot-check: II_ADD should exist with reasonable latency
        if let Some(ii_add) = itineraries.get("II_ADD") {
            assert!(ii_add.total_latency > 0, "II_ADD should have non-zero latency");
            assert!(!ii_add.stages.is_empty(), "II_ADD should have pipeline stages");
        }
    }

    #[test]
    fn test_load_register_model() {
        let td = td_file();
        if !td.exists() {
            eprintln!("Skipping: llvm-aie not found at {:?}", td);
            return;
        }
        let inc_paths = include_paths();
        let incs: Vec<&Path> = inc_paths.iter().map(|p| p.as_path()).collect();
        let model = load_register_model(&td, &incs).expect("should parse");

        // We expect ~50 registers
        assert!(
            model.registers.len() >= 20,
            "expected >= 20 registers, got {}",
            model.registers.len(),
        );

        // We expect ~30 register classes
        assert!(
            model.classes.len() >= 10,
            "expected >= 10 register classes, got {}",
            model.classes.len(),
        );

        // Spot-check: r0 should exist
        assert!(
            model.registers.contains_key("r0"),
            "r0 should be in register model",
        );

        // Spot-check: eR class should exist with members
        if let Some(er_class) = model.classes.get("eR") {
            assert!(
                !er_class.members.is_empty(),
                "eR class should have members",
            );
        }
    }
}
