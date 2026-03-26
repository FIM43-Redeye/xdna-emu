//! Build-time TableGen extraction via the `tblgen` crate.
//!
//! Adapted from `src/tablegen/native.rs`. Uses the tblgen crate API to parse
//! `.td` files in-process and produce build-time intermediate types
//! (`BuildInstrRecord`, etc.) that are then formatted as Rust source code.

use std::collections::HashMap;
use std::path::Path;

use tblgen::TableGenParser;

use super::records::{BuildEncodingBit, BuildInstrEncoding, BuildInstrRecord, BuildSlotEncoding};

// ============================================================================
// Build-time output types (no main-crate deps)
// ============================================================================

/// Processor scheduling model (build-time version).
pub struct BuildProcessorModel {
    pub load_latency: u8,
    pub high_latency: u8,
    pub mispredict_penalty: u8,
    pub issue_width: u16,
    pub itinerary_name: String,
}

/// Pipeline stage (build-time version).
#[derive(Clone)]
pub struct BuildPipelineStage {
    pub cycles: u8,
    pub units: Vec<String>,
    pub time_inc: i8,
}

/// Itinerary info (build-time version).
pub struct BuildItineraryInfo {
    pub class_name: String,
    pub total_latency: u8,
    pub operand_cycles: Vec<u8>,
    pub stages: Vec<BuildPipelineStage>,
    pub bypasses: Vec<String>,
}

/// Register definition (build-time version).
pub struct BuildRegisterDef {
    pub name: String,
    pub hw_encoding: u16,
    pub parents: Vec<String>,
}

/// Register class definition (build-time version).
pub struct BuildRegisterClassDef {
    pub name: String,
    pub members: Vec<String>,
    pub alignment: u16,
    pub parents: Vec<String>,
}

/// Register model (build-time version).
pub struct BuildRegisterModel {
    pub registers: HashMap<String, BuildRegisterDef>,
    pub classes: HashMap<String, BuildRegisterClassDef>,
}

impl Default for BuildRegisterModel {
    fn default() -> Self {
        Self {
            registers: HashMap::new(),
            classes: HashMap::new(),
        }
    }
}

/// Composite VLIW format definition (build-time version).
pub struct BuildCompositeFormat {
    pub name: String,
    pub total_bytes: u8,
    pub total_bits: u16,
    pub slots: Vec<(String, u16)>,
    pub fixed_mask: u128,
    pub fixed_value: u128,
    pub slot_maps: Vec<BuildSlotBitMap>,
}

/// Slot bit-map within a composite format.
pub struct BuildSlotBitMap {
    pub slot_name: String,
    pub width: u8,
    pub start_bit: u8,
}

/// Complete build-time extraction output.
pub struct BuildTblgenOutput {
    pub encodings_by_slot: HashMap<String, Vec<BuildInstrEncoding>>,
    pub processor_model: Option<BuildProcessorModel>,
    pub itineraries: HashMap<String, BuildItineraryInfo>,
    pub register_model: BuildRegisterModel,
    pub composite_formats: Vec<BuildCompositeFormat>,
    pub decoder_tables: HashMap<String, super::bytecode::BuildDecoderTable>,
}

impl BuildTblgenOutput {
    pub fn total_instructions(&self) -> usize {
        self.encodings_by_slot.values().map(|v| v.len()).sum()
    }

    pub fn slot_count(&self) -> usize {
        self.encodings_by_slot.len()
    }
}

// ============================================================================
// TableGen parsing
// ============================================================================

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
    parser
        .parse()
        .map_err(|e| format!("TableGen parse failed: {}", e))
}

// ============================================================================
// Instruction record extraction
// ============================================================================

fn load_instruction_records(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<Vec<BuildInstrRecord>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;
    let mut records = Vec::new();

    for record in keeper.all_derived_definitions("Instruction") {
        if let Some(instr) = extract_instruction_record(&record) {
            records.push(instr);
        }
    }

    eprintln!(
        "cargo:warning=TableGen: {} instruction records extracted",
        records.len()
    );
    Ok(records)
}

fn extract_instruction_record(record: &tblgen::Record<'_>) -> Option<BuildInstrRecord> {
    let name = record.name().ok()?.to_string();

    if record.subclass_of("AIE2CompositeInst") {
        return None;
    }

    let is_code_gen_only = record
        .bit_value("isCodeGenOnly")
        .or_else(|_| record.int_value("isCodeGenOnly").map(|v| v != 0))
        .unwrap_or(false);

    let is_composite = record.subclass_of("AIE2CompositeInst");

    let decoder_namespace = record.string_value("DecoderNamespace").ok()?;
    let decoder_namespace = decoder_namespace.trim().to_string();
    if decoder_namespace.is_empty() {
        return None;
    }

    let asm_string = record.string_value("AsmString").unwrap_or_default();
    let mnemonic = asm_string
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches('.')
        .to_string();

    let inst_val = record.value("Inst").ok()?;
    let bits_init = inst_val.init.as_bits().ok()?;
    let width = bits_init.num_bits();
    if width == 0 {
        return None;
    }

    let mut parts = Vec::with_capacity(width);
    for i in (0..width).rev() {
        let bit = match bits_init.bit(i) {
            Some(b) => b,
            None => {
                parts.push(BuildEncodingBit::DontCare);
                continue;
            }
        };
        if let Some((field_name, bit_idx)) = bit.as_var_bit() {
            parts.push(BuildEncodingBit::FieldBit {
                field: field_name.to_string(),
                bit: bit_idx as u8,
            });
        } else if let Some(val) = bit.as_literal() {
            parts.push(if val {
                BuildEncodingBit::One
            } else {
                BuildEncodingBit::Zero
            });
        } else {
            parts.push(BuildEncodingBit::DontCare);
        }
    }

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

    let slot_encoding = Some(BuildSlotEncoding {
        slot: slot.to_string(),
        width: width as u8,
        parts,
    });

    let inputs = extract_dag_operands(record, "InOperandList");
    let outputs = extract_dag_operands(record, "OutOperandList");
    let defs = extract_def_list(record, "Defs");
    let uses = extract_def_list(record, "Uses");

    let bool_field = |name: &str, default: bool| -> bool {
        record
            .bit_value(name)
            .or_else(|_| record.int_value(name).map(|v| v != 0))
            .unwrap_or(default)
    };

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

    let itinerary_class = record.value("Itinerary").ok().and_then(|v| {
        let def = v.init.as_def().ok()?;
        let rec: tblgen::Record = def.into();
        Some(rec.name().ok()?.to_string())
    });

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

    Some(BuildInstrRecord {
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
        has_complete_decoder,
        has_delay_slot,
        is_code_gen_only,
        is_move_imm,
        is_move_reg,
        is_slot_nop,
        is_branch,
        is_call,
        is_return,
        is_select,
        is_terminator,
        is_compare,
        is_composite,
        itinerary_class,
    })
}

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
// Pattern records: Pat<> -> semantic string mapping
// ============================================================================

fn load_pattern_records(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<HashMap<String, String>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;
    let mut candidates: HashMap<String, Vec<(String, bool)>> = HashMap::new();

    for record in keeper.all_derived_definitions("Pattern") {
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

        let first_result = match result_list.iter().next() {
            Some(item) => item,
            None => continue,
        };
        let result_dag: tblgen::init::DagInit = match first_result.as_dag() {
            Ok(d) => d,
            Err(_) => continue,
        };

        let instr_name = match result_dag.operator().name() {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };

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

        let is_swap = record.subclass_of("CmpSwapPat");
        let has_compound_result = result_dag_has_compound_ops(&result_dag);
        let is_simple = !is_swap && !has_compound_result;

        candidates
            .entry(instr_name)
            .or_default()
            .push((semantic, is_simple));
    }

    let mut result: HashMap<String, String> = HashMap::new();
    for (instr_name, cands) in &candidates {
        if let Some((op, _)) = cands.iter().find(|(_, is_simple)| *is_simple) {
            result.insert(instr_name.clone(), op.clone());
        } else if let Some((op, _)) = cands.first() {
            result.insert(instr_name.clone(), op.clone());
        }
    }

    Ok(result)
}

/// Map SDNode/intrinsic names to SemanticOp string representations.
fn semantic_from_sdnode(name: &str) -> Option<String> {
    let s = match name {
        "add" => "SemanticOp::Add",
        "sub" => "SemanticOp::Sub",
        "mul" => "SemanticOp::Mul",
        "sdiv" => "SemanticOp::SDiv",
        "udiv" => "SemanticOp::UDiv",
        "srem" => "SemanticOp::SRem",
        "urem" => "SemanticOp::URem",
        "abs" => "SemanticOp::Abs",
        "neg" | "ineg" => "SemanticOp::Neg",
        "and" => "SemanticOp::And",
        "or" => "SemanticOp::Or",
        "xor" => "SemanticOp::Xor",
        "not" => "SemanticOp::Not",
        "shl" => "SemanticOp::Shl",
        "sra" => "SemanticOp::Sra",
        "srl" => "SemanticOp::Srl",
        "rotl" => "SemanticOp::Rotl",
        "rotr" => "SemanticOp::Rotr",
        "seteq" => "SemanticOp::SetEq",
        "setne" => "SemanticOp::SetNe",
        "setlt" => "SemanticOp::SetLt",
        "setle" => "SemanticOp::SetLe",
        "setgt" => "SemanticOp::SetGt",
        "setge" => "SemanticOp::SetGe",
        "setult" => "SemanticOp::SetUlt",
        "setule" => "SemanticOp::SetUle",
        "setugt" => "SemanticOp::SetUgt",
        "setuge" => "SemanticOp::SetUge",
        "ctlz" | "ctlz_zero_undef" => "SemanticOp::Ctlz",
        "cttz" | "cttz_zero_undef" => "SemanticOp::Cttz",
        "ctpop" => "SemanticOp::Ctpop",
        "bswap" => "SemanticOp::Bswap",
        "fadd" => "SemanticOp::Add",
        "fsub" => "SemanticOp::Sub",
        "fmul" => "SemanticOp::Mul",
        "load" => "SemanticOp::Load",
        "store" => "SemanticOp::Store",
        "ptradd" => "SemanticOp::PointerAdd",
        "br" => "SemanticOp::Br",
        "brcond" => "SemanticOp::BrCond",
        "select" => "SemanticOp::Select",
        "sext" | "sign_extend" | "sext_inreg" => "SemanticOp::SignExtend",
        "zext" | "zero_extend" => "SemanticOp::ZeroExtend",
        "trunc" | "truncate" => "SemanticOp::Truncate",
        "copy" | "CopyToReg" | "CopyFromReg" => "SemanticOp::Copy",
        _ => return None,
    };
    Some(s.to_string())
}

/// Map intrinsic names to SemanticOp string representations.
fn semantic_from_intrinsic(name: &str) -> Option<String> {
    let stem = name.strip_prefix("int_aie2_")?;

    if let Some(before_conf) = stem.strip_suffix("_conf") {
        let base = before_conf.trim_end_matches(|c: char| c.is_ascii_digit());
        if base.ends_with("_negmac") || base.ends_with("_negmsc") {
            return Some("SemanticOp::NegMul".to_string());
        }
        if base.ends_with("_negmul") {
            return Some("SemanticOp::NegMatMul".to_string());
        }
        if base.ends_with("_addmac") || base.ends_with("_addmsc") {
            return Some("SemanticOp::AddMac".to_string());
        }
        if base.ends_with("_submac") || base.ends_with("_submsc") {
            return Some("SemanticOp::SubMac".to_string());
        }
        if base.ends_with("_mac") || base.ends_with("_msc") {
            return Some("SemanticOp::Mac".to_string());
        }
        if base.ends_with("_mul") {
            return Some("SemanticOp::MatMul".to_string());
        }
        if base.ends_with("_neg") {
            return Some("SemanticOp::Neg".to_string());
        }
        if base.starts_with("vaddsub") {
            return Some("SemanticOp::Add".to_string());
        }
        if base.starts_with("clr") {
            return Some("SemanticOp::VectorClear".to_string());
        }
    }

    if stem.starts_with("add_acc")
        || stem.starts_with("sub_acc")
        || stem.starts_with("negadd_acc")
        || stem.starts_with("negsub_acc")
    {
        return Some("SemanticOp::Accumulate".to_string());
    }
    if stem.starts_with("concat_") {
        return Some("SemanticOp::Shuffle".to_string());
    }
    if stem.ends_with("_ups") {
        return Some("SemanticOp::Ups".to_string());
    }
    if stem.ends_with("_srs") {
        return Some("SemanticOp::Srs".to_string());
    }
    if stem.starts_with("vshuffle") || stem.starts_with("vbcst_shuffle") {
        return Some("SemanticOp::Shuffle".to_string());
    }
    if stem.starts_with("vshift") {
        return Some("SemanticOp::Align".to_string());
    }
    if stem.starts_with("vsel") {
        return Some("SemanticOp::VectorSelect".to_string());
    }
    if stem.starts_with("vbroadcast_zero") {
        return Some("SemanticOp::VectorClear".to_string());
    }
    if stem.starts_with("vbroadcast") {
        return Some("SemanticOp::VectorBroadcast".to_string());
    }
    if stem.starts_with("vextract_broadcast") || stem.starts_with("vextract_bcast") {
        return Some("SemanticOp::VectorBroadcast".to_string());
    }
    if stem.starts_with("vextract_elem") {
        return Some("SemanticOp::VectorExtract".to_string());
    }
    if stem.starts_with("vinsert") {
        return Some("SemanticOp::VectorInsert".to_string());
    }
    if stem.starts_with("vmaxdiff_lt") {
        return Some("SemanticOp::MaxDiffLt".to_string());
    }
    if stem.starts_with("vsub_lt") {
        return Some("SemanticOp::SubLt".to_string());
    }
    if stem.starts_with("vsub_ge") {
        return Some("SemanticOp::SubGe".to_string());
    }
    if stem.starts_with("vmax_lt") {
        return Some("SemanticOp::MaxLt".to_string());
    }
    if stem.starts_with("vmin_ge") {
        return Some("SemanticOp::MinGe".to_string());
    }
    if stem.starts_with("vlt") {
        return Some("SemanticOp::SetLt".to_string());
    }
    if stem.starts_with("vge") {
        return Some("SemanticOp::SetGe".to_string());
    }
    if stem.starts_with("veqz") {
        return Some("SemanticOp::SetEq".to_string());
    }
    if stem.starts_with("vabs_gtz") {
        return Some("SemanticOp::AbsGtz".to_string());
    }
    if stem.starts_with("vneg_gtz") {
        return Some("SemanticOp::NegGtz".to_string());
    }
    if stem.starts_with("vbneg_ltz") {
        return Some("SemanticOp::NegLtz".to_string());
    }
    if stem.contains("_to_v") {
        return Some("SemanticOp::Convert".to_string());
    }
    if stem.starts_with("load_4x") {
        return Some("SemanticOp::Load".to_string());
    }
    if stem.starts_with("upd_") || stem.starts_with("set_") || stem.starts_with("ext_") {
        return Some("SemanticOp::Copy".to_string());
    }
    if stem.starts_with("acquire") {
        return Some("SemanticOp::LockAcquire".to_string());
    }
    if stem.starts_with("release") {
        return Some("SemanticOp::LockRelease".to_string());
    }
    if stem.starts_with("pack_") {
        return Some("SemanticOp::Pack".to_string());
    }
    if stem.starts_with("unpack_") {
        return Some("SemanticOp::Unpack".to_string());
    }
    if stem.starts_with("put_ms") || stem.starts_with("put_wss") {
        return Some("SemanticOp::StreamWrite".to_string());
    }
    if stem.starts_with("get_ss") || stem.starts_with("get_wss") {
        return Some("SemanticOp::StreamRead".to_string());
    }
    if stem.starts_with("scd_read") {
        return Some("SemanticOp::CascadeRead".to_string());
    }
    if stem.starts_with("scd_write") {
        return Some("SemanticOp::CascadeWrite".to_string());
    }

    match stem {
        "done" => Some("SemanticOp::Done".to_string()),
        "event" | "event0" | "event1" => Some("SemanticOp::Event".to_string()),
        "clb" => Some("SemanticOp::Clb".to_string()),
        "divs" => Some("SemanticOp::SDiv".to_string()),
        "sched_barrier" => Some("SemanticOp::Nop".to_string()),
        _ => None,
    }
}

fn extract_semantic_from_dag_init(dag: &tblgen::init::DagInit) -> Option<String> {
    let op_rec = dag.operator();
    if let Ok(name) = op_rec.name() {
        if let Some(op) = semantic_from_sdnode(&name) {
            return Some(op);
        }
        if let Some(op) = semantic_from_intrinsic(&name) {
            return Some(op);
        }
    }

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
// Pseudo expansion map
// ============================================================================

fn load_pseudo_expansion_map(
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

    Ok(result)
}

// ============================================================================
// Processor model
// ============================================================================

fn load_processor_model(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<Option<BuildProcessorModel>, String> {
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

        return Ok(Some(BuildProcessorModel {
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
// Itinerary data
// ============================================================================

fn load_itinerary_data(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<HashMap<String, BuildItineraryInfo>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;

    let mut stages_map: HashMap<String, BuildPipelineStage> = HashMap::new();
    for record in keeper.all_derived_definitions("InstrStage") {
        let name = match record.name() {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };
        let cycles = record.int_value("Cycles").unwrap_or(0) as u8;
        let time_inc = record.int_value("TimeInc").unwrap_or(1) as i8;
        let units = match record.value("Units") {
            Ok(v) => match v.init.as_list() {
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
            },
            Err(_) => Vec::new(),
        };
        stages_map.insert(
            name,
            BuildPipelineStage {
                cycles,
                units,
                time_inc,
            },
        );
    }

    let mut result: HashMap<String, BuildItineraryInfo> = HashMap::new();
    for record in keeper.all_derived_definitions("InstrItinData") {
        let class_name = match record.value("TheClass") {
            Ok(v) => match v.init.as_def() {
                Ok(def_init) => {
                    let rec: tblgen::Record = def_init.into();
                    match rec.name() {
                        Ok(n) => n.to_string(),
                        Err(_) => continue,
                    }
                }
                Err(_) => continue,
            },
            Err(_) => continue,
        };

        let stages: Vec<BuildPipelineStage> = match record.value("Stages") {
            Ok(v) => match v.init.as_list() {
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
            },
            Err(_) => Vec::new(),
        };

        let operand_cycles: Vec<u8> = match record.value("OperandCycles") {
            Ok(v) => match v.init.as_list() {
                Ok(list) => list
                    .iter()
                    .filter_map(|item| {
                        let int_init: tblgen::init::IntInit = item.as_int().ok()?;
                        let val: i64 = int_init.into();
                        Some(val as u8)
                    })
                    .collect(),
                Err(_) => Vec::new(),
            },
            Err(_) => Vec::new(),
        };

        let bypasses: Vec<String> = match record.value("Bypasses") {
            Ok(v) => match v.init.as_list() {
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
            },
            Err(_) => Vec::new(),
        };

        let total_latency = stages.iter().map(|s| s.cycles).sum::<u8>();

        result.insert(
            class_name.clone(),
            BuildItineraryInfo {
                class_name,
                total_latency,
                operand_cycles,
                stages,
                bypasses,
            },
        );
    }

    Ok(result)
}

// ============================================================================
// Register model
// ============================================================================

fn collect_dag_register_refs(dag: &tblgen::init::DagInit, out: &mut Vec<String>) {
    for i in 0..dag.num_args() {
        let arg_init = match dag.get(i) {
            Some(a) => a,
            None => continue,
        };
        if let Ok(def_init) = arg_init.as_def() {
            let rec: tblgen::Record = def_init.into();
            if let Ok(n) = rec.name() {
                out.push(n.to_string());
            }
        } else if let Ok(sub_dag) = arg_init.as_dag() {
            collect_dag_register_refs(&sub_dag, out);
        }
    }
}

fn load_register_model(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<BuildRegisterModel, String> {
    let keeper = parse_td_file(td_file, include_paths)?;

    let mut registers: HashMap<String, BuildRegisterDef> = HashMap::new();
    for record in keeper.all_derived_definitions("Register") {
        if record.subclass_of("RegisterClass") {
            continue;
        }
        let name = match record.name() {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };
        let namespace = record.string_value("Namespace").unwrap_or_default();
        if namespace != "AIE2" {
            continue;
        }
        let hw_encoding = match record.value("HWEncoding") {
            Ok(v) => match v.init.as_bits() {
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
            },
            Err(_) => continue,
        };

        let mut parents = Vec::new();
        for class in &["AIE2GPReg", "DwarfRegNum", "AIE2Reg"] {
            if record.subclass_of(class) {
                parents.push(class.to_string());
            }
        }

        registers.insert(
            name.clone(),
            BuildRegisterDef {
                name,
                hw_encoding,
                parents,
            },
        );
    }

    let mut classes: HashMap<String, BuildRegisterClassDef> = HashMap::new();
    let class_names = [
        "RegisterClass",
        "AIE2RegisterClass",
        "AIEBaseRegisterClass",
    ];
    let class_iter = class_names
        .iter()
        .flat_map(|cn| keeper.all_derived_definitions(cn));
    for record in class_iter {
        let name = match record.name() {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };
        let namespace = record.string_value("Namespace").unwrap_or_default();
        if namespace != "AIE2" {
            continue;
        }
        let members: Vec<String> = match record.value("MemberList") {
            Ok(v) => match v.init.as_dag() {
                Ok(dag) => {
                    let mut m = Vec::new();
                    collect_dag_register_refs(&dag, &mut m);
                    m
                }
                Err(_) => Vec::new(),
            },
            Err(_) => Vec::new(),
        };
        if members.is_empty() {
            continue;
        }
        let alignment = record.int_value("Alignment").unwrap_or(0) as u16;
        let mut parents = Vec::new();
        for class in &["RegisterClass", "AIE2RegisterClass"] {
            if record.subclass_of(class) {
                parents.push(class.to_string());
            }
        }
        classes.insert(
            name.clone(),
            BuildRegisterClassDef {
                name,
                members,
                alignment,
                parents,
            },
        );
    }

    Ok(BuildRegisterModel {
        registers,
        classes,
    })
}

// ============================================================================
// Composite format extraction
// ============================================================================

fn load_composite_formats(
    td_file: &Path,
    include_paths: &[&Path],
) -> Result<Vec<BuildCompositeFormat>, String> {
    let keeper = parse_td_file(td_file, include_paths)?;
    let mut formats = Vec::new();

    for record in keeper.all_derived_definitions("AIE2CompositeInst") {
        if let Some(fmt) = extract_composite_format(&record) {
            formats.push(fmt);
        }
    }

    Ok(formats)
}

fn extract_composite_format(record: &tblgen::Record<'_>) -> Option<BuildCompositeFormat> {
    let name = record.name().ok()?.to_string();
    let inst_val = record.value("Inst").ok()?;
    let bits_init = inst_val.init.as_bits().ok()?;
    let total_bits = bits_init.num_bits();
    if total_bits == 0 {
        return None;
    }
    let total_bytes = (total_bits as u8 + 7) / 8;

    let mut fixed_mask: u128 = 0;
    let mut fixed_value: u128 = 0;
    let mut slot_bits: HashMap<String, Vec<(usize, usize)>> = HashMap::new();

    for i in 0..total_bits {
        let bit = match bits_init.bit(i) {
            Some(b) => b,
            None => continue,
        };
        if let Some((var_name, var_bit)) = bit.as_var_bit() {
            slot_bits
                .entry(var_name.to_string())
                .or_default()
                .push((var_bit, i));
        } else if let Some(val) = bit.as_literal() {
            fixed_mask |= 1u128 << i;
            if val {
                fixed_value |= 1u128 << i;
            }
        }
    }

    let mut slot_maps = Vec::new();
    let mut slots = Vec::new();
    let mut slot_names: Vec<String> = slot_bits.keys().cloned().collect();
    slot_names.sort();

    for slot_name in &slot_names {
        let bits = &slot_bits[slot_name];
        if bits.is_empty() {
            continue;
        }
        let min_word_pos = bits.iter().map(|&(_, wp)| wp).min().unwrap();
        let max_word_pos = bits.iter().map(|&(_, wp)| wp).max().unwrap();
        let width = (max_word_pos - min_word_pos + 1) as u8;

        slot_maps.push(BuildSlotBitMap {
            slot_name: slot_name.clone(),
            width,
            start_bit: min_word_pos as u8,
        });
        slots.push((slot_name.clone(), width as u16));
    }

    Some(BuildCompositeFormat {
        name,
        total_bytes,
        total_bits: total_bits as u16,
        slots,
        fixed_mask,
        fixed_value,
        slot_maps,
    })
}

// ============================================================================
// Itinerary-based semantic inference (layer 4)
// ============================================================================

const ITINERARY_SEMANTICS: &[(&str, &str)] = &[
    ("II_PADD", "SemanticOp::PointerAdd"),
    ("II_STHB", "SemanticOp::Store"),
    ("II_MOV_CPH", "SemanticOp::StreamWritePacketHeader"),
    ("II_MOV_PH", "SemanticOp::StreamWritePacketHeader"),
    ("II_ST_MS", "SemanticOp::StreamWrite"),
    ("II_MOV_SS", "SemanticOp::StreamRead"),
    ("II_VMOV_CASCADE_READ", "SemanticOp::CascadeRead"),
    ("II_VMOV_CASCADE_WRITE", "SemanticOp::CascadeWrite"),
    ("II_VMOV", "SemanticOp::Copy"),
    ("II_VPACK", "SemanticOp::Pack"),
    ("II_VUNPACK", "SemanticOp::Unpack"),
    ("II_VPUSH_HI", "SemanticOp::VectorInsert"),
    ("II_VPUSH_LO", "SemanticOp::VectorInsert"),
    ("II_VSRSM", "SemanticOp::Srs"),
    ("II_VBCST", "SemanticOp::VectorBroadcast"),
    ("II_VEXTBCST", "SemanticOp::VectorBroadcast"),
    ("II_ADC", "SemanticOp::Adc"),
    ("II_SBC", "SemanticOp::Sbc"),
    ("II_ADD_NC", "SemanticOp::Add"),
    ("II_MOVd", "SemanticOp::Copy"),
    ("II_MOV_CNTR", "SemanticOp::Copy"),
    ("II_VFLOORs32bf16", "SemanticOp::Convert"),
    ("II_DIVS", "SemanticOp::SDiv"),
    ("II_VADDSUB", "SemanticOp::Add"),
    ("II_EXTENDs", "SemanticOp::SignExtend"),
    ("II_EXTENDu", "SemanticOp::ZeroExtend"),
];

// ============================================================================
// Main entry point: extract_all
// ============================================================================

/// Run the complete TableGen extraction pipeline at build time.
///
/// Extracts instruction encodings, scheduling model, register definitions,
/// composite formats, and decoder bytecode tables. Applies all four semantic
/// propagation layers (Pat<>, pseudo expansion, C++ switch, itinerary).
pub fn extract_all(llvm_aie_path: &Path) -> Result<BuildTblgenOutput, String> {
    let td_file = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2.td");
    let inc_path_bufs = vec![
        llvm_aie_path.join("llvm/include"),
        llvm_aie_path.join("llvm/lib/Target/AIE"),
    ];
    let inc_refs: Vec<&Path> = inc_path_bufs.iter().map(|p| p.as_path()).collect();

    // Extract instruction records
    let records = load_instruction_records(&td_file, &inc_refs)?;

    // Convert to encodings grouped by slot
    let mut by_slot: HashMap<String, Vec<BuildInstrEncoding>> = HashMap::new();
    for record in records {
        if let Some(encoding) = record.to_build_encoding() {
            by_slot
                .entry(encoding.slot.clone())
                .or_default()
                .push(encoding);
        }
    }

    // Layer 1: Pat<>-derived semantics
    let pattern_map = load_pattern_records(&td_file, &inc_refs).unwrap_or_default();
    let mut pattern_upgraded = 0usize;
    for encodings in by_slot.values_mut() {
        for enc in encodings.iter_mut() {
            if let Some(op) = pattern_map.get(enc.name.as_str()) {
                enc.semantic = Some(op.clone());
                pattern_upgraded += 1;
            }
        }
    }
    eprintln!(
        "cargo:warning=TableGen: {} pattern semantics applied",
        pattern_upgraded
    );

    // Layer 2: Pseudo expansion propagation
    let pseudo_map = load_pseudo_expansion_map(&td_file, &inc_refs).unwrap_or_default();
    let mut expansion_semantics: HashMap<String, String> = HashMap::new();
    for (pseudo_name, concretes) in &pseudo_map {
        if let Some(op) = pattern_map.get(pseudo_name.as_str()) {
            for concrete in concretes {
                expansion_semantics.insert(concrete.clone(), op.clone());
            }
        }
    }
    let mut pseudo_propagated = 0usize;
    for encodings in by_slot.values_mut() {
        for enc in encodings.iter_mut() {
            if enc.semantic.is_none() {
                if let Some(op) = expansion_semantics.get(enc.name.as_str()) {
                    enc.semantic = Some(op.clone());
                    pseudo_propagated += 1;
                }
            }
        }
    }
    if pseudo_propagated > 0 {
        eprintln!(
            "cargo:warning=TableGen: {} pseudo semantics propagated",
            pseudo_propagated
        );
    }

    // Layer 3: C++ selection propagation
    let cpp_map = super::cpp_switch::parse_cpp_opcode_switch(llvm_aie_path);
    let mut cpp_propagated = 0usize;
    for (intrinsic_stem, opcodes) in &cpp_map {
        if let Some(op) = semantic_from_intrinsic(&format!("int_aie2_{}", intrinsic_stem)) {
            for opcode_name in opcodes {
                for encodings in by_slot.values_mut() {
                    for enc in encodings.iter_mut() {
                        if enc.semantic.is_none() && enc.name == *opcode_name {
                            enc.semantic = Some(op.clone());
                            cpp_propagated += 1;
                        }
                    }
                }
            }
        }
    }
    if cpp_propagated > 0 {
        eprintln!(
            "cargo:warning=TableGen: {} C++ semantics propagated",
            cpp_propagated
        );
    }

    // Layer 4: Itinerary-based inference
    let mut itinerary_inferred = 0usize;
    for encodings in by_slot.values_mut() {
        for enc in encodings.iter_mut() {
            if enc.semantic.is_none() {
                if let Some(ref sched) = enc.sched_class {
                    for &(prefix, op_str) in ITINERARY_SEMANTICS {
                        if sched.starts_with(prefix) {
                            enc.semantic = Some(op_str.to_string());
                            itinerary_inferred += 1;
                            break;
                        }
                    }
                }
            }
        }
    }
    if itinerary_inferred > 0 {
        eprintln!(
            "cargo:warning=TableGen: {} itinerary semantics inferred",
            itinerary_inferred
        );
    }

    // Post-propagation: derive is_ptr_arithmetic from resolved semantics
    for encodings in by_slot.values_mut() {
        for enc in encodings.iter_mut() {
            if enc.semantic.as_deref() == Some("SemanticOp::PointerAdd") {
                enc.is_ptr_arithmetic = true;
            }
        }
    }

    // Extended data
    let processor_model = load_processor_model(&td_file, &inc_refs).unwrap_or(None);
    let itineraries = load_itinerary_data(&td_file, &inc_refs).unwrap_or_default();
    let register_model = load_register_model(&td_file, &inc_refs).unwrap_or_default();
    let composite_formats = load_composite_formats(&td_file, &inc_refs).unwrap_or_default();

    // Decoder bytecode tables from llvm-tblgen subprocess
    let decoder_tables = run_disassembler_and_parse(llvm_aie_path);

    Ok(BuildTblgenOutput {
        encodings_by_slot: by_slot,
        processor_model,
        itineraries,
        register_model,
        composite_formats,
        decoder_tables,
    })
}

// ============================================================================
// Disassembler subprocess for decoder bytecode
// ============================================================================

/// Run llvm-tblgen -gen-disassembler and parse the output.
fn run_disassembler_and_parse(
    llvm_aie_path: &Path,
) -> HashMap<String, super::bytecode::BuildDecoderTable> {
    let base = llvm_aie_path.join("llvm/lib/Target/AIE");

    // Find llvm-tblgen binary (same search logic as the runtime version)
    let tblgen_path = find_aie_tblgen(llvm_aie_path);
    let tblgen_path = match tblgen_path {
        Some(p) => p,
        None => {
            eprintln!("cargo:warning=No AIE-enabled llvm-tblgen found, decoder tables will be empty");
            return HashMap::new();
        }
    };

    eprintln!(
        "cargo:warning=Running llvm-tblgen -gen-disassembler from {}",
        tblgen_path.display()
    );

    let output = match std::process::Command::new(&tblgen_path)
        .arg("-gen-disassembler")
        .args(["AIE2.td", "-I.", "-I../../..", "-I../../../include"])
        .current_dir(&base)
        .env_remove("LD_LIBRARY_PATH")
        .output()
    {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).into_owned()
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!(
                "cargo:warning=llvm-tblgen -gen-disassembler failed: {}",
                stderr
            );
            return HashMap::new();
        }
        Err(e) => {
            eprintln!("cargo:warning=Failed to run llvm-tblgen: {}", e);
            return HashMap::new();
        }
    };

    super::bytecode::extract_all_tables(&output)
}

/// Find the AIE-specific llvm-tblgen binary.
fn find_aie_tblgen(llvm_aie_path: &Path) -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var("LLVM_AIE_TBLGEN") {
        let p = std::path::PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    let exe = if cfg!(target_os = "windows") {
        "llvm-tblgen.exe"
    } else {
        "llvm-tblgen"
    };

    let candidates: Vec<Option<std::path::PathBuf>> = vec![
        llvm_aie_path.parent().map(|p| {
            p.join(format!(
                "mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie/bin/{exe}"
            ))
        }),
        llvm_aie_path.parent().map(|p| {
            p.join(format!(
                "mlir-aie/ironenv/lib/python3.12/site-packages/llvm-aie/bin/{exe}"
            ))
        }),
        llvm_aie_path.parent().map(|p| {
            p.join(format!(
                "mlir-aie/ironenv/lib/python3.11/site-packages/llvm-aie/bin/{exe}"
            ))
        }),
        llvm_aie_path
            .parent()
            .map(|p| p.join(format!("mlir-aie/my_install/mlir/bin/{exe}"))),
        llvm_aie_path
            .parent()
            .map(|p| p.join(format!("mlir-aie/install/bin/{exe}"))),
        Some(llvm_aie_path.join(format!("build/bin/{exe}"))),
        Some(llvm_aie_path.join(format!("build/Release/bin/{exe}"))),
    ];

    for candidate in candidates.into_iter().flatten() {
        if let Ok(canonical) = candidate.canonicalize() {
            return Some(canonical);
        }
    }

    None
}
