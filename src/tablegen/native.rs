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

#[cfg(feature = "native-tblgen")]
use std::path::Path;

#[cfg(feature = "native-tblgen")]
use tblgen::TableGenParser;

#[cfg(feature = "native-tblgen")]
use super::tblgen_records::{EncodingBit, InstrRecord, SlotEncoding};
#[cfg(feature = "native-tblgen")]
use super::types::{CompositeFormatDef, SlotBitMap};

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
#[cfg(feature = "native-tblgen")]
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
#[cfg(feature = "native-tblgen")]
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
#[cfg(feature = "native-tblgen")]
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
#[cfg(feature = "native-tblgen")]
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
#[cfg(feature = "native-tblgen")]
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
#[cfg(feature = "native-tblgen")]
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
#[cfg(feature = "native-tblgen")]
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

#[cfg(test)]
#[cfg(feature = "native-tblgen")]
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
}
