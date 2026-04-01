//! Export AIE2 ISA metadata to JSON for the validation harness.
//!
//! Loads all instruction encodings from the build-time generated TableGen data
//! and serializes them to a JSON file. Each instruction includes its name,
//! mnemonic, assembly format string, slot, operand details, and flags.
//!
//! Usage:
//!   cargo run --bin export_isa -- --output tools/aie2-isa.json

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json;

use xdna_emu::tablegen::{
    self, InstrEncoding, OperandType, RegisterKind,
};

// ---------------------------------------------------------------------------
// Serializable mirror types (we don't want to add Serialize to core types)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct IsaOperand {
    name: String,
    bit_width: u8,
    is_output: bool,
    operand_type: String,
    register_kind: Option<String>,
    signed: bool,
    scale: Option<i32>,
}

#[derive(Serialize)]
struct IsaInstruction {
    name: String,
    mnemonic: String,
    asm_string: String,
    slot: String,
    width: u8,
    operands: Vec<IsaOperand>,
    may_load: bool,
    may_store: bool,
    is_vector: bool,
    has_complete_decoder: bool,
    sched_class: Option<String>,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn operand_type_str(ot: &OperandType) -> String {
    match ot {
        OperandType::Register(_) => "register".to_string(),
        OperandType::RegisterWithOffset(_, off) => format!("register+{}", off),
        OperandType::CompositeRegister(_) => "composite_register".to_string(),
        OperandType::Immediate { .. } => "immediate".to_string(),
        OperandType::LockId => "lock_id".to_string(),
        OperandType::Unknown => "unknown".to_string(),
    }
}

fn register_kind_str(ot: &OperandType) -> Option<String> {
    let kind = match ot {
        OperandType::Register(k) | OperandType::RegisterWithOffset(k, _) => k,
        OperandType::CompositeRegister(enc) => {
            return Some(format!("{:?}", enc));
        }
        _ => return None,
    };
    Some(match kind {
        RegisterKind::Scalar => "scalar",
        RegisterKind::Pointer => "pointer",
        RegisterKind::ModifierM => "modifier_m",
        RegisterKind::ModifierDN => "modifier_dn",
        RegisterKind::ModifierDJ => "modifier_dj",
        RegisterKind::ModifierDC => "modifier_dc",
        RegisterKind::Vector256 => "vector256",
        RegisterKind::Vector512 => "vector512",
        RegisterKind::Vector1024 => "vector1024",
        RegisterKind::Accumulator => "accumulator",
        RegisterKind::Control => "control",
        RegisterKind::SparseQx => "sparse_qx",
        RegisterKind::ScalarPair => "scalar_pair",
    }.to_string())
}

fn imm_scale(ot: &OperandType) -> Option<i32> {
    match ot {
        OperandType::Immediate { scale, .. } => Some(*scale),
        _ => None,
    }
}

fn convert_operand(field: &tablegen::OperandField) -> IsaOperand {
    IsaOperand {
        name: field.name.clone(),
        bit_width: field.width,
        is_output: field.is_output,
        operand_type: operand_type_str(&field.operand_type),
        register_kind: register_kind_str(&field.operand_type),
        signed: field.signed,
        scale: imm_scale(&field.operand_type),
    }
}

fn convert_instruction(enc: &InstrEncoding) -> IsaInstruction {
    IsaInstruction {
        name: enc.name.clone(),
        mnemonic: enc.mnemonic.clone(),
        asm_string: enc.asm_string.clone(),
        slot: enc.slot.clone(),
        width: enc.width,
        operands: enc.operand_fields.iter().map(convert_operand).collect(),
        may_load: enc.may_load,
        may_store: enc.may_store,
        is_vector: enc.is_vector,
        has_complete_decoder: enc.has_complete_decoder,
        sched_class: enc.sched_class.clone(),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let output_path = if let Some(pos) = args.iter().position(|a| a == "--output") {
        args.get(pos + 1)
            .expect("--output requires a path argument")
            .clone()
    } else {
        "tools/aie2-isa.json".to_string()
    };

    eprintln!("Loading TableGen data...");
    let tblgen = tablegen::load_from_generated();

    // Collect all instructions, sorted by slot then name for deterministic output.
    let mut by_slot: BTreeMap<String, Vec<IsaInstruction>> = BTreeMap::new();
    let mut total = 0usize;

    for (slot, encodings) in &tblgen.encodings_by_slot {
        let mut converted: Vec<IsaInstruction> = encodings
            .iter()
            .map(convert_instruction)
            .collect();
        converted.sort_by(|a, b| a.name.cmp(&b.name));
        total += converted.len();
        by_slot.insert(slot.clone(), converted);
    }

    eprintln!("Exporting {} instructions across {} slots to {}",
        total, by_slot.len(), output_path);

    let json = serde_json::to_string_pretty(&by_slot)
        .expect("JSON serialization failed");

    std::fs::write(&output_path, &json)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", output_path, e));

    eprintln!("Done. {} bytes written.", json.len());
}
