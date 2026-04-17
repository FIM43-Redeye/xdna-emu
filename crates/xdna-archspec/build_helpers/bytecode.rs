//! LLVM decoder table bytecode parser (build-time).
//!
//! Extracts decoder bytecode tables from `llvm-tblgen -gen-disassembler` output.
//! The runtime bytecode interpreter stays in `src/tablegen/decoder_bytecode.rs`.

use std::collections::HashMap;

/// LLVM decoder bytecode opcodes (from MCDecoderOps.h).
mod opc {
    pub const EXTRACT_FIELD: u8 = 1;
    pub const FILTER_VALUE: u8 = 2;
    pub const CHECK_FIELD: u8 = 3;
    pub const CHECK_PREDICATE: u8 = 4;
    pub const DECODE: u8 = 5;
    pub const TRY_DECODE: u8 = 6;
    pub const SOFT_FAIL: u8 = 7;
    pub const FAIL: u8 = 8;
}

/// A decoder table extracted at build time.
pub struct BuildDecoderTable {
    pub bytes: Vec<u8>,
    pub opcode_names: HashMap<u32, String>,
}

/// Read a ULEB128-encoded value from a byte slice.
fn read_uleb128(bytes: &[u8], ptr: &mut usize) -> u32 {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = bytes[*ptr];
        *ptr += 1;
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

/// Parse opcode name mapping from comments in LLVM's disassembler output.
pub fn parse_opcode_names_from_disasm(disasm_output: &str) -> HashMap<u32, String> {
    let mut names: HashMap<u32, String> = HashMap::new();

    for line in disasm_output.lines() {
        let line = line.trim();

        let (is_decode, is_try_decode) = (
            line.contains("MCD::OPC_Decode,"),
            line.contains("MCD::OPC_TryDecode,"),
        );
        if !is_decode && !is_try_decode {
            continue;
        }

        let name = if let Some(idx) = line.find("// Opcode: ") {
            let rest = &line[idx + 11..];
            let end = rest
                .find(|c: char| c == ',' || c == '\n')
                .unwrap_or(rest.len());
            rest[..end].trim().to_string()
        } else {
            continue;
        };

        let marker = if is_decode {
            "MCD::OPC_Decode,"
        } else {
            "MCD::OPC_TryDecode,"
        };
        let after_marker = if let Some(idx) = line.find(marker) {
            &line[idx + marker.len()..]
        } else {
            continue;
        };

        let bytes_str = if let Some(comment_idx) = after_marker.find("//") {
            &after_marker[..comment_idx]
        } else {
            after_marker
        };

        let byte_values: Vec<u8> = bytes_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u8>().ok())
            .collect();

        if byte_values.is_empty() {
            continue;
        }

        let mut ptr = 0;
        let opcode_id = read_uleb128(&byte_values, &mut ptr);
        names.insert(opcode_id, name);
    }

    names
}

/// Parse a single decoder table array from the C++ output.
pub fn parse_decoder_table_bytes(table_text: &str) -> Vec<u8> {
    let mut bytes = Vec::new();

    for line in table_text.lines() {
        let line = line.trim();

        if line.is_empty()
            || line == "{"
            || line.starts_with("};")
            || line.starts_with("static ")
        {
            continue;
        }

        let data_part = if let Some(idx) = line.find("*/") {
            &line[idx + 2..]
        } else {
            line
        };

        let data_part = if let Some(idx) = data_part.find("//") {
            &data_part[..idx]
        } else {
            data_part
        };

        for token in data_part.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }

            if let Some(val) = match_mcd_opcode(token) {
                bytes.push(val);
            } else if let Ok(val) = token.parse::<u8>() {
                bytes.push(val);
            }
        }
    }

    bytes
}

/// Map MCD::OPC_xxx symbolic names to their numeric values.
fn match_mcd_opcode(token: &str) -> Option<u8> {
    match token {
        "MCD::OPC_ExtractField" => Some(opc::EXTRACT_FIELD),
        "MCD::OPC_FilterValue" => Some(opc::FILTER_VALUE),
        "MCD::OPC_CheckField" => Some(opc::CHECK_FIELD),
        "MCD::OPC_CheckPredicate" => Some(opc::CHECK_PREDICATE),
        "MCD::OPC_Decode" => Some(opc::DECODE),
        "MCD::OPC_TryDecode" => Some(opc::TRY_DECODE),
        "MCD::OPC_SoftFail" => Some(opc::SOFT_FAIL),
        "MCD::OPC_Fail" => Some(opc::FAIL),
        _ => None,
    }
}

/// Extract all decoder tables from the full `-gen-disassembler` output.
pub fn extract_all_tables(disasm_output: &str) -> HashMap<String, BuildDecoderTable> {
    let mut result = HashMap::new();

    let slot_map: &[(&str, &str)] = &[
        ("DecoderTableAlu32", "alu"),
        ("DecoderTableLda32", "lda"),
        ("DecoderTableLdb32", "ldb"),
        ("DecoderTableMv32", "mv"),
        ("DecoderTableSt32", "st"),
        ("DecoderTableVec32", "vec"),
        ("DecoderTableNop16", "nop"),
        ("DecoderTableLng48", "lng"),
    ];

    for (table_name, slot_name) in slot_map {
        let search = format!("static const uint8_t {}[]", table_name);
        if let Some(start_idx) = disasm_output.find(&search) {
            let rest = &disasm_output[start_idx..];
            if let Some(end_offset) = find_table_end(rest) {
                let table_text = &rest[..end_offset];
                let bytes = parse_decoder_table_bytes(table_text);
                let opcode_names = parse_opcode_names_from_disasm(table_text);

                if !bytes.is_empty() {
                    result.insert(
                        slot_name.to_string(),
                        BuildDecoderTable {
                            bytes,
                            opcode_names,
                        },
                    );
                }
            }
        }
    }

    result
}

/// Find the end of a C array definition (`};`).
fn find_table_end(text: &str) -> Option<usize> {
    let mut offset = 0;
    for line in text.lines() {
        let next_offset = offset + line.len() + 1;
        let trimmed = line.trim();
        if trimmed == "};" || trimmed.starts_with("};") {
            return Some(next_offset);
        }
        offset = next_offset;
    }
    None
}
