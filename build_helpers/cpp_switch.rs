//! Parser for C++ instruction selection switches in AIE2InstrInfo.cpp.
//!
//! Build-time version of `src/tablegen/cpp_switch.rs`. Pure string parsing,
//! no external dependencies.

use std::collections::HashMap;
use std::path::Path;

/// Extract intrinsic-to-opcode mappings from AIE2InstrInfo.cpp.
///
/// Returns a map from intrinsic stem (e.g., `"acc32_v16_I256_ups"`) to a list
/// of opcode names (e.g., `["VUPS_S32_D16_mv_ups_w2b"]`).
pub fn parse_cpp_opcode_switch(llvm_aie_path: &Path) -> HashMap<String, Vec<String>> {
    let cpp_path = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2InstrInfo.cpp");
    let content = match std::fs::read_to_string(&cpp_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "cargo:warning=Could not read AIE2InstrInfo.cpp at {}: {}",
                cpp_path.display(),
                e
            );
            return HashMap::new();
        }
    };

    let mut result = HashMap::new();
    parse_switch_function(&content, "getOpCode", &mut result);
    parse_switch_function(&content, "getMoveToMSOpcode", &mut result);
    result
}

/// Parse a single switch function, extracting case/return pairs.
fn parse_switch_function(
    content: &str,
    function_name: &str,
    result: &mut HashMap<String, Vec<String>>,
) {
    let needle = format!("AIE2InstrInfo::{}", function_name);
    let func_start = match content.find(&needle) {
        Some(pos) => pos,
        None => return,
    };

    let switch_start = match content[func_start..].find("switch") {
        Some(pos) => func_start + pos,
        None => return,
    };

    let switch_body = match extract_brace_block(&content[switch_start..]) {
        Some(body) => body,
        None => return,
    };

    let mut pending_intrinsics: Vec<String> = Vec::new();
    let mut in_conditional_block = false;

    for line in switch_body.lines() {
        let trimmed = line.trim();

        if let Some(stem) = extract_intrinsic_stem(trimmed) {
            pending_intrinsics.push(stem);
            if trimmed.ends_with('{') {
                in_conditional_block = true;
            }
            continue;
        }

        if trimmed == "{" && !pending_intrinsics.is_empty() {
            in_conditional_block = true;
            continue;
        }

        let opcodes = extract_opcodes(trimmed);
        if !opcodes.is_empty() && !pending_intrinsics.is_empty() {
            for intrinsic in &pending_intrinsics {
                let entry = result.entry(intrinsic.clone()).or_default();
                for opcode in &opcodes {
                    if !entry.contains(opcode) {
                        entry.push(opcode.clone());
                    }
                }
            }
            if !in_conditional_block {
                pending_intrinsics.clear();
            }
        }

        if trimmed == "}" {
            if in_conditional_block {
                in_conditional_block = false;
            }
            pending_intrinsics.clear();
        }

        if trimmed.starts_with("default:") || trimmed == "break;" {
            pending_intrinsics.clear();
            in_conditional_block = false;
        }
    }
}

fn extract_intrinsic_stem(line: &str) -> Option<String> {
    let line = line.strip_prefix("case ")?;
    let line = line.strip_prefix("Intrinsic::aie2_")?;
    let stem = line.trim_end_matches(|c: char| c == ':' || c == ' ' || c == '{');
    if stem.is_empty() {
        return None;
    }
    Some(stem.to_string())
}

fn extract_opcodes(line: &str) -> Vec<String> {
    let mut opcodes = Vec::new();
    let mut search_from = 0;
    while let Some(pos) = line[search_from..].find("AIE2::") {
        let start = search_from + pos + 6;
        let end = line[start..]
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .map_or(line.len(), |e| start + e);
        let name = &line[start..end];
        if !name.is_empty() {
            opcodes.push(name.to_string());
        }
        search_from = end;
    }
    opcodes
}

fn extract_brace_block(text: &str) -> Option<&str> {
    let open = text.find('{')?;
    let mut depth = 0;
    let bytes = text.as_bytes();
    for i in open..bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[open + 1..i]);
                }
            }
            _ => {}
        }
    }
    None
}
