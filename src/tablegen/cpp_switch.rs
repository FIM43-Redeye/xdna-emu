//! Parser for C++ instruction selection switches in AIE2InstrInfo.cpp.
//!
//! The LLVM backend selects many instructions via C++ `switch` statements on
//! intrinsic IDs rather than TableGen `Pat<>` records. This module extracts
//! those `intrinsic -> opcode` mappings so the emulator can assign semantics
//! to instructions that have no Pat<> pattern and no pseudo expansion chain.
//!
//! ## Parsed Functions
//!
//! - `getOpCode()`: Main switch mapping ~48 intrinsics to concrete opcodes.
//!   Includes conditional cases (pack/unpack) where we emit all variants.
//! - `getMoveToMSOpcode()`: Stream write variants with TLast parameter.
//!
//! ## Format
//!
//! The switch format is extremely regular:
//! ```cpp
//! case Intrinsic::aie2_acc32_v16_I256_ups:
//!   return AIE2::VUPS_S32_D16_mv_ups_w2b;
//! ```
//!
//! Conditional cases emit multiple opcodes for the same intrinsic:
//! ```cpp
//! case Intrinsic::aie2_pack_I4_I8:
//! case Intrinsic::aie2_pack_I8_I16: {
//!   ...
//!   return isSigned ? AIE2::VPACK_S4_S8 : AIE2::VPACK_D4_D8;
//!   ...
//!   return isSigned ? AIE2::VPACK_S8_S16 : AIE2::VPACK_D8_D16;
//! }
//! ```

use std::collections::HashMap;
use std::path::Path;

/// Extract intrinsic-to-opcode mappings from AIE2InstrInfo.cpp.
///
/// Returns a map from intrinsic stem (e.g., `"acc32_v16_I256_ups"`) to a list
/// of opcode names (e.g., `["VUPS_S32_D16_mv_ups_w2b"]`). For conditional
/// cases, multiple opcodes are listed.
///
/// Returns an empty map if the file cannot be read (graceful fallback).
pub fn parse_cpp_opcode_switch(llvm_aie_path: &Path) -> HashMap<String, Vec<String>> {
    let cpp_path = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2InstrInfo.cpp");
    let content = match std::fs::read_to_string(&cpp_path) {
        Ok(c) => c,
        Err(e) => {
            log::warn!(
                "Could not read AIE2InstrInfo.cpp at {}: {}",
                cpp_path.display(),
                e
            );
            return HashMap::new();
        }
    };

    let mut result = HashMap::new();

    // Parse getOpCode() and getMoveToMSOpcode() functions
    parse_switch_function(&content, "getOpCode", &mut result);
    parse_switch_function(&content, "getMoveToMSOpcode", &mut result);

    log::info!(
        "Parsed {} intrinsic->opcode mappings from AIE2InstrInfo.cpp",
        result.len()
    );

    result
}

/// Parse a single switch function, extracting case/return pairs.
///
/// Handles three patterns:
/// 1. Simple: `case Intrinsic::aie2_X: return AIE2::Y;`
/// 2. Fall-through: multiple `case` lines followed by one `return`
/// 3. Conditional: `case` block with multiple `return AIE2::Y` (ternary)
fn parse_switch_function(
    content: &str,
    function_name: &str,
    result: &mut HashMap<String, Vec<String>>,
) {
    // Find the function by scanning for its signature
    let needle = format!("AIE2InstrInfo::{}", function_name);
    let func_start = match content.find(&needle) {
        Some(pos) => pos,
        None => return,
    };

    // Find the opening brace of the switch
    let switch_start = match content[func_start..].find("switch") {
        Some(pos) => func_start + pos,
        None => return,
    };

    // Extract the switch body by brace counting from the switch's opening brace
    let switch_body = match extract_brace_block(&content[switch_start..]) {
        Some(body) => body,
        None => return,
    };

    // Parse case/return pairs from the switch body.
    //
    // Simple cases: `case X: return Y;` -- pending cleared after return.
    // Fall-through: `case X: case Y: return Z;` -- both get Z, then cleared.
    // Conditional: `case X: case Y: { ... return A; ... return B; }` --
    //   both get A and B, cleared on closing brace.
    let mut pending_intrinsics: Vec<String> = Vec::new();
    let mut in_conditional_block = false;

    for line in switch_body.lines() {
        let trimmed = line.trim();

        // Match: case Intrinsic::aie2_XXX:
        if let Some(stem) = extract_intrinsic_stem(trimmed) {
            pending_intrinsics.push(stem);
            // Detect conditional block: `case X: {` or `case X: case Y: {`
            if trimmed.ends_with('{') {
                in_conditional_block = true;
            }
            continue;
        }

        // Standalone opening brace after case labels
        if trimmed == "{" && !pending_intrinsics.is_empty() {
            in_conditional_block = true;
            continue;
        }

        // Match: return AIE2::OPCODE_NAME;
        // Also match ternary: return cond ? AIE2::A : AIE2::B;
        let opcodes = extract_opcodes(trimmed);
        if !opcodes.is_empty() && !pending_intrinsics.is_empty() {
            for intrinsic in &pending_intrinsics {
                let entry = result.entry(intrinsic.clone()).or_insert_with(Vec::new);
                for opcode in &opcodes {
                    if !entry.contains(opcode) {
                        entry.push(opcode.clone());
                    }
                }
            }
            // In a simple case, clear pending after the return.
            // In a conditional block, keep pending until the closing brace
            // so multiple returns all contribute opcodes.
            if !in_conditional_block {
                pending_intrinsics.clear();
            }
        }

        // Closing brace ends a conditional block
        if trimmed == "}" {
            if in_conditional_block {
                in_conditional_block = false;
            }
            pending_intrinsics.clear();
        }

        // default/break clears any pending state
        if trimmed.starts_with("default:") || trimmed == "break;" {
            pending_intrinsics.clear();
            in_conditional_block = false;
        }
    }
}

/// Extract the intrinsic stem from a case label.
///
/// Input: `"case Intrinsic::aie2_acc32_v16_I256_ups:"`
/// Output: `Some("acc32_v16_I256_ups")`
fn extract_intrinsic_stem(line: &str) -> Option<String> {
    let line = line.strip_prefix("case ")?;
    let line = line.strip_prefix("Intrinsic::aie2_")?;
    // Strip trailing colon (and possible '{' for conditional blocks)
    let stem = line.trim_end_matches(|c: char| c == ':' || c == ' ' || c == '{');
    if stem.is_empty() {
        return None;
    }
    Some(stem.to_string())
}

/// Extract all AIE2:: opcode names from a return statement.
///
/// Handles:
/// - `return AIE2::VUPS_S32_D16_mv_ups_w2b;` -> `["VUPS_S32_D16_mv_ups_w2b"]`
/// - `return cond ? AIE2::A : AIE2::B;` -> `["A", "B"]`
fn extract_opcodes(line: &str) -> Vec<String> {
    let mut opcodes = Vec::new();

    // Find all AIE2::NAME occurrences
    let mut search_from = 0;
    while let Some(pos) = line[search_from..].find("AIE2::") {
        let start = search_from + pos + 6; // skip "AIE2::"
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

/// Extract the contents of a brace-delimited block.
///
/// Starts scanning from the first `{` found in `text` and returns everything
/// between the matching braces (exclusive).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_intrinsic_stem() {
        assert_eq!(
            extract_intrinsic_stem("case Intrinsic::aie2_acc32_v16_I256_ups:"),
            Some("acc32_v16_I256_ups".to_string())
        );
        assert_eq!(
            extract_intrinsic_stem("case Intrinsic::aie2_pack_I4_I8:"),
            Some("pack_I4_I8".to_string())
        );
        // Not an intrinsic case
        assert_eq!(extract_intrinsic_stem("case TargetOpcode::G_LOAD:"), None);
        assert_eq!(extract_intrinsic_stem("default:"), None);
    }

    #[test]
    fn test_extract_opcodes_simple() {
        assert_eq!(
            extract_opcodes("    return AIE2::VUPS_S32_D16_mv_ups_w2b;"),
            vec!["VUPS_S32_D16_mv_ups_w2b"]
        );
    }

    #[test]
    fn test_extract_opcodes_ternary() {
        assert_eq!(
            extract_opcodes("      return isSigned ? AIE2::VPACK_S4_S8 : AIE2::VPACK_D4_D8;"),
            vec!["VPACK_S4_S8", "VPACK_D4_D8"]
        );
    }

    #[test]
    fn test_extract_opcodes_none() {
        assert!(extract_opcodes("    bool isSigned = Sign && Sign->Value;").is_empty());
    }

    #[test]
    fn test_extract_brace_block() {
        let text = "switch (x) { case 1: return 2; }";
        assert_eq!(
            extract_brace_block(text),
            Some(" case 1: return 2; ")
        );
    }

    #[test]
    fn test_parse_switch_function_simple() {
        let content = r#"
unsigned AIE2InstrInfo::getOpCode(MachineInstr &I) const {
  unsigned IntrinsicID = cast<GIntrinsic>(I).getIntrinsicID();
  switch (IntrinsicID) {
  case Intrinsic::aie2_acc32_v16_I256_ups:
    return AIE2::VUPS_S32_D16_mv_ups_w2b;
  case Intrinsic::aie2_vmax_lt8:
    return AIE2::VMAX_LT_D8;
  default:
    llvm_unreachable("Unexpected Intrinsic ID");
  }
}
"#;
        let mut result = HashMap::new();
        parse_switch_function(content, "getOpCode", &mut result);
        assert_eq!(
            result.get("acc32_v16_I256_ups"),
            Some(&vec!["VUPS_S32_D16_mv_ups_w2b".to_string()])
        );
        assert_eq!(
            result.get("vmax_lt8"),
            Some(&vec!["VMAX_LT_D8".to_string()])
        );
    }

    #[test]
    fn test_parse_switch_function_fallthrough() {
        let content = r#"
unsigned AIE2InstrInfo::getOpCode(MachineInstr &I) const {
  switch (IntrinsicID) {
  case Intrinsic::aie2_scd_read_vec:
  case Intrinsic::aie2_scd_read_acc32:
    return AIE2::VMOV_mv_scd;
  default:
    break;
  }
}
"#;
        let mut result = HashMap::new();
        parse_switch_function(content, "getOpCode", &mut result);
        assert_eq!(
            result.get("scd_read_vec"),
            Some(&vec!["VMOV_mv_scd".to_string()])
        );
        assert_eq!(
            result.get("scd_read_acc32"),
            Some(&vec!["VMOV_mv_scd".to_string()])
        );
    }

    #[test]
    fn test_parse_switch_function_conditional() {
        let content = r#"
unsigned AIE2InstrInfo::getOpCode(MachineInstr &I) const {
  switch (IntrinsicID) {
  case Intrinsic::aie2_pack_I4_I8:
  case Intrinsic::aie2_pack_I8_I16: {
    bool isSigned = true;
    if (IntrinsicID == Intrinsic::aie2_pack_I4_I8)
      return isSigned ? AIE2::VPACK_S4_S8 : AIE2::VPACK_D4_D8;
    else
      return isSigned ? AIE2::VPACK_S8_S16 : AIE2::VPACK_D8_D16;
  }
  default:
    break;
  }
}
"#;
        let mut result = HashMap::new();
        parse_switch_function(content, "getOpCode", &mut result);
        // Both intrinsics should get all four opcodes
        let pack_i4 = result.get("pack_I4_I8").unwrap();
        assert!(pack_i4.contains(&"VPACK_S4_S8".to_string()));
        assert!(pack_i4.contains(&"VPACK_D4_D8".to_string()));
        assert!(pack_i4.contains(&"VPACK_S8_S16".to_string()));
        assert!(pack_i4.contains(&"VPACK_D8_D16".to_string()));
    }

    #[test]
    fn test_parse_from_real_file() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        let map = parse_cpp_opcode_switch(llvm_aie_path);

        // Should have extracted a reasonable number of mappings
        assert!(
            map.len() >= 30,
            "Expected at least 30 intrinsic mappings, got {}",
            map.len()
        );

        // Spot-check known mappings
        let ups = map.get("acc32_v16_I256_ups");
        assert!(ups.is_some(), "UPS mapping should be present");
        assert!(
            ups.unwrap().contains(&"VUPS_S32_D16_mv_ups_w2b".to_string()),
            "UPS should contain VUPS_S32_D16_mv_ups_w2b"
        );

        let vmax = map.get("vmax_lt8");
        assert!(vmax.is_some(), "VMAX mapping should be present");
        assert!(
            vmax.unwrap().contains(&"VMAX_LT_D8".to_string()),
            "VMAX should contain VMAX_LT_D8"
        );

        // Pack should have multiple opcodes (conditional)
        let pack = map.get("pack_I4_I8");
        assert!(pack.is_some(), "pack_I4_I8 should be present");
        assert!(
            pack.unwrap().len() >= 2,
            "pack_I4_I8 should have multiple opcodes"
        );

        // getMoveToMSOpcode should also be parsed
        assert!(
            map.get("put_ms").is_some(),
            "put_ms from getMoveToMSOpcode should be present"
        );
    }
}
