//! TableGen .td file extraction for compile-time architecture constants.
//!
//! Parses slot definitions from llvm-aie's `AIE2Slots.td` to extract VLIW
//! slot widths at build time. This is the authoritative source for slot widths;
//! manual constants in `populate_aie2_manual_constants()` serve as a
//! cross-validation baseline via `Confirmed<T>`.
//!
//! The parser is intentionally minimal -- no regex dependency, just enough
//! string parsing to extract `def NAME : InstSlot<"Display", BITS>` blocks.
//! The full instruction encoding parser (for decoder tables) stays in the
//! main crate's `src/tablegen/` module at runtime.

use std::path::Path;

/// A VLIW slot definition extracted from TableGen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotExtract {
    /// Internal name (e.g., "lda_slot").
    pub name: String,
    /// Display/field name (e.g., "lda").
    pub field: String,
    /// Bit width of this slot.
    pub bits: u8,
    /// Whether this is an artificial slot (e.g., nop_slot for alignment).
    pub artificial: bool,
}

/// Extract slot definitions from AIE2Slots.td content.
///
/// Parses all `def NAME : InstSlot<"Display", BITS> { ... }` blocks.
/// Returns one `SlotExtract` per non-artificial slot definition.
///
/// The parser handles the specific format used in llvm-aie:
/// ```text
/// def lda_slot : InstSlot<"Lda", 21> {
///     let FieldToFind = "lda";
///     let Artificial = true;  // optional
/// }
/// ```
pub fn extract_slots(content: &str) -> Vec<SlotExtract> {
    let mut slots = Vec::new();

    // Scan for `def NAME : InstSlot<"Display", BITS>` patterns.
    // We iterate line by line looking for the `def ... InstSlot` pattern,
    // then collect the block contents until the closing brace.
    let mut lines = content.lines().peekable();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();

        // Look for: def NAME : InstSlot<"Display", BITS>
        if !trimmed.starts_with("def ") {
            continue;
        }
        let Some((name, bits)) = parse_inst_slot_header(trimmed) else {
            continue;
        };

        // Collect block body (everything between { and })
        let mut block_body = String::new();
        let has_open_brace = trimmed.contains('{');

        if has_open_brace {
            // Opening brace on same line -- extract content after it
            if let Some(after_brace) = trimmed.split('{').nth(1) {
                let body = after_brace.trim_end_matches('}').trim();
                if !body.is_empty() {
                    block_body.push_str(body);
                    block_body.push('\n');
                }
            }
            // If closing brace was also on same line, we're done
            if trimmed.ends_with('}') {
                let (field, artificial) = parse_block_body(&block_body, &name);
                slots.push(SlotExtract {
                    name,
                    field,
                    bits,
                    artificial,
                });
                continue;
            }
        }

        // Read until closing brace
        for body_line in lines.by_ref() {
            let body_trimmed = body_line.trim();
            if body_trimmed == "}" || body_trimmed.ends_with('}') {
                // Include content before the closing brace
                let before_close = body_trimmed.trim_end_matches('}').trim();
                if !before_close.is_empty() {
                    block_body.push_str(before_close);
                    block_body.push('\n');
                }
                break;
            }
            if body_trimmed.contains('{') && !has_open_brace {
                // Opening brace on its own line -- skip it
                continue;
            }
            block_body.push_str(body_trimmed);
            block_body.push('\n');
        }

        let (field, artificial) = parse_block_body(&block_body, &name);
        slots.push(SlotExtract {
            name,
            field,
            bits,
            artificial,
        });
    }

    slots
}

/// Parse the `def NAME : InstSlot<"Display", BITS>` header line.
/// Returns (name, bits) on success.
fn parse_inst_slot_header(line: &str) -> Option<(String, u8)> {
    // Expected: "def NAME : InstSlot<\"Display\", BITS> {"
    // or:       "def NAME     : InstSlot<\"Display\", BITS> {"
    let rest = line.strip_prefix("def ")?.trim_start();

    // Extract name (up to whitespace or colon)
    let name_end = rest.find(|c: char| c.is_whitespace() || c == ':')?;
    let name = rest[..name_end].trim().to_string();

    // Find InstSlot< marker
    let inst_slot_pos = rest.find("InstSlot<")?;
    let after_marker = &rest[inst_slot_pos + "InstSlot<".len()..];

    // Extract: "Display", BITS>
    let closing = after_marker.find('>')?;
    let params = &after_marker[..closing];

    // Split on comma: "Display", BITS
    let comma = params.find(',')?;
    let bits_str = params[comma + 1..].trim();
    let bits: u8 = bits_str.parse().ok()?;

    Some((name, bits))
}

/// Parse the block body for FieldToFind and Artificial fields.
/// Returns (field_name, is_artificial).
fn parse_block_body(body: &str, slot_name: &str) -> (String, bool) {
    let mut field = None;
    let mut artificial = false;

    for line in body.lines() {
        let trimmed = line.trim();

        // Look for: let FieldToFind = "name";
        if let Some(rest) = trimmed.strip_prefix("let FieldToFind") {
            let rest = rest.trim_start().strip_prefix('=').unwrap_or(rest);
            let rest = rest.trim();
            // Extract quoted string
            if let Some(start) = rest.find('"') {
                if let Some(end) = rest[start + 1..].find('"') {
                    field = Some(rest[start + 1..start + 1 + end].to_string());
                }
            }
        }

        // Look for: let Artificial = true;
        if trimmed.starts_with("let Artificial") && trimmed.contains("true") {
            artificial = true;
        }
    }

    // Default field name: strip "_slot" suffix from the slot name
    let field = field.unwrap_or_else(|| {
        slot_name
            .strip_suffix("_slot")
            .unwrap_or(slot_name)
            .to_string()
    });

    (field, artificial)
}

/// Read AIE2Slots.td from the llvm-aie directory and extract slot definitions.
///
/// Returns only non-artificial slots (the real VLIW execution slots).
///
/// # Arguments
/// * `llvm_aie_path` -- Root of the llvm-aie repository (e.g., `../llvm-aie`)
pub fn extract_slots_from_llvm_aie(
    llvm_aie_path: &Path,
) -> Result<Vec<SlotExtract>, String> {
    let slots_td = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2Slots.td");

    if !slots_td.exists() {
        return Err(format!(
            "AIE2Slots.td not found at {}",
            slots_td.display()
        ));
    }

    let content = std::fs::read_to_string(&slots_td)
        .map_err(|e| format!("Failed to read {}: {}", slots_td.display(), e))?;

    let all_slots = extract_slots(&content);

    // Filter to non-artificial slots only (the real execution slots)
    let real_slots: Vec<_> = all_slots.into_iter().filter(|s| !s.artificial).collect();

    if real_slots.is_empty() {
        return Err(format!(
            "No real slot definitions found in {}",
            slots_td.display()
        ));
    }

    Ok(real_slots)
}

/// Cross-validate the ProcessorModel's slot widths against llvm-aie TableGen.
///
/// Reads AIE2Slots.td and confirms each slot width against the existing
/// manual constant. Panics (via `Confirmed::confirm()`) on mismatch.
///
/// If the model has no ProcessorModel yet, this sets one from the extracted
/// data (slots only; other fields like register sizes still require manual
/// population since they come from different .td files).
///
/// # Arguments
/// * `model` -- The ArchModel to validate (must already have ProcessorModel)
/// * `llvm_aie_path` -- Root of the llvm-aie repository
pub fn confirm_processor_slots(
    model: &mut crate::types::ArchModel,
    llvm_aie_path: &Path,
) -> Result<usize, String> {
    let slots = extract_slots_from_llvm_aie(llvm_aie_path)?;

    let proc = model.processor.as_ref().ok_or_else(|| {
        "ArchModel has no ProcessorModel -- call populate_manual_constants() first".to_string()
    })?;

    let tg_src = crate::types::SourceAttribution {
        origin: crate::types::Source::TableGen,
        file: format!(
            "{}/llvm/lib/Target/AIE/AIE2Slots.td",
            llvm_aie_path.display()
        ),
        detail: "InstSlot<> definitions, compile-time extraction".into(),
    };

    // Cross-validate: for each extracted slot, find the matching manual entry
    // and confirm the width matches.
    let mut confirmed_count = 0;
    let mut errors = Vec::new();

    for slot in &slots {
        let manual_entry = proc.slot_widths.iter().find(|(name, _)| *name == slot.field);
        match manual_entry {
            Some((_name, manual_width)) => {
                if *manual_width != slot.bits {
                    errors.push(format!(
                        "Slot '{}': manual={} bits, TableGen={} bits",
                        slot.field, manual_width, slot.bits
                    ));
                } else {
                    confirmed_count += 1;
                }
            }
            None => {
                errors.push(format!(
                    "Slot '{}' ({} bits) found in TableGen but not in manual ProcessorModel",
                    slot.field, slot.bits
                ));
            }
        }
    }

    // Also check for manual entries not in TableGen
    for (name, _width) in &proc.slot_widths {
        if !slots.iter().any(|s| s.field == *name) {
            errors.push(format!(
                "Slot '{}' in manual ProcessorModel but not found in TableGen",
                name
            ));
        }
    }

    if !errors.is_empty() {
        return Err(format!(
            "TableGen slot validation failed ({} errors):\n  {}",
            errors.len(),
            errors.join("\n  ")
        ));
    }

    // All slots confirmed -- record the TableGen source attribution.
    // We update the ProcessorModel's source to note TableGen confirmation.
    if let Some(ref mut proc) = model.processor {
        proc.source = tg_src;
    }

    Ok(confirmed_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIE2_SLOTS_TD: &str = r#"
let Namespace = "AIE2" in
{
  def lda_slot     : InstSlot<"Lda", 21> {
    let FieldToFind = "lda";
  }

  def ldb_slot     : InstSlot<"Ldb", 16> {
    let FieldToFind = "ldb";
  }

  def alu_slot     : InstSlot<"Alu", 20> {
    let FieldToFind = "alu";
  }

  def mv_slot     : InstSlot<"Mv", 22> {
    let FieldToFind = "mv";
  }

  def st_slot      : InstSlot<"St", 21> {
    let FieldToFind = "st";
  }

  def vec_slot : InstSlot<"Vec", 26> {
    let FieldToFind = "vec";
  }

  def lng_slot     : InstSlot<"Lng", 42> {
    let FieldToFind = "lng";
  }

  def nop_slot     : InstSlot<"Nop", 1> {
    let FieldToFind = "nop";
    let Artificial = true;
  }
}
"#;

    #[test]
    fn extract_all_aie2_slots() {
        let slots = extract_slots(AIE2_SLOTS_TD);

        assert_eq!(slots.len(), 8, "Expected 8 slots (7 real + 1 artificial)");

        // Verify each slot
        let find = |field: &str| slots.iter().find(|s| s.field == field).unwrap();

        assert_eq!(find("lda").bits, 21);
        assert_eq!(find("lda").name, "lda_slot");
        assert!(!find("lda").artificial);

        assert_eq!(find("ldb").bits, 16);
        assert_eq!(find("alu").bits, 20);
        assert_eq!(find("mv").bits, 22);
        assert_eq!(find("st").bits, 21);
        assert_eq!(find("vec").bits, 26);
        assert_eq!(find("lng").bits, 42);

        assert_eq!(find("nop").bits, 1);
        assert!(find("nop").artificial);
    }

    #[test]
    fn extract_filters_artificial() {
        let all = extract_slots(AIE2_SLOTS_TD);
        let real: Vec<_> = all.into_iter().filter(|s| !s.artificial).collect();

        assert_eq!(real.len(), 7);
        assert!(real.iter().all(|s| !s.artificial));
        assert!(!real.iter().any(|s| s.field == "nop"));
    }

    #[test]
    fn parse_minimal_slot() {
        let content = r#"def test_slot : InstSlot<"Test", 32> {
    let FieldToFind = "test";
}"#;
        let slots = extract_slots(content);
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].name, "test_slot");
        assert_eq!(slots[0].field, "test");
        assert_eq!(slots[0].bits, 32);
        assert!(!slots[0].artificial);
    }

    #[test]
    fn parse_slot_without_field_to_find() {
        // If FieldToFind is missing, derive from name by stripping _slot
        let content = r#"def foo_slot : InstSlot<"Foo", 10> {
}"#;
        let slots = extract_slots(content);
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].field, "foo");
    }

    #[test]
    fn empty_content_returns_empty() {
        let slots = extract_slots("");
        assert!(slots.is_empty());
    }

    #[test]
    fn non_slot_defs_ignored() {
        let content = r#"
def SomeOtherThing : Instruction<"foo"> {}
class NotASlot<int x> {}
def lda_slot : InstSlot<"Lda", 21> {
    let FieldToFind = "lda";
}
"#;
        let slots = extract_slots(content);
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].field, "lda");
    }

    #[test]
    fn confirm_processor_slots_matches() {
        // Build a model with matching manual constants
        use crate::types::*;
        let src = SourceAttribution {
            origin: Source::Am020,
            file: "test".into(),
            detail: "manual".into(),
        };
        let mut model = ArchModel {
            arch: Architecture::Aie2,
            generation: None,
            device_id: None,
            is_npu: true,
            tile_types: vec![],
            relationships: vec![],
            device_constants: None,
            array_topology: None,
            timing: None,
            packet: None,
            processor: Some(ProcessorModel {
                slot_widths: vec![
                    ("lda".into(), 21),
                    ("ldb".into(), 16),
                    ("alu".into(), 20),
                    ("mv".into(), 22),
                    ("st".into(), 21),
                    ("vec".into(), 26),
                    ("lng".into(), 42),
                ],
                vector_register_bits: 512,
                vector_pair_bits: 1024,
                accumulator_bits: 512,
                branch_delay_slots: 5,
                partial_store_data_latency: 6,
                srs_shift_bias: 4,
                source: src,
            }),
        };

        // This should succeed if llvm-aie exists at the expected path
        let llvm_aie = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("llvm-aie");
        if llvm_aie.join("llvm/lib/Target/AIE/AIE2Slots.td").exists() {
            let count = confirm_processor_slots(&mut model, &llvm_aie).unwrap();
            assert_eq!(count, 7, "Should confirm all 7 real slots");
            assert_eq!(
                model.processor.as_ref().unwrap().source.origin,
                Source::TableGen
            );
        }
    }
}
