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

use super::resolver::{InstrEncoding, OperandField, infer_semantic_from_mnemonic};
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

    /// Extract operand fields from the encoding.
    ///
    /// Groups consecutive field bit references into OperandField entries
    /// with their bit positions and widths.
    pub fn extract_operand_fields(&self) -> Vec<OperandField> {
        let mut fields: HashMap<String, (u8, u8)> = HashMap::new(); // name -> (low_bit, high_bit)

        for (i, part) in self.parts.iter().enumerate() {
            if let EncodingBit::FieldBit { field, bit: _ } = part {
                let bit_pos = self.width as usize - 1 - i;
                let entry = fields.entry(field.clone()).or_insert((bit_pos as u8, bit_pos as u8));
                // Expand range
                if (bit_pos as u8) < entry.0 {
                    entry.0 = bit_pos as u8;
                }
                if (bit_pos as u8) > entry.1 {
                    entry.1 = bit_pos as u8;
                }
            }
        }

        fields
            .into_iter()
            .map(|(name, (low, high))| OperandField::new(name, low, high - low + 1))
            .collect()
    }
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
    pub fn to_encoding(&self) -> Option<InstrEncoding> {
        let enc = self.slot_encoding.as_ref()?;
        let (fixed_mask, fixed_bits) = enc.compute_fixed_bits();
        let operand_fields = enc.extract_operand_fields();

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

        // Try to infer semantic from mnemonic
        let semantic = infer_semantic_from_mnemonic(&self.mnemonic);

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
        })
    }
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
