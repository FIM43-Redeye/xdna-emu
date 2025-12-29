//! Regex-based TableGen file parser.
//!
//! Parses LLVM TableGen (.td) files to extract instruction definitions.
//! This is a simplified parser that handles the specific patterns used
//! in the AIE2 instruction definitions from llvm-aie.
//!
//! ## Extended Parsing
//!
//! Beyond basic definitions, we also extract:
//! - **Attributes**: `mayLoad`, `mayStore`, `hasSideEffects`, `Defs`, `Uses`
//! - **Patterns**: `Pat<(add ...), (ADD ...)>` linking SDNodes to instructions

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use super::types::{
    EncodingPart, FormatClass, InstrAttributes, InstrDef, OperandDef, SemanticOp,
    SemanticPattern, SlotDef, TableGenData, TemplateParam,
};

/// Compiled regex patterns for TableGen parsing.
struct Patterns {
    /// Matches slot definitions: `def lda_slot : InstSlot<"Lda", 21>`
    slot_def: Regex,
    /// Matches FieldToFind: `let FieldToFind = "lda";`
    field_to_find: Regex,
    /// Matches Artificial: `let Artificial = true;`
    artificial: Regex,
    /// Matches class definitions: `class AIE2_foo<bits<4> op, ...> : Parent<...> {`
    class_def: Regex,
    /// Matches template params: `bits<4> op`
    template_param: Regex,
    /// Matches field declarations: `bits<5> mRx;` or `bits<5> mRx, mRy;`
    field_decl: Regex,
    /// Matches encoding statements: `let alu = {mRx0, mRx, op, 0b1};`
    encoding_stmt: Regex,
    /// Matches instruction defs: `def ADD : Format<args>;`
    instr_def: Regex,
    /// Matches binary literals: `0b0000`
    binary_literal: Regex,
    /// Matches outs dag: `(outs eR:$mRx)`
    outs_dag: Regex,
    /// Matches ins dag: `(ins eR:$mRx0, eR:$mRy)`
    ins_dag: Regex,
    /// Matches operand in dag: `eR:$mRx`
    operand: Regex,
    /// Matches bit slice: `i{10-6}` or `dontcare{5}`
    bit_slice: Regex,

    // Attribute patterns
    /// Matches mayLoad: `mayLoad = true`
    may_load: Regex,
    /// Matches mayStore: `mayStore = true`
    may_store: Regex,
    /// Matches hasSideEffects: `hasSideEffects = true`
    has_side_effects: Regex,
    /// Matches Defs: `Defs = [srCarry, srMS0]`
    defs: Regex,
    /// Matches Uses: `Uses = [srCarry]`
    uses: Regex,
    /// Matches register list contents: `[srCarry, srMS0]`
    reg_list: Regex,

    // Pattern patterns
    /// Matches simple Pat: `Pat<(sdnode ...), (INSTR ...)>`
    simple_pat: Regex,
    /// Matches PatGprGpr: `PatGprGpr<sdnode, INSTR, type>`
    pat_gpr_gpr: Regex,
    /// Matches PatGpr: `PatGpr<sdnode, INSTR, type>`
    pat_gpr: Regex,
}

static PATTERNS: LazyLock<Patterns> = LazyLock::new(|| Patterns {
    slot_def: Regex::new(r#"def\s+(\w+)\s*:\s*InstSlot<"(\w+)",\s*(\d+)>"#).unwrap(),
    field_to_find: Regex::new(r#"let\s+FieldToFind\s*=\s*"(\w+)"\s*;"#).unwrap(),
    artificial: Regex::new(r"let\s+Artificial\s*=\s*true\s*;").unwrap(),
    // Note: Class definitions can span multiple lines
    // Pattern: class Name<params> : ParentName<args> { body }
    // The colon and parent class may be on a new line
    // Template params may contain nested <> like `bits<4> op`, so we use a more permissive pattern
    class_def: Regex::new(
        r"class\s+(AIE2_\w+)\s*<((?:[^<>]|<[^>]*>)*)>\s*(?::\s*(AIE2_\w+)\s*<[^>]*>)?\s*\{",
    )
    .unwrap(),
    template_param: Regex::new(r"bits<(\d+)>\s+(\w+)").unwrap(),
    field_decl: Regex::new(r"bits<(\d+)>\s+([\w,\s]+);").unwrap(),
    encoding_stmt: Regex::new(r"let\s+(\w+)\s*=\s*\{([^}]+)\}\s*;").unwrap(),
    instr_def: Regex::new(
        r#"def\s+(\w+)\s*:\s*(AIE2_\w+)\s*<([^>]*)>\s*(?:,\s*(AIE2_\w+))?\s*;"#,
    )
    .unwrap(),
    binary_literal: Regex::new(r"0b([01]+)").unwrap(),
    outs_dag: Regex::new(r"\(outs\s+([^)]*)\)").unwrap(),
    ins_dag: Regex::new(r"\(ins\s+([^)]*)\)").unwrap(),
    operand: Regex::new(r"(\w+):\$(\w+)").unwrap(),
    bit_slice: Regex::new(r"(\w+)\{(\d+)(?:-(\d+))?\}").unwrap(),

    // Attribute patterns
    may_load: Regex::new(r"mayLoad\s*=\s*(true|false)").unwrap(),
    may_store: Regex::new(r"mayStore\s*=\s*(true|false)").unwrap(),
    has_side_effects: Regex::new(r"hasSideEffects\s*=\s*(true|false)").unwrap(),
    defs: Regex::new(r"Defs\s*=\s*\[([^\]]*)\]").unwrap(),
    uses: Regex::new(r"Uses\s*=\s*\[([^\]]*)\]").unwrap(),
    reg_list: Regex::new(r"(\w+)").unwrap(),

    // Pattern patterns - match Pat<(sdnode ...), (INSTR ...)>
    simple_pat: Regex::new(r"Pat<\s*\(\s*(\w+)\s+[^)]*\)\s*,\s*\(\s*(\w+)").unwrap(),
    pat_gpr_gpr: Regex::new(r"PatGprGpr<\s*(\w+)\s*,\s*(\w+)").unwrap(),
    pat_gpr: Regex::new(r"PatGpr<\s*(\w+)\s*,\s*(\w+)").unwrap(),
});

/// Error type for parsing failures.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid encoding part: {0}")]
    InvalidEncoding(String),
    #[error("Missing field width for: {0}")]
    MissingFieldWidth(String),
}

/// Parse a slot definition block.
///
/// Input example:
/// ```tablegen
/// def lda_slot : InstSlot<"Lda", 21> {
///   let FieldToFind = "lda";
/// }
/// ```
fn parse_slot_block(block: &str) -> Option<SlotDef> {
    let caps = PATTERNS.slot_def.captures(block)?;
    let name = caps.get(1)?.as_str().to_string();
    let display_name = caps.get(2)?.as_str().to_string();
    let bits: u8 = caps.get(3)?.as_str().parse().ok()?;

    let field = PATTERNS
        .field_to_find
        .captures(block)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| name.trim_end_matches("_slot").to_string());

    let artificial = PATTERNS.artificial.is_match(block);

    Some(SlotDef {
        name,
        display_name,
        bits,
        field,
        artificial,
    })
}

/// Parse slot definitions from AIE2Slots.td content.
pub fn parse_slots(content: &str) -> Vec<SlotDef> {
    let mut slots = Vec::new();

    // Find all `def X : InstSlot<...>` blocks
    let mut i = 0;
    let bytes = content.as_bytes();

    while i < bytes.len() {
        // Look for "def " followed by slot definition
        if let Some(pos) = content[i..].find("def ") {
            let start = i + pos;

            // Check if this is a slot definition
            if let Some(_slot_match) = PATTERNS.slot_def.captures(&content[start..]) {
                // Find the closing brace for this block
                let block_start = start;
                let mut brace_count = 0;
                let mut block_end = start;
                let mut in_block = false;

                for (j, &byte) in bytes[start..].iter().enumerate() {
                    if byte == b'{' {
                        brace_count += 1;
                        in_block = true;
                    } else if byte == b'}' {
                        brace_count -= 1;
                        if in_block && brace_count == 0 {
                            block_end = start + j + 1;
                            break;
                        }
                    }
                }

                let block = &content[block_start..block_end];
                if let Some(slot) = parse_slot_block(block) {
                    slots.push(slot);
                }

                i = block_end;
                continue;
            }

            i = start + 4; // Skip past "def "
        } else {
            break;
        }
    }

    slots
}

/// Parse an encoding part from a string like "mRx0", "0b11", or "i{10-6}".
fn parse_encoding_part(s: &str) -> Result<EncodingPart, ParseError> {
    let s = s.trim();

    // Check for binary literal: 0b0101
    if let Some(caps) = PATTERNS.binary_literal.captures(s) {
        let bits_str = caps.get(1).unwrap().as_str();
        let value = u64::from_str_radix(bits_str, 2)
            .map_err(|_| ParseError::InvalidEncoding(s.to_string()))?;
        return Ok(EncodingPart::Literal {
            value,
            width: bits_str.len() as u8,
        });
    }

    // Check for bit slice: field{high-low} or field{width}
    if let Some(caps) = PATTERNS.bit_slice.captures(s) {
        let name = caps.get(1).unwrap().as_str();

        // Handle dontcare specially
        if name == "dontcare" {
            let high: u8 = caps
                .get(2)
                .unwrap()
                .as_str()
                .parse()
                .map_err(|_| ParseError::InvalidEncoding(s.to_string()))?;
            let low: u8 = caps
                .get(3)
                .map(|m| m.as_str().parse().unwrap_or(0))
                .unwrap_or(0);
            return Ok(EncodingPart::DontCare {
                width: high - low + 1,
            });
        }

        let high: u8 = caps
            .get(2)
            .unwrap()
            .as_str()
            .parse()
            .map_err(|_| ParseError::InvalidEncoding(s.to_string()))?;
        let low: u8 = caps
            .get(3)
            .map(|m| m.as_str().parse().unwrap_or(0))
            .unwrap_or(0);

        return Ok(EncodingPart::FieldRef {
            name: name.to_string(),
            high: Some(high),
            low: Some(low),
        });
    }

    // Check for dontcare without braces
    if s == "dontcare" {
        return Ok(EncodingPart::DontCare { width: 1 });
    }

    // Simple field reference: mRx
    if s.chars().all(|c| c.is_alphanumeric() || c == '_') && !s.is_empty() {
        return Ok(EncodingPart::FieldRef {
            name: s.to_string(),
            high: None,
            low: None,
        });
    }

    Err(ParseError::InvalidEncoding(s.to_string()))
}

/// Parse template parameters from a class definition.
///
/// Input: "bits<4> op, dag outs, dag ins, string opcodestr = \"\""
/// Output: [TemplateParam { name: "op", bits: 4 }]
fn parse_template_params(params_str: &str) -> Vec<TemplateParam> {
    PATTERNS
        .template_param
        .captures_iter(params_str)
        .filter_map(|caps| {
            let bits: u8 = caps.get(1)?.as_str().parse().ok()?;
            let name = caps.get(2)?.as_str().to_string();
            Some(TemplateParam { name, bits })
        })
        .collect()
}

/// Parse field declarations from a class body.
///
/// Input: "bits<5> mRx;\nbits<7> c7s;\nbits<5> mRx0;"
/// Output: {"mRx": 5, "c7s": 7, "mRx0": 5}
fn parse_field_decls(body: &str) -> HashMap<String, u8> {
    let mut fields = HashMap::new();

    for caps in PATTERNS.field_decl.captures_iter(body) {
        let bits: u8 = match caps.get(1).and_then(|m| m.as_str().parse().ok()) {
            Some(b) => b,
            None => continue,
        };

        let names_str = match caps.get(2) {
            Some(m) => m.as_str(),
            None => continue,
        };

        // Handle comma-separated field names
        for name in names_str.split(',') {
            let name = name.trim();
            if !name.is_empty() {
                fields.insert(name.to_string(), bits);
            }
        }
    }

    fields
}

/// Parse an encoding statement.
///
/// Input: "let alu = {mRx0, mRx, c7s, 0b11, 0b0};"
/// Output: ("alu", [FieldRef("mRx0"), FieldRef("mRx"), FieldRef("c7s"), Literal(0b11, 2), Literal(0, 1)])
///
/// Handles nested braces like: "let mv = {dst,s0,imm{5-2},0b11,imm{1-0},0b11};"
fn parse_encoding(body: &str) -> Option<(String, Vec<EncodingPart>)> {
    // Find "let <slot> = {" pattern
    let let_pos = body.find("let ")?;
    let eq_pos = body[let_pos..].find('=')? + let_pos;
    let open_brace = body[eq_pos..].find('{')? + eq_pos;

    // Extract slot field name
    let slot_field = body[let_pos + 4..eq_pos].trim().to_string();

    // Find matching closing brace (handle nested braces)
    let mut brace_count = 0;
    let mut close_brace = None;
    for (i, c) in body[open_brace..].char_indices() {
        match c {
            '{' => brace_count += 1,
            '}' => {
                brace_count -= 1;
                if brace_count == 0 {
                    close_brace = Some(open_brace + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close_brace = close_brace?;

    // Extract content between braces
    let parts_str = &body[open_brace + 1..close_brace];

    let parts: Result<Vec<_>, _> = parts_str
        .split(',')
        .map(|s| parse_encoding_part(s.trim()))
        .collect();

    Some((slot_field, parts.ok()?))
}

/// Parse a single format class definition.
fn parse_class_block(block: &str) -> Option<FormatClass> {
    let caps = PATTERNS.class_def.captures(block)?;
    let name = caps.get(1)?.as_str().to_string();
    let params_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
    let parent = caps.get(3).map(|m| m.as_str().to_string());

    let template_params = parse_template_params(params_str);

    // Add template params to fields with their bit widths
    let mut fields = parse_field_decls(block);
    for param in &template_params {
        fields.insert(param.name.clone(), param.bits);
    }

    let (slot_field, encoding) = parse_encoding(block).unwrap_or_else(|| ("".to_string(), vec![]));
    let slot_field = if slot_field.is_empty() {
        None
    } else {
        Some(slot_field)
    };

    Some(FormatClass {
        name,
        parent,
        template_params,
        fields,
        slot_field,
        encoding,
    })
}

/// Parse format class definitions from AIE2GenInstrFormats.td content.
pub fn parse_format_classes(content: &str) -> Vec<FormatClass> {
    let mut classes = Vec::new();
    let bytes = content.as_bytes();

    let mut i = 0;
    while i < bytes.len() {
        // Look for "class AIE2_"
        if let Some(pos) = content[i..].find("class AIE2_") {
            let start = i + pos;

            // Find the opening brace
            let brace_pos = match content[start..].find('{') {
                Some(p) => start + p,
                None => {
                    i = start + 11;
                    continue;
                }
            };

            // Find matching closing brace
            let mut brace_count = 0;
            let mut block_end = brace_pos;

            for (j, &byte) in bytes[brace_pos..].iter().enumerate() {
                if byte == b'{' {
                    brace_count += 1;
                } else if byte == b'}' {
                    brace_count -= 1;
                    if brace_count == 0 {
                        block_end = brace_pos + j + 1;
                        break;
                    }
                }
            }

            let block = &content[start..block_end];
            if let Some(class) = parse_class_block(block) {
                classes.push(class);
            }

            i = block_end;
        } else {
            break;
        }
    }

    classes
}

/// Parse instruction attributes from a context string.
///
/// The context includes the `let` blocks surrounding an instruction definition:
/// ```tablegen
/// let hasSideEffects = false, mayLoad = false, mayStore = false in {
///   let Defs = [srCarry] in
///   def ADD : ...
/// }
/// ```
fn parse_attributes(context: &str) -> InstrAttributes {
    let mut attrs = InstrAttributes::default();

    // Parse boolean attributes
    if let Some(caps) = PATTERNS.may_load.captures(context) {
        attrs.may_load = caps.get(1).map(|m| m.as_str() == "true").unwrap_or(false);
    }
    if let Some(caps) = PATTERNS.may_store.captures(context) {
        attrs.may_store = caps.get(1).map(|m| m.as_str() == "true").unwrap_or(false);
    }
    if let Some(caps) = PATTERNS.has_side_effects.captures(context) {
        attrs.has_side_effects = caps.get(1).map(|m| m.as_str() == "true").unwrap_or(false);
    }

    // Parse Defs list
    if let Some(caps) = PATTERNS.defs.captures(context) {
        if let Some(list_content) = caps.get(1) {
            for reg_cap in PATTERNS.reg_list.captures_iter(list_content.as_str()) {
                if let Some(reg) = reg_cap.get(1) {
                    attrs.defs.insert(reg.as_str().to_string());
                }
            }
        }
    }

    // Parse Uses list
    if let Some(caps) = PATTERNS.uses.captures(context) {
        if let Some(list_content) = caps.get(1) {
            for reg_cap in PATTERNS.reg_list.captures_iter(list_content.as_str()) {
                if let Some(reg) = reg_cap.get(1) {
                    attrs.uses.insert(reg.as_str().to_string());
                }
            }
        }
    }

    attrs
}

/// Parse operands from a dag string like "eR:$mRx, eR:$mRy".
fn parse_operands(dag_content: &str, is_output: bool) -> Vec<OperandDef> {
    PATTERNS
        .operand
        .captures_iter(dag_content)
        .map(|caps| OperandDef {
            is_output,
            reg_class: caps.get(1).unwrap().as_str().to_string(),
            name: caps.get(2).unwrap().as_str().to_string(),
        })
        .collect()
}

/// Parse template arguments (the actual values passed to a format class).
///
/// Input: "0b0000, (outs eR:$mRx), (ins eR:$mRx0, eR:$mRy), \"add\", \"$mRx, $mRx0, $mRy\""
fn parse_template_args(args_str: &str) -> (Vec<u64>, String, String, Vec<OperandDef>, Vec<OperandDef>) {
    let mut template_args = Vec::new();
    let mut mnemonic = String::new();
    let mut asm_string = String::new();
    let mut outputs = Vec::new();
    let mut inputs = Vec::new();

    // Extract binary literal arguments
    for caps in PATTERNS.binary_literal.captures_iter(args_str) {
        let bits_str = caps.get(1).unwrap().as_str();
        if let Ok(value) = u64::from_str_radix(bits_str, 2) {
            template_args.push(value);
        }
    }

    // Extract outs dag
    if let Some(caps) = PATTERNS.outs_dag.captures(args_str) {
        let content = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        outputs = parse_operands(content, true);
    }

    // Extract ins dag
    if let Some(caps) = PATTERNS.ins_dag.captures(args_str) {
        let content = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        inputs = parse_operands(content, false);
    }

    // Extract mnemonic and asm string from quoted strings
    let quoted_strings: Vec<&str> = args_str
        .split('"')
        .enumerate()
        .filter(|(i, _)| i % 2 == 1) // Odd indices are inside quotes
        .map(|(_, s)| s)
        .collect();

    if !quoted_strings.is_empty() {
        mnemonic = quoted_strings[0].to_string();
    }
    if quoted_strings.len() >= 2 {
        asm_string = quoted_strings[1].to_string();
    }

    (template_args, mnemonic, asm_string, outputs, inputs)
}

/// Parse instruction definitions from AIE2GenInstrInfo.td content.
pub fn parse_instructions(content: &str) -> Vec<InstrDef> {
    let mut instructions = Vec::new();

    // The instruction defs can span multiple lines, so we need a more robust approach
    // Look for patterns like: def NAME : AIE2_format<...>;

    // First, normalize the content by joining continued lines
    let normalized = content.replace("\\\n", " ");

    // We need to track context for attributes. Look back from each def to find
    // enclosing `let ... in {` blocks.
    for caps in PATTERNS.instr_def.captures_iter(&normalized) {
        let match_start = caps.get(0).unwrap().start();
        let name = caps.get(1).unwrap().as_str().to_string();
        let format = caps.get(2).unwrap().as_str().to_string();
        let args_str = caps.get(3).map(|m| m.as_str()).unwrap_or("");

        let (template_args, mnemonic, asm_string, outputs, inputs) = parse_template_args(args_str);

        // Extract context for attributes by looking back ~500 chars
        // This captures the enclosing let blocks
        let context_start = match_start.saturating_sub(500);
        let context = &normalized[context_start..match_start];
        let attributes = parse_attributes(context);

        instructions.push(InstrDef {
            name,
            format,
            template_args,
            mnemonic,
            asm_string,
            outputs,
            inputs,
            attributes,
        });
    }

    instructions
}

/// Parse semantic patterns from AIE2InstrPatterns.td content.
///
/// Extracts mappings from LLVM SDNodes to AIE2 instructions.
pub fn parse_patterns(content: &str) -> Vec<SemanticPattern> {
    let mut patterns = Vec::new();

    // Parse simple Pat<(sdnode ...), (INSTR ...)> patterns
    for caps in PATTERNS.simple_pat.captures_iter(content) {
        let sdnode = caps.get(1).unwrap().as_str();
        let instr = caps.get(2).unwrap().as_str().to_string();

        if let Some(op) = SemanticOp::from_sdnode(sdnode) {
            patterns.push(SemanticPattern {
                operation: op,
                instruction: instr,
                operand_count: 2, // Default assumption
                intrinsic_name: None,
            });
        }
    }

    // Parse PatGprGpr<sdnode, INSTR, type> patterns
    for caps in PATTERNS.pat_gpr_gpr.captures_iter(content) {
        let sdnode = caps.get(1).unwrap().as_str();
        let instr = caps.get(2).unwrap().as_str().to_string();

        if let Some(op) = SemanticOp::from_sdnode(sdnode) {
            patterns.push(SemanticPattern {
                operation: op,
                instruction: instr,
                operand_count: 2,
                intrinsic_name: None,
            });
        }
    }

    // Parse PatGpr<sdnode, INSTR, type> patterns (unary ops)
    for caps in PATTERNS.pat_gpr.captures_iter(content) {
        let sdnode = caps.get(1).unwrap().as_str();
        let instr = caps.get(2).unwrap().as_str().to_string();

        if let Some(op) = SemanticOp::from_sdnode(sdnode) {
            patterns.push(SemanticPattern {
                operation: op,
                instruction: instr,
                operand_count: 1,
                intrinsic_name: None,
            });
        }
    }

    // Deduplicate patterns (keep first occurrence)
    let mut seen: HashSet<(SemanticOp, String)> = HashSet::new();
    patterns.retain(|p| {
        let key = (p.operation, p.instruction.clone());
        if seen.contains(&key) {
            false
        } else {
            seen.insert(key);
            true
        }
    });

    patterns
}

/// Parse all TableGen files and return combined data.
///
/// # Arguments
///
/// * `slots_content` - Content of AIE2Slots.td
/// * `formats_content` - Content of AIE2GenInstrFormats.td
/// * `instrs_content` - Content of AIE2GenInstrInfo.td
/// * `patterns_content` - Optional content of AIE2InstrPatterns.td
pub fn parse_tablegen_files(
    slots_content: &str,
    formats_content: &str,
    instrs_content: &str,
) -> TableGenData {
    parse_tablegen_files_with_patterns(slots_content, formats_content, instrs_content, None)
}

/// Parse all TableGen files including patterns.
pub fn parse_tablegen_files_with_patterns(
    slots_content: &str,
    formats_content: &str,
    instrs_content: &str,
    patterns_content: Option<&str>,
) -> TableGenData {
    let mut data = TableGenData::new();

    // Parse slots
    for slot in parse_slots(slots_content) {
        data.slots.insert(slot.name.clone(), slot);
    }

    // Parse format classes
    for format in parse_format_classes(formats_content) {
        data.formats.insert(format.name.clone(), format);
    }

    // Parse instructions
    for instr in parse_instructions(instrs_content) {
        data.instructions.insert(instr.name.clone(), instr);
    }

    // Parse patterns if provided
    if let Some(content) = patterns_content {
        data.patterns = parse_patterns(content);
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SLOT: &str = r#"
def lda_slot     : InstSlot<"Lda", 21> {
    let FieldToFind = "lda";
}
"#;

    const TEST_SLOT_ARTIFICIAL: &str = r#"
def nop_slot     : InstSlot<"Nop", 1> {
    let FieldToFind = "nop";
    let Artificial = true;
}
"#;

    const TEST_FORMAT: &str = r#"
class AIE2_add_r_ri_inst_alu <dag outs, dag ins,
      string opcodestr = "", string argstr = "">
    : AIE2_inst_alu_instr32 <outs, ins, opcodestr, argstr> {
  bits<5> mRx;
  bits<7> c7s;
  bits<5> mRx0;

  let alu = {mRx0,mRx,c7s,0b11,0b0};
}
"#;

    // Test multiline format class with bits<N> template parameter (like SBC, ADD)
    const TEST_FORMAT_ALU_RR: &str = r#"
class AIE2_alu_r_rr_inst_alu < bits<4> op, dag outs, dag ins, string opcodestr = "", string argstr = "">:
    AIE2_inst_alu_instr32 <outs, ins, opcodestr, argstr> {
  bits<5> mRx;
  bits<4> eBinArith;
  bits<5> mRx0;
  bits<5> mRy;

  let alu = {mRx0,mRx,mRy,op,0b1};
}
"#;

    const TEST_FORMAT_MV_ADD: &str = r#"
class AIE2_mv_add_inst_mv < dag outs, dag ins, string opcodestr = "", string argstr = "">:
    AIE2_inst_mv_instr32 <outs, ins, opcodestr, argstr> {
  bits<6> imm;
  bits<7> dst;
  bits<5> s0;
  let mv = {dst,s0,imm{5-2},0b11,imm{1-0},0b11};
}
"#;

    const TEST_INSTR: &str = r#"
def ADD : AIE2_alu_r_rr_inst_alu<0b0000, (outs eR:$mRx), (ins eR:$mRx0, eR:$mRy), "add", "$mRx, $mRx0, $mRy">;
"#;

    #[test]
    fn test_parse_slot() {
        let slots = parse_slots(TEST_SLOT);
        assert_eq!(slots.len(), 1);

        let slot = &slots[0];
        assert_eq!(slot.name, "lda_slot");
        assert_eq!(slot.display_name, "Lda");
        assert_eq!(slot.bits, 21);
        assert_eq!(slot.field, "lda");
        assert!(!slot.artificial);
    }

    #[test]
    fn test_parse_slot_artificial() {
        let slots = parse_slots(TEST_SLOT_ARTIFICIAL);
        assert_eq!(slots.len(), 1);

        let slot = &slots[0];
        assert_eq!(slot.name, "nop_slot");
        assert!(slot.artificial);
    }

    #[test]
    fn test_parse_encoding_part_binary() {
        let part = parse_encoding_part("0b11").unwrap();
        assert!(matches!(part, EncodingPart::Literal { value: 3, width: 2 }));

        let part = parse_encoding_part("0b0").unwrap();
        assert!(matches!(part, EncodingPart::Literal { value: 0, width: 1 }));

        let part = parse_encoding_part("0b0000").unwrap();
        assert!(matches!(part, EncodingPart::Literal { value: 0, width: 4 }));
    }

    #[test]
    fn test_parse_encoding_part_field() {
        let part = parse_encoding_part("mRx").unwrap();
        assert!(matches!(
            part,
            EncodingPart::FieldRef { ref name, high: None, low: None } if name == "mRx"
        ));
    }

    #[test]
    fn test_parse_encoding_part_slice() {
        let part = parse_encoding_part("i{10-6}").unwrap();
        assert!(matches!(
            part,
            EncodingPart::FieldRef { ref name, high: Some(10), low: Some(6) } if name == "i"
        ));
    }

    #[test]
    fn test_parse_encoding_part_dontcare() {
        let part = parse_encoding_part("dontcare{5}").unwrap();
        assert!(matches!(part, EncodingPart::DontCare { width: 6 })); // 5-0+1 = 6

        let part = parse_encoding_part("dontcare{5-1}").unwrap();
        assert!(matches!(part, EncodingPart::DontCare { width: 5 })); // 5-1+1 = 5
    }

    #[test]
    fn test_parse_format_class() {
        let classes = parse_format_classes(TEST_FORMAT);
        assert_eq!(classes.len(), 1);

        let class = &classes[0];
        assert_eq!(class.name, "AIE2_add_r_ri_inst_alu");
        assert_eq!(class.parent, Some("AIE2_inst_alu_instr32".to_string()));
        assert_eq!(class.slot_field, Some("alu".to_string()));

        // Check fields
        assert_eq!(class.fields.get("mRx"), Some(&5));
        assert_eq!(class.fields.get("c7s"), Some(&7));
        assert_eq!(class.fields.get("mRx0"), Some(&5));

        // Check encoding: {mRx0, mRx, c7s, 0b11, 0b0}
        assert_eq!(class.encoding.len(), 5);
        assert!(matches!(&class.encoding[0], EncodingPart::FieldRef { name, .. } if name == "mRx0"));
        assert!(matches!(&class.encoding[1], EncodingPart::FieldRef { name, .. } if name == "mRx"));
        assert!(matches!(&class.encoding[2], EncodingPart::FieldRef { name, .. } if name == "c7s"));
        assert!(matches!(&class.encoding[3], EncodingPart::Literal { value: 3, width: 2 }));
        assert!(matches!(&class.encoding[4], EncodingPart::Literal { value: 0, width: 1 }));

        // 5 + 5 + 7 + 2 + 1 = 20 bits
        assert_eq!(class.encoding_width(), Some(20));
    }

    #[test]
    fn test_parse_format_class_mv_add() {
        let classes = parse_format_classes(TEST_FORMAT_MV_ADD);
        eprintln!("Parsed {} classes from MV_ADD format", classes.len());
        for class in &classes {
            eprintln!(
                "  Class: name={}, slot_field={:?}, encoding={:?}",
                class.name, class.slot_field, class.encoding
            );
        }
        assert_eq!(classes.len(), 1);

        let class = &classes[0];
        assert_eq!(class.name, "AIE2_mv_add_inst_mv");
        assert_eq!(class.parent, Some("AIE2_inst_mv_instr32".to_string()));
        assert_eq!(class.slot_field, Some("mv".to_string()));

        // Check fields
        assert_eq!(class.fields.get("imm"), Some(&6));
        assert_eq!(class.fields.get("dst"), Some(&7));
        assert_eq!(class.fields.get("s0"), Some(&5));

        // Check encoding: {dst,s0,imm{5-2},0b11,imm{1-0},0b11}
        assert_eq!(class.encoding.len(), 6);
    }

    #[test]
    fn test_parse_format_class_alu_rr() {
        // This tests multiline format class with bits<N> template parameter
        let classes = parse_format_classes(TEST_FORMAT_ALU_RR);
        eprintln!("Parsed {} classes from ALU_RR format", classes.len());
        for class in &classes {
            eprintln!(
                "  Class: name={}, parent={:?}, slot_field={:?}, template_params={:?}",
                class.name, class.parent, class.slot_field, class.template_params
            );
        }
        assert_eq!(classes.len(), 1, "Should parse the alu_r_rr format class");

        let class = &classes[0];
        assert_eq!(class.name, "AIE2_alu_r_rr_inst_alu");
        assert_eq!(class.parent, Some("AIE2_inst_alu_instr32".to_string()));
        assert_eq!(class.slot_field, Some("alu".to_string()));

        // Check template params
        assert_eq!(class.template_params.len(), 1);
        assert_eq!(class.template_params[0].name, "op");
        assert_eq!(class.template_params[0].bits, 4);

        // Check fields
        assert_eq!(class.fields.get("mRx"), Some(&5));
        assert_eq!(class.fields.get("mRy"), Some(&5));
        assert_eq!(class.fields.get("mRx0"), Some(&5));
        assert_eq!(class.fields.get("op"), Some(&4)); // template param should be in fields too

        // Check encoding: {mRx0,mRx,mRy,op,0b1}
        assert_eq!(class.encoding.len(), 5);
    }

    #[test]
    fn test_parse_instruction() {
        let instrs = parse_instructions(TEST_INSTR);
        assert_eq!(instrs.len(), 1);

        let instr = &instrs[0];
        assert_eq!(instr.name, "ADD");
        assert_eq!(instr.format, "AIE2_alu_r_rr_inst_alu");
        assert_eq!(instr.template_args, vec![0b0000]);
        assert_eq!(instr.mnemonic, "add");
        assert_eq!(instr.asm_string, "$mRx, $mRx0, $mRy");

        assert_eq!(instr.outputs.len(), 1);
        assert_eq!(instr.outputs[0].reg_class, "eR");
        assert_eq!(instr.outputs[0].name, "mRx");

        assert_eq!(instr.inputs.len(), 2);
        assert_eq!(instr.inputs[0].name, "mRx0");
        assert_eq!(instr.inputs[1].name, "mRy");

        // Default attributes (no context)
        assert!(!instr.attributes.may_load);
        assert!(!instr.attributes.may_store);
    }

    const TEST_INSTR_WITH_ATTRS: &str = r#"
let hasSideEffects = false, mayLoad = true, mayStore = false in {
  let Defs = [srCarry] in
  def LDA_TEST : AIE2_lda_test<0b0001, (outs eR:$dst), (ins eP:$ptr), "lda", "$dst, [$ptr]">;
}
"#;

    #[test]
    fn test_parse_instruction_with_attributes() {
        let instrs = parse_instructions(TEST_INSTR_WITH_ATTRS);
        assert_eq!(instrs.len(), 1);

        let instr = &instrs[0];
        assert_eq!(instr.name, "LDA_TEST");
        assert!(instr.attributes.may_load);
        assert!(!instr.attributes.may_store);
        assert!(!instr.attributes.has_side_effects);
        assert!(instr.attributes.defs.contains("srCarry"));
    }

    const TEST_PATTERNS: &str = r#"
def : PatGprGpr<add, ADD, i32>;
def : PatGprGpr<sub, SUB, i32>;
def : PatGprGpr<and, AND, i32>;
def : PatGpr<abs, ABS, i32>;
def : Pat<(mul eR:$s1, eR:$s2), (MUL_mul_r_rr eR:$s1, eR:$s2)>;
"#;

    #[test]
    fn test_parse_patterns() {
        let patterns = parse_patterns(TEST_PATTERNS);

        // Should find add, sub, and, abs, mul
        assert!(patterns.len() >= 4);

        // Check we found the expected operations
        let ops: Vec<_> = patterns.iter().map(|p| p.operation).collect();
        assert!(ops.contains(&SemanticOp::Add));
        assert!(ops.contains(&SemanticOp::Sub));
        assert!(ops.contains(&SemanticOp::And));
        assert!(ops.contains(&SemanticOp::Abs));

        // Check instruction names
        let add_pat = patterns.iter().find(|p| p.operation == SemanticOp::Add).unwrap();
        assert_eq!(add_pat.instruction, "ADD");
    }

    #[test]
    fn test_parse_template_params() {
        let params = parse_template_params("bits<4> op, dag outs, dag ins, string opcodestr");
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "op");
        assert_eq!(params[0].bits, 4);
    }

    #[test]
    fn test_parse_field_decls() {
        let fields = parse_field_decls("bits<5> mRx, mRy;\nbits<7> c7s;");
        assert_eq!(fields.get("mRx"), Some(&5));
        assert_eq!(fields.get("mRy"), Some(&5));
        assert_eq!(fields.get("c7s"), Some(&7));
    }
}
