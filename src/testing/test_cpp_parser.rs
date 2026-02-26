//! Parse mlir-aie test.cpp files to extract buffer metadata.
//!
//! Each mlir-aie NPU test ships a `test.cpp` that is the XRT host program.
//! It contains all the information we previously hand-maintained in TOML
//! manifest files: buffer sizes, group IDs, element types, and input fill
//! patterns.
//!
//! This parser uses simple regex extraction -- no C++ AST required because
//! test.cpp files follow a highly consistent template structure.
//!
//! # What We Extract
//!
//! - `constexpr int SIZE = N;` and `#define SIZE N` -- size declarations
//! - `#define IN_DATATYPE int8_t` -- element type macros
//! - `xrt::bo(device, SIZE * sizeof(type), FLAGS, kernel.group_id(N))` -- buffer objects
//! - `push_back(expr)` in fill loops -- input patterns
//!
//! # What We Do NOT Extract
//!
//! - Validation logic (hardware reference replaces this entirely)
//! - Control packet bitfield construction (marked as `Opaque`)
//! - Multi-kernel buffer chaining (detected as `multi_kernel = true`)

use std::collections::HashMap;
use std::path::Path;

use regex::Regex;

/// Complete buffer specification extracted from a test.cpp file.
#[derive(Debug, Clone)]
pub struct BufferSpec {
    /// All buffer objects found (excluding instruction buffer at group_id 1).
    pub buffers: Vec<BufferDef>,
    /// Whether this test uses multiple kernels (bo0_*, bo1_* pattern).
    pub multi_kernel: bool,
}

/// A single buffer object extracted from an xrt::bo() declaration.
#[derive(Debug, Clone)]
pub struct BufferDef {
    /// Variable name in source (e.g. "bo_inA", "bo_out", "bo0_inA").
    pub name: String,
    /// XRT group_id from kernel.group_id(N).
    pub group_id: u32,
    /// Number of elements (resolved from constexpr/define).
    pub size_elements: usize,
    /// Element type (resolved from sizeof() argument or DATATYPE macro).
    pub element_type: ElementType,
    /// Direction: input or output (inferred from XRT flags and sync calls).
    pub direction: BufferDir,
    /// Input fill pattern (only meaningful for input buffers).
    pub input_pattern: InputPattern,
}

/// Element type for buffer data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl ElementType {
    /// Size of one element in bytes.
    pub fn byte_size(&self) -> usize {
        match self {
            ElementType::I8 | ElementType::U8 => 1,
            ElementType::I16 | ElementType::U16 => 2,
            ElementType::I32 | ElementType::U32 => 4,
            ElementType::I64 | ElementType::U64 => 8,
        }
    }

    /// Parse from a C++ type string (e.g. "int32_t", "uint8_t", "int").
    pub fn from_cpp_type(s: &str) -> Option<Self> {
        match s.trim() {
            "int8_t" => Some(ElementType::I8),
            "int16_t" => Some(ElementType::I16),
            "int32_t" | "int" => Some(ElementType::I32),
            "int64_t" => Some(ElementType::I64),
            "uint8_t" => Some(ElementType::U8),
            "uint16_t" => Some(ElementType::U16),
            "uint32_t" => Some(ElementType::U32),
            "uint64_t" => Some(ElementType::U64),
            _ => None,
        }
    }

    /// Short string for display/serialization (e.g. "i32", "u8").
    pub fn as_str(&self) -> &'static str {
        match self {
            ElementType::I8 => "i8",
            ElementType::I16 => "i16",
            ElementType::I32 => "i32",
            ElementType::I64 => "i64",
            ElementType::U8 => "u8",
            ElementType::U16 => "u16",
            ElementType::U32 => "u32",
            ElementType::U64 => "u64",
        }
    }

    /// Parse from a short string (e.g. "i32", "u8").
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim() {
            "i8" => Some(ElementType::I8),
            "i16" => Some(ElementType::I16),
            "i32" => Some(ElementType::I32),
            "i64" => Some(ElementType::I64),
            "u8" => Some(ElementType::U8),
            "u16" => Some(ElementType::U16),
            "u32" => Some(ElementType::U32),
            "u64" => Some(ElementType::U64),
            _ => None,
        }
    }
}

/// Buffer direction: input (host->device) or output (device->host).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferDir {
    Input,
    Output,
}

/// Pattern used to fill an input buffer before kernel execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputPattern {
    /// Sequential values: value[i] = start + i * step.
    Sequential { start: i64, step: i64 },
    /// All elements filled with the same constant.
    Constant(i64),
    /// Zero-filled or memset to zero.
    Zeros,
    /// Too complex to parse with regex (e.g. control packet headers).
    /// The hardware reference capture will provide the correct input.
    Opaque,
}

/// Parse a test.cpp file and extract buffer metadata.
///
/// Returns `None` if the file does not exist or cannot be parsed
/// (e.g. Python-only tests with no test.cpp).
pub fn parse_test_cpp(test_dir: &Path) -> Option<BufferSpec> {
    let test_cpp = test_dir.join("test.cpp");
    let content = std::fs::read_to_string(&test_cpp).ok()?;
    parse_test_cpp_content(&content)
}

/// Parse test.cpp content (for testability without filesystem).
pub fn parse_test_cpp_content(content: &str) -> Option<BufferSpec> {
    // Step 1: Collect type aliases first (needed for sizeof resolution in constants).
    let type_macros = extract_type_macros(content);

    // Step 2: Collect all constants (#define, constexpr, cxxopts defaults).
    let constants = extract_constants(content, &type_macros);

    // Step 3: Detect multi-kernel pattern (bo0_*, bo1_* prefixes).
    let multi_kernel = content.contains("bo0_in") || content.contains("kernel0");

    // Step 4: Extract xrt::bo() declarations (skip instruction buffer at gid 1).
    let buffers = extract_buffer_objects(content, &constants, &type_macros);

    // Step 5: Determine directions from sync calls and naming.
    let buffers = classify_directions(buffers, content);

    // Step 6: Extract input patterns from fill loops.
    let buffers = extract_input_patterns(buffers, content, &constants);

    if buffers.is_empty() {
        return None;
    }

    Some(BufferSpec {
        buffers,
        multi_kernel,
    })
}

/// Extract integer constants from `constexpr int NAME = EXPR;`,
/// `#define NAME EXPR`, and cxxopts default values.
///
/// The cxxopts pattern handles tests like `objectfifo_repeat/init_values_repeat`
/// where buffer sizes are computed from runtime arguments with known defaults:
///
/// ```cpp
/// options.add_options()("length,l", "...", cxxopts::value<int>()->default_value("4096"));
/// int N = vm["length"].as<int>();
/// ```
///
/// We extract the default value and map it to the variable name.
fn extract_constants(
    content: &str,
    type_macros: &HashMap<String, ElementType>,
) -> HashMap<String, usize> {
    let mut constants = HashMap::new();

    // constexpr int NAME = EXPR;
    let re_constexpr = Regex::new(
        r"constexpr\s+int\s+(\w+)\s*=\s*(.+?)\s*;"
    ).unwrap();
    for cap in re_constexpr.captures_iter(content) {
        let name = cap[1].to_string();
        if let Some(val) = eval_const_expr(&cap[2], &constants, type_macros) {
            constants.insert(name, val);
        }
    }

    // #define NAME EXPR (integer/expression values, skip type/string macros)
    let re_define = Regex::new(
        r#"#define\s+(\w+)\s+(.+)"#
    ).unwrap();
    for cap in re_define.captures_iter(content) {
        let name = cap[1].to_string();
        let value = cap[2].trim();
        // Skip type macros and string macros
        if name.contains("DATATYPE") || name.contains("KERNEL")
            || name.contains("XCLBIN") || name.contains("INSTS")
            || value.starts_with("std::") || value.starts_with('"')
        {
            continue;
        }
        // Strip trailing C/C++ comments
        let value = value.split("//").next().unwrap_or(value).trim();
        if let Some(val) = eval_const_expr(value, &constants, type_macros) {
            constants.insert(name, val);
        }
    }

    // cxxopts default values: extract option defaults and map to variables.
    //
    // Step 1: Collect option_name -> default_value from cxxopts declarations.
    //   Pattern: ("option_name,short", "desc", cxxopts::value<int>()->default_value("VALUE"))
    let cxxopts_defaults = extract_cxxopts_defaults(content);

    // Step 2: Map variable assignments from vm["option_name"].as<type>().
    //   Pattern: int VAR = vm["option_name"].as<int>();
    let re_vm_assign = Regex::new(
        r#"int\s+(\w+)\s*=\s*vm\["(\w+)"\]\.as<int>\(\)"#
    ).unwrap();
    for cap in re_vm_assign.captures_iter(content) {
        let var_name = cap[1].to_string();
        let option_name = &cap[2];
        if let Some(default_val) = cxxopts_defaults.get(option_name) {
            constants.entry(var_name).or_insert(*default_val);
        }
    }

    constants
}

/// Extract cxxopts default values from option declarations.
///
/// Matches patterns like:
///   ("length,l", "description", cxxopts::value<int>()->default_value("4096"))
///   ("repeat,r", "description", cxxopts::value<int>()->default_value("4"))
///
/// Returns a map from option name (e.g. "length") to default value.
fn extract_cxxopts_defaults(content: &str) -> HashMap<String, usize> {
    let mut defaults = HashMap::new();

    // Allow whitespace between ( and " for multiline chained options:
    //   options.add_options()("length,l", ..., default_value("4096"))(
    //       "repeat,r", ..., default_value("4"));
    let re = Regex::new(
        r#"\(\s*"(\w+)(?:,\w)?"[^)]*cxxopts::value<int>\(\)->default_value\("(\d+)"\)"#
    ).unwrap();
    for cap in re.captures_iter(content) {
        let option_name = cap[1].to_string();
        if let Ok(val) = cap[2].parse::<usize>() {
            defaults.insert(option_name, val);
        }
    }

    defaults
}

/// Evaluate a simple constant expression (supports +, *, -, /, parentheses,
/// sizeof(), and named constant references).
fn eval_const_expr(
    expr: &str,
    constants: &HashMap<String, usize>,
    type_macros: &HashMap<String, ElementType>,
) -> Option<usize> {
    let expr = expr.trim();

    // Direct integer literal
    if let Ok(v) = expr.parse::<usize>() {
        return Some(v);
    }

    // Named constant reference
    if let Some(v) = constants.get(expr) {
        return Some(*v);
    }

    // sizeof(type) -- resolve standard C++ types and type aliases
    if let Some(inner) = expr.strip_prefix("sizeof(").and_then(|s| s.strip_suffix(')')) {
        let inner = inner.trim();
        // Strip pointer suffix if present (e.g. "buf_in[0]")
        let type_name = if inner.contains('[') {
            inner.split('[').next().unwrap_or(inner).trim()
        } else {
            inner
        };
        // Try direct C++ type first, then type aliases (using/define)
        if let Some(t) = ElementType::from_cpp_type(type_name) {
            return Some(t.byte_size());
        }
        if let Some(t) = type_macros.get(type_name) {
            return Some(t.byte_size());
        }
        return None;
    }

    // Binary expressions: A * B, A + B, etc.
    // Handle parenthesized sub-expressions by stripping outer parens
    let expr = if expr.starts_with('(') && expr.ends_with(')') {
        &expr[1..expr.len()-1]
    } else {
        expr
    };

    // Try splitting on * (multiplication, most common in size expressions)
    if let Some(pos) = find_top_level_op(expr, '*') {
        let left = eval_const_expr(&expr[..pos], constants, type_macros)?;
        let right = eval_const_expr(&expr[pos+1..], constants, type_macros)?;
        return Some(left * right);
    }

    // Try splitting on + (addition)
    if let Some(pos) = find_top_level_op(expr, '+') {
        let left = eval_const_expr(&expr[..pos], constants, type_macros)?;
        let right = eval_const_expr(&expr[pos+1..], constants, type_macros)?;
        return Some(left + right);
    }

    None
}

/// Lenient expression evaluation: like `eval_const_expr` but treats unknown
/// addends as 0 in addition expressions.
///
/// This handles `C_SIZE + trace_size` where `trace_size` is an unresolvable
/// runtime variable that defaults to 0. For multiplication, both sides must
/// be known (unknown factor = undefined result, not safe to assume 0).
fn eval_const_expr_lenient(
    expr: &str,
    constants: &HashMap<String, usize>,
    type_macros: &HashMap<String, ElementType>,
) -> Option<usize> {
    // Try exact evaluation first
    if let Some(val) = eval_const_expr(expr, constants, type_macros) {
        return Some(val);
    }

    let expr = expr.trim();

    // Strip outer parens
    let expr = if expr.starts_with('(') && expr.ends_with(')') {
        &expr[1..expr.len()-1]
    } else {
        expr
    };

    // For addition: if one side resolves and the other doesn't, use the known side
    if let Some(pos) = find_top_level_op(expr, '+') {
        let left = eval_const_expr(&expr[..pos], constants, type_macros);
        let right = eval_const_expr(&expr[pos+1..], constants, type_macros);
        match (left, right) {
            (Some(l), Some(r)) => return Some(l + r),
            (Some(l), None) => return Some(l),
            (None, Some(r)) => return Some(r),
            (None, None) => {}
        }
    }

    None
}

/// Find a top-level binary operator (not inside parentheses).
fn find_top_level_op(expr: &str, op: char) -> Option<usize> {
    let mut depth = 0;
    // Scan right-to-left for left-associativity
    for (i, c) in expr.char_indices().rev() {
        match c {
            ')' => depth += 1,
            '(' => depth -= 1,
            c if c == op && depth == 0 && i > 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Extract type aliases from `#define` macros and `using` declarations.
///
/// Handles two patterns:
///   `#define IN_DATATYPE int8_t`          -- preprocessor macro
///   `using A_DATATYPE = std::int32_t;`    -- C++ type alias (matmul tests)
fn extract_type_macros(content: &str) -> HashMap<String, ElementType> {
    let mut macros = HashMap::new();

    // #define DATATYPE type
    let re_define = Regex::new(
        r"#define\s+(\w*DATATYPE\w*)\s+(\w+)"
    ).unwrap();
    for cap in re_define.captures_iter(content) {
        let name = cap[1].to_string();
        if let Some(elem_type) = ElementType::from_cpp_type(&cap[2]) {
            macros.insert(name, elem_type);
        }
    }

    // using NAME = std::type; (with or without std:: prefix)
    let re_using = Regex::new(
        r"using\s+(\w+)\s*=\s*(?:std::)?(\w+)\s*;"
    ).unwrap();
    for cap in re_using.captures_iter(content) {
        let name = cap[1].to_string();
        if let Some(elem_type) = ElementType::from_cpp_type(&cap[2]) {
            macros.insert(name, elem_type);
        }
    }

    macros
}

/// Extract xrt::bo() declarations, resolving size and type from context.
fn extract_buffer_objects(
    content: &str,
    constants: &HashMap<String, usize>,
    type_macros: &HashMap<String, ElementType>,
) -> Vec<BufferDef> {
    let mut buffers = Vec::new();

    // Match: auto VAR = xrt::bo(device, SIZE_EXPR, FLAGS, kernel.group_id(N));
    // Also handles: auto VAR = xrt::bo(device, SIZE_EXPR, FLAGS, kernelN.group_id(N));
    // The size expression can be multi-token: `IN_SIZE * sizeof(int32_t)`
    let re = Regex::new(
        r"auto\s+(\w+)\s*=\s*xrt::bo\(\s*device\s*,\s*(.+?)\s*,\s*(XRT_BO_FLAGS_HOST_ONLY|XCL_BO_FLAGS_CACHEABLE)\s*,\s*\w+\.group_id\((\d+)\)\s*\)"
    ).unwrap();

    for cap in re.captures_iter(content) {
        let var_name = cap[1].to_string();
        let size_expr = cap[2].trim();
        let flags = &cap[3];
        let group_id: u32 = cap[4].parse().unwrap_or(0);

        // Skip instruction buffer (always group_id 1, XCL_BO_FLAGS_CACHEABLE)
        if group_id <= 2 || flags == "XCL_BO_FLAGS_CACHEABLE" {
            continue;
        }

        // Parse size expression: typically "SIZE * sizeof(type)"
        let (size_elements, elem_type) = parse_size_expr(size_expr, constants, type_macros);

        buffers.push(BufferDef {
            name: var_name,
            group_id,
            size_elements,
            element_type: elem_type,
            direction: BufferDir::Input, // Will be classified later
            input_pattern: InputPattern::Zeros, // Will be extracted later
        });
    }

    buffers
}

/// Parse a buffer size expression like `IN_SIZE * sizeof(int32_t)` or `SIZE`.
///
/// Returns (element_count, element_type). When the expression is just a byte
/// count (like `SIZE` where SIZE is already in bytes), we divide by the
/// detected element size.
fn parse_size_expr(
    expr: &str,
    constants: &HashMap<String, usize>,
    type_macros: &HashMap<String, ElementType>,
) -> (usize, ElementType) {
    let default_type = ElementType::I32;

    // Pattern 1: EXPR * sizeof(TYPE)
    if let Some(pos) = expr.find("sizeof") {
        let count_part = expr[..pos].trim().trim_end_matches('*').trim();
        let sizeof_part = &expr[pos..];

        // Resolve count
        let count = eval_const_expr(count_part, constants, type_macros).unwrap_or(64);

        // Resolve type from sizeof argument
        let elem_type = parse_sizeof_type(sizeof_part, type_macros)
            .unwrap_or(default_type);

        return (count, elem_type);
    }

    // Pattern 2: general expression evaluation (handles constants, arithmetic)
    //
    // For expressions like `C_SIZE + trace_size` where one term is unknown,
    // eval_const_expr_lenient treats unknown addends as 0. This is correct
    // for the common `DATA_SIZE + trace_size` pattern where trace_size
    // defaults to 0 in production and the data size is the dominant term.
    if let Some(val) = eval_const_expr_lenient(expr.trim(), constants, type_macros) {
        if val > 0 {
            let elem_type = default_type;
            let size_elements = val / elem_type.byte_size();
            return (size_elements, elem_type);
        }
    }

    // Pattern 3: numeric literal
    if let Ok(val) = expr.trim().parse::<usize>() {
        return (val / default_type.byte_size(), default_type);
    }

    // Fallback
    (64, default_type)
}

/// Extract the element type from a sizeof() expression.
fn parse_sizeof_type(
    sizeof_expr: &str,
    type_macros: &HashMap<String, ElementType>,
) -> Option<ElementType> {
    // Extract the argument: sizeof(TYPE) or sizeof(TYPE *)
    let re = Regex::new(r"sizeof\(\s*(\w+)").unwrap();
    let cap = re.captures(sizeof_expr)?;
    let type_name = &cap[1];

    // Check type macros first (e.g. IN_DATATYPE -> int8_t)
    if let Some(t) = type_macros.get(type_name) {
        return Some(*t);
    }

    // Direct C++ type
    ElementType::from_cpp_type(type_name)
}

/// Classify buffer directions from sync calls and naming conventions.
fn classify_directions(mut buffers: Vec<BufferDef>, content: &str) -> Vec<BufferDef> {
    // Collect variable names that are synced FROM device (output buffers)
    let re_from = Regex::new(r"(\w+)\.sync\(XCL_BO_SYNC_BO_FROM_DEVICE\)").unwrap();
    let mut from_device: Vec<String> = Vec::new();
    for cap in re_from.captures_iter(content) {
        from_device.push(cap[1].to_string());
    }

    for buf in &mut buffers {
        if from_device.contains(&buf.name) {
            buf.direction = BufferDir::Output;
        } else if buf.name.contains("out") || buf.name.contains("Out") {
            // Naming convention fallback: buffers named "out" are outputs
            buf.direction = BufferDir::Output;
        } else {
            buf.direction = BufferDir::Input;
        }
    }

    buffers
}

/// Extract input fill patterns from push_back loops and memset calls.
fn extract_input_patterns(
    mut buffers: Vec<BufferDef>,
    content: &str,
    constants: &HashMap<String, usize>,
) -> Vec<BufferDef> {
    // For output buffers, pattern is always Zeros (they receive data).
    // For input buffers, look for the fill loop.

    // Common patterns in mlir-aie test.cpp files:
    //   push_back(i + C)  -> Sequential { start: C, step: 1 }
    //   push_back(C)      -> Constant(C)
    //   memset(buf, 0, N) -> Zeros
    //   push_back(1)      -> Constant(1)
    //
    // The fill loop is typically:
    //   for (int i = 0; i < SIZE; i++)
    //     srcVecX.push_back(EXPR);
    //   memcpy(bufX, srcVecX.data(), ...);
    //
    // We match push_back expressions to the buffer via variable naming.

    // Collect all push_back expressions with their vector names
    let re_pushback = Regex::new(
        r"(\w+)\.push_back\((.+?)\)"
    ).unwrap();

    let mut vec_patterns: HashMap<String, InputPattern> = HashMap::new();
    for cap in re_pushback.captures_iter(content) {
        let vec_name = cap[1].to_string();
        let expr = cap[2].trim();

        // Skip if already classified (take the first push_back in a loop)
        if vec_patterns.contains_key(&vec_name) {
            continue;
        }

        let pattern = parse_fill_expr(expr, constants);
        vec_patterns.insert(vec_name, pattern);
    }

    // Match vectors to buffer objects via memcpy:
    //   memcpy(bufInA, srcVecA.data(), ...)
    let re_memcpy = Regex::new(
        r"memcpy\(\s*(\w+)\s*,\s*(\w+)\.data\(\)"
    ).unwrap();

    let mut buf_to_vec: HashMap<String, String> = HashMap::new();
    for cap in re_memcpy.captures_iter(content) {
        let buf_ptr = cap[1].to_string();
        let vec_name = cap[2].to_string();
        buf_to_vec.insert(buf_ptr, vec_name);
    }

    // Also match direct map + assignment patterns (matrix_transpose style):
    //   TYPE *buf_ptr = bo_var.map<TYPE *>();
    let re_map = Regex::new(
        r"(\w+)\s*\*\s*(\w+)\s*=\s*(\w+)\.map<"
    ).unwrap();

    let mut bo_to_ptr: HashMap<String, String> = HashMap::new();
    for cap in re_map.captures_iter(content) {
        let ptr_name = cap[2].to_string();
        let bo_name = cap[3].to_string();
        bo_to_ptr.insert(bo_name, ptr_name);
    }

    // Check for memset(ptr, 0, ...) patterns
    let re_memset = Regex::new(
        r"memset\(\s*(\w+)\s*,\s*0\s*,"
    ).unwrap();
    let mut zeroed_ptrs: Vec<String> = Vec::new();
    for cap in re_memset.captures_iter(content) {
        zeroed_ptrs.push(cap[1].to_string());
    }

    // Also check for direct indexed assignment patterns:
    //   buf_ptr[i] = EXPR;
    let re_indexed = Regex::new(
        r"(\w+)\[(\w+)\]\s*=\s*(.+?)\s*;"
    ).unwrap();
    let mut ptr_patterns: HashMap<String, InputPattern> = HashMap::new();
    for cap in re_indexed.captures_iter(content) {
        let ptr_name = cap[1].to_string();
        let idx_var = &cap[2];
        let expr = cap[3].trim();

        if ptr_patterns.contains_key(&ptr_name) {
            continue;
        }

        // If index variable is 'i' and expr references i, it's sequential
        if idx_var == "i" {
            let pattern = parse_fill_expr(expr, constants);
            ptr_patterns.insert(ptr_name, pattern);
        }
    }

    // Now assign patterns to buffers
    for buf in &mut buffers {
        if buf.direction == BufferDir::Output {
            buf.input_pattern = InputPattern::Zeros;
            continue;
        }

        // Try matching via bo -> ptr -> vec -> pattern chain
        if let Some(ptr_name) = bo_to_ptr.get(&buf.name) {
            // Check memset zeros first
            if zeroed_ptrs.contains(ptr_name) {
                buf.input_pattern = InputPattern::Zeros;
                continue;
            }

            // Check direct indexed assignment
            if let Some(pattern) = ptr_patterns.get(ptr_name) {
                buf.input_pattern = pattern.clone();
                continue;
            }

            // Check memcpy from vector
            if let Some(vec_name) = buf_to_vec.get(ptr_name.as_str()) {
                if let Some(pattern) = vec_patterns.get(vec_name) {
                    buf.input_pattern = pattern.clone();
                    continue;
                }
            }
        }

        // Try matching via naming convention (bufInA -> srcVecA pattern)
        // The naming convention is: bo_inA -> mapped as bufInA -> filled from srcVecA
        let suffix = buf.name.trim_start_matches("bo_").trim_start_matches("bo0_").trim_start_matches("bo1_");
        for (ptr_name, vec_name) in &buf_to_vec {
            // Check if the pointer name contains the buffer suffix
            if ptr_name.to_lowercase().contains(&suffix.to_lowercase()) {
                if let Some(pattern) = vec_patterns.get(vec_name) {
                    buf.input_pattern = pattern.clone();
                    break;
                }
            }
        }
    }

    buffers
}

/// Parse a fill expression from push_back(EXPR) into an InputPattern.
fn parse_fill_expr(expr: &str, constants: &HashMap<String, usize>) -> InputPattern {
    let expr = expr.trim();

    // Constant literal: push_back(0) or push_back(1)
    if let Ok(val) = expr.parse::<i64>() {
        if val == 0 {
            return InputPattern::Zeros;
        }
        return InputPattern::Constant(val);
    }

    // Named constant: push_back(SOME_CONST)
    if let Some(val) = constants.get(expr) {
        if *val == 0 {
            return InputPattern::Zeros;
        }
        return InputPattern::Constant(*val as i64);
    }

    // i + C pattern: push_back(i + 1)
    let re_i_plus = Regex::new(r"^i\s*\+\s*(\d+)$").unwrap();
    if let Some(cap) = re_i_plus.captures(expr) {
        let start: i64 = cap[1].parse().unwrap_or(0);
        return InputPattern::Sequential { start, step: 1 };
    }

    // C + i pattern: push_back(1 + i)
    let re_plus_i = Regex::new(r"^(\d+)\s*\+\s*i$").unwrap();
    if let Some(cap) = re_plus_i.captures(expr) {
        let start: i64 = cap[1].parse().unwrap_or(0);
        return InputPattern::Sequential { start, step: 1 };
    }

    // Just `i`: push_back(i) -> Sequential { start: 0, step: 1 }
    if expr == "i" {
        return InputPattern::Sequential { start: 0, step: 1 };
    }

    // Anything more complex is opaque
    InputPattern::Opaque
}

/// Generate input data bytes from a BufferDef's pattern.
///
/// Returns a byte vector suitable for writing into host memory.
pub fn generate_input_data(buf: &BufferDef) -> Vec<u8> {
    let elem_size = buf.element_type.byte_size();
    let byte_count = buf.size_elements * elem_size;
    let mut data = vec![0u8; byte_count];

    match &buf.input_pattern {
        InputPattern::Zeros => {} // Already zero
        InputPattern::Constant(val) => {
            write_pattern(&mut data, buf.element_type, buf.size_elements, |_| *val);
        }
        InputPattern::Sequential { start, step } => {
            write_pattern(&mut data, buf.element_type, buf.size_elements, |i| {
                start + (i as i64) * step
            });
        }
        InputPattern::Opaque => {} // Leave as zeros; hardware reference will catch mismatches
    }

    data
}

/// Write a pattern into a byte buffer using the given element type.
fn write_pattern(
    data: &mut [u8],
    elem_type: ElementType,
    count: usize,
    value_fn: impl Fn(usize) -> i64,
) {
    let elem_size = elem_type.byte_size();
    for i in 0..count {
        let val = value_fn(i);
        let offset = i * elem_size;
        if offset + elem_size > data.len() {
            break;
        }
        match elem_type {
            ElementType::I8 | ElementType::U8 => {
                data[offset] = val as u8;
            }
            ElementType::I16 | ElementType::U16 => {
                let bytes = (val as u16).to_le_bytes();
                data[offset..offset+2].copy_from_slice(&bytes);
            }
            ElementType::I32 | ElementType::U32 => {
                let bytes = (val as u32).to_le_bytes();
                data[offset..offset+4].copy_from_slice(&bytes);
            }
            ElementType::I64 | ElementType::U64 => {
                let bytes = (val as u64).to_le_bytes();
                data[offset..offset+8].copy_from_slice(&bytes);
            }
        }
    }
}

/// Read element values from a byte buffer as i64 (sign-extended).
pub fn read_values(data: &[u8], elem_type: ElementType) -> Vec<i64> {
    let elem_size = elem_type.byte_size();
    let count = data.len() / elem_size;
    let mut values = Vec::with_capacity(count);

    for i in 0..count {
        let offset = i * elem_size;
        if offset + elem_size > data.len() {
            break;
        }
        let val = match elem_type {
            ElementType::I8 => data[offset] as i8 as i64,
            ElementType::U8 => data[offset] as u8 as i64,
            ElementType::I16 => {
                let bytes: [u8; 2] = data[offset..offset+2].try_into().unwrap();
                i16::from_le_bytes(bytes) as i64
            }
            ElementType::U16 => {
                let bytes: [u8; 2] = data[offset..offset+2].try_into().unwrap();
                u16::from_le_bytes(bytes) as i64
            }
            ElementType::I32 => {
                let bytes: [u8; 4] = data[offset..offset+4].try_into().unwrap();
                i32::from_le_bytes(bytes) as i64
            }
            ElementType::U32 => {
                let bytes: [u8; 4] = data[offset..offset+4].try_into().unwrap();
                u32::from_le_bytes(bytes) as i64
            }
            ElementType::I64 => {
                let bytes: [u8; 8] = data[offset..offset+8].try_into().unwrap();
                i64::from_le_bytes(bytes)
            }
            ElementType::U64 => {
                let bytes: [u8; 8] = data[offset..offset+8].try_into().unwrap();
                u64::from_le_bytes(bytes) as i64
            }
        };
        values.push(val);
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Constant extraction
    // ---------------------------------------------------------------

    #[test]
    fn test_extract_constexpr_int() {
        let content = r#"
constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;
"#;
        let types = HashMap::new();
        let constants = extract_constants(content, &types);
        assert_eq!(constants["IN_SIZE"], 64);
        assert_eq!(constants["OUT_SIZE"], 64);
    }

    #[test]
    fn test_extract_define_constants() {
        let content = r#"
#define MATRIX_ROWS 7
#define MATRIX_COLS 19
"#;
        let types = HashMap::new();
        let constants = extract_constants(content, &types);
        assert_eq!(constants["MATRIX_ROWS"], 7);
        assert_eq!(constants["MATRIX_COLS"], 19);
    }

    #[test]
    fn test_extract_type_macros() {
        let content = r#"
#define IN_DATATYPE int8_t
#define OUT_DATATYPE int8_t
"#;
        let macros = extract_type_macros(content);
        assert_eq!(macros["IN_DATATYPE"], ElementType::I8);
        assert_eq!(macros["OUT_DATATYPE"], ElementType::I8);
    }

    #[test]
    fn test_eval_const_expr_multiplication() {
        let mut constants = HashMap::new();
        constants.insert("IN_SIZE".to_string(), 64);
        let types = HashMap::new();
        assert_eq!(eval_const_expr("IN_SIZE * 4", &constants, &types), Some(256));
    }

    #[test]
    fn test_eval_const_expr_sizeof() {
        let constants = HashMap::new();
        let types = HashMap::new();
        assert_eq!(eval_const_expr("sizeof(int32_t)", &constants, &types), Some(4));
        assert_eq!(eval_const_expr("sizeof(int8_t)", &constants, &types), Some(1));
    }

    #[test]
    fn test_eval_const_expr_sizeof_type_alias() {
        let constants = HashMap::new();
        let mut types = HashMap::new();
        types.insert("A_DATATYPE".to_string(), ElementType::I32);
        assert_eq!(eval_const_expr("sizeof(A_DATATYPE)", &constants, &types), Some(4));
    }

    // ---------------------------------------------------------------
    // Element type parsing
    // ---------------------------------------------------------------

    #[test]
    fn test_element_type_from_cpp() {
        assert_eq!(ElementType::from_cpp_type("int32_t"), Some(ElementType::I32));
        assert_eq!(ElementType::from_cpp_type("uint8_t"), Some(ElementType::U8));
        assert_eq!(ElementType::from_cpp_type("int"), Some(ElementType::I32));
    }

    #[test]
    fn test_element_type_byte_size() {
        assert_eq!(ElementType::I8.byte_size(), 1);
        assert_eq!(ElementType::I16.byte_size(), 2);
        assert_eq!(ElementType::I32.byte_size(), 4);
        assert_eq!(ElementType::I64.byte_size(), 8);
    }

    // ---------------------------------------------------------------
    // Fill pattern parsing
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_fill_expr_constant() {
        let constants = HashMap::new();
        assert_eq!(parse_fill_expr("1", &constants), InputPattern::Constant(1));
        assert_eq!(parse_fill_expr("0", &constants), InputPattern::Zeros);
    }

    #[test]
    fn test_parse_fill_expr_sequential() {
        let constants = HashMap::new();
        assert_eq!(
            parse_fill_expr("i + 1", &constants),
            InputPattern::Sequential { start: 1, step: 1 }
        );
        assert_eq!(
            parse_fill_expr("i", &constants),
            InputPattern::Sequential { start: 0, step: 1 }
        );
    }

    // ---------------------------------------------------------------
    // Full test.cpp parsing: add_one_using_dma
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_add_one_using_dma() {
        let content = r#"
constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;

auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                        XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
auto bo_inB = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

uint32_t *bufInA = bo_inA.map<uint32_t *>();
std::vector<uint32_t> srcVecA;
for (int i = 0; i < IN_SIZE; i++)
  srcVecA.push_back(i + 1);
memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
"#;

        let spec = parse_test_cpp_content(content).unwrap();
        assert!(!spec.multi_kernel);
        assert_eq!(spec.buffers.len(), 3);

        // bo_inA: group_id=3, 64 elements, i32, input, sequential(1,1)
        let in_a = spec.buffers.iter().find(|b| b.name == "bo_inA").unwrap();
        assert_eq!(in_a.group_id, 3);
        assert_eq!(in_a.size_elements, 64);
        assert_eq!(in_a.element_type, ElementType::I32);
        assert_eq!(in_a.direction, BufferDir::Input);
        assert_eq!(in_a.input_pattern, InputPattern::Sequential { start: 1, step: 1 });

        // bo_inB: group_id=4, 64 elements, i32, input
        let in_b = spec.buffers.iter().find(|b| b.name == "bo_inB").unwrap();
        assert_eq!(in_b.group_id, 4);
        assert_eq!(in_b.size_elements, 64);
        assert_eq!(in_b.direction, BufferDir::Input);

        // bo_out: group_id=5, 64 elements, i32, output
        let out = spec.buffers.iter().find(|b| b.name == "bo_out").unwrap();
        assert_eq!(out.group_id, 5);
        assert_eq!(out.size_elements, 64);
        assert_eq!(out.element_type, ElementType::I32);
        assert_eq!(out.direction, BufferDir::Output);
    }

    // ---------------------------------------------------------------
    // Full test.cpp parsing: packet_flow (int8_t with DATATYPE macros)
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_packet_flow_i8() {
        let content = r#"
#define IN_DATATYPE int8_t
#define OUT_DATATYPE int8_t

constexpr int IN_SIZE = 64 * 64;
constexpr int OUT_SIZE = 64 * 64;

auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(IN_DATATYPE),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
auto bo_inB = xrt::bo(device, IN_SIZE * sizeof(IN_DATATYPE),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(OUT_DATATYPE),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

IN_DATATYPE *bufInA = bo_inA.map<IN_DATATYPE *>();
std::vector<IN_DATATYPE> srcVecA;
for (int i = 0; i < IN_SIZE; i++)
  srcVecA.push_back(1);
memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(IN_DATATYPE)));

bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
"#;

        let spec = parse_test_cpp_content(content).unwrap();
        assert!(!spec.multi_kernel);

        let in_a = spec.buffers.iter().find(|b| b.name == "bo_inA").unwrap();
        assert_eq!(in_a.element_type, ElementType::I8);
        assert_eq!(in_a.size_elements, 4096);
        assert_eq!(in_a.input_pattern, InputPattern::Constant(1));

        let out = spec.buffers.iter().find(|b| b.name == "bo_out").unwrap();
        assert_eq!(out.element_type, ElementType::I8);
        assert_eq!(out.direction, BufferDir::Output);
    }

    // ---------------------------------------------------------------
    // Full test.cpp parsing: matrix_transpose (byte-counted SIZE)
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_matrix_transpose() {
        let content = r#"
#define MATRIX_ROWS 7
#define MATRIX_COLS 19
#define SIZE (MATRIX_ROWS * MATRIX_COLS * sizeof(int32_t))

auto bo_in =
    xrt::bo(device, SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
auto bo_out =
    xrt::bo(device, SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

int32_t *buf_in = bo_in.map<int32_t *>();
for (int i = 0; i < SIZE / sizeof(buf_in[0]); i++) {
  buf_in[i] = i;
}
int32_t *buf_out = bo_out.map<int32_t *>();
memset(buf_out, 0, SIZE);

bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
"#;

        let spec = parse_test_cpp_content(content).unwrap();
        assert_eq!(spec.buffers.len(), 2);

        let bo_in = spec.buffers.iter().find(|b| b.name == "bo_in").unwrap();
        assert_eq!(bo_in.group_id, 3);
        assert_eq!(bo_in.direction, BufferDir::Input);
        // SIZE = 7 * 19 * 4 = 532 bytes, / 4 = 133 elements
        assert_eq!(bo_in.size_elements, 133);
        assert_eq!(bo_in.input_pattern, InputPattern::Sequential { start: 0, step: 1 });

        let bo_out = spec.buffers.iter().find(|b| b.name == "bo_out").unwrap();
        assert_eq!(bo_out.group_id, 4);
        assert_eq!(bo_out.direction, BufferDir::Output);
    }

    // ---------------------------------------------------------------
    // Multi-kernel detection
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_multi_kernel() {
        let content = r#"
constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;

auto bo0_inA = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(3));
auto bo0_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(5));
auto bo1_inA = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(3));
auto bo1_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(5));
"#;

        let spec = parse_test_cpp_content(content).unwrap();
        assert!(spec.multi_kernel);
        assert_eq!(spec.buffers.len(), 4);
    }

    // ---------------------------------------------------------------
    // Missing test.cpp
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_missing_test_cpp() {
        let result = parse_test_cpp(Path::new("/nonexistent/path"));
        assert!(result.is_none());
    }

    // ---------------------------------------------------------------
    // Data generation
    // ---------------------------------------------------------------

    #[test]
    fn test_generate_sequential_data() {
        let buf = BufferDef {
            name: "test".to_string(),
            group_id: 3,
            size_elements: 4,
            element_type: ElementType::I32,
            direction: BufferDir::Input,
            input_pattern: InputPattern::Sequential { start: 1, step: 1 },
        };
        let data = generate_input_data(&buf);
        assert_eq!(data.len(), 16); // 4 * 4 bytes
        let values = read_values(&data, ElementType::I32);
        assert_eq!(values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_generate_constant_data() {
        let buf = BufferDef {
            name: "test".to_string(),
            group_id: 3,
            size_elements: 4,
            element_type: ElementType::I8,
            direction: BufferDir::Input,
            input_pattern: InputPattern::Constant(1),
        };
        let data = generate_input_data(&buf);
        assert_eq!(data.len(), 4); // 4 * 1 byte
        assert_eq!(data, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_generate_zeros_data() {
        let buf = BufferDef {
            name: "test".to_string(),
            group_id: 5,
            size_elements: 4,
            element_type: ElementType::I32,
            direction: BufferDir::Output,
            input_pattern: InputPattern::Zeros,
        };
        let data = generate_input_data(&buf);
        assert_eq!(data.len(), 16);
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_read_values_i32() {
        let data: Vec<u8> = [1i32, 2, 3, 4].iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let values = read_values(&data, ElementType::I32);
        assert_eq!(values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_read_values_u8() {
        let data = vec![0u8, 1, 128, 255];
        let values = read_values(&data, ElementType::U8);
        assert_eq!(values, vec![0, 1, 128, 255]);
    }

    // ---------------------------------------------------------------
    // Parse real test.cpp files from mlir-aie (integration tests)
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_real_add_one_using_dma() {
        let test_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/test/npu-xrt/add_one_using_dma");
        if !test_dir.exists() {
            return; // Skip if mlir-aie not present
        }

        let spec = parse_test_cpp(&test_dir).unwrap();
        assert!(!spec.multi_kernel);
        assert_eq!(spec.buffers.len(), 3);

        let in_a = spec.buffers.iter().find(|b| b.group_id == 3).unwrap();
        assert_eq!(in_a.size_elements, 64);
        assert_eq!(in_a.element_type, ElementType::I32);
        assert_eq!(in_a.direction, BufferDir::Input);
        assert_eq!(in_a.input_pattern, InputPattern::Sequential { start: 1, step: 1 });

        let out = spec.buffers.iter().find(|b| b.group_id == 5).unwrap();
        assert_eq!(out.size_elements, 64);
        assert_eq!(out.direction, BufferDir::Output);
    }

    #[test]
    fn test_parse_real_vector_scalar_using_dma() {
        let test_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/test/npu-xrt/vector_scalar_using_dma");
        if !test_dir.exists() {
            return;
        }

        let spec = parse_test_cpp(&test_dir).unwrap();
        assert!(!spec.multi_kernel);

        let in_a = spec.buffers.iter().find(|b| b.group_id == 3).unwrap();
        assert_eq!(in_a.size_elements, 4096);
        assert_eq!(in_a.element_type, ElementType::I32);
        assert_eq!(in_a.input_pattern, InputPattern::Sequential { start: 1, step: 1 });
    }

    #[test]
    fn test_parse_real_matrix_transpose() {
        let test_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/test/npu-xrt/matrix_transpose");
        if !test_dir.exists() {
            return;
        }

        let spec = parse_test_cpp(&test_dir).unwrap();
        assert_eq!(spec.buffers.len(), 2);

        let bo_in = spec.buffers.iter().find(|b| b.group_id == 3).unwrap();
        assert_eq!(bo_in.direction, BufferDir::Input);
        // 7 * 19 = 133 elements (SIZE is in bytes, parser divides by element size)
        assert_eq!(bo_in.size_elements, 133);
    }

    #[test]
    fn test_parse_real_init_values_repeat() {
        let test_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/test/npu-xrt/objectfifo_repeat/init_values_repeat");
        if !test_dir.exists() {
            return;
        }

        let spec = parse_test_cpp(&test_dir).unwrap();
        assert_eq!(spec.buffers.len(), 3);

        // bo_inA: N=4096 elements (from cxxopts default)
        let in_a = spec.buffers.iter().find(|b| b.name == "bo_inA").unwrap();
        assert_eq!(in_a.group_id, 3);
        assert_eq!(in_a.size_elements, 4096);
        assert_eq!(in_a.element_type, ElementType::I32);
        assert_eq!(in_a.direction, BufferDir::Input);

        // bo_out: N * repeat_count = 4096 * 4 = 16384 elements
        let out = spec.buffers.iter().find(|b| b.name == "bo_out").unwrap();
        assert_eq!(out.group_id, 5);
        assert_eq!(out.size_elements, 16384);
        assert_eq!(out.element_type, ElementType::I32);
        assert_eq!(out.direction, BufferDir::Output);
    }

    #[test]
    fn test_cxxopts_default_extraction() {
        let content = r#"
options.add_options()("length,l", "the length", cxxopts::value<int>()->default_value("4096"))(
    "repeat,r", "the repeat count", cxxopts::value<int>()->default_value("4"));

int N = vm["length"].as<int>();
int repeat_count = vm["repeat"].as<int>();

auto bo_inA = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
auto bo_out = xrt::bo(device, N * repeat_count * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

int32_t *bufInA = bo_inA.map<int32_t *>();
std::vector<uint32_t> srcVecA;
for (int i = 0; i < N; i++)
  srcVecA.push_back(i + 1);
memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
"#;

        let spec = parse_test_cpp_content(content).unwrap();
        assert_eq!(spec.buffers.len(), 2);

        let in_a = spec.buffers.iter().find(|b| b.name == "bo_inA").unwrap();
        assert_eq!(in_a.size_elements, 4096);
        assert_eq!(in_a.element_type, ElementType::I32);
        assert_eq!(in_a.direction, BufferDir::Input);

        let out = spec.buffers.iter().find(|b| b.name == "bo_out").unwrap();
        assert_eq!(out.size_elements, 4096 * 4);
        assert_eq!(out.element_type, ElementType::I32);
        assert_eq!(out.direction, BufferDir::Output);
    }

    #[test]
    fn test_using_type_alias_in_sizeof() {
        let content = r#"
using A_DATATYPE = std::int32_t;
using B_DATATYPE = std::int32_t;
using C_DATATYPE = std::int32_t;

constexpr int M = 16;
constexpr int K = 16;
constexpr int N = 16;

constexpr int A_VOLUME = M * K;
constexpr int C_VOLUME = M * N;

constexpr int A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
constexpr int C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));

auto bo_a = xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
auto bo_c = xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
"#;

        let spec = parse_test_cpp_content(content).unwrap();
        assert_eq!(spec.buffers.len(), 2);

        // A_SIZE = 16 * 16 * 4 = 1024 bytes, / 4 = 256 elements
        let bo_a = spec.buffers.iter().find(|b| b.name == "bo_a").unwrap();
        assert_eq!(bo_a.size_elements, 256);
        assert_eq!(bo_a.element_type, ElementType::I32);
        assert_eq!(bo_a.direction, BufferDir::Input);

        // C_SIZE = 16 * 16 * 4 = 1024 bytes, / 4 = 256 elements
        let bo_c = spec.buffers.iter().find(|b| b.name == "bo_c").unwrap();
        assert_eq!(bo_c.size_elements, 256);
        assert_eq!(bo_c.element_type, ElementType::I32);
        assert_eq!(bo_c.direction, BufferDir::Output);
    }

    #[test]
    fn test_parse_real_matmul_cascade() {
        let test_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/test/npu-xrt/matrix_multiplication_using_cascade");
        if !test_dir.exists() {
            return;
        }

        let spec = parse_test_cpp(&test_dir).unwrap();
        assert_eq!(spec.buffers.len(), 3);

        // A: 256 elements (16x16), B: 256, C: 256
        let bo_a = spec.buffers.iter().find(|b| b.group_id == 3).unwrap();
        assert_eq!(bo_a.size_elements, 256);
        assert_eq!(bo_a.direction, BufferDir::Input);

        let bo_c = spec.buffers.iter().find(|b| b.group_id == 5).unwrap();
        assert_eq!(bo_c.size_elements, 256);
        assert_eq!(bo_c.direction, BufferDir::Output);
    }
}
