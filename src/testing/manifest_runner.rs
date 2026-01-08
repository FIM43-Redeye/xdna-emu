//! Test runner that executes tests defined by TOML manifests.
//!
//! This module provides infrastructure for running tests extracted from mlir-aie.
//! Tests are defined in TOML manifest files that specify:
//! - Input buffer patterns
//! - Expected output transformations
//! - References to xclbin files
//!
//! # Example Manifest
//!
//! ```toml
//! [test]
//! name = "add_one_using_dma"
//! source_dir = "test/npu-xrt/add_one_using_dma"
//!
//! [build]
//! mlir_file = "aie.mlir"
//! device = "npu1_1col"
//!
//! [buffers.input_a]
//! size = 64
//! element_type = "i32"
//! group_id = 3
//!
//! [buffers.input_a.pattern]
//! type = "sequential"
//! start = 1
//! step = 1
//!
//! [buffers.output]
//! size = 64
//! element_type = "i32"
//! group_id = 5
//!
//! [expected]
//! type = "transform"
//! transform = "input_a + 1"
//! ```

use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Test manifest defining a single test case.
#[derive(Debug, Clone, Deserialize)]
pub struct TestManifest {
    pub test: TestInfo,
    pub build: BuildInfo,
    pub buffers: HashMap<String, BufferDef>,
    pub expected: ExpectedDef,
}

/// Basic test metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct TestInfo {
    pub name: String,
    pub source_dir: String,
    #[serde(default)]
    pub description: String,
}

/// Build configuration for the test.
#[derive(Debug, Clone, Deserialize)]
pub struct BuildInfo {
    pub mlir_file: String,
    pub device: String,
}

/// Buffer definition (input or output).
#[derive(Debug, Clone, Deserialize)]
pub struct BufferDef {
    pub size: usize,
    pub element_type: String,
    pub group_id: u32,
    #[serde(default)]
    pub pattern: Option<PatternDef>,
}

/// Pattern for generating input data.
#[derive(Debug, Clone, Deserialize)]
pub struct PatternDef {
    #[serde(rename = "type")]
    pub pattern_type: String,
    #[serde(default)]
    pub start: i64,
    #[serde(default = "default_step")]
    pub step: i64,
    #[serde(default)]
    pub value: i64,
}

fn default_step() -> i64 {
    1
}

/// Expected output definition.
#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedDef {
    #[serde(rename = "type")]
    pub expected_type: String,
    #[serde(default)]
    pub transform: Option<String>,
    #[serde(default)]
    pub values: Option<Vec<i64>>,
}

/// Element type for buffer data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}

impl ElementType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "i8" => Some(ElementType::I8),
            "u8" => Some(ElementType::U8),
            "i16" => Some(ElementType::I16),
            "u16" => Some(ElementType::U16),
            "i32" => Some(ElementType::I32),
            "u32" => Some(ElementType::U32),
            "i64" => Some(ElementType::I64),
            "u64" => Some(ElementType::U64),
            _ => None,
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            ElementType::I8 | ElementType::U8 => 1,
            ElementType::I16 | ElementType::U16 => 2,
            ElementType::I32 | ElementType::U32 => 4,
            ElementType::I64 | ElementType::U64 => 8,
        }
    }
}

/// Result of running a test.
#[derive(Debug)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub correct_count: usize,
    pub total_count: usize,
    pub first_mismatch: Option<MismatchInfo>,
    pub error: Option<String>,
}

/// Information about a mismatched output value.
#[derive(Debug)]
pub struct MismatchInfo {
    pub index: usize,
    pub expected: i64,
    pub actual: i64,
}

impl TestManifest {
    /// Load a manifest from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read manifest: {}", e))?;
        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse manifest: {}", e))
    }

    /// Generate input data for a buffer based on its pattern.
    pub fn generate_input(&self, buffer_name: &str) -> Option<Vec<u8>> {
        let buffer = self.buffers.get(buffer_name)?;
        let elem_type = ElementType::from_str(&buffer.element_type)?;
        let pattern = buffer.pattern.as_ref()?;

        let mut data = Vec::with_capacity(buffer.size * elem_type.byte_size());

        match pattern.pattern_type.as_str() {
            "sequential" => {
                for i in 0..buffer.size {
                    let value = pattern.start + (i as i64) * pattern.step;
                    append_value(&mut data, value, elem_type);
                }
            }
            "constant" => {
                for _ in 0..buffer.size {
                    append_value(&mut data, pattern.value, elem_type);
                }
            }
            "zeros" => {
                data.resize(buffer.size * elem_type.byte_size(), 0);
            }
            _ => return None,
        }

        Some(data)
    }

    /// Generate expected output data based on the transform.
    pub fn generate_expected(&self, inputs: &HashMap<String, Vec<i64>>) -> Option<Vec<i64>> {
        let output_buf = self.buffers.get("output")?;

        match self.expected.expected_type.as_str() {
            "transform" => {
                let transform = self.expected.transform.as_ref()?;
                self.apply_transform(transform, inputs, output_buf.size)
            }
            "values" => self.expected.values.clone(),
            _ => None,
        }
    }

    /// Apply a simple transform expression.
    /// Supports: "input_a + N", "input_a - N", "input_a * N", "index + N"
    fn apply_transform(
        &self,
        transform: &str,
        inputs: &HashMap<String, Vec<i64>>,
        output_size: usize,
    ) -> Option<Vec<i64>> {
        // Parse simple expressions like "input_a + 41"
        let parts: Vec<&str> = transform.split_whitespace().collect();

        if parts.len() == 3 {
            let operand = parts[0];
            let op = parts[1];
            let constant: i64 = parts[2].parse().ok()?;

            let base_values: Vec<i64> = if operand == "index" {
                (0..output_size as i64).collect()
            } else if let Some(input) = inputs.get(operand) {
                input.clone()
            } else {
                return None;
            };

            let result: Vec<i64> = base_values
                .iter()
                .map(|&v| match op {
                    "+" => v + constant,
                    "-" => v - constant,
                    "*" => v * constant,
                    "/" => v / constant,
                    _ => v,
                })
                .collect();

            Some(result)
        } else if parts.len() == 1 {
            // Just a constant
            let constant: i64 = parts[0].parse().ok()?;
            Some(vec![constant; output_size])
        } else {
            None
        }
    }

    /// Get the input buffer definition.
    pub fn get_input(&self, name: &str) -> Option<&BufferDef> {
        self.buffers.get(name)
    }

    /// Get the output buffer definition.
    pub fn get_output(&self) -> Option<&BufferDef> {
        self.buffers.get("output")
    }
}

/// Append a value to a byte vector in little-endian format.
fn append_value(data: &mut Vec<u8>, value: i64, elem_type: ElementType) {
    match elem_type {
        ElementType::I8 => data.push(value as i8 as u8),
        ElementType::U8 => data.push(value as u8),
        ElementType::I16 => data.extend_from_slice(&(value as i16).to_le_bytes()),
        ElementType::U16 => data.extend_from_slice(&(value as u16).to_le_bytes()),
        ElementType::I32 => data.extend_from_slice(&(value as i32).to_le_bytes()),
        ElementType::U32 => data.extend_from_slice(&(value as u32).to_le_bytes()),
        ElementType::I64 => data.extend_from_slice(&value.to_le_bytes()),
        ElementType::U64 => data.extend_from_slice(&(value as u64).to_le_bytes()),
    }
}

/// Read values from a byte slice.
pub fn read_values(data: &[u8], elem_type: ElementType) -> Vec<i64> {
    let elem_size = elem_type.byte_size();
    let count = data.len() / elem_size;
    let mut values = Vec::with_capacity(count);

    for i in 0..count {
        let offset = i * elem_size;
        let value = match elem_type {
            ElementType::I8 => data[offset] as i8 as i64,
            ElementType::U8 => data[offset] as i64,
            ElementType::I16 => {
                i16::from_le_bytes([data[offset], data[offset + 1]]) as i64
            }
            ElementType::U16 => {
                u16::from_le_bytes([data[offset], data[offset + 1]]) as i64
            }
            ElementType::I32 => {
                i32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) as i64
            }
            ElementType::U32 => {
                u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) as i64
            }
            ElementType::I64 => i64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]),
            ElementType::U64 => {
                u64::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]) as i64
            }
        };
        values.push(value);
    }

    values
}

/// Test runner that executes manifest-based tests.
pub struct ManifestRunner {
    /// Path to mlir-aie repository (for finding xclbin files).
    mlir_aie_path: Option<PathBuf>,
    /// Path to xclbin cache directory.
    xclbin_cache: Option<PathBuf>,
}

impl ManifestRunner {
    /// Create a new manifest runner.
    pub fn new() -> Self {
        Self {
            mlir_aie_path: None,
            xclbin_cache: None,
        }
    }

    /// Set the path to the mlir-aie repository.
    pub fn with_mlir_aie_path(mut self, path: PathBuf) -> Self {
        self.mlir_aie_path = Some(path);
        self
    }

    /// Set the path to the xclbin cache directory.
    pub fn with_xclbin_cache(mut self, path: PathBuf) -> Self {
        self.xclbin_cache = Some(path);
        self
    }

    /// Find the xclbin file for a test.
    pub fn find_xclbin(&self, manifest: &TestManifest) -> Option<PathBuf> {
        // Try cache first
        if let Some(cache) = &self.xclbin_cache {
            let cached = cache.join(&manifest.test.name).join("aie.xclbin");
            if cached.exists() {
                return Some(cached);
            }
        }

        // Try mlir-aie build directory
        if let Some(mlir_aie) = &self.mlir_aie_path {
            // Build output is typically in build/test/npu-xrt/<test_name>/
            let build_path = mlir_aie
                .join("build")
                .join(&manifest.test.source_dir)
                .join("aie.xclbin");
            if build_path.exists() {
                return Some(build_path);
            }

            // Also check for the test directly in test directory
            let direct_path = mlir_aie
                .join(&manifest.test.source_dir)
                .join("build")
                .join("aie.xclbin");
            if direct_path.exists() {
                return Some(direct_path);
            }
        }

        None
    }
}

impl Default for ManifestRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_manifest() {
        let toml_content = r#"
[test]
name = "add_one_using_dma"
source_dir = "test/npu-xrt/add_one_using_dma"
description = "Test description"

[build]
mlir_file = "aie.mlir"
device = "npu1_1col"

[buffers.input_a]
size = 64
element_type = "i32"
group_id = 3

[buffers.input_a.pattern]
type = "sequential"
start = 1
step = 1

[buffers.output]
size = 64
element_type = "i32"
group_id = 5

[expected]
type = "transform"
transform = "input_a + 1"
"#;

        let manifest: TestManifest = toml::from_str(toml_content).unwrap();
        assert_eq!(manifest.test.name, "add_one_using_dma");
        assert_eq!(manifest.buffers.len(), 2);
        assert!(manifest.buffers.contains_key("input_a"));
        assert!(manifest.buffers.contains_key("output"));
    }

    #[test]
    fn test_generate_sequential_input() {
        let toml_content = r#"
[test]
name = "test"
source_dir = "test"

[build]
mlir_file = "aie.mlir"
device = "npu1_1col"

[buffers.input_a]
size = 4
element_type = "i32"
group_id = 3

[buffers.input_a.pattern]
type = "sequential"
start = 1
step = 1

[buffers.output]
size = 4
element_type = "i32"
group_id = 5

[expected]
type = "transform"
transform = "input_a + 41"
"#;

        let manifest: TestManifest = toml::from_str(toml_content).unwrap();
        let input = manifest.generate_input("input_a").unwrap();

        // Should generate [1, 2, 3, 4] as i32 little-endian
        assert_eq!(input.len(), 16); // 4 elements * 4 bytes
        let values = read_values(&input, ElementType::I32);
        assert_eq!(values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_apply_transform() {
        let toml_content = r#"
[test]
name = "test"
source_dir = "test"

[build]
mlir_file = "aie.mlir"
device = "npu1_1col"

[buffers.input_a]
size = 4
element_type = "i32"
group_id = 3

[buffers.input_a.pattern]
type = "sequential"
start = 1
step = 1

[buffers.output]
size = 4
element_type = "i32"
group_id = 5

[expected]
type = "transform"
transform = "input_a + 41"
"#;

        let manifest: TestManifest = toml::from_str(toml_content).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("input_a".to_string(), vec![1, 2, 3, 4]);

        let expected = manifest.generate_expected(&inputs).unwrap();
        assert_eq!(expected, vec![42, 43, 44, 45]);
    }

    #[test]
    fn test_element_types() {
        assert_eq!(ElementType::from_str("i8"), Some(ElementType::I8));
        assert_eq!(ElementType::from_str("i32"), Some(ElementType::I32));
        assert_eq!(ElementType::from_str("u64"), Some(ElementType::U64));
        assert_eq!(ElementType::from_str("invalid"), None);

        assert_eq!(ElementType::I8.byte_size(), 1);
        assert_eq!(ElementType::I32.byte_size(), 4);
        assert_eq!(ElementType::I64.byte_size(), 8);
    }
}
