//! AM025 register database parser.
//!
//! Loads register definitions from `aie_registers_aie2.json`, a structured
//! JSON file with 1,806 registers and 6,412 bit fields parsed from AMD's
//! AM025 (AIE-ML Register Reference) documentation.
//!
//! Data-model file: bitfield helpers (`extract`, `insert`, `set_bit`) and
//! query helpers (`non_zero_reset_values`, `registers_with_access`) are
//! intentional API for whichever consumer reaches for them next.
//! `dead_code` tolerated here -- delete explicitly if a method is
//! genuinely retired.
#![allow(dead_code)]
//!
//! This module provides the base parsing types used by both the architecture
//! graph (for extraction and cross-validation) and the emulator (for runtime
//! register access). The emulator extends these types with pre-resolved
//! field layouts for hot-path performance.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// JSON deserialization types (matching aie_registers_aie2.json schema)
// ============================================================================

/// Raw JSON structure matching the top level of aie_registers_aie2.json.
#[derive(Debug, Deserialize)]
struct RawRegisterDb {
    version: String,
    #[allow(dead_code)]
    source: Option<String>,
    #[allow(dead_code)]
    parsed_date: Option<String>,
    modules: HashMap<String, RawModule>,
}

/// A module section in the JSON (core, memory, memory_tile, shim).
#[derive(Debug, Deserialize)]
struct RawModule {
    registers: Vec<RawRegister>,
}

/// A single register definition in the JSON.
#[derive(Debug, Deserialize)]
struct RawRegister {
    name: String,
    /// Hex string like "0x000001D000"
    offset: String,
    /// Register width in bits (default 32).
    width: Option<u32>,
    /// Access mode string (e.g. "rwNormal read/write", "roRead-only").
    #[serde(rename = "type")]
    access_type: Option<String>,
    /// Reset value as hex string (e.g. "0x00000000").
    reset: Option<String>,
    /// Prose description (may encode address-mapped fields for special registers).
    #[serde(default)]
    description: Option<String>,
    bit_fields: Vec<RawBitField>,
}

/// A single bit field within a register.
#[derive(Debug, Deserialize)]
struct RawBitField {
    name: String,
    /// Array [lsb, msb], e.g. [14, 27] for bits 27:14
    bit_range: Vec<u32>,
}

// ============================================================================
// Processed types (efficient access)
// ============================================================================

/// A single bit field within a register.
///
/// Pre-computed mask and shift enable O(1) field extraction with the same
/// performance as hand-coded `(value >> SHIFT) & MASK` constants.
#[derive(Debug, Clone)]
pub struct BitField {
    pub name: String,
    pub lsb: u8,
    pub msb: u8,
    pub width: u8,
    /// Pre-computed mask: `(1 << width) - 1`
    pub mask: u32,
    /// Shift amount (same as lsb)
    pub shift: u8,
}

impl BitField {
    /// Extract this field's value from a raw 32-bit register value.
    #[inline]
    pub fn extract(&self, value: u32) -> u32 {
        (value >> self.shift) & self.mask
    }

    /// Extract this field as a boolean (for single-bit fields).
    #[inline]
    pub fn extract_bool(&self, value: u32) -> bool {
        self.extract(value) != 0
    }

    /// Insert a value into a register word at this field's position.
    ///
    /// Clears the field's bits in `word`, then ORs in `value` (masked to
    /// field width). Returns the modified word.
    #[inline]
    pub fn insert(&self, word: u32, value: u32) -> u32 {
        let cleared = word & !(self.mask << self.shift);
        cleared | ((value & self.mask) << self.shift)
    }

    /// Set this field's single bit to 1 in a register word.
    ///
    /// For multi-bit fields, sets only the LSB of the field range.
    #[inline]
    pub fn set_bit(&self, word: u32) -> u32 {
        word | (1 << self.shift)
    }

    /// Build a BitField from LSB and MSB bit positions.
    pub fn from_range(name: String, lsb: u8, msb: u8) -> Self {
        let width = msb - lsb + 1;
        let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
        Self { name, lsb, msb, width, mask, shift: lsb }
    }
}

/// Register access mode parsed from the JSON "type" field.
///
/// These come from the AM025 register reference and describe how the
/// hardware responds to reads and writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Normal read/write register.
    ReadWrite,
    /// Read-only register (writes ignored by hardware).
    ReadOnly,
    /// Write-only register (reads return reset value or 0).
    WriteOnly,
    /// Readable, write-1-to-clear: writing a 1 to a bit clears it.
    WriteToClear,
    /// Mixed: individual bit fields have different access modes.
    Mixed,
}

impl AccessMode {
    /// Parse from the JSON type string.
    pub fn from_json(s: &str) -> Self {
        if s.starts_with("rw") {
            AccessMode::ReadWrite
        } else if s.starts_with("ro") {
            AccessMode::ReadOnly
        } else if s.starts_with("wo") {
            AccessMode::WriteOnly
        } else if s.starts_with("wtc") {
            AccessMode::WriteToClear
        } else if s.starts_with("mixed") {
            AccessMode::Mixed
        } else {
            AccessMode::ReadWrite
        }
    }
}

/// A register definition with its offset, access mode, reset value,
/// and bit fields.
#[derive(Debug, Clone)]
pub struct RegisterDef {
    pub name: String,
    /// Byte offset within the module's address space
    pub offset: u32,
    /// Register width in bits (default 32).
    pub width: u32,
    /// Access mode (read-write, read-only, write-only, etc.)
    pub access: AccessMode,
    /// Power-on reset value for this register.
    pub reset_value: u32,
    /// Prose description (may encode address-mapped fields for special registers).
    pub description: Option<String>,
    pub fields: Vec<BitField>,
}

impl RegisterDef {
    /// Look up a bit field by name.
    pub fn field(&self, name: &str) -> Option<&BitField> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// A module (core, memory, memory_tile, shim) containing registers.
#[derive(Debug, Clone)]
pub struct ModuleDef {
    pub name: String,
    pub registers: Vec<RegisterDef>,
    /// Index by name for O(1) lookup
    register_index: HashMap<String, usize>,
}

impl ModuleDef {
    /// Look up a register by name.
    pub fn register(&self, name: &str) -> Option<&RegisterDef> {
        self.register_index.get(name).map(|&i| &self.registers[i])
    }

    /// Iterate over (offset, reset_value) pairs for registers with non-zero
    /// reset values.
    pub fn non_zero_reset_values(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.registers.iter()
            .filter(|r| r.reset_value != 0)
            .map(|r| (r.offset, r.reset_value))
    }

    /// Get all registers with a specific access mode.
    pub fn registers_with_access(&self, mode: AccessMode) -> impl Iterator<Item = &RegisterDef> {
        self.registers.iter().filter(move |r| r.access == mode)
    }
}

/// The complete register database for one architecture.
///
/// Loaded from the AMD AM025 JSON register reference.
#[derive(Debug, Clone)]
pub struct RegisterDb {
    pub version: String,
    pub modules: HashMap<String, ModuleDef>,
}

impl RegisterDb {
    /// Load a register database from a JSON file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        Self::from_json(&data)
    }

    /// Parse a register database from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let raw: RawRegisterDb = serde_json::from_str(json)
            .map_err(|e| format!("Failed to parse register database JSON: {}", e))?;

        let mut modules = HashMap::new();

        for (mod_name, raw_module) in raw.modules {
            let mut registers = Vec::with_capacity(raw_module.registers.len());
            let mut register_index = HashMap::new();

            for raw_reg in raw_module.registers {
                let offset = parse_hex_offset(&raw_reg.offset)?;

                let fields: Vec<BitField> = raw_reg.bit_fields.iter()
                    .filter(|f| f.name != "Reserved")
                    .map(|f| {
                        if f.bit_range.len() != 2 {
                            return Err(format!(
                                "Invalid bit_range for {}.{}: expected [lsb, msb]",
                                raw_reg.name, f.name
                            ));
                        }
                        let lsb = f.bit_range[0] as u8;
                        let msb = f.bit_range[1] as u8;
                        Ok(BitField::from_range(f.name.clone(), lsb, msb))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let width = raw_reg.width.unwrap_or(32);
                let access = AccessMode::from_json(
                    raw_reg.access_type.as_deref().unwrap_or("rwNormal")
                );
                let reset_value = raw_reg.reset.as_deref()
                    .map(parse_reset_value)
                    .unwrap_or(0);

                let idx = registers.len();
                register_index.insert(raw_reg.name.clone(), idx);
                registers.push(RegisterDef {
                    name: raw_reg.name,
                    offset,
                    width,
                    access,
                    reset_value,
                    description: raw_reg.description,
                    fields,
                });
            }

            modules.insert(mod_name.clone(), ModuleDef {
                name: mod_name,
                registers,
                register_index,
            });
        }

        Ok(Self {
            version: raw.version,
            modules,
        })
    }

    /// Get a module by name.
    pub fn module(&self, name: &str) -> Option<&ModuleDef> {
        self.modules.get(name)
    }
}

/// Parse a hex string like "0x000001D000" into a u32 offset.
///
/// The JSON uses full 40-bit addresses but we only need the 20-bit tile-local
/// offset (the lower portion). Since tile-local offsets fit in u32, we parse
/// as u64 then truncate.
fn parse_hex_offset(s: &str) -> Result<u32, String> {
    let hex_str = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))
        .ok_or_else(|| format!("Expected hex string, got: {}", s))?;
    let full = u64::from_str_radix(hex_str, 16)
        .map_err(|e| format!("Invalid hex '{}': {}", s, e))?;
    Ok(full as u32)
}

/// Parse a hex reset value string like "0x000006DB" into a u32.
fn parse_reset_value(s: &str) -> u32 {
    let hex_str = s.strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u32::from_str_radix(hex_str, 16).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_offsets() {
        assert_eq!(parse_hex_offset("0x000001D000").unwrap(), 0x1D000);
        assert_eq!(parse_hex_offset("0x000001F000").unwrap(), 0x1F000);
        assert_eq!(parse_hex_offset("0x0000000000").unwrap(), 0x0);
        assert_eq!(parse_hex_offset("0x00000A0000").unwrap(), 0xA0000);
    }

    #[test]
    fn parse_reset_values() {
        assert_eq!(parse_reset_value("0x00000000"), 0);
        assert_eq!(parse_reset_value("0x000006DB"), 0x6DB);
        assert_eq!(parse_reset_value("0xFFFFFFFF"), 0xFFFFFFFF);
        assert_eq!(parse_reset_value("0x00000002"), 2);
        assert_eq!(parse_reset_value("not_hex"), 0);
    }

    #[test]
    fn access_mode_parsing() {
        assert_eq!(AccessMode::from_json("rwNormal read/write"), AccessMode::ReadWrite);
        assert_eq!(AccessMode::from_json("roRead-only"), AccessMode::ReadOnly);
        assert_eq!(AccessMode::from_json("woWrite-only"), AccessMode::WriteOnly);
        assert_eq!(AccessMode::from_json("wtcReadable, write a 1 to clear"), AccessMode::WriteToClear);
        assert_eq!(AccessMode::from_json("mixedMixed types"), AccessMode::Mixed);
        assert_eq!(AccessMode::from_json("unknown"), AccessMode::ReadWrite);
    }
}
