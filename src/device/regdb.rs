//! Data-driven register database loaded from AMD AM025 JSON.
//!
//! This module loads register definitions from `aie_registers_aie2.json`,
//! a structured JSON file with 1,806 registers and 6,412 bit fields parsed
//! from AMD's AM025 (AIE-ML Register Reference) documentation.
//!
//! # Why data-driven?
//!
//! The emulator previously hand-coded every register offset and bit mask in
//! `registers_spec.rs`, transcribed from prose documentation. This was
//! error-prone and hard to keep in sync across architecture revisions.
//!
//! By loading from the same JSON that mlir-aie uses, we get:
//! - **Correctness by construction**: bit layouts come from AMD's own docs
//! - **Multi-architecture readiness**: load different JSON for AIE2 vs AIE2P
//! - **Cross-validation**: compare JSON values against our hand-coded constants
//!
//! # Performance
//!
//! Field extraction uses pre-computed masks and shifts, so it's zero-overhead
//! compared to hand-coded constants. The `BitField::extract()` method compiles
//! down to a shift and AND, same as `(value >> SHIFT) & MASK`.
//!
//! # Example
//!
//! ```ignore
//! let db = RegisterDb::load_for_device("aie2").unwrap();
//! let mem = db.module("memory").unwrap();
//! let bd0_0 = mem.register("DMA_BD0_0").unwrap();
//! let buf_len = bd0_0.field("Buffer_Length").unwrap();
//! assert_eq!(buf_len.extract(0x0000FFFF), 0x3FFF);
//! ```

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
// Processed runtime types (efficient access)
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
    fn from_range(name: String, lsb: u8, msb: u8) -> Self {
        let width = msb - lsb + 1;
        let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
        Self { name, lsb, msb, width, mask, shift: lsb }
    }
}

/// Register access mode parsed from the JSON "type" field.
///
/// These come from the AM025 register reference and describe how the
/// hardware responds to reads and writes. The emulator uses them to
/// enforce correct behavior (e.g., ignoring writes to read-only registers).
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
    fn from_json(s: &str) -> Self {
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
            // Default to ReadWrite for unknown types
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
    /// reset values. Used during tile initialization to set power-on state.
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

    /// Load from the mlir-aie install, using the config system.
    ///
    /// Resolves the JSON path via `Config::mlir_aie_subpath()`.
    pub fn load_for_device(device: &str) -> Result<Self, String> {
        let config = crate::config::Config::get();
        let json_path = config.mlir_aie_subpath(
            &format!("lib/Dialect/AIE/Util/aie_registers_{}.json", device)
        );
        Self::from_file(&json_path)
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
    // Tile-local offsets are 20 bits (TILE_OFFSET_MASK = 0xFFFFF)
    Ok(full as u32)
}

/// Parse a hex reset value string like "0x000006DB" into a u32.
///
/// Returns 0 for unparseable values (defensive -- the JSON is well-formed
/// but we don't want to panic on a missing or malformed reset field).
fn parse_reset_value(s: &str) -> u32 {
    let hex_str = s.strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u32::from_str_radix(hex_str, 16).unwrap_or(0)
}

// ============================================================================
// Pre-resolved field layouts for hot-path register parsing
// ============================================================================

/// Pre-resolved layout for compute tile BD registers (6 words).
///
/// Avoids repeated string lookups during BD parsing. Each field is a cloned
/// `BitField` with pre-computed mask/shift for O(1) extraction.
///
/// BD word layout (AM025 memory_module/dma/bd.txt):
/// - Word 0: Base_Address, Buffer_Length
/// - Word 1: Enable_Compression, Enable_Packet, Out_Of_Order_BD_ID, Packet_ID, Packet_Type
/// - Word 2: D0_Stepsize, D1_Stepsize
/// - Word 3: D0_Wrap, D1_Wrap, D2_Stepsize
/// - Word 4: Iteration_Current, Iteration_Wrap, Iteration_Stepsize
/// - Word 5: TLAST_Suppress, Next_BD, Use_Next_BD, Valid_BD, lock fields
#[derive(Debug, Clone)]
pub struct BdFieldLayout {
    // Word 0
    pub buffer_length: BitField,
    pub base_address: BitField,
    // Word 1
    pub enable_compression: BitField,
    pub enable_packet: BitField,
    pub out_of_order_bd_id: BitField,
    pub packet_id: BitField,
    pub packet_type: BitField,
    // Word 2
    pub d0_stepsize: BitField,
    pub d1_stepsize: BitField,
    // Word 3
    pub d0_wrap: BitField,
    pub d1_wrap: BitField,
    pub d2_stepsize: BitField,
    // Word 4
    pub iteration_current: BitField,
    pub iteration_wrap: BitField,
    pub iteration_stepsize: BitField,
    // Word 5
    pub tlast_suppress: BitField,
    pub next_bd: BitField,
    pub use_next_bd: BitField,
    pub valid_bd: BitField,
    pub lock_rel_value: BitField,
    pub lock_rel_id: BitField,
    pub lock_acq_enable: BitField,
    pub lock_acq_value: BitField,
    pub lock_acq_id: BitField,
}

impl BdFieldLayout {
    /// Build from the register database for a given module.
    ///
    /// Resolves all BD fields from DMA_BD0_0 through DMA_BD0_5 registers.
    pub fn from_regdb(db: &RegisterDb, module: &str) -> Result<Self, String> {
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        // Helper to get a field from a register, with clear error messages
        let get_field = |reg_name: &str, field_name: &str| -> Result<BitField, String> {
            let reg = m.register(reg_name)
                .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            // Word 0: DMA_BD0_0
            buffer_length: get_field("DMA_BD0_0", "Buffer_Length")?,
            base_address: get_field("DMA_BD0_0", "Base_Address")?,
            // Word 1: DMA_BD0_1
            enable_compression: get_field("DMA_BD0_1", "Enable_Compression")?,
            enable_packet: get_field("DMA_BD0_1", "Enable_Packet")?,
            out_of_order_bd_id: get_field("DMA_BD0_1", "Out_Of_Order_BD_ID")?,
            packet_id: get_field("DMA_BD0_1", "Packet_ID")?,
            packet_type: get_field("DMA_BD0_1", "Packet_Type")?,
            // Word 2: DMA_BD0_2
            d0_stepsize: get_field("DMA_BD0_2", "D0_Stepsize")?,
            d1_stepsize: get_field("DMA_BD0_2", "D1_Stepsize")?,
            // Word 3: DMA_BD0_3
            d0_wrap: get_field("DMA_BD0_3", "D0_Wrap")?,
            d1_wrap: get_field("DMA_BD0_3", "D1_Wrap")?,
            d2_stepsize: get_field("DMA_BD0_3", "D2_Stepsize")?,
            // Word 4: DMA_BD0_4
            iteration_current: get_field("DMA_BD0_4", "Iteration_Current")?,
            iteration_wrap: get_field("DMA_BD0_4", "Iteration_Wrap")?,
            iteration_stepsize: get_field("DMA_BD0_4", "Iteration_Stepsize")?,
            // Word 5: DMA_BD0_5
            tlast_suppress: get_field("DMA_BD0_5", "TLAST_Suppress")?,
            next_bd: get_field("DMA_BD0_5", "Next_BD")?,
            use_next_bd: get_field("DMA_BD0_5", "Use_Next_BD")?,
            valid_bd: get_field("DMA_BD0_5", "Valid_BD")?,
            lock_rel_value: get_field("DMA_BD0_5", "Lock_Rel_Value")?,
            lock_rel_id: get_field("DMA_BD0_5", "Lock_Rel_ID")?,
            lock_acq_enable: get_field("DMA_BD0_5", "Lock_Acq_Enable")?,
            lock_acq_value: get_field("DMA_BD0_5", "Lock_Acq_Value")?,
            lock_acq_id: get_field("DMA_BD0_5", "Lock_Acq_ID")?,
        })
    }
}

/// Pre-resolved layout for DMA channel control and start queue registers.
///
/// Covers both S2MM_Ctrl and MM2S_Ctrl field layouts (they share the same
/// bit assignments in AM025, differing only in which fields are meaningful).
#[derive(Debug, Clone)]
pub struct ChannelFieldLayout {
    // Control register fields
    pub fot_mode: BitField,
    pub controller_id: BitField,
    pub decompression_enable: BitField,
    pub enable_out_of_order: BitField,
    pub reset: BitField,
    // Start queue register fields
    pub enable_token_issue: BitField,
    pub repeat_count: BitField,
    pub start_bd_id: BitField,
}

impl ChannelFieldLayout {
    /// Build from the register database for a given module.
    ///
    /// Uses DMA_S2MM_0_Ctrl and DMA_S2MM_0_Start_Queue as the canonical
    /// source (all channels share the same field layout).
    pub fn from_regdb(db: &RegisterDb, module: &str) -> Result<Self, String> {
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        let get_field = |reg_name: &str, field_name: &str| -> Result<BitField, String> {
            let reg = m.register(reg_name)
                .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            fot_mode: get_field("DMA_S2MM_0_Ctrl", "FoT_Mode")?,
            controller_id: get_field("DMA_S2MM_0_Ctrl", "Controller_ID")?,
            decompression_enable: get_field("DMA_S2MM_0_Ctrl", "Decompression_Enable")?,
            enable_out_of_order: get_field("DMA_S2MM_0_Ctrl", "Enable_Out_of_Order")?,
            reset: get_field("DMA_S2MM_0_Ctrl", "Reset")?,
            enable_token_issue: get_field("DMA_S2MM_0_Start_Queue", "Enable_Token_Issue")?,
            repeat_count: get_field("DMA_S2MM_0_Start_Queue", "Repeat_Count")?,
            start_bd_id: get_field("DMA_S2MM_0_Start_Queue", "Start_BD_ID")?,
        })
    }
}

/// Pre-resolved layout for DMA channel status registers.
///
/// Covers `DMA_S2MM_Status_0` / `DMA_MM2S_Status_0` fields. The same
/// field layout applies to all channels within a module (S2MM and MM2S
/// share the same bit assignments for status).
///
/// Key difference between modules: the `cur_bd` field is 4 bits [27:24]
/// for compute tiles (16 BDs) but 6 bits [29:24] for memtiles (48 BDs).
#[derive(Debug, Clone)]
pub struct StatusFieldLayout {
    pub status: BitField,               // [1:0] IDLE/STARTING/RUNNING
    pub stalled_lock_acq: BitField,     // [2]
    pub stalled_lock_rel: BitField,     // [3]
    pub stalled_stream: BitField,       // [4]
    pub stalled_tct: BitField,          // [5]
    pub error_bd_unavailable: BitField, // [10]
    pub error_bd_invalid: BitField,     // [11]
    pub task_queue_overflow: BitField,  // [18]
    pub channel_running: BitField,      // [19]
    pub task_queue_size: BitField,      // [22:20]
    pub cur_bd: BitField,               // [27:24] compute / [29:24] memtile
}

impl StatusFieldLayout {
    /// Build from the register database for a given module.
    ///
    /// Uses `DMA_S2MM_Status_0` as the canonical source (all channels
    /// share the same field layout within a module).
    pub fn from_regdb(db: &RegisterDb, module: &str) -> Result<Self, String> {
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        let reg_name = "DMA_S2MM_Status_0";
        let reg = m.register(reg_name)
            .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;

        let get_field = |field_name: &str| -> Result<BitField, String> {
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            status: get_field("Status")?,
            stalled_lock_acq: get_field("Stalled_Lock_Acq")?,
            stalled_lock_rel: get_field("Stalled_Lock_Rel")?,
            stalled_stream: get_field("Stalled_Stream_Starvation")?,
            stalled_tct: get_field("Stalled_TCT_or_Count_FIFO_Full")?,
            error_bd_unavailable: get_field("Error_BD_Unavailable")?,
            error_bd_invalid: get_field("Error_BD_Invalid")?,
            task_queue_overflow: get_field("Task_Queue_Overflow")?,
            channel_running: get_field("Channel_Running")?,
            task_queue_size: get_field("Task_Queue_Size")?,
            cur_bd: get_field("Cur_BD")?,
        })
    }
}

/// Pre-resolved layout for MemTile BD registers (8 words).
///
/// MemTile BDs use 17-bit stepsizes and 19-bit addresses to cover the 512KB
/// address space. They also support 4-dimensional addressing (D3_Stepsize)
/// and zero-padding (D0/D1/D2 zero before/after) for MM2S transfers.
///
/// BD word layout (AM025 MEMORY_TILE_MODULE/DMA/BD):
/// - Word 0: Enable_Packet, Packet_Type, Packet_ID, OOO_BD_ID, Buffer_Length
/// - Word 1: D0_Zero_Before, Next_BD, Use_Next_BD, Base_Address
/// - Word 2: TLAST_Suppress, D0_Wrap, D0_Stepsize
/// - Word 3: D1_Zero_Before, D1_Wrap, D1_Stepsize
/// - Word 4: Enable_Compression, D2_Zero_Before, D2_Wrap, D2_Stepsize
/// - Word 5: D2_Zero_After, D1_Zero_After, D0_Zero_After, D3_Stepsize
/// - Word 6: Iteration_Current, Iteration_Wrap, Iteration_Stepsize
/// - Word 7: Valid_BD, Lock_Rel_Value, Lock_Rel_ID, Lock_Acq_Enable,
///           Lock_Acq_Value, Lock_Acq_ID
#[derive(Debug, Clone)]
pub struct MemTileBdFieldLayout {
    // Word 0
    pub enable_packet: BitField,
    pub packet_type: BitField,
    pub packet_id: BitField,
    pub out_of_order_bd_id: BitField,
    pub buffer_length: BitField,
    // Word 1
    pub d0_zero_before: BitField,
    pub next_bd: BitField,
    pub use_next_bd: BitField,
    pub base_address: BitField,
    // Word 2
    pub tlast_suppress: BitField,
    pub d0_wrap: BitField,
    pub d0_stepsize: BitField,
    // Word 3
    pub d1_zero_before: BitField,
    pub d1_wrap: BitField,
    pub d1_stepsize: BitField,
    // Word 4
    pub enable_compression: BitField,
    pub d2_zero_before: BitField,
    pub d2_wrap: BitField,
    pub d2_stepsize: BitField,
    // Word 5
    pub d2_zero_after: BitField,
    pub d1_zero_after: BitField,
    pub d0_zero_after: BitField,
    pub d3_stepsize: BitField,
    // Word 6
    pub iteration_current: BitField,
    pub iteration_wrap: BitField,
    pub iteration_stepsize: BitField,
    // Word 7
    pub valid_bd: BitField,
    pub lock_rel_value: BitField,
    pub lock_rel_id: BitField,
    pub lock_acq_enable: BitField,
    pub lock_acq_value: BitField,
    pub lock_acq_id: BitField,
}

impl MemTileBdFieldLayout {
    /// Build from the register database for the memory_tile module.
    ///
    /// Resolves all BD fields from DMA_BD0_0 through DMA_BD0_7 registers.
    pub fn from_regdb(db: &RegisterDb) -> Result<Self, String> {
        let module = "memory_tile";
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        let get_field = |reg_name: &str, field_name: &str| -> Result<BitField, String> {
            let reg = m.register(reg_name)
                .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            // Word 0: DMA_BD0_0
            enable_packet: get_field("DMA_BD0_0", "Enable_Packet")?,
            packet_type: get_field("DMA_BD0_0", "Packet_Type")?,
            packet_id: get_field("DMA_BD0_0", "Packet_ID")?,
            out_of_order_bd_id: get_field("DMA_BD0_0", "Out_Of_Order_BD_ID")?,
            buffer_length: get_field("DMA_BD0_0", "Buffer_Length")?,
            // Word 1: DMA_BD0_1
            d0_zero_before: get_field("DMA_BD0_1", "D0_Zero_Before")?,
            next_bd: get_field("DMA_BD0_1", "Next_BD")?,
            use_next_bd: get_field("DMA_BD0_1", "Use_Next_BD")?,
            base_address: get_field("DMA_BD0_1", "Base_Address")?,
            // Word 2: DMA_BD0_2
            tlast_suppress: get_field("DMA_BD0_2", "TLAST_Suppress")?,
            d0_wrap: get_field("DMA_BD0_2", "D0_Wrap")?,
            d0_stepsize: get_field("DMA_BD0_2", "D0_Stepsize")?,
            // Word 3: DMA_BD0_3
            d1_zero_before: get_field("DMA_BD0_3", "D1_Zero_Before")?,
            d1_wrap: get_field("DMA_BD0_3", "D1_Wrap")?,
            d1_stepsize: get_field("DMA_BD0_3", "D1_Stepsize")?,
            // Word 4: DMA_BD0_4
            enable_compression: get_field("DMA_BD0_4", "Enable_Compression")?,
            d2_zero_before: get_field("DMA_BD0_4", "D2_Zero_Before")?,
            d2_wrap: get_field("DMA_BD0_4", "D2_Wrap")?,
            d2_stepsize: get_field("DMA_BD0_4", "D2_Stepsize")?,
            // Word 5: DMA_BD0_5
            d2_zero_after: get_field("DMA_BD0_5", "D2_Zero_After")?,
            d1_zero_after: get_field("DMA_BD0_5", "D1_Zero_After")?,
            d0_zero_after: get_field("DMA_BD0_5", "D0_Zero_After")?,
            d3_stepsize: get_field("DMA_BD0_5", "D3_Stepsize")?,
            // Word 6: DMA_BD0_6
            iteration_current: get_field("DMA_BD0_6", "Iteration_Current")?,
            iteration_wrap: get_field("DMA_BD0_6", "Iteration_Wrap")?,
            iteration_stepsize: get_field("DMA_BD0_6", "Iteration_Stepsize")?,
            // Word 7: DMA_BD0_7
            valid_bd: get_field("DMA_BD0_7", "Valid_BD")?,
            lock_rel_value: get_field("DMA_BD0_7", "Lock_Rel_Value")?,
            lock_rel_id: get_field("DMA_BD0_7", "Lock_Rel_ID")?,
            lock_acq_enable: get_field("DMA_BD0_7", "Lock_Acq_Enable")?,
            lock_acq_value: get_field("DMA_BD0_7", "Lock_Acq_Value")?,
            lock_acq_id: get_field("DMA_BD0_7", "Lock_Acq_ID")?,
        })
    }
}

/// Pre-resolved layout for Shim (NOC module) BD registers (8 words).
///
/// Shim BDs target DDR via NoC with AXI parameters not found in other tile types.
///
/// BD word layout (AM025 shim/dma/bd.txt):
/// - Word 0: Buffer_Length (full 32 bits for DDR transfers)
/// - Word 1: Base_Address_Low[31:2] (lower 30 bits of 46-bit word address)
/// - Word 2: Enable_Packet, OOO_ID, Packet_ID, Packet_Type, Base_Address_High[15:0]
/// - Word 3: Secure_Access, D0_Wrap, D0_Stepsize (20-bit)
/// - Word 4: Burst_Length, D1_Wrap, D1_Stepsize (20-bit)
/// - Word 5: SMID, AxCache, AxQoS, D2_Stepsize (20-bit)
/// - Word 6: Iteration_Current, Iteration_Wrap, Iteration_Stepsize (20-bit)
/// - Word 7: TLAST_Suppress, Next_BD, Use_Next_BD, Valid_BD, lock fields
#[derive(Debug, Clone)]
pub struct ShimBdFieldLayout {
    // Word 0
    pub buffer_length: BitField,
    // Word 1
    pub base_address_low: BitField,
    // Word 2
    pub enable_packet: BitField,
    pub out_of_order_bd_id: BitField,
    pub packet_id: BitField,
    pub packet_type: BitField,
    pub base_address_high: BitField,
    // Word 3
    pub secure_access: BitField,
    pub d0_wrap: BitField,
    pub d0_stepsize: BitField,
    // Word 4
    pub burst_length: BitField,
    pub d1_wrap: BitField,
    pub d1_stepsize: BitField,
    // Word 5
    pub smid: BitField,
    pub axcache: BitField,
    pub axqos: BitField,
    pub d2_stepsize: BitField,
    // Word 6
    pub iteration_current: BitField,
    pub iteration_wrap: BitField,
    pub iteration_stepsize: BitField,
    // Word 7
    pub tlast_suppress: BitField,
    pub next_bd: BitField,
    pub use_next_bd: BitField,
    pub valid_bd: BitField,
    pub lock_rel_value: BitField,
    pub lock_rel_id: BitField,
    pub lock_acq_enable: BitField,
    pub lock_acq_value: BitField,
    pub lock_acq_id: BitField,
}

impl ShimBdFieldLayout {
    /// Build from the register database for the shim module.
    ///
    /// Resolves all BD fields from DMA_BD0_0 through DMA_BD0_7 registers.
    pub fn from_regdb(db: &RegisterDb) -> Result<Self, String> {
        let module = "shim";
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        let get_field = |reg_name: &str, field_name: &str| -> Result<BitField, String> {
            let reg = m.register(reg_name)
                .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            // Word 0: DMA_BD0_0
            buffer_length: get_field("DMA_BD0_0", "Buffer_Length")?,
            // Word 1: DMA_BD0_1
            base_address_low: get_field("DMA_BD0_1", "Base_Address_Low")?,
            // Word 2: DMA_BD0_2
            enable_packet: get_field("DMA_BD0_2", "Enable_Packet")?,
            out_of_order_bd_id: get_field("DMA_BD0_2", "Out_Of_Order_BD_ID")?,
            packet_id: get_field("DMA_BD0_2", "Packet_ID")?,
            packet_type: get_field("DMA_BD0_2", "Packet_Type")?,
            base_address_high: get_field("DMA_BD0_2", "Base_Address_High")?,
            // Word 3: DMA_BD0_3
            secure_access: get_field("DMA_BD0_3", "Secure_Access")?,
            d0_wrap: get_field("DMA_BD0_3", "D0_Wrap")?,
            d0_stepsize: get_field("DMA_BD0_3", "D0_Stepsize")?,
            // Word 4: DMA_BD0_4
            burst_length: get_field("DMA_BD0_4", "Burst_Length")?,
            d1_wrap: get_field("DMA_BD0_4", "D1_Wrap")?,
            d1_stepsize: get_field("DMA_BD0_4", "D1_Stepsize")?,
            // Word 5: DMA_BD0_5
            smid: get_field("DMA_BD0_5", "SMID")?,
            axcache: get_field("DMA_BD0_5", "AxCache")?,
            axqos: get_field("DMA_BD0_5", "AxQoS")?,
            d2_stepsize: get_field("DMA_BD0_5", "D2_Stepsize")?,
            // Word 6: DMA_BD0_6
            iteration_current: get_field("DMA_BD0_6", "Iteration_Current")?,
            iteration_wrap: get_field("DMA_BD0_6", "Iteration_Wrap")?,
            iteration_stepsize: get_field("DMA_BD0_6", "Iteration_Stepsize")?,
            // Word 7: DMA_BD0_7
            tlast_suppress: get_field("DMA_BD0_7", "TLAST_Suppress")?,
            next_bd: get_field("DMA_BD0_7", "Next_BD")?,
            use_next_bd: get_field("DMA_BD0_7", "Use_Next_BD")?,
            valid_bd: get_field("DMA_BD0_7", "Valid_BD")?,
            lock_rel_value: get_field("DMA_BD0_7", "Lock_Rel_Value")?,
            lock_rel_id: get_field("DMA_BD0_7", "Lock_Rel_ID")?,
            lock_acq_enable: get_field("DMA_BD0_7", "Lock_Acq_Enable")?,
            lock_acq_value: get_field("DMA_BD0_7", "Lock_Acq_Value")?,
            lock_acq_id: get_field("DMA_BD0_7", "Lock_Acq_ID")?,
        })
    }
}

/// A single shim mux/demux field mapping a register bit field to a
/// switchbox port index.
#[derive(Debug, Clone)]
pub struct ShimMuxField {
    /// Register bit field (2-bit select: 0=PL, 1=DMA, 2=NoC)
    pub field: BitField,
    /// Switchbox port index this field controls.
    /// For mux (input): slave port index.
    /// For demux (output): master port index.
    pub port_index: usize,
}

/// Pre-resolved layout for shim mux and demux configuration registers.
///
/// The shim mux selects which source (PL/DMA/NoC) feeds each South slave
/// port on the switchbox. The shim demux selects which destination receives
/// each South master port's output.
#[derive(Debug, Clone)]
pub struct ShimMuxLayout {
    /// Mux_Config register offset (e.g., 0x1F000)
    pub mux_offset: u32,
    /// Mux_Config fields: SouthN -> slave[N+2]
    pub mux_fields: Vec<ShimMuxField>,
    /// Demux_Config register offset (e.g., 0x1F004)
    pub demux_offset: u32,
    /// Demux_Config fields: SouthN -> master[N+2]
    pub demux_fields: Vec<ShimMuxField>,
}

impl ShimMuxLayout {
    /// Build from the register database for the shim module.
    ///
    /// Scans Mux_Config and Demux_Config register fields for "SouthN"
    /// names. Port index = N + 2 (South0=port[2] in shim switchbox layout).
    pub fn from_regdb(db: &RegisterDb) -> Result<Self, String> {
        let shim = db.module("shim")
            .ok_or("Module 'shim' not found in register database")?;

        let parse_south_fields = |reg_name: &str| -> Result<(u32, Vec<ShimMuxField>), String> {
            let reg = shim.register(reg_name)
                .ok_or_else(|| format!("shim.{} not found", reg_name))?;
            let mut fields = Vec::new();
            for bf in &reg.fields {
                if let Some(n_str) = bf.name.strip_prefix("South") {
                    if let Ok(n) = n_str.parse::<usize>() {
                        // SouthN maps to switchbox port N+2
                        // (South0=slave[2] / master[2] in shim layout)
                        fields.push(ShimMuxField {
                            field: bf.clone(),
                            port_index: n + 2,
                        });
                    }
                }
            }
            // Sort by port_index for deterministic ordering
            fields.sort_by_key(|f| f.port_index);
            Ok((reg.offset, fields))
        };

        let (mux_offset, mux_fields) = parse_south_fields("Mux_Config")?;
        let (demux_offset, demux_fields) = parse_south_fields("Demux_Config")?;

        Ok(Self {
            mux_offset,
            mux_fields,
            demux_offset,
            demux_fields,
        })
    }
}

/// Pre-resolved stream switch register address ranges for one tile type.
///
/// Base addresses are derived from the first register in each group
/// (e.g., `Stream_Switch_Master_Config_AIE_Core0`). End addresses are
/// the offset of the last register plus one stride (exclusive end for
/// range checks). This replaces 6 hardcoded constants per module.
#[derive(Debug, Clone)]
pub struct StreamSwitchLayout {
    /// Stream switch master config base address
    pub master_base: u32,
    /// Stream switch master config end (exclusive)
    pub master_end: u32,
    /// Stream switch slave config base address
    pub slave_base: u32,
    /// Stream switch slave config end (exclusive)
    pub slave_end: u32,
    /// Stream switch slave slot config base address
    pub slave_slot_base: u32,
    /// Stream switch slave slot config end (exclusive)
    pub slave_slot_end: u32,
    /// Bytes between consecutive slave ports' slot registers (derived from AM025)
    pub slave_slot_port_stride: u32,
    /// Number of slots per slave port (derived from register count)
    pub slave_slot_count: usize,
}

impl StreamSwitchLayout {
    /// Build from the register database for a given module.
    ///
    /// Scans for the first and last `Stream_Switch_Master_Config_*`,
    /// `Stream_Switch_Slave_Config_*`, and `Stream_Switch_Slave_*_Slot*`
    /// registers to determine the exact address ranges.
    pub fn from_regdb(db: &RegisterDb, module: &str) -> Result<Self, String> {
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        // Find all registers matching each group prefix
        let mut master_offsets = Vec::new();
        let mut slave_config_offsets = Vec::new();
        let mut slave_slot_offsets = Vec::new();

        for reg in &m.registers {
            if reg.name.starts_with("Stream_Switch_Master_Config_") {
                master_offsets.push(reg.offset);
            } else if reg.name.starts_with("Stream_Switch_Slave_Config_") {
                slave_config_offsets.push(reg.offset);
            } else if reg.name.contains("_Slot") && reg.name.starts_with("Stream_Switch_Slave_") {
                slave_slot_offsets.push(reg.offset);
            }
        }

        if master_offsets.is_empty() {
            return Err(format!("{}: no Stream_Switch_Master_Config registers found", module));
        }
        if slave_config_offsets.is_empty() {
            return Err(format!("{}: no Stream_Switch_Slave_Config registers found", module));
        }
        if slave_slot_offsets.is_empty() {
            return Err(format!("{}: no Stream_Switch_Slave slot registers found", module));
        }

        master_offsets.sort();
        slave_config_offsets.sort();
        slave_slot_offsets.sort();

        // Base = first register offset, End = last register offset + 4 (exclusive)
        let master_base = master_offsets[0];
        let master_end = master_offsets[master_offsets.len() - 1] + 4;
        let slave_base = slave_config_offsets[0];
        let slave_end = slave_config_offsets[slave_config_offsets.len() - 1] + 4;
        let slave_slot_base = slave_slot_offsets[0];
        let slave_slot_end = slave_slot_offsets[slave_slot_offsets.len() - 1] + 4;

        // Derive slave slot port stride and slot count from the register offsets.
        // Slot registers follow: Port0_Slot0, Port0_Slot1, ..., Port1_Slot0, ...
        // Within a port, slots are 4 bytes apart. Between ports, the stride is larger.
        // Find the first gap > 4 to determine where port 0 ends and port 1 begins.
        let num_slave_ports = slave_config_offsets.len();
        let (slave_slot_port_stride, slave_slot_count) = if num_slave_ports > 1
            && slave_slot_offsets.len() > 1
        {
            // Total slot registers / number of slave ports = slots per port
            let slots_per_port = slave_slot_offsets.len() / num_slave_ports;
            // Port stride = offset of port 1's first slot - offset of port 0's first slot
            let port_stride = if slots_per_port > 0 && slave_slot_offsets.len() > slots_per_port {
                slave_slot_offsets[slots_per_port] - slave_slot_offsets[0]
            } else {
                // Fallback: derive from gaps
                let mut stride = 0u32;
                for pair in slave_slot_offsets.windows(2) {
                    if pair[1] - pair[0] > 4 {
                        stride = pair[1] - pair[0] + 4 * (slots_per_port as u32 - 1);
                        break;
                    }
                }
                stride
            };
            (port_stride, slots_per_port)
        } else {
            (0x10, 4) // Shouldn't happen, but safe defaults
        };

        Ok(Self {
            master_base,
            master_end,
            slave_base,
            slave_end,
            slave_slot_base,
            slave_slot_end,
            slave_slot_port_stride,
            slave_slot_count,
        })
    }
}

/// Pre-resolved register layouts for one device architecture.
///
/// This aggregates all the field layouts needed by the emulator's hot paths
/// (BD parsing, channel control, lock access) into a single struct that is
/// resolved once at startup.
///
/// Structural constants (base addresses, strides) are derived from register
/// offsets in the JSON database, eliminating the need for hand-coded constants
/// in `registers_spec.rs`.
#[derive(Debug, Clone)]
pub struct DeviceRegLayout {
    /// Full register database (for ad-hoc lookups)
    pub db: RegisterDb,
    /// Compute tile BD field layout
    pub memory_bd: BdFieldLayout,
    /// DMA channel field layout (compute tiles)
    pub memory_channel: ChannelFieldLayout,
    /// DMA channel status field layout (compute tiles)
    pub memory_status: StatusFieldLayout,
    /// MemTile BD field layout
    pub memtile_bd: MemTileBdFieldLayout,
    /// MemTile DMA channel field layout
    pub memtile_channel: ChannelFieldLayout,
    /// MemTile DMA channel status field layout
    pub memtile_status: StatusFieldLayout,

    // -- Compute tile lock layout (derived from Lock0_value, Lock1_value) --
    /// Lock register base offset (memory module)
    pub memory_lock_base: u32,
    /// Lock register stride (memory module)
    pub memory_lock_stride: u32,
    /// Lock overflow status register (memory module, write-to-clear)
    pub memory_locks_overflow: u32,
    /// Lock underflow status register (memory module, write-to-clear)
    pub memory_locks_underflow: u32,

    // -- Compute tile BD layout (derived from DMA_BD0_0, DMA_BD1_0) --
    /// DMA BD base address (memory module)
    pub memory_bd_base: u32,
    /// DMA BD stride in bytes (memory module)
    pub memory_bd_stride: u32,
    /// Words per BD (number of DMA_BD0_N registers found)
    pub memory_bd_words: usize,

    // -- Compute tile channel layout (derived from DMA_S2MM_0_Ctrl etc.) --
    /// DMA channel control base address (memory module)
    pub memory_channel_base: u32,
    /// DMA channel stride in bytes (memory module)
    pub memory_channel_stride: u32,
    /// DMA channel status base address (memory module)
    pub memory_status_base: u32,

    // -- MemTile lock layout (derived from Lock0_value, Lock1_value) --
    /// Lock register base offset (memory tile)
    pub memtile_lock_base: u32,
    /// Lock register stride (memory tile)
    pub memtile_lock_stride: u32,
    /// Lock overflow status register 0 (memory tile, locks 0-31)
    pub memtile_locks_overflow_0: u32,
    /// Lock overflow status register 1 (memory tile, locks 32-63)
    pub memtile_locks_overflow_1: u32,
    /// Lock underflow status register 0 (memory tile, locks 0-31)
    pub memtile_locks_underflow_0: u32,
    /// Lock underflow status register 1 (memory tile, locks 32-63)
    pub memtile_locks_underflow_1: u32,

    // -- MemTile BD layout (derived from DMA_BD0_0, DMA_BD1_0) --
    /// DMA BD base address (memory tile)
    pub memtile_bd_base: u32,
    /// DMA BD stride in bytes (memory tile)
    pub memtile_bd_stride: u32,
    /// Words per BD (number of DMA_BD0_N registers found)
    pub memtile_bd_words: usize,

    // -- MemTile channel layout (derived from DMA_S2MM_0_Ctrl etc.) --
    /// S2MM channel control base address (memory tile)
    pub memtile_channel_s2mm_base: u32,
    /// MM2S channel control base address (memory tile)
    pub memtile_channel_mm2s_base: u32,
    /// DMA channel stride in bytes (memory tile)
    pub memtile_channel_stride: u32,

    // -- Stream switch layouts --
    /// Compute tile stream switch address ranges
    pub memory_stream_switch: StreamSwitchLayout,
    /// MemTile stream switch address ranges
    pub memtile_stream_switch: StreamSwitchLayout,

    // -- Shim mux/demux layout --
    /// Shim mux/demux register field layout
    pub shim_mux: ShimMuxLayout,

    // -- Shim BD field layout --
    /// Shim (NOC module) BD field layout
    pub shim_bd: ShimBdFieldLayout,

    // -- Shim BD structural layout (derived from DMA_BD0_0, DMA_BD1_0) --
    /// DMA BD base address (shim)
    pub shim_bd_base: u32,
    /// DMA BD stride in bytes (shim)
    pub shim_bd_stride: u32,
    /// Words per BD (number of DMA_BD0_N registers found)
    pub shim_bd_words: usize,

    // -- Shim channel layout (derived from DMA_S2MM_0_Ctrl etc.) --
    /// DMA channel control base address (shim)
    pub shim_channel_base: u32,
    /// DMA channel stride in bytes (shim)
    pub shim_channel_stride: u32,
}

impl DeviceRegLayout {
    /// Build from a register database, resolving all field layouts.
    ///
    /// Derives structural constants (base addresses, strides) from register
    /// offsets. For example, the BD base is DMA_BD0_0's offset, and the BD
    /// stride is DMA_BD1_0 - DMA_BD0_0. This eliminates hand-coded constants.
    pub fn from_regdb(db: RegisterDb) -> Result<Self, String> {
        let memory_bd = BdFieldLayout::from_regdb(&db, "memory")?;
        let memory_channel = ChannelFieldLayout::from_regdb(&db, "memory")?;
        let memory_status = StatusFieldLayout::from_regdb(&db, "memory")?;
        let memtile_bd = MemTileBdFieldLayout::from_regdb(&db)?;
        let memtile_channel = ChannelFieldLayout::from_regdb(&db, "memory_tile")?;
        let memtile_status = StatusFieldLayout::from_regdb(&db, "memory_tile")?;

        // Helper: get a register offset from a module
        let reg_offset = |module: &str, reg: &str| -> Result<u32, String> {
            db.module(module)
                .ok_or_else(|| format!("Module '{}' not found", module))?
                .register(reg)
                .ok_or_else(|| format!("{}.{} not found", module, reg))
                .map(|r| r.offset)
        };

        // -- Compute tile locks --
        let memory_lock_base = reg_offset("memory", "Lock0_value")?;
        let memory_lock_stride = reg_offset("memory", "Lock1_value")? - memory_lock_base;
        let memory_locks_overflow = reg_offset("memory", "Locks_Overflow")?;
        let memory_locks_underflow = reg_offset("memory", "Locks_Underflow")?;

        // -- Compute tile BDs --
        let memory_bd_base = reg_offset("memory", "DMA_BD0_0")?;
        let memory_bd_stride = reg_offset("memory", "DMA_BD1_0")? - memory_bd_base;

        // Count BD words: DMA_BD0_0 through DMA_BD0_N (contiguous registers)
        let mem = db.module("memory").unwrap();
        let memory_bd_words = (0..16)
            .take_while(|i| mem.register(&format!("DMA_BD0_{}", i)).is_some())
            .count();

        // -- Compute tile channels --
        let memory_channel_base = reg_offset("memory", "DMA_S2MM_0_Ctrl")?;
        let memory_channel_stride = reg_offset("memory", "DMA_S2MM_1_Ctrl")? - memory_channel_base;
        let memory_status_base = reg_offset("memory", "DMA_S2MM_Status_0")?;

        // -- MemTile locks --
        let memtile_lock_base = reg_offset("memory_tile", "Lock0_value")?;
        let memtile_lock_stride = reg_offset("memory_tile", "Lock1_value")? - memtile_lock_base;
        let memtile_locks_overflow_0 = reg_offset("memory_tile", "Locks_Overflow_0")?;
        let memtile_locks_overflow_1 = reg_offset("memory_tile", "Locks_Overflow_1")?;
        let memtile_locks_underflow_0 = reg_offset("memory_tile", "Locks_Underflow_0")?;
        let memtile_locks_underflow_1 = reg_offset("memory_tile", "Locks_Underflow_1")?;

        // -- MemTile BDs --
        let memtile_bd_base = reg_offset("memory_tile", "DMA_BD0_0")?;
        let memtile_bd_stride = reg_offset("memory_tile", "DMA_BD1_0")? - memtile_bd_base;

        let mt = db.module("memory_tile").unwrap();
        let memtile_bd_words = (0..16)
            .take_while(|i| mt.register(&format!("DMA_BD0_{}", i)).is_some())
            .count();

        // -- MemTile channels --
        let memtile_channel_s2mm_base = reg_offset("memory_tile", "DMA_S2MM_0_Ctrl")?;
        let memtile_channel_mm2s_base = reg_offset("memory_tile", "DMA_MM2S_0_Ctrl")?;
        let memtile_channel_stride = reg_offset("memory_tile", "DMA_S2MM_1_Ctrl")? - memtile_channel_s2mm_base;

        // -- Stream switch --
        // Note: AM025 JSON places compute tile stream switch registers under
        // "core" module (not "memory"), even though their addresses (0x3F000+)
        // are beyond the core module's general range (0x30000-0x3EFFF).
        let memory_stream_switch = StreamSwitchLayout::from_regdb(&db, "core")?;
        let memtile_stream_switch = StreamSwitchLayout::from_regdb(&db, "memory_tile")?;

        // -- Shim mux/demux --
        let shim_mux = ShimMuxLayout::from_regdb(&db)?;

        // -- Shim BD fields --
        let shim_bd = ShimBdFieldLayout::from_regdb(&db)?;

        // -- Shim BDs --
        let shim_bd_base = reg_offset("shim", "DMA_BD0_0")?;
        let shim_bd_stride = reg_offset("shim", "DMA_BD1_0")? - shim_bd_base;

        let sh = db.module("shim").unwrap();
        let shim_bd_words = (0..16)
            .take_while(|i| sh.register(&format!("DMA_BD0_{}", i)).is_some())
            .count();

        // -- Shim channels --
        let shim_channel_base = reg_offset("shim", "DMA_S2MM_0_Ctrl")?;
        let shim_channel_stride = reg_offset("shim", "DMA_S2MM_1_Ctrl")? - shim_channel_base;

        Ok(Self {
            memory_bd,
            memory_channel,
            memory_status,
            memtile_bd,
            memtile_channel,
            memtile_status,
            memory_lock_base,
            memory_lock_stride,
            memory_locks_overflow,
            memory_locks_underflow,
            memory_bd_base,
            memory_bd_stride,
            memory_bd_words,
            memory_channel_base,
            memory_channel_stride,
            memory_status_base,
            memtile_lock_base,
            memtile_lock_stride,
            memtile_locks_overflow_0,
            memtile_locks_overflow_1,
            memtile_locks_underflow_0,
            memtile_locks_underflow_1,
            memtile_bd_base,
            memtile_bd_stride,
            memtile_bd_words,
            memtile_channel_s2mm_base,
            memtile_channel_mm2s_base,
            memtile_channel_stride,
            memory_stream_switch,
            memtile_stream_switch,
            shim_mux,
            shim_bd,
            shim_bd_base,
            shim_bd_stride,
            shim_bd_words,
            shim_channel_base,
            shim_channel_stride,
            db,
        })
    }

    /// Load from the mlir-aie install for a given device.
    pub fn load_for_device(device: &str) -> Result<Self, String> {
        let db = RegisterDb::load_for_device(device)?;
        Self::from_regdb(db)
    }
}

// ============================================================================
// Global accessor with lazy initialization (JSON required)
// ============================================================================

use std::sync::OnceLock;

static DEVICE_REG_LAYOUT: OnceLock<DeviceRegLayout> = OnceLock::new();

/// Get the global register layout, loading from JSON on first access.
///
/// # Panics
///
/// Panics if the register database JSON file cannot be loaded. This requires
/// mlir-aie to be installed and MLIR_AIE_PATH configured. The fallback path
/// was removed because maintaining two sources of truth (JSON + hand-coded
/// constants) is error-prone; the JSON is the single authoritative source.
pub fn device_reg_layout() -> &'static DeviceRegLayout {
    DEVICE_REG_LAYOUT.get_or_init(|| {
        DeviceRegLayout::load_for_device("aie2").unwrap_or_else(|e| {
            panic!(
                "Failed to load register database: {}.\n\
                 The register database JSON (aie_registers_aie2.json) is required.\n\
                 Ensure mlir-aie is installed and MLIR_AIE_PATH is set.\n\
                 See CLAUDE.md for environment setup instructions.",
                e
            )
        })
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to load the real database (skips if not available).
    fn load_test_db() -> Option<RegisterDb> {
        RegisterDb::load_for_device("aie2").ok()
    }

    #[test]
    fn test_parse_register_database() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found (set MLIR_AIE_PATH)");
            return;
        };

        // Verify version string is present
        assert!(!db.version.is_empty(), "Version should not be empty");

        // Verify all expected modules exist
        assert!(db.module("core").is_some(), "core module missing");
        assert!(db.module("memory").is_some(), "memory module missing");
        assert!(db.module("memory_tile").is_some(), "memory_tile module missing");
        assert!(db.module("shim").is_some(), "shim module missing");

        // Verify register counts are substantial
        let mem = db.module("memory").unwrap();
        assert!(mem.registers.len() > 100,
            "Expected >100 registers in memory module, got {}", mem.registers.len());
    }

    #[test]
    fn test_bitfield_extract() {
        // Buffer_Length: bits 13:0 (14 bits)
        let bf = BitField::from_range("test".to_string(), 0, 13);
        assert_eq!(bf.mask, 0x3FFF);
        assert_eq!(bf.shift, 0);
        assert_eq!(bf.extract(0x0000FFFF), 0x3FFF);
        assert_eq!(bf.extract(0xFFFF0000), 0x0000);

        // Base_Address: bits 27:14 (14 bits)
        let bf2 = BitField::from_range("test".to_string(), 14, 27);
        assert_eq!(bf2.mask, 0x3FFF);
        assert_eq!(bf2.shift, 14);
        assert_eq!(bf2.extract(0x0FFFC000), 0x3FFF);
        assert_eq!(bf2.extract(0x00003FFF), 0x0000);

        // Single bit: bit 31
        let bf3 = BitField::from_range("test".to_string(), 31, 31);
        assert_eq!(bf3.mask, 1);
        assert!(bf3.extract_bool(0x80000000));
        assert!(!bf3.extract_bool(0x7FFFFFFF));
    }

    #[test]
    fn test_bitfield_insert() {
        // Base_Address: bits 27:14 (14 bits)
        let bf = BitField::from_range("test".to_string(), 14, 27);

        // Insert 0x100 into an empty word
        let word = bf.insert(0, 0x100);
        assert_eq!(word, 0x100 << 14);
        assert_eq!(bf.extract(word), 0x100);

        // Insert into a word with other fields set (should not disturb them)
        let word = bf.insert(0xFFFF, 0x200);
        // Low 14 bits should be preserved (0x3FFF from original 0xFFFF)
        assert_eq!(word & 0x3FFF, 0x3FFF);
        assert_eq!(bf.extract(word), 0x200);

        // Mask truncates values exceeding field width
        let word = bf.insert(0, 0xFFFF); // 16 bits into 14-bit field
        assert_eq!(bf.extract(word), 0x3FFF);
    }

    #[test]
    fn test_bitfield_set_bit() {
        // Single-bit field at bit 19
        let bf = BitField::from_range("test".to_string(), 19, 19);
        let word = bf.set_bit(0);
        assert_eq!(word, 1 << 19);
        assert!(bf.extract_bool(word));

        // set_bit on already-set word is idempotent
        let word2 = bf.set_bit(word);
        assert_eq!(word, word2);

        // set_bit preserves other bits
        let word = bf.set_bit(0xDEAD_0000);
        assert_eq!(word, 0xDEAD_0000 | (1 << 19));
    }

    #[test]
    fn test_bitfield_insert_roundtrip() {
        // Verify insert -> extract roundtrip for various field positions
        let fields = [
            BitField::from_range("low".to_string(), 0, 3),     // bits 3:0
            BitField::from_range("mid".to_string(), 8, 15),    // bits 15:8
            BitField::from_range("high".to_string(), 24, 31),  // bits 31:24
            BitField::from_range("single".to_string(), 19, 19), // single bit
        ];

        for bf in &fields {
            let max_val = bf.mask;
            let word = bf.insert(0, max_val);
            assert_eq!(bf.extract(word), max_val,
                "Roundtrip failed for field '{}' [{},{}]", bf.name, bf.lsb, bf.msb);
        }
    }

    #[test]
    fn test_parse_hex_offset() {
        assert_eq!(parse_hex_offset("0x000001D000").unwrap(), 0x1D000);
        assert_eq!(parse_hex_offset("0x000001F000").unwrap(), 0x1F000);
        assert_eq!(parse_hex_offset("0x0000000000").unwrap(), 0x0);
        assert_eq!(parse_hex_offset("0x00000A0000").unwrap(), 0xA0000);
    }

    #[test]
    fn test_bd_field_layout_from_regdb() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let layout = BdFieldLayout::from_regdb(&db, "memory")
            .expect("Failed to build BdFieldLayout");

        // Verify Buffer_Length
        assert_eq!(layout.buffer_length.lsb, 0);
        assert_eq!(layout.buffer_length.msb, 13);
        assert_eq!(layout.buffer_length.mask, 0x3FFF);

        // Verify Base_Address
        assert_eq!(layout.base_address.lsb, 14);
        assert_eq!(layout.base_address.msb, 27);

        // Verify a Word 5 field
        assert_eq!(layout.valid_bd.lsb, 25);
        assert_eq!(layout.valid_bd.msb, 25);
        assert_eq!(layout.valid_bd.mask, 1);
    }

    #[test]
    fn test_channel_field_layout_from_regdb() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let layout = ChannelFieldLayout::from_regdb(&db, "memory")
            .expect("Failed to build ChannelFieldLayout");

        // Verify FoT_Mode: bits 17:16
        assert_eq!(layout.fot_mode.lsb, 16);
        assert_eq!(layout.fot_mode.msb, 17);
        assert_eq!(layout.fot_mode.mask, 0x3);

        // Verify Start_BD_ID: bits 3:0
        assert_eq!(layout.start_bd_id.lsb, 0);
        assert_eq!(layout.start_bd_id.msb, 3);
        assert_eq!(layout.start_bd_id.mask, 0xF);
    }

    #[test]
    fn test_device_reg_layout_from_regdb() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let layout = DeviceRegLayout::from_regdb(db)
            .expect("Failed to build DeviceRegLayout");

        // Verify lock layout (AM025)
        assert_eq!(layout.memory_lock_base, 0x1F000);
        assert_eq!(layout.memory_lock_stride, 0x10);
        assert_eq!(layout.memtile_lock_base, 0xC0000);
        assert_eq!(layout.memtile_lock_stride, 0x10);

        // Verify compute tile BD layout (AM025)
        assert_eq!(layout.memory_bd_base, 0x1D000, "Compute BD base");
        assert_eq!(layout.memory_bd_stride, 0x20, "Compute BD stride");
        assert_eq!(layout.memory_bd_words, 6, "Compute BD words");

        // Verify compute tile channel layout (AM025)
        assert_eq!(layout.memory_channel_base, 0x1DE00, "Compute channel base");
        assert_eq!(layout.memory_channel_stride, 0x08, "Compute channel stride");
        assert_eq!(layout.memory_status_base, 0x1DF00, "Compute status base");

        // Verify memtile BD layout (AM025)
        assert_eq!(layout.memtile_bd_base, 0xA0000, "MemTile BD base");
        assert_eq!(layout.memtile_bd_stride, 0x20, "MemTile BD stride");
        assert_eq!(layout.memtile_bd_words, 8, "MemTile BD words");

        // Verify memtile channel layout (AM025)
        assert_eq!(layout.memtile_channel_s2mm_base, 0xA0600, "MemTile S2MM base");
        assert_eq!(layout.memtile_channel_mm2s_base, 0xA0630, "MemTile MM2S base");
        assert_eq!(layout.memtile_channel_stride, 0x08, "MemTile channel stride");

        // Verify shim BD layout (AM025)
        assert_eq!(layout.shim_bd_base, 0x1D000, "Shim BD base");
        assert_eq!(layout.shim_bd_stride, 0x20, "Shim BD stride");
        assert_eq!(layout.shim_bd_words, 8, "Shim BD words");

        // Verify shim channel layout (AM025)
        assert_eq!(layout.shim_channel_base, 0x1D200, "Shim channel base");
        assert_eq!(layout.shim_channel_stride, 0x08, "Shim channel stride");

        // Verify stream switch slave slot layout (AM025)
        assert_eq!(layout.memory_stream_switch.slave_slot_port_stride, 0x10,
            "Compute slave slot port stride");
        assert_eq!(layout.memory_stream_switch.slave_slot_count, 4,
            "Compute slave slots per port");
        assert_eq!(layout.memtile_stream_switch.slave_slot_port_stride, 0x10,
            "MemTile slave slot port stride");
        assert_eq!(layout.memtile_stream_switch.slave_slot_count, 4,
            "MemTile slave slots per port");

        // Verify shim BD field layout was populated
        assert_eq!(layout.shim_bd.buffer_length.width, 32, "Shim buffer_length is 32-bit");
        assert_eq!(layout.shim_bd.base_address_low.lsb, 2, "Shim addr_low starts at bit 2");
    }

    #[test]
    fn validate_status_field_layout() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let layout = DeviceRegLayout::from_regdb(db)
            .expect("Failed to build DeviceRegLayout");

        // Compute tile status fields (DMA_S2MM_Status_0)
        let cs = &layout.memory_status;
        assert_eq!(cs.status.lsb, 0, "Status[1:0]");
        assert_eq!(cs.status.msb, 1);
        assert_eq!(cs.stalled_lock_acq.lsb, 2, "Stalled_Lock_Acq[2]");
        assert_eq!(cs.stalled_lock_rel.lsb, 3, "Stalled_Lock_Rel[3]");
        assert_eq!(cs.stalled_stream.lsb, 4, "Stalled_Stream[4]");
        assert_eq!(cs.stalled_tct.lsb, 5, "Stalled_TCT[5]");
        assert_eq!(cs.error_bd_unavailable.lsb, 10, "Error_BD_Unavailable[10]");
        assert_eq!(cs.error_bd_invalid.lsb, 11, "Error_BD_Invalid[11]");
        assert_eq!(cs.task_queue_overflow.lsb, 18, "Task_Queue_Overflow[18]");
        assert_eq!(cs.channel_running.lsb, 19, "Channel_Running[19]");
        assert_eq!(cs.task_queue_size.lsb, 20, "Task_Queue_Size[22:20]");
        assert_eq!(cs.task_queue_size.msb, 22);
        assert_eq!(cs.cur_bd.lsb, 24, "Cur_BD[27:24] compute");
        assert_eq!(cs.cur_bd.msb, 27);
        assert_eq!(cs.cur_bd.width, 4, "Compute Cur_BD is 4 bits");

        // MemTile status fields -- key difference is Cur_BD width
        let ms = &layout.memtile_status;
        assert_eq!(ms.cur_bd.lsb, 24, "Cur_BD[29:24] memtile");
        assert_eq!(ms.cur_bd.msb, 29);
        assert_eq!(ms.cur_bd.width, 6, "MemTile Cur_BD is 6 bits (48 BDs)");
        // Other fields are the same
        assert_eq!(ms.channel_running.lsb, 19);
        assert_eq!(ms.task_queue_size.lsb, 20);
    }

    #[test]
    fn validate_shim_mux_layout() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let layout = DeviceRegLayout::from_regdb(db)
            .expect("Failed to build DeviceRegLayout");

        let mux = &layout.shim_mux;

        // Mux_Config at 0x1F000
        assert_eq!(mux.mux_offset, 0x1F000, "Mux_Config offset");

        // Mux fields: South2[9:8]->slave[4], South3[11:10]->slave[5],
        //             South6[13:12]->slave[8], South7[15:14]->slave[9]
        assert_eq!(mux.mux_fields.len(), 4, "Mux has 4 South port fields");
        // Sorted by port_index
        assert_eq!(mux.mux_fields[0].port_index, 4);
        assert_eq!(mux.mux_fields[0].field.lsb, 8);
        assert_eq!(mux.mux_fields[1].port_index, 5);
        assert_eq!(mux.mux_fields[1].field.lsb, 10);
        assert_eq!(mux.mux_fields[2].port_index, 8);
        assert_eq!(mux.mux_fields[2].field.lsb, 12);
        assert_eq!(mux.mux_fields[3].port_index, 9);
        assert_eq!(mux.mux_fields[3].field.lsb, 14);

        // Demux_Config at 0x1F004
        assert_eq!(mux.demux_offset, 0x1F004, "Demux_Config offset");

        // Demux fields: South2[5:4]->master[4], South3[7:6]->master[5],
        //               South4[9:8]->master[6], South5[11:10]->master[7]
        assert_eq!(mux.demux_fields.len(), 4, "Demux has 4 South port fields");
        assert_eq!(mux.demux_fields[0].port_index, 4);
        assert_eq!(mux.demux_fields[0].field.lsb, 4);
        assert_eq!(mux.demux_fields[1].port_index, 5);
        assert_eq!(mux.demux_fields[1].field.lsb, 6);
        assert_eq!(mux.demux_fields[2].port_index, 6);
        assert_eq!(mux.demux_fields[2].field.lsb, 8);
        assert_eq!(mux.demux_fields[3].port_index, 7);
        assert_eq!(mux.demux_fields[3].field.lsb, 10);
    }

    // ====================================================================
    // Spot-check: verify JSON field layouts match AM025 expected values.
    // These tests use inline expected values (from AM025) rather than
    // references to registers_spec.rs, since the JSON is now the single
    // authoritative source for bit field definitions.
    // ====================================================================

    #[test]
    fn validate_memory_module_bd_fields() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let mem = db.module("memory").unwrap();

        // BD base address (AM025: DMA_BD0_0 @ 0x1D000)
        let bd0_0 = mem.register("DMA_BD0_0").unwrap();
        assert_eq!(bd0_0.offset, 0x1D000, "DMA_BD0_0 offset");

        // BD stride: 0x20 (AM025: DMA_BD1_0 @ 0x1D020)
        let bd1_0 = mem.register("DMA_BD1_0").unwrap();
        assert_eq!(bd1_0.offset - bd0_0.offset, 0x20, "BD stride");

        // Word 0: Buffer_Length[13:0], Base_Address[27:14]
        let buf_len = bd0_0.field("Buffer_Length").unwrap();
        assert_eq!((buf_len.lsb, buf_len.msb), (0, 13), "Buffer_Length bits");
        assert_eq!(buf_len.mask, 0x3FFF);

        let base_addr = bd0_0.field("Base_Address").unwrap();
        assert_eq!((base_addr.lsb, base_addr.msb), (14, 27), "Base_Address bits");
        assert_eq!(base_addr.mask, 0x3FFF);

        // Word 1: packet control fields
        let bd0_1 = mem.register("DMA_BD0_1").unwrap();
        assert_eq!(bd0_1.field("Enable_Compression").unwrap().lsb, 31);
        assert_eq!(bd0_1.field("Enable_Packet").unwrap().lsb, 30);
        assert_eq!(bd0_1.field("Out_Of_Order_BD_ID").unwrap().lsb, 24);
        assert_eq!(bd0_1.field("Packet_ID").unwrap().lsb, 19);
        assert_eq!(bd0_1.field("Packet_Type").unwrap().lsb, 16);

        // Word 5: lock and chaining fields
        let bd0_5 = mem.register("DMA_BD0_5").unwrap();
        assert_eq!(bd0_5.field("TLAST_Suppress").unwrap().lsb, 31);
        assert_eq!(bd0_5.field("Next_BD").unwrap().lsb, 27);
        assert_eq!(bd0_5.field("Use_Next_BD").unwrap().lsb, 26);
        assert_eq!(bd0_5.field("Valid_BD").unwrap().lsb, 25);
        assert_eq!(bd0_5.field("Lock_Rel_Value").unwrap().lsb, 18);
        assert_eq!(bd0_5.field("Lock_Acq_Enable").unwrap().lsb, 12);
        assert_eq!(bd0_5.field("Lock_Acq_Value").unwrap().lsb, 5);
        assert_eq!(bd0_5.field("Lock_Acq_ID").unwrap().mask, 0xF);
    }

    #[test]
    fn validate_lock_registers() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        // Memory module locks (AM025: Lock0_value @ 0x1F000, stride 0x10)
        let mem = db.module("memory").unwrap();
        let lock0 = mem.register("Lock0_value").unwrap();
        assert_eq!(lock0.offset, 0x1F000, "Lock0 offset");

        let lock1 = mem.register("Lock1_value").unwrap();
        assert_eq!(lock1.offset - lock0.offset, 0x10, "Lock stride");

        let lock_field = lock0.field("Lock_value").unwrap();
        assert_eq!((lock_field.lsb, lock_field.msb), (0, 5), "Lock_value bits");

        // Memory tile locks (AM025: Lock0_value @ 0xC0000, stride 0x10)
        let mt_mod = db.module("memory_tile").unwrap();
        let mt_lock0 = mt_mod.register("Lock0_value").unwrap();
        assert_eq!(mt_lock0.offset, 0xC0000, "MemTile Lock0 offset");

        let mt_lock1 = mt_mod.register("Lock1_value").unwrap();
        assert_eq!(mt_lock1.offset - mt_lock0.offset, 0x10, "MemTile Lock stride");
    }

    #[test]
    fn validate_dma_channel_registers() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let mem = db.module("memory").unwrap();

        // S2MM channel control (AM025: 0x1DE00)
        let s2mm_ctrl = mem.register("DMA_S2MM_0_Ctrl").unwrap();
        assert_eq!(s2mm_ctrl.offset, 0x1DE00, "S2MM_0_Ctrl offset");

        // Start queue at +4
        let start_q = mem.register("DMA_S2MM_0_Start_Queue").unwrap();
        assert_eq!(start_q.offset, 0x1DE04, "S2MM_0_Start_Queue offset");

        // Channel stride: 0x08
        let s2mm1_ctrl = mem.register("DMA_S2MM_1_Ctrl").unwrap();
        assert_eq!(s2mm1_ctrl.offset - s2mm_ctrl.offset, 0x08, "Channel stride");

        // Channel control fields (AM025)
        assert_eq!(s2mm_ctrl.field("FoT_Mode").unwrap().lsb, 16);
        assert_eq!(s2mm_ctrl.field("FoT_Mode").unwrap().mask, 0x3);
        assert_eq!(s2mm_ctrl.field("Controller_ID").unwrap().lsb, 8);
        assert_eq!(s2mm_ctrl.field("Controller_ID").unwrap().mask, 0xFF);
        assert_eq!(s2mm_ctrl.field("Decompression_Enable").unwrap().lsb, 4);
        assert_eq!(s2mm_ctrl.field("Enable_Out_of_Order").unwrap().lsb, 3);
        assert_eq!(s2mm_ctrl.field("Reset").unwrap().lsb, 1);

        // Start queue fields
        assert_eq!(start_q.field("Enable_Token_Issue").unwrap().lsb, 31);
        assert_eq!(start_q.field("Repeat_Count").unwrap().lsb, 16);
        assert_eq!(start_q.field("Repeat_Count").unwrap().mask, 0xFF);
        assert_eq!(start_q.field("Start_BD_ID").unwrap().mask, 0xF);
    }

    #[test]
    fn validate_memtile_bd_fields() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let mt_mod = db.module("memory_tile").unwrap();

        // BD base (AM025: DMA_BD0_0 @ 0xA0000)
        let bd0_0 = mt_mod.register("DMA_BD0_0").unwrap();
        assert_eq!(bd0_0.offset, 0xA0000, "MemTile DMA_BD0_0 offset");

        // BD stride: 0x20
        let bd1_0 = mt_mod.register("DMA_BD1_0").unwrap();
        assert_eq!(bd1_0.offset - bd0_0.offset, 0x20, "MemTile BD stride");

        // Word 0: Buffer_Length[16:0] (17 bits for MemTile)
        let buf_len = bd0_0.field("Buffer_Length").unwrap();
        assert_eq!(buf_len.mask, 0x1FFFF, "MemTile Buffer_Length mask");

        // Word 1: Base_Address[18:0], Use_Next_BD[19], Next_BD[25:20]
        let bd0_1 = mt_mod.register("DMA_BD0_1").unwrap();
        assert_eq!(bd0_1.field("Base_Address").unwrap().mask, 0x7FFFF);
        assert_eq!(bd0_1.field("Use_Next_BD").unwrap().lsb, 19);
        assert_eq!(bd0_1.field("Next_BD").unwrap().lsb, 20);
        assert_eq!(bd0_1.field("Next_BD").unwrap().mask, 0x3F);

        // Word 7: Lock and valid fields
        let bd0_7 = mt_mod.register("DMA_BD0_7").unwrap();
        assert_eq!(bd0_7.field("Valid_BD").unwrap().lsb, 31);
        assert_eq!(bd0_7.field("Lock_Rel_Value").unwrap().lsb, 24);
        assert_eq!(bd0_7.field("Lock_Rel_Value").unwrap().mask, 0x7F);
        assert_eq!(bd0_7.field("Lock_Rel_ID").unwrap().lsb, 16);
        assert_eq!(bd0_7.field("Lock_Rel_ID").unwrap().mask, 0xFF);
        assert_eq!(bd0_7.field("Lock_Acq_Enable").unwrap().lsb, 15);
        assert_eq!(bd0_7.field("Lock_Acq_Value").unwrap().lsb, 8);
        assert_eq!(bd0_7.field("Lock_Acq_Value").unwrap().mask, 0x7F);
        assert_eq!(bd0_7.field("Lock_Acq_ID").unwrap().mask, 0xFF);
    }

    #[test]
    fn validate_memtile_channel_registers() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let mt_mod = db.module("memory_tile").unwrap();

        // S2MM channel base (AM025: 0xA0600)
        let s2mm_ctrl = mt_mod.register("DMA_S2MM_0_Ctrl").unwrap();
        assert_eq!(s2mm_ctrl.offset, 0xA0600, "MemTile S2MM_0_Ctrl offset");

        // MM2S channel base (AM025: 0xA0630)
        let mm2s_ctrl = mt_mod.register("DMA_MM2S_0_Ctrl").unwrap();
        assert_eq!(mm2s_ctrl.offset, 0xA0630, "MemTile MM2S_0_Ctrl offset");

        // Channel stride: 0x08
        let s2mm1_ctrl = mt_mod.register("DMA_S2MM_1_Ctrl").unwrap();
        assert_eq!(s2mm1_ctrl.offset - s2mm_ctrl.offset, 0x08, "MemTile channel stride");
    }

    #[test]
    fn validate_shim_bd_fields() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let shim = db.module("shim").unwrap();

        // BD base (AM025: DMA_BD0_0 @ 0x1D000)
        let bd0_0 = shim.register("DMA_BD0_0").unwrap();
        assert_eq!(bd0_0.offset, 0x1D000, "Shim DMA_BD0_0 offset");

        // BD stride: 0x20
        let bd1_0 = shim.register("DMA_BD1_0").unwrap();
        assert_eq!(bd1_0.offset - bd0_0.offset, 0x20, "Shim BD stride");

        // Word 0: Buffer_Length[31:0] (full 32 bits for DDR)
        let buf_len = bd0_0.field("Buffer_Length").unwrap();
        assert_eq!((buf_len.lsb, buf_len.msb), (0, 31), "Shim Buffer_Length bits");

        // Word 1: Base_Address_Low[31:2]
        let bd0_1 = shim.register("DMA_BD0_1").unwrap();
        let addr_low = bd0_1.field("Base_Address_Low").unwrap();
        assert_eq!((addr_low.lsb, addr_low.msb), (2, 31), "Shim Base_Address_Low bits");

        // Word 2: Base_Address_High[15:0], packet fields
        let bd0_2 = shim.register("DMA_BD0_2").unwrap();
        assert_eq!(bd0_2.field("Base_Address_High").unwrap().msb, 15);
        assert_eq!(bd0_2.field("Enable_Packet").unwrap().lsb, 30);
        assert_eq!(bd0_2.field("Out_Of_Order_BD_ID").unwrap().lsb, 24);

        // Word 3: D0_Stepsize[19:0] (20-bit for DDR range)
        let bd0_3 = shim.register("DMA_BD0_3").unwrap();
        assert_eq!(bd0_3.field("D0_Stepsize").unwrap().msb, 19, "Shim D0_Stepsize 20-bit");
        assert_eq!(bd0_3.field("Secure_Access").unwrap().lsb, 30);

        // Word 4: Burst_Length[31:30]
        let bd0_4 = shim.register("DMA_BD0_4").unwrap();
        assert_eq!(bd0_4.field("Burst_Length").unwrap().lsb, 30);

        // Word 5: SMID[31:28], AxCache[27:24], AxQoS[23:20]
        let bd0_5 = shim.register("DMA_BD0_5").unwrap();
        assert_eq!(bd0_5.field("SMID").unwrap().lsb, 28);
        assert_eq!(bd0_5.field("AxCache").unwrap().lsb, 24);
        assert_eq!(bd0_5.field("AxQoS").unwrap().lsb, 20);

        // Word 7: locks and chaining (same layout as compute BD word 5)
        let bd0_7 = shim.register("DMA_BD0_7").unwrap();
        assert_eq!(bd0_7.field("Valid_BD").unwrap().lsb, 25);
        assert_eq!(bd0_7.field("Lock_Acq_ID").unwrap().mask, 0xF);

        // Channel base (AM025: 0x1D200)
        let s2mm_ctrl = shim.register("DMA_S2MM_0_Ctrl").unwrap();
        assert_eq!(s2mm_ctrl.offset, 0x1D200, "Shim S2MM_0_Ctrl offset");

        // Channel stride: 0x08
        let s2mm1_ctrl = shim.register("DMA_S2MM_1_Ctrl").unwrap();
        assert_eq!(s2mm1_ctrl.offset - s2mm_ctrl.offset, 0x08, "Shim channel stride");
    }

    #[test]
    fn validate_core_module_registers() {
        use crate::device::registers_spec::core_module as cm;

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let core = db.module("core").unwrap();

        // Cross-validate all hardcoded core_module constants against AM025 JSON.
        // These remain hardcoded for hot-path match arms, but this test catches
        // drift if the toolchain or JSON evolves.
        //
        // Note: JSON register names omit the "Core_" prefix used in our
        // constants for some registers. E.g. "Core_Enable_Events" in our code
        // corresponds to "Enable_Events" in the JSON.
        assert_eq!(core.register("Core_Control").unwrap().offset, cm::CORE_CONTROL);
        assert_eq!(core.register("Core_Status").unwrap().offset, cm::CORE_STATUS);
        assert_eq!(core.register("Enable_Events").unwrap().offset, cm::CORE_ENABLE_EVENTS);
        assert_eq!(core.register("Reset_Event").unwrap().offset, cm::CORE_RESET_EVENT);
        assert_eq!(core.register("Core_PC").unwrap().offset, cm::CORE_PC);
        assert_eq!(core.register("Core_SP").unwrap().offset, cm::CORE_SP);
        assert_eq!(core.register("Core_LR").unwrap().offset, cm::CORE_LR);
        assert_eq!(core.register("Debug_Control0").unwrap().offset, cm::CORE_DEBUG_CONTROL0);
        assert_eq!(core.register("Tile_Control").unwrap().offset, cm::TILE_CONTROL);
        assert_eq!(core.register("Memory_Control").unwrap().offset, cm::MEMORY_CONTROL);
    }

    /// Spot check: extract a known BD configuration from raw words using
    /// the JSON-loaded layout, verify field extraction correctness.
    #[test]
    fn spot_check_bd_extraction() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let layout = DeviceRegLayout::from_regdb(db)
            .expect("Failed to build DeviceRegLayout");

        // Construct a realistic BD word 0:
        // Buffer_Length = 1024 words (0x400), Base_Address = 0x100
        let word0: u32 = (0x100 << 14) | 0x400;
        assert_eq!(layout.memory_bd.buffer_length.extract(word0), 0x400);
        assert_eq!(layout.memory_bd.base_address.extract(word0), 0x100);

        // Construct a realistic BD word 5:
        // Valid=1, Use_Next=1, Next_BD=3, Lock_Acq_ID=5, Lock_Acq_Value=1
        let word5: u32 = (1 << 25) | (1 << 26) | (3 << 27) | (1 << 5) | 5;
        assert!(layout.memory_bd.valid_bd.extract_bool(word5));
        assert!(layout.memory_bd.use_next_bd.extract_bool(word5));
        assert_eq!(layout.memory_bd.next_bd.extract(word5), 3);
        assert_eq!(layout.memory_bd.lock_acq_id.extract(word5), 5);
    }

    // ====================================================================
    // Tier 3: Register metadata (reset values, access modes, widths)
    // ====================================================================

    #[test]
    fn test_parse_reset_value() {
        assert_eq!(parse_reset_value("0x00000000"), 0);
        assert_eq!(parse_reset_value("0x000006DB"), 0x6DB);
        assert_eq!(parse_reset_value("0xFFFFFFFF"), 0xFFFFFFFF);
        assert_eq!(parse_reset_value("0x00000002"), 2);
        // Defensive: garbage input returns 0
        assert_eq!(parse_reset_value("not_hex"), 0);
    }

    #[test]
    fn test_access_mode_from_json() {
        assert_eq!(AccessMode::from_json("rwNormal read/write"), AccessMode::ReadWrite);
        assert_eq!(AccessMode::from_json("roRead-only"), AccessMode::ReadOnly);
        assert_eq!(AccessMode::from_json("woWrite-only"), AccessMode::WriteOnly);
        assert_eq!(AccessMode::from_json("wtcReadable, write a 1 to clear"), AccessMode::WriteToClear);
        assert_eq!(AccessMode::from_json("mixedMixed types"), AccessMode::Mixed);
        // Unknown defaults to ReadWrite
        assert_eq!(AccessMode::from_json("unknown"), AccessMode::ReadWrite);
    }

    #[test]
    fn test_register_width_and_access_parsed() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let core = db.module("core").unwrap();

        // Most core registers are 32-bit, but some (e.g. Program_Memory) are wider
        let wide_regs: Vec<&str> = core.registers.iter()
            .filter(|r| r.width != 32)
            .map(|r| r.name.as_str())
            .collect();
        // Program_Memory is 128-bit (VLIW bundle interface)
        assert!(wide_regs.contains(&"Program_Memory"),
            "Program_Memory should be wider than 32 bits");
        let pm = core.register("Program_Memory").unwrap();
        assert_eq!(pm.width, 128, "Program_Memory should be 128-bit");

        // Core_Status is read-only (hardware reports core state)
        let status = core.register("Core_Status").unwrap();
        assert_eq!(status.access, AccessMode::ReadOnly,
            "Core_Status should be read-only");

        // Core_Control is mixed (some bits are w1tc, others rw)
        let ctrl = core.register("Core_Control").unwrap();
        assert!(
            ctrl.access == AccessMode::Mixed || ctrl.access == AccessMode::ReadWrite,
            "Core_Control should be mixed or rw, got {:?}", ctrl.access
        );
    }

    #[test]
    fn test_known_nonzero_reset_values() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let core = db.module("core").unwrap();

        // Core_LE (Loop End) has a non-zero reset value (0x000FFFFF per AM025)
        let core_le = core.register("Core_LE").unwrap();
        assert_ne!(core_le.reset_value, 0,
            "Core_LE should have non-zero reset value");

        // Core_Control has reset 0x00000002 (bit 1 = Reset set on power-on)
        let core_ctrl = core.register("Core_Control").unwrap();
        assert_eq!(core_ctrl.reset_value, 0x00000002,
            "Core_Control reset should be 0x02 (Reset bit set)");
    }

    #[test]
    fn test_non_zero_reset_values_iterator() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let core = db.module("core").unwrap();
        let non_zero: Vec<(u32, u32)> = core.non_zero_reset_values().collect();

        // There should be some non-zero reset values in the core module
        assert!(!non_zero.is_empty(),
            "Core module should have at least one non-zero reset value");

        // Core_Control @ 0x32000 should be in the list with reset=0x02
        assert!(non_zero.iter().any(|&(off, val)| off == 0x32000 && val == 0x02),
            "Core_Control (0x32000) with reset 0x02 should be in non-zero list");
    }

    #[test]
    fn test_access_mode_distribution() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        // Count access modes across all modules
        let mut rw_count = 0usize;
        let mut ro_count = 0usize;
        let mut wo_count = 0usize;
        let mut wtc_count = 0usize;
        let mut mixed_count = 0usize;

        for module in db.modules.values() {
            for reg in &module.registers {
                match reg.access {
                    AccessMode::ReadWrite => rw_count += 1,
                    AccessMode::ReadOnly => ro_count += 1,
                    AccessMode::WriteOnly => wo_count += 1,
                    AccessMode::WriteToClear => wtc_count += 1,
                    AccessMode::Mixed => mixed_count += 1,
                }
            }
        }

        // Based on earlier exploration: rw~1592, wo~82, ro~70, wtc~36, mixed~26
        assert!(rw_count > 1000, "Expected >1000 rw registers, got {}", rw_count);
        assert!(ro_count > 30, "Expected >30 ro registers, got {}", ro_count);
        assert!(wo_count > 30, "Expected >30 wo registers, got {}", wo_count);
        assert!(wtc_count > 10, "Expected >10 wtc registers, got {}", wtc_count);
        assert!(mixed_count > 10, "Expected >10 mixed registers, got {}", mixed_count);
    }

    #[test]
    fn validate_stream_switch_layout() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        // Core module (compute tile) stream switch
        // AM025 JSON classifies these under "core", not "memory"
        let core_ss = StreamSwitchLayout::from_regdb(&db, "core")
            .expect("core stream switch layout");
        // Base addresses must match exactly (first register in each group)
        assert_eq!(core_ss.master_base, 0x3F000,
            "core master_base should be 0x3F000");
        assert_eq!(core_ss.slave_base, 0x3F100,
            "core slave_base should be 0x3F100");
        assert_eq!(core_ss.slave_slot_base, 0x3F200,
            "core slave_slot_base should be 0x3F200");
        // End addresses are last_register + 4. The old hardcoded values were
        // padded round numbers (0x3F058, 0x3F180, 0x3F390). The JSON-derived
        // values are tighter: exact end of the defined register space. Verify
        // they are above base and within the same address block.
        assert!(core_ss.master_end > core_ss.master_base
            && core_ss.master_end <= 0x3F100,
            "core master_end {:#X} should be in (0x3F000, 0x3F100]",
            core_ss.master_end);
        assert!(core_ss.slave_end > core_ss.slave_base
            && core_ss.slave_end <= 0x3F200,
            "core slave_end {:#X} should be in (0x3F100, 0x3F200]",
            core_ss.slave_end);
        assert!(core_ss.slave_slot_end > core_ss.slave_slot_base
            && core_ss.slave_slot_end <= 0x3F400,
            "core slave_slot_end {:#X} should be in (0x3F200, 0x3F400]",
            core_ss.slave_slot_end);

        // Memory tile stream switch
        let mt_ss = StreamSwitchLayout::from_regdb(&db, "memory_tile")
            .expect("memtile stream switch layout");
        assert_eq!(mt_ss.master_base, 0xB0000,
            "memtile master_base should be 0xB0000");
        assert_eq!(mt_ss.slave_base, 0xB0100,
            "memtile slave_base should be 0xB0100");
        assert_eq!(mt_ss.slave_slot_base, 0xB0200,
            "memtile slave_slot_base should be 0xB0200");
        assert!(mt_ss.master_end > mt_ss.master_base
            && mt_ss.master_end <= 0xB0100,
            "memtile master_end {:#X} should be in (0xB0000, 0xB0100]",
            mt_ss.master_end);
        assert!(mt_ss.slave_end > mt_ss.slave_base
            && mt_ss.slave_end <= 0xB0200,
            "memtile slave_end {:#X} should be in (0xB0100, 0xB0200]",
            mt_ss.slave_end);
        assert!(mt_ss.slave_slot_end > mt_ss.slave_slot_base
            && mt_ss.slave_slot_end <= 0xB0400,
            "memtile slave_slot_end {:#X} should be in (0xB0200, 0xB0400]",
            mt_ss.slave_slot_end);
    }

    #[test]
    fn test_registers_with_access_filter() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        let core = db.module("core").unwrap();

        // Core_Status should be in the read-only set
        let ro_regs: Vec<&str> = core.registers_with_access(AccessMode::ReadOnly)
            .map(|r| r.name.as_str())
            .collect();
        assert!(ro_regs.contains(&"Core_Status"),
            "Core_Status should be in read-only registers");

        // DMA status registers are typically read-only
        let mem = db.module("memory").unwrap();
        let mem_ro: Vec<&str> = mem.registers_with_access(AccessMode::ReadOnly)
            .map(|r| r.name.as_str())
            .collect();
        assert!(mem_ro.iter().any(|n| n.contains("Status")),
            "Memory module should have read-only status registers");
    }
}
