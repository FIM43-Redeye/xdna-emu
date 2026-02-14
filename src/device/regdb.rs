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
    #[allow(dead_code)]
    width: Option<u32>,
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

    /// Build a BitField from LSB and MSB bit positions.
    fn from_range(name: String, lsb: u8, msb: u8) -> Self {
        let width = msb - lsb + 1;
        let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
        Self { name, lsb, msb, width, mask, shift: lsb }
    }
}

/// A register definition with its offset and bit fields.
#[derive(Debug, Clone)]
pub struct RegisterDef {
    pub name: String,
    /// Byte offset within the module's address space
    pub offset: u32,
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

                let idx = registers.len();
                register_index.insert(raw_reg.name.clone(), idx);
                registers.push(RegisterDef {
                    name: raw_reg.name,
                    offset,
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

/// Pre-resolved layout for MemTile BD registers (8 words).
///
/// MemTile BDs have a different field layout from compute tile BDs:
/// - Word 0: Buffer_Length (17 bits)
/// - Word 1: Base_Address (19 bits), Use_Next_BD, Next_BD
/// - Word 7: Lock fields, Valid_BD
#[derive(Debug, Clone)]
pub struct MemTileBdFieldLayout {
    // Word 0
    pub buffer_length: BitField,
    // Word 1
    pub base_address: BitField,
    pub use_next_bd: BitField,
    pub next_bd: BitField,
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
            buffer_length: get_field("DMA_BD0_0", "Buffer_Length")?,
            base_address: get_field("DMA_BD0_1", "Base_Address")?,
            use_next_bd: get_field("DMA_BD0_1", "Use_Next_BD")?,
            next_bd: get_field("DMA_BD0_1", "Next_BD")?,
            valid_bd: get_field("DMA_BD0_7", "Valid_BD")?,
            lock_rel_value: get_field("DMA_BD0_7", "Lock_Rel_Value")?,
            lock_rel_id: get_field("DMA_BD0_7", "Lock_Rel_ID")?,
            lock_acq_enable: get_field("DMA_BD0_7", "Lock_Acq_Enable")?,
            lock_acq_value: get_field("DMA_BD0_7", "Lock_Acq_Value")?,
            lock_acq_id: get_field("DMA_BD0_7", "Lock_Acq_ID")?,
        })
    }
}

/// Pre-resolved register layouts for one device architecture.
///
/// This aggregates all the field layouts needed by the emulator's hot paths
/// (BD parsing, channel control, lock access) into a single struct that is
/// resolved once at startup.
#[derive(Debug, Clone)]
pub struct DeviceRegLayout {
    /// Full register database (for ad-hoc lookups)
    pub db: RegisterDb,
    /// Compute tile BD field layout
    pub memory_bd: BdFieldLayout,
    /// DMA channel field layout (compute tiles)
    pub memory_channel: ChannelFieldLayout,
    /// MemTile BD field layout
    pub memtile_bd: MemTileBdFieldLayout,
    /// MemTile DMA channel field layout
    pub memtile_channel: ChannelFieldLayout,
    /// Lock register base offset (memory module)
    pub memory_lock_base: u32,
    /// Lock register stride (memory module)
    pub memory_lock_stride: u32,
    /// Lock register base offset (memory tile)
    pub memtile_lock_base: u32,
    /// Lock register stride (memory tile)
    pub memtile_lock_stride: u32,
}

impl DeviceRegLayout {
    /// Build from a register database, resolving all field layouts.
    pub fn from_regdb(db: RegisterDb) -> Result<Self, String> {
        let memory_bd = BdFieldLayout::from_regdb(&db, "memory")?;
        let memory_channel = ChannelFieldLayout::from_regdb(&db, "memory")?;
        let memtile_bd = MemTileBdFieldLayout::from_regdb(&db)?;
        let memtile_channel = ChannelFieldLayout::from_regdb(&db, "memory_tile")?;

        // Derive lock base/stride from register definitions
        let mem = db.module("memory")
            .ok_or("Module 'memory' not found")?;
        let lock0 = mem.register("Lock0_value")
            .ok_or("Lock0_value not found in memory module")?;
        let lock1 = mem.register("Lock1_value")
            .ok_or("Lock1_value not found in memory module")?;

        let mt = db.module("memory_tile")
            .ok_or("Module 'memory_tile' not found")?;
        let mt_lock0 = mt.register("Lock0_value")
            .ok_or("Lock0_value not found in memory_tile module")?;
        let mt_lock1 = mt.register("Lock1_value")
            .ok_or("Lock1_value not found in memory_tile module")?;

        Ok(Self {
            memory_bd,
            memory_channel,
            memtile_bd,
            memtile_channel,
            memory_lock_base: lock0.offset,
            memory_lock_stride: lock1.offset - lock0.offset,
            memtile_lock_base: mt_lock0.offset,
            memtile_lock_stride: mt_lock1.offset - mt_lock0.offset,
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
// Global accessor with lazy initialization and fallback
// ============================================================================

use std::sync::OnceLock;

static DEVICE_REG_LAYOUT: OnceLock<DeviceRegLayout> = OnceLock::new();

/// Get the global register layout, loading from JSON on first access.
///
/// Falls back to hand-coded constants from `registers_spec.rs` if the JSON
/// file is not available (e.g., mlir-aie not installed).
pub fn device_reg_layout() -> &'static DeviceRegLayout {
    DEVICE_REG_LAYOUT.get_or_init(|| {
        match DeviceRegLayout::load_for_device("aie2") {
            Ok(layout) => {
                log::info!("Loaded register database: version {}", layout.db.version);
                layout
            }
            Err(e) => {
                log::warn!("Failed to load register database: {}. Using fallback.", e);
                fallback_aie2_layout()
            }
        }
    })
}

/// Build a fallback layout from our hand-coded constants.
///
/// This ensures the emulator works even without the JSON file. The fallback
/// constructs BitField structs with the same mask/shift values that
/// `registers_spec.rs` defines.
fn fallback_aie2_layout() -> DeviceRegLayout {
    use super::registers_spec::{memory_module as mm, mem_tile_module as mt};

    // Helper to build a BitField from explicit LSB and MSB
    let bf = |name: &str, lsb: u8, msb: u8| -> BitField {
        BitField::from_range(name.to_string(), lsb, msb)
    };

    let memory_bd = BdFieldLayout {
        buffer_length: bf("Buffer_Length", 0, 13),
        base_address: bf("Base_Address", 14, 27),
        enable_compression: bf("Enable_Compression", 31, 31),
        enable_packet: bf("Enable_Packet", 30, 30),
        out_of_order_bd_id: bf("Out_Of_Order_BD_ID", 24, 29),
        packet_id: bf("Packet_ID", 19, 23),
        packet_type: bf("Packet_Type", 16, 18),
        d0_stepsize: bf("D0_Stepsize", 0, 12),
        d1_stepsize: bf("D1_Stepsize", 13, 25),
        d0_wrap: bf("D0_Wrap", 13, 20),
        d1_wrap: bf("D1_Wrap", 21, 28),
        d2_stepsize: bf("D2_Stepsize", 0, 12),
        iteration_current: bf("Iteration_Current", 19, 24),
        iteration_wrap: bf("Iteration_Wrap", 13, 18),
        iteration_stepsize: bf("Iteration_Stepsize", 0, 12),
        tlast_suppress: bf("TLAST_Suppress", 31, 31),
        next_bd: bf("Next_BD", 27, 30),
        use_next_bd: bf("Use_Next_BD", 26, 26),
        valid_bd: bf("Valid_BD", 25, 25),
        lock_rel_value: bf("Lock_Rel_Value", 18, 24),
        lock_rel_id: bf("Lock_Rel_ID", 13, 16),
        lock_acq_enable: bf("Lock_Acq_Enable", 12, 12),
        lock_acq_value: bf("Lock_Acq_Value", 5, 11),
        lock_acq_id: bf("Lock_Acq_ID", 0, 3),
    };

    let memory_channel = ChannelFieldLayout {
        fot_mode: bf("FoT_Mode", 16, 17),
        controller_id: bf("Controller_ID", 8, 15),
        decompression_enable: bf("Decompression_Enable", 4, 4),
        enable_out_of_order: bf("Enable_Out_of_Order", 3, 3),
        reset: bf("Reset", 1, 1),
        enable_token_issue: bf("Enable_Token_Issue", 31, 31),
        repeat_count: bf("Repeat_Count", 16, 23),
        start_bd_id: bf("Start_BD_ID", 0, 3),
    };

    let memtile_bd = MemTileBdFieldLayout {
        buffer_length: bf("Buffer_Length", 0, 16),
        base_address: bf("Base_Address", 0, 18),
        use_next_bd: bf("Use_Next_BD", 19, 19),
        next_bd: bf("Next_BD", 20, 25),
        valid_bd: bf("Valid_BD", 31, 31),
        lock_rel_value: bf("Lock_Rel_Value", 24, 30),
        lock_rel_id: bf("Lock_Rel_ID", 16, 23),
        lock_acq_enable: bf("Lock_Acq_Enable", 15, 15),
        lock_acq_value: bf("Lock_Acq_Value", 8, 14),
        lock_acq_id: bf("Lock_Acq_ID", 0, 7),
    };

    let memtile_channel = ChannelFieldLayout {
        fot_mode: bf("FoT_Mode", 16, 17),
        controller_id: bf("Controller_ID", 8, 15),
        decompression_enable: bf("Decompression_Enable", 4, 4),
        enable_out_of_order: bf("Enable_Out_of_Order", 3, 3),
        reset: bf("Reset", 1, 1),
        enable_token_issue: bf("Enable_Token_Issue", 31, 31),
        repeat_count: bf("Repeat_Count", 16, 23),
        start_bd_id: bf("Start_BD_ID", 0, 5),
    };

    DeviceRegLayout {
        db: RegisterDb {
            version: "fallback-hand-coded".to_string(),
            modules: HashMap::new(),
        },
        memory_bd,
        memory_channel,
        memtile_bd,
        memtile_channel,
        memory_lock_base: mm::LOCK_BASE,
        memory_lock_stride: mm::LOCK_STRIDE,
        memtile_lock_base: mt::LOCK_BASE,
        memtile_lock_stride: mt::LOCK_STRIDE,
    }
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

        // Verify lock offsets match hand-coded constants
        assert_eq!(layout.memory_lock_base, 0x1F000);
        assert_eq!(layout.memory_lock_stride, 0x10);
        assert_eq!(layout.memtile_lock_base, 0xC0000);
        assert_eq!(layout.memtile_lock_stride, 0x10);
    }

    #[test]
    fn test_fallback_layout() {
        // Fallback should always succeed (no external dependencies)
        let layout = fallback_aie2_layout();

        assert_eq!(layout.memory_lock_base, 0x1F000);
        assert_eq!(layout.memory_lock_stride, 0x10);

        // Verify BD field extraction matches hand-coded constants
        assert_eq!(layout.memory_bd.buffer_length.mask, 0x3FFF);
        assert_eq!(layout.memory_bd.base_address.shift, 14);
        assert_eq!(layout.memory_bd.base_address.mask, 0x3FFF);
    }

    // ====================================================================
    // Cross-validation: JSON database vs hand-coded registers_spec.rs
    // ====================================================================

    #[test]
    fn validate_memory_module_bd_fields() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        use super::super::registers_spec::memory_module as mm;

        let mem = db.module("memory").unwrap();

        // BD base address
        let bd0_0 = mem.register("DMA_BD0_0").unwrap();
        assert_eq!(bd0_0.offset, mm::DMA_BD_BASE,
            "DMA_BD0_0 offset mismatch: JSON=0x{:X} spec=0x{:X}",
            bd0_0.offset, mm::DMA_BD_BASE);

        // BD stride (BD1 - BD0)
        let bd1_0 = mem.register("DMA_BD1_0").unwrap();
        assert_eq!(bd1_0.offset - bd0_0.offset, mm::DMA_BD_STRIDE,
            "BD stride mismatch");

        // Word 0: Buffer_Length
        let buf_len = bd0_0.field("Buffer_Length").unwrap();
        assert_eq!(buf_len.lsb, 0, "Buffer_Length LSB");
        assert_eq!(buf_len.msb, 13, "Buffer_Length MSB");
        assert_eq!(buf_len.mask, mm::bd::WORD0_BUFFER_LEN_MASK,
            "Buffer_Length mask: JSON=0x{:X} spec=0x{:X}",
            buf_len.mask, mm::bd::WORD0_BUFFER_LEN_MASK);

        // Word 0: Base_Address
        let base_addr = bd0_0.field("Base_Address").unwrap();
        assert_eq!(base_addr.lsb as u32, mm::bd::WORD0_BASE_ADDR_SHIFT,
            "Base_Address shift");
        assert_eq!(base_addr.mask, mm::bd::WORD0_BASE_ADDR_MASK,
            "Base_Address mask");

        // Word 1: DMA_BD0_1
        let bd0_1 = mem.register("DMA_BD0_1").unwrap();
        assert_eq!(bd0_1.offset, mm::DMA_BD_BASE + 4, "BD word 1 offset");

        let enable_comp = bd0_1.field("Enable_Compression").unwrap();
        assert_eq!(enable_comp.lsb as u32, mm::bd::WORD1_ENABLE_COMPRESSION_BIT,
            "Enable_Compression bit");

        let enable_pkt = bd0_1.field("Enable_Packet").unwrap();
        assert_eq!(enable_pkt.lsb as u32, mm::bd::WORD1_ENABLE_PACKET_BIT,
            "Enable_Packet bit");

        let ooo_id = bd0_1.field("Out_Of_Order_BD_ID").unwrap();
        assert_eq!(ooo_id.lsb as u32, mm::bd::WORD1_OOO_BD_ID_SHIFT,
            "OOO_BD_ID shift");
        assert_eq!(ooo_id.mask, mm::bd::WORD1_OOO_BD_ID_MASK,
            "OOO_BD_ID mask");

        let pkt_id = bd0_1.field("Packet_ID").unwrap();
        assert_eq!(pkt_id.lsb as u32, mm::bd::WORD1_PACKET_ID_SHIFT,
            "Packet_ID shift");
        assert_eq!(pkt_id.mask, mm::bd::WORD1_PACKET_ID_MASK,
            "Packet_ID mask");

        let pkt_type = bd0_1.field("Packet_Type").unwrap();
        assert_eq!(pkt_type.lsb as u32, mm::bd::WORD1_PACKET_TYPE_SHIFT,
            "Packet_Type shift");
        assert_eq!(pkt_type.mask, mm::bd::WORD1_PACKET_TYPE_MASK,
            "Packet_Type mask");

        // Word 2: DMA_BD0_2
        let bd0_2 = mem.register("DMA_BD0_2").unwrap();
        let d0_step = bd0_2.field("D0_Stepsize").unwrap();
        assert_eq!(d0_step.mask, mm::bd::WORD2_D0_STEPSIZE_MASK,
            "D0_Stepsize mask");

        let d1_step = bd0_2.field("D1_Stepsize").unwrap();
        assert_eq!(d1_step.lsb as u32, mm::bd::WORD2_D1_STEPSIZE_SHIFT,
            "D1_Stepsize shift");
        assert_eq!(d1_step.mask, mm::bd::WORD2_D1_STEPSIZE_MASK,
            "D1_Stepsize mask");

        // Word 3: DMA_BD0_3
        let bd0_3 = mem.register("DMA_BD0_3").unwrap();
        let d0_wrap = bd0_3.field("D0_Wrap").unwrap();
        assert_eq!(d0_wrap.lsb as u32, mm::bd::WORD3_D0_WRAP_SHIFT,
            "D0_Wrap shift");
        assert_eq!(d0_wrap.mask, mm::bd::WORD3_D0_WRAP_MASK,
            "D0_Wrap mask");

        let d1_wrap = bd0_3.field("D1_Wrap").unwrap();
        assert_eq!(d1_wrap.lsb as u32, mm::bd::WORD3_D1_WRAP_SHIFT,
            "D1_Wrap shift");
        assert_eq!(d1_wrap.mask, mm::bd::WORD3_D1_WRAP_MASK,
            "D1_Wrap mask");

        let d2_step = bd0_3.field("D2_Stepsize").unwrap();
        assert_eq!(d2_step.mask, mm::bd::WORD3_D2_STEPSIZE_MASK,
            "D2_Stepsize mask");

        // Word 4: DMA_BD0_4
        let bd0_4 = mem.register("DMA_BD0_4").unwrap();
        let iter_cur = bd0_4.field("Iteration_Current").unwrap();
        assert_eq!(iter_cur.lsb as u32, mm::bd::WORD4_ITERATION_CURRENT_SHIFT,
            "Iteration_Current shift");
        assert_eq!(iter_cur.mask, mm::bd::WORD4_ITERATION_CURRENT_MASK,
            "Iteration_Current mask");

        let iter_wrap = bd0_4.field("Iteration_Wrap").unwrap();
        assert_eq!(iter_wrap.lsb as u32, mm::bd::WORD4_ITERATION_WRAP_SHIFT,
            "Iteration_Wrap shift");
        assert_eq!(iter_wrap.mask, mm::bd::WORD4_ITERATION_WRAP_MASK,
            "Iteration_Wrap mask");

        let iter_step = bd0_4.field("Iteration_Stepsize").unwrap();
        assert_eq!(iter_step.mask, mm::bd::WORD4_ITERATION_STEPSIZE_MASK,
            "Iteration_Stepsize mask");

        // Word 5: DMA_BD0_5
        let bd0_5 = mem.register("DMA_BD0_5").unwrap();
        let tlast = bd0_5.field("TLAST_Suppress").unwrap();
        assert_eq!(tlast.lsb as u32, mm::bd::WORD5_TLAST_SUPPRESS_BIT,
            "TLAST_Suppress bit");

        let next_bd = bd0_5.field("Next_BD").unwrap();
        assert_eq!(next_bd.lsb as u32, mm::bd::WORD5_NEXT_BD_SHIFT,
            "Next_BD shift");
        assert_eq!(next_bd.mask, mm::bd::WORD5_NEXT_BD_MASK,
            "Next_BD mask");

        let use_next = bd0_5.field("Use_Next_BD").unwrap();
        assert_eq!(use_next.lsb as u32, mm::bd::WORD5_USE_NEXT_BD_BIT,
            "Use_Next_BD bit");

        let valid = bd0_5.field("Valid_BD").unwrap();
        assert_eq!(valid.lsb as u32, mm::bd::WORD5_VALID_BD_BIT,
            "Valid_BD bit");

        let acq_id = bd0_5.field("Lock_Acq_ID").unwrap();
        assert_eq!(acq_id.mask, mm::bd::WORD5_LOCK_ACQ_ID_MASK,
            "Lock_Acq_ID mask");

        let acq_val = bd0_5.field("Lock_Acq_Value").unwrap();
        assert_eq!(acq_val.lsb as u32, mm::bd::WORD5_LOCK_ACQ_VALUE_SHIFT,
            "Lock_Acq_Value shift");
        assert_eq!(acq_val.mask, mm::bd::WORD5_LOCK_ACQ_VALUE_MASK,
            "Lock_Acq_Value mask");

        let acq_en = bd0_5.field("Lock_Acq_Enable").unwrap();
        assert_eq!(acq_en.lsb as u32, mm::bd::WORD5_LOCK_ACQ_ENABLE_BIT,
            "Lock_Acq_Enable bit");

        let rel_id = bd0_5.field("Lock_Rel_ID").unwrap();
        assert_eq!(rel_id.lsb as u32, mm::bd::WORD5_LOCK_REL_ID_SHIFT,
            "Lock_Rel_ID shift");
        assert_eq!(rel_id.mask, mm::bd::WORD5_LOCK_REL_ID_MASK,
            "Lock_Rel_ID mask");

        let rel_val = bd0_5.field("Lock_Rel_Value").unwrap();
        assert_eq!(rel_val.lsb as u32, mm::bd::WORD5_LOCK_REL_VALUE_SHIFT,
            "Lock_Rel_Value shift");
        assert_eq!(rel_val.mask, mm::bd::WORD5_LOCK_REL_VALUE_MASK,
            "Lock_Rel_Value mask");
    }

    #[test]
    fn validate_lock_registers() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        use super::super::registers_spec::{memory_module as mm, mem_tile_module as mt};

        // Memory module locks
        let mem = db.module("memory").unwrap();
        let lock0 = mem.register("Lock0_value").unwrap();
        assert_eq!(lock0.offset, mm::LOCK_BASE,
            "Lock0 offset: JSON=0x{:X} spec=0x{:X}", lock0.offset, mm::LOCK_BASE);

        let lock1 = mem.register("Lock1_value").unwrap();
        assert_eq!(lock1.offset - lock0.offset, mm::LOCK_STRIDE,
            "Lock stride: JSON=0x{:X} spec=0x{:X}",
            lock1.offset - lock0.offset, mm::LOCK_STRIDE);

        let lock_field = lock0.field("Lock_value").unwrap();
        assert_eq!(lock_field.lsb, 0, "Lock_value LSB");
        assert_eq!(lock_field.msb, 5, "Lock_value MSB");

        // Memory tile locks
        let mt_mod = db.module("memory_tile").unwrap();
        let mt_lock0 = mt_mod.register("Lock0_value").unwrap();
        assert_eq!(mt_lock0.offset, mt::LOCK_BASE,
            "MemTile Lock0 offset: JSON=0x{:X} spec=0x{:X}", mt_lock0.offset, mt::LOCK_BASE);

        let mt_lock1 = mt_mod.register("Lock1_value").unwrap();
        assert_eq!(mt_lock1.offset - mt_lock0.offset, mt::LOCK_STRIDE,
            "MemTile Lock stride");
    }

    #[test]
    fn validate_dma_channel_registers() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        use super::super::registers_spec::memory_module as mm;
        use super::super::registers_spec::memory_module::channel as mm_ch;

        let mem = db.module("memory").unwrap();

        // S2MM channel control
        let s2mm_ctrl = mem.register("DMA_S2MM_0_Ctrl").unwrap();
        assert_eq!(s2mm_ctrl.offset, mm::DMA_CHANNEL_BASE,
            "S2MM_0_Ctrl offset: JSON=0x{:X} spec=0x{:X}",
            s2mm_ctrl.offset, mm::DMA_CHANNEL_BASE);

        // Start queue
        let start_q = mem.register("DMA_S2MM_0_Start_Queue").unwrap();
        assert_eq!(start_q.offset, mm::DMA_CHANNEL_BASE + 4,
            "S2MM_0_Start_Queue offset");

        // Channel stride: S2MM_1 - S2MM_0
        let s2mm1_ctrl = mem.register("DMA_S2MM_1_Ctrl").unwrap();
        assert_eq!(s2mm1_ctrl.offset - s2mm_ctrl.offset, mm::DMA_CHANNEL_STRIDE,
            "Channel stride: JSON=0x{:X} spec=0x{:X}",
            s2mm1_ctrl.offset - s2mm_ctrl.offset, mm::DMA_CHANNEL_STRIDE);

        // Channel field validation
        let fot = s2mm_ctrl.field("FoT_Mode").unwrap();
        assert_eq!(fot.lsb as u32, mm_ch::CTRL_FOT_MODE_SHIFT, "FoT_Mode shift");
        assert_eq!(fot.mask, mm_ch::CTRL_FOT_MODE_MASK, "FoT_Mode mask");

        let ctrl_id = s2mm_ctrl.field("Controller_ID").unwrap();
        assert_eq!(ctrl_id.lsb as u32, mm_ch::CTRL_CONTROLLER_ID_SHIFT, "Controller_ID shift");
        assert_eq!(ctrl_id.mask, mm_ch::CTRL_CONTROLLER_ID_MASK, "Controller_ID mask");

        let decomp = s2mm_ctrl.field("Decompression_Enable").unwrap();
        assert_eq!(decomp.lsb as u32, mm_ch::CTRL_COMPRESSION_ENABLE_BIT, "Decompression_Enable bit");

        let ooo = s2mm_ctrl.field("Enable_Out_of_Order").unwrap();
        assert_eq!(ooo.lsb as u32, mm_ch::CTRL_ENABLE_OUT_OF_ORDER_BIT, "Enable_Out_of_Order bit");

        let reset = s2mm_ctrl.field("Reset").unwrap();
        assert_eq!(reset.lsb as u32, mm_ch::CTRL_RESET_BIT, "Reset bit");

        // Start queue fields
        let token = start_q.field("Enable_Token_Issue").unwrap();
        assert_eq!(token.lsb as u32, mm_ch::START_QUEUE_ENABLE_TOKEN_ISSUE_BIT, "Enable_Token_Issue bit");

        let repeat = start_q.field("Repeat_Count").unwrap();
        assert_eq!(repeat.lsb as u32, mm_ch::START_QUEUE_REPEAT_COUNT_SHIFT, "Repeat_Count shift");
        assert_eq!(repeat.mask, mm_ch::START_QUEUE_REPEAT_COUNT_MASK, "Repeat_Count mask");

        let bd_id = start_q.field("Start_BD_ID").unwrap();
        assert_eq!(bd_id.mask, mm_ch::START_QUEUE_BD_ID_MASK, "Start_BD_ID mask");
    }

    #[test]
    fn validate_memtile_bd_fields() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        use super::super::registers_spec::mem_tile_module as mt;

        let mt_mod = db.module("memory_tile").unwrap();

        // BD base address
        let bd0_0 = mt_mod.register("DMA_BD0_0").unwrap();
        assert_eq!(bd0_0.offset, mt::DMA_BD_BASE,
            "MemTile DMA_BD0_0 offset: JSON=0x{:X} spec=0x{:X}",
            bd0_0.offset, mt::DMA_BD_BASE);

        // BD stride
        let bd1_0 = mt_mod.register("DMA_BD1_0").unwrap();
        assert_eq!(bd1_0.offset - bd0_0.offset, mt::DMA_BD_STRIDE,
            "MemTile BD stride");

        // Word 0: Buffer_Length (17 bits for MemTile)
        let buf_len = bd0_0.field("Buffer_Length").unwrap();
        assert_eq!(buf_len.mask, mt::bd::WORD0_BUFFER_LEN_MASK,
            "MemTile Buffer_Length mask: JSON=0x{:X} spec=0x{:X}",
            buf_len.mask, mt::bd::WORD0_BUFFER_LEN_MASK);

        // Word 1: Base_Address, Use_Next_BD, Next_BD
        let bd0_1 = mt_mod.register("DMA_BD0_1").unwrap();
        let base = bd0_1.field("Base_Address").unwrap();
        assert_eq!(base.mask, mt::bd::WORD1_BASE_ADDR_MASK,
            "MemTile Base_Address mask");

        let use_next = bd0_1.field("Use_Next_BD").unwrap();
        assert_eq!(use_next.lsb as u32, mt::bd::WORD1_USE_NEXT_BD_BIT,
            "MemTile Use_Next_BD bit");

        let next = bd0_1.field("Next_BD").unwrap();
        assert_eq!(next.lsb as u32, mt::bd::WORD1_NEXT_BD_SHIFT,
            "MemTile Next_BD shift");
        assert_eq!(next.mask, mt::bd::WORD1_NEXT_BD_MASK,
            "MemTile Next_BD mask");

        // Word 7: Lock and valid fields
        let bd0_7 = mt_mod.register("DMA_BD0_7").unwrap();
        let valid = bd0_7.field("Valid_BD").unwrap();
        assert_eq!(valid.lsb as u32, mt::bd::WORD7_VALID_BD_BIT,
            "MemTile Valid_BD bit");

        let acq_val = bd0_7.field("Lock_Acq_Value").unwrap();
        assert_eq!(acq_val.lsb as u32, mt::bd::WORD7_LOCK_ACQ_VALUE_SHIFT,
            "MemTile Lock_Acq_Value shift");
        assert_eq!(acq_val.mask, mt::bd::WORD7_LOCK_ACQ_VALUE_MASK,
            "MemTile Lock_Acq_Value mask");

        let acq_en = bd0_7.field("Lock_Acq_Enable").unwrap();
        assert_eq!(acq_en.lsb as u32, mt::bd::WORD7_LOCK_ACQ_ENABLE_BIT,
            "MemTile Lock_Acq_Enable bit");

        let rel_val = bd0_7.field("Lock_Rel_Value").unwrap();
        assert_eq!(rel_val.lsb as u32, mt::bd::WORD7_LOCK_REL_VALUE_SHIFT,
            "MemTile Lock_Rel_Value shift");
        assert_eq!(rel_val.mask, mt::bd::WORD7_LOCK_REL_VALUE_MASK,
            "MemTile Lock_Rel_Value mask");

        let rel_id = bd0_7.field("Lock_Rel_ID").unwrap();
        assert_eq!(rel_id.lsb as u32, mt::bd::WORD7_LOCK_REL_ID_SHIFT,
            "MemTile Lock_Rel_ID shift");
        assert_eq!(rel_id.mask, mt::bd::WORD7_LOCK_REL_ID_MASK,
            "MemTile Lock_Rel_ID mask");
    }

    #[test]
    fn validate_memtile_channel_registers() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        use super::super::registers_spec::mem_tile_module as mt;

        let mt_mod = db.module("memory_tile").unwrap();

        // S2MM channel base
        let s2mm_ctrl = mt_mod.register("DMA_S2MM_0_Ctrl").unwrap();
        assert_eq!(s2mm_ctrl.offset, mt::DMA_CHANNEL_S2MM_BASE,
            "MemTile S2MM_0_Ctrl offset: JSON=0x{:X} spec=0x{:X}",
            s2mm_ctrl.offset, mt::DMA_CHANNEL_S2MM_BASE);

        // MM2S channel base
        let mm2s_ctrl = mt_mod.register("DMA_MM2S_0_Ctrl").unwrap();
        assert_eq!(mm2s_ctrl.offset, mt::DMA_CHANNEL_MM2S_BASE,
            "MemTile MM2S_0_Ctrl offset: JSON=0x{:X} spec=0x{:X}",
            mm2s_ctrl.offset, mt::DMA_CHANNEL_MM2S_BASE);

        // Channel stride
        let s2mm1_ctrl = mt_mod.register("DMA_S2MM_1_Ctrl").unwrap();
        assert_eq!(s2mm1_ctrl.offset - s2mm_ctrl.offset, mt::DMA_CHANNEL_STRIDE,
            "MemTile channel stride");
    }

    #[test]
    fn validate_core_module_registers() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database JSON not found");
            return;
        };

        use super::super::registers_spec::core_module as cm;

        let core = db.module("core").unwrap();

        let ctrl = core.register("Core_Control").unwrap();
        assert_eq!(ctrl.offset, cm::CORE_CONTROL,
            "Core_Control offset: JSON=0x{:X} spec=0x{:X}",
            ctrl.offset, cm::CORE_CONTROL);

        let status = core.register("Core_Status").unwrap();
        assert_eq!(status.offset, cm::CORE_STATUS,
            "Core_Status offset");

        let pc = core.register("Core_PC").unwrap();
        assert_eq!(pc.offset, cm::CORE_PC,
            "Core_PC offset: JSON=0x{:X} spec=0x{:X}",
            pc.offset, cm::CORE_PC);

        let sp = core.register("Core_SP").unwrap();
        assert_eq!(sp.offset, cm::CORE_SP,
            "Core_SP offset: JSON=0x{:X} spec=0x{:X}",
            sp.offset, cm::CORE_SP);

        let lr = core.register("Core_LR").unwrap();
        assert_eq!(lr.offset, cm::CORE_LR,
            "Core_LR offset: JSON=0x{:X} spec=0x{:X}",
            lr.offset, cm::CORE_LR);
    }

    /// Spot check: extract a known BD configuration from raw words using
    /// both the RegisterDb layout and hand-coded constants, verify identical.
    #[test]
    fn spot_check_bd_extraction_matches() {
        let layout = fallback_aie2_layout();
        use super::super::registers_spec::memory_module::bd as mm_bd;

        // Construct a realistic BD word 0:
        // Buffer_Length = 1024 words (0x400), Base_Address = 0x100
        let word0: u32 = (0x100 << 14) | 0x400;

        // Extract with layout
        let layout_buf_len = layout.memory_bd.buffer_length.extract(word0);
        let layout_base_addr = layout.memory_bd.base_address.extract(word0);

        // Extract with hand-coded constants
        let spec_buf_len = word0 & mm_bd::WORD0_BUFFER_LEN_MASK;
        let spec_base_addr = (word0 >> mm_bd::WORD0_BASE_ADDR_SHIFT) & mm_bd::WORD0_BASE_ADDR_MASK;

        assert_eq!(layout_buf_len, spec_buf_len, "Buffer_Length extraction mismatch");
        assert_eq!(layout_base_addr, spec_base_addr, "Base_Address extraction mismatch");

        // Construct a realistic BD word 5:
        // Valid=1, Use_Next=1, Next_BD=3, Lock_Acq_ID=5, Lock_Acq_Value=1
        let word5: u32 = (1 << 25) | (1 << 26) | (3 << 27) | (1 << 5) | 5;

        let layout_valid = layout.memory_bd.valid_bd.extract_bool(word5);
        let layout_use_next = layout.memory_bd.use_next_bd.extract_bool(word5);
        let layout_next = layout.memory_bd.next_bd.extract(word5);
        let layout_acq_id = layout.memory_bd.lock_acq_id.extract(word5);

        let spec_valid = (word5 >> mm_bd::WORD5_VALID_BD_BIT) & 1 != 0;
        let spec_use_next = (word5 >> mm_bd::WORD5_USE_NEXT_BD_BIT) & 1 != 0;
        let spec_next = (word5 >> mm_bd::WORD5_NEXT_BD_SHIFT) & mm_bd::WORD5_NEXT_BD_MASK;
        let spec_acq_id = word5 & mm_bd::WORD5_LOCK_ACQ_ID_MASK;

        assert_eq!(layout_valid, spec_valid, "Valid_BD mismatch");
        assert_eq!(layout_use_next, spec_use_next, "Use_Next_BD mismatch");
        assert_eq!(layout_next, spec_next, "Next_BD mismatch");
        assert_eq!(layout_acq_id, spec_acq_id, "Lock_Acq_ID mismatch");
    }
}
