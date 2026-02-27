//! Data-driven device model parsed from mlir-aie's AIETargetModel.
//!
//! This module provides Rust types that mirror the JSON output of
//! `tools/mlir-aie-bridge.py`, which queries mlir-aie's Python bindings
//! to extract device configuration parameters for all NPU variants.
//!
//! The device model captures everything the compiler knows about the
//! hardware: array dimensions, memory sizes, lock/BD counts, tile
//! classification, and stream switch port topology. This is the same
//! pattern as our llvm-aie TableGen integration: external tool extracts
//! structured data from the source of truth, Rust parses it.
//!
//! # Usage
//!
//! ```no_run
//! use std::path::Path;
//! use xdna_emu::device::model::DeviceModelSet;
//!
//! // Load from pre-generated JSON file
//! let models = DeviceModelSet::load_from_json(
//!     Path::new("tools/aie-device-models.json")
//! ).unwrap();
//!
//! let npu1 = models.get("npu1").unwrap();
//! assert_eq!(npu1.columns, 4);
//! assert_eq!(npu1.rows, 6);
//! assert_eq!(npu1.local_memory_size, 65536);
//! ```

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Top-level container
// ---------------------------------------------------------------------------

/// Complete output of `aie-device-dump.py`: metadata + all device models.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceModelSet {
    /// Tool that generated this data.
    pub generator: String,

    /// Path to the mlir-aie Python bindings used for extraction.
    pub mlir_aie_python_path: String,

    /// Device models keyed by mlir-aie device name (e.g. "npu1", "npu2").
    pub devices: HashMap<String, DeviceModel>,
}

impl DeviceModelSet {
    /// Load device models from a pre-generated JSON file.
    ///
    /// The JSON file is produced by `tools/aie-device-dump.py` and checked
    /// into the repository so that builds don't depend on mlir-aie at
    /// compile time.
    pub fn load_from_json(path: &Path) -> Result<Self, DeviceModelError> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            DeviceModelError::Io(format!("reading {}: {}", path.display(), e))
        })?;
        Self::parse_json(&contents)
    }

    /// Parse device models from a JSON string.
    pub fn parse_json(json: &str) -> Result<Self, DeviceModelError> {
        serde_json::from_str(json).map_err(|e| {
            DeviceModelError::Parse(format!("parsing device model JSON: {}", e))
        })
    }

    /// Look up a device model by mlir-aie name.
    pub fn get(&self, name: &str) -> Option<&DeviceModel> {
        self.devices.get(name)
    }

    /// Get the NPU1 (Phoenix) base device model.
    pub fn npu1(&self) -> Option<&DeviceModel> {
        self.get("npu1")
    }

    /// Get the NPU2 (Strix) base device model.
    pub fn npu2(&self) -> Option<&DeviceModel> {
        self.get("npu2")
    }
}

// ---------------------------------------------------------------------------
// Per-device model
// ---------------------------------------------------------------------------

/// Device model for a single NPU variant, extracted from mlir-aie.
///
/// Maps to mlir-aie's `AIETargetModel` class hierarchy. Each device
/// variant (npu1, npu1_1col, npu2, etc.) has its own model instance.
///
/// # Naming conventions
///
/// The naming is confusing across projects:
///
/// | mlir-aie | xdna-driver | Architecture | Hardware |
/// |----------|-------------|--------------|----------|
/// | npu1     | NPU1        | AIE2 (XDNA)  | Phoenix  |
/// | npu2     | NPU4        | AIE2P (XDNA2)| Strix    |
///
/// The `_Ncol` variants (npu1_1col, etc.) are column-count subsets.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceModel {
    /// AIEDevice enum value from AIEAttrs.td.
    pub device_id: u32,

    /// Number of tile columns (compute columns, excludes any "column 0 offset").
    /// NPU1 = 4, NPU2 = 8.
    pub columns: u32,

    /// Number of tile rows (including shim row 0).
    /// Always 6 for current NPUs.
    pub rows: u32,

    /// Whether this is an NPU device (vs Versal FPGA).
    pub is_npu: bool,

    /// Core tile local data memory size in bytes (typically 64 KB).
    pub local_memory_size: u32,

    /// Memory tile size in bytes (typically 512 KB).
    pub mem_tile_size: u32,

    /// Number of memory tile rows (typically 1).
    pub num_mem_tile_rows: u32,

    /// Maximum value storable in a lock register (e.g. 63 for AIE2).
    pub max_lock_value: u32,

    /// Address generation granularity in bits (e.g. 32 for word-addressed).
    pub address_gen_granularity: u32,

    /// Bit shift to extract column from tile address.
    /// For NPU1/NPU2 this is 25 (32 MB per column).
    pub column_shift: u32,

    /// Bit shift to extract row from tile address.
    /// For NPU1/NPU2 this is 20 (1 MB per row).
    pub row_shift: u32,

    /// Base addresses for the four neighboring memory regions
    /// as seen from a compute tile's AGU.
    pub mem_base_addresses: MemBaseAddresses,

    /// Per-tile-type configuration (locks, BDs, stream switch ports).
    pub tile_types: HashMap<String, TileTypeConfig>,

    /// Full (col, row) -> tile type map for the entire array.
    pub tile_map: Vec<TileMapEntry>,
}

impl DeviceModel {
    /// Look up tile type configuration by name.
    pub fn tile_type_config(&self, name: &str) -> Option<&TileTypeConfig> {
        self.tile_types.get(name)
    }

    /// Get the core tile configuration.
    pub fn core_config(&self) -> Option<&TileTypeConfig> {
        self.tile_type_config("core")
    }

    /// Get the memory tile configuration.
    pub fn mem_tile_config(&self) -> Option<&TileTypeConfig> {
        self.tile_type_config("mem_tile")
    }

    /// Get the shim NoC tile configuration.
    pub fn shim_noc_config(&self) -> Option<&TileTypeConfig> {
        self.tile_type_config("shim_noc")
    }

    /// Classify a tile position by type name.
    pub fn classify_tile(&self, col: u32, row: u32) -> Option<&str> {
        self.tile_map
            .iter()
            .find(|t| t.col == col && t.row == row)
            .map(|t| t.tile_type.as_str())
    }
}

// ---------------------------------------------------------------------------
// Nested types
// ---------------------------------------------------------------------------

/// Memory base addresses for the four cardinal neighbors.
///
/// These are the byte addresses in the compute tile's AGU address space
/// where neighboring tiles' memories are mapped:
/// - South (0x40000): memory of tile below
/// - West (0x50000): memory of tile to the left
/// - North (0x60000): memory of tile above
/// - East (0x70000): memory of tile to the right
#[derive(Debug, Clone, Deserialize)]
pub struct MemBaseAddresses {
    pub south: u32,
    pub west: u32,
    pub north: u32,
    pub east: u32,
}

/// Configuration for a tile type (core, mem_tile, shim_noc).
#[derive(Debug, Clone, Deserialize)]
pub struct TileTypeConfig {
    /// Number of lock primitives in this tile type.
    pub num_locks: u32,

    /// Number of DMA buffer descriptors in this tile type.
    pub num_bds: u32,

    /// Number of memory banks in this tile type.
    /// Note: mlir-aie returns 4 for compute tiles while AM020 lists 8 physical
    /// banks. This is because mlir-aie counts bank groups (each containing 2
    /// interleaved banks). Use with care.
    pub num_banks: u32,

    /// Representative tile position [col, row] used for querying.
    /// An interior tile is preferred to capture maximum port counts.
    pub representative: [u32; 2],

    /// Stream switch port counts by bundle name (e.g. "Core", "DMA", "South").
    /// Each entry has master and slave counts.
    #[serde(default)]
    pub switchbox_ports: HashMap<String, PortCounts>,

    /// Shim mux port counts (only present for shim tiles).
    #[serde(default)]
    pub shim_mux_ports: HashMap<String, PortCounts>,
}

/// Master/slave port count pair for a stream switch bundle.
#[derive(Debug, Clone, Deserialize)]
pub struct PortCounts {
    /// Number of master (output) ports for this bundle.
    pub master: u32,
    /// Number of slave (input) ports for this bundle.
    pub slave: u32,
}

/// A single entry in the tile map: (col, row) -> type.
#[derive(Debug, Clone, Deserialize)]
pub struct TileMapEntry {
    pub col: u32,
    pub row: u32,
    /// Tile type name: "core", "mem_tile", "shim_noc", "shim_pl", or "unknown".
    #[serde(rename = "type")]
    pub tile_type: String,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from loading or validating device models.
#[derive(Debug)]
pub enum DeviceModelError {
    /// I/O error reading the model file.
    Io(String),
    /// JSON parse error.
    Parse(String),
    /// Validation error: parsed model doesn't match expected constants.
    Validation(String),
}

impl std::fmt::Display for DeviceModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceModelError::Io(msg) => write!(f, "I/O error: {}", msg),
            DeviceModelError::Parse(msg) => write!(f, "parse error: {}", msg),
            DeviceModelError::Validation(msg) => write!(f, "validation error: {}", msg),
        }
    }
}

impl std::error::Error for DeviceModelError {}

// ---------------------------------------------------------------------------
// Validation of model values against known-correct reference values
// ---------------------------------------------------------------------------

/// Validate that a parsed NPU1 device model contains the expected reference
/// values. These values were originally validated against our aie2_spec
/// constants and are now checked directly as regression guards -- if the
/// model JSON is regenerated and values change unexpectedly, this catches it.
///
/// Returns Ok(()) if all assertions pass, or Err with a list of mismatches.
pub fn validate_against_spec(model: &DeviceModel) -> Result<(), DeviceModelError> {
    let mut mismatches = Vec::new();

    // Helper macro to check a value and collect mismatches instead of panicking
    macro_rules! check {
        ($name:expr, $model_val:expr, $expected:expr) => {
            let model_val = $model_val;
            let expected = $expected;
            if model_val != expected {
                mismatches.push(format!(
                    "{}: model={}, expected={}",
                    $name, model_val, expected
                ));
            }
        };
    }

    // --- Global parameters ---
    check!("max_lock_value", model.max_lock_value, 63_u32);           // 0x3F
    check!("address_gen_granularity", model.address_gen_granularity, 32_u32); // 32-bit words

    // Cross-validate: the compile-time Lock::MAX_VALUE const must agree with
    // the model. This catches silent drift if a future architecture changes
    // the lock value range.
    check!(
        "Lock::MAX_VALUE vs model",
        crate::device::tile::Lock::MAX_VALUE as u32,
        model.max_lock_value
    );

    // --- Memory sizes ---
    check!("local_memory_size", model.local_memory_size, 65536_u32);   // 64 KB
    check!("mem_tile_size", model.mem_tile_size, 524288_u32);          // 512 KB

    // --- Tile type: core ---
    if let Some(core) = model.core_config() {
        check!("core.num_locks", core.num_locks, 16_u32);
        check!("core.num_bds", core.num_bds, 16_u32);

        // Stream switch port counts for compute tiles
        if let Some(dma) = core.switchbox_ports.get("DMA") {
            check!("core.dma_mm2s_ports", dma.master, 2_u32);
            check!("core.dma_s2mm_ports", dma.slave, 2_u32);
        } else {
            mismatches.push("core: missing DMA switchbox_ports".to_string());
        }
    } else {
        mismatches.push("missing core tile type".to_string());
    }

    // --- Tile type: mem_tile ---
    if let Some(mem) = model.mem_tile_config() {
        check!("mem_tile.num_locks", mem.num_locks, 64_u32);
        check!("mem_tile.num_bds", mem.num_bds, 48_u32);

        // DMA channels for memory tiles
        if let Some(dma) = mem.switchbox_ports.get("DMA") {
            check!("mem_tile.dma_mm2s_ports", dma.master, 6_u32);
            check!("mem_tile.dma_s2mm_ports", dma.slave, 6_u32);
        } else {
            mismatches.push("mem_tile: missing DMA switchbox_ports".to_string());
        }
    } else {
        mismatches.push("missing mem_tile tile type".to_string());
    }

    // --- Tile type: shim_noc ---
    if let Some(shim) = model.shim_noc_config() {
        check!("shim_noc.num_locks", shim.num_locks, 16_u32);
        check!("shim_noc.num_bds", shim.num_bds, 16_u32);
    } else {
        mismatches.push("missing shim_noc tile type".to_string());
    }

    // --- Memory base addresses ---
    // These map to the AGU address regions for neighboring tile memories.
    // The values 0x40000, 0x50000, 0x60000, 0x70000 are each 64 KB apart,
    // representing the four quadrants of the 256 KB addressable space.
    check!(
        "mem_base_south",
        model.mem_base_addresses.south as usize,
        0x40000
    );
    check!(
        "mem_base_west",
        model.mem_base_addresses.west as usize,
        0x50000
    );
    check!(
        "mem_base_north",
        model.mem_base_addresses.north as usize,
        0x60000
    );
    check!(
        "mem_base_east",
        model.mem_base_addresses.east as usize,
        0x70000
    );

    // --- Address space ---
    check!("column_shift", model.column_shift as usize, 25);
    check!("row_shift", model.row_shift as usize, 20);

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(DeviceModelError::Validation(format!(
            "{} mismatch(es):\n  {}",
            mismatches.len(),
            mismatches.join("\n  ")
        )))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Path to the pre-generated device model JSON.
    fn json_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/aie-device-models.json")
    }

    /// Verify we can find and parse the JSON file.
    fn load_models() -> DeviceModelSet {
        let path = json_path();
        if !path.exists() {
            panic!(
                "Device model JSON not found at {}. \
                 Run: python3 tools/mlir-aie-bridge.py device-model > tools/aie-device-models.json",
                path.display()
            );
        }
        DeviceModelSet::load_from_json(&path)
            .unwrap_or_else(|e| panic!("Failed to load device models: {}", e))
    }

    #[test]
    fn parse_device_model_json() {
        let models = load_models();

        // Should have at least npu1 and npu2
        assert!(
            models.devices.contains_key("npu1"),
            "missing npu1 device"
        );
        assert!(
            models.devices.contains_key("npu2"),
            "missing npu2 device"
        );
        assert!(
            models.generator.contains("mlir-aie-bridge") || models.generator.contains("aie-device-dump"),
            "unexpected generator: {}", models.generator
        );
    }

    #[test]
    fn npu1_basic_parameters() {
        let models = load_models();
        let npu1 = models.npu1().expect("npu1 not found");

        assert_eq!(npu1.device_id, 4);
        assert_eq!(npu1.columns, 4);
        assert_eq!(npu1.rows, 6);
        assert!(npu1.is_npu);
        assert_eq!(npu1.local_memory_size, 65536);
        assert_eq!(npu1.mem_tile_size, 524288);
        assert_eq!(npu1.num_mem_tile_rows, 1);
        assert_eq!(npu1.column_shift, 25);
        assert_eq!(npu1.row_shift, 20);
        assert_eq!(npu1.max_lock_value, 63);
        assert_eq!(npu1.address_gen_granularity, 32);
    }

    #[test]
    fn npu1_tile_type_banks() {
        let models = load_models();
        let npu1 = models.npu1().expect("npu1 not found");

        let core = npu1.core_config().expect("core config");
        // mlir-aie returns 4 (bank groups); AM020 says 8 (physical banks)
        assert_eq!(core.num_banks, 4);

        let mem = npu1.mem_tile_config().expect("mem_tile config");
        assert_eq!(mem.num_banks, 8);
    }

    #[test]
    fn npu1_tile_types() {
        let models = load_models();
        let npu1 = models.npu1().expect("npu1 not found");

        let core = npu1.core_config().expect("core config");
        assert_eq!(core.num_locks, 16);
        assert_eq!(core.num_bds, 16);

        let mem = npu1.mem_tile_config().expect("mem_tile config");
        assert_eq!(mem.num_locks, 64);
        assert_eq!(mem.num_bds, 48);

        let shim = npu1.shim_noc_config().expect("shim_noc config");
        assert_eq!(shim.num_locks, 16);
        assert_eq!(shim.num_bds, 16);
    }

    #[test]
    fn npu1_tile_map_coverage() {
        let models = load_models();
        let npu1 = models.npu1().expect("npu1 not found");

        // Should have cols * rows tile map entries
        let expected = (npu1.columns * npu1.rows) as usize;
        assert_eq!(npu1.tile_map.len(), expected);

        // Verify classification matches expected layout
        assert_eq!(npu1.classify_tile(0, 0), Some("shim_noc"));
        assert_eq!(npu1.classify_tile(0, 1), Some("mem_tile"));
        assert_eq!(npu1.classify_tile(0, 2), Some("core"));
        assert_eq!(npu1.classify_tile(0, 5), Some("core"));

        // Out-of-range returns None
        assert_eq!(npu1.classify_tile(99, 99), None);
    }

    #[test]
    fn npu1_mem_base_addresses() {
        let models = load_models();
        let npu1 = models.npu1().expect("npu1 not found");

        assert_eq!(npu1.mem_base_addresses.south, 0x40000);
        assert_eq!(npu1.mem_base_addresses.west, 0x50000);
        assert_eq!(npu1.mem_base_addresses.north, 0x60000);
        assert_eq!(npu1.mem_base_addresses.east, 0x70000);
    }

    #[test]
    fn npu1_stream_switch_ports() {
        let models = load_models();
        let npu1 = models.npu1().expect("npu1 not found");

        let core = npu1.core_config().expect("core config");
        let dma = core.switchbox_ports.get("DMA").expect("DMA ports");
        assert_eq!(dma.master, 2); // MM2S channels
        assert_eq!(dma.slave, 2); // S2MM channels

        let mem = npu1.mem_tile_config().expect("mem_tile config");
        let dma = mem.switchbox_ports.get("DMA").expect("DMA ports");
        assert_eq!(dma.master, 6);
        assert_eq!(dma.slave, 6);
    }

    #[test]
    fn npu2_basic_parameters() {
        let models = load_models();
        let npu2 = models.npu2().expect("npu2 not found");

        assert_eq!(npu2.device_id, 8);
        assert_eq!(npu2.columns, 8);
        assert_eq!(npu2.rows, 6);
        assert!(npu2.is_npu);
    }

    #[test]
    fn npu1_column_variants() {
        let models = load_models();

        // All column variants share the same per-tile parameters
        for (name, expected_cols) in [
            ("npu1_1col", 1),
            ("npu1_2col", 2),
            ("npu1_3col", 3),
            ("npu1", 4),
        ] {
            let dev = models.get(name).unwrap_or_else(|| panic!("missing {}", name));
            assert_eq!(dev.columns, expected_cols, "{} columns", name);
            assert_eq!(dev.rows, 6, "{} rows", name);
            assert_eq!(
                dev.local_memory_size, 65536,
                "{} local memory",
                name
            );
        }
    }

    #[test]
    fn validate_npu1_reference_values() {
        let models = load_models();
        let npu1 = models.npu1().expect("npu1 not found");

        validate_against_spec(npu1).unwrap_or_else(|e| {
            panic!("NPU1 model does not match expected reference values: {}", e);
        });
    }
}
