//! Device model extractor -- parses `tools/aie-device-models.json` into typed
//! `ArchModel` instances.
//!
//! One model per device name (12 total: npu1 + 3 column variants, npu2 + 7
//! column variants). Column variants are virtualized subsets of the full
//! device -- same tile-level parameters, different array widths.
//!
//! ## Strict parsing
//!
//! Every JSON object is validated for unknown fields. If the JSON generator
//! adds a field we haven't accounted for, the extractor fails loudly rather
//! than silently ignoring data.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::{Map, Value};

use crate::types::*;

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur during device model extraction.
#[derive(Debug)]
pub enum ExtractError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnknownField { context: String, field: String },
    MissingField { context: String, field: String },
    UnknownDevice { name: String },
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Json(e) => write!(f, "JSON parse error: {}", e),
            Self::UnknownField { context, field } => {
                write!(f, "unknown field '{}' in {}", field, context)
            }
            Self::MissingField { context, field } => {
                write!(f, "missing field '{}' in {}", field, context)
            }
            Self::UnknownDevice { name } => {
                write!(f, "unknown device '{}' (expected npu1* or npu2*)", name)
            }
        }
    }
}

impl std::error::Error for ExtractError {}

impl From<std::io::Error> for ExtractError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for ExtractError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

// ============================================================================
// Public interface
// ============================================================================

/// Parse all devices from a device model JSON file.
pub fn extract_device_models(path: &Path) -> Result<HashMap<String, ArchModel>, ExtractError> {
    let content = std::fs::read_to_string(path)?;
    let root: Value = serde_json::from_str(&content)?;

    let root_obj = root
        .as_object()
        .ok_or_else(|| ExtractError::MissingField {
            context: "root".into(),
            field: "(expected object)".into(),
        })?;

    check_keys(root_obj, &["generator", "mlir_aie_python_path", "devices"], "root")?;

    let devices_val = require_field(root_obj, "devices", "root")?;
    let devices_obj = devices_val.as_object().ok_or_else(|| ExtractError::MissingField {
        context: "root.devices".into(),
        field: "(expected object)".into(),
    })?;

    let file_str = path.display().to_string();
    let mut result = HashMap::new();
    for (name, value) in devices_obj {
        let model = extract_device(name, value, &file_str)?;
        result.insert(name.clone(), model);
    }
    Ok(result)
}

/// Parse a single named device from a device model JSON file.
pub fn extract_device_model(path: &Path, device: &str) -> Result<ArchModel, ExtractError> {
    let models = extract_device_models(path)?;
    models
        .into_iter()
        .find(|(name, _)| name == device)
        .map(|(_, model)| model)
        .ok_or_else(|| ExtractError::UnknownDevice {
            name: device.to_string(),
        })
}

// ============================================================================
// Strict field checking
// ============================================================================

/// Validate that all keys in `obj` are in the `known` set.
/// Returns `Err(UnknownField)` on the first unrecognized key.
fn check_keys(obj: &Map<String, Value>, known: &[&str], context: &str) -> Result<(), ExtractError> {
    for key in obj.keys() {
        if !known.contains(&key.as_str()) {
            return Err(ExtractError::UnknownField {
                context: context.into(),
                field: key.clone(),
            });
        }
    }
    Ok(())
}

/// Get a required field from a JSON object.
fn require_field<'a>(
    obj: &'a Map<String, Value>,
    field: &str,
    context: &str,
) -> Result<&'a Value, ExtractError> {
    obj.get(field).ok_or_else(|| ExtractError::MissingField {
        context: context.into(),
        field: field.into(),
    })
}

/// Get a required integer field.
fn require_u64(obj: &Map<String, Value>, field: &str, context: &str) -> Result<u64, ExtractError> {
    let val = require_field(obj, field, context)?;
    val.as_u64().ok_or_else(|| ExtractError::MissingField {
        context: context.into(),
        field: format!("{} (expected integer)", field),
    })
}

/// Get a required boolean field.
fn require_bool(
    obj: &Map<String, Value>,
    field: &str,
    context: &str,
) -> Result<bool, ExtractError> {
    let val = require_field(obj, field, context)?;
    val.as_bool().ok_or_else(|| ExtractError::MissingField {
        context: context.into(),
        field: format!("{} (expected bool)", field),
    })
}

// ============================================================================
// Architecture inference
// ============================================================================

/// Infer architecture from device name prefix.
fn infer_architecture(name: &str) -> Result<Architecture, ExtractError> {
    if name.starts_with("npu1") {
        Ok(Architecture::Aie2)
    } else if name.starts_with("npu2") {
        Ok(Architecture::Aie2p)
    } else {
        Err(ExtractError::UnknownDevice {
            name: name.to_string(),
        })
    }
}

// ============================================================================
// Device extraction
// ============================================================================

const DEVICE_KNOWN_FIELDS: &[&str] = &[
    "device_id",
    "columns",
    "rows",
    "is_npu",
    "local_memory_size",
    "mem_tile_size",
    "num_mem_tile_rows",
    "max_lock_value",
    "address_gen_granularity",
    "column_shift",
    "row_shift",
    "mem_base_addresses",
    "tile_types",
    "tile_map",
];

fn extract_device(name: &str, value: &Value, file: &str) -> Result<ArchModel, ExtractError> {
    let ctx = name;
    let obj = value.as_object().ok_or_else(|| ExtractError::MissingField {
        context: ctx.into(),
        field: "(expected object)".into(),
    })?;

    check_keys(obj, DEVICE_KNOWN_FIELDS, ctx)?;

    let arch = infer_architecture(name)?;
    let device_id = require_u64(obj, "device_id", ctx)? as u32;
    let columns = require_u64(obj, "columns", ctx)? as u8;
    let rows = require_u64(obj, "rows", ctx)? as u8;
    let is_npu = require_bool(obj, "is_npu", ctx)?;
    let local_memory_size = require_u64(obj, "local_memory_size", ctx)?;
    let mem_tile_size = require_u64(obj, "mem_tile_size", ctx)?;
    let num_mem_tile_rows = require_u64(obj, "num_mem_tile_rows", ctx)? as u8;
    let column_shift = require_u64(obj, "column_shift", ctx)? as u8;
    let row_shift = require_u64(obj, "row_shift", ctx)? as u8;

    let constants = extract_constants(obj, ctx, file)?;
    let tile_types = extract_tile_types(obj, ctx, file, local_memory_size, mem_tile_size)?;
    let tile_map = extract_tile_map(obj, ctx, file)?;

    let topology = ArrayTopology {
        columns,
        rows,
        num_mem_tile_rows,
        column_shift,
        row_shift,
        tile_map,
        source: SourceAttribution {
            origin: Source::DeviceModel,
            file: file.into(),
            detail: format!("{}.topology", name),
        },
    };

    Ok(ArchModel {
        arch,
        generation: None,
        device_id: Some(device_id),
        is_npu,
        tile_types,
        array_topology: Some(topology),
        device_constants: Some(constants),
        relationships: Vec::new(),
    })
}

// ============================================================================
// Constants extraction
// ============================================================================

fn extract_constants(
    obj: &Map<String, Value>,
    ctx: &str,
    file: &str,
) -> Result<DeviceConstants, ExtractError> {
    let max_lock_value = require_u64(obj, "max_lock_value", ctx)? as u32;
    let address_gen_granularity = require_u64(obj, "address_gen_granularity", ctx)? as u32;

    let mem_ctx = format!("{}.mem_base_addresses", ctx);
    let mem_val = require_field(obj, "mem_base_addresses", ctx)?;
    let mem_obj = mem_val.as_object().ok_or_else(|| ExtractError::MissingField {
        context: mem_ctx.clone(),
        field: "(expected object)".into(),
    })?;

    check_keys(mem_obj, &["south", "west", "north", "east"], &mem_ctx)?;

    let mut mem_base_addresses = BTreeMap::new();
    for (dir, val) in mem_obj {
        let addr = val.as_u64().ok_or_else(|| ExtractError::MissingField {
            context: mem_ctx.clone(),
            field: format!("{} (expected integer)", dir),
        })?;
        mem_base_addresses.insert(dir.clone(), addr);
    }

    // Architecture-appropriate defaults for fields not in the JSON.
    // min_lock_value: AIE2/AIE2P both use -64 (from aie-rt VAL_LOWER_BOUND).
    let min_lock_value = -64;

    Ok(DeviceConstants {
        max_lock_value,
        min_lock_value,
        address_gen_granularity,
        accumulator_cascade_bits: None, // populated by aie-rt extractor
        mem_base_addresses,
        properties: DeviceProperties {
            uses_semaphore_locks: true, // AIE2/AIE2P both use semaphore locks
            uses_multi_dim_bds: true,   // AIE2/AIE2P both use multi-dim BDs
        },
        source: SourceAttribution {
            origin: Source::DeviceModel,
            file: file.into(),
            detail: format!("{}.constants", ctx),
        },
    })
}

// ============================================================================
// Tile type extraction
// ============================================================================

const TILE_TYPE_KNOWN_FIELDS: &[&str] = &[
    "num_locks",
    "num_bds",
    "num_banks",
    "representative",
    "bank_size",
    "program_memory_size",
    "switchbox_ports",
    "shim_mux_ports",
];

fn extract_tile_types(
    device_obj: &Map<String, Value>,
    device_ctx: &str,
    file: &str,
    local_memory_size: u64,
    mem_tile_size: u64,
) -> Result<Vec<TileTypeModel>, ExtractError> {
    let ctx = format!("{}.tile_types", device_ctx);
    let types_val = require_field(device_obj, "tile_types", device_ctx)?;
    let types_obj = types_val.as_object().ok_or_else(|| ExtractError::MissingField {
        context: ctx.clone(),
        field: "(expected object)".into(),
    })?;

    let mut result = Vec::new();
    for (type_name, type_val) in types_obj {
        let tile_ctx = format!("{}.{}", ctx, type_name);
        let model = extract_tile_type(
            type_name,
            type_val,
            &tile_ctx,
            file,
            local_memory_size,
            mem_tile_size,
        )?;
        result.push(model);
    }
    Ok(result)
}

fn extract_tile_type(
    name: &str,
    value: &Value,
    ctx: &str,
    file: &str,
    local_memory_size: u64,
    mem_tile_size: u64,
) -> Result<TileTypeModel, ExtractError> {
    let obj = value.as_object().ok_or_else(|| ExtractError::MissingField {
        context: ctx.into(),
        field: "(expected object)".into(),
    })?;

    check_keys(obj, TILE_TYPE_KNOWN_FIELDS, ctx)?;

    let kind = match name {
        "core" => TileKind::Compute,
        "mem_tile" => TileKind::Mem,
        "shim_noc" => TileKind::ShimNoc,
        "shim_pl" => TileKind::ShimPl,
        _ => TileKind::Compute, // conservative fallback
    };

    let num_locks = require_u64(obj, "num_locks", ctx)? as u8;
    let num_bds = require_u64(obj, "num_bds", ctx)? as u8;
    let num_banks = require_u64(obj, "num_banks", ctx)? as u8;
    let bank_size = require_u64(obj, "bank_size", ctx)?;

    let representative = if let Some(rep_val) = obj.get("representative") {
        let arr = rep_val.as_array().ok_or_else(|| ExtractError::MissingField {
            context: ctx.into(),
            field: "representative (expected array)".into(),
        })?;
        if arr.len() == 2 {
            Some((arr[0].as_u64().unwrap_or(0) as u8, arr[1].as_u64().unwrap_or(0) as u8))
        } else {
            None
        }
    } else {
        None
    };

    let program_memory_bytes = obj
        .get("program_memory_size")
        .and_then(|v| v.as_u64());

    // Memory model: compute tiles use local_memory_size, memtiles use mem_tile_size,
    // shim tiles use bank_size * num_banks.
    let size_bytes = match kind {
        TileKind::Compute => local_memory_size,
        TileKind::Mem => mem_tile_size,
        TileKind::ShimNoc | TileKind::ShimPl => bank_size * num_banks as u64,
    };

    let mem_source = SourceAttribution {
        origin: Source::DeviceModel,
        file: file.into(),
        detail: format!("{}.memory", ctx),
    };
    let memory = Some(MemoryModel::new(
        size_bytes,
        BankingModel {
            num_banks,
            bank_size,
            // mlir-aie standard memory width is 128-bit (16 bytes per row).
            bank_width_bits: 128,
            source: mem_source.clone(),
        },
        None, // physical banking not available from device model
        program_memory_bytes,
        mem_source,
    ));

    // Extract port bundles
    let switchbox_ports = if let Some(sp_val) = obj.get("switchbox_ports") {
        extract_ports(sp_val, &format!("{}.switchbox_ports", ctx))?
    } else {
        Vec::new()
    };

    let shim_mux_ports = if let Some(sm_val) = obj.get("shim_mux_ports") {
        extract_ports(sm_val, &format!("{}.shim_mux_ports", ctx))?
    } else {
        Vec::new()
    };

    // DMA channel count: derived from DMA master count in the appropriate
    // port namespace. Shim tiles use shim_mux_ports, others use switchbox_ports.
    let channels = match kind {
        TileKind::ShimNoc | TileKind::ShimPl => {
            shim_mux_ports
                .iter()
                .find(|p| p.bundle == "DMA")
                .map(|p| p.masters)
                .unwrap_or(0)
        }
        _ => {
            switchbox_ports
                .iter()
                .find(|p| p.bundle == "DMA")
                .map(|p| p.masters)
                .unwrap_or(0)
        }
    };

    let instance_source = SourceAttribution {
        origin: Source::DeviceModel,
        file: file.into(),
        detail: format!("{}.instances", ctx),
    };
    Ok(TileTypeModel {
        kind,
        name: name.to_string(),
        representative,
        instances: InstanceCount {
            locks: Confirmed::new(num_locks, instance_source.clone()),
            bds: Confirmed::new(num_bds, instance_source.clone()),
            channels: Confirmed::new(channels, instance_source),
        },
        memory,
        dma_capabilities: None, // populated by aie-rt extractor
        switchbox_ports,
        shim_mux_ports,
        modules: Vec::new(), // populated by AM025 regdb extractor
        bd_schema: None,        // populated by BD schema extraction
        channel_schema: None,   // populated by channel schema extraction
        source: SourceAttribution {
            origin: Source::DeviceModel,
            file: file.into(),
            detail: ctx.into(),
        },
    })
}

// ============================================================================
// Port extraction
// ============================================================================

fn extract_ports(value: &Value, ctx: &str) -> Result<Vec<PortBundle>, ExtractError> {
    let obj = value.as_object().ok_or_else(|| ExtractError::MissingField {
        context: ctx.into(),
        field: "(expected object)".into(),
    })?;

    let mut result = Vec::new();
    for (bundle_name, bundle_val) in obj {
        let bundle_ctx = format!("{}.{}", ctx, bundle_name);
        let bundle_obj = bundle_val.as_object().ok_or_else(|| ExtractError::MissingField {
            context: bundle_ctx.clone(),
            field: "(expected object)".into(),
        })?;

        check_keys(bundle_obj, &["master", "slave"], &bundle_ctx)?;

        let masters = require_u64(bundle_obj, "master", &bundle_ctx)? as u8;
        let slaves = require_u64(bundle_obj, "slave", &bundle_ctx)? as u8;

        result.push(PortBundle {
            bundle: bundle_name.clone(),
            masters,
            slaves,
        });
    }

    // Sort by bundle name for deterministic ordering.
    result.sort_by(|a, b| a.bundle.cmp(&b.bundle));
    Ok(result)
}

// ============================================================================
// Tile map extraction
// ============================================================================

const TILE_MAP_ENTRY_KNOWN_FIELDS: &[&str] = &[
    "col",
    "row",
    "type",
    "is_internal",
    "edges",
    "mem_affinity",
];

const CARDINAL_KNOWN_FIELDS: &[&str] = &["north", "south", "east", "west"];

fn extract_tile_map(
    device_obj: &Map<String, Value>,
    device_ctx: &str,
    file: &str,
) -> Result<Vec<TilePlacement>, ExtractError> {
    let ctx = format!("{}.tile_map", device_ctx);
    let map_val = require_field(device_obj, "tile_map", device_ctx)?;
    let map_arr = map_val.as_array().ok_or_else(|| ExtractError::MissingField {
        context: ctx.clone(),
        field: "(expected array)".into(),
    })?;

    let mut result = Vec::new();
    for (i, entry_val) in map_arr.iter().enumerate() {
        let entry_ctx = format!("{}[{}]", ctx, i);
        let entry_obj = entry_val.as_object().ok_or_else(|| ExtractError::MissingField {
            context: entry_ctx.clone(),
            field: "(expected object)".into(),
        })?;

        check_keys(entry_obj, TILE_MAP_ENTRY_KNOWN_FIELDS, &entry_ctx)?;

        let col = require_u64(entry_obj, "col", &entry_ctx)? as u8;
        let row = require_u64(entry_obj, "row", &entry_ctx)? as u8;

        let tile_type_val = require_field(entry_obj, "type", &entry_ctx)?;
        let tile_type = tile_type_val
            .as_str()
            .ok_or_else(|| ExtractError::MissingField {
                context: entry_ctx.clone(),
                field: "type (expected string)".into(),
            })?
            .to_string();

        let is_internal = require_bool(entry_obj, "is_internal", &entry_ctx)?;

        let edges_ctx = format!("{}.edges", entry_ctx);
        let edges_val = require_field(entry_obj, "edges", &entry_ctx)?;
        let edges = extract_cardinal_flags(edges_val, &edges_ctx)?;

        let mem_affinity = if let Some(ma_val) = entry_obj.get("mem_affinity") {
            let ma_ctx = format!("{}.mem_affinity", entry_ctx);
            Some(extract_cardinal_flags(ma_val, &ma_ctx)?)
        } else {
            None
        };

        result.push(TilePlacement {
            col,
            row,
            tile_type,
            is_internal,
            edges,
            mem_affinity,
        });
    }

    let _ = file; // file is used at the device level for SourceAttribution
    Ok(result)
}

fn extract_cardinal_flags(value: &Value, ctx: &str) -> Result<CardinalFlags, ExtractError> {
    let obj = value.as_object().ok_or_else(|| ExtractError::MissingField {
        context: ctx.into(),
        field: "(expected object)".into(),
    })?;

    check_keys(obj, CARDINAL_KNOWN_FIELDS, ctx)?;

    Ok(CardinalFlags {
        north: require_bool(obj, "north", ctx)?,
        south: require_bool(obj, "south", ctx)?,
        east: require_bool(obj, "east", ctx)?,
        west: require_bool(obj, "west", ctx)?,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn json_path() -> PathBuf {
        // CARGO_MANIFEST_DIR is crates/xdna-archspec/, go up to xdna-emu root
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/aie-device-models.json")
    }

    // Test 1: Full parse -- all 12 devices parse without error.
    #[test]
    fn test_full_parse_all_devices() {
        let models = extract_device_models(&json_path()).expect("parse failed");
        assert_eq!(models.len(), 12, "expected 12 devices, got {}", models.len());

        // Verify all expected device names are present.
        let expected = [
            "npu1", "npu1_1col", "npu1_2col", "npu1_3col",
            "npu2", "npu2_1col", "npu2_2col", "npu2_3col",
            "npu2_4col", "npu2_5col", "npu2_6col", "npu2_7col",
        ];
        for name in &expected {
            assert!(models.contains_key(*name), "missing device: {}", name);
        }
    }

    // Test 2: Spot-check npu1.
    #[test]
    fn test_spot_check_npu1() {
        let model = extract_device_model(&json_path(), "npu1").expect("parse failed");
        assert_eq!(model.arch, Architecture::Aie2);
        assert_eq!(model.device_id, Some(4));
        assert!(model.is_npu);

        let topo = model.array_topology.as_ref().expect("missing topology");
        assert_eq!(topo.columns, 4);
        assert_eq!(topo.rows, 6);
        assert_eq!(model.tile_types.len(), 3);
        assert_eq!(topo.tile_map.len(), 24); // 4 cols * 6 rows
    }

    // Test 3: Spot-check npu2.
    #[test]
    fn test_spot_check_npu2() {
        let model = extract_device_model(&json_path(), "npu2").expect("parse failed");
        assert_eq!(model.arch, Architecture::Aie2p);
        assert_eq!(model.device_id, Some(8));

        let topo = model.array_topology.as_ref().expect("missing topology");
        assert_eq!(topo.columns, 8);
        assert_eq!(topo.rows, 6);
    }

    // Test 4: Tile type coverage.
    #[test]
    fn test_tile_type_coverage() {
        let model = extract_device_model(&json_path(), "npu1").expect("parse failed");

        // Core tile
        let core = model.tile_types.iter().find(|t| t.name == "core").expect("no core");
        assert_eq!(core.kind, TileKind::Compute);
        let core_mem = core.memory.as_ref().expect("core should have memory");
        assert_eq!(core_mem.program_memory_bytes, Some(16384));
        assert_eq!(core_mem.size_bytes, 65536);
        assert_eq!(*core.instances.locks.value(), 16);
        assert_eq!(*core.instances.bds.value(), 16);

        // Memtile
        let memtile = model.tile_types.iter().find(|t| t.name == "mem_tile").expect("no memtile");
        assert_eq!(memtile.kind, TileKind::Mem);
        assert_eq!(*memtile.instances.bds.value(), 48);
        assert_eq!(*memtile.instances.locks.value(), 64);
        let mt_mem = memtile.memory.as_ref().expect("memtile should have memory");
        assert_eq!(mt_mem.size_bytes, 524288);
        assert_eq!(mt_mem.program_memory_bytes, None);

        // Shim
        let shim = model.tile_types.iter().find(|t| t.name == "shim_noc").expect("no shim");
        assert_eq!(shim.kind, TileKind::ShimNoc);
        assert!(!shim.shim_mux_ports.is_empty());
    }

    // Test 5: Port bundle counts.
    #[test]
    fn test_port_bundle_counts() {
        let model = extract_device_model(&json_path(), "npu1").expect("parse failed");

        let core = model.tile_types.iter().find(|t| t.name == "core").expect("no core");
        assert_eq!(core.switchbox_ports.len(), 9, "core should have 9 switchbox bundles");

        let shim = model.tile_types.iter().find(|t| t.name == "shim_noc").expect("no shim");
        let shim_mux_dma = shim.shim_mux_ports.iter().find(|p| p.bundle == "DMA");
        assert!(shim_mux_dma.is_some(), "shim should have DMA in shim_mux");

        let shim_mux_south = shim.shim_mux_ports.iter().find(|p| p.bundle == "South");
        assert!(shim_mux_south.is_some(), "shim should have South in shim_mux");
    }

    // Test 6: Strict unknown field detection.
    #[test]
    fn test_strict_unknown_field() {
        let root = serde_json::json!({
            "generator": "test",
            "mlir_aie_python_path": "test",
            "devices": {
                "npu1_test": {
                    "device_id": 99,
                    "columns": 1,
                    "rows": 6,
                    "is_npu": true,
                    "local_memory_size": 65536,
                    "mem_tile_size": 524288,
                    "num_mem_tile_rows": 1,
                    "max_lock_value": 63,
                    "address_gen_granularity": 32,
                    "column_shift": 25,
                    "row_shift": 20,
                    "mem_base_addresses": { "south": 0, "west": 0, "north": 0, "east": 0 },
                    "tile_types": {},
                    "tile_map": [],
                    "bogus": 42
                }
            }
        });

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), root.to_string()).unwrap();

        let err = extract_device_models(tmp.path()).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unknown field"), "expected UnknownField, got: {}", msg);
        assert!(msg.contains("bogus"), "expected 'bogus' in error, got: {}", msg);
    }

    // Test 7: Missing required field.
    #[test]
    fn test_missing_required_field() {
        let root = serde_json::json!({
            "generator": "test",
            "mlir_aie_python_path": "test",
            "devices": {
                "npu1_test": {
                    "device_id": 99,
                    "rows": 6,
                    "is_npu": true,
                    "local_memory_size": 65536,
                    "mem_tile_size": 524288,
                    "num_mem_tile_rows": 1,
                    "max_lock_value": 63,
                    "address_gen_granularity": 32,
                    "column_shift": 25,
                    "row_shift": 20,
                    "mem_base_addresses": { "south": 0, "west": 0, "north": 0, "east": 0 },
                    "tile_types": {},
                    "tile_map": []
                }
            }
        });

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), root.to_string()).unwrap();

        let err = extract_device_models(tmp.path()).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("missing field"), "expected MissingField, got: {}", msg);
        assert!(msg.contains("columns"), "expected 'columns' in error, got: {}", msg);
    }

    // Test 8: Architecture inference.
    #[test]
    fn test_architecture_inference() {
        assert!(matches!(infer_architecture("npu1"), Ok(Architecture::Aie2)));
        assert!(matches!(infer_architecture("npu1_3col"), Ok(Architecture::Aie2)));
        assert!(matches!(infer_architecture("npu2"), Ok(Architecture::Aie2p)));
        assert!(matches!(infer_architecture("npu2_7col"), Ok(Architecture::Aie2p)));
        assert!(matches!(infer_architecture("xcvc1902"), Err(ExtractError::UnknownDevice { .. })));
    }

    // Test 9: Column variants share tile types and constants.
    #[test]
    fn test_column_variants() {
        let models = extract_device_models(&json_path()).expect("parse failed");

        let npu1 = &models["npu1"];
        let npu1_1col = &models["npu1_1col"];

        // Different column counts
        let topo1 = npu1.array_topology.as_ref().unwrap();
        let topo1c = npu1_1col.array_topology.as_ref().unwrap();
        assert_eq!(topo1.columns, 4);
        assert_eq!(topo1c.columns, 1);

        // Same instance counts (locks, BDs, channels)
        let core1 = npu1.tile_types.iter().find(|t| t.name == "core").unwrap();
        let core1c = npu1_1col.tile_types.iter().find(|t| t.name == "core").unwrap();
        assert_eq!(core1.instances, core1c.instances);

        // Port bundles may differ: 1-col variants lack East/West since there
        // are no neighboring columns. But shared bundles have the same counts.
        for port in &core1c.switchbox_ports {
            if let Some(p1) = core1.switchbox_ports.iter().find(|p| p.bundle == port.bundle) {
                assert_eq!(p1.masters, port.masters, "bundle {} masters differ", port.bundle);
                assert_eq!(p1.slaves, port.slaves, "bundle {} slaves differ", port.bundle);
            }
        }

        // Same constants
        let const1 = npu1.device_constants.as_ref().unwrap();
        let const1c = npu1_1col.device_constants.as_ref().unwrap();
        assert_eq!(const1.max_lock_value, const1c.max_lock_value);
        assert_eq!(const1.address_gen_granularity, const1c.address_gen_granularity);
    }

    // Test: DMA channel count derived from port bundles.
    #[test]
    fn test_dma_channel_count() {
        let model = extract_device_model(&json_path(), "npu1").expect("parse failed");

        let core = model.tile_types.iter().find(|t| t.name == "core").unwrap();
        assert_eq!(*core.instances.channels.value(), 2, "core should have 2 DMA channels");

        let memtile = model.tile_types.iter().find(|t| t.name == "mem_tile").unwrap();
        assert_eq!(*memtile.instances.channels.value(), 6, "memtile should have 6 DMA channels");

        let shim = model.tile_types.iter().find(|t| t.name == "shim_noc").unwrap();
        assert_eq!(*shim.instances.channels.value(), 2, "shim should have 2 DMA channels (from shim_mux)");
    }

    // Test: DeviceConstants populated correctly.
    #[test]
    fn test_device_constants_from_json() {
        let model = extract_device_model(&json_path(), "npu1").expect("parse failed");
        let constants = model.device_constants.as_ref().expect("missing constants");

        assert_eq!(constants.max_lock_value, 63);
        assert_eq!(constants.min_lock_value, -64);
        assert_eq!(constants.address_gen_granularity, 32);
        assert_eq!(constants.accumulator_cascade_bits, None);
        assert_eq!(constants.mem_base_addresses.len(), 4);
        assert_eq!(constants.mem_base_addresses["south"], 262144);
        assert!(constants.properties.uses_semaphore_locks);
        assert!(constants.properties.uses_multi_dim_bds);
    }

    // Test: SourceAttribution is populated with meaningful context.
    #[test]
    fn test_source_attribution() {
        let model = extract_device_model(&json_path(), "npu1").expect("parse failed");

        // Tile type has attribution
        let core = model.tile_types.iter().find(|t| t.name == "core").unwrap();
        assert_eq!(core.source.origin, Source::DeviceModel);
        assert!(core.source.detail.contains("core"), "detail should contain tile name");

        // Constants have attribution
        let constants = model.device_constants.as_ref().unwrap();
        assert_eq!(constants.source.origin, Source::DeviceModel);
        assert!(constants.source.detail.contains("npu1"));

        // Topology has attribution
        let topo = model.array_topology.as_ref().unwrap();
        assert_eq!(topo.source.origin, Source::DeviceModel);
    }
}
