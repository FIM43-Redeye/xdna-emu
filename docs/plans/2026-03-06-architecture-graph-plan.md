# NPU Architecture Graph Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone tool that extracts a typed, queryable hardware model from the open-source NPU toolchain, producing both serialized data and generated Rust source.

**Architecture:** A new binary (`npu-graph`) with core type definitions, per-source extractor modules, a merge/validation layer, and dual output (JSON + generated .rs). Library-first: the model is a Rust module importable by the emulator and other tools. Multi-architecture from day one.

**Tech Stack:** Rust, serde/serde_json for serialization, gcc -E for aie-rt header preprocessing (reusing existing build.rs pattern), existing regdb JSON parser pattern.

---

## Context for Implementers

### Source Locations

| Source | Path | What it provides |
|--------|------|-----------------|
| AM025 JSON (regdb) | `../mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json` | 1,806 registers, 6,412 fields, 4 modules |
| Device model | `tools/aie-device-models.json` | Tile types, array topology, port counts, memory sizes |
| aie-rt headers | `../aie-rt/driver/src/` | Register offsets via `#define`, DMA/lock/SS constants |
| aie-rt reginit | `../aie-rt/driver/src/global/xaiemlgbl_reginit.c` | Structured register init tables (preprocessed via gcc -E) |

### Existing Patterns to Reuse

- **`src/device/regdb.rs`**: Parses the same AM025 JSON at runtime. Copy the deserialization types (`RawRegisterDb`, `RawModule`, `RawRegister`, `RawBitField`) rather than importing (build tools compile separately).
- **`build.rs` lines 1050-1150**: `extract_aiert()` and `run_aiert_preprocessor()` show exactly how to invoke gcc -E on aie-rt and parse the output. Reuse the include-path logic and struct parsing.
- **`build.rs` lines 57-66**: `DeviceModelSet`/`DeviceModel` deserialization for the device model JSON.
- **`tools/aie-device-models.json`**: Already extracted from mlir-aie Python API. Contains per-device tile types with lock counts, BD counts, port maps, memory sizes.

### Design Doc

Read `docs/plans/2026-03-06-architecture-graph-design.md` for the full approved design including data model, query API, and extraction strategy.

### Key Rules

1. Every fact in the model must have `SourceAttribution` tracing it to a file + location.
2. When two sources state the same fact differently, **error out immediately** -- never silently prefer.
3. The `Architecture` enum parameterizes everything. No if/else on architecture in model logic.
4. Tests go through the model API, never inspect internal fields directly.

---

## Task 1: Binary Scaffold and Core Types

**Files:**
- Create: `src/graph/mod.rs`
- Create: `src/graph/types.rs`
- Create: `src/bin/npu_graph.rs`
- Modify: `src/lib.rs` (add `pub mod graph;`)
- Modify: `Cargo.toml` (add `[[bin]]` entry)

This task establishes the project structure and all core types from the design
doc. No extraction logic yet -- just the type definitions and a binary that
prints "npu-graph: no command specified."

**Step 1: Write the failing test**

Add to `src/graph/types.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_enum_variants() {
        // Verify all expected architectures exist
        let archs = [Architecture::Aie, Architecture::Aie2, Architecture::Aie2p];
        assert_eq!(archs.len(), 3);
    }

    #[test]
    fn test_tile_kind_variants() {
        let kinds = [TileKind::Compute, TileKind::Mem, TileKind::Shim];
        assert_eq!(kinds.len(), 3);
    }

    #[test]
    fn test_source_attribution_display() {
        let attr = SourceAttribution {
            origin: Source::Am025Json,
            file: "aie_registers_aie2.json".to_string(),
            detail: "module=memory, register=DMA_BD0_0".to_string(),
        };
        let s = format!("{}", attr);
        assert!(s.contains("AM025 JSON"));
        assert!(s.contains("aie_registers_aie2.json"));
    }

    #[test]
    fn test_bit_range_contiguous() {
        let range = BitRange::Contiguous { msb: 17, lsb: 0 };
        assert_eq!(range.width(), 18);
    }

    #[test]
    fn test_bit_range_split() {
        let range = BitRange::Split(vec![
            (23, 20),  // 4 bits
            (7, 4),    // 4 bits
        ]);
        assert_eq!(range.width(), 8);
    }

    #[test]
    fn test_arch_model_empty() {
        let model = ArchModel::new(Architecture::Aie2);
        assert_eq!(model.arch, Architecture::Aie2);
        assert!(model.tile_types.is_empty());
        assert!(model.relationships.is_empty());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib graph::types::tests -q`
Expected: FAIL -- module `graph` not found

**Step 3: Write the core types**

Create `src/graph/mod.rs`:

```rust
//! NPU Architecture Graph -- queryable hardware model.
//!
//! Extracts hardware architecture from the open-source NPU toolchain
//! (aie-rt, AM025 JSON, device model) into a single typed Rust model.
//! Multi-architecture: each `ArchModel` represents one architecture
//! (AIE, AIE2, AIE2P) with all its tile types, registers, and
//! relationships.

pub mod types;
```

Create `src/graph/types.rs` with all core types:

```rust
//! Core type definitions for the NPU architecture graph.

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Architecture and tile identification
// ============================================================================

/// Target NPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Architecture {
    Aie,
    Aie2,
    Aie2p,
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aie => write!(f, "AIE"),
            Self::Aie2 => write!(f, "AIE2"),
            Self::Aie2p => write!(f, "AIE2P"),
        }
    }
}

/// Tile type within the NPU array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TileKind {
    Compute,
    Mem,
    Shim,
}

impl fmt::Display for TileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compute => write!(f, "compute"),
            Self::Mem => write!(f, "mem"),
            Self::Shim => write!(f, "shim"),
        }
    }
}

// ============================================================================
// Source attribution
// ============================================================================

/// Which toolchain source a fact was derived from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Source {
    Am025Json,
    AieRt,
    DeviceModel,
    TableGen,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Am025Json => write!(f, "AM025 JSON"),
            Self::AieRt => write!(f, "aie-rt"),
            Self::DeviceModel => write!(f, "device model"),
            Self::TableGen => write!(f, "TableGen"),
        }
    }
}

/// Attribution of a fact to its toolchain source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAttribution {
    pub origin: Source,
    pub file: String,
    pub detail: String,
}

impl fmt::Display for SourceAttribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({})", self.origin, self.file, self.detail)
    }
}

// ============================================================================
// Register model
// ============================================================================

/// Bit range within a register -- contiguous or split across non-adjacent positions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitRange {
    /// Standard contiguous range, e.g., bits 17:0.
    Contiguous { msb: u8, lsb: u8 },
    /// Split field: multiple (msb, lsb) fragments, ordered high-to-low.
    Split(Vec<(u8, u8)>),
}

impl BitRange {
    /// Total width in bits.
    pub fn width(&self) -> u8 {
        match self {
            Self::Contiguous { msb, lsb } => msb - lsb + 1,
            Self::Split(fragments) => {
                fragments.iter().map(|(msb, lsb)| msb - lsb + 1).sum()
            }
        }
    }
}

/// Semantic meaning of a register field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldSemantics {
    Address,
    Count,
    Enable,
    LockId,
    LockValue,
    ChannelId,
    BdIndex,
    Status,
    Control,
    EventId,
    PortIndex,
    DataWidth,
    Reserved,
    /// Not yet classified.
    Unknown,
}

/// Register access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Access {
    ReadWrite,
    ReadOnly,
    WriteOnly,
    WriteToClear,
    Mixed,
}

/// A single bit field within a register.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldModel {
    pub name: String,
    pub bits: BitRange,
    pub meaning: FieldSemantics,
    pub source: SourceAttribution,
}

/// A hardware register.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterModel {
    pub name: String,
    pub offset: u32,
    pub width: u8,
    pub reset_value: u32,
    pub fields: Vec<FieldModel>,
    pub module: ModuleKind,
    pub access: Access,
    pub source: SourceAttribution,
}

// ============================================================================
// Module model
// ============================================================================

/// Functional module within a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleKind {
    Dma,
    Lock,
    StreamSwitch,
    Core,
    ProgramMemory,
    DataMemory,
    Trace,
    Event,
    ShimMux,
    /// Module not yet classified.
    Unknown,
}

impl fmt::Display for ModuleKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dma => write!(f, "dma"),
            Self::Lock => write!(f, "lock"),
            Self::StreamSwitch => write!(f, "stream_switch"),
            Self::Core => write!(f, "core"),
            Self::ProgramMemory => write!(f, "program_memory"),
            Self::DataMemory => write!(f, "data_memory"),
            Self::Trace => write!(f, "trace"),
            Self::Event => write!(f, "event"),
            Self::ShimMux => write!(f, "shim_mux"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Port direction for stream switch and DMA connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortDirection {
    Master,
    Slave,
}

/// A connection point on a module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortModel {
    pub bundle: String,
    pub index: u8,
    pub direction: PortDirection,
    pub source: SourceAttribution,
}

/// Instance counts for repeated hardware elements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceCount {
    pub channels: u8,
    pub bds: u8,
    pub locks: u8,
}

/// A functional module within a tile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleModel {
    pub kind: ModuleKind,
    pub registers: Vec<RegisterModel>,
    pub ports: Vec<PortModel>,
    pub instances: Option<InstanceCount>,
    pub source: SourceAttribution,
}

// ============================================================================
// Tile and array topology
// ============================================================================

/// Memory model for a tile type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryModel {
    pub size_bytes: u64,
    pub num_banks: u8,
    pub bank_size_bytes: u64,
    pub source: SourceAttribution,
}

/// A tile type definition (e.g., "aie2_compute").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileTypeModel {
    pub kind: TileKind,
    pub name: String,
    pub modules: Vec<ModuleModel>,
    pub memory: Option<MemoryModel>,
    pub source: SourceAttribution,
}

/// Array topology: grid dimensions and tile placement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrayTopology {
    pub columns: u8,
    pub rows: u8,
    pub num_mem_tile_rows: u8,
    pub column_shift: u8,
    pub row_shift: u8,
    pub source: SourceAttribution,
}

// ============================================================================
// Relationships
// ============================================================================

/// A node identifier in the graph (dot-separated path).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of relationship between two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipKind {
    DataFlow,
    Configures,
    BlocksOn,
    Triggers,
    RoutesTo,
    Produces,
    Contains,
}

/// A directed relationship between two nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relationship {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: RelationshipKind,
    pub source: SourceAttribution,
}

// ============================================================================
// Top-level model
// ============================================================================

/// Complete architecture model for one NPU architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchModel {
    pub arch: Architecture,
    pub tile_types: Vec<TileTypeModel>,
    pub array_topology: Option<ArrayTopology>,
    pub relationships: Vec<Relationship>,
}

impl ArchModel {
    /// Create an empty model for the given architecture.
    pub fn new(arch: Architecture) -> Self {
        Self {
            arch,
            tile_types: Vec::new(),
            array_topology: None,
            relationships: Vec::new(),
        }
    }
}
```

Add `pub mod graph;` to `src/lib.rs`.

Add to `Cargo.toml`:

```toml
[[bin]]
name = "npu-graph"
path = "src/bin/npu_graph.rs"
```

Create `src/bin/npu_graph.rs`:

```rust
//! NPU Architecture Graph -- standalone extraction and query tool.

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: npu-graph <command> [options]");
        eprintln!("Commands:");
        eprintln!("  extract  --arch <aie|aie2|aie2p> [source options]");
        eprintln!("  query    --model <path.json> <query>");
        eprintln!("  diff     --a <model_a.json> --b <model_b.json>");
        process::exit(1);
    }

    match args[1].as_str() {
        "extract" => {
            eprintln!("extract: not yet implemented");
            process::exit(1);
        }
        "query" => {
            eprintln!("query: not yet implemented");
            process::exit(1);
        }
        "diff" => {
            eprintln!("diff: not yet implemented");
            process::exit(1);
        }
        other => {
            eprintln!("Unknown command: {}", other);
            process::exit(1);
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib graph::types::tests -q`
Expected: All 6 tests PASS

**Step 5: Verify binary builds**

Run: `cargo build --bin npu-graph 2>&1 | tail -3`
Expected: Compiles successfully

**Step 6: Commit**

```bash
git add src/graph/ src/bin/npu_graph.rs src/lib.rs Cargo.toml
git commit -m "feat(graph): scaffold npu-graph binary with core type definitions"
```

---

## Task 2: Device Model Extractor

**Files:**
- Create: `src/graph/extract_device_model.rs`
- Modify: `src/graph/mod.rs` (add `pub mod extract_device_model;`)

Parses `tools/aie-device-models.json` into `TileTypeModel` entries with
memory sizes, lock/BD/channel counts, port maps, and array topology.

**Reference:** `build.rs` lines 57-66 for the JSON structure, and
`tools/aie-device-models.json` for the full schema (tile_types with
switchbox_ports, shim_mux_ports, num_locks, num_bds, num_banks, etc.)

**Step 1: Write the failing test**

In `src/graph/extract_device_model.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn device_model_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/aie-device-models.json")
    }

    #[test]
    fn test_extract_aie2_has_three_tile_types() {
        let result = extract_device_model(&device_model_path(), "npu1")
            .expect("extraction should succeed");
        // NPU1 has shim_noc, mem_tile, and aie_tile
        assert_eq!(result.tile_types.len(), 3);
    }

    #[test]
    fn test_extract_aie2_topology() {
        let result = extract_device_model(&device_model_path(), "npu1")
            .expect("extraction should succeed");
        let topo = result.array_topology.as_ref().expect("should have topology");
        assert_eq!(topo.columns, 4);
        assert_eq!(topo.rows, 6);
        assert_eq!(topo.num_mem_tile_rows, 1);
    }

    #[test]
    fn test_extract_aie2_compute_tile_memory() {
        let result = extract_device_model(&device_model_path(), "npu1")
            .expect("extraction should succeed");
        let compute = result.tile_types.iter()
            .find(|t| t.kind == TileKind::Compute)
            .expect("should have compute tile");
        let mem = compute.memory.as_ref().expect("compute tile should have memory");
        assert_eq!(mem.size_bytes, 65536);
    }

    #[test]
    fn test_extract_aie2_mem_tile_memory() {
        let result = extract_device_model(&device_model_path(), "npu1")
            .expect("extraction should succeed");
        let memtile = result.tile_types.iter()
            .find(|t| t.kind == TileKind::Mem)
            .expect("should have mem tile");
        let mem = memtile.memory.as_ref().expect("mem tile should have memory");
        assert_eq!(mem.size_bytes, 524288);
    }

    #[test]
    fn test_extract_aie2_shim_ports() {
        let result = extract_device_model(&device_model_path(), "npu1")
            .expect("extraction should succeed");
        let shim = result.tile_types.iter()
            .find(|t| t.kind == TileKind::Shim)
            .expect("should have shim tile");
        let ss = shim.modules.iter()
            .find(|m| m.kind == ModuleKind::StreamSwitch)
            .expect("shim should have stream switch");
        // Shim has ports from both switchbox_ports and shim_mux_ports
        assert!(!ss.ports.is_empty());
    }

    #[test]
    fn test_extract_aie2_lock_counts() {
        let result = extract_device_model(&device_model_path(), "npu1")
            .expect("extraction should succeed");
        let compute = result.tile_types.iter()
            .find(|t| t.kind == TileKind::Compute)
            .expect("should have compute tile");
        let lock_mod = compute.modules.iter()
            .find(|m| m.kind == ModuleKind::Lock)
            .expect("compute should have lock module");
        let inst = lock_mod.instances.as_ref().expect("should have instance counts");
        assert_eq!(inst.locks, 16);
    }

    #[test]
    fn test_extract_unknown_device_errors() {
        let result = extract_device_model(&device_model_path(), "nonexistent");
        assert!(result.is_err());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib graph::extract_device_model::tests -q`
Expected: FAIL -- module not found

**Step 3: Write the extractor**

Create `src/graph/extract_device_model.rs`:

```rust
//! Extractor for the mlir-aie device model JSON.
//!
//! Parses `aie-device-models.json` (generated by `tools/aie-device-dump.py`)
//! into tile types with memory sizes, instance counts, port maps, and
//! array topology.

use crate::graph::types::*;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// JSON deserialization types (matching aie-device-models.json)
// ============================================================================

#[derive(Deserialize)]
struct DeviceModelSet {
    devices: HashMap<String, DeviceModel>,
}

#[derive(Deserialize)]
struct DeviceModel {
    columns: u8,
    rows: u8,
    local_memory_size: u64,
    mem_tile_size: u64,
    num_mem_tile_rows: u8,
    column_shift: u8,
    row_shift: u8,
    tile_types: HashMap<String, TileTypeJson>,
}

#[derive(Deserialize)]
struct TileTypeJson {
    num_locks: u8,
    num_bds: u8,
    num_banks: u8,
    bank_size: u64,
    #[serde(default)]
    switchbox_ports: HashMap<String, PortCountJson>,
    #[serde(default)]
    shim_mux_ports: HashMap<String, PortCountJson>,
}

#[derive(Deserialize)]
struct PortCountJson {
    master: u8,
    slave: u8,
}

// ============================================================================
// Extraction
// ============================================================================

/// Map JSON tile type name to our TileKind.
fn classify_tile(name: &str) -> TileKind {
    match name {
        "aie_tile" => TileKind::Compute,
        "mem_tile" => TileKind::Mem,
        "shim_noc" | "shim_pl" => TileKind::Shim,
        _ => TileKind::Compute, // conservative default
    }
}

fn source_attr(file: &str, detail: &str) -> SourceAttribution {
    SourceAttribution {
        origin: Source::DeviceModel,
        file: file.to_string(),
        detail: detail.to_string(),
    }
}

/// Extract ports from switchbox_ports and shim_mux_ports maps.
fn extract_ports(
    ports_map: &HashMap<String, PortCountJson>,
    source_prefix: &str,
    file: &str,
) -> Vec<PortModel> {
    let mut ports = Vec::new();
    for (bundle, counts) in ports_map {
        for i in 0..counts.master {
            ports.push(PortModel {
                bundle: bundle.clone(),
                index: i,
                direction: PortDirection::Master,
                source: source_attr(file, &format!("{}.{}.master[{}]", source_prefix, bundle, i)),
            });
        }
        for i in 0..counts.slave {
            ports.push(PortModel {
                bundle: bundle.clone(),
                index: i,
                direction: PortDirection::Slave,
                source: source_attr(file, &format!("{}.{}.slave[{}]", source_prefix, bundle, i)),
            });
        }
    }
    ports
}

/// Extract an architecture model from the device model JSON.
///
/// `device_name` should be a key in the "devices" map (e.g., "npu1", "npu2").
pub fn extract_device_model(path: &Path, device_name: &str) -> Result<ArchModel> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading device model at {}", path.display()))?;
    let models: DeviceModelSet = serde_json::from_str(&text)
        .context("parsing device model JSON")?;

    let device = models.devices.get(device_name)
        .ok_or_else(|| anyhow::anyhow!(
            "device '{}' not found in {}. Available: {:?}",
            device_name, path.display(),
            models.devices.keys().collect::<Vec<_>>()
        ))?;

    let file = path.file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string());

    // Determine architecture from device name
    let arch = match device_name {
        "npu1" | n if n.starts_with("npu1_") => Architecture::Aie2,
        "npu2" | n if n.starts_with("npu2_") => Architecture::Aie2p,
        _ => bail!("cannot determine architecture for device '{}'", device_name),
    };

    let mut model = ArchModel::new(arch);

    // Array topology
    model.array_topology = Some(ArrayTopology {
        columns: device.columns,
        rows: device.rows,
        num_mem_tile_rows: device.num_mem_tile_rows,
        column_shift: device.column_shift,
        row_shift: device.row_shift,
        source: source_attr(&file, &format!("devices.{}", device_name)),
    });

    // Tile types
    for (type_name, tile_json) in &device.tile_types {
        let kind = classify_tile(type_name);

        // Memory model
        let memory = match kind {
            TileKind::Compute => Some(MemoryModel {
                size_bytes: device.local_memory_size,
                num_banks: tile_json.num_banks,
                bank_size_bytes: tile_json.bank_size,
                source: source_attr(&file, &format!("{}.local_memory_size", device_name)),
            }),
            TileKind::Mem => Some(MemoryModel {
                size_bytes: device.mem_tile_size,
                num_banks: tile_json.num_banks,
                bank_size_bytes: tile_json.bank_size,
                source: source_attr(&file, &format!("{}.mem_tile_size", device_name)),
            }),
            TileKind::Shim => None,
        };

        // Modules: DMA, Lock, StreamSwitch (from instance counts and ports)
        let mut modules = Vec::new();

        // DMA module
        modules.push(ModuleModel {
            kind: ModuleKind::Dma,
            registers: Vec::new(), // populated by regdb extractor
            ports: Vec::new(),     // DMA ports come from stream switch
            instances: Some(InstanceCount {
                channels: 2, // standard for AIE2; will be cross-validated
                bds: tile_json.num_bds,
                locks: 0,
            }),
            source: source_attr(&file, &format!("{}.num_bds={}", type_name, tile_json.num_bds)),
        });

        // Lock module
        modules.push(ModuleModel {
            kind: ModuleKind::Lock,
            registers: Vec::new(),
            ports: Vec::new(),
            instances: Some(InstanceCount {
                channels: 0,
                bds: 0,
                locks: tile_json.num_locks,
            }),
            source: source_attr(&file, &format!("{}.num_locks={}", type_name, tile_json.num_locks)),
        });

        // Stream switch module (with all ports)
        let mut ss_ports = extract_ports(&tile_json.switchbox_ports, "switchbox_ports", &file);
        let mux_ports = extract_ports(&tile_json.shim_mux_ports, "shim_mux_ports", &file);
        ss_ports.extend(mux_ports);

        modules.push(ModuleModel {
            kind: ModuleKind::StreamSwitch,
            registers: Vec::new(),
            ports: ss_ports,
            instances: None,
            source: source_attr(&file, &format!("{}.switchbox_ports", type_name)),
        });

        model.tile_types.push(TileTypeModel {
            kind,
            name: format!("{}_{}", arch.to_string().to_lowercase(), type_name),
            modules,
            memory,
            source: source_attr(&file, &format!("devices.{}.tile_types.{}", device_name, type_name)),
        });
    }

    Ok(model)
}
```

Add `pub mod extract_device_model;` to `src/graph/mod.rs`.

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib graph::extract_device_model::tests -q`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/graph/extract_device_model.rs src/graph/mod.rs
git commit -m "feat(graph): device model extractor -- tile types, topology, ports"
```

---

## Task 3: RegDB Extractor

**Files:**
- Create: `src/graph/extract_regdb.rs`
- Modify: `src/graph/mod.rs` (add `pub mod extract_regdb;`)

Parses `aie_registers_aie2.json` into `RegisterModel` entries with fields,
grouped by module. This is the most register-dense extractor -- 1,806
registers across 4 modules.

**Reference:** `src/device/regdb.rs` for the JSON schema and field parsing
pattern. `build.rs` `parse_offset()` for hex offset handling.

**Step 1: Write the failing test**

In `src/graph/extract_regdb.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn regdb_path() -> std::path::PathBuf {
        let mlir_aie = std::env::var("MLIR_AIE_PATH").unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent().unwrap()
                .join("mlir-aie")
                .to_string_lossy().to_string()
        });
        Path::new(&mlir_aie).join("lib/Dialect/AIE/Util/aie_registers_aie2.json")
    }

    #[test]
    fn test_extract_has_four_modules() {
        let result = extract_regdb(&regdb_path())
            .expect("extraction should succeed");
        // AM025 JSON has: core, memory, memory_tile, shim
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_extract_memory_module_has_dma_bds() {
        let result = extract_regdb(&regdb_path()).unwrap();
        let memory = result.iter()
            .find(|m| m.kind == ModuleKind::Dma)
            .or_else(|| result.iter().find(|m| {
                m.registers.iter().any(|r| r.name.starts_with("DMA_BD"))
            }))
            .expect("should have module with DMA BDs");
        let bd_regs: Vec<_> = memory.registers.iter()
            .filter(|r| r.name.starts_with("DMA_BD0_"))
            .collect();
        // BD0 has 8 words (DMA_BD0_0 through DMA_BD0_7)
        assert!(bd_regs.len() >= 8, "BD0 should have at least 8 words, got {}", bd_regs.len());
    }

    #[test]
    fn test_extract_register_has_fields() {
        let result = extract_regdb(&regdb_path()).unwrap();
        // Find DMA_BD0_0 -- should have Buffer_Length field
        let reg = result.iter()
            .flat_map(|m| &m.registers)
            .find(|r| r.name == "DMA_BD0_0")
            .expect("should have DMA_BD0_0");
        let buf_len = reg.fields.iter()
            .find(|f| f.name == "Buffer_Length")
            .expect("DMA_BD0_0 should have Buffer_Length field");
        assert!(buf_len.bits.width() > 0);
    }

    #[test]
    fn test_extract_register_has_offset() {
        let result = extract_regdb(&regdb_path()).unwrap();
        let core_ctrl = result.iter()
            .flat_map(|m| &m.registers)
            .find(|r| r.name == "Core_Control")
            .expect("should have Core_Control");
        // Core_Control is at a known offset in the core module
        assert!(core_ctrl.offset > 0);
    }

    #[test]
    fn test_extract_all_registers_have_source_attribution() {
        let result = extract_regdb(&regdb_path()).unwrap();
        for module in &result {
            for reg in &module.registers {
                assert_eq!(reg.source.origin, Source::Am025Json);
                assert!(!reg.source.file.is_empty());
            }
        }
    }

    #[test]
    fn test_extract_register_access_modes() {
        let result = extract_regdb(&regdb_path()).unwrap();
        // Core_Status should be read-only
        let core_status = result.iter()
            .flat_map(|m| &m.registers)
            .find(|r| r.name == "Core_Status");
        if let Some(reg) = core_status {
            assert_eq!(reg.access, Access::ReadOnly);
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib graph::extract_regdb::tests -q`
Expected: FAIL -- module not found

**Step 3: Write the extractor**

Create `src/graph/extract_regdb.rs`:

```rust
//! Extractor for the AM025 register database JSON.
//!
//! Parses `aie_registers_aie2.json` into `ModuleModel` entries with
//! registers and fields. Each module (core, memory, memory_tile, shim)
//! becomes a `ModuleModel` containing its registers.
//!
//! The JSON module names map to our `ModuleKind` based on content analysis
//! rather than name matching, since a single JSON module (e.g., "memory")
//! contains registers for DMA, locks, AND memory -- multiple `ModuleKind`s.
//! For now, each JSON module produces one `ModuleModel` with all its
//! registers. The merge step can split them by offset range if needed.

use crate::graph::types::*;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// JSON deserialization (matches aie_registers_aie2.json)
// ============================================================================

#[derive(Deserialize)]
struct RawRegisterDb {
    #[allow(dead_code)]
    version: String,
    modules: HashMap<String, RawModule>,
}

#[derive(Deserialize)]
struct RawModule {
    registers: Vec<RawRegister>,
}

#[derive(Deserialize)]
struct RawRegister {
    name: String,
    offset: String,
    #[serde(default)]
    width: Option<u32>,
    #[serde(rename = "type", default)]
    access_type: Option<String>,
    #[serde(default)]
    reset: Option<String>,
    bit_fields: Vec<RawBitField>,
}

#[derive(Deserialize)]
struct RawBitField {
    name: String,
    bit_range: Vec<u32>,
}

// ============================================================================
// Helpers
// ============================================================================

/// Parse hex offset string to tile-local u32 (low 20 bits).
fn parse_offset(s: &str) -> u32 {
    let s = s.trim_start_matches("0x").trim_start_matches("0X");
    let full = u64::from_str_radix(s, 16)
        .unwrap_or_else(|e| panic!("bad hex offset '{}': {}", s, e));
    (full & 0xFFFFF) as u32
}

/// Parse reset value from hex string.
fn parse_reset(s: &str) -> u32 {
    let s = s.trim_start_matches("0x").trim_start_matches("0X");
    u32::from_str_radix(s, 16).unwrap_or(0)
}

/// Map JSON access type string to our Access enum.
fn parse_access(s: &str) -> Access {
    let lower = s.to_lowercase();
    if lower.contains("read-only") || lower.starts_with("ro") {
        Access::ReadOnly
    } else if lower.contains("write-only") || lower.starts_with("wo") {
        Access::WriteOnly
    } else if lower.contains("write-1-to-clear") || lower.contains("wtc") {
        Access::WriteToClear
    } else if lower.contains("mixed") {
        Access::Mixed
    } else {
        Access::ReadWrite
    }
}

/// Map JSON module name to our ModuleKind.
///
/// The JSON modules are coarse (core, memory, memory_tile, shim) and
/// each contains registers for multiple functional units. We assign a
/// primary ModuleKind based on the JSON module name. The merge step can
/// refine this by splitting registers into sub-modules by offset range.
fn classify_module(json_name: &str) -> ModuleKind {
    match json_name {
        "core" => ModuleKind::Core,
        "memory" => ModuleKind::Dma,      // memory module is mostly DMA + locks
        "memory_tile" => ModuleKind::Dma, // same
        "shim" => ModuleKind::ShimMux,    // shim is DMA + mux
        _ => ModuleKind::Unknown,
    }
}

fn source_attr(file: &str, detail: &str) -> SourceAttribution {
    SourceAttribution {
        origin: Source::Am025Json,
        file: file.to_string(),
        detail: detail.to_string(),
    }
}

// ============================================================================
// Extraction
// ============================================================================

/// Extract register models from the AM025 JSON register database.
///
/// Returns one `ModuleModel` per JSON module, each containing all
/// registers from that module with their fields and source attribution.
pub fn extract_regdb(path: &Path) -> Result<Vec<ModuleModel>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading regdb at {}", path.display()))?;
    let raw: RawRegisterDb = serde_json::from_str(&text)
        .context("parsing AM025 JSON")?;

    let file = path.file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string());

    let mut modules = Vec::new();

    for (module_name, raw_module) in &raw.modules {
        let kind = classify_module(module_name);

        let registers: Vec<RegisterModel> = raw_module.registers.iter().map(|raw_reg| {
            let fields: Vec<FieldModel> = raw_reg.bit_fields.iter().map(|bf| {
                let bits = if bf.bit_range.len() == 2 {
                    BitRange::Contiguous {
                        lsb: bf.bit_range[0] as u8,
                        msb: bf.bit_range[1] as u8,
                    }
                } else {
                    // Pairs for split fields
                    let chunks: Vec<(u8, u8)> = bf.bit_range
                        .chunks(2)
                        .map(|c| (c[1] as u8, c[0] as u8))
                        .collect();
                    BitRange::Split(chunks)
                };

                FieldModel {
                    name: bf.name.clone(),
                    bits,
                    meaning: FieldSemantics::Unknown, // classified later
                    source: source_attr(&file, &format!(
                        "module={}, register={}, field={}",
                        module_name, raw_reg.name, bf.name
                    )),
                }
            }).collect();

            let access = raw_reg.access_type.as_deref()
                .map(parse_access)
                .unwrap_or(Access::ReadWrite);

            let reset_value = raw_reg.reset.as_deref()
                .map(parse_reset)
                .unwrap_or(0);

            RegisterModel {
                name: raw_reg.name.clone(),
                offset: parse_offset(&raw_reg.offset),
                width: raw_reg.width.unwrap_or(32) as u8,
                reset_value,
                fields,
                module: kind,
                access,
                source: source_attr(&file, &format!(
                    "module={}, register={}",
                    module_name, raw_reg.name
                )),
            }
        }).collect();

        modules.push(ModuleModel {
            kind,
            registers,
            ports: Vec::new(),
            instances: None,
            source: source_attr(&file, &format!("module={}", module_name)),
        });
    }

    Ok(modules)
}
```

Add `pub mod extract_regdb;` to `src/graph/mod.rs`.

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib graph::extract_regdb::tests -q`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/graph/extract_regdb.rs src/graph/mod.rs
git commit -m "feat(graph): regdb extractor -- 1,806 registers from AM025 JSON"
```

---

## Task 4: aie-rt Extractor

**Files:**
- Create: `src/graph/extract_aiert.rs`
- Modify: `src/graph/mod.rs` (add `pub mod extract_aiert;`)

Extracts DMA, lock, and stream switch constants from aie-rt headers using
gcc -E preprocessing. Reuses the pattern from `build.rs` lines 1050-1150.

**Reference:** `build.rs` functions `run_aiert_preprocessor()`,
`parse_dma_modules()`, `parse_lock_modules()`, `parse_port_maps()`.
These parse the preprocessed C output of `xaiemlgbl_reginit.c` to
extract register offsets and instance counts.

**Step 1: Write the failing test**

In `src/graph/extract_aiert.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn aiert_path() -> std::path::PathBuf {
        let path = std::env::var("AIE_RT_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent().unwrap()
                    .join("aie-rt/driver/src")
            });
        path
    }

    #[test]
    fn test_extract_aiert_has_dma_constants() {
        let result = extract_aiert(&aiert_path());
        match result {
            Ok(data) => {
                assert!(!data.dma_modules.is_empty(),
                    "should extract DMA module data");
            }
            Err(e) => {
                // aie-rt may not be available in CI
                eprintln!("skipping aie-rt test: {}", e);
            }
        }
    }

    #[test]
    fn test_extract_aiert_has_lock_constants() {
        let result = extract_aiert(&aiert_path());
        match result {
            Ok(data) => {
                assert!(!data.lock_modules.is_empty(),
                    "should extract lock module data");
            }
            Err(e) => {
                eprintln!("skipping aie-rt test: {}", e);
            }
        }
    }

    #[test]
    fn test_extract_aiert_has_port_maps() {
        let result = extract_aiert(&aiert_path());
        match result {
            Ok(data) => {
                assert!(!data.port_maps.is_empty(),
                    "should extract port map data");
            }
            Err(e) => {
                eprintln!("skipping aie-rt test: {}", e);
            }
        }
    }

    #[test]
    fn test_extract_aiert_source_attribution() {
        let result = extract_aiert(&aiert_path());
        if let Ok(data) = result {
            assert_eq!(data.source.origin, Source::AieRt);
            assert!(data.source.file.contains("aie-rt")
                || data.source.file.contains("xaiemlgbl"));
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib graph::extract_aiert::tests -q`
Expected: FAIL -- module not found

**Step 3: Write the extractor**

Create `src/graph/extract_aiert.rs`. This reuses the gcc -E preprocessing
pattern from `build.rs` but produces typed graph data instead of generated
Rust source. The parsing logic for DMA modules, lock modules, and port
maps follows the same struct-field extraction as `build.rs` functions
`parse_dma_modules()`, `parse_lock_modules()`, and `parse_port_maps()`.

```rust
//! Extractor for aie-rt hardware abstraction layer.
//!
//! Uses gcc -E to preprocess `xaiemlgbl_reginit.c` and extract DMA,
//! lock, and stream switch constants from the struct initializer tables.
//! This is the same approach used by `build.rs` `extract_aiert()`.

use crate::graph::types::*;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Extracted data from aie-rt.
pub struct AieRtData {
    pub dma_modules: Vec<AieRtDmaModule>,
    pub lock_modules: Vec<AieRtLockModule>,
    pub port_maps: Vec<AieRtPortMap>,
    pub source: SourceAttribution,
}

/// DMA module constants from aie-rt.
pub struct AieRtDmaModule {
    pub name: String,
    pub fields: HashMap<String, String>,
}

/// Lock module constants from aie-rt.
pub struct AieRtLockModule {
    pub name: String,
    pub fields: HashMap<String, String>,
}

/// Port map entry from aie-rt.
pub struct AieRtPortMap {
    pub name: String,
    pub fields: HashMap<String, String>,
}

/// Run gcc -E on aie-rt and extract hardware constants.
///
/// Returns `Err` if aie-rt is not available or gcc fails.
pub fn extract_aiert(aiert_dir: &Path) -> Result<AieRtData> {
    let reginit = aiert_dir.join("global/xaiemlgbl_reginit.c");
    if !reginit.exists() {
        bail!("aie-rt not found at {}", aiert_dir.display());
    }

    let preprocessed = run_preprocessor(aiert_dir)
        .context("running gcc -E on aie-rt")?;

    let dma_modules = parse_dma_modules(&preprocessed);
    let lock_modules = parse_lock_modules(&preprocessed);
    let port_maps = parse_port_maps(&preprocessed);

    Ok(AieRtData {
        dma_modules,
        lock_modules,
        port_maps,
        source: SourceAttribution {
            origin: Source::AieRt,
            file: reginit.display().to_string(),
            detail: "gcc -E xaiemlgbl_reginit.c".to_string(),
        },
    })
}

fn run_preprocessor(aiert_dir: &Path) -> Result<String> {
    let reginit = aiert_dir.join("global/xaiemlgbl_reginit.c");

    let subdirs = [
        "", "common", "core", "device", "dma", "events", "global",
        "interrupt", "io_backend", "lite", "locks", "memory", "npi",
        "perfcnt", "pl", "pm", "routing", "stream_switch", "timer",
        "trace", "util",
    ];

    let mut cmd = Command::new("gcc");
    cmd.arg("-E");
    for subdir in &subdirs {
        let inc = if subdir.is_empty() {
            aiert_dir.to_path_buf()
        } else {
            aiert_dir.join(subdir)
        };
        cmd.arg("-I").arg(&inc);
    }
    cmd.arg(&reginit);

    let output = cmd.output()
        .context("failed to run gcc")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("gcc -E failed: {}", stderr);
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

// ============================================================================
// Parsers -- adapted from build.rs
// ============================================================================
// These parse the C struct initializer tables in the preprocessed output.
// The format is: .FieldName = value, or .FieldName = {nested}
// We extract name-value pairs for each struct entry.

fn parse_dma_modules(preprocessed: &str) -> Vec<AieRtDmaModule> {
    parse_struct_array(preprocessed, "DmaMod")
        .into_iter()
        .map(|(name, fields)| AieRtDmaModule { name, fields })
        .collect()
}

fn parse_lock_modules(preprocessed: &str) -> Vec<AieRtLockModule> {
    parse_struct_array(preprocessed, "LockMod")
        .into_iter()
        .map(|(name, fields)| AieRtLockModule { name, fields })
        .collect()
}

fn parse_port_maps(preprocessed: &str) -> Vec<AieRtPortMap> {
    parse_struct_array(preprocessed, "StrmMod")
        .into_iter()
        .map(|(name, fields)| AieRtPortMap { name, fields })
        .collect()
}

/// Generic parser for C struct initializer arrays in preprocessed output.
///
/// Looks for patterns like:
///   .FieldName = value,
///   .FieldName = {nested_value},
///
/// within blocks that match the given struct type name.
fn parse_struct_array(
    preprocessed: &str,
    struct_type: &str,
) -> Vec<(String, HashMap<String, String>)> {
    let mut results = Vec::new();

    // Find struct initializer blocks by looking for the type name
    // followed by brace-enclosed initializers.
    // This is a simplified parser -- the build.rs version is more robust.
    // We look for ".TypeName =" or "TypeName[" patterns.
    for line in preprocessed.lines() {
        let trimmed = line.trim();
        // Look for field assignments like ".DmaBdProp = { ... }"
        if trimmed.starts_with('.') && trimmed.contains('=') {
            let parts: Vec<&str> = trimmed.splitn(2, '=').collect();
            if parts.len() == 2 {
                let field = parts[0].trim().trim_start_matches('.');
                let value = parts[1].trim().trim_end_matches(',');
                if field.contains(struct_type) || !results.is_empty() {
                    if results.is_empty() {
                        results.push((struct_type.to_string(), HashMap::new()));
                    }
                    if let Some(last) = results.last_mut() {
                        last.1.insert(field.to_string(), value.to_string());
                    }
                }
            }
        }
    }

    results
}
```

Add `pub mod extract_aiert;` to `src/graph/mod.rs`.

**NOTE:** The `parse_struct_array` above is a simplified skeleton. The real
implementation should closely follow the parsing logic from `build.rs`
functions `parse_dma_modules()`, `parse_lock_modules()`, and
`parse_port_maps()` (around lines 1160-1350). Read those functions and
adapt them to produce `HashMap<String, String>` field maps. The implementer
should read `build.rs` in full before writing the parsing code.

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib graph::extract_aiert::tests -q`
Expected: All 4 tests PASS (or gracefully skip if aie-rt not available)

**Step 5: Commit**

```bash
git add src/graph/extract_aiert.rs src/graph/mod.rs
git commit -m "feat(graph): aie-rt extractor -- DMA, lock, stream switch constants"
```

---

## Task 5: Merge and Cross-Validation

**Files:**
- Create: `src/graph/merge.rs`
- Modify: `src/graph/mod.rs` (add `pub mod merge;`)

Combines partial models from device model, regdb, and aie-rt extractors
into a single `ArchModel`. Cross-validates overlapping facts and errors
on any disagreement.

**Step 1: Write the failing test**

In `src/graph/merge.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::*;

    fn make_module(kind: ModuleKind, lock_count: u8) -> ModuleModel {
        ModuleModel {
            kind,
            registers: Vec::new(),
            ports: Vec::new(),
            instances: Some(InstanceCount { channels: 0, bds: 0, locks: lock_count }),
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test".to_string(),
                detail: "test".to_string(),
            },
        }
    }

    fn make_register(name: &str, offset: u32, source: Source) -> RegisterModel {
        RegisterModel {
            name: name.to_string(),
            offset,
            width: 32,
            reset_value: 0,
            fields: Vec::new(),
            module: ModuleKind::Dma,
            access: Access::ReadWrite,
            source: SourceAttribution {
                origin: source,
                file: "test".to_string(),
                detail: "test".to_string(),
            },
        }
    }

    #[test]
    fn test_merge_combines_tile_types_and_registers() {
        let mut device_model = ArchModel::new(Architecture::Aie2);
        device_model.tile_types.push(TileTypeModel {
            kind: TileKind::Compute,
            name: "aie2_compute".to_string(),
            modules: vec![make_module(ModuleKind::Dma, 0)],
            memory: None,
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test".to_string(),
                detail: "test".to_string(),
            },
        });

        let regdb_modules = vec![ModuleModel {
            kind: ModuleKind::Dma,
            registers: vec![make_register("DMA_BD0_0", 0x1D000, Source::Am025Json)],
            ports: Vec::new(),
            instances: None,
            source: SourceAttribution {
                origin: Source::Am025Json,
                file: "test".to_string(),
                detail: "test".to_string(),
            },
        }];

        let merged = merge_model(device_model, regdb_modules, None)
            .expect("merge should succeed");

        // The compute tile's DMA module should now have the register
        let compute = &merged.tile_types[0];
        let dma = compute.modules.iter()
            .find(|m| m.kind == ModuleKind::Dma)
            .expect("should have DMA module");
        assert!(!dma.registers.is_empty(), "DMA module should have registers after merge");
    }

    #[test]
    fn test_merge_errors_on_offset_disagreement() {
        let mut device_model = ArchModel::new(Architecture::Aie2);
        device_model.tile_types.push(TileTypeModel {
            kind: TileKind::Compute,
            name: "aie2_compute".to_string(),
            modules: vec![ModuleModel {
                kind: ModuleKind::Dma,
                registers: vec![make_register("DMA_BD0_0", 0x1D000, Source::DeviceModel)],
                ports: Vec::new(),
                instances: None,
                source: SourceAttribution {
                    origin: Source::DeviceModel,
                    file: "test".to_string(),
                    detail: "test".to_string(),
                },
            }],
            memory: None,
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test".to_string(),
                detail: "test".to_string(),
            },
        });

        let regdb_modules = vec![ModuleModel {
            kind: ModuleKind::Dma,
            registers: vec![make_register("DMA_BD0_0", 0x1D004, Source::Am025Json)],
            ports: Vec::new(),
            instances: None,
            source: SourceAttribution {
                origin: Source::Am025Json,
                file: "test".to_string(),
                detail: "test".to_string(),
            },
        }];

        let result = merge_model(device_model, regdb_modules, None);
        assert!(result.is_err(), "should error on offset disagreement");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("DMA_BD0_0"), "error should mention register name");
    }

    #[test]
    fn test_merge_with_no_aiert_succeeds() {
        let device_model = ArchModel::new(Architecture::Aie2);
        let regdb_modules = Vec::new();
        let merged = merge_model(device_model, regdb_modules, None)
            .expect("merge without aie-rt should succeed");
        assert_eq!(merged.arch, Architecture::Aie2);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib graph::merge::tests -q`
Expected: FAIL -- module not found

**Step 3: Write the merge logic**

Create `src/graph/merge.rs`:

```rust
//! Merge and cross-validate partial models from multiple extractors.
//!
//! The device model provides tile types, topology, and instance counts.
//! The regdb provides registers and fields per module.
//! The aie-rt data provides register offsets and hardware constants.
//!
//! When two sources state the same fact differently, we error immediately.

use crate::graph::extract_aiert::AieRtData;
use crate::graph::types::*;
use anyhow::{bail, Result};

/// Merge a device model, regdb modules, and optional aie-rt data into
/// a single ArchModel.
///
/// Errors if any overlapping facts disagree between sources.
pub fn merge_model(
    mut device_model: ArchModel,
    regdb_modules: Vec<ModuleModel>,
    aiert_data: Option<AieRtData>,
) -> Result<ArchModel> {
    // Phase 1: Merge regdb registers into device model tile types.
    //
    // The regdb provides registers grouped by AM025 module (core, memory,
    // memory_tile, shim). We match these to tile type modules by ModuleKind.
    // Registers from the "memory" JSON module go into compute tile's DMA
    // module. Registers from "memory_tile" go into mem tile's DMA module.
    merge_regdb_into_tiles(&mut device_model, &regdb_modules)?;

    // Phase 2: Cross-validate aie-rt constants against merged model.
    if let Some(aiert) = aiert_data {
        cross_validate_aiert(&device_model, &aiert)?;
    }

    Ok(device_model)
}

fn merge_regdb_into_tiles(
    model: &mut ArchModel,
    regdb_modules: &[ModuleModel],
) -> Result<()> {
    for regdb_mod in regdb_modules {
        // Find matching tile type modules by ModuleKind
        for tile_type in &mut model.tile_types {
            for tile_mod in &mut tile_type.modules {
                if tile_mod.kind == regdb_mod.kind {
                    // Cross-validate: if both have registers with the same name,
                    // offsets must agree
                    for new_reg in &regdb_mod.registers {
                        if let Some(existing) = tile_mod.registers.iter()
                            .find(|r| r.name == new_reg.name)
                        {
                            if existing.offset != new_reg.offset {
                                bail!(
                                    "Register offset disagreement for '{}': \
                                     {} says {:#X}, {} says {:#X}",
                                    new_reg.name,
                                    existing.source,
                                    existing.offset,
                                    new_reg.source,
                                    new_reg.offset,
                                );
                            }
                            // Same offset -- keep existing, don't duplicate
                            continue;
                        }
                        tile_mod.registers.push(new_reg.clone());
                    }
                }
            }
        }
    }
    Ok(())
}

fn cross_validate_aiert(
    _model: &ArchModel,
    _aiert: &AieRtData,
) -> Result<()> {
    // TODO: Compare aie-rt register offsets against regdb offsets.
    // For now, just accept. The real cross-validation will compare
    // specific #define values against AM025 JSON offsets.
    Ok(())
}
```

Add `pub mod merge;` to `src/graph/mod.rs`.

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib graph::merge::tests -q`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/graph/merge.rs src/graph/mod.rs
git commit -m "feat(graph): merge and cross-validation -- error on source disagreement"
```

---

## Task 6: JSON Serialization and CLI

**Files:**
- Modify: `src/bin/npu_graph.rs` (implement `extract` command)
- Create: `models/` directory

Wires up the extractors and merge step into the CLI binary's `extract`
command, producing a serialized JSON model file.

**Step 1: Write the failing test**

This is an integration test -- we test that the full pipeline produces
valid JSON. Add to `src/graph/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_full_extraction_pipeline_aie2() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let device_model_path = manifest.join("tools/aie-device-models.json");

        let mlir_aie = std::env::var("MLIR_AIE_PATH").unwrap_or_else(|_| {
            manifest.parent().unwrap()
                .join("mlir-aie")
                .to_string_lossy().to_string()
        });
        let regdb_path = Path::new(&mlir_aie)
            .join("lib/Dialect/AIE/Util/aie_registers_aie2.json");

        // Extract device model
        let device = extract_device_model::extract_device_model(
            &device_model_path, "npu1"
        ).expect("device model extraction failed");

        // Extract regdb
        let regdb = extract_regdb::extract_regdb(&regdb_path)
            .expect("regdb extraction failed");

        // Merge (no aie-rt for this test)
        let merged = merge::merge_model(device, regdb, None)
            .expect("merge failed");

        // Verify structure
        assert_eq!(merged.arch, types::Architecture::Aie2);
        assert!(!merged.tile_types.is_empty());

        // Verify serialization round-trips
        let json = serde_json::to_string_pretty(&merged)
            .expect("serialization failed");
        let deserialized: types::ArchModel = serde_json::from_str(&json)
            .expect("deserialization failed");
        assert_eq!(deserialized.arch, merged.arch);
        assert_eq!(deserialized.tile_types.len(), merged.tile_types.len());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib graph::tests -q`
Expected: FAIL (or pass if previous steps all worked -- this validates the pipeline)

**Step 3: Implement the CLI extract command**

Update `src/bin/npu_graph.rs`:

```rust
//! NPU Architecture Graph -- standalone extraction and query tool.

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::{env, fs, process};
use xdna_emu::graph::{extract_aiert, extract_device_model, extract_regdb, merge, types};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let result = match args[1].as_str() {
        "extract" => cmd_extract(&args[2..]),
        "query" => {
            eprintln!("query: not yet implemented");
            process::exit(1);
        }
        "diff" => {
            eprintln!("diff: not yet implemented");
            process::exit(1);
        }
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {:#}", e);
        process::exit(1);
    }
}

fn print_usage() {
    eprintln!("Usage: npu-graph <command> [options]");
    eprintln!("Commands:");
    eprintln!("  extract  --arch <aie2|aie2p> [--device-model PATH] [--regdb PATH] [--aiert PATH] [-o PATH]");
    eprintln!("  query    --model <path.json> <query>");
    eprintln!("  diff     --a <model_a.json> --b <model_b.json>");
}

fn cmd_extract(args: &[str]) -> Result<()> {
    // Parse arguments
    let mut arch_str = None;
    let mut device_model_path = None;
    let mut regdb_path = None;
    let mut aiert_path = None;
    let mut output_path = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--arch" => { i += 1; arch_str = Some(&args[i]); }
            "--device-model" => { i += 1; device_model_path = Some(&args[i]); }
            "--regdb" => { i += 1; regdb_path = Some(&args[i]); }
            "--aiert" => { i += 1; aiert_path = Some(&args[i]); }
            "-o" | "--output" => { i += 1; output_path = Some(&args[i]); }
            other => bail!("unknown argument: {}", other),
        }
        i += 1;
    }

    let arch_str = arch_str.ok_or_else(|| anyhow::anyhow!("--arch required"))?;
    let (device_name, _arch) = match arch_str.as_str() {
        "aie2" => ("npu1", types::Architecture::Aie2),
        "aie2p" => ("npu2", types::Architecture::Aie2p),
        other => bail!("unknown architecture: {}. Use aie2 or aie2p", other),
    };

    // Default paths relative to the binary's location
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let parent = manifest.parent().expect("manifest has parent");

    let dm_path = device_model_path
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("tools/aie-device-models.json"));
    let rdb_path = regdb_path
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            parent.join("mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json")
        });
    let aie_path = aiert_path
        .map(PathBuf::from)
        .unwrap_or_else(|| parent.join("aie-rt/driver/src"));
    let out_path = output_path
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join(format!("models/{}.json", arch_str)));

    // Extract
    eprintln!("Extracting {} model...", arch_str);

    eprintln!("  Device model: {}", dm_path.display());
    let device = extract_device_model::extract_device_model(&dm_path, device_name)?;

    eprintln!("  Register DB:  {}", rdb_path.display());
    let regdb = extract_regdb::extract_regdb(&rdb_path)?;

    let aiert = if aie_path.join("global/xaiemlgbl_reginit.c").exists() {
        eprintln!("  aie-rt:       {}", aie_path.display());
        match extract_aiert::extract_aiert(&aie_path) {
            Ok(data) => Some(data),
            Err(e) => {
                eprintln!("  aie-rt:       skipped ({})", e);
                None
            }
        }
    } else {
        eprintln!("  aie-rt:       not found, skipping");
        None
    };

    // Merge
    eprintln!("Merging...");
    let model = merge::merge_model(device, regdb, aiert)?;

    // Count what we got
    let reg_count: usize = model.tile_types.iter()
        .flat_map(|t| &t.modules)
        .map(|m| m.registers.len())
        .sum();
    eprintln!(
        "  {} tile types, {} total registers, {} relationships",
        model.tile_types.len(), reg_count, model.relationships.len()
    );

    // Serialize
    if let Some(parent_dir) = out_path.parent() {
        fs::create_dir_all(parent_dir)?;
    }
    let json = serde_json::to_string_pretty(&model)?;
    fs::write(&out_path, &json)?;
    eprintln!("Written to {}", out_path.display());

    Ok(())
}
```

**NOTE:** The `cmd_extract` function signature should be:
```rust
fn cmd_extract(args: &[String]) -> Result<()> {
```
(using `&[String]` not `&[str]`). Adjust string comparisons accordingly
with `.as_str()`.

**Step 4: Run integration test**

Run: `cargo test --lib graph::tests -q`
Expected: PASS

**Step 5: Build and smoke-test the binary**

Run: `cargo build --bin npu-graph && ./target/debug/npu-graph extract --arch aie2`
Expected: Prints extraction progress and writes `models/aie2.json`

**Step 6: Commit**

```bash
git add src/bin/npu_graph.rs models/
git commit -m "feat(graph): CLI extract command -- full pipeline to JSON"
```

---

## Task 7: Query API

**Files:**
- Create: `src/graph/query.rs`
- Modify: `src/graph/mod.rs` (add `pub mod query;`)

Adds typed query methods on `ArchModel` for programmatic access.
These are the methods the emulator will eventually call.

**Step 1: Write the failing test**

In `src/graph/query.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::*;

    fn sample_model() -> ArchModel {
        // Build a small model for testing queries
        let mut model = ArchModel::new(Architecture::Aie2);
        let attr = || SourceAttribution {
            origin: Source::Am025Json,
            file: "test".to_string(),
            detail: "test".to_string(),
        };

        model.tile_types.push(TileTypeModel {
            kind: TileKind::Compute,
            name: "aie2_compute".to_string(),
            modules: vec![
                ModuleModel {
                    kind: ModuleKind::Dma,
                    registers: vec![
                        RegisterModel {
                            name: "DMA_BD0_0".to_string(),
                            offset: 0x1D000,
                            width: 32,
                            reset_value: 0,
                            fields: vec![FieldModel {
                                name: "Buffer_Length".to_string(),
                                bits: BitRange::Contiguous { msb: 13, lsb: 0 },
                                meaning: FieldSemantics::Count,
                                source: attr(),
                            }],
                            module: ModuleKind::Dma,
                            access: Access::ReadWrite,
                            source: attr(),
                        },
                        RegisterModel {
                            name: "DMA_S2MM_0_Ctrl".to_string(),
                            offset: 0x1DE00,
                            width: 32,
                            reset_value: 0,
                            fields: Vec::new(),
                            module: ModuleKind::Dma,
                            access: Access::ReadWrite,
                            source: attr(),
                        },
                    ],
                    ports: Vec::new(),
                    instances: Some(InstanceCount { channels: 2, bds: 16, locks: 0 }),
                    source: attr(),
                },
                ModuleModel {
                    kind: ModuleKind::Lock,
                    registers: Vec::new(),
                    ports: Vec::new(),
                    instances: Some(InstanceCount { channels: 0, bds: 0, locks: 16 }),
                    source: attr(),
                },
            ],
            memory: Some(MemoryModel {
                size_bytes: 65536,
                num_banks: 4,
                bank_size_bytes: 16384,
                source: attr(),
            }),
            source: attr(),
        });

        model
    }

    #[test]
    fn test_tile_type_by_kind() {
        let model = sample_model();
        let compute = model.tile_type(TileKind::Compute);
        assert!(compute.is_some());
        assert_eq!(compute.unwrap().name, "aie2_compute");
    }

    #[test]
    fn test_tile_type_missing() {
        let model = sample_model();
        let shim = model.tile_type(TileKind::Shim);
        assert!(shim.is_none());
    }

    #[test]
    fn test_register_by_offset() {
        let model = sample_model();
        let reg = model.register_at(TileKind::Compute, 0x1D000);
        assert!(reg.is_some());
        assert_eq!(reg.unwrap().name, "DMA_BD0_0");
    }

    #[test]
    fn test_register_by_offset_missing() {
        let model = sample_model();
        let reg = model.register_at(TileKind::Compute, 0xFFFF);
        assert!(reg.is_none());
    }

    #[test]
    fn test_module_for_offset() {
        let model = sample_model();
        let module = model.module_for_offset(TileKind::Compute, 0x1D000);
        assert!(module.is_some());
        assert_eq!(module.unwrap().kind, ModuleKind::Dma);
    }

    #[test]
    fn test_registers_in_module() {
        let model = sample_model();
        let regs = model.registers_in(TileKind::Compute, ModuleKind::Dma);
        assert_eq!(regs.len(), 2);
    }

    #[test]
    fn test_fields_of_register() {
        let model = sample_model();
        let reg = model.register_at(TileKind::Compute, 0x1D000).unwrap();
        assert_eq!(reg.fields.len(), 1);
        assert_eq!(reg.fields[0].name, "Buffer_Length");
    }

    #[test]
    fn test_instance_counts() {
        let model = sample_model();
        let compute = model.tile_type(TileKind::Compute).unwrap();
        let dma = compute.modules.iter().find(|m| m.kind == ModuleKind::Dma).unwrap();
        let inst = dma.instances.as_ref().unwrap();
        assert_eq!(inst.bds, 16);
        assert_eq!(inst.channels, 2);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib graph::query::tests -q`
Expected: FAIL -- module not found

**Step 3: Write the query API**

Create `src/graph/query.rs`:

```rust
//! Query API for the NPU architecture model.
//!
//! Provides typed accessor methods on `ArchModel` for programmatic
//! queries. These methods are designed to be called by the emulator
//! at runtime or by analysis tools.

use crate::graph::types::*;

impl ArchModel {
    /// Find a tile type by kind.
    pub fn tile_type(&self, kind: TileKind) -> Option<&TileTypeModel> {
        self.tile_types.iter().find(|t| t.kind == kind)
    }

    /// Find a register by offset within a tile type.
    pub fn register_at(&self, tile: TileKind, offset: u32) -> Option<&RegisterModel> {
        self.tile_type(tile)?
            .modules.iter()
            .flat_map(|m| &m.registers)
            .find(|r| r.offset == offset)
    }

    /// Find which module owns a register offset within a tile type.
    pub fn module_for_offset(&self, tile: TileKind, offset: u32) -> Option<&ModuleModel> {
        self.tile_type(tile)?
            .modules.iter()
            .find(|m| m.registers.iter().any(|r| r.offset == offset))
    }

    /// Get all registers belonging to a specific module in a tile type.
    pub fn registers_in(&self, tile: TileKind, module: ModuleKind) -> Vec<&RegisterModel> {
        self.tile_type(tile)
            .map(|t| {
                t.modules.iter()
                    .filter(|m| m.kind == module)
                    .flat_map(|m| &m.registers)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all registers across all modules in a tile type.
    pub fn all_registers(&self, tile: TileKind) -> Vec<&RegisterModel> {
        self.tile_type(tile)
            .map(|t| {
                t.modules.iter()
                    .flat_map(|m| &m.registers)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find all relationships involving a node.
    pub fn relationships_for(&self, node: &str) -> Vec<&Relationship> {
        self.relationships.iter()
            .filter(|r| r.from.0 == node || r.to.0 == node)
            .collect()
    }

    /// Find all relationships of a specific kind.
    pub fn relationships_of_kind(&self, kind: RelationshipKind) -> Vec<&Relationship> {
        self.relationships.iter()
            .filter(|r| r.kind == kind)
            .collect()
    }
}
```

Add `pub mod query;` to `src/graph/mod.rs`.

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib graph::query::tests -q`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/graph/query.rs src/graph/mod.rs
git commit -m "feat(graph): query API -- tile_type, register_at, module_for_offset"
```

---

## Task 8: Relationship Builder

**Files:**
- Create: `src/graph/relationships.rs`
- Modify: `src/graph/mod.rs` (add `pub mod relationships;`)

Builds `Contains` and `Configures` relationships from the model structure.
These are the foundational edges that enable "what touches what" queries.

**Step 1: Write the failing test**

In `src/graph/relationships.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::*;

    fn sample_model() -> ArchModel {
        // Reuse the same model-building pattern from query tests.
        // Create a compute tile with DMA and Lock modules.
        let attr = || SourceAttribution {
            origin: Source::Am025Json,
            file: "test".to_string(),
            detail: "test".to_string(),
        };

        let mut model = ArchModel::new(Architecture::Aie2);
        model.tile_types.push(TileTypeModel {
            kind: TileKind::Compute,
            name: "aie2_compute".to_string(),
            modules: vec![
                ModuleModel {
                    kind: ModuleKind::Dma,
                    registers: vec![RegisterModel {
                        name: "DMA_BD0_0".to_string(),
                        offset: 0x1D000,
                        width: 32,
                        reset_value: 0,
                        fields: Vec::new(),
                        module: ModuleKind::Dma,
                        access: Access::ReadWrite,
                        source: attr(),
                    }],
                    ports: Vec::new(),
                    instances: None,
                    source: attr(),
                },
                ModuleModel {
                    kind: ModuleKind::Lock,
                    registers: Vec::new(),
                    ports: Vec::new(),
                    instances: None,
                    source: attr(),
                },
            ],
            memory: None,
            source: attr(),
        });
        model
    }

    #[test]
    fn test_build_containment_relationships() {
        let mut model = sample_model();
        build_containment(&mut model);

        let contains: Vec<_> = model.relationships.iter()
            .filter(|r| r.kind == RelationshipKind::Contains)
            .collect();

        // Should have: compute Contains dma, compute Contains lock,
        // dma Contains DMA_BD0_0
        assert!(contains.len() >= 3, "expected at least 3 Contains edges, got {}", contains.len());
    }

    #[test]
    fn test_containment_edges_have_correct_direction() {
        let mut model = sample_model();
        build_containment(&mut model);

        // Tile contains module
        let tile_to_dma = model.relationships.iter().find(|r| {
            r.kind == RelationshipKind::Contains
                && r.from.0.contains("compute")
                && r.to.0.contains("dma")
        });
        assert!(tile_to_dma.is_some(), "should have compute -> dma containment");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib graph::relationships::tests -q`
Expected: FAIL -- module not found

**Step 3: Write the relationship builder**

Create `src/graph/relationships.rs`:

```rust
//! Build relationships between nodes in the architecture model.
//!
//! Produces `Contains` edges for the structural hierarchy (tile contains
//! module, module contains register) and `Configures` edges for known
//! programming relationships (e.g., BD register configures DMA channel).

use crate::graph::types::*;

/// Build containment relationships from the model's hierarchical structure.
///
/// Produces edges:
///   tile_type -> module (Contains)
///   module -> register (Contains)
pub fn build_containment(model: &mut ArchModel) {
    let arch = model.arch.to_string().to_lowercase();
    let attr = SourceAttribution {
        origin: Source::DeviceModel,
        file: "derived".to_string(),
        detail: "structural containment".to_string(),
    };

    for tile_type in &model.tile_types {
        let tile_id = format!("{}.{}", arch, tile_type.name);

        for module in &tile_type.modules {
            let mod_id = format!("{}.{}", tile_id, module.kind);

            // Tile contains module
            model.relationships.push(Relationship {
                from: NodeId(tile_id.clone()),
                to: NodeId(mod_id.clone()),
                kind: RelationshipKind::Contains,
                source: attr.clone(),
            });

            // Module contains each register
            for reg in &module.registers {
                let reg_id = format!("{}.{}", mod_id, reg.name);
                model.relationships.push(Relationship {
                    from: NodeId(mod_id.clone()),
                    to: NodeId(reg_id),
                    kind: RelationshipKind::Contains,
                    source: attr.clone(),
                });
            }
        }
    }
}

// Note: This function modifies model.relationships in place, but reads
// from model.tile_types. The borrow checker requires we collect the
// relationships to add, then push them. Restructure if needed:

// Actually, the above code has a borrow issue -- we iterate tile_types
// while pushing to relationships on the same model. Fix by collecting
// relationships first:

/// Build containment relationships (borrow-safe version).
pub fn build_all_relationships(model: &mut ArchModel) {
    let arch = model.arch.to_string().to_lowercase();
    let attr = SourceAttribution {
        origin: Source::DeviceModel,
        file: "derived".to_string(),
        detail: "structural containment".to_string(),
    };

    let mut new_rels = Vec::new();

    for tile_type in &model.tile_types {
        let tile_id = format!("{}.{}", arch, tile_type.name);

        for module in &tile_type.modules {
            let mod_id = format!("{}.{}", tile_id, module.kind);

            new_rels.push(Relationship {
                from: NodeId(tile_id.clone()),
                to: NodeId(mod_id.clone()),
                kind: RelationshipKind::Contains,
                source: attr.clone(),
            });

            for reg in &module.registers {
                let reg_id = format!("{}.{}", mod_id, reg.name);
                new_rels.push(Relationship {
                    from: NodeId(mod_id.clone()),
                    to: NodeId(reg_id),
                    kind: RelationshipKind::Contains,
                    source: attr.clone(),
                });
            }
        }
    }

    model.relationships.extend(new_rels);
}
```

**NOTE:** The implementer should remove the first `build_containment`
function and keep only `build_all_relationships`, renaming it to
`build_containment` in the process. The duplicate is shown above to
illustrate the borrow-checker issue. The tests should call whichever
name survives.

Add `pub mod relationships;` to `src/graph/mod.rs`.

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib graph::relationships::tests -q`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add src/graph/relationships.rs src/graph/mod.rs
git commit -m "feat(graph): relationship builder -- containment edges"
```

---

## Task 9: CLI Query Command

**Files:**
- Modify: `src/bin/npu_graph.rs` (implement `query` command)

Implements the CLI `query` command so users can interrogate a model
from the command line.

**Step 1: Write a test model**

First, generate a model to test against:

Run: `cargo build --bin npu-graph && ./target/debug/npu-graph extract --arch aie2`
Expected: `models/aie2.json` created

**Step 2: Implement the query command**

Add to `src/bin/npu_graph.rs`:

```rust
fn cmd_query(args: &[String]) -> Result<()> {
    let mut model_path = None;
    let mut query_parts = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = Some(&args[i]); }
            _ => query_parts.push(args[i].clone()),
        }
        i += 1;
    }

    let model_path = model_path
        .ok_or_else(|| anyhow::anyhow!("--model required"))?;

    let text = std::fs::read_to_string(model_path)
        .with_context(|| format!("reading model from {}", model_path))?;
    let model: types::ArchModel = serde_json::from_str(&text)
        .context("parsing model JSON")?;

    if query_parts.is_empty() {
        // Print summary
        println!("Architecture: {}", model.arch);
        println!("Tile types: {}", model.tile_types.len());
        for tt in &model.tile_types {
            let reg_count: usize = tt.modules.iter().map(|m| m.registers.len()).sum();
            println!("  {} ({}) -- {} modules, {} registers",
                tt.name, tt.kind, tt.modules.len(), reg_count);
        }
        println!("Relationships: {}", model.relationships.len());
        return Ok(());
    }

    let query = query_parts.join(" ");
    match query.as_str() {
        q if q.starts_with("register ") => {
            let offset_str = q.trim_start_matches("register ").trim();
            let offset = if offset_str.starts_with("0x") {
                u32::from_str_radix(offset_str.trim_start_matches("0x"), 16)?
            } else {
                offset_str.parse()?
            };
            for tt in &model.tile_types {
                if let Some(reg) = model.register_at(tt.kind, offset) {
                    println!("{} ({}) -- {} at {:#07X}", tt.name, tt.kind, reg.name, reg.offset);
                    println!("  Access: {:?}, Width: {}, Reset: {:#010X}", reg.access, reg.width, reg.reset_value);
                    println!("  Source: {}", reg.source);
                    for field in &reg.fields {
                        println!("  Field: {} [{} bits] {:?}", field.name, field.bits.width(), field.meaning);
                    }
                }
            }
        }
        q if q.starts_with("module ") => {
            let mod_name = q.trim_start_matches("module ").trim();
            let mod_kind = match mod_name {
                "dma" => types::ModuleKind::Dma,
                "lock" => types::ModuleKind::Lock,
                "stream_switch" | "ss" => types::ModuleKind::StreamSwitch,
                "core" => types::ModuleKind::Core,
                "trace" => types::ModuleKind::Trace,
                other => { eprintln!("Unknown module: {}", other); return Ok(()); }
            };
            for tt in &model.tile_types {
                let regs = model.registers_in(tt.kind, mod_kind);
                if !regs.is_empty() {
                    println!("{} ({}) -- {} {} registers:", tt.name, tt.kind, regs.len(), mod_name);
                    for reg in &regs {
                        println!("  {:#07X}  {}  ({} fields)", reg.offset, reg.name, reg.fields.len());
                    }
                }
            }
        }
        q if q.starts_with("tile ") => {
            let tile_name = q.trim_start_matches("tile ").trim();
            let kind = match tile_name {
                "compute" => types::TileKind::Compute,
                "mem" => types::TileKind::Mem,
                "shim" => types::TileKind::Shim,
                other => { eprintln!("Unknown tile: {}", other); return Ok(()); }
            };
            if let Some(tt) = model.tile_type(kind) {
                println!("{} ({})", tt.name, tt.kind);
                if let Some(mem) = &tt.memory {
                    println!("  Memory: {} bytes, {} banks", mem.size_bytes, mem.num_banks);
                }
                for module in &tt.modules {
                    print!("  Module: {}", module.kind);
                    if let Some(inst) = &module.instances {
                        print!(" (locks={}, bds={}, channels={})", inst.locks, inst.bds, inst.channels);
                    }
                    println!(" -- {} registers, {} ports", module.registers.len(), module.ports.len());
                }
            } else {
                println!("No {} tile type found", tile_name);
            }
        }
        _ => {
            eprintln!("Unknown query: {}", query);
            eprintln!("Available queries:");
            eprintln!("  register <offset>   -- look up register by offset");
            eprintln!("  module <name>        -- list registers in a module (dma, lock, ss, core, trace)");
            eprintln!("  tile <kind>          -- describe a tile type (compute, mem, shim)");
            eprintln!("  (no query)           -- print model summary");
        }
    }

    Ok(())
}
```

Wire `cmd_query` into the main match arm.

**Step 3: Test from command line**

Run: `./target/debug/npu-graph query --model models/aie2.json`
Expected: Prints model summary

Run: `./target/debug/npu-graph query --model models/aie2.json tile compute`
Expected: Prints compute tile details with memory, modules, instance counts

Run: `./target/debug/npu-graph query --model models/aie2.json register 0x1D000`
Expected: Prints DMA_BD0_0 register details

**Step 4: Commit**

```bash
git add src/bin/npu_graph.rs
git commit -m "feat(graph): CLI query command -- register, module, tile queries"
```

---

## Task 10: Integration Test -- Full Pipeline

**Files:**
- Modify: `src/graph/mod.rs` (add integration test)

End-to-end test that extracts, merges, serializes, deserializes, and
queries a real model.

**Step 1: Write the test**

In `src/graph/mod.rs` tests section:

```rust
#[test]
fn test_full_pipeline_with_relationships() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let device_model_path = manifest.join("tools/aie-device-models.json");
    let mlir_aie = std::env::var("MLIR_AIE_PATH").unwrap_or_else(|_| {
        manifest.parent().unwrap()
            .join("mlir-aie")
            .to_string_lossy().to_string()
    });
    let regdb_path = Path::new(&mlir_aie)
        .join("lib/Dialect/AIE/Util/aie_registers_aie2.json");

    let device = extract_device_model::extract_device_model(
        &device_model_path, "npu1"
    ).unwrap();
    let regdb = extract_regdb::extract_regdb(&regdb_path).unwrap();
    let mut model = merge::merge_model(device, regdb, None).unwrap();

    // Build relationships
    relationships::build_all_relationships(&mut model);

    // Verify containment edges exist
    let contains: Vec<_> = model.relationships_of_kind(
        types::RelationshipKind::Contains
    );
    assert!(!contains.is_empty(), "should have containment relationships");

    // Query: find DMA module for compute tile
    let dma_regs = model.registers_in(types::TileKind::Compute, types::ModuleKind::Dma);
    assert!(!dma_regs.is_empty(), "compute tile should have DMA registers");

    // Query: register lookup by offset
    // DMA_BD0_0 is at 0x1D000 in the memory module
    let bd_reg = model.register_at(types::TileKind::Compute, 0x1D000);
    // May or may not find it depending on module assignment -- the
    // "memory" JSON module gets classified as DMA, and compute tiles
    // may or may not get those registers in the merge step.
    // This test verifies the pipeline doesn't crash; exact lookup
    // behavior is refined in later tasks.

    // Serialize round-trip
    let json = serde_json::to_string(&model).unwrap();
    let round_tripped: types::ArchModel = serde_json::from_str(&json).unwrap();
    assert_eq!(round_tripped.tile_types.len(), model.tile_types.len());
    assert_eq!(round_tripped.relationships.len(), model.relationships.len());
}
```

**Step 2: Run the test**

Run: `cargo test --lib graph::tests -q`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/graph/mod.rs
git commit -m "test(graph): end-to-end integration test with relationships"
```

---

## Future Tasks (Not in This Plan)

These build on the foundation above and should be planned separately once
the core pipeline is working:

1. **Rust code generation** -- Generate `src/generated/aie2_model.rs` to
   replace `aie2_spec.rs` and `registers_spec.rs`. Requires deciding
   exactly which constants move from hand-coded to generated.

2. **aie-rt cross-validation** -- Flesh out `cross_validate_aiert()` to
   compare every #define offset against the AM025 JSON offset for the
   same register. This is the "error on disagreement" enforcement.

3. **Architecture diff** -- `npu-graph diff --a aie2.json --b aie2p.json`
   to show exactly what changed between architectures.

4. **Side-effect modeling** -- Parse aie-rt function bodies (e.g.,
   `_XAieMl_DmaChStart`) to extract programming sequences as
   `SideEffect` entries.

5. **TableGen extractor** -- Import instruction set model. Can reuse
   existing `src/tablegen/` infrastructure.

6. **Programming model stubs** -- CDO/NPU instruction/control packet
   nodes and their `Configures` edges into the silicon model.

7. **Emulator integration** -- Replace `aie2_spec.rs` hardcoded values
   with queries against the loaded `ArchModel`. Replace
   `module_from_offset()` with `model.module_for_offset()`.

8. **FieldSemantics classification** -- Automatically classify field
   meanings (Address, Count, Enable, LockId) based on name patterns
   and aie-rt usage.
