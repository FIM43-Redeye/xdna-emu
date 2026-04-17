# Device Model Extractor Design

Task 2 of the architecture graph pipeline.

## Purpose

Parse `tools/aie-device-models.json` into typed `ArchModel` instances.
One model per device name (12 total: npu1 + 3 column variants, npu2 + 7
column variants). Column variants are virtualized subsets of the full
device -- same tile-level parameters, different array widths.

## File

`src/graph/device_model.rs`, registered in `src/graph/mod.rs`.

Future extractors follow the same pattern: `regdb.rs` (AM025 JSON),
`aiert.rs` (aie-rt headers). One file per source.

## Interface

```rust
/// Parse all devices from a device model JSON file.
pub fn extract_device_models(path: &Path) -> Result<HashMap<String, ArchModel>, ExtractError>

/// Parse a single named device from a device model JSON file.
pub fn extract_device_model(path: &Path, device: &str) -> Result<ArchModel, ExtractError>
```

## Error Type

```rust
pub enum ExtractError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnknownField { context: String, field: String },
    MissingField { context: String, field: String },
    UnknownDevice { name: String },
}
```

## Parsing Strategy

Use `serde_json::Value` for initial parse, then walk the tree manually
with small focused functions. NOT `#[derive(Deserialize)]` on intermediate
structs -- that would create a parallel type hierarchy mirroring the graph
types.

### Function decomposition

```
extract_device_models(path)
  -> for each device in JSON: extract_device(name, value)

extract_device(name, value) -> ArchModel
  -> extract_constants(value) -> DeviceConstants
  -> for each tile type: extract_tile_type(name, value)
  -> extract_tile_map(value) -> Vec<TilePlacement>
  -> assemble ArrayTopology, ArchModel

extract_tile_type(name, value) -> TileTypeModel
  -> extract_ports(value) -> Vec<PortBundle>
  -> extract_memory(value, tile_name) -> Option<MemoryModel>
  -> extract InstanceCount from num_locks, num_bds, DMA channel count

extract_ports(value) -> Vec<PortBundle>
  -> for each bundle: read master/slave counts

extract_tile_map(value) -> Vec<TilePlacement>
  -> for each entry: col, row, type, is_internal, edges, mem_affinity

extract_constants(value) -> DeviceConstants
  -> max_lock_value, address_gen_granularity, mem_base_addresses
  -> min_lock_value and properties inferred from architecture
```

### Strict field checking

Every function that reads a JSON object validates that ALL keys are
recognized. A generic helper does this:

```rust
fn check_keys(obj: &Map<String, Value>, known: &[&str], context: &str)
    -> Result<(), ExtractError>
```

If a JSON object contains a key not in `known`, return
`ExtractError::UnknownField`. This ensures the extractor breaks loudly
if the JSON generator adds new fields we haven't accounted for.

## Architecture Inference

The JSON doesn't declare architecture family. Infer from device name:

| Prefix | Architecture | Notes |
|--------|-------------|-------|
| `npu1` | `Architecture::Aie2` | Phoenix / Hawk Point |
| `npu2` | `Architecture::Aie2p` | Strix Point |

Unrecognized prefixes produce `ExtractError::UnknownDevice`.

Future: Versal devices (`xcvc1902`, `xcve2802`) would add entries here.

## DMA Channel Count

The JSON has no explicit `num_channels` field. DMA channels appear as
port bundles:
- Compute/memtile: `switchbox_ports["DMA"]` master count = channel count
- Shim: `shim_mux_ports["DMA"]` master count = channel count

The extractor reads DMA master count from the appropriate port namespace
and stores it in `InstanceCount.channels`.

## Fields NOT Populated (from other sources)

| Field | Default | Populated by |
|-------|---------|-------------|
| `generation` | `None` | aie-rt extractor |
| `dma_capabilities` | `None` | aie-rt extractor |
| `accumulator_cascade_bits` | `None` | mlir-aie (future) |
| `modules` (registers) | empty Vec | AM025 regdb extractor |
| `relationships` | empty Vec | relationship builder |

Fields that CAN be inferred from architecture but aren't in the JSON:

| Field | AIE2 value | AIE2P value | Source |
|-------|-----------|-------------|--------|
| `min_lock_value` | -64 | -64 | aie-rt `VAL_LOWER_BOUND` |
| `uses_semaphore_locks` | true | true | mlir-aie `ModelProperty` |
| `uses_multi_dim_bds` | true | true | mlir-aie `ModelProperty` |

These are set to architecture-appropriate defaults by the extractor.

## SourceAttribution

Every extracted item gets:
- `origin: Source::DeviceModel`
- `file`: the JSON file path
- `detail`: dot-path context, e.g., `"npu1.tile_types.core.switchbox_ports"`

## Tests

1. **Full parse**: Load real `aie-device-models.json`, verify all 12 devices
   parse without error.
2. **Spot-check npu1**: 4 columns, 6 rows, 3 tile types, 24 tiles in
   tile_map, device_id=4, is_npu=true.
3. **Spot-check npu2**: 8 columns, 6 rows, device_id=8.
4. **Tile type coverage**: Core tile has program_memory_bytes=16384,
   memtile has 48 BDs and 64 locks, shim has shim_mux_ports.
5. **Port bundle counts**: Core switchbox has 9 bundles, shim has
   shim_mux with DMA+South.
6. **Strict unknown field**: Inject `"bogus": 42` into a device object,
   verify `UnknownField` error.
7. **Missing required field**: Remove `columns` from a device, verify
   `MissingField` error.
8. **Architecture inference**: npu1 -> Aie2, npu2 -> Aie2p.
9. **Column variants**: npu1_1col has 1 column, npu1 has 4, same tile
   types and constants.

## Non-Goals

- Deduplication of column variants (caller's concern)
- Populating Layer 2+ data (registers, behavior)
- Merging with other sources (Task 5)
