# NPU Architecture Graph -- Queryable Hardware Model

**Date:** 2026-03-06
**Status:** Approved
**Scope:** Multi-architecture hardware model extracted from toolchain sources

## Problem

The emulator has accumulated architectural knowledge from multiple sources,
scattered across different modules with different representations:

- `regdb.rs` parses AM025 JSON for BD fields
- `build.rs` extracts aie-rt constants via gcc -E
- `aie2_spec.rs` hardcodes stream switch ports from AM025
- `aie-device-models.json` provides tile topology
- `state.rs` encodes side-effect knowledge inline in match arms

This fragmentation causes bugs (the register bus regression being a prime
example) because no single place describes "what the hardware actually does."
Each module has its own partial understanding, and they disagree silently.

## Solution

A standalone tool (`npu-graph`) that extracts hardware architecture from the
authoritative open-source toolchain sources, synthesizes them into a single
typed Rust model, and outputs both generated Rust source and a serialized
queryable graph. The emulator imports the generated source and can load the
graph at runtime for programmatic queries.

## Design Principles

1. **Derive from the toolchain.** Every fact in the model traces back to a
   specific source (aie-rt, AM025 JSON, llvm-aie TableGen, device model).
   Nothing is hardcoded that can be extracted.

2. **Hard error on disagreement.** When multiple sources describe the same
   fact and disagree, extraction fails immediately. Ambiguity is a bug that
   requires human resolution.

3. **Multi-architecture from day one.** AIE, AIE2, AIE2P, AIE2PS are
   different models, not special cases. The extraction tool runs once per
   architecture.

4. **Library-first.** The graph is a Rust library with a typed API. The CLI
   wraps it. The emulator imports it. Tests assert against it.

## Core Data Model

### Top Level

```rust
pub enum Architecture {
    Aie,      // AIE1 (Versal)
    Aie2,     // AIE-ML (Phoenix/Hawk Point)
    Aie2p,    // AIE-ML v2 (Strix Point)
}

pub struct ArchModel {
    pub arch: Architecture,
    pub tile_types: Vec<TileTypeModel>,
    pub array_topology: ArrayTopology,
    pub register_map: RegisterMap,
    pub instruction_set: InstructionSet,
    pub programming_model: ProgrammingModel,
    pub artifacts: ArtifactModel,
}
```

Each architecture produces a separate `ArchModel`. Differences between
architectures show up as different model contents, not if/else branches.
Comparing architectures is `aie2.diff(&aie2p)`.

### Tile and Module Internals

```rust
pub struct TileTypeModel {
    pub kind: TileKind,                    // Compute, Mem, Shim
    pub name: String,                      // "aie2_compute", "aie2p_mem", etc.
    pub modules: Vec<ModuleModel>,
    pub memory: MemoryModel,               // size, banks, address space
    pub register_range: (u32, u32),        // offset range owned by this tile type
}

pub struct ModuleModel {
    pub kind: ModuleKind,                  // Dma, Lock, StreamSwitch, Core, Trace, ...
    pub registers: Vec<RegisterModel>,
    pub side_effects: Vec<SideEffect>,
    pub ports: Vec<PortModel>,             // connection points to other modules
    pub instances: InstanceCount,          // 2 DMA channels, 64 locks, 16 BDs, etc.
}

pub struct RegisterModel {
    pub name: String,
    pub offset: u32,
    pub width: u8,
    pub reset_value: u32,
    pub fields: Vec<FieldModel>,
    pub module: ModuleKind,
    pub access: Access,                    // ReadWrite, ReadOnly, WriteOnly, WriteToClear
}

pub struct FieldModel {
    pub name: String,
    pub bits: BitRange,                    // contiguous or split
    pub meaning: FieldSemantics,           // Address, Count, Enable, LockId, ...
}
```

### Side Effects and Source Attribution

Every behavioral fact carries attribution to its source:

```rust
pub struct SideEffect {
    pub trigger: Trigger,                  // WriteToField("Start_Queue", any_value)
    pub effects: Vec<Effect>,              // [EnqueueBd, CheckLock, BeginTransfer, ...]
    pub source: SourceAttribution,         // "aie-rt: _XAieMl_DmaChStart line 234"
}

pub struct SourceAttribution {
    pub origin: Source,                    // AieRt, Am025Json, TableGen, DeviceModel
    pub file: String,
    pub detail: String,                    // function name, line number, or JSON path
}
```

### Relationships

Cross-module and cross-tile connections:

```rust
pub struct Relationship {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: RelationshipKind,
    pub source: SourceAttribution,
}

pub enum RelationshipKind {
    DataFlow,          // data moves from A to B
    Configures,        // writing A programs B's behavior
    BlocksOn,          // A stalls until B satisfies a condition
    Triggers,          // writing A causes B to act
    RoutesTo,          // stream switch connects A to B
    Produces,          // toolchain artifact A generates B
    Contains,          // hierarchical: tile contains modules
}
```

### Query API

```rust
let model = ArchModel::load("aie2")?;

// What does a DMA channel start touch?
model.what_touches("dma.channel_0.start_queue")
// -> [bd_chain, lock_arbiter, stream_port, status_register, event_generate]

// How does data flow from DDR to core memory?
model.data_path("shim.dma.mm2s_0", "compute.data_memory")
// -> [shim_dma, shim_stream_switch, noc, tile_stream_switch, tile_dma, data_memory]

// What registers configure a DMA transfer?
model.what_configures("dma.channel_0")
// -> [bd_word_0..7, channel_control, start_queue]

// Compare architectures
let diff = aie2.diff(&aie2p);
// -> [Added: compute.dma.channel_2, Changed: mem.bd_count 16->24, ...]
```

## Extractors

Each source gets a dedicated extractor module:

| Extractor | Source | Populates |
|-----------|--------|-----------|
| `extract_regdb` | AM025 JSON (`aie_registers_aie2.json`) | Registers, fields, module groupings |
| `extract_aiert` | aie-rt headers via gcc -E | Register offsets, bit masks, side-effect sequences |
| `extract_device_model` | `aie-device-models.json` | Tile types, array topology, instance counts |
| `extract_tablegen` | llvm-aie `.td` files | Instruction set, register files, VLIW structure |
| `extract_programming` | CDO/NPU format specs + aie-rt | Programming model nodes, configures edges |

Each extractor returns a partial model. The merge step combines them and
errors on disagreement. Ownership is explicit: regdb owns field bit
positions, aie-rt owns side effects, device model owns topology.

When sources are ambiguous and neither the open-source toolchain nor
aie-rt resolves the question, the extractor should report the ambiguity
for manual resolution rather than guessing.

## Outputs

Two artifacts from successful extraction:

1. **Generated Rust source** (`src/generated/aie2_model.rs`) -- register
   maps, field layouts, module structure as compiled-in constants. Replaces
   current `aie2_spec.rs`, `registers_spec.rs`, and build.rs-generated
   constants over time.

2. **Serialized graph** (`models/aie2.json`) -- the full model with all
   relationships, queryable from CLI and loadable as library.

## Tool Interface

```
npu-graph extract --arch aie2 \
    --aiert-src ../aie-rt/driver/src \
    --regdb ../mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json \
    --tablegen ../llvm-aie/llvm/lib/Target/AIE \
    --device-model tools/aie-device-models.json \
    --output-rs src/generated/aie2_model.rs \
    --output-json models/aie2.json

npu-graph query --model models/aie2.json "what_touches dma.channel_0"

npu-graph diff --a models/aie2.json --b models/aie2p.json
```

## Toolchain Stubs

The toolchain (mlir-aie, Peano) is modeled only at its output boundary:

- "The compiler produces an XCLBIN containing ELFs + CDO"
- "CDO configures these hardware registers in these modules"
- "ELFs contain VLIW code targeting these core resources"
- "NPU instructions write to these register offsets"
- "Control packets route register writes via stream switch"

No modeling of compiler internals (MLIR lowering, register allocation, etc.).

## Multi-Architecture Strategy

The `Architecture` enum and per-architecture extraction mean:

- AIE2 compute tile and AIE2P compute tile are different `TileTypeModel`
  entries with different contents (different BD counts, channel counts,
  register offsets)
- No if/else branches on architecture in the model -- differences are data
- `ArchModel::diff()` shows exactly what changed between architectures
- Adding a new architecture = running the extraction tool with new source
  paths

## Expected Outcome

- Single source of truth for NPU hardware architecture
- Every emulator module can query the model instead of hardcoding
- Source disagreements caught immediately instead of causing silent bugs
- Architecture differences explicit and queryable
- Foundation for project-wide cleanup: hold emulator against graph,
  fix every mismatch
