# Architecture Graph -- Source Map

What each toolchain component provides to the graph, what's wired up,
and what's next.

## The Principle

Every fact in the graph comes from one or more toolchain sources. Where
sources overlap, `Confirmed<T>` enforces agreement. Conflicts panic
with full attribution so we catch parser bugs and toolchain
inconsistencies immediately.

Sources are patchwork -- each knows different things. The graph assembles
the mosaic. Single-source facts are automatically correct; multi-source
facts must agree.

---

## Source Inventory

### 1. Device Model JSON (`tools/aie-device-models.json`)

**Generator**: `tools/aie-device-dump.py` (queries mlir-aie Python API)
**Extractor**: `src/graph/device_model.rs`
**Status**: WIRED

Provides the array skeleton -- everything about the physical layout and
tile-level resource counts.

| Data | Graph Type | Status |
|------|-----------|--------|
| Grid dimensions (cols, rows) | `ArrayTopology` | Wired |
| Tile placement + adjacency | `TilePlacement`, `CardinalFlags` | Wired |
| Memory affinity (core tiles) | `TilePlacement.mem_affinity` | Wired |
| Tile types (3: core, mem_tile, shim_noc) | `TileTypeModel` | Wired |
| Lock count per tile type | `InstanceCount.locks` | Wired, CONFIRMED by AM025 |
| BD count per tile type | `InstanceCount.bds` | Wired, CONFIRMED by AM025 |
| DMA channel count | `InstanceCount.channels` | Wired (single source) |
| Memory sizes + bank layout | `MemoryModel` | Wired |
| Program memory size (core only) | `MemoryModel.program_memory_bytes` | Wired |
| Port bundles (switchbox + shim_mux) | `PortBundle` | Wired |
| Device constants (lock range, address gran) | `DeviceConstants` | Wired |
| Memory base addresses (cardinal) | `DeviceConstants.mem_base_addresses` | Wired |

**12 devices**: npu1 + 3 column variants, npu2 + 7 column variants.
Column variants are identical tile-level hardware, different array widths.

### 2. AM025 Register Database (`../mlir-aie/.../aie_registers_aie2.json`)

**Extractor**: `src/graph/regdb_extractor.rs`
**Status**: WIRED

The register-level truth. Every register, every field, every bit position.

| Data | Graph Type | Status |
|------|-----------|--------|
| 1,806 register definitions | `RegisterModel` | Wired |
| 6,412 bit field definitions | `FieldModel` | Wired |
| Register offsets + widths | `RegisterModel.offset/width` | Wired |
| Access modes (RW/RO/WO/WTC/Mixed) | `Access` enum | Wired |
| Module grouping (core/memory/memory_tile/shim) | `ModuleModel` | Wired |
| Subsystem classification (15 kinds) | `SubsystemKind` | Wired |
| Instance counts from register groups | confirms `InstanceCount.locks/bds` | Wired |
| Field semantics (Address/Count/Enable/...) | `FieldSemantics` | Wired |
| Register categories (State/Operations/Events/Status) | `RegisterCategory` | Wired |

**Coverage**: Only AIE2 register file exists. No AIE2P variant in
mlir-aie yet. When one appears, the same extractor handles it.

### 3. aie-rt (`../aie-rt/driver/src/`)

**Extractor**: NOT YET BUILT (planned: tree-sitter-c parser)
**Status**: UNIMPLEMENTED

The host-side driver that programs real silicon. Its struct initializers
and #define constants are a goldmine of behavioral data that the register
database alone cannot express.

| Data | Where in aie-rt | Graph Target |
|------|-----------------|-------------|
| DMA capabilities (compression, padding, OOO, FIFO) | `XAie_DmaMod` struct init | `DmaCapabilities` |
| BD field register mappings | `XAie_DmaBdProp` / `XAie_RegFldAttr` | cross-ref with `FieldModel` |
| Lock module params (NumLocks confirmed) | `XAie_LockMod` struct init | confirms `InstanceCount.locks` |
| Stream switch port type assignments | `XAie_StrmMod` port arrays | extends `PortBundle` |
| ~500 event IDs per architecture | `xaie_events_aieml.h` #defines | future `EventModel` |
| Event-to-module mapping | `XAie_EvntMod.EventIdBase/EventIdMax` | future event subsystem |
| Performance counter config | `XAie_PerfMod` | future perf subsystem |
| Timer module config | `XAie_TimerMod` | future timer subsystem |
| Device generation constants | `XAIE_DEV_GEN_*` | `DeviceGeneration` |
| Architectural feature flags | struct field checks | `DmaCapabilities` flags |

**Scale**: 161K lines across 20 subsystem directories.
**Parser strategy**: tree-sitter-c for struct initializers and #defines.
Regex won't cut it -- the data is too structurally complex.

### 4. llvm-aie TableGen (`../llvm-aie/llvm/lib/Target/AIE/`)

**Extractor**: `src/tablegen/` (native parser, COMPLETE for instruction defs)
**Status**: PARTIALLY WIRED (used by emulator, not yet feeding graph)

The ISA specification. Already parsed for instruction decoding -- the
emulator's decoder is fully TableGen-driven.

| Data | Where in TableGen | Graph Target |
|------|-------------------|-------------|
| Instruction encodings (~600 AIE2) | `AIE2InstrInfo.td` | emulator decoder (done) |
| Register file structure (106 classes) | `AIE2RegisterInfo.td` | future `RegisterFileModel` |
| VLIW slot assignments (8 slots) | `AIE2InstrFormats.td` | emulator (done) |
| Scheduling latencies (275 classes) | `AIE2Schedule.td` | future timing model |
| Intrinsic signatures (317 AIE2) | `AIE2Intrinsics.td` | future vector dispatch |
| SemanticOp mappings (48 types) | attached to instruction defs | emulator (partial) |

**Not needed for the graph's register/subsystem layer.** Relevant when
we get to ISA and compute semantics. The native parser is already built
and battle-tested.

### 5. mlir-aie Device Model (`../mlir-aie/include/.../AIETargetModel.h`)

**Extractor**: via `tools/aie-device-dump.py` (Python API -> JSON)
**Status**: WIRED (through device model JSON)

The C++ `AIETargetModel` class provides the data that
`aie-device-dump.py` extracts. We don't parse the C++ directly -- the
Python API is the stable interface.

| Data | Method | Status |
|------|--------|--------|
| Lock local base index | `getLockLocalBaseIndex()` | Not extracted (behavioral) |
| Memory south base address | `getMemSouthBaseAddress()` | Extracted via JSON |
| Tile type queries | `isCoreTile()/isMemTile()/isShimNoc()` | Implicit in tile_map |
| Cascade direction | `getCascadeDir()` | Not extracted |

---

## Cross-Source Confirmation Matrix

Which sources can confirm which facts:

| Fact | Device Model | AM025 JSON | aie-rt | llvm-aie |
|------|:---:|:---:|:---:|:---:|
| Lock count per tile | YES | YES | (future) | - |
| BD count per tile | YES | YES | (future) | - |
| Channel count per tile | YES | - | (future) | - |
| Register offsets | - | YES | (future) | - |
| Register field positions | - | YES | (future) | - |
| Memory size | YES | - | (future) | - |
| Port bundle counts | YES | - | (future) | - |
| DMA capabilities | - | - | (future) | - |
| Instruction encoding | - | - | - | YES |

Currently confirmed: locks (2 sources), BDs (2 sources).
Adding aie-rt will bring most facts to 3-source confirmation.

---

## What's Closest to Registers (Next Candidates)

These are pieces that directly USE the register data we already have,
ordered by proximity to the existing graph:

### Tier 1: Direct register consumers (immediate)

1. **Subsystem behavioral profiles** -- We already compute
   `SubsystemProfile` (instance counts, register groups, categories)
   during `populate_tile_modules()` but discard them after confirming
   instance counts. Promoting these to graph-visible data would expose
   register group structure (how many BDs, how many fields per BD, which
   registers are State vs Operations vs Status).

2. **MemoryModel confirmation from registers** -- The AM025 register
   database has `DataMemory_*` registers whose address range reveals
   memory size. Could confirm `MemoryModel.size_bytes` from register
   offsets.

3. **DMA BD field schema** -- The register groups for DMA BDs
   (DMA_BD0_0 through DMA_BD0_5, 6 words x 16 BDs) define the complete
   BD programming interface. This is the bridge between "registers
   exist" and "what the DMA subsystem does." Already implicitly in
   the graph via `RegisterModel`, but not structured as a BD schema.

### Tier 2: Register + external data (needs aie-rt)

4. **DMA capabilities** -- `TileTypeModel.dma_capabilities` is `None`
   for all tile types. Requires aie-rt `XAie_DmaMod` struct fields
   (supports_compression, etc.). No AM025 signal for these.

5. **Stream switch port type mapping** -- Which port index maps to
   which bundle. Device model gives counts per bundle, aie-rt gives
   the actual port-to-bundle assignment arrays.

6. **Event ID catalog** -- 500+ event IDs in aie-rt header defines.
   No AM025 equivalent. Pure aie-rt extraction.

### Tier 3: ISA layer (separate domain)

7. **Register file model** -- TableGen register classes -> graph. Not
   register-map registers, but CPU register files (scalar, vector,
   accumulator). Different domain from the peripheral subsystem
   registers.

---

## File Paths Quick Reference

| Source | Path |
|--------|------|
| Device model JSON | `tools/aie-device-models.json` |
| Device model generator | `tools/aie-device-dump.py` |
| AM025 register DB | `../mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json` |
| aie-rt (official) | `../aie-rt/driver/src/` |
| aie-rt reginit | `../aie-rt/driver/src/global/xaiemlgbl_reginit.c` |
| aie-rt events | `../aie-rt/driver/src/events/xaie_events_aieml.h` |
| aie-rt DMA module | `../aie-rt/driver/src/dma/xaie_dma_aieml.h` |
| llvm-aie TableGen | `../llvm-aie/llvm/lib/Target/AIE/` |
| mlir-aie device model | `../mlir-aie/include/aie/Dialect/AIE/IR/AIETargetModel.h` |
| aietools (read-only) | `../aietools/data/aie_ml/lib/` |

## Graph Type Hierarchy

```
ArchModel
  +-- Architecture, DeviceGeneration, device_id, is_npu
  +-- ArrayTopology
  |     +-- columns, rows, shifts
  |     +-- TilePlacement[] (col, row, type, edges, mem_affinity)
  +-- DeviceConstants
  |     +-- lock range, address granularity, mem bases, DeviceProperties
  +-- TileTypeModel[]
        +-- TileKind, name, representative
        +-- InstanceCount (locks, bds, channels -- each Confirmed<u8>)
        +-- MemoryModel (size, banks, program memory)
        +-- DmaCapabilities (None -- awaiting aie-rt)
        +-- PortBundle[] (switchbox + shim_mux)
        +-- ModuleModel[]
              +-- ModuleKind
              +-- RegisterModel[]
                    +-- name, offset, width, reset_value
                    +-- SubsystemKind, Access
                    +-- FieldModel[] (name, BitRange, FieldSemantics)
```
