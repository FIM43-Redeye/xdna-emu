# SubsystemModel Design

**Date:** 2026-03-08
**Status:** Approved, ready for implementation

## Goal

Materialize hardware subsystems as queryable objects in the ArchModel,
replacing the hand-written `RegisterModule` enum in the emulator with
data-driven register routing based on `SubsystemKind`.

## Background

`SubsystemKind` (16 variants: DMA, Lock, StreamSwitch, Processor, etc.)
already exists in `types.rs` and every register in the AM025 database is
tagged with one by `regdb_extractor.rs`. But subsystems are not materialized
as objects -- there is no struct you can query for "what offset range does
the DMA subsystem occupy on a compute tile?"

The emulator's `RegisterModule::from_offset()` is a hand-written 20-line
match that conflates tile type with subsystem (e.g., `DmaBufferDescriptor`
vs `MemTileDmaBufferDescriptor` are the same subsystem on different tiles).
This is fragile and blocks multi-architecture support.

## Design

### SubsystemModel struct

```rust
pub struct SubsystemModel {
    pub kind: SubsystemKind,
    /// Start offset within the tile's 20-bit address space (inclusive).
    pub offset_start: Confirmed<u32>,
    /// End offset (exclusive).
    pub offset_end: Confirmed<u32>,
    /// Indices into parent ModuleModel.registers for registers owned
    /// by this subsystem.
    pub register_indices: Vec<usize>,
}
```

Lives on `ModuleModel` as `subsystems: Vec<SubsystemModel>`.

Hierarchy: `TileTypeModel -> ModuleModel -> SubsystemModel`

### Granularity

One SubsystemModel per functional block. DMA is one unit (not split into
BD/Channel/Status sub-subsystems). The BD/channel/status distinction is
already captured by `BdSchema` and `DmaChannelSchema`. Can be split later
if needed.

### Population & Cross-Validation

Two independent sources confirm each address range via `Confirmed<T>`:

**AM025 (bottom-up):** `regdb_extractor.rs` already tags every register
with `SubsystemKind`. After all registers are populated on a `ModuleModel`,
group by subsystem kind, compute `min(offset)` and `max(offset + width)`
to derive the address range. `Confirmed::new(range, am025_source)`.

**aie-rt (top-down):** The existing `gcc -E` pipeline in `build.rs`
already extracts subsystem base addresses from `xaiemlgbl_reginit.c`:
- `gen_aiert_dma.rs`: `compute_dma::BD_BASE = 0x1D000`, etc.
- `gen_aiert_locks.rs`: `compute_locks::SET_VAL_BASE = 0x1F000`, etc.

These feed `.confirm()` calls. No new C header parsing needed.

Conflict between AM025 and aie-rt = immediate panic with detailed message
showing which source disagrees. This is the standard `Confirmed<T>` pattern
used throughout the ArchModel.

### Generated Output

New `gen_subsystems.rs` alongside existing `gen_arch.rs`:

```rust
pub mod compute {
    pub mod dma {
        pub const OFFSET_START: u32 = 0x1D000;
        pub const OFFSET_END: u32 = ...;
    }
    pub mod lock { ... }
    pub mod stream_switch { ... }
    pub mod processor { ... }
    pub mod data_memory { ... }
    pub mod program_memory { ... }
}
pub mod memtile { ... }
pub mod shim { ... }
```

### Consumer Migration

`RegisterModule` enum (13 variants) is sunset. Replaced by:

```rust
pub fn subsystem_from_offset(offset: u32, tile_kind: TileKind) -> SubsystemKind
```

Callers pass tile type (which they all already know from tile row).
Returns `SubsystemKind` directly from archspec -- no emulator-local
translation enum.

## Scope

**In scope:**
- `SubsystemModel` struct on `ModuleModel`
- AM025 bottom-up range derivation in `regdb_extractor.rs`
- Cross-validation against existing `gen_aiert_*` constants
- `gen_subsystems.rs` code generation
- `subsystem_from_offset(offset, TileKind)` router
- Sunset `RegisterModule` enum, migrate all callers
- Tests: construction, cross-validation, routing correctness

**Out of scope (deferred):**
- Sub-subsystem splitting (one unit per functional block)
- Graph edge query API (backburner)
- Moving registers off `ModuleModel` onto subsystems (next step after this)
- New aie-rt C header parsing (existing pipeline sufficient)
- AIE2P population (structure supports it, only AIE2 for now)
- GUI changes

## Files

| File | Change |
|------|--------|
| `crates/xdna-archspec/src/types.rs` | Add `SubsystemModel` struct |
| `crates/xdna-archspec/src/regdb_extractor.rs` | Group registers -> subsystem ranges |
| `crates/xdna-archspec/src/lib.rs` | `populate_subsystems()` with cross-validation |
| `build.rs` | Generate `gen_subsystems.rs` from ArchModel |
| `src/arch.rs` or include site | `include!("gen_subsystems.rs")` |
| `src/device/registers.rs` | Replace `RegisterModule` with `SubsystemKind` routing |
| Callers of `RegisterModule` | Pass tile type, use `SubsystemKind` |

## Test Strategy

- Unit: SubsystemModel construction, offset range invariants
- Cross-validation: AM025 ranges match aie-rt ranges (real data, not synthetic)
- Migration: port `test_register_module_from_offset` to new API
- Green `cargo test --lib` throughout

## Future

After this work, the natural next step is moving registers from
`ModuleModel` onto their owning `SubsystemModel` directly, eliminating
the index indirection. The graph edge infrastructure remains deferred
until there are concrete emulator consumers.
