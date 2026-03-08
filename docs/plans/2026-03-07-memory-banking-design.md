# Multi-Level Memory Banking Model

Date: 2026-03-07

## Problem

The `MemoryModel` in xdna-graph stores a single `num_banks` / `bank_size_bytes`
pair, but hardware memory banking operates at two abstraction levels:

- **Physical banks**: The actual SRAM arrays. Compute tiles have 8 physical
  banks of 8 KB (128-bit wide, AM020 Ch2). MemTiles have 16 banks of 32 KB
  (AM020 Ch5) or 8 banks of 64 KB (AM027 Ch5, 256-bit wide) -- the exact
  physical structure may vary by silicon revision.
- **Logical banks**: The programmer/compiler view. Compute tiles present 4
  logical banks of 16 KB (pairs of physical banks interleaved). MemTiles
  present 8 logical banks of 64 KB. mlir-aie's `getNumBanks()` returns the
  logical count.

The emulator needs physical banks for cycle-accurate bank conflict detection
(`addr_to_bank()`, timing model). The graph currently stores only the logical
view (from mlir-aie), while `aie2_spec.rs` hardcodes the physical view from
AM020.

## Design

### New Types

```rust
/// Memory banking at a single abstraction level.
pub struct BankingModel {
    pub num_banks: u8,
    pub bank_size: u64,
    pub bank_width_bits: u16,   // 128 for most, 256 for AM027 MemTile
    pub source: SourceAttribution,
}

/// Memory model with explicit physical/logical banking.
pub struct MemoryModel {
    pub size_bytes: u64,
    pub logical: BankingModel,              // always present (from mlir-aie)
    pub physical: Option<BankingModel>,     // when available (from AM020/AM025)
    pub program_memory_bytes: Option<u64>,
    pub source: SourceAttribution,
}
```

### Structural Invariant

Validated at construction time (panic on violation):

```
logical.num_banks * logical.bank_size == size_bytes
physical.num_banks * physical.bank_size == size_bytes  (when Some)
```

This is a **law** (must hold for any valid memory description), not a **fact**
(which might vary by architecture). It expresses "these are views of the same
memory" without assuming any particular pairing ratio.

### Fallback Behavior

When `physical` is `None`, consumers that need physical bank information use
`logical` as the fallback (1:1 assumption -- each logical bank is one physical
bank). This is the accessor pattern:

```rust
impl MemoryModel {
    /// Banking model for physical conflict detection.
    /// Falls back to logical if physical is unspecified.
    pub fn effective_physical(&self) -> &BankingModel {
        self.physical.as_ref().unwrap_or(&self.logical)
    }
}
```

### FactEquals

`FactEquals` compares logical-to-logical and physical-to-physical independently.
Two `MemoryModel` values are fact-equal if:
- `size_bytes` matches
- `logical` banking matches (num_banks, bank_size, bank_width_bits)
- `physical` banking matches (both None, or both Some with matching values)
- `program_memory_bytes` matches

### Sourcing

| Level | Source | Populated By |
|-------|--------|-------------|
| Logical | mlir-aie `getNumBanks()` | `device_model.rs` extractor (existing) |
| Physical | AM020/AM025/aie-rt (when available) | Future: register event parsing, manual |

Initially, only logical banking is populated (from the existing device model
path). Physical banking starts as `None` for all tile types. This is safe
because the 1:1 fallback preserves current behavior where mlir-aie values
flow through.

### Consumer Migration

Consumers that currently use `aie2_spec.rs` hardcoded physical bank constants:

| Consumer | Current Source | New Source |
|----------|---------------|-----------|
| `addr_to_bank()` | `aie2_spec::COMPUTE_TILE_MEMORY_BANKS` (8) | `arch::compute::PHYSICAL_BANKS` (when available) |
| `banks_for_access()` | Same | Same |
| Timing model `NUM_BANKS` | `aie2_spec::COMPUTE_TILE_MEMORY_BANKS` (8) | Same |
| DMA engine `num_banks` | Tile type match on aie2_spec constants | `effective_physical().num_banks` |
| `gen_arch.rs` | Logical only | Both levels |

Migration happens incrementally: as physical banking data is wired in, each
consumer switches from `aie2_spec` constants to graph-derived values.

### Generated Constants (gen_arch.rs)

The generated `arch` module exposes both levels:

```rust
pub mod compute {
    pub const MEMORY_SIZE: u64 = 65536;
    pub const LOGICAL_BANKS: u8 = 4;
    pub const LOGICAL_BANK_SIZE: u64 = 16384;
    // Physical banking (when available from graph):
    // pub const PHYSICAL_BANKS: u8 = 8;
    // pub const PHYSICAL_BANK_SIZE: u64 = 8192;
}
```

Physical constants are only generated when `physical` is `Some` in the
ArchModel. Consumers that need physical values today continue using
`aie2_spec.rs` until the physical data flows through the graph.

## Values

### Compute Tile (64 KB)

| Level | Banks | Bank Size | Width | Source |
|-------|-------|-----------|-------|--------|
| Logical | 4 | 16 KB | 128-bit | mlir-aie `getNumBanks()` |
| Physical | 8 | 8 KB | 128-bit | AM020 Ch2 |

### MemTile (512 KB)

| Level | Banks | Bank Size | Width | Source |
|-------|-------|-----------|-------|--------|
| Logical | 8 | 64 KB | 128-bit | mlir-aie `getNumBanks()` |
| Physical | 16 | 32 KB | 128-bit | AM020 Ch5 |

Note: AM027 describes MemTile as 8 banks of 256-bit width. This may reflect
a hardware revision (AIE2P) or a different abstraction level. The physical
banking model can capture either representation.

### Shim Tile

Shim tiles report 4 banks / 16 KB from the device model. Physical banking
is likely not applicable (DMA address space, not SRAM). Physical stays `None`.

## Non-Goals

- Automatic derivation of physical-from-logical (the ratio varies by tile type)
- AM020 text parsing (physical values will be manually sourced or from register
  event counts until a machine-readable source is identified)
- Changing `addr_to_bank()` semantics (it already takes `num_banks` as a
  parameter; the caller just needs to pass the right value)
