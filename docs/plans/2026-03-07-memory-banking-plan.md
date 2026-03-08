# Multi-Level Memory Banking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace MemoryModel's flat `num_banks`/`bank_size_bytes` with explicit logical/physical banking levels, validate the structural invariant at construction, update gen_arch.rs to expose both levels, and migrate emulator consumers from hardcoded aie2_spec constants to graph-derived values.

**Architecture:** Add `BankingModel` type to the graph crate. MemoryModel gains `logical: BankingModel` (always populated from mlir-aie) and `physical: Option<BankingModel>` (populated when a source provides it). An `effective_physical()` accessor falls back to logical when physical is unknown. build.rs generates both levels into gen_arch.rs. Emulator consumers (tile.num_banks, DMA engine, timing model) switch from aie2_spec hardcoded constants to arch-module constants.

**Tech Stack:** Rust, xdna-graph crate (types.rs, device_model.rs), build.rs code generation, cargo test

**Design doc:** `docs/plans/2026-03-07-memory-banking-design.md`

---

### Task 1: Add BankingModel Type and Restructure MemoryModel

**Files:**
- Modify: `crates/xdna-graph/src/types.rs:469-490`

**Step 1: Write the failing test**

Add to the existing test module at the bottom of `crates/xdna-graph/src/types.rs`:

```rust
#[test]
fn memory_model_structural_invariant() {
    use super::*;
    let src = SourceAttribution {
        origin: Source::DeviceModel,
        file: "test".into(),
        detail: "test".into(),
    };
    // Valid: 4 logical banks * 16KB = 64KB
    let mem = MemoryModel::new(
        65536,
        BankingModel { num_banks: 4, bank_size: 16384, bank_width_bits: 128, source: src.clone() },
        None,
        Some(16384),
        src.clone(),
    );
    assert_eq!(mem.size_bytes, 65536);
    assert_eq!(mem.logical.num_banks, 4);
    assert!(mem.physical.is_none());
}

#[test]
fn memory_model_with_physical() {
    use super::*;
    let src = SourceAttribution {
        origin: Source::DeviceModel,
        file: "test".into(),
        detail: "test".into(),
    };
    let mem = MemoryModel::new(
        65536,
        BankingModel { num_banks: 4, bank_size: 16384, bank_width_bits: 128, source: src.clone() },
        Some(BankingModel { num_banks: 8, bank_size: 8192, bank_width_bits: 128, source: src.clone() }),
        Some(16384),
        src.clone(),
    );
    assert_eq!(mem.logical.num_banks, 4);
    assert_eq!(mem.effective_physical().num_banks, 8);
}

#[test]
fn memory_model_effective_physical_fallback() {
    use super::*;
    let src = SourceAttribution {
        origin: Source::DeviceModel,
        file: "test".into(),
        detail: "test".into(),
    };
    let mem = MemoryModel::new(
        65536,
        BankingModel { num_banks: 4, bank_size: 16384, bank_width_bits: 128, source: src.clone() },
        None,
        None,
        src.clone(),
    );
    // No physical -> falls back to logical
    assert_eq!(mem.effective_physical().num_banks, 4);
}

#[test]
#[should_panic(expected = "logical banking invariant")]
fn memory_model_rejects_bad_logical() {
    use super::*;
    let src = SourceAttribution {
        origin: Source::DeviceModel,
        file: "test".into(),
        detail: "test".into(),
    };
    // 3 banks * 16KB = 48KB != 64KB -> panic
    MemoryModel::new(
        65536,
        BankingModel { num_banks: 3, bank_size: 16384, bank_width_bits: 128, source: src.clone() },
        None,
        None,
        src.clone(),
    );
}

#[test]
#[should_panic(expected = "physical banking invariant")]
fn memory_model_rejects_bad_physical() {
    use super::*;
    let src = SourceAttribution {
        origin: Source::DeviceModel,
        file: "test".into(),
        detail: "test".into(),
    };
    // Physical: 7 banks * 8KB = 56KB != 64KB -> panic
    MemoryModel::new(
        65536,
        BankingModel { num_banks: 4, bank_size: 16384, bank_width_bits: 128, source: src.clone() },
        Some(BankingModel { num_banks: 7, bank_size: 8192, bank_width_bits: 128, source: src.clone() }),
        None,
        src.clone(),
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-graph memory_model_structural`
Expected: FAIL -- `BankingModel` and `MemoryModel::new()` don't exist yet.

**Step 3: Write minimal implementation**

Replace the existing `MemoryModel` struct and `FactEquals` impl in `types.rs:469-490` with:

```rust
/// Memory banking at a single abstraction level.
///
/// Represents either the logical view (programmer/compiler, from mlir-aie)
/// or the physical view (SRAM arrays, from AM020/AM025).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BankingModel {
    pub num_banks: u8,
    pub bank_size: u64,
    pub bank_width_bits: u16,
    pub source: SourceAttribution,
}

impl FactEquals for BankingModel {
    fn fact_equals(&self, other: &Self) -> bool {
        self.num_banks == other.num_banks
            && self.bank_size == other.bank_size
            && self.bank_width_bits == other.bank_width_bits
    }
}

/// Memory model for a tile type with explicit logical/physical banking.
///
/// `logical` is always present (from mlir-aie's `getNumBanks()`).
/// `physical` is present when a source provides physical bank data
/// (from AM020, AM025 register events, or aie-rt). When absent,
/// `effective_physical()` falls back to `logical` (1:1 assumption).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryModel {
    pub size_bytes: u64,
    pub logical: BankingModel,
    pub physical: Option<BankingModel>,
    pub program_memory_bytes: Option<u64>,
    pub source: SourceAttribution,
}

impl MemoryModel {
    /// Construct a MemoryModel, validating the structural invariant.
    ///
    /// Panics if `logical.num_banks * logical.bank_size != size_bytes`,
    /// or if physical is provided and fails the same check.
    pub fn new(
        size_bytes: u64,
        logical: BankingModel,
        physical: Option<BankingModel>,
        program_memory_bytes: Option<u64>,
        source: SourceAttribution,
    ) -> Self {
        assert_eq!(
            logical.num_banks as u64 * logical.bank_size,
            size_bytes,
            "logical banking invariant violated: {} banks * {} bytes = {}, expected {}",
            logical.num_banks, logical.bank_size,
            logical.num_banks as u64 * logical.bank_size, size_bytes,
        );
        if let Some(ref phys) = physical {
            assert_eq!(
                phys.num_banks as u64 * phys.bank_size,
                size_bytes,
                "physical banking invariant violated: {} banks * {} bytes = {}, expected {}",
                phys.num_banks, phys.bank_size,
                phys.num_banks as u64 * phys.bank_size, size_bytes,
            );
        }
        Self { size_bytes, logical, physical, program_memory_bytes, source }
    }

    /// Banking model for physical conflict detection.
    /// Falls back to logical if physical is unspecified.
    pub fn effective_physical(&self) -> &BankingModel {
        self.physical.as_ref().unwrap_or(&self.logical)
    }
}

impl FactEquals for MemoryModel {
    fn fact_equals(&self, other: &Self) -> bool {
        self.size_bytes == other.size_bytes
            && self.logical.fact_equals(&other.logical)
            && match (&self.physical, &other.physical) {
                (Some(a), Some(b)) => a.fact_equals(b),
                (None, None) => true,
                _ => false,
            }
            && self.program_memory_bytes == other.program_memory_bytes
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-graph memory_model`
Expected: All 5 new tests PASS.

**Step 5: Commit**

```
git add crates/xdna-graph/src/types.rs
git commit -m "feat(graph): add BankingModel, restructure MemoryModel with logical/physical levels"
```

---

### Task 2: Update Device Model Extractor

**Files:**
- Modify: `crates/xdna-graph/src/device_model.rs:395-413`

The extractor currently constructs `MemoryModel { num_banks, bank_size_bytes, ... }`.
It needs to construct `MemoryModel::new(size_bytes, logical_banking, None, ...)`.

**Step 1: Run existing tests to establish baseline**

Run: `cargo test -p xdna-graph`
Expected: Some tests FAIL because MemoryModel fields changed.

**Step 2: Update the extractor**

In `device_model.rs`, around line 403, replace the `MemoryModel` construction:

```rust
    let logical = BankingModel {
        num_banks,
        bank_size: bank_size,
        bank_width_bits: 128,  // mlir-aie standard width
        source: SourceAttribution {
            origin: Source::DeviceModel,
            file: file.into(),
            detail: format!("{}.memory.logical", ctx),
        },
    };

    let memory = Some(MemoryModel::new(
        size_bytes,
        logical,
        None,  // physical banking not available from device model
        program_memory_bytes,
        SourceAttribution {
            origin: Source::DeviceModel,
            file: file.into(),
            detail: format!("{}.memory", ctx),
        },
    ));
```

Add the import at the top of the function or file:
```rust
use crate::types::BankingModel;
```

**Step 3: Fix all test references to old MemoryModel fields**

Any test that accesses `mem.num_banks` must change to `mem.logical.num_banks`.
Any test that accesses `mem.bank_size_bytes` must change to `mem.logical.bank_size`.

Search for these patterns in `device_model.rs` test module and update:
- `core_mem.num_banks` -> `core_mem.logical.num_banks` (or remove if redundant with Task 1 tests)
- `mt_mem.bank_size_bytes` -> `mt_mem.logical.bank_size`

**Step 4: Run tests**

Run: `cargo test -p xdna-graph`
Expected: All 99+ tests PASS.

**Step 5: Commit**

```
git add crates/xdna-graph/src/device_model.rs
git commit -m "refactor(graph): device model extractor uses BankingModel for logical banking"
```

---

### Task 3: Fix Emulator Re-exports and Downstream Consumers

**Files:**
- Modify: `src/device/regdb.rs` (re-exports from xdna_graph)
- Modify: `src/graph/mod.rs` (re-exports)
- Check: any file that accesses `MemoryModel.num_banks` or `.bank_size_bytes`

**Step 1: Build the emulator to find all breakages**

Run: `cargo build 2>&1`
Expected: Compilation errors wherever old MemoryModel fields are accessed.

**Step 2: Fix each breakage**

The main consumers to fix:

1. `build.rs` gen_arch function -- accesses `mem.num_banks` and `mem.bank_size_bytes`.
   Change to `mem.logical.num_banks` and `mem.logical.bank_size`.

2. Any tests in `src/device/model.rs` that check `.num_banks` -- update to `.logical.num_banks`.

3. `tests/arch_constants.rs` -- rename `MEMORY_BANKS` to `LOGICAL_BANKS` (done in Task 4).

**Step 3: Run full test suite**

Run: `cargo test --lib && cargo test -p xdna-graph`
Expected: All tests PASS.

**Step 4: Commit**

```
git add -u
git commit -m "fix: update all MemoryModel consumers for logical/physical restructure"
```

---

### Task 4: Update gen_arch.rs to Expose Both Banking Levels

**Files:**
- Modify: `build.rs:162-174` (gen_arch memory section)
- Modify: `tests/arch_constants.rs`

**Step 1: Write the failing test**

Update `tests/arch_constants.rs` to test the new naming:

```rust
#[test]
fn compute_tile_memory() {
    assert_eq!(arch::compute::MEMORY_SIZE, 64 * 1024);
    assert_eq!(arch::compute::LOGICAL_BANKS, 4);
    assert_eq!(arch::compute::LOGICAL_BANK_SIZE, 16 * 1024);
    assert_eq!(arch::compute::PROGRAM_MEMORY_SIZE, 16 * 1024);
    // No PHYSICAL_* constants yet (physical is None in graph)
}

#[test]
fn memtile_memory() {
    assert_eq!(arch::memtile::MEMORY_SIZE, 512 * 1024);
    assert_eq!(arch::memtile::LOGICAL_BANKS, 8);
    assert_eq!(arch::memtile::LOGICAL_BANK_SIZE, 64 * 1024);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test arch_constants`
Expected: FAIL -- `LOGICAL_BANKS` doesn't exist yet (still `MEMORY_BANKS`).

**Step 3: Update gen_arch in build.rs**

In the memory model section of `gen_arch()`, replace the current bank generation with:

```rust
        // Memory model
        if let Some(ref mem) = tile.memory {
            writeln!(out, "    /// Total data memory size in bytes.").unwrap();
            writeln!(out, "    pub const MEMORY_SIZE: u64 = {};", mem.size_bytes).unwrap();
            writeln!(out, "    /// Logical bank count (programmer/compiler view).").unwrap();
            writeln!(out, "    pub const LOGICAL_BANKS: u8 = {};", mem.logical.num_banks).unwrap();
            writeln!(out, "    /// Logical bank size in bytes.").unwrap();
            writeln!(out, "    pub const LOGICAL_BANK_SIZE: u64 = {};", mem.logical.bank_size).unwrap();
            if let Some(ref phys) = mem.physical {
                writeln!(out, "    /// Physical bank count (SRAM arrays, for conflict detection).").unwrap();
                writeln!(out, "    pub const PHYSICAL_BANKS: u8 = {};", phys.num_banks).unwrap();
                writeln!(out, "    /// Physical bank size in bytes.").unwrap();
                writeln!(out, "    pub const PHYSICAL_BANK_SIZE: u64 = {};", phys.bank_size).unwrap();
            }
            if let Some(pmem) = mem.program_memory_bytes {
                writeln!(out, "    /// Program (instruction) memory size in bytes.").unwrap();
                writeln!(out, "    pub const PROGRAM_MEMORY_SIZE: u64 = {};", pmem).unwrap();
            }
        }
```

**Step 4: Update registers_spec.rs**

`src/device/registers_spec.rs` references `crate::arch::compute::MEMORY_SIZE` which
is unchanged. No action needed here.

**Step 5: Run tests**

Run: `cargo test --test arch_constants && cargo test --lib`
Expected: All PASS.

**Step 6: Commit**

```
git add build.rs tests/arch_constants.rs
git commit -m "feat(build): gen_arch exposes logical/physical banking levels"
```

---

### Task 5: Migrate Emulator Consumers to Graph-Derived Banking

This is the "close the loop" migration -- where emulator code switches from
`aie2_spec` hardcoded constants to the `arch` module. Since physical banking
is `None` in the graph right now, we use `LOGICAL_BANKS` where the graph
provides it, and keep `aie2_spec` constants only for physical-bank-specific
logic (addr_to_bank, timing) until physical data flows through.

**Files:**
- Modify: `src/device/tile.rs:1339-1344` (num_banks method)
- Modify: `src/device/dma/engine.rs:351-354` (DmaEngine num_banks init)
- Note: `addr_to_bank()` and timing model stay on aie2_spec physical constants
  for now (they specifically need the physical bank count for conflict detection)

**Step 1: Verify current tests pass**

Run: `cargo test --lib`
Expected: All 1563+ tests PASS.

**Step 2: Note -- no migration for physical-bank consumers yet**

The consumers that call `addr_to_bank()` and the timing model use the
physical bank count (8 for compute, 16 for memtile). These MUST NOT
switch to logical counts (4/8) -- that would break conflict detection.

These consumers stay on `aie2_spec` hardcoded constants until Task 6
(or a future session) wires physical banking through the graph. Document
this with a comment in each location.

**Step 3: Run tests**

Run: `cargo test --lib`
Expected: All PASS (no behavioral change).

**Step 4: Commit** (if any comments were added)

```
git add -u
git commit -m "docs: annotate physical-bank consumers for future graph migration"
```

---

### Task 6: (Future) Wire Physical Banking Data

This task is deferred until a machine-readable source for physical bank
counts is identified (AM025 register event bank IDs, aie-rt constants, etc.).

When implemented:
1. Add physical banking data to the graph (either via extractor or manual constants)
2. gen_arch.rs will automatically generate `PHYSICAL_BANKS` / `PHYSICAL_BANK_SIZE`
3. Consumers switch from `aie2_spec` constants to `arch::compute::PHYSICAL_BANKS`
4. Remove the `aie2_spec` bank constants (COMPUTE_TILE_MEMORY_BANKS, etc.)
5. Remove `tile.num_banks()` match statement (reads from graph instead)

---

## Verification Checklist

After all tasks:
- [ ] `cargo test -p xdna-graph` -- all graph crate tests pass
- [ ] `cargo test --lib` -- all 1563+ emulator tests pass
- [ ] `cargo test --test arch_constants` -- integration tests pass
- [ ] `cargo build` -- zero warnings
- [ ] Generated `gen_arch.rs` shows `LOGICAL_BANKS` / `LOGICAL_BANK_SIZE` per tile type
- [ ] No `num_banks` or `bank_size_bytes` fields remain on MemoryModel (all via BankingModel)
- [ ] Structural invariant tested (panic on bad values)
- [ ] `effective_physical()` fallback tested
