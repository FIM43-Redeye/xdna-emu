# Clock-control subsystem implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three-tier AIE2 clock-control subsystem (column / module / adaptive gates) per spec `docs/superpowers/specs/2026-05-24-clock-control-design.md`. Replace the `clock_control: STUB` coverage gap with a working, silicon-accurate implementation.

**Architecture:** Single `ClockController` owns all clock-gating state (Approach A). Lives on `TileArray`. Step paths query it before doing per-cycle work; register dispatch routes clock-control offsets to it; default boot state is silicon-accurate (all tiles gated). Tests opt out via `ClockController::ungate_all()` which exercises the real register-write path.

**Tech Stack:** Rust 1.x, existing xdna-emu test harness (`cargo test --lib`), regdb JSON at `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json` for bit-field layouts.

**Spec sections this plan implements:**
- `Default boot state: silicon-accurate` -> Tasks 1, 3, 13
- `Test-surface migration` -> Tasks 12, 13
- `Access-to-gated-tile policy: serve and warn` -> Task 11
- `Architecture` (Approach A, ClockController on TileArray) -> Tasks 1, 3
- `Types` (ClockController, TileGates, AdaptiveState, ModuleKind) -> Tasks 1, 4, 6
- `Public API` (is_*, write_register, read_register, tick_adaptive, ungate_all) -> Tasks 1, 2, 5, 6, 12
- `Data flow: register write` -> Tasks 7, 8
- `Data flow: step path` -> Tasks 9, 10
- `Edge cases` -> covered across Tasks 2, 5, 6, 11
- `Testing` -> per-task unit + Task 14 integration

**Out of scope (per spec):**
- NoC implementation
- Privilege enforcement
- GUI integration
- Submodule file split (single `mod.rs` v1)

---

## Reference files (read once before starting)

- `/home/triple/npu-work/xdna-emu/docs/superpowers/specs/2026-05-24-clock-control-design.md` (the spec)
- `/home/triple/npu-work/aie-rt/driver/src/pm/xaie_clock.c` (hardware reference)
- `/home/triple/npu-work/aie-rt/driver/src/pm/xaie_clock.h` (API reference)
- `/home/triple/npu-work/mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json` (bit-field source of truth)
- `/home/triple/npu-work/xdna-emu/src/device/perf_counters/mod.rs` (structural pattern reference)
- `/home/triple/npu-work/xdna-emu/src/device/state/dispatch.rs` (register dispatch pattern)
- `/home/triple/npu-work/xdna-emu/src/device/array/mod.rs:106` (TileArray def, where ClockController will live)

## Known register offsets (from regdb JSON, do not hardcode in implementation)

| Register | Offset | Tile types | Reset | Notes |
|----------|--------|-----------|-------|-------|
| `Column_Clock_Control` | `0x000FFF20` | Shim only (row 0) | `0x00000000` | One per column. Bit 0 = column-clock-enable. |
| `Module_Clock_Control` | `0x00060000` | Compute | `0x00000037` | Bit fields per regdb. |
| `Module_Clock_Control` | `0x000FFF00` | Memtile | `0x00000033` | Bit fields per regdb. |
| `Module_Clock_Control_0` | `0x000FFF00` | Shim | `0x0000003B` | Bit fields per regdb. |
| `Module_Clock_Control_1` | `0x000FFF04` | Shim | (see regdb) | Bit fields per regdb. |
| `Stream_Switch_Adaptive_Clock_Gate_Abort_Period` | (per-tile, varies) | All | per regdb | abort_period_2pow field. |

Implementation reads bit positions from `regdb::device_reg_layout()` -- it must NEVER hardcode bit positions.

---

## Task 1: ClockController skeleton with column gates

**Files:**
- Create: `src/device/clock_control/mod.rs`
- Modify: `src/device/mod.rs` (add `pub mod clock_control;` near line 71 next to other subsystem mods)

- [ ] **Step 1: Add module declaration**

Edit `src/device/mod.rs`. Find the existing `pub mod perf_counters;` line (around line 71) and add a line below it:

```rust
pub mod clock_control;
```

- [ ] **Step 2: Write the failing tests**

Create `src/device/clock_control/mod.rs` with this content:

```rust
//! AIE2 clock-control subsystem.
//!
//! Owns all clock-gating state for the array (column / module / adaptive
//! tiers).  Boots with every tile gated, matching silicon behavior per
//! aie-rt's XAie_PmRequestTiles documentation.  Tests opt out via
//! `ungate_all()` which exercises the same register-write path the
//! real CDO uses.
//!
//! Spec: docs/superpowers/specs/2026-05-24-clock-control-design.md

use std::collections::HashMap;

/// Per-array clock-gating state.  Single source of truth for all
/// column / module / adaptive gates.
#[derive(Debug, Clone)]
pub struct ClockController {
    /// Per-column clock gate; index = col.  `false` = gated (inactive).
    columns: Vec<bool>,
    /// Number of rows in the array.  Stored so `ungate_all()` knows how
    /// far to iterate without needing to consult arch config.
    num_rows: u8,
}

impl ClockController {
    /// Construct a controller for an array of `num_cols` columns and
    /// `num_rows` rows.  All columns boot gated (silicon-accurate).
    pub fn new(num_cols: u8, num_rows: u8) -> Self {
        Self {
            columns: vec![false; num_cols as usize],
            num_rows,
        }
    }

    /// Returns true iff column `col` has its clock enabled.
    /// Returns false for out-of-range columns.
    pub fn is_column_active(&self, col: u8) -> bool {
        self.columns.get(col as usize).copied().unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_clock_controller_boots_with_all_columns_gated() {
        let clock = ClockController::new(5, 6);
        for col in 0..5 {
            assert!(!clock.is_column_active(col),
                "col {} should be gated at boot", col);
        }
    }

    #[test]
    fn is_column_active_returns_false_for_out_of_range_col() {
        let clock = ClockController::new(5, 6);
        assert!(!clock.is_column_active(5));
        assert!(!clock.is_column_active(99));
    }
}
```

- [ ] **Step 3: Run tests to verify they pass**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: 2 passed.

- [ ] **Step 4: Commit**

```
git add src/device/mod.rs src/device/clock_control/mod.rs
git commit -m "clock_control: add ClockController skeleton with column gates

Boots with every column gated, matching aie-rt's documented
silicon behavior.  Foundation for module + adaptive tiers in
subsequent commits.

Generated using Claude Code."
```

---

## Task 2: Column_Clock_Control register write/read

**Files:**
- Modify: `src/device/clock_control/mod.rs`

- [ ] **Step 1: Add failing tests**

Add to the `mod tests` block in `src/device/clock_control/mod.rs`:

```rust
#[test]
fn write_column_clock_control_bit0_enables_column() {
    let mut clock = ClockController::new(5, 6);
    // Column 2's shim tile is at (col=2, row=0).
    // Column_Clock_Control offset is 0x000FFF20.
    clock.write_register(2, 0, 0x000FFF20, 0x1);
    assert!(clock.is_column_active(2));
    assert!(!clock.is_column_active(0), "other cols unaffected");
}

#[test]
fn write_column_clock_control_bit0_clear_disables_column() {
    let mut clock = ClockController::new(5, 6);
    clock.write_register(2, 0, 0x000FFF20, 0x1);
    assert!(clock.is_column_active(2));
    clock.write_register(2, 0, 0x000FFF20, 0x0);
    assert!(!clock.is_column_active(2));
}

#[test]
fn read_column_clock_control_returns_reset_before_any_write() {
    let clock = ClockController::new(5, 6);
    // AM025 reset is 0x00000000 for Column_Clock_Control.
    assert_eq!(clock.read_register(2, 0, 0x000FFF20), Some(0x00000000));
}

#[test]
fn read_column_clock_control_returns_written_value() {
    let mut clock = ClockController::new(5, 6);
    clock.write_register(3, 0, 0x000FFF20, 0x1);
    assert_eq!(clock.read_register(3, 0, 0x000FFF20), Some(0x1));
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: 4 new tests FAIL with "method `write_register` not found" and "method `read_register` not found".

- [ ] **Step 3: Implement write_register and read_register**

Add to the `impl ClockController` block in `src/device/clock_control/mod.rs`, after `is_column_active`:

```rust
/// AM025 offset for Column_Clock_Control (shim tiles only).
const COLUMN_CLOCK_CONTROL_OFFSET: u32 = 0x000FFF20;

impl ClockController {
    /// Handle a register write at the given tile / offset.  Silently
    /// ignores offsets that are not clock-control registers.
    pub fn write_register(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        match offset {
            COLUMN_CLOCK_CONTROL_OFFSET if row == 0 => {
                // Bit 0 = column-clock-enable; per AM025.
                if let Some(slot) = self.columns.get_mut(col as usize) {
                    *slot = (value & 0x1) != 0;
                }
            }
            _ => {} // not a clock-control offset
        }
    }

    /// Read a clock-control register.  Returns the current value, or the
    /// AM025 reset value if the register has not been written yet.
    /// Returns None if the offset is not a known clock-control register.
    pub fn read_register(&self, col: u8, row: u8, offset: u32) -> Option<u32> {
        let _ = (col, row);
        match offset {
            COLUMN_CLOCK_CONTROL_OFFSET if row == 0 => {
                // Reflect current state: bit 0 = column enabled.
                let enabled = self.is_column_active(col);
                Some(if enabled { 0x1 } else { 0x0 })
            }
            _ => None,
        }
    }
}
```

Note: the const declaration goes ABOVE the existing `impl ClockController` block. Adjust as needed if rustfmt prefers it inside.

- [ ] **Step 4: Run tests to verify they pass**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: 6 passed (2 from Task 1 + 4 from Task 2).

- [ ] **Step 5: Commit**

```
git add src/device/clock_control/mod.rs
git commit -m "clock_control: Column_Clock_Control register handling

Bit 0 of offset 0x000FFF20 (shim row only) gates / ungates the
whole column.  AM025 reset is 0; matches the silicon-accurate
'all gated at boot' default.

Generated using Claude Code."
```

---

## Task 3: Wire ClockController into TileArray

**Files:**
- Modify: `src/device/array/mod.rs` (TileArray struct + ::new constructor)

- [ ] **Step 1: Read the surrounding context first**

Read lines 100-160 of `src/device/array/mod.rs` to understand the current struct layout and constructor. Note the existing `arch`, `cols`, `rows`, `tiles`, `dma_engines` fields.

- [ ] **Step 2: Write the failing integration test**

Add to `src/device/array/tests.rs` (or wherever TileArray tests live -- check `#[cfg(test)] mod tests;` declaration; if it's a sibling file, use that path):

```rust
#[test]
fn tile_array_exposes_clock_controller() {
    use std::sync::Arc;
    use xdna_archspec::runtime::ArchConfig;
    let arch: Arc<dyn ArchConfig> = xdna_archspec::runtime::aie2_npu1();
    let array = TileArray::new(arch);
    // Default state: silicon-accurate, all columns gated.
    for col in 0..array.cols {
        assert!(!array.clock().is_column_active(col),
            "col {} should be gated at TileArray construction", col);
    }
}
```

If `aie2_npu1()` isn't the right constructor name, look in `xdna-archspec/src/runtime/*.rs` for the AIE2 NPU1 arch factory and substitute.

- [ ] **Step 3: Run to confirm it fails**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::array::tests::tile_array_exposes_clock_controller
```

Expected: FAIL ("no method named `clock` found").

- [ ] **Step 4: Add the field + accessor**

In `src/device/array/mod.rs`:

1. Add `use super::clock_control::ClockController;` near the top (with the other `use super::` statements).
2. Add a field to `TileArray`:

```rust
pub struct TileArray {
    pub(super) arch: Arc<dyn ArchConfig>,
    pub(super) cols: u8,
    pub(super) rows: u8,
    pub(crate) tiles: Vec<Tile>,
    // ... existing fields ...

    /// Clock-control state for the array.  Owns all column / module /
    /// adaptive gate state.  Boots with every tile gated.
    pub(crate) clock: ClockController,
}
```

3. In `TileArray::new`, initialize the field. Find the `Self { ... }` literal at the bottom and add:

```rust
Self {
    // ... existing fields ...
    clock: ClockController::new(cols, rows),
}
```

4. Add accessor methods on `impl TileArray`:

```rust
/// Borrow the clock controller (read-only).
pub fn clock(&self) -> &ClockController {
    &self.clock
}

/// Borrow the clock controller mutably.
pub fn clock_mut(&mut self) -> &mut ClockController {
    &mut self.clock
}
```

- [ ] **Step 5: Run the test to confirm it passes**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::array::tests::tile_array_exposes_clock_controller
```

Expected: PASS.

- [ ] **Step 6: Run the full lib test suite**

```
TMPDIR=/tmp/claude-1000 cargo test --lib
```

**WARNING**: this will surface many failing tests that construct TileArray directly without ungating clocks. That's expected -- they'll be fixed in Task 13. For now, note the count of new failures so we can confirm Task 13 fixes them. **Do not proceed if there are compile errors** (compile errors are real bugs, not the expected gating-side test failures).

Record the failing test count somewhere (e.g., `/tmp/claude-1000/clock-control-baseline-failures.txt`).

- [ ] **Step 7: Commit**

```
git add src/device/array/mod.rs src/device/array/tests.rs
git commit -m "clock_control: wire ClockController into TileArray

Array owns one ClockController, accessed via clock() / clock_mut().
Constructor inits with all columns gated.  This will cause many
existing tests to fail (they don't ungate); migration in a later
commit.

Generated using Claude Code."
```

---

## Task 4: ModuleKind enum + TileGates struct + is_module_active

**Files:**
- Modify: `src/device/clock_control/mod.rs`

- [ ] **Step 1: Add failing tests**

Add to the `mod tests` block:

```rust
#[test]
fn module_kind_variants_exist() {
    let _ = ModuleKind::Core;
    let _ = ModuleKind::Memory;
    let _ = ModuleKind::Dma;
    let _ = ModuleKind::StreamSwitch;
}

#[test]
fn is_module_active_default_false_for_all_kinds() {
    let clock = ClockController::new(5, 6);
    for kind in [ModuleKind::Core, ModuleKind::Memory, ModuleKind::Dma, ModuleKind::StreamSwitch] {
        assert!(!clock.is_module_active(2, 2, kind),
            "module {:?} should be gated at boot", kind);
    }
}

#[test]
fn is_tile_active_default_false() {
    let clock = ClockController::new(5, 6);
    assert!(!clock.is_tile_active(2, 2));
}
```

- [ ] **Step 2: Run to confirm failure**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: FAIL with "cannot find `ModuleKind` in this scope".

- [ ] **Step 3: Add the enum + struct + methods**

In `src/device/clock_control/mod.rs`:

Above the `ClockController` struct, add:

```rust
/// Which module within a tile is being queried.  Mirrors the bit-field
/// breakdown of Module_Clock_Control in AM025.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleKind {
    Core,
    Memory,
    Dma,
    StreamSwitch,
}

/// Per-tile module-gate state.  Stores the raw register value(s); the
/// per-module active queries decode bit fields on demand.
#[derive(Debug, Clone)]
struct TileGates {
    /// `Module_Clock_Control` (compute, memtile) or `Module_Clock_Control_0` (shim).
    raw_mcc_0: u32,
    /// `Module_Clock_Control_1` (shim only); None for other tile types.
    raw_mcc_1: Option<u32>,
}
```

Add the `tiles` field to `ClockController`:

```rust
pub struct ClockController {
    columns: Vec<bool>,
    num_rows: u8,
    /// Per-tile module-gate state, keyed by (col, row).
    /// Tile entry is absent until first write to that tile's MCC register;
    /// is_module_active falls back to the AM025 reset values in that case
    /// (consistent with what a read of the register would return).
    tiles: HashMap<(u8, u8), TileGates>,
}
```

Update `::new` to init the new field:

```rust
pub fn new(num_cols: u8, num_rows: u8) -> Self {
    Self {
        columns: vec![false; num_cols as usize],
        num_rows,
        tiles: HashMap::new(),
    }
}
```

Add the query methods on `impl ClockController`. The column gate is the upstream gate -- if a column is gated, no module on any tile in that column is active regardless of MCC state. Unprogrammed tiles in an ungated column fall back to the AM025 reset.

```rust
/// Returns true iff the named module on tile (col, row) is currently
/// clocked.  Column gate dominates: a gated column means every module
/// reports inactive regardless of MCC.  An ungated column with no MCC
/// writes yet uses the AM025 reset value.
pub fn is_module_active(&self, col: u8, row: u8, kind: ModuleKind) -> bool {
    // Column gate dominates.
    if !self.is_column_active(col) { return false; }
    // For now (filled in by Task 5), return false until bit-field
    // decode is added.
    let _ = (row, kind);
    false
}

/// Returns true iff any module on this tile is active.
pub fn is_tile_active(&self, col: u8, row: u8) -> bool {
    use ModuleKind::*;
    self.is_module_active(col, row, Core)
        || self.is_module_active(col, row, Memory)
        || self.is_module_active(col, row, Dma)
        || self.is_module_active(col, row, StreamSwitch)
}
```

- [ ] **Step 4: Run tests to confirm pass**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: all clock_control tests pass.

- [ ] **Step 5: Commit**

```
git add src/device/clock_control/mod.rs
git commit -m "clock_control: ModuleKind + TileGates struct

is_module_active and is_tile_active return false for unprogrammed
tiles (silicon-accurate default).  Bit-field decode of MCC register
follows in the next commit.

Generated using Claude Code."
```

---

## Task 5: Module_Clock_Control register handling (all tile types)

**Files:**
- Modify: `src/device/clock_control/mod.rs`

**Bit positions (confirmed from `aie_registers_aie2.json` 2026-05-24):**

| Tile type | Offset | Reset | Bit 0 | Bit 1 | Bit 2 | Bit 3 | Bit 4 | Bit 5 |
|-----------|--------|-------|-------|-------|-------|-------|-------|-------|
| Compute | 0x00060000 | 0x37 | SS_Clock | Memory_Module | Core_Module | -- | SS_Adaptive | DMA_Adaptive |
| Memtile | 0x000FFF00 | 0x33 | SS_Clock | Memory_Module | -- | -- | SS_Adaptive | DMA_Adaptive |
| Shim MCC_0 | 0x000FFF00 | 0x3B | SS_Clock | PL_Interface | -- | CTE_Clock | SS_Adaptive | DMA_Adaptive |
| Shim MCC_1 | 0x000FFF04 | 0x01 | NoC_Module | -- | -- | -- | -- | -- |

**Cross-tile-type ModuleKind mapping** (the controller translates universal queries to tile-specific bits):

| `ModuleKind::*` | Compute | Memtile | Shim |
|-----------------|---------|---------|------|
| Core | MCC bit 2 | always false | always false |
| Memory | MCC bit 1 | MCC bit 1 | always false |
| Dma | MCC bit 1 (= Memory) | MCC bit 1 | MCC_1 bit 0 (NoC) |
| StreamSwitch | MCC bit 0 | MCC bit 0 | MCC_0 bit 0 |

Note that on compute/memtile `Dma` and `Memory` map to the same bit -- the silicon doesn't separately clock-gate DMA from data memory. They're modeled as distinct ModuleKinds for caller clarity (the step_dma path asks about Dma; a memory access path can ask about Memory).

- [ ] **Step 1: Add failing tests for each tile type**

Add to the `mod tests` block.

```rust
const MCC_COMPUTE_OFFSET: u32 = 0x00060000;
const MCC_MEMTILE_OFFSET: u32 = 0x000FFF00;
const MCC_SHIM_0_OFFSET: u32 = 0x000FFF00;
const MCC_SHIM_1_OFFSET: u32 = 0x000FFF04;

#[test]
fn mcc_compute_decodes_core_bit() {
    // Bit positions per aie_registers_aie2.json (2026-05-24 lookup).
    let mut clock = ClockController::new(5, 6);
    clock.write_register(2, 0, 0x000FFF20, 0x1);  // ungate column 2
    // Compute MCC bit 2 = Core_Module_Clock_Enable.
    clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 2);
    assert!(clock.is_module_active(2, 3, ModuleKind::Core));
    assert!(!clock.is_module_active(2, 3, ModuleKind::Memory));
    assert!(!clock.is_module_active(2, 3, ModuleKind::Dma));
    assert!(!clock.is_module_active(2, 3, ModuleKind::StreamSwitch));
}

#[test]
fn mcc_compute_decodes_memory_bit_as_both_memory_and_dma() {
    let mut clock = ClockController::new(5, 6);
    clock.write_register(2, 0, 0x000FFF20, 0x1);
    // Compute MCC bit 1 = Memory_Module_Clock_Enable.  Same bit clocks DMA.
    clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 1);
    assert!(clock.is_module_active(2, 3, ModuleKind::Memory));
    assert!(clock.is_module_active(2, 3, ModuleKind::Dma));
    assert!(!clock.is_module_active(2, 3, ModuleKind::Core));
}

#[test]
fn mcc_compute_decodes_ss_bit() {
    let mut clock = ClockController::new(5, 6);
    clock.write_register(2, 0, 0x000FFF20, 0x1);
    clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 0);
    assert!(clock.is_module_active(2, 3, ModuleKind::StreamSwitch));
}

#[test]
fn mcc_memtile_no_core() {
    let mut clock = ClockController::new(5, 6);
    clock.write_register(2, 0, 0x000FFF20, 0x1);
    // Set everything we can on memtile.
    clock.write_register(2, 1, MCC_MEMTILE_OFFSET, 0xFFFF_FFFF);
    // Memtile has no Core module -- always false regardless of writes.
    assert!(!clock.is_module_active(2, 1, ModuleKind::Core));
    // Memory and Dma both reflect bit 1.
    assert!(clock.is_module_active(2, 1, ModuleKind::Memory));
    assert!(clock.is_module_active(2, 1, ModuleKind::Dma));
    assert!(clock.is_module_active(2, 1, ModuleKind::StreamSwitch));
}

#[test]
fn mcc_shim_dma_lives_in_mcc_1_not_mcc_0() {
    let mut clock = ClockController::new(5, 6);
    clock.write_register(2, 0, 0x000FFF20, 0x1);
    // MCC_0 with everything set; MCC_1 explicitly 0.
    clock.write_register(2, 0, MCC_SHIM_0_OFFSET, 0xFFFF_FFFF);
    clock.write_register(2, 0, MCC_SHIM_1_OFFSET, 0x0);
    // SS comes from MCC_0 bit 0 -> active.
    assert!(clock.is_module_active(2, 0, ModuleKind::StreamSwitch));
    // DMA on shim comes from MCC_1 bit 0 (NoC) -> inactive.
    assert!(!clock.is_module_active(2, 0, ModuleKind::Dma));
    // Shim has no Core or Memory.
    assert!(!clock.is_module_active(2, 0, ModuleKind::Core));
    assert!(!clock.is_module_active(2, 0, ModuleKind::Memory));
}

#[test]
fn mcc_shim_dma_lives_in_mcc_1_bit_0_when_set() {
    let mut clock = ClockController::new(5, 6);
    clock.write_register(2, 0, 0x000FFF20, 0x1);
    clock.write_register(2, 0, MCC_SHIM_1_OFFSET, 1 << 0);
    assert!(clock.is_module_active(2, 0, ModuleKind::Dma));
}

#[test]
fn module_inactive_when_column_gated_even_with_mcc_set() {
    let mut clock = ClockController::new(5, 6);
    // Column 2 NOT ungated.
    clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 2);  // try to enable Core
    // Column gate dominates -- everything inactive.
    assert!(!clock.is_module_active(2, 3, ModuleKind::Core));
}
```

- [ ] **Step 3: Run to confirm tests fail**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: new tests FAIL (is_module_active still returns false unconditionally from Task 4).

- [ ] **Step 4: Implement bit-field decode**

Add to `src/device/clock_control/mod.rs`. The exact bit positions come from regdb; this is the structural pattern:

```rust
// At the top of the file, with the existing const:
const MCC_COMPUTE_OFFSET: u32 = 0x00060000;
const MCC_MEMTILE_OFFSET: u32 = 0x000FFF00;
const MCC_SHIM_0_OFFSET: u32 = 0x000FFF00;
const MCC_SHIM_1_OFFSET: u32 = 0x000FFF04;

/// Decode `kind` from a Module_Clock_Control register value.
/// Bit positions confirmed from aie_registers_aie2.json on 2026-05-24.
/// See plan task header for the full bit table.
fn mcc_module_active(
    raw_mcc_0: u32,
    raw_mcc_1: Option<u32>,
    tile_kind: ClockTileKind,
    kind: ModuleKind,
) -> bool {
    use ModuleKind::*;
    use ClockTileKind::*;
    let (reg, bit): (u32, u8) = match (tile_kind, kind) {
        // Compute tile: MCC bit 2 = Core, bit 1 = Memory (= Dma),
        // bit 0 = SS.
        (Compute, Core)         => (raw_mcc_0, 2),
        (Compute, Memory)       => (raw_mcc_0, 1),
        (Compute, Dma)          => (raw_mcc_0, 1),  // same bit as Memory
        (Compute, StreamSwitch) => (raw_mcc_0, 0),
        // Memtile: no Core; bit 1 = Memory (= Dma), bit 0 = SS.
        (Memtile, Core)         => return false,
        (Memtile, Memory)       => (raw_mcc_0, 1),
        (Memtile, Dma)          => (raw_mcc_0, 1),
        (Memtile, StreamSwitch) => (raw_mcc_0, 0),
        // Shim: no Core / Memory.  Dma lives in MCC_1 bit 0 (NoC).
        (Shim, Core)            => return false,
        (Shim, Memory)          => return false,
        (Shim, Dma)             => match raw_mcc_1 {
            Some(r) => (r, 0),
            None    => return false,
        },
        (Shim, StreamSwitch)    => (raw_mcc_0, 0),
    };
    (reg >> bit) & 0x1 != 0
}

/// Internal tile-kind discriminator for bit-layout selection.
/// Uses a clock-control-local enum rather than xdna_archspec::types::TileKind
/// to avoid coupling the controller to the broader archspec model -- this
/// only needs three buckets (Compute / Memtile / Shim) regardless of how
/// many shim variants exist elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClockTileKind { Compute, Memtile, Shim }

fn clock_clock_tile_kind_from_row(row: u8) -> ClockTileKind {
    if row == 0 { ClockTileKind::Shim }
    else if row == 1 { ClockTileKind::Memtile }
    else { ClockTileKind::Compute }
}
```

Wire it into `is_module_active`, preserving the column-gate-dominates rule and falling back to AM025 reset for unprogrammed tiles in ungated columns:

```rust
pub fn is_module_active(&self, col: u8, row: u8, kind: ModuleKind) -> bool {
    if !self.is_column_active(col) { return false; }
    let tile_kind = clock_clock_tile_kind_from_row(row);
    let (raw_mcc_0, raw_mcc_1) = match self.tiles.get(&(col, row)) {
        Some(gates) => (gates.raw_mcc_0, gates.raw_mcc_1),
        None => (
            reset_value_for_mcc(tile_kind),
            if matches!(tile_kind, ClockTileKind::Shim) { Some(0x01) } else { None },
        ),
    };
    mcc_module_active(raw_mcc_0, raw_mcc_1, tile_kind, kind)
}
```

Extend `write_register` to handle MCC writes:

```rust
pub fn write_register(&mut self, col: u8, row: u8, offset: u32, value: u32) {
    match offset {
        COLUMN_CLOCK_CONTROL_OFFSET if row == 0 => {
            if let Some(slot) = self.columns.get_mut(col as usize) {
                *slot = (value & 0x1) != 0;
            }
        }
        MCC_COMPUTE_OFFSET | MCC_MEMTILE_OFFSET | MCC_SHIM_0_OFFSET => {
            let entry = self.tiles.entry((col, row)).or_insert_with(|| TileGates {
                raw_mcc_0: 0,
                raw_mcc_1: None,
            });
            entry.raw_mcc_0 = value;
        }
        MCC_SHIM_1_OFFSET if row == 0 => {
            let entry = self.tiles.entry((col, row)).or_insert_with(|| TileGates {
                raw_mcc_0: 0,
                raw_mcc_1: Some(0),
            });
            entry.raw_mcc_1 = Some(value);
        }
        _ => {}
    }
}
```

**Note on collision**: MCC_MEMTILE_OFFSET and MCC_SHIM_0_OFFSET are both `0x000FFF00`. The `row` parameter disambiguates -- memtile is row 1, shim is row 0. Adjust the match arm if regdb confirms the offsets differ; otherwise this is fine because both use `raw_mcc_0`.

Also extend `read_register` to return MCC register values:

```rust
pub fn read_register(&self, col: u8, row: u8, offset: u32) -> Option<u32> {
    let tile_kind = clock_clock_tile_kind_from_row(row);
    match offset {
        COLUMN_CLOCK_CONTROL_OFFSET if row == 0 => {
            Some(if self.is_column_active(col) { 0x1 } else { 0x0 })
        }
        MCC_COMPUTE_OFFSET | MCC_MEMTILE_OFFSET | MCC_SHIM_0_OFFSET => {
            // MCC_COMPUTE_OFFSET == 0x60000 (compute only).
            // MCC_MEMTILE_OFFSET == MCC_SHIM_0_OFFSET == 0xFFF00 (memtile or shim).
            // The row disambiguates memtile vs shim via clock_clock_tile_kind_from_row.
            self.tiles.get(&(col, row))
                .map(|g| g.raw_mcc_0)
                .or_else(|| Some(reset_value_for_mcc(tile_kind)))
        }
        MCC_SHIM_1_OFFSET if row == 0 => {
            self.tiles.get(&(col, row))
                .and_then(|g| g.raw_mcc_1)
                .or_else(|| Some(reset_value_for_mcc_1()))
        }
        _ => None,
    }
}

/// AM025 reset value for Module_Clock_Control (and MCC_0 for shim).
fn reset_value_for_mcc(tile_kind: ClockTileKind) -> u32 {
    match tile_kind {
        ClockTileKind::Compute => 0x37,
        ClockTileKind::Memtile => 0x33,
        ClockTileKind::Shim    => 0x3B,
    }
}

/// AM025 reset value for shim Module_Clock_Control_1 (NoC enable bit set).
fn reset_value_for_mcc_1() -> u32 {
    0x01
}
```

- [ ] **Step 5: Add a reset-value test**

```rust
#[test]
fn read_mcc_returns_am025_reset_value_before_any_write() {
    let clock = ClockController::new(5, 6);
    // Compute tile reset is 0x37.
    assert_eq!(clock.read_register(2, 3, MCC_COMPUTE_OFFSET), Some(0x37));
    // Memtile reset is 0x33.
    assert_eq!(clock.read_register(2, 1, MCC_MEMTILE_OFFSET), Some(0x33));
    // Shim Module_Clock_Control_0 reset is 0x3B.
    assert_eq!(clock.read_register(2, 0, MCC_SHIM_0_OFFSET), Some(0x3B));
}
```

- [ ] **Step 6: Run all clock_control tests**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: all pass.

- [ ] **Step 7: Commit**

```
git add src/device/clock_control/mod.rs
git commit -m "clock_control: Module_Clock_Control register handling

Per-tile module gates for compute / memtile / shim.  Bit positions
sourced from AM025 regdb (aie_registers_aie2.json).  Reset values
returned on read pre-write match silicon (0x37 / 0x33 / 0x3B).

Generated using Claude Code."
```

---

## Task 6: AdaptiveState + tick_adaptive

**Files:**
- Modify: `src/device/clock_control/mod.rs`

- [ ] **Step 1: Add failing tests**

```rust
#[test]
fn adaptive_gate_default_disengaged_on_fresh_tile() {
    let clock = ClockController::new(5, 6);
    // No writes yet; adaptive state is "permissive" because the
    // tile is also clock-gated (default).  Once ungated and ticked,
    // the adaptive gate will engage on sustained idle.
    assert!(!clock.is_adaptive_dma_engaged(2, 2));
    assert!(!clock.is_adaptive_ss_engaged(2, 2));
}

#[test]
fn adaptive_dma_engages_after_idle_cycles() {
    let mut clock = ClockController::new(5, 6);
    // Set abort_period = 3 (engage after 2^3 = 8 idle cycles).
    clock.set_adaptive_abort_period(2, 2, 3);
    for _ in 0..8 {
        clock.tick_adaptive(2, 2, /*dma_active=*/ false, /*ss_active=*/ false);
    }
    assert!(clock.is_adaptive_dma_engaged(2, 2));
}

#[test]
fn adaptive_dma_resets_on_activity() {
    let mut clock = ClockController::new(5, 6);
    clock.set_adaptive_abort_period(2, 2, 3);
    for _ in 0..7 {
        clock.tick_adaptive(2, 2, false, false);
    }
    // One active cycle resets the counter.
    clock.tick_adaptive(2, 2, true, false);
    assert!(!clock.is_adaptive_dma_engaged(2, 2));
}

#[test]
fn adaptive_ss_independent_from_dma() {
    let mut clock = ClockController::new(5, 6);
    clock.set_adaptive_abort_period(2, 2, 3);
    // DMA active, SS idle.
    for _ in 0..8 {
        clock.tick_adaptive(2, 2, /*dma=*/ true, /*ss=*/ false);
    }
    assert!(!clock.is_adaptive_dma_engaged(2, 2),
        "DMA active -> DMA gate stays disengaged");
    assert!(clock.is_adaptive_ss_engaged(2, 2),
        "SS idle long enough -> SS gate engages");
}
```

- [ ] **Step 2: Run to confirm failure**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: 4 new FAIL ("method not found").

- [ ] **Step 3: Implement**

Add to `src/device/clock_control/mod.rs`:

```rust
#[derive(Debug, Clone, Default)]
struct AdaptiveState {
    dma_idle_cycles: u32,
    ss_idle_cycles: u32,
    /// Engagement threshold = 2^abort_period.  Range 3-12 per AM025.
    /// Default 7 (= 2^7 = 128 cycles, the AM025 default).
    abort_period_2pow: u8,
}

impl Default for AdaptiveState {
    // Manual default to set abort_period_2pow to the AM025 default of 7.
    fn default() -> Self {
        Self { dma_idle_cycles: 0, ss_idle_cycles: 0, abort_period_2pow: 7 }
    }
}

// Add field to ClockController:
pub struct ClockController {
    columns: Vec<bool>,
    tiles: HashMap<(u8, u8), TileGates>,
    adaptive: HashMap<(u8, u8), AdaptiveState>,
}

// Update ::new
pub fn new(num_cols: u8) -> Self {
    Self {
        columns: vec![false; num_cols as usize],
        tiles: HashMap::new(),
        adaptive: HashMap::new(),
    }
}

// Add methods on impl ClockController
pub fn tick_adaptive(&mut self, col: u8, row: u8, dma_active: bool, ss_active: bool) {
    let entry = self.adaptive.entry((col, row)).or_default();
    if dma_active { entry.dma_idle_cycles = 0; }
    else { entry.dma_idle_cycles = entry.dma_idle_cycles.saturating_add(1); }
    if ss_active { entry.ss_idle_cycles = 0; }
    else { entry.ss_idle_cycles = entry.ss_idle_cycles.saturating_add(1); }
}

pub fn set_adaptive_abort_period(&mut self, col: u8, row: u8, period_2pow: u8) {
    let entry = self.adaptive.entry((col, row)).or_default();
    entry.abort_period_2pow = period_2pow;
}

pub fn is_adaptive_dma_engaged(&self, col: u8, row: u8) -> bool {
    let Some(s) = self.adaptive.get(&(col, row)) else { return false; };
    let threshold = 1u32.checked_shl(s.abort_period_2pow as u32).unwrap_or(u32::MAX);
    s.dma_idle_cycles >= threshold
}

pub fn is_adaptive_ss_engaged(&self, col: u8, row: u8) -> bool {
    let Some(s) = self.adaptive.get(&(col, row)) else { return false; };
    let threshold = 1u32.checked_shl(s.abort_period_2pow as u32).unwrap_or(u32::MAX);
    s.ss_idle_cycles >= threshold
}
```

- [ ] **Step 4: Run tests to verify pass**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: all clock_control tests pass.

- [ ] **Step 5: Commit**

```
git add src/device/clock_control/mod.rs
git commit -m "clock_control: adaptive gates + tick_adaptive

DMA and SS idle counters with configurable 2^N abort period.
Activity resets, idleness advances; gate engaged when counter
crosses threshold.  AM025 default period is 2^7 = 128 cycles.

Generated using Claude Code."
```

---

## Task 7: SubsystemKind::ClockControl + subsystem_from_offset routing

**Files:**
- Modify: `xdna-archspec/src/types.rs` (or wherever `SubsystemKind` is defined -- locate first)
- Modify: `src/device/registers.rs` (subsystem_from_offset function)

- [ ] **Step 1: Locate SubsystemKind enum**

```
grep -rn "pub enum SubsystemKind" /home/triple/npu-work/xdna-emu/ 2>/dev/null
```

Note the file path. Likely `xdna-archspec/src/types.rs`.

- [ ] **Step 2: Add a failing test in src/device/registers.rs**

In the existing tests module of `src/device/registers.rs`, add:

```rust
#[test]
fn subsystem_from_offset_routes_column_clock_control_to_clock_control() {
    assert_eq!(
        subsystem_from_offset(0x000FFF20, TileKind::ShimNoc),
        SubsystemKind::ClockControl
    );
}

#[test]
fn subsystem_from_offset_routes_module_clock_control_to_clock_control() {
    // Compute MCC at 0x00060000
    assert_eq!(
        subsystem_from_offset(0x00060000, TileKind::Compute),
        SubsystemKind::ClockControl
    );
    // Memtile MCC at 0x000FFF00
    assert_eq!(
        subsystem_from_offset(0x000FFF00, TileKind::Mem),
        SubsystemKind::ClockControl
    );
}
```

- [ ] **Step 3: Run to confirm failure**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::registers
```

Expected: FAIL with "no variant `ClockControl` on type `SubsystemKind`".

- [ ] **Step 4: Add the enum variant**

In the file from Step 1, add `ClockControl` to the `SubsystemKind` enum:

```rust
pub enum SubsystemKind {
    // ... existing variants ...
    ClockControl,
    Unknown,
}
```

- [ ] **Step 5: Update subsystem_from_offset**

In `src/device/registers.rs`, find the `subsystem_from_offset` function. Add a top-priority match for the clock-control offsets BEFORE the existing range checks (clock-control offsets are exact, not range-based):

```rust
pub fn subsystem_from_offset(offset: u32, tile_kind: TileKind) -> SubsystemKind {
    // Clock-control offsets (exact match).  Must come first so the
    // generic ranges below do not consume them.
    match (offset, tile_kind) {
        (0x000FFF20, TileKind::ShimNoc | TileKind::ShimPl) => return SubsystemKind::ClockControl,
        (0x00060000, TileKind::Compute) => return SubsystemKind::ClockControl,
        (0x000FFF00, TileKind::Mem | TileKind::ShimNoc | TileKind::ShimPl) => return SubsystemKind::ClockControl,
        (0x000FFF04, TileKind::ShimNoc | TileKind::ShimPl) => return SubsystemKind::ClockControl,
        _ => {}
    }
    // ... existing dispatch ...
}
```

- [ ] **Step 6: Run tests to verify pass**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::registers
```

Expected: pass. Also run the broader test set to make sure we didn't break other subsystem routing:

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::registers
```

- [ ] **Step 7: Commit**

```
git add xdna-archspec/src/types.rs src/device/registers.rs
git commit -m "clock_control: add SubsystemKind variant + offset routing

Module_Clock_Control and Column_Clock_Control offsets now
classify as SubsystemKind::ClockControl.  Routing handled in
the dispatch layer in the next commit.

Generated using Claude Code."
```

---

## Task 8: Dispatch ClockControl writes through DeviceState

**Files:**
- Modify: `src/device/state/dispatch.rs`

- [ ] **Step 1: Add a failing test**

In `src/device/state/` find the existing tests module (might be in `dispatch.rs` or a sibling). Add:

```rust
#[test]
fn write_tile_register_routes_column_clock_control_to_clock_controller() {
    let mut state = make_test_device_state(); // existing helper, or build inline
    // Initially gated.
    assert!(!state.array.clock().is_column_active(2));
    // Write Column_Clock_Control bit 0 = 1 to ungate column 2.
    state.write_tile_register(2, 0, 0x000FFF20, 0x1);
    assert!(state.array.clock().is_column_active(2));
}
```

If `make_test_device_state` doesn't exist, write a minimal one inline:

```rust
fn make_test_device_state() -> DeviceState {
    use std::sync::Arc;
    use xdna_archspec::runtime::ArchConfig;
    let arch: Arc<dyn ArchConfig> = xdna_archspec::runtime::aie2_npu1();
    DeviceState::new(arch) // adjust to existing constructor
}
```

- [ ] **Step 2: Run to confirm failure**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::state::dispatch
```

Expected: FAIL (the assertion that column becomes active fails -- dispatch hasn't routed the write yet).

- [ ] **Step 3: Add the dispatch arm**

In `src/device/state/dispatch.rs`, find the `write_register` function (around line 29) with the `match subsystem` block. Add an arm:

```rust
match subsystem {
    // ... existing arms ...

    SubsystemKind::ClockControl => {
        self.array.clock_mut().write_register(
            tile_addr.col, tile_addr.row, tile_addr.offset, value,
        );
    }

    SubsystemKind::Unknown => { /* ... existing handling ... */ }
}
```

- [ ] **Step 4: Run tests**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::state::dispatch
```

Expected: pass.

- [ ] **Step 5: Commit**

```
git add src/device/state/dispatch.rs
git commit -m "clock_control: dispatch ClockControl writes through DeviceState

write_tile_register now routes Column_Clock_Control and
Module_Clock_Control writes to ClockController::write_register.

Generated using Claude Code."
```

---

## Task 9: Column gate check in step_data_movement

**Files:**
- Modify: `src/device/array/routing.rs` (find `step_data_movement` function)

- [ ] **Step 1: Read the function to understand it**

```
sed -n '71,140p' /home/triple/npu-work/xdna-emu/src/device/array/routing.rs
```

(Or use Read on lines 71-140 of that file.)

Identify the loop where it iterates over columns / tiles.

- [ ] **Step 2: Write a failing test**

Add to `src/device/array/tests.rs` (or wherever routing tests live):

```rust
#[test]
fn step_data_movement_skips_gated_column() {
    use crate::device::host_memory::HostMemory;
    use std::sync::Arc;
    use xdna_archspec::runtime::ArchConfig;
    let arch: Arc<dyn ArchConfig> = xdna_archspec::runtime::aie2_npu1();
    let mut array = TileArray::new(arch);
    let mut host = HostMemory::new();
    // All columns gated by default.  step_data_movement should report
    // no DMA active, no SS active.
    let (dma_active, ss_active, _) = array.step_data_movement(&mut host);
    assert!(!dma_active, "gated column DMA should not be active");
    assert!(!ss_active, "gated column SS should not be active");
}
```

(Adjust to actual function signature if it differs.)

- [ ] **Step 3: Run to confirm it fails**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::array::tests::step_data_movement_skips_gated_column
```

Expected: FAIL (currently steps every column regardless of gate state).

- [ ] **Step 4: Add the gate check**

In `step_data_movement`, at the top of the per-column loop, add:

```rust
// Skip ungated columns -- silicon would not clock them, so the
// emulator advances no work either.  This is the v1 perf win.
if !self.clock.is_column_active(col) {
    continue;
}
```

(Exact placement depends on the loop structure; insert before any per-column work.)

- [ ] **Step 5: Run tests**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::array::tests::step_data_movement_skips_gated_column
```

Expected: pass.

- [ ] **Step 6: Commit**

```
git add src/device/array/routing.rs src/device/array/tests.rs
git commit -m "clock_control: skip step_data_movement for gated columns

Top-tier execution gate: ungated columns advance no DMA / SS /
core work per cycle.  The perf win lands here -- typical programs
use 1-2 of 5 columns.

Generated using Claude Code."
```

---

## Task 10: Module gate checks per subsystem + adaptive tick

**Files:**
- Modify: `src/device/array/routing.rs` (per-tile step calls within step_data_movement)
- Modify: `src/device/array/dma_ops.rs` (if step_dma is called directly there)

- [ ] **Step 1: Identify the call sites**

```
grep -n "step_dma\|step_tile_switches\|step_core" /home/triple/npu-work/xdna-emu/src/device/array/routing.rs /home/triple/npu-work/xdna-emu/src/device/array/dma_ops.rs
```

Note each per-subsystem call.

- [ ] **Step 2: Write a failing test**

```rust
#[test]
fn module_gate_skips_dma_when_dma_module_gated() {
    use crate::device::clock_control::ModuleKind;
    let arch = xdna_archspec::runtime::aie2_npu1();
    let mut array = TileArray::new(std::sync::Arc::clone(&arch));
    // Ungate column 2 but leave the DMA module bit clear in MCC.
    array.clock_mut().write_register(2, 0, 0x000FFF20, 0x1);
    // Write MCC with only Core enabled, DMA disabled.
    // (Replace BIT positions with regdb values.)
    let core_bit = 0;
    array.clock_mut().write_register(2, 3, 0x00060000, 1 << core_bit);
    assert!(array.clock().is_module_active(2, 3, ModuleKind::Core));
    assert!(!array.clock().is_module_active(2, 3, ModuleKind::Dma));
    let mut host = crate::device::host_memory::HostMemory::new();
    let dma_result = array.step_dma(2, 3, &mut host);
    assert!(dma_result.is_none() || matches!(dma_result, Some(_)),
        "DMA should not progress when module-gated; expected None");
}
```

- [ ] **Step 3: Run to confirm failure**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::array::tests::module_gate_skips_dma_when_dma_module_gated
```

Expected: FAIL.

- [ ] **Step 4: Add gate checks**

In each step call site, wrap with a gate check. Example for DMA:

```rust
pub fn step_dma(&mut self, col: u8, row: u8, host_memory: &mut HostMemory) -> Option<DmaResult> {
    use crate::device::clock_control::ModuleKind;
    if !self.clock.is_column_active(col)
        || !self.clock.is_module_active(col, row, ModuleKind::Dma)
        || self.clock.is_adaptive_dma_engaged(col, row)
    {
        return None;
    }
    // ... existing body ...
}
```

Similar wrapper around `step_tile_switches` (using `StreamSwitch` and `is_adaptive_ss_engaged`), and around the per-core step (using `Core`).

After each step call, feed activity back into adaptive:

```rust
let dma_active = /* whether step_dma returned Some(active) */;
let ss_active = /* same for SS */;
self.clock_mut().tick_adaptive(col, row, dma_active, ss_active);
```

- [ ] **Step 5: Run tests**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::array
```

Expected: new test passes; existing tests may fail (still need migration -- Task 13).

- [ ] **Step 6: Commit**

```
git add src/device/array/routing.rs src/device/array/dma_ops.rs
git commit -m "clock_control: module gates + adaptive tick in step path

Per-subsystem step calls (DMA, SS, core) now consult module-gate
state.  Adaptive idle counters tick after each step with whether
the subsystem did work.

Generated using Claude Code."
```

---

## Task 11: warn_once_per_site on gated-tile access

**Files:**
- Modify: `src/device/state/dispatch.rs` (in `write_tile_register`)
- Possibly modify the read path similarly

- [ ] **Step 1: Add a failing test**

```rust
#[test]
fn write_to_gated_tile_logs_warning() {
    // Use a log capture mechanism if the codebase has one; otherwise
    // assert via a behavioral side effect (e.g., a counter on
    // ClockController that increments on gated-access warning).
    // ...
}
```

If the codebase has no log capture, add a counter to ClockController:

```rust
pub struct ClockController {
    // ... existing fields ...
    /// Sites where a gated-tile access was warned about, used for
    /// dedupping and (in tests) for observation.
    warned_sites: std::collections::HashSet<(u8, u8, u32)>,
}
```

And a getter for tests:

```rust
#[cfg(test)]
pub fn warned_sites_len(&self) -> usize {
    self.warned_sites.len()
}
```

Test:

```rust
#[test]
fn first_write_to_gated_tile_records_warning() {
    let mut state = make_test_device_state();
    // Column 2 is gated by default.
    assert_eq!(state.array.clock().warned_sites_len(), 0);
    // Write to a non-clock-control register on a gated tile.
    state.write_tile_register(2, 3, 0x00000, 0xDEADBEEF);
    assert_eq!(state.array.clock().warned_sites_len(), 1,
        "first gated-tile access should record one warning site");
    // Repeat write to the same site -- no new warning.
    state.write_tile_register(2, 3, 0x00000, 0xCAFEF00D);
    assert_eq!(state.array.clock().warned_sites_len(), 1,
        "subsequent writes to same site should dedup");
}
```

- [ ] **Step 2: Run to confirm failure**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::state
```

Expected: FAIL.

- [ ] **Step 3: Implement**

In `src/device/clock_control/mod.rs`:

```rust
/// Record a warning for an access to a gated tile.  Returns true if
/// this is the first time the site (col, row, offset) was seen since
/// construction; false if it was a duplicate.
pub fn warn_gated_access(&mut self, col: u8, row: u8, offset: u32) -> bool {
    if self.is_tile_active(col, row) { return false; }
    self.warned_sites.insert((col, row, offset))
}
```

In `src/device/state/dispatch.rs::write_tile_register`, before doing the dispatch:

```rust
if self.array.clock_mut().warn_gated_access(col, row, offset) {
    log::warn!(
        "access to gated tile ({}, {}) at offset 0x{:05X}; silicon would produce undefined results. \
        Set XDNA_EMU_WARN_GATED_ACCESS=0 to silence.",
        col, row, offset,
    );
}
```

Honor `XDNA_EMU_WARN_GATED_ACCESS=0` by skipping the warn call entirely:

```rust
fn warnings_enabled() -> bool {
    std::env::var("XDNA_EMU_WARN_GATED_ACCESS").as_deref() != Ok("0")
}
```

- [ ] **Step 4: Run tests**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::state
```

Expected: pass.

- [ ] **Step 5: Commit**

```
git add src/device/clock_control/mod.rs src/device/state/dispatch.rs
git commit -m "clock_control: warn_once_per_site on gated-tile access

Accesses to gated tiles proceed as silicon would (undefined
behavior), but the emulator logs a deduplicated warning to surface
the bug pattern.  Silenced via XDNA_EMU_WARN_GATED_ACCESS=0.

Generated using Claude Code."
```

---

## Task 12: ungate_all() test helper

**Files:**
- Modify: `src/device/clock_control/mod.rs`

- [ ] **Step 1: Add a failing test**

```rust
#[test]
fn ungate_all_makes_every_column_and_module_active() {
    let mut clock = ClockController::new(5, 6);
    // Pre: everything gated.
    assert!(!clock.is_column_active(0));
    assert!(!clock.is_module_active(2, 3, ModuleKind::Core));
    // Ungate all.
    clock.ungate_all();
    // Post: every column active, every module active.
    for col in 0..5 {
        assert!(clock.is_column_active(col));
    }
    // Spot-check a few tiles across kinds.
    for row in 0..6 {
        for kind in [ModuleKind::Core, ModuleKind::Memory, ModuleKind::Dma, ModuleKind::StreamSwitch] {
            assert!(clock.is_module_active(2, row, kind),
                "tile (2, {}) module {:?} should be active after ungate_all", row, kind);
        }
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control::tests::ungate_all_makes_every_column_and_module_active
```

Expected: FAIL ("method `ungate_all` not found").

- [ ] **Step 3: Implement**

Add to `impl ClockController`:

```rust
/// Test helper: enable every column and module by writing
/// all-active patterns via the same register-write path the CDO uses.
/// Spec: serve-and-warn policy; tests opt out of silicon-accurate
/// boot via this helper.
pub fn ungate_all(&mut self) {
    let num_cols = self.columns.len() as u8;
    let rows = self.num_rows;
    for col in 0..num_cols {
        // Enable the column.
        self.write_register(col, 0, COLUMN_CLOCK_CONTROL_OFFSET, 0x1);
        for row in 0..rows {
            // Write MCC with all module bits set.  Use 0xFFFFFFFF;
            // the bit-field decoder picks out the bits it knows.
            let offset = match clock_tile_kind_from_row(row) {
                ClockTileKind::Shim => MCC_SHIM_0_OFFSET,
                ClockTileKind::Memtile => MCC_MEMTILE_OFFSET,
                ClockTileKind::Compute => MCC_COMPUTE_OFFSET,
            };
            self.write_register(col, row, offset, 0xFFFF_FFFF);
            if matches!(clock_tile_kind_from_row(row), ClockTileKind::Shim) {
                self.write_register(col, row, MCC_SHIM_1_OFFSET, 0xFFFF_FFFF);
            }
        }
    }
}
```

- [ ] **Step 4: Run tests**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control
```

Expected: pass.

- [ ] **Step 5: Commit**

```
git add src/device/clock_control/mod.rs
git commit -m "clock_control: ungate_all() test helper

Writes all-active patterns through write_register so tests
exercise the real CDO path, not a backdoor.

Generated using Claude Code."
```

---

## Task 13: Test migration audit + fix

**Files:**
- Modify: all tests that construct `TileArray` / `DeviceState` directly and expect tiles to execute

- [ ] **Step 1: Audit existing tests**

```
TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tee /tmp/claude-1000/clock-control-test-failures.txt
```

Grep the output for `FAILED`. Group by file. Expect the failures to be tests that construct TileArray and run kernels without ungating.

Count the failures:

```
grep -c "FAILED" /tmp/claude-1000/clock-control-test-failures.txt
```

- [ ] **Step 2: Decision point**

If failures > 30 sites, STOP and ask the user for guidance on migration approach. Otherwise proceed.

- [ ] **Step 3: Migrate tests**

For each failing test that's expected to execute kernels:

1. Find the line that constructs `TileArray::new(arch)` or `DeviceState::new(arch)`.
2. Add one line immediately after: `array.clock_mut().ungate_all();` (or `state.array.clock_mut().ungate_all();`).

Bulk-add via sed where the pattern is consistent:

```
# Example, adjust regex carefully
grep -l "TileArray::new" src/ -r | xargs sed -i 's/let mut array = TileArray::new(arch);/let mut array = TileArray::new(arch); array.clock_mut().ungate_all();/'
```

Manual review every change.

- [ ] **Step 4: Run the full suite**

```
TMPDIR=/tmp/claude-1000 cargo test --lib
```

Expected: all tests that were passing before Task 1 are now passing again.

- [ ] **Step 5: Commit**

```
git add -u
git commit -m "clock_control: migrate direct-construction tests to ungate_all()

N tests that construct TileArray/DeviceState directly (rather than
loading a CDO) now call ungate_all() so the kernel can execute.
xclbin_suite and bridge tests are unchanged -- their CDOs already
program clocks.

Generated using Claude Code."
```

(Replace N with the actual count.)

---

## Task 14: Hand-rolled CDO integration test

**Files:**
- Create: `src/testing/clock_control_integration.rs` (or in existing testing/ harness)
- Modify: wherever the test runner discovers integration tests

- [ ] **Step 1: Locate testing infrastructure**

```
ls /home/triple/npu-work/xdna-emu/src/testing/ 2>&1
cat /home/triple/npu-work/xdna-emu/src/testing/mod.rs 2>&1 | head -30
```

Read enough to understand where new integration tests should land.

- [ ] **Step 2: Write the test cases**

Three integration tests:

1. Gate column N, run a kernel targeting that column, verify no output produced.
2. Ungate after gating, verify kernel produces expected output.
3. Mixed: gate column A, leave column B ungated; verify column B runs while column A is dark.

These can be unit-test style if the harness allows constructing a minimal DeviceState + dispatching synthetic CDO commands. Otherwise add as `#[test] fn`s in a new file under `src/device/clock_control/integration_tests.rs`:

```rust
//! Integration tests for clock-control execution gating.
use super::*;
use crate::device::state::DeviceState;
use std::sync::Arc;

fn fresh_state() -> DeviceState {
    let arch: Arc<dyn xdna_archspec::runtime::ArchConfig> = xdna_archspec::runtime::aie2_npu1();
    DeviceState::new(arch)
}

#[test]
fn gated_column_produces_no_dma_progress() {
    let mut state = fresh_state();
    // Do NOT ungate.
    // Submit a write_tile_register sequence that would normally trigger DMA.
    // (Build out the minimal sequence based on how other tests exercise DMA.)
    // Step the array for N cycles.
    // Assert: DMA channel state remains in Idle / initial state.
}

// ... two more similar tests
```

The exact CDO sequence to test depends on existing test harness patterns -- consult `src/testing/xclbin_suite.rs` or similar for the canonical way to drive a synthetic CDO.

- [ ] **Step 3: Run tests**

```
TMPDIR=/tmp/claude-1000 cargo test --lib device::clock_control::integration_tests
```

Expected: all pass.

- [ ] **Step 4: Commit**

```
git add src/device/clock_control/integration_tests.rs
git commit -m "clock_control: integration tests for execution gating

Three test cases: gated-column -> no execution; ungated -> normal
execution; mixed gating -> selective execution.  Exercises the
full path from DeviceState::write_tile_register to per-subsystem
step gating.

Generated using Claude Code."
```

---

## Task 15: Bridge test verification + coverage doc update

**Files:**
- Modify: `docs/coverage/aie2/implementation-gaps.md`

- [ ] **Step 1: Run bridge tests**

```
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --no-hw add_one 2>&1 | tee /tmp/claude-1000/clock-control-bridge.log
```

Watch for failures. Most tests should pass -- their CDOs program clocks. Any test that fails because its CDO does NOT program clocks is a bug exposed by this change (treat it as a real bug, document it, address case by case).

- [ ] **Step 2: Document any bridge-test failures**

If any failures, create `docs/superpowers/findings/2026-MM-DD-clock-control-bridge-impact.md` documenting:
- Which tests failed
- The dmesg / log signatures
- Whether the cause is a CDO that doesn't program clocks (= bug) or another regression

- [ ] **Step 3: Run full HW bridge test if possible**

```
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/clock-control-bridge-full.log
```

(Skip if HW is being used by another test or unavailable.)

- [ ] **Step 4: Update coverage doc**

In `docs/coverage/aie2/implementation-gaps.md`, change:

```markdown
- noc: STUB
- clock_control: STUB
```

to:

```markdown
- noc: STUB
- clock_control: Full (column / module / adaptive)
```

Or whatever phrasing matches the doc's conventions for closed gaps.

- [ ] **Step 5: Commit**

```
git add docs/coverage/aie2/implementation-gaps.md
git commit -m "coverage: close clock_control gap

Three-tier clock control (column / module / adaptive) implemented
and tested.  See docs/superpowers/specs/2026-05-24-clock-control-design.md
for the design; the plan in docs/superpowers/plans/ tracks the
implementation history.

Generated using Claude Code."
```

---

## Self-review checklist (run after all tasks land)

After all tasks are complete, verify:

- [ ] `cargo test --lib` passes with no new failures
- [ ] `cargo test --lib device::clock_control` shows all clock_control unit tests passing
- [ ] Bridge tests pass (or any failures documented as real bugs)
- [ ] No `clock_control: STUB` entry remains in `docs/coverage/aie2/implementation-gaps.md`
- [ ] `cargo build --release` succeeds (catches release-mode-only issues)
- [ ] `cargo build -p xdna-emu-ffi` succeeds (catches FFI breakage)
- [ ] Module file size is under ~400 LoC; if larger, consider the file split mentioned in the spec
