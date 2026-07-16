# Visual Debugger v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live, steppable, multi-panel egui debugger that loads a real `.xclbin`, drives the emulator one cycle at a time, and shows the array plus one selected tile's live internals.

**Architecture:** Split logic from rendering. A **non-gated** `src/loading/` (shared xclbin load path) and `src/debugger/` (egui-free view-model + a frame-bounded stepper) hold all the testable logic and compile under the default `cargo test --lib`. A **`gui`-feature-gated** `src/visual/` holds only egui rendering that consumes those types. The app owns an `InterpreterEngine` (already a single-cycle stepper) and is a pure read-and-render viewer over existing state — no new emulator execution logic.

**Tech Stack:** Rust, egui/eframe (already a dep behind the optional `gui` feature), the existing `InterpreterEngine` / `TileArray` / `DmaEngine` / `NpuExecutor` APIs.

## Global Constraints

- **egui/eframe stays behind `feature = "gui"`.** Only `src/visual/` is gated. `src/loading/` and `src/debugger/` must be egui-free and compile without the feature, so their tests run under bare `cargo test --lib`.
- **No new emulator execution logic.** The GUI reads existing state and calls existing step/run APIs. The only in-crate additions are: the shared load helper, three thin DMA accessors, and the egui-free view-model.
- **Topology comes from the LIVE array** (`engine.device().array`, i.e. `TileArray::iter()` / `cols()` / `rows()`), NOT from `aie-device-models.json`. Each `Tile` carries its own `col`/`row` and `is_shim()`/`is_mem()`/`is_compute()`.
- **`cargo test --lib` stays green** after every task. It is the mandated regression gate (see CLAUDE.md).
- **No hardcoded topology, no hardcoded bit positions.** Derive from the toolchain/live state (project rule).
- **No emoji anywhere.** Commit messages end with `Generated using Claude Code.`
- Fixture tests load `MANIFEST/../mlir-aie/build/test/npu-xrt/<name>/chess/aie.xclbin` and **skip (return early with an eprintln) if the file is absent** — the established idiom (`xclbin_suite.rs:1764`). Never fail on a missing built fixture.

---

## File Structure

**New, non-gated (testable under default `cargo test --lib`):**
- `src/loading/mod.rs` — `load_engine(&Path) -> Result<InterpreterEngine, String>` (shared xclbin -> ready-to-step engine) + `default_host_buffers() -> Vec<HostBuffer>`.
- `src/debugger/mod.rs` — module root (`pub mod engine_host; pub mod model;`).
- `src/debugger/engine_host.rs` — `EngineHost` (owns engine + optional executor; load, step_one, step_bounded, reset, status, cycles, run-state).
- `src/debugger/model.rs` — egui-free view-model: `tile_grid()`, `tile_snapshot()`, and the plain structs they return.

**Modified (core, non-gated):**
- `src/device/dma/engine/mod.rs` — add `current_bd`, `queued_bd`, `channel_count` accessors over the `pub(super) channels`.
- `src/testing/xclbin_suite.rs` — `run_single_inner` refactored to call `load_engine` (removes the inline parse->cdo->elf->sync block).
- `src/lib.rs` — declare `pub mod loading;` and `pub mod debugger;` (ungated); keep `#[cfg(feature = "gui")] pub mod visual;`.

**Rewritten, `gui`-gated (smoke-tested by running):**
- `src/visual/mod.rs` — declares the new submodules; re-exports `DebuggerApp`.
- `src/visual/theme.rs` — debugger palette (replaces the trace-viewer palette).
- `src/visual/app.rs` — `DebuggerApp: eframe::App`; owns `EngineHost`, selected tile, run state; top-level layout.
- `src/visual/overview.rs` — array grid render + click-to-select.
- `src/visual/detail.rs` — selected-tile detail rows.
- `src/visual/controls.rs` — control bar (load-status, step/run/pause/reset, cycle+status readout).

**Deleted (old trace-comparison viewer):**
- `src/visual/{alignment,timeline,viewport,data,event_detail,tile_selector,app}.rs` (old `app.rs` replaced).

**Modified:**
- `src/main.rs` — `run_gui` repointed to construct `DebuggerApp`; drop the `--trace-view-hw/--trace-view-emu` wiring.

---

## Task 1: Shared load helper (`src/loading/`)

**Files:**
- Create: `src/loading/mod.rs`
- Modify: `src/lib.rs` (add `pub mod loading;`)
- Modify: `src/testing/xclbin_suite.rs:659-912` (`run_single_inner` — replace inline load block with a `load_engine` call; replace the inline default-buffer literal with `default_host_buffers()`)

**Interfaces:**
- Produces:
  - `pub fn load_engine(xclbin_path: &std::path::Path) -> Result<crate::interpreter::InterpreterEngine, String>` — parses the xclbin, builds an NPU1 engine, assigns partition columns, applies the CDO, loads every core ELF, and syncs cores. Returns a ready-to-step engine. Does NOT set a stall threshold, set up host input, or build an `NpuExecutor` (caller's concern).
  - `pub fn default_host_buffers() -> Vec<crate::npu::HostBuffer>` — the default DDR layout: `{0x0000, 4096}`, `{0x1000, 256}`, `{0x2000, 4096}`.

**Reference — the exact sequence to encapsulate** (from `run_single_inner`, verified): `Xclbin::from_file` -> `find_section(SectionKind::AiePartition)` -> `AiePartition::parse(section.data())` -> `partition.primary_pdi()` -> `find_cdo_offset(pdi.pdi_image)` -> `Cdo::parse(&pdi.pdi_image[off..])` -> `InterpreterEngine::new_npu1()` -> `engine.device_mut().assign_partition_columns(0, partition.column_width() as u8)` -> `engine.device_mut().apply_cdo(&cdo)?` -> for each ELF in the partition: `engine.load_elf_bytes(col, row, &data)?` -> `engine.sync_cores_from_device()`. Confirm the ELF-iteration shape by reading `run_single_inner` around lines 800-893 before writing (the ELF list comes from the partition; mirror it exactly).

- [ ] **Step 1: Declare the module.** In `src/lib.rs`, add `pub mod loading;` next to the other top-level `pub mod` declarations (ungated).

- [ ] **Step 2: Write the failing test.** Create `src/loading/mod.rs` with a test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin")
    }

    #[test]
    fn load_engine_produces_a_ready_engine() {
        let path = fixture();
        if !path.exists() {
            eprintln!("SKIP load_engine_produces_a_ready_engine: fixture not built at {}", path.display());
            return;
        }
        let engine = load_engine(&path).expect("load should succeed");
        // NPU1 array is instantiated and cores were enabled by ELF load.
        assert!(engine.device().array.cols() >= 1);
        assert!(engine.enabled_cores() >= 1, "at least one core should be enabled after ELF load");
    }

    #[test]
    fn default_host_buffers_match_the_default_ddr_layout() {
        let b = default_host_buffers();
        assert_eq!(b.len(), 3);
        assert_eq!((b[0].address, b[0].size), (0x0000, 4096));
        assert_eq!((b[1].address, b[1].size), (0x1000, 256));
        assert_eq!((b[2].address, b[2].size), (0x2000, 4096));
    }
}
```

- [ ] **Step 3: Run it, verify it fails to compile** (`load_engine`/`default_host_buffers` undefined).

Run: `cargo test --lib loading:: 2>&1 | tail -20`
Expected: compile error, `cannot find function load_engine`.

- [ ] **Step 4: Implement `load_engine` and `default_host_buffers`.** Write the two functions above the test module. Copy the exact call sequence out of `run_single_inner`, mapping each `TestOutcome::LoadError { message }` early-return to `return Err(message)`. Use the verified signatures:
  - `Xclbin::from_file`, `find_section`, `SectionKind::AiePartition` (`crate::parser::xclbin`)
  - `AiePartition::parse`, `primary_pdi`, `column_width` (`crate::parser::aie_partition`)
  - `find_cdo_offset`, `Cdo::parse` (`crate::parser::cdo`) — locate `find_cdo_offset` by reading its use at `run_single_inner:720`
  - `InterpreterEngine::new_npu1`, `device_mut`, `load_elf_bytes`, `sync_cores_from_device`, `enabled_cores`, `device` (`crate::interpreter`)
  - `assign_partition_columns`, `apply_cdo` (on `DeviceState` via `device_mut()`)
  - `HostBuffer { address, size }` (`crate::npu`)

`default_host_buffers`:

```rust
pub fn default_host_buffers() -> Vec<crate::npu::HostBuffer> {
    use crate::npu::HostBuffer;
    vec![
        HostBuffer { address: 0x0000, size: 4096 },
        HostBuffer { address: 0x1000, size: 256 },
        HostBuffer { address: 0x2000, size: 4096 },
    ]
}
```

Add a module doc comment: `//! Shared xclbin load path. Both the in-process test suite (xclbin_suite) and the visual debugger use load_engine so there is exactly one place that knows the parse -> CDO -> ELF -> sync incantation.`

- [ ] **Step 5: Run the tests, verify they pass** (or SKIP cleanly if the fixture is absent).

Run: `cargo test --lib loading:: 2>&1 | tail -20`
Expected: PASS (or the SKIP eprintln for the fixture test + PASS for the buffers test).

- [ ] **Step 6: Refactor `run_single_inner` to use `load_engine`.** Read `run_single_inner` (659 onward). Replace the inline block that goes from `Xclbin::from_file` (669) through `sync_cores_from_device` (893) with:

```rust
let mut engine = match crate::loading::load_engine(&test.xclbin_path) {
    Ok(e) => e,
    Err(message) => return (TestOutcome::LoadError { message }, None, None, warnings),
};
engine.set_stall_threshold(/* keep the existing threshold value/expression from line 746 */);
```

Keep everything that is genuinely test-specific and currently lives in that range: the stall-threshold set, the `input_values`/`host_buffers` setup (the `buffer_spec` vs default branch at 774-785), the `NpuExecutor` build (828-846), and the `run_engine` call (912). Replace the inline default-buffer `vec![HostBuffer {..}, ..]` in the `else` branch (774-785) with `crate::loading::default_host_buffers()`. **Preserve the exact return-tuple shape** of `run_single_inner` (verify the `warnings`/`Option<Vec<u8>>` fields it returns).

- [ ] **Step 7: Run the full library suite, verify no regression.**

Run: `cargo test --lib 2>&1 | tail -25`
Expected: same pass count as before this task, 0 failures. The three fixture-backed `xclbin_suite` tests (`add_one_using_dma`, etc.) still pass or skip exactly as before.

- [ ] **Step 8: Commit.**

```bash
git add src/loading/mod.rs src/lib.rs src/testing/xclbin_suite.rs
git commit -m "feat(loading): shared load_engine helper; refactor xclbin_suite onto it

Generated using Claude Code."
```

---

## Task 2: Live DMA accessors

**Files:**
- Modify: `src/device/dma/engine/mod.rs` (add three accessors on `impl DmaEngine`)

**Interfaces:**
- Consumes: `pub(super) channels: Vec<ChannelContext>` (mod.rs:80); `ChannelContext { pub current_bd: Option<u8>, pub queued_bd: Option<u8>, .. }` (channel.rs:303/309); `pub type ChannelId = u8`.
- Produces (on `DmaEngine`):
  - `pub fn current_bd(&self, channel: ChannelId) -> Option<u8>`
  - `pub fn queued_bd(&self, channel: ChannelId) -> Option<u8>`
  - `pub fn channel_count(&self) -> usize`

These are thin reads over `channels`, needed because `channels` is `pub(super)` so the fields are unreachable from `src/debugger/`.

- [ ] **Step 1: Write the failing test.** In the existing `#[cfg(test)] mod tests` at the bottom of `src/device/dma/engine/mod.rs` (or add one), add:

```rust
#[test]
fn bd_accessors_read_channel_state() {
    // Construct a DmaEngine the same way the other tests in this file do
    // (read an existing test above for the exact constructor/arch args).
    let engine = /* existing test constructor for a compute-tile DmaEngine */;
    // A freshly constructed engine has no active/queued BD on channel 0.
    assert_eq!(engine.current_bd(0), None);
    assert_eq!(engine.queued_bd(0), None);
    assert!(engine.channel_count() >= 1);
}
```

(Read the top of the existing test module in this file first and copy its `DmaEngine` construction verbatim — do not invent a constructor.)

- [ ] **Step 2: Run it, verify it fails** (methods undefined).

Run: `cargo test --lib dma::engine 2>&1 | tail -20`
Expected: compile error, `no method named current_bd`.

- [ ] **Step 3: Implement the accessors.** Add to `impl DmaEngine` (near `channel_state`, mod.rs:789):

```rust
/// The BD id currently active on `channel`, if any. Read-only live view.
pub fn current_bd(&self, channel: ChannelId) -> Option<u8> {
    self.channels.get(channel as usize).and_then(|c| c.current_bd)
}

/// The BD id queued to load next on `channel`, if any. Read-only live view.
pub fn queued_bd(&self, channel: ChannelId) -> Option<u8> {
    self.channels.get(channel as usize).and_then(|c| c.queued_bd)
}

/// Number of DMA channels on this engine.
pub fn channel_count(&self) -> usize {
    self.channels.len()
}
```

- [ ] **Step 4: Run it, verify it passes.**

Run: `cargo test --lib dma::engine 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/device/dma/engine/mod.rs
git commit -m "feat(dma): live current_bd/queued_bd/channel_count accessors

Generated using Claude Code."
```

---

## Task 3: `EngineHost` — load + frame-bounded stepper (`src/debugger/engine_host.rs`)

**Files:**
- Create: `src/debugger/mod.rs`
- Create: `src/debugger/engine_host.rs`
- Modify: `src/lib.rs` (add `pub mod debugger;`)

**Interfaces:**
- Consumes: `crate::loading::{load_engine, default_host_buffers}`; `InterpreterEngine::{step, total_cycles, status, reset, device, device_and_host_memory, enabled_cores}`; `EngineStatus` (Ready/Running/Paused/Halted/Stalled/Error); `NpuExecutor::{new, set_host_buffers, load, try_advance, is_done}`; `NpuInstructionStream::parse`; `AdvanceResult::{Idle, Done, Error}`.
- Produces:
  - `pub enum RunState { Paused, Running }`
  - `pub struct EngineHost { pub engine: InterpreterEngine, executor: Option<NpuExecutor>, pub run_state: RunState }`
  - `pub fn load(xclbin_path: &Path) -> Result<EngineHost, String>`
  - `pub fn step_one(&mut self)` — one executor-interleave + one `engine.step()`
  - `pub fn step_bounded(&mut self, budget: u32) -> EngineStatus` — up to `budget` steps, stopping early on a terminal status (Halted/Stalled/Error) or when the executor is done and the engine is idle; returns the status at stop
  - `pub fn reset(&mut self)`
  - `pub fn total_cycles(&self) -> u64`
  - `pub fn status(&self) -> EngineStatus`

**The interleave (mirror `run_engine`, verified):** each step does — if an executor exists, `let (device, host_mem) = self.engine.device_and_host_memory(); match executor.try_advance(device, host_mem) { AdvanceResult::Error(_) => stop, _ => {} }` — then `self.engine.step()`. The borrow of device+host_mem must be scoped so it ends before `engine.step()`.

- [ ] **Step 1: Declare the modules.** Create `src/debugger/mod.rs`:

```rust
//! Egui-free view-model and stepper for the visual debugger. Compiles without
//! the `gui` feature so its logic is covered by `cargo test --lib`. The egui
//! rendering layer (src/visual) consumes these types.
pub mod engine_host;
pub mod model;
```

(`model` is Task 4; create a placeholder `src/debugger/model.rs` with `//! view-model (Task 4)` so the module compiles now, or add `pub mod model;` in Task 4 — pick one and keep `mod.rs` compiling.) In `src/lib.rs`, add `pub mod debugger;` (ungated).

- [ ] **Step 2: Write the failing test.** In `src/debugger/engine_host.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin")
    }

    #[test]
    fn step_bounded_advances_cycles() {
        let path = fixture();
        if !path.exists() {
            eprintln!("SKIP step_bounded_advances_cycles: fixture not built at {}", path.display());
            return;
        }
        let mut host = load(&path).expect("load");
        let before = host.total_cycles();
        host.step_bounded(50);
        assert!(host.total_cycles() > before, "stepping must advance the cycle count");
    }

    #[test]
    fn reset_returns_to_zero_cycles() {
        let path = fixture();
        if !path.exists() {
            eprintln!("SKIP reset_returns_to_zero_cycles: fixture not built at {}", path.display());
            return;
        }
        let mut host = load(&path).expect("load");
        host.step_bounded(50);
        host.reset();
        assert_eq!(host.total_cycles(), 0);
    }
}
```

- [ ] **Step 3: Run it, verify it fails** (undefined).

Run: `cargo test --lib debugger::engine_host 2>&1 | tail -20`
Expected: compile error.

- [ ] **Step 4: Implement `EngineHost`.** Write:

```rust
use std::fs;
use std::path::Path;

use crate::interpreter::InterpreterEngine;
use crate::interpreter::engine::coordinator::EngineStatus; // confirm exact path when writing
use crate::loading::{default_host_buffers, load_engine};
use crate::npu::{AdvanceResult, NpuExecutor, NpuInstructionStream};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunState { Paused, Running }

pub struct EngineHost {
    pub engine: InterpreterEngine,
    executor: Option<NpuExecutor>,
    pub run_state: RunState,
}

/// Look for a companion instruction stream next to the xclbin. v1 uses the
/// simple convention (insts.bin / insts.elf in the same directory); zeroed
/// host memory is a valid default input, so a missing insts file just means
/// no control program runs.
fn find_companion_insts(xclbin_path: &Path) -> Option<NpuExecutor> {
    let dir = xclbin_path.parent()?;
    let data = ["insts.bin", "insts.elf"]
        .iter()
        .find_map(|n| fs::read(dir.join(n)).ok())?;
    let stream = NpuInstructionStream::parse(&data).ok()?;
    let mut ex = NpuExecutor::new();
    ex.set_host_buffers(default_host_buffers());
    ex.load(&stream);
    Some(ex)
}

pub fn load(xclbin_path: &Path) -> Result<EngineHost, String> {
    let engine = load_engine(xclbin_path)?;
    let executor = find_companion_insts(xclbin_path);
    Ok(EngineHost { engine, executor, run_state: RunState::Paused })
}

impl EngineHost {
    pub fn total_cycles(&self) -> u64 { self.engine.total_cycles() }
    pub fn status(&self) -> EngineStatus { self.engine.status() }

    pub fn step_one(&mut self) {
        if let Some(ex) = self.executor.as_mut() {
            let (device, host_mem) = self.engine.device_and_host_memory();
            if let AdvanceResult::Error(msg) = ex.try_advance(device, host_mem) {
                log::error!("NPU executor fatal: {}", msg);
                self.run_state = RunState::Paused;
                return;
            }
        }
        self.engine.step();
    }

    /// Up to `budget` steps; stops early on a terminal engine status.
    pub fn step_bounded(&mut self, budget: u32) -> EngineStatus {
        for _ in 0..budget {
            self.step_one();
            match self.engine.status() {
                EngineStatus::Halted | EngineStatus::Stalled | EngineStatus::Error => break,
                _ => {}
            }
        }
        self.engine.status()
    }

    pub fn reset(&mut self) {
        self.engine.reset();
        self.run_state = RunState::Paused;
    }
}
```

Confirm the exact import paths for `InterpreterEngine`, `EngineStatus`, `AdvanceResult`, `NpuExecutor`, `NpuInstructionStream`, `HostBuffer` by grepping their `pub use` re-exports (e.g. `crate::npu::AdvanceResult` was verified; `EngineStatus` lives in `interpreter::engine::coordinator` — prefer a crate-level re-export if one exists). Fix imports until it compiles.

- [ ] **Step 5: Run the tests, verify they pass** (or SKIP).

Run: `cargo test --lib debugger::engine_host 2>&1 | tail -20`
Expected: PASS (or SKIP eprintln).

- [ ] **Step 6: Commit.**

```bash
git add src/debugger/mod.rs src/debugger/engine_host.rs src/lib.rs
git commit -m "feat(debugger): EngineHost load + frame-bounded stepper

Generated using Claude Code."
```

---

## Task 4: View-model — `tile_grid` + `tile_snapshot` (`src/debugger/model.rs`)

**Files:**
- Modify/Create: `src/debugger/model.rs` (fill in the placeholder from Task 3)

**Interfaces:**
- Consumes: `TileArray::{iter, cols, rows}`; `Tile { pub col, pub row, is_shim(), is_mem(), is_compute() }`; `InterpreterEngine::{device, core_status, core_context}`; `CoreStatus`; `array.get(col,row)`; `Tile::{effective_lock_value, data_memory, read_data_u32}`; `Tile.stream_switch.{masters,slaves}` with `StreamPort.{cycle_active, cycle_stalled}`; `array.dma_engine(col,row)` -> `DmaEngine::{channel_count, channel_state, current_bd, queued_bd, task_queue_size}`.
- Produces:
  - `pub enum TileKindDisplay { Shim, Mem, Core }`
  - `pub struct TileCell { pub col: u8, pub row: u8, pub kind: TileKindDisplay }`
  - `pub fn tile_grid(array: &TileArray) -> Vec<TileCell>`
  - `pub struct ChannelSnapshot { pub index: u8, pub state: String, pub current_bd: Option<u8>, pub queued_bd: Option<u8>, pub queue_len: usize }`
  - `pub struct PortSnapshot { pub label: String, pub active: bool, pub stalled: bool }`
  - `pub struct TileSnapshot { pub col: u8, pub row: u8, pub kind: TileKindDisplay, pub core_status: Option<String>, pub pc: Option<u32>, pub dma: Vec<ChannelSnapshot>, pub locks: Vec<i8>, pub mem_size: usize, pub mem_peek: Vec<u32>, pub ports: Vec<PortSnapshot> }`
  - `pub fn tile_snapshot(engine: &InterpreterEngine, col: u8, row: u8) -> Option<TileSnapshot>`

- [ ] **Step 1: Write the failing tests.** Replace the `src/debugger/model.rs` placeholder with a test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::InterpreterEngine;

    #[test]
    fn tile_grid_matches_npu1_layout() {
        let engine = InterpreterEngine::new_npu1();
        let grid = tile_grid(&engine.device().array);
        // Every position in the flat array yields exactly one cell.
        let (cols, rows) = (engine.device().array.cols(), engine.device().array.rows());
        assert_eq!(grid.len(), cols as usize * rows as usize, "one cell per tile");
        // Row 0 is all shim; row 1 is all mem; rows >=2 are core.
        for c in &grid {
            match c.row {
                0 => assert_eq!(c.kind, TileKindDisplay::Shim),
                1 => assert_eq!(c.kind, TileKindDisplay::Mem),
                _ => assert_eq!(c.kind, TileKindDisplay::Core),
            }
        }
    }

    #[test]
    fn tile_snapshot_reports_locks_and_memory() {
        let engine = InterpreterEngine::new_npu1();
        // A compute tile exists at (0,2) on NPU1.
        let snap = tile_snapshot(&engine, 0, 2).expect("compute tile exists");
        assert_eq!(snap.locks.len(), 64);
        assert!(snap.mem_size > 0);
        assert_eq!(snap.col, 0);
        assert_eq!(snap.row, 2);
    }

    #[test]
    fn tile_snapshot_none_for_missing_tile() {
        let engine = InterpreterEngine::new_npu1();
        assert!(tile_snapshot(&engine, 99, 99).is_none());
    }
}
```

- [ ] **Step 2: Run, verify it fails.**

Run: `cargo test --lib debugger::model 2>&1 | tail -20`
Expected: compile error.

- [ ] **Step 3: Implement the structs and `tile_grid`.**

```rust
use crate::device::array::TileArray;      // confirm exact path
use crate::device::tile::Tile;            // confirm exact path
use crate::interpreter::InterpreterEngine;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileKindDisplay { Shim, Mem, Core }

impl TileKindDisplay {
    fn of(tile: &Tile) -> Self {
        if tile.is_shim() { TileKindDisplay::Shim }
        else if tile.is_mem() { TileKindDisplay::Mem }
        else { TileKindDisplay::Core }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TileCell { pub col: u8, pub row: u8, pub kind: TileKindDisplay }

pub fn tile_grid(array: &TileArray) -> Vec<TileCell> {
    array.iter()
        .map(|t| TileCell { col: t.col, row: t.row, kind: TileKindDisplay::of(t) })
        .collect()
}
```

- [ ] **Step 4: Implement `tile_snapshot`.**

```rust
#[derive(Clone, Debug)]
pub struct ChannelSnapshot { pub index: u8, pub state: String, pub current_bd: Option<u8>, pub queued_bd: Option<u8>, pub queue_len: usize }

#[derive(Clone, Debug)]
pub struct PortSnapshot { pub label: String, pub active: bool, pub stalled: bool }

#[derive(Clone, Debug)]
pub struct TileSnapshot {
    pub col: u8, pub row: u8, pub kind: TileKindDisplay,
    pub core_status: Option<String>, pub pc: Option<u32>,
    pub dma: Vec<ChannelSnapshot>, pub locks: Vec<i8>,
    pub mem_size: usize, pub mem_peek: Vec<u32>, pub ports: Vec<PortSnapshot>,
}

pub fn tile_snapshot(engine: &InterpreterEngine, col: u8, row: u8) -> Option<TileSnapshot> {
    let array = &engine.device().array;
    let tile = array.get(col, row)?;
    let kind = TileKindDisplay::of(tile);

    // Core status + live PC (only meaningful on compute tiles).
    let core_status = engine.core_status(col as usize, row as usize).map(|s| format!("{:?}", s));
    let pc = engine.core_context(col as usize, row as usize).map(|c| c.pc());

    // DMA channels (coarse live view).
    let mut dma = Vec::new();
    if let Some(eng) = array.dma_engine(col, row) {
        for ch in 0..eng.channel_count() as u8 {
            dma.push(ChannelSnapshot {
                index: ch,
                state: format!("{:?}", eng.channel_state(ch)),
                current_bd: eng.current_bd(ch),
                queued_bd: eng.queued_bd(ch),
                queue_len: eng.task_queue_size(ch),
            });
        }
    }

    // All 64 locks (effective value accounts for pending updates).
    let locks: Vec<i8> = (0..64).map(|i| tile.effective_lock_value(i)).collect();

    // Memory: size + a small word peek (first 8 words).
    let mem_size = tile.data_memory().len();
    let mem_peek: Vec<u32> = (0..8).filter_map(|w| tile.read_data_u32(w * 4)).collect();

    // Stream ports: master + slave activity.
    let mut ports = Vec::new();
    for (i, p) in tile.stream_switch.masters.iter().enumerate() {
        ports.push(PortSnapshot { label: format!("M{i}"), active: p.cycle_active, stalled: p.cycle_stalled });
    }
    for (i, p) in tile.stream_switch.slaves.iter().enumerate() {
        ports.push(PortSnapshot { label: format!("S{i}"), active: p.cycle_active, stalled: p.cycle_stalled });
    }

    Some(TileSnapshot { col, row, kind, core_status, pc, dma, locks, mem_size, mem_peek, ports })
}
```

Confirm exact import paths (`TileArray`, `Tile`) and that `data_memory().len()` / `read_data_u32(offset)` compile against the verified signatures. `effective_lock_value(i: usize) -> i8` was verified. If `stream_switch.masters`/`slaves` are named differently, correct against `FunctionalStreamSwitch`.

- [ ] **Step 5: Run the tests, verify they pass.**

Run: `cargo test --lib debugger::model 2>&1 | tail -20`
Expected: PASS (these use `new_npu1()` with no fixture, so they always run).

- [ ] **Step 6: Commit.**

```bash
git add src/debugger/model.rs src/debugger/mod.rs
git commit -m "feat(debugger): tile_grid + tile_snapshot view-model

Generated using Claude Code."
```

---

## Task 5: Retire old viewer, gated skeleton, repoint `main.rs`

**Files:**
- Delete: `src/visual/{alignment,timeline,viewport,data,event_detail,tile_selector}.rs`
- Rewrite: `src/visual/mod.rs`, `src/visual/app.rs`, `src/visual/theme.rs`
- Create: empty stubs `src/visual/{overview,detail,controls}.rs` (filled in Tasks 6-8)
- Modify: `src/main.rs` (`run_gui`)

**Interfaces:**
- Consumes: `crate::debugger::engine_host::{self, EngineHost, RunState}`.
- Produces: `pub struct DebuggerApp` implementing `eframe::App`; `DebuggerApp::new(xclbin: Option<PathBuf>) -> Self`.

This task delivers the thinnest vertical slice that runs: window opens, loads the CLI file if given, shows a cycle/status readout and a Step button that advances the emulator.

- [ ] **Step 1: Delete the old trace-viewer files.**

```bash
git rm src/visual/alignment.rs src/visual/timeline.rs src/visual/viewport.rs src/visual/data.rs src/visual/event_detail.rs src/visual/tile_selector.rs
```

- [ ] **Step 2: Rewrite `src/visual/theme.rs`** with a debugger palette (replace all trace-viewer constants):

```rust
//! Visual debugger palette. Functional-first; polish is a later pass.
use eframe::egui::Color32;

pub const BG: Color32 = Color32::from_rgb(24, 26, 30);
pub const TILE_SHIM: Color32 = Color32::from_rgb(90, 70, 120);
pub const TILE_MEM: Color32 = Color32::from_rgb(60, 100, 120);
pub const TILE_CORE: Color32 = Color32::from_rgb(70, 110, 80);
pub const TILE_SELECTED: Color32 = Color32::from_rgb(230, 200, 90);
pub const TILE_LABEL: Color32 = Color32::from_rgb(220, 220, 220);
pub const PORT_ACTIVE: Color32 = Color32::from_rgb(120, 200, 120);
pub const PORT_STALLED: Color32 = Color32::from_rgb(220, 140, 90);
```

- [ ] **Step 3: Rewrite `src/visual/mod.rs`.**

```rust
//! Live visual debugger (egui). Gated behind the `gui` feature. All logic
//! lives in crate::debugger (egui-free, tested); this layer only renders.
pub mod app;
pub mod controls;
pub mod detail;
pub mod overview;
pub mod theme;

pub use app::DebuggerApp;
```

- [ ] **Step 4: Create empty stubs** so the module compiles:

`src/visual/overview.rs`:
```rust
//! Array grid rendering (Task 6).
```
`src/visual/detail.rs`:
```rust
//! Selected-tile detail rows (Task 7).
```
`src/visual/controls.rs`:
```rust
//! Control bar (Task 8).
```

- [ ] **Step 5: Write `src/visual/app.rs`** — the skeleton app:

```rust
use std::path::PathBuf;

use eframe::egui;

use crate::debugger::engine_host::{self, EngineHost};

pub struct DebuggerApp {
    host: Option<EngineHost>,
    load_error: Option<String>,
    pub selected: Option<(u8, u8)>,
    /// Cycles advanced per frame while running (single tunable; a speed slider
    /// drops straight in here later).
    pub run_budget: u32,
}

impl DebuggerApp {
    pub fn new(xclbin: Option<PathBuf>) -> Self {
        let (host, load_error) = match xclbin {
            Some(p) => match engine_host::load(&p) {
                Ok(h) => (Some(h), None),
                Err(e) => (None, Some(e)),
            },
            None => (None, None),
        };
        Self { host, load_error, selected: None, run_budget: 32 }
    }
}

impl eframe::App for DebuggerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                match &self.host {
                    Some(h) => {
                        ui.label(format!("cycle: {}", h.total_cycles()));
                        ui.label(format!("status: {:?}", h.status()));
                    }
                    None => {
                        ui.label(self.load_error.clone().unwrap_or_else(|| "No design loaded".into()));
                    }
                }
                if ui.button("Step").clicked() {
                    if let Some(h) = self.host.as_mut() { h.step_one(); }
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("array + detail panels: Tasks 6-7");
        });
    }
}
```

- [ ] **Step 6: Repoint `src/main.rs::run_gui`.** Read `main.rs:96-124` and `run_gui` (491). Replace the body:
  - Remove `--trace-view-hw` / `--trace-view-emu` parsing (104-115) and the `trace_hw`/`trace_emu` args.
  - Change `run_gui` to `fn run_gui(file_arg: Option<String>) -> ...` and construct `xdna_emu::visual::DebuggerApp::new(file_arg.map(PathBuf::from))` instead of `TraceViewerApp::default()`.
  - Keep the `--gui`/`-g` flag and the no-args-opens-GUI behavior. Update the window title to `"xdna-emu Visual Debugger"`.
  - Keep the `#[cfg(feature = "gui")]` gate and the non-gui stub.

- [ ] **Step 7: Build with the gui feature.**

Run: `cargo build --features gui 2>&1 | tail -25`
Expected: builds clean.

- [ ] **Step 8: Smoke-test.** Load a built fixture (or run with no file). This opens a real window — run it, confirm it opens, click Step a few times, confirm the cycle counter increments, close it.

Run: `cargo run --features gui -- --gui ../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin`
Expected: window titled "xdna-emu Visual Debugger"; cycle counter starts at 0; each "Step" click increments it; status readout updates. (If the fixture is absent, run with no args and confirm "No design loaded".)

- [ ] **Step 9: Confirm `cargo test --lib` still green.**

Run: `cargo test --lib 2>&1 | tail -15`
Expected: no regression (the gui code isn't compiled here, but the delete/rewrite must not break the default build).

- [ ] **Step 10: Commit.**

```bash
git add -A src/visual src/main.rs
git commit -m "feat(visual): retire trace viewer; skeleton live debugger app

Generated using Claude Code."
```

---

## Task 6: Overview panel — array grid + click-select (`src/visual/overview.rs`)

**Files:**
- Modify: `src/visual/overview.rs`
- Modify: `src/visual/app.rs` (call the overview render in a side panel)

**Interfaces:**
- Consumes: `crate::debugger::model::{tile_grid, TileCell, TileKindDisplay}`; `crate::visual::theme`; `EngineHost` (for `&engine.device().array`).
- Produces: `pub fn show(ui: &mut egui::Ui, array: &TileArray, selected: &mut Option<(u8, u8)>)` — draws the grid, highlights `selected`, sets `selected` on click.

- [ ] **Step 1: Implement `overview::show`.** Draw the array as a spatial grid. Rows increase upward on hardware (row 0 = shim at the bottom); render row 0 at the bottom so the layout reads like the physical array. Use `egui::Grid` or a manual `painter` with clickable rects. Manual-rect version (gives click + highlight cheaply):

```rust
use eframe::egui::{self, Color32, Rect, Sense, Stroke, Vec2};

use crate::debugger::model::{tile_grid, TileCell, TileKindDisplay};
use crate::device::array::TileArray;   // confirm path
use crate::visual::theme;

fn tile_color(kind: TileKindDisplay) -> Color32 {
    match kind {
        TileKindDisplay::Shim => theme::TILE_SHIM,
        TileKindDisplay::Mem => theme::TILE_MEM,
        TileKindDisplay::Core => theme::TILE_CORE,
    }
}

pub fn show(ui: &mut egui::Ui, array: &TileArray, selected: &mut Option<(u8, u8)>) {
    let cells = tile_grid(array);
    let rows = array.rows();
    let cell = Vec2::new(64.0, 44.0);
    let pad = 6.0;
    let origin = ui.min_rect().min + Vec2::new(pad, pad);

    for c in &cells {
        // Flip row so row 0 (shim) sits at the bottom.
        let vis_row = (rows - 1 - c.row) as f32;
        let pos = origin + Vec2::new(c.col as f32 * (cell.x + pad), vis_row * (cell.y + pad));
        let rect = Rect::from_min_size(pos, cell);
        let resp = ui.allocate_rect(rect, Sense::click());
        if resp.clicked() { *selected = Some((c.col, c.row)); }

        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, tile_color(c.kind));
        if *selected == Some((c.col, c.row)) {
            painter.rect_stroke(rect, 4.0, Stroke::new(3.0, theme::TILE_SELECTED));
        }
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{},{}", c.col, c.row),
            egui::FontId::monospace(12.0),
            theme::TILE_LABEL,
        );
    }
    // Reserve the space we painted into so the panel sizes correctly.
    let width = array.cols() as f32 * (cell.x + pad) + pad;
    let height = rows as f32 * (cell.y + pad) + pad;
    ui.allocate_space(Vec2::new(width, height));
}
```

(If `allocate_rect` + `allocate_space` interact awkwardly, switch to computing the full desired size first with one `ui.allocate_at_least`, then paint — adjust until clicks land on the right tiles during the smoke test.)

- [ ] **Step 2: Wire into `app.rs`.** Add a left `SidePanel` before the `CentralPanel`:

```rust
egui::SidePanel::left("overview").resizable(true).show(ctx, |ui| {
    if let Some(h) = self.host.as_ref() {
        crate::visual::overview::show(ui, &h.engine.device().array, &mut self.selected);
    } else {
        ui.label("No design loaded");
    }
});
```

- [ ] **Step 3: Build.**

Run: `cargo build --features gui 2>&1 | tail -20`
Expected: clean.

- [ ] **Step 4: Smoke-test.** Run on the fixture; confirm the grid renders with the right tile count/colors (shim row at bottom), clicking a tile highlights it, and the selection persists.

Run: `cargo run --features gui -- --gui ../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin`
Expected: 5x6 grid (NPU1) of colored tiles; clicking selects/highlights.

- [ ] **Step 5: Commit.**

```bash
git add src/visual/overview.rs src/visual/app.rs
git commit -m "feat(visual): array overview grid with click-select

Generated using Claude Code."
```

---

## Task 7: Detail panel — selected-tile rows (`src/visual/detail.rs`)

**Files:**
- Modify: `src/visual/detail.rs`
- Modify: `src/visual/app.rs` (render detail in the central panel)

**Interfaces:**
- Consumes: `crate::debugger::model::{tile_snapshot, TileSnapshot}`; `EngineHost`.
- Produces: `pub fn show(ui: &mut egui::Ui, host: &EngineHost, selected: Option<(u8, u8)>)`.

- [ ] **Step 1: Implement `detail::show`.**

```rust
use eframe::egui;

use crate::debugger::engine_host::EngineHost;
use crate::debugger::model::tile_snapshot;
use crate::visual::theme;

pub fn show(ui: &mut egui::Ui, host: &EngineHost, selected: Option<(u8, u8)>) {
    let Some((col, row)) = selected else {
        ui.label("Select a tile to inspect it.");
        return;
    };
    let Some(snap) = tile_snapshot(&host.engine, col, row) else {
        ui.label(format!("No tile at ({col},{row})"));
        return;
    };

    ui.heading(format!("Tile ({},{})  [{:?}]", snap.col, snap.row, snap.kind));

    ui.separator();
    ui.label(format!("core: {}   pc: {}",
        snap.core_status.as_deref().unwrap_or("-"),
        snap.pc.map(|p| format!("0x{p:05x}")).unwrap_or_else(|| "-".into())));

    ui.separator();
    ui.label("DMA channels:");
    for ch in &snap.dma {
        ui.monospace(format!(
            "  ch{}: {}  cur_bd={:?} queued_bd={:?} queue={}",
            ch.index, ch.state, ch.current_bd, ch.queued_bd, ch.queue_len));
    }

    ui.separator();
    egui::CollapsingHeader::new(format!("locks (64)")).show(ui, |ui| {
        // 8 per row for compactness.
        for chunk in snap.locks.chunks(8) {
            ui.monospace(chunk.iter().map(|v| format!("{v:>3}")).collect::<Vec<_>>().join(" "));
        }
    });

    ui.separator();
    egui::CollapsingHeader::new(format!("memory ({} bytes)", snap.mem_size)).show(ui, |ui| {
        for (i, w) in snap.mem_peek.iter().enumerate() {
            ui.monospace(format!("  [0x{:04x}] 0x{:08x}", i * 4, w));
        }
    });

    ui.separator();
    ui.label("stream ports:");
    ui.horizontal_wrapped(|ui| {
        for p in &snap.ports {
            let color = if p.stalled { theme::PORT_STALLED }
                        else if p.active { theme::PORT_ACTIVE }
                        else { theme::TILE_LABEL };
            ui.colored_label(color, &p.label);
        }
    });
}
```

- [ ] **Step 2: Wire into `app.rs`** — replace the `CentralPanel` placeholder:

```rust
egui::CentralPanel::default().show(ctx, |ui| {
    match self.host.as_ref() {
        Some(h) => crate::visual::detail::show(ui, h, self.selected),
        None => { ui.label("No design loaded"); }
    }
});
```

- [ ] **Step 3: Build.**

Run: `cargo build --features gui 2>&1 | tail -20`
Expected: clean.

- [ ] **Step 4: Smoke-test.** Run on the fixture; select a compute tile, confirm core status/PC, DMA channels, 64 locks, memory size+peek, and stream ports render; click Step and confirm the values update (PC advances, lock/DMA state changes).

Run: `cargo run --features gui -- --gui ../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin`
Expected: detail panel shows live values that change as you Step.

- [ ] **Step 5: Commit.**

```bash
git add src/visual/detail.rs src/visual/app.rs
git commit -m "feat(visual): selected-tile detail panel

Generated using Claude Code."
```

---

## Task 8: Control bar + run-state loop (`src/visual/controls.rs`)

**Files:**
- Modify: `src/visual/controls.rs`
- Modify: `src/visual/app.rs` (use the control bar; drive `step_bounded` per frame when running)

**Interfaces:**
- Consumes: `EngineHost`, `RunState`, `DebuggerApp` fields.
- Produces: `pub fn show(ui: &mut egui::Ui, host: &mut EngineHost, run_budget: u32)` — renders cycle/status readout + step/run/pause/reset buttons and applies them to `host`.

- [ ] **Step 1: Implement `controls::show`.**

```rust
use eframe::egui;

use crate::debugger::engine_host::{EngineHost, RunState};

pub fn show(ui: &mut egui::Ui, host: &mut EngineHost, run_budget: u32) {
    ui.horizontal(|ui| {
        ui.monospace(format!("cycle {:>8}", host.total_cycles()));
        ui.monospace(format!("{:?}", host.status()));
        ui.separator();
        if ui.button("Step").clicked() { host.step_one(); }
        if ui.button("Step 100").clicked() { host.step_bounded(100); }
        match host.run_state {
            RunState::Paused => if ui.button("Run").clicked() { host.run_state = RunState::Running; },
            RunState::Running => if ui.button("Pause").clicked() { host.run_state = RunState::Paused; },
        }
        if ui.button("Reset").clicked() { host.reset(); }
        // run_budget shown for context; a slider replaces this later.
        ui.label(format!("budget/frame: {run_budget}"));
    });
}
```

- [ ] **Step 2: Wire into `app.rs`.** Replace the inline top-panel body with `controls::show`, and add the per-frame run advance:

```rust
fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
    // Advance while running, bounded per frame; request continuous repaint.
    if let Some(h) = self.host.as_mut() {
        if h.run_state == crate::debugger::engine_host::RunState::Running {
            let status = h.step_bounded(self.run_budget);
            use crate::interpreter::engine::coordinator::EngineStatus; // confirm path
            if matches!(status, EngineStatus::Halted | EngineStatus::Stalled | EngineStatus::Error) {
                h.run_state = crate::debugger::engine_host::RunState::Paused;
            } else {
                ctx.request_repaint();
            }
        }
    }

    egui::TopBottomPanel::top("controls").show(ctx, |ui| {
        match self.host.as_mut() {
            Some(h) => crate::visual::controls::show(ui, h, self.run_budget),
            None => { ui.label(self.load_error.clone().unwrap_or_else(|| "No design loaded".into())); }
        }
    });

    // ... SidePanel (overview) and CentralPanel (detail) as in Tasks 6-7 ...
}
```

- [ ] **Step 3: Build.**

Run: `cargo build --features gui 2>&1 | tail -20`
Expected: clean.

- [ ] **Step 4: Smoke-test the full v1.** Run on the fixture; confirm: Run animates the cycle counter and live tile state until the design halts (then auto-pauses); Pause freezes it; Step/Step 100 advance discretely; Reset returns to cycle 0; selecting different tiles during a run updates the detail panel live.

Run: `cargo run --features gui -- --gui ../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin`
Expected: all controls behave; the array + detail update as it runs.

- [ ] **Step 5: Final regression check.**

Run: `cargo test --lib 2>&1 | tail -15`
Expected: green, no regression from baseline.

- [ ] **Step 6: Commit.**

```bash
git add src/visual/controls.rs src/visual/app.rs
git commit -m "feat(visual): control bar + run/pause frame loop (v1 complete)

Generated using Claude Code."
```

---

## Success Criteria (v1)

- `cargo run --features gui -- --gui <path.xclbin>` opens the app, loads the design, and renders the array from the LIVE device model (correct tile count/types/positions for NPU1: 5 cols x 6 rows, row 0 shim, row 1 mem, rows 2-5 core).
- Step / Step 100 / Run / Pause / Reset visibly advance/control the emulator; the cycle counter and tile states change as it steps; Run auto-pauses on Halted/Stalled/Error.
- Selecting any tile shows its live core status+PC, DMA channels (state + current/queued BD + queue length), 64 lock values, memory size + word peek, and stream ports — values that update as the emulator steps.
- No emulator regression: `cargo test --lib` stays green; the shared load helper, DMA accessors, and view-model have unit coverage; `xclbin_suite` uses the shared load path.

## Out of Scope (v1)

Overview pan/zoom; connection/routing lines; animation; polish/prettiness; breakpoints; memory hex editor/editing; run-speed slider (the per-frame budget is a single tunable value now, ready for a slider); file-open dialog (v1 loads from the CLI arg — add `rfd` when wanted); per-resource drill-down tabs; multi-device beyond `new_npu1`.
