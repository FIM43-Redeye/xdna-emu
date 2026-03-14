# Phase 2: Module Integration and File Decomposition

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire 8 new subsystem modules into the existing tile/state/array/DMA infrastructure, replacing stubs and extracting code from bloated files.

**Architecture:** Each task replaces an old stub or scattered implementation with a delegation to the new self-contained module. The Tile struct gains new fields referencing the new modules, and register routing in state.rs delegates to the module's `read_register`/`write_register` interface. Every task is independent at the subsystem level but sequential at the file level (all touch tile.rs and state.rs).

**Tech Stack:** Rust, existing xdna-emu crate, new modules in `src/device/`.

**Key constraint:** All tasks modify `tile.rs` and `state.rs`. They MUST be executed sequentially. Run `cargo test --lib` after each task to catch regressions immediately.

---

## File Map

### Files being modified (shrinking)
- `src/device/tile.rs` (3393 lines) -- replace stub structs with new module types
- `src/device/state.rs` (2207 lines) -- update register routing to delegate
- `src/device/dma/engine.rs` (4257 lines) -- wire token queue and FIFO
- `src/device/dma/mod.rs` -- re-export new types
- `src/device/array.rs` (2104 lines) -- wire control packet processor
- `src/device/mod.rs` -- already done (modules declared)

### New modules being integrated (already created, 17 files)
- `src/device/perf_counters/mod.rs` -- PerfCounterBank (replaces PerfCounters stub)
- `src/device/timer.rs` -- TileTimer (new capability)
- `src/device/core_debug.rs` -- CoreDebugState (new capability)
- `src/device/events/` -- EventModule (new capability)
- `src/device/interrupts/` -- L1/L2InterruptController (new capability)
- `src/device/control_packets/` -- ControlPacketProcessor (replaces inline FSM)
- `src/device/dma/token.rs` -- TaskQueue, TokenState (new capability)
- `src/device/dma/fifo.rs` -- FotCountFifo (new capability)

---

## Task 1: Replace PerfCounters stub with PerfCounterBank

The old `PerfCounters` struct in tile.rs is a register-acceptance stub with no active counting. The new `perf_counters::PerfCounterBank` has full start/stop/reset event handling, tick(), and threshold events.

**Files:**
- Modify: `src/device/tile.rs` -- replace `PerfCounters` struct and fields
- Modify: `src/device/state.rs` -- update perf counter register routing

**Context to read:**
- `src/device/perf_counters/mod.rs` -- new PerfCounterBank API
- `src/device/tile.rs:1116-1263` -- old PerfCounters stub to remove
- `src/device/tile.rs:1041-1059` -- Tile struct fields to replace
- `src/device/tile.rs:1349-1360` -- constructor initialization
- `src/device/tile.rs:1632-1730` -- write_core_perf_register / write_mem_perf_register methods
- `src/device/state.rs:1703-1738` -- register routing dispatch

- [ ] **Step 1: Replace Tile struct fields**

In `tile.rs`, replace:
```rust
pub core_perf_counters: PerfCounters,
pub mem_perf_counters: PerfCounters,
```
with:
```rust
pub core_perf_counters: super::perf_counters::PerfCounterBank,
pub mem_perf_counters: super::perf_counters::PerfCounterBank,
```

Update the `Tile::new()` constructor to use `PerfCounterBank::new(N)` with the correct counter counts per tile type.

- [ ] **Step 2: Remove old PerfCounters struct**

Delete the old `PerfCounters` struct, `MAX_PERF_COUNTERS` const, and all its methods (tile.rs lines ~1116-1263). Also delete `write_core_perf_register()` and `write_mem_perf_register()` methods (~1632-1730).

- [ ] **Step 3: Update state.rs register routing**

In `apply_tile_local_effects()` (state.rs ~1703-1738), replace calls to `tile.write_core_perf_register(offset - base, value)` with `tile.core_perf_counters.write_register(offset - base, value, 7)` (7-bit event width for core/memory, 8-bit for memtile). Match the event width per module type.

- [ ] **Step 4: Fix any compilation errors**

Other files may reference `PerfCounters` or the old methods. Search for usages and update them.

- [ ] **Step 5: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: All tests pass including the 35 new perf_counters tests and existing perf counter register tests in state.rs.

- [ ] **Step 6: Commit**

```
feat: replace PerfCounters stub with active PerfCounterBank module
```

---

## Task 2: Add TileTimer to Tile

Tiles currently have no timer. Add `timer::TileTimer` instances to each tile module.

**Files:**
- Modify: `src/device/tile.rs` -- add timer fields to Tile struct
- Modify: `src/device/state.rs` -- add timer register routing

**Context to read:**
- `src/device/timer.rs` -- TileTimer API and register offsets
- aie-rt timer module register base addresses per tile type

- [ ] **Step 1: Add timer fields to Tile struct**

In tile.rs Tile struct, add:
```rust
/// Core module timer (compute and shim tiles).
pub core_timer: super::timer::TileTimer,
/// Memory module timer (compute and mem tiles).
pub mem_timer: super::timer::TileTimer,
```

Initialize in `Tile::new()`:
```rust
core_timer: super::timer::TileTimer::new(),
mem_timer: super::timer::TileTimer::new(),
```

- [ ] **Step 2: Add timer register routing in state.rs**

In `apply_tile_local_effects()`, add a new section for timer registers. Use the register offsets from `timer.rs` constants, adjusted by the module base address per tile type. The timer registers live within the core module and memory module address spaces.

- [ ] **Step 3: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: All tests pass including 24 timer tests.

- [ ] **Step 4: Commit**

```
feat: add TileTimer to all tile types with register routing
```

---

## Task 3: Add CoreDebugState to Tile

The coordinator already has some debug state scattered around. Add the proper `CoreDebugState` to compute tiles.

**Files:**
- Modify: `src/device/tile.rs` -- add core_debug field
- Modify: `src/device/state.rs` -- add core debug register routing
- Optionally modify: `src/interpreter/engine/coordinator.rs` -- wire debug state

**Context to read:**
- `src/device/core_debug.rs` -- CoreDebugState API and register offsets
- `src/device/tile.rs:768-803` -- existing CoreState struct
- `src/interpreter/engine/coordinator.rs:347` -- existing core_status()

- [ ] **Step 1: Add core_debug field to Tile struct**

```rust
/// Core debug state (compute tiles only).
pub core_debug: super::core_debug::CoreDebugState,
```

Initialize in `Tile::new()`.

- [ ] **Step 2: Add core debug register routing in state.rs**

Core debug registers (Core_Control, Core_Status, Debug_Control*, Core_PC/SP/LR) are in the core module address space. Route writes to `tile.core_debug.write_register()` and reads to `tile.core_debug.read_register()`. The existing `write_core_register()` in state.rs already handles some of these -- update it to delegate.

- [ ] **Step 3: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: All tests pass including 42 core_debug tests.

- [ ] **Step 4: Commit**

```
feat: add CoreDebugState to compute tiles with register routing
```

---

## Task 4: Add EventModule to Tile

Events are currently scattered across tile.rs (broadcast channels, event port selection, pending broadcasts). Consolidate into the new EventModule.

**Files:**
- Modify: `src/device/tile.rs` -- replace scattered event fields with EventModule
- Modify: `src/device/state.rs` -- update event register routing

**Context to read:**
- `src/device/events/mod.rs` -- EventModule API
- `src/device/tile.rs:1008-1102` -- existing event fields to replace
- `src/device/state.rs:1761-1888` -- existing event register routing

- [ ] **Step 1: Replace event fields in Tile struct**

Remove:
- `event_port_selection: [Option<(u8, bool)>; 8]`
- `broadcast_channels: [u8; 16]`
- `pending_broadcasts: Vec<u8>`

Add:
```rust
/// Core module event system (compute and shim tiles).
pub core_events: super::events::EventModule,
/// Memory module event system (compute and mem tiles).
pub mem_events: super::events::EventModule,
```

Initialize with appropriate `EventModuleType` per tile type.

- [ ] **Step 2: Update state.rs event register routing**

Replace the inline event broadcast / event_generate / port selection handling in `apply_tile_local_effects()` with delegation to `tile.core_events.write_register()` or `tile.mem_events.write_register()`.

- [ ] **Step 3: Update consumers of old fields**

Search for `broadcast_channels`, `pending_broadcasts`, `event_port_selection` in all files. Update to use the EventModule API instead.

- [ ] **Step 4: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: All tests pass including 124 events tests.

- [ ] **Step 5: Commit**

```
feat: replace scattered event fields with EventModule
```

---

## Task 5: Add L1/L2 InterruptControllers to shim tiles

**Files:**
- Modify: `src/device/tile.rs` -- add interrupt controller fields
- Modify: `src/device/state.rs` -- add interrupt register routing

**Context to read:**
- `src/device/interrupts/mod.rs` -- L1/L2 API and register offsets

- [ ] **Step 1: Add interrupt controller fields to Tile struct**

```rust
/// L1 interrupt controller (shim tiles only).
pub l1_irq: Option<super::interrupts::L1InterruptController>,
/// L2 interrupt controller (shim NoC tiles only).
pub l2_irq: Option<super::interrupts::L2InterruptController>,
```

Initialize: `Some(...)` for shim tiles, `None` for others.

- [ ] **Step 2: Add interrupt register routing in state.rs**

Route interrupt register writes to the appropriate controller's `write_register()`.

- [ ] **Step 3: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`

- [ ] **Step 4: Commit**

```
feat: add L1/L2 interrupt controllers to shim tiles
```

---

## Task 6: Wire ControlPacketProcessor into Tile

Replace the inline control packet FSM in tile.rs with the new `control_packets::ControlPacketProcessor`.

**Files:**
- Modify: `src/device/tile.rs` -- replace FSM with processor
- Modify: `src/device/array.rs` -- update control packet handling

**Context to read:**
- `src/device/control_packets/processor.rs` -- ControlPacketProcessor API
- `src/device/tile.rs:856-896` -- ControlPacketState enum
- `src/device/tile.rs:2170+` -- process_ctrl_packet_word, execute_ctrl_packet methods
- `src/device/array.rs:232-329` -- control packet action handling

- [ ] **Step 1: Assess compatibility**

Read both the old FSM and new processor API carefully. Determine if the new processor can be a drop-in replacement or needs adaptation. The old FSM accumulates words and dispatches; the new processor's `process()` method takes a parsed `ControlPacket`. There may be a gap in the word-accumulation layer.

- [ ] **Step 2: Integrate or adapt**

Either replace the old FSM with the new processor (if APIs align) or create a thin adapter that accumulates words (old pattern) and delegates to the processor for execution (new pattern).

- [ ] **Step 3: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`

- [ ] **Step 4: Commit**

```
feat: wire ControlPacketProcessor into tile control packet handling
```

---

## Task 7: Wire DMA TaskQueue and TokenState into DmaEngine

**Files:**
- Modify: `src/device/dma/engine.rs` -- integrate token.rs types

**Context to read:**
- `src/device/dma/token.rs` -- TaskQueue, TokenState, TaskQueueEntry API
- `src/device/dma/engine.rs:99-160` -- existing task queue types
- `src/device/dma/engine.rs:2420-2620` -- existing enqueue/dequeue/token methods

- [ ] **Step 1: Assess overlap**

The engine already has `TaskCompleteToken`, `TaskQueueEntry`, `ChannelTaskConfig` structs, plus task queue and token methods. Compare with the new `token.rs` types. Determine which to keep.

- [ ] **Step 2: Replace or delegate**

Replace the engine's inline task queue management with `token::TaskQueue` and `token::TokenState` instances. Move the engine's task queue fields to use the new types.

- [ ] **Step 3: Wire FotCountFifo**

Add a `FotCountFifo` per channel for FoT count tracking. Wire it into the transfer completion path.

- [ ] **Step 4: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`

- [ ] **Step 5: Commit**

```
feat: wire TaskQueue and FotCountFifo into DMA engine
```

---

## Task 8: Final cleanup and dead code removal

After all integrations, clean up any dead code.

**Files:**
- Modify: any files with dead code from replaced stubs

- [ ] **Step 1: Run clippy**

Run: `cargo clippy --lib 2>&1 | grep "warning\[dead_code\]"`

- [ ] **Step 2: Remove dead code**

Delete unused structs, methods, and imports from all files.

- [ ] **Step 3: Verify line counts improved**

Check that tile.rs, state.rs, engine.rs, array.rs are all smaller. Report before/after.

- [ ] **Step 4: Run full test suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: All 2519+ tests pass.

- [ ] **Step 5: Commit**

```
refactor: remove dead code from module integration
```
